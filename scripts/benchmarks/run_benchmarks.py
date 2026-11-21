"""Phase 3 benchmark suite: zero-shot vs conformal vs head_only vs LoRA vs full_ft (spec §10).

    uv run python scripts/benchmarks/run_benchmarks.py --device cuda --bf16 --sizes 5M 28M --sign-flip
    uv run python scripts/benchmarks/run_benchmarks.py --device cuda --bf16 --sizes 123M 209M \
        --max-steps 160 --batch-size 32 --full-ft-batch-size 16 --eval-batch-size 16 \
        --modes zero_shot conformal_aci lora_variate lora_temporal full_ft --sign-flip

Reproduces all benchmark tables in `docs/BENCHMARKS.md` (§1 and §3), reporting:
- Raw-space pinball, wQL, MASE, 80%/90% coverage and width through upstream `TimestralForecaster`.
- Chronological, horizon-delayed online ACI coverage (plus per-time-block coverage breakdown).
- Equal-weighted 3-domain macro relative forgetting (`ett1`, `jena_weather`, `loop_seattle`) plus
  per-domain forgetting and zero-shot baselines.
- Peak CUDA VRAM and wall-clock time per mode.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_general_forgetting_by_domain, load_realised_volatility, load_retail_demand

from timestral_tune import (
    ConformalCalibrator,
    LoraConfig,
    OnlineConformalCalibrator,
    TimestralTrainer,
    TrainingConfig,
    collect_conformal_data,
)
from timestral_tune.config import DEFAULT_LEARNING_RATES
from timestral_tune.models import TuneTimestralForecaster, load_model, merged_state_dict
from timestral_tune.models.forward import attention_context
from timestral_tune.utils import compute_forgetting, coverage, evaluate_forecaster, evaluate_predictions
from timestral_tune.utils.metrics import _apply_online_chronological, _collect_contexts

ALL_MODES = ("zero_shot", "conformal_cqr", "conformal_aci", "head_only", "lora_variate", "lora_temporal", "full_ft")


def _macro_forgetting(
    fc: TuneTimestralForecaster,
    gen_by_domain: dict[str, Any],
    zs_by_domain: dict[str, Any],
    eval_batch_size: int,
) -> dict[str, Any]:
    per_domain: dict[str, Any] = {}
    for dom, ds in gen_by_domain.items():
        m = evaluate_forecaster(fc, ds, seasonality=24, batch_size=eval_batch_size)
        per_domain[dom] = compute_forgetting(m, zs_by_domain[dom]).to_dict()
    return {
        "macro_relative_delta_wql": float(np.mean([v["relative_delta_wql"] for v in per_domain.values()])),
        "macro_relative_delta_mase": float(np.mean([v["relative_delta_mase"] for v in per_domain.values()])),
        "macro_delta_wql": float(np.mean([v["delta_wql"] for v in per_domain.values()])),
        "macro_delta_mase": float(np.mean([v["delta_mase"] for v in per_domain.values()])),
        "per_domain": per_domain,
    }


def _zero_forgetting(zs_by_domain: dict[str, Any]) -> dict[str, Any]:
    per_domain = {dom: compute_forgetting(m, m).to_dict() for dom, m in zs_by_domain.items()}
    return {
        "macro_relative_delta_wql": 0.0,
        "macro_relative_delta_mase": 0.0,
        "macro_delta_wql": 0.0,
        "macro_delta_mase": 0.0,
        "per_domain": per_domain,
    }


def _per_origin_coverage_80(preds: np.ndarray, actuals: np.ndarray, dataset) -> list[float]:
    origins = np.array(
        [
            dataset.series[int(dataset._series_index[i])].timestamps[int(dataset._origins[i])]
            for i in range(len(dataset))
        ]
    )
    out = []
    for orig in sorted(np.unique(origins)):
        mask = origins == orig
        out.append(round(coverage(preds[mask], actuals[mask], 0.80), 4))
    return out


def _run_dataset_for_model(
    model_id: str,
    splits,
    gen_by_domain: dict[str, Any],
    *,
    device: str,
    mixed_precision: str,
    attention_backend: str,
    compile_model: bool,
    modes: tuple[str, ...],
    max_steps: int,
    batch_size: int,
    eval_batch_size: int,
    full_ft_batch_size: int,
    lrs: dict[str, float],
    lora_rank: int,
    lora_targets: tuple[str, ...],
    sign_flip: bool,
    seed: int,
) -> dict[str, Any]:
    base_cpu = load_model(model_id, device="cpu")
    zs_model = copy.deepcopy(base_cpu).to(device).eval()
    zs_model.transformer.rope.rebuild_inv_freq()
    zs_fc = TuneTimestralForecaster(zs_model)
    tag = model_id.split("-")[-1]

    zs_by_domain = {
        dom: evaluate_forecaster(zs_fc, ds, seasonality=24, batch_size=eval_batch_size)
        for dom, ds in gen_by_domain.items()
    }
    zs_preds, test_y, test_scale = collect_conformal_data(zs_fc, splits.test, batch_size=eval_batch_size)
    test_ctx = _collect_contexts(splits.test)
    zs_test = evaluate_predictions(zs_preds, test_y, test_ctx, seasonality=splits.seasonality)
    zs_block_cov80 = _per_origin_coverage_80(zs_preds, test_y, splits.test)

    modes_out: dict[str, Any] = {}
    if "zero_shot" in modes:
        modes_out["zero_shot"] = {
            "test": zs_test.to_dict(),
            "per_block_cov80": zs_block_cov80,
            "forgetting": _zero_forgetting(zs_by_domain),
        }
        print(
            f"  [{tag} | {splits.name}] zero_shot     | "
            f"wQL={zs_test.wql:.4f} MASE={zs_test.mase:.4f} "
            f"cov80={zs_test.coverage_80:.3f} cov90={zs_test.coverage_90:.3f} blocks={zs_block_cov80}",
            flush=True,
        )

    if "conformal_cqr" in modes or "conformal_aci" in modes:
        val_preds, val_y, val_scale = collect_conformal_data(zs_fc, splits.val, batch_size=eval_batch_size)
        if "conformal_cqr" in modes:
            cqr = ConformalCalibrator(target_coverage="all", bucket_size=zs_fc.patch_size).fit(
                val_preds, val_y, val_scale
            )
            cqr_preds = cqr.transform(zs_preds, context_scale=test_scale)
            cqr_test = evaluate_predictions(cqr_preds, test_y, test_ctx, seasonality=splits.seasonality)
            modes_out["conformal_cqr"] = {
                "test": cqr_test.to_dict(),
                "per_block_cov80": _per_origin_coverage_80(cqr_preds, test_y, splits.test),
                "forgetting": _zero_forgetting(zs_by_domain),
            }
            print(
                f"  [{tag} | {splits.name}] conformal_cqr | "
                f"wQL={cqr_test.wql:.4f} MASE={cqr_test.mase:.4f} "
                f"cov80={cqr_test.coverage_80:.3f} cov90={cqr_test.coverage_90:.3f}",
                flush=True,
            )
        if "conformal_aci" in modes:
            aci = OnlineConformalCalibrator(target_coverage="all", gamma=0.01, window=500, bucket_size=zs_fc.patch_size)
            aci.update(val_preds, val_y, val_scale)
            aci_preds = _apply_online_chronological(aci, splits.test, zs_preds, test_y, test_scale)
            aci_test = evaluate_predictions(aci_preds, test_y, test_ctx, seasonality=splits.seasonality)
            aci_blocks = _per_origin_coverage_80(aci_preds, test_y, splits.test)
            modes_out["conformal_aci"] = {
                "test": aci_test.to_dict(),
                "per_block_cov80": aci_blocks,
                "forgetting": _zero_forgetting(zs_by_domain),
            }
            print(
                f"  [{tag} | {splits.name}] conformal_aci | "
                f"wQL={aci_test.wql:.4f} MASE={aci_test.mase:.4f} "
                f"cov80={aci_test.coverage_80:.3f} cov90={aci_test.coverage_90:.3f} blocks={aci_blocks}",
                flush=True,
            )

    del zs_model, zs_fc
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    for mode in ("head_only", "lora_variate", "lora_temporal", "full_ft"):
        if mode not in modes:
            continue
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model = copy.deepcopy(base_cpu)
        lora = (
            LoraConfig(
                r=lora_rank,
                lora_alpha=2.0 * lora_rank,
                axes=("variate" if mode == "lora_variate" else "time",),
                targets=lora_targets,  # type: ignore[arg-type]
                dropout=0.05,
            )
            if mode.startswith("lora_")
            else None
        )
        run_bs = full_ft_batch_size if mode == "full_ft" else batch_size
        cfg = TrainingConfig(
            model_name_or_path=model_id,
            mode=mode,  # type: ignore[arg-type]
            learning_rate=lrs[mode],
            batch_size=run_bs,
            eval_batch_size=eval_batch_size,
            max_steps=max_steps,
            warmup_steps=max(10, max_steps // 10),
            mixed_precision=mixed_precision,  # type: ignore[arg-type]
            device=device,
            seed=seed,
            eval_every=max(20, max_steps // 5),
            restore_best=True,
            sign_flip=sign_flip,
            compile_model=compile_model and mode != "head_only",
            compile_mode="default",
            log_every=max_steps + 1,
        )
        trainer = TimestralTrainer(cfg, splits.train, splits.val, lora_config=lora, model=model)
        with attention_context(attention_backend):
            res = trainer.fit()
        elapsed = time.perf_counter() - t0
        peak_gib = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0
        trainable_p = trainer.num_trainable_parameters

        eval_model = copy.deepcopy(base_cpu)
        eval_model.load_state_dict(merged_state_dict(trainer.model), strict=True)
        del trainer, model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        eval_model.to(device).eval()
        eval_model.transformer.rope.rebuild_inv_freq()
        tuned_fc = TuneTimestralForecaster(eval_model)

        test_m = evaluate_forecaster(tuned_fc, splits.test, seasonality=splits.seasonality, batch_size=eval_batch_size)
        forg = _macro_forgetting(tuned_fc, gen_by_domain, zs_by_domain, eval_batch_size)
        del eval_model, tuned_fc
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        modes_out[mode] = {
            "learning_rate": lrs[mode],
            "batch_size": run_bs,
            "trainable_params": trainable_p,
            "steps": res.steps,
            "best_step": res.best_step,
            "elapsed_s": round(elapsed, 2),
            "peak_vram_gib": round(peak_gib, 2),
            "test": test_m.to_dict(),
            "forgetting": forg,
        }
        print(
            f"  [{tag} | {splits.name}] {mode:13s} | "
            f"wQL={test_m.wql:.4f} MASE={test_m.mase:.4f} cov80={test_m.coverage_80:.3f} "
            f"cov90={test_m.coverage_90:.3f} | macro_forg={forg['macro_relative_delta_wql']:+.1%}/"
            f"{forg['macro_relative_delta_mase']:+.1%} ({elapsed:.1f}s, {peak_gib:.2f} GiB)",
            flush=True,
        )

    return {
        "dataset": splits.name,
        "license": splits.license,
        "source": splits.source,
        "test_date_range": splits.test_date_range,
        "contamination_note": splits.contamination_note,
        "counts": {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
        "general_zero_shot_by_domain": {dom: m.to_dict() for dom, m in zs_by_domain.items()},
        "modes": modes_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sizes", nargs="+", default=["5M", "28M"], choices=["5M", "28M", "123M", "209M"])
    parser.add_argument("--datasets", nargs="+", default=["realised_volatility", "retail_demand"])
    parser.add_argument("--modes", nargs="+", default=list(ALL_MODES), choices=list(ALL_MODES))
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--attention-backend", default="default", choices=["default", "math"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--full-ft-batch-size", type=int, default=None)
    parser.add_argument("--lr-head", type=float, default=DEFAULT_LEARNING_RATES["head_only"])
    parser.add_argument("--lr-lora-variate", type=float, default=DEFAULT_LEARNING_RATES["lora_variate"])
    parser.add_argument("--lr-lora-temporal", type=float, default=DEFAULT_LEARNING_RATES["lora_temporal"])
    parser.add_argument("--lr-full", type=float, default=DEFAULT_LEARNING_RATES["full_ft"])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-targets", nargs="+", default=["q", "k", "v", "o"])
    parser.add_argument("--sign-flip", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    precision = "bf16" if args.bf16 else "fp32"
    eval_bs = args.eval_batch_size or args.batch_size
    full_bs = args.full_ft_batch_size or args.batch_size
    lrs = {
        "head_only": args.lr_head,
        "lora_variate": args.lr_lora_variate,
        "lora_temporal": args.lr_lora_temporal,
        "full_ft": args.lr_full,
    }
    print("== Loading datasets (realised volatility, retail demand, GIFT-Eval by domain) ==", flush=True)
    ds_map = {}
    if "realised_volatility" in args.datasets:
        ds_map["realised_volatility"] = load_realised_volatility()
    if "retail_demand" in args.datasets:
        ds_map["retail_demand"] = load_retail_demand()
    gen_by_domain = load_general_forgetting_by_domain()
    for name, sp in ds_map.items():
        print(f"  {name}: tr={len(sp.train)} va={len(sp.val)} te={len(sp.test)} test_dates={sp.test_date_range}")
    print("  general_by_domain:", {k: len(v) for k, v in gen_by_domain.items()}, flush=True)

    report: dict[str, Any] = {
        "precision": precision,
        "attention_backend": args.attention_backend,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "eval_batch_size": eval_bs,
        "full_ft_batch_size": full_bs,
        "lrs": lrs,
        "lora_rank": args.lora_rank,
        "lora_targets": list(args.lora_targets),
        "sign_flip": args.sign_flip,
        "seed": args.seed,
        "results": {},
    }
    for size in args.sizes:
        model_id = f"mistralai/Timestral-{size}"
        report["results"][size] = {}
        for name, splits in ds_map.items():
            report["results"][size][name] = _run_dataset_for_model(
                model_id,
                splits,
                gen_by_domain,
                device=args.device,
                mixed_precision=precision,
                attention_backend=args.attention_backend,
                compile_model=args.compile,
                modes=tuple(args.modes),
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                eval_batch_size=eval_bs,
                full_ft_batch_size=full_bs,
                lrs=lrs,
                lora_rank=args.lora_rank,
                lora_targets=tuple(args.lora_targets),
                sign_flip=args.sign_flip,
                seed=args.seed,
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved benchmark report to {args.output}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
