"""Phase 3 benchmark suite: zero-shot vs conformal vs head_only vs LoRA vs full_ft (spec §10).

    uv run python scripts/benchmarks/run_benchmarks.py --device cuda --bf16 --sizes 5M 28M --output /tmp/bench.json

Evaluates all adaptation modes across realised volatility (post-Oct 27, 2026 test split) and
retail demand, reporting raw-space pinball, wQL, MASE, 80%/90% coverage and width, plus the
forgetting metric on the held-out GIFT-Eval general set (energy, weather, traffic).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_general_forgetting, load_realised_volatility, load_retail_demand

from timestral_tune import (
    ConformalCalibrator,
    LoraConfig,
    OnlineConformalCalibrator,
    TimestralTrainer,
    TrainingConfig,
    collect_conformal_data,
)
from timestral_tune.models import TuneTimestralForecaster, load_model
from timestral_tune.utils import compute_forgetting, evaluate_forecaster


def _run_dataset_for_model(
    model_id: str,
    splits,
    general_ds,
    *,
    device: str,
    mixed_precision: str,
    compile_model: bool,
    max_steps: int,
    batch_size: int,
    lrs: dict[str, float],
    lora_rank: int,
    lora_targets: tuple[str, ...],
    sign_flip: bool,
    seed: int,
) -> dict[str, Any]:
    base_cpu = load_model(model_id, device="cpu")
    zs_model = copy.deepcopy(base_cpu).to(device)
    zs_model.transformer.rope.rebuild_inv_freq()
    zs_fc = TuneTimestralForecaster(zs_model)

    zs_test = evaluate_forecaster(zs_fc, splits.test, seasonality=splits.seasonality, batch_size=batch_size)
    zs_general = evaluate_forecaster(zs_fc, general_ds, seasonality=24, batch_size=batch_size)
    tag = model_id.split("-")[-1]
    print(
        f"  [{tag} | {splits.name}] zero_shot     | "
        f"wQL={zs_test.wql:.4f} MASE={zs_test.mase:.4f} "
        f"cov80={zs_test.coverage_80:.3f} cov90={zs_test.coverage_90:.3f}",
        flush=True,
    )

    modes_out: dict[str, Any] = {
        "zero_shot": {
            "test": zs_test.to_dict(),
            "forgetting": compute_forgetting(zs_general, zs_general).to_dict(),
        }
    }

    # Conformal CQR (static, fitted on val split) and ACI (online, warmed up on val then online-updated on test).
    val_preds, val_y, val_scale = collect_conformal_data(zs_fc, splits.val, batch_size=batch_size)
    cqr = ConformalCalibrator(target_coverage="all", bucket_size=zs_fc.patch_size).fit(val_preds, val_y, val_scale)
    cqr_test = evaluate_forecaster(
        zs_fc, splits.test, calibrator=cqr, seasonality=splits.seasonality, batch_size=batch_size
    )
    modes_out["conformal_cqr"] = {
        "test": cqr_test.to_dict(),
        "forgetting": compute_forgetting(zs_general, zs_general).to_dict(),  # zero weight change -> zero forgetting
    }
    print(
        f"  [{tag} | {splits.name}] conformal_cqr | "
        f"wQL={cqr_test.wql:.4f} MASE={cqr_test.mase:.4f} "
        f"cov80={cqr_test.coverage_80:.3f} cov90={cqr_test.coverage_90:.3f}",
        flush=True,
    )

    aci = OnlineConformalCalibrator(target_coverage="all", gamma=0.01, window=500, bucket_size=zs_fc.patch_size)
    aci.update(val_preds, val_y, val_scale)
    aci_test = evaluate_forecaster(
        zs_fc, splits.test, calibrator=aci, online_update=True, seasonality=splits.seasonality, batch_size=batch_size
    )
    modes_out["conformal_aci"] = {
        "test": aci_test.to_dict(),
        "forgetting": compute_forgetting(zs_general, zs_general).to_dict(),
    }
    print(
        f"  [{tag} | {splits.name}] conformal_aci | "
        f"wQL={aci_test.wql:.4f} MASE={aci_test.mase:.4f} "
        f"cov80={aci_test.coverage_80:.3f} cov90={aci_test.coverage_90:.3f}",
        flush=True,
    )

    # Gradient adaptation modes.
    for mode in ("head_only", "lora_variate", "lora_temporal", "full_ft"):
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
        cfg = TrainingConfig(
            model_name_or_path=model_id,
            mode=mode,  # type: ignore[arg-type]
            learning_rate=lrs[mode],
            batch_size=batch_size,
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
        res = trainer.fit()
        elapsed = time.perf_counter() - t0

        # Evaluate in raw space via upstream TimestralForecaster (wrapping an uncompiled merged copy if compiled).
        eval_model = copy.deepcopy(base_cpu)
        from timestral_tune.models import merged_state_dict

        eval_model.load_state_dict(merged_state_dict(trainer.model), strict=True)
        eval_model.to(device).eval()
        eval_model.transformer.rope.rebuild_inv_freq()
        tuned_fc = TuneTimestralForecaster(eval_model)

        test_m = evaluate_forecaster(tuned_fc, splits.test, seasonality=splits.seasonality, batch_size=batch_size)
        gen_m = evaluate_forecaster(tuned_fc, general_ds, seasonality=24, batch_size=batch_size)
        forg = compute_forgetting(gen_m, zs_general)

        modes_out[mode] = {
            "learning_rate": lrs[mode],
            "trainable_params": trainer.num_trainable_parameters,
            "steps": res.steps,
            "best_step": res.best_step,
            "elapsed_s": round(elapsed, 2),
            "test": test_m.to_dict(),
            "forgetting": forg.to_dict(),
        }
        print(
            f"  [{model_id.split('-')[-1]} | {splits.name}] {mode:13s} | "
            f"wQL={test_m.wql:.4f} MASE={test_m.mase:.4f} cov80={test_m.coverage_80:.3f} "
            f"cov90={test_m.coverage_90:.3f} | forg_dWQL={forg.delta_wql:+.4f} forg_dMASE={forg.delta_mase:+.4f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    return {
        "dataset": splits.name,
        "license": splits.license,
        "source": splits.source,
        "contamination_note": splits.contamination_note,
        "counts": {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
        "modes": modes_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sizes", nargs="+", default=["5M", "28M"], choices=["5M", "28M"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-lora-variate", type=float, default=3e-4)
    parser.add_argument("--lr-lora-temporal", type=float, default=3e-4)
    parser.add_argument("--lr-full", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-targets", nargs="+", default=["q", "v"])
    parser.add_argument("--sign-flip", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    precision = "bf16" if args.bf16 else "fp32"
    lrs = {
        "head_only": args.lr_head,
        "lora_variate": args.lr_lora_variate,
        "lora_temporal": args.lr_lora_temporal,
        "full_ft": args.lr_full,
    }
    print("== Loading datasets (realised volatility, retail demand, GIFT-Eval forgetting) ==", flush=True)
    vol = load_realised_volatility()
    ret = load_retail_demand()
    general = load_general_forgetting()
    print(
        f"Loaded vol(tr={len(vol.train)}, va={len(vol.val)}, te={len(vol.test)}), "
        f"retail(tr={len(ret.train)}, va={len(ret.val)}, te={len(ret.test)}), "
        f"general(n={len(general)})",
        flush=True,
    )

    report: dict[str, Any] = {
        "precision": precision,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "lrs": lrs,
        "lora_rank": args.lora_rank,
        "lora_targets": list(args.lora_targets),
        "sign_flip": args.sign_flip,
        "seed": args.seed,
        "general_forgetting_windows": len(general),
        "results": {},
    }
    for size in args.sizes:
        model_id = f"mistralai/Timestral-{size}"
        report["results"][size] = {}
        for splits in (vol, ret):
            report["results"][size][splits.name] = _run_dataset_for_model(
                model_id,
                splits,
                general,
                device=args.device,
                mixed_precision=precision,
                compile_model=args.compile,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
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
