"""Phase 3 hyperparameter & sign-flip sweeps to back the defaults in spec §5.4.

    # Full unified 5M/28M sweep:
    uv run python scripts/benchmarks/run_sweeps.py \
        --device cuda --sizes 5M 28M --bf16 --check-forgetting --output docs/benchmarks/sweeps_bracketed.json

    # Targeted 123M/209M 3-seed decision sweep (lora_temporal, full_ft, sign_flip on/off, signed target):
    uv run python scripts/benchmarks/run_sweeps.py \
        --device cuda --sizes 123M 209M --bf16 --batch-size 16 --eval-batch-size 16 \
        --datasets realised_volatility retail_demand signed_returns \
        --modes lora_temporal full_ft --mode-lrs lora_temporal=1e-3,3e-3 full_ft=1e-4,3e-4 \
        --sign-flip-modes off on --seeds 0 1 2 --check-forgetting --skip-lora-sweep \
        --output docs/benchmarks/sweeps_large_decisions.json

Sweeps are scored strictly on validation splits and GIFT-Eval general forgetting — never on test sets.
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

import torch
from datasets import (
    load_general_forgetting_by_domain,
    load_realised_volatility,
    load_retail_demand,
    load_signed_returns,
)

from timestral_tune import LoraConfig, TimestralTrainer, TrainingConfig
from timestral_tune.models import TuneTimestralForecaster, load_model, merged_state_dict
from timestral_tune.utils import compute_forgetting, evaluate_forecaster

# Wide grids so every mode's validation optimum is bracketed strictly inside the grid.
LR_GRID: dict[str, tuple[float, ...]] = {
    "head_only": (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    "lora_variate": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    "lora_temporal": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    "full_ft": (1e-5, 3e-5, 1e-4, 3e-4, 1e-3),
}

DATASET_LOADERS = {
    "realised_volatility": load_realised_volatility,
    "retail_demand": load_retail_demand,
    "signed_returns": load_signed_returns,
}


def _eval_macro_forgetting(fc, zs_by_domain: dict[str, Any], gen_by_domain: dict[str, Any], batch_size: int) -> dict:
    per_domain = {}
    for dom, ds in gen_by_domain.items():
        m = evaluate_forecaster(fc, ds, seasonality=24, batch_size=batch_size)
        per_domain[dom] = compute_forgetting(m, zs_by_domain[dom]).to_dict()
    macro_rel_wql = sum(v["relative_delta_wql"] for v in per_domain.values()) / len(per_domain)
    macro_rel_mase = sum(v["relative_delta_mase"] for v in per_domain.values()) / len(per_domain)
    return {
        "macro_relative_delta_wql": macro_rel_wql,
        "macro_relative_delta_mase": macro_rel_mase,
        "per_domain": per_domain,
    }


def _train_and_eval(
    base_model,
    splits,
    *,
    mode: str,
    lr: float,
    device: str,
    mixed_precision: str,
    max_steps: int,
    batch_size: int,
    eval_batch_size: int | None = None,
    lora_config: LoraConfig | None = None,
    sign_flip: bool = False,
    seed: int = 0,
    gen_by_domain: dict[str, Any] | None = None,
    zs_by_domain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_bs = eval_batch_size or batch_size
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = copy.deepcopy(base_model)
    cfg = TrainingConfig(
        model_name_or_path="sweep",
        mode=mode,  # type: ignore[arg-type]
        learning_rate=lr,
        batch_size=batch_size,
        eval_batch_size=eval_bs,
        max_steps=max_steps,
        warmup_steps=max(5, max_steps // 10),
        mixed_precision=mixed_precision,  # type: ignore[arg-type]
        device=device,
        seed=seed,
        eval_every=max(20, max_steps // 4),
        restore_best=True,
        sign_flip=sign_flip,
        log_every=max_steps + 1,
    )
    trainer = TimestralTrainer(cfg, splits.train, splits.val, lora_config=lora_config, model=model)
    res = trainer.fit()
    peak_gib = (torch.cuda.max_memory_allocated() / (1024**3)) if device == "cuda" else 0.0

    ev_m = copy.deepcopy(base_model)
    ev_m.load_state_dict(merged_state_dict(trainer.model), strict=True)
    del trainer, model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    ev_m.to(device).eval()
    ev_m.transformer.rope.rebuild_inv_freq()
    fc = TuneTimestralForecaster(ev_m)
    metrics = evaluate_forecaster(fc, splits.val, seasonality=splits.seasonality, batch_size=eval_bs)
    forg = (
        _eval_macro_forgetting(fc, zs_by_domain, gen_by_domain, eval_bs)
        if (gen_by_domain is not None and zs_by_domain is not None)
        else None
    )
    del ev_m, fc
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "mode": mode,
        "lr": lr,
        "seed": seed,
        "batch_size": batch_size,
        "lora": None if lora_config is None else lora_config.to_dict(),
        "sign_flip": sign_flip,
        "steps": res.steps,
        "best_step": res.best_step,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "peak_vram_gib": round(peak_gib, 2),
        "val_metrics": metrics.to_dict(),
        "forgetting": forg,
    }


def _parse_mode_lrs(specs: list[str] | None) -> dict[str, tuple[float, ...]]:
    overrides: dict[str, tuple[float, ...]] = {}
    for spec in specs or ():
        if "=" not in spec:
            raise ValueError(f"--mode-lrs entry must look like 'mode=1e-3,3e-3', got {spec!r}")
        mode, vals = spec.split("=", 1)
        if mode not in LR_GRID:
            raise ValueError(f"unknown mode {mode!r} in --mode-lrs")
        overrides[mode] = tuple(float(v) for v in vals.split(",") if v.strip())
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sizes", nargs="+", default=["5M", "28M"], choices=["5M", "28M", "123M", "209M"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["realised_volatility", "retail_demand"],
        choices=list(DATASET_LOADERS),
    )
    parser.add_argument("--modes", nargs="+", default=list(LR_GRID), choices=list(LR_GRID))
    parser.add_argument(
        "--mode-lrs",
        nargs="+",
        default=None,
        help="Optional per-mode LR subsets, e.g. lora_temporal=1e-3,3e-3 full_ft=1e-4,3e-4",
    )
    parser.add_argument(
        "--sign-flip-modes",
        nargs="+",
        default=["off"],
        choices=["off", "on"],
        help="Sign-flip settings to sweep across the LR grid (default: ['off']).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--check-forgetting", action="store_true")
    parser.add_argument("--skip-lora-sweep", action="store_true")
    parser.add_argument("--skip-sign-flip-ablation", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    precision = "bf16" if args.bf16 else "fp32"
    eval_bs = args.eval_batch_size or args.batch_size
    lr_overrides = _parse_mode_lrs(args.mode_lrs)
    sf_flags = [m == "on" for m in args.sign_flip_modes]
    datasets = [DATASET_LOADERS[name]() for name in args.datasets]
    gen_by_domain = load_general_forgetting_by_domain() if args.check_forgetting else None

    all_results: dict[str, Any] = {
        "precision": precision,
        "sign_flip_modes": list(args.sign_flip_modes),
        "seeds": list(args.seeds),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "eval_batch_size": eval_bs,
        "sizes": {},
    }

    for size in args.sizes:
        model_id = f"mistralai/Timestral-{size}"
        print(f"\n================ Sweeps: {model_id} ================", flush=True)
        base_model = load_model(model_id, device="cpu")
        eval_dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        zs_m = copy.deepcopy(base_model).to(eval_dev).eval()
        zs_m.transformer.rope.rebuild_inv_freq()
        zs_fc = TuneTimestralForecaster(zs_m)
        zs_by_domain = (
            {
                dom: evaluate_forecaster(zs_fc, ds, seasonality=24, batch_size=eval_bs)
                for dom, ds in gen_by_domain.items()
            }
            if gen_by_domain is not None
            else None
        )

        size_res: dict[str, Any] = {
            "general_zero_shot_by_domain": None
            if zs_by_domain is None
            else {dom: m.to_dict() for dom, m in zs_by_domain.items()},
            "datasets": {},
        }
        for splits in datasets:
            print(
                f"\n--- {size} | {splits.name} (train={len(splits.train)}, val={len(splits.val)}) ---",
                flush=True,
            )
            zs = evaluate_forecaster(zs_fc, splits.val, seasonality=splits.seasonality, batch_size=eval_bs)
            print(f"  zero_shot val: wQL={zs.wql:.4f} | MASE={zs.mase:.4f} | cov80={zs.coverage_80:.3f}", flush=True)

            ds_res: dict[str, Any] = {"zero_shot_val": zs.to_dict(), "lr_sweep": [], "lora_sweep": [], "sign_flip": []}

            best_lr: dict[str, float] = {}
            for mode in args.modes:
                lrs = lr_overrides.get(mode, LR_GRID[mode])
                best_wql = float("inf")
                for sf in sf_flags:
                    for lr in lrs:
                        lora = (
                            LoraConfig(
                                r=8,
                                lora_alpha=16.0,
                                axes=("variate" if mode == "lora_variate" else "time",),
                                targets=("q", "k", "v", "o"),
                            )
                            if mode.startswith("lora_")
                            else None
                        )
                        run_bs = (
                            min(args.batch_size, 16)
                            if (size in ("123M", "209M") and mode == "full_ft")
                            else args.batch_size
                        )
                        seed_wqls = []
                        for seed in args.seeds:
                            row = _train_and_eval(
                                base_model,
                                splits,
                                mode=mode,
                                lr=lr,
                                device=eval_dev,
                                mixed_precision=precision,
                                max_steps=args.max_steps,
                                batch_size=run_bs,
                                eval_batch_size=eval_bs,
                                lora_config=lora,
                                sign_flip=sf,
                                seed=seed,
                                gen_by_domain=gen_by_domain,
                                zs_by_domain=zs_by_domain,
                            )
                            ds_res["lr_sweep"].append(row)
                            wql = row["val_metrics"]["wql"]
                            mase_v = row["val_metrics"]["mase"]
                            seed_wqls.append(wql)
                            f_str = (
                                f" | macro_forg={row['forgetting']['macro_relative_delta_wql']:+.1%}"
                                if row["forgetting"] is not None
                                else ""
                            )
                            print(
                                f"  [LR] {mode:13s} sf={int(sf)} lr={lr:.0e} s={seed} | "
                                f"wQL={wql:.4f} | MASE={mase_v:.4f}{f_str} | {row['elapsed_s']:.1f}s",
                                flush=True,
                            )
                        mean_wql = sum(seed_wqls) / len(seed_wqls)
                        if mean_wql < best_wql:
                            best_wql = mean_wql
                            best_lr[mode] = lr

            if size == "5M" and not args.skip_lora_sweep:
                for mode in ("lora_variate", "lora_temporal"):
                    if mode not in best_lr:
                        continue
                    axis = "variate" if mode == "lora_variate" else "time"
                    for r, targets in [
                        (4, ("q", "v")),
                        (8, ("q", "v")),
                        (16, ("q", "v")),
                        (8, ("q", "k", "v", "o")),
                    ]:
                        lora = LoraConfig(r=r, lora_alpha=2.0 * r, axes=(axis,), targets=targets)  # type: ignore[arg-type]
                        row = _train_and_eval(
                            base_model,
                            splits,
                            mode=mode,
                            lr=best_lr[mode],
                            device=eval_dev,
                            mixed_precision=precision,
                            max_steps=args.max_steps,
                            batch_size=args.batch_size,
                            eval_batch_size=eval_bs,
                            lora_config=lora,
                            sign_flip=sf_flags[-1],
                            seed=args.seeds[0],
                        )
                        ds_res["lora_sweep"].append(row)
                        print(
                            f"  [LoRA] {mode:13s} r={r:2d} targets={targets} | "
                            f"wQL={row['val_metrics']['wql']:.4f} | MASE={row['val_metrics']['mase']:.4f}",
                            flush=True,
                        )

            if size == "5M" and not args.skip_sign_flip_ablation and len(sf_flags) == 1:
                for mode in args.modes:
                    if mode not in best_lr:
                        continue
                    lora = (
                        LoraConfig(
                            r=8,
                            lora_alpha=16.0,
                            axes=("variate" if mode == "lora_variate" else "time",),
                            targets=("q", "k", "v", "o"),
                        )
                        if mode.startswith("lora_")
                        else None
                    )
                    for sf in (False, True):
                        row = _train_and_eval(
                            base_model,
                            splits,
                            mode=mode,
                            lr=best_lr[mode],
                            device=eval_dev,
                            mixed_precision=precision,
                            max_steps=args.max_steps,
                            batch_size=args.batch_size,
                            eval_batch_size=eval_bs,
                            lora_config=lora,
                            sign_flip=sf,
                            seed=args.seeds[0],
                        )
                        ds_res["sign_flip"].append(row)
                        print(
                            f"  [SignFlip] {mode:13s} sign_flip={str(sf):5s} | "
                            f"wQL={row['val_metrics']['wql']:.4f} | MASE={row['val_metrics']['mase']:.4f}",
                            flush=True,
                        )

            size_res["datasets"][splits.name] = ds_res

        del zs_m, zs_fc, base_model
        if eval_dev == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        all_results["sizes"][size] = size_res

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved sweep results to {args.output}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
