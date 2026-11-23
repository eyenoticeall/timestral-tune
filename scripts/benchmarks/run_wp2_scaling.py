"""Work Package 2: Training-length (200..2,000 steps) and data-size (10%/30%/100%) scaling curves.

    uv run python scripts/benchmarks/run_wp2_scaling.py \
        --device cuda --bf16 --sizes 5M 28M --seeds 0 1 2 \
        --output docs/benchmarks/wp2_scaling.json

Measures where fine-tuning gains saturate as a function of optimisation steps (with validation
early stopping) and fraction of training series (keeping val and test identical across fractions),
recording test decision metrics, GIFT-Eval forgetting, wall time, and peak GPU memory.
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
from datasets import (
    BenchmarkSplits,
    load_general_forgetting_by_domain,
    load_realised_volatility,
    load_retail_demand,
    subset_train_series,
)
from decision_metrics import evaluate_with_calibrators
from run_benchmarks import _macro_forgetting

from timestral_tune import LoraConfig, TimestralTrainer, TrainingConfig, collect_conformal_data
from timestral_tune.config import DEFAULT_LEARNING_RATES
from timestral_tune.models import TuneTimestralForecaster, load_model, merged_state_dict
from timestral_tune.utils import evaluate_forecaster


def _peak_mem_gib(device: str) -> float:
    if device == "cuda":
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    if device == "mps":
        return float(torch.mps.driver_allocated_memory() / (1024**3))
    return 0.0


def _reset_mem(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        torch.mps.empty_cache()
    gc.collect()


def _run_cell(
    base_cpu,
    model_id: str,
    splits: BenchmarkSplits,
    train_ds,
    gen_by_domain: dict[str, Any],
    zs_by_domain: dict[str, Any],
    *,
    mode: str,
    device: str,
    precision: str,
    max_steps: int,
    eval_every: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    _reset_mem(device)
    t0 = time.perf_counter()
    model = copy.deepcopy(base_cpu)
    lora = (
        LoraConfig(r=8, lora_alpha=16.0, axes=("time",), targets=("q", "k", "v", "o"), dropout=0.05)
        if mode == "lora_temporal"
        else None
    )
    cfg = TrainingConfig(
        model_name_or_path=model_id,
        mode=mode,  # type: ignore[arg-type]
        learning_rate=DEFAULT_LEARNING_RATES[mode],  # type: ignore[index]
        batch_size=batch_size,
        eval_batch_size=batch_size,
        max_steps=max_steps,
        warmup_steps=max(10, min(100, max_steps // 10)),
        mixed_precision=precision,  # type: ignore[arg-type]
        device=device,
        seed=seed,
        eval_every=eval_every,
        early_stopping_patience=patience,
        restore_best=True,
        sign_flip=True,
        log_every=max_steps + 1,
    )
    trainer = TimestralTrainer(cfg, train_ds, splits.val, lora_config=lora, model=model)
    res = trainer.fit()
    elapsed = time.perf_counter() - t0
    peak_gib = _peak_mem_gib(device)

    eval_model = copy.deepcopy(base_cpu)
    eval_model.load_state_dict(merged_state_dict(trainer.model), strict=True)
    del trainer, model
    _reset_mem(device)

    eval_model.to(device).eval()
    eval_model.transformer.rope.rebuild_inv_freq()
    fc = TuneTimestralForecaster(eval_model)
    val_p, val_y, val_s = collect_conformal_data(fc, splits.val, batch_size=batch_size)
    test_p, test_y, test_s = collect_conformal_data(fc, splits.test, batch_size=batch_size)
    metrics, _ = evaluate_with_calibrators(
        val_p,
        val_y,
        val_s,
        test_p,
        test_y,
        test_s,
        splits.test,
        seasonality=splits.seasonality,
        patch_size=fc.patch_size,
    )
    forg = _macro_forgetting(fc, gen_by_domain, zs_by_domain, batch_size)
    del eval_model, fc
    _reset_mem(device)

    return {
        "seed": seed,
        "train_series": len(set(int(i) for i in train_ds._series_index)),
        "train_windows": len(train_ds),
        "max_steps": max_steps,
        "steps_completed": res.steps,
        "best_step": res.best_step,
        "early_stopped": res.steps < max_steps,
        "best_eval_loss": res.best_eval_loss,
        "elapsed_s": round(elapsed, 2),
        "peak_vram_gib": round(peak_gib, 3),
        "raw": metrics["raw"],
        "cqr": metrics["cqr"],
        "forgetting": forg,
    }


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "wql_mean": float(np.mean([r["raw"]["wql"] for r in rows])),
        "wql_std": float(np.std([r["raw"]["wql"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
        "mase_mean": float(np.mean([r["raw"]["mase"] for r in rows])),
        "winkler_80_mean": float(np.mean([r["raw"]["winkler_80"] for r in rows])),
        "cqr_winkler_80_mean": float(np.mean([r["cqr"]["winkler_80"] for r in rows])),
        "cqr_width_80_mean": float(np.mean([r["cqr"]["width_80"] for r in rows])),
        "cqr_cov_80_mean": float(np.mean([r["cqr"]["coverage_80"] for r in rows])),
        "pinball_90_mean": float(np.mean([r["raw"]["pinball_90"] for r in rows])),
        "forgetting_wql_mean": float(np.mean([r["forgetting"]["macro_relative_delta_wql"] for r in rows])),
        "forgetting_wql_worst": float(np.max([r["forgetting"]["macro_relative_delta_wql"] for r in rows])),
        "best_step_mean": round(float(np.mean([r["best_step"] or 0 for r in rows])), 1),
        "steps_completed_mean": round(float(np.mean([r["steps_completed"] for r in rows])), 1),
        "elapsed_s_mean": round(float(np.mean([r["elapsed_s"] for r in rows])), 2),
        "peak_vram_gib_max": round(float(np.max([r["peak_vram_gib"] for r in rows])), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--sizes", nargs="+", default=["5M", "28M"], choices=["5M", "28M"])
    parser.add_argument("--datasets", nargs="+", default=["retail_demand", "realised_volatility"])
    parser.add_argument("--step-grid", type=int, nargs="+", default=[200, 500, 1000, 2000])
    parser.add_argument("--data-fractions", type=float, nargs="+", default=[0.10, 0.30, 1.00])
    parser.add_argument("--data-max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    precision = "bf16" if args.bf16 else "fp32"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds_map: dict[str, BenchmarkSplits] = {}
    if "retail_demand" in args.datasets:
        ds_map["retail_demand"] = load_retail_demand()
    if "realised_volatility" in args.datasets:
        ds_map["realised_volatility"] = load_realised_volatility()
    gen_by_domain = load_general_forgetting_by_domain()

    report: dict[str, Any] = {
        "precision": precision,
        "device": args.device,
        "batch_size": args.batch_size,
        "seeds": args.seeds,
        "step_grid": args.step_grid,
        "data_fractions": args.data_fractions,
        "data_max_steps": args.data_max_steps,
        "results": {},
    }

    for size in args.sizes:
        model_id = f"mistralai/Timestral-{size}"
        base_cpu = load_model(model_id, device="cpu")
        report["results"][size] = {}

        zs_model = copy.deepcopy(base_cpu).to(args.device).eval()
        zs_model.transformer.rope.rebuild_inv_freq()
        zs_fc = TuneTimestralForecaster(zs_model)
        zs_by_domain = {
            dom: evaluate_forecaster(zs_fc, ds, seasonality=24, batch_size=args.batch_size)
            for dom, ds in gen_by_domain.items()
        }

        for name, sp in ds_map.items():
            val_p, val_y, val_s = collect_conformal_data(zs_fc, sp.val, batch_size=args.batch_size)
            test_p, test_y, test_s = collect_conformal_data(zs_fc, sp.test, batch_size=args.batch_size)
            zs_m, _ = evaluate_with_calibrators(
                val_p,
                val_y,
                val_s,
                test_p,
                test_y,
                test_s,
                sp.test,
                seasonality=sp.seasonality,
                patch_size=zs_fc.patch_size,
            )
            del zs_model, zs_fc
            _reset_mem(args.device)

            entry: dict[str, Any] = {
                "zero_shot": {"raw": zs_m["raw"], "cqr": zs_m["cqr"]},
                "training_length": {},
                "data_size": {},
            }
            for mode in ("lora_temporal", "full_ft"):
                entry["training_length"][mode] = {}
                for max_steps in args.step_grid:
                    eval_every = 40 if max_steps <= 200 else 50
                    rows = [
                        _run_cell(
                            base_cpu,
                            model_id,
                            sp,
                            sp.train,
                            gen_by_domain,
                            zs_by_domain,
                            mode=mode,
                            device=args.device,
                            precision=precision,
                            max_steps=max_steps,
                            eval_every=eval_every,
                            patience=6,
                            batch_size=args.batch_size,
                            seed=sd,
                        )
                        for sd in args.seeds
                    ]
                    agg = _agg(rows)
                    entry["training_length"][mode][str(max_steps)] = {"seeds": rows, "summary": agg}
                    rel = agg["wql_mean"] / zs_m["raw"]["wql"] - 1.0
                    print(
                        f"  [steps | {size} | {name:19s} | {mode:13s} | {max_steps:4d}] "
                        f"wQL={agg['wql_mean']:.4f}±{agg['wql_std']:.4f} ({rel:+.1%}) "
                        f"best={agg['best_step_mean']:.0f}/{agg['steps_completed_mean']:.0f} "
                        f"forg={agg['forgetting_wql_mean']:+.1%} "
                        f"({agg['elapsed_s_mean']}s, {agg['peak_vram_gib_max']:.2f} GiB)",
                        flush=True,
                    )
                    report["results"][size][name] = entry
                    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

                entry["data_size"][mode] = {}
                for frac in args.data_fractions:
                    rows = [
                        _run_cell(
                            base_cpu,
                            model_id,
                            sp,
                            subset_train_series(sp.train, frac, seed=sd),
                            gen_by_domain,
                            zs_by_domain,
                            mode=mode,
                            device=args.device,
                            precision=precision,
                            max_steps=args.data_max_steps,
                            eval_every=25,
                            patience=5,
                            batch_size=args.batch_size,
                            seed=sd,
                        )
                        for sd in args.seeds
                    ]
                    agg = _agg(rows)
                    key = f"{int(round(frac * 100))}pct"
                    entry["data_size"][mode][key] = {"fraction": frac, "seeds": rows, "summary": agg}
                    rel = agg["wql_mean"] / zs_m["raw"]["wql"] - 1.0
                    print(
                        f"  [data  | {size} | {name:19s} | {mode:13s} | {key:5s}] "
                        f"wQL={agg['wql_mean']:.4f}±{agg['wql_std']:.4f} ({rel:+.1%}) "
                        f"best={agg['best_step_mean']:.0f}/{agg['steps_completed_mean']:.0f} "
                        f"forg={agg['forgetting_wql_mean']:+.1%} ({agg['elapsed_s_mean']}s)",
                        flush=True,
                    )
                    report["results"][size][name] = entry
                    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

            zs_model = copy.deepcopy(base_cpu).to(args.device).eval()
            zs_model.transformer.rope.rebuild_inv_freq()
            zs_fc = TuneTimestralForecaster(zs_model)

        del zs_model, zs_fc, base_cpu
        _reset_mem(args.device)

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved WP2 scaling report to {args.output}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
