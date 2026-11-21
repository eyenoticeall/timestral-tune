"""Phase 3 hyperparameter & sign-flip sweeps to back the defaults in spec §5.4.

    uv run python scripts/benchmarks/run_sweeps.py --device cuda --size 5M --output /tmp/sweeps.json

Sweeps:
1. Learning rates for `head_only`, `lora_variate`, `lora_temporal`, and `full_ft` on validation
   splits of realised volatility and retail demand, scored in raw space (wQL and MASE via
   `evaluate_forecaster`).
2. LoRA rank (`r in {4, 8, 16}`) and targets (`("q", "v")` vs `("q", "k", "v", "o")`).
3. Sign-flip augmentation (`sign_flip=False` vs `True`) on both one-signed domains.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_realised_volatility, load_retail_demand

from timestral_tune import LoraConfig, TimestralTrainer, TrainingConfig
from timestral_tune.models import TuneTimestralForecaster, load_model
from timestral_tune.utils import evaluate_forecaster

LR_GRID: dict[str, tuple[float, ...]] = {
    "head_only": (3e-5, 1e-4, 3e-4, 1e-3),
    "lora_variate": (1e-4, 3e-4, 1e-3),
    "lora_temporal": (1e-4, 3e-4, 1e-3),
    "full_ft": (3e-6, 1e-5, 3e-5, 1e-4),
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
    lora_config: LoraConfig | None = None,
    sign_flip: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = copy.deepcopy(base_model)
    cfg = TrainingConfig(
        model_name_or_path="sweep",
        mode=mode,  # type: ignore[arg-type]
        learning_rate=lr,
        batch_size=batch_size,
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
    fc = TuneTimestralForecaster(trainer.model)
    metrics = evaluate_forecaster(fc, splits.val, seasonality=splits.seasonality, batch_size=batch_size)
    return {
        "mode": mode,
        "lr": lr,
        "lora": None if lora_config is None else lora_config.to_dict(),
        "sign_flip": sign_flip,
        "steps": res.steps,
        "best_step": res.best_step,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "val_metrics": metrics.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--size", default="5M", choices=["5M", "28M"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    precision = "bf16" if args.bf16 else "fp32"
    model_id = f"mistralai/Timestral-{args.size}"
    print(f"== Loading {model_id} and datasets ==", flush=True)
    base_model = load_model(model_id, device="cpu")
    datasets = [load_realised_volatility(), load_retail_demand()]

    all_results: dict[str, Any] = {
        "model": model_id,
        "precision": precision,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "datasets": {},
    }

    for splits in datasets:
        print(f"\n=== Dataset: {splits.name} (train={len(splits.train)}, val={len(splits.val)}) ===", flush=True)
        zs = evaluate_forecaster(
            TuneTimestralForecaster(copy.deepcopy(base_model).to(args.device if args.device != "auto" else "cpu")),
            splits.val,
            seasonality=splits.seasonality,
            batch_size=args.batch_size,
        )
        print(f"  zero_shot val: wQL={zs.wql:.4f} | MASE={zs.mase:.4f} | cov80={zs.coverage_80:.3f}", flush=True)

        ds_res: dict[str, Any] = {"zero_shot_val": zs.to_dict(), "lr_sweep": [], "lora_sweep": [], "sign_flip": []}

        # 1. Learning-rate sweep per mode.
        best_lr: dict[str, float] = {}
        for mode, lrs in LR_GRID.items():
            best_wql = float("inf")
            for lr in lrs:
                lora = (
                    LoraConfig(r=8, lora_alpha=16, axes=("variate" if mode == "lora_variate" else "time",))
                    if mode.startswith("lora_")
                    else None
                )
                row = _train_and_eval(
                    base_model,
                    splits,
                    mode=mode,
                    lr=lr,
                    device=args.device,
                    mixed_precision=precision,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    lora_config=lora,
                )
                ds_res["lr_sweep"].append(row)
                wql = row["val_metrics"]["wql"]
                mase_v = row["val_metrics"]["mase"]
                print(
                    f"  [LR] {mode:13s} lr={lr:.0e} | wQL={wql:.4f} | MASE={mase_v:.4f} | {row['elapsed_s']:.1f}s",
                    flush=True,
                )
                if wql < best_wql:
                    best_wql = wql
                    best_lr[mode] = lr

        # 2. LoRA rank and targets sweep on lora_variate and lora_temporal at their best LRs.
        for mode in ("lora_variate", "lora_temporal"):
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
                    device=args.device,
                    mixed_precision=precision,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    lora_config=lora,
                )
                ds_res["lora_sweep"].append(row)
                print(
                    f"  [LoRA] {mode:13s} r={r:2d} targets={targets} | "
                    f"wQL={row['val_metrics']['wql']:.4f} | MASE={row['val_metrics']['mase']:.4f}",
                    flush=True,
                )

        # 3. Sign-flip ablation on lora_variate and head_only.
        for mode in ("head_only", "lora_variate", "full_ft"):
            lora = LoraConfig(r=8, lora_alpha=16, axes=("variate",)) if mode == "lora_variate" else None
            for sf in (False, True):
                row = _train_and_eval(
                    base_model,
                    splits,
                    mode=mode,
                    lr=best_lr[mode],
                    device=args.device,
                    mixed_precision=precision,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    lora_config=lora,
                    sign_flip=sf,
                )
                ds_res["sign_flip"].append(row)
                print(
                    f"  [SignFlip] {mode:13s} sign_flip={str(sf):5s} | "
                    f"wQL={row['val_metrics']['wql']:.4f} | MASE={row['val_metrics']['mase']:.4f}",
                    flush=True,
                )

        all_results["datasets"][splits.name] = ds_res

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved sweep results to {args.output}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
