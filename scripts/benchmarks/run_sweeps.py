"""Phase 3 hyperparameter & sign-flip sweeps to back the defaults in spec §5.4.

    uv run python scripts/benchmarks/run_sweeps.py --device cuda --sizes 5M 28M --bf16 --output /tmp/sweeps.json

Sweeps (scored strictly on validation splits — never on test sets):
1. Properly bracketed learning-rate grids for `head_only`, `lora_variate`, `lora_temporal`, and
   `full_ft` across realised volatility and retail demand, plus macro GIFT-Eval forgetting so
   aggressive LRs can be weighed against forgetting without touching the test split.
2. LoRA rank (`r in {4, 8, 16}`) and targets (`("q", "v")` vs `("q", "k", "v", "o")`).
3. Sign-flip augmentation (`sign_flip=False` vs `True`) on both one-signed domains.
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
from datasets import load_general_forgetting_by_domain, load_realised_volatility, load_retail_demand

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
    lora_config: LoraConfig | None = None,
    sign_flip: bool = True,
    seed: int = 0,
    gen_by_domain: dict[str, Any] | None = None,
    zs_by_domain: dict[str, Any] | None = None,
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
    ev_m = copy.deepcopy(base_model)
    ev_m.load_state_dict(merged_state_dict(trainer.model), strict=True)
    del trainer, model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    ev_m.to(device).eval()
    ev_m.transformer.rope.rebuild_inv_freq()
    fc = TuneTimestralForecaster(ev_m)
    metrics = evaluate_forecaster(fc, splits.val, seasonality=splits.seasonality, batch_size=batch_size)
    forg = (
        _eval_macro_forgetting(fc, zs_by_domain, gen_by_domain, batch_size)
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
        "lora": None if lora_config is None else lora_config.to_dict(),
        "sign_flip": sign_flip,
        "steps": res.steps,
        "best_step": res.best_step,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "val_metrics": metrics.to_dict(),
        "forgetting": forg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sizes", nargs="+", default=["5M", "28M"], choices=["5M", "28M", "123M", "209M"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--sign-flip", action="store_true", help="Enable sign-flip during LR/LoRA sweeps (default: False)"
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--check-forgetting", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    precision = "bf16" if args.bf16 else "fp32"
    eval_bs = args.eval_batch_size or args.batch_size
    datasets = [load_realised_volatility(), load_retail_demand()]
    gen_by_domain = load_general_forgetting_by_domain() if args.check_forgetting else None

    all_results: dict[str, Any] = {
        "precision": precision,
        "sign_flip": args.sign_flip,
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

        size_res: dict[str, Any] = {"datasets": {}}
        for splits in datasets:
            print(f"\n--- {size} | {splits.name} (train={len(splits.train)}, val={len(splits.val)}) ---", flush=True)
            zs = evaluate_forecaster(zs_fc, splits.val, seasonality=splits.seasonality, batch_size=eval_bs)
            print(f"  zero_shot val: wQL={zs.wql:.4f} | MASE={zs.mase:.4f} | cov80={zs.coverage_80:.3f}", flush=True)

            ds_res: dict[str, Any] = {"zero_shot_val": zs.to_dict(), "lr_sweep": [], "lora_sweep": [], "sign_flip": []}

            # 1. Properly bracketed learning-rate sweep per mode (at args.sign_flip, default False).
            best_lr: dict[str, float] = {}
            for mode, lrs in LR_GRID.items():
                best_wql = float("inf")
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
                    row = _train_and_eval(
                        base_model,
                        splits,
                        mode=mode,
                        lr=lr,
                        device=eval_dev,
                        mixed_precision=precision,
                        max_steps=args.max_steps,
                        batch_size=run_bs,
                        lora_config=lora,
                        sign_flip=args.sign_flip,
                        gen_by_domain=gen_by_domain,
                        zs_by_domain=zs_by_domain,
                    )
                    ds_res["lr_sweep"].append(row)
                    wql = row["val_metrics"]["wql"]
                    mase_v = row["val_metrics"]["mase"]
                    f_str = (
                        f" | macro_forg={row['forgetting']['macro_relative_delta_wql']:+.1%}"
                        if row["forgetting"] is not None
                        else ""
                    )
                    print(
                        f"  [LR] {mode:13s} lr={lr:.0e} | wQL={wql:.4f} | "
                        f"MASE={mase_v:.4f}{f_str} | {row['elapsed_s']:.1f}s",
                        flush=True,
                    )
                    if wql < best_wql:
                        best_wql = wql
                        best_lr[mode] = lr

            # 2. LoRA rank and targets sweep + sign-flip ablation on 5M.
            if size == "5M":
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
                            device=eval_dev,
                            mixed_precision=precision,
                            max_steps=args.max_steps,
                            batch_size=args.batch_size,
                            lora_config=lora,
                            sign_flip=True,
                        )
                        ds_res["lora_sweep"].append(row)
                        print(
                            f"  [LoRA] {mode:13s} r={r:2d} targets={targets} | "
                            f"wQL={row['val_metrics']['wql']:.4f} | MASE={row['val_metrics']['mase']:.4f}",
                            flush=True,
                        )

                for mode in ("head_only", "lora_variate", "lora_temporal", "full_ft"):
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
                            lora_config=lora,
                            sign_flip=sf,
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
