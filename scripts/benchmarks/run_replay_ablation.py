"""Phase 4 replay & frozen-base distillation ablation (spec §5.3).

    uv run python scripts/benchmarks/run_replay_ablation.py --device mps \
        --output docs/benchmarks/replay_ablation.json

Evaluates replay mixing (`replay_fraction`) and frozen-base quantile distillation
(`distill_weight`) using the built-in `SyntheticReplaySource` across the three high-forgetting
regimes measured in Phase 3:

1. `head_only` at shipped `3e-3` vs lower-LR `1e-3` control (`5M` across 3 seeds & `28M` seed 0,
   realised volatility & retail demand), where baseline macro GIFT-Eval forgetting reached `+5.0%`
   to `+13.7%`.
2. `lora_temporal` at higher LR `3e-3` vs shipped `1e-3` (`5M`, realised volatility, 3 seeds),
   where baseline macro forgetting spiked up to `+31.3%` in this harness on Apple GPU (`fp32`).
3. `full_ft` at higher LR `3e-4` vs `1e-4` and `2e-4` lower-LR controls, plus a 256-series
   (1,024-window) replay pool control (`5M`, retail demand, 3 seeds), where `3e-4` improved test
   wQL over `1e-4` at the cost of `+1.9%`–`+2.8%` retail forgetting.

Every run reports raw-space validation & test wQL/MASE through upstream `TimestralForecaster`
and equal-weighted 3-domain macro GIFT-Eval forgetting (`ett1`, `jena_weather`, `loop_seattle`).
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

from timestral_tune import LoraConfig, ReplayConfig, SyntheticReplaySource, TimestralTrainer, TrainingConfig
from timestral_tune.models import TuneTimestralForecaster, load_model, merged_state_dict
from timestral_tune.utils import compute_forgetting, evaluate_forecaster


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
        "per_domain": per_domain,
    }


def _run_one(
    base_cpu,
    splits,
    gen_by_domain: dict[str, Any],
    zs_by_domain: dict[str, Any],
    *,
    model_id: str,
    mode: str,
    lr: float,
    replay_cfg: ReplayConfig | None,
    device: str,
    mixed_precision: str,
    max_steps: int,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
    replay_series: int,
) -> dict[str, Any]:
    model = copy.deepcopy(base_cpu).to(device)
    model.transformer.rope.rebuild_inv_freq()

    lora_cfg = (
        LoraConfig(r=8, lora_alpha=16.0, axes=("time",), targets=("q", "k", "v", "o"), dropout=0.05)
        if mode == "lora_temporal"
        else None
    )
    replay_ds = (
        SyntheticReplaySource(
            num_series=replay_series,
            context_length=splits.train.context_length,
            forecast_horizon=splits.train.forecast_horizon,
            seed=seed + 2_000_003,
        )
        if replay_cfg is not None
        else None
    )
    cfg = TrainingConfig(
        model_name_or_path=model_id,
        mode=mode,  # type: ignore[arg-type]
        learning_rate=lr,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        max_steps=max_steps,
        warmup_steps=min(20, max_steps // 5),
        mixed_precision=mixed_precision,  # type: ignore[arg-type]
        device=device,
        seed=seed,
        eval_every=max(20, max_steps // 4),
        restore_best=True,
        sign_flip=True,
        log_every=max_steps + 1,
    )
    t0 = time.perf_counter()
    trainer = TimestralTrainer(
        cfg,
        splits.train,
        splits.val,
        lora_config=lora_cfg,
        replay_config=replay_cfg,
        replay_dataset=replay_ds,
        model=model,
    )
    res = trainer.fit()
    wall = time.perf_counter() - t0

    eval_model = copy.deepcopy(base_cpu)
    eval_model.load_state_dict(merged_state_dict(trainer.model), strict=True)
    eval_model = eval_model.to(device).eval()
    eval_model.transformer.rope.rebuild_inv_freq()
    fc = TuneTimestralForecaster(eval_model)

    val_m = evaluate_forecaster(fc, splits.val, seasonality=splits.seasonality, batch_size=eval_batch_size)
    test_m = evaluate_forecaster(fc, splits.test, seasonality=splits.seasonality, batch_size=eval_batch_size)
    forg = _macro_forgetting(fc, gen_by_domain, zs_by_domain, eval_batch_size)

    del trainer, model, eval_model, fc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return {
        "mode": mode,
        "lr": lr,
        "seed": seed,
        "replay_config": None if replay_cfg is None else replay_cfg.to_dict(),
        "wall_s": round(wall, 2),
        "best_step": res.best_step,
        "val": {"wql": val_m.wql, "mase": val_m.mase},
        "test": test_m.to_dict(),
        "forgetting": forg,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 4 replay and distillation ablation")
    p.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--replay-series", type=int, default=64)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--output", default="docs/benchmarks/replay_ablation.json")
    args = p.parse_args(argv)

    mp = "bf16" if args.bf16 else "fp32"
    print("== Loading pinned Phase 3 datasets ==", flush=True)
    vol = load_realised_volatility()
    ret = load_retail_demand()
    gen_by_domain = load_general_forgetting_by_domain()
    datasets_map = {"realised_volatility": vol, "retail_demand": ret}

    report: dict[str, Any] = {
        "device": args.device,
        "precision": mp,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "replay_series": args.replay_series,
        "seeds": args.seeds,
        "head_only": {},
        "lora_temporal_3e3": {},
        "full_ft_3e4": {},
    }

    # Preload 5M and 28M zero-shot baselines.
    models_cpu = {
        "5M": load_model("mistralai/Timestral-5M", device="cpu"),
        "28M": load_model("mistralai/Timestral-28M", device="cpu"),
    }
    zs_by_size: dict[str, dict[str, Any]] = {}
    for size, base_cpu in models_cpu.items():
        m_dev = copy.deepcopy(base_cpu).to(args.device).eval()
        m_dev.transformer.rope.rebuild_inv_freq()
        fc = TuneTimestralForecaster(m_dev)
        zs_by_size[size] = {
            dom: evaluate_forecaster(fc, ds, seasonality=24, batch_size=args.eval_batch_size)
            for dom, ds in gen_by_domain.items()
        }
        del m_dev, fc

    # 1. head_only at 3e-3 vs 1e-3 lower-LR control (5M across all seeds; 28M seed 0)
    head_variants: list[tuple[str, float, ReplayConfig | None]] = [
        ("lr1e-3_none", 1e-3, None),
        ("none", 3e-3, None),
        ("replay_0.25", 3e-3, ReplayConfig(replay_fraction=0.25, distill_weight=0.0)),
        ("distill_0.1", 3e-3, ReplayConfig(replay_fraction=0.0, distill_weight=0.1)),
        ("distill_0.5", 3e-3, ReplayConfig(replay_fraction=0.0, distill_weight=0.5)),
        ("distill_1.0", 3e-3, ReplayConfig(replay_fraction=0.0, distill_weight=1.0)),
        ("both_0.2_0.5", 3e-3, ReplayConfig(replay_fraction=0.2, distill_weight=0.5)),
    ]
    print("\n== 1. head_only (lr=3e-3 vs lr=1e-3 control) on 5M (multi-seed) & 28M (seed 0) ==", flush=True)
    for size in ("5M", "28M"):
        model_id = f"mistralai/Timestral-{size}"
        size_seeds = args.seeds if size == "5M" else [args.seeds[0]]
        report["head_only"][size] = {}
        for ds_name, splits in datasets_map.items():
            report["head_only"][size][ds_name] = {}
            for label, lr, rcfg in head_variants:
                rows = [
                    _run_one(
                        models_cpu[size],
                        splits,
                        gen_by_domain,
                        zs_by_size[size],
                        model_id=model_id,
                        mode="head_only",
                        lr=lr,
                        replay_cfg=rcfg,
                        device=args.device,
                        mixed_precision=mp,
                        max_steps=args.max_steps,
                        batch_size=args.batch_size,
                        eval_batch_size=args.eval_batch_size,
                        seed=s,
                        replay_series=args.replay_series,
                    )
                    for s in size_seeds
                ]
                report["head_only"][size][ds_name][label] = rows if len(rows) > 1 else rows[0]
                wqls = " / ".join(f"{r['test']['wql']:.4f}" for r in rows)
                forgs = " / ".join(f"{r['forgetting']['macro_relative_delta_wql']:+.1%}" for r in rows)
                print(
                    f"  [head_only {size} | {ds_name:19s} | {label:12s}] test={wqls} | forg={forgs}",
                    flush=True,
                )

    # 2. lora_temporal at 3e-3 vs 1e-3 (5M realised volatility, 3 seeds)
    lora_variants: list[tuple[str, float, ReplayConfig | None]] = [
        ("lr1e-3_none", 1e-3, None),
        ("lr3e-3_none", 3e-3, None),
        ("lr3e-3_replay_0.25", 3e-3, ReplayConfig(replay_fraction=0.25, distill_weight=0.0)),
        ("lr3e-3_distill_0.1", 3e-3, ReplayConfig(replay_fraction=0.0, distill_weight=0.1)),
        ("lr3e-3_distill_0.5", 3e-3, ReplayConfig(replay_fraction=0.0, distill_weight=0.5)),
        ("lr3e-3_distill_1.0", 3e-3, ReplayConfig(replay_fraction=0.0, distill_weight=1.0)),
        ("lr3e-3_both_0.2_0.5", 3e-3, ReplayConfig(replay_fraction=0.2, distill_weight=0.5)),
    ]
    print("\n== 2. lora_temporal (3e-3 forgetting cliff) on 5M volatility (3 seeds) ==", flush=True)
    for label, lr, rcfg in lora_variants:
        rows = []
        for seed in args.seeds:
            row = _run_one(
                models_cpu["5M"],
                vol,
                gen_by_domain,
                zs_by_size["5M"],
                model_id="mistralai/Timestral-5M",
                mode="lora_temporal",
                lr=lr,
                replay_cfg=rcfg,
                device=args.device,
                mixed_precision=mp,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                seed=seed,
                replay_series=args.replay_series,
            )
            rows.append(row)
        report["lora_temporal_3e3"][label] = rows
        wqls = " / ".join(f"{r['test']['wql']:.4f}" for r in rows)
        forgs = " / ".join(f"{r['forgetting']['macro_relative_delta_wql']:+.1%}" for r in rows)
        worst = max(r["forgetting"]["macro_relative_delta_wql"] for r in rows)
        print(f"  [lora_temporal 5M vol | {label:19s}] test={wqls} | forg={forgs} (worst {worst:+.1%})", flush=True)

    # 3. full_ft at 3e-4 vs 1e-4 & 2e-4 lower-LR controls + 256-series pool control (5M retail, 3 seeds)
    ft_variants: list[tuple[str, float, ReplayConfig | None, int]] = [
        ("lr1e-4_none", 1e-4, None, args.replay_series),
        ("lr2e-4_none", 2e-4, None, args.replay_series),
        ("lr3e-4_none", 3e-4, None, args.replay_series),
        ("lr3e-4_replay_0.25", 3e-4, ReplayConfig(replay_fraction=0.25, distill_weight=0.0), args.replay_series),
        ("lr3e-4_replay_0.25_256s", 3e-4, ReplayConfig(replay_fraction=0.25, distill_weight=0.0), 256),
        ("lr3e-4_distill_0.1", 3e-4, ReplayConfig(replay_fraction=0.0, distill_weight=0.1), args.replay_series),
        ("lr3e-4_distill_0.5", 3e-4, ReplayConfig(replay_fraction=0.0, distill_weight=0.5), args.replay_series),
        ("lr3e-4_distill_1.0", 3e-4, ReplayConfig(replay_fraction=0.0, distill_weight=1.0), args.replay_series),
        ("lr3e-4_both_0.2_0.5", 3e-4, ReplayConfig(replay_fraction=0.2, distill_weight=0.5), args.replay_series),
    ]
    print("\n== 3. full_ft (3e-4 vs 1e-4 & 2e-4 controls + 256-series pool) on 5M retail (3 seeds) ==", flush=True)
    for label, lr, rcfg, r_series in ft_variants:
        rows = []
        for seed in args.seeds:
            row = _run_one(
                models_cpu["5M"],
                ret,
                gen_by_domain,
                zs_by_size["5M"],
                model_id="mistralai/Timestral-5M",
                mode="full_ft",
                lr=lr,
                replay_cfg=rcfg,
                device=args.device,
                mixed_precision=mp,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                seed=seed,
                replay_series=r_series,
            )
            rows.append(row)
        report["full_ft_3e4"][label] = rows
        wqls = " / ".join(f"{r['test']['wql']:.4f}" for r in rows)
        forgs = " / ".join(f"{r['forgetting']['macro_relative_delta_wql']:+.1%}" for r in rows)
        worst = max(r["forgetting"]["macro_relative_delta_wql"] for r in rows)
        print(f"  [full_ft 5M retail    | {label:19s}] test={wqls} | forg={forgs} (worst {worst:+.1%})", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved replay ablation to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
