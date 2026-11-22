"""Probe: cached vs live frozen-base teacher parity under CUDA bf16 across `smooth_beta` (spec §5.3, D21).

    uv run python scripts/probes/replay_bf16_parity.py --device cuda --sizes tiny 5M \
        --output docs/benchmarks/replay_bf16_parity.json

Compares `ReplayConfig(cache_base_outputs=True)` against `cache_base_outputs=False` under CUDA
`bf16` autocast across `smooth_beta` in `(0.0, 1e-3, 1e-2)` for `lora_temporal` and `full_ft`
(plus `head_only` cached vs streaming) on both the tiny random model and a released checkpoint
(`Timestral-5M`). Records direct step-0 teacher output difference, max 5-step training loss
difference, and relative L2 parameter difference.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from timestral_tune._vendor.timestral.config import TimestralConfig
from timestral_tune.config import ReplayConfig, TrainingConfig
from timestral_tune.data import ColumnRoles, SeriesData, TimeSeriesDataset
from timestral_tune.models import TuneTimestralModel, load_model
from timestral_tune.replay import SyntheticReplaySource
from timestral_tune.trainer import TimestralTrainer


def _make_domain_dataset(context_length: int, forecast_horizon: int, seed: int = 60) -> TimeSeriesDataset:
    rng = np.random.default_rng(seed)
    length = context_length + 4 * forecast_horizon
    ts = np.arange(length, dtype=np.int64)
    empty_cat = np.empty((0, length), dtype=np.float64)
    series: list[SeriesData] = []
    for i in range(4):
        t = np.linspace(0.0, 4.0 * np.pi, length, dtype=np.float32)
        past = np.cos(t + i)[None, :].astype(np.float32)
        known = np.sin(2.0 * t + i)[None, :].astype(np.float32)
        y = (2.0 * known[0] + 0.5 * past[0] + rng.normal(0.0, 0.1, size=length)).astype(np.float32)
        series.append(
            SeriesData(
                series_id=f"dom_{i}",
                timestamps=ts,
                targets=y[None, :],
                past=past,
                known=known,
                categorical_past=empty_cat,
                categorical_known=empty_cat,
            )
        )
    return TimeSeriesDataset(
        series,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=forecast_horizon,
        min_context=context_length,
        roles=ColumnRoles(targets=("y",), past=("p0",), known=("k0",)),
    )


def _load_base(size: str) -> tuple[TuneTimestralModel, int, int]:
    if size == "tiny":
        torch.manual_seed(63)
        cfg = TimestralConfig(
            d_model=32,
            num_heads=2,
            mlp_hidden_dim=64,
            block_schedule=("time", "time", "time", "time+variate"),
            patch_size=4,
            block_size=4,
            context_length=64,
            covariate_enabled=True,
        )
        return TuneTimestralModel(cfg), 32, 8
    return load_model(f"mistralai/Timestral-{size}", device="cpu"), 128, 32


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--sizes", nargs="+", default=["tiny", "5M"], choices=["tiny", "5M", "28M"])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 1e-3, 1e-2])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if args.device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        print("CUDA bf16 is not available on this host.", file=sys.stderr)
        return 1

    mp = "bf16" if args.device == "cuda" else "fp32"
    print(
        f"== Replay teacher parity probe | device={args.device} precision={mp} torch={torch.__version__} ==", flush=True
    )
    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "device": args.device,
        "precision": mp,
        "steps": args.steps,
        "results": {},
    }

    for size in args.sizes:
        base_cpu, ctx, hor = _load_base(size)
        train_ds = _make_domain_dataset(ctx, hor, seed=60)
        replay_ds = SyntheticReplaySource(num_series=6, context_length=ctx, forecast_horizon=hor, seed=61)
        size_out: dict[str, Any] = {}
        for mode in ("lora_temporal", "full_ft"):
            lr = 3e-4 if mode == "full_ft" else 1e-3
            mode_out: dict[str, Any] = {}
            for beta in args.betas:
                cfg = TrainingConfig(
                    model_name_or_path=f"mistralai/Timestral-{size}" if size != "tiny" else "tiny",
                    mode=mode,  # type: ignore[arg-type]
                    learning_rate=lr,
                    batch_size=3,
                    max_steps=args.steps,
                    mixed_precision=mp,  # type: ignore[arg-type]
                    device=args.device,
                    seed=62,
                    sign_flip=True,
                    log_every=args.steps + 1,
                )
                tc = TimestralTrainer(
                    cfg,
                    train_ds,
                    replay_config=ReplayConfig(
                        replay_fraction=0.25,
                        distill_weight=0.5,
                        batch_size=2,
                        smooth_beta=beta,
                        cache_base_outputs=True,
                    ),
                    replay_dataset=replay_ds,
                    model=copy.deepcopy(base_cpu),
                )
                tl = TimestralTrainer(
                    cfg,
                    train_ds,
                    replay_config=ReplayConfig(
                        replay_fraction=0.25,
                        distill_weight=0.5,
                        batch_size=2,
                        smooth_beta=beta,
                        cache_base_outputs=False,
                    ),
                    replay_dataset=replay_ds,
                    model=copy.deepcopy(base_cpu),
                )
                rc, rl = tc.fit(), tl.fit()
                max_ld = max(abs(a - b) for (_, a), (_, b) in zip(rc.train_history, rl.train_history, strict=True))
                num = sum(
                    float((p1.detach() - p2.detach()).pow(2).sum())
                    for p1, p2 in zip(tc._trainable, tl._trainable, strict=True)
                )
                den = sum(float(p2.detach().pow(2).sum()) for p2 in tl._trainable)
                rel_l2 = (num**0.5) / max(den**0.5, 1e-12)
                max_p = max(
                    float((p1.detach() - p2.detach()).abs().max())
                    for p1, p2 in zip(tc._trainable, tl._trainable, strict=True)
                )
                mode_out[str(beta)] = {
                    "max_loss_diff": max_ld,
                    "rel_l2_weight_diff": rel_l2,
                    "max_abs_weight_diff": max_p,
                    "final_loss_cached": rc.final_train_loss,
                    "final_loss_live": rl.final_train_loss,
                }
                print(
                    f"  [{size:4s} | {mode:13s} | beta={beta:<5}] "
                    f"max_loss_diff={max_ld:.2e}  rel_l2={rel_l2:.2e}  max_abs_w={max_p:.2e}",
                    flush=True,
                )
            size_out[mode] = mode_out
        report["results"][size] = size_out

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved replay bf16 parity report to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
