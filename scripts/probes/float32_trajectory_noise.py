"""Probe: measure one-process float32 `TimestralTrainer.fit()` sensitivity to intra-op thread count.

    uv run python scripts/probes/float32_trajectory_noise.py
    uv run --isolated --python 3.12 --with "torch==2.4.0" python scripts/probes/float32_trajectory_noise.py

Runs entirely on CPU without network access and without importing from `tests/`. For each training
mode (`head_only`, `lora_variate`, `lora_temporal`, `full_ft`) and replay/distillation regime
(`none`, `replay_only`, `distill_only`, `both`), runs short deterministic one-process `fit()`
trajectories on the tiny random model (`_common.tiny_random_forecaster`) in float32 under intra-op
thread counts 1, 2, and 4, and reports pairwise `param_update = final - init` relative L2
differences (plus worst `final_params` and step-loss relative differences).

Always exits 0 (measurement probe, not a grader) and restores the caller's intra-op thread count
before exiting. On macOS arm64 thread count does not perturb float32 reductions, so every cell
prints `0.00e+00`; on Linux x86_64 changing thread count shifts float32 GEMM/reduction rounding
and reaches `O(1e-4)` on `lora_variate` `param_update`, motivating T15's `TRAJ_UPDATE_TOL` tier.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import platform
import sys

import numpy as np
import torch
from _common import PATCH, rel, tiny_random_forecaster

from timestral_tune import LoraConfig, ReplayConfig, TimestralTrainer, TrainingConfig
from timestral_tune.data import ColumnRoles, SeriesData, TimeSeriesDataset

MODES: tuple[str, ...] = ("head_only", "lora_variate", "lora_temporal", "full_ft")
REGIMES: tuple[tuple[str, float, float], ...] = (
    ("none", 0.0, 0.0),
    ("replay_only", 0.25, 0.0),
    ("distill_only", 0.0, 0.5),
    ("both", 0.25, 0.5),
)


def _make_dataset(
    num_windows: int,
    *,
    context_patches: int,
    horizon_patches: int,
    covariates: bool,
    seed: int,
) -> TimeSeriesDataset:
    """Self-contained synthetic dataset with unequal per-window scored counts."""
    rng = np.random.default_rng(seed)
    ctx_len = context_patches * PATCH
    hor_len = horizon_patches * PATCH
    total = ctx_len + hor_len
    ts = np.arange(total, dtype=np.int64)
    empty_f = np.empty((0, total), dtype=np.float32)
    empty_c = np.empty((0, total), dtype=np.float64)

    series: list[SeriesData] = []
    for i in range(num_windows):
        values = (15.0 + 3.0 * np.sin(0.2 * ts + 0.5 * i) + rng.normal(0.0, 0.8, size=total)).astype(np.float32)
        # Drop a varying tail of horizon targets while keeping >= 1 scored step in every window.
        drop = (i * 7) % (hor_len - 1)
        if drop > 0:
            values[total - drop :] = np.nan
        if covariates:
            past = rng.normal(0.0, 1.0, size=(1, total)).astype(np.float32)
            known = rng.normal(0.0, 1.0, size=(1, total)).astype(np.float32)
        else:
            past, known = empty_f, empty_f
        series.append(SeriesData(f"s{i}", ts, values[None, :], past, known, empty_c, empty_c))

    roles = ColumnRoles(targets=("y",), past=("p0",) if covariates else (), known=("k0",) if covariates else ())
    return TimeSeriesDataset(
        series,
        context_length=ctx_len,
        forecast_horizon=hor_len,
        stride=hor_len,
        min_context=ctx_len,
        roles=roles,
    )


def _lora_for(mode: str) -> LoraConfig | None:
    if mode == "lora_variate":
        return LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("q", "k", "v", "o"), dropout=0.0)
    if mode == "lora_temporal":
        return LoraConfig(r=4, lora_alpha=8, axes=("time",), targets=("q", "k", "v", "o"), dropout=0.0)
    return None


def _flat_trainable(trainer: TimestralTrainer) -> torch.Tensor:
    return torch.cat([p.detach().cpu().double().reshape(-1) for p in trainer._trainable])


def _scalar_rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def _run_one(
    base_model: torch.nn.Module,
    domain_ds: TimeSeriesDataset,
    replay_ds: TimeSeriesDataset,
    *,
    mode: str,
    replay_fraction: float,
    distill_weight: float,
    batch_size: int,
    steps: int,
    seed: int,
    threads: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    torch.set_num_threads(threads)
    cfg = TrainingConfig(
        model_name_or_path="local-probe",
        mode=mode,
        device="cpu",
        batch_size=batch_size,
        epochs=1,
        max_steps=steps,
        warmup_steps=1,
        lr_schedule="cosine",
        learning_rate=1e-3 if mode == "full_ft" else 1e-2,
        weight_decay=0.01,
        gradient_clip_val=1.0,
        shuffle=False,
        sign_flip=False,
        restore_best=False,
        cache_frozen_features=False,
        seed=seed + 107,
    )
    use_replay = replay_fraction > 0.0 or distill_weight > 0.0
    rcfg = (
        ReplayConfig(
            replay_fraction=replay_fraction,
            distill_weight=distill_weight,
            batch_size=batch_size,
            sign_flip=False,
            cache_base_outputs=True,
        )
        if use_replay
        else None
    )
    trainer = TimestralTrainer(
        cfg,
        domain_ds,
        lora_config=_lora_for(mode),
        replay_config=rcfg,
        replay_dataset=replay_ds if use_replay else None,
        model=copy.deepcopy(base_model),
    )
    init_p = _flat_trainable(trainer).clone()
    res = trainer.fit()
    final_p = _flat_trainable(trainer).clone()
    losses = [loss for _, loss in res.train_history]
    return final_p - init_p, final_p, losses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()

    orig_threads = torch.get_num_threads()
    thread_counts = tuple(args.threads)
    pairs = list(itertools.combinations(thread_counts, 2))

    print(
        f"== float32_trajectory_noise | torch {torch.__version__} | "
        f"{platform.system()} {platform.machine()} | threads={orig_threads} | "
        f"batch_sizes={list(args.batch_sizes)} | steps={args.steps} | seed={args.seed} =="
    )

    try:
        base_model = tiny_random_forecaster(seed=args.seed).model
        pair_headers = " ".join(f"{f'{a}v{b}':>10}" for a, b in pairs)
        print(f"\n{'bs':>3} {'mode':<14} {'regime':<13}  {pair_headers}  {'max_param':>10} {'max_loss':>10}")

        worst_update = 0.0
        worst_update_label = ""
        worst_param = 0.0
        worst_loss = 0.0

        for bs in args.batch_sizes:
            domain_ds = _make_dataset(
                bs * 2 + 1,
                context_patches=2,
                horizon_patches=1,
                covariates=True,
                seed=args.seed + 1001,
            )
            replay_ds = _make_dataset(
                bs + 3,
                context_patches=2,
                horizon_patches=1,
                covariates=False,
                seed=args.seed + 1002,
            )

            for mode in MODES:
                for regime, rf, dw in REGIMES:
                    runs = {
                        t: _run_one(
                            base_model,
                            domain_ds,
                            replay_ds,
                            mode=mode,
                            replay_fraction=rf,
                            distill_weight=dw,
                            batch_size=bs,
                            steps=args.steps,
                            seed=args.seed,
                            threads=t,
                        )
                        for t in thread_counts
                    }
                    upd_diffs: list[float] = []
                    row_max_param = 0.0
                    row_max_loss = 0.0
                    for a, b in pairs:
                        ua, pa, la = runs[a]
                        ub, pb, lb = runs[b]
                        u_diff = rel(ub, ua)
                        p_diff = rel(pb, pa)
                        l_diff = max(_scalar_rel(xb, xa) for xa, xb in zip(la, lb, strict=True))
                        upd_diffs.append(u_diff)
                        row_max_param = max(row_max_param, p_diff)
                        row_max_loss = max(row_max_loss, l_diff)
                        if u_diff > worst_update:
                            worst_update = u_diff
                            worst_update_label = f"bs={bs} {mode}/{regime} ({a}v{b})"
                    worst_param = max(worst_param, row_max_param)
                    worst_loss = max(worst_loss, row_max_loss)

                    pair_cols = " ".join(f"{d:>10.2e}" for d in upd_diffs)
                    print(f"{bs:>3} {mode:<14} {regime:<13}  {pair_cols}  {row_max_param:>10.2e} {row_max_loss:>10.2e}")

        print(
            f"\nsummary: max_param_update_rel_l2={worst_update:.2e} ({worst_update_label or 'none'}) | "
            f"max_final_params_rel_l2={worst_param:.2e} | max_loss_rel={worst_loss:.2e}"
        )
    finally:
        torch.set_num_threads(orig_threads)

    return 0


if __name__ == "__main__":
    sys.exit(main())
