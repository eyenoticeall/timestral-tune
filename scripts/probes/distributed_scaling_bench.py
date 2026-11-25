#!/usr/bin/env python3
"""Stage 3 probe: 1/2/4-GPU throughput, VRAM, and per-step overhead scaling benchmark (D22).

Directly measures the fourth open Phase 5 Stage 3 item (`docs/SPEC.md` §10, §12, D22;
`docs/phases/phase-5-multi-gpu.md` §3) without modifying `src/`:

- **Pass A (uninstrumented throughput & training VRAM):** runs warmup + timed steps with zero
  mid-step host-device syncs (using non-blocking CUDA events for median step time and wall-clock
  bracketing over the timed window for exact mean `step_ms` and `windows/s`), reporting weak and
  strong scaling speedup and efficiency across `N in {1, 2, 4}`, plus peak `train_vram_mib` and
  `train_reserved_mib` (including `full_ft` with live per-rank `_frozen_base` distillation).
- **Pass B (validation VRAM check):** runs `fit()` with `val_dataset` while `GradientReducer`'s
  flat gradient buffer remains allocated in accelerator memory, reporting `val_vram_mib` and
  `val_vram_delta_mib = val_vram_mib - train_vram_mib`.
- **Pass C (profiled per-step cost breakdown, `N > 1`):** instruments `_DistributedStrategy` and
  `GradientReducer.reduce()` (by wrapping `dist.all_reduce` during `reduce()`, duplicating zero
  library logic) to split step time into `precheck_ms`, `fwd_ms`, `stats_ar_ms`, `bwd_ms`,
  `pack_ms`, `grad_ar_ms`, `unpack_ms`, `opt_ms`, and `bcast_ms` — directly testing D22's
  `15–20%` flat-buffer `all_reduce` threshold.

Examples:
    # Local CPU/gloo smoke test on tiny_random:
    TMPDIR=/tmp uv run python scripts/probes/distributed_scaling_bench.py \\
        --backend gloo --sizes tiny --ranks 1 2 --warmup 1 --steps 3 --profile-steps 2 \\
        --json-out /tmp/scaling_cpu.json

    # Full 4-GPU A100 scaling & VRAM study:
    TMPDIR=/tmp uv run python scripts/probes/distributed_scaling_bench.py \\
        --backend nccl --sizes 28M 123M 209M --ranks 1 2 4 \\
        --json-out stage3_scaling.json
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from _common import PATCH, SIZES, load, tiny_random_forecaster
from _dist_common import (
    check_tmpdir_for_workers,
    cuda_env_banner_parts,
    git_head_commit,
    pick_free_port,
    require_positive_finite,
    validate_distributed_device_map,
    write_json_atomic,
)

from timestral_tune import LoraConfig, ReplayConfig, TimestralTrainer, TrainingConfig
from timestral_tune.config import AdaptationMode
from timestral_tune.data import ColumnRoles, SeriesData, TimeSeriesDataset
from timestral_tune.models import TuneTimestralModel

ALL_SIZE_CHOICES: tuple[str, ...] = ("tiny", *SIZES)
ScalingKind = Literal["weak", "strong"]


@dataclass(frozen=True)
class BenchSpec:
    """One scaling configuration benchmarked across requested `ranks`."""

    key: str
    size: str
    mode: AdaptationMode
    scaling: ScalingKind
    ctx_len: int
    horizon: int
    rows_per_series: int
    batch_spec: int  # per-rank batch if scaling=='weak'; global batch if scaling=='strong'
    compile_model: bool = False
    with_distill: bool = False
    distill_batch_per_rank: int = 8
    check_val_vram: bool = True


def build_bench_specs(
    sizes: Sequence[str],
    *,
    backend: Literal["gloo", "nccl"],
    include_compile: bool = True,
    include_long_context: bool = True,
) -> list[BenchSpec]:
    """Build the default Stage 3 scaling & VRAM benchmark matrix for `sizes`."""
    specs: list[BenchSpec] = []
    if backend == "gloo":
        for size in sizes:
            specs.extend(
                [
                    BenchSpec(
                        key=f"{size}_full_ft_eager_ctx64_weak_b2",
                        size=size,
                        mode="full_ft",
                        scaling="weak",
                        ctx_len=64,
                        horizon=64,
                        rows_per_series=5,
                        batch_spec=2,
                        compile_model=False,
                        with_distill=False,
                    ),
                    BenchSpec(
                        key=f"{size}_lora_temporal_eager_ctx64_weak_b2",
                        size=size,
                        mode="lora_temporal",
                        scaling="weak",
                        ctx_len=64,
                        horizon=64,
                        rows_per_series=5,
                        batch_spec=2,
                        compile_model=False,
                        with_distill=False,
                    ),
                    BenchSpec(
                        key=f"{size}_full_ft_distill_ctx64_weak_b2",
                        size=size,
                        mode="full_ft",
                        scaling="weak",
                        ctx_len=64,
                        horizon=64,
                        rows_per_series=5,
                        batch_spec=2,
                        compile_model=False,
                        with_distill=True,
                        distill_batch_per_rank=2,
                    ),
                    BenchSpec(
                        key=f"{size}_full_ft_eager_ctx64_strong_g4",
                        size=size,
                        mode="full_ft",
                        scaling="strong",
                        ctx_len=64,
                        horizon=64,
                        rows_per_series=5,
                        batch_spec=4,
                        compile_model=False,
                        with_distill=False,
                        check_val_vram=False,
                    ),
                ]
            )
        return specs

    # CUDA/NCCL matrix matching docs/phases/phase-5-multi-gpu.md §3 & spec §10/§12.
    compile_flags = (False, True) if include_compile else (False,)
    for size in sizes:
        # 1. Short context (ctx=256, H=96, cov5): weak (B_rank=16) and strong (B_global=32).
        for mode in ("full_ft", "lora_temporal"):
            for comp in compile_flags:
                tag = "compile" if comp else "eager"
                specs.append(
                    BenchSpec(
                        key=f"{size}_{mode}_{tag}_ctx256_weak_b16",
                        size=size,
                        mode=mode,
                        scaling="weak",
                        ctx_len=256,
                        horizon=96,
                        rows_per_series=5,
                        batch_spec=16,
                        compile_model=comp,
                        check_val_vram=not comp,
                    )
                )
                specs.append(
                    BenchSpec(
                        key=f"{size}_{mode}_{tag}_ctx256_strong_g32",
                        size=size,
                        mode=mode,
                        scaling="strong",
                        ctx_len=256,
                        horizon=96,
                        rows_per_series=5,
                        batch_spec=32,
                        compile_model=comp,
                        check_val_vram=False,
                    )
                )
        # 2. full_ft + distillation (live per-rank _frozen_base VRAM + 4th pre-check collective).
        specs.append(
            BenchSpec(
                key=f"{size}_full_ft_distill_eager_ctx256_weak_b16",
                size=size,
                mode="full_ft",
                scaling="weak",
                ctx_len=256,
                horizon=96,
                rows_per_series=5,
                batch_spec=16,
                compile_model=False,
                with_distill=True,
                distill_batch_per_rank=8,
                check_val_vram=True,
            )
        )
        # 3. Long context (ctx=1,024 & ctx=4,096) on 123M / 209M (or any non-tiny size if only 1 requested).
        if include_long_context and size in ("123M", "209M"):
            for mode in ("full_ft", "lora_temporal"):
                for comp in compile_flags:
                    tag = "compile" if comp else "eager"
                    specs.append(
                        BenchSpec(
                            key=f"{size}_{mode}_{tag}_ctx1024_weak_b8",
                            size=size,
                            mode=mode,
                            scaling="weak",
                            ctx_len=1024,
                            horizon=96,
                            rows_per_series=5,
                            batch_spec=8,
                            compile_model=comp,
                            check_val_vram=not comp,
                        )
                    )
                specs.append(
                    BenchSpec(
                        key=f"{size}_{mode}_eager_ctx4096_weak_b2",
                        size=size,
                        mode=mode,
                        scaling="weak",
                        ctx_len=4096,
                        horizon=96,
                        rows_per_series=5,
                        batch_spec=2,
                        compile_model=False,
                        check_val_vram=True,
                    )
                )
    return specs


def resolve_per_rank_batch(spec: BenchSpec, world_size: int) -> int | None:
    """Return per-rank batch size for `(spec, world_size)`, or `None` if indivisible."""
    if spec.scaling == "weak":
        return spec.batch_spec
    if spec.batch_spec % world_size != 0:
        return None
    return spec.batch_spec // world_size


# --------------------------------------------------------------------------- dataset & model helpers


def _make_cov_dataset(
    num_windows: int,
    *,
    ctx_len: int,
    horizon: int,
    rows_per_series: int,
    seed: int,
) -> TimeSeriesDataset:
    """Build a synthetic `TimeSeriesDataset` with 1 target + past/known covariates."""
    if ctx_len % PATCH != 0:
        raise ValueError(f"ctx_len must be a multiple of PATCH={PATCH}, got {ctx_len}")
    rng = np.random.default_rng(seed)
    total = ctx_len + horizon
    ts = np.arange(total, dtype=np.int64)
    empty_c = np.empty((0, total), dtype=np.float64)

    n_cov = max(0, rows_per_series - 1)
    n_known = n_cov // 2
    n_past = n_cov - n_known
    past_names = tuple(f"p{k}" for k in range(n_past))
    known_names = tuple(f"k{k}" for k in range(n_known))

    series: list[SeriesData] = []
    for i in range(num_windows):
        values = (100.0 + 3.0 * np.sin(0.05 * ts + 0.3 * i) + rng.normal(0.0, 1.0, size=total)).astype(np.float32)
        past = (
            rng.normal(0.0, 1.0, size=(n_past, total)).astype(np.float32)
            if n_past > 0
            else np.empty((0, total), dtype=np.float32)
        )
        known = (
            rng.normal(0.0, 1.0, size=(n_known, total)).astype(np.float32)
            if n_known > 0
            else np.empty((0, total), dtype=np.float32)
        )
        series.append(SeriesData(f"s{i}", ts, values[None, :], past, known, empty_c, empty_c))

    roles = ColumnRoles(targets=("y",), past=past_names, known=known_names)
    return TimeSeriesDataset(
        series,
        context_length=ctx_len,
        forecast_horizon=horizon,
        stride=horizon,
        min_context=ctx_len,
        roles=roles,
    )


class _ModelCache:
    def __init__(self) -> None:
        self._cpu_models: dict[str, TuneTimestralModel] = {}

    def clone(self, size: str) -> TuneTimestralModel:
        if size not in self._cpu_models:
            if size == "tiny":
                self._cpu_models[size] = tiny_random_forecaster(seed=0).model.cpu().eval()
            else:
                self._cpu_models[size] = load(size, device="cpu").model.cpu().eval()
        import copy

        return copy.deepcopy(self._cpu_models[size])


def _lora_for(mode: AdaptationMode) -> LoraConfig | None:
    if mode == "lora_temporal":
        return LoraConfig(r=8, lora_alpha=16.0, axes=("time",), targets=("q", "k", "v", "o"), dropout=0.0)
    if mode == "lora_variate":
        return LoraConfig(r=8, lora_alpha=16.0, axes=("variate",), targets=("q", "k", "v", "o"), dropout=0.0)
    return None


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_across_ranks(val: float, device: torch.device) -> float:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return val
    t = torch.tensor([float(val)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _barrier_rank(backend: Literal["gloo", "nccl"], device: torch.device) -> None:
    if not dist.is_initialized():
        return
    if backend == "nccl":
        idx = 0 if device.index is None else int(device.index)
        dist.barrier(device_ids=[idx])
    else:
        dist.barrier()


# --------------------------------------------------------------------------- Pass A, B, C execution


def _make_trainer(
    cache: _ModelCache,
    spec: BenchSpec,
    *,
    per_rank_batch: int,
    max_steps: int,
    device_arg: str,
    mixed_precision: Literal["fp32", "bf16"],
    dom_ds: TimeSeriesDataset,
    val_ds: TimeSeriesDataset | None = None,
    rep_ds: TimeSeriesDataset | None = None,
    eval_every: int | None = None,
) -> TimestralTrainer:
    rcfg: ReplayConfig | None = None
    if spec.with_distill:
        rcfg = ReplayConfig(
            replay_fraction=0.0,
            distill_weight=0.1,
            batch_size=spec.distill_batch_per_rank,
            sign_flip=False,
            cache_base_outputs=False,
            smooth_beta=1e-3,
        )
    cfg = TrainingConfig(
        model_name_or_path=f"bench-{spec.size}",
        mode=spec.mode,
        device=device_arg,
        mixed_precision=mixed_precision,
        compile_model=spec.compile_model,
        compile_mode="default",
        batch_size=per_rank_batch,
        eval_batch_size=per_rank_batch,
        max_steps=max_steps,
        eval_every=eval_every,
        warmup_steps=1,
        lr_schedule="constant",
        learning_rate=1e-4 if spec.mode == "full_ft" else 1e-3,
        weight_decay=0.01,
        gradient_clip_val=1.0,
        shuffle=False,
        sign_flip=False,
        cache_frozen_features=False,
        restore_best=False,
        seed=42,
        log_every=10_000,
    )
    if spec.compile_model and hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    model = cache.clone(spec.size)
    return TimestralTrainer(
        cfg,
        dom_ds,
        val_ds,
        lora_config=_lora_for(spec.mode),
        replay_config=rcfg,
        replay_dataset=rep_ds if spec.with_distill else None,
        model=model,
    )


def _run_pass_a_throughput(
    cache: _ModelCache,
    spec: BenchSpec,
    *,
    world_size: int,
    per_rank_batch: int,
    backend: Literal["gloo", "nccl"],
    device_arg: str,
    mixed_precision: Literal["fp32", "bf16"],
    dom_ds: TimeSeriesDataset,
    rep_ds: TimeSeriesDataset | None,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, Any]:
    """Uninstrumented throughput and training-only VRAM measurement (zero mid-step syncs)."""
    total_steps = warmup_steps + timed_steps
    trainer = _make_trainer(
        cache,
        spec,
        per_rank_batch=per_rank_batch,
        max_steps=total_steps,
        device_arg=device_arg,
        mixed_precision=mixed_precision,
        dom_ds=dom_ds,
        rep_ds=rep_ds,
    )
    device = trainer.device
    is_cuda = device.type == "cuda"

    step_counter = 0
    t_window_start: float | None = None
    t_window_end: float | None = None
    cuda_events: list[torch.cuda.Event] = []
    cpu_stamps: list[float] = []
    flat_buf_mib = 0.0

    orig_make_strategy = trainer._make_fit_strategy

    def _wrapped_make_strategy() -> Any:
        strategy = orig_make_strategy()
        orig_prepare = strategy.prepare_fit
        orig_on_step_end = strategy.on_step_end

        def _wrapped_prepare() -> None:
            nonlocal flat_buf_mib
            orig_prepare()
            reducer = getattr(strategy, "_reducer", None)
            if reducer is not None:
                flat_buf_mib = (reducer._buffer.numel() * reducer._buffer.element_size()) / (2**20)

        def _wrapped_on_step_end(step: int, loss: float, lr: float) -> float:
            nonlocal step_counter, t_window_start, t_window_end
            out = orig_on_step_end(step, loss, lr)
            step_counter += 1
            if step_counter == warmup_steps:
                _sync_device(device)
                _barrier_rank(backend, device)
                if is_cuda:
                    torch.cuda.reset_peak_memory_stats(device)
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    cuda_events.append(ev)
                else:
                    cpu_stamps.append(time.perf_counter())
                t_window_start = time.perf_counter()
            elif step_counter > warmup_steps:
                if is_cuda:
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    cuda_events.append(ev)
                else:
                    cpu_stamps.append(time.perf_counter())
                if step_counter == total_steps:
                    _sync_device(device)
                    t_window_end = time.perf_counter()
            return out

        strategy.prepare_fit = _wrapped_prepare
        strategy.on_step_end = _wrapped_on_step_end
        return strategy

    trainer._make_fit_strategy = _wrapped_make_strategy  # type: ignore[method-assign]
    trainable_params = sum(p.numel() for p in trainer._trainable)

    res = trainer.fit()
    assert res.steps == total_steps, f"expected {total_steps} steps, got {res.steps}"
    assert t_window_start is not None and t_window_end is not None

    if is_cuda:
        _sync_device(device)
        step_ms_samples = [cuda_events[i].elapsed_time(cuda_events[i + 1]) for i in range(len(cuda_events) - 1)]
        train_vram_mib = torch.cuda.max_memory_allocated(device) / (2**20)
        train_reserved_mib = torch.cuda.max_memory_reserved(device) / (2**20)
    else:
        step_ms_samples = [(cpu_stamps[i + 1] - cpu_stamps[i]) * 1e3 for i in range(len(cpu_stamps) - 1)]
        train_vram_mib = 0.0
        train_reserved_mib = 0.0

    mean_step_ms = ((t_window_end - t_window_start) / timed_steps) * 1e3
    median_step_ms = float(statistics.median(step_ms_samples)) if step_ms_samples else mean_step_ms

    # Aggregate max across ranks so slow-rank stragglers are visible.
    mean_step_ms = _max_across_ranks(mean_step_ms, device)
    median_step_ms = _max_across_ranks(median_step_ms, device)
    train_vram_mib = _max_across_ranks(train_vram_mib, device)
    train_reserved_mib = _max_across_ranks(train_reserved_mib, device)
    flat_buf_mib = _max_across_ranks(flat_buf_mib, device)

    eff_batch = world_size * per_rank_batch
    windows_per_s = (eff_batch / (mean_step_ms / 1e3)) if mean_step_ms > 0.0 else float("nan")

    del trainer
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    return {
        "trainable_params": trainable_params,
        "effective_batch": eff_batch,
        "mean_step_ms": mean_step_ms,
        "median_step_ms": median_step_ms,
        "windows_per_s": windows_per_s,
        "train_vram_mib": train_vram_mib,
        "train_reserved_mib": train_reserved_mib,
        "flat_buf_mib": flat_buf_mib,
    }


def _run_pass_b_val_vram(
    cache: _ModelCache,
    spec: BenchSpec,
    *,
    per_rank_batch: int,
    device_arg: str,
    mixed_precision: Literal["fp32", "bf16"],
    dom_ds: TimeSeriesDataset,
    val_ds: TimeSeriesDataset,
    rep_ds: TimeSeriesDataset | None,
) -> dict[str, float]:
    """Measure peak VRAM when `fit()` runs sharded validation while `GradientReducer`'s buffer is live."""
    trainer = _make_trainer(
        cache,
        spec,
        per_rank_batch=per_rank_batch,
        max_steps=2,
        device_arg=device_arg,
        mixed_precision=mixed_precision,
        dom_ds=dom_ds,
        val_ds=val_ds,
        rep_ds=rep_ds,
        eval_every=2,
    )
    device = trainer.device
    is_cuda = device.type == "cuda"
    if is_cuda:
        _sync_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    trainer.fit()
    if is_cuda:
        _sync_device(device)
        val_vram_mib = _max_across_ranks(torch.cuda.max_memory_allocated(device) / (2**20), device)
        val_reserved_mib = _max_across_ranks(torch.cuda.max_memory_reserved(device) / (2**20), device)
    else:
        val_vram_mib = 0.0
        val_reserved_mib = 0.0

    del trainer
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()
    return {"val_vram_mib": val_vram_mib, "val_reserved_mib": val_reserved_mib}


def _run_pass_c_breakdown(
    cache: _ModelCache,
    spec: BenchSpec,
    *,
    per_rank_batch: int,
    device_arg: str,
    mixed_precision: Literal["fp32", "bf16"],
    dom_ds: TimeSeriesDataset,
    rep_ds: TimeSeriesDataset | None,
    warmup_steps: int,
    profile_steps: int,
) -> dict[str, float]:
    """Measure per-step stage and `GradientReducer` pack/all_reduce/unpack breakdown (N > 1)."""
    total_steps = warmup_steps + profile_steps
    trainer = _make_trainer(
        cache,
        spec,
        per_rank_batch=per_rank_batch,
        max_steps=total_steps,
        device_arg=device_arg,
        mixed_precision=mixed_precision,
        dom_ds=dom_ds,
        rep_ds=rep_ds,
    )
    device = trainer.device
    keys = ("precheck", "fwd", "stats_ar", "bwd", "pack", "grad_ar", "unpack", "opt", "bcast")
    totals: dict[str, float] = dict.fromkeys(keys, 0.0)
    step_idx = 0

    def _active() -> bool:
        return step_idx >= warmup_steps

    orig_make_strategy = trainer._make_fit_strategy
    orig_apply_opt = trainer._apply_optimizer_step

    def _timed_apply_opt() -> None:
        if not _active():
            orig_apply_opt()
            return
        _sync_device(device)
        t0 = time.perf_counter()
        orig_apply_opt()
        _sync_device(device)
        totals["opt"] += (time.perf_counter() - t0) * 1e3

    def _wrapped_make_strategy() -> Any:
        strategy = orig_make_strategy()
        orig_prepare = strategy.prepare_fit
        orig_precheck = strategy._precheck_domain_for_replay
        orig_local = strategy._run_local_stage
        orig_stats = strategy._reduce_step_stats
        orig_bwd = strategy._backward_local_terms
        orig_on_step_end = strategy.on_step_end

        def _wrapped_prepare() -> None:
            orig_prepare()
            reducer = strategy._reducer
            assert reducer is not None
            orig_reduce = reducer.reduce

            def _timed_reduce() -> None:
                if not _active():
                    orig_reduce()
                    return
                ar_t0: float | None = None
                ar_t1: float | None = None
                orig_dist_ar = dist.all_reduce

                def _intercept_ar(*a: Any, **kw: Any) -> Any:
                    nonlocal ar_t0, ar_t1
                    _sync_device(device)
                    ar_t0 = time.perf_counter()
                    out = orig_dist_ar(*a, **kw)
                    _sync_device(device)
                    ar_t1 = time.perf_counter()
                    return out

                _sync_device(device)
                t0 = time.perf_counter()
                dist.all_reduce = _intercept_ar  # type: ignore[assignment]
                try:
                    orig_reduce()
                finally:
                    dist.all_reduce = orig_dist_ar  # type: ignore[assignment]
                _sync_device(device)
                t1 = time.perf_counter()
                if ar_t0 is not None and ar_t1 is not None:
                    totals["pack"] += (ar_t0 - t0) * 1e3
                    totals["grad_ar"] += (ar_t1 - ar_t0) * 1e3
                    totals["unpack"] += (t1 - ar_t1) * 1e3

            reducer.reduce = _timed_reduce  # type: ignore[method-assign]

        def _timed_precheck(*a: Any, **kw: Any) -> Any:
            if not _active():
                return orig_precheck(*a, **kw)
            _sync_device(device)
            t0 = time.perf_counter()
            out = orig_precheck(*a, **kw)
            _sync_device(device)
            totals["precheck"] += (time.perf_counter() - t0) * 1e3
            return out

        def _timed_local(*a: Any, **kw: Any) -> Any:
            if not _active():
                return orig_local(*a, **kw)
            _sync_device(device)
            t0 = time.perf_counter()
            out = orig_local(*a, **kw)
            _sync_device(device)
            totals["fwd"] += (time.perf_counter() - t0) * 1e3
            return out

        def _timed_stats(*a: Any, **kw: Any) -> Any:
            if not _active():
                return orig_stats(*a, **kw)
            _sync_device(device)
            t0 = time.perf_counter()
            out = orig_stats(*a, **kw)
            _sync_device(device)
            totals["stats_ar"] += (time.perf_counter() - t0) * 1e3
            return out

        def _timed_bwd(*a: Any, **kw: Any) -> Any:
            if not _active():
                return orig_bwd(*a, **kw)
            _sync_device(device)
            t0 = time.perf_counter()
            out = orig_bwd(*a, **kw)
            _sync_device(device)
            totals["bwd"] += (time.perf_counter() - t0) * 1e3
            return out

        def _timed_on_step_end(step: int, loss: float, lr: float) -> float:
            nonlocal step_idx
            if not _active():
                out = orig_on_step_end(step, loss, lr)
                step_idx += 1
                return out
            _sync_device(device)
            t0 = time.perf_counter()
            out = orig_on_step_end(step, loss, lr)
            _sync_device(device)
            totals["bcast"] += (time.perf_counter() - t0) * 1e3
            step_idx += 1
            return out

        strategy.prepare_fit = _wrapped_prepare
        strategy._precheck_domain_for_replay = _timed_precheck
        strategy._run_local_stage = _timed_local
        strategy._reduce_step_stats = _timed_stats
        strategy._backward_local_terms = _timed_bwd
        strategy.on_step_end = _timed_on_step_end
        return strategy

    trainer._apply_optimizer_step = _timed_apply_opt  # type: ignore[method-assign]
    trainer._make_fit_strategy = _wrapped_make_strategy  # type: ignore[method-assign]
    trainer.fit()

    means = {f"{k}_ms": _max_across_ranks(totals[k] / profile_steps, device) for k in keys}
    summed = sum(means.values())
    means["profiled_total_ms"] = summed
    means["grad_ar_pct"] = (100.0 * means["grad_ar_ms"] / summed) if summed > 0.0 else 0.0
    comm_plus_pack = (
        means["precheck_ms"]
        + means["stats_ar_ms"]
        + means["pack_ms"]
        + means["grad_ar_ms"]
        + means["unpack_ms"]
        + means["bcast_ms"]
    )
    means["comm_overhead_pct"] = (100.0 * comm_plus_pack / summed) if summed > 0.0 else 0.0

    del trainer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return means


# --------------------------------------------------------------------------- worker & orchestrator


def _worker_main(args: argparse.Namespace) -> int:
    job = json.loads(Path(args._worker_job).read_text(encoding="utf-8"))
    backend: Literal["gloo", "nccl"] = job["backend"]
    device_map: Literal["per-rank", "shared"] = job["device_map"]
    world_size: int = int(job["world_size"])
    warmup: int = int(job["warmup"])
    steps: int = int(job["steps"])
    profile_steps: int = int(job["profile_steps"])
    mixed_precision: Literal["fp32", "bf16"] = "bf16" if backend == "nccl" else "fp32"
    out_path = Path(job["out_json"])

    rank = 0
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        err = validate_distributed_device_map(
            backend=backend, device_map=device_map, world_size=world_size, local_rank=local_rank
        )
        if err is not None:
            raise RuntimeError(err)
        if backend == "nccl" and device_map == "shared":
            torch.cuda.set_device(0)
            device_arg = "cuda:0"
        elif backend == "nccl":
            device_arg = "auto"
        else:
            device_arg = "cpu"
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=300))
    else:
        device_arg = "cuda:0" if backend == "nccl" else "cpu"

    cache = _ModelCache()
    results: list[dict[str, Any]] = []
    try:
        for raw_spec in job["specs"]:
            spec = BenchSpec(**raw_spec)
            per_rank_batch = resolve_per_rank_batch(spec, world_size)
            if per_rank_batch is None:
                continue
            eff_warmup = max(warmup, 4 if spec.compile_model else warmup)
            # Enough windows for one clean full-size epoch across warmup + steps (no trailing shape change).
            n_dom_windows = world_size * per_rank_batch * (eff_warmup + max(steps, profile_steps) + 1)
            dom_ds = _make_cov_dataset(
                n_dom_windows,
                ctx_len=spec.ctx_len,
                horizon=spec.horizon,
                rows_per_series=spec.rows_per_series,
                seed=700,
            )
            rep_ds = (
                _make_cov_dataset(
                    world_size * spec.distill_batch_per_rank * 4,
                    ctx_len=min(spec.ctx_len, 256),
                    horizon=spec.horizon,
                    rows_per_series=1,
                    seed=701,
                )
                if spec.with_distill
                else None
            )

            pass_a = _run_pass_a_throughput(
                cache,
                spec,
                world_size=world_size,
                per_rank_batch=per_rank_batch,
                backend=backend,
                device_arg=device_arg,
                mixed_precision=mixed_precision,
                dom_ds=dom_ds,
                rep_ds=rep_ds,
                warmup_steps=eff_warmup,
                timed_steps=steps,
            )

            pass_b: dict[str, float] = {}
            if spec.check_val_vram:
                val_ds = _make_cov_dataset(
                    world_size * per_rank_batch * 2,
                    ctx_len=spec.ctx_len,
                    horizon=spec.horizon,
                    rows_per_series=spec.rows_per_series,
                    seed=702,
                )
                pass_b = _run_pass_b_val_vram(
                    cache,
                    spec,
                    per_rank_batch=per_rank_batch,
                    device_arg=device_arg,
                    mixed_precision=mixed_precision,
                    dom_ds=dom_ds,
                    val_ds=val_ds,
                    rep_ds=rep_ds,
                )
                pass_b["val_vram_delta_mib"] = pass_b["val_vram_mib"] - pass_a["train_vram_mib"]

            pass_c: dict[str, float] = {}
            if world_size > 1:
                pass_c = _run_pass_c_breakdown(
                    cache,
                    spec,
                    per_rank_batch=per_rank_batch,
                    device_arg=device_arg,
                    mixed_precision=mixed_precision,
                    dom_ds=dom_ds,
                    rep_ds=rep_ds,
                    warmup_steps=eff_warmup,
                    profile_steps=profile_steps,
                )

            if rank == 0:
                results.append(
                    {
                        "key": spec.key,
                        "world_size": world_size,
                        "per_rank_batch": per_rank_batch,
                        "spec": asdict(spec),
                        **pass_a,
                        **pass_b,
                        "breakdown": pass_c,
                    }
                )
            if world_size > 1:
                dev = (
                    torch.device("cuda", 0 if device_map == "shared" else rank)
                    if backend == "nccl"
                    else torch.device("cpu")
                )
                _barrier_rank(backend, dev)
        if rank == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(out_path, {"world_size": world_size, "rows": results})
    finally:
        if dist.is_initialized():
            if backend == "nccl" and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            dist.destroy_process_group()
    return 0


def _invoke_rank_worker(job: dict[str, Any], *, work_dir: Path) -> list[dict[str, Any]]:
    ws = int(job["world_size"])
    work_dir.mkdir(parents=True, exist_ok=True)
    job_path = work_dir / f"scaling_job_n{ws}.json"
    write_json_atomic(job_path, job)
    env = dict(os.environ)
    if job["backend"] == "nccl" and job["device_map"] == "per-rank":
        env.pop("NCCL_MULTI_RANK_GPU_ENABLE", None)
    script = str(Path(__file__).resolve())
    if ws == 1:
        cmd = [sys.executable, script, "--_worker-job", str(job_path)]
    else:
        port = pick_free_port()
        torchrun_bin = shutil.which("torchrun")
        if torchrun_bin is not None and Path(torchrun_bin).parent == Path(sys.executable).parent:
            cmd = [
                torchrun_bin,
                "--master-addr=127.0.0.1",
                f"--master-port={port}",
                f"--nproc_per_node={ws}",
                script,
                "--_worker-job",
                str(job_path),
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--master-addr=127.0.0.1",
                f"--master-port={port}",
                f"--nproc_per_node={ws}",
                script,
                "--_worker-job",
                str(job_path),
            ]
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"scaling worker (N={ws}) exited {proc.returncode}: {' '.join(cmd)}")
    data = json.loads(Path(job["out_json"]).read_text(encoding="utf-8"))
    return list(data["rows"])


def compute_scaling_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Join `N = 1, 2, 4` rows by `key` to compute speedup, efficiency, and D22 overhead summary."""
    by_key: dict[str, dict[int, Mapping[str, Any]]] = {}
    key_order: list[str] = []
    for row in rows:
        k = str(row["key"])
        if k not in by_key:
            by_key[k] = {}
            key_order.append(k)
        by_key[k][int(row["world_size"])] = row

    summary: list[dict[str, Any]] = []
    for k in key_order:
        ws_map = by_key[k]
        base = ws_map.get(1)
        for ws in sorted(ws_map):
            row = ws_map[ws]
            spec = row["spec"]
            scaling = spec["scaling"]
            speedup = float("nan")
            efficiency_pct = float("nan")
            if base is not None:
                if scaling == "weak":
                    # Weak scaling: throughput ratio (win/s_N / win/s_1); ideal = N.
                    speedup = row["windows_per_s"] / max(base["windows_per_s"], 1e-12)
                    efficiency_pct = 100.0 * speedup / ws
                else:
                    # Strong scaling: step-time speedup (step_ms_1 / step_ms_N); ideal = N.
                    speedup = base["mean_step_ms"] / max(row["mean_step_ms"], 1e-12)
                    efficiency_pct = 100.0 * speedup / ws
            bd = row.get("breakdown") or {}
            summary.append(
                {
                    "key": k,
                    "world_size": ws,
                    "scaling": scaling,
                    "size": spec["size"],
                    "mode": spec["mode"],
                    "compile_model": spec["compile_model"],
                    "with_distill": spec["with_distill"],
                    "ctx_len": spec["ctx_len"],
                    "per_rank_batch": row["per_rank_batch"],
                    "effective_batch": row["effective_batch"],
                    "mean_step_ms": row["mean_step_ms"],
                    "median_step_ms": row["median_step_ms"],
                    "windows_per_s": row["windows_per_s"],
                    "speedup_vs_1": speedup,
                    "efficiency_pct": efficiency_pct,
                    "train_vram_mib": row["train_vram_mib"],
                    "val_vram_mib": row.get("val_vram_mib"),
                    "val_vram_delta_mib": row.get("val_vram_delta_mib"),
                    "flat_buf_mib": row["flat_buf_mib"],
                    "grad_ar_ms": bd.get("grad_ar_ms"),
                    "grad_ar_pct": bd.get("grad_ar_pct"),
                    "pack_ms": bd.get("pack_ms"),
                    "unpack_ms": bd.get("unpack_ms"),
                    "small_coll_ms": (
                        bd.get("precheck_ms", 0.0) + bd.get("stats_ar_ms", 0.0) + bd.get("bcast_ms", 0.0)
                        if bd
                        else None
                    ),
                    "comm_overhead_pct": bd.get("comm_overhead_pct"),
                }
            )
    return summary


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--device-map", choices=("per-rank", "shared"), default="per-rank")
    parser.add_argument("--sizes", nargs="+", choices=ALL_SIZE_CHOICES, default=["28M", "123M", "209M"])
    parser.add_argument("--ranks", nargs="+", type=int, choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--no-compile", action="store_true", help="Skip compile_mode='default' rows.")
    parser.add_argument("--no-long-context", action="store_true", help="Skip ctx=1024 and ctx=4096 rows.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--_worker-job", type=str, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args._worker_job is not None:
        return _worker_main(args)

    require_positive_finite(parser, "--warmup", float(args.warmup))
    require_positive_finite(parser, "--steps", float(args.steps))
    require_positive_finite(parser, "--profile-steps", float(args.profile_steps))
    check_tmpdir_for_workers()

    dedup_ranks: list[int] = []
    for r in args.ranks:
        if r not in dedup_ranks:
            dedup_ranks.append(r)

    max_ws = max(dedup_ranks)
    if max_ws > 1:
        err = validate_distributed_device_map(
            backend=args.backend, device_map=args.device_map, world_size=max_ws, local_rank=None
        )
        if err is not None:
            print(f"error: {err}", file=sys.stderr)
            return 2
    elif args.backend == "nccl" and not torch.cuda.is_available():
        print("error: --backend nccl requires CUDA, but torch.cuda.is_available() is False", file=sys.stderr)
        return 2

    specs = build_bench_specs(
        args.sizes,
        backend=args.backend,
        include_compile=not args.no_compile,
        include_long_context=not args.no_long_context,
    )

    banner = [
        f"backend={args.backend}",
        f"device_map={args.device_map}",
        f"ranks={dedup_ranks}",
        f"sizes={list(args.sizes)}",
        f"torch={torch.__version__}",
        f"git={git_head_commit() or 'unknown'}",
    ]
    if args.backend == "nccl":
        banner.extend(cuda_env_banner_parts(allow_cuda_context=False))
    print("== distributed_scaling_bench | " + " | ".join(banner) + " ==")

    all_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="timestral_scale_") as tmp:
        work_dir = Path(tmp)
        for ws in dedup_ranks:
            out_json = work_dir / f"out_n{ws}.json"
            job = {
                "backend": args.backend,
                "device_map": args.device_map,
                "world_size": ws,
                "warmup": args.warmup,
                "steps": args.steps,
                "profile_steps": args.profile_steps,
                "out_json": str(out_json),
                "specs": [asdict(s) for s in specs],
            }
            print(f"\n--- Benchmarking world_size={ws} ({len(specs)} specs) ---")
            all_rows.extend(_invoke_rank_worker(job, work_dir=work_dir))

    summary = compute_scaling_summary(all_rows)
    print("\n=== Throughput, Scaling & VRAM Summary ===")
    print(
        f"{'scenario':<46} {'N':>1} {'B_eff':>5} | "
        f"{'mean_ms':>8} {'med_ms':>8} {'win/s':>8} {'spd':>5} {'eff%':>5} | "
        f"{'tr_MiB':>7} {'val+MiB':>7} {'bufMiB':>6} | "
        f"{'ar_ms':>6} {'ar%':>5} {'pk+up':>6} {'smCol':>6} {'comm%':>5}"
    )
    max_ar_pct = 0.0
    max_ar_key = "-"
    for s in summary:
        spd_s = f"{s['speedup_vs_1']:5.2f}" if not math.isnan(s["speedup_vs_1"]) else "    -"
        eff_s = f"{s['efficiency_pct']:5.1f}" if not math.isnan(s["efficiency_pct"]) else "    -"
        val_d = f"{s['val_vram_delta_mib']:+7.1f}" if s["val_vram_delta_mib"] is not None else "      -"
        ar_ms = f"{s['grad_ar_ms']:6.2f}" if s["grad_ar_ms"] is not None else "     -"
        ar_pct = f"{s['grad_ar_pct']:5.1f}" if s["grad_ar_pct"] is not None else "    -"
        pk_up = (
            f"{(s['pack_ms'] + s['unpack_ms']):6.2f}"
            if (s["pack_ms"] is not None and s["unpack_ms"] is not None)
            else "     -"
        )
        sm_c = f"{s['small_coll_ms']:6.2f}" if s["small_coll_ms"] is not None else "     -"
        cm_pct = f"{s['comm_overhead_pct']:5.1f}" if s["comm_overhead_pct"] is not None else "    -"
        if s["grad_ar_pct"] is not None and s["grad_ar_pct"] > max_ar_pct:
            max_ar_pct = s["grad_ar_pct"]
            max_ar_key = f"{s['key']} (N={s['world_size']})"
        print(
            f"{s['key']:<46} {s['world_size']:>1} {s['effective_batch']:>5} | "
            f"{s['mean_step_ms']:8.2f} {s['median_step_ms']:8.2f} {s['windows_per_s']:8.1f} {spd_s} {eff_s} | "
            f"{s['train_vram_mib']:7.1f} {val_d} {s['flat_buf_mib']:6.1f} | "
            f"{ar_ms} {ar_pct} {pk_up} {sm_c} {cm_pct}"
        )

    print(
        f"\nD22 check: peak flat-buffer all_reduce share = {max_ar_pct:.1f}% ({max_ar_key}); "
        f"D22 revisit threshold is > 15-20% on 123M/209M full_ft."
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "git_head": git_head_commit(),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "os": platform.system(),
                "arch": platform.machine(),
                "backend": args.backend,
                "device_map": args.device_map,
                "sizes": list(args.sizes),
                "ranks": dedup_ranks,
            },
            "raw_rows": all_rows,
            "summary": summary,
        }
        write_json_atomic(args.json_out, payload)
        print(f"Structured results written to: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
