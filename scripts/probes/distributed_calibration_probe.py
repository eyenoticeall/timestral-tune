#!/usr/bin/env python3
"""Stage 3 probe: `bf16` cause, compiled CUDA variation, and multi-seed cosine parity calibration.

Addresses the third open Phase 5 Stage 3 item (`docs/SPEC.md` §9, §10, §12;
`docs/phases/phase-5-multi-gpu.md` §3) without touching `src/`:

1. `bf16-cause` — Three-way decomposition at step 1 and over 6 steps:
     - `union_1p`: 1-process `TimestralTrainer` on union batch `B = 2 * N` (`cuda:0` / `cpu`)
     - `micro_accum_1p`: 1-process on the same device accumulating the exact `N` per-rank shards
       (`B = 2`) with exact count-weighted loss-share scaling in `fp32` (zero NCCL / zero `dist`)
     - `dist`: real `N`-rank `TimestralTrainer` under `torchrun` (`device="auto"` -> `cuda:LOCAL_RANK`)
   Comparing `union_vs_micro` (pure batch-size kernel quantization), `micro_vs_dist` (pure
   distributed reduction), and `union_vs_dist` (total gap) across `fp32` and `bf16` isolates
   whether larger `bf16` differences come from batch-size `bf16` quantization + AdamW sign flips
   or from multi-rank reduction.

2. `compile-var` — Repeated fresh-subprocess runs (`--compile-repeats`, default 3) of
   `eager_lora_variate` (control), `compile_default_lora_variate`, and
   `compile_default_lora_replay` (plus `TORCHINDUCTOR_FX_GRAPH_CACHE=0` on CUDA), reporting
   pairwise `ref_i vs ref_j`, `dist_i vs dist_j`, `dist_k vs ref_1`, and `dist_k vs ref_k` in
   exact `float64` to separate one-process Inductor/CUDA non-determinism from multi-rank drift.

3. `calibrate` — Multi-seed, multi-size (`tiny`, `5M`, `28M`, `123M`, `209M`) exact-`float64`
   parity calibration across `N in {2, 4}`, modes (`head_only`, `lora_variate`, `lora_temporal`,
   `full_ft`), regimes (`domain`, `replay_distill`), and precisions (`fp32`, `bf16`, optional
   `fp64`), reporting `xr_eq`, `xr_max`, `loss_rel`, `val_rel`, `step1_grad_l2`,
   `step1_1m_grad_cos`, `param_l2`, `upd_l2`, and `1m_upd_cos` (computed via `0.5 * ||u - v||^2`
   on unit-normalised `float64` vectors to avoid cancellation near `cos = 1`).

Examples:
    # Local CPU/gloo smoke test on tiny_random:
    TMPDIR=/tmp uv run python scripts/probes/distributed_calibration_probe.py \\
        --backend gloo --sizes tiny --ranks 2 --seeds 11 --compile-repeats 2 \\
        --json-out /tmp/calib_cpu.json

    # Full 4-GPU CUDA/NCCL calibration & investigation:
    TMPDIR=/tmp uv run python scripts/probes/distributed_calibration_probe.py \\
        --backend nccl --sizes tiny 5M 28M 123M 209M --ranks 2 4 --seeds 11 22 33 \\
        --json-out stage3_calibration.json
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from _common import SIZES, float64_reference, load, tiny_random_forecaster
from _dist_common import (
    DOM_HORIZON,
    REP_HORIZON,
    check_tmpdir_for_workers,
    cuda_env_banner_parts,
    git_head_commit,
    make_synthetic_dataset,
    pick_free_port,
    require_positive_finite,
    validate_distributed_device_map,
    write_json_atomic,
)
from torch import Tensor

from timestral_tune import LoraConfig, ReplayConfig, TimestralTrainer, TrainingConfig
from timestral_tune.config import AdaptationMode
from timestral_tune.data import TimeSeriesDataset
from timestral_tune.models import TuneTimestralModel

ALL_SIZE_CHOICES: tuple[str, ...] = ("tiny", *SIZES)
SECTIONS: tuple[str, ...] = ("bf16-cause", "compile-var", "calibrate")
MODES: tuple[AdaptationMode, ...] = ("head_only", "lora_variate", "lora_temporal", "full_ft")

DEFAULT_PER_RANK_BATCH = 2
DEFAULT_STEPS = 6
MOVED_WARMUP_STEPS = 3


# --------------------------------------------------------------------------- exact float64 metrics


def rel_l2_f64(a: Tensor, b: Tensor) -> float:
    """Exact relative L2 `||a - b||_2 / max(||b||_2, 1e-300)` in float64 (NaN-propagating)."""
    a64 = a.detach().cpu().double().reshape(-1)
    b64 = b.detach().cpu().double().reshape(-1)
    if a64.shape != b64.shape:
        raise ValueError(f"shape mismatch: {tuple(a64.shape)} vs {tuple(b64.shape)}")
    if not (bool(torch.isfinite(a64).all()) and bool(torch.isfinite(b64).all())):
        return float("nan")
    denom = float(b64.norm().item())
    diff = float((a64 - b64).norm().item())
    if denom == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / denom


def cosine_distance_f64(a: Tensor, b: Tensor) -> float:
    """Exact `1 - cos(a, b)` in float64 via `0.5 * ||u - v||^2` (avoids cancellation near 1)."""
    a64 = a.detach().cpu().double().reshape(-1)
    b64 = b.detach().cpu().double().reshape(-1)
    if a64.shape != b64.shape:
        raise ValueError(f"shape mismatch: {tuple(a64.shape)} vs {tuple(b64.shape)}")
    if not (bool(torch.isfinite(a64).all()) and bool(torch.isfinite(b64).all())):
        return float("nan")
    na = float(a64.norm().item())
    nb = float(b64.norm().item())
    if na == 0.0 and nb == 0.0:
        return 0.0
    if na == 0.0 or nb == 0.0:
        return float("nan")
    u = a64 / na
    v = b64 / nb
    return 0.5 * float((u - v).pow(2).sum().item())


def sign_flip_fraction(a: Tensor, b: Tensor, *, zero_tol: float = 0.0) -> float:
    """Fraction of coordinates where both tensors exceed `zero_tol` and have opposite signs."""
    a64 = a.detach().cpu().double().reshape(-1)
    b64 = b.detach().cpu().double().reshape(-1)
    if a64.shape != b64.shape or a64.numel() == 0:
        return float("nan")
    if not (bool(torch.isfinite(a64).all()) and bool(torch.isfinite(b64).all())):
        return float("nan")
    active = (a64.abs() > zero_tol) & (b64.abs() > zero_tol)
    n_active = int(active.sum().item())
    if n_active == 0:
        return 0.0
    flipped = int(((a64[active] * b64[active]) < 0.0).sum().item())
    return flipped / n_active


def scalar_rel(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    denom = max(abs(b), 1e-300)
    return abs(a - b) / denom


def max_list_rel(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return float("nan")
    out = 0.0
    for x, y in zip(xs, ys, strict=True):
        r = scalar_rel(x, y)
        if math.isnan(r):
            return float("nan")
        if r > out:
            out = r
    return out


def _flatten_tensors_cpu(tensors: Sequence[Tensor]) -> Tensor:
    parts = [t.detach().cpu().reshape(-1) for t in tensors]
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)


def _flatten_grads_cpu(params: Sequence[Tensor]) -> Tensor:
    parts: list[Tensor] = []
    for p in params:
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), dtype=p.dtype, device="cpu"))
        else:
            parts.append(p.grad.detach().cpu().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)


def _tensor_sha256(t: Tensor) -> str:
    arr = t.detach().cpu().contiguous()
    if arr.dtype == torch.bfloat16:
        raw = arr.view(torch.uint16).numpy().tobytes()
    else:
        raw = arr.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _model_state_sha256(model: TuneTimestralModel) -> str:
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(_tensor_sha256(param).encode("ascii"))
    for name, buf in sorted(model.named_buffers()):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(_tensor_sha256(buf).encode("ascii"))
    return h.hexdigest()


def _cross_rank_state_equality(model: TuneTimestralModel, device: torch.device) -> tuple[bool, float]:
    """Return `(bitwise_equal_all_ranks, max_abs_diff_f64)` across all parameters and buffers."""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return True, 0.0
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    my_hash = _model_state_sha256(model)
    # Fixed-size 32-byte SHA-256 compare via int64 tensor (no object collectives).
    digest_bytes = bytes.fromhex(my_hash)
    ref_digest = torch.tensor(list(digest_bytes), dtype=torch.int64, device=device)
    my_digest = ref_digest.clone()
    dist.broadcast(ref_digest, src=0)
    hash_mismatch = 0.0 if bool(torch.equal(my_digest, ref_digest)) else 1.0
    mismatch_t = torch.tensor([hash_mismatch], dtype=torch.float64, device=device)
    dist.all_reduce(mismatch_t, op=dist.ReduceOp.SUM)
    all_eq = float(mismatch_t.item()) == 0.0

    max_diff = 0.0
    if not all_eq:
        tensors = [t for _, t in sorted(model.named_parameters())] + [b for _, b in sorted(model.named_buffers())]
        for t in tensors:
            mine = t.detach().to(device=device).contiguous()
            r0 = mine.clone()
            dist.broadcast(r0, src=0)
            if rank != 0:
                diff = (mine.double() - r0.double()).abs()
                if not bool(torch.isfinite(diff).all()):
                    max_diff = float("nan")
                elif not math.isnan(max_diff):
                    cur = float(diff.max().item()) if diff.numel() > 0 else 0.0
                    if cur > max_diff:
                        max_diff = cur
        md_t = torch.tensor([0.0 if math.isnan(max_diff) else max_diff], dtype=torch.float64, device=device)
        nan_t = torch.tensor([1.0 if math.isnan(max_diff) else 0.0], dtype=torch.float64, device=device)
        dist.all_reduce(md_t, op=dist.ReduceOp.MAX)
        dist.all_reduce(nan_t, op=dist.ReduceOp.SUM)
        max_diff = float("nan") if float(nan_t.item()) > 0.0 else float(md_t.item())
    del world_size
    return all_eq, max_diff


# --------------------------------------------------------------------------- model & config builders


class _ModelCache:
    """Loads each requested base checkpoint onto CPU once per worker process."""

    def __init__(self) -> None:
        self._cpu_models: dict[tuple[str, int], TuneTimestralModel] = {}

    def clone(self, size: str, *, seed: int, dtype: torch.dtype) -> TuneTimestralModel:
        key = (size, seed if size == "tiny" else 0)
        if key not in self._cpu_models:
            if size == "tiny":
                self._cpu_models[key] = tiny_random_forecaster(seed=seed).model.cpu().eval()
            else:
                self._cpu_models[key] = load(size, device="cpu").model.cpu().eval()
        model = copy.deepcopy(self._cpu_models[key])
        if dtype == torch.float64:
            model = model.double()
        return model


def _lora_for(mode: AdaptationMode, *, dropout: float = 0.0) -> LoraConfig | None:
    if mode == "lora_variate":
        return LoraConfig(r=8, lora_alpha=16.0, axes=("variate",), targets=("q", "k", "v", "o"), dropout=dropout)
    if mode == "lora_temporal":
        return LoraConfig(r=8, lora_alpha=16.0, axes=("time",), targets=("q", "k", "v", "o"), dropout=dropout)
    return None


def _default_lr(mode: AdaptationMode) -> float:
    return {
        "head_only": 3e-3,
        "lora_variate": 3e-3,
        "lora_temporal": 1e-3,
        "full_ft": 1e-4,
    }[mode]


def _make_datasets(
    *, world_size: int, per_rank_batch: int, steps: int, seed: int, empty_trailing_shard: bool = False
) -> tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    if empty_trailing_shard:
        n_dom = max(1, (steps - 1) * world_size * per_rank_batch + max(1, world_size - 1))
    else:
        n_dom = steps * world_size * per_rank_batch
    dom_ds = make_synthetic_dataset(n_dom, horizon=DOM_HORIZON, covariates=True, seed=seed * 100 + 1)
    val_ds = make_synthetic_dataset(max(4, world_size * 2), horizon=DOM_HORIZON, covariates=True, seed=seed * 100 + 2)
    rep_ds = make_synthetic_dataset(
        world_size * per_rank_batch + world_size + 1,
        horizon=REP_HORIZON,
        covariates=False,
        seed=seed * 100 + 3,
    )
    return dom_ds, val_ds, rep_ds


def _build_warmed_full_ft_state(
    cache: _ModelCache,
    size: str,
    *,
    seed: int,
    dtype: torch.dtype,
    dom_ds: TimeSeriesDataset,
    world_size: int,
    per_rank_batch: int,
) -> dict[str, Tensor]:
    """Warm a `full_ft` student for 3 deterministic CPU steps so student != step-0 teacher."""
    model = cache.clone(size, seed=seed, dtype=dtype)
    cfg = TrainingConfig(
        model_name_or_path=f"probe-{size}",
        mode="full_ft",
        device="cpu",
        batch_size=world_size * per_rank_batch,
        max_steps=MOVED_WARMUP_STEPS,
        warmup_steps=0,
        lr_schedule="constant",
        learning_rate=_default_lr("full_ft"),
        weight_decay=0.01,
        gradient_clip_val=1.0,
        shuffle=False,
        sign_flip=False,
        seed=seed + 500,
        log_every=1000,
        restore_best=False,
    )
    ctx_mgr = float64_reference() if dtype == torch.float64 else contextlib.nullcontext()
    with ctx_mgr:
        # Construct trainer without active dist group? Inside a torchrun worker dist IS initialized,
        # so run pure manual AdamW steps using compute_loss directly (never touches dist!).
        from timestral_tune.data.collator import DHFDataCollator
        from timestral_tune.trainer import compute_loss

        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(list(model.parameters()), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        collator = DHFDataCollator.for_model(model.config, dom_ds.context_length)
        n_items = len(dom_ds)
        bs = world_size * per_rank_batch
        for step in range(MOVED_WARMUP_STEPS):
            start = (step * bs) % n_items
            idxs = [(start + i) % n_items for i in range(bs)]
            batch = collator([dom_ds[i] for i in idxs])
            opt.zero_grad(set_to_none=True)
            compute_loss(model, batch, cfg.target_frame, validate=False).loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _capture_step1_grad_during_fit(trainer: TimestralTrainer) -> list[Tensor]:
    """Hook `trainer._apply_optimizer_step` to snapshot the reduced step-1 gradient right before clip/step."""
    captured: list[Tensor] = []
    orig_apply = trainer._apply_optimizer_step

    def _wrapped_apply() -> None:
        if not captured:
            captured.append(_flatten_grads_cpu(trainer._trainable))
        orig_apply()

    trainer._apply_optimizer_step = _wrapped_apply  # type: ignore[method-assign]
    return captured


def _run_trainer_scenario(
    cache: _ModelCache,
    *,
    size: str,
    world_size: int,
    is_distributed: bool,
    device_arg: str,
    precision: Literal["fp64", "fp32", "bf16"],
    mode: AdaptationMode,
    regime: Literal["domain", "replay_distill", "replay_distill_moved"],
    seed: int,
    steps: int = DEFAULT_STEPS,
    per_rank_batch: int = DEFAULT_PER_RANK_BATCH,
    compile_model: bool = False,
    empty_trailing_shard: bool = False,
) -> dict[str, Any]:
    dtype = torch.float64 if precision == "fp64" else torch.float32
    mixed_precision: Literal["fp32", "bf16"] = "bf16" if precision == "bf16" else "fp32"
    dom_ds, val_ds, rep_ds = _make_datasets(
        world_size=world_size,
        per_rank_batch=per_rank_batch,
        steps=steps,
        seed=seed,
        empty_trailing_shard=empty_trailing_shard,
    )
    eff_bs = per_rank_batch if is_distributed else world_size * per_rank_batch
    rcfg: ReplayConfig | None = None
    rep_arg: TimeSeriesDataset | None = None
    if regime in ("replay_distill", "replay_distill_moved"):
        rcfg = ReplayConfig(
            replay_fraction=0.25,
            distill_weight=0.5,
            batch_size=eff_bs,
            sign_flip=False,
            cache_base_outputs=False,  # uncached live teacher on both 1p and Np for exact apples-to-apples
            smooth_beta=1e-3,
        )
        rep_arg = rep_ds

    cfg = TrainingConfig(
        model_name_or_path=f"probe-{size}",
        mode=mode,
        device=device_arg,
        mixed_precision=mixed_precision,
        compile_model=compile_model,
        compile_mode="default",
        batch_size=eff_bs,
        eval_batch_size=per_rank_batch if is_distributed else world_size * per_rank_batch,
        max_steps=steps,
        warmup_steps=min(2, steps),
        lr_schedule="cosine",
        learning_rate=_default_lr(mode),
        weight_decay=0.01,
        gradient_clip_val=1.0,
        shuffle=False,
        sign_flip=False,
        cache_frozen_features=False,  # streaming on both 1p and Np
        restore_best=False,
        seed=seed,
        log_every=1000,
    )

    if compile_model and hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

    base_model = cache.clone(size, seed=seed, dtype=dtype)
    warmed_state: dict[str, Tensor] | None = None
    if regime == "replay_distill_moved" and mode == "full_ft":
        warmed_state = _build_warmed_full_ft_state(
            cache,
            size,
            seed=seed,
            dtype=dtype,
            dom_ds=dom_ds,
            world_size=world_size,
            per_rank_batch=per_rank_batch,
        )

    ctx_mgr = float64_reference() if dtype == torch.float64 else contextlib.nullcontext()
    with ctx_mgr:
        trainer = TimestralTrainer(
            cfg,
            dom_ds,
            val_ds,
            lora_config=_lora_for(mode, dropout=0.0),
            replay_config=rcfg,
            replay_dataset=rep_arg,
            model=base_model,
        )
        if warmed_state is not None:
            trainer.model.load_state_dict(warmed_state)

        init_trainable = _flatten_tensors_cpu(trainer._trainable)
        init_sha = _tensor_sha256(init_trainable)
        step1_grads = _capture_step1_grad_during_fit(trainer)

        t0 = time.perf_counter()
        res = trainer.fit()
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
        wall_s = time.perf_counter() - t0

        if res.eval_history:
            val_loss = float(res.eval_history[-1][1])
        elif not is_distributed or dist.get_rank() == 0:
            val_loss = float(trainer.evaluate().loss)
        else:
            val_loss = float("nan")
        xr_eq, xr_max = _cross_rank_state_equality(trainer.model, trainer.device) if is_distributed else (True, 0.0)
        final_trainable = _flatten_tensors_cpu(trainer._trainable)

        unique_graphs: int | None = None
        if compile_model and hasattr(torch, "_dynamo"):
            try:
                unique_graphs = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            except Exception:
                unique_graphs = None

    step1_grad = step1_grads[0] if step1_grads else torch.empty(0, dtype=dtype)
    out = {
        "init_sha": init_sha,
        "init_trainable": init_trainable,
        "final_trainable": final_trainable,
        "step1_grad": step1_grad,
        "train_losses": [float(loss) for _, loss in res.train_history],
        "val_loss": val_loss,
        "steps": int(res.steps),
        "xr_eq": bool(xr_eq),
        "xr_max": float(xr_max),
        "wall_s": float(wall_s),
        "unique_graphs": unique_graphs,
    }
    del trainer, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _run_micro_accum_1p(
    cache: _ModelCache,
    *,
    size: str,
    world_size: int,
    device_arg: str,
    precision: Literal["fp64", "fp32", "bf16"],
    mode: AdaptationMode,
    regime: Literal["domain", "replay_distill"],
    seed: int,
    steps: int = DEFAULT_STEPS,
    per_rank_batch: int = DEFAULT_PER_RANK_BATCH,
) -> dict[str, Any]:
    """One-process micro-batch accumulation control (ZERO NCCL / ZERO `torch.distributed`).

    Executes the exact same `N` per-rank domain/replay shards (`B = per_rank_batch`) as `N`-rank
    distributed training, computes the exact same count-weighted loss-share backward on each shard,
    and sums gradients into `fp32` `.grad` tensors on one GPU before clipping and stepping AdamW.
    """
    dtype = torch.float64 if precision == "fp64" else torch.float32
    mixed_precision: Literal["fp32", "bf16"] = "bf16" if precision == "bf16" else "fp32"
    dom_ds, val_ds, rep_ds = _make_datasets(
        world_size=world_size, per_rank_batch=per_rank_batch, steps=steps, seed=seed
    )
    rcfg: ReplayConfig | None = None
    rep_arg: TimeSeriesDataset | None = None
    if regime == "replay_distill":
        rcfg = ReplayConfig(
            replay_fraction=0.25,
            distill_weight=0.5,
            batch_size=world_size * per_rank_batch,
            sign_flip=False,
            cache_base_outputs=False,
            smooth_beta=1e-3,
        )
        rep_arg = rep_ds

    cfg = TrainingConfig(
        model_name_or_path=f"probe-{size}",
        mode=mode,
        device=device_arg,
        mixed_precision=mixed_precision,
        batch_size=world_size * per_rank_batch,
        eval_batch_size=world_size * per_rank_batch,
        max_steps=steps,
        warmup_steps=min(2, steps),
        lr_schedule="cosine",
        learning_rate=_default_lr(mode),
        weight_decay=0.01,
        gradient_clip_val=1.0,
        shuffle=False,
        sign_flip=False,
        cache_frozen_features=False,
        restore_best=False,
        seed=seed,
        log_every=1000,
    )

    from timestral_tune.data.collator import DHFDataCollator
    from timestral_tune.trainer.distributed import rank_step_indices, steps_per_epoch

    base_model = cache.clone(size, seed=seed, dtype=dtype)
    ctx_mgr = float64_reference() if dtype == torch.float64 else contextlib.nullcontext()
    with ctx_mgr:
        trainer = TimestralTrainer(
            cfg,
            dom_ds,
            val_ds,
            lora_config=_lora_for(mode, dropout=0.0),
            replay_config=rcfg,
            replay_dataset=rep_arg,
            model=base_model,
        )
        trainer.scheduler = trainer._new_schedule()
        init_trainable = _flatten_tensors_cpu(trainer._trainable)
        init_sha = _tensor_sha256(init_trainable)
        dom_collator = DHFDataCollator.for_model(trainer.model.config, dom_ds.context_length)
        rep_collator = (
            DHFDataCollator.for_model(trainer.model.config, rep_ds.context_length) if rep_arg is not None else None
        )

        n_dom = len(dom_ds)
        n_rep = len(rep_ds) if rep_arg is not None else 0
        dom_steps_per_ep = steps_per_epoch(n_dom, world_size, per_rank_batch)
        rep_steps_per_ep = steps_per_epoch(n_rep, world_size, per_rank_batch) if rep_arg is not None else 0

        train_losses: list[float] = []
        step1_grad: Tensor | None = None
        trainer.model.train()

        for gstep in range(steps):
            trainer.optimizer.zero_grad(set_to_none=True)
            dom_step_in_ep = gstep % dom_steps_per_ep
            rep_step_in_ep = (gstep % rep_steps_per_ep) if rep_steps_per_ep > 0 else 0

            # Stage 1: prepare every rank's shard locally to gather global counts (matching _DistributedStrategy).
            shard_payloads: list[tuple[Any, Any, int, int, int]] = []
            w_d_tot = 0.0
            w_r_tot = 0.0
            m_tot = 0.0
            for r in range(world_size):
                d_idx = rank_step_indices(n_dom, world_size, per_rank_batch, r, dom_step_in_ep)
                raw_d = dom_collator([dom_ds[i] for i in d_idx]) if d_idx else None
                prep_d = None if raw_d is None else trainer._prepare_domain_batch(raw_d, None)
                prep_r = None
                if rep_arg is not None and rep_collator is not None:
                    r_idx = rank_step_indices(n_rep, world_size, per_rank_batch, r, rep_step_in_ep)
                    raw_r = rep_collator([rep_arg[i] for i in r_idx]) if r_idx else None
                    prep_r = None if raw_r is None else trainer._prepare_replay_batch(raw_r, None)
                wd = 0 if prep_d is None else prep_d.num_scored
                wr = 0 if prep_r is None else prep_r.num_scored
                # Count distillation target-step slots identically to trainer._compute_replay_terms_with_counts.
                if prep_r is not None and rcfg is not None and rcfg.distill_weight > 0.0:
                    real_targets = int(prep_r.batch.target_rows.sum().item())
                    h_eff = min(rep_arg.forecast_horizon, int(prep_r.batch.targets.shape[1]))
                    mc = real_targets * h_eff
                else:
                    mc = 0
                w_d_tot += float(wd)
                w_r_tot += float(wr)
                m_tot += float(mc)
                shard_payloads.append((prep_d, prep_r, wd, wr, mc))

            rf = 0.0 if rcfg is None else rcfg.replay_fraction
            dw = 0.0 if rcfg is None else rcfg.distill_weight
            replay_active = rf > 0.0 and w_r_tot > 0.0
            distill_active = dw > 0.0 and m_tot > 0.0
            dom_weight = (1.0 - rf) if replay_active else 1.0

            # Stage 2: forward + count-weighted backward per shard, accumulating into p.grad.
            sum_d_tot = 0.0
            sum_r_tot = 0.0
            sum_m_tot = 0.0
            for prep_d, prep_r, wd, wr, mc in shard_payloads:
                if prep_d is None and prep_r is None:
                    continue
                terms = trainer._compute_loss_terms(
                    None if prep_d is None else prep_d.batch,
                    domain_count=wd,
                    validate=False,
                    replay=prep_r,
                )
                assert terms.replay_count == wr, f"replay count mismatch: {terms.replay_count} != {wr}"
                assert terms.distill_count == mc, f"distill count mismatch: {terms.distill_count} != {mc}"
                local_terms: list[Tensor] = []
                if terms.domain is not None and wd > 0:
                    sum_d_tot += float(terms.domain.detach().item()) * float(wd)
                    local_terms.append((dom_weight * (float(wd) / w_d_tot)) * terms.domain)
                if replay_active and terms.replay is not None and wr > 0:
                    sum_r_tot += float(terms.replay.detach().item()) * float(wr)
                    local_terms.append((rf * (float(wr) / w_r_tot)) * terms.replay)
                if distill_active and terms.distill is not None and mc > 0:
                    sum_m_tot += float(terms.distill.detach().item()) * float(mc)
                    local_terms.append((dw * (float(mc) / m_tot)) * terms.distill)
                if local_terms:
                    loss_r = local_terms[0]
                    for extra in local_terms[1:]:
                        loss_r = loss_r + extra
                    loss_r.backward()

            global_loss = dom_weight * (sum_d_tot / w_d_tot)
            if replay_active:
                global_loss += rf * (sum_r_tot / w_r_tot)
            if distill_active:
                global_loss += dw * (sum_m_tot / m_tot)
            train_losses.append(global_loss)

            if step1_grad is None:
                step1_grad = _flatten_grads_cpu(trainer._trainable)
            trainer._apply_optimizer_step()
            trainer.global_step += 1

        trainer.model.eval()
        val_res = trainer.evaluate()
        final_trainable = _flatten_tensors_cpu(trainer._trainable)

    out = {
        "init_sha": init_sha,
        "init_trainable": init_trainable,
        "final_trainable": final_trainable,
        "step1_grad": step1_grad if step1_grad is not None else torch.empty(0, dtype=dtype),
        "train_losses": train_losses,
        "val_loss": float(val_res.loss),
        "steps": steps,
    }
    del trainer, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _compare_pair(cand: Mapping[str, Any], ref: Mapping[str, Any]) -> dict[str, float]:
    """Compare two run dicts (`cand` vs `ref`) with exact float64 L2, cosine distance, and sign flips."""
    if cand["init_sha"] != ref["init_sha"]:
        raise RuntimeError(f"initial trainable weights diverged: {cand['init_sha'][:12]} != {ref['init_sha'][:12]}")
    init = ref["init_trainable"]
    cand_upd = cand["final_trainable"].double() - init.double()
    ref_upd = ref["final_trainable"].double() - init.double()
    return {
        "step1_grad_l2": rel_l2_f64(cand["step1_grad"], ref["step1_grad"]),
        "step1_1m_grad_cos": cosine_distance_f64(cand["step1_grad"], ref["step1_grad"]),
        "step1_sign_flip": sign_flip_fraction(cand["step1_grad"], ref["step1_grad"]),
        "param_l2": rel_l2_f64(cand["final_trainable"], ref["final_trainable"]),
        "upd_l2": rel_l2_f64(cand_upd, ref_upd),
        "1m_upd_cos": cosine_distance_f64(cand_upd, ref_upd),
        "loss_rel": max_list_rel(cand["train_losses"], ref["train_losses"]),
        "val_rel": scalar_rel(float(cand["val_loss"]), float(ref["val_loss"])),
    }


def _save_run_tensors(path: Path, run: Mapping[str, Any]) -> None:
    payload = {
        "init_sha": run["init_sha"],
        "init_trainable": run["init_trainable"],
        "final_trainable": run["final_trainable"],
        "step1_grad": run["step1_grad"],
        "train_losses": run["train_losses"],
        "val_loss": run["val_loss"],
        "steps": run["steps"],
        "xr_eq": run.get("xr_eq", True),
        "xr_max": run.get("xr_max", 0.0),
        "wall_s": run.get("wall_s", 0.0),
        "unique_graphs": run.get("unique_graphs"),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_run_tensors(path: Path) -> dict[str, Any]:
    return dict(torch.load(path, map_location="cpu", weights_only=True))


# --------------------------------------------------------------------------- worker entry points


def _init_worker_pg(
    backend: Literal["gloo", "nccl"], device_map: Literal["per-rank", "shared"]
) -> tuple[int, int, str]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
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
    return rank, world_size, device_arg


def _worker_main(args: argparse.Namespace) -> int:
    job = json.loads(Path(args._worker_job).read_text(encoding="utf-8"))
    kind: str = job["kind"]
    backend: Literal["gloo", "nccl"] = job["backend"]
    device_map: Literal["per-rank", "shared"] = job["device_map"]
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = _ModelCache()

    if kind == "single_ref":
        device_arg = "cuda:0" if backend == "nccl" else "cpu"
        for item in job["scenarios"]:
            run = _run_trainer_scenario(
                cache,
                size=item["size"],
                world_size=item["world_size"],
                is_distributed=False,
                device_arg=device_arg,
                precision=item["precision"],
                mode=item["mode"],
                regime=item["regime"],
                seed=item["seed"],
                steps=item["steps"],
                per_rank_batch=item["per_rank_batch"],
                compile_model=item.get("compile_model", False),
                empty_trailing_shard=item.get("empty_trailing_shard", False),
            )
            _save_run_tensors(out_dir / f"{item['id']}.pt", run)
        return 0

    if kind == "micro_accum":
        device_arg = "cuda:0" if backend == "nccl" else "cpu"
        for item in job["scenarios"]:
            run = _run_micro_accum_1p(
                cache,
                size=item["size"],
                world_size=item["world_size"],
                device_arg=device_arg,
                precision=item["precision"],
                mode=item["mode"],
                regime=item["regime"],
                seed=item["seed"],
                steps=item["steps"],
                per_rank_batch=item["per_rank_batch"],
            )
            _save_run_tensors(out_dir / f"{item['id']}.pt", run)
        return 0

    if kind == "dist_run":
        rank, world_size, device_arg = _init_worker_pg(backend, device_map)
        try:
            for item in job["scenarios"]:
                assert item["world_size"] == world_size
                run = _run_trainer_scenario(
                    cache,
                    size=item["size"],
                    world_size=world_size,
                    is_distributed=True,
                    device_arg=device_arg,
                    precision=item["precision"],
                    mode=item["mode"],
                    regime=item["regime"],
                    seed=item["seed"],
                    steps=item["steps"],
                    per_rank_batch=item["per_rank_batch"],
                    compile_model=item.get("compile_model", False),
                    empty_trailing_shard=item.get("empty_trailing_shard", False),
                )
                if rank == 0:
                    _save_run_tensors(out_dir / f"{item['id']}.pt", run)
                if backend == "nccl":
                    idx = 0 if device_map == "shared" else rank
                    dist.barrier(device_ids=[idx])
                else:
                    dist.barrier()
        finally:
            if dist.is_initialized():
                if backend == "nccl" and torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                dist.destroy_process_group()
        return 0

    raise ValueError(f"unknown worker job kind: {kind!r}")


# --------------------------------------------------------------------------- parent orchestration


def _invoke_worker(
    job: dict[str, Any],
    *,
    work_dir: Path,
    nproc: int | None,
    extra_env: Mapping[str, str] | None = None,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    job_path = work_dir / f"job_{job['kind']}_{time.time_ns()}.json"
    write_json_atomic(job_path, job)
    env = dict(os.environ)
    if job["backend"] == "nccl" and job["device_map"] == "per-rank":
        env.pop("NCCL_MULTI_RANK_GPU_ENABLE", None)
    if extra_env:
        env.update(extra_env)
    script = str(Path(__file__).resolve())
    if nproc is None:
        cmd = [sys.executable, script, "--_worker-job", str(job_path)]
    else:
        port = pick_free_port()
        torchrun_bin = shutil.which("torchrun")
        if torchrun_bin is not None and Path(torchrun_bin).parent == Path(sys.executable).parent:
            cmd = [
                torchrun_bin,
                "--master-addr=127.0.0.1",
                f"--master-port={port}",
                f"--nproc_per_node={nproc}",
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
                f"--nproc_per_node={nproc}",
                script,
                "--_worker-job",
                str(job_path),
            ]
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"worker command exited {proc.returncode}: {' '.join(cmd)}")


def _fmt(val: float) -> str:
    if math.isnan(val):
        return "nan"
    if math.isinf(val):
        return "inf"
    return f"{val:.2e}"


def _run_section_bf16_cause(
    *,
    backend: Literal["gloo", "nccl"],
    device_map: Literal["per-rank", "shared"],
    sizes: Sequence[str],
    world_size: int,
    steps: int,
    work_root: Path,
) -> list[dict[str, Any]]:
    print(f"\n=== [1] bf16-cause: 3-way decomposition (union_1p vs micro_accum_1p vs {world_size}-rank {backend}) ===")
    target_sizes = [s for s in sizes if s in ("tiny", "5M", "28M")] or [sizes[0]]
    precisions: tuple[Literal["fp32", "bf16"], ...] = ("fp32", "bf16") if backend == "nccl" else ("fp32",)
    cases: list[tuple[AdaptationMode, Literal["domain", "replay_distill"]]] = [
        ("head_only", "domain"),
        ("lora_temporal", "domain"),
        ("full_ft", "domain"),
        ("lora_temporal", "replay_distill"),
        ("full_ft", "replay_distill"),
    ]

    rows: list[dict[str, Any]] = []
    print(
        f"{'size':<5} {'prec':<4} {'mode':<13} {'regime':<14} | "
        f"{'U-M s1_g':>9} {'M-D s1_g':>9} {'U-D s1_g':>9} {'flip%':>6} | "
        f"{'U-M upd':>9} {'M-D upd':>9} {'U-D upd':>9} {'U-D 1-cos':>9} {'xr':>3}"
    )

    for size in target_sizes:
        sec_dir = work_root / f"bf16_cause_{size}"
        u_dir, m_dir, d_dir = sec_dir / "union", sec_dir / "micro", sec_dir / "dist"
        scenarios = []
        for prec in precisions:
            for mode, regime in cases:
                sid = f"{size}_{prec}_{mode}_{regime}"
                scenarios.append(
                    {
                        "id": sid,
                        "size": size,
                        "world_size": world_size,
                        "precision": prec,
                        "mode": mode,
                        "regime": regime,
                        "seed": 11,
                        "steps": steps,
                        "per_rank_batch": DEFAULT_PER_RANK_BATCH,
                    }
                )
        _invoke_worker(
            {
                "kind": "single_ref",
                "backend": backend,
                "device_map": device_map,
                "out_dir": str(u_dir),
                "scenarios": scenarios,
            },
            work_dir=sec_dir,
            nproc=None,
        )
        _invoke_worker(
            {
                "kind": "micro_accum",
                "backend": backend,
                "device_map": device_map,
                "out_dir": str(m_dir),
                "scenarios": scenarios,
            },
            work_dir=sec_dir,
            nproc=None,
        )
        _invoke_worker(
            {
                "kind": "dist_run",
                "backend": backend,
                "device_map": device_map,
                "out_dir": str(d_dir),
                "scenarios": scenarios,
            },
            work_dir=sec_dir,
            nproc=world_size,
        )
        for item in scenarios:
            sid = item["id"]
            u_run = _load_run_tensors(u_dir / f"{sid}.pt")
            m_run = _load_run_tensors(m_dir / f"{sid}.pt")
            d_run = _load_run_tensors(d_dir / f"{sid}.pt")
            u_vs_m = _compare_pair(m_run, u_run)
            m_vs_d = _compare_pair(d_run, m_run)
            u_vs_d = _compare_pair(d_run, u_run)
            row = {
                "size": size,
                "world_size": world_size,
                "precision": item["precision"],
                "mode": item["mode"],
                "regime": item["regime"],
                "xr_eq": bool(d_run["xr_eq"]),
                "xr_max": float(d_run["xr_max"]),
                "union_vs_micro": u_vs_m,
                "micro_vs_dist": m_vs_d,
                "union_vs_dist": u_vs_d,
            }
            rows.append(row)
            flip_pct = 100.0 * u_vs_d["step1_sign_flip"]
            print(
                f"{size:<5} {item['precision']:<4} {item['mode']:<13} {item['regime']:<14} | "
                f"{_fmt(u_vs_m['step1_grad_l2']):>9} {_fmt(m_vs_d['step1_grad_l2']):>9} "
                f"{_fmt(u_vs_d['step1_grad_l2']):>9} {flip_pct:5.1f}% | "
                f"{_fmt(u_vs_m['upd_l2']):>9} {_fmt(m_vs_d['upd_l2']):>9} "
                f"{_fmt(u_vs_d['upd_l2']):>9} {_fmt(u_vs_d['1m_upd_cos']):>9} "
                f"{'yes' if row['xr_eq'] else 'NO':>3}"
            )
        shutil.rmtree(sec_dir, ignore_errors=True)
    return rows


def _run_section_compile_var(
    *,
    backend: Literal["gloo", "nccl"],
    device_map: Literal["per-rank", "shared"],
    sizes: Sequence[str],
    world_size: int,
    repeats: int,
    steps: int,
    work_root: Path,
) -> list[dict[str, Any]]:
    print(f"\n=== [2] compile-var: {repeats}x fresh-process repeats ({world_size}-rank {backend}) ===")
    target_sizes = [s for s in sizes if s in ("tiny", "5M")] or [sizes[0]]
    configs: list[tuple[str, AdaptationMode, Literal["domain", "replay_distill"], bool, dict[str, str]]] = [
        ("eager_lora_variate", "lora_variate", "domain", False, {}),
        ("compile_default_lora_variate", "lora_variate", "domain", True, {}),
        ("compile_default_lora_replay", "lora_variate", "replay_distill", True, {}),
    ]
    if backend == "nccl":
        configs.append(
            ("compile_nocache_lora_variate", "lora_variate", "domain", True, {"TORCHINDUCTOR_FX_GRAPH_CACHE": "0"})
        )

    rows: list[dict[str, Any]] = []
    print(
        f"{'size':<5} {'case':<29} | {'ref-ref upd':>11} {'dist-dist':>11} | "
        f"{'d_k vs r_1 (min..max)':>23} {'graphs':>8} {'xr_eq':>5}"
    )

    for size in target_sizes:
        sec_dir = work_root / f"compile_var_{size}"
        for label, mode, regime, do_compile, env_extra in configs:
            refs: list[dict[str, Any]] = []
            dists: list[dict[str, Any]] = []
            for rep_i in range(repeats):
                r_dir = sec_dir / f"{label}_r{rep_i}_ref"
                d_dir = sec_dir / f"{label}_r{rep_i}_dist"
                sc = [
                    {
                        "id": label,
                        "size": size,
                        "world_size": world_size,
                        "precision": "fp32",
                        "mode": mode,
                        "regime": regime,
                        "seed": 17,
                        "steps": steps,
                        "per_rank_batch": DEFAULT_PER_RANK_BATCH,
                        "compile_model": do_compile,
                        "empty_trailing_shard": True,
                    }
                ]
                _invoke_worker(
                    {
                        "kind": "single_ref",
                        "backend": backend,
                        "device_map": device_map,
                        "out_dir": str(r_dir),
                        "scenarios": sc,
                    },
                    work_dir=sec_dir,
                    nproc=None,
                    extra_env=env_extra,
                )
                _invoke_worker(
                    {
                        "kind": "dist_run",
                        "backend": backend,
                        "device_map": device_map,
                        "out_dir": str(d_dir),
                        "scenarios": sc,
                    },
                    work_dir=sec_dir,
                    nproc=world_size,
                    extra_env=env_extra,
                )
                refs.append(_load_run_tensors(r_dir / f"{label}.pt"))
                dists.append(_load_run_tensors(d_dir / f"{label}.pt"))

            ref_pair_upd = [
                _compare_pair(refs[j], refs[i])["upd_l2"] for i in range(repeats) for j in range(i + 1, repeats)
            ]
            dist_pair_upd = [
                _compare_pair(dists[j], dists[i])["upd_l2"] for i in range(repeats) for j in range(i + 1, repeats)
            ]
            d_vs_r1 = [_compare_pair(dists[k], refs[0]) for k in range(repeats)]
            d_vs_rk = [_compare_pair(dists[k], refs[k]) for k in range(repeats)]
            ref_max = max(ref_pair_upd) if ref_pair_upd else 0.0
            dist_max = max(dist_pair_upd) if dist_pair_upd else 0.0
            r1_upds = [m["upd_l2"] for m in d_vs_r1]
            all_xr = all(bool(d["xr_eq"]) for d in dists)
            graphs_vals = [d.get("unique_graphs") for d in dists]
            row = {
                "size": size,
                "world_size": world_size,
                "case": label,
                "repeats": repeats,
                "ref_pairwise_max_upd_l2": ref_max,
                "dist_pairwise_max_upd_l2": dist_max,
                "dist_vs_ref1": d_vs_r1,
                "dist_vs_refk": d_vs_rk,
                "all_xr_eq": all_xr,
                "unique_graphs": graphs_vals,
            }
            rows.append(row)
            g_str = ",".join(str(g) for g in graphs_vals) if any(g is not None for g in graphs_vals) else "-"
            span = f"{_fmt(min(r1_upds))}..{_fmt(max(r1_upds))}"
            print(
                f"{size:<5} {label:<29} | {_fmt(ref_max):>11} {_fmt(dist_max):>11} | "
                f"{span:>23} {g_str:>8} {'yes' if all_xr else 'NO':>5}"
            )
        shutil.rmtree(sec_dir, ignore_errors=True)
    return rows


def _run_section_calibrate(
    *,
    backend: Literal["gloo", "nccl"],
    device_map: Literal["per-rank", "shared"],
    sizes: Sequence[str],
    ranks: Sequence[int],
    seeds: Sequence[int],
    with_fp64: bool,
    steps: int,
    work_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(
        f"\n=== [3] calibrate: exact float64 L2 + cosine parity "
        f"sizes={list(sizes)} ranks={list(ranks)} seeds={list(seeds)} ==="
    )
    precisions: list[Literal["fp64", "fp32", "bf16"]] = []
    if with_fp64:
        precisions.append("fp64")
    precisions.append("fp32")
    if backend == "nccl":
        precisions.append("bf16")

    mode_regimes: list[tuple[AdaptationMode, Literal["domain", "replay_distill", "replay_distill_moved"]]] = [
        ("head_only", "domain"),
        ("lora_variate", "domain"),
        ("lora_temporal", "domain"),
        ("full_ft", "domain"),
        ("head_only", "replay_distill"),
        ("lora_variate", "replay_distill"),
        ("lora_temporal", "replay_distill"),
        ("full_ft", "replay_distill_moved"),
    ]

    all_rows: list[dict[str, Any]] = []
    print(
        f"{'size':<5} {'N':>1} {'prec':<4} {'mode':<13} {'regime':<20} {'sd':>2} | "
        f"{'s1_g_l2':>9} {'s1_1-cos':>9} {'param_l2':>9} {'upd_l2':>9} {'1-u_cos':>9} {'loss':>9} {'val':>9} {'xr':>3}"
    )

    for size in sizes:
        size_precs = [p for p in precisions if not (p == "fp64" and size in ("123M", "209M"))]
        for ws in ranks:
            sec_dir = work_root / f"calib_{size}_n{ws}"
            r_dir, d_dir = sec_dir / "ref", sec_dir / "dist"
            scenarios = []
            for prec in size_precs:
                for mode, regime in mode_regimes:
                    for seed in seeds:
                        sid = f"{size}_n{ws}_{prec}_{mode}_{regime}_s{seed}"
                        scenarios.append(
                            {
                                "id": sid,
                                "size": size,
                                "world_size": ws,
                                "precision": prec,
                                "mode": mode,
                                "regime": regime,
                                "seed": seed,
                                "steps": steps,
                                "per_rank_batch": DEFAULT_PER_RANK_BATCH,
                            }
                        )
            _invoke_worker(
                {
                    "kind": "single_ref",
                    "backend": backend,
                    "device_map": device_map,
                    "out_dir": str(r_dir),
                    "scenarios": scenarios,
                },
                work_dir=sec_dir,
                nproc=None,
            )
            _invoke_worker(
                {
                    "kind": "dist_run",
                    "backend": backend,
                    "device_map": device_map,
                    "out_dir": str(d_dir),
                    "scenarios": scenarios,
                },
                work_dir=sec_dir,
                nproc=ws,
            )
            for item in scenarios:
                sid = item["id"]
                ref = _load_run_tensors(r_dir / f"{sid}.pt")
                dist_r = _load_run_tensors(d_dir / f"{sid}.pt")
                cmp = _compare_pair(dist_r, ref)
                row = {
                    "size": size,
                    "world_size": ws,
                    "precision": item["precision"],
                    "mode": item["mode"],
                    "regime": item["regime"],
                    "seed": item["seed"],
                    "xr_eq": bool(dist_r["xr_eq"]),
                    "xr_max": float(dist_r["xr_max"]),
                    **cmp,
                }
                all_rows.append(row)
                print(
                    f"{size:<5} {ws:>1} {item['precision']:<4} {item['mode']:<13} "
                    f"{item['regime']:<20} {item['seed']:>2} | "
                    f"{_fmt(row['step1_grad_l2']):>9} {_fmt(row['step1_1m_grad_cos']):>9} "
                    f"{_fmt(row['param_l2']):>9} {_fmt(row['upd_l2']):>9} {_fmt(row['1m_upd_cos']):>9} "
                    f"{_fmt(row['loss_rel']):>9} {_fmt(row['val_rel']):>9} {'yes' if row['xr_eq'] else 'NO':>3}"
                )
            shutil.rmtree(sec_dir, ignore_errors=True)

    summary = summarize_calibration_rows(all_rows)
    print("\n--- Calibration worst-case summary by (precision, size, mode) ---")
    print(
        f"{'prec':<4} {'size':<5} {'mode':<13} {'n':>3} | "
        f"{'max_s1_1-c':>10} {'max_p_l2':>9} {'max_u_l2':>9} {'max_1-u_c':>10} "
        f"{'max_loss':>9} {'max_val':>9} {'xr_eq':>7}"
    )
    for s in summary:
        print(
            f"{s['precision']:<4} {s['size']:<5} {s['mode']:<13} {s['count']:>3} | "
            f"{_fmt(s['max_step1_1m_grad_cos']):>10} {_fmt(s['max_param_l2']):>9} "
            f"{_fmt(s['max_upd_l2']):>9} {_fmt(s['max_1m_upd_cos']):>10} "
            f"{_fmt(s['max_loss_rel']):>9} {_fmt(s['max_val_rel']):>9} "
            f"{s['xr_eq_count']}/{s['count']:>3}"
        )
    return all_rows, summary


def _safe_max(vals: Sequence[float]) -> float:
    if not vals:
        return float("nan")
    if any(math.isnan(v) for v in vals):
        return float("nan")
    return max(vals)


def summarize_calibration_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate worst-case exact metrics grouped by `(precision, size, mode)`."""
    buckets: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["precision"]), str(row["size"]), str(row["mode"]))
        buckets.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (prec, size, mode), group in buckets.items():
        summary.append(
            {
                "precision": prec,
                "size": size,
                "mode": mode,
                "count": len(group),
                "xr_eq_count": sum(1 for r in group if r["xr_eq"]),
                "max_step1_grad_l2": _safe_max([float(r["step1_grad_l2"]) for r in group]),
                "max_step1_1m_grad_cos": _safe_max([float(r["step1_1m_grad_cos"]) for r in group]),
                "max_param_l2": _safe_max([float(r["param_l2"]) for r in group]),
                "max_upd_l2": _safe_max([float(r["upd_l2"]) for r in group]),
                "max_1m_upd_cos": _safe_max([float(r["1m_upd_cos"]) for r in group]),
                "max_loss_rel": _safe_max([float(r["loss_rel"]) for r in group]),
                "max_val_rel": _safe_max([float(r["val_rel"]) for r in group]),
            }
        )
    return summary


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--device-map", choices=("per-rank", "shared"), default="per-rank")
    parser.add_argument("--sizes", nargs="+", choices=ALL_SIZE_CHOICES, default=["tiny", "5M", "28M", "123M", "209M"])
    parser.add_argument("--ranks", nargs="+", type=int, choices=(2, 4), default=[2, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--compile-repeats", type=int, default=3)
    parser.add_argument("--with-fp64", action="store_true", help="Also calibrate float64 on tiny/5M/28M.")
    parser.add_argument("--only", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path for structured results.")
    parser.add_argument("--_worker-job", type=str, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args._worker_job is not None:
        return _worker_main(args)

    require_positive_finite(parser, "--steps", float(args.steps))
    require_positive_finite(parser, "--compile-repeats", float(args.compile_repeats))
    check_tmpdir_for_workers()

    dedup_ranks: list[int] = []
    for r in args.ranks:
        if r not in dedup_ranks:
            dedup_ranks.append(r)

    max_ws = max(dedup_ranks)
    err = validate_distributed_device_map(
        backend=args.backend, device_map=args.device_map, world_size=max_ws, local_rank=None
    )
    if err is not None:
        print(f"error: {err}", file=sys.stderr)
        return 2

    banner = [
        f"backend={args.backend}",
        f"device_map={args.device_map}",
        f"torch={torch.__version__}",
        f"os={platform.system()} {platform.machine()}",
        f"git={git_head_commit() or 'unknown'}",
    ]
    if args.backend == "nccl":
        banner.extend(cuda_env_banner_parts(allow_cuda_context=False))
    print("== distributed_calibration_probe | " + " | ".join(banner) + " ==")

    payload: dict[str, Any] = {
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
            "seeds": list(args.seeds),
            "steps": args.steps,
        }
    }

    with tempfile.TemporaryDirectory(prefix="timestral_calib_") as tmp:
        work_root = Path(tmp)
        if "bf16-cause" in args.only:
            payload["bf16_cause"] = _run_section_bf16_cause(
                backend=args.backend,
                device_map=args.device_map,
                sizes=args.sizes,
                world_size=max_ws,
                steps=args.steps,
                work_root=work_root,
            )
        if "compile-var" in args.only:
            payload["compile_var"] = _run_section_compile_var(
                backend=args.backend,
                device_map=args.device_map,
                sizes=args.sizes,
                world_size=max_ws,
                repeats=args.compile_repeats,
                steps=args.steps,
                work_root=work_root,
            )
        if "calibrate" in args.only:
            cal_rows, cal_summary = _run_section_calibrate(
                backend=args.backend,
                device_map=args.device_map,
                sizes=args.sizes,
                ranks=dedup_ranks,
                seeds=args.seeds,
                with_fp64=bool(args.with_fp64),
                steps=args.steps,
                work_root=work_root,
            )
            payload["calibrate_rows"] = cal_rows
            payload["calibrate_summary"] = cal_summary

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.json_out, payload)
        print(f"\nStructured results written to: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
