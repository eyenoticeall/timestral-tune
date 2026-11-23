"""Probe: multi-rank `TimestralTrainer` parity, telemetry, and Stage 3 CUDA/NCCL driver.

Runs the real `TimestralTrainer` across ranks under `torchrun` on either `gloo` (CPU) or `nccl`
(CUDA) and compares against a one-process reference saved on the same device type. Reports
measured numbers without grading float32/bf16 tolerances; exits non-zero only on crashes, detected
hangs, cross-rank mismatches of model state (parameters + buffers) or `TrainingResult` in a
deterministic scenario, failed export checks, reference mismatches, or raised exceptions.

Two-step workflow (one-process reference first, then `torchrun`):

    # CPU / gloo (N=2):
    TMPDIR=/tmp uv run python scripts/probes/distributed_trainer_check.py \\
        --write-reference /tmp/ref_cpu_n2.zip --device cpu --world-size 2
    TMPDIR=/tmp uv run torchrun --master-addr=127.0.0.1 --master-port=29501 --nproc_per_node=2 \\
        scripts/probes/distributed_trainer_check.py \\
        --backend gloo --reference /tmp/ref_cpu_n2.zip

    # Single-GPU shared NCCL (N=2 on cuda:0):
    TMPDIR=/tmp uv run python scripts/probes/distributed_trainer_check.py \\
        --write-reference /tmp/ref_cuda_n2.zip --device cuda:0 --world-size 2
    NCCL_MULTI_RANK_GPU_ENABLE=1 TMPDIR=/tmp uv run torchrun --master-addr=127.0.0.1 \\
        --master-port=29501 --nproc_per_node=2 scripts/probes/distributed_trainer_check.py \\
        --backend nccl --device-map shared --reference /tmp/ref_cuda_n2.zip

    # Multi-GPU per-rank NCCL (N=4 on cuda:0..3, with released checkpoints):
    TMPDIR=/tmp uv run python scripts/probes/distributed_trainer_check.py \\
        --write-reference /tmp/ref_cuda_n4.zip --device cuda:0 --world-size 4 --sizes 5M 28M
    TMPDIR=/tmp uv run torchrun --master-addr=127.0.0.1 --master-port=29501 --nproc_per_node=4 \\
        scripts/probes/distributed_trainer_check.py \\
        --backend nccl --device-map per-rank --sizes 5M 28M --reference /tmp/ref_cuda_n4.zip

Reference storage (`--reference-mode`, default `auto`):
- `auto`: stores exact float64 `delta = final - init` for every `float64` scenario with
  `numel <= 1,000,000` (100% of `tiny_random` float64 scenarios, all `5M` LoRA + `head_only`, and
  all `28M` LoRA) and for any `float32`/`bf16` scenario with `numel <= 8192`; uses a deterministic
  `k = 8192` float64 CountSketch (`64 KiB`/scenario) for larger vectors. Keeps `tiny_random` at
  ~25 MiB (100% exact on float64) and `5M`+`28M` at ~37 MiB per world size with O(1) peak RAM.
- `sketch`: uses `k = 8192` CountSketch whenever `numel > 8192` (~1.9 MiB on `tiny_random`,
  ~4.8 MiB on `5M`+`28M`). CountSketch has ~0.78% relative L2 standard error well above its float64
  bucket-accumulation noise floor `floor_rel = 0.25 * sqrt(n / k) * eps_f64` (~4.3e-16 at 490K,
  ~1.4e-15 at 5M, ~3.2e-15 at 28M; measured 1.3x–7x inflation below ~1e-14 on 5M/28M). Any sketch
  estimate within 10x of `floor_rel` is flagged `sketch_bound(floor~...)` in `notes`, and any
  hash mismatch with zero sketch difference is flagged `sketch_underflow(sha_differs)`.
- `exact`: streams full native-dtype `final_trainable` vectors into the zip archive (O(1) peak
  memory; ~45 MiB on `tiny_random`, ~2.2 GiB on `5M`+`28M`).

Column definitions:
- `xr_eq`: exact 256-bit SHA-256 unanimity across all ranks over all `model.named_parameters()`
  and all `model.named_buffers()`. (`TrainingResult` unanimity across all 10 fields is checked
  separately; disagreement appends `RESULT_MISMATCH` to `notes` and fails deterministic runs.)
- `xr_max`: `max_{r >= 1} ||state_r - state_0||_inf` in float64 over all `model.named_parameters()`
  and all `model.named_buffers()` (propagates `NaN`).
- `param_l2`, `upd_l2`: relative L2 of `final_trainable` and `delta = final_trainable - init_trainable`
  computed over `trainer._trainable` only (for sensitivity in `head_only`/LoRA), against the
  reference's exact float64 norms. Bitwise-identical non-finite vectors report `nan`, never `0`.
- `loss_rel`, `val_rel`: NaN-propagating maximum relative step-loss and validation-loss difference.
- `wall_s`: `max_r` wall time of `trainer.fit()` starting after a cross-rank barrier + CUDA sync
  and ending after a final CUDA sync.
- `step_ms`: `max_r` mean full per-step wall time strictly after the first two optimiser steps
  (steps `3..K`), measured symmetrically in both one-process reference and multi-process modes from
  the end of `strategy.on_step_end(k-1)` to the end of `strategy.on_step_end(k)` (with CUDA sync)
  minus any validation time (`strategy.evaluate_val()` + `strategy.on_evaluate()`) occurring between
  them. Includes epoch rollover, DataLoader fetch, batch/replay preparation and host-to-device
  transfer, `zero_grad`, forward, loss, backward, distributed collectives, gradient clipping,
  `optimizer.step()`, `scheduler.step()`, and `on_step_end` callbacks/sync; excludes step-1/2
  startup/compilation and all periodic validation (including lazy validation worker spawn).
- `tail_s`: `max_r` wall time from the end of the final `strategy.on_step_end(K)` (with CUDA sync)
  to `fit()` returning (includes any final-step validation, `finalize_best()`, `on_fit_end()`, and
  DataLoader worker teardown).
- `vram_mib`: `max_r` peak `torch.cuda.max_memory_allocated()` during `fit()` in MiB (`2**20` bytes, CUDA only).
- `val_s` (in `notes` when validating): `max_r` cumulative wall time spent inside
  `strategy.evaluate_val()` and `strategy.on_evaluate()` (including lazy validation worker spawn).
- `graphs=s1:A->end:B(+C,bumps_post_s1=D)` (in `notes` for compiled scenarios): `A` is Dynamo
  `unique_graphs` after optimiser step 1, `B` is `unique_graphs` after `fit()`, `+C = B - A` new
  graphs compiled after step 1, and `D` is the number of distinct optimiser steps after step 1 on
  which `unique_graphs` increased (`bumps_post_s1=1` indicates one specialisation, e.g. for the
  shorter trailing batch, rather than repeated recompilation).

See `docs/phases/phase-5-multi-gpu.md` §3 and `docs/SPEC.md` §5.6, §12.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import datetime
import gc
import hashlib
import json
import math
import os
import platform
import re
import struct
import sys
import tempfile
import time
import traceback
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from _common import SIZES, float64_reference, load, rel, tiny_random_forecaster
from _dist_common import (
    CTX_PATCHES,
    DOM_HORIZON,
    HOR_PATCHES,
    REP_HORIZON,
    check_tmpdir_for_workers,
    cuda_env_banner_parts,
    files_sha256,
    git_head_commit,
    make_synthetic_dataset,
    require_positive_finite,
    validate_distributed_device_map,
)
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor

from timestral_tune import LoraConfig, ReplayConfig, TimestralTrainer, TrainingConfig
from timestral_tune.config import AdaptationMode
from timestral_tune.data import DHFDataCollator, TimeSeriesDataset
from timestral_tune.export import dataset_fingerprint
from timestral_tune.models import TuneTimestralModel, denormalise_quantiles, load_model, training_forward
from timestral_tune.trainer import TrainingResult, compute_loss


def _torch_major_minor() -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if match is None:
        return (0, 0)
    return int(match.group(1)), int(match.group(2))


# PyTorch < 2.8 on macOS hangs at interpreter exit after DataLoader worker processes run (VERIFICATION.md §10).
WORKERS_HANG_AT_EXIT = sys.platform == "darwin" and _torch_major_minor() < (2, 8)

SCHEMA_VERSION = 5
DEFAULT_BATCH_SIZE = 2
DEFAULT_REPLAY_BATCH_SIZE = 2
DEFAULT_STEPS = 6
DEFAULT_TIMEOUT_S = 180.0
SKETCH_DIM = 8192
SKETCH_CHUNK = 262_144
AUTO_EXACT_F64_MAX_NUMEL = 1_000_000
MOVED_WARMUP_STEPS = 3
MOVED_WARMUP_BATCH_SIZE = 4

ReferenceMode = Literal["auto", "sketch", "exact"]

_DTYPE_FROM_NAME: dict[str, torch.dtype] = {
    "float64": torch.float64,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
_NAME_FROM_DTYPE: dict[torch.dtype, str] = {v: k for k, v in _DTYPE_FROM_NAME.items()}
_DTYPE_NBYTES: dict[str, int] = {"float64": 8, "float32": 4, "bfloat16": 2, "float16": 2}
_PROBE_SOURCE_FILES: tuple[Path, ...] = (
    Path(__file__).resolve(),
    Path(__file__).resolve().with_name("_dist_common.py"),
    Path(__file__).resolve().with_name("_common.py"),
)


def _current_script_sha256() -> str:
    return files_sha256(_PROBE_SOURCE_FILES)


# --------------------------------------------------------------------------- declarative scenarios


@dataclass(frozen=True)
class ScenarioSpec:
    """Declarative specification for one `TimestralTrainer.fit()` measurement scenario."""

    name: str
    group: int
    mode: AdaptationMode
    dtype: Literal["float64", "float32"] = "float32"
    mixed_precision: Literal["fp32", "bf16"] = "fp32"
    replay_fraction: float = 0.0
    distill_weight: float = 0.0
    moved_student: bool = False
    with_val: bool = False
    eval_every: int | None = None
    restore_best: bool = False
    early_stopping_patience: int | None = None
    compile_model: bool = False
    num_workers: int = 0
    shuffle: bool = False
    sign_flip: bool = False
    lora_dropout: float = 0.0
    test_export: bool = False
    cuda_only: bool = False
    compare_ref: bool = True
    deterministic: bool = True
    empty_trailing_shard: bool = False
    learning_rate_override: float | None = None
    steps_override: int | None = None
    static_note: str = ""

    @property
    def uses_replay(self) -> bool:
        return self.replay_fraction > 0.0 or self.distill_weight > 0.0

    def to_meta(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "mode": self.mode,
            "dtype": self.dtype,
            "mixed_precision": self.mixed_precision,
            "replay_fraction": self.replay_fraction,
            "distill_weight": self.distill_weight,
            "moved_student": self.moved_student,
            "with_val": self.with_val,
            "eval_every": self.eval_every,
            "restore_best": self.restore_best,
            "early_stopping_patience": self.early_stopping_patience,
            "compile_model": self.compile_model,
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "sign_flip": self.sign_flip,
            "lora_dropout": self.lora_dropout,
            "test_export": self.test_export,
            "cuda_only": self.cuda_only,
            "compare_ref": self.compare_ref,
            "deterministic": self.deterministic,
            "empty_trailing_shard": self.empty_trailing_shard,
            "learning_rate_override": self.learning_rate_override,
            "steps_override": self.steps_override,
            "static_note": self.static_note,
        }


def build_scenarios() -> tuple[ScenarioSpec, ...]:
    """Return the complete declarative scenario table covering Stage 3 groups 1–8."""
    items: list[ScenarioSpec] = []

    # 1. Domain loss only: all four modes in float64 and float32, deterministic, trailing partial batch.
    for dt in ("float64", "float32"):
        suffix = "f64" if dt == "float64" else "f32"
        for mode in ("head_only", "lora_variate", "lora_temporal", "full_ft"):
            items.append(ScenarioSpec(name=f"domain_{mode}_{suffix}", group=1, mode=mode, dtype=dt))

    # 2. Replay regimes (`replay_only`, `distill_only`, `both`) for head_only, lora_variate, full_ft.
    # For float32 `full_ft` + distillation, keep the identical-start rows explicitly labelled with the
    # PyTorch `F.linear` `requires_grad` dispatch caveat (spec §12, `linear_requires_grad_dispatch.py`)
    # AND include moved-student variants (matching T15(j)'s `test_t15_j_full_ft_distill_float32`) so
    # float32 `full_ft` distillation parity against one-process is meaningfully measured.
    for dt in ("float64", "float32"):
        suffix = "f64" if dt == "float64" else "f32"
        for mode in ("head_only", "lora_variate", "full_ft"):
            for regime, rf, dw in (
                ("replay_only", 0.25, 0.0),
                ("distill_only", 0.0, 0.5),
                ("both", 0.25, 0.5),
            ):
                is_f32_full_ft_distill = dt == "float32" and mode == "full_ft" and dw > 0.0
                items.append(
                    ScenarioSpec(
                        name=f"replay_{mode}_{regime}_{suffix}",
                        group=2,
                        mode=mode,
                        dtype=dt,
                        replay_fraction=rf,
                        distill_weight=dw,
                        static_note="identical_start(F.linear_rg_dispatch,spec_§12)" if is_f32_full_ft_distill else "",
                    )
                )
                if is_f32_full_ft_distill:
                    items.append(
                        ScenarioSpec(
                            name=f"replay_{mode}_{regime}_moved_{suffix}",
                            group=2,
                            mode=mode,
                            dtype=dt,
                            replay_fraction=rf,
                            distill_weight=dw,
                            moved_student=True,
                            static_note="moved_student",
                        )
                    )

    # 3. Periodic validation with restore_best=True and early stopping.
    for dt in ("float64", "float32"):
        suffix = "f64" if dt == "float64" else "f32"
        items.append(
            ScenarioSpec(
                name=f"val_restore_early_stop_{suffix}",
                group=3,
                mode="head_only",
                dtype=dt,
                with_val=True,
                eval_every=2,
                restore_best=True,
                early_stopping_patience=2,
                learning_rate_override=5e-3,
                steps_override=10,
            )
        )

    # 4. CUDA bf16 autocast across all four modes (with and without replay/distillation).
    items.extend(
        [
            ScenarioSpec(name="bf16_head_only", group=4, mode="head_only", mixed_precision="bf16", cuda_only=True),
            ScenarioSpec(
                name="bf16_lora_variate", group=4, mode="lora_variate", mixed_precision="bf16", cuda_only=True
            ),
            ScenarioSpec(
                name="bf16_lora_temporal", group=4, mode="lora_temporal", mixed_precision="bf16", cuda_only=True
            ),
            ScenarioSpec(name="bf16_full_ft", group=4, mode="full_ft", mixed_precision="bf16", cuda_only=True),
            ScenarioSpec(
                name="bf16_lora_variate_replay",
                group=4,
                mode="lora_variate",
                mixed_precision="bf16",
                replay_fraction=0.25,
                distill_weight=0.5,
                cuda_only=True,
            ),
            ScenarioSpec(
                name="bf16_full_ft_replay",
                group=4,
                mode="full_ft",
                mixed_precision="bf16",
                replay_fraction=0.25,
                distill_weight=0.5,
                cuda_only=True,
            ),
        ]
    )

    # 5. compile_model=True, compile_mode="default" with shorter trailing batch and empty trailing shard.
    items.extend(
        [
            ScenarioSpec(
                name="compile_default_lora_variate",
                group=5,
                mode="lora_variate",
                compile_model=True,
                empty_trailing_shard=True,
            ),
            ScenarioSpec(
                name="compile_default_lora_replay",
                group=5,
                mode="lora_variate",
                replay_fraction=0.25,
                distill_weight=0.5,
                compile_model=True,
                empty_trailing_shard=True,
            ),
        ]
    )

    # 6. num_workers=2 with uneven trailing training and validation shards (replay stays num_workers=0).
    items.append(
        ScenarioSpec(
            name="workers_uneven_f32",
            group=6,
            mode="head_only",
            with_val=True,
            eval_every=3,
            restore_best=False,
            replay_fraction=0.25,
            distill_weight=0.3,
            num_workers=2,
            empty_trailing_shard=True,
        )
    )

    # 7. Multi-rank export() check: rank 0 writes valid export + tune_metadata.json, peers write nothing.
    items.append(
        ScenarioSpec(
            name="export_multirank_f32",
            group=7,
            mode="lora_variate",
            replay_fraction=0.25,
            distill_weight=0.5,
            test_export=True,
            steps_override=4,
        )
    )

    # 8. Stochastic run (shuffle + sign_flip + LoRA dropout): cross-rank equality only.
    items.append(
        ScenarioSpec(
            name="stochastic_lora_variate_f32",
            group=8,
            mode="lora_variate",
            replay_fraction=0.25,
            distill_weight=0.3,
            shuffle=True,
            sign_flip=True,
            lora_dropout=0.2,
            compare_ref=False,
            deterministic=False,
        )
    )

    return tuple(items)


ALL_SCENARIOS: tuple[ScenarioSpec, ...] = build_scenarios()
SCENARIO_BY_NAME: dict[str, ScenarioSpec] = {spec.name: spec for spec in ALL_SCENARIOS}


# --------------------------------------------------------------------------- configs, digests & sketches


def _lora_for(mode: AdaptationMode, dropout: float) -> LoraConfig | None:
    if mode == "lora_variate":
        return LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("q", "k", "v", "o"), dropout=dropout)
    if mode == "lora_temporal":
        return LoraConfig(r=4, lora_alpha=8, axes=("time",), targets=("q", "k", "v", "o"), dropout=dropout)
    return None


def _scenario_lr(spec: ScenarioSpec, *, is_released: bool) -> float:
    if spec.learning_rate_override is not None:
        return spec.learning_rate_override
    if is_released:
        return 1e-4 if spec.mode == "full_ft" else 1e-3
    return 1e-3 if spec.mode == "full_ft" else 1e-2


def _moved_warmup_lr(*, is_released: bool) -> float:
    return 1e-4 if is_released else 5e-3


def _build_trainer_configs(
    spec: ScenarioSpec,
    *,
    is_released: bool,
    world_size: int,
    per_rank_batch: int,
    per_rank_replay_batch: int,
    steps: int,
    seed: int,
    config_device: str,
    distributed: bool,
) -> tuple[TrainingConfig, LoraConfig | None, ReplayConfig | None]:
    """Construct resolved `(TrainingConfig, LoraConfig, ReplayConfig)` for a scenario."""
    eff_batch = per_rank_batch if distributed else world_size * per_rank_batch
    eff_replay_batch = per_rank_replay_batch if distributed else world_size * per_rank_replay_batch
    eff_steps = spec.steps_override if spec.steps_override is not None else steps
    lr = _scenario_lr(spec, is_released=is_released)

    cfg = TrainingConfig(
        model_name_or_path="local-stage3-probe",
        mode=spec.mode,
        device=config_device,
        batch_size=eff_batch,
        epochs=max(1, eff_steps),
        max_steps=eff_steps,
        warmup_steps=1,
        lr_schedule="cosine",
        learning_rate=lr,
        weight_decay=0.01,
        gradient_clip_val=1.0,
        mixed_precision=spec.mixed_precision,
        eval_every=spec.eval_every,
        eval_batch_size=eff_batch,
        early_stopping_patience=spec.early_stopping_patience,
        restore_best=spec.restore_best,
        log_every=max(1000, eff_steps + 1),
        shuffle=spec.shuffle,
        sign_flip=spec.sign_flip,
        cache_frozen_features=False,
        compile_model=spec.compile_model,
        compile_mode="default",
        num_workers=spec.num_workers,
        seed=seed + 107,
    )
    lora_cfg = _lora_for(spec.mode, spec.lora_dropout)
    rcfg: ReplayConfig | None = None
    if spec.uses_replay:
        # In one-process mode, `_precompute_replay_base_outputs` casts cached teacher quantiles to
        # `torch.float32` regardless of model dtype, so float64 references must run uncached
        # (`cache_base_outputs=False`) to stay in float64 (matching T15). In float32/bf16, keep the
        # default `cache_base_outputs=True` so multi-process live teachers compare against shipped defaults.
        cache_base = spec.dtype != "float64"
        rcfg = ReplayConfig(
            replay_fraction=spec.replay_fraction,
            distill_weight=spec.distill_weight,
            batch_size=eff_replay_batch,
            sign_flip=spec.sign_flip,
            cache_base_outputs=cache_base,
            smooth_beta=1e-3,
        )
    return cfg, lora_cfg, rcfg


def _scenario_config_digest(
    cfg: TrainingConfig,
    lora_cfg: LoraConfig | None,
    rcfg: ReplayConfig | None,
    *,
    spec: ScenarioSpec,
    per_rank_batch: int,
    per_rank_replay_batch: int,
    is_released: bool,
) -> str:
    """Canonical SHA-256 of the actual resolved training/LoRA/replay configs (invariant across 1p vs mp)."""
    cfg_dict: dict[str, Any] = {}
    for field_info in dataclasses.fields(cfg):
        if field_info.name in ("device", "output_dir"):
            continue
        val = getattr(cfg, field_info.name)
        if isinstance(val, Path):
            val = str(val)
        cfg_dict[field_info.name] = val
    # Normalise effective batch sizes to per-rank B so 1-process (N*B) and distributed (B) hash identically.
    cfg_dict["batch_size"] = per_rank_batch
    cfg_dict["eval_batch_size"] = per_rank_batch
    cfg_dict["resolved_learning_rate"] = cfg.resolved_learning_rate

    rcfg_dict = None
    if rcfg is not None:
        rcfg_dict = dict(rcfg.to_dict())
        rcfg_dict["batch_size"] = per_rank_replay_batch

    payload = {
        "training_config": cfg_dict,
        "lora_config": None if lora_cfg is None else lora_cfg.to_dict(),
        "replay_config": rcfg_dict,
        "moved_student_warmup": (
            {
                "steps": MOVED_WARMUP_STEPS,
                "batch_size": MOVED_WARMUP_BATCH_SIZE,
                "lr": _moved_warmup_lr(is_released=is_released),
            }
            if spec.moved_student
            else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flat_trainable_native(trainer: TimestralTrainer) -> Tensor:
    """Concatenate trainable parameters on CPU in their native dtype."""
    return torch.cat([p.detach().cpu().reshape(-1) for p in trainer._trainable])


def _tensor_sha256(tensor: Tensor) -> str:
    arr = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _sketch_seed(base_seed: int, model_name: str, scenario_name: str) -> int:
    digest = hashlib.sha256(f"timestral-tune:stage3-sketch:{base_seed}:{model_name}:{scenario_name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _sketch_floor_rel(numel: int, sketch_dim: int) -> float:
    """Estimated float64 bucket-accumulation relative noise floor: `0.25 * sqrt(n / k) * eps_f64`."""
    eps_f64 = float(torch.finfo(torch.float64).eps)
    return 0.25 * math.sqrt(max(1.0, float(numel) / float(sketch_dim))) * eps_f64


def _count_sketch_delta(
    init_native: Tensor,
    final_native: Tensor,
    *,
    sketch_dim: int,
    seed: int,
) -> tuple[Tensor, float, float, Literal["exact_delta", "count_sketch"]]:
    """Compute `(compact_vector_f64, final_norm_f64, update_norm_f64, kind)` in O(chunk) memory.

    When `numel <= sketch_dim`, returns the exact float64 `delta = final - init` vector (`exact_delta`).
    Otherwise returns a deterministic linear CountSketch of `delta` in float64 (`count_sketch`).
    Above ~10x the float64 bucket-accumulation noise floor (`_sketch_floor_rel`), its relative L2
    standard error is <= `1 / sqrt(2 * sketch_dim)` (~0.78% at `sketch_dim = 8192`).
    """
    n = init_native.numel()
    final_sq = 0.0
    upd_sq = 0.0
    if n <= sketch_dim:
        f64 = final_native.double()
        d64 = f64 - init_native.double()
        return d64, float(f64.norm().item()), float(d64.norm().item()), "exact_delta"

    g = torch.Generator().manual_seed(seed)
    sketch = torch.zeros(sketch_dim, dtype=torch.float64)
    for start in range(0, n, SKETCH_CHUNK):
        end = min(n, start + SKETCH_CHUNK)
        m = end - start
        f_chunk = final_native[start:end].double()
        d_chunk = f_chunk - init_native[start:end].double()
        final_sq += float(torch.dot(f_chunk, f_chunk).item())
        upd_sq += float(torch.dot(d_chunk, d_chunk).item())
        buckets = torch.randint(0, sketch_dim, (m,), generator=g, dtype=torch.int64)
        signs = torch.randint(0, 2, (m,), generator=g, dtype=torch.int64).mul_(2).sub_(1).to(torch.float64)
        sketch.scatter_add_(0, buckets, d_chunk * signs)
    return sketch, math.sqrt(final_sq), math.sqrt(upd_sq), "count_sketch"


def _select_reference_payload(
    init_vec: Tensor,
    final_vec: Tensor,
    *,
    ref_mode: ReferenceMode,
    spec: ScenarioSpec,
    seed: int,
    model_name: str,
) -> tuple[Tensor, float, float, Literal["exact_final", "exact_delta", "count_sketch"]]:
    """Choose and compute the reference tensor payload according to `ref_mode`."""
    if ref_mode == "exact":
        f64 = final_vec.double()
        d64 = f64 - init_vec.double()
        return final_vec, float(f64.norm().item()), float(d64.norm().item()), "exact_final"
    if ref_mode == "auto" and spec.dtype == "float64" and final_vec.numel() <= AUTO_EXACT_F64_MAX_NUMEL:
        f64 = final_vec.double()
        d64 = f64 - init_vec.double()
        return d64, float(f64.norm().item()), float(d64.norm().item()), "exact_delta"
    return _count_sketch_delta(
        init_vec,
        final_vec,
        sketch_dim=SKETCH_DIM,
        seed=_sketch_seed(seed, model_name, spec.name),
    )


def _tensor_to_bytes(tensor: Tensor) -> tuple[bytes, str, list[int]]:
    arr = tensor.detach().cpu().contiguous()
    dtype_name = _NAME_FROM_DTYPE.get(arr.dtype)
    if dtype_name is None:
        raise ValueError(f"unsupported tensor dtype for archive: {arr.dtype}")
    return arr.reshape(-1).view(torch.uint8).numpy().tobytes(), dtype_name, list(arr.shape)


def _bytes_to_tensor(raw: bytes, dtype_name: str, shape: Sequence[int]) -> Tensor:
    dtype = _DTYPE_FROM_NAME[dtype_name]
    buf = bytearray(raw)
    return torch.frombuffer(buf, dtype=torch.uint8).view(dtype).reshape(tuple(shape)).clone()


def _scalar_rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def _max_history_rel_nan_safe(
    got_hist: Sequence[tuple[int, float]],
    ref_hist: Sequence[Sequence[Any]],
) -> float:
    """Compute max relative error across aligned `(step, value)` histories, propagating any non-finite value."""
    if (
        len(got_hist) != len(ref_hist)
        or len(ref_hist) == 0
        or not all(s_got == int(s_ref) for (s_got, _), (s_ref, _) in zip(got_hist, ref_hist, strict=True))
    ):
        return float("nan")
    worst = 0.0
    for (_, v_got), (_, v_ref_raw) in zip(got_hist, ref_hist, strict=True):
        v_got_f = float(v_got)
        v_ref_f = float(v_ref_raw)
        if not math.isfinite(v_got_f) or not math.isfinite(v_ref_f):
            return float("nan")
        d = _scalar_rel(v_got_f, v_ref_f)
        if d > worst:
            worst = d
    return worst


def _encode_digest_words(digest: bytes) -> list[int]:
    return list(struct.unpack("<4q", hashlib.sha256(digest).digest()))


def _hash_model_state(model: TuneTimestralModel) -> bytes:
    """SHA-256 over every parameter in `sorted(named_parameters())` and every buffer in `sorted(named_buffers())`."""
    h = hashlib.sha256()
    for kind, items in (
        (b"param", sorted(model.named_parameters(), key=lambda kv: kv[0])),
        (b"buffer", sorted(model.named_buffers(), key=lambda kv: kv[0])),
    ):
        for name, tensor in items:
            arr = tensor.detach().cpu().contiguous()
            h.update(kind)
            h.update(name.encode("utf-8"))
            h.update(str(tuple(arr.shape)).encode("utf-8"))
            h.update(str(arr.dtype).encode("utf-8"))
            h.update(arr.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.digest()


def _hash_training_result(result: TrainingResult) -> bytes:
    """Canonical SHA-256 over all ten `TrainingResult` fields using exact `float.hex()` encodings."""

    def _f(val: float | None) -> str | None:
        return None if val is None else float(val).hex()

    payload = {
        "steps": result.steps,
        "epochs": result.epochs,
        "final_train_loss": _f(result.final_train_loss),
        "best_eval_loss": _f(result.best_eval_loss),
        "best_step": result.best_step,
        "stopped_early": result.stopped_early,
        "skipped_batches": result.skipped_batches,
        "restored_best": result.restored_best,
        "train_history": [[s, float(v).hex()] for s, v in result.train_history],
        "eval_history": [[s, float(v).hex()] for s, v in result.eval_history],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest()


def _dynamo_unique_graphs() -> int:
    try:
        return int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0))
    except Exception:
        return 0


def _reset_dynamo() -> None:
    try:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
    except Exception:
        pass


def _cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _format_num(value: float | None, *, width: int = 9, fmt: str = ".2e") -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width}{fmt}}"


def _load_models(sizes: Sequence[str], *, include_tiny: bool, seed: int) -> list[tuple[str, TuneTimestralModel]]:
    models: list[tuple[str, TuneTimestralModel]] = []
    if include_tiny:
        models.append(("tiny_random", tiny_random_forecaster(seed=seed).model))
    for size in sizes:
        models.append((size, load(size, device="cpu").model))
    return models


def _skip_reason(spec: ScenarioSpec, *, is_cuda: bool) -> str | None:
    if spec.cuda_only and not is_cuda:
        return "cuda_only"
    if spec.num_workers > 0 and WORKERS_HANG_AT_EXIT:
        return "macos_torch_lt_2_8_workers"
    return None


def _build_datasets_for_scenario(
    spec: ScenarioSpec,
    *,
    world_size: int,
    per_rank_batch: int,
    per_rank_replay_batch: int,
    seed: int,
) -> tuple[TimeSeriesDataset, TimeSeriesDataset | None, TimeSeriesDataset | None]:
    trailing = 1 if spec.empty_trailing_shard else max(1, world_size - 1)
    dom_len = world_size * per_rank_batch + trailing
    dom_ramp = 40.0 if spec.early_stopping_patience is not None else 0.0
    dom_ds = make_synthetic_dataset(
        dom_len, horizon=DOM_HORIZON, covariates=True, seed=seed + 1001, future_ramp=dom_ramp
    )

    val_ds: TimeSeriesDataset | None = None
    if spec.with_val:
        val_ramp = -40.0 if spec.early_stopping_patience is not None else 0.0
        val_ds = make_synthetic_dataset(
            2 * world_size + 1,
            horizon=DOM_HORIZON,
            covariates=True,
            seed=seed + 1002,
            future_ramp=val_ramp,
        )

    rep_ds: TimeSeriesDataset | None = None
    if spec.uses_replay:
        rep_len = world_size * per_rank_replay_batch + max(1, world_size - 1)
        rep_ds = make_synthetic_dataset(rep_len, horizon=REP_HORIZON, covariates=False, seed=seed + 1003)

    return dom_ds, val_ds, rep_ds


def _build_moved_full_ft_params(
    base_model_cpu: TuneTimestralModel,
    dom_ds: TimeSeriesDataset,
    *,
    is_released: bool,
) -> dict[str, Tensor]:
    """Run deterministic domain-only `full_ft` warmup on CPU (1 thread, f32) and return **parameters only**.

    Returning parameters only (excluding registered buffers such as `transformer.rope.inv_freq`) ensures
    that applying the warmed parameters never overwrites the trainer's device-built RoPE table.
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = copy.deepcopy(base_model_cpu).float().train()
        for p in model.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(model.parameters(), lr=_moved_warmup_lr(is_released=is_released), weight_decay=0.0)
        collator = dom_ds.collator(model.config)
        n_win = len(dom_ds)
        for step_idx in range(MOVED_WARMUP_STEPS):
            start = (step_idx * MOVED_WARMUP_BATCH_SIZE) % n_win
            idxs = [(start + j) % n_win for j in range(min(MOVED_WARMUP_BATCH_SIZE, n_win))]
            batch = collator([dom_ds[i] for i in idxs])
            batch.validate(model.config.patch_size)
            opt.zero_grad(set_to_none=True)
            compute_loss(model, batch, validate=False).loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    finally:
        torch.set_num_threads(prev_threads)


# --------------------------------------------------------------------------- distributed helpers & checks


def _propagate_rank_error(
    local_err: Exception | None,
    *,
    device: torch.device,
    world_size: int,
    rank: int,
    stage: str,
) -> None:
    """Coordinate a rank-local error across all ranks so no peer hangs in a later collective."""
    flags = torch.zeros(world_size, dtype=torch.int64, device=device)
    if local_err is not None:
        flags[rank] = 1
    dist.all_reduce(flags, op=dist.ReduceOp.SUM)
    failing = [i for i, v in enumerate(flags.tolist()) if v != 0]
    if local_err is not None:
        raise local_err
    if failing:
        raise RuntimeError(f"peer rank(s) {failing} failed during {stage!r}")


def _verify_cuda_device_binding(
    trainer: TimestralTrainer,
    *,
    rank: int,
    expected_device_index: int,
    label: str,
) -> None:
    expected = torch.device("cuda", expected_device_index)
    if trainer.device != expected or torch.cuda.current_device() != expected_device_index:
        raise RuntimeError(
            f"rank {rank}: expected {label} binding {expected}, "
            f"got trainer.device={trainer.device} current_device={torch.cuda.current_device()}"
        )
    for name, param in trainer.model.named_parameters():
        if param.device != expected:
            raise RuntimeError(f"rank {rank}: parameter {name!r} resides on {param.device}, expected {expected}")
    for name, buf in trainer.model.named_buffers():
        if buf.device != expected:
            raise RuntimeError(f"rank {rank}: buffer {name!r} resides on {buf.device}, expected {expected}")


def _apply_moved_student_params(
    trainer: TimestralTrainer,
    *,
    model_name: str,
    dom_fp: str,
    base_model_cpu: TuneTimestralModel,
    dom_ds: TimeSeriesDataset,
    is_released: bool,
    distributed: bool,
    rank: int,
    device: torch.device,
    moved_params_cache: dict[str, dict[str, Tensor]],
) -> None:
    """Transfer moved-student parameters (never buffers) to `trainer.model`, keeping peers in lockstep."""
    local_err: Exception | None = None
    if not distributed or rank == 0:
        try:
            cache_key = f"{model_name}:{dom_fp}"
            if cache_key not in moved_params_cache:
                moved_params_cache[cache_key] = _build_moved_full_ft_params(
                    base_model_cpu,
                    dom_ds,
                    is_released=is_released,
                )
            moved_params = moved_params_cache[cache_key]
            model_params = dict(trainer.model.named_parameters())
            if set(moved_params) != set(model_params):
                raise RuntimeError("moved_student parameter keys do not match trainer.model.named_parameters()")
            with torch.no_grad():
                for name, param in model_params.items():
                    param.copy_(moved_params[name].to(device=param.device, dtype=param.dtype))
        except Exception as exc:
            local_err = exc

    if distributed:
        flag = torch.tensor([1 if local_err is not None else 0], dtype=torch.int64, device=device)
        dist.broadcast(flag, src=0)
        if local_err is not None:
            raise local_err
        if int(flag.item()) != 0:
            raise RuntimeError("rank 0 failed while preparing moved_student parameters")
        with torch.no_grad():
            for _, param in sorted(trainer.model.named_parameters(), key=lambda kv: kv[0]):
                if param.device == device and param.is_contiguous():
                    dist.broadcast(param.detach(), src=0)
                else:
                    staged = param.detach().to(device=device).contiguous()
                    dist.broadcast(staged, src=0)
                    param.copy_(staged)
    elif local_err is not None:
        raise local_err


def _validate_exported_dir(
    out_dir: Path,
    *,
    trainer: TimestralTrainer,
    spec: ScenarioSpec,
    distributed: bool,
    world_size: int,
    per_rank_batch: int,
) -> tuple[bool, str]:
    """Verify exported files, metadata, strict reloadability via `load_model`, and forecast parity."""
    required_files = ["config.json", "model.safetensors", "tune_metadata.json", "LICENSE", "NOTICE"]
    if spec.mode.startswith("lora_"):
        required_files.extend(["adapter.safetensors", "adapter_config.json"])
    missing = [fname for fname in required_files if not (out_dir / fname).is_file()]
    if missing:
        return False, f"missing expected export file(s): {missing}"

    meta = json.loads((out_dir / "tune_metadata.json").read_text(encoding="utf-8"))
    if distributed:
        if meta.get("world_size") != world_size:
            return False, f"world_size={meta.get('world_size')} != {world_size}"
        if meta.get("effective_batch_size") != world_size * per_rank_batch:
            return (
                False,
                f"effective_batch_size={meta.get('effective_batch_size')} != {world_size * per_rank_batch}",
            )
        hp = meta.get("hyperparameters") or {}
        if hp.get("batch_size") != per_rank_batch:
            return False, f"hyperparameters.batch_size={hp.get('batch_size')} != {per_rank_batch}"
    else:
        if "world_size" in meta or "effective_batch_size" in meta:
            return False, "one-process export unexpectedly recorded world_size or effective_batch_size"

    if spec.uses_replay and (meta.get("replay_config") is None or meta.get("replay_source") is None):
        return False, "missing replay_config or replay_source in tune_metadata.json"

    if spec.mode.startswith("lora_"):
        reloaded_lora = LoraConfig.from_dict(json.loads((out_dir / "adapter_config.json").read_text(encoding="utf-8")))
        if trainer.lora_config is not None and reloaded_lora != trainer.lora_config:
            return False, f"reloaded adapter_config {reloaded_lora} != trainer.lora_config {trainer.lora_config}"
        adapter_tensors = safetensors_load_file(str(out_dir / "adapter.safetensors"))
        if not adapter_tensors or not all(torch.isfinite(t).all() for t in adapter_tensors.values()):
            return False, "adapter.safetensors is empty or contains non-finite tensors"

    # Reload merged checkpoint offline with strict=True via load_model and verify forecast parity
    # against the live in-memory model on a deterministic window from trainer.train_dataset.
    model_dtype = next(trainer.model.parameters()).dtype
    reloaded = load_model(out_dir, device=trainer.device, dtype=model_dtype, local_files_only=True)
    try:
        patch = trainer.model.config.patch_size
        collator = DHFDataCollator.for_model(trainer.model.config, trainer.train_dataset.context_length)
        sample_batch = collator([trainer.train_dataset[0]]).to(trainer.device)
        was_training = trainer.model.training
        trainer.model.eval()
        try:
            with torch.no_grad():
                live_out = training_forward(trainer.model, sample_batch, validate=False)
                live_raw = denormalise_quantiles(
                    trainer.model, live_out.quantiles, live_out.anchor_loc, live_out.anchor_scale
                )
                reloaded_raw = reloaded.predict_block(
                    sample_batch.context,
                    sample_batch.context_mask,
                    sample_batch.group_ids,
                    sample_batch.horizon_steps // patch,
                    sample_batch.role_ids,
                    sample_batch.known_future,
                    sample_batch.known_future_mask,
                )
        finally:
            trainer.model.train(was_training)
        if not torch.isfinite(live_raw).all() or not torch.isfinite(reloaded_raw).all():
            return False, "non-finite forecast produced during export reload parity check"
        if live_raw.shape != reloaded_raw.shape:
            return False, f"reloaded forecast shape {tuple(reloaded_raw.shape)} != live {tuple(live_raw.shape)}"
        max_abs = float((reloaded_raw.double() - live_raw.double()).abs().max().item())
        scale = max(float(live_raw.double().abs().max().item()), 1e-12)
        rel_err = max_abs / scale
        # `timestral_tune.models.loader.load_model` builds `TuneTimestralModel(config)` in float32,
        # loads `model.safetensors` into those float32 parameters (truncating float64 weights to
        # float32), and only then casts via `.to(device=target, dtype=dtype)`. Until `load_model`
        # preserves float64 precision on load, a reloaded float64 export matches the live float64
        # model only to float32 precision (~2.5e-9 to 1.65e-8 measured across `tiny_random`
        # adaptation modes), not float64 round-trip precision (~1e-10). Use 1e-6 for float64
        # (~60x margin over measured tiny_random truncation error, 100x tighter than float32's
        # 1e-4 so silent corruption or drift is still caught).
        tol = 1e-6 if model_dtype == torch.float64 else 1e-4
        if not math.isfinite(rel_err) or rel_err > tol:
            return False, f"export forecast parity mismatch: rel_err={rel_err:.2e} > tol={tol:.1e}"
    finally:
        del reloaded
    return True, ""


def _broadcast_path(path_str: str, *, rank: int, device: torch.device) -> str:
    """Broadcast a UTF-8 path string from rank 0 to all ranks using tensor collectives."""
    raw = path_str.encode("utf-8") if rank == 0 else b""
    length = torch.tensor([len(raw)], dtype=torch.int64, device=device)
    dist.broadcast(length, src=0)
    n = int(length.item())
    if rank == 0:
        buf = torch.tensor(list(raw), dtype=torch.uint8, device=device)
    else:
        buf = torch.empty(n, dtype=torch.uint8, device=device)
    dist.broadcast(buf, src=0)
    return bytes(buf.cpu().tolist()).decode("utf-8")


def _verify_export(
    trainer: TimestralTrainer,
    *,
    spec: ScenarioSpec,
    distributed: bool,
    rank: int,
    world_size: int,
    per_rank_batch: int,
    device: torch.device,
) -> tuple[bool, str]:
    """Verify `export()` on both a shared `output_dir` (real multi-rank usage) and a private non-write check.

    In distributed mode:
    1. Every rank calls `trainer.export(shared_out_dir)` with the exact same path created by rank 0;
       after a barrier, rank 0 verifies all files, `tune_metadata.json`, strict reloadability via
       `load_model(..., local_files_only=True)`, and forecast parity against `trainer.model`, while
       all ranks verify no error occurred and the returned path matches.
    2. Each non-zero rank additionally calls `trainer.export(private_nonexistent_dir)` and asserts
       that directory is NOT created, strictly proving non-zero ranks perform zero disk writes.

    Never raises: catches unexpected I/O or serialisation errors while preserving collective lockstep
    so all ranks still reach `_gather_distributed_checks`.
    """
    if not distributed:
        try:
            with tempfile.TemporaryDirectory(prefix="tt_stage3_exp_1p_") as tmpdir:
                out_dir = Path(tmpdir) / "ckpt"
                returned = trainer.export(out_dir)
                if returned != out_dir:
                    return False, f"returned {returned} != {out_dir}"
                return _validate_exported_dir(
                    out_dir,
                    trainer=trainer,
                    spec=spec,
                    distributed=False,
                    world_size=world_size,
                    per_rank_batch=per_rank_batch,
                )
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    # Distributed mode: keep collectives (path broadcast + post-export/pre-cleanup barriers) outside
    # the local try/except so every rank remains strictly in lockstep even if one rank fails.
    shared_ctx = tempfile.TemporaryDirectory(prefix="tt_stage3_exp_shared_") if rank == 0 else contextlib.nullcontext()
    ok = True
    err = ""
    with shared_ctx as shared_tmp:
        shared_dir_str = str(Path(shared_tmp) / "ckpt") if rank == 0 else ""
        shared_out_dir = Path(_broadcast_path(shared_dir_str, rank=rank, device=device))

        # 1. Shared output_dir (exact real multi-rank usage).
        try:
            returned = trainer.export(shared_out_dir)
            if returned != shared_out_dir:
                ok, err = False, f"shared export returned {returned} != {shared_out_dir}"
        except Exception as exc:
            ok, err = False, f"shared export raised {type(exc).__name__}: {exc}"

        # Synchronise so rank 0 has completed writing before verification and peers have returned.
        dist.all_reduce(torch.zeros(1, dtype=torch.int64, device=device), op=dist.ReduceOp.SUM)

        if rank == 0 and ok:
            try:
                ok, err = _validate_exported_dir(
                    shared_out_dir,
                    trainer=trainer,
                    spec=spec,
                    distributed=True,
                    world_size=world_size,
                    per_rank_batch=per_rank_batch,
                )
            except Exception as exc:
                ok, err = False, f"shared export validation raised {type(exc).__name__}: {exc}"

        # 2. Private non-write proof on non-zero ranks (proves peers never create output_dir).
        if rank != 0 and ok:
            try:
                with tempfile.TemporaryDirectory(prefix=f"tt_stage3_exp_r{rank}_") as priv_tmp:
                    priv_out_dir = Path(priv_tmp) / "ckpt"
                    returned_priv = trainer.export(priv_out_dir)
                    if returned_priv != priv_out_dir:
                        ok, err = False, f"private export returned {returned_priv} != {priv_out_dir}"
                    elif priv_out_dir.exists():
                        ok, err = False, f"rank {rank} created private export directory"
            except Exception as exc:
                ok, err = False, f"private export check raised {type(exc).__name__}: {exc}"

        # Keep shared_tmp alive on rank 0 until all ranks finish before context exit deletes it.
        dist.all_reduce(torch.zeros(1, dtype=torch.int64, device=device), op=dist.ReduceOp.SUM)
    return ok, err


def _gather_distributed_checks(
    model: TuneTimestralModel,
    result: TrainingResult,
    local: dict[str, Any],
    *,
    world_size: int,
    rank: int,
) -> dict[str, Any]:
    """Collect cross-rank state equality, max-abs diff, and max-over-ranks telemetry via collectives.

    What is compared across ranks:
    - `result_eq`: SHA-256 of all ten `TrainingResult` fields (with exact `float.hex()` encodings)
      broadcast from rank 0 as 4 signed `int64` words and checked for unanimity via `all_reduce(SUM)`.
    - `xrank_exact`: SHA-256 of every parameter in `sorted(model.named_parameters())` AND every
      buffer in `sorted(model.named_buffers())` (hashing `(kind, name, shape, dtype, raw uint8 bytes)`),
      broadcast from rank 0 as 4 signed `int64` words and checked for unanimity via `all_reduce(SUM)`.
    - `xrank_max_abs`: exact elementwise `max(abs(rank_r.double() - rank_0.double()))` over every
      parameter in `sorted(model.named_parameters())` AND every buffer in `sorted(model.named_buffers())`,
      computed by broadcasting each tensor from rank 0 and reducing the per-rank maximum via `all_reduce(MAX)`.
    - Timing/memory/compile counters (`wall_s`, `steady_step_ms`, `tail_s`, `val_s`, `peak_vram_mib`,
      `graphs_s1`, `graphs_end`, `graph_bumps_post_s1`): maximum across ranks via `all_reduce(MAX)`.
    """
    device: torch.device = local["device"]

    # 1. Exact TrainingResult equality via 4x int64 SHA-256 digest words.
    res_words = _encode_digest_words(_hash_training_result(result))
    res_buf = torch.tensor(res_words, dtype=torch.int64, device=device)
    dist.broadcast(res_buf, src=0)
    res_match = torch.tensor([1 if res_buf.tolist() == res_words else 0], dtype=torch.int64, device=device)
    dist.all_reduce(res_match, op=dist.ReduceOp.SUM)
    result_eq = int(res_match.item()) == world_size

    # 2. Exact full model state (all parameters + all buffers) bitwise equality.
    state_words = _encode_digest_words(_hash_model_state(model))
    state_buf = torch.tensor(state_words, dtype=torch.int64, device=device)
    dist.broadcast(state_buf, src=0)
    state_match = torch.tensor([1 if state_buf.tolist() == state_words else 0], dtype=torch.int64, device=device)
    dist.all_reduce(state_match, op=dist.ReduceOp.SUM)
    xrank_exact = int(state_match.item()) == world_size

    # 3. Cross-rank max-abs difference across every parameter and every buffer.
    local_max_abs = 0.0
    local_nonfinite_diff = 0.0
    state_tensors = [
        *[p for _, p in sorted(model.named_parameters(), key=lambda kv: kv[0])],
        *[b for _, b in sorted(model.named_buffers(), key=lambda kv: kv[0])],
    ]
    with torch.no_grad():
        for tensor in state_tensors:
            if tensor.numel() == 0:
                continue
            src_tensor = tensor.detach().to(device=device).contiguous()
            # Cast bool buffers to uint8 for NCCL broadcast compatibility.
            bcast_src = src_tensor.to(torch.uint8) if src_tensor.dtype == torch.bool else src_tensor
            buf = bcast_src.clone() if rank == 0 else torch.empty_like(bcast_src)
            dist.broadcast(buf, src=0)
            if rank != 0:
                diff = float((bcast_src.double() - buf.double()).abs().max().item())
                if math.isnan(diff):
                    local_nonfinite_diff = 1.0
                elif diff > local_max_abs:
                    local_max_abs = diff

    # 4. Reduce max-over-ranks telemetry + export failure indicator in one float64 collective.
    steady_val = -1.0 if local["steady_step_ms"] is None else float(local["steady_step_ms"])
    metrics_max = torch.tensor(
        [
            local_max_abs,
            local_nonfinite_diff,
            float(local["wall_s"]),
            steady_val,
            float(local["tail_s"]),
            float(local["val_s"]),
            float(local["peak_vram_mib"]),
            float(local["graphs_s1"]),
            float(local["graphs_end"]),
            float(local["graph_bumps_post_s1"]),
            0.0 if local["export_ok"] else 1.0,
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(metrics_max, op=dist.ReduceOp.MAX)
    (
        xrank_max_abs,
        nonfinite_diff_any,
        wall_max_s,
        step_max_ms_raw,
        tail_max_s,
        val_max_s,
        vram_max_mib,
        graphs_s1_max,
        graphs_end_max,
        graph_bumps_max,
        export_any_failed,
    ) = metrics_max.tolist()
    if nonfinite_diff_any > 0.0:
        xrank_max_abs = float("nan")
    step_max_ms: float | None = None if step_max_ms_raw < 0.0 else step_max_ms_raw

    export_pass_count = torch.tensor([1 if local["export_ok"] else 0], dtype=torch.int64, device=device)
    dist.all_reduce(export_pass_count, op=dist.ReduceOp.SUM)

    return {
        "result_eq": result_eq,
        "xrank_exact": xrank_exact,
        "xrank_max_abs": xrank_max_abs,
        "wall_max_s": wall_max_s,
        "step_max_ms": step_max_ms,
        "tail_max_s": tail_max_s,
        "val_max_s": val_max_s,
        "vram_max_mib": vram_max_mib,
        "graphs_s1_max": int(round(graphs_s1_max)),
        "graphs_end_max": int(round(graphs_end_max)),
        "graph_bumps_max": int(round(graph_bumps_max)),
        "export_all_ok": export_any_failed == 0.0,
        "export_pass_count": int(export_pass_count.item()),
    }


def _timed_fit(
    trainer: TimestralTrainer,
    *,
    spec: ScenarioSpec,
    device: torch.device,
    distributed: bool,
) -> tuple[TrainingResult, float, float | None, float, float, float, int, int, int]:
    """Execute `trainer.fit()` with a pre-`t0` barrier, symmetric per-step wall timing, and validation timing.

    Measures full per-step wall time identically in one-process and multi-process modes: each step
    interval spans from the end of `strategy.on_step_end(k-1)` to the end of `strategy.on_step_end(k)`
    (both with CUDA sync) minus any `strategy.evaluate_val()` + `strategy.on_evaluate()` duration
    executed between those two steps. `steady_step_ms` averages steps `3..K` (`step_intervals[1:]`).
    """
    step_end_times: list[float] = []
    step_intervals: list[float] = []
    val_since_last_step: list[float] = [0.0]
    val_total_s: list[float] = [0.0]
    step_graph_counts: list[int] = []

    orig_make_strategy = trainer._make_fit_strategy

    def _timed_make_strategy() -> Any:
        strategy = orig_make_strategy()
        orig_on_step_end = strategy.on_step_end
        orig_eval = strategy.evaluate_val
        orig_on_eval = strategy.on_evaluate

        def _timed_on_step_end(step: int, loss: float, learning_rate: float) -> float:
            ret = orig_on_step_end(step, loss, learning_rate)
            _cuda_sync_if_needed(device)
            t_now = time.perf_counter()
            if step_end_times:
                raw_dt = t_now - step_end_times[-1]
                step_intervals.append(max(0.0, raw_dt - val_since_last_step[0]))
            val_since_last_step[0] = 0.0
            step_end_times.append(t_now)
            if spec.compile_model:
                step_graph_counts.append(_dynamo_unique_graphs())
            return ret

        def _timed_eval() -> Any:
            _cuda_sync_if_needed(device)
            vt0 = time.perf_counter()
            try:
                return orig_eval()
            finally:
                _cuda_sync_if_needed(device)
                dt = time.perf_counter() - vt0
                val_total_s[0] += dt
                val_since_last_step[0] += dt

        def _timed_on_eval(step: int, result: Any, bookkeeper: Any) -> Any:
            _cuda_sync_if_needed(device)
            vt0 = time.perf_counter()
            try:
                return orig_on_eval(step, result, bookkeeper)
            finally:
                _cuda_sync_if_needed(device)
                dt = time.perf_counter() - vt0
                val_total_s[0] += dt
                val_since_last_step[0] += dt

        strategy.on_step_end = _timed_on_step_end
        strategy.evaluate_val = _timed_eval
        strategy.on_evaluate = _timed_on_eval
        return strategy

    trainer._make_fit_strategy = _timed_make_strategy  # type: ignore[method-assign]
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        if distributed:
            # 1-scalar all_reduce on `device` synchronises all ranks immediately before t0 without
            # relying on `dist.barrier(device_ids=...)` warnings on NCCL.
            dist.all_reduce(torch.zeros(1, dtype=torch.int64, device=device), op=dist.ReduceOp.SUM)
        _cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        result = trainer.fit()
        _cuda_sync_if_needed(device)
        t1 = time.perf_counter()
    finally:
        trainer._make_fit_strategy = orig_make_strategy  # type: ignore[method-assign]

    wall_s = t1 - t0
    # step_intervals[0] is step 2 (1->2); step_intervals[1:] covers steps 3..K (strictly after steps 1-2).
    steady_intervals = step_intervals[1:]
    steady_step_ms: float | None = sum(steady_intervals) / len(steady_intervals) * 1000.0 if steady_intervals else None
    tail_s = (t1 - step_end_times[-1]) if step_end_times else wall_s
    peak_vram_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0) if device.type == "cuda" else 0.0
    if spec.compile_model:
        graphs_s1 = step_graph_counts[0] if step_graph_counts else 0
        graphs_end = _dynamo_unique_graphs()
        graph_bumps_post_s1 = sum(
            1 for prev, cur in zip(step_graph_counts, step_graph_counts[1:], strict=False) if cur > prev
        )
        if step_graph_counts and graphs_end > step_graph_counts[-1]:
            graph_bumps_post_s1 += 1
    else:
        graphs_s1 = 0
        graphs_end = 0
        graph_bumps_post_s1 = 0
    return (
        result,
        wall_s,
        steady_step_ms,
        tail_s,
        val_total_s[0],
        peak_vram_mib,
        graphs_s1,
        graphs_end,
        graph_bumps_post_s1,
    )


def _run_scenario(
    spec: ScenarioSpec,
    model_name: str,
    base_model_cpu: TuneTimestralModel,
    *,
    world_size: int,
    per_rank_batch: int,
    per_rank_replay_batch: int,
    steps: int,
    seed: int,
    config_device: str,
    distributed: bool,
    rank: int,
    expected_cuda_index: int | None,
    binding_label: str,
    moved_params_cache: dict[str, dict[str, Tensor]],
) -> dict[str, Any]:
    """Execute one scenario in either 1-process reference mode or multi-rank mode."""
    if spec.num_workers > 0:
        check_tmpdir_for_workers()

    is_released = model_name != "tiny_random"
    dom_ds, val_ds, rep_ds = _build_datasets_for_scenario(
        spec,
        world_size=world_size,
        per_rank_batch=per_rank_batch,
        per_rank_replay_batch=per_rank_replay_batch,
        seed=seed,
    )
    dom_fp = dataset_fingerprint(dom_ds)
    val_fp = None if val_ds is None else dataset_fingerprint(val_ds)
    rep_fp = None if rep_ds is None else dataset_fingerprint(rep_ds)

    cfg, lora_cfg, rcfg = _build_trainer_configs(
        spec,
        is_released=is_released,
        world_size=world_size,
        per_rank_batch=per_rank_batch,
        per_rank_replay_batch=per_rank_replay_batch,
        steps=steps,
        seed=seed,
        config_device=config_device,
        distributed=distributed,
    )
    cfg_digest = _scenario_config_digest(
        cfg,
        lora_cfg,
        rcfg,
        spec=spec,
        per_rank_batch=per_rank_batch,
        per_rank_replay_batch=per_rank_replay_batch,
        is_released=is_released,
    )

    if spec.compile_model:
        _reset_dynamo()

    model = copy.deepcopy(base_model_cpu)
    if spec.dtype == "float64":
        model = model.double()

    ctx_mgr = float64_reference() if spec.dtype == "float64" else contextlib.nullcontext()
    trainer: TimestralTrainer | None = None
    device: torch.device | None = None
    try:
        with ctx_mgr:
            trainer = TimestralTrainer(
                cfg,
                dom_ds,
                val_ds,
                lora_config=lora_cfg,
                replay_config=rcfg,
                replay_dataset=rep_ds,
                model=model,
            )
            device = trainer.device
            if distributed and expected_cuda_index is not None and device.type == "cuda":
                bind_err: Exception | None = None
                try:
                    _verify_cuda_device_binding(
                        trainer,
                        rank=rank,
                        expected_device_index=expected_cuda_index,
                        label=binding_label,
                    )
                except Exception as exc:
                    bind_err = exc
                _propagate_rank_error(
                    bind_err,
                    device=device,
                    world_size=world_size,
                    rank=rank,
                    stage="cuda_device_binding",
                )

            if spec.moved_student:
                _apply_moved_student_params(
                    trainer,
                    model_name=model_name,
                    dom_fp=dom_fp,
                    base_model_cpu=base_model_cpu,
                    dom_ds=dom_ds,
                    is_released=is_released,
                    distributed=distributed,
                    rank=rank,
                    device=device,
                    moved_params_cache=moved_params_cache,
                )

            init_model_sha256 = _hash_model_state(trainer.model).hex()
            frozen_base_sha256 = (
                _hash_model_state(trainer._frozen_base).hex() if trainer._frozen_base is not None else None
            )
            init_trainable_native = _flat_trainable_native(trainer).clone()
            init_trainable_sha256 = _tensor_sha256(init_trainable_native)

            (
                result,
                wall_s,
                steady_step_ms,
                tail_s,
                val_s,
                peak_vram_mib,
                graphs_s1,
                graphs_end,
                graph_bumps_post_s1,
            ) = _timed_fit(trainer, spec=spec, device=device, distributed=distributed)

            export_ok = True
            export_err = ""
            if spec.test_export:
                export_ok, export_err = _verify_export(
                    trainer,
                    spec=spec,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                    per_rank_batch=per_rank_batch,
                    device=device,
                )

            final_trainable_native = _flat_trainable_native(trainer).clone()
            final_trainable_sha256 = _tensor_sha256(final_trainable_native)
            final_all_finite = bool(torch.isfinite(final_trainable_native).all().item())

            summary: dict[str, Any] = {
                "result": result,
                "config_digest": cfg_digest,
                "init_model_sha256": init_model_sha256,
                "frozen_base_sha256": frozen_base_sha256,
                "init_trainable_native": init_trainable_native,
                "init_trainable_sha256": init_trainable_sha256,
                "final_trainable_native": final_trainable_native,
                "final_trainable_sha256": final_trainable_sha256,
                "final_all_finite": final_all_finite,
                "dom_fp": dom_fp,
                "val_fp": val_fp,
                "rep_fp": rep_fp,
                "wall_s": wall_s,
                "steady_step_ms": steady_step_ms,
                "tail_s": tail_s,
                "val_s": val_s,
                "peak_vram_mib": peak_vram_mib,
                "graphs_s1": graphs_s1,
                "graphs_end": graphs_end,
                "graph_bumps_post_s1": graph_bumps_post_s1,
                "export_ok": export_ok,
                "export_err": export_err,
                "device": device,
            }

            if distributed:
                summary.update(
                    _gather_distributed_checks(trainer.model, result, summary, world_size=world_size, rank=rank)
                )
    finally:
        del trainer, model
        if device is not None and device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    return summary


# --------------------------------------------------------------------------- reference archive I/O & validation


def _expected_config_digests(
    models: Sequence[tuple[str, TuneTimestralModel]],
    scenarios: Sequence[ScenarioSpec],
    *,
    is_cuda: bool,
    world_size: int,
    per_rank_batch: int,
    per_rank_replay_batch: int,
    steps: int,
    seed: int,
) -> dict[str, str]:
    """Precompute canonical per-(model, scenario) resolved config digests without touching GPU."""
    digests: dict[str, str] = {}
    for model_name, _ in models:
        is_released = model_name != "tiny_random"
        for spec in scenarios:
            if not spec.compare_ref or _skip_reason(spec, is_cuda=is_cuda) is not None:
                continue
            cfg, lora_cfg, rcfg = _build_trainer_configs(
                spec,
                is_released=is_released,
                world_size=world_size,
                per_rank_batch=per_rank_batch,
                per_rank_replay_batch=per_rank_replay_batch,
                steps=steps,
                seed=seed,
                config_device="cpu",
                distributed=True,
            )
            digests[f"{model_name}:{spec.name}"] = _scenario_config_digest(
                cfg,
                lora_cfg,
                rcfg,
                spec=spec,
                per_rank_batch=per_rank_batch,
                per_rank_replay_batch=per_rank_replay_batch,
                is_released=is_released,
            )
    return digests


def _make_run_spec(
    *,
    world_size: int,
    device_type: str,
    models: Sequence[tuple[str, TuneTimestralModel]],
    scenarios: Sequence[ScenarioSpec],
    seed: int,
    per_rank_batch: int,
    per_rank_replay_batch: int,
    steps: int,
    cpu_threads: int | None,
    reference_mode: ReferenceMode,
    entry_keys: Sequence[str],
) -> dict[str, Any]:
    is_cuda = device_type == "cuda"
    executed_scenarios = [s for s in scenarios if s.compare_ref and _skip_reason(s, is_cuda=is_cuda) is None]
    model_digests = {name: _hash_model_state(base_model).hex() for name, base_model in models}
    cfg_digests = _expected_config_digests(
        models,
        executed_scenarios,
        is_cuda=is_cuda,
        world_size=world_size,
        per_rank_batch=per_rank_batch,
        per_rank_replay_batch=per_rank_replay_batch,
        steps=steps,
        seed=seed,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_mode": reference_mode,
        "sketch_dim": SKETCH_DIM if reference_mode in ("auto", "sketch") else None,
        "sketch_chunk": SKETCH_CHUNK if reference_mode in ("auto", "sketch") else None,
        "auto_exact_f64_max_numel": AUTO_EXACT_F64_MAX_NUMEL if reference_mode == "auto" else None,
        "world_size": world_size,
        "device_type": device_type,
        "models": [name for name, _ in models],
        "model_digests": model_digests,
        "scenarios": [s.to_meta() for s in executed_scenarios],
        "scenario_config_digests": cfg_digests,
        "entry_keys": sorted(entry_keys),
        "seed": seed,
        "per_rank_batch_size": per_rank_batch,
        "per_rank_replay_batch_size": per_rank_replay_batch,
        "steps": steps,
        "ctx_patches": CTX_PATCHES,
        "hor_patches": HOR_PATCHES,
        "dom_horizon": DOM_HORIZON,
        "rep_horizon": REP_HORIZON,
        "cpu_threads": cpu_threads,
    }


class _StreamingReferenceWriter:
    """Writes a pickle-free zip reference archive (`spec.json`, `entries.json`, `tensors/<idx>.bin`).

    Streams each scenario's tensor payload directly to disk as it finishes so peak memory stays
    O(1 scenario) even for released checkpoints under `--reference-mode exact`.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_path = path.with_suffix(path.suffix + ".tmp")
        self._zf = zipfile.ZipFile(self._tmp_path, mode="w", compression=zipfile.ZIP_STORED)
        self._entries: dict[str, dict[str, Any]] = {}
        self._next_idx = 0
        self._closed = False

    @property
    def entry_keys(self) -> list[str]:
        return sorted(self._entries.keys())

    def add_entry(self, key: str, metadata: dict[str, Any], tensor: Tensor) -> None:
        member = f"tensors/{self._next_idx:04d}.bin"
        self._next_idx += 1
        raw, dtype_name, shape = _tensor_to_bytes(tensor)
        self._zf.writestr(member, raw)
        entry = dict(metadata)
        entry["tensor_member"] = member
        entry["tensor_dtype"] = dtype_name
        entry["tensor_shape"] = shape
        self._entries[key] = entry

    def commit(self, *, spec: dict[str, Any], env: dict[str, Any]) -> None:
        spec_bytes = json.dumps({"spec": spec, "environment": env}, sort_keys=True, indent=2).encode("utf-8")
        entries_bytes = json.dumps(self._entries, sort_keys=True, indent=2).encode("utf-8")
        self._zf.writestr("spec.json", spec_bytes)
        self._zf.writestr("entries.json", entries_bytes)
        self._zf.close()
        self._closed = True
        os.replace(self._tmp_path, self._path)

    def abort(self) -> None:
        if not self._closed:
            self._zf.close()
            self._closed = True
        if self._tmp_path.exists():
            self._tmp_path.unlink()


class _ReferenceReader:
    """Lazy reader for a zip reference archive; loads one scenario tensor at a time on rank 0."""

    def __init__(self, path: Path) -> None:
        self._zf = zipfile.ZipFile(path, mode="r")
        try:
            self.meta: dict[str, Any] = json.loads(self._zf.read("spec.json").decode("utf-8"))
            self.entries: dict[str, dict[str, Any]] = json.loads(self._zf.read("entries.json").decode("utf-8"))
        except BaseException:
            self._zf.close()
            raise

    def load_tensor(self, entry: dict[str, Any]) -> Tensor:
        raw = self._zf.read(entry["tensor_member"])
        return _bytes_to_tensor(raw, entry["tensor_dtype"], entry["tensor_shape"])

    def close(self) -> None:
        self._zf.close()


def _read_reference_metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """CRC-verify the zip archive and JSON-validate `spec.json`, `entries.json`, and member sizes pre-init."""
    if not zipfile.is_zipfile(path):
        raise ValueError("reference file is not a valid zip archive")
    with zipfile.ZipFile(path, mode="r") as zf:
        try:
            bad_member = zf.testzip()
        except Exception as exc:
            raise ValueError(f"reference archive failed CRC integrity check: {exc}") from exc
        if bad_member is not None:
            raise ValueError(f"reference archive member {bad_member!r} failed CRC integrity check")
        member_sizes = {info.filename: info.file_size for info in zf.infolist()}
        if "spec.json" not in member_sizes or "entries.json" not in member_sizes:
            raise ValueError("reference archive is malformed (missing 'spec.json' or 'entries.json')")
        try:
            meta = json.loads(zf.read("spec.json").decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"reference archive spec.json is malformed: {exc}") from exc
        try:
            entries = json.loads(zf.read("entries.json").decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"reference archive entries.json is malformed: {exc}") from exc
        if not isinstance(entries, dict):
            raise ValueError("reference archive entries.json is malformed (expected JSON object)")
        return meta, entries, member_sizes


def _validate_reference_spec(
    meta: dict[str, Any],
    entries: dict[str, Any],
    member_sizes: Mapping[str, int],
    *,
    world_size: int,
    device_type: str,
    models: Sequence[tuple[str, TuneTimestralModel]],
    scenarios: Sequence[ScenarioSpec],
    seed: int,
    per_rank_batch: int,
    per_rank_replay_batch: int,
    steps: int,
    cpu_threads: int | None,
    allow_env_mismatch: bool = False,
) -> tuple[ReferenceMode, list[str]]:
    """Refuse to compare if any run, model, config, member, or environment field mismatches.

    Returns `(ref_mode, env_warnings)`. Environment/script-revision differences are refused unless
    `allow_env_mismatch=True`, in which case they are returned in `env_warnings` for rank 0 to print.
    Missing/corrupt zip members, schema, model, config, and scenario mismatches are always refused.
    """
    if not isinstance(meta, dict) or "spec" not in meta or not isinstance(meta["spec"], dict):
        raise ValueError("reference metadata is malformed (missing 'spec')")
    ref_spec = meta["spec"]
    ref_env = meta.get("environment")
    diffs: list[str] = []
    env_diffs: list[str] = []

    if ref_spec.get("schema_version") != SCHEMA_VERSION:
        diffs.append(f"schema_version: reference={ref_spec.get('schema_version')!r}, current={SCHEMA_VERSION!r}")

    ref_mode = ref_spec.get("reference_mode")
    if ref_mode not in ("auto", "sketch", "exact"):
        diffs.append(f"reference_mode: unsupported value {ref_mode!r}")
    else:
        if ref_mode in ("auto", "sketch"):
            if ref_spec.get("sketch_dim") != SKETCH_DIM:
                diffs.append(f"sketch_dim: reference={ref_spec.get('sketch_dim')!r}, current={SKETCH_DIM!r}")
            if ref_spec.get("sketch_chunk") != SKETCH_CHUNK:
                diffs.append(f"sketch_chunk: reference={ref_spec.get('sketch_chunk')!r}, current={SKETCH_CHUNK!r}")
        if ref_mode == "auto" and ref_spec.get("auto_exact_f64_max_numel") != AUTO_EXACT_F64_MAX_NUMEL:
            diffs.append(
                f"auto_exact_f64_max_numel: reference={ref_spec.get('auto_exact_f64_max_numel')!r}, "
                f"current={AUTO_EXACT_F64_MAX_NUMEL!r}"
            )

    scalar_checks: tuple[tuple[str, Any], ...] = (
        ("world_size", world_size),
        ("device_type", device_type),
        ("seed", seed),
        ("per_rank_batch_size", per_rank_batch),
        ("per_rank_replay_batch_size", per_rank_replay_batch),
        ("steps", steps),
        ("ctx_patches", CTX_PATCHES),
        ("hor_patches", HOR_PATCHES),
        ("dom_horizon", DOM_HORIZON),
        ("rep_horizon", REP_HORIZON),
    )
    if device_type == "cpu":
        scalar_checks = (*scalar_checks, ("cpu_threads", cpu_threads))
    for key, expected in scalar_checks:
        actual = ref_spec.get(key)
        if actual != expected:
            diffs.append(f"{key}: reference={actual!r}, current={expected!r}")

    # Environment and script-revision checks (overridable only via --allow-env-mismatch).
    if not isinstance(ref_env, dict):
        env_diffs.append("environment metadata missing from reference spec.json")
    else:
        if ref_env.get("torch_version") != torch.__version__:
            env_diffs.append(
                f"torch_version: reference={ref_env.get('torch_version')!r}, current={torch.__version__!r}"
            )
        if device_type == "cuda" and ref_env.get("cuda_version") != torch.version.cuda:
            env_diffs.append(f"cuda_version: reference={ref_env.get('cuda_version')!r}, current={torch.version.cuda!r}")
        cur_script_sha = _current_script_sha256()
        ref_script_sha = ref_env.get("script_sha256")
        if ref_script_sha != cur_script_sha:
            env_diffs.append(f"script_sha256: reference={str(ref_script_sha)[:12]!r}, current={cur_script_sha[:12]!r}")
        cur_git_head = git_head_commit(Path(__file__).resolve().parent)
        ref_git_head = ref_env.get("git_head")
        if ref_git_head is not None and cur_git_head is not None and ref_git_head != cur_git_head:
            env_diffs.append(f"git_head: reference={ref_git_head[:12]!r}, current={cur_git_head[:12]!r}")

    model_names = [name for name, _ in models]
    ref_models = set(ref_spec.get("models", []))
    missing_models = [m for m in model_names if m not in ref_models]
    if missing_models:
        diffs.append(f"models: missing {missing_models} (reference has {sorted(ref_models)})")

    ref_model_digests = ref_spec.get("model_digests")
    if not isinstance(ref_model_digests, dict):
        diffs.append("model_digests: missing from reference spec")
    else:
        for model_name, base_model in models:
            if model_name not in ref_models:
                continue
            cur_digest = _hash_model_state(base_model).hex()
            ref_digest = ref_model_digests.get(model_name)
            if ref_digest != cur_digest:
                diffs.append(
                    f"base model state {model_name!r} differs (ref={str(ref_digest)[:12]}, cur={cur_digest[:12]})"
                )

    is_cuda = device_type == "cuda"
    ref_scen_by_name = {entry["name"]: entry for entry in ref_spec.get("scenarios", [])}
    ref_cfg_digests = ref_spec.get("scenario_config_digests")
    cur_cfg_digests = _expected_config_digests(
        models,
        scenarios,
        is_cuda=is_cuda,
        world_size=world_size,
        per_rank_batch=per_rank_batch,
        per_rank_replay_batch=per_rank_replay_batch,
        steps=steps,
        seed=seed,
    )
    for spec in scenarios:
        if not spec.compare_ref or _skip_reason(spec, is_cuda=is_cuda) is not None:
            continue
        if spec.name not in ref_scen_by_name:
            diffs.append(f"scenario {spec.name!r}: missing from reference spec (possibly skipped when ref was built)")
            continue
        expected_meta = spec.to_meta()
        actual_meta = ref_scen_by_name[spec.name]
        if actual_meta != expected_meta:
            all_keys = sorted(set(actual_meta) | set(expected_meta))
            field_diffs = [
                f"{k} (ref={actual_meta.get(k)!r} vs cur={expected_meta.get(k)!r})"
                for k in all_keys
                if actual_meta.get(k) != expected_meta.get(k)
            ]
            diffs.append(f"scenario {spec.name!r} differs in: {', '.join(field_diffs)}")
        for model_name in model_names:
            if model_name not in ref_models:
                continue
            key = f"{model_name}:{spec.name}"
            if key not in entries:
                diffs.append(f"entry {key!r}: missing from reference entries.json")
                continue
            entry = entries[key]
            if not isinstance(entry, dict):
                diffs.append(f"entry {key!r}: malformed entry record")
                continue
            member = entry.get("tensor_member")
            dtype_name = entry.get("tensor_dtype")
            shape = entry.get("tensor_shape")
            if not isinstance(member, str) or member not in member_sizes:
                diffs.append(f"entry {key!r}: tensor member {member!r} missing from zip archive")
            elif dtype_name not in _DTYPE_NBYTES or not isinstance(shape, list):
                diffs.append(f"entry {key!r}: invalid tensor_dtype={dtype_name!r} or tensor_shape={shape!r}")
            else:
                expected_bytes = math.prod(int(d) for d in shape) * _DTYPE_NBYTES[dtype_name]
                actual_bytes = member_sizes[member]
                if actual_bytes != expected_bytes:
                    diffs.append(
                        f"entry {key!r}: tensor member {member!r} has {actual_bytes} bytes, expected {expected_bytes}"
                    )
            cur_cd = cur_cfg_digests[key]
            ref_cd = ref_cfg_digests.get(key) if isinstance(ref_cfg_digests, dict) else None
            if ref_cd != cur_cd:
                diffs.append(f"resolved config for {key!r} differs (ref={str(ref_cd)[:12]}, cur={cur_cd[:12]})")

    if env_diffs and not allow_env_mismatch:
        diffs.extend(f"{d} (pass --allow-env-mismatch to override)" for d in env_diffs)

    if diffs:
        raise ValueError("; ".join(diffs))
    assert ref_mode in ("auto", "sketch", "exact")
    return ref_mode, (env_diffs if allow_env_mismatch else [])


# --------------------------------------------------------------------------- comparison & row formatting


def _compare_to_reference(
    out: dict[str, Any],
    reader: _ReferenceReader | None,
    *,
    model_name: str,
    spec: ScenarioSpec,
    seed: int,
) -> tuple[float | None, float | None, float | None, float | None, list[str]]:
    if reader is None or not spec.compare_ref:
        return None, None, None, None, []

    key = f"{model_name}:{spec.name}"
    ref = reader.entries.get(key)
    if ref is None:
        raise ValueError(f"{key}: missing from reference archive")

    # Guard against complete initial model state, frozen teacher, config, or dataset drift.
    if ref.get("init_model_sha256") != out["init_model_sha256"]:
        raise ValueError(f"{key}: initial full model state (params+buffers) SHA-256 differs from reference")
    if ref.get("init_trainable_sha256") != out["init_trainable_sha256"]:
        raise ValueError(f"{key}: initial trainable weights SHA-256 differs from reference")
    if ref.get("frozen_base_sha256") is not None and out["frozen_base_sha256"] is not None:
        if ref.get("frozen_base_sha256") != out["frozen_base_sha256"]:
            raise ValueError(f"{key}: frozen_base teacher state SHA-256 differs from reference")
    if ref.get("config_digest") != out["config_digest"]:
        raise ValueError(f"{key}: resolved TrainingConfig/LoraConfig/ReplayConfig digest differs from reference")
    for fp_key in ("dom_fp", "val_fp", "rep_fp"):
        if ref.get(fp_key) != out[fp_key]:
            raise ValueError(f"{key}: dataset fingerprint {fp_key!r} differs from reference")

    init_native: Tensor = out["init_trainable_native"]
    final_native: Tensor = out["final_trainable_native"]
    if int(final_native.numel()) != int(ref["numel"]):
        raise ValueError(f"{key}: trainable numel {final_native.numel()} != reference {ref['numel']}")

    extra_notes: list[str] = []
    storage_kind = ref["storage_kind"]
    got_finite = bool(out.get("final_all_finite", bool(torch.isfinite(final_native).all().item())))
    ref_finite = bool(ref.get("final_all_finite", True))

    if not got_finite or not ref_finite:
        # Never report 0.0 for bitwise-identical NaN/Inf vectors.
        param_l2 = float("nan")
        upd_l2 = float("nan")
        extra_notes.append("NONFINITE_PARAMS")
    elif out["final_trainable_sha256"] == ref["final_trainable_sha256"]:
        param_l2 = 0.0
        upd_l2 = 0.0
    else:
        ref_tensor = reader.load_tensor(ref)
        if storage_kind == "exact_final":
            init_f64 = init_native.double()
            got_final_f64 = final_native.double()
            ref_final_f64 = ref_tensor.double()
            param_l2 = rel(got_final_f64, ref_final_f64)
            upd_l2 = rel(got_final_f64 - init_f64, ref_final_f64 - init_f64)
        elif storage_kind == "exact_delta":
            got_delta_f64 = final_native.double() - init_native.double()
            diff_norm = float((got_delta_f64 - ref_tensor.double()).norm().item())
            param_l2 = diff_norm / max(float(ref["final_norm_f64"]), 1e-300)
            upd_l2 = diff_norm / max(float(ref["update_norm_f64"]), 1e-300)
        elif storage_kind == "count_sketch":
            sketch_dim = int(ref_tensor.numel())
            got_sketch, _, _, _ = _count_sketch_delta(
                init_native,
                final_native,
                sketch_dim=sketch_dim,
                seed=_sketch_seed(seed, model_name, spec.name),
            )
            diff_norm = float((got_sketch - ref_tensor.double()).norm().item())
            param_l2 = diff_norm / max(float(ref["final_norm_f64"]), 1e-300)
            upd_l2 = diff_norm / max(float(ref["update_norm_f64"]), 1e-300)
            if diff_norm == 0.0:
                extra_notes.append("sketch_underflow(sha_differs)")
            else:
                floor_rel = _sketch_floor_rel(int(final_native.numel()), sketch_dim)
                if upd_l2 <= 10.0 * floor_rel:
                    extra_notes.append(f"sketch_bound(floor~{floor_rel:.1e})")
        else:
            raise ValueError(f"{key}: unknown reference storage_kind {storage_kind!r}")

    res: TrainingResult = out["result"]
    ref_hist = ref["train_history"]
    loss_rel = _max_history_rel_nan_safe(res.train_history, ref_hist)
    if len(res.train_history) != len(ref_hist):
        extra_notes.append(f"ref_steps={ref['steps']}")

    val_rel: float | None = None
    if spec.with_val:
        val_rel = _max_history_rel_nan_safe(res.eval_history, ref["eval_history"])
        if res.best_step != ref.get("best_step"):
            extra_notes.append(f"ref_best_step={ref.get('best_step')}")
        if res.restored_best != ref.get("restored_best"):
            extra_notes.append(f"ref_restored={ref.get('restored_best')}")
        if res.stopped_early != ref.get("stopped_early"):
            extra_notes.append(f"ref_early={ref.get('stopped_early')}")

    return param_l2, upd_l2, loss_rel, val_rel, extra_notes


def _format_graph_note(graphs_s1: int, graphs_end: int, bumps_post_s1: int) -> str:
    delta = max(0, graphs_end - graphs_s1)
    return f"graphs=s1:{graphs_s1}->end:{graphs_end}(+{delta},bumps_post_s1={bumps_post_s1})"


def _format_ref_row(spec: ScenarioSpec, out: dict[str, Any], *, is_cuda: bool) -> str:
    res: TrainingResult = out["result"]
    notes: list[str] = []
    if spec.static_note:
        notes.append(spec.static_note)
    if spec.with_val:
        notes.append(
            f"early={res.stopped_early} restored={res.restored_best} best_step={res.best_step} val_s={out['val_s']:.2f}"
        )
    if spec.compile_model:
        notes.append(_format_graph_note(out["graphs_s1"], out["graphs_end"], out["graph_bumps_post_s1"]))
    if spec.test_export:
        notes.append("export=ok" if out["export_ok"] else f"export=FAIL({out['export_err']})")
    step_str = _format_num(out["steady_step_ms"], width=8, fmt=".2f")
    vram_str = _format_num(out["peak_vram_mib"] if is_cuda else None, width=8, fmt=".1f")
    return (
        f"{spec.name:<42} {res.steps:>5d} {res.skipped_batches:>4d} "
        f"{out['wall_s']:>7.2f} {step_str} {out['tail_s']:>7.2f} {vram_str}  {' | '.join(notes)}"
    )


def _format_dist_row(
    spec: ScenarioSpec,
    out: dict[str, Any],
    *,
    param_l2: float | None,
    upd_l2: float | None,
    loss_rel: float | None,
    val_rel: float | None,
    extra_notes: Sequence[str],
    world_size: int,
    is_cuda: bool,
) -> str:
    res: TrainingResult = out["result"]
    notes: list[str] = []
    if spec.static_note:
        notes.append(spec.static_note)
    if not out["xrank_exact"]:
        notes.append("XRANK_STATE_MISMATCH")
    if not out["result_eq"]:
        notes.append("RESULT_MISMATCH")
    if spec.with_val:
        notes.append(
            f"early={res.stopped_early} restored={res.restored_best} "
            f"best_step={res.best_step} val_s={out['val_max_s']:.2f}"
        )
    if spec.compile_model:
        notes.append(_format_graph_note(out["graphs_s1_max"], out["graphs_end_max"], out["graph_bumps_max"]))
    if spec.test_export:
        cnt = out["export_pass_count"]
        notes.append(f"export={'ok' if out['export_all_ok'] else 'FAIL'}({cnt}/{world_size})")
    if not spec.compare_ref:
        notes.append("xrank_only")
    notes.extend(extra_notes)

    step_str = _format_num(out["step_max_ms"], width=8, fmt=".2f")
    vram_str = _format_num(out["vram_max_mib"] if is_cuda else None, width=8, fmt=".1f")
    return (
        f"{spec.name:<42} {res.steps:>5d} {res.skipped_batches:>4d} "
        f"{'yes' if out['xrank_exact'] else 'no':>5} {out['xrank_max_abs']:>9.2e} "
        f"{_format_num(param_l2)} {_format_num(upd_l2)} "
        f"{_format_num(loss_rel)} {_format_num(val_rel)} "
        f"{out['wall_max_s']:>7.2f} {step_str} {out['tail_max_s']:>7.2f} {vram_str}  "
        f"{' | '.join(notes)}"
    )


# --------------------------------------------------------------------------- CLI helpers


def _parse_name_filter(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    parsed: list[str] = []
    for item in values:
        for part in item.split(","):
            name = part.strip()
            if name:
                parsed.append(name)
    return parsed


def _select_scenarios(only: Sequence[str] | None, skip: Sequence[str] | None) -> list[ScenarioSpec]:
    only_names = _parse_name_filter(only)
    skip_names = _parse_name_filter(skip)
    unknown = [name for name in (*only_names, *skip_names) if name not in SCENARIO_BY_NAME]
    if unknown:
        valid = ", ".join(s.name for s in ALL_SCENARIOS)
        raise SystemExit(f"error: unknown scenario name(s): {unknown}. Valid names: {valid}")

    selected = list(ALL_SCENARIOS)
    if only_names:
        keep = set(only_names)
        selected = [s for s in selected if s.name in keep]
    if skip_names:
        drop = set(skip_names)
        selected = [s for s in selected if s.name not in drop]
    if not selected:
        raise SystemExit("error: no scenarios left after applying --only / --skip")
    return selected


def _resolve_cpu_threads(args: argparse.Namespace, *, is_cuda: bool) -> int | None:
    if is_cuda:
        return None
    threads = 1 if args.cpu_threads is None else args.cpu_threads
    if threads < 1:
        raise SystemExit(f"error: --cpu-threads must be >= 1, got {threads}")
    torch.set_num_threads(threads)
    return threads


def _ref_mode_label(ref_mode: ReferenceMode) -> str:
    if ref_mode == "auto":
        return f"auto(f64<={AUTO_EXACT_F64_MAX_NUMEL},k={SKETCH_DIM})"
    if ref_mode == "sketch":
        return f"sketch(k={SKETCH_DIM})"
    return "exact"


def _env_banner(
    *,
    mode_label: str,
    backend: str,
    world_size: int,
    device_label: str,
    device_map: str | None,
    ref_mode_label: str | None,
    seed: int,
    per_rank_batch: int,
    per_rank_replay_batch: int,
    steps: int,
) -> str:
    parts = [
        f"distributed_trainer_check ({mode_label})",
        f"torch {torch.__version__}",
        f"python {platform.python_version()}",
        f"{platform.system()} {platform.machine()}",
        f"backend={backend}",
        f"world_size={world_size}",
        f"device={device_label}",
    ]
    if device_map is not None:
        parts.append(f"device_map={device_map}")
    if ref_mode_label is not None:
        parts.append(f"ref_mode={ref_mode_label}")
    if device_label.startswith("cuda") and torch.cuda.is_available():
        parts.extend(cuda_env_banner_parts(allow_cuda_context=True))
    parts.extend(
        [
            f"threads={torch.get_num_threads()}",
            f"seed={seed}",
            f"B={per_rank_batch} (eff={world_size * per_rank_batch})",
            f"b={per_rank_replay_batch} (eff={world_size * per_rank_replay_batch})",
            f"steps={steps}",
        ]
    )
    return "== " + " | ".join(parts) + " =="


# --------------------------------------------------------------------------- reference & torchrun modes


def _run_write_reference(args: argparse.Namespace, scenarios: Sequence[ScenarioSpec]) -> int:
    if dist.is_available() and dist.is_initialized():
        print(
            "error: --write-reference must be run as a plain single process, not inside a process group",
            file=sys.stderr,
        )
        return 2
    if args.world_size is None or args.world_size < 2:
        print("error: --write-reference requires --world-size N with N >= 2", file=sys.stderr)
        return 2

    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    parsed_dev = torch.device(device_str)
    if parsed_dev.type not in ("cpu", "cuda"):
        print(f"error: --device must be cpu or cuda[:index], got {device_str!r}", file=sys.stderr)
        return 2
    if parsed_dev.type == "cuda" and not torch.cuda.is_available():
        print("error: --device cuda requested, but CUDA is not available", file=sys.stderr)
        return 2
    is_cuda = parsed_dev.type == "cuda"
    cpu_threads = _resolve_cpu_threads(args, is_cuda=is_cuda)
    ref_mode: ReferenceMode = args.reference_mode
    ref_label = _ref_mode_label(ref_mode)

    models = _load_models(args.sizes, include_tiny=not args.no_tiny, seed=args.seed)
    if not models:
        print("error: --no-tiny requires at least one --sizes entry", file=sys.stderr)
        return 2

    print(
        _env_banner(
            mode_label="reference",
            backend="single-process",
            world_size=args.world_size,
            device_label=device_str,
            device_map=None,
            ref_mode_label=ref_label,
            seed=args.seed,
            per_rank_batch=args.batch_size,
            per_rank_replay_batch=args.replay_batch_size,
            steps=args.steps,
        ),
        flush=True,
    )

    out_path = Path(args.write_reference)
    writer = _StreamingReferenceWriter(out_path)
    moved_params_cache: dict[str, dict[str, Tensor]] = {}
    had_failure = False
    entry_count = 0

    try:
        for model_name, base_model in models:
            print(f"\n=== Model: {model_name} (reference for N={args.world_size}) ===", flush=True)
            print(
                f"{'scenario':<42} {'steps':>5} {'skip':>4} {'wall_s':>7} "
                f"{'step_ms':>8} {'tail_s':>7} {'vram_mib':>8}  notes",
                flush=True,
            )
            for spec in scenarios:
                if not spec.compare_ref:
                    print(
                        f"{spec.name:<42} {'-':>5} {'-':>4} {'-':>7} {'-':>8} {'-':>7} {'-':>8}"
                        "  SKIP (cross-rank only)",
                        flush=True,
                    )
                    continue
                reason = _skip_reason(spec, is_cuda=is_cuda)
                if reason is not None:
                    print(
                        f"{spec.name:<42} {'-':>5} {'-':>4} {'-':>7} {'-':>8} {'-':>7} {'-':>8}  SKIP ({reason})",
                        flush=True,
                    )
                    continue
                try:
                    out = _run_scenario(
                        spec,
                        model_name,
                        base_model,
                        world_size=args.world_size,
                        per_rank_batch=args.batch_size,
                        per_rank_replay_batch=args.replay_batch_size,
                        steps=args.steps,
                        seed=args.seed,
                        config_device=device_str,
                        distributed=False,
                        rank=0,
                        expected_cuda_index=None,
                        binding_label="single-process",
                        moved_params_cache=moved_params_cache,
                    )
                except Exception as exc:
                    had_failure = True
                    print(
                        f"{spec.name:<42} {'ERR':>5} {'-':>4} {'-':>7} {'-':>8} {'-':>7} {'-':>8}  "
                        f"ERROR: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc(file=sys.stderr)
                    continue

                res: TrainingResult = out["result"]
                key = f"{model_name}:{spec.name}"
                init_vec: Tensor = out["init_trainable_native"]
                final_vec: Tensor = out["final_trainable_native"]
                stored_tensor, final_norm, update_norm, storage_kind = _select_reference_payload(
                    init_vec,
                    final_vec,
                    ref_mode=ref_mode,
                    spec=spec,
                    seed=args.seed,
                    model_name=model_name,
                )

                writer.add_entry(
                    key,
                    {
                        "storage_kind": storage_kind,
                        "numel": int(final_vec.numel()),
                        "config_digest": out["config_digest"],
                        "init_model_sha256": out["init_model_sha256"],
                        "frozen_base_sha256": out["frozen_base_sha256"],
                        "init_trainable_sha256": out["init_trainable_sha256"],
                        "final_trainable_sha256": out["final_trainable_sha256"],
                        "final_all_finite": out["final_all_finite"],
                        "final_norm_f64": final_norm,
                        "update_norm_f64": update_norm,
                        "dom_fp": out["dom_fp"],
                        "val_fp": out["val_fp"],
                        "rep_fp": out["rep_fp"],
                        "train_history": [[s, float(v)] for s, v in res.train_history],
                        "eval_history": [[s, float(v)] for s, v in res.eval_history],
                        "steps": res.steps,
                        "skipped_batches": res.skipped_batches,
                        "best_eval_loss": res.best_eval_loss,
                        "best_step": res.best_step,
                        "stopped_early": res.stopped_early,
                        "restored_best": res.restored_best,
                        "wall_s": out["wall_s"],
                        "steady_step_ms": out["steady_step_ms"],
                        "tail_s": out["tail_s"],
                        "val_s": out["val_s"],
                        "peak_vram_mib": out["peak_vram_mib"],
                    },
                    stored_tensor,
                )
                entry_count += 1
                if spec.test_export and not out["export_ok"]:
                    had_failure = True
                print(_format_ref_row(spec, out, is_cuda=is_cuda), flush=True)

        if had_failure:
            writer.abort()
            print("\nerror: one or more reference scenarios failed; reference file not written", file=sys.stderr)
            return 1

        run_spec = _make_run_spec(
            world_size=args.world_size,
            device_type=parsed_dev.type,
            models=models,
            scenarios=scenarios,
            seed=args.seed,
            per_rank_batch=args.batch_size,
            per_rank_replay_batch=args.replay_batch_size,
            steps=args.steps,
            cpu_threads=cpu_threads,
            reference_mode=ref_mode,
            entry_keys=writer.entry_keys,
        )
        env_info = {
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "device": device_str,
            "cuda_version": torch.version.cuda if is_cuda else None,
            "threads": torch.get_num_threads(),
            "script_sha256": _current_script_sha256(),
            "git_head": git_head_commit(Path(__file__).resolve().parent),
            "archive_id": uuid.uuid4().hex,
        }
        writer.commit(spec=run_spec, env=env_info)
    except BaseException:
        writer.abort()
        raise

    size_mib = out_path.stat().st_size / (1024.0 * 1024.0)
    print(
        f"\nWrote {entry_count} reference entries for N={args.world_size} ({parsed_dev.type}, "
        f"{ref_label}, {size_mib:.2f} MiB).",
        flush=True,
    )
    return 0


def _prepare_distributed_setup(
    args: argparse.Namespace,
    scenarios: Sequence[ScenarioSpec],
    *,
    rank: int,
    world_size: int,
    local_rank: int,
) -> tuple[
    str,
    int | None,
    str,
    str,
    list[tuple[str, TuneTimestralModel]],
    _ReferenceReader | None,
    str | None,
]:
    """Validate device mapping, load CPU models, and validate reference archive on every rank before init."""
    backend: Literal["gloo", "nccl"] = args.backend
    is_cuda = backend == "nccl"
    device_map: Literal["per-rank", "shared"] = args.device_map
    cpu_threads = _resolve_cpu_threads(args, is_cuda=is_cuda)

    dev_err = validate_distributed_device_map(
        backend=backend,
        device_map=device_map,
        world_size=world_size,
        local_rank=local_rank,
    )
    if dev_err is not None:
        if rank == 0:
            print(f"error: {dev_err}", file=sys.stderr)
        raise SystemExit(2)

    if any(s.num_workers > 0 and _skip_reason(s, is_cuda=is_cuda) is None for s in scenarios):
        try:
            check_tmpdir_for_workers()
        except Exception as exc:
            if rank == 0:
                print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(2) from None

    if is_cuda:
        if device_map == "shared":
            config_device = "cuda:0"
            expected_cuda_index: int | None = 0
            binding_label = "shared"
            device_label = "cuda:0 (shared)"
        else:
            # Pass device="auto" and do NOT call torch.cuda.set_device beforehand so TimestralTrainer's
            # DistributedContext.detect() binds cuda:LOCAL_RANK live.
            config_device = "auto"
            expected_cuda_index = local_rank
            binding_label = "per-rank"
            device_label = f"cuda:0..{world_size - 1}"
    else:
        config_device = "cpu"
        expected_cuda_index = None
        binding_label = "gloo"
        device_label = "cpu"

    models = _load_models(args.sizes, include_tiny=not args.no_tiny, seed=args.seed)
    if not models:
        if rank == 0:
            print("error: --no-tiny requires at least one --sizes entry", file=sys.stderr)
        raise SystemExit(2)

    ref_reader: _ReferenceReader | None = None
    ref_label: str | None = None
    if args.reference is not None:
        ref_path = Path(args.reference)
        if not ref_path.is_file():
            if rank == 0:
                print(f"error: reference file not found: {ref_path}", file=sys.stderr)
            raise SystemExit(2)
        try:
            ref_meta, ref_entries, ref_member_sizes = _read_reference_metadata(ref_path)
            ref_mode, env_warnings = _validate_reference_spec(
                ref_meta,
                ref_entries,
                ref_member_sizes,
                world_size=world_size,
                device_type="cuda" if is_cuda else "cpu",
                models=models,
                scenarios=scenarios,
                seed=args.seed,
                per_rank_batch=args.batch_size,
                per_rank_replay_batch=args.replay_batch_size,
                steps=args.steps,
                cpu_threads=cpu_threads,
                allow_env_mismatch=args.allow_env_mismatch,
            )
            if env_warnings and rank == 0:
                print(
                    f"warning: overriding reference environment mismatch (--allow-env-mismatch): "
                    f"{'; '.join(env_warnings)}",
                    file=sys.stderr,
                )
            ref_label = _ref_mode_label(ref_mode)
            if rank == 0:
                ref_reader = _ReferenceReader(ref_path)
        except Exception as exc:
            if rank == 0:
                print(f"error: refusing to compare against reference file: {exc}", file=sys.stderr)
            raise SystemExit(2) from None

    return config_device, expected_cuda_index, binding_label, device_label, models, ref_reader, ref_label


def _run_distributed_mode(args: argparse.Namespace, scenarios: Sequence[ScenarioSpec]) -> int:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print(
            "error: RANK and WORLD_SIZE are not set. Launch with:\n"
            "  torchrun --master-addr=127.0.0.1 --master-port=29501 --nproc_per_node=N "
            "scripts/probes/distributed_trainer_check.py --backend ...",
            file=sys.stderr,
        )
        return 2
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size < 2:
        if rank == 0:
            print(f"error: multi-rank check requires WORLD_SIZE >= 2, got {world_size}", file=sys.stderr)
        return 2

    backend: Literal["gloo", "nccl"] = args.backend
    is_cuda = backend == "nccl"
    device_map: Literal["per-rank", "shared"] = args.device_map

    try:
        (
            config_device,
            expected_cuda_index,
            binding_label,
            device_label,
            models,
            ref_reader,
            ref_label,
        ) = _prepare_distributed_setup(
            args,
            scenarios,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    had_failure = False
    clean_collective_exit = False
    moved_params_cache: dict[str, dict[str, Tensor]] = {}
    try:
        # In shared single-GPU mode, bind cuda:0 before init_process_group; in per-rank mode, leave
        # CUDA device binding to TimestralTrainer so the trainer's LOCAL_RANK path is genuinely exercised.
        if is_cuda and device_map == "shared":
            torch.cuda.set_device(0)

        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(seconds=args.timeout),
        )

        if rank == 0:
            print(
                _env_banner(
                    mode_label="torchrun",
                    backend=backend,
                    world_size=world_size,
                    device_label=device_label,
                    device_map=device_map if is_cuda else None,
                    ref_mode_label=ref_label,
                    seed=args.seed,
                    per_rank_batch=args.batch_size,
                    per_rank_replay_batch=args.replay_batch_size,
                    steps=args.steps,
                ),
                flush=True,
            )
            if args.reference is None:
                print("(no --reference supplied: reporting cross-rank agreement and timings only)", flush=True)

        header = (
            f"{'scenario':<42} {'steps':>5} {'skip':>4} {'xr_eq':>5} {'xr_max':>9} "
            f"{'param_l2':>9} {'upd_l2':>9} {'loss_rel':>9} {'val_rel':>9} "
            f"{'wall_s':>7} {'step_ms':>8} {'tail_s':>7} {'vram_mib':>8}  notes"
        )
        for model_name, base_model in models:
            if rank == 0:
                print(f"\n=== Model: {model_name} ===", flush=True)
                print(header, flush=True)
            for spec in scenarios:
                reason = _skip_reason(spec, is_cuda=is_cuda)
                if reason is not None:
                    if rank == 0:
                        print(
                            f"{spec.name:<42} {'-':>5} {'-':>4} {'-':>5} {'-':>9} "
                            f"{'-':>9} {'-':>9} {'-':>9} {'-':>9} "
                            f"{'-':>7} {'-':>8} {'-':>7} {'-':>8}  SKIP ({reason})",
                            flush=True,
                        )
                    continue

                try:
                    out = _run_scenario(
                        spec,
                        model_name,
                        base_model,
                        world_size=world_size,
                        per_rank_batch=args.batch_size,
                        per_rank_replay_batch=args.replay_batch_size,
                        steps=args.steps,
                        seed=args.seed,
                        config_device=config_device,
                        distributed=True,
                        rank=rank,
                        expected_cuda_index=expected_cuda_index,
                        binding_label=binding_label,
                        moved_params_cache=moved_params_cache,
                    )
                except Exception as exc:
                    had_failure = True
                    print(
                        f"[rank {rank}] ERROR in {model_name}:{spec.name}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    traceback.print_exc(file=sys.stderr)
                    # Leave clean_collective_exit=False so finally skips destroy_process_group()
                    # and torchrun reaps peers promptly without stalling on broken NCCL teardown.
                    raise

                if spec.test_export and not out["export_ok"]:
                    print(
                        f"[rank {rank}] export check failed in {model_name}:{spec.name}: {out['export_err']}",
                        file=sys.stderr,
                        flush=True,
                    )

                if (spec.deterministic and (not out["xrank_exact"] or not out["result_eq"])) or (
                    spec.test_export and not out["export_all_ok"]
                ):
                    had_failure = True

                # Rank 0 compares against reference and broadcasts ref_error_flag BEFORE row formatting
                # so no formatting exception can ever desynchronise the collective.
                ref_error_flag = torch.zeros(1, dtype=torch.int64, device=out["device"])
                cmp_result: tuple[float | None, float | None, float | None, float | None, list[str]] | None = None
                if rank == 0:
                    try:
                        cmp_result = _compare_to_reference(
                            out,
                            ref_reader,
                            model_name=model_name,
                            spec=spec,
                            seed=args.seed,
                        )
                    except Exception as exc:
                        ref_error_flag.fill_(1)
                        print(f"error: reference comparison failed: {exc}", file=sys.stderr, flush=True)

                dist.broadcast(ref_error_flag, src=0)
                if int(ref_error_flag.item()) != 0:
                    clean_collective_exit = True
                    return 2

                if rank == 0 and cmp_result is not None:
                    param_l2, upd_l2, loss_rel, val_rel, extra_notes = cmp_result
                    print(
                        _format_dist_row(
                            spec,
                            out,
                            param_l2=param_l2,
                            upd_l2=upd_l2,
                            loss_rel=loss_rel,
                            val_rel=val_rel,
                            extra_notes=extra_notes,
                            world_size=world_size,
                            is_cuda=is_cuda,
                        ),
                        flush=True,
                    )
        clean_collective_exit = True
    finally:
        if ref_reader is not None:
            ref_reader.close()
        if clean_collective_exit and dist.is_initialized():
            dist.destroy_process_group()

    return 1 if had_failure else 0


# --------------------------------------------------------------------------- CLI entrypoint


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list",
        "--list-scenarios",
        dest="list_scenarios",
        action="store_true",
        help="Print all available scenario names and exit.",
    )
    parser.add_argument(
        "--write-reference",
        type=str,
        default=None,
        help="Run in plain single-process mode and write reference archive to this path.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("auto", "sketch", "exact"),
        default="auto",
        help=(
            f"Reference vector storage for --write-reference: 'auto' (default; exact float64 delta when "
            f"dtype=float64 and numel <= {AUTO_EXACT_F64_MAX_NUMEL:,} or numel <= {SKETCH_DIM}, else "
            f"CountSketch dim {SKETCH_DIM}), 'sketch' (CountSketch whenever numel > {SKETCH_DIM}), "
            "or 'exact' (full native-dtype trainable vectors streamed to zip)."
        ),
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to a reference archive written by --write-reference (validated before init, loaded on rank 0).",
    )
    parser.add_argument(
        "--allow-env-mismatch",
        action="store_true",
        help=(
            "Allow comparing against a reference archive built with a different torch/CUDA version or "
            "probe script revision (prints a warning with every difference; never overrides schema, "
            "model, config, or missing-member mismatches)."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("gloo", "nccl"),
        default=None,
        help="Process-group backend for torchrun mode.",
    )
    parser.add_argument(
        "--device-map",
        choices=("per-rank", "shared"),
        default="per-rank",
        help="NCCL device mapping: per-rank (device='auto' -> cuda:LOCAL_RANK) or shared (cuda:0 on all ranks).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for --write-reference (cpu or cuda[:index]; defaults to cuda:0 if available else cpu).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Target world size N for --write-reference (sets effective batch sizes N*B and N*b).",
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        choices=SIZES,
        default=[],
        help="Additional released checkpoint sizes to load via _common.load (e.g. --sizes 5M 28M).",
    )
    parser.add_argument(
        "--no-tiny",
        action="store_true",
        help="Skip tiny_random and run only models named in --sizes.",
    )
    parser.add_argument(
        "--only",
        action="extend",
        nargs="+",
        default=None,
        help="Run only the named scenario(s) (space- or comma-separated, repeatable).",
    )
    parser.add_argument(
        "--skip",
        action="extend",
        nargs="+",
        default=None,
        help="Skip the named scenario(s) (space- or comma-separated, repeatable).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed (default: 0).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Per-rank domain batch size B (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--replay-batch-size",
        type=int,
        default=DEFAULT_REPLAY_BATCH_SIZE,
        help=f"Per-rank replay batch size b (default: {DEFAULT_REPLAY_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS, help=f"Default max_steps (default: {DEFAULT_STEPS})."
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Intra-op CPU threads for CPU/gloo runs (default: 1, matching torchrun's OMP_NUM_THREADS=1 default).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"init_process_group timeout in seconds for torchrun mode (default: {DEFAULT_TIMEOUT_S}).",
    )
    return parser


def _validate_numeric_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        parser.error(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.replay_batch_size < 1:
        parser.error(f"--replay-batch-size must be >= 1, got {args.replay_batch_size}")
    if args.steps < 1:
        parser.error(f"--steps must be >= 1, got {args.steps}")
    require_positive_finite(parser, "--timeout", args.timeout)
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error(f"--cpu-threads must be >= 1, got {args.cpu_threads}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_scenarios:
        for spec in ALL_SCENARIOS:
            flags: list[str] = [f"g{spec.group}", spec.mode, spec.dtype]
            if spec.mixed_precision != "fp32":
                flags.append(spec.mixed_precision)
            if spec.uses_replay:
                flags.append(f"rf={spec.replay_fraction},dw={spec.distill_weight}")
            if spec.moved_student:
                flags.append("moved_student")
            if spec.with_val:
                flags.append("val")
            if spec.compile_model:
                flags.append("compile")
            if spec.num_workers > 0:
                flags.append(f"workers={spec.num_workers}")
            if spec.test_export:
                flags.append("export")
            if not spec.compare_ref:
                flags.append("xrank_only")
            print(f"{spec.name:<42} ({', '.join(flags)})")
        return 0

    _validate_numeric_args(parser, args)
    scenarios = _select_scenarios(args.only, args.skip)

    if args.write_reference is not None:
        if args.backend is not None:
            parser.error("--write-reference cannot be combined with --backend")
        return _run_write_reference(args, scenarios)

    if args.backend is None:
        parser.error("specify either --write-reference PATH or --backend gloo|nccl")
    return _run_distributed_mode(args, scenarios)


if __name__ == "__main__":
    sys.exit(main())
