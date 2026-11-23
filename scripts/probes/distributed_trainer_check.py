"""Probe: multi-rank `TimestralTrainer` parity, telemetry, and Stage 3 CUDA/NCCL driver.

Runs the real `TimestralTrainer` across ranks under `torchrun` on either `gloo` (CPU) or `nccl`
(CUDA) and compares against a one-process reference saved on the same device type. Reports
measured numbers without grading float32/bf16 tolerances; exits non-zero only on crashes, detected
hangs, cross-rank mismatches of parameters or `TrainingResult` in a deterministic scenario,
failed export checks, reference mismatches, or raised exceptions.

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

Reference storage (`--reference-mode`, default `sketch`):
- `sketch`: stores exact float64 `delta = final - init` whenever `numel <= 8192` (all tiny LoRA
  scenarios are 100% exact) and a deterministic `k = 8192` float64 CountSketch of `delta`
  (`64 KiB` per scenario, ~0.8% relative L2 standard error across 15+ orders of magnitude) when
  `numel > 8192`, alongside exact SHA-256 digests (`init_sha256`, `final_sha256` -> exact `0.0`
  on bitwise matches) and exact float64 `||final||_2` and `||delta||_2`. Keeps reference archives
  under ~2 MiB for `tiny_random` and ~5 MiB for `5M`+`28M` with O(1) peak memory.
- `exact`: streams full native-dtype trainable vectors into the zip archive (one entry per
  scenario, O(1) peak memory; large on disk for released `full_ft`).

See `docs/phases/phase-5-multi-gpu.md` §3 and `docs/SPEC.md` §5.6, §12.
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
import re
import struct
import sys
import tempfile
import time
import traceback
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from _common import PATCH, SIZES, float64_reference, load, rel, tiny_random_forecaster
from torch import Tensor

from timestral_tune import LoraConfig, ReplayConfig, TimestralTrainer, TrainingConfig
from timestral_tune.config import AdaptationMode
from timestral_tune.data import ColumnRoles, SeriesData, TimeSeriesDataset
from timestral_tune.export import dataset_fingerprint
from timestral_tune.models import TuneTimestralModel
from timestral_tune.trainer import TrainingResult, compute_loss


def _torch_major_minor() -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if match is None:
        return (0, 0)
    return int(match.group(1)), int(match.group(2))


# PyTorch < 2.8 on macOS hangs at interpreter exit after DataLoader worker processes run (VERIFICATION.md §10).
WORKERS_HANG_AT_EXIT = sys.platform == "darwin" and _torch_major_minor() < (2, 8)

SCHEMA_VERSION = 3
CTX_PATCHES = 2
HOR_PATCHES = 2
DOM_HORIZON = HOR_PATCHES * PATCH  # 64
REP_HORIZON = PATCH + PATCH // 2  # 48 (non-multiple of PATCH; exercises distillation mask padding)
DEFAULT_BATCH_SIZE = 2
DEFAULT_REPLAY_BATCH_SIZE = 2
DEFAULT_STEPS = 6
DEFAULT_TIMEOUT_S = 180.0
SKETCH_DIM = 8192
SKETCH_CHUNK = 262_144

# macOS AF_UNIX sun_path limit is 103 bytes (107 on Linux); torch_shm_manager reads raw getenv("TMPDIR")
# (falling back to "/tmp") without stripping a trailing slash and opens
# `<TMPDIR>/torch-shm-dir-XXXXXX/manager.sock`. If this overflows inside a DataLoader worker's
# Queue._feed background thread, Queue._feed catches the RuntimeError, drops the batch, and leaves
# workers alive while DataLoader waits forever.
_SHM_SUFFIX_LEN = len("/torch-shm-dir-XXXXXX/manager.sock")
_UNIX_PATH_MAX = 103 if sys.platform == "darwin" else 107

_DTYPE_FROM_NAME: dict[str, torch.dtype] = {
    "float64": torch.float64,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
_NAME_FROM_DTYPE: dict[torch.dtype, str] = {v: k for k, v in _DTYPE_FROM_NAME.items()}


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


# --------------------------------------------------------------------------- synthetic datasets & helpers


def _make_dataset(
    num_windows: int,
    *,
    horizon: int,
    covariates: bool,
    seed: int,
    future_ramp: float = 0.0,
) -> TimeSeriesDataset:
    """Self-contained synthetic `TimeSeriesDataset` with unequal per-window scored counts."""
    rng = np.random.default_rng(seed)
    ctx_len = CTX_PATCHES * PATCH
    total = ctx_len + horizon
    ts = np.arange(total, dtype=np.int64)
    empty_f = np.empty((0, total), dtype=np.float32)
    empty_c = np.empty((0, total), dtype=np.float64)
    ramp = np.linspace(0.0, future_ramp, horizon, dtype=np.float32) if future_ramp != 0.0 else None

    series: list[SeriesData] = []
    for i in range(num_windows):
        values = (15.0 + 3.0 * np.sin(0.2 * ts + 0.5 * i) + rng.normal(0.0, 0.8, size=total)).astype(np.float32)
        if ramp is not None:
            values[ctx_len:] = values[ctx_len:] + ramp
        # Drop a varying tail of horizon targets while keeping >= 1 scored step in every window.
        drop = (i * 11) % max(1, horizon - 1)
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
        forecast_horizon=horizon,
        stride=horizon,
        min_context=ctx_len,
        roles=roles,
    )


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


def _flat_trainable_native(trainer: TimestralTrainer) -> Tensor:
    """Concatenate trainable parameters on CPU in their native dtype."""
    return torch.cat([p.detach().cpu().reshape(-1) for p in trainer._trainable])


def _tensor_sha256(tensor: Tensor) -> str:
    arr = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _sketch_seed(base_seed: int, model_name: str, scenario_name: str) -> int:
    digest = hashlib.sha256(f"timestral-tune:stage3-sketch:{base_seed}:{model_name}:{scenario_name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _count_sketch_delta(
    init_native: Tensor,
    final_native: Tensor,
    *,
    sketch_dim: int,
    seed: int,
) -> tuple[Tensor, float, float, Literal["exact_delta", "count_sketch"]]:
    """Compute `(compact_vector_f64, final_norm_f64, update_norm_f64, kind)` in O(chunk) memory.

    When `numel <= sketch_dim`, returns the exact float64 `delta = final - init` vector (`exact_delta`).
    Otherwise returns a deterministic linear CountSketch of `delta` in float64 (`count_sketch`),
    giving an unbiased L2 estimator for `delta_mp - delta_ref == final_mp - final_ref` with
    relative standard error <= `1 / sqrt(2 * sketch_dim)` (~0.78% at `sketch_dim = 8192`).
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


def _tensor_to_bytes(tensor: Tensor) -> tuple[bytes, str, list[int]]:
    arr = tensor.detach().cpu().contiguous()
    dtype_name = _NAME_FROM_DTYPE.get(arr.dtype)
    if dtype_name is None:
        raise ValueError(f"unsupported tensor dtype for archive: {arr.dtype}")
    return arr.view(torch.uint8).numpy().tobytes(), dtype_name, list(arr.shape)


def _bytes_to_tensor(raw: bytes, dtype_name: str, shape: Sequence[int]) -> Tensor:
    dtype = _DTYPE_FROM_NAME[dtype_name]
    buf = bytearray(raw)
    return torch.frombuffer(buf, dtype=torch.uint8).view(dtype).reshape(tuple(shape)).clone()


def _scalar_rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def _encode_digest_words(digest: bytes) -> list[int]:
    return list(struct.unpack("<4q", hashlib.sha256(digest).digest()))


def _hash_all_parameters(model: TuneTimestralModel) -> bytes:
    """SHA-256 over every parameter in `sorted(model.named_parameters())` (name, shape, dtype, raw bytes)."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        arr = param.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        # View as uint8 to hash exact bit representation across fp64, fp32, and bf16.
        h.update(arr.view(torch.uint8).numpy().tobytes())
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


def _check_tmpdir_for_workers() -> None:
    """Fail fast before spawning DataLoader workers if TMPDIR overflows torch_shm_manager's sun_path."""
    try:
        sharing = torch.multiprocessing.get_sharing_strategy()
    except Exception:
        sharing = "file_system" if sys.platform == "darwin" else "file_descriptor"
    if sharing != "file_system":
        return
    # torch_shm_manager reads raw getenv("TMPDIR") (defaulting to "/tmp") without stripping a
    # trailing slash and appends "/torch-shm-dir-XXXXXX/manager.sock" (34 bytes).
    raw_tmp = os.environ.get("TMPDIR") or "/tmp"
    raw_bytes = len(raw_tmp.encode("utf-8"))
    sock_len = raw_bytes + _SHM_SUFFIX_LEN
    if sock_len > _UNIX_PATH_MAX:
        raise RuntimeError(
            f"TMPDIR length ({raw_bytes} bytes) makes torch_shm_manager's Unix socket path "
            f"{sock_len} bytes (> {_UNIX_PATH_MAX}-byte limit), which silently deadlocks DataLoader "
            f"workers inside Queue._feed; re-run with TMPDIR=/tmp"
        )


def _build_datasets_for_scenario(
    spec: ScenarioSpec,
    *,
    world_size: int,
    per_rank_batch: int,
    per_rank_replay_batch: int,
    seed: int,
) -> tuple[TimeSeriesDataset, TimeSeriesDataset | None, TimeSeriesDataset | None]:
    # Trailing partial batch:
    # - `empty_trailing_shard`: `N * B + 1` gives rank 0 a 1-window trailing batch and ranks 1..N-1 empty shards.
    # - default: `N * B + max(1, N - 1)` gives unequal trailing counts and leaves rank N - 1 empty.
    trailing = 1 if spec.empty_trailing_shard else max(1, world_size - 1)
    dom_len = world_size * per_rank_batch + trailing
    dom_ramp = 40.0 if spec.early_stopping_patience is not None else 0.0
    dom_ds = _make_dataset(dom_len, horizon=DOM_HORIZON, covariates=True, seed=seed + 1001, future_ramp=dom_ramp)

    val_ds: TimeSeriesDataset | None = None
    if spec.with_val:
        # Uneven validation shards across ranks (`2 * N + 1` windows). Opposite future ramp vs train_ds
        # causes validation loss to rise after initial steps so EarlyStopping and restore_best=True fire.
        val_ramp = -40.0 if spec.early_stopping_patience is not None else 0.0
        val_ds = _make_dataset(
            2 * world_size + 1,
            horizon=DOM_HORIZON,
            covariates=True,
            seed=seed + 1002,
            future_ramp=val_ramp,
        )

    rep_ds: TimeSeriesDataset | None = None
    if spec.uses_replay:
        rep_len = world_size * per_rank_replay_batch + max(1, world_size - 1)
        rep_ds = _make_dataset(rep_len, horizon=REP_HORIZON, covariates=False, seed=seed + 1003)

    return dom_ds, val_ds, rep_ds


def _build_moved_full_ft_state(
    base_model_cpu: TuneTimestralModel,
    dom_ds: TimeSeriesDataset,
    *,
    is_released: bool,
    steps: int = 3,
    batch_size: int = 4,
) -> dict[str, Tensor]:
    """Run 3 deterministic domain-only `full_ft` steps on CPU (1 thread, float32) so student != teacher.

    Runs outside `TimestralTrainer` so it works identically in plain reference mode and inside a
    live `torchrun` process group without triggering distributed startup collectives.
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = copy.deepcopy(base_model_cpu).float().train()
        for p in model.parameters():
            p.requires_grad_(True)
        lr = 1e-4 if is_released else 5e-3
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        collator = dom_ds.collator(model.config)
        n_win = len(dom_ds)
        for step_idx in range(steps):
            start = (step_idx * batch_size) % n_win
            idxs = [(start + j) % n_win for j in range(min(batch_size, n_win))]
            batch = collator([dom_ds[i] for i in idxs])
            batch.validate(model.config.patch_size)
            opt.zero_grad(set_to_none=True)
            compute_loss(model, batch, validate=False).loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    finally:
        torch.set_num_threads(prev_threads)


# --------------------------------------------------------------------------- core execution


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


def _verify_export(
    trainer: TimestralTrainer,
    *,
    spec: ScenarioSpec,
    distributed: bool,
    rank: int,
    world_size: int,
    per_rank_batch: int,
) -> tuple[bool, str]:
    """Verify `export()`: rank 0 writes complete checkpoint + metadata, peers write nothing."""
    with tempfile.TemporaryDirectory(prefix=f"tt_stage3_exp_r{rank}_") as tmpdir:
        out_dir = Path(tmpdir) / "ckpt"
        returned = trainer.export(out_dir)
        if returned != out_dir:
            return False, f"returned {returned} != {out_dir}"
        if distributed and rank != 0:
            if out_dir.exists():
                return False, f"rank {rank} created export directory"
            return True, ""

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
        return True, ""


def _gather_distributed_checks(
    model: TuneTimestralModel,
    result: TrainingResult,
    local: dict[str, Any],
    *,
    world_size: int,
    rank: int,
) -> dict[str, Any]:
    """Collect cross-rank equality, max-abs parameter diff, and max-over-ranks telemetry via collectives.

    What is compared across ranks:
    - `result_eq`: SHA-256 of all ten `TrainingResult` fields (with exact `float.hex()` encodings)
      broadcast from rank 0 as 4 signed `int64` words and checked for unanimity via `all_reduce(SUM)`.
    - `xrank_exact`: SHA-256 of every parameter in `sorted(model.named_parameters())` (both trainable
      and frozen, including LoRA A/B adapters when present), hashing `(name, shape, dtype, raw uint8 bytes)`,
      broadcast from rank 0 as 4 signed `int64` words and checked for unanimity via `all_reduce(SUM)`.
    - `xrank_max_abs`: exact elementwise `max(abs(rank_r.double() - rank_0.double()))` over every
      parameter in `sorted(model.named_parameters())`, computed by broadcasting each parameter from
      rank 0 to all ranks and reducing the per-rank maximum across ranks with `all_reduce(MAX)`.
    - Timing/memory/compile counters (`wall_s`, `steady_step_ms`, `tail_s`, `peak_vram_mib`,
      `total_graphs`, `extra_graphs`): maximum across ranks via `all_reduce(MAX)`.
    """
    device: torch.device = local["device"]

    # 1. Exact TrainingResult equality via 4x int64 SHA-256 digest words.
    res_words = _encode_digest_words(_hash_training_result(result))
    res_buf = torch.tensor(res_words, dtype=torch.int64, device=device)
    dist.broadcast(res_buf, src=0)
    res_match = torch.tensor([1 if res_buf.tolist() == res_words else 0], dtype=torch.int64, device=device)
    dist.all_reduce(res_match, op=dist.ReduceOp.SUM)
    result_eq = int(res_match.item()) == world_size

    # 2. Exact all-parameter bitwise equality via 4x int64 SHA-256 digest words.
    param_words = _encode_digest_words(_hash_all_parameters(model))
    param_buf = torch.tensor(param_words, dtype=torch.int64, device=device)
    dist.broadcast(param_buf, src=0)
    param_match = torch.tensor([1 if param_buf.tolist() == param_words else 0], dtype=torch.int64, device=device)
    dist.all_reduce(param_match, op=dist.ReduceOp.SUM)
    xrank_exact = int(param_match.item()) == world_size

    # 3. Cross-rank max-abs difference across every parameter in sorted(model.named_parameters()).
    local_max_abs = 0.0
    local_nonfinite_diff = 0.0
    with torch.no_grad():
        for _, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            if param.numel() == 0:
                continue
            src_tensor = param.detach().to(device=device).contiguous()
            buf = src_tensor.clone() if rank == 0 else torch.empty_like(src_tensor)
            dist.broadcast(buf, src=0)
            if rank != 0:
                diff = float((src_tensor.double() - buf.double()).abs().max().item())
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
            float(local["peak_vram_mib"]),
            float(local["total_graphs"]),
            float(local["extra_graphs"]),
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
        vram_max_mib,
        graphs_max,
        extra_graphs_max,
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
        "vram_max_mib": vram_max_mib,
        "graphs_max": int(round(graphs_max)),
        "extra_graphs_max": int(round(extra_graphs_max)),
        "export_all_ok": export_any_failed == 0.0,
        "export_pass_count": int(export_pass_count.item()),
    }


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
    config_device: str | None,
    distributed: bool,
    rank: int,
    expected_cuda_index: int | None,
    binding_label: str,
    moved_state_cache: dict[str, dict[str, Tensor]],
) -> dict[str, Any]:
    """Execute one scenario in either 1-process reference mode or multi-rank mode."""
    if spec.num_workers > 0:
        _check_tmpdir_for_workers()

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

    eff_batch = per_rank_batch if distributed else world_size * per_rank_batch
    eff_replay_batch = per_rank_replay_batch if distributed else world_size * per_rank_replay_batch
    eff_steps = spec.steps_override if spec.steps_override is not None else steps
    lr = _scenario_lr(spec, is_released=is_released)

    if spec.compile_model:
        _reset_dynamo()

    model = copy.deepcopy(base_model_cpu)
    if spec.dtype == "float64":
        model = model.double()

    cfg = TrainingConfig(
        model_name_or_path="local-stage3-probe",
        mode=spec.mode,
        device=config_device,  # type: ignore[arg-type]
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

    ctx_mgr = float64_reference() if spec.dtype == "float64" else contextlib.nullcontext()
    trainer: TimestralTrainer | None = None
    device: torch.device | None = None
    try:
        with ctx_mgr:
            trainer = TimestralTrainer(
                cfg,
                dom_ds,
                val_ds,
                lora_config=_lora_for(spec.mode, spec.lora_dropout),
                replay_config=rcfg,
                replay_dataset=rep_ds,
                model=model,
            )
            device = trainer.device
            if distributed and expected_cuda_index is not None and device.type == "cuda":
                _verify_cuda_device_binding(
                    trainer,
                    rank=rank,
                    expected_device_index=expected_cuda_index,
                    label=binding_label,
                )

            if spec.moved_student:
                # Load pre-warmed student weights AFTER `TimestralTrainer.__init__` so `_frozen_base`
                # and cached teacher quantiles stay anchored at the original step-0 base model.
                # In multi-rank mode, only rank 0 runs the CPU warmup and broadcasts to peers.
                if not distributed or rank == 0:
                    cache_key = f"{model_name}:{dom_fp}"
                    if cache_key not in moved_state_cache:
                        moved_state_cache[cache_key] = _build_moved_full_ft_state(
                            base_model_cpu,
                            dom_ds,
                            is_released=is_released,
                        )
                    trainer.model.load_state_dict(moved_state_cache[cache_key])
                if distributed:
                    with torch.no_grad():
                        for _, p in sorted(trainer.model.named_parameters(), key=lambda kv: kv[0]):
                            if p.device == device and p.is_contiguous():
                                dist.broadcast(p.detach(), src=0)
                            else:
                                staged = p.detach().to(device=device).contiguous()
                                dist.broadcast(staged, src=0)
                                p.copy_(staged)

            init_trainable_native = _flat_trainable_native(trainer).clone()
            init_sha256 = _tensor_sha256(init_trainable_native)

            # Record optimiser-step completion timestamps (CUDA-synchronised) and step-1 Dynamo graph count.
            step_times: list[float] = []
            graphs_after_step1 = [0]

            def _on_opt_step(*_args: Any) -> None:
                assert device is not None
                _cuda_sync_if_needed(device)
                step_times.append(time.perf_counter())
                if spec.compile_model and len(step_times) == 1:
                    graphs_after_step1[0] = _dynamo_unique_graphs()

            hook_handle = trainer.optimizer.register_step_post_hook(_on_opt_step)
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                result = trainer.fit()
                _cuda_sync_if_needed(device)
                t1 = time.perf_counter()
            finally:
                hook_handle.remove()

            wall_s = t1 - t0
            # Mean step time strictly after the first two optimiser steps.
            if len(step_times) >= 3:
                steady_step_ms: float | None = (step_times[-1] - step_times[1]) / (len(step_times) - 2) * 1000.0
            else:
                steady_step_ms = None
            tail_s = (t1 - step_times[-1]) if step_times else wall_s
            peak_vram_mib = (
                float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0) if device.type == "cuda" else 0.0
            )
            total_graphs = _dynamo_unique_graphs() if spec.compile_model else 0
            extra_graphs = max(0, total_graphs - graphs_after_step1[0]) if spec.compile_model else 0

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
                )

            final_trainable_native = _flat_trainable_native(trainer).clone()
            final_sha256 = _tensor_sha256(final_trainable_native)

            summary: dict[str, Any] = {
                "result": result,
                "init_trainable_native": init_trainable_native,
                "init_sha256": init_sha256,
                "final_trainable_native": final_trainable_native,
                "final_sha256": final_sha256,
                "dom_fp": dom_fp,
                "val_fp": val_fp,
                "rep_fp": rep_fp,
                "wall_s": wall_s,
                "steady_step_ms": steady_step_ms,
                "tail_s": tail_s,
                "peak_vram_mib": peak_vram_mib,
                "total_graphs": total_graphs,
                "extra_graphs": extra_graphs,
                "export_ok": export_ok,
                "export_err": export_err,
                "device": device,
            }

            if distributed:
                summary.update(
                    _gather_distributed_checks(trainer.model, result, summary, world_size=world_size, rank=rank)
                )
    finally:
        # Release model, frozen teacher, and optimiser buffers promptly so CUDA peak VRAM stays isolated per scenario.
        del trainer, model
        if device is not None and device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    return summary


# --------------------------------------------------------------------------- reference archive I/O & validation


def _make_run_spec(
    *,
    world_size: int,
    device_type: str,
    model_names: Sequence[str],
    scenarios: Sequence[ScenarioSpec],
    seed: int,
    per_rank_batch: int,
    per_rank_replay_batch: int,
    steps: int,
    cpu_threads: int | None,
    reference_mode: Literal["sketch", "exact"],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_mode": reference_mode,
        "sketch_dim": SKETCH_DIM if reference_mode == "sketch" else None,
        "world_size": world_size,
        "device_type": device_type,
        "models": list(model_names),
        "scenarios": [s.to_meta() for s in scenarios if s.compare_ref],
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
        self.meta: dict[str, Any] = json.loads(self._zf.read("spec.json").decode("utf-8"))
        self.entries: dict[str, dict[str, Any]] = json.loads(self._zf.read("entries.json").decode("utf-8"))

    def load_tensor(self, entry: dict[str, Any]) -> Tensor:
        raw = self._zf.read(entry["tensor_member"])
        return _bytes_to_tensor(raw, entry["tensor_dtype"], entry["tensor_shape"])

    def close(self) -> None:
        self._zf.close()


def _read_reference_spec(path: Path) -> dict[str, Any]:
    """Read only `spec.json` from the reference archive (fast, zero tensor memory)."""
    if not zipfile.is_zipfile(path):
        raise ValueError("reference file is not a valid reference archive (missing spec.json)")
    with zipfile.ZipFile(path, mode="r") as zf:
        names = set(zf.namelist())
        if "spec.json" not in names or "entries.json" not in names:
            raise ValueError("reference archive is malformed (missing 'spec.json' or 'entries.json')")
        return json.loads(zf.read("spec.json").decode("utf-8"))


def _validate_reference_spec(
    meta: dict[str, Any],
    *,
    world_size: int,
    device_type: str,
    model_names: Sequence[str],
    scenarios: Sequence[ScenarioSpec],
    seed: int,
    per_rank_batch: int,
    per_rank_replay_batch: int,
    steps: int,
    cpu_threads: int | None,
) -> str:
    """Refuse to compare if any run or scenario spec field mismatches, listing every difference.

    Returns the reference archive's storage mode (`"sketch"` or `"exact"`).
    """
    if not isinstance(meta, dict) or "spec" not in meta:
        raise ValueError("reference metadata is malformed (missing 'spec')")
    ref_spec = meta["spec"]
    diffs: list[str] = []

    if ref_spec.get("schema_version") != SCHEMA_VERSION:
        diffs.append(f"schema_version: reference={ref_spec.get('schema_version')!r}, current={SCHEMA_VERSION!r}")

    ref_mode = ref_spec.get("reference_mode")
    if ref_mode not in ("sketch", "exact"):
        diffs.append(f"reference_mode: unsupported value {ref_mode!r}")
    elif ref_mode == "sketch" and ref_spec.get("sketch_dim") != SKETCH_DIM:
        diffs.append(f"sketch_dim: reference={ref_spec.get('sketch_dim')!r}, current={SKETCH_DIM!r}")

    scalar_checks = (
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

    ref_models = set(ref_spec.get("models", []))
    missing_models = [m for m in model_names if m not in ref_models]
    if missing_models:
        diffs.append(f"models: missing {missing_models} (reference has {sorted(ref_models)})")

    ref_scen_by_name = {entry["name"]: entry for entry in ref_spec.get("scenarios", [])}
    is_cuda = device_type == "cuda"
    for spec in scenarios:
        if not spec.compare_ref or _skip_reason(spec, is_cuda=is_cuda) is not None:
            continue
        if spec.name not in ref_scen_by_name:
            diffs.append(f"scenario {spec.name!r}: missing from reference file")
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

    if diffs:
        raise ValueError("; ".join(diffs))
    assert isinstance(ref_mode, str)
    return ref_mode


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
        parts.append(f"cuda={torch.version.cuda}")
        nccl_ver = torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else None
        if nccl_ver is not None:
            parts.append(f"nccl={'.'.join(str(x) for x in nccl_ver)}")
        parts.append(f"gpu={torch.cuda.get_device_name(0)}")
        parts.append(f"gpu_count={torch.cuda.device_count()}")
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
    ref_mode: Literal["sketch", "exact"] = args.reference_mode
    ref_mode_label = f"sketch(k={SKETCH_DIM})" if ref_mode == "sketch" else "exact"

    models = _load_models(args.sizes, include_tiny=not args.no_tiny, seed=args.seed)
    model_names = [name for name, _ in models]
    if not model_names:
        print("error: --no-tiny requires at least one --sizes entry", file=sys.stderr)
        return 2

    print(
        _env_banner(
            mode_label="reference",
            backend="single-process",
            world_size=args.world_size,
            device_label=device_str,
            device_map=None,
            ref_mode_label=ref_mode_label,
            seed=args.seed,
            per_rank_batch=args.batch_size,
            per_rank_replay_batch=args.replay_batch_size,
            steps=args.steps,
        ),
        flush=True,
    )

    out_path = Path(args.write_reference)
    writer = _StreamingReferenceWriter(out_path)
    moved_state_cache: dict[str, dict[str, Tensor]] = {}
    had_failure = False
    entry_count = 0

    try:
        for model_name, base_model in models:
            print(f"\n=== Model: {model_name} (reference for N={args.world_size}) ===", flush=True)
            print(
                f"{'scenario':<42} {'steps':>5} {'skip':>4} {'wall_s':>7} "
                f"{'step_ms':>8} {'tail_s':>7} {'vram_mb':>8}  notes",
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
                        moved_state_cache=moved_state_cache,
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

                if ref_mode == "sketch":
                    stored_tensor, final_norm, update_norm, storage_kind = _count_sketch_delta(
                        init_vec,
                        final_vec,
                        sketch_dim=SKETCH_DIM,
                        seed=_sketch_seed(args.seed, model_name, spec.name),
                    )
                else:
                    stored_tensor = final_vec
                    final_norm = float(final_vec.double().norm().item())
                    update_norm = float((final_vec.double() - init_vec.double()).norm().item())
                    storage_kind = "exact_final"

                writer.add_entry(
                    key,
                    {
                        "storage_kind": storage_kind,
                        "numel": int(final_vec.numel()),
                        "init_sha256": out["init_sha256"],
                        "final_sha256": out["final_sha256"],
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
                        "peak_vram_mib": out["peak_vram_mib"],
                    },
                    stored_tensor,
                )
                entry_count += 1

                notes: list[str] = []
                if spec.static_note:
                    notes.append(spec.static_note)
                if spec.with_val:
                    notes.append(f"early={res.stopped_early} restored={res.restored_best} best_step={res.best_step}")
                if spec.compile_model:
                    notes.append(
                        f"graphs={out['total_graphs']} recompiled={'yes' if out['extra_graphs'] > 0 else 'no'}"
                    )
                if spec.test_export:
                    notes.append("export=ok" if out["export_ok"] else f"export=FAIL({out['export_err']})")
                    if not out["export_ok"]:
                        had_failure = True
                step_str = _format_num(out["steady_step_ms"], width=8, fmt=".2f")
                vram_str = _format_num(out["peak_vram_mib"] if is_cuda else None, width=8, fmt=".1f")
                print(
                    f"{spec.name:<42} {res.steps:>5d} {res.skipped_batches:>4d} "
                    f"{out['wall_s']:>7.2f} {step_str} {out['tail_s']:>7.2f} {vram_str}  {' | '.join(notes)}",
                    flush=True,
                )

        if had_failure:
            writer.abort()
            print("\nerror: one or more reference scenarios failed; reference file not written", file=sys.stderr)
            return 1

        run_spec = _make_run_spec(
            world_size=args.world_size,
            device_type=parsed_dev.type,
            model_names=model_names,
            scenarios=scenarios,
            seed=args.seed,
            per_rank_batch=args.batch_size,
            per_rank_replay_batch=args.replay_batch_size,
            steps=args.steps,
            cpu_threads=cpu_threads,
            reference_mode=ref_mode,
        )
        env_info = {
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "device": device_str,
            "cuda_version": torch.version.cuda if is_cuda else None,
            "threads": torch.get_num_threads(),
        }
        writer.commit(spec=run_spec, env=env_info)
    except BaseException:
        writer.abort()
        raise

    size_mib = out_path.stat().st_size / (1024.0 * 1024.0)
    print(
        f"\nWrote {entry_count} reference entries for N={args.world_size} ({parsed_dev.type}, "
        f"{ref_mode_label}, {size_mib:.2f} MiB).",
        flush=True,
    )
    return 0


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

    # Guard against subtle weight-init or dataset drift between reference and multi-rank runs.
    if ref.get("init_sha256") != out["init_sha256"]:
        raise ValueError(f"{key}: initial trainable weights SHA-256 differs from reference")
    for fp_key in ("dom_fp", "val_fp", "rep_fp"):
        if ref.get(fp_key) != out[fp_key]:
            raise ValueError(f"{key}: dataset fingerprint {fp_key!r} differs from reference")

    init_native: Tensor = out["init_trainable_native"]
    final_native: Tensor = out["final_trainable_native"]
    if int(final_native.numel()) != int(ref["numel"]):
        raise ValueError(f"{key}: trainable numel {final_native.numel()} != reference {ref['numel']}")

    extra_notes: list[str] = []
    storage_kind = ref["storage_kind"]
    if out["final_sha256"] == ref["final_sha256"]:
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
            got_sketch, _, _, _ = _count_sketch_delta(
                init_native,
                final_native,
                sketch_dim=int(ref_tensor.numel()),
                seed=_sketch_seed(seed, model_name, spec.name),
            )
            diff_norm = float((got_sketch - ref_tensor.double()).norm().item())
            param_l2 = diff_norm / max(float(ref["final_norm_f64"]), 1e-300)
            upd_l2 = diff_norm / max(float(ref["update_norm_f64"]), 1e-300)
        else:
            raise ValueError(f"{key}: unknown reference storage_kind {storage_kind!r}")

    res: TrainingResult = out["result"]
    ref_hist = ref["train_history"]
    if (
        len(res.train_history) == len(ref_hist)
        and len(ref_hist) > 0
        and all(s_got == s_ref for (s_got, _), (s_ref, _) in zip(res.train_history, ref_hist, strict=True))
    ):
        loss_rel: float | None = max(
            _scalar_rel(l_got, l_ref) for (_, l_got), (_, l_ref) in zip(res.train_history, ref_hist, strict=True)
        )
    else:
        loss_rel = float("nan")
        extra_notes.append(f"ref_steps={ref['steps']}")

    val_rel: float | None = None
    if spec.with_val:
        ref_eval = ref["eval_history"]
        if (
            len(res.eval_history) == len(ref_eval)
            and len(ref_eval) > 0
            and all(s_got == s_ref for (s_got, _), (s_ref, _) in zip(res.eval_history, ref_eval, strict=True))
        ):
            val_rel = max(
                _scalar_rel(v_got, v_ref) for (_, v_got), (_, v_ref) in zip(res.eval_history, ref_eval, strict=True)
            )
        else:
            val_rel = float("nan")
    return param_l2, upd_l2, loss_rel, val_rel, extra_notes


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
    cpu_threads = _resolve_cpu_threads(args, is_cuda=is_cuda)

    if backend == "gloo" and device_map == "shared":
        if rank == 0:
            print("error: --device-map shared applies only to --backend nccl", file=sys.stderr)
        return 2

    if is_cuda:
        if not torch.cuda.is_available():
            if rank == 0:
                print("error: --backend nccl requires CUDA, but torch.cuda.is_available() is False", file=sys.stderr)
            return 2
        if device_map == "shared":
            if os.environ.get("NCCL_MULTI_RANK_GPU_ENABLE") != "1":
                if rank == 0:
                    print(
                        "error: --device-map shared on NCCL requires NCCL_MULTI_RANK_GPU_ENABLE=1 in the environment "
                        "(without it NCCL rejects multiple ranks on cuda:0 at init_process_group)",
                        file=sys.stderr,
                    )
                return 2
            config_device: str | None = "cuda:0"
            expected_cuda_index: int | None = 0
            binding_label = "shared"
            device_label = "cuda:0 (shared)"
        else:
            gpu_count = torch.cuda.device_count()
            if not (0 <= local_rank < gpu_count):
                print(
                    f"error: rank {rank} LOCAL_RANK={local_rank} out of range for {gpu_count} available GPU(s); "
                    "use --device-map shared with NCCL_MULTI_RANK_GPU_ENABLE=1 on a single GPU",
                    file=sys.stderr,
                )
                return 2
            # Pass device=None and do NOT call torch.cuda.set_device beforehand so TimestralTrainer's own
            # DistributedContext.detect() binds cuda:LOCAL_RANK live.
            config_device = None
            expected_cuda_index = local_rank
            binding_label = "per-rank"
            device_label = f"cuda:0..{world_size - 1}"
    else:
        config_device = "cpu"
        expected_cuda_index = None
        binding_label = "gloo"
        device_label = "cpu"

    model_names = (["tiny_random"] if not args.no_tiny else []) + list(args.sizes)
    if not model_names:
        if rank == 0:
            print("error: --no-tiny requires at least one --sizes entry", file=sys.stderr)
        return 2

    # Validate the tiny `spec.json` member on EVERY rank BEFORE entering `init_process_group`.
    # Every rank exits 2 cleanly on mismatch without entering a collective or touching GPU binding,
    # while only rank 0 opens the archive for per-scenario lazy tensor reads.
    ref_reader: _ReferenceReader | None = None
    ref_mode_label: str | None = None
    if args.reference is not None:
        ref_path = Path(args.reference)
        if not ref_path.is_file():
            if rank == 0:
                print(f"error: reference file not found: {ref_path}", file=sys.stderr)
            return 2
        try:
            ref_meta = _read_reference_spec(ref_path)
            ref_mode = _validate_reference_spec(
                ref_meta,
                world_size=world_size,
                device_type="cuda" if is_cuda else "cpu",
                model_names=model_names,
                scenarios=scenarios,
                seed=args.seed,
                per_rank_batch=args.batch_size,
                per_rank_replay_batch=args.replay_batch_size,
                steps=args.steps,
                cpu_threads=cpu_threads,
            )
            ref_mode_label = f"sketch(k={SKETCH_DIM})" if ref_mode == "sketch" else "exact"
        except Exception as exc:
            if rank == 0:
                print(f"error: refusing to compare against reference file: {exc}", file=sys.stderr)
            return 2
        if rank == 0:
            ref_reader = _ReferenceReader(ref_path)

    # In shared single-GPU mode, bind cuda:0 before init_process_group; in per-rank mode, leave
    # CUDA device binding to TimestralTrainer so the trainer's LOCAL_RANK path is genuinely exercised.
    if is_cuda and device_map == "shared":
        torch.cuda.set_device(0)

    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(seconds=args.timeout),
    )
    had_failure = False
    moved_state_cache: dict[str, dict[str, Tensor]] = {}
    try:
        models = _load_models(args.sizes, include_tiny=not args.no_tiny, seed=args.seed)
        if rank == 0:
            print(
                _env_banner(
                    mode_label="torchrun",
                    backend=backend,
                    world_size=world_size,
                    device_label=device_label,
                    device_map=device_map if is_cuda else None,
                    ref_mode_label=ref_mode_label,
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
            f"{'wall_s':>7} {'step_ms':>8} {'tail_s':>7} {'vram_mb':>8}  notes"
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
                        moved_state_cache=moved_state_cache,
                    )
                except Exception as exc:
                    had_failure = True
                    print(
                        f"[rank {rank}] ERROR in {model_name}:{spec.name}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    traceback.print_exc(file=sys.stderr)
                    # Abort immediately so ranks cannot desynchronise on subsequent scenarios.
                    raise

                # Exit non-zero only on cross-rank mismatch in a deterministic scenario or export failure.
                if (spec.deterministic and (not out["xrank_exact"] or not out["result_eq"])) or (
                    spec.test_export and not out["export_all_ok"]
                ):
                    had_failure = True

                # Rank 0 compares against reference; broadcast a 1-scalar error flag so any reference
                # content mismatch (e.g. init_sha256 or dataset fingerprint drift) aborts all ranks in lockstep.
                ref_error_flag = torch.zeros(1, dtype=torch.int64, device=out["device"])
                if rank == 0:
                    try:
                        param_l2, upd_l2, loss_rel, val_rel, extra_notes = _compare_to_reference(
                            out,
                            ref_reader,
                            model_name=model_name,
                            spec=spec,
                            seed=args.seed,
                        )
                    except Exception as exc:
                        ref_error_flag.fill_(1)
                        print(f"error: reference comparison failed: {exc}", file=sys.stderr, flush=True)
                    else:
                        res: TrainingResult = out["result"]
                        notes: list[str] = []
                        if spec.static_note:
                            notes.append(spec.static_note)
                        if not out["xrank_exact"]:
                            notes.append("XRANK_PARAM_MISMATCH")
                        if not out["result_eq"]:
                            notes.append("RESULT_MISMATCH")
                        if spec.with_val:
                            notes.append(
                                f"early={res.stopped_early} restored={res.restored_best} best_step={res.best_step}"
                            )
                        if spec.compile_model:
                            recomp = "yes" if out["extra_graphs_max"] > 0 else "no"
                            notes.append(f"graphs={out['graphs_max']} recompiled={recomp}")
                        if spec.test_export:
                            cnt = out["export_pass_count"]
                            notes.append(f"export={'ok' if out['export_all_ok'] else 'FAIL'}({cnt}/{world_size})")
                        if not spec.compare_ref:
                            notes.append("xrank_only")
                        notes.extend(extra_notes)

                        step_str = _format_num(out["step_max_ms"], width=8, fmt=".2f")
                        vram_str = _format_num(out["vram_max_mib"] if is_cuda else None, width=8, fmt=".1f")
                        print(
                            f"{spec.name:<42} {res.steps:>5d} {res.skipped_batches:>4d} "
                            f"{'yes' if out['xrank_exact'] else 'no':>5} {out['xrank_max_abs']:>9.2e} "
                            f"{_format_num(param_l2)} {_format_num(upd_l2)} "
                            f"{_format_num(loss_rel)} {_format_num(val_rel)} "
                            f"{out['wall_max_s']:>7.2f} {step_str} {out['tail_max_s']:>7.2f} {vram_str}  "
                            f"{' | '.join(notes)}",
                            flush=True,
                        )

                dist.broadcast(ref_error_flag, src=0)
                if int(ref_error_flag.item()) != 0:
                    return 2
    finally:
        if ref_reader is not None:
            ref_reader.close()
        if dist.is_initialized():
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
        choices=("sketch", "exact"),
        default="sketch",
        help=(
            f"Reference vector storage for --write-reference: 'sketch' (default; exact when numel <= {SKETCH_DIM}, "
            f"else deterministic float64 CountSketch of dim {SKETCH_DIM} = 64 KiB/scenario) or 'exact' "
            "(full native-dtype trainable vectors streamed to zip)."
        ),
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to a reference archive written by --write-reference (validated before init, loaded on rank 0).",
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
        help="NCCL device mapping: per-rank (device=None -> cuda:LOCAL_RANK) or shared (cuda:0 on all ranks).",
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
