"""Shared helpers for the Stage 3 multi-process measurement probes.

Used by `scripts/probes/distributed_trainer_check.py` and
`scripts/probes/distributed_rank_failure_check.py`. Imports only standard library,
PyTorch/NumPy, `_common`, and public `timestral_tune` APIs (never `tests/`).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from _common import PATCH

from timestral_tune.data import ColumnRoles, SeriesData, TimeSeriesDataset

CTX_PATCHES = 2
HOR_PATCHES = 2
DOM_HORIZON = HOR_PATCHES * PATCH  # 64
REP_HORIZON = PATCH + PATCH // 2  # 48 (non-multiple of PATCH; exercises distillation mask padding)

# macOS AF_UNIX sun_path limit is 103 bytes (107 on Linux); torch_shm_manager reads raw getenv("TMPDIR")
# (falling back to "/tmp") without stripping a trailing slash and opens
# `<TMPDIR>/torch-shm-dir-XXXXXX/manager.sock`. If this overflows inside a DataLoader worker's
# Queue._feed background thread, Queue._feed catches the RuntimeError, drops the batch, and leaves
# workers alive while DataLoader waits forever.
_SHM_SUFFIX_LEN = len("/torch-shm-dir-XXXXXX/manager.sock")
_UNIX_PATH_MAX = 103 if sys.platform == "darwin" else 107


def pick_free_port() -> int:
    """Bind an ephemeral IPv4 loopback socket and return its free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON to `path` via a sibling `.tmp` file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def describe_exitcode(exitcode: int | None) -> str:
    """Format a subprocess exit code as `running`, `signal:SIG*`, or `exit:N`."""
    if exitcode is None:
        return "running"
    if exitcode < 0:
        sig_num = -exitcode
        try:
            sig_name = signal.Signals(sig_num).name
        except ValueError:
            sig_name = f"SIG{sig_num}"
        return f"signal:{sig_name}"
    return f"exit:{exitcode}"


def check_tmpdir_for_workers() -> None:
    """Fail fast before spawning DataLoader workers if TMPDIR overflows torch_shm_manager's sun_path."""
    try:
        sharing = torch.multiprocessing.get_sharing_strategy()
    except Exception:
        sharing = "file_system" if sys.platform == "darwin" else "file_descriptor"
    if sharing != "file_system":
        return
    raw_tmp = os.environ.get("TMPDIR") or "/tmp"
    raw_bytes = len(raw_tmp.encode("utf-8"))
    sock_len = raw_bytes + _SHM_SUFFIX_LEN
    if sock_len > _UNIX_PATH_MAX:
        raise RuntimeError(
            f"TMPDIR length ({raw_bytes} bytes) makes torch_shm_manager's Unix socket path "
            f"{sock_len} bytes (> {_UNIX_PATH_MAX}-byte limit), which silently deadlocks DataLoader "
            f"workers inside Queue._feed; re-run with TMPDIR=/tmp"
        )


def make_synthetic_dataset(
    num_windows: int,
    *,
    horizon: int = DOM_HORIZON,
    covariates: bool = False,
    seed: int = 0,
    future_ramp: float = 0.0,
    drop_tail_targets: bool = True,
) -> TimeSeriesDataset:
    """Build a deterministic self-contained `TimeSeriesDataset` for multi-rank probes."""
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
        if drop_tail_targets:
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


def validate_distributed_device_map(
    *,
    backend: Literal["gloo", "nccl"],
    device_map: Literal["per-rank", "shared"],
    world_size: int,
    local_rank: int | None = None,
) -> str | None:
    """Validate `(backend, device_map, world_size, local_rank)` uniformly on every rank before init.

    Returns an actionable error string on conflict, or `None` when valid. Checking `world_size > gpu_count`
    on every rank ensures no subset of ranks enters `init_process_group` when GPUs are insufficient.
    """
    if backend == "gloo":
        if device_map == "shared":
            return "--device-map shared applies only to --backend nccl"
        return None

    if not torch.cuda.is_available():
        return "--backend nccl requires CUDA, but torch.cuda.is_available() is False"
    if device_map == "shared":
        if os.environ.get("NCCL_MULTI_RANK_GPU_ENABLE") != "1":
            return (
                "--device-map shared on NCCL requires NCCL_MULTI_RANK_GPU_ENABLE=1 in the environment "
                "(without it NCCL rejects multiple ranks on cuda:0 at init_process_group)"
            )
        return None

    gpu_count = torch.cuda.device_count()
    if world_size > gpu_count:
        return (
            f"--device-map per-rank needs {world_size} GPUs, found {gpu_count}; "
            "use --device-map shared with NCCL_MULTI_RANK_GPU_ENABLE=1 on 1 GPU"
        )
    if local_rank is not None and not (0 <= local_rank < gpu_count):
        return (
            f"LOCAL_RANK={local_rank} out of range for {gpu_count} available GPU(s); "
            "use --device-map shared with NCCL_MULTI_RANK_GPU_ENABLE=1 on 1 GPU"
        )
    return None


def _nvidia_smi_first_gpu_name() -> str | None:
    """Query the first visible GPU name via `nvidia-smi` without creating a CUDA context."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    return lines[0] if lines else None


def cuda_env_banner_parts(*, allow_cuda_context: bool = True) -> list[str]:
    """Return CUDA/NCCL/GPU header fragments for probe banners.

    When `allow_cuda_context=False` (used by parent launcher processes that spawn CUDA children),
    avoids calling `torch.cuda.get_device_name(0)` so the parent never allocates a CUDA context.
    """
    parts: list[str] = [f"cuda={torch.version.cuda}"]
    nccl_ver = torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else None
    if nccl_ver is not None:
        parts.append(f"nccl={'.'.join(str(x) for x in nccl_ver)}")
    if allow_cuda_context:
        parts.append(f"gpu={torch.cuda.get_device_name(0)}")
    else:
        smi_name = _nvidia_smi_first_gpu_name()
        if smi_name is not None:
            parts.append(f"gpu={smi_name}")
    parts.append(f"gpu_count={torch.cuda.device_count()}")
    return parts
