"""Reusable multi-process test harness for `torch.distributed` scenarios (CPU/`gloo` default, opt-in CUDA/`nccl`).

Spawns real separate Python processes using `torch.multiprocessing`'s `spawn` start method and
coordinates rendezvous through a file-based store (`init_method="file://..."`) inside a fresh
temporary directory, avoiding ephemeral TCP port collisions.

Backend and device-mapping selection (opt-in via environment variables; CI default is CPU/`gloo`):
- `TIMESTRAL_TEST_DIST_BACKEND`: `"gloo"` (default when unset) or `"nccl"`.
- `TIMESTRAL_TEST_DIST_DEVICE_MAP`: `"per-rank"` (default; one GPU per rank via `LOCAL_RANK` and
  `DistributedContext.detect("auto")`) or `"shared"` (development-only single-GPU mode, all ranks on
  `cuda:0`, requiring `TIMESTRAL_TEST_DIST_BACKEND=nccl` and `NCCL_MULTI_RANK_GPU_ENABLE=1`).

Design guarantees:
- Each worker uses 1 intra-op CPU thread, initialises the process group with a configurable timeout,
  invokes the target module-level function as `fn(rank, world_size, *args)`, records either its
  picklable return value or its exception type/message/traceback to a per-rank file, and always
  destroys the process group in `finally`. On CUDA/`nccl` backends, each spawned worker starts in its
  own session (`os.setsid`) so timeout/crash teardown can signal its entire process group; CPU/`gloo`
  keeps workers in the parent's process group (unchanged default).
- On CUDA/`nccl` with `device_map="per-rank"`, `_worker_main` sets `LOCAL_RANK=str(rank)` and does NOT
  pre-bind `torch.cuda.set_device(...)`, so `DistributedContext.detect("auto")` exercises the real
  `cuda:LOCAL_RANK` binding path.
- One worker raising a Python exception never kills sibling workers prematurely; every rank finishes
  and reports what it saw (essential for testing failure propagation and consistency checks).
- The parent enforces a hard wall-clock deadline (and on CUDA/`nccl`, a 5 s crash grace when a worker
  exits abnormally via signal or non-zero exit code) and terminates/kills live workers so GPU
  processes do not leak on failure.
- `run_scenarios` amortises process-spawn overhead by running multiple named scenarios sequentially
  inside a single spawned process group.
"""

from __future__ import annotations

import datetime
import math
import os
import pickle
import signal
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

__all__ = [
    "DIST_BACKEND_ENV",
    "DIST_DEVICE_MAP_ENV",
    "DistributedTimeoutError",
    "EVAL_TOL",
    "F32_TOL_SCALE",
    "F32_TOL_SCALE_ENV",
    "RankError",
    "RankResult",
    "STEP_PARAM_TOL",
    "STEP_TOL",
    "TRAJ_TOL",
    "TRAJ_UPDATE_TOL",
    "active_dist_device",
    "barrier",
    "dist_backend",
    "dist_device_map",
    "expected_worker_device",
    "f32_tol_scale",
    "is_cuda_backend",
    "is_cuda_dist_mode",
    "reference_device_for_mode",
    "run_distributed",
    "run_scenarios",
    "scale_timeout",
    "trainer_device_for_mode",
    "unwrap_results",
    "validate_cuda_harness_prereqs",
]

_ALLOWED_CPU_BACKENDS = frozenset({"gloo", "cpu:gloo"})
_ALLOWED_CUDA_BACKENDS = frozenset({"nccl", "cuda:nccl"})
_ALLOWED_BACKENDS = _ALLOWED_CPU_BACKENDS | _ALLOWED_CUDA_BACKENDS
_VALID_ENV_BACKENDS = frozenset({"gloo", "nccl"})
_VALID_DEVICE_MAPS = frozenset({"per-rank", "shared"})

# Opt-in environment switches for running the multi-process test suite on CUDA/NCCL.
# Unset defaults to CPU/gloo ("per-rank"), leaving CI and local CPU runs unchanged.
DIST_BACKEND_ENV = "TIMESTRAL_TEST_DIST_BACKEND"
DIST_DEVICE_MAP_ENV = "TIMESTRAL_TEST_DIST_DEVICE_MAP"

# Inside `_worker_main`, tracks the backend and device map actually passed to `run_distributed`
# so explicit per-test backend overrides (such as `backend="cpu:gloo"`) stay isolated from outer
# environment variables without mutating `os.environ`.
_ACTIVE_WORKER_BACKEND: str | None = None
_ACTIVE_WORKER_DEVICE_MAP: str | None = None

# Environment knob multiplying every float32 parity tolerance in the multi-process test suite
# (float64 tiers are never scaled). A run that passes at 0.1 has at least 10x headroom on that machine;
# at 0.2, at least 5x. Values above 1 loosen the tiers, for exploring a new platform. Must be a finite
# positive float.
F32_TOL_SCALE_ENV = "TIMESTRAL_TEST_F32_TOL_SCALE"


def is_cuda_backend(backend: str) -> bool:
    """Return True if `backend` is a CUDA/NCCL backend accepted by the harness."""
    return backend in _ALLOWED_CUDA_BACKENDS


def dist_backend(env_var: str = DIST_BACKEND_ENV) -> str:
    """Return the active distributed test backend (`'gloo'` default, or `'nccl'`).

    Inside a spawned worker executing `_worker_main`, returns that worker's bound backend when
    `env_var` is `DIST_BACKEND_ENV`. Otherwise reads and validates `os.environ[env_var]`.
    """
    if env_var == DIST_BACKEND_ENV and _ACTIVE_WORKER_BACKEND is not None:
        return _ACTIVE_WORKER_BACKEND
    raw = os.environ.get(env_var)
    if raw is None:
        return "gloo"
    value = raw.strip()
    if value not in _VALID_ENV_BACKENDS:
        raise ValueError(f"{env_var} must be one of {sorted(_VALID_ENV_BACKENDS)}, got {raw!r}")
    return value


def dist_device_map(env_var: str = DIST_DEVICE_MAP_ENV, *, backend: str | None = None) -> str:
    """Return the active CUDA device-mapping mode (`'per-rank'` default, or `'shared'`).

    Rejects `'shared'` when paired with a CPU/`gloo` backend so single-GPU debug mode cannot be
    silently ignored.
    """
    if env_var == DIST_DEVICE_MAP_ENV and backend is None and _ACTIVE_WORKER_DEVICE_MAP is not None:
        return _ACTIVE_WORKER_DEVICE_MAP
    resolved_backend = dist_backend() if backend is None else backend
    raw = os.environ.get(env_var)
    if raw is None:
        mode = "per-rank"
    else:
        mode = raw.strip()
        if mode not in _VALID_DEVICE_MAPS:
            raise ValueError(f"{env_var} must be one of {sorted(_VALID_DEVICE_MAPS)}, got {raw!r}")
    if mode == "shared" and not is_cuda_backend(resolved_backend):
        raise ValueError(
            f"{env_var}='shared' requires a CUDA/NCCL backend ({DIST_BACKEND_ENV}='nccl'); "
            f"got backend={resolved_backend!r}"
        )
    return mode


def is_cuda_dist_mode() -> bool:
    """Return True when the active multi-process test mode targets CUDA/NCCL."""
    return is_cuda_backend(dist_backend())


def trainer_device_for_mode() -> str:
    """Return the `TrainingConfig.device` / `DistributedContext.detect(...)` argument for workers.

    - CPU/`gloo`: `'cpu'` (unchanged default).
    - CUDA/`nccl` with `'per-rank'`: `'auto'` so `DistributedContext.detect('auto')` performs real
      `LOCAL_RANK` -> `cuda:<rank>` binding.
    - CUDA/`nccl` with `'shared'`: `'cuda:0'` so all ranks explicitly share GPU 0.
    """
    if not is_cuda_dist_mode():
        return "cpu"
    if dist_device_map() == "shared":
        return "cuda:0"
    return "auto"


def reference_device_for_mode() -> str:
    """Return the device string for one-process references (`'cpu'` or `'cuda:0'`).

    Fails loudly with `RuntimeError` if CUDA/NCCL mode is active without an available CUDA GPU.
    """
    if not is_cuda_dist_mode():
        return "cpu"
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            f"CUDA/NCCL multi-process tests requested ({DIST_BACKEND_ENV}={dist_backend()!r}), "
            "but no CUDA GPU is available for the one-process reference"
        )
    return "cuda:0"


def expected_worker_device(rank: int) -> torch.device:
    """Return the `torch.device` expected for `rank` under the active backend and device map."""
    if not is_cuda_dist_mode():
        return torch.device("cpu")
    if dist_device_map() == "shared":
        return torch.device("cuda", 0)
    return torch.device("cuda", rank)


def _default_cuda_device_index() -> int:
    """Return this rank's target CUDA device index (0 in `shared` mode, `rank` in `per-rank` mode)."""
    if _ACTIVE_WORKER_DEVICE_MAP == "shared":
        return 0
    return int(dist.get_rank())


def _ensure_cuda_device(idx: int) -> None:
    if not torch.cuda.is_initialized() or torch.cuda.current_device() != idx:
        torch.cuda.set_device(idx)


def active_dist_device() -> torch.device:
    """Return the device to place ad-hoc collective tensors on inside a running worker.

    On CUDA/NCCL, always resolves to this rank's own target GPU (`0` in `shared` mode, `rank` in
    `per-rank` mode) and calls `torch.cuda.set_device(idx)` if needed, never falling back to GPU 0
    on non-zero ranks even if CUDA was already lazily initialised on device 0 before `detect()`.
    """
    if dist.is_initialized() and is_cuda_backend(dist.get_backend()):
        idx = _default_cuda_device_index()
        _ensure_cuda_device(idx)
        return torch.device("cuda", idx)
    return torch.device("cpu")


def barrier(device: torch.device | None = None) -> None:
    """Run `dist.barrier`, passing explicit `device_ids` on NCCL to avoid device-0 fallback."""
    if not dist.is_initialized():
        return
    if is_cuda_backend(dist.get_backend()):
        if device is not None and device.type == "cuda":
            idx = 0 if device.index is None else device.index
        else:
            idx = _default_cuda_device_index()
        _ensure_cuda_device(idx)
        dist.barrier(device_ids=[idx])
    else:
        dist.barrier()


def scale_timeout(cpu_seconds: float, nccl_seconds: float) -> float:
    """Return `nccl_seconds` in CUDA/NCCL mode and `cpu_seconds` unchanged in CPU/`gloo` mode."""
    return nccl_seconds if is_cuda_dist_mode() else cpu_seconds


def validate_cuda_harness_prereqs(
    world_size: int,
    *,
    backend: str | None = None,
    device_map: str | None = None,
) -> None:
    """Validate hardware and environment prerequisites before spawning CUDA/NCCL workers.

    Raises `ValueError` on configuration conflicts (e.g. `device_map='shared'` without
    `NCCL_MULTI_RANK_GPU_ENABLE=1`) and `RuntimeError` when CUDA is unavailable or too few GPUs
    are visible. Never skips silently.
    """
    resolved_backend = dist_backend() if backend is None else backend
    if not is_cuda_backend(resolved_backend):
        return
    resolved_map = dist_device_map(backend=resolved_backend) if device_map is None else device_map
    if resolved_map not in _VALID_DEVICE_MAPS:
        raise ValueError(f"device_map must be one of {sorted(_VALID_DEVICE_MAPS)}, got {resolved_map!r}")

    if resolved_map == "shared" and os.environ.get("NCCL_MULTI_RANK_GPU_ENABLE") != "1":
        raise ValueError(
            f"{DIST_DEVICE_MAP_ENV}='shared' requires NCCL_MULTI_RANK_GPU_ENABLE=1 "
            "(development-only single-GPU mode; distinct from per-rank multi-GPU verification)"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA/NCCL multi-process run requested (backend={resolved_backend!r}, "
            f"device_map={resolved_map!r}, world_size={world_size}), "
            "but torch.cuda.is_available() is False"
        )

    gpu_count = torch.cuda.device_count()
    if resolved_map == "shared":
        if gpu_count < 1:
            raise RuntimeError(f"CUDA/NCCL shared-GPU run requires at least 1 visible CUDA GPU, found {gpu_count}")
        return

    if gpu_count < world_size:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        raise RuntimeError(
            f"CUDA/NCCL multi-process run with device_map='per-rank' requires at least {world_size} "
            f"visible CUDA GPUs for world_size={world_size}, found {gpu_count} "
            f"(CUDA_VISIBLE_DEVICES={cvd!r}). Run on a host with >= {world_size} GPUs, filter to "
            f"smaller world sizes (e.g. scripts/run_t15_nccl.py --ranks 2), or set "
            f"{DIST_DEVICE_MAP_ENV}=shared with NCCL_MULTI_RANK_GPU_ENABLE=1 for single-GPU debugging."
        )


def f32_tol_scale(env_var: str = F32_TOL_SCALE_ENV) -> float:
    """Read and validate the float32 tolerance headroom multiplier from `env_var` (default 1.0)."""
    raw = os.environ.get(env_var)
    if raw is None:
        return 1.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be a finite positive float, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{env_var} must be a finite positive float, got {raw!r}")
    return value


F32_TOL_SCALE: float = f32_tol_scale()

# Shared parity tolerance tiers for all multi-process tests. The float64 tiers check the logic and are
# never scaled. The float32 tiers are calibrated against measured round-off, which depends on the
# platform: on macOS arm64 one-process `fit()` is bit-identical across intra-op thread counts, while on
# Linux x86_64 it is not (`scripts/probes/float32_trajectory_noise.py`). Headroom is checked by running
# the four multi-process test files with `TIMESTRAL_TEST_F32_TOL_SCALE` below 1 (`docs/VERIFICATION.md`
# §17): they pass at 0.1 on macOS arm64, and on Linux x86_64 at 0.2 with 1, 2 and 4 intra-op threads.
#
# - EVAL_TOL: sharded validation loss at identical weights.
# - STEP_TOL: loss and gradients of one direct `_run_step`.
# - STEP_PARAM_TOL: parameters after that one step. Split from STEP_TOL because AdamW divides by
#   `sqrt(v) + eps`, so gradient components at float32 round-off level become parameter differences
#   orders of magnitude larger in relative terms (3.4e-06 was observed on Linux x86_64).
# - TRAJ_TOL: step losses, validation losses and `final_trainable` relative L2 over a multi-step `fit()`.
# - TRAJ_UPDATE_TOL: `param_update = final - init` relative L2 over a multi-step `fit()`. This is 1e-3 in
#   float32, not 1e-4, because the update is much smaller than the parameters for LoRA and the same
#   AdamW amplification applies at every step: with the tier at 1e-4, Linux x86_64 runs reached 1.9e-04
#   against the one-process reference, and the one-process reference differed from itself by 1.1e-04
#   between 1 and 2 intra-op threads on a T15(j) `lora_variate` distillation scenario.
EVAL_TOL: Mapping[str, float] = MappingProxyType({"float64": 1e-12, "float32": 1e-5 * F32_TOL_SCALE})
STEP_TOL: Mapping[str, float] = MappingProxyType({"float64": 1e-12, "float32": 1e-5 * F32_TOL_SCALE})
STEP_PARAM_TOL: Mapping[str, float] = MappingProxyType({"float64": 1e-12, "float32": 1e-4 * F32_TOL_SCALE})
TRAJ_TOL: Mapping[str, float] = MappingProxyType({"float64": 1e-10, "float32": 1e-4 * F32_TOL_SCALE})
TRAJ_UPDATE_TOL: Mapping[str, float] = MappingProxyType({"float64": 1e-10, "float32": 1e-3 * F32_TOL_SCALE})


class DistributedTimeoutError(TimeoutError):
    """Raised when a multi-process test exceeds its wall-clock deadline."""


@dataclass(frozen=True)
class RankError:
    """Captured exception details returned from a worker rank."""

    exc_type: type[BaseException] | None
    exc_type_name: str
    message: str
    traceback: str
    exception: BaseException | None = None


@dataclass(frozen=True)
class RankResult:
    """Outcome produced by a single worker rank."""

    rank: int
    value: Any = None
    error: RankError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> Any:
        """Return `self.value`, or raise `RuntimeError` with the worker's traceback if it failed."""
        if self.error is not None:
            raise RuntimeError(
                f"rank {self.rank} raised {self.error.exc_type_name}: {self.error.message}\n{self.error.traceback}"
            )
        return self.value


def unwrap_results(results: Sequence[RankResult]) -> list[Any]:
    """Return `[r.value for r in results]`, or raise `RuntimeError` reporting every failing rank."""
    failures = [r for r in results if not r.ok]
    if failures:
        parts = []
        for r in failures:
            assert r.error is not None
            parts.append(f"--- rank {r.rank} ({r.error.exc_type_name}: {r.error.message}) ---\n{r.error.traceback}")
        raise RuntimeError(f"{len(failures)}/{len(results)} rank(s) failed:\n" + "\n".join(parts))
    return [r.value for r in results]


def _capture_error(exc: BaseException) -> RankError:
    exc_type: type[BaseException] | None = type(exc)
    try:
        pickle.dumps(exc_type)
    except Exception:
        exc_type = None

    pickled_exc: BaseException | None = exc
    try:
        pickle.dumps(exc)
    except Exception:
        pickled_exc = None

    return RankError(
        exc_type=exc_type,
        exc_type_name=type(exc).__qualname__,
        message=str(exc),
        traceback=traceback.format_exc(),
        exception=pickled_exc,
    )


def _write_result(path: Path, result: RankResult) -> None:
    tmp_path = path.with_suffix(".pkl.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _worker_main(
    rank: int,
    world_size: int,
    store_path: str,
    result_path: str,
    pg_timeout: float,
    backend: str,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    device_map: str = "per-rank",
    new_session: bool = False,
) -> None:
    """Top-level worker entry point for `spawn`."""
    global _ACTIVE_WORKER_BACKEND, _ACTIVE_WORKER_DEVICE_MAP

    if new_session and hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass

    prev_threads = torch.get_num_threads()
    prev_backend = _ACTIVE_WORKER_BACKEND
    prev_device_map = _ACTIVE_WORKER_DEVICE_MAP
    cuda_mode = is_cuda_backend(backend)
    prev_local_rank = os.environ.get("LOCAL_RANK") if cuda_mode else None

    torch.set_num_threads(1)
    _ACTIVE_WORKER_BACKEND = backend
    _ACTIVE_WORKER_DEVICE_MAP = device_map
    if cuda_mode:
        # Expose each spawned rank's local GPU index so DistributedContext.detect("auto")
        # exercises its real LOCAL_RANK -> cuda:<rank> binding path without pre-binding.
        os.environ["LOCAL_RANK"] = str(rank)

    outcome: RankResult | None = None
    try:
        try:
            if cuda_mode and device_map == "shared":
                # Shared-GPU dev mode explicitly binds every rank to GPU 0.
                # Per-rank mode intentionally does NOT call torch.cuda.set_device here so
                # DistributedContext.detect("auto") performs real cuda:LOCAL_RANK binding.
                torch.cuda.set_device(0)
            dist.init_process_group(
                backend=backend,
                init_method=f"file://{store_path}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=pg_timeout),
            )
            try:
                value = fn(rank, world_size, *args)
                outcome = RankResult(rank=rank, value=value)
            except BaseException as exc:
                outcome = RankResult(rank=rank, error=_capture_error(exc))
            finally:
                if dist.is_initialized():
                    try:
                        if cuda_mode and outcome is not None and outcome.ok and torch.cuda.is_initialized():
                            torch.cuda.synchronize()
                        dist.destroy_process_group()
                    except BaseException as cleanup_exc:
                        if outcome is None or outcome.ok:
                            outcome = RankResult(rank=rank, error=_capture_error(cleanup_exc))
        except BaseException as exc:
            if outcome is None or outcome.ok:
                outcome = RankResult(rank=rank, error=_capture_error(exc))
        assert outcome is not None
        _write_result(Path(result_path), outcome)
    finally:
        if cuda_mode:
            if prev_local_rank is None:
                os.environ.pop("LOCAL_RANK", None)
            else:
                os.environ["LOCAL_RANK"] = prev_local_rank
        _ACTIVE_WORKER_BACKEND = prev_backend
        _ACTIVE_WORKER_DEVICE_MAP = prev_device_map
        torch.set_num_threads(prev_threads)


def _signal_label(sig_num: int) -> str:
    try:
        return f"{signal.Signals(sig_num).name} ({sig_num})"
    except ValueError:
        return f"signal {sig_num}"


def _signal_worker_group(proc: BaseProcess, sig: signal.Signals) -> None:
    """Signal the worker's own process group (when it is a session leader) or the worker itself."""
    pid = proc.pid
    if pid is not None and hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, sig)
                return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        if sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.terminate()
    except (ProcessLookupError, OSError):
        pass


def _terminate_workers(
    procs: Sequence[BaseProcess],
    *,
    grace_s: float = 0.5,
    signal_groups: bool = False,
) -> None:
    for proc in procs:
        if proc.is_alive():
            if signal_groups:
                _signal_worker_group(proc, signal.SIGTERM)
            else:
                proc.terminate()
    grace_deadline = time.monotonic() + grace_s
    for proc in procs:
        if proc.is_alive():
            remaining = max(0.0, grace_deadline - time.monotonic())
            proc.join(timeout=remaining)
    for proc in procs:
        if proc.is_alive():
            if signal_groups:
                _signal_worker_group(proc, signal.SIGKILL)
            else:
                proc.kill()
    for proc in procs:
        proc.join(timeout=2.0)


def run_distributed(
    fn: Callable[..., Any],
    world_size: int,
    args: Sequence[Any] = (),
    *,
    pg_timeout: float = 5.0,
    wall_timeout: float = 30.0,
    backend: str | None = None,
    device_map: str | None = None,
) -> list[RankResult]:
    """Run module-level `fn(rank, world_size, *args)` on `world_size` spawned processes.

    When `backend` is `None`, resolves via `dist_backend()` (CPU/`gloo` default; opt-in CUDA/`nccl`
    via `TIMESTRAL_TEST_DIST_BACKEND=nccl`). When `device_map` is `None`, resolves via
    `dist_device_map()` for CUDA backends (`'per-rank'` default or `'shared'`) and `'per-rank'`
    for CPU backends.

    Returns a list of `RankResult` indexed by rank `0..world_size-1`. Workers raising Python
    exceptions are recorded in `RankResult.error` without aborting sibling ranks. Raises
    `DistributedTimeoutError` if `wall_timeout` elapses before all ranks finish, or `RuntimeError`
    if CUDA prerequisites fail or any worker exits via a signal or abnormal non-zero exit code.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    resolved_backend = dist_backend() if backend is None else backend
    if resolved_backend not in _ALLOWED_BACKENDS:
        raise ValueError(f"harness supports backends {sorted(_ALLOWED_BACKENDS)} only; got {resolved_backend!r}")
    cuda_mode = is_cuda_backend(resolved_backend)
    if device_map is not None:
        if device_map not in _VALID_DEVICE_MAPS:
            raise ValueError(f"device_map must be one of {sorted(_VALID_DEVICE_MAPS)}, got {device_map!r}")
        if device_map == "shared" and not cuda_mode:
            raise ValueError(f"device_map='shared' requires a CUDA/NCCL backend; got backend={resolved_backend!r}")
        resolved_device_map = device_map
    elif cuda_mode:
        resolved_device_map = dist_device_map(backend=resolved_backend)
    else:
        resolved_device_map = "per-rank"

    if pg_timeout <= 0.0:
        raise ValueError(f"pg_timeout must be > 0, got {pg_timeout}")
    if wall_timeout <= 0.0:
        raise ValueError(f"wall_timeout must be > 0, got {wall_timeout}")

    validate_cuda_harness_prereqs(world_size, backend=resolved_backend, device_map=resolved_device_map)

    ctx = mp.get_context("spawn")
    args_tuple = tuple(args)
    # Keep CPU/gloo process semantics identical to the original harness (workers remain in the
    # parent's process group, no crash-grace early abort, 0.5s termination grace). Enable
    # per-worker session isolation, PGID teardown, and 5s crash grace only for CUDA/NCCL so
    # dead GPU ranks cannot stall long NCCL wall timeouts or leak GPU/DataLoader processes.
    term_grace = 1.5 if cuda_mode else 0.5

    with tempfile.TemporaryDirectory(prefix="timestral_dist_") as tmpdir:
        tmp = Path(tmpdir)
        store_path = str(tmp / "rendezvous")
        result_paths = [tmp / f"rank_{rank}.pkl" for rank in range(world_size)]
        procs = [
            ctx.Process(
                target=_worker_main,
                args=(
                    rank,
                    world_size,
                    store_path,
                    str(result_paths[rank]),
                    pg_timeout,
                    resolved_backend,
                    fn,
                    args_tuple,
                    resolved_device_map,
                    cuda_mode,
                ),
                name=f"dist-rank-{rank}",
            )
            for rank in range(world_size)
        ]

        try:
            for proc in procs:
                proc.start()

            deadline = time.monotonic() + wall_timeout
            crash_deadline: float | None = None
            while True:
                alive = [proc for proc in procs if proc.is_alive()]
                if not alive:
                    break
                now = time.monotonic()
                # On CUDA/NCCL, if a worker died abnormally (signal or non-zero exit code; caught
                # Python exceptions exit 0), cap remaining wait so siblings blocked in NCCL are
                # torn down promptly instead of waiting out multi-minute wall timeouts.
                if (
                    cuda_mode
                    and crash_deadline is None
                    and any(proc.exitcode is not None and proc.exitcode != 0 for proc in procs)
                ):
                    crash_deadline = min(deadline, now + 5.0)
                effective_deadline = deadline if crash_deadline is None else crash_deadline
                remaining = effective_deadline - now
                if remaining <= 0.0:
                    break
                alive[0].join(timeout=min(0.05, remaining))

            unfinished = [rank for rank, proc in enumerate(procs) if proc.is_alive()]
            if unfinished:
                crashed_before_kill = [
                    (rank, proc.exitcode)
                    for rank, proc in enumerate(procs)
                    if not proc.is_alive() and proc.exitcode is not None and proc.exitcode != 0
                ]
                _terminate_workers(procs, grace_s=term_grace, signal_groups=cuda_mode)
                if cuda_mode and crashed_before_kill and crash_deadline is not None and crash_deadline < deadline:
                    parts = []
                    for rank, code in crashed_before_kill:
                        if code < 0:
                            parts.append(f"rank {rank}: terminated by signal {_signal_label(-code)}")
                        else:
                            parts.append(f"rank {rank}: exited with code {code}")
                    parts.append(f"terminated surviving ranks: {unfinished}")
                    raise RuntimeError("worker process failure: " + "; ".join(parts))
                raise DistributedTimeoutError(
                    f"distributed run timed out after {wall_timeout:.2f}s; "
                    f"unfinished ranks: {unfinished} (world_size={world_size})"
                )

            for proc in procs:
                proc.join()

            abnormal: list[str] = []
            for rank, proc in enumerate(procs):
                code = proc.exitcode
                if code is None:
                    abnormal.append(f"rank {rank}: did not report an exit code")
                elif code < 0:
                    abnormal.append(f"rank {rank}: terminated by signal {_signal_label(-code)}")
                elif code > 0:
                    abnormal.append(f"rank {rank}: exited with code {code}")
            if abnormal:
                raise RuntimeError("worker process failure: " + "; ".join(abnormal))

            results: list[RankResult] = []
            for rank, path in enumerate(result_paths):
                if not path.exists():
                    raise RuntimeError(f"rank {rank} exited with code 0 but wrote no result file")
                with path.open("rb") as handle:
                    results.append(pickle.load(handle))
            return results
        finally:
            if any(proc.is_alive() for proc in procs):
                _terminate_workers(procs, grace_s=term_grace, signal_groups=cuda_mode)


def _scenario_runner(
    rank: int,
    world_size: int,
    scenarios: Sequence[tuple[str, Callable[..., Any], tuple[Any, ...]]],
) -> dict[str, RankResult]:
    """Run named scenarios sequentially inside one process group, capturing per-scenario outcomes."""
    outcomes: dict[str, RankResult] = {}
    for name, fn, fn_args in scenarios:
        try:
            value = fn(rank, world_size, *fn_args)
            outcomes[name] = RankResult(rank=rank, value=value)
        except BaseException as exc:
            outcomes[name] = RankResult(rank=rank, error=_capture_error(exc))
        barrier()
    return outcomes


def run_scenarios(
    scenarios: Mapping[str, Callable[..., Any] | tuple[Callable[..., Any], Sequence[Any]]],
    world_size: int,
    *,
    pg_timeout: float = 5.0,
    wall_timeout: float = 30.0,
    backend: str | None = None,
    device_map: str | None = None,
) -> dict[str, list[RankResult]]:
    """Run multiple named scenarios sequentially inside a single spawned process group.

    Each mapping entry is either a module-level callable `fn(rank, world_size)` or a pair
    `(fn, args)` invoked as `fn(rank, world_size, *args)`. Every scenario must leave the process
    group alive and balanced; a barrier separates scenarios so stray collectives surface immediately.
    Returns `{scenario_name: [rank_0_result, ..., rank_{N-1}_result]}`.
    """
    normalised: list[tuple[str, Callable[..., Any], tuple[Any, ...]]] = []
    for name, spec in scenarios.items():
        if isinstance(spec, tuple):
            fn, fn_args = spec
            normalised.append((name, fn, tuple(fn_args)))
        else:
            normalised.append((name, spec, ()))

    rank_results = run_distributed(
        _scenario_runner,
        world_size=world_size,
        args=(normalised,),
        pg_timeout=pg_timeout,
        wall_timeout=wall_timeout,
        backend=backend,
        device_map=device_map,
    )
    per_rank_dicts = unwrap_results(rank_results)
    return {name: [per_rank_dicts[rank][name] for rank in range(world_size)] for name, _, _ in normalised}
