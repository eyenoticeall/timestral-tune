"""Reusable multi-process test harness for CPU/gloo `torch.distributed` scenarios.

Spawns real separate Python processes using the `spawn` start method and coordinates rendezvous
through a file-based store (`init_method="file://..."`) inside a fresh temporary directory, avoiding
ephemeral TCP port collisions.

Design guarantees:
- Each worker uses 1 intra-op CPU thread, initialises the process group with a configurable short
  timeout, invokes the target module-level function as `fn(rank, world_size, *args)`, records either
  its picklable return value or its exception type/message/traceback to a per-rank file, and always
  destroys the process group in `finally`.
- One worker raising never kills sibling workers prematurely; every rank finishes and reports what
  it saw (essential for testing failure propagation and consistency checks).
- The parent enforces a hard wall-clock deadline: on expiry it terminates/kills all live workers and
  raises `DistributedTimeoutError` naming the unfinished ranks.
- Workers killed by a signal or exiting abnormally are reported with rank and signal/exit code.
- `run_scenarios` amortises process-spawn overhead by running multiple named scenarios sequentially
  inside a single spawned process group.
"""

from __future__ import annotations

import datetime
import math
import multiprocessing
import os
import pickle
import signal
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
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
    "f32_tol_scale",
    "run_distributed",
    "run_scenarios",
    "unwrap_results",
]

_ALLOWED_CPU_BACKENDS = frozenset({"gloo", "cpu:gloo"})

# Environment knob multiplying every float32 parity tolerance in the multi-process test suite
# (float64 tiers are never scaled). A run that passes at 0.1 has at least 10x headroom on that machine;
# at 0.2, at least 5x. Values above 1 loosen the tiers, for exploring a new platform. Must be a finite
# positive float.
F32_TOL_SCALE_ENV = "TIMESTRAL_TEST_F32_TOL_SCALE"


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
) -> None:
    """Top-level worker entry point for `spawn`."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    outcome: RankResult | None = None
    try:
        try:
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
        torch.set_num_threads(prev_threads)


def _signal_label(sig_num: int) -> str:
    try:
        return f"{signal.Signals(sig_num).name} ({sig_num})"
    except ValueError:
        return f"signal {sig_num}"


def _terminate_workers(procs: Sequence[multiprocessing.process.BaseProcess]) -> None:
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    grace_deadline = time.monotonic() + 0.5
    for proc in procs:
        if proc.is_alive():
            remaining = max(0.0, grace_deadline - time.monotonic())
            proc.join(timeout=remaining)
    for proc in procs:
        if proc.is_alive():
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
    backend: str = "gloo",
) -> list[RankResult]:
    """Run module-level `fn(rank, world_size, *args)` on `world_size` spawned CPU/gloo processes.

    Returns a list of `RankResult` indexed by rank `0..world_size-1`. Workers raising Python
    exceptions are recorded in `RankResult.error` without aborting sibling ranks. Raises
    `DistributedTimeoutError` if `wall_timeout` elapses before all ranks finish, or `RuntimeError`
    if any worker exits via a signal or abnormal non-zero exit code.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if backend not in _ALLOWED_CPU_BACKENDS:
        raise ValueError(f"harness supports CPU/gloo only (one of {sorted(_ALLOWED_CPU_BACKENDS)}); got {backend!r}")
    if pg_timeout <= 0.0:
        raise ValueError(f"pg_timeout must be > 0, got {pg_timeout}")
    if wall_timeout <= 0.0:
        raise ValueError(f"wall_timeout must be > 0, got {wall_timeout}")

    ctx = multiprocessing.get_context("spawn")
    args_tuple = tuple(args)

    with tempfile.TemporaryDirectory(prefix="timestral_dist_") as tmpdir:
        tmp = Path(tmpdir)
        store_path = str(tmp / "rendezvous")
        result_paths = [tmp / f"rank_{rank}.pkl" for rank in range(world_size)]
        procs = [
            ctx.Process(
                target=_worker_main,
                args=(rank, world_size, store_path, str(result_paths[rank]), pg_timeout, backend, fn, args_tuple),
                name=f"dist-rank-{rank}",
            )
            for rank in range(world_size)
        ]

        try:
            for proc in procs:
                proc.start()

            deadline = time.monotonic() + wall_timeout
            while True:
                alive = [proc for proc in procs if proc.is_alive()]
                if not alive:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                alive[0].join(timeout=min(0.05, remaining))

            unfinished = [rank for rank, proc in enumerate(procs) if proc.is_alive()]
            if unfinished:
                _terminate_workers(procs)
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
                _terminate_workers(procs)


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
        if dist.is_initialized():
            dist.barrier()
    return outcomes


def run_scenarios(
    scenarios: Mapping[str, Callable[..., Any] | tuple[Callable[..., Any], Sequence[Any]]],
    world_size: int,
    *,
    pg_timeout: float = 5.0,
    wall_timeout: float = 30.0,
    backend: str = "gloo",
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
    )
    per_rank_dicts = unwrap_results(rank_results)
    return {name: [per_rank_dicts[rank][name] for rank in range(world_size)] for name, _, _ in normalised}
