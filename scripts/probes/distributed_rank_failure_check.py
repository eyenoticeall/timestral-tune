"""Probe: measure surviving-rank abort timing when one rank dies (`os._exit(1)`) mid-`fit()`.

Kept separate from `distributed_trainer_check.py` because `torchrun` immediately sends SIGTERM
to sibling ranks as soon as one worker exits non-zero, which would hide PyTorch's process-group
timeout behaviour on surviving ranks. Instead, this probe spawns `N` independent worker processes
via `torch.multiprocessing` (`spawn`), lets `--kill-rank` call `os._exit(1)` mid-`fit()` at
`--kill-step`, records per-rank exception and exit timings relative to the kill, and enforces two
separate hard deadlines in the parent:

- `--startup-timeout`: budget from spawn until `--kill-rank` exits (default: `max(90, wall_timeout)`).
- `--wall-timeout`: post-kill budget armed once `--kill-rank` exits (default: `max(60, pg_timeout * 2.5 + 15)`).

Recognised survivor abort criteria (strict allow-list):
- A native `SIGABRT` received inside `fit()` after `kill_rank` exits (e.g. NCCL watchdog abort).
- A Python exception raised inside `fit()` after `kill_rank` exits (never OOM or `RankFailureError`)
  that is either an explicit distributed/network exception type (`DistBackendError`,
  `DistNetworkError`, `DistStoreError`, `DistError`, `TimeoutError`, `BrokenPipeError`,
  `ConnectionResetError`, `ConnectionAbortedError`, `ConnectionRefusedError`) or a `RuntimeError` /
  `OSError` whose message matches a known Gloo/NCCL/c10d transport or collective failure pattern
  (such as macOS Gloo `"Timed out waiting ... for recv/send operation"`, Linux Gloo `"Read error"` /
  `"Connection closed by peer"` / `"Software caused connection abort"` / `"short read"`, or NCCL /
  c10d `"ncclRemoteError"` / `"ncclInternalError"` / `"watchdog"` / `"communicator was aborted"` /
  `"work was aborted"` / `"remote process exited"` / `"ProcessGroup"`).
- Any other exception is reported as `FAIL (unrecognised exception, treated as FAIL)` with its class
  and message so unrelated bugs can never silently pass as a clean abort.

Examples:

    # CPU / gloo (N=2, 15 s process-group timeout):
    TMPDIR=/tmp PYTHONFAULTHANDLER=1 uv run python scripts/probes/distributed_rank_failure_check.py \\
        --backend gloo --world-size 2 --pg-timeout 15

    # Single-GPU shared NCCL (N=2 on cuda:0):
    NCCL_MULTI_RANK_GPU_ENABLE=1 TMPDIR=/tmp PYTHONFAULTHANDLER=1 \\
        uv run python scripts/probes/distributed_rank_failure_check.py \\
        --backend nccl --device-map shared --world-size 2 --pg-timeout 20

    # Multi-GPU per-rank NCCL (N=4 on cuda:0..3):
    TMPDIR=/tmp PYTHONFAULTHANDLER=1 uv run python scripts/probes/distributed_rank_failure_check.py \\
        --backend nccl --device-map per-rank --world-size 4 --pg-timeout 20

See `docs/phases/phase-5-multi-gpu.md` §3 and `docs/SPEC.md` §5.6 (D35), §12.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _common import tiny_random_forecaster
from _dist_common import (
    DOM_HORIZON,
    cuda_env_banner_parts,
    describe_exitcode,
    make_synthetic_dataset,
    pick_free_port,
    require_positive_finite,
    validate_distributed_device_map,
    write_json_atomic,
)

from timestral_tune import TimestralTrainer, TrainingConfig
from timestral_tune.trainer.distributed import RankFailureError, is_oom_error

# Small clock slack for comparing independent process monotonic timestamps around the kill event.
_PRE_KILL_SLACK_S = 0.05

# Explicit distributed / transport exception types accepted unconditionally as clean survivor aborts.
_EXPECTED_ABORT_TYPES: tuple[type[BaseException], ...] = tuple(
    t
    for t in (
        getattr(dist, "DistBackendError", None),
        getattr(dist, "DistNetworkError", None),
        getattr(dist, "DistStoreError", None),
        getattr(dist, "DistError", None),
        TimeoutError,
        BrokenPipeError,
        ConnectionResetError,
        ConnectionAbortedError,
        ConnectionRefusedError,
    )
    if isinstance(t, type) and issubclass(t, BaseException)
)

# Case-insensitive patterns matching genuine Gloo / NCCL / c10d transport & collective errors
# when raised as generic `RuntimeError` or `OSError`.
_EXPECTED_ABORT_MESSAGE_RE = re.compile(
    r"|".join(
        (
            r"timed out waiting",
            r"work timed out",
            r"\btimeout\b",
            r"\btimed out\b",
            r"read error",
            r"write error",
            r"\bshort read\b",
            r"connection closed by peer",
            r"connection reset by peer",
            r"connection refused",
            r"software caused connection abort",
            r"connection abort(?:ed)?",
            r"broken pipe",
            r"\beof\b",
            r"\bgloo\b",
            r"\bnccl\b",
            r"\bnccl(?:[a-z]*error|invalid[a-z]*)\b",
            r"\bc10d\b",
            r"process\s*group",
            r"\bwatchdog\b",
            r"\bcollective\b",
            r"communicator was aborted",
            r"comm aborted",
            r"work was aborted",
            r"remote process exited",
            r"remote peer",
            r"\btransport\b",
            r"\bsocket\b",
            r"tcpstore",
            r"store error",
        )
    ),
    re.IGNORECASE,
)


def _is_expected_survivor_exception(exc: Exception) -> bool:
    """Return True iff `exc` matches the allow-list of distributed transport/collective aborts."""
    if is_oom_error(exc) or isinstance(exc, RankFailureError):
        return False
    if isinstance(exc, _EXPECTED_ABORT_TYPES):
        return True
    if isinstance(exc, (RuntimeError, OSError)):
        return bool(_EXPECTED_ABORT_MESSAGE_RE.search(str(exc)))
    return False


def _init_worker_trainer(
    *,
    rank: int,
    world_size: int,
    port: int,
    backend: Literal["gloo", "nccl"],
    device_map: Literal["per-rank", "shared"],
    pg_timeout: float,
    kill_step: int,
    seed: int,
) -> TimestralTrainer:
    """Initialise the process group and construct a verified `TimestralTrainer` for one worker."""
    if backend == "gloo":
        torch.set_num_threads(1)
        config_device = "cpu"
        expected_cuda_idx: int | None = None
    else:
        os.environ["LOCAL_RANK"] = str(rank)
        if device_map == "shared":
            torch.cuda.set_device(0)
            config_device = "cuda:0"
            expected_cuda_idx = 0
        else:
            # Pass "auto" so TimestralTrainer's DistributedContext.detect() binds cuda:LOCAL_RANK.
            config_device = "auto"
            expected_cuda_idx = rank

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=pg_timeout),
    )

    ds = make_synthetic_dataset(
        world_size * 4,
        horizon=DOM_HORIZON,
        covariates=False,
        seed=seed + 100,
        drop_tail_targets=False,
    )
    max_steps = max(kill_step + 4, 6)
    cfg = TrainingConfig(
        model_name_or_path="local-rank-failure-probe",
        mode="head_only",
        device=config_device,
        batch_size=2,
        epochs=max_steps,
        max_steps=max_steps,
        learning_rate=1e-2,
        shuffle=False,
        sign_flip=False,
        restore_best=False,
        cache_frozen_features=False,
        log_every=max_steps + 10,
        seed=seed,
    )
    trainer = TimestralTrainer(cfg, ds, model=tiny_random_forecaster(seed=seed).model)
    if expected_cuda_idx is not None:
        expected_dev = torch.device("cuda", expected_cuda_idx)
        if trainer.device != expected_dev or torch.cuda.current_device() != expected_cuda_idx:
            raise RuntimeError(
                f"rank {rank}: expected CUDA binding {expected_dev}, "
                f"got trainer.device={trainer.device} current_device={torch.cuda.current_device()}"
            )
    return trainer


def _worker_main(
    rank: int,
    world_size: int,
    port: int,
    backend: Literal["gloo", "nccl"],
    device_map: Literal["per-rank", "shared"],
    pg_timeout: float,
    kill_rank: int,
    kill_step: int,
    seed: int,
    status_dir: str,
) -> None:
    """Spawned worker process: runs `TimestralTrainer.fit()` and triggers `os._exit(1)` on `kill_rank`."""
    report_path = Path(status_dir) / f"rank_{rank}.json"
    try:
        trainer = _init_worker_trainer(
            rank=rank,
            world_size=world_size,
            port=port,
            backend=backend,
            device_map=device_map,
            pg_timeout=pg_timeout,
            kill_step=kill_step,
            seed=seed,
        )
    except Exception as exc:
        write_json_atomic(
            report_path,
            {
                "rank": rank,
                "stage": "init",
                "event": "init_error",
                "exc_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "exc_message": str(exc).splitlines()[0][:240] if str(exc) else "",
                "event_mono": time.monotonic(),
            },
        )
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        sys.exit(3)

    handle = None
    try:
        if rank == kill_rank:

            def _kill_mid_fit(_mod: Any, _inp: Any) -> None:
                if not trainer.model.training:
                    return
                # global_step increments after each completed optimiser step, so active 1-based step is +1.
                active_step = trainer.global_step + 1
                if active_step == kill_step:
                    write_json_atomic(
                        report_path,
                        {
                            "rank": rank,
                            "stage": "fit",
                            "event": "os_exit_1",
                            "step": active_step,
                            "event_mono": time.monotonic(),
                        },
                    )
                    os._exit(1)

            handle = trainer.model.transformer.register_forward_pre_hook(_kill_mid_fit)

        # Mark entry into fit() before running so the parent distinguishes startup crashes from
        # in-fit native signals (e.g. NCCL watchdog SIGABRT before Python catches an exception).
        t_fit_start = time.monotonic()
        write_json_atomic(
            report_path,
            {
                "rank": rank,
                "stage": "fit",
                "event": "fit_started",
                "event_mono": t_fit_start,
            },
        )
        try:
            res = trainer.fit()
            write_json_atomic(
                report_path,
                {
                    "rank": rank,
                    "stage": "fit",
                    "event": "returned_normally",
                    "steps": res.steps,
                    "event_mono": time.monotonic(),
                    "fit_elapsed_s": time.monotonic() - t_fit_start,
                },
            )
        except Exception as exc:
            t_exc = time.monotonic()
            write_json_atomic(
                report_path,
                {
                    "rank": rank,
                    "stage": "fit",
                    "event": "exception",
                    "exc_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "exc_message": str(exc).splitlines()[0][:240] if str(exc) else "",
                    "is_expected_abort": _is_expected_survivor_exception(exc),
                    "event_mono": t_exc,
                    "fit_elapsed_s": t_exc - t_fit_start,
                },
            )
            sys.exit(2)
        finally:
            if handle is not None:
                handle.remove()
    finally:
        # Intentionally call destroy_process_group() here (matching docs/SPEC.md §7.6) while the
        # parent tracks exc_s vs exit_s separately so any teardown hang on a broken group is exposed.
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _spawn_and_monitor(
    *,
    args: argparse.Namespace,
    port: int,
    status_dir: str,
    startup_budget: float,
    post_kill_budget: float,
) -> tuple[list[int | None], list[float | None], list[bool], Literal["none", "startup_timeout", "wall_timeout"]]:
    """Spawn all `N` workers inside `try/finally` and enforce separate startup and post-kill deadlines."""
    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    exit_times: list[float | None] = [None] * args.world_size
    killed_by_parent: list[bool] = [False] * args.world_size
    expired_budget: Literal["none", "startup_timeout", "wall_timeout"] = "none"

    try:
        t_spawn = time.monotonic()
        for r in range(args.world_size):
            proc = ctx.Process(
                target=_worker_main,
                args=(
                    r,
                    args.world_size,
                    port,
                    args.backend,
                    args.device_map,
                    args.pg_timeout,
                    args.kill_rank,
                    args.kill_step,
                    args.seed,
                    status_dir,
                ),
            )
            proc.start()
            procs.append(proc)

        deadline = t_spawn + startup_budget
        post_kill_armed = False
        while True:
            now = time.monotonic()
            all_done = True
            for r, proc in enumerate(procs):
                if exit_times[r] is None:
                    if not proc.is_alive():
                        exit_times[r] = now
                    else:
                        all_done = False
            if not post_kill_armed and exit_times[args.kill_rank] is not None:
                # Arm the post-kill timeout budget once kill_rank actually exits, so cold startup
                # and CUDA/NCCL initialisation stay governed strictly by startup_budget.
                deadline = exit_times[args.kill_rank] + post_kill_budget
                post_kill_armed = True
            if all_done or now >= deadline:
                break
            time.sleep(0.05)

        stragglers = [r for r, p in enumerate(procs) if p.is_alive()]
        if stragglers:
            expired_budget = "wall_timeout" if post_kill_armed else "startup_timeout"
            for r in stragglers:
                killed_by_parent[r] = True
                procs[r].terminate()
            grace_end = time.monotonic() + 2.0
            while time.monotonic() < grace_end and any(procs[r].is_alive() for r in stragglers):
                time.sleep(0.05)
            for r in stragglers:
                if procs[r].is_alive():
                    procs[r].kill()
        for r, proc in enumerate(procs):
            proc.join(timeout=2.0)
            if exit_times[r] is None:
                exit_times[r] = time.monotonic()
        exitcodes = [p.exitcode for p in procs]
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2.0)

    return exitcodes, exit_times, killed_by_parent, expired_budget


def _evaluate_and_print_report(
    *,
    args: argparse.Namespace,
    reports: list[dict[str, Any]],
    exitcodes: list[int | None],
    exit_times: list[float | None],
    killed_by_parent: list[bool],
    expired_budget: Literal["none", "startup_timeout", "wall_timeout"],
    startup_budget: float,
    post_kill_budget: float,
) -> bool:
    """Print the per-rank table and return True iff kill_rank died on schedule and every survivor aborted cleanly."""
    kill_rep = reports[args.kill_rank]
    kill_succeeded = (
        not killed_by_parent[args.kill_rank]
        and kill_rep.get("event") == "os_exit_1"
        and exitcodes[args.kill_rank] == 1
        and isinstance(kill_rep.get("event_mono"), float)
    )
    t_kill: float | None = (
        float(kill_rep["event_mono"])
        if (kill_rep.get("event") == "os_exit_1" and isinstance(kill_rep.get("event_mono"), float))
        else exit_times[args.kill_rank]
    )

    print(f"\n{'rank':>4} {'role':<9} {'status':<16} {'exc_s':>8} {'exit_s':>8}  detail", flush=True)
    ok = kill_succeeded
    parent_killed_ranks: list[int] = []

    for r in range(args.world_size):
        role = "killed" if r == args.kill_rank else "survivor"
        rep = reports[r]
        t_ex = exit_times[r]
        exit_delta = (t_ex - t_kill) if (t_ex is not None and t_kill is not None) else None
        exit_str = f"{exit_delta:>8.2f}" if exit_delta is not None else f"{'-':>8}"

        ev_mono = rep.get("event_mono")
        exc_delta = (
            (ev_mono - t_kill)
            if (
                r != args.kill_rank
                and rep.get("event") == "exception"
                and isinstance(ev_mono, float)
                and t_kill is not None
            )
            else None
        )
        exc_str = f"{exc_delta:>8.2f}" if exc_delta is not None else f"{'-':>8}"

        if killed_by_parent[r]:
            parent_killed_ranks.append(r)
            status_str = "HUNG_KILLED"
            if expired_budget == "startup_timeout":
                detail = f"killed by parent after startup_timeout={startup_budget:.1f}s (before kill_rank exited)"
            else:
                detail = f"killed by parent after wall_timeout={post_kill_budget:.1f}s (post-kill)"
            if rep.get("event") == "exception":
                detail += f" (caught {rep.get('exc_type')} before teardown hang)"
            ok = False
        else:
            status_str = describe_exitcode(exitcodes[r])
            if rep.get("event") == "init_error":
                detail = f"INIT ERROR {rep.get('exc_type')}: {rep.get('exc_message')}"
                ok = False
            elif r == args.kill_rank:
                if kill_succeeded:
                    detail = f"os._exit(1) at step {rep.get('step')}"
                else:
                    detail = f"unexpected kill_rank outcome: {rep or status_str}"
                    ok = False
            elif rep.get("event") == "exception":
                exc_desc = f"{rep.get('exc_type')}: {rep.get('exc_message')}"
                if not kill_succeeded or (exc_delta is not None and exc_delta < -_PRE_KILL_SLACK_S):
                    detail = f"FAIL (aborted before kill_rank exited): {exc_desc}"
                    ok = False
                elif not rep.get("is_expected_abort", False):
                    detail = f"FAIL (unrecognised exception, treated as FAIL): {exc_desc}"
                    ok = False
                else:
                    detail = exc_desc
            elif rep.get("event") == "returned_normally":
                detail = f"ERROR: returned normally after {rep.get('steps')} steps"
                ok = False
            elif rep.get("stage") == "fit" and exitcodes[r] == -int(signal.SIGABRT):
                if not kill_succeeded or (exit_delta is not None and exit_delta < -_PRE_KILL_SLACK_S):
                    detail = f"FAIL ({status_str} before kill_rank exited)"
                    ok = False
                else:
                    # Expected native abort inside fit(), e.g. NCCL watchdog SIGABRT before Python caught an exception.
                    detail = f"aborted in fit() by {status_str} (no Python exception captured)"
            elif rep.get("stage") == "fit" and exitcodes[r] is not None and exitcodes[r] < 0:
                detail = f"FAIL (unexpected crash signal {status_str} in fit())"
                ok = False
            else:
                detail = f"unexpected exit ({status_str}; report={rep or 'none'})"
                ok = False

        print(f"{r:>4} {role:<9} {status_str:<16} {exc_str} {exit_str}  {detail}", flush=True)

    killed_note = f"parent_killed={parent_killed_ranks}" if parent_killed_ranks else "parent_killed=none"
    budget_note = f" | expired_budget={expired_budget}" if expired_budget != "none" else ""
    print(f"\nresult: {'PASS (no hang)' if ok else 'FAIL'} | {killed_note}{budget_note}", flush=True)
    return ok


def _parse_and_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device-map", choices=("per-rank", "shared"), default="per-rank")
    parser.add_argument("--pg-timeout", type=float, default=20.0, help="Process-group timeout in seconds.")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=None,
        help="Hard startup budget in seconds from spawn until kill-rank exits (default: max(90, wall_timeout)).",
    )
    parser.add_argument(
        "--wall-timeout",
        type=float,
        default=None,
        help="Hard post-kill parent deadline in seconds (default: max(60, pg_timeout * 2.5 + 15)).",
    )
    parser.add_argument("--kill-rank", type=int, default=1, help="Rank index that calls os._exit(1) mid-fit().")
    parser.add_argument("--kill-step", type=int, default=2, help="1-based training step at which kill-rank exits.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.world_size < 2:
        parser.error(f"--world-size must be >= 2, got {args.world_size}")
    if not (0 <= args.kill_rank < args.world_size):
        parser.error(f"--kill-rank must be in [0, {args.world_size - 1}], got {args.kill_rank}")
    if args.kill_step < 1:
        parser.error(f"--kill-step must be >= 1, got {args.kill_step}")
    require_positive_finite(parser, "--pg-timeout", args.pg_timeout)
    if args.startup_timeout is not None:
        require_positive_finite(parser, "--startup-timeout", args.startup_timeout)
    if args.wall_timeout is not None:
        require_positive_finite(parser, "--wall-timeout", args.wall_timeout)

    err = validate_distributed_device_map(
        backend=args.backend,
        device_map=args.device_map,
        world_size=args.world_size,
    )
    if err is not None:
        print(f"error: {err}", file=sys.stderr)
        sys.exit(2)
    return args


def main() -> int:
    args = _parse_and_validate_args()
    post_kill_budget = args.wall_timeout if args.wall_timeout is not None else max(60.0, args.pg_timeout * 2.5 + 15.0)
    startup_budget = args.startup_timeout if args.startup_timeout is not None else max(90.0, post_kill_budget)
    port = pick_free_port()

    header_parts = [
        "distributed_rank_failure_check",
        f"torch {torch.__version__}",
        f"python {platform.python_version()}",
        f"{platform.system()} {platform.machine()}",
        f"backend={args.backend}",
        f"world_size={args.world_size}",
        f"pg_timeout={args.pg_timeout:.1f}s",
        f"startup_timeout={startup_budget:.1f}s",
        f"wall_timeout={post_kill_budget:.1f}s",
        f"kill_rank={args.kill_rank}",
        f"kill_step={args.kill_step}",
    ]
    if args.backend == "nccl":
        header_parts.append(f"device_map={args.device_map}")
        header_parts.extend(cuda_env_banner_parts(allow_cuda_context=False))
        async_eh = os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING", "default")
        header_parts.append(f"async_eh={async_eh}")
    print("== " + " | ".join(header_parts) + " ==", flush=True)

    with tempfile.TemporaryDirectory(prefix="tt_rank_fail_") as status_dir:
        exitcodes, exit_times, killed_by_parent, expired_budget = _spawn_and_monitor(
            args=args,
            port=port,
            status_dir=status_dir,
            startup_budget=startup_budget,
            post_kill_budget=post_kill_budget,
        )
        reports: list[dict[str, Any]] = []
        for r in range(args.world_size):
            rp = Path(status_dir) / f"rank_{r}.json"
            reports.append(json.loads(rp.read_text(encoding="utf-8")) if rp.is_file() else {})

        ok = _evaluate_and_print_report(
            args=args,
            reports=reports,
            exitcodes=exitcodes,
            exit_times=exit_times,
            killed_by_parent=killed_by_parent,
            expired_budget=expired_budget,
            startup_budget=startup_budget,
            post_kill_budget=post_kill_budget,
        )
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
