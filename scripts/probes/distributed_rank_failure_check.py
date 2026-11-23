"""Probe: measure surviving-rank abort timing when one rank dies (`os._exit(1)`) mid-`fit()`.

Kept separate from `distributed_trainer_check.py` because `torchrun` immediately sends SIGTERM
to sibling ranks as soon as one worker exits non-zero, which would hide PyTorch's process-group
timeout behaviour on surviving ranks. Instead, this probe spawns `N` independent worker processes
via `torch.multiprocessing` (`spawn`), lets `--kill-rank` call `os._exit(1)` mid-`fit()` at
`--kill-step`, records per-rank exception and exit timings relative to the kill, and enforces a
hard wall-clock deadline in the parent (terminating/killing stragglers and reporting if it had to).

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
import signal
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _common import PATCH, tiny_random_forecaster

from timestral_tune import TimestralTrainer, TrainingConfig
from timestral_tune.data import ColumnRoles, SeriesData, TimeSeriesDataset

CTX_PATCHES = 2
HOR_PATCHES = 2
HORIZON = HOR_PATCHES * PATCH


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_dataset(num_windows: int, seed: int) -> TimeSeriesDataset:
    rng = np.random.default_rng(seed)
    ctx_len = CTX_PATCHES * PATCH
    total = ctx_len + HORIZON
    ts = np.arange(total, dtype=np.int64)
    empty_f = np.empty((0, total), dtype=np.float32)
    empty_c = np.empty((0, total), dtype=np.float64)
    series: list[SeriesData] = []
    for i in range(num_windows):
        values = (15.0 + 3.0 * np.sin(0.2 * ts + 0.5 * i) + rng.normal(0.0, 0.8, size=total)).astype(np.float32)
        series.append(SeriesData(f"s{i}", ts, values[None, :], empty_f, empty_f, empty_c, empty_c))
    return TimeSeriesDataset(
        series,
        context_length=ctx_len,
        forecast_horizon=HORIZON,
        stride=HORIZON,
        min_context=ctx_len,
        roles=ColumnRoles(targets=("y",)),
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


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
        if backend == "gloo":
            torch.set_num_threads(1)
            config_device: str | None = "cpu"
            expected_cuda_idx: int | None = None
        else:
            os.environ["LOCAL_RANK"] = str(rank)
            if device_map == "shared":
                torch.cuda.set_device(0)
                config_device = "cuda:0"
                expected_cuda_idx = 0
            else:
                # Leave device binding to TimestralTrainer's LOCAL_RANK path and verify afterwards.
                config_device = None
                expected_cuda_idx = rank

        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=pg_timeout),
        )

        ds = _make_dataset(world_size * 4, seed=seed + 100)
        max_steps = max(kill_step + 4, 6)
        cfg = TrainingConfig(
            model_name_or_path="local-rank-failure-probe",
            mode="head_only",
            device=config_device,  # type: ignore[arg-type]
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
    except BaseException as exc:
        _write_json_atomic(
            report_path,
            {
                "rank": rank,
                "stage": "init",
                "event": "init_error",
                "exc_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "exc_message": str(exc).splitlines()[0][:200] if str(exc) else "",
                "event_mono": time.monotonic(),
            },
        )
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        sys.exit(3)

    try:
        handle = None
        if rank == kill_rank:

            def _kill_mid_fit(_mod: Any, _inp: Any) -> None:
                if not trainer.model.training:
                    return
                # global_step increments after each completed optimiser step, so active 1-based step is +1.
                active_step = trainer.global_step + 1
                if active_step == kill_step:
                    _write_json_atomic(
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
        _write_json_atomic(
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
            _write_json_atomic(
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
        except BaseException as exc:
            t_exc = time.monotonic()
            _write_json_atomic(
                report_path,
                {
                    "rank": rank,
                    "stage": "fit",
                    "event": "exception",
                    "exc_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "exc_message": str(exc).splitlines()[0][:200] if str(exc) else "",
                    "event_mono": t_exc,
                    "fit_elapsed_s": t_exc - t_fit_start,
                },
            )
            sys.exit(2)
        finally:
            if handle is not None:
                handle.remove()
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _describe_exitcode(exitcode: int | None) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device-map", choices=("per-rank", "shared"), default="per-rank")
    parser.add_argument("--pg-timeout", type=float, default=20.0, help="Process-group timeout in seconds.")
    parser.add_argument(
        "--wall-timeout",
        type=float,
        default=None,
        help="Hard parent deadline in seconds (default: max(60, pg_timeout * 2.5 + 15)).",
    )
    parser.add_argument("--kill-rank", type=int, default=1, help="Rank index that calls os._exit(1) mid-fit().")
    parser.add_argument("--kill-step", type=int, default=2, help="1-based training step at which kill-rank exits.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.world_size < 2:
        print(f"error: --world-size must be >= 2, got {args.world_size}", file=sys.stderr)
        return 2
    if not (0 <= args.kill_rank < args.world_size):
        print(f"error: --kill-rank must be in [0, {args.world_size - 1}], got {args.kill_rank}", file=sys.stderr)
        return 2
    if args.kill_step < 1:
        print(f"error: --kill-step must be >= 1, got {args.kill_step}", file=sys.stderr)
        return 2
    if args.pg_timeout <= 0:
        print(f"error: --pg-timeout must be > 0, got {args.pg_timeout}", file=sys.stderr)
        return 2

    if args.backend == "gloo" and args.device_map == "shared":
        print("error: --device-map shared applies only to --backend nccl", file=sys.stderr)
        return 2
    if args.backend == "nccl":
        if not torch.cuda.is_available():
            print("error: --backend nccl requires CUDA, but torch.cuda.is_available() is False", file=sys.stderr)
            return 2
        if args.device_map == "shared":
            if os.environ.get("NCCL_MULTI_RANK_GPU_ENABLE") != "1":
                print(
                    "error: --device-map shared on NCCL requires NCCL_MULTI_RANK_GPU_ENABLE=1 in the environment",
                    file=sys.stderr,
                )
                return 2
        elif torch.cuda.device_count() < args.world_size:
            print(
                f"error: --device-map per-rank needs {args.world_size} GPUs, found {torch.cuda.device_count()}; "
                "use --device-map shared with NCCL_MULTI_RANK_GPU_ENABLE=1 on 1 GPU",
                file=sys.stderr,
            )
            return 2

    wall_timeout = args.wall_timeout if args.wall_timeout is not None else max(60.0, args.pg_timeout * 2.5 + 15.0)
    port = _pick_free_port()

    header_parts = [
        "distributed_rank_failure_check",
        f"torch {torch.__version__}",
        f"python {platform.python_version()}",
        f"{platform.system()} {platform.machine()}",
        f"backend={args.backend}",
        f"world_size={args.world_size}",
        f"pg_timeout={args.pg_timeout:.1f}s",
        f"wall_timeout={wall_timeout:.1f}s",
        f"kill_rank={args.kill_rank}",
        f"kill_step={args.kill_step}",
    ]
    if args.backend == "nccl":
        header_parts.append(f"device_map={args.device_map}")
        header_parts.append(f"cuda={torch.version.cuda}")
        nccl_ver = torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else None
        if nccl_ver is not None:
            header_parts.append(f"nccl={'.'.join(str(x) for x in nccl_ver)}")
        header_parts.append(f"gpu={torch.cuda.get_device_name(0)}")
        async_eh = os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING", "default")
        header_parts.append(f"async_eh={async_eh}")
    print("== " + " | ".join(header_parts) + " ==", flush=True)

    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="tt_rank_fail_") as status_dir:
        procs: list[mp.Process] = []
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

        exit_times: list[float | None] = [None] * args.world_size
        killed_by_parent: list[bool] = [False] * args.world_size
        deadline = t_spawn + wall_timeout

        try:
            while True:
                now = time.monotonic()
                all_done = True
                for r, proc in enumerate(procs):
                    if exit_times[r] is None:
                        if not proc.is_alive():
                            exit_times[r] = now
                        else:
                            all_done = False
                if all_done or now >= deadline:
                    break
                time.sleep(0.05)

            # Hard deadline enforcement: terminate then SIGKILL any surviving straggler.
            stragglers = [r for r, p in enumerate(procs) if p.is_alive()]
            if stragglers:
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
        finally:
            for proc in procs:
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1.0)

        reports: list[dict[str, Any]] = []
        for r in range(args.world_size):
            rp = Path(status_dir) / f"rank_{r}.json"
            reports.append(json.loads(rp.read_text(encoding="utf-8")) if rp.is_file() else {})

        # Anchor elapsed times at the exact monotonic instant right before os._exit(1) (falling back to observed exit).
        kill_rep = reports[args.kill_rank]
        t_kill = (
            kill_rep.get("event_mono")
            if (kill_rep.get("event") == "os_exit_1" and isinstance(kill_rep.get("event_mono"), float))
            else exit_times[args.kill_rank]
        )

        print(f"\n{'rank':>4} {'role':<9} {'status':<16} {'exc_s':>8} {'exit_s':>8}  detail", flush=True)
        ok = True
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
                detail = f"killed by parent after wall_timeout={wall_timeout:.1f}s"
                if rep.get("event") == "exception":
                    detail += f" (caught {rep.get('exc_type')} before teardown hang)"
                ok = False
            else:
                status_str = _describe_exitcode(procs[r].exitcode)
                if rep.get("event") == "init_error":
                    detail = f"INIT ERROR {rep.get('exc_type')}: {rep.get('exc_message')}"
                    ok = False
                elif r == args.kill_rank:
                    if rep.get("event") == "os_exit_1" and procs[r].exitcode == 1:
                        detail = f"os._exit(1) at step {rep.get('step')}"
                    else:
                        detail = f"unexpected kill_rank outcome: {rep or status_str}"
                        ok = False
                else:
                    if rep.get("event") == "exception":
                        detail = f"{rep.get('exc_type')}: {rep.get('exc_message')}"
                    elif rep.get("event") == "returned_normally":
                        detail = f"ERROR: returned normally after {rep.get('steps')} steps"
                        ok = False
                    elif rep.get("stage") == "fit" and procs[r].exitcode is not None and procs[r].exitcode < 0:
                        # Native abort inside fit(), e.g. NCCL watchdog SIGABRT before Python caught an exception.
                        detail = f"aborted in fit() by {status_str} (no Python exception captured)"
                    else:
                        detail = f"unexpected exit ({status_str}; report={rep or 'none'})"
                        ok = False
            print(f"{r:>4} {role:<9} {status_str:<16} {exc_str} {exit_str}  {detail}", flush=True)

        killed_note = f"parent_killed={parent_killed_ranks}" if parent_killed_ranks else "parent_killed=none"
        print(f"\nresult: {'PASS (no hang)' if ok else 'FAIL'} | {killed_note}", flush=True)
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
