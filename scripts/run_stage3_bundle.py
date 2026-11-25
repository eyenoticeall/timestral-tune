#!/usr/bin/env python3
"""Master stdlib-only runner for a bundled Phase 5 Stage 3 multi-GPU session (e.g. 4x A100 SXM4).

Runs all remaining Stage 3 hardware verification items sequentially in one resumable, signal-safe
session and writes all logs, structured JSON outputs, repeat-determinism diffs, and a final
`summary.txt` into `--out-dir`:

  00_host_preflight     GPU inventory, `can_device_access_peer` matrix, `nvidia-smi topo -m`,
                        CPU/RAM/`/dev/shm`/disk, and a 4-rank `NCCL_DEBUG=INFO` transport probe
                        confirming NVLink / GPU P2P transport.
  01_t15_nccl           All 142 multi-process acceptance + harness tests on 2 & 4 GPUs (0 skips).
  02_t15_nccl_scale02   Same at `TIMESTRAL_TEST_F32_TOL_SCALE=0.2` (5x float32 headroom check).
  03_multi_gpu_check_r1 Full 18-stage `scripts/multi_gpu_check.sh` session (including `01_env`
                        CPU/peer-access/topology, 4-GPU prototype parity, 316 driver rows, and
                        NVLink/P2P NCCL rank-failure probe).
  04_multi_gpu_check_r2 Second independent `scripts/multi_gpu_check.sh` run on the same host +
                        programmatic row-by-row parity diff (`04_repeat_diff.txt`) for cross-GPU
                        run-to-run determinism.
  05_calibration_probe  `bf16-cause` 3-way decomposition, `compile-var` 3x fresh-process repeats,
                        and multi-seed (`11 22 33`), multi-size (`tiny 5M 28M 123M 209M`) exact
                        float64 L2 + cosine-similarity calibration (`05_calibration.json`).
  06_scaling_bench      1/2/4-GPU throughput, weak/strong scaling, long-context (`1024`/`4096`),
                        validation + `_frozen_base` VRAM, and pack/`all_reduce`/unpack breakdown
                        to confirm D22 (`06_scaling.json`).
  07_linear_dispatch    `scripts/probes/linear_requires_grad_dispatch.py --device cuda`.
  08_cuda_check         Optional (`--include-cuda-check`): single-GPU `scripts/cuda_check.sh`.

Examples:
    # Full 4x A100 SXM4 unattended session:
    uv run python scripts/run_stage3_bundle.py --python 3.12 --out-dir stage3_a100_bundle

    # Resume after interruption (completed steps with `.ok` markers are skipped automatically):
    uv run python scripts/run_stage3_bundle.py --python 3.12 --out-dir stage3_a100_bundle

    # Local CPU/gloo end-to-end dry run (macOS or Linux, no GPU required):
    TMPDIR=/tmp uv run python scripts/run_stage3_bundle.py --dry-run-cpu --out-dir /tmp/stage3_dry
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parent.parent

ALL_STEP_NAMES: tuple[str, ...] = (
    "00_host_preflight",
    "01_t15_nccl",
    "02_t15_nccl_scale02",
    "03_multi_gpu_check_r1",
    "04_multi_gpu_check_r2",
    "05_calibration_probe",
    "06_scaling_bench",
    "07_linear_dispatch",
    "08_cuda_check",
)

_PREFLIGHT_SNIPPET = """\
import datetime, os, platform, shutil, subprocess, sys, torch
print("timestamp_utc=" + datetime.datetime.now(datetime.timezone.utc).isoformat())
print(
    f"os={platform.system()} release={platform.release()} "
    f"arch={platform.machine()} python={platform.python_version()}"
)
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_runtime={torch.version.cuda}")
print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
print(f"nccl_multi_rank_gpu_enable={os.environ.get('NCCL_MULTI_RANK_GPU_ENABLE', 'unset')}")
df_cmd = ["df", "-h", ".", "/dev/shm"] if os.path.exists("/dev/shm") else ["df", "-h", "."]
for cmd in (["lscpu"], ["free", "-h"], df_cmd):
    if shutil.which(cmd[0]) is not None:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
        print(f"--- {' '.join(cmd)} (rc={r.returncode}) ---")
        print(r.stdout.rstrip())
if torch.cuda.is_available():
    nccl = torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else None
    n = torch.cuda.device_count()
    print(f"nccl={'.'.join(str(x) for x in nccl) if nccl else None} gpu_count={n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"gpu[{i}]={props.name} cc={props.major}.{props.minor} vram_gib={props.total_memory / (2**30):.2f}")
    for i in range(n):
        row = [
            "-" if i == j else ("1" if torch.cuda.can_device_access_peer(i, j) else "0")
            for j in range(n)
        ]
        print(f"gpu_peer_access[{i}]=" + ",".join(row))
    if shutil.which("nvidia-smi") is not None:
        q_fields = (
            "index,name,driver_version,memory.total,"
            "pcie.link.gen.current,pcie.link.width.current"
        )
        for smi_args in (
            ["nvidia-smi", f"--query-gpu={q_fields}", "--format=csv"],
            ["nvidia-smi", "topo", "-m"],
            ["nvidia-smi", "nvlink", "--status"],
        ):
            r = subprocess.run(smi_args, check=False, capture_output=True, text=True, timeout=15)
            print(f"--- {' '.join(smi_args)} (rc={r.returncode}) ---")
            if r.stdout.strip():
                print(r.stdout.rstrip())
            elif r.stderr.strip():
                print(r.stderr.rstrip())
"""


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_nccl_transport_worker() -> int:
    """Internal `torchrun` worker verifying real multi-GPU NCCL all_reduce transport."""
    import datetime

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=60))
    try:
        t = torch.full((1024 * 1024,), float(rank + 1), dtype=torch.float32, device=f"cuda:{local_rank}")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(local_rank)
        expected = sum(range(1, world_size + 1))
        assert float(t[0].item()) == float(expected), f"got {t[0].item()} != {expected}"
        if rank == 0:
            print(f"nccl_transport_all_reduce_ok world_size={world_size} expected_sum={expected}")
    finally:
        dist.destroy_process_group()
    return 0


@dataclass(frozen=True)
class BundleConfig:
    out_dir: Path
    python_version: str | None
    ranks: tuple[int, ...]
    calib_sizes: tuple[str, ...]
    scaling_sizes: tuple[str, ...]
    seeds: tuple[int, ...]
    only: tuple[str, ...]
    skip: tuple[str, ...]
    force: bool
    include_cuda_check: bool
    dry_run_cpu: bool


@dataclass(frozen=True)
class BundleStep:
    name: str
    commands: tuple[tuple[str, ...], ...]
    extra_env: tuple[tuple[str, str], ...] = ()
    post_action: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("stage3_a100_bundle"))
    parser.add_argument("--python", dest="python_version", default=None, help="Python version for uv (e.g. 3.12).")
    parser.add_argument("--ranks", nargs="+", type=int, choices=(2, 4), default=[2, 4])
    parser.add_argument(
        "--calib-sizes",
        nargs="+",
        choices=("tiny", "5M", "28M", "123M", "209M"),
        default=["tiny", "5M", "28M", "123M", "209M"],
    )
    parser.add_argument(
        "--scaling-sizes",
        nargs="+",
        choices=("tiny", "5M", "28M", "123M", "209M"),
        default=["28M", "123M", "209M"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--only", nargs="+", choices=ALL_STEP_NAMES, default=[])
    parser.add_argument("--skip", nargs="+", choices=ALL_STEP_NAMES, default=[])
    parser.add_argument("--force", action="store_true", help="Re-run steps even if their `.ok` marker exists.")
    parser.add_argument(
        "--include-cuda-check",
        action="store_true",
        help="Also run single-GPU `scripts/cuda_check.sh` as step 08_cuda_check.",
    )
    parser.add_argument(
        "--dry-run-cpu",
        action="store_true",
        help="Run a fast CPU/gloo tiny-model subset locally to verify the entire bundle pipeline without GPUs.",
    )
    parser.add_argument("--_nccl-transport-check", action="store_true", help=argparse.SUPPRESS)
    return parser


def parse_args(argv: Sequence[str]) -> BundleConfig:
    ns = build_parser().parse_args(list(argv))
    dedup_ranks: list[int] = []
    for r in ns.ranks:
        if r not in dedup_ranks:
            dedup_ranks.append(r)
    return BundleConfig(
        out_dir=ns.out_dir,
        python_version=ns.python_version,
        ranks=tuple(dedup_ranks),
        calib_sizes=tuple(ns.calib_sizes),
        scaling_sizes=tuple(ns.scaling_sizes),
        seeds=tuple(ns.seeds),
        only=tuple(ns.only),
        skip=tuple(ns.skip),
        force=bool(ns.force),
        include_cuda_check=bool(ns.include_cuda_check),
        dry_run_cpu=bool(ns.dry_run_cpu),
    )


def _uv_prefix(uv_bin: str, python_version: str | None) -> tuple[str, ...]:
    cmd = [uv_bin, "run"]
    if python_version is not None:
        cmd.extend(["--python", python_version])
    return tuple(cmd)


def build_bundle_steps(uv_bin: str, config: BundleConfig) -> list[BundleStep]:
    """Build the ordered list of Stage 3 bundle steps respecting CLI filters."""
    uv_pre = _uv_prefix(uv_bin, config.python_version)
    py_flags: tuple[str, ...] = ("--python", config.python_version) if config.python_version else ()
    ranks_str = tuple(str(r) for r in config.ranks)
    out = config.out_dir

    if config.dry_run_cpu:
        steps = [
            BundleStep(
                name="00_host_preflight",
                commands=((*uv_pre, "python", "-c", _PREFLIGHT_SNIPPET),),
            ),
            BundleStep(
                name="05_calibration_probe",
                commands=(
                    (
                        *uv_pre,
                        "python",
                        "scripts/probes/distributed_calibration_probe.py",
                        "--backend",
                        "gloo",
                        "--sizes",
                        "tiny",
                        "--ranks",
                        "2",
                        "--seeds",
                        "11",
                        "--steps",
                        "3",
                        "--compile-repeats",
                        "2",
                        "--only",
                        "bf16-cause",
                        "calibrate",
                        "--json-out",
                        str(out / "05_calibration.json"),
                    ),
                ),
            ),
            BundleStep(
                name="06_scaling_bench",
                commands=(
                    (
                        *uv_pre,
                        "python",
                        "scripts/probes/distributed_scaling_bench.py",
                        "--backend",
                        "gloo",
                        "--sizes",
                        "tiny",
                        "--ranks",
                        "1",
                        "2",
                        "--warmup",
                        "1",
                        "--steps",
                        "2",
                        "--profile-steps",
                        "2",
                        "--json-out",
                        str(out / "06_scaling.json"),
                    ),
                ),
            ),
            BundleStep(
                name="07_linear_dispatch",
                commands=((*uv_pre, "python", "scripts/probes/linear_requires_grad_dispatch.py", "--device", "cpu"),),
            ),
        ]
    else:
        max_rank = max(config.ranks)
        transport_port = _pick_free_port()
        steps = [
            BundleStep(
                name="00_host_preflight",
                commands=(
                    (*uv_pre, "python", "-c", _PREFLIGHT_SNIPPET),
                    (
                        *uv_pre,
                        "torchrun",
                        "--master-addr=127.0.0.1",
                        f"--master-port={transport_port}",
                        f"--nproc_per_node={max_rank}",
                        "scripts/run_stage3_bundle.py",
                        "--_nccl-transport-check",
                    ),
                ),
                extra_env=(("NCCL_DEBUG", "INFO"), ("NCCL_DEBUG_SUBSYS", "INIT,GRAPH,NET,P2P")),
            ),
            BundleStep(
                name="01_t15_nccl",
                commands=(
                    (
                        *uv_pre,
                        "python",
                        "scripts/run_t15_nccl.py",
                        *py_flags,
                        "--ranks",
                        *ranks_str,
                        "--log-file",
                        str(out / "01_t15_nccl_inner.log"),
                    ),
                ),
            ),
            BundleStep(
                name="02_t15_nccl_scale02",
                commands=(
                    (
                        *uv_pre,
                        "python",
                        "scripts/run_t15_nccl.py",
                        *py_flags,
                        "--ranks",
                        *ranks_str,
                        "--f32-tol-scale",
                        "0.2",
                        "--log-file",
                        str(out / "02_t15_nccl_scale02_inner.log"),
                    ),
                ),
            ),
            BundleStep(
                name="03_multi_gpu_check_r1",
                commands=(
                    (
                        "bash",
                        "scripts/multi_gpu_check.sh",
                        *py_flags,
                        "--ranks",
                        *ranks_str,
                        "--log-dir",
                        str(out / "03_mgc_r1"),
                    ),
                ),
            ),
            BundleStep(
                name="04_multi_gpu_check_r2",
                commands=(
                    (
                        "bash",
                        "scripts/multi_gpu_check.sh",
                        *py_flags,
                        "--ranks",
                        *ranks_str,
                        "--log-dir",
                        str(out / "04_mgc_r2"),
                    ),
                ),
                post_action="diff_mgc_repeats",
            ),
            BundleStep(
                name="05_calibration_probe",
                commands=(
                    (
                        *uv_pre,
                        "python",
                        "scripts/probes/distributed_calibration_probe.py",
                        "--backend",
                        "nccl",
                        "--device-map",
                        "per-rank",
                        "--sizes",
                        *config.calib_sizes,
                        "--ranks",
                        *ranks_str,
                        "--seeds",
                        *(str(s) for s in config.seeds),
                        "--with-fp64",
                        "--json-out",
                        str(out / "05_calibration.json"),
                    ),
                ),
            ),
            BundleStep(
                name="06_scaling_bench",
                commands=(
                    (
                        *uv_pre,
                        "python",
                        "scripts/probes/distributed_scaling_bench.py",
                        "--backend",
                        "nccl",
                        "--device-map",
                        "per-rank",
                        "--sizes",
                        *config.scaling_sizes,
                        "--ranks",
                        "1",
                        *ranks_str,
                        "--json-out",
                        str(out / "06_scaling.json"),
                    ),
                ),
            ),
            BundleStep(
                name="07_linear_dispatch",
                commands=((*uv_pre, "python", "scripts/probes/linear_requires_grad_dispatch.py", "--device", "cuda"),),
            ),
        ]
        if config.include_cuda_check:
            steps.append(
                BundleStep(
                    name="08_cuda_check",
                    commands=(("bash", "scripts/cuda_check.sh"),),
                )
            )

    if config.only:
        only_set = set(config.only)
        steps = [s for s in steps if s.name in only_set]
    if config.skip:
        skip_set = set(config.skip)
        steps = [s for s in steps if s.name not in skip_set]
    return steps


# --------------------------------------------------------------------------- repeat-determinism diff


_DRIVER_ROW_RE = re.compile(
    r"^(?P<model>\S+)\s+(?P<scenario>\S+)\s+(?P<xr_eq>yes|no)\s+(?P<xr_max>\S+)\s+"
    r"(?P<param_l2>\S+)\s+(?P<upd_l2>\S+)\s+(?P<loss_rel>\S+)\s+(?P<val_rel>\S+)\s+"
    r"(?P<wall_s>\S+)\s+(?P<step_ms>\S+)\s+(?P<tail_s>\S+)\s+(?P<vram_mib>\S+)(?P<rest>.*)$"
)


def parse_driver_parity_rows(log_text: str) -> dict[tuple[str, str], tuple[str, ...]]:
    """Extract `(model, scenario) -> (xr_eq, xr_max, param_l2, upd_l2, loss_rel, val_rel, vram_mib)`."""
    rows: dict[tuple[str, str], tuple[str, ...]] = {}
    for raw in log_text.splitlines():
        line = raw.strip()
        m = _DRIVER_ROW_RE.match(line)
        if m is None:
            continue
        key = (m.group("model"), m.group("scenario"))
        if key[0] == "model" and key[1] == "scenario":
            continue
        rows[key] = (
            m.group("xr_eq"),
            m.group("xr_max"),
            m.group("param_l2"),
            m.group("upd_l2"),
            m.group("loss_rel"),
            m.group("val_rel"),
            m.group("vram_mib"),
        )
    return rows


def diff_mgc_repeats(dir_r1: Path, dir_r2: Path, out_file: Path) -> str:
    """Compare parity & `vram_mib` columns across matching driver stage logs in two `multi_gpu_check` dirs."""
    lines: list[str] = [f"=== Repeat determinism diff: {dir_r1} vs {dir_r2} ==="]
    stage_logs = sorted(p.name for p in dir_r1.glob("0[468]_*.log") if (dir_r2 / p.name).is_file())
    if not stage_logs:
        lines.append("no matching 04_*/06_*/08_* stage logs found")
    total_rows = 0
    total_diffs = 0
    for name in stage_logs:
        r1 = parse_driver_parity_rows((dir_r1 / name).read_text(encoding="utf-8", errors="replace"))
        r2 = parse_driver_parity_rows((dir_r2 / name).read_text(encoding="utf-8", errors="replace"))
        common = [k for k in r1 if k in r2]
        diffs: list[str] = []
        for k in common:
            if r1[k] != r2[k]:
                diffs.append(f"  DIFF {k[0]} {k[1]}: r1={r1[k]} vs r2={r2[k]}")
        missing = sorted(set(r1) ^ set(r2))
        total_rows += len(common)
        total_diffs += len(diffs)
        lines.append(
            f"{name}: matched={len(common) - len(diffs)}/{len(common)} diffs={len(diffs)} missing={len(missing)}"
        )
        lines.extend(diffs)
        for mk in missing:
            lines.append(f"  MISSING_KEY {mk}")
    lines.append(f"TOTAL: identical={total_rows - total_diffs}/{total_rows}, differed={total_diffs}")
    report = "\n".join(lines) + "\n"
    out_file.write_text(report, encoding="utf-8")
    return report


# --------------------------------------------------------------------------- execution helpers


def _emit(text: str, log_fh: TextIO) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()
    log_fh.write(text)
    log_fh.flush()


def _kill_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=3.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def _run_cmd(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    log_fh: TextIO,
) -> tuple[int, int | None]:
    interrupted: list[int] = []
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    def _on_sig(signum: int, _frame: Any) -> None:
        if not interrupted:
            interrupted.append(signum)
        _kill_group(proc)

    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    if prev_int != signal.SIG_IGN:
        signal.signal(signal.SIGINT, _on_sig)
    if prev_term != signal.SIG_IGN:
        signal.signal(signal.SIGTERM, _on_sig)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            _emit(line, log_fh)
        rc = proc.wait()
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        if proc.poll() is None:
            _kill_group(proc)
    if interrupted:
        return 128 + interrupted[0], interrupted[0]
    return rc, None


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("NCCL_MULTI_RANK_GPU_ENABLE", None)
    tmpdir = env.get("TMPDIR", "")
    if not tmpdir or len(tmpdir) > 60:
        env["TMPDIR"] = "/tmp"
    env.setdefault("PYTHONFAULTHANDLER", "1")
    return env


def _check_residual_gpu_procs() -> str:
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi unavailable"
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if r.returncode != 0:
        return f"nvidia-smi rc={r.returncode}"
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return "0 residual GPU compute processes" if not lines else f"{len(lines)} residual GPU processes: {lines}"


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--_nccl-transport-check" in raw_argv:
        return _run_nccl_transport_worker()
    config = parse_args(raw_argv)
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        print("error: 'uv' executable not found on PATH", file=sys.stderr)
        return 2

    try:
        config.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create --out-dir {str(config.out_dir)!r}: {exc}", file=sys.stderr)
        return 2

    steps = build_bundle_steps(uv_bin, config)
    base_env = _base_env()
    summary_rows: list[tuple[str, str, float]] = []
    overall_rc = 0
    t_all = time.monotonic()

    print(
        f"=== timestral-tune Stage 3 bundle | out_dir={config.out_dir} | "
        f"ranks={list(config.ranks)} | dry_run_cpu={config.dry_run_cpu} | "
        f"host={platform.system()} {platform.machine()} ==="
    )

    for step in steps:
        ok_marker = config.out_dir / f"{step.name}.ok"
        log_path = config.out_dir / f"{step.name}.log"
        if ok_marker.is_file() and not config.force:
            print(f"[SKIP resumed] {step.name} ({ok_marker.name} present)")
            summary_rows.append((step.name, "RESUMED", 0.0))
            continue

        print(f"\n================ [{step.name}] ================")
        t0 = time.monotonic()
        step_env = dict(base_env)
        for k, v in step.extra_env:
            step_env[k] = v

        step_rc = 0
        interrupted_sig: int | None = None
        with log_path.open("w", encoding="utf-8") as log_fh:
            for cmd in step.commands:
                _emit(f"$ {' '.join(cmd)}\n", log_fh)
                rc, sig = _run_cmd(cmd, env=step_env, cwd=REPO_ROOT, log_fh=log_fh)
                if sig is not None:
                    interrupted_sig = sig
                    step_rc = rc
                    break
                if rc != 0:
                    step_rc = rc
                    break
            if step_rc == 0 and step.post_action == "diff_mgc_repeats":
                report = diff_mgc_repeats(
                    config.out_dir / "03_mgc_r1",
                    config.out_dir / "04_mgc_r2",
                    config.out_dir / "04_repeat_diff.txt",
                )
                _emit("\n" + report, log_fh)

        elapsed = time.monotonic() - t0
        if interrupted_sig is not None:
            summary_rows.append((step.name, f"INTERRUPTED (sig {interrupted_sig})", elapsed))
            overall_rc = step_rc
            break
        if step_rc == 0:
            ok_marker.write_text(f"ok elapsed_s={elapsed:.2f}\n", encoding="utf-8")
            summary_rows.append((step.name, "PASS", elapsed))
        else:
            summary_rows.append((step.name, f"FAIL (exit {step_rc})", elapsed))
            overall_rc = 1

    total_elapsed = time.monotonic() - t_all
    gpu_check = _check_residual_gpu_procs()
    summary_lines = [
        f"=== Stage 3 bundle summary (total {total_elapsed:.1f}s) ===",
        *(f"  {name:<24} {status:<22} {sec:7.1f}s" for name, status, sec in summary_rows),
        f"GPU process check: {gpu_check}",
        f"Artifacts in: {config.out_dir}",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    (config.out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text, end="")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
