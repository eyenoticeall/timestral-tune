#!/usr/bin/env python3
"""Run acceptance test T15 and the multi-process test suite on CUDA/NCCL.

Standard-library-only runner (Python 3.10+) intended for a rented multi-GPU host. Keeps
`scripts/multi_gpu_check.py`'s frozen Stage 3 probe surface untouched while providing one
repeatable command to run the graded pytest acceptance tests on 2 GPUs and 4 GPUs (one rank
per GPU via real `DistributedContext.detect("auto")` / `LOCAL_RANK` binding) and capture a
combined log.

Examples:
    # 4-GPU host: run both 2-GPU and 4-GPU passes across T15 and the other 3 harness test files
    uv run python scripts/run_t15_nccl.py --python 3.12 --log-file t15_nccl.log

    # 2-GPU host: run the 2-GPU pass only (explicitly excludes 4-rank tests)
    uv run python scripts/run_t15_nccl.py --ranks 2 --python 3.12 --log-file t15_nccl_2gpu.log

    # T15 acceptance file only (tests/acceptance/test_distributed.py)
    uv run python scripts/run_t15_nccl.py --t15-only --python 3.12 --log-file t15_only_nccl.log

    # Single-GPU development debugging (all ranks share cuda:0; NOT separate-GPU verification)
    uv run python scripts/run_t15_nccl.py --shared-gpu --ranks 2 --log-file t15_shared.log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parent.parent

T15_TEST_FILE = "tests/acceptance/test_distributed.py"
ALL_MULTIPROC_TEST_FILES: tuple[str, ...] = (
    "tests/test_distributed.py",
    "tests/test_trainer_distributed.py",
    "tests/test_trainer_distributed_replay.py",
    T15_TEST_FILE,
)
FOUR_RANK_TEST_FILES: tuple[str, ...] = (
    "tests/test_distributed.py",
    "tests/test_trainer_distributed_replay.py",
    T15_TEST_FILE,
)

# Exact disjoint partition of the 142 items across ALL_MULTIPROC_TEST_FILES:
#   - 2-rank pass: 124 tests (18 in T15 + 106 in the other 3 files)
#   - 4-rank pass: 18 tests (16 in T15 including test_t15_f_dataloader_workers + 2 in the other 2 files)
PASS_2_K_EXPR = "not [4] and not test_t15_f_dataloader_workers"
PASS_4_K_EXPR = "[4] or test_t15_f_dataloader_workers"

_PROBE_PREFIX = "__TIMESTRAL_CUDA_PROBE__="
_CUDA_PROBE_SNIPPET = """\
import json, os, platform, torch
cuda_avail = bool(torch.cuda.is_available())
gpu_count = int(torch.cuda.device_count()) if cuda_avail else 0
gpus = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if cuda_avail else []
nccl = torch.cuda.nccl.version() if (cuda_avail and hasattr(torch.cuda, "nccl")) else None
payload = {
    "os": platform.system(),
    "arch": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": cuda_avail,
    "cuda_runtime": torch.version.cuda,
    "nccl": ".".join(str(x) for x in nccl) if nccl is not None else None,
    "gpu_count": gpu_count,
    "gpus": gpus,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
}
print("__TIMESTRAL_CUDA_PROBE__=" + json.dumps(payload))
"""


@dataclass(frozen=True)
class RunnerConfig:
    """Validated CLI configuration for `run_t15_nccl.py`."""

    ranks: tuple[int, ...]
    shared_gpu: bool
    t15_only: bool
    f32_tol_scale: float | None
    log_file: Path
    python_version: str | None
    no_sync: bool
    extra_pytest_args: tuple[str, ...]


@dataclass(frozen=True)
class PassPlan:
    """Single pytest pass targeting one world size (2 or 4)."""

    world_size: int
    label: str
    files: tuple[str, ...]
    k_expr: str


@dataclass(frozen=True)
class PytestCounts:
    """Parsed test outcome counts and `-rs` skip reasons from one pytest pass."""

    passed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: int = 0
    deselected: int = 0
    skip_reasons: tuple[str, ...] = ()
    summary_found: bool = False


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SUMMARY_FOOTER_RE = re.compile(r"^=+\s+(.+?)\s+=+$")
_COUNT_TOKEN_RE = re.compile(r"\b(\d+)\s+(passed|skipped|failed|errors?|deselected)\b")


def parse_pytest_counts(output: str) -> PytestCounts:
    """Extract outcome counts and `-rs` skip reasons from pytest's combined output."""
    clean_lines = [_ANSI_RE.sub("", line).rstrip("\r\n") for line in output.splitlines()]

    skip_reasons: list[str] = []
    in_short_summary = False
    for line in clean_lines:
        stripped = line.strip()
        if "short test summary info" in stripped and stripped.startswith("="):
            in_short_summary = True
            continue
        if in_short_summary:
            if stripped.startswith("="):
                in_short_summary = False
                continue
            if stripped.startswith("SKIPPED "):
                skip_reasons.append(stripped)

    counts = {"passed": 0, "skipped": 0, "failed": 0, "errors": 0, "deselected": 0}
    summary_found = False
    for line in reversed(clean_lines):
        m = _SUMMARY_FOOTER_RE.match(line.strip())
        if not m:
            continue
        body = m.group(1)
        tokens = _COUNT_TOKEN_RE.findall(body)
        if tokens or "no tests ran" in body:
            summary_found = True
            for num_str, kind in tokens:
                key = "errors" if kind.startswith("error") else kind
                counts[key] += int(num_str)
            break

    return PytestCounts(
        passed=counts["passed"],
        skipped=counts["skipped"],
        failed=counts["failed"],
        errors=counts["errors"],
        deselected=counts["deselected"],
        skip_reasons=tuple(skip_reasons),
        summary_found=summary_found,
    )


def evaluate_pass_result(rc: int, counts: PytestCounts) -> tuple[bool, str]:
    """Decide whether a CUDA/NCCL pytest pass succeeded.

    In CUDA mode a pass must never succeed silently: non-zero exit codes, zero passing tests,
    any failure/error, or any skipped test all fail the pass (exit code 1). Deselected tests
    (for example 4-rank tests excluded under `--ranks 2`) are expected and do not fail a pass.
    """
    if rc != 0:
        return False, f"FAIL (exit {rc})"
    if not counts.summary_found or counts.passed == 0:
        return False, "FAIL (0 tests passed)"
    if counts.failed > 0 or counts.errors > 0:
        return False, f"FAIL (failed={counts.failed}, errors={counts.errors})"
    if counts.skipped > 0:
        return False, f"FAIL (skipped={counts.skipped}; skips are not allowed in CUDA mode)"
    return True, "PASS"


def _parse_positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a finite positive float, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be a finite positive float, got {raw!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_t15_nccl.py",
        description=(
            "Run acceptance test T15 and the multi-process harness test suite on CUDA/NCCL "
            "(one rank per GPU by default) and write a combined log."
        ),
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        choices=(2, 4),
        default=[2, 4],
        help="World sizes to run (default: 2 4). Pass --ranks 2 on a 2-GPU host.",
    )
    parser.add_argument(
        "--shared-gpu",
        action="store_true",
        help=(
            "Development-only single-GPU mode: sets TIMESTRAL_TEST_DIST_DEVICE_MAP=shared and "
            "NCCL_MULTI_RANK_GPU_ENABLE=1 so all ranks share cuda:0. Not a substitute for "
            "separate-GPU verification."
        ),
    )
    parser.add_argument(
        "--t15-only",
        action="store_true",
        help="Run only tests/acceptance/test_distributed.py instead of all four multi-process test files.",
    )
    parser.add_argument(
        "--f32-tol-scale",
        type=_parse_positive_float,
        default=None,
        help="Optional TIMESTRAL_TEST_F32_TOL_SCALE multiplier (e.g. 0.1 or 0.2 to verify headroom).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("t15_nccl.log"),
        help="File to write combined stdout and stderr (default: t15_nccl.log).",
    )
    parser.add_argument(
        "--python",
        dest="python_version",
        default=None,
        help="Optional Python version forwarded to `uv run --python VERSION` (e.g. 3.12).",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Pass --no-sync to `uv run` (assumes the environment is already synced).",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to pytest (repeatable; e.g. --pytest-arg=-vv).",
    )
    return parser


def parse_args(argv: Sequence[str]) -> RunnerConfig:
    parser = build_parser()
    ns = parser.parse_args(list(argv))
    dedup_ranks: list[int] = []
    for r in ns.ranks:
        if r not in dedup_ranks:
            dedup_ranks.append(r)
    return RunnerConfig(
        ranks=tuple(dedup_ranks),
        shared_gpu=bool(ns.shared_gpu),
        t15_only=bool(ns.t15_only),
        f32_tol_scale=ns.f32_tol_scale,
        log_file=ns.log_file,
        python_version=ns.python_version,
        no_sync=bool(ns.no_sync),
        extra_pytest_args=tuple(ns.pytest_arg),
    )


def build_pass_plans(config: RunnerConfig) -> list[PassPlan]:
    """Build ordered pytest pass plans for the requested world sizes."""
    plans: list[PassPlan] = []
    for world_size in config.ranks:
        if world_size == 2:
            files = (T15_TEST_FILE,) if config.t15_only else ALL_MULTIPROC_TEST_FILES
            plans.append(
                PassPlan(
                    world_size=2,
                    label="2-GPU (world_size=2)" if not config.shared_gpu else "2-rank shared-GPU (world_size=2)",
                    files=files,
                    k_expr=PASS_2_K_EXPR,
                )
            )
        elif world_size == 4:
            files = (T15_TEST_FILE,) if config.t15_only else FOUR_RANK_TEST_FILES
            plans.append(
                PassPlan(
                    world_size=4,
                    label="4-GPU (world_size=4)" if not config.shared_gpu else "4-rank shared-GPU (world_size=4)",
                    files=files,
                    k_expr=PASS_4_K_EXPR,
                )
            )
        else:
            raise ValueError(f"unsupported world_size: {world_size}")
    return plans


def build_pass_env(base_env: Mapping[str, str], config: RunnerConfig) -> dict[str, str]:
    """Build the child environment for CUDA/NCCL pytest passes."""
    env = dict(base_env)
    env["TIMESTRAL_TEST_DIST_BACKEND"] = "nccl"
    env["TIMESTRAL_TEST_DIST_DEVICE_MAP"] = "shared" if config.shared_gpu else "per-rank"
    if config.shared_gpu:
        env["NCCL_MULTI_RANK_GPU_ENABLE"] = "1"
    else:
        # Keep NCCL's same-GPU guard active in per-rank mode even if set in the caller's shell.
        env.pop("NCCL_MULTI_RANK_GPU_ENABLE", None)
    if config.f32_tol_scale is not None:
        env["TIMESTRAL_TEST_F32_TOL_SCALE"] = str(config.f32_tol_scale)
    tmpdir = env.get("TMPDIR", "")
    if not tmpdir or len(tmpdir) > 60:
        env["TMPDIR"] = "/tmp"
    env.setdefault("PYTHONFAULTHANDLER", "1")
    return env


def build_uv_prefix(uv_bin: str, config: RunnerConfig) -> list[str]:
    cmd = [uv_bin, "run"]
    if config.no_sync:
        cmd.append("--no-sync")
    if config.python_version is not None:
        cmd.extend(["--python", config.python_version])
    return cmd


def build_pytest_command(uv_bin: str, config: RunnerConfig, plan: PassPlan) -> list[str]:
    cmd = [
        *build_uv_prefix(uv_bin, config),
        "pytest",
        "-v",
        "-rs",
        "-p",
        "no:cacheprovider",
        "-k",
        plan.k_expr,
        *config.extra_pytest_args,
        *plan.files,
    ]
    return cmd


def parse_probe_output(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_PROBE_PREFIX):
            return dict(json.loads(line[len(_PROBE_PREFIX) :]))
    raise RuntimeError(f"CUDA environment probe did not emit {_PROBE_PREFIX!r} output")


def validate_probe_for_config(probe: Mapping[str, Any], config: RunnerConfig) -> str | None:
    """Return an error string if the host lacks required GPUs for `config`, else `None`."""
    cuda_avail = bool(probe.get("cuda_available"))
    gpu_count = int(probe.get("gpu_count", 0))
    cvd = probe.get("cuda_visible_devices", "unset")
    if not cuda_avail:
        return (
            "CUDA/NCCL run requested, but PyTorch reported cuda_available=False "
            f"(torch={probe.get('torch')}, cuda_runtime={probe.get('cuda_runtime')})."
        )
    required = 1 if config.shared_gpu else max(config.ranks)
    if gpu_count < required:
        if config.shared_gpu:
            return f"--shared-gpu requires at least 1 visible CUDA GPU, found {gpu_count} (CUDA_VISIBLE_DEVICES={cvd})."
        hint = " Pass --ranks 2 on a 2-GPU host." if gpu_count >= 2 and 4 in config.ranks else ""
        return (
            f"per-rank CUDA/NCCL run for ranks {list(config.ranks)} requires at least {required} visible CUDA GPUs, "
            f"found {gpu_count} (CUDA_VISIBLE_DEVICES={cvd}).{hint}"
        )
    return None


def _emit(line: str, log_fh: TextIO) -> None:
    sys.stdout.write(line)
    sys.stdout.flush()
    log_fh.write(line)
    log_fh.flush()


def _terminate_child_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=3.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def _run_streaming(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    log_fh: TextIO,
) -> tuple[int, int | None, str]:
    """Run `cmd` in its own session, teeing combined output to stdout and `log_fh`.

    Returns `(returncode, interrupted_signum, captured_output)`.
    """
    interrupted_sig: list[int] = []
    captured: list[str] = []
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

    def _on_signal(signum: int, _frame: Any) -> None:
        if not interrupted_sig:
            interrupted_sig.append(signum)
        _terminate_child_group(proc)

    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    if prev_int != signal.SIG_IGN:
        signal.signal(signal.SIGINT, _on_signal)
    if prev_term != signal.SIG_IGN:
        signal.signal(signal.SIGTERM, _on_signal)

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            captured.append(line)
            _emit(line, log_fh)
        rc = proc.wait()
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        if proc.poll() is None:
            _terminate_child_group(proc)

    joined = "".join(captured)
    if interrupted_sig:
        return 128 + interrupted_sig[0], interrupted_sig[0], joined
    return rc, None, joined


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(sys.argv[1:] if argv is None else argv)

    uv_bin = shutil.which("uv")
    if uv_bin is None:
        print("error: 'uv' executable not found on PATH", file=sys.stderr)
        return 2

    try:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fh = config.log_file.open("w", encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot open log file {str(config.log_file)!r}: {exc}", file=sys.stderr)
        return 2

    with log_fh:
        mode_desc = (
            "SHARED SINGLE-GPU DEBUG MODE (device_map=shared, NCCL_MULTI_RANK_GPU_ENABLE=1; NOT separate-GPU)"
            if config.shared_gpu
            else "PER-RANK MULTI-GPU MODE (device_map=per-rank, real LOCAL_RANK binding)"
        )
        _emit("=== timestral-tune CUDA/NCCL multi-process test runner ===\n", log_fh)
        _emit(f"mode={mode_desc}\n", log_fh)
        _emit(f"ranks={list(config.ranks)} t15_only={config.t15_only} f32_tol_scale={config.f32_tol_scale}\n", log_fh)
        _emit(f"host_os={platform.system()} host_arch={platform.machine()} log_file={config.log_file}\n", log_fh)

        probe_cmd = [*build_uv_prefix(uv_bin, config), "python", "-c", _CUDA_PROBE_SNIPPET]
        probe_env = build_pass_env(os.environ, config)
        probe_proc = subprocess.run(
            probe_cmd,
            cwd=str(REPO_ROOT),
            env=probe_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe_proc.returncode != 0:
            _emit(f"error: CUDA probe failed (exit {probe_proc.returncode})\n", log_fh)
            if probe_proc.stderr:
                _emit(probe_proc.stderr, log_fh)
            return 2
        try:
            probe = parse_probe_output(probe_proc.stdout)
        except Exception as exc:
            _emit(f"error: {exc}\n", log_fh)
            return 2

        _emit(
            f"python={probe['python']} torch={probe['torch']} cuda_available={probe['cuda_available']} "
            f"cuda_runtime={probe['cuda_runtime']} nccl={probe['nccl']} gpu_count={probe['gpu_count']} "
            f"cuda_visible_devices={probe['cuda_visible_devices']}\n",
            log_fh,
        )
        for idx, name in enumerate(probe.get("gpus", [])):
            _emit(f"gpu[{idx}]={name}\n", log_fh)

        prereq_err = validate_probe_for_config(probe, config)
        if prereq_err is not None:
            _emit(f"error: {prereq_err}\n", log_fh)
            return 2

        if 4 not in config.ranks:
            _emit(
                "note: --ranks excluded 4, so 4-rank tests ([4] and test_t15_f_dataloader_workers) will not run.\n",
                log_fh,
            )

        plans = build_pass_plans(config)
        summary: list[tuple[str, bool, str, PytestCounts, float]] = []
        overall_rc = 0
        for plan in plans:
            cmd = build_pytest_command(uv_bin, config, plan)
            _emit(f"\n--- Running {plan.label} ---\n", log_fh)
            _emit(f"command: {' '.join(cmd)}\n", log_fh)
            t0 = time.monotonic()
            rc, interrupted, output = _run_streaming(cmd, env=probe_env, cwd=REPO_ROOT, log_fh=log_fh)
            elapsed = time.monotonic() - t0
            if interrupted is not None:
                _emit(f"\ninterrupted by signal {interrupted} after {elapsed:.1f}s\n", log_fh)
                return rc
            counts = parse_pytest_counts(output)
            ok, status = evaluate_pass_result(rc, counts)
            summary.append((plan.label, ok, status, counts, elapsed))
            if not ok:
                overall_rc = 1

        _emit("\n=== Summary ===\n", log_fh)
        for label, _ok, status, counts, elapsed in summary:
            _emit(
                f"  {label}: {status} in {elapsed:.1f}s "
                f"(passed={counts.passed}, skipped={counts.skipped}, failed={counts.failed}, "
                f"errors={counts.errors}, deselected={counts.deselected})\n",
                log_fh,
            )
            for reason in counts.skip_reasons:
                _emit(f"    skip: {reason}\n", log_fh)
        _emit(f"Log written to: {config.log_file}\n", log_fh)
        return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
