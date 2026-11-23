"""Portable, resumable Stage 3 multi-GPU / multi-process verification session driver.

Runs the full Stage 3 verification sequence (environment check, prototype CUDA parity on 4 GPUs,
one-process references, `torchrun` trainer checks, and rank-failure timeout probes) under a single
standard-library-only Python process (Python >= 3.10; never imports `torch` or third-party packages).

Usage (either directly or through the thin `scripts/multi_gpu_check.sh` wrapper):

    # Four-GPU machine (auto-detects visible GPUs via PyTorch, runs N=2 and N=4 on NCCL + CPU gloo):
    bash scripts/multi_gpu_check.sh --python 3.12

    # Single-GPU shared NCCL bench (exports NCCL_MULTI_RANK_GPU_ENABLE=1, runs N=2 and N=4 on cuda:0):
    bash scripts/multi_gpu_check.sh --shared-gpu --python 3.12

    # CPU-only gloo dry run:
    bash scripts/multi_gpu_check.sh --backend gloo --ranks 2 4 --log-dir /tmp/multi_gpu_dry

Portability and process management:
- Uses no `sudo`, cloud-specific commands, GNU `timeout`, `nvidia-smi`, or user/home assumptions.
- Streams stage output directly to stdout and `<log-dir>/<stage>.log` via a non-blocking pipe
  (no shell `| tee` pipeline that orphaned descendants could hold open) and drains continuously
  through `SIGTERM` grace and `SIGKILL` teardown so shutting-down workers never block on a full pipe.
- Signal handling: for each of `SIGINT`, `SIGTERM`, and `SIGHUP`, inspects the disposition inherited
  on entry. Any signal set to `SIG_IGN` on entry (such as `SIGHUP` under `nohup` over SSH) remains
  ignored for the entire session across start-up, mid-stage, timeout teardown, and stage boundaries;
  child stages run in their own session (`start_new_session=True`), so terminal/group signals sent to
  the orchestrator never reach them and they inherit `SIG_IGN` via standard POSIX `exec` semantics.
  Any signal not ignored on entry interrupts cleanly at every moment: kills the active stage tree,
  removes incomplete `.done` markers and partial artifacts, writes an interrupted `summary.txt`,
  skips remaining stages, and exits `128 + signum`.
- Runs each stage in its own session (`start_new_session=True`) tagged with an inherited per-stage
  environment token (`_TIMESTRAL_STAGE_TOKEN`), and re-collects confirmed-live non-zombie PIDs from
  `/proc` or `ps` immediately before every signal so reaped/reused PIDs are never signalled.
- Limitation: a descendant that clears its environment (`_TIMESTRAL_STAGE_TOKEN`) and outlives its
  parent (so it is reparented out of the tracked PPID tree — even without double-forking or calling
  `setsid()`, because a PGID is signalled only when every live member is a known target), or runs on
  a system with neither `/proc` nor `ps`, can escape tree termination; neither `torchrun` nor PyTorch
  `DataLoader` workers do this.

Resume markers (`<stage>.done`) are content-addressed over tree state (git HEAD + tracked status/diff
+ untracked files under `scripts/` and `src/`, or deterministic content hashing of `scripts/`, `src/`,
`pyproject.toml`, and `uv.lock` excluding `--log-dir` and Python bytecode caches when git is absent
or cwd is not a git work-tree top-level), resolved Python/torch/CUDA runtime identity, exact stage
arguments separated by a domain boundary from tracked environment variables (`NCCL_*`, `GLOO_*`,
`TORCH_*`, `PYTORCH_*`, `CUDA_*`, `CUDNN_*`, `CUBLAS_*`, `OMP_*`, `MKL_*`, `OPENBLAS_*`), and the
verified SHA-256 content digest of any reference archive produced or consumed by the stage.

See `docs/phases/phase-5-multi-gpu.md` §3 and `docs/SPEC.md` §5.6, §12.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent

Backend = Literal["nccl", "gloo"]
ReferenceMode = Literal["auto", "sketch", "exact"]
TreeStateMethod = Literal["git", "content_hash"]

VALID_SIZES: tuple[str, ...] = ("5M", "28M", "123M", "209M")
TRACKED_ENV_PREFIXES: tuple[str, ...] = (
    "NCCL_",
    "GLOO_",
    "TORCH_",
    "PYTORCH_",
    "CUDA_",
    "CUDNN_",
    "CUBLAS_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
)
MANAGED_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_STAGE_TOKEN_ENV_KEY = "_TIMESTRAL_STAGE_TOKEN"
_DTYPE_NBYTES: dict[str, int] = {"float64": 8, "float32": 4, "bfloat16": 2, "float16": 2}
_POS_NUMBER_RE = re.compile(r"^[0-9]+([.][0-9]+)?$")
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

USAGE = """\
Usage: bash scripts/multi_gpu_check.sh [options]

Options:
  --backend nccl|gloo        Backend to verify (default: nccl).
  --shared-gpu               Run 2 and 4 NCCL ranks on one GPU (exports NCCL_MULTI_RANK_GPU_ENABLE=1
                             and passes --device-map shared).
  --ranks N [N ...]          World sizes to test: space-, quote-, or comma-separated (default: all of
                             {2, 4} that fit visible GPUs, or "2 4" under --shared-gpu / --backend gloo).
  --sizes SIZE [SIZE ...]    Released checkpoint sizes for NCCL runs: space-, quote-, or comma-separated
                             (default: "5M 28M" for nccl, empty for gloo; pass "" to skip released sizes).
  --python VERSION           Python version passed to `uv` (e.g. 3.12).
  --log-dir PATH             Directory for per-stage logs, reference archives, and completion markers
                             (default: multi_gpu_check_logs).
  --force                    Re-run stages even if already marked completed in --log-dir.
  --reference-mode MODE      Reference storage for --write-reference: auto (default), sketch, or exact.
  --pg-timeout SEC           Process-group timeout for the rank-failure check (default: 20).
  --step-timeout SEC         init_process_group timeout for distributed_trainer_check.py (default: 180).
  --stage-timeout SEC        Hard wall-clock limit per stage in seconds (default: 900).
  --only SCENARIO            Forward --only SCENARIO to distributed_trainer_check.py (repeatable).
  --skip SCENARIO            Forward --skip SCENARIO to distributed_trainer_check.py (repeatable).
  -h, --help                 Show this help message."""

_RUNTIME_ENV_SIG_SNIPPET = """\
import hashlib, platform, sys, torch
h = hashlib.sha256()
for part in (
    sys.executable,
    sys.version,
    platform.python_version(),
    torch.__version__,
    str(torch.version.cuda),
):
    h.update(part.encode("utf-8"))
    h.update(b"\\0")
print(h.hexdigest())
"""

_GPU_COUNT_SNIPPET = "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"

_ENV_PROBE_SNIPPET = """\
import os, platform, shutil, subprocess, sys, sysconfig, torch

backend, tree_method, tree_sig_short = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"os={platform.system()} arch={platform.machine()} python={platform.python_version()}")
print(f"tree_state_method={tree_method} tree_state_sig={tree_sig_short}")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_runtime={torch.version.cuda}")
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
print(f"cuda_visible_devices={cvd}")
if torch.cuda.is_available():
    nccl = torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else None
    nccl_str = ".".join(str(x) for x in nccl) if nccl is not None else "unknown"
    print(f"nccl={nccl_str} gpu_count={torch.cuda.device_count()}")
    if shutil.which("nvidia-smi") is not None:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False, capture_output=True, text=True,
        )
        if out.returncode == 0:
            for idx, line in enumerate(out.stdout.strip().splitlines()):
                print(f"gpu[{idx}]={line.strip()}")
    else:
        print("nvidia_smi=unavailable")
has_header = os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h"))
has_cc = shutil.which("cc") is not None or shutil.which("gcc") is not None
print(f"python_h={has_header} c_compiler={has_cc}")
nccl_shared = os.environ.get("NCCL_MULTI_RANK_GPU_ENABLE", "unset")
print(f"nccl_multi_rank_gpu_enable={nccl_shared}")
if backend == "nccl" and not torch.cuda.is_available():
    print("error: --backend nccl requested, but this PyTorch build cannot use CUDA", file=sys.stderr)
    sys.exit(3)
"""

_ENV_STAGE_BOOTSTRAP = """\
import subprocess, sys

uv_bin, backend, tree_method, tree_sig_short, probe_code = sys.argv[1:6]
uv_py_args = sys.argv[6:]
r1 = subprocess.run([uv_bin, "sync", "--locked", "--quiet", *uv_py_args], check=False)
if r1.returncode != 0:
    sys.exit(r1.returncode)
r2 = subprocess.run(
    [uv_bin, "run", "--no-sync", *uv_py_args, "python", "-c", probe_code, backend, tree_method, tree_sig_short],
    check=False,
)
sys.exit(r2.returncode)
"""


class CliError(Exception):
    """Raised when CLI arguments fail validation (mapped to exit code 2)."""

    def __init__(self, message: str, *, show_usage: bool = False) -> None:
        super().__init__(message)
        self.show_usage = show_usage


class HelpRequested(Exception):
    """Raised when `-h` or `--help` is passed (mapped to exit code 0)."""


@dataclass(frozen=True)
class SessionConfig:
    """Validated configuration for one multi-GPU verification session."""

    backend: Backend = "nccl"
    shared_gpu: bool = False
    ranks: tuple[int, ...] = ()
    ranks_specified: bool = False
    sizes: tuple[str, ...] = ("5M", "28M")
    python_version: str = ""
    log_dir: Path = Path("multi_gpu_check_logs")
    force: bool = False
    reference_mode: ReferenceMode = "auto"
    pg_timeout: str = "20"
    step_timeout: str = "180"
    stage_timeout: float = 900.0
    stage_timeout_raw: str = "900"
    extra_driver_args: tuple[str, ...] = ()

    @property
    def device_map(self) -> Literal["per-rank", "shared"]:
        return "shared" if self.shared_gpu else "per-rank"

    @property
    def uv_py_args(self) -> tuple[str, ...]:
        return ("--python", self.python_version) if self.python_version else ()


@dataclass(frozen=True)
class StageRecord:
    """One row recorded in `summary.txt`."""

    name: str
    status: str
    log_file: str


@dataclass(frozen=True)
class StageOutcome:
    """Execution outcome of a single stage run."""

    ok: bool
    returncode: int
    resumed: bool = False
    artifact_sha256: str = ""
    interrupted_sig: int | None = None


# --------------------------------------------------------------------------- CLI parsing


def split_tokens(raw: str) -> list[str]:
    """Split a space- or comma-separated CLI value into non-empty tokens without shell globbing."""
    return [tok for tok in raw.replace(",", " ").split() if tok]


def validate_positive_number(opt: str, val: str) -> float:
    """Validate that `val` is a finite positive decimal number and return its float value."""
    if not _POS_NUMBER_RE.match(val):
        raise CliError(f"error: {opt} must be a finite positive number, got '{val}'")
    parsed = float(val)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise CliError(f"error: {opt} must be a finite positive number, got '{val}'")
    return parsed


def parse_args(argv: Sequence[str]) -> SessionConfig:
    """Parse and validate CLI arguments identically to the shell interface."""
    backend_raw = "nccl"
    shared_gpu = False
    ranks_raw = ""
    ranks_specified = False
    sizes_raw: str | None = None
    sizes_specified = False
    python_version = ""
    log_dir_raw = "multi_gpu_check_logs"
    force = False
    reference_mode_raw = "auto"
    pg_timeout_raw = "20"
    step_timeout_raw = "180"
    stage_timeout_raw = "900"
    extra_driver_args: list[str] = []

    i = 0
    n_args = len(argv)

    def _require_val(opt: str) -> str:
        nonlocal i
        if i + 1 >= n_args:
            raise CliError(f"error: option '{opt}' requires an argument")
        i += 1
        return argv[i]

    while i < n_args:
        arg = argv[i]
        if arg == "--backend":
            backend_raw = _require_val(arg)
        elif arg == "--shared-gpu":
            shared_gpu = True
        elif arg == "--ranks":
            ranks_specified = True
            parts = [_require_val(arg)]
            while i + 1 < n_args and not argv[i + 1].startswith("-"):
                i += 1
                parts.append(argv[i])
            ranks_raw = " ".join(parts)
        elif arg == "--sizes":
            sizes_specified = True
            parts = [_require_val(arg)]
            while i + 1 < n_args and not argv[i + 1].startswith("-"):
                i += 1
                parts.append(argv[i])
            sizes_raw = " ".join(parts)
        elif arg == "--python":
            python_version = _require_val(arg)
        elif arg == "--log-dir":
            log_dir_raw = _require_val(arg)
        elif arg == "--force":
            force = True
        elif arg == "--reference-mode":
            reference_mode_raw = _require_val(arg)
        elif arg == "--pg-timeout":
            pg_timeout_raw = _require_val(arg)
        elif arg == "--step-timeout":
            step_timeout_raw = _require_val(arg)
        elif arg == "--stage-timeout":
            stage_timeout_raw = _require_val(arg)
        elif arg in ("--only", "--skip"):
            val = _require_val(arg)
            extra_driver_args.extend((arg, val))
        elif arg in ("-h", "--help"):
            raise HelpRequested()
        else:
            raise CliError(f"error: unknown option '{arg}'", show_usage=True)
        i += 1

    if backend_raw not in ("nccl", "gloo"):
        raise CliError(f"error: --backend must be nccl or gloo, got '{backend_raw}'")
    backend: Backend = "nccl" if backend_raw == "nccl" else "gloo"

    if backend == "gloo" and shared_gpu:
        raise CliError("error: --shared-gpu applies only to --backend nccl")

    if reference_mode_raw not in ("auto", "sketch", "exact"):
        raise CliError(f"error: --reference-mode must be auto, sketch, or exact, got '{reference_mode_raw}'")
    reference_mode: ReferenceMode = (
        "auto" if reference_mode_raw == "auto" else ("sketch" if reference_mode_raw == "sketch" else "exact")
    )

    validate_positive_number("--pg-timeout", pg_timeout_raw)
    validate_positive_number("--step-timeout", step_timeout_raw)
    stage_timeout = validate_positive_number("--stage-timeout", stage_timeout_raw)

    ranks: list[int] = []
    if ranks_specified:
        rank_tokens = split_tokens(ranks_raw)
        if not rank_tokens:
            raise CliError(f"error: --ranks must contain at least one world size (integer >= 2), got '{ranks_raw}'")
        for tok in rank_tokens:
            if not tok.isdigit() or int(tok) < 2:
                raise CliError(f"error: each entry in --ranks must be an integer >= 2, got '{tok}'")
            ranks.append(int(tok))

    if not sizes_specified and sizes_raw is None:
        sizes_raw = "5M 28M" if backend == "nccl" else ""
    sizes: list[str] = []
    if sizes_raw:
        size_tokens = split_tokens(sizes_raw)
        if not size_tokens:
            raise CliError('error: --sizes contained no valid size tokens (pass "" to skip released sizes)')
        for sz in size_tokens:
            if sz not in VALID_SIZES:
                raise CliError(f"error: invalid checkpoint size '{sz}' (expected one of: 5M, 28M, 123M, 209M)")
            sizes.append(sz)

    return SessionConfig(
        backend=backend,
        shared_gpu=shared_gpu,
        ranks=tuple(ranks),
        ranks_specified=ranks_specified,
        sizes=tuple(sizes),
        python_version=python_version,
        log_dir=Path(log_dir_raw),
        force=force,
        reference_mode=reference_mode,
        pg_timeout=pg_timeout_raw,
        step_timeout=step_timeout_raw,
        stage_timeout=stage_timeout,
        stage_timeout_raw=stage_timeout_raw,
        extra_driver_args=tuple(extra_driver_args),
    )


# --------------------------------------------------------------------------- signatures & archives


def _is_inside_dir(path: Path, ancestor_resolved: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == ancestor_resolved or ancestor_resolved in resolved.parents


def _is_bytecode_cache(name: str) -> bool:
    return name == "__pycache__" or name.endswith((".pyc", ".pyo"))


def _try_git_tree_state(repo_root: Path, log_dir_resolved: Path, *, env: Mapping[str, str] | None = None) -> str | None:
    """Compute git-based tree digest when `git` is available and `repo_root` is a work-tree top-level."""
    path_env = (env or os.environ).get("PATH")
    git_bin = shutil.which("git", path=path_env)
    if git_bin is None:
        return None

    def _run(args: Sequence[str]) -> tuple[int, bytes]:
        try:
            proc = subprocess.run(
                [git_bin, *args],
                cwd=str(repo_root),
                env=None if env is None else dict(env),
                check=False,
                capture_output=True,
            )
        except OSError:
            return 1, b""
        return proc.returncode, proc.stdout

    rc_wt, out_wt = _run(["rev-parse", "--is-inside-work-tree"])
    if rc_wt != 0 or out_wt.strip() != b"true":
        return None
    rc_top, out_top = _run(["rev-parse", "--show-toplevel"])
    if rc_top != 0:
        return None
    try:
        top_resolved = Path(os.fsdecode(out_top.strip())).resolve()
    except OSError:
        return None
    if top_resolved != repo_root.resolve():
        return None

    rc_head, head = _run(["rev-parse", "HEAD"])
    rc_status, status = _run(["status", "--porcelain=v1", "-z", "--untracked-files=no"])
    rc_diff, diff = _run(["diff", "HEAD"])
    rc_untracked, untracked = _run(["ls-files", "--others", "--exclude-standard", "-z", "scripts", "src"])
    if rc_head != 0 or not head.strip() or rc_status != 0 or rc_diff != 0 or rc_untracked != 0:
        return None

    h = hashlib.sha256()
    h.update(b"method=git\0")
    h.update(b"head=")
    h.update(head.strip())
    h.update(b"\0")
    h.update(b"status=")
    h.update(status)
    h.update(b"\0")
    h.update(b"diff=")
    h.update(diff)
    h.update(b"\0")

    if untracked:
        for rel_b in sorted(x for x in untracked.split(b"\0") if x):
            rel_path = Path(os.fsdecode(rel_b))
            if any(_is_bytecode_cache(part) for part in rel_path.parts):
                continue
            full_path = repo_root / rel_path
            if _is_inside_dir(full_path, log_dir_resolved):
                continue
            h.update(b"untracked=")
            h.update(rel_b)
            h.update(b"\0")
            if full_path.is_symlink():
                h.update(b"symlink=")
                h.update(os.fsencode(os.readlink(full_path)))
            elif full_path.is_file():
                with open(full_path, "rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        h.update(chunk)
            h.update(b"\0")
    return h.hexdigest()


def _content_hash_tree_state(repo_root: Path, log_dir_resolved: Path) -> str:
    """Deterministically hash `scripts/`, `src/`, `pyproject.toml`, and `uv.lock`."""
    h = hashlib.sha256()
    h.update(b"method=content_hash\0")
    hashed_files = 0

    def _hash_file(rel_posix: str, fp: Path) -> None:
        nonlocal hashed_files
        h.update(b"file=")
        h.update(rel_posix.encode("utf-8", errors="surrogateescape"))
        h.update(b"\0")
        size = fp.stat().st_size
        h.update(f"{size}:".encode("ascii"))
        with open(fp, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        h.update(b"\0")
        hashed_files += 1

    for root_name in ("scripts", "src"):
        root_p = repo_root / root_name
        if not root_p.is_dir():
            h.update(b"missing_dir=")
            h.update(root_name.encode("ascii"))
            h.update(b"\0")
            continue
        for dirpath, dirnames, filenames in os.walk(root_p, topdown=True, followlinks=False):
            dp = Path(dirpath)
            dirnames[:] = sorted(
                d for d in dirnames if not _is_bytecode_cache(d) and not _is_inside_dir(dp / d, log_dir_resolved)
            )
            for fn in sorted(filenames):
                if _is_bytecode_cache(fn):
                    continue
                fp = dp / fn
                if _is_inside_dir(fp, log_dir_resolved):
                    continue
                rel_posix = fp.relative_to(repo_root).as_posix()
                if fp.is_symlink():
                    h.update(b"symlink=")
                    h.update(rel_posix.encode("utf-8", errors="surrogateescape"))
                    h.update(b"\0")
                    h.update(os.fsencode(os.readlink(fp)))
                    h.update(b"\0")
                    hashed_files += 1
                elif fp.is_file():
                    _hash_file(rel_posix, fp)

    for top_file in ("pyproject.toml", "uv.lock"):
        fp = repo_root / top_file
        if fp.is_symlink():
            h.update(b"symlink=")
            h.update(top_file.encode("ascii"))
            h.update(b"\0")
            h.update(os.fsencode(os.readlink(fp)))
            h.update(b"\0")
            hashed_files += 1
        elif fp.is_file():
            _hash_file(top_file, fp)
        else:
            h.update(b"missing_file=")
            h.update(top_file.encode("ascii"))
            h.update(b"\0")

    if hashed_files == 0:
        raise RuntimeError("content_hash found no source files under scripts/, src/, pyproject.toml, or uv.lock")
    return h.hexdigest()


def compute_tree_state_sig(
    repo_root: Path,
    log_dir: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[TreeStateMethod, str]:
    """Return `(method, sha256_hex)` describing the current repository tree state.

    Uses git when `git` is on `PATH` and `repo_root` is the top-level of a git work tree; otherwise
    falls back to deterministic content hashing of `scripts/`, `src/`, `pyproject.toml`, and `uv.lock`
    excluding `log_dir` and Python bytecode caches. Raises `RuntimeError` if no valid digest can be
    produced so callers never proceed with an empty or constant signature.
    """
    log_dir_resolved = log_dir.resolve()
    git_digest = _try_git_tree_state(repo_root, log_dir_resolved, env=env)
    if git_digest is not None:
        method: TreeStateMethod = "git"
        digest = git_digest
    else:
        method = "content_hash"
        digest = _content_hash_tree_state(repo_root, log_dir_resolved)

    if not _HEX_SHA256_RE.match(digest):
        raise RuntimeError(f"invalid tree-state digest {digest!r} from method {method!r}")
    return method, digest


def stage_signature(
    tree_method: str,
    tree_sig: str,
    runtime_env_sig: str,
    parts: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Compute a collision-free SHA-256 stage signature over arguments and tracked env variables."""
    source_env = os.environ if env is None else env
    h = hashlib.sha256()
    h.update(b"args\0")
    for part in (tree_method, tree_sig, runtime_env_sig, *parts):
        b = part.encode("utf-8")
        h.update(f"{len(b)}:".encode("ascii"))
        h.update(b)
        h.update(b"\0")
    h.update(b"--env--\0")
    for env_k in sorted(source_env):
        if env_k.startswith(TRACKED_ENV_PREFIXES):
            kb = env_k.encode("utf-8")
            vb = source_env[env_k].encode("utf-8")
            h.update(f"{len(kb)}:".encode("ascii"))
            h.update(kb)
            h.update(b"=")
            h.update(f"{len(vb)}:".encode("ascii"))
            h.update(vb)
            h.update(b"\0")
    return h.hexdigest()


def verify_and_hash_ref_archive(ref_path: Path) -> str | None:
    """Verify zip CRC integrity, metadata JSON, and exact tensor member byte sizes; return file SHA-256."""
    try:
        if not ref_path.is_file() or ref_path.stat().st_size <= 0:
            return None
        with zipfile.ZipFile(ref_path, "r") as zf:
            if zf.testzip() is not None:
                return None
            member_sizes = {info.filename: info.file_size for info in zf.infolist()}
            if "spec.json" not in member_sizes or "entries.json" not in member_sizes:
                return None
            meta = json.loads(zf.read("spec.json").decode("utf-8"))
            entries = json.loads(zf.read("entries.json").decode("utf-8"))
            if not isinstance(meta, dict) or not isinstance(meta.get("spec"), dict) or not isinstance(entries, dict):
                return None
            for entry in entries.values():
                if not isinstance(entry, dict):
                    return None
                member = entry.get("tensor_member")
                dtype_name = entry.get("tensor_dtype")
                shape = entry.get("tensor_shape")
                if not isinstance(member, str) or member not in member_sizes:
                    return None
                if dtype_name not in _DTYPE_NBYTES or not isinstance(shape, list):
                    return None
                expected_bytes = math.prod(int(d) for d in shape) * _DTYPE_NBYTES[dtype_name]
                if member_sizes[member] != expected_bytes:
                    return None
    except Exception:
        return None

    h = hashlib.sha256()
    try:
        with open(ref_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def pick_free_port() -> int:
    """Bind an ephemeral IPv4 loopback socket and return its free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def signal_names(sig_num: int) -> tuple[str, str]:
    """Return `(full_name, short_name)` such as `('SIGINT', 'INT')` for `sig_num`."""
    try:
        sig_label = signal.Signals(sig_num).name
    except ValueError:
        sig_label = f"SIG{sig_num}"
    short_sig = sig_label[3:] if sig_label.startswith("SIG") else sig_label
    return sig_label, short_sig


# --------------------------------------------------------------------------- process-tree inspection & cleanup


def _resolve_ps_bin(env: Mapping[str, str] | None = None) -> str | None:
    path_env = (env or os.environ).get("PATH")
    found = shutil.which("ps", path=path_env)
    if found is not None:
        return found
    return "/bin/ps" if os.path.isfile("/bin/ps") else None


def read_proc_table(ps_bin: str | None) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Return `(children_by_ppid, pgid_by_pid)` of live (non-zombie) processes via `/proc` or `ps`."""
    children: dict[int, list[int]] = {}
    pgids: dict[int, int] = {}
    if os.path.isdir("/proc"):
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                stat = Path(f"/proc/{pid}/stat").read_bytes()
                rparen = stat.rfind(b")")
                if rparen < 0:
                    continue
                fields = stat[rparen + 2 :].split()
                if len(fields) < 3 or fields[0] == b"Z":
                    continue
                ppid = int(fields[1])
                pgid = int(fields[2])
                children.setdefault(ppid, []).append(pid)
                pgids[pid] = pgid
            except OSError:
                continue
        if pgids:
            return children, pgids
    if ps_bin is not None:
        try:
            out = subprocess.run(
                [ps_bin, "-axo", "pid=,ppid=,pgid=,stat="],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 4 and all(p.isdigit() for p in parts[:3]):
                        if parts[3].startswith("Z"):
                            continue
                        pid, ppid, pgid = int(parts[0]), int(parts[1]), int(parts[2])
                        children.setdefault(ppid, []).append(pid)
                        pgids[pid] = pgid
        except Exception:
            pass
    return children, pgids


def find_token_pids(token_needle: str, *, self_pid: int, ps_bin: str | None) -> set[int]:
    """Find live processes carrying our inherited stage environment token (catches reparented/new-session workers)."""
    found: set[int] = set()
    if os.path.isdir("/proc"):
        needle_b = token_needle.encode("utf-8")
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == self_pid:
                continue
            try:
                env_bytes = Path(f"/proc/{pid}/environ").read_bytes()
                if needle_b in env_bytes:
                    found.add(pid)
            except OSError:
                continue
        return found
    if ps_bin is not None:
        try:
            out = subprocess.run(
                [ps_bin, "eww", "-axo", "pid=,command="],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    if token_needle not in line:
                        continue
                    parts = line.strip().split(None, 1)
                    if parts and parts[0].isdigit():
                        pid = int(parts[0])
                        if pid != self_pid:
                            found.add(pid)
        except Exception:
            pass
    return found


def collect_live_targets(
    proc: subprocess.Popen[Any],
    token_needle: str,
    *,
    self_pid: int,
    self_pgid: int,
    ps_bin: str | None,
) -> tuple[set[int], set[int]]:
    """Collect only currently confirmed-live stage PIDs and safe PGIDs (never reaped/reused PIDs)."""
    children, pgids = read_proc_table(ps_bin)
    token_pids = find_token_pids(token_needle, self_pid=self_pid, ps_bin=ps_bin)

    seeds = [p for p in token_pids if p != self_pid and (not pgids or p in pgids)]
    # Only seed proc.pid while our un-reaped Popen handle confirms it has not exited yet so the
    # kernel cannot have reused proc.pid for an unrelated process.
    if proc.poll() is None and proc.pid != self_pid:
        seeds.append(proc.pid)

    live_pids: set[int] = set()
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        if cur in live_pids or cur == self_pid:
            continue
        if pgids and cur not in pgids and cur != proc.pid:
            continue
        live_pids.add(cur)
        for ch in children.get(cur, ()):
            stack.append(ch)

    # Signal a PGID only when every observed live member belongs to this stage (and never our own PGID).
    members_by_pgid: dict[int, set[int]] = {}
    for pid, pg in pgids.items():
        members_by_pgid.setdefault(pg, set()).add(pid)

    live_pgids: set[int] = set()
    for pid in live_pids:
        pg = pgids.get(pid)
        if pg is not None and pg > 0 and pg != self_pgid:
            members = members_by_pgid.get(pg, set())
            if members and members.issubset(live_pids):
                live_pgids.add(pg)
    if proc.poll() is None and proc.pid != self_pgid and not pgids:
        live_pgids.add(proc.pid)

    return live_pids, live_pgids


def signal_live(pids: set[int], pgids: set[int], sig: signal.Signals) -> None:
    """Deliver `sig` to safe stage PGIDs and live stage PIDs, ignoring already-exited targets."""
    for pg in pgids:
        try:
            os.killpg(pg, sig)
        except OSError:
            pass
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


# --------------------------------------------------------------------------- resume markers & summary


def write_done_marker(done_file: Path, sig: str, artifact_sha: str = "") -> None:
    """Write a stage completion marker binding the stage signature and optional artifact digest."""
    lines = [f"sig={sig}"]
    if artifact_sha:
        lines.append(f"artifact_sha256={artifact_sha}")
    done_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_done_marker(done_file: Path) -> tuple[str, str]:
    """Return `(saved_sig, saved_artifact_sha)` parsed from `done_file`, or empty strings."""
    saved_sig = ""
    saved_sha = ""
    try:
        content = done_file.read_text(encoding="utf-8")
    except OSError:
        return "", ""
    for line in content.splitlines():
        if line.startswith("sig="):
            saved_sig = line[4:]
        elif line.startswith("artifact_sha256="):
            saved_sha = line[16:]
    return saved_sig, saved_sha


def format_summary(
    *,
    backend: Backend,
    device_map: str,
    ranks: Sequence[int],
    stages: Sequence[StageRecord],
    interrupted_sig: str = "",
    interrupted_code: int = 0,
) -> str:
    """Render `summary.txt` using relative stage log filenames so no absolute paths or usernames leak."""
    hdr = f"backend={backend}"
    if backend == "nccl":
        hdr += f", device_map={device_map}"
    if ranks:
        hdr += f", ranks={' '.join(str(r) for r in ranks)}"
    else:
        hdr += ", ranks=none"
    if interrupted_sig:
        hdr += f", INTERRUPTED by SIG{interrupted_sig} (exit {interrupted_code})"
    lines = [
        "",
        f"== multi_gpu_check summary ({hdr}; logs relative to --log-dir) ==",
        f"{'stage':<36} {'exit_code':<28} log_file",
        f"{'-' * 36} {'-' * 28} {'-' * 40}",
    ]
    for rec in stages:
        lines.append(f"{rec.name:<36} {rec.status:<28} {rec.log_file}")
    if interrupted_sig:
        lines.append("")
        lines.append(
            f"Session interrupted by SIG{interrupted_sig} (exit {interrupted_code}); remaining stages were not started."
        )
    return "\n".join(lines) + "\n"


def _prepare_session_env(base_env: Mapping[str, str], *, shared_gpu: bool) -> dict[str, str]:
    """Apply required session environment defaults (`TMPDIR`, `PYTHONFAULTHANDLER`, shared-GPU NCCL)."""
    env = dict(base_env)
    # Keep TMPDIR short so PyTorch's torch_shm_manager Unix socket stays below the 103-byte macOS limit.
    tmpdir = env.get("TMPDIR", "")
    if not tmpdir or len(tmpdir) > 60:
        env["TMPDIR"] = "/tmp"
    env["PYTHONFAULTHANDLER"] = "1"
    if shared_gpu:
        env["NCCL_MULTI_RANK_GPU_ENABLE"] = "1"
    return env


# --------------------------------------------------------------------------- session runner


@dataclass
class SessionRunner:
    """Single-process orchestrator for a resumable multi-GPU verification session."""

    config: SessionConfig
    repo_root: Path = REPO_ROOT
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    uv_bin: str = "uv"
    teardown_grace_s: float = 3.0
    teardown_reap_s: float = 2.0
    on_before_stage_spawn: Callable[[str], None] | None = None
    on_timeout_teardown: Callable[[str], None] | None = None

    stages: list[StageRecord] = field(default_factory=list, init=False)
    ranks: list[int] = field(default_factory=list, init=False)
    any_failed: int = field(default=0, init=False)
    tree_method: TreeStateMethod = field(default="git", init=False)
    tree_sig: str = field(default="", init=False)
    runtime_env_sig: str = field(default="uninitialized", init=False)
    interrupted_sig_num: int | None = field(default=None, init=False)
    interrupted_short_sig: str = field(default="", init=False)
    interrupted_exit_code: int = field(default=0, init=False)
    _stdout_open: bool = field(default=True, init=False)
    _ps_bin: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.env = _prepare_session_env(self.env, shared_gpu=self.config.shared_gpu)
        self.ranks = list(self.config.ranks)
        self._ps_bin = _resolve_ps_bin(self.env)

    @property
    def log_dir(self) -> Path:
        if self.config.log_dir.is_absolute():
            return self.config.log_dir
        return self.repo_root / self.config.log_dir

    @property
    def refs_dir(self) -> Path:
        return self.log_dir / "refs"

    def _on_signal(self, signum: int, _frame: Any) -> None:
        # Record first signal only; ignore duplicates so teardown cannot be aborted mid-cleanup.
        if self.interrupted_sig_num is None:
            self.interrupted_sig_num = signum

    def _install_signal_handlers(self) -> dict[int, Any]:
        """Install handlers for managed signals not ignored on entry and return previous dispositions.

        Preserves `SIG_IGN` inherited on entry (for example `SIGHUP` under `nohup` over SSH) for the
        entire session. Child stages run in their own session (`start_new_session=True`) and inherit
        `SIG_IGN` under POSIX `exec`, so neither terminal/group signals nor direct inheritance breaks
        `nohup`.
        """
        previous: dict[int, Any] = {}
        for sig in MANAGED_SIGNALS:
            prev = signal.getsignal(sig)
            if prev == signal.SIG_IGN:
                continue
            previous[int(sig)] = prev
            signal.signal(sig, self._on_signal)
        return previous

    def _restore_signal_handlers(self, previous: Mapping[int, Any]) -> None:
        for sig_int, handler in previous.items():
            try:
                signal.signal(sig_int, handler)
            except (OSError, ValueError):
                pass

    def _write_stdout_bytes(self, data: bytes) -> None:
        if not self._stdout_open:
            return
        try:
            buf = getattr(sys.stdout, "buffer", None)
            if buf is not None:
                buf.write(data)
                buf.flush()
            else:
                sys.stdout.write(data.decode("utf-8", errors="replace"))
                sys.stdout.flush()
        except OSError:
            self._stdout_open = False

    def _write_stderr_bytes(self, data: bytes) -> None:
        try:
            buf = getattr(sys.stderr, "buffer", None)
            if buf is not None:
                buf.write(data)
                buf.flush()
            else:
                sys.stderr.write(data.decode("utf-8", errors="replace"))
                sys.stderr.flush()
        except OSError:
            pass

    def _print_line(self, text: str) -> None:
        self._write_stdout_bytes((text + "\n").encode("utf-8"))

    def record_stage(self, name: str, status: str, log_file: str) -> None:
        self.stages.append(StageRecord(name=name, status=status, log_file=log_file))

    def write_summary(self) -> str:
        text = format_summary(
            backend=self.config.backend,
            device_map=self.config.device_map,
            ranks=self.ranks,
            stages=self.stages,
            interrupted_sig=self.interrupted_short_sig,
            interrupted_code=self.interrupted_exit_code,
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        summary_file = self.log_dir / "summary.txt"
        summary_file.write_text(text, encoding="utf-8")
        self._write_stdout_bytes(text.encode("utf-8"))
        return text

    def _finalize_interrupted(self) -> int:
        assert self.interrupted_sig_num is not None
        _, short_sig = signal_names(self.interrupted_sig_num)
        self.interrupted_short_sig = short_sig
        self.interrupted_exit_code = 128 + self.interrupted_sig_num
        self.any_failed = 1
        self.write_summary()
        return self.interrupted_exit_code

    def _run_interruptible_capture(self, cmd: Sequence[str], *, timeout: float = 30.0) -> tuple[int, str]:
        """Run a short helper command while polling for session interrupts so signals never wait on it."""
        if self.interrupted_sig_num is not None:
            return 128 + self.interrupted_sig_num, ""
        try:
            proc = subprocess.Popen(
                list(cmd),
                cwd=str(self.repo_root),
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            return 1, ""
        deadline = time.monotonic() + timeout
        while True:
            rc = proc.poll()
            if rc is not None:
                out_b = proc.stdout.read() if proc.stdout is not None else b""
                if proc.stdout is not None:
                    proc.stdout.close()
                return rc, out_b.decode("utf-8", errors="replace").strip()
            if self.interrupted_sig_num is not None or time.monotonic() >= deadline:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
                if proc.stdout is not None:
                    proc.stdout.close()
                if self.interrupted_sig_num is not None:
                    return 128 + self.interrupted_sig_num, ""
                return 124, ""
            time.sleep(0.02)

    def compute_runtime_env_sig(self) -> str:
        """Query the synced uv environment's interpreter, PyTorch, and CUDA identity."""
        cmd = [
            self.uv_bin,
            "run",
            "--no-sync",
            *self.config.uv_py_args,
            "python",
            "-c",
            _RUNTIME_ENV_SIG_SNIPPET,
        ]
        rc, out = self._run_interruptible_capture(cmd)
        if rc == 0 and out:
            return out.splitlines()[-1].strip()
        return "uninitialized"

    def query_gpu_count(self) -> int:
        """Query visible CUDA GPU count via PyTorch inside the synced project environment."""
        cmd = [
            self.uv_bin,
            "run",
            "--no-sync",
            *self.config.uv_py_args,
            "python",
            "-c",
            _GPU_COUNT_SNIPPET,
        ]
        rc, out = self._run_interruptible_capture(cmd)
        if rc == 0 and out.isdigit():
            return int(out)
        return 0

    def is_stage_resumed(self, name: str, expected_sig: str, required_artifact: Path | None) -> tuple[bool, str]:
        """Return `(resumed, artifact_sha)` when `<name>.done` matches `expected_sig` and artifact integrity."""
        done_file = self.log_dir / f"{name}.done"
        if self.config.force or not done_file.is_file():
            return False, ""
        saved_sig, saved_sha = read_done_marker(done_file)
        if not saved_sig or saved_sig != expected_sig:
            return False, ""
        if required_artifact is not None:
            if not saved_sha:
                return False, ""
            cur_sha = verify_and_hash_ref_archive(required_artifact)
            if cur_sha is None or cur_sha != saved_sha:
                return False, ""
            return True, cur_sha
        return True, ""

    def run_stage_process(self, stage_name: str, log_file: Path, cmd: Sequence[str]) -> tuple[int, int | None]:
        """Run one stage process with timeout, live non-blocking pipe drain, and confirmed-live tree kill.

        Returns `(exit_code, interrupted_sig_num)`. On any exit path, a recorded session interrupt
        takes precedence over timeout (`124`) or normal process return code.
        """
        self_pid = os.getpid()
        self_pgid = os.getpgrp()
        token_val = f"tt_{self_pid}_{uuid.uuid4().hex}"
        token_needle = f"{_STAGE_TOKEN_ENV_KEY}={token_val}"
        child_env = dict(self.env)
        child_env[_STAGE_TOKEN_ENV_KEY] = token_val

        r_fd, w_fd = os.pipe()
        os.set_blocking(r_fd, False)
        sel = selectors.DefaultSelector()
        sel.register(r_fd, selectors.EVENT_READ)

        try:
            with open(log_file, "wb") as log_f:
                header = f"== stage: {stage_name} (tree_state={self.tree_method}:{self.tree_sig[:12]}) ==\n".encode()
                log_f.write(header)
                log_f.flush()
                self._write_stdout_bytes(header)

                if self.on_before_stage_spawn is not None:
                    self.on_before_stage_spawn(stage_name)

                if self.interrupted_sig_num is not None:
                    sig_num = self.interrupted_sig_num
                    sig_label, _ = signal_names(sig_num)
                    msg = (
                        f"ERROR: stage {stage_name} interrupted by {sig_label} "
                        "(during startup before launch) and killed (signalled 0 pid(s))\n"
                    ).encode()
                    log_f.write(msg)
                    log_f.flush()
                    self._write_stderr_bytes(msg)
                    return 128 + sig_num, sig_num

                proc = subprocess.Popen(
                    list(cmd),
                    cwd=str(self.repo_root),
                    stdout=w_fd,
                    stderr=w_fd,
                    start_new_session=True,
                    env=child_env,
                )
                os.close(w_fd)
                w_fd = -1

                deadline = time.monotonic() + self.config.stage_timeout
                timed_out = False
                eof = False

                def _drain_available() -> None:
                    nonlocal eof
                    while not eof:
                        try:
                            chunk = os.read(r_fd, 65536)
                        except BlockingIOError:
                            return
                        except InterruptedError:
                            continue
                        except OSError as exc:
                            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                                return
                            eof = True
                            return
                        if not chunk:
                            eof = True
                            return
                        log_f.write(chunk)
                        log_f.flush()
                        self._write_stdout_bytes(chunk)

                def _wait_and_drain(duration: float) -> None:
                    end = time.monotonic() + duration
                    while True:
                        _drain_available()
                        rem = end - time.monotonic()
                        if rem <= 0:
                            return
                        if eof:
                            time.sleep(min(0.05, rem))
                            return
                        try:
                            events = sel.select(timeout=min(0.05, rem))
                        except InterruptedError:
                            continue
                        if events:
                            _drain_available()

                def _kill_stage_tree(*, notify_timeout_teardown: bool = False) -> int:
                    ever_signalled: set[int] = set()
                    _drain_available()

                    live_pids, live_pgids = collect_live_targets(
                        proc, token_needle, self_pid=self_pid, self_pgid=self_pgid, ps_bin=self._ps_bin
                    )
                    if live_pids or live_pgids:
                        ever_signalled |= live_pids
                        signal_live(live_pids, live_pgids, signal.SIGTERM)

                    if notify_timeout_teardown and self.on_timeout_teardown is not None:
                        self.on_timeout_teardown(stage_name)

                    grace_end = time.monotonic() + self.teardown_grace_s
                    while time.monotonic() < grace_end:
                        _wait_and_drain(0.05)
                        live_pids, live_pgids = collect_live_targets(
                            proc, token_needle, self_pid=self_pid, self_pgid=self_pgid, ps_bin=self._ps_bin
                        )
                        if not live_pids and proc.poll() is not None:
                            _wait_and_drain(0.1)
                            break
                        newly_seen = live_pids - ever_signalled
                        if newly_seen:
                            ever_signalled |= newly_seen
                            signal_live(newly_seen, set(), signal.SIGTERM)

                    live_pids, live_pgids = collect_live_targets(
                        proc, token_needle, self_pid=self_pid, self_pgid=self_pgid, ps_bin=self._ps_bin
                    )
                    if live_pids or live_pgids:
                        ever_signalled |= live_pids
                        signal_live(live_pids, live_pgids, signal.SIGKILL)

                        reap_end = time.monotonic() + self.teardown_reap_s
                        while time.monotonic() < reap_end:
                            _wait_and_drain(0.05)
                            live_pids, live_pgids = collect_live_targets(
                                proc, token_needle, self_pid=self_pid, self_pgid=self_pgid, ps_bin=self._ps_bin
                            )
                            if not live_pids and proc.poll() is not None:
                                break
                            if live_pids or live_pgids:
                                ever_signalled |= live_pids
                                signal_live(live_pids, live_pgids, signal.SIGKILL)

                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
                    _wait_and_drain(0.1)
                    _drain_available()
                    return len(ever_signalled)

                def _log_interrupted(killed_n: int, context: str = "") -> tuple[int, int]:
                    assert self.interrupted_sig_num is not None
                    sig_num = self.interrupted_sig_num
                    sig_label, _ = signal_names(sig_num)
                    suffix = f" ({context})" if context else ""
                    msg = (
                        f"ERROR: stage {stage_name} interrupted by {sig_label}{suffix} "
                        f"and killed (signalled {killed_n} pid(s))\n"
                    ).encode()
                    log_f.write(msg)
                    log_f.flush()
                    self._write_stderr_bytes(msg)
                    return 128 + sig_num, sig_num

                while True:
                    if self.interrupted_sig_num is not None:
                        break
                    rc = proc.poll()
                    remaining = deadline - time.monotonic()
                    if rc is not None:
                        _drain_available()
                        if eof:
                            break
                        if remaining <= 0:
                            timed_out = True
                            break
                        try:
                            events = sel.select(timeout=min(0.05, remaining))
                        except InterruptedError:
                            continue
                        if events:
                            _drain_available()
                        continue

                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        events = sel.select(timeout=min(0.05, remaining))
                    except InterruptedError:
                        continue
                    if events:
                        _drain_available()

                if self.interrupted_sig_num is not None:
                    killed_n = _kill_stage_tree()
                    return _log_interrupted(killed_n)

                if timed_out:
                    killed_n = _kill_stage_tree(notify_timeout_teardown=True)
                    msg = (
                        f"ERROR: stage {stage_name} timed out after {self.config.stage_timeout:.0f}s "
                        f"and was killed (signalled {killed_n} pid(s))\n"
                    ).encode()
                    log_f.write(msg)
                    log_f.flush()
                    self._write_stderr_bytes(msg)
                    if self.interrupted_sig_num is not None:
                        return _log_interrupted(killed_n, context="during timeout teardown")
                    return 124, None

                rc = proc.wait()
                _drain_available()
                if self.interrupted_sig_num is not None:
                    killed_n = _kill_stage_tree()
                    return _log_interrupted(killed_n, context="at stage exit")
                exit_code = 128 + (-rc) if rc < 0 else rc
                return exit_code, None
        finally:
            sel.close()
            if w_fd >= 0:
                try:
                    os.close(w_fd)
                except OSError:
                    pass
            try:
                os.close(r_fd)
            except OSError:
                pass

    def run_stage_with_sig(
        self,
        name: str,
        sig: str,
        required_artifact: Path | None,
        cmd: Sequence[str],
    ) -> StageOutcome:
        """Execute or resume one stage with a pre-computed signature."""
        if self.interrupted_sig_num is not None:
            return StageOutcome(
                ok=False, returncode=128 + self.interrupted_sig_num, interrupted_sig=self.interrupted_sig_num
            )

        log_file = self.log_dir / f"{name}.log"
        done_file = self.log_dir / f"{name}.done"
        shown_log = log_file.name

        resumed, artifact_sha = self.is_stage_resumed(name, sig, required_artifact)
        if resumed:
            self._print_line(f"== [SKIP resumed] {name} ==")
            self.record_stage(name, "0 (resumed)", shown_log)
            return StageOutcome(ok=True, returncode=0, resumed=True, artifact_sha256=artifact_sha)

        done_file.unlink(missing_ok=True)
        if required_artifact is not None:
            required_artifact.unlink(missing_ok=True)
            Path(f"{required_artifact}.tmp").unlink(missing_ok=True)

        self._print_line(f"== [RUN] {name} ==")
        rc, int_sig = self.run_stage_process(name, log_file, cmd)
        if int_sig is not None:
            done_file.unlink(missing_ok=True)
            if required_artifact is not None:
                required_artifact.unlink(missing_ok=True)
                Path(f"{required_artifact}.tmp").unlink(missing_ok=True)
            _, short_sig = signal_names(int_sig)
            self.record_stage(name, f"{128 + int_sig} (interrupted:SIG{short_sig})", shown_log)
            self.any_failed = 1
            return StageOutcome(ok=False, returncode=128 + int_sig, interrupted_sig=int_sig)

        if rc == 0:
            if required_artifact is not None:
                produced_sha = verify_and_hash_ref_archive(required_artifact)
                if produced_sha is not None:
                    write_done_marker(done_file, sig, produced_sha)
                    self.record_stage(name, "0", shown_log)
                    return StageOutcome(ok=True, returncode=0, artifact_sha256=produced_sha)
                self._write_stderr_bytes(
                    f"error: stage {name} exited 0, but artifact {required_artifact} "
                    "failed integrity verification\n".encode()
                )
                required_artifact.unlink(missing_ok=True)
                self.any_failed = 1
                self.record_stage(name, "1 (corrupt artifact)", shown_log)
                return StageOutcome(ok=False, returncode=1)
            write_done_marker(done_file, sig)
            self.record_stage(name, "0", shown_log)
            return StageOutcome(ok=True, returncode=0)

        self.any_failed = 1
        self.record_stage(name, str(rc), shown_log)
        return StageOutcome(ok=False, returncode=rc)

    def run_stage(self, name: str, required_artifact: Path | None, cmd: Sequence[str]) -> StageOutcome:
        sig = stage_signature(self.tree_method, self.tree_sig, self.runtime_env_sig, [name, *cmd], env=self.env)
        return self.run_stage_with_sig(name, sig, required_artifact, cmd)

    def run_torchrun_stage(
        self,
        name: str,
        world_size: int,
        ref_sha: str,
        driver_args: Sequence[str],
    ) -> StageOutcome:
        """Run or resume one `torchrun` stage, binding its signature to `ref_sha` and omitting `--master-port`."""
        if self.interrupted_sig_num is not None:
            return StageOutcome(
                ok=False, returncode=128 + self.interrupted_sig_num, interrupted_sig=self.interrupted_sig_num
            )

        shown_log = f"{name}.log"
        sig_parts = [name, "torchrun", f"n={world_size}", f"ref_sha={ref_sha}", *self.config.uv_py_args, *driver_args]
        sig = stage_signature(self.tree_method, self.tree_sig, self.runtime_env_sig, sig_parts, env=self.env)
        resumed, _ = self.is_stage_resumed(name, sig, None)
        if resumed:
            self._print_line(f"== [SKIP resumed] {name} ==")
            self.record_stage(name, "0 (resumed)", shown_log)
            return StageOutcome(ok=True, returncode=0, resumed=True)

        try:
            port = pick_free_port()
        except OSError:
            self._write_stderr_bytes(f"error: failed to allocate a free TCP port for {name}\n".encode())
            self.any_failed = 1
            self.record_stage(name, "1 (no port)", shown_log)
            return StageOutcome(ok=False, returncode=1)

        cmd = [
            self.uv_bin,
            "run",
            "--no-sync",
            *self.config.uv_py_args,
            "torchrun",
            "--max-restarts=0",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            f"--nproc_per_node={world_size}",
            *driver_args,
        ]
        return self.run_stage_with_sig(name, sig, None, cmd)

    def _build_env_stage_cmd(self) -> list[str]:
        return [
            sys.executable,
            "-c",
            _ENV_STAGE_BOOTSTRAP,
            self.uv_bin,
            self.config.backend,
            self.tree_method,
            self.tree_sig[:12],
            _ENV_PROBE_SNIPPET,
            *self.config.uv_py_args,
        ]

    def _run_ref_and_torchrun_pair(
        self,
        *,
        ref_stage_name: str,
        dist_stage_name: str,
        ref_file: Path,
        world_size: int,
        ref_device: str,
        dist_backend: Backend,
        device_map_args: Sequence[str],
        extra_model_args: Sequence[str],
    ) -> bool:
        """Run one reference stage followed by its dependent `torchrun` stage. Returns False if interrupted."""
        ref_cmd = [
            self.uv_bin,
            "run",
            "--no-sync",
            *self.config.uv_py_args,
            "python",
            "scripts/probes/distributed_trainer_check.py",
            "--write-reference",
            str(ref_file),
            "--reference-mode",
            self.config.reference_mode,
            "--device",
            ref_device,
            "--world-size",
            str(world_size),
            *extra_model_args,
            *self.config.extra_driver_args,
        ]
        ref_out = self.run_stage(ref_stage_name, ref_file, ref_cmd)
        if self.interrupted_sig_num is not None:
            return False

        if ref_out.ok and ref_out.artifact_sha256 and ref_file.is_file():
            driver_args = [
                "scripts/probes/distributed_trainer_check.py",
                "--backend",
                dist_backend,
                *device_map_args,
                "--timeout",
                self.config.step_timeout,
                *extra_model_args,
                "--reference",
                str(ref_file),
                *self.config.extra_driver_args,
            ]
            self.run_torchrun_stage(dist_stage_name, world_size, ref_out.artifact_sha256, driver_args)
            if self.interrupted_sig_num is not None:
                return False
        else:
            self._print_line(f"== [SKIP] {dist_stage_name} (reference stage failed) ==")
            self.record_stage(dist_stage_name, "SKIPPED (ref failed)", "-")
            self.any_failed = 1
        return True

    def run(self) -> int:
        """Execute the full Stage 3 verification session and return the POSIX exit code."""
        if self.uv_bin == "uv" and shutil.which("uv", path=self.env.get("PATH")) is None:
            self._write_stderr_bytes(b"error: 'uv' not found on PATH; install uv before running this script\n")
            return 2

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.refs_dir.mkdir(parents=True, exist_ok=True)

        previous_handlers = self._install_signal_handlers()
        try:
            try:
                self.tree_method, self.tree_sig = compute_tree_state_sig(self.repo_root, self.log_dir, env=self.env)
            except Exception as exc:
                self._write_stderr_bytes(f"error: failed to compute tree-state signature: {exc}\n".encode())
                return 2

            if self.interrupted_sig_num is not None:
                return self._finalize_interrupted()

            self.runtime_env_sig = self.compute_runtime_env_sig()
            if self.interrupted_sig_num is not None:
                return self._finalize_interrupted()

            # Stage 1: environment sync and runtime facts.
            env_cmd = self._build_env_stage_cmd()
            env_sig_parts = [
                "01_env",
                self.config.backend,
                self.tree_method,
                self.tree_sig[:12],
                *self.config.uv_py_args,
            ]
            env_sig = stage_signature(
                self.tree_method, self.tree_sig, self.runtime_env_sig, env_sig_parts, env=self.env
            )
            env_out = self.run_stage_with_sig("01_env", env_sig, None, env_cmd)
            if self.interrupted_sig_num is not None:
                return self._finalize_interrupted()

            if not env_out.ok:
                self._write_stderr_bytes(
                    f"error: stage 01_env failed (exit {env_out.returncode}); "
                    f"aborting {self.config.backend} session (see 01_env.log)\n".encode()
                )
                self.any_failed = 1
                self.write_summary()
                return env_out.returncode

            # Refresh resolved runtime identity after uv sync; if 01_env just synced a new venv,
            # rebind 01_env.done to the post-sync runtime identity so identical re-runs resume cleanly.
            post_env_runtime_sig = self.compute_runtime_env_sig()
            if self.interrupted_sig_num is not None:
                return self._finalize_interrupted()
            if post_env_runtime_sig != self.runtime_env_sig:
                self.runtime_env_sig = post_env_runtime_sig
                if not env_out.resumed:
                    post_env_sig = stage_signature(
                        self.tree_method, self.tree_sig, self.runtime_env_sig, env_sig_parts, env=self.env
                    )
                    write_done_marker(self.log_dir / "01_env.done", post_env_sig)

            # Query visible GPU count via PyTorch in the synced project environment.
            gpu_count = 0
            if self.config.backend == "nccl":
                gpu_count = self.query_gpu_count()
                if self.interrupted_sig_num is not None:
                    return self._finalize_interrupted()
                if gpu_count < 1:
                    self._write_stderr_bytes(
                        b"error: --backend nccl requested, but PyTorch reported 0 visible CUDA GPUs (see 01_env.log)\n"
                    )
                    self.any_failed = 1
                    self.write_summary()
                    return 2

            # Resolve default ranks and validate per-rank GPU count.
            if not self.config.ranks_specified:
                if self.config.backend == "gloo" or self.config.shared_gpu:
                    self.ranks = [2, 4]
                else:
                    self.ranks = []
                    if gpu_count >= 2:
                        self.ranks.append(2)
                    if gpu_count >= 4:
                        self.ranks.append(4)
                    if not self.ranks:
                        self._write_stderr_bytes(
                            f"error: only {gpu_count} visible GPU detected; "
                            "pass --shared-gpu to run N=2 and N=4 on one GPU\n".encode()
                        )
                        self.any_failed = 1
                        self.write_summary()
                        return 2

            has_rank_4 = 4 in self.ranks
            for n in self.ranks:
                if self.config.backend == "nccl" and not self.config.shared_gpu and n > gpu_count:
                    self._write_stderr_bytes(
                        f"error: requested --ranks {n} with per-rank NCCL, "
                        f"but only {gpu_count} visible GPU(s) detected "
                        "(pass --shared-gpu to share 1 GPU)\n".encode()
                    )
                    self.any_failed = 1
                    self.write_summary()
                    return 2

            if self.interrupted_sig_num is not None:
                return self._finalize_interrupted()

            # Stage 2: 4-GPU prototype parity probe.
            if self.config.backend == "nccl" and not self.config.shared_gpu and gpu_count >= 4 and has_rank_4:
                proto_size_args = ["--sizes", *self.config.sizes] if self.config.sizes else []
                self.run_stage(
                    "02_prototype_parity_cuda",
                    None,
                    [
                        self.uv_bin,
                        "run",
                        "--no-sync",
                        *self.config.uv_py_args,
                        "python",
                        "scripts/probes/distributed_parity.py",
                        "--device",
                        "cuda",
                        *proto_size_args,
                    ],
                )
                if self.interrupted_sig_num is not None:
                    return self._finalize_interrupted()
            else:
                if self.config.backend != "nccl":
                    proto_skip_reason = "gloo backend"
                elif self.config.shared_gpu:
                    proto_skip_reason = "shared GPU"
                elif gpu_count < 4:
                    proto_skip_reason = "needs 4 GPUs"
                else:
                    proto_skip_reason = "rank 4 not requested"
                self._print_line(f"== [SKIP] 02_prototype_parity_cuda ({proto_skip_reason}) ==")
                self.record_stage("02_prototype_parity_cuda", f"SKIPPED ({proto_skip_reason})", "-")

            # Stages 3-4: tiny_random reference + torchrun on the selected backend.
            ref_device = "cuda:0" if self.config.backend == "nccl" else "cpu"
            device_map_args: tuple[str, ...] = (
                ("--device-map", self.config.device_map) if self.config.backend == "nccl" else ()
            )
            for n in self.ranks:
                if not self._run_ref_and_torchrun_pair(
                    ref_stage_name=f"03_ref_{self.config.backend}_tiny_n{n}",
                    dist_stage_name=f"04_{self.config.backend}_tiny_n{n}",
                    ref_file=self.refs_dir / f"ref_{self.config.backend}_tiny_n{n}.zip",
                    world_size=n,
                    ref_device=ref_device,
                    dist_backend=self.config.backend,
                    device_map_args=device_map_args,
                    extra_model_args=(),
                ):
                    return self._finalize_interrupted()

            # Stages 5-6: released checkpoints when requested.
            if self.config.sizes:
                released_args = ("--no-tiny", "--sizes", *self.config.sizes)
                for n in self.ranks:
                    if not self._run_ref_and_torchrun_pair(
                        ref_stage_name=f"05_ref_{self.config.backend}_released_n{n}",
                        dist_stage_name=f"06_{self.config.backend}_released_n{n}",
                        ref_file=self.refs_dir / f"ref_{self.config.backend}_released_n{n}.zip",
                        world_size=n,
                        ref_device=ref_device,
                        dist_backend=self.config.backend,
                        device_map_args=device_map_args,
                        extra_model_args=released_args,
                    ):
                        return self._finalize_interrupted()

            # Stages 7-9: CPU gloo comparison (including gloo rank-failure check) when primary backend is nccl.
            if self.config.backend == "nccl":
                for n in self.ranks:
                    if not self._run_ref_and_torchrun_pair(
                        ref_stage_name=f"07_ref_gloo_tiny_n{n}",
                        dist_stage_name=f"08_gloo_tiny_n{n}",
                        ref_file=self.refs_dir / f"ref_gloo_tiny_n{n}.zip",
                        world_size=n,
                        ref_device="cpu",
                        dist_backend="gloo",
                        device_map_args=(),
                        extra_model_args=(),
                    ):
                        return self._finalize_interrupted()
                for n in self.ranks:
                    self.run_stage(
                        f"09_rank_failure_gloo_n{n}",
                        None,
                        [
                            self.uv_bin,
                            "run",
                            "--no-sync",
                            *self.config.uv_py_args,
                            "python",
                            "scripts/probes/distributed_rank_failure_check.py",
                            "--backend",
                            "gloo",
                            "--world-size",
                            str(n),
                            "--pg-timeout",
                            self.config.pg_timeout,
                        ],
                    )
                    if self.interrupted_sig_num is not None:
                        return self._finalize_interrupted()

            # Rank-failure timeout check for the primary backend.
            rf_prefix = "10" if self.config.backend == "nccl" else "09"
            for n in self.ranks:
                self.run_stage(
                    f"{rf_prefix}_rank_failure_{self.config.backend}_n{n}",
                    None,
                    [
                        self.uv_bin,
                        "run",
                        "--no-sync",
                        *self.config.uv_py_args,
                        "python",
                        "scripts/probes/distributed_rank_failure_check.py",
                        "--backend",
                        self.config.backend,
                        *device_map_args,
                        "--world-size",
                        str(n),
                        "--pg-timeout",
                        self.config.pg_timeout,
                    ],
                )
                if self.interrupted_sig_num is not None:
                    return self._finalize_interrupted()

            self.write_summary()
            return self.any_failed
        finally:
            self._restore_signal_handlers(previous_handlers)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `scripts/multi_gpu_check.py`."""
    try:
        config = parse_args(sys.argv[1:] if argv is None else argv)
    except HelpRequested:
        print(USAGE)
        return 0
    except CliError as exc:
        print(str(exc), file=sys.stderr)
        if exc.show_usage:
            print(USAGE, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130

    os.chdir(REPO_ROOT)
    runner = SessionRunner(config=config, repo_root=REPO_ROOT)
    try:
        return runner.run()
    except KeyboardInterrupt:
        runner.interrupted_sig_num = int(signal.SIGINT)
        return runner._finalize_interrupted()


if __name__ == "__main__":
    sys.exit(main())
