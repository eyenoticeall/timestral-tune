#!/usr/bin/env bash
# Portable, resumable Stage 3 multi-GPU / multi-process verification session driver.
#
# Usage:
#   bash scripts/multi_gpu_check.sh [options]
#
# Examples:
#   # Four-GPU machine (auto-detects visible GPUs via PyTorch, runs N=2 and N=4 on NCCL + CPU gloo):
#   bash scripts/multi_gpu_check.sh --python 3.12
#
#   # Single-GPU shared NCCL bench (exports NCCL_MULTI_RANK_GPU_ENABLE=1, runs N=2 and N=4 on cuda:0):
#   bash scripts/multi_gpu_check.sh --shared-gpu --python 3.12
#
#   # CPU-only gloo dry run (exercises stage runner, resume logic, trainer check, and rank-failure check):
#   bash scripts/multi_gpu_check.sh --backend gloo --ranks 2 4 --log-dir /tmp/multi_gpu_dry
#
# Portable: uses no sudo, no cloud-specific commands, and no user/home assumptions; works as root
# inside Docker containers as well as on Linux/macOS developer machines (bash 3.2+). Does not depend
# on GNU `timeout` or `nvidia-smi`; uses an embedded Python process-tree runner that streams directly
# to stdout and the stage log without a shell `| tee` pipe that orphaned workers could hold open,
# continuously drains output during SIGTERM grace and SIGKILL teardown, handles SIGINT/SIGTERM/SIGHUP
# by killing the active stage tree and writing an interrupted summary, and re-collects confirmed-live
# stage PIDs immediately before every signal so reaped/reused PIDs are never signalled.
#
# Limitation (A1): a process that double-forks, calls `setsid()`, and clears its environment (or runs
# on a system with neither `/proc` nor `ps`) can escape tree termination; neither `torchrun` nor
# PyTorch `DataLoader` workers do this.
#
# Resume markers (`<stage>.done`) are content-addressed over git tree state (HEAD + tracked diffs +
# untracked code under scripts/ and src/), resolved Python/torch/CUDA runtime identity, all
# `NCCL_*`/`TORCH_*`/`CUDA_*`/`CUBLAS_*`/`OMP_*` environment variables, exact stage arguments, and
# the verified SHA-256 content digest of any reference archive produced or consumed by the stage.
set -uo pipefail
cd "$(dirname "$0")/.."

BACKEND="nccl"
SHARED_GPU=0
RANKS_ARG=""
RANKS_SPECIFIED=0
SIZES_ARG="__DEFAULT__"
SIZES_SPECIFIED=0
PYTHON_VERSION=""
LOG_DIR="multi_gpu_check_logs"
FORCE=0
PG_TIMEOUT="20"
STEP_TIMEOUT="180"
STAGE_TIMEOUT="900"
REFERENCE_MODE="auto"
EXTRA_DRIVER_ARGS=()

usage() {
    cat <<'EOF'
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
  -h, --help                 Show this help message.
EOF
}

require_arg() {
    if [[ "$2" -lt 2 ]]; then
        echo "error: option '$1' requires an argument" >&2
        exit 2
    fi
}

validate_positive_number() {
    local opt="$1"
    local val="$2"
    if ! [[ "$val" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "error: ${opt} must be a finite positive number, got '${val}'" >&2
        exit 2
    fi
    if ! "${HELPER_PY[@]}" -c 'import math, sys; v = float(sys.argv[1]); sys.exit(0 if math.isfinite(v) and v > 0.0 else 1)' "$val"; then
        echo "error: ${opt} must be a finite positive number, got '${val}'" >&2
        exit 2
    fi
}

# Tokenise space- or comma-separated option values with globbing disabled (bash 3.2 and 5 safe).
declare -a SPLIT_RESULT=()
split_tokens() {
    local raw="${1//,/ }"
    local old_opts="$-"
    SPLIT_RESULT=()
    set -f
    # shellcheck disable=SC2206
    SPLIT_RESULT=($raw)
    if [[ "$old_opts" != *f* ]]; then
        set +f
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            require_arg "$1" "$#"
            BACKEND="$2"
            shift 2
            ;;
        --shared-gpu)
            SHARED_GPU=1
            shift
            ;;
        --ranks)
            require_arg "$1" "$#"
            RANKS_SPECIFIED=1
            RANKS_ARG="$2"
            shift 2
            while [[ $# -gt 0 && "$1" != -* ]]; do
                RANKS_ARG="${RANKS_ARG} $1"
                shift
            done
            ;;
        --sizes)
            require_arg "$1" "$#"
            SIZES_SPECIFIED=1
            SIZES_ARG="$2"
            shift 2
            while [[ $# -gt 0 && "$1" != -* ]]; do
                SIZES_ARG="${SIZES_ARG} $1"
                shift
            done
            ;;
        --python)
            require_arg "$1" "$#"
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --log-dir)
            require_arg "$1" "$#"
            LOG_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --reference-mode)
            require_arg "$1" "$#"
            REFERENCE_MODE="$2"
            shift 2
            ;;
        --pg-timeout)
            require_arg "$1" "$#"
            PG_TIMEOUT="$2"
            shift 2
            ;;
        --step-timeout)
            require_arg "$1" "$#"
            STEP_TIMEOUT="$2"
            shift 2
            ;;
        --stage-timeout)
            require_arg "$1" "$#"
            STAGE_TIMEOUT="$2"
            shift 2
            ;;
        --only|--skip)
            require_arg "$1" "$#"
            EXTRA_DRIVER_ARGS+=("$1" "$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$BACKEND" != "nccl" && "$BACKEND" != "gloo" ]]; then
    echo "error: --backend must be nccl or gloo, got '$BACKEND'" >&2
    exit 2
fi
if [[ "$BACKEND" == "gloo" && "$SHARED_GPU" -eq 1 ]]; then
    echo "error: --shared-gpu applies only to --backend nccl" >&2
    exit 2
fi
if [[ "$REFERENCE_MODE" != "auto" && "$REFERENCE_MODE" != "sketch" && "$REFERENCE_MODE" != "exact" ]]; then
    echo "error: --reference-mode must be auto, sketch, or exact, got '$REFERENCE_MODE'" >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' not found on PATH; install uv before running this script" >&2
    exit 2
fi

declare -a UV_PY_ARGS=()
if [[ -n "$PYTHON_VERSION" ]]; then
    UV_PY_ARGS=(--python "$PYTHON_VERSION")
fi

# Standalone Python for timeout/port/signature helpers (avoids running `uv sync` on every helper call).
declare -a HELPER_PY=()
if command -v python3 >/dev/null 2>&1; then
    HELPER_PY=(python3)
else
    HELPER_PY=(uv run --no-project ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python)
fi

validate_positive_number "--pg-timeout" "$PG_TIMEOUT"
validate_positive_number "--step-timeout" "$STEP_TIMEOUT"
validate_positive_number "--stage-timeout" "$STAGE_TIMEOUT"

# Validate explicit --ranks now (default RANKS resolved after 01_env reports visible GPUs via PyTorch).
declare -a RANKS=()
if [[ "$RANKS_SPECIFIED" -eq 1 ]]; then
    split_tokens "$RANKS_ARG"
    if [[ "${#SPLIT_RESULT[@]}" -eq 0 ]]; then
        echo "error: --ranks must contain at least one world size (integer >= 2), got '${RANKS_ARG}'" >&2
        exit 2
    fi
    RANKS=("${SPLIT_RESULT[@]}")
    for n in "${RANKS[@]}"; do
        if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -lt 2 ]]; then
            echo "error: each entry in --ranks must be an integer >= 2, got '$n'" >&2
            exit 2
        fi
    done
fi

# Resolve and validate SIZES (glob-safe).
if [[ "$SIZES_SPECIFIED" -eq 0 && "$SIZES_ARG" == "__DEFAULT__" ]]; then
    if [[ "$BACKEND" == "nccl" ]]; then
        SIZES_ARG="5M 28M"
    else
        SIZES_ARG=""
    fi
fi
declare -a SIZES=()
if [[ -n "$SIZES_ARG" ]]; then
    split_tokens "$SIZES_ARG"
    if [[ "${#SPLIT_RESULT[@]}" -eq 0 ]]; then
        echo "error: --sizes contained no valid size tokens (pass \"\" to skip released sizes)" >&2
        exit 2
    fi
    SIZES=("${SPLIT_RESULT[@]}")
    for sz in "${SIZES[@]}"; do
        case "$sz" in
            5M|28M|123M|209M) ;;
            *)
                echo "error: invalid checkpoint size '$sz' (expected one of: 5M, 28M, 123M, 209M)" >&2
                exit 2
                ;;
        esac
    done
fi

# Ensure short TMPDIR so PyTorch's torch_shm_manager Unix socket stays below the 103-byte macOS limit.
if [[ -z "${TMPDIR:-}" || "${#TMPDIR}" -gt 60 ]]; then
    export TMPDIR="/tmp"
fi
export PYTHONFAULTHANDLER=1

if [[ "$SHARED_GPU" -eq 1 ]]; then
    export NCCL_MULTI_RANK_GPU_ENABLE=1
fi

mkdir -p "$LOG_DIR"
REFS_DIR="${LOG_DIR}/refs"
mkdir -p "$REFS_DIR"

DEVICE_MAP="per-rank"
if [[ "$SHARED_GPU" -eq 1 ]]; then
    DEVICE_MAP="shared"
fi

# Format stage log path relative to LOG_DIR so no absolute path, $HOME, or username prefix leaks into summary.txt.
display_log_path() {
    local raw="$1"
    printf "%s" "$(basename "$raw")"
}

# Pick a free TCP port on 127.0.0.1 so torchrun never relies on hostname/IPv6 reverse-DNS.
pick_free_port() {
    "${HELPER_PY[@]}" -c '
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
'
}

# Digest of actual checkout state: HEAD commit + tracked git status/diff + untracked code files under
# scripts/ and src/ (excluding LOG_DIR so writing logs inside the checkout never invalidates resume).
compute_tree_state_sig() {
    "${HELPER_PY[@]}" -c '
import hashlib, os, subprocess, sys
from pathlib import Path

log_dir_resolved = Path(sys.argv[1]).resolve()
h = hashlib.sha256()

def _run(args):
    p = subprocess.run(args, check=False, capture_output=True)
    return p.returncode, p.stdout

rc, head = _run(["git", "rev-parse", "HEAD"])
h.update(b"head=")
h.update(head.strip() if rc == 0 else b"unknown")
h.update(b"\0")

rc, status = _run(["git", "status", "--porcelain=v1", "-z", "--untracked-files=no"])
h.update(b"status=")
h.update(status if rc == 0 else b"unknown")
h.update(b"\0")

rc, diff = _run(["git", "diff", "HEAD"])
h.update(b"diff=")
h.update(diff if rc == 0 else b"unknown")
h.update(b"\0")

rc, untracked = _run(["git", "ls-files", "--others", "--exclude-standard", "-z", "scripts", "src"])
if rc == 0 and untracked:
    for rel_b in sorted(x for x in untracked.split(b"\0") if x):
        p = Path(rel_b.decode("utf-8", errors="surrogateescape"))
        try:
            resolved = p.resolve()
        except OSError:
            continue
        if resolved == log_dir_resolved or log_dir_resolved in resolved.parents:
            continue
        h.update(b"untracked=")
        h.update(rel_b)
        h.update(b"\0")
        if p.is_file():
            try:
                h.update(p.read_bytes())
            except OSError:
                pass
        h.update(b"\0")

print(h.hexdigest())
' "$LOG_DIR"
}

# Resolved interpreter + PyTorch/CUDA runtime identity in the synced uv environment.
compute_runtime_env_sig() {
    uv run --no-sync ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python -c '
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
    h.update(b"\0")
print(h.hexdigest())
' 2>/dev/null || echo "uninitialized"
}

TREE_STATE_SIG="$(compute_tree_state_sig)"
RUNTIME_ENV_SIG="$(compute_runtime_env_sig)"

# Content-addressed signature for a stage: tree state + resolved runtime identity + stage name +
# exact canonical arguments + every environment variable matching NCCL_*, TORCH_*, CUDA_*, CUBLAS_*, OMP_*.
stage_signature() {
    "${HELPER_PY[@]}" -c '
import hashlib, os, sys
h = hashlib.sha256()
for part in sys.argv[1:]:
    h.update(part.encode("utf-8"))
    h.update(b"\0")
prefixes = ("NCCL_", "TORCH_", "CUDA_", "CUBLAS_", "OMP_")
for env_k in sorted(os.environ):
    if env_k.startswith(prefixes):
        h.update(f"{env_k}={os.environ[env_k]}".encode("utf-8"))
        h.update(b"\0")
print(h.hexdigest())
' "$TREE_STATE_SIG" "$RUNTIME_ENV_SIG" "$@"
}

# Verify a reference zip archive is non-empty, passes full zip CRC integrity (`testzip()`), has valid
# `spec.json` and `entries.json`, and contains every expected tensor member with exact byte length;
# prints the archive file's SHA-256 hex digest on stdout and exits non-zero on any corruption.
verify_and_hash_ref_archive() {
    local ref_path="$1"
    "${HELPER_PY[@]}" -c '
import hashlib, json, math, os, sys, zipfile

path = sys.argv[1]
if not os.path.isfile(path) or os.path.getsize(path) <= 0:
    sys.exit(1)

dtype_nbytes = {"float64": 8, "float32": 4, "bfloat16": 2, "float16": 2}
try:
    with zipfile.ZipFile(path, "r") as zf:
        if zf.testzip() is not None:
            sys.exit(1)
        member_sizes = {info.filename: info.file_size for info in zf.infolist()}
        if "spec.json" not in member_sizes or "entries.json" not in member_sizes:
            sys.exit(1)
        meta = json.loads(zf.read("spec.json").decode("utf-8"))
        entries = json.loads(zf.read("entries.json").decode("utf-8"))
        if not isinstance(meta, dict) or not isinstance(meta.get("spec"), dict) or not isinstance(entries, dict):
            sys.exit(1)
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                sys.exit(1)
            member = entry.get("tensor_member")
            dtype_name = entry.get("tensor_dtype")
            shape = entry.get("tensor_shape")
            if not isinstance(member, str) or member not in member_sizes:
                sys.exit(1)
            if dtype_name not in dtype_nbytes or not isinstance(shape, list):
                sys.exit(1)
            expected_bytes = math.prod(int(d) for d in shape) * dtype_nbytes[dtype_name]
            if member_sizes[member] != expected_bytes:
                sys.exit(1)
except Exception:
    sys.exit(1)

h = hashlib.sha256()
with open(path, "rb") as f:
    while True:
        chunk = f.read(1024 * 1024)
        if not chunk:
            break
        h.update(chunk)
print(h.hexdigest())
' "$ref_path"
}

declare -a STAGE_NAMES=()
declare -a STAGE_RESULTS=()
declare -a STAGE_LOGS=()
ANY_FAILED=0
LAST_STAGE_OK=0
LAST_STAGE_RC=0
LAST_STAGE_RESUMED=0
LAST_STAGE_SIG=""
LAST_ARTIFACT_SHA=""
INTERRUPTED_SIG=""
INTERRUPTED_CODE=0
ACTIVE_STAGE_NAME=""
ACTIVE_STAGE_LOG=""
ACTIVE_REQUIRED_ARTIFACT=""
STAGE_RUNNER_PID=""
INT_MARKER="${LOG_DIR}/.stage_interrupted"

record_stage() {
    STAGE_NAMES+=("$1")
    STAGE_RESULTS+=("$2")
    STAGE_LOGS+=("$3")
}

write_summary() {
    local summary_file="${LOG_DIR}/summary.txt"
    local hdr="backend=${BACKEND}"
    if [[ "$BACKEND" == "nccl" ]]; then
        hdr="${hdr}, device_map=${DEVICE_MAP}"
    fi
    if [[ "${#RANKS[@]}" -gt 0 ]]; then
        hdr="${hdr}, ranks=${RANKS[*]}"
    else
        hdr="${hdr}, ranks=none"
    fi
    if [[ -n "$INTERRUPTED_SIG" ]]; then
        hdr="${hdr}, INTERRUPTED by SIG${INTERRUPTED_SIG} (exit ${INTERRUPTED_CODE})"
    fi
    {
        echo ""
        echo "== multi_gpu_check summary (${hdr}; logs relative to --log-dir) =="
        printf "%-36s %-28s %s\n" "stage" "exit_code" "log_file"
        printf "%-36s %-28s %s\n" "------------------------------------" "----------------------------" "----------------------------------------"
        for i in "${!STAGE_NAMES[@]}"; do
            printf "%-36s %-28s %s\n" "${STAGE_NAMES[$i]}" "${STAGE_RESULTS[$i]}" "${STAGE_LOGS[$i]}"
        done
        if [[ -n "$INTERRUPTED_SIG" ]]; then
            echo ""
            echo "Session interrupted by SIG${INTERRUPTED_SIG} (exit ${INTERRUPTED_CODE}); remaining stages were not started."
        fi
    } >"$summary_file"
    cat "$summary_file" 2>/dev/null || true
}

on_interrupt() {
    local sig_name="$1"
    local exit_code="$2"
    if [[ -n "$INTERRUPTED_SIG" ]]; then
        return
    fi
    INTERRUPTED_SIG="$sig_name"
    INTERRUPTED_CODE="$exit_code"
    ANY_FAILED=1

    if [[ -n "$STAGE_RUNNER_PID" ]]; then
        # Forward the signal to the Python stage runner (idempotent if delivered via PGID) and wait
        # for it to drain output and terminate the entire stage tree.
        kill "-${sig_name}" "$STAGE_RUNNER_PID" 2>/dev/null || true
        local waited=0
        while kill -0 "$STAGE_RUNNER_PID" 2>/dev/null; do
            wait "$STAGE_RUNNER_PID" 2>/dev/null || true
            if ! kill -0 "$STAGE_RUNNER_PID" 2>/dev/null; then
                break
            fi
            waited=$((waited + 1))
            if [[ "$waited" -eq 2 ]]; then
                # Re-signal once in case the runner was mid-startup before installing its handler.
                kill "-${sig_name}" "$STAGE_RUNNER_PID" 2>/dev/null || true
            elif [[ "$waited" -ge 120 ]]; then
                kill -KILL "$STAGE_RUNNER_PID" 2>/dev/null || true
                wait "$STAGE_RUNNER_PID" 2>/dev/null || true
                break
            fi
            sleep 0.1
        done
        STAGE_RUNNER_PID=""
    fi

    if [[ -n "$ACTIVE_STAGE_NAME" ]]; then
        rm -f "${LOG_DIR}/${ACTIVE_STAGE_NAME}.done" "$INT_MARKER"
        if [[ -n "$ACTIVE_REQUIRED_ARTIFACT" ]]; then
            rm -f "$ACTIVE_REQUIRED_ARTIFACT" "${ACTIVE_REQUIRED_ARTIFACT}.tmp"
        fi
        record_stage "$ACTIVE_STAGE_NAME" "${exit_code} (interrupted:SIG${sig_name})" "$ACTIVE_STAGE_LOG"
        ACTIVE_STAGE_NAME=""
        ACTIVE_STAGE_LOG=""
        ACTIVE_REQUIRED_ARTIFACT=""
    else
        rm -f "$INT_MARKER"
    fi

    write_summary
    exit "$exit_code"
}

trap 'on_interrupt INT 130' INT
trap 'on_interrupt TERM 143' TERM
trap 'on_interrupt HUP 129' HUP

# Run one stage under a hard wall-clock timeout and interrupt handling, streaming directly to stdout
# and `$log_file` without a shell `| tee` pipe. Drains child output continuously during both SIGTERM
# grace and SIGKILL teardown, re-collects confirmed-live stage PIDs immediately before every signal
# (never signalling reaped/reused PIDs), and returns POSIX 128+sig on signal death.
run_stage_with_timeout() {
    local limit_s="$1"
    local stage_name="$2"
    local log_file="$3"
    shift 3
    rm -f "$INT_MARKER"

    "${HELPER_PY[@]}" -c '
import errno, os, selectors, shutil, signal, subprocess, sys, time, uuid

limit = float(sys.argv[1])
stage_name = sys.argv[2]
log_path = sys.argv[3]
int_marker_path = sys.argv[4]
cmd = sys.argv[5:]

self_pid = os.getpid()
self_pgid = os.getpgrp()
token_key = "_TIMESTRAL_STAGE_TOKEN"
token_val = f"tt_{self_pid}_{uuid.uuid4().hex}"
token_needle = f"{token_key}={token_val}"
ps_bin = shutil.which("ps") or ("/bin/ps" if os.path.isfile("/bin/ps") else None)

interrupted_sig = [None]


def _on_signal(signum, _frame):
    # Record first signal only; ignore duplicates so teardown cannot be aborted mid-cleanup.
    if interrupted_sig[0] is None:
        interrupted_sig[0] = signum


for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(_sig, _on_signal)


def _read_proc_table():
    """Return `(children_by_ppid, pgid_by_pid)` of live (non-zombie) processes via `/proc` or `ps`."""
    children = {}
    pgids = {}
    if os.path.isdir("/proc"):
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                stat = open(f"/proc/{pid}/stat", "rb").read()
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


def _find_token_pids():
    """Find live processes carrying our inherited stage token (catches reparented/new-session workers)."""
    found = set()
    if os.path.isdir("/proc"):
        needle_b = token_needle.encode("utf-8")
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == self_pid:
                continue
            try:
                env_bytes = open(f"/proc/{pid}/environ", "rb").read()
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


def _collect_live_targets(proc):
    """Collect ONLY currently confirmed-live stage PIDs and safe PGIDs (never reaped/reused PIDs)."""
    children, pgids = _read_proc_table()
    token_pids = _find_token_pids()

    seeds = [p for p in token_pids if p != self_pid and (not pgids or p in pgids)]
    # Only seed proc.pid while our un-reaped Popen handle confirms it has not exited yet (so the
    # kernel cannot have reused proc.pid for an unrelated process).
    if proc.poll() is None and proc.pid != self_pid:
        seeds.append(proc.pid)

    live_pids = set()
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

    # Group all observed live PIDs by PGID so we only call killpg() on PGIDs whose EVERY live member
    # belongs to this stage (never touching our own PGID or any unrelated process sharing a PGID).
    members_by_pgid = {}
    for pid, pg in pgids.items():
        members_by_pgid.setdefault(pg, set()).add(pid)

    live_pgids = set()
    for pid in live_pids:
        pg = pgids.get(pid)
        if pg is not None and pg > 0 and pg != self_pgid:
            members = members_by_pgid.get(pg, set())
            if members and members.issubset(live_pids):
                live_pgids.add(pg)
    if proc.poll() is None and proc.pid != self_pgid and not pgids:
        live_pgids.add(proc.pid)

    return live_pids, live_pgids


def _signal_live(pids, pgids, sig):
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


r_fd, w_fd = os.pipe()
os.set_blocking(r_fd, False)
child_env = dict(os.environ)
child_env[token_key] = token_val
stdout_open = [True]


def _write_stdout(data):
    if not stdout_open[0]:
        return
    try:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    except OSError:
        stdout_open[0] = False


def _write_stderr(data):
    try:
        sys.stderr.buffer.write(data)
        sys.stderr.buffer.flush()
    except OSError:
        pass


with open(log_path, "wb") as log_f:
    header = f"== stage: {stage_name} ==\n".encode("utf-8")
    log_f.write(header)
    log_f.flush()
    _write_stdout(header)

    proc = subprocess.Popen(cmd, stdout=w_fd, stderr=w_fd, start_new_session=True, env=child_env)
    os.close(w_fd)

    sel = selectors.DefaultSelector()
    sel.register(r_fd, selectors.EVENT_READ)
    deadline = time.monotonic() + limit
    timed_out = False
    eof = [False]

    def _drain_available():
        while not eof[0]:
            try:
                chunk = os.read(r_fd, 65536)
            except BlockingIOError:
                return
            except InterruptedError:
                continue
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    return
                eof[0] = True
                return
            if not chunk:
                eof[0] = True
                return
            log_f.write(chunk)
            log_f.flush()
            _write_stdout(chunk)

    def _wait_and_drain(duration):
        end = time.monotonic() + duration
        while True:
            _drain_available()
            rem = end - time.monotonic()
            if rem <= 0:
                return
            if eof[0]:
                time.sleep(min(0.05, rem))
                return
            try:
                events = sel.select(timeout=min(0.05, rem))
            except InterruptedError:
                continue
            if events:
                _drain_available()

    def _kill_stage_tree():
        ever_signalled = set()
        _drain_available()

        live_pids, live_pgids = _collect_live_targets(proc)
        if live_pids or live_pgids:
            ever_signalled |= live_pids
            _signal_live(live_pids, live_pgids, signal.SIGTERM)

        # Drain continuously through the 3s SIGTERM grace window so dumping children never block on the pipe.
        grace_end = time.monotonic() + 3.0
        while time.monotonic() < grace_end:
            _wait_and_drain(0.05)
            live_pids, live_pgids = _collect_live_targets(proc)
            if not live_pids and proc.poll() is not None:
                # Re-drain briefly to collect trailing buffered bytes until EOF.
                _wait_and_drain(0.1)
                break
            newly_seen = live_pids - ever_signalled
            if newly_seen:
                ever_signalled |= newly_seen
                _signal_live(newly_seen, set(), signal.SIGTERM)

        # Re-collect fresh immediately before SIGKILL so already-exited/reused PIDs are NEVER signalled.
        live_pids, live_pgids = _collect_live_targets(proc)
        if live_pids or live_pgids:
            ever_signalled |= live_pids
            _signal_live(live_pids, live_pgids, signal.SIGKILL)

            reap_end = time.monotonic() + 2.0
            while time.monotonic() < reap_end:
                _wait_and_drain(0.05)
                live_pids, live_pgids = _collect_live_targets(proc)
                if not live_pids and proc.poll() is not None:
                    break
                if live_pids or live_pgids:
                    ever_signalled |= live_pids
                    _signal_live(live_pids, live_pgids, signal.SIGKILL)

        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        _wait_and_drain(0.1)
        _drain_available()
        return len(ever_signalled)

    try:
        while True:
            if interrupted_sig[0] is not None:
                break
            rc = proc.poll()
            remaining = deadline - time.monotonic()
            if rc is not None:
                _drain_available()
                if eof[0]:
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
                events = sel.select(timeout=min(0.1, remaining))
            except InterruptedError:
                continue
            if events:
                _drain_available()

        if interrupted_sig[0] is not None:
            sig_num = interrupted_sig[0]
            try:
                sig_label = signal.Signals(sig_num).name
            except ValueError:
                sig_label = f"SIG{sig_num}"
            short_sig = sig_label[3:] if sig_label.startswith("SIG") else sig_label
            killed_n = _kill_stage_tree()
            msg = f"ERROR: stage {stage_name} interrupted by {sig_label} and killed (signalled {killed_n} pid(s))\n".encode("utf-8")
            log_f.write(msg)
            log_f.flush()
            _write_stderr(msg)
            try:
                with open(int_marker_path, "w", encoding="utf-8") as mf:
                    mf.write(f"{short_sig} {128 + sig_num}\n")
            except OSError:
                pass
            sys.exit(128 + sig_num)

        if timed_out:
            killed_n = _kill_stage_tree()
            msg = f"ERROR: stage {stage_name} timed out after {limit:.0f}s and was killed (signalled {killed_n} pid(s))\n".encode("utf-8")
            log_f.write(msg)
            log_f.flush()
            _write_stderr(msg)
            sys.exit(124)

        rc = proc.wait()
        _drain_available()
        if rc < 0:
            sys.exit(128 + (-rc))
        sys.exit(rc)
    finally:
        sel.close()
        try:
            os.close(r_fd)
        except OSError:
            pass
' "$limit_s" "$stage_name" "$log_file" "$INT_MARKER" "$@" &
    STAGE_RUNNER_PID=$!
    local rc=0
    wait "$STAGE_RUNNER_PID" || rc=$?
    STAGE_RUNNER_PID=""
    return "$rc"
}

write_done_marker() {
    local done_file="$1"
    local sig="$2"
    local artifact_sha="$3"
    {
        printf "sig=%s\n" "$sig"
        if [[ -n "$artifact_sha" ]]; then
            printf "artifact_sha256=%s\n" "$artifact_sha"
        fi
    } >"$done_file"
}

is_stage_resumed() {
    local name="$1"
    local expected_sig="$2"
    local required_artifact="$3"
    local done_file="${LOG_DIR}/${name}.done"
    LAST_ARTIFACT_SHA=""
    if [[ "$FORCE" -ne 0 || ! -f "$done_file" ]]; then
        return 1
    fi
    local saved_sig=""
    local saved_sha=""
    local line=""
    while IFS= read -r line || [[ -n "$line" ]]; do
        case "$line" in
            sig=*) saved_sig="${line#sig=}" ;;
            artifact_sha256=*) saved_sha="${line#artifact_sha256=}" ;;
        esac
    done <"$done_file"
    if [[ -z "$saved_sig" || "$saved_sig" != "$expected_sig" ]]; then
        return 1
    fi
    if [[ -n "$required_artifact" ]]; then
        if [[ -z "$saved_sha" ]]; then
            return 1
        fi
        local cur_sha=""
        if ! cur_sha="$(verify_and_hash_ref_archive "$required_artifact")"; then
            return 1
        fi
        if [[ "$cur_sha" != "$saved_sha" ]]; then
            return 1
        fi
        LAST_ARTIFACT_SHA="$cur_sha"
    fi
    return 0
}

run_stage_with_sig() {
    local name="$1"
    local sig="$2"
    local required_artifact="$3"
    shift 3
    local log_file="${LOG_DIR}/${name}.log"
    local done_file="${LOG_DIR}/${name}.done"
    local shown_log
    shown_log="$(display_log_path "$log_file")"
    LAST_STAGE_SIG="$sig"
    LAST_STAGE_RESUMED=0

    if is_stage_resumed "$name" "$sig" "$required_artifact"; then
        echo "== [SKIP resumed] ${name} =="
        record_stage "$name" "0 (resumed)" "$shown_log"
        LAST_STAGE_OK=1
        LAST_STAGE_RC=0
        LAST_STAGE_RESUMED=1
        return 0
    fi
    rm -f "$done_file"
    if [[ -n "$required_artifact" ]]; then
        rm -f "$required_artifact" "${required_artifact}.tmp"
    fi

    echo "== [RUN] ${name} =="
    ACTIVE_STAGE_NAME="$name"
    ACTIVE_STAGE_LOG="$shown_log"
    ACTIVE_REQUIRED_ARTIFACT="$required_artifact"
    run_stage_with_timeout "$STAGE_TIMEOUT" "$name" "$log_file" "$@"
    local rc="$?"
    if [[ -f "$INT_MARKER" ]]; then
        local int_sig=""
        local int_code=""
        read -r int_sig int_code <"$INT_MARKER" || true
        rm -f "$INT_MARKER"
        on_interrupt "${int_sig:-TERM}" "${int_code:-$rc}"
    fi
    ACTIVE_STAGE_NAME=""
    ACTIVE_STAGE_LOG=""
    ACTIVE_REQUIRED_ARTIFACT=""

    LAST_STAGE_RC="$rc"
    LAST_ARTIFACT_SHA=""
    if [[ "$rc" -eq 0 ]]; then
        if [[ -n "$required_artifact" ]]; then
            local produced_sha=""
            if produced_sha="$(verify_and_hash_ref_archive "$required_artifact")"; then
                LAST_ARTIFACT_SHA="$produced_sha"
                write_done_marker "$done_file" "$sig" "$produced_sha"
                record_stage "$name" "0" "$shown_log"
                LAST_STAGE_OK=1
            else
                echo "error: stage ${name} exited 0, but artifact ${required_artifact} failed integrity verification" >&2
                rm -f "$required_artifact"
                ANY_FAILED=1
                LAST_STAGE_OK=0
                LAST_STAGE_RC=1
                record_stage "$name" "1 (corrupt artifact)" "$shown_log"
            fi
        else
            write_done_marker "$done_file" "$sig" ""
            record_stage "$name" "0" "$shown_log"
            LAST_STAGE_OK=1
        fi
    else
        ANY_FAILED=1
        LAST_STAGE_OK=0
        record_stage "$name" "${rc}" "$shown_log"
    fi
    return 0
}

run_stage() {
    local name="$1"
    local required_artifact="$2"
    shift 2
    local sig
    sig="$(stage_signature "$name" "$@")"
    run_stage_with_sig "$name" "$sig" "$required_artifact" "$@"
}

run_torchrun_stage() {
    local name="$1"
    local n="$2"
    local ref_sha="$3"
    shift 3
    local log_file="${LOG_DIR}/${name}.log"
    local shown_log
    shown_log="$(display_log_path "$log_file")"
    # Signature omits ephemeral --master-port so identical re-runs resume cleanly, while binding
    # directly to `ref_sha` (the verified SHA-256 of the reference archive file on disk).
    local sig
    sig="$(stage_signature "$name" "torchrun" "n=${n}" "ref_sha=${ref_sha}" ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} "$@")"
    LAST_STAGE_SIG="$sig"
    if is_stage_resumed "$name" "$sig" ""; then
        echo "== [SKIP resumed] ${name} =="
        record_stage "$name" "0 (resumed)" "$shown_log"
        LAST_STAGE_OK=1
        LAST_STAGE_RC=0
        LAST_STAGE_RESUMED=1
        return 0
    fi
    local port
    if ! port="$(pick_free_port)" || ! [[ "$port" =~ ^[0-9]+$ ]]; then
        echo "error: failed to allocate a free TCP port for ${name}" >&2
        ANY_FAILED=1
        LAST_STAGE_OK=0
        LAST_STAGE_RC=1
        record_stage "$name" "1 (no port)" "$shown_log"
        return 0
    fi
    run_stage_with_sig "$name" "$sig" "" \
        uv run --no-sync ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} torchrun \
            --max-restarts=0 --master-addr=127.0.0.1 --master-port="$port" --nproc_per_node="$n" \
            "$@"
}

# Stage 1: environment (sync locked venv quietly and record hardware/runtime without hostnames, zones, or usernames).
ENV_STAGE_CMD=(
    bash -c '
set -euo pipefail
backend="$1"
shift
uv sync --locked --quiet "$@"
uv run --no-sync "$@" python -c '\''
import os, platform, shutil, subprocess, sys, sysconfig, torch

backend = sys.argv[1]
print(f"os={platform.system()} arch={platform.machine()} python={platform.python_version()}")
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
'\'' "$backend"
' _ "$BACKEND" ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"}
)
run_stage "01_env" "" "${ENV_STAGE_CMD[@]}"
ENV_OK="$LAST_STAGE_OK"
ENV_RC="$LAST_STAGE_RC"

# Refresh resolved runtime identity after uv sync; if 01_env just ran and synced a new venv, bind
# 01_env.done to the post-sync runtime identity so the next identical invocation resumes cleanly.
if [[ "$ENV_OK" -eq 1 ]]; then
    POST_ENV_RUNTIME_SIG="$(compute_runtime_env_sig)"
    if [[ "$POST_ENV_RUNTIME_SIG" != "$RUNTIME_ENV_SIG" ]]; then
        RUNTIME_ENV_SIG="$POST_ENV_RUNTIME_SIG"
        if [[ "$LAST_STAGE_RESUMED" -eq 0 ]]; then
            POST_ENV_SIG="$(stage_signature "01_env" "${ENV_STAGE_CMD[@]}")"
            write_done_marker "${LOG_DIR}/01_env.done" "$POST_ENV_SIG" ""
        fi
    fi
fi

# Query visible GPU count via PyTorch in the synced project environment (respects CUDA_VISIBLE_DEVICES,
# works inside containers without `nvidia-smi`).
GPU_COUNT=0
if [[ "$BACKEND" == "nccl" ]]; then
    if [[ "$ENV_OK" -ne 1 ]]; then
        echo "error: stage 01_env failed (exit ${ENV_RC}); aborting NCCL session (see 01_env.log)" >&2
        ANY_FAILED=1
        write_summary
        exit "$ENV_RC"
    fi
    GPU_COUNT_RAW="$(uv run --no-sync ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null || echo 0)"
    if [[ "$GPU_COUNT_RAW" =~ ^[0-9]+$ ]]; then
        GPU_COUNT="$GPU_COUNT_RAW"
    fi
    if [[ "$GPU_COUNT" -lt 1 ]]; then
        echo "error: --backend nccl requested, but PyTorch reported 0 visible CUDA GPUs (see 01_env.log)" >&2
        ANY_FAILED=1
        write_summary
        exit 2
    fi
fi

# Resolve default RANKS if not explicitly supplied, and check per-rank GPU count.
if [[ "$RANKS_SPECIFIED" -eq 0 ]]; then
    if [[ "$BACKEND" == "gloo" || "$SHARED_GPU" -eq 1 ]]; then
        RANKS=(2 4)
    else
        if [[ "$GPU_COUNT" -ge 2 ]]; then
            RANKS+=(2)
        fi
        if [[ "$GPU_COUNT" -ge 4 ]]; then
            RANKS+=(4)
        fi
        if [[ "${#RANKS[@]}" -eq 0 ]]; then
            echo "error: only $GPU_COUNT visible GPU detected; pass --shared-gpu to run N=2 and N=4 on one GPU" >&2
            exit 2
        fi
    fi
fi

HAS_RANK_4=0
for n in "${RANKS[@]}"; do
    if [[ "$n" -eq 4 ]]; then
        HAS_RANK_4=1
    fi
    if [[ "$BACKEND" == "nccl" && "$SHARED_GPU" -eq 0 && "$n" -gt "$GPU_COUNT" ]]; then
        echo "error: requested --ranks $n with per-rank NCCL, but only $GPU_COUNT visible GPU(s) detected (pass --shared-gpu to share 1 GPU)" >&2
        exit 2
    fi
done

# Stage 2: prototype probe on 4 GPUs (distributed_parity.py binds cuda:rank and requires >= 4 GPUs;
# run only when 4 distinct GPUs are available and rank 4 was requested/selected, forwarding --sizes).
if [[ "$BACKEND" == "nccl" && "$SHARED_GPU" -eq 0 && "$GPU_COUNT" -ge 4 && "$HAS_RANK_4" -eq 1 ]]; then
    declare -a PROTO_SIZE_ARGS=()
    if [[ "${#SIZES[@]}" -gt 0 ]]; then
        PROTO_SIZE_ARGS=(--sizes "${SIZES[@]}")
    fi
    run_stage "02_prototype_parity_cuda" "" \
        uv run --no-sync ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python scripts/probes/distributed_parity.py \
            --device cuda \
            ${PROTO_SIZE_ARGS[@]+"${PROTO_SIZE_ARGS[@]}"}
else
    PROTO_SKIP_REASON="needs 4 GPUs"
    if [[ "$BACKEND" != "nccl" ]]; then
        PROTO_SKIP_REASON="gloo backend"
    elif [[ "$SHARED_GPU" -eq 1 ]]; then
        PROTO_SKIP_REASON="shared GPU"
    elif [[ "$GPU_COUNT" -lt 4 ]]; then
        PROTO_SKIP_REASON="needs 4 GPUs"
    elif [[ "$HAS_RANK_4" -eq 0 ]]; then
        PROTO_SKIP_REASON="rank 4 not requested"
    fi
    echo "== [SKIP] 02_prototype_parity_cuda (${PROTO_SKIP_REASON}) =="
    record_stage "02_prototype_parity_cuda" "SKIPPED (${PROTO_SKIP_REASON})" "-"
fi

# Stages 3-4: tiny_random reference + torchrun on the selected backend for each N.
REF_DEVICE="cpu"
declare -a DEVICE_MAP_ARGS=()
if [[ "$BACKEND" == "nccl" ]]; then
    REF_DEVICE="cuda:0"
    DEVICE_MAP_ARGS=(--device-map "$DEVICE_MAP")
fi

for n in "${RANKS[@]}"; do
    ref_file="${REFS_DIR}/ref_${BACKEND}_tiny_n${n}.zip"
    run_stage "03_ref_${BACKEND}_tiny_n${n}" "$ref_file" \
        uv run --no-sync ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python scripts/probes/distributed_trainer_check.py \
            --write-reference "$ref_file" \
            --reference-mode "$REFERENCE_MODE" \
            --device "$REF_DEVICE" \
            --world-size "$n" \
            ${EXTRA_DRIVER_ARGS[@]+"${EXTRA_DRIVER_ARGS[@]}"}

    if [[ "$LAST_STAGE_OK" -eq 1 && -n "$LAST_ARTIFACT_SHA" && -f "$ref_file" ]]; then
        run_torchrun_stage "04_${BACKEND}_tiny_n${n}" "$n" "$LAST_ARTIFACT_SHA" \
            scripts/probes/distributed_trainer_check.py \
            --backend "$BACKEND" \
            ${DEVICE_MAP_ARGS[@]+"${DEVICE_MAP_ARGS[@]}"} \
            --timeout "$STEP_TIMEOUT" \
            --reference "$ref_file" \
            ${EXTRA_DRIVER_ARGS[@]+"${EXTRA_DRIVER_ARGS[@]}"}
    else
        echo "== [SKIP] 04_${BACKEND}_tiny_n${n} (reference stage failed) =="
        record_stage "04_${BACKEND}_tiny_n${n}" "SKIPPED (ref failed)" "-"
        ANY_FAILED=1
    fi
done

# Stages 5-6: released checkpoints (e.g. 5M 28M) when requested.
if [[ "${#SIZES[@]}" -gt 0 ]]; then
    for n in "${RANKS[@]}"; do
        ref_file="${REFS_DIR}/ref_${BACKEND}_released_n${n}.zip"
        run_stage "05_ref_${BACKEND}_released_n${n}" "$ref_file" \
            uv run --no-sync ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python scripts/probes/distributed_trainer_check.py \
                --write-reference "$ref_file" \
                --reference-mode "$REFERENCE_MODE" \
                --device "$REF_DEVICE" \
                --world-size "$n" \
                --no-tiny \
                --sizes "${SIZES[@]}" \
                ${EXTRA_DRIVER_ARGS[@]+"${EXTRA_DRIVER_ARGS[@]}"}

        if [[ "$LAST_STAGE_OK" -eq 1 && -n "$LAST_ARTIFACT_SHA" && -f "$ref_file" ]]; then
            run_torchrun_stage "06_${BACKEND}_released_n${n}" "$n" "$LAST_ARTIFACT_SHA" \
                scripts/probes/distributed_trainer_check.py \
                --backend "$BACKEND" \
                ${DEVICE_MAP_ARGS[@]+"${DEVICE_MAP_ARGS[@]}"} \
                --timeout "$STEP_TIMEOUT" \
                --no-tiny \
                --sizes "${SIZES[@]}" \
                --reference "$ref_file" \
                ${EXTRA_DRIVER_ARGS[@]+"${EXTRA_DRIVER_ARGS[@]}"}
        else
            echo "== [SKIP] 06_${BACKEND}_released_n${n} (reference stage failed) =="
            record_stage "06_${BACKEND}_released_n${n}" "SKIPPED (ref failed)" "-"
            ANY_FAILED=1
        fi
    done
fi

# Stages 7-9: CPU gloo comparison (including gloo rank-failure check) when the main session backend is nccl.
if [[ "$BACKEND" == "nccl" ]]; then
    for n in "${RANKS[@]}"; do
        ref_file="${REFS_DIR}/ref_gloo_tiny_n${n}.zip"
        run_stage "07_ref_gloo_tiny_n${n}" "$ref_file" \
            uv run --no-sync ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python scripts/probes/distributed_trainer_check.py \
                --write-reference "$ref_file" \
                --reference-mode "$REFERENCE_MODE" \
                --device cpu \
                --world-size "$n" \
                ${EXTRA_DRIVER_ARGS[@]+"${EXTRA_DRIVER_ARGS[@]}"}

        if [[ "$LAST_STAGE_OK" -eq 1 && -n "$LAST_ARTIFACT_SHA" && -f "$ref_file" ]]; then
            run_torchrun_stage "08_gloo_tiny_n${n}" "$n" "$LAST_ARTIFACT_SHA" \
                scripts/probes/distributed_trainer_check.py \
                --backend gloo \
                --timeout "$STEP_TIMEOUT" \
                --reference "$ref_file" \
                ${EXTRA_DRIVER_ARGS[@]+"${EXTRA_DRIVER_ARGS[@]}"}
        else
            echo "== [SKIP] 08_gloo_tiny_n${n} (reference stage failed) =="
            record_stage "08_gloo_tiny_n${n}" "SKIPPED (ref failed)" "-"
            ANY_FAILED=1
        fi
    done
    for n in "${RANKS[@]}"; do
        run_stage "09_rank_failure_gloo_n${n}" "" \
            uv run --no-sync ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python scripts/probes/distributed_rank_failure_check.py \
                --backend gloo \
                --world-size "$n" \
                --pg-timeout "$PG_TIMEOUT"
    done
fi

# Rank-failure timeout check for the primary backend.
RF_PREFIX="09"
if [[ "$BACKEND" == "nccl" ]]; then
    RF_PREFIX="10"
fi
for n in "${RANKS[@]}"; do
    run_stage "${RF_PREFIX}_rank_failure_${BACKEND}_n${n}" "" \
        uv run --no-sync ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python scripts/probes/distributed_rank_failure_check.py \
            --backend "$BACKEND" \
            ${DEVICE_MAP_ARGS[@]+"${DEVICE_MAP_ARGS[@]}"} \
            --world-size "$n" \
            --pg-timeout "$PG_TIMEOUT"
done

write_summary
exit "$ANY_FAILED"
