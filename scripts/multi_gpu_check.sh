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
# on GNU `timeout` or `nvidia-smi`; uses an embedded Python process-tree + PGID timeout runner that
# streams directly to stdout and the stage log without a pipe that orphaned workers could hold open.
# Records a content-addressed stage signature (commit + exact stage args) in `<stage>.done` so
# re-running resumes only identical completed stages (pass --force to redo completed stages).
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
        echo "error: ${opt} must be a positive number, got '${val}'" >&2
        exit 2
    fi
    if ! "${HELPER_PY[@]}" -c 'import sys; sys.exit(0 if float(sys.argv[1]) > 0.0 else 1)' "$val"; then
        echo "error: ${opt} must be > 0, got '${val}'" >&2
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
            if [[ -n "$SIZES_ARG" ]]; then
                while [[ $# -gt 0 && "$1" != -* ]]; do
                    SIZES_ARG="${SIZES_ARG} $1"
                    shift
                done
            fi
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
        --only)
            require_arg "$1" "$#"
            EXTRA_DRIVER_ARGS+=("--only" "$2")
            shift 2
            ;;
        --skip)
            require_arg "$1" "$#"
            EXTRA_DRIVER_ARGS+=("--skip" "$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$BACKEND" != "nccl" && "$BACKEND" != "gloo" ]]; then
    echo "error: --backend must be nccl or gloo, got '$BACKEND'" >&2
    exit 2
fi
if [[ "$SHARED_GPU" -eq 1 && "$BACKEND" != "nccl" ]]; then
    echo "error: --shared-gpu applies only to --backend nccl" >&2
    exit 2
fi
if [[ "$REFERENCE_MODE" != "auto" && "$REFERENCE_MODE" != "sketch" && "$REFERENCE_MODE" != "exact" ]]; then
    echo "error: --reference-mode must be auto, sketch, or exact, got '$REFERENCE_MODE'" >&2
    exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' not found on PATH" >&2
    exit 2
fi

declare -a UV_PY_ARGS=()
if [[ -n "$PYTHON_VERSION" ]]; then
    UV_PY_ARGS=(--python "$PYTHON_VERSION")
fi

# Choose a standalone Python interpreter for timeout, signature, and free-port helpers so they
# never trigger an implicit `uv sync` on the project venv before stage `01_env` runs `uv sync --locked`.
declare -a HELPER_PY=()
if command -v python3 >/dev/null 2>&1; then
    HELPER_PY=(python3)
elif command -v python >/dev/null 2>&1; then
    HELPER_PY=(python)
else
    HELPER_PY=(uv run --no-project ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python)
fi

validate_positive_number "--pg-timeout" "$PG_TIMEOUT"
validate_positive_number "--step-timeout" "$STEP_TIMEOUT"
validate_positive_number "--stage-timeout" "$STAGE_TIMEOUT"

# Validate explicit --ranks up front (before syncing the venv) with globbing disabled.
declare -a RANKS=()
if [[ "$RANKS_SPECIFIED" -eq 1 ]]; then
    split_tokens "$RANKS_ARG"
    if [[ "${#SPLIT_RESULT[@]}" -eq 0 ]]; then
        echo "error: --ranks requires at least one integer >= 2, got '$RANKS_ARG'" >&2
        exit 2
    fi
    RANKS=("${SPLIT_RESULT[@]}")
    for n in "${RANKS[@]}"; do
        if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -lt 2 ]]; then
            echo "error: rank count must be an integer >= 2, got '$n'" >&2
            exit 2
        fi
    done
fi

# Resolve and validate released checkpoint sizes (globbing disabled).
if [[ "$SIZES_SPECIFIED" -eq 0 ]]; then
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

REPO_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

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

# Content-addressed signature for a stage: commit HEAD + stage name + exact canonical arguments + key env.
stage_signature() {
    "${HELPER_PY[@]}" -c '
import hashlib, os, sys
h = hashlib.sha256()
for part in sys.argv[1:]:
    h.update(part.encode("utf-8"))
    h.update(b"\0")
for env_k in ("NCCL_MULTI_RANK_GPU_ENABLE", "CUDA_VISIBLE_DEVICES"):
    val = os.environ.get(env_k, "<unset>")
    h.update(f"{env_k}={val}".encode("utf-8"))
    h.update(b"\0")
print(h.hexdigest())
' "$REPO_HEAD" "$@"
}

# Run one stage under a hard wall-clock timeout, streaming directly to stdout and `$log_file`
# without a shell `| tee` pipe that orphaned children could hold open. On timeout, kills the
# entire descendant tree AND every session/process-group spawned under this stage (including
# `torchrun`'s `start_new_session=True` rank workers and DataLoader workers) via both PPID
# recursion and an inherited stage environment token.
run_stage_with_timeout() {
    local limit_s="$1"
    local stage_name="$2"
    local log_file="$3"
    shift 3
    "${HELPER_PY[@]}" -c '
import errno, os, selectors, signal, subprocess, sys, time, uuid

limit = float(sys.argv[1])
stage_name = sys.argv[2]
log_path = sys.argv[3]
cmd = sys.argv[4:]

self_pid = os.getpid()
self_pgid = os.getpgrp()
token_key = "_TIMESTRAL_STAGE_TOKEN"
token_val = f"tt_{self_pid}_{uuid.uuid4().hex}"
token_needle = f"{token_key}={token_val}"


def _read_proc_table():
    """Return `(children_by_ppid, pgid_by_pid)` across Linux `/proc` and POSIX `ps`."""
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
                ppid = int(fields[1])
                pgid = int(fields[2])
                children.setdefault(ppid, []).append(pid)
                pgids[pid] = pgid
            except OSError:
                continue
        if pgids:
            return children, pgids
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                parts = line.split()
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    pid, ppid, pgid = int(parts[0]), int(parts[1]), int(parts[2])
                    children.setdefault(ppid, []).append(pid)
                    pgids[pid] = pgid
    except Exception:
        pass
    return children, pgids


def _find_token_pids():
    """Find any process carrying our inherited stage token (catches reparented/new-session workers)."""
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
    try:
        out = subprocess.run(
            ["ps", "eww", "-axo", "pid=,command="],
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


def _collect_targets(root_pid):
    children, pgids = _read_proc_table()
    pids = set()
    stack = [root_pid]
    while stack:
        cur = stack.pop()
        if cur in pids or cur == self_pid:
            continue
        pids.add(cur)
        for ch in children.get(cur, ()):
            stack.append(ch)
    for tpid in _find_token_pids():
        if tpid not in pids and tpid != self_pid:
            stack = [tpid]
            while stack:
                cur = stack.pop()
                if cur in pids or cur == self_pid:
                    continue
                pids.add(cur)
                for ch in children.get(cur, ()):
                    stack.append(ch)
    target_pgids = set()
    for pid in pids:
        pg = pgids.get(pid)
        if pg is not None and pg != self_pgid and pg > 0:
            target_pgids.add(pg)
    if root_pid != self_pgid:
        target_pgids.add(root_pid)
    return pids, target_pgids


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_all(pids, pgids, sig):
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


def _kill_stage_tree(proc):
    seen_pids, seen_pgids = _collect_targets(proc.pid)
    _signal_all(seen_pids, seen_pgids, signal.SIGTERM)

    grace_end = time.monotonic() + 3.0
    while time.monotonic() < grace_end:
        proc.poll()
        new_pids, new_pgids = _collect_targets(proc.pid)
        seen_pids |= new_pids
        seen_pgids |= new_pgids
        if not any(_pid_alive(p) for p in seen_pids) and proc.poll() is not None:
            break
        time.sleep(0.05)

    # Always run SIGKILL against every tracked PGID and PID, even if proc itself exited early.
    new_pids, new_pgids = _collect_targets(proc.pid)
    seen_pids |= new_pids
    seen_pgids |= new_pgids
    _signal_all(seen_pids, seen_pgids, signal.SIGKILL)

    reap_end = time.monotonic() + 2.0
    while time.monotonic() < reap_end:
        proc.poll()
        if not any(_pid_alive(p) for p in seen_pids) and proc.poll() is not None:
            break
        time.sleep(0.05)
        new_pids, new_pgids = _collect_targets(proc.pid)
        seen_pids |= new_pids
        seen_pgids |= new_pgids
        _signal_all(seen_pids, seen_pgids, signal.SIGKILL)
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass
    return len(seen_pids)


r_fd, w_fd = os.pipe()
os.set_blocking(r_fd, False)
child_env = dict(os.environ)
child_env[token_key] = token_val

with open(log_path, "wb") as log_f:
    header = f"== stage: {stage_name} ==\n".encode("utf-8")
    log_f.write(header)
    log_f.flush()
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.flush()

    proc = subprocess.Popen(cmd, stdout=w_fd, stderr=w_fd, start_new_session=True, env=child_env)
    os.close(w_fd)

    sel = selectors.DefaultSelector()
    sel.register(r_fd, selectors.EVENT_READ)
    deadline = time.monotonic() + limit
    timed_out = False
    eof = [False]

    def _drain_available():
        while True:
            try:
                chunk = os.read(r_fd, 65536)
            except BlockingIOError:
                return
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
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

    try:
        while True:
            rc = proc.poll()
            remaining = deadline - time.monotonic()
            if rc is not None:
                _drain_available()
                if eof[0]:
                    break
                events = sel.select(timeout=0.05)
                if events:
                    _drain_available()
                elif proc.poll() is not None:
                    # Direct child finished; check if any descendant still holds w_fd.
                    if remaining <= 0:
                        timed_out = True
                        break
                    continue
                continue

            if remaining <= 0:
                timed_out = True
                break
            events = sel.select(timeout=min(0.1, remaining))
            if events:
                _drain_available()

        if timed_out:
            killed_n = _kill_stage_tree(proc)
            _drain_available()
            msg = f"ERROR: stage {stage_name} timed out after {limit:.0f}s and was killed (signalled {killed_n} pid(s))\n".encode("utf-8")
            log_f.write(msg)
            log_f.flush()
            sys.stderr.buffer.write(msg)
            sys.stderr.buffer.flush()
            sys.exit(124)

        rc = proc.wait()
        sys.exit(rc)
    finally:
        sel.close()
        try:
            os.close(r_fd)
        except OSError:
            pass
' "$limit_s" "$stage_name" "$log_file" "$@"
}

declare -a STAGE_NAMES=()
declare -a STAGE_RESULTS=()
declare -a STAGE_LOGS=()
ANY_FAILED=0
LAST_STAGE_OK=0
LAST_STAGE_SIG=""

record_stage() {
    STAGE_NAMES+=("$1")
    STAGE_RESULTS+=("$2")
    STAGE_LOGS+=("$3")
}

is_stage_resumed() {
    local name="$1"
    local expected_sig="$2"
    local required_artifact="$3"
    local done_file="${LOG_DIR}/${name}.done"
    if [[ "$FORCE" -eq 0 && -f "$done_file" ]]; then
        local saved_sig
        saved_sig="$(cat "$done_file" 2>/dev/null || true)"
        if [[ "$saved_sig" == "$expected_sig" ]]; then
            if [[ -z "$required_artifact" || -f "$required_artifact" ]]; then
                return 0
            fi
        fi
    fi
    return 1
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

    if is_stage_resumed "$name" "$sig" "$required_artifact"; then
        echo "== [SKIP resumed] ${name} =="
        record_stage "$name" "0 (resumed)" "$shown_log"
        LAST_STAGE_OK=1
        return 0
    fi
    rm -f "$done_file"
    if [[ -n "$required_artifact" ]]; then
        rm -f "$required_artifact"
    fi

    echo "== [RUN] ${name} =="
    run_stage_with_timeout "$STAGE_TIMEOUT" "$name" "$log_file" "$@"
    local rc="$?"

    if [[ "$rc" -eq 0 && ( -z "$required_artifact" || -f "$required_artifact" ) ]]; then
        printf "%s\n" "$sig" >"$done_file"
        record_stage "$name" "0" "$shown_log"
        LAST_STAGE_OK=1
    else
        ANY_FAILED=1
        LAST_STAGE_OK=0
        if [[ "$rc" -eq 0 ]]; then
            rc=1
        fi
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
    local dep_sig="$3"
    shift 3
    local log_file="${LOG_DIR}/${name}.log"
    local shown_log
    shown_log="$(display_log_path "$log_file")"
    # Signature omits ephemeral --master-port so identical re-runs resume cleanly, while binding
    # to the paired reference stage's signature (`dep_sig`) so any reference flag change invalidates it.
    local sig
    sig="$(stage_signature "$name" "torchrun" "n=${n}" "dep=${dep_sig}" ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} "$@")"
    LAST_STAGE_SIG="$sig"
    if is_stage_resumed "$name" "$sig" ""; then
        echo "== [SKIP resumed] ${name} =="
        record_stage "$name" "0 (resumed)" "$shown_log"
        LAST_STAGE_OK=1
        return 0
    fi
    local port
    if ! port="$(pick_free_port)" || ! [[ "$port" =~ ^[0-9]+$ ]]; then
        echo "error: failed to allocate a free TCP port for ${name}" >&2
        ANY_FAILED=1
        LAST_STAGE_OK=0
        record_stage "$name" "1 (no port)" "$shown_log"
        return 0
    fi
    run_stage_with_sig "$name" "$sig" "" \
        uv run --no-sync ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} torchrun \
            --max-restarts=0 --master-addr=127.0.0.1 --master-port="$port" --nproc_per_node="$n" \
            "$@"
}

# Stage 1: environment (sync locked venv quietly and record hardware/runtime without hostnames, zones, or usernames).
run_stage "01_env" "" bash -c '
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

# Query visible GPU count via PyTorch in the project environment (respects CUDA_VISIBLE_DEVICES,
# works inside containers without `nvidia-smi`).
GPU_COUNT=0
ENV_OK="$LAST_STAGE_OK"
if [[ "$BACKEND" == "nccl" ]]; then
    if [[ "$ENV_OK" -eq 1 ]]; then
        GPU_COUNT_RAW="$(uv run --no-sync ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null || echo 0)"
        if [[ "$GPU_COUNT_RAW" =~ ^[0-9]+$ ]]; then
            GPU_COUNT="$GPU_COUNT_RAW"
        fi
    fi
    if [[ "$GPU_COUNT" -lt 1 ]]; then
        echo "error: --backend nccl requested, but PyTorch reported 0 visible CUDA GPUs (see 01_env.log)" >&2
        ANY_FAILED=1
        SUMMARY_FILE="${LOG_DIR}/summary.txt"
        {
            echo ""
            echo "== multi_gpu_check summary (backend=${BACKEND}, device_map=${DEVICE_MAP}, ranks=none; logs relative to --log-dir) =="
            printf "%-36s %-24s %s\n" "stage" "exit_code" "log_file"
            printf "%-36s %-24s %s\n" "------------------------------------" "------------------------" "----------------------------------------"
            for i in "${!STAGE_NAMES[@]}"; do
                printf "%-36s %-24s %s\n" "${STAGE_NAMES[$i]}" "${STAGE_RESULTS[$i]}" "${STAGE_LOGS[$i]}"
            done
        } | tee "$SUMMARY_FILE"
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
# run only when 4 ranks were requested/selected, forwarding any configured released --sizes).
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
    echo "== [SKIP] 02_prototype_parity_cuda (requires 4 distinct GPUs and rank 4 selected) =="
    record_stage "02_prototype_parity_cuda" "SKIPPED (needs 4 GPUs)" "-"
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

    if [[ "$LAST_STAGE_OK" -eq 1 && -f "$ref_file" ]]; then
        run_torchrun_stage "04_${BACKEND}_tiny_n${n}" "$n" "$LAST_STAGE_SIG" \
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

        if [[ "$LAST_STAGE_OK" -eq 1 && -f "$ref_file" ]]; then
            run_torchrun_stage "06_${BACKEND}_released_n${n}" "$n" "$LAST_STAGE_SIG" \
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

        if [[ "$LAST_STAGE_OK" -eq 1 && -f "$ref_file" ]]; then
            run_torchrun_stage "08_gloo_tiny_n${n}" "$n" "$LAST_STAGE_SIG" \
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

# Final summary table: stage -> exit_code -> log_file (relative to --log-dir; no hostnames, zones, usernames, or paths).
SUMMARY_FILE="${LOG_DIR}/summary.txt"
SUMMARY_HDR="backend=${BACKEND}"
if [[ "$BACKEND" == "nccl" ]]; then
    SUMMARY_HDR="${SUMMARY_HDR}, device_map=${DEVICE_MAP}"
fi
SUMMARY_HDR="${SUMMARY_HDR}, ranks=${RANKS[*]}"
{
    echo ""
    echo "== multi_gpu_check summary (${SUMMARY_HDR}; logs relative to --log-dir) =="
    printf "%-36s %-24s %s\n" "stage" "exit_code" "log_file"
    printf "%-36s %-24s %s\n" "------------------------------------" "------------------------" "----------------------------------------"
    for i in "${!STAGE_NAMES[@]}"; do
        printf "%-36s %-24s %s\n" "${STAGE_NAMES[$i]}" "${STAGE_RESULTS[$i]}" "${STAGE_LOGS[$i]}"
    done
} | tee "$SUMMARY_FILE"

exit "$ANY_FAILED"
