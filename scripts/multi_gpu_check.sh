#!/usr/bin/env bash
# Portable, resumable Stage 3 multi-GPU / multi-process verification session driver.
#
# Usage:
#   bash scripts/multi_gpu_check.sh [options]
#
# Examples:
#   # Four-GPU machine (auto-detects GPU count, runs N=2 and N=4 on NCCL + CPU gloo comparison):
#   bash scripts/multi_gpu_check.sh --python 3.12
#
#   # Single-GPU shared NCCL bench (exports NCCL_MULTI_RANK_GPU_ENABLE=1, runs N=2 and N=4 on cuda:0):
#   bash scripts/multi_gpu_check.sh --shared-gpu --python 3.12
#
#   # CPU-only gloo dry run (exercises stage runner, resume logic, trainer check, and rank-failure check):
#   bash scripts/multi_gpu_check.sh --backend gloo --ranks 2 4 --log-dir /tmp/multi_gpu_dry
#
# Portable: uses no sudo, no cloud-specific commands, and no user/home assumptions; works as root
# inside Docker containers as well as on Linux/macOS developer machines. Does not depend on GNU
# `timeout` (uses a small Python subprocess process-group timeout wrapper). Records completed
# stages so re-running resumes from where a prior run stopped (pass --force to redo completed stages).
set -uo pipefail
cd "$(dirname "$0")/.."

BACKEND="nccl"
SHARED_GPU=0
RANKS_ARG=""
SIZES_ARG="__DEFAULT__"
PYTHON_VERSION=""
LOG_DIR="multi_gpu_check_logs"
FORCE=0
PG_TIMEOUT="20"
STEP_TIMEOUT="180"
STAGE_TIMEOUT="900"
REFERENCE_MODE="sketch"
EXTRA_DRIVER_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/multi_gpu_check.sh [options]

Options:
  --backend nccl|gloo        Backend to verify (default: nccl).
  --shared-gpu               Run 2 and 4 NCCL ranks on one GPU (exports NCCL_MULTI_RANK_GPU_ENABLE=1
                             and passes --device-map shared).
  --ranks N [N ...]          World sizes to test: space-, quote-, or comma-separated (default: all of
                             {2, 4} that fit available GPUs, or "2 4" under --shared-gpu / --backend gloo).
  --sizes SIZE [SIZE ...]    Released checkpoint sizes for NCCL runs: space-, quote-, or comma-separated
                             (default: "5M 28M" for nccl, empty for gloo; pass "" to skip released sizes).
  --python VERSION           Python version passed to `uv` (e.g. 3.12).
  --log-dir PATH             Directory for per-stage logs, reference archives, and completion markers
                             (default: multi_gpu_check_logs).
  --force                    Re-run stages even if already marked completed in --log-dir.
  --reference-mode MODE      Reference storage passed to --write-reference: sketch (default) or exact.
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
            RANKS_ARG="$2"
            shift 2
            while [[ $# -gt 0 && "$1" != -* ]]; do
                RANKS_ARG="${RANKS_ARG} $1"
                shift
            done
            ;;
        --sizes)
            require_arg "$1" "$#"
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
if [[ "$REFERENCE_MODE" != "sketch" && "$REFERENCE_MODE" != "exact" ]]; then
    echo "error: --reference-mode must be sketch or exact, got '$REFERENCE_MODE'" >&2
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

# Choose a standalone Python interpreter for the wall-clock timeout and free-port helpers so they
# never trigger an implicit `uv sync` on the project venv before stage `01_env` runs `uv sync --locked`.
declare -a HELPER_PY=()
if command -v python3 >/dev/null 2>&1; then
    HELPER_PY=(python3)
elif command -v python >/dev/null 2>&1; then
    HELPER_PY=(python)
else
    HELPER_PY=(uv run --no-project ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python)
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

GPU_COUNT=0
if [[ "$BACKEND" == "nccl" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c . || true)"
    fi
    if [[ "$GPU_COUNT" -lt 1 ]]; then
        echo "error: --backend nccl requested, but nvidia-smi reported 0 GPUs" >&2
        exit 2
    fi
fi

# Resolve rank counts (space- or comma-separated).
declare -a RANKS=()
if [[ -n "$RANKS_ARG" ]]; then
    RANKS_NORM="${RANKS_ARG//,/ }"
    # shellcheck disable=SC2206
    RANKS=($RANKS_NORM)
elif [[ "$BACKEND" == "gloo" || "$SHARED_GPU" -eq 1 ]]; then
    RANKS=(2 4)
else
    if [[ "$GPU_COUNT" -ge 2 ]]; then
        RANKS+=(2)
    fi
    if [[ "$GPU_COUNT" -ge 4 ]]; then
        RANKS+=(4)
    fi
    if [[ "${#RANKS[@]}" -eq 0 ]]; then
        echo "error: only $GPU_COUNT GPU detected; pass --shared-gpu to run N=2 and N=4 on one GPU" >&2
        exit 2
    fi
fi

for n in "${RANKS[@]}"; do
    if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -lt 2 ]]; then
        echo "error: rank count must be an integer >= 2, got '$n'" >&2
        exit 2
    fi
    if [[ "$BACKEND" == "nccl" && "$SHARED_GPU" -eq 0 && "$n" -gt "$GPU_COUNT" ]]; then
        echo "error: requested --ranks $n with per-rank NCCL, but only $GPU_COUNT GPU(s) detected (pass --shared-gpu to share 1 GPU)" >&2
        exit 2
    fi
done

# Resolve released checkpoint sizes (space- or comma-separated).
if [[ "$SIZES_ARG" == "__DEFAULT__" ]]; then
    if [[ "$BACKEND" == "nccl" ]]; then
        SIZES_ARG="5M 28M"
    else
        SIZES_ARG=""
    fi
fi
declare -a SIZES=()
if [[ -n "$SIZES_ARG" ]]; then
    SIZES_NORM="${SIZES_ARG//,/ }"
    # shellcheck disable=SC2206
    SIZES=($SIZES_NORM)
fi

DEVICE_MAP="per-rank"
if [[ "$SHARED_GPU" -eq 1 ]]; then
    DEVICE_MAP="shared"
fi

# Format log paths for summary.txt without embedding $HOME if LOG_DIR sits under $HOME.
display_log_path() {
    local raw="$1"
    if [[ -n "${HOME:-}" && "$raw" == "${HOME}"/* ]]; then
        printf "%s" "$(basename "$raw")"
    else
        printf "%s" "$raw"
    fi
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

# Run one command under a hard wall-clock timeout without relying on GNU coreutils `timeout`.
run_with_timeout() {
    local limit_s="$1"
    shift
    "${HELPER_PY[@]}" -c '
import os, signal, subprocess, sys
limit = float(sys.argv[1])
cmd = sys.argv[2:]
proc = subprocess.Popen(cmd, start_new_session=True)
try:
    rc = proc.wait(timeout=limit)
except subprocess.TimeoutExpired:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    print(f"ERROR: stage timed out after {limit:.0f}s and was killed", file=sys.stderr, flush=True)
    sys.exit(124)
sys.exit(rc)
' "$limit_s" "$@"
}

declare -a STAGE_NAMES=()
declare -a STAGE_RESULTS=()
declare -a STAGE_LOGS=()
ANY_FAILED=0

record_stage() {
    STAGE_NAMES+=("$1")
    STAGE_RESULTS+=("$2")
    STAGE_LOGS+=("$3")
}

is_stage_resumed() {
    local name="$1"
    local required_artifact="$2"
    local done_file="${LOG_DIR}/${name}.done"
    if [[ "$FORCE" -eq 0 && -f "$done_file" ]]; then
        if [[ -z "$required_artifact" || -f "$required_artifact" ]]; then
            return 0
        fi
    fi
    return 1
}

run_stage() {
    local name="$1"
    local required_artifact="$2"
    shift 2
    local log_file="${LOG_DIR}/${name}.log"
    local done_file="${LOG_DIR}/${name}.done"
    local shown_log
    shown_log="$(display_log_path "$log_file")"

    if is_stage_resumed "$name" "$required_artifact"; then
        echo "== [SKIP resumed] ${name} =="
        record_stage "$name" "0 (resumed)" "$shown_log"
        return 0
    fi
    rm -f "$done_file"

    echo "== [RUN] ${name} =="
    {
        echo "== stage: ${name} =="
        run_with_timeout "$STAGE_TIMEOUT" "$@"
    } 2>&1 | tee "$log_file"
    local rc="${PIPESTATUS[0]}"

    if [[ "$rc" -eq 0 ]]; then
        echo "0" >"$done_file"
        record_stage "$name" "0" "$shown_log"
    else
        ANY_FAILED=1
        record_stage "$name" "${rc}" "$shown_log"
    fi
    return 0
}

run_torchrun_stage() {
    local name="$1"
    local n="$2"
    shift 2
    local log_file="${LOG_DIR}/${name}.log"
    local shown_log
    shown_log="$(display_log_path "$log_file")"
    if is_stage_resumed "$name" ""; then
        echo "== [SKIP resumed] ${name} =="
        record_stage "$name" "0 (resumed)" "$shown_log"
        return 0
    fi
    local port
    port="$(pick_free_port)"
    run_stage "$name" "" \
        uv run ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} torchrun \
            --master-addr=127.0.0.1 --master-port="$port" --nproc_per_node="$n" \
            "$@"
}

# Stage 1: environment (sync locked venv quietly and record hardware/runtime without hostnames, zones, or usernames).
run_stage "01_env" "" bash -c '
set -euo pipefail
backend="$1"
shift
uv sync --locked --quiet "$@"
uv run "$@" python -c '\''
import os, platform, shutil, subprocess, sys, sysconfig, torch

backend = sys.argv[1]
print(f"os={platform.system()} arch={platform.machine()} python={platform.python_version()}")
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_runtime={torch.version.cuda}")
if torch.cuda.is_available():
    nccl = torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else None
    nccl_str = ".".join(str(x) for x in nccl) if nccl is not None else "unknown"
    print(f"nccl={nccl_str} gpu_count={torch.cuda.device_count()}")
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=False, capture_output=True, text=True,
    )
    if out.returncode == 0:
        for idx, line in enumerate(out.stdout.strip().splitlines()):
            print(f"gpu[{idx}]={line.strip()}")
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

# Stage 2: prototype probe on 4 GPUs (distributed_parity.py binds cuda:rank and requires >= 4 GPUs).
if [[ "$BACKEND" == "nccl" && "$SHARED_GPU" -eq 0 && "$GPU_COUNT" -ge 4 ]]; then
    run_stage "02_prototype_parity_cuda" "" \
        uv run ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python scripts/probes/distributed_parity.py --device cuda
else
    echo "== [SKIP] 02_prototype_parity_cuda (requires 4 distinct GPUs) =="
    record_stage "02_prototype_parity_cuda" "SKIPPED (needs 4 GPUs)" "-"
fi

# Stages 3-4: tiny_random reference + torchrun on the selected backend for each N.
REF_DEVICE="cpu"
if [[ "$BACKEND" == "nccl" ]]; then
    REF_DEVICE="cuda:0"
fi

for n in "${RANKS[@]}"; do
    ref_file="${REFS_DIR}/ref_${BACKEND}_tiny_n${n}.zip"
    run_stage "03_ref_${BACKEND}_tiny_n${n}" "$ref_file" \
        uv run ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python scripts/probes/distributed_trainer_check.py \
            --write-reference "$ref_file" \
            --reference-mode "$REFERENCE_MODE" \
            --device "$REF_DEVICE" \
            --world-size "$n" \
            ${EXTRA_DRIVER_ARGS[@]+"${EXTRA_DRIVER_ARGS[@]}"}

    if [[ -f "$ref_file" ]]; then
        run_torchrun_stage "04_${BACKEND}_tiny_n${n}" "$n" \
            scripts/probes/distributed_trainer_check.py \
            --backend "$BACKEND" \
            --device-map "$DEVICE_MAP" \
            --timeout "$STEP_TIMEOUT" \
            --reference "$ref_file" \
            ${EXTRA_DRIVER_ARGS[@]+"${EXTRA_DRIVER_ARGS[@]}"}
    else
        echo "== [SKIP] 04_${BACKEND}_tiny_n${n} (reference file missing due to prior stage failure) =="
        record_stage "04_${BACKEND}_tiny_n${n}" "SKIPPED (no reference)" "-"
        ANY_FAILED=1
    fi
done

# Stages 5-6: released checkpoints (e.g. 5M 28M) when requested.
if [[ "${#SIZES[@]}" -gt 0 ]]; then
    for n in "${RANKS[@]}"; do
        ref_file="${REFS_DIR}/ref_${BACKEND}_released_n${n}.zip"
        run_stage "05_ref_${BACKEND}_released_n${n}" "$ref_file" \
            uv run ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python scripts/probes/distributed_trainer_check.py \
                --write-reference "$ref_file" \
                --reference-mode "$REFERENCE_MODE" \
                --device "$REF_DEVICE" \
                --world-size "$n" \
                --no-tiny \
                --sizes "${SIZES[@]}" \
                ${EXTRA_DRIVER_ARGS[@]+"${EXTRA_DRIVER_ARGS[@]}"}

        if [[ -f "$ref_file" ]]; then
            run_torchrun_stage "06_${BACKEND}_released_n${n}" "$n" \
                scripts/probes/distributed_trainer_check.py \
                --backend "$BACKEND" \
                --device-map "$DEVICE_MAP" \
                --timeout "$STEP_TIMEOUT" \
                --no-tiny \
                --sizes "${SIZES[@]}" \
                --reference "$ref_file" \
                ${EXTRA_DRIVER_ARGS[@]+"${EXTRA_DRIVER_ARGS[@]}"}
        else
            echo "== [SKIP] 06_${BACKEND}_released_n${n} (reference file missing due to prior stage failure) =="
            record_stage "06_${BACKEND}_released_n${n}" "SKIPPED (no reference)" "-"
            ANY_FAILED=1
        fi
    done
fi

# Stages 7-8: CPU gloo comparison when the main session backend is nccl.
if [[ "$BACKEND" == "nccl" ]]; then
    for n in "${RANKS[@]}"; do
        ref_file="${REFS_DIR}/ref_gloo_tiny_n${n}.zip"
        run_stage "07_ref_gloo_tiny_n${n}" "$ref_file" \
            uv run ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python scripts/probes/distributed_trainer_check.py \
                --write-reference "$ref_file" \
                --reference-mode "$REFERENCE_MODE" \
                --device cpu \
                --world-size "$n" \
                ${EXTRA_DRIVER_ARGS[@]+"${EXTRA_DRIVER_ARGS[@]}"}

        if [[ -f "$ref_file" ]]; then
            run_torchrun_stage "08_gloo_tiny_n${n}" "$n" \
                scripts/probes/distributed_trainer_check.py \
                --backend gloo \
                --timeout "$STEP_TIMEOUT" \
                --reference "$ref_file" \
                ${EXTRA_DRIVER_ARGS[@]+"${EXTRA_DRIVER_ARGS[@]}"}
        else
            echo "== [SKIP] 08_gloo_tiny_n${n} (reference file missing due to prior stage failure) =="
            record_stage "08_gloo_tiny_n${n}" "SKIPPED (no reference)" "-"
            ANY_FAILED=1
        fi
    done
fi

# Stage 9: rank-failure timeout check for each configured rank count.
for n in "${RANKS[@]}"; do
    run_stage "09_rank_failure_${BACKEND}_n${n}" "" \
        uv run ${UV_PY_ARGS[@]+"${UV_PY_ARGS[@]}"} python scripts/probes/distributed_rank_failure_check.py \
            --backend "$BACKEND" \
            --device-map "$DEVICE_MAP" \
            --world-size "$n" \
            --pg-timeout "$PG_TIMEOUT"
done

# Final summary table: stage -> exit_code -> log_file (no hostnames, zones, usernames, or $HOME paths).
SUMMARY_FILE="${LOG_DIR}/summary.txt"
{
    echo ""
    echo "== multi_gpu_check summary (backend=${BACKEND}, device_map=${DEVICE_MAP}, ranks=${RANKS[*]}) =="
    printf "%-36s %-24s %s\n" "stage" "exit_code" "log_file"
    printf "%-36s %-24s %s\n" "------------------------------------" "------------------------" "----------------------------------------"
    for i in "${!STAGE_NAMES[@]}"; do
        printf "%-36s %-24s %s\n" "${STAGE_NAMES[$i]}" "${STAGE_RESULTS[$i]}" "${STAGE_LOGS[$i]}"
    done
} | tee "$SUMMARY_FILE"

exit "$ANY_FAILED"
