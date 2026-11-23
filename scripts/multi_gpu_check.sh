#!/usr/bin/env bash
# Thin bash 3.2+ wrapper around scripts/multi_gpu_check.py.
#
# Keeps `bash scripts/multi_gpu_check.sh [options]` working unchanged while keeping all signal
# handling, process-tree management, resume signatures, and stage sequencing in a single Python
# process. Selects Python >= 3.10 (preferring `python3` on PATH, then resolving a Python >= 3.10
# interpreter directly via `uv python find --no-project` before falling back to `uv run
# --no-project python`) and `exec`s scripts/multi_gpu_check.py so the Python process owns its own
# PID, inherited signal dispositions, and exit code.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

_is_py310() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

_uv_find_py310() {
    local req="$1"
    local found=""
    if found="$(uv python find --no-project --resolve-links "$req" 2>/dev/null)" \
        && [[ -n "$found" && -x "$found" ]] && _is_py310 "$found"; then
        printf "%s\n" "$found"
        return 0
    fi
    if found="$(uv python find --no-project "$req" 2>/dev/null)" \
        && [[ -n "$found" && -x "$found" ]] && _is_py310 "$found"; then
        printf "%s\n" "$found"
        return 0
    fi
    return 1
}

if command -v python3 >/dev/null 2>&1 && _is_py310 python3; then
    exec python3 "${SCRIPT_DIR}/multi_gpu_check.py" ${1+"$@"}
fi

if command -v uv >/dev/null 2>&1; then
    py_req=""
    declare -a args=()
    if [[ $# -gt 0 ]]; then
        args=("$@")
    fi
    idx=0
    n="${#args[@]}"
    while [[ "$idx" -lt "$n" ]]; do
        arg="${args[$idx]}"
        case "$arg" in
            --python)
                if [[ $((idx + 1)) -lt "$n" ]]; then
                    py_req="${args[$((idx + 1))]}"
                    idx=$((idx + 2))
                else
                    idx=$((idx + 1))
                fi
                ;;
            --backend|--log-dir|--reference-mode|--pg-timeout|--step-timeout|--stage-timeout|--only|--skip)
                idx=$((idx + 2))
                ;;
            --ranks|--sizes)
                idx=$((idx + 2))
                while [[ "$idx" -lt "$n" && "${args[$idx]}" != -* ]]; do
                    idx=$((idx + 1))
                done
                ;;
            *)
                idx=$((idx + 1))
                ;;
        esac
    done
    if [[ -z "$py_req" ]]; then
        py_req=">=3.10"
    fi

    py_bin=""
    if py_bin="$(_uv_find_py310 "$py_req")"; then
        exec "$py_bin" "${SCRIPT_DIR}/multi_gpu_check.py" ${1+"$@"}
    fi
    if [[ "$py_req" != ">=3.10" ]] && py_bin="$(_uv_find_py310 ">=3.10")"; then
        exec "$py_bin" "${SCRIPT_DIR}/multi_gpu_check.py" ${1+"$@"}
    fi
    exec uv run --no-project --python "$py_req" python "${SCRIPT_DIR}/multi_gpu_check.py" ${1+"$@"}
fi

echo "error: Python >= 3.10 or 'uv' is required on PATH to run scripts/multi_gpu_check.py" >&2
exit 2
