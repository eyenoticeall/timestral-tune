#!/usr/bin/env bash
# Thin bash 3.2+ wrapper around scripts/multi_gpu_check.py.
#
# Keeps `bash scripts/multi_gpu_check.sh [options]` working unchanged while keeping all CLI parsing,
# signal handling, process-tree management, resume signatures, and stage sequencing in a single
# stdlib-only Python process. Selects any Python >= 3.10 (preferring `python3` on PATH, then
# resolving an installed interpreter via `uv python find --no-project ">=3.10"`, before falling back
# to `uv run --no-project --python ">=3.10" python`) without inspecting its arguments, and `exec`s
# scripts/multi_gpu_check.py so the Python process owns its own PID, signal dispositions, and exit
# code. `--python` is forwarded untouched to select the uv project environment for the stages.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

_is_py310() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

_uv_find_py310() {
    local found=""
    if found="$(uv python find --no-project --resolve-links ">=3.10" 2>/dev/null)" \
        && [[ -n "$found" && -x "$found" ]] && _is_py310 "$found"; then
        printf "%s\n" "$found"
        return 0
    fi
    if found="$(uv python find --no-project ">=3.10" 2>/dev/null)" \
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
    py_bin=""
    if py_bin="$(_uv_find_py310)"; then
        exec "$py_bin" "${SCRIPT_DIR}/multi_gpu_check.py" ${1+"$@"}
    fi
    exec uv run --no-project --python ">=3.10" python "${SCRIPT_DIR}/multi_gpu_check.py" ${1+"$@"}
fi

echo "error: Python >= 3.10 or 'uv' is required on PATH to run scripts/multi_gpu_check.py" >&2
exit 2
