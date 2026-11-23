#!/usr/bin/env bash
# Thin bash 3.2+ wrapper around scripts/multi_gpu_check.py.
#
# Keeps `bash scripts/multi_gpu_check.sh [options]` working unchanged while keeping all signal
# handling, process-tree management, resume signatures, and stage sequencing in a single Python
# process. Selects Python >= 3.10 (preferring `python3` on PATH, falling back to `uv run
# --no-project python` before `uv sync` has run) and `exec`s scripts/multi_gpu_check.py directly so
# the Python process owns its own inherited signal dispositions and exit code.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if command -v python3 >/dev/null 2>&1 \
    && python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    exec python3 "${SCRIPT_DIR}/multi_gpu_check.py" ${1+"$@"}
fi

if command -v uv >/dev/null 2>&1; then
    declare -a UV_PY_ARGS=()
    prev=""
    for arg in ${1+"$@"}; do
        if [[ "$prev" == "--python" ]]; then
            UV_PY_ARGS=(--python "$arg")
            break
        fi
        prev="$arg"
    done
    if [[ "${#UV_PY_ARGS[@]}" -eq 0 ]]; then
        UV_PY_ARGS=(--python ">=3.10")
    fi
    exec uv run --no-project "${UV_PY_ARGS[@]}" python "${SCRIPT_DIR}/multi_gpu_check.py" ${1+"$@"}
fi

echo "error: Python >= 3.10 or 'uv' is required on PATH to run scripts/multi_gpu_check.py" >&2
exit 2
