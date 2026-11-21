#!/usr/bin/env bash
# Run everything that has never been run on CUDA (spec §10, Phase 1; spec §12).
#
#   bash scripts/cuda_check.sh            # on a machine with an NVIDIA GPU, from the repository root
#
# Needs uv and network access to the Hugging Face Hub (it downloads the 5M and 28M checkpoints).
# Writes cuda_check.log next to the repository's README. Record its numbers in docs/VERIFICATION.md,
# then tick the T14-on-CUDA box in docs/SPEC.md §10.
set -uo pipefail
cd "$(dirname "$0")/.."

{
    echo "== environment =="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "no NVIDIA GPU found"; exit 2; }
    uv sync --locked
    uv run python -c "import torch, sys; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| cuda runtime', torch.version.cuda)"
    uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)" || { echo "this PyTorch build cannot use the GPU"; exit 3; }

    echo "== T14 and every other test with a device parameter (-s prints the gradient errors) =="
    uv run pytest -s -q -rs tests/acceptance/test_device_parity.py tests/test_trainer.py tests/test_validation.py tests/test_loader_device.py
    tests_status=$?

    echo "== the whole suite =="
    uv run pytest -q
    suite_status=$?

    echo "== probes =="
    uv run python scripts/probes/device_parity.py --device cuda --size 5M
    uv run python scripts/probes/device_parity.py --device cuda --size 28M
    uv run python scripts/probes/gradient_accuracy.py --device cuda
    uv run python scripts/probes/step_timing.py --device cuda

    echo "== result: device tests exit $tests_status, whole suite exit $suite_status =="
    exit $(( tests_status || suite_status ))
} 2>&1 | tee cuda_check.log
exit "${PIPESTATUS[0]}"
