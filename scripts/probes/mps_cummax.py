"""Probe: does this PyTorch implement `cummax` on the Apple GPU? (spec §5.4, decision D9)

    uv run --isolated --python 3.12 --with "torch==2.7.1" python scripts/probes/mps_cummax.py
    uv run --isolated --python 3.12 --with "torch==2.8.0" python scripts/probes/mps_cummax.py

Upstream's input assembly forward-fills missing values with `Tensor.cummax` (`ffill_with_mask`).
PyTorch releases before 2.8 do not implement that operator on the Apple GPU, which is why the library
refuses the device there (`MIN_TORCH_FOR_MPS`). This script imports nothing from timestral-tune, so it
can be run against any PyTorch version to re-check the floor. Results are in docs/VERIFICATION.md §10.
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    if not torch.backends.mps.is_available():
        print(f"torch {torch.__version__}: no Apple GPU on this machine; nothing to check")
        return 2
    values = torch.arange(6, device="mps").reshape(2, 3)
    try:
        result = values.cummax(dim=-1).values.cpu()
    except NotImplementedError as error:
        print(f"torch {torch.__version__}: cummax on the Apple GPU raises NotImplementedError ({str(error)[:70]}...)")
        return 1
    correct = torch.equal(result, torch.arange(6).reshape(2, 3))
    print(
        f"torch {torch.__version__}: cummax on the Apple GPU works" + ("" if correct else ", but returns wrong values")
    )
    return 0 if correct else 1


if __name__ == "__main__":
    sys.exit(main())
