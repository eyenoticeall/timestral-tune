"""Probe: does a DataLoader with a worker process leave the interpreter hanging at exit?

    uv run --isolated --python 3.12 --with "torch==2.4.0" python scripts/probes/worker_exit_hang.py

This script deliberately imports nothing from timestral-tune. On macOS with PyTorch 2.4.0 it prints
its result and then never exits; on PyTorch 2.8.0 and newer it exits normally.

What was observed while it hung: PyTorch's shared-memory helper process (`torch_shm_manager`) held
the write end of the pipe that Python's multiprocessing resource tracker reads. The tracker
therefore never saw end-of-file, and the interpreter waited for it forever at shutdown.

The same happens to any training run that sets `num_workers > 0` on that combination, and to the
test suite, which is why one test is skipped there. Results are recorded in docs/VERIFICATION.md.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Arrays(Dataset):
    def __init__(self) -> None:
        self.values = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.values[index]


def collate(rows: list[np.ndarray]) -> dict[str, torch.Tensor]:
    values = torch.from_numpy(np.stack(rows))
    return {"values": values, "mask": torch.isfinite(values)}


if __name__ == "__main__":
    batches = list(DataLoader(Arrays(), batch_size=4, collate_fn=collate, num_workers=1))
    print(f"read {len(batches)} batches through a worker on torch {torch.__version__}; exiting now")
