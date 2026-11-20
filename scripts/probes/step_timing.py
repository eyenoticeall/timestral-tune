"""Probe: time one full training step on CPU and on another device (spec §5.4).

    uv run python scripts/probes/step_timing.py --device mps
    uv run python scripts/probes/step_timing.py --device mps --sizes 5M 28M --batch 32

The step is the real one: normalised forward, median-folded target frame (with its CPU round
trip), pinball loss on target rows, backward, AdamW update. Context 1,024 steps, horizon 96.
Reports the median of the steps after the first. Results depend on the machine; the ones behind
the specification are recorded in docs/VERIFICATION.md.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from _common import install_cpu_offload, load, make_batch, real_loss


def median_step_seconds(model, device: str, batch, steps: int) -> float:
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-7)
    timings = []
    for _ in range(steps):
        start = time.perf_counter()
        optimiser.zero_grad()
        real_loss(model, batch, device).backward()
        optimiser.step()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    timings = sorted(timings[1:])
    return timings[len(timings) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", required=True, choices=["mps", "cuda"])
    parser.add_argument("--sizes", nargs="+", default=["5M", "28M"], choices=["5M", "28M", "123M", "209M"])
    parser.add_argument("--batch", type=int, default=32, help="series per batch")
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    install_cpu_offload()
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"== step timing | torch {torch.__version__} | cpu threads {torch.get_num_threads()} | batch {args.batch} ==")

    layouts = (("univariate", 1, False), ("5 rows per series, with covariates", 5, True))
    for size in args.sizes:
        m_cpu = load(size, "cpu").model.train()
        m_dev = load(size, args.device).model.train()
        for label, rows_per_series, covariates in layouts:
            batch = make_batch(args.batch, rows_per_series, 32, 3, covariates, seed=3)
            t_cpu = median_step_seconds(m_cpu, "cpu", batch, args.steps)
            t_dev = median_step_seconds(m_dev, args.device, batch, args.steps)
            if args.device == "mps":
                allocated = torch.mps.driver_allocated_memory() / 2**30
            else:
                allocated = torch.cuda.max_memory_allocated() / 2**30
            print(f"{size:5s} {label:36s} cpu {t_cpu * 1e3:6.0f} ms | {args.device} {t_dev * 1e3:6.0f} ms | "
                  f"{t_cpu / t_dev:3.1f}x | {args.device} memory allocated {allocated:4.1f} GiB")  # fmt: skip


if __name__ == "__main__":
    main()
