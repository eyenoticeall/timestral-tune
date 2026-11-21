"""Probe: time one full training step on CPU and on an accelerator (spec §5.4).

    uv run python scripts/probes/step_timing.py --device mps
    uv run python scripts/probes/step_timing.py --device cuda --bf16 --compile default
    uv run python scripts/probes/step_timing.py --device cuda --mode lora_variate --bf16

The step is the library's own: `compute_loss` (normalised forward, median-folded target frame,
pinball loss), backward over trainable parameters, AdamW update. Context 1,024 steps, horizon 96.
Reports the median of the steps after warm-up. Results depend on the machine; the ones behind
the specification are recorded in docs/VERIFICATION.md.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import time

import torch
from _common import SIZES, load, make_batch, on_device

from timestral_tune.config import LoraConfig
from timestral_tune.models import inject_lora, lora_parameters
from timestral_tune.trainer import compute_loss


def median_step_seconds(
    model,
    device: str,
    batch,
    steps: int,
    *,
    mode: str = "full_ft",
    bf16: bool = False,
    compile_mode: str | None = None,
) -> tuple[float, float]:
    model = copy.deepcopy(model).train()
    if mode == "lora_variate":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        inject_lora(model, LoraConfig(r=8, lora_alpha=16, axes=("variate",), targets=("q", "v")))
        trainable = list(lora_parameters(model))
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        trainable = list(model.parameters())
    if compile_mode is not None:
        model.transformer = torch.compile(model.transformer, mode=compile_mode)

    adamw_kwargs: dict[str, object] = {"lr": 1e-7}
    if device == "cuda":
        adamw_kwargs["fused"] = True
    optimiser = torch.optim.AdamW(trainable, **adamw_kwargs)
    batch = batch.to(device)
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if (bf16 and device == "cuda") else contextlib.nullcontext()
    )
    warmup = 3 if compile_mode is not None else 1

    def run_one() -> None:
        optimiser.zero_grad(set_to_none=True)
        with amp_ctx:
            loss = compute_loss(model, batch, validate=False).loss
        loss.backward()
        optimiser.step()

    for _ in range(warmup):
        run_one()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timings = []
    for _ in range(steps):
        start = time.perf_counter()
        run_one()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    timings.sort()
    allocated = (
        torch.mps.driver_allocated_memory() / 2**30
        if device == "mps"
        else (torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0)
    )
    return timings[len(timings) // 2], allocated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", required=True, choices=["mps", "cuda"])
    parser.add_argument("--sizes", nargs="+", default=["5M", "28M"], choices=SIZES)
    parser.add_argument("--batch", type=int, default=32, help="series per batch")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--mode", default="full_ft", choices=["full_ft", "lora_variate"])
    parser.add_argument("--bf16", action="store_true", help="use bfloat16 autocast on CUDA")
    parser.add_argument("--compile", dest="compile_mode", default=None, choices=["default", "reduce-overhead"])
    args = parser.parse_args()
    if args.bf16 and args.device != "cuda":
        parser.error("--bf16 is supported on CUDA only")
    torch.set_num_threads(os.cpu_count() or 1)
    print(
        f"== step timing | torch {torch.__version__} | cpu threads {torch.get_num_threads()} "
        f"| batch {args.batch} | mode {args.mode} | bf16 {args.bf16} | compile {args.compile_mode} =="
    )

    layouts = (
        (("5 rows per series, with covariates", 5, True),)
        if args.mode == "lora_variate"
        else (("univariate", 1, False), ("5 rows per series, with covariates", 5, True))
    )
    for size in args.sizes:
        cpu_model = load(size, "cpu").model.train()
        device_model = on_device(load(size, "cpu"), args.device).model.train()
        for label, rows_per_series, covariates in layouts:
            batch = make_batch(args.batch, rows_per_series, 32, 3, covariates, seed=3)
            t_cpu, _ = median_step_seconds(cpu_model, "cpu", batch, args.steps, mode=args.mode)
            t_dev, allocated = median_step_seconds(
                device_model,
                args.device,
                batch,
                args.steps,
                mode=args.mode,
                bf16=args.bf16,
                compile_mode=args.compile_mode,
            )
            print(
                f"{size:5s} {label:36s} cpu {t_cpu * 1e3:6.0f} ms | {args.device} {t_dev * 1e3:6.0f} ms | "
                f"{t_cpu / t_dev:3.1f}x | {args.device} memory allocated {allocated:4.2f} GiB"
            )


if __name__ == "__main__":
    main()
