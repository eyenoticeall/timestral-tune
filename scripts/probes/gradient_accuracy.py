"""Probe: how accurate are float32 gradients through the released checkpoints? (spec §12)

    uv run python scripts/probes/gradient_accuracy.py
    uv run python scripts/probes/gradient_accuracy.py --sizes 5M 28M --device mps

Observation being tracked: gradients of the training loss computed in float32 differ from a
float64-weights reference by far more than float32 round-off on the released checkpoints, while
forward outputs agree to about 3e-7. A randomly initialised model of the same architecture does
not show this. CPU is usually 10 to 100 times worse than the Apple GPU. No cause is identified.

Three experiments, so the investigation can continue from evidence:

1. Matrix: relative L2 error of the float32 gradient, per model (a tiny random model and the
   released sizes), per input layout, on CPU and on the accelerator.
2. In-model attention swap: the CPU error after replacing upstream's attention call
   (`F.scaled_dot_product_attention`) with an explicit matmul-and-softmax.
3. Standalone attention: float32 against float64 gradients of the stock and the explicit attention
   call, outside the model, as the attention logits are scaled up.

Results are recorded in docs/VERIFICATION.md.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from _common import SIZES, load, loss_gradient, make_batch, on_device, reference_gradient, rel

import timestral_tune._vendor.timestral.layers as layers
from timestral_tune._vendor.timestral import TimestralConfig
from timestral_tune.models import TuneTimestralForecaster, TuneTimestralModel

LAYOUTS = (("univariate", 1, False), ("multivariate", 3, False), ("covariates", 5, True))


def tiny_random_forecaster() -> TuneTimestralForecaster:
    """A small randomly initialised model with the released patch size, to separate architecture from weights."""
    torch.manual_seed(0)
    levels = (0.01, *(round(0.05 * i, 2) for i in range(1, 20)), 0.99)
    config = TimestralConfig(
        patch_size=32,
        d_model=64,
        num_heads=2,
        mlp_hidden_dim=128,
        block_schedule=("time", "time_variate", "time", "time_variate"),
        quantile_knots=levels,
        block_size=32,
        context_length=32768,
    )
    model = TuneTimestralModel(config)
    with torch.no_grad():
        model.representation.forecast_query.normal_(std=0.5)
    return TuneTimestralForecaster(model.eval())


def manual_sdpa(query, key, value, mask, is_causal=False):
    """Drop-in for the vendored `_sdpa`: the same maths, written out."""
    batch, seq, _, _ = query.shape
    q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    elif is_causal:
        causal = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
    out = scores.softmax(-1) @ v
    return out.transpose(1, 2).contiguous().view(batch, seq, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=["5M", "28M"], choices=SIZES)
    parser.add_argument("--device", default=None, choices=["mps", "cuda"], help="also measure this accelerator")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(f"== float32 gradient accuracy | torch {torch.__version__} ==")

    print("\n[1] relative L2 error of the float32 training-loss gradient vs the float64-weights reference")
    forecasters = {"tiny random": tiny_random_forecaster()}
    forecasters.update({size: load(size) for size in args.sizes})
    for name, forecaster in forecasters.items():
        device_model = on_device(forecaster, args.device).model if args.device else None
        for label, rows_per_series, covariates in LAYOUTS:
            batch = make_batch(2, rows_per_series, 10, 3, covariates, seed=args.seed)
            _, reference = reference_gradient(forecaster.model, batch)
            _, cpu = loss_gradient(forecaster.model, batch)
            line = f"    {name:12s} {label:13s} cpu {rel(cpu, reference):.1e}"
            if device_model is not None:
                _, accelerated = loss_gradient(device_model, batch.to(args.device))
                line += f" | {args.device} {rel(accelerated, reference):.1e}"
            print(line)

    print("\n[2] CPU error with upstream's attention call replaced by an explicit matmul and softmax")
    stock = layers._sdpa
    for size in args.sizes:
        model = forecasters[size].model
        batch = make_batch(2, 5, 10, 3, True, seed=args.seed)
        _, reference = reference_gradient(model, batch)
        _, with_stock = loss_gradient(model, batch)
        layers._sdpa = manual_sdpa
        try:
            _, with_manual = loss_gradient(model, batch)
        finally:
            layers._sdpa = stock
        stock_error, explicit_error = rel(with_stock, reference), rel(with_manual, reference)
        print(f"    {size:4s} covariates  stock {stock_error:.1e} | explicit {explicit_error:.1e}")

    print("\n[3] standalone attention on CPU: float32 vs float64 gradient, by attention-logit scale")
    torch.manual_seed(1)
    q0, k0, v0 = (torch.randn(8, 4, 70, 64, dtype=torch.float64) for _ in range(3))
    mask = torch.ones(70, 70, dtype=torch.bool).tril()[None, None]

    def stock_call(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def explicit_call(q, k, v):
        return ((q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).masked_fill(~mask, float("-inf")).softmax(-1)) @ v

    def gradient(dtype, call, scale):
        q, k, v = ((t * (scale if i < 2 else 1)).to(dtype).requires_grad_(True) for i, t in enumerate((q0, k0, v0)))
        (call(q, k, v).double() ** 2).mean().backward()
        return torch.cat([t.grad.double().flatten() for t in (q, k, v)])

    for scale in (1, 4, 16):
        reference = gradient(torch.float64, explicit_call, scale)
        max_logit = ((q0 * scale) @ (k0 * scale).transpose(-2, -1) / 8).abs().max().item()
        stock_error = rel(gradient(torch.float32, stock_call, scale), reference)
        explicit_error = rel(gradient(torch.float32, explicit_call, scale), reference)
        print(f"    max |logit| {max_logit:7.1f}  stock {stock_error:.1e} | explicit {explicit_error:.1e}")


if __name__ == "__main__":
    main()
