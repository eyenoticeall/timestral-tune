"""Probe: the unexplained CPU fp32 gradient gap (handoff §12).

    uv run python scripts/probes/cpu_gradient_gap.py            # 5M
    uv run python scripts/probes/cpu_gradient_gap.py --size 28M

Observation being tracked: with float32 weights on CPU, gradients of the training loss differ from
a float64-weights reference by far more than forward outputs do, while the Apple GPU does not show
this (see scripts/probes/device_parity.py). No cause is established. This script records three
experiments so that the investigation can continue from evidence:

1. The gap with the stock attention call (`F.scaled_dot_product_attention`).
2. The gap after replacing that call with an explicit matmul-and-softmax, in the same model.
3. A standalone attention call, fp32 against fp64, outside the model.

Results are recorded in docs/VERIFICATION.md.
"""

from __future__ import annotations

import argparse
import copy
import math

import torch
import torch.nn.functional as F
from _common import flat_grads, load, make_batch, real_loss, rel

import timestral_tune._vendor.timestral.layers as layers


def manual_sdpa(query, key, value, mask, is_causal=False):
    """Drop-in for the vendored `_sdpa`: same maths, written out."""
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
    parser.add_argument("--size", default="5M", choices=["5M", "28M", "123M", "209M"])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(f"== Timestral-{args.size} on cpu | torch {torch.__version__} ==")

    model = load(args.size, "cpu").model.train()
    reference = copy.deepcopy(model).double().train()
    batch = make_batch(2, 3, 32, 3, True, seed=args.seed)

    def gradient(m, double=False):
        m.zero_grad()
        real_loss(m, batch, "cpu", double=double).backward()
        return flat_grads(m)

    ref = gradient(reference, double=True)
    gap_stock = rel(gradient(model), ref)
    print(f"[1] stock attention:   cpu fp32 gradient rel L2 vs float64-weights reference: {gap_stock:.2e}")
    stock = layers._sdpa
    layers._sdpa = manual_sdpa
    try:
        gap_manual = rel(gradient(model), ref)
    finally:
        layers._sdpa = stock
    print(f"[2] manual attention:  cpu fp32 gradient rel L2 vs the same reference:        {gap_manual:.2e}")

    torch.manual_seed(1)
    q, k, v = (torch.randn(8, 4, 70, 64, dtype=torch.float64) for _ in range(3))
    mask = torch.ones(70, 70, dtype=torch.bool).tril()[None, None]

    def standalone(dtype, fn):
        qq, kk, vv = (t.to(dtype).clone().requires_grad_(True) for t in (q, k, v))
        out = fn(qq, kk, vv)
        (out**2).mean().backward()
        return torch.cat([t.grad.double().flatten() for t in (qq, kk, vv)])

    def stock_call(a, b, c):
        return F.scaled_dot_product_attention(a, b, c, attn_mask=mask)

    gap = rel(standalone(torch.float32, stock_call), standalone(torch.float64, stock_call))
    print(f"[3] standalone stock attention, fp32 vs fp64 gradient rel L2:                  {gap:.2e}")


if __name__ == "__main__":
    main()
