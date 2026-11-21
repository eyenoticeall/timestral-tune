"""Probe: how accurate are float32 gradients through the released checkpoints? (spec §12)

    uv run python scripts/probes/gradient_accuracy.py
    uv run python scripts/probes/gradient_accuracy.py --sizes 5M 28M --device mps

Observation being tracked: gradients of the training loss computed in float32 differ from a
float64 reference by far more than float32 round-off on the released checkpoints, while forward
outputs agree to about 3e-7. A randomly initialised model of the same architecture does not show
this. The released checkpoints produce attention logits of ~200k-500k (where one float32 ULP is
0.015625 below 262k and 0.03125 above it), causing two float32 effects: fused SDPA backward loses
accuracy on typical CPU/CUDA batches (reduced by `attention_context("math")` or explicit
matmul-and-softmax), and near-tied large-logit rows still suffer float32 ULP quantization under any
float32 attention kernel.

Six experiments, so the investigation can continue from evidence:

1. Matrix: relative L2 error of the float32 gradient, per model (a tiny random model and the
   released sizes), per input layout, on CPU and on the accelerator.
2. Attention comparison: the default kernel, `attention_context("math")` (unfused math
   SDPA), and explicit matmul-and-softmax (`manual_sdpa`) across all layouts on CPU and on the
   accelerator.
3. Standalone attention: float32 against float64 gradients of default SDPA, math SDPA, and explicit
   matmul-and-softmax from small logits up to ~400k (large-gap and near-tie keys).
4. The reference itself: how far a float64-weights reference moves when upstream's RMSNorm is left
   computing in float32, as it does by default. All other experiments use float64 RMSNorm in the
   reference (`_common.reference_gradient`).
5. Model attention logits and weight norms: per layout, the query row with the largest `|logit|`
   together with its own top-1/top-2 gap, the near-tie query row (`gap < 10`, `top1 > 1000`) with
   the largest logit together with its own gap, and the peak layer's Q/K/V/RMSNorm norms.
6. Multi-seed sweep on 5M past-only (CPU): median and max error across seeds for `default`,
   `math`, and `explicit`, showing typical improvement alongside occasional near-tie spikes.

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
from timestral_tune._vendor.timestral.layers import Attention, TwoStreamMask, _apply_rotary
from timestral_tune.models import TuneTimestralForecaster, TuneTimestralModel
from timestral_tune.models.forward import attention_context
from timestral_tune.trainer import compute_loss

# (label, rows per series, covariates, a known-future row among them)
LAYOUTS = (
    ("univariate", 1, False, True),
    ("multivariate", 3, False, True),
    ("covariates", 5, True, True),
    ("past-only", 3, True, False),
)


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


def explicit_loss_gradient(model: TuneTimestralModel, batch) -> tuple[float, torch.Tensor]:
    stock = layers._sdpa
    layers._sdpa = manual_sdpa
    try:
        return loss_gradient(model, batch)
    finally:
        layers._sdpa = stock


def inspect_attention_logits(
    model: TuneTimestralModel, batch
) -> tuple[tuple[float, float, str], tuple[float, float, str]]:
    """Return same-row `(peak_abs_logit, its_gap, module)` and `(max_near_tie_logit, its_gap, module)`.

    Both `(logit, gap)` pairs come from the exact same query row: the first from the query row
    with the largest finite logit overall, the second from the query row with the largest top-1
    logit among rows whose top-1 minus top-2 gap is below 10 (and top-1 > 1,000).
    """
    worst_peak = (0.0, float("nan"), "")
    worst_near_tie = (0.0, float("nan"), "")

    def inspect_scores(scores: torch.Tensor, mask: torch.Tensor | None, name: str) -> None:
        nonlocal worst_peak, worst_near_tie
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        if scores.shape[-1] < 2:
            return
        top2 = scores.topk(2, dim=-1).values
        top1, top2_val = top2[..., 0], top2[..., 1]
        finite_top1 = top1.isfinite()
        if not finite_top1.any():
            return
        gap = top1 - top2_val

        peak_flat = top1.masked_fill(~finite_top1, float("-inf")).argmax().item()
        peak_pos = torch.unravel_index(torch.tensor(peak_flat), top1.shape)
        peak_val = abs(top1[peak_pos].item())
        peak_gap = gap[peak_pos].item() if top2_val[peak_pos].isfinite() else float("inf")
        if peak_val > worst_peak[0]:
            worst_peak = (peak_val, peak_gap, name)

        near = finite_top1 & top2_val.isfinite() & (gap < 10.0) & (top1 > 1000.0)
        if near.any():
            near_flat = top1.masked_fill(~near, float("-inf")).argmax().item()
            near_pos = torch.unravel_index(torch.tensor(near_flat), top1.shape)
            near_val = top1[near_pos].item()
            near_gap = gap[near_pos].item()
            if near_val > worst_near_tie[0]:
                worst_near_tie = (near_val, near_gap, name)

    def make_hook(name: str):
        def hook(module: Attention, args: tuple) -> None:
            x, mask, cos, sin = args
            inputs = x if module.is_time else x.permute(1, 0, 2)
            q, k, _ = module._qkv(inputs)
            if module.is_time:
                q, k = _apply_rotary(q, k, cos, sin)
            qt, kt = q.transpose(1, 2).double(), k.transpose(1, 2).double()
            scale = math.sqrt(qt.shape[-1])
            if isinstance(mask, TwoStreamMask):
                n = mask.enc_len
                inspect_scores((qt[:, :, :n] @ kt[:, :, :n].transpose(-2, -1)) / scale, mask.mask[:, :, :n, :n], name)
                inspect_scores((qt[:, :, n:] @ kt.transpose(-2, -1)) / scale, mask.mask[:, :, n:, :], name)
            else:
                inspect_scores((qt @ kt.transpose(-2, -1)) / scale, mask, name)

        return hook

    handles = [
        module.register_forward_pre_hook(make_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, Attention)
    ]
    try:
        with torch.no_grad():
            compute_loss(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    return worst_peak, worst_near_tie


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=["5M", "28M"], choices=SIZES)
    parser.add_argument("--device", default=None, choices=["mps", "cuda"], help="also measure this accelerator")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sweep-seeds", type=int, default=15, help="number of seeds for the 5M past-only sweep")
    args = parser.parse_args()
    print(f"== float32 gradient accuracy | torch {torch.__version__} ==")

    print("\n[1] relative L2 error of the float32 training-loss gradient vs the float64-weights reference")
    forecasters = {"tiny random": tiny_random_forecaster()}
    forecasters.update({size: load(size) for size in args.sizes})
    for name, forecaster in forecasters.items():
        device_model = on_device(forecaster, args.device).model if args.device else None
        for label, rows_per_series, covariates, known in LAYOUTS:
            batch = make_batch(2, rows_per_series, 10, 3, covariates, seed=args.seed, known_future_row=known)
            _, reference = reference_gradient(forecaster.model, batch)
            _, cpu = loss_gradient(forecaster.model, batch)
            line = f"    {name:12s} {label:13s} cpu {rel(cpu, reference):.1e}"
            if device_model is not None:
                _, accelerated = loss_gradient(device_model, batch.to(args.device))
                line += f" | {args.device} {rel(accelerated, reference):.1e}"
            print(line)

    print("\n[2] default kernel vs attention_context('math') vs explicit matmul-and-softmax")
    for size in args.sizes:
        models = {"cpu": forecasters[size].model}
        if args.device:
            models[args.device] = on_device(forecasters[size], args.device).model
        for label, rows_per_series, covariates, known in LAYOUTS:
            batch = make_batch(2, rows_per_series, 10, 3, covariates, seed=args.seed, known_future_row=known)
            _, reference = reference_gradient(forecasters[size].model, batch)
            for device, model in models.items():
                on_dev = batch.to(device)
                _, with_default = loss_gradient(model, on_dev, attention_backend="default")
                _, with_math = loss_gradient(model, on_dev, attention_backend="math")
                _, with_explicit = explicit_loss_gradient(model, on_dev)
                print(
                    f"    {size:4s} {label:12s} {device:4s} default {rel(with_default, reference):.1e} | "
                    f"math {rel(with_math, reference):.1e} | explicit {rel(with_explicit, reference):.1e}"
                )

    print("\n[3] standalone attention on CPU: float32 vs float64 gradient, by attention-logit scale")
    torch.manual_seed(1)
    q0, k0, v0 = (torch.randn(8, 4, 70, 64, dtype=torch.float64) for _ in range(3))
    mask = torch.ones(70, 70, dtype=torch.bool).tril()[None, None]

    def stock_call(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def math_call(q, k, v):
        with attention_context("math"):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def explicit_call(q, k, v):
        return ((q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).masked_fill(~mask, float("-inf")).softmax(-1)) @ v

    def gradient(dtype, call, q_in, k_in, v_in):
        q, k, v = (t.to(dtype).detach().requires_grad_(True) for t in (q_in, k_in, v_in))
        (call(q, k, v).double() ** 2).mean().backward()
        return torch.cat([t.grad.double().flatten() for t in (q, k, v)])

    for scale in (1, 4, 16, 300):
        q_s, k_s = q0 * scale, k0 * scale
        reference = gradient(torch.float64, explicit_call, q_s, k_s, v0)
        max_logit = (q_s @ k_s.transpose(-2, -1) / 8).masked_fill(~mask, 0).abs().max().item()
        print(
            f"    max |logit| {max_logit:8.1f}  "
            f"default {rel(gradient(torch.float32, stock_call, q_s, k_s, v0), reference):.1e} | "
            f"math {rel(gradient(torch.float32, math_call, q_s, k_s, v0), reference):.1e} | "
            f"explicit {rel(gradient(torch.float32, explicit_call, q_s, k_s, v0), reference):.1e}"
        )

    # Near-tie at ~350k logits (similar patches in a large-logit layer): float32 ULP is 0.03125.
    q_big, k_big = q0 * 300, k0 * 300
    k_tie = k_big.clone()
    k_tie[:, :, 1:, :] = k_tie[:, :, :1, :] + 1e-4 * torch.randn_like(k_tie[:, :, 1:, :]) * 300
    ref_tie = gradient(torch.float64, explicit_call, q_big, k_tie, v0)
    max_tie = (q_big @ k_tie.transpose(-2, -1) / 8).masked_fill(~mask, 0).abs().max().item()
    print(
        f"    max |logit| {max_tie:8.1f} (near-tie)  "
        f"default {rel(gradient(torch.float32, stock_call, q_big, k_tie, v0), ref_tie):.1e} | "
        f"math {rel(gradient(torch.float32, math_call, q_big, k_tie, v0), ref_tie):.1e} | "
        f"explicit {rel(gradient(torch.float32, explicit_call, q_big, k_tie, v0), ref_tie):.1e}"
    )

    print("\n[4] how much upstream's float32 RMSNorm moves the float64-weights reference itself")
    for name, forecaster in forecasters.items():
        for label, rows_per_series, covariates, known in LAYOUTS:
            batch = make_batch(2, rows_per_series, 10, 3, covariates, seed=args.seed, known_future_row=known)
            _, reference = reference_gradient(forecaster.model, batch)
            _, float32_norm = reference_gradient(forecaster.model, batch, float64_norm=False)
            print(f"    {name:12s} {label:13s} {rel(float32_norm, reference):.1e}")

    print("\n[5] same-row peak attention logits, near-tie rows (gap < 10), and peak-layer weight norms")
    for size in args.sizes:
        model = forecasters[size].model
        d_model = model.config.d_model
        best_logit, best_layer = 0.0, ""
        for label, rows_per_series, covariates, known in LAYOUTS:
            batch = make_batch(2, rows_per_series, 10, 3, covariates, seed=args.seed, known_future_row=known)
            (peak_val, peak_gap, peak_mod), (tie_val, tie_gap, tie_mod) = inspect_attention_logits(model, batch)
            if peak_val > best_logit:
                best_logit, best_layer = peak_val, peak_mod
            tie_str = f"{tie_val:9.1f} (gap {tie_gap:4.2f}, {tie_mod})" if tie_val > 0 else "none"
            print(f"    {size:4s} {label:12s} peak {peak_val:9.1f} (gap {peak_gap:7.1f}, {peak_mod}) | tie {tie_str}")
        if best_layer:
            attn = model.get_submodule(best_layer)
            wq, wk, wv = attn.wQKV.weight.split(d_model, dim=0)
            parts = best_layer.split(".")
            norm = model.get_submodule(f"{parts[0]}.{parts[1]}.{parts[2]}.attn_norms.{parts[4]}")
            print(
                f"    {size:4s} {best_layer}: ||norm.scale||_inf={norm.scale.abs().max():.2f}, "
                f"||Wq||_2={torch.linalg.matrix_norm(wq, 2):.1f}, "
                f"||Wk||_2={torch.linalg.matrix_norm(wk, 2):.1f}, "
                f"||Wv||_2={torch.linalg.matrix_norm(wv, 2):.1f}"
            )

    if "5M" in forecasters and args.sweep_seeds > 0:
        print(f"\n[6] 5M past-only CPU sweep across {args.sweep_seeds} seeds: default vs math vs explicit")
        model = forecasters["5M"].model
        errors = {"default": [], "math": [], "explicit": []}
        worst_math = (0.0, -1, 0.0, float("nan"))
        for seed in range(args.sweep_seeds):
            batch = make_batch(2, 3, 10, 3, True, seed=seed, known_future_row=False)
            _, reference = reference_gradient(model, batch)
            e_def = rel(loss_gradient(model, batch, attention_backend="default")[1], reference)
            e_math = rel(loss_gradient(model, batch, attention_backend="math")[1], reference)
            e_exp = rel(explicit_loss_gradient(model, batch)[1], reference)
            errors["default"].append(e_def)
            errors["math"].append(e_math)
            errors["explicit"].append(e_exp)
            if e_math > worst_math[0]:
                _, (tie_val, tie_gap, _) = inspect_attention_logits(model, batch)
                worst_math = (e_math, seed, tie_val, tie_gap)
        for name, vals in errors.items():
            ordered = sorted(vals)
            print(f"    {name:8s} median {ordered[len(ordered) // 2]:.2e} | max {ordered[-1]:.2e}")
        print(
            f"    worst math seed={worst_math[1]}: rel L2 {worst_math[0]:.2e} "
            f"(same-row max near-tie {worst_math[2]:.1f}, gap {worst_math[3]:.2f})"
        )


if __name__ == "__main__":
    main()
