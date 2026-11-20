"""Probe: does a non-CPU device reproduce CPU forecasts and accurate gradients? (spec §5.4, T14)

    uv run python scripts/probes/device_parity.py --device mps            # 5M
    uv run python scripts/probes/device_parity.py --device mps --size 28M
    uv run python scripts/probes/device_parity.py --device cuda

Prints forward parity against CPU, gradients of the real training loss against a float64-weights
reference (for both CPU fp32 and the device), and what autocast does on the device.

The reference is the same model with float64 weights and inputs on CPU. Upstream's RMSNorm and
RoPE still compute in float32 internally, so it is not a pure float64 computation.
Expected output is recorded in docs/VERIFICATION.md. This is a prototype of acceptance test T14.
"""

from __future__ import annotations

import argparse
import copy

import torch
from _common import flat_grads, install_cpu_offload, load, make_batch, real_loss, rel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", required=True, choices=["mps", "cuda"])
    parser.add_argument("--size", default="5M", choices=["5M", "28M", "123M", "209M"])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    dev = args.device
    install_cpu_offload()
    print(f"== Timestral-{args.size} on {dev} | torch {torch.__version__} ==")

    f_cpu, f_dev = load(args.size, "cpu"), load(args.size, dev)
    m_cpu, m_dev = f_cpu.model, f_dev.model

    def on(device, tensors):
        return [None if t is None else t.to(device) for t in tensors]

    # Forward parity: predict_block in both layouts.
    for label, covariates in (("no covariates", False), ("with covariates", True)):
        x, obs, gid, _, roles, kf, kfm = make_batch(2, 3, 32, 3, covariates, seed=args.seed)
        with torch.no_grad():
            a = m_cpu.predict_block(x, obs, gid, 3, roles, kf, kfm)
            b = m_dev.predict_block(*on(dev, (x, obs, gid)), 3, *on(dev, (roles, kf, kfm)))
        print(f"[fwd]  predict_block, {label:16s} rel diff vs cpu: {rel(b, a):.2e}")

    # Forward parity: the full forecaster, single block and block-autoregressive.
    g = torch.Generator().manual_seed(args.seed)
    context = (torch.randn(4, 2, 1500, generator=g).cumsum(-1) + 50).numpy()
    for horizon in (96, 2100):
        a = f_cpu.forecast(context, prediction_length=horizon).quantiles
        b = f_dev.forecast(context, prediction_length=horizon).quantiles
        print(f"[fwd]  forecast, horizon {horizon:<5d}        rel diff vs cpu: {rel(b, a):.2e}")
    target = context[:, :1]
    past = torch.randn(4, 2, 1500, generator=g).numpy()
    known = torch.randn(4, 1, 1596, generator=g).numpy()
    codes = torch.randint(0, 5, (4, 1, 1596), generator=g).double().numpy()
    kwargs = dict(past_covariates=past, known_covariates=known, categorical_known_covariates=codes)
    a = f_cpu.forecast(target, 96, **kwargs).quantiles
    b = f_dev.forecast(target, 96, **kwargs).quantiles
    print(f"[fwd]  forecast, numeric + categorical  rel diff vs cpu: {rel(b, a):.2e}")

    # Gradients of the real training loss against the float64-weights reference.
    batch = make_batch(2, 3, 32, 3, True, seed=args.seed)
    m_ref = copy.deepcopy(m_cpu).double().train()
    m_cpu.train()
    m_dev.train()
    results = {}
    for name, model, device, double in (
        ("ref", m_ref, "cpu", True),
        ("cpu", m_cpu, "cpu", False),
        (dev, m_dev, dev, False),
    ):
        model.zero_grad()
        loss = real_loss(model, batch, device, double=double)
        loss.backward()
        results[name] = (loss.item(), flat_grads(model))
    print(f"[grad] training loss: reference {results['ref'][0]:.7f} | cpu fp32 {results['cpu'][0]:.7f} | "
          f"{dev} fp32 {results[dev][0]:.7f}")  # fmt: skip
    gap_cpu, gap_dev = rel(results["cpu"][1], results["ref"][1]), rel(results[dev][1], results["ref"][1])
    print(f"[grad] gradient rel L2 vs float64-weights reference: cpu fp32 {gap_cpu:.2e} | {dev} fp32 {gap_dev:.2e}")
    cosine = torch.nn.functional.cosine_similarity(results["cpu"][1], results[dev][1], dim=0).item()
    print(f"[grad] cosine(cpu fp32, {dev} fp32) = {cosine:.8f}")

    # Autocast on the device.
    x, obs, gid, _, roles, kf, kfm = make_batch(2, 3, 32, 3, True, seed=args.seed)
    with torch.no_grad():
        a = m_cpu.predict_block(x, obs, gid, 3, roles, kf, kfm)
    for dtype in (torch.bfloat16, torch.float16):
        try:
            with torch.no_grad(), torch.autocast(dev, dtype=dtype):
                b = m_dev.predict_block(*on(dev, (x, obs, gid)), 3, *on(dev, (roles, kf, kfm)))
            m_dev.zero_grad()
            with torch.autocast(dev, dtype=dtype):
                out = m_dev.predict_block(*on(dev, (x, obs, gid)), 3, *on(dev, (roles, kf, kfm)))
            out.float().abs().mean().backward()
            finite = all(torch.isfinite(p.grad).all().item() for p in m_dev.parameters())
            gap = rel(b, a)
            print(f"[amp]  autocast {str(dtype):15s} forward rel diff vs cpu fp32: {gap:.2e} | grads finite: {finite}")
        except Exception as error:  # noqa: BLE001 - report whatever the backend raises
            print(f"[amp]  autocast {dtype} failed: {type(error).__name__}: {str(error)[:140]}")


if __name__ == "__main__":
    main()
