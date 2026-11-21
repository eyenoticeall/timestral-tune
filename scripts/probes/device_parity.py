"""Probe: does an accelerator reproduce CPU forecasts and accurate gradients? (spec §5.4, T14)

    uv run python scripts/probes/device_parity.py --device mps            # 5M
    uv run python scripts/probes/device_parity.py --device mps --size 28M
    uv run python scripts/probes/device_parity.py --device cuda

Prints forward parity against CPU, gradients of the training loss against a float64-weights
reference (for CPU float32 and for the device), and what autocast does on the device.

The same checks run as acceptance test T14; this probe prints the numbers behind them.
Expected output is recorded in docs/VERIFICATION.md.
"""

from __future__ import annotations

import argparse

import torch
from _common import SIZES, load, loss_gradient, make_batch, on_device, reference_gradient, rel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", required=True, choices=["mps", "cuda"])
    parser.add_argument("--size", default="5M", choices=SIZES)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    dev = args.device
    print(f"== Timestral-{args.size} on {dev} | torch {torch.__version__} ==")

    f_cpu = load(args.size, "cpu")
    f_dev = on_device(f_cpu, dev)
    m_cpu, m_dev = f_cpu.model, f_dev.model

    def predict(model, batch):
        return model.predict_block(
            batch.context,
            batch.context_mask,
            batch.group_ids,
            batch.horizon_steps // 32,
            batch.role_ids,
            batch.known_future,
            batch.known_future_mask,
        )

    # Forward parity: predict_block in both layouts.
    for label, covariates in (("no covariates", False), ("with covariates", True)):
        batch = make_batch(2, 3, 32, 3, covariates, seed=args.seed)
        with torch.no_grad():
            expected, actual = predict(m_cpu, batch), predict(m_dev, batch.to(dev))
        print(f"[fwd]  predict_block, {label:16s} rel diff vs cpu: {rel(actual, expected):.2e}")

    # Forward parity: the full forecaster, single block and block-autoregressive.
    g = torch.Generator().manual_seed(args.seed)
    context = (torch.randn(4, 2, 1500, generator=g).cumsum(-1) + 50).numpy()
    for horizon in (96, 2100):
        expected = f_cpu.forecast(context, prediction_length=horizon).quantiles
        actual = f_dev.forecast(context, prediction_length=horizon).quantiles
        print(f"[fwd]  forecast, horizon {horizon:<5d}        rel diff vs cpu: {rel(actual, expected):.2e}")
    target = context[:, :1]
    kwargs = dict(
        past_covariates=torch.randn(4, 2, 1500, generator=g).numpy(),
        known_covariates=torch.randn(4, 1, 1596, generator=g).numpy(),
        categorical_known_covariates=torch.randint(0, 5, (4, 1, 1596), generator=g).double().numpy(),
    )
    expected = f_cpu.forecast(target, 96, **kwargs).quantiles
    actual = f_dev.forecast(target, 96, **kwargs).quantiles
    print(f"[fwd]  forecast, numeric + categorical  rel diff vs cpu: {rel(actual, expected):.2e}")

    # Gradients of the training loss against the float64-weights reference.
    batch = make_batch(2, 3, 32, 3, True, seed=args.seed)
    reference_loss, reference = reference_gradient(m_cpu, batch)
    cpu_loss, cpu_gradient = loss_gradient(m_cpu, batch)
    dev_loss, dev_gradient = loss_gradient(m_dev, batch.to(dev))
    print(f"[grad] training loss: reference {reference_loss:.7f} | cpu fp32 {cpu_loss:.7f} | {dev} fp32 {dev_loss:.7f}")
    print(f"[grad] gradient rel L2 vs float64-weights reference: cpu fp32 {rel(cpu_gradient, reference):.2e} | "
          f"{dev} fp32 {rel(dev_gradient, reference):.2e}")  # fmt: skip

    # Autocast on the device.
    with torch.no_grad():
        expected = predict(m_cpu, batch)
    for dtype in (torch.bfloat16, torch.float16):
        try:
            with torch.no_grad(), torch.autocast(dev, dtype=dtype):
                actual = predict(m_dev, batch.to(dev))
            print(f"[amp]  autocast {str(dtype):15s} forward rel diff vs cpu fp32: {rel(actual, expected):.2e}")
        except Exception as error:  # noqa: BLE001 - report whatever the backend raises
            print(f"[amp]  autocast {dtype} failed: {type(error).__name__}: {str(error)[:140]}")


if __name__ == "__main__":
    main()
