"""Device parity (spec §9, T14).

Forecasts on an accelerator are compared with CPU. Gradients are compared with a float64-weights
reference on CPU, on every device including CPU float32, because float32 backward passes through
the released checkpoints are not accurate to float32 round-off on any device:

- On the tiny random model, CPU and Apple GPU gradients agree with the reference to about 3e-7.
  That is the structural check, and it is held to 1e-5.
- On the released 5M and 28M checkpoints the same code is off by up to 3.4e-3 (CPU) and 1.1e-3
  (Apple GPU); one minus the cosine similarity is at most 5.7e-6 and 6.4e-7. The cause inside the
  network is not identified (spec §12, docs/VERIFICATION.md §6). These are held to a looser bound
  that still catches a wrong gradient: relative L2 below 1e-2, which implies cosine similarity
  above 0.99995.

The reference uses float64 weights and inputs; upstream's RMSNorm and RoPE still compute in float32
internally, so it is not a pure float64 computation.
"""

from __future__ import annotations

import copy

import pytest
import torch

from ..conftest import available_accelerators
from ..helpers import as_double, forecaster_on, l2_rel, loss_gradient, make_raw, max_rel, upstream_batch

FORECAST_RTOL = 1e-4
GRADIENT_RTOL_RANDOM_WEIGHTS = 1e-5
GRADIENT_RTOL_RELEASED_WEIGHTS = 1e-2

ACCELERATORS = available_accelerators()
accelerator = pytest.mark.parametrize(
    "device",
    ACCELERATORS or [pytest.param("none", marks=pytest.mark.skip(reason="no CUDA or Apple GPU device available"))],
)


@pytest.mark.acceptance
@accelerator
def test_t14_device_parity_forecasts(forecaster, raw, device):
    """T14 (forecasts). Passes when: on each available accelerator, forecasts match CPU within 1e-4
    relative. Skipped where no accelerator is present.
    """
    expected = forecaster.forecast(raw.context, raw.horizon, **raw.forecast_kwargs()).quantiles
    actual = forecaster_on(forecaster, device).forecast(raw.context, raw.horizon, **raw.forecast_kwargs()).quantiles
    assert actual.device.type == device
    assert max_rel(actual, expected) < FORECAST_RTOL


@pytest.mark.acceptance
@accelerator
def test_t14_device_parity_block_autoregressive(forecaster, device):
    """T14 (long horizons). A horizon of several decode blocks feeds medians back as context, so device
    differences could compound; they must stay within the same 1e-4.
    """
    patch, block = forecaster.patch_size, forecaster.config.block_size
    raw = make_raw(11, series=2, context_len=6 * patch + 3, horizon=2 * block * patch + patch)
    expected = forecaster.forecast(raw.context, raw.horizon).quantiles
    actual = forecaster_on(forecaster, device).forecast(raw.context, raw.horizon).quantiles
    assert max_rel(actual, expected) < FORECAST_RTOL


@pytest.mark.acceptance
@pytest.mark.parametrize("device", ["cpu", *ACCELERATORS])
def test_t14_device_parity_gradients(forecaster, raw, device, is_released_checkpoint):
    """T14 (gradients). Passes when: on every device, float32 training-loss gradients match a
    float64-weights reference within 1e-5 relative on the tiny random model, and within 1e-2 relative
    on released checkpoints.
    """
    batch, _, _ = upstream_batch(forecaster, raw)
    reference_loss, reference = loss_gradient(copy.deepcopy(forecaster.model).double(), as_double(batch))

    on_device = forecaster if device == "cpu" else forecaster_on(forecaster, device)
    loss, gradient = loss_gradient(on_device.model, batch.to(device))

    assert loss == pytest.approx(reference_loss, rel=1e-4)
    assert torch.isfinite(gradient).all()
    bound = GRADIENT_RTOL_RELEASED_WEIGHTS if is_released_checkpoint else GRADIENT_RTOL_RANDOM_WEIGHTS
    relative = l2_rel(gradient, reference)
    cosine = torch.nn.functional.cosine_similarity(gradient, reference, dim=0).item()
    print(f"gradient on {device}: relative L2 error {relative:.2e}, 1 - cosine {1 - cosine:.1e}")  # `pytest -s`
    assert relative < bound, f"relative L2 {relative:.2e}"
