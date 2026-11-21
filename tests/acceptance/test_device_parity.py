"""Device parity (spec §9, T14).

Forecasts on an accelerator are compared with CPU. Gradients are compared with a float64-weights
reference on CPU, on every device including CPU float32, in three tiers:

- Tiny random model: CPU, Apple GPU and CUDA gradients agree with the reference to about 3e-7.
  Held to 1e-5. This is the structural check.
- Released 5M and 28M checkpoints, attention written out as matmul and softmax: at most 1.1e-4
  on every device, except one scenario (28M with covariates: 1.1e-4 on CUDA, 5.4e-4 on CPU,
  1.2e-3 on the Apple GPU), which is unexplained. Held to 5e-3. This shows the library's own
  forward, frames, loss and device handling are accurate on real weights.
- Released checkpoints as shipped, with upstream's `F.scaled_dot_product_attention`: off by up to
  3.4e-3 on CPU, 1.1e-3 on the Apple GPU and 1.2e-2 on CUDA. The float32 backward pass of that
  call loses accuracy with these weights (spec §12, docs/VERIFICATION.md §6). Held to 5e-2, which
  still catches a wrong gradient; one minus the cosine similarity stays below 1e-4.

The reference uses float64 weights and inputs; upstream's RMSNorm and RoPE still compute in float32
internally, so it is not a pure float64 computation.
"""

from __future__ import annotations

import copy

import pytest
import torch

import timestral_tune._vendor.timestral.layers as upstream_layers

from ..conftest import available_accelerators
from ..helpers import (
    as_double,
    explicit_attention,
    forecaster_on,
    l2_rel,
    loss_gradient,
    make_raw,
    max_rel,
    upstream_batch,
)

FORECAST_RTOL = 1e-4
GRADIENT_RTOL_RANDOM_WEIGHTS = 1e-5
GRADIENT_RTOL_RELEASED_WEIGHTS = 5e-2
GRADIENT_RTOL_EXPLICIT_ATTENTION = 5e-3

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
    float64-weights reference within 1e-5 relative on the tiny random model, and within 5e-2 relative
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


@pytest.mark.acceptance
@pytest.mark.parametrize("device", ["cpu", *ACCELERATORS])
def test_t14_gradients_with_explicit_attention(forecaster, raw, device, is_released_checkpoint, monkeypatch):
    """T14 (gradients, explicit attention). Passes when: with attention written out as matmul and softmax,
    float32 gradients match the reference within 5e-3 on released checkpoints (1e-5 on the tiny random
    model), on every device.

    Together with the test above this locates the float32 inaccuracy in the backward pass of upstream's
    attention call, and shows the rest of the training path is accurate on real weights.
    """
    batch, _, _ = upstream_batch(forecaster, raw)
    _, reference = loss_gradient(copy.deepcopy(forecaster.model).double(), as_double(batch))

    monkeypatch.setattr(upstream_layers, "_sdpa", explicit_attention)
    on_device = forecaster if device == "cpu" else forecaster_on(forecaster, device)
    _, gradient = loss_gradient(on_device.model, batch.to(device))
    bound = GRADIENT_RTOL_EXPLICIT_ATTENTION if is_released_checkpoint else GRADIENT_RTOL_RANDOM_WEIGHTS
    relative = l2_rel(gradient, reference)
    print(f"gradient on {device}, explicit attention: relative L2 error {relative:.2e}")  # `pytest -s`
    assert relative < bound, f"relative L2 {relative:.2e}"
