"""Device parity (spec §9, T14).

Forecasts on an accelerator are compared with CPU. Gradients are compared with a float64 reference
on CPU, on every device including CPU float32. Over all parameters there are four checks:

- Tiny random model: gradients agree with the reference to about 4e-7 on every device and under
  every attention kernel. Held to 1e-5. This is the structural check.
- Released 5M and 28M checkpoints as shipped, with PyTorch's default attention kernel: off by up
  to 3.4e-3 on CPU, 2.1e-3 on the Apple GPU and 1.1e-2 on CUDA, because the float32 backward pass
  of the fused kernel loses accuracy at these checkpoints' attention logits of 200k to 500k
  (spec §12, docs/VERIFICATION.md §6, §13). Held to 5e-2 relative and 1e-4 on one minus the
  cosine, which still catches a wrong gradient.
- Released checkpoints with attention written out as matmul and softmax, and
- released checkpoints with PyTorch's unfused kernel (`attention_context("math")`, experimental,
  spec D15): for nine of the ten cases, at most 6.1e-4 on CPU (Apple and x86) and CUDA, and 1.2e-3
  on the Apple GPU. Held to 5e-3.
  These show the library's own forward, frames, loss and device handling are accurate on real
  weights.

The tenth case, 5M with past-only covariates, is ill-conditioned in float32 whatever the kernel:
one query has two keys 3.46 apart at a logit of 351k, where one float32 step is 0.03, so the
softmax weights depend on how each kernel happens to round. Which kernel does worst differs by
machine: explicit attention gives 1.2e-3 on an Apple CPU, 5.3e-3 on an x86 CPU and 1.9e-3 on CUDA;
the unfused kernel 9.0e-3, 1.4e-3 and 1.4e-3. That case alone is held to 2e-2 under both kernels
(`NEAR_TIE_CASES`), instead of loosening the bound for all.

With everything but the head frozen (`head_only`, the one mode the trainer implements so far), the
backward pass never enters attention or the body's norms, and gradients match the reference within
1e-5 on released weights too (measured at most 2.8e-6 on CPU, the Apple GPU and CUDA).

The reference uses float64 weights and inputs, and computes upstream's RMSNorm in float64
(`helpers.reference_gradient`). Upstream's RMSNorm always computes in float32, which made an
earlier reference itself inaccurate by up to 3.8e-3. RoPE still builds its table in float32, so the
reference is not a pure float64 computation.
"""

from __future__ import annotations

import copy

import pytest
import torch

import timestral_tune._vendor.timestral.layers as upstream_layers
from timestral_tune.trainer import compute_loss

from ..conftest import available_accelerators
from ..helpers import (
    as_double,
    explicit_attention,
    forecaster_on,
    l2_rel,
    loss_gradient,
    make_raw,
    max_rel,
    reference_gradient,
    upstream_batch,
)

FORECAST_RTOL = 1e-4
GRADIENT_RTOL_RANDOM_WEIGHTS = 1e-5
GRADIENT_RTOL_RELEASED_WEIGHTS = 5e-2
GRADIENT_RTOL_UNFUSED_ATTENTION = 5e-3
GRADIENT_RTOL_NEAR_TIE = 2e-2
# (model, scenario) pairs whose batch has a near-tie between large attention logits; see the module docstring.
NEAR_TIE_CASES = {("5M", "past_only_covariates")}
GRADIENT_COSINE_RTOL = 1e-4

ACCELERATORS = available_accelerators()


@pytest.fixture
def unfused_bound(request, is_released_checkpoint) -> float:
    """The gradient bound for this model and scenario when attention runs unfused (explicitly or via PyTorch)."""
    if not is_released_checkpoint:
        return GRADIENT_RTOL_RANDOM_WEIGHTS
    params = request.node.callspec.params
    near_tie = (params["forecaster"], params["scenario"]) in NEAR_TIE_CASES
    return GRADIENT_RTOL_NEAR_TIE if near_tie else GRADIENT_RTOL_UNFUSED_ATTENTION


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
    reference_loss, reference = reference_gradient(forecaster.model, batch)

    on_device = forecaster if device == "cpu" else forecaster_on(forecaster, device)
    loss, gradient = loss_gradient(on_device.model, batch.to(device))

    assert loss == pytest.approx(reference_loss, rel=1e-4)
    assert torch.isfinite(gradient).all()
    bound = GRADIENT_RTOL_RELEASED_WEIGHTS if is_released_checkpoint else GRADIENT_RTOL_RANDOM_WEIGHTS
    relative = l2_rel(gradient, reference)
    cosine = torch.nn.functional.cosine_similarity(gradient, reference, dim=0).item()
    print(f"gradient on {device}: relative L2 error {relative:.2e}, 1 - cosine {1 - cosine:.1e}")  # `pytest -s`
    if device == "cpu":
        # Why the reference computes RMSNorm in float64: how far it moves if upstream's float32 one is kept.
        _, float32_norm = loss_gradient(copy.deepcopy(forecaster.model).double(), as_double(batch))
        print(f"reference with upstream's float32 RMSNorm: relative L2 distance {l2_rel(float32_norm, reference):.2e}")
    assert relative < bound, f"relative L2 {relative:.2e}"
    assert 1 - cosine < GRADIENT_COSINE_RTOL, f"1 - cosine {1 - cosine:.2e}"


@pytest.mark.acceptance
@pytest.mark.parametrize("device", ["cpu", *ACCELERATORS])
def test_t14_gradients_with_explicit_attention(forecaster, raw, device, unfused_bound, monkeypatch):
    """T14 (gradients, explicit attention). Passes when: with attention written out as matmul and softmax,
    float32 gradients match the reference within 5e-3 on released checkpoints (2e-2 on the one near-tie
    case, 1e-5 on the tiny random model), on every device.

    Together with the test above this locates the float32 inaccuracy in the backward pass of upstream's
    attention call, and shows the rest of the training path is accurate on real weights.
    """
    batch, _, _ = upstream_batch(forecaster, raw)
    _, reference = reference_gradient(forecaster.model, batch)

    monkeypatch.setattr(upstream_layers, "_sdpa", explicit_attention)
    on_device = forecaster if device == "cpu" else forecaster_on(forecaster, device)
    _, gradient = loss_gradient(on_device.model, batch.to(device))
    relative = l2_rel(gradient, reference)
    print(f"gradient on {device}, explicit attention: relative L2 error {relative:.2e}")  # `pytest -s`
    assert relative < unfused_bound, f"relative L2 {relative:.2e}"


@pytest.mark.acceptance
@pytest.mark.parametrize("device", ["cpu", *ACCELERATORS])
def test_t14_gradients_with_math_sdpa(forecaster, raw, device, unfused_bound):
    """T14 (gradients, PyTorch's unfused kernel). Passes when: with `attention_context("math")`, float32
    gradients match the reference within 5e-3 relative and 1e-4 on one minus the cosine on released
    checkpoints (2e-2 on the one near-tie case, 1e-5 on the tiny random model), on every device.

    The kernel switch is experimental (spec D15). This is the same bound as the explicit-attention test.
    """
    batch, _, _ = upstream_batch(forecaster, raw)
    _, reference = reference_gradient(forecaster.model, batch)

    on_device = forecaster if device == "cpu" else forecaster_on(forecaster, device)
    _, gradient = loss_gradient(on_device.model, batch.to(device), attention_backend="math")
    relative = l2_rel(gradient, reference)
    cosine = torch.nn.functional.cosine_similarity(gradient, reference, dim=0).item()
    print(
        f"gradient on {device}, math SDPA: relative L2 error {relative:.2e}, 1 - cosine {1 - cosine:.1e}"
    )  # `pytest -s`
    assert relative < unfused_bound, f"relative L2 {relative:.2e}"
    assert 1 - cosine < GRADIENT_COSINE_RTOL, f"1 - cosine {1 - cosine:.2e}"


@pytest.mark.acceptance
@pytest.mark.parametrize("device", ["cpu", *ACCELERATORS])
def test_t14_head_only_gradients(forecaster, raw, device):
    """T14 (gradients, `head_only`). Passes when: with everything but the head frozen, float32 gradients
    match the float64-weights reference within 1e-5 on every model and device.

    With the body frozen, the backward pass stops at the head and never enters attention, so the
    inaccuracy of the tiers above does not reach the one mode the trainer implements so far.
    """

    def head_gradient(model, batch):
        model.zero_grad()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
        compute_loss(model, batch).loss.backward()
        return 0.0, torch.cat([p.grad.detach().cpu().double().flatten() for p in model.head.parameters()])

    batch, _, _ = upstream_batch(forecaster, raw)
    _, reference = reference_gradient(forecaster.model, batch, head_gradient)
    model = copy.deepcopy(forecaster.model) if device == "cpu" else forecaster_on(forecaster, device).model
    relative = l2_rel(head_gradient(model, batch.to(device))[1], reference)
    print(f"head-only gradient on {device}: relative L2 error {relative:.2e}")  # `pytest -s`
    assert relative < GRADIENT_RTOL_RANDOM_WEIGHTS, f"relative L2 {relative:.2e}"
