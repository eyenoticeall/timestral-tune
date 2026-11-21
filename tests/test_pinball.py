"""Unit tests for the pinball loss."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from timestral_tune.losses import PinballLoss, pinball_loss

from .conftest import available_accelerators

LEVELS = torch.tensor([0.1, 0.5, 0.9])


def test_matches_a_hand_computed_value():
    quantiles = torch.tensor([[1.0, 2.0, 4.0]])
    targets = torch.tensor([3.0])
    # errors: 2, 1, -1 -> 0.1*2, 0.5*1, (0.9-1)*(-1)
    expected = (0.2 + 0.5 + 0.1) / 3
    assert pinball_loss(quantiles, targets, LEVELS).item() == pytest.approx(expected)


def test_matches_numpy_on_random_data_with_a_mask():
    rng = np.random.default_rng(0)
    q = np.sort(rng.standard_normal((5, 7, 3)), axis=-1)
    z = rng.standard_normal((5, 7))
    w = rng.random((5, 7)) < 0.6
    error = z[..., None] - q
    tau = LEVELS.numpy().astype(np.float64)
    expected = (np.maximum(tau * error, (tau - 1) * error).mean(-1) * w).sum() / w.sum()
    actual = pinball_loss(torch.tensor(q), torch.tensor(z), LEVELS, torch.tensor(w))
    assert actual.dtype == torch.float64
    assert actual.item() == pytest.approx(expected, rel=1e-12)


def test_is_minimised_by_the_true_quantile():
    samples = torch.linspace(0, 1, 10_001)
    candidates = torch.linspace(0, 1, 101)
    level = torch.tensor([0.9])
    losses = torch.stack([pinball_loss(c.expand(samples.numel(), 1), samples, level) for c in candidates])
    assert candidates[losses.argmin()].item() == pytest.approx(0.9, abs=0.011)


def test_unscored_positions_contribute_nothing_even_when_not_finite():
    quantiles = torch.tensor([[0.0, 1.0, 2.0], [float("nan"), float("inf"), 3.0]], requires_grad=True)
    targets = torch.tensor([0.5, float("nan")])
    weight = torch.tensor([True, False])
    loss = pinball_loss(quantiles, targets, LEVELS, weight)
    loss.backward()
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(pinball_loss(quantiles[:1].detach(), targets[:1], LEVELS).item())
    assert (quantiles.grad[1] == 0).all()
    assert (quantiles.grad[0] != 0).all()


def test_no_scored_position_gives_exact_zero_and_zero_gradients():
    quantiles = torch.randn(4, 3, requires_grad=True)
    loss = pinball_loss(quantiles, torch.randn(4), LEVELS, torch.zeros(4, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0.0
    assert (quantiles.grad == 0).all()


def test_reduction_none_returns_per_position_losses():
    quantiles, targets = torch.randn(2, 5, 3), torch.randn(2, 5)
    weight = torch.rand(2, 5) < 0.5
    per_position = pinball_loss(quantiles, targets, LEVELS, weight, reduction="none")
    assert per_position.shape == (2, 5)
    assert (per_position[~weight] == 0).all()
    mean = pinball_loss(quantiles, targets, LEVELS, weight)
    assert mean.item() == pytest.approx(per_position.sum().item() / weight.sum().item(), rel=1e-6)


def test_low_precision_inputs_are_computed_in_float32():
    quantiles = torch.randn(3, 3, dtype=torch.bfloat16)
    assert pinball_loss(quantiles, torch.randn(3, dtype=torch.bfloat16), LEVELS).dtype == torch.float32


@pytest.mark.parametrize(
    ("quantiles", "targets", "weight", "match"),
    [
        (torch.zeros(2, 3), torch.zeros(3), None, "must match quantiles"),
        (torch.zeros(2, 4), torch.zeros(2), None, "levels on the last axis"),
        (torch.zeros(2, 3), torch.zeros(2), torch.zeros(3, dtype=torch.bool), "shaped like targets"),
        (torch.zeros(2, 3), torch.zeros(2), torch.ones(2), "bool mask"),
    ],
)
def test_rejects_malformed_inputs(quantiles, targets, weight, match):
    with pytest.raises(ValueError, match=match):
        pinball_loss(quantiles, targets, LEVELS, weight)


@pytest.mark.parametrize("levels", [[], [0.0, 0.5], [0.5, 1.0], [0.5, 0.5], [0.9, 0.1]])
def test_module_rejects_invalid_levels(levels):
    with pytest.raises(ValueError):
        PinballLoss(levels)


def test_module_keeps_levels_out_of_the_state_dict_and_matches_the_function():
    levels = [0.1, 0.5, 0.9]
    loss = PinballLoss(levels)
    assert loss.state_dict() == {} and loss.levels.dtype == torch.float64
    assert loss(torch.tensor([[1.0, 2.0, 4.0]]), torch.tensor([3.0])).item() == pytest.approx(0.8 / 3)
    # In float64 the module is bit-identical to the function given float64 levels: no rounding of the levels.
    generator = torch.Generator().manual_seed(0)
    quantiles = torch.randn(5, 7, 3, dtype=torch.float64, generator=generator).sort(-1).values
    targets = torch.randn(5, 7, dtype=torch.float64, generator=generator)
    reference = pinball_loss(quantiles, targets, torch.tensor(levels, dtype=torch.float64))
    assert torch.equal(loss(quantiles, targets), reference)


@pytest.mark.parametrize("device", available_accelerators())
def test_module_can_be_moved_to_an_accelerator(device):
    """The Apple GPU has no float64, so the module must not hold a float64 buffer."""
    loss = PinballLoss([0.1, 0.5, 0.9]).to(device)
    parent = torch.nn.ModuleDict({"loss": PinballLoss([0.1, 0.5, 0.9]), "head": torch.nn.Linear(2, 2)}).to(device)
    quantiles, targets = torch.tensor([[1.0, 2.0, 4.0]], device=device), torch.tensor([3.0], device=device)
    for module in (loss, parent["loss"]):
        value = module(quantiles, targets)
        assert value.device.type == device and value.item() == pytest.approx(0.8 / 3)
