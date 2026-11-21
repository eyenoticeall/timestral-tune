"""Tests for the reusable verifiers themselves: that they run on every device, skip when they do not apply,
and fail when the property they check is broken.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from timestral_tune.models import forward as forward_module
from timestral_tune.utils import (
    CheckResult,
    check_firewall,
    check_forward_parity,
    check_known_future_sensitivity,
    check_label_isolation,
    check_loss_mask,
    run_checks,
)

from .conftest import available_accelerators
from .helpers import forecaster_on, make_raw, upstream_batch

DEVICES = ["cpu", *available_accelerators()]


def covariate_batch(forecaster):
    patch = forecaster.patch_size
    raw = make_raw(31, targets=1, past=1, known=1, context_len=6 * patch + 1, horizon=2 * patch + 1)
    return upstream_batch(forecaster, raw)[0]


def univariate_batch(forecaster):
    patch = forecaster.patch_size
    return upstream_batch(forecaster, make_raw(32, context_len=6 * patch + 1, horizon=2 * patch + 1))[0]


@pytest.mark.parametrize("device", DEVICES)
def test_every_check_passes_on_every_device(tiny_forecaster, device):
    forecaster = tiny_forecaster if device == "cpu" else forecaster_on(tiny_forecaster, device)
    results = run_checks(forecaster.model, covariate_batch(tiny_forecaster).to(device))
    assert len(results) == 6
    assert all(isinstance(result, CheckResult) for result in results)
    assert all(result.passed and not result.skipped for result in results), "\n".join(map(str, results))
    assert all(str(result).startswith("[PASS]") for result in results)


def test_checks_that_do_not_apply_are_skipped_not_passed(tiny_forecaster):
    model, batch = tiny_forecaster.model, univariate_batch(tiny_forecaster)
    for result in (check_firewall(model, batch), check_known_future_sensitivity(model, batch)):
        assert result.skipped and not result.passed and str(result).startswith("[SKIP]")
    unscored = dataclasses.replace(batch, loss_weight=torch.zeros_like(batch.loss_weight))
    assert check_loss_mask(model, unscored).skipped
    assert check_forward_parity(model, batch).passed and check_label_isolation(model, batch).passed


def test_label_isolation_fails_when_labels_reach_the_inputs(tiny_forecaster, monkeypatch):
    """A forward that reads the horizon targets must be caught."""
    model, batch = tiny_forecaster.model, univariate_batch(tiny_forecaster)
    real = forward_module.training_forward

    def leaky(model, batch, **kwargs):
        peeked = dataclasses.replace(batch, context=batch.context + batch.targets.mean(dim=1, keepdim=True))
        return real(model, peeked, **kwargs)

    monkeypatch.setattr("timestral_tune.utils.validation.training_forward", leaky)
    result = check_label_isolation(model, batch)
    assert not result.passed and str(result).startswith("[FAIL]")


def test_firewall_fails_when_roles_are_wrong(tiny_forecaster, monkeypatch):
    """With the role firewall removed (every row treated as a target), covariate rows do read the targets."""
    model, batch = tiny_forecaster.model, covariate_batch(tiny_forecaster)
    real = forward_module.training_forward

    def no_firewall(model, batch, **kwargs):
        return real(
            model, dataclasses.replace(batch, role_ids=None, known_future=None, known_future_mask=None), **kwargs
        )

    monkeypatch.setattr("timestral_tune.utils.validation.training_forward", no_firewall)
    result = check_firewall(model, batch)
    assert not result.skipped and not result.passed


def test_forward_parity_fails_on_a_wrong_forward(tiny_forecaster, monkeypatch):
    model, batch = tiny_forecaster.model, univariate_batch(tiny_forecaster)
    real = forward_module.training_forward

    def shifted(model, batch, **kwargs):
        output = real(model, batch, **kwargs)
        return dataclasses.replace(output, quantiles=output.quantiles + 1e-3)

    monkeypatch.setattr("timestral_tune.utils.validation.training_forward", shifted)
    assert not check_forward_parity(model, batch).passed


@pytest.mark.parametrize("trainable", ["nothing", "head", "everything"])
def test_checks_run_whatever_is_frozen_and_leave_the_flags_alone(tiny_forecaster, trainable):
    """`check_loss_mask` differentiates with respect to the predictions, so it needs no trainable parameter
    and must not change which parameters are trainable.
    """
    import copy

    model = copy.deepcopy(tiny_forecaster.model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(trainable == "everything" or (trainable == "head" and name.startswith("head.")))
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    results = run_checks(model, covariate_batch(tiny_forecaster))
    assert len(results) == 6 and all(result.passed for result in results), "\n".join(map(str, results))
    assert flags == {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert all(parameter.grad is None for parameter in model.parameters())
