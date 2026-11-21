"""Leakage and point-in-time tests on the training path (spec §9, T4-T7 and T12).

T4, T5 and T12 go through the whole data path: a long-format frame, the dataset, the collator and
the training forward. Each builds two frames that differ only in values a forecast made at the
window's origin must not see, and asserts that what the model is given, and what it predicts, is
identical. T6 and T7 perturb a collated batch directly.

Perturbations are non-constant noise: a constant offset is absorbed by the normalisation and would
make a firewall test pass vacuously.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from timestral_tune.data import TimeSeriesDataset, TrainingBatch, encode_categorical_window, encoding_target
from timestral_tune.models import training_forward
from timestral_tune.utils import (
    check_firewall,
    check_forward_parity,
    check_known_future_sensitivity,
    check_label_isolation,
)

from ..helpers import make_columns, role_kwargs, to_frame, upstream_batch

INPUT_FIELDS = ("context", "context_mask", "group_ids", "role_ids", "known_future", "known_future_mask")
ROLES = {"targets": 1, "past": 1, "known": 1, "categorical_past": 1, "categorical_known": 1}


def steps_for(forecaster) -> int:
    """Series length, sized to the model's patch so that the last window has a full, unaligned context."""
    return 12 * forecaster.patch_size + 3


def last_window_batch(forecaster, columns) -> tuple[TrainingBatch, TimeSeriesDataset]:
    """The batch of each series' last window: origin at `steps - horizon`, so the frame's final `horizon`
    rows of every series are exactly the window's horizon.
    """
    patch = forecaster.patch_size
    horizon = 2 * patch + 1
    steps = steps_for(forecaster)
    dataset = TimeSeriesDataset.from_frame(
        to_frame(columns, "pandas"),
        context_length=9 * patch + 1,
        forecast_horizon=horizon,
        stride=10_000,
        **role_kwargs(columns),
    )
    assert all(dataset[i].origin == steps - horizon for i in range(len(dataset)))
    assert all(dataset[i].context_steps == 9 * patch + 1 for i in range(len(dataset)))
    return dataset.collator(forecaster.config)([dataset[i] for i in range(len(dataset))]), dataset


def after_origin(forecaster, columns) -> np.ndarray:
    """Row mask: the last `horizon` steps of every series, i.e. everything at or after the last origin."""
    steps, horizon = steps_for(forecaster), 2 * forecaster.patch_size + 1
    step_in_series = np.arange(columns["time"].shape[0]) % steps
    return step_in_series >= steps - horizon


def inputs_equal(a: TrainingBatch, b: TrainingBatch) -> bool:
    for name in INPUT_FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        if (x is None) != (y is None) or (x is not None and not torch.equal(x, y)):
            return False
    return True


@pytest.fixture
def columns(forecaster):
    return make_columns(21, series=3, steps=steps_for(forecaster), **ROLES)


@pytest.mark.acceptance
def test_t04_label_isolation(forecaster, columns):
    """T4. Passes when: Replacing horizon target values with noise leaves predictions bit-identical."""
    changed = dict(columns, y0=columns["y0"].copy())
    rows = after_origin(forecaster, columns)
    changed["y0"][rows] = np.random.default_rng(4).standard_normal(int(rows.sum())) * 50

    batch, _ = last_window_batch(forecaster, columns)
    other, _ = last_window_batch(forecaster, changed)
    assert not torch.equal(batch.targets, other.targets), "the labels were meant to change"
    assert inputs_equal(batch, other), "horizon target values reached the model's inputs"
    with torch.no_grad():
        assert torch.equal(
            training_forward(forecaster.model, batch).quantiles, training_forward(forecaster.model, other).quantiles
        )
    # The same property, checked by the reusable verifier directly on the batch.
    result = check_label_isolation(forecaster.model, batch)
    assert result.passed, str(result)


@pytest.mark.acceptance
def test_t05_past_covariate_cut(forecaster, columns):
    """T5. Passes when: Changing a past-only covariate after the origin (in the raw frame) changes nothing."""
    rows = after_origin(forecaster, columns)
    rng = np.random.default_rng(5)
    changed = dict(columns, p0=columns["p0"].copy(), cp0=columns["cp0"].copy())
    changed["p0"][rows] = rng.standard_normal(int(rows.sum())) * 50
    changed["cp0"][rows] = "d"

    batch, _ = last_window_batch(forecaster, columns)
    other, _ = last_window_batch(forecaster, changed)
    assert inputs_equal(batch, other), "a past-only covariate value from after the origin reached the inputs"
    assert torch.equal(batch.targets, other.targets) and torch.equal(batch.loss_weight, other.loss_weight)


@pytest.mark.acceptance
@pytest.mark.parametrize("scenario", ["covariates", "multi_target_covariates"], indirect=True)
def test_t06_firewall(forecaster, raw):
    """T6. Passes when: Noise on a target's context changes no covariate row's hidden state or output."""
    batch, _, _ = upstream_batch(forecaster, raw)
    result = check_firewall(forecaster.model, batch)
    assert not result.skipped
    assert result.passed, str(result)


@pytest.mark.acceptance
@pytest.mark.parametrize("scenario", ["covariates", "multi_target_covariates"], indirect=True)
def test_t07_known_future_positive_control(forecaster, raw):
    """T7. Passes when: Noise on a known-future covariate in the horizon does change the target forecast."""
    batch, _, _ = upstream_batch(forecaster, raw)
    result = check_known_future_sensitivity(forecaster.model, batch)
    assert not result.skipped
    assert result.passed, str(result)


@pytest.mark.acceptance
def test_t12_encoding_causality(forecaster, columns):
    """T12. Passes when: A window's categorical encodings are unchanged when data after its origin changes."""
    rows = after_origin(forecaster, columns)
    rng = np.random.default_rng(12)
    changed = dict(columns, y0=columns["y0"].copy(), cp0=columns["cp0"].copy())
    changed["y0"][rows] = rng.standard_normal(int(rows.sum())) * 500 + 1000
    changed["cp0"][rows] = "a"

    batch, dataset = last_window_batch(forecaster, columns)
    other, _ = last_window_batch(forecaster, changed)
    # Rows per series: target, numeric past, encoded categorical past, numeric known, encoded categorical known.
    categorical_rows = torch.zeros(batch.num_rows, dtype=torch.bool)
    categorical_rows[2::5] = True
    categorical_rows[4::5] = True
    assert batch.role_ids.tolist()[:5] == [0, 1, 1, 2, 2]
    for name in ("context", "context_mask", "known_future", "known_future_mask"):
        assert torch.equal(getattr(batch, name)[categorical_rows], getattr(other, name)[categorical_rows]), name
    assert batch.context_mask[categorical_rows].any() and batch.known_future_mask[4::5].any()

    # Positive control: the test can see a leak. An encoding fitted on context *and* horizon targets, which
    # is what encoding a whole series at once would do, gives different numbers.
    window = dataset[0]
    leaky_target = encoding_target(np.concatenate([window.target_context[0], window.target_future[0]]))
    leaky, _ = encode_categorical_window(leaky_target, window.categorical_known[0])
    honest, honest_future = encode_categorical_window(
        encoding_target(window.target_context[0]),
        window.categorical_known[0, : window.context_steps],
        window.categorical_known[0, window.context_steps :],
    )
    assert not np.allclose(leaky[: window.context_steps], honest, equal_nan=True)
    assert np.isfinite(honest_future).all(), "horizon codes are looked up in the context's table"


@pytest.mark.acceptance
def test_collated_batches_pass_forward_parity(forecaster, columns):
    """The batch built by the dataset and collator, not only upstream's own assembly, reproduces `predict_block`."""
    batch, _ = last_window_batch(forecaster, columns)
    result = check_forward_parity(forecaster.model, batch)
    assert result.passed, str(result)
