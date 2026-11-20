"""Forward and frame parity with upstream inference (spec §9, T1-T3).

The training forward runs the same operations as upstream's `predict_block`, so on CPU the
denormalised output is expected to be bit-identical, and is asserted to be. The frame tests go
through upstream's `_refine_block`, which renormalises raw-space values in float32, so they allow
float32 round-off and no more.
"""

from __future__ import annotations

import pytest
import torch

from timestral_tune.models import denormalise_quantiles, target_frames, training_forward

from ..conftest import SCENARIOS
from ..helpers import max_rel, upstream_batch

FRAME_RTOL = 1e-5  # measured 1e-7 to 2e-7; upstream's float32 renormalisation round-trip


def _predict_block(forecaster, batch):
    patch = forecaster.patch_size
    return forecaster.model.predict_block(
        batch.context,
        batch.context_mask,
        batch.group_ids,
        batch.horizon_steps // patch,
        batch.role_ids,
        batch.known_future,
        batch.known_future_mask,
    )


def _assert_forward_parity(forecaster, raw):
    model = forecaster.model
    batch, _, _ = upstream_batch(forecaster, raw)
    with torch.no_grad():
        output = training_forward(model, batch)
        ours = denormalise_quantiles(model, output.quantiles, output.anchor_loc, output.anchor_scale)
        reference = _predict_block(forecaster, batch)
    assert ours.shape == reference.shape
    assert torch.equal(ours, reference), f"max relative difference {max_rel(ours, reference):.2e}"
    assert (output.quantiles.diff(dim=-1) >= 0).all(), "normalised quantiles must be non-decreasing"


@pytest.mark.acceptance
@pytest.mark.parametrize("scenario", ["univariate", "multivariate"], indirect=True)
def test_t01_forward_parity_no_covariates(forecaster, raw, scenario):
    """T1. Passes when: `sinh(training_forward) * sigma_0 + mu_0` equals `predict_block` output within
    tolerance, no covariates.
    """
    assert SCENARIOS[scenario][1:] == (0, 0)
    _assert_forward_parity(forecaster, raw)


@pytest.mark.acceptance
@pytest.mark.parametrize("scenario", ["covariates", "multi_target_covariates"], indirect=True)
def test_t02_forward_parity_with_covariates(forecaster, raw, scenario):
    """T2. Passes when: Same as T1 for the `2n`-token layout with target, past-only and known-future rows."""
    assert all(count > 0 for count in SCENARIOS[scenario])
    _assert_forward_parity(forecaster, raw)


@pytest.mark.acceptance
def test_t03_frame_parity(forecaster, raw):
    """T3. Passes when: Applying the median-folded frame reproduces `_refine_block`, and the full pipeline
    matches `forecast(symmetric_averaging=False, clip_nonnegative=False)` on a forecaster built with
    `quantile_real_cap_k=0`.
    """
    assert forecaster.quantile_real_cap_k == 0
    model = forecaster.model
    batch, n_targets, (series, rows_per_series) = upstream_batch(forecaster, raw)
    target_rows = batch.target_rows
    with torch.no_grad():
        output = training_forward(model, batch)
        loc, scale = target_frames(model, batch, output, "median_folded")
        framed = denormalise_quantiles(model, output.quantiles, loc, scale)

        refined = forecaster._refine_block(
            _predict_block(forecaster, batch),
            batch.context,
            batch.context_mask,
            None if batch.role_ids is None else target_rows,
        )
        forecast = forecaster.forecast(
            raw.context,
            raw.horizon,
            symmetric_averaging=False,
            clip_nonnegative=False,
            **raw.forecast_kwargs(),
        ).quantiles

    assert max_rel(framed[target_rows], refined[target_rows]) < FRAME_RTOL
    ours = framed.reshape(series, rows_per_series, batch.horizon_steps, -1)[:, :n_targets, : raw.horizon]
    assert ours.shape == forecast.shape
    assert max_rel(ours, forecast) < FRAME_RTOL


@pytest.mark.acceptance
def test_t03_first_patch_frame_is_the_anchor(forecaster, raw):
    """Both frame definitions frame horizon patch 0 by the context alone, bit-identically to the forward's
    anchor; they may differ only from patch 1 on.
    """
    model = forecaster.model
    batch, _, _ = upstream_batch(forecaster, raw)
    with torch.no_grad():
        output = training_forward(model, batch)
        folded = target_frames(model, batch, output, "median_folded")
        true = target_frames(model, batch, output, "true_values")
    for loc, scale in (folded, true):
        assert loc.shape == (batch.num_rows, batch.horizon_steps // forecaster.patch_size)
        assert torch.equal(loc[:, 0], output.anchor_loc)
        assert torch.equal(scale[:, 0], output.anchor_scale)
        assert torch.isfinite(loc).all() and (scale > 0).all()
