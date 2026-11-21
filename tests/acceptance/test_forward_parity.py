"""Forward and frame parity with upstream inference (spec §9, T1-T3).

The training forward runs the same operations as upstream's `predict_block`, so the denormalised
output must agree to float32 round-off. Measured: bit-identical on PyTorch 2.4.0 and 2.15.0; one
tiny-model case differs by 8e-10 on 2.8.0, because the head is applied to a different number of
tokens in the two code paths. The frame tests go through upstream's `_refine_block`, which
renormalises raw-space values in float32; measured 1e-7 to 2e-7.
"""

from __future__ import annotations

import pytest
import torch

from timestral_tune.models import denormalise_quantiles, target_frames, training_forward

from ..conftest import SCENARIOS
from ..helpers import max_rel, upstream_batch

FORWARD_RTOL = 1e-6
FRAME_RTOL = 1e-5


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
    assert max_rel(ours, reference) < FORWARD_RTOL
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
@pytest.mark.parametrize("scenario", ["past_only_covariates"], indirect=True)
def test_t02_forward_parity_with_past_only_covariates(forecaster, raw, scenario):
    """T2, the common case of covariates none of which are known in advance: the `2n`-token layout with no
    known-future values at all (`known_future` is None).
    """
    batch, _, _ = upstream_batch(forecaster, raw)
    assert batch.role_ids is not None and batch.known_future is None and batch.known_future_mask is None
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


@pytest.mark.acceptance
def test_t03_true_values_frame_is_the_causal_frame_of_the_true_series(forecaster, raw):
    """The `true_values` frame of horizon patch j is the frame an origin j patches later would use: the
    causal statistics of the context followed by the true values of horizon patches 0..j-1, and nothing
    from patch j on.

    The reference is computed a different way from `target_frames`: by extending the context with the true
    values and asking upstream's `model.scale` for the frame of the last patch.
    """
    model, patch = forecaster.model, forecaster.patch_size
    batch, _, _ = upstream_batch(forecaster, raw)
    horizon_patches = batch.horizon_steps // patch
    assert horizon_patches >= 3, "the scenario must have patches beyond the first"
    with torch.no_grad():
        output = training_forward(model, batch)
        loc, scale = target_frames(model, batch, output, "true_values")
        folded_loc, _ = target_frames(model, batch, output, "median_folded")
    rows = batch.target_rows
    for j in range(1, horizon_patches):
        steps = j * patch
        extended = torch.cat([batch.context, batch.targets[:, :steps]], dim=-1)
        observed = torch.cat([batch.context_mask, batch.loss_weight[:, :steps]], dim=-1)
        _, reference_loc, reference_scale = model.scale(extended, observed)
        assert max_rel(loc[rows, j], reference_loc[rows, -1]) < FORWARD_RTOL, f"location of patch {j}"
        assert max_rel(scale[rows, j], reference_scale[rows, -1]) < FORWARD_RTOL, f"scale of patch {j}"
    # The two frame definitions are genuinely different from patch 1 on, so the check above is not vacuous.
    assert not torch.allclose(loc[rows, 1:], folded_loc[rows, 1:], rtol=1e-3)
