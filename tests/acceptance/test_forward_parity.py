"""Forward and frame parity with upstream inference (spec §9, T1-T3).

Without covariates (T1), the training forward runs the same operations as upstream's `predict_block`
and agrees to 1e-6 relative. With covariates (T2), the training forward drops the `context_patches`
inactive decoder tokens from upstream's `2n`-token layout (they attend only to themselves and are
discarded before the head). That is mathematically exact (`<= 1e-12` relative in float64) and agrees
with upstream's `2n`-token `predict_block` within 1e-5 in float32 (measured 6e-8 to 5.2e-6 from
matrix-tiling order). The frame tests (T3) go through upstream's `_refine_block`, which renormalises
raw-space values in float32; measured 1e-7 to 2e-7.
"""

from __future__ import annotations

import copy

import pytest
import torch

from timestral_tune._vendor.timestral.masks import build_two_stream_time_mask, compute_enc_key_validity
from timestral_tune._vendor.timestral.model import ROLE_KNOWN_COV
from timestral_tune.models import denormalise_quantiles, target_frames, training_forward
from timestral_tune.models.forward import (
    _build_covariate_time_mask,
    advancing_block_frames_from_state,
    context_welford_state,
)

from ..conftest import SCENARIOS
from ..helpers import as_double, float64_reference, make_raw, max_rel, upstream_batch

FORWARD_RTOL = 1e-6
COVARIATE_FORWARD_RTOL = 1e-5
FLOAT64_FORWARD_RTOL = 1e-12
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


def _assert_forward_parity(forecaster, raw, *, rtol: float = FORWARD_RTOL, check_float64: bool = False):
    model = forecaster.model
    batch, _, _ = upstream_batch(forecaster, raw)
    with torch.no_grad():
        output = training_forward(model, batch)
        ours = denormalise_quantiles(model, output.quantiles, output.anchor_loc, output.anchor_scale)
        reference = _predict_block(forecaster, batch)
    assert ours.shape == reference.shape
    assert max_rel(ours, reference) < rtol
    assert (output.quantiles.diff(dim=-1) >= 0).all(), "normalised quantiles must be non-decreasing"

    if check_float64:
        model64 = copy.deepcopy(model).double()
        batch64 = as_double(batch)
        with torch.no_grad(), float64_reference():
            out64 = training_forward(model64, batch64)
            ours64 = denormalise_quantiles(model64, out64.quantiles, out64.anchor_loc, out64.anchor_scale)
            ref64 = model64.predict_block(
                batch64.context,
                batch64.context_mask,
                batch64.group_ids,
                batch64.horizon_steps // forecaster.patch_size,
                batch64.role_ids,
                batch64.known_future,
                batch64.known_future_mask,
            )
        assert max_rel(ours64, ref64) < FLOAT64_FORWARD_RTOL


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
    """T2. Passes when: Same as T1 for the covariate layout with target, past-only and known-future rows,
    within 1e-5 in float32 and 1e-12 in float64.
    """
    assert all(count > 0 for count in SCENARIOS[scenario])
    _assert_forward_parity(forecaster, raw, rtol=COVARIATE_FORWARD_RTOL, check_float64=True)


@pytest.mark.acceptance
@pytest.mark.parametrize("scenario", ["past_only_covariates"], indirect=True)
def test_t02_forward_parity_with_past_only_covariates(forecaster, raw, scenario):
    """T2, the common case of covariates none of which are known in advance: the covariate layout with no
    known-future values at all (`known_future` is None).
    """
    batch, _, _ = upstream_batch(forecaster, raw)
    assert batch.role_ids is not None and batch.known_future is None and batch.known_future_mask is None
    _assert_forward_parity(forecaster, raw, rtol=COVARIATE_FORWARD_RTOL, check_float64=True)


@pytest.mark.acceptance
@pytest.mark.parametrize("scenario", ["covariates", "past_only_covariates"], indirect=True)
def test_t02_trimmed_covariate_mask_equals_sliced_upstream_mask(forecaster, raw, scenario):
    """Our trimmed `(wp + hp, wp + hp)` covariate time mask is bit-identical to slicing upstream's
    `build_two_stream_time_mask` to the kept tokens, and every dropped decoder token in upstream's full
    mask has zero incoming or outgoing edges to any kept token. Guards against upstream mask drift.
    """
    patch = forecaster.patch_size
    batch, _, _ = upstream_batch(forecaster, raw)
    cp, hp = batch.context_steps // patch, batch.horizon_steps // patch
    wp = cp + hp
    known_rows = batch.role_ids == ROLE_KNOWN_COV
    has_known = batch.known_future is not None
    kfm = batch.known_future_mask if has_known else torch.zeros(batch.num_rows, batch.horizon_steps, dtype=torch.bool)
    window_mask = torch.cat([batch.context_mask, kfm], dim=-1)
    obs_per_patch = window_mask.unflatten(-1, (wp, patch)).any(dim=-1)
    key_valid = compute_enc_key_validity(obs_per_patch, forecaster.config.attend_missing_patches)

    block_start = torch.full((batch.num_rows,), cp, dtype=torch.long)
    full = build_two_stream_time_mask(
        block_start, wp, forecaster.config.block_size, key_valid, bidirectional_rows=known_rows
    )
    trimmed = _build_covariate_time_mask(cp, hp, key_valid, known_rows, has_known=has_known)

    keep = torch.cat([torch.arange(wp), torch.arange(wp + cp, 2 * wp)])
    dropped = torch.arange(wp, wp + cp)
    assert torch.equal(trimmed, full[:, :, keep][:, :, :, keep])
    assert not full[:, :, keep][:, :, :, dropped].any(), "a kept token read a dropped decoder token"
    assert not full[:, :, dropped][:, :, :, keep].any(), "a dropped decoder token read a kept token"


def test_target_frames_single_patch_and_welford_state_parity(tiny_forecaster):
    """Single-patch horizons short-circuit to the anchor, and `advancing_block_frames_from_state` matches
    upstream's `advancing_block_frames` to float64 round-off (bit-identical after float32 cast).
    """
    from timestral_tune._vendor.timestral.scaler import advancing_block_frames

    patch = tiny_forecaster.patch_size
    raw1 = make_raw(41, targets=1, past=1, known=1, context_len=5 * patch, horizon=patch)
    batch1, _, _ = upstream_batch(tiny_forecaster, raw1)
    with torch.no_grad():
        out1 = training_forward(tiny_forecaster.model, batch1)
        for frame in ("median_folded", "true_values"):
            loc, scale = target_frames(tiny_forecaster.model, batch1, out1, frame)
            assert torch.equal(loc, out1.anchor_loc[:, None]) and torch.equal(scale, out1.anchor_scale[:, None])

    raw3 = make_raw(42, targets=2, past=1, known=1, context_len=7 * patch, horizon=3 * patch)
    batch3, _, _ = upstream_batch(tiny_forecaster, raw3)
    with torch.no_grad():
        out3 = training_forward(tiny_forecaster.model, batch3)
        median = out3.quantiles[..., tiny_forecaster.config.median_index].double().sinh().unflatten(-1, (3, patch))
        expected_loc, expected_scale = advancing_block_frames(
            batch3.context, batch3.context_mask, median, tiny_forecaster.config.minimum_scale
        )
        cnt, mean, m2 = context_welford_state(batch3.context, batch3.context_mask, patch)
        actual_loc, actual_scale = advancing_block_frames_from_state(
            cnt, mean, m2, median, tiny_forecaster.config.minimum_scale
        )
    assert max_rel(actual_loc, expected_loc) < 1e-14 and max_rel(actual_scale, expected_scale) < 1e-14
    assert torch.equal(actual_loc.float(), expected_loc.float())
    assert torch.equal(actual_scale.float(), expected_scale.float())


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
