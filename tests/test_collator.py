"""The collator against upstream's own input assembly, and the encoders against upstream's.

Ground truth is the vendored forecaster: `_assemble_rows`, `_encoding_target` and
`_encode_categorical`, called on the same raw arrays. Tensors are compared for exact equality.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from timestral_tune.data import DHFDataCollator, Window, encode_categorical_window, encoding_target, forward_fill
from timestral_tune.models import denormalise_quantiles, target_frames, training_forward

from .helpers import RawSeries, make_raw, max_rel, upstream_batch

EMPTY_FLOAT64 = np.empty((0, 0), dtype=np.float64)


def windows_from_raw(raw: RawSeries, codes_past=None, codes_known=None) -> list[Window]:
    """One full-context `Window` per series of `raw`."""
    series, _, context_len = raw.context.shape
    windows = []
    for i in range(series):
        windows.append(
            Window(
                series_index=i,
                origin=context_len,
                target_context=raw.context[i],
                target_future=raw.future[i],
                past=raw.past[i] if raw.past is not None else np.empty((0, context_len), dtype=np.float32),
                known=raw.known[i]
                if raw.known is not None
                else np.empty((0, context_len + raw.horizon), dtype=np.float32),
                categorical_past=codes_past[i]
                if codes_past is not None
                else np.empty((0, context_len), dtype=np.float64),
                categorical_known=codes_known[i]
                if codes_known is not None
                else np.empty((0, context_len + raw.horizon), dtype=np.float64),
            )
        )
    return windows


def collator_for(forecaster, context_length: int) -> DHFDataCollator:
    return DHFDataCollator.for_model(forecaster.config, context_length)


def assert_batches_equal(ours, expected) -> None:
    for name in (
        "context",
        "context_mask",
        "group_ids",
        "role_ids",
        "known_future",
        "known_future_mask",
        "targets",
        "loss_weight",
    ):
        a, b = getattr(ours, name), getattr(expected, name)
        assert (a is None) == (b is None), f"{name}: one side is None"
        if a is not None:
            assert a.dtype == b.dtype and a.shape == b.shape, (
                f"{name}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"
            )
            assert torch.equal(a, b), f"{name} differs"


def test_batch_equals_upstream_assembly(forecaster, raw):
    """Every tensor of the collated batch equals what upstream's `_assemble_rows` builds from the same arrays."""
    expected, _, _ = upstream_batch(forecaster, raw)
    ours = collator_for(forecaster, raw.context.shape[-1])(windows_from_raw(raw))
    ours.validate(forecaster.patch_size)
    assert_batches_equal(ours, expected)


def test_layout_details(tiny_forecaster):
    patch = tiny_forecaster.patch_size
    raw = make_raw(5, series=2, targets=2, past=1, known=1, context_len=3 * patch + 1, horizon=patch + 1)
    batch = collator_for(tiny_forecaster, raw.context.shape[-1])(windows_from_raw(raw))
    assert batch.context.shape == (8, 4 * patch) and batch.targets.shape == (8, 2 * patch)
    assert batch.group_ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert batch.role_ids.tolist() == [0, 0, 1, 2] * 2
    # Left padding is unobserved zeros; horizon padding and covariate rows are never scored.
    assert not batch.context_mask[:, : patch - 1].any() and (batch.context[:, : patch - 1] == 0).all()
    assert not batch.loss_weight[:, patch + 1 :].any()
    assert not batch.loss_weight[batch.role_ids != 0].any()
    assert not batch.known_future_mask[batch.role_ids != 2].any()
    assert batch.num_scored == int(np.isfinite(raw.future).sum())


def test_target_rows_are_forward_filled_and_covariates_zero_filled(tiny_forecaster):
    patch = tiny_forecaster.patch_size
    context = np.array([[[np.nan, 1.0, np.nan, np.nan, 5.0, np.nan, 7.0, 8.0]]], dtype=np.float32)
    past = np.array([[[np.nan, 2.0, np.nan, 4.0, np.nan, 6.0, 7.0, 8.0]]], dtype=np.float32)
    raw = RawSeries(context=context, future=np.ones((1, 1, patch), dtype=np.float32), past=past, known=None)
    batch = collator_for(tiny_forecaster, 8)(windows_from_raw(raw))
    assert batch.context[0].tolist() == [0.0, 1.0, 1.0, 1.0, 5.0, 5.0, 7.0, 8.0]
    assert batch.context_mask[0].tolist() == [False, True, False, False, True, False, True, True]
    assert batch.context[1].tolist() == [0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 7.0, 8.0]
    assert batch.known_future is None and batch.role_ids.tolist() == [0, 1]


@pytest.mark.parametrize("extra_patches", [1, 3])
@pytest.mark.parametrize("scenario", ["univariate", "covariates", "multi_target_covariates"], indirect=True)
def test_extra_left_padding_does_not_change_the_forecast(forecaster, raw, extra_patches):
    """Batches are padded to a common context length, further than upstream pads. Leading unobserved
    patches must leave the model's output unchanged, in both layouts.
    """
    model = forecaster.model
    context_len = raw.context.shape[-1]
    windows = windows_from_raw(raw)
    tight = collator_for(forecaster, context_len)(windows)
    padded = collator_for(forecaster, context_len + extra_patches * forecaster.patch_size)(windows)
    assert padded.context_steps == tight.context_steps + extra_patches * forecaster.patch_size
    with torch.no_grad():
        results = []
        for batch in (tight, padded):
            output = training_forward(model, batch)
            loc, scale = target_frames(model, batch, output)
            results.append(denormalise_quantiles(model, output.quantiles, loc, scale)[batch.target_rows])
    assert max_rel(results[1], results[0]) < 1e-5


def test_categorical_encoding_equals_upstream(tiny_forecaster):
    """Univariate target: encoded categorical rows equal upstream's `_encode_categorical` output."""
    forecaster = tiny_forecaster
    patch = forecaster.patch_size
    raw = make_raw(8, series=3, targets=1, past=1, known=1, context_len=7 * patch + 2, horizon=2 * patch, missing=0.15)
    raw.context[0, 0, :5] = np.nan  # a late-starting target
    rng = np.random.default_rng(8)
    context_len, horizon = raw.context.shape[-1], raw.horizon
    codes_past = rng.integers(0, 4, (3, 2, context_len)).astype(np.float64)
    codes_known = rng.integers(0, 5, (3, 1, context_len + horizon)).astype(np.float64)
    codes_past[rng.random(codes_past.shape) < 0.1] = np.nan
    codes_known[:, :, -3:] = 99.0  # a category never seen in the context
    codes_past[0, :, :3] = np.nan  # missing codes before the target starts: padding, not a category

    inputs, mask = forecaster._as_context(raw.context, None)
    target = forecaster._encoding_target(inputs, mask)
    expected_past = forecaster._encode_categorical(torch.as_tensor(codes_past), target)
    expected_known = forecaster._encode_categorical(torch.as_tensor(codes_known), target)

    for i in range(3):
        ours_target = encoding_target(raw.context[i, 0])
        assert np.array_equal(ours_target, target[i].double().numpy(), equal_nan=True)
        for row in range(2):
            encoded, future = encode_categorical_window(ours_target, codes_past[i, row])
            assert future is None
            assert np.array_equal(encoded, expected_past[i, row].numpy(), equal_nan=True)
        encoded, future = encode_categorical_window(
            ours_target, codes_known[i, 0, :context_len], codes_known[i, 0, context_len:]
        )
        assert np.array_equal(np.concatenate([encoded, future]), expected_known[i, 0].numpy(), equal_nan=True)

    # And the whole batch equals upstream's assembly with the encoded rows appended after the numeric ones.
    with_encoded = RawSeries(
        context=raw.context,
        future=raw.future,
        past=np.concatenate([raw.past, expected_past.numpy()], axis=1),
        known=np.concatenate([raw.known, expected_known.numpy()], axis=1),
    )
    expected_batch, _, _ = upstream_batch(forecaster, with_encoded)
    ours = collator_for(forecaster, context_len)(windows_from_raw(raw, codes_past, codes_known))
    assert_batches_equal(ours, expected_batch)


def test_several_targets_receive_raw_category_codes(tiny_forecaster):
    forecaster = tiny_forecaster
    patch = forecaster.patch_size
    raw = make_raw(9, series=2, targets=2, context_len=4 * patch, horizon=patch)
    codes_known = np.random.default_rng(9).integers(0, 6, (2, 1, 5 * patch)).astype(np.float64)
    batch = collator_for(forecaster, 4 * patch)(windows_from_raw(raw, None, codes_known))
    known_rows = batch.role_ids == 2
    assert torch.equal(batch.context[known_rows], torch.as_tensor(codes_known[:, 0, : 4 * patch], dtype=torch.float32))
    assert torch.equal(
        batch.known_future[known_rows], torch.as_tensor(codes_known[:, 0, 4 * patch :], dtype=torch.float32)
    )


def test_forward_fill():
    values = np.array([[np.nan, 1.0, np.nan, 3.0, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan]])
    filled = forward_fill(values)
    assert np.array_equal(filled, [[np.nan, 1.0, 1.0, 3.0, 3.0], [np.nan] * 5], equal_nan=True)
    assert filled.dtype == values.dtype and np.isnan(values[0, 2])  # input untouched


def test_collator_rejects_bad_input(tiny_forecaster):
    patch = tiny_forecaster.patch_size
    config = tiny_forecaster.config
    raw = make_raw(1, series=2, context_len=3 * patch, horizon=patch)
    windows = windows_from_raw(raw)
    collate = collator_for(tiny_forecaster, 3 * patch)

    with pytest.raises(ValueError, match="empty list"):
        collate([])
    with pytest.raises(ValueError, match="context cap"):
        DHFDataCollator.for_model(config, config.context_length + 1)
    with pytest.raises(ValueError, match="context steps"):
        collator_for(tiny_forecaster, 2 * patch)(windows)
    too_long = make_raw(1, series=1, context_len=3 * patch, horizon=config.block_size * patch + 1)
    with pytest.raises(ValueError, match="forecast horizon"):
        collate(windows_from_raw(too_long))
    other_layout = windows_from_raw(make_raw(1, series=1, targets=2, context_len=3 * patch, horizon=patch))
    with pytest.raises(ValueError, match="same number"):
        collate([windows[0], other_layout[0]])
    raw.context[0, 0, 2] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        collate(windows_from_raw(raw))
