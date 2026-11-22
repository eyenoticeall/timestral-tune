"""Unit tests for raw-space evaluation metrics and forgetting measurement (spec §5.3, §5.5)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from timestral_tune import ConformalCalibrator, OnlineConformalCalibrator, collect_conformal_data
from timestral_tune.data import SeriesData, TimeSeriesDataset
from timestral_tune.models import DEFAULT_QUANTILE_LEVELS, TuneTimestralForecaster
from timestral_tune.utils import (
    MetricsSummary,
    compute_forgetting,
    coverage,
    evaluate_forecaster,
    evaluate_forgetting,
    evaluate_predictions,
    interval_width,
    mase,
    pinball_loss_raw,
    weighted_quantile_loss,
)

from .helpers import make_toy_dataset


def test_raw_metrics_exact_analytical_values_and_nan_masking():
    levels = DEFAULT_QUANTILE_LEVELS
    q_count = len(levels)
    # 2 series, horizon 2: actuals = [[10, 12], [20, NaN]]; constant predictions = 10 everywhere except bounds.
    actuals = np.array([[10.0, 12.0], [20.0, np.nan]])
    preds = np.full((2, 2, q_count), 10.0)
    # Give 80% and 90% strictly distinct coverages AND widths so swapping 80% <-> 90% always fails.
    lo80, hi80 = levels.index(0.10), levels.index(0.90)
    lo90, hi90 = levels.index(0.05), levels.index(0.95)
    preds[..., lo80] = 9.0
    preds[..., hi80] = 13.0
    preds[..., lo90] = 8.0
    preds[..., hi90] = 22.0

    # Coverage over the 3 finite actuals: 80% covers 10 and 12 (misses 20) -> 2/3; 90% covers all 3 -> 1.0.
    assert coverage(preds, actuals, 0.80) == pytest.approx(2.0 / 3.0)
    assert coverage(preds, actuals, 0.90) == pytest.approx(1.0)
    assert interval_width(preds, 0.80, actuals=actuals) == pytest.approx(4.0)
    assert interval_width(preds, 0.90, actuals=actuals) == pytest.approx(14.0)

    # MASE: context diffs have |diff| = 2 for series 0 (MAE=(0+2)/2=1 -> 0.5) and |diff|=5 for series 1 (MAE=10 -> 2.0).
    context = np.array([[0.0, 2.0, 4.0, 6.0], [0.0, 5.0, 10.0, 15.0]])
    assert mase(preds, actuals, context, seasonality=1) == pytest.approx(0.5 * (0.5 + 2.0))
    # With seasonality=2, context diffs double -> MASE halves.
    assert mase(preds, actuals, context, seasonality=2) == pytest.approx(0.5 * (0.25 + 1.0))

    # Pinball loss and wQL check on exact hand-computed values (uniform preds = 10 on all levels).
    flat_preds = np.full((2, 2, q_count), 10.0)
    # Finite residuals y - q: 0, +2, +10 -> sum of positive residuals = 12, mean residual = 4; sum(|y|) = 42.
    # For diff >= 0, pinball(tau) = tau * diff; mean(levels) = 0.5 exactly by symmetry.
    expected_pinball = 4.0 * 0.5
    assert pinball_loss_raw(flat_preds, actuals) == pytest.approx(expected_pinball)
    # wQL = mean_q(2 * tau * 12 / 42) = 2 * 0.5 * 12 / 42 = 12 / 42.
    assert weighted_quantile_loss(flat_preds, actuals) == pytest.approx(12.0 / 42.0)

    # Assert EVERY single field of MetricsSummary (catching 80%/90% swaps, pinball/wql/mase wiring bugs).
    summary = evaluate_predictions(torch.tensor(preds), torch.tensor(actuals), torch.tensor(context))
    assert summary.num_series == 2 and summary.num_scored == 3
    assert summary.pinball == pytest.approx(pinball_loss_raw(preds, actuals))
    assert summary.wql == pytest.approx(weighted_quantile_loss(preds, actuals))
    assert summary.mase == pytest.approx(1.25)
    assert summary.coverage_80 == pytest.approx(2.0 / 3.0)
    assert summary.width_80 == pytest.approx(4.0)
    assert summary.coverage_90 == pytest.approx(1.0)
    assert summary.width_90 == pytest.approx(14.0)


def test_metrics_validation_guards():
    q = len(DEFAULT_QUANTILE_LEVELS)
    preds = np.zeros((2, 4, q))
    y = np.ones((2, 4))
    ctx = np.arange(16, dtype=np.float64).reshape(2, 8)

    with pytest.raises(ValueError, match="symmetric pair"):
        coverage(preds, y, 0.75)
    with pytest.raises(ValueError, match="sum\\(\\|actuals\\|\\) == 0"):
        weighted_quantile_loss(preds, np.zeros((2, 4)))
    with pytest.raises(ValueError, match="no finite observations"):
        pinball_loss_raw(preds, np.full((2, 4), np.nan))
    with pytest.raises(ValueError, match="positive in-sample seasonal scale"):
        mase(preds, y, np.ones((2, 8)))
    with pytest.raises(ValueError, match="exceed seasonality"):
        mase(preds, y, ctx[:, :2], seasonality=2)


def test_forgetting_sign_convention_exact():
    zs = MetricsSummary(
        pinball=0.20,
        wql=0.10,
        mase=1.00,
        coverage_80=0.80,
        width_80=1.0,
        coverage_90=0.90,
        width_90=1.5,
        num_series=4,
        num_scored=64,
    )
    worse = MetricsSummary(
        pinball=0.50,
        wql=0.15,
        mase=1.40,
        coverage_80=0.70,
        width_80=1.2,
        coverage_90=0.82,
        width_90=1.8,
        num_series=4,
        num_scored=64,
    )
    better = MetricsSummary(
        pinball=0.10,
        wql=0.08,
        mase=0.80,
        coverage_80=0.81,
        width_80=0.9,
        coverage_90=0.91,
        width_90=1.4,
        num_series=4,
        num_scored=64,
    )
    # Degradation (forgetting) MUST be strictly positive; transfer gain MUST be strictly negative.
    f_worse = compute_forgetting(worse, zs)
    assert f_worse.delta_pinball == pytest.approx(+0.30)
    assert f_worse.relative_delta_pinball == pytest.approx(+1.50)
    assert f_worse.delta_wql == pytest.approx(+0.05)
    assert f_worse.relative_delta_wql == pytest.approx(+0.50)
    assert f_worse.delta_mase == pytest.approx(+0.40)
    assert f_worse.relative_delta_mase == pytest.approx(+0.40)

    f_better = compute_forgetting(better, zs)
    assert f_better.delta_pinball == pytest.approx(-0.10)
    assert f_better.delta_wql == pytest.approx(-0.02)
    assert f_better.delta_mase == pytest.approx(-0.20)


def test_evaluate_forecaster_window_context_alignment_and_conformal(tiny_forecaster):
    # Two series with 10,000x different scales so swapping or reversing contexts in MASE fails loudly.
    patch = tiny_forecaster.patch_size
    steps = 8 * patch
    rng = np.random.default_rng(7)
    s_small = (1.0 + np.cumsum(rng.normal(0.0, 0.01, size=(1, steps)), axis=-1)).astype(np.float32)
    s_large = (1000.0 + np.cumsum(rng.normal(0.0, 100.0, size=(1, steps)), axis=-1)).astype(np.float32)
    ds = TimeSeriesDataset(
        [
            SeriesData(
                series_id="small",
                timestamps=np.arange(steps, dtype=np.int64),
                targets=s_small,
                past=np.empty((0, steps), dtype=np.float32),
                known=np.empty((0, steps), dtype=np.float32),
                categorical_past=np.empty((0, steps), dtype=np.float64),
                categorical_known=np.empty((0, steps), dtype=np.float64),
            ),
            SeriesData(
                series_id="large",
                timestamps=np.arange(steps, dtype=np.int64),
                targets=s_large,
                past=np.empty((0, steps), dtype=np.float32),
                known=np.empty((0, steps), dtype=np.float32),
                categorical_past=np.empty((0, steps), dtype=np.float64),
                categorical_known=np.empty((0, steps), dtype=np.float64),
            ),
        ],
        context_length=4 * patch,
        forecast_horizon=2 * patch,
        stride=2 * patch,
        min_context=4 * patch,
    )
    preds, actuals, scale = collect_conformal_data(tiny_forecaster, ds, batch_size=2)
    # Compute exact hand-aligned MASE per window.
    med_i = list(tiny_forecaster.quantile_levels).index(0.5)
    expected_mases = []
    for i in range(len(ds)):
        ctx_i = ds[i].target_context[0].astype(np.float64)
        denom = float(np.mean(np.abs(np.diff(ctx_i))))
        mae_i = float(np.mean(np.abs(actuals[i] - preds[i, :, med_i])))
        expected_mases.append(mae_i / denom)
    m_base = evaluate_forecaster(tiny_forecaster, ds, batch_size=2)
    assert m_base.mase == pytest.approx(float(np.mean(expected_mases)), rel=1e-6)

    # Verify evaluate_forecaster actually forwards `seasonality` into mase() (catches ignoring seasonality).
    expected_mases_s2 = []
    for i in range(len(ds)):
        ctx_i = ds[i].target_context[0].astype(np.float64)
        denom_s2 = float(np.mean(np.abs(ctx_i[2:] - ctx_i[:-2])))
        mae_i = float(np.mean(np.abs(actuals[i] - preds[i, :, med_i])))
        expected_mases_s2.append(mae_i / denom_s2)
    m_s2 = evaluate_forecaster(tiny_forecaster, ds, seasonality=2, batch_size=2)
    assert m_s2.mase == pytest.approx(float(np.mean(expected_mases_s2)), rel=1e-6)
    assert m_s2.mase != pytest.approx(m_base.mase, rel=1e-3)

    # Static CQR calibrator actually changes evaluated forecasts to match cqr.transform(preds, scale).
    cal = ConformalCalibrator(target_coverage="all", bucket_size=patch).fit(preds, actuals, scale)
    m_cal = evaluate_forecaster(tiny_forecaster, ds, calibrator=cal, batch_size=2)
    expected_cal = evaluate_predictions(
        cal.transform(preds, context_scale=scale),
        actuals,
        np.stack([ds[i].target_context[0].astype(np.float64) for i in range(len(ds))], axis=0),
    )
    assert m_cal.width_80 == pytest.approx(expected_cal.width_80, rel=1e-6)
    assert m_cal.width_90 == pytest.approx(expected_cal.width_90, rel=1e-6)
    assert m_cal.width_80 != pytest.approx(m_base.width_80)

    # Forgetting against identical model is zero; adding large head shift degrades loss strictly (> 0).
    same = evaluate_forgetting(tiny_forecaster, tiny_forecaster, ds, batch_size=2)
    assert same.delta_pinball == 0.0 and same.delta_mase == 0.0 and same.delta_wql == 0.0

    perturbed_model = copy.deepcopy(tiny_forecaster.model)
    with torch.no_grad():
        for p in perturbed_model.head.parameters():
            p.add_(2.0)
    diff = evaluate_forgetting(TuneTimestralForecaster(perturbed_model), tiny_forecaster, ds, batch_size=2)
    assert diff.delta_pinball > 0.0 and diff.delta_wql > 0.0 and diff.delta_mase > 0.0


def test_evaluate_forecaster_online_aci_is_chronological_and_never_peeks(tiny_forecaster):
    """Online ACI must run strictly in origin-time order across series, never peek before transform,
    and delay updates until a prior window's full horizon has completed (`t_end < t_origin`).
    """
    # Build two series with STAGGERED timestamps so series 0 starts LATER than series 1:
    # dict insertion order of origins is then strictly non-chronological (catching both dataset-order
    # iteration AND dropping `sorted(by_origin)`).
    base_ds = make_toy_dataset(tiny_forecaster.patch_size, series=3, covariates=True)
    staggered_series = [
        SeriesData(
            series_id="late-0",
            timestamps=base_ds.series[0].timestamps + 1000,
            targets=base_ds.series[0].targets,
            past=base_ds.series[0].past,
            known=base_ds.series[0].known,
            categorical_past=base_ds.series[0].categorical_past,
            categorical_known=base_ds.series[0].categorical_known,
        ),
        SeriesData(
            series_id="early-1",
            timestamps=base_ds.series[1].timestamps,
            targets=base_ds.series[1].targets,
            past=base_ds.series[1].past,
            known=base_ds.series[1].known,
            categorical_past=base_ds.series[1].categorical_past,
            categorical_known=base_ds.series[1].categorical_known,
        ),
        SeriesData(
            series_id="shared-2",
            timestamps=base_ds.series[2].timestamps + 1000,
            targets=base_ds.series[2].targets,
            past=base_ds.series[2].past,
            known=base_ds.series[2].known,
            categorical_past=base_ds.series[2].categorical_past,
            categorical_known=base_ds.series[2].categorical_known,
        ),
    ]
    dataset = TimeSeriesDataset(
        staggered_series,
        context_length=base_ds.context_length,
        forecast_horizon=base_ds.forecast_horizon,
        stride=base_ds.stride,
        min_context=base_ds.context_length,
        roles=base_ds.roles,
    )
    origins = [
        int(dataset.series[dataset._series_index[i]].timestamps[dataset._origins[i]]) for i in range(len(dataset))
    ]
    unique_insertion = list(dict.fromkeys(origins))
    assert unique_insertion != sorted(unique_insertion), "test requires non-chronological first-seen origin order"

    preds_raw, actuals_raw, _ = collect_conformal_data(tiny_forecaster, dataset, batch_size=4)

    class _SpyOnline(OnlineConformalCalibrator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.events: list[tuple[str, int, float, np.ndarray]] = []

        def transform(self, predictions, context_scale=None):
            h = predictions.shape[1]
            off_sum = float(np.sum(np.abs(self.current_offsets(h)))) if self.alpha_t.size > 0 else 0.0
            self.events.append(("transform", int(predictions.shape[0]), off_sum, predictions.copy()))
            return super().transform(predictions, context_scale=context_scale)

        def update(self, predictions, actuals, context_scale=None):
            self.events.append(("update", int(predictions.shape[0]), float(np.mean(actuals)), actuals.copy()))
            return super().update(predictions, actuals, context_scale=context_scale)

    spy = _SpyOnline(target_coverage=0.80, gamma=0.05, window=50, bucket_size=tiny_forecaster.patch_size)
    evaluate_forecaster(tiny_forecaster, dataset, calibrator=spy, online_update=True, batch_size=4)

    # 1. Verify exact chronological grouping: each transform call receives the EXACT windows at
    #    that sorted origin timestamp across all 3 series (catches dataset-order processing!).
    unique_origins = sorted(set(origins))
    transforms = [ev for ev in spy.events if ev[0] == "transform"]
    updates = [ev for ev in spy.events if ev[0] == "update"]
    assert len(transforms) == len(unique_origins) and len(updates) == len(unique_origins)
    assert spy.events[0][0] == "transform" and spy.events[0][2] == 0.0
    for block_idx, orig in enumerate(unique_origins):
        idxs = [i for i, o in enumerate(origins) if o == orig]
        np.testing.assert_allclose(transforms[block_idx][3], preds_raw[idxs])
        np.testing.assert_allclose(updates[block_idx][3], actuals_raw[idxs])
        # For non-overlapping stride == horizon, events strictly alternate: transform(k), update(k).
        assert spy.events[2 * block_idx][0] == "transform"
        assert spy.events[2 * block_idx + 1][0] == "update"
    assert any(ev[2] > 0.0 for ev in transforms[1:])

    # 2. Overlapping stride (`stride < forecast_horizon`): verify `update` is delayed until
    #    `t_end < t_origin` (catches updating a window before its full horizon finishes!).
    patch = tiny_forecaster.patch_size
    horizon = 2 * patch
    stride = patch  # overlaps by `patch` steps!
    overlap_ds = TimeSeriesDataset(
        base_ds.series[:2],
        context_length=base_ds.context_length,
        forecast_horizon=horizon,
        stride=stride,
        min_context=base_ds.context_length,
        roles=base_ds.roles,
    )
    ov_origins = sorted(
        {
            int(overlap_ds.series[overlap_ds._series_index[i]].timestamps[overlap_ds._origins[i]])
            for i in range(len(overlap_ds))
        }
    )
    assert len(ov_origins) >= 3 and ov_origins[1] < ov_origins[0] + horizon - 1
    spy_ov = _SpyOnline(target_coverage=0.80, gamma=0.05, window=50, bucket_size=patch)
    evaluate_forecaster(tiny_forecaster, overlap_ds, calibrator=spy_ov, online_update=True, batch_size=4)
    # Origin 0 has t_end = orig0 + horizon - 1 >= orig1, so origin 1 MUST be transformed BEFORE any update!
    assert spy_ov.events[0][0] == "transform" and spy_ov.events[0][2] == 0.0
    assert spy_ov.events[1][0] == "transform" and spy_ov.events[1][2] == 0.0, (
        f"overlapping window updated before its horizon finished: events={[e[:3] for e in spy_ov.events[:4]]}"
    )
    # Origin 2 (`orig0 + 2*patch`) is strictly after `t_end` (`orig0 + 2*patch - 1`), so origin 0's
    # update fires right before origin 2's transform!
    assert spy_ov.events[2][0] == "update" and spy_ov.events[3][0] == "transform" and spy_ov.events[3][2] > 0.0
