"""Unit tests for raw-space evaluation metrics and forgetting measurement (spec §5.3, §5.5)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from timestral_tune import ConformalCalibrator, OnlineConformalCalibrator
from timestral_tune.models import DEFAULT_QUANTILE_LEVELS, TuneTimestralForecaster
from timestral_tune.utils import (
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
    # Put 0.10/0.90 at [9, 13] (width 4) and 0.05/0.95 at [8, 22] (width 14).
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
    # For diff >= 0, pinball(tau) = tau * diff; mean(levels) = 0.5 exactly by symmetry!
    expected_pinball = 4.0 * 0.5
    assert pinball_loss_raw(flat_preds, actuals) == pytest.approx(expected_pinball)
    # wQL = mean_q(2 * tau * 12 / 42) = 2 * 0.5 * 12 / 42 = 12 / 42.
    assert weighted_quantile_loss(flat_preds, actuals) == pytest.approx(12.0 / 42.0)

    # Works cleanly with PyTorch tensors as well.
    summary = evaluate_predictions(torch.tensor(preds), torch.tensor(actuals), torch.tensor(context))
    assert summary.num_series == 2 and summary.num_scored == 3
    assert summary.coverage_80 == pytest.approx(2.0 / 3.0)
    assert summary.mase == pytest.approx(1.25)


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


def test_evaluate_forecaster_conformal_and_forgetting(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size, series=4, covariates=True)
    m_base = evaluate_forecaster(tiny_forecaster, dataset, batch_size=3)
    assert m_base.num_series == len(dataset) and m_base.num_scored > 0
    assert m_base.pinball > 0 and m_base.wql > 0 and m_base.mase > 0
    assert 0.0 <= m_base.coverage_80 <= 1.0 and m_base.width_80 >= 0.0

    # Conformal calibrator plugs cleanly into evaluate_forecaster (static and online).
    from timestral_tune import collect_conformal_data

    preds, actuals, scale = collect_conformal_data(tiny_forecaster, dataset, batch_size=3)
    cal = ConformalCalibrator(target_coverage="all", bucket_size=tiny_forecaster.patch_size).fit(preds, actuals, scale)
    m_cal = evaluate_forecaster(tiny_forecaster, dataset, calibrator=cal, batch_size=3)
    assert m_cal.num_scored == m_base.num_scored
    assert m_cal.coverage_80 >= 0.70

    online = OnlineConformalCalibrator(
        target_coverage=0.80, gamma=0.05, window=20, bucket_size=tiny_forecaster.patch_size
    )
    m_online = evaluate_forecaster(tiny_forecaster, dataset, calibrator=online, online_update=True, batch_size=3)
    assert m_online.num_scored == m_base.num_scored and online.alpha_t.size > 0

    # Forgetting against identical model is identically zero; perturbing weights produces non-zero deltas.
    same = evaluate_forgetting(tiny_forecaster, tiny_forecaster, dataset, batch_size=3)
    assert same.delta_pinball == 0.0 and same.delta_mase == 0.0 and same.delta_wql == 0.0

    perturbed_model = copy.deepcopy(tiny_forecaster.model)
    with torch.no_grad():
        for p in perturbed_model.head.parameters():
            p.add_(0.5)
    perturbed_fc = TuneTimestralForecaster(perturbed_model)
    diff = evaluate_forgetting(perturbed_fc, tiny_forecaster, dataset, batch_size=3)
    assert diff.delta_pinball != 0.0 and diff.relative_delta_pinball == pytest.approx(
        diff.delta_pinball / diff.zero_shot.pinball
    )
    assert compute_forgetting(diff.tuned, diff.zero_shot).to_dict()["delta_mase"] == pytest.approx(diff.delta_mase)
