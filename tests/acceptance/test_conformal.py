"""Conformal calibration (spec §9, T13)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from timestral_tune import ConformalCalibrator, OnlineConformalCalibrator
from timestral_tune.models import DEFAULT_QUANTILE_LEVELS


def _normal_z(levels: tuple[float, ...]) -> np.ndarray:
    return torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(levels, dtype=torch.float64)).numpy()


def _synthetic_exchangeable(
    n: int,
    horizon: int,
    *,
    width_factor: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic heteroscedastic series with systematically miscalibrated quantile widths.

    Each series has a random positive `context_scale` spanning two orders of magnitude and
    horizon-dependent noise (`sigma_h` grows across horizon buckets). Predicted quantiles are
    scaled by `width_factor` (e.g. 0.45 = severely under-covering, 2.2 = severely over-covering).
    """
    rng = np.random.default_rng(seed)
    z = _normal_z(DEFAULT_QUANTILE_LEVELS)
    scale = rng.lognormal(mean=0.0, sigma=1.2, size=n)
    loc = rng.normal(0.0, 50.0, size=(n, horizon))
    horizon_growth = np.linspace(0.6, 1.8, horizon)[None, :]
    true_sigma = scale[:, None] * horizon_growth
    actuals = loc + rng.normal(0.0, 1.0, size=(n, horizon)) * true_sigma
    pred_sigma = true_sigma * width_factor
    predictions = loc[..., None] + pred_sigma[..., None] * z[None, None, :]
    return predictions, actuals, scale


@pytest.mark.acceptance
@pytest.mark.parametrize("width_factor", [0.45, 2.2], ids=["under_covering", "over_covering"])
def test_t13_conformal_coverage_and_monotonicity(width_factor):
    """T13. Passes when: On synthetic exchangeable data, calibrated coverage meets the nominal level;
    output quantiles are non-decreasing.
    """
    horizon = 64
    cal_preds, cal_y, cal_scale = _synthetic_exchangeable(400, horizon, width_factor=width_factor, seed=13)
    test_preds, test_y, test_scale = _synthetic_exchangeable(400, horizon, width_factor=width_factor, seed=14)

    # Confirm the raw 80% interval is genuinely miscalibrated before conformal calibration.
    low_80, high_80 = DEFAULT_QUANTILE_LEVELS.index(0.10), DEFAULT_QUANTILE_LEVELS.index(0.90)
    raw_cov = float(np.mean((test_y >= test_preds[..., low_80]) & (test_y <= test_preds[..., high_80])))
    if width_factor < 1.0:
        assert raw_cov < 0.60, f"expected raw under-coverage, got {raw_cov:.3f}"
    else:
        assert raw_cov > 0.95, f"expected raw over-coverage, got {raw_cov:.3f}"

    # Calibrate all ten symmetric pairs simultaneously (and verify median is never shifted).
    calibrator = ConformalCalibrator(target_coverage="all", bucket_size=32)
    calibrator.fit(cal_preds, cal_y, context_scale=cal_scale)
    calibrated = calibrator.transform(test_preds, context_scale=test_scale)

    assert calibrated.shape == test_preds.shape
    assert np.all(np.diff(calibrated, axis=-1) >= 0.0), "calibrated quantiles must be non-decreasing"
    median_idx = DEFAULT_QUANTILE_LEVELS.index(0.50)
    assert np.array_equal(calibrated[..., median_idx], test_preds[..., median_idx]), "median must not shift"

    # Check empirical coverage across key nominal pairs (within finite-sample margin on N=400).
    for cov, low_idx, high_idx in calibrator.pairs:
        empirical = float(np.mean((test_y >= calibrated[..., low_idx]) & (test_y <= calibrated[..., high_idx])))
        assert empirical >= cov - 0.025, f"coverage {cov:.2f}: empirical {empirical:.3f} below nominal"
        assert empirical <= min(1.0, cov + 0.045), f"coverage {cov:.2f}: empirical {empirical:.3f} too conservative"

    # Over-covering raw forecasts must yield negative CQR offsets (intervals narrow to nominal).
    if width_factor > 1.0:
        assert calibrator.offsets_ is not None and (calibrator.offsets_ < 0.0).all()


@pytest.mark.acceptance
def test_t13_online_conformal_tracks_coverage_under_shift():
    """T13 (online ACI). Under a sudden variance shift, `OnlineConformalCalibrator` adapts `alpha_t`
    without clipping, tracks nominal coverage in the long run, and outputs non-decreasing quantiles.
    """
    n_pre, n_post, horizon = 120, 480, 32
    p1, y1, s1 = _synthetic_exchangeable(n_pre, horizon, width_factor=1.0, seed=101)
    p2, y2, s2 = _synthetic_exchangeable(n_post, horizon, width_factor=0.35, seed=102)
    preds = np.concatenate([p1, p2], axis=0)
    actuals = np.concatenate([y1, y2], axis=0)
    scales = np.concatenate([s1, s2], axis=0)

    online = OnlineConformalCalibrator(target_coverage=0.80, gamma=0.02, window=200, bucket_size=32)
    low_80, high_80 = DEFAULT_QUANTILE_LEVELS.index(0.10), DEFAULT_QUANTILE_LEVELS.index(0.90)

    covered: list[float] = []
    for i in range(preds.shape[0]):
        cal_i = online.transform(preds[i : i + 1], context_scale=scales[i : i + 1])
        assert np.all(np.diff(cal_i, axis=-1) >= 0.0), "online calibrated quantiles must be non-decreasing"
        if i >= n_pre + 80:  # after adaptation transient
            covered.append(float(np.mean((actuals[i] >= cal_i[0, :, low_80]) & (actuals[i] <= cal_i[0, :, high_80]))))
        online.update(preds[i : i + 1], actuals[i : i + 1], context_scale=scales[i : i + 1])

    post_shift_coverage = float(np.mean(covered))
    assert 0.74 <= post_shift_coverage <= 0.86, f"post-shift ACI coverage {post_shift_coverage:.3f} drifted from 0.80"
