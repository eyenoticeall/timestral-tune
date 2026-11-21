"""Unit tests for ConformalCalibrator (CQR) and OnlineConformalCalibrator (ACI) (spec §5.2, §7.3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from timestral_tune import ConformalCalibrator, OnlineConformalCalibrator
from timestral_tune.models import DEFAULT_QUANTILE_LEVELS

from .conftest import available_accelerators

DEVICES = ["cpu", *available_accelerators()]


def _make_simple(n: int = 40, h: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = np.linspace(-2.0, 2.0, len(DEFAULT_QUANTILE_LEVELS))[None, None, :]
    loc = rng.normal(0.0, 5.0, size=(n, h))
    scale = rng.uniform(0.5, 4.0, size=n)
    preds = loc[..., None] + scale[:, None, None] * base
    actuals = loc + rng.normal(0.0, 1.5, size=(n, h)) * scale[:, None]
    return preds, actuals, scale


def test_conformal_rejects_invalid_coverages_and_inputs():
    # Invalid coverages (not matching any symmetric pair of Timestral's 21 levels).
    for bad in (0.75, 0.95, 0.0, 1.0, "invalid", (), [0.80, 0.80], [True], 0.85):
        with pytest.raises(ValueError):
            ConformalCalibrator(target_coverage=bad)  # type: ignore[arg-type]

    # All ten symmetric coverages are valid.
    cal_all = ConformalCalibrator(target_coverage="all")
    assert cal_all.coverages == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98)

    preds, actuals, scale = _make_simple(10, 8)
    cal = ConformalCalibrator(target_coverage=0.80, bucket_size=4)
    with pytest.raises(RuntimeError, match="fitted"):
        cal.transform(preds, context_scale=scale)

    with pytest.raises(ValueError, match="positive"):
        cal.fit(preds, actuals, context_scale=np.zeros_like(scale))
    with pytest.raises(ValueError, match="shape"):
        cal.fit(preds, actuals[:, :-1], context_scale=scale)
    with pytest.raises(ValueError, match="finite"):
        bad_preds = preds.copy()
        bad_preds[0, 0, 0] = np.nan
        cal.fit(bad_preds, actuals, context_scale=scale)
    with pytest.raises(ValueError, match="infinities"):
        bad_y = actuals.copy()
        bad_y[0, 0] = np.inf
        cal.fit(preds, bad_y, context_scale=scale)

    # Horizon longer than fitted buckets raises.
    cal.fit(preds, actuals, context_scale=scale)
    longer_preds, _, _ = _make_simple(10, 16)
    with pytest.raises(ValueError, match="bucket"):
        cal.transform(longer_preds, context_scale=scale)


def test_cqr_scale_invariance_nan_handling_and_save_load(tmp_path):
    preds, actuals, scale = _make_simple(60, 16, seed=1)
    # Missing actuals (NaN) are ignored cleanly when computing bucket scores.
    actuals_with_nan = actuals.copy()
    actuals_with_nan[:5, :3] = np.nan

    cal1 = ConformalCalibrator(target_coverage=(0.80, 0.90), bucket_size=8).fit(preds, actuals_with_nan, scale)
    # Scaling predictions, actuals, and context_scale by 10x must leave scaled offsets identical.
    cal10 = ConformalCalibrator(target_coverage=(0.80, 0.90), bucket_size=8).fit(
        preds * 10.0, actuals_with_nan * 10.0, scale * 10.0
    )
    assert cal1.offsets_ is not None and cal10.offsets_ is not None
    np.testing.assert_allclose(cal1.offsets_, cal10.offsets_, rtol=1e-12, atol=1e-12)

    # Save / load round-trip reproduces transform output bit-identically.
    path = cal1.save(tmp_path / "conformal.json")
    reloaded = ConformalCalibrator.load(path)
    np.testing.assert_array_equal(cal1.transform(preds, scale), reloaded.transform(preds, scale))


def test_outward_monotonicity_repair_widens_only_and_keeps_median_fixed():
    """Calibrating a single narrow pair (0.50) by a large shift must push outer quantiles outward,
    never move the median, and leave intervals non-decreasing.
    """
    preds, actuals, scale = _make_simple(20, 4, seed=2)
    # Make actuals extreme so the 50% pair gets a huge positive shift exceeding the raw 90%/98% bounds.
    extreme_y = actuals + 50.0 * scale[:, None]
    cal = ConformalCalibrator(target_coverage=0.50, bucket_size=4).fit(preds, extreme_y, scale)
    out = cal.transform(preds, scale)

    m = DEFAULT_QUANTILE_LEVELS.index(0.50)
    assert np.array_equal(out[..., m], preds[..., m]), "median must remain unshifted"
    assert np.all(np.diff(out, axis=-1) >= 0.0), "quantiles must be non-decreasing"
    assert np.all(out[..., m:] >= preds[..., m:]), "outward repair must never shrink upper quantiles"
    assert np.all(out[..., : m + 1] <= preds[..., : m + 1]), "outward repair must never shrink lower quantiles"


@pytest.mark.parametrize("device", DEVICES)
def test_conformal_preserves_torch_tensor_device_and_dtype(device):
    preds, actuals, scale = _make_simple(30, 8, seed=3)
    t_preds = torch.tensor(preds, dtype=torch.float32, device=device)
    t_actuals = torch.tensor(actuals, dtype=torch.float32, device=device)
    t_scale = torch.tensor(scale, dtype=torch.float32, device=device)

    cal = ConformalCalibrator(target_coverage=0.80, bucket_size=4).fit(t_preds, t_actuals, t_scale)
    out = cal.transform(t_preds, t_scale)
    assert isinstance(out, torch.Tensor) and out.device.type == device and out.dtype == torch.float32
    assert (out.diff(dim=-1) >= 0).all()

    online = OnlineConformalCalibrator(target_coverage=0.80, bucket_size=4)
    online.update(t_preds, t_actuals, t_scale)
    out_on = online.transform(t_preds, t_scale)
    assert isinstance(out_on, torch.Tensor) and out_on.device.type == device and out_on.dtype == torch.float32


def test_aci_never_clips_alpha_t_sets_saturated_and_round_trips(tmp_path):
    preds, actuals, scale = _make_simple(15, 4, seed=4)
    # Escalating misses (each step exceeds all prior scores in the window) push alpha_t below 0 and set `.saturated`.
    escalating_y = actuals + (np.arange(1, 16)[:, None] * 100.0) * scale[:, None]
    online = OnlineConformalCalibrator(target_coverage=0.80, gamma=0.1, window=20, bucket_size=4)
    assert not online.saturated
    online.update(preds, escalating_y, scale)
    assert float(online.alpha_t[0, 0]) < 0.0, "ACI must not clip alpha_t at 0 (Lemma 4.1)"
    assert online.saturated, "saturated flag must be raised when alpha_t <= 0"

    # Persistent hits push alpha_t above 1 without clipping and use the smallest window score.
    online_hi = OnlineConformalCalibrator(target_coverage=0.20, gamma=0.2, window=20, bucket_size=4)
    median = preds[..., DEFAULT_QUANTILE_LEVELS.index(0.50)]
    online_hi.update(preds, median, scale)
    assert float(online_hi.alpha_t[0, 0]) > 1.0, "ACI must not clip alpha_t at 1 (Lemma 4.1)"
    assert not online_hi.saturated

    # Save / load round-trip preserves online state exactly.
    path = online.save(tmp_path / "aci.json")
    reloaded = OnlineConformalCalibrator.load(path)
    np.testing.assert_array_equal(online.alpha_t, reloaded.alpha_t)
    assert online.saturated == reloaded.saturated
    np.testing.assert_array_equal(online.transform(preds, scale), reloaded.transform(preds, scale))
