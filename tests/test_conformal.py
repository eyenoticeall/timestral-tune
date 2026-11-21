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


def test_cqr_exact_order_statistic_and_edge_cases():
    """Verify exact CQR eq. 10 order statistic `ceil((n + 1) * cov)` (e.g. n=9, cov=0.8 -> 8th of 9),
    plus `n_valid=1`, `bucket_size > H`, `bucket_size=1`, `H=0`, `N=0`, and all-NaN bucket guards.
    """
    # Construct 9 calibration points with known scaled scores 1..9 for the 0.80 pair.
    n, h = 9, 1
    preds = np.zeros((n, h, len(DEFAULT_QUANTILE_LEVELS)), dtype=np.float64)
    actuals = np.arange(1.0, 10.0, dtype=np.float64)[:, None]  # scores = 1, 2, ..., 9
    cal = ConformalCalibrator(target_coverage=0.80, bucket_size=1).fit(preds, actuals)
    # ceil((9 + 1) * 0.8) = 8 -> exact 8th smallest score = 8.0 (not 9.0).
    assert cal.offsets_ is not None and cal.offsets_[0, 0] == pytest.approx(8.0)

    # n_valid = 1 hits the finite-sample rank > n upper bound -> max(scores).
    cal1 = ConformalCalibrator(target_coverage=0.80, bucket_size=32).fit(preds[:1], actuals[:1])
    assert cal1.offsets_ is not None and cal1.offsets_[0, 0] == pytest.approx(1.0)

    # Empty N=0 or H=0 raises ValueError cleanly (never IndexError).
    empty_h = np.zeros((4, 0, len(DEFAULT_QUANTILE_LEVELS)))
    empty_y = np.zeros((4, 0))
    with pytest.raises(ValueError, match="non-zero"):
        ConformalCalibrator(0.80).fit(empty_h, empty_y)
    with pytest.raises(ValueError, match="non-zero"):
        cal.transform(empty_h)
    with pytest.raises(ValueError, match="non-zero"):
        OnlineConformalCalibrator(0.80).transform(empty_h)

    # All-NaN bucket raises ValueError cleanly.
    nan_y = np.full((5, 4), np.nan)
    with pytest.raises(ValueError, match="no finite actuals"):
        ConformalCalibrator(0.80, bucket_size=4).fit(np.zeros((5, 4, len(DEFAULT_QUANTILE_LEVELS))), nan_y)


def test_from_dict_rejects_corrupted_cqr_and_aci_payloads():
    preds, actuals, scale = _make_simple(12, 8, seed=7)
    cqr_data = ConformalCalibrator(0.80, bucket_size=4).fit(preds, actuals, scale).to_dict()

    for mutate in (
        lambda d: d.update(offsets=[[float("nan"), 0.1]]),
        lambda d: d.update(offsets=[[0.1, 0.2], [0.3, 0.4]]),  # wrong pair count
        lambda d: d.update(offsets=[[]]),  # 0 buckets
        lambda d: d.update(bucket_counts=[1]),  # mismatched bucket count
    ):
        bad = dict(cqr_data)
        mutate(bad)
        with pytest.raises(ValueError):
            ConformalCalibrator.from_dict(bad)

    aci = OnlineConformalCalibrator(0.80, bucket_size=4).update(preds, actuals, scale)
    aci_data = aci.to_dict()
    for mutate in (
        lambda d: d.update(alpha_t=[[0.2]]),  # truncated bucket row
        lambda d: d.update(alpha_t=[[float("nan"), 0.2]]),
        lambda d: d.update(alpha_t=[[True, 0.2]]),
        lambda d: d.update(alpha_t=[["0.2", 0.2]]),
        lambda d: d.update(alpha_t=[[None, 0.2]]),
        lambda d: d.update(scores=[[[True], [0.1]]]),
        lambda d: d.update(scores=[]),
        lambda d: d.update(num_buckets=-1),
    ):
        bad = dict(aci_data)
        mutate(bad)
        with pytest.raises(ValueError):
            OnlineConformalCalibrator.from_dict(bad)


def test_aci_transform_is_read_only_and_update_uses_repaired_interval():
    preds, actuals, scale = _make_simple(6, 8, seed=8)
    online = OnlineConformalCalibrator(target_coverage=0.80, bucket_size=4)
    before = online.to_dict()
    out = online.transform(preds, scale)
    assert online.to_dict() == before, "transform() must not mutate calibrator state"
    assert np.all(np.diff(out, axis=-1) >= 0.0)

    # Verify update evaluates err_t on the *repaired* interval: if 0.50's huge offset widens the 0.80
    # interval to cover y, 0.80 registers err=0 (so its alpha_t rises rather than falling).
    multi = OnlineConformalCalibrator(target_coverage=(0.50, 0.80), gamma=0.1, window=20, bucket_size=4)
    p1 = np.zeros((1, 4, len(DEFAULT_QUANTILE_LEVELS)))
    # Seed 0.50 with score 10.0 and 0.80 with score 0.0, with alpha_t=0.5 / 0.2.
    multi._ensure_buckets(1)
    multi._push_observation(0, 0, [10.0])
    multi._push_observation(1, 0, [0.0])
    alpha_80_before = float(multi.alpha_t[1, 0])
    # y=5.0 lies outside raw 0.80 [0, 0] + 0 = [0, 0], but INSIDE repaired 0.80 [-10, 10] pushed by 0.50!
    multi.update(p1, np.full((1, 4), 5.0), np.ones(1))
    assert float(multi.alpha_t[1, 0]) > alpha_80_before, "ACI should score err_t against the repaired interval"

    # bfloat16 and requires_grad=True inputs work cleanly without error.
    t_bf16 = torch.tensor(preds, dtype=torch.bfloat16, requires_grad=True)
    t_out = multi.transform(t_bf16)
    assert isinstance(t_out, torch.Tensor) and t_out.dtype == torch.bfloat16 and (t_out.diff(dim=-1) >= 0).all()


def test_aci_window_eviction_and_anti_windup_regime_recovery():
    """Verify rolling forecast-observation window semantics & eviction, median clamp on extreme
    narrowing, and bounded Lemma 4.1 recovery on both the lower (-gamma) and upper (1 + gamma) bounds.
    """
    # 1. Window counts FORECASTS (not individual steps) and evicts old observations cleanly:
    # with window=2, bucket_size=4 (4 step scores per forecast), pushing 1 outlier forecast (100.0)
    # then 1 tiny forecast (0.1) MUST retain 100.0 (step-counting window=2 would have evicted it already);
    # pushing a 2nd tiny forecast evicts the outlier and drops the offset to 0.1.
    online = OnlineConformalCalibrator(target_coverage=0.80, gamma=0.005, window=2, bucket_size=4)
    zero_preds = np.zeros((1, 4, len(DEFAULT_QUANTILE_LEVELS)), dtype=np.float64)
    online.update(zero_preds, np.full((1, 4), 100.0), np.ones(1))
    online.update(zero_preds, np.full((1, 4), 0.1), np.ones(1))
    assert online.current_offsets(4)[0, 0] == pytest.approx(100.0), "window counted steps instead of forecasts"
    online.update(zero_preds, np.full((1, 4), 0.1), np.ones(1))
    assert online.current_offsets(4)[0, 0] == pytest.approx(0.1), "old outlier forecast was not evicted"

    # 2. Extreme negative offset clamps calibrated bounds at the unshifted median (never crosses median).
    median_idx = DEFAULT_QUANTILE_LEVELS.index(0.50)
    low_80, high_80 = DEFAULT_QUANTILE_LEVELS.index(0.10), DEFAULT_QUANTILE_LEVELS.index(0.90)
    wide = np.linspace(-10.0, 10.0, len(DEFAULT_QUANTILE_LEVELS))[None, None, :].repeat(4, axis=1) + 5.0
    cal_clamp = ConformalCalibrator(target_coverage=0.80, bucket_size=4).fit(wide, np.full((1, 4), 5.0), np.ones(1))
    cal_clamp.offsets_ = np.array([[-50.0]])
    clamped = cal_clamp.transform(wide, np.ones(1))
    assert np.all(np.diff(clamped, axis=-1) >= 0.0)
    np.testing.assert_array_equal(clamped[..., median_idx], wide[..., median_idx])
    np.testing.assert_array_equal(clamped[..., low_80], wide[..., median_idx])
    np.testing.assert_array_equal(clamped[..., high_80], wide[..., median_idx])

    # 3. Over-wide -> narrow regime shift with default single coverage 0.80 (reviewer Finding 2).
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(DEFAULT_QUANTILE_LEVELS, dtype=torch.float64)).numpy()
    rng = np.random.default_rng(42)
    gamma = 0.01
    aci = OnlineConformalCalibrator(target_coverage=0.80, gamma=gamma, window=200, bucket_size=32)

    for _ in range(600):
        y = rng.normal(0.0, 1.0, size=(1, 32))
        p = (2.2 * z)[None, None, :].repeat(32, axis=1)
        aci.update(p, y, np.ones(1))
        assert -gamma - 1e-12 <= float(aci.alpha_t[0, 0]) <= 1.0 + gamma + 1e-12
    assert float(aci.alpha_t[0, 0]) < 0.45, f"alpha_t wound up on over-wide forecasts: {float(aci.alpha_t[0, 0]):.3f}"

    post_cov: list[float] = []
    for step in range(250):
        y = rng.normal(0.0, 1.0, size=(1, 32))
        p = (0.45 * z)[None, None, :].repeat(32, axis=1)
        cal = aci.transform(p, np.ones(1))
        if step >= 60:
            post_cov.append(float(np.mean((y >= cal[..., low_80]) & (y <= cal[..., high_80]))))
        aci.update(p, y, np.ones(1))
    assert float(np.mean(post_cov)) >= 0.72, f"post-shift recovery too slow: {float(np.mean(post_cov)):.3f}"

    # 4. Escalating errors (lower bound): alpha_t saturates at -gamma (never winds down to -2.2).
    aci_esc = OnlineConformalCalibrator(target_coverage=0.80, gamma=0.05, window=50, bucket_size=4)
    for k in range(1, 80):
        aci_esc.update(zero_preds, np.full((1, 4), float(k)), np.ones(1))
    assert aci_esc.saturated and float(aci_esc.alpha_t[0, 0]) == pytest.approx(-0.05)

    # 5. Zero-noise exact-median stream (upper bound): err_t == 0 forever, so alpha_t caps at 1 + gamma.
    aci_hi = OnlineConformalCalibrator(target_coverage=0.80, gamma=0.05, window=50, bucket_size=4)
    for _ in range(150):
        aci_hi.update(zero_preds, np.zeros((1, 4)), np.ones(1))
    assert float(aci_hi.alpha_t[0, 0]) == pytest.approx(1.0 + 0.05)


def test_collect_conformal_data_matches_upstream_and_preserves_small_scales(tiny_forecaster):
    from timestral_tune import collect_conformal_data
    from timestral_tune.data import SeriesData, TimeSeriesDataset

    from .helpers import make_toy_dataset

    dataset = make_toy_dataset(tiny_forecaster.patch_size, series=4, covariates=True)
    preds, actuals, scale = collect_conformal_data(tiny_forecaster, dataset, batch_size=3, symmetric_averaging=True)
    preds_nosym, _, _ = collect_conformal_data(tiny_forecaster, dataset, batch_size=3, symmetric_averaging=False)

    # 1. Exact agreement with single-window upstream forecast (including known covariates) & actual alignment.
    w0 = dataset[0]
    ref_sym = (
        tiny_forecaster.forecast(
            w0.target_context[None],
            dataset.forecast_horizon,
            past_covariates=w0.past[None],
            known_covariates=w0.known[None],
            symmetric_averaging=True,
        )
        .quantiles.cpu()
        .numpy()[0, 0]
    )
    ref_nosym = (
        tiny_forecaster.forecast(
            w0.target_context[None],
            dataset.forecast_horizon,
            past_covariates=w0.past[None],
            known_covariates=w0.known[None],
            symmetric_averaging=False,
        )
        .quantiles.cpu()
        .numpy()[0, 0]
    )
    np.testing.assert_allclose(preds[0], ref_sym, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(preds_nosym[0], ref_nosym, rtol=1e-5, atol=1e-5)
    assert not np.allclose(preds[0], preds_nosym[0])
    for i in range(len(dataset)):
        np.testing.assert_allclose(actuals[i], dataset[i].target_future[0], rtol=1e-6, atol=1e-6)
        assert scale[i] == pytest.approx(float(np.std(dataset[i].target_context[0])), rel=1e-6)

    # 2. Small-magnitude data (realised-variance range ~3e-6 to 3e-5): true std preserved, NOT jumped to 1.0!
    rng = np.random.default_rng(123)
    patch = tiny_forecaster.patch_size
    steps = 6 * patch
    small_series = []
    expected_stds = []
    for idx, sigma in enumerate((3e-6, 5e-6, 9e-6, 2e-5, 3e-5)):
        vals = (5 * sigma + rng.normal(0.0, sigma, size=(1, steps))).astype(np.float32)
        small_series.append(
            SeriesData(
                series_id=f"rv-{idx}",
                timestamps=np.arange(steps, dtype=np.int64),
                targets=vals,
                past=np.empty((0, steps), dtype=np.float32),
                known=np.empty((0, steps), dtype=np.float32),
                categorical_past=np.empty((0, steps), dtype=np.float64),
                categorical_known=np.empty((0, steps), dtype=np.float64),
            )
        )
        expected_stds.append(float(np.std(vals[0, : 4 * patch].astype(np.float64))))
    small_ds = TimeSeriesDataset(
        small_series, context_length=4 * patch, forecast_horizon=2 * patch, stride=2 * patch, min_context=4 * patch
    )
    s_preds, s_actuals, s_scale = collect_conformal_data(tiny_forecaster, small_ds, batch_size=3)
    for got, exp in zip(s_scale, expected_stds, strict=True):
        assert got == pytest.approx(exp, rel=1e-5), f"small-scale std corrupted: got {got:.3e}, expected {exp:.3e}"
    cal = ConformalCalibrator(target_coverage=0.80, bucket_size=patch).fit(s_preds, s_actuals, s_scale)
    out = cal.transform(s_preds, s_scale)
    assert out.shape == s_preds.shape and np.all(np.diff(out, axis=-1) >= 0.0)

    # 3. Flat non-zero small-magnitude context falls back to its own magnitude (not 1.0), and
    # categorical covariates + clip_nonnegative flag are forwarded accurately to upstream forecast().
    flat_vals = np.full((1, steps), 3e-5, dtype=np.float32)
    # Non-negative high-variance context touching 0 so unclipped lower quantiles go negative.
    ramp_vals = np.where(np.arange(steps) % 2 == 0, 0.0, 10.0).astype(np.float32)[None, :]
    cat_past = (np.arange(steps) % 3).astype(np.float64)[None, :]
    cat_known = (np.arange(steps) % 4).astype(np.float64)[None, :]
    cat_ds = TimeSeriesDataset(
        [
            SeriesData(
                series_id="flat-small",
                timestamps=np.arange(steps, dtype=np.int64),
                targets=flat_vals,
                past=np.empty((0, steps), dtype=np.float32),
                known=np.empty((0, steps), dtype=np.float32),
                categorical_past=cat_past,
                categorical_known=cat_known,
            ),
            SeriesData(
                series_id="nonneg-ramp",
                timestamps=np.arange(steps, dtype=np.int64),
                targets=ramp_vals,
                past=np.empty((0, steps), dtype=np.float32),
                known=np.empty((0, steps), dtype=np.float32),
                categorical_past=cat_past,
                categorical_known=cat_known,
            ),
        ],
        context_length=4 * patch,
        forecast_horizon=2 * patch,
        stride=2 * patch,
        min_context=4 * patch,
    )
    c_preds, _, c_scale = collect_conformal_data(tiny_forecaster, cat_ds, batch_size=2, clip_nonnegative=True)
    c_preds_noclip, _, _ = collect_conformal_data(tiny_forecaster, cat_ds, batch_size=2, clip_nonnegative=False)
    assert c_scale[0] == pytest.approx(3e-5, rel=1e-5), f"flat small context jumped to {c_scale[0]}"
    assert np.all(c_preds >= 0.0), "clip_nonnegative=True must clamp non-negative context forecasts at 0"
    assert not np.allclose(c_preds, c_preds_noclip), "clip_nonnegative=False should differ from True"
    w_cat = cat_ds[1]
    ref_cat = (
        tiny_forecaster.forecast(
            w_cat.target_context[None],
            cat_ds.forecast_horizon,
            categorical_past_covariates=w_cat.categorical_past[None],
            categorical_known_covariates=w_cat.categorical_known[None],
            clip_nonnegative=False,
        )
        .quantiles.cpu()
        .numpy()[0, 0]
    )
    np.testing.assert_allclose(c_preds_noclip[1], ref_cat, rtol=1e-5, atol=1e-5)
