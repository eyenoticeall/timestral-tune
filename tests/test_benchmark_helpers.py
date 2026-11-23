"""Network-free unit tests for practical-value benchmark metrics and dataset helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = str(Path(__file__).resolve().parent.parent / "scripts" / "benchmarks")
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

from datasets import _align_series_for_cutoff, subset_train_series  # noqa: E402
from decision_metrics import (  # noqa: E402
    paired_decision_bootstrap,
    paired_decision_bootstrap_multiseed,
    service_level_pinball,
    winkler_interval_score,
)

from timestral_tune.data import ColumnRoles, SeriesData, TimeSeriesDataset  # noqa: E402
from timestral_tune.models import DEFAULT_QUANTILE_LEVELS  # noqa: E402
from timestral_tune.utils import pinball_loss_raw  # noqa: E402


def test_winkler_interval_score_exact_formula_and_nan_mask() -> None:
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
    # 3 observations across (N=1, H=3): inside [10, 20], below lower by 2, above upper by 3
    preds = np.zeros((1, 4, levels.size), dtype=np.float64)
    lo80 = int(np.where(np.isclose(levels, 0.10))[0][0])
    hi80 = int(np.where(np.isclose(levels, 0.90))[0][0])
    preds[:, :, lo80] = 10.0
    preds[:, :, hi80] = 20.0
    y = np.array([[15.0, 8.0, 23.0, np.nan]], dtype=np.float64)

    # alpha = 0.2 -> 2/alpha = 10. Width = 10 everywhere.
    # Obs 0: 10. Obs 1: 10 + 10*2 = 30. Obs 2: 10 + 10*3 = 40. Obs 3: NaN (masked). Mean = 80 / 3.
    score = winkler_interval_score(preds, y, nominal_coverage=0.80)
    assert score == pytest.approx(80.0 / 3.0, rel=1e-12)


def test_service_level_pinball_matches_mean_and_newsvendor_formula() -> None:
    rng = np.random.default_rng(7)
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
    preds = np.sort(rng.normal(5.0, 2.0, size=(8, 12, levels.size)), axis=-1)
    y = rng.normal(5.5, 2.0, size=(8, 12))

    per_q = [service_level_pinball(preds, y, float(q))[0] for q in levels]
    assert float(np.mean(per_q)) == pytest.approx(pinball_loss_raw(preds, y), rel=1e-12)

    # Analytical check at tau = 0.90: under-prediction costs 0.9*(y-q), over-prediction costs 0.1*(q-y).
    q90_idx = int(np.where(np.isclose(levels, 0.90))[0][0])
    q_hat = preds[..., q90_idx]
    expected_cost = float(np.mean(np.where(y >= q_hat, 0.90 * (y - q_hat), 0.10 * (q_hat - y))))
    expected_fill = float(np.mean(y <= q_hat))
    cost, fill = service_level_pinball(preds, y, 0.90)
    assert cost == pytest.approx(expected_cost, rel=1e-12)
    assert fill == pytest.approx(expected_fill, rel=1e-12)


def test_paired_decision_bootstrap_detects_known_improvement() -> None:
    rng = np.random.default_rng(123)
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
    n, h = 40, 8
    y = rng.uniform(5.0, 15.0, size=(n, h))
    ctx = rng.uniform(5.0, 15.0, size=(n, 32))
    # Model A is tighter and closer to y than Model B across every cluster.
    z = (levels - 0.5)[None, None, :]
    preds_a = y[:, :, None] + 0.5 * z
    preds_b = y[:, :, None] + 2.0 * z
    clusters = np.arange(n) % 20

    res = paired_decision_bootstrap(preds_a, preds_b, y, ctx, clusters, seasonality=4, n_boot=1000, seed=0)
    assert res["wql"]["obs_delta"] < 0.0
    assert res["wql"]["ci95_rel_delta"][1] < 0.0
    assert res["winkler_80"]["ci95_rel_delta"][1] < 0.0
    assert res["width_80"]["obs_rel_delta"] == pytest.approx(-0.75, rel=1e-9)
    assert res["pinball_90"]["ci95_rel_delta"][1] < 0.0


def test_paired_decision_bootstrap_multiseed_contains_mean_across_divergent_seeds() -> None:
    rng = np.random.default_rng(99)
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
    n, h = 40, 8
    y = rng.uniform(5.0, 15.0, size=(n, h))
    ctx = rng.uniform(5.0, 15.0, size=(n, 32))
    z = (levels - 0.5)[None, None, :]
    preds_b = y[:, :, None] + 2.0 * z
    # Seed 0 is 6% narrower, seeds 1 and 2 are 1.5% narrower -> 3-seed mean is -3.0% (outside seed-0-only CI!)
    preds_a_seeds = [
        y[:, :, None] + 1.88 * z,
        y[:, :, None] + 1.97 * z,
        y[:, :, None] + 1.97 * z,
    ]
    clusters = np.arange(n) % 20
    res = paired_decision_bootstrap_multiseed(
        preds_a_seeds, preds_b, y, ctx, clusters, seasonality=4, n_boot=3000, seed=0
    )
    w = res["width_80"]
    assert res["n_seeds"] == 3
    assert w["obs_rel_delta"] == pytest.approx(-0.03, rel=1e-9)
    assert w["ci95_rel_delta"][0] <= w["obs_rel_delta"] <= w["ci95_rel_delta"][1]
    assert w["ci95_rel_delta"][0] == pytest.approx(-0.06, abs=1e-6)
    assert w["ci95_rel_delta"][1] == pytest.approx(-0.015, abs=1e-6)


def test_subset_train_series_and_cutoff_alignment() -> None:
    series = [
        SeriesData(
            series_id=f"s{i}",
            timestamps=np.arange(100, dtype=np.int64),
            targets=np.ones((1, 100), dtype=np.float32) * (i + 1),
            past=np.empty((0, 100), dtype=np.float32),
            known=np.empty((0, 100), dtype=np.float32),
            categorical_past=np.empty((0, 100), dtype=np.float64),
            categorical_known=np.empty((0, 100), dtype=np.float64),
        )
        for i in range(20)
    ]
    aligned = _align_series_for_cutoff(series, cutoff=73, forecast_horizon=5, stride=5)
    ds = TimeSeriesDataset(
        aligned,
        context_length=32,
        forecast_horizon=5,
        stride=5,
        min_context=32,
        roles=ColumnRoles(targets=("y",)),
    )
    _, te = ds.split_by_time(cutoff=73)
    first_ts = min(int(te.series[int(te._series_index[i])].timestamps[int(te._origins[i])]) for i in range(len(te)))
    assert first_ts == 73

    sub10 = subset_train_series(ds, 0.10, seed=0)
    sub30 = subset_train_series(ds, 0.30, seed=0)
    sub10_repeat = subset_train_series(ds, 0.10, seed=0)
    assert len(set(int(i) for i in sub10._series_index)) == 2
    assert len(set(int(i) for i in sub30._series_index)) == 6
    assert np.array_equal(sub10._series_index, sub10_repeat._series_index)
    assert subset_train_series(ds, 1.0, seed=0) is ds


def test_gbt_early_stop_does_not_freeze_on_zero_inflated_targets() -> None:
    pytest.importorskip("sklearn")
    from baselines import _fit_quantile_with_val_early_stop

    rng = np.random.default_rng(7)
    n_tr, n_va = 1200, 300
    x_tr = np.column_stack([rng.integers(0, 4, size=n_tr), rng.uniform(0.0, 1000.0, size=n_tr)])
    x_va = np.column_stack([rng.integers(0, 4, size=n_va), rng.uniform(0.0, 1000.0, size=n_va)])
    y_tr = np.where(x_tr[:, 1] < 300.0, 0.0, x_tr[:, 1] + rng.normal(0.0, 20.0, size=n_tr))
    y_va = np.where(x_va[:, 1] < 300.0, 0.0, x_va[:, 1] + rng.normal(0.0, 20.0, size=n_va))
    cfg = {"max_leaf_nodes": 15, "min_samples_leaf": 20, "learning_rate": 0.05, "l2_regularization": 0.0}
    pv, _, best_iter, _ = _fit_quantile_with_val_early_stop(
        x_tr,
        y_tr,
        x_va,
        y_va,
        np.zeros(n_va),
        np.ones(n_va),
        tau=0.1,
        tf="raw",
        cfg=cfg,
        max_iter=60,
        eval_stride=10,
        clip_nonnegative=True,
        seed=0,
    )
    assert np.std(pv) > 100.0
    assert best_iter >= 20

    from baselines import _fit_median_anchor

    cfg_l2 = {**cfg, "l2_regularization": 10.0}
    p50_tr, p50_va, _, bi50 = _fit_median_anchor(
        x_tr, y_tr, x_va, y_va, cfg_l2, max_iter=60, eval_stride=10, clip_nonnegative=True, seed=0
    )
    pv_res, _, bi_res, _ = _fit_quantile_with_val_early_stop(
        x_tr,
        y_tr - p50_tr,
        x_va,
        y_va,
        p50_va,
        np.ones(n_va),
        tau=0.1,
        tf="median_residual",
        cfg=cfg_l2,
        max_iter=60,
        eval_stride=10,
        clip_nonnegative=True,
        seed=0,
    )
    assert bi50 >= 10
    assert bi_res >= 10
    assert np.std(pv_res) > 100.0
