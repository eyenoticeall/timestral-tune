"""Decision-relevant forecast metrics, conformal wrappers, and paired cluster bootstraps.

Reuses `timestral_tune.utils.metrics`, `timestral_tune.models.conformal`, and
`scripts/benchmarks/significance.py` so every benchmark shares one exact scoring path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from significance import _clustered_paired_stats

from timestral_tune import ConformalCalibrator, OnlineConformalCalibrator
from timestral_tune.data import TimeSeriesDataset
from timestral_tune.models import DEFAULT_QUANTILE_LEVELS
from timestral_tune.utils import coverage, evaluate_predictions
from timestral_tune.utils.metrics import _apply_online_chronological, _collect_contexts, _pair_indices


def _quantile_index(level: float, levels: np.ndarray) -> int:
    matches = np.where(np.isclose(levels, float(level), atol=1e-9))[0]
    if matches.size != 1:
        raise ValueError(f"quantile level {level!r} not found in {levels.tolist()}")
    return int(matches[0])


def winkler_interval_score(
    predictions: np.ndarray,
    actuals: np.ndarray,
    nominal_coverage: float = 0.80,
    *,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> float:
    """Mean Winkler (1972) interval score in raw units over finite `(n, h)` observations.

    For nominal coverage `1 - alpha` and predicted interval `[l, u]`:
        S_alpha(l, u; y) = (u - l) + (2 / alpha) * max(l - y, 0) + (2 / alpha) * max(y - u, 0).
    """
    preds = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(actuals, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    low_idx, high_idx = _pair_indices(nominal_coverage, levels)
    alpha = 1.0 - float(nominal_coverage)
    valid = np.isfinite(y)
    if not valid.any():
        raise ValueError("actuals have no finite observations")
    lo = preds[..., low_idx][valid]
    hi = preds[..., high_idx][valid]
    yv = y[valid]
    score = (hi - lo) + (2.0 / alpha) * np.maximum(lo - yv, 0.0) + (2.0 / alpha) * np.maximum(yv - hi, 0.0)
    return float(np.mean(score))


def service_level_pinball(
    predictions: np.ndarray,
    actuals: np.ndarray,
    quantile_level: float,
    *,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> tuple[float, float]:
    """Return `(mean_pinball_loss, empirical_fill_rate)` at a single service level `tau`.

    For retail demand, `mean_pinball_loss` is the raw-unit newsvendor cost at critical fractile `tau`
    (with unit costs normalised to `c_u + c_o = 1`), and `empirical_fill_rate = P(y <= q_tau)`.
    """
    preds = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(actuals, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    q_idx = _quantile_index(quantile_level, levels)
    valid = np.isfinite(y)
    if not valid.any():
        raise ValueError("actuals have no finite observations")
    q_hat = preds[..., q_idx][valid]
    yv = y[valid]
    diff = yv - q_hat
    loss = np.maximum(quantile_level * diff, (quantile_level - 1.0) * diff)
    fill = float(np.mean(yv <= q_hat))
    return float(np.mean(loss)), fill


def evaluate_decision_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    contexts: np.ndarray,
    *,
    seasonality: int,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> dict[str, Any]:
    """Compute standard `MetricsSummary` plus Winkler interval scores and 0.80/0.90/0.95 service costs."""
    base = evaluate_predictions(
        predictions,
        actuals,
        contexts,
        seasonality=seasonality,
        quantile_levels=quantile_levels,
    ).to_dict()
    base["winkler_80"] = winkler_interval_score(predictions, actuals, 0.80, quantile_levels=quantile_levels)
    base["winkler_90"] = winkler_interval_score(predictions, actuals, 0.90, quantile_levels=quantile_levels)
    for tau, tag in ((0.80, "80"), (0.90, "90"), (0.95, "95")):
        pb, fill = service_level_pinball(predictions, actuals, tau, quantile_levels=quantile_levels)
        base[f"pinball_{tag}"] = pb
        base[f"fill_rate_{tag}"] = fill
    return base


def per_origin_coverage(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dataset: TimeSeriesDataset,
    nominal_coverage: float = 0.80,
) -> list[float]:
    origins = np.array(
        [
            dataset.series[int(dataset._series_index[i])].timestamps[int(dataset._origins[i])]
            for i in range(len(dataset))
        ]
    )
    out = []
    for orig in sorted(np.unique(origins)):
        mask = origins == orig
        out.append(round(coverage(predictions[mask], actuals[mask], nominal_coverage), 4))
    return out


def cluster_ids_for_dataset(dataset: TimeSeriesDataset) -> tuple[np.ndarray, np.ndarray]:
    """Return `(series_ids, origin_times)` aligned with `collect_conformal_data` rows."""
    n_targets = dataset.series[0].targets.shape[0]
    series_ids = np.repeat(
        np.array([int(dataset._series_index[i]) for i in range(len(dataset))], dtype=np.int64),
        n_targets,
    )
    origin_times = np.repeat(
        np.array(
            [
                dataset.series[int(dataset._series_index[i])].timestamps[int(dataset._origins[i])]
                for i in range(len(dataset))
            ]
        ),
        n_targets,
    )
    return series_ids, origin_times


def evaluate_with_calibrators(
    val_preds: np.ndarray,
    val_y: np.ndarray,
    val_scale: np.ndarray,
    test_preds: np.ndarray,
    test_y: np.ndarray,
    test_scale: np.ndarray,
    test_ds: TimeSeriesDataset,
    *,
    seasonality: int,
    patch_size: int = 32,
    aci_gamma: float = 0.01,
    aci_window: int = 500,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate raw predictions plus static CQR `(0.80, 0.90)` / `"all"` and online ACI `(0.80, 0.90)` / `"all"`.

    Returns `(metrics_dict, preds_dict)` where `preds_dict` contains `"raw"`, `"cqr"`, `"cqr_all"`,
    `"aci"`, `"aci_all"` for downstream paired cluster bootstrap comparisons.
    """
    test_ctx = _collect_contexts(test_ds)
    raw_m = evaluate_decision_metrics(test_preds, test_y, test_ctx, seasonality=seasonality)
    raw_m["per_block_cov80"] = per_origin_coverage(test_preds, test_y, test_ds, 0.80)

    cqr = ConformalCalibrator(target_coverage=(0.80, 0.90), bucket_size=patch_size).fit(val_preds, val_y, val_scale)
    cqr_preds = cqr.transform(test_preds, context_scale=test_scale)
    cqr_m = evaluate_decision_metrics(cqr_preds, test_y, test_ctx, seasonality=seasonality)
    cqr_m["per_block_cov80"] = per_origin_coverage(cqr_preds, test_y, test_ds, 0.80)

    cqr_all = ConformalCalibrator(target_coverage="all", bucket_size=patch_size).fit(val_preds, val_y, val_scale)
    cqr_all_preds = cqr_all.transform(test_preds, context_scale=test_scale)
    cqr_all_m = evaluate_decision_metrics(cqr_all_preds, test_y, test_ctx, seasonality=seasonality)

    aci = OnlineConformalCalibrator(
        target_coverage=(0.80, 0.90), gamma=aci_gamma, window=aci_window, bucket_size=patch_size
    )
    aci.update(val_preds, val_y, val_scale)
    aci_preds = _apply_online_chronological(aci, test_ds, test_preds, test_y, test_scale)
    aci_m = evaluate_decision_metrics(aci_preds, test_y, test_ctx, seasonality=seasonality)
    aci_m["per_block_cov80"] = per_origin_coverage(aci_preds, test_y, test_ds, 0.80)

    aci_all = OnlineConformalCalibrator(
        target_coverage="all", gamma=aci_gamma, window=aci_window, bucket_size=patch_size
    )
    aci_all.update(val_preds, val_y, val_scale)
    aci_all_preds = _apply_online_chronological(aci_all, test_ds, test_preds, test_y, test_scale)
    aci_all_m = evaluate_decision_metrics(aci_all_preds, test_y, test_ctx, seasonality=seasonality)
    aci_all_m["per_block_cov80"] = per_origin_coverage(aci_all_preds, test_y, test_ds, 0.80)

    return (
        {
            "raw": raw_m,
            "cqr": cqr_m,
            "cqr_all": cqr_all_m,
            "aci": aci_m,
            "aci_all": aci_all_m,
        },
        {
            "raw": test_preds,
            "cqr": cqr_preds,
            "cqr_all": cqr_all_preds,
            "aci": aci_preds,
            "aci_all": aci_all_preds,
        },
    )


def _window_decision_sums(
    predictions: np.ndarray,
    actuals: np.ndarray,
    levels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-window finite count and score sums for exact cluster bootstrap of decision metrics."""
    preds = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(actuals, dtype=np.float64)
    valid = np.isfinite(y)
    counts = valid.sum(axis=1).astype(np.float64)

    out: dict[str, np.ndarray] = {"count": counts}
    for cov, tag in ((0.80, "80"), (0.90, "90")):
        lo_i, hi_i = _pair_indices(cov, levels)
        alpha = 1.0 - cov
        lo = preds[..., lo_i]
        hi = preds[..., hi_i]
        width = np.where(valid, hi - lo, 0.0)
        cov_hit = np.where(valid, ((y >= lo) & (y <= hi)).astype(np.float64), 0.0)
        wink = np.where(
            valid,
            (hi - lo) + (2.0 / alpha) * np.maximum(lo - y, 0.0) + (2.0 / alpha) * np.maximum(y - hi, 0.0),
            0.0,
        )
        out[f"width_{tag}"] = width.sum(axis=1)
        out[f"coverage_{tag}"] = cov_hit.sum(axis=1)
        out[f"winkler_{tag}"] = wink.sum(axis=1)

    for tau, tag in ((0.80, "80"), (0.90, "90"), (0.95, "95")):
        q_i = _quantile_index(tau, levels)
        diff = np.where(valid, y - preds[..., q_i], 0.0)
        pb = np.where(valid, np.maximum(tau * diff, (tau - 1.0) * diff), 0.0)
        out[f"pinball_{tag}"] = pb.sum(axis=1)

    return out


def _cluster_wql_arrays(
    predictions: np.ndarray,
    actuals: np.ndarray,
    cluster_ids: np.ndarray,
    uniq: np.ndarray,
    levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k = uniq.size
    c_num = np.empty(k, dtype=np.float64)
    c_den = np.empty(k, dtype=np.float64)
    valid = np.isfinite(actuals)
    for idx, cid in enumerate(uniq):
        m = cluster_ids == cid
        yv = actuals[m][valid[m]]
        pv = predictions[m][valid[m]]
        diff = yv[:, None] - pv
        pb = np.maximum(levels[None, :] * diff, (levels[None, :] - 1.0) * diff)
        c_num[idx] = float(2.0 * np.mean(np.sum(pb, axis=0)))
        c_den[idx] = float(np.sum(np.abs(yv)))
    return c_num, c_den


def _window_mase(
    predictions: np.ndarray, actuals: np.ndarray, contexts: np.ndarray, seasonality: int, levels: np.ndarray
) -> np.ndarray:
    med_i = int(np.where(np.isclose(levels, 0.5, atol=1e-9))[0][0])
    n = predictions.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        v = np.isfinite(actuals[i])
        if not v.any():
            continue
        cd = contexts[i, seasonality:] - contexts[i, :-seasonality]
        cv = np.isfinite(cd)
        scale = float(np.mean(np.abs(cd[cv]))) if cv.any() else 0.0
        if scale > 0.0:
            out[i] = float(np.mean(np.abs(actuals[i, v] - predictions[i, v, med_i])) / scale)
    return out


def paired_decision_bootstrap_multiseed(
    preds_a_seeds: Sequence[np.ndarray],
    preds_b_or_seeds: np.ndarray | Sequence[np.ndarray],
    actuals: np.ndarray,
    contexts: np.ndarray,
    cluster_ids: np.ndarray,
    seasonality: int,
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Single-run cluster+seed paired bootstrap across all training seeds (no ensemble bias).

    Each bootstrap iteration draws one seed `s ~ Uniform{0..M-1}` and resamples `K` clusters
    with replacement from seed `s`'s cluster sums, capturing both data-cluster and training-seed
    variance. Observed point estimates equal the exact M-seed mean of single-run metrics.
    """
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
    a_list = [np.asarray(p, dtype=np.float64) for p in preds_a_seeds]
    m_seeds = len(a_list)
    if m_seeds == 0:
        raise ValueError("preds_a_seeds must be non-empty")
    if isinstance(preds_b_or_seeds, np.ndarray):
        b_list = [np.asarray(preds_b_or_seeds, dtype=np.float64)] * m_seeds
    else:
        b_list = [np.asarray(p, dtype=np.float64) for p in preds_b_or_seeds]
        if len(b_list) != m_seeds:
            raise ValueError(f"preds_b_or_seeds length {len(b_list)} != preds_a_seeds length {m_seeds}")

    uniq = np.unique(cluster_ids)
    k = uniq.size
    rng = np.random.default_rng(seed)
    idx_draw = rng.integers(0, k, size=(n_boot, k))
    seed_draw = rng.integers(0, m_seeds, size=(n_boot,)) if m_seeds > 1 else np.zeros(n_boot, dtype=np.int64)

    # Precompute cluster-level wQL & MASE arrays per seed: shapes (M, K)
    wql_a = np.empty((m_seeds, k), dtype=np.float64)
    wql_b = np.empty((m_seeds, k), dtype=np.float64)
    wql_den = np.empty(k, dtype=np.float64)
    c_mase_a = np.empty((m_seeds, k), dtype=np.float64)
    c_mase_b = np.empty((m_seeds, k), dtype=np.float64)
    c_cnt = np.empty(k, dtype=np.float64)

    metric_keys = (
        "winkler_80",
        "winkler_90",
        "width_80",
        "width_90",
        "coverage_80",
        "coverage_90",
        "pinball_80",
        "pinball_90",
        "pinball_95",
    )
    ca_all = {mk: np.empty((m_seeds, k), dtype=np.float64) for mk in metric_keys}
    cb_all = {mk: np.empty((m_seeds, k), dtype=np.float64) for mk in metric_keys}

    for s_i in range(m_seeds):
        na, den = _cluster_wql_arrays(a_list[s_i], actuals, cluster_ids, uniq, levels)
        nb, _ = _cluster_wql_arrays(b_list[s_i], actuals, cluster_ids, uniq, levels)
        wql_a[s_i] = na
        wql_b[s_i] = nb
        if s_i == 0:
            wql_den = den

        ma_w = _window_mase(a_list[s_i], actuals, contexts, seasonality, levels)
        mb_w = _window_mase(b_list[s_i], actuals, contexts, seasonality, levels)
        sa = _window_decision_sums(a_list[s_i], actuals, levels)
        sb = _window_decision_sums(b_list[s_i], actuals, levels)
        for idx, cid in enumerate(uniq):
            mask = cluster_ids == cid
            if s_i == 0:
                c_cnt[idx] = sa["count"][mask].sum()
            vm = np.isfinite(ma_w[mask]) & np.isfinite(mb_w[mask])
            c_mase_a[s_i, idx] = float(np.mean(ma_w[mask][vm])) if vm.any() else np.nan
            c_mase_b[s_i, idx] = float(np.mean(mb_w[mask][vm])) if vm.any() else np.nan
            for mk in metric_keys:
                ca_all[mk][s_i, idx] = sa[mk][mask].sum()
                cb_all[mk][s_i, idx] = sb[mk][mask].sum()

    # Exact M-seed mean wQL point estimates + cluster+seed bootstrap draws
    tot_den = max(float(wql_den.sum()), 1e-30)
    per_seed_wql_a = wql_a.sum(axis=1) / tot_den
    per_seed_wql_b = wql_b.sum(axis=1) / tot_den
    per_seed_rel_wql = (per_seed_wql_a - per_seed_wql_b) / np.maximum(per_seed_wql_b, 1e-30)
    mean_wql_a = float(np.mean(per_seed_wql_a))
    mean_wql_b = float(np.mean(per_seed_wql_b))
    obs_delta_wql = mean_wql_a - mean_wql_b
    obs_rel_wql = obs_delta_wql / max(mean_wql_b, 1e-30)

    b_wql_a = wql_a[seed_draw[:, None], idx_draw].sum(axis=1)
    b_wql_b = wql_b[seed_draw[:, None], idx_draw].sum(axis=1)
    b_den = np.maximum(wql_den[idx_draw].sum(axis=1), 1e-30)
    b_delta_wql = (b_wql_a - b_wql_b) / b_den
    b_rel_wql = (b_wql_a - b_wql_b) / np.maximum(b_wql_b, 1e-30)
    boot_p_wql = float(2.0 * min(np.mean(b_delta_wql >= 0.0), np.mean(b_delta_wql <= 0.0)))

    # Cluster wins & Diebold-Mariano on seed-averaged per-cluster ΔwQL
    c_dwql_per_seed = (wql_a - wql_b) / np.maximum(wql_den[None, :], 1e-30)
    per_seed_wins = [int(np.sum(c_dwql_per_seed[s_i] < 0.0)) for s_i in range(m_seeds)]
    d_mean_c = c_dwql_per_seed.mean(axis=0)
    wins_mean_c = int(np.sum(d_mean_c < 0.0))
    d_bar = float(np.mean(d_mean_c))
    d_c = d_mean_c - d_bar
    gamma0 = float(np.mean(d_c**2))
    gamma1 = float(np.mean(d_c[1:] * d_c[:-1])) if k > 2 else 0.0
    var_nw = max(gamma0 + 2.0 * 0.5 * gamma1, gamma0 * 0.25, 1e-30)
    dm_stat = float(d_bar / math.sqrt(var_nw / k)) if k > 1 else float("nan")
    dm_p = float(math.erfc(abs(dm_stat) / math.sqrt(2.0))) if math.isfinite(dm_stat) else float("nan")

    c_dmase = c_mase_a - c_mase_b
    b_dmase = np.nanmean(c_dmase[seed_draw[:, None], idx_draw], axis=1)

    metrics_boot: dict[str, Any] = {
        "n_seeds": m_seeds,
        "wql": {
            "obs_delta": obs_delta_wql,
            "obs_rel_delta": obs_rel_wql,
            "per_seed_rel_delta": [float(x) for x in per_seed_rel_wql],
            "base_value": mean_wql_b,
            "ci95_delta": [float(np.quantile(b_delta_wql, 0.025)), float(np.quantile(b_delta_wql, 0.975))],
            "ci95_rel_delta": [float(np.quantile(b_rel_wql, 0.025)), float(np.quantile(b_rel_wql, 0.975))],
            "clusters_improved": wins_mean_c,
            "per_seed_clusters_improved": per_seed_wins,
            "n_clusters": int(k),
            "boot_p": boot_p_wql,
            "dm_p": dm_p,
        },
        "mase": {
            "obs_delta": float(np.nanmean(c_dmase)),
            "ci95_delta": [float(np.nanquantile(b_dmase, 0.025)), float(np.nanquantile(b_dmase, 0.975))],
        },
    }

    tot_cnt = max(float(c_cnt.sum()), 1.0)
    b_cnt = np.maximum(c_cnt[idx_draw].sum(axis=1), 1.0)
    for mk in metric_keys:
        ca = ca_all[mk]
        cb = cb_all[mk]
        per_seed_a = ca.sum(axis=1) / tot_cnt
        per_seed_b = cb.sum(axis=1) / tot_cnt
        mean_a = float(np.mean(per_seed_a))
        mean_b = float(np.mean(per_seed_b))
        obs_delta = mean_a - mean_b
        obs_rel = obs_delta / mean_b if abs(mean_b) > 1e-30 else float("nan")
        per_seed_rel = [
            float((per_seed_a[s_i] - per_seed_b[s_i]) / per_seed_b[s_i])
            if abs(per_seed_b[s_i]) > 1e-30
            else float("nan")
            for s_i in range(m_seeds)
        ]

        ba = ca[seed_draw[:, None], idx_draw].sum(axis=1) / b_cnt
        bb = cb[seed_draw[:, None], idx_draw].sum(axis=1) / b_cnt
        b_delta = ba - bb
        b_rel = b_delta / np.where(np.abs(bb) > 1e-30, bb, np.nan)
        lower_is_better = not mk.startswith("coverage_")
        c_delta_mean = ((ca - cb) / np.maximum(c_cnt[None, :], 1.0)).mean(axis=0)
        wins = int(np.sum(c_delta_mean < 0.0)) if lower_is_better else int(np.sum(c_delta_mean > 0.0))
        boot_p = float(2.0 * min(np.mean(b_delta >= 0.0), np.mean(b_delta <= 0.0)))

        entry: dict[str, Any] = {
            "obs_delta": obs_delta,
            "base_value": mean_b,
            "ci95_delta": [float(np.quantile(b_delta, 0.025)), float(np.quantile(b_delta, 0.975))],
            "clusters_improved": wins,
            "n_clusters": int(k),
            "boot_p": boot_p,
        }
        if not mk.startswith("coverage_") and math.isfinite(obs_rel):
            entry["obs_rel_delta"] = float(obs_rel)
            entry["per_seed_rel_delta"] = per_seed_rel
            entry["ci95_rel_delta"] = [float(np.nanquantile(b_rel, 0.025)), float(np.nanquantile(b_rel, 0.975))]
        metrics_boot[mk] = entry

    return metrics_boot


def paired_decision_bootstrap(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    actuals: np.ndarray,
    contexts: np.ndarray,
    cluster_ids: np.ndarray,
    seasonality: int,
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired cluster bootstrap comparing `preds_a` minus `preds_b` (single-seed convenience wrapper)."""
    core = _clustered_paired_stats(
        preds_a,
        preds_b,
        actuals,
        contexts,
        cluster_ids,
        seasonality,
        n_boot=n_boot,
        seed=seed,
    )
    res = paired_decision_bootstrap_multiseed(
        [preds_a], preds_b, actuals, contexts, cluster_ids, seasonality, n_boot=n_boot, seed=seed
    )
    res["wql"]["dm_p"] = core["dm_p_wql"]
    return res


def paired_both_clusters_multiseed(
    preds_a_seeds: Sequence[np.ndarray],
    preds_b_or_seeds: np.ndarray | Sequence[np.ndarray],
    actuals: np.ndarray,
    contexts: np.ndarray,
    dataset: TimeSeriesDataset,
    seasonality: int,
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run multi-seed cluster+seed paired bootstrap clustered separately by series and by time block."""
    series_ids, origin_times = cluster_ids_for_dataset(dataset)
    return {
        "by_series": paired_decision_bootstrap_multiseed(
            preds_a_seeds, preds_b_or_seeds, actuals, contexts, series_ids, seasonality, n_boot=n_boot, seed=seed
        ),
        "by_time_block": paired_decision_bootstrap_multiseed(
            preds_a_seeds, preds_b_or_seeds, actuals, contexts, origin_times, seasonality, n_boot=n_boot, seed=seed
        ),
    }


def paired_both_clusters(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    actuals: np.ndarray,
    contexts: np.ndarray,
    dataset: TimeSeriesDataset,
    seasonality: int,
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run paired cluster bootstrap clustered separately by series and by time block."""
    return paired_both_clusters_multiseed(
        [preds_a], preds_b, actuals, contexts, dataset, seasonality, n_boot=n_boot, seed=seed
    )
