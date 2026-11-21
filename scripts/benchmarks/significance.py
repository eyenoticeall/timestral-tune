"""Statistical significance, multi-seed variance, extended volatility window, and classical baselines.

    uv run python scripts/benchmarks/significance.py --device cuda --bf16 --output /tmp/sig.json

Quantifies the three caveats noted in Phase 3 benchmarks (`docs/BENCHMARKS.md`):

1. **Multi-seed variance:** runs `seeds=(0, 1, 2)` on Timestral-28M for `lora_variate`,
   `lora_temporal`, and `full_ft` on both realised volatility and retail demand.
2. **Paired significance tests:** computes series-clustered and time-block-clustered paired
   bootstrap 95% CIs (10,000 resamples) plus Diebold-Mariano test statistics on per-time-block
   wQL/MASE differences vs `zero_shot`:
   - Retail demand: 64 store-SKU series x 6 non-overlapping 48h test blocks (384 windows).
   - Realised volatility (strict post-Oct 27, 2026): 51 equities x 3 blocks (153 windows).
   - Realised volatility (extended 10-month 2026 holdout, trained strictly on 2024-2025 only):
     51 equities x ~40 non-overlapping 32h blocks (~2,000 windows), providing a large number of
     independent market regimes for time-block bootstrap & Diebold-Mariano inference.
3. **Classical non-foundation baselines** scored on the exact same test windows with the exact same
   raw-space `evaluate_predictions`:
   - Seasonal Naive (s=7 on volatility, s=24 on retail) with empirical validation residual quantiles.
   - HAR-RV + covariates (Corsi 2009 daily/weekly/monthly volatility lags + macro/volume/calendar
     Ridge regression + validation residual quantiles) on realised volatility.
   - Seasonal + Covariate Ridge (lag-1/24/48/168 sales + stockout/weather/discount/holiday/calendar
     Ridge regression + validation residual quantiles, clipped non-negative) on retail demand.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from datasets import (
    TWELVEDATA_REPO,
    TWELVEDATA_REV,
    BenchmarkSplits,
    load_realised_volatility,
    load_retail_demand,
)
from huggingface_hub import hf_hub_download

from timestral_tune import LoraConfig, TimestralTrainer, TrainingConfig, collect_conformal_data
from timestral_tune.data import ColumnRoles, SeriesData, TimeSeriesDataset
from timestral_tune.models import DEFAULT_QUANTILE_LEVELS, TuneTimestralForecaster, load_model, merged_state_dict
from timestral_tune.utils import evaluate_predictions
from timestral_tune.utils.metrics import _collect_contexts


def load_realised_volatility_2026_holdout(
    *,
    context_length: int = 256,
    forecast_horizon: int = 32,
    train_stride: int = 32,
    eval_stride: int = 32,
) -> BenchmarkSplits:
    """Extended 10-month 2026 holdout (train: 2024-01..2025-09, val: 2025-10..2025-12, test: 2026-01..2026-11).

    Provides ~45 non-overlapping 32-hour market blocks across all of 2026 so time-block bootstrap
    and Diebold-Mariano tests have strong statistical power across many distinct market regimes.
    """
    p_val = hf_hub_download(TWELVEDATA_REPO, "bars_1h/val.parquet", repo_type="dataset", revision=TWELVEDATA_REV)
    p_test = hf_hub_download(TWELVEDATA_REPO, "bars_1h/test.parquet", repo_type="dataset", revision=TWELVEDATA_REV)
    cols = [
        "symbol",
        "datetime",
        "rv_20",
        "volume",
        "vix_level",
        "spy_logret_1",
        "sector_logret_1",
        "atr_14",
    ]
    df = pl.concat([pl.read_parquet(p_val, columns=cols), pl.read_parquet(p_test, columns=cols)])
    df = df.sort(["symbol", "datetime"]).with_columns(
        pl.col("datetime").dt.convert_time_zone("UTC").alias("dt_utc"),
        pl.col("volume").clip(lower_bound=0.0).log1p().cast(pl.Float32).alias("log_volume"),
        pl.col("datetime").dt.hour().cast(pl.Float32).alias("hour_of_day"),
        pl.col("datetime").dt.weekday().cast(pl.Float32).alias("day_of_week"),
    )

    test_cutoff_ns = int(np.datetime64("2026-01-02T00:00:00", "ns").astype(np.int64))
    series_list: list[SeriesData] = []
    for sym, grp in df.group_by("symbol", maintain_order=True):
        sym_name = str(sym[0] if isinstance(sym, tuple) else sym)
        n = grp.height
        dt_ns = grp["dt_utc"].dt.epoch("ns").to_numpy()
        test_pos = int(np.searchsorted(dt_ns, test_cutoff_ns))
        if test_pos < context_length + 500 or (n - test_pos) < forecast_horizon:
            continue
        bar_idx = np.arange(n, dtype=np.int64) - test_pos
        series_list.append(
            SeriesData(
                series_id=sym_name,
                timestamps=bar_idx,
                targets=grp["rv_20"].to_numpy().astype(np.float32)[None, :],
                past=np.stack(
                    [
                        grp["log_volume"].to_numpy().astype(np.float32),
                        grp["vix_level"].to_numpy().astype(np.float32),
                        grp["spy_logret_1"].to_numpy().astype(np.float32),
                        grp["sector_logret_1"].to_numpy().astype(np.float32),
                        grp["atr_14"].to_numpy().astype(np.float32),
                    ],
                    axis=0,
                ),
                known=np.stack(
                    [
                        grp["hour_of_day"].to_numpy().astype(np.float32),
                        grp["day_of_week"].to_numpy().astype(np.float32),
                    ],
                    axis=0,
                ),
                categorical_past=np.empty((0, n), dtype=np.float64),
                categorical_known=np.empty((0, n), dtype=np.float64),
            )
        )

    roles = ColumnRoles(
        targets=("rv_20",),
        past=("log_volume", "vix_level", "spy_logret_1", "sector_logret_1", "atr_14"),
        known=("hour_of_day", "day_of_week"),
    )
    val_start_bar = -450  # ~3 months of late-2025 hourly trading bars
    full_train = TimeSeriesDataset(
        series_list,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=train_stride,
        min_context=context_length,
        roles=roles,
    )
    full_eval = TimeSeriesDataset(
        series_list,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=eval_stride,
        min_context=context_length,
        roles=roles,
    )
    train_ds, _ = full_train.split_by_time(cutoff=val_start_bar)
    pre_test, test_ds = full_eval.split_by_time(cutoff=0)
    _, val_ds = pre_test.split_by_time(cutoff=val_start_bar)
    return BenchmarkSplits(
        name="realised_volatility_2026_10m",
        license="MIT",
        source=f"{TWELVEDATA_REPO}@{TWELVEDATA_REV} (bars_1h, full 2026 holdout)",
        contamination_note=(
            "Trained strictly on 2024-2025. Test spans Jan-Nov 2026 (mostly pre-release: Jan-Oct "
            "precedes Timestral's 2026-10-27 release; only Nov 3-20 is strictly post-release). "
            "Used for multi-regime statistical power."
        ),
        test_date_range="2026-01 to 2026-11-20",
        seasonality=7,
        train=train_ds,
        val=val_ds,
        test=test_ds,
    )


def _window_components(
    preds: np.ndarray,
    actuals: np.ndarray,
    contexts: np.ndarray,
    seasonality: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-window `(2*sum_pinball, sum_|y|, window_mase)` arrays for exact clustered bootstrap."""
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
    med_i = levels.tolist().index(0.5)
    n = preds.shape[0]
    num = np.empty(n, dtype=np.float64)
    den = np.empty(n, dtype=np.float64)
    mase_i = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        v = np.isfinite(actuals[i])
        yv = actuals[i, v]
        diff = yv[:, None] - preds[i, v]
        pb = np.maximum(levels[None, :] * diff, (levels[None, :] - 1.0) * diff)
        num[i] = float(2.0 * np.mean(np.sum(pb, axis=0)))
        den[i] = float(np.sum(np.abs(yv)))
        cd = contexts[i, seasonality:] - contexts[i, :-seasonality]
        cv = np.isfinite(cd)
        scale = float(np.mean(np.abs(cd[cv]))) if cv.any() else 0.0
        if scale > 0.0:
            mase_i[i] = float(np.mean(np.abs(yv - preds[i, v, med_i])) / scale)
    return num, den, mase_i


def _clustered_paired_stats(
    preds_tuned: np.ndarray,
    preds_base: np.ndarray,
    actuals: np.ndarray,
    contexts: np.ndarray,
    cluster_ids: np.ndarray,
    seasonality: int,
    *,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Cluster-level paired bootstrap 95% CI + Diebold-Mariano z/p on ΔwQL and ΔMASE."""
    num_t, den_t, mase_t = _window_components(preds_tuned, actuals, contexts, seasonality)
    num_b, den_b, mase_b = _window_components(preds_base, actuals, contexts, seasonality)

    uniq = np.unique(cluster_ids)
    k = uniq.size
    c_num_t = np.empty(k, dtype=np.float64)
    c_num_b = np.empty(k, dtype=np.float64)
    c_den = np.empty(k, dtype=np.float64)
    c_dmase = np.empty(k, dtype=np.float64)
    c_dwql = np.empty(k, dtype=np.float64)
    for idx, cid in enumerate(uniq):
        mask = cluster_ids == cid
        c_num_t[idx] = num_t[mask].sum()
        c_num_b[idx] = num_b[mask].sum()
        c_den[idx] = den_t[mask].sum()
        c_dwql[idx] = (c_num_t[idx] - c_num_b[idx]) / max(c_den[idx], 1e-30)
        valid_m = np.isfinite(mase_t[mask]) & np.isfinite(mase_b[mask])
        c_dmase[idx] = float(np.mean(mase_t[mask][valid_m] - mase_b[mask][valid_m])) if valid_m.any() else np.nan

    obs_dwql = float((c_num_t.sum() - c_num_b.sum()) / c_den.sum())
    obs_dmase = float(np.nanmean(mase_t - mase_b))
    win_clusters = int(np.sum(c_dwql < 0.0))

    # Diebold-Mariano / paired cluster t-test on cluster-level ΔwQL (with lag-1 Newey-West variance).
    d = c_dwql
    d_mean = float(np.mean(d))
    d_c = d - d_mean
    gamma0 = float(np.mean(d_c**2))
    gamma1 = float(np.mean(d_c[1:] * d_c[:-1])) if k > 2 else 0.0
    var_nw = max(gamma0 + 2.0 * 0.5 * gamma1, gamma0 * 0.25, 1e-30)
    dm_stat = float(d_mean / math.sqrt(var_nw / k)) if k > 1 else float("nan")
    # Two-sided normal p-value (exact asymptotic DM)
    dm_p = float(math.erfc(abs(dm_stat) / math.sqrt(2.0))) if math.isfinite(dm_stat) else float("nan")

    # Paired cluster bootstrap (10,000 resamples).
    rng = np.random.default_rng(seed)
    idx_draw = rng.integers(0, k, size=(n_boot, k))
    b_dwql = (c_num_t[idx_draw].sum(axis=1) - c_num_b[idx_draw].sum(axis=1)) / c_den[idx_draw].sum(axis=1)
    b_dmase = np.nanmean(c_dmase[idx_draw], axis=1)
    boot_p_wql = float(2.0 * min(np.mean(b_dwql >= 0.0), np.mean(b_dwql <= 0.0)))
    return {
        "n_clusters": int(k),
        "clusters_improved_wql": win_clusters,
        "obs_delta_wql": obs_dwql,
        "ci95_delta_wql": [float(np.quantile(b_dwql, 0.025)), float(np.quantile(b_dwql, 0.975))],
        "boot_p_wql": boot_p_wql,
        "dm_stat_wql": dm_stat,
        "dm_p_wql": dm_p,
        "obs_delta_mase": obs_dmase,
        "ci95_delta_mase": [float(np.quantile(b_dmase, 0.025)), float(np.quantile(b_dmase, 0.975))],
    }


def _seasonal_naive_baseline(splits: BenchmarkSplits, *, clip_nonnegative: bool) -> dict[str, Any]:
    """Seasonal naive median + empirical per-horizon validation residual quantiles."""
    s = splits.seasonality
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)

    def _point_and_y(ds: TimeSeriesDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = ds.forecast_horizon
        n = len(ds)
        pts = np.empty((n, h), dtype=np.float64)
        ys = np.empty((n, h), dtype=np.float64)
        for i in range(n):
            w = ds[i]
            ctx = w.target_context[0].astype(np.float64)
            tail = ctx[-s:]
            pts[i] = np.resize(tail, h)
            ys[i] = w.target_future[0].astype(np.float64)
        return pts, ys, _collect_contexts(ds)

    val_pt, val_y, _ = _point_and_y(splits.val)
    test_pt, test_y, test_ctx = _point_and_y(splits.test)
    resid = val_y - val_pt  # (N_val, H)
    # Empirical quantile offsets per horizon step h from validation residuals
    q_off = np.quantile(resid, levels, axis=0).T  # (H, Q)
    test_preds = test_pt[:, :, None] + q_off[None, :, :]
    if clip_nonnegative:
        test_preds = np.maximum(test_preds, 0.0)
    return evaluate_predictions(test_preds, test_y, test_ctx, seasonality=s).to_dict()


def _ridge_har_or_seasonal_baseline(
    splits: BenchmarkSplits,
    *,
    kind: str,
    clip_nonnegative: bool = True,
    alpha: float = 10.0,
) -> dict[str, Any]:
    """Direct multi-horizon Ridge regression (HAR-RV + covariates for volatility; Seasonal+Covariates for retail)."""
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
    h = splits.test.forecast_horizon

    def _extract_xy(ds: TimeSeriesDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats: list[np.ndarray] = []
        ys = np.empty((len(ds), h), dtype=np.float64)
        for i in range(len(ds)):
            w = ds[i]
            ctx = w.target_context[0].astype(np.float64)
            ys[i] = w.target_future[0].astype(np.float64)
            if kind == "har_rv":
                # Corsi (2009) HAR lags: last bar, 1 trading day (7), 1 trading week (35), 1 trading month (140)
                f = [
                    ctx[-1],
                    float(np.mean(ctx[-7:])),
                    float(np.mean(ctx[-35:])),
                    float(np.mean(ctx[-140:])),
                    float(np.std(ctx[-35:])),
                ]
                # Past covariates: latest values + 7-bar means
                for row in w.past:
                    arr = np.nan_to_num(row.astype(np.float64), nan=0.0)
                    f.extend([arr[-1], float(np.mean(arr[-7:]))])
                # Known future calendar at forecast origin & horizon midpoint
                for row in w.known:
                    arr = np.nan_to_num(row.astype(np.float64), nan=0.0)
                    f.extend([arr[ctx.size], arr[min(arr.size - 1, ctx.size + h // 2)]])
            else:
                # Retail seasonal lags (1h, 24h, 48h, 168h) + 24h profile + past & known covariates
                f = [
                    ctx[-1],
                    ctx[-24],
                    ctx[-48],
                    ctx[-168],
                    float(np.mean(ctx[-24:])),
                    float(np.mean(ctx[-168:])),
                    float(np.std(ctx[-168:])),
                ]
                # Full last 24h seasonal profile
                f.extend(ctx[-24:].tolist())
                # Past covariates: recent 24h means
                for row in w.past:
                    arr = np.nan_to_num(row.astype(np.float64), nan=0.0)
                    f.extend([arr[-1], float(np.mean(arr[-24:]))])
                # Known future covariates: mean over forecast horizon + day-1/day-2 values
                for row in w.known:
                    arr = np.nan_to_num(row.astype(np.float64), nan=0.0)
                    fut = arr[ctx.size : ctx.size + h]
                    f.extend([float(np.mean(fut)), float(fut[0]), float(fut[-1])])
            feats.append(np.asarray(f, dtype=np.float64))
        return np.stack(feats, axis=0), ys, _collect_contexts(ds)

    x_tr, y_tr, _ = _extract_xy(splits.train)
    x_va, y_va, _ = _extract_xy(splits.val)
    x_te, y_te, ctx_te = _extract_xy(splits.test)

    # Standardise features on train only (zero leakage!).
    mu = x_tr.mean(axis=0, keepdims=True)
    sd = np.where(x_tr.std(axis=0, keepdims=True) > 1e-12, x_tr.std(axis=0, keepdims=True), 1.0)
    x_tr_s = np.concatenate([(x_tr - mu) / sd, np.ones((x_tr.shape[0], 1))], axis=1)
    x_va_s = np.concatenate([(x_va - mu) / sd, np.ones((x_va.shape[0], 1))], axis=1)
    x_te_s = np.concatenate([(x_te - mu) / sd, np.ones((x_te.shape[0], 1))], axis=1)

    # Direct multi-horizon Ridge: W = (X^T X + alpha I)^{-1} X^T Y (do not penalise intercept).
    reg = np.eye(x_tr_s.shape[1]) * alpha
    reg[-1, -1] = 0.0
    w_hat = np.linalg.solve(x_tr_s.T @ x_tr_s + reg, x_tr_s.T @ y_tr)  # (D, H)

    val_pt = x_va_s @ w_hat
    test_pt = x_te_s @ w_hat
    q_off = np.quantile(y_va - val_pt, levels, axis=0).T  # (H, Q)
    test_preds = test_pt[:, :, None] + q_off[None, :, :]
    if clip_nonnegative:
        test_preds = np.maximum(test_preds, 0.0)
    return evaluate_predictions(test_preds, y_te, ctx_te, seasonality=splits.seasonality).to_dict()


def _train_and_predict(
    base_cpu,
    splits: BenchmarkSplits,
    *,
    mode: str,
    lr: float,
    device: str,
    mixed_precision: str,
    max_steps: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = copy.deepcopy(base_cpu)
    lora = (
        LoraConfig(
            r=8,
            lora_alpha=16.0,
            axes=("variate" if mode == "lora_variate" else "time",),
            targets=("q", "k", "v", "o"),
            dropout=0.05,
        )
        if mode.startswith("lora_")
        else None
    )
    cfg = TrainingConfig(
        model_name_or_path="mistralai/Timestral-28M",
        mode=mode,  # type: ignore[arg-type]
        learning_rate=lr,
        batch_size=batch_size,
        max_steps=max_steps,
        warmup_steps=max(10, max_steps // 10),
        mixed_precision=mixed_precision,  # type: ignore[arg-type]
        device=device,
        seed=seed,
        eval_every=max(20, max_steps // 5),
        restore_best=True,
        sign_flip=True,
        log_every=max_steps + 1,
    )
    tr = TimestralTrainer(cfg, splits.train, splits.val, lora_config=lora, model=m)
    tr.fit()
    ev_m = copy.deepcopy(base_cpu)
    ev_m.load_state_dict(merged_state_dict(tr.model), strict=True)
    del tr, m
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    ev_m.to(device).eval()
    ev_m.transformer.rope.rebuild_inv_freq()
    preds, actuals, _ = collect_conformal_data(TuneTimestralForecaster(ev_m), splits.test, batch_size=batch_size)
    del ev_m
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return preds, actuals, _collect_contexts(splits.test)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    precision = "bf16" if args.bf16 else "fp32"
    t0 = time.perf_counter()
    print("== Loading 28M and datasets (post-Oct vol, 10-month 2026 vol, retail) ==", flush=True)
    base_cpu = load_model("mistralai/Timestral-28M", device="cpu")
    vol_post = load_realised_volatility()
    vol_10m = load_realised_volatility_2026_holdout()
    retail = load_retail_demand()
    print(
        f"Loaded: vol_post(te={len(vol_post.test)}), vol_10m(tr={len(vol_10m.train)}, "
        f"va={len(vol_10m.val)}, te={len(vol_10m.test)}), retail(te={len(retail.test)})",
        flush=True,
    )

    out: dict[str, Any] = {"model": "mistralai/Timestral-28M", "seeds": args.seeds, "datasets": {}}

    for splits, baseline_kind in [
        (vol_post, "har_rv"),
        (vol_10m, "har_rv"),
        (retail, "seasonal_ridge"),
    ]:
        print(f"\n================ {splits.name} (te={len(splits.test)}) ================", flush=True)
        # 1. Classical baselines
        sn = _seasonal_naive_baseline(splits, clip_nonnegative=True)
        cl = _ridge_har_or_seasonal_baseline(splits, kind=baseline_kind, clip_nonnegative=True)
        print(f"  [baseline] seasonal_naive | wQL={sn['wql']:.4f} MASE={sn['mase']:.4f} cov80={sn['coverage_80']:.3f}")
        print(
            f"  [baseline] {baseline_kind:14s} | "
            f"wQL={cl['wql']:.4f} MASE={cl['mase']:.4f} cov80={cl['coverage_80']:.3f}",
            flush=True,
        )

        # 2. Zero-shot predictions
        zs_m = copy.deepcopy(base_cpu).to(args.device).eval()
        zs_m.transformer.rope.rebuild_inv_freq()
        zs_preds, ys, _ = collect_conformal_data(TuneTimestralForecaster(zs_m), splits.test, batch_size=args.batch_size)
        ctxs = _collect_contexts(splits.test)
        del zs_m
        if args.device == "cuda":
            torch.cuda.empty_cache()
        zs_metrics = evaluate_predictions(zs_preds, ys, ctxs, seasonality=splits.seasonality).to_dict()
        print(
            f"  [28M] zero_shot          | wQL={zs_metrics['wql']:.4f} "
            f"MASE={zs_metrics['mase']:.4f} cov80={zs_metrics['coverage_80']:.3f}",
            flush=True,
        )

        # Extract cluster IDs for BOTH series-level and time-block-level paired tests.
        series_ids = np.array([splits.test._series_index[i] for i in range(len(splits.test))])
        # Bin origins into exact time blocks (each series shares the same origin coordinate grid).
        origin_times = np.array(
            [
                splits.test.series[splits.test._series_index[i]].timestamps[splits.test._origins[i]]
                for i in range(len(splits.test))
            ]
        )

        # 3. Multi-seed runs for lora_variate, lora_temporal, full_ft
        modes_data: dict[str, Any] = {}
        mode_list = (
            [("lora_variate", 1e-3), ("lora_temporal", 1e-3), ("full_ft", 5e-5)]
            if splits is not vol_10m
            else [("lora_variate", 1e-3), ("lora_temporal", 1e-3), ("full_ft", 5e-5)]
        )
        run_seeds = args.seeds if splits is not vol_10m else [0]
        for mode, lr in mode_list:
            seed_rows = []
            first_preds = None
            for sd in run_seeds:
                p, _, _ = _train_and_predict(
                    base_cpu,
                    splits,
                    mode=mode,
                    lr=lr,
                    device=args.device,
                    mixed_precision=precision,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    seed=sd,
                )
                if first_preds is None:
                    first_preds = p
                met = evaluate_predictions(p, ys, ctxs, seasonality=splits.seasonality).to_dict()
                seed_rows.append({"seed": sd, "metrics": met})
            wqls = [r["metrics"]["wql"] for r in seed_rows]
            mases = [r["metrics"]["mase"] for r in seed_rows]
            by_series = _clustered_paired_stats(first_preds, zs_preds, ys, ctxs, series_ids, splits.seasonality)
            by_time = _clustered_paired_stats(first_preds, zs_preds, ys, ctxs, origin_times, splits.seasonality)
            modes_data[mode] = {
                "seeds": seed_rows,
                "wql_mean": float(np.mean(wqls)),
                "wql_std": float(np.std(wqls, ddof=1)) if len(wqls) > 1 else 0.0,
                "mase_mean": float(np.mean(mases)),
                "mase_std": float(np.std(mases, ddof=1)) if len(mases) > 1 else 0.0,
                "paired_by_series": by_series,
                "paired_by_time_block": by_time,
            }
            print(
                f"  [28M] {mode:14s} | wQL={np.mean(wqls):.4f}±{modes_data[mode]['wql_std']:.4f} "
                f"MASE={np.mean(mases):.4f}±{modes_data[mode]['mase_std']:.4f} | "
                f"series CI95=[{by_series['ci95_delta_wql'][0]:+.4f},{by_series['ci95_delta_wql'][1]:+.4f}] "
                f"(p={by_series['dm_p_wql']:.2e}, {by_series['clusters_improved_wql']}/{by_series['n_clusters']}) | "
                f"time CI95=[{by_time['ci95_delta_wql'][0]:+.4f},{by_time['ci95_delta_wql'][1]:+.4f}] "
                f"(DM p={by_time['dm_p_wql']:.2e}, {by_time['clusters_improved_wql']}/{by_time['n_clusters']})",
                flush=True,
            )

        out["datasets"][splits.name] = {
            "baselines": {"seasonal_naive": sn, baseline_kind: cl},
            "zero_shot": zs_metrics,
            "modes": modes_data,
        }

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved significance report to {args.output} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
