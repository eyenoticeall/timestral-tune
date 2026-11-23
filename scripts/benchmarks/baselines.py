"""Validation-tuned classical and gradient-boosted tree baselines sharing exact BenchmarkSplits.

Requires `scikit-learn` (e.g. `uv run --with scikit-learn==1.6.1 ...`) for `run_tuned_gbt`.
Every baseline receives the exact same target history, past covariates, and known-future
covariates as Timestral, tunes all hyperparameters strictly on `splits.val` (never on test),
and emits full 21-quantile raw-space predictions scored by `evaluate_decision_metrics` and
`evaluate_with_calibrators`.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from datasets import BenchmarkSplits
from decision_metrics import evaluate_with_calibrators

from timestral_tune.data import TimeSeriesDataset
from timestral_tune.models import DEFAULT_QUANTILE_LEVELS
from timestral_tune.utils import weighted_quantile_loss


def _context_scales(dataset: TimeSeriesDataset) -> np.ndarray:
    """Compute scale-invariant per-window context std matching `collect_conformal_data`."""
    scales = np.empty(len(dataset), dtype=np.float64)
    for i in range(len(dataset)):
        ctx = dataset[i].target_context[0].astype(np.float64)
        valid = ctx[np.isfinite(ctx)]
        if valid.size == 0:
            scales[i] = 1.0
            continue
        sd = float(np.std(valid))
        max_abs = float(np.max(np.abs(valid)))
        if sd > 0.0 and sd > 1e-12 * max_abs:
            scales[i] = sd
        elif max_abs > 0.0:
            scales[i] = max_abs
        else:
            scales[i] = 1.0
    return scales


def _targets_array(dataset: TimeSeriesDataset) -> np.ndarray:
    return np.stack([dataset[i].target_future[0].astype(np.float64) for i in range(len(dataset))], axis=0)


def run_seasonal_naive(
    splits: BenchmarkSplits,
    *,
    clip_nonnegative: bool = True,
    patch_size: int = 32,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Seasonal naive point forecast + empirical per-horizon validation residual quantiles."""
    t0 = time.perf_counter()
    s = splits.seasonality
    h = splits.test.forecast_horizon
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)

    def _pt(ds: TimeSeriesDataset) -> tuple[np.ndarray, np.ndarray]:
        pts = np.empty((len(ds), h), dtype=np.float64)
        ys = np.empty((len(ds), h), dtype=np.float64)
        for i in range(len(ds)):
            w = ds[i]
            ctx = np.nan_to_num(w.target_context[0].astype(np.float64), nan=0.0)
            pts[i] = np.resize(ctx[-s:], h)
            ys[i] = w.target_future[0].astype(np.float64)
        return pts, ys

    val_pt, val_y = _pt(splits.val)
    test_pt, test_y = _pt(splits.test)
    q_off = np.nanquantile(val_y - val_pt, levels, axis=0).T  # (H, Q)
    val_preds = val_pt[:, :, None] + q_off[None, :, :]
    test_preds = test_pt[:, :, None] + q_off[None, :, :]
    if clip_nonnegative:
        val_preds = np.maximum(val_preds, 0.0)
        test_preds = np.maximum(test_preds, 0.0)

    val_scale = _context_scales(splits.val)
    test_scale = _context_scales(splits.test)
    metrics, preds_dict = evaluate_with_calibrators(
        val_preds,
        val_y,
        val_scale,
        test_preds,
        test_y,
        test_scale,
        splits.test,
        seasonality=s,
        patch_size=patch_size,
    )
    return {"elapsed_s": round(time.perf_counter() - t0, 2), "seasonality": s, "metrics": metrics}, preds_dict


def _build_stacked_features(
    dataset: TimeSeriesDataset,
    *,
    seasonality: int,
    dataset_name: str,
    include_series_id: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build `(N * H, D)` step-aligned feature matrix where every `(window, h)` gets exact `known[T + h]`.

    Returns `(X, y_flat, mu_flat, sd_flat, feature_names)` where `X` has column 0 as integer
    `series_idx` when `include_series_id=True` (for native categorical splits in GBT).
    """
    n = len(dataset)
    h = dataset.forecast_horizon
    s = seasonality
    w0 = dataset[0]
    n_past = w0.past.shape[0]
    n_known = w0.known.shape[0]
    is_de_power = dataset_name == "de_power_15m" and n_known >= 1

    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    rows_mu: list[np.ndarray] = []
    rows_sd: list[np.ndarray] = []

    h_idx = np.arange(h, dtype=np.float64)
    for i in range(n):
        w = dataset[i]
        sidx = float(dataset._series_index[i])
        ctx_raw = w.target_context[0].astype(np.float64)
        finite_ctx = ctx_raw[np.isfinite(ctx_raw)]
        mu_ctx = float(np.mean(finite_ctx)) if finite_ctx.size else 0.0
        sd_ctx = float(np.std(finite_ctx)) if finite_ctx.size else 1.0
        if not (sd_ctx > 1e-12 * max(abs(mu_ctx), 1.0)):
            sd_ctx = max(abs(mu_ctx), 1.0)
        ctx = np.nan_to_num(ctx_raw, nan=mu_ctx)
        t_len = ctx.size

        # Window-level scalar features (replicated across all H lead steps).
        short_w = min(7, t_len)
        seas_w = min(s, t_len)
        w_feats = [
            mu_ctx,
            sd_ctx,
            ctx[-1],
            ctx[-min(2, t_len)],
            ctx[-min(3, t_len)],
            ctx[-min(4, t_len)],
            ctx[-short_w],
            ctx[-seas_w],
            ctx[-min(2 * s, t_len)],
            float(np.mean(ctx[-short_w:])),
            float(np.std(ctx[-short_w:])),
            float(np.mean(ctx[-seas_w:])),
            float(np.std(ctx[-seas_w:])),
            float(np.min(ctx[-seas_w:])),
            float(np.max(ctx[-seas_w:])),
            float(np.quantile(ctx, 0.10)),
            float(np.quantile(ctx, 0.50)),
            float(np.quantile(ctx, 0.90)),
            float(np.mean(ctx[-seas_w:]) - np.mean(ctx[-min(2 * s, t_len) : -seas_w])) if t_len >= 2 * s else 0.0,
        ]
        # Domain-specific Corsi HAR lags for realised volatility
        if "volatility" in dataset_name:
            w_feats.extend(
                [
                    float(np.mean(ctx[-min(35, t_len) :])),
                    float(np.std(ctx[-min(35, t_len) :])),
                    float(np.mean(ctx[-min(140, t_len) :])),
                ]
            )
        # Past covariates: latest, short mean, seasonal mean & std
        for r in range(n_past):
            arr = np.nan_to_num(w.past[r].astype(np.float64), nan=0.0)
            w_feats.extend(
                [
                    arr[-1],
                    float(np.mean(arr[-short_w:])),
                    float(np.mean(arr[-seas_w:])),
                    float(np.std(arr[-seas_w:])),
                ]
            )
        # Known covariates window-level summaries (origin + horizon mean + recent forecast error for DE power)
        for r in range(n_known):
            arr = np.nan_to_num(w.known[r].astype(np.float64), nan=0.0)
            fut = arr[t_len : t_len + h]
            w_feats.extend([arr[t_len - 1], float(np.mean(fut)), float(np.std(fut))])
        if is_de_power:
            da_ctx = np.nan_to_num(w.known[0, :t_len].astype(np.float64), nan=0.0)
            err = ctx - da_ctx
            w_feats.extend(
                [
                    err[-1],
                    float(np.mean(err[-4:])),
                    float(np.mean(err[-96:])),
                    float(np.std(err[-96:])),
                ]
            )

        # Step-varying features (H, K_step): lead time h, seasonal lags, exact known[T + h], and DA ramp.
        seas_lag = np.array([ctx[-s + (step % s)] if t_len >= s else ctx[-1] for step in range(h)], dtype=np.float64)
        seas_lag2 = np.array(
            [ctx[-2 * s + (step % s)] if t_len >= 2 * s else seas_lag[step] for step in range(h)],
            dtype=np.float64,
        )
        step_cols = [h_idx, seas_lag, seas_lag2]
        for r in range(n_known):
            arr = np.nan_to_num(w.known[r].astype(np.float64), nan=0.0)
            fut = arr[t_len : t_len + h]
            step_cols.append(fut)
            if is_de_power and r == 0:
                # Step-to-step ramp of the day-ahead TSO forecast
                prev_da = np.concatenate([[arr[t_len - 1]], fut[:-1]])
                step_cols.append(fut - prev_da)
                step_cols.append(fut - mu_ctx)

        base_mat = np.broadcast_to(np.asarray(w_feats, dtype=np.float64)[None, :], (h, len(w_feats)))
        step_mat = np.stack(step_cols, axis=1)
        if include_series_id:
            sid_col = np.full((h, 1), sidx, dtype=np.float64)
            x_i = np.concatenate([sid_col, step_mat, base_mat], axis=1)
        else:
            x_i = np.concatenate([step_mat, base_mat], axis=1)

        rows_x.append(x_i)
        rows_y.append(w.target_future[0].astype(np.float64))
        rows_mu.append(np.full(h, mu_ctx, dtype=np.float64))
        rows_sd.append(np.full(h, sd_ctx, dtype=np.float64))

    X = np.concatenate(rows_x, axis=0)
    y_flat = np.concatenate(rows_y, axis=0)
    mu_flat = np.concatenate(rows_mu, axis=0)
    sd_flat = np.concatenate(rows_sd, axis=0)
    names = (["series_idx"] if include_series_id else []) + [
        f"f_{j}" for j in range(X.shape[1] - int(include_series_id))
    ]
    return X, y_flat, mu_flat, sd_flat, names


def run_tuned_classical(
    splits: BenchmarkSplits,
    *,
    clip_nonnegative: bool = True,
    patch_size: int = 32,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Direct per-horizon Ridge ARX with exact step-aligned known covariates and per-series intercepts.

    Tunes `(target_transform in {raw, log1p}, alpha in {0.01..10000})` strictly on validation wQL,
    then fits empirical per-horizon validation residual quantiles (and optional CQR/ACI).
    """
    t0 = time.perf_counter()
    h = splits.test.forecast_horizon
    s = splits.seasonality
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
    n_series = len(splits.train.series)

    x_tr, y_tr_f, _, _, _ = _build_stacked_features(
        splits.train, seasonality=s, dataset_name=splits.name, include_series_id=True
    )
    x_va, y_va_f, _, _, _ = _build_stacked_features(
        splits.val, seasonality=s, dataset_name=splits.name, include_series_id=True
    )
    x_te, y_te_f, _, _, _ = _build_stacked_features(
        splits.test, seasonality=s, dataset_name=splits.name, include_series_id=True
    )
    # Reshape back to (N, H, D) for exact per-horizon direct regression
    n_tr, n_va, n_te = len(splits.train), len(splits.val), len(splits.test)
    d = x_tr.shape[1]
    X_tr = x_tr.reshape(n_tr, h, d)
    X_va = x_va.reshape(n_va, h, d)
    X_te = x_te.reshape(n_te, h, d)
    Y_tr = y_tr_f.reshape(n_tr, h)
    Y_va = y_va_f.reshape(n_va, h)
    Y_te = y_te_f.reshape(n_te, h)

    # One-hot series fixed effects from column 0, numeric features from columns 1: (drop constant lead-time h col 1)
    def _design(X_mat: np.ndarray, step: int, mu: np.ndarray | None, sd: np.ndarray | None):
        sidx = X_mat[:, step, 0].astype(np.int64)
        num = X_mat[:, step, 2:]
        if mu is None or sd is None:
            mu = num.mean(axis=0, keepdims=True)
            sd = np.where(num.std(axis=0, keepdims=True) > 1e-12, num.std(axis=0, keepdims=True), 1.0)
        num_s = (num - mu) / sd
        oh = np.zeros((X_mat.shape[0], n_series), dtype=np.float64)
        valid_s = (sidx >= 0) & (sidx < n_series)
        oh[np.arange(X_mat.shape[0])[valid_s], sidx[valid_s]] = 1.0
        design = np.concatenate([num_s, oh, np.ones((X_mat.shape[0], 1), dtype=np.float64)], axis=1)
        return design, mu, sd

    # Precompute standardised per-horizon design matrices and Gram matrices once
    tr_designs, va_designs, te_designs = [], [], []
    grams = []
    for step in range(h):
        d_tr, mu_s, sd_s = _design(X_tr, step, None, None)
        d_va, _, _ = _design(X_va, step, mu_s, sd_s)
        d_te, _, _ = _design(X_te, step, mu_s, sd_s)
        tr_designs.append(d_tr)
        va_designs.append(d_va)
        te_designs.append(d_te)
        grams.append(d_tr.T @ d_tr)

    d_tot = tr_designs[0].shape[1]
    eye = np.eye(d_tot, dtype=np.float64)
    eye[-1, -1] = 0.0  # do not penalise intercept

    transforms = ("raw", "log1p") if (Y_tr >= 0.0).all() and (Y_va >= 0.0).all() else ("raw",)
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
    best_cfg: tuple[str, float] | None = None
    best_val_wql = float("inf")
    best_val_preds = None
    best_test_preds = None
    tuning_log: list[dict[str, Any]] = []

    for tf in transforms:
        y_tr_t = np.log1p(Y_tr) if tf == "log1p" else Y_tr
        y_va_t = np.log1p(Y_va) if tf == "log1p" else Y_va
        rhs = [tr_designs[step].T @ y_tr_t[:, step] for step in range(h)]
        cap_hi = float(np.max(y_tr_t) + 2.0) if tf == "log1p" else float("inf")

        for alpha in alphas:
            val_pt_t = np.empty((n_va, h), dtype=np.float64)
            test_pt_t = np.empty((n_te, h), dtype=np.float64)
            for step in range(h):
                w_h = np.linalg.solve(grams[step] + alpha * eye, rhs[step])
                val_pt_t[:, step] = va_designs[step] @ w_h
                test_pt_t[:, step] = te_designs[step] @ w_h

            # Empirical per-horizon residual quantiles fitted on train residuals so val scoring is strictly OOS.
            tr_pt_t = np.empty((n_tr, h), dtype=np.float64)
            for step in range(h):
                w_h = np.linalg.solve(grams[step] + alpha * eye, rhs[step])
                tr_pt_t[:, step] = tr_designs[step] @ w_h
            q_off_tr = np.nanquantile(y_tr_t - tr_pt_t, levels, axis=0).T  # (H, Q)
            val_q = val_pt_t[:, :, None] + q_off_tr[None, :, :]
            if tf == "log1p":
                val_q = np.expm1(np.clip(val_q, 0.0, cap_hi))
            elif clip_nonnegative:
                val_q = np.maximum(val_q, 0.0)
            val_wql = weighted_quantile_loss(val_q, Y_va)
            tuning_log.append({"transform": tf, "alpha": alpha, "val_wql": round(val_wql, 6)})

            if val_wql < best_val_wql:
                best_val_wql = val_wql
                best_cfg = (tf, alpha)
                # Final test residual quantiles calibrated on validation residuals (matching significance.py)
                q_off_va = np.nanquantile(y_va_t - val_pt_t, levels, axis=0).T
                vp = val_pt_t[:, :, None] + q_off_tr[None, :, :]
                tp = test_pt_t[:, :, None] + q_off_va[None, :, :]
                if tf == "log1p":
                    vp = np.expm1(np.clip(vp, 0.0, cap_hi))
                    tp = np.expm1(np.clip(tp, 0.0, cap_hi))
                elif clip_nonnegative:
                    vp = np.maximum(vp, 0.0)
                    tp = np.maximum(tp, 0.0)
                best_val_preds = np.sort(vp, axis=-1)
                best_test_preds = np.sort(tp, axis=-1)

    assert best_cfg is not None and best_val_preds is not None and best_test_preds is not None
    val_scale = _context_scales(splits.val)
    test_scale = _context_scales(splits.test)
    metrics, preds_dict = evaluate_with_calibrators(
        best_val_preds,
        Y_va,
        val_scale,
        best_test_preds,
        Y_te,
        test_scale,
        splits.test,
        seasonality=s,
        patch_size=patch_size,
    )
    return (
        {
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "best_transform": best_cfg[0],
            "best_alpha": best_cfg[1],
            "best_val_wql": best_val_wql,
            "tuning_grid_size": len(tuning_log),
            "metrics": metrics,
        },
        preds_dict,
    )


def _fit_median_anchor(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    cfg: dict[str, Any],
    *,
    max_iter: int,
    eval_stride: int,
    clip_nonnegative: bool,
    seed: int,
    X_te: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """Fit an L1 (`absolute_error`) median GBT and return `(p50_tr, p50_va, p50_te, best_iter)`."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    reg = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=max_iter,
        max_leaf_nodes=cfg["max_leaf_nodes"],
        min_samples_leaf=cfg["min_samples_leaf"],
        learning_rate=cfg["learning_rate"],
        l2_regularization=cfg["l2_regularization"],
        categorical_features=[0],
        early_stopping=False,
        random_state=seed,
    )
    reg.fit(X_tr, y_tr)

    valid_va = np.isfinite(y_va)
    yv = y_va[valid_va]
    best_loss = float("inf")
    best_iter = max_iter
    best_pv = None
    n_built = int(reg.n_iter_)
    for step, pv in enumerate(reg.staged_predict(X_va), start=1):
        if step % eval_stride != 0 and step != n_built:
            continue
        if clip_nonnegative:
            pv = np.maximum(pv, 0.0)
        loss = float(np.mean(np.abs(yv - pv[valid_va])))
        if loss < best_loss:
            best_loss = loss
            best_iter = step
            best_pv = pv

    assert best_pv is not None
    p50_tr = None
    for step, pt in enumerate(reg.staged_predict(X_tr), start=1):
        if step == best_iter:
            p50_tr = np.maximum(pt, 0.0) if clip_nonnegative else pt
            break
    assert p50_tr is not None

    p50_te: np.ndarray | None = None
    if X_te is not None:
        for step, pt in enumerate(reg.staged_predict(X_te), start=1):
            if step == best_iter:
                p50_te = np.maximum(pt, 0.0) if clip_nonnegative else pt
                break
        assert p50_te is not None
    return p50_tr, best_pv, p50_te, best_iter


def _fit_quantile_with_val_early_stop(
    X_tr: np.ndarray,
    y_fit: np.ndarray,
    X_va: np.ndarray,
    y_va_raw: np.ndarray,
    mu_va: np.ndarray,
    sd_va: np.ndarray,
    *,
    tau: float,
    tf: str,
    cfg: dict[str, Any],
    max_iter: int,
    eval_stride: int,
    clip_nonnegative: bool,
    seed: int,
    X_te: np.ndarray | None = None,
    mu_te: np.ndarray | None = None,
    sd_te: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, int, float]:
    """Fit one quantile GBT and select exact `best_iter` chronologically on `splits.val` via `staged_predict`.

    Supports `tf in {"raw", "context_norm", "median_residual"}` (for `"median_residual"`, `mu_va`
    and `mu_te` carry the median anchor `p50` and `y_fit` is `y_tr - p50_tr`).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    reg = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(tau),
        max_iter=max_iter,
        max_leaf_nodes=cfg["max_leaf_nodes"],
        min_samples_leaf=cfg["min_samples_leaf"],
        learning_rate=cfg["learning_rate"],
        l2_regularization=cfg["l2_regularization"],
        categorical_features=[0],
        early_stopping=False,
        random_state=seed,
    )
    reg.fit(X_tr, y_fit)

    valid_va = np.isfinite(y_va_raw)
    yv = y_va_raw[valid_va]
    best_loss = float("inf")
    best_iter = max_iter
    best_pv = None
    n_built = int(reg.n_iter_)
    for step, pv in enumerate(reg.staged_predict(X_va), start=1):
        if step % eval_stride != 0 and step != n_built:
            continue
        if tf == "context_norm":
            pv = pv * sd_va + mu_va
        elif tf == "median_residual":
            pv = pv + mu_va
        if clip_nonnegative:
            pv = np.maximum(pv, 0.0)
        diff = yv - pv[valid_va]
        loss = float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))
        if loss < best_loss:
            best_loss = loss
            best_iter = step
            best_pv = pv

    assert best_pv is not None
    pt_out: np.ndarray | None = None
    if X_te is not None:
        assert mu_te is not None and sd_te is not None
        for step, pt in enumerate(reg.staged_predict(X_te), start=1):
            if step == best_iter:
                if tf == "context_norm":
                    pt = pt * sd_te + mu_te
                elif tf == "median_residual":
                    pt = pt + mu_te
                if clip_nonnegative:
                    pt = np.maximum(pt, 0.0)
                pt_out = pt
                break
        assert pt_out is not None
    return best_pv, pt_out, best_iter, best_loss


def run_tuned_gbt(
    splits: BenchmarkSplits,
    *,
    clip_nonnegative: bool = True,
    patch_size: int = 32,
    seed: int = 0,
    max_grid_rows: int = 40_000,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Two-stage validation-tuned global stacked quantile GBT with chronological early stopping.

    Stage 1 sweeps 45 broad configs (`target_transform in {raw, context_norm, median_residual}` x
    `(max_leaf_nodes, min_samples_leaf) in {(4,200), (7,100), (15,50), (31,30), (63,20)}` x
    `learning_rate in {0.02, 0.05, 0.10}` at `l2=0.0`, avoiding scikit-learn's `loss="quantile"`
    split freeze under `l2 > 0` on zero-inflated targets) with exact chronological early stopping on
    `splits.val` via `staged_predict` (`1..350` trees). Stage 2 refines the top Stage-1 config of
    each of the three target transforms (plus the overall runner-up if distinct) across
    `l2 in {0.0, 1.0, 10.0, 50.0}` and adjacent leaf/min-leaf regularisation (~85 total configs,
    `1..600` trees). Finally fits all 21 quantiles on 100% of training data with per-quantile
    chronological early stopping (`1..800` trees), reporting `tuning_time_s` and single-model
    `fit_time_s` separately alongside boundary diagnostics.
    """
    t0 = time.perf_counter()
    h = splits.test.forecast_horizon
    s = splits.seasonality
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
    anchors = (0.10, 0.50, 0.90)
    stage1_max_iter = 350
    stage2_max_iter = 600
    final_max_iter = 800

    X_tr, y_tr, mu_tr, sd_tr, _ = _build_stacked_features(
        splits.train, seasonality=s, dataset_name=splits.name, include_series_id=True
    )
    X_va, y_va, mu_va, sd_va, _ = _build_stacked_features(
        splits.val, seasonality=s, dataset_name=splits.name, include_series_id=True
    )
    X_te, y_te, mu_te, sd_te, _ = _build_stacked_features(
        splits.test, seasonality=s, dataset_name=splits.name, include_series_id=True
    )
    n_va, n_te = len(splits.val), len(splits.test)
    Y_va = y_va.reshape(n_va, h)
    Y_te = y_te.reshape(n_te, h)

    valid_tr = np.isfinite(y_tr)
    X_tr_v = X_tr[valid_tr]
    y_tr_v = y_tr[valid_tr]
    mu_tr_v = mu_tr[valid_tr]
    sd_tr_v = sd_tr[valid_tr]

    if X_tr_v.shape[0] > max_grid_rows:
        rng = np.random.default_rng(seed + 77)
        sub_idx = rng.choice(X_tr_v.shape[0], size=max_grid_rows, replace=False)
        X_grid, y_grid, mu_grid, sd_grid = X_tr_v[sub_idx], y_tr_v[sub_idx], mu_tr_v[sub_idx], sd_tr_v[sub_idx]
    else:
        X_grid, y_grid, mu_grid, sd_grid = X_tr_v, y_tr_v, mu_tr_v, sd_tr_v

    def _eval_cfg(cfg: dict[str, Any], max_iter: int) -> tuple[float, int]:
        tf = cfg["target_transform"]
        val_anchor = np.empty((n_va, h, len(anchors)), dtype=np.float64)
        best_iters = []
        if tf == "median_residual":
            p50_tr, p50_va, _, bi50 = _fit_median_anchor(
                X_grid,
                y_grid,
                X_va,
                y_va,
                cfg,
                max_iter=max_iter,
                eval_stride=10,
                clip_nonnegative=clip_nonnegative,
                seed=seed,
            )
            y_fit = y_grid - p50_tr
            anc_va = p50_va
        else:
            y_fit = (y_grid - mu_grid) / sd_grid if tf == "context_norm" else y_grid
            anc_va = mu_va
            bi50 = 0
        for q_i, tau in enumerate(anchors):
            if tf == "median_residual" and abs(tau - 0.50) < 1e-9:
                val_anchor[:, :, q_i] = anc_va.reshape(n_va, h)
                best_iters.append(bi50)
                continue
            pv, _, bi, _ = _fit_quantile_with_val_early_stop(
                X_grid,
                y_fit,
                X_va,
                y_va,
                anc_va,
                sd_va,
                tau=tau,
                tf=tf,
                cfg=cfg,
                max_iter=max_iter,
                eval_stride=10,
                clip_nonnegative=clip_nonnegative,
                seed=seed,
            )
            val_anchor[:, :, q_i] = pv.reshape(n_va, h)
            best_iters.append(bi)
        val_anchor = np.sort(val_anchor, axis=-1)
        if clip_nonnegative:
            val_anchor = np.maximum(val_anchor, 0.0)
        return weighted_quantile_loss(val_anchor, Y_va, quantile_levels=anchors), int(
            round(float(np.median(best_iters)))
        )

    leaf_ladder = ((4, 200), (7, 100), (15, 50), (31, 30), (63, 20))
    stage1_grid = [
        {
            "target_transform": tf,
            "max_leaf_nodes": leaves,
            "min_samples_leaf": min_leaf,
            "learning_rate": lr,
            "l2_regularization": 0.0,
        }
        for tf in ("raw", "context_norm", "median_residual")
        for leaves, min_leaf in leaf_ladder
        for lr in (0.02, 0.05, 0.10)
    ]

    stage1_results: list[tuple[float, int, dict[str, Any]]] = []
    for cfg in stage1_grid:
        wql_a, med_iter = _eval_cfg(cfg, stage1_max_iter)
        stage1_results.append((wql_a, med_iter, cfg))
    stage1_results.sort(key=lambda item: item[0])
    print(
        f"    [gbt stage1 | {splits.name}] best={stage1_results[0][2]} "
        f"(val_wql={stage1_results[0][0]:.5f}, iter={stage1_results[0][1]}, {time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    seen_keys: set[tuple[Any, ...]] = set()
    stage2_grid: list[dict[str, Any]] = []
    leaf_neighbors = {
        4: ((4, 200), (4, 100), (7, 200)),
        7: ((4, 200), (7, 200), (7, 100), (15, 100)),
        15: ((7, 100), (15, 100), (15, 50), (31, 50)),
        31: ((15, 50), (31, 50), (31, 30), (63, 30)),
        63: ((31, 30), (63, 30), (63, 20)),
    }
    stage2_seeds: list[dict[str, Any]] = []
    for tf_name in ("raw", "context_norm", "median_residual"):
        for _, _, cfg in stage1_results:
            if cfg["target_transform"] == tf_name:
                stage2_seeds.append(cfg)
                break
    for _, _, cfg in stage1_results[:2]:
        if cfg not in stage2_seeds:
            stage2_seeds.append(cfg)

    for base_cfg in stage2_seeds:
        for l2 in (0.0, 1.0, 10.0, 50.0):
            cand = {**base_cfg, "l2_regularization": l2}
            k = (
                cand["target_transform"],
                cand["max_leaf_nodes"],
                cand["min_samples_leaf"],
                cand["learning_rate"],
                cand["l2_regularization"],
            )
            if k not in seen_keys:
                seen_keys.add(k)
                stage2_grid.append(cand)
        for nb_leaves, nb_min_leaf in leaf_neighbors[base_cfg["max_leaf_nodes"]]:
            for l2 in (0.0, 10.0):
                cand = {
                    **base_cfg,
                    "max_leaf_nodes": nb_leaves,
                    "min_samples_leaf": nb_min_leaf,
                    "l2_regularization": l2,
                }
                k = (
                    cand["target_transform"],
                    cand["max_leaf_nodes"],
                    cand["min_samples_leaf"],
                    cand["learning_rate"],
                    cand["l2_regularization"],
                )
                if k not in seen_keys:
                    seen_keys.add(k)
                    stage2_grid.append(cand)

    best_cfg: dict[str, Any] | None = None
    best_val_wql = float("inf")
    best_med_iter = 200
    for cfg in stage2_grid:
        wql_a, med_iter = _eval_cfg(cfg, stage2_max_iter)
        if wql_a < best_val_wql:
            best_val_wql = wql_a
            best_cfg = cfg
            best_med_iter = med_iter

    assert best_cfg is not None
    tuning_time_s = round(time.perf_counter() - t0, 2)
    print(
        f"    [gbt stage2 | {splits.name}] winner={best_cfg} "
        f"(val_wql={best_val_wql:.5f}, iter={best_med_iter}, tune={tuning_time_s}s)",
        flush=True,
    )

    # Final single-config 21-quantile fit on 100% of training data with per-quantile chronological early stopping.
    t_fit = time.perf_counter()
    tf = best_cfg["target_transform"]
    if tf == "median_residual":
        p50_tr_full, p50_va_full, p50_te_full, bi50_full = _fit_median_anchor(
            X_tr_v,
            y_tr_v,
            X_va,
            y_va,
            best_cfg,
            max_iter=final_max_iter,
            eval_stride=10,
            clip_nonnegative=clip_nonnegative,
            seed=seed,
            X_te=X_te,
        )
        assert p50_te_full is not None
        y_fit_full = y_tr_v - p50_tr_full
        anc_va_full = p50_va_full
        anc_te_full = p50_te_full
    else:
        y_fit_full = (y_tr_v - mu_tr_v) / sd_tr_v if tf == "context_norm" else y_tr_v
        anc_va_full = mu_va
        anc_te_full = mu_te
        bi50_full = 0

    val_preds = np.empty((n_va, h, levels.size), dtype=np.float64)
    test_preds = np.empty((n_te, h, levels.size), dtype=np.float64)
    per_q_iters: list[int] = []
    for q_i, tau in enumerate(levels):
        if tf == "median_residual" and abs(float(tau) - 0.50) < 1e-9:
            val_preds[:, :, q_i] = anc_va_full.reshape(n_va, h)
            test_preds[:, :, q_i] = anc_te_full.reshape(n_te, h)
            per_q_iters.append(bi50_full)
            continue
        pv, pt, bi, _ = _fit_quantile_with_val_early_stop(
            X_tr_v,
            y_fit_full,
            X_va,
            y_va,
            anc_va_full,
            sd_va,
            tau=float(tau),
            tf=tf,
            cfg=best_cfg,
            max_iter=final_max_iter,
            eval_stride=10,
            clip_nonnegative=clip_nonnegative,
            seed=seed,
            X_te=X_te,
            mu_te=anc_te_full,
            sd_te=sd_te,
        )
        assert pt is not None
        val_preds[:, :, q_i] = pv.reshape(n_va, h)
        test_preds[:, :, q_i] = pt.reshape(n_te, h)
        per_q_iters.append(bi)

    val_preds = np.sort(val_preds, axis=-1)
    test_preds = np.sort(test_preds, axis=-1)
    if clip_nonnegative:
        val_preds = np.maximum(val_preds, 0.0)
        test_preds = np.maximum(test_preds, 0.0)
    fit_time_s = round(time.perf_counter() - t_fit, 2)

    val_scale = _context_scales(splits.val)
    test_scale = _context_scales(splits.test)
    metrics, preds_dict = evaluate_with_calibrators(
        val_preds,
        Y_va,
        val_scale,
        test_preds,
        Y_te,
        test_scale,
        splits.test,
        seasonality=s,
        patch_size=patch_size,
    )
    boundary_flags = {
        "min_leaves_hit": best_cfg["max_leaf_nodes"] == 4,
        "max_leaves_hit": best_cfg["max_leaf_nodes"] == 63,
        "max_l2_hit": best_cfg["l2_regularization"] == 50.0,
        "max_iter_hit": max(per_q_iters) == final_max_iter,
    }
    return (
        {
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "tuning_time_s": tuning_time_s,
            "fit_time_s": fit_time_s,
            "best_config": {**best_cfg, "median_best_iter": int(round(float(np.median(per_q_iters))))},
            "per_quantile_best_iters": per_q_iters,
            "boundary_flags": boundary_flags,
            "best_val_wql_anchor": round(best_val_wql, 6),
            "val_wql_21q": round(weighted_quantile_loss(val_preds, Y_va), 6),
            "train_rows": int(X_tr_v.shape[0]),
            "num_features": int(X_tr_v.shape[1]),
            "tuning_grid_size": len(stage1_grid) + len(stage2_grid),
            "stage2_median_iter": best_med_iter,
            "metrics": metrics,
        },
        preds_dict,
    )


def run_tso_day_ahead_baseline(
    splits: BenchmarkSplits,
    *,
    patch_size: int = 32,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Domain sanity check for `de_power_15m`: raw ENTSO-E day-ahead forecast + val residual quantiles."""
    t0 = time.perf_counter()
    h = splits.test.forecast_horizon
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)

    def _da_pt(ds: TimeSeriesDataset) -> tuple[np.ndarray, np.ndarray]:
        pts = np.empty((len(ds), h), dtype=np.float64)
        ys = np.empty((len(ds), h), dtype=np.float64)
        for i in range(len(ds)):
            w = ds[i]
            t_len = w.target_context.shape[1]
            pts[i] = w.known[0, t_len : t_len + h].astype(np.float64)
            ys[i] = w.target_future[0].astype(np.float64)
        return pts, ys

    val_pt, val_y = _da_pt(splits.val)
    test_pt, test_y = _da_pt(splits.test)
    q_off = np.nanquantile(val_y - val_pt, levels, axis=0).T
    val_preds = np.maximum(val_pt[:, :, None] + q_off[None, :, :], 0.0)
    test_preds = np.maximum(test_pt[:, :, None] + q_off[None, :, :], 0.0)
    metrics, preds_dict = evaluate_with_calibrators(
        val_preds,
        val_y,
        _context_scales(splits.val),
        test_preds,
        test_y,
        _context_scales(splits.test),
        splits.test,
        seasonality=splits.seasonality,
        patch_size=patch_size,
    )
    return {"elapsed_s": round(time.perf_counter() - t0, 2), "metrics": metrics}, preds_dict


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from datasets import (
        load_de_power_15m,
        load_india_air_quality_daily,
        load_realised_volatility,
        load_retail_demand,
        load_sl_marine_waves_daily,
    )
    from significance import load_realised_volatility_2026_holdout

    parser = argparse.ArgumentParser(description="Precompute validation-tuned baselines and test prediction caches.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "retail_demand",
            "realised_volatility",
            "realised_volatility_2026_10m",
            "de_power_15m",
            "india_air_quality_daily",
            "sl_marine_waves_daily",
        ],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "retail_demand": load_retail_demand,
        "realised_volatility": load_realised_volatility,
        "realised_volatility_2026_10m": load_realised_volatility_2026_holdout,
        "de_power_15m": load_de_power_15m,
        "india_air_quality_daily": load_india_air_quality_daily,
        "sl_marine_waves_daily": load_sl_marine_waves_daily,
    }
    for name in args.datasets:
        sp = loaders[name]()
        print(
            f"\n== Precomputing baselines for {name} (tr={len(sp.train)}, va={len(sp.val)}, te={len(sp.test)}) ==",
            flush=True,
        )
        sn_res, _ = run_seasonal_naive(sp)
        cl_res, cl_p = run_tuned_classical(sp)
        gbt_res, gbt_p = run_tuned_gbt(sp, seed=args.seed)
        entry: dict[str, Any] = {
            "seasonal_naive": sn_res,
            "classical_domain": cl_res,
            "gbt_tuned": gbt_res,
        }
        if name == "de_power_15m":
            tso_res, _ = run_tso_day_ahead_baseline(sp)
            entry["tso_day_ahead"] = tso_res
        (args.output_dir / f"baseline_{name}.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        np.savez(args.output_dir / f"baseline_gbt_{name}.npz", raw=gbt_p["raw"], cqr=gbt_p["cqr"])
        np.savez(args.output_dir / f"baseline_classical_{name}.npz", raw=cl_p["raw"], cqr=cl_p["cqr"])
        print(
            f"  [{name:26s}] naive={sn_res['metrics']['raw']['wql']:.4f} | "
            f"cl={cl_res['metrics']['raw']['wql']:.4f} | "
            f"gbt={gbt_res['metrics']['raw']['wql']:.4f} "
            f"(cfg={gbt_res['best_config']}, bounds={gbt_res['boundary_flags']}, "
            f"tune={gbt_res['tuning_time_s']}s, fit={gbt_res['fit_time_s']}s)",
            flush=True,
        )
