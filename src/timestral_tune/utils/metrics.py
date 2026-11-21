"""Raw-space evaluation metrics and forgetting measurement (spec §5.3, §5.5).

All reported metrics operate on raw-space forecast quantiles produced through the vendored
upstream `TimestralForecaster` with its defaults (`symmetric_averaging=True`,
`clip_nonnegative=True`), so tuned and zero-shot models share one code path.

Metrics:
- `pinball_loss_raw`: mean raw-space pinball (quantile) loss over all 21 quantile levels.
- `weighted_quantile_loss` (wQL): `2 * sum(pinball) / sum(|y|)` averaged across the 21 levels.
- `mase`: Mean Absolute Scaled Error of the median forecast against the in-sample seasonal
  naive error computed from each window's context.
- `coverage` and `interval_width`: empirical central coverage and mean width of any symmetric
  quantile pair (e.g. 0.80 and 0.90).
- `compute_forgetting`: change in raw-space pinball loss, wQL, and MASE (tuned minus zero-shot)
  on a held-out non-target-domain set (spec §5.3).

- Spec: `docs/SPEC.md` §5.3, §5.5
- Roadmap phase: 3
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from ..models.conformal import (
    DEFAULT_QUANTILE_LEVELS,
    ConformalCalibrator,
    OnlineConformalCalibrator,
    collect_conformal_data,
)

if TYPE_CHECKING:
    from .._vendor.timestral.forecaster import TimestralForecaster
    from ..data.dataset import TimeSeriesDataset

__all__ = [
    "ForgettingSummary",
    "MetricsSummary",
    "compute_forgetting",
    "coverage",
    "evaluate_forecaster",
    "evaluate_forgetting",
    "evaluate_predictions",
    "interval_width",
    "mase",
    "pinball_loss_raw",
    "weighted_quantile_loss",
]

ArrayOrTensor = np.ndarray | Tensor


def _as_f64(value: ArrayOrTensor, name: str) -> np.ndarray:
    if isinstance(value, Tensor):
        arr = value.detach().cpu().to(torch.float64).numpy()
    else:
        arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be an array or tensor, got scalar {value!r}")
    return arr


def _validate_preds_actuals(
    predictions: ArrayOrTensor,
    actuals: ArrayOrTensor,
    quantile_levels: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = _as_f64(predictions, "predictions")
    y = _as_f64(actuals, "actuals")
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if levels.ndim != 1 or levels.size < 3:
        raise ValueError(f"quantile_levels must be 1-D with >=3 entries, got shape {levels.shape}")
    if not (np.isfinite(levels).all() and (levels > 0.0).all() and (levels < 1.0).all()):
        raise ValueError("quantile_levels must lie strictly in (0, 1)")
    if not (np.diff(levels) > 0.0).all():
        raise ValueError("quantile_levels must be strictly increasing")
    if preds.ndim != 3 or preds.shape[0] == 0 or preds.shape[1] == 0:
        raise ValueError(f"predictions must have non-empty shape (N, H, Q), got {preds.shape}")
    if preds.shape[2] != levels.size:
        raise ValueError(f"predictions have {preds.shape[2]} quantiles, expected {levels.size}")
    if not np.isfinite(preds).all():
        raise ValueError("predictions must be finite")
    if y.shape != preds.shape[:2]:
        raise ValueError(f"actuals shape {y.shape} must match predictions (N, H) = {preds.shape[:2]}")
    if np.isinf(y).any():
        raise ValueError("actuals must not contain infinities (use NaN for missing values)")
    return preds, y, levels


def _pair_indices(nominal_coverage: float, levels: np.ndarray) -> tuple[int, int]:
    if isinstance(nominal_coverage, bool) or not (
        isinstance(nominal_coverage, numbers.Real) and 0.0 < float(nominal_coverage) < 1.0
    ):
        raise ValueError(f"nominal_coverage must lie in (0, 1), got {nominal_coverage!r}")
    tau = 0.5 * (1.0 - float(nominal_coverage))
    low_matches = np.where(np.isclose(levels, tau, atol=1e-9))[0]
    high_matches = np.where(np.isclose(levels, 1.0 - tau, atol=1e-9))[0]
    if low_matches.size != 1 or high_matches.size != 1:
        raise ValueError(
            f"nominal_coverage={nominal_coverage!r} has no symmetric pair "
            f"({tau:.4g}, {1.0 - tau:.4g}) in quantile_levels"
        )
    return int(low_matches[0]), int(high_matches[0])


def _median_index(levels: np.ndarray) -> int:
    matches = np.where(np.isclose(levels, 0.5, atol=1e-9))[0]
    if matches.size != 1:
        raise ValueError("quantile_levels must contain the median 0.5")
    return int(matches[0])


def pinball_loss_raw(
    predictions: ArrayOrTensor,
    actuals: ArrayOrTensor,
    *,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> float:
    """Mean raw-space pinball (quantile) loss over all finite `(n, h)` positions and quantile levels."""
    preds, y, levels = _validate_preds_actuals(predictions, actuals, quantile_levels)
    valid = np.isfinite(y)
    if not valid.any():
        raise ValueError("actuals have no finite observations")
    diff = y[valid][:, None] - preds[valid]  # (M, Q)
    loss = np.maximum(levels[None, :] * diff, (levels[None, :] - 1.0) * diff)
    return float(np.mean(loss))


def weighted_quantile_loss(
    predictions: ArrayOrTensor,
    actuals: ArrayOrTensor,
    *,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> float:
    """Mean weighted quantile loss (wQL) across all quantile levels: `mean_q(2 * sum(pinball_q) / sum(|y|))`.

    Scale-free relative probabilistic accuracy metric matching standard forecasting benchmarks
    (Chronos / GIFT-Eval / Timestral). Raises `ValueError` if all observed actuals are zero.
    """
    preds, y, levels = _validate_preds_actuals(predictions, actuals, quantile_levels)
    valid = np.isfinite(y)
    if not valid.any():
        raise ValueError("actuals have no finite observations")
    y_valid = y[valid]
    denom = float(np.sum(np.abs(y_valid)))
    if denom <= 0.0:
        raise ValueError("weighted_quantile_loss is undefined when sum(|actuals|) == 0")
    diff = y_valid[:, None] - preds[valid]  # (M, Q)
    pinball = np.maximum(levels[None, :] * diff, (levels[None, :] - 1.0) * diff)
    per_q = 2.0 * np.sum(pinball, axis=0) / denom
    return float(np.mean(per_q))


def mase(
    predictions: ArrayOrTensor,
    actuals: ArrayOrTensor,
    context: ArrayOrTensor,
    *,
    seasonality: int = 1,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006) of the median forecast.

    For each series `i`, the forecast MAE over finite horizon steps is divided by the in-sample
    mean absolute `seasonality`-step difference over finite context pairs. Series with no valid
    horizon steps or zero in-sample seasonal scale (flat context) are excluded from the average;
    raises `ValueError` if no series has a positive finite seasonal scale.
    """
    if not isinstance(seasonality, numbers.Integral) or isinstance(seasonality, bool) or seasonality < 1:
        raise ValueError(f"seasonality must be a positive integer, got {seasonality!r}")
    s = int(seasonality)
    preds, y, levels = _validate_preds_actuals(predictions, actuals, quantile_levels)
    ctx = _as_f64(context, "context")
    if ctx.ndim != 2 or ctx.shape[0] != preds.shape[0]:
        raise ValueError(f"context must have shape (N={preds.shape[0]}, T), got {ctx.shape}")
    if ctx.shape[1] <= s:
        raise ValueError(f"context length {ctx.shape[1]} must exceed seasonality {s}")
    if np.isinf(ctx).any():
        raise ValueError("context must not contain infinities (use NaN for missing values)")

    med = preds[..., _median_index(levels)]  # (N, H)
    per_series: list[float] = []
    for i in range(preds.shape[0]):
        h_valid = np.isfinite(y[i])
        if not h_valid.any():
            continue
        diffs = ctx[i, s:] - ctx[i, :-s]
        c_valid = np.isfinite(diffs)
        if not c_valid.any():
            continue
        scale = float(np.mean(np.abs(diffs[c_valid])))
        if not (math.isfinite(scale) and scale > 0.0):
            continue
        mae = float(np.mean(np.abs(y[i, h_valid] - med[i, h_valid])))
        per_series.append(mae / scale)
    if not per_series:
        raise ValueError("no series had both finite horizon actuals and positive in-sample seasonal scale")
    return float(np.mean(per_series))


def coverage(
    predictions: ArrayOrTensor,
    actuals: ArrayOrTensor,
    nominal_coverage: float = 0.80,
    *,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> float:
    """Empirical coverage of the central `nominal_coverage` prediction interval over finite actuals."""
    preds, y, levels = _validate_preds_actuals(predictions, actuals, quantile_levels)
    low_idx, high_idx = _pair_indices(nominal_coverage, levels)
    valid = np.isfinite(y)
    if not valid.any():
        raise ValueError("actuals have no finite observations")
    lo = preds[..., low_idx][valid]
    hi = preds[..., high_idx][valid]
    y_v = y[valid]
    return float(np.mean((y_v >= lo) & (y_v <= hi)))


def interval_width(
    predictions: ArrayOrTensor,
    nominal_coverage: float = 0.80,
    *,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    actuals: ArrayOrTensor | None = None,
) -> float:
    """Mean width `q_{1 - tau} - q_tau` of the central `nominal_coverage` prediction interval.

    If `actuals` is provided, averages only over finite `(n, h)` positions so width is measured
    on the exact same positions as `coverage()`.
    """
    preds = _as_f64(predictions, "predictions")
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if preds.ndim != 3 or preds.shape[0] == 0 or preds.shape[1] == 0 or preds.shape[2] != levels.size:
        raise ValueError(f"predictions shape {preds.shape} does not match (N, H, {levels.size})")
    if not np.isfinite(preds).all():
        raise ValueError("predictions must be finite")
    low_idx, high_idx = _pair_indices(nominal_coverage, levels)
    widths = preds[..., high_idx] - preds[..., low_idx]
    if actuals is not None:
        y = _as_f64(actuals, "actuals")
        if y.shape != preds.shape[:2]:
            raise ValueError(f"actuals shape {y.shape} must match predictions (N, H) = {preds.shape[:2]}")
        valid = np.isfinite(y)
        if not valid.any():
            raise ValueError("actuals have no finite observations")
        return float(np.mean(widths[valid]))
    return float(np.mean(widths))


@dataclass(frozen=True)
class MetricsSummary:
    """Full raw-space evaluation report of spec §5.5."""

    pinball: float
    wql: float
    mase: float
    coverage_80: float
    width_80: float
    coverage_90: float
    width_90: float
    num_series: int
    num_scored: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForgettingSummary:
    """Change in raw-space accuracy (tuned minus zero-shot) on a non-target-domain set (spec §5.3).

    Positive deltas indicate forgetting (degradation vs zero-shot); negative deltas indicate
    positive transfer.
    """

    delta_pinball: float
    relative_delta_pinball: float
    delta_wql: float
    relative_delta_wql: float
    delta_mase: float
    relative_delta_mase: float
    tuned: MetricsSummary
    zero_shot: MetricsSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_pinball": self.delta_pinball,
            "relative_delta_pinball": self.relative_delta_pinball,
            "delta_wql": self.delta_wql,
            "relative_delta_wql": self.relative_delta_wql,
            "delta_mase": self.delta_mase,
            "relative_delta_mase": self.relative_delta_mase,
            "tuned": self.tuned.to_dict(),
            "zero_shot": self.zero_shot.to_dict(),
        }


def evaluate_predictions(
    predictions: ArrayOrTensor,
    actuals: ArrayOrTensor,
    context: ArrayOrTensor,
    *,
    seasonality: int = 1,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> MetricsSummary:
    """Compute every raw-space metric of spec §5.5 from `(predictions, actuals, context)` arrays."""
    preds, y, levels = _validate_preds_actuals(predictions, actuals, quantile_levels)
    return MetricsSummary(
        pinball=pinball_loss_raw(preds, y, quantile_levels=levels),
        wql=weighted_quantile_loss(preds, y, quantile_levels=levels),
        mase=mase(preds, y, context, seasonality=seasonality, quantile_levels=levels),
        coverage_80=coverage(preds, y, 0.80, quantile_levels=levels),
        width_80=interval_width(preds, 0.80, quantile_levels=levels, actuals=y),
        coverage_90=coverage(preds, y, 0.90, quantile_levels=levels),
        width_90=interval_width(preds, 0.90, quantile_levels=levels, actuals=y),
        num_series=int(preds.shape[0]),
        num_scored=int(np.isfinite(y).sum()),
    )


def _collect_contexts(dataset: TimeSeriesDataset) -> np.ndarray:
    """Left-padded float64 `(N_windows * n_targets, max_context)` contexts aligned with `collect_conformal_data`."""
    max_t = max(dataset[i].target_context.shape[1] for i in range(len(dataset)))
    rows: list[np.ndarray] = []
    for i in range(len(dataset)):
        ctx = dataset[i].target_context.astype(np.float64)  # (V, T_i)
        pad = max_t - ctx.shape[1]
        if pad > 0:
            ctx = np.concatenate([np.full((ctx.shape[0], pad), np.nan, dtype=np.float64), ctx], axis=1)
        rows.append(ctx)
    return np.concatenate(rows, axis=0)


def evaluate_forecaster(
    forecaster: TimestralForecaster,
    dataset: TimeSeriesDataset,
    *,
    calibrator: ConformalCalibrator | OnlineConformalCalibrator | None = None,
    online_update: bool = False,
    seasonality: int = 1,
    batch_size: int = 32,
    symmetric_averaging: bool = True,
    clip_nonnegative: bool = True,
) -> MetricsSummary:
    """Evaluate `forecaster` on `dataset` in raw space through upstream `TimestralForecaster.forecast()`.

    When `calibrator` is supplied, transforms raw forecasts before scoring. If `calibrator` is an
    `OnlineConformalCalibrator` and `online_update=True`, evaluates chronologically one window at
    a time (`transform` then `update`), simulating live deployment.
    """
    preds, actuals, scale = collect_conformal_data(
        forecaster,
        dataset,
        batch_size=batch_size,
        symmetric_averaging=symmetric_averaging,
        clip_nonnegative=clip_nonnegative,
    )
    if calibrator is not None:
        if isinstance(calibrator, OnlineConformalCalibrator) and online_update:
            calibrated = np.empty_like(preds)
            for i in range(preds.shape[0]):
                calibrated[i : i + 1] = calibrator.transform(preds[i : i + 1], context_scale=scale[i : i + 1])
                calibrator.update(preds[i : i + 1], actuals[i : i + 1], context_scale=scale[i : i + 1])
            preds = calibrated
        else:
            preds = calibrator.transform(preds, context_scale=scale)
    contexts = _collect_contexts(dataset)
    return evaluate_predictions(
        preds,
        actuals,
        contexts,
        seasonality=seasonality,
        quantile_levels=forecaster.quantile_levels,
    )


def compute_forgetting(tuned: MetricsSummary, zero_shot: MetricsSummary) -> ForgettingSummary:
    """Compute the forgetting metric (tuned minus zero-shot) from two `MetricsSummary` objects."""

    def _rel(delta: float, base: float) -> float:
        return delta / base if base > 0.0 else float("nan")

    d_pin = tuned.pinball - zero_shot.pinball
    d_wql = tuned.wql - zero_shot.wql
    d_mase = tuned.mase - zero_shot.mase
    return ForgettingSummary(
        delta_pinball=d_pin,
        relative_delta_pinball=_rel(d_pin, zero_shot.pinball),
        delta_wql=d_wql,
        relative_delta_wql=_rel(d_wql, zero_shot.wql),
        delta_mase=d_mase,
        relative_delta_mase=_rel(d_mase, zero_shot.mase),
        tuned=tuned,
        zero_shot=zero_shot,
    )


def evaluate_forgetting(
    tuned_forecaster: TimestralForecaster,
    zero_shot_forecaster: TimestralForecaster,
    general_dataset: TimeSeriesDataset,
    *,
    calibrator: ConformalCalibrator | OnlineConformalCalibrator | None = None,
    seasonality: int = 1,
    batch_size: int = 32,
) -> ForgettingSummary:
    """Evaluate `tuned_forecaster` vs `zero_shot_forecaster` on a non-target-domain `general_dataset`."""
    zs = evaluate_forecaster(zero_shot_forecaster, general_dataset, seasonality=seasonality, batch_size=batch_size)
    tuned = evaluate_forecaster(
        tuned_forecaster,
        general_dataset,
        calibrator=calibrator,
        seasonality=seasonality,
        batch_size=batch_size,
    )
    return compute_forgetting(tuned, zs)
