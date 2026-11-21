"""Post-hoc conformal calibration for Timestral's 21-quantile forecasts.

Two distribution-free calibrators operating on raw-space forecast quantiles without modifying
model weights (spec §5.2, decision D6):

- `ConformalCalibrator`: static split Conformalized Quantile Regression (Romano et al., 2019,
  eqs. 9-11), with scale-normalised conformity scores, horizon buckets, exact finite-sample
  order statistic `ceil((n + 1) * (1 - alpha))` computed in O(n) via `np.partition`, and
  anchor-aware monotonicity repair around the unshifted median (so narrowing a single pair like
  the default `target_coverage=0.80` is not undone by uncalibrated neighbours).
- `OnlineConformalCalibrator`: Adaptive Conformal Inference (Gibbs & Candès, 2021) for
  non-exchangeable series, maintaining per-bucket working miscoverage `alpha_t` over a rolling
  window of `window` recent forecast observations (kept sorted for O(1) quantile lookup).
  `err_t` is evaluated on the actual emitted (repaired) interval; `alpha_t` is never clipped at
  `[0, 1]` and is bounded only at Lemma 4.1's theoretical envelope `[-gamma, 1 + gamma]` so
  finite-window extremes (`max`/`min`) cannot cause unbounded wind-up. Saturation at
  `alpha_t <= 0` uses the largest score in the window and raises `.saturated`.
- `collect_conformal_data`: runs an upstream-compatible `TimestralForecaster` over a
  `TimeSeriesDataset` and returns `(predictions, actuals, context_scale)` ready for either
  calibrator.

- Spec: `docs/SPEC.md` §5.2, §7.3, §11 (D6)
- Acceptance tests: T13
"""

from __future__ import annotations

import bisect
import json
import math
import numbers
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from .._vendor.timestral.forecaster import TimestralForecaster
    from ..data.dataset import TimeSeriesDataset

__all__ = [
    "DEFAULT_QUANTILE_LEVELS",
    "ConformalCalibrator",
    "OnlineConformalCalibrator",
    "collect_conformal_data",
]

DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (0.01, *(round(0.05 * i, 2) for i in range(1, 20)), 0.99)
CoverageSpec = float | Literal["all"] | Sequence[float]
ArrayOrTensor = np.ndarray | Tensor


def _validate_quantile_levels(levels: Sequence[float]) -> tuple[tuple[float, ...], int, dict[float, tuple[int, int]]]:
    """Validate strictly increasing levels containing 0.5 and return `(levels, median_idx, coverage_to_pair)`."""
    arr = tuple(float(x) for x in levels)
    if len(arr) < 3:
        raise ValueError(f"quantile_levels must have at least 3 levels including 0.5, got {len(arr)}")
    if not all(math.isfinite(x) and 0.0 < x < 1.0 for x in arr):
        raise ValueError("quantile_levels must lie strictly in (0, 1)")
    if not all(a < b for a, b in zip(arr, arr[1:], strict=False)):
        raise ValueError("quantile_levels must be strictly increasing")
    median_matches = [i for i, x in enumerate(arr) if math.isclose(x, 0.5, abs_tol=1e-9)]
    if len(median_matches) != 1:
        raise ValueError("quantile_levels must contain the median 0.5")
    median_idx = median_matches[0]

    pairs: dict[float, tuple[int, int]] = {}
    for low_idx in range(median_idx):
        tau = arr[low_idx]
        target_hi = 1.0 - tau
        for high_idx in range(median_idx + 1, len(arr)):
            if math.isclose(arr[high_idx], target_hi, abs_tol=1e-9):
                cov = round(1.0 - 2.0 * tau, 10)
                pairs[cov] = (low_idx, high_idx)
                break
    if not pairs:
        raise ValueError("quantile_levels has no symmetric (tau, 1 - tau) pairs around 0.5")
    return arr, median_idx, pairs


def _resolve_coverages(
    target_coverage: CoverageSpec,
    available: dict[float, tuple[int, int]],
) -> tuple[tuple[float, int, int], ...]:
    """Resolve `target_coverage` into sorted `(coverage, low_idx, high_idx)` triples."""
    valid_sorted = sorted(available)
    if target_coverage == "all":
        return tuple((cov, *available[cov]) for cov in valid_sorted)
    if isinstance(target_coverage, str):
        raise ValueError(
            f"target_coverage must be 'all' or valid nominal coverage(s) {valid_sorted}, got {target_coverage!r}"
        )
    requested: Sequence[float]
    if isinstance(target_coverage, numbers.Real) and not isinstance(target_coverage, bool):
        requested = (float(target_coverage),)
    elif isinstance(target_coverage, Sequence) and len(target_coverage) > 0:
        if any(isinstance(v, bool) or not isinstance(v, numbers.Real) for v in target_coverage):
            raise ValueError(f"target_coverage entries must be numbers, got {target_coverage!r}")
        requested = tuple(float(v) for v in target_coverage)
    else:
        raise ValueError(
            f"target_coverage must be 'all' or valid nominal coverage(s) {valid_sorted}, got {target_coverage!r}"
        )

    resolved: list[tuple[float, int, int]] = []
    seen: set[float] = set()
    for cov in requested:
        match = next((k for k in valid_sorted if math.isclose(cov, k, abs_tol=1e-9)), None)
        if match is None:
            raise ValueError(
                f"coverage {cov!r} does not match any symmetric quantile pair; valid coverages are {valid_sorted}"
            )
        if match in seen:
            raise ValueError(f"duplicate coverage {match!r} in target_coverage")
        seen.add(match)
        resolved.append((match, *available[match]))
    return tuple(resolved)


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    return float(value)


def _to_numpy_f64(value: ArrayOrTensor, name: str) -> tuple[np.ndarray, bool, torch.device | None, torch.dtype | None]:
    if isinstance(value, Tensor):
        return value.detach().cpu().to(torch.float64).numpy(), True, value.device, value.dtype
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be an array or tensor, got scalar {value!r}")
    return arr, False, None, None


def _validate_inputs(
    predictions: ArrayOrTensor,
    num_quantiles: int,
    actuals: ArrayOrTensor | None = None,
    context_scale: ArrayOrTensor | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, bool, torch.device | None, torch.dtype | None]:
    preds, is_tensor, device, dtype = _to_numpy_f64(predictions, "predictions")
    if preds.ndim != 3:
        raise ValueError(f"predictions must have shape (N, H, Q), got {preds.shape}")
    if preds.shape[0] == 0 or preds.shape[1] == 0:
        raise ValueError(f"predictions must have non-zero N and H, got shape {preds.shape}")
    if preds.shape[2] != num_quantiles:
        raise ValueError(f"predictions have {preds.shape[2]} quantiles on the last axis, expected {num_quantiles}")
    if not np.isfinite(preds).all():
        raise ValueError("predictions must be finite")

    n, h, _ = preds.shape
    y: np.ndarray | None = None
    if actuals is not None:
        y, _, _, _ = _to_numpy_f64(actuals, "actuals")
        if y.shape != (n, h):
            raise ValueError(f"actuals shape {y.shape} must match predictions' (N, H) = {(n, h)}")
        if np.isinf(y).any():
            raise ValueError("actuals must not contain infinities (use NaN for missing values)")

    if context_scale is None:
        scale = np.ones((n,), dtype=np.float64)
    else:
        scale, _, _, _ = _to_numpy_f64(context_scale, "context_scale")
        if scale.shape != (n,):
            raise ValueError(f"context_scale must have shape ({n},), got {scale.shape}")
        if not np.isfinite(scale).all() or (scale <= 0.0).any():
            raise ValueError("context_scale must be strictly positive and finite")

    return preds, y, scale, is_tensor, device, dtype


def _cqr_order_statistic(scores: np.ndarray, coverage: float) -> float:
    """Exact CQR finite-sample order statistic `ceil((n + 1) * coverage)` in O(n) (Romano et al., eq. 10).

    When `ceil((n + 1) * coverage) > n` (finite-sample boundary at very small `n` or high
    coverage), returns `max(scores)` as the finite upper bound.
    """
    n = scores.size
    rank = math.ceil((n + 1) * coverage)
    if rank > n:
        return float(np.max(scores))
    idx = max(0, rank - 1)
    return float(np.partition(scores, idx)[idx])


def _repair_monotonicity_anchored(
    quantiles: np.ndarray,
    median_idx: int,
    pairs: tuple[tuple[float, int, int], ...],
) -> np.ndarray:
    """Repair monotonicity around fixed anchors (unshifted median + calibrated levels).

    Calibrated levels are clamped so they never cross the median and repaired outward relative
    to each other (so calibrated intervals can only widen against other calibrated pairs).
    Uncalibrated levels are then reconciled into the bounds between adjacent anchors (or outward
    beyond the outermost anchor) without moving any calibrated level. When every symmetric pair
    is calibrated (`target_coverage="all"`), this reduces identically to pure outward repair.
    """
    out = quantiles.copy()
    num_q = out.shape[-1]
    cal_upper = sorted(high_idx for _, _, high_idx in pairs)
    cal_lower = sorted((low_idx for _, low_idx, _ in pairs), reverse=True)

    # Upper half: calibrated anchors repair outward from the unshifted median.
    prev = out[..., median_idx]
    for c in cal_upper:
        out[..., c] = np.maximum(out[..., c], prev)
        prev = out[..., c]
    upper_anchors = [median_idx, *cal_upper]
    for a_lo, a_hi in zip(upper_anchors, upper_anchors[1:], strict=False):
        if a_hi - a_lo > 1:
            seg = np.clip(out[..., a_lo + 1 : a_hi], out[..., a_lo : a_lo + 1], out[..., a_hi : a_hi + 1])
            out[..., a_lo + 1 : a_hi] = np.maximum.accumulate(seg, axis=-1)
    last_hi = upper_anchors[-1]
    if last_hi < num_q - 1:
        out[..., last_hi:] = np.maximum.accumulate(out[..., last_hi:], axis=-1)

    # Lower half: calibrated anchors repair outward (towards index 0) from the unshifted median.
    prev = out[..., median_idx]
    for c in cal_lower:
        out[..., c] = np.minimum(out[..., c], prev)
        prev = out[..., c]
    lower_anchors = [median_idx, *cal_lower]
    for a_hi, a_lo in zip(lower_anchors, lower_anchors[1:], strict=False):
        if a_hi - a_lo > 1:
            seg = np.clip(out[..., a_lo + 1 : a_hi], out[..., a_lo : a_lo + 1], out[..., a_hi : a_hi + 1])
            out[..., a_lo + 1 : a_hi] = np.minimum.accumulate(seg[..., ::-1], axis=-1)[..., ::-1]
    last_lo = lower_anchors[-1]
    if last_lo > 0:
        out[..., : last_lo + 1] = np.minimum.accumulate(out[..., last_lo::-1], axis=-1)[..., ::-1]

    return out


def _apply_offsets(
    preds: np.ndarray,
    scale: np.ndarray,
    pairs: tuple[tuple[float, int, int], ...],
    offsets: np.ndarray,
    bucket_size: int,
    median_idx: int,
) -> np.ndarray:
    """Apply `(num_pairs, num_buckets)` scaled offsets and anchor-aware monotonicity repair."""
    _, h, _ = preds.shape
    bucket_ids = np.arange(h) // bucket_size
    if int(bucket_ids[-1]) >= offsets.shape[1]:
        raise ValueError(
            f"horizon {h} requires bucket {int(bucket_ids[-1])}, but calibrator only has {offsets.shape[1]} bucket(s)"
        )
    out = preds.copy()
    for pair_idx, (_, low_idx, high_idx) in enumerate(pairs):
        shift = offsets[pair_idx, bucket_ids][None, :] * scale[:, None]
        out[..., low_idx] -= shift
        out[..., high_idx] += shift
    return _repair_monotonicity_anchored(out, median_idx, pairs)


def _restore_type(arr: np.ndarray, is_tensor: bool, device: torch.device | None, dtype: torch.dtype | None) -> Any:
    if is_tensor:
        assert device is not None and dtype is not None
        return torch.from_numpy(arr).to(device=device, dtype=dtype)
    return arr


class ConformalCalibrator:
    """Static split Conformalized Quantile Regression (CQR; Romano et al., 2019, eqs. 9-11).

    Args:
        target_coverage: Nominal central coverage matching a symmetric pair of `quantile_levels`
            (e.g. `0.80`, `0.90`), a sequence of valid coverages, or `"all"` for all ten pairs.
        bucket_size: Horizon steps pooled per calibration bucket (default 32, one Timestral patch;
            set to 1 for per-step calibration).
        quantile_levels: Sorted quantile levels matching the model's output axis.
    """

    def __init__(
        self,
        target_coverage: CoverageSpec = 0.80,
        *,
        bucket_size: int = 32,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    ) -> None:
        self.quantile_levels, self.median_idx, available = _validate_quantile_levels(quantile_levels)
        self.pairs = _resolve_coverages(target_coverage, available)
        self.bucket_size = _positive_int("bucket_size", bucket_size)
        self.offsets_: np.ndarray | None = None
        self.bucket_counts_: np.ndarray | None = None

    @property
    def coverages(self) -> tuple[float, ...]:
        return tuple(cov for cov, _, _ in self.pairs)

    @property
    def is_fitted(self) -> bool:
        return self.offsets_ is not None

    def fit(
        self,
        predictions: ArrayOrTensor,
        actuals: ArrayOrTensor,
        context_scale: ArrayOrTensor | None = None,
    ) -> ConformalCalibrator:
        """Fit per-bucket scaled conformity quantiles from held-out calibration forecasts."""
        preds, y, scale, _, _, _ = _validate_inputs(predictions, len(self.quantile_levels), actuals, context_scale)
        assert y is not None
        _, h, _ = preds.shape
        num_buckets = math.ceil(h / self.bucket_size)
        offsets = np.empty((len(self.pairs), num_buckets), dtype=np.float64)
        counts = np.zeros(num_buckets, dtype=np.int64)

        for b in range(num_buckets):
            start = b * self.bucket_size
            end = min(h, start + self.bucket_size)
            y_b = y[:, start:end]
            valid = np.isfinite(y_b)
            n_valid = int(valid.sum())
            if n_valid == 0:
                raise ValueError(f"horizon bucket {b} (steps {start}..{end - 1}) has no finite actuals")
            counts[b] = n_valid
            scale_b = np.broadcast_to(scale[:, None], y_b.shape)[valid]
            y_valid = y_b[valid]
            for pair_idx, (cov, low_idx, high_idx) in enumerate(self.pairs):
                q_low = preds[:, start:end, low_idx][valid]
                q_high = preds[:, start:end, high_idx][valid]
                scores = np.maximum(q_low - y_valid, y_valid - q_high) / scale_b
                offsets[pair_idx, b] = _cqr_order_statistic(scores, cov)

        self.offsets_ = offsets
        self.bucket_counts_ = counts
        return self

    @overload
    def transform(self, predictions: np.ndarray, context_scale: ArrayOrTensor | None = None) -> np.ndarray: ...

    @overload
    def transform(self, predictions: Tensor, context_scale: ArrayOrTensor | None = None) -> Tensor: ...

    def transform(
        self,
        predictions: ArrayOrTensor,
        context_scale: ArrayOrTensor | None = None,
    ) -> ArrayOrTensor:
        """Apply calibrated shifts and anchor-aware monotonicity repair to raw-space quantile forecasts."""
        if self.offsets_ is None:
            raise RuntimeError("ConformalCalibrator must be fitted before calling transform()")
        preds, _, scale, is_tensor, device, dtype = _validate_inputs(
            predictions, len(self.quantile_levels), None, context_scale
        )
        calibrated = _apply_offsets(preds, scale, self.pairs, self.offsets_, self.bucket_size, self.median_idx)
        return _restore_type(calibrated, is_tensor, device, dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "cqr",
            "quantile_levels": list(self.quantile_levels),
            "coverages": list(self.coverages),
            "bucket_size": self.bucket_size,
            "offsets": None if self.offsets_ is None else self.offsets_.tolist(),
            "bucket_counts": None if self.bucket_counts_ is None else self.bucket_counts_.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConformalCalibrator:
        if not isinstance(data, dict) or data.get("kind") != "cqr":
            raise ValueError(f"expected kind='cqr', got {data.get('kind') if isinstance(data, dict) else data!r}")
        for required in ("coverages", "bucket_size", "quantile_levels"):
            if required not in data:
                raise ValueError(f"missing required key {required!r} in CQR payload")
        cal = cls(
            target_coverage=data["coverages"],
            bucket_size=data["bucket_size"],
            quantile_levels=data["quantile_levels"],
        )
        if data.get("offsets") is not None:
            try:
                offsets = np.asarray(data["offsets"], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("offsets must be a 2-D numeric array") from exc
            if offsets.ndim != 2 or offsets.shape[0] != len(cal.pairs) or offsets.shape[1] < 1:
                raise ValueError(
                    f"offsets shape {offsets.shape} invalid for {len(cal.pairs)} coverage pair(s) "
                    "(need 2-D with at least 1 bucket)"
                )
            if not np.isfinite(offsets).all():
                raise ValueError("offsets must be finite")
            cal.offsets_ = offsets
            if data.get("bucket_counts") is not None:
                counts = np.asarray(data["bucket_counts"], dtype=np.int64)
                if counts.shape != (offsets.shape[1],) or (counts < 0).any():
                    raise ValueError(f"bucket_counts shape {counts.shape} does not match {offsets.shape[1]} buckets")
                cal.bucket_counts_ = counts
        return cal

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> ConformalCalibrator:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class OnlineConformalCalibrator:
    """Adaptive Conformal Inference (ACI; Gibbs & Candès, 2021) for non-exchangeable series.

    Maintains a rolling window of up to `window` recent forecast observations per
    `(coverage_pair, horizon_bucket)` (with a sorted step-score mirror for O(1) quantile lookup)
    and working miscoverage `alpha_t`. Realised miscoverage `err_t` is evaluated on the actual
    emitted (anchor-repaired) interval; with `bucket_size > 1`, `err_t` is the mean miscoverage
    fraction across observed steps in the bucket.

    `alpha_t` is never clipped at `[0, 1]`: it crosses `0` (emitting the largest score in the
    window and setting `.saturated = True`) and `1` (emitting the smallest score). Because finite
    window extremes replace the paper's infinite / empty intervals, `alpha_t` is bounded only at
    Lemma 4.1's theoretical envelope `[-gamma, 1 + gamma]` to prevent unbounded wind-up.

    Args:
        target_coverage: Nominal central coverage(s) or `"all"`.
        gamma: Online step size in `alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`.
        window: Rolling window length in forecast observations per `(pair, bucket)` (default 500;
            with `bucket_size=32`, holds up to 500 forecasts / 16,000 step scores — ~1 MB JSON per
            pair or ~10 MB for `"all"` at horizon 96).
        bucket_size: Horizon steps pooled per bucket (default 32, down to 1).
        quantile_levels: Sorted quantile levels matching the model's output axis.
    """

    def __init__(
        self,
        target_coverage: CoverageSpec = 0.80,
        *,
        gamma: float = 0.005,
        window: int = 500,
        bucket_size: int = 32,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    ) -> None:
        self.quantile_levels, self.median_idx, available = _validate_quantile_levels(quantile_levels)
        self.pairs = _resolve_coverages(target_coverage, available)
        if isinstance(gamma, bool) or not (isinstance(gamma, numbers.Real) and math.isfinite(gamma) and gamma > 0):
            raise ValueError(f"gamma must be positive and finite, got {gamma!r}")
        self.gamma = float(gamma)
        self.window = _positive_int("window", window)
        self.bucket_size = _positive_int("bucket_size", bucket_size)
        self._num_buckets: int = 0
        self._alpha_t: list[list[float]] = [[] for _ in self.pairs]
        self._scores: list[list[deque[tuple[float, ...]]]] = [[] for _ in self.pairs]
        self._sorted: list[list[list[float]]] = [[] for _ in self.pairs]

    @property
    def coverages(self) -> tuple[float, ...]:
        return tuple(cov for cov, _, _ in self.pairs)

    def _ensure_buckets(self, num_buckets: int) -> None:
        if num_buckets <= self._num_buckets:
            return
        for pair_idx, (cov, _, _) in enumerate(self.pairs):
            alpha = 1.0 - cov
            for _ in range(num_buckets - self._num_buckets):
                self._alpha_t[pair_idx].append(alpha)
                self._scores[pair_idx].append(deque())
                self._sorted[pair_idx].append([])
        self._num_buckets = num_buckets

    def _push_observation(self, pair_idx: int, bucket_idx: int, obs_scores: Sequence[float]) -> None:
        """Append one forecast observation's finite step scores and evict the oldest observation if full."""
        if not obs_scores:
            return
        fifo = self._scores[pair_idx][bucket_idx]
        ordered = self._sorted[pair_idx][bucket_idx]
        if len(fifo) >= self.window:
            evicted = fifo.popleft()
            for old in evicted:
                del ordered[bisect.bisect_left(ordered, old)]
        entry = tuple(float(v) for v in obs_scores)
        fifo.append(entry)
        for val in entry:
            bisect.insort(ordered, val)

    def _bucket_offset(self, pair_idx: int, bucket_idx: int) -> tuple[float, bool]:
        if bucket_idx >= self._num_buckets:
            return 0.0, False
        ordered = self._sorted[pair_idx][bucket_idx]
        m = len(ordered)
        if m == 0:
            return 0.0, False
        alpha_t = self._alpha_t[pair_idx][bucket_idx]
        if alpha_t <= 0.0:
            return ordered[-1], True
        if alpha_t >= 1.0:
            return ordered[0], False
        idx = min(m - 1, max(0, math.ceil(m * (1.0 - alpha_t)) - 1))
        return ordered[idx], False

    @property
    def alpha_t(self) -> np.ndarray:
        """Current working miscoverage `(num_pairs, num_buckets)`."""
        if self._num_buckets == 0:
            return np.empty((len(self.pairs), 0), dtype=np.float64)
        return np.asarray(self._alpha_t, dtype=np.float64)

    @property
    def saturated(self) -> bool:
        """True if any calibrated `(coverage, bucket)` currently has `alpha_t <= 0` with scores present."""
        return any(
            bool(self._sorted[p][b]) and self._alpha_t[p][b] <= 0.0
            for p in range(len(self.pairs))
            for b in range(self._num_buckets)
        )

    def current_offsets(self, horizon: int) -> np.ndarray:
        """Current `(num_pairs, num_buckets)` offsets for `horizon`, without mutating calibrator state."""
        h = _positive_int("horizon", horizon)
        num_buckets = math.ceil(h / self.bucket_size)
        offsets = np.zeros((len(self.pairs), num_buckets), dtype=np.float64)
        for p in range(len(self.pairs)):
            for b in range(min(num_buckets, self._num_buckets)):
                offsets[p, b], _ = self._bucket_offset(p, b)
        return offsets

    @overload
    def transform(self, predictions: np.ndarray, context_scale: ArrayOrTensor | None = None) -> np.ndarray: ...

    @overload
    def transform(self, predictions: Tensor, context_scale: ArrayOrTensor | None = None) -> Tensor: ...

    def transform(
        self,
        predictions: ArrayOrTensor,
        context_scale: ArrayOrTensor | None = None,
    ) -> ArrayOrTensor:
        """Apply the current online offsets and anchor-aware monotonicity repair without mutating state."""
        preds, _, scale, is_tensor, device, dtype = _validate_inputs(
            predictions, len(self.quantile_levels), None, context_scale
        )
        offsets = self.current_offsets(preds.shape[1])
        calibrated = _apply_offsets(preds, scale, self.pairs, offsets, self.bucket_size, self.median_idx)
        return _restore_type(calibrated, is_tensor, device, dtype)

    def update(
        self,
        predictions: ArrayOrTensor,
        actuals: ArrayOrTensor,
        context_scale: ArrayOrTensor | None = None,
    ) -> OnlineConformalCalibrator:
        """Update working miscoverages and rolling observation windows chronologically as actuals arrive."""
        preds, y, scale, _, _, _ = _validate_inputs(predictions, len(self.quantile_levels), actuals, context_scale)
        assert y is not None
        n, h, _ = preds.shape
        num_buckets = math.ceil(h / self.bucket_size)
        self._ensure_buckets(num_buckets)

        for i in range(n):
            s_i = float(scale[i])
            # Evaluate miscoverage against the actual emitted (anchor-repaired) interval before updating.
            offsets_i = self.current_offsets(h)
            emitted_i = _apply_offsets(
                preds[i : i + 1], scale[i : i + 1], self.pairs, offsets_i, self.bucket_size, self.median_idx
            )[0]
            for b in range(num_buckets):
                start = b * self.bucket_size
                end = min(h, start + self.bucket_size)
                y_b = y[i, start:end]
                valid = np.isfinite(y_b)
                if not valid.any():
                    continue
                y_v = y_b[valid]
                for p, (cov, low_idx, high_idx) in enumerate(self.pairs):
                    alpha = 1.0 - cov
                    lo = emitted_i[start:end, low_idx][valid]
                    hi = emitted_i[start:end, high_idx][valid]
                    err = float(np.mean((y_v < lo) | (y_v > hi)))
                    # Never clipped at [0, 1]; bounded at Lemma 4.1's [-gamma, 1 + gamma] envelope so
                    # finite-window extremes (max / min) do not wind up unboundedly under regime shifts.
                    next_alpha = self._alpha_t[p][b] + self.gamma * (alpha - err)
                    self._alpha_t[p][b] = min(1.0 + self.gamma, max(-self.gamma, next_alpha))
                    q_low = preds[i, start:end, low_idx][valid]
                    q_high = preds[i, start:end, high_idx][valid]
                    new_scores = (np.maximum(q_low - y_v, y_v - q_high) / s_i).tolist()
                    self._push_observation(p, b, new_scores)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "aci",
            "quantile_levels": list(self.quantile_levels),
            "coverages": list(self.coverages),
            "gamma": self.gamma,
            "window": self.window,
            "bucket_size": self.bucket_size,
            "num_buckets": self._num_buckets,
            "alpha_t": [list(row) for row in self._alpha_t],
            "scores": [[[list(obs) for obs in dq] for dq in row] for row in self._scores],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OnlineConformalCalibrator:
        if not isinstance(data, dict) or data.get("kind") != "aci":
            raise ValueError(f"expected kind='aci', got {data.get('kind') if isinstance(data, dict) else data!r}")
        for required in ("coverages", "gamma", "window", "bucket_size", "quantile_levels", "alpha_t", "scores"):
            if required not in data:
                raise ValueError(f"missing required key {required!r} in ACI payload")
        cal = cls(
            target_coverage=data["coverages"],
            gamma=data["gamma"],
            window=data["window"],
            bucket_size=data["bucket_size"],
            quantile_levels=data["quantile_levels"],
        )
        raw_nb = data.get("num_buckets", 0)
        if not isinstance(raw_nb, numbers.Integral) or isinstance(raw_nb, bool) or raw_nb < 0:
            raise ValueError(f"num_buckets must be a non-negative integer, got {raw_nb!r}")
        num_buckets = int(raw_nb)
        alpha_t = data["alpha_t"]
        scores = data["scores"]
        if not isinstance(alpha_t, list) or len(alpha_t) != len(cal.pairs):
            got = len(alpha_t) if isinstance(alpha_t, list) else type(alpha_t).__name__
            raise ValueError(f"alpha_t must have {len(cal.pairs)} rows, got {got}")
        if not isinstance(scores, list) or len(scores) != len(cal.pairs):
            got = len(scores) if isinstance(scores, list) else type(scores).__name__
            raise ValueError(f"scores must have {len(cal.pairs)} rows, got {got}")

        cal._ensure_buckets(num_buckets)
        for p in range(len(cal.pairs)):
            if not isinstance(alpha_t[p], list) or len(alpha_t[p]) != num_buckets:
                raise ValueError(f"alpha_t[{p}] must have length {num_buckets}")
            if not isinstance(scores[p], list) or len(scores[p]) != num_buckets:
                raise ValueError(f"scores[{p}] must have length {num_buckets}")
            for b in range(num_buckets):
                cal._alpha_t[p][b] = _finite_real(f"alpha_t[{p}][{b}]", alpha_t[p][b])
                bucket_scores = scores[p][b]
                if not isinstance(bucket_scores, list):
                    raise ValueError(f"scores[{p}][{b}] must be a list")
                for entry in bucket_scores[-cal.window :]:
                    if isinstance(entry, list):
                        if not entry:
                            raise ValueError(f"empty observation list in scores[{p}][{b}]")
                        cal._push_observation(p, b, [_finite_real(f"scores[{p}][{b}] entry", s) for s in entry])
                    else:
                        # Legacy flat-score format: treat each scalar as a single-step observation.
                        cal._push_observation(p, b, [_finite_real(f"scores[{p}][{b}] entry", entry)])
        return cal

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> OnlineConformalCalibrator:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@torch.no_grad()
def collect_conformal_data(
    forecaster: TimestralForecaster,
    dataset: TimeSeriesDataset,
    *,
    batch_size: int = 32,
    symmetric_averaging: bool = True,
    clip_nonnegative: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run `forecaster.forecast()` over `dataset` and return `(predictions, actuals, context_scale)`.

    Bridges a `TimeSeriesDataset` directly to `ConformalCalibrator.fit()` / `OnlineConformalCalibrator.update()`
    using the exact upstream inference path (symmetric averaging and non-negative clipping on by
    default; spec §5.2, §5.5). Each `(window, target_variate)` becomes one row of the returned
    arrays: `predictions` `(N, H, Q)`, `actuals` `(N, H)`, and `context_scale` `(N,)`.
    """
    bs = _positive_int("batch_size", batch_size)
    n_windows = len(dataset)
    if n_windows == 0:
        raise ValueError("dataset has no windows")

    horizon = dataset.forecast_horizon
    preds_parts: list[np.ndarray] = []
    actuals_parts: list[np.ndarray] = []
    scales_parts: list[np.ndarray] = []

    was_training = forecaster.model.training
    forecaster.model.eval()
    try:
        idx = 0
        while idx < n_windows:
            first = dataset[idx]
            ctx_len = first.target_context.shape[1]
            group = [first]
            idx += 1
            while idx < n_windows and len(group) < bs:
                nxt = dataset[idx]
                if nxt.target_context.shape[1] != ctx_len:
                    break
                group.append(nxt)
                idx += 1

            context = np.stack([w.target_context for w in group], axis=0)  # (B, V, T)
            actuals_b = np.stack([w.target_future for w in group], axis=0)  # (B, V, H)
            kwargs: dict[str, Any] = {
                "symmetric_averaging": symmetric_averaging,
                "clip_nonnegative": clip_nonnegative,
            }
            if group[0].past.shape[0] > 0:
                kwargs["past_covariates"] = np.stack([w.past for w in group], axis=0)
            if group[0].known.shape[0] > 0:
                kwargs["known_covariates"] = np.stack([w.known for w in group], axis=0)
            if group[0].categorical_past.shape[0] > 0:
                kwargs["categorical_past_covariates"] = np.stack([w.categorical_past for w in group], axis=0)
            if group[0].categorical_known.shape[0] > 0:
                kwargs["categorical_known_covariates"] = np.stack([w.categorical_known for w in group], axis=0)

            forecast = forecaster.forecast(context, prediction_length=horizon, **kwargs)
            q_np = forecast.quantiles.detach().cpu().to(torch.float64).numpy()  # (B, V, H, Q)
            b_size, n_targets, _, num_q = q_np.shape

            # Scale-invariant context std per (window, target_variate): preserve arbitrarily small
            # real scales (e.g. realised variance ~1e-6); on flat non-zero contexts fall back to
            # context magnitude `max(|obs|)` rather than 1.0, and to 1.0 only when identically zero.
            flat_ctx = context.astype(np.float64).reshape(b_size * n_targets, ctx_len)
            scales_b = np.empty(b_size * n_targets, dtype=np.float64)
            for row_i in range(flat_ctx.shape[0]):
                obs = flat_ctx[row_i][np.isfinite(flat_ctx[row_i])]
                if obs.size >= 2:
                    std = float(np.std(obs))
                    mag = float(np.max(np.abs(obs)))
                    if math.isfinite(std) and std > 0.0 and std > 1e-12 * mag:
                        scales_b[row_i] = std
                    elif math.isfinite(mag) and mag > 0.0:
                        scales_b[row_i] = mag
                    else:
                        scales_b[row_i] = 1.0
                elif obs.size == 1 and math.isfinite(float(abs(obs[0]))) and abs(float(obs[0])) > 0.0:
                    scales_b[row_i] = float(abs(obs[0]))
                else:
                    scales_b[row_i] = 1.0

            preds_parts.append(q_np.reshape(b_size * n_targets, horizon, num_q))
            actuals_parts.append(actuals_b.astype(np.float64).reshape(b_size * n_targets, horizon))
            scales_parts.append(scales_b)
    finally:
        forecaster.model.train(was_training)

    return (
        np.concatenate(preds_parts, axis=0),
        np.concatenate(actuals_parts, axis=0),
        np.concatenate(scales_parts, axis=0),
    )
