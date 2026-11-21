"""Post-hoc conformal calibration for Timestral's 21-quantile forecasts.

Two distribution-free calibrators operating on raw-space forecast quantiles without modifying
model weights (spec §5.2, decision D6):

- `ConformalCalibrator`: static split Conformalized Quantile Regression (Romano et al., 2019,
  eqs. 9-11), with scale-normalised conformity scores, horizon buckets, finite-sample quantile
  level `(1 - alpha) * (1 + 1 / n)`, and outward monotonicity repair from the unshifted median.
- `OnlineConformalCalibrator`: Adaptive Conformal Inference (Gibbs & Candès, 2021) for
  non-exchangeable series, maintaining per-bucket working miscoverage `alpha_t` over a rolling
  score window. `alpha_t` is never clipped (ACI Lemma 4.1); saturation at `alpha_t <= 0` uses the
  largest score in the window and raises `.saturated`.

- Spec: `docs/SPEC.md` §5.2, §7.3, §11 (D6)
- Acceptance tests: T13
"""

from __future__ import annotations

import json
import math
import numbers
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "DEFAULT_QUANTILE_LEVELS",
    "ConformalCalibrator",
    "OnlineConformalCalibrator",
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


def _repair_monotonicity_outward(quantiles: np.ndarray, median_idx: int) -> np.ndarray:
    """Outward monotonicity repair from the unshifted median (spec §5.2): intervals can only widen."""
    out = quantiles.copy()
    # Upper half: q[k] = max(q[k], q[k - 1]) moving outward from median.
    out[..., median_idx:] = np.maximum.accumulate(out[..., median_idx:], axis=-1)
    # Lower half: q[k] = min(q[k], q[k + 1]) moving outward from median.
    out[..., : median_idx + 1] = np.minimum.accumulate(out[..., median_idx::-1], axis=-1)[..., ::-1]
    return out


def _apply_offsets(
    preds: np.ndarray,
    scale: np.ndarray,
    pairs: tuple[tuple[float, int, int], ...],
    offsets: np.ndarray,
    bucket_size: int,
    median_idx: int,
) -> np.ndarray:
    """Apply `(num_pairs, num_buckets)` scaled offsets and outward monotonicity repair."""
    _, h, _ = preds.shape
    bucket_ids = np.arange(h) // bucket_size
    if bucket_ids[-1] >= offsets.shape[1]:
        raise ValueError(
            f"horizon {h} requires bucket {int(bucket_ids[-1])}, but calibrator only has {offsets.shape[1]} bucket(s)"
        )
    out = preds.copy()
    for pair_idx, (_, low_idx, high_idx) in enumerate(pairs):
        shift = offsets[pair_idx, bucket_ids][None, :] * scale[:, None]
        out[..., low_idx] -= shift
        out[..., high_idx] += shift
    return _repair_monotonicity_outward(out, median_idx)


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
                level = cov * (1.0 + 1.0 / n_valid)
                if level >= 1.0:
                    offsets[pair_idx, b] = float(np.max(scores))
                else:
                    offsets[pair_idx, b] = float(np.quantile(scores, level, method="higher"))

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
        """Apply calibrated shifts and outward monotonicity repair to raw-space quantile forecasts."""
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
        if data.get("kind") != "cqr":
            raise ValueError(f"expected kind='cqr', got {data.get('kind')!r}")
        cal = cls(
            target_coverage=data["coverages"],
            bucket_size=data["bucket_size"],
            quantile_levels=data["quantile_levels"],
        )
        if data.get("offsets") is not None:
            cal.offsets_ = np.asarray(data["offsets"], dtype=np.float64)
            cal.bucket_counts_ = (
                None if data.get("bucket_counts") is None else np.asarray(data["bucket_counts"], dtype=np.int64)
            )
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

    Maintains a rolling score window and unclipped working miscoverage `alpha_t` per
    `(coverage_pair, horizon_bucket)`. When `alpha_t <= 0`, the largest score in the window is
    used and `.saturated` is True; when `alpha_t >= 1`, the smallest score is used.

    Args:
        target_coverage: Nominal central coverage(s) or `"all"`.
        gamma: Online step size in `alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`.
        window: Rolling window length of recent scaled conformity scores per `(pair, bucket)`.
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
        # Per-bucket state grows lazily as horizons are seen.
        self._num_buckets: int = 0
        self._alpha_t: list[list[float]] = [[] for _ in self.pairs]
        self._scores: list[list[deque[float]]] = [[] for _ in self.pairs]

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
                self._scores[pair_idx].append(deque(maxlen=self.window))
        self._num_buckets = num_buckets

    def _bucket_offset(self, pair_idx: int, bucket_idx: int) -> tuple[float, bool]:
        scores = self._scores[pair_idx][bucket_idx]
        if not scores:
            return 0.0, False
        alpha_t = self._alpha_t[pair_idx][bucket_idx]
        arr = np.fromiter(scores, dtype=np.float64, count=len(scores))
        if alpha_t <= 0.0:
            return float(np.max(arr)), True
        if alpha_t >= 1.0:
            return float(np.min(arr)), False
        return float(np.quantile(arr, 1.0 - alpha_t, method="higher")), False

    @property
    def alpha_t(self) -> np.ndarray:
        """Current unclipped working miscoverage `(num_pairs, num_buckets)`."""
        if self._num_buckets == 0:
            return np.empty((len(self.pairs), 0), dtype=np.float64)
        return np.asarray(self._alpha_t, dtype=np.float64)

    @property
    def saturated(self) -> bool:
        """True if any calibrated `(coverage, bucket)` currently has `alpha_t <= 0` with scores present."""
        return any(
            bool(self._scores[p][b]) and self._alpha_t[p][b] <= 0.0
            for p in range(len(self.pairs))
            for b in range(self._num_buckets)
        )

    def current_offsets(self, horizon: int) -> np.ndarray:
        num_buckets = math.ceil(horizon / self.bucket_size)
        self._ensure_buckets(num_buckets)
        offsets = np.zeros((len(self.pairs), num_buckets), dtype=np.float64)
        for p in range(len(self.pairs)):
            for b in range(num_buckets):
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
        """Apply the current online offsets and outward monotonicity repair to raw-space forecasts."""
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
        """Update working miscoverages and rolling score windows chronologically as actuals arrive."""
        preds, y, scale, _, _, _ = _validate_inputs(predictions, len(self.quantile_levels), actuals, context_scale)
        assert y is not None
        n, h, _ = preds.shape
        num_buckets = math.ceil(h / self.bucket_size)
        self._ensure_buckets(num_buckets)

        for i in range(n):
            s_i = float(scale[i])
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
                    q_cur, _ = self._bucket_offset(p, b)
                    q_low = preds[i, start:end, low_idx][valid]
                    q_high = preds[i, start:end, high_idx][valid]
                    lo = q_low - q_cur * s_i
                    hi = q_high + q_cur * s_i
                    err = float(np.mean((y_v < lo) | (y_v > hi)))
                    # ACI update: alpha_t is never clipped (Lemma 4.1).
                    self._alpha_t[p][b] = self._alpha_t[p][b] + self.gamma * (alpha - err)
                    new_scores = np.maximum(q_low - y_v, y_v - q_high) / s_i
                    self._scores[p][b].extend(float(v) for v in new_scores)
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
            "scores": [[list(dq) for dq in row] for row in self._scores],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OnlineConformalCalibrator:
        if data.get("kind") != "aci":
            raise ValueError(f"expected kind='aci', got {data.get('kind')!r}")
        cal = cls(
            target_coverage=data["coverages"],
            gamma=data["gamma"],
            window=data["window"],
            bucket_size=data["bucket_size"],
            quantile_levels=data["quantile_levels"],
        )
        num_buckets = int(data.get("num_buckets", 0))
        cal._ensure_buckets(num_buckets)
        for p in range(len(cal.pairs)):
            for b in range(num_buckets):
                cal._alpha_t[p][b] = float(data["alpha_t"][p][b])
                cal._scores[p][b] = deque((float(v) for v in data["scores"][p][b]), maxlen=cal.window)
        return cal

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> OnlineConformalCalibrator:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
