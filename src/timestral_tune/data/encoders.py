"""Leak-free categorical covariate encoding.

Timestral takes numeric covariates only, so a categorical covariate is target-encoded before it
reaches the model: each category becomes a smoothed mean of the target over the time steps where
that category occurred. Upstream does this per series at inference, against the context only.

Training must do the same thing per window, or it leaks: an encoding fitted on a whole series
would carry target values from after the forecast origin into the inputs. The functions here
encode one window using nothing after its origin, by calling upstream's `target_encode` on the
window's context. Horizon codes of a known-future covariate are looked up in the context's table;
a category never seen in the context maps to the context's target mean.

These functions reproduce upstream's `TimestralForecaster._encoding_target` and
`._encode_categorical` for a single series, so training inputs equal inference inputs.

- Spec: `docs/SPEC.md` §6.3 rule 4
- Acceptance tests: T12
"""

from __future__ import annotations

import numpy as np

from .._vendor.timestral.covariate_encoding import target_encode

__all__ = ["encode_categorical_window", "encoding_target", "forward_fill"]


def forward_fill(values: np.ndarray) -> np.ndarray:
    """Forward-fill non-finite entries along the last axis; entries before the first finite value stay NaN.

    Returns a new array with the dtype of `values` (which must be floating).
    """
    values = np.asarray(values)
    observed = np.isfinite(values)
    steps = np.arange(values.shape[-1])
    last_seen = np.maximum.accumulate(np.where(observed, steps, -1), axis=-1)
    filled = np.take_along_axis(values, np.maximum(last_seen, 0), axis=-1)
    return np.where(last_seen >= 0, filled, np.nan).astype(values.dtype, copy=False)


def encoding_target(target_context: np.ndarray) -> np.ndarray:
    """The target a window's categorical covariates are encoded against: `(T,)` float64.

    The forward-filled target from its first observation on; NaN before it, so that leading
    padding or a late start does not enter the category means. Values pass through float32 first,
    exactly as they do in upstream's forecaster.

    Args:
        target_context: `(T,)` target values over the window's context; NaN marks a missing value.
    """
    target_context = np.asarray(target_context)
    if target_context.ndim != 1:
        raise ValueError(f"target_context must be (T,), got {target_context.shape}")
    return forward_fill(target_context.astype(np.float32)).astype(np.float64)


def encode_categorical_window(
    target: np.ndarray,
    codes_context: np.ndarray,
    codes_horizon: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Target-encode one categorical covariate for one window, using the context only.

    Args:
        target: `(T,)` float64 encoding target from `encoding_target`.
        codes_context: `(T,)` numeric category codes over the context; NaN marks a missing value,
            which is encoded as a category of its own.
        codes_horizon: `(H,)` codes over the horizon, for a known-future covariate.

    Returns:
        `(encoded_context, encoded_horizon)` as float32; the second is None when `codes_horizon`
        is None. A context position is NaN (to be masked as unobserved) where the code is missing
        and the target has not started: that is padding, not the "missing" category.
    """
    codes_context = np.asarray(codes_context, dtype=np.float64)
    if target.shape != codes_context.shape or target.ndim != 1:
        raise ValueError(f"target {target.shape} and codes_context {codes_context.shape} must both be (T,)")
    if np.isinf(codes_context).any() or (codes_horizon is not None and np.isinf(codes_horizon).any()):
        raise ValueError("category codes must be finite or NaN")
    horizon = None if codes_horizon is None else np.asarray(codes_horizon, dtype=np.float64)
    encoded_context, encoded_horizon = target_encode(target, codes_context, horizon)
    is_padding = np.isnan(target) & np.isnan(codes_context)
    return np.where(is_padding, np.float32(np.nan), encoded_context), encoded_horizon
