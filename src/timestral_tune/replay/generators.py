"""Simple built-in synthetic generators for replay mixing (spec §5.3).

Provides `SyntheticReplaySource`, a deterministic, seeded `TimeSeriesDataset` generated from
simple classical time-series families:

- `"trend_seasonal"`: linear/quadratic trend + multi-frequency sinusoids + noise
- `"random_walk"`: drifted random walk / integrated innovations with volatility bursts
- `"ar"`: stationary AR(1)/AR(2) processes (both monotonic and damped-oscillatory)
- `"regime_shift"`: piecewise level/slope shifts with additive noise
- `"positive_seasonal"`: non-negative seasonal demand/volatility-like patterns with occasional spikes

This is a lightweight regulariser only and makes no claim to match Mistral's unreleased
pretraining corpus (decision D7).

- Spec: `docs/SPEC.md` §5.3, §11 (D7)
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import Literal, get_args

import numpy as np

from ..data.dataset import ColumnRoles, SeriesData, TimeSeriesDataset

__all__ = ["ALL_SYNTHETIC_FAMILIES", "SyntheticFamily", "SyntheticReplaySource"]

SyntheticFamily = Literal["trend_seasonal", "random_walk", "ar", "regime_shift", "positive_seasonal"]
ALL_SYNTHETIC_FAMILIES: tuple[SyntheticFamily, ...] = get_args(SyntheticFamily)


def _sample_scale(rng: np.random.Generator) -> float:
    """Log-uniform scale spanning [1e-2, 1e3] so replay covers diverse magnitudes."""
    return float(10.0 ** rng.uniform(-2.0, 3.0))


def _generate_trend_seasonal(rng: np.random.Generator, length: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    slope = rng.normal(0.0, 1.5)
    quad = rng.normal(0.0, 0.5) if rng.random() < 0.4 else 0.0
    signal = slope * t + quad * (t**2)
    n_harmonics = int(rng.integers(1, 4))
    for _ in range(n_harmonics):
        period = float(10.0 ** rng.uniform(np.log10(6.0), np.log10(max(24.0, min(256.0, length / 2.0)))))
        amp = float(rng.uniform(0.3, 1.5))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        signal = signal + amp * np.sin(2.0 * math.pi * np.arange(length, dtype=np.float64) / period + phase)
    noise_std = float(rng.uniform(0.05, 0.5))
    noise = rng.normal(0.0, noise_std, size=length)
    return _rescale(rng, signal + noise)


def _generate_random_walk(rng: np.random.Generator, length: int) -> np.ndarray:
    drift = float(rng.normal(0.0, 0.03))
    step_scale = np.ones(length, dtype=np.float64)
    if rng.random() < 0.5:
        # Smooth volatility burst via low-frequency envelope.
        burst_center = rng.uniform(0.15, 0.85) * length
        burst_width = max(8.0, rng.uniform(0.05, 0.2) * length)
        burst_gain = rng.uniform(1.5, 4.0)
        idx = np.arange(length, dtype=np.float64)
        step_scale = 1.0 + burst_gain * np.exp(-0.5 * ((idx - burst_center) / burst_width) ** 2)
    innovations = rng.normal(drift, 1.0, size=length) * step_scale
    series = np.cumsum(innovations)
    return _rescale(rng, series)


def _generate_ar(rng: np.random.Generator, length: int) -> np.ndarray:
    burn = 64
    total = length + burn
    if rng.random() < 0.5:
        # Stationary AR(1)
        phi1 = float(rng.uniform(-0.92, 0.96))
        phi2 = 0.0
    else:
        # Stationary AR(2) constructed from roots strictly inside the unit disk.
        if rng.random() < 0.5:
            r1 = float(rng.uniform(-0.88, 0.92))
            r2 = float(rng.uniform(-0.88, 0.92))
            phi1 = r1 + r2
            phi2 = -(r1 * r2)
        else:
            radius = float(rng.uniform(0.5, 0.95))
            angle = float(rng.uniform(0.15, math.pi - 0.15))
            phi1 = 2.0 * radius * math.cos(angle)
            phi2 = -(radius**2)
    eps = rng.normal(0.0, 1.0, size=total)
    x = np.zeros(total, dtype=np.float64)
    for t in range(2, total):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + eps[t]
    return _rescale(rng, x[burn:])


def _generate_regime_shift(rng: np.random.Generator, length: int) -> np.ndarray:
    n_regimes = int(rng.integers(2, 5))
    cuts = np.sort(rng.integers(max(1, length // 8), max(2, length - length // 8), size=n_regimes - 1))
    edges = np.concatenate([[0], cuts, [length]])
    series = np.empty(length, dtype=np.float64)
    level = float(rng.normal(0.0, 1.0))
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        seg_len = int(end - start)
        if seg_len <= 0:
            continue
        slope = float(rng.normal(0.0, 0.015))
        noise_std = float(rng.uniform(0.1, 0.5))
        t = np.arange(seg_len, dtype=np.float64)
        series[start:end] = level + slope * t + rng.normal(0.0, noise_std, size=seg_len)
        level = float(series[end - 1]) + float(rng.normal(0.0, 2.0))
    return _rescale(rng, series)


def _generate_positive_seasonal(rng: np.random.Generator, length: int) -> np.ndarray:
    idx = np.arange(length, dtype=np.float64)
    period1 = float(rng.choice([7.0, 12.0, 24.0, 48.0, 168.0]))
    period2 = float(rng.choice([24.0, 168.0]))
    wave = (
        1.2
        + 0.8 * np.sin(2.0 * math.pi * idx / period1 + rng.uniform(0.0, 2.0 * math.pi))
        + 0.4 * np.sin(2.0 * math.pi * idx / period2 + rng.uniform(0.0, 2.0 * math.pi))
    )
    noise = rng.normal(0.0, 0.25, size=length)
    raw = np.maximum(0.0, wave + noise)
    # Occasional demand/volatility spikes.
    n_spikes = int(rng.integers(0, max(1, length // 64) + 1))
    if n_spikes > 0:
        spike_pos = rng.integers(0, length, size=n_spikes)
        raw[spike_pos] += rng.uniform(1.0, 4.0, size=n_spikes)
    scale = _sample_scale(rng)
    return (raw * scale).astype(np.float32)


def _rescale(rng: np.random.Generator, values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std > 1e-12:
        values = (values - float(np.mean(values))) / std
    scale = _sample_scale(rng)
    offset = float(rng.normal(0.0, 2.0 * scale))
    return (values * scale + offset).astype(np.float32)


_GENERATORS = {
    "trend_seasonal": _generate_trend_seasonal,
    "random_walk": _generate_random_walk,
    "ar": _generate_ar,
    "regime_shift": _generate_regime_shift,
    "positive_seasonal": _generate_positive_seasonal,
}


class SyntheticReplaySource(TimeSeriesDataset):
    """Deterministic `TimeSeriesDataset` of synthetic univariate series for replay mixing.

    Subclasses `TimeSeriesDataset` directly so it works seamlessly with `DHFDataCollator`,
    `TimestralTrainer`, frozen-feature caching, `evaluate_forecaster`, and `dataset_fingerprint`.

    Args:
        num_series: Number of synthetic series to generate.
        context_length: Context length in time steps for each window.
        forecast_horizon: Forecast horizon in time steps for each window.
        series_length: Total length of each generated series. Defaults to
            `context_length + 4 * forecast_horizon` (producing multiple windows per series).
        stride: Distance between consecutive forecast origins. Defaults to `forecast_horizon`.
        seed: Random seed controlling all generated series (fully deterministic).
        missing_rate: Optional fraction of steps randomly masked as `NaN` in `[0.0, 0.5)`.
            At least one step in every window's context and horizon stays observed.
        families: Subset of synthetic generator families to cycle/sample across. Defaults to all
            five built-in families.
    """

    def __init__(
        self,
        *,
        num_series: int = 64,
        context_length: int = 256,
        forecast_horizon: int = 32,
        series_length: int | None = None,
        stride: int | None = None,
        seed: int = 0,
        missing_rate: float = 0.0,
        families: Sequence[SyntheticFamily] = ALL_SYNTHETIC_FAMILIES,
    ) -> None:
        if not isinstance(num_series, numbers.Integral) or isinstance(num_series, bool) or num_series < 1:
            raise ValueError(f"num_series must be a positive integer, got {num_series!r}")
        if not isinstance(context_length, numbers.Integral) or isinstance(context_length, bool) or context_length < 1:
            raise ValueError(f"context_length must be a positive integer, got {context_length!r}")
        if (
            not isinstance(forecast_horizon, numbers.Integral)
            or isinstance(forecast_horizon, bool)
            or forecast_horizon < 1
        ):
            raise ValueError(f"forecast_horizon must be a positive integer, got {forecast_horizon!r}")
        if not isinstance(seed, numbers.Integral) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        if isinstance(missing_rate, bool) or not (
            isinstance(missing_rate, numbers.Real) and math.isfinite(missing_rate) and 0.0 <= missing_rate < 0.5
        ):
            raise ValueError(f"missing_rate must be in [0, 0.5), got {missing_rate!r}")
        if isinstance(families, str) or len(families) == 0:
            raise ValueError(f"families must be a non-empty sequence of {ALL_SYNTHETIC_FAMILIES}")
        invalid = [f for f in families if f not in _GENERATORS]
        if invalid:
            raise ValueError(f"unknown synthetic families {invalid!r}; expected entries from {ALL_SYNTHETIC_FAMILIES}")

        num_series = int(num_series)
        context_length = int(context_length)
        forecast_horizon = int(forecast_horizon)
        resolved_length = context_length + 4 * forecast_horizon if series_length is None else int(series_length)
        if resolved_length < context_length + forecast_horizon:
            raise ValueError(
                f"series_length ({resolved_length}) must be at least context_length + forecast_horizon "
                f"({context_length + forecast_horizon})"
            )
        resolved_stride = forecast_horizon if stride is None else int(stride)

        rng = np.random.default_rng(int(seed))
        family_tuple = tuple(families)
        series_list: list[SeriesData] = []
        series_families: list[str] = []
        timestamps = np.arange(resolved_length, dtype=np.int64)
        empty_f32 = np.empty((0, resolved_length), dtype=np.float32)
        empty_f64 = np.empty((0, resolved_length), dtype=np.float64)

        for idx in range(num_series):
            family = family_tuple[idx % len(family_tuple)]
            values = _GENERATORS[family](rng, resolved_length).copy()
            if missing_rate > 0.0:
                drop = rng.random(resolved_length) < float(missing_rate)
                # Keep endpoints observed so context and horizon always contain observed targets.
                drop[0] = False
                drop[-1] = False
                values[drop] = np.nan
            series_families.append(family)
            series_list.append(
                SeriesData(
                    series_id=f"synth_{family}_{idx:04d}",
                    timestamps=timestamps,
                    targets=values[None, :],
                    past=empty_f32,
                    known=empty_f32,
                    categorical_past=empty_f64,
                    categorical_known=empty_f64,
                )
            )

        self.seed = int(seed)
        self.missing_rate = float(missing_rate)
        self.families: tuple[str, ...] = family_tuple
        self.series_families: tuple[str, ...] = tuple(series_families)
        super().__init__(
            series_list,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
            stride=resolved_stride,
            min_context=context_length,
            roles=ColumnRoles(targets=("target",)),
        )
