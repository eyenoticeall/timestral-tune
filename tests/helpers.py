"""Shared builders for tests: a tiny random model, synthetic raw series, and reference batches.

`upstream_batch` builds a `TrainingBatch` from upstream's own input assembly
(`TimestralForecaster._assemble_rows`). It is the ground truth that the collator is tested against,
and lets the forward and frame tests run independently of the collator.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from timestral_tune._vendor.timestral import TimestralConfig, TimestralForecaster
from timestral_tune.data.batch import TrainingBatch
from timestral_tune.models import TuneTimestralForecaster, TuneTimestralModel
from timestral_tune.trainer.step import compute_loss

QUANTILE_LEVELS = (0.01, *(round(0.05 * i, 2) for i in range(1, 20)), 0.99)

TINY_CONFIG = TimestralConfig(
    patch_size=4,
    d_model=32,
    num_heads=2,
    mlp_hidden_dim=64,
    block_schedule=("time", "time_variate", "time", "time_variate"),
    quantile_knots=QUANTILE_LEVELS,
    block_size=4,
    context_length=64,
)


def make_tiny_forecaster(seed: int = 0) -> TuneTimestralForecaster:
    """A small randomly initialised model with two variate-attention layers. Not a released checkpoint.

    Structural properties (parity with upstream, the role firewall, label isolation) hold for any
    weights, so tests of those run on this model without network access.
    """
    torch.manual_seed(seed)
    model = TuneTimestralModel(TINY_CONFIG)
    with torch.no_grad():
        model.representation.forecast_query.normal_(std=0.5)
    return TuneTimestralForecaster(model.eval(), quantile_real_cap_k=0)


@dataclass(frozen=True)
class RawSeries:
    """Synthetic raw inputs in the shapes upstream's `forecast` takes. NaN marks a missing value."""

    context: np.ndarray  # (B, V, T)
    future: np.ndarray  # (B, V, H) true continuation of the targets
    past: np.ndarray | None  # (B, C_past, T)
    known: np.ndarray | None  # (B, C_known, T + H)

    @property
    def horizon(self) -> int:
        return self.future.shape[-1]

    def forecast_kwargs(self) -> dict[str, np.ndarray | None]:
        return {"past_covariates": self.past, "known_covariates": self.known}


def make_raw(
    seed: int,
    *,
    series: int = 3,
    targets: int = 1,
    past: int = 0,
    known: int = 0,
    context_len: int,
    horizon: int,
    missing: float = 0.03,
) -> RawSeries:
    """Random-walk targets around a level of 100 with some missing values, plus noise covariates."""
    rng = np.random.default_rng(seed)
    walk = rng.standard_normal((series, targets, context_len + horizon)).cumsum(-1) * 3 + 100
    context = walk[..., :context_len].astype(np.float32)
    future = walk[..., context_len:].astype(np.float32)
    context[rng.random(context.shape) < missing] = np.nan
    future[rng.random(future.shape) < missing] = np.nan
    past_values = rng.standard_normal((series, past, context_len)).astype(np.float32) if past else None
    known_values = rng.standard_normal((series, known, context_len + horizon)).astype(np.float32) if known else None
    if past_values is not None:
        past_values[rng.random(past_values.shape) < missing] = np.nan
    return RawSeries(context=context, future=future, past=past_values, known=known_values)


def upstream_batch(forecaster: TimestralForecaster, raw: RawSeries) -> tuple[TrainingBatch, int, tuple[int, int]]:
    """A `TrainingBatch` assembled by upstream's own code, plus `(n_targets, (B, V_all))`."""
    horizon = raw.horizon
    inputs, mask = forecaster._as_context(raw.context, None)
    series, context_len = inputs.shape[0], inputs.shape[-1]
    past = forecaster._as_covariates(raw.past, series, context_len, "past_covariates")
    known = forecaster._as_covariates(raw.known, series, context_len + horizon, "known_covariates")
    rows, row_mask, role_ids, known_future, known_future_mask, n_targets = forecaster._assemble_rows(
        inputs, mask, horizon, past, known
    )
    batch_size, rows_per_series, steps = rows.shape
    patch = forecaster.patch_size
    horizon_steps = -(-horizon // patch) * patch

    targets = torch.zeros(batch_size, rows_per_series, horizon_steps)
    weight = torch.zeros(batch_size, rows_per_series, horizon_steps, dtype=torch.bool)
    future = torch.as_tensor(raw.future, dtype=torch.float32)
    observed = torch.isfinite(future)
    targets[:, :n_targets, :horizon] = torch.where(observed, future, torch.zeros_like(future))
    weight[:, :n_targets, :horizon] = observed

    batch = TrainingBatch(
        context=rows.reshape(batch_size * rows_per_series, steps),
        context_mask=row_mask.reshape(batch_size * rows_per_series, steps),
        group_ids=torch.arange(batch_size).repeat_interleave(rows_per_series),
        role_ids=role_ids,
        known_future=known_future,
        known_future_mask=known_future_mask,
        targets=targets.reshape(batch_size * rows_per_series, horizon_steps),
        loss_weight=weight.reshape(batch_size * rows_per_series, horizon_steps),
    )
    return batch, n_targets, (batch_size, rows_per_series)


def forecaster_on(forecaster: TuneTimestralForecaster, device: str) -> TuneTimestralForecaster:
    """A copy of `forecaster` (same weights and settings) on another device."""
    model = copy.deepcopy(forecaster.model).to(device)
    model.transformer.rope.rebuild_inv_freq()
    return TuneTimestralForecaster(
        model, forecaster.max_context, forecaster.per_shot_horizon_patches, forecaster.quantile_real_cap_k
    )


def as_double(batch: TrainingBatch) -> TrainingBatch:
    """`batch` with float64 values, for the float64-weights reference."""
    return dataclasses.replace(
        batch,
        context=batch.context.double(),
        targets=batch.targets.double(),
        known_future=None if batch.known_future is None else batch.known_future.double(),
    )


def loss_gradient(
    model: TuneTimestralModel, batch: TrainingBatch, frame: str = "median_folded"
) -> tuple[float, Tensor]:
    """The training loss and its gradient with respect to all parameters, flattened to float64 on CPU."""
    was_training = model.training
    model.train()
    model.zero_grad()
    loss = compute_loss(model, batch, frame).loss
    loss.backward()
    gradient = torch.cat([p.grad.detach().cpu().double().flatten() for p in model.parameters()])
    model.zero_grad()
    model.train(was_training)
    return loss.item(), gradient


def max_rel(actual: Tensor, expected: Tensor) -> float:
    """Largest absolute difference relative to the largest expected magnitude."""
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    return ((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-300)).item()


def l2_rel(actual: Tensor, expected: Tensor) -> float:
    """Relative L2 difference."""
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-300)).item()


FRAME_KINDS = ("pandas", "polars", "arrow")


def make_columns(
    seed: int,
    *,
    series: int = 3,
    steps: int = 60,
    targets: int = 1,
    past: int = 0,
    known: int = 0,
    categorical_past: int = 0,
    categorical_known: int = 0,
    missing: float = 0.05,
) -> dict[str, np.ndarray]:
    """Long-format columns: `id`, hourly `time`, targets `y*`, numeric `p*` / `k*`, categorical `cp*` / `ck*`.

    Missing values are NaN in numeric columns and None in categorical ones. Every series is complete
    (no absent time steps) and has `steps` rows.
    """
    rng = np.random.default_rng(seed)
    rows = series * steps
    start = np.datetime64("2024-01-01T00:00", "us")
    columns: dict[str, np.ndarray] = {
        "id": np.repeat(np.array([f"s{i}" for i in range(series)], dtype=object), steps),
        "time": np.tile(start + np.arange(steps) * np.timedelta64(1, "h"), series),
    }

    def numeric(walk: bool) -> np.ndarray:
        noise = rng.standard_normal((series, steps))
        values = (noise.cumsum(-1) * 3 + 100 if walk else noise).reshape(rows)
        values[rng.random(rows) < missing] = np.nan
        return values

    for i in range(targets):
        columns[f"y{i}"] = numeric(walk=True)
    for i in range(past):
        columns[f"p{i}"] = numeric(walk=False)
    for i in range(known):
        columns[f"k{i}"] = numeric(walk=False)
    labels = np.array(["a", "b", "c", "d"], dtype=object)
    for prefix, count in (("cp", categorical_past), ("ck", categorical_known)):
        for i in range(count):
            values = labels[rng.integers(0, len(labels), rows)]
            values[rng.random(rows) < missing] = None
            columns[f"{prefix}{i}"] = values
    return columns


def to_frame(columns: dict[str, np.ndarray], kind: str):
    """`columns` as a pandas or polars DataFrame, or a pyarrow Table."""
    if kind == "pandas":
        import pandas

        return pandas.DataFrame(columns)
    if kind == "polars":
        import polars

        return polars.DataFrame({name: list(v) if v.dtype == object else v for name, v in columns.items()})
    if kind == "arrow":
        import pyarrow

        return pyarrow.table({name: list(v) if v.dtype == object else v for name, v in columns.items()})
    raise ValueError(kind)


def role_kwargs(columns: dict[str, np.ndarray]) -> dict[str, object]:
    """The `from_frame` role arguments for columns made by `make_columns`."""

    def named(prefix: str) -> list[str]:
        return sorted(name for name in columns if name.rstrip("0123456789") == prefix)

    return {
        "id_col": "id",
        "time_col": "time",
        "target_cols": named("y"),
        "past_covariates": named("p"),
        "known_covariates": named("k"),
        "categorical_past_covariates": named("cp"),
        "categorical_known_covariates": named("ck"),
    }


def make_toy_dataset(patch_size: int, *, series: int = 6, seed: int = 0, covariates: bool = False):
    """A small training set sized to a model's patch: noisy seasonal series, optionally with covariates.

    Context is 8 patches, horizon 2 patches, and each series yields a handful of windows.
    """
    from timestral_tune.data import SeriesData, TimeSeriesDataset

    rng = np.random.default_rng(seed)
    steps = 16 * patch_size
    time = np.arange(steps)
    data = []
    for index in range(series):
        period = patch_size * rng.choice([1, 2, 3])
        driver = rng.standard_normal(steps)
        values = 20 + 5 * np.sin(2 * np.pi * time / period + rng.uniform(0, 6)) + rng.standard_t(3, steps)
        if covariates:
            values = values + 2 * driver
        empty = np.empty((0, steps))
        data.append(
            SeriesData(
                series_id=f"toy{index}",
                timestamps=time.astype(np.int64),
                targets=values[None].astype(np.float32),
                past=(rng.standard_normal((1, steps)) if covariates else empty).astype(np.float32),
                known=(driver[None] if covariates else empty).astype(np.float32),
                categorical_past=empty.astype(np.float64),
                categorical_known=empty.astype(np.float64),
            )
        )
    return TimeSeriesDataset(
        data,
        context_length=8 * patch_size,
        forecast_horizon=2 * patch_size,
        stride=2 * patch_size,
        min_context=8 * patch_size,
    )
