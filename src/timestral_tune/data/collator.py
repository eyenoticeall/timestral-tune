"""DHF batch collator: raw windows in, packed `TrainingBatch` out.

The collator turns the windows a `TimeSeriesDataset` yields into the exact inputs upstream's
forecaster would build for the same data (`TimestralForecaster._assemble_rows`), so that what the
model sees in training is what it sees at inference:

- target rows are forward-filled, with the observation mask kept;
- covariate rows hold 0 where unobserved;
- categorical covariates are target-encoded per window against the context only (for a single
  target variate; several target variates receive the raw codes, as upstream does) and are
  appended after the numeric covariates of the same kind;
- the row order within a series is targets, past-only covariates, known-future covariates;
- contexts are left-padded with unobserved zeros and horizons right-padded to whole patches.

One thing differs from upstream by design: every window in a batch is left-padded to the same
context length, not just to the next patch multiple. Leading all-unobserved patches are never
attended to and do not shift positions, so this does not change the model's output; a test asserts
that.

Horizon target values go to `TrainingBatch.targets` only. They are labels and are never model
inputs (spec §6.3 rule 1).

- Spec: `docs/SPEC.md` §3.3, §3.4, §4.2, §6.2, §6.3
- Acceptance tests: T1, T2, T4, T5, T8, T12
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .._vendor.timestral.config import TimestralConfig
from .._vendor.timestral.forecaster import ffill_with_mask
from .._vendor.timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET
from .batch import TrainingBatch
from .encoders import encode_categorical_window, encoding_target

__all__ = ["DHFDataCollator", "Window"]


@dataclass(frozen=True)
class Window:
    """One forecast-origin window of one series, as raw values. NaN marks a missing value.

    `L` is the number of context steps available (at most the dataset's `context_length`; fewer
    near the start of a series) and `H` the forecast horizon.

    Attributes:
        series_index: Index of the series in the dataset.
        origin: Index, within the series, of the first horizon step.
        target_context: `(V, L)` target variates over the context.
        target_future: `(V, H)` target variates over the horizon. Labels only.
        past: `(C_past, L)` numeric past-only covariates.
        known: `(C_known, L + H)` numeric known-future covariates.
        categorical_past: `(D_past, L)` category codes of past-only categorical covariates.
        categorical_known: `(D_known, L + H)` category codes of known-future categorical covariates.
    """

    series_index: int
    origin: int
    target_context: np.ndarray
    target_future: np.ndarray
    past: np.ndarray
    known: np.ndarray
    categorical_past: np.ndarray
    categorical_known: np.ndarray

    @property
    def context_steps(self) -> int:
        return self.target_context.shape[-1]

    @property
    def horizon(self) -> int:
        return self.target_future.shape[-1]

    @property
    def layout(self) -> tuple[int, int, int, int, int, int]:
        """`(V, C_past, C_known, D_past, D_known, H)`: must agree across the windows of a batch."""
        return (
            self.target_context.shape[0],
            self.past.shape[0],
            self.known.shape[0],
            self.categorical_past.shape[0],
            self.categorical_known.shape[0],
            self.horizon,
        )


def _round_up(steps: int, patch_size: int) -> int:
    return -(-steps // patch_size) * patch_size


class DHFDataCollator:
    """Collates `Window`s into a `TrainingBatch` for a given model.

    Args:
        patch_size: The model's patch size.
        context_length: Context length every batch is padded to, in time steps; rounded up to a
            whole number of patches. Use the dataset's `context_length`.
        max_context: The model's context cap (`config.context_length`).
        max_horizon: The longest horizon one training forward can cover
            (`config.block_size * config.patch_size`).

    Use `DHFDataCollator.for_model(config, context_length)` to fill the model-dependent arguments.
    The collator holds no tensors and no model, so it is cheap to copy into DataLoader workers.
    """

    def __init__(self, patch_size: int, context_length: int, max_context: int, max_horizon: int) -> None:
        if patch_size < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if context_length < 1:
            raise ValueError(f"context_length must be positive, got {context_length}")
        if context_length > max_context:
            raise ValueError(f"context_length {context_length} exceeds the model's context cap of {max_context} steps")
        self.patch_size = patch_size
        self.context_length = context_length
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.padded_context = _round_up(context_length, patch_size)

    @classmethod
    def for_model(cls, config: TimestralConfig, context_length: int) -> DHFDataCollator:
        """A collator sized to a model's patch size, context cap and decode block."""
        return cls(
            patch_size=config.patch_size,
            context_length=context_length,
            max_context=config.context_length,
            max_horizon=config.block_size * config.patch_size,
        )

    def __repr__(self) -> str:
        return (
            f"DHFDataCollator(patch_size={self.patch_size}, context_length={self.context_length}, "
            f"max_context={self.max_context}, max_horizon={self.max_horizon})"
        )

    def _check(self, windows: Sequence[Window]) -> tuple[int, int, int, int, int, int]:
        if len(windows) == 0:
            raise ValueError("cannot collate an empty list of windows")
        layout = windows[0].layout
        for window in windows:
            if window.layout != layout:
                raise ValueError(
                    "windows in one batch must have the same number of target, covariate and categorical rows and "
                    f"the same horizon; got {window.layout} and {layout}"
                )
            if not 1 <= window.context_steps <= self.context_length:
                raise ValueError(
                    f"a window has {window.context_steps} context steps; expected between 1 and {self.context_length}"
                )
        targets, _, _, _, _, horizon = layout
        if targets < 1:
            raise ValueError("windows must have at least one target variate")
        if not 1 <= horizon <= self.max_horizon:
            raise ValueError(
                f"forecast horizon {horizon} is outside 1..{self.max_horizon}, the range one training forward covers"
            )
        return layout

    def _encode_categoricals(self, window: Window) -> tuple[np.ndarray, np.ndarray]:
        """Categorical rows as float32 model inputs: `(D_past, L)` and `(D_known, L + H)`."""
        codes_past, codes_known = window.categorical_past, window.categorical_known
        context_steps = window.context_steps
        if window.target_context.shape[0] != 1:
            # Several target variates: there is no single target to encode against, so upstream feeds the codes.
            return codes_past.astype(np.float32), codes_known.astype(np.float32)
        target = encoding_target(window.target_context[0])
        past = np.empty(codes_past.shape, dtype=np.float32)
        for row, codes in enumerate(codes_past):
            past[row], _ = encode_categorical_window(target, codes)
        known = np.empty(codes_known.shape, dtype=np.float32)
        for row, codes in enumerate(codes_known):
            known[row, :context_steps], known[row, context_steps:] = encode_categorical_window(
                target, codes[:context_steps], codes[context_steps:]
            )
        return past, known

    def __call__(self, windows: Sequence[Window]) -> TrainingBatch:
        n_targets, n_past, n_known, d_past, d_known, horizon = self._check(windows)
        batch_size = len(windows)
        past_rows, known_rows = n_past + d_past, n_known + d_known
        rows_per_series = n_targets + past_rows + known_rows
        context_steps = self.padded_context
        horizon_steps = _round_up(horizon, self.patch_size)

        # Raw values with NaN for "missing"; left padding stays NaN and so ends up unobserved.
        context = np.full((batch_size, rows_per_series, context_steps), np.nan, dtype=np.float32)
        known_future = np.full((batch_size, rows_per_series, horizon_steps), np.nan, dtype=np.float32)
        future = np.full((batch_size, rows_per_series, horizon_steps), np.nan, dtype=np.float32)
        known_slice = slice(n_targets + past_rows, rows_per_series)
        for index, window in enumerate(windows):
            steps = window.context_steps
            start = context_steps - steps
            encoded_past, encoded_known = self._encode_categoricals(window)
            past = np.concatenate([window.past, encoded_past], axis=0)
            known = np.concatenate([window.known, encoded_known], axis=0)
            context[index, :n_targets, start:] = window.target_context
            context[index, n_targets : n_targets + past_rows, start:] = past
            context[index, known_slice, start:] = known[:, :steps]
            known_future[index, known_slice, :horizon] = known[:, steps:]
            future[index, :n_targets, :horizon] = window.target_future

        for name, array in (("context", context), ("known-future covariates", known_future), ("targets", future)):
            if np.isinf(array).any():
                raise ValueError(f"{name} hold infinite values; use NaN for missing values and remove infinities")

        values = torch.from_numpy(context)
        observed = torch.isfinite(values)
        values = torch.where(observed, values, torch.zeros_like(values))
        # Forward-fill target rows only, keeping the mask: upstream's `_assemble_rows`.
        values[:, :n_targets] = ffill_with_mask(values[:, :n_targets], observed[:, :n_targets])

        labels = torch.from_numpy(future)
        scored = torch.isfinite(labels)
        labels = torch.where(scored, labels, torch.zeros_like(labels))

        total_rows = batch_size * rows_per_series
        role_ids = None
        known_values = known_mask = None
        if past_rows + known_rows > 0:
            roles = [ROLE_TARGET] * n_targets + [ROLE_PAST_COV] * past_rows + [ROLE_KNOWN_COV] * known_rows
            role_ids = torch.tensor(roles, dtype=torch.long).repeat(batch_size)
        if known_rows > 0:
            known_tensor = torch.from_numpy(known_future)
            known_mask = torch.isfinite(known_tensor)
            known_values = torch.where(known_mask, known_tensor, torch.zeros_like(known_tensor))
            known_values = known_values.reshape(total_rows, horizon_steps)
            known_mask = known_mask.reshape(total_rows, horizon_steps)

        return TrainingBatch(
            context=values.reshape(total_rows, context_steps),
            context_mask=observed.reshape(total_rows, context_steps),
            group_ids=torch.arange(batch_size).repeat_interleave(rows_per_series),
            role_ids=role_ids,
            known_future=known_values,
            known_future_mask=known_mask,
            targets=labels.reshape(total_rows, horizon_steps),
            loss_weight=scored.reshape(total_rows, horizon_steps),
        )
