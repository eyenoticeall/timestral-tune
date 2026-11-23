"""Shared result, batch-preparation, bookkeeping, and strategy types for `trainer`.

Held in a leaf module with no dependency on `trainer.py` so both `trainer.py` and
`_distributed_strategy.py` can import them at module level without circular imports.

Public types (`EvalResult`, `TrainingResult`) remain re-exported from `timestral_tune.trainer`
and `timestral_tune.trainer.trainer`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor

from ..data.batch import TrainingBatch
from .step import CachedHeadBatch

__all__ = [
    "EvalResult",
    "TrainingResult",
    "_FitBookkeeper",
    "_FitStrategy",
    "_LossTerms",
    "_PreparedDomainBatch",
    "_PreparedReplayBatch",
    "_StepOutcome",
]


@dataclass(frozen=True)
class _PreparedDomainBatch:
    """One domain batch prepared on the target device with its CPU-counted scored total."""

    batch: TrainingBatch | CachedHeadBatch
    num_scored: int


@dataclass(frozen=True)
class _PreparedReplayBatch:
    """One replay micro-batch prepared on the target device for replay pinball and distillation."""

    batch: TrainingBatch | CachedHeadBatch
    real_series: int | None = None
    teacher_quantiles: Tensor | None = None
    num_scored: int | None = None


@dataclass(frozen=True)
class _LossTerms:
    """Unmixed per-term training losses and active position counts from one forward pass.

    `domain_count` is `0` when no domain batch was evaluated, the scored-position count when
    supplied by the caller, or `None` when omitted to avoid an extra device-to-host sync on direct
    `training_step()` calls.
    """

    domain: Tensor | None
    domain_count: int | None = 0
    replay: Tensor | None = None
    replay_count: int = 0
    distill: Tensor | None = None
    distill_count: int = 0


@dataclass(frozen=True)
class EvalResult:
    """Loss over a dataset.

    Attributes:
        loss: Pinball loss in normalised space, averaged over quantile levels and over every
            scored position of the dataset (not a mean of per-batch means).
        scored_positions: Number of scored positions the average is over.
        batches: Number of batches evaluated.
    """

    loss: float
    scored_positions: int
    batches: int


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a `fit` call.

    Attributes:
        steps: Optimiser steps taken.
        epochs: Passes over the training set started.
        final_train_loss: Training loss of the last step taken, or None if no step was taken (a
            callback stopped training before the first one).
        best_eval_loss: Lowest validation loss seen, or None without a validation set.
        best_step: Step of that evaluation, or None.
        stopped_early: Whether a callback stopped training before the planned number of steps.
        skipped_batches: Training batches skipped because they had no scored position.
        restored_best: Whether the trainable parameters were put back to those of `best_step`.
        train_history: `(step, loss)` for every optimiser step.
        eval_history: `(step, loss)` for every evaluation.
    """

    steps: int
    epochs: int
    final_train_loss: float | None
    best_eval_loss: float | None
    best_step: int | None
    stopped_early: bool
    skipped_batches: int
    restored_best: bool
    train_history: list[tuple[int, float]] = field(repr=False)
    eval_history: list[tuple[int, float]] = field(repr=False)


class _FitBookkeeper:
    """Tracks histories, best-validation snapshot, and constructs `TrainingResult` for one `fit()`.

    When `track_best` is False (non-zero ranks in multi-process mode), validation losses are
    recorded in `eval_history` without comparing losses or snapshotting weights; the authoritative
    `(best_eval_loss, best_step)` summary is supplied via `apply_best_summary()` at the end of fit.
    """

    def __init__(self, *, start_step: int, final_step: int, track_best: bool, restore_best: bool) -> None:
        self.start_step = start_step
        self.final_step = final_step
        self.track_best = track_best
        self.restore_best = restore_best
        self.train_history: list[tuple[int, float]] = []
        self.eval_history: list[tuple[int, float]] = []
        self.best_eval_loss: float | None = None
        self.best_step: int | None = None
        self._best_state: list[Tensor] | None = None
        self.skipped_batches = 0
        self.epochs = 0
        self.last_evaluated_step = -1

    def record_step(self, step: int, loss: float) -> None:
        self.train_history.append((step, loss))

    def record_eval(self, step: int, loss: float, trainable: Sequence[Tensor]) -> None:
        self.last_evaluated_step = step
        self.eval_history.append((step, loss))
        if self.track_best and (self.best_eval_loss is None or loss < self.best_eval_loss):
            self.best_eval_loss = loss
            self.best_step = step
            if self.restore_best:
                # Keep best-state snapshot on CPU so `full_ft` does not double GPU VRAM.
                self._best_state = [parameter.detach().cpu().clone() for parameter in trainable]

    def restore_best_weights(self, current_step: int, trainable: Sequence[Tensor], *, non_blocking: bool) -> bool:
        """Copy the best CPU weight snapshot into `trainable` if an earlier step had lower loss."""
        if self._best_state is None or self.best_step == current_step:
            return False
        with torch.no_grad():
            for parameter, saved in zip(trainable, self._best_state, strict=True):
                parameter.copy_(saved, non_blocking=non_blocking)
        return True

    def apply_best_summary(self, best_eval_loss: float | None, best_step: int | None) -> None:
        """Set `(best_eval_loss, best_step)` from an authoritative source (e.g. rank 0 broadcast)."""
        self.best_eval_loss = best_eval_loss
        self.best_step = best_step

    def build_result(self, current_step: int, *, restored_best: bool) -> TrainingResult:
        return TrainingResult(
            steps=current_step - self.start_step,
            epochs=self.epochs,
            final_train_loss=self.train_history[-1][1] if self.train_history else None,
            best_eval_loss=self.best_eval_loss,
            best_step=self.best_step,
            stopped_early=current_step < self.final_step,
            skipped_batches=self.skipped_batches,
            restored_best=restored_best,
            train_history=self.train_history,
            eval_history=self.eval_history,
        )


@dataclass(frozen=True)
class _StepOutcome:
    """Outcome of one loop iteration yielded by a `_FitStrategy`."""

    skipped: bool
    loss: float = 0.0
    learning_rate: float = 0.0


class _FitStrategy(Protocol):
    """Execution strategy plugged into `TimestralTrainer._run_fit_loop`.

    Separates execution mechanics (seeding, loaders, per-epoch step execution, validation,
    callback dispatch and status/loss synchronisation, best-weight restore/broadcast, cleanup)
    from the shared epoch, step, and evaluation loop skeleton.
    """

    @property
    def is_primary(self) -> bool:
        """Whether this process owns best-loss tracking and summary logging (True in 1-process and rank 0)."""
        ...

    def prepare_fit(self) -> None:
        """Seed RNGs and construct training/replay loaders."""
        ...

    def on_fit_start(self) -> None:
        """Run `on_fit_start` callbacks (and sync stop/error status across ranks)."""
        ...

    def iter_epoch_steps(self, epoch: int) -> Iterator[_StepOutcome]:
        """Yield one `_StepOutcome` per loop iteration in 0-based `epoch`."""
        ...

    def on_step_end(self, step: int, loss: float, learning_rate: float) -> float:
        """Run `on_step_end` callbacks (and broadcast `[status, loss]`); return authoritative `loss`."""
        ...

    def evaluate_val(self) -> EvalResult:
        """Evaluate `val_dataset` for `fit()` (rank-local in 1-process, sharded in multi-process)."""
        ...

    def on_evaluate(self, step: int, result: EvalResult, bookkeeper: _FitBookkeeper) -> None:
        """Record eval loss (and best snapshot on primary), run `on_evaluate` callbacks, and sync."""
        ...

    def finalize_best(self, bookkeeper: _FitBookkeeper, current_step: int) -> bool:
        """Restore best weights if warranted (and broadcast summary/weights); return `restored_best`."""
        ...

    def on_fit_end(self, result: TrainingResult) -> None:
        """Run `on_fit_end` callbacks (rank 0 only in multi-process mode)."""
        ...

    def cleanup(self) -> None:
        """Release flip generators, loaders, samplers, and buffers after end-of-fit collectives finish."""
        ...
