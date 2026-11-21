"""Trainer callbacks: progress logging and early stopping.

A callback subclasses `Callback` and overrides any of its four hooks; the rest are no-ops. Hooks
receive the trainer, so a callback can read its state or ask it to stop by setting
`trainer.should_stop = True`, from any hook including `on_fit_start`.

Steps are the trainer's global step: it counts optimiser steps from 1 and keeps counting across
`fit` calls. `trainer.fit_start_step` is its value when the current `fit` began, and
`trainer.total_steps` the number of steps one `fit` plans.

- Spec: `docs/SPEC.md` §5.4
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import EvalResult, TimestralTrainer, TrainingResult

__all__ = ["Callback", "EarlyStopping", "ProgressLogger"]

logger = logging.getLogger("timestral_tune.trainer")


class Callback:
    """Base class with no-op hooks. Override the ones you need."""

    def on_fit_start(self, trainer: TimestralTrainer) -> None:
        """Called once, before the first step."""

    def on_step_end(self, trainer: TimestralTrainer, step: int, loss: float, learning_rate: float) -> None:
        """Called after every optimiser step. `step` is the global step, counting from 1 across `fit` calls."""

    def on_evaluate(self, trainer: TimestralTrainer, step: int, result: EvalResult) -> None:
        """Called after every evaluation on the validation set."""

    def on_fit_end(self, trainer: TimestralTrainer, result: TrainingResult) -> None:
        """Called once, after training has finished or stopped."""


class ProgressLogger(Callback):
    """Logs the training loss every `every` steps, and every evaluation, through `logging`."""

    def __init__(self, every: int = 50) -> None:
        if every < 1:
            raise ValueError(f"every must be positive, got {every}")
        self.every = every

    def on_fit_start(self, trainer: TimestralTrainer) -> None:
        logger.info(
            "training %d parameters (%s) for %d steps on %s",
            trainer.num_trainable_parameters,
            trainer.config.mode,
            trainer.total_steps,
            trainer.device,
        )

    def on_step_end(self, trainer: TimestralTrainer, step: int, loss: float, learning_rate: float) -> None:
        done = step - trainer.fit_start_step  # steps of this `fit` call
        if done % self.every == 0 or done == trainer.total_steps:
            logger.info("step %d/%d  loss %.5f  lr %.3g", done, trainer.total_steps, loss, learning_rate)

    def on_evaluate(self, trainer: TimestralTrainer, step: int, result: EvalResult) -> None:
        logger.info(
            "step %d  validation loss %.5f over %d scored positions", step, result.loss, result.scored_positions
        )

    def on_fit_end(self, trainer: TimestralTrainer, result: TrainingResult) -> None:
        logger.info(
            "finished after %d steps%s; best validation loss %s",
            result.steps,
            " (stopped early)" if result.stopped_early else "",
            "n/a" if result.best_eval_loss is None else f"{result.best_eval_loss:.5f} at step {result.best_step}",
        )


class EarlyStopping(Callback):
    """Stops training when the validation loss has not improved for `patience` evaluations.

    An evaluation counts as an improvement when its loss is lower than the best so far by more
    than `min_delta`.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError(f"patience must be positive, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.evaluations_without_improvement = 0

    def on_fit_start(self, trainer: TimestralTrainer) -> None:
        self.best = math.inf
        self.evaluations_without_improvement = 0

    def on_evaluate(self, trainer: TimestralTrainer, step: int, result: EvalResult) -> None:
        if result.loss < self.best - self.min_delta:
            self.best = result.loss
            self.evaluations_without_improvement = 0
            return
        self.evaluations_without_improvement += 1
        if self.evaluations_without_improvement >= self.patience:
            logger.info("early stopping at step %d: no improvement in %d evaluations", step, self.patience)
            trainer.should_stop = True
