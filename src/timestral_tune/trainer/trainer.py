"""The training loop.

`TimestralTrainer` loads a checkpoint, freezes everything the chosen adaptation mode does not
train, and minimises the pinball loss of spec §4 over the windows of a `TimeSeriesDataset`.

Scope in this release (roadmap Phase 1): the `head_only` mode in float32, on CPU, CUDA or the Apple
GPU. The LoRA modes, full fine-tuning, `bf16` autocast, `torch.compile` and `export()` arrive in
Phase 2; asking for them raises `NotImplementedError` rather than doing something else.

- Spec: `docs/SPEC.md` §4, §5, §5.4, §7.1
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import TrainingConfig
from ..data.batch import TrainingBatch
from ..data.dataset import TimeSeriesDataset
from ..models.device import TuneTimestralModel, supports_float64
from ..models.loader import load_model, resolve_device
from .callbacks import Callback, EarlyStopping, ProgressLogger
from .step import compute_loss

__all__ = ["EvalResult", "TimestralTrainer", "TrainingResult"]

logger = logging.getLogger("timestral_tune.trainer")


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
        final_train_loss: Training loss of the last step taken.
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
    final_train_loss: float
    best_eval_loss: float | None
    best_step: int | None
    stopped_early: bool
    skipped_batches: int
    restored_best: bool
    train_history: list[tuple[int, float]] = field(repr=False)
    eval_history: list[tuple[int, float]] = field(repr=False)


def _learning_rate_factor(step: int, warmup_steps: int, total_steps: int, schedule: str) -> float:
    """Multiplier on the peak learning rate for the optimiser step with 0-based index `step`."""
    if step < warmup_steps:
        return (step + 1) / (warmup_steps + 1)
    if schedule == "constant":
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class TimestralTrainer:
    """Fine-tunes a Timestral checkpoint on a `TimeSeriesDataset`.

    Args:
        config: Training settings.
        train_dataset: Windows to train on.
        val_dataset: Optional windows to evaluate on; enables early stopping and best-state
            restoration.
        lora_config: Reserved for the LoRA modes (roadmap Phase 2).
        callbacks: Extra callbacks, run after the built-in progress logger and early stopping.
        model: An already loaded model to train, instead of loading `config.model_name_or_path`.
            It is moved to the configured device and modified in place.
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_dataset: TimeSeriesDataset,
        val_dataset: TimeSeriesDataset | None = None,
        *,
        lora_config: object | None = None,
        callbacks: Sequence[Callback] = (),
        model: TuneTimestralModel | None = None,
    ) -> None:
        if config.mode != "head_only":
            raise NotImplementedError(
                f"mode {config.mode!r} is not implemented yet; only 'head_only' is (the others are roadmap Phase 2)"
            )
        if lora_config is not None:
            raise NotImplementedError("LoRA is not implemented yet (roadmap Phase 2); pass lora_config=None")
        if config.mixed_precision != "fp32":
            raise NotImplementedError("mixed precision is not implemented yet (roadmap Phase 2); use 'fp32'")

        self.config = config
        self.device = resolve_device(config.device)
        if model is None:
            model = load_model(config.model_name_or_path, device=self.device, revision=config.revision)
        elif not isinstance(model, TuneTimestralModel):
            raise TypeError("model must be a TuneTimestralModel; convert with TuneTimestralModel.from_upstream")
        else:
            dtype = next(model.parameters()).dtype
            if not supports_float64(self.device) and dtype != torch.float32:
                raise ValueError(f"the {self.device.type} backend supports float32 weights only, got {dtype}")
            model.to(self.device)
            model.transformer.rope.rebuild_inv_freq()
        self.model = model

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self._check_dataset(train_dataset, "train_dataset")
        if val_dataset is not None:
            self._check_dataset(val_dataset, "val_dataset")
            if (val_dataset.series[0].layout, val_dataset.forecast_horizon) != (
                train_dataset.series[0].layout,
                train_dataset.forecast_horizon,
            ):
                raise ValueError(
                    "val_dataset must have the same columns per role and forecast horizon as train_dataset"
                )

        self._trainable = self._select_trainable()
        self.optimizer = torch.optim.AdamW(
            self._trainable, lr=config.resolved_learning_rate, weight_decay=config.weight_decay
        )
        steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
        self.total_steps = config.max_steps if config.max_steps is not None else config.epochs * steps_per_epoch
        self.scheduler = self._new_schedule()

        self.callbacks: list[Callback] = [ProgressLogger(config.log_every)]
        if config.early_stopping_patience is not None:
            if val_dataset is None:
                raise ValueError("early_stopping_patience needs a val_dataset")
            self.callbacks.append(EarlyStopping(config.early_stopping_patience))
        self.callbacks.extend(callbacks)

        self.global_step = 0
        self.should_stop = False

    # ----------------------------------------------------------------- setup

    def _check_dataset(self, dataset: TimeSeriesDataset, name: str) -> None:
        """Fail at construction, not at the first batch, if the dataset cannot be used with this model."""
        model_config = self.model.config
        dataset.collator(model_config)  # raises if context_length exceeds the model's context cap
        max_horizon = model_config.block_size * model_config.patch_size
        if dataset.forecast_horizon > max_horizon:
            raise ValueError(
                f"{name}: forecast_horizon {dataset.forecast_horizon} exceeds the {max_horizon} steps one training "
                "forward covers (one decode block)"
            )
        _, past, known, categorical_past, categorical_known = dataset.series[0].layout
        if past + known + categorical_past + categorical_known > 0 and not model_config.covariate_enabled:
            raise ValueError(f"{name} has covariates, but this checkpoint was not trained with covariates")
        if known + categorical_known > 0 and not model_config.supports_known_covariates:
            raise ValueError(f"{name} has known-future covariates, which this checkpoint cannot condition on")

    def _select_trainable(self) -> list[Tensor]:
        """Freeze everything, then unfreeze what `config.mode` trains. Returns the trainable parameters."""
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        trainable = list(self.model.head.parameters())  # head_only: the quantile output projection
        for parameter in trainable:
            parameter.requires_grad_(True)
        return trainable

    def _new_schedule(self) -> torch.optim.lr_scheduler.LambdaLR:
        """A fresh warm-up and decay schedule over `total_steps`, starting at the peak learning rate."""
        config = self.config
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] = config.resolved_learning_rate
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: _learning_rate_factor(step, config.warmup_steps, self.total_steps, config.lr_schedule),
        )

    @property
    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self._trainable)

    def _loader(self, dataset: TimeSeriesDataset, *, training: bool) -> DataLoader:
        config = self.config
        generator = torch.Generator().manual_seed(config.seed) if training else None
        batch_size = config.batch_size if training else (config.eval_batch_size or config.batch_size)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training and config.shuffle,
            collate_fn=dataset.collator(self.model.config),
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )

    # ------------------------------------------------------------------ loop

    def training_step(self, batch: TrainingBatch) -> float:
        """One optimiser step on `batch` (already on the device). Returns the loss."""
        self.optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(self.model, batch, self.config.target_frame).loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"the training loss became {loss.item()} at step {self.global_step + 1}. Lower the learning rate, "
                "or check the data for extreme values"
            )
        loss.backward()
        if self.config.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self._trainable, self.config.gradient_clip_val)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def fit(self) -> TrainingResult:
        """Train for the configured number of steps and return a summary.

        With a validation set, evaluation runs every `eval_every` steps (or at the end of each
        epoch) and once more at the end if the last step was not evaluated.

        Calling `fit` again continues from the current weights and optimiser state with a fresh
        learning-rate schedule and the same batch order. It seeds PyTorch's global generator.
        """
        config = self.config
        torch.manual_seed(config.seed)
        self.scheduler = self._new_schedule()
        loader = self._loader(self.train_dataset, training=True)
        train_history: list[tuple[int, float]] = []
        eval_history: list[tuple[int, float]] = []
        best_loss: float | None = None
        best_step: int | None = None
        best_state: list[Tensor] | None = None
        skipped = epochs = 0
        last_evaluated_step = -1
        self.should_stop = False
        start_step = self.global_step
        final_step = start_step + self.total_steps

        def evaluate_now() -> None:
            nonlocal best_loss, best_step, best_state, last_evaluated_step
            result = self.evaluate()
            last_evaluated_step = self.global_step
            eval_history.append((self.global_step, result.loss))
            if best_loss is None or result.loss < best_loss:
                best_loss, best_step = result.loss, self.global_step
                if config.restore_best:
                    best_state = [parameter.detach().clone() for parameter in self._trainable]
            for callback in self.callbacks:
                callback.on_evaluate(self, self.global_step, result)

        for callback in self.callbacks:
            callback.on_fit_start(self)
        self.model.train()
        try:
            while self.global_step < final_step and not self.should_stop:
                epochs += 1
                steps_before_epoch = self.global_step
                for batch in loader:
                    if batch.num_scored == 0:
                        skipped += 1
                        continue
                    learning_rate = self.optimizer.param_groups[0]["lr"]
                    loss = self.training_step(batch.to(self.device))
                    self.global_step += 1
                    train_history.append((self.global_step, loss))
                    for callback in self.callbacks:
                        callback.on_step_end(self, self.global_step, loss, learning_rate)
                    due = config.eval_every is not None and (self.global_step - start_step) % config.eval_every == 0
                    if self.val_dataset is not None and due:
                        evaluate_now()
                    if self.global_step >= final_step or self.should_stop:
                        break
                if self.global_step == steps_before_epoch:
                    raise RuntimeError(
                        f"an epoch produced no optimiser step: all {skipped} batches had no scored position"
                    )
                if self.val_dataset is not None and config.eval_every is None and not self.should_stop:
                    evaluate_now()
            if self.val_dataset is not None and last_evaluated_step != self.global_step:
                evaluate_now()
        finally:
            self.model.eval()

        restored = False
        if best_state is not None and best_step != self.global_step:
            with torch.no_grad():
                for parameter, saved in zip(self._trainable, best_state, strict=True):
                    parameter.copy_(saved)
            restored = True
        if skipped:
            logger.warning("skipped %d training batches that had no scored position", skipped)

        result = TrainingResult(
            steps=self.global_step - start_step,
            epochs=epochs,
            final_train_loss=train_history[-1][1],
            best_eval_loss=best_loss,
            best_step=best_step,
            stopped_early=self.global_step < final_step,
            skipped_batches=skipped,
            restored_best=restored,
            train_history=train_history,
            eval_history=eval_history,
        )
        for callback in self.callbacks:
            callback.on_fit_end(self, result)
        return result

    @torch.no_grad()
    def evaluate(self, dataset: TimeSeriesDataset | None = None) -> EvalResult:
        """The loss over `dataset` (default: the validation set), in the configured target frame."""
        dataset = self.val_dataset if dataset is None else dataset
        if dataset is None:
            raise ValueError("no dataset to evaluate: pass one, or give the trainer a val_dataset")
        self._check_dataset(dataset, "dataset")
        was_training = self.model.training
        self.model.eval()
        try:
            return self._evaluate(self._loader(dataset, training=False))
        finally:
            self.model.train(was_training)

    def _evaluate(self, batches: Iterable[TrainingBatch]) -> EvalResult:
        total = 0.0
        scored = count = 0
        for batch in batches:
            count += 1
            positions = batch.num_scored
            if positions == 0:
                continue
            loss = compute_loss(self.model, batch.to(self.device), self.config.target_frame).loss
            # Weight each batch by its scored positions, so the result is the mean over the dataset.
            total += loss.item() * positions  # .item() is a Python float, so the sum accumulates in float64
            scored += positions
        if scored == 0:
            raise ValueError("the dataset has no scored position to evaluate")
        return EvalResult(loss=total / scored, scored_positions=scored, batches=count)

    def export(self, output_dir: str | Path | None = None) -> None:
        """Write an upstream-loadable checkpoint (spec §7.4). Not implemented yet: roadmap Phase 2."""
        raise NotImplementedError(
            "export() is not implemented yet (roadmap Phase 2). The trained model is available as trainer.model"
        )
