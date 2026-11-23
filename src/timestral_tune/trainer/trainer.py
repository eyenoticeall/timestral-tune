"""The training loop.

`TimestralTrainer` loads a checkpoint, freezes everything the chosen adaptation mode does not
train (`head_only`, `lora_variate`, `lora_temporal`, `full_ft`), and minimises the pinball loss
of spec §4 over the windows of a `TimeSeriesDataset`.

- Spec: `docs/SPEC.md` §4, §5, §5.1, §5.4, §7.1
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import logging
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from ..config import LoraConfig, ReplayConfig, TrainingConfig
from ..data.batch import TrainingBatch, flip_batch_signs
from ..data.collator import DHFDataCollator, Window
from ..data.dataset import TimeSeriesDataset
from ..export import _validate_export_inputs, dataset_fingerprint, export_checkpoint
from ..models.adapter import disable_lora, has_lora, inject_lora, lora_parameters, unmerge_lora, validate_lora_for_rows
from ..models.conformal import ConformalCalibrator, OnlineConformalCalibrator
from ..models.device import TuneTimestralModel, supports_float64
from ..models.forward import context_welford_state, normalise_targets, target_frames, training_forward
from ..models.loader import load_model, resolve_device
from ..replay import SyntheticReplaySource, distillation_loss
from .callbacks import Callback, EarlyStopping, ProgressLogger
from .distributed import (
    DistributedContext,
    broadcast_module_state,
    is_oom_error,
    steps_per_epoch,
)
from .step import CachedHeadBatch, compute_loss

__all__ = ["EvalResult", "TimestralTrainer", "TrainingResult"]

logger = logging.getLogger("timestral_tune.trainer")

# Cap per-dataset CPU memory for cached `head_only` representations and frozen-base teacher outputs (256 MiB).
_MAX_HEAD_CACHE_BYTES = 256 * 1024 * 1024


class _IndexedReplayDataset(Dataset[tuple[int, Window]]):
    """Wraps a `TimeSeriesDataset` to emit `(window_index, Window)` for cached teacher lookup."""

    def __init__(self, dataset: TimeSeriesDataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[int, Window]:
        return index, self._dataset[index]


@dataclass(frozen=True)
class _ReplayBatchItem:
    """One collated replay micro-batch paired with its window indices and unpadded series count."""

    batch: TrainingBatch
    indices: Tensor
    real_series: int

    def pin_memory(self) -> _ReplayBatchItem:
        return _ReplayBatchItem(batch=self.batch.pin_memory(), indices=self.indices, real_series=self.real_series)


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


def _apply_series_flip(batch: TrainingBatch, flip_mask: Tensor, rows_per_series: int) -> TrainingBatch:
    """Negate every row of each real series `i` where `flip_mask[i]` is True; padded tail series stay +1."""
    real_series = int(flip_mask.numel())
    row_signs = torch.ones((batch.num_rows,), dtype=batch.context.dtype, device=batch.device)
    real_signs = torch.where(flip_mask, -1.0, 1.0).to(dtype=batch.context.dtype, device=batch.device)
    row_signs[: real_series * rows_per_series] = real_signs.repeat_interleave(rows_per_series)
    signs = row_signs.unsqueeze(-1)
    return TrainingBatch(
        context=batch.context * signs,
        context_mask=batch.context_mask,
        group_ids=batch.group_ids,
        role_ids=batch.role_ids,
        known_future=None if batch.known_future is None else batch.known_future * signs,
        known_future_mask=batch.known_future_mask,
        targets=batch.targets * signs,
        loss_weight=batch.loss_weight,
    )


def _cycle_loader(loader: DataLoader) -> Iterator[Any]:
    while True:
        yield from loader


class _CachedHeadDataset(Dataset[int]):
    """Per-window frozen-body representations on CPU, indexed directly by window index."""

    def __init__(
        self,
        horizon_hidden: Tensor,
        anchor_loc: Tensor,
        anchor_scale: Tensor,
        targets: Tensor,
        loss_weight: Tensor,
        context_welford: tuple[Tensor, Tensor, Tensor] | None,
        normalised_targets: Tensor | None,
        frame_loc: Tensor | None,
        frame_scale: Tensor | None,
        negated: _CachedHeadDataset | None = None,
    ) -> None:
        self.horizon_hidden = horizon_hidden
        self.anchor_loc = anchor_loc
        self.anchor_scale = anchor_scale
        self.targets = targets
        self.loss_weight = loss_weight
        self.context_welford = context_welford
        self.normalised_targets = normalised_targets
        self.frame_loc = frame_loc
        self.frame_scale = frame_scale
        self.negated = negated

    def __len__(self) -> int:
        return self.horizon_hidden.shape[0]

    def __getitem__(self, index: int) -> int:
        return index

    def collate(self, indices: Sequence[int]) -> CachedHeadBatch:
        idx = torch.as_tensor(indices, dtype=torch.long)
        hidden = self.horizon_hidden[idx].flatten(0, 1)
        loc = self.anchor_loc[idx].flatten(0, 1)
        scale = self.anchor_scale[idx].flatten(0, 1)
        targets = self.targets[idx].flatten(0, 1)
        weight = self.loss_weight[idx].flatten(0, 1)
        welford = None if self.context_welford is None else tuple(t[idx].flatten(0, 1) for t in self.context_welford)
        normalised = None if self.normalised_targets is None else self.normalised_targets[idx].flatten(0, 1)
        f_loc = None if self.frame_loc is None else self.frame_loc[idx].flatten(0, 1)
        f_scale = None if self.frame_scale is None else self.frame_scale[idx].flatten(0, 1)
        return CachedHeadBatch(
            horizon_hidden=hidden,
            anchor_loc=loc,
            anchor_scale=scale,
            targets=targets,
            loss_weight=weight,
            context_welford=welford,
            normalised_targets=normalised,
            frame_loc=f_loc,
            frame_scale=f_scale,
        )

    def collate_sign_flipped(self, indices: Sequence[int], generator: torch.Generator) -> CachedHeadBatch:
        """Collate `indices`, randomly selecting per-window from `+x` or cached `-x` with `p=0.5`."""
        if self.negated is None:
            return self.collate(indices)
        idx = torch.as_tensor(indices, dtype=torch.long)
        flip = torch.rand((idx.numel(),), generator=generator) < 0.5  # (B,)

        def _pick(pos_t: Tensor, neg_t: Tensor) -> Tensor:
            p_sel = pos_t[idx]
            n_sel = neg_t[idx]
            mask = flip.view(-1, *([1] * (p_sel.ndim - 1)))
            return torch.where(mask, n_sel, p_sel).flatten(0, 1)

        neg = self.negated
        welford = (
            None
            if self.context_welford is None or neg.context_welford is None
            else tuple(_pick(pt, nt) for pt, nt in zip(self.context_welford, neg.context_welford, strict=True))
        )
        normalised = (
            None
            if self.normalised_targets is None or neg.normalised_targets is None
            else _pick(self.normalised_targets, neg.normalised_targets)
        )
        f_loc = None if self.frame_loc is None or neg.frame_loc is None else _pick(self.frame_loc, neg.frame_loc)
        f_scale = (
            None if self.frame_scale is None or neg.frame_scale is None else _pick(self.frame_scale, neg.frame_scale)
        )
        return CachedHeadBatch(
            horizon_hidden=_pick(self.horizon_hidden, neg.horizon_hidden),
            anchor_loc=_pick(self.anchor_loc, neg.anchor_loc),
            anchor_scale=_pick(self.anchor_scale, neg.anchor_scale),
            targets=_pick(self.targets, neg.targets),
            loss_weight=_pick(self.loss_weight, neg.loss_weight),
            context_welford=welford,
            normalised_targets=normalised,
            frame_loc=f_loc,
            frame_scale=f_scale,
        )


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

    def set_last_eval_loss(self, loss: float) -> None:
        """Overwrite the most recent `eval_history` loss with an authoritative value (e.g. rank 0)."""
        step, _ = self.eval_history[-1]
        self.eval_history[-1] = (step, loss)

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
        """Seed RNGs, attach per-fit flip generators, and construct training/replay loaders."""
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
        """Release per-fit state inside `fit()`'s `finally:` block."""
        ...


class _SingleProcessStrategy:
    """Default one-process execution strategy for `TimestralTrainer._run_fit_loop`."""

    def __init__(self, trainer: TimestralTrainer) -> None:
        self._trainer = trainer
        self._flip_generator: torch.Generator | None = None
        self._replay_flip_generator: torch.Generator | None = None
        self._loader: Iterable[TrainingBatch | CachedHeadBatch] | None = None
        self._replay_iter: Iterator[Any] | None = None

    @property
    def is_primary(self) -> bool:
        return True

    def prepare_fit(self) -> None:
        trainer = self._trainer
        config = trainer.config
        torch.manual_seed(config.seed)
        self._flip_generator = torch.Generator().manual_seed(config.seed + 1_000_003) if config.sign_flip else None
        trainer._fit_flip_generator = self._flip_generator
        self._replay_flip_generator = (
            torch.Generator().manual_seed(config.seed + 3_000_007) if trainer._replay_sign_flip else None
        )
        trainer._replay_flip_generator = self._replay_flip_generator
        self._loader = trainer._loader(trainer.train_dataset, training=True)
        self._replay_iter = _cycle_loader(trainer._replay_loader()) if trainer.replay_config is not None else None

    def on_fit_start(self) -> None:
        for callback in self._trainer.callbacks:
            callback.on_fit_start(self._trainer)

    def iter_epoch_steps(self, epoch: int) -> Iterator[_StepOutcome]:
        del epoch
        assert self._loader is not None
        trainer = self._trainer
        for batch in self._loader:
            prepared = trainer._prepare_domain_batch(batch, self._flip_generator)
            if prepared is None:
                yield _StepOutcome(skipped=True)
                continue
            prepared_replay = (
                trainer._prepare_replay_batch(next(self._replay_iter), self._replay_flip_generator)
                if self._replay_iter is not None
                else None
            )
            learning_rate = trainer.optimizer.param_groups[0]["lr"]
            loss = trainer.training_step(
                prepared.batch,
                validate=False,
                replay_batch=None if prepared_replay is None else prepared_replay.batch,
                replay_real_series=None if prepared_replay is None else prepared_replay.real_series,
                replay_teacher_quantiles=None if prepared_replay is None else prepared_replay.teacher_quantiles,
            )
            yield _StepOutcome(skipped=False, loss=loss, learning_rate=learning_rate)

    def on_step_end(self, step: int, loss: float, learning_rate: float) -> float:
        for callback in self._trainer.callbacks:
            callback.on_step_end(self._trainer, step, loss, learning_rate)
        return loss

    def evaluate_val(self) -> EvalResult:
        return self._trainer.evaluate()

    def on_evaluate(self, step: int, result: EvalResult, bookkeeper: _FitBookkeeper) -> None:
        bookkeeper.record_eval(step, result.loss, self._trainer._trainable)
        for callback in self._trainer.callbacks:
            callback.on_evaluate(self._trainer, step, result)

    def finalize_best(self, bookkeeper: _FitBookkeeper, current_step: int) -> bool:
        return bookkeeper.restore_best_weights(
            current_step,
            self._trainer._trainable,
            non_blocking=self._trainer.device.type == "cuda",
        )

    def on_fit_end(self, result: TrainingResult) -> None:
        for callback in self._trainer.callbacks:
            callback.on_fit_end(self._trainer, result)

    def cleanup(self) -> None:
        self._trainer._fit_flip_generator = None
        self._trainer._replay_flip_generator = None
        self._flip_generator = None
        self._replay_flip_generator = None
        self._loader = None
        self._replay_iter = None


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
        lora_config: Optional `LoraConfig` for `lora_variate` and `lora_temporal` (defaults to a
            mode-appropriate config when omitted; rejected for `head_only` and `full_ft`).
        replay_config: Optional `ReplayConfig` enabling replay mixing and/or frozen-base
            distillation (spec §5.3). When distilling, the frozen base is the unadapted base model
            (via `disable_lora`) for LoRA modes even when resuming a `load_lora` checkpoint, and
            the starting `model` weights at construction time for `head_only` and `full_ft`.
        replay_dataset: Optional `TimeSeriesDataset` supplying replay windows when `replay_config`
            is set. Must not be `val_dataset`. When omitted and `replay_config` is set, a
            deterministic `SyntheticReplaySource` matching `train_dataset`'s context and horizon
            is created. May have a different covariate layout or horizon from `train_dataset`.
        callbacks: Extra `Callback` instances, run after the built-in progress logger and early stopping.
        model: An already loaded model to train, instead of loading `config.model_name_or_path`.
            It is moved to the configured device and modified in place. For LoRA modes, a model
            already adapted via `load_lora()` is resumed in place.
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_dataset: TimeSeriesDataset,
        val_dataset: TimeSeriesDataset | None = None,
        *,
        lora_config: LoraConfig | None = None,
        replay_config: ReplayConfig | None = None,
        replay_dataset: TimeSeriesDataset | None = None,
        callbacks: Sequence[Callback] = (),
        model: TuneTimestralModel | None = None,
    ) -> None:
        self.config = config
        self._dist: DistributedContext | None = DistributedContext.detect(config.device)
        if self._dist is not None:
            self._init_distributed(
                self._dist,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                lora_config=lora_config,
                replay_config=replay_config,
                replay_dataset=replay_dataset,
                callbacks=callbacks,
                model=model,
            )
            return

        # One-process path: (i) validate/load -> (iii) runtime setup -> (ii) callbacks,
        # keeping exact historical check, exception, and message ordering.
        self.device = resolve_device(config.device)
        self._init_core_components(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            lora_config=lora_config,
            replay_config=replay_config,
            replay_dataset=replay_dataset,
            model=model,
            distributed=False,
        )
        self._init_runtime(epoch_steps=math.ceil(len(train_dataset) / config.batch_size))
        self.callbacks = self._build_callbacks(val_dataset, callbacks)

    # ----------------------------------------------------------------- setup

    def _init_core_components(
        self,
        *,
        train_dataset: TimeSeriesDataset,
        val_dataset: TimeSeriesDataset | None,
        lora_config: LoraConfig | None,
        replay_config: ReplayConfig | None,
        replay_dataset: TimeSeriesDataset | None,
        model: TuneTimestralModel | None,
        distributed: bool,
    ) -> None:
        """Shared constructor step (i): validate args, load/move model, check datasets, select trainables."""
        config = self.config
        if config.mixed_precision == "bf16":
            if self.device.type != "cuda":
                raise ValueError(
                    f"mixed_precision='bf16' is supported on CUDA only (MPS runs fp32 only); got {self.device.type!r}"
                )
            if not torch.cuda.is_bf16_supported():
                raise ValueError("mixed_precision='bf16' requested, but this CUDA device does not support bfloat16")

        if distributed and config.compile_model and config.compile_mode == "reduce-overhead":
            raise ValueError(
                "compile_mode='reduce-overhead' is not supported in multi-process mode; "
                "use compile_mode='default' or compile_model=False"
            )

        if lora_config is not None and not isinstance(lora_config, LoraConfig):
            raise TypeError(f"lora_config must be a LoraConfig or None, got {type(lora_config).__name__}")
        explicit_lora_config = lora_config is not None
        self.lora_config = self._resolve_lora_config(config.mode, lora_config)
        self._explicit_lora_config = explicit_lora_config

        if replay_config is not None and not isinstance(replay_config, ReplayConfig):
            raise TypeError(f"replay_config must be a ReplayConfig or None, got {type(replay_config).__name__}")
        if replay_dataset is not None and replay_config is None:
            raise ValueError("replay_dataset requires a ReplayConfig")
        if replay_dataset is not None and not isinstance(replay_dataset, TimeSeriesDataset):
            raise TypeError(f"replay_dataset must be a TimeSeriesDataset or None, got {type(replay_dataset).__name__}")
        if val_dataset is not None and replay_dataset is val_dataset:
            raise ValueError("replay_dataset must not be the same object as val_dataset")
        if distributed and replay_config is not None:
            raise NotImplementedError(
                "multi-process training with replay_config is not implemented yet (planned for Stage 2)"
            )
        self.replay_config = replay_config

        self._model_provided_directly = model is not None
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

        self.replay_dataset: TimeSeriesDataset | None = None
        if replay_config is not None:
            if replay_dataset is None:
                replay_dataset = SyntheticReplaySource(
                    num_series=64,
                    context_length=train_dataset.context_length,
                    forecast_horizon=train_dataset.forecast_horizon,
                    seed=config.seed + 2_000_003,
                )
            self._check_dataset(replay_dataset, "replay_dataset")
            if len(replay_dataset) == 0:
                raise ValueError("replay_dataset has no windows")
            self.replay_dataset = replay_dataset

        self._trainable = self._select_trainable()

    def _build_callbacks(self, val_dataset: TimeSeriesDataset | None, callbacks: Sequence[Callback]) -> list[Callback]:
        """Shared constructor step (ii): validate and build the callback list."""
        built: list[Callback] = [ProgressLogger(self.config.log_every)]
        if self.config.early_stopping_patience is not None:
            if val_dataset is None:
                raise ValueError("early_stopping_patience needs a val_dataset")
            built.append(EarlyStopping(self.config.early_stopping_patience))
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise TypeError(f"callbacks must be Callback instances, got {type(callback).__name__}")
        built.extend(callbacks)
        return built

    def _init_runtime(self, *, epoch_steps: int) -> None:
        """Shared constructor step (iii): teacher, compile, optimiser, schedule, and mutable fit state."""
        config = self.config
        self._frozen_head: torch.nn.Module | None = None
        self._frozen_base: TuneTimestralModel | None = None
        self._replay_base_pos: Tensor | None = None
        self._replay_base_neg: Tensor | None = None
        if self.replay_config is not None:
            self._setup_replay_teacher()

        if config.compile_model:
            self.model.transformer = torch.compile(self.model.transformer, mode=config.compile_mode)

        use_fused = self.device.type == "cuda" and all(p.is_cuda and p.is_floating_point() for p in self._trainable)
        adamw_kwargs: dict[str, object] = {
            "lr": config.resolved_learning_rate,
            "weight_decay": config.weight_decay,
        }
        if use_fused:
            adamw_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(self._trainable, **adamw_kwargs)
        self.total_steps = config.max_steps if config.max_steps is not None else config.epochs * epoch_steps
        self.scheduler = self._new_schedule()

        self.global_step = 0
        self.fit_start_step = 0
        self.should_stop = False
        self._train_head_cache: _CachedHeadDataset | None = None
        self._val_head_cache: _CachedHeadDataset | None = None
        self._replay_head_cache: _CachedHeadDataset | None = None
        self._fit_flip_generator: torch.Generator | None = None
        self._replay_flip_generator: torch.Generator | None = None
        self._val_loader: DataLoader | None = None

    def _log_distributed_startup(self, dist_ctx: DistributedContext) -> None:
        """Emit one-time rank-0 startup notices for multi-process configuration."""
        if not dist_ctx.is_rank_zero:
            return
        config = self.config
        effective_batch = dist_ctx.world_size * config.batch_size
        logger.info(
            "multi-process training enabled: world_size=%d backend=%s per_rank_batch_size=%d effective_batch_size=%d",
            dist_ctx.world_size,
            dist_ctx.backend,
            config.batch_size,
            effective_batch,
        )
        if config.learning_rate is None and effective_batch > 32:
            logger.warning(
                "effective batch size is %d (%d ranks x %d per rank) with the default learning rate %g "
                "(defaults were swept at batch sizes 16-32 and are not scaled automatically; set "
                "learning_rate explicitly if needed)",
                effective_batch,
                dist_ctx.world_size,
                config.batch_size,
                config.resolved_learning_rate,
            )
        if config.mode == "head_only" and config.cache_frozen_features:
            logger.info(
                "multi-process mode skips the head_only frozen-feature cache (cache_frozen_features=True "
                "is ignored); running the uncached forward path"
            )

    def _init_distributed(
        self,
        dist_ctx: DistributedContext,
        *,
        train_dataset: TimeSeriesDataset,
        val_dataset: TimeSeriesDataset | None,
        lora_config: LoraConfig | None,
        replay_config: ReplayConfig | None,
        replay_dataset: TimeSeriesDataset | None,
        callbacks: Sequence[Callback],
        model: TuneTimestralModel | None,
    ) -> None:
        """Multi-process startup sequence (spec §5.6, D28–D29, steps (a)–(e)).

        Backend check and device resolution/binding have already run in `DistributedContext.detect()`
        outside failure propagation.

        Ranks: all ranks must call in lockstep.
        Collectives: 1 (`propagate_failure` in step (b)) + 2 (`check_consistency` in step (c))
            + `P + B` (`broadcast_module_state` in step (d), one per unique parameter/buffer).
        """
        config = self.config
        self.device = dist_ctx.device

        # Step (a): shared steps (i) and (ii) plus multi-process trainable invariants inside failure protection.
        local_error: Exception | None = None
        try:
            self._init_core_components(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                lora_config=lora_config,
                replay_config=replay_config,
                replay_dataset=replay_dataset,
                model=model,
                distributed=True,
            )
            if not self._trainable:
                raise ValueError("no trainable parameters selected for multi-process training")
            trainable_dtypes = {parameter.dtype for parameter in self._trainable}
            if len(trainable_dtypes) != 1:
                dtype_list = ", ".join(sorted(str(d) for d in trainable_dtypes))
                raise ValueError(
                    f"multi-process training requires all trainable parameters to share one dtype; got {dtype_list}"
                )
            if any(parameter.device != self.device for parameter in self._trainable):
                raise ValueError(f"all trainable parameters must reside on {self.device}")
            self.callbacks = self._build_callbacks(val_dataset, callbacks)
        except Exception as exc:
            if is_oom_error(exc):
                raise
            local_error = exc

        # Step (b): startup failure collective.
        dist_ctx.propagate_failure(local_error, stage="startup")

        # Step (c): consistency digest check (mismatched ranks raise ValueError; matching peers raise RankFailureError).
        dist_ctx.check_consistency(self._startup_consistency_digest(), stage="startup_consistency")

        # Step (d): broadcast all parameters and all buffers from rank 0.
        broadcast_module_state(dist_ctx, self.model)

        # Step (e): shared step (iii) with global epoch step count, plus rank-0 notices.
        epoch_steps = steps_per_epoch(len(train_dataset), dist_ctx.world_size, config.batch_size)
        self._init_runtime(epoch_steps=epoch_steps)
        self._log_distributed_startup(dist_ctx)

    def _startup_consistency_digest(self) -> bytes:
        """Compute a 32-byte SHA-256 digest over model schema, config, and dataset fingerprints.

        Hashes each present dataset (`train_dataset`, `val_dataset`, `replay_dataset`) once via
        `dataset_fingerprint`, which is linear in the dataset's total array bytes.
        """
        config_fields: dict[str, Any] = {}
        for field_info in dataclasses.fields(self.config):
            if field_info.name in ("device", "output_dir"):
                continue
            value = getattr(self.config, field_info.name)
            if isinstance(value, Path):
                value = str(value)
            elif hasattr(value, "item") and callable(value.item):
                value = value.item()
            config_fields[field_info.name] = value

        param_names_by_id = {id(parameter): name for name, parameter in self.model.named_parameters()}
        trainable_names = [param_names_by_id[id(parameter)] for parameter in self._trainable]

        def _dataset_entry(dataset: TimeSeriesDataset | None) -> dict[str, Any] | None:
            if dataset is None:
                return None
            return {"length": len(dataset), "fingerprint": dataset_fingerprint(dataset)}

        payload = {
            "model_config": self.model.config.to_dict(),
            "parameters": [
                [name, list(parameter.shape), str(parameter.dtype)]
                for name, parameter in sorted(self.model.named_parameters(), key=lambda item: item[0])
            ],
            "buffers": [
                [name, list(buf.shape), str(buf.dtype)]
                for name, buf in sorted(self.model.named_buffers(), key=lambda item: item[0])
            ],
            "trainable": trainable_names,
            "training_config": config_fields,
            "lora_config": None if self.lora_config is None else self.lora_config.to_dict(),
            "replay_config": None if self.replay_config is None else self.replay_config.to_dict(),
            "train_dataset": _dataset_entry(self.train_dataset),
            "val_dataset": _dataset_entry(self.val_dataset),
            "replay_dataset": _dataset_entry(self.replay_dataset),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).digest()

    @staticmethod
    def _resolve_lora_config(mode: str, lora_config: LoraConfig | None) -> LoraConfig | None:
        if mode in ("head_only", "full_ft"):
            if lora_config is not None:
                raise ValueError(f"lora_config is only valid for LoRA modes, not mode={mode!r}")
            return None
        required_axis = "variate" if mode == "lora_variate" else "time"
        if lora_config is None:
            return LoraConfig(axes=(required_axis,))
        if required_axis not in lora_config.axes:
            raise ValueError(f"mode={mode!r} requires {required_axis!r} in lora_config.axes, got {lora_config.axes!r}")
        return lora_config

    def _autocast(self):
        if self.config.mixed_precision == "bf16":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

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
        """Freeze everything, then unfreeze (or inject) what `config.mode` trains. Returns trainable parameters."""
        mode = self.config.mode
        if mode in ("head_only", "full_ft") and has_lora(self.model):
            raise ValueError(
                f"cannot run mode={mode!r} on a model that already has LoRA adapters; "
                "use a LoRA mode or load the merged weights into a base model"
            )
        if mode == "full_ft":
            trainable = list(self.model.parameters())
            for parameter in trainable:
                parameter.requires_grad_(True)
            return trainable

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        if mode == "head_only":
            trainable = list(self.model.head.parameters())
            for parameter in trainable:
                parameter.requires_grad_(True)
            return trainable

        assert self.lora_config is not None
        rows_per_series = sum(self.train_dataset.series[0].layout)
        if has_lora(self.model):
            existing = getattr(self.model, "_lora_config", None)
            if existing is not None:
                if self._explicit_lora_config and existing != self.lora_config:
                    raise ValueError(
                        f"pre-injected model has {existing!r}, "
                        f"which does not match passed lora_config={self.lora_config!r}"
                    )
                self.lora_config = self._resolve_lora_config(mode, existing)
            validate_lora_for_rows(self.lora_config, rows_per_series)
            unmerge_lora(self.model)
        else:
            generator = torch.Generator().manual_seed(self.config.seed)
            inject_lora(self.model, self.lora_config, rows_per_series=rows_per_series, generator=generator)
        trainable = list(lora_parameters(self.model))
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

    def _build_data_loader(
        self,
        dataset: TimeSeriesDataset,
        *,
        batch_size: int | None = None,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
        batch_sampler: Sampler[list[int]] | None = None,
        indices: Sequence[int] | None = None,
        bucket_to_max: int | None = None,
    ) -> DataLoader:
        """Build a `DataLoader` over `dataset` (or an index `Subset`) with a model-matched `DHFDataCollator`."""
        if batch_sampler is not None and (
            batch_size is not None or shuffle or generator is not None or indices is not None
        ):
            raise ValueError("batch_sampler cannot be combined with batch_size, shuffle, generator, or indices")
        if batch_sampler is None and batch_size is None:
            raise ValueError("either batch_size or batch_sampler must be specified")
        config = self.config
        collator = DHFDataCollator.for_model(self.model.config, dataset.context_length, bucket_to_max=bucket_to_max)
        source: Dataset[Any] = dataset if indices is None else Subset(dataset, indices)
        common: dict[str, Any] = {
            "collate_fn": collator,
            "num_workers": config.num_workers,
            "persistent_workers": config.num_workers > 0,
            "pin_memory": self.device.type == "cuda",
        }
        if batch_sampler is not None:
            return DataLoader(source, batch_sampler=batch_sampler, **common)
        return DataLoader(source, batch_size=batch_size, shuffle=shuffle, generator=generator, **common)

    def _raw_loader(self, dataset: TimeSeriesDataset, *, training: bool) -> DataLoader:
        config = self.config
        generator = torch.Generator().manual_seed(config.seed) if training else None
        batch_size = config.batch_size if training else (config.eval_batch_size or config.batch_size)
        bucket_max = batch_size if (config.compile_model and config.compile_mode == "reduce-overhead") else None
        return self._build_data_loader(
            dataset,
            batch_size=batch_size,
            shuffle=training and config.shuffle,
            generator=generator,
            bucket_to_max=bucket_max,
        )

    def _estimate_head_cache_bytes(self, dataset: TimeSeriesDataset) -> int:
        model_config = self.model.config
        patch = model_config.patch_size
        n_targets = dataset.series[0].layout[0]
        horizon_patches = -(-dataset.forecast_horizon // patch)
        horizon_steps = horizon_patches * patch
        bytes_per_target = (
            horizon_patches * model_config.d_model * 4  # horizon_hidden (fp32)
            + horizon_steps * 5  # targets (fp32) + loss_weight (bool)
            + 8  # anchor_loc + anchor_scale (fp32)
            + (24 if self.config.target_frame == "median_folded" and horizon_patches > 1 else 0)  # 3x fp64 Welford
            + (horizon_steps * 4 if self.config.target_frame == "true_values" else 0)  # precomputed normalised targets
        )
        return len(dataset) * n_targets * bytes_per_target

    @property
    def _replay_sign_flip(self) -> bool:
        if self.replay_config is None:
            return False
        return self.config.sign_flip if self.replay_config.sign_flip is None else self.replay_config.sign_flip

    def _estimate_replay_base_cache_bytes(self, dataset: TimeSeriesDataset) -> int:
        model_config = self.model.config
        patch = model_config.patch_size
        n_targets = dataset.series[0].layout[0]
        horizon_steps = -(-dataset.forecast_horizon // patch) * patch
        bytes_per_target = horizon_steps * model_config.num_quantiles * 4  # fp32 normalised quantiles
        return len(dataset) * n_targets * bytes_per_target * (2 if self._replay_sign_flip else 1)

    def _setup_replay_teacher(self) -> None:
        """Prepare the step-0 frozen base reference for replay distillation (before `torch.compile`)."""
        assert self.replay_config is not None and self.replay_dataset is not None
        if self.replay_config.distill_weight <= 0.0:
            return
        if self.config.mode == "head_only":
            frozen_head = copy.deepcopy(self.model.head).to(self.device).eval()
            for parameter in frozen_head.parameters():
                parameter.requires_grad_(False)
            self._frozen_head = frozen_head
            return

        estimated = self._estimate_replay_base_cache_bytes(self.replay_dataset)
        if self.replay_config.cache_base_outputs and estimated <= _MAX_HEAD_CACHE_BYTES:
            self._replay_base_pos = self._precompute_replay_base_outputs(self.replay_dataset, negate=False)
            if self._replay_sign_flip:
                self._replay_base_neg = self._precompute_replay_base_outputs(self.replay_dataset, negate=True)
            return

        if self.replay_config.cache_base_outputs and estimated > _MAX_HEAD_CACHE_BYTES:
            logger.info(
                "skipping frozen-base teacher cache for %d replay windows (estimated %.1f MiB exceeds %.0f MiB cap)",
                len(self.replay_dataset),
                estimated / (1024 * 1024),
                _MAX_HEAD_CACHE_BYTES / (1024 * 1024),
            )
        if self.config.mode == "full_ft":
            frozen_base = copy.deepcopy(self.model).to(self.device).eval()
            for parameter in frozen_base.parameters():
                parameter.requires_grad_(False)
            self._frozen_base = frozen_base

    @torch.no_grad()
    def _precompute_replay_base_outputs(self, dataset: TimeSeriesDataset, *, negate: bool) -> Tensor:
        """Precompute `(N, n_targets, horizon_steps, Q)` frozen-base normalised quantiles on CPU."""
        patch = self.model.config.patch_size
        n_targets = dataset.series[0].layout[0]
        batch_size = self.config.eval_batch_size or self.config.batch_size
        collator = DHFDataCollator.for_model(self.model.config, dataset.context_length, bucket_to_max=None)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        non_blocking = self.device.type == "cuda"
        outputs: list[Tensor] = []
        was_training = self.model.training
        self.model.eval()
        try:
            with disable_lora(self.model):
                for batch in loader:
                    if negate:
                        batch = flip_batch_signs(batch, p=1.0)
                    batch.validate(patch)
                    on_device = batch.to(self.device, non_blocking=non_blocking)
                    with self._autocast():
                        out = training_forward(self.model, on_device, validate=False)
                    target_q = out.quantiles[on_device.target_rows].detach().to(torch.float32).cpu()
                    series_count = target_q.shape[0] // n_targets
                    outputs.append(target_q.unflatten(0, (series_count, n_targets)))
        finally:
            self.model.train(was_training)
        return torch.cat(outputs, dim=0)

    def _get_head_cache(
        self,
        dataset: TimeSeriesDataset,
        *,
        slot: Literal["train", "val", "replay"] | None = None,
    ) -> _CachedHeadDataset | None:
        if self._dist is not None or self.config.mode != "head_only" or not self.config.cache_frozen_features:
            return None
        if slot is None:
            if dataset is self.train_dataset:
                slot = "train"
            elif dataset is self.val_dataset:
                slot = "val"
            elif dataset is self.replay_dataset:
                slot = "replay"
            else:
                return None

        if slot == "train":
            dual = self.config.sign_flip
            if self._train_head_cache is not None:
                return self._train_head_cache
        elif slot == "val":
            dual = False
            if self._val_head_cache is not None:
                return self._val_head_cache
        else:
            dual = self._replay_sign_flip
            if self._replay_head_cache is not None:
                return self._replay_head_cache

        estimated = self._estimate_head_cache_bytes(dataset) * (2 if dual else 1)
        if estimated > _MAX_HEAD_CACHE_BYTES:
            logger.info(
                "skipping frozen-feature cache for %d windows (estimated %.1f MiB exceeds %.0f MiB cap)",
                len(dataset),
                estimated / (1024 * 1024),
                _MAX_HEAD_CACHE_BYTES / (1024 * 1024),
            )
            return None
        cache = self._build_head_cache(dataset, negate=False)
        if dual:
            cache.negated = self._build_head_cache(dataset, negate=True)
        if slot == "train":
            self._train_head_cache = cache
        elif slot == "val":
            self._val_head_cache = cache
        else:
            self._replay_head_cache = cache
        return cache

    @torch.no_grad()
    def _build_head_cache(self, dataset: TimeSeriesDataset, *, negate: bool = False) -> _CachedHeadDataset:
        model_config = self.model.config
        patch = model_config.patch_size
        n_targets = dataset.series[0].layout[0]
        horizon_patches = -(-dataset.forecast_horizon // patch)
        frame = self.config.target_frame
        non_blocking = self.device.type == "cuda"

        hiddens: list[Tensor] = []
        locs: list[Tensor] = []
        scales: list[Tensor] = []
        targets_list: list[Tensor] = []
        weights_list: list[Tensor] = []
        cnts: list[Tensor] = []
        means: list[Tensor] = []
        m2s: list[Tensor] = []
        norm_list: list[Tensor] = []
        floc_list: list[Tensor] = []
        fscale_list: list[Tensor] = []

        was_training = self.model.training
        self.model.eval()
        seen_series = 0
        total_series = len(dataset)
        try:
            for batch in self._raw_loader(dataset, training=False):
                if negate:
                    batch = flip_batch_signs(batch, p=1.0)
                batch.validate(patch)
                on_device = batch.to(self.device, non_blocking=non_blocking)
                with self._autocast():
                    output = training_forward(self.model, on_device, return_hidden=True, validate=False)
                assert output.hidden is not None
                target_rows_cpu = batch.target_rows
                target_rows_dev = on_device.target_rows
                series_count = int(target_rows_cpu.sum()) // n_targets
                real_series = min(series_count, total_series - seen_series)
                seen_series += real_series
                if real_series == 0:
                    continue

                horizon_hidden = output.hidden[target_rows_dev, output.num_context_tokens :, :].detach().cpu()
                hiddens.append(horizon_hidden.unflatten(0, (series_count, n_targets))[:real_series])
                locs.append(
                    output.anchor_loc[target_rows_dev]
                    .detach()
                    .cpu()
                    .unflatten(0, (series_count, n_targets))[:real_series]
                )
                scales.append(
                    output.anchor_scale[target_rows_dev]
                    .detach()
                    .cpu()
                    .unflatten(0, (series_count, n_targets))[:real_series]
                )
                targets_list.append(
                    batch.targets[target_rows_cpu].unflatten(0, (series_count, n_targets))[:real_series]
                )
                weights_list.append(
                    batch.loss_weight[target_rows_cpu].unflatten(0, (series_count, n_targets))[:real_series]
                )

                if frame == "true_values":
                    f_loc, f_scale = target_frames(self.model, on_device, output, "true_values")
                    normalised = normalise_targets(self.model, on_device, f_loc, f_scale)[target_rows_dev].cpu()
                    norm_list.append(normalised.unflatten(0, (series_count, n_targets))[:real_series])
                    floc_list.append(f_loc[target_rows_dev].cpu().unflatten(0, (series_count, n_targets))[:real_series])
                    fscale_list.append(
                        f_scale[target_rows_dev].cpu().unflatten(0, (series_count, n_targets))[:real_series]
                    )
                elif horizon_patches > 1:
                    cnt, mean, m2 = context_welford_state(
                        batch.context[target_rows_cpu], batch.context_mask[target_rows_cpu], patch
                    )
                    cnts.append(cnt.unflatten(0, (series_count, n_targets))[:real_series])
                    means.append(mean.unflatten(0, (series_count, n_targets))[:real_series])
                    m2s.append(m2.unflatten(0, (series_count, n_targets))[:real_series])
        finally:
            self.model.train(was_training)

        welford = (torch.cat(cnts), torch.cat(means), torch.cat(m2s)) if cnts else None
        normalised_all = torch.cat(norm_list) if norm_list else None
        floc_all = torch.cat(floc_list) if floc_list else None
        fscale_all = torch.cat(fscale_list) if fscale_list else None
        return _CachedHeadDataset(
            horizon_hidden=torch.cat(hiddens),
            anchor_loc=torch.cat(locs),
            anchor_scale=torch.cat(scales),
            targets=torch.cat(targets_list),
            loss_weight=torch.cat(weights_list),
            context_welford=welford,
            normalised_targets=normalised_all,
            frame_loc=floc_all,
            frame_scale=fscale_all,
        )

    def _loader(self, dataset: TimeSeriesDataset, *, training: bool) -> DataLoader:
        slot: Literal["train", "val", "replay"] | None = (
            "train" if dataset is self.train_dataset else ("val" if dataset is self.val_dataset else None)
        )
        cache = self._get_head_cache(dataset, slot=slot)
        if cache is None:
            return self._raw_loader(dataset, training=training)
        config = self.config
        generator = torch.Generator().manual_seed(config.seed) if training else None
        batch_size = config.batch_size if training else (config.eval_batch_size or config.batch_size)
        flip_gen = self._fit_flip_generator if training else None
        collate_fn = (
            (lambda idx: cache.collate_sign_flipped(idx, flip_gen))
            if (flip_gen is not None and cache.negated is not None)
            else cache.collate
        )
        return DataLoader(
            cache,
            batch_size=batch_size,
            shuffle=training and config.shuffle,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )

    def _replay_loader(self) -> DataLoader:
        assert self.replay_config is not None and self.replay_dataset is not None
        dataset = self.replay_dataset
        config = self.config
        replay_batch_size = self.replay_config.resolved_batch_size(config.batch_size)
        shuffle_gen = torch.Generator().manual_seed(config.seed + 2_000_009)

        cache = self._get_head_cache(dataset, slot="replay")
        if cache is not None:
            flip_gen = self._replay_flip_generator
            collate_fn = (
                (lambda idx: cache.collate_sign_flipped(idx, flip_gen))
                if (flip_gen is not None and cache.negated is not None)
                else cache.collate
            )
            return DataLoader(
                cache,
                batch_size=replay_batch_size,
                shuffle=config.shuffle,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=self.device.type == "cuda",
                generator=shuffle_gen,
            )

        bucket_max = replay_batch_size if (config.compile_model and config.compile_mode == "reduce-overhead") else None
        collator = DHFDataCollator.for_model(self.model.config, dataset.context_length, bucket_to_max=bucket_max)

        def collate_indexed(items: Sequence[tuple[int, Window]]) -> _ReplayBatchItem:
            indices = torch.as_tensor([idx for idx, _ in items], dtype=torch.long)
            windows = [window for _, window in items]
            return _ReplayBatchItem(batch=collator(windows), indices=indices, real_series=len(windows))

        return DataLoader(
            _IndexedReplayDataset(dataset),
            batch_size=replay_batch_size,
            shuffle=config.shuffle,
            collate_fn=collate_indexed,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            generator=shuffle_gen,
        )

    def _compute_replay_terms_with_counts(
        self,
        replay_batch: TrainingBatch | CachedHeadBatch,
        *,
        real_series: int | None,
        teacher_quantiles: Tensor | None,
        validate: bool,
        replay_scored: int | None = None,
    ) -> tuple[Tensor | None, int, Tensor | None, int]:
        """Compute `(replay_pinball, replay_count, distill_loss, distill_count)` on one device replay batch."""
        assert self.replay_config is not None and self.replay_dataset is not None
        want_pinball = self.replay_config.replay_fraction > 0.0
        w_r = (replay_batch.num_scored if replay_scored is None else replay_scored) if want_pinball else 0
        need_pinball = want_pinball and w_r > 0
        need_distill = self.replay_config.distill_weight > 0.0
        if not need_pinball and not need_distill:
            return None, 0, None, 0

        need_hidden = need_distill and isinstance(replay_batch, TrainingBatch) and self._frozen_head is not None
        if need_pinball:
            out = compute_loss(
                self.model,
                replay_batch,
                self.config.target_frame,
                return_hidden=need_hidden,
                validate=validate,
            )
            replay_pinball: Tensor | None = out.loss
            student_quantiles = out.forward.quantiles
            student_hidden = out.forward.hidden
            num_context_tokens = out.forward.num_context_tokens
        else:
            replay_pinball = None
            w_r = 0
            if isinstance(replay_batch, CachedHeadBatch):
                student_quantiles = self.model.head(replay_batch.horizon_hidden)
                student_hidden = None
                num_context_tokens = 0
            else:
                fwd = training_forward(self.model, replay_batch, return_hidden=need_hidden, validate=validate)
                student_quantiles = fwd.quantiles
                student_hidden = fwd.hidden
                num_context_tokens = fwd.num_context_tokens

        if not need_distill:
            return replay_pinball, w_r, None, 0

        horizon_steps = student_quantiles.shape[1]
        requested_horizon = min(self.replay_dataset.forecast_horizon, horizon_steps)
        step_mask = torch.arange(horizon_steps, device=student_quantiles.device) < requested_horizon

        if isinstance(replay_batch, CachedHeadBatch):
            assert self._frozen_head is not None
            with torch.no_grad():
                teacher_q = self._frozen_head(replay_batch.horizon_hidden)
            student_q = student_quantiles
            distill_mask = step_mask.unsqueeze(0).expand(student_q.shape[0], horizon_steps)
            distill_count = int(student_q.shape[0]) * requested_horizon
            return (
                replay_pinball,
                w_r,
                distillation_loss(student_q, teacher_q, distill_mask, smooth_beta=self.replay_config.smooth_beta),
                distill_count,
            )

        rows_per_series = sum(self.replay_dataset.series[0].layout)
        real_row_limit = replay_batch.num_rows if real_series is None else real_series * rows_per_series
        real_target_mask = replay_batch.target_rows & (
            torch.arange(replay_batch.num_rows, device=student_quantiles.device) < real_row_limit
        )
        student_q = student_quantiles[real_target_mask]

        if teacher_quantiles is not None:
            teacher_q = teacher_quantiles
        elif self._frozen_head is not None:
            assert student_hidden is not None
            with torch.no_grad():
                teacher_q = self._frozen_head(student_hidden[real_target_mask, num_context_tokens:, :])
        elif self._frozen_base is not None:
            with torch.no_grad():
                teacher_q = training_forward(self._frozen_base, replay_batch, validate=False).quantiles[
                    real_target_mask
                ]
        else:
            # Zero-VRAM frozen-base forward for LoRA modes when teacher caching is disabled/exceeded.
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad(), disable_lora(self.model):
                    teacher_q = training_forward(self.model, replay_batch, validate=False).quantiles[real_target_mask]
            finally:
                self.model.train(was_training)

        distill_mask = step_mask.unsqueeze(0).expand(student_q.shape[0], horizon_steps)
        distill_count = int(student_q.shape[0]) * requested_horizon
        return (
            replay_pinball,
            w_r,
            distillation_loss(student_q, teacher_q, distill_mask, smooth_beta=self.replay_config.smooth_beta),
            distill_count,
        )

    def _compute_replay_terms(
        self,
        replay_batch: TrainingBatch | CachedHeadBatch,
        *,
        real_series: int | None,
        teacher_quantiles: Tensor | None,
        validate: bool,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Compute `(replay_pinball_loss, distill_loss)` on one replay micro-batch already on device."""
        replay_pinball, _, distill_term, _ = self._compute_replay_terms_with_counts(
            replay_batch,
            real_series=real_series,
            teacher_quantiles=teacher_quantiles,
            validate=validate,
        )
        return replay_pinball, distill_term

    # ------------------------------------------------------------------ loop

    def _prepare_domain_batch(
        self,
        batch: TrainingBatch | CachedHeadBatch,
        flip_generator: torch.Generator | None,
    ) -> _PreparedDomainBatch | None:
        """Sign-flip, validate on CPU, and transfer a scored domain batch to `self.device`.

        Returns None when `batch` has no scored position, leaving `flip_generator` untouched and
        skipping validation and device transfer.
        """
        scored = batch.num_scored
        if scored == 0:
            return None
        if isinstance(batch, TrainingBatch):
            if flip_generator is not None:
                batch = flip_batch_signs(batch, flip_generator)
            batch.validate(self.model.config.patch_size)
        return _PreparedDomainBatch(
            batch=batch.to(self.device, non_blocking=self.device.type == "cuda"),
            num_scored=scored,
        )

    def _prepare_replay_batch(
        self,
        replay_item: _ReplayBatchItem | CachedHeadBatch,
        flip_generator: torch.Generator | None,
    ) -> _PreparedReplayBatch:
        """Sign-flip, validate on CPU, resolve cached teacher quantiles, and transfer a replay batch."""
        non_blocking = self.device.type == "cuda"
        if isinstance(replay_item, CachedHeadBatch):
            return _PreparedReplayBatch(
                batch=replay_item.to(self.device, non_blocking=non_blocking),
                num_scored=replay_item.num_scored,
            )
        assert isinstance(replay_item, _ReplayBatchItem)
        assert self.replay_dataset is not None
        r_batch = replay_item.batch
        real_series = replay_item.real_series
        flip_mask: Tensor | None = None
        if flip_generator is not None:
            rows_per_series = sum(self.replay_dataset.series[0].layout)
            flip_mask = torch.rand((real_series,), generator=flip_generator) < 0.5
            r_batch = _apply_series_flip(r_batch, flip_mask, rows_per_series)
        r_batch.validate(self.model.config.patch_size)
        scored = r_batch.num_scored
        replay_dev = r_batch.to(self.device, non_blocking=non_blocking)
        teacher_dev: Tensor | None = None
        if self._replay_base_pos is not None:
            pos_q = self._replay_base_pos[replay_item.indices]
            if flip_mask is not None and self._replay_base_neg is not None:
                neg_q = self._replay_base_neg[replay_item.indices]
                sel = flip_mask.view(-1, 1, 1, 1)
                teacher_cpu = torch.where(sel, neg_q, pos_q).flatten(0, 1)
            else:
                teacher_cpu = pos_q.flatten(0, 1)
            teacher_dev = teacher_cpu.to(self.device, non_blocking=non_blocking)
        return _PreparedReplayBatch(
            batch=replay_dev,
            real_series=real_series,
            teacher_quantiles=teacher_dev,
            num_scored=scored,
        )

    def _mark_cudagraph_step_begin(self) -> None:
        """Notify PyTorch CUDA graphs that a new compiled step is starting, when applicable."""
        if (
            self.config.compile_model
            and self.config.compile_mode == "reduce-overhead"
            and self.device.type == "cuda"
            and hasattr(torch, "compiler")
            and hasattr(torch.compiler, "cudagraph_mark_step_begin")
        ):
            torch.compiler.cudagraph_mark_step_begin()

    def _compute_loss_terms(
        self,
        batch: TrainingBatch | CachedHeadBatch | None,
        *,
        domain_count: int | None = None,
        validate: bool = True,
        replay: _PreparedReplayBatch | None = None,
    ) -> _LossTerms:
        """Compute unmixed per-term losses and active counts under autocast on prepared device batches.

        When `batch` is None or `domain_count == 0`, the domain forward is skipped (`domain=None`,
        `domain_count=0`). Replay terms still run when `replay` and `self.replay_config` are set.
        """
        with self._autocast():
            domain_loss: Tensor | None = None
            w_d: int | None = 0
            if batch is not None and domain_count != 0:
                domain_loss = compute_loss(self.model, batch, self.config.target_frame, validate=validate).loss
                w_d = domain_count

            replay_loss: Tensor | None = None
            w_r = 0
            distill_term: Tensor | None = None
            m_count = 0
            if replay is not None and self.replay_config is not None:
                replay_loss, w_r, distill_term, m_count = self._compute_replay_terms_with_counts(
                    replay.batch,
                    real_series=replay.real_series,
                    teacher_quantiles=replay.teacher_quantiles,
                    validate=validate,
                    replay_scored=replay.num_scored,
                )
        return _LossTerms(
            domain=domain_loss,
            domain_count=w_d,
            replay=replay_loss,
            replay_count=w_r,
            distill=distill_term,
            distill_count=m_count,
        )

    def _mix_loss_terms(self, terms: _LossTerms) -> Tensor:
        """Combine unmixed terms in exact one-process order: domain/replay mix, then distillation."""
        if terms.domain is None:
            raise ValueError("cannot mix loss terms without a domain loss")
        loss = terms.domain
        if self.replay_config is not None:
            if terms.replay is not None:
                frac = self.replay_config.replay_fraction
                loss = (1.0 - frac) * terms.domain + frac * terms.replay
            if terms.distill is not None:
                loss = loss + self.replay_config.distill_weight * terms.distill
        return loss

    def _apply_optimizer_step(self) -> None:
        """Clip trainable gradients (if configured) and advance the optimiser and LR scheduler."""
        if self.config.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self._trainable, self.config.gradient_clip_val)
        self.optimizer.step()
        self.scheduler.step()

    def training_step(
        self,
        batch: TrainingBatch | CachedHeadBatch,
        *,
        validate: bool = True,
        replay_batch: TrainingBatch | CachedHeadBatch | None = None,
        replay_real_series: int | None = None,
        replay_teacher_quantiles: Tensor | None = None,
    ) -> float:
        """One optimiser step on `batch` (already on the device), with optional replay micro-batch."""
        if self._dist is not None:
            raise RuntimeError(
                "training_step() is single-process only and cannot be called on a multi-process trainer; "
                "call fit() instead to run synchronised multi-process training"
            )
        self._mark_cudagraph_step_begin()
        self.optimizer.zero_grad(set_to_none=True)
        replay = (
            None
            if replay_batch is None
            else _PreparedReplayBatch(
                batch=replay_batch,
                real_series=replay_real_series,
                teacher_quantiles=replay_teacher_quantiles,
            )
        )
        terms = self._compute_loss_terms(batch, validate=validate, replay=replay)
        loss = self._mix_loss_terms(terms)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"the training loss became {loss.item()} at step {self.global_step + 1}. Lower the learning rate, "
                "or check the data for extreme values"
            )
        loss.backward()
        self._apply_optimizer_step()
        return loss.item()

    def _make_fit_strategy(self) -> _FitStrategy:
        """Select the execution strategy for `fit()`."""
        if self._dist is not None:
            from ._distributed_strategy import _DistributedStrategy

            return _DistributedStrategy(self, self._dist)
        return _SingleProcessStrategy(self)

    def _run_fit_loop(self, strategy: _FitStrategy) -> TrainingResult:
        """Run the shared epoch, step, and evaluation loop skeleton driven by `strategy`."""
        config = self.config
        self.scheduler = self._new_schedule()
        self.should_stop = False
        start_step = self.fit_start_step = self.global_step
        final_step = start_step + self.total_steps
        bookkeeper = _FitBookkeeper(
            start_step=start_step,
            final_step=final_step,
            track_best=strategy.is_primary,
            restore_best=config.restore_best and strategy.is_primary,
        )

        def evaluate_now() -> None:
            result = strategy.evaluate_val()
            strategy.on_evaluate(self.global_step, result, bookkeeper)

        try:
            try:
                strategy.prepare_fit()
                strategy.on_fit_start()
                self.model.train()
                while self.global_step < final_step and not self.should_stop:
                    bookkeeper.epochs += 1
                    steps_before_epoch = self.global_step
                    skipped_before_epoch = bookkeeper.skipped_batches
                    for outcome in strategy.iter_epoch_steps(bookkeeper.epochs - 1):
                        if outcome.skipped:
                            bookkeeper.skipped_batches += 1
                            continue
                        self.global_step += 1
                        loss = strategy.on_step_end(self.global_step, outcome.loss, outcome.learning_rate)
                        bookkeeper.record_step(self.global_step, loss)
                        due = config.eval_every is not None and (self.global_step - start_step) % config.eval_every == 0
                        if self.val_dataset is not None and due:
                            evaluate_now()
                        if self.global_step >= final_step or self.should_stop:
                            break
                    if self.global_step == steps_before_epoch:
                        epoch_skipped = bookkeeper.skipped_batches - skipped_before_epoch
                        raise RuntimeError(
                            f"an epoch produced no optimiser step: all {epoch_skipped} batches had no scored position"
                        )
                    if self.val_dataset is not None and config.eval_every is None and not self.should_stop:
                        evaluate_now()
                if self.val_dataset is not None and bookkeeper.last_evaluated_step != self.global_step:
                    evaluate_now()
            finally:
                self.model.eval()
                self._fit_flip_generator = None
                self._replay_flip_generator = None

            restored = strategy.finalize_best(bookkeeper, self.global_step)
            if bookkeeper.skipped_batches and strategy.is_primary:
                logger.warning("skipped %d training batches that had no scored position", bookkeeper.skipped_batches)

            result = bookkeeper.build_result(self.global_step, restored_best=restored)
            strategy.on_fit_end(result)
            return result
        finally:
            strategy.cleanup()

    def fit(self) -> TrainingResult:
        """Train for the configured number of steps and return a summary.

        With a validation set, evaluation runs every `eval_every` steps (or at the end of each
        epoch) and once more at the end if the last step was not evaluated.

        Calling `fit` again continues from the current weights and optimiser state with a fresh
        learning-rate schedule and the same batch order. It seeds PyTorch's global generator.
        """
        if self._dist is not None:
            self._dist.check_active()
        return self._run_fit_loop(self._make_fit_strategy())

    @torch.no_grad()
    def evaluate(self, dataset: TimeSeriesDataset | None = None) -> EvalResult:
        """The loss over `dataset` (default: the validation set), in the configured target frame."""
        dataset = self.val_dataset if dataset is None else dataset
        if dataset is None:
            raise ValueError("no dataset to evaluate: pass one, or give the trainer a val_dataset")
        if not isinstance(dataset, TimeSeriesDataset):
            raise TypeError(f"dataset must be a TimeSeriesDataset, got {type(dataset).__name__}")
        self._check_dataset(dataset, "dataset")
        was_training = self.model.training
        self.model.eval()
        try:
            if dataset is self.val_dataset:
                if self._val_loader is None:
                    self._val_loader = self._loader(dataset, training=False)
                loader = self._val_loader
            else:
                loader = self._loader(dataset, training=False)
            return self._evaluate(loader)
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def _accumulate_eval_totals(self, batches: Iterable[TrainingBatch | CachedHeadBatch]) -> tuple[float, int, int]:
        """Accumulate `(loss_sum, scored_positions, batches)` over `batches` in float64."""
        patch = self.model.config.patch_size
        non_blocking = self.device.type == "cuda"
        total = 0.0
        scored = count = 0
        for batch in batches:
            count += 1
            positions = batch.num_scored
            if positions == 0:
                continue
            if isinstance(batch, TrainingBatch):
                batch.validate(patch)
            with self._autocast():
                loss = compute_loss(
                    self.model,
                    batch.to(self.device, non_blocking=non_blocking),
                    self.config.target_frame,
                    validate=False,
                ).loss
            # Weight each batch by its scored positions, so the result is the mean over the dataset.
            total += loss.item() * positions  # .item() is a Python float, so the sum accumulates in float64
            scored += positions
        return total, scored, count

    def _evaluate(self, batches: Iterable[TrainingBatch | CachedHeadBatch]) -> EvalResult:
        total, scored, count = self._accumulate_eval_totals(batches)
        if scored == 0:
            raise ValueError("the dataset has no scored position to evaluate")
        return EvalResult(loss=total / scored, scored_positions=scored, batches=count)

    def export(
        self,
        output_dir: str | Path | None = None,
        *,
        calibrator: ConformalCalibrator | OnlineConformalCalibrator | None = None,
        metrics: Mapping[str, Any] | None = None,
        data_fingerprint: str | None = None,
        license_path: str | Path | None = None,
    ) -> Path:
        """Write an upstream-loadable checkpoint directory (spec §7.4, acceptance test T11).

        Args:
            output_dir: Destination directory. Defaults to `config.output_dir`.
            calibrator: Optional fitted `ConformalCalibrator` or `OnlineConformalCalibrator` to
                write as `conformal.json`.
            metrics: Optional evaluation metrics recorded in `tune_metadata.json`.
            data_fingerprint: Optional dataset fingerprint; defaults to a SHA-256 structural
                summary of `train_dataset`.
            license_path: Optional explicit path to the base weights' `LICENSE` file.
        """
        target = output_dir if output_dir is not None else self.config.output_dir
        if target is None:
            raise ValueError("no output directory: pass output_dir to export() or set TrainingConfig.output_dir")
        world_size = None if self._dist is None else self._dist.world_size
        effective_batch_size = None if self._dist is None else self._dist.world_size * self.config.batch_size
        if self._dist is not None and not self._dist.is_rank_zero:
            # Skip dataset_fingerprint() hashing on non-zero ranks; validate only caller-supplied inputs.
            _validate_export_inputs(
                self.model,
                lora_config=self.lora_config,
                calibrator=calibrator,
                metrics=metrics,
                license_path=license_path,
                data_fingerprint=data_fingerprint,
                world_size=world_size,
                effective_batch_size=effective_batch_size,
            )
            logger.info(
                "skipping checkpoint write on rank %d/%d; rank 0 writes %s",
                self._dist.rank,
                self._dist.world_size,
                Path(target),
            )
            return Path(target)

        fingerprint = data_fingerprint if data_fingerprint is not None else dataset_fingerprint(self.train_dataset)
        replay_source: dict[str, Any] | None = None
        if self.replay_dataset is not None:
            replay_source = {
                "type": type(self.replay_dataset).__name__,
                "num_series": len(self.replay_dataset.series),
                "num_windows": len(self.replay_dataset),
                "context_length": self.replay_dataset.context_length,
                "forecast_horizon": self.replay_dataset.forecast_horizon,
                "data_fingerprint": dataset_fingerprint(self.replay_dataset),
            }
            if isinstance(self.replay_dataset, SyntheticReplaySource):
                replay_source["seed"] = self.replay_dataset.seed
                replay_source["families"] = list(self.replay_dataset.families)
                replay_source["missing_rate"] = self.replay_dataset.missing_rate

        return export_checkpoint(
            self.model,
            target,
            base_repo_or_path=self.config.model_name_or_path,
            base_revision=self.config.revision,
            mode=self.config.mode,
            training_config=self.config,
            lora_config=self.lora_config,
            replay_config=self.replay_config,
            replay_source=replay_source,
            calibrator=calibrator,
            metrics=metrics,
            data_fingerprint=fingerprint,
            license_path=license_path,
            model_provided_directly=self._model_provided_directly,
            world_size=world_size,
            effective_batch_size=effective_batch_size,
        )
