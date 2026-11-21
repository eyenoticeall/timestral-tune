"""The training loop.

`TimestralTrainer` loads a checkpoint, freezes everything the chosen adaptation mode does not
train (`head_only`, `lora_variate`, `lora_temporal`, `full_ft`), and minimises the pinball loss
of spec §4 over the windows of a `TimeSeriesDataset`.

- Spec: `docs/SPEC.md` §4, §5, §5.1, §5.4, §7.1
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..config import LoraConfig, TrainingConfig
from ..data.batch import TrainingBatch
from ..data.dataset import TimeSeriesDataset
from ..models.adapter import inject_lora, lora_parameters
from ..models.device import TuneTimestralModel, supports_float64
from ..models.forward import context_welford_state, normalise_targets, target_frames, training_forward
from ..models.loader import load_model, resolve_device
from .callbacks import Callback, EarlyStopping, ProgressLogger
from .step import CachedHeadBatch, compute_loss

__all__ = ["EvalResult", "TimestralTrainer", "TrainingResult"]

logger = logging.getLogger("timestral_tune.trainer")

# Cap per-dataset CPU memory for cached `head_only` representations (256 MiB ~ 45k windows on 28M).
_MAX_HEAD_CACHE_BYTES = 256 * 1024 * 1024


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
        callbacks: Extra `Callback` instances, run after the built-in progress logger and early stopping.
        model: An already loaded model to train, instead of loading `config.model_name_or_path`.
            It is moved to the configured device and modified in place.
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_dataset: TimeSeriesDataset,
        val_dataset: TimeSeriesDataset | None = None,
        *,
        lora_config: LoraConfig | None = None,
        callbacks: Sequence[Callback] = (),
        model: TuneTimestralModel | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        if config.mixed_precision == "bf16":
            if self.device.type != "cuda":
                raise ValueError(
                    f"mixed_precision='bf16' is supported on CUDA only (MPS runs fp32 only); got {self.device.type!r}"
                )
            if not torch.cuda.is_bf16_supported():
                raise ValueError("mixed_precision='bf16' requested, but this CUDA device does not support bfloat16")

        if lora_config is not None and not isinstance(lora_config, LoraConfig):
            raise TypeError(f"lora_config must be a LoraConfig or None, got {type(lora_config).__name__}")
        self.lora_config = self._resolve_lora_config(config.mode, lora_config)

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

        if config.compile_model and config.compile_mode == "reduce-overhead":
            self._check_fixed_batch_shapes(train_dataset, config.batch_size, "train_dataset")
            if val_dataset is not None:
                eval_bs = config.eval_batch_size or config.batch_size
                self._check_fixed_batch_shapes(val_dataset, eval_bs, "val_dataset")

        self._trainable = self._select_trainable()
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
        steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
        self.total_steps = config.max_steps if config.max_steps is not None else config.epochs * steps_per_epoch
        self.scheduler = self._new_schedule()

        self.callbacks: list[Callback] = [ProgressLogger(config.log_every)]
        if config.early_stopping_patience is not None:
            if val_dataset is None:
                raise ValueError("early_stopping_patience needs a val_dataset")
            self.callbacks.append(EarlyStopping(config.early_stopping_patience))
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise TypeError(f"callbacks must be Callback instances, got {type(callback).__name__}")
        self.callbacks.extend(callbacks)

        self.global_step = 0
        self.fit_start_step = 0
        self.should_stop = False
        self._head_cache: dict[int, _CachedHeadDataset] = {}

    # ----------------------------------------------------------------- setup

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

    @staticmethod
    def _check_fixed_batch_shapes(dataset: TimeSeriesDataset, batch_size: int, name: str) -> None:
        if len(dataset) > batch_size and len(dataset) % batch_size != 0:
            raise ValueError(
                f"compile_mode='reduce-overhead' requires fixed batch shapes, but {name} has {len(dataset)} windows "
                f"which is not divisible by batch_size={batch_size}; use compile_mode='default' or bucket batches"
            )

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
        torch.manual_seed(self.config.seed)
        inject_lora(self.model, self.lora_config, rows_per_series=rows_per_series)
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

    def _raw_loader(self, dataset: TimeSeriesDataset, *, training: bool) -> DataLoader:
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

    def _get_head_cache(self, dataset: TimeSeriesDataset) -> _CachedHeadDataset | None:
        if self.config.mode != "head_only" or not self.config.cache_frozen_features:
            return None
        if dataset is not self.train_dataset and dataset is not self.val_dataset:
            return None
        key = id(dataset)
        if key in self._head_cache:
            return self._head_cache[key]
        estimated = self._estimate_head_cache_bytes(dataset)
        if estimated > _MAX_HEAD_CACHE_BYTES:
            logger.info(
                "skipping frozen-feature cache for %d windows (estimated %.1f MiB exceeds %.0f MiB cap)",
                len(dataset),
                estimated / (1024 * 1024),
                _MAX_HEAD_CACHE_BYTES / (1024 * 1024),
            )
            return None
        cache = self._build_head_cache(dataset)
        self._head_cache[key] = cache
        return cache

    @torch.no_grad()
    def _build_head_cache(self, dataset: TimeSeriesDataset) -> _CachedHeadDataset:
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
        try:
            for batch in self._raw_loader(dataset, training=False):
                batch.validate(patch)
                on_device = batch.to(self.device, non_blocking=non_blocking)
                output = training_forward(self.model, on_device, return_hidden=True, validate=False)
                assert output.hidden is not None
                target_rows_cpu = batch.target_rows
                target_rows_dev = on_device.target_rows
                series_count = int(target_rows_cpu.sum()) // n_targets

                horizon_hidden = output.hidden[target_rows_dev, output.num_context_tokens :, :].detach().cpu()
                hiddens.append(horizon_hidden.unflatten(0, (series_count, n_targets)))
                locs.append(output.anchor_loc[target_rows_dev].detach().cpu().unflatten(0, (series_count, n_targets)))
                scales.append(
                    output.anchor_scale[target_rows_dev].detach().cpu().unflatten(0, (series_count, n_targets))
                )
                targets_list.append(batch.targets[target_rows_cpu].unflatten(0, (series_count, n_targets)))
                weights_list.append(batch.loss_weight[target_rows_cpu].unflatten(0, (series_count, n_targets)))

                if frame == "true_values":
                    f_loc, f_scale = target_frames(self.model, on_device, output, "true_values")
                    normalised = normalise_targets(self.model, on_device, f_loc, f_scale)[target_rows_dev].cpu()
                    norm_list.append(normalised.unflatten(0, (series_count, n_targets)))
                    floc_list.append(f_loc[target_rows_dev].cpu().unflatten(0, (series_count, n_targets)))
                    fscale_list.append(f_scale[target_rows_dev].cpu().unflatten(0, (series_count, n_targets)))
                elif horizon_patches > 1:
                    cnt, mean, m2 = context_welford_state(
                        batch.context[target_rows_cpu], batch.context_mask[target_rows_cpu], patch
                    )
                    cnts.append(cnt.unflatten(0, (series_count, n_targets)))
                    means.append(mean.unflatten(0, (series_count, n_targets)))
                    m2s.append(m2.unflatten(0, (series_count, n_targets)))
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
        cache = self._get_head_cache(dataset)
        if cache is None:
            return self._raw_loader(dataset, training=training)
        config = self.config
        generator = torch.Generator().manual_seed(config.seed) if training else None
        batch_size = config.batch_size if training else (config.eval_batch_size or config.batch_size)
        return DataLoader(
            cache,
            batch_size=batch_size,
            shuffle=training and config.shuffle,
            collate_fn=cache.collate,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )

    # ------------------------------------------------------------------ loop

    def training_step(self, batch: TrainingBatch | CachedHeadBatch, *, validate: bool = True) -> float:
        """One optimiser step on `batch` (already on the device). Returns the loss."""
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            loss = compute_loss(self.model, batch, self.config.target_frame, validate=validate).loss
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
        start_step = self.fit_start_step = self.global_step
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

        patch = self.model.config.patch_size
        non_blocking = self.device.type == "cuda"
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
                    if isinstance(batch, TrainingBatch):
                        batch.validate(patch)
                    learning_rate = self.optimizer.param_groups[0]["lr"]
                    loss = self.training_step(batch.to(self.device, non_blocking=non_blocking), validate=False)
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
            final_train_loss=train_history[-1][1] if train_history else None,
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

    def _evaluate(self, batches: Iterable[TrainingBatch | CachedHeadBatch]) -> EvalResult:
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
        if scored == 0:
            raise ValueError("the dataset has no scored position to evaluate")
        return EvalResult(loss=total / scored, scored_positions=scored, batches=count)

    def export(self, output_dir: str | Path | None = None) -> None:
        """Write an upstream-loadable checkpoint (spec §7.4). Not implemented yet: roadmap Phase 2."""
        raise NotImplementedError(
            "export() is not implemented yet (roadmap Phase 2). The trained model is available as trainer.model"
        )
