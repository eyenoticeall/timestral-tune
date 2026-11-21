"""Configuration dataclasses.

`TrainingConfig` holds everything the trainer needs apart from the data. It accepts every
adaptation mode and precision the specification names, so that configurations written today stay
valid; the trainer raises a clear `NotImplementedError` for the ones that are not built yet.

`LoraConfig` (roadmap Phase 2) and `ReplayConfig` (Phase 4) are not defined yet.

- Spec: `docs/SPEC.md` §5 (modes), §5.4 (trainer), §7.1 (API example)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from .models.forward import TARGET_FRAMES, TargetFrame

__all__ = ["DEFAULT_LEARNING_RATES", "AdaptationMode", "TrainingConfig"]

AdaptationMode = Literal["head_only", "lora_variate", "lora_temporal", "full_ft"]
MixedPrecision = Literal["fp32", "bf16"]
LearningRateSchedule = Literal["cosine", "constant"]

# Starting points only. The paper publishes no optimiser settings; these are placeholders until the
# Phase 3 sweeps (spec §5.4, §12).
DEFAULT_LEARNING_RATES: dict[str, float] = {
    "head_only": 1e-4,
    "lora_variate": 3e-4,
    "lora_temporal": 3e-4,
    "full_ft": 1e-5,
}


@dataclass(frozen=True)
class TrainingConfig:
    """Settings for one training run.

    Attributes:
        model_name_or_path: Hub repo id (e.g. `mistralai/Timestral-28M`) or a local checkpoint
            directory.
        revision: Hub revision. Official repos default to the revision this library was verified
            against.
        mode: Which parameters are trained (spec §5).
        learning_rate: Peak learning rate. None picks the default for `mode`.
        weight_decay: AdamW weight decay.
        batch_size: Series windows per training batch.
        epochs: Passes over the training set. Ignored when `max_steps` is set.
        max_steps: Total optimiser steps, overriding `epochs`.
        warmup_steps: Steps of linear warm-up from 0 to the peak learning rate.
        lr_schedule: `"cosine"` decays to 0 over the remaining steps; `"constant"` holds the peak.
        gradient_clip_val: Maximum gradient norm, or None to disable clipping.
        mixed_precision: `"fp32"`, or `"bf16"` autocast (CUDA only).
        target_frame: Frame the loss targets are normalised in (spec §4.3).
        device: `"auto"`, `"cpu"`, `"cuda"` or `"mps"`.
        seed: Seeds weight-independent randomness: batch order and any sampling.
        num_workers: DataLoader worker processes. On macOS with PyTorch 2.4.0, any worker leaves
            the process hanging at exit (a PyTorch issue; `docs/VERIFICATION.md` §10).
        eval_every: Evaluate on the validation set every this many steps. None evaluates at the
            end of each epoch.
        eval_batch_size: Batch size for evaluation. None uses `batch_size`.
        early_stopping_patience: Stop after this many evaluations without improvement. None
            disables early stopping.
        restore_best: After training with a validation set, put back the trainable parameters
            from the evaluation with the lowest loss.
        log_every: Log training progress every this many steps.
        shuffle: Shuffle the training windows each epoch.
        output_dir: Where `export()` will write (roadmap Phase 2). Nothing is written before then.
    """

    model_name_or_path: str | Path
    revision: str | None = None
    mode: AdaptationMode = "head_only"
    learning_rate: float | None = None
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 1
    max_steps: int | None = None
    warmup_steps: int = 0
    lr_schedule: LearningRateSchedule = "cosine"
    gradient_clip_val: float | None = 1.0
    mixed_precision: MixedPrecision = "fp32"
    target_frame: TargetFrame = "median_folded"
    device: str = "auto"
    seed: int = 0
    num_workers: int = 0
    eval_every: int | None = None
    eval_batch_size: int | None = None
    early_stopping_patience: int | None = None
    restore_best: bool = True
    log_every: int = 50
    shuffle: bool = True
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        _one_of("mode", self.mode, get_args(AdaptationMode))
        _one_of("mixed_precision", self.mixed_precision, get_args(MixedPrecision))
        _one_of("lr_schedule", self.lr_schedule, get_args(LearningRateSchedule))
        _one_of("target_frame", self.target_frame, TARGET_FRAMES)
        if self.learning_rate is not None and not (math.isfinite(self.learning_rate) and self.learning_rate > 0):
            raise ValueError(f"learning_rate must be positive and finite, got {self.learning_rate}")
        if not (math.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise ValueError(f"weight_decay must be non-negative and finite, got {self.weight_decay}")
        if self.gradient_clip_val is not None and not (
            math.isfinite(self.gradient_clip_val) and self.gradient_clip_val > 0
        ):
            raise ValueError(f"gradient_clip_val must be positive and finite or None, got {self.gradient_clip_val}")
        for name in ("batch_size", "epochs", "log_every"):
            _positive_int(name, getattr(self, name))
        for name in ("max_steps", "eval_every", "eval_batch_size", "early_stopping_patience"):
            if getattr(self, name) is not None:
                _positive_int(name, getattr(self, name))
        for name in ("warmup_steps", "num_workers", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")

    @property
    def resolved_learning_rate(self) -> float:
        """`learning_rate`, or the default starting point for `mode`."""
        return DEFAULT_LEARNING_RATES[self.mode] if self.learning_rate is None else self.learning_rate


def _one_of(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
