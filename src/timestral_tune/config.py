"""Configuration dataclasses.

`TrainingConfig` holds everything the trainer needs apart from the data, and `LoraConfig`
configures selective slice LoRA on attention projections (spec §5.1).

`ReplayConfig` (roadmap Phase 4) is not defined yet.

- Spec: `docs/SPEC.md` §5 (modes), §5.1 (LoRA), §5.4 (trainer), §7.1 (API example)
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from .models.forward import TARGET_FRAMES, TargetFrame

__all__ = [
    "DEFAULT_LEARNING_RATES",
    "AdaptationMode",
    "CompileMode",
    "LoraAxis",
    "LoraConfig",
    "LoraTarget",
    "TrainingConfig",
]

AdaptationMode = Literal["head_only", "lora_variate", "lora_temporal", "full_ft"]
MixedPrecision = Literal["fp32", "bf16"]
LearningRateSchedule = Literal["cosine", "constant"]
CompileMode = Literal["default", "reduce-overhead"]
LoraAxis = Literal["time", "variate"]
LoraTarget = Literal["q", "k", "v", "o", "gate"]

LORA_AXES: tuple[str, ...] = get_args(LoraAxis)
LORA_TARGETS: tuple[str, ...] = get_args(LoraTarget)
SINGLE_ROW_ALLOWED_TARGETS: tuple[str, ...] = ("v", "o", "gate")

# Starting points only. The paper publishes no optimiser settings; these are placeholders until the
# Phase 3 sweeps (spec §5.4, §12).
DEFAULT_LEARNING_RATES: dict[str, float] = {
    "head_only": 1e-4,
    "lora_variate": 3e-4,
    "lora_temporal": 3e-4,
    "full_ft": 1e-5,
}


@dataclass(frozen=True)
class LoraConfig:
    """Settings for selective slice LoRA on Timestral's attention modules (spec §5.1).

    Attributes:
        r: Low-rank bottleneck dimension.
        lora_alpha: Scaling numerator; each adapter adds `(lora_alpha / r) * B @ A`.
        axes: Attention axes to adapt, selected from `"time"` and `"variate"` via each attention
            module's `is_time` attribute.
        targets: Projections/slices to adapt, selected from `"q"`, `"k"`, `"v"` (row slices of
            the fused `wQKV`), `"o"` (`wO`) and `"gate"` (`_gate`).
        dropout: Dropout rate applied to the adapter input path during training (`0 <= dropout < 1`).
        allow_single_row: Override the single-row guard for `"variate"` attention. When True on
            single-row data, only `"v"`, `"o"` and `"gate"` are permitted (`"q"` and `"k"` are
            inert there and rejected).
    """

    r: int = 8
    lora_alpha: float = 16.0
    axes: tuple[LoraAxis, ...] = ("variate",)
    targets: tuple[LoraTarget, ...] = ("q", "v")
    dropout: float = 0.0
    allow_single_row: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.r, numbers.Integral) or isinstance(self.r, bool) or self.r < 1:
            raise ValueError(f"r must be a positive integer, got {self.r!r}")
        object.__setattr__(self, "r", int(self.r))

        if isinstance(self.lora_alpha, bool) or not (
            isinstance(self.lora_alpha, numbers.Real) and math.isfinite(self.lora_alpha) and self.lora_alpha > 0
        ):
            raise ValueError(f"lora_alpha must be positive and finite, got {self.lora_alpha!r}")
        object.__setattr__(self, "lora_alpha", float(self.lora_alpha))

        if isinstance(self.dropout, bool) or not (
            isinstance(self.dropout, numbers.Real) and math.isfinite(self.dropout) and 0.0 <= self.dropout < 1.0
        ):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout!r}")
        object.__setattr__(self, "dropout", float(self.dropout))

        if not isinstance(self.allow_single_row, bool):
            raise ValueError(f"allow_single_row must be a bool, got {self.allow_single_row!r}")

        object.__setattr__(self, "axes", _unique_choices("axes", self.axes, LORA_AXES))
        object.__setattr__(self, "targets", _unique_choices("targets", self.targets, LORA_TARGETS))

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.r

    def to_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "axes": list(self.axes),
            "targets": list(self.targets),
            "dropout": self.dropout,
            "allow_single_row": self.allow_single_row,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoraConfig:
        known = {"r", "lora_alpha", "axes", "targets", "dropout", "allow_single_row"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown LoraConfig keys: {sorted(unknown)}")
        return cls(**data)


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
        cache_frozen_features: In `head_only` mode, cache the frozen transformer's per-window
            horizon representations on CPU (up to 256 MiB per dataset) so multi-epoch training and
            repeated evaluations skip re-running the frozen body. Ignored in other modes, and falls
            back to streaming when the estimated cache exceeds the memory cap.
        compile_model: Compile the transformer with `torch.compile`.
        compile_mode: Compilation mode when `compile_model=True`. `"reduce-overhead"` uses CUDA
            graphs and automatically pads partial tail batches into power-of-two shape buckets with
            zero-weighted, group-isolated dummy series so no data is dropped.
        output_dir: Default directory for `trainer.export()`.
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
    cache_frozen_features: bool = True
    compile_model: bool = False
    compile_mode: CompileMode = "default"
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        _one_of("mode", self.mode, get_args(AdaptationMode))
        _one_of("mixed_precision", self.mixed_precision, get_args(MixedPrecision))
        _one_of("lr_schedule", self.lr_schedule, get_args(LearningRateSchedule))
        _one_of("target_frame", self.target_frame, TARGET_FRAMES)
        _one_of("compile_mode", self.compile_mode, get_args(CompileMode))
        for flag in ("cache_frozen_features", "compile_model", "restore_best", "shuffle"):
            if not isinstance(getattr(self, flag), bool):
                raise ValueError(f"{flag} must be a bool, got {getattr(self, flag)!r}")
        if self.learning_rate is not None and not (math.isfinite(self.learning_rate) and self.learning_rate > 0):
            raise ValueError(f"learning_rate must be positive and finite, got {self.learning_rate}")
        if not (math.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise ValueError(f"weight_decay must be non-negative and finite, got {self.weight_decay}")
        if self.gradient_clip_val is not None and not (
            math.isfinite(self.gradient_clip_val) and self.gradient_clip_val > 0
        ):
            raise ValueError(f"gradient_clip_val must be positive and finite or None, got {self.gradient_clip_val}")
        for name in ("batch_size", "epochs", "log_every"):
            self._set_integer(name, minimum=1)
        for name in ("max_steps", "eval_every", "eval_batch_size", "early_stopping_patience"):
            if getattr(self, name) is not None:
                self._set_integer(name, minimum=1)
        for name in ("warmup_steps", "num_workers", "seed"):
            self._set_integer(name, minimum=0)

    def _set_integer(self, name: str, *, minimum: int) -> None:
        """Validate an integer field and store it as a Python `int`.

        NumPy integers are accepted (values often come from an array or a sweep) and converted, so
        that code downstream which insists on `int`, such as the DataLoader, is not surprised.
        Booleans are rejected: `True` is an `int` to Python, and never what was meant here.
        """
        value = getattr(self, name)
        if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < minimum:
            kind = "a positive integer" if minimum == 1 else "a non-negative integer"
            raise ValueError(f"{name} must be {kind}, got {value!r}")
        object.__setattr__(self, name, int(value))

    @property
    def resolved_learning_rate(self) -> float:
        """`learning_rate`, or the default starting point for `mode`."""
        return DEFAULT_LEARNING_RATES[self.mode] if self.learning_rate is None else self.learning_rate


def _one_of(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _unique_choices(name: str, values: Sequence[str], allowed: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence of {allowed}, got {values!r}")
    items = tuple(values)
    if len(items) == 0:
        raise ValueError(f"{name} must not be empty")
    invalid = [v for v in items if v not in allowed]
    if invalid:
        raise ValueError(f"{name} holds invalid entries {invalid!r}; expected entries from {allowed}")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} holds duplicate entries: {items!r}")
    return items
