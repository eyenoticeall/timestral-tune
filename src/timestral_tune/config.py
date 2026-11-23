"""Configuration dataclasses.

`TrainingConfig` holds everything the trainer needs apart from the data, `LoraConfig`
configures selective slice LoRA on attention projections (spec §5.1), and `ReplayConfig`
configures anti-forgetting replay mixing and frozen-base distillation (spec §5.3).

- Spec: `docs/SPEC.md` §5 (modes), §5.1 (LoRA), §5.3 (replay), §5.4 (trainer), §7.1 (API example)
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
    "ReplayConfig",
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

# Sweep-backed default learning rates (bracketed Phase 3 validation sweeps on 5M, 28M, 123M, and
# 209M across realised volatility, retail demand, and signed returns; see `docs/BENCHMARKS.md` §2
# and `docs/SPEC.md` D19).
DEFAULT_LEARNING_RATES: dict[AdaptationMode, float] = {
    "head_only": 3e-3,
    "lora_variate": 3e-3,
    "lora_temporal": 1e-3,
    "full_ft": 1e-4,
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
    targets: tuple[LoraTarget, ...] = ("q", "k", "v", "o")
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
class ReplayConfig:
    """Settings for anti-forgetting replay mixing and frozen-base distillation (spec §5.3).

    At each training step the trainer combines the domain pinball loss with a replay micro-batch:

    `loss = (1 - replay_fraction) * L_domain + replay_fraction * L_replay + distill_weight * L_distill`

    where `L_replay` is the pinball loss on labelled replay windows (skipped when
    `replay_fraction == 0`) and `L_distill` is the smooth L1 (Huber, controlled by `smooth_beta`)
    difference between the tuned model's and the frozen base model's normalised quantiles on
    replay windows (skipped when `distill_weight == 0`, and needs no labels).

    Defaults to light label-free distillation (`replay_fraction=0.0`, `distill_weight=0.1`), which
    avoids fitting synthetic generator labels in `full_ft` while anchoring predictions toward the
    base model across all adaptation modes. For LoRA modes, the frozen base is always the
    unadapted base model (`disable_lora`), even when resuming from a `load_lora` checkpoint; for
    `head_only` and `full_ft`, it is the starting `model` passed to `TimestralTrainer`.

    Attributes:
        replay_fraction: Weight of replay pinball loss relative to domain pinball loss, in
            `[0.0, 1.0)`. Defaults to `0.0` (distillation-only anchoring).
        distill_weight: Non-negative weight on frozen-base quantile distillation over replay
            windows. Defaults to `0.1`. At least one of `replay_fraction` or `distill_weight`
            must be positive.
        batch_size: Replay windows per training step, per process (per rank in multi-process mode).
            `None` picks `max(1, round(train_batch_size * replay_fraction))` when
            `replay_fraction > 0`, or `max(1, train_batch_size // 4)` for distillation-only replay.
            In multi-process mode, this resolution uses each rank's `TrainingConfig.batch_size`, so
            the global effective replay batch size is `world_size * resolved_batch_size(batch_size)`
            (which may differ from `resolved_batch_size(world_size * batch_size)` when
            `batch_size=None` due to rounding or the minimum of 1).
        sign_flip: Whether to apply random per-series sign flips (`p=0.5`) to replay batches.
            `None` inherits `TrainingConfig.sign_flip`. When distillation is enabled, both the
            tuned and frozen base models see the same orientation.
        cache_base_outputs: When `distill_weight > 0`, precompute the frozen base model's
            normalised quantiles on CPU (up to 256 MiB) before training starts so `full_ft` and
            LoRA avoid both a second copy of the base model in accelerator memory and a second
            forward pass every step. Falls back to live frozen-base evaluation (`disable_lora`
            for LoRA; a frozen model copy on the device for `full_ft`) above the cap or when
            False.
        smooth_beta: Huber/Smooth-L1 transition threshold passed to `distillation_loss` (`>= 0`).
            Default `1e-3` avoids `+/-1` subgradient kicks from `~1e-7` fp32 GEMM tiling
            round-off at step 0; set to `0.0` for exact unsmoothed L1.
    """

    replay_fraction: float = 0.0
    distill_weight: float = 0.1
    batch_size: int | None = None
    sign_flip: bool | None = None
    cache_base_outputs: bool = True
    smooth_beta: float = 1e-3

    def __post_init__(self) -> None:
        if isinstance(self.replay_fraction, bool) or not (
            isinstance(self.replay_fraction, numbers.Real)
            and math.isfinite(self.replay_fraction)
            and 0.0 <= self.replay_fraction < 1.0
        ):
            raise ValueError(f"replay_fraction must be in [0, 1), got {self.replay_fraction!r}")
        object.__setattr__(self, "replay_fraction", float(self.replay_fraction))

        if isinstance(self.distill_weight, bool) or not (
            isinstance(self.distill_weight, numbers.Real)
            and math.isfinite(self.distill_weight)
            and self.distill_weight >= 0.0
        ):
            raise ValueError(f"distill_weight must be non-negative and finite, got {self.distill_weight!r}")
        object.__setattr__(self, "distill_weight", float(self.distill_weight))

        if self.replay_fraction == 0.0 and self.distill_weight == 0.0:
            raise ValueError("at least one of replay_fraction or distill_weight must be positive")

        if self.batch_size is not None:
            if (
                not isinstance(self.batch_size, numbers.Integral)
                or isinstance(self.batch_size, bool)
                or self.batch_size < 1
            ):
                raise ValueError(f"batch_size must be a positive integer or None, got {self.batch_size!r}")
            object.__setattr__(self, "batch_size", int(self.batch_size))

        if self.sign_flip is not None and not isinstance(self.sign_flip, bool):
            raise ValueError(f"sign_flip must be a bool or None, got {self.sign_flip!r}")

        if not isinstance(self.cache_base_outputs, bool):
            raise ValueError(f"cache_base_outputs must be a bool, got {self.cache_base_outputs!r}")

        if isinstance(self.smooth_beta, bool) or not (
            isinstance(self.smooth_beta, numbers.Real) and math.isfinite(self.smooth_beta) and self.smooth_beta >= 0.0
        ):
            raise ValueError(f"smooth_beta must be non-negative and finite, got {self.smooth_beta!r}")
        object.__setattr__(self, "smooth_beta", float(self.smooth_beta))

    def resolved_batch_size(self, train_batch_size: int) -> int:
        """Number of replay windows drawn per training step (per rank in multi-process mode)."""
        if self.batch_size is not None:
            return self.batch_size
        if self.replay_fraction > 0.0:
            return max(1, round(train_batch_size * self.replay_fraction))
        return max(1, train_batch_size // 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_fraction": self.replay_fraction,
            "distill_weight": self.distill_weight,
            "batch_size": self.batch_size,
            "sign_flip": self.sign_flip,
            "cache_base_outputs": self.cache_base_outputs,
            "smooth_beta": self.smooth_beta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayConfig:
        known = {"replay_fraction", "distill_weight", "batch_size", "sign_flip", "cache_base_outputs", "smooth_beta"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown ReplayConfig keys: {sorted(unknown)}")
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
        learning_rate: Peak learning rate. None picks the default for `mode`. For `full_ft` on
            `5M` and `28M`, setting `learning_rate=3e-4` is recommended for lower error (three
            seeds on `5M`, one on `28M`), at the cost of roughly two to three percentage points
            more forgetting; keep `1e-4` on `123M` and `209M` (see `docs/BENCHMARKS.md` §2.2).
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
        num_workers: DataLoader worker processes per persistent pool. When `num_workers > 0`, each
            process (or rank in multi-process mode) keeps up to two persistent worker pools during
            `fit()`: training, plus validation when `val_dataset` is given. The replay loader always
            runs in the main process (`num_workers=0`). On macOS with PyTorch 2.4.0, any worker
            leaves the process hanging at exit (a PyTorch issue; `docs/VERIFICATION.md` §10).
        eval_every: Evaluate on the validation set every this many steps. None evaluates at the
            end of each epoch.
        eval_batch_size: Batch size for evaluation. None uses `batch_size`.
        early_stopping_patience: Stop after this many evaluations without improvement. None
            disables early stopping.
        restore_best: After training with a validation set, put back the trainable parameters
            from the evaluation with the lowest loss.
        log_every: Log training progress every this many steps.
        shuffle: Shuffle the training windows each epoch.
        sign_flip: Randomly negate whole series (targets and covariates together, p=0.5 per series)
            during training so fine-tuning stays coherent with upstream's symmetric averaging
            (default True; spec §5.3, D19). In `head_only` with `cache_frozen_features=True`, both
            `+x` and `-x` orientations are cached up to the 256 MiB cap.
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
    sign_flip: bool = True
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
        for flag in ("cache_frozen_features", "compile_model", "restore_best", "shuffle", "sign_flip"):
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
