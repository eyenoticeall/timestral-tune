"""Model layer: loading, device shims, the training forward, LoRA and conformal calibration.

See spec §3 (model facts), §4.3 (normalised space) and §5 (adaptation modes).
"""

from .device import TuneTimestralForecaster, TuneTimestralModel, supports_float64
from .forward import (
    ForwardOutput,
    TargetFrame,
    denormalise_quantiles,
    normalise_targets,
    target_frames,
    training_forward,
)
from .loader import PINNED_REVISIONS, load_forecaster, load_model, resolve_device, resolve_revision

__all__ = [
    "PINNED_REVISIONS",
    "ForwardOutput",
    "TargetFrame",
    "TuneTimestralForecaster",
    "TuneTimestralModel",
    "denormalise_quantiles",
    "load_forecaster",
    "load_model",
    "normalise_targets",
    "resolve_device",
    "resolve_revision",
    "supports_float64",
    "target_frames",
    "training_forward",
]
