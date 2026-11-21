"""Model layer: loading, device shims, the training forward, LoRA and conformal calibration.

See spec §3 (model facts), §4.3 (normalised space) and §5 (adaptation modes).
"""

from .adapter import (
    LoRASliceLinear,
    extract_lora_state_dict,
    has_lora,
    inject_lora,
    load_lora,
    load_lora_state_dict,
    lora_parameters,
    merge_lora,
    merged_state_dict,
    unmerge_lora,
    validate_lora_for_rows,
)
from .conformal import (
    DEFAULT_QUANTILE_LEVELS,
    ConformalCalibrator,
    OnlineConformalCalibrator,
    collect_conformal_data,
)
from .device import MIN_TORCH_FOR_MPS, TuneTimestralForecaster, TuneTimestralModel, mps_is_usable, supports_float64
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
    "DEFAULT_QUANTILE_LEVELS",
    "MIN_TORCH_FOR_MPS",
    "PINNED_REVISIONS",
    "ConformalCalibrator",
    "ForwardOutput",
    "LoRASliceLinear",
    "OnlineConformalCalibrator",
    "TargetFrame",
    "TuneTimestralForecaster",
    "TuneTimestralModel",
    "collect_conformal_data",
    "denormalise_quantiles",
    "extract_lora_state_dict",
    "has_lora",
    "inject_lora",
    "load_forecaster",
    "load_lora",
    "load_lora_state_dict",
    "load_model",
    "lora_parameters",
    "merge_lora",
    "merged_state_dict",
    "mps_is_usable",
    "normalise_targets",
    "resolve_device",
    "resolve_revision",
    "supports_float64",
    "target_frames",
    "training_forward",
    "unmerge_lora",
    "validate_lora_for_rows",
]
