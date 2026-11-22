"""timestral-tune: fine-tuning, selective LoRA and conformal calibration for Timestral models.

Status: alpha (v0.1; roadmap Phases 0-4). Supports loading on CPU/CUDA/MPS (5M-209M, contexts up to 32,768),
leak-free datasets and collation, normalised-space training across `head_only`, `lora_variate`,
`lora_temporal`, and `full_ft`, sign-flip augmentation, replay mixing and frozen-base distillation,
post-hoc CQR/ACI conformal calibration, raw-space evaluation and forgetting metrics, optional Lacuna
integration, and upstream-loadable checkpoint export.
See `README.md`, `docs/STATUS.md` and `docs/SPEC.md`.
"""

from .config import LoraConfig, ReplayConfig, TrainingConfig
from .export import export_checkpoint
from .models.adapter import load_lora
from .models.conformal import ConformalCalibrator, OnlineConformalCalibrator, collect_conformal_data
from .replay import SyntheticReplaySource, distillation_loss
from .trainer import TimestralTrainer

__version__ = "0.1.0"

__all__ = [
    "ConformalCalibrator",
    "LoraConfig",
    "OnlineConformalCalibrator",
    "ReplayConfig",
    "SyntheticReplaySource",
    "TimestralTrainer",
    "TrainingConfig",
    "__version__",
    "collect_conformal_data",
    "distillation_loss",
    "export_checkpoint",
    "load_lora",
]
