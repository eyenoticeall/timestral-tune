"""timestral-tune: fine-tuning, selective LoRA and conformal calibration for Timestral models.

Status: pre-alpha (Phases 0-2). Supports loading on CPU/CUDA/MPS, leak-free datasets and
collation, normalised-space training across `head_only`, `lora_variate`, `lora_temporal`, and
`full_ft`, post-hoc CQR/ACI conformal calibration, and upstream-loadable checkpoint export.
See `README.md`, `docs/STATUS.md` and `docs/SPEC.md`.
"""

from .config import LoraConfig, TrainingConfig
from .export import export_checkpoint
from .models.adapter import load_lora
from .models.conformal import ConformalCalibrator, OnlineConformalCalibrator, collect_conformal_data
from .trainer import TimestralTrainer

__version__ = "0.1.0.dev0"

__all__ = [
    "ConformalCalibrator",
    "LoraConfig",
    "OnlineConformalCalibrator",
    "TimestralTrainer",
    "TrainingConfig",
    "__version__",
    "collect_conformal_data",
    "export_checkpoint",
    "load_lora",
]
