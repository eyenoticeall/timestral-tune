"""timestral-tune: fine-tuning, selective LoRA and conformal calibration for Timestral models.

Status: pre-alpha. Implemented so far: loading on any device, the dataset and collator, the
normalised-space training forward and loss, and `head_only` training. The LoRA modes, full
fine-tuning, conformal calibration and export are not built yet. See `README.md`,
`docs/STATUS.md` and `docs/SPEC.md`.
"""

from .config import LoraConfig, TrainingConfig
from .trainer import TimestralTrainer

__version__ = "0.1.0.dev0"

__all__ = ["LoraConfig", "TimestralTrainer", "TrainingConfig", "__version__"]
