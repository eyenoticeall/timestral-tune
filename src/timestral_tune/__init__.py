"""timestral-tune: fine-tuning, selective LoRA and conformal calibration for Timestral models.

Status: pre-alpha scaffold. No training, adaptation or calibration functionality is implemented
yet. See `README.md`, `docs/STATUS.md` and `docs/SPEC.md`.

The public API of spec §7 (`TimestralTrainer`, `TrainingConfig`, `LoraConfig`, ...) is exported
from here as each piece lands. Until then only `__version__` is exported.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
