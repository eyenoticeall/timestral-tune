"""Anti-forgetting: replay sources and distillation (spec §5.3)."""

from .distill import distillation_loss
from .generators import ALL_SYNTHETIC_FAMILIES, SyntheticFamily, SyntheticReplaySource

__all__ = [
    "ALL_SYNTHETIC_FAMILIES",
    "SyntheticFamily",
    "SyntheticReplaySource",
    "distillation_loss",
]
