"""Training loop and callbacks.

See spec §5.4.
"""

from .callbacks import Callback, EarlyStopping, ProgressLogger
from .step import LossOutput, compute_loss
from .trainer import EvalResult, TimestralTrainer, TrainingResult

__all__ = [
    "Callback",
    "EarlyStopping",
    "EvalResult",
    "LossOutput",
    "ProgressLogger",
    "TimestralTrainer",
    "TrainingResult",
    "compute_loss",
]
