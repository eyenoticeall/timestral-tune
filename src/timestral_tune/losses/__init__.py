"""Loss functions.

See spec §4.
"""

from .pinball import PinballLoss, pinball_loss

__all__ = ["PinballLoss", "pinball_loss"]
