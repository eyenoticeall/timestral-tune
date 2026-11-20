# Vendored from https://github.com/mistralai/timestral at commit a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff (Apache-2.0).
# Modified by timestral-tune: absolute `timestral.*` imports were rewritten to relative
# imports so that this copy never resolves to an installed `timestral`. No other changes.
# Regenerate with scripts/vendor_upstream.py; do not edit by hand.
"""Standalone inference for Timestral time-series forecasting models."""

from .config import TimestralConfig
from .forecaster import Forecast, TimestralForecaster
from .loading import load_model
from .model import TimestralModel

__all__ = ["Forecast", "TimestralConfig", "TimestralForecaster", "TimestralModel", "load_model"]
__version__ = "1.0.0"
