"""Data layer: long-frame loading, point-in-time windowing, DHF collation and encoders.

See spec §6.2 (data contract) and §6.3 (point-in-time rules).
"""

from .batch import TrainingBatch, bucket_series_count, pad_batch_to_series
from .collator import DHFDataCollator, Window
from .dataset import ColumnRoles, SeriesData, TimeSeriesDataset, WindowStats
from .encoders import encode_categorical_window, encoding_target, forward_fill, window_category_codes

__all__ = [
    "ColumnRoles",
    "DHFDataCollator",
    "SeriesData",
    "TimeSeriesDataset",
    "TrainingBatch",
    "Window",
    "WindowStats",
    "bucket_series_count",
    "encode_categorical_window",
    "encoding_target",
    "forward_fill",
    "pad_batch_to_series",
    "window_category_codes",
]
