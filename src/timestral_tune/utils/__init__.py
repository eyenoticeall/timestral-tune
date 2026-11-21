"""Metrics and verification helpers.

See spec §5.5 and §9.
"""

from .metrics import (
    ForgettingSummary,
    MetricsSummary,
    compute_forgetting,
    coverage,
    evaluate_forecaster,
    evaluate_forgetting,
    evaluate_predictions,
    interval_width,
    mase,
    pinball_loss_raw,
    weighted_quantile_loss,
)
from .validation import (
    CheckResult,
    check_firewall,
    check_forward_parity,
    check_known_future_sensitivity,
    check_label_isolation,
    check_loss_mask,
    run_checks,
)

__all__ = [
    "CheckResult",
    "ForgettingSummary",
    "MetricsSummary",
    "check_firewall",
    "check_forward_parity",
    "check_known_future_sensitivity",
    "check_label_isolation",
    "check_loss_mask",
    "compute_forgetting",
    "coverage",
    "evaluate_forecaster",
    "evaluate_forgetting",
    "evaluate_predictions",
    "interval_width",
    "mase",
    "pinball_loss_raw",
    "run_checks",
    "weighted_quantile_loss",
]
