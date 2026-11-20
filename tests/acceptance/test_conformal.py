"""Conformal calibration (spec §9, T13).

Placeholders only: every test here is skipped until it is implemented. A skipped placeholder is
not a passing acceptance test. Replace the skip marker when implementing; keep the test name.
"""

import pytest

NOT_IMPLEMENTED = pytest.mark.skip(reason="acceptance test not implemented yet (roadmap phase 2)")


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t13_conformal_coverage_and_monotonicity():
    """T13. Passes when: On synthetic exchangeable data, calibrated coverage meets the nominal level;
    output quantiles are non-decreasing.
    """
