"""Device parity (handoff §9, T14).

A prototype of this check is `scripts/probes/device_parity.py`. Compare gradients against the
float64-weights reference, not against CPU fp32 (see handoff §12 and docs/VERIFICATION.md).

Placeholders only: every test here is skipped until it is implemented. A skipped placeholder is
not a passing acceptance test. Replace the skip marker when implementing; keep the test name.
"""

import pytest

NOT_IMPLEMENTED = pytest.mark.skip(reason="acceptance test not implemented yet (roadmap phase 0)")


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t14_device_parity():
    """T14. Passes when: On each available device, forecasts match CPU within 1e-4 relative, and training-
    loss gradients match a float64-weights reference within 1e-3 relative. Skipped where the device is
    absent.
    """
