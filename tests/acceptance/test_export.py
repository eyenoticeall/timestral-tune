"""Export round-trip (spec §9, T11).

Placeholders only: every test here is skipped until it is implemented. A skipped placeholder is
not a passing acceptance test. Replace the skip marker when implementing; keep the test name.
"""

import pytest

NOT_IMPLEMENTED = pytest.mark.skip(reason="acceptance test not implemented yet (roadmap phase 2)")


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t11_export_round_trip():
    """T11. Passes when: The exported directory loads in the vendored upstream loader with `strict=True`
    and reproduces the trainer's forecasts.
    """
