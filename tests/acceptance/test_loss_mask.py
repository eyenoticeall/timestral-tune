"""Loss-mask test (handoff §9, T8).

Placeholders only: every test here is skipped until it is implemented. A skipped placeholder is
not a passing acceptance test. Replace the skip marker when implementing; keep the test name.
"""

import pytest

NOT_IMPLEMENTED = pytest.mark.skip(reason="acceptance test not implemented yet (roadmap phase 1)")


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t08_loss_mask():
    """T8. Passes when: The gradient of the loss with respect to predictions is exactly zero at covariate
    rows, unobserved steps and horizon padding.
    """
