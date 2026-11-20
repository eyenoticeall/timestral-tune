"""LoRA adapter tests (handoff §9, T9-T10).

Placeholders only: every test here is skipped until it is implemented. A skipped placeholder is
not a passing acceptance test. Replace the skip marker when implementing; keep the test name.
"""

import pytest

NOT_IMPLEMENTED = pytest.mark.skip(reason="acceptance test not implemented yet (roadmap phase 2)")


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t09_lora_identity_and_merge():
    """T9. Passes when: A freshly initialised adapter leaves outputs unchanged; merged and unmerged outputs
    agree within tolerance.
    """


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t10_single_row_guard():
    """T10. Passes when: `lora_variate` on single-row data raises; with the override, query and key slices
    are rejected.
    """
