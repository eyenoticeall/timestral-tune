"""Leakage and point-in-time tests on the training forward (spec §9, T4-T7 and T12).

Perturbations must be non-constant noise: a constant offset is absorbed by the normalisation
and would make a firewall test pass vacuously.

Placeholders only: every test here is skipped until it is implemented. A skipped placeholder is
not a passing acceptance test. Replace the skip marker when implementing; keep the test name.
"""

import pytest

NOT_IMPLEMENTED = pytest.mark.skip(reason="acceptance test not implemented yet (roadmap phase 1)")


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t04_label_isolation():
    """T4. Passes when: Replacing horizon target values with noise leaves predictions bit-identical."""


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t05_past_covariate_cut():
    """T5. Passes when: Changing a past-only covariate after the origin (in the raw frame) changes nothing."""


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t06_firewall():
    """T6. Passes when: Noise on a target's context changes no covariate row's hidden state or output."""


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t07_known_future_positive_control():
    """T7. Passes when: Noise on a known-future covariate in the horizon does change the target forecast."""


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t12_encoding_causality():
    """T12. Passes when: A window's categorical encodings are unchanged when data after its origin changes."""
