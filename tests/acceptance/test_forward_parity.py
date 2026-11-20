"""Forward and frame parity with upstream inference (spec §9, T1-T3).

Placeholders only: every test here is skipped until it is implemented. A skipped placeholder is
not a passing acceptance test. Replace the skip marker when implementing; keep the test name.
"""

import pytest

NOT_IMPLEMENTED = pytest.mark.skip(reason="acceptance test not implemented yet (roadmap phase 1)")


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t01_forward_parity_no_covariates():
    """T1. Passes when: `sinh(training_forward) * sigma_0 + mu_0` equals `predict_block` output within
    tolerance, no covariates.
    """


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t02_forward_parity_with_covariates():
    """T2. Passes when: Same as T1 for the `2n`-token layout with target, past-only and known-future rows."""


@pytest.mark.acceptance
@NOT_IMPLEMENTED
def test_t03_frame_parity():
    """T3. Passes when: Applying the median-folded frame reproduces `_refine_block`, and the full pipeline
    matches `forecast(symmetric_averaging=False, clip_nonnegative=False)` on a forecaster built with
    `quantile_real_cap_k=0`.
    """
