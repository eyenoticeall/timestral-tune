"""Loss-mask test (spec §9, T8)."""

from __future__ import annotations

import pytest
import torch

from timestral_tune.utils import check_loss_mask

from ..helpers import upstream_batch


@pytest.mark.acceptance
@pytest.mark.parametrize("scenario", ["covariates", "multi_target_covariates", "past_only_covariates"], indirect=True)
@pytest.mark.parametrize("frame", ["median_folded", "true_values"])
def test_t08_loss_mask(forecaster, raw, frame):
    """T8. Passes when: The gradient of the loss with respect to predictions is exactly zero at covariate
    rows, unobserved steps and horizon padding.
    """
    batch, _, _ = upstream_batch(forecaster, raw)
    weight = batch.loss_weight
    target_rows = batch.target_rows

    # The batch must actually contain each kind of unscored position, or the test proves nothing.
    padding = torch.zeros_like(weight)
    padding[:, raw.horizon :] = True
    unobserved = target_rows[:, None] & ~padding & ~weight
    assert (~target_rows).any(), "no covariate rows in the batch"
    assert padding.any(), "no horizon padding in the batch"
    assert unobserved.any(), "no unobserved target step in the batch"

    result = check_loss_mask(forecaster.model, batch, frame)
    assert not result.skipped
    assert result.passed, str(result)
    assert all(parameter.grad is None for parameter in forecaster.model.parameters()), "the check left gradients behind"
