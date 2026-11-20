"""Loss-mask test (spec §9, T8)."""

from __future__ import annotations

import pytest
import torch

from timestral_tune.trainer.step import compute_loss

from ..helpers import upstream_batch


@pytest.mark.acceptance
@pytest.mark.parametrize("scenario", ["covariates", "multi_target_covariates"], indirect=True)
@pytest.mark.parametrize("frame", ["median_folded", "true_values"])
def test_t08_loss_mask(forecaster, raw, frame):
    """T8. Passes when: The gradient of the loss with respect to predictions is exactly zero at covariate
    rows, unobserved steps and horizon padding.
    """
    model = forecaster.model
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

    result = compute_loss(model, batch, frame)
    quantiles = result.forward.quantiles
    quantiles.retain_grad()
    result.loss.backward()
    loss = result.loss

    gradient = quantiles.grad
    assert torch.isfinite(loss) and loss > 0
    assert torch.isfinite(gradient).all()
    assert (gradient[~weight] == 0).all(), "unscored positions received a gradient"
    assert (gradient[weight].abs().sum(dim=-1) > 0).all(), "a scored position received no gradient"
    model.zero_grad()
