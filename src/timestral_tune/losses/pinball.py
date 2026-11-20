"""Quantile (pinball) loss with a weight mask.

For a target `z` and predicted quantile `q` at level `tau`:

    rho_tau(e) = max(tau * e, (tau - 1) * e),   e = z - q

The batch loss averages equally over quantile levels and over scored positions (spec §4.2, the
paper's eq. 3):

    L = sum_i w_i * sum_tau rho_tau(z_i - q_{tau,i}) / (|Q| * sum_i w_i)

Positions with `w_i = 0` contribute exactly nothing: neither value nor gradient, even if their
target or prediction is not finite. When no position is scored the loss is exactly 0 with zero
gradients; the trainer skips such batches rather than dividing by an epsilon.

- Spec: `docs/SPEC.md` §4.1, §4.2
- Acceptance tests: T8
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["PinballLoss", "pinball_loss"]

Reduction = Literal["mean", "none"]


def _check_levels(levels: Tensor) -> None:
    if levels.dim() != 1 or levels.numel() == 0:
        raise ValueError(f"quantile levels must be a non-empty 1-D sequence, got shape {tuple(levels.shape)}")
    if not bool(((levels > 0) & (levels < 1)).all()):
        raise ValueError("quantile levels must lie strictly between 0 and 1")
    if not bool((levels.diff() > 0).all()):
        raise ValueError("quantile levels must be strictly increasing")


def pinball_loss(
    quantiles: Tensor,
    targets: Tensor,
    levels: Tensor,
    weight: Tensor | None = None,
    reduction: Reduction = "mean",
) -> Tensor:
    """Pinball loss of `quantiles` `(..., Q)` against `targets` `(...)`.

    Args:
        quantiles: Predicted quantiles, last axis indexed by level.
        targets: Target values, shaped like `quantiles` without its last axis.
        levels: `(Q,)` quantile levels in (0, 1), strictly increasing.
        weight: Optional bool mask shaped like `targets`; True marks a scored position.
        reduction: `"mean"` returns the scalar batch loss. `"none"` returns the per-position loss
            `(...)`, averaged over levels, with unscored positions set to 0.

    The computation runs in at least float32, whatever the input dtype.
    """
    if quantiles.shape[:-1] != targets.shape:
        raise ValueError(
            f"targets {tuple(targets.shape)} must match quantiles {tuple(quantiles.shape)} without the last axis"
        )
    if quantiles.shape[-1] != levels.numel():
        raise ValueError(f"quantiles have {quantiles.shape[-1]} levels on the last axis, expected {levels.numel()}")
    if weight is not None:
        if weight.shape != targets.shape:
            raise ValueError(f"weight {tuple(weight.shape)} must be shaped like targets {tuple(targets.shape)}")
        if weight.dtype != torch.bool:
            raise ValueError(f"weight must be a bool mask, got {weight.dtype}")
    if reduction not in ("mean", "none"):
        raise ValueError(f"unknown reduction {reduction!r}; expected 'mean' or 'none'")

    work = torch.promote_types(quantiles.dtype, torch.float32)
    predictions = quantiles.to(work)
    values = targets.to(work)
    if weight is not None:
        # Unscored targets may hold anything, including NaN; they must not reach the arithmetic.
        values = torch.where(weight, values, torch.zeros_like(values))
    tau = levels.to(device=predictions.device, dtype=work)

    error = values.unsqueeze(-1) - predictions
    per_level = torch.maximum(tau * error, (tau - 1) * error)
    if weight is not None:
        # `where` (not multiplication) so that unscored positions get an exactly zero gradient.
        per_level = torch.where(weight.unsqueeze(-1), per_level, torch.zeros_like(per_level))
    per_position = per_level.mean(dim=-1)
    if reduction == "none":
        return per_position

    scored = targets.numel() if weight is None else weight.sum()
    # With no scored position the numerator is an exact, graph-connected zero.
    return per_position.sum() / torch.as_tensor(scored, dtype=work, device=predictions.device).clamp_min(1)


class PinballLoss(nn.Module):
    """`pinball_loss` bound to a fixed set of quantile levels.

    Args:
        quantile_levels: The levels of the model's quantile head, e.g. `model.config.quantile_knots`.
    """

    levels: Tensor

    def __init__(self, quantile_levels: Sequence[float]) -> None:
        super().__init__()
        levels = torch.as_tensor(list(quantile_levels), dtype=torch.float64)
        _check_levels(levels)
        self.register_buffer("levels", levels, persistent=False)

    def forward(
        self,
        quantiles: Tensor,
        targets: Tensor,
        weight: Tensor | None = None,
        reduction: Reduction = "mean",
    ) -> Tensor:
        return pinball_loss(quantiles, targets, self.levels, weight, reduction)

    def extra_repr(self) -> str:
        return f"num_levels={self.levels.numel()}"
