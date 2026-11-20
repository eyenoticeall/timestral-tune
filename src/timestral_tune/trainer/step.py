"""One loss evaluation: forward, target frame, normalised targets, pinball loss.

This is the whole of spec §4 in one function, shared by the trainer, the evaluation loop, the
acceptance tests and the verification probes, so that they cannot drift apart.

- Spec: `docs/SPEC.md` §4, §5.4
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .._vendor.timestral.model import TimestralModel
from ..data.batch import TrainingBatch
from ..losses.pinball import pinball_loss
from ..models.forward import ForwardOutput, TargetFrame, normalise_targets, target_frames, training_forward

__all__ = ["LossOutput", "compute_loss"]


@dataclass(frozen=True)
class LossOutput:
    """Result of `compute_loss`.

    Attributes:
        loss: Scalar pinball loss in normalised space, averaged over quantile levels and scored
            positions. Exactly 0 when the batch has no scored position.
        forward: The forward pass the loss was computed from.
        normalised_targets: `(R, H)` targets in the chosen frame; 0 where unscored.
        frame_loc: `(R, horizon_patches)` location of each horizon patch's frame.
        frame_scale: `(R, horizon_patches)` scale of each horizon patch's frame.
    """

    loss: Tensor
    forward: ForwardOutput
    normalised_targets: Tensor
    frame_loc: Tensor
    frame_scale: Tensor


def compute_loss(
    model: TimestralModel,
    batch: TrainingBatch,
    frame: TargetFrame = "median_folded",
    *,
    return_hidden: bool = False,
) -> LossOutput:
    """The training loss of `model` on `batch`, with gradients with respect to the model's parameters.

    The target frame carries no gradient: with `median_folded` the model's own medians define the
    frame, but only as constants.
    """
    forward = training_forward(model, batch, return_hidden=return_hidden)
    loc, scale = target_frames(model, batch, forward, frame)
    normalised = normalise_targets(model, batch, loc, scale)
    levels = forward.quantiles.new_tensor(model.config.quantile_knots)
    loss = pinball_loss(forward.quantiles, normalised, levels, batch.loss_weight)
    return LossOutput(loss=loss, forward=forward, normalised_targets=normalised, frame_loc=loc, frame_scale=scale)
