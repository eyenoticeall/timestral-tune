"""One loss evaluation: forward, target frame, normalised targets, pinball loss.

This is the whole of spec §4 in one function, shared by the trainer, the evaluation loop, the
acceptance tests and the verification probes, so that they cannot drift apart.

- Spec: `docs/SPEC.md` §4, §5.4
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .._vendor.timestral.model import TimestralModel
from .._vendor.timestral.scaler import apply_affine_asinh_masked
from ..data.batch import TrainingBatch
from ..losses.pinball import pinball_loss
from ..models.device import supports_float64
from ..models.forward import (
    TARGET_FRAMES,
    ForwardOutput,
    TargetFrame,
    advancing_block_frames_from_state,
    normalise_targets,
    target_frames,
    training_forward,
)

__all__ = ["CachedHeadBatch", "LossOutput", "compute_loss"]

_LEVELS_CACHE: dict[tuple[tuple[float, ...], torch.device, torch.dtype], Tensor] = {}


def _cached_levels(knots: tuple[float, ...], device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (knots, device, dtype)
    cached = _LEVELS_CACHE.get(key)
    if cached is None:
        if len(_LEVELS_CACHE) >= 32:
            _LEVELS_CACHE.clear()
        cached = torch.tensor(knots, device=device, dtype=dtype)
        _LEVELS_CACHE[key] = cached
    return cached


@dataclass(frozen=True)
class CachedHeadBatch:
    """Precomputed frozen-body outputs for target rows, used to avoid re-running the body in `head_only`.

    Attributes:
        horizon_hidden: `(R_tgt, horizon_patches, D)` frozen transformer outputs at the horizon
            prediction tokens of target rows.
        anchor_loc: `(R_tgt,)` location of the last context patch.
        anchor_scale: `(R_tgt,)` scale of the last context patch.
        targets: `(R_tgt, H)` raw target labels; 0 where unscored.
        loss_weight: `(R_tgt, H)` bool mask of scored positions.
        context_welford: Optional `(cnt, mean, m2)` float64 Welford summary of each row's context
            for `median_folded` horizons of more than one patch. Kept on CPU when the target
            device lacks float64.
        normalised_targets: Optional `(R_tgt, H)` pre-normalised targets when `true_values` is
            used (independent of the head's weights).
    """

    horizon_hidden: Tensor
    anchor_loc: Tensor
    anchor_scale: Tensor
    targets: Tensor
    loss_weight: Tensor
    context_welford: tuple[Tensor, Tensor, Tensor] | None = None
    normalised_targets: Tensor | None = None
    frame_loc: Tensor | None = None
    frame_scale: Tensor | None = None

    @property
    def num_scored(self) -> int:
        return int(self.loss_weight.sum())

    @property
    def horizon_steps(self) -> int:
        return self.targets.shape[1]

    @property
    def device(self) -> torch.device:
        return self.horizon_hidden.device

    def to(self, device: torch.device | str, non_blocking: bool = False) -> CachedHeadBatch:
        target_device = torch.device(device)
        welford = self.context_welford
        if welford is not None:
            stats_device = target_device if supports_float64(target_device) else torch.device("cpu")
            welford = tuple(t.to(stats_device, non_blocking=non_blocking) for t in welford)
        return CachedHeadBatch(
            horizon_hidden=self.horizon_hidden.to(target_device, non_blocking=non_blocking),
            anchor_loc=self.anchor_loc.to(target_device, non_blocking=non_blocking),
            anchor_scale=self.anchor_scale.to(target_device, non_blocking=non_blocking),
            targets=self.targets.to(target_device, non_blocking=non_blocking),
            loss_weight=self.loss_weight.to(target_device, non_blocking=non_blocking),
            context_welford=welford,
            normalised_targets=(
                None
                if self.normalised_targets is None
                else self.normalised_targets.to(target_device, non_blocking=non_blocking)
            ),
            frame_loc=(None if self.frame_loc is None else self.frame_loc.to(target_device, non_blocking=non_blocking)),
            frame_scale=(
                None if self.frame_scale is None else self.frame_scale.to(target_device, non_blocking=non_blocking)
            ),
        )

    def pin_memory(self) -> CachedHeadBatch:
        return CachedHeadBatch(
            horizon_hidden=self.horizon_hidden.pin_memory(),
            anchor_loc=self.anchor_loc.pin_memory(),
            anchor_scale=self.anchor_scale.pin_memory(),
            targets=self.targets.pin_memory(),
            loss_weight=self.loss_weight.pin_memory(),
            context_welford=(
                None if self.context_welford is None else tuple(t.pin_memory() for t in self.context_welford)
            ),
            normalised_targets=(None if self.normalised_targets is None else self.normalised_targets.pin_memory()),
            frame_loc=(None if self.frame_loc is None else self.frame_loc.pin_memory()),
            frame_scale=(None if self.frame_scale is None else self.frame_scale.pin_memory()),
        )


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


def _compute_cached_head_loss(model: TimestralModel, batch: CachedHeadBatch, frame: TargetFrame) -> LossOutput:
    if frame not in TARGET_FRAMES:
        raise ValueError(f"unknown target frame {frame!r}; expected one of {TARGET_FRAMES}")
    config = model.config
    patch = config.patch_size
    horizon_patches = batch.horizon_steps // patch
    device = batch.device
    dtype = batch.targets.dtype

    quantiles = model.head(batch.horizon_hidden)
    forward = ForwardOutput(
        quantiles=quantiles,
        anchor_loc=batch.anchor_loc,
        anchor_scale=batch.anchor_scale,
        hidden=None,
        num_context_tokens=0,
    )
    if frame == "true_values":
        if batch.normalised_targets is None or batch.frame_loc is None or batch.frame_scale is None:
            raise ValueError("CachedHeadBatch for 'true_values' requires precomputed normalised_targets and frames")
        loc, scale, normalised = batch.frame_loc, batch.frame_scale, batch.normalised_targets
    elif horizon_patches == 1:
        loc, scale = batch.anchor_loc[:, None], batch.anchor_scale[:, None]
        normalised = apply_affine_asinh_masked(
            batch.targets,
            loc.repeat_interleave(patch, dim=-1),
            scale.repeat_interleave(patch, dim=-1),
            batch.loss_weight,
            config.use_asinh,
        )
    else:
        if batch.context_welford is None:
            raise ValueError("CachedHeadBatch for multi-patch 'median_folded' requires context_welford")
        with torch.no_grad():
            stats_device = device if supports_float64(device) else torch.device("cpu")
            cnt, mean, m2 = (t.to(stats_device) for t in batch.context_welford)
            median = quantiles[..., config.median_index].detach().to(stats_device).to(torch.float64)
            standardised = median.sinh() if config.use_asinh else median
            loc, scale = advancing_block_frames_from_state(
                cnt, mean, m2, standardised.unflatten(-1, (horizon_patches, patch)), config.minimum_scale
            )
            loc, scale = loc.to(device, dtype).clone(), scale.to(device, dtype).clone()
            loc[:, 0] = batch.anchor_loc
            scale[:, 0] = batch.anchor_scale
            normalised = apply_affine_asinh_masked(
                batch.targets,
                loc.repeat_interleave(patch, dim=-1),
                scale.repeat_interleave(patch, dim=-1),
                batch.loss_weight,
                config.use_asinh,
            )

    levels = _cached_levels(
        config.quantile_knots, quantiles.device, torch.promote_types(quantiles.dtype, torch.float32)
    )
    loss = pinball_loss(quantiles, normalised, levels, batch.loss_weight)
    return LossOutput(loss=loss, forward=forward, normalised_targets=normalised, frame_loc=loc, frame_scale=scale)


def compute_loss(
    model: TimestralModel,
    batch: TrainingBatch | CachedHeadBatch,
    frame: TargetFrame = "median_folded",
    *,
    return_hidden: bool = False,
    validate: bool = True,
) -> LossOutput:
    """The training loss of `model` on `batch`, with gradients with respect to the model's parameters.

    The target frame carries no gradient: with `median_folded` the model's own medians define the
    frame, but only as constants.
    """
    if isinstance(batch, CachedHeadBatch):
        return _compute_cached_head_loss(model, batch, frame)
    forward = training_forward(model, batch, return_hidden=return_hidden, validate=validate)
    loc, scale = target_frames(model, batch, forward, frame)
    normalised = normalise_targets(model, batch, loc, scale)
    levels = _cached_levels(
        model.config.quantile_knots,
        forward.quantiles.device,
        torch.promote_types(forward.quantiles.dtype, torch.float32),
    )
    loss = pinball_loss(forward.quantiles, normalised, levels, batch.loss_weight)
    return LossOutput(loss=loss, forward=forward, normalised_targets=normalised, frame_loc=loc, frame_scale=scale)
