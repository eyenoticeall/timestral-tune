"""Training forward pass in normalised space, and the target frames the loss is computed in.

Upstream only exposes `predict_block`, which returns raw-space quantiles after `sinh`
denormalisation. Training needs the normalised head output (spec §3.5, §4.3). `training_forward`
therefore runs the same building blocks as upstream's two inference layouts and stops before
denormalisation:

- no covariates: `L` context tokens followed by `h` prediction tokens
  (upstream `TimestralModel._predict_context_only`);
- with covariates: a window of `n = L + h` patches, `n` context tokens and `n` prediction tokens,
  with only the horizon prediction tokens read out (upstream `TimestralModel._predict_window`).

Everything that shapes information flow (normalisation, embeddings, masks, positions, the
transformer, the head) is upstream's code, called from here. Acceptance tests T1 and T2 assert
that `sinh(output) * anchor_scale + anchor_loc` equals `predict_block`.

The loss compares predictions with targets normalised in a per-patch *frame* `(loc_j, scale_j)`:

- `median_folded` (default): patch 0 uses the context statistics; patch `j >= 1` uses the context
  plus the model's own predicted medians of patches `0..j-1`, with gradients stopped. This is the
  frame upstream's forecaster denormalises in, so training and deployment match (decision D2).
- `true_values`: patch `j` uses the context plus the true values of patches `0..j-1`.

- Spec: `docs/SPEC.md` §3.4, §4.3, §5.4
- Acceptance tests: T1, T2, T3, T4, T6, T7
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, get_args

import torch
from torch import Tensor

from .._vendor.timestral.layers import TwoStreamMask
from .._vendor.timestral.masks import (
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_positions,
    build_two_stream_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    group_first_observed_patch,
)
from .._vendor.timestral.model import ROLE_KNOWN_COV, ROLE_TARGET, TimestralModel
from .._vendor.timestral.scaler import (
    advancing_block_frames,
    apply_affine_asinh_masked,
    causal_patch_loc_scale,
    denormalize_quantiles,
)
from ..data.batch import TrainingBatch
from .device import supports_float64

__all__ = [
    "AttentionBackend",
    "ForwardOutput",
    "TargetFrame",
    "attention_context",
    "denormalise_quantiles",
    "normalise_targets",
    "target_frames",
    "training_forward",
]

AttentionBackend = Literal["default", "math"]
ATTENTION_BACKENDS: tuple[str, ...] = get_args(AttentionBackend)
TargetFrame = Literal["median_folded", "true_values"]
TARGET_FRAMES: tuple[str, ...] = get_args(TargetFrame)

_SDPA_BACKENDS = ("math", "flash", "mem_efficient", "cudnn")
_MATH_SDPA_LOCK = threading.Lock()
_MATH_SDPA_DEPTH = 0
_MATH_SDPA_SAVED: dict[str, bool] | None = None


def _snapshot_sdpa_flags() -> dict[str, bool]:
    return {name: getattr(torch.backends.cuda, f"{name}_sdp_enabled")() for name in _SDPA_BACKENDS}


def _apply_sdpa_flags(flags: dict[str, bool]) -> None:
    for name, enabled in flags.items():
        getattr(torch.backends.cuda, f"enable_{name}_sdp")(enabled)


@contextlib.contextmanager
def attention_context(backend: AttentionBackend = "default") -> Iterator[None]:
    """Experimental: select the PyTorch attention kernel for calls inside the block.

    This is a measurement tool, used by acceptance test T14 and the gradient-accuracy probe. It is
    not part of the public API and no training code path uses it; whether training should is a
    Phase 2 question (spec §11, D15).

    `"default"` leaves PyTorch's dispatch alone, matching upstream inference. `"math"` disables the
    fused scaled-dot-product-attention kernels (flash, memory-efficient, cuDNN) so PyTorch runs its
    unfused one. On typical CPU batches that avoids the float32 backward error of the fused kernel,
    at the cost of materialising the O(S^2) attention matrix. It can still be less accurate than
    `"default"` on individual batches with near-tied large logits (spec §12), and is identical to
    `"default"` on the Apple GPU. On CUDA it lowers the error 50 to 900 times for roughly 5% to 25%
    more step time at a context of 1,024 (docs/VERIFICATION.md §13).

    PyTorch's flags are process-global, so other threads also see math-only attention while any
    `"math"` block is active. Overlapping or nested blocks are reference-counted, so the original
    flags are restored when the last one exits.
    """
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(f"unknown attention backend {backend!r}; expected one of {ATTENTION_BACKENDS}")
    if backend != "math":
        yield
        return

    global _MATH_SDPA_DEPTH, _MATH_SDPA_SAVED
    with _MATH_SDPA_LOCK:
        if _MATH_SDPA_DEPTH == 0:
            _MATH_SDPA_SAVED = _snapshot_sdpa_flags()
            _apply_sdpa_flags({name: name == "math" for name in _SDPA_BACKENDS})
        _MATH_SDPA_DEPTH += 1
    try:
        yield
    finally:
        with _MATH_SDPA_LOCK:
            _MATH_SDPA_DEPTH -= 1
            if _MATH_SDPA_DEPTH == 0 and _MATH_SDPA_SAVED is not None:
                saved = _MATH_SDPA_SAVED
                _MATH_SDPA_SAVED = None
                _apply_sdpa_flags(saved)


@dataclass(frozen=True)
class ForwardOutput:
    """Result of `training_forward`.

    Attributes:
        quantiles: `(R, H, Q)` normalised-space quantiles at the horizon prediction tokens of every
            row, non-decreasing along `Q`, in at least float32. Only target rows are meaningful
            forecasts; covariate rows are returned so that leakage tests can inspect them.
        anchor_loc: `(R,)` location of the last context patch: the frame of horizon patch 0.
        anchor_scale: `(R,)` scale of the last context patch.
        hidden: `(R, S, D)` final transformer states over the whole `[context ; prediction]` token
            sequence, or None unless requested. For leakage tests.
        num_context_tokens: Number of leading context tokens in `hidden`.
    """

    quantiles: Tensor
    anchor_loc: Tensor
    anchor_scale: Tensor
    hidden: Tensor | None
    num_context_tokens: int


def _run_transformer(
    model: TimestralModel,
    x_enc: Tensor,
    x_dec: Tensor,
    time_mask: Tensor,
    variate_mask: Tensor | None,
    positions: Tensor,
) -> Tensor:
    """Transformer over `[context ; prediction]` tokens; mirrors upstream `_run_two_stream` without the head."""
    x = torch.cat([x_enc, x_dec], dim=1)
    mask: Tensor | TwoStreamMask = time_mask
    if model.config.split_context_horizon_attn:
        mask = TwoStreamMask(mask=time_mask, enc_len=x_enc.shape[1])
    return model.transformer(x, mask, variate_mask, positions)


def _first_observed_patch(obs_per_patch: Tensor, group_ids: Tensor) -> Tensor:
    unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
    return group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]


def _uses_covariate_layout(model: TimestralModel, batch: TrainingBatch) -> bool:
    """The same rule upstream's `predict_block` uses to choose between its two layouts."""
    if batch.role_ids is None or not model.config.covariate_enabled:
        return False
    return not bool((batch.role_ids == ROLE_TARGET).all())


def training_forward(model: TimestralModel, batch: TrainingBatch, *, return_hidden: bool = False) -> ForwardOutput:
    """Normalised-space quantiles for the horizon of `batch`, with gradients.

    Target values over the horizon (`batch.targets`) are never read here: they are labels.

    Args:
        model: The network. Use a `TuneTimestralModel` on devices without float64.
        batch: Inputs in the packed row layout, on the model's device.
        return_hidden: Also return the final transformer states.

    Raises:
        ValueError: If the batch breaks the layout contract, the horizon exceeds one decode block,
            or the batch has known-future covariates the checkpoint cannot use.
    """
    config = model.config
    patch = config.patch_size
    batch.validate(patch)
    horizon_patches = batch.horizon_steps // patch
    if horizon_patches > config.block_size:
        raise ValueError(
            f"training horizon of {horizon_patches} patches exceeds one decode block ({config.block_size} patches, "
            f"{config.block_size * patch} steps)"
        )
    rows = batch.num_rows
    context_patches = batch.context_steps // patch
    device = batch.device

    if not _uses_covariate_layout(model, batch):
        scaled, loc, scale = model.scale(batch.context, batch.context_mask)
        x_enc = model.representation(scaled, batch.context_mask)
        x_dec = model.representation.forecast_tokens(rows, horizon_patches)
        obs_per_patch = batch.context_mask.unflatten(-1, (context_patches, patch)).any(dim=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, config.attend_missing_patches)
        time_mask = build_forecast_time_mask(context_patches, horizon_patches, key_valid, device)
        variate_mask = build_two_stream_variate_mask(batch.group_ids, key_valid, horizon_patches)
        positions = build_forecast_positions(
            context_patches, horizon_patches, _first_observed_patch(obs_per_patch, batch.group_ids)
        )
        hidden = _run_transformer(model, x_enc, x_dec, time_mask, variate_mask, positions)
        quantiles = model.head(hidden[:, context_patches:, :])
        num_context_tokens = context_patches
    else:
        role_ids = batch.role_ids
        assert role_ids is not None  # guaranteed by _uses_covariate_layout
        target_rows = role_ids == ROLE_TARGET
        known_rows = role_ids == ROLE_KNOWN_COV
        if bool(known_rows.any()) and not config.supports_known_covariates:
            raise ValueError("this checkpoint cannot condition on known-future covariates")
        if batch.known_future is None:
            known_future = torch.zeros(rows, batch.horizon_steps, device=device, dtype=batch.context.dtype)
            known_future_mask = torch.zeros(rows, batch.horizon_steps, device=device, dtype=torch.bool)
        else:
            known_future, known_future_mask = batch.known_future, batch.known_future_mask
        # Target and past-only rows hold a blank, unobserved horizon; known-future rows hold their values.
        window = torch.cat([batch.context, known_future], dim=-1)
        window_mask = torch.cat([batch.context_mask, known_future_mask], dim=-1)
        window_patches = context_patches + horizon_patches

        scaled, loc, scale = model.scale(window, window_mask)
        scaled, loc, scale = model._known_cov_scaling(window, window_mask, scaled, loc, scale, known_rows)
        x_enc = model.representation(scaled, window_mask)
        x_dec = model.representation.forecast_tokens(rows, window_patches)
        obs_per_patch = window_mask.unflatten(-1, (window_patches, patch)).any(dim=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, config.attend_missing_patches)
        block_start = torch.full((rows,), context_patches, device=device, dtype=torch.long)
        time_mask = build_two_stream_time_mask(
            block_start, window_patches, config.block_size, key_valid, bidirectional_rows=known_rows
        )
        variate_mask = build_two_stream_variate_mask(
            batch.group_ids, key_valid, window_patches, target_rows=target_rows, bidirectional_rows=known_rows
        )
        positions = build_two_stream_positions(window_patches, _first_observed_patch(obs_per_patch, batch.group_ids))
        hidden = _run_transformer(model, x_enc, x_dec, time_mask, variate_mask, positions)
        # Prediction tokens exist for every window patch; only the horizon ones are forecasts.
        quantiles = model.head(hidden[:, window_patches + context_patches :, :])
        num_context_tokens = window_patches

    per_patch_loc = loc.unflatten(-1, (-1, patch))[..., 0]
    per_patch_scale = scale.unflatten(-1, (-1, patch))[..., 0]
    return ForwardOutput(
        quantiles=quantiles,
        anchor_loc=per_patch_loc[:, context_patches - 1],
        anchor_scale=per_patch_scale[:, context_patches - 1],
        hidden=hidden if return_hidden else None,
        num_context_tokens=num_context_tokens,
    )


@torch.no_grad()
def target_frames(
    model: TimestralModel,
    batch: TrainingBatch,
    output: ForwardOutput,
    frame: TargetFrame = "median_folded",
) -> tuple[Tensor, Tensor]:
    """Per-patch normalisation frames for the horizon: `(loc, scale)`, each `(R, horizon_patches)`.

    No gradient flows through the frames. The statistics are computed in float64, on CPU when the
    batch's device has no float64, and returned in the batch's dtype on its device.

    Frames are returned for every row; only target rows are meaningful.
    """
    if frame not in TARGET_FRAMES:
        raise ValueError(f"unknown target frame {frame!r}; expected one of {TARGET_FRAMES}")
    config = model.config
    patch = config.patch_size
    horizon_patches = batch.horizon_steps // patch
    device = batch.device
    stats_device = device if supports_float64(device) else torch.device("cpu")
    context = batch.context.to(stats_device)
    context_mask = batch.context_mask.to(stats_device)

    if frame == "median_folded":
        # The block's median in the last context patch's frame with the asinh undone, as upstream's
        # `_refine_block` passes it to `advancing_block_frames`.
        # Move first, then widen: converting to float64 on the source device fails where it has none.
        median = output.quantiles[..., config.median_index].detach().to(stats_device).to(torch.float64)
        standardised = median.sinh() if config.use_asinh else median
        loc, scale = advancing_block_frames(
            context, context_mask, standardised.unflatten(-1, (horizon_patches, patch)), config.minimum_scale
        )
    else:
        # Causal statistics of the context followed by the true horizon values; patch j is framed by
        # everything observed up to the end of patch j-1.
        series = torch.cat([context, batch.targets.to(stats_device)], dim=-1)
        observed = torch.cat([context_mask, batch.loss_weight.to(stats_device)], dim=-1)
        loc_steps, scale_steps = causal_patch_loc_scale(
            series.unsqueeze(1), observed.unsqueeze(1), patch, config.minimum_scale, config.stabilize_with_global
        )
        context_patches = batch.context_steps // patch
        frame_patches = slice(context_patches - 1, context_patches - 1 + horizon_patches)
        loc = loc_steps.squeeze(1).unflatten(-1, (-1, patch))[..., 0][:, frame_patches]
        scale = scale_steps.squeeze(1).unflatten(-1, (-1, patch))[..., 0][:, frame_patches]

    dtype = batch.context.dtype
    loc, scale = loc.to(device, dtype).clone(), scale.to(device, dtype).clone()
    # Patch 0 is framed by the context alone under both definitions. Use the forward's own anchor so
    # that the frame is bit-identical to the one upstream denormalises patch 0 with.
    loc[:, 0] = output.anchor_loc.detach().to(dtype)
    scale[:, 0] = output.anchor_scale.detach().to(dtype)
    return loc, scale


def _per_step(frame_value: Tensor, patch_size: int) -> Tensor:
    """`(R, horizon_patches)` to `(R, H)`: repeat each patch's value over its time steps."""
    return frame_value.repeat_interleave(patch_size, dim=-1)


def normalise_targets(model: TimestralModel, batch: TrainingBatch, loc: Tensor, scale: Tensor) -> Tensor:
    """`batch.targets` in the model's normalised space: `asinh((y - loc_j) / scale_j)`, `(R, H)`.

    Positions that are not scored (`batch.loss_weight` False) are 0.
    """
    patch = model.config.patch_size
    return apply_affine_asinh_masked(
        batch.targets, _per_step(loc, patch), _per_step(scale, patch), batch.loss_weight, model.config.use_asinh
    )


def denormalise_quantiles(model: TimestralModel, quantiles: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    """Raw-space quantiles `sinh(q) * scale_j + loc_j`, `(R, H, Q)`, with infinities clamped as upstream does.

    `loc` and `scale` are either per-patch frames `(R, horizon_patches)` or per-row anchors `(R,)`.
    """
    patch = model.config.patch_size
    if loc.dim() == 1:
        loc_steps, scale_steps = loc[:, None, None], scale[:, None, None]
    else:
        loc_steps, scale_steps = _per_step(loc, patch)[..., None], _per_step(scale, patch)[..., None]
    return denormalize_quantiles(quantiles, loc_steps, scale_steps, model.config.use_asinh)
