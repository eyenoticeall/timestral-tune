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
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    enc_causal_mask,
    group_first_observed_patch,
)
from .._vendor.timestral.model import ROLE_KNOWN_COV, ROLE_TARGET, TimestralModel
from .._vendor.timestral.scaler import (
    advancing_block_frames,
    apply_affine_asinh_masked,
    causal_patch_loc_scale,
    cumulative_welford,
    denormalize_quantiles,
    welford_combine,
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


def _build_covariate_time_mask(
    context_patches: int,
    horizon_patches: int,
    key_valid: Tensor,
    known_rows: Tensor,
    *,
    has_known: bool,
) -> Tensor:
    """`(R, 1, wp + hp, wp + hp)` trimmed time mask for single-horizon covariate training.

    Mathematically identical to upstream's `build_two_stream_time_mask` restricted to the `wp`
    encoder tokens and the `hp` active horizon decoder tokens: the `cp` inactive context-position
    decoder tokens in upstream's `2 * wp` layout attend only to themselves, are never read by any
    other token, and are discarded before the head.
    """
    rows, window_patches = key_valid.shape
    device = key_valid.device
    enc_enc = enc_causal_mask(key_valid)
    if has_known:
        eye = torch.eye(window_patches, dtype=torch.bool, device=device)
        enc_bi = key_valid[:, None, :].expand(rows, window_patches, window_patches) | eye[None]
        enc_enc = torch.where(known_rows[:, None, None], enc_bi, enc_enc)
    enc_dec = torch.zeros(rows, window_patches, horizon_patches, dtype=torch.bool, device=device)
    ctx_valid = torch.cat(
        [key_valid[:, :context_patches], torch.zeros(rows, horizon_patches, dtype=torch.bool, device=device)], dim=1
    )
    dec_enc = ctx_valid[:, None, :].expand(rows, horizon_patches, window_patches)
    dec_dec = torch.tril(torch.ones(horizon_patches, horizon_patches, dtype=torch.bool, device=device))[None].expand(
        rows, horizon_patches, horizon_patches
    )
    top = torch.cat([enc_enc, enc_dec], dim=2)
    bottom = torch.cat([dec_enc, dec_dec], dim=2)
    return torch.cat([top, bottom], dim=1).unsqueeze(1)


def _build_covariate_positions(context_patches: int, horizon_patches: int, first_obs: Tensor) -> Tensor:
    """`(R, wp + hp)` positions matching upstream's `build_two_stream_positions` on the kept tokens."""
    window_patches = context_patches + horizon_patches
    p_enc = torch.arange(window_patches, device=first_obs.device)[None, :] - first_obs[:, None]
    p_dec = torch.arange(context_patches, window_patches, device=first_obs.device)[None, :] - first_obs[:, None]
    return torch.cat([p_enc, p_dec], dim=-1)


def training_forward(
    model: TimestralModel,
    batch: TrainingBatch,
    *,
    return_hidden: bool = False,
    validate: bool = True,
) -> ForwardOutput:
    """Normalised-space quantiles for the horizon of `batch`, with gradients.

    Target values over the horizon (`batch.targets`) are never read here: they are labels.

    Args:
        model: The network. Use a `TuneTimestralModel` on devices without float64.
        batch: Inputs in the packed row layout, on the model's device.
        return_hidden: Also return the final transformer states.
        validate: Run `batch.validate(patch_size)`. Set to False only when the caller has already
            validated the batch (for example on CPU before moving it to an accelerator).

    Raises:
        ValueError: If the batch breaks the layout contract, the horizon exceeds one decode block,
            or the batch has known-future covariates the checkpoint cannot use.
    """
    config = model.config
    patch = config.patch_size
    if validate:
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
        has_known = batch.known_future is not None
        if not config.supports_known_covariates and bool(known_rows.any()):
            raise ValueError("this checkpoint cannot condition on known-future covariates")
        if not has_known:
            known_future = torch.zeros(rows, batch.horizon_steps, device=device, dtype=batch.context.dtype)
            known_future_mask = torch.zeros(rows, batch.horizon_steps, device=device, dtype=torch.bool)
        else:
            known_future, known_future_mask = batch.known_future, batch.known_future_mask
        # Target and past-only rows hold a blank, unobserved horizon; known-future rows hold their values.
        window = torch.cat([batch.context, known_future], dim=-1)
        window_mask = torch.cat([batch.context_mask, known_future_mask], dim=-1)
        window_patches = context_patches + horizon_patches

        scaled, loc, scale = model.scale(window, window_mask)
        if has_known:
            scaled, loc, scale = model._known_cov_scaling(window, window_mask, scaled, loc, scale, known_rows)
        x_enc = model.representation(scaled, window_mask)
        # Only the `horizon_patches` active decoder tokens are needed; the `context_patches` inactive
        # decoder slots of upstream's `2 * window_patches` layout are decoupled and discarded.
        x_dec = model.representation.forecast_tokens(rows, horizon_patches)
        obs_per_patch = window_mask.unflatten(-1, (window_patches, patch)).any(dim=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, config.attend_missing_patches)
        time_mask = _build_covariate_time_mask(
            context_patches, horizon_patches, key_valid, known_rows, has_known=has_known
        )
        variate_mask = build_two_stream_variate_mask(
            batch.group_ids, key_valid, horizon_patches, target_rows=target_rows, bidirectional_rows=known_rows
        )
        positions = _build_covariate_positions(
            context_patches, horizon_patches, _first_observed_patch(obs_per_patch, batch.group_ids)
        )
        hidden = _run_transformer(model, x_enc, x_dec, time_mask, variate_mask, positions)
        quantiles = model.head(hidden[:, window_patches:, :])
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
def context_welford_state(context: Tensor, observation_mask: Tensor, patch_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """Float64 Welford summary `(cnt, mean, m2)` of each row's observed context, matching `advancing_block_frames`.

    Each tensor has shape `(R,)` in float64 on `context.device`. Computing this once lets cached
    `head_only` steps evaluate `advancing_block_frames_from_state` without retaining or rescanning
    the context.
    """
    values = context.unflatten(-1, (-1, patch_size)).to(torch.float64)
    weights = observation_mask.unflatten(-1, (-1, patch_size)).to(torch.float64)
    cnt = weights.sum(-1)
    mean = (values * weights).sum(-1) / cnt.clamp(min=1.0)
    m2 = (weights * (values - mean.unsqueeze(-1)).square()).sum(-1)
    is_zero = cnt == 0
    mean = torch.where(is_zero, torch.zeros_like(mean), mean)
    m2 = torch.where(is_zero, torch.zeros_like(m2), m2)
    cnt, mean, m2 = (s[:, -1] for s in cumulative_welford(cnt, mean, m2))
    return cnt, mean, m2


@torch.no_grad()
def advancing_block_frames_from_state(
    cnt: Tensor,
    mean: Tensor,
    m2: Tensor,
    sinh_median: Tensor,
    minimum_scale: float,
) -> tuple[Tensor, Tensor]:
    """Per-patch `(loc, scale)` in the advancing refine frame from a precomputed context Welford state.

    Matches upstream's `advancing_block_frames` to float64 round-off without rescanning the context.
    """
    _, n_pred, ps = sinh_median.shape
    offset = mean
    clip = torch.finfo(torch.float32).max ** 0.5 / 1e3
    locs, scales = [], []
    for h in range(n_pred):
        var = (m2 / (cnt - 1.0).clamp(min=1.0)).clamp(min=0.0)
        scale = var.sqrt().clamp(min=minimum_scale)
        locs.append(mean)
        scales.append(scale)
        folded = (sinh_median[:, h] * scale[:, None] + mean[:, None] - offset[:, None]).clamp(-clip, clip)
        folded = folded + offset[:, None]
        b_mean = folded.mean(-1)
        b_m2 = (folded - b_mean[:, None]).square().sum(-1)
        cnt, mean, m2 = welford_combine(cnt, mean, m2, torch.full_like(cnt, ps), b_mean, b_m2)
    return torch.stack(locs, dim=1), torch.stack(scales, dim=1)


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
    dtype = batch.context.dtype
    anchor_loc = output.anchor_loc.detach().to(dtype)
    anchor_scale = output.anchor_scale.detach().to(dtype)

    # Patch 0 is framed by the context alone under both definitions, so a single-patch horizon needs no scan.
    if horizon_patches == 1:
        return anchor_loc[:, None], anchor_scale[:, None]

    stats_device = device if supports_float64(device) else torch.device("cpu")

    def compute_for(context: Tensor, context_mask: Tensor, quantiles: Tensor, targets: Tensor, weight: Tensor):
        ctx = context.to(stats_device)
        ctx_mask = context_mask.to(stats_device)
        if frame == "median_folded":
            # Move first, then widen: converting to float64 on the source device fails where it has none.
            median = quantiles[..., config.median_index].detach().to(stats_device).to(torch.float64)
            standardised = median.sinh() if config.use_asinh else median
            return advancing_block_frames(
                ctx, ctx_mask, standardised.unflatten(-1, (horizon_patches, patch)), config.minimum_scale
            )
        series = torch.cat([ctx, targets.to(stats_device)], dim=-1)
        observed = torch.cat([ctx_mask, weight.to(stats_device)], dim=-1)
        loc_steps, scale_steps = causal_patch_loc_scale(
            series.unsqueeze(1), observed.unsqueeze(1), patch, config.minimum_scale, config.stabilize_with_global
        )
        context_patches = batch.context_steps // patch
        frame_patches = slice(context_patches - 1, context_patches - 1 + horizon_patches)
        return (
            loc_steps.squeeze(1).unflatten(-1, (-1, patch))[..., 0][:, frame_patches],
            scale_steps.squeeze(1).unflatten(-1, (-1, patch))[..., 0][:, frame_patches],
        )

    if batch.role_ids is None:
        loc, scale = compute_for(batch.context, batch.context_mask, output.quantiles, batch.targets, batch.loss_weight)
        loc, scale = loc.to(device, dtype).clone(), scale.to(device, dtype).clone()
        loc[:, 0] = anchor_loc
        scale[:, 0] = anchor_scale
        return loc, scale

    # Covariate rows are never scored; compute frames on target rows only and fill covariate rows with their anchor.
    loc = anchor_loc[:, None].expand(batch.num_rows, horizon_patches).clone()
    scale = anchor_scale[:, None].expand(batch.num_rows, horizon_patches).clone()
    target_rows = batch.role_ids == ROLE_TARGET
    if bool(target_rows.any()):
        t_loc, t_scale = compute_for(
            batch.context[target_rows],
            batch.context_mask[target_rows],
            output.quantiles[target_rows],
            batch.targets[target_rows],
            batch.loss_weight[target_rows],
        )
        loc[target_rows, 1:] = t_loc[:, 1:].to(device, dtype)
        scale[target_rows, 1:] = t_scale[:, 1:].to(device, dtype)
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
