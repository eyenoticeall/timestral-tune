"""Shared helpers for the verification probes.

The probes are the evidence behind every **[V]** tag in the handoff document. They run the
released checkpoints through the vendored upstream code, so they need network access to the
Hugging Face Hub the first time. Results are recorded in docs/VERIFICATION.md.

Two things here are prototypes of library code that does not exist yet:

- `install_cpu_offload` monkeypatches the vendored modules so that the float64 statistics run on
  CPU. The library version (`timestral_tune/models/device.py`) will use subclasses instead.
- `norm_forward` and `real_loss` are a working sketch of the training forward and loss of
  handoff §4. The library version is `timestral_tune/models/forward.py`.
"""

from __future__ import annotations

import torch
from torch import Tensor

import timestral_tune._vendor.timestral.forecaster as forecaster_module
import timestral_tune._vendor.timestral.scaler as scaler
from timestral_tune._vendor.timestral import TimestralForecaster
from timestral_tune._vendor.timestral.masks import (
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_positions,
    build_two_stream_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    group_first_observed_patch,
)
from timestral_tune._vendor.timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET, TimestralModel

# Hub revisions pinned in the handoff document.
REVISIONS = {
    "5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    "28M": "49b26409ac87b5935539a5d8ffe7f397b720db47",
    "123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
    "209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
}
PATCH = 32

_offload_installed = False


def load(size: str, device: str = "cpu") -> TimestralForecaster:
    """Load a released checkpoint at its pinned revision through the vendored loader."""
    return TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device=device, revision=REVISIONS[size])


def install_cpu_offload() -> None:
    """Run upstream's float64 statistics on CPU when tensors live on a device without float64 (MPS)."""
    global _offload_installed
    if _offload_installed:
        return
    _offload_installed = True

    original_stats = scaler.compute_causal_statistics

    def stats_on_cpu(data, observation_mask, *args, **kwargs):
        if data.device.type != "mps":
            return original_stats(data, observation_mask, *args, **kwargs)
        means, scales = original_stats(data.cpu(), observation_mask.cpu(), *args, **kwargs)
        return means.to(data.device), scales.to(data.device)

    scaler.compute_causal_statistics = stats_on_cpu

    original_refine = forecaster_module.TimestralForecaster._refine_block

    def refine_on_cpu(self, quantiles, context, mask, target_rows):
        if quantiles.device.type != "mps":
            return original_refine(self, quantiles, context, mask, target_rows)
        rows = None if target_rows is None else target_rows.cpu()
        return original_refine(self, quantiles.cpu(), context.cpu(), mask.cpu(), rows).to(quantiles.device)

    forecaster_module.TimestralForecaster._refine_block = refine_on_cpu

    original_encode = forecaster_module.TimestralForecaster._encode_categorical

    def encode_on_cpu(self, codes, target_context):
        if target_context is not None and target_context.device.type == "mps":
            target_context = target_context.cpu()
        return original_encode(self, codes, target_context)

    forecaster_module.TimestralForecaster._encode_categorical = encode_on_cpu


def rel(a: Tensor, b: Tensor) -> float:
    """Relative L2 difference of `a` from the reference `b`, computed in float64 on CPU."""
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def flat_grads(model: TimestralModel) -> Tensor:
    return torch.cat([p.grad.detach().cpu().double().flatten() for p in model.parameters()])


def make_batch(
    series: int, rows_per_series: int, context_patches: int, horizon_patches: int, covariates: bool, seed: int
):
    """Synthetic batch `(x, obs, group_ids, y, roles, known_future, known_future_mask)` on CPU.

    With `covariates`, each series has one target, `rows_per_series - 2` past-only covariates and one
    known-future covariate. About 2% of context values are marked missing.
    """
    g = torch.Generator().manual_seed(seed)
    rows = series * rows_per_series
    x = torch.randn(rows, context_patches * PATCH, generator=g).cumsum(-1) * 3 + 100
    obs = torch.rand(rows, context_patches * PATCH, generator=g) > 0.02
    group_ids = torch.arange(series).repeat_interleave(rows_per_series)
    y = x[:, -1:] + torch.randn(rows, horizon_patches * PATCH, generator=g).cumsum(-1) * 3
    if not covariates:
        return x, obs, group_ids, y, None, None, None
    roles = torch.tensor(([ROLE_TARGET] + [ROLE_PAST_COV] * (rows_per_series - 2) + [ROLE_KNOWN_COV]) * series)
    known = (roles == ROLE_KNOWN_COV)[:, None]
    known_future = torch.randn(rows, horizon_patches * PATCH, generator=g) * known
    known_future_mask = known.expand(rows, horizon_patches * PATCH).clone()
    return x, obs, group_ids, y, roles, known_future, known_future_mask


def norm_forward(
    model: TimestralModel, x, obs, group_ids, horizon_patches, roles=None, known_future=None, known_mask=None
):
    """Normalised-space quantiles at the horizon prediction tokens, for both upstream layouts.

    Mirrors `_predict_context_only` and `_predict_window` of the vendored model, stopping before
    denormalisation. Returns `(quantiles (R, H*P, Q), anchor_loc (R,), anchor_scale (R,))`.
    """
    config = model.config
    rows, device = x.shape[0], x.device
    h = horizon_patches
    if roles is None:
        ll = x.shape[1] // PATCH
        scaled, loc, scale = model.scale(x, obs)
        x_enc = model.representation(scaled, obs)
        x_dec = model.representation.forecast_tokens(rows, h)
        obs_per_patch = obs.unflatten(-1, (ll, PATCH)).any(-1)
        key_valid = compute_enc_key_validity(obs_per_patch, config.attend_missing_patches)
        time_mask = build_forecast_time_mask(ll, h, key_valid, device)
        variate_mask = build_two_stream_variate_mask(group_ids, key_valid, h)
        unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
        first_obs = group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]
        positions = build_forecast_positions(ll, h, first_obs)
        preds = model._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
    else:
        full_x = torch.cat([x, known_future], -1)
        full_obs = torch.cat([obs, known_mask], -1)
        n = full_x.shape[1] // PATCH
        ll = n - h
        target_rows, known_rows = roles == ROLE_TARGET, roles == ROLE_KNOWN_COV
        scaled, loc, scale = model.scale(full_x, full_obs)
        scaled, loc, scale = model._known_cov_scaling(full_x, full_obs, scaled, loc, scale, known_rows)
        x_enc = model.representation(scaled, full_obs)
        x_dec = model.representation.forecast_tokens(rows, n)
        obs_per_patch = full_obs.unflatten(-1, (n, PATCH)).any(-1)
        key_valid = compute_enc_key_validity(obs_per_patch, config.attend_missing_patches)
        start = torch.full((rows,), ll, device=device, dtype=torch.long)
        time_mask = build_two_stream_time_mask(start, n, config.block_size, key_valid, bidirectional_rows=known_rows)
        variate_mask = build_two_stream_variate_mask(
            group_ids, key_valid, n, target_rows=target_rows, bidirectional_rows=known_rows
        )
        unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
        first_obs = group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]
        positions = build_two_stream_positions(n, first_obs)
        preds = model._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
        preds = preds.unflatten(1, (n, PATCH))[:, ll:n].reshape(rows, h * PATCH, -1)
    anchor_loc = loc.unflatten(-1, (-1, PATCH))[..., 0][:, ll - 1]
    anchor_scale = scale.unflatten(-1, (-1, PATCH))[..., 0][:, ll - 1]
    return preds, anchor_loc, anchor_scale


def real_loss(model: TimestralModel, batch, device: str, double: bool = False) -> Tensor:
    """The training step of handoff §4: normalised forward, median-folded frame, pinball on targets.

    The frame statistics are computed on CPU in float64 with gradients stopped, as the spec requires.
    With `double`, inputs are float64 (for the float64-weights reference; the model must be `.double()`).
    """
    x, obs, group_ids, y, roles, known_future, known_mask = batch
    if double:
        x, y = x.double(), y.double()
        known_future = None if known_future is None else known_future.double()

    def to_device(t):
        return None if t is None else t.to(device)

    h = y.shape[1] // PATCH
    preds, anchor_loc, anchor_scale = norm_forward(
        model, to_device(x), to_device(obs), to_device(group_ids), h, to_device(roles), to_device(known_future),
        to_device(known_mask),
    )  # fmt: skip
    with torch.no_grad():
        median = preds[..., model.config.median_index].detach().cpu().double()
        sinh_median = median.sinh().unflatten(-1, (h, PATCH))
        loc_j, scale_j = scaler.advancing_block_frames(x, obs, sinh_median, model.config.minimum_scale)
        loc_j, scale_j = loc_j.to(y.dtype), scale_j.to(y.dtype)
        loc_j[:, 0], scale_j[:, 0] = anchor_loc.cpu().to(y.dtype), anchor_scale.cpu().to(y.dtype)
        z = torch.asinh((y - loc_j.repeat_interleave(PATCH, -1)) / scale_j.repeat_interleave(PATCH, -1)).to(device)
    target_rows = torch.ones(x.shape[0], dtype=torch.bool) if roles is None else roles == ROLE_TARGET
    target_rows = target_rows.to(device)
    levels = torch.tensor(model.config.quantile_knots, device=device, dtype=preds.dtype)
    error = z[target_rows][..., None].to(preds.dtype) - preds[target_rows]
    return torch.maximum(levels * error, (levels - 1) * error).mean()
