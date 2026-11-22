"""Shared helpers for the verification probes.

The probes are the evidence behind the **[V]** tags in the specification. They run the released
checkpoints through the library (and so through the vendored upstream code), and need network
access to the Hugging Face Hub the first time. Results are recorded in docs/VERIFICATION.md.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses

import torch
from torch import Tensor

import timestral_tune._vendor.timestral.layers as layers
from timestral_tune._vendor.timestral import TimestralConfig
from timestral_tune._vendor.timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET
from timestral_tune.data import TrainingBatch
from timestral_tune.models import TuneTimestralForecaster, TuneTimestralModel, load_forecaster
from timestral_tune.models.forward import attention_context
from timestral_tune.trainer import compute_loss

SIZES = ("5M", "28M", "123M", "209M")
PATCH = 32


def load(size: str, device: str = "cpu", **forecaster_kwargs) -> TuneTimestralForecaster:
    """Load a released checkpoint at its pinned revision."""
    return load_forecaster(f"mistralai/Timestral-{size}", device=device, **forecaster_kwargs)


def tiny_random_forecaster(seed: int = 0) -> TuneTimestralForecaster:
    """A small randomly initialised model with the released patch size, for network-free probes."""
    torch.manual_seed(seed)
    levels = (0.01, *(round(0.05 * i, 2) for i in range(1, 20)), 0.99)
    config = TimestralConfig(
        patch_size=PATCH,
        d_model=64,
        num_heads=2,
        mlp_hidden_dim=128,
        block_schedule=("time", "time_variate", "time", "time_variate"),
        quantile_knots=levels,
        block_size=32,
        context_length=32768,
    )
    model = TuneTimestralModel(config)
    with torch.no_grad():
        model.representation.forecast_query.normal_(std=0.5)
    return TuneTimestralForecaster(model.eval())


def rel(a: Tensor, b: Tensor) -> float:
    """Relative L2 difference of `a` from the reference `b`, computed in float64 on CPU."""
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def make_batch(
    series: int,
    rows_per_series: int,
    context_patches: int,
    horizon_patches: int,
    covariates: bool,
    seed: int,
    *,
    known_future_row: bool = True,
) -> TrainingBatch:
    """A synthetic batch on CPU: random-walk rows around a level of 100, about 2% of context values missing.

    With `covariates`, each series has one target and `rows_per_series - 1` covariates: all past-only
    but the last, which is known in advance unless `known_future_row` is False. Otherwise every row is
    a target.
    """
    g = torch.Generator().manual_seed(seed)
    rows = series * rows_per_series
    context = torch.randn(rows, context_patches * PATCH, generator=g).cumsum(-1) * 3 + 100
    mask = torch.rand(rows, context_patches * PATCH, generator=g) > 0.02
    future = context[:, -1:] + torch.randn(rows, horizon_patches * PATCH, generator=g).cumsum(-1) * 3
    role_ids = known_future = known_mask = None
    target_rows = torch.ones(rows, dtype=torch.bool)
    if covariates:
        last = ROLE_KNOWN_COV if known_future_row else ROLE_PAST_COV
        role_ids = torch.tensor(([ROLE_TARGET] + [ROLE_PAST_COV] * (rows_per_series - 2) + [last]) * series)
        target_rows = role_ids == ROLE_TARGET
        if known_future_row:
            known = (role_ids == ROLE_KNOWN_COV)[:, None]
            known_future = torch.randn(rows, horizon_patches * PATCH, generator=g) * known
            known_mask = known.expand(rows, horizon_patches * PATCH).clone()
    weight = target_rows[:, None].expand(rows, horizon_patches * PATCH).clone()
    return TrainingBatch(
        context=torch.where(mask, context, torch.zeros_like(context)),
        context_mask=mask,
        group_ids=torch.arange(series).repeat_interleave(rows_per_series),
        role_ids=role_ids,
        known_future=known_future,
        known_future_mask=known_mask,
        targets=torch.where(weight, future, torch.zeros_like(future)),
        loss_weight=weight,
    )


def as_double(batch: TrainingBatch) -> TrainingBatch:
    """`batch` with float64 values, for the float64-weights reference."""
    return dataclasses.replace(
        batch,
        context=batch.context.double(),
        targets=batch.targets.double(),
        known_future=None if batch.known_future is None else batch.known_future.double(),
    )


def loss_gradient(
    model: TuneTimestralModel, batch: TrainingBatch, *, attention_backend: str = "default"
) -> tuple[float, Tensor]:
    """The training loss (spec §4) and its gradient over all parameters, flattened to float64 on CPU."""
    model.zero_grad()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    with attention_context(attention_backend):
        loss = compute_loss(model, batch).loss
    loss.backward()
    gradient = torch.cat([p.grad.detach().cpu().double().flatten() for p in model.parameters()])
    model.zero_grad()
    return loss.item(), gradient


@contextlib.contextmanager
def float64_reference():
    """While active, upstream's RMSNorm computes in its input's own dtype instead of always in float32.

    Upstream's RMSNorm casts to float32 internally, even in a float64 model, so its backward pass is only
    float32-accurate. With the released weights that alone moves a "float64" gradient by up to 4e-3, so
    reference computations run inside this context. The formula is upstream's, unchanged.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale

    original = layers.RMSNorm.forward
    layers.RMSNorm.forward = forward
    try:
        yield
    finally:
        layers.RMSNorm.forward = original


def reference_gradient(
    model: TuneTimestralModel, batch: TrainingBatch, *, float64_norm: bool = True
) -> tuple[float, Tensor]:
    """`loss_gradient` of a float64-weights copy of `model` on CPU, with float64 RMSNorm.

    RoPE still builds its rotation table in float32, so this is not a pure float64 computation; it is the
    most accurate reference available without editing vendored code. `float64_norm=False` gives the
    reference used before 2026-11-21, with upstream's float32 RMSNorm, for comparison.
    """
    reference = copy.deepcopy(model).to("cpu").double()
    if not float64_norm:
        return loss_gradient(reference, as_double(batch.to("cpu")))
    with float64_reference():
        return loss_gradient(reference, as_double(batch.to("cpu")))


def on_device(forecaster: TuneTimestralForecaster, device: str) -> TuneTimestralForecaster:
    """A copy of `forecaster` on another device."""
    model = copy.deepcopy(forecaster.model).to(device)
    model.transformer.rope.rebuild_inv_freq()
    return TuneTimestralForecaster(
        model, forecaster.max_context, forecaster.per_shot_horizon_patches, forecaster.quantile_real_cap_k
    )
