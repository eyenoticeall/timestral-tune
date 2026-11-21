"""Shared helpers for the verification probes.

The probes are the evidence behind the **[V]** tags in the specification. They run the released
checkpoints through the library (and so through the vendored upstream code), and need network
access to the Hugging Face Hub the first time. Results are recorded in docs/VERIFICATION.md.
"""

from __future__ import annotations

import copy
import dataclasses

import torch
from torch import Tensor

from timestral_tune._vendor.timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET
from timestral_tune.data import TrainingBatch
from timestral_tune.models import TuneTimestralForecaster, TuneTimestralModel, load_forecaster
from timestral_tune.trainer import compute_loss

SIZES = ("5M", "28M", "123M", "209M")
PATCH = 32


def load(size: str, device: str = "cpu", **forecaster_kwargs) -> TuneTimestralForecaster:
    """Load a released checkpoint at its pinned revision."""
    return load_forecaster(f"mistralai/Timestral-{size}", device=device, **forecaster_kwargs)


def rel(a: Tensor, b: Tensor) -> float:
    """Relative L2 difference of `a` from the reference `b`, computed in float64 on CPU."""
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def make_batch(
    series: int, rows_per_series: int, context_patches: int, horizon_patches: int, covariates: bool, seed: int
) -> TrainingBatch:
    """A synthetic batch on CPU: random-walk rows around a level of 100, about 2% of context values missing.

    With `covariates`, each series has one target, `rows_per_series - 2` past-only covariates and one
    known-future covariate; otherwise every row is a target.
    """
    g = torch.Generator().manual_seed(seed)
    rows = series * rows_per_series
    context = torch.randn(rows, context_patches * PATCH, generator=g).cumsum(-1) * 3 + 100
    mask = torch.rand(rows, context_patches * PATCH, generator=g) > 0.02
    future = context[:, -1:] + torch.randn(rows, horizon_patches * PATCH, generator=g).cumsum(-1) * 3
    role_ids = known_future = known_mask = None
    target_rows = torch.ones(rows, dtype=torch.bool)
    if covariates:
        role_ids = torch.tensor(([ROLE_TARGET] + [ROLE_PAST_COV] * (rows_per_series - 2) + [ROLE_KNOWN_COV]) * series)
        target_rows = role_ids == ROLE_TARGET
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


def loss_gradient(model: TuneTimestralModel, batch: TrainingBatch) -> tuple[float, Tensor]:
    """The training loss (spec §4) and its gradient over all parameters, flattened to float64 on CPU."""
    model.zero_grad()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    loss = compute_loss(model, batch).loss
    loss.backward()
    gradient = torch.cat([p.grad.detach().cpu().double().flatten() for p in model.parameters()])
    model.zero_grad()
    return loss.item(), gradient


def reference_gradient(model: TuneTimestralModel, batch: TrainingBatch) -> tuple[float, Tensor]:
    """`loss_gradient` of a float64-weights copy of `model` on CPU.

    Upstream's RMSNorm and RoPE still compute in float32 internally, so this is not a pure float64
    computation; it is the most accurate reference available without editing vendored code.
    """
    reference = copy.deepcopy(model).to("cpu").double()
    return loss_gradient(reference, as_double(batch.to("cpu")))


def on_device(forecaster: TuneTimestralForecaster, device: str) -> TuneTimestralForecaster:
    """A copy of `forecaster` on another device."""
    model = copy.deepcopy(forecaster.model).to(device)
    model.transformer.rope.rebuild_inv_freq()
    return TuneTimestralForecaster(
        model, forecaster.max_context, forecaster.per_shot_horizon_patches, forecaster.quantile_real_cap_k
    )
