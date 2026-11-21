"""Device shims: run upstream's float64 statistics on CPU.

The vendored code computes its normalisation statistics in float64 (spec §3.5). The Apple GPU
backend (MPS) has no float64, so those code paths raise there. The statistics carry no gradients
and are cheap, so the fix is to compute them on CPU and move the float32 results to the device.

This module does that with two subclasses that override exactly the methods that reach a float64
site, move their tensor arguments to CPU, call the parent implementation, and move the result
back. Nothing in the vendored package is edited or monkeypatched.

- `TuneTimestralModel.scale` (reached from both forecasting layouts)
- `TuneTimestralForecaster._cap_quantiles`, `._refine_block` and `._encode_categorical`

On devices that support float64 the overrides call the parent directly, so behaviour there is
exactly upstream's.

- Spec: `docs/SPEC.md` §3.5, §5.4
- Acceptance tests: T14
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from .._vendor.timestral.forecaster import TimestralForecaster
from .._vendor.timestral.model import TimestralModel

__all__ = [
    "MIN_TORCH_FOR_MPS",
    "TuneTimestralForecaster",
    "TuneTimestralModel",
    "mps_is_usable",
    "supports_float64",
]

_CPU = torch.device("cpu")

# Upstream's input assembly uses `cummax`, which the Apple GPU backend implements from PyTorch 2.8.
# Measured: the operator raises NotImplementedError on 2.4.0, 2.5.1, 2.6.0 and 2.7.1 and works on
# 2.8.0, where the whole test suite passes on the Apple GPU (docs/VERIFICATION.md).
MIN_TORCH_FOR_MPS = (2, 8)


def _torch_version() -> tuple[int, int]:
    major, minor = torch.__version__.split(".")[:2]
    return int(major), int(minor)


def mps_is_usable() -> bool:
    """Whether the Apple GPU backend is present and this PyTorch is new enough to run Timestral on it."""
    return torch.backends.mps.is_available() and _torch_version() >= MIN_TORCH_FOR_MPS


def supports_float64(device: torch.device | str) -> bool:
    """Whether tensors on `device` can be float64. False for the Apple GPU backend."""
    return torch.device(device).type != "mps"


def _to_cpu(tensor: Tensor | None) -> Tensor | None:
    return None if tensor is None else tensor.to(_CPU)


class TuneTimestralModel(TimestralModel):
    """The upstream network, with normalisation statistics computed on CPU where float64 is missing.

    Parameter names and the forward computation are unchanged, so released weights load with
    `strict=True` and weights saved from this class load in the official package.
    """

    @classmethod
    def from_upstream(cls, model: TimestralModel) -> TuneTimestralModel:
        """Build a `TuneTimestralModel` holding a copy of `model`'s weights, on the same device and dtype."""
        if isinstance(model, cls):
            return model
        parameter = next(model.parameters())
        # Cast before loading: loading float64 weights into float32 parameters would round them.
        tuned = cls(model.config).to(device=parameter.device, dtype=parameter.dtype)
        tuned.load_state_dict(model.state_dict(), strict=True)
        tuned.transformer.rope.rebuild_inv_freq()
        return tuned.train(model.training)

    def scale(self, inputs: Tensor, observed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Causal per-patch normalisation; see the parent. Statistics run on CPU when the device needs it."""
        device = inputs.device
        if supports_float64(device):
            return super().scale(inputs, observed)
        scaled, loc, scale = super().scale(inputs.to(_CPU), observed.to(_CPU))
        return scaled.to(device), loc.to(device), scale.to(device)


class TuneTimestralForecaster(TimestralForecaster):
    """The upstream forecaster, usable on devices without float64.

    Build it with `timestral_tune.models.load_forecaster`, or pass a `TuneTimestralModel`.
    """

    def __init__(self, model: TimestralModel, *args, **kwargs) -> None:
        if not isinstance(model, TuneTimestralModel):
            raise TypeError(
                "TuneTimestralForecaster needs a TuneTimestralModel; load one with "
                "timestral_tune.models.load_model or convert with TuneTimestralModel.from_upstream"
            )
        super().__init__(model, *args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        device: str | torch.device = "cpu",
        max_context: int | None = None,
        per_shot_horizon_patches: int = 32,
        quantile_real_cap_k: float = 1e4,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> TuneTimestralForecaster:
        """Same arguments as upstream's `from_pretrained`; see `timestral_tune.models.load_forecaster`."""
        from .loader import load_forecaster  # local import: loader imports this module

        return load_forecaster(
            path_or_repo_id,
            device=device,
            max_context=max_context,
            per_shot_horizon_patches=per_shot_horizon_patches,
            quantile_real_cap_k=quantile_real_cap_k,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )

    def _cap_quantiles(self, quantiles: Tensor, inputs: Tensor, observation_mask: Tensor) -> Tensor:
        device = quantiles.device
        if supports_float64(device):
            return super()._cap_quantiles(quantiles, inputs, observation_mask)
        return super()._cap_quantiles(quantiles.to(_CPU), inputs.to(_CPU), observation_mask.to(_CPU)).to(device)

    def _refine_block(self, quantiles: Tensor, context: Tensor, mask: Tensor, target_rows: Tensor | None) -> Tensor:
        device = quantiles.device
        if supports_float64(device):
            return super()._refine_block(quantiles, context, mask, target_rows)
        refined = super()._refine_block(quantiles.to(_CPU), context.to(_CPU), mask.to(_CPU), _to_cpu(target_rows))
        return refined.to(device)

    def _encode_categorical(self, codes: Tensor | None, target_context: Tensor | None) -> Tensor | None:
        # The parent converts `target_context` to float64 before moving it to CPU; do the move first.
        return super()._encode_categorical(codes, _to_cpu(target_context))
