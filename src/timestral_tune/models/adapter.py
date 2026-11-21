"""Slice LoRA for Timestral's fused attention projections.

Timestral's attention modules live at `transformer.layers.{i}.attns.{j}` and share one class,
told apart by `is_time: bool`, with three bias-free linears: `wQKV` (`D -> 3D`), `wO` (`D -> D`),
and optional `_gate` (`D -> D`). In `wQKV.weight`, rows `[0, D)` are query, `[D, 2D)` are key,
and `[2D, 3D)` are value (spec §3.2).

`LoRASliceLinear` wraps a linear layer and applies low-rank updates `(alpha / r) * B @ A` to one
or more contiguous output row slices, using one fused input projection across all adapted slices
of the wrapped linear. `A` is Gaussian-initialised and `B` is zero-initialised, so an adapted
model equals the base model at step 0 (acceptance test T9).

- Spec: `docs/SPEC.md` §3.2, §5.1, §7.1, §7.4
- Acceptance tests: T9, T10
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .._vendor.timestral.layers import Attention
from .._vendor.timestral.model import TimestralModel

if TYPE_CHECKING:
    from ..config import LoraConfig

__all__ = [
    "LoRASliceLinear",
    "extract_lora_state_dict",
    "has_lora",
    "inject_lora",
    "lora_parameters",
    "merge_lora",
    "merged_state_dict",
    "unmerge_lora",
    "validate_lora_for_rows",
]

_QKV_SLICES: dict[str, int] = {"q": 0, "k": 1, "v": 2}
_SINGLE_ROW_ALLOWED_TARGETS: tuple[str, ...] = ("v", "o", "gate")


class LoRASliceLinear(nn.Module):
    """Low-rank adapter targeting named contiguous output row slices of a wrapped `nn.Linear`.

    Args:
        base: The frozen base `nn.Linear` layer.
        slices: Sequence of `(name, start_row, slice_dim)` triples targeting output rows
            `[start_row, start_row + slice_dim)` of `base.weight`.
        r: Low-rank bottleneck dimension.
        lora_alpha: Scaling numerator (`scaling = lora_alpha / r`).
        dropout: Dropout probability applied to the adapter input path while training.
    """

    def __init__(
        self,
        base: nn.Linear,
        slices: Sequence[tuple[str, int, int]],
        *,
        r: int,
        lora_alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"base must be an nn.Linear, got {type(base).__name__}")
        if len(slices) == 0:
            raise ValueError("at least one slice is required")
        out_features, in_features = base.weight.shape
        for name, start, dim in slices:
            if start < 0 or dim <= 0 or start + dim > out_features:
                raise ValueError(
                    f"slice {name!r} [{start}, {start + dim}) is out of bounds for out_features={out_features}"
                )

        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.slices: tuple[tuple[str, int, int], ...] = tuple(slices)
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / self.r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.merged = False

        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name, _, dim in self.slices:
            a = nn.Parameter(torch.empty((r, in_features), device=device, dtype=dtype))
            b = nn.Parameter(torch.zeros((dim, r), device=device, dtype=dtype))
            # Gaussian A, zero B (LoRA paper eq. 3 / §4.1; spec §5.1) -> exact identity at step 0.
            nn.init.normal_(a, mean=0.0, std=1.0 / math.sqrt(r))
            self.lora_A[name] = a
            self.lora_B[name] = b

        # Fast-path flags for forward layout: full single-slice linear (wO, _gate) vs 3-way QKV slices.
        self._is_full = len(self.slices) == 1 and self.slices[0][1] == 0 and self.slices[0][2] == out_features
        self._qkv_dim = out_features // 3 if (out_features % 3 == 0 and not self._is_full) else 0
        self._is_qkv = self._qkv_dim > 0 and all(
            dim == self._qkv_dim and start in (0, self._qkv_dim, 2 * self._qkv_dim) for _, start, dim in self.slices
        )

    def _slice_delta(self, name: str, dtype: torch.dtype) -> Tensor:
        """Low-rank weight update `(slice_dim, in_features)` computed in at least float32."""
        work = torch.promote_types(dtype, torch.float32)
        delta = (self.lora_B[name].to(work) @ self.lora_A[name].to(work)) * self.scaling
        return delta.to(dtype)

    @torch.no_grad()
    def merge(self) -> None:
        """Add every slice's low-rank update into `base.weight` in place. Idempotent."""
        if self.merged:
            return
        weight = self.base.weight
        for name, start, dim in self.slices:
            weight[start : start + dim].add_(self._slice_delta(name, weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Subtract every slice's low-rank update back out of `base.weight` in place. Idempotent."""
        if not self.merged:
            return
        weight = self.base.weight
        for name, start, dim in self.slices:
            weight[start : start + dim].sub_(self._slice_delta(name, weight.dtype))
        self.merged = False

    @torch.no_grad()
    def merged_weight(self) -> Tensor:
        """A copy of `base.weight` with all slice updates applied, without mutating this module."""
        weight = self.base.weight.detach().clone()
        if not self.merged:
            for name, start, dim in self.slices:
                weight[start : start + dim].add_(self._slice_delta(name, weight.dtype))
        return weight

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if self.merged:
            return out

        x_in = self.dropout(x)
        if self._is_full:
            name = self.slices[0][0]
            delta = F.linear(F.linear(x_in, self.lora_A[name]), self.lora_B[name]) * self.scaling
            return out + delta

        # One fused input projection across all adapted slices reads `x_in` once from memory.
        if len(self.slices) == 1:
            name = self.slices[0][0]
            hidden_parts = (F.linear(x_in, self.lora_A[name]),)
        else:
            a_cat = torch.cat([self.lora_A[name] for name, _, _ in self.slices], dim=0)
            hidden_parts = F.linear(x_in, a_cat).split(self.r, dim=-1)

        if self._is_qkv:
            parts = list(out.chunk(3, dim=-1))
            for (name, start, _), h in zip(self.slices, hidden_parts, strict=True):
                slot = start // self._qkv_dim
                parts[slot] = parts[slot] + F.linear(h, self.lora_B[name]) * self.scaling
            return torch.cat(parts, dim=-1)

        # Generic fallback for arbitrary non-QKV slices (out-of-place for autograd / compile safety).
        result = out.clone()
        for (name, start, dim), h in zip(self.slices, hidden_parts, strict=True):
            result[..., start : start + dim] = result[..., start : start + dim] + (
                F.linear(h, self.lora_B[name]) * self.scaling
            )
        return result

    def extra_repr(self) -> str:
        names = [name for name, _, _ in self.slices]
        return (
            f"in_features={self.base.in_features}, out_features={self.base.out_features}, "
            f"slices={names}, r={self.r}, alpha={self.lora_alpha}, merged={self.merged}"
        )


def validate_lora_for_rows(lora_config: LoraConfig, rows_per_series: int) -> None:
    """Enforce the single-row variate-attention guard (spec §5.1, decision D5, T10).

    With one row per series (univariate, no covariates), variate attention has a single key so
    softmax is identically 1 and query/key projections have zero effect on the output.
    """
    if rows_per_series < 1:
        raise ValueError(f"rows_per_series must be positive, got {rows_per_series}")
    if "variate" not in lora_config.axes or rows_per_series > 1:
        return
    if not lora_config.allow_single_row:
        raise ValueError(
            "variate-axis LoRA is disabled on single-row series (1 target, no covariates): variate attention "
            "has only one key so query and key adapters are inert. Use mode='lora_temporal', add covariates, "
            "or pass LoraConfig(allow_single_row=True, targets=('v',)) for a per-token value/output/gate transform"
        )
    disallowed = [t for t in lora_config.targets if t not in _SINGLE_ROW_ALLOWED_TARGETS]
    if disallowed:
        raise ValueError(
            f"with allow_single_row=True on single-row series, targets {disallowed!r} are inert and rejected; "
            f"allowed targets are {_SINGLE_ROW_ALLOWED_TARGETS}"
        )


def has_lora(model: nn.Module) -> bool:
    """Whether `model` contains any `LoRASliceLinear` adapter."""
    return any(isinstance(module, LoRASliceLinear) for module in model.modules())


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """All LoRA `A` and `B` parameters in `model`, in module order."""
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRASliceLinear):
            params.extend(module.lora_A.values())
            params.extend(module.lora_B.values())
    return params


def inject_lora(
    model: TimestralModel,
    lora_config: LoraConfig,
    *,
    rows_per_series: int | None = None,
) -> TimestralModel:
    """Wrap selected attention projections in `model` with `LoRASliceLinear` in place.

    Args:
        model: The Timestral network to adapt.
        lora_config: Which axes (`"time"`, `"variate"`) and slices (`"q"`, `"k"`, `"v"`, `"o"`,
            `"gate"`) to adapt, rank, alpha, dropout and single-row override.
        rows_per_series: Number of rows per series in the training data, used to enforce the
            single-row guard for variate attention (spec §5.1, T10).

    Returns:
        `model` with adapters injected in place. Base weights of wrapped layers are frozen; LoRA
        `A` and `B` parameters have `requires_grad=True`.
    """
    if has_lora(model):
        raise ValueError("LoRA adapters are already injected into this model")
    if rows_per_series is not None:
        validate_lora_for_rows(lora_config, rows_per_series)

    want_time = "time" in lora_config.axes
    want_variate = "variate" in lora_config.axes
    qkv_targets = [t for t in ("q", "k", "v") if t in lora_config.targets]
    want_o = "o" in lora_config.targets
    want_gate = "gate" in lora_config.targets

    wrapped = 0
    for module in model.modules():
        if not isinstance(module, Attention):
            continue
        if (module.is_time and not want_time) or (not module.is_time and not want_variate):
            continue
        d = module.embed_dim
        if qkv_targets:
            slices = [(name, _QKV_SLICES[name] * d, d) for name in qkv_targets]
            module.wQKV = LoRASliceLinear(
                module.wQKV, slices, r=lora_config.r, lora_alpha=lora_config.lora_alpha, dropout=lora_config.dropout
            )
            wrapped += 1
        if want_o:
            module.wO = LoRASliceLinear(
                module.wO,
                [("o", 0, d)],
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                dropout=lora_config.dropout,
            )
            wrapped += 1
        if want_gate and module._gate is not None:
            module._gate = LoRASliceLinear(
                module._gate,
                [("gate", 0, d)],
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                dropout=lora_config.dropout,
            )
            wrapped += 1

    if wrapped == 0:
        raise ValueError(
            f"LoraConfig(axes={lora_config.axes!r}, targets={lora_config.targets!r}) matched no attention projection "
            f"in this model (block_schedule={model.config.block_schedule!r})"
        )
    model._lora_config = lora_config  # type: ignore[attr-defined]
    return model


def merge_lora(model: nn.Module) -> None:
    """Merge every `LoRASliceLinear` adapter in `model` into its base linear weight in place."""
    for module in model.modules():
        if isinstance(module, LoRASliceLinear):
            module.merge()


def unmerge_lora(model: nn.Module) -> None:
    """Unmerge every `LoRASliceLinear` adapter in `model` back out of its base linear weight in place."""
    for module in model.modules():
        if isinstance(module, LoRASliceLinear):
            module.unmerge()


def extract_lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Unmerged LoRA `A` and `B` tensors on CPU, keyed by `<module>.lora_A.<slice>` / `lora_B.<slice>`."""
    tensors: dict[str, Tensor] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, LoRASliceLinear):
            for slice_name, param in module.lora_A.items():
                tensors[f"{module_name}.lora_A.{slice_name}"] = param.detach().cpu().contiguous()
            for slice_name, param in module.lora_B.items():
                tensors[f"{module_name}.lora_B.{slice_name}"] = param.detach().cpu().contiguous()
    return tensors


def merged_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Upstream-compatible state dict on CPU with all LoRA adapters merged, without mutating `model`.

    Parameter names match upstream `TimestralModel` exactly so the result loads with `strict=True`
    in the official package (spec §7.4, acceptance test T11).
    """
    lora_prefixes: dict[str, LoRASliceLinear] = {
        name: module for name, module in model.named_modules() if isinstance(module, LoRASliceLinear)
    }
    raw = model.state_dict()
    if not lora_prefixes:
        return {name: tensor.detach().cpu().contiguous() for name, tensor in raw.items()}

    out: dict[str, Tensor] = {}
    emitted: set[str] = set()
    for key, tensor in raw.items():
        matched_prefix: str | None = None
        for prefix in lora_prefixes:
            if key.startswith(prefix + "."):
                matched_prefix = prefix
                break
        if matched_prefix is None:
            out[key] = tensor.detach().cpu().contiguous()
            continue
        if matched_prefix in emitted:
            continue
        emitted.add(matched_prefix)
        adapter = lora_prefixes[matched_prefix]
        out[f"{matched_prefix}.weight"] = adapter.merged_weight().cpu().contiguous()
        if adapter.base.bias is not None:
            out[f"{matched_prefix}.bias"] = adapter.base.bias.detach().cpu().contiguous()
    return out
