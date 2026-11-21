"""Slice LoRA for Timestral's fused attention projections.

Timestral's attention modules live at `transformer.layers.{i}.attns.{j}` and share one class,
told apart by `is_time: bool`, with three bias-free linears: `wQKV` (`D -> 3D`), `wO` (`D -> D`),
and optional `_gate` (`D -> D`). In `wQKV.weight`, rows `[0, D)` are query, `[D, 2D)` are key,
and `[2D, 3D)` are value (spec §3.2).

`LoRASliceLinear` wraps a linear layer and applies low-rank updates `(alpha / r) * B @ A` to one
or more contiguous output row slices, using one fused input projection across all adapted slices
of the wrapped linear. `A` is Gaussian-initialised and `B` is zero-initialised, so an adapted
model equals the base model at step 0 (acceptance test T9). `merge()` snapshots the original base
rows so `unmerge()` restores them bit-identically (even in `bf16` or if adapters are touched
while merged).

- Spec: `docs/SPEC.md` §3.2, §5.1, §7.1, §7.4
- Acceptance tests: T9, T10
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
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
    "load_lora",
    "load_lora_state_dict",
    "lora_parameters",
    "merge_lora",
    "merged_state_dict",
    "unmerge_lora",
    "validate_lora_for_rows",
]

logger = logging.getLogger("timestral_tune.lora")

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
        generator: Optional local RNG for `lora_A` Gaussian initialisation (avoids mutating
            PyTorch's global RNG).
    """

    def __init__(
        self,
        base: nn.Linear,
        slices: Sequence[tuple[str, int, int]],
        *,
        r: int,
        lora_alpha: float,
        dropout: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"base must be an nn.Linear, got {type(base).__name__}")
        if len(slices) == 0:
            raise ValueError("at least one slice is required")
        out_features, in_features = base.weight.shape
        seen_names: set[str] = set()
        for name, start, dim in slices:
            if name in seen_names:
                raise ValueError(f"duplicate slice name {name!r}")
            seen_names.add(name)
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
        self._saved_base_slices: dict[str, Tensor] | None = None

        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name, _, dim in self.slices:
            # Draw on CPU when a local CPU generator is supplied, then move to the base weight's device.
            if generator is not None and generator.device != device:
                init_a = torch.empty((r, in_features), device=generator.device, dtype=dtype)
                nn.init.normal_(init_a, mean=0.0, std=1.0 / math.sqrt(r), generator=generator)
                a = nn.Parameter(init_a.to(device=device))
            else:
                a = nn.Parameter(torch.empty((r, in_features), device=device, dtype=dtype))
                nn.init.normal_(a, mean=0.0, std=1.0 / math.sqrt(r), generator=generator)
            b = nn.Parameter(torch.zeros((dim, r), device=device, dtype=dtype))
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
        """Add every slice's low-rank update into `base.weight` in place. Idempotent.

        Snapshots the pre-merge base rows so `unmerge()` restores them bit-identically regardless
        of dtype (including `bf16`) or later changes to `lora_A`/`lora_B`. Note that raw
        `state_dict()` does not persist the `merged` flag; use `merged_state_dict()` to serialise
        merged weights.
        """
        if self.merged:
            return
        weight = self.base.weight
        saved: dict[str, Tensor] = {}
        for name, start, dim in self.slices:
            saved[name] = weight[start : start + dim].detach().clone()
            weight[start : start + dim].add_(self._slice_delta(name, weight.dtype))
        self._saved_base_slices = saved
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Restore `base.weight` to its exact pre-merge values in place. Idempotent."""
        if not self.merged:
            return
        weight = self.base.weight
        if self._saved_base_slices is not None:
            for name, start, dim in self.slices:
                weight[start : start + dim].copy_(self._saved_base_slices[name])
            self._saved_base_slices = None
        else:
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
    softmax is identically 1 and query/key projections have zero effect on the output. Because
    `LoraConfig.targets` applies across all configured `axes`, `"q"` and `"k"` are rejected
    whenever `"variate"` is adapted on single-row data.
    """
    if rows_per_series < 1:
        raise ValueError(f"rows_per_series must be positive, got {rows_per_series}")
    if "variate" not in lora_config.axes or rows_per_series > 1:
        return
    if not lora_config.allow_single_row:
        suggestion = (
            "Drop 'variate' from lora_config.axes" if "time" in lora_config.axes else "Use mode='lora_temporal'"
        )
        raise ValueError(
            "variate-axis LoRA is disabled on single-row series (1 target, no covariates): variate attention "
            f"has only one key so query and key adapters are inert. {suggestion}, add covariates, "
            "or pass LoraConfig(allow_single_row=True, targets=('v',)) for a per-token value/output/gate transform"
        )
    disallowed = [t for t in lora_config.targets if t not in _SINGLE_ROW_ALLOWED_TARGETS]
    if disallowed:
        raise ValueError(
            f"with allow_single_row=True on single-row series, targets {disallowed!r} are inert on variate "
            f"attention and rejected; allowed targets are {_SINGLE_ROW_ALLOWED_TARGETS}"
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
    generator: torch.Generator | None = None,
) -> TimestralModel:
    """Wrap selected attention projections in `model` with `LoRASliceLinear` in place.

    Args:
        model: The Timestral network to adapt.
        lora_config: Which axes (`"time"`, `"variate"`) and slices (`"q"`, `"k"`, `"v"`, `"o"`,
            `"gate"`) to adapt, rank, alpha, dropout and single-row override.
        rows_per_series: Number of rows per series in the training data, used to enforce the
            single-row guard for variate attention (spec §5.1, T10).
        generator: Optional local RNG for deterministic `lora_A` initialisation without mutating
            PyTorch's global RNG.

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
    matched_attns = 0
    gates_wrapped = 0
    for module in model.modules():
        if not isinstance(module, Attention):
            continue
        if (module.is_time and not want_time) or (not module.is_time and not want_variate):
            continue
        matched_attns += 1
        d = module.embed_dim
        if qkv_targets:
            slices = [(name, _QKV_SLICES[name] * d, d) for name in qkv_targets]
            module.wQKV = LoRASliceLinear(
                module.wQKV,
                slices,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                dropout=lora_config.dropout,
                generator=generator,
            )
            wrapped += 1
        if want_o:
            module.wO = LoRASliceLinear(
                module.wO,
                [("o", 0, d)],
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                dropout=lora_config.dropout,
                generator=generator,
            )
            wrapped += 1
        if want_gate and module._gate is not None:
            module._gate = LoRASliceLinear(
                module._gate,
                [("gate", 0, d)],
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                dropout=lora_config.dropout,
                generator=generator,
            )
            wrapped += 1
            gates_wrapped += 1

    if wrapped == 0:
        raise ValueError(
            f"LoraConfig(axes={lora_config.axes!r}, targets={lora_config.targets!r}) matched no attention projection "
            f"in this model (block_schedule={model.config.block_schedule!r})"
        )
    if want_gate and gates_wrapped < matched_attns:
        logger.warning(
            "target 'gate' adapted %d of %d matched attention modules (the others have no gate)",
            gates_wrapped,
            matched_attns,
        )
    model._lora_config = lora_config  # type: ignore[attr-defined]
    return model


def merge_lora(model: nn.Module) -> None:
    """Merge every `LoRASliceLinear` adapter in `model` into its base linear weight in place."""
    for module in model.modules():
        if isinstance(module, LoRASliceLinear):
            module.merge()


def unmerge_lora(model: nn.Module) -> None:
    """Restore every `LoRASliceLinear` adapter's base weight to its exact pre-merge state in place."""
    for module in model.modules():
        if isinstance(module, LoRASliceLinear):
            module.unmerge()


def _clean_compiled_key(key: str) -> str:
    """Strip `_orig_mod.` segments inserted when a submodule is wrapped by `torch.compile`."""
    return key.replace("._orig_mod.", ".").removeprefix("_orig_mod.")


def extract_lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Unmerged LoRA `A` and `B` tensors on CPU, keyed by `<module>.lora_A.<slice>` / `lora_B.<slice>`."""
    tensors: dict[str, Tensor] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, LoRASliceLinear):
            clean = _clean_compiled_key(module_name)
            for slice_name, param in module.lora_A.items():
                tensors[f"{clean}.lora_A.{slice_name}"] = param.detach().cpu().contiguous()
            for slice_name, param in module.lora_B.items():
                tensors[f"{clean}.lora_B.{slice_name}"] = param.detach().cpu().contiguous()
    return tensors


@torch.no_grad()
def load_lora_state_dict(model: nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    """Load saved LoRA `A` and `B` tensors into an already-injected model (strict key & shape check)."""
    if not has_lora(model):
        raise ValueError("model has no LoRA adapters; call inject_lora() or load_lora() first")
    expected: dict[str, nn.Parameter] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, LoRASliceLinear):
            if module.merged:
                raise ValueError(f"cannot load LoRA weights into merged adapter {module_name!r}; unmerge first")
            clean = _clean_compiled_key(module_name)
            for slice_name, param in module.lora_A.items():
                expected[f"{clean}.lora_A.{slice_name}"] = param
            for slice_name, param in module.lora_B.items():
                expected[f"{clean}.lora_B.{slice_name}"] = param

    provided_keys = set(state_dict)
    expected_keys = set(expected)
    missing = sorted(expected_keys - provided_keys)
    unexpected = sorted(provided_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(f"LoRA state_dict key mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")

    for key, param in expected.items():
        src = state_dict[key]
        if tuple(src.shape) != tuple(param.shape):
            raise ValueError(f"shape mismatch for {key}: expected {tuple(param.shape)}, got {tuple(src.shape)}")
        param.copy_(src.to(device=param.device, dtype=param.dtype))


def load_lora(
    model: TimestralModel,
    adapter_dir_or_weights: str | Path,
    *,
    adapter_config_path: str | Path | None = None,
    rows_per_series: int | None = None,
) -> tuple[TimestralModel, LoraConfig]:
    """Attach exported LoRA adapters (`adapter.safetensors` + `adapter_config.json`) onto a base model.

    Args:
        model: Base `TimestralModel` (without LoRA adapters injected yet).
        adapter_dir_or_weights: Export directory containing `adapter.safetensors` and
            `adapter_config.json`, or direct path to `adapter.safetensors`.
        adapter_config_path: Explicit path to `adapter_config.json` (required only if
            `adapter_dir_or_weights` is a file not sitting next to `adapter_config.json`).
        rows_per_series: Optional row count for single-row guard validation.

    Returns:
        `(model, lora_config)` with adapters injected and loaded in place.
    """
    from ..config import LoraConfig

    path = Path(adapter_dir_or_weights)
    if path.is_dir():
        weights_file = path / "adapter.safetensors"
        config_file = Path(adapter_config_path) if adapter_config_path is not None else path / "adapter_config.json"
    else:
        weights_file = path
        config_file = (
            Path(adapter_config_path) if adapter_config_path is not None else path.with_name("adapter_config.json")
        )
    if not weights_file.is_file():
        raise FileNotFoundError(f"adapter weights not found at {str(weights_file)!r}")
    if not config_file.is_file():
        raise FileNotFoundError(f"adapter config not found at {str(config_file)!r}")

    lora_config = LoraConfig.from_dict(json.loads(config_file.read_text(encoding="utf-8")))
    # Use a local generator so reloading adapters never mutates PyTorch's global RNG.
    inject_lora(model, lora_config, rows_per_series=rows_per_series, generator=torch.Generator().manual_seed(0))
    load_lora_state_dict(model, load_file(str(weights_file), device="cpu"))
    return model, lora_config


def merged_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Upstream-compatible state dict on CPU with all LoRA adapters merged, without mutating `model`.

    Parameter names match upstream `TimestralModel` exactly (including when a submodule has been
    wrapped with `torch.compile`) so the result loads with `strict=True` in the official package
    (spec §7.4, acceptance test T11).
    """
    lora_prefixes: dict[str, LoRASliceLinear] = {
        _clean_compiled_key(name): module
        for name, module in model.named_modules()
        if isinstance(module, LoRASliceLinear)
    }
    raw = model.state_dict()
    if not lora_prefixes:
        return {_clean_compiled_key(name): tensor.detach().cpu().contiguous() for name, tensor in raw.items()}

    out: dict[str, Tensor] = {}
    emitted: set[str] = set()
    for raw_key, tensor in raw.items():
        key = _clean_compiled_key(raw_key)
        matches = [prefix for prefix in lora_prefixes if key.startswith(prefix + ".")]
        if not matches:
            out[key] = tensor.detach().cpu().contiguous()
            continue
        if len(matches) > 1:
            raise ValueError(f"nested LoRASliceLinear adapters are not supported: {matches!r}")
        matched_prefix = matches[0]
        if matched_prefix in emitted:
            continue
        emitted.add(matched_prefix)
        adapter = lora_prefixes[matched_prefix]
        out[f"{matched_prefix}.weight"] = adapter.merged_weight().cpu().contiguous()
        if adapter.base.bias is not None:
            out[f"{matched_prefix}.bias"] = adapter.base.bias.detach().cpu().contiguous()
    return out
