"""Checkpoint loading at pinned Hub revisions.

`load_model` and `load_forecaster` return the device-aware subclasses of `models/device.py`, so
the result works on CPU, CUDA and the Apple GPU alike.

Two rules from the spec are enforced here:

- Hyperparameters are always read from the checkpoint's `config.json`, never from code defaults
  (the released values differ from the defaults, spec §3.1).
- The four official repos resolve to the Hub revisions this library was verified against, unless
  the caller names another revision.

- Spec: `docs/SPEC.md` §3.1, §5.4, §7.2
- Acceptance tests: T14 (and T11 once export exists)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

from .._vendor.timestral.config import TimestralConfig
from .._vendor.timestral.loading import CONFIG_FILE, WEIGHTS_FILE, resolve_checkpoint
from .device import MIN_TORCH_FOR_MPS, TuneTimestralForecaster, TuneTimestralModel, mps_is_usable, supports_float64

__all__ = ["PINNED_REVISIONS", "load_forecaster", "load_model", "resolve_device", "resolve_revision"]

logger = logging.getLogger(__name__)

# Hub revisions of the official checkpoints that this library was verified against (docs/UPSTREAM.md).
PINNED_REVISIONS: dict[str, str] = {
    "mistralai/Timestral-5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    "mistralai/Timestral-28M": "49b26409ac87b5935539a5d8ffe7f397b720db47",
    "mistralai/Timestral-123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
    "mistralai/Timestral-209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
}


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Turn a device request into a `torch.device`.

    `"auto"` (or None) picks CUDA, then the Apple GPU, then CPU. An explicit request for a device
    that is not available raises instead of silently falling back.

    The Apple GPU needs PyTorch 2.8 or newer: older versions lack an operator upstream's code uses.
    On an older PyTorch `"auto"` skips it, and an explicit `"mps"` raises.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_is_usable():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' was requested but CUDA is not available")
    if resolved.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device 'mps' was requested but the Apple GPU backend is not available")
        if not mps_is_usable():
            minimum = ".".join(map(str, MIN_TORCH_FOR_MPS))
            raise RuntimeError(
                f"the Apple GPU backend needs PyTorch {minimum} or newer to run Timestral; this is "
                f"{torch.__version__}. Upgrade PyTorch or use device='cpu'"
            )
    return resolved


def resolve_revision(path_or_repo_id: str | Path, revision: str | None) -> str | None:
    """The Hub revision to load.

    An explicit `revision` always wins. Otherwise an official repo id resolves to its pinned
    revision, and anything else (a local directory, a third-party repo) resolves to None, which
    means the repo's default branch.
    """
    if revision is not None:
        return revision
    return PINNED_REVISIONS.get(str(path_or_repo_id))


def load_model(
    path_or_repo_id: str | Path,
    device: str | torch.device | None = "cpu",
    dtype: torch.dtype = torch.float32,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> TuneTimestralModel:
    """Load a Timestral checkpoint as a `TuneTimestralModel`, in eval mode.

    Args:
        path_or_repo_id: A local directory holding `config.json` and `model.safetensors`, or a
            Hugging Face Hub repo id such as `mistralai/Timestral-28M`.
        device: Target device, or `"auto"`.
        dtype: Weight dtype. The Apple GPU backend supports float32 only.
        revision: Hub branch, tag or commit. Defaults to the pinned revision for official repos.
        cache_dir, token, local_files_only: As in `huggingface_hub`.
    """
    target = resolve_device(device)
    if not supports_float64(target) and dtype != torch.float32:
        raise ValueError(f"the {target.type} backend supports float32 weights only, got {dtype}")

    resolved_revision = resolve_revision(path_or_repo_id, revision)
    if revision is None and resolved_revision is not None:
        logger.info("loading %s at pinned revision %s", path_or_repo_id, resolved_revision)
    checkpoint = resolve_checkpoint(
        path_or_repo_id,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config = TimestralConfig.load(checkpoint / CONFIG_FILE)
    model = TuneTimestralModel(config)
    model.load_state_dict(load_file(checkpoint / WEIGHTS_FILE, device="cpu"), strict=True)
    model.to(device=target, dtype=dtype)
    # The RoPE table is rebuilt on the target device, as upstream's loader does: a table computed on
    # CPU and then moved could differ in the last bit.
    model.transformer.rope.rebuild_inv_freq()
    return model.eval()


def load_forecaster(
    path_or_repo_id: str | Path,
    device: str | torch.device | None = "cpu",
    *,
    max_context: int | None = None,
    per_shot_horizon_patches: int = 32,
    quantile_real_cap_k: float = 1e4,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> TuneTimestralForecaster:
    """Load a checkpoint as a `TuneTimestralForecaster`: upstream's forecaster, usable on any device.

    This is the inference entry point of spec §7.2. The forecaster arguments (`max_context`,
    `per_shot_horizon_patches`, `quantile_real_cap_k`) are upstream's and keep their defaults.
    """
    model = load_model(
        path_or_repo_id,
        device=device,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    return TuneTimestralForecaster(model, max_context, per_shot_horizon_patches, quantile_real_cap_k)
