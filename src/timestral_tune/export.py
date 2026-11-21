"""Checkpoint export in the upstream-compatible layout of spec §7.4.

Writes:
- `config.json`: exact upstream `TimestralConfig` schema, never adding extra keys.
- `model.safetensors`: merged weights under exact upstream parameter names, loadable in the
  official `timestral` package with `strict=True` (acceptance test T11).
- `adapter.safetensors` and `adapter_config.json`: unmerged LoRA weights and `LoraConfig`
  (written when the model has LoRA adapters).
- `conformal.json`: serialised `ConformalCalibrator` or `OnlineConformalCalibrator` (if passed).
- `tune_metadata.json`: sidecar recording base model, revision, mode, hyperparameters, data
  fingerprint, metrics, library version, and vendored upstream commit.
- `LICENSE` and `NOTICE`: base weights licence and derivative notice (spec §8).

Exporting is non-destructive: a live LoRA model stays unmerged and ready for further training.

- Spec: `docs/SPEC.md` §7.4, §8, §11 (D10)
- Acceptance tests: T11
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from ._vendor.timestral.loading import CONFIG_FILE, WEIGHTS_FILE
from ._vendor.timestral.model import TimestralModel
from .config import LoraConfig, TrainingConfig
from .data.dataset import TimeSeriesDataset
from .models.adapter import extract_lora_state_dict, has_lora, merged_state_dict
from .models.conformal import ConformalCalibrator, OnlineConformalCalibrator
from .models.loader import resolve_revision

__all__ = ["dataset_fingerprint", "export_checkpoint"]

logger = logging.getLogger("timestral_tune.export")

ADAPTER_WEIGHTS_FILE = "adapter.safetensors"
ADAPTER_CONFIG_FILE = "adapter_config.json"
CONFORMAL_FILE = "conformal.json"
METADATA_FILE = "tune_metadata.json"
LICENSE_FILE = "LICENSE"
NOTICE_FILE = "NOTICE"


def _vendored_upstream_commit() -> str | None:
    manifest = Path(__file__).resolve().parent / "_vendor" / "timestral" / "VENDOR_MANIFEST.json"
    if not manifest.is_file():
        return None
    try:
        commit = json.loads(manifest.read_text(encoding="utf-8")).get("commit")
        return None if commit is None else str(commit)
    except (OSError, ValueError):
        return None


def _library_version() -> str:
    from . import __version__

    return __version__


def dataset_fingerprint(dataset: TimeSeriesDataset) -> str:
    """Deterministic SHA-256 summary of a `TimeSeriesDataset`'s structure and window boundaries."""
    h = hashlib.sha256()
    header = {
        "num_series": len(dataset.series),
        "num_windows": len(dataset),
        "context_length": dataset.context_length,
        "forecast_horizon": dataset.forecast_horizon,
        "stride": dataset.stride,
        "layout": list(dataset.series[0].layout),
        "series_lengths": [int(s.timestamps.shape[0]) for s in dataset.series],
    }
    h.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


def _resolve_license_bytes(
    base_repo_or_path: str | Path | None,
    base_revision: str | None,
    license_path: str | Path | None,
) -> bytes:
    if license_path is not None:
        src = Path(license_path)
        if not src.is_file():
            raise FileNotFoundError(f"license_path {str(src)!r} does not exist")
        return src.read_bytes()

    if base_repo_or_path is not None:
        local = Path(base_repo_or_path).expanduser()
        if local.is_dir():
            candidate = local / LICENSE_FILE
            if candidate.is_file():
                return candidate.read_bytes()
        elif "/" in str(base_repo_or_path):
            try:
                downloaded = hf_hub_download(
                    repo_id=str(base_repo_or_path),
                    filename=LICENSE_FILE,
                    revision=base_revision,
                )
                return Path(downloaded).read_bytes()
            except Exception as exc:
                logger.warning("could not fetch %s from %s: %s", LICENSE_FILE, base_repo_or_path, exc)

    base_label = str(base_repo_or_path) if base_repo_or_path is not None else "the base Timestral checkpoint"
    fallback = (
        f"This checkpoint is derived from {base_label}.\n"
        "No LICENSE file was found at the base checkpoint location during export.\n"
        "Derivatives of Mistral Timestral checkpoints are governed by Mistral AI's Modified MIT License\n"
        "(see the base checkpoint repository on Hugging Face Hub for the verbatim licence text).\n"
    )
    return fallback.encode("utf-8")


def _notice_text(
    base_repo_or_path: str | Path | None,
    base_revision: str | None,
    *,
    has_adapters: bool,
) -> str:
    base_label = str(base_repo_or_path) if base_repo_or_path is not None else "unspecified base checkpoint"
    rev_label = base_revision if base_revision is not None else "unspecified"
    adapter_note = (
        "Both the merged weights (`model.safetensors`) and the adapter files (`adapter.safetensors`,\n"
        "`adapter_config.json`) are treated as derivatives of the base weights.\n"
        if has_adapters
        else "The exported weights (`model.safetensors`) are a derivative of the base weights.\n"
    )
    return (
        "NOTICE\n"
        "======\n\n"
        f"Base model:    {base_label}\n"
        f"Base revision: {rev_label}\n\n"
        f"{adapter_note}\n"
        "Weights derived from Mistral AI Timestral checkpoints are governed by Mistral AI's\n"
        "Modified MIT License (see `LICENSE` in this directory), including its commercial\n"
        "revenue restriction. The `timestral-tune` adaptation toolkit itself is an independent\n"
        "community project licensed under Apache-2.0 and is not affiliated with Mistral AI.\n"
    )


def _serialise_training_config(config: TrainingConfig) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "learning_rate": config.resolved_learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "max_steps": config.max_steps,
        "warmup_steps": config.warmup_steps,
        "lr_schedule": config.lr_schedule,
        "gradient_clip_val": config.gradient_clip_val,
        "mixed_precision": config.mixed_precision,
        "target_frame": config.target_frame,
        "seed": config.seed,
        "cache_frozen_features": config.cache_frozen_features,
        "compile_model": config.compile_model,
        "compile_mode": config.compile_mode,
    }


def export_checkpoint(
    model: TimestralModel,
    output_dir: str | Path,
    *,
    base_repo_or_path: str | Path | None = None,
    base_revision: str | None = None,
    mode: str | None = None,
    training_config: TrainingConfig | None = None,
    lora_config: LoraConfig | None = None,
    calibrator: ConformalCalibrator | OnlineConformalCalibrator | None = None,
    metrics: Mapping[str, Any] | None = None,
    data_fingerprint: str | None = None,
    license_path: str | Path | None = None,
) -> Path:
    """Export `model` to `output_dir` in the upstream-loadable layout of spec §7.4.

    Does not mutate `model`: LoRA weights are merged into a copy for `model.safetensors` while
    unmerged adapter matrices are written to `adapter.safetensors`.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    resolved_base = (
        base_repo_or_path
        if base_repo_or_path is not None
        else (training_config.model_name_or_path if training_config is not None else None)
    )
    resolved_revision = base_revision
    if resolved_revision is None and resolved_base is not None:
        explicit_rev = training_config.revision if training_config is not None else None
        resolved_revision = resolve_revision(resolved_base, explicit_rev)

    # 1. Exact upstream config.json (zero extra keys) and merged model.safetensors.
    model.config.save(out / CONFIG_FILE)
    save_file(merged_state_dict(model), str(out / WEIGHTS_FILE))

    # 2. Adapter files (LoRA only).
    model_has_lora = has_lora(model)
    effective_lora = lora_config if lora_config is not None else getattr(model, "_lora_config", None)
    if model_has_lora:
        save_file(extract_lora_state_dict(model), str(out / ADAPTER_WEIGHTS_FILE))
        if effective_lora is not None:
            (out / ADAPTER_CONFIG_FILE).write_text(
                json.dumps(effective_lora.to_dict(), indent=2) + "\n", encoding="utf-8"
            )

    # 3. Conformal offsets (if fitted / provided).
    if calibrator is not None:
        calibrator.save(out / CONFORMAL_FILE)

    # 4. Sidecar metadata (never inside config.json).
    metadata: dict[str, Any] = {
        "base_model": None if resolved_base is None else str(resolved_base),
        "base_revision": resolved_revision,
        "mode": mode if mode is not None else (training_config.mode if training_config is not None else None),
        "hyperparameters": None if training_config is None else _serialise_training_config(training_config),
        "lora_config": None if effective_lora is None else effective_lora.to_dict(),
        "has_conformal": calibrator is not None,
        "data_fingerprint": data_fingerprint,
        "metrics": None if metrics is None else dict(metrics),
        "library_version": _library_version(),
        "vendored_upstream_commit": _vendored_upstream_commit(),
    }
    (out / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # 5. Licence and derivative notice (spec §8).
    (out / LICENSE_FILE).write_bytes(_resolve_license_bytes(resolved_base, resolved_revision, license_path))
    (out / NOTICE_FILE).write_text(
        _notice_text(resolved_base, resolved_revision, has_adapters=model_has_lora), encoding="utf-8"
    )
    return out
