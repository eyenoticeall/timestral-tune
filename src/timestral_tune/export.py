"""Checkpoint export in the upstream-compatible layout of spec §7.4.

Writes:
- `config.json`: exact upstream `TimestralConfig` schema, never adding extra keys.
- `model.safetensors`: merged weights under exact upstream parameter names, loadable in the
  official `timestral` package with `strict=True` (acceptance test T11).
- `adapter.safetensors` and `adapter_config.json`: unmerged LoRA weights and `LoraConfig`
  (written when the model has LoRA adapters; stale files are removed otherwise).
- `conformal.json`: serialised fitted `ConformalCalibrator` or `OnlineConformalCalibrator`
  (stale file removed if omitted).
- `tune_metadata.json`: sidecar recording base model, revision, mode, hyperparameters, content
  and window-aware data fingerprint, metrics, library version, and vendored upstream commit.
- `LICENSE` and `NOTICE`: base weights licence and derivative notice (spec §8).

All JSON payloads and licence/notice artefacts are validated and serialised in memory before any
file is written, so a serialisation failure never leaves partial weights without `LICENSE` or
`NOTICE`. Exporting is non-destructive: a live LoRA model stays unmerged and ready for further
training.

- Spec: `docs/SPEC.md` §7.4, §8, §11 (D10)
- Acceptance tests: T11
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from ._vendor.timestral.loading import CONFIG_FILE, WEIGHTS_FILE
from ._vendor.timestral.model import TimestralModel
from .config import LoraConfig, ReplayConfig, TrainingConfig
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

_FALLBACK_LICENSE_MARKER = b"No verbatim LICENSE file was found at the base checkpoint location during export."


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


def _normalise_json(value: Any) -> Any:
    """Convert NumPy scalars to Python primitives while leaving non-JSON types to fail strictly."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _normalise_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_normalise_json(v) for v in value]
    return value


def _dumps_json(payload: Any) -> str:
    return json.dumps(_normalise_json(payload), indent=2, allow_nan=False) + "\n"


def dataset_fingerprint(dataset: TimeSeriesDataset) -> str:
    """Deterministic SHA-256 content digest over a `TimeSeriesDataset`'s windows, timestamps, and values."""
    h = hashlib.sha256()
    header = {
        "num_series": len(dataset.series),
        "num_windows": len(dataset),
        "context_length": dataset.context_length,
        "forecast_horizon": dataset.forecast_horizon,
        "stride": dataset.stride,
        "min_context": dataset.min_context,
        "layout": list(dataset.series[0].layout),
    }
    h.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    # Include the active window selection so train and validation splits sharing `series` hash distinctly.
    for index_arr in (dataset._series_index, dataset._origins):
        contig = np.ascontiguousarray(index_arr, dtype=np.int64)
        h.update(str((contig.shape, contig.dtype.str)).encode("utf-8"))
        h.update(contig.tobytes())
    for series in dataset.series:
        h.update(repr(series.series_id).encode("utf-8"))
        for arr in (
            series.timestamps,
            series.targets,
            series.past,
            series.known,
            series.categorical_past,
            series.categorical_known,
        ):
            contig = np.ascontiguousarray(arr)
            h.update(str((contig.shape, contig.dtype.str)).encode("utf-8"))
            h.update(contig.tobytes())
    return f"sha256:{h.hexdigest()}"


def _resolve_license_bytes(
    base_repo_or_path: str | Path | None,
    base_revision: str | None,
    license_path: str | Path | None,
) -> tuple[bytes, bool]:
    """Return `(license_bytes, is_verbatim)`."""
    if license_path is not None:
        src = Path(license_path)
        if not src.is_file():
            raise FileNotFoundError(f"license_path {str(src)!r} does not exist")
        raw = src.read_bytes()
        is_verbatim = _FALLBACK_LICENSE_MARKER not in raw
        if not is_verbatim:
            logger.warning("license_path %r contains a fallback pointer, not a verbatim base LICENSE", str(src))
        return raw, is_verbatim

    if base_repo_or_path is not None:
        local = Path(base_repo_or_path).expanduser()
        if local.is_dir():
            candidate = local / LICENSE_FILE
            if candidate.is_file():
                raw = candidate.read_bytes()
                if _FALLBACK_LICENSE_MARKER not in raw:
                    return raw, True
                logger.warning(
                    "base directory %r holds a fallback %s pointer rather than a verbatim licence",
                    str(local),
                    LICENSE_FILE,
                )
        elif "/" in str(base_repo_or_path):
            try:
                downloaded = hf_hub_download(
                    repo_id=str(base_repo_or_path),
                    filename=LICENSE_FILE,
                    revision=base_revision,
                )
                return Path(downloaded).read_bytes(), True
            except Exception as exc:
                logger.warning("could not fetch %s from %s: %s", LICENSE_FILE, base_repo_or_path, exc)

    base_label = str(base_repo_or_path) if base_repo_or_path is not None else "the base Timestral checkpoint"
    logger.warning(
        "no verbatim %s found for %s; writing a licence pointer instead "
        "(pass license_path= to embed the verbatim text)",
        LICENSE_FILE,
        base_label,
    )
    fallback = (
        f"This checkpoint is derived from {base_label}.\n"
        f"{_FALLBACK_LICENSE_MARKER.decode('utf-8')}\n"
        "Derivatives of Mistral Timestral checkpoints are governed by Mistral AI's Modified MIT License\n"
        "(obtain the verbatim licence text from the base checkpoint repository on Hugging Face Hub).\n"
    )
    return fallback.encode("utf-8"), False


def _notice_text(
    base_repo_or_path: str | Path | None,
    base_revision: str | None,
    *,
    has_adapters: bool,
    verbatim_license: bool,
) -> str:
    base_label = str(base_repo_or_path) if base_repo_or_path is not None else "unspecified base checkpoint"
    rev_label = base_revision if base_revision is not None else "unspecified"
    adapter_note = (
        "Both the merged weights (`model.safetensors`) and the adapter files (`adapter.safetensors`,\n"
        "`adapter_config.json`) are treated as derivatives of the base weights.\n"
        if has_adapters
        else "The exported weights (`model.safetensors`) are a derivative of the base weights.\n"
    )
    license_ref = (
        "(see `LICENSE` in this directory)"
        if verbatim_license
        else "(no verbatim LICENSE was found at export time; obtain it from the base checkpoint repository)"
    )
    return (
        "NOTICE\n"
        "======\n\n"
        f"Base model:    {base_label}\n"
        f"Base revision: {rev_label}\n\n"
        f"{adapter_note}\n"
        "Weights derived from Mistral AI Timestral checkpoints are governed by Mistral AI's\n"
        f"Modified MIT License {license_ref}, including its commercial\n"
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
        "device": config.device,
        "seed": config.seed,
        "num_workers": config.num_workers,
        "eval_every": config.eval_every,
        "eval_batch_size": config.eval_batch_size,
        "early_stopping_patience": config.early_stopping_patience,
        "restore_best": config.restore_best,
        "shuffle": config.shuffle,
        "sign_flip": config.sign_flip,
        "cache_frozen_features": config.cache_frozen_features,
        "compile_model": config.compile_model,
        "compile_mode": config.compile_mode,
    }


@dataclass(frozen=True)
class _ValidatedExport:
    """Validated export inputs and in-memory JSON payloads produced without I/O or weight merges."""

    model_has_lora: bool
    effective_lora: LoraConfig | None
    adapter_config_json: str | None
    conformal_json: str | None
    replay_source_dict: dict[str, Any] | None
    metrics_dict: dict[str, Any] | None


def _validate_export_inputs(
    model: TimestralModel,
    *,
    lora_config: LoraConfig | None = None,
    calibrator: ConformalCalibrator | OnlineConformalCalibrator | None = None,
    metrics: Mapping[str, Any] | None = None,
    license_path: str | Path | None = None,
    replay_source: Mapping[str, Any] | None = None,
    data_fingerprint: str | None = None,
) -> _ValidatedExport:
    """Validate `export_checkpoint` inputs without Hub calls, weight merges, or disk writes.

    Checks, in the same order as `export_checkpoint`: fitted `ConformalCalibrator`, LoRA config
    match and presence when the model has adapters, strict (`allow_nan=False`) JSON serialisability
    of adapter config, calibrator, `replay_source`, `data_fingerprint`, and `metrics`, and existence
    of an explicit `license_path`.
    """
    if isinstance(calibrator, ConformalCalibrator) and not calibrator.is_fitted:
        raise ValueError("cannot export an unfitted ConformalCalibrator; call fit() first")

    model_has_lora = has_lora(model)
    injected_lora = getattr(model, "_lora_config", None)
    if model_has_lora:
        if injected_lora is not None and lora_config is not None and lora_config != injected_lora:
            raise ValueError(
                f"passed lora_config={lora_config!r} does not match the model's injected {injected_lora!r}"
            )
        effective_lora = lora_config if lora_config is not None else injected_lora
        if effective_lora is None:
            raise ValueError("cannot export LoRA adapters without a LoraConfig; pass lora_config=")
    else:
        effective_lora = None

    adapter_config_json = _dumps_json(effective_lora.to_dict()) if effective_lora is not None else None
    conformal_json = _dumps_json(calibrator.to_dict()) if calibrator is not None else None
    replay_source_dict = None if replay_source is None else dict(replay_source)
    metrics_dict = None if metrics is None else dict(metrics)
    if replay_source_dict is not None:
        _dumps_json(replay_source_dict)
    if data_fingerprint is not None:
        _dumps_json(data_fingerprint)
    if metrics_dict is not None:
        _dumps_json(metrics_dict)

    if license_path is not None:
        src = Path(license_path)
        if not src.is_file():
            raise FileNotFoundError(f"license_path {str(src)!r} does not exist")

    return _ValidatedExport(
        model_has_lora=model_has_lora,
        effective_lora=effective_lora,
        adapter_config_json=adapter_config_json,
        conformal_json=conformal_json,
        replay_source_dict=replay_source_dict,
        metrics_dict=metrics_dict,
    )


def export_checkpoint(
    model: TimestralModel,
    output_dir: str | Path,
    *,
    base_repo_or_path: str | Path | None = None,
    base_revision: str | None = None,
    mode: str | None = None,
    training_config: TrainingConfig | None = None,
    lora_config: LoraConfig | None = None,
    replay_config: ReplayConfig | None = None,
    replay_source: Mapping[str, Any] | None = None,
    calibrator: ConformalCalibrator | OnlineConformalCalibrator | None = None,
    metrics: Mapping[str, Any] | None = None,
    data_fingerprint: str | None = None,
    license_path: str | Path | None = None,
    model_provided_directly: bool = False,
) -> Path:
    """Export `model` to `output_dir` in the upstream-loadable layout of spec §7.4.

    Does not mutate `model`: LoRA weights are merged into a copy for `model.safetensors` while
    unmerged adapter matrices are written to `adapter.safetensors`. All metadata, JSON payloads,
    and licence files are validated and serialised before writing anything to disk, so errors
    never leave a partial export without `LICENSE` or `NOTICE`. Stale conditional files from a
    previous export in `output_dir` are removed when not part of this export.
    """
    validated = _validate_export_inputs(
        model,
        lora_config=lora_config,
        calibrator=calibrator,
        metrics=metrics,
        license_path=license_path,
        replay_source=replay_source,
        data_fingerprint=data_fingerprint,
    )
    model_has_lora = validated.model_has_lora
    effective_lora = validated.effective_lora
    adapter_config_json = validated.adapter_config_json
    conformal_json = validated.conformal_json

    resolved_base = (
        base_repo_or_path
        if base_repo_or_path is not None
        else (training_config.model_name_or_path if training_config is not None else None)
    )
    resolved_revision = base_revision
    if resolved_revision is None and resolved_base is not None:
        explicit_rev = training_config.revision if training_config is not None else None
        resolved_revision = resolve_revision(resolved_base, explicit_rev)

    # Prepare and validate all serialised payloads in memory BEFORE touching `output_dir`.
    config_json = json.dumps(model.config.to_dict(), indent=2) + "\n"
    metadata: dict[str, Any] = {
        "base_model": None if resolved_base is None else str(resolved_base),
        "base_revision": resolved_revision,
        "model_provided_directly": model_provided_directly,
        "mode": mode if mode is not None else (training_config.mode if training_config is not None else None),
        "hyperparameters": None if training_config is None else _serialise_training_config(training_config),
        "lora_config": None if effective_lora is None else effective_lora.to_dict(),
        "replay_config": None if replay_config is None else replay_config.to_dict(),
        "replay_source": validated.replay_source_dict,
        "has_conformal": calibrator is not None,
        "data_fingerprint": data_fingerprint,
        "metrics": validated.metrics_dict,
        "library_version": _library_version(),
        "vendored_upstream_commit": _vendored_upstream_commit(),
    }
    metadata_json = _dumps_json(metadata)
    license_bytes, is_verbatim = _resolve_license_bytes(resolved_base, resolved_revision, license_path)
    notice_text = _notice_text(
        resolved_base, resolved_revision, has_adapters=model_has_lora, verbatim_license=is_verbatim
    )
    merged_weights = merged_state_dict(model)
    adapter_weights = extract_lora_state_dict(model) if model_has_lora else None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Exact upstream config.json (zero extra keys) and merged model.safetensors.
    (out / CONFIG_FILE).write_text(config_json, encoding="utf-8")
    save_file(merged_weights, str(out / WEIGHTS_FILE))

    # 2. Adapter files (LoRA only); remove stale files when exporting a non-LoRA checkpoint.
    if model_has_lora:
        assert adapter_weights is not None and adapter_config_json is not None
        save_file(adapter_weights, str(out / ADAPTER_WEIGHTS_FILE))
        (out / ADAPTER_CONFIG_FILE).write_text(adapter_config_json, encoding="utf-8")
    else:
        (out / ADAPTER_WEIGHTS_FILE).unlink(missing_ok=True)
        (out / ADAPTER_CONFIG_FILE).unlink(missing_ok=True)

    # 3. Conformal offsets (if provided); remove stale file when absent.
    if conformal_json is not None:
        (out / CONFORMAL_FILE).write_text(conformal_json, encoding="utf-8")
    else:
        (out / CONFORMAL_FILE).unlink(missing_ok=True)

    # 4. Sidecar metadata, licence and derivative notice (spec §8).
    (out / METADATA_FILE).write_text(metadata_json, encoding="utf-8")
    (out / LICENSE_FILE).write_bytes(license_bytes)
    (out / NOTICE_FILE).write_text(notice_text, encoding="utf-8")
    return out
