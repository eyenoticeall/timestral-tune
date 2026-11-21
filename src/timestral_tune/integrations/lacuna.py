"""Optional Lacuna integration (Python 3.11+, `timestral-tune[lacuna]`).

Appends trial records and model selections to a `lacuna.ExperimentRegistry` and writes long-format
signal frames with the default columns (`time`, `instrument`, `signal`) expected by
`lacuna.SignalStudy`. Does not write Lacuna bundles and computes no Sharpe ratios (spec §7.5, D8).

Nothing in this module is imported by the core package. `lacuna` is imported lazily so core
installations on Python 3.10+ never require `lacuna-quant`.

- Spec: `docs/SPEC.md` §7.5, §11 (D8)
- Roadmap phase: 3
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from torch import Tensor

from ..config import LoraConfig, TrainingConfig
from ..export import _normalise_json, _serialise_training_config, _vendored_upstream_commit, dataset_fingerprint
from ..models.conformal import DEFAULT_QUANTILE_LEVELS
from ..utils.metrics import ForgettingSummary, MetricsSummary

if TYPE_CHECKING:
    from ..data.dataset import TimeSeriesDataset
    from ..trainer.trainer import TrainingResult

__all__ = ["LacunaExporter", "write_signal_frame"]

SignalSpec = Literal["median", "spread_80", "spread_90", "iqr"] | Callable[[np.ndarray], np.ndarray]
FrameFormat = Literal["auto", "pyarrow", "polars", "pandas", "dict"]


def _import_lacuna() -> Any:
    if sys.version_info < (3, 11):
        raise RuntimeError(
            f"the optional Lacuna integration requires Python 3.11+, running on {sys.version.split()[0]}"
        )
    try:
        import lacuna  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "lacuna-quant is not installed; install the optional extra with: pip install 'timestral-tune[lacuna]'"
        ) from exc
    return lacuna


def _code_fingerprint() -> str:
    from .. import __version__

    commit = _vendored_upstream_commit() or "unknown"
    return f"timestral-tune:{__version__}+upstream:{commit[:12]}"


def _result_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_normalise_json(payload), sort_keys=True, allow_nan=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class LacunaExporter:
    """Thin adapter around `lacuna.ExperimentRegistry` for recording `timestral-tune` trials (spec §7.5).

    Records every training or evaluation run (with parameters, metric, data/code/result
    fingerprints, and metadata) and logs model selection via `record_selection` so Lacuna holds
    the full candidate set for multiple-testing audits. Never writes Lacuna bundles and never
    computes Sharpe ratios.

    Args:
        registry_or_name: Either an existing `lacuna.ExperimentRegistry` instance or an experiment
            name string used to construct one.
        path: Optional SQLite path passed to `lacuna.ExperimentRegistry(name, path=...)` when
            `registry_or_name` is a string.
        family: Optional experiment family passed to `lacuna.ExperimentRegistry`.
    """

    def __init__(
        self,
        registry_or_name: Any,
        *,
        path: str | Path | None = None,
        family: str | None = None,
    ) -> None:
        if isinstance(registry_or_name, str):
            lacuna = _import_lacuna()
            self.registry = lacuna.ExperimentRegistry(registry_or_name, path=path, family=family)
            self._owns_registry = True
        else:
            if not (hasattr(registry_or_name, "record") and hasattr(registry_or_name, "record_selection")):
                raise TypeError("registry_or_name must be an experiment name or a lacuna.ExperimentRegistry")
            self.registry = registry_or_name
            self._owns_registry = False

    def __enter__(self) -> LacunaExporter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_registry and hasattr(self.registry, "close"):
            self.registry.close()

    def record_run(
        self,
        *,
        metric: float | None,
        metric_name: str = "wql",
        training_config: TrainingConfig | None = None,
        lora_config: LoraConfig | None = None,
        parameters: Mapping[str, Any] | None = None,
        dataset: TimeSeriesDataset | None = None,
        data_fingerprint: str | None = None,
        metrics: MetricsSummary | Mapping[str, Any] | None = None,
        training_result: TrainingResult | None = None,
        forgetting: ForgettingSummary | None = None,
        status: str = "completed",
        method: str = "timestral_tune.evaluation",
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Append one training or evaluation run to the registry and return Lacuna's `AttemptRecord`."""
        params: dict[str, Any] = {}
        if training_config is not None:
            params.update(_serialise_training_config(training_config))
            params["model_name_or_path"] = str(training_config.model_name_or_path)
            if training_config.revision is not None:
                params["revision"] = training_config.revision
        if lora_config is not None:
            params["lora"] = lora_config.to_dict()
        if parameters is not None:
            params.update(parameters)
        params = _normalise_json(params)

        resolved_data_fp = data_fingerprint
        if resolved_data_fp is None and dataset is not None:
            resolved_data_fp = dataset_fingerprint(dataset)

        meta: dict[str, Any] = {}
        if metrics is not None:
            meta["metrics"] = metrics.to_dict() if isinstance(metrics, MetricsSummary) else dict(metrics)
        if training_result is not None:
            meta["training"] = {
                "steps": training_result.steps,
                "epochs": training_result.epochs,
                "final_train_loss": training_result.final_train_loss,
                "best_eval_loss": training_result.best_eval_loss,
                "best_step": training_result.best_step,
                "stopped_early": training_result.stopped_early,
                "restored_best": training_result.restored_best,
            }
        if forgetting is not None:
            meta["forgetting"] = forgetting.to_dict()
        if metadata is not None:
            meta.update(metadata)
        meta = _normalise_json(meta)

        result_fp = _result_fingerprint({"metric_name": metric_name, "metric": metric, "metadata": meta})
        metric_val = None if metric is None else float(metric)

        return self.registry.record(
            parameters=params,
            metric=metric_val,
            metric_name=metric_name,
            status=status,
            method=method,
            data_fingerprint=resolved_data_fp,
            code_fingerprint=_code_fingerprint(),
            result_fingerprint=result_fp,
            metadata=meta or None,
        )

    def record_selection(
        self,
        eligible_trial_ids: Sequence[str],
        selected_trial_ids: Sequence[str],
        *,
        metric: str = "wql",
        direction: Literal["minimize", "maximize"] = "minimize",
        used_holdout: bool = True,
        actor: str | None = None,
        exclusion_reasons: Mapping[str, Any] | None = None,
    ) -> Any:
        """Record model selection among candidate `trial_id`s in `lacuna.ExperimentRegistry`."""
        return self.registry.record_selection(
            eligible_trial_ids=list(eligible_trial_ids),
            selected_trial_ids=list(selected_trial_ids),
            metric=metric,
            direction=direction,
            used_holdout=used_holdout,
            actor=actor,
            exclusion_reasons=None if exclusion_reasons is None else _normalise_json(dict(exclusion_reasons)),
        )

    def attempts(self) -> tuple[Any, ...]:
        return tuple(self.registry.attempts())

    def selections(self) -> tuple[Any, ...]:
        return tuple(self.registry.selections())

    def to_result(self) -> Any:
        return self.registry.to_result()


def _extract_signal(
    preds: np.ndarray,
    signal_fn: SignalSpec,
    horizon_index: int | None,
    levels: tuple[float, ...],
) -> np.ndarray:
    if callable(signal_fn):
        values = np.asarray(signal_fn(preds), dtype=np.float64)
        if values.shape != (preds.shape[0],):
            raise ValueError(f"signal_fn must return shape ({preds.shape[0]},), got {values.shape}")
        return values

    def _idx(tau: float) -> int:
        for i, q in enumerate(levels):
            if abs(q - tau) < 1e-9:
                return i
        raise ValueError(f"quantile level {tau} not found in quantile_levels")

    if signal_fn == "median":
        grid = preds[..., _idx(0.50)]
    elif signal_fn == "iqr":
        grid = preds[..., _idx(0.75)] - preds[..., _idx(0.25)]
    elif signal_fn == "spread_80":
        grid = preds[..., _idx(0.90)] - preds[..., _idx(0.10)]
    elif signal_fn == "spread_90":
        grid = preds[..., _idx(0.95)] - preds[..., _idx(0.05)]
    else:
        raise ValueError(
            f"signal_fn must be 'median', 'iqr', 'spread_80', 'spread_90', or a callable; got {signal_fn!r}"
        )

    if horizon_index is None:
        return np.mean(grid, axis=1)
    if not (-preds.shape[1] <= horizon_index < preds.shape[1]):
        raise ValueError(f"horizon_index {horizon_index} out of range for horizon {preds.shape[1]}")
    return grid[:, horizon_index]


def _build_frame(columns: dict[str, Any], output_format: FrameFormat) -> Any:
    if output_format == "dict":
        return columns
    if output_format in ("auto", "pyarrow"):
        try:
            import pyarrow as pa

            return pa.table(columns)
        except ImportError:
            if output_format == "pyarrow":
                raise
    if output_format in ("auto", "polars"):
        try:
            import polars as pl

            return pl.DataFrame(columns)
        except ImportError:
            if output_format == "polars":
                raise
    if output_format in ("auto", "pandas"):
        try:
            import pandas as pd

            return pd.DataFrame(columns)
        except ImportError:
            if output_format == "pandas":
                raise
    return columns


def write_signal_frame(
    predictions: np.ndarray | Tensor,
    *,
    timestamps: Sequence[Any] | np.ndarray,
    instruments: Sequence[Any] | np.ndarray,
    signal_fn: SignalSpec = "median",
    horizon_index: int | None = 0,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
    path: str | Path | None = None,
    output_format: FrameFormat = "auto",
) -> Any:
    """Build (and optionally write) a long signal frame for `lacuna.SignalStudy` (spec §7.5).

    Emits columns `time`, `instrument`, `signal` — the exact defaults of `lacuna.SignalStudy`.
    The user supplies prices to Lacuna; `timestral-tune` computes no returns or Sharpe ratios.

    Args:
        predictions: `(N, H, Q)` raw-space forecast quantiles.
        timestamps: Length-`N` forecast-origin timestamps (datetime / numpy datetime64 / int).
        instruments: Length-`N` instrument / symbol identifiers.
        signal_fn: Built-in scalar reduction (`"median"`, `"iqr"`, `"spread_80"`, `"spread_90"`)
            or a callable mapping `(N, H, Q)` float64 quantiles to a length-`N` 1-D array.
        horizon_index: Horizon step to read for built-in reductions (default `0`, first forecast
            step; pass `None` to average across the full horizon `H`).
        quantile_levels: Quantile levels along the last axis of `predictions`.
        path: Optional file path to write (`.parquet` or `.csv`).
        output_format: Return type: `"auto"` (prefers `pyarrow.Table`, then `polars.DataFrame`,
            then `pandas.DataFrame`, then `dict`), or an explicit format.
    """
    preds = (
        predictions.detach().cpu().to(torch.float64).numpy()
        if isinstance(predictions, Tensor)
        else np.asarray(predictions, dtype=np.float64)
    )
    if preds.ndim != 3 or preds.shape[0] == 0 or preds.shape[1] == 0:
        raise ValueError(f"predictions must have non-empty shape (N, H, Q), got {preds.shape}")
    levels = tuple(float(q) for q in quantile_levels)
    if preds.shape[2] != len(levels):
        raise ValueError(f"predictions have {preds.shape[2]} quantiles, expected {len(levels)}")
    if not np.isfinite(preds).all():
        raise ValueError("predictions must be finite")

    n = preds.shape[0]
    time_list = list(timestamps)
    inst_list = [str(x) for x in instruments]
    if len(time_list) != n:
        raise ValueError(f"timestamps length {len(time_list)} must match predictions N={n}")
    if len(inst_list) != n:
        raise ValueError(f"instruments length {len(inst_list)} must match predictions N={n}")

    signal = _extract_signal(preds, signal_fn, horizon_index, levels)
    if not np.isfinite(signal).all():
        raise ValueError("extracted signal contains non-finite values")

    columns = {"time": time_list, "instrument": inst_list, "signal": signal.tolist()}

    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()
        if suffix == ".csv":
            with target.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["time", "instrument", "signal"])
                for t, inst, val in zip(time_list, inst_list, columns["signal"], strict=True):
                    writer.writerow([t, inst, val])
        elif suffix == ".parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.table(columns), target)
        else:
            raise ValueError(f"unsupported signal frame file extension {suffix!r}; use '.parquet' or '.csv'")

    return _build_frame(columns, output_format)
