"""Reproducible WP3 candidate screening: released Timestral-28M vs final classical baseline.

    uv run python scripts/benchmarks/screen_wp3_candidates.py \
        --device mps --output docs/benchmarks/wp3_screening.json

Evaluates zero-shot `Timestral-28M` against the final `run_tuned_classical` (and seasonal naive)
baseline on BOTH the validation split (`splits.val`) and the test split (`splits.test`) with
zero fine-tuning. Exists to document the exact WP3 selection audit (including the disclosure that
initial exploratory screening had checked the test split and that `india_air_quality_daily` does
not trail the final classical baseline by >= 5%).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from baselines import _build_stacked_features, run_seasonal_naive, run_tuned_classical
from datasets import (
    BenchmarkSplits,
    load_de_power_15m,
    load_india_air_quality_daily,
    load_sl_marine_waves_daily,
)

from timestral_tune import collect_conformal_data
from timestral_tune.models import DEFAULT_QUANTILE_LEVELS, TuneTimestralForecaster, load_model
from timestral_tune.utils import evaluate_predictions
from timestral_tune.utils.metrics import _collect_contexts


def _eval_classical_val_wql(splits: BenchmarkSplits, *, transform: str, alpha: float) -> float:
    """Recompute the validation-split wQL of the selected classical Ridge config (strictly OOS)."""
    h = splits.test.forecast_horizon
    s = splits.seasonality
    levels = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)
    n_series = len(splits.train.series)
    x_tr, y_tr_f, _, _, _ = _build_stacked_features(
        splits.train, seasonality=s, dataset_name=splits.name, include_series_id=True
    )
    x_va, y_va_f, _, _, _ = _build_stacked_features(
        splits.val, seasonality=s, dataset_name=splits.name, include_series_id=True
    )
    n_tr, n_va = len(splits.train), len(splits.val)
    d = x_tr.shape[1]
    X_tr = x_tr.reshape(n_tr, h, d)
    X_va = x_va.reshape(n_va, h, d)
    Y_tr = y_tr_f.reshape(n_tr, h)
    Y_va = y_va_f.reshape(n_va, h)

    y_tr_t = np.log1p(Y_tr) if transform == "log1p" else Y_tr
    cap_hi = float(np.max(y_tr_t) + 2.0) if transform == "log1p" else float("inf")
    tr_pt = np.empty((n_tr, h), dtype=np.float64)
    va_pt = np.empty((n_va, h), dtype=np.float64)
    for step in range(h):
        s_tr = X_tr[:, step, 0].astype(np.int64)
        s_va = X_va[:, step, 0].astype(np.int64)
        num_tr = X_tr[:, step, 2:]
        num_va = X_va[:, step, 2:]
        mu = num_tr.mean(axis=0, keepdims=True)
        sd = np.where(num_tr.std(axis=0, keepdims=True) > 1e-12, num_tr.std(axis=0, keepdims=True), 1.0)
        oh_tr = np.zeros((n_tr, n_series), dtype=np.float64)
        oh_va = np.zeros((n_va, n_series), dtype=np.float64)
        oh_tr[np.arange(n_tr), s_tr] = 1.0
        oh_va[np.arange(n_va), s_va] = 1.0
        d_tr = np.concatenate([(num_tr - mu) / sd, oh_tr, np.ones((n_tr, 1))], axis=1)
        d_va = np.concatenate([(num_va - mu) / sd, oh_va, np.ones((n_va, 1))], axis=1)
        eye = np.eye(d_tr.shape[1], dtype=np.float64)
        eye[-1, -1] = 0.0
        w_h = np.linalg.solve(d_tr.T @ d_tr + alpha * eye, d_tr.T @ y_tr_t[:, step])
        tr_pt[:, step] = d_tr @ w_h
        va_pt[:, step] = d_va @ w_h
    q_off = np.nanquantile(y_tr_t - tr_pt, levels, axis=0).T
    val_q = va_pt[:, :, None] + q_off[None, :, :]
    val_q = np.expm1(np.clip(val_q, 0.0, cap_hi)) if transform == "log1p" else np.maximum(val_q, 0.0)
    return float(evaluate_predictions(np.sort(val_q, axis=-1), Y_va, _collect_contexts(splits.val), seasonality=s).wql)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/wp3_screening.json"))
    args = parser.parse_args()

    model = load_model("mistralai/Timestral-28M", device=args.device).eval()
    model.transformer.rope.rebuild_inv_freq()
    fc = TuneTimestralForecaster(model)

    report: dict[str, Any] = {
        "model": "mistralai/Timestral-28M",
        "device": args.device,
        "disclosure": (
            "Initial ad-hoc screening prior to pre-registration evaluated zero-shot 28M vs an "
            "exploratory Ridge baseline on the test split. This script records both validation-split "
            "and test-split comparisons against the final run_tuned_classical baseline."
        ),
        "datasets": {},
    }

    for loader in (load_de_power_15m, load_india_air_quality_daily, load_sl_marine_waves_daily):
        sp = loader()
        sn_res, _ = run_seasonal_naive(sp)
        cl_res, _ = run_tuned_classical(sp)
        cl_val_wql = _eval_classical_val_wql(sp, transform=cl_res["best_transform"], alpha=cl_res["best_alpha"])
        cl_test_wql = float(cl_res["metrics"]["raw"]["wql"])

        val_p, val_y, _ = collect_conformal_data(fc, sp.val, batch_size=args.batch_size)
        test_p, test_y, _ = collect_conformal_data(fc, sp.test, batch_size=args.batch_size)
        zs_val_wql = float(
            evaluate_predictions(val_p, val_y, _collect_contexts(sp.val), seasonality=sp.seasonality).wql
        )
        zs_test_wql = float(
            evaluate_predictions(test_p, test_y, _collect_contexts(sp.test), seasonality=sp.seasonality).wql
        )

        val_gap = zs_val_wql / cl_val_wql - 1.0
        test_gap = zs_test_wql / cl_test_wql - 1.0
        report["datasets"][sp.name] = {
            "source": sp.source,
            "license": sp.license,
            "test_date_range": sp.test_date_range,
            "classical_config": {"transform": cl_res["best_transform"], "alpha": cl_res["best_alpha"]},
            "validation": {
                "classical_wql": round(cl_val_wql, 6),
                "zs_28m_wql": round(zs_val_wql, 6),
                "zs_vs_classical_rel": round(val_gap, 4),
                "meets_5pct_rule_on_val": val_gap >= 0.05,
            },
            "test": {
                "naive_wql": round(float(sn_res["metrics"]["raw"]["wql"]), 6),
                "classical_wql": round(cl_test_wql, 6),
                "zs_28m_wql": round(zs_test_wql, 6),
                "zs_vs_classical_rel": round(test_gap, 4),
                "meets_5pct_rule_on_test": test_gap >= 0.05,
            },
        }
        print(
            f"  [{sp.name:24s}] VAL: cl={cl_val_wql:.4f} zs={zs_val_wql:.4f} ({val_gap:+.1%}) | "
            f"TEST: cl={cl_test_wql:.4f} zs={zs_test_wql:.4f} ({test_gap:+.1%})",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved screening report to {args.output}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
