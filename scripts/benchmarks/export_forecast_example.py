"""Export one week of forecasts on German power for the README figure.

    uv run --with scikit-learn==1.6.1 python scripts/benchmarks/export_forecast_example.py \
        --device mps --output docs/benchmarks/forecast_example_de_power.json

Runs Timestral-28M as released and after `full_ft` (seed 0, the Work Package 3 settings) on
`de_power_15m`, and writes the forecasts of two series for the first seven test days together with
whole-test statistics for every series. The two series are chosen by rule, not by eye: the ten series
are ranked by fine-tuned wQL over released wQL on all 25 test days and the middle two are exported.

The benchmark tables in `docs/BENCHMARKS.md` §6.3 come from `run_wp3_unfamiliar.py` (three seeds on an
L4); this is a single-seed re-run whose only purpose is to save forecast values, which that runner
does not keep. `scripts/make_forecast_figure.py` draws `assets/forecast_example.png` from the output.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from baselines import DEFAULT_QUANTILE_LEVELS
from datasets import DE_POWER_REPO, DE_POWER_REV, load_de_power_15m, load_general_forgetting_by_domain
from decision_metrics import evaluate_with_calibrators
from run_wp1_decision_metrics import _train_one_seed

from timestral_tune import collect_conformal_data
from timestral_tune.models import TuneTimestralForecaster, load_model
from timestral_tune.utils import evaluate_forecaster

MODEL_ID = "mistralai/Timestral-28M"
DAYS_SHOWN = 7
LEVELS = np.asarray(DEFAULT_QUANTILE_LEVELS, dtype=np.float64)


def _quantile(preds: np.ndarray, level: float) -> np.ndarray:
    return preds[..., int(np.argmin(np.abs(LEVELS - level)))]


def _wql(preds: np.ndarray, actual: np.ndarray) -> float:
    diff = actual[..., None] - preds
    return float(2.0 * np.maximum(LEVELS * diff, (LEVELS - 1.0) * diff).mean(-1).sum() / np.abs(actual).sum())


def _stats(preds: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    low, mid, high = (_quantile(preds, level) for level in (0.1, 0.5, 0.9))
    return {
        "wql": _wql(preds, actual),
        "mean_abs_error_mw": float(np.abs(mid - actual).mean()),
        "width_80_mw": float((high - low).mean()),
        "coverage_80": float(((actual >= low) & (actual <= high)).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    precision = "bf16" if args.bf16 else "fp32"

    splits = load_de_power_15m()
    test = splits.test
    base = load_model(MODEL_ID, device="cpu")
    released = copy.deepcopy(base).to(args.device).eval()
    released.transformer.rope.rebuild_inv_freq()
    forecaster = TuneTimestralForecaster(released)
    val_p, val_y, val_s = collect_conformal_data(forecaster, splits.val, batch_size=32)
    test_p, test_y, test_s = collect_conformal_data(forecaster, test, batch_size=32)
    _, released_preds = evaluate_with_calibrators(
        val_p,
        val_y,
        val_s,
        test_p,
        test_y,
        test_s,
        test,
        seasonality=splits.seasonality,
        patch_size=forecaster.patch_size,
    )

    general = load_general_forgetting_by_domain()
    released_general = {
        name: evaluate_forecaster(forecaster, ds, seasonality=24, batch_size=32) for name, ds in general.items()
    }
    row, tuned_preds = _train_one_seed(
        base,
        MODEL_ID,
        splits,
        general,
        released_general,
        mode="full_ft",
        device=args.device,
        precision=precision,
        max_steps=400,
        batch_size=32,
        eval_batch_size=32,
        seed=args.seed,
        eval_every=40,
        early_stopping_patience=5,
    )

    preds = {"released": released_preds["raw"], "fine_tuned": tuned_preds["raw"]}
    series_index = np.asarray(test._series_index)
    origins = np.asarray(test._origins)
    horizon = test.forecast_horizon
    series_ids = np.array([str(test.series[i].series_id) for i in series_index])
    names = sorted(set(series_ids.tolist()))
    first_ns = min(int(test.series[i].timestamps[o]) for i, o in zip(series_index, origins, strict=True))
    last_ns = max(int(test.series[i].timestamps[o + horizon - 1]) for i, o in zip(series_index, origins, strict=True))

    per_series: dict[str, Any] = {}
    for name in names:
        rows = series_ids == name
        per_series[name] = {model: _stats(p[rows], test_y[rows]) for model, p in preds.items()}
        per_series[name]["wql_ratio"] = per_series[name]["fine_tuned"]["wql"] / per_series[name]["released"]["wql"]
    ranked = sorted(names, key=lambda name: per_series[name]["wql_ratio"])
    chosen = ranked[len(ranked) // 2 - 1 : len(ranked) // 2 + 1]

    shown: dict[str, Any] = {}
    for name in chosen:
        rows = np.where(series_ids == name)[0][:DAYS_SHOWN]
        stamps = np.concatenate(
            [np.asarray(test.series[series_index[r]].timestamps[origins[r] : origins[r] + horizon]) for r in rows]
        ).astype("datetime64[ns]")
        entry: dict[str, Any] = {
            "start_utc": str(stamps[0].astype("datetime64[m]")),
            "step_minutes": int((stamps[1] - stamps[0]) / np.timedelta64(1, "m")),
            "steps_per_forecast": horizon,
            "actual_mw": np.round(test_y[rows].reshape(-1), 1).tolist(),
        }
        for model, p in preds.items():
            entry[model] = {
                f"q{int(100 * level):02d}_mw": np.round(_quantile(p[rows], level).reshape(-1), 1).tolist()
                for level in (0.1, 0.5, 0.9)
            }
        shown[name] = entry

    result = {
        "dataset": "de_power_15m",
        "source": f"{DE_POWER_REPO}@{DE_POWER_REV}",
        "licence": "CC BY 4.0",
        "attribution": (
            f"Actual generation values are from the Hugging Face dataset {DE_POWER_REPO} (CC BY 4.0), "
            "which republishes ENTSO-E Transparency Platform data."
        ),
        "test_date_range": f"{str(np.datetime64(first_ns, 'ns'))[:10]} to {str(np.datetime64(last_ns, 'ns'))[:10]}",
        "model": MODEL_ID,
        "mode": "full_ft",
        "seed": args.seed,
        "device": args.device,
        "precision": precision,
        "max_steps": 400,
        "batch_size": 32,
        "best_step": row["best_step"],
        "intervals": "raw model quantiles (no conformal calibration)",
        "selection_rule": (
            "rank the series by fine-tuned wQL / released wQL over all test days; export the middle two; "
            f"first {DAYS_SHOWN} test days"
        ),
        "ranking": ranked,
        "overall": {model: _stats(p, test_y) for model, p in preds.items()},
        "per_series": per_series,
        "shown": shown,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: shown {chosen}")
    print({model: {key: round(value, 4) for key, value in stats.items()} for model, stats in result["overall"].items()})


if __name__ == "__main__":
    main()
