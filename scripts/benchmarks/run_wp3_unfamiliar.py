"""Work Package 3: Three non-hourly, post-release niche datasets vs tuned gradient-boosted trees.

    uv run --with scikit-learn==1.6.1 python scripts/benchmarks/run_wp3_unfamiliar.py \
        --device cuda --bf16 --sizes 28M 5M --seeds 0 1 2 \
        --output docs/benchmarks/wp3_unfamiliar_domains.json

Evaluates `de_power_15m` (15-min German renewable power + ENTSO-E day-ahead TSO forecasts),
`india_air_quality_daily` (daily 50-city Indian PM2.5 + weather/pollutants), and
`sl_marine_waves_daily` (daily 16-site Sri Lanka coastal wave height + ocean/weather) —
all strictly post-2026-10-27 test splits — comparing seasonal naive, validation-tuned domain Ridge,
validation-tuned 21-quantile GBT, released Timestral, conformal calibration alone, and all four
Timestral adaptation modes across three seeds with paired cluster bootstrap 95% CIs.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from baselines import (
    run_seasonal_naive,
    run_tso_day_ahead_baseline,
    run_tuned_classical,
    run_tuned_gbt,
)
from datasets import (
    BenchmarkSplits,
    load_de_power_15m,
    load_general_forgetting_by_domain,
    load_india_air_quality_daily,
    load_sl_marine_waves_daily,
)
from decision_metrics import evaluate_with_calibrators, paired_both_clusters, paired_both_clusters_multiseed
from run_wp1_decision_metrics import TRAIN_MODES, _reset_mem, _summarise_seeds, _train_one_seed

from timestral_tune import collect_conformal_data
from timestral_tune.models import TuneTimestralForecaster, load_model
from timestral_tune.utils import evaluate_forecaster
from timestral_tune.utils.metrics import _collect_contexts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--sizes", nargs="+", default=["28M", "5M"], choices=["5M", "28M", "123M", "209M"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["de_power_15m", "india_air_quality_daily", "sl_marine_waves_daily"],
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--precomputed-baselines-dir",
        type=Path,
        default=None,
        help="Optional dir holding precomputed baseline_<ds>.json, baseline_gbt_<ds>.npz, baseline_classical_<ds>.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    precision = "bf16" if args.bf16 else "fp32"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("== Loading WP3 non-hourly post-release datasets & GIFT-Eval ==", flush=True)
    loaders = {
        "de_power_15m": load_de_power_15m,
        "india_air_quality_daily": load_india_air_quality_daily,
        "sl_marine_waves_daily": load_sl_marine_waves_daily,
    }
    ds_map: dict[str, BenchmarkSplits] = {name: loaders[name]() for name in args.datasets}
    gen_by_domain = load_general_forgetting_by_domain()

    report: dict[str, Any] = {
        "precision": precision,
        "device": args.device,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "seeds": args.seeds,
        "datasets": {},
        "baselines": {},
        "models": {},
        "cross_size_5m_vs_28m_released": {},
    }

    baseline_preds: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, sp in ds_map.items():
        report["datasets"][name] = {
            "license": sp.license,
            "source": sp.source,
            "test_date_range": sp.test_date_range,
            "contamination_note": sp.contamination_note,
            "seasonality": sp.seasonality,
            "context_length": sp.test.context_length,
            "forecast_horizon": sp.test.forecast_horizon,
            "num_series": len(sp.test.series),
            "counts": {"train": len(sp.train), "val": len(sp.val), "test": len(sp.test)},
        }
        pre_json = args.precomputed_baselines_dir / f"baseline_{name}.json" if args.precomputed_baselines_dir else None
        pre_gbt = (
            args.precomputed_baselines_dir / f"baseline_gbt_{name}.npz" if args.precomputed_baselines_dir else None
        )
        pre_cl = (
            args.precomputed_baselines_dir / f"baseline_classical_{name}.npz"
            if args.precomputed_baselines_dir
            else None
        )
        if (
            pre_json is not None
            and pre_json.exists()
            and pre_gbt is not None
            and pre_gbt.exists()
            and pre_cl is not None
            and pre_cl.exists()
        ):
            report["baselines"][name] = json.loads(pre_json.read_text(encoding="utf-8"))
            zg = np.load(pre_gbt)
            zc = np.load(pre_cl)
            baseline_preds[name] = {
                "gbt_tuned": {"raw": zg["raw"], "cqr": zg["cqr"]},
                "classical_domain": {"raw": zc["raw"], "cqr": zc["cqr"]},
            }
            print(f"  [precomputed] loaded baselines for {name} from {pre_json}", flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            continue

        print(
            f"\n== Baselines for {name} (series={len(sp.test.series)}, tr={len(sp.train)}, "
            f"va={len(sp.val)}, te={len(sp.test)}, test={sp.test_date_range}) ==",
            flush=True,
        )
        sn_res, sn_p = run_seasonal_naive(sp)
        cl_res, cl_p = run_tuned_classical(sp)
        gbt_res, gbt_p = run_tuned_gbt(sp, seed=0)
        b_entry: dict[str, Any] = {
            "seasonal_naive": sn_res,
            "classical_domain": cl_res,
            "gbt_tuned": gbt_res,
        }
        baseline_preds[name] = {
            "seasonal_naive": sn_p,
            "classical_domain": cl_p,
            "gbt_tuned": gbt_p,
        }
        if name == "de_power_15m":
            tso_res, tso_p = run_tso_day_ahead_baseline(sp)
            b_entry["tso_day_ahead"] = tso_res
            baseline_preds[name]["tso_day_ahead"] = tso_p
            print(f"  [baseline | {name}] tso_day_ahead wQL={tso_res['metrics']['raw']['wql']:.4f}", flush=True)
        report["baselines"][name] = b_entry
        print(
            f"  [baseline | {name}] naive={sn_res['metrics']['raw']['wql']:.4f} | "
            f"classical({cl_res['best_transform']},a={cl_res['best_alpha']})={cl_res['metrics']['raw']['wql']:.4f} | "
            f"gbt_tuned({gbt_res['best_config']})={gbt_res['metrics']['raw']['wql']:.4f} "
            f"(wink80={gbt_res['metrics']['raw']['winkler_80']:.4f}, "
            f"cqr_w80={gbt_res['metrics']['cqr']['width_80']:.4f}, {gbt_res['elapsed_s']}s)",
            flush=True,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    zs_28m_cache: dict[str, dict[str, np.ndarray]] = {}
    tuned_5m_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for size in args.sizes:
        model_id = f"mistralai/Timestral-{size}"
        base_cpu = load_model(model_id, device="cpu")
        report["models"][size] = {}

        zs_model = copy.deepcopy(base_cpu).to(args.device).eval()
        zs_model.transformer.rope.rebuild_inv_freq()
        zs_fc = TuneTimestralForecaster(zs_model)
        zs_by_domain = {
            dom: evaluate_forecaster(zs_fc, ds, seasonality=24, batch_size=args.batch_size)
            for dom, ds in gen_by_domain.items()
        }

        for name, sp in ds_map.items():
            print(f"\n================ {size} | {name} ================", flush=True)
            val_p, val_y, val_s = collect_conformal_data(zs_fc, sp.val, batch_size=args.batch_size)
            test_p, test_y, test_s = collect_conformal_data(zs_fc, sp.test, batch_size=args.batch_size)
            test_ctx = _collect_contexts(sp.test)
            zs_metrics, zs_preds = evaluate_with_calibrators(
                val_p,
                val_y,
                val_s,
                test_p,
                test_y,
                test_s,
                sp.test,
                seasonality=sp.seasonality,
                patch_size=zs_fc.patch_size,
            )
            if size == "28M":
                zs_28m_cache[name] = {k: zs_preds[k].copy() for k in ("raw", "cqr", "aci")}
            print(
                f"  [{size} | {name}] zero_shot | wQL={zs_metrics['raw']['wql']:.4f} "
                f"wink80={zs_metrics['raw']['winkler_80']:.4f} cov80={zs_metrics['raw']['coverage_80']:.3f} "
                f"cqr_cov80={zs_metrics['cqr']['coverage_80']:.3f} cqr_w80={zs_metrics['cqr']['width_80']:.4f}",
                flush=True,
            )

            del zs_model, zs_fc
            _reset_mem(args.device)

            ds_entry: dict[str, Any] = {
                "zero_shot": zs_metrics,
                "paired_zs_vs_gbt": paired_both_clusters(
                    zs_preds["raw"], baseline_preds[name]["gbt_tuned"]["raw"], test_y, test_ctx, sp.test, sp.seasonality
                ),
                "modes": {},
            }
            for mode in TRAIN_MODES:
                seed_rows: list[dict[str, Any]] = []
                seed_raw: list[np.ndarray] = []
                seed_cqr: list[np.ndarray] = []
                seed_aci: list[np.ndarray] = []
                for sd in args.seeds:
                    row, p_dict = _train_one_seed(
                        base_cpu,
                        model_id,
                        sp,
                        gen_by_domain,
                        zs_by_domain,
                        mode=mode,
                        device=args.device,
                        precision=precision,
                        max_steps=args.max_steps,
                        batch_size=args.batch_size,
                        eval_batch_size=args.batch_size,
                        seed=sd,
                        eval_every=40,
                        early_stopping_patience=5,
                    )
                    seed_rows.append(row)
                    seed_raw.append(p_dict["raw"].copy())
                    seed_cqr.append(p_dict["cqr"].copy())
                    seed_aci.append(p_dict["aci"].copy())

                summary = _summarise_seeds(seed_rows)
                if size == "5M":
                    tuned_5m_cache[(name, mode)] = {"raw": seed_raw, "cqr": seed_cqr}

                gbt_p = baseline_preds[name]["gbt_tuned"]
                cl_p = baseline_preds[name]["classical_domain"]
                ds_entry["modes"][mode] = {
                    "seeds": seed_rows,
                    "summary": summary,
                    "paired_vs_zero_shot": {
                        "raw": paired_both_clusters_multiseed(
                            seed_raw, zs_preds["raw"], test_y, test_ctx, sp.test, sp.seasonality
                        ),
                        "cqr": paired_both_clusters_multiseed(
                            seed_cqr, zs_preds["cqr"], test_y, test_ctx, sp.test, sp.seasonality
                        ),
                        "aci": paired_both_clusters_multiseed(
                            seed_aci, zs_preds["aci"], test_y, test_ctx, sp.test, sp.seasonality
                        ),
                    },
                    "paired_vs_gbt": {
                        "raw": paired_both_clusters_multiseed(
                            seed_raw, gbt_p["raw"], test_y, test_ctx, sp.test, sp.seasonality
                        ),
                        "cqr": paired_both_clusters_multiseed(
                            seed_cqr, gbt_p["cqr"], test_y, test_ctx, sp.test, sp.seasonality
                        ),
                    },
                    "paired_vs_classical": {
                        "raw": paired_both_clusters_multiseed(
                            seed_raw, cl_p["raw"], test_y, test_ctx, sp.test, sp.seasonality
                        ),
                    },
                }
                vs_gbt = ds_entry["modes"][mode]["paired_vs_gbt"]["raw"]["by_series"]["wql"]
                ci_lo, ci_hi = vs_gbt["ci95_rel_delta"]
                print(
                    f"  [{size} | {name}] {mode:13s} | "
                    f"wQL={summary['raw']['wql_mean']:.4f}±{summary['raw']['wql_std']:.4f} "
                    f"(vs GBT {vs_gbt['obs_rel_delta']:+.1%}, CI=[{ci_lo:+.1%},{ci_hi:+.1%}]) "
                    f"| wink80={summary['raw']['winkler_80_mean']:.4f} | cqr_w80={summary['cqr']['width_80_mean']:.4f} "
                    f"| forg={summary['macro_forgetting_wql_mean']:+.1%} ({summary['elapsed_s_mean']}s)",
                    flush=True,
                )

            report["models"][size][name] = ds_entry
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

            zs_model = copy.deepcopy(base_cpu).to(args.device).eval()
            zs_model.transformer.rope.rebuild_inv_freq()
            zs_fc = TuneTimestralForecaster(zs_model)

        del zs_model, zs_fc, base_cpu
        gc.collect()
        _reset_mem(args.device)

    # Cross-size: fine-tuned 5M vs released 28M on new datasets
    if "5M" in args.sizes and "28M" in args.sizes:
        for name, sp in ds_map.items():
            test_y = np.stack([sp.test[i].target_future[0].astype(np.float64) for i in range(len(sp.test))], axis=0)
            test_ctx = _collect_contexts(sp.test)
            report["cross_size_5m_vs_28m_released"][name] = {}
            for mode in TRAIN_MODES:
                c5 = tuned_5m_cache[(name, mode)]
                z28 = zs_28m_cache[name]
                report["cross_size_5m_vs_28m_released"][name][mode] = {
                    "raw": paired_both_clusters_multiseed(
                        c5["raw"], z28["raw"], test_y, test_ctx, sp.test, sp.seasonality
                    ),
                    "cqr": paired_both_clusters_multiseed(
                        c5["cqr"], z28["cqr"], test_y, test_ctx, sp.test, sp.seasonality
                    ),
                }

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved WP3 report to {args.output}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
