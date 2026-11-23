"""Work Package 1 & 4a: Decision metrics, calibration, cross-size CIs, and 3-seed 5M-209M tables.

    uv run --with scikit-learn==1.6.1 python scripts/benchmarks/run_wp1_decision_metrics.py \
        --device cuda --bf16 --sizes 5M 28M 123M 209M --seeds 0 1 2 \
        --output docs/benchmarks/wp1_decision_metrics.json

Evaluates `retail_demand`, `realised_volatility`, and (for `5M`/`28M`) `realised_volatility_2026_10m`
across all four model sizes, three seeds, all four gradient modes, static CQR, online ACI,
validation-tuned classical Ridge, and validation-tuned GBT, computing Winkler interval scores,
matched-coverage interval widths, 0.80/0.90/0.95 service-level pinball costs, GIFT-Eval forgetting,
and paired cluster bootstrap 95% CIs (by series and by time block).
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from baselines import run_seasonal_naive, run_tuned_classical, run_tuned_gbt
from datasets import (
    BenchmarkSplits,
    load_general_forgetting_by_domain,
    load_realised_volatility,
    load_retail_demand,
)
from decision_metrics import evaluate_with_calibrators, paired_both_clusters_multiseed
from run_benchmarks import _macro_forgetting
from significance import load_realised_volatility_2026_holdout

from timestral_tune import LoraConfig, TimestralTrainer, TrainingConfig, collect_conformal_data
from timestral_tune.config import DEFAULT_LEARNING_RATES
from timestral_tune.models import TuneTimestralForecaster, load_model, merged_state_dict
from timestral_tune.utils.metrics import _collect_contexts

TRAIN_MODES = ("head_only", "lora_variate", "lora_temporal", "full_ft")


def _peak_mem_gib(device: str) -> float:
    if device == "cuda":
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    if device == "mps":
        return float(torch.mps.driver_allocated_memory() / (1024**3))
    return 0.0


def _reset_mem(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        torch.mps.empty_cache()
    gc.collect()


def _train_one_seed(
    base_cpu,
    model_id: str,
    splits: BenchmarkSplits,
    gen_by_domain: dict[str, Any],
    zs_by_domain: dict[str, Any],
    *,
    mode: str,
    device: str,
    precision: str,
    max_steps: int,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
    eval_every: int | None = None,
    early_stopping_patience: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _reset_mem(device)
    t0 = time.perf_counter()
    model = copy.deepcopy(base_cpu)
    lora = (
        LoraConfig(
            r=8,
            lora_alpha=16.0,
            axes=("variate" if mode == "lora_variate" else "time",),
            targets=("q", "k", "v", "o"),
            dropout=0.05,
        )
        if mode.startswith("lora_")
        else None
    )
    cfg = TrainingConfig(
        model_name_or_path=model_id,
        mode=mode,  # type: ignore[arg-type]
        learning_rate=DEFAULT_LEARNING_RATES[mode],  # type: ignore[index]
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        max_steps=max_steps,
        warmup_steps=max(10, min(100, max_steps // 10)),
        mixed_precision=precision,  # type: ignore[arg-type]
        device=device,
        seed=seed,
        eval_every=eval_every or max(20, max_steps // 5),
        early_stopping_patience=early_stopping_patience,
        restore_best=True,
        sign_flip=True,
        log_every=max_steps + 1,
    )
    trainer = TimestralTrainer(cfg, splits.train, splits.val, lora_config=lora, model=model)
    res = trainer.fit()
    elapsed = time.perf_counter() - t0
    peak_gib = _peak_mem_gib(device)
    trainable_p = trainer.num_trainable_parameters

    eval_model = copy.deepcopy(base_cpu)
    eval_model.load_state_dict(merged_state_dict(trainer.model), strict=True)
    del trainer, model
    _reset_mem(device)

    eval_model.to(device).eval()
    eval_model.transformer.rope.rebuild_inv_freq()
    tuned_fc = TuneTimestralForecaster(eval_model)

    val_preds, val_y, val_scale = collect_conformal_data(tuned_fc, splits.val, batch_size=eval_batch_size)
    test_preds, test_y, test_scale = collect_conformal_data(tuned_fc, splits.test, batch_size=eval_batch_size)
    metrics_bundle, preds_dict = evaluate_with_calibrators(
        val_preds,
        val_y,
        val_scale,
        test_preds,
        test_y,
        test_scale,
        splits.test,
        seasonality=splits.seasonality,
        patch_size=tuned_fc.patch_size,
    )
    forg = _macro_forgetting(tuned_fc, gen_by_domain, zs_by_domain, eval_batch_size)
    del eval_model, tuned_fc
    _reset_mem(device)

    row = {
        "seed": seed,
        "learning_rate": DEFAULT_LEARNING_RATES[mode],  # type: ignore[index]
        "batch_size": batch_size,
        "trainable_params": trainable_p,
        "steps": res.steps,
        "best_step": res.best_step,
        "best_eval_loss": res.best_eval_loss,
        "elapsed_s": round(elapsed, 2),
        "peak_vram_gib": round(peak_gib, 3),
        "metrics": metrics_bundle,
        "forgetting": forg,
    }
    return row, preds_dict


def _summarise_seeds(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "elapsed_s_mean": round(float(np.mean([r["elapsed_s"] for r in seed_rows])), 2),
        "peak_vram_gib_max": round(float(np.max([r["peak_vram_gib"] for r in seed_rows])), 3),
        "macro_forgetting_wql_mean": float(np.mean([r["forgetting"]["macro_relative_delta_wql"] for r in seed_rows])),
        "macro_forgetting_wql_worst": float(np.max([r["forgetting"]["macro_relative_delta_wql"] for r in seed_rows])),
        "macro_forgetting_mase_mean": float(np.mean([r["forgetting"]["macro_relative_delta_mase"] for r in seed_rows])),
    }
    for cal_key in ("raw", "cqr", "cqr_all", "aci", "aci_all"):
        cal_sum: dict[str, Any] = {}
        for mk in (
            "wql",
            "mase",
            "pinball",
            "coverage_80",
            "width_80",
            "winkler_80",
            "coverage_90",
            "width_90",
            "winkler_90",
            "pinball_80",
            "pinball_90",
            "pinball_95",
            "fill_rate_80",
            "fill_rate_90",
            "fill_rate_95",
        ):
            vals = [r["metrics"][cal_key][mk] for r in seed_rows]
            cal_sum[f"{mk}_mean"] = float(np.mean(vals))
            cal_sum[f"{mk}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary[cal_key] = cal_sum
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--sizes", nargs="+", default=["5M", "28M", "123M", "209M"], choices=["5M", "28M", "123M", "209M"]
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["retail_demand", "realised_volatility", "realised_volatility_2026_10m"]
    )
    parser.add_argument(
        "--extended-sizes", nargs="+", default=["5M", "28M"], help="Sizes to run on realised_volatility_2026_10m"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip-gbt", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed entries already in --output")
    parser.add_argument(
        "--precomputed-baselines-dir",
        type=Path,
        default=None,
        help="Optional dir holding precomputed baseline_<ds>.json and baseline_gbt_<ds>.npz",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/tt_wp1_preds"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    precision = "bf16" if args.bf16 else "fp32"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("== Loading datasets (retail, post-release vol, 10m 2026 vol, GIFT-Eval) ==", flush=True)
    ds_map: dict[str, BenchmarkSplits] = {}
    if "retail_demand" in args.datasets:
        ds_map["retail_demand"] = load_retail_demand()
    if "realised_volatility" in args.datasets:
        ds_map["realised_volatility"] = load_realised_volatility()
    if "realised_volatility_2026_10m" in args.datasets:
        ds_map["realised_volatility_2026_10m"] = load_realised_volatility_2026_holdout()
    gen_by_domain = load_general_forgetting_by_domain()

    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        report.setdefault("baselines", {})
        report.setdefault("models", {})
        report.setdefault("cross_size_5m_vs_released", {})
        print(f"Resuming from existing {args.output}", flush=True)
    else:
        report = {
            "precision": precision,
            "device": args.device,
            "seeds": args.seeds,
            "datasets": {},
            "baselines": {},
            "models": {},
            "cross_size_5m_vs_released": {},
        }
    for name, sp in ds_map.items():
        report["datasets"][name] = {
            "license": sp.license,
            "source": sp.source,
            "test_date_range": sp.test_date_range,
            "contamination_note": sp.contamination_note,
            "seasonality": sp.seasonality,
            "counts": {"train": len(sp.train), "val": len(sp.val), "test": len(sp.test)},
        }

    # 1. Load precomputed or run non-foundation baselines once per dataset and cache test predictions.
    baseline_preds: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, sp in ds_map.items():
        gbt_npz = args.cache_dir / f"baseline_gbt_{name}.npz"
        pre_json = args.precomputed_baselines_dir / f"baseline_{name}.json" if args.precomputed_baselines_dir else None
        pre_npz = (
            args.precomputed_baselines_dir / f"baseline_gbt_{name}.npz" if args.precomputed_baselines_dir else None
        )
        if pre_json is not None and pre_json.exists() and (args.skip_gbt or (pre_npz is not None and pre_npz.exists())):
            report["baselines"][name] = json.loads(pre_json.read_text(encoding="utf-8"))
            baseline_preds[name] = {}
            if not args.skip_gbt and pre_npz is not None and pre_npz.exists():
                z = np.load(pre_npz)
                baseline_preds[name]["gbt_tuned"] = {"raw": z["raw"], "cqr": z["cqr"]}
            print(f"  [precomputed] loaded baselines for {name} from {pre_json}", flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            continue
        if args.resume and name in report["baselines"] and (args.skip_gbt or "gbt_tuned" in report["baselines"][name]):
            baseline_preds[name] = {}
            if gbt_npz.exists():
                z = np.load(gbt_npz)
                baseline_preds[name]["gbt_tuned"] = {"raw": z["raw"], "cqr": z["cqr"]}
            print(f"  [resume] reusing baselines for {name}", flush=True)
            continue
        print(f"\n== Baselines for {name} (tr={len(sp.train)}, va={len(sp.val)}, te={len(sp.test)}) ==", flush=True)
        sn_res, sn_p = run_seasonal_naive(sp)
        cl_res, cl_p = run_tuned_classical(sp)
        report["baselines"][name] = {"seasonal_naive": sn_res, "classical_domain": cl_res}
        baseline_preds[name] = {"seasonal_naive": sn_p, "classical_domain": cl_p}
        print(
            f"  [baseline | {name}] naive wQL={sn_res['metrics']['raw']['wql']:.4f} | "
            f"classical({cl_res['best_transform']},a={cl_res['best_alpha']}) wQL={cl_res['metrics']['raw']['wql']:.4f} "
            f"wink80={cl_res['metrics']['raw']['winkler_80']:.4f}",
            flush=True,
        )
        if not args.skip_gbt:
            gbt_res, gbt_p = run_tuned_gbt(sp, seed=0)
            report["baselines"][name]["gbt_tuned"] = gbt_res
            baseline_preds[name]["gbt_tuned"] = gbt_p
            np.savez(gbt_npz, raw=gbt_p["raw"], cqr=gbt_p["cqr"])
            print(
                f"  [baseline | {name}] gbt_tuned({gbt_res['best_config']}) "
                f"wQL={gbt_res['metrics']['raw']['wql']:.4f} wink80={gbt_res['metrics']['raw']['winkler_80']:.4f} "
                f"cqr_w80={gbt_res['metrics']['cqr']['width_80']:.4f} ({gbt_res['elapsed_s']}s)",
                flush=True,
            )
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Cache predictions needed for cross-size 5M vs released comparisons.
    zs_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    tuned_5m_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    # 2. Run all model sizes x datasets x modes x seeds.
    for size in args.sizes:
        model_id = f"mistralai/Timestral-{size}"
        report["models"].setdefault(size, {})
        is_large = size in ("123M", "209M")
        max_steps = 160 if is_large else 200
        batch_size = 32
        full_ft_bs = 16 if is_large else 32
        eval_bs = 16 if is_large else 32

        # Determine whether any dataset still needs full training or prediction caching for cross-size CIs.
        ds_to_run = [
            name for name in ds_map if not (name == "realised_volatility_2026_10m" and size not in args.extended_sizes)
        ]
        need_train = [name for name in ds_to_run if not (args.resume and name in report["models"][size])]
        need_zs_cache = [
            name
            for name in ds_to_run
            if size in ("28M", "123M", "209M") and not (args.cache_dir / f"zs_{size}_{name}.npz").exists()
        ]
        need_5m_cache = [
            (name, mode)
            for name in ds_to_run
            for mode in TRAIN_MODES
            if size == "5M" and not (args.cache_dir / f"tuned_5M_{name}_{mode}.npz").exists()
        ]
        if not need_train and not need_zs_cache and not need_5m_cache:
            print(f"  [resume] skipping {size} (all datasets & caches present)", flush=True)
            for name in ds_to_run:
                zp = args.cache_dir / f"zs_{size}_{name}.npz"
                if zp.exists():
                    z = np.load(zp)
                    zs_cache[(size, name)] = {"raw": z["raw"], "cqr": z["cqr"], "aci": z["aci"]}
                if size == "5M":
                    for mode in TRAIN_MODES:
                        tp = args.cache_dir / f"tuned_5M_{name}_{mode}.npz"
                        if tp.exists():
                            z = np.load(tp)
                            tuned_5m_cache[(name, mode)] = {
                                "raw": list(z["raw"]) if z["raw"].ndim == 4 else [z["raw"]],
                                "cqr": list(z["cqr"]) if z["cqr"].ndim == 4 else [z["cqr"]],
                            }
            continue

        base_cpu = load_model(model_id, device="cpu")
        zs_model = copy.deepcopy(base_cpu).to(args.device).eval()
        zs_model.transformer.rope.rebuild_inv_freq()
        zs_fc = TuneTimestralForecaster(zs_model)
        from timestral_tune.utils import evaluate_forecaster

        zs_by_domain = (
            {
                dom: evaluate_forecaster(zs_fc, ds, seasonality=24, batch_size=eval_bs)
                for dom, ds in gen_by_domain.items()
            }
            if need_train or need_5m_cache
            else {}
        )

        for name in ds_to_run:
            sp = ds_map[name]
            zp = args.cache_dir / f"zs_{size}_{name}.npz"
            already_done = args.resume and name in report["models"][size]
            if already_done:
                if zp.exists():
                    z = np.load(zp)
                    zs_cache[(size, name)] = {"raw": z["raw"], "cqr": z["cqr"], "aci": z["aci"]}
                elif size in ("28M", "123M", "209M"):
                    print(f"  [resume] collecting zero-shot cache for {size} | {name}", flush=True)
                    val_p, val_y, val_s = collect_conformal_data(zs_fc, sp.val, batch_size=eval_bs)
                    test_p, test_y, test_s = collect_conformal_data(zs_fc, sp.test, batch_size=eval_bs)
                    _, zs_preds = evaluate_with_calibrators(
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
                    np.savez(zp, raw=zs_preds["raw"], cqr=zs_preds["cqr"], aci=zs_preds["aci"])
                    zs_cache[(size, name)] = {k: zs_preds[k].copy() for k in ("raw", "cqr", "aci")}
                if size == "5M":
                    del zs_model, zs_fc
                    _reset_mem(args.device)
                    for mode in TRAIN_MODES:
                        tp = args.cache_dir / f"tuned_5M_{name}_{mode}.npz"
                        if tp.exists():
                            z = np.load(tp)
                            tuned_5m_cache[(name, mode)] = {
                                "raw": list(z["raw"]) if z["raw"].ndim == 4 else [z["raw"]],
                                "cqr": list(z["cqr"]) if z["cqr"].ndim == 4 else [z["cqr"]],
                            }
                        else:
                            print(f"  [resume] regenerating 5M seed-0 cache for {name} | {mode}", flush=True)
                            _, p_dict = _train_one_seed(
                                base_cpu,
                                model_id,
                                sp,
                                gen_by_domain,
                                zs_by_domain,
                                mode=mode,
                                device=args.device,
                                precision=precision,
                                max_steps=max_steps,
                                batch_size=batch_size,
                                eval_batch_size=eval_bs,
                                seed=args.seeds[0],
                            )
                            np.savez(tp, raw=p_dict["raw"], cqr=p_dict["cqr"])
                            tuned_5m_cache[(name, mode)] = {"raw": p_dict["raw"].copy(), "cqr": p_dict["cqr"].copy()}
                    zs_model = copy.deepcopy(base_cpu).to(args.device).eval()
                    zs_model.transformer.rope.rebuild_inv_freq()
                    zs_fc = TuneTimestralForecaster(zs_model)
                continue

            print(f"\n================ {size} | {name} ================", flush=True)
            val_p, val_y, val_s = collect_conformal_data(zs_fc, sp.val, batch_size=eval_bs)
            test_p, test_y, test_s = collect_conformal_data(zs_fc, sp.test, batch_size=eval_bs)
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
            np.savez(zp, raw=zs_preds["raw"], cqr=zs_preds["cqr"], aci=zs_preds["aci"])
            zs_cache[(size, name)] = {k: zs_preds[k].copy() for k in ("raw", "cqr", "aci")}
            print(
                f"  [{size} | {name}] zero_shot | wQL={zs_metrics['raw']['wql']:.4f} "
                f"wink80={zs_metrics['raw']['winkler_80']:.4f} cov80={zs_metrics['raw']['coverage_80']:.3f} "
                f"cqr_cov80={zs_metrics['cqr']['coverage_80']:.3f} cqr_w80={zs_metrics['cqr']['width_80']:.4f} "
                f"aci_cov80={zs_metrics['aci']['coverage_80']:.3f}",
                flush=True,
            )

            # Free GPU before training loop, then reload zs_fc for next dataset if needed
            ds_entry: dict[str, Any] = {
                "max_steps": max_steps,
                "general_zero_shot_by_domain": {dom: m.to_dict() for dom, m in zs_by_domain.items()},
                "zero_shot": zs_metrics,
                "modes": {},
            }
            del zs_model, zs_fc
            _reset_mem(args.device)

            for mode in TRAIN_MODES:
                run_bs = full_ft_bs if mode == "full_ft" else batch_size
                seed_rows: list[dict[str, Any]] = []
                seed_preds_raw: list[np.ndarray] = []
                seed_preds_cqr: list[np.ndarray] = []
                seed_preds_aci: list[np.ndarray] = []

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
                        max_steps=max_steps,
                        batch_size=run_bs,
                        eval_batch_size=eval_bs,
                        seed=sd,
                    )
                    seed_rows.append(row)
                    seed_preds_raw.append(p_dict["raw"])
                    seed_preds_cqr.append(p_dict["cqr"])
                    seed_preds_aci.append(p_dict["aci"])

                summary = _summarise_seeds(seed_rows)
                if size == "5M":
                    np.savez(
                        args.cache_dir / f"tuned_5M_{name}_{mode}.npz",
                        raw=np.stack(seed_preds_raw, axis=0),
                        cqr=np.stack(seed_preds_cqr, axis=0),
                    )
                    tuned_5m_cache[(name, mode)] = {
                        "raw": list(seed_preds_raw),
                        "cqr": list(seed_preds_cqr),
                    }

                paired_vs_zs = {
                    "raw": paired_both_clusters_multiseed(
                        seed_preds_raw, zs_preds["raw"], test_y, test_ctx, sp.test, sp.seasonality
                    ),
                    "cqr": paired_both_clusters_multiseed(
                        seed_preds_cqr, zs_preds["cqr"], test_y, test_ctx, sp.test, sp.seasonality
                    ),
                    "aci": paired_both_clusters_multiseed(
                        seed_preds_aci, zs_preds["aci"], test_y, test_ctx, sp.test, sp.seasonality
                    ),
                }
                paired_vs_gbt = None
                if "gbt_tuned" in baseline_preds[name]:
                    gp = baseline_preds[name]["gbt_tuned"]
                    paired_vs_gbt = {
                        "raw": paired_both_clusters_multiseed(
                            seed_preds_raw, gp["raw"], test_y, test_ctx, sp.test, sp.seasonality
                        ),
                        "cqr": paired_both_clusters_multiseed(
                            seed_preds_cqr, gp["cqr"], test_y, test_ctx, sp.test, sp.seasonality
                        ),
                    }

                ds_entry["modes"][mode] = {
                    "seeds": seed_rows,
                    "summary": summary,
                    "paired_vs_zero_shot": paired_vs_zs,
                    "paired_vs_gbt": paired_vs_gbt,
                }
                w_rel = paired_vs_zs["cqr"]["by_series"]["width_80"]["obs_rel_delta"]
                wk_rel = paired_vs_zs["cqr"]["by_series"]["winkler_80"]["obs_rel_delta"]
                print(
                    f"  [{size} | {name}] {mode:13s} | "
                    f"wQL={summary['raw']['wql_mean']:.4f}±{summary['raw']['wql_std']:.4f} "
                    f"wink80={summary['raw']['winkler_80_mean']:.4f} | "
                    f"CQR cov80={summary['cqr']['coverage_80_mean']:.3f} width80={w_rel:+.1%} wink80={wk_rel:+.1%} | "
                    f"pb90={summary['raw']['pinball_90_mean']:.4f} | forg={summary['macro_forgetting_wql_mean']:+.1%} "
                    f"({summary['elapsed_s_mean']}s, {summary['peak_vram_gib_max']:.2f} GiB)",
                    flush=True,
                )

            report["models"][size][name] = ds_entry
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

            # Reload zs_fc for next dataset of same size
            zs_model = copy.deepcopy(base_cpu).to(args.device).eval()
            zs_model.transformer.rope.rebuild_inv_freq()
            zs_fc = TuneTimestralForecaster(zs_model)

        del zs_model, zs_fc, base_cpu
        _reset_mem(args.device)

    # 3. Cross-size paired cluster bootstrap: fine-tuned 5M vs each larger size as released.
    if "5M" in args.sizes:
        print("\n== Cross-size paired CIs: fine-tuned 5M vs larger sizes as released ==", flush=True)
        for name, sp in ds_map.items():
            test_y = np.stack([sp.test[i].target_future[0].astype(np.float64) for i in range(len(sp.test))], axis=0)
            test_ctx = _collect_contexts(sp.test)
            report["cross_size_5m_vs_released"][name] = {}
            for target_size in ("28M", "123M", "209M"):
                if (target_size, name) not in zs_cache:
                    continue
                report["cross_size_5m_vs_released"][name][target_size] = {}
                for mode in TRAIN_MODES:
                    if (name, mode) not in tuned_5m_cache:
                        continue
                    c5 = tuned_5m_cache[(name, mode)]
                    zt = zs_cache[(target_size, name)]
                    c5_raw = list(c5["raw"]) if isinstance(c5["raw"], list) else [c5["raw"]]
                    c5_cqr = list(c5["cqr"]) if isinstance(c5["cqr"], list) else [c5["cqr"]]
                    pair = {
                        "raw": paired_both_clusters_multiseed(
                            c5_raw, zt["raw"], test_y, test_ctx, sp.test, sp.seasonality
                        ),
                        "cqr": paired_both_clusters_multiseed(
                            c5_cqr, zt["cqr"], test_y, test_ctx, sp.test, sp.seasonality
                        ),
                    }
                    report["cross_size_5m_vs_released"][name][target_size][mode] = pair
                    dw = pair["raw"]["by_series"]["wql"]
                    print(
                        f"  [5M {mode:13s} vs {target_size:4s} ZS | {name}] "
                        f"ΔwQL={dw['obs_rel_delta']:+.1%} "
                        f"series_CI95=[{dw['ci95_rel_delta'][0]:+.1%}, {dw['ci95_rel_delta'][1]:+.1%}] "
                        f"({dw['clusters_improved']}/{dw['n_clusters']})",
                        flush=True,
                    )

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved WP1 report to {args.output}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
