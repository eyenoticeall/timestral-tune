"""Work Package 4b/4c: Retail covariate ablation and inference latency/throughput across sizes.

Covariate ablation (L4 CUDA):
    uv run python scripts/benchmarks/run_wp4_breadth_serving.py \
        --run-ablation --device cuda --bf16 --sizes 5M 28M --seeds 0 1 2 \
        --ablation-output docs/benchmarks/wp4_covariate_ablation.json

Inference latency & throughput (L4 CUDA + x86 CPU):
    uv run python scripts/benchmarks/run_wp4_breadth_serving.py \
        --run-latency --latency-devices cuda cpu --sizes 5M 28M 123M 209M \
        --latency-output docs/benchmarks/wp4_inference_latency_cuda_cpu.json

Inference latency & throughput (Apple M4 CPU + MPS):
    uv run python scripts/benchmarks/run_wp4_breadth_serving.py \
        --run-latency --latency-devices mps cpu --sizes 5M 28M 123M 209M \
        --latency-output docs/benchmarks/wp4_inference_latency_apple.json
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_general_forgetting_by_domain, load_retail_demand
from decision_metrics import evaluate_with_calibrators, paired_both_clusters_multiseed
from run_wp1_decision_metrics import _reset_mem, _summarise_seeds, _train_one_seed

from timestral_tune import collect_conformal_data
from timestral_tune.models import TuneTimestralForecaster, load_model
from timestral_tune.utils import evaluate_forecaster
from timestral_tune.utils.metrics import _collect_contexts


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _mem_gib(device: str) -> float:
    if device == "cuda":
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    if device == "mps":
        return float(torch.mps.driver_allocated_memory() / (1024**3))
    return 0.0


def run_covariate_ablation(
    *,
    device: str,
    precision: str,
    sizes: list[str],
    seeds: list[int],
    max_steps: int,
    batch_size: int,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sp_cov = load_retail_demand(include_covariates=True)
    sp_uni = load_retail_demand(include_covariates=False)
    gen_by_domain = load_general_forgetting_by_domain()
    test_y = np.stack([sp_cov.test[i].target_future[0].astype(np.float64) for i in range(len(sp_cov.test))], axis=0)
    test_ctx = _collect_contexts(sp_cov.test)

    report: dict[str, Any] = {
        "precision": precision,
        "device": device,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "seeds": seeds,
        "note": (
            "lora_variate is evaluated with covariates only; on univariate series (1 row) variate "
            "Q/K attention is mathematically inert and rejected by the T10 single-row guard."
        ),
        "models": {},
    }

    for size in sizes:
        model_id = f"mistralai/Timestral-{size}"
        base_cpu = load_model(model_id, device="cpu")
        zs_model = copy.deepcopy(base_cpu).to(device).eval()
        zs_model.transformer.rope.rebuild_inv_freq()
        zs_fc = TuneTimestralForecaster(zs_model)
        zs_by_domain = {
            dom: evaluate_forecaster(zs_fc, ds, seasonality=24, batch_size=batch_size)
            for dom, ds in gen_by_domain.items()
        }

        size_entry: dict[str, Any] = {"with_covariates": {}, "no_covariates": {}, "paired_cov_minus_uni": {}}
        preds_store: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
            "with_covariates": {},
            "no_covariates": {},
        }

        for label, sp, modes in (
            ("with_covariates", sp_cov, ("head_only", "lora_variate", "lora_temporal", "full_ft")),
            ("no_covariates", sp_uni, ("head_only", "lora_temporal", "full_ft")),
        ):
            print(f"\n================ Covariate Ablation: {size} | {label} ================", flush=True)
            zs_m = copy.deepcopy(base_cpu).to(device).eval()
            zs_m.transformer.rope.rebuild_inv_freq()
            fc = TuneTimestralForecaster(zs_m)
            vp, vy, vs = collect_conformal_data(fc, sp.val, batch_size=batch_size)
            tp, ty, ts = collect_conformal_data(fc, sp.test, batch_size=batch_size)
            zs_metrics, zs_preds = evaluate_with_calibrators(
                vp, vy, vs, tp, ty, ts, sp.test, seasonality=sp.seasonality, patch_size=fc.patch_size
            )
            preds_store[label]["zero_shot"] = {"raw": [zs_preds["raw"].copy()], "cqr": [zs_preds["cqr"].copy()]}
            del zs_m, fc
            _reset_mem(device)

            size_entry[label]["zero_shot"] = zs_metrics
            size_entry[label]["modes"] = {}
            print(
                f"  [{size} | {label:15s}] zero_shot     | wQL={zs_metrics['raw']['wql']:.4f} "
                f"wink80={zs_metrics['raw']['winkler_80']:.4f} cqr_w80={zs_metrics['cqr']['width_80']:.4f}",
                flush=True,
            )

            for mode in modes:
                seed_rows: list[dict[str, Any]] = []
                seed_raw: list[np.ndarray] = []
                seed_cqr: list[np.ndarray] = []
                for sd in seeds:
                    row, p_dict = _train_one_seed(
                        base_cpu,
                        model_id,
                        sp,
                        gen_by_domain,
                        zs_by_domain,
                        mode=mode,
                        device=device,
                        precision=precision,
                        max_steps=max_steps,
                        batch_size=batch_size,
                        eval_batch_size=batch_size,
                        seed=sd,
                    )
                    seed_rows.append(row)
                    seed_raw.append(p_dict["raw"].copy())
                    seed_cqr.append(p_dict["cqr"].copy())
                preds_store[label][mode] = {"raw": seed_raw, "cqr": seed_cqr}
                summary = _summarise_seeds(seed_rows)
                size_entry[label]["modes"][mode] = {
                    "seeds": seed_rows,
                    "summary": summary,
                    "paired_vs_zero_shot": {
                        "raw": paired_both_clusters_multiseed(
                            seed_raw, zs_preds["raw"], test_y, test_ctx, sp.test, sp.seasonality
                        ),
                        "cqr": paired_both_clusters_multiseed(
                            seed_cqr, zs_preds["cqr"], test_y, test_ctx, sp.test, sp.seasonality
                        ),
                    },
                }
                print(
                    f"  [{size} | {label:15s}] {mode:13s} | "
                    f"wQL={summary['raw']['wql_mean']:.4f}±{summary['raw']['wql_std']:.4f} "
                    f"wink80={summary['raw']['winkler_80_mean']:.4f} cqr_w80={summary['cqr']['width_80_mean']:.4f} "
                    f"({summary['elapsed_s_mean']}s)",
                    flush=True,
                )

        # Paired delta: with_covariates minus no_covariates across all seeds for every shared mode
        for mode in ("zero_shot", "head_only", "lora_temporal", "full_ft"):
            size_entry["paired_cov_minus_uni"][mode] = {
                "raw": paired_both_clusters_multiseed(
                    preds_store["with_covariates"][mode]["raw"],
                    preds_store["no_covariates"][mode]["raw"],
                    test_y,
                    test_ctx,
                    sp_cov.test,
                    sp_cov.seasonality,
                ),
                "cqr": paired_both_clusters_multiseed(
                    preds_store["with_covariates"][mode]["cqr"],
                    preds_store["no_covariates"][mode]["cqr"],
                    test_y,
                    test_ctx,
                    sp_cov.test,
                    sp_cov.seasonality,
                ),
            }
            dw = size_entry["paired_cov_minus_uni"][mode]["raw"]["by_series"]["wql"]
            print(
                f"  [{size} | cov - uni] {mode:13s} | ΔwQL={dw['obs_rel_delta']:+.1%} "
                f"CI95=[{dw['ci95_rel_delta'][0]:+.1%}, {dw['ci95_rel_delta'][1]:+.1%}] "
                f"({dw['clusters_improved']}/{dw['n_clusters']})",
                flush=True,
            )

        report["models"][size] = size_entry
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        del zs_model, zs_fc, base_cpu
        _reset_mem(device)

    print(f"\nSaved WP4 covariate ablation report to {output}", flush=True)


def run_inference_latency(
    *,
    devices: list[str],
    sizes: list[str],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "os": platform.system(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "results": {},
    }

    rng = np.random.default_rng(0)
    # Workloads: (label, ctx, horizon, n_past, n_known, gpu_batches, cpu_batches)
    workloads = (
        ("ctx256_h48_uni", 256, 48, 0, 0, (1, 16, 64), (1, 16)),
        ("ctx256_h48_cov8", 256, 48, 3, 5, (1, 16, 64), (1, 16)),
        ("ctx512_h96_uni", 512, 96, 0, 0, (1, 16), (1,)),
        ("ctx512_h96_cov8", 512, 96, 3, 5, (1, 16), (1,)),
    )

    for dev in devices:
        precisions = ("bf16", "fp32") if dev == "cuda" else ("fp32",)
        report["results"][dev] = {}
        for prec in precisions:
            report["results"][dev][prec] = {}
            for size in sizes:
                model_id = f"mistralai/Timestral-{size}"
                model = load_model(model_id, device=dev).eval()
                model.transformer.rope.rebuild_inv_freq()
                fc = TuneTimestralForecaster(model)
                size_rows: dict[str, Any] = {}
                print(f"\n== Latency: {dev} | {prec} | {size} ==", flush=True)

                for wlabel, ctx_len, horizon, n_past, n_known, gpu_bs, cpu_bs in workloads:
                    bs_list = cpu_bs if dev == "cpu" else gpu_bs
                    for bs in bs_list:
                        # Measure both symmetric_averaging=True (upstream default) and False at B=1/16 on ctx256
                        sym_opts = (True, False) if (wlabel.startswith("ctx256") and bs in (1, 16)) else (True,)
                        ctx = rng.normal(10.0, 2.0, size=(bs, 1, ctx_len)).astype(np.float32)
                        past = (
                            rng.normal(0.0, 1.0, size=(bs, n_past, ctx_len)).astype(np.float32) if n_past > 0 else None
                        )
                        known = (
                            rng.normal(0.0, 1.0, size=(bs, n_known, ctx_len + horizon)).astype(np.float32)
                            if n_known > 0
                            else None
                        )
                        for sym in sym_opts:
                            n_warm = 2 if dev == "cpu" else 4
                            n_rep = 5 if (dev == "cpu" and size in ("123M", "209M")) else (8 if dev == "cpu" else 15)
                            if dev == "cuda":
                                torch.cuda.reset_peak_memory_stats()

                            def _run_once(
                                _fc=fc,
                                _dev=dev,
                                _prec=prec,
                                _ctx=ctx,
                                _h=horizon,
                                _past=past,
                                _known=known,
                                _sym=sym,
                            ) -> None:
                                with torch.inference_mode():
                                    if _dev == "cuda" and _prec == "bf16":
                                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                            _fc.forecast(
                                                _ctx,
                                                prediction_length=_h,
                                                past_covariates=_past,
                                                known_covariates=_known,
                                                symmetric_averaging=_sym,
                                            )
                                    else:
                                        _fc.forecast(
                                            _ctx,
                                            prediction_length=_h,
                                            past_covariates=_past,
                                            known_covariates=_known,
                                            symmetric_averaging=_sym,
                                        )

                            for _ in range(n_warm):
                                _run_once()
                            _sync(dev)

                            times_ms: list[float] = []
                            for _ in range(n_rep):
                                t0 = time.perf_counter()
                                _run_once()
                                _sync(dev)
                                times_ms.append((time.perf_counter() - t0) * 1000.0)

                            p50 = float(np.median(times_ms))
                            p95 = float(np.quantile(times_ms, 0.95))
                            ms_per_series = p50 / bs
                            series_per_sec = bs / (p50 / 1000.0)
                            mem = _mem_gib(dev)
                            key = f"{wlabel}_b{bs}_sym{int(sym)}"
                            size_rows[key] = {
                                "workload": wlabel,
                                "context_length": ctx_len,
                                "horizon": horizon,
                                "num_covariates": n_past + n_known,
                                "batch_size": bs,
                                "symmetric_averaging": sym,
                                "n_reps": n_rep,
                                "p50_ms": round(p50, 2),
                                "p95_ms": round(p95, 2),
                                "ms_per_series": round(ms_per_series, 2),
                                "series_per_sec": round(series_per_sec, 1),
                                "peak_mem_gib": round(mem, 3),
                            }
                            print(
                                f"  [{dev:4s} {prec:4s} {size:4s}] {key:26s} | "
                                f"p50={p50:7.1f} ms  p95={p95:7.1f} ms  "
                                f"({ms_per_series:6.1f} ms/ser, {series_per_sec:6.1f} ser/s, {mem:.2f} GiB)",
                                flush=True,
                            )

                report["results"][dev][prec][size] = size_rows
                output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                del fc, model
                gc.collect()
                if dev == "cuda":
                    torch.cuda.empty_cache()
                elif dev == "mps":
                    torch.mps.empty_cache()

    print(f"\nSaved WP4 inference latency report to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--run-latency", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--sizes", nargs="+", default=["5M", "28M"], choices=["5M", "28M", "123M", "209M"])
    parser.add_argument("--latency-devices", nargs="+", default=["cuda", "cpu"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ablation-output", type=Path, default=Path("docs/benchmarks/wp4_covariate_ablation.json"))
    parser.add_argument("--latency-output", type=Path, default=Path("docs/benchmarks/wp4_inference_latency.json"))
    args = parser.parse_args()

    if not args.run_ablation and not args.run_latency:
        parser.error("pass --run-ablation and/or --run-latency")

    if args.run_ablation:
        run_covariate_ablation(
            device=args.device,
            precision="bf16" if args.bf16 else "fp32",
            sizes=args.sizes,
            seeds=args.seeds,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            output=args.ablation_output,
        )
    if args.run_latency:
        run_inference_latency(
            devices=args.latency_devices,
            sizes=args.sizes,
            output=args.latency_output,
        )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
