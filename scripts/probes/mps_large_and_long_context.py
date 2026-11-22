"""Probe: Apple GPU (MPS) coverage for 123M, 209M, and long contexts up to 32,768 steps (spec §5.4, §10).

    uv run python scripts/probes/mps_large_and_long_context.py \
        --output docs/benchmarks/mps_coverage.json

Measures on Apple Silicon (`fp32`, MPS):
1. `123M` and `209M` forward parity vs CPU (`predict_block` univariate + covariate, `forecast`
   with numeric/categorical covariates), training-loss gradient parity vs CPU fp32 (plus float64
   reference on a small batch), and step timings + peak MPS allocated/driver memory across
   `head_only`, `lora_variate`, `lora_temporal`, and `full_ft`.
2. Long-context inference and training (`4,096`, `8,192`, `16,384`, `32,768` steps = `128..1,024`
   patches) on `28M`, `123M`, and `209M`, recording parity vs CPU, latency, peak MPS memory, and
   any backend/memory ceiling cleanly.
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

import torch
import torch.nn.functional as F
from _common import PATCH, load, loss_gradient, make_batch, on_device, reference_gradient, rel

from timestral_tune.config import LoraConfig
from timestral_tune.models import TuneTimestralModel, inject_lora, mps_is_usable
from timestral_tune.trainer import compute_loss


def _mps_sync() -> None:
    torch.mps.synchronize()


def _mps_mem_gib() -> tuple[float, float]:
    cur = torch.mps.current_allocated_memory() / (1024**3)
    drv = torch.mps.driver_allocated_memory() / (1024**3)
    return round(cur, 3), round(drv, 3)


def _clear_mps() -> None:
    gc.collect()
    torch.mps.empty_cache()


def _predict_block(model: TuneTimestralModel, batch):
    return model.predict_block(
        batch.context,
        batch.context_mask,
        batch.group_ids,
        batch.horizon_steps // PATCH,
        batch.role_ids,
        batch.known_future,
        batch.known_future_mask,
    )


def _probe_large_model(size: str, *, seed: int = 7) -> dict[str, Any]:
    print(f"\n== Large-checkpoint MPS probe: Timestral-{size} ==", flush=True)
    out: dict[str, Any] = {"size": size}
    f_cpu = load(size, "cpu")
    f_mps = on_device(f_cpu, "mps")
    m_cpu, m_mps = f_cpu.model, f_mps.model

    # 1. Forward parity vs CPU (ctx=1024, h=96)
    fwd: dict[str, Any] = {}
    for label, covariates in (("predict_block_uni", False), ("predict_block_cov", True)):
        batch = make_batch(2, 3, 32, 3, covariates, seed=seed)
        with torch.no_grad():
            expected = _predict_block(m_cpu, batch)
            _mps_sync()
            t0 = time.perf_counter()
            actual = _predict_block(m_mps, batch.to("mps"))
            _mps_sync()
            ms = (time.perf_counter() - t0) * 1000.0
        err = rel(actual, expected)
        fwd[label] = {"rel_l2_vs_cpu": err, "mps_ms": round(ms, 1)}
        print(f"  [fwd] {label:20s} rel_l2={err:.2e}  ({ms:.1f} ms)", flush=True)

    g = torch.Generator().manual_seed(seed)
    ctx = (torch.randn(2, 1, 1024, generator=g).cumsum(-1) + 50).numpy()
    kwargs = dict(
        past_covariates=torch.randn(2, 2, 1024, generator=g).numpy(),
        known_covariates=torch.randn(2, 1, 1120, generator=g).numpy(),
        categorical_known_covariates=torch.randint(0, 5, (2, 1, 1120), generator=g).double().numpy(),
    )
    exp_fc = f_cpu.forecast(ctx, 96, **kwargs).quantiles
    _mps_sync()
    t0 = time.perf_counter()
    act_fc = f_mps.forecast(ctx, 96, **kwargs).quantiles
    _mps_sync()
    fc_ms = (time.perf_counter() - t0) * 1000.0
    fc_err = rel(act_fc, exp_fc)
    fwd["forecast_96_cov_cat"] = {"rel_l2_vs_cpu": fc_err, "mps_ms": round(fc_ms, 1)}
    print(f"  [fwd] forecast_96_cov_cat  rel_l2={fc_err:.2e}  ({fc_ms:.1f} ms)", flush=True)
    out["forward_parity"] = fwd

    # 2. Gradient parity (small batch: 1 series, 3 rows, ctx=256, h=64) vs CPU fp32 and float64 ref
    grad_batch = make_batch(1, 3, 8, 2, True, seed=seed)
    ref_loss, ref_grad = reference_gradient(m_cpu, grad_batch)
    cpu_loss, cpu_grad = loss_gradient(m_cpu, grad_batch)
    mps_loss, mps_grad = loss_gradient(m_mps, grad_batch.to("mps"))
    cos_mps_ref = float(F.cosine_similarity(mps_grad.unsqueeze(0), ref_grad.unsqueeze(0)).item())
    cos_mps_cpu = float(F.cosine_similarity(mps_grad.unsqueeze(0), cpu_grad.unsqueeze(0)).item())
    out["gradient_parity"] = {
        "ref_loss": ref_loss,
        "cpu_loss": cpu_loss,
        "mps_loss": mps_loss,
        "cpu_rel_l2_vs_f64": rel(cpu_grad, ref_grad),
        "mps_rel_l2_vs_f64": rel(mps_grad, ref_grad),
        "mps_rel_l2_vs_cpu": rel(mps_grad, cpu_grad),
        "mps_one_minus_cos_vs_f64": 1.0 - cos_mps_ref,
        "mps_one_minus_cos_vs_cpu": 1.0 - cos_mps_cpu,
    }
    print(
        f"  [grad] rel_l2 vs f64: cpu={out['gradient_parity']['cpu_rel_l2_vs_f64']:.2e} "
        f"mps={out['gradient_parity']['mps_rel_l2_vs_f64']:.2e} "
        f"(1-cos={out['gradient_parity']['mps_one_minus_cos_vs_f64']:.2e}; "
        f"mps vs cpu={out['gradient_parity']['mps_rel_l2_vs_cpu']:.2e})",
        flush=True,
    )
    del ref_grad, cpu_grad, mps_grad, f_mps, m_mps
    _clear_mps()

    # 3. Training-step benchmark across all four modes on MPS (ctx=256, h=32, cov5, batch=8)
    step_results: dict[str, Any] = {}
    step_batch_cpu = make_batch(8, 5, 8, 1, True, seed=seed)
    for mode in ("head_only", "lora_variate", "lora_temporal", "full_ft"):
        try:
            m = copy.deepcopy(m_cpu).to("mps")
            m.transformer.rope.rebuild_inv_freq()
            if mode == "head_only":
                for p in m.parameters():
                    p.requires_grad_(False)
                for p in m.head.parameters():
                    p.requires_grad_(True)
                params = list(m.head.parameters())
            elif mode in ("lora_variate", "lora_temporal"):
                for p in m.parameters():
                    p.requires_grad_(False)
                axis = "variate" if mode == "lora_variate" else "time"
                inject_lora(m, LoraConfig(r=8, lora_alpha=16.0, axes=(axis,), targets=("q", "k", "v", "o")))
                params = [p for p in m.parameters() if p.requires_grad]
            else:
                for p in m.parameters():
                    p.requires_grad_(True)
                params = list(m.parameters())
            opt = torch.optim.AdamW(params, lr=1e-4)
            b_dev = step_batch_cpu.to("mps")

            # Warmup 1 step, time 3 steps
            def run_step(model=m, optimizer=opt, batch_dev=b_dev, trainable=params) -> float:
                optimizer.zero_grad(set_to_none=True)
                loss = compute_loss(model, batch_dev, validate=False).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                return float(loss.item())

            run_step()
            _mps_sync()
            times = []
            last_loss = 0.0
            for _ in range(3):
                t0 = time.perf_counter()
                last_loss = run_step()
                _mps_sync()
                times.append((time.perf_counter() - t0) * 1000.0)
            cur_gib, drv_gib = _mps_mem_gib()
            step_results[mode] = {
                "ok": True,
                "median_ms": round(sorted(times)[1], 1),
                "allocated_gib": cur_gib,
                "driver_gib": drv_gib,
                "loss": round(last_loss, 5),
            }
            print(
                f"  [step ctx=256 cov5 B=8] {mode:14s} median={sorted(times)[1]:6.1f} ms  "
                f"alloc={cur_gib:.2f} GiB  driver={drv_gib:.2f} GiB",
                flush=True,
            )
            del opt, m, b_dev, params
            _clear_mps()
        except Exception as exc:  # noqa: BLE001
            step_results[mode] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
            print(f"  [step ctx=256 cov5 B=8] {mode:14s} FAILED: {type(exc).__name__}: {exc}", flush=True)
            _clear_mps()
    out["training_steps_ctx256_cov5_b8"] = step_results
    del f_cpu, m_cpu
    _clear_mps()
    return out


def _probe_long_contexts(sizes: tuple[str, ...], contexts: tuple[int, ...], *, seed: int = 7) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for size in sizes:
        print(f"\n== Long-context MPS probe: Timestral-{size} ==", flush=True)
        f_cpu = load(size, "cpu")
        f_mps = on_device(f_cpu, "mps")
        m_cpu, m_mps = f_cpu.model, f_mps.model
        size_res: dict[str, Any] = {}

        for ctx_len in contexts:
            ctx_patches = ctx_len // PATCH
            entry: dict[str, Any] = {"context_steps": ctx_len, "context_patches": ctx_patches}
            batch = make_batch(1, 1, ctx_patches, 3, False, seed=seed)

            # Inference parity vs CPU
            try:
                with torch.no_grad():
                    expected = _predict_block(m_cpu, batch)
                    b_dev = batch.to("mps")
                    _mps_sync()
                    t0 = time.perf_counter()
                    actual = _predict_block(m_mps, b_dev)
                    _mps_sync()
                    ms = (time.perf_counter() - t0) * 1000.0
                    cur_gib, drv_gib = _mps_mem_gib()
                err = rel(actual, expected)
                entry["inference"] = {
                    "ok": True,
                    "rel_l2_vs_cpu": err,
                    "mps_ms": round(ms, 1),
                    "allocated_gib": cur_gib,
                    "driver_gib": drv_gib,
                }
                print(
                    f"  [ctx={ctx_len:5d} inf B=1 uni] rel_l2={err:.2e}  {ms:7.1f} ms  "
                    f"alloc={cur_gib:.2f} GiB  driver={drv_gib:.2f} GiB",
                    flush=True,
                )
                del expected, actual, b_dev
                _clear_mps()
            except Exception as exc:  # noqa: BLE001
                entry["inference"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
                print(f"  [ctx={ctx_len:5d} inf B=1 uni] FAILED: {type(exc).__name__}: {exc}", flush=True)
                _clear_mps()

            # Training step (`lora_temporal`, B=1 univariate) on MPS
            try:
                m_train = copy.deepcopy(m_cpu).to("mps")
                m_train.transformer.rope.rebuild_inv_freq()
                for p in m_train.parameters():
                    p.requires_grad_(False)
                inject_lora(m_train, LoraConfig(r=8, lora_alpha=16.0, axes=("time",), targets=("q", "k", "v", "o")))
                params = [p for p in m_train.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=1e-3)
                b_dev = batch.to("mps")

                opt.zero_grad(set_to_none=True)
                _mps_sync()
                t0 = time.perf_counter()
                loss = compute_loss(m_train, b_dev, validate=False).loss
                loss.backward()
                opt.step()
                _mps_sync()
                ms = (time.perf_counter() - t0) * 1000.0
                cur_gib, drv_gib = _mps_mem_gib()
                entry["train_lora_temporal_b1"] = {
                    "ok": True,
                    "mps_ms": round(ms, 1),
                    "allocated_gib": cur_gib,
                    "driver_gib": drv_gib,
                    "loss": round(float(loss.item()), 5),
                }
                print(
                    f"  [ctx={ctx_len:5d} trn lora_t B=1] {ms:7.1f} ms  "
                    f"alloc={cur_gib:.2f} GiB  driver={drv_gib:.2f} GiB  loss={loss.item():.4f}",
                    flush=True,
                )
                del opt, m_train, b_dev, params, loss
                _clear_mps()
            except Exception as exc:  # noqa: BLE001
                entry["train_lora_temporal_b1"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
                print(f"  [ctx={ctx_len:5d} trn lora_t B=1] FAILED: {type(exc).__name__}: {exc}", flush=True)
                _clear_mps()

            size_res[str(ctx_len)] = entry

        # Also test full_ft at full 32,768-step context (B=1 univariate) and covariate long context (cov3, B=1).
        extra: dict[str, Any] = {}
        try:
            batch_32k = make_batch(1, 1, 32768 // PATCH, 3, False, seed=seed)
            m_ft = copy.deepcopy(m_cpu).to("mps")
            m_ft.transformer.rope.rebuild_inv_freq()
            for p in m_ft.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(list(m_ft.parameters()), lr=1e-4)
            b_dev = batch_32k.to("mps")
            opt.zero_grad(set_to_none=True)
            _mps_sync()
            t0 = time.perf_counter()
            loss = compute_loss(m_ft, b_dev, validate=False).loss
            loss.backward()
            opt.step()
            _mps_sync()
            ms = (time.perf_counter() - t0) * 1000.0
            cur_gib, drv_gib = _mps_mem_gib()
            extra["full_ft_32768_b1_uni"] = {
                "ok": True,
                "mps_ms": round(ms, 1),
                "allocated_gib": cur_gib,
                "driver_gib": drv_gib,
                "loss": round(float(loss.item()), 5),
            }
            print(
                f"  [ctx=32768 trn full_ft B=1] {ms:7.1f} ms  "
                f"alloc={cur_gib:.2f} GiB  driver={drv_gib:.2f} GiB  loss={loss.item():.4f}",
                flush=True,
            )
            del opt, m_ft, b_dev, loss, batch_32k
            _clear_mps()
        except Exception as exc:  # noqa: BLE001
            extra["full_ft_32768_b1_uni"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
            print(f"  [ctx=32768 trn full_ft B=1] FAILED: {type(exc).__name__}: {exc}", flush=True)
            _clear_mps()

        for cov_ctx in (8192, 16384):
            key = f"cov3_{cov_ctx}_b1"
            try:
                b_cov = make_batch(1, 3, cov_ctx // PATCH, 3, True, seed=seed)
                with torch.no_grad():
                    exp_cov = _predict_block(m_cpu, b_cov)
                    b_dev = b_cov.to("mps")
                    _mps_sync()
                    t0 = time.perf_counter()
                    act_cov = _predict_block(m_mps, b_dev)
                    _mps_sync()
                    inf_ms = (time.perf_counter() - t0) * 1000.0
                err_cov = rel(act_cov, exp_cov)
                del exp_cov, act_cov

                m_tr = copy.deepcopy(m_cpu).to("mps")
                m_tr.transformer.rope.rebuild_inv_freq()
                for p in m_tr.parameters():
                    p.requires_grad_(False)
                inject_lora(m_tr, LoraConfig(r=8, lora_alpha=16.0, axes=("time",), targets=("q", "k", "v", "o")))
                opt = torch.optim.AdamW([p for p in m_tr.parameters() if p.requires_grad], lr=1e-3)
                opt.zero_grad(set_to_none=True)
                _mps_sync()
                t0 = time.perf_counter()
                loss = compute_loss(m_tr, b_dev, validate=False).loss
                loss.backward()
                opt.step()
                _mps_sync()
                trn_ms = (time.perf_counter() - t0) * 1000.0
                cur_gib, drv_gib = _mps_mem_gib()
                extra[key] = {
                    "ok": True,
                    "inf_rel_l2_vs_cpu": err_cov,
                    "inf_ms": round(inf_ms, 1),
                    "train_lora_t_ms": round(trn_ms, 1),
                    "allocated_gib": cur_gib,
                    "driver_gib": drv_gib,
                }
                print(
                    f"  [ctx={cov_ctx:5d} cov3 B=1] inf rel_l2={err_cov:.2e} ({inf_ms:.1f} ms) | "
                    f"trn lora_t={trn_ms:.1f} ms  alloc={cur_gib:.2f} GiB  driver={drv_gib:.2f} GiB",
                    flush=True,
                )
                del opt, m_tr, b_dev, loss, b_cov
                _clear_mps()
            except Exception as exc:  # noqa: BLE001
                extra[key] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
                print(f"  [ctx={cov_ctx:5d} cov3 B=1] FAILED: {type(exc).__name__}: {exc}", flush=True)
                _clear_mps()

        size_res["extra"] = extra
        results[size] = size_res
        del f_cpu, f_mps, m_cpu, m_mps
        _clear_mps()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=["123M", "209M"], choices=["28M", "123M", "209M"])
    parser.add_argument("--long-sizes", nargs="+", default=["28M", "123M", "209M"], choices=["28M", "123M", "209M"])
    parser.add_argument("--contexts", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if not mps_is_usable():
        print("Apple GPU (MPS) is not available or PyTorch is below 2.8.", file=sys.stderr)
        return 1

    print(f"== Apple GPU (MPS) Coverage Probe | torch {torch.__version__} ==", flush=True)
    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "device": "mps",
        "precision": "fp32",
        "large_models": {},
        "long_contexts": {},
    }
    for size in args.sizes:
        report["large_models"][size] = _probe_large_model(size, seed=args.seed)
    report["long_contexts"] = _probe_long_contexts(tuple(args.long_sizes), tuple(args.contexts), seed=args.seed)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved MPS coverage report to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
