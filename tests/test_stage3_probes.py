"""Unit tests for Stage 3 calibration probe, scaling benchmark, linear dispatch CUDA option, and bundle runner.

Verifies everything that can be tested honestly without a CUDA GPU:
- Exact float64 L2, cancellation-free cosine distance (`0.5 * ||u - v||^2`), sign-flip fraction,
  and worst-case calibration summarization in `scripts/probes/distributed_calibration_probe.py`.
- Spec generation, weak/strong per-rank batch resolution, and speedup/efficiency/D22 summary math
  in `scripts/probes/distributed_scaling_bench.py`.
- CLI parsing, step planning, repeat-determinism driver-log parsing/diffing, and `--only`/`--skip`/
  resume semantics in `scripts/run_stage3_bundle.py` (including Python 3.10 AST compatibility).
- Fail-loud exit code 2 (never silent skip) when CUDA is requested without GPUs across all probes.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(rel_path: str, mod_name: str, *, add_probes_dir: bool = False) -> ModuleType:
    probes_dir = str(REPO_ROOT / "scripts" / "probes")
    added = False
    if add_probes_dir and probes_dir not in sys.path:
        sys.path.insert(0, probes_dir)
        added = True
    try:
        spec = importlib.util.spec_from_file_location(mod_name, REPO_ROOT / rel_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        if added and probes_dir in sys.path:
            sys.path.remove(probes_dir)


# --------------------------------------------------------------------------- calibration probe metrics & CLI


def test_calibration_probe_metrics_and_summary() -> None:
    calib = _load_script(
        "scripts/probes/distributed_calibration_probe.py",
        "stage3_calib_under_test",
        add_probes_dir=True,
    )

    # Identical vectors -> 0.0 across all error metrics.
    a = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    assert calib.rel_l2_f64(a, a) == 0.0
    assert calib.cosine_distance_f64(a, a) == 0.0
    assert calib.sign_flip_fraction(a, a) == 0.0

    # Orthogonal vectors -> 1 - cos = 1.0; opposite vectors -> 1 - cos = 2.0 and 100% sign flips.
    e1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    e2 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    assert calib.cosine_distance_f64(e1, e2) == pytest.approx(1.0, abs=1e-15)
    assert calib.cosine_distance_f64(a, -a) == pytest.approx(2.0, abs=1e-15)
    assert calib.sign_flip_fraction(a, -a) == 1.0

    # Small perturbation: 0.5 * ||u - v||^2 resolves tiny angles cleanly below float64 1e-16 floor.
    eps = 1e-10
    v1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    v2 = torch.tensor([1.0, eps], dtype=torch.float64)
    # 1 - cos(theta) = 0.5 * theta^2 + O(theta^4) = 0.5e-20
    assert calib.cosine_distance_f64(v1, v2) == pytest.approx(0.5 * eps * eps, rel=1e-6)

    # Non-finite values propagate NaN (never silently 0).
    nan_t = torch.tensor([1.0, float("nan")], dtype=torch.float64)
    assert math.isnan(calib.rel_l2_f64(nan_t, nan_t))
    assert math.isnan(calib.cosine_distance_f64(nan_t, nan_t))
    assert math.isnan(calib.sign_flip_fraction(nan_t, nan_t))

    # Sign-flip fraction ignores zero entries and counts opposite-sign pairs accurately.
    x = torch.tensor([1.0, -2.0, 0.0, 4.0], dtype=torch.float64)
    y = torch.tensor([-1.0, -3.0, 5.0, 2.0], dtype=torch.float64)
    # Active coords where both non-zero: idx 0 (flip), idx 1 (same), idx 3 (same) -> 1/3.
    assert calib.sign_flip_fraction(x, y) == pytest.approx(1.0 / 3.0)

    # Summarization aggregates worst-case maxima and xr_eq counts per (precision, size, mode).
    rows = [
        {
            "precision": "bf16",
            "size": "5M",
            "mode": "full_ft",
            "xr_eq": True,
            "step1_grad_l2": 0.1,
            "step1_1m_grad_cos": 0.01,
            "param_l2": 1e-4,
            "upd_l2": 0.4,
            "1m_upd_cos": 0.05,
            "loss_rel": 1e-2,
            "val_rel": 2e-3,
        },
        {
            "precision": "bf16",
            "size": "5M",
            "mode": "full_ft",
            "xr_eq": True,
            "step1_grad_l2": 0.25,
            "step1_1m_grad_cos": 0.03,
            "param_l2": 2e-4,
            "upd_l2": 0.6,
            "1m_upd_cos": 0.08,
            "loss_rel": 3e-2,
            "val_rel": 5e-3,
        },
    ]
    summary = calib.summarize_calibration_rows(rows)
    assert len(summary) == 1
    assert summary[0]["count"] == 2 and summary[0]["xr_eq_count"] == 2
    assert summary[0]["max_upd_l2"] == pytest.approx(0.6)
    assert summary[0]["max_1m_upd_cos"] == pytest.approx(0.08)


def test_calibration_and_scaling_probes_fail_loudly_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    calib = _load_script(
        "scripts/probes/distributed_calibration_probe.py",
        "stage3_calib_nocuda",
        add_probes_dir=True,
    )
    scale = _load_script(
        "scripts/probes/distributed_scaling_bench.py",
        "stage3_scale_nocuda",
        add_probes_dir=True,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert calib.main(["--backend", "nccl", "--sizes", "tiny", "--ranks", "2"]) == 2
    assert scale.main(["--backend", "nccl", "--sizes", "tiny", "--ranks", "1", "2"]) == 2

    # linear_requires_grad_dispatch --device cuda also exits 2 when CUDA is unavailable.
    if not torch.cuda.is_available():
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/probes/linear_requires_grad_dispatch.py"), "--device", "cuda"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 2
        assert "torch.cuda.is_available() is False" in proc.stderr


# --------------------------------------------------------------------------- scaling benchmark math & specs


def test_scaling_bench_specs_batch_resolution_and_summary() -> None:
    scale = _load_script(
        "scripts/probes/distributed_scaling_bench.py",
        "stage3_scale_under_test",
        add_probes_dir=True,
    )

    cuda_specs = scale.build_bench_specs(["28M", "209M"], backend="nccl")
    keys = {s.key for s in cuda_specs}
    # Covers short-context weak/strong (eager + compile), full_ft distill, and 209M long-context 1024 & 4096.
    assert "28M_full_ft_eager_ctx256_weak_b16" in keys
    assert "28M_full_ft_compile_ctx256_strong_g32" in keys
    assert "209M_full_ft_distill_eager_ctx256_weak_b16" in keys
    assert "209M_full_ft_eager_ctx1024_weak_b8" in keys
    assert "209M_lora_temporal_eager_ctx4096_weak_b2" in keys

    weak_spec = next(s for s in cuda_specs if s.key == "28M_full_ft_eager_ctx256_weak_b16")
    strong_spec = next(s for s in cuda_specs if s.key == "28M_full_ft_compile_ctx256_strong_g32")
    assert [scale.resolve_per_rank_batch(weak_spec, ws) for ws in (1, 2, 4)] == [16, 16, 16]
    assert [scale.resolve_per_rank_batch(strong_spec, ws) for ws in (1, 2, 4)] == [32, 16, 8]
    assert scale.resolve_per_rank_batch(strong_spec, 3) is None

    # Weak and strong scaling speedup / efficiency math.
    raw_rows = [
        {
            "key": "w",
            "world_size": 1,
            "per_rank_batch": 16,
            "effective_batch": 16,
            "spec": {
                "scaling": "weak",
                "size": "209M",
                "mode": "full_ft",
                "compile_model": False,
                "with_distill": False,
                "ctx_len": 256,
            },
            "mean_step_ms": 100.0,
            "median_step_ms": 100.0,
            "windows_per_s": 160.0,
            "train_vram_mib": 5000.0,
            "val_vram_mib": 5100.0,
            "val_vram_delta_mib": 100.0,
            "flat_buf_mib": 0.0,
            "breakdown": {},
        },
        {
            "key": "w",
            "world_size": 4,
            "per_rank_batch": 16,
            "effective_batch": 64,
            "spec": {
                "scaling": "weak",
                "size": "209M",
                "mode": "full_ft",
                "compile_model": False,
                "with_distill": False,
                "ctx_len": 256,
            },
            "mean_step_ms": 110.0,
            "median_step_ms": 110.0,
            "windows_per_s": 64.0 / 0.11,
            "train_vram_mib": 5800.0,
            "val_vram_mib": 5900.0,
            "val_vram_delta_mib": 100.0,
            "flat_buf_mib": 798.5,
            "breakdown": {
                "precheck_ms": 0.0,
                "stats_ar_ms": 0.2,
                "pack_ms": 1.0,
                "grad_ar_ms": 8.0,
                "unpack_ms": 1.0,
                "bcast_ms": 0.1,
                "grad_ar_pct": 7.27,
                "comm_overhead_pct": 9.36,
            },
        },
        {
            "key": "s",
            "world_size": 1,
            "per_rank_batch": 32,
            "effective_batch": 32,
            "spec": {
                "scaling": "strong",
                "size": "209M",
                "mode": "full_ft",
                "compile_model": False,
                "with_distill": False,
                "ctx_len": 256,
            },
            "mean_step_ms": 200.0,
            "median_step_ms": 200.0,
            "windows_per_s": 160.0,
            "train_vram_mib": 6000.0,
            "flat_buf_mib": 0.0,
            "breakdown": {},
        },
        {
            "key": "s",
            "world_size": 4,
            "per_rank_batch": 8,
            "effective_batch": 32,
            "spec": {
                "scaling": "strong",
                "size": "209M",
                "mode": "full_ft",
                "compile_model": False,
                "with_distill": False,
                "ctx_len": 256,
            },
            "mean_step_ms": 60.0,
            "median_step_ms": 60.0,
            "windows_per_s": 32.0 / 0.06,
            "train_vram_mib": 4200.0,
            "flat_buf_mib": 798.5,
            "breakdown": {
                "precheck_ms": 0.0,
                "stats_ar_ms": 0.2,
                "pack_ms": 1.0,
                "grad_ar_ms": 8.0,
                "unpack_ms": 1.0,
                "bcast_ms": 0.1,
                "grad_ar_pct": 13.3,
                "comm_overhead_pct": 17.2,
            },
        },
    ]
    summary = scale.compute_scaling_summary(raw_rows)
    w4 = next(r for r in summary if r["key"] == "w" and r["world_size"] == 4)
    s4 = next(r for r in summary if r["key"] == "s" and r["world_size"] == 4)
    assert w4["speedup_vs_1"] == pytest.approx((64.0 / 0.11) / 160.0, rel=1e-6)
    assert w4["efficiency_pct"] == pytest.approx(100.0 * (100.0 / 110.0), rel=1e-6)
    assert w4["small_coll_ms"] == pytest.approx(0.3)
    assert s4["speedup_vs_1"] == pytest.approx(200.0 / 60.0, rel=1e-6)
    assert s4["efficiency_pct"] == pytest.approx(100.0 * (200.0 / 60.0) / 4.0, rel=1e-6)


# --------------------------------------------------------------------------- bundle runner tests


def test_stage3_bundle_runner_plans_ast_and_repeat_diff(tmp_path: Path) -> None:
    # Verify Python 3.10 syntax compatibility on stdlib-only bundle runner.
    bundle_src = (REPO_ROOT / "scripts/run_stage3_bundle.py").read_text(encoding="utf-8")
    ast.parse(bundle_src, feature_version=(3, 10))

    bundle = _load_script("scripts/run_stage3_bundle.py", "stage3_bundle_under_test")
    cfg = bundle.parse_args(["--python", "3.12", "--out-dir", str(tmp_path / "out"), "--include-cuda-check"])
    steps = bundle.build_bundle_steps("/usr/bin/uv", cfg)
    names = [s.name for s in steps]
    assert names == list(bundle.ALL_STEP_NAMES)
    # 00_host_preflight uses --_nccl-transport-check with torchrun (never invalid `-c`).
    preflight_cmds = steps[0].commands
    assert len(preflight_cmds) == 2
    assert "torchrun" in preflight_cmds[1] and "--_nccl-transport-check" in preflight_cmds[1]
    assert "-c" not in preflight_cmds[1]
    # 04_multi_gpu_check_r2 wires repeat diff post-action.
    step4 = next(s for s in steps if s.name == "04_multi_gpu_check_r2")
    assert step4.post_action == "diff_mgc_repeats"

    # --only and --skip filtering.
    cfg_sub = bundle.parse_args(["--only", "01_t15_nccl", "05_calibration_probe", "--skip", "01_t15_nccl"])
    sub_steps = bundle.build_bundle_steps("/usr/bin/uv", cfg_sub)
    assert [s.name for s in sub_steps] == ["05_calibration_probe"]

    # Repeat determinism diff detects matching vs differing rows accurately.
    r1 = tmp_path / "r1"
    r2 = tmp_path / "r2"
    r1.mkdir()
    r2.mkdir()
    header = (
        "model        scenario                        xr_eq xr_max   param_l2  "
        "upd_l2    loss_rel  val_rel   wall_s  step_ms tail_s  vram_mib notes\n"
    )
    row_same = (
        "tiny_random  domain_head_only_f32            yes   0.00e+00 1.23e-07  "
        "4.56e-06  1.00e-07  2.00e-07  1.01    12.3    0.05    120.0\n"
    )
    row_c1 = (
        "tiny_random  compile_default_lora_variate    yes   0.00e+00 3.00e-06  "
        "1.10e-04  5.00e-06  -         2.50    15.0    0.06    130.0    graphs=s1:1->end:2\n"
    )
    row_c2 = (
        "tiny_random  compile_default_lora_variate    yes   0.00e+00 3.20e-06  "
        "1.45e-04  5.50e-06  -         2.40    14.8    0.06    130.0    graphs=s1:1->end:2\n"
    )
    (r1 / "04_nccl_tiny_n4.log").write_text(header + row_same + row_c1, encoding="utf-8")
    (r2 / "04_nccl_tiny_n4.log").write_text(header + row_same + row_c2, encoding="utf-8")
    diff_out = tmp_path / "diff.txt"
    report = bundle.diff_mgc_repeats(r1, r2, diff_out)
    assert "04_nccl_tiny_n4.log: matched=1/2 diffs=1 missing=0" in report
    assert "DIFF tiny_random compile_default_lora_variate" in report
    assert "TOTAL: identical=1/2, differed=1" in report
