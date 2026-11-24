"""Unit tests for CUDA/NCCL distributed harness selection, device binding, and runner logic.

Exercises everything that can be verified honestly without a CUDA GPU:
- `TIMESTRAL_TEST_DIST_BACKEND` and `TIMESTRAL_TEST_DIST_DEVICE_MAP` env-var parsing and validation.
- Fail-loud behavior (raising `RuntimeError` / `ValueError`, never silently skipping) when CUDA/NCCL
  is requested with 0 GPUs, too few GPUs for `world_size`, or `shared` without
  `NCCL_MULTI_RANK_GPU_ENABLE=1`.
- `pytest_configure` fail-fast validation (`pytest.UsageError`) when env vars are invalid, and no-op
  when unset.
- Device resolution helpers (`trainer_device_for_mode`, `reference_device_for_mode`,
  `expected_worker_device`, `scale_timeout`).
- `_worker_main` setting and restoring `LOCAL_RANK=str(rank)` on CUDA backends without pre-binding
  `torch.cuda.set_device(...)` in `per-rank` mode (verifying `DistributedContext.detect("auto")`
  performs real `cuda:LOCAL_RANK` binding), versus binding `cuda:0` in `shared` mode, plus
  `torch.cuda.synchronize()` before `destroy_process_group()` on clean exit only.
- `barrier()` and `_scenario_runner` passing explicit `device_ids=[idx]` on NCCL without touching
  GPU 0 on non-zero ranks.
- `run_distributed` backend validation, pre-spawn CUDA prerequisite checks, and crash-grace cleanup.
- `scripts/run_t15_nccl.py` CLI parsing, GPU-count validation, pass plan partitioning, and child
  environment / pytest command construction.
"""

from __future__ import annotations

import importlib.util
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
import torch.distributed as dist

from timestral_tune.trainer.distributed import DistributedContext

from . import conftest as test_conftest
from . import distributed_harness as harness


def _load_runner_module() -> ModuleType:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_t15_nccl.py"
    spec = importlib.util.spec_from_file_location("run_t15_nccl_under_test", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- env-var selection & validation


def test_dist_backend_and_device_map_defaults_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(harness.DIST_BACKEND_ENV, raising=False)
    monkeypatch.delenv(harness.DIST_DEVICE_MAP_ENV, raising=False)

    assert harness.dist_backend() == "gloo"
    assert harness.dist_device_map() == "per-rank"
    assert harness.is_cuda_dist_mode() is False
    assert harness.trainer_device_for_mode() == "cpu"
    assert harness.reference_device_for_mode() == "cpu"
    assert harness.expected_worker_device(0) == torch.device("cpu")
    assert harness.expected_worker_device(3) == torch.device("cpu")
    assert harness.scale_timeout(5.0, 30.0) == 5.0

    # Valid explicit values (with surrounding whitespace).
    monkeypatch.setenv(harness.DIST_BACKEND_ENV, " nccl ")
    assert harness.dist_backend() == "nccl"
    assert harness.is_cuda_dist_mode() is True
    assert harness.dist_device_map() == "per-rank"
    assert harness.trainer_device_for_mode() == "auto"
    assert harness.expected_worker_device(0) == torch.device("cuda", 0)
    assert harness.expected_worker_device(3) == torch.device("cuda", 3)
    assert harness.scale_timeout(5.0, 30.0) == 30.0

    monkeypatch.setenv(harness.DIST_DEVICE_MAP_ENV, " shared ")
    assert harness.dist_device_map() == "shared"
    assert harness.trainer_device_for_mode() == "cuda:0"
    assert harness.expected_worker_device(0) == torch.device("cuda", 0)
    assert harness.expected_worker_device(3) == torch.device("cuda", 0)

    # Invalid backend strings fail loudly (including device-qualified names and empty strings).
    for bad in ("", "   ", "cpu:gloo", "cuda:nccl", "mpi", "NCCL"):
        monkeypatch.setenv(harness.DIST_BACKEND_ENV, bad)
        with pytest.raises(ValueError, match=harness.DIST_BACKEND_ENV):
            harness.dist_backend()

    # Invalid device map strings fail loudly.
    monkeypatch.setenv(harness.DIST_BACKEND_ENV, "nccl")
    for bad_map in ("", "   ", "auto", "single", "PER-RANK"):
        monkeypatch.setenv(harness.DIST_DEVICE_MAP_ENV, bad_map)
        with pytest.raises(ValueError, match=harness.DIST_DEVICE_MAP_ENV):
            harness.dist_device_map()

    # `shared` with CPU/gloo is rejected so it cannot be silently ignored.
    monkeypatch.setenv(harness.DIST_BACKEND_ENV, "gloo")
    monkeypatch.setenv(harness.DIST_DEVICE_MAP_ENV, "shared")
    with pytest.raises(ValueError, match="requires a CUDA/NCCL backend"):
        harness.dist_device_map()


def test_pytest_configure_validates_env_vars_and_is_noop_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(harness.DIST_BACKEND_ENV, raising=False)
    monkeypatch.delenv(harness.DIST_DEVICE_MAP_ENV, raising=False)
    monkeypatch.delenv("NCCL_MULTI_RANK_GPU_ENABLE", raising=False)
    test_conftest.pytest_configure(None)  # type: ignore[arg-type]

    monkeypatch.setenv(harness.DIST_BACKEND_ENV, "bogus")
    with pytest.raises(pytest.UsageError, match=harness.DIST_BACKEND_ENV):
        test_conftest.pytest_configure(None)  # type: ignore[arg-type]

    # `shared` without NCCL_MULTI_RANK_GPU_ENABLE=1 fails fast at configure time.
    monkeypatch.setenv(harness.DIST_BACKEND_ENV, "nccl")
    monkeypatch.setenv(harness.DIST_DEVICE_MAP_ENV, "shared")
    with pytest.raises(pytest.UsageError, match="NCCL_MULTI_RANK_GPU_ENABLE=1"):
        test_conftest.pytest_configure(None)  # type: ignore[arg-type]

    # With NCCL_MULTI_RANK_GPU_ENABLE=1, configure passes (hardware check runs per test).
    monkeypatch.setenv("NCCL_MULTI_RANK_GPU_ENABLE", "1")
    test_conftest.pytest_configure(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- too-few-GPUs & prereq failures


def test_cuda_prereqs_fail_loudly_without_enough_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(harness.DIST_BACKEND_ENV, "nccl")
    monkeypatch.delenv(harness.DIST_DEVICE_MAP_ENV, raising=False)
    monkeypatch.delenv("NCCL_MULTI_RANK_GPU_ENABLE", raising=False)

    # 1. No CUDA available -> RuntimeError (both reference helper and harness prereq check).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="no CUDA GPU is available"):
        harness.reference_device_for_mode()
    with pytest.raises(RuntimeError, match=r"torch\.cuda\.is_available\(\) is False"):
        harness.validate_cuda_harness_prereqs(2)

    # 2. 1 GPU visible in per-rank mode -> world_size=2 fails clearly; reference_device succeeds.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert harness.reference_device_for_mode() == "cuda:0"
    with pytest.raises(RuntimeError, match="requires at least 2 visible CUDA GPUs.*found 1"):
        harness.validate_cuda_harness_prereqs(2)

    # 3. 2 GPUs visible in per-rank mode -> world_size=2 passes; world_size=4 fails clearly.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    harness.validate_cuda_harness_prereqs(2)
    with pytest.raises(RuntimeError, match="requires at least 4 visible CUDA GPUs.*found 2"):
        harness.validate_cuda_harness_prereqs(4)

    # 4. 4 GPUs visible -> both world_size=2 and world_size=4 pass.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    harness.validate_cuda_harness_prereqs(2)
    harness.validate_cuda_harness_prereqs(4)

    # 5. Shared mode requires NCCL_MULTI_RANK_GPU_ENABLE=1 AND >= 1 GPU.
    monkeypatch.setenv(harness.DIST_DEVICE_MAP_ENV, "shared")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="NCCL_MULTI_RANK_GPU_ENABLE=1"):
        harness.validate_cuda_harness_prereqs(4)
    monkeypatch.setenv("NCCL_MULTI_RANK_GPU_ENABLE", "1")
    harness.validate_cuda_harness_prereqs(2)
    harness.validate_cuda_harness_prereqs(4)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="at least 1 visible CUDA GPU"):
        harness.validate_cuda_harness_prereqs(2)


def test_run_distributed_checks_backend_and_cuda_prereqs_before_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    def _dummy(rank: int, world_size: int) -> int:
        return rank + world_size

    def _forbid_spawn(_method: str) -> Any:
        raise AssertionError("mp.get_context('spawn') must not be called when validation fails")

    monkeypatch.setattr(harness.mp, "get_context", _forbid_spawn)

    with pytest.raises(ValueError, match="harness supports backends"):
        harness.run_distributed(_dummy, world_size=2, backend="mpi")

    with pytest.raises(ValueError, match="device_map='shared' requires a CUDA/NCCL backend"):
        harness.run_distributed(_dummy, world_size=2, backend="gloo", device_map="shared")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="requires at least 4 visible CUDA GPUs"):
        harness.run_distributed(_dummy, world_size=4, backend="nccl", device_map="per-rank")


# --------------------------------------------------------------------------- _worker_main & real detect() binding


def test_worker_main_per_rank_leaves_device_binding_to_detect_and_restores_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In `per-rank` CUDA mode, `_worker_main` sets `LOCAL_RANK=str(rank)` without calling
    `torch.cuda.set_device(...)`, so `DistributedContext.detect("auto")` performs the real binding.
    """
    set_device_calls: list[int] = []
    sync_calls: list[int] = []
    destroy_calls: list[int] = []
    init_kwargs: list[dict[str, Any]] = []
    cuda_initialized = False

    def _fake_set_device(idx: int) -> None:
        nonlocal cuda_initialized
        set_device_calls.append(idx)
        cuda_initialized = True

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: cuda_initialized)
    monkeypatch.setattr(torch.cuda, "set_device", _fake_set_device)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(1))
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: init_kwargs.append(kw))
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 2)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(dist, "destroy_process_group", lambda: destroy_calls.append(1))
    monkeypatch.setenv("LOCAL_RANK", "99")

    def _worker_fn(rank: int, world_size: int) -> dict[str, Any]:
        # Before detect(), _worker_main must NOT have pre-bound CUDA in per-rank mode.
        assert set_device_calls == []
        assert os.environ.get("LOCAL_RANK") == str(rank)
        assert harness.dist_backend() == "nccl"
        assert harness.dist_device_map() == "per-rank"
        assert harness.trainer_device_for_mode() == "auto"

        ctx = DistributedContext.detect(harness.trainer_device_for_mode())
        assert ctx is not None
        assert set_device_calls == [rank]
        return {"rank": ctx.rank, "world_size": ctx.world_size, "device": str(ctx.device)}

    res_path = tmp_path / "rank_2.pkl"
    harness._worker_main(2, 4, str(tmp_path / "store"), str(res_path), 10.0, "nccl", _worker_fn, (), "per-rank", False)

    # Outer environment and active-worker state fully restored.
    assert os.environ.get("LOCAL_RANK") == "99"
    assert harness._ACTIVE_WORKER_BACKEND is None
    assert harness._ACTIVE_WORKER_DEVICE_MAP is None
    assert init_kwargs[0]["backend"] == "nccl"
    assert sync_calls == [1]
    assert destroy_calls == [1]

    res = pickle.loads(res_path.read_bytes())
    assert res.ok
    assert res.value == {"rank": 2, "world_size": 4, "device": "cuda:2"}


def test_worker_main_shared_mode_binds_cuda0_and_skips_sync_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_device_calls: list[int] = []
    sync_calls: list[int] = []

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: set_device_calls.append(idx))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(1))
    monkeypatch.setattr(dist, "init_process_group", lambda **_kw: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(dist, "destroy_process_group", lambda: None)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    def _failing_shared_worker(rank: int, world_size: int) -> None:
        del world_size
        assert set_device_calls == [0]
        assert os.environ.get("LOCAL_RANK") == str(rank)
        assert harness.trainer_device_for_mode() == "cuda:0"
        ctx = DistributedContext.detect(harness.trainer_device_for_mode())
        assert ctx is not None and ctx.device == torch.device("cuda", 0)
        raise RuntimeError("boom_in_shared")

    res_path = tmp_path / "shared_r1.pkl"
    harness._worker_main(
        1, 2, str(tmp_path / "store"), str(res_path), 10.0, "nccl", _failing_shared_worker, (), "shared", False
    )

    assert "LOCAL_RANK" not in os.environ
    # On worker exception, synchronize() is skipped so a poisoned CUDA stream cannot mask the error.
    assert sync_calls == []
    res = pickle.loads(res_path.read_bytes())
    assert not res.ok and res.error is not None
    assert res.error.exc_type is RuntimeError and "boom_in_shared" in res.error.message


def test_barrier_and_scenario_runner_pass_device_ids_on_nccl(monkeypatch: pytest.MonkeyPatch) -> None:
    barrier_calls: list[dict[str, Any]] = []
    set_device_calls: list[int] = []
    cuda_init = False

    def _fake_set_device(idx: int) -> None:
        nonlocal cuda_init
        set_device_calls.append(idx)
        cuda_init = True

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(dist, "get_rank", lambda: 3)
    monkeypatch.setattr(dist, "barrier", lambda **kw: barrier_calls.append(kw))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: cuda_init)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "set_device", _fake_set_device)
    monkeypatch.setattr(harness, "_ACTIVE_WORKER_DEVICE_MAP", "per-rank")

    # Scenario fails before CUDA init -> barrier binds rank 3 (never GPU 0) and passes device_ids=[3].
    def _fail_early(rank: int, world_size: int) -> None:
        del rank, world_size
        raise ValueError("early_fail")

    def _ok_second(rank: int, world_size: int) -> int:
        return rank + world_size

    out = harness._scenario_runner(3, 4, [("s1", _fail_early, ()), ("s2", _ok_second, ())])
    assert not out["s1"].ok and out["s2"].ok and out["s2"].value == 7
    assert set_device_calls == [3]
    assert barrier_calls == [{"device_ids": [3]}, {"device_ids": [3]}]

    # In shared mode, uninitialized barrier binds device 0.
    barrier_calls.clear()
    set_device_calls.clear()
    cuda_init = False
    monkeypatch.setattr(harness, "_ACTIVE_WORKER_DEVICE_MAP", "shared")
    harness.barrier()
    assert set_device_calls == [0]
    assert barrier_calls == [{"device_ids": [0]}]

    # On gloo, bare dist.barrier() is called with no device_ids.
    barrier_calls.clear()
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    harness.barrier()
    assert barrier_calls == [{}]


def test_barrier_and_active_dist_device_ignore_stale_cuda0_in_per_rank_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if CUDA is already initialised on device 0 before `detect()`, `per-rank` helpers bind `rank`."""
    current_dev = 0
    set_device_calls: list[int] = []
    barrier_calls: list[dict[str, Any]] = []

    def _fake_set_device(idx: int) -> None:
        nonlocal current_dev
        set_device_calls.append(idx)
        current_dev = idx

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(dist, "get_rank", lambda: 2)
    monkeypatch.setattr(dist, "barrier", lambda **kw: barrier_calls.append(kw))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_dev)
    monkeypatch.setattr(torch.cuda, "set_device", _fake_set_device)
    monkeypatch.setattr(harness, "_ACTIVE_WORKER_DEVICE_MAP", "per-rank")

    # 1. barrier() on rank 2 with stale current_device() == 0 must switch to GPU 2 and pass device_ids=[2].
    harness.barrier()
    assert set_device_calls == [2]
    assert barrier_calls == [{"device_ids": [2]}]

    # 2. active_dist_device() on rank 2 with stale current_device() == 0 must switch to GPU 2 and return cuda:2.
    current_dev = 0
    set_device_calls.clear()
    dev = harness.active_dist_device()
    assert dev == torch.device("cuda", 2)
    assert set_device_calls == [2]

    # 3. In shared mode on rank 2, even if current_device() were 2, both helpers bind GPU 0.
    current_dev = 2
    set_device_calls.clear()
    barrier_calls.clear()
    monkeypatch.setattr(harness, "_ACTIVE_WORKER_DEVICE_MAP", "shared")
    assert harness.active_dist_device() == torch.device("cuda", 0)
    harness.barrier()
    assert set_device_calls == [0]
    assert barrier_calls == [{"device_ids": [0]}]


# --------------------------------------------------------------------------- scripts/run_t15_nccl.py tests


def test_runner_cli_plans_env_and_validation() -> None:
    runner = _load_runner_module()

    cfg = runner.parse_args(["--python", "3.12", "--f32-tol-scale", "0.2", "--log-file", "out.log"])
    assert cfg.ranks == (2, 4)
    assert cfg.shared_gpu is False
    assert cfg.t15_only is False
    assert cfg.f32_tol_scale == pytest.approx(0.2)
    assert cfg.log_file == Path("out.log")
    assert cfg.python_version == "3.12"

    plans = runner.build_pass_plans(cfg)
    assert [p.world_size for p in plans] == [2, 4]
    assert plans[0].files == runner.ALL_MULTIPROC_TEST_FILES
    assert plans[0].k_expr == runner.PASS_2_K_EXPR
    assert plans[1].files == runner.FOUR_RANK_TEST_FILES
    assert plans[1].k_expr == runner.PASS_4_K_EXPR

    cmd2 = runner.build_pytest_command("uv", cfg, plans[0])
    assert cmd2[:7] == ["uv", "run", "--python", "3.12", "pytest", "-v", "-rs"]
    assert runner.PASS_2_K_EXPR in cmd2

    env = runner.build_pass_env(
        {"PATH": "/usr/bin", "TMPDIR": "/a" * 80, "NCCL_MULTI_RANK_GPU_ENABLE": "1"},
        cfg,
    )
    assert env["TIMESTRAL_TEST_DIST_BACKEND"] == "nccl"
    assert env["TIMESTRAL_TEST_DIST_DEVICE_MAP"] == "per-rank"
    assert env["TIMESTRAL_TEST_F32_TOL_SCALE"] == "0.2"
    assert env["TMPDIR"] == "/tmp"
    assert env["PYTHONFAULTHANDLER"] == "1"
    assert "NCCL_MULTI_RANK_GPU_ENABLE" not in env

    # Shared-GPU + --t15-only + --ranks 2
    cfg_shared = runner.parse_args(["--shared-gpu", "--t15-only", "--ranks", "2", "2", "--no-sync"])
    assert cfg_shared.ranks == (2,)
    shared_plans = runner.build_pass_plans(cfg_shared)
    assert len(shared_plans) == 1 and shared_plans[0].files == (runner.T15_TEST_FILE,)
    shared_env = runner.build_pass_env({"TMPDIR": "/tmp"}, cfg_shared)
    assert shared_env["TIMESTRAL_TEST_DIST_DEVICE_MAP"] == "shared"
    assert shared_env["NCCL_MULTI_RANK_GPU_ENABLE"] == "1"
    shared_cmd = runner.build_pytest_command("uv", cfg_shared, shared_plans[0])
    assert "--no-sync" in shared_cmd

    # Probe validation catches missing CUDA and insufficient GPU counts (never skips silently).
    no_cuda = {"cuda_available": False, "gpu_count": 0, "torch": "2.6.0", "cuda_runtime": None}
    assert "cuda_available=False" in (runner.validate_probe_for_config(no_cuda, cfg) or "")

    two_gpus = {"cuda_available": True, "gpu_count": 2, "cuda_visible_devices": "0,1"}
    err_2on4 = runner.validate_probe_for_config(two_gpus, cfg)
    assert err_2on4 is not None and "at least 4" in err_2on4 and "--ranks 2" in err_2on4
    cfg_2 = runner.parse_args(["--ranks", "2"])
    assert runner.validate_probe_for_config(two_gpus, cfg_2) is None

    one_gpu = {"cuda_available": True, "gpu_count": 1, "cuda_visible_devices": "0"}
    assert runner.validate_probe_for_config(one_gpu, cfg_2) is not None
    assert runner.validate_probe_for_config(one_gpu, cfg_shared) is None

    # Probe output parsing.
    parsed = runner.parse_probe_output('some warning\n__TIMESTRAL_CUDA_PROBE__={"gpu_count": 4}\n')
    assert parsed == {"gpu_count": 4}
    with pytest.raises(RuntimeError, match="__TIMESTRAL_CUDA_PROBE__"):
        runner.parse_probe_output("no marker here\n")


def test_runner_main_fails_loudly_without_cuda(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`run_t15_nccl.py` returns exit code 2 and logs the reason when CUDA is absent (never skips)."""
    runner = _load_runner_module()
    log_path = tmp_path / "no_cuda.log"

    class _FakeCompleted:
        returncode = 0
        stdout = (
            '__TIMESTRAL_CUDA_PROBE__={"os": "Linux", "arch": "x86_64", "python": "3.12.0", '
            '"torch": "2.6.0", "cuda_available": false, "cuda_runtime": null, "nccl": null, '
            '"gpu_count": 0, "gpus": [], "cuda_visible_devices": "unset"}\n'
        )
        stderr = ""

    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_kw: _FakeCompleted())

    rc = runner.main(["--log-file", str(log_path)])
    assert rc == 2
    text = log_path.read_text(encoding="utf-8")
    assert "cuda_available=False" in text
    assert "PER-RANK MULTI-GPU MODE" in text


def test_runner_parses_counts_and_rejects_skips_or_zero_passed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Runner parses pytest counts, prints them, and exits non-zero on skips or 0 passed (never silently passes)."""
    runner = _load_runner_module()

    # 1. Clean pass with expected deselections (e.g. --ranks 2).
    ok_out = "\x1b[32m================ 124 passed, 18 deselected in 42.10s =================\x1b[0m\n"
    c_ok = runner.parse_pytest_counts(ok_out)
    assert (c_ok.passed, c_ok.skipped, c_ok.failed, c_ok.errors, c_ok.deselected) == (124, 0, 0, 0, 18)
    assert c_ok.skip_reasons == ()
    assert runner.evaluate_pass_result(0, c_ok) == (True, "PASS")

    # 2. Exit 0 with skipped tests -> FAIL with captured -rs reasons.
    skip_out = (
        "=========================== short test summary info ============================\n"
        "SKIPPED [1] tests/acceptance/test_distributed.py:2950: torch.compile requires a C/C++ compiler\n"
        "SKIPPED [1] tests/acceptance/test_distributed.py:4995: torch.compile requires a C/C++ compiler\n"
        "================== 122 passed, 2 skipped, 18 deselected in 10.0s ==================\n"
    )
    c_skip = runner.parse_pytest_counts(skip_out)
    assert (c_skip.passed, c_skip.skipped, c_skip.deselected) == (122, 2, 18)
    assert len(c_skip.skip_reasons) == 2
    ok_flag, status = runner.evaluate_pass_result(0, c_skip)
    assert ok_flag is False and "skipped=2" in status

    # 3. 0 passed (all skipped or empty run) -> FAIL.
    c_zero = runner.parse_pytest_counts("=================== 0 passed, 2 skipped in 0.10s ===================\n")
    assert runner.evaluate_pass_result(0, c_zero)[0] is False
    c_none = runner.parse_pytest_counts("======================= no tests ran in 0.02s ========================\n")
    assert runner.evaluate_pass_result(5, c_none)[0] is False
    assert runner.evaluate_pass_result(0, c_none)[0] is False

    # 4. End-to-end main() with 4-GPU probe stub: skips -> exit 1; clean pass with deselections -> exit 0.
    class _FourGpuProbe:
        returncode = 0
        stdout = (
            '__TIMESTRAL_CUDA_PROBE__={"os": "Linux", "arch": "x86_64", "python": "3.12.0", '
            '"torch": "2.6.0", "cuda_available": true, "cuda_runtime": "12.4", "nccl": "2.21.5", '
            '"gpu_count": 4, "gpus": ["GPU0", "GPU1", "GPU2", "GPU3"], "cuda_visible_devices": "unset"}\n'
        )
        stderr = ""

    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_kw: _FourGpuProbe())

    log_skip = tmp_path / "skip_run.log"
    monkeypatch.setattr(runner, "_run_streaming", lambda *_a, **_kw: (0, None, skip_out))
    assert runner.main(["--ranks", "2", "--log-file", str(log_skip)]) == 1
    skip_log_text = log_skip.read_text(encoding="utf-8")
    assert "passed=122, skipped=2, failed=0, errors=0, deselected=18" in skip_log_text
    assert "torch.compile requires a C/C++ compiler" in skip_log_text

    log_pass = tmp_path / "pass_run.log"
    monkeypatch.setattr(runner, "_run_streaming", lambda *_a, **_kw: (0, None, ok_out))
    assert runner.main(["--ranks", "2", "--log-file", str(log_pass)]) == 0
    pass_log_text = log_pass.read_text(encoding="utf-8")
    assert "PASS in" in pass_log_text
    assert "passed=124, skipped=0, failed=0, errors=0, deselected=18" in pass_log_text


def _collect_node_ids(
    files: tuple[str, ...], *, k_expr: str | None = None, env_extra: dict[str, str] | None = None
) -> set[str]:
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
    if k_expr is not None:
        cmd.extend(["-k", k_expr])
    cmd.extend(files)
    env = dict(os.environ)
    env.pop(harness.DIST_BACKEND_ENV, None)
    env.pop(harness.DIST_DEVICE_MAP_ENV, None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"pytest --collect-only failed:\n{proc.stdout}\n{proc.stderr}"
    return {line.strip() for line in proc.stdout.splitlines() if "::" in line}


def test_runner_rank_k_expressions_partition_all_142_tests_without_cuda() -> None:
    """Prove the 2-rank and 4-rank `-k` expressions are disjoint, complete, and need no CUDA to collect."""
    runner = _load_runner_module()

    all_ids = _collect_node_ids(runner.ALL_MULTIPROC_TEST_FILES)
    pass2_ids = _collect_node_ids(runner.ALL_MULTIPROC_TEST_FILES, k_expr=runner.PASS_2_K_EXPR)
    pass4_ids = _collect_node_ids(runner.FOUR_RANK_TEST_FILES, k_expr=runner.PASS_4_K_EXPR)

    assert len(all_ids) == 142
    assert pass2_ids.isdisjoint(pass4_ids), f"overlap: {pass2_ids & pass4_ids}"
    assert pass2_ids | pass4_ids == all_ids, f"missed: {all_ids - (pass2_ids | pass4_ids)}"
    assert "tests/acceptance/test_distributed.py::test_t15_f_dataloader_workers" in pass4_ids
    assert len(pass2_ids) == 124 and len(pass4_ids) == 18

    # --t15-only partition covers all 34 T15 items cleanly (18 + 16).
    t15_all = _collect_node_ids((runner.T15_TEST_FILE,))
    t15_2 = _collect_node_ids((runner.T15_TEST_FILE,), k_expr=runner.PASS_2_K_EXPR)
    t15_4 = _collect_node_ids((runner.T15_TEST_FILE,), k_expr=runner.PASS_4_K_EXPR)
    assert len(t15_all) == 34 and len(t15_2) == 18 and len(t15_4) == 16
    assert t15_2.isdisjoint(t15_4) and (t15_2 | t15_4) == t15_all

    # Collection itself succeeds with TIMESTRAL_TEST_DIST_BACKEND=nccl even without a GPU.
    nccl_collected = _collect_node_ids(
        runner.ALL_MULTIPROC_TEST_FILES,
        env_extra={harness.DIST_BACKEND_ENV: "nccl"},
    )
    assert nccl_collected == all_ids


def _worker_report_pgid(rank: int, world_size: int) -> tuple[int, int]:
    del rank, world_size
    return os.getpid(), os.getpgid(0)


def _worker_rank1_crashes_rank0_sleeps(rank: int, world_size: int) -> str:
    del world_size
    if rank == 1:
        os._exit(2)
    time.sleep(60.0)
    return "unreachable"


def _worker_rank0_spawns_grandchild_rank1_crashes(rank: int, world_size: int, pid_file: str) -> str:
    del world_size
    pid_path = Path(pid_file)
    if rank == 0:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        tmp = pid_path.with_suffix(".tmp")
        tmp.write_text(str(child.pid), encoding="utf-8")
        os.replace(tmp, pid_path)
        time.sleep(60.0)
        return "unreachable"
    deadline = time.monotonic() + 10.0
    while not pid_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    os._exit(2)


def test_cpu_gloo_keeps_original_process_group_and_timeout_semantics() -> None:
    """CPU/`gloo` default must not detach workers with `setsid` or change timeout exception type."""
    parent_pgid = os.getpgid(0)
    pgids = harness.unwrap_results(harness.run_distributed(_worker_report_pgid, world_size=2, backend="gloo"))
    for pid, pgid in pgids:
        assert pgid == parent_pgid, f"CPU/gloo worker {pid} unexpectedly detached into PGID {pgid} != {parent_pgid}"

    # On CPU/gloo, if rank 1 exits abnormally while rank 0 sleeps past wall_timeout, original
    # DistributedTimeoutError semantics are preserved (crash grace is CUDA-only).
    with pytest.raises(harness.DistributedTimeoutError, match=r"unfinished ranks: \[0\]"):
        harness.run_distributed(
            _worker_rank1_crashes_rank0_sleeps,
            world_size=2,
            pg_timeout=10.0,
            wall_timeout=2.5,
            backend="gloo",
        )


def test_cuda_crash_grace_and_pgid_teardown_kills_sibling_and_grandchild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In CUDA mode, an abnormally exiting worker triggers 5 s crash-grace PGID teardown killing grandchildren."""
    pid_file = tmp_path / "grandchild.pid"
    # Enable CUDA session/PGID/crash-grace handling in the parent while spawned workers run gloo (no GPU needed).
    monkeypatch.setattr(harness, "is_cuda_backend", lambda _backend: True)
    monkeypatch.setattr(harness, "validate_cuda_harness_prereqs", lambda *_a, **_kw: None)

    t0 = time.monotonic()
    with pytest.raises(
        RuntimeError,
        match=r"worker process failure:.*rank 1: exited with code 2; terminated surviving ranks: \[0\]",
    ):
        harness.run_distributed(
            _worker_rank0_spawns_grandchild_rank1_crashes,
            world_size=2,
            args=(str(pid_file),),
            pg_timeout=30.0,
            wall_timeout=45.0,
            backend="gloo",
            device_map="per-rank",
        )
    elapsed = time.monotonic() - t0
    assert elapsed < 15.0, f"CUDA crash-grace teardown took too long: {elapsed:.2f}s"
    assert pid_file.is_file(), "rank 0 did not start grandchild before crash"
    gc_pid = int(pid_file.read_text(encoding="utf-8").strip())
    # Give the OS a moment to reap the killed grandchild, then verify it is no longer alive.
    reap_deadline = time.monotonic() + 2.0
    alive = True
    while time.monotonic() < reap_deadline:
        try:
            os.kill(gc_pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            alive = False
            break
    if alive:
        try:
            os.kill(gc_pid, 9)
        except ProcessLookupError:
            pass
    assert not alive, f"grandchild PID {gc_pid} survived CUDA PGID teardown"
