"""Multi-process `TimestralTrainer` tests (startup, consistency, guards, export, fit).

Grouped into small spawned `gloo` process groups via `tests.distributed_harness.run_scenarios`
with short timeouts so every hang fails promptly and the suite stays fast.

- Spec: `docs/SPEC.md` §5.6, §7.4, §9 (T15), §11 (D22–D35)
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import logging
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist

from timestral_tune import (
    ConformalCalibrator,
    LoraConfig,
    ReplayConfig,
    TimestralTrainer,
    TrainingConfig,
)
from timestral_tune import export as export_module
from timestral_tune.data import SeriesData, TimeSeriesDataset
from timestral_tune.data.batch import flip_batch_signs
from timestral_tune.models import TuneTimestralModel
from timestral_tune.trainer import Callback, RankFailureError
from timestral_tune.trainer import __all__ as trainer_all
from timestral_tune.trainer.distributed import DistributedContext, steps_per_epoch
from timestral_tune.trainer.step import compute_loss

from .distributed_harness import run_distributed, run_scenarios, unwrap_results
from .helpers import TINY_CONFIG, as_double, float64_reference, l2_rel, make_tiny_forecaster, make_toy_dataset

WORKERS_HANG_AT_EXIT = sys.platform == "darwin" and tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


class _RecordHandler(logging.Handler):
    """In-worker log collector returning `(levelno, message)` tuples."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelno, record.getMessage()))


def _capture_trainer_logs() -> tuple[logging.Logger, _RecordHandler, int]:
    tlog = logging.getLogger("timestral_tune.trainer")
    prev_level = tlog.level
    handler = _RecordHandler()
    tlog.setLevel(logging.DEBUG)
    tlog.addHandler(handler)
    return tlog, handler, prev_level


def _release_trainer_logs(tlog: logging.Logger, handler: _RecordHandler, prev_level: int) -> None:
    tlog.removeHandler(handler)
    tlog.setLevel(prev_level)


def _followup_sum(ctx: DistributedContext) -> int:
    probe = torch.tensor([1], dtype=torch.int64)
    ctx.all_reduce_sum(probe)
    return int(probe.item())


def _base_config(**overrides: Any) -> TrainingConfig:
    defaults: dict[str, Any] = {
        "model_name_or_path": "local-test",
        "mode": "head_only",
        "device": "cpu",
        "batch_size": 2,
        "epochs": 2,
        "learning_rate": 1e-3,
        "shuffle": False,
        "sign_flip": False,
        "restore_best": False,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _fresh_model(seed: int) -> TuneTimestralModel:
    return copy.deepcopy(make_tiny_forecaster(seed=seed).model)


# --------------------------------------------------------------------------- in-process surface


def test_rank_failure_error_exported_from_trainer_package() -> None:
    assert "RankFailureError" in trainer_all
    assert issubclass(RankFailureError, RuntimeError)


# --------------------------------------------------------------------------- startup, guards, export scenarios


def _sc_startup_broadcast_all_modes(rank: int, world_size: int) -> dict[str, Any]:
    ds = make_toy_dataset(TINY_CONFIG.patch_size, series=4, seed=7, covariates=True)
    out: dict[str, Any] = {}
    for mode in ("head_only", "lora_variate", "full_ft"):
        # Give each rank completely different starting weights (trainable + frozen) and a custom buffer.
        model = _fresh_model(seed=100 + rank)
        model.register_buffer("rank_sentinel", torch.full((3,), float(rank + 10), dtype=torch.float32))
        # Different per-rank output_dir must be excluded from startup consistency digest.
        cfg = _base_config(mode=mode, batch_size=2, epochs=3, output_dir=f"/tmp/rank-out-{rank}")
        lora_cfg = LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("q", "v")) if "lora" in mode else None
        trainer = TimestralTrainer(cfg, ds, lora_config=lora_cfg, model=model)
        params_snap = {name: p.detach().clone().tolist() for name, p in sorted(trainer.model.named_parameters())}
        bufs_snap = {name: b.detach().clone().tolist() for name, b in sorted(trainer.model.named_buffers())}
        out[mode] = {
            "params": params_snap,
            "buffers": bufs_snap,
            "total_steps": trainer.total_steps,
            "expected_total_steps": 3 * steps_per_epoch(len(ds), world_size, 2),
        }
    return out


def _sc_consistency_mismatches(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect()
    assert ctx is not None
    ds = make_toy_dataset(TINY_CONFIG.patch_size, series=4, seed=3, covariates=True)
    train_ds, val_ds = ds.split_by_time(12 * TINY_CONFIG.patch_size)
    out: dict[str, Any] = {}

    # 1. Different dataset lengths.
    keep = np.ones(len(ds), dtype=bool)
    if rank == 1:
        keep[-1] = False
    rank_ds = ds._with_windows(keep)
    try:
        TimestralTrainer(_base_config(), rank_ds, model=_fresh_model(0))
        out["dataset_len"] = {"raised": False}
    except Exception as exc:
        out["dataset_len"] = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "followup": _followup_sum(ctx),
        }

    # 2. Different TrainingConfig field (seed).
    cfg_seed = _base_config(seed=99 if rank == world_size - 1 else 0)
    try:
        TimestralTrainer(cfg_seed, ds, model=_fresh_model(0))
        out["config_seed"] = {"raised": False}
    except Exception as exc:
        out["config_seed"] = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "followup": _followup_sum(ctx),
        }

    # 3. Different model architecture (d_model).
    alt_model = _fresh_model(0)
    if rank == 1:
        alt_cfg = dataclasses.replace(TINY_CONFIG, d_model=16)
        alt_model = TuneTimestralModel(alt_cfg)
    try:
        TimestralTrainer(_base_config(), ds, model=alt_model)
        out["model_shape"] = {"raised": False}
    except Exception as exc:
        out["model_shape"] = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "followup": _followup_sum(ctx),
        }

    # 4. Different val_dataset presence.
    rank_val = val_ds if rank == 0 else None
    try:
        TimestralTrainer(_base_config(), train_ds, rank_val, model=_fresh_model(0))
        out["val_presence"] = {"raised": False}
    except Exception as exc:
        out["val_presence"] = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "followup": _followup_sum(ctx),
        }

    return out


def _sc_startup_failure_propagation(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect()
    assert ctx is not None
    ds = make_toy_dataset(TINY_CONFIG.patch_size, series=4, seed=5, covariates=True)
    out: dict[str, Any] = {}

    # 1. Rank 0 fails in step (a) (early_stopping_patience without val_dataset).
    cfg_r0 = _base_config(early_stopping_patience=2 if rank == 0 else None)
    try:
        TimestralTrainer(cfg_r0, ds, model=_fresh_model(0))
        out["fail_r0"] = {"raised": False}
    except Exception as exc:
        out["fail_r0"] = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "failing_ranks": getattr(exc, "failing_ranks", None),
            "stage": getattr(exc, "stage", None),
            "followup": _followup_sum(ctx),
        }

    # 2. Non-zero rank (rank 1) fails in step (a) (invalid callback type).
    cbs: tuple[Any, ...] = (object(),) if rank == 1 else ()
    try:
        TimestralTrainer(_base_config(), ds, callbacks=cbs, model=_fresh_model(0))
        out["fail_r1"] = {"raised": False}
    except Exception as exc:
        out["fail_r1"] = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "failing_ranks": getattr(exc, "failing_ranks", None),
            "stage": getattr(exc, "stage", None),
            "followup": _followup_sum(ctx),
        }

    # 3. Two ranks fail simultaneously in step (a).
    pair = (0, 1) if world_size == 2 else (1, 3)
    bad_lora: Any = None
    bad_model: Any = _fresh_model(0)
    if rank == pair[0]:
        bad_lora = LoraConfig()  # invalid for head_only -> ValueError
    elif rank == pair[1]:
        bad_model = object()  # wrong type -> TypeError
    try:
        TimestralTrainer(_base_config(), ds, lora_config=bad_lora, model=bad_model)
        out["fail_two"] = {"raised": False}
    except Exception as exc:
        out["fail_two"] = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "failing_ranks": getattr(exc, "failing_ranks", None),
            "stage": getattr(exc, "stage", None),
            "followup": _followup_sum(ctx),
        }

    # 4. Mixed trainable parameter dtypes on rank 0 rejected in step (a) and propagated.
    mixed_model = _fresh_model(0)
    if rank == 0:
        first_head_param = next(mixed_model.head.parameters())
        first_head_param.data = first_head_param.data.to(torch.float64)
    try:
        TimestralTrainer(_base_config(mode="head_only"), ds, model=mixed_model)
        out["mixed_dtype"] = {"raised": False}
    except Exception as exc:
        out["mixed_dtype"] = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "failing_ranks": getattr(exc, "failing_ranks", None),
            "stage": getattr(exc, "stage", None),
            "followup": _followup_sum(ctx),
        }

    return out


def _sc_stage1_refusals(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect()
    assert ctx is not None
    ds = make_toy_dataset(TINY_CONFIG.patch_size, series=4, seed=9, covariates=True)
    train_ds, val_ds = ds.split_by_time(12 * TINY_CONFIG.patch_size)
    out: dict[str, Any] = {}

    # 1. compile_model=True + compile_mode='reduce-overhead' -> ValueError.
    try:
        TimestralTrainer(_base_config(compile_model=True, compile_mode="reduce-overhead"), ds, model=_fresh_model(0))
        out["reduce_overhead_on"] = None
    except Exception as exc:
        out["reduce_overhead_on"] = (type(exc), str(exc))

    # 2. compile_model=False + compile_mode='reduce-overhead' -> allowed.
    trainer_ok = TimestralTrainer(
        _base_config(compile_model=False, compile_mode="reduce-overhead"), ds, model=_fresh_model(0)
    )
    out["reduce_overhead_off_ok"] = trainer_ok.total_steps > 0

    # 3. Replay type/value checks run BEFORE Stage 1 NotImplementedError.
    replay_cases: list[tuple[str, dict[str, Any]]] = [
        ("bad_replay_cfg_type", {"replay_config": "bad"}),
        ("replay_ds_without_cfg", {"replay_config": None, "replay_dataset": train_ds}),
        ("bad_replay_ds_type", {"replay_config": ReplayConfig(), "replay_dataset": "bad"}),
        ("replay_ds_is_val_ds", {"replay_config": ReplayConfig(), "replay_dataset": val_ds}),
        ("valid_replay_cfg", {"replay_config": ReplayConfig(), "replay_dataset": None}),
    ]
    for label, kwargs in replay_cases:
        try:
            TimestralTrainer(_base_config(), train_ds, val_ds, model=_fresh_model(0), **kwargs)
            out[label] = None
        except Exception as exc:
            out[label] = (type(exc), str(exc))

    # 4. Pre-collective device/backend mismatch raises ValueError immediately (gloo rejects cuda/mps).
    for bad_dev in ("cuda", "mps"):
        try:
            TimestralTrainer(_base_config(device=bad_dev), train_ds, model=_fresh_model(0))
            out[f"bad_dev_{bad_dev}"] = None
        except Exception as exc:
            out[f"bad_dev_{bad_dev}"] = type(exc)

    out["followup"] = _followup_sum(ctx)
    return out


def _sc_guards_logs_and_export(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect()
    assert ctx is not None
    ds = make_toy_dataset(TINY_CONFIG.patch_size, series=4, seed=11, covariates=True)
    out: dict[str, Any] = {}

    # 1. Default LR with effective_batch > 32 logs warning on rank 0 only; head_only cache bypass logged once on rank 0.
    big_bs = 32  # N * 32 >= 64 > 32
    tlog, handler, prev = _capture_trainer_logs()
    try:
        trainer_big = TimestralTrainer(
            _base_config(mode="head_only", batch_size=big_bs, learning_rate=None, cache_frozen_features=True),
            ds,
            model=_fresh_model(0),
        )
    finally:
        _release_trainer_logs(tlog, handler, prev)

    out["big_warnings"] = [msg for lvl, msg in handler.records if lvl == logging.WARNING]
    out["big_infos"] = [msg for lvl, msg in handler.records if lvl == logging.INFO]
    out["head_cache_is_none"] = trainer_big._get_head_cache(ds) is None

    # 2. Explicit LR with effective_batch > 32 emits NO warning.
    tlog, handler, prev = _capture_trainer_logs()
    try:
        TimestralTrainer(
            _base_config(mode="head_only", batch_size=big_bs, learning_rate=1e-3, cache_frozen_features=False),
            ds,
            model=_fresh_model(0),
        )
    finally:
        _release_trainer_logs(tlog, handler, prev)
    out["explicit_lr_warnings"] = [msg for lvl, msg in handler.records if lvl == logging.WARNING]
    out["cache_off_bypass_infos"] = [
        msg for lvl, msg in handler.records if lvl == logging.INFO and "frozen-feature cache" in msg
    ]

    # 3. Default LR with effective_batch <= 32 emits NO warning.
    small_bs = 4  # N * 4 <= 16 <= 32
    tlog, handler, prev = _capture_trainer_logs()
    try:
        TimestralTrainer(
            _base_config(mode="head_only", batch_size=small_bs, learning_rate=None),
            ds,
            model=_fresh_model(0),
        )
    finally:
        _release_trainer_logs(tlog, handler, prev)
    out["small_bs_warnings"] = [msg for lvl, msg in handler.records if lvl == logging.WARNING]

    # 4. Public training_step() raises RuntimeError immediately with no collective (called on rank 0 only).
    if rank == 0:
        batch = ds.collator(trainer_big.model.config)([ds[0]])
        try:
            trainer_big.training_step(batch)
            out["training_step_err"] = None
        except Exception as exc:
            out["training_step_err"] = (type(exc), str(exc))
    else:
        out["training_step_err"] = "skipped_on_peer"

    # 5. Export: rank 0 writes metadata with world_size/effective_batch_size; peers validate, log, and write nothing.
    with tempfile.TemporaryDirectory(prefix=f"tt_exp_r{rank}_") as tmpdir:
        base_dir = Path(tmpdir)
        export_target = base_dir / "ckpt"

        # Missing output_dir raises ValueError on every rank.
        try:
            trainer_big.export()
            out["export_no_dir"] = None
        except Exception as exc:
            out["export_no_dir"] = type(exc)

        # Invalid inputs raise on every rank (including non-zero ranks) without writing files.
        try:
            trainer_big.export(export_target, calibrator=ConformalCalibrator(0.80))
            out["export_unfitted_cal"] = None
        except Exception as exc:
            out["export_unfitted_cal"] = type(exc)

        try:
            trainer_big.export(export_target, metrics={"bad": float("nan")})
            out["export_nan_metrics"] = None
        except Exception as exc:
            out["export_nan_metrics"] = type(exc)

        try:
            trainer_big.export(export_target, license_path=base_dir / "nonexistent_license")
            out["export_missing_license"] = None
        except Exception as exc:
            out["export_missing_license"] = type(exc)

        assert not export_target.exists()

        # On non-zero ranks, verify export() never touches Hub, weight merges, or disk.
        orig_download = export_module.hf_hub_download
        orig_merge = export_module.merged_state_dict
        orig_save = export_module.save_file

        def _forbid(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("non-zero rank export() must not call Hub, merge weights, or save files")

        if rank != 0:
            export_module.hf_hub_download = _forbid
            export_module.merged_state_dict = _forbid
            export_module.save_file = _forbid

        tlog, handler, prev = _capture_trainer_logs()
        try:
            returned = trainer_big.export(export_target, metrics={"val_loss": 0.25})
        finally:
            _release_trainer_logs(tlog, handler, prev)
            export_module.hf_hub_download = orig_download
            export_module.merged_state_dict = orig_merge
            export_module.save_file = orig_save

        out["export_returned_matches"] = returned == export_target
        out["export_dir_exists"] = export_target.exists()
        out["export_infos"] = [msg for lvl, msg in handler.records if lvl == logging.INFO]
        if rank == 0:
            meta = json.loads((export_target / "tune_metadata.json").read_text(encoding="utf-8"))
            out["export_meta"] = {
                "world_size": meta.get("world_size"),
                "effective_batch_size": meta.get("effective_batch_size"),
                "per_rank_batch_size": meta["hyperparameters"]["batch_size"],
                "metrics": meta["metrics"],
            }
        else:
            out["export_meta"] = None

    # 5b. Non-zero-rank LoRA export validates without merging/extracting adapters or writing files.
    lora_trainer = TimestralTrainer(
        _base_config(mode="lora_variate", batch_size=2),
        ds,
        lora_config=LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("q", "v")),
        model=_fresh_model(0),
    )
    with tempfile.TemporaryDirectory(prefix=f"tt_lora_exp_r{rank}_") as tmpdir:
        lora_target = Path(tmpdir) / "lora_ckpt"
        orig_download = export_module.hf_hub_download
        orig_merge = export_module.merged_state_dict
        orig_extract = export_module.extract_lora_state_dict
        orig_save = export_module.save_file

        def _forbid_lora(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("non-zero rank LoRA export() must not call Hub, merge/extract, or save files")

        if rank != 0:
            export_module.hf_hub_download = _forbid_lora
            export_module.merged_state_dict = _forbid_lora
            export_module.extract_lora_state_dict = _forbid_lora
            export_module.save_file = _forbid_lora
        try:
            lora_returned = lora_trainer.export(lora_target)
        finally:
            export_module.hf_hub_download = orig_download
            export_module.merged_state_dict = orig_merge
            export_module.extract_lora_state_dict = orig_extract
            export_module.save_file = orig_save
        out["lora_export_returned_matches"] = lora_returned == lora_target
        out["lora_export_adapter_exists"] = (lora_target / "adapter.safetensors").is_file()
        out["lora_export_dir_exists"] = lora_target.exists()

    out["followup"] = _followup_sum(ctx)
    return out


@pytest.mark.parametrize("world_size", [2, 4])
def test_trainer_distributed_startup_guards_and_export(world_size: int) -> None:
    scenarios = {
        "startup_broadcast": _sc_startup_broadcast_all_modes,
        "consistency_mismatches": _sc_consistency_mismatches,
        "startup_failures": _sc_startup_failure_propagation,
        "stage1_refusals": _sc_stage1_refusals,
        "guards_logs_export": _sc_guards_logs_and_export,
    }
    outcomes = run_scenarios(scenarios, world_size=world_size, pg_timeout=5.0, wall_timeout=30.0)

    # 1. Initial-state broadcast across all modes (including frozen weights and custom buffers).
    sb = unwrap_results(outcomes["startup_broadcast"])
    for mode in ("head_only", "lora_variate", "full_ft"):
        r0 = sb[0][mode]
        assert r0["total_steps"] == r0["expected_total_steps"]
        assert r0["buffers"]["rank_sentinel"] == [10.0, 10.0, 10.0]
        for r in range(1, world_size):
            assert sb[r][mode]["params"] == r0["params"]
            assert sb[r][mode]["buffers"] == r0["buffers"]
            assert sb[r][mode]["total_steps"] == r0["total_steps"]

    # 2. Consistency mismatches raise ValueError on EVERY rank and keep group clean.
    cm = unwrap_results(outcomes["consistency_mismatches"])
    for key in ("dataset_len", "config_seed", "model_shape", "val_presence"):
        for r in range(world_size):
            entry = cm[r][key]
            assert entry["raised"] is True
            assert entry["exc_type"] is ValueError
            assert entry["followup"] == world_size

    # 3. Startup single-rank and two-rank failure propagation.
    sf = unwrap_results(outcomes["startup_failures"])
    for r in range(world_size):
        # Rank 0 failed.
        e0 = sf[r]["fail_r0"]
        assert e0["raised"] is True and e0["followup"] == world_size
        if r == 0:
            assert e0["exc_type"] is ValueError and "val_dataset" in e0["message"]
        else:
            assert e0["exc_type"] is RankFailureError
            assert e0["failing_ranks"] == (0,) and e0["stage"] == "startup"

        # Rank 1 failed.
        e1 = sf[r]["fail_r1"]
        assert e1["raised"] is True and e1["followup"] == world_size
        if r == 1:
            assert e1["exc_type"] is TypeError and "Callback" in e1["message"]
        else:
            assert e1["exc_type"] is RankFailureError
            assert e1["failing_ranks"] == (1,) and e1["stage"] == "startup"

        # Two ranks failed.
        pair = (0, 1) if world_size == 2 else (1, 3)
        e2 = sf[r]["fail_two"]
        assert e2["raised"] is True and e2["followup"] == world_size
        if r == pair[0]:
            assert e2["exc_type"] is ValueError
        elif r == pair[1]:
            assert e2["exc_type"] is TypeError
        else:
            assert e2["exc_type"] is RankFailureError
            assert e2["failing_ranks"] == pair and e2["stage"] == "startup"

        # Mixed trainable dtype on rank 0.
        em = sf[r]["mixed_dtype"]
        assert em["raised"] is True and em["followup"] == world_size
        if r == 0:
            assert em["exc_type"] is ValueError and "share one dtype" in em["message"]
        else:
            assert em["exc_type"] is RankFailureError
            assert em["failing_ranks"] == (0,) and em["stage"] == "startup"

    # 4. Stage 1 refusals (reduce-overhead and ReplayConfig ordering).
    sr = unwrap_results(outcomes["stage1_refusals"])
    for r in range(world_size):
        assert sr[r]["reduce_overhead_on"][0] is ValueError
        assert "reduce-overhead" in sr[r]["reduce_overhead_on"][1]
        assert sr[r]["reduce_overhead_off_ok"] is True
        assert sr[r]["bad_replay_cfg_type"][0] is TypeError
        assert sr[r]["replay_ds_without_cfg"][0] is ValueError
        assert sr[r]["bad_replay_ds_type"][0] is TypeError
        assert sr[r]["replay_ds_is_val_ds"][0] is ValueError
        assert sr[r]["valid_replay_cfg"][0] is NotImplementedError
        assert sr[r]["bad_dev_cuda"] is ValueError
        assert sr[r]["bad_dev_mps"] is ValueError
        assert sr[r]["followup"] == world_size

    # 5. Guards, logs, training_step(), and export().
    gle = unwrap_results(outcomes["guards_logs_export"])
    for r in range(world_size):
        entry = gle[r]
        assert entry["head_cache_is_none"] is True
        assert entry["explicit_lr_warnings"] == []
        assert entry["cache_off_bypass_infos"] == []
        assert entry["small_bs_warnings"] == []
        assert entry["export_no_dir"] is ValueError
        assert entry["export_unfitted_cal"] is ValueError
        assert entry["export_nan_metrics"] is ValueError
        assert entry["export_missing_license"] is FileNotFoundError
        assert entry["export_returned_matches"] is True
        assert entry["lora_export_returned_matches"] is True
        assert entry["followup"] == world_size
        if r == 0:
            assert len(entry["big_warnings"]) == 1 and "effective batch size" in entry["big_warnings"][0]
            assert sum("frozen-feature cache" in m for m in entry["big_infos"]) == 1
            assert entry["training_step_err"][0] is RuntimeError
            assert entry["export_dir_exists"] is True
            assert entry["lora_export_dir_exists"] is True and entry["lora_export_adapter_exists"] is True
            assert entry["export_meta"] == {
                "world_size": world_size,
                "effective_batch_size": world_size * 32,
                "per_rank_batch_size": 32,
                "metrics": {"val_loss": 0.25},
            }
        else:
            assert entry["big_warnings"] == []
            assert entry["big_infos"] == []
            assert entry["export_dir_exists"] is False
            assert entry["lora_export_dir_exists"] is False and entry["lora_export_adapter_exists"] is False
            assert len(entry["export_infos"]) == 1 and "rank 0" in entry["export_infos"][0]


def _worker_stale_group_fit_and_post_destroy_export(rank: int, world_size: int) -> dict[str, Any]:
    ds = make_toy_dataset(TINY_CONFIG.patch_size, series=4, seed=13, covariates=True)
    trainer = TimestralTrainer(_base_config(batch_size=3), ds, model=_fresh_model(0))
    dist.barrier()
    dist.destroy_process_group()

    # Stale group: fit() raises RuntimeError before mutating trainer state.
    trainer.should_stop = True
    prev_scheduler = trainer.scheduler
    fit_err_type = None
    try:
        trainer.fit()
    except Exception as exc:
        fit_err_type = type(exc)
    state_untouched = trainer.should_stop is True and trainer.scheduler is prev_scheduler and trainer.global_step == 0

    # Stale group (world size changed): fit() also raises RuntimeError without mutating state.
    size_changed_err_type = None
    with tempfile.TemporaryDirectory(prefix=f"tt_ws1_r{rank}_") as ws1_dir:
        store = Path(ws1_dir) / "store"
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
        try:
            trainer.fit()
        except Exception as exc:
            size_changed_err_type = type(exc)
        finally:
            dist.destroy_process_group()

    # export() still works after destroy_process_group() using construction-time snapshot.
    with tempfile.TemporaryDirectory(prefix=f"tt_post_destroy_r{rank}_") as tmpdir:
        target = Path(tmpdir) / "exported"
        returned = trainer.export(target)
        exists = target.exists()
        meta = json.loads((target / "tune_metadata.json").read_text(encoding="utf-8")) if exists else None

    return {
        "fit_err_type": fit_err_type,
        "size_changed_err_type": size_changed_err_type,
        "state_untouched": state_untouched,
        "returned_matches": returned == target,
        "exists": exists,
        "world_size": None if meta is None else meta.get("world_size"),
        "effective_batch_size": None if meta is None else meta.get("effective_batch_size"),
    }


def test_trainer_stale_group_fit_and_export_after_destroy() -> None:
    results = unwrap_results(
        run_distributed(
            _worker_stale_group_fit_and_post_destroy_export, world_size=2, pg_timeout=5.0, wall_timeout=30.0
        )
    )
    assert results[0] == {
        "fit_err_type": RuntimeError,
        "size_changed_err_type": RuntimeError,
        "state_untouched": True,
        "returned_matches": True,
        "exists": True,
        "world_size": 2,
        "effective_batch_size": 6,
    }
    assert results[1] == {
        "fit_err_type": RuntimeError,
        "size_changed_err_type": RuntimeError,
        "state_untouched": True,
        "returned_matches": True,
        "exists": False,
        "world_size": None,
        "effective_batch_size": None,
    }


# --------------------------------------------------------------------------- fit() parity & step-level scenarios


def _make_unequal_dataset(
    num_windows: int,
    *,
    seed: int = 17,
    covariates: bool = True,
) -> TimeSeriesDataset:
    """Build a dataset of exactly `num_windows` windows with unequal horizon missingness across windows."""
    patch = TINY_CONFIG.patch_size
    ctx_len = 8 * patch
    horizon = 2 * patch
    rng = np.random.default_rng(seed)
    steps = ctx_len + horizon
    time = np.arange(steps, dtype=np.int64)
    series_list: list[SeriesData] = []
    for idx in range(num_windows):
        driver = rng.standard_normal(steps)
        values = 20.0 + 4.0 * np.sin(2.0 * np.pi * time / (2 * patch) + idx * 0.3) + rng.normal(0.0, 1.0, steps)
        if covariates:
            values = values + 1.5 * driver
        values = values.astype(np.float32)
        # Vary the number of missing horizon steps (0 to horizon - 2) so scored counts differ per window.
        n_missing = idx % (horizon - 1)
        if n_missing > 0:
            miss_positions = rng.choice(horizon, size=n_missing, replace=False)
            values[ctx_len + miss_positions] = np.nan
        empty = np.empty((0, steps))
        series_list.append(
            SeriesData(
                series_id=f"u{idx}",
                timestamps=time,
                targets=values[None],
                past=(rng.standard_normal((1, steps)) if covariates else empty).astype(np.float32),
                known=(driver[None] if covariates else empty).astype(np.float32),
                categorical_past=empty.astype(np.float64),
                categorical_known=empty.astype(np.float64),
            )
        )
    ds = TimeSeriesDataset(
        series_list,
        context_length=ctx_len,
        forecast_horizon=horizon,
        stride=horizon,
        min_context=ctx_len,
    )
    assert len(ds) == num_windows
    return ds


def _mask_windows_unscored(ds: TimeSeriesDataset, window_indices: list[int]) -> TimeSeriesDataset:
    """Return a copy of `ds` whose selected windows have all-NaN horizon targets (`num_scored == 0`)."""
    cloned = copy.deepcopy(ds)
    horizon = cloned.forecast_horizon
    for win_idx in window_indices:
        s_idx = int(cloned._series_index[win_idx])
        origin = int(cloned._origins[win_idx])
        cloned.series[s_idx].targets[:, origin : origin + horizon] = np.nan
    return cloned


def _lora_cfg_for_mode(mode: str) -> LoraConfig | None:
    if mode == "lora_variate":
        return LoraConfig(r=4, lora_alpha=8.0, axes=("variate",), targets=("q", "v"), dropout=0.0)
    if mode == "lora_temporal":
        return LoraConfig(r=4, lora_alpha=8.0, axes=("time",), targets=("q", "v"), dropout=0.0)
    return None


def _result_as_dict(res: Any) -> dict[str, Any]:
    return {
        "steps": res.steps,
        "epochs": res.epochs,
        "final_train_loss": res.final_train_loss,
        "best_eval_loss": res.best_eval_loss,
        "best_step": res.best_step,
        "stopped_early": res.stopped_early,
        "skipped_batches": res.skipped_batches,
        "restored_best": res.restored_best,
        "train_history": list(res.train_history),
        "eval_history": list(res.eval_history),
    }


def _flat_trainable(trainer: TimestralTrainer) -> torch.Tensor:
    return torch.cat([p.detach().cpu().reshape(-1) for p in trainer._trainable])


def _flat_all_params(trainer: TimestralTrainer) -> torch.Tensor:
    return torch.cat([p.detach().cpu().reshape(-1) for p in trainer.model.parameters()])


def _sc_fit_parity_all_modes(rank: int, world_size: int) -> dict[str, Any]:
    # 9 windows for N=2 (4, 4, 1) and 17 windows for N=4 (8, 8, 1): 3 steps/epoch x 3 epochs = 9 steps,
    # with unequal horizon missingness and a trailing partial step of 1 window (< N, so peers are empty).
    per_rank_bs = 2
    num_windows = 2 * world_size * per_rank_bs + 1
    ds = _make_unequal_dataset(num_windows, seed=21, covariates=True)
    out: dict[str, Any] = {}

    for dtype_name, use_f64 in (("f64", True), ("f32", False)):
        for mode in ("head_only", "lora_variate", "lora_temporal", "full_ft"):
            model = _fresh_model(seed=31)
            if use_f64:
                model = model.double()
            cfg = TrainingConfig(
                model_name_or_path="local-test",
                mode=mode,  # type: ignore[arg-type]
                device="cpu",
                batch_size=per_rank_bs,
                epochs=3,
                max_steps=None,
                warmup_steps=2,
                lr_schedule="cosine",
                learning_rate=2e-3 if mode != "full_ft" else 5e-4,
                weight_decay=0.05,
                gradient_clip_val=1.0,
                shuffle=False,
                sign_flip=False,
                restore_best=False,
                seed=7,
            )
            lora_cfg = _lora_cfg_for_mode(mode)
            if use_f64:
                with float64_reference():
                    trainer = TimestralTrainer(cfg, ds, lora_config=lora_cfg, model=model)
                    init_trainable = _flat_trainable(trainer).clone()
                    res = trainer.fit()
            else:
                trainer = TimestralTrainer(cfg, ds, lora_config=lora_cfg, model=model)
                init_trainable = _flat_trainable(trainer).clone()
                res = trainer.fit()

            out[f"{dtype_name}:{mode}"] = {
                "result": _result_as_dict(res),
                "init": init_trainable,
                "final": _flat_trainable(trainer).clone(),
                "all_params": _flat_all_params(trainer).clone(),
            }
    return out


def _sc_step_level_and_non_finite(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect()
    assert ctx is not None
    # Build 5 windows with unequal missingness; split across ranks so shards have unequal sizes,
    # a zero-scored shard, and an empty None shard.
    ds = _make_unequal_dataset(6, seed=41, covariates=True)
    # Mark window index 2 unscored so rank 1's shard in a zero-scored test has 0 scored positions.
    ds_with_unscored = _mask_windows_unscored(ds, [2])
    out: dict[str, Any] = {}

    for dtype_name, use_f64 in (("f64", True), ("f32", False)):
        for mode in ("head_only", "lora_variate", "lora_temporal", "full_ft"):
            model = _fresh_model(seed=43)
            if use_f64:
                model = model.double()
            cfg = _base_config(
                mode=mode,
                batch_size=2,
                max_steps=3,
                lr_schedule="constant",
                learning_rate=1e-2,
                weight_decay=0.0,
                gradient_clip_val=None,
            )
            lora_cfg = _lora_cfg_for_mode(mode)
            collator = ds.collator(model.config)

            if world_size == 2:
                all_step_shards: tuple[list[list[int] | None], ...] = (
                    [[0, 1], [3]],
                    [[0, 4], [2]],  # rank 1 zero-scored
                    [[1, 5], None],  # rank 1 empty None
                )
            else:
                all_step_shards = (
                    [[0, 1], [2], [3], None],  # unequal + zero-scored + empty all at once
                    [[4], [0, 5], [1], [2]],  # step 2 ensures LoRA A & B both have non-zero grads
                    [[3, 4], None, [5], None],
                )

            with float64_reference() if use_f64 else contextlib.nullcontext():
                trainer = TimestralTrainer(cfg, ds, lora_config=lora_cfg, model=model)
                strategy = trainer._make_fit_strategy()
                strategy.prepare_fit()
                trainer.model.train()
                step_records = []
                try:
                    for step_shards in all_step_shards:
                        indices = step_shards[rank]
                        if indices is None:
                            shard = None
                        else:
                            raw_shard = collator([ds_with_unscored[i] for i in indices])
                            shard = as_double(raw_shard) if use_f64 else raw_shard
                        outcome = strategy._run_step(lambda s=shard: s)
                        grads = torch.cat(
                            [
                                (torch.zeros_like(p) if p.grad is None else p.grad).detach().cpu().reshape(-1)
                                for p in trainer._trainable
                            ]
                        )
                        step_records.append(
                            {
                                "loss": outcome.loss,
                                "grads": grads,
                                "params": _flat_trainable(trainer).clone(),
                            }
                        )
                finally:
                    strategy.cleanup()
                    assert strategy._reducer is None
                    trainer.model.eval()
            out[f"{dtype_name}:{mode}"] = {"steps": step_records}

    # Single-rank non-finite loss raises FloatingPointError cleanly on every rank.
    nf_model = _fresh_model(seed=47)
    nf_trainer = TimestralTrainer(_base_config(mode="head_only", batch_size=2), ds, model=nf_model)
    hook_handle = None
    if rank == 1:
        hook_handle = nf_trainer.model.head.register_forward_hook(lambda _m, _inp, y: y * float("nan"))
    try:
        nf_trainer.fit()
        out["non_finite"] = {"raised": False}
    except Exception as exc:
        out["non_finite"] = {
            "raised": True,
            "exc_type": type(exc),
            "model_training": nf_trainer.model.training,
            "followup": _followup_sum(ctx),
        }
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return out


@pytest.mark.parametrize("world_size", [2, 4])
def test_trainer_distributed_parity_and_step(world_size: int) -> None:
    scenarios = {
        "fit_parity": _sc_fit_parity_all_modes,
        "step_level": _sc_step_level_and_non_finite,
    }
    outcomes = run_scenarios(scenarios, world_size=world_size, pg_timeout=5.0, wall_timeout=30.0)

    # 1. Multi-epoch fit() parity against 1-process fit() at batch_size = world_size * 2.
    fp = unwrap_results(outcomes["fit_parity"])
    per_rank_bs = 2
    num_windows = 2 * world_size * per_rank_bs + 1
    ds = _make_unequal_dataset(num_windows, seed=21, covariates=True)

    for dtype_name, use_f64, tol_param, tol_loss in (
        ("f64", True, 1e-10, 1e-10),
        ("f32", False, 1e-4, 1e-5),
    ):
        for mode in ("head_only", "lora_variate", "lora_temporal", "full_ft"):
            key = f"{dtype_name}:{mode}"
            ref_model = _fresh_model(seed=31)
            if use_f64:
                ref_model = ref_model.double()
            ref_cfg = TrainingConfig(
                model_name_or_path="local-test",
                mode=mode,  # type: ignore[arg-type]
                device="cpu",
                batch_size=world_size * per_rank_bs,
                epochs=3,
                max_steps=None,
                warmup_steps=2,
                lr_schedule="cosine",
                learning_rate=2e-3 if mode != "full_ft" else 5e-4,
                weight_decay=0.05,
                gradient_clip_val=1.0,
                shuffle=False,
                sign_flip=False,
                restore_best=False,
                seed=7,
            )
            lora_cfg = _lora_cfg_for_mode(mode)
            if use_f64:
                with float64_reference():
                    ref_trainer = TimestralTrainer(ref_cfg, ds, lora_config=lora_cfg, model=ref_model)
                    ref_init = _flat_trainable(ref_trainer).clone()
                    ref_res = ref_trainer.fit()
            else:
                ref_trainer = TimestralTrainer(ref_cfg, ds, lora_config=lora_cfg, model=ref_model)
                ref_init = _flat_trainable(ref_trainer).clone()
                ref_res = ref_trainer.fit()
            ref_final = _flat_trainable(ref_trainer).clone()
            ref_update = ref_final - ref_init

            r0 = fp[0][key]
            assert r0["result"]["steps"] == ref_res.steps == 9
            assert r0["result"]["epochs"] == ref_res.epochs == 3
            assert len(r0["result"]["train_history"]) == len(ref_res.train_history) == 9
            for (s_dist, l_dist), (s_ref, l_ref) in zip(
                r0["result"]["train_history"], ref_res.train_history, strict=True
            ):
                assert s_dist == s_ref
                assert abs(l_dist - l_ref) / max(abs(l_ref), 1e-12) <= tol_loss, f"{key} step {s_dist}"

            dist_update = r0["final"] - r0["init"]
            assert l2_rel(r0["final"], ref_final) <= tol_param, f"{key} final param mismatch"
            assert l2_rel(dist_update, ref_update) <= tol_param, f"{key} param update mismatch"

            # Cross-rank bitwise 0.0 equality on parameters and identical TrainingResult.
            for r in range(1, world_size):
                assert torch.equal(fp[r][key]["all_params"], r0["all_params"]), f"{key} rank {r} diverged"
                assert fp[r][key]["result"] == r0["result"], f"{key} rank {r} result mismatch"

    # 2. Step-level constructed-batch gradient/loss parity & non-finite FloatingPointError.
    sl = unwrap_results(outcomes["step_level"])
    ds_step = _mask_windows_unscored(_make_unequal_dataset(6, seed=41, covariates=True), [2])
    if world_size == 2:
        union_steps = [[0, 1, 3], [0, 4, 2], [1, 5]]
    else:
        union_steps = [[0, 1, 2, 3], [4, 0, 5, 1, 2], [3, 4, 5]]

    for dtype_name, use_f64, tol in (("f64", True, 1e-12), ("f32", False, 1e-5)):
        for mode in ("head_only", "lora_variate", "lora_temporal", "full_ft"):
            key = f"{dtype_name}:{mode}"
            ref_model = _fresh_model(seed=43)
            if use_f64:
                ref_model = ref_model.double()
            ref_cfg = _base_config(
                mode=mode,
                batch_size=2,
                max_steps=3,
                lr_schedule="constant",
                learning_rate=1e-2,
                weight_decay=0.0,
                gradient_clip_val=None,
            )
            lora_cfg = _lora_cfg_for_mode(mode)
            collator = ds_step.collator(ref_model.config)

            with float64_reference() if use_f64 else contextlib.nullcontext():
                ref_t = TimestralTrainer(ref_cfg, ds_step, lora_config=lora_cfg, model=ref_model)
                ref_t.model.train()
                ref_recs = []
                for idx_list in union_steps:
                    batch = collator([ds_step[i] for i in idx_list])
                    if use_f64:
                        batch = as_double(batch)
                    ref_t.optimizer.zero_grad(set_to_none=True)
                    loss_val = compute_loss(ref_t.model, batch, ref_cfg.target_frame, validate=False).loss
                    loss_val.backward()
                    grads = torch.cat(
                        [
                            (torch.zeros_like(p) if p.grad is None else p.grad).detach().cpu().reshape(-1)
                            for p in ref_t._trainable
                        ]
                    )
                    ref_t._apply_optimizer_step()
                    ref_recs.append({"loss": loss_val.item(), "grads": grads, "params": _flat_trainable(ref_t).clone()})

            for step_i, ref_rec in enumerate(ref_recs):
                r0_rec = sl[0][key]["steps"][step_i]
                assert abs(r0_rec["loss"] - ref_rec["loss"]) / max(abs(ref_rec["loss"]), 1e-12) <= tol
                assert l2_rel(r0_rec["grads"], ref_rec["grads"]) <= tol, f"{key} step {step_i} grad mismatch"
                assert l2_rel(r0_rec["params"], ref_rec["params"]) <= tol, f"{key} step {step_i} param mismatch"
                for r in range(1, world_size):
                    assert torch.equal(sl[r][key]["steps"][step_i]["grads"], r0_rec["grads"])
                    assert torch.equal(sl[r][key]["steps"][step_i]["params"], r0_rec["params"])

    for r in range(world_size):
        nf = sl[r]["non_finite"]
        assert nf["raised"] is True
        assert nf["exc_type"] is FloatingPointError
        assert nf["model_training"] is False
        assert nf["followup"] == world_size


# --------------------------------------------------------------------------- lifecycle, validation, failure scenarios


class _RecordingCallback(Callback):
    def __init__(self, *, stop_on_start: bool = False, stop_at_step: int | None = None) -> None:
        self.stop_on_start = stop_on_start
        self.stop_at_step = stop_at_step
        self.events: list[str] = []

    def on_fit_start(self, trainer: TimestralTrainer) -> None:
        self.events.append("start")
        if self.stop_on_start:
            trainer.should_stop = True

    def on_step_end(self, trainer: TimestralTrainer, step: int, loss: float, learning_rate: float) -> None:
        self.events.append(f"step:{step}")
        if self.stop_at_step is not None and step >= self.stop_at_step:
            trainer.should_stop = True

    def on_evaluate(self, trainer: TimestralTrainer, step: int, result: Any) -> None:
        self.events.append(f"eval:{step}")

    def on_fit_end(self, trainer: TimestralTrainer, result: Any) -> None:
        self.events.append("end")


class _FailingCallback(Callback):
    def __init__(self, fail_hook: str) -> None:
        self.fail_hook = fail_hook

    def on_fit_start(self, trainer: TimestralTrainer) -> None:
        if self.fail_hook == "on_fit_start":
            raise ValueError("cb_start_boom")

    def on_step_end(self, trainer: TimestralTrainer, step: int, loss: float, learning_rate: float) -> None:
        if self.fail_hook == "on_step_end":
            raise ValueError("cb_step_boom")

    def on_evaluate(self, trainer: TimestralTrainer, step: int, result: Any) -> None:
        if self.fail_hook == "on_evaluate":
            raise ValueError("cb_eval_boom")

    def on_fit_end(self, trainer: TimestralTrainer, result: Any) -> None:
        if self.fail_hook == "on_fit_end":
            raise ValueError("cb_end_boom")


def _sc_zero_scored_and_empty_epoch(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect()
    assert ctx is not None
    global_bs = world_size * 2
    base_ds = _make_unequal_dataset(global_bs * 2, seed=51, covariates=True)

    # 1. First global batch is completely unscored (W_D == 0), second global batch is active.
    partial_ds = _mask_windows_unscored(base_ds, list(range(global_bs)))
    trainer = TimestralTrainer(
        _base_config(mode="head_only", batch_size=2, epochs=1, max_steps=1, learning_rate=1e-2),
        partial_ds,
        model=_fresh_model(53),
    )
    res = trainer.fit()

    # 2. All batches unscored -> RuntimeError on every rank.
    all_unscored = _mask_windows_unscored(base_ds, list(range(len(base_ds))))
    trainer_empty = TimestralTrainer(
        _base_config(mode="head_only", batch_size=2, epochs=1, max_steps=None),
        all_unscored,
        model=_fresh_model(53),
    )
    try:
        trainer_empty.fit()
        empty_err: dict[str, Any] = {"raised": False}
    except Exception as exc:
        empty_err = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "followup": _followup_sum(ctx),
        }

    return {
        "partial_result": _result_as_dict(res),
        "partial_global_step": trainer.global_step,
        "partial_lr": trainer.optimizer.param_groups[0]["lr"],
        "partial_params": _flat_trainable(trainer).clone(),
        "empty_err": empty_err,
    }


def _sc_stop_restore_and_callbacks(rank: int, world_size: int) -> dict[str, Any]:
    ds = _make_unequal_dataset(world_size * 2 * 2 + 1, seed=61, covariates=True)
    val_ds = _make_unequal_dataset(world_size + 2, seed=62, covariates=True)
    out: dict[str, Any] = {}

    # 1. Stop from on_fit_start (with val_dataset=None -> 0 finalize_best collectives).
    cb_start = _RecordingCallback(stop_on_start=True)
    t_start = TimestralTrainer(
        _base_config(mode="head_only", batch_size=2, max_steps=4),
        ds,
        callbacks=[cb_start],
        model=_fresh_model(63),
    )
    res_start = t_start.fit()
    out["stop_start"] = {"result": _result_as_dict(res_start), "events": cb_start.events}

    # 2. Stop from on_step_end at step 2.
    cb_step = _RecordingCallback(stop_at_step=2)
    t_step = TimestralTrainer(
        _base_config(mode="head_only", batch_size=2, max_steps=5),
        ds,
        val_ds,
        callbacks=[cb_step],
        model=_fresh_model(63),
    )
    res_step = t_step.fit()
    out["stop_step"] = {"result": _result_as_dict(res_step), "events": cb_step.events}

    # 3. EarlyStopping + restore_best=True where best step is strictly earlier than final step.
    # High LR on head_only causes initial improvement then overshoot so EarlyStopping stops and restores best.
    cb_es = _RecordingCallback()
    t_es = TimestralTrainer(
        _base_config(
            mode="head_only",
            batch_size=2,
            max_steps=8,
            eval_every=1,
            early_stopping_patience=2,
            restore_best=True,
            learning_rate=0.5,
            lr_schedule="constant",
        ),
        ds,
        val_ds,
        callbacks=[cb_es],
        model=_fresh_model(64),
    )
    res_es = t_es.fit()
    out["early_stop_restore"] = {
        "result": _result_as_dict(res_es),
        "events": cb_es.events,
        "params": _flat_trainable(t_es).clone(),
    }
    return out


def _sc_sampling_flips_and_workers(rank: int, world_size: int) -> dict[str, Any]:
    per_rank_bs = 2
    num_windows = world_size * per_rank_bs + 1  # trailing partial step has 1 window (< N)
    ds = _make_unequal_dataset(num_windows, seed=71, covariates=True)
    val_ds = _make_unequal_dataset(world_size + 1, seed=72, covariates=True)

    # 1. shuffle=True + sign_flip=True across 2 epochs with an empty trailing rank.
    t_shuf = TimestralTrainer(
        _base_config(mode="head_only", batch_size=per_rank_bs, epochs=2, shuffle=True, sign_flip=True, seed=77),
        ds,
        val_ds,
        model=_fresh_model(73),
    )
    res_shuf = t_shuf.fit()

    # 2. Rank 0's sign-flip stream matches one-process at the same per-rank batch_size while peers differ.
    flip_bs = 4
    flip_ds = _make_unequal_dataset(world_size * flip_bs * 2, seed=75, covariates=True)
    t_flip = TimestralTrainer(
        _base_config(
            mode="head_only",
            batch_size=flip_bs,
            max_steps=2,
            shuffle=False,
            sign_flip=True,
            seed=88,
        ),
        flip_ds,
        model=_fresh_model(73),
    )
    captured_targets: list[torch.Tensor] = []
    orig_prepare = t_flip._prepare_domain_batch

    def _spy_prepare(batch: Any, flip_gen: Any) -> Any:
        prepared = orig_prepare(batch, flip_gen)
        if prepared is not None:
            captured_targets.append(prepared.batch.targets.detach().cpu().clone())
        return prepared

    t_flip._prepare_domain_batch = _spy_prepare  # type: ignore[method-assign]
    t_flip.fit()

    # Compare the 2 batches (8 series flips) on each rank against the single-process flip seed stream.
    collator = flip_ds.collator(t_flip.model.config)
    one_proc_gen = torch.Generator().manual_seed(88 + 1_000_003)
    ref_flipped = []
    for step_i in range(2):
        base_idx = step_i * world_size * flip_bs + rank * flip_bs
        raw_batch = collator([flip_ds[base_idx + i] for i in range(flip_bs)])
        ref_flipped.append(flip_batch_signs(raw_batch, one_proc_gen).targets.clone())
    flip_matches = all(torch.equal(got, ref) for got, ref in zip(captured_targets, ref_flipped, strict=True))

    return {
        "shuf_result": _result_as_dict(res_shuf),
        "shuf_params": _flat_trainable(t_shuf).clone(),
        "flip_matches_single_process_seed": flip_matches,
    }


def _sc_validation_and_evaluate(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect()
    assert ctx is not None
    train_ds = _make_unequal_dataset(world_size * 2, seed=81, covariates=True)
    val_ds = _make_unequal_dataset(world_size + 3, seed=82, covariates=True)
    out: dict[str, Any] = {}

    # 1. Sharded validation parity in float64 and float32, plus rank-0-only evaluate() after fit().
    for dtype_name, use_f64 in (("f64", True), ("f32", False)):
        model = _fresh_model(83)
        if use_f64:
            model = model.double()
        cfg = _base_config(mode="head_only", batch_size=2, max_steps=1, eval_every=1, eval_batch_size=2)
        if use_f64:
            with float64_reference():
                trainer = TimestralTrainer(cfg, train_ds, val_ds, model=model)
                res = trainer.fit()
                val_loader_untouched = trainer._val_loader is None
                rank0_eval = trainer.evaluate().loss if rank == 0 else None
        else:
            trainer = TimestralTrainer(cfg, train_ds, val_ds, model=model)
            res = trainer.fit()
            val_loader_untouched = trainer._val_loader is None
            rank0_eval = trainer.evaluate().loss if rank == 0 else None
        out[dtype_name] = {
            "eval_loss": res.eval_history[0][1],
            "val_loader_untouched": val_loader_untouched,
            "rank0_eval": rank0_eval,
        }

    # 2. len(val_dataset) < world_size (1-window val set: peers have empty shards).
    tiny_val = _make_unequal_dataset(1, seed=84, covariates=True)
    t_tiny = TimestralTrainer(
        _base_config(mode="head_only", batch_size=2, max_steps=1, eval_every=1),
        train_ds,
        tiny_val,
        model=_fresh_model(83),
    )
    res_tiny = t_tiny.fit()
    out["tiny_val_loss"] = res_tiny.eval_history[0][1]

    # 3. All-unscored val_dataset -> ValueError on every rank.
    unscored_val = _mask_windows_unscored(val_ds, list(range(len(val_ds))))
    t_bad_val = TimestralTrainer(
        _base_config(mode="head_only", batch_size=2, max_steps=1, eval_every=1),
        train_ds,
        unscored_val,
        model=_fresh_model(83),
    )
    try:
        t_bad_val.fit()
        out["unscored_val"] = {"raised": False}
    except Exception as exc:
        out["unscored_val"] = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "followup": _followup_sum(ctx),
        }

    return out


def _sc_fit_failure_propagation(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect()
    assert ctx is not None
    train_ds = _make_unequal_dataset(world_size * 2, seed=91, covariates=True)
    val_ds = _make_unequal_dataset(world_size + 2, seed=92, covariates=True)
    out: dict[str, Any] = {}

    # 1. Local step failure on rank 1.
    t_step_fail = TimestralTrainer(_base_config(batch_size=2, max_steps=2), train_ds, model=_fresh_model(93))
    if rank == 1:

        def _boom_prepare(_batch: Any, _gen: Any) -> Any:
            raise KeyError("step_boom")

        t_step_fail._prepare_domain_batch = _boom_prepare  # type: ignore[method-assign]
    try:
        t_step_fail.fit()
        out["step_fail"] = {"raised": False}
    except Exception as exc:
        out["step_fail"] = {
            "raised": True,
            "exc_type": type(exc),
            "failing_ranks": getattr(exc, "failing_ranks", None),
            "stage": getattr(exc, "stage", None),
            "model_training": t_step_fail.model.training,
            "flip_gen_cleared": t_step_fail._fit_flip_generator is None,
            "followup": _followup_sum(ctx),
        }

    # 2. Validation shard failure on rank 1.
    t_val_fail = TimestralTrainer(
        _base_config(batch_size=2, max_steps=1, eval_every=1), train_ds, val_ds, model=_fresh_model(93)
    )
    if rank == 1:

        def _boom_eval(_batches: Any) -> Any:
            raise RuntimeError("val_boom")

        t_val_fail._accumulate_eval_totals = _boom_eval  # type: ignore[method-assign]
    try:
        t_val_fail.fit()
        out["val_fail"] = {"raised": False}
    except Exception as exc:
        out["val_fail"] = {
            "raised": True,
            "exc_type": type(exc),
            "failing_ranks": getattr(exc, "failing_ranks", None),
            "stage": getattr(exc, "stage", None),
            "model_training": t_val_fail.model.training,
            "followup": _followup_sum(ctx),
        }

    # 3. Failing rank-0 callback in each of the four hooks.
    for hook in ("on_fit_start", "on_step_end", "on_evaluate", "on_fit_end"):
        t_cb = TimestralTrainer(
            _base_config(batch_size=2, max_steps=1, eval_every=1),
            train_ds,
            val_ds,
            callbacks=[_FailingCallback(hook)],
            model=_fresh_model(93),
        )
        try:
            t_cb.fit()
            out[f"cb_{hook}"] = {"raised": False}
        except Exception as exc:
            out[f"cb_{hook}"] = {
                "raised": True,
                "exc_type": type(exc),
                "failing_ranks": getattr(exc, "failing_ranks", None),
                "stage": getattr(exc, "stage", None),
                "model_training": t_cb.model.training,
                "followup": _followup_sum(ctx),
            }

    return out


@pytest.mark.parametrize("world_size", [2, 4])
def test_trainer_distributed_lifecycle_validation_and_failures(world_size: int) -> None:
    scenarios = {
        "zero_scored": _sc_zero_scored_and_empty_epoch,
        "stop_restore": _sc_stop_restore_and_callbacks,
        "sampling_flips": _sc_sampling_flips_and_workers,
        "validation": _sc_validation_and_evaluate,
        "failures": _sc_fit_failure_propagation,
    }
    outcomes = run_scenarios(scenarios, world_size=world_size, pg_timeout=5.0, wall_timeout=30.0)

    # 1. Zero-scored global skip and empty epoch RuntimeError.
    zs = unwrap_results(outcomes["zero_scored"])
    global_bs = world_size * 2
    base_ds = _make_unequal_dataset(global_bs * 2, seed=51, covariates=True)
    ref_partial = TimestralTrainer(
        _base_config(mode="head_only", batch_size=global_bs, epochs=1, max_steps=1, learning_rate=1e-2),
        _mask_windows_unscored(base_ds, list(range(global_bs))),
        model=_fresh_model(53),
    )
    ref_partial_res = ref_partial.fit()
    assert ref_partial_res.skipped_batches == 1 and ref_partial_res.steps == 1
    for r in range(world_size):
        assert zs[r]["partial_result"]["skipped_batches"] == 1
        assert zs[r]["partial_result"]["steps"] == 1
        assert zs[r]["partial_global_step"] == 1
        assert zs[r]["partial_lr"] == pytest.approx(ref_partial.optimizer.param_groups[0]["lr"])
        assert zs[r]["partial_result"] == zs[0]["partial_result"]
        assert torch.equal(zs[r]["partial_params"], zs[0]["partial_params"])
        assert l2_rel(zs[r]["partial_params"], _flat_trainable(ref_partial)) <= 1e-5
        assert zs[r]["empty_err"]["raised"] is True
        assert zs[r]["empty_err"]["exc_type"] is RuntimeError
        assert "no optimiser step" in zs[r]["empty_err"]["message"]
        assert zs[r]["empty_err"]["followup"] == world_size

    # 2. Stop from on_fit_start / on_step_end / EarlyStopping, restore_best, rank-0 callbacks, identical results.
    sr = unwrap_results(outcomes["stop_restore"])
    ds_sr = _make_unequal_dataset(world_size * 2 * 2 + 1, seed=61, covariates=True)
    val_sr = _make_unequal_dataset(world_size + 2, seed=62, covariates=True)
    ref_es = TimestralTrainer(
        _base_config(
            mode="head_only",
            batch_size=world_size * 2,
            max_steps=8,
            eval_every=1,
            early_stopping_patience=2,
            restore_best=True,
            learning_rate=0.5,
            lr_schedule="constant",
        ),
        ds_sr,
        val_sr,
        model=_fresh_model(64),
    )
    ref_es_res = ref_es.fit()
    assert ref_es_res.restored_best is True and ref_es_res.best_step is not None
    assert ref_es_res.best_step < ref_es_res.steps

    for r in range(world_size):
        # Stop on fit_start.
        s0 = sr[r]["stop_start"]
        assert s0["result"] == sr[0]["stop_start"]["result"]
        assert s0["result"]["steps"] == 0 and s0["result"]["epochs"] == 0
        assert s0["result"]["final_train_loss"] is None and s0["result"]["stopped_early"] is True
        assert s0["result"]["best_eval_loss"] is None and s0["result"]["restored_best"] is False
        assert (s0["events"] == ["start", "end"]) if r == 0 else (s0["events"] == [])

        # Stop on step_end.
        s1 = sr[r]["stop_step"]
        assert s1["result"] == sr[0]["stop_step"]["result"]
        assert s1["result"]["steps"] == 2 and s1["result"]["stopped_early"] is True
        assert (s1["events"] == ["start", "step:1", "step:2", "eval:2", "end"]) if r == 0 else (s1["events"] == [])

        # EarlyStopping + restore_best matches one-process.
        s2 = sr[r]["early_stop_restore"]
        assert s2["result"] == sr[0]["early_stop_restore"]["result"]
        assert s2["result"]["steps"] == ref_es_res.steps
        assert s2["result"]["best_step"] == ref_es_res.best_step
        assert s2["result"]["restored_best"] is True
        assert s2["result"]["stopped_early"] == ref_es_res.stopped_early
        assert torch.equal(s2["params"], sr[0]["early_stop_restore"]["params"])
        assert l2_rel(s2["params"], _flat_trainable(ref_es)) <= 1e-4
        if r == 0:
            assert s2["events"][0] == "start" and s2["events"][-1] == "end"
        else:
            assert s2["events"] == []

    # 3. Shuffle and sign_flip stream parity on rank 0 (peers differ).
    sf = unwrap_results(outcomes["sampling_flips"])
    assert sf[0]["flip_matches_single_process_seed"] is True
    for r in range(1, world_size):
        assert sf[r]["flip_matches_single_process_seed"] is False
        assert sf[r]["shuf_result"] == sf[0]["shuf_result"]
        assert torch.equal(sf[r]["shuf_params"], sf[0]["shuf_params"])

    # 4. Internal sharded validation parity, tiny val_dataset < N, unscored val error, rank-0 evaluate().
    val_out = unwrap_results(outcomes["validation"])
    train_v = _make_unequal_dataset(world_size * 2, seed=81, covariates=True)
    val_v = _make_unequal_dataset(world_size + 3, seed=82, covariates=True)
    tiny_val_v = _make_unequal_dataset(1, seed=84, covariates=True)
    for dtype_name, use_f64, tol in (("f64", True, 1e-12), ("f32", False, 1e-5)):
        ref_m = _fresh_model(83).double() if use_f64 else _fresh_model(83)
        ref_cfg = _base_config(
            mode="head_only", batch_size=world_size * 2, max_steps=1, eval_every=1, eval_batch_size=2
        )
        if use_f64:
            with float64_reference():
                ref_t = TimestralTrainer(ref_cfg, train_v, val_v, model=ref_m)
                ref_res = ref_t.fit()
                ref_eval = ref_t.evaluate().loss
        else:
            ref_t = TimestralTrainer(ref_cfg, train_v, val_v, model=ref_m)
            ref_res = ref_t.fit()
            ref_eval = ref_t.evaluate().loss
        assert math.isclose(val_out[0][dtype_name]["eval_loss"], ref_res.eval_history[0][1], rel_tol=tol)
        assert math.isclose(val_out[0][dtype_name]["rank0_eval"], ref_eval, rel_tol=tol)
        for r in range(world_size):
            assert val_out[r][dtype_name]["val_loader_untouched"] is True
            assert val_out[r][dtype_name]["eval_loss"] == val_out[0][dtype_name]["eval_loss"]

    ref_tiny = TimestralTrainer(
        _base_config(mode="head_only", batch_size=world_size * 2, max_steps=1, eval_every=1),
        train_v,
        tiny_val_v,
        model=_fresh_model(83),
    )
    ref_tiny_res = ref_tiny.fit()
    for r in range(world_size):
        assert math.isclose(val_out[r]["tiny_val_loss"], ref_tiny_res.eval_history[0][1], rel_tol=1e-5)
        assert val_out[r]["unscored_val"]["raised"] is True
        assert val_out[r]["unscored_val"]["exc_type"] is ValueError
        assert "no scored position" in val_out[r]["unscored_val"]["message"]
        assert val_out[r]["unscored_val"]["followup"] == world_size

    # 5. Failure propagation during fit(): step, validation, and all four rank-0 callback hooks.
    fl = unwrap_results(outcomes["failures"])
    for r in range(world_size):
        # Step failure on rank 1.
        st = fl[r]["step_fail"]
        assert st["raised"] is True and st["model_training"] is False and st["flip_gen_cleared"] is True
        assert st["followup"] == world_size
        if r == 1:
            assert st["exc_type"] is KeyError
        else:
            assert st["exc_type"] is RankFailureError
            assert st["failing_ranks"] == (1,) and st["stage"] == "training_step"

        # Validation failure on rank 1.
        vf = fl[r]["val_fail"]
        assert vf["raised"] is True and vf["model_training"] is False and vf["followup"] == world_size
        if r == 1:
            assert vf["exc_type"] is RuntimeError
        else:
            assert vf["exc_type"] is RankFailureError
            assert vf["failing_ranks"] == (1,) and vf["stage"] == "validation"

        # Rank-0 callback failures across all 4 hooks.
        for hook in ("on_fit_start", "on_step_end", "on_evaluate", "on_fit_end"):
            cf = fl[r][f"cb_{hook}"]
            assert cf["raised"] is True and cf["model_training"] is False and cf["followup"] == world_size
            if r == 0:
                assert cf["exc_type"] is ValueError
            else:
                assert cf["exc_type"] is RankFailureError
                assert cf["failing_ranks"] == (0,) and cf["stage"] == hook


def _worker_num_workers_fit(rank: int, world_size: int) -> dict[str, Any]:
    per_rank_bs = 2
    # Trailing partial step has 1 window (< N) so rank 0 fetches 2 batches/epoch and peers fetch 1 batch/epoch,
    # exercising persistent_workers=True across uneven batch counts and verifying cleanup runs after on_fit_end.
    ds = _make_unequal_dataset(world_size * per_rank_bs + 1, seed=71, covariates=True)
    trainer = TimestralTrainer(
        _base_config(
            mode="head_only",
            batch_size=per_rank_bs,
            epochs=2,
            shuffle=True,
            sign_flip=True,
            num_workers=2,
            seed=79,
        ),
        ds,
        model=_fresh_model(73),
    )
    res = trainer.fit()
    return {"result": _result_as_dict(res), "params": _flat_trainable(trainer).clone()}


@pytest.mark.skipif(WORKERS_HANG_AT_EXIT, reason="PyTorch below 2.8 on macOS: a DataLoader worker hangs the exit")
@pytest.mark.parametrize("world_size", [2, 4])
def test_trainer_distributed_num_workers(world_size: int) -> None:
    """Multi-process `fit()` with `num_workers=2` and `persistent_workers=True` across uneven trailing shards.

    Runs in its own group (not followed by a tight post-scenario barrier) because on macOS
    PyTorch's DataLoader worker shutdown waits `MP_STATUS_CHECK_INTERVAL` (5s) per active worker.
    Because `_run_fit_loop` runs `strategy.cleanup()` after `finalize_best()` and `on_fit_end()`,
    `fit()` itself stays safe under a 5s process-group timeout.
    """
    results = unwrap_results(
        run_distributed(_worker_num_workers_fit, world_size=world_size, pg_timeout=5.0, wall_timeout=45.0)
    )
    for r in range(1, world_size):
        assert results[r]["result"] == results[0]["result"]
        assert torch.equal(results[r]["params"], results[0]["params"])
