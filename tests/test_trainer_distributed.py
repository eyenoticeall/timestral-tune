"""Multi-process `TimestralTrainer` tests (startup, consistency, guards, export, fit).

Grouped into small spawned `gloo` process groups via `tests.distributed_harness.run_scenarios`
with short timeouts so every hang fails promptly and the suite stays fast.

- Spec: `docs/SPEC.md` §5.6, §7.4, §9 (T15), §11 (D22–D35)
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
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
from timestral_tune.models import TuneTimestralModel
from timestral_tune.trainer import RankFailureError
from timestral_tune.trainer import __all__ as trainer_all
from timestral_tune.trainer.distributed import DistributedContext, steps_per_epoch

from .distributed_harness import run_distributed, run_scenarios, unwrap_results
from .helpers import TINY_CONFIG, make_tiny_forecaster, make_toy_dataset


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
        assert entry["followup"] == world_size
        if r == 0:
            assert len(entry["big_warnings"]) == 1 and "effective batch size" in entry["big_warnings"][0]
            assert sum("frozen-feature cache" in m for m in entry["big_infos"]) == 1
            assert entry["training_step_err"][0] is RuntimeError
            assert entry["export_dir_exists"] is True
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

    # export() still works after destroy_process_group() using construction-time snapshot.
    with tempfile.TemporaryDirectory(prefix=f"tt_post_destroy_r{rank}_") as tmpdir:
        target = Path(tmpdir) / "exported"
        returned = trainer.export(target)
        exists = target.exists()
        meta = json.loads((target / "tune_metadata.json").read_text(encoding="utf-8")) if exists else None

    return {
        "fit_err_type": fit_err_type,
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
        "state_untouched": True,
        "returned_matches": True,
        "exists": True,
        "world_size": 2,
        "effective_batch_size": 6,
    }
    assert results[1] == {
        "fit_err_type": RuntimeError,
        "state_untouched": True,
        "returned_matches": True,
        "exists": False,
        "world_size": None,
        "effective_batch_size": None,
    }
