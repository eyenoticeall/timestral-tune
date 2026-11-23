"""White-box internal tests for multi-process `TimestralTrainer` and `distributed_harness`.

End-to-end spec §9 acceptance clauses live in `tests/acceptance/test_distributed.py` (T15).
This module covers only white-box internals and invariants that §9 does not specify directly:
- Package `__all__` export of `RankFailureError`.
- `tests/distributed_harness.py:_worker_main` preserving the original scenario exception if
  `dist.destroy_process_group()` raises during `finally:` cleanup.
- Non-zero rank `export()` monkeypatch forbid spies (`hf_hub_download`, `merged_state_dict`,
  `extract_lora_state_dict`, `save_file`, `dataset_fingerprint`) for `head_only` and LoRA.
- `_DistributedStrategy` private state cleanup (`_flip_generator`, `_reducer`, loaders, sampler,
  `_val_loader_ready`, and untouched `trainer._val_loader`) on both normal runs and propagated step failures.
- Mixed trainable parameter dtypes rejected in startup step (a) and propagated to peers.
- Rank-0-only startup log isolation (`WARNING`/`INFO` suppressed on non-zero ranks, and no cache
  bypass notice when `cache_frozen_features=False`).

- Spec: `docs/SPEC.md` §5.6, §7.4, D22–D35
"""

from __future__ import annotations

import copy
import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist

from timestral_tune import LoraConfig, TimestralTrainer, TrainingConfig
from timestral_tune import export as export_module
from timestral_tune.models import TuneTimestralModel
from timestral_tune.trainer import RankFailureError
from timestral_tune.trainer import __all__ as trainer_all
from timestral_tune.trainer import trainer as trainer_module
from timestral_tune.trainer.distributed import DistributedContext

from . import distributed_harness as harness
from .distributed_harness import run_scenarios, unwrap_results
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
        "epochs": 1,
        "max_steps": 1,
        "learning_rate": 1e-3,
        "shuffle": False,
        "sign_flip": False,
        "restore_best": False,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _fresh_model(seed: int = 0) -> TuneTimestralModel:
    return copy.deepcopy(make_tiny_forecaster(seed=seed).model)


# --------------------------------------------------------------------------- in-process unit checks


def test_rank_failure_error_exported_from_trainer_package() -> None:
    assert "RankFailureError" in trainer_all
    assert issubclass(RankFailureError, RuntimeError)


def test_harness_preserves_scenario_error_when_destroy_process_group_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If `dist.destroy_process_group()` raises in `_worker_main`'s `finally:`, keep the scenario error."""
    monkeypatch.setattr(dist, "init_process_group", lambda **_kw: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def _boom_destroy() -> None:
        raise RuntimeError("destroy_pg_boom")

    monkeypatch.setattr(dist, "destroy_process_group", _boom_destroy)

    def _failing_fn(rank: int, world_size: int) -> None:
        del rank, world_size
        raise ValueError("original_scenario_error")

    res_fail_path = tmp_path / "fail.pkl"
    harness._worker_main(0, 2, str(tmp_path / "store"), str(res_fail_path), 5.0, "gloo", _failing_fn, ())
    res_fail = pickle.loads(res_fail_path.read_bytes())
    assert not res_fail.ok and res_fail.error is not None
    assert res_fail.error.exc_type is ValueError
    assert "original_scenario_error" in res_fail.error.message

    # When the scenario itself succeeds, a cleanup error from destroy_process_group() is still surfaced.
    def _ok_fn(rank: int, world_size: int) -> str:
        del rank, world_size
        return "ok"

    res_ok_path = tmp_path / "ok.pkl"
    harness._worker_main(0, 2, str(tmp_path / "store"), str(res_ok_path), 5.0, "gloo", _ok_fn, ())
    res_ok = pickle.loads(res_ok_path.read_bytes())
    assert not res_ok.ok and res_ok.error is not None
    assert res_ok.error.exc_type is RuntimeError
    assert "destroy_pg_boom" in res_ok.error.message


# --------------------------------------------------------------------------- white-box multi-process internals


def _sc_non_zero_export_forbid_spies(rank: int, world_size: int) -> dict[str, Any]:
    del world_size
    ds = make_toy_dataset(TINY_CONFIG.patch_size, series=4, seed=11, covariates=True)
    head_trainer = TimestralTrainer(_base_config(mode="head_only"), ds, model=_fresh_model(0))
    lora_trainer = TimestralTrainer(
        _base_config(mode="lora_variate"),
        ds,
        lora_config=LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("q", "v")),
        model=_fresh_model(0),
    )

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix=f"tt_wb_exp_r{rank}_") as tmpdir:
        base = Path(tmpdir)
        orig_download = export_module.hf_hub_download
        orig_merge = export_module.merged_state_dict
        orig_extract = export_module.extract_lora_state_dict
        orig_save = export_module.save_file
        orig_fp = trainer_module.dataset_fingerprint

        def _forbid(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("non-zero rank export() must not hash datasets, call Hub, merge/extract, or write")

        if rank != 0:
            export_module.hf_hub_download = _forbid
            export_module.merged_state_dict = _forbid
            export_module.extract_lora_state_dict = _forbid
            export_module.save_file = _forbid
            trainer_module.dataset_fingerprint = _forbid
        try:
            head_dir = base / "head_ckpt"
            lora_dir = base / "lora_ckpt"
            out["head_returned"] = head_trainer.export(head_dir) == head_dir
            out["lora_returned"] = lora_trainer.export(lora_dir) == lora_dir
            out["head_exists"] = head_dir.exists()
            out["lora_exists"] = lora_dir.exists()
            out["lora_adapter_exists"] = (lora_dir / "adapter.safetensors").is_file()
        finally:
            export_module.hf_hub_download = orig_download
            export_module.merged_state_dict = orig_merge
            export_module.extract_lora_state_dict = orig_extract
            export_module.save_file = orig_save
            trainer_module.dataset_fingerprint = orig_fp
    return out


def _sc_strategy_cleanup_and_private_state(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect()
    assert ctx is not None
    ds = make_toy_dataset(TINY_CONFIG.patch_size, series=4, seed=13, covariates=True)
    train_ds, val_ds = ds.split_by_time(12 * TINY_CONFIG.patch_size)

    # 1. Normal strategy lifecycle populates private state and `cleanup()` releases the flip
    # generator, flat gradient buffer, loaders, sampler, and val-loader flag; internal sharded
    # validation never populates `trainer._val_loader`.
    trainer = TimestralTrainer(
        _base_config(mode="head_only", sign_flip=True, max_steps=1, eval_every=1),
        train_ds,
        val_ds,
        model=_fresh_model(1),
    )
    strategy = trainer._make_fit_strategy()
    strategy.prepare_fit()
    trainer.model.train()
    for _outcome in strategy.iter_epoch_steps(0):
        break
    strategy.evaluate_val()
    populated_before_cleanup = (
        strategy._flip_generator is not None
        and strategy._reducer is not None
        and strategy._train_loader is not None
        and strategy._sharded_val_loader is not None
        and strategy._sampler is not None
        and strategy._val_loader_ready is True
    )
    val_loader_untouched = trainer._val_loader is None
    strategy.cleanup()
    cleanup_cleared = (
        strategy._flip_generator is None
        and strategy._reducer is None
        and strategy._train_loader is None
        and strategy._sharded_val_loader is None
        and strategy._sampler is None
        and strategy._val_loader_ready is False
    )
    trainer.model.eval()

    # 2. Propagated step failure still runs `strategy.cleanup()` and restores `model.eval()` on every rank.
    t_fail = TimestralTrainer(
        _base_config(mode="head_only", sign_flip=True, max_steps=1),
        train_ds,
        model=_fresh_model(2),
    )
    captured_strategy: Any = None
    orig_make_strategy = t_fail._make_fit_strategy

    def _capture_strategy() -> Any:
        nonlocal captured_strategy
        captured_strategy = orig_make_strategy()
        return captured_strategy

    t_fail._make_fit_strategy = _capture_strategy  # type: ignore[method-assign]
    if rank == 1:

        def _boom_prepare(_batch: Any, _gen: Any) -> Any:
            raise KeyError("whitebox_step_boom")

        t_fail._prepare_domain_batch = _boom_prepare  # type: ignore[method-assign]
    exc_type: type[BaseException] | None = None
    try:
        t_fail.fit()
    except Exception as exc:
        exc_type = type(exc)

    fail_cleanup_cleared = (
        captured_strategy is not None
        and captured_strategy._flip_generator is None
        and captured_strategy._reducer is None
        and captured_strategy._train_loader is None
        and captured_strategy._sharded_val_loader is None
        and captured_strategy._sampler is None
        and captured_strategy._val_loader_ready is False
    )

    return {
        "populated_before_cleanup": populated_before_cleanup,
        "val_loader_untouched": val_loader_untouched,
        "cleanup_cleared": cleanup_cleared,
        "fail_exc_type": exc_type,
        "fail_model_eval": not t_fail.model.training,
        "fail_cleanup_cleared": fail_cleanup_cleared,
        "followup": _followup_sum(ctx),
    }


def _sc_mixed_trainable_dtype_startup(rank: int, world_size: int) -> dict[str, Any]:
    del world_size
    ctx = DistributedContext.detect()
    assert ctx is not None
    ds = make_toy_dataset(TINY_CONFIG.patch_size, series=4, seed=15, covariates=True)
    mixed_model = _fresh_model(0)
    if rank == 0:
        first_head_param = next(mixed_model.head.parameters())
        first_head_param.data = first_head_param.data.to(torch.float64)
    try:
        TimestralTrainer(_base_config(mode="head_only"), ds, model=mixed_model)
        return {"raised": False}
    except Exception as exc:
        return {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "failing_ranks": getattr(exc, "failing_ranks", None),
            "stage": getattr(exc, "stage", None),
            "followup": _followup_sum(ctx),
        }


def _sc_rank0_only_startup_log_isolation(rank: int, world_size: int) -> dict[str, Any]:
    del rank, world_size
    ds = make_toy_dataset(TINY_CONFIG.patch_size, series=4, seed=17, covariates=True)

    # Default LR with effective_batch > 32 + cache_frozen_features=True: rank 0 logs, peers emit nothing.
    tlog, handler, prev = _capture_trainer_logs()
    try:
        TimestralTrainer(
            _base_config(mode="head_only", batch_size=32, learning_rate=None, cache_frozen_features=True),
            ds,
            model=_fresh_model(0),
        )
    finally:
        _release_trainer_logs(tlog, handler, prev)
    big_warnings = [msg for lvl, msg in handler.records if lvl == logging.WARNING]
    big_infos = [msg for lvl, msg in handler.records if lvl == logging.INFO]

    # cache_frozen_features=False emits no cache-bypass info log on any rank.
    tlog, handler, prev = _capture_trainer_logs()
    try:
        TimestralTrainer(
            _base_config(mode="head_only", batch_size=2, learning_rate=1e-3, cache_frozen_features=False),
            ds,
            model=_fresh_model(0),
        )
    finally:
        _release_trainer_logs(tlog, handler, prev)
    cache_off_bypass = [msg for lvl, msg in handler.records if lvl == logging.INFO and "frozen-feature cache" in msg]
    return {
        "big_warnings": big_warnings,
        "big_infos": big_infos,
        "cache_off_bypass": cache_off_bypass,
    }


def test_distributed_trainer_whitebox_internals() -> None:
    """White-box internal invariants of multi-process `TimestralTrainer` and `_DistributedStrategy`."""
    outcomes = run_scenarios(
        {
            "export_spies": _sc_non_zero_export_forbid_spies,
            "strategy_cleanup": _sc_strategy_cleanup_and_private_state,
            "mixed_dtype": _sc_mixed_trainable_dtype_startup,
            "startup_logs": _sc_rank0_only_startup_log_isolation,
        },
        world_size=2,
        pg_timeout=5.0,
        wall_timeout=30.0,
    )

    # 1. Non-zero rank export() never calls Hub, dataset_fingerprint, merge/extract, or save_file.
    exp = unwrap_results(outcomes["export_spies"])
    assert exp[0] == {
        "head_returned": True,
        "lora_returned": True,
        "head_exists": True,
        "lora_exists": True,
        "lora_adapter_exists": True,
    }
    assert exp[1] == {
        "head_returned": True,
        "lora_returned": True,
        "head_exists": False,
        "lora_exists": False,
        "lora_adapter_exists": False,
    }

    # 2. Private strategy cleanup and post-failure `cleanup()` + `model.eval()` teardown.
    sc = unwrap_results(outcomes["strategy_cleanup"])
    for r, entry in enumerate(sc):
        assert entry["populated_before_cleanup"] is True
        assert entry["val_loader_untouched"] is True
        assert entry["cleanup_cleared"] is True
        assert entry["fail_model_eval"] is True
        assert entry["fail_cleanup_cleared"] is True
        assert entry["followup"] == 2
        assert entry["fail_exc_type"] is (KeyError if r == 1 else RankFailureError)

    # 3. Single trainable parameter dtype invariant in startup step (a).
    md = unwrap_results(outcomes["mixed_dtype"])
    assert md[0]["raised"] is True and md[0]["exc_type"] is ValueError and "share one dtype" in md[0]["message"]
    assert md[0]["followup"] == 2
    assert md[1]["raised"] is True and md[1]["exc_type"] is RankFailureError
    assert md[1]["failing_ranks"] == (0,) and md[1]["stage"] == "startup" and md[1]["followup"] == 2

    # 4. Startup log isolation (rank 0 only) and cache_frozen_features=False silence.
    sl = unwrap_results(outcomes["startup_logs"])
    assert len(sl[0]["big_warnings"]) == 1 and "effective batch size" in sl[0]["big_warnings"][0]
    assert sum("frozen-feature cache" in m for m in sl[0]["big_infos"]) == 1
    assert sl[0]["cache_off_bypass"] == []
    assert sl[1]["big_warnings"] == [] and sl[1]["big_infos"] == [] and sl[1]["cache_off_bypass"] == []
