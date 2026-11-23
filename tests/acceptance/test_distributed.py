"""Multi-process training acceptance tests (spec §5.6, §7.4, §9 T15, D22–D35).

Covers T15 cases (a)–(i) network-free on CPU/`gloo` for `N = 2` and `N = 4` using the tiny random
model and `tests/distributed_harness.py` (spawn start method, file-based rendezvous, 1 intra-op
thread per worker, hard wall-clock deadlines, multiple scenarios grouped per spawned group).

All parity tests compare against the shipped one-process `TimestralTrainer` at `batch_size = N * B`
with `shuffle=False`, `sign_flip=False`, and LoRA `dropout=0`, using the exact tolerance tiers from
spec §9:

- Trajectory parity (a): `1e-10` relative (`float64` under `float64_reference()`) / `1e-4` (`float32`).
- Step-level gradients and loss (b): `1e-12` (`float64`) / `1e-5` (`float32`), bitwise `0.0` cross-rank.
- Internal sharded validation (g): `1e-12` (`float64`) / `1e-5` (`float32`).
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader

import timestral_tune.trainer as trainer_pkg
from timestral_tune import LoraConfig, ReplayConfig, TimestralTrainer, TrainingConfig
from timestral_tune._vendor.timestral import TimestralModel
from timestral_tune.data import ColumnRoles, SeriesData, TimeSeriesDataset
from timestral_tune.models import ConformalCalibrator, TuneTimestralModel
from timestral_tune.models.adapter import merged_state_dict
from timestral_tune.trainer import Callback, EvalResult, TrainingResult
from timestral_tune.trainer.distributed import (
    RankFailureError,
    rank_seed,
    steps_per_epoch,
    validation_shard_indices,
)
from timestral_tune.trainer.step import compute_loss

from ..distributed_harness import run_distributed, run_scenarios, unwrap_results
from ..helpers import as_double, float64_reference, l2_rel, make_tiny_forecaster

# PyTorch < 2.8 on macOS hangs at process exit after DataLoader workers run (tests/test_dataset.py:451).
WORKERS_HANG_AT_EXIT = sys.platform == "darwin" and tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)

ALL_MODES: tuple[str, ...] = ("head_only", "lora_variate", "lora_temporal", "full_ft")

TRAJ_TOL: dict[str, float] = {"float64": 1e-10, "float32": 1e-4}
STEP_TOL: dict[str, float] = {"float64": 1e-12, "float32": 1e-5}
EVAL_TOL: dict[str, float] = {"float64": 1e-12, "float32": 1e-5}

PG_TIMEOUT = 10.0
WALL_TIMEOUT = 45.0


# --------------------------------------------------------------------------- shared dataset, model & callback helpers


class _RecordingDataset(TimeSeriesDataset):
    """`TimeSeriesDataset` subclass recording window indices fetched by `DataLoader` (`num_workers=0`)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accessed_indices: list[int] = []

    def __getitem__(self, index: int):
        self.accessed_indices.append(int(index))
        return super().__getitem__(index)


def _make_uneven_dataset(
    scored_counts: Sequence[int],
    *,
    seed: int = 0,
    covariates: bool = True,
    positive_only: bool = False,
    recording: bool = False,
    patch_size: int = 4,
) -> TimeSeriesDataset:
    """Build a 1-window-per-series dataset with exact per-window horizon scored counts (`0..2*patch_size`).

    Each series has `context_length = 4 * patch_size` (16) and `forecast_horizon = 2 * patch_size`
    (8), so window `i` corresponds 1-to-1 to series `i` with origin `16`. Entries with
    `scored_counts[i] == 0` are constructed with a placeholder finite point and mutated to NaN
    afterward so the window remains in the dataset with `batch.num_scored == 0`.
    """
    ctx_len = 4 * patch_size
    hor_len = 2 * patch_size
    total = ctx_len + hor_len
    rng = np.random.default_rng(seed)
    ts = np.arange(total, dtype=np.int64)
    empty_cat = np.empty((0, total), dtype=np.float64)

    series_list: list[SeriesData] = []
    zero_indices: list[int] = []
    for idx, count in enumerate(scored_counts):
        if not 0 <= count <= hor_len:
            raise ValueError(f"scored count {count} out of range 0..{hor_len}")
        scale = 1.5 + 0.4 * (idx % 5)
        phase = 0.5 * idx
        t = np.arange(total, dtype=np.float64)
        base_level = 30.0 + 3.0 * (idx % 4) if positive_only else 15.0 + 2.0 * (idx % 4)
        walk = (
            base_level + scale * np.sin(2.0 * np.pi * t / (2 * patch_size) + phase) + 0.6 * rng.standard_normal(total)
        )
        if positive_only:
            walk = np.abs(walk) + 20.0
        target = walk[None, :].astype(np.float32)

        if covariates:
            driver = rng.standard_normal((1, total)).astype(np.float32)
            past = rng.standard_normal((1, total)).astype(np.float32)
            target = target + 1.25 * driver
            if positive_only:
                target = np.abs(target) + 20.0
            known = driver
        else:
            past = np.empty((0, total), dtype=np.float32)
            known = np.empty((0, total), dtype=np.float32)

        if count == 0:
            # Keep one finite horizon entry during TimeSeriesDataset construction, then mask to NaN below.
            target[:, ctx_len + 1 :] = np.nan
            zero_indices.append(idx)
        elif count < hor_len:
            target[:, ctx_len + count :] = np.nan

        series_list.append(SeriesData(f"u{idx}", ts, target, past, known, empty_cat, empty_cat))

    roles = ColumnRoles(targets=("y",), past=("p0",) if covariates else (), known=("k0",) if covariates else ())
    cls = _RecordingDataset if recording else TimeSeriesDataset
    ds = cls(
        series_list,
        context_length=ctx_len,
        forecast_horizon=hor_len,
        stride=hor_len,
        min_context=ctx_len,
        roles=roles,
    )
    for idx in zero_indices:
        ds.series[idx].targets[:, ctx_len:] = np.nan
    return ds


def _fresh_tiny_model(seed: int = 0, dtype: torch.dtype = torch.float32) -> TuneTimestralModel:
    model = make_tiny_forecaster(seed=seed).model
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
        model.transformer.rope.rebuild_inv_freq()
    return model


def _lora_for(mode: str, *, dropout: float = 0.0) -> LoraConfig | None:
    if mode in ("head_only", "full_ft"):
        return None
    axis = "variate" if mode == "lora_variate" else "time"
    return LoraConfig(r=4, lora_alpha=8, axes=(axis,), dropout=dropout)


def _flat_params(model: TuneTimestralModel) -> Tensor:
    return torch.cat([p.detach().cpu().double().reshape(-1) for _, p in sorted(model.named_parameters())])


def _flat_trainable(trainer: TimestralTrainer) -> Tensor:
    return torch.cat([p.detach().cpu().double().reshape(-1) for p in trainer._trainable])


def _flat_trainable_grads(trainer: TimestralTrainer) -> Tensor:
    parts: list[Tensor] = []
    for p in trainer._trainable:
        assert p.grad is not None, "expected populated gradient on trainable parameter"
        parts.append(p.grad.detach().cpu().double().reshape(-1))
    return torch.cat(parts)


def _scalar_rel(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-300)


def _base_config(mode: str = "head_only", **overrides: Any) -> TrainingConfig:
    defaults: dict[str, Any] = {
        "model_name_or_path": "tiny-local-test",
        "mode": mode,
        "device": "cpu",
        "log_every": 1000,
        "shuffle": False,
        "sign_flip": False,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _followup_collective_sum(world_size: int) -> int:
    probe = torch.ones(1, dtype=torch.int64)
    dist.all_reduce(probe, op=dist.ReduceOp.SUM)
    return int(probe.item())


def _record_propagated_exception(fn: Any, world_size: int) -> dict[str, Any]:
    exc_type: type[BaseException] | None = None
    exc_msg = ""
    failing_ranks: tuple[int, ...] | None = None
    stage: str | None = None
    try:
        fn()
    except Exception as exc:
        exc_type = type(exc)
        exc_msg = str(exc)
        if isinstance(exc, RankFailureError):
            failing_ranks = exc.failing_ranks
            stage = exc.stage
    return {
        "exc_type": exc_type,
        "exc_msg": exc_msg,
        "failing_ranks": failing_ranks,
        "stage": stage,
        "followup": _followup_collective_sum(world_size),
    }


def _assert_propagated(
    outs: list[dict[str, Any]],
    world_size: int,
    expected_local: dict[int, type[BaseException]],
    label: str,
) -> None:
    expected_failing = tuple(sorted(expected_local))
    # Check locally failing ranks first so a missed failure is reported directly before peer propagation checks.
    ordered_ranks = [*expected_failing, *(r for r in range(world_size) if r not in expected_local)]
    for r in ordered_ranks:
        out = outs[r]
        assert out["followup"] == world_size, f"{label} r{r}: follow-up collective failed"
        if r in expected_local:
            assert out["exc_type"] is expected_local[r], (
                f"{label} r{r}: expected {expected_local[r].__name__}, got {out['exc_type']}"
            )
        else:
            assert out["exc_type"] is RankFailureError, (
                f"{label} r{r}: expected RankFailureError, got {out['exc_type']}"
            )
            assert out["failing_ranks"] == expected_failing, (
                f"{label} r{r}: failing_ranks {out['failing_ranks']} != {expected_failing}"
            )
            assert out["stage"], f"{label} r{r}: missing stage label on RankFailureError"
            assert out["stage"] in out["exc_msg"]
            for fr in expected_failing:
                assert str(fr) in out["exc_msg"]


class _CaptureHandler(logging.Handler):
    """In-memory logging handler for asserting warning/info logs inside worker processes."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelno, record.getMessage()))


class _RecorderCallback(Callback):
    """Records every hook invocation and optionally triggers stop on `on_fit_start` or `on_step_end`."""

    def __init__(self, *, stop_on_start: bool = False, stop_at_step: int | None = None) -> None:
        self.stop_on_start = stop_on_start
        self.stop_at_step = stop_at_step
        self.started = 0
        self.ended = 0
        self.steps: list[int] = []
        self.lrs: list[float] = []
        self.evals: list[tuple[int, EvalResult]] = []

    def on_fit_start(self, trainer: TimestralTrainer) -> None:
        self.started += 1
        if self.stop_on_start:
            trainer.should_stop = True

    def on_step_end(self, trainer: TimestralTrainer, step: int, loss: float, learning_rate: float) -> None:
        del loss
        self.steps.append(step)
        self.lrs.append(learning_rate)
        if self.stop_at_step is not None and step == self.stop_at_step:
            trainer.should_stop = True

    def on_evaluate(self, trainer: TimestralTrainer, step: int, result: EvalResult) -> None:
        del trainer
        self.evals.append((step, result))

    def on_fit_end(self, trainer: TimestralTrainer, result: TrainingResult) -> None:
        del trainer, result
        self.ended += 1

    def summary(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "ended": self.ended,
            "steps": list(self.steps),
            "lrs": list(self.lrs),
            "eval_steps": [s for s, _ in self.evals],
            "eval_scored": [r.scored_positions for _, r in self.evals],
        }


class _BoomCallback(Callback):
    """Raises `KeyError` on the configured hook so rank-0 callback failure propagation is tested."""

    def __init__(self, hook: str = "on_step_end") -> None:
        self._hook = hook

    def on_fit_start(self, trainer: TimestralTrainer) -> None:
        del trainer
        if self._hook == "on_fit_start":
            raise KeyError("callback_start_boom")

    def on_step_end(self, trainer: TimestralTrainer, step: int, loss: float, learning_rate: float) -> None:
        del trainer, loss, learning_rate
        if self._hook == "on_step_end" and step == 1:
            raise KeyError("callback_step_boom")

    def on_evaluate(self, trainer: TimestralTrainer, step: int, result: EvalResult) -> None:
        del trainer, step, result
        if self._hook == "on_evaluate":
            raise KeyError("callback_eval_boom")

    def on_fit_end(self, trainer: TimestralTrainer, result: TrainingResult) -> None:
        del trainer, result
        if self._hook == "on_fit_end":
            raise KeyError("callback_end_boom")


class _SetClearStopCallback(Callback):
    """Sets `trainer.should_stop = True` in one hook and clears it to `False` in a subsequent reachable hook."""

    def __init__(self, mode: str) -> None:
        self._mode = mode
        self.steps: list[int] = []
        self.evals: list[int] = []

    def on_fit_start(self, trainer: TimestralTrainer) -> None:
        if self._mode == "fit_start_to_step0_eval":
            trainer.should_stop = True

    def on_step_end(self, trainer: TimestralTrainer, step: int, loss: float, learning_rate: float) -> None:
        del loss, learning_rate
        self.steps.append(step)
        if self._mode == "step_end_to_eval" and step <= 2:
            trainer.should_stop = True

    def on_evaluate(self, trainer: TimestralTrainer, step: int, result: EvalResult) -> None:
        del result
        self.evals.append(step)
        if self._mode in ("step_end_to_eval", "fit_start_to_step0_eval"):
            trainer.should_stop = False


# --------------------------------------------------------------------------- T15 (a): multi-epoch trajectory parity


def _t15a_counts(world_size: int) -> list[int]:
    # Length = 2 * (N * B) + R with B = 2 and R < N (trailing empty rank(s)) and sharp scored imbalance across ranks.
    if world_size == 2:
        # 9 windows: step 0 = r0:[8,8] r1:[1,1]; step 1 = r0:[1,2] r1:[8,8]; step 2 = r0:[5] r1:[]
        return [8, 8, 1, 1, 1, 2, 8, 8, 5]
    # N = 4 -> 18 windows: trailing R = 2 < 4 (ranks 2, 3 empty on step 2).
    return [8, 8, 1, 1, 6, 6, 1, 2, 1, 1, 8, 7, 2, 1, 8, 8, 8, 1]


def _run_t15a_one(world_size: int, mode: str, dtype_name: str, *, per_rank_batch: int) -> dict[str, Any]:
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    ds = _make_uneven_dataset(_t15a_counts(world_size), seed=101, covariates=True)
    lr = 1e-3 if mode == "full_ft" else 1e-2
    cfg = _base_config(
        mode=mode,
        batch_size=per_rank_batch,
        epochs=3,
        max_steps=None,
        warmup_steps=2,
        lr_schedule="cosine",
        learning_rate=lr,
        weight_decay=0.01,
        gradient_clip_val=1.0,
        seed=7,
    )
    ctx_mgr = float64_reference() if dtype == torch.float64 else torch.enable_grad()
    pre_clip_norms: list[float] = []
    with ctx_mgr:
        model = _fresh_tiny_model(seed=0, dtype=dtype)
        trainer = TimestralTrainer(cfg, ds, lora_config=_lora_for(mode, dropout=0.0), model=model)
        init_trainable = _flat_trainable(trainer).clone()
        orig_apply = trainer._apply_optimizer_step

        def _spy_apply() -> None:
            grads = [p.grad.detach() for p in trainer._trainable if p.grad is not None]
            if grads:
                norm = float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])).item())
                pre_clip_norms.append(norm)
            orig_apply()

        trainer._apply_optimizer_step = _spy_apply  # type: ignore[method-assign]
        res = trainer.fit()
        final_trainable = _flat_trainable(trainer).clone()
        final_all = _flat_params(trainer.model).clone()
    clipped_steps = sum(1 for n in pre_clip_norms if cfg.gradient_clip_val is not None and n > cfg.gradient_clip_val)
    return {
        "result": res,
        "init_trainable": init_trainable,
        "final_trainable": final_trainable,
        "final_all": final_all,
        "clipped_steps": clipped_steps,
    }


def _sc_t15a_trajectory(rank: int, world_size: int, mode: str, dtype_name: str) -> dict[str, Any]:
    del rank
    return _run_t15a_one(world_size, mode, dtype_name, per_rank_batch=2)


def _run_t15a_repeated_det(world_size: int, *, per_rank_batch: int) -> dict[str, Any]:
    ds = _make_uneven_dataset(_t15a_counts(world_size), seed=105, covariates=True)
    cfg = _base_config(
        mode="lora_variate",
        batch_size=per_rank_batch,
        epochs=2,
        max_steps=None,
        warmup_steps=2,
        lr_schedule="cosine",
        learning_rate=1e-2,
        weight_decay=0.01,
        gradient_clip_val=1.0,
        seed=17,
    )
    cb = _RecorderCallback()
    trainer = TimestralTrainer(
        cfg,
        ds,
        lora_config=_lora_for("lora_variate", dropout=0.0),
        callbacks=[cb],
        model=_fresh_tiny_model(seed=10),
    )
    res1 = trainer.fit()
    lrs1 = list(cb.lrs)
    step1 = trainer.global_step
    params1_train = _flat_trainable(trainer).clone()
    params1_all = _flat_params(trainer.model).clone()

    res2 = trainer.fit()
    lrs2 = cb.lrs[len(lrs1) :]
    step2 = trainer.global_step
    params2_train = _flat_trainable(trainer).clone()
    params2_all = _flat_params(trainer.model).clone()
    return {
        "res1": res1,
        "res2": res2,
        "lrs1": lrs1,
        "lrs2": lrs2,
        "step1": step1,
        "step2": step2,
        "params1_train": params1_train,
        "params2_train": params2_train,
        "params1_all": params1_all,
        "params2_all": params2_all,
    }


def _sc_t15a_repeated_det(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    return _run_t15a_repeated_det(world_size, per_rank_batch=2)


def _sc_t15a_repeated_shuf(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    n_items = world_size * 2 + world_size + 1
    ds = _make_uneven_dataset([8 - (i % 7) for i in range(n_items)], seed=106, covariates=True, recording=True)
    assert isinstance(ds, _RecordingDataset)
    cfg = _base_config(
        mode="lora_temporal",
        batch_size=2,
        epochs=2,
        max_steps=None,
        warmup_steps=1,
        lr_schedule="cosine",
        learning_rate=1e-2,
        shuffle=True,
        sign_flip=False,
        seed=55,
    )
    trainer = TimestralTrainer(
        cfg,
        ds,
        lora_config=_lora_for("lora_temporal", dropout=0.0),
        model=_fresh_tiny_model(seed=11),
    )
    ds.accessed_indices.clear()
    res1 = trainer.fit()
    idx1 = list(ds.accessed_indices)
    ds.accessed_indices.clear()
    res2 = trainer.fit()
    idx2 = list(ds.accessed_indices)
    return {
        "res1": res1,
        "res2": res2,
        "idx1": idx1,
        "idx2": idx2,
        "params_all": _flat_params(trainer.model).clone(),
    }


@pytest.mark.acceptance
@pytest.mark.parametrize("world_size", [2, 4])
def test_t15_a_multi_epoch_trajectory_parity(world_size: int) -> None:
    """T15(a). Multi-epoch `fit()` parity across all four modes, float32 and float64, with unequal
    scored counts, trailing empty rank(s), warmup+cosine LR, weight decay, and gradient clipping,
    plus calling `fit()` twice on the same trainer (fresh schedule, identical batch order, parity).
    """
    scenarios: dict[str, Any] = {
        f"{mode}_{dt}": (_sc_t15a_trajectory, (mode, dt)) for dt in ("float64", "float32") for mode in ALL_MODES
    }
    scenarios["repeated_det"] = _sc_t15a_repeated_det
    scenarios["repeated_shuf"] = _sc_t15a_repeated_shuf
    outcomes = run_scenarios(scenarios, world_size=world_size, pg_timeout=PG_TIMEOUT, wall_timeout=WALL_TIMEOUT)

    for dt in ("float64", "float32"):
        tol = TRAJ_TOL[dt]
        for mode in ALL_MODES:
            ref = _run_t15a_one(world_size, mode, dt, per_rank_batch=world_size * 2)
            ref_res: TrainingResult = ref["result"]
            assert ref_res.steps == 9 and ref_res.epochs == 3 and ref_res.skipped_batches == 0
            ref_update = ref["final_trainable"] - ref["init_trainable"]
            assert float(ref_update.norm()) > 1e-4, "reference update must be non-trivial"
            if mode in ("head_only", "full_ft"):
                assert ref["clipped_steps"] >= 1, f"{mode}/{dt}: trajectory never triggered gradient clipping"

            rank_outs = unwrap_results(outcomes[f"{mode}_{dt}"])
            r0 = rank_outs[0]
            for r, out in enumerate(rank_outs):
                res: TrainingResult = out["result"]
                assert res.steps == ref_res.steps, f"{mode}/{dt} r{r}: steps {res.steps} != {ref_res.steps}"
                assert res.epochs == ref_res.epochs, f"{mode}/{dt} r{r}: epochs {res.epochs} != {ref_res.epochs}"
                assert res.skipped_batches == 0
                assert res == r0["result"], f"{mode}/{dt} r{r}: TrainingResult differs from rank 0"
                assert torch.equal(out["final_all"], r0["final_all"]), (
                    f"{mode}/{dt} r{r}: cross-rank parameters not bitwise identical"
                )
                for (step, loss), (ref_step, ref_loss) in zip(res.train_history, ref_res.train_history, strict=True):
                    assert step == ref_step
                    assert _scalar_rel(loss, ref_loss) <= tol, (
                        f"{mode}/{dt} r{r} step {step}: loss {loss} vs ref {ref_loss}"
                    )
                param_err = l2_rel(out["final_trainable"], ref["final_trainable"])
                update_err = l2_rel(out["final_trainable"] - out["init_trainable"], ref_update)
                assert param_err <= tol, f"{mode}/{dt} r{r}: final_params rel L2 {param_err:.2e} > {tol:.1e}"
                assert update_err <= tol, f"{mode}/{dt} r{r}: param_update rel L2 {update_err:.2e} > {tol:.1e}"

    # Repeated fit() (shuffle=False): fresh warmup+cosine schedule, continuous global_step, parity with 1-process.
    ref_rep = _run_t15a_repeated_det(world_size, per_rank_batch=world_size * 2)
    assert ref_rep["res1"].steps == 6 and ref_rep["res2"].steps == 6
    assert ref_rep["step1"] == 6 and ref_rep["step2"] == 12
    assert ref_rep["lrs2"] == pytest.approx(ref_rep["lrs1"], rel=1e-12)
    assert ref_rep["res2"].train_history[0][0] == 7
    assert float((ref_rep["params2_train"] - ref_rep["params1_train"]).norm()) > 1e-4

    rep_outs = unwrap_results(outcomes["repeated_det"])
    r0_rep = rep_outs[0]
    assert r0_rep["lrs1"] == pytest.approx(ref_rep["lrs1"], rel=1e-12)
    assert r0_rep["lrs2"] == pytest.approx(r0_rep["lrs1"], rel=1e-12)
    for _r, out in enumerate(rep_outs):
        assert out["res1"] == r0_rep["res1"] and out["res2"] == r0_rep["res2"]
        assert out["step1"] == 6 and out["step2"] == 12
        assert out["res2"].train_history[0][0] == 7
        assert torch.equal(out["params1_all"], r0_rep["params1_all"])
        assert torch.equal(out["params2_all"], r0_rep["params2_all"])
        assert l2_rel(out["params1_train"], ref_rep["params1_train"]) <= TRAJ_TOL["float32"]
        assert l2_rel(out["params2_train"], ref_rep["params2_train"]) <= TRAJ_TOL["float32"]
        for (s, loss), (rs, rloss) in zip(out["res2"].train_history, ref_rep["res2"].train_history, strict=True):
            assert s == rs and _scalar_rel(loss, rloss) <= TRAJ_TOL["float32"]

    # Repeated fit() (shuffle=True): identical batch order on second fit() on every rank and bitwise param equality.
    shuf_rep_outs = unwrap_results(outcomes["repeated_shuf"])
    r0_srep = shuf_rep_outs[0]
    for r, out in enumerate(shuf_rep_outs):
        assert out["idx1"] == out["idx2"] and len(out["idx1"]) > 0, f"repeated_shuf r{r}: batch order changed"
        assert out["res1"] == r0_srep["res1"] and out["res2"] == r0_srep["res2"]
        assert out["res2"].train_history[0][0] == out["res1"].steps + 1
        assert torch.equal(out["params_all"], r0_srep["params_all"])


# --------------------------------------------------------------------------- T15 (b): step gradients, loss & non-finite


def _t15b_spec(world_size: int) -> tuple[list[int], tuple[list[list[int] | None], ...]]:
    """Construct unequal-count windows (with index 2 zero-scored) and 3 constructed shard layouts:
    step 0 unequal counts, step 1 rank-1 zero-scored shard (also LoRA step 1 with lora_B != 0),
    and step 2 rank-(N-1) empty `None` shard.
    """
    if world_size == 2:
        counts = [8, 8, 0, 1, 7, 2]
        shards: tuple[list[list[int] | None], ...] = (
            [[0, 1], [3, 5]],
            [[0, 4], [2]],
            [[1, 4], None],
        )
        return counts, shards
    counts = [8, 8, 0, 1, 7, 2, 6, 1]
    shards = (
        [[0, 1], [3, 7], [4], [5, 6]],
        [[0, 4], [2], [6], [3, 5]],
        [[1, 4], [3], [6, 7], None],
    )
    return counts, shards


def _sc_t15b_direct_steps(rank: int, world_size: int, mode: str, dtype_name: str) -> dict[str, Any]:
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    counts, step_shards = _t15b_spec(world_size)
    ds = _make_uneven_dataset(counts, seed=202, covariates=True)
    lr = 1e-3 if mode == "full_ft" else 1e-2
    cfg = _base_config(
        mode=mode,
        batch_size=2,
        max_steps=len(step_shards),
        warmup_steps=0,
        lr_schedule="constant",
        learning_rate=lr,
        weight_decay=0.0,
        gradient_clip_val=None,
        seed=13,
    )
    ctx_mgr = float64_reference() if dtype == torch.float64 else contextlib.nullcontext()
    records: list[dict[str, Any]] = []
    with ctx_mgr:
        model = _fresh_tiny_model(seed=1, dtype=dtype)
        trainer = TimestralTrainer(cfg, ds, lora_config=_lora_for(mode, dropout=0.0), model=model)
        collator = ds.collator(trainer.model.config)
        strategy = trainer._make_fit_strategy()
        strategy.prepare_fit()
        trainer.model.train()
        try:
            for shard_map in step_shards:
                indices = shard_map[rank]
                if indices is None:
                    shard = None
                else:
                    raw = collator([ds[i] for i in indices])
                    shard = as_double(raw) if dtype == torch.float64 else raw
                outcome = strategy._run_step(lambda s=shard: s)
                grad = _flat_trainable_grads(trainer).clone()
                params = _flat_trainable(trainer).clone()
                extra: dict[str, float] = {}
                if mode.startswith("lora_"):
                    a_grads = [
                        p.grad.detach().double().reshape(-1)
                        for n, p in trainer.model.named_parameters()
                        if ".lora_A." in n and p.grad is not None
                    ]
                    b_grads = [
                        p.grad.detach().double().reshape(-1)
                        for n, p in trainer.model.named_parameters()
                        if ".lora_B." in n and p.grad is not None
                    ]
                    b_max = max(
                        float(p.detach().abs().max().item())
                        for n, p in trainer.model.named_parameters()
                        if ".lora_B." in n
                    )
                    extra = {
                        "a_grad_norm": float(torch.cat(a_grads).norm()),
                        "b_grad_norm": float(torch.cat(b_grads).norm()),
                        "b_max": b_max,
                    }
                records.append(
                    {
                        "skipped": outcome.skipped,
                        "loss": outcome.loss,
                        "grad": grad,
                        "params": params,
                        **extra,
                    }
                )
        finally:
            strategy.cleanup()
            trainer.model.eval()
    return {"steps": records}


def _run_t15b_union_ref(world_size: int, mode: str, dtype_name: str) -> list[dict[str, Any]]:
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    counts, step_shards = _t15b_spec(world_size)
    ds = _make_uneven_dataset(counts, seed=202, covariates=True)
    lr = 1e-3 if mode == "full_ft" else 1e-2
    cfg = _base_config(
        mode=mode,
        batch_size=2 * world_size,
        max_steps=len(step_shards),
        warmup_steps=0,
        lr_schedule="constant",
        learning_rate=lr,
        weight_decay=0.0,
        gradient_clip_val=None,
        seed=13,
    )
    ctx_mgr = float64_reference() if dtype == torch.float64 else contextlib.nullcontext()
    records: list[dict[str, Any]] = []
    with ctx_mgr:
        model = _fresh_tiny_model(seed=1, dtype=dtype)
        trainer = TimestralTrainer(cfg, ds, lora_config=_lora_for(mode, dropout=0.0), model=model)
        collator = ds.collator(trainer.model.config)
        trainer.model.train()
        for shard_map in step_shards:
            union_indices = [idx for shard in shard_map if shard is not None for idx in shard]
            raw_union = collator([ds[i] for i in union_indices])
            union_batch = as_double(raw_union) if dtype == torch.float64 else raw_union
            trainer.optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(trainer.model, union_batch, cfg.target_frame, validate=False).loss
            loss.backward()
            grad = _flat_trainable_grads(trainer).clone()
            trainer._apply_optimizer_step()
            records.append(
                {
                    "loss": loss.item(),
                    "grad": grad,
                    "params": _flat_trainable(trainer).clone(),
                }
            )
        trainer.model.eval()
    return records


def _sc_t15b_non_finite_direct(rank: int, world_size: int) -> dict[str, Any]:
    counts, step_shards = _t15b_spec(world_size)
    ds = _make_uneven_dataset(counts, seed=204, covariates=True)
    cfg = _base_config(mode="full_ft", batch_size=2, max_steps=1, learning_rate=1e-3, seed=18)
    trainer = TimestralTrainer(cfg, ds, model=_fresh_tiny_model(seed=2))
    collator = ds.collator(trainer.model.config)
    indices = step_shards[0][rank]
    assert indices is not None
    shard = collator([ds[i] for i in indices])
    before = _flat_params(trainer.model).clone()

    strategy = trainer._make_fit_strategy()
    strategy.prepare_fit()
    trainer.model.train()
    handle = None
    if rank == world_size - 1:
        handle = trainer.model.head.register_forward_hook(lambda _mod, _inp, out: out * float("nan"))

    exc_type: type[BaseException] | None = None
    try:
        strategy._run_step(lambda s=shard: s)
    except Exception as exc:
        exc_type = type(exc)
    finally:
        if handle is not None:
            handle.remove()
        strategy.cleanup()
        trainer.model.eval()

    return {
        "exc_type": exc_type,
        "unchanged": torch.equal(_flat_params(trainer.model), before),
        "no_grads": all(p.grad is None for p in trainer._trainable),
        "followup": _followup_collective_sum(world_size),
    }


def _sc_t15b_non_finite_fit(rank: int, world_size: int) -> dict[str, Any]:
    ds = _make_uneven_dataset([8, 8, 1, 1] if world_size == 2 else [8, 8, 1, 1, 7, 5, 1, 2], seed=205, covariates=True)
    cfg = _base_config(mode="full_ft", batch_size=2, max_steps=1, learning_rate=1e-3, seed=19)
    model = _fresh_tiny_model(seed=2)
    trainer = TimestralTrainer(cfg, ds, model=model)
    before = _flat_params(trainer.model).clone()

    handle = None
    if rank == world_size - 1:
        handle = trainer.model.head.register_forward_hook(lambda _mod, _inp, out: out * float("nan"))

    exc_type: type[BaseException] | None = None
    try:
        trainer.fit()
    except Exception as exc:
        exc_type = type(exc)
    finally:
        if handle is not None:
            handle.remove()

    return {
        "exc_type": exc_type,
        "unchanged": torch.equal(_flat_params(trainer.model), before),
        "no_grads": all(p.grad is None for p in trainer._trainable),
        "eval_mode": not trainer.model.training,
        "followup": _followup_collective_sum(world_size),
    }


def _sc_t15b_all_none_grad_param(rank: int, world_size: int) -> dict[str, Any]:
    """Unused trainable parameter keeps `.grad is None` and skips AdamW weight decay on every rank."""
    counts, step_shards = _t15b_spec(world_size)
    ds = _make_uneven_dataset(counts, seed=206, covariates=True)
    model = _fresh_tiny_model(seed=3)
    init_unused = torch.tensor([2.0, -4.0], dtype=torch.float32)
    model.register_parameter("_t15_unused_param", torch.nn.Parameter(init_unused.clone()))
    cfg = _base_config(
        mode="full_ft",
        batch_size=2,
        max_steps=1,
        warmup_steps=0,
        lr_schedule="constant",
        learning_rate=0.1,
        weight_decay=0.5,
        gradient_clip_val=None,
        seed=21,
    )
    trainer = TimestralTrainer(cfg, ds, model=model)
    collator = ds.collator(trainer.model.config)
    indices = step_shards[0][rank]
    assert indices is not None
    shard = collator([ds[i] for i in indices])
    strategy = trainer._make_fit_strategy()
    strategy.prepare_fit()
    trainer.model.train()
    try:
        outcome = strategy._run_step(lambda s=shard: s)
        unused_param = trainer.model._t15_unused_param
        unused_grad_is_none = unused_param.grad is None
        unused_unchanged = torch.equal(unused_param.detach().cpu(), init_unused)
    finally:
        strategy.cleanup()
        trainer.model.eval()
    return {
        "skipped": outcome.skipped,
        "unused_grad_is_none": unused_grad_is_none,
        "unused_unchanged": unused_unchanged,
    }


@pytest.mark.acceptance
@pytest.mark.parametrize("world_size", [2, 4])
def test_t15_b_step_gradients_loss_and_non_finite(world_size: int) -> None:
    """T15(b). Direct internal distributed step (`_run_step`) on constructed batches (unequal,
    zero-scored, and empty `None` shards; `as_double` inputs for `float64` tier) across all four
    modes including LoRA step 1 (both A and B gradients non-zero), all-`None` gradient preservation
    (no spurious AdamW weight decay), plus direct and end-to-end `fit()` single-rank non-finite
    `FloatingPointError` propagation.
    """
    step_cases = [(f"{mode}_{dt}", mode, dt) for dt in ("float64", "float32") for mode in ALL_MODES]
    scenarios: dict[str, Any] = {label: (_sc_t15b_direct_steps, (mode, dt)) for label, mode, dt in step_cases}
    scenarios["all_none_grad"] = _sc_t15b_all_none_grad_param
    scenarios["non_finite_direct"] = _sc_t15b_non_finite_direct
    scenarios["non_finite_fit"] = _sc_t15b_non_finite_fit

    outcomes = run_scenarios(scenarios, world_size=world_size, pg_timeout=PG_TIMEOUT, wall_timeout=WALL_TIMEOUT)

    for label, mode, dt in step_cases:
        tol = STEP_TOL[dt]
        param_tol = TRAJ_TOL[dt]
        ref_steps = _run_t15b_union_ref(world_size, mode, dt)
        rank_outs = unwrap_results(outcomes[label])
        r0_steps = rank_outs[0]["steps"]
        for step_idx, ref_rec in enumerate(ref_steps):
            r0_rec = r0_steps[step_idx]
            for r, out in enumerate(rank_outs):
                rec = out["steps"][step_idx]
                assert rec["skipped"] is False, f"{label} step {step_idx} r{r}: unexpectedly skipped"
                assert torch.equal(rec["grad"], r0_rec["grad"]), (
                    f"{label} step {step_idx} r{r}: cross-rank grads differ"
                )
                assert torch.equal(rec["params"], r0_rec["params"]), (
                    f"{label} step {step_idx} r{r}: cross-rank params differ"
                )
                loss_err = _scalar_rel(rec["loss"], ref_rec["loss"])
                grad_err = l2_rel(rec["grad"], ref_rec["grad"])
                param_err = l2_rel(rec["params"], ref_rec["params"])
                assert loss_err <= tol, f"{label} step {step_idx} r{r}: loss rel diff {loss_err:.2e} > {tol:.1e}"
                assert grad_err <= tol, f"{label} step {step_idx} r{r}: grad rel L2 {grad_err:.2e} > {tol:.1e}"
                assert param_err <= param_tol, (
                    f"{label} step {step_idx} r{r}: param rel L2 {param_err:.2e} > {param_tol:.1e}"
                )
                if mode.startswith("lora_") and step_idx == 1:
                    assert rec["b_max"] > 0.0 and rec["a_grad_norm"] > 0.0 and rec["b_grad_norm"] > 0.0

    none_outs = unwrap_results(outcomes["all_none_grad"])
    for r, out in enumerate(none_outs):
        assert out["skipped"] is False
        assert out["unused_grad_is_none"], f"all_none_grad r{r}: unused trainable param received a non-None grad"
        assert out["unused_unchanged"], f"all_none_grad r{r}: AdamW decayed unused param with all-None grad"

    for nf_key in ("non_finite_direct", "non_finite_fit"):
        nf_outs = unwrap_results(outcomes[nf_key])
        for r, out in enumerate(nf_outs):
            assert out["exc_type"] is FloatingPointError, (
                f"{nf_key} r{r}: expected FloatingPointError, got {out['exc_type']}"
            )
            assert out["unchanged"], f"{nf_key} r{r}: weights mutated on non-finite step"
            assert out["no_grads"], f"{nf_key} r{r}: backward ran on non-finite step"
            if nf_key == "non_finite_fit":
                assert out["eval_mode"], f"{nf_key} r{r}: model left in train mode"
            assert out["followup"] == world_size


# --------------------------------------------------------------------------- T15 (c): zero-scored step & empty epoch


def _t15c_counts(world_size: int) -> list[int]:
    # 3 global batches of size 2 * N: batch 0 scored, batch 1 all zero-scored (W_D == 0), batch 2 scored.
    g = 2 * world_size
    b0 = [8 if i % 2 == 0 else 2 for i in range(g)]
    b1 = [0] * g
    b2 = [3 if i % 2 == 0 else 7 for i in range(g)]
    return b0 + b1 + b2


def _run_t15c_skip(world_size: int, *, per_rank_batch: int) -> dict[str, Any]:
    ds = _make_uneven_dataset(_t15c_counts(world_size), seed=301, covariates=True)
    cfg = _base_config(
        mode="full_ft",
        batch_size=per_rank_batch,
        max_steps=4,
        warmup_steps=1,
        lr_schedule="cosine",
        learning_rate=5e-3,
        weight_decay=0.01,
        gradient_clip_val=1.0,
        seed=23,
    )
    rec = _RecorderCallback()
    trainer = TimestralTrainer(cfg, ds, callbacks=[rec], model=_fresh_tiny_model(seed=3))
    res = trainer.fit()
    first_param = trainer._trainable[0]
    opt_step = int(trainer.optimizer.state[first_param]["step"].item())
    return {
        "result": res,
        "global_step": trainer.global_step,
        "opt_step": opt_step,
        "sched_last_epoch": trainer.scheduler.last_epoch,
        "cb": rec.summary(),
        "params": _flat_params(trainer.model),
    }


def _sc_t15c_mid_epoch_skip(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    return _run_t15c_skip(world_size, per_rank_batch=2)


def _sc_t15c_empty_epoch(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    ds = _make_uneven_dataset([0] * (2 * world_size), seed=302, covariates=True)
    cfg = _base_config(mode="head_only", batch_size=2, max_steps=2, learning_rate=1e-2, seed=29)
    trainer = TimestralTrainer(cfg, ds, model=_fresh_tiny_model(seed=4))
    before = _flat_params(trainer.model).clone()
    exc_type: type[BaseException] | None = None
    exc_msg = ""
    try:
        trainer.fit()
    except Exception as exc:
        exc_type = type(exc)
        exc_msg = str(exc)
    return {
        "exc_type": exc_type,
        "exc_msg": exc_msg,
        "global_step": trainer.global_step,
        "unchanged": torch.equal(_flat_params(trainer.model), before),
        "eval_mode": not trainer.model.training,
        "followup": _followup_collective_sum(world_size),
    }


@pytest.mark.acceptance
@pytest.mark.parametrize("world_size", [2, 4])
def test_t15_c_global_zero_scored_step_and_empty_epoch(world_size: int) -> None:
    """T15(c). Globally zero-scored batch (`W_D == 0`) skips backward/optimiser/scheduler and
    increments `skipped_batches` on every rank; an all-unscored epoch raises `RuntimeError` on every rank.
    """
    ref = _run_t15c_skip(world_size, per_rank_batch=2 * world_size)
    assert ref["result"].steps == 4 and ref["result"].skipped_batches == 2 and ref["result"].epochs == 2

    outcomes = run_scenarios(
        {"mid_skip": _sc_t15c_mid_epoch_skip, "empty_epoch": _sc_t15c_empty_epoch},
        world_size=world_size,
        pg_timeout=PG_TIMEOUT,
        wall_timeout=WALL_TIMEOUT,
    )

    skip_outs = unwrap_results(outcomes["mid_skip"])
    r0 = skip_outs[0]
    assert r0["cb"]["lrs"] == pytest.approx(ref["cb"]["lrs"], rel=1e-12)
    for r, out in enumerate(skip_outs):
        assert out["result"] == r0["result"]
        assert out["result"].steps == 4
        assert out["result"].epochs == 2
        assert out["result"].skipped_batches == 2
        assert out["global_step"] == 4
        assert out["opt_step"] == 4
        assert out["sched_last_epoch"] == 4
        assert torch.equal(out["params"], r0["params"]), f"mid_skip r{r}: cross-rank params differ"
        assert l2_rel(out["params"], ref["params"]) <= TRAJ_TOL["float32"]

    empty_outs = unwrap_results(outcomes["empty_epoch"])
    for r, out in enumerate(empty_outs):
        assert out["exc_type"] is RuntimeError, f"empty_epoch r{r}: expected RuntimeError, got {out['exc_type']}"
        assert "no optimiser step" in out["exc_msg"]
        assert out["global_step"] == 0 and out["unchanged"] and out["eval_mode"]
        assert out["followup"] == world_size


# --------------------------------------------------------------------------- T15 (d): initial broadcast & consistency


def _sc_t15d_initial_broadcast(rank: int, world_size: int, mode: str, base_dir: str) -> dict[str, Any]:
    del world_size
    ds = _make_uneven_dataset([8, 4, 6, 2], seed=401, covariates=True)
    model = _fresh_tiny_model(seed=100 + rank)
    model.register_buffer("_t15_rank_buf", torch.full((4,), float(rank + 1), dtype=torch.float32))
    # `representation.patch_proj.hidden_layer.weight` is never LoRA-wrapped and is frozen in every non-full_ft mode.
    init_body = model.representation.patch_proj.hidden_layer.weight.detach().clone()

    # Deliberately give each rank a distinct `output_dir`: spec §5.6 excludes rank-specific `output_dir` from digest.
    cfg = _base_config(mode=mode, batch_size=2, max_steps=1, seed=31, output_dir=Path(base_dir) / f"out_r{rank}")
    trainer = TimestralTrainer(cfg, ds, lora_config=_lora_for(mode), model=model)
    after_body_param = trainer.model.representation.patch_proj.hidden_layer.weight
    after_body = after_body_param.detach().clone()
    req_grad = {name: p.requires_grad for name, p in sorted(trainer.model.named_parameters())}
    return {
        "all_params": _flat_params(trainer.model),
        "rank_buf": trainer.model._t15_rank_buf.detach().cpu().clone(),
        "body_changed_from_local_init": not torch.equal(init_body, after_body),
        "body_frozen": not after_body_param.requires_grad,
        "req_grad": req_grad,
    }


def _sc_t15d_mismatch_dataset_length(rank: int, world_size: int) -> dict[str, Any]:
    counts = [8, 4, 6] if rank == world_size - 1 else [8, 4, 6, 2]
    ds = _make_uneven_dataset(counts, seed=402, covariates=True)
    cfg = _base_config(mode="head_only", batch_size=2, max_steps=1, seed=31)
    return _record_propagated_exception(
        lambda: TimestralTrainer(cfg, ds, model=_fresh_tiny_model(seed=5)),
        world_size,
    )


def _sc_t15d_mismatch_dataset_content(rank: int, world_size: int) -> dict[str, Any]:
    # Same length (4) on every rank, different target values on rank 1 -> catches checks missing dataset_fingerprint.
    ds = _make_uneven_dataset([8, 4, 6, 2], seed=405 if rank == 1 else 404, covariates=True)
    cfg = _base_config(mode="head_only", batch_size=2, max_steps=1, seed=31)
    return _record_propagated_exception(
        lambda: TimestralTrainer(cfg, ds, model=_fresh_tiny_model(seed=5)),
        world_size,
    )


def _sc_t15d_mismatch_config(rank: int, world_size: int) -> dict[str, Any]:
    bad_ranks = (1,) if world_size == 2 else (1, 3)
    ds = _make_uneven_dataset([8, 4, 6, 2], seed=403, covariates=True)
    cfg = _base_config(mode="head_only", batch_size=3 if rank in bad_ranks else 2, max_steps=1, seed=31)
    return _record_propagated_exception(
        lambda: TimestralTrainer(cfg, ds, model=_fresh_tiny_model(seed=5)),
        world_size,
    )


@pytest.mark.acceptance
@pytest.mark.parametrize("world_size", [2, 4])
def test_t15_d_initial_broadcast_and_consistency(world_size: int, tmp_path: Path) -> None:
    """T15(d). `__init__` broadcasts all parameters (frozen and trainable) and buffers from rank 0
    across all modes, allows rank-specific `output_dir`, and rejects mismatched datasets or configs
    (`ValueError` on mismatched rank(s), `RankFailureError` naming them on matching peers).
    """
    scenarios: dict[str, Any] = {
        f"bcast_{mode}": (_sc_t15d_initial_broadcast, (mode, str(tmp_path))) for mode in ALL_MODES
    }
    scenarios["mismatch_ds_len"] = _sc_t15d_mismatch_dataset_length
    scenarios["mismatch_ds_content"] = _sc_t15d_mismatch_dataset_content
    scenarios["mismatch_cfg"] = _sc_t15d_mismatch_config

    outcomes = run_scenarios(scenarios, world_size=world_size, pg_timeout=PG_TIMEOUT, wall_timeout=WALL_TIMEOUT)

    for mode in ALL_MODES:
        outs = unwrap_results(outcomes[f"bcast_{mode}"])
        r0 = outs[0]
        assert torch.equal(r0["rank_buf"], torch.full((4,), 1.0))
        for r, out in enumerate(outs):
            assert torch.equal(out["all_params"], r0["all_params"]), f"{mode} r{r}: weights differ from rank 0"
            assert torch.equal(out["rank_buf"], r0["rank_buf"]), f"{mode} r{r}: buffer differs from rank 0"
            assert out["req_grad"] == r0["req_grad"], f"{mode} r{r}: requires_grad map differs"
            assert out["body_frozen"] is (mode != "full_ft")
            if r > 0:
                assert out["body_changed_from_local_init"], f"{mode} r{r}: frozen body weights were not broadcast"

    # Decided consistency semantics: mismatched rank(s) raise ValueError; matching peers raise RankFailureError.
    cfg_bad = {1: ValueError} if world_size == 2 else {1: ValueError, 3: ValueError}
    _assert_propagated(
        unwrap_results(outcomes["mismatch_ds_content"]), world_size, {1: ValueError}, "mismatch_ds_content"
    )
    _assert_propagated(
        unwrap_results(outcomes["mismatch_ds_len"]), world_size, {world_size - 1: ValueError}, "mismatch_ds_len"
    )
    _assert_propagated(unwrap_results(outcomes["mismatch_cfg"]), world_size, cfg_bad, "mismatch_cfg")


# --------------------------------------------------------------------------- T15 (e): stop, restore, and TrainingResult


def _sc_t15e_stop_on_start(rank: int, world_size: int) -> dict[str, Any]:
    del rank, world_size
    ds = _make_uneven_dataset([8, 4, 6, 2, 7, 1, 5, 3], seed=501, covariates=True)
    cfg = _base_config(mode="head_only", batch_size=2, max_steps=6, learning_rate=1e-2, seed=41)
    cb = _RecorderCallback(stop_on_start=True)
    trainer = TimestralTrainer(cfg, ds, callbacks=[cb], model=_fresh_tiny_model(seed=6))
    before = _flat_params(trainer.model).clone()
    res = trainer.fit()
    return {
        "result": res,
        "cb": cb.summary(),
        "unchanged": torch.equal(_flat_params(trainer.model), before),
    }


def _sc_t15e_stop_on_step_end(rank: int, world_size: int) -> dict[str, Any]:
    del rank, world_size
    ds = _make_uneven_dataset([8, 4, 6, 2, 7, 1, 5, 3] * 2, seed=502, covariates=True)
    cfg = _base_config(mode="head_only", batch_size=2, max_steps=8, learning_rate=1e-2, seed=42)
    cb = _RecorderCallback(stop_at_step=2)
    trainer = TimestralTrainer(cfg, ds, callbacks=[cb], model=_fresh_tiny_model(seed=7))
    res = trainer.fit()
    return {
        "result": res,
        "global_step": trainer.global_step,
        "cb": cb.summary(),
        "params": _flat_params(trainer.model),
    }


def _run_t15e_early_stop_restore(world_size: int, *, per_rank_batch: int) -> dict[str, Any]:
    train_ds = _make_uneven_dataset([8, 2, 7, 1, 8, 3, 6, 2] * (1 if world_size == 2 else 2), seed=503, covariates=True)
    val_ds = _make_uneven_dataset([8, 1, 6, 2, 7, 3, 5], seed=504, covariates=True)
    # High constant LR degrades validation loss after early steps, triggering EarlyStopping and restore_best.
    cfg = _base_config(
        mode="head_only",
        batch_size=per_rank_batch,
        eval_batch_size=2,
        max_steps=16,
        eval_every=1,
        learning_rate=0.5,
        lr_schedule="constant",
        early_stopping_patience=2,
        restore_best=True,
        seed=43,
    )
    cb = _RecorderCallback()
    trainer = TimestralTrainer(cfg, train_ds, val_ds, callbacks=[cb], model=_fresh_tiny_model(seed=8))
    res = trainer.fit()
    return {
        "result": res,
        "cb": cb.summary(),
        "params": _flat_trainable(trainer),
        "all_params": _flat_params(trainer.model),
    }


def _sc_t15e_early_stop_restore(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    return _run_t15e_early_stop_restore(world_size, per_rank_batch=2)


def _sc_t15e_broadcast_loss_adoption(rank: int, world_size: int) -> dict[str, Any]:
    """Perturb rank 0's reduced step/eval loss sums so non-zero ranks visibly adopt rank 0's broadcast values."""
    train_ds = _make_uneven_dataset([8, 4, 6, 2] * world_size, seed=507, covariates=True)
    val_ds = _make_uneven_dataset([8, 2, 7, 3, 5], seed=508, covariates=True)
    cfg = _base_config(
        mode="head_only",
        batch_size=2,
        eval_batch_size=2,
        max_steps=2,
        eval_every=1,
        learning_rate=1e-2,
        lr_schedule="constant",
        restore_best=False,
        seed=45,
    )
    trainer = TimestralTrainer(cfg, train_ds, val_ds, model=_fresh_tiny_model(seed=8))

    orig_all_reduce = dist.all_reduce
    perturbed_steps = 0
    perturbed_evals = 0
    unperturbed_step_losses: list[float] = []
    unperturbed_eval_losses: list[float] = []

    def _rank0_perturbed_all_reduce(tensor: Tensor, *args: Any, **kwargs: Any) -> Any:
        nonlocal perturbed_steps, perturbed_evals
        work = orig_all_reduce(tensor, *args, **kwargs)
        if tensor.dtype == torch.float64 and tensor.numel() == 8 and tensor[0].item() > 0.0:
            w_d = tensor[0].item()
            unperturbed_step_losses.append(tensor[1].item() / w_d)
            # Perturb sum_d only (slot 1); w_d (slot 0) and gradients remain exact.
            tensor[1].add_(100.0 * w_d)
            perturbed_steps += 1
        elif tensor.dtype == torch.float64 and tensor.numel() == 4 and tensor[1].item() > 0.0:
            scored = tensor[1].item()
            unperturbed_eval_losses.append(tensor[0].item() / scored)
            tensor[0].add_(200.0 * scored)
            perturbed_evals += 1
        return work

    if rank == 0:
        dist.all_reduce = _rank0_perturbed_all_reduce  # type: ignore[assignment]
    try:
        res = trainer.fit()
    finally:
        dist.all_reduce = orig_all_reduce  # type: ignore[assignment]

    return {
        "result": res,
        "perturbed_steps": perturbed_steps,
        "perturbed_evals": perturbed_evals,
        "unperturbed_step_losses": unperturbed_step_losses,
        "unperturbed_eval_losses": unperturbed_eval_losses,
        "all_params": _flat_params(trainer.model),
    }


@pytest.mark.acceptance
@pytest.mark.parametrize("world_size", [2, 4])
def test_t15_e_stop_restore_and_training_result(world_size: int) -> None:
    """T15(e). Stop from `on_fit_start`, `on_step_end`, and `EarlyStopping` halts all ranks together;
    callbacks run on rank 0 only; `restore_best=True` restores best weights on every rank matching
    one-process `fit()`; all ten `TrainingResult` fields are equal on every rank, with non-zero ranks
    visibly adopting rank 0's broadcast step and evaluation losses.
    """
    ref_es = _run_t15e_early_stop_restore(world_size, per_rank_batch=2 * world_size)
    ref_res: TrainingResult = ref_es["result"]
    assert ref_res.stopped_early and ref_res.restored_best
    assert ref_res.best_step is not None and ref_res.best_step < ref_res.steps
    assert ref_res.steps == ref_res.best_step + 2

    outcomes = run_scenarios(
        {
            "stop_start": _sc_t15e_stop_on_start,
            "stop_step": _sc_t15e_stop_on_step_end,
            "early_stop": _sc_t15e_early_stop_restore,
            "broadcast_loss": _sc_t15e_broadcast_loss_adoption,
        },
        world_size=world_size,
        pg_timeout=PG_TIMEOUT,
        wall_timeout=WALL_TIMEOUT,
    )

    # 1. Stop in on_fit_start.
    start_outs = unwrap_results(outcomes["stop_start"])
    assert start_outs[0]["cb"] == {
        "started": 1,
        "ended": 1,
        "steps": [],
        "lrs": [],
        "eval_steps": [],
        "eval_scored": [],
    }
    for r, out in enumerate(start_outs):
        res: TrainingResult = out["result"]
        assert res == start_outs[0]["result"]
        assert res.steps == 0 and res.epochs == 0 and res.stopped_early and res.final_train_loss is None
        assert res.train_history == [] and out["unchanged"]
        if r > 0:
            assert out["cb"]["started"] == 0 and out["cb"]["ended"] == 0 and out["cb"]["steps"] == []

    # 2. Stop in on_step_end.
    step_outs = unwrap_results(outcomes["stop_step"])
    assert step_outs[0]["cb"]["started"] == 1 and step_outs[0]["cb"]["ended"] == 1
    assert step_outs[0]["cb"]["steps"] == [1, 2]
    for r, out in enumerate(step_outs):
        res = out["result"]
        assert res == step_outs[0]["result"]
        assert res.steps == 2 and res.stopped_early and out["global_step"] == 2
        assert torch.equal(out["params"], step_outs[0]["params"])
        if r > 0:
            assert out["cb"]["started"] == 0 and out["cb"]["ended"] == 0 and out["cb"]["steps"] == []

    # 3. EarlyStopping + restore_best=True + 1-process parity.
    es_outs = unwrap_results(outcomes["early_stop"])
    r0_es = es_outs[0]
    assert r0_es["cb"]["started"] == 1 and r0_es["cb"]["ended"] == 1
    assert len(r0_es["cb"]["steps"]) == ref_res.steps
    assert len(r0_es["cb"]["eval_steps"]) == len(ref_res.eval_history)
    for r, out in enumerate(es_outs):
        res = out["result"]
        assert res == r0_es["result"], f"early_stop r{r}: TrainingResult differs from rank 0"
        assert res.steps == ref_res.steps and res.best_step == ref_res.best_step
        assert res.stopped_early and res.restored_best
        assert res.best_eval_loss is not None and ref_res.best_eval_loss is not None
        assert _scalar_rel(res.best_eval_loss, ref_res.best_eval_loss) <= TRAJ_TOL["float32"]
        assert torch.equal(out["all_params"], r0_es["all_params"]), f"early_stop r{r}: restored weights differ"
        assert l2_rel(out["params"], ref_es["params"]) <= TRAJ_TOL["float32"]
        if r > 0:
            assert out["cb"] == {"started": 0, "ended": 0, "steps": [], "lrs": [], "eval_steps": [], "eval_scored": []}

    # 4. Non-zero ranks adopt rank 0's broadcast step & eval loss values (catches M3a and M3b).
    bl_outs = unwrap_results(outcomes["broadcast_loss"])
    r0_bl = bl_outs[0]
    assert r0_bl["perturbed_steps"] == 2 and r0_bl["perturbed_evals"] == 2
    for (_step, recorded), raw in zip(r0_bl["result"].train_history, r0_bl["unperturbed_step_losses"], strict=True):
        assert recorded == pytest.approx(raw + 100.0, rel=1e-9)
    for (_step, recorded), raw in zip(r0_bl["result"].eval_history, r0_bl["unperturbed_eval_losses"], strict=True):
        assert recorded == pytest.approx(raw + 200.0, rel=1e-9)
    for r, out in enumerate(bl_outs):
        assert out["result"] == r0_bl["result"], (
            f"broadcast_loss r{r}: rank did not adopt rank 0's broadcast step/eval losses"
        )
        assert torch.equal(out["all_params"], r0_bl["all_params"])
        if r > 0:
            assert out["perturbed_steps"] == 0 and out["perturbed_evals"] == 0


def _run_t15e_set_clear_stop(world_size: int, *, mode: str, per_rank_batch: int) -> dict[str, Any]:
    train_ds = _make_uneven_dataset([8, 4, 6, 2] * world_size, seed=505, covariates=True)
    val_ds = _make_uneven_dataset([8, 2, 7, 3, 5], seed=506, covariates=True)
    cfg = _base_config(
        mode="head_only",
        batch_size=per_rank_batch,
        eval_batch_size=2,
        max_steps=4,
        eval_every=1,
        learning_rate=1e-2,
        lr_schedule="constant",
        restore_best=False,
        seed=44,
    )
    cb = _SetClearStopCallback(mode)
    trainer = TimestralTrainer(cfg, train_ds, val_ds, callbacks=[cb], model=_fresh_tiny_model(seed=8))
    res = trainer.fit()
    return {
        "result": res,
        "should_stop": trainer.should_stop,
        "cb_steps": list(cb.steps),
        "cb_evals": list(cb.evals),
        "params": _flat_trainable(trainer),
        "all_params": _flat_params(trainer.model),
    }


def _worker_t15e_should_stop_cleared(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    # Reachable set->clear order 1: `on_fit_start` sets True, post-loop step-0 `on_evaluate` clears False
    # (`_run_fit_loop` skips the while loop, then runs final `evaluate_now()` because `last_evaluated_step != 0`).
    start_to_eval = _run_t15e_set_clear_stop(world_size, mode="fit_start_to_step0_eval", per_rank_batch=2)
    dist.barrier()
    # Reachable set->clear order 2: `on_step_end` sets True, same-step `on_evaluate` (`eval_every=1`) clears False
    # before the loop's `if self.global_step >= final_step or self.should_stop: break` check.
    # (Reverse orders -- `on_evaluate` -> next `on_step_end`, and `on_fit_start` -> first `on_step_end` -- are
    # unreachable in `_run_fit_loop`, which immediately checks `self.should_stop` before entering the next step.)
    step_to_eval = _run_t15e_set_clear_stop(world_size, mode="step_end_to_eval", per_rank_batch=2)
    return {
        "fit_start_to_step0_eval": start_to_eval,
        "step_end_to_eval": step_to_eval,
    }


@pytest.mark.acceptance
@pytest.mark.parametrize("world_size", [2, 4])
def test_t15_e_should_stop_cleared_after_set(world_size: int) -> None:
    """T15(e) regression. Clearing `trainer.should_stop` after an earlier hook set it must sync
    `should_stop = False` on every non-zero rank so training continues to `max_steps` without
    deadlocking and matches one-process `fit()` with the same callback.
    """
    ref_step = _run_t15e_set_clear_stop(world_size, mode="step_end_to_eval", per_rank_batch=2 * world_size)
    assert ref_step["result"].steps == 4 and not ref_step["result"].stopped_early
    assert ref_step["should_stop"] is False
    assert ref_step["cb_steps"] == [1, 2, 3, 4] and ref_step["cb_evals"] == [1, 2, 3, 4]

    ref_start = _run_t15e_set_clear_stop(world_size, mode="fit_start_to_step0_eval", per_rank_batch=2 * world_size)
    assert ref_start["result"].steps == 0 and ref_start["result"].stopped_early
    assert ref_start["should_stop"] is False
    assert ref_start["cb_steps"] == [] and ref_start["cb_evals"] == [0]

    outs = unwrap_results(
        run_distributed(
            _worker_t15e_should_stop_cleared,
            world_size=world_size,
            pg_timeout=3.0,
            wall_timeout=15.0,
        )
    )

    r0_step = outs[0]["step_end_to_eval"]
    assert r0_step["cb_steps"] == [1, 2, 3, 4] and r0_step["cb_evals"] == [1, 2, 3, 4]
    for r, out in enumerate(outs):
        se = out["step_end_to_eval"]
        assert se["result"] == r0_step["result"], f"step_end_to_eval r{r}: TrainingResult differs from rank 0"
        assert se["result"].steps == 4 and not se["result"].stopped_early
        assert se["should_stop"] is False, f"step_end_to_eval r{r}: should_stop left True on non-zero rank"
        assert torch.equal(se["all_params"], r0_step["all_params"]), f"step_end_to_eval r{r}: params differ"
        assert l2_rel(se["params"], ref_step["params"]) <= TRAJ_TOL["float32"]
        for (s, loss), (rs, rloss) in zip(se["result"].train_history, ref_step["result"].train_history, strict=True):
            assert s == rs and _scalar_rel(loss, rloss) <= TRAJ_TOL["float32"]
        for (s, loss), (rs, rloss) in zip(se["result"].eval_history, ref_step["result"].eval_history, strict=True):
            assert s == rs and _scalar_rel(loss, rloss) <= TRAJ_TOL["float32"]
        if r > 0:
            assert se["cb_steps"] == [] and se["cb_evals"] == []

        fs = out["fit_start_to_step0_eval"]
        assert fs["result"] == outs[0]["fit_start_to_step0_eval"]["result"]
        assert fs["result"].steps == 0 and fs["result"].stopped_early
        assert fs["should_stop"] is False, f"fit_start_to_step0_eval r{r}: should_stop left True on non-zero rank"
        if r == 0:
            assert fs["cb_steps"] == [] and fs["cb_evals"] == [0]
        else:
            assert fs["cb_steps"] == [] and fs["cb_evals"] == []


# --------------------------------------------------------------------------- T15 (f): sampling, workers, and flips


def _sc_t15f_shuffled_epochs(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    # N*B + (N + 1) windows with B=3: trailing R = N + 1 splits evenly ([2,1] at N=2, [2,1,1,1] at N=4),
    # distinguishing balanced splitting from contiguous-B splitting ([3,0] / [3,2,0,0]) at both N=2 and N=4.
    n_items = world_size * 3 + world_size + 1
    counts = [8 - (i % 7) for i in range(n_items)]
    ds = _make_uneven_dataset(counts, seed=601, covariates=True, recording=True)
    assert isinstance(ds, _RecordingDataset)

    cfg = _base_config(
        mode="lora_temporal",
        batch_size=3,
        epochs=3,
        max_steps=None,
        learning_rate=1e-2,
        shuffle=True,
        sign_flip=False,
        seed=51,
    )
    trainer = TimestralTrainer(cfg, ds, lora_config=_lora_for("lora_temporal"), model=_fresh_tiny_model(seed=9))
    ds.accessed_indices.clear()
    rows_per_series = ds.rows_per_series
    step_sizes: list[int] = []
    handle = trainer.model.transformer.register_forward_pre_hook(
        lambda _m, inp: step_sizes.append(inp[0].shape[0] // rows_per_series)
    )
    try:
        res = trainer.fit()
    finally:
        handle.remove()

    return {
        "n_items": n_items,
        "result": res,
        "accessed": list(ds.accessed_indices),
        "step_sizes": step_sizes,
        "params": _flat_params(trainer.model),
    }


def _run_t15f_sign_flip(world_size: int, *, per_rank_batch: int, max_steps: int | None) -> dict[str, Any]:
    # 8 full global steps + 1 trailing window -> R = 1 < N (ranks 1..N-1 empty on step 8).
    n_items = 8 * world_size * 2 + 1
    counts = [8 - (i % 6) for i in range(n_items)]
    ds = _make_uneven_dataset(counts, seed=602, covariates=False, positive_only=True)
    cfg = _base_config(
        mode="lora_temporal",
        batch_size=per_rank_batch,
        epochs=1,
        max_steps=max_steps,
        learning_rate=1e-2,
        shuffle=False,
        sign_flip=True,
        seed=61,
    )
    trainer = TimestralTrainer(cfg, ds, lora_config=_lora_for("lora_temporal"), model=_fresh_tiny_model(seed=10))
    flips: list[bool] = []
    orig_scale = trainer.model.scale

    def _recording_scale(inputs: Tensor, mask: Tensor):
        # Strictly positive series (>= 20.0) become strictly negative iff flipped.
        flips.extend(bool(v < 0.0) for v in inputs[:, 0].detach().cpu().tolist())
        return orig_scale(inputs, mask)

    trainer.model.scale = _recording_scale  # type: ignore[method-assign]
    res = trainer.fit()
    return {"result": res, "flips": flips, "params": _flat_params(trainer.model)}


def _sc_t15f_sign_flip(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    return _run_t15f_sign_flip(world_size, per_rank_batch=2, max_steps=None)


def _run_t15f_lora_dropout(world_size: int, *, per_rank_batch: int, copies: int) -> dict[str, Any]:
    """Run `lora_variate` with active dropout and record packed zero-masks from `LoRASliceLinear.dropout`."""
    base_ds = _make_uneven_dataset([8, 6], seed=604, covariates=True)
    series_copies = [
        SeriesData(
            f"d{c}_{s.series_id}",
            s.timestamps.copy(),
            s.targets.copy(),
            s.past.copy(),
            s.known.copy(),
            s.categorical_past.copy(),
            s.categorical_known.copy(),
        )
        for c in range(copies)
        for s in base_ds.series
    ]
    ds = TimeSeriesDataset(
        series_copies,
        context_length=base_ds.context_length,
        forecast_horizon=base_ds.forecast_horizon,
        stride=base_ds.forecast_horizon,
        min_context=base_ds.context_length,
        roles=base_ds.roles,
    )
    cfg = _base_config(
        mode="lora_variate",
        batch_size=per_rank_batch,
        epochs=1,
        max_steps=2,
        learning_rate=1e-2,
        shuffle=False,
        sign_flip=False,
        seed=63,
    )
    trainer = TimestralTrainer(
        cfg, ds, lora_config=_lora_for("lora_variate", dropout=0.5), model=_fresh_tiny_model(seed=10)
    )
    _activate_lora_b(trainer.model)

    masks: list[bytes] = []
    dropped_counts: list[int] = []
    seen_initial_seeds: list[int] = []
    handles = []
    for mod in trainer.model.modules():
        if isinstance(mod, torch.nn.Dropout):

            def _record_drop(_m: torch.nn.Module, inp: tuple[Tensor, ...], out: Tensor) -> None:
                if not seen_initial_seeds:
                    seen_initial_seeds.append(int(torch.initial_seed()))
                dropped = (out.detach() == 0) & (inp[0].detach() != 0)
                dropped_counts.append(int(dropped.sum().item()))
                masks.append(np.packbits(dropped.cpu().numpy().reshape(-1).astype(np.uint8)).tobytes())

            handles.append(mod.register_forward_hook(_record_drop))
    assert handles, "expected at least one LoRA nn.Dropout module"
    try:
        res = trainer.fit()
    finally:
        for h in handles:
            h.remove()
    return {
        "result": res,
        "initial_seed": seen_initial_seeds[0],
        "masks": masks,
        "dropped_counts": dropped_counts,
        "params": _flat_params(trainer.model),
    }


def _sc_t15f_lora_dropout(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    # 2 identical windows per rank across 2 steps so any mask difference comes solely from rank-offset RNG.
    return _run_t15f_lora_dropout(world_size, per_rank_batch=2, copies=world_size * 2)


def _run_t15f_short_dataset(
    world_size: int, *, with_val: bool, per_rank_batch: int, num_workers: int = 0, shuffle: bool = False
) -> dict[str, Any]:
    """Train on a dataset shorter than `world_size` (`len = world_size - 1`) and count `DataLoader.__iter__`."""
    train_counts = [8, 2, 6][: world_size - 1]
    val_counts = [7, 3, 5][: world_size - 1]
    train_ds = _make_uneven_dataset(train_counts, seed=605, covariates=True, recording=(num_workers == 0))
    val_ds = (
        _make_uneven_dataset(val_counts, seed=606, covariates=True, recording=(num_workers == 0)) if with_val else None
    )
    cfg = _base_config(
        mode="head_only",
        batch_size=per_rank_batch,
        eval_batch_size=2,
        epochs=2,
        max_steps=None,
        eval_every=1 if with_val else None,
        learning_rate=1e-2,
        lr_schedule="constant",
        shuffle=shuffle,
        sign_flip=False,
        restore_best=False,
        num_workers=num_workers,
        seed=65,
    )
    trainer = TimestralTrainer(cfg, train_ds, val_ds, model=_fresh_tiny_model(seed=11))
    if isinstance(train_ds, _RecordingDataset):
        train_ds.accessed_indices.clear()
    if isinstance(val_ds, _RecordingDataset):
        val_ds.accessed_indices.clear()

    iter_calls = 0
    orig_iter = DataLoader.__iter__

    def _spied_iter(self_loader: DataLoader[Any]) -> Any:
        nonlocal iter_calls
        iter_calls += 1
        return orig_iter(self_loader)

    DataLoader.__iter__ = _spied_iter  # type: ignore[method-assign]
    try:
        res = trainer.fit()
    finally:
        DataLoader.__iter__ = orig_iter  # type: ignore[method-assign]

    return {
        "result": res,
        "iter_calls": iter_calls,
        "train_accessed": list(train_ds.accessed_indices) if isinstance(train_ds, _RecordingDataset) else None,
        "val_accessed": list(val_ds.accessed_indices) if isinstance(val_ds, _RecordingDataset) else None,
        "params": _flat_trainable(trainer),
        "all_params": _flat_params(trainer.model),
    }


def _sc_t15f_short_dataset(rank: int, world_size: int, with_val: bool) -> dict[str, Any]:
    del rank
    return _run_t15f_short_dataset(world_size, with_val=with_val, per_rank_batch=2, num_workers=0, shuffle=False)


@pytest.mark.acceptance
@pytest.mark.parametrize("world_size", [2, 4])
def test_t15_f_sampling_and_sign_flips(world_size: int) -> None:
    """T15(f). Shuffled multi-epoch `fit()` uses distinct per-epoch permutations, equal step counts,
    exact once-per-epoch union coverage, and balanced trailing splits; `sign_flip=True` and LoRA
    dropout use rank-0's one-process RNG stream while peers draw pairwise distinct streams; datasets
    shorter than `world_size` (with and without `val_dataset`) never create a `DataLoader` iterator
    on idle ranks and match one-process `fit()`.
    """
    ref_flip = _run_t15f_sign_flip(world_size, per_rank_batch=2, max_steps=8)
    assert len(ref_flip["flips"]) == 16

    ref_short_noval = _run_t15f_short_dataset(world_size, with_val=False, per_rank_batch=2 * world_size)
    ref_short_val = _run_t15f_short_dataset(world_size, with_val=True, per_rank_batch=2 * world_size)

    outcomes = run_scenarios(
        {
            "shuffled": _sc_t15f_shuffled_epochs,
            "sign_flip": _sc_t15f_sign_flip,
            "lora_dropout": _sc_t15f_lora_dropout,
            "short_no_val": (_sc_t15f_short_dataset, (False,)),
            "short_with_val": (_sc_t15f_short_dataset, (True,)),
        },
        world_size=world_size,
        pg_timeout=PG_TIMEOUT,
        wall_timeout=WALL_TIMEOUT,
    )

    # 1. Shuffled multi-epoch sampling checks.
    shuf_outs = unwrap_results(outcomes["shuffled"])
    n_items = shuf_outs[0]["n_items"]
    n_steps = steps_per_epoch(n_items, world_size, 3)
    assert n_steps == 2  # 1 full step + 1 trailing step of size N + 1
    for r, out in enumerate(shuf_outs):
        assert out["result"].epochs == 3
        assert out["result"].steps == 3 * n_steps
        assert len(out["step_sizes"]) == 3 * n_steps
        assert sum(out["step_sizes"]) == len(out["accessed"])
        assert torch.equal(out["params"], shuf_outs[0]["params"]), f"shuffled r{r}: weights diverged"

    # Reconstruct per-epoch, per-step index lists per rank and verify union coverage & trailing balance.
    per_rank_batches: list[list[list[int]]] = []
    for out in shuf_outs:
        batches: list[list[int]] = []
        cursor = 0
        for sz in out["step_sizes"]:
            batches.append(out["accessed"][cursor : cursor + sz])
            cursor += sz
        per_rank_batches.append(batches)

    epoch_orders: list[list[int]] = []
    for epoch in range(3):
        order: list[int] = []
        for step in range(n_steps):
            global_step_idx = epoch * n_steps + step
            sizes = [len(per_rank_batches[r][global_step_idx]) for r in range(world_size)]
            assert max(sizes) - min(sizes) <= 1, f"epoch {epoch} step {step}: unbalanced sizes {sizes}"
            for r in range(world_size):
                order.extend(per_rank_batches[r][global_step_idx])
        assert sorted(order) == list(range(n_items)), f"epoch {epoch}: did not cover 0..{n_items - 1} once"
        epoch_orders.append(order)
    assert epoch_orders[0] != epoch_orders[1] and epoch_orders[1] != epoch_orders[2]

    # 2. Sign-flip with empty trailing rank + rank-0 vs peer streams.
    flip_outs = unwrap_results(outcomes["sign_flip"])
    for _r, out in enumerate(flip_outs):
        assert out["result"].steps == 9  # 8 full steps + 1 trailing step
        assert out["result"] == flip_outs[0]["result"]
        assert torch.equal(out["params"], flip_outs[0]["params"])
    # Rank 0 draws 16 flips on the 8 full steps (+1 on the trailing step = 17); peers draw 16 flips (empty on step 8).
    assert len(flip_outs[0]["flips"]) == 17
    assert flip_outs[0]["flips"][:16] == ref_flip["flips"]
    seen_streams = {tuple(flip_outs[0]["flips"][:16])}
    for r in range(1, world_size):
        assert len(flip_outs[r]["flips"]) == 16
        stream = tuple(flip_outs[r]["flips"])
        assert stream not in seen_streams, f"rank {r} flip stream collided with another rank"
        seen_streams.add(stream)

    # 3. Rank-offset global seed (LoRA dropout stream): rank 0 keeps config.seed while all ranks draw distinct masks.
    drop_outs = unwrap_results(outcomes["lora_dropout"])
    seen_drop_streams: set[tuple[bytes, ...]] = set()
    for r, out in enumerate(drop_outs):
        assert len(out["masks"]) > 0 and all(c > 0 for c in out["dropped_counts"]), (
            f"lora_dropout r{r}: dropout did not activate"
        )
        key = tuple(out["masks"])
        assert key not in seen_drop_streams, f"lora_dropout r{r}: shared dropout mask stream with another rank"
        seen_drop_streams.add(key)
        assert out["initial_seed"] == rank_seed(63, r), (
            f"lora_dropout r{r}: initial_seed {out['initial_seed']} != {rank_seed(63, r)}"
        )
        assert out["result"] == drop_outs[0]["result"]
        assert torch.equal(out["params"], drop_outs[0]["params"])

    # 4. Dataset shorter than world_size (`len = N - 1`), with and without val_dataset: idle rank never creates a
    # DataLoader iterator and results match one-process fit().
    for key, ref_short, has_val in (
        ("short_no_val", ref_short_noval, False),
        ("short_with_val", ref_short_val, True),
    ):
        s_outs = unwrap_results(outcomes[key])
        r0_s = s_outs[0]
        assert ref_short["result"].steps == 2 and ref_short["result"].epochs == 2
        for r, out in enumerate(s_outs):
            assert out["result"] == r0_s["result"], f"{key} r{r}: TrainingResult differs from rank 0"
            assert out["result"].steps == 2 and out["result"].epochs == 2
            assert torch.equal(out["all_params"], r0_s["all_params"])
            assert l2_rel(out["params"], ref_short["params"]) <= TRAJ_TOL["float32"]
            for (s, loss), (rs, rloss) in zip(
                out["result"].train_history, ref_short["result"].train_history, strict=True
            ):
                assert s == rs and _scalar_rel(loss, rloss) <= TRAJ_TOL["float32"]
            if has_val:
                for (s, loss), (rs, rloss) in zip(
                    out["result"].eval_history, ref_short["result"].eval_history, strict=True
                ):
                    assert s == rs and _scalar_rel(loss, rloss) <= TRAJ_TOL["float32"]
            if r == world_size - 1:
                assert out["iter_calls"] == 0, f"{key} r{r}: idle rank created a DataLoader iterator"
                assert out["train_accessed"] == []
                if has_val:
                    assert out["val_accessed"] == []
            else:
                assert out["iter_calls"] >= 2, f"{key} r{r}: active rank did not iterate its DataLoader"


def _sc_t15f_workers(rank: int, world_size: int, num_workers: int) -> dict[str, Any]:
    del rank
    n_items = world_size * 2 + 1  # trailing R = 1 < N (empty ranks)
    ds = _make_uneven_dataset([8 - (i % 6) for i in range(n_items)], seed=603, covariates=True)
    # Validation set larger than N on uneven shards + short validation set (< N) tested in the short case below.
    val_ds = _make_uneven_dataset([8, 2, 7, 1, 6][: world_size + 1], seed=607, covariates=True)
    cfg = _base_config(
        mode="head_only",
        batch_size=2,
        eval_batch_size=2,
        epochs=2,
        max_steps=None,
        eval_every=1,
        learning_rate=1e-2,
        shuffle=True,
        sign_flip=False,
        restore_best=False,
        num_workers=num_workers,
        seed=67,
    )
    trainer = TimestralTrainer(cfg, ds, val_ds, model=_fresh_tiny_model(seed=11))
    res = trainer.fit()
    trailing = {"result": res, "params": _flat_params(trainer.model)}

    # Also test len(train_dataset) < N and len(val_dataset) < N with num_workers (idle rank never creates an iterator).
    short = _run_t15f_short_dataset(world_size, with_val=True, per_rank_batch=2, num_workers=num_workers, shuffle=True)
    return {"trailing": trailing, "short": short}


def _worker_t15f_workers_paired(rank: int, world_size: int) -> dict[str, Any]:
    """Run `num_workers=0` reference first, then `num_workers=2` in one spawned process group."""
    w0 = _sc_t15f_workers(rank, world_size, 0)
    dist.barrier()
    w2 = _sc_t15f_workers(rank, world_size, 2)
    return {"w0": w0, "w2": w2}


@pytest.mark.acceptance
@pytest.mark.skipif(WORKERS_HANG_AT_EXIT, reason="PyTorch < 2.8 on macOS: DataLoader worker hangs at exit")
def test_t15_f_dataloader_workers() -> None:
    """T15(f). Multi-process `fit()` at `world_size=4` with `num_workers=2` matches `num_workers=0`
    across (1) uneven trailing training shards + `val_dataset`, and (2) `len(train_dataset) < N`
    with `len(val_dataset) < N` where idle ranks never create a `DataLoader` iterator.

    Runs at `world_size=4` (which exercises active non-zero peers, uneven trailing shards, empty
    trailing ranks, and idle ranks in one spawn) with a generous process-group timeout (30s) because
    on macOS each active `DataLoader` worker waits `MP_STATUS_CHECK_INTERVAL` (~5s) on shutdown.
    """
    world_size = 4
    outs = unwrap_results(
        run_distributed(
            _worker_t15f_workers_paired,
            world_size=world_size,
            pg_timeout=30.0,
            wall_timeout=90.0,
        )
    )
    expected_steps = 2 * steps_per_epoch(world_size * 2 + 1, world_size, 2)
    r0_w0_trail = outs[0]["w0"]["trailing"]
    r0_w0_short = outs[0]["w0"]["short"]
    for r, out in enumerate(outs):
        w2_trail = out["w2"]["trailing"]
        assert w2_trail["result"].steps == expected_steps
        assert len(w2_trail["result"].eval_history) == expected_steps
        assert w2_trail["result"] == r0_w0_trail["result"]
        assert torch.equal(w2_trail["params"], r0_w0_trail["params"])

        w2_short = out["w2"]["short"]
        assert w2_short["result"].steps == 2 and w2_short["result"].epochs == 2
        assert len(w2_short["result"].eval_history) == 2
        assert w2_short["result"] == r0_w0_short["result"]
        assert torch.equal(w2_short["all_params"], r0_w0_short["all_params"])
        if r == world_size - 1:
            assert w2_short["iter_calls"] == 0, f"workers short r{r}: idle rank created a DataLoader iterator"
        else:
            assert w2_short["iter_calls"] >= 2


# --------------------------------------------------------------------------- T15 (g): validation & post-fit evaluate()


def _t15g_val_counts(world_size: int) -> list[int]:
    # 2 * N + 3 windows with strongly unequal missingness so every shard has a different mean than the full set.
    base = [8, 1, 7, 1, 8, 2, 1, 6, 1, 8, 3]
    return base[: 2 * world_size + 3]


def _activate_lora_b(model: TuneTimestralModel) -> None:
    """Populate `lora_B` with deterministic non-zero weights so active LoRA dropout visibly perturbs outputs."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_B." in name:
                param.fill_(0.25)


def _run_t15g_eval_ref(
    val_counts: Sequence[int], *, dtype_name: str = "float32", mode: str = "full_ft", dropout: float = 0.0
) -> EvalResult:
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    train_ds = _make_uneven_dataset([8, 6, 4, 2], seed=700, covariates=True)
    val_ds = _make_uneven_dataset(val_counts, seed=701, covariates=True)
    cfg = _base_config(mode=mode, batch_size=2, eval_batch_size=2, max_steps=1, seed=71)
    ctx_mgr = float64_reference() if dtype == torch.float64 else torch.enable_grad()
    with ctx_mgr:
        trainer = TimestralTrainer(
            cfg,
            train_ds,
            val_ds,
            lora_config=_lora_for(mode, dropout=dropout),
            model=_fresh_tiny_model(seed=12, dtype=dtype),
        )
        if mode.startswith("lora_") and dropout > 0.0:
            _activate_lora_b(trainer.model)
        return trainer.evaluate()


def _sc_t15g_sharded_val_and_rank0_eval(
    rank: int, world_size: int, dtype_name: str, mode: str, dropout: float
) -> dict[str, Any]:
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    train_ds = _make_uneven_dataset([8, 6, 4, 2], seed=700, covariates=True)
    val_counts = _t15g_val_counts(world_size)
    val_ds = _make_uneven_dataset(val_counts, seed=701, covariates=True)
    cfg = _base_config(mode=mode, batch_size=2, eval_batch_size=2, max_steps=1, seed=71)
    cb = _RecorderCallback(stop_on_start=True)  # triggers step-0 internal validation on exact initial weights

    ctx_mgr = float64_reference() if dtype == torch.float64 else torch.enable_grad()
    with ctx_mgr:
        trainer = TimestralTrainer(
            cfg,
            train_ds,
            val_ds,
            lora_config=_lora_for(mode, dropout=dropout),
            callbacks=[cb],
            model=_fresh_tiny_model(seed=12, dtype=dtype),
        )
        if mode.startswith("lora_") and dropout > 0.0:
            _activate_lora_b(trainer.model)
        rows_per_series = val_ds.rows_per_series
        seen_windows: list[int] = []
        handle = trainer.model.transformer.register_forward_pre_hook(
            lambda _m, inp: seen_windows.append(inp[0].shape[0] // rows_per_series)
        )
        try:
            res = trainer.fit()
        finally:
            handle.remove()
        val_windows_seen = sum(seen_windows)

        # Call public evaluate() on rank 0 ALONE after fit(): must not hang and must score the full val_dataset.
        rank0_eval = trainer.evaluate() if rank == 0 else None

    return {
        "result": res,
        "cb": cb.summary(),
        "val_windows_seen": val_windows_seen,
        "rank0_eval": rank0_eval,
    }


def _sc_t15g_val_smaller_than_world(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    train_ds = _make_uneven_dataset([8, 6, 4, 2], seed=700, covariates=True)
    val_counts = [8, 2, 5][: world_size - 1]
    val_ds = _make_uneven_dataset(val_counts, seed=701, covariates=True)
    cfg = _base_config(mode="full_ft", batch_size=2, eval_batch_size=2, max_steps=1, seed=72)
    trainer = TimestralTrainer(
        cfg, train_ds, val_ds, callbacks=[_RecorderCallback(stop_on_start=True)], model=_fresh_tiny_model(seed=12)
    )
    rows_per_series = val_ds.rows_per_series
    seen: list[int] = []
    handle = trainer.model.transformer.register_forward_pre_hook(
        lambda _m, inp: seen.append(inp[0].shape[0] // rows_per_series)
    )
    try:
        res = trainer.fit()
    finally:
        handle.remove()
    return {"result": res, "val_windows_seen": sum(seen)}


def _sc_t15g_val_one_rank_zero_scored(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    # 2 * N windows; rank 1's stride indices (1, 1 + N) have 0 scored positions while other ranks are scored.
    val_counts = [0 if i % world_size == 1 else (8 - (i % 5)) for i in range(2 * world_size)]
    train_ds = _make_uneven_dataset([8, 6, 4, 2], seed=700, covariates=True)
    val_ds = _make_uneven_dataset(val_counts, seed=701, covariates=True)
    cfg = _base_config(mode="full_ft", batch_size=2, eval_batch_size=2, max_steps=1, seed=73)
    trainer = TimestralTrainer(
        cfg, train_ds, val_ds, callbacks=[_RecorderCallback(stop_on_start=True)], model=_fresh_tiny_model(seed=12)
    )
    res = trainer.fit()
    return {"result": res}


def _sc_t15g_val_all_zero_scored(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    train_ds = _make_uneven_dataset([8, 6, 4, 2], seed=700, covariates=True)
    val_ds = _make_uneven_dataset([0] * world_size, seed=701, covariates=True)
    cfg = _base_config(mode="full_ft", batch_size=2, eval_batch_size=2, max_steps=1, seed=74)
    trainer = TimestralTrainer(
        cfg, train_ds, val_ds, callbacks=[_RecorderCallback(stop_on_start=True)], model=_fresh_tiny_model(seed=12)
    )
    exc_type: type[BaseException] | None = None
    exc_msg = ""
    try:
        trainer.fit()
    except Exception as exc:
        exc_type = type(exc)
        exc_msg = str(exc)
    return {"exc_type": exc_type, "exc_msg": exc_msg, "followup": _followup_collective_sum(world_size)}


def _sc_t15g_training_mode_restored_after_val(rank: int, world_size: int) -> dict[str, Any]:
    """Verify `model.training` is restored to `True` (and LoRA dropout stays active) after internal validation."""
    del rank
    train_ds = _make_uneven_dataset([8, 4] * world_size, seed=705, covariates=True)
    val_ds = _make_uneven_dataset([8, 6] * world_size, seed=706, covariates=True)
    cfg = _base_config(
        mode="lora_variate",
        batch_size=2,
        eval_batch_size=2,
        max_steps=2,
        eval_every=1,
        learning_rate=1e-2,
        shuffle=False,
        sign_flip=False,
        restore_best=False,
        seed=75,
    )
    trainer = TimestralTrainer(
        cfg,
        train_ds,
        val_ds,
        lora_config=_lora_for("lora_variate", dropout=0.5),
        model=_fresh_tiny_model(seed=12),
    )
    _activate_lora_b(trainer.model)

    first_dropout = next(m for m in trainer.model.modules() if isinstance(m, torch.nn.Dropout))
    trace: list[tuple[bool, bool, int]] = []

    def _record(_mod: torch.nn.Module, inp: tuple[Tensor, ...], out: Tensor) -> None:
        dropped = int(((out.detach() == 0) & (inp[0].detach() != 0)).sum().item())
        trace.append((trainer.model.training, first_dropout.training, dropped))

    handle = first_dropout.register_forward_hook(_record)
    try:
        res = trainer.fit()
    finally:
        handle.remove()
    return {
        "result": res,
        "trace": trace,
        "post_fit_training": trainer.model.training,
        "params": _flat_params(trainer.model),
    }


@pytest.mark.acceptance
@pytest.mark.parametrize("world_size", [2, 4])
def test_t15_g_validation_and_post_fit_evaluate(world_size: int) -> None:
    """T15(g). Internal sharded validation matches one-process `evaluate()` in `float64` (`1e-12`)
    and `float32` (`1e-5`) including under LoRA dropout, restores `model.training` after validation
    so subsequent training steps keep LoRA dropout active, handles `len(val) < N` and per-rank
    zero-scored shards, raises `ValueError` on every rank when globally unscored, and leaves rank-0
    post-`fit()` `evaluate()` collective-free on the full validation set.
    """
    val_counts = _t15g_val_counts(world_size)
    ref_f64 = _run_t15g_eval_ref(val_counts, dtype_name="float64", mode="full_ft")
    ref_f32 = _run_t15g_eval_ref(val_counts, dtype_name="float32", mode="full_ft")
    ref_lora = _run_t15g_eval_ref(val_counts, dtype_name="float32", mode="lora_variate", dropout=0.5)
    ref_small = _run_t15g_eval_ref([8, 2, 5][: world_size - 1], dtype_name="float32", mode="full_ft")
    one_zero_counts = [0 if i % world_size == 1 else (8 - (i % 5)) for i in range(2 * world_size)]
    ref_one_zero = _run_t15g_eval_ref(one_zero_counts, dtype_name="float32", mode="full_ft")

    outcomes = run_scenarios(
        {
            "val_f64": (_sc_t15g_sharded_val_and_rank0_eval, ("float64", "full_ft", 0.0)),
            "val_f32": (_sc_t15g_sharded_val_and_rank0_eval, ("float32", "full_ft", 0.0)),
            "val_lora_drop": (_sc_t15g_sharded_val_and_rank0_eval, ("float32", "lora_variate", 0.5)),
            "val_small": _sc_t15g_val_smaller_than_world,
            "val_one_zero": _sc_t15g_val_one_rank_zero_scored,
            "val_all_zero": _sc_t15g_val_all_zero_scored,
            "val_restore_train": _sc_t15g_training_mode_restored_after_val,
        },
        world_size=world_size,
        pg_timeout=PG_TIMEOUT,
        wall_timeout=WALL_TIMEOUT,
    )

    for key, ref_ev, tol in (
        ("val_f64", ref_f64, EVAL_TOL["float64"]),
        ("val_f32", ref_f32, EVAL_TOL["float32"]),
        ("val_lora_drop", ref_lora, EVAL_TOL["float32"]),
    ):
        outs = unwrap_results(outcomes[key])
        assert outs[0]["cb"]["eval_scored"] == [ref_ev.scored_positions]
        # Rank-0-only post-fit evaluate() scored the entire validation dataset without hanging.
        r0_eval: EvalResult = outs[0]["rank0_eval"]
        assert r0_eval.scored_positions == ref_ev.scored_positions
        assert r0_eval.batches == ref_ev.batches
        assert _scalar_rel(r0_eval.loss, ref_ev.loss) <= tol
        for r, out in enumerate(outs):
            expected_shard = len(validation_shard_indices(len(val_counts), world_size, r))
            assert out["val_windows_seen"] == expected_shard, (
                f"{key} r{r}: expected sharded val to process {expected_shard} windows, saw {out['val_windows_seen']}"
            )
            assert out["result"] == outs[0]["result"]
            sharded_loss = out["result"].eval_history[0][1]
            assert _scalar_rel(sharded_loss, ref_ev.loss) <= tol, (
                f"{key} r{r}: sharded val loss {sharded_loss} vs ref {ref_ev.loss}"
            )

    small_outs = unwrap_results(outcomes["val_small"])
    for r, out in enumerate(small_outs):
        expected_shard = len(validation_shard_indices(world_size - 1, world_size, r))
        assert out["val_windows_seen"] == expected_shard
        assert _scalar_rel(out["result"].eval_history[0][1], ref_small.loss) <= EVAL_TOL["float32"]

    one_zero_outs = unwrap_results(outcomes["val_one_zero"])
    for _r, out in enumerate(one_zero_outs):
        assert out["result"] == one_zero_outs[0]["result"]
        assert _scalar_rel(out["result"].eval_history[0][1], ref_one_zero.loss) <= EVAL_TOL["float32"]

    all_zero_outs = unwrap_results(outcomes["val_all_zero"])
    for r, out in enumerate(all_zero_outs):
        assert out["exc_type"] is ValueError, f"val_all_zero r{r}: expected ValueError, got {out['exc_type']}"
        assert "no scored position" in out["exc_msg"]
        assert out["followup"] == world_size

    # Training mode and LoRA dropout must be restored after step-1 internal validation for step-2 training (catches M4).
    rt_outs = unwrap_results(outcomes["val_restore_train"])
    for r, out in enumerate(rt_outs):
        assert out["result"] == rt_outs[0]["result"]
        assert torch.equal(out["params"], rt_outs[0]["params"])
        assert out["post_fit_training"] is False
        trace = out["trace"]
        assert len(trace) == 4, f"val_restore_train r{r}: expected 4 forwards (train, val, train, val), got {trace}"
        # Step 1 train, step 1 val, step 2 train (post-val), step 2 val.
        assert trace[0][0] is True and trace[0][1] is True and trace[0][2] > 0
        assert trace[1] == (False, False, 0)
        assert trace[2][0] is True and trace[2][1] is True and trace[2][2] > 0, (
            f"val_restore_train r{r}: model left in eval mode after internal validation: {trace[2]}"
        )
        assert trace[3] == (False, False, 0)


# --------------------------------------------------------------------------- T15 (h): guards & export


def _sc_t15h_guards_and_export(rank: int, world_size: int, base_dir: str) -> dict[str, Any]:
    ds = _make_uneven_dataset([8, 4, 6, 2] * world_size, seed=801, covariates=True)
    val_ds = _make_uneven_dataset([8, 4, 6, 2], seed=802, covariates=True)

    # 1. compile_mode="reduce-overhead" with compile_model=True -> ValueError; compile_model=False -> OK.
    reduce_overhead_err: type[BaseException] | None = None
    try:
        TimestralTrainer(
            _base_config(compile_model=True, compile_mode="reduce-overhead", batch_size=2),
            ds,
            model=_fresh_tiny_model(seed=13),
        )
    except Exception as exc:
        reduce_overhead_err = type(exc)
    # Positive control: compile_model=False ignores compile_mode="reduce-overhead".
    TimestralTrainer(
        _base_config(compile_model=False, compile_mode="reduce-overhead", batch_size=2),
        ds,
        model=_fresh_tiny_model(seed=13),
    )

    # 2. ReplayConfig: existing type/value checks run BEFORE Stage 1 NotImplementedError.
    replay_errs: dict[str, type[BaseException] | None] = {}
    for label, kwargs in (
        ("bad_type", {"replay_config": object()}),
        ("ds_without_cfg", {"replay_dataset": ds}),
        ("ds_is_val", {"replay_config": ReplayConfig(), "replay_dataset": val_ds}),
        ("valid_cfg", {"replay_config": ReplayConfig()}),
    ):
        err: type[BaseException] | None = None
        try:
            TimestralTrainer(_base_config(batch_size=2), ds, val_ds, model=_fresh_tiny_model(seed=13), **kwargs)
        except Exception as exc:
            err = type(exc)
        replay_errs[label] = err

    # 3. Device/backend mismatches on gloo raise ValueError immediately.
    dev_errs: dict[str, type[BaseException] | None] = {}
    for dev in ("cuda", "mps"):
        err = None
        try:
            TimestralTrainer(_base_config(device=dev, batch_size=2), ds, model=_fresh_tiny_model(seed=13))
        except Exception as exc:
            err = type(exc)
        dev_errs[dev] = err

    # 4. Effective batch > 32 with learning_rate=None emits logger.warning; explicit LR or N*B <= 32 does not.
    trainer_logger = logging.getLogger("timestral_tune.trainer")
    big_b = (32 // world_size) + 1  # 17 at N=2 (34 > 32), 9 at N=4 (36 > 32)
    warn_counts: dict[str, int] = {}
    for label, cfg in (
        ("default_lr_big", _base_config(batch_size=big_b, learning_rate=None)),
        ("explicit_lr_big", _base_config(batch_size=big_b, learning_rate=1e-3)),
        ("default_lr_small", _base_config(batch_size=2, learning_rate=None)),
    ):
        handler = _CaptureHandler()
        prev_level = trainer_logger.level
        trainer_logger.setLevel(logging.INFO)
        trainer_logger.addHandler(handler)
        try:
            TimestralTrainer(cfg, ds, model=_fresh_tiny_model(seed=13))
        finally:
            trainer_logger.removeHandler(handler)
            trainer_logger.setLevel(prev_level)
        warn_counts[label] = sum(1 for lvl, _ in handler.records if lvl == logging.WARNING)

    # 5. head_only skips frozen-feature cache (logged once at construction, body runs on every step).
    init_handler = _CaptureHandler()
    fit_handler = _CaptureHandler()
    prev_level = trainer_logger.level
    trainer_logger.setLevel(logging.INFO)
    trainer_logger.addHandler(init_handler)
    try:
        head_trainer = TimestralTrainer(
            _base_config(
                mode="head_only",
                batch_size=2,
                epochs=3,
                max_steps=None,
                learning_rate=1e-2,
                cache_frozen_features=True,
            ),
            ds,
            model=_fresh_tiny_model(seed=14),
        )
    finally:
        trainer_logger.removeHandler(init_handler)
        trainer_logger.setLevel(prev_level)

    body_calls = 0

    def _bump(*_a: Any) -> None:
        nonlocal body_calls
        body_calls += 1

    handle = head_trainer.model.transformer.register_forward_hook(_bump)
    trainer_logger.setLevel(logging.INFO)
    trainer_logger.addHandler(fit_handler)
    try:
        head_res = head_trainer.fit()
    finally:
        handle.remove()
        trainer_logger.removeHandler(fit_handler)
        trainer_logger.setLevel(prev_level)

    # Also verify `_get_head_cache` stays disabled for `evaluate()` / direct calls in multi-process mode.
    head_trainer.evaluate(val_ds)
    head_cache_disabled = (
        head_trainer._train_head_cache is None
        and head_trainer._val_head_cache is None
        and head_trainer._get_head_cache(ds) is None
    )

    cache_init_logs = [
        msg
        for lvl, msg in init_handler.records
        if lvl == logging.INFO and ("cach" in msg.lower() or "frozen" in msg.lower())
    ]
    cache_fit_logs = [
        msg
        for lvl, msg in fit_handler.records
        if lvl == logging.INFO and ("cach" in msg.lower() or "frozen" in msg.lower())
    ]

    # 6. Public training_step() raises RuntimeError in multi-process mode.
    sample_batch = ds.collator(head_trainer.model.config)([ds[0], ds[1]])
    training_step_err: type[BaseException] | None = None
    try:
        head_trainer.training_step(sample_batch)
    except Exception as exc:
        training_step_err = type(exc)

    # 7. export(): distinct path per rank proves non-zero ranks write nothing, log, return Path, and validate inputs.
    export_dir = Path(base_dir) / f"export_r{rank}"
    export_handler = _CaptureHandler()
    trainer_logger.setLevel(logging.INFO)
    trainer_logger.addHandler(export_handler)
    try:
        returned_path = head_trainer.export(export_dir)
    finally:
        trainer_logger.removeHandler(export_handler)
        trainer_logger.setLevel(prev_level)

    export_validation_errs: dict[str, type[BaseException] | None] = {}
    for label, call_kw in (
        ("no_output_dir", {"output_dir": None}),
        (
            "unfitted_calibrator",
            {"output_dir": Path(base_dir) / f"bad_cal_r{rank}", "calibrator": ConformalCalibrator(0.80)},
        ),
        (
            "missing_license",
            {"output_dir": Path(base_dir) / f"bad_lic_r{rank}", "license_path": Path(base_dir) / "nope"},
        ),
        ("nan_metrics", {"output_dir": Path(base_dir) / f"bad_met_r{rank}", "metrics": {"loss": float("nan")}}),
    ):
        err = None
        try:
            head_trainer.export(**call_kw)
        except Exception as exc:
            err = type(exc)
        export_validation_errs[label] = err

    return {
        "reduce_overhead_err": reduce_overhead_err,
        "replay_errs": replay_errs,
        "dev_errs": dev_errs,
        "warn_counts": warn_counts,
        "head_cache_is_none": head_cache_disabled,
        "head_body_calls": body_calls,
        "head_steps": head_res.steps,
        "cache_init_logs": len(cache_init_logs),
        "cache_fit_logs": len(cache_fit_logs),
        "training_step_err": training_step_err,
        "returned_path": str(returned_path),
        "export_dir": str(export_dir),
        "export_dir_exists": export_dir.exists(),
        "export_info_logs": len([msg for lvl, msg in export_handler.records if lvl == logging.INFO]),
        "export_validation_errs": export_validation_errs,
    }


@pytest.mark.acceptance
@pytest.mark.parametrize("world_size", [2, 4])
def test_t15_h_guards_and_export(world_size: int, tmp_path: Path) -> None:
    """T15(h). Stage 1 guards (`reduce-overhead`, `ReplayConfig`, device mismatch, default-LR warning,
    `head_only` cache bypass, `training_step()` `RuntimeError`) and collective-free `export()` with
    rank-0 metadata keys and non-zero-rank validation/no-write semantics.
    """
    # Positive control: one-process export must NOT contain `world_size` or `effective_batch_size` (spec §7.4).
    one_ds = _make_uneven_dataset([8, 4, 6, 2], seed=801, covariates=True)
    one_trainer = TimestralTrainer(_base_config(batch_size=2, max_steps=1), one_ds, model=_fresh_tiny_model(seed=14))
    one_out = one_trainer.export(tmp_path / "one_process_export")
    one_meta = json.loads((one_out / "tune_metadata.json").read_text(encoding="utf-8"))
    assert "world_size" not in one_meta and "effective_batch_size" not in one_meta

    outcomes = run_scenarios(
        {"guards": (_sc_t15h_guards_and_export, (str(tmp_path),))},
        world_size=world_size,
        pg_timeout=PG_TIMEOUT,
        wall_timeout=WALL_TIMEOUT,
    )
    outs = unwrap_results(outcomes["guards"])

    # Rank 0 logs warnings & cache bypass once.
    assert outs[0]["warn_counts"]["default_lr_big"] >= 1
    assert outs[0]["warn_counts"]["explicit_lr_big"] == 0
    assert outs[0]["warn_counts"]["default_lr_small"] == 0
    assert outs[0]["cache_init_logs"] == 1 and outs[0]["cache_fit_logs"] == 0

    expected_head_steps = 3 * steps_per_epoch(4 * world_size, world_size, 2)
    for r, out in enumerate(outs):
        assert out["reduce_overhead_err"] is ValueError
        assert out["replay_errs"] == {
            "bad_type": TypeError,
            "ds_without_cfg": ValueError,
            "ds_is_val": ValueError,
            "valid_cfg": NotImplementedError,
        }
        assert out["dev_errs"] == {"cuda": ValueError, "mps": ValueError}
        assert out["head_cache_is_none"] is True
        assert out["head_steps"] == expected_head_steps
        assert out["head_body_calls"] == expected_head_steps, (
            f"r{r}: head_only body ran {out['head_body_calls']} times, expected streaming {expected_head_steps}"
        )
        assert out["training_step_err"] is RuntimeError
        assert out["returned_path"] == out["export_dir"]
        assert out["export_validation_errs"] == {
            "no_output_dir": ValueError,
            "unfitted_calibrator": ValueError,
            "missing_license": FileNotFoundError,
            "nan_metrics": ValueError,
        }
        if r == 0:
            assert out["export_dir_exists"] is True
            meta = json.loads((Path(out["export_dir"]) / "tune_metadata.json").read_text(encoding="utf-8"))
            assert meta["world_size"] == world_size
            assert meta["effective_batch_size"] == world_size * 2
            assert meta["hyperparameters"]["batch_size"] == 2
        else:
            assert out["export_dir_exists"] is False, f"non-zero rank {r} wrote export directory"
            assert out["export_info_logs"] >= 1, f"non-zero rank {r} did not log that rank 0 writes the export"


def _worker_t15h_cpu_gloo_backend(rank: int, world_size: int) -> type[BaseException] | None:
    del rank, world_size
    ds = _make_uneven_dataset([8, 4, 6, 2], seed=803, covariates=True)
    try:
        TimestralTrainer(_base_config(batch_size=2), ds, model=_fresh_tiny_model(seed=15))
    except Exception as exc:
        return type(exc)
    return None


def _worker_t15h_post_destroy_snapshot(rank: int, world_size: int, base_dir: str) -> dict[str, Any]:
    del world_size
    ds = _make_uneven_dataset([8, 4, 6, 2], seed=804, covariates=True)
    trainer = TimestralTrainer(_base_config(batch_size=2, max_steps=1), ds, model=_fresh_tiny_model(seed=16))
    dist.barrier()
    dist.destroy_process_group()

    trainer.should_stop = True
    prev_scheduler = trainer.scheduler
    fit_err: type[BaseException] | None = None
    try:
        trainer.fit()
    except Exception as exc:
        fit_err = type(exc)
    state_untouched = trainer.should_stop is True and trainer.scheduler is prev_scheduler and trainer.global_step == 0

    export_dir = Path(base_dir) / f"after_destroy_r{rank}"
    returned = trainer.export(export_dir)
    return {
        "fit_err": fit_err,
        "state_untouched": state_untouched,
        "returned": str(returned),
        "export_dir": str(export_dir),
        "exists": export_dir.exists(),
    }


@pytest.mark.acceptance
def test_t15_h_unsupported_backend_and_post_destroy_snapshot(tmp_path: Path) -> None:
    """T15(h) & §5.6 snapshot semantics. Device-qualified `"cpu:gloo"` raises `ValueError` at
    construction; after `destroy_process_group()`, `fit()` raises `RuntimeError` without mutating
    trainer state while `export()` still uses the construction snapshot cleanly.
    """
    bad_backend_outs = unwrap_results(
        run_distributed(
            _worker_t15h_cpu_gloo_backend,
            world_size=2,
            backend="cpu:gloo",
            pg_timeout=PG_TIMEOUT,
            wall_timeout=WALL_TIMEOUT,
        )
    )
    assert bad_backend_outs == [ValueError, ValueError]

    snap_outs = unwrap_results(
        run_distributed(
            _worker_t15h_post_destroy_snapshot,
            world_size=2,
            args=(str(tmp_path),),
            pg_timeout=PG_TIMEOUT,
            wall_timeout=WALL_TIMEOUT,
        )
    )
    for r, out in enumerate(snap_outs):
        assert out["fit_err"] is RuntimeError, f"r{r}: fit() after destroy expected RuntimeError, got {out['fit_err']}"
        assert out["state_untouched"] is True, f"r{r}: stale-group fit() mutated trainer state before failing"
        assert out["returned"] == out["export_dir"]
    assert snap_outs[0]["exists"] is True
    meta = json.loads((Path(snap_outs[0]["export_dir"]) / "tune_metadata.json").read_text(encoding="utf-8"))
    assert meta["world_size"] == 2 and meta["effective_batch_size"] == 4
    assert snap_outs[1]["exists"] is False


def _worker_t15h_compile_default(rank: int, world_size: int, base_dir: str) -> dict[str, Any]:
    del world_size
    ds = _make_uneven_dataset([8, 4, 6, 2, 7, 3, 5, 1], seed=805, covariates=True)
    cfg = _base_config(
        mode="lora_variate",
        batch_size=2,
        max_steps=3,
        learning_rate=2e-2,
        lr_schedule="constant",
        compile_model=True,
        compile_mode="default",
        seed=81,
    )
    trainer = TimestralTrainer(cfg, ds, lora_config=_lora_for("lora_variate"), model=_fresh_tiny_model(seed=17))
    res = trainer.fit()
    state = merged_state_dict(trainer.model)
    has_orig_mod = any("_orig_mod" in k for k in state)
    reloaded = TimestralModel(trainer.model.config)
    reloaded.load_state_dict(state, strict=True)
    export_dir = Path(base_dir) / f"compile_export_r{rank}"
    returned = trainer.export(export_dir)
    return {
        "result": res,
        "params": _flat_params(trainer.model),
        "has_orig_mod": has_orig_mod,
        "returned": str(returned),
        "export_dir": str(export_dir),
        "exists": export_dir.exists(),
    }


@pytest.mark.acceptance
def test_t15_h_compile_default_smoke(tmp_path: Path) -> None:
    """T15(h). `compile_model=True` with `compile_mode="default"` trains cleanly across ranks on CPU,
    produces bitwise identical weights, merges without `_orig_mod` keys, and exports on rank 0 only.
    """
    if shutil.which("g++") is None and shutil.which("clang++") is None:
        pytest.skip("CPU torch.compile requires a C++ compiler (g++ or clang++)")
    outs = unwrap_results(
        run_distributed(
            _worker_t15h_compile_default,
            world_size=2,
            args=(str(tmp_path),),
            pg_timeout=60.0,
            wall_timeout=180.0,
        )
    )
    for r, out in enumerate(outs):
        assert out["result"].steps == 3
        assert out["result"] == outs[0]["result"]
        assert torch.equal(out["params"], outs[0]["params"]), f"compile_default r{r}: cross-rank params differ"
        assert out["has_orig_mod"] is False
        assert out["returned"] == out["export_dir"]
    assert outs[0]["exists"] is True
    meta = json.loads((Path(outs[0]["export_dir"]) / "tune_metadata.json").read_text(encoding="utf-8"))
    assert meta["world_size"] == 2 and meta["effective_batch_size"] == 4
    assert outs[1]["exists"] is False


# --------------------------------------------------------------------------- T15 (i): single-rank failure propagation


def _sc_t15i_init_fail_rank0(rank: int, world_size: int) -> dict[str, Any]:
    ds = _make_uneven_dataset([8, 4, 6, 2], seed=901, covariates=True)
    cbs: list[Any] = [object()] if rank == 0 else []
    return _record_propagated_exception(
        lambda: TimestralTrainer(_base_config(batch_size=2), ds, callbacks=cbs, model=_fresh_tiny_model(seed=17)),
        world_size,
    )


def _sc_t15i_init_fail_nonzero(rank: int, world_size: int) -> dict[str, Any]:
    ds = _make_uneven_dataset([8, 4, 6, 2], seed=902, covariates=True)
    lcfg = LoraConfig() if rank == 1 else None
    return _record_propagated_exception(
        lambda: TimestralTrainer(
            _base_config(mode="head_only", batch_size=2), ds, lora_config=lcfg, model=_fresh_tiny_model(seed=17)
        ),
        world_size,
    )


def _sc_t15i_init_fail_two_ranks(rank: int, world_size: int) -> dict[str, Any]:
    ds = _make_uneven_dataset([8, 4, 6, 2], seed=903, covariates=True)
    failing = (0, 1) if world_size == 2 else (1, 3)
    cbs: list[Any] = [object()] if rank == failing[0] else []
    lcfg = LoraConfig() if rank == failing[1] else None
    return _record_propagated_exception(
        lambda: TimestralTrainer(
            _base_config(mode="head_only", batch_size=2),
            ds,
            lora_config=lcfg,
            callbacks=cbs,
            model=_fresh_tiny_model(seed=17),
        ),
        world_size,
    )


def _sc_t15i_fit_fail_rank(rank: int, world_size: int, failing_ranks: tuple[int, ...]) -> dict[str, Any]:
    ds = _make_uneven_dataset([8, 4] * world_size, seed=904, covariates=True)
    trainer = TimestralTrainer(
        _base_config(mode="full_ft", batch_size=2, max_steps=1), ds, model=_fresh_tiny_model(seed=18)
    )
    if rank in failing_ranks:
        # Poison the first window of this rank's step-0 contiguous slice so DHFDataCollator raises ValueError in fit().
        poison_idx = rank * 2
        trainer.train_dataset.series[poison_idx].targets[0, -1] = np.inf
    return _record_propagated_exception(trainer.fit, world_size)


def _sc_t15i_fit_fail_callback(rank: int, world_size: int, hook: str) -> dict[str, Any]:
    del rank
    ds = _make_uneven_dataset([8, 4] * world_size, seed=905, covariates=True)
    val_ds = _make_uneven_dataset([8, 4] * world_size, seed=908, covariates=True) if hook == "on_evaluate" else None
    cbs: list[Callback] = [_BoomCallback(hook)]
    if hook == "on_evaluate":
        cbs.insert(0, _RecorderCallback(stop_on_start=True))
    # Pass the same BoomCallback on every rank: callbacks run on rank 0 only, so peers get RankFailureError.
    trainer = TimestralTrainer(
        _base_config(mode="head_only", batch_size=2, max_steps=2),
        ds,
        val_ds,
        callbacks=cbs,
        model=_fresh_tiny_model(seed=19),
    )
    return _record_propagated_exception(trainer.fit, world_size)


def _sc_t15i_fit_fail_val_shard(rank: int, world_size: int) -> dict[str, Any]:
    train_ds = _make_uneven_dataset([8, 4] * world_size, seed=906, covariates=True)
    val_ds = _make_uneven_dataset([8, 4] * world_size, seed=907, covariates=True)
    trainer = TimestralTrainer(
        _base_config(mode="head_only", batch_size=2, eval_batch_size=2, max_steps=1, eval_every=1),
        train_ds,
        val_ds,
        model=_fresh_tiny_model(seed=20),
    )
    if rank == 1:
        # Index 1 belongs to rank 1's stride validation shard (`r, r + N, ...`).
        assert trainer.val_dataset is not None
        trainer.val_dataset.series[1].targets[0, -1] = np.inf
    return _record_propagated_exception(trainer.fit, world_size)


def _sc_t15i_fit_fail_over_non_finite(rank: int, world_size: int) -> dict[str, Any]:
    """Failure check must take precedence over non-finite check when both occur in the same step (catches M5)."""
    ds = _make_uneven_dataset([8, 4] * world_size, seed=910, covariates=True)
    trainer = TimestralTrainer(
        _base_config(mode="full_ft", batch_size=2, max_steps=1), ds, model=_fresh_tiny_model(seed=18)
    )
    handle = None
    if rank == 0:
        handle = trainer.model.head.register_forward_hook(lambda _mod, _inp, out: out * float("nan"))
    elif rank == 1:
        trainer.train_dataset.series[2].targets[0, -1] = np.inf
    try:
        return _record_propagated_exception(trainer.fit, world_size)
    finally:
        if handle is not None:
            handle.remove()


@pytest.mark.acceptance
@pytest.mark.parametrize("world_size", [2, 4])
def test_t15_i_single_rank_failure_propagation(world_size: int) -> None:
    """T15(i). Single-rank and two-rank failures in `__init__`, `fit()` local stage (including when
    another rank simultaneously reports a non-finite loss), rank-0 callbacks (`on_fit_start`,
    `on_step_end`, `on_evaluate`, `on_fit_end`), and sharded validation propagate to all ranks
    (`RankFailureError` naming failing ranks and stage on peers) and leave the process group clean
    for follow-up collectives.
    """
    two_ranks = (0, 1) if world_size == 2 else (1, 3)
    scenarios: dict[str, Any] = {
        "init_r0": _sc_t15i_init_fail_rank0,
        "init_r1": _sc_t15i_init_fail_nonzero,
        "init_two": _sc_t15i_init_fail_two_ranks,
        "fit_r1": (_sc_t15i_fit_fail_rank, ((1,),)),
        "fit_r0": (_sc_t15i_fit_fail_rank, ((0,),)),
        "fit_two": (_sc_t15i_fit_fail_rank, (two_ranks,)),
        "fit_fail_over_nf": _sc_t15i_fit_fail_over_non_finite,
        "fit_cb_start": (_sc_t15i_fit_fail_callback, ("on_fit_start",)),
        "fit_cb_step": (_sc_t15i_fit_fail_callback, ("on_step_end",)),
        "fit_cb_eval": (_sc_t15i_fit_fail_callback, ("on_evaluate",)),
        "fit_cb_end": (_sc_t15i_fit_fail_callback, ("on_fit_end",)),
        "fit_val": _sc_t15i_fit_fail_val_shard,
    }
    outcomes = run_scenarios(scenarios, world_size=world_size, pg_timeout=PG_TIMEOUT, wall_timeout=WALL_TIMEOUT)

    _assert_propagated(unwrap_results(outcomes["init_r0"]), world_size, {0: TypeError}, "init_r0")
    _assert_propagated(unwrap_results(outcomes["init_r1"]), world_size, {1: ValueError}, "init_r1")
    _assert_propagated(
        unwrap_results(outcomes["init_two"]),
        world_size,
        {two_ranks[0]: TypeError, two_ranks[1]: ValueError},
        "init_two",
    )
    _assert_propagated(unwrap_results(outcomes["fit_r1"]), world_size, {1: ValueError}, "fit_r1")
    _assert_propagated(unwrap_results(outcomes["fit_r0"]), world_size, {0: ValueError}, "fit_r0")
    _assert_propagated(
        unwrap_results(outcomes["fit_two"]),
        world_size,
        {two_ranks[0]: ValueError, two_ranks[1]: ValueError},
        "fit_two",
    )
    _assert_propagated(
        unwrap_results(outcomes["fit_fail_over_nf"]), world_size, {1: ValueError}, "fit_fail_over_non_finite"
    )
    _assert_propagated(unwrap_results(outcomes["fit_cb_start"]), world_size, {0: KeyError}, "fit_cb_start")
    _assert_propagated(unwrap_results(outcomes["fit_cb_step"]), world_size, {0: KeyError}, "fit_cb_step")
    _assert_propagated(unwrap_results(outcomes["fit_cb_eval"]), world_size, {0: KeyError}, "fit_cb_eval")
    _assert_propagated(unwrap_results(outcomes["fit_cb_end"]), world_size, {0: KeyError}, "fit_cb_end")
    _assert_propagated(unwrap_results(outcomes["fit_val"]), world_size, {1: ValueError}, "fit_val")

    # Spec §5.6: `RankFailureError` must be exported from `timestral_tune.trainer`.
    assert getattr(trainer_pkg, "RankFailureError", None) is RankFailureError, (
        "RankFailureError must be exported from timestral_tune.trainer (spec §5.6)"
    )


def _worker_t15i_startup_digest_failure(rank: int, world_size: int) -> dict[str, Any]:
    ds = _make_uneven_dataset([8, 4, 6, 2], seed=909, covariates=True)
    # Non-JSON-serialisable `model_name_or_path` with explicit `model=` bypasses `load_model()` in
    # step (a) and raises `TypeError` inside `_startup_consistency_digest()` on rank 1 only.
    bad_path: Any = object() if rank == 1 else "tiny-local-test"
    cfg = _base_config(model_name_or_path=bad_path, mode="head_only", batch_size=2, max_steps=1, seed=31)
    return _record_propagated_exception(
        lambda: TimestralTrainer(cfg, ds, model=_fresh_tiny_model(seed=17)),
        world_size,
    )


@pytest.mark.acceptance
@pytest.mark.parametrize("world_size", [2, 4])
def test_t15_i_startup_digest_failure_propagation(world_size: int) -> None:
    """T15(i) regression. A single-rank failure while computing the startup consistency digest
    must propagate (`TypeError` on the failing rank, `RankFailureError` naming it on every peer)
    and leave the process group clean for follow-up collectives instead of timing out peers.
    """
    outs = unwrap_results(
        run_distributed(
            _worker_t15i_startup_digest_failure,
            world_size=world_size,
            pg_timeout=3.0,
            wall_timeout=15.0,
        )
    )
    _assert_propagated(outs, world_size, {1: TypeError}, "init_digest_r1")
