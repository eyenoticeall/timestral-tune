"""White-box and complementary Stage 2 multi-process replay and distillation tests.

End-to-end acceptance clauses (trajectory parity across replay/distillation regimes, skipped-domain
steps, unscored/partial replay, repeated `fit()`, flips, teachers/caches, startup consistency,
failures, and replay export) live in `tests/acceptance/test_distributed.py` (T15(j)–(l)).

This module keeps only checks that are white-box or complementary to T15:
- White-box `_DistributedStrategy._run_step` parity on a warmed-up student with a non-patch-multiple
  replay horizon (`REP_HORIZON = 6`, `PATCH = 4`, exercising distillation mask padding), at strict
  `1e-12` (float64) and `1e-5` (float32) for loss and gradients (`1e-4` float32 for post-AdamW
  parameters).
- Complementary trajectory parity combining replay with periodic validation + `restore_best=True`,
  with `lora_temporal`, and with moved-student `full_ft` + distillation, all at strict `1e-10`
  (float64) and `1e-4` (float32).
- Direct white-box assertion that both flip generators and the replay cursor are untouched across a
  globally zero-scored skipped step.
- White-box replay loader generator seed check (`rank_seed(seed + 2_000_009, rank)`), `num_workers=0`
  enforcement, global RNG preservation, and stochastic `SyntheticReplaySource` `lora_temporal` run.
- Real multi-process `fit()` with `num_workers=1`, validation, and replay verifying training and
  validation run in worker processes while replay runs in the main process, matching `num_workers=0`.

- Spec: `docs/SPEC.md` §5.3, §5.6, §9 (T15), D22–D37
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import os
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from timestral_tune import LoraConfig, ReplayConfig, TimestralTrainer, TrainingConfig
from timestral_tune.data import ColumnRoles, SeriesData, TimeSeriesDataset, TrainingBatch
from timestral_tune.models import TuneTimestralModel, inject_lora, lora_parameters
from timestral_tune.replay import SyntheticReplaySource
from timestral_tune.trainer import compute_loss
from timestral_tune.trainer.distributed import rank_seed

from .distributed_harness import (
    STEP_PARAM_TOL,
    STEP_TOL,
    TRAJ_TOL,
    run_distributed,
    run_scenarios,
    unwrap_results,
)
from .helpers import as_double, float64_reference, l2_rel, make_tiny_forecaster

# PyTorch < 2.8 on macOS hangs at process exit after DataLoader worker processes run (docs/VERIFICATION.md §10).
WORKERS_HANG_AT_EXIT = sys.platform == "darwin" and tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)

PATCH = 4
CTX_STEPS = 4 * PATCH  # 16
DOM_HORIZON = 2 * PATCH  # 8
REP_HORIZON = 6  # deliberately not a multiple of PATCH (exercises distillation mask padding)
COV_ROWS = 3


# --------------------------------------------------------------------------- helpers


def _fresh_model(seed: int = 0, *, dtype: torch.dtype = torch.float32) -> TuneTimestralModel:
    model = copy.deepcopy(make_tiny_forecaster(seed=seed).model)
    if dtype == torch.float64:
        model = model.double()
    return model


def _flat_tensors(tensors: Sequence[Tensor]) -> Tensor:
    return torch.cat([t.detach().cpu().double().reshape(-1) for t in tensors])


def _flat_grads(params: Sequence[Tensor]) -> Tensor:
    return torch.cat(
        [(torch.zeros_like(p) if p.grad is None else p.grad).detach().cpu().double().reshape(-1) for p in params]
    )


def _base_config(**overrides: Any) -> TrainingConfig:
    defaults: dict[str, Any] = {
        "model_name_or_path": "local-test",
        "mode": "full_ft",
        "device": "cpu",
        "batch_size": 2,
        "epochs": 1,
        "max_steps": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "warmup_steps": 1,
        "lr_schedule": "cosine",
        "gradient_clip_val": 1.0,
        "shuffle": False,
        "sign_flip": False,
        "restore_best": False,
        "cache_frozen_features": False,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _make_single_window_dataset(
    num_windows: int,
    *,
    seed: int,
    covariates: bool,
    horizon: int,
    unequal_missing: bool = True,
) -> TimeSeriesDataset:
    """Build a dataset with exactly `num_windows` windows (1 window per series) and varied scored counts."""
    rng = np.random.default_rng(seed)
    total = CTX_STEPS + horizon
    ts = np.arange(total, dtype=np.int64)
    empty_f = np.empty((0, total), dtype=np.float32)
    empty_c = np.empty((0, total), dtype=np.float64)
    series: list[SeriesData] = []
    for i in range(num_windows):
        values = (15.0 + 3.0 * np.sin(0.4 * ts + 0.5 * i) + rng.normal(0.0, 0.8, size=total)).astype(np.float32)
        if unequal_missing:
            # Drop a varying tail of horizon targets while keeping >= 1 scored step per window.
            drop = i % max(1, horizon - 1)
            if drop > 0:
                values[total - drop :] = np.nan
        if covariates:
            past = rng.normal(0.0, 1.0, size=(1, total)).astype(np.float32)
            known = rng.normal(0.0, 1.0, size=(1, total)).astype(np.float32)
        else:
            past, known = empty_f, empty_f
        series.append(SeriesData(f"s{i}", ts, values[None, :], past, known, empty_c, empty_c))
    roles = ColumnRoles(targets=("y",), past=("p0",) if covariates else (), known=("k0",) if covariates else ())
    return TimeSeriesDataset(
        series,
        context_length=CTX_STEPS,
        forecast_horizon=horizon,
        stride=horizon,
        min_context=CTX_STEPS,
        roles=roles,
    )


def _make_sharded_batch_from_counts(
    spec: tuple[tuple[int, ...], ...],
    *,
    covariates: bool,
    horizon: int,
    seed: int,
    dtype: torch.dtype,
    model_config: Any,
) -> tuple[TrainingBatch, list[TrainingBatch | None]]:
    """Build a union `TrainingBatch` and per-rank disjoint shards matching exact per-series scored counts."""
    per_series = [k for counts in spec for k in counts]
    rps = COV_ROWS if covariates else 1
    rng = np.random.default_rng(seed)
    total = CTX_STEPS + horizon
    ts = np.arange(total, dtype=np.int64)
    empty_f = np.empty((0, total), dtype=np.float32)
    empty_c = np.empty((0, total), dtype=np.float64)
    series: list[SeriesData] = []
    for i in range(len(per_series)):
        vals = (12.0 + 2.5 * np.cos(0.35 * ts + 0.3 * i) + rng.normal(0.0, 0.7, size=total)).astype(np.float32)
        if covariates:
            past = rng.normal(0.0, 1.0, size=(1, total)).astype(np.float32)
            known = rng.normal(0.0, 1.0, size=(1, total)).astype(np.float32)
        else:
            past, known = empty_f, empty_f
        series.append(SeriesData(f"b{i}", ts, vals[None, :], past, known, empty_c, empty_c))
    roles = ColumnRoles(targets=("y",), past=("p0",) if covariates else (), known=("k0",) if covariates else ())
    ds = TimeSeriesDataset(
        series, context_length=CTX_STEPS, forecast_horizon=horizon, stride=horizon, min_context=CTX_STEPS, roles=roles
    )
    collator = ds.collator(model_config)
    base = collator([ds[i] for i in range(len(per_series))])

    keep = torch.zeros_like(base.loss_weight)
    for i, scored in enumerate(per_series):
        if scored > 0:
            keep[i * rps : (i + 1) * rps, :scored] = True
    weight = base.loss_weight & keep
    targets = torch.where(weight, base.targets, torch.zeros_like(base.targets))
    union = dataclasses.replace(base, targets=targets, loss_weight=weight)
    if dtype == torch.float64:
        union = as_double(union)

    shards: list[TrainingBatch | None] = []
    offset = 0
    for counts in spec:
        n = len(counts)
        if n == 0:
            shards.append(None)
        else:
            r0, r1 = offset * rps, (offset + n) * rps
            gids = torch.arange(n, dtype=torch.long).repeat_interleave(rps)
            shards.append(
                TrainingBatch(
                    context=union.context[r0:r1].clone(),
                    context_mask=union.context_mask[r0:r1].clone(),
                    group_ids=gids,
                    role_ids=None if union.role_ids is None else union.role_ids[r0:r1].clone(),
                    known_future=None if union.known_future is None else union.known_future[r0:r1].clone(),
                    known_future_mask=(
                        None if union.known_future_mask is None else union.known_future_mask[r0:r1].clone()
                    ),
                    targets=union.targets[r0:r1].clone(),
                    loss_weight=union.loss_weight[r0:r1].clone(),
                )
            )
        offset += n
    return union, shards


def _warmed_student_state(
    mode: str,
    dtype: torch.dtype,
    dom_ds: TimeSeriesDataset,
    *,
    model_seed: int = 0,
) -> dict[str, Tensor]:
    """Take 1 deterministic step from `_fresh_model(model_seed, dtype)` so student != base (and LoRA B != 0)."""
    model = _fresh_model(model_seed, dtype=dtype)
    if mode == "full_ft":
        trainable = list(model.parameters())
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        if mode == "head_only":
            trainable = list(model.head.parameters())
        else:
            gen = torch.Generator().manual_seed(0)
            inject_lora(model, LoraConfig(axes=("variate",), dropout=0.0), rows_per_series=COV_ROWS, generator=gen)
            trainable = list(lora_parameters(model))
    for p in trainable:
        p.requires_grad_(True)
    ctx = float64_reference() if dtype == torch.float64 else contextlib.nullcontext()
    with ctx:
        model.train()
        opt = torch.optim.AdamW(trainable, lr=2e-2 if mode != "full_ft" else 5e-3, weight_decay=0.0)
        batch = dom_ds.collator(model.config)([dom_ds[i] for i in range(len(dom_ds))])
        if dtype == torch.float64:
            batch = as_double(batch)
        opt.zero_grad(set_to_none=True)
        compute_loss(model, batch, validate=False).loss.backward()
        opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------- Scenario 1: White-box step parity


def _step_parity_regimes(
    world_size: int,
) -> list[tuple[str, float, float, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]]:
    """Mixed unequal, single-rank zero replay, distill-only, all-zero fallback, and empty-domain active-replay."""
    if world_size == 2:
        uneq_d: tuple[tuple[int, ...], ...] = ((8, 6, 4), (2,))
        eq_d: tuple[tuple[int, ...], ...] = ((8, 6), (8, 6))
        empty_d: tuple[tuple[int, ...], ...] = ((), (8, 5))
        return [
            ("a_mixed_unequal", 0.25, 0.5, uneq_d, ((6, 4, 6), (2,))),
            ("b_replay_zero_one_rank", 0.25, 0.5, ((8, 6), (4,)), ((6, 4), (0,))),
            ("c_distill_only", 0.0, 0.5, eq_d, ((6, 6, 6), (6,))),
            ("d_replay_all_zero_fallback", 0.25, 0.5, uneq_d, ((0, 0), (0,))),
            ("d2_replay_only_all_zero_fallback", 0.25, 0.0, uneq_d, ((0, 0), (0,))),
            ("e_empty_domain_active_replay", 0.25, 0.5, empty_d, ((6, 4), (5,))),
        ]
    uneq_d = ((8, 6, 4), (6, 4), (4,), (2,))
    eq_d = ((8, 6),) * 4
    empty_d = ((8, 6), (6,), (4,), ())
    return [
        ("a_mixed_unequal", 0.25, 0.5, uneq_d, ((6, 6, 4), (6, 2), (4,), (2,))),
        ("b_replay_zero_one_rank", 0.25, 0.5, ((8, 6), (6,), (4,), (2,)), ((6, 4), (4,), (2,), (0,))),
        ("c_distill_only", 0.0, 0.5, eq_d, ((6, 6, 4), (6, 6), (6,), ())),
        ("d_replay_all_zero_fallback", 0.25, 0.5, uneq_d, ((0, 0), (0,), (0,), (0,))),
        ("d2_replay_only_all_zero_fallback", 0.25, 0.0, uneq_d, ((0, 0), (0,), (0,), (0,))),
        ("e_empty_domain_active_replay", 0.25, 0.5, empty_d, ((6, 4), (4,), (2,), (6, 5))),
    ]


def _sc_step_level_parity(rank: int, world_size: int) -> dict[str, Any]:
    dom_ds = _make_single_window_dataset(4, seed=10, covariates=True, horizon=DOM_HORIZON)
    rep_ds = _make_single_window_dataset(4, seed=11, covariates=False, horizon=REP_HORIZON)
    regimes = _step_parity_regimes(world_size)
    modes = ("head_only", "lora_variate", "full_ft")
    out: dict[str, Any] = {}

    for dtype_name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        ctx_mgr = float64_reference() if dtype == torch.float64 else contextlib.nullcontext()
        with ctx_mgr:
            for mode in modes:
                student_state = _warmed_student_state(mode, dtype, dom_ds)
                lora_cfg = LoraConfig(axes=("variate",), dropout=0.0) if mode == "lora_variate" else None
                for label, rf, dw, dom_spec, rep_spec in regimes:
                    key = f"{dtype_name}:{mode}:{label}"
                    base_model = _fresh_model(0, dtype=dtype)
                    _, dom_s = _make_sharded_batch_from_counts(
                        dom_spec,
                        covariates=True,
                        horizon=DOM_HORIZON,
                        seed=200 + len(label),
                        dtype=dtype,
                        model_config=base_model.config,
                    )
                    _, rep_s = _make_sharded_batch_from_counts(
                        rep_spec,
                        covariates=False,
                        horizon=REP_HORIZON,
                        seed=300 + len(label),
                        dtype=dtype,
                        model_config=base_model.config,
                    )
                    cfg = _base_config(
                        mode=mode,
                        batch_size=2,
                        max_steps=1,
                        learning_rate=1e-3,
                        weight_decay=0.01,
                        warmup_steps=0,
                        lr_schedule="constant",
                        gradient_clip_val=1.0,
                    )
                    rcfg = ReplayConfig(
                        replay_fraction=rf,
                        distill_weight=dw,
                        batch_size=2,
                        cache_base_outputs=True,
                        smooth_beta=1e-3,
                    )
                    trainer = TimestralTrainer(
                        cfg,
                        dom_ds,
                        lora_config=lora_cfg,
                        replay_config=rcfg,
                        replay_dataset=rep_ds,
                        model=base_model,
                    )
                    # Load warmed-up student weights in place while keeping step-0 frozen teacher intact.
                    trainer.model.load_state_dict(student_state)
                    strategy = trainer._make_fit_strategy()
                    strategy.prepare_fit()
                    trainer.model.train()
                    try:
                        outcome = strategy._run_step(
                            lambda s=dom_s[rank]: s,
                            lambda r=rep_s[rank]: r,
                        )
                        out[key] = {
                            "skipped": outcome.skipped,
                            "loss": outcome.loss,
                            "grads": _flat_grads(trainer._trainable),
                            "params": _flat_tensors(trainer._trainable),
                        }
                    finally:
                        strategy.cleanup()
                        trainer.model.eval()
                    assert trainer._replay_base_pos is None and trainer._replay_base_neg is None
                    assert trainer._train_head_cache is None and trainer._replay_head_cache is None
    return out


# --------------------------------------------------------------------------- Scenario 2: Complementary trajectories


def _unique_trajectory_cases(world_size: int) -> list[dict[str, Any]]:
    dom_len = 2 * world_size * 2 + 1
    non_mult_rep = world_size * 2 + (world_size // 2)  # 5 at N=2, 10 at N=4
    short_rep = max(1, world_size - 2) if world_size == 4 else 1
    return [
        {
            "name": "both_non_mult_lora_temporal",
            "mode": "lora_temporal",
            "dom_len": dom_len,
            "rep_len": non_mult_rep,
            "rf": 0.25,
            "dw": 0.5,
            "steps": 5,
        },
        {
            "name": "both_non_mult_head_only_with_val",
            "mode": "head_only",
            "dom_len": dom_len,
            "rep_len": non_mult_rep,
            "rf": 0.25,
            "dw": 0.5,
            "steps": 5,
            "with_val": True,
        },
        {
            # Starting from a moved student avoids the step-0 identical-weight regime where
            # `_frozen_base` (`requires_grad=False`) and student (`requires_grad=True`) take
            # different `F.linear` paths on permuted float32 inputs (measured in
            # `scripts/probes/linear_requires_grad_dispatch.py`), holding strict 1e-4 / 1e-10.
            "name": "both_short_full_ft_moved",
            "mode": "full_ft",
            "dom_len": dom_len,
            "rep_len": short_rep,
            "rf": 0.25,
            "dw": 0.5,
            "steps": 5,
            "moved_student": True,
        },
    ]


def _run_trajectory_case(
    case: dict[str, Any],
    *,
    dtype_name: str,
    batch_size: int,
    replay_batch_size: int,
) -> dict[str, Any]:
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    dom_ds = _make_single_window_dataset(case["dom_len"], seed=40, covariates=True, horizon=DOM_HORIZON)
    rep_ds = _make_single_window_dataset(case["rep_len"], seed=41, covariates=False, horizon=REP_HORIZON)
    val_ds = (
        _make_single_window_dataset(6, seed=42, covariates=True, horizon=DOM_HORIZON) if case.get("with_val") else None
    )
    mode = case["mode"]
    lora_cfg = LoraConfig(axes=("time",), dropout=0.0) if mode == "lora_temporal" else None
    cfg = _base_config(
        mode=mode,
        batch_size=batch_size,
        max_steps=case["steps"],
        eval_every=2 if val_ds is not None else None,
        restore_best=val_ds is not None,
        learning_rate=2e-3,
        warmup_steps=2,
        lr_schedule="cosine",
        weight_decay=0.01,
        gradient_clip_val=1.0,
        seed=7,
    )
    rcfg = ReplayConfig(
        replay_fraction=case["rf"],
        distill_weight=case["dw"],
        batch_size=replay_batch_size,
        cache_base_outputs=False,
        smooth_beta=1e-3,
    )
    ctx_mgr = float64_reference() if dtype == torch.float64 else contextlib.nullcontext()
    with ctx_mgr:
        trainer = TimestralTrainer(
            cfg,
            dom_ds,
            val_ds,
            lora_config=lora_cfg,
            replay_config=rcfg,
            replay_dataset=rep_ds,
            model=_fresh_model(3, dtype=dtype),
        )
        if case.get("moved_student"):
            moved_state = _warmed_student_state(mode, dtype, dom_ds, model_seed=3)
            trainer.model.load_state_dict(moved_state)
        res = trainer.fit()
        params = _flat_tensors(trainer._trainable)
    return {"result": res, "params": params}


def _sc_unique_trajectory_parity(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    out: dict[str, Any] = {}
    for dtype_name in ("float64", "float32"):
        for case in _unique_trajectory_cases(world_size):
            out[f"{dtype_name}:{case['name']}"] = _run_trajectory_case(
                case, dtype_name=dtype_name, batch_size=2, replay_batch_size=2
            )
    return out


# --------------------------------------------------------------------------- Scenario 3: Skipped-step white-box check


class _ZeroScoredDataset(TimeSeriesDataset):
    """`TimeSeriesDataset` whose windows have all-NaN futures (zero scored)."""

    def __init__(self, base: TimeSeriesDataset) -> None:
        self.__dict__.update(base.__dict__)

    def __getitem__(self, index: int):
        win = super().__getitem__(index)
        return dataclasses.replace(win, target_future=np.full_like(win.target_future, np.nan))


def _sc_skipped_step_preserves_generators_and_cursor(rank: int, world_size: int) -> dict[str, Any]:
    per_rank_b = 2
    dom_ds = _ZeroScoredDataset(
        _make_single_window_dataset(world_size * per_rank_b, seed=55, covariates=True, horizon=DOM_HORIZON)
    )
    rep_ds = _make_single_window_dataset(world_size * per_rank_b * 2, seed=56, covariates=False, horizon=REP_HORIZON)
    cfg = _base_config(
        mode="head_only",
        batch_size=per_rank_b,
        max_steps=2,
        shuffle=False,
        sign_flip=True,
        seed=29,
    )
    rcfg = ReplayConfig(
        replay_fraction=0.25, distill_weight=0.2, batch_size=per_rank_b, sign_flip=True, cache_base_outputs=False
    )
    trainer = TimestralTrainer(cfg, dom_ds, replay_config=rcfg, replay_dataset=rep_ds, model=_fresh_model(5))
    strat = trainer._make_fit_strategy()
    strat.prepare_fit()
    trainer.model.train()
    try:
        assert strat._replay_cycle is not None
        assert strat._flip_generator is not None and strat._replay_flip_generator is not None
        collator = dom_ds.collator(trainer.model.config)
        zero_batch = collator([dom_ds[rank]])
        dom_flip_before = strat._flip_generator.get_state().clone()
        rep_flip_before = strat._replay_flip_generator.get_state().clone()
        outcome = strat._run_step(lambda: zero_batch, strat._replay_cycle.next_batch)
        return {
            "skipped": outcome.skipped,
            "replay_step": strat._replay_cycle.global_step,
            "dom_flip_unchanged": bool(torch.equal(strat._flip_generator.get_state(), dom_flip_before)),
            "rep_flip_unchanged": bool(torch.equal(strat._replay_flip_generator.get_state(), rep_flip_before)),
        }
    finally:
        strat.cleanup()
        trainer.model.eval()


# --------------------------------------------------------------------------- Scenario 4: Loader seed & stochastic synth


def _sc_loader_seed_and_stochastic_synth(rank: int, world_size: int) -> dict[str, Any]:
    dom_ds = _make_single_window_dataset(world_size * 2 + 1, seed=70, covariates=True, horizon=DOM_HORIZON)
    t_synth = TimestralTrainer(
        _base_config(mode="lora_temporal", batch_size=2, max_steps=3, shuffle=True, sign_flip=True, seed=19),
        dom_ds,
        lora_config=LoraConfig(axes=("time",), dropout=0.1),
        replay_config=ReplayConfig(replay_fraction=0.2, distill_weight=0.2, batch_size=2, sign_flip=True),
        model=_fresh_model(8),
    )
    assert isinstance(t_synth.replay_dataset, SyntheticReplaySource)

    strat_check = t_synth._make_fit_strategy()
    strat_check.prepare_fit()
    try:
        assert strat_check._replay_loader is not None and strat_check._replay_loader.generator is not None
        loader_seed_ok = strat_check._replay_loader.generator.initial_seed() == rank_seed(19 + 2_000_009, rank)
        loader_workers_ok = (
            strat_check._replay_loader.num_workers == 0 and strat_check._replay_loader.persistent_workers is False
        )
        assert strat_check._replay_cycle is not None
        rng_before = torch.get_rng_state()
        _ = strat_check._replay_cycle.next_batch()
        replay_rng_preserved = bool(torch.equal(torch.get_rng_state(), rng_before))
    finally:
        strat_check.cleanup()

    synth_res = t_synth.fit()
    return {
        "loader_seed_ok": loader_seed_ok,
        "loader_workers_ok": loader_workers_ok,
        "replay_rng_preserved": replay_rng_preserved,
        "steps": synth_res.steps,
        "train_history": synth_res.train_history,
        "params": _flat_tensors(t_synth._trainable),
    }


# --------------------------------------------------------------------------- Main multi-process test entry points


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_replay_whitebox_scenarios(world_size: int) -> None:
    scenarios = {
        "step_parity": _sc_step_level_parity,
        "trajectory_unique": _sc_unique_trajectory_parity,
        "skipped_step_whitebox": _sc_skipped_step_preserves_generators_and_cursor,
        "loader_seed_and_synth": _sc_loader_seed_and_stochastic_synth,
    }
    outcomes = run_scenarios(scenarios, world_size=world_size, pg_timeout=15.0, wall_timeout=120.0)

    # --- 1. Step-level parity vs 1-process shipped TimestralTrainer.training_step on union batches ---
    sp = unwrap_results(outcomes["step_parity"])
    dom_ds = _make_single_window_dataset(4, seed=10, covariates=True, horizon=DOM_HORIZON)
    rep_ds = _make_single_window_dataset(4, seed=11, covariates=False, horizon=REP_HORIZON)
    regimes = _step_parity_regimes(world_size)

    for dtype_name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        tol = STEP_TOL[dtype_name]
        param_tol = STEP_PARAM_TOL[dtype_name]
        ctx_mgr = float64_reference() if dtype == torch.float64 else contextlib.nullcontext()
        with ctx_mgr:
            for mode in ("head_only", "lora_variate", "full_ft"):
                student_state = _warmed_student_state(mode, dtype, dom_ds)
                lora_cfg = LoraConfig(axes=("variate",), dropout=0.0) if mode == "lora_variate" else None
                for label, rf, dw, dom_spec, rep_spec in regimes:
                    key = f"{dtype_name}:{mode}:{label}"
                    base_model = _fresh_model(0, dtype=dtype)
                    dom_u, _ = _make_sharded_batch_from_counts(
                        dom_spec,
                        covariates=True,
                        horizon=DOM_HORIZON,
                        seed=200 + len(label),
                        dtype=dtype,
                        model_config=base_model.config,
                    )
                    rep_u, _ = _make_sharded_batch_from_counts(
                        rep_spec,
                        covariates=False,
                        horizon=REP_HORIZON,
                        seed=300 + len(label),
                        dtype=dtype,
                        model_config=base_model.config,
                    )
                    cfg = _base_config(
                        mode=mode,
                        batch_size=2,
                        max_steps=1,
                        learning_rate=1e-3,
                        weight_decay=0.01,
                        warmup_steps=0,
                        lr_schedule="constant",
                        gradient_clip_val=1.0,
                    )
                    rcfg = ReplayConfig(
                        replay_fraction=rf, distill_weight=dw, batch_size=2, cache_base_outputs=False, smooth_beta=1e-3
                    )
                    ref_trainer = TimestralTrainer(
                        cfg, dom_ds, lora_config=lora_cfg, replay_config=rcfg, replay_dataset=rep_ds, model=base_model
                    )
                    ref_trainer.model.load_state_dict(student_state)
                    ref_trainer.model.train()
                    ref_loss = ref_trainer.training_step(dom_u, validate=False, replay_batch=rep_u)
                    ref_grads = _flat_grads(ref_trainer._trainable)
                    ref_params = _flat_tensors(ref_trainer._trainable)

                    r0 = sp[0][key]
                    assert r0["skipped"] is False
                    assert abs(r0["loss"] - ref_loss) <= tol * max(1.0, abs(ref_loss)), (
                        f"{key}: loss {r0['loss']} vs ref {ref_loss}"
                    )
                    assert l2_rel(r0["grads"], ref_grads) <= tol, (
                        f"{key}: grad rel L2 = {l2_rel(r0['grads'], ref_grads)}"
                    )
                    assert l2_rel(r0["params"], ref_params) <= param_tol, (
                        f"{key}: param rel L2 = {l2_rel(r0['params'], ref_params)}"
                    )
                    for r in range(1, world_size):
                        assert torch.equal(sp[r][key]["grads"], r0["grads"]), f"{key}: rank {r} grad mismatch"
                        assert torch.equal(sp[r][key]["params"], r0["params"]), f"{key}: rank {r} param mismatch"
                        assert sp[r][key]["loss"] == pytest.approx(r0["loss"], rel=0.0, abs=1e-15)

                    # Prove the empty-domain rank's replay slice actually contributed.
                    if label == "e_empty_domain_active_replay" and dtype_name == "float32" and mode == "full_ft":
                        empty_rank = 0 if world_size == 2 else 3
                        non_empty_rep_spec = tuple(
                            () if idx == empty_rank else counts for idx, counts in enumerate(rep_spec)
                        )
                        drop_rep_u, _ = _make_sharded_batch_from_counts(
                            non_empty_rep_spec,
                            covariates=False,
                            horizon=REP_HORIZON,
                            seed=300 + len(label),
                            dtype=dtype,
                            model_config=base_model.config,
                        )
                        drop_trainer = TimestralTrainer(
                            cfg,
                            dom_ds,
                            lora_config=lora_cfg,
                            replay_config=rcfg,
                            replay_dataset=rep_ds,
                            model=_fresh_model(0, dtype=dtype),
                        )
                        drop_trainer.model.load_state_dict(student_state)
                        drop_trainer.model.train()
                        drop_trainer.training_step(dom_u, validate=False, replay_batch=drop_rep_u)
                        drop_grads = _flat_grads(drop_trainer._trainable)
                        assert l2_rel(r0["grads"], drop_grads) > 1e-3, (
                            "empty-domain rank's replay contribution was not distinguishable from dropping it"
                        )

    # --- 2. Complementary trajectories (lora_temporal, periodic val+restore, moved-student full_ft) ---
    tp = unwrap_results(outcomes["trajectory_unique"])
    for dtype_name in ("float64", "float32"):
        tol = TRAJ_TOL[dtype_name]
        for case in _unique_trajectory_cases(world_size):
            key = f"{dtype_name}:{case['name']}"
            ref = _run_trajectory_case(
                case, dtype_name=dtype_name, batch_size=world_size * 2, replay_batch_size=world_size * 2
            )
            r0 = tp[0][key]
            assert r0["result"].steps == ref["result"].steps == case["steps"]
            assert len(r0["result"].train_history) == len(ref["result"].train_history)
            for (s_got, l_got), (s_ref, l_ref) in zip(
                r0["result"].train_history, ref["result"].train_history, strict=True
            ):
                assert s_got == s_ref
                assert abs(l_got - l_ref) <= tol * max(1.0, abs(l_ref)), f"{key} step {s_got}: {l_got} vs {l_ref}"
            if case.get("with_val"):
                assert len(r0["result"].eval_history) == len(ref["result"].eval_history) > 0
                assert r0["result"].best_step == ref["result"].best_step
                assert r0["result"].restored_best == ref["result"].restored_best
                for (s_got, l_got), (s_ref, l_ref) in zip(
                    r0["result"].eval_history, ref["result"].eval_history, strict=True
                ):
                    assert s_got == s_ref
                    assert abs(l_got - l_ref) <= tol * max(1.0, abs(l_ref)), f"{key} eval {s_got}: {l_got} vs {l_ref}"
            assert l2_rel(r0["params"], ref["params"]) <= tol, (
                f"{key}: param rel L2 {l2_rel(r0['params'], ref['params'])}"
            )
            for r in range(1, world_size):
                assert torch.equal(tp[r][key]["params"], r0["params"]), f"{key}: rank {r} param divergence"
                assert tp[r][key]["result"] == r0["result"], f"{key}: rank {r} TrainingResult divergence"

    # --- 3. Skipped step leaves both flip generators and replay cursor untouched ---
    sk = unwrap_results(outcomes["skipped_step_whitebox"])
    for r in range(world_size):
        assert sk[r] == {
            "skipped": True,
            "replay_step": 0,
            "dom_flip_unchanged": True,
            "rep_flip_unchanged": True,
        }

    # --- 4. Replay loader seed, num_workers=0, RNG isolation, and stochastic SyntheticReplaySource ---
    ls = unwrap_results(outcomes["loader_seed_and_synth"])
    for r in range(world_size):
        assert ls[r]["loader_seed_ok"] is True
        assert ls[r]["loader_workers_ok"] is True
        assert ls[r]["replay_rng_preserved"] is True
        assert ls[r]["steps"] == 3
        assert torch.equal(ls[r]["params"], ls[0]["params"])
        assert ls[r]["train_history"] == ls[0]["train_history"]


# --------------------------------------------------------------------------- Persistent DataLoader workers with replay


class _PidRecordingDataset(TimeSeriesDataset):
    """Picklable `TimeSeriesDataset` recording `os.getpid()` for each `__getitem__` call in-process."""

    def __init__(self, base: TimeSeriesDataset) -> None:
        self.__dict__.update(base.__dict__)
        self.pids: list[int] = []

    def __getitem__(self, index: int):
        self.pids.append(os.getpid())
        return super().__getitem__(index)


def _worker_fit_replay_persistent_workers(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    main_pid = os.getpid()
    dom_base = _make_single_window_dataset(2 * world_size * 2 + 1, seed=101, covariates=True, horizon=DOM_HORIZON)
    rep_base = _make_single_window_dataset(world_size * 2 + 1, seed=102, covariates=False, horizon=REP_HORIZON)
    val_base = _make_single_window_dataset(world_size * 2 + 1, seed=103, covariates=True, horizon=DOM_HORIZON)
    rcfg = ReplayConfig(replay_fraction=0.25, distill_weight=0.3, batch_size=2, sign_flip=True)

    outputs: dict[str, Any] = {"main_pid": main_pid}
    for workers in (0, 1):
        dom_ds = _PidRecordingDataset(dom_base)
        rep_ds = _PidRecordingDataset(rep_base)
        val_ds = _PidRecordingDataset(val_base)
        cfg = _base_config(
            mode="head_only",
            batch_size=2,
            max_steps=4,
            eval_every=2,
            restore_best=True,
            num_workers=workers,
            shuffle=True,
            sign_flip=True,
            seed=23,
        )
        trainer = TimestralTrainer(
            cfg, dom_ds, val_ds, replay_config=rcfg, replay_dataset=rep_ds, model=_fresh_model(12)
        )

        loader_info: dict[str, Any] = {}
        real_make = trainer._make_fit_strategy

        def _wrapped_make(info: dict[str, Any] = loader_info, make_fn: Any = real_make) -> Any:
            strat = make_fn()
            real_prepare = strat.prepare_fit
            real_eval = strat.evaluate_val

            def _wrapped_prepare() -> None:
                real_prepare()
                assert strat._train_loader is not None and strat._replay_loader is not None
                info["train_num_workers"] = strat._train_loader.num_workers
                info["train_persistent"] = strat._train_loader.persistent_workers
                info["replay_num_workers"] = strat._replay_loader.num_workers
                info["replay_persistent"] = strat._replay_loader.persistent_workers

            def _wrapped_eval() -> Any:
                res = real_eval()
                assert strat._sharded_val_loader is not None
                info["val_num_workers"] = strat._sharded_val_loader.num_workers
                info["val_persistent"] = strat._sharded_val_loader.persistent_workers
                return res

            strat.prepare_fit = _wrapped_prepare  # type: ignore[method-assign]
            strat.evaluate_val = _wrapped_eval  # type: ignore[method-assign]
            return strat

        trainer._make_fit_strategy = _wrapped_make  # type: ignore[method-assign]
        dom_ds.pids.clear()
        rep_ds.pids.clear()
        val_ds.pids.clear()
        res = trainer.fit()
        outputs[f"result_w{workers}"] = res
        outputs[f"params_w{workers}"] = _flat_tensors(trainer._trainable)
        outputs[f"training_after_w{workers}"] = trainer.model.training
        outputs[f"loaders_w{workers}"] = loader_info
        outputs[f"dom_main_pids_w{workers}"] = list(dom_ds.pids)
        outputs[f"val_main_pids_w{workers}"] = list(val_ds.pids)
        outputs[f"rep_main_pids_w{workers}"] = list(rep_ds.pids)
    return outputs


@pytest.mark.skipif(WORKERS_HANG_AT_EXIT, reason="PyTorch < 2.8 on macOS: DataLoader worker hangs at process exit")
def test_distributed_fit_with_replay_and_persistent_workers() -> None:
    """With `num_workers=1`, training and validation use workers while replay runs in the main process."""
    results = unwrap_results(
        run_distributed(_worker_fit_replay_persistent_workers, world_size=2, pg_timeout=20.0, wall_timeout=60.0)
    )
    for r in range(2):
        main_pid = results[r]["main_pid"]
        assert results[r]["training_after_w0"] is False
        assert results[r]["training_after_w1"] is False
        assert results[r]["result_w1"] == results[r]["result_w0"]
        assert torch.equal(results[r]["params_w1"], results[r]["params_w0"])
        assert results[r]["loaders_w1"] == {
            "train_num_workers": 1,
            "train_persistent": True,
            "val_num_workers": 1,
            "val_persistent": True,
            "replay_num_workers": 0,
            "replay_persistent": False,
        }
        # With num_workers=1, train/val __getitem__ ran in worker processes (so main-process lists stay empty),
        # while replay ran in the main process (matching one-process _replay_loader).
        assert results[r]["dom_main_pids_w1"] == []
        assert results[r]["val_main_pids_w1"] == []
        assert len(results[r]["rep_main_pids_w1"]) > 0
        assert set(results[r]["rep_main_pids_w1"]) == {main_pid}
        assert len(results[r]["dom_main_pids_w0"]) > 0 and set(results[r]["dom_main_pids_w0"]) == {main_pid}
        assert len(results[r]["val_main_pids_w0"]) > 0 and set(results[r]["val_main_pids_w0"]) == {main_pid}
        assert len(results[r]["rep_main_pids_w0"]) > 0 and set(results[r]["rep_main_pids_w0"]) == {main_pid}
    assert results[1]["result_w1"] == results[0]["result_w1"]
    assert torch.equal(results[1]["params_w1"], results[0]["params_w1"])
