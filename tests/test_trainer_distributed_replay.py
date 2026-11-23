"""Stage 2 multi-process replay and distillation tests.

Also temporarily hosts unit tests for `DistributedBatchCycle` (from `trainer/distributed.py`),
which can be moved into `tests/test_distributed.py` once parallel test edits merge.

- Spec: `docs/SPEC.md` §5.3, §5.6, §9 (T15), D22–D35
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import logging
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from timestral_tune import LoraConfig, ReplayConfig, TimestralTrainer, TrainingConfig
from timestral_tune.data import ColumnRoles, SeriesData, TimeSeriesDataset, TrainingBatch
from timestral_tune.models import TuneTimestralModel, inject_lora, lora_parameters
from timestral_tune.replay import SyntheticReplaySource
from timestral_tune.trainer import RankFailureError, compute_loss
from timestral_tune.trainer.distributed import (
    DistributedBatchCycle,
    DistributedBatchSampler,
    DistributedContext,
    rank_seed,
    rank_step_indices,
)

from .distributed_harness import run_scenarios, unwrap_results
from .helpers import as_double, float64_reference, l2_rel, make_tiny_forecaster

# PyTorch < 2.8 on macOS hangs at process exit after DataLoader worker processes run (docs/VERIFICATION.md §10).
WORKERS_HANG_AT_EXIT = sys.platform == "darwin" and tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)

PATCH = 4
CTX_STEPS = 4 * PATCH  # 16
DOM_HORIZON = 2 * PATCH  # 8
REP_HORIZON = 6  # deliberately different from DOM_HORIZON (non-multiple of PATCH)
COV_ROWS = 3


class _IndexDataset(Dataset[int]):
    """Minimal picklable dataset returning each sample's integer index."""

    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> int:
        return index


def _identity_collate(items: list[int]) -> list[int]:
    return list(items)


# --------------------------------------------------------------------------- DistributedBatchCycle unit tests
# (Located here due to parallel file ownership; candidate to move to tests/test_distributed.py.)


def test_distributed_batch_cycle_validation_and_docstrings() -> None:
    empty_sampler = DistributedBatchSampler(0, world_size=2, rank=0, batch_size=2)
    with pytest.raises(ValueError, match="non-empty"):
        DistributedBatchCycle(empty_sampler, [])
    with pytest.raises(TypeError, match="DistributedBatchSampler"):
        DistributedBatchCycle(object(), [])  # type: ignore[arg-type]

    assert DistributedBatchCycle.__doc__
    for attr_name in ("epoch", "step_in_epoch", "global_step", "next_batch"):
        attr = getattr(DistributedBatchCycle, attr_name)
        target = attr.fget if isinstance(attr, property) else attr
        assert target is not None and target.__doc__, f"DistributedBatchCycle.{attr_name} missing docstring"


@pytest.mark.parametrize(
    "num_items,world_size,batch_size",
    [
        (1, 2, 2),  # shorter than world_size: rank 1 empty on every step
        (2, 4, 2),  # shorter than world_size: ranks 2, 3 empty on every step
        (5, 2, 2),  # not a multiple of N*b: trailing partial step (rank 0 gets 1, rank 1 empty)
        (10, 4, 2),  # trailing step has 2 items across 4 ranks (ranks 0, 1 get 1; ranks 2, 3 empty)
        (11, 4, 2),  # trailing step has 3 items across 4 ranks
        (16, 4, 2),  # exact multiple of N*b
    ],
)
@pytest.mark.parametrize("shuffle", [False, True])
def test_distributed_batch_cycle_wraps_epochs_and_matches_union(
    num_items: int, world_size: int, batch_size: int, shuffle: bool
) -> None:
    dataset = _IndexDataset(num_items)
    cycles: list[DistributedBatchCycle[list[int]]] = []
    iter_counts = [0] * world_size

    class _CountingLoader:
        def __init__(self, rank_idx: int, inner: DataLoader[list[int]]) -> None:
            self._rank_idx = rank_idx
            self._inner = inner

        def __iter__(self):
            iter_counts[self._rank_idx] += 1
            return iter(self._inner)

    for rank in range(world_size):
        sampler = DistributedBatchSampler(
            num_items,
            world_size=world_size,
            rank=rank,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=2_000_009,
        )
        loader: DataLoader[list[int]] = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=_identity_collate,
            num_workers=0,
            generator=torch.Generator().manual_seed(100 + rank),
        )
        cycles.append(DistributedBatchCycle(sampler, _CountingLoader(rank, loader)))

    steps_per_ep = cycles[0]._sampler.num_global_steps
    total_steps = steps_per_ep * 3 + 1

    # Also build one-process cycling reference at batch_size = world_size * batch_size for shuffle=False.
    global_batch = world_size * batch_size
    one_proc_batches: list[list[int]] = [
        list(range(start, min(start + global_batch, num_items))) for start in range(0, num_items, global_batch)
    ]

    for gstep in range(total_steps):
        expected_epoch, expected_step = divmod(gstep, steps_per_ep)
        step_union: list[int] = []
        for rank, cycle in enumerate(cycles):
            assert cycle.epoch == expected_epoch
            assert cycle.step_in_epoch == expected_step
            assert cycle.global_step == gstep
            got = cycle.next_batch()
            expected_indices = rank_step_indices(
                num_items,
                world_size,
                batch_size,
                rank,
                expected_step,
                shuffle=shuffle,
                seed=2_000_009,
                epoch=expected_epoch,
            )
            if not expected_indices:
                assert got is None
            else:
                assert got == expected_indices
                step_union.extend(got)

        if not shuffle:
            assert step_union == one_proc_batches[expected_step]

    # Ranks that never had a non-empty slice in any epoch must never have invoked iter(loader).
    for rank in range(world_size):
        if rank >= num_items:
            assert iter_counts[rank] == 0
        else:
            assert iter_counts[rank] == 4  # 3 full epochs + 1 step into epoch 3


@pytest.mark.skipif(WORKERS_HANG_AT_EXIT, reason="PyTorch < 2.8 on macOS: DataLoader worker hangs at process exit")
def test_distributed_batch_cycle_with_persistent_workers_and_global_rng() -> None:
    dataset = _IndexDataset(10)
    sampler = DistributedBatchSampler(10, world_size=4, rank=0, batch_size=2, shuffle=True, seed=2_000_009)
    loader: DataLoader[list[int]] = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=_identity_collate,
        num_workers=1,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(42),
    )
    cycle = DistributedBatchCycle(sampler, loader)

    rng_before = torch.get_rng_state()
    seen = [cycle.next_batch() for _ in range(sampler.num_global_steps * 3)]
    assert torch.equal(torch.get_rng_state(), rng_before)

    expected = [
        rank_step_indices(10, 4, 2, 0, s, shuffle=True, seed=2_000_009, epoch=ep)
        for ep in range(3)
        for s in range(sampler.num_global_steps)
    ]
    assert seen == [idx if idx else None for idx in expected]


# --------------------------------------------------------------------------- multi-process helpers


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
) -> dict[str, Tensor]:
    """Take 1 deterministic step from `_fresh_model(0, dtype)` so student != base (and LoRA B != 0)."""
    model = _fresh_model(0, dtype=dtype)
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


# --------------------------------------------------------------------------- Scenario 1: Step-level parity


def _step_parity_regimes(
    world_size: int,
) -> list[tuple[str, float, float, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]]:
    """Probe section [4] regimes + empty-domain-with-active-replay + replay-only all-zero fallback."""
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
                    dom_u, dom_s = _make_sharded_batch_from_counts(
                        dom_spec,
                        covariates=True,
                        horizon=DOM_HORIZON,
                        seed=200 + len(label),
                        dtype=dtype,
                        model_config=base_model.config,
                    )
                    rep_u, rep_s = _make_sharded_batch_from_counts(
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
                        cache_base_outputs=True,  # must still be ignored in multi-process mode
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
                    # Confirm no caches were built.
                    assert trainer._replay_base_pos is None and trainer._replay_base_neg is None
                    assert trainer._train_head_cache is None and trainer._replay_head_cache is None
    return out


# --------------------------------------------------------------------------- Scenario 2: Trajectory parity via fit()


def _trajectory_cases(world_size: int) -> list[dict[str, Any]]:
    # Domain has 2*N*B + 1 windows (with B=2) so a trailing partial step exercises unequal/empty domain slices.
    dom_len = 2 * world_size * 2 + 1
    non_mult_rep = world_size * 2 + (world_size // 2)  # 5 at N=2 (N*b=4); 10 at N=4 (N*b=8)
    short_rep = max(1, world_size - 2) if world_size == 4 else 1  # 1 at N=2; 2 at N=4 (< N, empty trailing ranks)
    return [
        {
            "name": "replay_only_non_mult_full_ft",
            "mode": "full_ft",
            "dom_len": dom_len,
            "rep_len": non_mult_rep,
            "rf": 0.25,
            "dw": 0.0,
            "steps": 5,
        },
        {
            "name": "distill_only_short_head_only",
            "mode": "head_only",
            "dom_len": dom_len,
            "rep_len": short_rep,
            "rf": 0.0,
            "dw": 0.5,
            "steps": 5,
        },
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
            "name": "both_short_full_ft",
            "mode": "full_ft",
            "dom_len": dom_len,
            "rep_len": short_rep,
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
    ]


def _run_trajectory_case(case: dict[str, Any], *, batch_size: int, replay_batch_size: int) -> dict[str, Any]:
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
    trainer = TimestralTrainer(
        cfg,
        dom_ds,
        val_ds,
        lora_config=lora_cfg,
        replay_config=rcfg,
        replay_dataset=rep_ds,
        model=_fresh_model(3),
    )
    res = trainer.fit()
    return {
        "result": res,
        "params": _flat_tensors(trainer._trainable),
    }


def _sc_trajectory_parity(rank: int, world_size: int) -> dict[str, Any]:
    del rank
    return {
        case["name"]: _run_trajectory_case(case, batch_size=2, replay_batch_size=2)
        for case in _trajectory_cases(world_size)
    }


# --------------------------------------------------------------------------- Scenario 3: Globally skipped domain step


class _ZeroScoredMiddleDataset(TimeSeriesDataset):
    """`TimeSeriesDataset` whose windows in `[zero_start, zero_end)` have all-NaN futures (zero scored)."""

    def __init__(self, base: TimeSeriesDataset, zero_start: int, zero_end: int) -> None:
        self.__dict__.update(base.__dict__)
        self._zero_start = zero_start
        self._zero_end = zero_end

    def __getitem__(self, index: int):
        win = super().__getitem__(index)
        if self._zero_start <= index < self._zero_end:
            return dataclasses.replace(win, target_future=np.full_like(win.target_future, np.nan))
        return win


def _sc_skipped_domain_step_preserves_replay(rank: int, world_size: int) -> dict[str, Any]:
    per_rank_b = 2
    global_b = world_size * per_rank_b
    base_dom = _make_single_window_dataset(3 * global_b, seed=55, covariates=True, horizon=DOM_HORIZON)
    dom_with_skip = _ZeroScoredMiddleDataset(base_dom, global_b, 2 * global_b)
    # Replay dataset with distinct windows so advancing an extra replay step would visibly change loss & weights.
    rep_ds = _make_single_window_dataset(global_b * 3, seed=56, covariates=False, horizon=REP_HORIZON)

    cfg = _base_config(
        mode="full_ft",
        batch_size=per_rank_b,
        max_steps=4,
        learning_rate=2e-3,
        warmup_steps=1,
        lr_schedule="cosine",
        weight_decay=0.01,
        gradient_clip_val=1.0,
        seed=13,
    )
    rcfg = ReplayConfig(replay_fraction=0.3, distill_weight=0.4, batch_size=per_rank_b, cache_base_outputs=False)
    trainer = TimestralTrainer(cfg, dom_with_skip, replay_config=rcfg, replay_dataset=rep_ds, model=_fresh_model(5))

    # Spy on replay cursor advancement inside fit() by wrapping _make_fit_strategy.
    real_make = trainer._make_fit_strategy
    recorded_replay_steps: list[tuple[int, int]] = []  # (trainer.global_step before step, replay global_step)

    def _wrapped_make():
        strat = real_make()
        real_prepare = strat.prepare_fit

        def _wrapped_prepare() -> None:
            real_prepare()
            assert strat._replay_cycle is not None
            real_next = strat._replay_cycle.next_batch

            def _spied_next():
                recorded_replay_steps.append((trainer.global_step, strat._replay_cycle.global_step))
                return real_next()

            strat._replay_cycle.next_batch = _spied_next  # type: ignore[method-assign]

        strat.prepare_fit = _wrapped_prepare  # type: ignore[method-assign]
        return strat

    trainer._make_fit_strategy = _wrapped_make  # type: ignore[method-assign]
    res = trainer.fit()

    # Also verify stochastic skipped step leaves replay cursor and both flip generators untouched.
    stoch_cfg = _base_config(
        mode="head_only",
        batch_size=per_rank_b,
        max_steps=2,
        shuffle=False,
        sign_flip=True,
        seed=29,
    )
    stoch_rcfg = ReplayConfig(
        replay_fraction=0.25, distill_weight=0.2, batch_size=per_rank_b, sign_flip=True, cache_base_outputs=False
    )
    stoch_trainer = TimestralTrainer(
        stoch_cfg, dom_with_skip, replay_config=stoch_rcfg, replay_dataset=rep_ds, model=_fresh_model(5)
    )
    strat = stoch_trainer._make_fit_strategy()
    strat.prepare_fit()
    stoch_trainer.model.train()
    try:
        assert strat._replay_cycle is not None
        assert strat._flip_generator is not None and strat._replay_flip_generator is not None
        collator = dom_with_skip.collator(stoch_trainer.model.config)
        zero_batch = collator([dom_with_skip[global_b + rank]])
        dom_flip_before = strat._flip_generator.get_state().clone()
        rep_flip_before = strat._replay_flip_generator.get_state().clone()
        outcome = strat._run_step(lambda: zero_batch, strat._replay_cycle.next_batch)
        stoch_skip_ok = (
            outcome.skipped
            and strat._replay_cycle.global_step == 0
            and torch.equal(strat._flip_generator.get_state(), dom_flip_before)
            and torch.equal(strat._replay_flip_generator.get_state(), rep_flip_before)
        )
    finally:
        strat.cleanup()
        stoch_trainer.model.eval()

    return {
        "result": res,
        "params": _flat_tensors(trainer._trainable),
        "recorded_replay_steps": recorded_replay_steps,
        "stoch_skip_ok": stoch_skip_ok,
    }


# --------------------------------------------------------------------------- Scenario 4: Replay failures & non-finite


class _FailingOnRankDataset(TimeSeriesDataset):
    """Raises `RuntimeError` when indexed on `fail_rank`."""

    def __init__(self, base: TimeSeriesDataset, *, should_fail: bool) -> None:
        self.__dict__.update(base.__dict__)
        self._should_fail = should_fail

    def __getitem__(self, index: int):
        if self._should_fail:
            raise RuntimeError("replay_fetch_boom")
        return super().__getitem__(index)


def _sc_replay_failures(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect()
    assert ctx is not None
    dom_ds = _make_single_window_dataset(world_size * 2, seed=60, covariates=True, horizon=DOM_HORIZON)
    rep_ds = _make_single_window_dataset(world_size * 2, seed=61, covariates=False, horizon=REP_HORIZON)
    cfg = _base_config(mode="full_ft", batch_size=2, max_steps=2)
    rcfg = ReplayConfig(replay_fraction=0.25, distill_weight=0.5, batch_size=2, cache_base_outputs=False)
    out: dict[str, Any] = {}

    # 1. Rank 1 fails inside _prepare_replay_batch.
    t1 = TimestralTrainer(cfg, dom_ds, replay_config=rcfg, replay_dataset=rep_ds, model=_fresh_model(6))
    if rank == 1:

        def _boom_prep(_item: Any, _gen: Any) -> Any:
            raise KeyError("replay_prep_boom")

        t1._prepare_replay_batch = _boom_prep  # type: ignore[method-assign]
    try:
        t1.fit()
        out["prep_fail"] = {"raised": False}
    except Exception as exc:
        out["prep_fail"] = {
            "raised": True,
            "exc_type": type(exc),
            "failing_ranks": getattr(exc, "failing_ranks", None),
            "stage": getattr(exc, "stage", None),
            "model_training": t1.model.training,
            "followup": _followup_sum(ctx),
        }

    # 2. Rank 1 fails inside replay dataset __getitem__ (fetch_replay inside Stage 1).
    # Keep identical dataset fingerprints at startup, then flip _should_fail on rank 1 before fit().
    failing_rep_ds = _FailingOnRankDataset(rep_ds, should_fail=False)
    t2 = TimestralTrainer(cfg, dom_ds, replay_config=rcfg, replay_dataset=failing_rep_ds, model=_fresh_model(6))
    if rank == 1:
        failing_rep_ds._should_fail = True
    try:
        t2.fit()
        out["fetch_fail"] = {"raised": False}
    except Exception as exc:
        out["fetch_fail"] = {
            "raised": True,
            "exc_type": type(exc),
            "failing_ranks": getattr(exc, "failing_ranks", None),
            "stage": getattr(exc, "stage", None),
            "model_training": t2.model.training,
            "followup": _followup_sum(ctx),
        }

    # 3. Rank 1 fails inside Phase 0 domain fetch (with replay configured) -> propagates without advancing replay.
    failing_dom_ds = _FailingOnRankDataset(dom_ds, should_fail=False)
    t3 = TimestralTrainer(cfg, failing_dom_ds, replay_config=rcfg, replay_dataset=rep_ds, model=_fresh_model(6))
    if rank == 1:
        failing_dom_ds._should_fail = True
    try:
        t3.fit()
        out["pre_fetch_fail"] = {"raised": False}
    except Exception as exc:
        out["pre_fetch_fail"] = {
            "raised": True,
            "exc_type": type(exc),
            "failing_ranks": getattr(exc, "failing_ranks", None),
            "stage": getattr(exc, "stage", None),
            "model_training": t3.model.training,
            "followup": _followup_sum(ctx),
        }

    # 4. Single-rank non-finite replay loss raises FloatingPointError on all ranks.
    t4 = TimestralTrainer(cfg, dom_ds, replay_config=rcfg, replay_dataset=rep_ds, model=_fresh_model(6))
    real_prep = t4._prepare_replay_batch
    if rank == 1:

        def _nan_replay(item: Any, gen: Any):
            prepared = real_prep(item, gen)
            bad_targets = prepared.batch.targets.clone()
            bad_targets[0, 0] = float("nan")
            return dataclasses.replace(prepared, batch=dataclasses.replace(prepared.batch, targets=bad_targets))

        t4._prepare_replay_batch = _nan_replay  # type: ignore[method-assign]
    try:
        t4.fit()
        out["non_finite_replay"] = {"raised": False}
    except Exception as exc:
        out["non_finite_replay"] = {
            "raised": True,
            "exc_type": type(exc),
            "model_training": t4.model.training,
            "followup": _followup_sum(ctx),
        }

    return out


# --------------------------------------------------------------------------- Scenario 5: Caches, logs, broadcast, synth


def _sc_caches_logs_startup_and_stochastic(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect()
    assert ctx is not None
    dom_ds = _make_single_window_dataset(world_size * 2 + 1, seed=70, covariates=True, horizon=DOM_HORIZON)
    rep_ds = _make_single_window_dataset(world_size * 2 + 1, seed=71, covariates=False, horizon=REP_HORIZON)
    out: dict[str, Any] = {}

    # 1. Caches off + rank-0-only log notices across head_only, lora_variate, full_ft.
    cache_checks: dict[str, Any] = {}
    for mode in ("head_only", "lora_variate", "full_ft"):
        lora_cfg = LoraConfig(axes=("variate",), dropout=0.0) if mode == "lora_variate" else None
        # Perturb rank 1's initial model weights to verify teacher copies happen AFTER startup broadcast!
        m = _fresh_model(7)
        if rank != 0:
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(rank + 1.0)
        tlog, handler, prev = _capture_trainer_logs()
        try:
            tr = TimestralTrainer(
                _base_config(mode=mode, batch_size=2, max_steps=2, cache_frozen_features=True),
                dom_ds,
                lora_config=lora_cfg,
                replay_config=ReplayConfig(
                    replay_fraction=0.25, distill_weight=0.5, batch_size=2, cache_base_outputs=True
                ),
                replay_dataset=rep_ds,
                model=m,
            )
            init_logs = list(handler.records)
            handler.records.clear()

            teacher_params: Tensor | None = None
            if mode == "head_only":
                assert tr._frozen_head is not None and tr._frozen_base is None
                teacher_params = _flat_tensors(list(tr._frozen_head.parameters()))
            elif mode == "full_ft":
                assert tr._frozen_base is not None and tr._frozen_head is None
                teacher_params = _flat_tensors(list(tr._frozen_base.parameters()))
            else:
                assert tr._frozen_head is None and tr._frozen_base is None

            tr.fit()
            fit_logs = list(handler.records)
        finally:
            _release_trainer_logs(tlog, handler, prev)

        cache_checks[mode] = {
            "caches_none": (
                tr._train_head_cache is None
                and tr._val_head_cache is None
                and tr._replay_head_cache is None
                and tr._replay_base_pos is None
                and tr._replay_base_neg is None
            ),
            "teacher_params": teacher_params,
            "final_params": _flat_tensors(tr._trainable),
            "init_base_cache_logs": sum(1 for _, msg in init_logs if "cache_base_outputs=True" in msg),
            "fit_base_cache_logs": sum(1 for _, msg in fit_logs if "cache_base_outputs=True" in msg),
            "init_head_cache_logs": sum(1 for _, msg in init_logs if "cache_frozen_features=True" in msg),
            "fit_head_cache_logs": sum(1 for _, msg in fit_logs if "cache_frozen_features=True" in msg),
        }
    out["cache_checks"] = cache_checks

    # 2. No cache_base_outputs bypass notice when cache_base_outputs=False or distill_weight=0.
    tlog, handler, prev = _capture_trainer_logs()
    try:
        TimestralTrainer(
            _base_config(mode="full_ft", batch_size=2, max_steps=1),
            dom_ds,
            replay_config=ReplayConfig(replay_fraction=0.25, distill_weight=0.5, cache_base_outputs=False),
            replay_dataset=rep_ds,
            model=_fresh_model(7),
        )
        TimestralTrainer(
            _base_config(mode="full_ft", batch_size=2, max_steps=1),
            dom_ds,
            replay_config=ReplayConfig(replay_fraction=0.25, distill_weight=0.0, cache_base_outputs=True),
            replay_dataset=rep_ds,
            model=_fresh_model(7),
        )
        silent_logs = list(handler.records)
    finally:
        _release_trainer_logs(tlog, handler, prev)
    out["silent_base_cache_logs"] = sum(1 for _, msg in silent_logs if "cache_base_outputs=True" in msg)

    # 3. Default SyntheticReplaySource (replay_dataset=None) + startup consistency + stochastic fit().
    t_synth = TimestralTrainer(
        _base_config(mode="lora_temporal", batch_size=2, max_steps=3, shuffle=True, sign_flip=True, seed=19),
        dom_ds,
        lora_config=LoraConfig(axes=("time",), dropout=0.1),
        replay_config=ReplayConfig(replay_fraction=0.2, distill_weight=0.2, batch_size=2, sign_flip=True),
        model=_fresh_model(8),
    )
    assert isinstance(t_synth.replay_dataset, SyntheticReplaySource)
    # Verify replay loader generator is rank-seeded and leaves global RNG untouched.
    strat_check = t_synth._make_fit_strategy()
    strat_check.prepare_fit()
    try:
        assert strat_check._replay_loader is not None and strat_check._replay_loader.generator is not None
        assert strat_check._replay_loader.generator.initial_seed() == rank_seed(19 + 2_000_009, rank)
        assert strat_check._replay_cycle is not None
        rng_before = torch.get_rng_state()
        _ = strat_check._replay_cycle.next_batch()
        replay_rng_preserved = torch.equal(torch.get_rng_state(), rng_before)
    finally:
        strat_check.cleanup()

    synth_res = t_synth.fit()
    out["synth"] = {
        "steps": synth_res.steps,
        "train_history": synth_res.train_history,
        "params": _flat_tensors(t_synth._trainable),
        "replay_rng_preserved": replay_rng_preserved,
    }

    # 4. Mismatched replay_dataset on rank 1 caught by startup_consistency.
    bad_rep = _make_single_window_dataset(world_size * 2 + 1, seed=999, covariates=False, horizon=REP_HORIZON)
    try:
        TimestralTrainer(
            _base_config(mode="head_only", batch_size=2, max_steps=1),
            dom_ds,
            replay_config=ReplayConfig(replay_fraction=0.25, distill_weight=0.5, batch_size=2),
            replay_dataset=bad_rep if rank == 1 else rep_ds,
            model=_fresh_model(8),
        )
        out["replay_mismatch"] = {"raised": False}
    except Exception as exc:
        out["replay_mismatch"] = {
            "raised": True,
            "exc_type": type(exc),
            "failing_ranks": getattr(exc, "failing_ranks", None),
            "stage": getattr(exc, "stage", None),
            "followup": _followup_sum(ctx),
        }

    return out


# --------------------------------------------------------------------------- Main multi-process test entry points


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_replay_and_distillation_scenarios(world_size: int) -> None:
    scenarios = {
        "step_parity": _sc_step_level_parity,
        "trajectory_parity": _sc_trajectory_parity,
        "skipped_domain": _sc_skipped_domain_step_preserves_replay,
        "failures": _sc_replay_failures,
        "caches_and_startup": _sc_caches_logs_startup_and_stochastic,
    }
    outcomes = run_scenarios(scenarios, world_size=world_size, pg_timeout=15.0, wall_timeout=120.0)

    # --- 1. Step-level parity vs 1-process shipped TimestralTrainer.training_step on union batches ---
    sp = unwrap_results(outcomes["step_parity"])
    dom_ds = _make_single_window_dataset(4, seed=10, covariates=True, horizon=DOM_HORIZON)
    rep_ds = _make_single_window_dataset(4, seed=11, covariates=False, horizon=REP_HORIZON)
    regimes = _step_parity_regimes(world_size)

    for dtype_name, dtype, tol in (("float64", torch.float64, 1e-12), ("float32", torch.float32, 1e-5)):
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
                    rep_u, rep_s = _make_sharded_batch_from_counts(
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
                    assert l2_rel(r0["params"], ref_params) <= tol, (
                        f"{key}: param rel L2 = {l2_rel(r0['params'], ref_params)}"
                    )
                    # Cross-rank exact equality on gradients and updated parameters.
                    for r in range(1, world_size):
                        assert torch.equal(sp[r][key]["grads"], r0["grads"]), f"{key}: rank {r} grad mismatch"
                        assert torch.equal(sp[r][key]["params"], r0["params"]), f"{key}: rank {r} param mismatch"
                        assert sp[r][key]["loss"] == pytest.approx(r0["loss"], rel=0.0, abs=1e-15)

                    # For e_empty_domain_active_replay (float32 full_ft), also prove the empty-domain rank's
                    # replay slice actually contributed (dropping it yields strictly different loss/grads).
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

    # --- 2. Trajectory parity through fit() vs 1-process fit() at (N*B, N*b) ---
    tp = unwrap_results(outcomes["trajectory_parity"])
    for case in _trajectory_cases(world_size):
        name = case["name"]
        ref = _run_trajectory_case(case, batch_size=world_size * 2, replay_batch_size=world_size * 2)
        r0 = tp[0][name]
        assert r0["result"].steps == ref["result"].steps == case["steps"]
        assert len(r0["result"].train_history) == len(ref["result"].train_history)
        # For `full_ft` starting at student == frozen_base with distillation, PyTorch CPU `bs=2` grad vs
        # `no_grad` differs by ~3.6e-7 in fp32, which `smooth_l1_loss(beta=1e-3)` scales by 1/beta (~3.6e-4)
        # on step 1 (verified 6.5e-14 in float64); all other regimes stay well inside 1e-4.
        loss_tol = 5e-4 if (case["mode"] == "full_ft" and case["dw"] > 0.0) else 1e-4
        for (s_got, l_got), (s_ref, l_ref) in zip(r0["result"].train_history, ref["result"].train_history, strict=True):
            assert s_got == s_ref
            assert abs(l_got - l_ref) <= loss_tol * max(1.0, abs(l_ref)), f"{name} step {s_got}: {l_got} vs {l_ref}"
        assert l2_rel(r0["params"], ref["params"]) <= 1e-4, (
            f"{name}: param rel L2 {l2_rel(r0['params'], ref['params'])}"
        )
        for r in range(1, world_size):
            assert torch.equal(tp[r][name]["params"], r0["params"]), f"{name}: rank {r} param divergence"
            assert tp[r][name]["result"] == r0["result"], f"{name}: rank {r} TrainingResult divergence"

    # --- 3. Globally skipped domain step preserves replay position & matches 1-process fit() ---
    sk = unwrap_results(outcomes["skipped_domain"])
    global_b = world_size * 2
    base_dom = _make_single_window_dataset(3 * global_b, seed=55, covariates=True, horizon=DOM_HORIZON)
    dom_with_skip = _ZeroScoredMiddleDataset(base_dom, global_b, 2 * global_b)
    rep_ds_skip = _make_single_window_dataset(global_b * 3, seed=56, covariates=False, horizon=REP_HORIZON)
    ref_skip_trainer = TimestralTrainer(
        _base_config(
            mode="full_ft",
            batch_size=global_b,
            max_steps=4,
            learning_rate=2e-3,
            warmup_steps=1,
            lr_schedule="cosine",
            weight_decay=0.01,
            gradient_clip_val=1.0,
            seed=13,
        ),
        dom_with_skip,
        replay_config=ReplayConfig(
            replay_fraction=0.3, distill_weight=0.4, batch_size=global_b, cache_base_outputs=False
        ),
        replay_dataset=rep_ds_skip,
        model=_fresh_model(5),
    )
    ref_skip_res = ref_skip_trainer.fit()
    ref_skip_params = _flat_tensors(ref_skip_trainer._trainable)

    r0_sk = sk[0]
    assert r0_sk["result"].steps == 4
    assert r0_sk["result"].skipped_batches == 2  # 2 epochs over 3 global batches (1 skipped per epoch)
    # Every active step 0..3 consumed replay steps 0, 1, 2, 3 (never advanced on skipped middle batches!).
    assert r0_sk["recorded_replay_steps"] == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert l2_rel(r0_sk["params"], ref_skip_params) <= 5e-4
    for (s_got, l_got), (s_ref, l_ref) in zip(r0_sk["result"].train_history, ref_skip_res.train_history, strict=True):
        assert s_got == s_ref and abs(l_got - l_ref) <= 5e-4 * max(1.0, abs(l_ref))
    for r in range(world_size):
        assert sk[r]["stoch_skip_ok"] is True
        assert sk[r]["recorded_replay_steps"] == [(0, 0), (1, 1), (2, 2), (3, 3)]
        assert torch.equal(sk[r]["params"], r0_sk["params"])
        assert sk[r]["result"] == r0_sk["result"]

    # --- 4. Replay failure propagation & non-finite replay loss ---
    fl = unwrap_results(outcomes["failures"])
    for key, rank1_exc in (
        ("prep_fail", KeyError),
        ("fetch_fail", RuntimeError),
        ("pre_fetch_fail", RuntimeError),
    ):
        for r in range(world_size):
            entry = fl[r][key]
            assert entry["raised"] is True
            assert entry["model_training"] is False
            assert entry["followup"] == world_size
            if r == 1:
                assert entry["exc_type"] is rank1_exc
            else:
                assert entry["exc_type"] is RankFailureError
                assert entry["failing_ranks"] == (1,)
                assert entry["stage"] == "training_step"
    for r in range(world_size):
        nf = fl[r]["non_finite_replay"]
        assert nf["raised"] is True
        assert nf["exc_type"] is FloatingPointError
        assert nf["model_training"] is False
        assert nf["followup"] == world_size

    # --- 5. Caches off, rank-0 logs once, post-broadcast teachers, SyntheticReplaySource, consistency ---
    cs = unwrap_results(outcomes["caches_and_startup"])
    for mode in ("head_only", "lora_variate", "full_ft"):
        r0_cc = cs[0]["cache_checks"][mode]
        for r in range(world_size):
            cc = cs[r]["cache_checks"][mode]
            assert cc["caches_none"] is True
            assert torch.equal(cc["final_params"], r0_cc["final_params"])
            if mode in ("head_only", "full_ft"):
                assert cc["teacher_params"] is not None and r0_cc["teacher_params"] is not None
                assert torch.equal(cc["teacher_params"], r0_cc["teacher_params"])
            if r == 0:
                assert cc["init_base_cache_logs"] == (0 if mode == "head_only" else 1)
                assert cc["fit_base_cache_logs"] == 0
                assert cc["init_head_cache_logs"] == (1 if mode == "head_only" else 0)
                assert cc["fit_head_cache_logs"] == 0
            else:
                assert cc["init_base_cache_logs"] == 0 and cc["fit_base_cache_logs"] == 0
                assert cc["init_head_cache_logs"] == 0 and cc["fit_head_cache_logs"] == 0

    for r in range(world_size):
        assert cs[r]["silent_base_cache_logs"] == 0
        assert cs[r]["synth"]["steps"] == 3
        assert cs[r]["synth"]["replay_rng_preserved"] is True
        assert torch.equal(cs[r]["synth"]["params"], cs[0]["synth"]["params"])
        assert cs[r]["synth"]["train_history"] == cs[0]["synth"]["train_history"]

        mm = cs[r]["replay_mismatch"]
        assert mm["raised"] is True and mm["followup"] == world_size
        if r == 1:
            assert mm["exc_type"] is ValueError
        else:
            assert mm["exc_type"] is RankFailureError
            assert mm["failing_ranks"] == (1,)
            assert mm["stage"] == "startup_consistency"
