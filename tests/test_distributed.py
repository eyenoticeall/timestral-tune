"""Unit and multi-process tests for `timestral_tune.trainer.distributed` and `distributed_harness`."""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset

from timestral_tune.trainer import distributed as dist_mod
from timestral_tune.trainer.distributed import (
    ControlStatus,
    DistributedBatchCycle,
    DistributedBatchSampler,
    DistributedContext,
    GradientReducer,
    RankFailureError,
    broadcast_module_state,
    broadcast_status,
    is_oom_error,
    rank_seed,
    rank_slice_size,
    rank_step_indices,
    steps_per_epoch,
    validation_shard_indices,
)

from .distributed_harness import (
    F32_TOL_SCALE_ENV,
    DistributedTimeoutError,
    RankResult,
    barrier,
    dist_backend,
    dist_device_map,
    expected_worker_device,
    f32_tol_scale,
    is_cuda_dist_mode,
    run_distributed,
    run_scenarios,
    scale_timeout,
    trainer_device_for_mode,
    unwrap_results,
)

# PyTorch < 2.8 on macOS hangs at process exit after DataLoader worker processes run (docs/VERIFICATION.md §10).
WORKERS_HANG_AT_EXIT = sys.platform == "darwin" and tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


# --------------------------------------------------------------------------- helpers & tiny fixtures


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


class _ToyModule(nn.Module):
    """Module exercising trainable params, frozen params, float/int buffers, and non-contiguous buffers."""

    def __init__(self, rank: int, device: torch.device | str = "cpu") -> None:
        super().__init__()
        base = float(rank + 1)
        self.weight = nn.Parameter(torch.full((3, 4), base, dtype=torch.float32, device=device), requires_grad=True)
        self.frozen = nn.Parameter(
            torch.full((2, 3), base * 10.0, dtype=torch.float32, device=device), requires_grad=False
        )
        self.register_buffer("scale_buf", torch.full((4,), base * 0.5, dtype=torch.float64, device=device))
        self.register_buffer("step_buf", torch.tensor([rank, rank + 7], dtype=torch.int64, device=device))
        # Non-contiguous transposed buffer.
        nc = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3).t() + base
        assert not nc.is_contiguous()
        self.register_buffer("non_contig_buf", nc)


# --------------------------------------------------------------------------- sampling arithmetic (in-process)


SAMPLING_GRID = [
    # (num_items, world_size, batch_size)
    (0, 2, 2),
    (0, 4, 3),
    (1, 2, 2),
    (1, 4, 2),
    (2, 4, 1),
    (3, 4, 2),
    (4, 4, 2),
    (5, 4, 2),
    (7, 2, 3),
    (8, 2, 2),
    (9, 4, 2),
    (10, 4, 2),
    (11, 4, 2),
    (12, 4, 2),
    (13, 4, 2),
    (15, 4, 2),
    (16, 4, 2),
    (17, 3, 2),
    (11, 1, 4),
    (12, 1, 4),
]


@pytest.mark.parametrize("num_items,world_size,batch_size", SAMPLING_GRID)
@pytest.mark.parametrize("shuffle", [False, True])
def test_sampling_arithmetic_exact_coverage_union_and_slice_sizes(
    num_items: int, world_size: int, batch_size: int, shuffle: bool
) -> None:
    expected_steps = 0 if num_items == 0 else math.ceil(num_items / (world_size * batch_size))
    n_steps = steps_per_epoch(num_items, world_size, batch_size)
    assert n_steps == expected_steps

    global_batch = world_size * batch_size
    seen: list[int] = []
    for step in range(n_steps):
        step_start = step * global_batch
        remaining = min(global_batch, num_items - step_start)
        step_union: list[int] = []
        for rank in range(world_size):
            size = rank_slice_size(num_items, world_size, batch_size, rank, step)
            expected_size = remaining // world_size + (1 if rank < (remaining % world_size) else 0)
            assert size == expected_size
            # Empty slice occurs if and only if remaining < world_size and rank >= remaining.
            assert (size == 0) == (remaining < world_size and rank >= remaining)

            indices = rank_step_indices(
                num_items, world_size, batch_size, rank, step, shuffle=shuffle, seed=17, epoch=2
            )
            assert len(indices) == size
            step_union.extend(indices)

        assert len(step_union) == remaining
        if not shuffle:
            assert step_union == list(range(step_start, step_start + remaining))
        seen.extend(step_union)

    assert len(seen) == num_items
    assert sorted(seen) == list(range(num_items))
    if not shuffle:
        assert seen == list(range(num_items))


def test_shuffled_epochs_differ_and_agree_across_ranks() -> None:
    num_items, world_size, batch_size = 40, 4, 3
    n_steps = steps_per_epoch(num_items, world_size, batch_size)

    def epoch_order(seed: int, epoch: int) -> list[int]:
        out: list[int] = []
        for step in range(n_steps):
            for rank in range(world_size):
                out.extend(
                    rank_step_indices(
                        num_items, world_size, batch_size, rank, step, shuffle=True, seed=seed, epoch=epoch
                    )
                )
        return out

    e0 = epoch_order(seed=42, epoch=0)
    e1 = epoch_order(seed=42, epoch=1)
    e2 = epoch_order(seed=42, epoch=2)
    s_other = epoch_order(seed=43, epoch=0)
    # Guard against naive `seed + epoch` aliasing: (seed=0, epoch=1) vs (seed=1, epoch=0).
    alias_a = epoch_order(seed=0, epoch=1)
    alias_b = epoch_order(seed=1, epoch=0)

    assert len({tuple(e0), tuple(e1), tuple(e2), tuple(s_other)}) == 4
    assert alias_a != alias_b
    assert e0 != list(range(num_items))
    # Repeatability: re-computing gives bit-identical order.
    assert epoch_order(seed=42, epoch=1) == e1


@pytest.mark.parametrize("num_items,world_size,batch_size", [(0, 4, 2), (3, 4, 2), (10, 4, 2), (13, 4, 2), (16, 4, 2)])
@pytest.mark.parametrize("shuffle", [False, True])
def test_batch_sampler_matches_arithmetic_pickles_and_sets_epoch(
    num_items: int, world_size: int, batch_size: int, shuffle: bool
) -> None:
    n_steps = steps_per_epoch(num_items, world_size, batch_size)
    for rank in range(world_size):
        sampler = DistributedBatchSampler(
            num_items, world_size=world_size, rank=rank, batch_size=batch_size, shuffle=shuffle, seed=99
        )
        assert sampler.num_global_steps == n_steps
        for step in range(n_steps):
            assert sampler.slice_size(step) == rank_slice_size(num_items, world_size, batch_size, rank, step)
        for epoch in (0, 1, 2):
            sampler.set_epoch(epoch)
            assert sampler.epoch == epoch
            batches = list(sampler)
            assert len(batches) == len(sampler)
            assert all(len(b) > 0 for b in batches)
            expected = [
                idx
                for step in range(n_steps)
                if (
                    idx := rank_step_indices(
                        num_items, world_size, batch_size, rank, step, shuffle=shuffle, seed=99, epoch=epoch
                    )
                )
            ]
            assert batches == expected

        # Pickle round-trip preserves epoch and yields identical batches.
        restored = pickle.loads(pickle.dumps(sampler))
        assert restored.epoch == sampler.epoch
        assert list(restored) == list(sampler)
        restored.set_epoch(5)
        sampler.set_epoch(5)
        assert list(restored) == list(sampler)


def test_batch_sampler_in_dataloader_single_process() -> None:
    dataset = _IndexDataset(13)
    sampler = DistributedBatchSampler(len(dataset), world_size=4, rank=1, batch_size=2, shuffle=True, seed=7)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=_identity_collate, num_workers=0)
    sampler.set_epoch(0)
    e0 = list(loader)
    sampler.set_epoch(1)
    e1 = list(loader)
    expected_e0 = DistributedBatchSampler(13, world_size=4, rank=1, batch_size=2, shuffle=True, seed=7, epoch=0)
    assert e0 == list(expected_e0)
    assert e1 != e0
    assert len(e0) == len(expected_e0)


@pytest.mark.skipif(WORKERS_HANG_AT_EXIT, reason="PyTorch < 2.8 on macOS: DataLoader worker hangs at process exit")
def test_batch_sampler_with_persistent_workers() -> None:
    # Regular multi-epoch run with num_workers > 0 and persistent_workers=True.
    dataset = _IndexDataset(15)
    sampler = DistributedBatchSampler(len(dataset), world_size=4, rank=2, batch_size=2, shuffle=True, seed=11)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=_identity_collate,
        num_workers=2,
        persistent_workers=True,
    )
    epoch_outputs: list[list[list[int]]] = []
    for epoch in range(3):
        sampler.set_epoch(epoch)
        got = list(loader)
        assert got == list(sampler)
        epoch_outputs.append(got)
    assert epoch_outputs[0] != epoch_outputs[1]
    assert epoch_outputs[1] != epoch_outputs[2]

    # Empty-rank case (num_items < world_size, rank >= num_items) must also work cleanly.
    small_ds = _IndexDataset(2)
    empty_sampler = DistributedBatchSampler(len(small_ds), world_size=4, rank=3, batch_size=2, shuffle=False)
    empty_loader = DataLoader(
        small_ds,
        batch_sampler=empty_sampler,
        collate_fn=_identity_collate,
        num_workers=1,
        persistent_workers=True,
    )
    assert list(empty_loader) == []


@pytest.mark.parametrize("num_items,world_size", [(0, 2), (1, 4), (3, 4), (4, 4), (10, 4), (11, 3)])
def test_validation_shard_indices_partition(num_items: int, world_size: int) -> None:
    shards = [validation_shard_indices(num_items, world_size, r) for r in range(world_size)]
    for r, shard in enumerate(shards):
        assert shard == list(range(r, num_items, world_size))
    flattened = [i for shard in shards for i in shard]
    assert len(flattened) == num_items
    assert sorted(flattened) == list(range(num_items))


def test_rank_seed_formula_and_wraparound() -> None:
    for base in (0, 42, -5, (1 << 63) - 10):
        assert rank_seed(base, 0) == base
        seen = {rank_seed(base, 0)}
        for r in range(1, 8):
            expected = (base + r * 1_000_000_007) % (1 << 63)
            got = rank_seed(base, r)
            assert got == expected
            assert 0 <= got < (1 << 63)
            seen.add(got)
        assert len(seen) == 8


def test_sampling_and_seed_validation_errors() -> None:
    with pytest.raises(ValueError):
        steps_per_epoch(-1, 2, 2)
    with pytest.raises(ValueError):
        steps_per_epoch(4, 0, 2)
    with pytest.raises(ValueError):
        steps_per_epoch(4, 2, 0)
    with pytest.raises(ValueError):
        rank_slice_size(0, 2, 2, 0, 0)
    with pytest.raises(ValueError):
        rank_slice_size(4, 2, 2, 2, 0)
    with pytest.raises(ValueError):
        rank_slice_size(4, 2, 2, 0, 1)
    sampler = DistributedBatchSampler(4, world_size=2, rank=0, batch_size=2)
    with pytest.raises(ValueError):
        sampler.slice_size(1)
    with pytest.raises(ValueError):
        sampler.slice_size(-1)
    with pytest.raises(ValueError):
        rank_step_indices(4, 2, 2, 0, 0, epoch=-1)
    with pytest.raises(ValueError):
        validation_shard_indices(4, 2, 2)
    with pytest.raises(ValueError):
        rank_seed(0, -1)
    with pytest.raises(TypeError):
        rank_seed(True, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        steps_per_epoch(4, True, 2)  # type: ignore[arg-type]


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


# --------------------------------------------------------------------------- OOM, errors, backend rules (in-process)


def test_is_oom_error_predicate() -> None:
    cuda_oom = torch.cuda.OutOfMemoryError("CUDA out of memory")
    assert is_oom_error(cuda_oom) is True

    class _SubOOM(torch.cuda.OutOfMemoryError):
        pass

    assert is_oom_error(_SubOOM("sub")) is True

    if hasattr(torch, "OutOfMemoryError"):
        assert is_oom_error(torch.OutOfMemoryError("torch oom")) is True

    assert is_oom_error(RuntimeError("CUDA out of memory")) is False
    assert is_oom_error(MemoryError("cpu oom")) is False
    assert is_oom_error(ValueError("bad")) is False
    assert is_oom_error(KeyboardInterrupt()) is False


def test_rank_failure_error_contract_and_pickling() -> None:
    err = RankFailureError([3, 1, 3], stage="startup")
    assert isinstance(err, RuntimeError)
    assert err.failing_ranks == (1, 3)
    assert err.stage == "startup"
    msg = str(err)
    assert "startup" in msg
    assert "1" in msg and "3" in msg

    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is RankFailureError
    assert restored.failing_ranks == (1, 3)
    assert restored.stage == "startup"
    assert str(restored) == msg

    with pytest.raises(ValueError):
        RankFailureError([], stage="startup")
    with pytest.raises(ValueError):
        RankFailureError([0], stage="")


def test_detect_inactive_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DistributedContext.detect() is None

    # Initialised group with world_size == 1 is NOT multi-process mode.
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_backend", lambda: "unsupported_ignored_at_ws1")
    assert DistributedContext.detect() is None


@pytest.mark.parametrize("bad_backend", ["cpu:gloo", "cuda:nccl", "mpi", "ucc", "GLOO", "NCCL", ""])
def test_detect_rejects_unsupported_backends(monkeypatch: pytest.MonkeyPatch, bad_backend: str) -> None:
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_backend", lambda: bad_backend)
    with pytest.raises(ValueError, match="unsupported torch.distributed backend"):
        DistributedContext.detect()


def test_detect_gloo_device_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")

    for ok_dev in ("auto", None, "cpu", torch.device("cpu")):
        ctx = DistributedContext.detect(ok_dev)
        assert ctx == DistributedContext(rank=1, world_size=2, backend="gloo", device=torch.device("cpu"))
        assert not ctx.is_rank_zero

    for bad_dev in ("cuda", "cuda:0", "mps", "cpu:0", torch.device("cuda", 0), torch.device("mps"), "bogus"):
        with pytest.raises(ValueError):
            DistributedContext.detect(bad_dev)


def test_detect_nccl_device_rules_monkeypatched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercises all NCCL device-selection branches in-process by monkeypatching `torch.cuda`."""
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "get_rank", lambda: 2)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    selected: list[int] = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: selected.append(idx))

    # 1. Explicit indexed CUDA device overrides LOCAL_RANK (even if LOCAL_RANK is unparseable).
    monkeypatch.setenv("LOCAL_RANK", "not-an-int")
    ctx = DistributedContext.detect("cuda:3")
    assert ctx == DistributedContext(rank=2, world_size=4, backend="nccl", device=torch.device("cuda", 3))
    assert selected == [3]

    selected.clear()
    ctx = DistributedContext.detect(torch.device("cuda", 1))
    assert ctx is not None and ctx.device == torch.device("cuda", 1)
    assert selected == [1]

    # 2. LOCAL_RANK used when device is "auto", None, or unindexed "cuda".
    monkeypatch.setenv("LOCAL_RANK", "1")
    for req in ("auto", None, "cuda", torch.device("cuda")):
        selected.clear()
        ctx = DistributedContext.detect(req)
        assert ctx is not None and ctx.device == torch.device("cuda", 1)
        assert selected == [1]

    # 3. Rank used when LOCAL_RANK is unset.
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    selected.clear()
    ctx = DistributedContext.detect("auto")
    assert ctx is not None and ctx.device == torch.device("cuda", 2)
    assert selected == [2]

    # 4. Invalid LOCAL_RANK raises ValueError without calling set_device.
    for bad_env in ("", "abc", "1.5"):
        monkeypatch.setenv("LOCAL_RANK", bad_env)
        selected.clear()
        with pytest.raises(ValueError, match="LOCAL_RANK"):
            DistributedContext.detect("auto")
        assert selected == []

    # 5. Out-of-range index (negative, >= device_count, or device_count == 0) raises ValueError.
    monkeypatch.setenv("LOCAL_RANK", "-1")
    with pytest.raises(ValueError, match="out of range"):
        DistributedContext.detect("auto")
    monkeypatch.setenv("LOCAL_RANK", "4")
    with pytest.raises(ValueError, match="out of range"):
        DistributedContext.detect("auto")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with pytest.raises(ValueError, match="out of range"):
        DistributedContext.detect("cuda:4")

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="out of range"):
        DistributedContext.detect("auto")

    # 6. Non-CUDA devices (including MPS and CPU) raise ValueError without calling set_device.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    for bad_dev in ("cpu", "mps", torch.device("cpu"), torch.device("mps"), "bogus"):
        selected.clear()
        with pytest.raises(ValueError):
            DistributedContext.detect(bad_dev)
        assert selected == []


def test_distributed_context_direct_construction_validation() -> None:
    """Direct `DistributedContext(...)` normalises integer fields and enforces backend/device compatibility."""
    ctx_gloo = DistributedContext(
        rank=np.int64(1),  # type: ignore[arg-type]
        world_size=np.int64(4),  # type: ignore[arg-type]
        backend="gloo",
        device=torch.device("cpu"),
    )
    assert type(ctx_gloo.rank) is int and ctx_gloo.rank == 1
    assert type(ctx_gloo.world_size) is int and ctx_gloo.world_size == 4

    ctx_nccl = DistributedContext(
        rank=np.int32(0),  # type: ignore[arg-type]
        world_size=np.int32(2),  # type: ignore[arg-type]
        backend="nccl",
        device=torch.device("cuda", 0),
    )
    assert type(ctx_nccl.rank) is int and ctx_nccl.rank == 0
    assert type(ctx_nccl.world_size) is int and ctx_nccl.world_size == 2

    # Impossible backend/device pairs rejected just as in detect().
    for bad_gloo_dev in (torch.device("cuda", 0), torch.device("cuda"), torch.device("cpu", 0), torch.device("mps")):
        with pytest.raises(ValueError, match="gloo"):
            DistributedContext(rank=0, world_size=2, backend="gloo", device=bad_gloo_dev)

    for bad_nccl_dev in (torch.device("cpu"), torch.device("cuda"), torch.device("mps")):
        with pytest.raises(ValueError, match="nccl"):
            DistributedContext(rank=0, world_size=2, backend="nccl", device=bad_nccl_dev)

    with pytest.raises(TypeError, match="bool"):
        DistributedContext(rank=False, world_size=2, backend="gloo", device=torch.device("cpu"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.device"):
        DistributedContext(rank=0, world_size=2, backend="gloo", device="cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="world_size"):
        DistributedContext(rank=0, world_size=1, backend="gloo", device=torch.device("cpu"))
    with pytest.raises(ValueError, match="rank"):
        DistributedContext(rank=2, world_size=2, backend="gloo", device=torch.device("cpu"))
    with pytest.raises(ValueError, match="backend"):
        DistributedContext(rank=0, world_size=2, backend="cpu:gloo", device=torch.device("cpu"))  # type: ignore[arg-type]


def test_stale_group_check_and_non_propagated_errors_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = DistributedContext(rank=0, world_size=2, backend="gloo", device=torch.device("cpu"))

    # Group destroyed -> RuntimeError.
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="destroyed"):
        ctx.check_active()

    # World size changed -> RuntimeError.
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    with pytest.raises(RuntimeError, match="world size changed"):
        ctx.check_active()

    # OOM and non-Exception BaseException re-raise immediately WITHOUT entering collectives.
    called_all_reduce = False

    def _forbidden_all_reduce(*_a: Any, **_kw: Any) -> None:
        nonlocal called_all_reduce
        called_all_reduce = True

    monkeypatch.setattr(dist, "all_reduce", _forbidden_all_reduce)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        ctx.propagate_failure(torch.cuda.OutOfMemoryError("oom"), stage="step")
    with pytest.raises(KeyboardInterrupt):
        ctx.propagate_failure(KeyboardInterrupt(), stage="step")  # type: ignore[arg-type]
    assert not called_all_reduce

    # raise_propagated_failure asserts if invoked when no rank actually supplied an error.
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None: None)
    with pytest.raises(AssertionError, match="no rank supplied a local_error"):
        ctx.raise_propagated_failure(None, stage="training_step")


def test_gradient_reducer_construction_validation() -> None:
    ctx = DistributedContext(rank=0, world_size=2, backend="gloo", device=torch.device("cpu"))
    p_f32 = nn.Parameter(torch.ones(2, dtype=torch.float32))
    p_f64 = nn.Parameter(torch.ones(2, dtype=torch.float64))
    p_meta = nn.Parameter(torch.empty(2, dtype=torch.float32, device="meta"))

    with pytest.raises(ValueError, match="at least one parameter"):
        GradientReducer(ctx, [])
    with pytest.raises(ValueError, match="duplicate"):
        GradientReducer(ctx, [p_f32, p_f32])
    with pytest.raises(ValueError, match="share one dtype"):
        GradientReducer(ctx, [p_f32, p_f64])
    with pytest.raises(ValueError, match="share one device"):
        GradientReducer(ctx, [p_f32, p_meta])
    with pytest.raises(ValueError, match="DistributedContext uses"):
        GradientReducer(ctx, [p_meta])


def test_gradient_reducer_uses_single_collective_with_presence_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """`GradientReducer.reduce()` carries presence flags inside the single `all_reduce(SUM)` buffer."""
    ctx = DistributedContext(rank=0, world_size=2, backend="gloo", device=torch.device("cpu"))
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")

    calls: list[tuple[int, Any]] = []

    def _fake_all_reduce(tensor: torch.Tensor, op: Any = None) -> None:
        calls.append((int(tensor.numel()), op))
        # Simulate summing two identical ranks: multiply buffer in place by 2.
        tensor.mul_(2.0)

    monkeypatch.setattr(dist, "all_reduce", _fake_all_reduce)

    p_none = nn.Parameter(torch.ones(3, dtype=torch.float32))
    p_zero = nn.Parameter(torch.ones(2, dtype=torch.float32))
    p_vals = nn.Parameter(torch.ones(4, dtype=torch.float32))
    reducer = GradientReducer(ctx, [p_none, p_zero, p_vals])
    assert reducer.numel == 3 + 2 + 4

    p_none.grad = None
    p_zero.grad = torch.zeros(2, dtype=torch.float32)
    p_vals.grad = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    reducer.reduce()

    assert calls == [(3 + 2 + 4 + 3, dist.ReduceOp.SUM)]
    assert p_none.grad is None
    assert p_zero.grad is not None and p_zero.grad.tolist() == [0.0, 0.0]
    assert p_vals.grad is not None and p_vals.grad.tolist() == [2.0, 4.0, 6.0, 8.0]


def test_gradient_reducer_all_present_skips_readback_and_all_absent_reuses_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-present skips `_flags_slice.tolist()`; all-absent uses preallocated zeros and reads flags."""
    ctx = DistributedContext(rank=0, world_size=2, backend="gloo", device=torch.device("cpu"))
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")

    p0 = nn.Parameter(torch.ones(2, dtype=torch.float32))
    p1 = nn.Parameter(torch.ones(3, dtype=torch.float32))
    reducer = GradientReducer(ctx, [p0, p1])

    # 1. All local gradients present: single all_reduce carries [1.0, 1.0] flags, skips `.tolist()`.
    calls: list[int] = []

    def _double_all_reduce(tensor: torch.Tensor, op: Any = None) -> None:
        assert op == dist.ReduceOp.SUM
        calls.append(int(tensor.numel()))
        tensor.mul_(2.0)

    monkeypatch.setattr(dist, "all_reduce", _double_all_reduce)

    real_flags = reducer._flags_slice

    class _NoReadbackFlags:
        def copy_(self, src: torch.Tensor) -> torch.Tensor:
            return real_flags.copy_(src)

        def tolist(self) -> list[float]:
            raise AssertionError("all-present fast path must not read back presence flags")

    reducer._flags_slice = _NoReadbackFlags()  # type: ignore[assignment]
    p0.grad = torch.tensor([1.5, -2.5], dtype=torch.float32)
    p1.grad = torch.tensor([0.0, 3.0, 4.0], dtype=torch.float32)
    reducer.reduce()

    assert calls == [2 + 3 + 2]
    assert real_flags.tolist() == [2.0, 2.0]
    assert p0.grad is not None and p0.grad.tolist() == [3.0, -5.0]
    assert p1.grad is not None and p1.grad.tolist() == [0.0, 6.0, 8.0]

    # 2. All local gradients None: uses preallocated zero flags and reads back peer presence.
    reducer._flags_slice = real_flags

    def _peer_populated_p1(tensor: torch.Tensor, op: Any = None) -> None:
        assert op == dist.ReduceOp.SUM
        # Verify caller wrote [0.0, 0.0] into flags before collective.
        assert tensor[-2:].tolist() == [0.0, 0.0]
        # Simulate peer rank contributing grad for p1 only (flag=1.0, values=[5, 6, 7]).
        tensor[2:5].copy_(torch.tensor([5.0, 6.0, 7.0]))
        tensor[-1].fill_(1.0)

    monkeypatch.setattr(dist, "all_reduce", _peer_populated_p1)
    p0.grad = None
    p1.grad = None
    reducer.reduce()
    assert p0.grad is None
    assert p1.grad is not None and p1.grad.tolist() == [5.0, 6.0, 7.0]


# --------------------------------------------------------------------------- multi-process scenarios (N=2 and N=4)


def _sc_snapshot_and_device_guards(rank: int, world_size: int) -> dict[str, Any]:
    assert torch.get_num_threads() == 1
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    assert ctx.rank == rank
    assert ctx.world_size == world_size
    assert ctx.backend == dist_backend()
    expected_dev = expected_worker_device(rank)
    assert ctx.device == expected_dev
    assert ctx.is_rank_zero == (rank == 0)
    ctx.check_active()

    if is_cuda_dist_mode():
        assert os.environ.get("LOCAL_RANK") == str(rank)
        assert expected_dev.index is not None
        assert torch.cuda.current_device() == expected_dev.index
        if dist_device_map() == "per-rank":
            assert DistributedContext.detect() == ctx
            assert DistributedContext.detect(None) == ctx
            assert DistributedContext.detect("cuda") == ctx
            assert DistributedContext.detect(f"cuda:{rank}") == ctx
        else:
            assert DistributedContext.detect("cuda:0") == ctx
        with pytest.raises(ValueError):
            DistributedContext.detect("cpu")
        with pytest.raises(ValueError):
            DistributedContext.detect("mps")
    else:
        assert DistributedContext.detect("cpu") == ctx
        assert DistributedContext.detect(None) == ctx
        with pytest.raises(ValueError):
            DistributedContext.detect("cuda")
        with pytest.raises(ValueError):
            DistributedContext.detect("mps")
    return {"rank": ctx.rank, "world_size": ctx.world_size}


def _sc_sum_and_broadcast_roundtrips(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    dev = ctx.device

    # SUM reductions across float64, float32, int64, and non-contiguous views.
    t_f64 = torch.tensor([float(rank + 1), float((rank + 1) ** 2)], dtype=torch.float64, device=dev)
    ctx.all_reduce_sum(t_f64)
    t_i64 = torch.tensor([rank + 10], dtype=torch.int64, device=dev)
    ctx.all_reduce_sum(t_i64)
    nc = torch.full((3, 2), float(rank + 1), dtype=torch.float32, device=dev).t()
    assert not nc.is_contiguous()
    ctx.all_reduce_sum(nc)

    # Broadcasts from rank 0: plain tensor, non-contiguous view, and requires_grad=True parameter.
    b_i64 = torch.tensor([100 + rank, 200 + rank], dtype=torch.int64, device=dev)
    ctx.broadcast(b_i64)
    b_nc = torch.full((3, 2), float(rank + 5), dtype=torch.float32, device=dev).t()
    assert not b_nc.is_contiguous()
    ctx.broadcast(b_nc)
    param = nn.Parameter(torch.full((2, 2), float(rank + 1), dtype=torch.float32, device=dev), requires_grad=True)
    ctx.broadcast(param)
    assert param.requires_grad is True

    return {
        "sum_f64": t_f64.tolist(),
        "sum_i64": t_i64.tolist(),
        "sum_nc": nc.tolist(),
        "b_i64": b_i64.tolist(),
        "b_nc": b_nc.tolist(),
        "param": param.detach().tolist(),
    }


def _run_failure_case_with_followup(
    ctx: DistributedContext, local_error: Exception | None, stage: str
) -> dict[str, Any]:
    caught: dict[str, Any]
    try:
        ctx.propagate_failure(local_error, stage=stage)
        caught = {"raised": False}
    except Exception as exc:
        caught = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "failing_ranks": exc.failing_ranks if isinstance(exc, RankFailureError) else None,
            "stage": exc.stage if isinstance(exc, RankFailureError) else None,
        }
    # Follow-up collective must succeed cleanly on every rank.
    probe = torch.tensor([1], dtype=torch.int64, device=ctx.device)
    ctx.all_reduce_sum(probe)
    caught["followup_sum"] = int(probe.item())
    return caught


def _sc_failure_nonzero_rank(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    err = KeyError("missing_col") if rank == 1 else None
    return _run_failure_case_with_followup(ctx, err, stage="data_load")


def _sc_failure_rank_zero(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    err = ValueError("bad_rank0_cfg") if rank == 0 else None
    return _run_failure_case_with_followup(ctx, err, stage="startup")


def _sc_failure_two_ranks(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    failing = (0, 1) if world_size == 2 else (1, 3)
    err: Exception | None = None
    if rank == failing[0]:
        err = ValueError("first_bad")
    elif rank == failing[1]:
        err = KeyError("second_bad")
    return _run_failure_case_with_followup(ctx, err, stage="train_step")


def _sc_failure_none(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    return _run_failure_case_with_followup(ctx, None, stage="startup")


def _sc_failure_uncaught_propagates(rank: int, world_size: int) -> None:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    err = ValueError("uncaught_rank1") if rank == 1 else None
    ctx.propagate_failure(err, stage="uncaught_stage")


def _sc_consistency_match(rank: int, world_size: int) -> bool:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    ctx.check_consistency(hashlib.sha256(b"same-model-layout").digest(), stage="layout")
    ctx.check_consistency(b"arbitrary-length-digest", stage="config")
    return True


def _sc_consistency_mismatch_single(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    # Deliberately use a different byte length on the last rank to verify fixed-size encoding avoids deadlock.
    digest = b"short-mismatch" if rank == world_size - 1 else hashlib.sha256(b"canonical").digest()
    try:
        ctx.check_consistency(digest, stage="model_schema")
        out: dict[str, Any] = {"raised": False}
    except Exception as exc:
        out = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "failing_ranks": exc.failing_ranks if isinstance(exc, RankFailureError) else None,
            "stage": exc.stage if isinstance(exc, RankFailureError) else None,
        }
    probe = torch.tensor([1], dtype=torch.int64, device=ctx.device)
    ctx.all_reduce_sum(probe)
    out["followup_sum"] = int(probe.item())
    return out


def _sc_consistency_mismatch_two(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    mismatched = (1,) if world_size == 2 else (1, 2)
    digest = f"divergent-{rank}".encode() if rank in mismatched else b"canonical"
    try:
        ctx.check_consistency(digest, stage="weights_digest")
        out: dict[str, Any] = {"raised": False}
    except Exception as exc:
        out = {
            "raised": True,
            "exc_type": type(exc),
            "message": str(exc),
            "failing_ranks": exc.failing_ranks if isinstance(exc, RankFailureError) else None,
            "stage": exc.stage if isinstance(exc, RankFailureError) else None,
        }
    probe = torch.tensor([1], dtype=torch.int64, device=ctx.device)
    ctx.all_reduce_sum(probe)
    out["followup_sum"] = int(probe.item())
    return out


def _sc_module_state_broadcast(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    module = _ToyModule(rank, device=ctx.device)
    broadcast_module_state(ctx, module)
    return {
        "weight": module.weight.detach().tolist(),
        "weight_req_grad": module.weight.requires_grad,
        "frozen": module.frozen.detach().tolist(),
        "frozen_req_grad": module.frozen.requires_grad,
        "scale_buf": module.scale_buf.tolist(),
        "scale_buf_req_grad": module.scale_buf.requires_grad,
        "step_buf": module.step_buf.tolist(),
        "non_contig_buf": module.non_contig_buf.tolist(),
        "non_contig_is_contig": module.non_contig_buf.is_contiguous(),
    }


def _sc_gradient_reducer(rank: int, world_size: int) -> dict[str, Any]:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    dev = ctx.device
    p0 = nn.Parameter(torch.zeros(3, dtype=torch.float64, device=dev))
    p1 = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64, device=dev))
    p2 = nn.Parameter(torch.zeros(4, dtype=torch.float64, device=dev))
    p3 = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=dev))  # 0-D scalar parameter
    reducer = GradientReducer(ctx, [p0, p1, p2, p3])
    assert reducer.numel == 3 + 4 + 4 + 1

    # Round 1: p0 present on all ranks, p1 None on rank 0 only, p2 None on all ranks, p3 scalar on all ranks.
    p0.grad = torch.tensor([rank + 1.0, (rank + 1.0) * 2.0, -float(rank)], dtype=torch.float64, device=dev)
    p1.grad = None if rank == 0 else torch.full((2, 2), float(rank + 3), dtype=torch.float64, device=dev)
    p2.grad = None
    p3.grad = torch.tensor(float(rank + 2), dtype=torch.float64, device=dev)
    reducer.reduce()
    assert p0.grad is not None and p1.grad is not None and p2.grad is None and p3.grad is not None
    assert p3.grad.shape == ()
    r1 = {"p0": p0.grad.tolist(), "p1": p1.grad.tolist(), "p2": None, "p3": p3.grad.item()}

    # Round 2 (reusing the same flat buffer): flip None pattern; p2 is an all-zero tensor on rank 0 and None on peers.
    p0.grad = None if rank == world_size - 1 else torch.full((3,), 0.5 * (rank + 1), dtype=torch.float64, device=dev)
    p1.grad = torch.arange(4, dtype=torch.float64, device=dev).reshape(2, 2) * (rank + 1)
    p2.grad = torch.full((4,), float(rank), dtype=torch.float64, device=dev) if rank == 0 else None
    p3.grad = None if rank != 0 else torch.tensor(-7.5, dtype=torch.float64, device=dev)
    reducer.reduce()
    assert p0.grad is not None and p1.grad is not None and p2.grad is not None and p3.grad is not None
    r2 = {"p0": p0.grad.tolist(), "p1": p1.grad.tolist(), "p2": p2.grad.tolist(), "p3": p3.grad.item()}

    # Round 3: previously populated parameters that become all-None return to None; present zero scalar stays tensor.
    p0.grad = None
    p1.grad = torch.full((2, 2), 2.5, dtype=torch.float64, device=dev) if rank == 1 else None
    p2.grad = None
    p3.grad = torch.tensor(0.0, dtype=torch.float64, device=dev) if rank == 0 else None
    reducer.reduce()
    assert p0.grad is None and p1.grad is not None and p2.grad is None and p3.grad is not None
    assert p3.grad.shape == ()
    r3 = {"p0": None, "p1": p1.grad.tolist(), "p2": None, "p3": p3.grad.item()}

    # Round 4: asymmetric fast path — rank 0 has all grads present (skips flag read-back),
    # while rank 1 has all grads None (reads back flags and allocates every param's .grad).
    if rank == 0:
        p0.grad = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=dev)
        p1.grad = torch.full((2, 2), 4.0, dtype=torch.float64, device=dev)
        p2.grad = torch.zeros(4, dtype=torch.float64, device=dev)
        p3.grad = torch.tensor(5.0, dtype=torch.float64, device=dev)
    elif rank == 1:
        p0.grad = p1.grad = p2.grad = p3.grad = None
    else:
        p0.grad = torch.full((3,), 0.5, dtype=torch.float64, device=dev)
        p1.grad = torch.full((2, 2), 1.0, dtype=torch.float64, device=dev)
        p2.grad = None
        p3.grad = torch.tensor(2.0, dtype=torch.float64, device=dev)
    reducer.reduce()
    assert p0.grad is not None and p1.grad is not None and p2.grad is not None and p3.grad is not None
    r4 = {"p0": p0.grad.tolist(), "p1": p1.grad.tolist(), "p2": p2.grad.tolist(), "p3": p3.grad.item()}

    # AdamW weight-decay parity: all-None param skips decay; present all-zero param decays identically across ranks.
    p_unused = nn.Parameter(torch.tensor([2.0, -4.0], dtype=torch.float64, device=dev))
    p_zero = nn.Parameter(torch.tensor([2.0, -4.0], dtype=torch.float64, device=dev))
    wd_reducer = GradientReducer(ctx, [p_unused, p_zero])
    opt = torch.optim.AdamW([p_unused, p_zero], lr=0.1, weight_decay=0.5)
    opt.zero_grad(set_to_none=True)
    p_unused.grad = None
    p_zero.grad = torch.zeros_like(p_zero) if rank == 0 else None
    wd_reducer.reduce()
    assert p_unused.grad is None and p_zero.grad is not None
    opt.step()
    return {
        "round1": r1,
        "round2": r2,
        "round3": r3,
        "round4": r4,
        "wd_unused": p_unused.detach().tolist(),
        "wd_zero": p_zero.detach().tolist(),
    }


def _sc_status_broadcast(rank: int, world_size: int) -> list[tuple[ControlStatus, float]]:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    results: list[tuple[ControlStatus, float]] = []
    cases = [
        (ControlStatus.CONTINUE, 0.0),
        (ControlStatus.STOP, 12.5),
        (ControlStatus.FAILED, -3.25),
        (ControlStatus.STOP, float("nan")),
    ]
    for idx, (status, value) in enumerate(cases):
        if rank == 0:
            results.append(broadcast_status(ctx, status, value))
        elif idx % 2 == 0:
            results.append(broadcast_status(ctx))
        else:
            # Conflicting values on non-zero ranks must be overwritten by rank 0's payload.
            results.append(broadcast_status(ctx, ControlStatus.FAILED, 999.0))
    return results


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_primitives_multiprocess(world_size: int) -> None:
    scenarios = {
        "snapshot": _sc_snapshot_and_device_guards,
        "sum_broadcast": _sc_sum_and_broadcast_roundtrips,
        "fail_nonzero": _sc_failure_nonzero_rank,
        "fail_rank0": _sc_failure_rank_zero,
        "fail_two": _sc_failure_two_ranks,
        "fail_none": _sc_failure_none,
        "fail_uncaught": _sc_failure_uncaught_propagates,
        "consistency_match": _sc_consistency_match,
        "consistency_mismatch_single": _sc_consistency_mismatch_single,
        "consistency_mismatch_two": _sc_consistency_mismatch_two,
        "module_broadcast": _sc_module_state_broadcast,
        "grad_reducer": _sc_gradient_reducer,
        "status_broadcast": _sc_status_broadcast,
    }
    outcomes = run_scenarios(
        scenarios,
        world_size=world_size,
        pg_timeout=scale_timeout(5.0, 30.0),
        wall_timeout=scale_timeout(30.0, 120.0),
    )

    # 1. Snapshot and device guards.
    snap = unwrap_results(outcomes["snapshot"])
    assert snap == [{"rank": r, "world_size": world_size} for r in range(world_size)]

    # 2. SUM and broadcast round-trips.
    sb = unwrap_results(outcomes["sum_broadcast"])
    expected_s1 = float(sum(r + 1 for r in range(world_size)))
    expected_s2 = float(sum((r + 1) ** 2 for r in range(world_size)))
    expected_i64 = sum(r + 10 for r in range(world_size))
    for r_out in sb:
        assert r_out["sum_f64"] == [expected_s1, expected_s2]
        assert r_out["sum_i64"] == [expected_i64]
        assert r_out["sum_nc"] == [[expected_s1] * 3, [expected_s1] * 3]
        assert r_out["b_i64"] == [100, 200]
        assert r_out["b_nc"] == [[5.0] * 3, [5.0] * 3]
        assert r_out["param"] == [[1.0, 1.0], [1.0, 1.0]]

    # 3. Failure propagation: non-zero rank fails.
    fnz = unwrap_results(outcomes["fail_nonzero"])
    for r, res in enumerate(fnz):
        assert res["raised"] is True
        assert res["followup_sum"] == world_size
        if r == 1:
            assert res["exc_type"] is KeyError
            assert "missing_col" in res["message"]
        else:
            assert res["exc_type"] is RankFailureError
            assert res["failing_ranks"] == (1,)
            assert res["stage"] == "data_load"
            assert "data_load" in res["message"] and "1" in res["message"]

    # 4. Failure propagation: rank 0 fails.
    fr0 = unwrap_results(outcomes["fail_rank0"])
    for r, res in enumerate(fr0):
        assert res["raised"] is True
        assert res["followup_sum"] == world_size
        if r == 0:
            assert res["exc_type"] is ValueError
            assert "bad_rank0_cfg" in res["message"]
        else:
            assert res["exc_type"] is RankFailureError
            assert res["failing_ranks"] == (0,)
            assert res["stage"] == "startup"
            assert "startup" in res["message"] and "0" in res["message"]

    # 5. Failure propagation: two ranks fail.
    f2 = unwrap_results(outcomes["fail_two"])
    failing_pair = (0, 1) if world_size == 2 else (1, 3)
    for r, res in enumerate(f2):
        assert res["raised"] is True
        assert res["followup_sum"] == world_size
        if r == failing_pair[0]:
            assert res["exc_type"] is ValueError
            assert "first_bad" in res["message"]
        elif r == failing_pair[1]:
            assert res["exc_type"] is KeyError
            assert "second_bad" in res["message"]
        else:
            assert res["exc_type"] is RankFailureError
            assert res["failing_ranks"] == failing_pair
            assert res["stage"] == "train_step"
            assert "train_step" in res["message"]
            assert str(failing_pair[0]) in res["message"] and str(failing_pair[1]) in res["message"]

    # 6. Failure propagation: nobody fails.
    fnone = unwrap_results(outcomes["fail_none"])
    assert all(res == {"raised": False, "followup_sum": world_size} for res in fnone)

    # 6b. Failure propagation: uncaught exceptions returned via harness RankResult.error (incl. RankFailureError).
    func = outcomes["fail_uncaught"]
    for r, rank_res in enumerate(func):
        assert rank_res.error is not None
        if r == 1:
            assert rank_res.error.exc_type is ValueError
            assert "uncaught_rank1" in rank_res.error.message
        else:
            assert rank_res.error.exc_type is RankFailureError
            assert isinstance(rank_res.error.exception, RankFailureError)
            assert rank_res.error.exception.failing_ranks == (1,)
            assert rank_res.error.exception.stage == "uncaught_stage"

    # 7. Consistency check: matching digests.
    assert unwrap_results(outcomes["consistency_match"]) == [True] * world_size

    # 8. Consistency check: single mismatching rank (with different byte length).
    cm1 = unwrap_results(outcomes["consistency_mismatch_single"])
    for r, res in enumerate(cm1):
        assert res["raised"] is True
        assert res["followup_sum"] == world_size
        if r == world_size - 1:
            assert res["exc_type"] is ValueError
            assert "model_schema" in res["message"]
        else:
            assert res["exc_type"] is RankFailureError
            assert res["failing_ranks"] == (world_size - 1,)
            assert res["stage"] == "model_schema"

    # 9. Consistency check: two mismatching ranks (at N=4; 1 at N=2).
    cm2 = unwrap_results(outcomes["consistency_mismatch_two"])
    expected_bad = (1,) if world_size == 2 else (1, 2)
    for r, res in enumerate(cm2):
        assert res["raised"] is True
        assert res["followup_sum"] == world_size
        if r in expected_bad:
            assert res["exc_type"] is ValueError
        else:
            assert res["exc_type"] is RankFailureError
            assert res["failing_ranks"] == expected_bad
            assert res["stage"] == "weights_digest"

    # 10. Module state broadcast.
    ref_mod = _ToyModule(0)
    mb = unwrap_results(outcomes["module_broadcast"])
    for res in mb:
        assert res["weight"] == ref_mod.weight.detach().tolist()
        assert res["weight_req_grad"] is True
        assert res["frozen"] == ref_mod.frozen.detach().tolist()
        assert res["frozen_req_grad"] is False
        assert res["scale_buf"] == ref_mod.scale_buf.tolist()
        assert res["scale_buf_req_grad"] is False
        assert res["step_buf"] == ref_mod.step_buf.tolist()
        assert res["non_contig_buf"] == ref_mod.non_contig_buf.tolist()
        assert res["non_contig_is_contig"] is False

    # 11. Gradient reducer (three rounds against manual SUM, exact None vs present-zero, and AdamW weight decay).
    gr = unwrap_results(outcomes["grad_reducer"])
    exp_r1_p0 = [
        sum(r + 1.0 for r in range(world_size)),
        sum((r + 1.0) * 2.0 for r in range(world_size)),
        sum(-float(r) for r in range(world_size)),
    ]
    exp_r1_p1_val = sum(float(r + 3) for r in range(1, world_size))
    exp_r1_p1 = [[exp_r1_p1_val, exp_r1_p1_val], [exp_r1_p1_val, exp_r1_p1_val]]
    exp_r1_p3 = sum(float(r + 2) for r in range(world_size))

    exp_r2_p0_val = sum(0.5 * (r + 1) for r in range(world_size - 1))
    exp_r2_p0 = [exp_r2_p0_val] * 3
    scale_r2 = sum(r + 1 for r in range(world_size))
    exp_r2_p1 = [[0.0, 1.0 * scale_r2], [2.0 * scale_r2, 3.0 * scale_r2]]
    exp_r2_p2 = [0.0, 0.0, 0.0, 0.0]  # rank 0 contributed 0.0 tensor, others None
    exp_r2_p3 = -7.5

    ref_unused = nn.Parameter(torch.tensor([2.0, -4.0], dtype=torch.float64))
    ref_zero = nn.Parameter(torch.tensor([2.0, -4.0], dtype=torch.float64))
    ref_opt = torch.optim.AdamW([ref_unused, ref_zero], lr=0.1, weight_decay=0.5)
    ref_unused.grad = None
    ref_zero.grad = torch.zeros_like(ref_zero)
    ref_opt.step()

    for res in gr:
        assert res["round1"]["p0"] == exp_r1_p0
        assert res["round1"]["p1"] == exp_r1_p1
        assert res["round1"]["p2"] is None
        assert res["round1"]["p3"] == exp_r1_p3
        assert res["round2"]["p0"] == exp_r2_p0
        assert res["round2"]["p1"] == exp_r2_p1
        assert res["round2"]["p2"] == exp_r2_p2
        assert res["round2"]["p3"] == exp_r2_p3
        assert res["round3"]["p0"] is None
        assert res["round3"]["p1"] == [[2.5, 2.5], [2.5, 2.5]]
        assert res["round3"]["p2"] is None
        assert res["round3"]["p3"] == 0.0
        extra_ranks = max(0, world_size - 2)
        assert res["round4"]["p0"] == [1.0 + 0.5 * extra_ranks, 2.0 + 0.5 * extra_ranks, 3.0 + 0.5 * extra_ranks]
        assert res["round4"]["p1"] == [[4.0 + extra_ranks] * 2, [4.0 + extra_ranks] * 2]
        assert res["round4"]["p2"] == [0.0, 0.0, 0.0, 0.0]
        assert res["round4"]["p3"] == 5.0 + 2.0 * extra_ranks
        assert res["wd_unused"] == [2.0, -4.0]
        assert res["wd_zero"] == ref_zero.detach().tolist()
        assert res["wd_zero"] != [2.0, -4.0]

    # 12. Status broadcast.
    sb_status = unwrap_results(outcomes["status_broadcast"])
    for res in sb_status:
        assert res[0] == (ControlStatus.CONTINUE, 0.0)
        assert res[1] == (ControlStatus.STOP, 12.5)
        assert res[2] == (ControlStatus.FAILED, -3.25)
        assert res[3][0] is ControlStatus.STOP and math.isnan(res[3][1])


# --------------------------------------------------------------------------- dedicated multi-process edge tests


def _worker_stale_group_after_destroy(rank: int, world_size: int) -> dict[str, bool]:
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    barrier(ctx.device)
    dist.destroy_process_group()
    check_raised = False
    try:
        ctx.check_active()
    except RuntimeError:
        check_raised = True
    collective_raised = False
    try:
        ctx.all_reduce_sum(torch.ones(1, device=ctx.device))
    except RuntimeError:
        collective_raised = True

    size_changed_raised = False
    with tempfile.TemporaryDirectory(prefix="timestral_ws1_") as tmpdir:
        store = os.path.join(tmpdir, "store")
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
        try:
            ctx.check_active()
        except RuntimeError as exc:
            size_changed_raised = "world size changed" in str(exc)
        finally:
            dist.destroy_process_group()

    return {
        "check_raised": check_raised,
        "collective_raised": collective_raised,
        "size_changed_raised": size_changed_raised,
    }


def test_stale_group_raises_after_destroy_in_real_group() -> None:
    results = unwrap_results(
        run_distributed(
            _worker_stale_group_after_destroy,
            world_size=2,
            pg_timeout=scale_timeout(5.0, 30.0),
            wall_timeout=scale_timeout(30.0, 90.0),
        )
    )
    assert results == [{"check_raised": True, "collective_raised": True, "size_changed_raised": True}] * 2


def _worker_reject_cpu_gloo_backend(rank: int, world_size: int) -> None:
    DistributedContext.detect()


def test_device_qualified_cpu_gloo_backend_rejected_in_real_group() -> None:
    results = run_distributed(_worker_reject_cpu_gloo_backend, world_size=2, backend="cpu:gloo", pg_timeout=5.0)
    for res in results:
        assert res.error is not None
        assert res.error.exc_type is ValueError
        assert "cpu:gloo" in res.error.message


def _worker_unbalanced_hang(rank: int, world_size: int, ready_dir: str) -> str:
    # Bind device and synchronise once after startup, then signal readiness via file before blocking.
    ctx = DistributedContext.detect(trainer_device_for_mode())
    assert ctx is not None
    barrier(ctx.device)
    Path(ready_dir, f"rank_{rank}.ready").write_text("ready", encoding="utf-8")
    if rank == 0:
        # Rank 0 blocks in a collective that rank 1 will never enter.
        dist.all_reduce(torch.ones(1, device=ctx.device), op=dist.ReduceOp.SUM)
        if ctx.device.type == "cuda":
            # NCCL collectives enqueue asynchronously on the CUDA stream; wait on the stream so
            # rank 0 blocks inside this scenario rather than returning to teardown cleanup.
            torch.cuda.synchronize(ctx.device)
    else:
        # Rank 1 sleeps so neither rank destroys the group early; both remain blocked until killed.
        time.sleep(60.0)
    return "unreachable"


def test_harness_kills_blocked_group_quickly_and_reports_unfinished_ranks(tmp_path: Path) -> None:
    t0 = time.monotonic()
    with pytest.raises(DistributedTimeoutError) as exc_info:
        run_distributed(
            _worker_unbalanced_hang,
            world_size=2,
            args=(str(tmp_path),),
            pg_timeout=scale_timeout(30.0, 90.0),
            wall_timeout=scale_timeout(6.0, 25.0),
        )
    elapsed = time.monotonic() - t0
    max_elapsed = scale_timeout(20.0, 45.0)
    assert elapsed < max_elapsed, f"harness took too long to kill blocked group: {elapsed:.2f}s"
    assert (tmp_path / "rank_0.ready").is_file() and (tmp_path / "rank_1.ready").is_file(), (
        "workers did not finish start-up and reach the blocking point before wall_timeout"
    )
    msg = str(exc_info.value)
    assert "unfinished ranks" in msg
    assert "0" in msg and "1" in msg


def _worker_only_rank1_blocks(rank: int, world_size: int, ready_dir: str) -> str:
    Path(ready_dir, f"rank_{rank}.ready").write_text("ready", encoding="utf-8")
    if rank == 1:
        time.sleep(60.0)
    return "finished"


def test_harness_reports_only_unfinished_rank_on_timeout(tmp_path: Path) -> None:
    with pytest.raises(DistributedTimeoutError) as exc_info:
        run_distributed(
            _worker_only_rank1_blocks,
            world_size=2,
            args=(str(tmp_path),),
            pg_timeout=scale_timeout(30.0, 90.0),
            wall_timeout=scale_timeout(6.0, 25.0),
        )
    assert (tmp_path / "rank_0.ready").is_file() and (tmp_path / "rank_1.ready").is_file(), (
        "workers did not finish start-up before wall_timeout"
    )
    assert "unfinished ranks: [1]" in str(exc_info.value)


def _worker_rank1_killed_by_signal(rank: int, world_size: int) -> str:
    if rank == 1:
        os.kill(os.getpid(), signal.SIGKILL)
    return "ok"


def test_harness_reports_worker_terminated_by_signal() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        run_distributed(
            _worker_rank1_killed_by_signal,
            world_size=2,
            pg_timeout=scale_timeout(5.0, 30.0),
            wall_timeout=scale_timeout(10.0, 30.0),
        )
    msg = str(exc_info.value)
    assert "rank 1" in msg
    assert "SIGKILL" in msg


def _worker_rank1_raises_without_killing_rank0(rank: int, world_size: int) -> str:
    if rank == 1:
        raise ValueError("rank1_immediate_error")
    # Rank 0 takes a moment before finishing; it must NOT be killed by rank 1 raising.
    time.sleep(0.2)
    return "rank0_completed"


def test_harness_does_not_kill_siblings_when_one_rank_raises() -> None:
    results: list[RankResult] = run_distributed(
        _worker_rank1_raises_without_killing_rank0,
        world_size=2,
        pg_timeout=scale_timeout(5.0, 30.0),
        wall_timeout=scale_timeout(15.0, 45.0),
    )
    assert results[0].ok and results[0].value == "rank0_completed"
    assert results[1].error is not None
    assert results[1].error.exc_type is ValueError
    assert "rank1_immediate_error" in results[1].error.message
    assert "ValueError" in results[1].error.traceback


def test_f32_tol_scale_env_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """`TIMESTRAL_TEST_F32_TOL_SCALE` defaults to 1.0, accepts finite positive floats, and rejects garbage."""
    monkeypatch.delenv(F32_TOL_SCALE_ENV, raising=False)
    assert f32_tol_scale() == 1.0

    for raw, expected in (("0.1", 0.1), ("0.2", 0.2), ("1.0", 1.0), (" 2.5 ", 2.5)):
        monkeypatch.setenv(F32_TOL_SCALE_ENV, raw)
        assert f32_tol_scale() == pytest.approx(expected)

    for bad in ("", "   ", "abc", "0", "-0.1", "nan", "inf", "-inf", "1e400", "1e-400"):
        monkeypatch.setenv(F32_TOL_SCALE_ENV, bad)
        with pytest.raises(ValueError, match=F32_TOL_SCALE_ENV):
            f32_tol_scale()


def test_public_all_and_docstrings_complete() -> None:
    """Every public name in `__all__` and every public member exists and is documented."""
    assert len(dist_mod.__all__) == len(set(dist_mod.__all__))
    for name in dist_mod.__all__:
        obj = getattr(dist_mod, name)
        assert obj.__doc__, f"{name} missing docstring"
    for cls in (DistributedContext, GradientReducer, DistributedBatchSampler, DistributedBatchCycle):
        for attr_name, attr in vars(cls).items():
            if attr_name.startswith("_"):
                continue
            target = attr.fget if isinstance(attr, property) else attr
            if isinstance(target, (classmethod, staticmethod)):
                target = target.__func__
            assert target is not None and target.__doc__, f"{cls.__name__}.{attr_name} missing docstring"
