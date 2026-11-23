"""Stage 2 multi-process replay and distillation tests.

Also temporarily hosts unit tests for `DistributedBatchCycle` (from `trainer/distributed.py`),
which can be moved into `tests/test_distributed.py` once parallel test edits merge.

- Spec: `docs/SPEC.md` §5.3, §5.6, §9 (T15), D22–D35
"""

from __future__ import annotations

import sys

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from timestral_tune.trainer.distributed import (
    DistributedBatchCycle,
    DistributedBatchSampler,
    rank_step_indices,
)

# PyTorch < 2.8 on macOS hangs at process exit after DataLoader worker processes run (docs/VERIFICATION.md §10).
WORKERS_HANG_AT_EXIT = sys.platform == "darwin" and tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


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
