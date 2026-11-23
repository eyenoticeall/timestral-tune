"""Multi-process primitives for single-node data-parallel training.

This module is the only place in the package that imports `torch.distributed`. It provides a
minimal, lockstep-friendly surface for the trainer:

- Process-group detection, backend/device validation, and stale-group checks (`DistributedContext`).
- Fixed-size tensor collectives built exclusively on `ReduceOp.SUM` and rank-0 broadcasts
  (`DistributedContext.all_reduce_sum`, `DistributedContext.broadcast`,
  `DistributedContext.propagate_failure`, `DistributedContext.check_consistency`,
  `broadcast_module_state`, `GradientReducer`, `ControlStatus`, `broadcast_status`).
- Failure classification (`RankFailureError`, `is_oom_error`).
- Deterministic epoch sharding, validation sharding, and rank-offset seeding (`steps_per_epoch`,
  `rank_slice_size`, `rank_step_indices`, `DistributedBatchSampler`, `validation_shard_indices`,
  `rank_seed`).

Every communicating entry point (`DistributedContext` methods, `broadcast_module_state`,
`broadcast_status`, and `GradientReducer.reduce`) documents which ranks must call it and how many
collectives it executes so the integrator can keep all ranks strictly in lockstep. Everything in
the sampling and seeding section (`steps_per_epoch`, `rank_slice_size`, `rank_step_indices`,
`DistributedBatchSampler`, `validation_shard_indices`, `rank_seed`), along with `is_oom_error`,
`RankFailureError`, `ControlStatus`, and read-only properties, is pure and rank-local (0
collectives).

- Spec: `docs/SPEC.md` §5.6, D22–D35
"""

from __future__ import annotations

import hashlib
import math
import operator
import os
import struct
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.data import Sampler

__all__ = [
    "ControlStatus",
    "DistributedBatchSampler",
    "DistributedContext",
    "GradientReducer",
    "RankFailureError",
    "broadcast_module_state",
    "broadcast_status",
    "is_oom_error",
    "rank_seed",
    "rank_slice_size",
    "rank_step_indices",
    "steps_per_epoch",
    "validation_shard_indices",
]

BackendName = Literal["gloo", "nccl"]

_RANK_SEED_STRIDE = 1_000_000_007
_SEED_MODULUS = 1 << 63
_DIGEST_WORDS = 4


def _build_oom_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = []
    for candidate in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)):
        if isinstance(candidate, type) and issubclass(candidate, BaseException) and candidate not in types:
            types.append(candidate)
    return tuple(types)


_OOM_TYPES: tuple[type[BaseException], ...] = _build_oom_types()


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, not bool")
    try:
        parsed = operator.index(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {parsed}")
    return parsed


def _require_stage(stage: str) -> str:
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a non-empty string")
    return stage


def _encode_digest(digest: bytes) -> tuple[int, int, int, int]:
    """Normalise an arbitrary `bytes` digest via SHA-256 into four signed int64 words."""
    if not isinstance(digest, bytes):
        raise TypeError(f"digest must be bytes, got {type(digest).__name__}")
    raw = hashlib.sha256(digest).digest()
    w0, w1, w2, w3 = struct.unpack("<4q", raw)
    return int(w0), int(w1), int(w2), int(w3)


def _resolve_rank_device(
    backend: BackendName,
    rank: int,
    requested: str | torch.device | None,
) -> torch.device:
    """Resolve and validate the rank's device according to the active backend.

    Raises `ValueError` immediately on any backend/device conflict or invalid CUDA index.
    Never selects MPS.
    """
    if backend == "gloo":
        if requested is None or requested == "auto" or requested == "cpu":
            return torch.device("cpu")
        if isinstance(requested, torch.device) and requested.type == "cpu" and requested.index is None:
            return torch.device("cpu")
        raise ValueError(f"backend 'gloo' requires device 'auto', None, or 'cpu'; got {requested!r}")

    explicit_index: int | None
    if requested is None or requested == "auto":
        explicit_index = None
    elif isinstance(requested, (str, torch.device)):
        if isinstance(requested, torch.device):
            parsed = requested
        else:
            try:
                parsed = torch.device(requested)
            except RuntimeError as exc:
                raise ValueError(f"invalid device {requested!r}") from exc
        if parsed.type != "cuda":
            raise ValueError(f"backend 'nccl' requires a CUDA device (or 'auto'/None); got {requested!r}")
        explicit_index = parsed.index
    else:
        raise ValueError(f"backend 'nccl' requires a CUDA device (or 'auto'/None); got {requested!r}")

    if explicit_index is not None:
        index = explicit_index
    else:
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is None:
            index = rank
        else:
            try:
                index = int(local_rank_env)
            except ValueError as exc:
                raise ValueError(
                    f"LOCAL_RANK environment variable must parse as an integer, got {local_rank_env!r}"
                ) from exc

    device_count = torch.cuda.device_count()
    if index < 0 or index >= device_count:
        raise ValueError(f"CUDA device index {index} out of range for {device_count} available CUDA device(s)")

    torch.cuda.set_device(index)
    return torch.device("cuda", index)


# --------------------------------------------------------------------------- errors & predicates


def is_oom_error(exc: BaseException) -> bool:
    """Return True if `exc` is a PyTorch out-of-memory error that must never be propagated.

    Recognises `torch.cuda.OutOfMemoryError` and `torch.OutOfMemoryError` (where available in
    the installed PyTorch). Out-of-memory errors leave device contexts unreliable for collectives
    and must be re-raised immediately rather than routed through `propagate_failure`.
    """
    return isinstance(exc, _OOM_TYPES)


class RankFailureError(RuntimeError):
    """Raised on healthy ranks when one or more peer ranks fail in a coordinated stage.

    Attributes:
        failing_ranks: Sorted tuple of unique rank indices that reported a failure.
        stage: Non-empty label identifying the stage where the failure occurred.
    """

    def __init__(self, failing_ranks: Sequence[int], stage: str) -> None:
        ranks = tuple(sorted({_require_int("rank", r, minimum=0) for r in failing_ranks}))
        if not ranks:
            raise ValueError("failing_ranks must not be empty")
        stage = _require_stage(stage)
        self.failing_ranks: tuple[int, ...] = ranks
        self.stage: str = stage
        ranks_text = ", ".join(str(r) for r in ranks)
        noun = "rank" if len(ranks) == 1 else "ranks"
        super().__init__(f"distributed failure during {stage!r} on {noun} [{ranks_text}]")

    def __reduce__(self) -> tuple[type[RankFailureError], tuple[tuple[int, ...], str]]:
        return (type(self), (self.failing_ranks, self.stage))


class ControlStatus(IntEnum):
    """Rank-0 loop control status carried by `broadcast_status`.

    Members:
        CONTINUE: All ranks continue to the next step or epoch.
        STOP: All ranks stop cleanly (for example, early stopping or a callback stop request).
        FAILED: Rank 0 encountered an error (for example, in a callback) and all ranks must abort.
    """

    CONTINUE = 0
    STOP = 1
    FAILED = 2


# --------------------------------------------------------------------------- process-group snapshot


@dataclass(frozen=True)
class DistributedContext:
    """Immutable snapshot of an active multi-process `torch.distributed` group and rank device.

    Created once via `DistributedContext.detect` after the caller initialises the process group.
    The library never creates or destroys process groups. All collective methods on this object
    use only `ReduceOp.SUM` reductions and rank-0 broadcasts over fixed-size tensors, and verify
    that the process group is still alive and unchanged before executing.

    Attributes:
        rank: This process's rank (`0 <= rank < world_size`).
        world_size: Number of processes in the group (`world_size > 1`).
        backend: Active backend name, strictly `"gloo"` or `"nccl"`.
        device: Validated device assigned to this rank (`cpu` for `"gloo"`, `cuda:<index>` for
            `"nccl"`).

    Ranks: captured independently on every rank via `detect()`; collective methods require all
        ranks in lockstep as documented per method.
    Collectives: 0 to construct or inspect attributes; see individual methods for collective counts.
    """

    rank: int
    world_size: int
    backend: BackendName
    device: torch.device

    def __post_init__(self) -> None:
        _require_int("world_size", self.world_size, minimum=2)
        _require_int("rank", self.rank, minimum=0, maximum=self.world_size - 1)
        if not isinstance(self.backend, str) or self.backend not in ("gloo", "nccl"):
            raise ValueError(f"unsupported backend {self.backend!r}; expected exactly 'gloo' or 'nccl'")
        if not isinstance(self.device, torch.device):
            raise TypeError(f"device must be a torch.device, got {type(self.device).__name__}")

    @property
    def is_rank_zero(self) -> bool:
        """Whether this process is rank 0."""
        return self.rank == 0

    @classmethod
    def detect(cls, device: str | torch.device | None = "auto") -> DistributedContext | None:
        """Capture a snapshot if a multi-process group (`world_size > 1`) is active, else `None`.

        Returns `None` when `torch.distributed` is unavailable, uninitialised, or has
        `world_size <= 1`. When active:

        - Checks `torch.distributed.get_backend()` by exact string equality against `"gloo"` and
          `"nccl"`, raising `ValueError` for anything else (including `"cpu:gloo"`).
        - Resolves the rank's device (`gloo` -> CPU, rejecting anything other than `"auto"`,
          `None`, or `"cpu"`; `nccl` -> CUDA, honouring an explicit CUDA index, else integer
          `LOCAL_RANK`, else `rank`, validating `0 <= index < torch.cuda.device_count()`, and
          calling `torch.cuda.set_device`). MPS is never selected.
        - Errors raise immediately and are not routed through any collective.

        Ranks: any rank (or single process); every rank in a multi-process run should call it once
            at startup.
        Collectives: 0.
        """
        if not dist.is_available() or not dist.is_initialized():
            return None
        world_size = int(dist.get_world_size())
        if world_size <= 1:
            return None
        raw_backend = dist.get_backend()
        if isinstance(raw_backend, str) and raw_backend == "gloo":
            backend: BackendName = "gloo"
        elif isinstance(raw_backend, str) and raw_backend == "nccl":
            backend = "nccl"
        else:
            raise ValueError(
                f"unsupported torch.distributed backend {raw_backend!r}; expected exactly 'gloo' or 'nccl'"
            )
        rank = int(dist.get_rank())
        resolved_device = _resolve_rank_device(backend, rank, device)
        return cls(rank=rank, world_size=world_size, backend=backend, device=resolved_device)

    def check_active(self) -> None:
        """Verify that the process group is still initialised and matches this snapshot.

        Raises `RuntimeError` if the group has been destroyed or if its world size, rank, or
        backend has changed since `detect()`.

        Ranks: any rank (local check).
        Collectives: 0.
        """
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "the torch.distributed process group has been destroyed since this DistributedContext was created"
            )
        current_world_size = int(dist.get_world_size())
        if current_world_size != self.world_size:
            raise RuntimeError(f"torch.distributed world size changed from {self.world_size} to {current_world_size}")
        current_rank = int(dist.get_rank())
        if current_rank != self.rank:
            raise RuntimeError(f"torch.distributed rank changed from {self.rank} to {current_rank}")
        current_backend = dist.get_backend()
        if current_backend != self.backend:
            raise RuntimeError(f"torch.distributed backend changed from {self.backend!r} to {current_backend!r}")

    def all_reduce_sum(self, tensor: Tensor) -> Tensor:
        """In-place `ReduceOp.SUM` all-reduce of `tensor` across all ranks; returns `tensor`.

        `tensor` must reside on `self.device` and have `requires_grad=False`. Non-contiguous
        tensors are staged through a contiguous copy and written back in place.

        Ranks: all ranks must call in lockstep with matching tensor shape and dtype.
        Collectives: 1 (`all_reduce(SUM)`).
        """
        self.check_active()
        if tensor.device != self.device:
            raise ValueError(f"tensor resides on {tensor.device}, expected {self.device}")
        if tensor.requires_grad:
            raise ValueError("all_reduce_sum does not accept tensors with requires_grad=True")
        if tensor.is_contiguous():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        else:
            staged = tensor.contiguous()
            dist.all_reduce(staged, op=dist.ReduceOp.SUM)
            tensor.copy_(staged)
        return tensor

    def broadcast(self, tensor: Tensor) -> Tensor:
        """In-place broadcast of `tensor` from rank 0 to all ranks under `no_grad`; returns `tensor`.

        `tensor` must reside on `self.device`. `tensor.requires_grad` is preserved unchanged.
        Non-contiguous tensors are staged through a contiguous copy and written back in place.

        Ranks: all ranks must call in lockstep with matching tensor shape and dtype.
        Collectives: 1 (`broadcast(src=0)`).
        """
        self.check_active()
        if tensor.device != self.device:
            raise ValueError(f"tensor resides on {tensor.device}, expected {self.device}")
        with torch.no_grad():
            if tensor.is_contiguous():
                dist.broadcast(tensor.detach(), src=0)
            else:
                staged = tensor.detach().contiguous()
                dist.broadcast(staged, src=0)
                tensor.copy_(staged)
        return tensor

    def propagate_failure(self, local_error: Exception | None, *, stage: str) -> None:
        """Coordinate exception reporting across ranks so every rank raises if any rank failed.

        Executes an `all_reduce(SUM)` over an `int64` vector of length `world_size` with `1` at
        `self.rank` when `local_error` is not `None`. If any rank failed:

        - Failing ranks re-raise their original `local_error`.
        - All other ranks raise `RankFailureError(failing_ranks, stage=stage)`.

        The collective completes before any rank raises, leaving the process group clean for
        follow-up collectives if callers handle the exception.

        Only `Exception` instances (excluding PyTorch out-of-memory errors recognised by
        `is_oom_error`) may be propagated; passing a non-`Exception` `BaseException` or an OOM
        error re-raises it immediately without entering the collective.

        In a training step, callers typically carry a 0/1 error flag inside the step's reduced
        statistics vector on the fast path and invoke this helper only when the reduced flag is
        non-zero.

        Ranks: all ranks must call in lockstep.
        Collectives: 1 (`all_reduce(SUM)` of an `int64` vector of length `world_size`).
        """
        stage = _require_stage(stage)
        if local_error is not None:
            if not isinstance(local_error, Exception) or is_oom_error(local_error):
                raise local_error
        self.check_active()
        flags = torch.zeros(self.world_size, dtype=torch.int64, device=self.device)
        if local_error is not None:
            flags[self.rank] = 1
        dist.all_reduce(flags, op=dist.ReduceOp.SUM)
        failing_ranks = tuple(i for i, v in enumerate(flags.tolist()) if v != 0)
        if local_error is not None:
            raise local_error
        if failing_ranks:
            raise RankFailureError(failing_ranks, stage=stage)

    def check_consistency(self, digest: bytes, *, stage: str = "consistency_check") -> None:
        """Verify that every rank computed the same `bytes` digest as rank 0.

        Each rank hashes `digest` with SHA-256 and unpacks the 32 bytes into four signed `int64`
        integers, so arbitrarily sized caller digests always map to a fixed-size `(4,)` `int64`
        tensor. Rank 0 broadcasts its four integers, each rank compares against its own, and
        `propagate_failure` runs so mismatched ranks raise `ValueError` and all other ranks raise
        `RankFailureError`.

        Ranks: all ranks must call in lockstep.
        Collectives: 2 (1 `broadcast(src=0)` of 4 `int64`s + 1 `all_reduce(SUM)` via
            `propagate_failure`).
        """
        stage = _require_stage(stage)
        local_words = _encode_digest(digest)
        self.check_active()
        local_tensor = torch.tensor(local_words, dtype=torch.int64, device=self.device)
        ref_tensor = (
            local_tensor.clone()
            if self.rank == 0
            else torch.empty(_DIGEST_WORDS, dtype=torch.int64, device=self.device)
        )
        dist.broadcast(ref_tensor, src=0)
        local_error: Exception | None = None
        if not torch.equal(local_tensor, ref_tensor):
            local_error = ValueError(f"rank {self.rank} digest mismatch during {stage!r} (differs from rank 0)")
        self.propagate_failure(local_error, stage=stage)


# --------------------------------------------------------------------------- higher-level collectives


def broadcast_module_state(ctx: DistributedContext, module: nn.Module) -> None:
    """Broadcast every unique parameter and buffer of `module` from rank 0 under `no_grad`.

    Tensors are visited in deterministic sorted-name order: all unique `module.named_parameters()`
    first, followed by all unique `module.named_buffers()`. Every tensor's `requires_grad` flag is
    preserved unchanged. Non-contiguous tensors or tensors stored on a different device than
    `ctx.device` are staged through `ctx.device` and copied back in place.

    A parameter or buffer name, shape, or dtype mismatch across ranks is a caller bug that
    `ctx.check_consistency` exists to catch before calling this function.

    Ranks: all ranks must call in lockstep on structurally identical modules.
    Collectives: one broadcast from rank 0 per unique parameter tensor and one per unique buffer
        tensor (the same number on every rank provided the modules are structurally identical).
    """
    ctx.check_active()
    params = [param for _, param in sorted(module.named_parameters(), key=lambda item: item[0])]
    buffers = [buf for _, buf in sorted(module.named_buffers(), key=lambda item: item[0])]
    with torch.no_grad():
        for tensor in (*params, *buffers):
            if tensor.device == ctx.device and tensor.is_contiguous():
                dist.broadcast(tensor.detach(), src=0)
            else:
                staged = tensor.detach().to(device=ctx.device).contiguous()
                dist.broadcast(staged, src=0)
                tensor.copy_(staged)


def broadcast_status(
    ctx: DistributedContext,
    status: ControlStatus = ControlStatus.CONTINUE,
    value: float = 0.0,
) -> tuple[ControlStatus, float]:
    """Broadcast a `(ControlStatus, float)` pair from rank 0 as a 2-element `float64` tensor.

    Rank 0 supplies the authoritative `(status, value)`. Non-zero ranks may omit them (defaults
    are ignored and overwritten by rank 0's payload). Returns the decoded `(status, value)` on
    every rank.

    Ranks: all ranks must call in lockstep.
    Collectives: 1 (`broadcast(src=0)` of a 2-element `float64` tensor).
    """
    if not isinstance(status, ControlStatus):
        raise TypeError(f"status must be a ControlStatus, got {type(status).__name__}")
    ctx.check_active()
    if ctx.rank == 0:
        payload = torch.tensor([float(status.value), float(value)], dtype=torch.float64, device=ctx.device)
    else:
        payload = torch.empty(2, dtype=torch.float64, device=ctx.device)
    dist.broadcast(payload, src=0)
    raw_code, raw_value = payload.tolist()
    if not math.isfinite(raw_code) or raw_code != int(raw_code):
        raise ValueError(f"invalid ControlStatus code received from broadcast: {raw_code!r}")
    return ControlStatus(int(raw_code)), float(raw_value)


class GradientReducer:
    """Reusable flat-buffer gradient SUM-reducer over a fixed list of trainable parameters.

    Constructed once from the trainable parameter sequence, allocating a single flat buffer of the
    shared dtype and device. All parameters must be non-empty in count, distinct, share one dtype
    and one device, and reside on `ctx.device` (`ValueError` otherwise).

    Calling `reduce()` packs each parameter's `.grad` into the flat buffer (writing zeros where
    `.grad` is `None`), executes a single `ReduceOp.SUM` all-reduce with no division, and copies
    the reduced slices back into every parameter's `.grad` (allocating `.grad` where it was
    `None`).

    Ranks: constructed independently on each rank over matching parameter lists; `reduce()` must
        be called by all ranks in lockstep.
    Collectives: 0 in `__init__`; 1 (`all_reduce(SUM)`) per `reduce()` call.
    """

    def __init__(self, ctx: DistributedContext, parameters: Iterable[Tensor]) -> None:
        params = tuple(parameters)
        if not params:
            raise ValueError("GradientReducer requires at least one parameter")
        seen: set[int] = set()
        for param in params:
            if not isinstance(param, Tensor):
                raise TypeError(f"parameters must contain Tensor instances, got {type(param).__name__}")
            pid = id(param)
            if pid in seen:
                raise ValueError("GradientReducer received a duplicate parameter object")
            seen.add(pid)
        dtype = params[0].dtype
        device = params[0].device
        for param in params[1:]:
            if param.dtype != dtype:
                raise ValueError(f"all parameters must share one dtype; got {dtype} and {param.dtype}")
            if param.device != device:
                raise ValueError(f"all parameters must share one device; got {device} and {param.device}")
        if device != ctx.device:
            raise ValueError(f"parameters reside on {device}, but DistributedContext uses {ctx.device}")

        total_numel = sum(param.numel() for param in params)
        self._ctx = ctx
        self._dtype = dtype
        self._device = device
        self._buffer = torch.empty(total_numel, dtype=dtype, device=device)

        entries: list[tuple[Tensor, Tensor]] = []
        offset = 0
        for param in params:
            count = param.numel()
            entries.append((param, self._buffer.narrow(0, offset, count)))
            offset += count
        self._entries: tuple[tuple[Tensor, Tensor], ...] = tuple(entries)

    @property
    def numel(self) -> int:
        """Total number of scalar elements in the flat buffer."""
        return int(self._buffer.numel())

    def reduce(self) -> None:
        """Pack gradients, SUM-reduce the flat buffer once without division, and write `.grad` back.

        Parameters whose `.grad` is `None` contribute zeros and receive a newly allocated `.grad`
        tensor containing the reduced sum.

        Ranks: all ranks must call in lockstep on identically ordered, matching parameter lists.
        Collectives: 1 (`all_reduce(SUM)`).
        """
        self._ctx.check_active()
        with torch.no_grad():
            for param, buf_slice in self._entries:
                grad = param.grad
                if grad is None:
                    buf_slice.zero_()
                else:
                    if grad.shape != param.shape or grad.dtype != self._dtype or grad.device != self._device:
                        raise ValueError(
                            f"gradient metadata mismatch: expected shape={tuple(param.shape)} "
                            f"dtype={self._dtype} device={self._device}, got shape={tuple(grad.shape)} "
                            f"dtype={grad.dtype} device={grad.device}"
                        )
                    buf_slice.copy_(grad.detach().reshape(-1))

            dist.all_reduce(self._buffer, op=dist.ReduceOp.SUM)

            for param, buf_slice in self._entries:
                if param.grad is None:
                    param.grad = torch.empty_like(param)
                param.grad.copy_(buf_slice.view_as(param))


# --------------------------------------------------------------------------- sampling & seeding


def _epoch_seed(seed: int, epoch: int) -> int:
    """Derive a 63-bit non-negative seed from `(seed, epoch)` via SHA-256."""
    digest = hashlib.sha256(f"timestral-tune:epoch:{seed}:{epoch}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & (_SEED_MODULUS - 1)


def _epoch_ordering(num_items: int, *, shuffle: bool, seed: int, epoch: int) -> list[int]:
    if num_items == 0:
        return []
    if not shuffle:
        return list(range(num_items))
    generator = torch.Generator().manual_seed(_epoch_seed(seed, epoch))
    return torch.randperm(num_items, generator=generator).tolist()


def _rank_slice_bounds(num_items: int, world_size: int, batch_size: int, rank: int, step: int) -> tuple[int, int]:
    """Return `[start, end)` positions in the global epoch ordering for `(rank, step)`."""
    num_items = _require_int("num_items", num_items, minimum=0)
    world_size = _require_int("world_size", world_size, minimum=1)
    batch_size = _require_int("batch_size", batch_size, minimum=1)
    rank = _require_int("rank", rank, minimum=0, maximum=world_size - 1)
    total_steps = steps_per_epoch(num_items, world_size, batch_size)
    if total_steps == 0:
        raise ValueError("step is out of range when num_items == 0")
    step = _require_int("step", step, minimum=0, maximum=total_steps - 1)

    global_batch = world_size * batch_size
    step_start = step * global_batch
    remaining = min(global_batch, num_items - step_start)
    base, extra = divmod(remaining, world_size)
    length = base + (1 if rank < extra else 0)
    offset = rank * base + min(rank, extra)
    start = step_start + offset
    return start, start + length


def steps_per_epoch(num_items: int, world_size: int, batch_size: int) -> int:
    """Global optimiser-step count per epoch: `ceil(num_items / (world_size * batch_size))`.

    Returns `0` when `num_items == 0`. Identical on every rank.
    """
    num_items = _require_int("num_items", num_items, minimum=0)
    world_size = _require_int("world_size", world_size, minimum=1)
    batch_size = _require_int("batch_size", batch_size, minimum=1)
    if num_items == 0:
        return 0
    global_batch = world_size * batch_size
    return (num_items + global_batch - 1) // global_batch


def rank_slice_size(num_items: int, world_size: int, batch_size: int, rank: int, step: int) -> int:
    """Number of items assigned to `rank` at global step `step` (`0 <= step < steps_per_epoch`).

    Full steps give every rank `batch_size`. A trailing partial step with `R` items assigns
    `floor(R / world_size) + (1 if rank < R mod world_size else 0)` items, so a rank receives
    `0` items only when `R < world_size` and `rank >= R`.
    """
    start, end = _rank_slice_bounds(num_items, world_size, batch_size, rank, step)
    return end - start


def rank_step_indices(
    num_items: int,
    world_size: int,
    batch_size: int,
    rank: int,
    step: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
    epoch: int = 0,
) -> list[int]:
    """Return the item indices assigned to `rank` at global step `step` of `epoch`.

    The global epoch ordering over `0..num_items-1` is natural order when `shuffle=False`, or a
    deterministic permutation derived solely from `(seed, epoch)` via SHA-256 into a 63-bit seed
    for `torch.randperm` when `shuffle=True`. Across ranks `0..world_size-1` in rank order, the
    concatenation at step `step` is the exact contiguous global batch of size up to
    `world_size * batch_size`, with no index duplicated or dropped. May return `[]` on a trailing
    step when fewer than `world_size` items remain.
    """
    seed = _require_int("seed", seed)
    epoch = _require_int("epoch", epoch, minimum=0)
    start, end = _rank_slice_bounds(num_items, world_size, batch_size, rank, step)
    if start == end:
        return []
    ordering = _epoch_ordering(num_items, shuffle=shuffle, seed=seed, epoch=epoch)
    return ordering[start:end]


class DistributedBatchSampler(Sampler[list[int]]):
    """Main-process `DataLoader(batch_sampler=...)` yielding a rank's non-empty batches in step order.

    For each epoch, derives one global ordering of `0..num_items-1` (natural order when
    `shuffle=False`; SHA-256-seeded `torch.randperm` from `(seed, epoch)` when `shuffle=True`)
    and partitions each global step of `world_size * batch_size` positions contiguously across
    ranks, splitting any trailing partial step `R` as `floor(R / N) + (1 if r < R mod N else 0)`.
    Only non-empty slices are yielded, so `DHFDataCollator` is never called with an empty list.
    Call `set_epoch(epoch)` before iterating each epoch.

    Stores only primitive integers and booleans, so it is picklable and works with
    `DataLoader(..., num_workers > 0, persistent_workers=True)` (the sampler executes in the
    main process).
    """

    def __init__(
        self,
        num_items: int,
        *,
        world_size: int,
        rank: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        self._num_items = _require_int("num_items", num_items, minimum=0)
        self._world_size = _require_int("world_size", world_size, minimum=1)
        self._rank = _require_int("rank", rank, minimum=0, maximum=self._world_size - 1)
        self._batch_size = _require_int("batch_size", batch_size, minimum=1)
        self._shuffle = bool(shuffle)
        self._seed = _require_int("seed", seed)
        self._epoch = _require_int("epoch", epoch, minimum=0)

    @property
    def epoch(self) -> int:
        """Current epoch index."""
        return self._epoch

    @property
    def num_global_steps(self) -> int:
        """Total global steps in an epoch, including steps where this rank is empty."""
        return steps_per_epoch(self._num_items, self._world_size, self._batch_size)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch index used for subsequent iterations."""
        self._epoch = _require_int("epoch", epoch, minimum=0)

    def __len__(self) -> int:
        """Number of non-empty batches this rank yields in an epoch."""
        total_steps = self.num_global_steps
        if total_steps == 0:
            return 0
        last_size = rank_slice_size(self._num_items, self._world_size, self._batch_size, self._rank, total_steps - 1)
        return total_steps if last_size > 0 else total_steps - 1

    def __iter__(self) -> Iterator[list[int]]:
        total_steps = self.num_global_steps
        if total_steps == 0:
            return
        ordering = _epoch_ordering(self._num_items, shuffle=self._shuffle, seed=self._seed, epoch=self._epoch)
        for step in range(total_steps):
            start, end = _rank_slice_bounds(self._num_items, self._world_size, self._batch_size, self._rank, step)
            if end > start:
                yield ordering[start:end]


def validation_shard_indices(num_items: int, world_size: int, rank: int) -> list[int]:
    """Return strided validation indices `[rank, rank + world_size, ...]` below `num_items`.

    Partitions `0..num_items-1` disjointly across ranks without padding or duplication; ranks with
    `rank >= num_items` receive `[]`.
    """
    num_items = _require_int("num_items", num_items, minimum=0)
    world_size = _require_int("world_size", world_size, minimum=1)
    rank = _require_int("rank", rank, minimum=0, maximum=world_size - 1)
    return list(range(rank, num_items, world_size))


def rank_seed(base_seed: int, rank: int) -> int:
    """Derive a per-rank RNG seed from `base_seed`.

    Rank 0 keeps `base_seed` unchanged; rank `r > 0` uses `(base_seed + r * 1_000_000_007) mod 2^63`.
    """
    base_seed = _require_int("base_seed", base_seed)
    rank = _require_int("rank", rank, minimum=0)
    if rank == 0:
        return base_seed
    return (base_seed + rank * _RANK_SEED_STRIDE) % _SEED_MODULUS
