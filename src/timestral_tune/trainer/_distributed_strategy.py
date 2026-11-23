"""Multi-process execution strategy for `TimestralTrainer._run_fit_loop`.

Implements the five-stage distributed training step (domain loss, optional sharded replay mixing,
and frozen-base distillation), main-process epoch batch sampling and replay epoch cycling,
rank-offset stochastic streams, rank-0 callback / best-state ownership with fixed-size `float64`
control broadcasts, and internal sharded validation of spec §5.6 (all four modes, no caches).

- Spec: `docs/SPEC.md` §5.3, §5.6, §9 (T15), §11 (D22–D35)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..data.batch import TrainingBatch
from ._types import EvalResult, TrainingResult, _FitBookkeeper, _LossTerms, _PreparedReplayBatch, _StepOutcome
from .distributed import (
    ControlStatus,
    DistributedBatchCycle,
    DistributedBatchSampler,
    DistributedContext,
    GradientReducer,
    RankFailureError,
    broadcast_status,
    is_oom_error,
    rank_seed,
    validation_shard_indices,
)

if TYPE_CHECKING:
    from .trainer import TimestralTrainer

# Fixed layout of the 8-element float64 statistics all-reduce buffer (spec §5.6, D23).
_STAT_W_D = 0
_STAT_SUM_D = 1
_STAT_W_R = 2
_STAT_SUM_R = 3
_STAT_M = 4
_STAT_SUM_M = 5
_STAT_NON_FINITE = 6
_STAT_FAILED = 7
_STAT_SIZE = 8


@dataclass(frozen=True)
class _LocalStepStats:
    """One rank's local statistics before the 8-scalar float64 SUM all-reduce."""

    w_d: int = 0
    sum_d: float = 0.0
    w_r: int = 0
    sum_r: float = 0.0
    m_count: int = 0
    sum_m: float = 0.0
    non_finite: bool = False
    failed: bool = False

    def to_tensor(self, device: torch.device) -> Tensor:
        """Pack into the 8-element float64 statistics vector ordered by `_STAT_*`."""
        values = [0.0] * _STAT_SIZE
        values[_STAT_W_D] = float(self.w_d)
        values[_STAT_SUM_D] = self.sum_d
        values[_STAT_W_R] = float(self.w_r)
        values[_STAT_SUM_R] = self.sum_r
        values[_STAT_M] = float(self.m_count)
        values[_STAT_SUM_M] = self.sum_m
        values[_STAT_NON_FINITE] = 1.0 if self.non_finite else 0.0
        values[_STAT_FAILED] = 1.0 if self.failed else 0.0
        return torch.tensor(values, dtype=torch.float64, device=device)


@dataclass(frozen=True)
class _ReducedStepStats:
    """Globally summed statistics unpacked from the 8-element float64 all-reduce buffer."""

    w_d_tot: float
    sum_d_tot: float
    w_r_tot: float
    sum_r_tot: float
    m_tot: float
    sum_m_tot: float
    non_finite_tot: float
    failed_tot: float

    @classmethod
    def from_tensor(cls, stats: Tensor) -> _ReducedStepStats:
        """Unpack an all-reduced 8-element float64 statistics vector using `_STAT_*`."""
        values = stats.tolist()
        return cls(
            w_d_tot=values[_STAT_W_D],
            sum_d_tot=values[_STAT_SUM_D],
            w_r_tot=values[_STAT_W_R],
            sum_r_tot=values[_STAT_SUM_R],
            m_tot=values[_STAT_M],
            sum_m_tot=values[_STAT_SUM_M],
            non_finite_tot=values[_STAT_NON_FINITE],
            failed_tot=values[_STAT_FAILED],
        )


@dataclass(frozen=True)
class _LocalStepResult:
    """Output of the protected local stage before the statistics all-reduce."""

    stats: _LocalStepStats
    terms: _LossTerms | None = None
    error: Exception | None = None
    first_non_finite: float | None = None


@dataclass(frozen=True)
class _StepDecision:
    """Global control-flow decision and active term weights derived from reduced statistics."""

    skipped: bool
    global_loss: float = 0.0
    dom_weight: float = 1.0
    replay_fraction: float = 0.0
    replay_active: bool = False
    distill_weight: float = 0.0
    distill_active: bool = False
    w_d_tot: float = 0.0
    w_r_tot: float = 0.0
    m_tot: float = 0.0


def _term_weighted_sum(loss: Tensor | None, count: int) -> tuple[int, float, float | None, bool]:
    """Convert one loss term and its weight count into `(count, weighted_sum, raw_val, non_finite)`.

    Returns `(0, 0.0, None, False)` when `loss is None` or `count <= 0`. When `loss.item()` or
    `loss.item() * count` is non-finite, `weighted_sum` stays `0.0` (so the SUM all-reduce is
    not poisoned with `NaN`/`Inf`) and `non_finite` is `True`.
    """
    if loss is None or count <= 0:
        return 0, 0.0, None, False
    val = loss.item()
    if not math.isfinite(val):
        return count, 0.0, val, True
    weighted = val * count
    if not math.isfinite(weighted):
        return count, 0.0, val, True
    return count, weighted, val, False


class _DistributedStrategy:
    """Multi-process `_FitStrategy` driving `TimestralTrainer._run_fit_loop` across ranks."""

    def __init__(self, trainer: TimestralTrainer, ctx: DistributedContext) -> None:
        self._trainer = trainer
        self._ctx = ctx
        self._flip_generator: torch.Generator | None = None
        self._replay_flip_generator: torch.Generator | None = None
        self._sampler: DistributedBatchSampler | None = None
        self._train_loader: DataLoader | None = None
        self._replay_sampler: DistributedBatchSampler | None = None
        self._replay_loader: DataLoader | None = None
        self._replay_cycle: DistributedBatchCycle[TrainingBatch] | None = None
        self._sharded_val_loader: DataLoader | None = None
        self._val_loader_ready = False
        self._reducer: GradientReducer | None = None

    @property
    def is_primary(self) -> bool:
        """True on rank 0, which owns callbacks, best-loss tracking, and summary logging."""
        return self._ctx.is_rank_zero

    def prepare_fit(self) -> None:
        """Verify group liveness, seed rank-offset streams, and build samplers, loaders, and reducer.

        Ranks: all ranks must call in lockstep.
        Collectives: 0.
        """
        self._ctx.check_active()
        trainer = self._trainer
        config = trainer.config
        rank = self._ctx.rank

        torch.manual_seed(rank_seed(config.seed, rank))
        self._flip_generator = (
            torch.Generator().manual_seed(rank_seed(config.seed + 1_000_003, rank)) if config.sign_flip else None
        )

        self._sampler = DistributedBatchSampler(
            len(trainer.train_dataset),
            world_size=self._ctx.world_size,
            rank=rank,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            seed=config.seed,
            epoch=0,
        )
        train_loader_generator = torch.Generator().manual_seed(rank_seed(config.seed, rank))
        self._train_loader = trainer._build_data_loader(
            trainer.train_dataset,
            batch_sampler=self._sampler,
            generator=train_loader_generator,
        )

        if trainer.replay_config is not None:
            assert trainer.replay_dataset is not None
            self._replay_flip_generator = (
                torch.Generator().manual_seed(rank_seed(config.seed + 3_000_007, rank))
                if trainer._replay_sign_flip
                else None
            )
            replay_batch_size = trainer.replay_config.resolved_batch_size(config.batch_size)
            self._replay_sampler = DistributedBatchSampler(
                len(trainer.replay_dataset),
                world_size=self._ctx.world_size,
                rank=rank,
                batch_size=replay_batch_size,
                shuffle=config.shuffle,
                seed=config.seed + 2_000_009,
                epoch=0,
            )
            replay_loader_generator = torch.Generator().manual_seed(rank_seed(config.seed + 2_000_009, rank))
            self._replay_loader = trainer._build_data_loader(
                trainer.replay_dataset,
                batch_sampler=self._replay_sampler,
                generator=replay_loader_generator,
            )
            self._replay_cycle = DistributedBatchCycle(self._replay_sampler, self._replay_loader)
        else:
            self._replay_flip_generator = None
            self._replay_sampler = None
            self._replay_loader = None
            self._replay_cycle = None

        self._sharded_val_loader = None
        self._val_loader_ready = False
        self._reducer = GradientReducer(self._ctx, trainer._trainable)

    def _broadcast_rank0_hook(self, stage: str, fn: Callable[[], float]) -> float:
        """Run `fn` on rank 0 only and broadcast `[ControlStatus, value]` to every rank.

        Non-OOM `Exception`s raised inside `fn` on rank 0 broadcast `ControlStatus.FAILED`,
        re-raise the original exception on rank 0, and raise `RankFailureError((0,), stage=stage)`
        on non-zero ranks. Out-of-memory errors and `BaseException`s re-raise immediately without
        entering a collective.

        Ranks: all ranks must call in lockstep.
        Collectives: 1 (`broadcast_status`, 2 float64 scalars from rank 0).
        """
        if self._ctx.is_rank_zero:
            local_error: Exception | None = None
            value = 0.0
            try:
                value = fn()
            except Exception as exc:
                if is_oom_error(exc):
                    raise
                local_error = exc

            if local_error is not None:
                broadcast_status(self._ctx, ControlStatus.FAILED, 0.0)
                raise local_error

            status = ControlStatus.STOP if self._trainer.should_stop else ControlStatus.CONTINUE
            _, out_value = broadcast_status(self._ctx, status, value)
            return out_value

        status, out_value = broadcast_status(self._ctx)
        if status is ControlStatus.FAILED:
            raise RankFailureError((0,), stage=stage)
        self._trainer.should_stop = status is ControlStatus.STOP
        return out_value

    def on_fit_start(self) -> None:
        """Run `on_fit_start` callbacks on rank 0 and broadcast stop/failure status.

        Ranks: all ranks must call in lockstep.
        Collectives: 1 (`broadcast_status`).
        """

        def _run() -> float:
            for callback in self._trainer.callbacks:
                callback.on_fit_start(self._trainer)
            return 0.0

        self._broadcast_rank0_hook("on_fit_start", _run)

    def _precheck_domain_for_replay(
        self, fetch_batch: Callable[[], TrainingBatch | None]
    ) -> tuple[TrainingBatch | None, bool]:
        """Fetch this rank's domain batch and check global `W_D` before advancing replay.

        Runs a 2-scalar `float64` SUM collective (`[w_d, failed]`), plus `propagate_failure` if
        any rank raised a non-OOM exception during `fetch_batch()`. Returns `(raw_domain, should_skip)`.
        """
        fetch_error: Exception | None = None
        raw_domain: TrainingBatch | None = None
        pre_w_d = 0
        fetch_failed = 0.0
        try:
            raw_domain = fetch_batch()
            pre_w_d = 0 if raw_domain is None else raw_domain.num_scored
        except Exception as exc:
            if is_oom_error(exc):
                raise
            fetch_error = exc
            raw_domain = None
            pre_w_d = 0
            fetch_failed = 1.0

        pre_stats = torch.tensor(
            [float(pre_w_d), fetch_failed],
            dtype=torch.float64,
            device=self._ctx.device,
        )
        self._ctx.all_reduce_sum(pre_stats)
        pre_w_d_tot, pre_failed_tot = pre_stats.tolist()
        if pre_failed_tot > 0.0:
            self._ctx.raise_propagated_failure(fetch_error, stage="training_step")
        return raw_domain, pre_w_d_tot == 0.0

    def _run_local_stage(
        self,
        *,
        raw_domain: TrainingBatch | None,
        fetch_batch: Callable[[], TrainingBatch | None],
        fetch_replay: Callable[[], TrainingBatch | None] | None,
        has_replay: bool,
    ) -> _LocalStepResult:
        """Run local domain fetch (when no replay), batch prep, and forward.

        Runs no collectives. Non-OOM `Exception`s are caught and returned as a failed
        `_LocalStepStats` so every rank still enters the statistics all-reduce in lockstep.
        """
        trainer = self._trainer
        try:
            if not has_replay:
                raw_domain = fetch_batch()
            prepared_domain = (
                None if raw_domain is None else trainer._prepare_domain_batch(raw_domain, self._flip_generator)
            )
            prepared_replay: _PreparedReplayBatch | None = None
            if has_replay and fetch_replay is not None:
                raw_replay = fetch_replay()
                if raw_replay is not None:
                    prepared_replay = trainer._prepare_replay_batch(raw_replay, self._replay_flip_generator)

            if prepared_domain is None and prepared_replay is None:
                return _LocalStepResult(stats=_LocalStepStats())

            domain_count = 0 if prepared_domain is None else prepared_domain.num_scored
            terms = trainer._compute_loss_terms(
                None if prepared_domain is None else prepared_domain.batch,
                domain_count=domain_count,
                validate=False,
                replay=prepared_replay,
            )
            w_d, sum_d, domain_val, nf_d = _term_weighted_sum(terms.domain, domain_count)
            w_r, sum_r, replay_val, nf_r = _term_weighted_sum(terms.replay, terms.replay_count)
            m_count, sum_m, distill_val, nf_m = _term_weighted_sum(terms.distill, terms.distill_count)
            first_non_finite = next(
                (val for val in (domain_val, replay_val, distill_val) if val is not None and not math.isfinite(val)),
                None,
            )
            return _LocalStepResult(
                stats=_LocalStepStats(
                    w_d=w_d,
                    sum_d=sum_d,
                    w_r=w_r,
                    sum_r=sum_r,
                    m_count=m_count,
                    sum_m=sum_m,
                    non_finite=nf_d or nf_r or nf_m,
                ),
                terms=terms,
                first_non_finite=first_non_finite,
            )
        except Exception as exc:
            if is_oom_error(exc):
                raise
            return _LocalStepResult(stats=_LocalStepStats(failed=True), error=exc)

    def _reduce_step_stats(self, local_stats: _LocalStepStats) -> _ReducedStepStats:
        """All-reduce local step statistics across all ranks.

        Runs one `float64` SUM collective (`all_reduce_sum` over the 8-element statistics vector).
        """
        stats = local_stats.to_tensor(self._ctx.device)
        self._ctx.all_reduce_sum(stats)
        return _ReducedStepStats.from_tensor(stats)

    def _decide_step(self, reduced: _ReducedStepStats, local: _LocalStepResult) -> _StepDecision:
        """Apply global control-flow precedence (failure -> non-finite -> skip -> active step).

        Runs no collectives on normal, skipped, or non-finite steps; runs `propagate_failure` if
        any rank reported a local failure in `reduced`.
        """
        trainer = self._trainer
        if reduced.failed_tot > 0.0:
            self._ctx.raise_propagated_failure(local.error, stage="training_step")

        if reduced.non_finite_tot > 0.0:
            bad_value: object = "non-finite" if local.first_non_finite is None else local.first_non_finite
            raise FloatingPointError(
                f"the training loss became {bad_value} at step {trainer.global_step + 1}. "
                "Lower the learning rate, or check the data for extreme values"
            )

        if reduced.w_d_tot == 0.0:
            return _StepDecision(skipped=True)

        replay_cfg = trainer.replay_config
        replay_fraction = 0.0 if replay_cfg is None else replay_cfg.replay_fraction
        distill_weight = 0.0 if replay_cfg is None else replay_cfg.distill_weight
        replay_active = replay_fraction > 0.0 and reduced.w_r_tot > 0.0
        distill_active = distill_weight > 0.0 and reduced.m_tot > 0.0
        dom_weight = (1.0 - replay_fraction) if replay_active else 1.0

        global_loss = dom_weight * (reduced.sum_d_tot / reduced.w_d_tot)
        if replay_active:
            global_loss += replay_fraction * (reduced.sum_r_tot / reduced.w_r_tot)
        if distill_active:
            global_loss += distill_weight * (reduced.sum_m_tot / reduced.m_tot)
        if not math.isfinite(global_loss):
            raise FloatingPointError(
                f"the training loss became {global_loss} at step {trainer.global_step + 1}. "
                "Lower the learning rate, or check the data for extreme values"
            )

        return _StepDecision(
            skipped=False,
            global_loss=global_loss,
            dom_weight=dom_weight,
            replay_fraction=replay_fraction,
            replay_active=replay_active,
            distill_weight=distill_weight,
            distill_active=distill_active,
            w_d_tot=reduced.w_d_tot,
            w_r_tot=reduced.w_r_tot,
            m_tot=reduced.m_tot,
        )

    def _backward_local_terms(self, local: _LocalStepResult, decision: _StepDecision) -> None:
        """Run count-weighted local backward over this rank's active loss terms.

        Runs no collectives.
        """
        terms = local.terms
        if terms is None:
            return
        stats = local.stats
        local_terms: list[Tensor] = []
        if terms.domain is not None and stats.w_d > 0:
            local_terms.append((decision.dom_weight * (stats.w_d / decision.w_d_tot)) * terms.domain)
        if decision.replay_active and terms.replay is not None and stats.w_r > 0:
            local_terms.append((decision.replay_fraction * (stats.w_r / decision.w_r_tot)) * terms.replay)
        if decision.distill_active and terms.distill is not None and stats.m_count > 0:
            local_terms.append((decision.distill_weight * (stats.m_count / decision.m_tot)) * terms.distill)
        if not local_terms:
            return
        weighted_loss = local_terms[0]
        for extra in local_terms[1:]:
            weighted_loss = weighted_loss + extra
        weighted_loss.backward()

    def _reduce_gradients_and_update(self, global_loss: float) -> _StepOutcome:
        """All-reduce gradients across ranks, clip, and step the optimiser and scheduler.

        Runs one gradient SUM collective (`GradientReducer.reduce()`).
        """
        trainer = self._trainer
        assert self._reducer is not None
        learning_rate = trainer.optimizer.param_groups[0]["lr"]
        self._reducer.reduce()
        trainer._apply_optimizer_step()
        return _StepOutcome(skipped=False, loss=global_loss, learning_rate=learning_rate)

    def _run_step(
        self,
        fetch_batch: Callable[[], TrainingBatch | None],
        fetch_replay: Callable[[], TrainingBatch | None] | None = None,
    ) -> _StepOutcome:
        """Execute one distributed training step (spec §5.6, D22–D24).

        Stages:
            1. Pre-check (only when `trainer.replay_config` is set): `_precheck_domain_for_replay`.
            2. Local stage: `_run_local_stage`.
            3. Reduce statistics: `_reduce_step_stats`.
            4. Decide control flow and active weights: `_decide_step`.
            5. Local backward: `_backward_local_terms`.
            6. Reduce gradients and update: `_reduce_gradients_and_update`.

        Ranks: all ranks must call in lockstep.
        Collectives:
            - Without replay:
              - Normal step: 2 (`all_reduce_sum` 8 float64 + `GradientReducer.reduce()`).
              - Skipped (`W_D == 0`) or non-finite step: 1 (`all_reduce_sum` 8 float64).
              - Local failure: 2 (`all_reduce_sum` 8 float64 + `propagate_failure` N int64).
            - With replay:
              - Normal step: 3 (pre-check `all_reduce_sum` 2 float64 + stats `all_reduce_sum`
                8 float64 + `GradientReducer.reduce()`).
              - Skipped (`W_D == 0`) step: 1 (pre-check `all_reduce_sum` 2 float64).
              - Non-finite step: 2 (pre-check `all_reduce_sum` 2 float64 + stats `all_reduce_sum`
                8 float64).
              - Local failure during domain fetch: 2 (pre-check `all_reduce_sum` 2 float64 +
                `propagate_failure` N int64).
              - Local failure during prep/replay/forward: 3 (pre-check `all_reduce_sum` 2 float64
                + stats `all_reduce_sum` 8 float64 + `propagate_failure` N int64).
        """
        trainer = self._trainer
        assert self._reducer is not None
        trainer.optimizer.zero_grad(set_to_none=True)

        has_replay = trainer.replay_config is not None
        raw_domain: TrainingBatch | None = None
        if has_replay:
            raw_domain, should_skip = self._precheck_domain_for_replay(fetch_batch)
            if should_skip:
                return _StepOutcome(skipped=True)

        local = self._run_local_stage(
            raw_domain=raw_domain,
            fetch_batch=fetch_batch,
            fetch_replay=fetch_replay,
            has_replay=has_replay,
        )
        reduced = self._reduce_step_stats(local.stats)
        decision = self._decide_step(reduced, local)
        if decision.skipped:
            return _StepOutcome(skipped=True)

        self._backward_local_terms(local, decision)
        return self._reduce_gradients_and_update(decision.global_loss)

    def iter_epoch_steps(self, epoch: int) -> Iterator[_StepOutcome]:
        """Drive one epoch by global step index with lazy per-epoch `DataLoader` iterator creation.

        Ranks: all ranks must iterate in lockstep for the same 0-based `epoch`.
        Collectives: delegates per global iteration to `_run_step()`.
        """
        assert self._sampler is not None and self._train_loader is not None
        self._sampler.set_epoch(epoch)
        total_epoch_steps = self._sampler.num_global_steps
        loader_iter: Iterator[TrainingBatch] | None = None
        fetch_replay = None if self._replay_cycle is None else self._replay_cycle.next_batch

        for step_idx in range(total_epoch_steps):
            slice_size = self._sampler.slice_size(step_idx)

            def _fetch(size: int = slice_size) -> TrainingBatch | None:
                nonlocal loader_iter
                if size == 0:
                    return None
                if loader_iter is None:
                    assert self._train_loader is not None
                    loader_iter = iter(self._train_loader)
                return next(loader_iter)

            yield self._run_step(_fetch, fetch_replay)

    def on_step_end(self, step: int, loss: float, learning_rate: float) -> float:
        """Run `on_step_end` callbacks on rank 0 and broadcast `[status, rank0_loss]`.

        Ranks: all ranks must call in lockstep.
        Collectives: 1 (`broadcast_status`).
        """

        def _run() -> float:
            for callback in self._trainer.callbacks:
                callback.on_step_end(self._trainer, step, loss, learning_rate)
            return loss

        return self._broadcast_rank0_hook("on_step_end", _run)

    def _get_sharded_val_loader(self) -> DataLoader | None:
        """Lazily build and cache the internal strided validation shard loader (not `trainer._val_loader`)."""
        if self._val_loader_ready:
            return self._sharded_val_loader
        trainer = self._trainer
        val_dataset = trainer.val_dataset
        assert val_dataset is not None
        indices = validation_shard_indices(len(val_dataset), self._ctx.world_size, self._ctx.rank)
        if not indices:
            self._sharded_val_loader = None
            self._val_loader_ready = True
            return None
        batch_size = trainer.config.eval_batch_size or trainer.config.batch_size
        val_loader_generator = torch.Generator().manual_seed(rank_seed(trainer.config.seed, self._ctx.rank))
        self._sharded_val_loader = trainer._build_data_loader(
            val_dataset,
            batch_size=batch_size,
            indices=indices,
            generator=val_loader_generator,
        )
        self._val_loader_ready = True
        return self._sharded_val_loader

    def evaluate_val(self) -> EvalResult:
        """Evaluate `trainer.val_dataset` across strided shards for `fit()` (spec §5.6, D31).

        Never touches `trainer._val_loader`. Runs under `model.eval()` and `torch.no_grad()`,
        restoring `model.training` in a `finally:` block.

        Ranks: all ranks must call in lockstep.
        Collectives: 1 `all_reduce_sum` (4 float64 scalars `[loss_sum, scored, batches, failed]`),
            plus 1 conditional `propagate_failure` (`int64` length-`N`) only if any rank failed.
        """
        trainer = self._trainer
        assert trainer.val_dataset is not None
        was_training = trainer.model.training
        trainer.model.eval()
        local_error: Exception | None = None
        loss_sum = 0.0
        scored = 0
        batches = 0
        failed = 0.0
        try:
            with torch.no_grad():
                loader = self._get_sharded_val_loader()
                if loader is not None:
                    loss_sum, scored, batches = trainer._accumulate_eval_totals(loader)
        except Exception as exc:
            if is_oom_error(exc):
                raise
            local_error = exc
            loss_sum = 0.0
            scored = 0
            batches = 0
            failed = 1.0
        finally:
            trainer.model.train(was_training)

        totals = torch.tensor(
            [loss_sum, float(scored), float(batches), failed],
            dtype=torch.float64,
            device=self._ctx.device,
        )
        self._ctx.all_reduce_sum(totals)
        total_loss_sum, total_scored_f, total_batches_f, failed_tot = totals.tolist()

        if failed_tot > 0.0:
            self._ctx.raise_propagated_failure(local_error, stage="validation")

        total_scored = int(round(total_scored_f))
        total_batches = int(round(total_batches_f))
        if total_scored == 0:
            raise ValueError("the dataset has no scored position to evaluate")
        return EvalResult(
            loss=total_loss_sum / total_scored,
            scored_positions=total_scored,
            batches=total_batches,
        )

    def on_evaluate(self, step: int, result: EvalResult, bookkeeper: _FitBookkeeper) -> None:
        """Protect rank 0's whole post-validation block (best tracking, snapshot, callbacks) and sync status/loss.

        Ranks: all ranks must call in lockstep.
        Collectives: 1 (`broadcast_status`).
        """

        def _run() -> float:
            bookkeeper.record_eval(step, result.loss, self._trainer._trainable)
            for callback in self._trainer.callbacks:
                callback.on_evaluate(self._trainer, step, result)
            return result.loss

        broadcast_loss = self._broadcast_rank0_hook("on_evaluate", _run)
        if not self._ctx.is_rank_zero:
            bookkeeper.record_eval(step, broadcast_loss, self._trainer._trainable)

    def finalize_best(self, bookkeeper: _FitBookkeeper, current_step: int) -> bool:
        """Broadcast end-of-fit best summary and restored trainable weights when `val_dataset` exists.

        Ranks: all ranks must call in lockstep.
        Collectives: 0 when `val_dataset is None`; otherwise 1 `broadcast` of 4 float64 scalars
            `[has_best, best_eval_loss, best_step, restored]` plus `len(trainer._trainable)`
            `broadcast` calls only when `restored == 1.0`.
        """
        trainer = self._trainer
        if trainer.val_dataset is None:
            return False

        if self._ctx.is_rank_zero:
            restored = bookkeeper.restore_best_weights(
                current_step,
                trainer._trainable,
                non_blocking=trainer.device.type == "cuda",
            )
            has_best = bookkeeper.best_eval_loss is not None and bookkeeper.best_step is not None
            record = torch.tensor(
                [
                    1.0 if has_best else 0.0,
                    float(bookkeeper.best_eval_loss) if has_best else 0.0,
                    float(bookkeeper.best_step) if has_best else 0.0,
                    1.0 if restored else 0.0,
                ],
                dtype=torch.float64,
                device=self._ctx.device,
            )
        else:
            record = torch.empty(4, dtype=torch.float64, device=self._ctx.device)

        self._ctx.broadcast(record)
        has_best_f, best_loss_f, best_step_f, restored_f = record.tolist()
        restored_bool = restored_f == 1.0
        if not self._ctx.is_rank_zero:
            if has_best_f == 1.0:
                bookkeeper.apply_best_summary(best_loss_f, int(round(best_step_f)))
            else:
                bookkeeper.apply_best_summary(None, None)
        if restored_bool:
            for parameter in trainer._trainable:
                self._ctx.broadcast(parameter)
        return restored_bool

    def on_fit_end(self, result: TrainingResult) -> None:
        """Run `on_fit_end` callbacks on rank 0 and broadcast completion/failure status.

        Ranks: all ranks must call in lockstep.
        Collectives: 1 (`broadcast_status`).
        """

        def _run() -> float:
            for callback in self._trainer.callbacks:
                callback.on_fit_end(self._trainer, result)
            return 0.0

        self._broadcast_rank0_hook("on_fit_end", _run)

    def cleanup(self) -> None:
        """Release flip generators, gradient buffer, loaders, and samplers after fit (0 collectives)."""
        self._flip_generator = None
        self._replay_flip_generator = None
        self._sampler = None
        self._train_loader = None
        self._replay_sampler = None
        self._replay_loader = None
        self._replay_cycle = None
        self._sharded_val_loader = None
        self._val_loader_ready = False
        self._reducer = None
