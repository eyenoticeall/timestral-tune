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

    def _run_step(
        self,
        fetch_batch: Callable[[], TrainingBatch | None],
        fetch_replay: Callable[[], TrainingBatch | None] | None = None,
    ) -> _StepOutcome:
        """Execute one five-stage distributed training step (spec §5.6, D22–D24).

        When `trainer.replay_config` is set, a 2-scalar `float64` pre-check `[w_d, failed]`
        runs first so a globally zero-scored domain step (`W_D == 0`) skips before any rank
        advances its replay iterator. Without replay, no pre-check runs.

        Ranks: all ranks must call in lockstep.
        Collectives:
            - Without replay:
              - Normal step: 1 stats `all_reduce_sum` (8 float64) + 1 `GradientReducer.reduce()`.
              - Skipped step (`W_D == 0`): 1 stats `all_reduce_sum` (8 float64).
              - Local failure: 1 stats `all_reduce_sum` (8 float64) + 1 `propagate_failure` (N int64).
            - With replay:
              - Normal step: 1 pre-check `all_reduce_sum` (2 float64) + 1 stats `all_reduce_sum`
                (8 float64) + 1 `GradientReducer.reduce()`.
              - Skipped step (`W_D == 0`): 1 pre-check `all_reduce_sum` (2 float64) only.
              - Local failure during domain fetch: 1 pre-check `all_reduce_sum` (2 float64) + 1
                `propagate_failure` (N int64).
              - Local failure during prep/replay/forward: 1 pre-check `all_reduce_sum` (2 float64)
                + 1 stats `all_reduce_sum` (8 float64) + 1 `propagate_failure` (N int64).
        """
        trainer = self._trainer
        assert self._reducer is not None
        trainer.optimizer.zero_grad(set_to_none=True)

        replay_cfg = trainer.replay_config
        has_replay = replay_cfg is not None
        raw_domain: TrainingBatch | None = None

        # Pre-replay domain-count check: skip globally zero-scored steps before advancing replay.
        if has_replay:
            fetch_error: Exception | None = None
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
            if pre_w_d_tot == 0.0:
                return _StepOutcome(skipped=True)

        # Stage 1: protected local stage (domain fetch if no replay, domain & replay prep, forward).
        local_error: Exception | None = None
        terms: _LossTerms | None = None
        domain_val: float | None = None
        replay_val: float | None = None
        distill_val: float | None = None
        w_d = 0
        sum_d = 0.0
        w_r = 0
        sum_r = 0.0
        m_count = 0
        sum_m = 0.0
        non_finite = 0.0
        failed = 0.0
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

            if prepared_domain is not None or prepared_replay is not None:
                terms = trainer._compute_loss_terms(
                    None if prepared_domain is None else prepared_domain.batch,
                    domain_count=0 if prepared_domain is None else prepared_domain.num_scored,
                    validate=False,
                    replay=prepared_replay,
                )

                if terms.domain is not None:
                    assert prepared_domain is not None
                    w_d = prepared_domain.num_scored
                    domain_val = terms.domain.item()
                    if math.isfinite(domain_val):
                        weighted_d = domain_val * w_d
                        if math.isfinite(weighted_d):
                            sum_d = weighted_d
                        else:
                            non_finite = 1.0
                    else:
                        non_finite = 1.0

                if terms.replay is not None and terms.replay_count > 0:
                    w_r = terms.replay_count
                    replay_val = terms.replay.item()
                    if math.isfinite(replay_val):
                        weighted_r = replay_val * w_r
                        if math.isfinite(weighted_r):
                            sum_r = weighted_r
                        else:
                            non_finite = 1.0
                    else:
                        non_finite = 1.0

                if terms.distill is not None and terms.distill_count > 0:
                    m_count = terms.distill_count
                    distill_val = terms.distill.item()
                    if math.isfinite(distill_val):
                        weighted_m = distill_val * m_count
                        if math.isfinite(weighted_m):
                            sum_m = weighted_m
                        else:
                            non_finite = 1.0
                    else:
                        non_finite = 1.0
        except Exception as exc:
            if is_oom_error(exc):
                raise
            local_error = exc
            terms = None
            domain_val = None
            replay_val = None
            distill_val = None
            w_d = 0
            sum_d = 0.0
            w_r = 0
            sum_r = 0.0
            m_count = 0
            sum_m = 0.0
            non_finite = 0.0
            failed = 1.0

        # Stage 2: 8-scalar float64 SUM statistics collective.
        stats = torch.tensor(
            [float(w_d), sum_d, float(w_r), sum_r, float(m_count), sum_m, non_finite, failed],
            dtype=torch.float64,
            device=self._ctx.device,
        )
        self._ctx.all_reduce_sum(stats)
        reduced = stats.tolist()
        w_d_tot = reduced[_STAT_W_D]
        sum_d_tot = reduced[_STAT_SUM_D]
        w_r_tot = reduced[_STAT_W_R]
        sum_r_tot = reduced[_STAT_SUM_R]
        m_tot = reduced[_STAT_M]
        sum_m_tot = reduced[_STAT_SUM_M]
        non_finite_tot = reduced[_STAT_NON_FINITE]
        failed_tot = reduced[_STAT_FAILED]

        # Stage 3: globally decided control flow (failure -> non-finite -> global skip -> active step).
        if failed_tot > 0.0:
            self._ctx.raise_propagated_failure(local_error, stage="training_step")

        if non_finite_tot > 0.0:
            bad_value: object = "non-finite"
            for candidate in (domain_val, replay_val, distill_val):
                if candidate is not None and not math.isfinite(candidate):
                    bad_value = candidate
                    break
            raise FloatingPointError(
                f"the training loss became {bad_value} at step {trainer.global_step + 1}. "
                "Lower the learning rate, or check the data for extreme values"
            )

        if w_d_tot == 0.0:
            return _StepOutcome(skipped=True)

        replay_fraction = 0.0 if replay_cfg is None else replay_cfg.replay_fraction
        distill_weight = 0.0 if replay_cfg is None else replay_cfg.distill_weight
        replay_active = replay_fraction > 0.0 and w_r_tot > 0.0
        distill_active = distill_weight > 0.0 and m_tot > 0.0
        dom_weight = (1.0 - replay_fraction) if replay_active else 1.0

        global_loss = dom_weight * (sum_d_tot / w_d_tot)
        if replay_active:
            global_loss += replay_fraction * (sum_r_tot / w_r_tot)
        if distill_active:
            global_loss += distill_weight * (sum_m_tot / m_tot)
        if not math.isfinite(global_loss):
            raise FloatingPointError(
                f"the training loss became {global_loss} at step {trainer.global_step + 1}. "
                "Lower the learning rate, or check the data for extreme values"
            )

        # Stage 4: count-weighted local backward over active local terms.
        if terms is not None:
            local_terms: list[Tensor] = []
            if terms.domain is not None and w_d > 0:
                local_terms.append((dom_weight * (w_d / w_d_tot)) * terms.domain)
            if replay_active and terms.replay is not None and w_r > 0:
                local_terms.append((replay_fraction * (w_r / w_r_tot)) * terms.replay)
            if distill_active and terms.distill is not None and m_count > 0:
                local_terms.append((distill_weight * (m_count / m_tot)) * terms.distill)
            if local_terms:
                weighted_loss = local_terms[0]
                for extra in local_terms[1:]:
                    weighted_loss = weighted_loss + extra
                weighted_loss.backward()

        # Stage 5: single flat-buffer SUM gradient collective, clipping, optimiser and scheduler step.
        learning_rate = trainer.optimizer.param_groups[0]["lr"]
        self._reducer.reduce()
        trainer._apply_optimizer_step()
        return _StepOutcome(skipped=False, loss=global_loss, learning_rate=learning_rate)

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
