"""The batch layout shared by the collator, the training forward and the loss.

A `TrainingBatch` uses upstream's packed row layout: one row per variate of every series in the
batch. With `B` series of `V` rows each, `R = B * V`. Rows of one series are contiguous and share a
`group_ids` value. Within a series the row order is: target variates, past-only covariates,
known-future covariates (categorical covariates, once encoded, follow the numeric ones of the same
kind).

Shapes use `T` for the context length in time steps and `H` for the horizon length in time steps.
Both are whole multiples of the model's patch size; the collator pads to make them so.

- Spec: `docs/SPEC.md` §3.3, §3.4, §4.2, §6.2
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from .._vendor.timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET

__all__ = ["TrainingBatch"]

_VALID_ROLES = (ROLE_TARGET, ROLE_PAST_COV, ROLE_KNOWN_COV)


@dataclass(frozen=True)
class TrainingBatch:
    """One training or evaluation batch in the packed row layout.

    Attributes:
        context: `(R, T)` float. Model inputs over the context. Target rows are forward-filled;
            covariate rows hold 0 where unobserved.
        context_mask: `(R, T)` bool. True where `context` is an observed value.
        group_ids: `(R,)` long. Series index of each row; rows of one series attend to each other.
        role_ids: `(R,)` long, or None when every row is a target. Values are upstream's
            `ROLE_TARGET`, `ROLE_PAST_COV`, `ROLE_KNOWN_COV`.
        known_future: `(R, H)` float, or None when there is no known-future covariate. The values
            known in advance over the horizon; 0 on every other row.
        known_future_mask: `(R, H)` bool, present exactly when `known_future` is.
        targets: `(R, H)` float. Label values over the horizon on target rows. These are labels
            only and are never fed to the model. 0 wherever `loss_weight` is False.
        loss_weight: `(R, H)` bool. True exactly where a prediction is scored: a target row, inside
            the requested horizon (not the padding that rounds it up to whole patches), observed.
    """

    context: Tensor
    context_mask: Tensor
    group_ids: Tensor
    role_ids: Tensor | None
    known_future: Tensor | None
    known_future_mask: Tensor | None
    targets: Tensor
    loss_weight: Tensor

    @property
    def num_rows(self) -> int:
        return self.context.shape[0]

    @property
    def context_steps(self) -> int:
        return self.context.shape[1]

    @property
    def horizon_steps(self) -> int:
        return self.targets.shape[1]

    @property
    def device(self) -> torch.device:
        return self.context.device

    @property
    def target_rows(self) -> Tensor:
        """`(R,)` bool: which rows are target variates."""
        if self.role_ids is None:
            return torch.ones(self.num_rows, dtype=torch.bool, device=self.device)
        return self.role_ids == ROLE_TARGET

    @property
    def num_scored(self) -> int:
        """Number of scored positions. A batch with 0 contributes no loss and is skipped by the trainer."""
        return int(self.loss_weight.sum())

    def to(self, device: torch.device | str, non_blocking: bool = False) -> TrainingBatch:
        """A copy with every tensor on `device`."""
        moved = {
            f.name: None if (value := getattr(self, f.name)) is None else value.to(device, non_blocking=non_blocking)
            for f in fields(self)
        }
        return TrainingBatch(**moved)

    def pin_memory(self) -> TrainingBatch:
        """Used by `torch.utils.data.DataLoader(pin_memory=True)`."""
        pinned = {
            f.name: None if (value := getattr(self, f.name)) is None else value.pin_memory() for f in fields(self)
        }
        return TrainingBatch(**pinned)

    def validate(self, patch_size: int) -> None:
        """Raise `ValueError` if shapes, dtypes or values break the layout's contract."""
        rows, context_steps = self.context.shape if self.context.dim() == 2 else (None, None)
        if rows is None:
            raise ValueError(f"context must be (R, T), got {tuple(self.context.shape)}")
        if self.targets.dim() != 2 or self.targets.shape[0] != rows:
            raise ValueError(f"targets must be (R={rows}, H), got {tuple(self.targets.shape)}")
        horizon_steps = self.targets.shape[1]
        if context_steps == 0 or context_steps % patch_size != 0:
            raise ValueError(f"context length {context_steps} must be a positive multiple of patch size {patch_size}")
        if horizon_steps == 0 or horizon_steps % patch_size != 0:
            raise ValueError(f"horizon length {horizon_steps} must be a positive multiple of patch size {patch_size}")

        expected = {
            "context_mask": ((rows, context_steps), torch.bool),
            "group_ids": ((rows,), torch.long),
            "loss_weight": ((rows, horizon_steps), torch.bool),
        }
        if self.role_ids is not None:
            expected["role_ids"] = ((rows,), torch.long)
        if (self.known_future is None) != (self.known_future_mask is None):
            raise ValueError("known_future and known_future_mask must be given together")
        if self.known_future is not None:
            if self.known_future.shape != (rows, horizon_steps):
                raise ValueError(
                    f"known_future must be (R={rows}, H={horizon_steps}), got {tuple(self.known_future.shape)}"
                )
            expected["known_future_mask"] = ((rows, horizon_steps), torch.bool)
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"{name} must be {shape} {dtype}, got {tuple(value.shape)} {value.dtype}")

        if self.role_ids is not None:
            unknown = ~torch.isin(self.role_ids, torch.tensor(_VALID_ROLES, device=self.role_ids.device))
            if bool(unknown.any()):
                raise ValueError(f"role_ids holds values outside {_VALID_ROLES}")
            known_rows = self.role_ids == ROLE_KNOWN_COV
            if bool(known_rows.any()) and self.known_future is None:
                raise ValueError("known-future covariate rows are present but known_future is None")
            if self.known_future is not None and bool(self.known_future_mask[~known_rows].any()):
                raise ValueError("known_future_mask may be True on known-future covariate rows only")
        elif self.known_future is not None:
            raise ValueError("known_future needs role_ids to say which rows are known-future covariates")

        if bool(self.loss_weight[~self.target_rows].any()):
            raise ValueError("loss_weight may be True on target rows only")
        for name in ("context", "targets", "known_future"):
            value = getattr(self, name)
            if value is not None and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} holds non-finite values; missing values must be masked and set to 0")
