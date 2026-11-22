"""Distillation to the frozen base model (spec §5.3).

Mean absolute difference (L1) between the tuned model's and the frozen base model's normalised
quantiles on replay samples. Because the supervision comes from the base model's own forecast
distribution, distillation requires no target labels.

- Spec: `docs/SPEC.md` §5.3
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["distillation_loss"]


def distillation_loss(
    student_quantiles: Tensor,
    teacher_quantiles: Tensor,
    weight_mask: Tensor | None = None,
) -> Tensor:
    """Mean absolute difference between student and frozen-base normalised quantiles.

    Computes in at least float32 (matching `pinball_loss` promotion) and detaches
    `teacher_quantiles` so gradients never flow into the reference model. Averages equally over
    quantile levels and active `(row, step)` positions; returns exact `0.0` (connected to
    `student_quantiles` so `.backward()` stays safe) when `weight_mask` has no active positions.

    Args:
        student_quantiles: `(R, H, Q)` normalised quantiles from the model being trained.
        teacher_quantiles: `(R, H, Q)` normalised quantiles from the frozen base model.
        weight_mask: Optional `(R, H)` boolean or non-negative float mask selecting active
            `(row, step)` positions (e.g. real target rows inside the requested unpadded horizon).
            If omitted, every `(row, step)` position receives weight 1.
    """
    if student_quantiles.ndim != 3:
        raise ValueError(f"student_quantiles must be 3-D (R, H, Q), got {tuple(student_quantiles.shape)}")
    if teacher_quantiles.shape != student_quantiles.shape:
        raise ValueError(
            f"teacher_quantiles shape {tuple(teacher_quantiles.shape)} does not match "
            f"student_quantiles shape {tuple(student_quantiles.shape)}"
        )

    work_dtype = torch.promote_types(student_quantiles.dtype, torch.float32)
    student = student_quantiles.to(work_dtype)
    teacher = teacher_quantiles.detach().to(device=student.device, dtype=work_dtype)
    per_pos = (student - teacher).abs().mean(dim=-1)  # (R, H)

    if weight_mask is None:
        if per_pos.numel() == 0:
            return student.sum() * 0.0
        return per_pos.mean()

    expected_mask_shape = student_quantiles.shape[:2]
    if tuple(weight_mask.shape) != expected_mask_shape:
        raise ValueError(f"weight_mask must have shape {expected_mask_shape}, got {tuple(weight_mask.shape)}")
    mask = weight_mask.to(device=student.device, dtype=work_dtype)
    if weight_mask.dtype != torch.bool:
        if not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()):
            raise ValueError("weight_mask must be finite and non-negative")
    total_weight = mask.sum()
    if float(total_weight) == 0.0:
        return student.sum() * 0.0
    return (per_pos * mask).sum() / total_weight
