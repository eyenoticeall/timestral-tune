"""Distillation to the frozen base model (spec §5.3).

Smooth L1 (Huber) difference between the tuned model's and the frozen base model's normalised
quantiles on replay samples (default transition `smooth_beta=1e-3`, falling back to exact L1 when
`smooth_beta=0.0`). Smoothing the kink at zero prevents `~1e-7` fp32 GEMM tiling round-off
between cached and live teachers from firing full-strength `+/-1` subgradients at step 0 when the
student still equals the frozen base. Because supervision comes from the base model's own
forecast distribution, distillation requires no target labels.

- Spec: `docs/SPEC.md` §5.3, §11 (D21)
"""

from __future__ import annotations

import math
import numbers

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["distillation_loss"]


def distillation_loss(
    student_quantiles: Tensor,
    teacher_quantiles: Tensor,
    weight_mask: Tensor | None = None,
    *,
    smooth_beta: float = 1e-3,
) -> Tensor:
    """Mean absolute (or smooth L1) difference between student and frozen-base normalised quantiles.

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
        smooth_beta: Huber/Smooth-L1 transition threshold in normalised quantile units (`>= 0`).
            Default `1e-3` behaves as exact L1 for all meaningful drift (`|diff| >= 1e-3`) while
            preventing `+/-1` subgradient kicks from `1e-7` step-0 floating-point round-off.
            Set to `0.0` for exact unsmoothed L1.
    """
    if student_quantiles.ndim != 3:
        raise ValueError(f"student_quantiles must be 3-D (R, H, Q), got {tuple(student_quantiles.shape)}")
    if teacher_quantiles.shape != student_quantiles.shape:
        raise ValueError(
            f"teacher_quantiles shape {tuple(teacher_quantiles.shape)} does not match "
            f"student_quantiles shape {tuple(student_quantiles.shape)}"
        )
    if isinstance(smooth_beta, bool) or not (
        isinstance(smooth_beta, numbers.Real) and math.isfinite(smooth_beta) and smooth_beta >= 0.0
    ):
        raise ValueError(f"smooth_beta must be non-negative and finite, got {smooth_beta!r}")

    work_dtype = torch.promote_types(student_quantiles.dtype, torch.float32)
    student = student_quantiles.to(work_dtype)
    teacher = teacher_quantiles.detach().to(device=student.device, dtype=work_dtype)
    if smooth_beta == 0.0:
        per_pos = (student - teacher).abs().mean(dim=-1)  # (R, H)
    else:
        per_pos = F.smooth_l1_loss(student, teacher, reduction="none", beta=float(smooth_beta)).mean(dim=-1)

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
