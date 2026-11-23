"""Probe: measure `F.linear` and step-0 distillation differences under `requires_grad` flags.

    uv run python scripts/probes/linear_requires_grad_dispatch.py
    uv run --isolated --python 3.12 --with "torch==2.4.0" python scripts/probes/linear_requires_grad_dispatch.py

Runs entirely on CPU without network access and prints three measurements:

(a) Standalone `torch.nn.functional.linear` outputs across contiguous and permuted non-contiguous
    inputs, with `weight.requires_grad` set to `True` versus `False`, in both `enable_grad` and
    `no_grad` modes (float32 and float64).
(b) On the tiny random model (`_common.tiny_random_forecaster`), forward quantile differences
    between a `full_ft` student (`requires_grad=True`) and an identical-weight teacher in float32
    and float64, both when the teacher is frozen as `TimestralTrainer` freezes it
    (`requires_grad=False`) and when the teacher's weights retain `requires_grad=True` (still run
    under `torch.no_grad()`).
(c) The resulting step-0 `distillation_loss` value and student gradient L2 norm in each case
    (at `smooth_beta=1e-3` and `smooth_beta=0.0`).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import platform
import sys

import torch
import torch.nn.functional as F
from _common import as_double, float64_reference, make_batch, rel, tiny_random_forecaster

from timestral_tune.models.forward import training_forward
from timestral_tune.replay import distillation_loss


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().double() - b.detach().double()).abs().max().item())


def _grad_l2(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().double().pow(2).sum().item())
    return total**0.5


def _section_a(seed: int) -> None:
    print("\n[a] Standalone F.linear: contiguous vs permuted non-contiguous inputs (CPU)")
    print(
        f"{'dtype':<7} {'base_shape':<13} {'perm':<11} {'layout':<11} {'out':>4}  "
        f"{'grad:T-F':>10} {'nograd:T-F':>10} {'rgT:g-ng':>10} {'rgF:g-ng':>10}"
    )
    cases: list[tuple[tuple[int, int, int], tuple[int, int, int], int]] = [
        ((8, 5, 32), (1, 0, 2), 32),
        ((1, 6, 64), (1, 0, 2), 64),
        ((2, 6, 64), (1, 0, 2), 64),
        ((4, 6, 64), (1, 0, 2), 64),
        ((8, 6, 64), (1, 0, 2), 64),
        ((2, 6, 64), (1, 0, 2), 192),
        ((8, 6, 64), (1, 0, 2), 192),
    ]
    for dtype_name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        g = torch.Generator().manual_seed(seed)
        for base_shape, perm, out_features in cases:
            in_features = base_shape[-1]
            x_base = torch.randn(base_shape, generator=g, dtype=dtype)
            w_data = torch.randn((out_features, in_features), generator=g, dtype=dtype)
            b_data = torch.randn((out_features,), generator=g, dtype=dtype)

            x_perm = x_base.permute(*perm)
            x_contig = x_perm.contiguous()
            for layout_name, x in (("contiguous", x_contig), ("permuted", x_perm)):
                w_t = w_data.clone().requires_grad_(True)
                b_t = b_data.clone().requires_grad_(True)
                w_f = w_data.clone().requires_grad_(False)
                b_f = b_data.clone().requires_grad_(False)

                with torch.enable_grad():
                    y_g_t = F.linear(x, w_t, b_t)
                    y_g_f = F.linear(x, w_f, b_f)
                with torch.no_grad():
                    y_ng_t = F.linear(x, w_t, b_t)
                    y_ng_f = F.linear(x, w_f, b_f)

                print(
                    f"{dtype_name:<7} {str(base_shape):<13} {str(perm):<11} {layout_name:<11} {out_features:>4}  "
                    f"{_max_abs(y_g_t, y_g_f):>10.2e} {_max_abs(y_ng_t, y_ng_f):>10.2e} "
                    f"{_max_abs(y_g_t, y_ng_t):>10.2e} {_max_abs(y_g_f, y_ng_f):>10.2e}"
                )


def _section_b_and_c(seed: int) -> None:
    print("\n[b] Tiny random model: student (train, rg=True) vs identical-weight teacher (eval, no_grad)")
    print(
        f"{'dtype':<7} {'layout':<11} {'bs':>3}  "
        f"{'frozen max_abs':>14} {'frozen rel_l2':>14}  "
        f"{'rgTrue max_abs':>14} {'rgTrue rel_l2':>14}"
    )

    grad_rows: list[str] = []
    base_f32 = tiny_random_forecaster(seed=seed).model

    for dtype_name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        ctx = float64_reference() if dtype == torch.float64 else contextlib.nullcontext()
        with ctx:
            base = copy.deepcopy(base_f32).double() if dtype == torch.float64 else copy.deepcopy(base_f32)
            for layout_name, rows_per_series, covariates, batch_seed in (
                ("univariate", 1, False, seed + 1),
                ("covariates", 3, True, seed + 2),
            ):
                for bs in (1, 2, 4, 8):
                    batch = make_batch(
                        series=bs,
                        rows_per_series=rows_per_series,
                        context_patches=4,
                        horizon_patches=2,
                        covariates=covariates,
                        seed=batch_seed,
                    )
                    if dtype == torch.float64:
                        batch = as_double(batch)

                    student = copy.deepcopy(base).train()
                    for p in student.parameters():
                        p.requires_grad_(True)

                    teacher_frozen = copy.deepcopy(base).eval()
                    for p in teacher_frozen.parameters():
                        p.requires_grad_(False)

                    teacher_rg_true = copy.deepcopy(base).eval()
                    for p in teacher_rg_true.parameters():
                        p.requires_grad_(True)

                    with torch.no_grad():
                        q_frozen = training_forward(teacher_frozen, batch, validate=False).quantiles[batch.target_rows]
                        q_rg_true = training_forward(teacher_rg_true, batch, validate=False).quantiles[
                            batch.target_rows
                        ]

                    with torch.enable_grad():
                        q_student_probe = training_forward(student, batch, validate=False).quantiles[batch.target_rows]

                    print(
                        f"{dtype_name:<7} {layout_name:<11} {bs:>3}  "
                        f"{_max_abs(q_student_probe, q_frozen):>14.2e} {rel(q_student_probe, q_frozen):>14.2e}  "
                        f"{_max_abs(q_student_probe, q_rg_true):>14.2e} {rel(q_student_probe, q_rg_true):>14.2e}"
                    )

                    student_params = list(student.parameters())
                    for teacher_label, q_teacher in (("frozen(rg=F)", q_frozen), ("unfrozen(rg=T)", q_rg_true)):
                        for beta in (1e-3, 0.0):
                            student.zero_grad(set_to_none=True)
                            q_student = training_forward(student, batch, validate=False).quantiles[batch.target_rows]
                            loss = distillation_loss(q_student, q_teacher, smooth_beta=beta)
                            loss.backward()
                            gnorm = _grad_l2(student_params)
                            grad_rows.append(
                                f"{dtype_name:<7} {layout_name:<11} {bs:>3}  "
                                f"{teacher_label:<14} {beta:>6.0e}  {loss.item():>12.2e} {gnorm:>12.2e}"
                            )

    print("\n[c] Step-0 distillation loss and student gradient L2 norm at identical weights")
    print(f"{'dtype':<7} {'layout':<11} {'bs':>3}  {'teacher':<14} {'beta':>6}  {'distill_loss':>12} {'grad_l2':>12}")
    for row in grad_rows:
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"== linear_requires_grad_dispatch | torch {torch.__version__} | "
        f"{platform.system()} {platform.machine()} | threads={torch.get_num_threads()} | seed={args.seed} =="
    )
    _section_a(args.seed)
    _section_b_and_c(args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
