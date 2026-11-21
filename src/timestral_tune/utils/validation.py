"""Parity, leakage and loss-mask verifiers.

These are the checks behind the acceptance tests, packaged so that they can be run on any model
and any batch, including a user's own data:

    from timestral_tune.utils.validation import run_checks
    for result in run_checks(model, batch):
        print(result)

Each check returns a `CheckResult`. A check that does not apply to the batch (for example the
firewall check on a batch without covariates) is reported as skipped, not as passed.

Perturbations are non-constant noise on observed values. A constant offset would be absorbed by the
normalisation and make a leakage check pass vacuously.

- Spec: `docs/SPEC.md` §9
- Acceptance tests: T1, T2, T4, T6, T7, T8
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from torch import Tensor

from .._vendor.timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, TimestralModel
from ..data.batch import TrainingBatch
from ..losses.pinball import pinball_loss
from ..models.forward import TargetFrame, denormalise_quantiles, training_forward
from ..trainer.step import compute_loss

__all__ = [
    "CheckResult",
    "check_firewall",
    "check_forward_parity",
    "check_known_future_sensitivity",
    "check_label_isolation",
    "check_loss_mask",
    "run_checks",
]


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one verifier.

    Attributes:
        name: Which check, with its acceptance-test number.
        passed: True if the property holds. False for a skipped check.
        skipped: True if the check does not apply to this batch.
        detail: What was measured, or why the check was skipped.
    """

    name: str
    passed: bool
    detail: str
    skipped: bool = False

    def __str__(self) -> str:
        status = "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")
        return f"[{status}] {self.name}: {self.detail}"


def _skip(name: str, reason: str) -> CheckResult:
    return CheckResult(name=name, passed=False, detail=reason, skipped=True)


def _noise_like(values: Tensor, generator: torch.Generator) -> Tensor:
    """Non-constant noise on the scale of `values`, generated on CPU for reproducibility across devices."""
    scale = values.std().clamp_min(1.0).item()
    noise = torch.randn(values.shape, generator=generator, dtype=torch.float32) * scale
    return noise.to(values.device, values.dtype)


def _max_abs_difference(a: Tensor, b: Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    # On CPU first: the Apple GPU backend has no float64.
    return (a.detach().cpu().double() - b.detach().cpu().double()).abs().max().item()


@torch.no_grad()
def check_forward_parity(model: TimestralModel, batch: TrainingBatch, rtol: float = 1e-5) -> CheckResult:
    """T1/T2: the denormalised training forward equals upstream's `predict_block`.

    The two run the same operations, so they agree to float32 round-off. Measured: at most 8e-10 on
    CPU, where the acceptance tests hold it to 1e-6, and up to 8e-7 on the Apple GPU. The default
    tolerance leaves room for accelerators.
    """
    name = "forward parity with upstream predict_block (T1/T2)"
    patch = model.config.patch_size
    output = training_forward(model, batch)
    ours = denormalise_quantiles(model, output.quantiles, output.anchor_loc, output.anchor_scale)
    reference = model.predict_block(
        batch.context,
        batch.context_mask,
        batch.group_ids,
        batch.horizon_steps // patch,
        batch.role_ids,
        batch.known_future,
        batch.known_future_mask,
    )
    difference = _max_abs_difference(ours, reference) / max(reference.abs().max().item(), 1e-300)
    return CheckResult(name, difference <= rtol, f"max relative difference {difference:.2e} (tolerance {rtol:.0e})")


@torch.no_grad()
def check_label_isolation(model: TimestralModel, batch: TrainingBatch, seed: int = 0) -> CheckResult:
    """T4: replacing the horizon target values with noise leaves the predictions bit-identical."""
    name = "label isolation (T4)"
    generator = torch.Generator().manual_seed(seed)
    noisy = dataclasses.replace(batch, targets=batch.targets + _noise_like(batch.targets, generator))
    original = training_forward(model, batch).quantiles
    perturbed = training_forward(model, noisy).quantiles
    identical = torch.equal(original, perturbed)
    return CheckResult(name, identical, f"max change in predictions {_max_abs_difference(original, perturbed):.2e}")


@torch.no_grad()
def check_firewall(model: TimestralModel, batch: TrainingBatch, seed: int = 0) -> CheckResult:
    """T6: covariate rows never read target rows, and known-future rows never read past-only rows.

    Noise is added to the observed context values of the target rows; no covariate row's hidden
    state or output may change, while target rows must change (otherwise the perturbation did
    nothing and the check would be vacuous). The same is then done from past-only rows to
    known-future rows.
    """
    name = "role firewall (T6)"
    if batch.role_ids is None or bool(batch.target_rows.all()):
        return _skip(name, "the batch has no covariate rows")
    generator = torch.Generator().manual_seed(seed)
    baseline = training_forward(model, batch, return_hidden=True)
    details = []
    passed = True

    def perturb(source: Tensor) -> tuple[Tensor, Tensor]:
        noise = _noise_like(batch.context, generator) * (source[:, None] & batch.context_mask)
        noisy = training_forward(model, dataclasses.replace(batch, context=batch.context + noise), return_hidden=True)
        return noisy.hidden, noisy.quantiles

    target_rows = batch.target_rows
    hidden, quantiles = perturb(target_rows)
    leak = max(
        _max_abs_difference(hidden[~target_rows], baseline.hidden[~target_rows]),
        _max_abs_difference(quantiles[~target_rows], baseline.quantiles[~target_rows]),
    )
    moved = _max_abs_difference(quantiles[target_rows], baseline.quantiles[target_rows])
    passed &= leak == 0.0 and moved > 0.0
    details.append(f"noise on targets changed covariate rows by {leak:.2e} (targets themselves by {moved:.2e})")

    past_rows, known_rows = batch.role_ids == ROLE_PAST_COV, batch.role_ids == ROLE_KNOWN_COV
    if bool(past_rows.any()) and bool(known_rows.any()):
        hidden, quantiles = perturb(past_rows)
        leak = max(
            _max_abs_difference(hidden[known_rows], baseline.hidden[known_rows]),
            _max_abs_difference(quantiles[known_rows], baseline.quantiles[known_rows]),
        )
        moved = _max_abs_difference(quantiles[past_rows], baseline.quantiles[past_rows])
        passed &= leak == 0.0 and moved > 0.0
        details.append(f"noise on past-only rows changed known-future rows by {leak:.2e}")
    return CheckResult(name, bool(passed), "; ".join(details))


@torch.no_grad()
def check_known_future_sensitivity(model: TimestralModel, batch: TrainingBatch, seed: int = 0) -> CheckResult:
    """T7: a positive control. Noise on known-future covariate values in the horizon changes the target forecast."""
    name = "known-future covariates reach the targets (T7)"
    if batch.known_future is None or not bool(batch.known_future_mask.any()):
        return _skip(name, "the batch has no observed known-future covariate values in the horizon")
    generator = torch.Generator().manual_seed(seed)
    noise = _noise_like(batch.known_future, generator) * batch.known_future_mask
    noisy = dataclasses.replace(batch, known_future=batch.known_future + noise)
    target_rows = batch.target_rows
    original = training_forward(model, batch).quantiles[target_rows]
    perturbed = training_forward(model, noisy).quantiles[target_rows]
    change = _max_abs_difference(original, perturbed)
    return CheckResult(name, change > 0.0, f"max change in target predictions {change:.2e}")


def check_loss_mask(model: TimestralModel, batch: TrainingBatch, frame: TargetFrame = "median_folded") -> CheckResult:
    """T8: the loss gradient with respect to the predictions is exactly zero at every unscored position
    (covariate rows, unobserved steps, horizon padding) and non-zero at every scored one.
    """
    name = f"loss mask, {frame} frame (T8)"
    if batch.num_scored == 0:
        return _skip(name, "the batch has no scored position")
    with torch.no_grad():
        result = compute_loss(model, batch, frame)
    # Differentiate the loss with respect to the predictions themselves, as a leaf. This needs no gradient
    # through the model, so it works whatever is frozen and leaves every `requires_grad` flag alone. The
    # target frame is a constant of the loss under both definitions, so nothing is lost.
    predictions = result.forward.quantiles.detach().requires_grad_(True)
    levels = predictions.new_tensor(model.config.quantile_knots)
    with torch.enable_grad():
        loss = pinball_loss(predictions, result.normalised_targets, levels, batch.loss_weight)
        (gradient,) = torch.autograd.grad(loss, predictions)
    weight = batch.loss_weight
    leaked = int((gradient[~weight] != 0).sum())
    missing = int((gradient[weight].abs().sum(dim=-1) == 0).sum())
    finite = bool(torch.isfinite(gradient).all()) and bool(torch.isfinite(loss))
    passed = finite and leaked == 0 and missing == 0
    detail = (
        f"{leaked} unscored positions with a gradient, {missing} scored positions without one, "
        f"{int(weight.sum())} scored of {weight.numel()}"
    )
    return CheckResult(name, passed, detail if finite else "non-finite loss or gradient; " + detail)


def run_checks(model: TimestralModel, batch: TrainingBatch, seed: int = 0) -> list[CheckResult]:
    """Run every verifier on `batch` and return the results in acceptance-test order."""
    return [
        check_forward_parity(model, batch),
        check_label_isolation(model, batch, seed),
        check_firewall(model, batch, seed),
        check_known_future_sensitivity(model, batch, seed),
        check_loss_mask(model, batch, "median_folded"),
        check_loss_mask(model, batch, "true_values"),
    ]
