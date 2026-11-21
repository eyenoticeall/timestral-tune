"""LoRA adapter tests (spec §9, T9-T10)."""

from __future__ import annotations

import copy

import pytest
import torch

from timestral_tune import LoraConfig
from timestral_tune._vendor.timestral import TimestralModel
from timestral_tune.models import (
    inject_lora,
    lora_parameters,
    merge_lora,
    merged_state_dict,
    training_forward,
    unmerge_lora,
)

from ..helpers import as_double, float64_reference, max_rel, upstream_batch

MERGE_RTOL = 1e-4
FLOAT64_MERGE_RTOL = 1e-12


@pytest.mark.acceptance
@pytest.mark.parametrize("axes", [("variate",), ("time",)], ids=["variate", "time"])
@pytest.mark.parametrize("scenario", ["covariates"], indirect=True)
def test_t09_lora_identity_and_merge(forecaster, raw, axes):
    """T9. Passes when: A freshly initialised adapter leaves outputs unchanged; merged and unmerged outputs
    agree within tolerance.
    """
    batch, _, _ = upstream_batch(forecaster, raw)
    base_model = forecaster.model
    with torch.no_grad():
        base_out = training_forward(base_model, batch).quantiles

    torch.manual_seed(9)
    adapted = copy.deepcopy(base_model)
    config = LoraConfig(r=8, lora_alpha=16, axes=axes, targets=("q", "v", "o", "gate"), dropout=0.05)
    inject_lora(adapted, config)
    adapted.eval()

    # Step-0 identity: zero-initialised B leaves predictions unchanged (float32 BLAS/layout round-off
    # on 28M's 500k-logit layers reaches 2.8e-6 on x86 CPU; exact identity is also checked in float64 below).
    assert all(p.abs().max().item() == 0.0 for n, p in adapted.named_parameters() if ".lora_B." in n)
    with torch.no_grad():
        fresh_out = training_forward(adapted, batch).quantiles
    assert max_rel(fresh_out, base_out) < 1e-5, (
        f"freshly initialised LoRA moved output by {max_rel(fresh_out, base_out):.2e}"
    )

    base64 = copy.deepcopy(base_model).double().eval()
    fresh64 = copy.deepcopy(adapted).double().eval()
    batch64 = as_double(batch)
    with torch.no_grad(), float64_reference():
        assert (
            max_rel(training_forward(fresh64, batch64).quantiles, training_forward(base64, batch64).quantiles) < 1e-12
        )

    # Perturb B matrices to simulate training: output must move, and merged/unmerged must agree.
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for name, param in adapted.named_parameters():
            if ".lora_B." in name:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.02)
        unmerged_out = training_forward(adapted, batch).quantiles
    assert max_rel(unmerged_out, base_out) > 1e-4, "non-zero LoRA weights did not change the output"

    # Exact algebraic merge equivalence verified in float64 (catches any slice-index or scaling bug).
    adapted64 = copy.deepcopy(adapted).double().eval()
    batch64 = as_double(batch)
    with torch.no_grad(), float64_reference():
        unmerged64 = training_forward(adapted64, batch64).quantiles
        merge_lora(adapted64)
        merged64 = training_forward(adapted64, batch64).quantiles
    assert max_rel(merged64, unmerged64) < FLOAT64_MERGE_RTOL

    # Non-destructive merged state dict loads in upstream TimestralModel with strict=True and matches outputs.
    state = merged_state_dict(adapted)
    reloaded = TimestralModel(adapted.config).eval()
    reloaded.load_state_dict(state, strict=True)
    with torch.no_grad():
        reloaded_out = training_forward(reloaded, batch).quantiles
    assert max_rel(reloaded_out, unmerged_out) < MERGE_RTOL

    # In-place merge and unmerge round-trip cleanly.
    merge_lora(adapted)
    with torch.no_grad():
        merged_out = training_forward(adapted, batch).quantiles
    assert max_rel(merged_out, unmerged_out) < MERGE_RTOL

    unmerge_lora(adapted)
    with torch.no_grad():
        restored_out = training_forward(adapted, batch).quantiles
    assert max_rel(restored_out, unmerged_out) < MERGE_RTOL


@pytest.mark.acceptance
def test_t10_single_row_guard(forecaster):
    """T10. Passes when: `lora_variate` on single-row data raises; with the override, query and key slices
    are rejected.
    """
    # Default variate LoRA on single-row data raises.
    with pytest.raises(ValueError, match="single-row"):
        inject_lora(
            copy.deepcopy(forecaster.model), LoraConfig(axes=("variate",), targets=("q", "v")), rows_per_series=1
        )

    # With allow_single_row=True, q and k slices are still rejected.
    for bad_targets in (("q",), ("k",), ("q", "v"), ("k", "o")):
        with pytest.raises(ValueError, match="inert"):
            inject_lora(
                copy.deepcopy(forecaster.model),
                LoraConfig(axes=("variate",), targets=bad_targets, allow_single_row=True),
                rows_per_series=1,
            )

    # With allow_single_row=True, v/o/gate are permitted.
    ok_model = copy.deepcopy(forecaster.model)
    inject_lora(
        ok_model,
        LoraConfig(axes=("variate",), targets=("v", "o", "gate"), allow_single_row=True),
        rows_per_series=1,
    )
    assert len(lora_parameters(ok_model)) > 0

    # Multi-row data (e.g. target + covariates) allows q and k without any override.
    multi_model = copy.deepcopy(forecaster.model)
    inject_lora(multi_model, LoraConfig(axes=("variate",), targets=("q", "v")), rows_per_series=3)
    assert len(lora_parameters(multi_model)) > 0
