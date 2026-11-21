"""Unit tests for LoraConfig and Slice LoRA (spec §3.2, §5.1)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from timestral_tune import LoraConfig
from timestral_tune._vendor.timestral.layers import Attention
from timestral_tune.models import (
    LoRASliceLinear,
    extract_lora_state_dict,
    has_lora,
    inject_lora,
    load_lora,
    load_lora_state_dict,
    lora_parameters,
    merge_lora,
    merged_state_dict,
    training_forward,
    unmerge_lora,
)

from .conftest import available_accelerators
from .helpers import make_raw, max_rel, upstream_batch

DEVICES = ["cpu", *available_accelerators()]


def test_lora_config_validation_and_round_trip():
    cfg = LoraConfig(r=np.int64(4), lora_alpha=8, axes=["variate"], targets=["q", "v"], dropout=0.1)
    assert cfg.r == 4 and type(cfg.r) is int
    assert cfg.lora_alpha == 8.0 and cfg.scaling == 2.0
    assert cfg.axes == ("variate",) and cfg.targets == ("q", "v")
    assert LoraConfig.from_dict(cfg.to_dict()) == cfg

    for kwargs in (
        {"r": 0},
        {"r": True},
        {"lora_alpha": 0.0},
        {"lora_alpha": float("nan")},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"axes": ()},
        {"axes": "variate"},
        {"axes": ("spatial",)},
        {"axes": ("time", "time")},
        {"targets": ()},
        {"targets": ("q_proj",)},
        {"targets": ("q", "q")},
        {"allow_single_row": 1},
    ):
        with pytest.raises(ValueError):
            LoraConfig(**kwargs)
    with pytest.raises(ValueError, match="unknown"):
        LoraConfig.from_dict({"r": 4, "extra": 1})


def test_parameter_counts_match_spec_formula(tiny_forecaster):
    """Trainable parameters equal `modules * slices * 2 * r * D` (spec §5.1)."""
    d = tiny_forecaster.model.config.d_model
    attns = [m for m in tiny_forecaster.model.modules() if isinstance(m, Attention)]
    n_time = sum(1 for m in attns if m.is_time)
    n_variate = sum(1 for m in attns if not m.is_time)

    for axes, n_mod in ((("variate",), n_variate), (("time",), n_time), (("time", "variate"), n_time + n_variate)):
        for targets in (("q", "v"), ("q", "k", "v", "o", "gate")):
            m = copy.deepcopy(tiny_forecaster.model)
            inject_lora(m, LoraConfig(r=4, lora_alpha=8, axes=axes, targets=targets))
            expected = n_mod * len(targets) * 2 * 4 * d
            actual = sum(p.numel() for p in lora_parameters(m))
            assert actual == expected
            assert has_lora(m)


def test_slice_isolation_modifies_only_targeted_qkv_rows(tiny_forecaster):
    """Adapting only `'v'` changes only rows `[2D, 3D)` of `wQKV.weight`, leaving Q and K bit-identical."""
    model = copy.deepcopy(tiny_forecaster.model)
    d = model.config.d_model
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    inject_lora(model, LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("v",)))
    with torch.no_grad():
        for p in lora_parameters(model):
            p.normal_(std=0.1)
    merged = merged_state_dict(model)
    for name, orig in before.items():
        if name.endswith(".wQKV.weight") and ".attns.1." in name:
            assert torch.equal(merged[name][: 2 * d], orig[: 2 * d]), "Q/K rows changed when only V was adapted"
            assert not torch.equal(merged[name][2 * d :], orig[2 * d :]), "V rows did not change"
        else:
            assert torch.equal(merged[name], orig), f"non-targeted weight {name} changed"


def test_merge_and_unmerge_are_idempotent_and_non_destructive(tiny_forecaster):
    model = copy.deepcopy(tiny_forecaster.model).eval()
    inject_lora(model, LoraConfig(r=4, lora_alpha=8, axes=("time", "variate"), targets=("q", "k", "v", "o", "gate")))
    with torch.no_grad():
        for p in lora_parameters(model):
            p.normal_(std=0.05)

    # merged_state_dict must not flip `.merged` on any adapter.
    state_before = merged_state_dict(model)
    assert all(not m.merged for m in model.modules() if isinstance(m, LoRASliceLinear))

    merge_lora(model)
    state_once = merged_state_dict(model)
    merge_lora(model)  # idempotent
    state_twice = merged_state_dict(model)
    for k in state_before:
        assert torch.equal(state_before[k], state_once[k])
        assert torch.equal(state_once[k], state_twice[k])

    unmerge_lora(model)
    unmerge_lora(model)  # idempotent
    assert all(not m.merged for m in model.modules() if isinstance(m, LoRASliceLinear))

    # Adapter extraction holds only A/B tensors with expected shapes.
    adapters = extract_lora_state_dict(model)
    assert len(adapters) == len(lora_parameters(model))
    assert all(".lora_A." in k or ".lora_B." in k for k in adapters)


def test_inject_lora_guards_against_double_injection_and_zero_matches(tiny_forecaster):
    model = copy.deepcopy(tiny_forecaster.model)
    inject_lora(model, LoraConfig(axes=("variate",)))
    with pytest.raises(ValueError, match="already injected"):
        inject_lora(model, LoraConfig(axes=("time",)))

    import dataclasses

    from timestral_tune.models import TuneTimestralModel

    time_only = TuneTimestralModel(dataclasses.replace(tiny_forecaster.config, block_schedule=("time", "time")))
    with pytest.raises(ValueError, match="matched no attention"):
        inject_lora(time_only, LoraConfig(axes=("variate",)))


@pytest.mark.parametrize("device", DEVICES)
def test_lora_backward_and_dropout_on_device(tiny_forecaster, device):
    patch = tiny_forecaster.patch_size
    raw = make_raw(23, targets=1, past=1, known=1, context_len=6 * patch, horizon=2 * patch)
    batch, _, _ = upstream_batch(tiny_forecaster, raw)
    model = copy.deepcopy(tiny_forecaster.model).to(device).train()
    model.transformer.rope.rebuild_inv_freq()
    for p in model.parameters():
        p.requires_grad_(False)
    inject_lora(
        model, LoraConfig(r=4, lora_alpha=8, axes=("time", "variate"), targets=("q", "v", "o", "gate"), dropout=0.1)
    )

    # Gradients flow into B (non-zero at step 0 because A is non-zero) and never into frozen base weights.
    out = training_forward(model, batch.to(device)).quantiles
    out.pow(2).mean().backward()
    for name, p in model.named_parameters():
        if ".lora_" in name:
            assert p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
            if ".lora_B." in name:
                assert p.grad.abs().sum().item() > 0
        else:
            assert not p.requires_grad and p.grad is None

    # Eval mode disables dropout, so repeated calls are bit-identical even with non-zero B.
    with torch.no_grad():
        for p in lora_parameters(model):
            p.normal_(std=0.05)
    model.eval()
    with torch.no_grad():
        a = training_forward(model, batch.to(device)).quantiles
        b = training_forward(model, batch.to(device)).quantiles
    assert torch.equal(a, b)
    merge_lora(model)
    with torch.no_grad():
        c = training_forward(model, batch.to(device)).quantiles
    assert max_rel(c, a) < 1e-5


def test_k_only_and_gate_only_slice_isolation_and_generic_slice_fallback(tiny_forecaster):
    """Middle slice (`'k'`), `'gate'`-only, and arbitrary non-QKV slices each modify only their own rows."""
    model = copy.deepcopy(tiny_forecaster.model)
    d = model.config.d_model
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    inject_lora(model, LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("k",)))
    with torch.no_grad():
        for p in lora_parameters(model):
            p.normal_(std=0.1)
    merged = merged_state_dict(model)
    for name, orig in before.items():
        if name.endswith(".wQKV.weight") and ".attns.1." in name:
            assert torch.equal(merged[name][:d], orig[:d]), "Q rows changed under k-only LoRA"
            assert not torch.equal(merged[name][d : 2 * d], orig[d : 2 * d]), "K rows did not change"
            assert torch.equal(merged[name][2 * d :], orig[2 * d :]), "V rows changed under k-only LoRA"
        else:
            assert torch.equal(merged[name], orig)

    # Gate-only injection adapts only _gate weights.
    gate_model = copy.deepcopy(tiny_forecaster.model)
    inject_lora(gate_model, LoraConfig(r=4, lora_alpha=8, axes=("time",), targets=("gate",)))
    wrapped_names = [n for n, m in gate_model.named_modules() if isinstance(m, LoRASliceLinear)]
    assert wrapped_names and all(n.endswith("._gate") for n in wrapped_names)

    # Generic fallback path (arbitrary non-QKV slice bounds).
    lin = torch.nn.Linear(16, 20, bias=False)
    wrapped = LoRASliceLinear(lin, [("mid", 4, 6)], r=3, lora_alpha=6.0)
    assert not wrapped._is_full and not wrapped._is_qkv
    with torch.no_grad():
        wrapped.lora_B["mid"].normal_(std=0.2)
    x = torch.randn(2, 5, 16)
    unmerged = wrapped(x)
    wrapped.merge()
    merged_out = wrapped(x)
    torch.testing.assert_close(merged_out, unmerged, rtol=1e-5, atol=1e-6)


def test_unmerge_restores_exact_base_weights_even_after_mutation_and_in_bf16(tiny_forecaster):
    """Snapshotted pre-merge rows prevent base corruption on merge->mutate->unmerge and bf16 drift."""
    for dtype in (torch.float32, torch.bfloat16):
        lin = torch.nn.Linear(32, 96, bias=False).to(dtype)
        orig_weight = lin.weight.detach().clone()
        wrapped = LoRASliceLinear(lin, [("q", 0, 32), ("v", 64, 32)], r=4, lora_alpha=8.0)
        with torch.no_grad():
            for p in wrapped.lora_B.values():
                p.normal_(std=0.1)
        wrapped.merge()
        # Mutate LoRA parameters while merged: unmerge must still restore original base weights bit-identically.
        with torch.no_grad():
            for p in wrapped.lora_B.values():
                p.add_(1.0)
        wrapped.unmerge()
        assert torch.equal(wrapped.base.weight, orig_weight), f"base weight drifted in {dtype}"


def test_load_lora_round_trips_exported_adapters_and_rejects_mismatches(tiny_forecaster, tmp_path):
    from timestral_tune import export_checkpoint

    patch = tiny_forecaster.patch_size
    raw = make_raw(29, targets=1, past=1, known=1, context_len=6 * patch, horizon=2 * patch)
    batch, _, _ = upstream_batch(tiny_forecaster, raw)

    source = copy.deepcopy(tiny_forecaster.model).eval()
    cfg = LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("q", "v", "o"))
    inject_lora(source, cfg, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        for p in lora_parameters(source):
            p.normal_(std=0.05)
        expected = training_forward(source, batch).quantiles

    out_dir = export_checkpoint(source, tmp_path / "lora-exp", lora_config=cfg)
    fresh = copy.deepcopy(tiny_forecaster.model).eval()
    torch.manual_seed(555)
    rng_before = torch.get_rng_state()
    reloaded, reloaded_cfg = load_lora(fresh, out_dir, rows_per_series=3)
    assert torch.equal(torch.get_rng_state(), rng_before), "load_lora must not mutate global PyTorch RNG"
    assert reloaded_cfg == cfg
    with torch.no_grad():
        actual = training_forward(reloaded, batch).quantiles
    assert torch.equal(actual, expected)

    # Strict key/shape validation in load_lora_state_dict.
    bad_state = extract_lora_state_dict(source)
    bad_key = next(iter(bad_state))
    bad_state[bad_key] = bad_state[bad_key][:-1]
    with pytest.raises(ValueError, match="shape mismatch"):
        load_lora_state_dict(reloaded, bad_state)
    with pytest.raises(ValueError, match="key mismatch"):
        load_lora_state_dict(reloaded, {})
    with pytest.raises(ValueError, match="duplicate slice name"):
        LoRASliceLinear(torch.nn.Linear(16, 48, bias=False), [("q", 0, 16), ("q", 16, 16)], r=2, lora_alpha=4.0)
