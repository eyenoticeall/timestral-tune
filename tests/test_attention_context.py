"""Tests for `attention_context`, the experimental switch between PyTorch's attention kernels.

It is a measurement tool for acceptance test T14 and the gradient-accuracy probe (spec §12), not part of
the public API.
"""

from __future__ import annotations

import threading

import pytest
import torch

from timestral_tune.models import training_forward
from timestral_tune.models.forward import attention_context

from .helpers import make_raw, upstream_batch

MATH_ONLY = (True, False, False, False)


def sdpa_flags() -> tuple[bool, bool, bool, bool]:
    return (
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    )


def test_math_disables_the_fused_kernels_and_restores_the_flags():
    before = sdpa_flags()
    with attention_context("default"):
        assert sdpa_flags() == before
    with attention_context("math"):
        assert sdpa_flags() == MATH_ONLY
        with attention_context("math"):
            assert sdpa_flags() == MATH_ONLY
        assert sdpa_flags() == MATH_ONLY, "leaving a nested block must not restore the flags early"
    assert sdpa_flags() == before

    with pytest.raises(RuntimeError, match="boom"), attention_context("math"):
        raise RuntimeError("boom")
    assert sdpa_flags() == before, "an exception inside the block must still restore the flags"

    with pytest.raises(ValueError, match="unknown attention backend"), attention_context("flash"):
        pass


def test_overlapping_threads_restore_the_original_flags():
    """Out-of-order exits across threads (A enters, B enters, A exits, B exits) must not leave the fused
    kernels disabled, and A's exit must not re-enable them while B is still inside.
    """
    before = sdpa_flags()
    a_entered, b_entered, a_exited = threading.Event(), threading.Event(), threading.Event()
    observed_in_b: list[tuple[bool, bool, bool, bool]] = []

    def thread_a() -> None:
        with attention_context("math"):
            a_entered.set()
            b_entered.wait(timeout=5)
        a_exited.set()

    def thread_b() -> None:
        a_entered.wait(timeout=5)
        with attention_context("math"):
            b_entered.set()
            a_exited.wait(timeout=5)
            observed_in_b.append(sdpa_flags())

    threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert observed_in_b == [MATH_ONLY], "thread A's exit restored the flags while B was still inside"
    assert sdpa_flags() == before, "overlapping exits left the fused kernels disabled"


def test_the_model_runs_its_attention_under_the_selected_kernel(tiny_forecaster):
    """The switch reaches upstream's attention call, and the forward agrees between the two kernels."""
    patch = tiny_forecaster.patch_size
    raw = make_raw(17, targets=1, past=1, known=1, context_len=6 * patch + 1, horizon=2 * patch)
    batch, _, _ = upstream_batch(tiny_forecaster, raw)
    model = tiny_forecaster.model
    seen: list[tuple[bool, bool, bool, bool]] = []
    handle = model.transformer.layers[0].attns[0].register_forward_hook(lambda *_: seen.append(sdpa_flags()))
    try:
        before = sdpa_flags()
        with torch.no_grad():
            default = training_forward(model, batch).quantiles
            assert seen and all(flags == before for flags in seen)
            seen.clear()
            with attention_context("math"):
                math = training_forward(model, batch).quantiles
            assert seen and all(flags == MATH_ONLY for flags in seen)
        assert sdpa_flags() == before
        assert torch.allclose(default, math, rtol=1e-5, atol=1e-6)
    finally:
        handle.remove()
