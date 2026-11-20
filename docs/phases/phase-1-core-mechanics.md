# Phase 1 — Core Mechanics & Verification

**Status:** Not started
**Goal:** A training forward pass and loss that provably match upstream inference, a leak-free data layer, and the first end-to-end `head_only` run.
**Spec sections:** `docs/SPEC.md` §3.3, §3.4, §4, §6.2, §6.3, §9 (T1–T8, T12), §10

The task checklist and exit criteria are in the spec, §10. This file holds working notes for whoever
implements the phase. Add to it as you learn things; do not copy the checklist here.

## Notes for the implementer

- **Start with `models/forward.py` and T1–T3.** `norm_forward` in `scripts/probes/_common.py` is a
  working prototype for both layouts; `real_loss` shows the median-folded frame.
- **Parity target.** `sinh(normalised output) * anchor_scale + anchor_loc` must equal `predict_block`.
  For T3, build the forecaster with `quantile_real_cap_k=0`, or its output clamp will interfere.
- **Collator.** Mirror vendored `_assemble_rows`: forward-fill targets while keeping the mask, zero-fill
  covariate NaNs as unobserved, truncate to the context cap, left-pad to a patch multiple.
- **Frame statistics** run on CPU in float64 without gradients, also on GPU devices.
- **Perturbation tests** need non-constant noise (see `docs/DEVELOPMENT.md`).
- **Target-frame check.** Compare the pretrained model's normalised pinball loss and interval coverage
  under `median_folded` and `true_values` on horizons longer than one patch. Record the outcome in
  spec §4.3 and §12, and revise decision D2 if `true_values` is clearly better.

## Open questions

- Decision D2 (target frame) is unconfirmed.
- Which dataframe libraries to support first, and their version floors.

## Log of what was done

Add dated bullets here as work lands, with the evidence (which tests pass, on what).
