# Phase 2 — Adaptation & Calibration

**Status:** Not started
**Goal:** All five adaptation modes, both conformal calibrators, and exports that load in the official package.
**Spec sections:** `timestral_tune_handoff_document.md` §3.2, §5.1, §5.2, §7.3, §7.4, §9 (T9–T11, T13), §10

The task checklist and exit criteria are in the handoff, §10. This file holds working notes for whoever
implements the phase. Add to it as you learn things; do not copy the checklist here.

## Notes for the implementer

- **LoRA slices.** In `wQKV.weight`, rows `[0, D)` are query, `[D, 2D)` key, `[2D, 3D)` value. The
  query/key order is from the code only; the value third is measured (`docs/VERIFICATION.md` §3).
- **Adapter naming.** Wrapping a linear changes parameter names, so always export merged weights under
  upstream names. T11 guards this.
- **Single-row guard** is measured, not theoretical: query and key perturbations change nothing on
  univariate input.
- **Conformal formulas** are quoted from the papers in handoff §5.2; the PDFs are in `papers/` locally.
- **`config.json` must stay byte-compatible** with upstream; metadata goes in `tune_metadata.json`.

## Open questions

- Default LoRA rank and which slices to enable by default (settled by Phase 3 sweeps).

## Log of what was done

Add dated bullets here as work lands, with the evidence (which tests pass, on what).
