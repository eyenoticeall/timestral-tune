# Phase 0 — Foundations

**Status:** Done, except that the weekly drift job has not had its first scheduled run
**Goal:** Make the repository buildable and the upstream dependency safe, so later phases only add logic.
**Spec sections:** `docs/SPEC.md` §3.5, §5.4 (devices), §6.1 (vendoring), §9 (T14), §10

The task checklist and exit criteria are in the spec, §10. This file holds working notes for whoever
implements the phase. Add to it as you learn things; do not copy the checklist here.

## Notes for the implementer

- **Vendoring is done.** `scripts/vendor_upstream.py` rewrites absolute sibling imports to relative ones;
  five of nine files are modified that way and carry a notice header. See `docs/UPSTREAM.md`.
- **Line offsets.** In those five vendored files, lines sit four lower than the `[U file:line]` references.
- **Packaging.** Hatchling, src layout, version read from `src/timestral_tune/__init__.py`. The wheel
  includes the vendored `LICENSE` and manifest.
- **Core dependencies** are exactly upstream's four. The data layer reads polars, pandas and pyarrow frames
  without importing any of them, so none is a dependency or an extra; all three are in the dev group for tests.
- **Loader and device shims are done** (`models/loader.py`, `models/device.py`). The shims are subclasses
  that override four methods (model `scale`; forecaster `_cap_quantiles`, `_refine_block`,
  `_encode_categorical`), move tensors to CPU and call the parent. On devices with float64 they call the
  parent directly.
- **The loader** builds the config with `TimestralConfig.load` from the checkpoint directory, never from
  defaults, and rebuilds the RoPE table on the target device as upstream's loader does.
- **The Apple GPU needs PyTorch 2.8** (`cummax`). `resolve_device` enforces it.

## Open questions

- None for this phase.

## Log of what was done

Add dated bullets here as work lands, with the evidence (which tests pass, on what).

- 2026-11-21: Vendored upstream at `a534bdb`; package skeleton, packaging, scaffold tests, probes, CI
  workflow files and process docs added. Evidence: 6 scaffold tests pass on Python 3.10 and 3.13; wheel
  builds with the vendored licence inside; drift check reports no drift. CI itself has not run.
- 2026-11-21: Pushed `main`; first CI run green (lint, build, tests on Python 3.10–3.13).
- 2026-11-21: Loader, device shims and T14 done. T14 passes on CPU and the Apple GPU on the tiny model, 5M
  and 28M. Apple GPU floor found to be PyTorch 2.8.
