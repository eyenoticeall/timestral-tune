# Phase 0 — Foundations

**Status:** In progress
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
- **Core dependencies** are exactly upstream's four. Dataframe libraries will be extras, added in Phase 1.
- **Still to do:** `models/loader.py`, `models/device.py`, acceptance test T14.
- **Device shims.** The prototype (`install_cpu_offload`) patches three functions, which also covers
  `_cap_quantiles` because it patches the module-level statistics function. The library version uses
  subclasses and must override four methods: model `scale`, and forecaster `_cap_quantiles`,
  `_refine_block`, `_encode_categorical`.
- **Loader.** Build the config with `TimestralConfig.load` from the checkpoint directory, never from
  defaults. After moving a model between devices, call `model.transformer.rope.rebuild_inv_freq()` as
  upstream's loader does.

## Open questions

- Whether MPS needs a higher PyTorch floor than 2.4 (untested below 2.15.0).

## Log of what was done

Add dated bullets here as work lands, with the evidence (which tests pass, on what).

- 2026-11-21: Vendored upstream at `a534bdb`; package skeleton, packaging, scaffold tests, probes, CI
  workflow files and process docs added. Evidence: 6 scaffold tests pass on Python 3.10 and 3.13; wheel
  builds with the vendored licence inside; drift check reports no drift. CI itself has not run.
- 2026-11-21: Pushed `main`; first CI run green (lint, build, tests on Python 3.10–3.13).
