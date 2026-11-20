# Phase 3 — Benchmarks & Documentation

**Status:** Not started
**Goal:** Evidence that the modes work, defaults backed by sweeps, the Lacuna integration, and a first release.
**Spec sections:** `docs/SPEC.md` §5.3 (forgetting metric), §5.4, §5.5, §7.5, §8, §10, §12

The task checklist and exit criteria are in the spec, §10. This file holds working notes for whoever
implements the phase. Add to it as you learn things; do not copy the checklist here.

## Notes for the implementer

- **Evaluate through the vendored forecaster with its defaults**, so tuned and zero-shot numbers share one
  code path.
- **Contamination.** The paper does not list the real datasets used in post-training, so any public
  benchmark may have been seen. State this with every result.
- **Lacuna.** Use `lacuna.ExperimentRegistry.record` / `record_selection` and the default column names of
  `lacuna.SignalStudy` (`time`, `instrument`, `signal`), as of `lacuna-quant` 0.14.1. Re-check its API at
  the time of implementation.
- **Release.** The PyPI names `timestral-tune` and `timestral` were both unclaimed on 2026-11-21.

## Open questions

- Benchmark datasets are not chosen; licences must be checked first.
- Whether sign-flip augmentation helps (spec §5.3).

## Log of what was done

Add dated bullets here as work lands, with the evidence (which tests pass, on what).
