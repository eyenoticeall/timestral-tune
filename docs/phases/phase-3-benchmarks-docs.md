# Phase 3 — Benchmarks & Documentation

**Status:** Complete on branch `phase-3` (2026-11-22; PyPI release pending maintainer approval).
**Goal:** Evidence that the modes work, defaults backed by sweeps, the Lacuna integration, and a first release.
**Spec sections:** `docs/SPEC.md` §5.3 (forgetting metric), §5.4, §5.5, §7.5, §8, §10, §12

The task checklist and exit criteria are in the spec, §10. This file holds working notes for whoever
implements the phase. Add to it as you learn things; do not copy the checklist here.

## Notes for the implementer

- **Evaluate through the vendored forecaster with its defaults**, so tuned and zero-shot numbers share one
  code path (`utils/metrics.py::evaluate_forecaster`).
- **Datasets chosen (pinned revisions, permissive licences):**
  - Realised volatility: `twelvedata/financial-world-model@da186e5fd362` (MIT, 51 US equities, 1h bars,
    post-Oct 27, 2026 test split — zero pretraining contamination on test targets).
  - Retail demand: `Dingdong-Inc/FreshRetailNet-50K@c374e85975c8` (CC BY 4.0, top-64 active store-SKU
    hourly series, 2024 dates — potential post-training contamination noted in `docs/BENCHMARKS.md`).
  - General forgetting set: `Salesforce/GiftEval@30841734ac5c` (Apache-2.0, `ett1/H`, `jena_weather/H`,
    `LOOP_SEATTLE/H`).
- **Lacuna:** `LacunaExporter` uses `AttemptRecord.trial_id` (not `attempt_id`) in `record_selection`, and
  `write_signal_frame` emits `pyarrow.Table` / `polars.DataFrame` with `("time", "instrument", "signal")`
  verified live against `lacuna-quant==0.14.1`.
- **Sign-flip augmentation:** `TrainingConfig.sign_flip=True` bypasses the training `head_only` cache
  (keeping validation caching) and improved one-signed benchmarks (up to −4.0% wQL on retail `full_ft`).
- **Release:** Wheel + sdist verified with `uv build`; vendored Apache-2.0 `LICENSE` and `NOTICE` included.
  PyPI names `timestral-tune` and `timestral` were unclaimed on 2026-11-21; publish awaits maintainer action.

## Log of what was done

- 2026-11-22 (`0dc3f1a`): implemented raw-space metrics (`mase`, `weighted_quantile_loss`, `pinball_loss_raw`,
  `coverage`, `interval_width`, `evaluate_forecaster`, `evaluate_forgetting`) in `utils/metrics.py` + unit tests.
- 2026-11-22 (`8cbb7b2`): implemented `TrainingConfig.sign_flip` and `flip_batch_signs` + unit tests.
- 2026-11-22 (`f8b9c9c`): implemented `integrations/lacuna.py` (`LacunaExporter`, `write_signal_frame`) and
  verified live against `lacuna-quant==0.14.1` (`ExperimentRegistry` + `SignalStudy`).
- 2026-11-22 (`604d8d5` + L4 GPU run): built `scripts/benchmarks/{datasets,run_sweeps,run_benchmarks}.py`,
  ran LR/rank/targets/sign-flip sweeps on 5M & 28M and the full 5M + 28M 7-mode benchmark + forgetting matrix
  on an NVIDIA L4 VM (VM deleted immediately after), updated sweep-backed defaults in `config.py` and
  `SPEC.md` §5.4, and published `docs/BENCHMARKS.md`.
