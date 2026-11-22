# Phase 3 — Benchmarks & Documentation

**Status:** Complete and merged into `main` (2026-11-22, fast-forwarded from `phase-3` at `9c5817c`; PyPI release pending maintainer approval).
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
- 2026-11-22 (`604d8d5`..`3f12fe2`, `cf0520c`): built `scripts/benchmarks/{datasets,run_sweeps,run_benchmarks,significance}.py`,
  ran 5M/28M sweeps + benchmarks, `123M`/`209M` CUDA gradient probes & benchmarks (`docs/VERIFICATION.md` §13),
  and 28M multi-seed / clustered CI / classical baseline checks (`docs/BENCHMARKS.md` §4).
- 2026-11-22 (`55d57d2`, `1185f82`, `7ebe45d`, `2556a7e`): closed the remaining 4 mutation-test gaps,
  added `load_signed_returns()` (`logret_20`) and full multi-seed CLI flags to `run_sweeps.py`, ran the
  144-run 3-seed `123M`/`209M` decision sweep (`sweeps_large_decisions.json`), adopted Decisions 1–3
  (`sign_flip=True` with dual-orientation `+x`/`-x` `head_only` cache, `lora_temporal=1e-3`,
  `full_ft=1e-4`, D19), unified `sweeps_bracketed.json` (dropping `sweeps_extra.json`), re-ran all
  benchmarks, fresh train-only precision ablation, and `sig_v2.json` at shipped defaults on L4 (VM
  deleted immediately after), and added `docs/VERIFICATION.md` §14 backing all `[V]` claims.
- 2026-11-22 (`40d29fe`, `542568c`, and final CUDA verification): closed the last 3 mutation-test gaps
  (staggered ACI origin order, 2× dual-cache memory cap, unflipped post-`fit()` training evaluation),
  documented the maintainer decision keeping `full_ft=1e-4` as the single cross-size default while
  recommending `3e-4` for `full_ft` on `5M` and `28M`, and verified `bash scripts/cuda_check.sh` on NVIDIA
  L4 at `542568c`: **255 passed, 2 skipped** (device tests) and **538 passed, 3 skipped** (whole suite, exit 0;
  VM deleted immediately after).
