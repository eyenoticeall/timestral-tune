# Project Status

**Last updated:** 2026-11-21
**Stage:** pre-alpha. Phases 0, 1, and 2 are built and verified (`phase-2` branch off `main`).
**Next phase:** Phase 3 (Benchmarks, metrics, Lacuna integration, sweeps). See spec §10.

## What works today

| Area | State |
|---|---|
| Loading | `load_model` / `load_forecaster`: pinned Hub revisions, config from the checkpoint, any device |
| Devices | CPU (PyTorch 2.4+), Apple GPU (PyTorch 2.8+, fp32), CUDA (verified on NVIDIA L4, PyTorch 2.15.0, fp32 and `bf16`; commit `b0980cb`) |
| Data | `TimeSeriesDataset` from polars, pandas or pyarrow; point-in-time windows; time splits; `DHFDataCollator`; per-window categorical encoding |
| Training forward and loss | Normalised-space forward (trimmed `L + 2h` covariate layout), both target frames, pinball loss, `compute_loss` |
| Adaptation & training | `TimestralTrainer` across `head_only` (with frozen-feature caching), `lora_variate` (with single-row guard), `lora_temporal`, and `full_ft`; fp32 everywhere, `bf16` + fused AdamW on CUDA, optional `torch.compile`, early stopping, best-state restore |
| Conformal calibration | `ConformalCalibrator` (static split CQR) and `OnlineConformalCalibrator` (online ACI) with scale normalisation, horizon buckets, outward monotonicity repair, and JSON persistence |
| Export | `export_checkpoint` and `TimestralTrainer.export()`: upstream-loadable `config.json` + merged `model.safetensors`, unmerged LoRA adapters, `conformal.json`, `tune_metadata.json`, `LICENSE`, `NOTICE` |
| Verifiers | `timestral_tune.utils.run_checks`: parity, label isolation, firewall, known-future sensitivity, loss mask |
| Acceptance tests | **14 of 14 implemented and passing**: T1–T14 (zero placeholders remaining) |
| Test suite | 519 passed, 4 skipped (CUDA-only `bf16`), on tiny, 5M and 28M (PyTorch 2.15.0 on 3.13; earlier commit `3fe5bc9`: 510/4 on 2.14.1 and 2.8.0, 411/26 on 2.4.0; macOS). On NVIDIA L4 CUDA (Linux, `b0980cb`): **521 passed, 2 skipped** |
| CI | Green on `main` (`fc26d03`, Python 3.10–3.13 on Linux); `phase-2` is pushed to `origin/phase-2` and awaits a PR/merge for CI |

## What does not exist yet

- Phase 3 raw-space evaluation metrics (`utils/metrics.py`), benchmark scripts, LR/rank sweeps, and Lacuna integration (`integrations/lacuna.py`).
- Phase 4 replay mixing, built-in generators, distillation (`replay/`), and tiled training.

## What has been verified, and where

All locally on Apple M4 (macOS, PyTorch 2.4/2.8/2.14/2.15) and on an NVIDIA L4 (Linux, PyTorch 2.15.0+cu132); details in `docs/VERIFICATION.md`.

- Training inputs and forward equal upstream inference (T1–T3, collator and dataset parity tests).
- No leakage found on the training path (T4–T7, T12, with one and with several targets) and an exact loss mask (T8).
- Slice LoRA step-0 identity, merge/unmerge parity (`< 1e-12` fp64, `< 1e-4` fp32), and single-row guard (T9, T10).
- Upstream `strict=True` export round-trip across all four modes on tiny, 5M, and 28M, plus verbatim Hub `LICENSE` copy (T11).
- Static CQR and online ACI conformal coverage, monotonicity repair, saturation, and persistence (T13).
- Device parity and gradients on CPU, Apple GPU, and CUDA (T14), plus `bf16` autocast and `torch.compile` on L4 CUDA (3.3× faster 28M steps, 45% less VRAM).
- Decision D15 resolved on CUDA: 20-step training trajectories under `fp32` default SDPA, `fp32` unfused math, and `bf16` match closely in validation loss across `lora_variate`, `lora_temporal`, and `full_ft`.

Not verified anywhere: the 123M and 209M checkpoints, real datasets, multi-GPU.

## Next steps

1. **Merge `phase-2` into `main`** once reviewed and let CI run on Linux Python 3.10–3.13.
2. **Start Phase 3** (spec §10): raw-space evaluation metrics (`utils/metrics.py`), real-dataset benchmarks (realised volatility + retail demand), forgetting metric, LR/rank sweeps to replace placeholder defaults, and Lacuna integration.

## Open questions and known issues

Full list in spec §12. The ones that matter next:

- **CUDA needs a C compiler and Python headers** with recent PyTorch (spec §12). `scripts/cuda_check.sh` checks for them.
- **The target-frame evidence is synthetic** (univariate, fully observed, one seed, 5M and 28M).
- **Hyperparameter defaults are placeholders** until the Phase 3 sweeps.
- **PyTorch 2.4.0 on macOS hangs at exit when a DataLoader uses worker processes.** Gone by 2.8.0 (`docs/VERIFICATION.md` §10). `num_workers` defaults to 0.
- **Validation loss under `median_folded` is in units that move with the model** (spec §12). Use it as a training signal, not for model selection across runs, until Phase 3 raw-space metrics exist.
- **CI actions use a deprecated Node.js version.** GitHub warns that `actions/checkout@v4` and `astral-sh/setup-uv@v5` target Node.js 20. They still run.

## Branch state

- `main`: Phases 0 and 1 (including the trimmed covariate layout and `head_only` caching), pushed to
  `origin/main` (`fc26d03`). `perf/phase1-kernels` (merged into `main`) and `fix/math-sdpa-backend`
  (closed by D15) have been deleted locally and on `origin`.
- `phase-2`: all of Phase 2 (LoRA, `full_ft`, CQR/ACI conformal calibration, upstream-loadable export,
  `bf16` + `torch.compile` on CUDA, T9–T11 & T13 passing), verified locally and on NVIDIA L4; pushed to
  `origin/phase-2`, not yet merged into `main`.

Branch policy is in `AGENTS.md` §6.
