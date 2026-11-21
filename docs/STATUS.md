# Project Status

**Last updated:** 2026-11-21
**Stage:** pre-alpha. Phase 0 and Phase 1 are built, verified and merged into `main`.
**Next phase:** Phase 2 (Adaptation & Calibration), not started. See spec §10.

## What works today

| Area | State |
|---|---|
| Loading | `load_model` / `load_forecaster`: pinned Hub revisions, config from the checkpoint, any device |
| Devices | CPU (PyTorch 2.4+), Apple GPU (PyTorch 2.8+, fp32), CUDA (verified once: NVIDIA L4, PyTorch 2.15.0, fp32) |
| Data | `TimeSeriesDataset` from polars, pandas or pyarrow; point-in-time windows; time splits; `DHFDataCollator`; per-window categorical encoding |
| Training forward and loss | Normalised-space forward for both layouts, both target frames, pinball loss, `compute_loss` |
| Training | `TimestralTrainer` in `head_only` mode, fp32: AdamW, warm-up, cosine or constant schedule, clipping, evaluation, early stopping, best-state restore |
| Verifiers | `timestral_tune.utils.run_checks`: parity, label isolation, firewall, known-future sensitivity, loss mask |
| Acceptance tests | **10 of 14 implemented and passing**: T1–T8, T12, T14. T9, T10, T11, T13 are Phase 2 placeholders |
| Test suite | 438 passed, 4 skipped, on the tiny random model, 5M and 28M (PyTorch 2.15.0 on Python 3.13; also 2.14.1 on Python 3.10 and 2.8.0 on 3.12; 349 passed, 26 skipped on 2.4.0; macOS). On CUDA, at an earlier commit: 290 passed, 6 skipped |
| CI | Green on Phase 1 (pull request #1, 2026-11-21): lint, build, and the network-free tests on Python 3.10–3.13, on Linux |

## What does not exist yet

- LoRA modes and `full_ft` (the trainer raises `NotImplementedError`).
- `bf16` autocast and `torch.compile`.
- `export()`: there is no way to save a tuned model in the upstream layout yet.
- Conformal calibration, metrics, replay, the Lacuna integration.

## What has been verified, and where

All on one machine (Apple M4, macOS); details in `docs/VERIFICATION.md`.

- Training inputs and forward equal upstream inference (T1–T3, collator and dataset parity tests).
- No leakage found on the training path (T4–T7, T12, with one and with several targets) and an exact loss
  mask (T8). A review on 2026-11-21 found and fixed two leaks the earlier tests could not see: dataset-wide
  category numbering with several targets, and time-zone-aware timestamps from polars and pyarrow.
- The released checkpoints are calibrated in the median-folded target frame (decision D2 confirmed).
- Apple GPU parity with CPU, and gradient accuracy on both (T14).
- PyTorch floors: CPU on 2.4.0; Apple GPU from 2.8.0.
- `head_only` training lowers the loss and changes nothing outside the head, on CPU and the Apple GPU.

Not verified anywhere: the 123M and 209M checkpoints, real datasets, `bf16`. Everything changed after the
CUDA run (T14's `head_only` check, the past-only-covariate scenario, the review fixes, the float64-RMSNorm
gradient reference) has run on CPU and the Apple GPU only. CI covers the network-free tests on Linux; the tests on the released
checkpoints run only locally.

## Next steps

1. **Start Phase 2** (spec §10) when the maintainer says so: the slice LoRA adapter (T9, T10), `full_ft`, the
   conformal calibrators (T13), export (T11), `bf16` on CUDA.
2. **Re-run `bash scripts/cuda_check.sh`** on a GPU machine early in Phase 2: nothing since commit `ecbab9d`
   has run on CUDA.
3. **Decide the attention kernel** once a mode trains body weights (spec D15); the held option is on
   `fix/math-sdpa-backend`.

## Open questions and known issues

Full list in spec §12. The ones that matter next:

- **CUDA needs a C compiler and Python headers** with recent PyTorch (spec §12). `scripts/cuda_check.sh`
  checks for them.
- **Float32 gradients through released weights are not at round-off**, worst on CUDA (up to 1.2e-2).
  The released checkpoints have peak attention logits of 370k–404k (5M) and 203k–503k (28M), where float32
  ULP is 0.015625 below 262k and 0.03125 above it, triggering both fused-SDPA backward inaccuracy and ULP
  quantization on near-tied large-logit rows (`docs/VERIFICATION.md` §6, §13). PyTorch's unfused
  kernel cuts the CPU error on typical batches (2.5×–79× on the probe layouts) but is worse on one batch,
  changes nothing in `head_only`, and is untested on CUDA, so no choice of kernel is exposed yet (spec D15).
- **Re-run `bash scripts/cuda_check.sh` on a GPU machine** before relying on CUDA for Phase 2: the CUDA
  gradient numbers predate the improved reference, the review fixes, and `attention_context("math")`.
- **The target-frame evidence is synthetic** (univariate, fully observed, one seed, 5M and 28M).
- **Hyperparameter defaults are placeholders** until the Phase 3 sweeps.
- **No export yet**, so a trained model lives only in memory (`trainer.model`).
- **PyTorch 2.4.0 on macOS hangs at exit when a DataLoader uses worker processes.** A PyTorch issue, gone
  by 2.8.0 (`docs/VERIFICATION.md` §10). `num_workers` defaults to 0.
- **Validation loss under `median_folded` is in units that move with the model** (spec §12). Use it as a
  training signal, not for model selection across runs, until the raw-space metrics exist.
- **CI actions use a deprecated Node.js version.** GitHub warns that `actions/checkout@v4` and
  `astral-sh/setup-uv@v5` target Node.js 20. They still run.

## Branch state

- `main`: Phases 0 and 1.
- `fix/math-sdpa-backend`: a public `attention_backend` option and its plumbing through the config, trainer
  and verifiers, held until Phase 2 can show whether it is needed (spec D15). Its measurements and probe
  work are already in `main`.

Branch policy is in `AGENTS.md` §6.
