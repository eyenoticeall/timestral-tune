# Project Status

**Last updated:** 2026-11-21
**Stage:** pre-alpha. Phase 0 and Phase 1 are built and verified.
**Next phase:** Phase 2 (Adaptation & Calibration). See spec §10.

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
| Test suite | 292 passed, 4 skipped, on the tiny random model, 5M and 28M (PyTorch 2.15.0 on Python 3.13; also 2.14.1 on Python 3.10 and 2.8.0 on 3.12; macOS). On CUDA: 290 passed, 6 skipped |
| CI | Green on the last push (scaffold only). The Phase 1 work has not been pushed, so CI has not run on it |

## What does not exist yet

- LoRA modes and `full_ft` (the trainer raises `NotImplementedError`).
- `bf16` autocast and `torch.compile`.
- `export()`: there is no way to save a tuned model in the upstream layout yet.
- Conformal calibration, metrics, replay, the Lacuna integration.

## What has been verified, and where

All on one machine (Apple M4, macOS); details in `docs/VERIFICATION.md`.

- Training inputs and forward equal upstream inference (T1–T3, collator and dataset parity tests).
- No leakage on the training path (T4–T7, T12) and an exact loss mask (T8).
- The released checkpoints are calibrated in the median-folded target frame (decision D2 confirmed).
- Apple GPU parity with CPU, and gradient accuracy on both (T14).
- PyTorch floors: CPU on 2.4.0; Apple GPU from 2.8.0.
- `head_only` training lowers the loss and changes nothing outside the head, on CPU and the Apple GPU.

Not verified anywhere: the 123M and 209M checkpoints, real datasets, `bf16`. CI has run only the scaffold.

## Next steps

1. **Push `phase-1` and let CI run** on the new tests (it has only ever run on the scaffold).
2. **Start Phase 2** (spec §10): the slice LoRA adapter (T9, T10), `full_ft`, the conformal calibrators
   (T13), export (T11), `bf16` on CUDA.

## Open questions and known issues

Full list in spec §12. The ones that matter next:

- **CUDA needs a C compiler and Python headers** with recent PyTorch (spec §12). `scripts/cuda_check.sh`
  checks for them.
- **Float32 gradients through released weights are not at round-off**, worst on CUDA (up to 1.2e-2).
  Mostly the backward pass of upstream's attention call (`docs/VERIFICATION.md` §6, §13). Training is not
  expected to be affected, but that is untested; T14 has three tiers because of it.
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

- `main`: the scaffold, the spec rename and the README banner. Two banner commits are ahead of `origin`
  and not pushed.
- `phase-1`: Phase 0's remainder and all of Phase 1, rebased on `main`. Not merged, not pushed.

Branch policy is in `AGENTS.md` §6.
