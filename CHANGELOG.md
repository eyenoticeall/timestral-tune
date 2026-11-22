# Changelog

All notable changes to `timestral-tune` are listed here. The project follows
[Semantic Versioning](https://semver.org/); while the version is 0.x, the public API may change between
minor releases, and such changes are called out under **Changed**.

## 0.1.0 — 2026-11-23

First public release. It adapts the released Timestral checkpoints (5M, 28M, 123M and 209M) and vendors
Mistral AI's `timestral` inference package at commit `a534bdb`.

### Added

- **Loading** of pinned checkpoints from the Hugging Face Hub on CPU, CUDA and the Apple GPU
  (`load_model`, `load_forecaster`).
- **Data pipeline**: `TimeSeriesDataset` from polars, pandas or pyarrow frames, with target, past-only and
  known-future roles, point-in-time windows, time splits with an embargo, and `DHFDataCollator`.
- **Training** in normalised space with the paper's pinball loss, equal to the official inference path
  (`TimestralTrainer`, `TrainingConfig`, `compute_loss`).
- **Adaptation modes**: `head_only` (with frozen-feature caching), `lora_variate`, `lora_temporal`
  (`LoraConfig`, mergeable slice LoRA) and `full_ft`, with sign-flip augmentation and sweep-backed default
  learning rates.
- **CUDA acceleration**: `bf16` autocast, fused AdamW and optional `torch.compile`.
- **Conformal calibration**: `ConformalCalibrator` (split CQR) and `OnlineConformalCalibrator` (ACI).
- **Anti-forgetting**: `ReplayConfig`, `SyntheticReplaySource` and `distillation_loss` for replay mixing
  and distillation towards the frozen base model.
- **Export** of checkpoints that load in the official package with `strict=True`, plus unmerged LoRA
  adapters (`load_lora`), `conformal.json` and a `tune_metadata.json` sidecar (`export_checkpoint`).
- **Evaluation**: raw-space pinball loss, weighted quantile loss, MASE, coverage and a forgetting metric
  (`timestral_tune.utils`), and `run_checks` for leakage and parity checks on your own data.
- **Optional Lacuna integration** (`timestral-tune[lacuna]`, Python 3.11+).

### Verified on

- CPU with PyTorch 2.4, 2.8, 2.14 and 2.15 on Python 3.10 to 3.13; one Apple M4 (PyTorch 2.8 and 2.15,
  `fp32`); one NVIDIA L4 (PyTorch 2.15, `fp32` and `bf16`). Details and limits are in
  `docs/VERIFICATION.md` and `docs/SPEC.md` §12.

### Known issues

- `compile_model=True` on PyTorch 2.4.0 with Python older than 3.12 needs `setuptools` installed; that
  PyTorch release declares the dependency only from Python 3.12 (seen on Python 3.10).
- PyTorch 2.4.0 on macOS can hang at exit when a DataLoader uses worker processes; `num_workers`
  defaults to 0.
- Multi-GPU training is untested.
