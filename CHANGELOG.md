# Changelog

All notable changes to `timestral-tune` are listed here. The project follows
[Semantic Versioning](https://semver.org/); while the version is 0.x, the public API may change between
minor releases, and such changes are called out under **Changed**.

## Unreleased

### Added

- **Experimental multi-process training (Phase 5 Stages 1–2):** `TimestralTrainer` activates single-node
  data-parallel training automatically when a caller-owned `torch.distributed` process group (`"gloo"` or
  `"nccl"`, `world_size > 1`) is active, across `head_only`, `lora_variate`, `lora_temporal`, and `full_ft`,
  including `ReplayConfig` replay mixing and uncached frozen-base distillation (`DistributedBatchCycle` with
  shared replay epoch seed, main-process `num_workers=0` replay loader, per-rank `ReplayConfig.batch_size`
  resolution and rank-0 startup logging, and replay-only pre-check collective), count-weighted gradient
  reduction (`ReduceOp.SUM`), startup consistency checks and rank-0 weight/buffer broadcast, sharded
  validation inside `fit()`, rank-0 callbacks/early-stopping/restore with identical `TrainingResult` fields
  across ranks, and best-effort single-rank failure propagation (`docs/SPEC.md` §5.6). Experimental: verified
  on CPU/`gloo` (macOS arm64 and Linux x86_64, `N = 2` and `N = 4`, acceptance test T15 cases `(a)–(l)`), and
  exercised on one NVIDIA L4 with 2 and 4 NCCL ranks sharing that GPU (`docs/VERIFICATION.md` §17.4);
  separate-GPU verification remains open (Stage 3).
- **Stage 3 multi-process verification tooling:** `scripts/probes/distributed_trainer_check.py` (41-scenario
  multi-rank `TimestralTrainer` measuring driver under `torchrun` with `--reference-mode auto|sketch|exact`,
  schema v5 archive verification, and parameter+buffer cross-rank equality),
  `scripts/probes/distributed_rank_failure_check.py` (mid-`fit()` rank-death probe with separate startup and
  post-kill deadlines and a strict survivor-abort allow-list), `scripts/probes/_dist_common.py`, and
  `scripts/multi_gpu_check.sh` wrapping `scripts/multi_gpu_check.py` (single stdlib-only Python `>= 3.10`
  session driver with content-bound resume, preserved ignored-on-entry signals for `nohup`, confirmed-live
  `/proc` and pure-BSD `ps` process-tree cleanup, pre-stage `--log-dir` writability checks and clean `OSError`
  handling, and 105 stub-stage tests in `tests/test_multi_gpu_check.py`; `docs/DEVELOPMENT.md`,
  `docs/VERIFICATION.md` §17.4).
- **`RankFailureError`** exported from `timestral_tune.trainer` (`RuntimeError` subclass raised on healthy
  peers when another rank fails during multi-process setup, training, validation, or rank-0 callbacks).
- **Multi-process export metadata:** optional paired `world_size` and `effective_batch_size` arguments on
  `export_checkpoint`, writing top-level `world_size` and `effective_batch_size` keys to
  `tune_metadata.json` when `world_size > 1` (`TimestralTrainer.export()` writes files on rank 0 only and
  validates without writing on non-zero ranks).

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
