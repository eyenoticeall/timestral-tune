# Project Status

**Last updated:** 2026-11-22
**Stage:** pre-alpha. Phases 0, 1, 2, and 3 are built and verified (`phase-3` branch off `main`; PyPI publish pending maintainer action).
**Next phase:** Phase 4 (Stretch: replay mixing, distillation, tiled training) or v0.1.0 PyPI release. See spec §10.

## What works today

| Area | State |
|---|---|
| Loading | `load_model` / `load_forecaster`: pinned Hub revisions, config from the checkpoint, any device |
| Devices | CPU (PyTorch 2.4+), Apple GPU (PyTorch 2.8+, fp32), CUDA (verified on NVIDIA L4, PyTorch 2.15.0, fp32 and `bf16`; commit `6880061`) |
| Data | `TimeSeriesDataset` from polars, pandas or pyarrow; point-in-time windows; time splits; `DHFDataCollator` (with power-of-two shape bucketing); per-window categorical encoding; `flip_batch_signs` |
| Training forward and loss | Normalised-space forward (trimmed `L + 2h` covariate layout), both target frames, pinball loss, `compute_loss` |
| Adaptation & training | `TimestralTrainer` across `head_only` (with frozen-feature caching), `lora_variate` (with single-row guard), `lora_temporal`, and `full_ft`; sweep-backed LR & LoRA defaults; `sign_flip` augmentation; fp32 everywhere, `bf16` + fused AdamW + `torch.compile` on CUDA; early stopping, CPU best-state restore |
| Conformal calibration | `ConformalCalibrator` (static split CQR), `OnlineConformalCalibrator` (online ACI, forecast-count rolling window, Lemma 4.1 anti-windup), and `collect_conformal_data` helper, with anchor-aware monotonicity repair and JSON persistence |
| Evaluation & metrics | `utils/metrics.py`: raw-space `pinball_loss_raw`, `weighted_quantile_loss`, `mase`, `coverage`, `interval_width`, `evaluate_forecaster`, and `evaluate_forgetting` |
| Benchmarks & sweeps | `scripts/benchmarks/` + `docs/BENCHMARKS.md`: 5M & 28M tables across all 7 modes + forgetting metrics on realised volatility (post-Oct 27, 2026), retail demand, and GIFT-Eval |
| Optional Lacuna integration | `integrations/lacuna.py` (`timestral-tune[lacuna]`, Python 3.11+): `LacunaExporter` (`ExperimentRegistry`) and `write_signal_frame` (`SignalStudy`), verified live against `lacuna-quant==0.14.1` |
| Export | `export_checkpoint` and `TimestralTrainer.export()`: upstream-loadable `config.json` + merged `model.safetensors`, unmerged LoRA adapters (`load_lora`), `conformal.json`, `tune_metadata.json`, `LICENSE`, `NOTICE` |
| Verifiers | `timestral_tune.utils.run_checks`: parity, label isolation, firewall, known-future sensitivity, loss mask |
| Acceptance tests | **14 of 14 implemented and passing**: T1–T14 (zero placeholders remaining) |
| Test suite | 536 passed, 5 skipped (4 CUDA-only `bf16`, 1 optional `lacuna`), on tiny, 5M and 28M (PyTorch 2.15.0 on 3.13; live `lacuna-quant==0.14.1`: 4/4 passed; NVIDIA L4 CUDA at `6880061`: 528/2) |
| CI | Green on `main` (run `44936917337`, Python 3.10–3.13 on Linux); `phase-3` pending PR/merge |

## What does not exist yet

- Phase 4 stretch features: replay mixing, built-in generators, distillation (`replay/`), tiled training, and 123M/209M Apple GPU coverage.
- PyPI release (packaged and audited via `uv build`; publishing requires maintainer approval).

## What has been verified, and where

All locally on Apple M4 (macOS, PyTorch 2.4/2.8/2.14/2.15) and on an NVIDIA L4 (Linux, PyTorch 2.15.0+cu132); details in `docs/VERIFICATION.md`.

- Training inputs and forward equal upstream inference (T1–T3, collator and dataset parity tests).
- No leakage found on the training path (T4–T7, T12, with one and with several targets) and an exact loss mask (T8).
- Slice LoRA step-0 identity, merge/unmerge parity (`< 1e-12` fp64, `< 1e-4` fp32), and single-row guard (T9, T10).
- Upstream `strict=True` export round-trip across all four modes on tiny, 5M, and 28M, plus verbatim Hub `LICENSE` copy (T11).
- Static CQR and online ACI conformal coverage, monotonicity repair, saturation, and persistence (T13).
- Device parity and gradients on CPU, Apple GPU, and CUDA (T14), plus `bf16` autocast and `torch.compile` on L4 CUDA (3.3× faster 28M steps, 45% less VRAM).
- Decision D15 resolved on CUDA: 20-step training trajectories under `fp32` default SDPA, `fp32` unfused math, and `bf16` match closely in validation loss across `lora_variate`, `lora_temporal`, and `full_ft`.

- Real-dataset benchmarks and sweeps across all four checkpoints (`5M`, `28M`, `123M`, `209M`) on realised volatility (post-Oct 27, 2026), retail demand, and GIFT-Eval forgetting (`docs/BENCHMARKS.md`), `123M`/`209M` CUDA gradient accuracy and `fp32`/`math`/`bf16` training parity (`docs/VERIFICATION.md` §13), plus live `lacuna-quant==0.14.1` integration.

Not verified anywhere: Apple GPU on `123M`/`209M`, multi-GPU.

## Next steps

1. **Review & merge `phase-3` into `main`**, then publish v0.1.0 to PyPI when ready.
2. **Optional Phase 4** (spec §10): replay mixing, built-in generators, distillation, tiled training.

## Open questions and known issues

Full list in spec §12. The ones that matter next:

- **CUDA needs a C compiler and Python headers** with recent PyTorch (spec §12). `scripts/cuda_check.sh` checks for them.
- **Apple GPU on `123M`/`209M` and multi-GPU training** remain untested (`123M` and `209M` are verified on L4 CUDA; `docs/BENCHMARKS.md` §3, `docs/VERIFICATION.md` §13).
- **PyTorch 2.4.0 on macOS hangs at exit when a DataLoader uses worker processes.** Gone by 2.8.0 (`docs/VERIFICATION.md` §10). `num_workers` defaults to 0.
- **CI actions use a deprecated Node.js version.** GitHub warns that `actions/checkout@v4` and `astral-sh/setup-uv@v5` target Node.js 20. They still run.

## Branch state

- `main`: Phases 0, 1, and 2 (`1b3dec1`, CI green).
- `phase-3`: Phase 3 (raw-space metrics, forgetting metric, sign-flip augmentation, Lacuna integration, reproducible sweeps & 5M/28M benchmarks, sweep-backed defaults, `docs/BENCHMARKS.md`), open off `main`, unpushed.

Branch policy is in `AGENTS.md` §6.
