# Project Status

**Last updated:** 2026-11-22
**Stage:** pre-alpha. Phases 0, 1, 2, 3, and rescoped Phase 4 (replay mixing, frozen-base distillation, and Apple GPU `123M`/`209M`/long-context coverage) are built, verified, and merged into `main` (tiled DHF deferred as exploratory; PyPI publish pending maintainer action).
**Next phase:** v0.1.0 PyPI release (when ready), or exploratory evaluation of tiled DHF. See spec §10.

## What works today

| Area | State |
|---|---|
| Loading | `load_model` / `load_forecaster`: pinned Hub revisions, config from the checkpoint, any device |
| Devices | CPU (PyTorch 2.4+), Apple GPU (PyTorch 2.8+, fp32; `5M`/`28M` on 2.8 & 2.15, `123M`/`209M` & contexts up to `32,768` on 2.15), CUDA (verified on NVIDIA L4 Linux x86, PyTorch 2.15.0, fp32 and `bf16`; commit `89ac249`) |
| Data | `TimeSeriesDataset` from polars, pandas or pyarrow; point-in-time windows; time splits; `DHFDataCollator` (with power-of-two shape bucketing); per-window categorical encoding; `flip_batch_signs` |
| Training forward and loss | Normalised-space forward (trimmed `L + 2h` covariate layout), both target frames, pinball loss, `compute_loss` |
| Adaptation & training | `TimestralTrainer` across `head_only` (with dual-orientation `+x`/`-x` frozen-feature caching), `lora_variate` (with single-row guard), `lora_temporal`, and `full_ft`; `sign_flip=True` and sweep-backed single cross-size LR & LoRA defaults (D19; `full_ft` recommendation of `3e-4` on `5M`/`28M` documented in `docs/BENCHMARKS.md` §2.2); fp32 everywhere, `bf16` + fused AdamW + `torch.compile` on CUDA; early stopping, CPU best-state restore |
| Replay & distillation | `ReplayConfig` (default `replay_fraction=0.0, distill_weight=0.1, smooth_beta=1e-3`), `SyntheticReplaySource` (5 classical families), `distillation_loss`, `disable_lora` zero-VRAM bypass, CPU-cached frozen-base teacher outputs, and `replay_source` export provenance (D21); empirically ablated with lower-LR and pool-size controls in `docs/BENCHMARKS.md` §5 |
| Conformal calibration | `ConformalCalibrator` (static split CQR), `OnlineConformalCalibrator` (online ACI, forecast-count rolling window, Lemma 4.1 anti-windup), and `collect_conformal_data` helper, with anchor-aware monotonicity repair and JSON persistence |
| Evaluation & metrics | `utils/metrics.py`: raw-space `pinball_loss_raw`, `weighted_quantile_loss`, `mase`, `coverage`, `interval_width`, `evaluate_forecaster`, and `evaluate_forgetting` |
| Benchmarks & sweeps | `scripts/benchmarks/` + `docs/BENCHMARKS.md`: `5M`–`209M` tables + raw JSONs (`docs/benchmarks/`) across all modes, forgetting metrics, 123M/209M 3-seed decision sweeps, and 28M significance & classical baselines |
| Optional Lacuna integration | `integrations/lacuna.py` (`timestral-tune[lacuna]`, Python 3.11+): `LacunaExporter` (`ExperimentRegistry`) and `write_signal_frame` (`SignalStudy`), verified live against `lacuna-quant==0.14.1` |
| Export | `export_checkpoint` and `TimestralTrainer.export()`: upstream-loadable `config.json` + merged `model.safetensors`, unmerged LoRA adapters (`load_lora`), `conformal.json`, `tune_metadata.json`, `LICENSE`, `NOTICE` |
| Verifiers | `timestral_tune.utils.run_checks`: parity, label isolation, firewall, known-future sensitivity, loss mask |
| Acceptance tests | **14 of 14 implemented and passing**: T1–T14 (zero placeholders remaining) |
| Test suite | 567 passed, 17 skipped (16 CUDA-only, 1 optional `lacuna`), on tiny, 5M, 28M, and 123M MPS parity (`main`, PyTorch 2.15.0 on 3.13; PyTorch 2.4.0 `-m "not network"`: 313/25; PyTorch 2.8.0 & Python 3.10/2.14.1: 356/15; live `lacuna-quant==0.14.1`: 4/4; NVIDIA L4 Linux x86 CUDA at `89ac249`: **580 passed, 4 skipped**) |
| CI | Running on `main` (Python 3.10–3.13 on Linux) |

## What does not exist yet

- Deferred exploratory items: fixed-length tiled DHF training and sampled block lengths (`docs/phases/phase-4-stretch.md`).
- Multi-GPU training.
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

- Real-dataset benchmarks and sweeps across all four checkpoints (`5M`, `28M`, `123M`, `209M`) on realised volatility (post-Oct 27, 2026), retail demand, and GIFT-Eval forgetting (`docs/BENCHMARKS.md`), `123M`/`209M` CUDA gradient accuracy and `fp32`/`math`/`bf16` training parity (`docs/VERIFICATION.md` §13), live `lacuna-quant==0.14.1` integration, Phase 4 replay/distillation forgetting ablations (`docs/BENCHMARKS.md` §5), and Apple GPU `123M`/`209M`/long-context (`4,096..32,768`) parity & training (`docs/VERIFICATION.md` §15).

Not verified anywhere: multi-GPU, real general-domain replay datasets (only built-in synthetic replay was benchmarked), long-context (`4,096..32,768`) gradient parity on MPS.

## Next steps

1. **Publish v0.1.0 to PyPI** when ready.
2. **Optional exploratory work:** evaluate whether fixed-length tiled DHF warrants its token and categorical-encoding trade-offs on long fine-tuning series.

## Open questions and known issues

Full list in spec §12. The ones that matter next:

- **CUDA needs a C compiler and Python headers** with recent PyTorch (spec §12). `scripts/cuda_check.sh` checks for them.
- **Multi-GPU training** remains untested (`123M`, `209M`, and contexts up to `32,768` are now verified on Apple GPU in `fp32`, and `123M`/`209M` on NVIDIA L4 CUDA in `fp32`/`bf16`).
- **PyTorch 2.4.0 on macOS hangs at exit when a DataLoader uses worker processes.** Gone by 2.8.0 (`docs/VERIFICATION.md` §10). `num_workers` defaults to 0.
- **CI actions use a deprecated Node.js version.** GitHub warns that `actions/checkout@v4` and `astral-sh/setup-uv@v5` target Node.js 20. They still run.

## Branch state

- `main`: Phases 0–4 (rescoped Phase 4 fast-forwarded from `phase-4` at `9261c75` on 2026-11-22). All prior branches (`phase-1`, `perf/phase1-kernels`, `fix/math-sdpa-backend`, `phase-2`, `phase-3`, `phase-4`) are closed and deleted locally and on `origin`.

Branch policy is in `AGENTS.md` §6.
