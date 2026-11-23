# Project Status

**Last updated:** 2026-11-23
**Stage:** alpha. Version `0.1.0` is published on PyPI (`pip install timestral-tune`, tag `v0.1.0`, 2026-11-23). Phases 0–4 (rescoped Phase 4: replay mixing, frozen-base distillation, and Apple GPU `123M`/`209M`/long-context coverage; tiled DHF deferred as exploratory) are on `main`; Phase 5 Stages 1–2 (single-node multi-process training across all four modes, including `ReplayConfig` replay mixing and uncached frozen-base distillation, verified on CPU/`gloo` only) are implemented ahead of the next release. The repository is public.
**Next phase:** Phase 5 Stage 3 (2-GPU/4-GPU CUDA/NCCL hardware verification; multi-rank NCCL has never executed); respond to first outside use of `0.1.0`. See spec §10.

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
| Export | `export_checkpoint` and `TimestralTrainer.export()`: upstream-loadable `config.json` + merged `model.safetensors`, unmerged LoRA adapters (`load_lora`), `conformal.json`, `tune_metadata.json` (with conditional `world_size` / `effective_batch_size` when `world_size > 1`), `LICENSE`, `NOTICE` |
| Multi-process training (Stages 1–2, experimental) | Caller-owned `torch.distributed` group (`"gloo"` or `"nccl"`, `world_size > 1`) activates data-parallel training in `TimestralTrainer` across all four modes (`head_only`, `lora_variate`, `lora_temporal`, `full_ft`), including `ReplayConfig` replay mixing and uncached frozen-base distillation (`DistributedBatchCycle`, main-process `num_workers=0` replay loader, per-rank `ReplayConfig.batch_size`), count-weighted `GradientReducer` (`ReduceOp.SUM`), startup consistency & rank-0 broadcast, sharded validation, rank-0 callbacks/restore, `RankFailureError`, and rank-0-only `export()`. Verified on CPU/`gloo` on macOS arm64 only (`N = 2` and `N = 4`, PyTorch `2.4.0` & `2.15.0`, `docs/VERIFICATION.md` §17); multi-rank NCCL has never executed (Stage 3 open) |
| Verifiers | `timestral_tune.utils.run_checks`: parity, label isolation, firewall, known-future sensitivity, loss mask |
| Acceptance tests | **15 of 15 implemented and passing**: T1–T14 (tiny, 5M, 28M across CPU, Apple GPU, L4 CUDA) and T15 cases `(a)–(l)` (CPU/`gloo`, tiny random model, `N = 2` and `N = 4`; CUDA/NCCL in Stage 3) |
| Test suite | `0.1.0` `main` baseline and L4 CUDA counts in `docs/VERIFICATION.md` §10 & §13; Phase 5 Stages 1–2 network-free and multi-process suite counts on CPU/`gloo` (PyTorch `2.15.0` and `2.4.0`) in `docs/VERIFICATION.md` §17 |
| CI | Green on `main` (run `45046304547`, Python 3.10–3.13 on Linux: lint, build, and network-free tests) |

## What does not exist yet

- Phase 5 Stage 3: CUDA/NCCL multi-GPU hardware verification (multi-rank NCCL has never executed; `bf16`, CUDA `compile_mode="default"`, `torchrun` on Linux, scaling/VRAM measurements, released-weights T15 bounds).
- Optional (unscheduled): gradient accumulation.
- Deferred exploratory items: fixed-length tiled DHF training and sampled block lengths (`docs/phases/phase-4-stretch.md`).

## What has been verified, and where

All locally on Apple M4 (macOS, PyTorch 2.4/2.8/2.14/2.15) and on an NVIDIA L4 (Linux, PyTorch 2.15.0+cu132); details in `docs/VERIFICATION.md`.

- Training inputs and forward equal upstream inference (T1–T3, collator and dataset parity tests).
- No leakage found on the training path (T4–T7, T12, with one and with several targets) and an exact loss mask (T8).
- Slice LoRA step-0 identity, merge/unmerge parity (`< 1e-12` fp64, `< 1e-4` fp32), and single-row guard (T9, T10).
- Upstream `strict=True` export round-trip across all four modes on tiny, 5M, and 28M, plus verbatim Hub `LICENSE` copy (T11).
- Static CQR and online ACI conformal coverage, monotonicity repair, saturation, and persistence (T13).
- Device parity and gradients on CPU, Apple GPU, and CUDA (T14), plus `bf16` autocast and `torch.compile` on L4 CUDA (3.3× faster 28M steps, 45% less VRAM).
- Decision D15 resolved on CUDA: 20-step training trajectories under `fp32` default SDPA, `fp32` unfused math, and `bf16` match closely in validation loss across `lora_variate`, `lora_temporal`, and `full_ft`.
- Multi-process training Stages 1–2 (domain loss and `ReplayConfig` replay/distillation: parity, sampling, cycling, sharded validation, guards, export, and single-rank failure propagation) on CPU/`gloo` on macOS arm64 for `N = 2` and `N = 4` across PyTorch `2.4.0` and `2.15.0` (T15 cases `(a)–(l)`, `tests/test_distributed.py`, `tests/test_trainer_distributed.py`, `tests/test_trainer_distributed_replay.py`; `docs/VERIFICATION.md` §17).
- Real-dataset benchmarks and sweeps across all four checkpoints (`5M`, `28M`, `123M`, `209M`) on realised volatility (post-Oct 27, 2026), retail demand, and GIFT-Eval forgetting (`docs/BENCHMARKS.md`), `123M`/`209M` CUDA gradient accuracy and `fp32`/`math`/`bf16` training parity (`docs/VERIFICATION.md` §13), live `lacuna-quant==0.14.1` integration, Phase 4 replay/distillation forgetting ablations (`docs/BENCHMARKS.md` §5), and Apple GPU `123M`/`209M`/long-context (`4,096..32,768`) parity & training (`docs/VERIFICATION.md` §15).

Not verified anywhere: multi-GPU CUDA/NCCL hardware (multi-rank NCCL has never executed; Stages 1–2 multi-process is verified on CPU/`gloo` on macOS arm64 only), released weights under T15 (with or without replay), live concurrent Hub downloads at `huggingface_hub==0.23.0` (source reading only, `docs/VERIFICATION.md` §12), real general-domain replay datasets (only built-in synthetic replay was benchmarked), long-context (`4,096..32,768`) gradient parity on MPS, any checkpoint larger than 209M (other sizes are supported by design and tested only on small random models; spec §3.1).

## Next steps

1. **Phase 5 Stage 3:** run 2-GPU and 4-GPU Linux CUDA/NCCL hardware verification (`torchrun`, `bf16`, `compile_mode="default"` with and without replay, Linux worker teardown, per-step cost/VRAM/scaling breakdown including `_frozen_base` VRAM, released-weights T15 bounds; `docs/phases/phase-5-multi-gpu.md` §3).
2. **Watch for issues from first users of `0.1.0`**; fixes go out as patch releases (`docs/DEVELOPMENT.md`, "Releasing").
3. **Maintainer decisions on spec §12 findings:** PyTorch `F.linear` `requires_grad` dispatch remedies for `full_ft` distillation and the ten pre-existing one-process observations.
4. **Optional exploratory work:** evaluate gradient accumulation or whether fixed-length tiled DHF warrants its token and categorical-encoding trade-offs on long fine-tuning series.

## Open questions and known issues

Full list in spec §12. The ones that matter next:

- **CUDA needs a C compiler and Python headers** with recent PyTorch (spec §12). `scripts/cuda_check.sh` checks for them.
- **Multi-GPU CUDA/NCCL training** has never executed on GPU hardware (Stages 1–2 are implemented and verified on CPU/`gloo` on macOS arm64 only; Stage 3 CUDA/NCCL remains open).
- **PyTorch `F.linear` `requires_grad` dispatch on non-contiguous `float32` inputs** (spec §12, `docs/VERIFICATION.md` §17.2): causes step-0 round-off differences between student and identical-weight frozen teacher in `full_ft` distillation on macOS arm64 CPU (affecting one-process runs too; left unchanged pending maintainer decision).
- **PyTorch 2.4.0 on macOS hangs at exit when a DataLoader uses worker processes.** Gone by 2.8.0 (`docs/VERIFICATION.md` §10). `num_workers` defaults to 0.
- **`compile_model=True` on PyTorch 2.4.0 with Python older than 3.12 needs `setuptools` installed.** That PyTorch wheel does not declare it (`docs/VERIFICATION.md` §10).
- **CI actions use a deprecated Node.js version.** GitHub warns that `actions/checkout@v4` and `astral-sh/setup-uv@v5` target Node.js 20. They still run.

## Branch state

- `main`: Phases 0–4 (rescoped Phase 4 fast-forwarded from `phase-4` at `9261c75` on 2026-11-22). All prior Phase 1–4 branches are closed and deleted locally and on `origin`. Phase 5 Stages 1–2 are developed on the multi-GPU work branch pending merge.

Branch policy is in `AGENTS.md` §6.

## Multi-GPU training (Phase 5)

Phase 5 Stages 1–2 (single-node multi-process training across `head_only`, `lora_variate`, `lora_temporal`, and `full_ft`, including `ReplayConfig` replay mixing and uncached frozen-base distillation) are implemented (`src/timestral_tune/trainer/_types.py`, `distributed.py`, `_distributed_strategy.py`, `trainer.py`, and `export.py`) and verified on CPU/`gloo` on macOS arm64 for `N = 2` and `N = 4` across PyTorch `2.4.0` and `2.15.0` (acceptance test T15 cases `(a)–(l)`, `docs/VERIFICATION.md` §17; `docs/SPEC.md` §5.6, §7.4, §7.6, §9, §10, D22–D37; `docs/phases/phase-5-multi-gpu.md`). Stage 3 (CUDA/NCCL GPU hardware verification; multi-rank NCCL has never executed) remains open, and gradient accumulation stays an unticked, unscheduled option.
