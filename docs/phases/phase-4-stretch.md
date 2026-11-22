# Phase 4 — Anti-forgetting and Apple GPU Scaling (rescoped)

**Status:** Complete on branch `phase-4` (2026-11-22; Items 1 & 4 done; Items 2 & 3 deferred as exploratory).
**Goal:** Replay mixing, built-in synthetic generators, frozen-base distillation, and Apple GPU (MPS) coverage for `123M`, `209M`, and contexts up to `32,768`.
**Spec sections:** `docs/SPEC.md` §5.3, §5.4, §10, decisions D1, D7, D21

The task checklist and exit criteria are in the spec, §10.

## Scope decision (2026-11-22)

Phase 4 was rescoped to **Item 1 (replay + distillation)** and **Item 4 (Apple GPU 123M/209M/long-context)**, with **Items 2 & 3 (fixed-length tiled DHF and sampled block lengths)** deferred as exploratory:
- Fixed-length tiling undoes D16's `L + 2h` token saving (roughly doubling tokens), conflicts fundamentally with per-window categorical target encoding (D1, §6.3 rule 4), assumes known-future covariates are known across the entire tiled span, and requires a separate multi-origin batch/collator/frame pipeline whose payoff on small fine-tuning datasets is unproven.
- Sampled block lengths depend on tiled training and Mistral's unpublished block-length distribution.

## Notes for the implementer

- **Replay & distillation (D21):**
  - Replay runs as a separate micro-batch so domain datasets with covariates pair cleanly with univariate `SyntheticReplaySource` or general-domain datasets without forcing matching layouts/horizons.
  - `distillation_loss` defaults to Huber/Smooth-L1 with `smooth_beta=1e-3` (`0.0` for exact L1) to eliminate the `+/-1` subgradient kink at zero from `~1e-7` fp32 GEMM round-off between cached and live teacher outputs at step 0.
  - Low-VRAM frozen-base teacher: frozen `model.head` in `head_only` (preserving dual `+x`/`-x` frozen-body caching), CPU-cached base quantiles up to 256 MiB in `lora_*` and `full_ft`, and zero-VRAM `disable_lora` (LoRA) or frozen base copy (`full_ft`) fallbacks.
  - Empirical caveat: on `full_ft`, pure synthetic pinball replay can slightly worsen forgetting because full fine-tuning adapts to the synthetic generator; use light frozen-base distillation (`distill_weight ~ 0.1`) or real general-domain replay data.
- **Apple GPU (MPS) coverage:**
  - Verified on 16 GiB Apple M4 (`fp32`, PyTorch 2.15.0) across `123M`, `209M`, and contexts `4,096..32,768` steps (`128..1,024` patches), including `209M` `full_ft` at `32,768` steps (`B=1`, `6.09 GiB` driver) and `cov3` `lora_temporal` at `16,384` steps (`6.10 GiB` driver).

## Log of what was done

- 2026-11-22 (`68c64c5`, `7304681`, `258965d`, `e3c6883`): implemented `ReplayConfig` (defaulting to
  light label-free distillation `replay_fraction=0.0, distill_weight=0.1, smooth_beta=1e-3`),
  `SyntheticReplaySource` (5 classical families), `distillation_loss` (Huber `smooth_beta=1e-3`, exact L1
  at `0.0`), `disable_lora`, full `TimestralTrainer` integration across all 4 modes with low-VRAM teacher
  caching/bypass, val-as-replay rejection, isolated head-cache slots (`"train"`, `"val"`, `"replay"`),
  `tune_metadata.json` `replay_config` + `replay_source` export, and unit/parity/mutation/CUDA tests
  (`tests/test_replay.py`: 39 tests total — 30 CPU + 9 CUDA `fp32`/`bf16`/`reduce-overhead`).
- 2026-11-22 (`e31a2b2`, `4db0fa1`): ran empirical Apple GPU replay & distillation ablation with lower-LR
  (`head_only 1e-3`, `full_ft 2e-4`) and 256-series pool controls (`scripts/benchmarks/run_replay_ablation.py`,
  `docs/benchmarks/replay_ablation.json`).
- 2026-11-22 (`3608c60`, `e3c6883`): added and ran `scripts/probes/mps_large_and_long_context.py`
  (`docs/benchmarks/mps_coverage.json`) on 16 GiB Apple M4 (PyTorch 2.15.0) and ran full
  `scripts/cuda_check.sh` on NVIDIA L4 Linux x86 (`255/3` device tests, `577/4` whole suite, exit 0; VM
  deleted immediately after).
