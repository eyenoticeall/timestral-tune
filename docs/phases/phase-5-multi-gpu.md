# Phase 5 — Multi-Process / Multi-GPU Training

**Status:** Stages 1–2 implemented and verified on CPU/`gloo` on macOS arm64 (2026-11-23); Stage 3 (CUDA/NCCL hardware verification; multi-rank NCCL has never executed) is open; gradient accumulation remains unscheduled.
**Goal:** Data-parallel multi-process training across `N` ranks on a single node (CUDA/NCCL and CPU/`gloo`) with exact step-level equivalence to single-process training on the union batch.
**Spec sections:** `docs/SPEC.md` §5.6, §6, §7.4, §7.6, §9 (T15), §10 (Phase 5), §11 (D22–D37), §12; prototype and Stages 1–2 evidence in `docs/VERIFICATION.md` §16–§17.

The task checklist and exit criteria are in `docs/SPEC.md` §10.

---

## 1. Motivation and honest scope

Before committing engineering time or multi-GPU cloud spend, the proposal must be weighed against what single-GPU fine-tuning actually costs today.

### What single-L4 fine-tunes take today

Re-derived from the measured NVIDIA L4 (`23 GB`, `docs/VERIFICATION.md`) tables in `docs/BENCHMARKS.md`:

- **Main `bf16` benchmark tables** (`docs/BENCHMARKS.md` §1 `bench_5m_28m.json` and §3 `bench_123m_209m.json`; 28 gradient runs across `5M`–`209M`, `B = 16`–`32`, `160`–`200` steps): every run finished in **4.4 s** (`5M` retail `head_only`, `0.10 GiB`) to **53.2 s** (`209M` volatility `lora_temporal`, `3.60 GiB`) and used **0.08 GiB to 4.91 GiB** peak VRAM (`209M` retail `full_ft`, `37.0 s`) — under a quarter of one L4 card's `23 GB` memory.
- **Including `fp32` precision-ablation rows** (`docs/BENCHMARKS.md` §3 `precision_ablation_123m_209m.json`; `lora_temporal`, `B = 16`, `120` steps on retail): wall time reaches **53.7 s** (`209M` `fp32` default SDPA, `3.36 GiB`; `53.4 s` / `3.47 GiB` for `fp32` math), while peak VRAM across all printed tables stays at `4.91 GiB`.
- *(Note on unprinted sweep JSONs: the 144 single-L4 `bf16` runs behind `docs/BENCHMARKS.md` §2.1 in `docs/benchmarks/sweeps_large_decisions.json`, run at `B = 16`, `eval_batch_size = 32`, `120` steps, ranged `28.7 s`–`57.8 s` and `1.95`–`5.70 GiB`.)*

### Where multi-GPU could plausibly matter (estimates, not measurements)

Multi-GPU only pays off on jobs much larger than the Phase 3 benchmarks: datasets on the order of **10,000+ series** (tens of thousands of windows), or **long-context** (`4,096`–`32,768` steps) runs on `123M`/`209M`.

The wall-time figures below are **arithmetic estimates assuming linear scaling with step count**, not measurements, and make **no prediction of multi-GPU speed-up or scaling efficiency**:

$$T_{\text{single}} \approx N_{\text{steps}} \times t_{\text{step}}, \qquad N_{\text{steps}} = E \cdot \left\lceil \frac{W}{B} \right\rceil$$

- **Large short-context job (`209M`, `full_ft`, `bf16`, L4):** in `docs/BENCHMARKS.md` §3 (`bench_123m_209m.json`, `B = 16`, `ctx = 256`, `8`–`9` rows/series), `160` steps took `37.0 s` on retail (`231.25 ms/step`) and `46.8 s` on volatility (`292.50 ms/step`). Because `scripts/benchmarks/run_benchmarks.py`'s `perf_counter` span wraps `copy.deepcopy(base_cpu)`, `TimestralTrainer` construction, and `fit()` (including periodic validation and final `restore_best=True`), those `ms/step` figures are **upper bounds on pure per-step time**. Scaling linearly at those upper bounds for `10,000` series × `5` windows (`W = 50,000`) over `E = 5` epochs at `B = 16` ($N_{\text{steps}} = 5 \times \lceil 50{,}000 / 16 \rceil = 15{,}625$ steps) gives an estimated single-L4 wall time of up to `~3,613 s` (`~60.2 min`) at `37.0 s / 160` or `~4,570 s` (`~76.2 min`) at `46.8 s / 160`.
- **Longer-context `28M` pure step (`ctx = 1,024`, `H = 96`, `B = 32`, `5` rows/series, `bf16`, L4):** `docs/VERIFICATION.md` §13 measured `93.8 ms/step` eager and `59.8 ms/step` with `compile_mode="default"` (`full_ft`, excluding validation). At `10,000` steps, linear arithmetic gives `~938 s` (`~15.6 min`) eager or `~598 s` (`~10.0 min`) compiled.
- **Long-context `123M`/`209M` (`4,096`–`32,768` steps):** no L4 CUDA step timings exist yet. The only measured points are Apple M4 MPS `fp32` single steps (`docs/VERIFICATION.md` §15.2: `209M full_ft` at `32,768` steps `B = 1` took `1,469 ms/step`; `209M lora_temporal` `cov3` at `16,384` steps `B = 1` took `1,692 ms/step`).
- **Cached `head_only`:** below the 256 MiB per-dataset CPU cache cap (`_MAX_HEAD_CACHE_BYTES`, doubled when `sign_flip=True` caches both `+x` and `-x`), the frozen transformer body—the expensive part—runs once to build the cache; above that cap, the one-process path streams too.
- **No speed-up or NCCL efficiency has been measured.** Neither NCCL collective cost nor multi-GPU scaling has been observed anywhere.

### Alternative / prerequisite option: gradient accumulation first

An alternative raised while preparing this proposal is to defer multi-GPU and implement **gradient accumulation** first, optionally paired with refactoring `pinball_loss` and `distillation_loss` (or an internal helper) to return `(loss_sum, active_count)`:

- **Shared arithmetic:** exact gradient accumulation over $K$ micro-batches with variable scored counts uses the exact same count-weighted combination $\sum_k \frac{W_k}{W_{\text{tot}}} \nabla L_k$ as multi-process training (D24). Having the loss helpers return `(sum, count)` lets both features share one code path.
- **Immediate single-GPU payoff:** on long-context `123M`/`209M` runs where VRAM forces small per-step batch sizes (`B = 1`–`4`), gradient accumulation restores an effective batch of `16`–`32` (where shipped learning-rate defaults D19 were swept) on one GPU.
- **Zero hardware risk:** fully testable on existing CPU, Apple GPU, and single-L4 setups with no NCCL unknowns or multi-GPU VM cost.
- **Trade-off:** gradient accumulation does not cut wall-clock time on compute-bound large datasets; multi-GPU addresses throughput (subject to unmeasured NCCL scaling).

**Status:** Stages 1–2 of Phase 5 are implemented and verified on CPU/`gloo` on macOS arm64 (2026-11-23); Stage 3 (CUDA/NCCL) and gradient accumulation remain open (`docs/SPEC.md` §10).

---

## 2. Module and function map

All behavioural rules live in `docs/SPEC.md` §5.6, §7.4, §7.6, and D22–D37; this section maps the as-built Stages 1–2 implementation to modules and functions.

### Stage 1 — Core multi-process training (implemented)

- **`models/loader.py`** — unchanged (`resolve_device`, `load_model`, `load_forecaster` stay pure and collective-free; `huggingface_hub==0.23.0` `WeakFileLock` confirmed by source reading in `docs/VERIFICATION.md` §12, not a concurrent-download test).
- **`trainer/__init__.py`** — exports `RankFailureError` (`RuntimeError` subclass) alongside `TimestralTrainer`, `TrainingResult`, `EvalResult`, and callbacks.
- **`trainer/_types.py`** — shared dataclasses (`EvalResult`, `TrainingResult`, `_StepOutcome`, `_StepLoss`, `_FitBookkeeper`, etc.) and the `_FitStrategy` protocol, keeping `trainer.py` and `_distributed_strategy.py` acyclic.
- **`trainer/distributed.py`** — sole importer of `torch.distributed`: `DistributedContext` (`detect`, `check_active` verifying liveness/`world_size`/`rank`/`backend`, `all_reduce_sum`, `broadcast`, `propagate_failure`, `raise_propagated_failure`, `check_consistency`), `RankFailureError`, `is_oom_error`, `ControlStatus`, `broadcast_status`, `broadcast_module_state`, `GradientReducer` (flat buffer with trailing presence flags and all-present read-back skip), `DistributedBatchSampler` (SHA-256 epoch permutation, `num_global_steps`, `slice_size`), `DistributedBatchCycle` (Stage 2), `steps_per_epoch`, `rank_slice_size`, `rank_step_indices`, `validation_shard_indices`, and `rank_seed`.
- **`trainer/_distributed_strategy.py`** — `_DistributedStrategy` implementing `_FitStrategy`:
  - `prepare_fit` / `cleanup` — allocates `GradientReducer` per `fit()`, builds `DistributedBatchSampler` and rank-seeded train/val loaders (plus Stage 2 replay sampler/cycle/loader and replay flip generator), and releases them in `cleanup()` after end-of-fit collectives.
  - `iter_epoch_steps` / `_run_step` — global-iteration stepping with lazy per-epoch iterator creation, local zero-scored skip, six-stage step (Stage 1 stages 2–6: local forward/loss, 8-scalar `float64` `ReduceOp.SUM` stats collective, failure-over-non-finite precedence, `float64` overflow checks, count-weighted backward, and `GradientReducer.reduce()`).
  - `evaluate_val` — internal sharded validation (never touches `trainer._val_loader`), `eval()`/`no_grad()` restoring `training`, lazy cached stride loader skipped on empty shards, `[loss_sum, scored_positions, batches, failed]` reduction.
  - `on_fit_start` / `on_step_end` / `on_evaluate` / `finalize_best` / `on_fit_end` — rank-0-only callbacks, best-loss tracking, early stopping, and restore; fixed-size `float64` `[code, value]` broadcasts mirroring `should_stop` (both sets and clears) and propagating rank-0 losses; `[has_best, best_eval_loss, best_step, restored]` plus per-parameter broadcast when `restored == 1`.
- **`trainer/trainer.py`** — `_SingleProcessStrategy` and shared `_run_fit_loop` skeleton (`model.eval()` in inner `finally` before `finalize_best()`/`on_fit_end()`, `strategy.cleanup()` in outer `finally` after `on_fit_end()`); `_init_distributed` running five-step startup `(a)–(e)` (including protected `_startup_consistency_digest()` with `model.config.to_dict()`, uniform trainable non-emptiness/device/dtype checks, and `ValueError`/`RankFailureError` consistency check); rank-0-only startup logs; `head_only` cache bypass; public `training_step()` `RuntimeError`; unchanged collective-free `evaluate()`; snapshot-based `export()`.
- **`export.py`** — shared `_validate_export_inputs` helper (called by `export_checkpoint` and non-zero-rank `TimestralTrainer.export()` without hashing datasets) and optional paired `world_size` / `effective_batch_size` arguments (positive `numbers.Integral` excluding `bool`, `effective_batch_size >= world_size`), writing top-level `world_size` and `effective_batch_size` keys to `tune_metadata.json` only when `world_size > 1`.
- **`tests/distributed_harness.py`** — CPU/`gloo` `spawn` harness (file-based rendezvous, `torch.set_num_threads(1)`, hard wall-clock deadlines raising `DistributedTimeoutError`, `run_scenarios`).
- **`tests/acceptance/test_distributed.py`** — T15 (`test_t15_*`, Stage 1 cases `(a)–(i)` plus Stage 2 cases `(j)–(l)`).
- **`tests/test_distributed.py`**, **`tests/test_trainer_distributed.py`**, **`tests/test_trainer.py`**, **`tests/test_export.py`** — unit, multi-process primitive/harness, white-box trainer, and export-metadata tests.

### Stage 2 — Replay and distillation (implemented)

- **`trainer/distributed.py`** — `DistributedBatchCycle` cycling a replay `DistributedBatchSampler` (shared replay epoch seed `config.seed + 2_000_009`, D37) and `DataLoader` (rank-seeded generator, `num_workers=0`, `persistent_workers=False`, D36) across wrapping replay epochs, handling `len(replay_dataset) < N * b` and `< N` cleanly and resetting to replay epoch 0 step 0 on repeated `fit()`.
- **`trainer/_distributed_strategy.py`** — six-stage `_DistributedStrategy._run_step` adding stage 1 (`_precheck_domain_for_replay`: replay-only 2-element `float64` `[w_d, failed]` `ReduceOp.SUM` collective with failure-over-skip precedence before advancing replay), replay fetching/flipping (`rank_seed(config.seed + 3_000_007, rank)`), count-weighted `[W_r, sum_r, M, sum_m]` terms in the 8-scalar stats collective and `_backward_local_terms` (including empty/zero-scored domain ranks contributing replay), global `W_R > 0` / `M_tot > 0` mixing, and replay cleanup in `cleanup()`.
- **`trainer/trainer.py`** & **`config.py`** — removed Stage 1 `ReplayConfig` `NotImplementedError`; validated `replay_config`/`replay_dataset` (default `SyntheticReplaySource` with shared seed `config.seed + 2_000_003`) and hashed them into `_startup_consistency_digest()`; moved `_setup_replay_teacher()` after step (d)'s rank-0 broadcast using uncached teacher paths only (`_frozen_head`, live `disable_lora`, live `_frozen_base`, with a one-time rank-0 cache-bypass log; no head or teacher caches built, D32); logged `per_rank_replay_batch_size` and `effective_replay_batch_size` on rank 0 at startup (D27); skipped `dataset_fingerprint(self.replay_dataset)` on non-zero ranks in `export()` (D31); updated `TrainingConfig.num_workers` and `ReplayConfig.batch_size` docstrings.
- **`tests/acceptance/test_distributed.py`**, **`tests/test_trainer_distributed_replay.py`**, & **`scripts/probes/linear_requires_grad_dispatch.py`** — T15 cases `(j)–(l)` (including `test_t15_j_full_ft_distill_float32` and 2-rank CPU `test_t15_l_compile_default_replay_distill_smoke`), dedicated Stage 2 unit/integration replay suite (including `lora_temporal`, non-patch-multiple replay horizon, periodic validation + `restore_best=True`, cleanup), and the standalone `F.linear` `requires_grad` dispatch probe (`docs/VERIFICATION.md` §17.2).

### Optional (unscheduled): gradient accumulation

- **`losses/pinball.py`** (`pinball_loss`) and **`replay/distill.py`** (`distillation_loss`) — internal `(loss_sum, active_count)` helper if gradient accumulation is scheduled later.

---

## 3. Hardware verification session plan (Stage 3: CUDA / NCCL)

Multi-rank NCCL has never executed; nothing on CUDA or NCCL has been run yet for multi-process training. Before declaring multi-process training verified on GPUs, run one focused verification session on Linux CUDA hardware with 2 GPUs and 4 GPUs (note: a single 4-GPU VM can run both `N = 2` and `N = 4` tests).

### What to run

1. **Preflight:** run `bash scripts/cuda_check.sh` to verify compiler/headers and baseline single-GPU suite health.
2. **Prototype probe on NCCL:**
   - `uv run python scripts/probes/distributed_parity.py --device cuda` (`tiny_random`, `N = 2` and `N = 4`, NCCL).
   - `uv run python scripts/probes/distributed_parity.py --device cuda --sizes 5M 28M` (released weights on NCCL; measure cross-rank diffs and body-backprop diffs vs union reference, including `[4]` replay/distillation).
3. **Trainer acceptance & launcher checks:**
   - `torchrun --nproc_per_node=2` and `torchrun --nproc_per_node=4` running `T15` (`test_t15_*`, cases `(a)–(l)`) on CUDA/NCCL.
   - Smoke tests for `mixed_precision="bf16"` (with and without `ReplayConfig`), `compile_mode="default"` (including epochs with shorter or empty trailing shards and replay/distillation, measuring graph recompilation and timeout interaction), and `num_workers > 0` on Linux (measuring worker shutdown time across uneven trailing shards vs macOS's ~5 s/worker `MP_STATUS_CHECK_INTERVAL` wait).
   - Rank-failure smoke test: launch with a short explicit `timeout` (e.g. `15 s`), terminate rank 1 mid-step, and verify surviving ranks abort cleanly at timeout rather than hanging indefinitely.
4. **Throughput, per-step cost breakdown, & scaling micro-benchmark:**
   - Measure 1-GPU vs 2-GPU vs 4-GPU step time, samples/s, peak VRAM (including whether keeping `GradientReducer`'s flat gradient buffer allocated across validation inside `fit()` and holding live `_frozen_base` on every rank in `full_ft` distillation raise peak VRAM materially), and per-step overhead breakdown: Python per-parameter pack/unpack loops in `GradientReducer`, the small collectives per active step (3 without replay; 4 with replay adding the 2-element pre-check `all_reduce`), and flat-buffer `all_reduce` wall-time share on `28M`, `123M`, and `209M` (`full_ft` and `lora_temporal`, `bf16`, eager and `compile_mode="default"`, at `ctx = 256` and `ctx = 1,024`–`4,096`).

### What to record (in `docs/VERIFICATION.md` §17)

- Exact VM type, GPU count, CUDA/NCCL/PyTorch/Python versions, and commands.
- Probe and T15 pass/fail counts and `[INFO]` ranges on NCCL for `tiny_random`, `5M`, and `28M` (plus multi-seed and cosine-similarity measurements—with and without replay—used to set released-weights T15 bounds).
- Whether NCCL cross-rank gradients and final parameters remain bitwise `0.0` or exhibit non-zero reduction variance.
- `bf16`, `compile_mode="default"` (including trailing-shape recompilation and replay), Linux `num_workers > 0` teardown latency, and rank-failure timeout exit results.
- Measured 1 / 2 / 4 GPU step times, scaling efficiency, peak VRAM (training vs validation with flat buffer live, plus `full_ft` distillation with per-rank `_frozen_base`), and per-step pack/unpack + collective time shares.

### What findings would change the design

- **Communication overhead (revisiting D22):** if unoverlapped flat-buffer `all_reduce` accounts for a material fraction of step time (for example `> 15–20%` on `123M`/`209M` `full_ft` at realistic batch sizes), benchmark a thin `nn.Module.forward` wrapper under `DistributedDataParallel` to overlap communication with backward.
- **NCCL determinism:** check whether NCCL cross-rank gradients and final parameters remain bitwise `0.0`; if NCCL `all_reduce` exhibits non-zero cross-rank variance, relax T15's NCCL cross-rank tolerance and evaluate periodic rank-0 parameter broadcast.
- **Empty shards under `compile_mode="default"`:** if compiled graphs fail or recompile excessively on short/empty trailing shards, restrict `compile_model=True` in multi-process mode or pad/guard trailing steps.
- **Released-weights T15 bounds:** set released-weights bounds in spec §9 from Stage 3 measurements across sizes (`5M`/`28M`/`123M`/`209M`), seeds, devices, and replay regimes—including cosine similarity, which `scripts/probes/distributed_parity.py` does not yet report.

---

## 4. Estimated cloud cost for hardware verification

Provider pricing below is Google Cloud `us-central1` on-demand, retrieved **2026-11-23** from the provider's published pricing page (prices change over time and vary by region and commitment):

| Instance | GPUs | On-demand price (`us-central1`, retrieved 2026-11-23) |
|---|---|---:|
| `g2-standard-24` | 2× NVIDIA L4 | `$2.000832696 / hr` |
| `g2-standard-48` | 4× NVIDIA L4 | `$4.001665392 / hr` |

A single 1.5–2 hour session on `g2-standard-48` (which covers both `N = 2` and `N = 4` runs without provisioning a second VM) costs roughly `$6.00`–`$8.00` in compute (plus minor boot-disk charges). Provisioning `g2-standard-24` (`~1 hr`, `~$2.00`) and `g2-standard-48` (`~1–1.5 hr`, `~$4.00`–`$6.00`) separately gives a similar `$6`–`$8` total. Delete the VM immediately after collecting outputs.

---

## 5. Log of what was done

- **2026-11-23 (proposal & prototype probe):** wrote the documentation-only single-node multi-process design proposal across `docs/SPEC.md` (§5.6, §7.4, §9 T15 cases `(a)–(i)`, §10 Phase 5, §11 D22–D35, §12), `docs/VERIFICATION.md` (§11 one-off `ReduceOp.AVG` check and §16), `docs/STATUS.md`, and these notes, backed by the standalone CPU/`gloo` prototype probe `scripts/probes/distributed_parity.py` (`326/326` passed on `tiny_random` on PyTorch `2.15.0` and `2.4.0`; `516/516` passed + `68` `[INFO]` on `--sizes 5M`).
- **2026-11-23 (approved & pre-implementation design revision):** maintainer approved Phase 5 (implementation starting; gradient accumulation stays unscheduled). Updated `docs/SPEC.md`, `docs/VERIFICATION.md`, `docs/STATUS.md`, and these notes to lock down exact backend checks, pre-collective device binding, `__init__` startup failure/consistency collectives and rank/world-size/backend snapshot, binary failure flags + conditional `int64` length-`N` vectors, `Exception`-only / OOM-excluded propagation and `RankFailureError`, fixed-size `float64` control broadcasts, contiguous slices, local zero-scored skip order, `(base + r * 1_000_000_007) mod 2^63` rank seeds, reused flat gradient buffer, sharded validation `eval()`/`no_grad()` & shared `export()` validation, Stage 2 decisions (per-rank cycling replay loaders, replay-only pre-forward count collective, no multi-process head/teacher caches), and tightened T15 cases `(a)–(i)`; trimmed repeated numbers and cut §2 to a module/function map.
- **2026-11-23 (Stage 1 implemented & CPU/`gloo` verified):** implemented `trainer/_types.py`, `trainer/distributed.py`, `trainer/_distributed_strategy.py`, shared `_run_fit_loop` strategy dispatch in `trainer/trainer.py`, `RankFailureError` export, shared `_validate_export_inputs` and paired `world_size`/`effective_batch_size` on `export_checkpoint`, `tests/distributed_harness.py`, acceptance test T15 Stage 1 cases `(a)–(i)` (`tests/acceptance/test_distributed.py`), and unit/white-box distributed tests. Fixed review-found hang paths (`should_stop` clearing sync and protected startup digest in `290cd6c`; `strategy.cleanup()` after end-of-fit collectives in `d181c7a`), preserved `RankFailureError` on matching ranks during consistency checks (`09de038`), packed gradient presence flags into `GradientReducer` (`ac9ab04`), and added rank-seeded loader generators (`e3a4f6e`). Verified Stage 1 T15 on CPU/`gloo` on PyTorch `2.4.0` and `2.15.0`, plus `huggingface_hub==0.23.0` `WeakFileLock` source reading (`docs/VERIFICATION.md` §12).
- **2026-11-23 (Stage 2 implemented & CPU/`gloo` verified at `3347f70`):** added `DistributedBatchCycle` with shared replay epoch seed (`+2_000_009`, D37) and main-process `num_workers=0` replay loader (D36), refactored `_DistributedStrategy._run_step` into six named stages with the replay-only 2-element `[w_d, failed]` pre-check collective (D23) and `[W_r, sum_r, M, sum_m]` count-weighted reduction, moved uncached `_setup_replay_teacher()` after rank-0 broadcast (one-time rank-0 bypass log; no head or teacher caches in multi-process mode, D32), added per-rank replay batch logging (D27), startup digest hashing over `replay_config`/`replay_dataset`, and peer fingerprint skip in `export()`. Extended T15 to cases `(a)–(l)` (including `test_t15_j_full_ft_distill_float32` and 2-rank CPU `test_t15_l_compile_default_replay_distill_smoke`), added `tests/test_trainer_distributed_replay.py` and `scripts/probes/linear_requires_grad_dispatch.py`, and verified the full network-free suite and four distributed test files on CPU/`gloo` on macOS arm64 across PyTorch `2.4.0` and `2.15.0` (`docs/VERIFICATION.md` §17). Stage 3 (CUDA/NCCL hardware verification; multi-rank NCCL has never executed) remains open.
