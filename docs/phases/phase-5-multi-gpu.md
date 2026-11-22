# Phase 5 — Multi-Process / Multi-GPU Training (Proposed)

**Status:** Proposed design only — awaiting maintainer go/no-go and sequencing decision (nothing in `src/` or `tests/` implements this yet).
**Goal:** Data-parallel multi-process training across `N` ranks on a single node (CUDA/NCCL and CPU/`gloo`) with exact step-level equivalence to single-process training on the union batch.
**Spec sections:** `docs/SPEC.md` §5.6, §9 (T15), §10 (Phase 5), §11 (D22–D35), §12; prototype evidence in `docs/VERIFICATION.md` §16.

The task checklist and proposed exit criteria are in `docs/SPEC.md` §10.

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

**Decision for the maintainer:** go/no-go on Phase 5, and whether to land gradient accumulation (and `(sum, count)` loss helpers) before Stage 1, alongside it, or instead of it for now.

---

## 2. Staged implementation plan & trainer touch points

Design details and rationale are in `docs/SPEC.md` §5.6 and D22–D35. All touch points below name modules and functions rather than line numbers.

### Stage 1 — Core multi-process training (domain loss, four modes, no head cache)

- **`models/loader.py` (no changes)**
  - `resolve_device()`, `load_model()`, and `load_forecaster()` remain pure and collective-free (D28, D29). Every rank calls `load_model()` as today without a rank-0 download barrier (Stage 1 includes checking `huggingface_hub==0.23` lock behaviour, §12).
- **`trainer/trainer.py`**
  - `TimestralTrainer.__init__`:
    - Detect caller-owned multi-process group (`dist.is_initialized()` with `world_size > 1`); keep single-process code path byte-for-byte unchanged otherwise (D25, D28).
    - Run all rank-local setup and validation that can fail first, catching any exception locally (D28, D29, D32): backend-driven device resolution (`dist.get_backend()`: `nccl`→CUDA, `gloo`→CPU, `gloo`+CUDA refused in v1, `"auto"` never picks MPS, contradictions raise; for CUDA honour explicit index else `LOCAL_RANK` else `rank`, validate against `torch.cuda.device_count()`, and call `torch.cuda.set_device` for every CUDA device before `torch.cuda.is_bf16_supported()`), Stage 1 restrictions (`ValueError` for `compile_mode="reduce-overhead"` when `compile_model=True`; `NotImplementedError` for `replay_config is not None` following D13), `lora_config`/`replay_config`/`replay_dataset` checks, model loading or validation, dataset checks (`_check_dataset` and train/val layout equality), `_select_trainable()`, and `early_stopping_patience`/callback validation (moved ahead of the first collective).
    - Run one small startup failure-status collective (`all_reduce(ReduceOp.SUM)` of per-rank failure flags): if any rank failed, failing rank(s) re-raise their own exception and peers raise an error naming the failing rank(s) without entering the broadcast (best-effort and unprototyped `[D]`; ranks unable to enter a collective and post-broadcast steps fall under D35) (D29).
    - Only if all ranks succeed, broadcast **all model parameters and all buffers** from rank 0 (before `_setup_replay_teacher()` and `torch.compile`, which depend on the synced eager model) so caller-supplied `model=` weights/buffers and LoRA adapters start identical across ranks (D29).
    - Log effective global batch `world_size * config.batch_size` and emit a warning when `config.learning_rate is None` and effective global batch `> 32` (D27).
    - Compute `iters_per_epoch = math.ceil(len(train_dataset) / (world_size * config.batch_size))` loop iterations per epoch and the corresponding global `total_steps` budget (D26).
  - `TimestralTrainer._raw_loader` and `_loader`:
    - In multi-process mode, bypass `_get_head_cache` in Stage 1 and log that `head_only` is streaming (D32).
    - Use a main-process epoch batch sampler derived from `(config.seed, epoch)` (or natural order when `shuffle=False`) giving `config.batch_size` indices per rank on full steps and splitting the final partial global batch evenly (sizes differ by at most 1; empty only when `< N` windows remain), yielding only non-empty index lists to `DataLoader(..., batch_sampler=...)` so `DHFDataCollator` is never called empty and `num_workers > 0` / `persistent_workers=True` work cleanly (D26).
  - `TimestralTrainer.training_step` and internal distributed step helper:
    - Public `training_step()` raises `RuntimeError` in an active multi-process group directing callers to `fit()` (D23, D31).
    - Internal distributed step helper invoked by `fit()` implements the five-stage step (D22, D23, D24): local stage (empty ranks skip `DataLoader` fetch, sign flip, `batch.validate`, and forward; local exceptions caught into error flag) → 8-scalar `float64` `ReduceOp.SUM` statistics collective → globally decided local-error propagation, `FloatingPointError` (all ranks), or global skip (`W_D == 0`, preserving one-process skip semantics: no backward, no optimiser/scheduler step, `global_step` unchanged, `skipped_batches` incremented, identical per-epoch loop iterations across ranks) → local backward of `(W_d / W_D) * L_d` (backward failures fall under D35, not a gradient-buffer flag) → flat-buffer `ReduceOp.SUM` gradient collective (`None` grads filled with zeros, no division by `N`) → `clip_grad_norm_` → `optimizer.step()` → `scheduler.step()`.
    - Preserve the collective-safety invariant: no rank-local branch may skip or add a collective (D23).
  - `TimestralTrainer.fit`:
    - Drive each epoch by global iteration index (`0..iters_per_epoch - 1`), fetching from the `DataLoader` only when the rank's shard is non-empty (D26).
    - Seed epoch permutations identically across ranks; offset `_fit_flip_generator` and global `torch.manual_seed` (LoRA dropout) so rank 0 keeps today's seeds and distinct ranks get distinct streams (D33).
    - Run `self.callbacks` (`ProgressLogger`, `EarlyStopping`, user callbacks), best-loss tracking, early stopping, and the restore decision on rank 0 only (non-zero ranks never compare validation losses); broadcast a status (`continue` / `stop` / `rank-0 callback failed`) after `on_fit_start`, `on_step_end`, and `on_evaluate` (D30).
    - At the end of `fit()`, broadcast `(best_eval_loss, best_step, restored_best)` from rank 0 and, only if `restored_best` is True, broadcast restored `self._trainable`; construct equal `TrainingResult` across ranks (spec §5.6, D30).
    - Sharded validation inside `fit()`: separate internal path and sharded loader (never `self._val_loader`); disjoint stride sharding (`r, r + N, …`), empty or zero-scored shards contribute zeros, one end-of-eval `float64` `ReduceOp.SUM` collective over `[loss_sum, scored_positions, batches, err]`, raising `"the dataset has no scored position to evaluate"` only after reduction from the global count on every rank (D31).
  - `TimestralTrainer.evaluate` and `TimestralTrainer.export`:
    - `evaluate()` remains rank-local and collective-free on the full dataset (unchanged).
    - `export()` stays collective-free: performs identical argument validation on all ranks; rank 0 writes the export files (passing `world_size` and `effective_batch_size` for `tune_metadata.json`), while non-zero ranks skip Hub calls, weight merges, and disk writes, log that rank 0 writes the files, and return the same `Path` (D31).
- **`export.py`**
  - Accept optional multi-process metadata (`world_size`, `effective_batch_size`) in `export_checkpoint` for `tune_metadata.json` (proposed addition to spec §7.4).
- **Acceptance tests (`tests/acceptance/test_distributed.py`)**
  - Implement `test_t15_*` covering cases `(a)–(i)` in `docs/SPEC.md` §9 (including the non-zero-rank `export()` guard in case `(h)`).

### Stage 2 — Replay, distillation, and head-cache decision

- **`trainer/trainer.py`**
  - `_setup_replay_teacher`: run after rank-0 all-parameter/buffer broadcast (D29). How the frozen-base teacher output cache (`_replay_base_pos` / `_replay_base_neg`) is built across ranks is an open Stage 2 question.
  - `_replay_loader` and `_compute_replay_terms`: `ReplayConfig.batch_size` is per rank (D27); shard `replay_dataset` per global step with the balanced epoch sampler (rank-offset `_replay_flip_generator`, D33), populate `[W_r, sum_r, M, sum_m]` in the stats collective, decide replay mixing globally from `W_R > 0`, and weight `L_replay` by `f * (W_r / W_R)` and `L_distill` by `distill_weight * (M / M_tot)` (D23, D24). Open Stage 2 decision (not settled; current leaning: a small pre-forward count collective used only when `replay_config` is set, leaving Stage 1 unaffected): scored counts are known from the CPU batches before any forward pass, so checking `W_D == 0` first preserves exact one-process skip semantics (`trainer/trainer.py`) without advancing `replay_iter`.
  - `head_only` frozen-feature cache (open Stage 2 decision, not a commitment): decide whether a multi-process cache is worth having, given that per-epoch global permutations move windows between ranks across epochs (so a per-rank cache cannot serve later epochs without holding the whole dataset on every rank or exchanging caches across ranks).
  - Extend `test_t15_*` to cover multi-process `ReplayConfig` (matching the four replay/distillation regimes in `scripts/probes/distributed_parity.py` §4) and any Stage 2 caching adopted.

### Optional shared refactor (if gradient accumulation precedes or joins Stage 1)

- **`losses/pinball.py`** (`pinball_loss`) and **`replay/distill.py`** (`distillation_loss`): expose an internal `(loss_sum, active_count)` helper so gradient accumulation and multi-process count weighting share one denominator-free path without changing public signatures.

---

## 3. Hardware verification session plan (Stage 3: CUDA / NCCL)

Nothing on CUDA or NCCL has been run yet. Before declaring multi-process training verified, run one focused verification session on Linux CUDA hardware with 2 GPUs and 4 GPUs (note: a single 4-GPU VM can run both `N = 2` and `N = 4` tests).

### What to run

1. **Preflight:** run `bash scripts/cuda_check.sh` to verify compiler/headers and baseline single-GPU suite health.
2. **Prototype probe on NCCL:**
   - `uv run python scripts/probes/distributed_parity.py --device cuda` (`tiny_random`, `N = 2` and `N = 4`, NCCL).
   - `uv run python scripts/probes/distributed_parity.py --device cuda --sizes 5M 28M` (released weights on NCCL; measure cross-rank diffs and body-backprop diffs vs union reference).
3. **Trainer acceptance & launcher checks (once Stage 1 lands):**
   - `torchrun --nproc_per_node=2` and `torchrun --nproc_per_node=4` running `T15` (`test_t15_*`) on CUDA/NCCL.
   - Smoke tests for `mixed_precision="bf16"`, `compile_mode="default"` (including epochs with empty trailing shards), and `num_workers > 0` on Linux.
   - Rank-failure smoke test: launch with a short explicit `timeout` (e.g. `15 s`), terminate rank 1 mid-step, and verify surviving ranks abort cleanly at timeout rather than hanging indefinitely.
4. **Throughput & scaling micro-benchmark:**
   - Measure 1-GPU vs 2-GPU vs 4-GPU step time, samples/s, peak VRAM, and flat-buffer `all_reduce` wall-time share on `28M`, `123M`, and `209M` (`full_ft` and `lora_temporal`, `bf16`, eager and `compile_mode="default"`, at `ctx = 256` and `ctx = 1,024`–`4,096`).

### What to record (in `docs/VERIFICATION.md` §16)

- Exact VM type, GPU count, CUDA/NCCL/PyTorch/Python versions, and commands.
- Probe pass/fail counts and `[INFO]` ranges on NCCL for `tiny_random`, `5M`, and `28M` (plus multi-seed and cosine-similarity measurements used to set released-weights T15 bounds).
- Whether NCCL cross-rank gradients and final parameters remain bitwise `0.0` or exhibit non-zero reduction variance.
- `bf16` and `compile_mode="default"` pass/fail results, plus rank-failure timeout exit behaviour.
- Measured 1 / 2 / 4 GPU step times, scaling efficiency, and `all_reduce` fraction of step time.

### What findings would change the design

- **Communication overhead (revisiting D22):** if unoverlapped flat-buffer `all_reduce` accounts for a material fraction of step time (for example `> 15–20%` on `123M`/`209M` `full_ft` at realistic batch sizes), benchmark a thin `nn.Module.forward` wrapper under `DistributedDataParallel` to overlap communication with backward.
- **NCCL determinism:** check whether NCCL cross-rank gradients and final parameters remain bitwise `0.0`; if NCCL `all_reduce` exhibits non-zero cross-rank variance, relax T15's NCCL cross-rank tolerance and evaluate periodic rank-0 parameter broadcast.
- **Empty shards under `compile_mode="default"`:** if compiled graphs fail or recompile excessively on short/empty trailing shards, restrict `compile_model=True` in multi-process mode or pad/guard trailing steps.
- **Released-weights T15 bounds:** set released-weights bounds in spec §9 from Stage 3 measurements across sizes (`5M`/`28M`/`123M`/`209M`), seeds, and devices—including cosine similarity, which `scripts/probes/distributed_parity.py` does not yet report.

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

- **2026-11-23 (proposal & prototype probe):** wrote the documentation-only single-node multi-process design proposal across `docs/SPEC.md` (§5.6, §7.4, §9 T15 cases `(a)–(i)`, §10 Phase 5, §11 D22–D35, §12), `docs/VERIFICATION.md` (§11 one-off `ReduceOp.AVG` check and §16), `docs/STATUS.md`, and these notes, backed by the standalone CPU/`gloo` prototype probe `scripts/probes/distributed_parity.py`. Final probe totals on macOS arm64 CPU/`gloo`: `326/326` passed (`0` failed) on `tiny_random` on both PyTorch `2.15.0` and `2.4.0` (verified against both `probe_ref` and independent one-process `TimestralTrainer` `shipped_trainer`, covering scalar global loss, exact `0.00e+00` cross-rank parity, LoRA step-0/1 invariants, empty-rank-0 delivery, global skip, non-finite propagation, and 10-active-step + 1-skip AdamW trajectories with `wd=0.01`, warmup + cosine, and clip `1.0` across `full_ft` and `lora_variate`); and `516/516` passed (`326` `tiny_random` + `190` enforced `Timestral-5M`), `0` failed, `68` ungraded `[INFO]` `Timestral-5M` body-backprop values on PyTorch `2.15.0` (`--sizes 5M`). Nothing in `src/` or `tests/` implements multi-process training; awaiting maintainer go/no-go and sequencing against gradient accumulation.
