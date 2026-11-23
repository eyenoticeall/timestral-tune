# Phase 5 — Multi-Process / Multi-GPU Training

**Status:** Approved by the maintainer on 2026-11-23; implementation in progress, nothing implemented yet in `src/` or `tests/`.
**Goal:** Data-parallel multi-process training across `N` ranks on a single node (CUDA/NCCL and CPU/`gloo`) with exact step-level equivalence to single-process training on the union batch.
**Spec sections:** `docs/SPEC.md` §5.6, §7.4, §9 (T15), §10 (Phase 5), §11 (D22–D35), §12; prototype evidence in `docs/VERIFICATION.md` §16.

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

**Status:** Phase 5 is approved (2026-11-23) and implementation is starting; gradient accumulation remains an unticked, unscheduled option (`docs/SPEC.md` §10).

---

## 2. Module and function map

All behavioural rules live in `docs/SPEC.md` §5.6, §7.4, and D22–D35; this section maps those rules to modules and functions.

### Stage 1 — Core multi-process training

- **`models/loader.py`** — unchanged (`resolve_device`, `load_model`, `load_forecaster` stay pure and collective-free; Stage 1 checks `huggingface_hub==0.23` lock behaviour, §12).
- **`trainer/__init__.py`** — export `RankFailureError` (`RuntimeError` subclass).
- **`trainer/trainer.py`**
  - `TimestralTrainer.__init__` — process-group detection & snapshot, exact backend string check, pre-collective device resolution/binding, startup sequence `(a)–(e)` (rank-local setup/validation + single trainable dtype check, startup `int64` failure collective, integer consistency digest + collective, all-parameter/all-buffer broadcast, post-broadcast setup & pre-allocated flat gradient buffer, one-time logs), and global `total_steps`.
  - `TimestralTrainer._raw_loader` / `_loader` — main-process contiguous epoch batch sampler (balanced trailing split, non-empty slices only) and `head_only` cache bypass.
  - `TimestralTrainer.training_step` + internal distributed step helper — public `training_step()` raises `RuntimeError`; internal helper implements stages 1–5 (local skip on empty/`num_scored == 0`, 8-scalar `ReduceOp.SUM`, conditional length-`N` `int64` rank vector, count-weighted backward, reused flat-buffer `ReduceOp.SUM`).
  - `TimestralTrainer.fit` — process-group liveness/`world_size` check, global-iteration loop with lazy per-epoch iterator creation, rank-offset RNG streams, `on_fit_start` inside `try/finally`, rank-0 callbacks/best-loss/restore with fixed-size `float64` `[code, value]` and conditional `[has_best, best_eval_loss, best_step, restored]` broadcasts.
  - Internal sharded validation helper (separate from `self._val_loader`) — `eval()`/`no_grad()` restoring `training`, cached stride loader skipped on empty shards, per-rank `eval_batch_size`, `[loss_sum, scored_positions, batches, failed]` reduction.
  - `TimestralTrainer.evaluate` — unchanged (rank-local, collective-free).
  - `TimestralTrainer.export` — uses snapshotted `rank`/`world_size`, calls shared validation helper on all ranks; rank 0 writes files with `world_size`/`effective_batch_size`, non-zero ranks log, no-op, and return `Path`.
- **`export.py`** — extract input validation into a shared helper called by `export_checkpoint` and non-zero-rank `export()`; write top-level `world_size` and `effective_batch_size` to `tune_metadata.json` only when `world_size > 1`.
- **`tests/acceptance/test_distributed.py`** — `test_t15_*` covering cases `(a)–(i)` and harness rules in `docs/SPEC.md` §9.

### Stage 2 — Replay and distillation

- **`trainer/trainer.py`** — `_setup_replay_teacher` (uncached paths only, logged once), `_replay_loader` (per-rank cycling loader with rank-offset shuffle seed and per-rank `ReplayConfig.batch_size`), internal distributed step (replay-only 1-element pre-forward domain count collective + `[W_r, sum_r, M, sum_m]` terms), and Stage 2 T15 coverage.

### Optional (unscheduled): gradient accumulation

- **`losses/pinball.py`** (`pinball_loss`) and **`replay/distill.py`** (`distillation_loss`) — internal `(loss_sum, active_count)` helper if gradient accumulation is scheduled later.

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

- **2026-11-23 (proposal & prototype probe):** wrote the documentation-only single-node multi-process design proposal across `docs/SPEC.md` (§5.6, §7.4, §9 T15 cases `(a)–(i)`, §10 Phase 5, §11 D22–D35, §12), `docs/VERIFICATION.md` (§11 one-off `ReduceOp.AVG` check and §16), `docs/STATUS.md`, and these notes, backed by the standalone CPU/`gloo` prototype probe `scripts/probes/distributed_parity.py` (`326/326` passed on `tiny_random` on PyTorch `2.15.0` and `2.4.0`; `516/516` passed + `68` `[INFO]` on `--sizes 5M`).
- **2026-11-23 (approved & pre-implementation design revision):** maintainer approved Phase 5 (implementation starting; gradient accumulation stays unscheduled). Updated `docs/SPEC.md`, `docs/VERIFICATION.md`, `docs/STATUS.md`, and these notes to lock down exact backend checks, pre-collective device binding, `__init__` startup failure/consistency collectives and rank/world-size/backend snapshot, binary failure flags + conditional `int64` length-`N` vectors, `Exception`-only / OOM-excluded propagation and `RankFailureError`, fixed-size `float64` control broadcasts, contiguous slices, local zero-scored skip order, `(base + r * 1_000_000_007) mod 2^63` rank seeds, reused flat gradient buffer, sharded validation `eval()`/`no_grad()` & shared `export()` validation, Stage 2 decisions (per-rank cycling replay loaders, replay-only pre-forward count collective, no multi-process head/teacher caches), and tightened T15 cases `(a)–(i)`; trimmed repeated numbers and cut §2 to a module/function map.
