# Development Log

Append-only, oldest first. One entry per working session. Keep entries factual: what was done, what was
decided and why, what evidence exists, what is left open. Decisions are recorded in full in the handoff,
§11; link to them rather than restating them.

---

## 2026-11-21 — Audit, specification rewrite, Apple GPU check, scaffold

**Starting point.** The repository held a licence, a two-line README and a draft handoff document.

**1. Audit of the draft specification.**
Checked it against the Timestral paper (arXiv:2610.37585), the official `mistralai/timestral` code and the
released 5M checkpoint. The foundations were real, but the draft could not be implemented as written.
Main findings:

- The LoRA target modules it named do not exist; attention uses one fused `wQKV` projection.
- It described "Decoupled Horizon Masking"; the method is Decoupled Horizon Forecasting, and its point is
  that context is not masked.
- The loss section omitted that the loss is computed in the model's normalised space.
- Its gradient-leakage test was mis-specified.
- The weights licence has a revenue cap that also covers fine-tuned derivatives.
- The Lacuna integration pointed the wrong way: Lacuna emits bundles, it does not consume them.
- The planned PyPI dependency on `timestral` is impossible; it is not published there.

The original draft is kept at `archive/timestral_tune_handoff_document.v0.md`.

**2. Specification rewrite.**
Rewrote `docs/SPEC.md` on the original's structure, with provenance tags on every claim, a model-facts section,
acceptance tests T1–T13, a decisions log (§11) and open risks (§12). Decisions D1–D10 were made here.
The one resting on interpretation rather than evidence is D2 (the target frame).

**3. Apple GPU.**
Found that the official code fails on the Apple GPU because four sites use float64. Running those on CPU
fixes it. Measured forward and gradient parity and a 1.4–2.7× faster training step. Moved Apple GPU
support into v0.1 (decision D9, acceptance test T14, fp32 only).
Side finding: CPU fp32 gradients are less accurate than the Apple GPU's against a float64-weights
reference, cause unknown. Recorded in handoff §12 and `docs/VERIFICATION.md` §6.

**4. Scaffold.**

- Vendored upstream at commit `a534bdb` with `scripts/vendor_upstream.py`. Upstream files import each
  other absolutely, so the script rewrites those imports to relative ones; this is the only modification.
  The spec previously said "unmodified" and was corrected.
- Package skeleton under `src/timestral_tune/`: documented stubs only.
- Tests: 6 scaffold checks, 14 skipped acceptance placeholders.
- Ported the ad-hoc verification scripts into `scripts/probes/` and re-ran them against the vendored code;
  the specification's measured numbers were updated to the reproducible runs.
- CI and drift workflows written, not run (nothing pushed).
- Process docs: `AGENTS.md` and `docs/`.

**Evidence.** `uv run pytest`: 6 passed, 14 skipped, on Python 3.13 and 3.10. Lint clean. Wheel builds and
contains the vendored licence. Drift check: no drift. Probe results: `docs/VERIFICATION.md`.

**Commits.** The scaffold was committed as a series on `phase-0/scaffold`. The "every commit passes tests"
rule can only hold from the commit that adds the tests onward; earlier commits in the series add the
specification, vendored code and packaging before any test exists.

**Merge.** At the maintainer's prompt, `phase-0/scaffold` was fast-forwarded into `main`. This merges the
scaffold only; Phase 0 itself is still open.

**Correction.** Earlier notes in this session, and the body of the `ci:` commit, say the repository has no
remote. That was wrong: a private `origin` exists and held only the initial commit. The accurate statement
is that nothing had been pushed. The docs were corrected; the commit message was left as is.

**First push and CI.** `main` was pushed to the private `origin`. The first CI run passed: lint, build, and
the network-free tests on Python 3.10, 3.11, 3.12 and 3.13 (Linux runners). GitHub flagged the checkout and
setup-uv actions as targeting a deprecated Node.js version; they still ran.

**Rename.** The specification was renamed from `timestral_tune_handoff_document.md` at the repository root to `docs/SPEC.md`, since it
is no longer a handoff note but the project's standing specification. References across the repo now say
"spec". Earlier entries in this log, and older commit messages, use the old name and the word "handoff".

**Branch policy.** The maintainer asked for fewer branches: work on `main`, and reserve branches for
phases, new versions and large adaptations. `AGENTS.md` §6 was rewritten accordingly (it previously
required a branch per unit of work), and the two merged branches were deleted.

**Left open.**

- Phase 0 is not finished: loader, device shims and T14 are still stubs.
- The target-frame check (D2) has not been run.
- The CPU gradient gap is unexplained.
- Nothing has run on CUDA or on the PyTorch 2.4 floor. Linux has run only the scaffold tests, in CI.
- The weekly drift job has not had its first scheduled run.

---

## 2026-11-21 (second session) — README banner; Phase 0 remainder and Phase 1

**Banner.** A subagent drew a pixel-art README banner in code (`scripts/make_banner.py`, `assets/banner.png`).
The first version's zoomed crop read as disconnected blocks; the maintainer asked for a redo with QA loops,
and the second version (a plain framed hourglass whose sand settles into rising bars) replaced it. Both
commits are on `main`, unpushed.

**Phase 0 remainder.** `models/loader.py` and `models/device.py`: pinned loading and two subclasses that
run upstream's four float64 sites on CPU where the device has no float64. No vendored file is touched.

**Phase 1, in the order built.**

1. `models/forward.py`, `data/batch.py`: the normalised-space training forward for both layouts, the two
   target frames, and the packed batch layout. T1–T3.
2. `losses/pinball.py`, `trainer/step.py`: the loss and the one function that composes forward, frame and
   loss. T8.
3. Target-frame check (`scripts/probes/target_frame_check.py`), before any trainer existed.
4. `data/`: encoders, collator, dataset. T4–T7, T12.
5. `utils/validation.py`: the acceptance checks as reusable verifiers.
6. `config.py`, `trainer/`: `head_only` training end to end.
7. PyTorch floor runs; probes moved onto the library.

**Findings.**

- **D2 confirmed.** The checkpoints are calibrated in the median-folded frame: near-nominal coverage there,
  heavy over-coverage in the true-values frame. The decision rule written before the run also compared
  pinball loss and, by its letter, came out "inconclusive"; that criterion was ill-posed (each frame's loss
  is in its own units) and is recorded as such rather than rewritten.
- **The "CPU gradient gap" is not CPU-specific.** Float32 gradients through the released weights are off
  by up to 3.4e-3 on CPU and 1.1e-3 on the Apple GPU; a random model is at 3e-7 on both. Still unexplained.
  T14's single 1e-3 gradient bound was replaced by two tiers (decision D12). An earlier statement in this
  project that Apple GPU gradients sit within 8e-5 of the reference is superseded: that was one batch.
- **The Apple GPU needs PyTorch 2.8.** Older versions lack `cummax` on that backend. The library now
  refuses it below 2.8 (decision in spec §5.4).
- **PyTorch 2.4.0 on macOS hangs at exit after a DataLoader worker has run.** First misread as a test
  hanging on the Apple GPU (the commit message of the PyTorch 2.8 gate says so; it is wrong on that point).
  A stack sample and `lsof` showed the interpreter waiting at shutdown for the multiprocessing resource
  tracker, whose pipe PyTorch's `torch_shm_manager` kept open. It reproduces without this library
  (`scripts/probes/worker_exit_hang.py`) and not on 2.8.0. One test is skipped on that combination.
- **Bit-identity is not a safe assertion across PyTorch versions.** T1/T2 are bit-identical on 2.4.0 and
  2.15.0 but one case differs by 8e-10 on 2.8.0. The bound is now 1e-6.
- **`split_by_time` semantics were tightened** (decision D11): training gets nothing at or after the
  cutoff, so labels cannot overlap even without an embargo. The spec's rule 5 was reworded.

**Bugs the tests caught during development.** Three float64 conversions executed on the Apple GPU before
moving to CPU (in the frame computation, the device-order of a `.to()`, and the evaluation loop's
accumulator). All three were found by running on the device, none by reading.

**Evidence.** 268 passed, 4 skipped on PyTorch 2.15.0 (Python 3.13), on 2.14.1 (Python 3.10) and on 2.8.0
(Python 3.12); 231 passed, 22 skipped on 2.4.0 (Python 3.12). T1–T8 and T12 pass on 5M and 28M, which is Phase 1's exit criterion.
Numbers: `docs/VERIFICATION.md`.

**Left open.**

- T14 on CUDA: no hardware. One command when there is.
- Gradient cosine against the reference was first recorded as above 0.999999 on both devices without a CPU
  measurement. Measured since: one minus the cosine is at most 5.7e-6 on CPU and 6.4e-7 on the Apple GPU.
  The docs carry the measured values.
- Validation loss under `median_folded` is in units that move with the model (spec §12); unmeasured.
- `phase-1` is not merged or pushed; CI has not run on it.
- The gradient-accuracy cause.
- The target-frame evidence is synthetic only.
- Phase 2 has not started. There is no export, so a tuned model cannot be saved in the upstream layout yet.

---

## 2026-11-21 (third session) — T14 on CUDA

**What.** The maintainer approved a short-lived Google Cloud GPU VM. T4s were out of stock in every zone
tried; a watcher loop got an NVIDIA L4 after a few minutes. The VM ran for about half an hour and was
deleted. `scripts/cuda_check.sh` is the whole procedure.

**Results.** T14 and the whole suite pass on CUDA: 127 device tests, then 290 passed, 6 skipped overall, at
commit `ecbab9d` (PyTorch 2.15.0+cu132, Python 3.12). Forecasts match CPU to 1e-7 to 1.5e-5. Numbers are in
`docs/VERIFICATION.md` §13. Phase 1's last roadmap box is ticked.

**Findings.**

- **CUDA float32 gradients on released weights are the least accurate of the three devices**: up to 7.4e-3
  on T14's scenarios and 1.2e-2 on the probe's, against a bound of 1e-2. T14 passed, but with no margin.
- **Most of the gradient inaccuracy is the float32 backward pass of upstream's attention call.** With
  attention written out explicitly, CUDA drops from 1.2e-2 to 2e-5 and CPU improves likewise; float64
  weights on CUDA give 1e-5; TF32 and Triton FP fusion do nothing. This replaces "cause unknown" in spec §12
  with a located cause and a smaller unexplained remainder (one scenario, 28M with covariates).
- **Decision D12 revised** to three tiers: random model 1e-5, released weights with explicit attention
  5e-3, released weights as shipped 5e-2. The middle tier is the meaningful one for our own code.
- **CUDA needs a C compiler and Python headers** with recent PyTorch, because its Triton component compiles
  a helper on first use. The first run, on the VM's system Python 3.10, failed every CUDA test for that
  reason. The script now checks first.

**After the run.** A `head_only` gradient test was added: with the body frozen the backward pass never
enters attention, and head gradients match the reference within 2.6e-6 on CPU and the Apple GPU. So the
inaccuracy does not touch the one mode built so far. That test has not run on CUDA (the VM was gone). The
attention explanation holds for CUDA and CPU only; on the Apple GPU the swap changes nothing.

Connecting with `gcloud compute ssh` had added a login key to the cloud project's shared metadata. It was
removed after the VM was deleted.

**Final counts** (macOS): 316 passed, 4 skipped on PyTorch 2.15.0 (Python 3.13), 2.14.1 (Python 3.10) and
2.8.0 (Python 3.12); 255 passed, 22 skipped on 2.4.0.

**Not done / open.** The library still trains with upstream's attention call. Whether explicit attention
changes training results is untested; if it matters, it belongs with Phase 2's attention wrapping for LoRA.
`bf16` on CUDA is untested. `phase-1` is still unmerged and unpushed.

---

## 2026-11-21 (fourth session) — review of Phase 1 and fixes

**What.** The maintainer asked for a review workflow: five reviewers (forward and loss, data, trainer and
loader, tests and verifiers, spec and docs), each followed by a skeptic that tried to refute its findings.
22 findings survived (21 distinct: none high, 8 medium, 13 low) and 4 were refuted. All 21 are fixed.

**Defects in the library.**

- **Category codes depended on future data with several targets.** Codes were numbered over the whole
  frame, and with more than one target they are model inputs. Windows now number their categories among
  those they can see (decision D14, spec §6.3 rule 4).
- **Time-zone-aware timestamps from polars and pyarrow were accepted silently** and converted to UTC, moving
  local-time split cutoffs. The check now runs on the frame's column type.
- Smaller: `PinballLoss` could not be moved to the Apple GPU (float64 buffer); integer-dtype batches passed
  validation; `fit()` raised if a callback stopped it before the first step; callbacks were documented as
  duck-typed but are not; the progress logger miscounted on a second `fit()`; `from_upstream` rounded
  float64 weights through float32; the config rejected NumPy integers; `check_loss_mask` raised on a fully
  frozen model.

**Tests that could not fail.** The checkpoint fixture turned every loader error into a skip; the time-split
test used one shared time axis; the `true_values` frame was only checked at the patch it is assigned from.
All three are fixed, and each new test was run against the reviewer's mutation and seen to fail.

**A finding of our own, while fixing.** Adding the missing past-only-covariate scenario made one T14 case
fail (explicit attention, Apple GPU, 5M: 5.9e-3 against 5e-3). The bound was not loosened. The reference
was at fault: upstream's RMSNorm computes in float32 even in a float64 model, and in that scenario this
moved the "float64" reference by 3.8e-3. References now compute RMSNorm in float64, and every local
gradient number was re-measured. This is a second identified source of the float32 gradient inaccuracy,
next to the attention call.

**Documents.** Spec §6.2 now describes what the collator does with long and short contexts; §9 no longer
says every acceptance test is a placeholder; §12 says which evidence comes from which probe or test; the
Hub pins are documented where they live, with a test that keeps the table and the code in step; the PyTorch
2.8 floor has a probe.

**Evidence.** 405 passed, 4 skipped on PyTorch 2.15.0 (Python 3.13), 2.14.1 (Python 3.10) and 2.8.0 (Python
3.12); 331 passed, 26 skipped on 2.4.0.

**Not done / open.** None of this has run on CUDA; the CUDA gradient numbers in `docs/VERIFICATION.md` §13
predate the improved reference. The earlier claim "no leakage on the training path" was too strong: it
held for what the tests covered, which did not include several targets with categoricals or zoned
timestamps. `phase-1` is still unmerged and unpushed.

---

## 2026-11-21 (fifth session) — `attention_backend="math"`, large-logit analysis, and review fixes

**What.** On a new branch `fix/math-sdpa-backend` off `phase-1`, investigated the float32 gradient
inaccuracy, added an opt-in `attention_backend="math"` option without touching `_vendor/`, and then
addressed all findings from a review of the branch.

**Measured cause.** The released checkpoints have large Q/K spectral norms (5M layer 5: `||norm.scale||_∞
= 5.01`, `||Wq||_2 = 141.6`, `||Wk||_2 = 82.4`; 28M layer 1: `||norm.scale||_∞ = 2.10`, `||Wq||_2 = 196.1`,
`||Wk||_2 = 160.6`), producing peak attention logits of 370k–404k in 5M and 203k–503k in 28M across the
probe layouts (and same-row near-ties with `gap < 10` up to 335k in 5M and 37.5k in 28M), where one float32
ULP is `0.015625` below 262k and `0.03125` from 262k to 524k. That produces two effects:

1. Fused SDPA backward (CPU `FLASH_ATTENTION`, CUDA fused SDPA) loses accuracy on typical batches (4.6e-4
   to 2.1e-3 on CPU, up to 1.2e-2 on CUDA). `attention_backend="math"` cuts CPU probe error 2.5×–79× (to
   1.1e-5–8.3e-5 on seven layouts and 3.6e-4 on 28M covariates), drops the 15-seed 5M past-only median 85×
   (1.05e-3 to 1.24e-5), and drops 8 of 10 T14 CPU scenarios to 4.0e-6–3.9e-5. On the Apple GPU `"math"`
   is identical to `"default"`.
2. When two keys in the same query row sit at large logits within an O(1) gap, float32's ULP quantizes the
   gap in *any* float32 attention kernel. Consequently `"math"` can be *less* accurate than `"default"` on
   individual batches: on T14's 5M `past_only_covariates` batch, `"math"` gives 9.0e-3 on CPU against
   2.2e-3 for `"default"` and 1.2e-3 for explicit matmul-and-softmax (`1 - cosine = 3.7e-5`). A one-off
   trace (`docs/VERIFICATION.md` §11) showed 100% of layer 5's decoder `dq` error on that batch sits at one
   query `(row=3, head=2, query=11)` whose top-2 keys sit at ~351k with true gap 3.4577, where `"math"`
   quantized the gap 2 ULPs to 3.375 (`+8.3%` on `p2`).

**Review findings fixed on this branch.**

- **Restored T14's strict `5e-3` middle tier** (`test_t14_gradients_with_explicit_attention` using
  `tests/helpers.py::explicit_attention`, never loosened) and added a separate
  `test_t14_gradients_with_math_sdpa` (`2e-2` rel L2, `1e-4` on `1 - cosine`).
- **Restored `manual_sdpa` in `scripts/probes/gradient_accuracy.py`** (preserving the CUDA reproducer in
  `docs/VERIFICATION.md` §13), fixed Experiment [5] so peak logits and near-tie gaps are strictly
  **same-row** (correcting the earlier cross-row error that misdescribed 28M covariates' 503k peak as a tie
  when its own gap is 20.2k), and added Experiment [6] (15-seed 5M past-only CPU sweep).
- **Moved one-off inspection figures to `docs/VERIFICATION.md` §11** instead of tagging unreproducible
  figures as `[V]`, reconciled all numbers across probe docstring and docs, and qualified docstrings, spec
  §5.4, D15 and `STATUS.md` honestly (including that `"math"` has no gradient effect in `head_only`, is a
  no-op on MPS, and can regress on individual near-tie batches).
- **Fixed overlapping-thread flag corruption in `attention_context`**: uses a lock-guarded reference counter
  over `torch.backends.cuda.enable_*_sdp` (avoiding `sdpa_kernel`'s cross-thread `ContextVar` exit failure)
  with an overlapping-thread unit test, while documenting that PyTorch's SDPA flags remain process-global.
- **Plumbed `attention_backend` through `utils/validation.py`** (`run_checks` and every individual verifier)
  with a device-parametrised unit test.

**Evidence.** `uv run ruff check . && uv run ruff format --check .` clean. Full test suite: 444 passed, 4
skipped on PyTorch 2.15.0 (Python 3.13), 2.14.1 (Python 3.10), and 2.8.0 (Python 3.12); 353 passed, 26
skipped on PyTorch 2.4.0 (Python 3.12), macOS. Probe numbers recorded in `docs/VERIFICATION.md` §6.

**Not done / open.** `attention_backend="math"` and the updated T14 / probe have not yet been run or timed
on CUDA. `phase-1` and `fix/math-sdpa-backend` remain unmerged and unpushed.

**Afterwards: the branch was split.** A second review and the advisor agreed the investigation was worth
landing and the option was not, yet: it changes no gradient that runs today, is unverified on CUDA, and may
be the wrong mechanism next to explicit attention. At the maintainer's direction:

- `phase-1` took the measurements, the probe, the T14 tiers and `attention_context` as an experimental
  helper that no training path uses (spec D15 records the deferral and what should reopen it).
- `fix/math-sdpa-backend` keeps only the public `attention_backend` option and its plumbing through the
  config, trainer, loss step and verifiers, with their tests, on top of `main`.

Where this entry says the option was "added", read: added on that branch. Suite on `phase-1` after the
split: 438 passed, 4 skipped on PyTorch 2.15.0, 2.14.1 (Python 3.10) and 2.8.0; 349 passed, 26 skipped on
2.4.0.

**Merged.** `phase-1` was rebased on `main`, pushed, and run through CI as pull request #1: lint, build and
the network-free tests passed on Python 3.10 to 3.13 on Linux, the first CI run on any Phase 1 work. `main`
was fast-forwarded and pushed, and `phase-1` deleted locally and on the remote. `fix/math-sdpa-backend` is
one commit on top of `main`, pushed, and stays open. Phase 2 has not been started.

---

## 2026-11-21 (sixth session) — CUDA re-run before Phase 2

**What.** With the maintainer's clearance, a second short-lived Google Cloud VM (one NVIDIA L4, about 35
minutes, deleted afterwards). The login key was scoped to the instance this time, so the project's shared
metadata was not touched. Three things ran: `scripts/cuda_check.sh` on `main`, a one-off timing of each
attention kernel, and the suite of the held branch `fix/math-sdpa-backend`.

**Results.** At commit `50c170a`: 218 device tests and then 436 passed, 6 skipped overall; the held branch
443 passed, 6 skipped. `head_only` gradients on CUDA are within 2.2e-6 of the reference. Numbers are in
`docs/VERIFICATION.md` §6 and §13.

**Findings.**

- **On CUDA the unfused kernels remove most of the gradient error**: 50 to 900 times lower across the eight
  probe layouts, for PyTorch's unfused kernel and for explicit attention alike, at 5% to 25% more step
  time and at most 0.2 GiB more memory at a context of 1,024. This is the CUDA evidence decision D15 was
  waiting for. The option stays deferred: it still changes nothing in `head_only`, and whether it changes
  training results is untested.
- **One T14 case failed on the VM's x86 CPU, not on CUDA**: explicit attention on 5M with past-only
  covariates, 5.3e-3 against 5e-3, where the Mac gives 1.2e-3. It is the batch with a near-tie at a logit of
  351k. Across four device and three kernel combinations it lands between 1.2e-3 and 1.1e-2, and the worst
  kernel differs by machine. That case now has its own 2e-2 bound, and PyTorch's unfused kernel is held to
  the same 5e-3 as explicit attention on the other nine (it was 2e-2). Decision D12 records the history of
  this criterion; this is the third time a bound turned out to be about one batch and not about our code.

**Not done / open.** A short training run under each kernel (needs a mode that trains body weights), and
the kernels' cost at long contexts. Phase 2 has not been started.

---

## 2026-11-21 (seventh session) — Phase 1 kernel optimisations and L4 CUDA verification

**What.** After profiling the current Phase 1 hot paths on CPU and Apple GPU, implemented the Phase 1-applicable
kernel and pipeline optimisations on a new branch `perf/phase1-kernels` off `main`, verified locally across all
four PyTorch/Python environments, and ran a short-lived NVIDIA L4 CUDA VM (instance-scoped SSH key, deleted
immediately after) for full `scripts/cuda_check.sh` verification and same-VM A/B timing against `main`.

**Optimisations implemented (decision D16, zero `_vendor/` edits):**

- **Trimmed covariate layout (`models/forward.py`):** dropped the `context_patches` inactive decoder tokens
  from upstream's `2(L + h)` layout (`x_dec` has `h` tokens instead of `L + h`), building the `(wp + hp, wp + hp)`
  time mask from upstream's `enc_causal_mask`. Verified mathematically exact in float64 (1.1e-16 to 9.8e-15
  against `predict_block`) and bit-identical to slicing upstream's `build_two_stream_time_mask` with zero
  edges to dropped tokens. T2 pairs a `1e-5` float32 bound (measured `6.1e-8`–`5.2e-6` from GEMM tiling) with
  a `1e-12` float64 parity check and the mask-equivalence check.
- **Frozen-feature caching in `head_only` (`config.py`, `trainer/step.py`, `trainer/trainer.py`):**
  `TrainingConfig.cache_frozen_features=True` (256 MiB cap per dataset) caches per-window target-row horizon
  hidden states, anchors, targets, and 3-scalar float64 context Welford summaries (`context_welford_state` /
  `advancing_block_frames_from_state`) or pre-normalised targets/frames for `true_values` on CPU, skipping
  re-running the frozen transformer body across epochs and repeated evaluations.
- **Pipeline cleanups (`models/forward.py`, `trainer/step.py`, `trainer/trainer.py`):** short-circuited
  `target_frames` on single-patch horizons and computed multi-patch frames on target rows only; skipped
  `_known_cov_scaling` when `known_future` is None and short-circuited `supports_known_covariates` before the
  GPU sync; cached `quantile_knots` per `(knots, device, dtype)`; moved `TrainingBatch.validate` to CPU before
  `batch.to(device, non_blocking=self.device.type == "cuda")` and passed `validate=False` inside the loop.

**Measured results (same-machine before vs after):**

- **CUDA (NVIDIA L4, `step_timing.py`, batch 32, ctx 1,024, horizon 96):** 5M covariates `94 → 56 ms` (1.68×,
  VRAM `1.9 → 1.1 GiB`, −42%); 28M covariates `390 → 208 ms` (1.88×, VRAM `5.7 → 3.3 GiB`, −42%); univariate
  unchanged (`47 ms` 5M, `74 → 73 ms` 28M). Warmed-up 5-epoch `head_only` run with per-epoch validation on 28M
  covariates: `850 → 273 ms` (3.1×).
- **Apple M4 (`step_timing.py`):** 5M covariates `907 → 492 ms` CPU (1.84×), `474 → 276 ms` MPS (1.72×);
  28M covariates `6,216 → 1,845 ms` CPU (3.37×), `1,957 → 986 ms` MPS (1.98×, MPS memory `7.3 → 4.5 GiB`,
  −38%). 5-epoch `head_only` (24 covariate series, batch 16): 3.5×–4.3× faster across 5M/28M and CPU/MPS.

**Suite evidence.** Lint and format clean. Local macOS: 457 passed, 4 skipped on PyTorch 2.15.0 (3.13),
2.14.1 (3.10), and 2.8.0 (3.12); 363 passed, 26 skipped on PyTorch 2.4.0 (3.12). CUDA (L4, commit `9fc6735`):
230 passed, 2 skipped on device tests; 455 passed, 6 skipped overall (`head_only` gradients <= 1.7e-6).

**Not done / open.** `perf/phase1-kernels` is not merged or pushed yet. Phase 2 has not been started.

---

## 2026-11-21 (eighth session) — Phase 2 built and verified locally and on NVIDIA L4

**What.** Fast-forwarded `main` to `perf/phase1-kernels`, created branch `phase-2` off `main`, and
implemented all of Phase 2 (spec §10): Slice LoRA (`models/adapter.py`, `LoraConfig`), all adaptation modes
(`lora_variate`, `lora_temporal`, `full_ft`), `bf16` autocast + fused `AdamW` + optional `torch.compile` in
`TimestralTrainer`, post-hoc CQR and online ACI conformal calibration (`models/conformal.py`), and
upstream-loadable checkpoint export (`export.py`, `TimestralTrainer.export()`). Verified locally across all
four PyTorch/Python environments and on a short-lived NVIDIA L4 VM (instance-scoped SSH key, deleted
immediately after).

**Design & kernel highlights (zero `_vendor/` edits):**

- **Slice LoRA (`models/adapter.py`):** `LoRASliceLinear` fuses input projections across adapted `wQKV` slices
  into one `F.linear(x, cat(A))` call, uses out-of-place `chunk(3) + cat` (compile- and autograd-safe),
  accumulates `merge()`/`unmerge()` in at least float32, enforces the single-row guard (D5, T10), and strips
  `_orig_mod.` prefixes in `merged_state_dict` / `extract_lora_state_dict` so compiled models export cleanly.
- **Trainer (`trainer/trainer.py`, `config.py`):** supports `head_only`, `lora_variate`, `lora_temporal`, and
  `full_ft`; uses `AdamW(..., fused=True)` on CUDA; supports `mixed_precision="bf16"` on CUDA (rejecting CPU,
  MPS, and non-bf16 hardware); supports `compile_model=True` on `model.transformer` with fixed-shape
  enforcement for `compile_mode="reduce-overhead"`.
- **Conformal calibration (`models/conformal.py`):** `ConformalCalibrator` (CQR eqs. 9–11, finite-sample
  `(1 - alpha)(1 + 1 / n)` quantile allowing negative offsets) and `OnlineConformalCalibrator` (ACI,
  unclipped `alpha_t` per Lemma 4.1 with `.saturated` flag when `alpha_t <= 0`), both with scale
  normalisation, configurable horizon buckets, outward monotonicity repair from the unshifted median,
  NumPy/PyTorch type+device preservation, and JSON save/load.
- **Export (`export.py`):** writes `config.json` (exact upstream schema), merged `model.safetensors`,
  unmerged `adapter.safetensors` + `adapter_config.json` (LoRA only), `conformal.json`, `tune_metadata.json`,
  base `LICENSE` (copied verbatim from local dir or fetched from Hub), and derivative `NOTICE` — non-destructively.

**CUDA findings & D15 resolution (`docs/VERIFICATION.md` §13):**

- **`bf16` + `torch.compile` on L4 (ctx 1,024, batch 32, 5 rows covariates):** 28M `full_ft` dropped from
  `197.0 ms` / `3.28 GiB` (`fp32`) to `93.8 ms` / `2.30 GiB` (`bf16`) and **`59.8 ms` / `1.80 GiB`**
  (`bf16 + compile("default")`, **3.3× faster**, 45% less VRAM); 28M `lora_variate` dropped from `139.8 ms` /
  `1.63 GiB` to **`42.9 ms` / `0.94 GiB`** (**3.3× faster**).
- **Decision D15 closed:** 20-step AdamW trajectories on L4 across `lora_variate`, `lora_temporal`, and
  `full_ft` gave virtually identical validation losses between `fp32` default SDPA and `fp32` unfused math
  (identical to 5 decimal places on 28M: `0.09029`, `0.08629`, `0.09022`; within `0.0002` on 5M), and `bf16`
  also stayed within 0.1%–0.2%. No public attention-kernel option is needed.
- **Two Linux/CUDA test catches fixed (`3fe5bc9`):** x86 float32 gate layout moved 28M step-0 LoRA by `2.8e-6`
  (T9 now asserts `< 1e-12` in float64 + `< 1e-5` in float32); compiled export test now targets CUDA when
  available and skips CPU compile when no C++ compiler (`g++`/`clang++`) is installed.

**Evidence.** Ruff check & format clean; `uv build` clean; `check_upstream_drift.py` reports no drift.
All 14/14 acceptance tests pass (zero placeholders). Local macOS: **510 passed, 4 skipped** (CUDA-only
`bf16`) on PyTorch 2.15.0 (3.13), 2.14.1 (3.10), and 2.8.0 (3.12); **411 passed, 26 skipped** on PyTorch
2.4.0 (3.12). NVIDIA L4 CUDA (`3fe5bc9`): **512 passed, 2 skipped** (only Apple GPU tests skipped), plus
verbatim 1,687-byte Hub `LICENSE` export + upstream `strict=True` reload on `mistralai/Timestral-5M`.

**Review-fix pass (same session).** Deployed a four-agent parallel read-only review (LoRA/adapter,
trainer/`bf16`/`compile`, conformal calibration, export/licensing/docs) and resolved all Important findings
plus clean Minor/perf fixes:
- LoRA (`4ec2e46`): added `load_lora`/`load_lora_state_dict` (adapter reload round-trip), snapshotted
  pre-merge base rows so `unmerge()` restores exact bits (including `bf16` and post-merge mutations),
  added local `torch.Generator` support, warned on gate-less axes, and hardened nested-adapter checks.
- Conformal (`6a81d79`): rejected `N=0`/`H=0`, switched CQR to exact O(n) order statistic
  `ceil((n + 1)(1 - alpha))` via `np.partition`, added strict `from_dict` validation, made online
  `transform()`/`current_offsets()` read-only, scored ACI `err_t` against the actual emitted (repaired)
  interval, and added a `bisect`-sorted score mirror for O(1) quantile lookup.
- Trainer (`498851a`): moved `restore_best` snapshots to CPU (avoiding 2× VRAM in `full_ft`), used local
  RNG for LoRA init, enforced `reduce-overhead` fixed shapes in `evaluate()` and `_build_head_cache`,
  added `cudagraph_mark_step_begin()`, cached `_val_loader`, and promoted cached quantile levels to fp32.
- Export (`352f832`): made `dataset_fingerprint` hash actual timestamps/values, unlinked stale conditional
  artifacts on re-export, rejected unfitted calibrators, warned on fallback `LICENSE` with honest `NOTICE`,
  and recorded full hyperparameters + `model_provided_directly`.
- Probe & docs (`ed0df48`, `b0980cb`): added `--bf16`/`--compile`/`--mode` to `scripts/probes/step_timing.py`,
  fixed stale Phase 2 / D15 text, and after a second-pass 2-agent review (merge-ready, zero regressions) added
  `load_lora` resume support in `TimestralTrainer` plus strict ACI `from_dict` scalar checks and duplicate-slice
  rejection.
- Final NVIDIA L4 CUDA re-verification (`b0980cb`, short-lived VM deleted immediately after): `cuda_check.sh`
  passed with **249 passed, 2 skipped** on device tests and **521 passed, 2 skipped** overall, and
  `step_timing.py --device cuda --bf16 --compile default` / `--mode lora_variate --bf16 --compile reduce-overhead`
  reproduced the L4 timings (28M cov5: `62 ms` / `1.90 GiB` `full_ft`, `43 ms` `lora_variate`).

**Not done / open.** `phase-2` is pushed to `origin/phase-2` and awaits review/merge into `main`;
`perf/phase1-kernels` and `fix/math-sdpa-backend` have both been deleted locally and on `origin`.
Phase 3 has not started.

---

## 2026-11-22 — External review fixes (conformal narrowing & ACI wind-up, shape bucketing, export atomicity)

**What.** Addressed all findings from an independent review agent on `phase-2` (including three
mutation-test gaps) and verified a 5M LoRA export inside a standalone isolated install of the official
`mistralai/timestral@a534bdb` package (bit-identical forecasts: `max_abs = 0.0`, `4.9%` diff vs zero-shot):

- **Conformal calibration (`d0077ea`):**
  - Fixed Finding 1: replaced blind outward repair with `_repair_monotonicity_anchored` so uncalibrated
    interior levels (`q_0.85`, `q_0.15`) no longer undo negative offsets on single-coverage calibrators
    (default `ConformalCalibrator(0.80)` on 2.2× over-wide data now hits `80.1%` coverage instead of `97.6%`).
  - Fixed Finding 2: bounded ACI `alpha_t` at Lemma 4.1's theoretical envelope `[-gamma, 1 + gamma]`
    (still unclipped at `[0, 1]`), eliminating unbounded wind-up (`+2.84` / `-2.2`) from finite-window
    extremes (3,000 over-wide steps → `alpha_t = 0.1975`, immediate recovery on regime shift).
  - Fixed Finding 6a/6d: `OnlineConformalCalibrator.window` now counts forecast observations per bucket
    (500 forecasts at `bucket_size=32`, instead of ~15), and added `collect_conformal_data(forecaster, dataset)`.
  - Added ACI window-eviction test (mutation gap 1).
- **Shape bucketing & trainer (`dc9409f`):**
  - Fixed Finding 3: added `bucket_series_count`, `pad_batch_to_series` (zero-weighted, group-isolated
    tail padding, verified `< 1e-12` float64 loss/gradient equivalence), and `DHFDataCollator(bucket_to_max=...)`,
    wired automatically under `compile_mode="reduce-overhead"` so non-divisible datasets work without
    dropping data.
  - Fixed Finding 6c: wrapped `_build_head_cache` forward in `self._autocast()` for `bf16` parity.
  - Added gradient-clipping tests for `lora_variate`/`full_ft` and `restore_best` test for `full_ft`
    (mutation gaps 2 and 3).
- **Export & LoRA (`5292b43`):**
  - Fixed Finding 4: hashed `dataset._series_index` and `dataset._origins` in `dataset_fingerprint` so
    equal-sized train/val splits sharing `series` hash distinctly.
  - Fixed Findings 5a–c & 6b: rejected mismatched `lora_config` vs `model._lora_config`, pre-serialised
    all JSON/licence artefacts (coercing NumPy scalars) before writing any file (no partial exports),
    detected fallback `LICENSE` pointers on chained exports, and used a local `torch.Generator` in `load_lora`.

- **Follow-up review fixes & CUDA verification (`2d79832`, `576c770`, `163ae4a`, `6880061`):** replaced
  `collect_conformal_data`'s absolute `1e-5` std cutoff with scale-invariant `std > 0 and std > 1e-12 * max(|obs|)`
  (falling back to context magnitude `max(|obs|)` on flat non-zero contexts instead of `1.0`), closed all
  mutation-test gaps (single-window `forecast` parity with numeric and categorical covariates, actuals/scales,
  `symmetric_averaging`, `clip_nonnegative`, forecast-vs-step ACI window, median clamp, upper `1 + gamma`
  bound, wired trainer bucketing & head-cache trimming, `_series_index` fingerprint, fallback `license_path`),
  fixed `test_bf16_training_on_cuda`'s head-cache check to assert `cache.horizon_hidden`, streaming `bf16`
  parity, and difference from an `fp32` cache, fixed `SPEC.md` §5.2's repair guarantee wording, added
  D17/D18 to §11, and restored the `b0980cb` L4 log line in Phase 2 notes. The first L4 attempt at
  `163ae4a` failed `test_shape_bucketing_preserves_exact_loss_gradients_and_works_under_reduce_overhead`
  because `head_bucket_trainer` hardcoded `device="cpu"` with `compile_model=True` on a minimal Linux VM
  without `g++`; fixing it to use `device` (`"cuda"` when available) in `6880061` resolved the failure.

**Evidence.** Ruff check & format clean; `uv build` clean; `check_upstream_drift.py` reports no drift.
Local macOS suite: **526 passed, 4 skipped** (PyTorch 2.15.0, Python 3.13); PyTorch 2.4.0 floor
network-free pass green. NVIDIA L4 CUDA (`bash scripts/cuda_check.sh` at `6880061`, short-lived VM
deleted immediately after): **253 passed, 2 skipped** on device tests and **528 passed, 2 skipped**
overall (exit 0), exercising all 4 `bf16` CUDA modes (including `bf16` head-cache parity vs streaming and
difference vs `fp32`) and shape-bucketed `compile_mode="reduce-overhead"` on CUDA graphs.

**Not done / open.** Phase 2 fast-forwarded into `main` at `38d4ace`, pushed to `origin/main` (CI run
`44936917337` green across Python 3.10–3.13), and `phase-2` deleted locally and on `origin`. Phase 3 has
not started.

---

## 2026-11-22 (ninth session) — Phase 3 built: metrics, sign-flip, Lacuna, sweeps & 5M/28M benchmarks

**What.** Created branch `phase-3` off `main` (`1b3dec1`) and built all Phase 3 deliverables (spec §10):

- **Raw-space evaluation & forgetting (`0dc3f1a`, `utils/metrics.py`):** implemented `pinball_loss_raw`,
  `weighted_quantile_loss`, `mase`, `coverage`, `interval_width`, `MetricsSummary`, `evaluate_predictions`,
  `evaluate_forecaster` (through vendored upstream `TimestralForecaster.forecast()` with its defaults, plus
  optional CQR/ACI calibrators), `ForgettingSummary`, `compute_forgetting`, and `evaluate_forgetting`.
- **Sign-flip augmentation (`8cbb7b2`):** added `TrainingConfig.sign_flip: bool = False`, `flip_batch_signs`
  (coherent per-series negation of targets + covariates), and training head-cache bypass when enabled.
- **Optional Lacuna integration (`f8b9c9c`, `integrations/lacuna.py`):** implemented `LacunaExporter`
  (wrapping `lacuna.ExperimentRegistry.record` and `record_selection` using `AttemptRecord.trial_id`) and
  `write_signal_frame` (long `time`/`instrument`/`signal` frame for `lacuna.SignalStudy`), lazy-imported and
  verified live against `lacuna-quant==0.14.1`.
- **Reproducible sweeps & 5M/28M benchmarks (`604d8d5`, `scripts/benchmarks/`, `docs/BENCHMARKS.md`):**
  - Pinned three permissively licensed Hugging Face datasets: `twelvedata/financial-world-model@da186e5fd362`
    (MIT; 51 US equities, 1h realised volatility `rv_20`, **post-Oct 27, 2026 test split** — zero contamination
    on test targets), `Dingdong-Inc/FreshRetailNet-50K@c374e85975c8` (CC BY 4.0; top-64 active store-SKU
    hourly series, 9 rows/series), and `Salesforce/GiftEval@30841734ac5c` (Apache-2.0; 48 hourly energy,
    weather, and traffic series for the forgetting metric).
  - Ran LR, LoRA rank/targets, and sign-flip sweeps on 5M and 28M plus the full 5M + 28M 7-mode benchmark
    matrix and 200-step 28M `bf16` vs `fp32` check on a short-lived NVIDIA L4 VM (deleted and verified gone
    immediately afterwards).
  - Updated `DEFAULT_LEARNING_RATES` (`head_only=3e-4`, `lora_*=1e-3`, `full_ft=5e-5`) and `LoraConfig`
    default `targets=("q", "k", "v", "o")` (`r=8, alpha=16`) in `config.py` and `SPEC.md` §5.4 from sweep
    evidence.

**Key findings (`docs/BENCHMARKS.md`):**
- Every mode beat zero-shot on both datasets and both sizes (`4.6%–10.7%` wQL reduction).
- `lora_variate` (0.26% of 28M params) matched `full_ft` on post-release equity volatility (`0.2187` vs
  `0.2188` wQL) with negative forgetting (`ΔwQL = -0.0005`); `lora_temporal` beat `full_ft` on both 5M and
  28M retail demand (`0.1957` vs `0.1963` on 28M).
- Online ACI lifted 80% coverage on the post-Oct 2026 volatility shift from `66.3% → 76.5%` (5M) and
  `70.2% → 76.6%` (28M) with zero weight changes or forgetting.
- `sign_flip=True` reduced retail `full_ft` validation wQL by 4.0% (`0.2230 → 0.2141`).
- 200-step 28M `bf16` vs `fp32` full fine-tuning matched within `0.35%` raw test wQL (`0.1963` vs `0.1956`).

- **Large-checkpoint (`123M` & `209M`) CUDA probes & benchmarks (same session, L4 VM deleted immediately after):**
  - Probed peak attention logits and CUDA gradient accuracy vs float64-RMSNorm reference: `123M` (`131.6k–132.4k`
    peak logits, default-SDPA grad error `4.96e-4–6.94e-4`, `1 - cos <= 2.4e-7`, `math` `7.9e-6–9.4e-6`, `head_only`
    `<= 1.9e-6`) and `209M` (`123.4k–131.4k` peak logits, default-SDPA grad error `2.63e-4–4.85e-4`,
    `1 - cos <= 1.2e-7`, `math` `4.0e-6–2.3e-5`, `head_only` `<= 5.5e-7`) are significantly better-conditioned than
    `5M` and `28M`.
  - Direct 120-step real-data retail ablation (`lora_temporal`): `fp32` default, `fp32` math, and `bf16` agreed
    within `0.0002–0.0007` test wQL (`123M`: `0.1963 / 0.1965 / 0.1970`; `209M`: `0.1947 / 0.1952 / 0.1943` with
    `bf16` 1.76× faster in `2.79 GiB` VRAM).
  - Full 160-step real-dataset benchmarks (`docs/BENCHMARKS.md` §3): all modes including `209M full_ft` ran in
    `< 5 GiB` VRAM and `25–45 s` on L4; `lora_temporal` achieved the lowest test wQL across both domains on both
    sizes (`123M`: `0.2111` vol / `0.1950` retail; `209M`: `0.2127` vol / `0.1924` retail — −13.5% vs zero-shot).

- **Statistical significance, multi-seed variance, extended 2026 volatility holdout & classical baselines (`cf0520c`, `scripts/benchmarks/significance.py`, `docs/BENCHMARKS.md` §4, L4 VM deleted immediately after):**
  - Quantified 3-seed variance on `28M` (`wQL` std `0.0002–0.0010` on retail, `0.0002–0.0017` on post-Oct volatility).
  - Paired clustered bootstrap 95% CIs (10,000 resamples) + Newey-West Diebold-Mariano tests:
    - Retail (`64` SKUs × `6` blocks): `lora_temporal` `0.1969 ± 0.0010` vs zero-shot `0.2192` and `seasonal_ridge` `0.2189`; improves `64/64` SKUs (`CI [-0.0263, -0.0206]`) and `6/6` time blocks (`CI [-0.0282, -0.0187]`).
    - Extended 10-month 2026 volatility (`51` equities × `49` market blocks, trained strictly on 2024–2025): `lora_temporal` `0.1819` vs zero-shot `0.1926` and Corsi `har_rv` `0.1981`; improves `49/51` equities (`CI [-0.0128, -0.0087]`) and `44/49` market blocks (`CI [-0.0140, -0.0076]`, `DM p = 4.1e-10`).
    - Strict 3-week post-Oct 27, 2026 slice (`3` blocks): beats `har_rv` (`0.2171 ± 0.0016` vs `0.2253`) and `38/51` equities (`series CI [-0.0208, -0.0006]`), while 3-block time CI straddles zero (`[-0.0293, +0.0061]`, `2/3` blocks) as expected for `n=3`.

- **Independent Phase 3 review remediation (`ae120e2`, `ecbd2d8`, and final sweep/benchmark re-run on L4, VM deleted immediately after):**
  - Fixed online ACI evaluation order in `utils/metrics.py::_apply_online_chronological` (D20): groups
    windows by forecast-origin timestamp in ascending chronological order and flushes `calibrator.update()`
    only once a prior window's full horizon completes (`t_end < t_origin`), eliminating cross-series lookahead.
  - Closed all 9 review mutation-test gaps across `tests/test_metrics.py`, `tests/test_trainer.py`, and
    `tests/test_lacuna.py` (including `sign_flip=True` vs `False` weight divergence, exact forgetting sign,
    10,000x-scale MASE context alignment, distinct 80%/90% coverage/width assertions, chronological
    non-peeking ACI spy test, and all 4 built-in `write_signal_frame` reductions & `horizon_index`).
  - Fixed retail series selection in `scripts/benchmarks/datasets.py` to rank strictly on training-period
    sales (first 70 days, zero val/test leakage) and print exact scored test date ranges (`2026-11-03` to
    `2026-11-20` for volatility, `2024-06-21` to `2024-07-02` for retail).
  - Properly bracketed all four LR defaults strictly on validation splits across `5M`, `28M`, and `123M`
    (`head_only=3e-3`, `lora_variate=3e-3`, `lora_temporal=3e-3`, `full_ft=3e-4`, D19) with macro
    GIFT-Eval forgetting visible alongside validation wQL (zero test-set peeking).
  - Added full CLI reproducibility flags (`--modes`, `--datasets`, `--eval-batch-size`, `--full-ft-batch-size`,
    `--attention-backend`, peak VRAM, per-block ACI coverage, equal-weighted 3-domain macro forgetting) to
    `run_benchmarks.py`, re-ran `5M`, `28M`, `123M`, `209M` benchmarks and `28M` significance checks on L4,
    committed raw JSON outputs in `docs/benchmarks/`, added D19/D20 to `SPEC.md` §11, fixed stale `STATUS.md`
    wording, and corrected claims in `docs/BENCHMARKS.md`.
- **Large-checkpoint 3-seed decision runs & final Phase 3 defaults (`55d57d2`, `1185f82`, `7ebe45d`, and final docs/JSONs, short-lived L4 VM deleted and verified gone immediately after):**
  - Closed the remaining 4 mutation-test gaps (`55d57d2`: `sign_flip` `p=0.5` vs `p=1.0`, `evaluate_forecaster`
    `seasonality` forwarding, exact chronological array order in `_SpyOnline`, and overlapping-stride horizon-reveal
    delay).
  - Added reproducible `load_signed_returns()` (`logret_20`) to `datasets.py` and `--datasets`/`--modes`/`--mode-lrs`/
    `--sign-flip-modes`/`--seeds` flags to `run_sweeps.py` (`1185f82`), then ran the full 144-run 3-seed factorial on
    `123M` and `209M` (`docs/benchmarks/sweeps_large_decisions.json`).
  - Applied Decisions 1–3 (`7ebe45d`, D19):
    1. `TrainingConfig.sign_flip = True` (Option B: within seed spread on signed `logret_20` on both `123M` and `209M`,
       improving 5/6 cells, and cutting retail val wQL 5.9%–6.7%), with dual-orientation `+x`/`-x` frozen-feature caching
       in `head_only` (3 body passes across 4 epochs, `< 1e-4` L2 parity vs streaming).
    2. `lora_temporal` default LR → `1e-3` (`3e-3` never beat `1e-3` by more than seed spread on `123M`/`209M` and hit
       `+19.6%`/`+23.8%` worst-seed forgetting on signed returns).
    3. `full_ft` default LR → `1e-4` (won 3/6 `123M`/`209M` splits outright, `3e-4` never exceeded seed spread on the
       other 3, and `1e-4` halved worst-seed retail forgetting to `+1.6%`/`+2.1%`; `5M`/`28M` preference for `3e-4`
       noted in D19 and `docs/BENCHMARKS.md`).
  - Re-ran unified `sweeps_bracketed.json` (removing split `sweeps_extra.json`), `bench_5m_28m.json`,
    `bench_123m_209m.json`, `precision_ablation_123m_209m.json` (on train-only retail selection), and `sig_v2.json`
    at the final shipped defaults, added §14 to `docs/VERIFICATION.md` for all new `[V]` claims, fixed all four
    report claims, and removed the GCP zone string from earlier devlog prose.
  - Closed the last three mutation-test gaps (`40d29fe`: staggered ACI timestamps, dual-cache 2× memory cap,
    and unflipped post-`fit()` training-set evaluation) and documented the maintainer decision to keep
    `full_ft=1e-4` as the single cross-size default while recommending `learning_rate=3e-4` for `full_ft` on
    `5M` and `28M` (three seeds on `5M`, one seed on `28M`, at the cost of roughly two to three percentage
    points more forgetting) across `README.md`, `TrainingConfig` docstring, `docs/SPEC.md` (§5.4, D19, §12),
    `docs/BENCHMARKS.md` §2.2, `docs/STATUS.md`, and Phase 3 notes.

  - Re-ran full `bash scripts/cuda_check.sh` on an NVIDIA L4 VM at commit `542568c` (post dual-orientation
    `head_only` cache; VM and disk deleted immediately after): **255 passed, 2 skipped** on device tests and
    **538 passed, 3 skipped** overall (exit 0), exercising all 4 `bf16` CUDA modes, dual-orientation caching,
    shape-bucketed `reduce-overhead` compile, and T1–T14.

**Not done / open.** Phase 3 fast-forwarded into `main` at `9c5817c`, pushed to `origin/main` (CI run
`45006472178` green across Python 3.10–3.13), and `phase-3` deleted locally and on `origin`. Post-merge
polish on `main` (`638e9c1`, pushed): re-verified Phase 3 network-free tests on PyTorch 2.4.0 (`283/15`),
PyTorch 2.8.0 (`326/5`), and Python 3.10 / PyTorch 2.14.1 (`326/5`); committed
`docs/benchmarks/bench_5m_full_ft_3e4_vs_1e4.json`; clarified explicit `dropout=0.05` vs
`LoraConfig.dropout=0.0` in `docs/BENCHMARKS.md` §1; and cleaned stale wording.

---

## 2026-11-22 — Phase 4 (rescoped): replay mixing, frozen-base distillation, and Apple GPU scaling

**Done (branch `phase-4`, unpushed/unmerged).** Rescoped Phase 4 per maintainer direction to **Item 1**
(replay mixing, built-in synthetic generators, frozen-base distillation) and **Item 4** (Apple GPU `123M`,
`209M`, and long-context coverage), deferring Items 2 & 3 (fixed-length tiled DHF and sampled block
lengths) as exploratory due to token doubling (undoing D16), per-window categorical encoding conflict
(D1), whole-span known-future normalisation, and separate batch/collator complexity.

- **Item 1 — Replay & distillation (`68c64c5`, `7304681`):**
  - Implemented `ReplayConfig` (`config.py`), `SyntheticReplaySource` (`replay/generators.py`, 5 classical
    families subclassing `TimeSeriesDataset`), `distillation_loss` (`replay/distill.py`), and `disable_lora`
    (`models/adapter.py`).
  - Wired into `TimestralTrainer` (`trainer/trainer.py`) and `export_checkpoint` (`export.py`, recording
    `replay_config` in `tune_metadata.json` while keeping `config.json` strictly upstream-compatible).
  - **Key finding / Decision D21:** running replay as a separate micro-batch allows heterogeneous covariate
    layouts and horizons between domain and replay datasets. On our first cached-vs-live teacher parity
    test, raw L1 (`|student - teacher|`) exhibited a real `+/-1` subgradient kink at `0`: `~3.5e-7` fp32
    GEMM tiling round-off between cached and live teachers at step 0 flipped AdamW step signs (`2*lr`
    divergence). Switching `distillation_loss` to Huber/Smooth-L1 (`smooth_beta=1e-3`, with `0.0`
    preserving exact L1) eliminated the kink while remaining exact L1 for real drift (`>= 1e-3`), bringing
    5-step cached-vs-live trajectories within `< 5e-4` rel L2 and direct teacher outputs within `< 1e-5`.
  - Low-VRAM frozen-base teacher: frozen `model.head` in `head_only` (preserving dual `+x`/`-x` head
    caching), CPU-cached base quantiles up to 256 MiB in `lora_*`/`full_ft`, and zero-VRAM `disable_lora`
    (LoRA) or frozen base copy (`full_ft`) fallbacks.
  - Added 24 unit/parity/mutation tests in `tests/test_replay.py` (all passing).

- **Empirical replay ablation (`e31a2b2`, `docs/benchmarks/replay_ablation.json`, `docs/BENCHMARKS.md` §5):**
  - Ran `scripts/benchmarks/run_replay_ablation.py` on Apple M4 GPU (`fp32`) across the three high-forgetting
    regimes from Phase 3:
    1. `head_only` at `3e-3` (`5M` & `28M`): `replay_fraction=0.25` cuts vol forgetting 2.5×–2.7×
       (`+13.7% → +5.0%` on 5M, `+10.3% → +4.1%` on 28M) and retail forgetting (`+5.0% → +3.8%`,
       `+7.0% → +5.3%`) at `+4–6%` wall time on 28M; `distill_weight=0.5` brings vol forgetting to `+0.7%`.
    2. `lora_temporal` at `3e-3` (`5M` vol, 3 seeds): `replay_fraction=0.25` eliminates the `+31.3%`
       worst-seed forgetting cliff down to `+1.1%` while keeping mean test wQL `0.2181` (vs `0.2186` at `1e-3`).
    3. `full_ft` at `3e-4` (`5M` retail, 3 seeds): pure synthetic pinball replay slightly increases
       forgetting (`+2.8% → +4.1%`) as full FT adapts to the synthetic generator, whereas light frozen-base
       distillation (`distill_weight=0.1`) halves worst-seed forgetting (`+2.8% → +1.3%`) while retaining
       most of `3e-4`'s test wQL gain over `1e-4` (`0.1999` vs `0.2016`).

- **Item 4 — Apple GPU (MPS) `123M`, `209M`, and long-context coverage (`3608c60`, `docs/benchmarks/mps_coverage.json`, `docs/VERIFICATION.md` §15):**
  - Added and ran `scripts/probes/mps_large_and_long_context.py` on a 16 GiB Apple M4 (`fp32`, PyTorch
    2.15.0): `123M` and `209M` match CPU to `9.5e-8..1.1e-6` forward and `4.5e-6..8.0e-6` gradient vs
    float64-RMSNorm reference (cleaner than CPU fp32); all four modes run cleanly at `ctx=256 cov5 B=8`;
    and contexts `4,096..32,768` steps (`128..1,024` patches) pass inference (`6.8e-8..5.3e-7` vs CPU),
    `lora_temporal`, and `full_ft` (`209M full_ft` at `32,768` steps `B=1`: `1.31 s/step`, `6.09 GiB` driver)
    plus `cov3` at `8,192` and `16,384` steps. Added network-marked MPS parity test in
    `tests/test_loader_device.py`.

- **Phase 4 review remediation (`258965d`, `4db0fa1`, `e3c6883`, and final `docs:` commit):**
  - Changed `ReplayConfig()` default from `0.25 / 0.0` (which worsened `full_ft` forgetting) to light
    label-free distillation (`replay_fraction=0.0, distill_weight=0.1, smooth_beta=1e-3`); wired
    `ReplayConfig.smooth_beta` into both `distillation_loss` calls in `TimestralTrainer`; removed the
    untested `bf16` round-off claim from `replay/distill.py`; rejected `replay_dataset is val_dataset`;
    isolated `head_only` cache slots (`"train"`, `"val"`, `"replay"`) so a shared `train_dataset` never
    silently ignores `ReplayConfig.sign_flip`; documented frozen-base semantics on resume; recorded
    `replay_source` (type, sizes, fingerprint, synthetic seed/families) in `tune_metadata.json`; and
    added tests for multi-target covariate replay, `smooth_beta`, val rejection, slot isolation,
    `replay_source` export, and 9 CUDA `fp32`/`bf16`/`reduce-overhead` replay tests (39 tests total in
    `tests/test_replay.py`).
  - Added scientific controls to `run_replay_ablation.py` & `docs/benchmarks/replay_ablation.json`:
    `head_only` `1e-3` lower-LR control (3 seeds on `5M`, 1 seed on `28M`), `full_ft` `2e-4` lower-LR
    control, and `full_ft` 256-series (1,024-window) synthetic pool control. Findings (reported without
    overclaiming in `docs/BENCHMARKS.md` §5 and `docs/SPEC.md` §5.3): plain `full_ft 2e-4` (`0.1995`,
    worst `+1.3%`) matches or slightly beats `3e-4 + distill_0.1` (`0.1999`, worst `+1.3%`); expanding the
    synthetic pool from 64 to 256 series lowers `full_ft` synthetic-pinball forgetting from `+4.1%` back to
    `+2.8%` (showing small-pool memorisation drove the extra jump, while synthetic pinball still fails to
    improve on plain `3e-4`'s `+2.8%`); `lora_temporal 3e-3 + replay_0.25` only recovers shipped `1e-3`
    without beating it; and on `5M head_only` volatility `3e-3 + replay_0.25` beats plain `1e-3` at
    matched wQL (`0.2231 / +5.0%` vs `0.2236 / +7.8%`), while on `28M` vol plain `1e-3` gives lower wQL.
  - Ran `tests/test_replay.py`, a one-off tiny-model CUDA `bf16` `smooth_beta` check (no committed script;
    `0.00e+00` cached-vs-live diff across `{0.0, 1e-3, 1e-2}`), and full `bash scripts/cuda_check.sh` on a
    short-lived NVIDIA L4 Linux x86 VM (`e3c6883`; caught and fixed a Linux x86 fp32 merged-vs-unmerged
    tolerance in `test_disable_lora`): **255 passed, 3 skipped** (device tests) and **577 passed, 4 skipped**
    (whole suite, exit 0; VM and disk deleted immediately after). The earlier project-wide `ssh-keys`
    metadata entry was subsequently removed via `gcloud compute project-info remove-metadata --keys=ssh-keys`.
  - Re-verified the full local macOS PyTorch matrix on `phase-4`: PyTorch `2.4.0` (`313/24`), `2.8.0`
    (`356/14`), Python `3.10` / `2.14.1` (`356/14`), and `2.15.0` full suite (`567/14`); fixed every
    remaining doc contradiction across `SPEC.md`, `BENCHMARKS.md`, `VERIFICATION.md`, `README.md`,
    `STATUS.md`, and `phases/phase-4-stretch.md`.
  - Closed all five remaining non-blocking gaps (`58c1cc6`, `89ac249`, final `docs:` commit):
    1. Added `test_cuda_bf16_replay_released_5m_checkpoint` (released `Timestral-5M` `head_only` &
       `lora_temporal` under CUDA `bf16` + upstream export) and parametrised
       `test_cuda_bf16_replay_with_reduce_overhead_compile` over `cache_base_outputs=[True, False]`
       (exercising live `disable_lora` under CUDA graphs).
    2. Committed `scripts/probes/replay_bf16_parity.py` and its NVIDIA L4 CUDA `bf16` output
       `docs/benchmarks/replay_bf16_parity.json` (`0.00e+00` cached-vs-live diff on `tiny` and released
       `Timestral-5M` across `smooth_beta` `{0.0, 1e-3, 1e-2}` for `lora_temporal` and `full_ft`).
    3. Added `distill_0.1` (`ReplayConfig()` default) cells to `head_only` (`5M` 3 seeds, `28M` seed 0)
       and `lora_temporal` (`5M` vol, 3 seeds) in `docs/benchmarks/replay_ablation.json`: strictly beat
       lower-LR `1e-3` on `head_only` retail (`5M` `0.2105/+2.5%` vs `0.2114/+3.6%`; `28M` `0.2079/+3.6%`
       vs `0.2090/+4.2%`), and beat shipped `1e-3` on `lora_temporal` 5M vol on 3/3 seeds (`0.2178/+0.6%`
       vs `0.2186/+1.6%`, small margin vs seed spread).
    4. Regenerated `docs/benchmarks/mps_coverage.json` with `assert torch.isfinite(loss)` active.
    5. Ran full `scripts/cuda_check.sh` on a fresh OS-Login NVIDIA L4 VM (`89ac249`, no new project-wide
       `ssh-keys` added; VM and disk deleted immediately after): **42/42** in `tests/test_replay.py`,
       **255 passed, 3 skipped** (device tests), and **580 passed, 4 skipped** (whole suite, exit 0).

**Not done / open.** Phase 4 fast-forwarded into `main` at `9261c75`, pushed to `origin/main` (CI run
`45046304547` green across Python 3.10–3.13), and `phase-4` deleted locally and on `origin`. PyPI
publish awaits maintainer approval; tiled DHF remains deferred as exploratory.

---

## 2026-11-23 — Release preparation for v0.1.0

**Done.** The maintainer made the repository public and asked for a first release. It is `0.1.0`, not
`1.0`: the tests are thorough, but nobody outside the project has used the API yet and upstream is a
first release with no stability promise, so the API must stay free to change.

- Version `0.1.0`, classifier Alpha, and "pre-alpha" wording removed from the README, `AGENTS.md`,
  `STATUS.md` and the package docstring.
- README made fit to be the PyPI page: absolute links and banner URL, an install section, and a
  `ReplayConfig` example. `pyproject.toml` gained `[project.urls]`; `CHANGELOG.md` added and shipped in
  the sdist.
- `.github/workflows/release.yml`: a `v*` tag builds once, checks tag, changelog and metadata, publishes
  through PyPI trusted publishing, and creates the GitHub release. Run by hand it builds and checks
  without publishing. Procedure in `docs/DEVELOPMENT.md`, "Releasing".

**Evidence** (`docs/VERIFICATION.md` §10).

- The wheel, installed into a clean environment outside the repository, runs the README quick start as
  written against `Timestral-5M`, and the `ReplayConfig()` example.
- The declared dependency floors pass the network-free suite together on Python 3.10 (313 passed,
  25 skipped). One finding: on torch 2.4.0 with Python 3.10, `torch.compile` needs `setuptools`, which
  that wheel does not declare. Recorded as a known issue (spec §12); no code changed.
- Tracked files and all commit messages were scanned for private details and secrets before release:
  none found.

**Not done / open.** Not yet published: PyPI needs a trusted publisher registered for this repository,
and the tag and upload wait for the maintainer's go-ahead. The spec §10 release box stays unticked until
the upload has happened. Tiled DHF remains deferred as exploratory.

---

## 2026-11-23 — v0.1.0 published

**Done.** With the maintainer's go-ahead, a PyPI trusted publisher was registered for this repository and
tag `v0.1.0` was pushed. The release workflow built the package, uploaded the wheel and sdist to PyPI, and
created the GitHub release from the changelog. This was the first run of its publish and release jobs.

**Evidence.** PyPI serves `timestral-tune` `0.1.0` with both files. Installed from PyPI into a clean
environment, the package runs the README quick start against `Timestral-5M` with the same results as the
local wheel (`docs/VERIFICATION.md` §10). CI and a build-only run of the release workflow were green on
the tagged commit before tagging.

**Not done / open.** `main` still carries version `0.1.0`; set the next version when the next change to
the package lands. Tiled DHF remains deferred as exploratory.

---

## 2026-11-23 — README rewritten for outside readers; repository topics and homepage set

**Done.** The README now leads with what the library is for and why to trust it, and adds sections on
choosing a mode, headline results with their limits, conformal calibration, forecasting from an export,
hardware, and limits. The internal status table and repository layout were removed; they live in
`docs/STATUS.md` and `docs/DEVELOPMENT.md`. Conformal calibration is no longer typeset as a
`TrainingConfig` mode, which it is not. The licence section is unchanged.

**Evidence.** All five Python blocks were extracted from the README and run in order against
`timestral-tune==0.1.0` installed from PyPI. The results table was recomputed from
`docs/benchmarks/sig_v2.json` and the cross-size ranges from `bench_5m_28m.json` and
`bench_123m_209m.json`. All 19 links resolve and `twine check --strict` passes.

**Not done / open.** The PyPI project page keeps the `0.1.0` README until the next release.

---

## 2026-11-23 — README: tagline, results chart, and a memory and speed section

**Done.** At the maintainer's request the README gained an opening tagline, a results section built
around one chart, and a section on memory use and training time.

- `assets/results.png` is drawn by `scripts/make_results_chart.py` from `docs/benchmarks/sig_v2.json`.
  Per dataset it shows three bars: the best classical model, Timestral-28M as released, and
  Timestral-28M after full fine-tuning. Bars start at zero, and the fine-tuned bar carries the number
  of series that improved and the 95% confidence interval of the gain.
- The 28M comparison was chosen because it is the best-evidenced one: three seeds, paired confidence
  intervals and classical baselines all exist for it. The volatility panel uses the ten-month 2026
  test (2,499 windows); the README's fine print says that most of it predates the models' release
  and gives the 18-day post-release result alongside.
- Drafts that were rejected, and why: a before-and-after range chart across all four sizes (too dense
  to read); a chart plotting only the differences (it exaggerated a 10% gap); and a comparison of the
  fine-tuned 5M model with the 209M model as released as the headline (true on retail demand, but a
  tie on volatility, and on retail the 209M is the weakest model as released).
- Memory and speed figures are the peak GPU memory and wall time recorded in `bench_5m_28m.json` and
  `bench_123m_209m.json` (NVIDIA L4, `bf16`).

**Not done / open.** The PyPI project page keeps the `0.1.0` README until the next release.

---

## 2026-11-23 — Document support for checkpoints of other sizes

**Done.** The maintainer asked whether the library is tied to the four released sizes, and then to
document that it is not. Checked before writing it:

- The size names appear in non-vendored code only as pinned Hub revisions; every model dimension comes
  from the checkpoint's `config.json`.
- The four released configs differ only in width, depth, head count and MLP width. They share layer
  kinds (`time`, `time_variate`), the `sdpa_sigmoid` gate and the 21 quantile knots. The claim is scoped
  to that: the vendored config also allows `variate` and `variate_time` layers and an ungated
  attention, which no released model uses and no test here trains.
- New test `test_local_checkpoint_of_an_unreleased_shape_trains_and_exports`: a third shape, saved to
  a directory, loaded by path in the trainer, trained in all four modes, exported, and reloaded by the
  vendored upstream loader with `strict=True`. Until now the trainer's load-by-local-path route was
  only exercised through `load_model`, not end to end.

README gained an "Other model sizes" section; spec §3.1, `docs/VERIFICATION.md` §8 and STATUS updated.

**Not done / open.** Nothing above 209M has been run. Full fine-tuning at that scale would need
features the library lacks (multi-GPU, gradient accumulation, resumable runs).

---

## 2026-11-23 — Practical-value benchmarks (WP1–WP4) on branch `practical-benchmarks`

**Done (branch `practical-benchmarks`, unpushed/unmerged; `README.md` and version `0.1.0` untouched).**
Pre-registered all comparisons, metrics, non-hourly dataset selections (screened on zero-shot vs
classical before any fine-tuning), and quantitative success thresholds in
`docs/phases/practical-benchmarks.md`, then executed WP1–WP4 on a short-lived OS-Login NVIDIA L4 VM
plus local Apple M4 CPU/MPS serving latency:

- **Harness & tests (`scripts/benchmarks/`, `tests/test_benchmark_helpers.py`):** added Winkler
  interval scores, 0.80/0.90/0.95 service-level pinball costs, paired cluster bootstraps across
  decision metrics (`decision_metrics.py`), validation-tuned direct ARX Ridge and 24-config
  21-quantile `HistGradientBoostingRegressor` baselines (`baselines.py`), three strictly
  post-`2026-10-27` `CC BY 4.0` non-hourly datasets (`de_power_15m`, `india_air_quality_daily`,
  `sl_marine_waves_daily`), retail covariate toggle, and series-fraction subsetting (`datasets.py`).
  Unit test caught backward-anchored `TimeSeriesDataset` origin alignment, leading to exact
  `00:00 UTC` / `2026-10-28` boundary alignment across all three new datasets.
- **WP1 & WP4a (`docs/benchmarks/wp1_decision_metrics.json`):** 3-seed decision metrics across all
  four sizes (`5M`–`209M`) on retail and volatility (`123M`/`209M` 3-seed means match §3 within
  `0.0022` wQL, zero README contradictions). On retail, fine-tuning cuts CQR 80% width `−16.0%` to
  `−27.7%` at equal/higher coverage, cuts 90% newsvendor cost `−13.7%` to `−22.9%`, beats CQR alone
  by `−11.3%` to `−18.4%` Winkler_80, beats tuned GBT (`−5.7%` to `−9.1%`), and `5M full_ft` beats
  released `28M`/`123M`/`209M` (`−8.3%`/`−7.0%`/`−9.8%`, all 95% CIs `< 0`). On volatility, raw ZS
  under-covers post-release (`0.663–0.734`); ACI restores `0.782–0.791` for `+12.5%–+29.4%` width;
  CQR width reduction is smaller (`−1%` to `−12.6%`, often raising coverage instead); `28M full_ft`
  beats `gbt_tuned` on the 49-block 2026 holdout (`−6.5%`, `45/51`) but trails `gbt_tuned` (`0.2099`)
  on the 3-block post-release slice (`+3.1%`, CI `[-1.1%, +7.4%]`).
- **WP2 (`docs/benchmarks/wp2_scaling.json`):** `28M` retail plateaus by `200` steps (`0.1951`
  across `200..2,000`); `5M` gains `~1.3 pp` to `500` steps then flattens. `30%` of training series
  (`15–19` series) captures `80%–113%` of full-data gain (`10%` / `5–6` series captures `41%–78%`)
  without inflating forgetting.
- **WP3 (`docs/benchmarks/wp3_unfamiliar_domains.json`):** on all three non-hourly post-release
  domains, `28M full_ft` (36–46 s, forgetting `−0.8%..+1.3%`) cuts wQL `−9.0%` to `−23.0%` and
  `pb90` `−15.0%` to `−26.1%` vs zero-shot. Against tuned GBT (287–912 s): statistical tie on
  `de_power_15m` (`−0.8%`, `[-4.2%, +3.4%]`), small wQL edge + clear CQR Winkler_80 win (`−4.0%`,
  `[-7.1%, -0.6%]`) on `india_air_quality_daily`, and clear wQL win (`−8.4%`, `[-13.4%, -3.4%]`,
  `13/16` sites) on `sl_marine_waves_daily`. `head_only` forgets `+20.8%–+21.6%` on both daily sets.
- **WP4b/4c (`wp4_covariate_ablation.json`, `wp4_inference_latency_{cuda_cpu,apple}.json`):**
  zero-shot ignores retail covariates (`+0.1%`/`+0.4%`) while fine-tuning unlocks `−2.2%`/`−2.6%`
  (`55/64` SKUs) on top of univariate adaptation. Serving `5M` instead of `209M` (`ctx=256, cov=8,
  sym=True`) is `11.9×` faster on L4 `bf16` (`B=64`, `6.0×` less VRAM), `18.9×` on L4 `fp32`,
  `24.4×` on x86 CPU (`B=16`), and `17.5×`/`25.1×` on Apple M4 CPU/MPS.
- **Cloud cleanup verified:** OS-Login L4 VM deleted; `gcloud compute instances list` and `disks
  list` confirm only the pre-existing `e2-medium` VM and its 40 GB disk remain; `commonInstanceMetadata`
  remains completely empty (zero project-wide SSH keys added).

**Not done / open.** Branch `practical-benchmarks` is unpushed and unmerged awaiting maintainer
review; `README.md` and `assets/results.png` were deliberately left unchanged.

---

## 2026-11-23 — Review follow-up on `practical-benchmarks` (all 7 findings addressed)

**Done.** Addressed every methodological and reporting issue flagged during branch review:

1. **Screening audit & India rule (`scripts/benchmarks/screen_wp3_candidates.py`, `docs/benchmarks/wp3_screening.json`):**
   committed a reproducible zero-shot `28M` vs final `run_tuned_classical` screening script reporting both
   validation and test splits; disclosed plainly that initial exploratory screening used test splits and
   that `india_air_quality_daily` does not meet the `>= 5%` classical-gap rule under the final baseline
   (`−4.5%` val / `−0.2%` test; niche-domain branch only).
2. **Upgraded chronologically early-stopped GBT (`scripts/benchmarks/baselines.py`):** expanded the search
   to a two-stage bracketed grid (`4..63` leaves, `min_samples_leaf 20..200`, `l2 0..50`) with exact
   per-quantile chronological validation early stopping via `staged_predict` (`1..800` trees), separating
   single-model CPU `fit_time_s` (`55–492 s`) from full-grid `tuning_time_s` (`502–996 s`) and removing
   all GPU-default-vs-CPU-grid speed ratios. Also caught and fixed a scikit-learn `loss="quantile"` bug
   exposed on `de_power_15m` (~14% night-time solar zeros): fixing Stage 1 to `l2=0.0` (avoiding `l2 > 0`
   split freeze on `tau <= 0.10`) and adding validation-selected `target_transform="median_residual"`
   (fit L1 median `p50`, then fit quantile residuals `y - p50`) eliminated lower-tail collapse, bringing
   `de_power_15m` GBT from broken `0.1143` to `0.0972` with zero boundary hits (`28M full_ft` `+2.3%`,
   CI `[-0.9%, +6.4%]` — not separable, GBT slightly ahead). Across all splits, no dataset hit the
   small-tree corner (`4` leaves); GBT tightened on retail (`0.2134 → 0.2114`), 10m volatility
   (`0.1942 → 0.1935`), India PM2.5 (`0.2138 → 0.2114`, dead tie `−0.5%`), and German power (`0.1009 → 0.0972`,
   not separable `+2.3%`), and stayed flat on post-release volatility (`0.2102`) and Sri Lanka waves (`0.0826`).
3. **3-seed cluster+seed paired bootstraps (`scripts/benchmarks/decision_metrics.py`, `tests/test_benchmark_helpers.py`):**
   upgraded all WP1, WP3, and WP4b paired CIs to resample both training seed `s ∈ {0, 1, 2}` and clusters,
   verified zero consistency violations across all artefacts (fixing e.g. `28M` 10m volatility CQR width
   `−3.1%`, now CI `[-6.8%, +0.5%]`), and re-ran WP1, WP3, and WP4b on a fresh short-lived OS-Login L4 VM.
4. **Tightened framing (`docs/BENCHMARKS.md` §6, `docs/VERIFICATION.md` §16, `docs/phases/practical-benchmarks.md`):**
   explicitly stated that retail equal-coverage width passes cleanly only for `28M` (`123M`/`209M` give up
   `4.0–4.6 pp` coverage, failing the registered condition, though CQR Winkler improves `14%–18%`), counted
   5 distinct datasets rather than double-counting overlapping volatility splits, and stated plainly that
   against tuned GBT fine-tuned `28M` leads on the two pre-release/mostly-pre-release splits and is not
   cleanly separable across the four post-release test slices.
5. **Cloud cleanup verified:** deleted the second OS-Login L4 VM immediately after WP1/WP3/WP4b completed;
   `gcloud compute instances list`, `disks list`, and `commonInstanceMetadata` confirm zero temporary VMs,
   disks, or project-wide SSH keys remain.

**Not done / open.** Branch `practical-benchmarks` remains unpushed and unmerged awaiting maintainer
review; `README.md`, `assets/results.png`, and version `0.1.0` remain untouched.

---

## 2026-11-23 — Single-pass Revision-3 baseline regeneration across all 6 splits on `practical-benchmarks`

**Done.** Regenerated all six GBT baselines (`retail_demand`, `realised_volatility`,
`realised_volatility_2026_10m`, `de_power_15m`, `india_air_quality_daily`, `sl_marine_waves_daily`)
and all WP1 (`5M`–`209M`) and WP3 (`28M`, `5M`) paired comparisons in one pass from the tip
Revision-3 code on a single short-lived OS-Login `g2-standard-32` (`OMP_NUM_THREADS=4` per baseline,
L4 `bf16` training), removing all stale Revision-2 rows:

- **Updated tip-code GBT baselines & `28M full_ft` 3-seed cluster+seed comparisons:**
  - `retail_demand` (2024 pre-release): GBT `0.2114` (`−7.7%`, `[-9.5%, -5.9%]`, time `[-11.3%, -4.7%]` — clear lead).
  - `realised_volatility_2026_10m` (mostly pre-release): GBT `0.1928` (`−6.0%`, `[-7.9%, -4.2%]`, time `[-8.6%, -3.3%]` — clear lead).
  - `realised_volatility` (18d post-release tail, zero boundary hits): GBT `0.2104` (`+2.7%`, `[-1.5%, +7.3%]` — not separable, GBT ahead).
  - `de_power_15m` (post-release, zero boundary hits): GBT `0.0972` (`+2.3%`, `[-0.9%, +6.4%]` — not separable, GBT ahead).
  - `india_air_quality_daily` (post-release, zero boundary hits): GBT `0.2106` (`−0.1%`, `[-2.1%, +2.2%]` — dead tie).
  - `sl_marine_waves_daily` (post-release, zero boundary hits): GBT `0.0798` (`−5.1%`, `[-13.3%, +2.9%]`, time `[-20.5%, +10.5%]` — not separable; loses prior "lead by series" label).
  - `5M full_ft` vs tip GBT: beats GBT on pre-release/mostly-pre-release retail (`−4.7%`, `[-6.4%, -3.0%]`) and 10m volatility (`−3.9%`, `[-5.6%, -2.1%]`), and trails GBT on all four post-release slices (`+4.5%` 18d vol, `+10.4%` German power, `+3.4%` India PM2.5, `+3.3%` Sri Lanka waves). Registered conclusion stated plainly: gradient boosting is the better practical choice for the four clean post-release domains.
- **Disclosures & smaller items addressed:** recorded Revision 3 explicitly alongside Revisions 1–2 and
  up to `~0.4%` cross-machine GBT CPU spread in `docs/phases/practical-benchmarks.md` and `docs/BENCHMARKS.md`,
  fixed `run_tuned_gbt` docstring (`45` Stage-1 configs across `3` scalings, ~`85` total), corrected
  `de_power_15m` training zero-target percentage to measured `10.22%` (`41,496 / 406,080`), documented
  `a69_da` CET/CEST vs `00:00 UTC` delivery-day timing — explicitly noting that for the last hour of each
  winter window (last two hours in summer) every method sees a day-ahead forecast not yet published at the
  `00:00 UTC` origin — documented PyTorch's physical/performance core thread defaults (`2` on `g2-standard-4`
  vs `4` on Apple M4) in `docs/BENCHMARKS.md` §6.5, and documented the intentional `pytest.importorskip("sklearn")`
  skip in plain CI in `docs/VERIFICATION.md` §16.
- **Cloud cleanup verified:** temporary OS-Login L4 VM deleted; `gcloud compute instances list`, `disks list`,
  and `commonInstanceMetadata` confirm zero temporary VMs, disks, or project-wide SSH keys remain.

**Not done / open.** Branch `practical-benchmarks` remains unpushed and unmerged awaiting maintainer
review; `README.md`, `assets/results.png`, and version `0.1.0` remain untouched.


## 2026-11-24 — Squash-merge `practical-benchmarks` into `main`; README states the GBT comparison

**Done.**

- Independent review of the final branch state before merging: recomputed the six-dataset table from
  `docs/benchmarks/wp1_decision_metrics.json` and `wp3_unfamiliar_domains.json`, checked that every
  paired interval contains its own estimate, confirmed the fine-tuned scores did not change across the
  three baseline revisions, and re-ran the Revision-3 GBT search on all six datasets on a second machine
  (Apple M4). Waves (`0.0798`) and German power (`0.0972`) reproduce exactly; the other four differ by
  at most `0.4%`, which is the cross-machine spread recorded in `docs/BENCHMARKS.md` §6.3. No reading
  depends on it.
- Squash-merged the branch into `main` as one commit, so that the superseded copies of the result files
  (about 13–15 MB each revision) stay out of history. The tree of the squash commit is identical to the
  branch tip.
- `README.md`, Results: added the comparison with the tuned gradient-boosted model, scoped to
  `Timestral-28M` (8% and 6% less error on the two charted datasets, level on the four post-release test
  sets) and noting that fine-tuned `5M` trails on those four. The chart is unchanged.

**Decisions.** The README claim is limited to what the registered protocol supports: the two leads are on
test periods that mostly predate the release, so the sentence says so next to the numbers. The other
README proposals from the benchmark report (interval width, serving latency) were not applied.

**Not done / open.** `main` is not pushed. The README on PyPI does not change until a release. The
result files (`docs/benchmarks`, 14.7 MB) are included in the source package through the `docs` entry
in `pyproject.toml`; excluding them is undecided.
