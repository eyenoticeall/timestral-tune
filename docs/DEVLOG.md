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

## 2026-11-23 — Multi-process training design proposal (documentation only)

**Done.** Added a standalone CPU/`gloo` prototype probe (`scripts/probes/distributed_parity.py`) and wrote a
documentation-only design proposal for single-node multi-process / multi-GPU training for a maintainer
go/no-go decision. Nothing in `src/`, `tests/`, `README.md`, or `CHANGELOG.md` changed.

- `scripts/probes/distributed_parity.py`: standalone CPU/`gloo` prototype (`torch.multiprocessing.spawn`)
  comparing `ddp.module` bypass against explicit flat-buffer `all_reduce(ReduceOp.SUM)` and count-weighted
  loss-share scaling across domain-loss edge cases (`equal`, `unequal`, `zero_scored`, `empty_shard` with
  empty-rank-0 delivery at `N=2` and multi-rank sum at `N=4`, `all_zero_skip`, single-rank `non_finite`),
  all four adaptation modes (including `CachedHeadBatch` and LoRA step-0/1 invariants), four replay and
  distillation regimes, 10-active-step (+1 zero-scored skip) AdamW trajectories (`wd=0.01`, `warmup=3` +
  cosine, `clip=1.0`, `full_ft` and `lora_variate`), and `DistributedSampler` behaviour, graded against both
  a union-batch `probe_ref` and an independent one-process `TimestralTrainer` (`shipped_trainer`, on
  `tiny_random`).
- `docs/SPEC.md`: added proposed §5.6 (step-level contract, collective boundaries, explicit flat-buffer
  `ReduceOp.SUM` reduction instead of DDP, five-stage step and collective-safety invariant, count-weighting
  derivation, balanced trailing split with a global-iteration-driven loop and main-process batch sampler,
  per-rank `batch_size` with a warning on default LRs above effective batch 32, backend-driven device
  resolution inside `TimestralTrainer.__init__`, startup failure-status collective before the rank-0
  all-parameter/all-buffer broadcast without a download barrier, rank-0 callbacks / best-loss / restore
  ownership with equal `TrainingResult` fields, separate internal sharded validation, collective-free
  `evaluate()` and `export()`, rank-offset RNG streams, timeouts, and Stage 1 restrictions), proposed
  `tune_metadata.json` fields (`world_size`, `effective_batch_size`) in §7.4, proposed acceptance test T15
  (`test_t15_*`, cases `(a)–(i)`, with `(a)` anchoring its `1e-10`/`1e-4` trajectory tiers at ~10 optimiser
  steps across >1 epoch and `(i)` covering both `__init__` and `fit()` single-rank failures) in §9 with
  released-weights bounds deferred to Stage 3, proposed Phase 5 checklist (three stages plus the
  gradient-accumulation alternative/prerequisite option) in §10, proposed decisions D22–D35 in §11, and
  multi-process risks in §12.
- `docs/VERIFICATION.md`: added the one-off `ReduceOp.AVG` vs `ReduceOp.SUM` CPU/`gloo` check in §11 and
  §16 recording all three runs of `scripts/probes/distributed_parity.py`, keeping enforced checks and
  ungraded `[INFO]` `Timestral-5M` body-backprop values strictly separate and marking every row as a
  prototype outside `src/`.
- `docs/phases/phase-5-multi-gpu.md`: documented single-L4 benchmark wall times and peak VRAM re-derived
  from `docs/BENCHMARKS.md` (`4.4 s`–`53.2 s` / `0.08`–`4.91 GiB` across main `bf16` tables, up to `53.7 s`
  with §3's `fp32` precision-ablation rows; `run_benchmarks.py`'s `ms/step` qualified as upper bounds),
  linear-scaling arithmetic estimates for large jobs, the gradient-accumulation option sharing `(loss_sum,
  count)` helpers, function-level trainer touch points, the Stage 3 2-GPU/4-GPU CUDA/NCCL verification
  plan, and cloud cost estimates (Google Cloud `us-central1` on-demand retrieved `2026-11-23`).
- `docs/STATUS.md`: added a short subsection noting the pending proposal and clarifying that T15 is
  unimplemented (the 14-of-14 acceptance-test count refers to T1–T14), while keeping multi-GPU training in
  the does-not-exist list.

**Decisions.** Proposed D22–D35 (all `[D]`, awaiting sign-off): explicit flat-buffer
`all_reduce(ReduceOp.SUM)` after backward instead of DDP (revisit if NCCL measurements show overlap
matters); five-stage internal distributed step (`ReduceOp.SUM` only) with an 8-scalar `float64` stats
collective, count-weighted loss-share scaling, and best-effort unprototyped single-rank exception
propagation for `__init__` startup and `fit()` local/validation/callback stages (excluding backward
failures, post-broadcast `__init__` steps, and unavailable device contexts, which fall under D35);
collectives confined to `TimestralTrainer.__init__` and `fit()`, keeping `models/loader.py`,
`resolve_device()`, `evaluate()`, `export()`, and public `training_step()` collective-free (`training_step()`
raises `RuntimeError` in a multi-process group; non-zero-rank `export()` validates, logs, no-ops, and
returns `Path`); no rank-0 download barrier (relying on `huggingface_hub`'s per-file `WeakFileLock`, with
a Stage 1 check at the `>=0.23` floor), with `__init__` running rank-local setup/validation before a
startup failure-status collective and broadcasting all parameters and buffers from rank 0 only if all ranks
succeed; backend-driven device resolution in `__init__` (`nccl`→CUDA, `gloo`→CPU, `gloo`+CUDA and MPS
refused, explicit indexed > `LOCAL_RANK` > `rank`, `torch.cuda.set_device` before
`torch.cuda.is_bf16_supported()`); balanced trailing split and global-iteration-driven loop with a
main-process batch sampler instead of `DistributedSampler`; per-rank `batch_size` with no automatic LR
scaling; rank-0 ownership of callbacks, best-loss tracking, early stopping, and restore decisions with
end-of-`fit()` record and conditional trainable-weight broadcast; Stage 1 restrictions following D13
(`head_only` streaming fallback, `reduce-overhead` `ValueError`, `ReplayConfig` `NotImplementedError`);
rank-offset augmentation/dropout streams; and rejection of `nn.DataParallel`, `accelerate`,
`lightning-fabric`, multi-node training, and benchmark-script edits.

**Design-shaping facts checked against existing code and prototype runs:**

- Adding the independent one-process `TimestralTrainer` reference (`shipped_trainer`) exposed and fixed a
  `float64 → float32` state-dict truncation in the prototype's `_load_model` helper (casting to `dtype` now
  precedes `load_state_dict`) and confirmed that `TimeSeriesDataset` stores `np.float32` inputs (a
  `float64` model works with the shipped head cache in `fit()`, but `float64` input values cannot be fed
  through `TimeSeriesDataset`, so `float64` `head_only_cached` is reported `n/a`).
- Inspecting existing trainer and data code established that `DHFDataCollator` raises `ValueError` on an
  empty list (`data/collator.py:155-156`), motivating the global-iteration-driven loop and non-empty batch
  sampler; `torch.cuda.is_bf16_supported()` queries the current CUDA device (`trainer/trainer.py:292`),
  requiring `torch.cuda.set_device` on every CUDA device before precision checks; `early_stopping_patience`
  and callback validation (`trainer/trainer.py:375-383`) can move ahead of `__init__`'s startup status
  collective while `_setup_replay_teacher()` and `torch.compile` must stay after the startup broadcast; and
  one-process `fit()` skips zero-scored batches before advancing `replay_iter`
  (`trainer/trainer.py:1013-1025`), leaving Stage 2 skip ordering open (current leaning: a replay-only
  pre-forward count collective, leaving Stage 1 unaffected).

**Evidence.** Ran `scripts/probes/distributed_parity.py` on macOS arm64 CPU/`gloo`:

- PyTorch `2.15.0` (Python `3.12.13`), `tiny_random`: `326/326` passed, `0` failed (`probe_ref` and
  `shipped_trainer`, `0.00e+00` cross-rank).
- PyTorch `2.4.0` (Python `3.12`), `tiny_random`: `326/326` passed, `0` failed (`probe_ref` and
  `shipped_trainer`, `0.00e+00` cross-rank).
- PyTorch `2.15.0`, `--sizes 5M`: `516/516` passed (`326` `tiny_random` + `190` enforced `Timestral-5M`:
  cross-rank `0.00e+00`, scalar global loss `<= 6.77e-16` f64 / `<= 4.75e-07` f32, `N=2 empty_shard`
  `empty_r0_recv_ok=True`, global skip, non-finite, `head_only` live/cached `<= 1.12e-15` f64 /
  `<= 4.00e-07` f32, LoRA step-0/1 invariants, and `[5]` clipping and skipped-step no-op invariants), `0`
  failed, plus `68` ungraded `[INFO]` `Timestral-5M` body-backprop values (single-step up to `2.42e-12` f64
  / `1.58e-03` f32; 10-step `full_ft` `param_update` up to `7.40e-10` f64 / `4.80e-02` f32; `lora_variate`
  up to `1.22e-04` `final_params` / `1.06e-02` `param_update` in f32).
- One-off CPU/`gloo` check (`docs/VERIFICATION.md` §11): PyTorch `2.4.0` raises
  `RuntimeError: Cannot use ReduceOp.AVG with Gloo` while `ReduceOp.SUM` succeeds; PyTorch `2.15.0` accepts
  both.
- Verified PyTorch `2.15.0` / `2.4.0` source citations (`DistributedDataParallel` hooks, `nn.DataParallel`
  / `scatter_gather.scatter`, `init_process_group` timeout defaults of `10 min` NCCL / `30 min` other
  backends), `huggingface_hub 2.4.0` `WeakFileLock` citations (`file_download.py:1258, 1273, 1485, 1503`),
  and PyPI dependency metadata (`2026-11-23`) for `accelerate` (`1.16.0`: `python >= 3.11`, `torch >= 2.6.0`)
  and `lightning-fabric` (`2.6.6`: 4 extra dependencies).

**Not done / open.** Maintainer go/no-go and sequencing against gradient accumulation; Stage 2 open
decisions (replay skip ordering, teacher output caching across ranks, and whether a multi-process
`head_only` cache is worth having); and unverified items (`docs/SPEC.md` §12): CUDA/NCCL, `bf16`,
`torch.compile`, `torchrun` on Linux, released-weights T15 bounds, `huggingface_hub==0.23` lock behaviour,
large-batch LR sweeps, unprototyped single-rank exception propagation, and NCCL rank-failure behaviour.

---

## 2026-11-23 — Multi-process design approved; pre-implementation spec revision

**Done.** Recorded maintainer approval of Phase 5 (implementation starting; nothing implemented yet in
`src/` or `tests/`, gradient accumulation unscheduled) and applied the pre-implementation design revision
across `docs/SPEC.md`, `docs/VERIFICATION.md`, `docs/STATUS.md`, `docs/phases/phase-5-multi-gpu.md`, and
one message string in `scripts/probes/distributed_parity.py`:

- **Devices, backend, and startup (`§5.6`, D28, D29):** exact `get_backend()` check (`"gloo"` or `"nccl"`
  only); device resolution/binding before any collective and outside failure propagation; snapshotting
  `rank`, `world_size`, and `backend` (`fit()` checks liveness and `world_size`; `export()` uses snapshot);
  five-step `__init__` sequence `(a)–(e)` adding a fixed-size integer consistency digest over model/config
  (excluding `device` and `output_dir`)/datasets (`export.py:dataset_fingerprint`, lines `91–122`) and a
  single trainable dtype check before the rank-0 broadcast.
- **Failure propagation and rank-0 broadcasts (`§5.6`, D23, D30, D35):** replaced the float64 bitmask with
  a `0/1` flag plus conditional `int64` length-`N` rank vector (used directly at startup); restricted
  catching to `Exception` only excluding OOM in forward and backward; introduced `RankFailureError`
  (`RuntimeError` subclass exported from `timestral_tune.trainer`); protected lazy iterator creation and
  rank-0 post-validation, moved `on_fit_start` inside `try/finally` (`trainer/trainer.py:1004-1007`), and
  switched all control broadcasts (including `on_fit_end`) to fixed-size `float64` `[code, value]` tensors
  propagating rank-0 losses plus `[has_best, best_eval_loss, best_step, restored]` when `val_dataset`
  exists, making all ten `TrainingResult` fields identical by construction (removing the NCCL history
  caveat).
- **Sampling, validation, export, Stage 2, and T15 (`§5.6`, `§7.4`, `§9`, `§10`, D22–D33):** specified
  contiguous slices, local zero-scored skip before flip/validate/transfer/forward
  (`trainer/trainer.py:1013-1020`), `(base + r * 1_000_000_007) mod 2^63` rank seeds (checked collision-free
  against `trainer/trainer.py:344, 781, 967-972`), reused flat gradient buffer, sharded validation under
  `eval()`/`no_grad()` restoring `training` (`models/adapter.py:112, 194`), shared `export()` validation
  helper (`export.py:131-134, 267-312`) with conditional top-level `world_size`/`effective_batch_size`
  keys, Stage 2 decisions (per-rank cycling replay loaders, replay-only pre-forward domain count
  collective, no multi-process head/teacher caches), and tightened T15 cases `(a)–(i)` plus harness notes
  (confirming `TuneTimestralModel` has only `transformer.rope.inv_freq` and citing
  `tests/test_dataset.py:451` for worker skips). Trimmed repeated numbers in `docs/SPEC.md` and cut
  `docs/phases/phase-5-multi-gpu.md` §2 to a module/function map.
- **Wording fix:** corrected `scripts/probes/distributed_parity.py:655`, `docs/VERIFICATION.md` §16, and
  this devlog so `float64` `head_only_cached` `n/a` is attributed to `TimeSeriesDataset` storing
  `np.float32` inputs rather than claiming the head cache supports `float32` only.

**Evidence.** Re-ran `uv run python scripts/probes/distributed_parity.py` after the message-string edit:
`total: 326/326 passed, 0 failed`. Lint and `pytest -m "not network"` green.

**Not done / open.** Phase 5 Stages 1–3 implementation and CUDA/NCCL hardware verification (`docs/SPEC.md`
§10, §12).

---

## 2026-11-23 — Implement Phase 5 Stage 1 multi-process training

**Done.** Implemented Phase 5 Stage 1 single-node multi-process training (domain loss across all four
adaptation modes: `head_only`, `lora_variate`, `lora_temporal`, `full_ft`), verified it network-free on
CPU/`gloo` for `N = 2` and `N = 4` across PyTorch `2.4.0` and `2.15.0`, and updated documentation to match
the as-built Stage 1 code.

- **Modules & architecture:**
  - `src/timestral_tune/trainer/_types.py`: shared dataclasses (`EvalResult`, `TrainingResult`, `_StepOutcome`,
    `_StepLoss`, `_FitBookkeeper`, etc.) and `_FitStrategy` protocol, keeping `trainer.py` and
    `_distributed_strategy.py` acyclic.
  - `src/timestral_tune/trainer/distributed.py`: sole importer of `torch.distributed`, providing
    `DistributedContext` (`detect`, `check_active`, `all_reduce_sum`, `broadcast`, `propagate_failure`,
    `raise_propagated_failure`, `check_consistency`), `RankFailureError`, `is_oom_error`, `ControlStatus`,
    `broadcast_status`, `broadcast_module_state`, `GradientReducer`, `DistributedBatchSampler`,
    `steps_per_epoch`, `rank_slice_size`, `rank_step_indices`, `validation_shard_indices`, and `rank_seed`.
  - `src/timestral_tune/trainer/_distributed_strategy.py`: `_DistributedStrategy` implementing per-`fit()`
    `GradientReducer` lifecycle, global-iteration stepping with lazy rank-seeded train/val loaders, five-stage
    count-weighted `_run_step`, internal sharded validation (`evaluate_val`), rank-0 callbacks/restore with
    fixed-size `float64` `[code, value]` broadcasts, and `finalize_best()`.
  - `src/timestral_tune/trainer/trainer.py` & `trainer/__init__.py`: extracted `_SingleProcessStrategy` and
    unified `TimestralTrainer._run_fit_loop` across single- and multi-process strategies; implemented
    five-step `__init__` startup `(a)–(e)`, rank-0-only startup logs, `head_only` cache bypass, public
    `training_step()` `RuntimeError`, snapshot-based `export()`, and exported `RankFailureError` from
    `timestral_tune.trainer`.
  - `src/timestral_tune/export.py`: extracted `_validate_export_inputs` (shared with non-zero-rank
    `TimestralTrainer.export()` without dataset hashing) and added optional paired `world_size` and
    `effective_batch_size` keyword arguments on `export_checkpoint` (writing top-level keys to
    `tune_metadata.json` only when `world_size > 1`).
- **Tests:** added `tests/distributed_harness.py`, acceptance test T15 in `tests/acceptance/test_distributed.py`
  (cases `(a)–(i)` plus regressions and 2-rank CPU `compile_mode="default"` smoke test),
  `tests/test_distributed.py`, `tests/test_trainer_distributed.py`, and targeted unit tests in
  `tests/test_trainer.py` and `tests/test_export.py`; updated the `acceptance` pytest marker in
  `pyproject.toml` to T1–T15.
- **Documentation:** updated `docs/SPEC.md` (§5.6, §6, §7.4, §7.6, §9, §10, D22–D35, §12),
  `docs/VERIFICATION.md` (§12 source reading, §16 clarification, §17 Stage 1 measurements), `docs/STATUS.md`,
  `docs/phases/phase-5-multi-gpu.md`, `docs/DEVELOPMENT.md`, `README.md`, and `CHANGELOG.md`.

**Design adjustments made during implementation:**

- **Consistency mismatch exceptions (D29, `09de038`):** `check_consistency` raises `ValueError` on mismatched
  rank(s) and `RankFailureError(failing_ranks, stage="startup_consistency")` on matching peers (including rank
  0 when another rank mismatches), matching every other failure stage rather than converting to `ValueError`
  on every rank.
- **`GradientReducer` lifecycle & presence flags (D22, `ac9ab04`, `1ae2dc3`):** allocated per `fit()` in
  `prepare_fit()` and released in `cleanup()` rather than in `__init__` (trainable non-emptiness, uniform
  device, and single dtype remain checked at construction). Packs one `0.0`/`1.0` presence flag per trainable
  parameter into the tail of the single `all_reduce(ReduceOp.SUM)` buffer so all-`None` gradients stay `None`
  (preserving one-process AdamW weight-decay semantics) while skipping post-collective `.tolist()` host
  read-back when all local gradients are present.
- **Process-group checks & `gloo` devices (D28, `1ae2dc3`):** `DistributedContext.check_active()` verifies
  liveness, `world_size`, `rank`, and `backend` before every collective; `gloo` accepts `None`, `"auto"`,
  `"cpu"`, or `torch.device("cpu")` (no index).
- **Loader RNG isolation & epoch permutation (D26, `e3a4f6e`):** epoch shuffling hashes
  `"timestral-tune:epoch:{seed}:{epoch}"` via SHA-256 (avoiding `seed + epoch` aliasing), and multi-process
  train and sharded validation loaders receive explicit rank-seeded `torch.Generator`s so iterator creation
  never consumes PyTorch's global RNG stream (LoRA dropout).
- **Step overflow, failure precedence, and OOM scope (D23):** `failed` takes precedence over `non_finite` when
  both occur in the same step; `loss_val * w_d` and `sum_d_tot / w_d_tot` are checked with `math.isfinite`;
  `is_oom_error` matches `torch.OutOfMemoryError` and `torch.cuda.OutOfMemoryError` by type only.
- **Export validation & metadata (D31, `ca07a6b`, `1c83a8d`):** non-zero-rank `export()` skips
  `dataset_fingerprint()` (validating caller-supplied `data_fingerprint` only); `export_checkpoint()` requires
  `world_size` and `effective_batch_size` to be passed together as positive `numbers.Integral` values
  (excluding `bool`) with `effective_batch_size >= world_size`, writing the keys only when `world_size > 1`.
- **Digest contents & rank-0 logs (D27, D29):** `_startup_consistency_digest()` includes
  `model.config.to_dict()` alongside parameter/buffer schemas, config, and dataset fingerprints; startup
  effective-batch and cache-bypass notices emit on rank 0 only.

**Hang paths found in review and fixed (code facts):**

1. **`should_stop` clearing (`290cd6c`):** `_DistributedStrategy._broadcast_rank0_hook` previously set
   `trainer.should_stop = True` on `ControlStatus.STOP` without clearing it on `ControlStatus.CONTINUE`; a
   callback setting `should_stop` in `on_step_end` and clearing it in `on_evaluate` left non-zero ranks
   exiting the loop while rank 0 kept training, deadlocking the next collective. Fixed by unconditionally
   assigning `self._trainer.should_stop = status is ControlStatus.STOP`; covered by
   `test_t15_e_should_stop_cleared_after_set`.
2. **Unprotected startup consistency digest (`290cd6c`):** `_startup_consistency_digest()` previously ran
   after step (b)'s startup failure collective, so a single-rank serialisation error inside the digest left
   peers blocked in step (c)'s broadcast. Moved inside step (a)'s protected `try/except` without adding a
   collective; covered by `test_t15_i_startup_digest_failure_propagation`.
3. **Cleanup before end-of-fit collectives (`d181c7a`):** calling `strategy.cleanup()` before
   `finalize_best()` and `on_fit_end()` triggered PyTorch's macOS `DataLoader` worker shutdown (~5 s per
   active worker under `MP_STATUS_CHECK_INTERVAL`) asymmetrically across uneven trailing shards, timing out
   `on_fit_end()`'s broadcast. Fixed by running `model.eval()` in the inner `finally` before end-of-fit
   collectives and `strategy.cleanup()` in an outer `finally` after `on_fit_end()`.

**Evidence (Stage 1 completion, macOS arm64, Python `3.12.13`, CPU/`gloo`):**

- `TMPDIR=/tmp uv run pytest tests/acceptance/test_distributed.py -v` on PyTorch `2.15.0`: **`25 passed`** in
  `80.22 s` (`docs/VERIFICATION.md` §17).
- `TMPDIR=/tmp uv run --isolated --python 3.12 --with "torch==2.4.0" pytest tests/acceptance/test_distributed.py tests/test_distributed.py -v`:
  **`110 passed, 2 skipped`** in `39.52 s` (`24 passed, 1 skipped` in T15; `86 passed, 1 skipped` in
  `tests/test_distributed.py`; both skips are `num_workers > 0` on macOS PyTorch `< 2.8`,
  `docs/VERIFICATION.md` §10 & §17).
- T15 was hardened by a mutation campaign of deliberate breakages.
- One-off pre-refactor 1-process bit-identity check (run via an out-of-repo script, not a repo test or probe):
  confirmed bitwise identity across 12 scenarios (all four modes, replay and distillation, cached and streamed
  `head_only`, sign flips, early stopping, and restore) against the pre-refactor one-process trainer.
- Source reading of `huggingface_hub==0.23.0` (`docs/VERIFICATION.md` §12): `_snapshot_download.py:267-268`
  delegates to `hf_hub_download` and `file_download.py:797, 1366, 1478, 1485` acquires `WeakFileLock` (source
  reading only, not a concurrent-download test).

**Not done / open:**

- Stage 2 (multi-process `ReplayConfig` / distillation) and Stage 3 (CUDA/NCCL hardware verification: real
  `torch.cuda.set_device` / `LOCAL_RANK` binding, `bf16`, CUDA `compile_mode="default"` + trailing-shape
  recompilation, `torchrun` on Linux + worker shutdown latency, per-step pack/unpack & collective cost and
  validation-time flat-buffer VRAM measurements, rank-failure timeout check, and released-weights T15 bounds).
- Live concurrent multi-process Hub download check at `huggingface_hub==0.23.0`.
- Parallel fix in progress for `tests/test_trainer_distributed.py::test_distributed_trainer_whitebox_internals`,
  which then called the removed `strategy.after_train_loop()` method and read the removed
  `trainer._fit_flip_generator` attribute (remaining `-m "not network"` suite at Stage 1 completion on PyTorch `2.15.0`:
  `483 passed, 15 skipped`).

---

## 2026-11-23 — Implement Phase 5 Stage 2 multi-process replay and distillation

**Done.** Implemented Phase 5 Stage 2 single-node multi-process `ReplayConfig` replay mixing and uncached
frozen-base distillation across all four adaptation modes (`head_only`, `lora_variate`, `lora_temporal`,
`full_ft`), extended acceptance test T15 to cases `(a)–(l)`, added the `F.linear` `requires_grad` dispatch
probe, verified the full network-free and multi-process suites on CPU/`gloo` on macOS arm64 across PyTorch
`2.4.0` and `2.15.0` at `3347f70`, and reconciled all tracked documentation with the as-built code.

- **Modules & implementation:**
  - `src/timestral_tune/trainer/distributed.py`: added `DistributedBatchCycle` wrapping a replay
    `DistributedBatchSampler` and `DataLoader` to cycle global replay epochs disjointly across ranks,
    wrapping independently of domain epochs and handling `len(replay_dataset) < N * b` and `< N` without
    invoking `DHFDataCollator` empty.
  - `src/timestral_tune/trainer/_distributed_strategy.py`: structured `_DistributedStrategy._run_step` into
    six named stages (`_precheck_domain_for_replay`, `_run_local_stage`, `_reduce_step_stats`,
    `_decide_step`, `_backward_local_terms`, `_reduce_gradients_and_update`); wired the replay-only
    2-element `float64` `[w_d, failed]` pre-check collective (failure taking precedence over skip before
    advancing replay), shared replay epoch seed (`config.seed + 2_000_009`) with rank-seeded replay loader
    generator (`rank_seed(config.seed + 2_000_009, rank)`) and main-process execution (`num_workers=0`,
    `persistent_workers=False`), rank-offset replay sign flips (`rank_seed(config.seed + 3_000_007, rank)`),
    count-weighted `[W_r, sum_r, M, sum_m]` terms in the 8-scalar stats collective and backward (including
    empty/zero-scored domain ranks contributing replay), and cleanup of replay cycle/loader/generator state.
  - `src/timestral_tune/trainer/trainer.py` & `src/timestral_tune/config.py`: removed Stage 1's
    `ReplayConfig` `NotImplementedError`, validated `replay_config`/`replay_dataset` (constructing default
    `SyntheticReplaySource` with shared seed `config.seed + 2_000_003`) and included them in
    `_startup_consistency_digest()`, moved `_setup_replay_teacher()` into step (e) after rank-0 weight
    broadcast (building no teacher caches in multi-process mode; using `_frozen_head`, live `disable_lora`,
    or live `_frozen_base`, with a one-time rank-0 bypass log), logged `per_rank_replay_batch_size` and
    `effective_replay_batch_size` on rank 0 at startup, skipped `dataset_fingerprint(self.replay_dataset)`
    on non-zero ranks in `export()`, and updated `TrainingConfig.num_workers` and `ReplayConfig.batch_size`
    docstrings.
- **Tests & probes:**
  - `tests/acceptance/test_distributed.py`: extended T15 to cases `(a)–(l)` (19 test functions, 34 collected
    items), adding `test_t15_j_replay_and_distillation_trajectory_parity`,
    `test_t15_j_full_ft_distill_float32` (replacing an earlier xfail with moved-student `1e-4` parity and
    identical-start cross-rank bitwise checks), `test_t15_k_replay_skip_unscored_and_repeated_fit`,
    `test_t15_l_replay_flips_teachers_startup_failures_and_export`, and 2-rank CPU compiled replay smoke
    `test_t15_l_compile_default_replay_distill_smoke`.
  - `tests/test_trainer_distributed_replay.py`, `tests/test_trainer_distributed.py`, and
    `tests/test_distributed.py`: added dedicated Stage 2 unit/integration replay tests (including
    `lora_temporal`, non-patch-multiple replay horizon, periodic validation + `restore_best=True`, and
    cleanup), `DistributedBatchCycle` tests, and updated white-box trainer tests.
  - `scripts/probes/linear_requires_grad_dispatch.py`: added standalone probe measuring PyTorch's
    `F.linear` `weight.requires_grad` dispatch difference on non-contiguous `float32` inputs and its step-0
    effect on `full_ft` distillation (`docs/VERIFICATION.md` §17.2).
- **Documentation:** updated `docs/SPEC.md` (§5.3, §5.6, §6, §7.4, §7.6, §9 T15, §10 Phase 5, §11 D22–D37,
  §12 including the `F.linear` finding and ten verified pre-existing one-process observations),
  `docs/VERIFICATION.md` (§16–§17), `docs/STATUS.md`, `docs/phases/phase-5-multi-gpu.md`,
  `docs/DEVELOPMENT.md`, `README.md`, `CHANGELOG.md`, and `AGENTS.md` §1.

**Decisions (spec §11):**

- **2-element `[w_d, failed]` replay pre-check collective (D23):** expanded Stage 1's planned 1-element
  domain-count collective to `[float(pre_w_d), failed]` so single-rank domain-fetch failures propagate via
  `raise_propagated_failure(..., stage="training_step")` with failure taking precedence over skipping before
  any rank advances replay.
- **Main-process replay loader (D36):** the multi-process replay `DataLoader` runs with `num_workers=0` and
  `persistent_workers=False` regardless of `config.num_workers`, matching one-process `_replay_loader` and
  capping persistent worker pools at two per rank during `fit()`.
- **Shared replay epoch seed with `DistributedBatchCycle` (D37):** replaced Stage 1's planned rank-offset
  replay shuffling with a shared replay epoch seed (`config.seed + 2_000_009`) partitioned by
  `DistributedBatchSampler` and cycled by `DistributedBatchCycle`, giving disjoint rank slices, exact
  once-per-replay-epoch union coverage, and clean short-dataset handling; only the replay loader generator
  and replay sign-flip stream (`+3_000_007`) are rank-offset (D33).

**Development history (not `[V]` claims):**

- Acceptance tests T15(j)–(l) were written by a different engineer from the Stage 2 implementation.
- The distributed tests were hardened with mutation campaigns: engineers reported 56 seeded implementation
  mutations killed across three rounds. Because those mutation runs are not reproducible from the repository,
  they are noted here as reported development history only and excluded from `docs/VERIFICATION.md`.
- Investigating the `float32` `full_ft` identical-start distillation gap initially produced a wrong
  explanation ("grad vs no-grad" mode difference); the standalone probe
  `scripts/probes/linear_requires_grad_dispatch.py` disproved that explanation (`0.00e+00` between
  `enable_grad` and `no_grad` for the same flag) and established the measured `weight.requires_grad`
  dispatch cause on non-contiguous `float32` inputs (`docs/VERIFICATION.md` §17.2, spec §12).

**Evidence (`3347f70`, macOS arm64, Python `3.12.13`, CPU/`gloo`; full commands and tables in
`docs/VERIFICATION.md` §17):**

- Full network-free suite (`-m "not network"`, PyTorch `2.15.0`) and acceptance test T15 (`tests/acceptance/test_distributed.py`, cases `(a)–(l)`) pass on CPU/`gloo` with zero failures; the four distributed test files pass on PyTorch `2.4.0` with only the four expected macOS `< 2.8` `num_workers > 0` skips.
- `scripts/probes/linear_requires_grad_dispatch.py` ran on both PyTorch `2.15.0` and `2.4.0` on macOS arm64
  CPU (`docs/VERIFICATION.md` §17.2), confirming identical standalone `F.linear` diffs and documenting the
  tiny-model differences between `2.4.0` and `2.15.0`.
- `uv run ruff check .` and `uv run ruff format --check .` clean.

**Not done / open:**

- **Stage 3 (CUDA/NCCL hardware verification):** multi-rank NCCL has never executed; all multi-process runs
  remain CPU/`gloo` on macOS arm64 only (`docs/phases/phase-5-multi-gpu.md` §3, spec §10 & §12).
- Live concurrent multi-process Hub download check at `huggingface_hub==0.23.0` (source reading only,
  `docs/VERIFICATION.md` §12).
- Maintainer decisions on the two candidate remedies for PyTorch's `F.linear` `requires_grad` dispatch in
  `full_ft` distillation and on the ten pre-existing one-process observations recorded in spec §12.
- Optional gradient accumulation (unticked, unscheduled).

---

## 2026-11-23 — Linux x86_64 validation, float32 tolerance recalibration, and single-L4 CUDA/NCCL checks

**What was found on Linux x86_64 (at `3347f70`).** Running the Stages 1–2 suite on a rented 8-vCPU Linux
x86_64 cloud VM (Ubuntu 22.04) surfaced three platform differences not visible on macOS arm64:

1. **Test-order-dependent float32 `param_update` failures caused by a thread leak plus Linux thread-count
   sensitivity.** The full network-free suite (`-m "not network"`, where `tests/acceptance/test_distributed.py`
   runs before `tests/test_trainer_distributed.py`) passed on both managed Python `3.12.16` (`torch 2.15.0+cu132`)
   and managed Python `3.10.22` (`torch 2.14.1+cu130`). Running the four multi-process files alone (with the
   acceptance file last) failed two float32 `param_update` checks against the old `1e-4` bound:
   `T15(j)[2]` `lora_variate_distill_only_wrap_float32` (`1.07e-04`) and `T15(l)[4]` `default_synth`
   (`1.71e-04`). Root causes:
   - `tests/test_trainer_distributed.py::test_harness_preserves_scenario_error_when_destroy_process_group_raises`
     called `distributed_harness._worker_main` in-process, which set `torch.set_num_threads(1)` without
     restoring it, leaking 1 intra-op thread into every later test in the parent process.
   - Unlike macOS arm64 (where one-process `fit()` is bit-identical across intra-op thread counts), Linux
     x86_64 one-process `fit()` varies with thread count: a one-off pure one-process check on T15(j) scenarios
     showed `param_update` rel L2 up to `1.14e-04` between 1 and 2 threads (`N=2 lora_variate distill_only`),
     identically on `torch 2.14.1+cu130` and `torch 2.15.0+cu132`, and pre-fix 10×-tightened margin scans
     showed `T15(l)[4]` `default_synth` reaching `1.94e-04` at 2 threads even without the leak (because
     `default_synth` used `lora_variate`, whose Q/K slices are inert on univariate `SyntheticReplaySource`
     windows per `validate_lora_for_rows` and were dominated by near-$\epsilon$ AdamW amplification;
     `docs/VERIFICATION.md` §11).
2. **`F.linear` `requires_grad` dispatch on Linux x86_64 is non-zero in `float64` too and varies across CPUs.**
   Re-running `scripts/probes/linear_requires_grad_dispatch.py` across three Linux x86_64 VMs showed
   non-zero standalone permuted differences in `float64` as well as `float32`, non-zero tiny-model differences
   in 7 of 8 cells (including `cov bs=4,8`, which are zero on macOS arm64), identical results across
   `torch 2.14.1+cu130` and `2.15.0+cu132` on the AMD EPYC 7B12 VM, and different shapes/magnitudes across
   x86_64 CPUs even on the same PyTorch wheel (`docs/VERIFICATION.md` §17.2).
3. **Ubuntu's system Python `3.10.12` lacks `Python.h`.** Running with `/usr/bin/python3.10` failed all five
   `torch.compile` tests on missing `Python.h`; switching to `uv`-managed Python `3.10.22` (which ships its
   headers) passed cleanly (`docs/VERIFICATION.md` §17.1).

**Fixes (`e33cbf5`, `0f80f4c`, `71514a6`, `40a580a`).**

- `e33cbf5`: `_worker_main` in `tests/distributed_harness.py` restores the caller's thread count in `finally`;
  added autouse `_guard_torch_global_state` in `tests/conftest.py` to fail any test leaking `num_threads`,
  `grad_enabled`, or `default_dtype`.
- `0f80f4c`: centralised all five parity tiers (`EVAL_TOL`, `STEP_TOL`, `STEP_PARAM_TOL`, `TRAJ_TOL`,
  `TRAJ_UPDATE_TOL`) and the validated `TIMESTRAL_TEST_F32_TOL_SCALE` headroom knob (default `1.0`, scaling
  float32 tiers only) in `tests/distributed_harness.py` with unit coverage in `tests/test_distributed.py`;
  set `TRAJ_UPDATE_TOL["float32"] = 1e-3` (keeping `TRAJ_TOL["float32"] = 1e-4` for losses and
  `final_trainable`); split `STEP_PARAM_TOL` (`1e-12` float64 / `1e-4` float32), tightening T15(b)'s float64
  post-step parameter tier from `1e-10` to `1e-12` while aligning T15(b) to a uniform `learning_rate=1e-3`;
  and switched T15(l)'s `default_synth` check to `mode="lora_temporal"` (with an explicit `> 1e-4` update-norm
  check).
- `71514a6` & `40a580a`: added standalone `scripts/probes/float32_trajectory_noise.py` and aligned harness/probe
  comments with the actual measured headroom and Linux numbers.

**Development history (not `[V]` claims):**

- The Stage 1–2 float32 parity tiers had originally been calibrated on macOS arm64 only and failed on Linux
  x86_64 when first exercised there.
- The earlier mutation campaign was **not** re-run after `TRAJ_UPDATE_TOL["float32"]` was loosened from `1e-4`
  to `1e-3`; the `float64` tiers that guard the distributed logic were untouched (and T15(b)'s `float64`
  post-step parameter bound was tightened from `1e-10` to `1e-12`).

**Single-L4 CUDA/NCCL results (at `71514a6`, one NVIDIA L4, driver `580.178.04`, `uv`-selected Python `3.15.2`,
`torch 2.15.0+cu132`, CUDA `13.2`, NCCL `2.30.7`):**

- `bash scripts/cuda_check.sh` passed (`266 passed, 3 skipped` device tests; `735 passed, 4 skipped` whole
  suite including all 142 CPU/`gloo` multi-process items; `docs/VERIFICATION.md` §13). Note that `uv` selected
  `CPython 3.15.2`, which is outside the CI matrix (`3.10–3.13`). An L4 was available in only one of roughly
  ten zones tried.
- One-off 12-scenario one-process bit-identity check against pre-implementation `57a781c` matched bitwise
  (`BITWISE MATCH`, including `57a781c` self-reproduction) on both CPU and CUDA (`docs/VERIFICATION.md` §11).
- One-off single-rank NCCL smoke passed `19/19`, confirming NCCL accepts all collective dtypes and CUDA
  primitive paths (`docs/VERIFICATION.md` §11).
- One-off multi-rank NCCL trainer trial ran 2 and 4 ranks sharing the single L4 (requiring
  `NCCL_MULTI_RANK_GPU_ENABLE=1` because NCCL `2.30.7` refuses multiple ranks per GPU by default): all 23
  scenarios per rank count exited cleanly with bitwise cross-rank equality (`xrank_equal=True`), `float64`
  parity `<= 2.99e-14`, `float32` `update_rel <= 1.80e-04`, and `bf16` `update_rel` `1.03e-02`–`2.50e-02`
  (no `bf16` tier set; `docs/VERIFICATION.md` §11).

**Evidence (`40a580a` on macOS arm64; `71514a6` on Linux x86_64 and one NVIDIA L4; full commands and tables in
`docs/VERIFICATION.md` §11, §13, §16, §17):**

- macOS arm64 (`40a580a`, PyTorch `2.15.0`): `-m "not network"` `511 passed, 15 skipped, 213 deselected`; four
  multi-process files `142 passed` at `TIMESTRAL_TEST_F32_TOL_SCALE=0.1`; `torch==2.4.0` `138 passed, 4 skipped`;
  `float32_trajectory_noise.py` all `0.00e+00` on `2.15.0` and `2.4.0`.
- Linux x86_64 AMD EPYC 7B12 (`71514a6`): `-m "not network"` `469 passed, 24 skipped, 169 deselected` on both
  managed Python `3.12.16` (`2.15.0+cu132`) and managed Python `3.10.22` (`2.14.1+cu130`); four multi-process
  files `142 passed` at `TIMESTRAL_TEST_F32_TOL_SCALE=0.2` across `OMP_NUM_THREADS` `1`, `2`, `4`, acceptance
  first, and Python `3.10.22` (`t=2`), plus `142 passed` at `0.1` at default threads; `float32_trajectory_noise.py`
  max `8.08e-06` (`2.15.0+cu132` and `2.14.1+cu130`, identical).
- `uv run ruff check .` and `uv run ruff format --check .` clean.

**Not done / open:**

- **Stage 3 (multi-GPU CUDA/NCCL hardware verification):** prerequisite backend/device switch in
  `tests/distributed_harness.py` (currently CPU/`gloo` only via `_ALLOWED_CPU_BACKENDS`), T15 on NCCL, ranks
  on separate GPUs and real `LOCAL_RANK` binding, `torchrun`, `bf16` parity tier calibration, CUDA
  `compile_mode="default"` and `num_workers > 0` across ranks, NCCL rank-failure behaviour, throughput/VRAM/
  scaling, and released-weights T15 bounds (`docs/phases/phase-5-multi-gpu.md` §3, spec §10 & §12).
- Live concurrent multi-process Hub download check at `huggingface_hub==0.23.0` (source reading only,
  `docs/VERIFICATION.md` §12).
- Maintainer decisions on `F.linear` `requires_grad` dispatch remedies and the ten pre-existing one-process
  observations in spec §12.
- Optional gradient accumulation (unticked, unscheduled).

---

## 2026-11-23 — Stage 3 measuring scripts, first tracked single-L4 NCCL session, and script review findings

**What was built (`b25854e`, `70c3561`, `82a21ca`).** Three tracked verification scripts were added ahead of
the Stage 3 multi-GPU session (separate from acceptance test T15, which remains CPU/`gloo` only in
`tests/distributed_harness.py`):

1. `scripts/probes/distributed_trainer_check.py` (`b25854e`): a 41-scenario (`g1`–`g8`) ungraded measuring
   driver that runs the real `TimestralTrainer` across ranks under `torchrun` (`gloo` or `nccl`;
   `--device-map per-rank|shared`), compares against a one-process reference archive written beforehand on
   the same device (`--reference-mode sketch` by default — SHA-256 hashes, exact norms, exact deltas for
   parameter sets `<= 8,192` elements, and a `k = 8,192` CountSketch for larger sets — with
   `--reference-mode exact` available), and reports cross-rank parameter equality, parity metrics, timing,
   and peak VRAM without grading them.
2. `scripts/probes/distributed_rank_failure_check.py` (`70c3561`): spawns `N` ranks on `gloo` or `nccl`,
   terminates rank 1 via `os._exit(1)` mid-`fit()` at step 2, and records each rank's exit status, exception
   (if any), and elapsed time against explicit process-group and wall timeouts.
3. `scripts/multi_gpu_check.sh` (`82a21ca`): portable session runner automating environment capture,
   optional 4-GPU `distributed_parity.py --device cuda`, reference generation, `torchrun` driver runs, and
   rank-failure checks in numbered stages (`01_env`–`09_rank_failure_*`) with `.done` markers and macOS
   `TMPDIR=/tmp` handling.

**Development history (not `[V]` claims):**

- The measuring driver was written across three engineers because two ran out of usage quota mid-task before
  the full script suite landed at `82a21ca`.
- **macOS `DataLoader` worker deadlock diagnosis:** while exercising `num_workers > 0` on macOS, the driver
  engineer reproduced a silent hang with no project code: PyTorch `DataLoader` workers share tensors through
  `torch_shm_manager`, which binds an `AF_UNIX` socket under `TMPDIR`. When the socket path exceeds macOS's
  103-byte `sun_path` limit (a `TMPDIR` of ~70 bytes is enough), the worker's feeder thread raises, drops the
  batch, and stays alive while the main process waits forever at 0% CPU with no error. This affects
  one-process `TimestralTrainer.fit()` with `num_workers > 0` identically to multi-process `fit()`;
  recorded in spec §12 (pre-existing one-process observation #11, with a candidate fail-fast remedy in
  `_build_data_loader`) and `docs/DEVELOPMENT.md`.

**First tracked hardware session (`82a21ca`, one rented NVIDIA L4, Linux x86_64, 8 vCPU, driver `595.112.07`,
Python `3.12.16`, `torch 2.15.0+cu132`, CUDA `13.2`, NCCL `2.30.7`; full tables in `docs/VERIFICATION.md`
§13 and §17.4):**

- `bash scripts/multi_gpu_check.sh --shared-gpu --python 3.12` ran 16 stages with `N = 2` and `N = 4` NCCL
  ranks sharing the single GPU (`NCCL_MULTI_RANK_GPU_ENABLE=1`, `--device-map shared`): 15 stages exited `0`,
  1 skipped (`02_prototype_parity_cuda`, requires 4 GPUs), zero hangs, and exact parameter-level cross-rank
  equality (`xr_eq=yes, xr_max=0.00e+00`) in all 246 NCCL rows (`tiny_random`, `5M`, `28M` × `N = 2, 4`) and
  all 70 Linux CPU/`gloo` rows (`tiny_random` × `N = 2, 4`). Reproduces and supersedes the earlier one-off
  shared-GPU trial in `docs/VERIFICATION.md` §11.
- `float64` stayed at round-off on almost every scenario, peaking on identical-start `full_ft` distillation
  (`upd_l2` up to `7.81e-12` on `tiny` at `N = 4` — `12.8×` below T15's `1e-10`; up to `8.73e-11` on `5M` at
  `N = 4` and `1.55e-10` on `28M` at `N = 2`, with no tier set for released weights).
- `float32` stayed `<= 1.00e-04` (`upd_l2`) on `tiny_random` NCCL (`<= 1.62e-04` Linux `gloo`, `<= 1.73e-04`
  macOS `gloo`, excluding identical-start `full_ft` distillation); on released `5M`/`28M`, `final_trainable`
  agreed to `2.38e-08`–`5.58e-04` (`5.61e-04` exact) while `upd_l2` reached `1.04e-01` (`5M`
  `domain_lora_temporal_f32`) / `7.33e-02` (`28M` `domain_lora_temporal_f32`) even though `float64` on those
  scenarios stayed at `8.87e-14`–`5.94e-12` (consistent with spec §12's CUDA default-SDPA `float32` gradient
  inaccuracy, recorded as an interpretation rather than a demonstrated cause).
- `bf16` `upd_l2` ranged `4.18e-03`–`5.75e-02` on `tiny_random` and `3.37e-02`–`3.48e-01` (`3.51e-01` exact)
  on released `5M`/`28M`, with step-loss rel diff up to `1.27e-01` (`5M`, `N = 2`, `bf16_full_ft_replay`);
  cause uninvestigated, recorded as an open question in spec §12 with no tier set.
- `compile_mode="default"` (`2` graphs without replay, `5` graphs with replay), `num_workers=2` (`tail_s`
  `0.16–0.32 s` on Linux L4 NCCL vs `~20 s` on macOS `gloo`), `export()`, early stopping with restore, and
  stochastic training all ran cleanly across ranks on CUDA.
- **Per-backend rank-failure behaviour:** on macOS arm64 CPU/`gloo` (`pg_timeout=10 s`, reproduced locally),
  surviving ranks raised Python `RuntimeError` at `10.00–10.01 s` and exited cleanly (`10.42–10.58 s`); on
  shared-L4 NCCL (`pg_timeout=20 s`), surviving ranks were killed by `SIGABRT` at `35.85–36.19 s` (~20.5–20.8 s
  watchdog timeout + 15 s health-check/flight-recorder wait) with **no Python exception raised**, so neither
  `except` nor `finally` blocks ran. Neither backend hung.
- **Reference-mode check:** `ref_exact.zip` (`109.17 MiB`) and `ref_sketch.zip` (`0.64 MiB`) matched within
  `0.00%`–`1.60%` across 10 `tiny`/`5M` `N = 2` entries (`0.00%` in `loss_rel`).
- **Python `3.12.16` `cuda_check.sh`:** passed at `82a21ca` (`266 passed, 3 skipped` device tests;
  `735 passed, 4 skipped` whole suite; `docs/VERIFICATION.md` §13), closing the Python `3.12` branch gap left
  by the earlier `71514a6` `CPython 3.15.2` run.
- **Local macOS arm64 verification (`82a21ca`, Python `3.12.13`, PyTorch `2.15.0`):** `uv run ruff check . &&
  uv run ruff format --check .` clean; `TMPDIR=/tmp uv run pytest -m "not network" -q -p no:cacheprovider`
  passed (`511 passed, 15 skipped, 213 deselected`); `bash scripts/multi_gpu_check.sh --backend gloo --ranks 2
  4 --pg-timeout 10` passed (`7` stages exit `0`, `1` skipped, `350.0 s`; `docs/VERIFICATION.md` §17.4).

**Independent script review findings (re-run planned after fixes):**

- An independent code review of the `82a21ca` scripts confirmed that parameter-level `xr_eq`/`xr_max` and the
  parity columns (`param_l2`, `upd_l2`, `loss_rel`, `val_rel`) compare the right tensors with the right
  denominators (with `sketch` matching `exact` to ~1% for `float32`/`bf16` and `float64 >= 1e-14`), while
  identifying tooling limitations documented explicitly in `docs/VERIFICATION.md` §17.4: `xr_eq`/`xr_max` omit
  buffers (`rope.inv_freq`), `param_l2`/`upd_l2` cover trainable parameters only, `sketch` has a `float64`
  bucket-accumulation floor below `~1e-14` on `5M`/`28M`, `step_ms` includes validation/worker spawn on
  validating scenarios and `wall_s` lacks a pre-`t0` barrier, the two CUDA `moved_student` rows mixed CPU and
  CUDA `rope.inv_freq` across ranks (provisional parity), the reference mismatch guard omits frozen base
  weights in LoRA modes and some hardcoded hyperparameters, and `multi_gpu_check.sh`'s stage timeout and
  `.done` resume markers need hardening (separate-session `torchrun` orphans on hang; flag-blind resume and
  stale-reference gating). The GPU session will be repeated after those script fixes land.

**Documentation updated (`docs/SPEC.md` §5.6, §7.6, §9, §10, D28, D35, §12; `docs/VERIFICATION.md` §11, §13,
§17.4; `docs/phases/phase-5-multi-gpu.md`; `docs/DEVELOPMENT.md`; `docs/STATUS.md`; `README.md`;
`CHANGELOG.md`; `AGENTS.md` §1):** recorded the `82a21ca` session and qualifications, updated §5.6/§7.6/D35
for per-backend peer-death behaviour (`gloo` `RuntimeError` vs NCCL `SIGABRT` bypassing `finally`), added the
four API/runtime observations from the driver author and the macOS `TMPDIR` hazard to spec §12, and kept
Phase 5 Stage 3 unticked.

**Not done / open:**

- Fix the `82a21ca` review findings in the three Stage 3 scripts and re-run the GPU session.
- **Stage 3 (separate-GPU CUDA/NCCL hardware verification):** backend/device switch in
  `tests/distributed_harness.py` to run T15 on NCCL, separate-GPU `multi_gpu_check.sh --device-map per-rank`
  (real `LOCAL_RANK` binding, cross-GPU NCCL determinism, 4-GPU `distributed_parity.py --device cuda` on tiny
  and `5M`/`28M`, separate-GPU NCCL rank-failure checks), `bf16` investigation and tier calibration,
  released-weights T15 bounds (including cosine similarity), and 1/2/4-GPU throughput/VRAM/scaling
  measurements (`docs/phases/phase-5-multi-gpu.md` §3, spec §10 & §12).
- Live concurrent multi-process Hub download check at `huggingface_hub==0.23.0` (source reading only,
  `docs/VERIFICATION.md` §12).
- Maintainer decisions on `F.linear` `requires_grad` dispatch remedies, the eleven pre-existing one-process
  observations (including the macOS `TMPDIR` guard), and the four Stage 3 driver API observations in spec §12.
- Optional gradient accumulation (unticked, unscheduled).

---

## 2026-11-24 — Stage 3 tooling review fixes, Python session driver rewrite, and final-tip shared-GPU verification

**What was built (`713aaac`–`8e8f490`).** The limitations flagged in the `82a21ca` Stage 3 script review were
addressed across two stretches of work (library code under `src/` was unchanged; the CountSketch `float64`
floor is resolved for vectors `<= 1,000,000` parameters under `--reference-mode auto` and remains present with
explicit floor flags on larger `full_ft` / `28M head_only` entries):

1. **Three probe and session-script review-and-fix rounds (`713aaac`–`1963050`):**
   - Extracted shared helpers into `scripts/probes/_dist_common.py` (`713aaac`) and hardened
     `scripts/probes/distributed_trainer_check.py` (`fd24a2a`, `9ab347d`, `ef92d53`, `d6ad220`): cross-rank
     equality (`xr_eq`, `xr_max`) now hashes and compares **both** parameters and buffers (`rope.inv_freq`),
     `--reference-mode auto` became the default (exact `float64` deltas up to `1,000,000` parameters and exact
     deltas for any dtype up to `8,192` parameters, `k = 8,192` CountSketch otherwise), archive schema v5 added
     every-rank pre-flight zip CRC, metadata, tensor byte-size, model/config, and script-revision checks plus
     per-scenario rank-0 `frozen_base_sha256` and dataset-fingerprint verification, timing and memory columns
     became `wall_s`, `step_ms` (largest-over-ranks mean per-step wall time over steps `3..K` excluding
     validation), `tail_s`, and `vram_mib` (with `val_s` in `notes` on validating scenarios), compiled graph
     notes switched to `graphs=s1:<k0>->end:<k1>(+<delta>,bumps_post_s1=<b1>)`, CUDA `moved_student` rows now
     copy warmed parameters only onto each rank's device (`_build_moved_full_ft_params`,
     `_apply_moved_student_params`) without touching device-built `rope.inv_freq`, and `export()` reload parity
     was added (`1e-6` float64 / `1e-4` float32).
   - Hardened `scripts/probes/distributed_rank_failure_check.py` (`0edb56e`, `15bf1c5`, `e1dece0`): separate
     `--startup-timeout` and post-kill `--wall-timeout`, strict allow-list of distributed transport/collective
     exceptions and in-`fit()` `SIGABRT`, finite-positive CLI guards, and orphan cleanup.
   - Split rank-failure stages into `09_rank_failure_gloo_n<N>` and `10_rank_failure_nccl_n<N>` so `--backend
     nccl` exercises both Linux CPU/`gloo` and CUDA/NCCL rank failures (`a20d386`, `866816a`, `1963050`),
     forwarded `--sizes` into `02_prototype_parity_cuda` (requiring `4` in `--ranks`), and added content-bound
     resume markers.
2. **Rewrite of the session driver in standard-library Python (`bd24e47`–`8e8f490`):**
   - Replaced the bash + embedded-Python implementation in `scripts/multi_gpu_check.sh` with a single
     standard-library-only Python `>= 3.10` program (`scripts/multi_gpu_check.py`, structured around
     `SessionRunner` phase methods and `StageProcess` in `64f4dbe`) behind a thin bash wrapper
     (`scripts/multi_gpu_check.sh`) that locates a `>= 3.10` interpreter without parsing CLI flags and `exec`s
     `multi_gpu_check.py`.
   - Added 103 stub-stage unit tests in `tests/test_multi_gpu_check.py` (included in `-m "not network"`).

**Why the session script was rewritten.** Three consecutive review rounds (`a20d386`, `866816a`, `1963050`)
kept surfacing subtle defects at the boundary between bash (bash 3.2 on macOS vs GNU bash on Linux, `| tee`
pipelines, `trap` handling, subshell exit codes) and its embedded Python snippets. In addition, an unattended
GPU session started under `nohup` was terminated by `SIGHUP` because the shell runner unconditionally installed
a `HUP` handler, overriding the inherited `SIG_IGN` disposition. Consolidating argument parsing, stage
sequencing, content-bound resume signatures, signal handling, pipe draining, stage timeouts, and process-tree
cleanup into one standard-library Python process eliminates the shell/Python boundary and makes the driver
directly testable with stub stages.

**Key driver decisions:**

- **Argument-transparent `exec` wrapper (`33d16e9`):** `scripts/multi_gpu_check.sh` never parses user options
  (avoiding flag drift and `--python` vs driver-Python confusion); `exec` transfers PID, signal dispositions,
  and exit status directly to `multi_gpu_check.py`.
- **Honour inherited `SIG_IGN` (`4682669`):** signals already ignored on entry (`SIGHUP` under `nohup`, or
  ignored `SIGINT`/`SIGTERM`) remain ignored for the entire session and across stage `exec`s; non-ignored
  `SIGINT`/`SIGTERM`/`SIGHUP` trigger clean teardown and exit `128 + signum`.
- **Content-bound resume with git-free fallback (`1963050`, `d7c63ff`):** `.done` markers bind `sig=` to tree
  state (`git` with automatic `content_hash` fallback over `scripts/`, `src/`, `pyproject.toml`, `uv.lock` when
  `.git` is absent), resolved runtime versions, stage args (excluding ephemeral `--master-port` and binding
  verified `ref_sha=`), and tracked env vars, plus `artifact_sha256=` and full archive verification for
  reference stages.
- **Confirmed-live tree kill and bounded post-exit drain (`12e3dc7`, `e2e7b6a`):** stage cleanup tracks
  `_TIMESTRAL_STAGE_TOKEN` across confirmed-live non-zombie PIDs and safe stage-only PGIDs, removes partial
  reference archives on every failure/timeout/interrupt path, and bounds post-exit pipe draining to `2 s`
  before killing lingering descendants.

**Defects caught by review and live gates worth a successor knowing:**

- **Closed stdout/stderr turning exit `0` (or the real exit code) into `120` (`8e8f490`):** when a caller
  closed stdout/stderr (for example piping into `head`), Python's buffered stdio flush during interpreter
  shutdown raised `BrokenPipeError` and overwrote the exit status with `120`. The failure was hidden in one
  engineer's shell because `PYTHONUNBUFFERED=1` was set in the environment. Fixed by redirecting broken stdio
  fds to `/dev/null` and flushing in `_silence_broken_stream`, and tested with `env -u PYTHONUNBUFFERED`.
- **Relative `--log-dir` expanded to an absolute path (`bbb65a2`):** resolving `--log-dir` eagerly leaked
  home/checkout paths into stage CLI arguments, `.done` signatures, and `summary.txt`. Fixed so a relative
  `--log-dir` stays relative everywhere the driver writes paths (Python warnings and `faulthandler` tracebacks
  inside stage logs still print their own paths).
- **Finished stage with a lingering descendant holding stdout open hit full `--stage-timeout` (`e2e7b6a`):**
  an exited stage leader with a backgrounded grandchild holding the stdout pipe open previously blocked until
  the full stage timeout (`900 s`) fired. Bounded to a `2 s` post-exit drain grace before killing and logging
  the lingering descendants.
- **Wrapper CLI parsing (`33d16e9`) and `SIG_IGN` override (`4682669`):** removed option parsing from
  `multi_gpu_check.sh` and preserved inherited `SIG_IGN` across `SIGINT`/`SIGTERM`/`SIGHUP`.
- **Pre-existing `load_model(..., dtype=torch.float64)` float32 truncation (`d6ad220`, spec §12 observation #12,
  `docs/VERIFICATION.md` §11):** tightening float64 export-reload checks exposed that `load_model` constructs
  `TuneTimestralModel(config)` in default `float32`, loads weights into it (truncating `float64` weights to
  `float32`), and casts afterward. Reproduced locally at `8e8f490` on `tiny_random` across all six adaptation
  modes (one-off measurements in `docs/VERIFICATION.md` §11). Recorded without changing
  `src/timestral_tune/models/loader.py`; the probe retains `1e-6` float64 export-reload tolerance in the
  meantime.
- **Open Linux `ps`-fallback unit-test failure in `tests/test_multi_gpu_check.py`:** at `8e8f490` on the Linux
  L4 VM, `test_find_token_pids_exact_matching_empty_proc_fallback_and_scan_order` failed (mocking `/proc` empty
  to exercise the non-Linux `ps`-based scan fallback, which returned no hits on that Linux host; the live Linux
  `/proc` scan path passed every signal and timeout gate; counts in `docs/VERIFICATION.md` §17.4). Fix pending.

**Evidence (full tables, commands, and cross-session comparisons in `docs/VERIFICATION.md` §11 and §17.4):**

- **Final-tip shared-GPU L4 session (`8e8f490`, one rented NVIDIA L4, driver `580.178.04`, Python `3.12.16`,
  `torch 2.15.0+cu132`, CUDA `13.2`, NCCL `2.30.7`, driver on `python3` `3.10.12` on `PATH`):**
  `bash scripts/multi_gpu_check.sh --shared-gpu --python 3.12 --log-dir mgc_logs` finished in `47 min 51 s`:
  **17 of 17 runnable stages exited `0` (18 listed, `02_prototype_parity_cuda` skipped as `shared GPU`)**, `0`
  leftover CPU/GPU processes, bitwise parameter **and** buffer equality (`xr_eq=yes, xr_max=0.00e+00`) in all
  `316` rows (`246` shared-GPU NCCL + `70` Linux CPU/`gloo`), plus Linux `gloo` (Python `RuntimeError` on TCP
  transport reset) and shared-L4 NCCL (`SIGABRT` without Python cleanup) rank-failure stages and live gates for
  resume, `SIGINT`/`SIGTERM`/`SIGHUP` (default and `SIG_IGN`) on real NCCL stages, reference-stage interrupt
  artifact removal, `--stage-timeout 20` tree-kill, no-`.git` `content_hash`, and relative `--log-dir` hygiene
  (`docs/VERIFICATION.md` §17.4).
- **Cross-session reproducibility across `a20d386`, `866816a`, and `8e8f490` (`docs/VERIFICATION.md` §17.4):**
  between S3 (`866816a`) and S4 (`8e8f490`, same machine type, different VM instance per the provisioning
  record; same driver `580.178.04`), all non-compiled NCCL rows and all Linux CPU/`gloo` rows matched in the
  printed parity columns and `vram_mib` (timing columns and `val_s` varied), whereas
  `compile_default_lora_variate` / `compile_default_lora_replay` varied run-to-run on CUDA (10 of 12 compiled
  NCCL rows differed between S3 and S4; cause uninvestigated). Against S2 (`a20d386`, a different host per the
  provisioning record), CPU-dependent rows (Linux `gloo` and CPU-warmed `*_moved_f32`) differed in their printed
  values (up to `12×` on `gloo` and `4.1×` on NCCL `*_moved_f32`; full breakdown in `docs/VERIFICATION.md`
  §17.4).
- **Local macOS arm64 verification (`8e8f490`, Python `3.12.13`, `torch 2.15.0`):**
  `--backend gloo --ranks 2 4 --pg-timeout 10` passed (**7 of 7 runnable stages exited `0`**, 8 listed,
  `xr_eq=yes` in all `70` rows, `gloo` rank failure raised `RuntimeError` at timeout); all 6 signal/disposition
  live checks passed; `-m "not network"` **`614 passed, 15 skipped, 213 deselected`** and
  `tests/test_multi_gpu_check.py` **`103 passed`** across 3 consecutive runs; one-off
  `load_model(..., dtype=torch.float64)` reproduction recorded in `docs/VERIFICATION.md` §11.
- `uv run ruff check . && uv run ruff format --check .` clean.

**Documentation updated:** `docs/VERIFICATION.md` (§11, §13, §17.1, §17.4), `docs/SPEC.md` (§5.6, §7.6, §9,
§10, D35, §12 — Phase 5 Stage 3 kept unticked; no §11 D-entry added for the session driver, matching existing
practice where §11 records library decisions only), `docs/DEVELOPMENT.md`, `docs/STATUS.md`,
`docs/phases/phase-5-multi-gpu.md`, `CHANGELOG.md`, and this entry.

**Not done / open:**

- Fix the 1 failing Linux `ps`-fallback unit test in `tests/test_multi_gpu_check.py`
  (`test_find_token_pids_exact_matching_empty_proc_fallback_and_scan_order`).
- **Stage 3 (separate-GPU CUDA/NCCL hardware verification):** run `bash scripts/multi_gpu_check.sh --python
  3.12 --log-dir <dir>` on a 4-GPU host (or `--ranks 2` on 2 GPUs; real `LOCAL_RANK` binding, 4-GPU
  `02_prototype_parity_cuda` with `--sizes 5M 28M`, cross-GPU NCCL determinism, separate-GPU rank-failure
  checks), add a backend/device switch to `tests/distributed_harness.py` for T15 on NCCL, investigate the
  larger `bf16` differences and compiled CUDA run-to-run variation, calibrate `bf16` and released-weights T15
  bounds (including cosine similarity), and measure 1/2/4-GPU throughput/VRAM/scaling
  (`docs/phases/phase-5-multi-gpu.md` §3, spec §10 & §12).
- Live concurrent multi-process Hub download check at `huggingface_hub==0.23.0` (source reading only,
  `docs/VERIFICATION.md` §12).
- Maintainer decisions on `F.linear` `requires_grad` dispatch remedies, the twelve pre-existing one-process
  observations (including `load_model(..., dtype=torch.float64)` truncation and the macOS `TMPDIR` guard), and
  the four Stage 3 driver API observations in spec §12.
- Optional gradient accumulation (unticked, unscheduled).
