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
