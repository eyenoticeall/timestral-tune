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
  refuses it below 2.8 (decision in spec §5.4); before that, the suite on 2.4.0 failed and then hung.
- **Bit-identity is not a safe assertion across PyTorch versions.** T1/T2 are bit-identical on 2.4.0 and
  2.15.0 but one case differs by 8e-10 on 2.8.0. The bound is now 1e-6.
- **`split_by_time` semantics were tightened** (decision D11): training gets nothing at or after the
  cutoff, so labels cannot overlap even without an embargo. The spec's rule 5 was reworded.

**Bugs the tests caught during development.** Three float64 conversions executed on the Apple GPU before
moving to CPU (in the frame computation, the device-order of a `.to()`, and the evaluation loop's
accumulator). All three were found by running on the device, none by reading.

**Evidence.** 262 passed, 4 skipped on PyTorch 2.15.0 (Python 3.13); 262 passed on 2.8.0 and 227 passed,
21 skipped on 2.4.0 (Python 3.12). T1–T8 and T12 pass on 5M and 28M, which is Phase 1's exit criterion.
Numbers: `docs/VERIFICATION.md`.

**Left open.**

- T14 on CUDA: no hardware. One command when there is.
- `phase-1` is not merged or pushed; CI has not run on it.
- The gradient-accuracy cause.
- The target-frame evidence is synthetic only.
- Phase 2 has not started. There is no export, so a tuned model cannot be saved in the upstream layout yet.
