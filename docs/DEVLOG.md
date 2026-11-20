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
Rewrote `timestral_tune_handoff_document.md` on the original's structure, with provenance tags on every claim, a model-facts section,
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
- CI and drift workflows written, not run (no remote).
- Process docs: `AGENTS.md` and `docs/`.

**Evidence.** `uv run pytest`: 6 passed, 14 skipped, on Python 3.13 and 3.10. Lint clean. Wheel builds and
contains the vendored licence. Drift check: no drift. Probe results: `docs/VERIFICATION.md`.

**Commits.** The scaffold was committed as a series on `phase-0/scaffold`. The "every commit passes tests"
rule can only hold from the commit that adds the tests onward; earlier commits in the series add the
specification, vendored code and packaging before any test exists.

**Left open.**

- Phase 0 is not finished: loader, device shims and T14 are still stubs.
- The target-frame check (D2) has not been run.
- The CPU gradient gap is unexplained.
- Nothing has run on CUDA, on Linux, or on the PyTorch 2.4 floor.
- CI has never run.
