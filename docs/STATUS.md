# Project Status

**Last updated:** 2026-11-21
**Stage:** pre-alpha scaffold. No library functionality is implemented.
**Active phase:** Phase 0 (Foundations), partly done. See spec §10.

## What exists

| Area | State |
|---|---|
| Specification (`docs/SPEC.md`, revision 4) | Complete for v0.1; verified against the paper, upstream code and the 5M / 28M checkpoints |
| Vendored upstream (`src/timestral_tune/_vendor/timestral/`) | Done at commit `a534bdb`; manifest and drift check in place; no drift as of 2026-11-21 |
| Package skeleton | 18 documented stub modules in 7 subpackages; only `__version__` is exported |
| Packaging | `pyproject.toml` (hatchling), `uv.lock`; wheel builds and contains the vendored licence |
| Scaffold tests | 6 pass (Python 3.10 and 3.13, macOS) |
| Acceptance tests T1–T14 | **0 of 14 implemented.** All 14 are skipped placeholders in `tests/acceptance/` |
| Verification probes | 4 working probes in `scripts/probes/`; results in `docs/VERIFICATION.md` |
| CI | Green on the first run (2026-11-21): lint, build, network-free tests on Python 3.10–3.13, Linux. The weekly drift job is enabled; its first scheduled run is pending |
| Process docs | `AGENTS.md`, this folder |

## What has been verified, and where

Only on one machine (Apple M4, PyTorch 2.15.0, macOS). See `docs/VERIFICATION.md`.

- Model structure facts the LoRA design depends on.
- Role firewall and gradient behaviour.
- Apple GPU forward and gradient parity, with the float64 statistics on CPU.
- A prototype of the full training step (normalised forward, median-folded frame, pinball loss) runs and
  differentiates on CPU and the Apple GPU.

Linux has only run the network-free scaffold tests (in CI). Not verified anywhere yet: CUDA, the PyTorch
2.4 floor, the 123M and 209M checkpoints.

## Next steps

In order. Details and exit criteria are in spec §10.

1. **Finish Phase 0.**
   - `models/loader.py`: load at pinned Hub revisions, config always from `config.json`.
   - `models/device.py`: the float64-on-CPU shims as subclasses (prototype: `install_cpu_offload` in
     `scripts/probes/_common.py`).
   - Implement acceptance test T14 (prototype: `scripts/probes/device_parity.py`).
2. **Start Phase 1.** `models/forward.py` and tests T1–T3 first (prototype: `norm_forward` in
   `scripts/probes/_common.py`), then the loss (T8), then the data layer (T4–T7, T12).
3. **Run the target-frame check** (spec §4.3 and §12): pretrained loss and coverage under
   `median_folded` against `true_values`, horizons longer than one patch.

## Open questions and known issues

Full list in spec §12. The ones that affect the next steps:

- **Target frame (decision D2)** rests on a reading of one paper sentence. Unconfirmed until step 3 above.
- **CPU fp32 gradient gap**, cause unknown (`docs/VERIFICATION.md` §6). Compare gradients against the
  float64-weights reference.
- **Hyperparameter defaults are placeholders** until the Phase 3 sweeps.
- **CI actions use a deprecated Node.js version.** GitHub warns that `actions/checkout@v4` and
  `astral-sh/setup-uv@v5` target Node.js 20. They still run; bump them when newer majors are confirmed.
- **PyTorch floor.** The spec says 2.4+; everything so far ran on 2.15.0.

## Branch state

See `git log` and `git branch`. The scaffold was merged into `main` by fast-forward and pushed to the
private `origin` on 2026-11-21. The rest of Phase 0 (loader, device shims, T14) is not done and goes on a
new branch off `main`.
