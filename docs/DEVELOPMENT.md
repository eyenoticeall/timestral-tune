# Development Guide

Practical reference for working in this repository. Rules and commit conventions are in `AGENTS.md`;
design is in `docs/SPEC.md`.

## Setup

Requires Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # creates .venv with the package (editable) and dev tools
uv sync --python 3.10        # to work on the supported floor
```

The lockfile `uv.lock` is committed. Use `uv add` / `uv add --group dev` to change dependencies so that
the lockfile stays in step, and commit it with the change.

## Everyday commands

```bash
uv run pytest                          # everything; placeholders show as skipped
uv run pytest -m "not network"         # what CI runs
uv run pytest tests/test_scaffold.py   # scaffold checks only
uv run pytest -m acceptance            # acceptance tests only
uv run ruff check .                    # lint
uv run ruff format .                   # format
uv build                               # build wheel and sdist
```

## Layout

```
src/timestral_tune/
  _vendor/timestral/     Official inference package at a pinned commit (generated; do not edit)
  config.py              TrainingConfig
  data/                  Batch layout, dataset, collator, categorical encoders
  models/                Loader, device shims, training forward (built); LoRA adapter, conformal (stubs)
  losses/                Pinball loss
  trainer/               Loss step, training loop, callbacks
  utils/                 Verifiers (built); metrics (stub)
  replay/, export.py, integrations/    Stubs for later phases
tests/
  conftest.py, helpers.py   Fixtures and builders shared by all tests
  test_*.py                 Unit tests per module
  acceptance/               T1–T14 from spec §9
scripts/
  vendor_upstream.py     Regenerates _vendor/timestral
  check_upstream_drift.py
  probes/                Verification probes behind the [V] tags
```

Every stub module's docstring names its spec section, roadmap phase, acceptance tests and planned public
names. Start there when implementing a module.

## Implementing a module

1. Read the stub's docstring and the spec sections it names.
2. Find its acceptance tests in `tests/acceptance/`. Implement the test first where practical: replace
   the `@NOT_IMPLEMENTED` marker, keep the test's name and docstring.
3. Implement against the vendored code. Import from `timestral_tune._vendor.timestral`, never from a
   top-level `timestral`.
4. Export new public names from the subpackage `__init__.py` and, when they are part of spec §7, from
   `timestral_tune/__init__.py`.
5. Update the stub's "Status" line, then follow the end-of-session steps in `AGENTS.md`.

Shared building blocks to reuse rather than re-derive: `trainer/step.py::compute_loss` (the training loss),
`models/forward.py` (forward and frames), `utils/validation.py` (the acceptance checks as functions), and
`tests/helpers.py` (a tiny random model, synthetic raw series and frames, and `upstream_batch`, which builds
a batch with upstream's own input assembly as ground truth).

## Testing strategy

- **Three models per test.** Most acceptance tests take the `forecaster` fixture, which runs them on a tiny
  random model (network-free, so CI covers the logic) and on the released 5M and 28M checkpoints (marked
  `network`). The roadmap's exit criteria are about the released checkpoints, so run the full suite locally
  before claiming an item done: `uv run pytest`.
- **Acceptance tests** (`tests/acceptance/`) are the definition of done and carry the `acceptance` marker.
- **Ground truth is upstream.** Compare with the vendored forecaster on the same inputs
  (`tests/helpers.py::upstream_batch`, `forecast()`), not with hand-computed expectations.
- **Tolerances.** Forward results agree with upstream at float32 round-off (1e-6); do not assert
  bit-identity, which does not hold across PyTorch versions. Compare gradients with the float64-weights
  reference, never with CPU float32 (`docs/VERIFICATION.md` §6).
- **Perturbation tests** must use non-constant noise. A constant offset is absorbed by the normalisation
  and makes leakage tests pass vacuously.
- **Device tests** take their device list from `tests/conftest.py::available_accelerators` and skip
  cleanly when a device is absent or the PyTorch version is too old for it.
- **Older PyTorch.** `uv run --isolated --python 3.12 --with "torch==2.4.0" pytest` runs the suite on the
  floor without touching the project environment.
- **Training tests** copy the model first (`copy.deepcopy`); the fixtures are shared across tests.

## Devices

- CPU works everywhere, from PyTorch 2.4.
- The Apple GPU needs PyTorch 2.8 or newer, runs float32 only, and has no float64: move tensors to CPU
  before converting them to float64. `models/device.py` does this for upstream's code.
- CUDA is the intended main training target and has never been run in this project.

## Probes

Probes are small scripts that measure facts the specification relies on. Run them with
`uv run python scripts/probes/<name>.py --help`. They download checkpoints on first use.

Add a probe when the spec needs a new measured fact. Keep each probe single-purpose, print labelled
lines, pin revisions through `_common.load`, and record the result in `docs/VERIFICATION.md`.

## Style

- Ruff handles lint and formatting; line length 120. Vendored code and Markdown are excluded.
- Type hints on public functions. Docstrings state shapes for tensors, as upstream does.
- Prefer boolean masks with `True` meaning "attend" or "observed", matching upstream.
