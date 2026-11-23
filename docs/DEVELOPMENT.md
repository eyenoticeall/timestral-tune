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
uv run pytest                          # everything (downloads 5M and 28M on first run)
uv run pytest -m "not network"         # what CI runs
uv run pytest tests/test_scaffold.py   # scaffold checks only
uv run pytest -m acceptance            # acceptance tests only
uv run ruff check .                    # lint
uv run ruff format .                   # format
uv build                               # build wheel and sdist
```

## Releasing

Releases are published by `.github/workflows/release.yml` through PyPI trusted publishing; no token is
stored anywhere. Publishing happens only when the maintainer asks (`AGENTS.md` §6).

1. Set `__version__` in `src/timestral_tune/__init__.py` and add a `## X.Y.Z — date` section to
   `CHANGELOG.md`. Commit, push, and wait for CI to pass on that commit.
2. Run the release workflow by hand (`gh workflow run release.yml`). This builds and checks the package
   without publishing.
3. Tag that same commit and push the tag by name: `git tag vX.Y.Z && git push origin vX.Y.Z`. Never push
   all tags. The workflow checks that the tag matches the package version and the changelog, uploads the
   build to PyPI, and creates the GitHub release from the changelog section.
4. Install the published version in a clean environment and run the README quick start.

A version number can never be reused on PyPI, so a broken release is fixed by publishing the next patch
version.

## Layout

```
src/timestral_tune/
  _vendor/timestral/     Official inference package at a pinned commit (generated; do not edit)
  config.py              TrainingConfig, LoraConfig, ReplayConfig
  data/                  Batch layout, dataset, collator, categorical encoders
  models/                Loader, device shims, training forward, slice LoRA adapter, conformal calibrators
  losses/                Pinball loss
  trainer/               Loss step, training loop, callbacks, multi-process primitives and strategy
  export.py              Upstream-loadable checkpoint export, adapters, conformal, sidecar, LICENSE/NOTICE
  utils/                 Verifiers and raw-space evaluation / forgetting metrics
  integrations/          Optional Lacuna integration (`timestral-tune[lacuna]`, Python 3.11+)
  replay/                Replay sources and distillation to the frozen base model
tests/
  conftest.py, helpers.py   Fixtures, global PyTorch state leak guard, and builders shared by all tests
  distributed_harness.py    Multi-process CPU/gloo spawn harness, shared parity tiers, and headroom knob
  test_*.py                 Unit tests per module (including test_distributed.py, test_trainer_distributed.py, test_trainer_distributed_replay.py)
  acceptance/               T1–T15 from spec §9 (all 15 implemented and passing; T15 on CPU/gloo)
scripts/
  vendor_upstream.py     Regenerates _vendor/timestral
  check_upstream_drift.py
  probes/                Verification probes behind the [V] tags
  benchmarks/            Benchmark, sweep and significance runs behind docs/BENCHMARKS.md
  make_banner.py, make_results_chart.py   Draw the README images in assets/
  cuda_check.sh          Runs the device tests and the whole suite on a CUDA machine
```

No stub modules remain. Every module's docstring names the spec section it implements; start there when
changing a module.

## Implementing a module

1. Read the stub's docstring and the spec sections it names.
2. Find or add its acceptance tests in `tests/acceptance/` (following `test_tNN_*`), keeping the test
   names aligned with spec §9.
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
  before claiming an item done: `uv run pytest`. Without network access, or with a broken loader, those tests
  error; they are never skipped silently. Use `uv run pytest -m "not network"` to leave them out.
- **Acceptance tests** (`tests/acceptance/`) are the definition of done and carry the `acceptance` marker.
- **Ground truth is upstream.** Compare with the vendored forecaster on the same inputs
  (`tests/helpers.py::upstream_batch`, `forecast()`), not with hand-computed expectations.
- **Tolerances.** Forward results agree with upstream at float32 round-off (1e-6 for T1; 1e-5 fp32 and
  1e-12 fp64 for T2's trimmed covariate layout); do not assert bit-identity, which does not hold across
  PyTorch versions. Compare gradients with the float64-weights reference, never with CPU float32
  (`docs/VERIFICATION.md` §6).
- **Perturbation tests** must use non-constant noise. A constant offset is absorbed by the normalisation
  and makes leakage tests pass vacuously.
- **Device tests** take their device list from `tests/conftest.py::available_accelerators` and skip
  cleanly when a device is absent or the PyTorch version is too old for it.
- **Older PyTorch.** `uv run --isolated --python 3.12 --with "torch==2.4.0" pytest` runs the suite on the
  floor without touching the project environment. Each such run downloads a PyTorch build of a few hundred
  megabytes into the uv cache; `uv cache prune` reclaims the space afterwards.
- **Training tests** copy the model first (`copy.deepcopy`); the fixtures are shared across tests.
- **Global PyTorch state leak guard.** An autouse fixture (`_guard_torch_global_state` in `tests/conftest.py`)
  records `torch.get_num_threads()`, `torch.is_grad_enabled()`, and `torch.get_default_dtype()` around every
  test, restores any mutated value, and fails the test if anything leaked. Any test or helper (such as
  `tests/distributed_harness.py::_worker_main`) that changes process-global PyTorch state must restore it in
  `finally`.
- **Multi-process tests** (`tests/acceptance/test_distributed.py` for T15 cases `(a)–(l)`,
  `tests/test_distributed.py` for primitives and the harness, `tests/test_trainer_distributed.py` for
  white-box trainer internals, and `tests/test_trainer_distributed_replay.py` for multi-process replay and
  distillation) spawn separate Python processes on CPU/`gloo` through `tests/distributed_harness.py`. The
  harness uses the `spawn` start method, file-based rendezvous inside a fresh temporary directory (avoiding
  ephemeral TCP port collisions), `torch.set_num_threads(1)` per worker (restored in `finally`),
  `run_scenarios` to amortise process startup across balanced scenarios, and a hard wall-clock deadline that
  terminates/kills stuck workers (`DistributedTimeoutError`) so collective hangs fail fast. Run them with:
  `TMPDIR=/tmp uv run pytest tests/test_distributed.py tests/test_trainer_distributed.py tests/test_trainer_distributed_replay.py tests/acceptance/test_distributed.py`
  (`TMPDIR=/tmp` avoids macOS socket path-length failures with multi-process `DataLoader`s), or on the floor
  with
  `TMPDIR=/tmp uv run --isolated --python 3.12 --with "torch==2.4.0" pytest tests/test_distributed.py tests/test_trainer_distributed.py tests/test_trainer_distributed_replay.py tests/acceptance/test_distributed.py`
  (the four `num_workers > 0` tests skip on macOS below PyTorch 2.8; see `docs/VERIFICATION.md` §10, §17).
- **Float32 multi-process parity tolerances & headroom knob.** All five parity tiers (`EVAL_TOL`, `STEP_TOL`,
  `STEP_PARAM_TOL`, `TRAJ_TOL`, `TRAJ_UPDATE_TOL`) live in `tests/distributed_harness.py`. The `float64`
  tiers check the distributed logic and are never scaled; `TIMESTRAL_TEST_F32_TOL_SCALE` (default `1.0`,
  validated finite positive float) multiplies the `float32` tiers only. Use a value below `1.0` to verify
  headroom (`TIMESTRAL_TEST_F32_TOL_SCALE=0.1` checks `>= 10×` headroom on macOS arm64; `0.2` checks `>= 5×`
  on Linux x86_64), or above `1.0` when exploring a new platform to measure where assertions land before
  calibrating a tier. Any new or tightened `float32` tolerance must be checked on Linux x86_64 (across
  intra-op thread counts `1`, `2`, and `4`) as well as macOS arm64: on macOS arm64 one-process `fit()` is
  bit-identical across thread counts, whereas on Linux x86_64 it is not (`scripts/probes/float32_trajectory_noise.py`,
  `docs/VERIFICATION.md` §11, §17.1, §17.3).
- **Ubuntu system Python & `torch.compile` tests.** Ubuntu's system Python (for example `/usr/bin/python3.10`)
  ships without `Python.h` unless `python3-dev` is installed, causing all `torch.compile` tests to fail with
  a C++ compiler error even on CPU. Use a `uv`-managed Python (which includes the headers, e.g.
  `UV_PYTHON_PREFERENCE=only-managed uv sync --python 3.10`) when running the suite on Linux.

## Devices

- CPU works everywhere, from PyTorch 2.4.
- The Apple GPU needs PyTorch 2.8 or newer, runs float32 only, and has no float64: move tensors to CPU
  before converting them to float64. `models/device.py` does this for upstream's code.
- CUDA is the intended main training target. Verified on an NVIDIA L4 in fp32 and `bf16`
  (`docs/VERIFICATION.md` §13); there is no CUDA machine in CI. After changing anything device-related, run
  `bash scripts/cuda_check.sh` on a GPU machine. It needs a C compiler and Python headers; a uv-managed
  Python has them.

## Probes

Probes are small scripts that measure facts the specification relies on. Run them with
`uv run python scripts/probes/<name>.py --help`. They download checkpoints on first use.

Add a probe when the spec needs a new measured fact. Keep each probe single-purpose, print labelled
lines, pin revisions through `_common.load`, and record the result in `docs/VERIFICATION.md`.

## Style

- Ruff handles lint and formatting; line length 120. Vendored code and Markdown are excluded.
- Type hints on public functions. Docstrings state shapes for tensors, as upstream does.
- Prefer boolean masks with `True` meaning "attend" or "observed", matching upstream.
