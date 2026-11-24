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
  test_*.py                 Unit tests per module (including test_distributed.py, test_trainer_distributed.py, test_trainer_distributed_replay.py, test_multi_gpu_check.py)
  acceptance/               T1–T15 from spec §9 (all 15 implemented and passing; T15 on CPU/gloo)
scripts/
  vendor_upstream.py     Regenerates _vendor/timestral
  check_upstream_drift.py
  probes/                Verification probes behind the [V] tags
  benchmarks/            Benchmark, sweep and significance runs behind docs/BENCHMARKS.md
  make_banner.py, make_results_chart.py   Draw the README images in assets/
  cuda_check.sh          Runs the device tests and the whole suite on a CUDA machine
  multi_gpu_check.sh     Thin bash 3.2+ wrapper that execs multi_gpu_check.py
  multi_gpu_check.py     Stdlib-only Stage 3 multi-process/multi-GPU verification session driver
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
  or on the floor with
  `TMPDIR=/tmp uv run --isolated --python 3.12 --with "torch==2.4.0" pytest tests/test_distributed.py tests/test_trainer_distributed.py tests/test_trainer_distributed_replay.py tests/acceptance/test_distributed.py`
  (the four `num_workers > 0` tests skip on macOS below PyTorch 2.8; see `docs/VERIFICATION.md` §10, §17).
  **Why `TMPDIR=/tmp` on macOS:** PyTorch `DataLoader` workers share tensors through `torch_shm_manager`,
  which binds an `AF_UNIX` socket under `TMPDIR`. If the socket path exceeds macOS's 103-byte `sun_path`
  limit (a `TMPDIR` of roughly 70 bytes is enough, common in sandboxed or deeply nested environments), the
  worker's feeder thread raises, drops the batch, and stays alive, leaving the main process waiting forever at
  0% CPU with no error. This affects one-process `TimestralTrainer.fit()` with `num_workers > 0` the same way
  as multi-process `fit()` (there is no library guard yet; spec §12). Setting `TMPDIR=/tmp` avoids it.
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
- CUDA is the intended main training target. Verified on an NVIDIA L4 and on 4× NVIDIA GeForce RTX 3060 in
  fp32 and `bf16` (`docs/VERIFICATION.md` §13, §17.4); there is no CUDA machine in CI. After changing
  anything device-related, run `bash scripts/cuda_check.sh` on a GPU machine. It needs a C compiler and
  Python headers; a uv-managed Python has them.

## Probes

Probes are small scripts that measure facts the specification relies on. Run them with
`uv run python scripts/probes/<name>.py --help`. They download checkpoints on first use.

Add a probe when the spec needs a new measured fact. Keep each probe single-purpose, print labelled
lines, pin revisions through `_common.load`, and record the result in `docs/VERIFICATION.md`.

### Multi-process / Stage 3 verification scripts

Tracked scripts under `scripts/` drive Stage 3 hardware measurement on CPU/`gloo`, a single shared GPU, or a
multi-GPU host (they are ungraded measuring instruments, separate from T15, which runs on CPU/`gloo` through
`tests/distributed_harness.py`):

- **`scripts/multi_gpu_check.sh` and `scripts/multi_gpu_check.py`** — `multi_gpu_check.sh` is a thin bash
  3.2+ wrapper that locates any Python `>= 3.10` and `exec`s `scripts/multi_gpu_check.py`, a single
  standard-library-only Python program (never imports `torch` or third-party packages) that owns CLI parsing,
  stage sequencing, content-bound resume signatures, signal handling, per-stage timeouts, and process-tree
  teardown.
- **`scripts/probes/distributed_trainer_check.py`** — runs the real `TimestralTrainer` across ranks under
  `torchrun` (`--backend gloo|nccl`, `--device-map per-rank|shared`), compares against a one-process
  reference archive written beforehand on the same device (`--write-reference PATH`, then
  `torchrun ... --reference PATH`), and prints ungraded parity/timing/VRAM metrics across 41 scenarios in
  groups `g1`–`g8` (`--list`, `--only`, `--skip`, `--sizes 5M 28M`). Checks cross-rank equality over all
  parameters **and** buffers (`xr_eq`, `xr_max`). Defaults to `--reference-mode auto` (exact `float64` deltas
  when `dtype=float64` and `numel <= 1,000,000` or any dtype with `numel <= 8,192`; `k = 8,192` CountSketch
  otherwise; schema v5 archive with pre-flight CRC and digest validation), with `sketch` and `exact` also
  available.
- **`scripts/probes/distributed_rank_failure_check.py`** — spawns `N` independent ranks via
  `torch.multiprocessing` (`--backend gloo|nccl`, `--world-size N`, `--device-map per-rank|shared`,
  `--pg-timeout S`, separate `--startup-timeout` and post-kill `--wall-timeout`), lets `--kill-rank` (default
  `1`) call `os._exit(1)` mid-`fit()` at `--kill-step` (default `2`), and grades surviving ranks against a
  strict allow-list of distributed transport/collective exceptions or in-`fit()` `SIGABRT`.
- **`scripts/probes/_dist_common.py`** — shared constants, synthetic dataset builders, device-map validation,
  and `TMPDIR` socket-length checks shared by both probes.

**Which Python runs what:** the wrapper picks the interpreter that runs `multi_gpu_check.py` itself—first
`python3` on `PATH` if it is `>= 3.10`, then an installed interpreter found by
`uv python find --no-project --resolve-links ">=3.10"` (or without `--resolve-links`), falling back to
`uv run --no-project --python ">=3.10" python`—without inspecting CLI flags, and `exec`s `multi_gpu_check.py`
so Python owns its PID, signal dispositions, and exit code. Passing **`--python VERSION`** (for example
`--python 3.12`) selects the **project virtual environment's** Python passed to `uv sync --locked --quiet`
and `uv run --no-sync` inside every stage (where PyTorch, CUDA, and `Python.h` live; use a `uv`-managed
Python on Ubuntu hosts whose system Python lacks `Python.h`).

```bash
# 1. CPU/gloo dry run (macOS or Linux; tiny model only)
bash scripts/multi_gpu_check.sh --backend gloo --ranks 2 4 --pg-timeout 10 --log-dir multi_gpu_dry

# 2. Single-GPU shared NCCL (testing convenience; sets NCCL_MULTI_RANK_GPU_ENABLE=1 and --device-map shared)
bash scripts/multi_gpu_check.sh --shared-gpu --python 3.12 --log-dir multi_gpu_shared

# 3. Multi-GPU host (4 separate GPUs; --device-map per-rank, real LOCAL_RANK binding; pass --ranks 2 on 2 GPUs)
bash scripts/multi_gpu_check.sh --python 3.12 --log-dir multi_gpu_4gpu
```

**Stages executed:** `01_env` (syncs the locked venv and records OS/arch/Python/tree-state/PyTorch/CUDA/NCCL/GPU
facts); `02_prototype_parity_cuda` (`distributed_parity.py --device cuda`, forwarding `--sizes`; runs only for
`--backend nccl` without `--shared-gpu` when `>= 4` GPUs are visible and `4` is in `--ranks`, skipped
otherwise); `03_ref_<backend>_tiny_n<N>` / `04_<backend>_tiny_n<N>`; `05_ref_<backend>_released_n<N>` /
`06_<backend>_released_n<N>` (default `--sizes 5M 28M` for `nccl`, empty for `gloo`; pass `--sizes ""` to
skip); when `--backend nccl`, Linux CPU/`gloo` comparison stages `07_ref_gloo_tiny_n<N>` / `08_gloo_tiny_n<N>`
and `09_rank_failure_gloo_n<N>`; and primary rank-failure checks `09_rank_failure_gloo_n<N>` (for `gloo`) or
`10_rank_failure_nccl_n<N>` (for `nccl`). Full CLI options: `bash scripts/multi_gpu_check.sh --help`.

**Logs, markers, reference archives, and relative `--log-dir`:** per-stage logs (`<stage>.log`), completion
markers (`<stage>.done`), `summary.txt`, and reference archives (`refs/ref_*.zip`) are written under
`--log-dir` (default `multi_gpu_check_logs`; untracked, do not commit). Before any stage runs, the driver
creates and probes writability of both `--log-dir` and `<log-dir>/refs`: if either path is unwritable,
inaccessible, or blocked by a regular file, it prints one `error:` line naming the path as typed (without
expanding a relative path to `REPO_ROOT` or printing a traceback) and exits `2`. Other file `OSError`s that
occur mid-session (clearing stale files, writing `.done` markers, or writing `summary.txt`) are reported as
stage failures or warnings rather than raw tracebacks. Passing a **relative** `--log-dir` (such as the
default or `--log-dir mgc_logs`) keeps stage arguments, `.done` signatures, and `summary.txt` relative to the
repository root so machine and home paths do not leak into those files (note that Python itself may still
print absolute paths inside stage logs for warnings or `faulthandler` tracebacks).

**Content-bound resume and `--force`:** every passing stage writes `<stage>.done` containing a SHA-256 `sig=`
(plus `artifact_sha256=` for reference stages) bound to:
1. Tree state: `git` (`HEAD`, tracked `status`, `diff HEAD`, and untracked files under `scripts/` and `src/`
   excluding bytecode caches and `--log-dir`), automatically falling back to deterministic `content_hash`
   over `scripts/`, `src/`, `pyproject.toml`, and `uv.lock` when `git` is absent or the directory is not a
   git work-tree top-level (for example an exported archive).
2. Resolved project runtime identity (interpreter executable/version, `torch.__version__`,
   `torch.version.cuda`).
3. Exact stage arguments (omitting the ephemeral `--master-port` on `torchrun` stages and binding `ref_sha=`
   to the verified reference archive SHA-256 instead), separated by a domain boundary from tracked
   environment variables (`NCCL_*`, `GLOO_*`, `TORCH_*`, `PYTORCH_*`, `CUDA_*`, `CUDNN_*`, `CUBLAS_*`,
   `OMP_*`, `MKL_*`, `OPENBLAS_*`).
4. For reference stages, full zip CRC, metadata JSON, tensor member byte-size verification, and SHA-256
   equality of the archive on disk.

Re-running with the same `--log-dir` skips any stage whose `.done` signature and verified artifact still
match (`0 (resumed)`); changing code, CLI flags, tracked environment variables, runtime versions, or a
reference archive automatically invalidates and re-runs affected stages. Pass `--force` to ignore `.done`
markers and re-run every stage. Whenever a reference stage fails, times out, or is interrupted, any partial
`.zip` / `.zip.tmp` archive is deleted and its dependent `torchrun` stage is recorded as
`SKIPPED (ref failed)`.

**Signals (`SIGINT` / Ctrl-C, `SIGTERM`, `SIGHUP`) and `nohup`:** on entry, `multi_gpu_check.py` inspects the
inherited disposition of `SIGINT`, `SIGTERM`, and `SIGHUP`:
- Any signal already set to `SIG_IGN` on entry—such as `SIGHUP` under `nohup bash scripts/multi_gpu_check.sh ... &`
  over SSH—stays ignored for the entire session (child stages start in their own session via
  `start_new_session=True` and inherit `SIG_IGN` across `exec`), so disconnecting SSH does not abort the run.
- Any signal not ignored on entry interrupts cleanly at any point (start-up, mid-stage, timeout teardown, or
  stage boundaries): kills the active stage's process tree, removes incomplete `.done` markers and partial
  reference archives, writes an interrupted `summary.txt`, skips remaining stages, and exits `128 + signum`.

**Stage timeout and process-tree handling:** each stage runs in its own session tagged with a per-stage
environment variable (`_TIMESTRAL_STAGE_TOKEN`), streaming combined stdout/stderr through a non-blocking pipe
(no shell `| tee` pipeline). If a stage exceeds `--stage-timeout` (default `900` s), the driver collects
confirmed-live non-zombie PIDs carrying the stage token from `/proc` (primary on Linux) or pure-BSD `ps`
(`axewwo`/`axo` on macOS/BSD, or as fallback if Python cannot enumerate `/proc`), their live descendants, and
safe stage-only PGIDs, sends `SIGTERM` with a `3 s` pipe-draining grace period, then `SIGKILL` with a `2 s`
reap loop, deletes any partial artifact, records `124`, and continues to independent remaining stages. If a
stage leader exits while descendants still hold stdio open or stay alive, the driver drains for up to a `2 s`
post-exit grace period and then kills and logs a warning for any surviving descendants without waiting out
`--stage-timeout`. Runtime/GPU helper probe failures or timeouts (`180 s` helper limit) are reported
explicitly with stderr snippets (never silently as "0 GPUs"), and broken stdout/stderr pipes are redirected
to `/dev/null` so interpreter shutdown never overwrites the real exit code with `120`.

**Exit codes:**
- `0`: all runnable stages passed (or `-h`/`--help`).
- `1`: at least one stage failed or timed out (`124` in `summary.txt`, corrupt archive, port allocation
  failure, stage exception, or mid-session stale-file / `.done`-marker write `OSError`).
- `2`: invalid CLI arguments, missing Python `>= 3.10` or `uv`, inability to change to the repository root,
  unusable or unwritable `--log-dir` / `<log-dir>/refs`, failed tree-state/runtime/GPU helper probe, `0`
  visible CUDA GPUs for `--backend nccl`, only `1` visible GPU with default ranks and no `--shared-gpu`, or
  `--ranks` exceeding visible GPUs without `--shared-gpu`.
- Any non-zero exit code from `01_env`: `01_env` failed and aborted the session (for example `3` when
  `--backend nccl` is requested on a non-CUDA PyTorch build, `uv sync`'s or the environment probe's non-zero
  exit code, `124` on a stage timeout, or `127` if the command cannot be launched).
- `129` / `130` / `143`: interrupted by `SIGHUP` / `SIGINT` / `SIGTERM` (when not ignored on entry).

**Known process-escape limitation:** a descendant that strips `_TIMESTRAL_STAGE_TOKEN` from its environment
and outlives its parent (being reparented outside the tracked PPID tree), or runs on a system where neither
`/proc` nor `ps` can inspect processes (note that on Linux `procps-ng` `ps` itself reads `/proc`, so an
OS-level unmounted or inaccessible `/proc` cannot be worked around via `ps`), can escape tree termination;
neither `torchrun` nor PyTorch `DataLoader` workers do this.

**macOS `TMPDIR` handling:** `multi_gpu_check.py` automatically exports `TMPDIR=/tmp` whenever `TMPDIR` is
unset or longer than 60 characters (and `distributed_trainer_check.py` fails fast via
`check_tmpdir_for_workers()` before spawning workers if `TMPDIR` would overflow macOS's 103-byte `sun_path`
limit; see "Testing strategy" above).

**Driver unit tests (`tests/test_multi_gpu_check.py`):** 105 stub-stage tests covering CLI parsing, wrapper
selection, resume signatures, `git` and `content_hash` tree hashing, signal dispositions, timeouts,
process-tree cleanup (including pure-BSD `ps` fallback), pre-stage `--log-dir` and `OSError` handling,
partial-artifact removal, relative log paths, and closed stdio streams (included in
`uv run pytest -m "not network"`; the tests strip `PYTHONUNBUFFERED` internally):

```bash
TMPDIR=/tmp uv run pytest tests/test_multi_gpu_check.py
```

## Style

- Ruff handles lint and formatting; line length 120. Vendored code and Markdown are excluded.
- Type hints on public functions. Docstrings state shapes for tensors, as upstream does.
- Prefer boolean masks with `True` meaning "attend" or "observed", matching upstream.
