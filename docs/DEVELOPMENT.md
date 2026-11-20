# Development Guide

Practical reference for working in this repository. Rules and commit conventions are in `AGENTS.md`;
design is in `timestral_tune_handoff_document.md`.

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
  config.py              Configuration dataclasses
  data/                  Dataset, collator, categorical encoders
  models/                Loader, device shims, training forward, LoRA adapter, conformal calibrators
  losses/                Pinball loss
  trainer/               Training loop and callbacks
  replay/                Replay generators and distillation (stretch)
  export.py              Checkpoint export
  integrations/          Optional integrations (Lacuna)
  utils/                 Metrics and verifiers
tests/
  test_scaffold.py       Network-free checks of the scaffold itself
  acceptance/            T1–T14 from handoff §9
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
4. Export new public names from the subpackage `__init__.py` and, when they are part of handoff §7, from
   `timestral_tune/__init__.py`.
5. Update the stub's "Status" line, then follow the end-of-session steps in `AGENTS.md`.

`scripts/probes/_common.py` contains working prototypes worth reading before writing the real thing:
`norm_forward` and `real_loss` (the training forward and loss) and `install_cpu_offload` (the device
shims). They are prototypes: the library versions must not monkeypatch the vendored modules.

## Testing strategy

- **Scaffold tests** (`tests/test_scaffold.py`) need no network. They build a tiny random model in code.
- **Acceptance tests** (`tests/acceptance/`) are the definition of done. Each carries the `acceptance`
  marker. Most need released checkpoints; mark those `network` so CI can exclude them.
- **Tests that need a checkpoint** should use the smallest one that proves the point (5M), at its pinned
  Hub revision, and repeat on 28M where the handoff asks for it.
- **Tolerance.** Compare forward results with a relative tolerance, and gradients against the
  float64-weights reference, not against CPU fp32 (see `docs/VERIFICATION.md` §6).
- **Perturbation tests** must use non-constant noise. A constant offset is absorbed by the normalisation
  and makes leakage tests pass vacuously.
- **Device tests** skip cleanly when the device is absent.

## Devices

- CPU works everywhere.
- The Apple GPU needs the float64 statistics to run on CPU and supports fp32 only (handoff §5.4).
- CUDA is planned as the main training target but has not been exercised yet.

## Probes

Probes are small scripts that measure facts the specification relies on. Run them with
`uv run python scripts/probes/<name>.py --help`. They download checkpoints on first use.

Add a probe when the spec needs a new measured fact. Keep each probe single-purpose, print labelled
lines, pin revisions through `_common.load`, and record the result in `docs/VERIFICATION.md`.

## Style

- Ruff handles lint and formatting; line length 120. Vendored code and Markdown are excluded.
- Type hints on public functions. Docstrings state shapes for tensors, as upstream does.
- Prefer boolean masks with `True` meaning "attend" or "observed", matching upstream.
