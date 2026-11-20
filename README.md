# timestral-tune

Fine-tuning, selective LoRA, and conformal calibration for Mistral's Timestral time-series foundation models in PyTorch.

> **Status: pre-alpha scaffold. Nothing here trains, adapts or calibrates a model yet.**
> The repository currently holds the specification, a vendored copy of the official inference code,
> an empty package skeleton, verification probes, and placeholder tests. See [docs/STATUS.md](docs/STATUS.md).

## What this will be

[Timestral](https://arxiv.org/abs/2610.37585) is Mistral AI's family of time-series foundation models
(5M, 28M, 123M and 209M parameters). The [official repository](https://github.com/mistralai/timestral)
provides inference only. `timestral-tune` aims to be an open adaptation toolkit for the released weights,
with a ladder of increasingly invasive options:

| Mode | What changes |
|---|---|
| `conformal` | Nothing in the model; forecast quantiles are recalibrated after the fact |
| `head_only` | The quantile head |
| `lora_variate` | Low-rank adapters on cross-series (variate) attention |
| `lora_temporal` | Low-rank adapters on temporal attention |
| `full_ft` | All weights, ideally with replay to limit forgetting |

Design goals: reproduce the paper's objective, masks and normalisation exactly and prove it with tests
against the official code; export checkpoints that load in the official package; make runs auditable.

The full design is in **[docs/SPEC.md](docs/SPEC.md)**, which
is the project's overview guide and source of truth.

## What exists today

| Piece | State |
|---|---|
| Specification | Written and verified against the paper, the official code and the released checkpoints |
| Vendored upstream code | Done, at a pinned commit, with a manifest and a drift check |
| Package `timestral_tune` | Skeleton only: every module is a documented stub |
| Acceptance tests T1–T14 | 0 of 14 implemented; all are skipped placeholders |
| Scaffold tests | 6 passing (imports, vendoring integrity, a tiny random model) |
| Verification probes | Working; they reproduce the measurements the specification cites |
| CI | Passing: lint, build, and the network-free tests on Python 3.10–3.13 |

## Repository layout

```
docs/SPEC.md                         Specification and overview guide (start here)
AGENTS.md                            How to work in this repo: rules, commands, commit conventions
docs/                                Also: status, development log, verification record, upstream and phase notes
src/timestral_tune/                  The package (stubs) and the vendored upstream code in _vendor/
tests/                               Scaffold checks and acceptance-test placeholders
scripts/                             Vendoring, drift check, and verification probes
papers/                              Reference papers (only the CC BY Timestral paper is tracked)
archive/                             The original draft of the specification
```

## Development

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # create the environment
uv run pytest                # scaffold tests pass; acceptance placeholders are skipped
uv run ruff check . && uv run ruff format --check .
```

The verification probes download released checkpoints from the Hugging Face Hub on first use:

```bash
uv run python scripts/probes/model_facts.py
uv run python scripts/probes/device_parity.py --device mps    # or cuda
```

More in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Contributors and coding agents should read
[AGENTS.md](AGENTS.md) first.

## Licence and important restrictions

- **This software** is licensed under [Apache 2.0](LICENSE). It includes a vendored copy of Mistral AI's
  Apache-2.0 `timestral` package; see [NOTICE](NOTICE).
- **Timestral model weights are not covered by Apache 2.0.** They are downloaded from the Hugging Face Hub
  under Mistral AI's Modified MIT License. That licence grants no rights to anyone whose company (or
  employer) had global consolidated revenue above **$20 million** in the preceding month, and it applies to
  derivatives and modifications. **Weights fine-tuned with this toolkit are derivatives** and carry the same
  restriction. Read the licence in the model repositories before using the weights. This is a summary, not
  legal advice.
- **Independence.** This is an independent community project, not affiliated with or endorsed by
  Mistral AI. "Timestral", the base architecture and the weights are Mistral AI's.
