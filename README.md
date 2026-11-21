<p align="center">
  <img src="assets/banner.png" width="100%"
       alt="timestral-tune: fine-tuning, selective LoRA and conformal calibration for Timestral time-series models">
</p>

# timestral-tune

Fine-tuning, selective LoRA, and conformal calibration for Mistral's Timestral time-series foundation models in PyTorch.

> **Status: pre-alpha.** Loading, the data pipeline and `head_only` training work and are tested against
> the official code. LoRA, full fine-tuning, conformal calibration and checkpoint export are not built yet,
> so a tuned model cannot be saved yet. See [docs/STATUS.md](docs/STATUS.md).

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
| Loading on any device | Done: CUDA, CPU, Apple GPU (PyTorch 2.8+) |
| Data pipeline | Done: polars / pandas / pyarrow frames, point-in-time windows, time splits, collation |
| Training forward and loss | Done; equal to official inference at float32 round-off |
| `head_only` training | Done, fp32 |
| LoRA, full fine-tuning, `bf16` | Not built (Phase 2) |
| Conformal calibration, export | Not built (Phase 2) |
| Benchmarks, metrics, Lacuna integration | Not built (Phase 3) |
| Acceptance tests T1–T14 | 10 of 14 implemented and passing on 5M and 28M; 4 are Phase 2 placeholders |
| CI | Passing on Python 3.10–3.13 as of the last push |

## What works today: `head_only` training

```python
from timestral_tune import TimestralTrainer, TrainingConfig
from timestral_tune.data import TimeSeriesDataset

# A long-format frame: one row per (series, timestamp). polars, pandas and pyarrow all work.
dataset = TimeSeriesDataset.from_polars(
    df,
    id_col="symbol",
    time_col="timestamp",
    target_cols=["load"],
    past_covariates=["temperature"],     # known up to the forecast origin only
    known_covariates=["is_holiday"],     # known in advance over the horizon
    context_length=512,
    forecast_horizon=96,
    stride=96,
)
train, validation = dataset.split_by_time("2024-02-15")   # training sees nothing from this date on

config = TrainingConfig(model_name_or_path="mistralai/Timestral-5M", mode="head_only", batch_size=8, epochs=3)
trainer = TimestralTrainer(config, train, validation)
print(trainer.evaluate().loss)   # zero-shot validation loss
result = trainer.fit()           # trains the quantile head only; everything else stays frozen
print(result.best_eval_loss)
```

The tuned model is `trainer.model`. There is no `export()` yet, so it cannot be saved in the official
checkpoint layout until Phase 2. Learning-rate defaults are placeholders that have not been tuned.

To check your own data for leakage and parity with the official inference path:

```python
from timestral_tune.utils import run_checks

batch = dataset.collator(trainer.model.config)([dataset[i] for i in range(8)]).to(trainer.device)
for check in run_checks(trainer.model, batch):
    print(check)
```

## Repository layout

```
docs/SPEC.md                         Specification and overview guide (start here)
AGENTS.md                            How to work in this repo: rules, commands, commit conventions
docs/                                Also: status, development log, verification record, upstream and phase notes
src/timestral_tune/                  The package (stubs) and the vendored upstream code in _vendor/
tests/                               Scaffold checks and acceptance-test placeholders
scripts/                             Vendoring, drift check, and verification probes
papers/                              Reference papers (only the CC BY Timestral paper is tracked)
assets/                              README banner (drawn by scripts/make_banner.py)
archive/                             The original draft of the specification
```

## Development

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # create the environment
uv run pytest                # downloads the 5M and 28M checkpoints on first run
uv run pytest -m "not network"   # no downloads; what CI runs
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
