<p align="center">
  <img src="assets/banner.png" width="100%"
       alt="timestral-tune: fine-tuning, selective LoRA and conformal calibration for Timestral time-series models">
</p>

# timestral-tune

Fine-tuning, selective LoRA, and conformal calibration for Mistral's Timestral time-series foundation models in PyTorch.

> **Status: pre-alpha (Phases 0–3).** Loading, the data pipeline, all five adaptation modes (`conformal`,
> `head_only`, `lora_variate`, `lora_temporal`, `full_ft`), `bf16` + `torch.compile` on CUDA,
> upstream-loadable checkpoint export, raw-space evaluation & forgetting metrics, optional Lacuna
> integration, and real-dataset benchmarks (5M & 28M) are built and verified (14/14 acceptance tests
> passing). See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [docs/STATUS.md](docs/STATUS.md).

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
| Training forward and loss | Done; trimmed covariate layout (`L + 2h` tokens), equal to official inference (`<= 1e-12` fp64) |
| Adaptation modes | Done: `head_only` (with frozen-feature caching), `lora_variate` (with single-row guard), `lora_temporal`, `full_ft` |
| CUDA acceleration | Done: `bf16` autocast, fused `AdamW`, optional `torch.compile` (verified on NVIDIA L4) |
| Conformal calibration | Done: `ConformalCalibrator` (CQR) and `OnlineConformalCalibrator` (ACI) |
| Checkpoint export | Done: upstream-loadable `config.json` + merged `model.safetensors`, LoRA adapters, `conformal.json`, sidecar, `LICENSE`, `NOTICE` |
| Benchmarks, metrics, Lacuna integration | Done: `utils/metrics.py`, `integrations/lacuna.py`, sweep-backed defaults, 5M & 28M tables in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| Acceptance tests T1–T14 | All 14 of 14 implemented and passing on tiny, 5M and 28M (CPU, Apple GPU, CUDA) |
| CI | Passing on `main` (Python 3.10–3.13 on Linux) |

## Quick start

```python
from timestral_tune import LoraConfig, TimestralTrainer, TrainingConfig
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

config = TrainingConfig(
    model_name_or_path="mistralai/Timestral-5M",
    mode="lora_variate",
    batch_size=8,
    epochs=3,
    output_dir="./runs/load-lora",
)
lora = LoraConfig(r=8, lora_alpha=16, axes=("variate",), targets=("q", "v"))
trainer = TimestralTrainer(config, train, validation, lora_config=lora)
print(trainer.evaluate().loss)   # zero-shot validation loss
result = trainer.fit()           # trains only the variate LoRA adapters
print(result.best_eval_loss)
trainer.export()                 # writes upstream-loadable merged checkpoint + adapters + NOTICE/LICENSE
```

Default settings (`sign_flip=True`, learning rates `head_only=3e-3`, `lora_variate=3e-3`,
`lora_temporal=1e-3`, `full_ft=1e-4`, and LoRA `r=8, targets=("q","k","v","o")`) are backed by
Phase 3 validation and forgetting sweeps across `5M`–`209M` in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
For `full_ft` on the two smaller checkpoints (`5M` and `28M`), we recommend setting
`learning_rate=3e-4`: it achieved lower error in our benchmarks (three seeds on `5M`, one seed on
`28M`), at the cost of about two percentage points more retail forgetting on `5M`. Keep the
`1e-4` default on `123M` and `209M`, where the two rates tie or favour `1e-4` on validation and
`1e-4` forgets less.

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
src/timestral_tune/                  The package (Phases 0-3 built, Phase 4 replay stubbed) and the vendored upstream code in _vendor/
tests/                               Unit tests and the acceptance tests of spec §9
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
