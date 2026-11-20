# `timestral-tune` — Project Specification & Technical Handoff

> **Repository Name:** `timestral-tune`  
> **Tagline:** Fine-tuning, selective LoRA, and conformal calibration for Mistral's Timestral time-series foundation models in PyTorch.  
> **License:** Apache 2.0 (Software) | Weights governed by Mistral Modified MIT  
> **Target Python Version:** 3.10+ (PyTorch 2.4+)  

---

## 1. Executive Summary & Vision

`timestral-tune` is an open-source, pure-PyTorch adaptation toolkit designed specifically for Mistral AI's **Timestral** time-series foundation model family (`Timestral-5M`, `28M`, `123M`, `209M`).

While the official `mistralai/timestral` repository provides standalone inference utilities (`TimestralForecaster`), domain adaptation is relegated to closed/paid enterprise products (Mistral Forge). `timestral-tune` bridges this open-source gap by providing a lightweight, high-throughput training engine faithful to the original paper (*arXiv:2610.37585*). 

The library prioritizes **anti-forgetting adaptation**—offering a progressive ladder from zero-shot conformal calibration and head probing to selective low-rank adaptation (LoRA) and synthetic replay mixing—without requiring custom C++/CUDA compilation steps.

---

## 2. Core Value Proposition & System Architecture

```
                                  ┌─────────────────────────────────────────┐
                                  │           Raw Time Series Data          │
                                  └────────────────────┬────────────────────┘
                                                       │
                                  ┌────────────────────▼────────────────────┐
                                  │      Strict Point-In-Time Collator      │
                                  │     (Leak-Free Covariate Encoder)       │
                                  └────────────────────┬────────────────────┘
                                                       │
                                  ┌────────────────────▼────────────────────┐
                                  │   Decoupled Horizon Masking (DHF)       │
                                  └────────────────────┬────────────────────┘
                                                       │
         ┌─────────────────────────────────────────────┼─────────────────────────────────────────────┐
         │                                             │                                             │
┌────────▼─────────┐                         ┌─────────▼─────────┐                         ┌─────────▼─────────┐
│ Conformal Calib. │                         │   Selective LoRA  │                         │ Head Probe / Full │
│   (Post-Hoc)     │                         │ (Variate/Temporal)│                         │    Fine-Tuning    │
└────────┬─────────┘                         └─────────┬─────────┘                         └─────────┬─────────┘
         │                                             │                                             │
         └─────────────────────────────────────────────┼─────────────────────────────────────────────┘
                                                       │
                                  ┌────────────────────▼────────────────────┐
                                  │ 21-Quantile Pinball Loss (Target Only)  │
                                  └────────────────────┬────────────────────┘
                                                       │
                                  ┌────────────────────▼────────────────────┐
                                  │  Lacuna Audit Bundle & HF Weights Out   │
                                  └─────────────────────────────────────────┘
```

### Key Technical Pillars

1. **Faithful DHF Masking & Supervision:** Implements Decoupled Horizon Forecasting (DHF) attention matrices that keep historical context causal while allowing target forecast positions to receive gradient updates without leaking target values into context.
2. **Covariate-Aware Pinball Loss:** Restricts the 21-quantile pinball loss computation strictly to target variate forecast positions, leaving `past-only` and `known-future` covariates unscored.
3. **Anti-Forgetting Adaptation Ladder:** Provides modular entry points for adaptation to prevent catastrophic destruction of pretrained temporal priors on small domain datasets.
4. **Integration with Validation Tooling:** Designed out-of-the-box to output trial distributions directly consumable by **Lacuna** for out-of-sample (OOS) audit gating, Deflated Sharpe Ratio (DSR) checks, and parameter-decay checks.

---

## 3. Mathematical & Loss Mechanics

### 3.1. 21-Quantile Pinball Loss

Timestral predicts non-crossing quantiles across 21 evaluation levels:

$$\mathcal{T} = \{0.01, 0.05, 0.10, 0.15, \dots, 0.85, 0.90, 0.95, 0.99\}$$

For a target value $y$ and predicted quantile $\hat{y}_\tau$ at level $\tau \in \mathcal{T}$, the pinball loss is defined as:

$$\mathcal{L}_\tau(y, \hat{y}_\tau) = \max \left( \tau (y - \hat{y}_\tau), (\tau - 1)(y - \hat{y}_\tau) \right)$$

The total step loss for a target variate $v$ at horizon position $t$ is averaged across all 21 quantiles:

$$\mathcal{L}_{v, t} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \mathcal{L}_\tau(y_{v, t}, \hat{y}_{v, t, \tau})$$

### 3.2. Masked Batch Loss Aggregation

Let $M_{v, t} \in \{0, 1\}$ be an indicator matrix where $M_{v, t} = 1$ if and only if position $(v, t)$ is a **target variable at a forecast horizon position** and is non-missing (`observation_mask == 1`).

The global batch loss is:

$$\mathcal{L}_{\text{batch}} = \frac{\sum_{b=1}^{B} \sum_{v=1}^{V} \sum_{t=1}^{T} M_{b, v, t} \cdot \mathcal{L}_{b, v, t}}{\sum_{b=1}^{B} \sum_{v=1}^{V} \sum_{t=1}^{T} M_{b, v, t} + \epsilon}$$

---

## 4. Adaptation Modes

| Mode | Trainable Parameters | Primary Use Case | Overfitting Risk |
|---|---|---|---|
| `conformal` | $0$ (Post-hoc quantile scaling) | Low-data regimes, real-time risk buffers | Very Low |
| `head_only` | Quantile output projection head only | Financial return distributions, micro-domain shifts | Low |
| `lora_variate` | LoRA rank adapters on Variate Attention (`r=4..16`) | Cross-asset correlation changes while locking temporal priors | Low–Moderate |
| `lora_temporal` | LoRA rank adapters on Temporal Attention (`r=4..16`) | Seasonality/regime adaptation | Moderate |
| `full_ft` | 100% of weights | Large-scale datasets (1,000+ series) | High |

---

## 5. Proposed Package Architecture

```
timestral_tune/
│
├── __init__.py
├── config.py                 # Training, LoRA, and DHF configuration dataclasses
├── data/
│   ├── collator.py           # DHF batch collator, attention mask generator
│   ├── dataset.py            # Polars/PyArrow/Pandas zero-copy dataset loader
│   └── encoders.py           # Leak-free target encoder for categorical covariates
│
├── models/
│   ├── adapter.py            # LoRA injection into temporal/variate attention blocks
│   ├── conformal.py          # Online & rolling conformal quantile calibration engine
│   └── loader.py             # Hugging Face checkpoint wrapper for Timestral models
│
├── losses/
│   └── pinball.py            # Vectorized 21-quantile loss with boolean mask reduction
│
├── trainer/
│   ├── trainer.py            # Main PyTorch training loop (AMP, torch.compile, gradient clipping)
│   └── callbacks.py          # Progress logging, early stopping, Lacuna exporter
│
└── utils/
    ├── metrics.py            # CRPS, MASE, pinball loss, empirical coverage (P10-P90)
    └── validation.py         # Temporal integrity and gradient flow verifiers
```

---

## 6. Public API Spec (Draft v0.1)

### 6.1. Basic Training Interface

```python
from timestral_tune import TimestralTrainer, TrainingConfig, LoraConfig
from timestral_tune.data import TimeSeriesDataset

# 1. Load dataset with explicit covariate roles
dataset = TimeSeriesDataset.from_polars(
    df,
    target_cols=["realized_vol"],
    past_covariates=["volume", "spread"],
    future_covariates=["is_fomc_day", "hour_of_day"],
    context_length=1024,
    forecast_horizon=96,
)

# 2. Configure selective LoRA adaptation
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["variate_attn.q_proj", "variate_attn.v_proj"], # Target variate attention
    dropout=0.05,
)

train_config = TrainingConfig(
    model_name_or_path="mistralai/Timestral-28M",
    learning_rate=3e-4,
    batch_size=32,
    epochs=5,
    mixed_precision="bf16",
    gradient_clip_val=1.0,
    output_dir="./checkpoints/timestral-28m-vol",
)

# 3. Instantiate and run trainer
trainer = TimestralTrainer(
    config=train_config,
    lora_config=lora_config,
    train_dataset=dataset,
)

trainer.fit()
```

### 6.2. Conformal Recalibration (Zero-Gradient Baseline)

```python
from timestral_tune.models import ConformalCalibrator

# Calibrate raw quantiles post-hoc over a rolling calibration window
calibrator = ConformalCalibrator(target_coverage=0.80)
calibrator.fit(predictions=val_preds, actuals=val_targets)

calibrated_quantiles = calibrator.transform(test_preds)
```

---

## 7. v0.1 Development Roadmap

### Phase 1: Core Mechanics & Verification (Week 1)
- [ ] Implement `PinballLoss` with strict target-masking reduction.
- [ ] Implement `DHFDataCollator` for context/forecast token separation and asymmetric covariate masking.
- [ ] Add unit test suite for gradient leakage verification (assert zero gradients on non-target covariate positions).

### Phase 2: Adaptation & Calibration (Week 2)
- [ ] Add `LoraConfig` and PEFT/custom injection for Timestral's temporal vs. variate attention layers.
- [ ] Implement post-hoc `ConformalCalibrator` for 80% and 90% interval coverage adjustment.
- [ ] Ensure full compatibility with `torch.compile(mode="reduce-overhead")` and `bf16`/`fp32` precision.

### Phase 3: Benchmarks & Documentation (Week 3)
- [ ] Run benchmark suite comparing **Zero-Shot vs. Conformal vs. LoRA vs. Full FT** on realized volatility and retail demand datasets.
- [ ] Export OOS test outputs to **Lacuna** format for statistical audit verification.
- [ ] Publish documentation, launch benchmarks on GitHub, and release wheel on PyPI.

---

## 8. License & Compliance Guidelines

- **Code Base:** Published under **Apache 2.0**.
- **Model Weights:** Downloaded dynamically from Hugging Face at runtime (`mistralai/Timestral-*`); governed by Mistral AI's **Modified MIT License**.
- **Notice File:** Include an explicit license disclaimer in the repository root and `README.md` clarifying independent community authorship and attribution to Mistral AI for the base architecture and weights.