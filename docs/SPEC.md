# `timestral-tune` — Project Specification

> **Repository Name:** `timestral-tune`
> **Tagline:** Fine-tuning, selective LoRA, and conformal calibration for Mistral's Timestral time-series foundation models in PyTorch.
> **License:** Apache 2.0 (software) | Weights governed by Mistral's Modified MIT License, which carries a revenue cap (see §8)
> **Target Python Version:** 3.10+ with PyTorch 2.4+, matching upstream; the Apple GPU needs PyTorch 2.8+. The optional Lacuna integration needs Python 3.11+.
> **Devices (v0.1):** CUDA, CPU, and Apple GPU (MPS, fp32 only, PyTorch 2.8+). See §5.4.
> **Spec revision:** 5, dated 2026-11-21, describing library release v0.1. The original draft is kept at `archive/timestral_tune_handoff_document.v0.md`.
> **File name:** this document was `timestral_tune_handoff_document.md` at the repository root until 2026-11-21; older commit messages use that name.
> **Working in this repo:** start with `AGENTS.md`; current state is in `docs/STATUS.md`.

### How to read this document

Every technical claim carries a provenance tag, so a reader can tell what is established from what we chose.

| Tag | Meaning |
|---|---|
| **[P §x]** | Stated in the Timestral paper, arXiv:2610.37585v1, section x (`papers/timestral_arxiv_2610.37585v1.pdf`) |
| **[U file:line]** | Read from the official `mistralai/timestral` code at the pinned commit below |
| **[H]** | Read from the `config.json` of the released checkpoints on the Hugging Face Hub |
| **[V]** | Verified by running the released 5M and 28M checkpoints on 2026-11-21 (torch 2.15.0, Apple M4, 16 GB) |
| **[D]** | A design decision of this project. Not from Mistral; open to revision |

The **[V]** claims are recorded in `docs/VERIFICATION.md`. Most are reproduced by a probe in `scripts/probes/`; a few came from one-off runs and are marked as such there. Tests T6, T7, T10 and T14 in §9 turn the behavioural ones into permanent tests.

### Pinned upstream versions

| Artifact | Revision |
|---|---|
| `mistralai/timestral` (code, Apache 2.0) | commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` (2026-10-15, "Timestral release") |
| `mistralai/Timestral-5M` | `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` |
| `mistralai/Timestral-28M` | `49b26409ac87b5935539a5d8ffe7f397b720db47` |
| `mistralai/Timestral-123M` | `5a147287addda85f6acd8da3fb8858fe9af7e8f7` |
| `mistralai/Timestral-209M` | `9874d252ab0aae16f7f7c999f98f98d6ad7d8365` |
| Lacuna (`lacuna-quant`) | 0.14.1 |

Upstream does not accept pull requests and makes no stability promise for its private functions, so `main` may move without notice. All line references below are to the pinned commit. In the vendored copy, five files carry a four-line notice header, so their lines sit four lower (`docs/UPSTREAM.md`).

---

## 1. Executive Summary & Vision

`timestral-tune` is an open-source, pure-PyTorch adaptation toolkit for Mistral AI's **Timestral** time-series foundation models (`Timestral-5M`, `28M`, `123M`, `209M`).

The official `mistralai/timestral` repository provides inference only (`TimestralForecaster`). Mistral's announcement offers fine-tuning, training-mix control and replay of pre-training data through its commercial Forge product, and publishes no training code. `timestral-tune` fills that gap for the open weights.

The library prioritises **anti-forgetting adaptation**. It offers a ladder of increasing intervention: post-hoc conformal calibration, head probing, selective low-rank adaptation (LoRA), full fine-tuning, and replay mixing. No custom C++ or CUDA build steps are needed.

### What "faithful to the paper" does and does not mean

The paper specifies the objective, the attention masks, the normalisation, and the architecture. This library reproduces those exactly and tests the reproduction against the released code and weights.

The paper does **not** specify an optimiser, learning rate, schedule, batch size, block-length distribution, or augmentation settings, and Mistral has not released its synthetic corpus. Everything in those areas is our design **[D]** and is validated empirically in Phase 3.

### Non-goals for v0.1

- Reproducing Mistral's pretraining or post-training.
- Architectural extensions (new covariate types, new heads).
- Training horizons longer than one decode block (1,024 steps) in a single forward pass.
- Re-hosting or redistributing Mistral's weights.

---

## 2. Core Value Proposition & System Architecture

```
                         ┌─────────────────────────────────────────┐
                         │   Raw long-format frame (id, time, …)   │
                         └────────────────────┬────────────────────┘
                                              │
                         ┌────────────────────▼────────────────────┐
                         │      Strict point-in-time windowing     │
                         │  (per-window, context-only encodings)   │
                         └────────────────────┬────────────────────┘
                                              │
                         ┌────────────────────▼────────────────────┐
                         │  DHF collator: packed rows, roles,      │
                         │  [context tokens ; prediction tokens]   │
                         └────────────────────┬────────────────────┘
                                              │
     ┌────────────────────┬───────────────────┼───────────────────┬────────────────────┐
     │                    │                   │                   │                    │
┌────▼─────┐      ┌───────▼──────┐    ┌───────▼───────┐   ┌───────▼───────┐    ┌───────▼──────┐
│Conformal │      │  Head probe  │    │ LoRA: variate │   │ LoRA: temporal│    │   Full FT    │
│(post-hoc)│      │              │    │   attention   │   │   attention   │    │ (+ replay)   │
└────┬─────┘      └───────┬──────┘    └───────┬───────┘   └───────┬───────┘    └───────┬──────┘
     │                    └───────────────────┼───────────────────┴────────────────────┘
     │                                        │
     │                   ┌────────────────────▼────────────────────┐
     │                   │ 21-quantile pinball loss, normalised    │
     │                   │ space, observed target horizon only     │
     │                   └────────────────────┬────────────────────┘
     │                                        │
     │                   ┌────────────────────▼────────────────────┐
     └──────────────────►│ Export: upstream-loadable checkpoint,   │
                         │ adapter, conformal offsets, metadata    │
                         └────────────────────┬────────────────────┘
                                              │
                         ┌────────────────────▼────────────────────┐
                         │ Evaluation via the upstream forecaster  │
                         │ → metrics, trial records, signal frames │
                         └────────────────────┬────────────────────┘
                                              │ (optional)
                         ┌────────────────────▼────────────────────┐
                         │ Lacuna audits them and emits its bundle │
                         └─────────────────────────────────────────┘
```

### Key Technical Pillars

1. **Faithful Decoupled Horizon Forecasting (DHF).** Context patches and forecast positions use separate tokens. Context is encoded causally and is never masked to create a horizon; prediction tokens read the context and never write to it **[P §3.1]**. The library reuses the upstream mask builders unchanged and proves its training forward pass equal to upstream inference.
2. **Role-aware pinball loss in normalised space.** The 21-quantile pinball loss is computed in the model's normalised space and only on observed target values in the forecast horizon. Past-only and known-future covariates are inputs and are never scored **[P §4.1, §3.2]**.
3. **Anti-forgetting adaptation ladder.** Five modes of increasing capacity, plus replay mixing and a forgetting metric, so that small domain datasets do not destroy the pretrained priors.
4. **Auditable outputs.** Every run records its parameters, data fingerprint and metrics. An optional integration writes trial records and signal frames in the form Lacuna consumes, so Lacuna can apply its multiple-testing and overfitting checks.

---

## 3. Model Facts the Design Depends On

This section did not exist in the original draft. Implementers need it before §4.

### 3.1 Architecture

| | 5M | 28M | 123M | 209M |
|---|---|---|---|---|
| `d_model` | 192 | 384 | 640 | 768 |
| Blocks | 8 | 12 | 20 | 24 |
| Heads (head dim 64) | 3 | 6 | 10 | 12 |
| SwiGLU width | 512 | 1024 | 1712 | 2048 |
| Temporal-attention modules | 8 | 12 | 20 | 24 |
| Variate-attention modules | 2 | 3 | 5 | 6 |
| Layers holding variate attention | 3, 7 | 3, 7, 11 | 3, 7, …, 19 | 3, 7, …, 23 |
| Parameters | 5.2M | 27.8M | 122.6M | 209.3M |

Sources: **[P Table 1]**, **[H]**. All four checkpoints share `patch_size=32`, `block_size=32`, `context_length=32768`, and `quantile_width_factor=3.0` **[H]**.

- **Patching.** Each patch is 32 time steps, embedded from its normalised values plus per-step missing indicators **[P §3.3]**.
- **Limits.** Context up to 32,768 steps (1,024 patches). One decode block covers up to 32 patches (1,024 steps); longer horizons are decoded block by block, feeding medians back **[P §3.1]**.
- **Blocks.** Pre-norm, RMSNorm, SwiGLU. Every block has temporal attention; every fourth block also has variate attention. Both use a sigmoid output gate. No dropout or stochastic depth was used **[P §3.3]**.
- **Head.** A two-layer residual MLP emits 21 non-crossing quantiles per time step. The median is free; other levels are cumulative softplus increments **[P §3.3]**, **[U model.py:57-76]**.
- **Read the checkpoint config.** `quantile_width_factor` is 3.0 in every released checkpoint but defaults to 2.0 in the code **[H]**, **[U config.py:63]**. Never construct a config from code defaults.

### 3.2 Module naming, which dictates the LoRA design

- Attention modules live at `transformer.layers.{i}.attns.{j}`. Temporal and variate attention are the same class, told apart only by the `is_time` attribute **[U layers.py:143-159]**. No module is named `variate_attn`, `q_proj` or `v_proj` **[V]**.
- Each attention module has three bias-free linears: `wQKV` (`d_model → 3·d_model`, fused), `wO`, and `_gate` **[U layers.py:157-159]**, **[V]**.
- In `wQKV.weight`, rows `[0, D)` are the query projection, `[D, 2D)` the key projection, and `[2D, 3D)` the value projection, where `D = d_model` **[U layers.py:161-166]**. The run confirmed that the last third is the value projection and the first two-thirds are query and key together **[V]**; the order of query and key within those two-thirds is from the code only.
- The head is `head.projection`, three bias-free linears **[U model.py:87-89]**. Its size is `4D² + 3360·D` parameters: 792,576 for 5M, which is 15.2% of the model **[V]**. The same formula gives about 6.8% for 28M, 3.1% for 123M and 2.4% for 209M.

### 3.3 Roles, masks and normalisation

Each row is a target, a past-only covariate or a known-future covariate. Roles are not embedded; they act only through masks and the choice of normalisation **[P §3.2]**.

| | Temporal attention | Variate attention | Normalisation |
|---|---|---|---|
| Target | causal | reads targets and both covariate kinds | causal per patch, then asinh |
| Past-only covariate | causal | reads past-only and known-future covariates; never targets | causal per patch, then asinh |
| Known-future covariate | whole window | reads known-future covariates only | whole-window statistics; 0/1 indicator rows left unchanged |

Sources: **[P §3.2, §3.3]**, **[U masks.py:104-118]**, **[U model.py:135-151]**.

Verified on the 5M checkpoint **[V]**: noise added to a target's context changed neither covariate row's output (0.00); noise on a past-only covariate changed the target and itself but not the known-future row; noise on a known-future covariate changed all three.

### 3.4 Two inference layouts

Upstream uses two token layouts, and training mirrors both.

- **No covariates** **[U model.py:204-229]**: `L` context tokens followed by `h` prediction tokens. Masks come from `build_forecast_time_mask` **[U masks.py:34-47]**.
- **With covariates** **[U model.py:231-273]**: the window is context plus horizon, `n = L + h` patches. There are `n` context tokens and `n` prediction tokens (`2n` in total). Target and past-only rows hold a blank, unobserved horizon; known-future rows hold their real future values. Masks come from `build_two_stream_time_mask` **[U masks.py:68-101]** and `build_two_stream_variate_mask` **[U masks.py:121-146]**. Only the horizon prediction tokens are read out.

### 3.5 What upstream does not expose

- `predict_block` returns raw-space quantiles after `sinh` denormalisation **[U model.py:229, 273]**. The normalised head output needed for the loss is only reachable through the private `_run_two_stream` **[U model.py:153-162]**.
- `TimestralForecaster.forecast` runs under `torch.inference_mode()` **[U forecaster.py:160]**, so it cannot be used for training.
- Statistics are computed in float64, which the Apple GPU backend does not support **[V]**. There are four such sites: the causal statistics **[U scaler.py:29]** (reached from `model.scale` and from the forecaster's `_cap_quantiles`), `advancing_block_frames` **[U scaler.py:185]**, `_refine_block` **[U forecaster.py:472]**, and `_encode_categorical` **[U forecaster.py:265]**. §5.4 describes the workaround.
- `TimestralConfig.from_dict` rejects unknown keys **[U config.py:121-124]**, so no extra metadata may be written into `config.json`.

---

## 4. Mathematical & Loss Mechanics

### 4.1 Quantile levels and pinball loss

The model predicts 21 non-crossing quantiles **[P §4.1]**:

$$\mathcal{Q} = \{0.01, 0.05, 0.10, 0.15, \dots, 0.85, 0.90, 0.95, 0.99\}$$

For a normalised target $z$ and predicted quantile $\hat{q}_\tau$ at level $\tau \in \mathcal{Q}$ **[P eq. 2]**:

$$\rho_\tau(e) = \max\big(\tau e,\ (\tau - 1)e\big), \qquad e = z - \hat{q}_\tau$$

### 4.2 Scored positions and batch loss

Let $i$ index (row, horizon time step) pairs in a batch, and let $w_i \in \{0, 1\}$ with $w_i = 1$ if and only if all of the following hold:

- the row is a **target** (not a covariate);
- the step lies in the **requested forecast horizon** (not in the padding that rounds the horizon up to a whole patch);
- the value is **observed**.

The loss averages equally over quantile levels and scored positions **[P eq. 3]**:

$$\mathcal{L} = \frac{1}{|\mathcal{Q}| \sum_i w_i} \sum_i w_i \sum_{\tau \in \mathcal{Q}} \rho_\tau\big(z_i - \hat{q}_{\tau, i}\big)$$

A batch with $\sum_i w_i = 0$ is skipped rather than divided by an epsilon **[D]**. Context representations and covariates receive no direct loss; they are trained through their influence on scored predictions **[P §4.1]**.

### 4.3 Normalised space

Targets and predictions are compared in the model's normalised space **[P §4.1]**. For a step in horizon patch $j$ of a block (with $j = 0$ the first):

$$z = \operatorname{asinh}\!\left(\frac{y - \mu_j}{\sigma_j}\right)$$

Because this map is increasing for a fixed frame $(\mu_j, \sigma_j)$, quantiles of $z$ map one-to-one onto quantiles of $y$. Raw-space forecasts are $\sinh(\hat{q}) \cdot \sigma_j + \mu_j$.

**The frame $(\mu_j, \sigma_j)$:**

- **First patch ($j = 0$).** Mean and standard deviation of all observed context values up to the end of the last context patch **[P §3.3]**, **[U model.py:227-228]**. Settled.
- **Later patches ($j \ge 1$).** The paper standardises each forecast patch with all values up to the end of the preceding patch. Inside a block those include earlier forecast patches, "which are unknown at inference and are therefore replaced by their predicted medians. The same statistics are used during training to standardise the targets, so that training and inference match" **[P §3.3]**.

We read that as: training folds the model's own predicted medians into the statistics. The upstream forecaster always denormalises later patches this way, in `_refine_block` and `advancing_block_frames` **[U forecaster.py:458-483]**, **[U scaler.py:165-208]**.

**Decision [D]:** train in the **median-folded frame** with stop-gradient on the statistics, reusing `advancing_block_frames` unchanged. Whatever pretraining did, this makes the tuned model's training frame identical to its deployment frame.

**Evidence [V].** The released checkpoints are calibrated in this frame. On synthetic series with short contexts and a 16-patch horizon, the 5M and 28M models' 80% interval covers 76–85% of targets normalised in the median-folded frame on horizon patches 1–15, against 88–100% of targets normalised in the true-values frame. Patch 0, which is identical under both frames, sits at 80–83%. A model trained against true-values targets would be calibrated under that frame instead of over-covering, so pretraining is consistent with the median-folded frame and not with the other. This supports the reading above by measurement, not only by the paper's wording.

Limits of that evidence: synthetic, univariate, fully observed series; one seed; two model sizes. Under the median-folded frame the 50% interval under-covers at long lead times (0.40–0.44 on patches 4–15). Numbers and the probe are in `docs/VERIFICATION.md`.

An alternative frame, `true_values` (statistics from the true series up to the preceding patch), is available as `target_frame="true_values"` for ablation. Pinball losses are not comparable between the two frames: each is measured in its own normalised units, and true-values frames have larger scales on drifting series, so they score lower whatever the model predicts.

Reported metrics (§5.5) are always computed in raw space.

---

## 5. Adaptation Modes

| Mode | Trainable parameters | Primary use case | Overfitting risk |
|---|---|---|---|
| `conformal` | None by gradient; a small table of offsets is fitted | Low-data regimes, real-time risk buffers | Very low |
| `head_only` | `head.projection` only (15.2% of 5M, about 2.4% of 209M) | Distribution-shape shifts, e.g. financial return tails | Low; moderate on 5M, where the head is a large share |
| `lora_variate` | LoRA on variate attention, `r = 4..16` | Cross-series dependency changes while temporal priors stay frozen | Low to moderate |
| `lora_temporal` | LoRA on temporal attention, `r = 4..16` | Seasonality and regime adaptation | Moderate |
| `full_ft` | All weights | Large datasets (1,000+ series) | High; pair with replay (§5.3) |

Modes are presets over one mechanism: a set of trainable parameter groups plus optional LoRA wrappers **[D]**.

### 5.1 LoRA design

Because query, key and value share one fused `wQKV` linear (§3.2), name-based injection through a generic library cannot target Q and V separately. The library ships a small custom adapter **[D]**.

- **Slices.** `LoraConfig.targets` selects from `"q"`, `"k"`, `"v"` (row slices of `wQKV`), `"o"` (`wO`) and `"gate"` (`_gate`). `LoraConfig.axes` selects `"time"`, `"variate"`, or both, by reading each module's `is_time`.
- **Update.** For each selected slice, $\Delta W = \frac{\alpha}{r} B A$ with $A \in \mathbb{R}^{r \times D}$ Gaussian-initialised and $B \in \mathbb{R}^{D \times r}$ zero-initialised, so the adapted model equals the base model at step 0 (LoRA paper, eq. 3 and §4.1).
- **Size.** Trainable parameters are `modules × slices × 2·r·D`. For 28M with `r=8` on Q and V: 36,864 for `lora_variate` (3 modules) and 147,456 for `lora_temporal` (12 modules).
- **Merge.** `merge()` adds $\Delta W$ into the slice rows in place; export writes merged weights under upstream parameter names (§7.4).

**Single-row guard.** Variate attention attends across the rows of one series. When a series has a single row, there is one key, the softmax is identically 1, and the query and key projections have no effect on the output. Verified: perturbing them on a univariate input changed the forecast by exactly 0, while perturbing the value projection changed it **[V]**. Therefore:

- `lora_variate` raises a configuration error if every series in the dataset has exactly one row and no covariates.
- It can be overridden with `allow_single_row=True`, in which case only `"v"`, `"o"` and `"gate"` are permitted, and the docs state that this is a per-token transform, not a cross-series one.

### 5.2 Conformal calibration

Conformal calibration adjusts the final forecast quantiles after all inference steps (including the forecaster's symmetric averaging and clipping). It never touches model weights.

**Level pairs.** The 21 levels form ten symmetric pairs $(\tau, 1 - \tau)$ around the median, giving nominal central coverages of 98%, 90%, 80%, 70%, …, 10%. A requested coverage must be one of these; the calibrator raises otherwise.

**Static method: Conformalized Quantile Regression** (`fit` / `transform`). For a pair with miscoverage $\alpha = 2\tau$ and a calibration set of size $n$:

$$E_i = \max\big(\hat{q}_{\tau}(x_i) - y_i,\ \ y_i - \hat{q}_{1-\tau}(x_i)\big)$$

$$Q = \text{the } (1 - \alpha)(1 + 1/n)\text{-th empirical quantile of } \{E_i\}$$

$$C(x) = \big[\hat{q}_{\tau}(x) - Q,\ \ \hat{q}_{1-\tau}(x) + Q\big]$$

(CQR paper, eqs. 9–11.) $Q$ may be negative, which narrows an over-wide interval.

Design choices **[D]**:

- **Horizon buckets.** Forecast error grows with lead time, so offsets are fitted per horizon bucket (default: one per patch of 32 steps; configurable down to one per step).
- **Score scaling.** Scores are divided by the context standard deviation of each series before pooling, and multiplied back at transform time, so that series with different scales share one calibration set.
- **Monotonicity repair.** Independent offsets per pair can break nesting or cross the median. The median is never shifted. After applying offsets, levels are repaired outward from it: each upper level is raised to at least the level inside it, and each lower level is lowered to at most the level inside it.
- **Calibration data.** Must be disjoint from training data and later in time than it.

**Online method: Adaptive Conformal Inference** (`update` / `transform`). The working miscoverage level is adjusted after each realised observation:

$$\alpha_{t+1} = \alpha_t + \gamma\,(\alpha - \mathrm{err}_t), \qquad \mathrm{err}_t = \mathbb{1}\{y_t \notin C_t(\alpha_t)\}$$

(ACI paper, eq. 2.) The offset at time $t$ is the $(1 - \alpha_t)$ quantile of a rolling window of recent scores. For multi-step forecasts, the error at lead time $h$ is only known $h$ steps later, so each horizon bucket keeps its own $\alpha_t$ and updates when its observation arrives.

The recursion can push $\alpha_t$ slightly outside $[0, 1]$; the ACI paper bounds it to $[-\gamma, 1 + \gamma]$ (Lemma 4.1). $\alpha_t$ itself is never clipped, so the recursion stays as published. When $\alpha_t \le 0$ the calibrator uses the largest score in the window and sets a `saturated` flag (the paper's interval would be infinite); when $\alpha_t \ge 1$ it uses the smallest score.

**What each method guarantees.**

- CQR guarantees marginal coverage of at least $1 - \alpha$ when calibration and test points are exchangeable (CQR paper, Theorem 1). Time series generally violate this, and overlapping forecast windows are dependent, so the guarantee is approximate here. The library reports empirical coverage on a held-out period rather than asserting the nominal level.
- ACI makes no distributional assumption; its long-run average miscoverage converges to $\alpha$, with the gap bounded by $(\max\{\alpha_1, 1 - \alpha_1\} + \gamma) / (T\gamma)$ after $T$ steps (ACI paper, Proposition 4.1). It guarantees nothing for any single period.
- After the monotonicity repair, intervals can only widen, so a repaired pair covers at least as often as the unrepaired one.

### 5.3 Anti-forgetting: replay and distillation

Stretch scope, Phase 4. All of this is our design **[D]**; Mistral has not released its synthetic corpus, so "replay" here cannot mean replaying Mistral's pretraining data.

- **Replay mixing.** A fraction `replay_fraction` of each batch is drawn from a replay source: a user-supplied dataset, or simple built-in generators (trend plus seasonality plus noise, random walks, autoregressive processes). The built-in generators are a regulariser and make no claim to match Mistral's priors.
- **Distillation to the frozen base.** On replay samples, an extra term penalises the mean absolute difference between the tuned model's and the frozen base model's normalised quantiles, weighted by `distill_weight`. This needs no labels.
- **Forgetting metric.** Every evaluation reports the change in pinball loss and MASE, tuned versus zero-shot, on a held-out general set that is not from the target domain. This runs in all modes from Phase 3 onward, independent of whether replay is enabled.
- **Sign-flip augmentation.** The upstream forecaster averages each forecast with the reflected forecast of the negated series by default **[U forecaster.py:221-227]**. Fine-tuning on one-signed data can make the two passes disagree. Randomly negating whole series (targets and covariates together) during training keeps them coherent. Off by default; evaluated in Phase 3.

### 5.4 Trainer

- **Forward pass.** The trainer never calls `forecast`. It builds the same inputs as upstream's `_assemble_rows` **[U forecaster.py:332-394]**, runs the two-stream transformer, and takes normalised quantiles at the horizon prediction tokens of target rows.
- **Optimiser.** AdamW with cosine decay and linear warm-up. Suggested starting learning rates: `3e-4` for LoRA, `1e-4` for `head_only`, `1e-5` for `full_ft`. These are starting points only **[D]**; the paper publishes none.
- **Precision.** `bf16` autocast on CUDA; `fp32` on CPU and on the Apple GPU. The head already promotes to at least fp32 **[U model.py:96]** and statistics run in float64 without gradients **[U scaler.py:28-29]**. The paper reports bf16 inference within 0.2% of fp32 **[P §4.3]**.
- **Compilation.** `torch.compile` is optional. `mode="reduce-overhead"` is only used when batches are bucketed to fixed shapes.
- **Gradient clipping** at 1.0 by default.
- **Dropout.** The base model has none **[P §3.3]**. LoRA dropout applies to the adapter path only.
- **CUDA.** Verified **[V]** on an NVIDIA L4 (PyTorch 2.15.0, fp32): forecasts match CPU to 1e-7 to 1.5e-5 relative, T14 and the whole suite pass, and a full training step is 6× to 28× faster than that VM's 4 CPU threads (context 1,024, horizon 96, batch 32). Float32 gradients are less accurate than on the other devices (see §12). The environment needs a C compiler and Python headers (§12).
- **Apple GPU (MPS).** Supported in v0.1 by running the four float64 statistics sites of §3.5 on CPU and moving their float32 results to the device **[D]**. The statistics carry no gradients, so this changes nothing in the maths. Measured **[V]**:
  - *Forward parity with CPU:* relative difference of 2e-7 to 3e-6 for `predict_block` in both layouts, 1e-7 for a 96-step forecast, up to 2e-5 for a 2,100-step block-autoregressive forecast, and 1e-7 with numeric and categorical covariates.
  - *Gradients of the training loss:* within 2e-7 to 1.1e-3 of a float64-weights reference across the T14 scenarios, against 2e-7 to 3.4e-3 for CPU float32. §12 explains why released weights are not at round-off on either device.
  - *Speed of the full training step* (normalised forward, median-folded frame with its CPU round trip, pinball loss, AdamW; context 1,024, horizon 96, batch 32): between 1.4× and 3.2× faster than CPU across four runs, for both univariate batches and batches with five rows per series. Timings vary from run to run.
  - *Memory:* about 7 GiB allocated for 28M with five rows per series at batch 32.
  - *Mixed precision:* no speed-up from `bf16` or `fp16` autocast, and `bf16` moved outputs by 1e-2 to 6e-2, so MPS runs fp32 only and `mixed_precision` other than `"fp32"` raises there.
  - *Requirements and scope:* PyTorch 2.8 or newer. Older versions lack the `cummax` operator on this backend, which upstream's input assembly uses; `resolve_device` skips the Apple GPU under `"auto"` and raises for an explicit `"mps"`. Verified on 2.8.0 and 2.15.0, on one machine. 123M, 209M and contexts above 4,096 steps are untested on MPS.
- **Device shims.** The offload is implemented as subclasses outside `_vendor/` that move tensors to CPU and call the parent: a model subclass overriding `scale`, and a forecaster subclass overriding `_cap_quantiles`, `_refine_block` and `_encode_categorical`. The training forward calls `advancing_block_frames` on CPU directly. No vendored file is edited for this (§6.1).
- **Memory.** With covariates the token count is `2n` per row (§3.4), about double the no-covariate layout.

### 5.5 Evaluation

All reported forecasts go through the vendored upstream `TimestralForecaster` with its defaults (symmetric averaging and non-negative clipping on), so tuned and zero-shot numbers are produced by the same code path.

Metrics, in raw space: MASE, mean weighted quantile loss over the 21 levels, pinball loss, empirical coverage and mean width of the 80% and 90% intervals, and the forgetting metric of §5.3.

---

## 6. Package Architecture

```
src/timestral_tune/
│
├── __init__.py
├── config.py                 # TrainingConfig (built); LoraConfig, ReplayConfig (later phases)
├── _vendor/
│   └── timestral/            # Upstream code at the pinned commit (imports made relative), LICENSE, manifest
│
├── data/
│   ├── batch.py              # TrainingBatch: the packed row layout shared by collator, forward and loss
│   ├── dataset.py            # Long-frame loader (Polars / PyArrow / pandas), windowing, time splits
│   ├── collator.py           # Packed rows, roles, group ids, padding, loss weights
│   └── encoders.py           # Per-window, context-only target encoding of categorical covariates
│
├── models/
│   ├── loader.py             # Checkpoint loading at a pinned Hub revision
│   ├── device.py             # Subclasses that run float64 statistics on CPU (needed for MPS)
│   ├── forward.py            # Training forward: normalised quantiles and target frames
│   ├── adapter.py            # Slice LoRA for wQKV / wO / gate; merge and unmerge
│   └── conformal.py          # Static (CQR) and online (ACI) calibrators
│
├── losses/
│   └── pinball.py            # 21-quantile pinball loss with weight-mask reduction
│
├── replay/
│   ├── generators.py         # Simple synthetic generators
│   └── distill.py            # Frozen-base distillation term
│
├── trainer/
│   ├── step.py               # compute_loss: forward, target frame and pinball loss in one place
│   ├── trainer.py            # Training loop (clipping, evaluation, early stopping; AMP and compile in Phase 2)
│   └── callbacks.py          # Progress logging, early stopping
│
├── export.py                 # Upstream-loadable checkpoint, adapter, metadata sidecar
│
├── integrations/
│   └── lacuna.py             # Optional: trial records and signal frames (Python 3.11+)
│
└── utils/
    ├── metrics.py            # MASE, weighted quantile loss, pinball, coverage, width
    └── validation.py         # Parity, leakage and loss-mask verifiers (§9)
```

### 6.1 Dependency on upstream: vendored **[D]**

`timestral` is not on PyPI, and PyPI rejects packages that depend on a git URL. The PyPI name `timestral` is also unclaimed, so a bare requirement on it would be a dependency-confusion risk. Training additionally needs upstream's private functions, which a version pin cannot protect.

- `scripts/vendor_upstream.py` copies the upstream package into `src/timestral_tune/_vendor/timestral/` at the pinned commit, with its Apache 2.0 `LICENSE`. The repository `NOTICE` names the source and commit.
- **One mechanical modification is made:** upstream's files import each other absolutely (`from timestral.config import …`), which would fail, or silently resolve to an installed official `timestral`. The script rewrites these to relative imports and adds a notice header to each of the five files it changes. The other four source files are byte-identical. A manifest records the pristine and vendored hash of every file.
- All new logic lives outside `_vendor/` and imports from it. Vendored files are never edited by hand; a test compares them with the manifest.
- A scheduled CI job runs `scripts/check_upstream_drift.py`, which compares upstream `main` with the manifest's pristine hashes and fails on drift.
- Parameter names are never changed, so exported weights load in the official package with `strict=True` **[U loading.py]**.

### 6.2 Data contract

- **Input.** A long-format frame with one row per (series, timestamp), an id column and a time column. Each series must be regularly sampled; gaps become missing values.
- **Roles.** `target_cols`, `past_covariates`, `known_covariates`, and their categorical counterparts. The name `known_covariates` follows upstream.
- **Windows.** A window is a forecast origin plus up to `context_length` steps before it and `forecast_horizon` steps from it. Origins are anchored at the end of each series and step back by `stride`, so the most recent data is always used. A window is dropped, and counted, when no target is observed in its context or in its horizon.
- **Shape rules.** The context is truncated to at most 32,768 steps and left-padded to a multiple of 32 with unobserved zeros **[U forecaster.py:70-77]**. The horizon is rounded up to whole patches; padded steps get zero loss weight. `forecast_horizon` must be at most 1,024 for training.
- **Preprocessing parity.** Target rows are forward-filled with the observation mask preserved; covariate NaNs become unobserved zeros. This mirrors `_assemble_rows` exactly.
- **Storage.** Frames from polars, pandas or pyarrow are converted once to contiguous float32 arrays per series; none of the three is a runtime dependency. Windows are views into those arrays, and the collator makes the padded copies.
- **Validation.** Null ids or timestamps, duplicate (series, timestamp) pairs, timestamps off the regular grid, non-numeric or infinite values and time-zone-aware timestamps raise. `regularize=False` treats consecutive rows as consecutive steps, for calendar-irregular data such as monthly series.

### 6.3 Point-in-time rules

1. Target values after the forecast origin are labels only. They are never model inputs.
2. Past-only covariates are cut at the origin. The user is responsible for supplying values as they were known at each timestamp; a per-column `available_lag` shifts a covariate that is published late.
3. Known-future covariates must be genuinely known in advance over the horizon.
4. Categorical covariates are target-encoded **per window, using the context only**, with horizon codes looked up in the context's table. This matches upstream inference exactly **[U covariate_encoding.py]**, **[U forecaster.py:246-276]**. Series with several target variates have no single target to encode against, so the category codes go to the model as numbers, as upstream does **[U forecaster.py:264-267]**. The numbering must then not depend on the future either: each window numbers its categories by rank among the categories it can see (context for past-only covariates, context and horizon for known-future ones), so a category that first occurs after the origin or after a split cutoff cannot change a window's inputs **[D]**. `window_category_codes` is that rule; apply it to a window's codes before calling upstream's `forecast` on a model trained this way.
5. `split_by_time(cutoff, embargo)` gives training windows nothing at or after the cutoff, context and horizon alike. Validation origins start `embargo` steps after it, and windows that straddle the cutoff are dropped. No validation label can appear in a training window even with `embargo=0`; the embargo adds a gap for strongly dependent data.

---

## 7. Public API Spec (Draft, library v0.1)

### 7.1 Basic training interface

```python
from timestral_tune import TimestralTrainer, TrainingConfig, LoraConfig
from timestral_tune.data import TimeSeriesDataset

# 1. Load a long-format frame with explicit covariate roles
dataset = TimeSeriesDataset.from_polars(
    df,
    id_col="symbol",
    time_col="timestamp",
    target_cols=["realized_vol"],
    past_covariates=["volume", "spread"],
    known_covariates=["is_fomc_day", "hour_of_day"],
    context_length=1024,      # 32 patches
    forecast_horizon=96,      # 3 patches
    stride=96,
)
train_ds, val_ds = dataset.split_by_time(cutoff="2025-01-01", embargo=96)

# 2. Configure selective LoRA on variate attention.
#    Each series here has 5 rows (1 target + 4 covariates), so variate attention is live.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    axes=("variate",),
    targets=("q", "v"),
    dropout=0.05,
)

train_config = TrainingConfig(
    model_name_or_path="mistralai/Timestral-28M",
    revision="49b26409ac87b5935539a5d8ffe7f397b720db47",
    mode="lora_variate",
    learning_rate=3e-4,
    batch_size=32,
    epochs=5,
    mixed_precision="bf16",
    gradient_clip_val=1.0,
    target_frame="median_folded",
    output_dir="./checkpoints/timestral-28m-vol",
)

# 3. Train and export
trainer = TimestralTrainer(
    config=train_config,
    lora_config=lora_config,
    train_dataset=train_ds,
    val_dataset=val_ds,
)
trainer.fit()
trainer.export()
```

### 7.2 Using the result with the official package

```python
from timestral import TimestralForecaster   # the official inference package

forecaster = TimestralForecaster.from_pretrained("./checkpoints/timestral-28m-vol")
forecast = forecaster.forecast(
    context,
    prediction_length=96,
    past_covariates=past,
    known_covariates=known,
)
```

On a Mac, the official package itself only runs on CPU, which is its default device; passing `device="mps"` to it fails **[V]**. For Apple GPU inference, load the export through `timestral_tune.models.load_forecaster(path, device="mps")`, which returns the shimmed forecaster of §5.4.

### 7.3 Conformal recalibration (zero-gradient baseline)

```python
from timestral_tune.models import ConformalCalibrator, OnlineConformalCalibrator

# Static: fit offsets on a calibration period, apply to later forecasts
calibrator = ConformalCalibrator(target_coverage=0.80)   # the (0.10, 0.90) pair; "all" calibrates all ten
calibrator.fit(predictions=val_preds, actuals=val_targets, context_scale=val_scale)
calibrated_quantiles = calibrator.transform(test_preds, context_scale=test_scale)

# Online: adapt as observations arrive
online = OnlineConformalCalibrator(target_coverage=0.80, gamma=0.005, window=500)
calibrated = online.transform(new_preds, context_scale=new_scale)
online.update(predictions=new_preds, actuals=realised, context_scale=new_scale)
```

`predictions` are `(N, H, 21)` quantile arrays, `actuals` are `(N, H)`, and `context_scale` is `(N,)`.

### 7.4 Export layout

```
checkpoints/timestral-28m-vol/
├── config.json            # Upstream schema, byte-compatible; no extra keys
├── model.safetensors      # Merged weights under upstream parameter names
├── adapter.safetensors    # LoRA modes only: the unmerged A and B matrices
├── adapter_config.json    # LoRA modes only
├── conformal.json         # If a calibrator was fitted
├── tune_metadata.json     # Base repo and revision, mode, hyperparameters, data fingerprint,
│                          # metrics, library version, vendored upstream commit
├── LICENSE                # Mistral's Modified MIT License, copied from the base repo
└── NOTICE                 # States that the weights are a derivative of the named base model
```

### 7.5 Optional Lacuna integration

Installed with `pip install "timestral-tune[lacuna]"`; requires Python 3.11+ and `lacuna-quant>=0.14.1`.

Lacuna consumes signals, prices, returns and experiment history, and itself produces the audit bundle (`report.bundle(...)`). We found no forecast or quantile ingestion in 0.14.1. So this integration writes Lacuna's inputs; it does not write bundles.

- **Trial records.** Each training or evaluation run is appended to a `lacuna.ExperimentRegistry` with `record(parameters=..., metric=..., metric_name=..., data_fingerprint=..., code_fingerprint=..., result_fingerprint=..., metadata=...)`. Model selection is logged with `record_selection(eligible_trial_ids=..., selected_trial_ids=..., metric=..., direction="minimize", used_holdout=...)`. This gives Lacuna the full candidate set for multiple-testing corrections.
- **Signal frames.** A helper writes a long frame with columns `time`, `instrument`, `signal`, which are the default column names of `lacuna.SignalStudy`. The signal is a user-chosen function of the forecast, for example the median or an interquantile spread. The user supplies prices to Lacuna.
- **Deflated Sharpe Ratio.** This needs strategy returns, which a forecaster does not produce. It applies only when the user builds a strategy on the signal and gives Lacuna its returns. `timestral-tune` computes no Sharpe ratios.

---

## 8. License & Compliance Guidelines

- **Code.** Published under **Apache 2.0**. The vendored upstream code is also Apache 2.0 and keeps its licence and a NOTICE.
- **Weights.** Downloaded from Hugging Face at runtime (`mistralai/Timestral-*`) and never re-hosted. They are under Mistral AI's **Modified MIT License**, which has two conditions:
  1. The attribution notice and permission notice must accompany all copies or substantial portions of the model.
  2. **No rights are granted to anyone whose company (or employer) had global consolidated revenue above $20 million in the preceding month.** The licence states that this restriction "applies to the Model and any derivatives, modifications, or combined works based on it". Above the threshold, a commercial licence from Mistral is needed.
- **Fine-tuned outputs are derivatives.** Merged checkpoints fall under clause 2. Adapter files should be treated the same way. Every export therefore includes the licence and a NOTICE (§7.4), and the README says so plainly. This matters for the finance audience this library targets. This is a summary, not legal advice.
- **Independence.** The README and repository root carry a disclaimer: this is an independent community project, not affiliated with or endorsed by Mistral AI; "Timestral" and the base architecture and weights are Mistral AI's.
- **Reference papers.** Only the Timestral paper (CC BY 4.0) is tracked in `papers/`. Other PDFs there are local reading copies under arXiv's non-redistribution licence and are git-ignored.
- **Benchmark data.** Use only datasets whose licences allow publishing derived metrics.

---

## 9. Acceptance Tests

Placeholders for all fourteen tests are in `tests/acceptance/`, skipped until implemented. A skipped placeholder is not a pass.

These replace the original "assert zero gradients on non-target covariate positions" test, which was mis-specified: a target-only loss produces **non-zero** gradients on covariate inputs **[V]**, and that is intended, since targets read covariates.

| # | Test | Passes when |
|---|---|---|
| T1 | Forward parity, no covariates | `sinh(training_forward) · σ₀ + μ₀` equals `predict_block` output within 1e-6 relative, i.e. float32 round-off |
| T2 | Forward parity, with covariates | Same, for the `2n`-token layout with all three roles |
| T3 | Frame parity | Applying the median-folded frame reproduces `_refine_block`, and the full pipeline matches `forecast(symmetric_averaging=False, clip_nonnegative=False)` on a forecaster built with `quantile_real_cap_k=0`, which disables its output clamp |
| T4 | Label isolation | Replacing horizon target values with noise leaves predictions bit-identical |
| T5 | Past-covariate cut | Changing a past-only covariate after the origin (in the raw frame) changes nothing |
| T6 | Firewall | Noise on a target's context changes no covariate row's hidden state or output |
| T7 | Positive control | Noise on a known-future covariate in the horizon does change the target forecast |
| T8 | Loss mask | The gradient of the loss with respect to predictions is exactly zero at covariate rows, unobserved steps and horizon padding |
| T9 | LoRA identity | A freshly initialised adapter leaves outputs unchanged; merged and unmerged outputs agree within tolerance |
| T10 | Single-row guard | `lora_variate` on single-row data raises; with the override, query and key slices are rejected |
| T11 | Export round-trip | The exported directory loads in the vendored upstream loader with `strict=True` and reproduces the trainer's forecasts |
| T12 | Encoding causality | A window's categorical encodings are unchanged when data after its origin changes |
| T13 | Conformal | On synthetic exchangeable data, calibrated coverage meets the nominal level; output quantiles are non-decreasing |
| T14 | Device parity | On each available accelerator, forecasts match CPU within 1e-4 relative. On every device including CPU, float32 training-loss gradients match a float64-weights reference in three tiers: within 1e-5 relative on a randomly initialised model; within 5e-3 on released checkpoints with attention written out explicitly; within 5e-2 on released checkpoints as shipped (§12 explains the tiers). With everything but the head frozen, head gradients match within 1e-5 on every model. Skipped where a device is absent |

Perturbation tests use non-constant noise. A constant offset is absorbed by the normalisation and would make a firewall test pass vacuously.

---

## 10. Development Roadmap

### Phase 0: Foundations
- [x] Vendor upstream at the pinned commit with LICENSE and NOTICE; drift-check script (run locally: no drift).
- [x] Package skeleton of documented stubs; scaffold tests pass on Python 3.10 and 3.13. The `[lacuna]` extra is declared with a 3.11 marker and resolves in the lockfile, but has not been installed or exercised.
- [x] Verification probes in `scripts/probes/`, with results in `docs/VERIFICATION.md`.
- [x] CI green on Python 3.10–3.13 (first run 2026-11-21, Linux runners).
- [ ] Weekly drift job: enabled, but its first scheduled run has not happened yet.
- [x] Checkpoint loader with pinned Hub revisions; config always read from `config.json`.
- [x] Device shims for the float64 statistics (§5.4); T14 passes on CPU and the Apple GPU, on the tiny model, 5M and 28M.

### Phase 1: Core mechanics & verification
- [x] Training forward for both layouts; T1–T3 pass (agreement with upstream at float32 round-off).
- [x] `PinballLoss` with the weight mask of §4.2; T8 passes under both target frames.
- [x] Dataset, windowing, collator and per-window encoders; T4–T7 and T12 pass. Collated batches equal upstream's own input assembly tensor for tensor.
- [x] Target-frame check on horizons longer than one patch. Result: the checkpoints are calibrated in the median-folded frame, which confirms D2 (§4.3).
- [x] `head_only` training end to end on a toy dataset, on CPU and the Apple GPU, on the tiny model, 5M and 28M: the loss falls and no weight outside the head changes.
- [x] T14's gradient check on the PyTorch 2.4 floor: passes on CPU. The Apple GPU does not work below PyTorch 2.8, which is now its stated floor.
- [x] T14 on CUDA: passes on an NVIDIA L4 with PyTorch 2.15.0, on the tiny model, 5M and 28M, as does the whole suite (`bash scripts/cuda_check.sh`).

**Exit:** T1–T8 and T12 pass on 5M and 28M; the frame decision is confirmed or revised. **Met on 2026-11-21.**

### Phase 2: Adaptation & calibration
- [ ] Slice LoRA adapter, mode presets, single-row guard; tests T9 and T10.
- [ ] `full_ft`.
- [ ] `ConformalCalibrator` and `OnlineConformalCalibrator`; test T13.
- [ ] Export layout and metadata sidecar; test T11.
- [ ] `bf16` on CUDA and optional `torch.compile` with shape bucketing.

**Exit:** all tests T1–T14 pass; an exported LoRA checkpoint loads in the official package.

### Phase 3: Benchmarks & documentation
- [ ] Benchmark **zero-shot vs. conformal vs. head-only vs. LoRA (variate, temporal) vs. full FT** on a realised-volatility dataset and a retail-demand dataset, across at least two model sizes.
- [ ] Report the forgetting metric for every mode.
- [ ] Learning-rate and rank sweeps to replace the placeholder defaults of §5.4.
- [ ] Evaluate sign-flip augmentation.
- [ ] Lacuna integration: trial records and signal frames.
- [ ] Documentation, licence notices, release on PyPI.

**Exit:** published tables with code to reproduce them; each default in §5.4 is backed by a sweep.

### Phase 4: Anti-forgetting and denser supervision (stretch)
- [ ] Replay mixing, built-in generators, and base-model distillation.
- [ ] Tiled training with a fixed block length: the upstream builder already takes `min_context` and `block_size` **[U masks.py:56-101]**, so one forward pass can supervise every patch after a minimum context. This needs its own target-frame definition and parity tests.
- [ ] Sampled block lengths within a sequence, as in the paper **[P §3.1]**. This needs a new mask builder; the paper does not publish the block-length distribution.
- [ ] Apple GPU coverage for 123M, 209M and long contexts.

---

## 11. Decisions Log

| # | Decision | Why |
|---|---|---|
| D1 | Default training scheme is a single context-plus-horizon split that mirrors upstream inference | Provably equal to deployment (T1–T3); keeps per-window encodings identical to inference; the paper's block-length distribution is unpublished |
| D2 | Train in the median-folded target frame, stop-gradient | Matches the frame the upstream forecaster always uses; our reading of **[P §3.3]**; confirmed by calibration measurements on 5M and 28M (§4.3) |
| D3 | Vendor upstream at a pinned commit | Not on PyPI; training needs private functions; avoids a dependency-confusion risk |
| D4 | Custom slice LoRA on the fused `wQKV` | No separate Q or V modules exist |
| D5 | `lora_variate` refuses single-row data by default | Query and key adapters are provably inert there |
| D6 | CQR as the static conformal method, ACI as the online one | CQR works directly on the model's quantile pairs; ACI needs no exchangeability |
| D7 | Replay uses user data or simple generators, with optional distillation | Mistral's corpus is unreleased |
| D8 | Lacuna integration is optional and writes trial records and signal frames only | Lacuna consumes those and emits its own bundle; it needs Python 3.11+ |
| D9 | Apple GPU supported in v0.1, fp32 only, PyTorch 2.8+, by running float64 statistics on CPU in subclasses outside the vendored code | Measured parity with CPU and a 1.4–3.2× faster training step; mixed precision gave no gain there |
| D10 | Exports are merged, upstream-loadable, with a sidecar for metadata | `config.json` rejects unknown keys; users can deploy with the official package |
| D11 | `split_by_time` gives training nothing at or after the cutoff; the embargo is an optional extra gap | Stronger and simpler than relying on an embargo of one horizon to separate labels |
| D12 | T14's gradient criterion has three tiers on every device: 1e-5 on a randomly initialised model; 5e-3 on released checkpoints with explicit attention; 5e-2 on released checkpoints as shipped | Float32 gradients through the released weights are off by up to 3.4e-3 on CPU, 1.1e-3 on the Apple GPU and 1.2e-2 on CUDA, while a random model is at 3e-7 everywhere. The middle tier shows the library's own code is accurate on real weights; the loose tier only guards against a wrong gradient. A separate `head_only` check holds 1e-5 on every model, since that mode's backward pass never enters attention. Revised on 2026-11-21 after the first CUDA run: the as-shipped bound was 1e-2, which CUDA exceeds on some batches |
| D13 | Phase 1 trains `head_only` in fp32 only; other modes, `bf16` and `export()` raise `NotImplementedError` | No interim behaviour to unlearn later; each arrives with its own acceptance tests in Phase 2 |
| D14 | Category codes are numbered per window, among the categories the window can see | Numbering them once over the whole frame let a category that first occurs later renumber earlier windows' inputs when several targets are forecast together (found in review, 2026-11-21). Per-window ranks depend on nothing outside the window, keep the order of ordered categories, and need no stored mapping at inference |

## 12. Open Questions & Risks

- **Target frame (D2).** Confirmed by calibration measurements on synthetic data for 5M and 28M (§4.3). Still not confirmed by Mistral, and not checked on real data, with covariates, or on 123M and 209M.
- **Hyperparameters.** No published optimiser settings exist. All defaults are provisional until the Phase 3 sweeps.
- **Benchmark contamination.** The paper says the models were post-trained on "a mixture of real and synthetic data" and does not list the real datasets. Any public benchmark dataset may have been seen. Results must say so, and should favour data published after October 2026 or private data where possible.
- **Upstream drift.** Upstream is a first release with no stability promise. The drift-check job detects changes; re-vendoring requires re-running T1–T3 and T11.
- **Benchmark dataset choice.** The realised-volatility and retail-demand datasets are not yet selected; licences must be checked first.
- **Float32 gradients through the released weights are not at round-off.** Against a float64-weights reference (upstream's RMSNorm and RoPE still compute in float32 internally), float32 gradients of the training loss differ by up to 3.4e-3 on CPU, 1.1e-3 on the Apple GPU and 1.2e-2 on CUDA **[V]**, while forward outputs agree to about 3e-7. It depends on the weights, not the architecture: a randomly initialised model is at 3e-7 on every device. On CUDA and CPU most of it is the float32 backward pass of upstream's `F.scaled_dot_product_attention`: with attention written out as matmul and softmax the error falls to 1.1e-4 or less on CUDA (from 1.2e-2) and on CPU, and float64 weights on CUDA give 1e-5; TF32 and Triton's FP fusion make no difference. On the Apple GPU the same swap changes nothing. The same call is accurate in isolation at every attention-logit scale tried, so what about these weights triggers it is not known. One scenario (28M with covariates) keeps a smaller unexplained error with explicit attention: 1.1e-4 on CUDA, 5.4e-4 on CPU, 1.2e-3 on the Apple GPU; it is the Apple GPU's worst case either way. **`head_only` training is not affected:** with the body frozen the backward pass never enters attention, and head gradients match the reference within 2.6e-6 on CPU and the Apple GPU (not yet measured on CUDA). The inaccuracy therefore matters from Phase 2 on, when attention weights are trained. The gradient direction is nearly unaffected (one minus the cosine is at most 2.7e-5, on CUDA), so training should be too; that has not been tested by training both ways. The library trains with upstream's attention call unchanged. `scripts/probes/gradient_accuracy.py` reproduces all of this.
- **Validation loss under `median_folded` is measured in units that move with the model.** From the second horizon patch on, the frame is built from the model's own predicted medians, so two checkpoints are scored against slightly different normalised targets. Early stopping and best-state restoration compare those losses across steps. The effect shrinks with context length (the frame is dominated by the context) and is absent for horizons of one patch, but it has not been measured. Until the raw-space metrics of §5.5 exist (Phase 3), treat the validation loss as a training signal, not as a model-selection metric across runs.
- **Apple GPU scope.** Needs PyTorch 2.8 or newer. Verified on one machine, on 2.8.0 and 2.15.0.
- **Worker processes on macOS with PyTorch 2.4.0.** A DataLoader with `num_workers > 0` leaves the process hanging at exit there **[V]**. It is a PyTorch issue that reproduces without this library and is gone by 2.8.0; `num_workers` defaults to 0.
- **CUDA scope.** Verified once, on one NVIDIA L4 with PyTorch 2.15.0 (CUDA 13.2) on Python 3.12, in fp32. `bf16`, multi-GPU, older GPUs and other PyTorch versions are untested. On CUDA, recent PyTorch compiles a small C helper the first time some operations run (through its Triton component), which needs a C compiler and the interpreter's `Python.h` **[V]**; without them every CUDA forward fails with a compiler error. A uv-managed Python ships the headers.
- **Licence reach.** Whether adapter-only files count as derivatives is a legal question. The library takes the cautious position (§8).

## 13. References

- Timestral. Mistral AI, 2026. arXiv:2610.37585. `papers/timestral_arxiv_2610.37585v1.pdf`
- Official code: https://github.com/mistralai/timestral
- Weights: https://huggingface.co/collections/mistralai/timestral
- Announcement: https://mistral.ai/news/timestral
- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685
- Romano, Patterson, Candès. Conformalized Quantile Regression. arXiv:1905.03222
- Gibbs, Candès. Adaptive Conformal Inference Under Distribution Shift. arXiv:2106.00170
- Lacuna: `lacuna-quant` 0.14.1
