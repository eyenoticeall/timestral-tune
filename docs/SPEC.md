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

- **Other sizes.** The library does not depend on these four sizes. Every dimension in the table, and the patch size, block size and context length, is read from the checkpoint's `config.json` **[U config.py:42-49]**; the four size names appear in our code only as pinned Hub revisions **[D]**. A checkpoint of another size is supported when it keeps the released architecture: layer kinds `time` and `time_variate`, the `sdpa_sigmoid` gate, and the 21 quantile knots **[H]**. Unknown config keys and unknown or missing parameters are load errors **[U config.py:120-123]**. The vendored config also accepts the layer kinds `variate` and `variate_time`, the `none` gate and other quantile knots **[U config.py:14-15, 70-76]**; such a checkpoint loads, but no released model uses them and no test here trains them. Tested network-free: one non-released shape goes from a local directory through the trainer in all four modes to an export the upstream loader accepts, and the suite's tiny model, a second non-released shape, covers the same steps separately (`docs/VERIFICATION.md` §8). Nothing above 209M has been run, and the default learning rates were swept on 5M to 209M only.
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
- **With covariates** **[U model.py:231-273]**: upstream uses a window of `n = L + h` patches with `n` encoder tokens and `n` prediction tokens (`2n` in total); target and past-only rows hold a blank, unobserved horizon while known-future rows hold their real future values, and only the `h` horizon prediction tokens are read out. In single-horizon training, the leading `L` prediction tokens are inactive (they attend only to themselves and are never read by any other token), so `training_forward` drops them and runs `L + 2h` tokens (`n` encoder tokens plus `h` active horizon prediction tokens) **[D]** (decision D16). Masks come from `_build_covariate_time_mask` (built from upstream's `enc_causal_mask`) and `build_two_stream_variate_mask` **[U masks.py:121-146]**, verified bit-identical to slicing upstream's `build_two_stream_time_mask` **[U masks.py:68-101]**.

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
| `full_ft` | All weights | Large datasets (1,000+ series) | High; lower LR or pair with distillation / real-data replay (§5.3) |

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

(CQR paper, eqs. 9–11.) $Q$ is the exact finite-sample order statistic $\lceil (n + 1)(1 - \alpha) \rceil$ (taking $\max_i E_i$ when the rank exceeds $n$) and may be negative, which narrows an over-wide interval.

Design choices **[D]**:

- **Horizon buckets.** Forecast error grows with lead time, so offsets are fitted per horizon bucket (default: one per patch of 32 steps; configurable down to one per step).
- **Score scaling.** Scores are divided by the context standard deviation of each series before pooling, and multiplied back at transform time, so that series with different scales share one calibration set.
- **Monotonicity repair.** Independent offsets per pair can break nesting or cross the median. The median is never shifted. Calibrated levels are clamped so they do not cross the median and repaired outward relative to each other (so calibrated pairs can only widen against other calibrated pairs); uncalibrated levels are then reconciled into the bounds between adjacent fixed anchors (or outward beyond the outermost anchor) without moving any calibrated level. When all ten pairs are calibrated (`target_coverage="all"`), this reduces identically to pure outward repair; when a single pair such as `0.80` is calibrated, negative offsets still narrow that pair cleanly instead of being pulled back outward by uncalibrated neighbours.
- **Calibration data.** Must be disjoint from training data and later in time than it. `collect_conformal_data(forecaster, dataset)` produces `(predictions, actuals, context_scale)` from a `TimeSeriesDataset` through upstream `TimestralForecaster.forecast()`.

**Online method: Adaptive Conformal Inference** (`update` / `transform`). The working miscoverage level is adjusted after each realised observation:

$$\alpha_{t+1} = \alpha_t + \gamma\,(\alpha - \mathrm{err}_t), \qquad \mathrm{err}_t = \mathbb{1}\{y_t \notin C_t(\alpha_t)\}$$

(ACI paper, eq. 2.) The offset at time $t$ is the $(1 - \alpha_t)$ quantile of a rolling window of the last `window` forecast observations per bucket (default 500 forecasts; serialised `conformal.json` is ~1 MB for one pair or ~10 MB for `"all"` at horizon 96), and $\mathrm{err}_t$ is evaluated against the actual emitted (anchor-repaired) interval. For multi-step forecasts, each horizon bucket keeps its own $\alpha_t$ and updates when its observation arrives; when `bucket_size > 1`, $\mathrm{err}_t \in [0, 1]$ is the mean miscoverage fraction over observed steps in the bucket (**[D]** batch-ACI).

The recursion can push $\alpha_t$ slightly outside $[0, 1]$; the ACI paper proves it stays in $[-\gamma, 1 + \gamma]$ (Lemma 4.1) because the paper emits $(-\infty, +\infty)$ for $\alpha_t \le 0$ and $\emptyset$ for $\alpha_t \ge 1$. Because a finite rolling window substitutes the largest score in the window at $\alpha_t \le 0$ (setting `.saturated = True`) and the smallest score at $\alpha_t \ge 1$, $\alpha_t$ is never clipped at $[0, 1]$ but is bounded at Lemma 4.1's theoretical envelope $[-\gamma, 1 + \gamma]$ (**[D]**, D18) so finite-window extremes cannot cause unbounded wind-up during regime shifts.

**What each method guarantees.**

- CQR guarantees marginal coverage of at least $1 - \alpha$ when calibration and test points are exchangeable (CQR paper, Theorem 1). Time series generally violate this, and overlapping forecast windows are dependent, so the guarantee is approximate here. The library reports empirical coverage on a held-out period rather than asserting the nominal level.
- ACI makes no distributional assumption; its long-run average miscoverage converges to $\alpha$, with the gap bounded by $(\max\{\alpha_1, 1 - \alpha_1\} + \gamma) / (T\gamma)$ after $T$ steps (ACI paper, Proposition 4.1). It guarantees nothing for any single period.
- After the anchor-aware repair (D17), calibrated pairs can only widen relative to each other (so every calibrated pair covers at least as often as its own CQR offset); uncalibrated interior levels may be pulled inward to stay inside a narrowed calibrated pair.

### 5.3 Anti-forgetting: replay and distillation

Implemented in Phase 4 (`ReplayConfig`, `SyntheticReplaySource`, `distillation_loss`). All of this is our design **[D]**; Mistral has not released its synthetic corpus, so "replay" here cannot mean replaying Mistral's pretraining data.

- **Replay mixing.** When `replay_config: ReplayConfig` is passed to `TimestralTrainer`, each training step draws a replay micro-batch from `replay_dataset` (a user-supplied `TimeSeriesDataset`, or a default seeded `SyntheticReplaySource` covering trend+seasonality, random walks, stationary AR(1)/AR(2), regime shifts, and positive seasonal patterns across log-uniform scales) and combines losses as `(1 - replay_fraction) * L_domain + replay_fraction * L_replay + distill_weight * L_distill`. `ReplayConfig()` defaults to light label-free distillation (`replay_fraction=0.0, distill_weight=0.1, smooth_beta=1e-3`), measured across `head_only`, `lora_temporal`, and `full_ft` (`docs/BENCHMARKS.md` §5). Running replay as a separate micro-batch allows heterogeneous covariate layouts and horizons between domain and replay data (decision D21). `replay_dataset` must not be `val_dataset`. Built-in generators are a regulariser only and make no claim to match Mistral's priors.
- **Distillation to the frozen base.** On replay samples, `distillation_loss` penalises the smooth L1 (Huber, controlled by `ReplayConfig.smooth_beta`, default `1e-3`; exact L1 when `0.0`) difference between the tuned model's and the frozen base model's normalised quantiles, weighted by `distill_weight` (decision D21). This needs no target labels: its mask covers all real target rows inside the requested unpadded horizon, excluding covariates, right-padding steps, and shape-bucketed dummy series. **Frozen-base semantics on resume:** for LoRA modes, the frozen base is always the unadapted base model (`disable_lora`), even when resuming from `load_lora`; for `head_only` and `full_ft`, it is the starting `model` passed to `TimestralTrainer`. **Memory:** `head_only` keeps only a frozen `model.head` copy; LoRA uses CPU-cached step-0 base quantiles (up to 256 MiB, dual `+x`/`-x` when `sign_flip` is on) or zero-extra-VRAM `disable_lora` fallback; `full_ft` uses CPU-cached base quantiles within the 256 MiB cap and falls back to a full frozen model copy on the device when `cache_base_outputs=False` or above the cap. **Measured ablation with built-in synthetic replay (`docs/BENCHMARKS.md` §5, `docs/VERIFICATION.md` §15) [V]:** on `head_only` retail, default `distill_weight=0.1` strictly beats the lower-LR `1e-3` control on both wQL and forgetting (`5M` 3/3 seeds: `0.2105 / +2.5%` vs `0.2114 / +3.6%`; `28M`: `0.2079 / +3.6%` vs `0.2090 / +4.2%`); on `5M head_only` volatility (3 seeds), `replay_fraction=0.25` beats plain `1e-3` at matched wQL (`0.2231 / +5.0%` vs `0.2236 / +7.8%`), while on `28M` vol (1 seed) plain `1e-3` gives lower wQL (`0.2169` vs `0.2195–0.2196`). On `lora_temporal=3e-3` (3 seeds), `distill_weight=0.1` eliminates the `+31.3%` forgetting cliff (`+0.6%` worst-seed) and edges out shipped `1e-3` on 3/3 seeds (`0.2178` vs `0.2186`, small margin relative to seed spread). On `full_ft=3e-4` retail (3 seeds), neither synthetic pinball (`+4.1%` at 64 series; `+2.8%` at 256 series) nor `distill_weight=0.1` (`0.1999`, worst `+1.3%`) beats simply lowering the learning rate to `2e-4` (`0.1995`, worst `+1.3%`). Real general-domain replay datasets are untested.
- **Forgetting metric.** Raw-space evaluation (`timestral_tune.utils.evaluate_forgetting` / `compute_forgetting`) reports the absolute and relative change in pinball loss, wQL, and MASE (tuned minus zero-shot) on a held-out general set that is not from the target domain (macro-averaged equally across domains in benchmarks so high-magnitude series do not dominate; decision D20). This runs across all modes from Phase 3 onward, independent of whether replay is enabled; `TimestralTrainer` itself logs training/validation loss only and leaves raw-space target and forgetting evaluation to `utils/metrics.py`.
- **Sign-flip augmentation.** The upstream forecaster averages each forecast with the reflected forecast of the negated series by default **[U forecaster.py:221-227]**. Fine-tuning on one-signed data can make the two passes disagree. Randomly negating whole series (targets and covariates together, `p = 0.5`) during training keeps them coherent. **On by default (`TrainingConfig.sign_flip=True`)** from Phase 3 onward (decision D19): verified on `123M` and `209M` across 3 seeds to improve one-signed volatility and retail validation wQL by up to 6.7% while staying within seed spread (improving 5/6 cells) on signed 20-bar log returns **[V]**. In `head_only` with `cache_frozen_features=True`, both `+x` and `-x` orientations are cached on CPU within the 256 MiB cap.

### 5.4 Trainer

- **Forward pass.** The trainer never calls `forecast`. It builds the same inputs as upstream's `_assemble_rows` **[U forecaster.py:332-394]**, runs the two-stream transformer, and takes normalised quantiles at the horizon prediction tokens of target rows.
- **Optimiser.** AdamW with cosine decay and linear warm-up. Sweep-backed default learning rates (`docs/BENCHMARKS.md` §2, decision D19): `3e-3` for `head_only` and `lora_variate`, `1e-3` for `lora_temporal` (`r=8, alpha=16, targets=("q", "k", "v", "o")`), and `1e-4` for `full_ft` **[V]**. As a user recommendation (not a default), setting `learning_rate=3e-4` for `full_ft` on `5M` and `28M` gave lower error (three seeds on `5M`, one seed on `28M`, mainly on volatility for `28M`) at the cost of roughly two to three percentage points more forgetting; keep `1e-4` on `123M` and `209M`, where the two rates tie or favour `1e-4` on validation and `1e-4` forgets less (`docs/BENCHMARKS.md` §2.2).
- **Sign-flip augmentation.** `TrainingConfig.sign_flip=True` by default (decision D19); randomly negates whole series (targets + covariates together, `p=0.5`) during training to match upstream's `symmetric_averaging=True`. Verified across 3 seeds on `123M` and `209M` (`docs/BENCHMARKS.md` §2.1, `docs/VERIFICATION.md` §14): stays within seed spread on signed `logret_20` returns while cutting retail validation wQL by 5.9%–6.7% and volatility wQL on 3/3 seeds **[V]**.
- **Precision.** `bf16` autocast on CUDA (verified on an NVIDIA L4 **[V]** across all four trainable modes via `test_bf16_training_on_cuda`; raises on CPU, MPS, and non-bf16 GPUs); `fp32` on CPU and on the Apple GPU. The head already promotes to at least fp32 **[U model.py:96]**, statistics run in float64 without gradients **[U scaler.py:28-29]**, and `pinball_loss` promotes to float32. The paper reports bf16 inference within 0.2% of fp32 **[P §4.3]**; in one-off 20-step L4 training trajectories across 5M and 28M, `bf16` validation loss stayed within 0.1%–0.2% of `fp32` while running 2.0×–2.1× faster on 28M and cutting VRAM by ~30% (`docs/VERIFICATION.md` §13; reproducible via `scripts/probes/step_timing.py --device cuda --bf16`).
- **Attention kernel.** Training uses upstream's attention call with PyTorch's default kernel selection, exactly as upstream inference does. An experimental helper, `timestral_tune.models.forward.attention_context("math")`, switches PyTorch to its unfused kernel for the calls inside it; acceptance test T14 and the gradient probe use it to measure the float32 gradient inaccuracy of §12. Twenty-step L4 CUDA training trajectories across `lora_variate`, `lora_temporal`, and `full_ft` (`docs/VERIFICATION.md` §13) show `fp32` default SDPA and `fp32` unfused math matching in validation loss within 0.0002 (identical to 5 decimal places on 28M), so no public attention-kernel toggle is needed (decision D15).
- **Compilation.** `TrainingConfig.compile_model=True` compiles `model.transformer` with `torch.compile` (`compile_mode="default"` or `"reduce-overhead"`). Under `mode="reduce-overhead"`, `DHFDataCollator` automatically pads partial tail batches into power-of-two shape buckets using zero-weighted, group-isolated dummy series (`pad_batch_to_series`), bounding distinct compiled shapes to $\lceil \log_2 B \rceil + 1$ without dropping data or changing loss/gradients. On L4 CUDA, `bf16 + compile("default")` ran 28M `full_ft` at 59.8 ms / 1.80 GiB (3.3× faster than `fp32`, 45% less VRAM) and `lora_variate` at 42.9 ms / 0.94 GiB (`docs/VERIFICATION.md` §13; `scripts/probes/step_timing.py --device cuda --bf16 --compile default`).
- **Gradient clipping** at 1.0 by default.
- **Dropout.** The base model has none **[P §3.3]**. LoRA dropout applies to the adapter path only.
- **CUDA.** Verified **[V]** on an NVIDIA L4 (PyTorch 2.15.0, fp32 and bf16): forecasts match CPU to 1e-7 to 1.5e-5 relative, T1–T14 and the whole suite pass (580 passed, 4 skipped at `89ac249`, including all four modes with replay/distillation in `fp32` and `bf16`, both cached and live-LoRA `reduce-overhead` compile under CUDA graphs, and released `Timestral-5M` `bf16` replay in `head_only` and `lora_temporal`; `full_ft` on released `Timestral-5M` is covered by `scripts/probes/replay_bf16_parity.py`, not by the suite), `AdamW` runs `fused=True`, `bf16` dual-orientation head caching matches streaming `bf16`, shape-bucketed `reduce-overhead` compiles cleanly on non-divisible datasets, and `bf16 + torch.compile` speeds up 28M covariate steps 3.1×–3.3× over fp32. Float32 gradients through the default attention kernel are less accurate than on the other devices (up to 1.1e-2; §12), though 20-step training trajectories match unfused math closely (D15). The environment needs a C compiler and Python headers (§12).
- **Apple GPU (MPS).** Supported in v0.1 by running the four float64 statistics sites of §3.5 on CPU and moving their float32 results to the device **[D]**. The statistics carry no gradients, so this changes nothing in the maths. Measured **[V]**:
  - *Forward parity with CPU:* relative difference of 1.9e-7 to 3.4e-6 for `predict_block` in both layouts, 1e-7 for a 96-step forecast, up to 2e-5 for a 2,100-step block-autoregressive forecast, and 1e-7 with numeric and categorical covariates.
  - *Gradients of the training loss:* within 2e-7 to 2.1e-3 of a float64 reference across the T14 scenarios, against 2e-7 to 3.4e-3 for CPU float32. §12 explains why released weights are not at round-off on any device.
  - *Speed of the full training step* (normalised forward, median-folded frame with its CPU round trip, pinball loss, AdamW; context 1,024, horizon 96, batch 32): between 1.4× and 3.2× faster than CPU across four runs, for both univariate batches and batches with five rows per series. Timings vary from run to run.
  - *Memory:* about 4.5 GiB allocated for 28M with five rows per series at batch 32 (down from 7.3 GiB before D16).
  - *Mixed precision:* no speed-up from `bf16` or `fp16` autocast, and `bf16` moved outputs by 1e-2 to 6e-2, so MPS runs fp32 only and `mixed_precision` other than `"fp32"` raises there.
  - *Requirements and scope:* PyTorch 2.8 or newer. Older versions lack the `cummax` operator on this backend, which upstream's input assembly uses; `resolve_device` skips the Apple GPU under `"auto"` and raises for an explicit `"mps"`. `5M` and `28M` are verified on PyTorch 2.8.0 and 2.15.0; **`123M`, `209M`, and long contexts up to `32,768` steps (`1,024` patches)** are verified on PyTorch **2.15.0** on one 16 GiB Apple M4 machine (`scripts/probes/mps_large_and_long_context.py`, `docs/VERIFICATION.md` §15) **[V]**: `123M` and `209M` match CPU to `9.5e-8..1.1e-6` forward and `4.5e-6..8.0e-6` gradient vs float64 reference (cleaner than CPU fp32), and `209M` completes one finite optimiser step in `full_ft` at `32,768` steps (`B=1`, `6.09 GiB` driver) and `cov3` `lora_temporal` at `16,384` steps (`6.10 GiB` driver; long-context gradient parity is untested).
- **Device shims.** The offload is implemented as subclasses outside `_vendor/` that move tensors to CPU and call the parent: a model subclass overriding `scale`, and a forecaster subclass overriding `_cap_quantiles`, `_refine_block` and `_encode_categorical`. The training forward calls `advancing_block_frames` on CPU directly. No vendored file is edited for this (§6.1).
- **Memory.** With covariates the trimmed single-horizon layout uses `L + 2h` tokens per row (§3.4, decision D16) instead of upstream's `2(L + h)`, cutting covariate activation memory by ~42% on CUDA (5.7 → 3.3 GiB on 28M batch 32) and ~38% on the Apple GPU (7.3 → 4.5 GiB) **[V]**.
- **Frozen-feature caching (`head_only`) [D].** `TrainingConfig.cache_frozen_features=True` (default) precomputes each window's frozen horizon hidden states, anchors, targets, and 3-scalar float64 context Welford summaries (or pre-normalised targets for `true_values`) on CPU up to a 256 MiB cap per dataset, skipping re-execution of the frozen transformer body across epochs and repeated evaluations (decision D16). When `sign_flip=True`, both `+x` and `-x` orientations are cached for the training set (counted together against the 256 MiB cap) and selected per window each step, matching streaming `sign_flip=True` to `< 1e-4` L2. Falls back to streaming above the cap or when disabled.

### 5.5 Evaluation

All reported forecasts go through the vendored upstream `TimestralForecaster` with its defaults (symmetric averaging and non-negative clipping on), so tuned and zero-shot numbers are produced by the same code path.

Metrics, in raw space: MASE, mean weighted quantile loss over the 21 levels, pinball loss, empirical coverage and mean width of the 80% and 90% intervals, and the forgetting metric of §5.3.

### 5.6 Multi-process training (proposed, not implemented)

> **Status: proposed design only.** Nothing in `src/` or `tests/` implements multi-process training; decisions D22–D35 and acceptance test T15 await maintainer go/no-go and sequencing against gradient accumulation (`docs/phases/phase-5-multi-gpu.md`). The single-process path of §5.4 is unchanged.

- **Contract [D] (D25).** For the deterministic path (`sign_flip=False`, LoRA `dropout=0`), one optimiser step across $N \ge 2$ ranks on shards $\{B_0, \dots, B_{N-1}\}$ matches one single-process step on the union batch $B = \bigcup_{r=0}^{N-1} B_r$ in loss scalar, gradient, clipping decision, and parameter update up to floating-point rounding. With `sign_flip=True` or LoRA `dropout > 0`, ranks draw from rank-offset streams (D33), so an $N$-rank run and a one-process run execute the same statistical procedure rather than identical numbers. Reproducing single-process `DataLoader` batch order across different $N$ is not claimed. When no multi-rank process group is active (`not dist.is_initialized()` or `world_size == 1`), the one-process path runs byte-for-byte unchanged.
- **Where collectives happen [D] (D29, D31).** Collectives happen in exactly two places: `TimestralTrainer.__init__` (startup failure-status check and initial-state broadcast) and `TimestralTrainer.fit()`. `load_model()`, `load_forecaster()`, `resolve_device()`, `evaluate()`, `export()`, and public `training_step()` are collective-free. Calling public `training_step()` in an active multi-process group raises `RuntimeError` directing callers to `fit()`, which runs an internal distributed step.
- **The step and explicit gradient reduction [D] / [V] (D22, D23).** `TimestralModel` defines no `forward` (inheriting `nn.Module._forward_unimplemented`, PyTorch 2.15.0 `torch/nn/modules/module.py:391-404`), and training calls `compute_loss` or `training_forward` directly (§5.4). `DistributedDataParallel` arms reduction only inside `DistributedDataParallel.forward` (PyTorch 2.15.0 `torch/nn/parallel/distributed.py:1906-1914`; `_pre_forward` at lines 1776–1780, `_post_forward` at lines 1850–1861), so calling `compute_loss(ddp.module, ...)` bypasses both hooks and leaves gradients unsynchronised: in the CPU/`gloo` prototype (`scripts/probes/distributed_parity.py`, `docs/VERIFICATION.md` §16), 2-rank `ddp.module` cross-rank gradient relative L2 was `2.46e+00` (`max_abs=5.49e-02`) on `tiny_random` (PyTorch 2.15.0 and 2.4.0) and `1.04e+00` (`max_abs=7.52e-01` float64 / `7.53e-01` float32) on `Timestral-5M` (PyTorch 2.15.0), versus `0.00e+00` for the proposed step **[V]**. Instead, trainable gradients are packed into one flat buffer after backward and reduced with `all_reduce(..., op=ReduceOp.SUM)`.

  *Collective-safety invariant:* every branch that changes which collectives a rank executes must depend only on globally reduced values, never on rank-local batch state. Each global step inside `fit()` runs five stages using `ReduceOp.SUM` only (`ReduceOp.AVG` raises on CPU/`gloo` in PyTorch 2.4.0; one-off check in `docs/VERIFICATION.md` §11):
  1. *Local stage:* if the rank's shard for iteration $k$ is non-empty, it fetches from its `DataLoader`, applies `sign_flip` and `batch.validate`, moves the batch to its device, and computes local domain mean loss $L_d^{(r)}$ and scored count $W_d^{(r)}$ (skipping forward if empty or $W_d^{(r)} = 0$), plus—in Stage 2—replay pinball $(L_r^{(r)}, W_r^{(r)})$, distillation $(L_m^{(r)}, M^{(r)})$, and non-finite flag $\text{nf}^{(r)} \in \{0, 1\}$. If a local exception occurs in data loading, `batch.validate`, or forward, the rank catches it, zeroes its statistics entries, and sets local-error flag $\text{err}^{(r)}$ (e.g. rank bitmask $2^r$, exact in `float64` up to 53 single-node ranks).
  2. *Statistics collective:* one 8-element `float64` `all_reduce(ReduceOp.SUM)` over $[W_d^{(r)}, s_d^{(r)}, W_r^{(r)}, s_r^{(r)}, M^{(r)}, s_m^{(r)}, \text{nf}^{(r)}, \text{err}^{(r)}]$, with $s_d^{(r)} = L_d^{(r)} W_d^{(r)}$ when $W_d^{(r)} > 0$ and finite (else $0$), and similarly for $s_r^{(r)}$ and $s_m^{(r)}$, yielding $[W_D, S_D, W_R, S_R, M_{\text{tot}}, S_M, \text{nf}_{\text{tot}}, \text{err}_{\text{tot}}]$.
  3. *Globally decided control flow:* if $\text{err}_{\text{tot}} > 0$, failing rank(s) re-raise their original exception and peers raise an error naming the failing rank(s) without running backward. If $\text{nf}_{\text{tot}} > 0$, every rank raises `FloatingPointError` without running backward. If $W_D = 0$, every rank skips the step with exact one-process skip semantics: no backward, no optimiser or scheduler step, `global_step` unchanged, `skipped_batches` incremented, identical per-epoch loop iterations, and `RuntimeError` on every rank if a whole epoch produces no optimiser step. *(Open Stage 2 decision: the one-process loop skips zero-scored batches before advancing `replay_iter`, `trainer/trainer.py:1013-1025`, whereas skipping after stage 1 would advance `replay_iter`; the current leaning is a small pre-forward count collective used only when `replay_config` is set, leaving Stage 1 unaffected.)* Replay pinball mixes iff $f > 0$ and $W_R > 0$ ($a_d = 1 - f$, else $a_d = 1$); distillation is added iff $\lambda > 0$ and $M_{\text{tot}} > 0$. Reported global loss is $a_d \frac{S_D}{W_D} + \mathbb{1}_{\text{replay}}\, f \frac{S_R}{W_R} + \mathbb{1}_{\text{distill}}\, \lambda \frac{S_M}{M_{\text{tot}}}$.
  4. *Weighted local backward:* each rank with an active local term backpropagates
     $$\tilde{\mathcal{L}}^{(r)} = a_d \frac{W_d^{(r)}}{W_D}\, L_d^{(r)} + \mathbb{1}_{\text{replay}}\, f \frac{W_r^{(r)}}{W_R}\, L_r^{(r)} + \mathbb{1}_{\text{distill}}\, \lambda \frac{M^{(r)}}{M_{\text{tot}}}\, L_m^{(r)},$$
     and skips `.backward()` otherwise.
  5. *Gradient collective and update:* pack trainable gradients into one flat buffer (`None` or absent gradients as zeros), call `all_reduce(buf, op=ReduceOp.SUM)` without dividing by $N$, copy back into `p.grad`, clip (`clip_grad_norm_`), then step the optimiser and scheduler.

  Single-rank exception propagation—via the `__init__` startup status collective below, the 8th statistics entry, the 4th validation entry, and the callback-failure stop status—is **best-effort and unprototyped** (only the non-finite flag has been prototyped; §12). It excludes failures in backward, post-broadcast `__init__` failures, killed processes, and broken or unavailable device contexts unable to enter a collective; those end by launcher teardown or process-group timeout (D35).
- **Why count weighting [D] / [V] (D24).** `pinball_loss` (§4.2) averages equally over scored positions, so the union-batch loss is a count-weighted sum of rank means:
  $$\mathcal{L}(B) = \frac{1}{|\mathcal{Q}|\, W_D} \sum_{r=0}^{N-1} \sum_{i \in B_r} w_i \sum_{\tau \in \mathcal{Q}} \rho_\tau(z_i - \hat{q}_{\tau, i}) = \sum_{r=0}^{N-1} \frac{W_d^{(r)}}{W_D}\, L_d^{(r)}, \qquad W_D = \sum_{r=0}^{N-1} W_d^{(r)}.$$
  By linearity of differentiation, $\nabla_\theta \mathcal{L}(B) = \sum_{r=0}^{N-1} \frac{W_d^{(r)}}{W_D} \nabla_\theta L_d^{(r)}$, and the same identity holds for `L_replay` and `L_distill`. Equal-weight averaging $\frac{1}{N} \sum_r \nabla_\theta L_d^{(r)}$ disagrees whenever active counts differ across ranks. In the CPU/`gloo` prototype (`docs/VERIFICATION.md` §16), plain averaging had relative L2 gradient error `8.97e-02`–`2.78e+00` across non-equal cases on `tiny_random` (`2.39e-01`–`1.79e+00` on `Timestral-5M`), while count weighting matched `probe_ref` to `1.49e-16`–`6.89e-16` (`float64`) and `1.01e-07`–`5.57e-07` (`float32`) on PyTorch 2.15.0 (`1.53e-16`–`7.51e-16` and `1.00e-07`–`3.76e-07` on 2.4.0), with scalar global loss within `0.00e+00`–`3.01e-16` (`float64`) and `0.00e+00`–`1.29e-07` (`2.15.0`) / `0.00e+00`–`1.48e-07` (`2.4.0`) (`float32`), and matched independent one-process `TimestralTrainer` (`shipped_trainer`) within `1.49e-16`–`6.64e-16` / `1.01e-07`–`5.57e-07` (`2.15.0`) and `1.53e-16`–`7.44e-16` / `1.00e-07`–`3.48e-07` (`2.4.0`), passing `326/326` checks on both versions including 10-active-step (+1 zero-scored skip) AdamW trajectories across `full_ft` and `lora_variate` (`final_params` `<= 5.32e-16` f64 / `<= 3.01e-07` f32, `param_update` `<= 1.84e-14` f64 / `<= 1.26e-05` f32 across both versions; cross-rank `0.00e+00`; exact clipping and skipped-step no-op parity), global skip, and non-finite propagation **[V]**.
- **Sampling and batch size [D] (D26, D27).** Each epoch generates one global permutation of `0..len(dataset)-1` from `(seed, epoch)` (natural order when `shuffle=False`). Full iterations assign each rank $B = \text{config.batch\_size}$ indices; a trailing partial batch of length $R = \text{len}(\text{dataset}) \bmod (N \cdot B) > 0$ splits evenly so rank sizes differ by at most one ($\lfloor R / N \rfloor + \mathbb{1}_{r < R \bmod N}$), leaving a rank empty only when $R < N$. The per-iteration union is unchanged, preserving `shuffle=False` equivalence with a one-process run at `batch_size = N * B`. Every rank runs $\lceil \text{len}(\text{dataset}) / (N \cdot B) \rceil$ loop iterations per epoch; `total_steps`, the LR schedule, `max_steps`, `eval_every`, and `log_every` count global optimiser steps. The epoch loop is driven by global iteration index rather than by iterating `DataLoader`. A main-process batch sampler yields only non-empty index lists, so `num_workers > 0` and `persistent_workers=True` work and `DHFDataCollator` (which raises `ValueError` on an empty list at `data/collator.py:155-156`) is never called empty. An empty rank skips fetching, `sign_flip`, `batch.validate`, and forward, contributing zero statistics and zero gradients: in the prototype `[2]` domain-loss cases, empty-rank-0 delivery (`N=2 empty_shard`, `W_d=[0, 112]`) passed with `empty_r0_recv_ok=True` and cross-rank `0.00e+00`, and multi-rank cases including `N=4 empty_shard` matched `probe_ref` to `3.31e-16`–`4.72e-16` f64 / `1.68e-07`–`2.82e-07` f32 on PyTorch 2.15.0 and `3.11e-16`–`4.58e-16` / `1.64e-07`–`2.69e-07` on 2.4.0 (`docs/VERIFICATION.md` §16.1 **[V]**). PyTorch's `DistributedSampler` is not used because for 10 items across 4 replicas it duplicates `[0, 1]` when `drop_last=False` or drops `[8, 9]` when `drop_last=True` (`docs/VERIFICATION.md` §16.1 **[V]**). `batch_size` (and Stage 2 `ReplayConfig.batch_size`) is per rank, giving effective batch $N \cdot B$. Because shipped defaults (D19) were swept only at batch sizes 16 (`docs/BENCHMARKS.md` §2.1) and 32 (§2.2), multi-process mode logs $N \cdot B$ and warns when `learning_rate is None` with $N \cdot B > 32$, with no automatic LR scaling.
- **Devices [D] (D28).** The caller owns process-group lifecycle (`torchrun` / `dist.init_process_group`), and `TimestralTrainer` activates multi-process mode when `dist.is_initialized()` with `world_size > 1`. `resolve_device()` stays pure and unchanged. Multi-process device resolution lives in `TimestralTrainer.__init__` and follows `dist.get_backend()`: `nccl` requires CUDA, `gloo` runs on CPU (`gloo` with CUDA is refused in the first version), `"auto"` resolves accordingly and never picks MPS, and contradicting requests (`"cpu"` with `nccl`, CUDA with `gloo`, `"mps"` with either) raise at construction. For CUDA, an explicitly indexed device (e.g. `cuda:1`) is honoured; otherwise `LOCAL_RANK` is used when set, else the process-group `rank`. The index is validated against `torch.cuda.device_count()`, and `torch.cuda.set_device` is called for every CUDA device (explicit ones included) before anything querying the current device, notably `torch.cuda.is_bf16_supported()` (`trainer/trainer.py:292`).
- **Loading and startup broadcast [D] (D29).** Every rank calls `load_model()` as today with no rank-0 download barrier. Vendored `resolve_checkpoint` calls `snapshot_download` (`_vendor/timestral/loading.py:44`), which in installed `huggingface_hub 2.4.0` delegates to `hf_hub_download` (`_snapshot_download.py:521`) and serialises concurrent per-file downloads with `WeakFileLock` (`file_download.py:1258, 1273` for the cache dir; `1485, 1503` for a local dir). Whether the `huggingface_hub>=0.23` floor (`pyproject.toml:27`) uses the same lock is unverified (§12). Inside `TimestralTrainer.__init__`, all rank-local work that can fail runs first—backend-driven device resolution and `bf16`/Stage 1 restriction checks, `lora_config`/`replay_config`/`replay_dataset` checks, model loading or validation, dataset checks (`_check_dataset` and train/val layout equality), `_select_trainable()`, and `early_stopping_patience`/callback validation (`trainer/trainer.py:286-351, 375-383`)—catching any exception locally. All ranks then execute one small startup status collective (`all_reduce(ReduceOp.SUM)` of per-rank failure flags, e.g. rank bitmask $2^r$); if any rank failed, failing rank(s) re-raise their original exception and peers raise an error naming the failing rank(s) without entering the broadcast. Only if all ranks succeed does `__init__` broadcast **all model parameters and all buffers** from rank 0 (before `_setup_replay_teacher()` and `torch.compile`, which depend on the synced eager model) so caller-supplied `model=` weights agree even in `head_only` and LoRA modes. Startup failure propagation is **best-effort and unprototyped** (`[D]`, §12): it requires every rank to be able to enter the status collective (on NCCL, a rank without a valid CUDA device context—for example `device_count() == 0` or a rank exceeding available GPUs—cannot enter an NCCL collective and falls under D35, as do post-broadcast failures in `_setup_replay_teacher()` or `torch.compile`). As parameter-count arithmetic (not a measurement), `Timestral-209M`'s `209,312,256` parameters (§3.1, `docs/BENCHMARKS.md` §3) in `fp32` (`4 B/param`) represent a one-time `~798 MiB` (`~0.78 GiB`) broadcast.
- **Seeding and timeouts [D] (D33, D35).** Epoch sampling and LoRA init share `config.seed`. `sign_flip`, replay flips, and LoRA dropout use rank-offset streams keeping rank 0 on the single-process seed and giving distinct ranks distinct streams (formula left to implementation). Process death or backward failure ends by launcher teardown or process-group timeout; callers should pass an explicit `timeout` to `init_process_group` (defaults and citations in D35).
- **Rank 0's role and end of `fit()` [D] (D30).** Callbacks (`ProgressLogger`, `EarlyStopping`, user callbacks), best-loss tracking, early stopping, and the restore decision run on rank 0 only; non-zero ranks never compare validation losses. After `on_fit_start`, `on_step_end`, and `on_evaluate`, rank 0 broadcasts a status (`continue`, `stop`, or `rank-0 callback failed`). At the end of `fit()`, rank 0 broadcasts `(best_eval_loss, best_step, restored_best)` and, only when `restored_best` is True, broadcasts restored `self._trainable`. All ten `TrainingResult` fields (`trainer/trainer.py:227-236`) aim to be equal on every rank: `steps`, `epochs`, `stopped_early`, and `skipped_batches` are equal by construction from lockstep control flow; `best_eval_loss`, `best_step`, and `restored_best` come verbatim from rank 0's end-of-`fit()` broadcast; `final_train_loss`, `train_history`, and `eval_history` come from the per-step and per-eval reduced `float64` losses (bitwise identical when `float64` `all_reduce(ReduceOp.SUM)` is bitwise deterministic as on `gloo`, or differing by at most `float64` reduction rounding if NCCL order varies across ranks, §12).
- **Validation, `evaluate()`, and `export()` [D] (D31).** Validation inside `fit()` uses its own internal path and sharded loader—never `self._val_loader`—so a later rank-local `evaluate()` still scores the full `val_dataset`. It shards `val_dataset` by stride (`r, r+N, …`) with no per-batch collectives and one end-of-eval `float64` `ReduceOp.SUM` over `[loss_sum, scored_positions, batches, err]`. Empty or zero-scored shards contribute zeros, and `ValueError("the dataset has no scored position to evaluate")` is raised only after reduction from the global count on every rank. Under sharding, `EvalResult.loss` and `EvalResult.scored_positions` match one-process `evaluate()`, while `EvalResult.batches` is the sum of per-rank batch counts (generally larger than the one-process count). Public `evaluate()` stays rank-local and collective-free. Public `export()` also stays collective-free: rank 0 writes the export directory and adds `world_size` and `effective_batch_size` to `tune_metadata.json` (§7.4), while non-zero ranks run identical argument validation without Hub calls, weight merges, or disk writes, log that rank 0 writes the files, and return the same `Path`. Returning on a non-zero rank does not guarantee files exist on disk yet; scripts reading exported files on another rank must synchronise themselves (e.g. `dist.barrier()`).
- **First-version restrictions [D] (D32, D34).** In multi-process mode, `head_only` skips the frozen-feature cache and streams instead (logged; whether a multi-process head cache is worth having is an open Stage 2 decision, §10), `compile_mode="reduce-overhead"` raises `ValueError` while `compile_mode="default"` is allowed (unverified on NCCL, §12), and `ReplayConfig` raises `NotImplementedError` in Stage 1 (added in Stage 2). `nn.DataParallel`, third-party wrappers (`accelerate`, `lightning-fabric`), multi-node training, and benchmark-script changes are out of scope (D34).

---

## 6. Package Architecture

```
src/timestral_tune/
│
├── __init__.py
├── config.py                 # TrainingConfig, LoraConfig, ReplayConfig
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
│   ├── trainer.py            # Training loop (all modes, clipping, eval, early stopping, bf16, compile)
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
- **Shape rules.** `context_length` may not exceed the model's context cap (32,768 steps for the released checkpoints); a longer one raises instead of being truncated, so that nothing is dropped silently. Every window of a dataset is left-padded with unobserved zeros to `context_length` rounded up to a whole patch, so all batches have one shape. Upstream pads each call only to the next multiple of 32 **[U forecaster.py:70-77]**; the extra padding is never attended to and leaves forecasts unchanged within 1e-5 **[V]**. The horizon is rounded up to whole patches; padded steps get zero loss weight. `forecast_horizon` must be at most 1,024 for training.
- **Preprocessing parity.** Target rows are forward-filled with the observation mask preserved; covariate NaNs become unobserved zeros. Apart from the extra left padding above, the collated tensors equal those of upstream's `_assemble_rows` **[V]**.
- **Storage.** Frames from polars, pandas or pyarrow are converted once to contiguous float32 arrays per series; none of the three is a runtime dependency. A window's numeric arrays are views into those arrays, its category codes are renumbered copies (§6.3 rule 4), and the collator makes the padded copies.
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
    mixed_precision="bf16",  # CUDA only; use "fp32" on CPU and Apple GPU
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
from timestral_tune.models import ConformalCalibrator, OnlineConformalCalibrator, collect_conformal_data

# Optional helper: collect raw-space (predictions, actuals, context_scale) via TimestralForecaster
val_preds, val_targets, val_scale = collect_conformal_data(forecaster, val_ds)

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
├── conformal.json         # If a fitted calibrator was passed (unfitted CQR raises)
├── tune_metadata.json     # Base repo, revision, model_provided_directly, mode, hyperparameters,
│                          # lora_config, replay_config, replay_source, has_conformal, content data
│                          # fingerprint, metrics, library version, vendored upstream commit
├── LICENSE                # Mistral's Modified MIT License, copied from the base repo
└── NOTICE                 # States that the weights are a derivative of the named base model
```

Saved LoRA adapters can be re-attached to a base model (and resumed in `TimestralTrainer`) with `load_lora(base_model, export_dir)`.

*(Proposed addition for multi-process runs, §5.6 / D31: `tune_metadata.json` — where per-rank `batch_size` is recorded at `hyperparameters["batch_size"]` by `export.py:_serialise_training_config` — also records `world_size` and `effective_batch_size`.)*

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

The 14 v0.1 acceptance tests (T1–T14) are implemented in `tests/acceptance/` and pass; T15 (multi-process training, §5.6) is proposed and not yet implemented.

These replace the original "assert zero gradients on non-target covariate positions" test, which was mis-specified: a target-only loss produces **non-zero** gradients on covariate inputs **[V]**, and that is intended, since targets read covariates.

| # | Test | Passes when |
|---|---|---|
| T1 | Forward parity, no covariates | `sinh(training_forward) · σ₀ + μ₀` equals `predict_block` output within 1e-6 relative, i.e. float32 round-off |
| T2 | Forward parity, with covariates | Same, for the trimmed `(L + 2h)`-token covariate layout with all three roles (and past-only covariates): within 1e-5 relative in float32, within 1e-12 relative in float64, and trimmed time mask bit-identical to slicing upstream's `build_two_stream_time_mask` with zero edges to dropped tokens |
| T3 | Frame parity | Applying the median-folded frame reproduces `_refine_block`, and the full pipeline matches `forecast(symmetric_averaging=False, clip_nonnegative=False)` on a forecaster built with `quantile_real_cap_k=0`, which disables its output clamp |
| T4 | Label isolation | Replacing horizon target values with noise leaves predictions bit-identical |
| T5 | Past-covariate cut | Changing a past-only covariate after the origin (in the raw frame) changes nothing |
| T6 | Firewall | Noise on a target's context changes no covariate row's hidden state or output |
| T7 | Positive control | Noise on a known-future covariate in the horizon does change the target forecast |
| T8 | Loss mask | The gradient of the loss with respect to predictions is exactly zero at covariate rows, unobserved steps and horizon padding |
| T9 | LoRA identity | A freshly initialised adapter leaves outputs unchanged; merged and unmerged outputs agree within tolerance |
| T10 | Single-row guard | `lora_variate` on single-row data raises; with the override, query and key slices are rejected |
| T11 | Export round-trip | The exported directory loads in the vendored upstream loader with `strict=True` and reproduces the trainer's forecasts |
| T12 | Encoding causality | A window's categorical inputs are unchanged when data after its origin changes: the target encodings with one target, and the category numbering with several |
| T13 | Conformal | On synthetic exchangeable data, calibrated coverage meets the nominal level; output quantiles are non-decreasing |
| T14 | Device parity | On each available accelerator, forecasts match CPU within 1e-4 relative. On every device including CPU, float32 training-loss gradients match a float64 reference (float64 weights and RMSNorm): within 1e-5 relative on a randomly initialised model; within 5e-2, and 1e-4 on one minus the cosine, on released checkpoints as shipped; within 5e-3 on released checkpoints when attention runs unfused, either written out explicitly or through PyTorch's unfused kernel, except one ill-conditioned case held to 2e-2 (§12 explains the tiers). With everything but the head frozen, head gradients match within 1e-5 on every model. Skipped where a device is absent |
| T15 *(proposed, not implemented)* | Multi-process parity (`test_t15_*`) | Network-free on CPU/`gloo` with the tiny random model for `N=2` and `N=4`, covering cases `(a)–(i)` below |

Perturbation tests use non-constant noise. A constant offset is absorbed by the normalisation and would make a firewall test pass vacuously.

**Proposed T15 cases (`test_t15_*`, CPU/`gloo`, `N=2` and `N=4`, tiny random model) [D].** Because `DHFDataCollator` emits `float32` inputs even for a `float64` model (`data/collator.py:204-206`), `float64` tiers use the setup of existing `float64` parity tests (`tests/helpers.py`: `float64_reference()` for float64 RMSNorm and `as_double`, as in `tests/acceptance/test_forward_parity.py`, `test_lora.py`, `test_device_parity.py`, and `tests/test_trainer.py`). Entry points:

- **(a) Multi-epoch trajectory parity (`fit()`):** dataset length not divisible by `N * B`, >1 epoch with `max_steps=None` sized so the whole run is about ten optimiser steps (the length at which the `1e-10` / `1e-4` trajectory tiers below were set and measured; §5.6, `docs/VERIFICATION.md` §16.1), warmup + cosine schedule, `weight_decay > 0`, all four modes, `shuffle=False`, `sign_flip=False`, LoRA `dropout=0`, clip `1.0`, vs one-process `fit()` at `batch_size = N * B`. Asserts final parameters and parameter updates match within `1e-10` relative (`float64`) / `1e-4` (`float32`) at that run length, recorded losses and optimiser step counts match, and cross-rank parameters are bitwise `0.0`.
- **(b) Step-level gradients & non-finite (internal distributed step):** constructed batches (unequal, zero-scored, empty shards; all four modes) match union-batch gradients within `1e-12` (`float64`) / `1e-5` (`float32`) with `0.0` cross-rank diff; single-rank non-finite loss raises `FloatingPointError` on all ranks cleanly.
- **(c) Global zero-scored step (`fit()`):** `W_D == 0` runs no optimiser/scheduler step, leaves `global_step` unchanged, and increments `skipped_batches` on every rank (`fit()` owns both, `trainer/trainer.py:983, 1013-1015, 1056`).
- **(d) Initial broadcast (`__init__`):** ranks constructed with different weights, frozen weights and buffers included (`head_only`, LoRA, `full_ft`), match rank 0 after construction.
- **(e) Stop, restore, and `TrainingResult` (`fit()`):** early stopping stops all ranks at the same step; callbacks run on rank 0 only; with `restore_best=True`, all ranks end with identical weights matching one-process `fit()`; `TrainingResult` is equal across ranks.
- **(f) Sampling and flips (epoch batch sampler & `fit()`):** `shuffle=True` gives equal loop iteration counts (`ceil(len / (N * B))`) and exact once-per-epoch union coverage; trailing partial batch splits within 1; `sign_flip=True` with an empty trailing rank succeeds; rank 0's flip stream matches one-process while other ranks differ.
- **(g) Validation (`fit()` & `evaluate()`):** internal sharded validation matches one-process `evaluate()` within `1e-12` (`float64`) / `1e-5` (`float32`); `len(val_dataset) < N` neither hangs nor raises; rank-0 `evaluate()` after `fit()` scores the full dataset.
- **(h) Guards (`__init__`, `training_step()`, `export()`):** `reduce-overhead` raises `ValueError`; Stage 1 `ReplayConfig` raises `NotImplementedError`; device/backend mismatches raise; `learning_rate is None` with `N * B > 32` warns; `head_only` skips the feature cache; public `training_step()` raises `RuntimeError`; non-zero-rank `export()` validates arguments, logs, writes nothing, and returns `Path`.
- **(i) Single-rank failure propagation (`__init__` & `fit()`):** an exception on one rank during construction (e.g. a rank-local dataset or device/config error before the startup broadcast) or during `fit()` data validation raises on all ranks well inside the process-group timeout (failing rank re-raises its original exception; peers raise naming the failing rank).

**Released-weights measurements for T15 (not graded; tiers deferred to Stage 3, §12) [D].** In the CPU/`gloo` prototype (`scripts/probes/distributed_parity.py --sizes 5M`, PyTorch 2.15.0, `docs/VERIFICATION.md` §16.2), `Timestral-5M` body-backprop gradient and parameter-trajectory checks were ungraded (`68` `[INFO]` values; cross-rank `0.00e+00`, scalar global loss, empty-rank-0 delivery, skip, non-finite, `head_only`, LoRA step-0/1 invariants, and `[5]` clipping/skip-no-op invariants passed: `190` on `5M`, `516/516` total with `tiny_random`) **[V]**. **Both `float64` and `float32` body-backprop values exceeded the tiny-model bounds** **[V]**: single-step reached `2.42e-12` (`float64`, above `1e-12`) and `1.58e-03` (`float32`, above `1e-5`), and 10-step `full_ft` `param_update` reached `7.40e-10` (`float64`, above `1e-10`) and `4.80e-02` (`float32`, above `1e-4`) (`lora_variate` reached `final_params` `1.22e-04` and `param_update` `1.06e-02` in `float32`), while enforced scalar global loss stayed within `6.77e-16` (`float64`) / `4.75e-07` (`float32`), `head_only` gradients stayed within `1.12e-15` (`float64`) / `4.00e-07` (`float32`), and cross-rank diffs stayed `0.00e+00`. The larger `float32` differences are consistent with the large-logit float32 sensitivity of §12. Released-weights bounds are not fixed from this single-size, single-seed CPU run and will be set in Stage 3 across sizes, seeds, devices, and cosine similarity (which the probe does not yet report).

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
- [x] T14 on CUDA: passes on an NVIDIA L4 with PyTorch 2.15.0, on the tiny model, 5M and 28M, as does the whole suite (`bash scripts/cuda_check.sh`). Last run at commit `50c170a`, which includes the `head_only` gradient check, all five scenarios and the float64-RMSNorm reference.

**Exit:** T1–T8 and T12 pass on 5M and 28M; the frame decision is confirmed or revised. **Met on 2026-11-21.**

### Phase 2: Adaptation & calibration
- [x] Slice LoRA adapter, mode presets, single-row guard; tests T9 and T10.
- [x] `full_ft`.
- [x] `ConformalCalibrator` and `OnlineConformalCalibrator`; test T13.
- [x] Export layout and metadata sidecar; test T11.
- [x] `bf16` on CUDA and optional `torch.compile` with shape bucketing.

**Exit:** all tests T1–T14 pass; an exported LoRA checkpoint loads in the official package. **Met on 2026-11-21; merged into `main` on 2026-11-22.**

### Phase 3: Benchmarks & documentation
- [x] Benchmark **zero-shot vs. conformal vs. head-only vs. LoRA (variate, temporal) vs. full FT** on a realised-volatility dataset and a retail-demand dataset, across at least two model sizes (`docs/BENCHMARKS.md`, 5M and 28M).
- [x] Report the forgetting metric for every mode (`docs/BENCHMARKS.md`, GIFT-Eval energy/weather/traffic).
- [x] Learning-rate and rank sweeps to replace the placeholder defaults of §5.4 (`config.py`, `docs/BENCHMARKS.md` §2).
- [x] Evaluate sign-flip augmentation (`TrainingConfig.sign_flip`, `docs/BENCHMARKS.md` §2).
- [x] Lacuna integration: trial records and signal frames (`integrations/lacuna.py`, verified live against `lacuna-quant==0.14.1`).
- [x] Documentation, licence notices, release on PyPI: `timestral-tune` `0.1.0` published on 2026-11-23 from tag `v0.1.0` through the release workflow; installed from PyPI into a clean environment, it runs the README quick start (`docs/VERIFICATION.md` §10).

**Exit:** published tables with code to reproduce them; each default in §5.4 is backed by a sweep. **Met and merged into `main` on 2026-11-22; released on PyPI as `0.1.0` on 2026-11-23.**

### Phase 4: Anti-forgetting and Apple GPU scaling (rescoped)
- [x] Replay mixing, built-in synthetic generators (`SyntheticReplaySource`), and frozen-base distillation (`distillation_loss`, `ReplayConfig`, `disable_lora`), verified with unit tests and real-dataset forgetting ablations (`docs/BENCHMARKS.md` §5, `docs/VERIFICATION.md` §15).
- [x] Apple GPU (MPS) coverage for `123M`, `209M`, and long contexts up to `32,768` steps (`scripts/probes/mps_large_and_long_context.py`, `docs/VERIFICATION.md` §15).
- **Deferred (exploratory):**
  - [ ] Tiled training with a fixed block length: upstream's `build_two_stream_time_mask` takes `min_context` and `block_size` **[U masks.py:56-101]**, but tiling undoes D16's `L + 2h` token saving (roughly doubling tokens), conflicts with per-window categorical encoding (D1, §6.3 rule 4), requires whole-span known-future covariates, and requires a separate multi-origin batch/collator/frame pipeline whose payoff on small fine-tuning datasets is unproven.
  - [ ] Sampled block lengths within a sequence **[P §3.1]**: depends on tiled training and an unpublished block-length distribution.

**Exit (rescoped Phase 4):** replay + distillation implemented, parity-tested, and empirically evaluated against the forgetting metric; Apple GPU verified on `123M`, `209M`, and contexts up to `32,768`. **Met and merged into `main` on 2026-11-22.**

### Phase 5: Multi-process training (proposed — awaiting maintainer decision)

All items below are proposed and unimplemented (`docs/phases/phase-5-multi-gpu.md`). Sequencing against gradient accumulation is left to the maintainer.

- **Alternative / prerequisite option (pending maintainer decision):**
  - [ ] Gradient accumulation (`gradient_accumulation_steps`) sharing `(loss_sum, count)` helpers with `pinball_loss` and `distillation_loss`: uses the same count-weighting arithmetic as multi-process training, raises effective batch size on memory-limited long-context runs, and is fully testable on existing single-GPU/CPU/MPS hardware.
- **Stage 1 — Core multi-process training (domain loss, four modes, no head cache):**
  - [ ] Caller-owned process group detection, backend-driven device resolution in `TimestralTrainer` (`nccl`→CUDA, `gloo`→CPU, explicit indexed > `LOCAL_RANK` > `rank`, validated against `device_count()`, `torch.cuda.set_device` before `is_bf16_supported()`, `resolve_device()` unchanged), rank-local setup and validation followed by a startup failure-status collective and all-parameter/all-buffer rank-0 broadcast in `__init__` (no loader barrier; verify `huggingface_hub==0.23` lock behaviour), and public `training_step()` `RuntimeError` (D28, D29).
  - [ ] Global-iteration-driven loop with main-process epoch batch sampler, balanced trailing split (empty only when `< N` remain; empty ranks skip `DataLoader` fetch, sign flip, validation, and forward), rank-offset RNG streams, and effective-batch default-LR warning (D26, D27, D33).
  - [ ] Internal distributed step (8-scalar `float64` `ReduceOp.SUM` stats collective, globally decided skip/non-finite/error branches, count-weighted backward, flat-buffer `ReduceOp.SUM` gradient collective), internal sharded validation (separate from `self._val_loader`), rank-0 callbacks / best-loss tracking / status & end-of-`fit()` `(best_eval_loss, best_step, restored_best)` + trainable-weight broadcast, collective-free `export()` with non-zero-rank validation + logging + `tune_metadata.json` `world_size`/`effective_batch_size`, and Stage 1 restrictions (logged head-cache streaming fallback, `compile_mode="reduce-overhead"` `ValueError`, `ReplayConfig` `NotImplementedError`) (D22–D25, D30–D32).
  - [ ] Acceptance test T15 (`tests/acceptance/test_distributed.py`, `test_t15_*`, cases `(a)–(i)` in §9) passing on CPU/`gloo` for `N=2` and `N=4` on PyTorch 2.4+ and 2.15.
- **Stage 2 — Replay, distillation, and head-cache decision:**
  - [ ] Multi-process `ReplayConfig` (`ReplayConfig.batch_size` per rank, sharded replay micro-batches, `[W_r, sum_r, M, sum_m]` in the stats collective, global `W_R > 0` mixing branch), settling two open Stage 2 questions: global skip ordering (leaning toward a replay-only pre-forward count collective, leaving Stage 1 unaffected), and how the frozen-base teacher output cache is built across ranks.
  - [ ] Decide whether a multi-process `head_only` frozen-feature cache is worth having (per-epoch global permutations move windows between ranks across epochs, so caching requires either holding the whole dataset on every rank or exchanging caches across ranks).
- **Stage 3 — Hardware verification (CUDA / NCCL):**
  - [ ] Run `scripts/probes/distributed_parity.py --device cuda` and T15 on 2-GPU and 4-GPU Linux CUDA (NCCL), plus `bf16`, `compile_mode="default"`, `torchrun` launch, rank-failure timeout check, and throughput/scaling measurements to confirm D22 and fix released-weights T15 bounds (across sizes, seeds, devices, and cosine similarity).

**Exit (proposed):** T15 passes on CPU/`gloo` (Stages 1–2) and on 2-GPU/4-GPU CUDA/NCCL (Stage 3), with measured numbers recorded in `docs/VERIFICATION.md`.

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
| D12 | T14's gradient criterion, on every device: 1e-5 on a randomly initialised model; 5e-2 relative and 1e-4 on one minus the cosine on released checkpoints as shipped; 5e-3 on released checkpoints with attention unfused (explicit, and PyTorch's unfused kernel), with one named near-tie case held to 2e-2; 1e-5 for `head_only` gradients | Float32 gradients through the released weights with the default kernel are off by up to 3.4e-3 on CPU, 2.1e-3 on the Apple GPU and 1.1e-2 on CUDA, while a random model is at 4e-7 everywhere, so the as-shipped tier can only guard against a wrong gradient. The 5e-3 unfused tier is the check on our own code: its nine well-conditioned cases measure at most 1.2e-3 on any device. The tenth, 5M with past-only covariates, has two keys 3.46 apart at a logit of 351k and lands anywhere from 1.2e-3 to 9.0e-3 depending on kernel and machine, so it is bounded on its own instead of loosening the tier for all. The reference computes upstream's RMSNorm in float64: left in float32, as upstream has it, the reference itself was off by up to 3.8e-3. History: one tier at 1e-3, then two, then three, on 2026-11-21, as CUDA, a fifth scenario and an x86 CPU each showed the previous bound to be about that batch rather than about our code |
| D13 | Phase 1 trains `head_only` in fp32 only; other modes, `bf16` and `export()` raise `NotImplementedError` | No interim behaviour to unlearn later; each arrives with its own acceptance tests in Phase 2 |
| D14 | Category codes are numbered per window, among the categories the window can see | Numbering them once over the whole frame let a category that first occurs later renumber earlier windows' inputs when several targets are forecast together (found in review, 2026-11-21). Per-window ranks depend on nothing outside the window, keep the order of ordered categories, and need no stored mapping at inference |
| D15 | Keep PyTorch's default attention kernel for training; keep `attention_context` as an experimental measurement helper only | Verified across all four released checkpoints (`5M`, `28M`, `123M`, `209M`) on NVIDIA L4 CUDA (`docs/VERIFICATION.md` §13–§14, `docs/BENCHMARKS.md` §3): on `5M`/`28M`, 20-step AdamW trajectories under `fp32` default SDPA and `fp32` unfused math converge to virtually identical validation losses (`0.00000` diff on 28M, `<= 0.0002` on 5M); on `123M`/`209M`, peak attention logits are 3×–4× smaller (`123k–132k`), default-SDPA CUDA gradient error drops to `2.6e-4–6.9e-4` (`cos >= 0.9999997`), and 120-step real-data retail training gives `fp32` default / `fp32` math / `bf16` default test wQL of `0.1982 / 0.1978 / 0.1981` (`123M`) and `0.1980 / 0.1951 / 0.1947` (`209M`, with `bf16` 1.8× faster). Exposing process-global SDPA flag switching in the training API is unnecessary |
| D16 | Drop the `L` dead decoder tokens in the covariate training layout (`L + 2h` tokens instead of `2(L + h)`), tighten `target_frames` / batch validation / pinned transfers, and cache frozen-body features on CPU in `head_only` | In single-horizon training, upstream's `L` context-position decoder tokens are mathematically decoupled from every kept token (float64 parity 1e-16..9.8e-15; trimmed mask bit-identical to sliced upstream mask), and dropping them cuts covariate step time 1.7×–2.0× and VRAM ~42% on CUDA (390 → 208 ms, 5.7 → 3.3 GiB on 28M) and ~2× on CPU/MPS without touching `_vendor/`. T2 pairs a 1e-5 float32 bound (for GEMM tiling differences of 6e-8..5.2e-6) with a 1e-12 float64 parity check and exact mask check. Caching frozen horizon hidden states + 3-scalar Welford summaries on CPU (256 MiB cap) speeds up multi-epoch `head_only` runs 3.1×–4.3× over 5 epochs with matching losses and head weights |
| D17 | Anchor-aware monotonicity repair in conformal calibration (`_repair_monotonicity_anchored`) | Blind outward `maximum.accumulate` across all 21 levels allowed uncalibrated interior neighbours (e.g. `q_0.85`) to pull a negatively shifted single target coverage (default `0.80`) right back outward, leaving 2.2× over-wide intervals at 97.6% coverage instead of 80%. Repairing calibrated levels outward among themselves (clamped at the unshifted median) and reconciling uncalibrated levels between fixed anchors lets single-coverage calibrators narrow cleanly while keeping all 21 levels non-decreasing and remaining identical to pure outward repair for `target_coverage="all"` |
| D18 | Bound online ACI `alpha_t` at Lemma 4.1's `[-gamma, 1 + gamma]` envelope and count `window` in forecast observations | Because finite rolling-window extremes (`max`/`min`) replace ACI's infinite / empty intervals at `alpha_t <= 0` and `alpha_t >= 1`, unbounded updates allowed `alpha_t` to wind up to `+2.84` / `-2.2` under sustained shifts, causing 500–2,200 step recovery lags. Bounding only at `[-gamma, 1 + gamma]` preserves unclipped crossing of `0` and `1` (`.saturated` and `min`/`max` branches still fire) while restoring Lemma 4.1's recovery bound. Counting `window` in forecast observations keeps `window=500` meaningful at `bucket_size=32` (500 forecasts rather than ~15) |
| D19 | Shipped Phase 3 defaults: `sign_flip=True`, `head_only=3e-3`, `lora_variate=3e-3`, `lora_temporal=1e-3`, `full_ft=1e-4`, and LoRA `r=8, alpha=16, targets=("q","k","v","o")` | Decided from 3-seed validation wQL and worst-seed macro GIFT-Eval forgetting on `123M` and `209M` (`docs/benchmarks/sweeps_large_decisions.json`, `docs/VERIFICATION.md` §14; zero test peeking) plus full-grid `5M`/`28M` sweeps (`sweeps_bracketed.json`): **(1) `sign_flip=True` (Rule: default True if within seed spread on signed target on both sizes):** on signed `logret_20`, flipping improved 5/6 mode/LR cells and stayed within 1/3 of seed spread on the 6th (`123M full_ft 1e-4`: `0.6176` vs `0.6168`, spread `0.0025`), while reducing retail validation wQL by 5.9%–6.7% and improving volatility on 3/3 seeds; dual-orientation `+x`/`-x` caching preserves `head_only` speed. **(2) `lora_temporal=1e-3` (Rule: take higher LR only if gain exceeds seed spread and worst-seed forgetting is acceptable):** `3e-3` failed to exceed seed spread on any `123M`/`209M` split, lost outright on 3/6, and hit `+19.6%`/`+23.8%` worst-seed forgetting on signed returns (vs `+0.3%`/`+0.7%` at `1e-3`). **(3) `full_ft=1e-4` (same rule + single default across sizes):** on `123M`/`209M`, `1e-4` won 3/6 splits outright (including both `209M` volatility and retail), `3e-4` never beat `1e-4` by more than seed spread on the other 3, and `1e-4` cut worst-seed retail forgetting in half (`+1.6%`/`+2.1%` vs `+4.0%`/`+4.2%`). We keep `1e-4` as the single cross-size default without size-dependent logic; as a user recommendation (not a default), setting `learning_rate=3e-4` for `full_ft` on `5M` and `28M` gave lower error (three seeds on `5M`, one seed on `28M`, mainly on volatility for `28M`) at the cost of roughly two to three percentage points more forgetting (`docs/BENCHMARKS.md` §2.2), while `123M` and `209M` should stay on `1e-4` |
| D20 | Chronological horizon-delayed online ACI evaluation (`_apply_online_chronological`) and equal-weighted 3-domain macro relative forgetting | Looping series-by-series over `TimeSeriesDataset` leaked one series's later outcomes into another series's earlier forecasts; grouping by forecast-origin timestamp and flushing `calibrator.update()` only after `t_end < t_origin` eliminates lookahead across series and overlapping horizons. Equal-weighted macro relative forgetting across `ett1`, `jena_weather`, and `loop_seattle` prevents high-magnitude pressure channels in `jena_weather` (~72% of pooled wQL denominator) from masking forgetting on other domains |
| D21 | Separate replay micro-batches, low-VRAM frozen-base teacher strategies (`disable_lora` / frozen head / CPU-cached base quantiles), and smooth L1 (`smooth_beta=1e-3`) in `distillation_loss` | `DHFDataCollator` requires uniform covariate layouts and horizons within a single `TrainingBatch`, whereas domain sets typically carry covariates and replay sources are univariate or general; separate replay micro-batches decouple layouts cleanly. Pure L1 (`|student - teacher|`) has a discontinuous `+/-1` subgradient kink at 0, so `~1e-7` fp32 GEMM tiling round-off between cached and live teachers at step 0 flipped AdamW step signs (`2*lr` divergence); Huber/Smooth-L1 (`smooth_beta=1e-3`, with `0.0` preserving exact L1) eliminates the kink while staying exact L1 for real drift (`>= 1e-3`). Precomputing frozen-base quantiles on CPU (256 MiB cap, dual `+x`/`-x` for `sign_flip`) avoids duplicating 209M weights in accelerator memory or running a second transformer pass per step |
| D22 | *(Proposed [D])* Explicit flat-buffer `all_reduce(ReduceOp.SUM)` after backward instead of `DistributedDataParallel` | `TimestralModel` has no `forward`, so calling `compute_loss` / `training_forward` bypasses DDP's forward/backward hooks and leaves gradients unsynchronised (§5.6, `docs/VERIFICATION.md` §16). *Alternative:* wrap `compute_loss` in a thin `nn.Module.forward` under DDP for communication overlap. *Reversible by:* NCCL measurements on `123M`/`209M` showing unoverlapped `all_reduce` costs materially |
| D23 | *(Proposed [D])* Five-stage internal distributed step (`ReduceOp.SUM` only), collective-safety invariant, best-effort exception propagation for local/validation/callback stages (excluding backward), and public `training_step()` raising `RuntimeError` | Deciding every control-flow branch from reduced values (`local_error`, `non_finite`, `W_D == 0`, `W_R > 0`) prevents mismatched collectives and preserves one-process skip semantics, while `ReduceOp.SUM` avoids PyTorch 2.4.0's `gloo` `ReduceOp.AVG` failure (§5.6, `docs/VERIFICATION.md` §11). Backward failures are excluded because on CUDA they are mostly OOM or device errors after which entering a collective is unreliable (they fall under D35); public `training_step()` raises on first call because a single-rank step cannot be correct without collectives. Stage 2 skip ordering remains open, leaning toward a replay-only pre-forward count collective (§5.6, §12) |
| D24 | *(Proposed [D])* Count-weighted loss-share scaling (`W_d/W_D`, `W_r/W_R`, `M/M_tot`) instead of `1/N` rank averaging | `pinball_loss` and `distillation_loss` average over local active positions, so equal-weight averaging disagrees with the union-batch gradient whenever active counts differ across ranks (§5.6, `docs/VERIFICATION.md` §16) |
| D25 | *(Proposed [D])* Step-level equivalence contract for the deterministic path (`sign_flip=False`, LoRA `dropout=0`); stochastic runs follow the same procedure; batch order across `N` not preserved; 1-process path unchanged | Exact deterministic parity is directly testable without perturbing the shipped one-process path, whereas matching epoch order or stochastic masks across arbitrary `N` adds coupling with no mathematical benefit (§5.6) |
| D26 | *(Proposed [D])* Global-iteration-driven loop with a main-process epoch batch sampler and balanced trailing split, instead of `DistributedSampler` | Avoids `DistributedSampler`'s item duplication or dropping (§5.6, `docs/VERIFICATION.md` §16), balances trailing memory across ranks, prevents empty calls to `DHFDataCollator`, and works with `num_workers > 0` and `persistent_workers=True` |
| D27 | *(Proposed [D])* Per-rank `batch_size` (and Stage 2 `ReplayConfig.batch_size`); warn on default LR above effective batch 32; no automatic LR scaling | Matches PyTorch data-parallel convention and keeps per-GPU memory predictable; shipped LR defaults (D19) were swept only at batch sizes 16 (`docs/BENCHMARKS.md` §2.1) and 32 (§2.2). *Reversible by:* large-batch LR sweeps (refusing automatic scaling currently rests on interpretation) |
| D28 | *(Proposed [D])* Caller-owned process group; backend-driven device resolution in `TimestralTrainer.__init__` (`nccl`→CUDA, `gloo`→CPU, `gloo`+CUDA and MPS refused, `set_device` before `is_bf16_supported()`); `resolve_device()` stays pure | Composes with `torchrun` with no new config fields, keeps `resolve_device()` free of distributed state, rejects backend/device mismatches at construction, and ensures current-device CUDA queries inspect the rank's assigned GPU (§5.6). Falling back from `LOCAL_RANK` to `rank` holds while multi-node is out of scope (D34) |
| D29 | *(Proposed [D])* No rank-0 download barrier (every rank calls `load_model()` as today); `__init__` runs rank-local setup/validation first, checks a startup failure-status `ReduceOp.SUM` collective, and broadcasts all parameters and buffers from rank 0 only if all ranks succeed | Dropping the barrier removes hangs on rank-0 load failure or single-rank `load_model()` calls, relying on `huggingface_hub`'s per-file `WeakFileLock` (§5.6; `>=0.23` floor check in Stage 1, §12). Checking a startup status collective before broadcasting stops surviving ranks from hanging when rank-local construction validation fails on a subset of ranks (best-effort and unprototyped, excluding ranks unable to enter a collective and post-broadcast steps, §5.6, §12). Broadcasting all parameters and buffers keeps caller-supplied `model=` weights aligned across ranks in `head_only` and LoRA modes |
| D30 | *(Proposed [D])* Rank 0 owns callbacks, best-loss tracking, early stopping, and the restore decision; hook status and end-of-`fit()` record/weight broadcasts keep `TrainingResult` equal across ranks | Prevents duplicate side effects and divergent floating-point comparisons across ranks, avoids broadcasting weights when no restore occurs, and keeps every rank in lockstep on stop or failure (§5.6; NCCL float-history caveat in §12) |
| D31 | *(Proposed [D])* Collectives only in `__init__` and `fit()`; loader, `evaluate()`, `export()`, and public `training_step()` are collective-free; separate internal sharded validation; non-zero-rank `export()` validates, logs, no-ops, and returns `Path`; `tune_metadata.json` adds `world_size` and `effective_batch_size` (§7.4) | Guarding `evaluate()` or `export()` under `if rank == 0:` never hangs, post-`fit()` `evaluate()` still scores the full validation set, and unguarded multi-rank `export()` avoids both directory clobbering and mid-write `torchrun` teardown (non-zero ranks reading exported files must synchronise themselves, §12). *Considered alternatives* (resting on API-ergonomics interpretation): raising on non-zero-rank `export()`, or an alternative raised while preparing this proposal to make `evaluate()` and `export()` collective |
| D32 | *(Proposed [D])* Stage 1 restrictions: stream `head_only` (multi-process head cache is an open Stage 2 decision), raise `ValueError` on `reduce-overhead`, and raise `NotImplementedError` on `ReplayConfig` (Stage 2) | Follows D13 ("no interim behaviour to unlearn later"), keeping Stage 1 small until core NCCL parity is verified (§5.6) |
| D33 | *(Proposed [D])* Shared seed across ranks for epoch permutation and LoRA init; rank-offset streams for `sign_flip`, replay flips, and LoRA dropout (rank 0 unchanged, distinct streams per rank; formula left to implementation) | Ranks must agree on sharding and initial weights while drawing independent augmentations and dropout, leaving single-process runs unchanged |
| D34 | *(Proposed [D])* Reject `nn.DataParallel`, `accelerate`, `lightning-fabric`, multi-node training, and benchmark-script edits | `nn.DataParallel` duplicates `TrainingBatch` unsplit (`torch/nn/parallel/scatter_gather.py:49-76`, PyTorch 2.15.0) and calls unimplemented `TimestralModel.forward` (`data_parallel.py:196`, `parallel_apply.py:114`, `module.py:391-404`). `accelerate` (`1.16.0`, PyPI 2026-11-23) requires `python >= 3.11` and `torch >= 2.6.0` (above our `>=3.10`/`>=2.4` floors) plus `packaging`, `psutil`, `pyyaml`; `lightning-fabric` (`2.6.6`, PyPI 2026-11-23) adds `fsspec[http]`, `packaging`, `typing-extensions`, and `lightning-utilities` outside this package's four runtime dependencies (`pyproject.toml`). Multi-node fault tolerance and benchmark re-runs are out of scope |
| D35 | *(Proposed [D])* Fail-by-timeout on rank death, backward failures, or other unrecoverable rank errors (no elastic recovery); recommend an explicit `timeout` in `init_process_group` | Surviving ranks block in the next collective until timeout (PyTorch 2.15.0 `torch/distributed/distributed_c10d.py:2515-2520`, identical wording in PyTorch 2.4.0 `1259–1264`: default 10 minutes for NCCL, 30 minutes for other backends). Elastic recovery is disproportionate for single-node fine-tuning; NCCL failure cleanup is unverified (§12) |

## 12. Open Questions & Risks

- **Target frame (D2).** Confirmed by calibration measurements on synthetic data for 5M and 28M (§4.3). Still not confirmed by Mistral, and not checked on real data, with covariates, or on 123M and 209M.
- **Hyperparameters.** Default `sign_flip=True`, learning rates (`head_only=3e-3`, `lora_variate=3e-3`, `lora_temporal=1e-3`, `full_ft=1e-4`), and LoRA defaults (`r=8, alpha=16, targets=("q","k","v","o")`) are backed by 3-seed validation & forgetting sweeps on `123M` and `209M` plus full-grid sweeps on `5M` and `28M` (`docs/BENCHMARKS.md` §2, `docs/VERIFICATION.md` §14, D19). As a user recommendation (not a default), setting `learning_rate=3e-4` for `full_ft` on `5M` and `28M` gave lower error (three seeds on `5M`, one seed on `28M`, mainly on volatility for `28M`) at the cost of roughly two to three percentage points more forgetting; keep `1e-4` on `123M` and `209M`, where the two rates tie or favour `1e-4` on validation and `1e-4` forgets less (`docs/BENCHMARKS.md` §2.2).
- **Benchmark contamination.** The paper says the models were post-trained on "a mixture of real and synthetic data" and does not list the real datasets. Any public benchmark dataset may have been seen. Results in `docs/BENCHMARKS.md` state this explicitly: the strict realised-volatility test targets (`2026-11-03` to `2026-11-20`) are post-release, while the retail-demand test targets (`2024-06-21` to `2024-07-02`) and the extended 10-month 2026 volatility holdout (`2026-01` to `2026-11-20`) are mostly or wholly pre-release.
- **Upstream drift.** Upstream is a first release with no stability promise. The drift-check job detects changes; re-vendoring requires re-running T1–T3 and T11.
- **Benchmark dataset choice.** Selected and pinned with permissive licences (`twelvedata/financial-world-model` MIT, `Dingdong-Inc/FreshRetailNet-50K` CC BY 4.0, `Salesforce/GiftEval` Apache-2.0; `docs/BENCHMARKS.md`).
- **Float32 gradients through the released weights are not at round-off.** Against a float64 reference, float32 gradients of the training loss with PyTorch's default attention kernel differ by up to 3.4e-3 on CPU, 2.1e-3 on the Apple GPU and 1.1e-2 on CUDA **[V]**, while forward outputs agree to about 3e-7. It depends on the weights, not the architecture: a randomly initialised model is at 4e-7 on every device.
  - *Measured cause* **[V]**. The released checkpoints have large query and key weights (5M layer 5: `||Wq||_2 = 141.6`, `||Wk||_2 = 82.4`; 28M layer 1: `196.1` and `160.6`), which produce attention logits of 370k–404k in 5M and 203k–503k in 28M across the probe layouts. One float32 step at those magnitudes is 0.016 to 0.031.
  - *Effect 1: the fused attention kernel.* At such logits the float32 backward pass of PyTorch's fused scaled-dot-product-attention kernel loses accuracy. On CPU it is the flash kernel: disabling that flag alone takes a 5M error from 1.1e-3 to 1.4e-5. Running attention unfused, either through PyTorch's unfused kernel or written out as matmul and softmax, lowers the error by 50 to 900 times on CUDA and 2.5 to 130 times on CPU across the eight probe layouts. On the Apple GPU the unfused kernel is identical to the default.
  - *Effect 2: near-ties between large logits.* When two keys of one query sit within an $O(1)$ gap at a logit of several hundred thousand, float32 cannot represent the gap accurately in any kernel, and the softmax weights depend on how each kernel rounds. T14's 5M batch with past-only covariates has such a query (two keys 3.46 apart at 351k; a one-off trace, `docs/VERIFICATION.md` §11). On that batch the worst kernel differs by machine: the unfused kernel on an Apple CPU (9.0e-3), explicit attention on an x86 CPU (5.3e-3), the default on CUDA (1.1e-2). A standalone attention call reproduces errors of about 1.5e-2 at logits of 350k–400k under every kernel.
  - *A third, separate source: upstream's RMSNorm* computes in float32 whatever the model's dtype. Left that way in the float64 reference, it moved the reference itself by up to 3.8e-3, so the reference computes RMSNorm in float64, without editing vendored code. RoPE still builds its table in float32, which remains a limit of the reference.
  - *Consequences.* The gradient direction is nearly unaffected: one minus the cosine is at most 5.7e-6 with the default kernel on CPU and the Apple GPU, 5.6e-5 on 5M/28M CUDA, and `2.4e-7` on `123M`/`209M` CUDA (whose peak logits are 3×–4× lower at `123k–132k`, giving default-SDPA CUDA gradient errors of `2.6e-4–6.9e-4`). **`head_only` training is not affected:** head gradients match the reference within 2.8e-6 on all four model sizes. Across 20-step 5M/28M trajectories and 120-step real-data `123M`/`209M` retail training (`docs/VERIFICATION.md` §13–§14, `docs/BENCHMARKS.md` §3), `bf16` with default SDPA matches or beats `fp32` math SDPA (`123M`: `0.1981` vs `0.1978`; `209M`: `0.1947` vs `0.1951`), so the library keeps PyTorch's default kernel (D15).
  - *Evidence.* Acceptance test T14 for the per-scenario maxima, the cosines, the near-tie case and `head_only`; `scripts/probes/gradient_accuracy.py` for the error matrix, the three-way kernel comparison, the standalone attention call, the effect of float32 RMSNorm on the reference, the peak and near-tie logits with weight norms, and the seed sweep; `scripts/cuda_check.sh` for all of these on CUDA; one-off runs for the 351k query trace, the kernel timings, and the float64-weights, TF32 and FP-fusion checks on CUDA (`docs/VERIFICATION.md` §6, §11, §13).
- **Validation loss under `median_folded` is measured in units that move with the model.** From the second horizon patch on, the frame is built from the model's own predicted medians, so two checkpoints are scored against slightly different normalised targets. Early stopping and best-state restoration compare those losses across steps. The effect shrinks with context length (the frame is dominated by the context) and is absent for horizons of one patch. Use `timestral_tune.utils.evaluate_forecaster` (raw-space MASE, wQL, pinball, coverage) for model selection across runs.
- **Apple GPU scope.** Needs PyTorch 2.8 or newer. Verified on one 16 GiB Apple M4 machine (`5M`/`28M` on 2.8.0 and 2.15.0; `123M`, `209M`, and contexts `4,096..32,768` on 2.15.0 only; `docs/VERIFICATION.md` §15).
- **Synthetic replay scope (D7).** Built-in `SyntheticReplaySource` is a simple regulariser, not Mistral's unreleased corpus. In our 5M retail `full_ft` ablation (`docs/BENCHMARKS.md` §5), neither synthetic pinball replay (`+4.1%` at 64 series, `+2.8%` at 256 series) nor synthetic-context distillation (`distill_weight=0.1`: `0.1999`, worst `+1.3%`) beat simply lowering the learning rate to `2e-4` (`0.1995`, worst `+1.3%`); and on `lora_temporal`, `3e-3 + replay_0.25` only recovers the accuracy/forgetting of shipped `1e-3`. Real general-domain replay datasets remain untested.
- **`torch.compile` on PyTorch 2.4.0 with Python older than 3.12.** `compile_model=True` needs `setuptools` installed there: PyTorch's compiler imports it, and that release's Python 3.10 wheel does not declare it **[V]** (`docs/VERIFICATION.md` §10). torch 2.8.0 on Python 3.10 installs it.
- **Worker processes on macOS with PyTorch 2.4.0.** A DataLoader with `num_workers > 0` leaves the process hanging at exit there **[V]**. It is a PyTorch issue that reproduces without this library and is gone by 2.8.0; `num_workers` defaults to 0.
- **CUDA scope.** Verified on one NVIDIA L4 with PyTorch 2.15.0 (CUDA 13.2) on Python 3.12, in both `fp32` and `bf16` (including `torch.compile`, power-of-two shape bucketing under `reduce-overhead`, `bf16` dual-orientation head-cache autocast, all four training modes, replay/distillation on both tiny and released `Timestral-5M` plus cached and live-LoRA `reduce-overhead` CUDA graphs, `load_lora` resume, and T1–T14, at commit `89ac249`). There is no CUDA machine in CI. Multi-GPU, older GPUs and other PyTorch versions are untested. On CUDA, recent PyTorch compiles a small C helper the first time some operations run (through its Triton component), which needs a C compiler and the interpreter's `Python.h` **[V]**; without them every CUDA forward fails with a compiler error. A uv-managed Python ships the headers.
- **Multi-process training risks (proposed, §5.6, D22–D35):**
  - *NCCL never run:* the prototype probe (`scripts/probes/distributed_parity.py`) has run only on macOS CPU with `gloo`. NCCL reduction determinism (whether cross-rank tensors—and therefore `TrainingResult`'s `final_train_loss`, `train_history`, and `eval_history`—remain bitwise `0.0`), flat-buffer `all_reduce` overhead vs DDP backward overlap (D22), and multi-GPU scaling efficiency are unmeasured.
  - *`bf16` autocast and `torch.compile` under multiple processes never run:* neither `mixed_precision="bf16"` nor `compile_mode="default"` (including empty trailing shards) has been exercised across ranks.
  - *`torchrun` on Linux never run here:* launcher rendezvous, `LOCAL_RANK` CUDA device binding, and `DataLoader` worker behaviour under `torchrun` on Linux are untested.
  - *Released-weights T15 tolerance tiers not fixed:* only `Timestral-5M` on CPU/`gloo` (one seed) has been measured (`68` ungraded `[INFO]` body-backprop values, `docs/VERIFICATION.md` §16.2); measurements across sizes (`5M`/`28M`/`123M`/`209M`), seeds, and devices—including cosine similarity, which the probe does not yet report—are required in Stage 3 before setting released-weights bounds.
  - *Learning rates at effective batch `> 32` not swept:* shipped defaults (D19) were swept only at batch sizes 16 and 32 (`docs/BENCHMARKS.md` §2); convergence and forgetting at effective batches 64–128 are unmeasured.
  - *Single-rank exception propagation unprototyped (`[D]`):* only the non-finite flag has been prototyped; the `__init__` startup failure-status collective, `local_error` statistics flag, sharded-validation error flag, and rank-0 callback-failure status are unprototyped and best-effort (backward failures, post-broadcast `__init__` failures in `_setup_replay_teacher()` or `torch.compile`, killed processes, and broken or unavailable device contexts unable to enter a collective are excluded and fall under D35).
  - *NCCL rank-failure behaviour unverified:* `init_process_group` documents asynchronous abort / `TORCH_NCCL_BLOCKING_WAIT` timeout behaviour (D35), but actual failure cleanup when a rank dies on CUDA/NCCL has not been observed here.
  - *Hub download lock at the dependency floor unverified:* `WeakFileLock` serialisation is verified in installed `huggingface_hub 2.4.0` (`file_download.py:1258, 1273, 1485, 1503`), but the package floor is `huggingface_hub>=0.23` (`pyproject.toml:27`); verify lock behaviour at `0.23` in Stage 1 (D29).
  - *Flat gradient buffer doubles trainable gradient memory (arithmetic):* allocating one full flat reduction buffer alongside `p.grad` doubles accelerator gradient memory for the trainable set. As arithmetic on `Timestral-209M`'s `209,312,256` parameters (§3.1, `docs/BENCHMARKS.md` §3), `full_ft` with `fp32` gradients (`4 B/param`) adds `209,312,256 × 4 B ≈ 798 MiB` (`~0.78 GiB`) per rank (not measured); chunked/bucketed flat-buffer reduction (or a DDP wrapper, D22) is the mitigation to evaluate if VRAM becomes tight.
  - *Host memory scales with `N`:* each rank is a separate OS process holding its own copy of `train_dataset` / `val_dataset`, its own `DataLoader` worker processes (`num_workers > 0`), and (if adopted in Stage 2) any CPU caches.
  - *`export()` read race:* because `export()` is collective-free (D31), returning on a non-zero rank does not guarantee rank 0 has finished writing files; multi-process scripts reading the export from non-zero ranks must synchronise explicitly (e.g. `dist.barrier()`).
  - *Open Stage 2 replay and cache decisions:* skip ordering with replay (leaning toward a replay-only pre-forward count collective, leaving Stage 1 unaffected), how to build the frozen-base teacher output cache across ranks, and whether a multi-process `head_only` feature cache is worth its host-RAM or communication trade-off.
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
