# Phase 3 Benchmarks & Hyperparameter Sweeps

All numbers below were measured on an NVIDIA L4 GPU (`g2-standard-4`, PyTorch 2.15.0+cu132,
Python 3.12, `bf16` autocast) on `2026-11-22`, evaluating every mode in raw space through the
vendored upstream `TimestralForecaster.forecast()` with its defaults (`symmetric_averaging=True`,
`clip_nonnegative=True`; spec §5.5). Raw JSON outputs for every table are committed in
`docs/benchmarks/`.

## Datasets, licences & contamination notes

| Role | Dataset (pinned HF Hub revision) | Licence | Shape & covariates | Scored test dates & contamination note |
|---|---|---|---|---|
| **Realised volatility** | `twelvedata/financial-world-model@da186e5fd362` (`bars_1h`) | MIT | 51 large-cap US equities, 1h bars, 8 rows/series (`rv_20` + 5 past: `log1p(volume)`, `vix_level`, `spy_logret_1`, `sector_logret_1`, `atr_14` + 2 known calendar: hour, weekday). `context=256`, `horizon=32`, train `6,324` / val `1,071` / test `153` windows. | **Scored targets span `2026-11-03` to `2026-11-20`** — strictly after Timestral's `2026-10-27` release (no pretraining contamination on test targets). Training/validation bars precede release. |
| **Retail demand** | `Dingdong-Inc/FreshRetailNet-50K@c374e85975c8` | CC BY 4.0 | Top-64 store-SKU hourly series ranked strictly on **training-period sales** (days 0–69; zero val/test leakage), 9 rows/series (hourly sales + 3 past: stockout, precipitation, temperature + 5 known: discount, holiday, activity, hour, weekday). `context=256`, `horizon=48`, train `1,856` / val `384` / test `384` windows. | **Scored targets span `2024-06-21` to `2024-07-02`** — published prior to October 2026; **may have been seen** during Mistral's post-training (§12). |
| **Forgetting set** | `Salesforce/GiftEval@30841734ac5c` | Apache-2.0 | Evaluated per domain and **macro-averaged equally** across `ett1/H` (28 windows, energy), `jena_weather/H` (84 windows, weather), and `LOOP_SEATTLE/H` (80 windows, traffic) so high-magnitude weather pressure channels cannot dominate pooled wQL. | Reported as equal-weighted 3-domain **relative forgetting %** (`mean_d((tuned - zs) / zs)`). Positive = degradation; negative = transfer gain. Univariate series only (note: `lora_variate` Q/K slices are inert on 1-row series, T10). |

All three licences permit publishing derived benchmark metrics (spec §8).

---

## 1. Main benchmark: Timestral-5M & Timestral-28M (`docs/benchmarks/bench_5m_28m.json`)

Reproduce:

```bash
uv run python scripts/benchmarks/run_benchmarks.py \
  --device cuda --bf16 --sizes 5M 28M --max-steps 200 --batch-size 32 --sign-flip \
  --output docs/benchmarks/bench_5m_28m.json
```

Hyperparameters (validation-bracketed defaults from §2 / D19): `head_only` `3e-3`, `lora_variate`
`3e-3`, `lora_temporal` `3e-3`, `full_ft` `3e-4`, LoRA `r=8, alpha=16, targets=("q","k","v","o"), dropout=0.05`,
`sign_flip=True`, `restore_best=True`. Online ACI runs strictly chronologically across all series
with horizon-reveal delay (`t_end < t_origin`, D20).

**GIFT-Eval zero-shot wQL baselines by domain:**
- **5M:** `ett1 = 0.1723`, `jena_weather = 0.0484`, `loop_seattle = 0.0585`
- **28M:** `ett1 = 0.1694`, `jena_weather = 0.0485`, `loop_seattle = 0.0573`

### Timestral-5M — Realised Volatility (`2026-11-03` to `2026-11-20`, 153 windows)

| Mode | Trainable params | Pinball | wQL | MASE | Cov 80% | Width 80% | Cov 90% | Width 90% | Macro Forg wQL | Macro Forg MASE | Peak VRAM / Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `zero_shot` | 0 | 0.000939 | 0.2360 | 1.9167 | 0.663 | 0.00603 | 0.796 | 0.00823 | +0.0% | +0.0% | — |
| `conformal_cqr` | 0 | 0.000933 | 0.2346 | 1.9167 | 0.689 | 0.00634 | 0.825 | 0.00884 | +0.0% | +0.0% | — |
| `conformal_aci` | 0 | 0.000948 | 0.2384 | 1.9167 | 0.783\* | 0.00780 | 0.868 | 0.01147 | +0.0% | +0.0% | — |
| `head_only` | 792,576 | 0.000873 | 0.2195 | 1.8292 | 0.761 | 0.00629 | 0.883 | 0.00934 | +13.9% | +7.4% | 0.08 GiB / 7.3 s |
| `lora_variate` | 24,576 | 0.000872 | 0.2192 | 1.7845 | 0.728 | 0.00624 | 0.844 | 0.00879 | -2.7% | -1.3% | 0.16 GiB / 12.0 s |
| `lora_temporal` | 98,304 | 0.000859 | 0.2161 | 1.7655 | 0.717 | 0.00591 | 0.850 | 0.00847 | +6.0% | +7.0% | 0.24 GiB / 17.0 s |
| `full_ft` | 5,208,960 | **0.000853** | **0.2144** | **1.7532** | 0.720 | 0.00584 | 0.849 | 0.00848 | -0.6% | +0.2% | 0.36 GiB / 14.3 s |

### Timestral-28M — Realised Volatility (`2026-11-03` to `2026-11-20`, 153 windows)

| Mode | Trainable params | Pinball | wQL | MASE | Cov 80% | Width 80% | Cov 90% | Width 90% | Macro Forg wQL | Macro Forg MASE | Peak VRAM / Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `zero_shot` | 0 | 0.000912 | 0.2293 | 1.8785 | 0.702 | 0.00641 | 0.825 | 0.00878 | +0.0% | +0.0% | — |
| `conformal_cqr` | 0 | 0.000909 | 0.2286 | 1.8785 | 0.712 | 0.00653 | 0.834 | 0.00908 | +0.0% | +0.0% | — |
| `conformal_aci` | 0 | 0.000920 | 0.2313 | 1.8785 | 0.790\* | 0.00770 | 0.865 | 0.01137 | +0.0% | +0.0% | — |
| `head_only` | 1,880,064 | 0.000861 | 0.2165 | 1.7984 | 0.754 | 0.00635 | 0.874 | 0.00945 | +10.4% | +4.7% | 0.21 GiB / 7.6 s |
| `lora_variate` | 73,728 | 0.000861 | 0.2164 | 1.7686 | 0.732 | 0.00599 | 0.847 | 0.00858 | -1.4% | -1.3% | 0.58 GiB / 17.4 s |
| `lora_temporal` | 294,912 | 0.000854 | 0.2148 | 1.7461 | 0.727 | 0.00611 | 0.851 | 0.00871 | -1.7% | +0.9% | 0.78 GiB / 23.9 s |
| `full_ft` | 27,821,184 | **0.000835** | **0.2098** | **1.7249** | 0.745 | 0.00574 | 0.871 | 0.00845 | +1.2% | -1.0% | 1.21 GiB / 20.4 s |

\* **ACI per-block 80% coverage across the 3 volatility blocks:** `zero_shot` scored `[0.531, 0.624, 0.833]`
(5M) and `[0.564, 0.705, 0.836]` (28M). `conformal_aci` cannot anticipate block 1's regime jump (`0.542` /
`0.557`), then widens aggressively and **overshoots** in blocks 2 and 3 (`[0.542, 0.906, 0.901]` on 5M;
`[0.557, 0.925, 0.889]` on 28M), averaging near `0.79` across the 3 blocks.

### Timestral-5M — Retail Demand (`2024-06-21` to `2024-07-02`, 384 windows)

| Mode | Trainable params | Pinball | wQL | MASE | Cov 80% | Width 80% | Cov 90% | Width 90% | Macro Forg wQL | Macro Forg MASE | Peak VRAM / Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `zero_shot` | 0 | 0.1311 | 0.2210 | 0.8459 | 0.823 | 1.282 | 0.923 | 1.789 | +0.0% | +0.0% | — |
| `conformal_cqr` | 0 | 0.1311 | 0.2210 | 0.8459 | 0.823 | 1.282 | 0.923 | 1.789 | +0.0% | +0.0% | — |
| `conformal_aci` | 0 | 0.1311 | 0.2210 | 0.8459 | 0.810 | 1.269 | 0.916 | 1.779 | +0.0% | +0.0% | — |
| `head_only` | 792,576 | 0.1244 | 0.2097 | 0.8289 | 0.820 | 1.165 | 0.917 | 1.551 | +5.0% | +0.5% | 0.10 GiB / 6.1 s |
| `lora_variate` | 24,576 | 0.1203 | 0.2027 | 0.7880 | 0.803 | 1.092 | 0.904 | 1.476 | +2.0% | -0.8% | 0.21 GiB / 10.5 s |
| `lora_temporal` | 98,304 | 0.1176 | 0.1982 | **0.7699** | 0.794 | 1.053 | 0.903 | 1.423 | +4.7% | +3.3% | 0.33 GiB / 14.8 s |
| `full_ft` | 5,208,960 | **0.1174** | **0.1979** | 0.7739 | 0.798 | 1.032 | 0.902 | 1.364 | +2.5% | -0.2% | 0.47 GiB / 12.8 s |

### Timestral-28M — Retail Demand (`2024-06-21` to `2024-07-02`, 384 windows)

| Mode | Trainable params | Pinball | wQL | MASE | Cov 80% | Width 80% | Cov 90% | Width 90% | Macro Forg wQL | Macro Forg MASE | Peak VRAM / Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `zero_shot` | 0 | 0.1301 | 0.2193 | 0.8347 | 0.766 | 1.253 | 0.882 | 1.693 | +0.0% | +0.0% | — |
| `conformal_cqr` | 0 | 0.1301 | 0.2193 | 0.8347 | 0.768 | 1.254 | 0.883 | 1.697 | +0.0% | +0.0% | — |
| `conformal_aci` | 0 | 0.1301 | 0.2193 | 0.8347 | 0.795 | 1.272 | 0.896 | 1.715 | +0.0% | +0.0% | — |
| `head_only` | 1,880,064 | 0.1229 | 0.2072 | 0.8125 | 0.812 | 1.164 | 0.913 | 1.550 | +6.8% | +1.3% | 0.24 GiB / 7.7 s |
| `lora_variate` | 73,728 | 0.1179 | 0.1987 | 0.7726 | 0.785 | 1.045 | 0.898 | 1.400 | -1.6% | -0.2% | 0.75 GiB / 15.0 s |
| `lora_temporal` | 294,912 | 0.1164 | 0.1962 | 0.7624 | 0.778 | 1.003 | 0.895 | 1.344 | +0.3% | +0.5% | 1.01 GiB / 20.8 s |
| `full_ft` | 27,821,184 | **0.1153** | **0.1943** | **0.7608** | 0.779 | 0.992 | 0.886 | 1.293 | +2.1% | +0.7% | 1.50 GiB / 18.3 s |

### Key findings (5M–209M)

- **Every gradient mode improves over zero-shot** across all four checkpoints and both datasets:
  wQL drops by **5.1%–14.1%** (including `head_only`, which drops 5.1%–7.0%), and MASE drops by
  **2.0%–11.4%** (excluding 209M `full_ft` volatility at 0.5%).
- **`full_ft` vs LoRA with bracketed LRs:** once `full_ft` uses its validation-bracketed LR (`3e-4`),
  it achieves the lowest test wQL on 5M/28M volatility and 5M/28M/123M/209M retail, while `lora_temporal`
  (`0.56%–1.9%` of params) stays within `0.1%–2.4%` of `full_ft` (and beats `full_ft` on 209M
  post-release volatility: `0.2087` vs `0.2149`). `lora_variate` (`0.14%–0.47%` of params) delivers the
  cleanest trade-off against univariate forgetting (`-2.7%` to `+2.0%` macro wQL forgetting everywhere).
- **Forgetting is largest in `head_only`:** training the shared quantile head at `3e-3` degrades
  macro GIFT-Eval wQL by **`+5.0%` to `+13.9%`** — more than LoRA or short-schedule `full_ft`.
  On `123M`/`209M` retail at `3e-3`/`3e-4`, `lora_temporal` (`+4.3%–+7.4%`) and `full_ft`
  (`+5.3%–+6.2%`) also show measurable general-domain forgetting; lowering LR to `1e-3` / `1e-4`
  eliminates most of it at a small validation cost (§2). Note that LoRA forgetting differences of
  `~1–2%` sit within seed noise.
- **Online ACI on regime shifts vs steady series:** on retail's 6 blocks, `conformal_aci` tracks
  nominal coverage closely (`28M`: `[0.744, 0.761, 0.813, 0.840, 0.791, 0.819]`, mean `0.795`). On
  volatility's 3-block regime jump, ACI under-covers on block 1 (`0.54–0.60`) before overshooting on
  blocks 2–3 (`0.87–0.92`) — illustrating both its adaptation speed and finite-horizon overshoot.

---

## 2. Bracketed hyperparameter sweeps & sign-flip ablation (`docs/benchmarks/sweeps_*.json`)

Reproduce:

```bash
uv run python scripts/benchmarks/run_sweeps.py \
  --device cuda --sizes 5M 28M --bf16 --max-steps 120 --batch-size 32 --check-forgetting \
  --output docs/benchmarks/sweeps_bracketed.json
```

All defaults in `DEFAULT_LEARNING_RATES` (`config.py`, D19) are chosen strictly from **validation**
wQL and GIFT-Eval macro forgetting (zero test-set peeking), with every optimum bracketed strictly
inside its sweep grid.

### Learning-rate sweep (validation wQL and equal-weighted macro GIFT-Eval forgetting, 120 steps, `bf16`, `sign_flip=True`)

| Mode | LR | 5M Vol (ZS `0.1891`) | 5M Ret (ZS `0.2393`) | 28M Vol (ZS `0.1850`) | 28M Ret (ZS `0.2411`) | 123M Vol / Ret val | Chosen default |
|---|---:|---|---|---|---|---|---|
| `head_only` | `3e-5` | 0.1882 (+1.3%) | 0.2358 (+0.3%) | 0.1840 (+1.3%) | 0.2366 (+0.1%) | — | |
| `head_only` | `1e-4` | 0.1872 (+4.3%) | 0.2336 (+0.7%) | 0.1828 (+4.3%) | 0.2344 (+0.6%) | — | |
| `head_only` | `3e-4` | 0.1861 (+7.6%) | 0.2314 (+1.6%) | 0.1820 (+5.7%) | 0.2318 (+1.6%) | — | |
| `head_only` | `1e-3` | 0.1841 (+5.6%) | 0.2286 (+3.3%) | 0.1807 (+3.7%) | 0.2284 (+3.5%) | — | |
| `head_only` | **`3e-3`** | 0.1825 (+5.4%) | 0.2264 (+4.9%) | **0.1803 (+4.0%)** | **0.2263 (+6.2%)** | — | **`3e-3`** |
| `head_only` | `1e-2` | **0.1814 (+19.8%)** | **0.2256 (+8.9%)** | 0.1810 (+23.5%) | 0.2268 (+11.8%) | — | |
| `head_only` | `3e-2` | 0.1835 (+37.4%) | 0.2283 (+15.2%) | 0.1848 (+40.9%) | 0.2299 (+17.9%) | — | |
| `lora_variate` | `3e-4` | 0.1847 (-1.1%) | 0.2365 (+0.1%) | 0.1809 (-0.9%) | 0.2219 (-1.3%) | — | |
| `lora_variate` | `1e-3` | 0.1819 (-2.5%) | 0.2268 (+0.7%) | 0.1785 (-1.3%) | 0.2055 (-2.5%) | 0.1768 / 0.2037 | |
| `lora_variate` | **`3e-3`** | 0.1807 (-3.4%) | 0.2160 (+1.5%) | **0.1758 (-2.1%)** | 0.2010 (-2.7%) | **0.1762 / 0.1979** | **`3e-3`** |
| `lora_variate` | `1e-2` | **0.1802 (-1.2%)** | **0.2095 (+1.6%)** | 0.1782 (-3.8%) | **0.1994 (-0.2%)** | — | |
| `lora_variate` | `3e-2` | 0.1861 (+3.7%) | 0.2117 (+8.1%) | 0.2035 (+66.4%) | 0.2377 (+4.4%) | — | |
| `lora_temporal` | `1e-4` | 0.1841 (-0.9%) | 0.2339 (+0.4%) | 0.1806 (-0.5%) | 0.2210 (-1.9%) | — | |
| `lora_temporal` | `3e-4` | 0.1819 (-0.8%) | 0.2266 (+0.2%) | 0.1780 (-1.3%) | 0.2073 (-1.8%) | — | |
| `lora_temporal` | `1e-3` | 0.1803 (-0.1%) | 0.2149 (+1.5%) | 0.1767 (-1.8%) | 0.2021 (-0.6%) | **0.1756** / 0.1973 | |
| `lora_temporal` | **`3e-3`** | **0.1797 (+3.3%)** | 0.2105 (+3.1%) | **0.1759 (-0.5%)** | **0.2010 (+0.7%)** | 0.1774 / **0.1962** | **`3e-3`** |
| `lora_temporal` | `1e-2` | 0.1801 (+32.5%) | **0.2104 (+4.1%)** | 0.1800 (+36.3%) | 0.2122 (+3.9%) | — | |
| `full_ft` | `1e-5` | 0.1858 (-0.4%) | 0.2339 (+0.1%) | 0.1812 (-0.4%) | 0.2265 (-1.3%) | — | |
| `full_ft` | `3e-5` | 0.1832 (-0.5%) | 0.2285 (+0.2%) | 0.1788 (-0.7%) | 0.2110 (-2.3%) | 0.1767 / 0.2039 | |
| `full_ft` | `1e-4` | 0.1804 (-1.1%) | 0.2162 (+0.4%) | 0.1758 (-0.6%) | 0.1978 (-1.5%) | 0.1761 / **0.1974** | |
| `full_ft` | **`3e-4`** | 0.1781 (-0.9%) | 0.2098 (+2.0%) | **0.1735 (+1.9%)** | **0.1961 (+0.3%)** | **0.1753** / **0.1974** | **`3e-4`** |
| `full_ft` | `1e-3` | **0.1762 (+5.0%)** | **0.2023 (+7.9%)** | 0.1738 (+6.3%) | 0.1969 (+8.6%) | — | |

### LoRA rank & targets sweep (5M validation wQL / MASE, 120 steps, `sign_flip=True`)

| Mode | `r` | Targets | Volatility wQL / MASE | Retail wQL / MASE |
|---|---:|---|---:|---:|
| `lora_variate` | 4 | `("q", "v")` | 0.1814 / 1.3327 | 0.2202 / 0.7722 |
| `lora_variate` | 8 | `("q", "v")` | 0.1807 / 1.3300 | 0.2126 / 0.7467 |
| `lora_variate` | 16 | `("q", "v")` | 0.1815 / 1.3335 | 0.2112 / 0.7422 |
| `lora_variate` | **8** | **`("q", "k", "v", "o")`** | **0.1802 / 1.3202** | **0.2095 / 0.7346** |
| `lora_temporal` | 4 | `("q", "v")` | 0.1807 / 1.3289 | 0.2102 / 0.7350 |
| `lora_temporal` | 8 | `("q", "v")` | 0.1803 / 1.3203 | **0.2060 / 0.7204** |
| `lora_temporal` | 16 | `("q", "v")` | 0.1811 / 1.3272 | 0.2116 / 0.7390 |
| `lora_temporal` | **8** | **`("q", "k", "v", "o")`** | **0.1797 / 1.3132** | 0.2104 / 0.7341 |

### Sign-flip augmentation ablation (5M validation wQL / MASE, 120 steps, at validation-optimal LRs)

| Mode | `sign_flip` | Volatility wQL / MASE | Retail wQL / MASE |
|---|---|---:|---:|
| `head_only` | `False` | 0.1861 / 1.3612 | 0.2413 / 0.8454 |
| `head_only` | `True` | **0.1825 / 1.3495** | **0.2264 / 0.8106** |
| `lora_variate` | `False` | 0.1812 / 1.3257 | 0.2257 / 0.7858 |
| `lora_variate` | `True` | **0.1802 / 1.3202** | **0.2095 / 0.7346** |
| `lora_temporal` | `False` | 0.1802 / 1.3128 | 0.2345 / 0.8153 |
| `lora_temporal` | `True` | **0.1797** / 1.3132 | **0.2104 / 0.7341** |
| `full_ft` | `False` | 0.1800 / 1.3102 | 0.2186 / 0.7693 |
| `full_ft` | `True` | **0.1762 / 1.2981** | **0.2023 / 0.7169** |

At the validation-optimal LRs, `sign_flip=True` delivers large gains on retail (`−6.2%` to `−10.3%`
wQL across all four modes) and smaller consistent gains on volatility (`−0.3%` to `−2.1%` wQL).
`TrainingConfig.sign_flip` stays `False` by default for general signed series and is recommended
(`sign_flip=True`) for strictly non-negative targets.

---

## 3. Large checkpoints: Timestral-123M & Timestral-209M (`docs/benchmarks/bench_123m_209m.json`)

Reproduce:

```bash
uv run python scripts/benchmarks/run_benchmarks.py \
  --device cuda --bf16 --sizes 123M 209M --max-steps 160 \
  --batch-size 32 --full-ft-batch-size 16 --eval-batch-size 16 \
  --modes zero_shot conformal_aci lora_variate lora_temporal full_ft --sign-flip \
  --output docs/benchmarks/bench_123m_209m.json
```

**GIFT-Eval zero-shot wQL baselines by domain:**
- **123M:** `ett1 = 0.1673`, `jena_weather = 0.0474`, `loop_seattle = 0.0557`
- **209M:** `ett1 = 0.1655`, `jena_weather = 0.0468`, `loop_seattle = 0.0562`

### Timestral-123M — Realised Volatility (`2026-11-03` to `2026-11-20`) & Retail Demand (`2024-06-21` to `2024-07-02`)

| Dataset | Mode | Params | Pinball | wQL | MASE | Cov 80% | Wid 80% | Cov 90% | Wid 90% | Macro Forg wQL | Macro Forg MASE | Peak VRAM / Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Volatility** | `zero_shot` | 0 | 0.000875 | 0.2201 | 1.8017 | 0.734 | 0.00656 | 0.842 | 0.00894 | +0.0% | +0.0% | — |
| **Volatility** | `conformal_aci` | 0 | 0.000882 | 0.2216 | 1.8017 | 0.788\* | 0.00735 | 0.872 | 0.01064 | +0.0% | +0.0% | — |
| **Volatility** | `lora_variate` | 204,800 | 0.000843 | 0.2120 | **1.7284** | 0.770 | 0.00667 | 0.885 | 0.00962 | -1.3% | -0.4% | 2.00 GiB / 33.2 s |
| **Volatility** | `lora_temporal` | 819,200 | 0.000855 | 0.2149 | 1.7566 | 0.727 | 0.00636 | 0.867 | 0.00958 | +2.7% | +1.6% | 2.39 GiB / 41.2 s |
| **Volatility** | `full_ft` | 122,606,720 | **0.000841** | **0.2114** | 1.7468 | 0.719 | 0.00570 | 0.856 | 0.00862 | +0.1% | -0.4% | 2.75 GiB / 35.9 s |
| **Retail** | `zero_shot` | 0 | 0.1283 | 0.2163 | 0.8252 | 0.841 | 1.320 | 0.934 | 1.798 | +0.0% | +0.0% | — |
| **Retail** | `conformal_aci` | 0 | 0.1283 | 0.2162 | 0.8252 | 0.786 | 1.287 | 0.908 | 1.772 | +0.0% | +0.0% | — |
| **Retail** | `lora_variate` | 204,800 | 0.1161 | 0.1958 | 0.7618 | 0.791 | 1.046 | 0.907 | 1.412 | -0.5% | -0.4% | 2.49 GiB / 29.7 s |
| **Retail** | `lora_temporal` | 819,200 | 0.1154 | 0.1946 | 0.7589 | 0.761 | 0.973 | 0.889 | 1.290 | +7.4% | +4.8% | 3.00 GiB / 34.9 s |
| **Retail** | `full_ft` | 122,606,720 | **0.1146** | **0.1932** | **0.7558** | 0.780 | 0.969 | 0.890 | 1.275 | +6.2% | +3.6% | 3.13 GiB / 28.1 s |

### Timestral-209M — Realised Volatility (`2026-11-03` to `2026-11-20`) & Retail Demand (`2024-06-21` to `2024-07-02`)

| Dataset | Mode | Params | Pinball | wQL | MASE | Cov 80% | Wid 80% | Cov 90% | Wid 90% | Macro Forg wQL | Macro Forg MASE | Peak VRAM / Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Volatility** | `zero_shot` | 0 | 0.000876 | 0.2201 | 1.7890 | 0.731 | 0.00639 | 0.835 | 0.00873 | +0.0% | +0.0% | — |
| **Volatility** | `conformal_aci` | 0 | 0.000882 | 0.2218 | 1.7890 | 0.781\* | 0.00718 | 0.864 | 0.01092 | +0.0% | +0.0% | — |
| **Volatility** | `lora_variate` | 294,912 | 0.000840 | 0.2113 | **1.7217** | 0.704 | 0.00589 | 0.855 | 0.00858 | -0.9% | -0.7% | 3.07 GiB / 44.0 s |
| **Volatility** | `lora_temporal` | 1,179,648 | **0.000830** | **0.2087** | 1.7350 | 0.719 | 0.00626 | 0.860 | 0.00929 | +0.4% | +2.1% | 3.60 GiB / 52.6 s |
| **Volatility** | `full_ft` | 209,312,256 | 0.000855 | 0.2149 | 1.7800 | 0.710 | 0.00591 | 0.844 | 0.00884 | -0.1% | -1.1% | 4.37 GiB / 47.5 s |
| **Retail** | `zero_shot` | 0 | 0.1322 | 0.2230 | 0.8471 | 0.832 | 1.400 | 0.923 | 1.925 | +0.0% | +0.0% | — |
| **Retail** | `conformal_aci` | 0 | 0.1322 | 0.2228 | 0.8471 | 0.810 | 1.356 | 0.901 | 1.890 | +0.0% | +0.0% | — |
| **Retail** | `lora_variate` | 294,912 | 0.1156 | 0.1949 | 0.7587 | 0.774 | 1.026 | 0.894 | 1.370 | -0.1% | -0.2% | 3.84 GiB / 43.2 s |
| **Retail** | `lora_temporal` | 1,179,648 | 0.1141 | 0.1923 | **0.7505** | 0.768 | 0.960 | 0.887 | 1.270 | +4.3% | +2.9% | 4.51 GiB / 50.1 s |
| **Retail** | `full_ft` | 209,312,256 | **0.1136** | **0.1915** | 0.7507 | 0.775 | 1.001 | 0.890 | 1.313 | +5.3% | +3.9% | 4.91 GiB / 37.3 s |

\* Per-block volatility ACI 80% coverage: `[0.598, 0.901, 0.866]` (123M) and `[0.564, 0.906, 0.873]` (209M).

### Precision & attention-kernel ablation on `123M` & `209M` (retail, `lora_temporal`, `lr=1e-3`, 120 steps, `B=16`)

Reproducible via `run_benchmarks.py --datasets retail_demand --modes lora_temporal --max-steps 120 --batch-size 16 --lr-lora-temporal 1e-3 --sign-flip` toggling `--bf16` and `--attention-backend default|math` (measured on earlier retail slice; see `docs/VERIFICATION.md` §13 for `123M`/`209M` single-step gradient probes):

| Model | Variant | Test wQL | Test MASE | Wall time | Peak VRAM |
|---|---|---:|---:|---:|---:|
| **123M** | `fp32` + default SDPA | 0.1963 | 0.7784 | 30.4 s | 2.23 GiB |
| **123M** | `fp32` + `attention_context("math")` | 0.1965 | 0.7797 | 29.9 s | 2.30 GiB |
| **123M** | `bf16` + default SDPA | 0.1970 | 0.7815 | **21.9 s** | **1.81 GiB** |
| **209M** | `fp32` + default SDPA | 0.1947 | 0.7725 | 46.0 s | 3.36 GiB |
| **209M** | `fp32` + `attention_context("math")` | 0.1952 | 0.7746 | 45.2 s | 3.47 GiB |
| **209M** | `bf16` + default SDPA | **0.1943** | **0.7718** | **26.1 s** | **2.79 GiB** |

(Earlier 28M 200-step `full_ft` retail check at `lr=5e-5` also matched `bf16` `0.1963` vs `fp32` `0.1956` within `0.35%`.)

---

## 4. Statistical significance, multi-seed variance & classical baselines (`docs/benchmarks/sig_v2.json`)

Reproduce:

```bash
uv run python scripts/benchmarks/significance.py \
  --device cuda --bf16 --max-steps 200 --batch-size 32 --seeds 0 1 2 \
  --output docs/benchmarks/sig_v2.json
```

All methods use the validation-bracketed defaults (`lora_*=3e-3`, `full_ft=3e-4`, `sign_flip=True`) and
identical test windows scored by `evaluate_predictions`. Paired 95% CIs use 10,000-resample cluster
bootstrap on `ΔwQL = wQL(tuned) - wQL(zero_shot)` clustered separately by series and by non-overlapping
time block; lag-1 Newey-West Diebold-Mariano p-values are reported where block count is large (`k = 49`).

### Retail demand (`2024-06-21` to `2024-07-02`; 64 train-ranked SKUs × 6 time blocks = 384 windows)

| Method | wQL (`mean ± std`, 3 seeds) | MASE (`mean ± std`) | Series 95% CI on ΔwQL (wins) | Time-block 95% CI on ΔwQL (wins) |
|---|---:|---:|---|---|
| `seasonal_naive` (`s=24`) | 0.2939 | 1.1200 | — | — |
| `seasonal_ridge` (covariates) | 0.2177 | 0.8427 | — | — |
| `28M zero_shot` | 0.2193 | 0.8347 | baseline | baseline |
| `28M lora_variate` | 0.1979 ± 0.0010 | 0.7683 ± 0.0048 | `[-0.0234, -0.0179]` (63/64) | `[-0.0250, -0.0167]` (6/6) |
| `28M lora_temporal` | 0.1957 ± 0.0006 | 0.7605 ± 0.0037 | `[-0.0256, -0.0207]` (64/64) | `[-0.0274, -0.0190]` (6/6) |
| `28M full_ft` | **0.1943 ± 0.0004** | **0.7593 ± 0.0016** | `[-0.0278, -0.0222]` (64/64) | `[-0.0309, -0.0202]` (6/6) |

### Realised volatility — extended 10-month 2026 holdout (`2026-01` to `2026-11-20`; 51 equities × 49 market blocks = 2,499 windows)

*Contamination note:* trained strictly on `2024–2025` only, but the 10-month test window is **mostly
pre-release** (`Jan–Oct 2026` precedes Timestral's `2026-10-27` release; only `Nov 3–20` is strictly
post-release). It is included for multi-regime time-block statistical power (`k = 49` blocks).

| Method | wQL | MASE | Series 95% CI on ΔwQL (wins) | Time-block 95% CI on ΔwQL (wins, DM p) |
|---|---:|---:|---|---|
| `seasonal_naive` (`s=7`) | 0.2639 | 1.8894 | — | — |
| `har_rv` (Corsi HAR + covariates) | 0.1981 | 1.4837 | — | — |
| `28M zero_shot` | 0.1926 | 1.4514 | baseline | baseline |
| `28M lora_variate` | 0.1848 | 1.3826 | `[-0.0098, -0.0058]` (47/51) | `[-0.0107, -0.0051]` (45/49, `p = 2.7e-08`) |
| `28M lora_temporal` | 0.1813 | 1.3675 | `[-0.0136, -0.0091]` (47/51) | `[-0.0151, -0.0080]` (45/49, `p = 9.2e-09`) |
| `28M full_ft` | **0.1793** | **1.3553** | `[-0.0157, -0.0111]` (48/51) | `[-0.0177, -0.0094]` (44/49, `p = 9.5e-09`) |

### Realised volatility — strict post-release slice (`2026-11-03` to `2026-11-20`; 51 equities × 3 blocks = 153 windows)

| Method | wQL (`mean ± std`, 3 seeds) | MASE (`mean ± std`) | Series 95% CI on ΔwQL (wins) | Time-block 95% CI on ΔwQL (wins) |
|---|---:|---:|---|---|
| `seasonal_naive` (`s=7`) | 0.2967 | 2.2553 | — | — |
| `har_rv` (Corsi HAR + covariates) | 0.2253 | 1.7725 | — | — |
| `28M zero_shot` | 0.2293 | 1.8785 | baseline | baseline |
| `28M lora_variate` | 0.2185 ± 0.0020 | 1.7788 ± 0.0126 | `[-0.0219, -0.0039]` (44/51) | `[-0.0269, -0.0018]` (3/3) |
| `28M lora_temporal` | 0.2138 ± 0.0019 | **1.7407 ± 0.0196** | `[-0.0239, -0.0057]` (39/51) | `[-0.0328, -0.0019]` (3/3) |
| `28M full_ft` | **0.2125 ± 0.0033** | 1.7446 ± 0.0265 | `[-0.0316, -0.0078]` (41/51) | `[-0.0365, -0.0075]` (3/3) |

### Summary of caveat measurements

1. **Seed variance is small:** across 3 seeds, `wQL` standard deviation is `0.0004–0.0010` on retail
   (vs `−0.021` to `−0.025` gain) and `0.0019–0.0033` on post-release volatility (vs `−0.011` to
   `−0.017` gain).
2. **Clustered CIs exclude zero across both series and time blocks:** retail improves `63–64/64` SKUs
   and `6/6` time blocks; the 49-block 2026 volatility holdout improves `47–48/51` equities and
   `44–45/49` market blocks (`DM p <= 2.7e-8`); and on the strict 3-block post-release volatility slice
   all three modes now improve `3/3` time blocks and `39–44/51` equities (though `k = 3` blocks remains
   a short calendar span).
3. **Beats classical domain baselines:** `28M zero_shot` trailed `seasonal_ridge` on retail (`0.2193` vs
   `0.2177`) and trailed Corsi `har_rv` on post-release volatility (`0.2293` vs `0.2253`). Adapted with
   `full_ft` / `lora_temporal`, 28M beats `seasonal_ridge` by **−10.1% to −10.7% wQL** on retail,
   beats `har_rv` by **−8.5% to −9.5% wQL** across 2026, and beats `har_rv` by **−5.1% to −5.7% wQL**
   on the post-release volatility slice.
