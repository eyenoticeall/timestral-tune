# Phase 3 Benchmarks & Hyperparameter Sweeps

All numbers below were measured on an NVIDIA L4 GPU (`g2-standard-4`, PyTorch 2.15.0+cu132,
Python 3.12, `bf16` autocast) on `2026-11-22`, evaluating every mode in raw space through the
vendored upstream `TimestralForecaster.forecast()` with its defaults (`symmetric_averaging=True`,
`clip_nonnegative=True`; spec §5.5). Raw JSON outputs for every table are committed in
`docs/benchmarks/`.

## Datasets, licences & contamination notes

| Role | Dataset (pinned HF Hub revision) | Licence | Shape & covariates | Scored test dates & contamination note |
|---|---|---|---|---|
| **Realised volatility** | `twelvedata/financial-world-model@da186e5fd362` (`bars_1h`, `target=rv_20`) | MIT | 51 large-cap US equities, 1h bars, 8 rows/series (`rv_20` + 5 past: `log1p(volume)`, `vix_level`, `spy_logret_1`, `sector_logret_1`, `atr_14` + 2 known calendar: hour, weekday). `context=256`, `horizon=32`, train `6,324` / val `1,071` / test `153` windows. | **Scored targets span `2026-11-03` to `2026-11-20`** — strictly after Timestral's `2026-10-27` release (no pretraining contamination on test targets). Training/validation bars precede release. |
| **Signed returns** | `twelvedata/financial-world-model@da186e5fd362` (`bars_1h`, `target=logret_20`) | MIT | Same 51 equities, covariates, and splits as realised volatility, with signed 20-bar log returns (`val` min `-0.225`, mean `+0.0009`, max `+0.247`). Used in §2.1 to test `sign_flip` on a signed target. | Same chronological split (`val` precedes release; test targets span `2026-11-03` to `2026-11-20`). |
| **Retail demand** | `Dingdong-Inc/FreshRetailNet-50K@c374e85975c8` | CC BY 4.0 | Top-64 store-SKU hourly series ranked strictly on **training-period sales** (days 0–69; zero val/test leakage), 9 rows/series (hourly sales + 3 past: stockout, precipitation, temperature + 5 known: discount, holiday, activity, hour, weekday). `context=256`, `horizon=48`, train `1,856` / val `384` / test `384` windows. | **Scored targets span `2024-06-21` to `2024-07-02`** — published prior to October 2026; **may have been seen** during Mistral's post-training (§12). |
| **Forgetting set** | `Salesforce/GiftEval@30841734ac5c` | Apache-2.0 | Evaluated per domain and **macro-averaged equally** across `ett1/H` (28 windows, energy), `jena_weather/H` (84 windows, weather), and `LOOP_SEATTLE/H` (80 windows, traffic) so high-magnitude weather pressure channels cannot dominate pooled wQL. | Reported as equal-weighted 3-domain **relative forgetting %** (`mean_d((tuned - zs) / zs)`). Positive = degradation; negative = transfer gain. Univariate series only (note: `lora_variate` Q/K slices are inert on 1-row series, T10). |

All licences permit publishing derived benchmark metrics (spec §8).

---

## 1. Main benchmark: Timestral-5M & Timestral-28M (`docs/benchmarks/bench_5m_28m.json`)

Reproduce:

```bash
uv run python scripts/benchmarks/run_benchmarks.py \
  --device cuda --bf16 --sizes 5M 28M --max-steps 200 --batch-size 32 \
  --output docs/benchmarks/bench_5m_28m.json
```

Uses the shipped defaults (D19): `sign_flip=True`, `head_only` `3e-3` (with dual-orientation `+x`/`-x`
frozen-feature cache), `lora_variate` `3e-3`, `lora_temporal` `1e-3`, `full_ft` `1e-4`, LoRA
`r=8, alpha=16, targets=("q","k","v","o")`, and `restore_best=True`. Benchmark runs pass
`dropout=0.05` explicitly (`LoraConfig.dropout` defaults to `0.0` in `config.py`). Online ACI runs
strictly chronologically across all series with horizon-reveal delay (`t_end < t_origin`, D20).

**GIFT-Eval zero-shot wQL baselines by domain:**
- **5M:** `ett1 = 0.1723`, `jena_weather = 0.0484`, `loop_seattle = 0.0585`
- **28M:** `ett1 = 0.1694`, `jena_weather = 0.0485`, `loop_seattle = 0.0573`

### Timestral-5M — Realised Volatility (`2026-11-03` to `2026-11-20`, 153 windows)

| Mode | Trainable params | Pinball | wQL | MASE | Cov 80% | Width 80% | Cov 90% | Width 90% | Macro Forg wQL | Macro Forg MASE | Peak VRAM / Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `zero_shot` | 0 | 0.000939 | 0.2360 | 1.9167 | 0.663 | 0.00603 | 0.796 | 0.00823 | +0.0% | +0.0% | — |
| `conformal_cqr` | 0 | 0.000933 | 0.2346 | 1.9167 | 0.689 | 0.00634 | 0.825 | 0.00884 | +0.0% | +0.0% | — |
| `conformal_aci` | 0 | 0.000948 | 0.2384 | 1.9167 | 0.783\* | 0.00780 | 0.868 | 0.01147 | +0.0% | +0.0% | — |
| `head_only` | 792,576 | 0.000873 | 0.2195 | 1.8292 | 0.761 | 0.00629 | 0.884 | 0.00934 | +14.0% | +7.5% | 0.08 GiB / 12.5 s |
| `lora_variate` | 24,576 | **0.000872** | **0.2192** | **1.7845** | 0.728 | 0.00624 | 0.844 | 0.00879 | -2.7% | -1.3% | 0.16 GiB / 12.9 s |
| `lora_temporal` | 98,304 | 0.000880 | 0.2213 | 1.8007 | 0.719 | 0.00628 | 0.837 | 0.00878 | -1.0% | +1.4% | 0.24 GiB / 17.7 s |
| `full_ft` | 5,208,960 | 0.000876 | 0.2202 | 1.8009 | 0.726 | 0.00622 | 0.846 | 0.00882 | -1.0% | -1.0% | 0.36 GiB / 15.2 s |

### Timestral-28M — Realised Volatility (`2026-11-03` to `2026-11-20`, 153 windows)

| Mode | Trainable params | Pinball | wQL | MASE | Cov 80% | Width 80% | Cov 90% | Width 90% | Macro Forg wQL | Macro Forg MASE | Peak VRAM / Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `zero_shot` | 0 | 0.000912 | 0.2293 | 1.8785 | 0.702 | 0.00641 | 0.825 | 0.00878 | +0.0% | +0.0% | — |
| `conformal_cqr` | 0 | 0.000909 | 0.2286 | 1.8785 | 0.712 | 0.00653 | 0.834 | 0.00908 | +0.0% | +0.0% | — |
| `conformal_aci` | 0 | 0.000920 | 0.2313 | 1.8785 | 0.790\* | 0.00770 | 0.865 | 0.01137 | +0.0% | +0.0% | — |
| `head_only` | 1,880,064 | 0.000861 | 0.2165 | 1.7985 | 0.755 | 0.00635 | 0.873 | 0.00943 | +10.4% | +4.6% | 0.21 GiB / 13.5 s |
| `lora_variate` | 73,728 | 0.000861 | 0.2164 | 1.7686 | 0.732 | 0.00599 | 0.847 | 0.00858 | -1.4% | -1.3% | 0.58 GiB / 18.3 s |
| `lora_temporal` | 294,912 | 0.000870 | 0.2188 | 1.7826 | 0.733 | 0.00608 | 0.855 | 0.00866 | -2.0% | -1.2% | 0.78 GiB / 24.7 s |
| `full_ft` | 27,821,184 | **0.000860** | **0.2163** | **1.7669** | 0.729 | 0.00582 | 0.860 | 0.00858 | -0.7% | -1.5% | 1.21 GiB / 21.4 s |

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
| `head_only` | 792,576 | 0.1244 | 0.2098 | 0.8291 | 0.821 | 1.168 | 0.918 | 1.554 | +5.0% | +0.5% | 0.10 GiB / 4.4 s |
| `lora_variate` | 24,576 | 0.1203 | 0.2027 | 0.7880 | 0.803 | 1.092 | 0.904 | 1.476 | +2.0% | -0.8% | 0.21 GiB / 11.2 s |
| `lora_temporal` | 98,304 | 0.1207 | 0.2035 | 0.7883 | 0.807 | 1.084 | 0.908 | 1.471 | +1.8% | +1.3% | 0.33 GiB / 15.7 s |
| `full_ft` | 5,208,960 | **0.1193** | **0.2011** | **0.7847** | 0.803 | 1.080 | 0.908 | 1.449 | +0.4% | -1.1% | 0.47 GiB / 13.5 s |

### Timestral-28M — Retail Demand (`2024-06-21` to `2024-07-02`, 384 windows)

| Mode | Trainable params | Pinball | wQL | MASE | Cov 80% | Width 80% | Cov 90% | Width 90% | Macro Forg wQL | Macro Forg MASE | Peak VRAM / Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `zero_shot` | 0 | 0.1301 | 0.2193 | 0.8347 | 0.766 | 1.253 | 0.882 | 1.693 | +0.0% | +0.0% | — |
| `conformal_cqr` | 0 | 0.1301 | 0.2193 | 0.8347 | 0.768 | 1.254 | 0.883 | 1.697 | +0.0% | +0.0% | — |
| `conformal_aci` | 0 | 0.1301 | 0.2193 | 0.8347 | 0.795 | 1.272 | 0.896 | 1.715 | +0.0% | +0.0% | — |
| `head_only` | 1,880,064 | 0.1229 | 0.2072 | 0.8125 | 0.815 | 1.164 | 0.913 | 1.546 | +6.9% | +1.3% | 0.24 GiB / 5.4 s |
| `lora_variate` | 73,728 | 0.1179 | 0.1987 | 0.7726 | 0.785 | 1.045 | 0.898 | 1.400 | -1.6% | -0.2% | 0.75 GiB / 15.6 s |
| `lora_temporal` | 294,912 | 0.1172 | 0.1976 | 0.7678 | 0.780 | 1.036 | 0.899 | 1.401 | +0.8% | +1.8% | 1.01 GiB / 21.7 s |
| `full_ft` | 27,821,184 | **0.1155** | **0.1946** | **0.7596** | 0.781 | 1.006 | 0.894 | 1.333 | -1.2% | -1.0% | 1.50 GiB / 18.9 s |

### Key findings across all four sizes (`5M`, `28M`, `123M`, `209M`)

- **Every gradient mode improves over zero-shot** across all four checkpoints and both datasets:
  wQL drops by **3.0%–14.0%** (lowest: `123M` volatility `full_ft` at 3.0%; highest: `209M` retail
  `full_ft` at 14.0%) and MASE drops by **1.6%–11.4%** (lowest: `209M` volatility `full_ft` at 1.6%;
  highest: `209M` retail `full_ft` at 11.4%).
- **Mode winners across the 8 `(size × dataset)` cells:** `full_ft` achieves the lowest test wQL in
  **5 of 8** cells (all four retail cells plus `28M` volatility), `lora_temporal` leads **2 of 8**
  (`123M` volatility `0.2111` and `209M` volatility `0.2084`), and `lora_variate` leads **1 of 8**
  (`5M` volatility `0.2192`).
- **Forgetting stays low for LoRA and `full_ft` at the shipped defaults (`1e-3` / `1e-4`), and is
  largest in `head_only`:** training the shared quantile head at `3e-3` degrades macro GIFT-Eval wQL
  by **`+5.0%` to `+14.0%`**. By contrast, `lora_variate` stays between **`-2.7%` and `+2.0%`**,
  `full_ft` (`1e-4`) stays between **`-1.6%` and `+2.5%`**, and `lora_temporal` (`1e-3`) stays
  between **`-2.0%` and `+4.6%`** across all 8 cells (avoiding the `+19.6%`–`+37.2%` worst-seed
  forgetting cliff seen at `3e-3`, §2.1).
- **Online ACI on regime shifts vs steady series:** on retail's 6 blocks, `conformal_aci` tracks
  nominal coverage closely (`28M`: `[0.744, 0.761, 0.813, 0.840, 0.791, 0.819]`, mean `0.795`). On
  volatility's 3-block regime jump, ACI under-covers on block 1 (`0.54–0.60`) before overshooting on
  blocks 2–3 (`0.87–0.92`).

---

## 2. Hyperparameter sweeps & default-selection decisions

### 2.1 Large-checkpoint 3-seed decision sweep (`docs/benchmarks/sweeps_large_decisions.json`)

Reproduce:

```bash
uv run python scripts/benchmarks/run_sweeps.py \
  --device cuda --sizes 123M 209M --bf16 --batch-size 16 --eval-batch-size 32 \
  --datasets realised_volatility retail_demand signed_returns \
  --modes lora_temporal full_ft --mode-lrs lora_temporal=1e-3,3e-3 full_ft=1e-4,3e-4 \
  --sign-flip-modes off on --seeds 0 1 2 --check-forgetting --skip-lora-sweep \
  --output docs/benchmarks/sweeps_large_decisions.json
```

All three default decisions (D19) were made strictly from **validation wQL** and **worst-seed macro
GIFT-Eval forgetting** across `seeds = [0, 1, 2]` on `123M` and `209M` (144 runs total; zero test
results consulted).

#### Timestral-123M (validation wQL, seeds `0 / 1 / 2`, and macro forgetting `0 / 1 / 2`)

| Dataset (ZS val wQL) | Mode | LR | `sign_flip` | Val wQL (`s0 / s1 / s2`; mean, spread) | Macro Forg (`s0 / s1 / s2`; worst) |
|---|---|---:|---|---|---|
| **Volatility** (`0.1834`) | `lora_temporal` | `1e-3` | `False` | `0.1787 / 0.1776 / 0.1778` (0.1781, 0.0011) | `+0.3% / +2.3% / -0.2%` (`+2.3%`) |
| **Volatility** (`0.1834`) | `lora_temporal` | `3e-3` | `False` | `0.1795 / 0.1788 / 0.1806` (0.1796, 0.0018) | `+2.1% / +8.0% / +8.8%` (`+8.8%`) |
| **Volatility** (`0.1834`) | `lora_temporal` | `1e-3` | **`True`** | **`0.1756 / 0.1749 / 0.1741`** (0.1749, 0.0015) | **`-0.4% / +0.5% / -0.2%` (`+0.5%`)** |
| **Volatility** (`0.1834`) | `lora_temporal` | `3e-3` | `True` | `0.1774 / 0.1755 / 0.1758` (0.1762, 0.0020) | `+3.5% / +2.8% / +1.9%` (`+3.5%`) |
| **Volatility** (`0.1834`) | `full_ft` | `1e-4` | `False` | `0.1775 / 0.1769 / 0.1765` (0.1770, 0.0010) | `-1.3% / -0.8% / -1.5%` (`-0.8%`) |
| **Volatility** (`0.1834`) | `full_ft` | `3e-4` | `False` | `0.1779 / 0.1772 / 0.1799` (0.1783, 0.0027) | `-2.0% / -0.4% / -0.7%` (`-0.4%`) |
| **Volatility** (`0.1834`) | `full_ft` | `1e-4` | **`True`** | **`0.1761 / 0.1745 / 0.1727`** (0.1745, 0.0034) | **`-0.8% / -1.2% / -1.2%` (`-0.8%`)** |
| **Volatility** (`0.1834`) | `full_ft` | `3e-4` | `True` | `0.1753 / 0.1746 / 0.1739` (0.1746, 0.0014) | `+1.0% / -0.4% / -0.1%` (`+1.0%`) |
| **Retail** (`0.2346`) | `lora_temporal` | `1e-3` | `False` | `0.2156 / 0.2132 / 0.2114` (0.2134, 0.0042) | `+3.1% / +3.5% / +2.6%` (`+3.5%`) |
| **Retail** (`0.2346`) | `lora_temporal` | `3e-3` | `False` | `0.2145 / 0.2145 / 0.2209` (0.2166, 0.0064) | `+3.6% / +4.0% / +7.6%` (`+7.6%`) |
| **Retail** (`0.2346`) | `lora_temporal` | `1e-3` | **`True`** | `0.1973 / 0.1999 / 0.2004` (0.1992, 0.0031) | `+5.5% / +5.3% / +3.4%` (`+5.5%`) |
| **Retail** (`0.2346`) | `lora_temporal` | `3e-3` | `True` | `0.1962 / 0.1996 / 0.1951` (0.1970, 0.0045) | `+5.8% / +5.3% / +5.6%` (`+5.8%`) |
| **Retail** (`0.2346`) | `full_ft` | `1e-4` | `False` | `0.2077 / 0.2084 / 0.2031` (0.2064, 0.0053) | `+0.8% / +0.6% / +1.1%` (`+1.1%`) |
| **Retail** (`0.2346`) | `full_ft` | `3e-4` | `False` | `0.2081 / 0.2086 / 0.2111` (0.2093, 0.0030) | `+3.4% / +2.8% / +3.4%` (`+3.4%`) |
| **Retail** (`0.2346`) | `full_ft` | `1e-4` | **`True`** | `0.1974 / 0.1962 / 0.1957` (0.1965, 0.0017) | **`+1.5% / +2.1% / +1.1%` (`+2.1%`)** |
| **Retail** (`0.2346`) | `full_ft` | `3e-4` | `True` | `0.1974 / 0.1949 / 0.1938` (0.1954, 0.0036) | `+4.0% / +3.8% / +2.8%` (`+4.0%`) |
| **Signed returns** (`0.6550`) | `lora_temporal` | `1e-3` | `False` | `0.6165 / 0.6164 / 0.6189` (0.6173, 0.0025) | `+1.7% / +1.6% / -0.8%` (`+1.7%`) |
| **Signed returns** (`0.6550`) | `lora_temporal` | `3e-3` | `False` | `0.6156 / 0.6156 / 0.6275` (0.6196, 0.0120) | `+8.0% / +30.8% / +13.9%` (**`+30.8%`**) |
| **Signed returns** (`0.6550`) | `lora_temporal` | `1e-3` | **`True`** | **`0.6158 / 0.6171 / 0.6179`** (0.6169, 0.0022) | **`-0.6% / +0.7% / -0.5%` (`+0.7%`)** |
| **Signed returns** (`0.6550`) | `lora_temporal` | `3e-3` | `True` | `0.6141 / 0.6165 / 0.6174` (0.6160, 0.0033) | `+9.6% / +23.8% / +16.6%` (**`+23.8%`**) |
| **Signed returns** (`0.6550`) | `full_ft` | `1e-4` | `False` | `0.6155 / 0.6168 / 0.6181` (0.6168, 0.0025) | `-2.0% / -2.8% / -2.3%` (`-2.0%`) |
| **Signed returns** (`0.6550`) | `full_ft` | `3e-4` | `False` | `0.6124 / 0.6182 / 0.6189` (0.6165, 0.0065) | `-0.4% / +0.4% / +0.7%` (`+0.7%`) |
| **Signed returns** (`0.6550`) | `full_ft` | `1e-4` | **`True`** | `0.6190 / 0.6166 / 0.6172` (0.6176, 0.0024) | **`-1.9% / -2.3% / -2.6%` (`-1.9%`)** |
| **Signed returns** (`0.6550`) | `full_ft` | `3e-4` | `True` | `0.6136 / 0.6145 / 0.6179` (0.6153, 0.0042) | `+0.4% / -1.1% / +0.2%` (`+0.4%`) |

#### Timestral-209M (validation wQL, seeds `0 / 1 / 2`, and macro forgetting `0 / 1 / 2`)

| Dataset (ZS val wQL) | Mode | LR | `sign_flip` | Val wQL (`s0 / s1 / s2`; mean, spread) | Macro Forg (`s0 / s1 / s2`; worst) |
|---|---|---:|---|---|---|
| **Volatility** (`0.1841`) | `lora_temporal` | `1e-3` | `False` | `0.1805 / 0.1781 / 0.1805` (0.1797, 0.0024) | `+1.0% / -1.2% / +0.9%` (`+1.0%`) |
| **Volatility** (`0.1841`) | `lora_temporal` | `3e-3` | `False` | `0.1808 / 0.1784 / 0.1797` (0.1796, 0.0025) | `+5.2% / +0.2% / +4.1%` (`+5.2%`) |
| **Volatility** (`0.1841`) | `lora_temporal` | `1e-3` | **`True`** | **`0.1745 / 0.1736 / 0.1742`** (0.1741, 0.0009) | **`-1.3% / -2.5% / -1.5%` (`-1.3%`)** |
| **Volatility** (`0.1841`) | `lora_temporal` | `3e-3` | `True` | `0.1757 / 0.1744 / 0.1736` (0.1746, 0.0021) | `+3.4% / +2.5% / -0.7%` (`+3.4%`) |
| **Volatility** (`0.1841`) | `full_ft` | `1e-4` | `False` | `0.1787 / 0.1773 / 0.1785` (0.1782, 0.0014) | `-0.3% / -1.2% / -0.7%` (`-0.3%`) |
| **Volatility** (`0.1841`) | `full_ft` | `3e-4` | `False` | `0.1794 / 0.1770 / 0.1779` (0.1781, 0.0024) | `-0.7% / -0.7% / +0.2%` (`+0.2%`) |
| **Volatility** (`0.1841`) | `full_ft` | `1e-4` | **`True`** | **`0.1746 / 0.1731 / 0.1725`** (0.1734, 0.0022) | **`-1.7% / -1.0% / -1.3%` (`-1.0%`)** |
| **Volatility** (`0.1841`) | `full_ft` | `3e-4` | `True` | `0.1749 / 0.1735 / 0.1727` (0.1737, 0.0021) | `-0.9% / -0.9% / -0.4%` (`-0.4%`) |
| **Retail** (`0.2401`) | `lora_temporal` | `1e-3` | `False` | `0.2077 / 0.2070 / 0.2100` (0.2082, 0.0030) | `+2.5% / +2.8% / +3.3%` (`+3.3%`) |
| **Retail** (`0.2401`) | `lora_temporal` | `3e-3` | `False` | `0.2114 / 0.2202 / 0.2089` (0.2135, 0.0114) | `+2.9% / +4.0% / +2.8%` (`+4.0%`) |
| **Retail** (`0.2401`) | `lora_temporal` | `1e-3` | **`True`** | `0.1946 / 0.1977 / 0.1944` (0.1956, 0.0034) | **`+4.1% / +4.1% / +1.0%` (`+4.1%`)** |
| **Retail** (`0.2401`) | `lora_temporal` | `3e-3` | `True` | `0.1938 / 0.1981 / 0.1943` (0.1954, 0.0043) | `+4.8% / +4.1% / +3.3%` (`+4.8%`) |
| **Retail** (`0.2401`) | `full_ft` | `1e-4` | `False` | `0.2015 / 0.2067 / 0.2034` (0.2039, 0.0053) | `+0.5% / +0.7% / +0.5%` (`+0.7%`) |
| **Retail** (`0.2401`) | `full_ft` | `3e-4` | `False` | `0.2066 / 0.2087 / 0.2060` (0.2071, 0.0027) | `+2.9% / +2.8% / +1.3%` (`+2.9%`) |
| **Retail** (`0.2401`) | `full_ft` | `1e-4` | **`True`** | **`0.1894 / 0.1930 / 0.1931`** (0.1918, 0.0037) | **`+1.6% / +1.4% / -0.2%` (`+1.6%`)** |
| **Retail** (`0.2401`) | `full_ft` | `3e-4` | `True` | `0.1922 / 0.1945 / 0.1930` (0.1932, 0.0023) | `+4.2% / +4.0% / +3.5%` (`+4.2%`) |
| **Signed returns** (`0.6531`) | `lora_temporal` | `1e-3` | `False` | `0.6144 / 0.6153 / 0.6185` (0.6160, 0.0041) | `-1.5% / -0.3% / -0.5%` (`-0.3%`) |
| **Signed returns** (`0.6531`) | `lora_temporal` | `3e-3` | `False` | `0.6214 / 0.6158 / 0.6177` (0.6183, 0.0056) | `+2.5% / +6.6% / +4.7%` (`+6.6%`) |
| **Signed returns** (`0.6531`) | `lora_temporal` | `1e-3` | **`True`** | **`0.6153 / 0.6141 / 0.6168`** (0.6154, 0.0028) | **`+0.3% / -0.8% / -0.8%` (`+0.3%`)** |
| **Signed returns** (`0.6531`) | `lora_temporal` | `3e-3` | `True` | `0.6161 / 0.6129 / 0.6179` (0.6156, 0.0050) | `+5.9% / +19.6% / +3.0%` (**`+19.6%`**) |
| **Signed returns** (`0.6531`) | `full_ft` | `1e-4` | `False` | `0.6125 / 0.6149 / 0.6186` (0.6153, 0.0062) | `-1.8% / -1.6% / -1.4%` (`-1.4%`) |
| **Signed returns** (`0.6531`) | `full_ft` | `3e-4` | `False` | `0.6084 / 0.6115 / 0.6187` (0.6129, 0.0103) | `+1.3% / +0.2% / +0.7%` (`+1.3%`) |
| **Signed returns** (`0.6531`) | `full_ft` | `1e-4` | **`True`** | `0.6131 / 0.6119 / 0.6155` (0.6135, 0.0036) | **`-1.2% / -1.2% / -1.1%` (`-1.1%`)** |
| **Signed returns** (`0.6531`) | `full_ft` | `3e-4` | `True` | `0.6118 / 0.6101 / 0.6154` (0.6124, 0.0053) | `+1.1% / +0.6% / +1.0%` (`+1.1%`) |

### 2.2 Unified 5M & 28M full-grid sweep (`docs/benchmarks/sweeps_bracketed.json`)

Reproduce:

```bash
uv run python scripts/benchmarks/run_sweeps.py \
  --device cuda --sizes 5M 28M --bf16 --max-steps 120 --batch-size 32 \
  --sign-flip-modes on --check-forgetting --output docs/benchmarks/sweeps_bracketed.json
```

All `head_only` rates (`3e-5` to `3e-2`), LoRA rank/target rows, and sign-flip ablations live in this
single unified JSON file.

| Mode | LR | 5M Vol (ZS `0.1891`) | 5M Ret (ZS `0.2393`) | 28M Vol (ZS `0.1850`) | 28M Ret (ZS `0.2411`) | Shipped default |
|---|---:|---|---|---|---|---|
| `head_only` | `3e-5` | 0.1882 (+1.3%) | 0.2358 (+0.3%) | 0.1840 (+1.3%) | 0.2366 (+0.1%) | |
| `head_only` | `1e-4` | 0.1872 (+4.3%) | 0.2336 (+0.7%) | 0.1828 (+4.3%) | 0.2344 (+0.6%) | |
| `head_only` | `3e-4` | 0.1861 (+7.6%) | 0.2314 (+1.6%) | 0.1820 (+5.7%) | 0.2318 (+1.6%) | |
| `head_only` | `1e-3` | 0.1841 (+5.6%) | 0.2286 (+3.3%) | 0.1807 (+3.7%) | 0.2284 (+3.5%) | |
| `head_only` | **`3e-3`** | 0.1824 (+5.4%) | 0.2264 (+4.9%) | **0.1803 (+4.0%)** | **0.2263 (+6.2%)** | **`3e-3`** |
| `head_only` | `1e-2` | **0.1814 (+19.8%)** | **0.2256 (+8.9%)** | 0.1810 (+23.5%) | 0.2268 (+11.8%) | |
| `head_only` | `3e-2` | 0.1834 (+36.7%) | 0.2283 (+15.2%) | 0.1848 (+40.9%) | 0.2299 (+17.9%) | |
| `lora_variate` | `3e-4` | 0.1847 (-1.1%) | 0.2365 (+0.1%) | 0.1809 (-0.9%) | 0.2219 (-1.3%) | |
| `lora_variate` | `1e-3` | 0.1819 (-2.5%) | 0.2268 (+0.7%) | 0.1785 (-1.3%) | 0.2055 (-2.5%) | |
| `lora_variate` | **`3e-3`** | 0.1807 (-3.4%) | 0.2160 (+1.5%) | **0.1758 (-2.1%)** | 0.2010 (-2.7%) | **`3e-3`** |
| `lora_variate` | `1e-2` | **0.1802 (-1.2%)** | **0.2095 (+1.6%)** | 0.1782 (-3.8%) | **0.1994 (-0.2%)** | |
| `lora_variate` | `3e-2` | 0.1861 (+3.7%) | 0.2117 (+8.1%) | 0.2035 (+66.4%) | 0.2377 (+4.4%) | |
| `lora_temporal` | `1e-4` | 0.1841 (-0.9%) | 0.2339 (+0.4%) | 0.1806 (-0.5%) | 0.2210 (-1.9%) | |
| `lora_temporal` | `3e-4` | 0.1819 (-0.8%) | 0.2266 (+0.2%) | 0.1780 (-1.3%) | 0.2073 (-1.8%) | |
| `lora_temporal` | **`1e-3`** | 0.1803 (-0.1%) | 0.2149 (+1.5%) | 0.1767 (-1.8%) | 0.2021 (-0.6%) | **`1e-3`** (D19) |
| `lora_temporal` | `3e-3` | **0.1797 (+3.3%)** | 0.2105 (+3.1%) | **0.1759 (-0.5%)** | **0.2010 (+0.7%)** | |
| `lora_temporal` | `1e-2` | 0.1801 (+32.5%) | **0.2104 (+4.1%)** | 0.1800 (+36.3%) | 0.2122 (+3.9%) | |
| `full_ft` | `1e-5` | 0.1858 (-0.4%) | 0.2339 (+0.1%) | 0.1812 (-0.4%) | 0.2265 (-1.3%) | |
| `full_ft` | `3e-5` | 0.1832 (-0.5%) | 0.2285 (+0.2%) | 0.1788 (-0.7%) | 0.2110 (-2.3%) | |
| `full_ft` | **`1e-4`** | 0.1804 (-1.1%) | 0.2162 (+0.4%) | 0.1758 (-0.6%) | 0.1978 (-1.5%) | **`1e-4`** (D19) |
| `full_ft` | `3e-4` | 0.1781 (-0.9%) | 0.2098 (+2.0%) | **0.1735 (+1.9%)** | **0.1961 (+0.3%)** | |
| `full_ft` | `1e-3` | **0.1762 (+5.0%)** | **0.2023 (+7.9%)** | 0.1738 (+6.3%) | 0.1969 (+8.6%) | |

*Bracketing & size-tradeoff note:* on `28M`, all four modes are strictly bracketed inside the grid
(`head_only` peaks at `3e-3`, `lora_variate` at `3e-3`–`1e-2`, `lora_temporal` at `3e-3`, `full_ft`
at `3e-4`). On `5M`, `full_ft` is still improving at the top grid edge `1e-3` (with `+5.0%–+7.9%`
forgetting). We keep `full_ft=1e-4` as the single cross-size default (D19), and **all §1 tables use
`1e-4`**. As a user recommendation (not a default), setting `learning_rate=3e-4` for `full_ft` on
`5M` and `28M` gave lower error (three seeds on `5M`, one seed on `28M` — where the gain is the ~3%
drop on volatility, while `28M` retail is essentially flat at `0.1943` vs `0.1946`), at the cost of
roughly two to three percentage points more forgetting. Do not extend this recommendation to `123M`
or `209M`, where `3e-4` ties or trails `1e-4` on validation and `1e-4` forgets less (§2.1).

**Evidence for the `5M` / `28M` `full_ft=3e-4` recommendation** (validation in committed
`docs/benchmarks/sweeps_bracketed.json`, one seed: `5M` retail `0.2098` vs `0.2162`, `28M`
volatility `0.1735` vs `0.1758`; test wQL and macro forgetting below):

| Model, dataset | `3e-4` test wQL | `1e-4` test wQL | Macro forgetting (`3e-4` vs `1e-4`) | Provenance |
|---|---:|---:|---|---|
| `5M` volatility | `0.2149 / 0.2162 / 0.2165` | `0.2203 / 0.2201 / 0.2204` | `-1.1% / -0.4% / +0.1%` vs `-1.0% / -0.5% / -1.5%` | Apple GPU `fp32`, 3 seeds (`docs/benchmarks/bench_5m_full_ft_3e4_vs_1e4.json`) |
| `5M` retail | `0.1983 / 0.1983 / 0.1995` | `0.2014 / 0.2012 / 0.2022` | `+2.8% / +2.0% / +2.1%` vs `+0.4% / +0.5% / +0.6%` | Apple GPU `fp32`, 3 seeds (`docs/benchmarks/bench_5m_full_ft_3e4_vs_1e4.json`) |
| `28M` volatility | `0.2098` | `0.2163` | `+1.2%` vs `-0.7%` | L4 `bf16`, 1 seed each (`3e-4` from `docs/benchmarks/bench_5m_28m.json` at commit `3fe64e0`; `1e-4` from `HEAD`) |
| `28M` retail | `0.1943` | `0.1946` | `+2.1%` vs `-1.2%` | L4 `bf16`, 1 seed each (`3e-4` from `docs/benchmarks/bench_5m_28m.json` at commit `3fe64e0`; `1e-4` from `HEAD`) |

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

### Sign-flip augmentation ablation (5M validation wQL / MASE, 120 steps, at per-mode grid optima)

| Mode | `sign_flip` | Volatility wQL / MASE | Retail wQL / MASE |
|---|---|---:|---:|
| `head_only` | `False` | 0.1861 / 1.3612 | 0.2413 / 0.8454 |
| `head_only` | `True` | **0.1814 / 1.3410** | **0.2256 / 0.8033** |
| `lora_variate` | `False` | 0.1812 / 1.3257 | 0.2257 / 0.7858 |
| `lora_variate` | `True` | **0.1802 / 1.3202** | **0.2095 / 0.7346** |
| `lora_temporal` | `False` | 0.1802 / 1.3128 | 0.2345 / 0.8153 |
| `lora_temporal` | `True` | **0.1797** / 1.3132 | **0.2104 / 0.7341** |
| `full_ft` | `False` | 0.1800 / 1.3102 | 0.2186 / 0.7693 |
| `full_ft` | `True` | **0.1762 / 1.2981** | **0.2023 / 0.7169** |

---

## 3. Large checkpoints: Timestral-123M & Timestral-209M (`docs/benchmarks/bench_123m_209m.json`)

Reproduce:

```bash
uv run python scripts/benchmarks/run_benchmarks.py \
  --device cuda --bf16 --sizes 123M 209M --max-steps 160 \
  --batch-size 32 --full-ft-batch-size 16 --eval-batch-size 16 \
  --modes zero_shot conformal_aci lora_variate lora_temporal full_ft \
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
| **Volatility** | `lora_variate` | 204,800 | 0.000843 | 0.2120 | 1.7284 | 0.770 | 0.00667 | 0.885 | 0.00962 | -1.3% | -0.4% | 2.00 GiB / 34.5 s |
| **Volatility** | `lora_temporal` | 819,200 | **0.000840** | **0.2111** | **1.7242** | 0.717 | 0.00586 | 0.853 | 0.00858 | +0.6% | +0.7% | 2.39 GiB / 42.4 s |
| **Volatility** | `full_ft` | 122,606,720 | 0.000849 | 0.2134 | 1.7405 | 0.705 | 0.00563 | 0.840 | 0.00814 | -1.1% | -0.9% | 2.75 GiB / 37.6 s |
| **Retail** | `zero_shot` | 0 | 0.1283 | 0.2163 | 0.8252 | 0.841 | 1.320 | 0.934 | 1.798 | +0.0% | +0.0% | — |
| **Retail** | `conformal_aci` | 0 | 0.1283 | 0.2162 | 0.8252 | 0.786 | 1.287 | 0.908 | 1.772 | +0.0% | +0.0% | — |
| **Retail** | `lora_variate` | 204,800 | 0.1161 | 0.1958 | 0.7618 | 0.791 | 1.046 | 0.907 | 1.412 | -0.5% | -0.4% | 2.49 GiB / 30.6 s |
| **Retail** | `lora_temporal` | 819,200 | 0.1166 | 0.1967 | 0.7659 | 0.783 | 1.000 | 0.898 | 1.330 | +4.6% | +3.4% | 3.00 GiB / 36.0 s |
| **Retail** | `full_ft` | 122,606,720 | **0.1150** | **0.1939** | **0.7574** | 0.776 | 0.996 | 0.895 | 1.317 | +2.5% | +0.8% | 3.13 GiB / 29.2 s |

### Timestral-209M — Realised Volatility (`2026-11-03` to `2026-11-20`) & Retail Demand (`2024-06-21` to `2024-07-02`)

| Dataset | Mode | Params | Pinball | wQL | MASE | Cov 80% | Wid 80% | Cov 90% | Wid 90% | Macro Forg wQL | Macro Forg MASE | Peak VRAM / Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Volatility** | `zero_shot` | 0 | 0.000876 | 0.2201 | 1.7890 | 0.731 | 0.00639 | 0.835 | 0.00873 | +0.0% | +0.0% | — |
| **Volatility** | `conformal_aci` | 0 | 0.000882 | 0.2218 | 1.7890 | 0.781\* | 0.00718 | 0.864 | 0.01092 | +0.0% | +0.0% | — |
| **Volatility** | `lora_variate` | 294,912 | 0.000840 | 0.2113 | **1.7217** | 0.704 | 0.00589 | 0.855 | 0.00858 | -0.9% | -0.7% | 3.07 GiB / 44.8 s |
| **Volatility** | `lora_temporal` | 1,179,648 | **0.000829** | **0.2084** | 1.7250 | 0.729 | 0.00587 | 0.863 | 0.00859 | -0.5% | -0.5% | 3.60 GiB / 53.2 s |
| **Volatility** | `full_ft` | 209,312,256 | 0.000848 | 0.2132 | 1.7611 | 0.704 | 0.00583 | 0.847 | 0.00860 | -1.6% | -1.7% | 4.37 GiB / 46.8 s |
| **Retail** | `zero_shot` | 0 | 0.1322 | 0.2230 | 0.8471 | 0.832 | 1.400 | 0.923 | 1.925 | +0.0% | +0.0% | — |
| **Retail** | `conformal_aci` | 0 | 0.1322 | 0.2228 | 0.8471 | 0.810 | 1.356 | 0.901 | 1.890 | +0.0% | +0.0% | — |
| **Retail** | `lora_variate` | 294,912 | 0.1156 | 0.1949 | 0.7587 | 0.774 | 1.026 | 0.894 | 1.370 | -0.1% | -0.2% | 3.84 GiB / 44.0 s |
| **Retail** | `lora_temporal` | 1,179,648 | 0.1148 | 0.1936 | 0.7544 | 0.779 | 0.977 | 0.892 | 1.298 | +4.6% | +1.7% | 4.51 GiB / 51.0 s |
| **Retail** | `full_ft` | 209,312,256 | **0.1137** | **0.1918** | **0.7508** | 0.772 | 0.996 | 0.890 | 1.315 | +1.7% | +0.5% | 4.91 GiB / 37.0 s |

\* Per-block volatility ACI 80% coverage: `[0.598, 0.901, 0.866]` (123M) and `[0.564, 0.906, 0.873]` (209M).

### Precision & attention-kernel ablation on `123M` & `209M` (`docs/benchmarks/precision_ablation_123m_209m.json`)

Reproduce (train-only retail selection, `lora_temporal`, `lr=1e-3`, 120 steps, `B=16`):

```bash
for variant in fp32_default fp32_math bf16_default; do
  uv run python scripts/benchmarks/run_benchmarks.py --device cuda \
    $([ "$variant" = bf16_default ] && echo --bf16) \
    --attention-backend $([ "$variant" = fp32_math ] && echo math || echo default) \
    --sizes 123M 209M --datasets retail_demand --modes lora_temporal \
    --max-steps 120 --batch-size 16 --eval-batch-size 16
done
```

| Model | Variant | Test wQL | Test MASE | Macro Forg wQL | Wall time | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| **123M** | `fp32` + default SDPA | 0.1982 | 0.7725 | +4.9% | 37.4 s | 2.23 GiB |
| **123M** | `fp32` + `attention_context("math")` | 0.1978 | 0.7713 | +4.8% | 36.6 s | 2.30 GiB |
| **123M** | `bf16` + default SDPA | 0.1981 | 0.7716 | +6.8% | **27.1 s** | **1.81 GiB** |
| **209M** | `fp32` + default SDPA | 0.1980 | 0.7727 | +3.8% | 53.7 s | 3.36 GiB |
| **209M** | `fp32` + `attention_context("math")` | 0.1951 | 0.7617 | +4.1% | 53.4 s | 3.47 GiB |
| **209M** | `bf16` + default SDPA | **0.1947** | **0.7603** | +4.1% | **29.9 s** | **2.79 GiB** |

---

## 4. Statistical significance, multi-seed variance & classical baselines (`docs/benchmarks/sig_v2.json`)

Reproduce:

```bash
uv run python scripts/benchmarks/significance.py \
  --device cuda --bf16 --max-steps 200 --batch-size 32 --seeds 0 1 2 \
  --output docs/benchmarks/sig_v2.json
```

All methods use the shipped defaults (`sign_flip=True`, `lora_variate=3e-3`, `lora_temporal=1e-3`,
`full_ft=1e-4`) and identical test windows scored by `evaluate_predictions`. Paired 95% CIs use
10,000-resample cluster bootstrap on `ΔwQL = wQL(tuned) - wQL(zero_shot)` clustered separately by
series and by non-overlapping time block; lag-1 Newey-West Diebold-Mariano p-values are reported
where block count is large (`k = 49`).

### Retail demand (`2024-06-21` to `2024-07-02`; 64 train-ranked SKUs × 6 time blocks = 384 windows)

| Method | wQL (`mean ± std`, 3 seeds) | MASE (`mean ± std`) | Series 95% CI on ΔwQL (wins) | Time-block 95% CI on ΔwQL (wins) |
|---|---:|---:|---|---|
| `seasonal_naive` (`s=24`) | 0.2939 | 1.1200 | — | — |
| `seasonal_ridge` (covariates) | 0.2177 | 0.8427 | — | — |
| `28M zero_shot` | 0.2193 | 0.8347 | baseline | baseline |
| `28M lora_variate` | 0.1979 ± 0.0010 | 0.7683 ± 0.0048 | `[-0.0234, -0.0179]` (63/64) | `[-0.0250, -0.0167]` (6/6) |
| `28M lora_temporal` | 0.1977 ± 0.0001 | 0.7680 ± 0.0003 | `[-0.0245, -0.0190]` (64/64) | `[-0.0259, -0.0176]` (6/6) |
| `28M full_ft` | **0.1951 ± 0.0004** | **0.7608 ± 0.0010** | `[-0.0276, -0.0218]` (64/64) | `[-0.0285, -0.0209]` (6/6) |

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
| `28M lora_temporal` | 0.1819 | 1.3700 | `[-0.0128, -0.0087]` (49/51) | `[-0.0140, -0.0076]` (44/49, `p = 4.1e-10`) |
| `28M full_ft` | **0.1816** | **1.3663** | `[-0.0130, -0.0092]` (50/51) | `[-0.0142, -0.0081]` (46/49, `p = 1.5e-11`) |

### Realised volatility — strict post-release slice (`2026-11-03` to `2026-11-20`; 51 equities × 3 blocks = 153 windows)

| Method | wQL (`mean ± std`, 3 seeds) | MASE (`mean ± std`) | Series 95% CI on ΔwQL (wins) | Time-block 95% CI on ΔwQL (wins) |
|---|---:|---:|---|---|
| `seasonal_naive` (`s=7`) | 0.2967 | 2.2553 | — | — |
| `har_rv` (Corsi HAR + covariates) | 0.2253 | 1.7725 | — | — |
| `28M zero_shot` | 0.2293 | 1.8785 | baseline | baseline |
| `28M lora_variate` | 0.2185 ± 0.0020 | 1.7788 ± 0.0126 | `[-0.0219, -0.0039]` (44/51) | `[-0.0269, -0.0018]` (3/3) |
| `28M lora_temporal` | 0.2171 ± 0.0016 | **1.7633 ± 0.0168** | `[-0.0208, -0.0006]` (38/51) | `[-0.0293, +0.0061]` (2/3) |
| `28M full_ft` | **0.2161 ± 0.0012** | 1.7662 ± 0.0113 | `[-0.0228, -0.0035]` (39/51) | `[-0.0341, +0.0029]` (2/3) |

### Summary of statistical & baseline measurements

1. **Seed variance is small:** across 3 seeds, `wQL` standard deviation is `0.0001–0.0010` on retail
   (vs `−0.021` to `−0.024` gain) and `0.0012–0.0020` on post-release volatility (vs `−0.011` to
   `−0.013` gain).
2. **Time-block significance holds on retail and the 49-block 2026 volatility holdout:** retail
   improves `63–64/64` SKUs and `6/6` time blocks; the 49-block 2026 volatility holdout improves
   `47–50/51` equities and `44–46/49` market blocks (`DM p <= 2.7e-8`). On the strict 3-block
   post-release slice alone, series-level CIs exclude zero for all three modes (`38–44/51` equities),
   while only `lora_variate` wins `3/3` time blocks (`lora_temporal` and `full_ft` win `2/3` and their
   3-block time CIs straddle zero, as expected for `k = 3`).
3. **Beats classical domain baselines:** `28M zero_shot` trailed `seasonal_ridge` on retail (`0.2193` vs
   `0.2177`) and trailed Corsi `har_rv` on post-release volatility (`0.2293` vs `0.2253`). At the shipped
   defaults, `full_ft` / `lora_temporal` beat `seasonal_ridge` by **−9.2% to −10.4% wQL** on retail,
   beat `har_rv` by **−8.2% to −8.3% wQL** across 2026, and beat `har_rv` by **−3.6% to −4.1% wQL**
   on the post-release volatility slice.

---

## 5. Phase 4: Replay mixing & frozen-base distillation ablation (`docs/benchmarks/replay_ablation.json`)

Reproduce (Apple M4 GPU, `fp32`, `max_steps=200`, `batch_size=32`, built-in `SyntheticReplaySource` with 64 series):

```bash
uv run python scripts/benchmarks/run_replay_ablation.py \
  --device mps --output docs/benchmarks/replay_ablation.json
```

Evaluates `ReplayConfig` (`replay_fraction` and `distill_weight`) across the three high-forgetting regimes
identified in Phase 3:

### 5.1 `head_only` at shipped `3e-3` vs `1e-3` lower-LR control (`5M` 3 seeds; `28M` seed `0`)

Note: most `head_only` runs select their final step (`200`), so longer schedules may behave differently.

#### `5M` (3 seeds `0 / 1 / 2`)

| Dataset | Variant (`lr`, `replay_fraction / distill_weight`) | Test wQL (`s0 / s1 / s2`; mean) | Macro Forg wQL (`s0 / s1 / s2`; **worst**) |
|---|---|---|---|
| Volatility | `lr1e-3_none` (lower-LR control) | `0.2231 / 0.2234 / 0.2242` (0.2236) | `+6.9% / +7.8% / +7.4%` (**+7.8%**) |
| Volatility | `none` (`3e-3`) | `0.2196 / 0.2203 / 0.2210` (0.2203) | `+13.7% / +13.0% / +14.1%` (**+14.1%**) |
| Volatility | `replay_0.25` (`3e-3`, `0.25 / 0.0`) | `0.2232 / 0.2216 / 0.2246` (**0.2231**) | `+5.0% / +3.5% / +4.0%` (**+5.0%**) |
| Volatility | `distill_0.5` (`3e-3`, `0.0 / 0.5`) | `0.2302 / 0.2302 / 0.2308` (0.2304) | `+0.7% / +0.9% / +0.6%` (**+0.9%**) |
| Volatility | `both_0.2_0.5` (`3e-3`, `0.2 / 0.5`) | `0.2308 / 0.2309 / 0.2312` (0.2310) | `+1.1% / +0.7% / +0.5%` (**+1.1%**) |
| Retail | `lr1e-3_none` (lower-LR control) | `0.2115 / 0.2114 / 0.2114` (0.2114) | `+3.6% / +3.6% / +3.4%` (**+3.6%**) |
| Retail | `none` (`3e-3`) | `0.2098 / 0.2099 / 0.2095` (0.2097) | `+5.0% / +5.0% / +4.8%` (**+5.0%**) |
| Retail | `replay_0.25` (`3e-3`, `0.25 / 0.0`) | `0.2099 / 0.2103 / 0.2098` (0.2100) | `+3.8% / +5.6% / +3.8%` (**+5.6%**) |
| Retail | `distill_0.5` (`3e-3`, `0.0 / 0.5`) | `0.2123 / 0.2120 / 0.2118` (0.2120) | `+1.9% / +1.7% / +1.6%` (**+1.9%**) |

#### `28M` (seed `0`, single run)

| Dataset | Variant (`lr`, `replay_fraction / distill_weight`) | Val wQL | Test wQL | Macro Forg wQL | Wall time (vs `none`) |
|---|---|---:|---:|---:|---:|
| Volatility | `lr1e-3_none` (lower-LR control) | 0.1800 | 0.2169 | +6.7% | 81.0 s (−20%) |
| Volatility | `none` (`3e-3`) | 0.1794 | 0.2165 | +10.3% | 101.5 s (0%) |
| Volatility | `replay_0.25` (`3e-3`, `0.25 / 0.0`) | 0.1805 | 0.2195 | +4.1% | 105.1 s (+4%) |
| Volatility | `distill_0.5` (`3e-3`, `0.0 / 0.5`) | 0.1822 | 0.2235 | +0.7% | 107.1 s (+6%) |
| Volatility | `distill_1.0` (`3e-3`, `0.0 / 1.0`) | 0.1829 | 0.2252 | +0.9% | 114.2 s (+13%) |
| Volatility | `both_0.2_0.5` (`3e-3`, `0.2 / 0.5`) | 0.1827 | 0.2247 | +0.4% | 117.3 s (+16%) |
| Retail | `lr1e-3_none` (lower-LR control) | 0.2271 | 0.2090 | +4.2% | 34.5 s (−29%) |
| Retail | `none` (`3e-3`) | 0.2256 | 0.2072 | +7.0% | 48.4 s (0%) |
| Retail | `replay_0.25` (`3e-3`, `0.25 / 0.0`) | 0.2262 | 0.2075 | +5.3% | 49.3 s (+2%) |
| Retail | `distill_0.5` (`3e-3`, `0.0 / 0.5`) | 0.2279 | 0.2095 | +2.7% | 45.8 s (−5%) |
| Retail | `distill_1.0` (`3e-3`, `0.0 / 1.0`) | 0.2288 | 0.2104 | +2.1% | 45.3 s (−6%) |
| Retail | `both_0.2_0.5` (`3e-3`, `0.2 / 0.5`) | 0.2284 | 0.2098 | +2.4% | 49.1 s (+2%) |

### 5.2 `lora_temporal` at higher LR `3e-3` (`5M` volatility, 3 seeds `0 / 1 / 2`)

| Variant (`lr`, `replay_fraction / distill_weight`) | Test wQL (`s0 / s1 / s2`; mean) | Macro Forg wQL (`s0 / s1 / s2`; **worst**) |
|---|---|---|
| `lr1e-3_none` (shipped default) | `0.2175 / 0.2185 / 0.2198` (0.2186) | `-0.2% / -1.2% / +1.6%` (**+1.6%**) |
| `lr3e-3_none` (forgetting cliff) | `0.2167 / 0.2167 / 0.2161` (0.2165) | `+20.8% / +31.3% / +7.6%` (**+31.3%**) |
| `lr3e-3_replay_0.25` (`0.25 / 0.0`) | `0.2201 / 0.2155 / 0.2186` (0.2181) | `+0.1% / +1.1% / -0.2%` (**+1.1%**) |
| `lr3e-3_distill_0.5` (`0.0 / 0.5`) | `0.2209 / 0.2227 / 0.2221` (0.2219) | `-1.2% / -1.3% / -0.4%` (**-0.4%**) |
| `lr3e-3_distill_1.0` (`0.0 / 1.0`) | `0.2252 / 0.2254 / 0.2248` (0.2251) | `-0.4% / -1.5% / +0.4%` (**+0.4%**) |
| `lr3e-3_both_0.2_0.5` (`0.2 / 0.5`) | `0.2231 / 0.2247 / 0.2223` (0.2234) | `+1.4% / -1.5% / +0.4%` (**+1.4%**) |

### 5.3 `full_ft` at higher LR `3e-4` vs `1e-4` & `2e-4` controls (`5M` retail demand, 3 seeds `0 / 1 / 2`)

| Variant (`lr`, `replay_fraction / distill_weight`, pool) | Test wQL (`s0 / s1 / s2`; mean) | Macro Forg wQL (`s0 / s1 / s2`; mean, **worst**) |
|---|---|---|
| `lr1e-4_none` (shipped default) | `0.2014 / 0.2012 / 0.2022` (0.2016) | `+0.4% / +0.5% / +0.6%` (+0.5%, **+0.6%**) |
| `lr2e-4_none` (lower-LR control) | `0.1993 / 0.1989 / 0.2003` (0.1995) | `+1.3% / +1.0% / +1.3%` (+1.2%, **+1.3%**) |
| `lr3e-4_none` | `0.1983 / 0.1981 / 0.1998` (**0.1988**) | `+2.8% / +1.9% / +2.1%` (+2.3%, **+2.8%**) |
| `lr3e-4_replay_0.25` (`0.25 / 0.0`, 64 series) | `0.1988 / 0.1986 / 0.2004` (0.1993) | `+4.1% / +3.1% / +3.7%` (+3.6%, **+4.1%**) |
| `lr3e-4_replay_0.25_256s` (`0.25 / 0.0`, 256 series) | `0.2004 / 0.1989 / 0.2005` (0.1999) | `+2.7% / +2.8% / +2.2%` (+2.5%, **+2.8%**) |
| `lr3e-4_distill_0.1` (`0.0 / 0.1`, 64 series) | `0.1993 / 0.1993 / 0.2010` (0.1999) | `+1.3% / +0.6% / +0.9%` (+0.9%, **+1.3%**) |
| `lr3e-4_distill_0.5` (`0.0 / 0.5`, 64 series) | `0.2011 / 0.2017 / 0.2020` (0.2016) | `+1.0% / +0.4% / +1.1%` (+0.8%, **+1.1%**) |
| `lr3e-4_distill_1.0` (`0.0 / 1.0`, 64 series) | `0.2046 / 0.2036 / 0.2037` (0.2040) | `+1.2% / +0.4% / +0.6%` (+0.7%, **+1.2%**) |
| `lr3e-4_both_0.2_0.5` (`0.2 / 0.5`, 64 series) | `0.2026 / 0.2027 / 0.2027` (0.2027) | `+0.5% / +0.6% / +0.8%` (+0.7%, **+0.8%**) |

**What the data actually shows (with lower-LR and pool-size controls):**
- **Lowering the learning rate is often as effective as synthetic replay/distillation:** on `5M full_ft` retail, plain `lr=2e-4` (`0.1995`, worst forg `+1.3%`) matches or slightly beats `3e-4 + distill_0.1` (`0.1999`, worst `+1.3%`, mean `+0.9%` vs `+1.2%`), and plain `lr=1e-4` (`0.2016`, worst `+0.6%`) strictly beats `3e-4 + distill_0.5` (`0.2016`, worst `+1.1%`). On `5M lora_temporal` volatility, `3e-3 + replay_0.25` (`0.2181`, worst `+1.1%`) rescues a non-default higher LR from its `+31.3%` cliff, but is indistinguishable across 3 seeds from shipped `1e-3` without replay (`0.2186`, worst `+1.6%`).
- **Where built-in replay/distillation beats the lower-LR control:** on `5M head_only` volatility (3 seeds), `3e-3 + replay_0.25` matches plain `1e-3`'s test wQL (`0.2231` vs `0.2236`) while lowering worst-seed forgetting from `+7.8%` to `+5.0%`; on `5M` and `28M` retail `head_only`, `3e-3 + distill_0.5` roughly halves forgetting relative to plain `1e-3` (`+1.9%` vs `+3.6%` on `5M`, `+2.7%` vs `+4.2%` on `28M`) for `+0.0005–0.0006` wQL. On `28M` volatility (single seed), however, plain `1e-3` gives lower wQL (`0.2169` vs `0.2195`). Across `28M` single runs, `head_only` wall time with dual-cached replay/distillation ranged from `−6%` to `+16%` relative to `none`.
- **Synthetic pool size vs generator mismatch (`full_ft`):** expanding the `SyntheticReplaySource` pool from 64 series (256 windows, cycled ~6×) to 256 series (1,024 windows, cycled ~1.5×) dropped `full_ft 3e-4` synthetic-pinball worst-seed forgetting from `+4.1%` back to `+2.8%`, showing that memorising a small pool explained the `+4.1%` spike — while synthetic pinball still did not improve forgetting over plain `3e-4` (`+2.8%`), consistent with D7. Real general-domain replay sets remain untested.
- **Note on `ReplayConfig()` default (`replay_fraction=0.0, distill_weight=0.1`):** chosen as a conservative default that avoids synthetic pinball worsening `full_ft`, but it is lightly evidenced — `distill_weight=0.1` was measured only on `5M full_ft` (where it tied plain `2e-4`), whereas the `head_only` reductions above used `distill_weight=0.5` or `replay_fraction=0.25`.
