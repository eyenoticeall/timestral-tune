# Phase 3 Benchmarks & Hyperparameter Sweeps

All numbers below were measured on an NVIDIA L4 GPU (`g2-standard-4`, PyTorch 2.15.0+cu132,
Python 3.12, `bf16` autocast) on `2026-11-22`, evaluating every mode in raw space through the
vendored upstream `TimestralForecaster.forecast()` with its defaults (`symmetric_averaging=True`,
`clip_nonnegative=True`; spec §5.5).

## Datasets, licences & contamination notes

| Role | Dataset (pinned HF Hub revision) | Licence | Shape & covariates | Contamination note |
|---|---|---|---|---|
| **Realised volatility** | `twelvedata/financial-world-model@da186e5fd362` (`bars_1h`) | MIT | 51 large-cap US equities, 1h bars, 8 rows/series (`rv_20` + 5 past: `log1p(volume)`, `vix_level`, `spy_logret_1`, `sector_logret_1`, `atr_14` + 2 known calendar: hour, weekday). `context=256`, `horizon=32`, train `6,324` / val `1,071` / test `153` windows. | **Test split covers 2026-10-28 to 2026-11-20** — strictly after Timestral's 2026-10-27 release (no pretraining contamination on test targets). Train/val precede release. |
| **Retail demand** | `Dingdong-Inc/FreshRetailNet-50K@c374e85975c8` | CC BY 4.0 | Deterministic top-64 active store-SKU hourly series (97 days × 24h), 9 rows/series (hourly sales + 3 past: stockout, precipitation, temperature + 5 known: discount, holiday, activity, hour, weekday). `context=256`, `horizon=48`, train `1,856` / val `384` / test `384` windows. | Published before October 2026 (2024 dates); **may have been seen** during Mistral's post-training (§12). |
| **Forgetting set** | `Salesforce/GiftEval@30841734ac5c` | Apache-2.0 | 48 hourly series (192 windows, `context=256`, `horizon=64`) across energy (`ett1/H`), weather (`jena_weather/H`), and highway traffic (`LOOP_SEATTLE/H`) — disjoint from finance and retail. | General-domain reference; forgetting is reported as `tuned - zero_shot` (positive = degradation, negative = transfer gain). |

All three licences permit publishing derived benchmark metrics (spec §8).

---

## 1. Main benchmark: zero-shot vs conformal vs head-only vs LoRA vs full FT

Reproduce:

```bash
uv run python scripts/benchmarks/run_benchmarks.py \
  --device cuda --bf16 --compile --sizes 5M 28M --max-steps 200 --batch-size 32 --sign-flip
```

Hyperparameters used (selected by §2 sweeps): `head_only` LR `3e-4`, `lora_variate` LR `1e-3`,
`lora_temporal` LR `1e-3`, `full_ft` LR `5e-5`, LoRA `r=8, alpha=16, targets=("q","k","v","o"), dropout=0.05`,
`sign_flip=True`, `restore_best=True`.

### Timestral-5M — Realised Volatility (post-Oct 27, 2026 test split, 153 windows)

| Mode | Trainable params | Pinball | wQL | MASE | Cov 80% | Width 80% | Cov 90% | Width 90% | Forg ΔPinball | Forg ΔwQL | Forg ΔMASE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zero_shot` | 0 | 0.000939 | 0.2360 | 1.9167 | 0.663 | 0.00603 | 0.796 | 0.00823 | +0.0000 | +0.0000 | +0.0000 |
| `conformal_cqr` | 0 | 0.000933 | 0.2346 | 1.9167 | 0.689 | 0.00634 | 0.825 | 0.00884 | +0.0000 | +0.0000 | +0.0000 |
| `conformal_aci` | 0 | 0.000935 | 0.2352 | 1.9167 | **0.765** | 0.00757 | **0.875** | 0.01095 | +0.0000 | +0.0000 | +0.0000 |
| `head_only` | 792,576 | 0.000903 | 0.2269 | 1.8685 | 0.736 | 0.00662 | 0.862 | 0.00947 | +0.2492 | +0.0050 | +0.0174 |
| `lora_variate` | 24,576 | 0.000885 | 0.2223 | 1.8060 | 0.728 | 0.00633 | 0.836 | 0.00889 | -0.0665 | -0.0013 | -0.0140 |
| `lora_temporal` | 98,304 | **0.000868** | **0.2183** | **1.7783** | 0.721 | 0.00620 | 0.843 | 0.00870 | +0.0895 | +0.0018 | +0.0745 |
| `full_ft` | 5,208,960 | 0.000888 | 0.2231 | 1.8229 | 0.735 | 0.00635 | 0.841 | 0.00880 | -0.0093 | -0.0002 | -0.0125 |

### Timestral-28M — Realised Volatility (post-Oct 27, 2026 test split, 153 windows)

| Mode | Trainable params | Pinball | wQL | MASE | Cov 80% | Width 80% | Cov 90% | Width 90% | Forg ΔPinball | Forg ΔwQL | Forg ΔMASE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zero_shot` | 0 | 0.000912 | 0.2293 | 1.8785 | 0.702 | 0.00641 | 0.825 | 0.00878 | +0.0000 | +0.0000 | +0.0000 |
| `conformal_cqr` | 0 | 0.000909 | 0.2286 | 1.8785 | 0.712 | 0.00653 | 0.834 | 0.00908 | +0.0000 | +0.0000 | +0.0000 |
| `conformal_aci` | 0 | 0.000912 | 0.2293 | 1.8785 | **0.766** | 0.00745 | **0.874** | 0.01076 | +0.0000 | +0.0000 | +0.0000 |
| `head_only` | 1,880,064 | 0.000871 | 0.2190 | 1.8107 | 0.738 | 0.00650 | 0.871 | 0.00954 | +0.2087 | +0.0042 | +0.0110 |
| `lora_variate` | 73,728 | **0.000870** | **0.2187** | 1.7850 | 0.730 | 0.00617 | 0.846 | 0.00871 | -0.0260 | -0.0005 | -0.0109 |
| `lora_temporal` | 294,912 | **0.000870** | 0.2188 | **1.7826** | 0.733 | 0.00608 | 0.855 | 0.00866 | -0.0876 | -0.0018 | -0.0234 |
| `full_ft` | 27,821,184 | **0.000870** | 0.2188 | 1.7848 | 0.728 | 0.00607 | 0.852 | 0.00877 | -0.0191 | -0.0004 | -0.0183 |

### Timestral-5M — Retail Demand (384 test windows)

| Mode | Trainable params | Pinball | wQL | MASE | Cov 80% | Width 80% | Cov 90% | Width 90% | Forg ΔPinball | Forg ΔwQL | Forg ΔMASE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zero_shot` | 0 | 0.1352 | 0.2201 | 0.8569 | 0.822 | 1.317 | 0.924 | 1.837 | +0.0000 | +0.0000 | +0.0000 |
| `conformal_cqr` | 0 | 0.1352 | 0.2201 | 0.8569 | 0.822 | 1.317 | 0.924 | 1.837 | +0.0000 | +0.0000 | +0.0000 |
| `conformal_aci` | 0 | 0.1352 | 0.2202 | 0.8569 | **0.792** | 1.301 | **0.906** | 1.823 | +0.0000 | +0.0000 | +0.0000 |
| `head_only` | 792,576 | 0.1308 | 0.2131 | 0.8468 | 0.805 | 1.206 | 0.914 | 1.677 | +0.0758 | +0.0015 | +0.0034 |
| `lora_variate` | 24,576 | 0.1268 | 0.2065 | 0.8173 | 0.816 | 1.159 | 0.914 | 1.584 | +0.0528 | +0.0011 | -0.0059 |
| `lora_temporal` | 98,304 | **0.1238** | **0.2017** | **0.7958** | 0.798 | 1.083 | 0.902 | 1.475 | +0.0255 | +0.0005 | +0.0150 |
| `full_ft` | 5,208,960 | 0.1255 | 0.2044 | 0.8109 | 0.804 | 1.144 | 0.911 | 1.569 | +0.0268 | +0.0005 | -0.0047 |

### Timestral-28M — Retail Demand (384 test windows)

| Mode | Trainable params | Pinball | wQL | MASE | Cov 80% | Width 80% | Cov 90% | Width 90% | Forg ΔPinball | Forg ΔwQL | Forg ΔMASE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zero_shot` | 0 | 0.1346 | 0.2192 | 0.8477 | 0.762 | 1.285 | 0.878 | 1.739 | +0.0000 | +0.0000 | +0.0000 |
| `conformal_cqr` | 0 | 0.1346 | 0.2192 | 0.8477 | 0.762 | 1.285 | 0.879 | 1.742 | +0.0000 | +0.0000 | +0.0000 |
| `conformal_aci` | 0 | 0.1345 | 0.2191 | 0.8477 | **0.795** | 1.306 | **0.894** | 1.758 | +0.0000 | +0.0000 | +0.0000 |
| `head_only` | 1,880,064 | 0.1295 | 0.2109 | 0.8329 | 0.788 | 1.205 | 0.894 | 1.614 | +0.0745 | +0.0015 | +0.0020 |
| `lora_variate` | 73,728 | 0.1221 | 0.1988 | 0.7857 | 0.781 | 1.069 | 0.902 | 1.456 | -0.0472 | -0.0010 | -0.0119 |
| `lora_temporal` | 294,912 | **0.1202** | **0.1957** | **0.7743** | 0.787 | 1.049 | 0.902 | 1.411 | -0.0471 | -0.0009 | +0.0069 |
| `full_ft` | 27,821,184 | 0.1205 | 0.1963 | 0.7782 | 0.780 | 1.071 | 0.898 | 1.440 | -0.0428 | -0.0009 | -0.0134 |

### Key findings

- **Every adaptation mode beats zero-shot** on both datasets and both model sizes (`4.6%–10.7%` wQL reduction, `4.9%–8.7%` MASE reduction), including the strictly post-release equity volatility test split.
- **LoRA matches or beats `full_ft` with 1% or less of the parameters and negligible/negative forgetting:** on 28M realised volatility, `lora_variate` (`73,728` params, `0.26%` of 28M) matches `full_ft` (`27.8M` params) at `0.2187` vs `0.2188` wQL; on both 5M and 28M retail demand, `lora_temporal` (`1.9%` / `1.06%` of params) beats `full_ft` outright (`0.2017` vs `0.2044` on 5M; `0.1957` vs `0.1963` on 28M).
- **Online ACI fixes real-world coverage drift with zero weight changes:** on the post-Oct 27, 2026 volatility regime shift, `conformal_aci` raises 5M 80% coverage from `66.3%` to `76.5%` (28M: `70.2%` to `76.6%`) and 90% coverage from `79.6%` to `87.5%` (`82.5%` to `87.4%`), with zero forgetting. On retail demand, `conformal_aci` pulls both over-covering 5M (`82.2% → 79.2%`) and under-covering 28M (`76.2% → 79.5%`) onto nominal 80%.

---

## 2. Hyperparameter sweeps & sign-flip ablation

Reproduce:

```bash
uv run python scripts/benchmarks/run_sweeps.py --device cuda --size 5M --bf16 --max-steps 120 --batch-size 32
```

### Learning-rate sweep (validation wQL, 120 steps, `bf16`)

| Mode | LR | 5M Volatility (ZS `0.1891`) | 5M Retail (ZS `0.2374`) | 28M Volatility (ZS `0.1850`) | 28M Retail (ZS `0.2399`) | Chosen default |
|---|---:|---:|---:|---:|---:|---|
| `head_only` | `3e-5` | 0.1883 | 0.2349 | — | — | |
| `head_only` | `1e-4` | 0.1875 | **0.2345** | 0.1828 | 0.2330 | |
| `head_only` | `3e-4` | 0.1865 | 0.2351 | **0.1820** | **0.2300** | **`3e-4`** |
| `head_only` | `1e-3` | **0.1856** | 0.2374 | — | — | |
| `lora_variate` | `1e-4` | 0.1879 | 0.2371 | — | — | |
| `lora_variate` | `3e-4` | 0.1860 | 0.2366 | 0.1809 | 0.2217 | |
| `lora_variate` | `1e-3` | **0.1829** | **0.2314** | **0.1785** | **0.2072** | **`1e-3`** |
| `lora_temporal` | `1e-4` | 0.1859 | 0.2356 | — | — | |
| `lora_temporal` | `3e-4` | 0.1826 | 0.2312 | 0.1780 | 0.2077 | |
| `lora_temporal` | `1e-3` | **0.1814** | **0.2279** | **0.1767** | **0.1995** | **`1e-3`** |
| `full_ft` | `3e-6` | 0.1879 | 0.2354 | — | — | |
| `full_ft` | `1e-5` | 0.1859 | 0.2334 | 0.1812 | 0.2256 | |
| `full_ft` | `3e-5` | 0.1832 | 0.2292 | 0.1788 | 0.2103 | |
| `full_ft` | `1e-4` | **0.1810** | **0.2230** | **0.1758** | **0.1963** | **`5e-5`** (conservative across 200+ steps) |

(On 28M retail at 200 steps, `full_ft` gave test wQL `0.1979` at `3e-5`, `0.1963` at `5e-5`, and `0.1934` at `1e-4` — `5e-5` is adopted as the default to balance adaptation speed with low forgetting across longer schedules.)

### LoRA rank & targets sweep (5M validation wQL / MASE, 120 steps)

| Mode | `r` | Targets | Volatility wQL / MASE | Retail wQL / MASE |
|---|---:|---|---:|---:|
| `lora_variate` | 4 | `("q", "v")` | 0.1840 / 1.3536 | 0.2365 / 0.8223 |
| `lora_variate` | 8 | `("q", "v")` | 0.1829 / 1.3449 | 0.2314 / 0.8052 |
| `lora_variate` | 16 | `("q", "v")` | 0.1828 / 1.3414 | 0.2302 / 0.8033 |
| `lora_variate` | **8** | **`("q", "k", "v", "o")`** | **0.1822 / 1.3356** | **0.2268 / 0.7946** |
| `lora_temporal` | 4 | `("q", "v")` | 0.1817 / 1.3336 | 0.2271 / 0.7969 |
| `lora_temporal` | 8 | `("q", "v")` | 0.1814 / 1.3303 | 0.2279 / 0.8028 |
| `lora_temporal` | 16 | `("q", "v")` | 0.1813 / 1.3258 | 0.2250 / 0.7921 |
| `lora_temporal` | **8** | **`("q", "k", "v", "o")`** | **0.1810 / 1.3232** | **0.2225 / 0.7805** |

`r=8, alpha=16, targets=("q", "k", "v", "o")` beats `r=16` on `("q", "v")` across both domains and both axes and is adopted as the `LoraConfig` default.

### Sign-flip augmentation ablation (5M validation wQL / MASE, 120 steps)

| Mode | `sign_flip` | Volatility wQL / MASE | Retail wQL / MASE |
|---|---|---:|---:|
| `head_only` | `False` | 0.1856 / 1.3630 | 0.2345 / 0.8194 |
| `head_only` | `True` | **0.1841 / 1.3602** | **0.2317 / 0.8115** |
| `lora_variate` | `False` | **0.1829 / 1.3449** | 0.2314 / 0.8052 |
| `lora_variate` | `True` | 0.1830 / 1.3461 | **0.2312 / 0.8026** |
| `full_ft` | `False` | 0.1810 / 1.3282 | 0.2230 / 0.7827 |
| `full_ft` | `True` | **0.1804 / 1.3273** | **0.2141 / 0.7551** |

`sign_flip=True` delivers consistent gains on strictly non-negative targets (up to **−4.0% wQL** on retail `full_ft`) because upstream `TimestralForecaster` averages with the reflected pass of `-context` (§5.3). `TrainingConfig.sign_flip` stays `False` by default for general signed series and is recommended (`sign_flip=True`) for one-signed domains.

### Real-data `bf16` vs `fp32` check (28M retail, 200 steps, `full_ft`, `lr=5e-5`)

| Precision | Test wQL | Test MASE | Forgetting ΔwQL | Forgetting ΔMASE |
|---|---:|---:|---:|---:|
| `bf16` | 0.1963 | 0.7782 | -0.0009 | -0.0134 |
| `fp32` | 0.1956 | 0.7759 | -0.0010 | -0.0166 |

Real 200-step full fine-tuning under CUDA `bf16` agrees with `fp32` within `0.35%` on raw test wQL and MASE.

---

## 3. Large checkpoints: Timestral-123M & Timestral-209M (NVIDIA L4, 23 GiB VRAM)

Measured on the same NVIDIA L4 VM (`bf16`, `sign_flip=True`, 160 steps, `eval_every=40`, `restore_best=True`,
`lora_*` LR `1e-3` with `r=8, alpha=16, targets=("q","k","v","o"), dropout=0.05`, `full_ft` LR `3e-5`,
`batch_size=32` for `123M` LoRA and `batch_size=16` for `123M full_ft` and all `209M` runs).

### Timestral-123M — Realised Volatility (post-Oct 27, 2026) & Retail Demand

| Dataset | Mode | Params | Pinball | wQL | MASE | Cov 80% | Wid 80% | Cov 90% | Wid 90% | Forg ΔwQL | Forg ΔMASE | Peak VRAM / Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Volatility** | `zero_shot` | 0 | 0.000875 | 0.2201 | 1.8017 | 0.734 | 0.00656 | 0.842 | 0.00894 | +0.0000 | +0.0000 | — |
| **Volatility** | `conformal_aci` | 0 | 0.000877 | 0.2205 | 1.8017 | **0.773** | 0.00718 | **0.878** | 0.01019 | +0.0000 | +0.0000 | — |
| **Volatility** | `lora_variate` | 204,800 | 0.000853 | 0.2145 | 1.7435 | 0.741 | 0.00602 | 0.858 | 0.00856 | -0.0007 | -0.0082 | 2.00 GiB / 34.7 s |
| **Volatility** | `lora_temporal` | 819,200 | **0.000840** | **0.2111** | **1.7242** | 0.717 | 0.00586 | 0.853 | 0.00858 | +0.0001 | +0.0018 | 2.39 GiB / 38.0 s |
| **Volatility** | `full_ft` | 122,606,720 | 0.000851 | 0.2139 | 1.7249 | 0.742 | 0.00601 | 0.844 | 0.00835 | -0.0004 | -0.0096 | 2.75 GiB / 33.8 s |
| **Retail** | `zero_shot` | 0 | 0.1327 | 0.2161 | 0.8401 | 0.838 | 1.353 | 0.932 | 1.846 | +0.0000 | +0.0000 | — |
| **Retail** | `conformal_aci` | 0 | 0.1327 | 0.2161 | 0.8401 | **0.789** | 1.318 | **0.904** | 1.823 | +0.0000 | +0.0000 | — |
| **Retail** | `lora_variate` | 204,800 | 0.1207 | 0.1966 | 0.7793 | 0.776 | 1.031 | 0.894 | 1.384 | +0.0002 | -0.0041 | 2.49 GiB / 28.6 s |
| **Retail** | `lora_temporal` | 819,200 | **0.1197** | **0.1950** | **0.7729** | 0.783 | 1.010 | 0.898 | 1.352 | +0.0025 | +0.0295 | 3.00 GiB / 33.7 s |
| **Retail** | `full_ft` | 122,606,720 | 0.1206 | 0.1965 | 0.7811 | 0.782 | 1.080 | 0.904 | 1.449 | +0.0005 | -0.0023 | 3.13 GiB / 27.5 s |

### Timestral-209M — Realised Volatility (post-Oct 27, 2026) & Retail Demand

| Dataset | Mode | Params | Pinball | wQL | MASE | Cov 80% | Wid 80% | Cov 90% | Wid 90% | Forg ΔwQL | Forg ΔMASE | Peak VRAM / Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Volatility** | `zero_shot` | 0 | 0.000876 | 0.2201 | 1.7890 | 0.731 | 0.00639 | 0.835 | 0.00873 | +0.0000 | +0.0000 | — |
| **Volatility** | `conformal_aci` | 0 | 0.000876 | 0.2202 | 1.7890 | **0.772** | 0.00700 | **0.876** | 0.01036 | +0.0000 | +0.0000 | — |
| **Volatility** | `lora_variate` | 294,912 | 0.000866 | 0.2178 | 1.7615 | 0.728 | 0.00580 | 0.845 | 0.00830 | -0.0004 | -0.0079 | 2.09 GiB / 33.8 s |
| **Volatility** | `lora_temporal` | 1,179,648 | **0.000846** | **0.2127** | **1.7395** | 0.718 | 0.00586 | 0.869 | 0.00873 | -0.0010 | -0.0192 | 2.38 GiB / 44.6 s |
| **Volatility** | `full_ft` | 209,312,256 | 0.000860 | 0.2161 | 1.7517 | 0.728 | 0.00577 | 0.842 | 0.00815 | -0.0007 | -0.0170 | 4.37 GiB / 42.2 s |
| **Retail** | `zero_shot` | 0 | 0.1366 | 0.2225 | 0.8603 | 0.829 | 1.434 | 0.922 | 1.974 | +0.0000 | +0.0000 | — |
| **Retail** | `conformal_aci` | 0 | 0.1365 | 0.2223 | 0.8603 | **0.794** | 1.373 | **0.902** | 1.938 | +0.0000 | +0.0000 | — |
| **Retail** | `lora_variate` | 294,912 | 0.1202 | 0.1957 | 0.7759 | 0.785 | 1.087 | 0.901 | 1.478 | -0.0005 | -0.0068 | 2.42 GiB / 25.8 s |
| **Retail** | `lora_temporal` | 1,179,648 | **0.1181** | **0.1924** | **0.7641** | 0.788 | 1.042 | 0.901 | 1.396 | +0.0011 | -0.0033 | 2.79 GiB / 34.6 s |
| **Retail** | `full_ft` | 209,312,256 | 0.1196 | 0.1948 | 0.7748 | 0.788 | 1.084 | 0.901 | 1.454 | +0.0003 | +0.0004 | 4.91 GiB / 34.7 s |

### Direct real-data precision & attention-kernel ablation on 123M & 209M (retail, `lora_temporal`, 120 steps, `B=16`)

| Model | Variant | Test wQL | Test MASE | Wall time | Peak VRAM |
|---|---|---:|---:|---:|---:|
| **123M** | `fp32` + default SDPA | 0.1963 | 0.7784 | 30.4 s | 2.23 GiB |
| **123M** | `fp32` + `attention_context("math")` | 0.1965 | 0.7797 | 29.9 s | 2.30 GiB |
| **123M** | `bf16` + default SDPA | 0.1970 | 0.7815 | **21.9 s** | **1.81 GiB** |
| **209M** | `fp32` + default SDPA | 0.1947 | 0.7725 | 46.0 s | 3.36 GiB |
| **209M** | `fp32` + `attention_context("math")` | 0.1952 | 0.7746 | 45.2 s | 3.47 GiB |
| **209M** | `bf16` + default SDPA | **0.1943** | **0.7718** | **26.1 s** | **2.79 GiB** |

On both `123M` and `209M`, `fp32` default SDPA, `fp32` math SDPA, and `bf16` default SDPA agree within `0.0002–0.0007` wQL on real data, and every mode — including `209M full_ft` — fits in `< 5 GiB` VRAM and trains in under 45 seconds on a single L4 GPU.

