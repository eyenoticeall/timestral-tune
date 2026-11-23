# Practical Value Benchmarks — Pre-Registered Plan & Working Notes

**Status:** In progress (branch `practical-benchmarks`, off `main` at `f4c7eb7`, package version `0.1.0`).
**Goal:** Test whether fine-tuning Timestral with `timestral-tune` is worth using in practice — measured on decision-relevant metrics (interval width at matched coverage, Winkler interval score, service-level newsvendor pinball cost), calibration vs fine-tuning, training-length and data-size scaling curves, three non-hourly strictly post-release niche datasets against carefully tuned gradient-boosted trees, covariate ablation, multi-seed 123M/209M tables, and serving latency/throughput.

All comparisons, metrics, dataset selections, and success thresholds below were written down **before running any fine-tuning experiment** in this work package. Every experiment run will be reported verbatim in `docs/BENCHMARKS.md`, including null or negative results.

---

## Ground Rules

1. **Validation-only model selection:** All hyperparameters (classical ridge $\alpha$ and raw/log target transform, GBT leaf/depth/learning-rate/regularisation/raw-vs-normalised target, and Timestral early-stopping checkpoints via `restore_best=True`) are chosen strictly on the chronological validation split. Never on test.
2. **Three seeds & paired cluster CIs:** Every fine-tuned configuration runs across `seeds = (0, 1, 2)`. Paired 95% confidence intervals come from the 10,000-resample paired cluster bootstrap (`scripts/benchmarks/significance.py`), clustered separately by **series** and by **non-overlapping time block** (using seed-0 predictions for exact per-window pairing and also reporting the 3-seed mean/std and seed-averaged ensemble CI where relevant).
3. **Common scoring path:** Every Timestral forecast runs through vendored upstream `TimestralForecaster.forecast()` with its defaults (`symmetric_averaging=True`, `clip_nonnegative=True`), scored in raw units via `timestral_tune.utils.metrics`.
4. **Resource accounting:** Every training and inference run records wall-clock time (`elapsed_s`) and peak GPU memory (`peak_vram_gib` on CUDA; allocated/driver memory on Apple GPU).
5. **Contamination & licence hygiene:** Every dataset has a pinned Hugging Face Hub revision, a permissive licence permitting derived benchmark publication, and an explicit contamination note relative to Timestral's `2026-10-27` release date.
6. **README untouched:** `README.md` results and `assets/results.png` are not modified in this branch; proposed updates go in the maintainer summary.

---

## Metric Definitions

Let $\hat{q}_\tau(i, h)$ be the predicted $\tau$-quantile for window $i$, horizon step $h$, and $y(i, h)$ the finite ground-truth observation.

- **Winkler Interval Score** at nominal central coverage $1 - \alpha \in \{0.80, 0.90\}$ ($\alpha \in \{0.20, 0.10\}$, $l = \hat{q}_{\alpha/2}$, $u = \hat{q}_{1-\alpha/2}$):
  $$S_\alpha(l, u; y) = (u - l) + \frac{2}{\alpha}(l - y)\,\mathbb{1}[y < l] + \frac{2}{\alpha}(y - u)\,\mathbb{1}[y > u]$$
  Strictly proper in the data's raw units; combines interval width and miscoverage penalty (`10×` at 80%, `20×` at 90%).
- **Matched-coverage interval width:** Fit `ConformalCalibrator(target_coverage=(0.80, 0.90), bucket_size=32)` strictly on each model's validation split, apply to test, and report achieved test coverage and mean test interval width alongside the paired 95% CI on relative width reduction `(width_tuned_cqr - width_zs_cqr) / width_zs_cqr`. Note: for fine-tuned models, validation is also used to select `best_step` (1 discrete choice among 4–12 checkpoints); achieved test coverage and the proper Winkler score verify whether CQR over-narrows.
- **Service-level pinball (newsvendor) cost** at $\tau \in \{0.80, 0.90, 0.95\}$:
  $$\text{Pinball}_\tau(\hat{q}_\tau, y) = \text{mean}\bigl[\max(\tau(y - \hat{q}_\tau), (\tau - 1)(y - \hat{q}_\tau))\bigr]$$
  Reported in raw units, empirical fill rate $\mathbb{P}(y \le \hat{q}_\tau)$, and relative `%` change vs released model (with paired 95% CIs).
- **Standard raw-space metrics:** `wql`, `pinball`, `mase`, `coverage_80/90`, `width_80/90`, and equal-weighted 3-domain GIFT-Eval macro forgetting (`ett1`, `jena_weather`, `loop_seattle`).

---

## Pre-Registered Experiments, Comparisons & Success Thresholds

### Work Package 1 — Decision metrics on existing datasets (+ 3-seed main table for all 4 sizes)

- **Datasets:** Existing `retail_demand` (`FreshRetailNet-50K@c374e85975c8`, CC BY 4.0, test `2024-06-21`..`2024-07-02` [pre-release], 64 SKUs × 6 blocks = 384 windows) and `realised_volatility` (`twelvedata/financial-world-model@da186e5fd362`, MIT, test `2026-11-03`..`2026-11-20` [strictly post-release], 51 equities × 3 blocks = 153 windows), plus extended `realised_volatility_2026_10m` (`2026-01`..`2026-11-20` [mostly pre-release], 51 equities × 49 blocks = 2,499 windows) for multi-regime statistical power on `5M` and `28M`.
- **Models & modes:** All four sizes (`5M`, `28M`, `123M`, `209M`), `zero_shot`, `zero_shot + CQR`, `zero_shot + ACI`, and all four gradient modes (`head_only`, `lora_variate`, `lora_temporal`, `full_ft`) at shipped defaults (`5M`/`28M`: `max_steps=200, batch_size=32`; `123M`/`209M`: `max_steps=160, batch_size=32, full_ft_batch_size=16, eval_batch_size=16` matching `docs/BENCHMARKS.md` §1 & §3), 3 seeds (`0, 1, 2`), evaluated raw, `+ CQR`, and `+ ACI`. Also evaluated against tuned GBT (`HistGradientBoostingRegressor(loss="quantile")`) on the exact same splits.
- **Pre-registered comparisons & thresholds:**
  1. *Width at matched coverage:* `tuned + CQR` vs `zero_shot + CQR` at 80% and 90%. **Success threshold:** mean interval width is **at least 10% narrower** (`<= -10%`) with achieved test coverage within 2 percentage points of (or higher than) `zero_shot + CQR`, and series-clustered 95% CI on width reduction strictly below 0.
  2. *Winkler interval score (raw & calibrated):* `tuned` vs `zero_shot` and `tuned + CQR` vs `zero_shot + CQR` at 80% and 90%. **Success threshold:** relative Winkler reduction **>= 5%** with series-clustered 95% CI strictly below 0.
  3. *Service-level cost (0.80, 0.90, 0.95):* `tuned` vs `zero_shot` raw-unit pinball cost. **Success threshold:** at `tau = 0.90`, relative cost reduction **>= 5%** with series-clustered 95% CI strictly below 0.
  4. *Calibration as released:* Report 80%/90% coverage and width of `zero_shot`, `zero_shot + CQR`, and `zero_shot + ACI` across all 4 sizes. **Diagnostic threshold:** calibrated coverage within ±3 percentage points of nominal (`[0.77, 0.83]` for 80%; `[0.87, 0.93]` for 90%) on stationary splits, documenting regime-shift lag on volatility.
  5. *Fine-tuning + calibration vs calibration alone:* `tuned + CQR` vs `zero_shot + CQR` and `tuned + ACI` vs `zero_shot + ACI` on Winkler 80%/90%, wQL, and 0.90 pinball. **Success threshold:** `tuned + CQR` beats `zero_shot + CQR` by **>= 5%** on Winkler_80/90 with series 95% CI strictly below 0.
  6. *Cross-size claim:* Paired cluster bootstrap 95% CI for fine-tuned `5M` (best validation mode and all 4 modes) against released `28M`, `123M`, and `209M` on `wql`, `winkler_80`, `pinball_0.90`, and CQR `width_80`. **Success threshold:** 95% CI strictly below 0 (fine-tuned `5M` beats larger zero-shot model).

### Work Package 2 — Where the gain levels off (scaling curves)

- **Datasets & models:** `retail_demand` and `realised_volatility`, `5M` and `28M`, `lora_temporal` (`1e-3`) and `full_ft` (`1e-4`), 3 seeds (`0, 1, 2`).
- **Training length curve:** `max_steps ∈ {200, 500, 1000, 2000}`, each an independent cosine-schedule run with `warmup_steps = max(10, min(100, max_steps // 10))`, `eval_every = 40` (at 200) or `50` (at 500/1000/2000), `early_stopping_patience = 6`, `restore_best = True`.
  - **Question & threshold:** Does training beyond 200 steps add **>= 2% further relative wQL reduction** without pushing macro GIFT-Eval forgetting above `+5%`? Record where validation early stopping triggers and where test gain saturates (< 1% incremental gain).
- **Data size curve:** `train_series_fraction ∈ {0.10, 0.30, 1.00}` (`retail`: 6, 19, 64 series; `volatility`: 5, 15, 51 series; seeded subset of training series per seed; identical full `val` and `test` sets), `max_steps = 500`, `eval_every = 25`, `early_stopping_patience = 5`, `restore_best = True`.
  - **Question & threshold:** What fraction of the 100%-data wQL gain is already captured at 10% and 30% of training series (threshold: **>= 50% of full gain at 10%**, **>= 80% at 30%**), and does small-data fine-tuning inflate forgetting above `+5%`?

### Work Package 3 — Three non-hourly, post-release niche datasets vs tuned GBT

Selected **strictly before running any fine-tuning** using zero-shot `Timestral-28M` vs classical baselines (`Apple M4 fp32` screening, 2026-11-23):

| Dataset | Pinned HF Hub source | Licence | Sampling rate | Series & covariates | Post-release test period | Pre-selection screening (zero-shot 28M vs classical Ridge) |
|---|---|---|---|---|---|---|
| **`de_power_15m`** | `akderekaan/de-power-forecast-data@2dba9a08d042` | CC BY 4.0 | **15-minute** (`ctx=384, H=96, s=96`, aligned to `00:00 UTC`) | 10 German regional TSO×tech series (`50HZ`/`AMP`/`TBW`/`TTG` × `solar`/`wind_on`/`wind_off`); known-future: ENTSO-E Day-Ahead TSO forecast (`a69_da`, published D-1) + quarter-hour/dow/doy | **`2026-10-28`..`2026-11-21`** (strictly post-release; 25 daily blocks × 10 series = 250 windows) | `naive=0.4649`, `ridge=0.1004`, `28M_zs=0.1175` (**`+17.1%` worse than classical**) |
| **`india_air_quality_daily`** | `AkshayKarthick/india-weather-air-quality@ba3f7ad78883` | CC BY 4.0 | **Daily** (`ctx=128, H=5, s=7`, aligned to `2026-10-28`) | 50 Indian cities (`pm2_5_mean`); 12 past co-pollutant & weather covariates + 3 known calendar covariates | **`2026-10-28`..`2026-11-21`** (strictly post-release; 5 blocks × 50 cities = 250 windows) | `log_ridge=0.2163`, `28M_zs=0.2312` (**`+6.9%` worse than classical**) |
| **`sl_marine_waves_daily`** | `tharinduperera/sl-fisheries-weather-daily@3bca7287be47` | CC BY 4.0 | **Daily** (`ctx=128, H=4, s=7`, aligned to `2026-10-28`) | 16 Sri Lanka coastal fishing sites (`wave_height_max`, consistent `era5_ocean` + `era5` reanalysis 2020–2026); 9 past ocean/weather covariates + 2 known seasonal covariates | **`2026-10-28`..`2026-11-16`** (strictly post-release; 5 blocks × 16 sites = 80 windows) | At `H=7`: `naive=0.1518`, `ridge=0.0829`, `28M_zs=0.0979` (**`+18.0%` worse than classical**) |

- **All three** have zero missing values, zero timestamp gaps, permissive `CC BY 4.0` licences, strictly post-`2026-10-27` test splits (zero pretraining contamination on test targets), two non-hourly sampling rates (`15-minute`, `daily`), and released `Timestral-28M` trailing the classical baseline by **`+6.9%` to `+18.0%`** (all > 5%).
- **Methods compared on every dataset:**
  - `seasonal_naive` (+ validation residual quantiles)
  - `classical_domain` (per-horizon direct ARX/bias-correction Ridge tuned on validation over $\alpha \in \{10^{-2}..10^4\}$ and raw vs $\log(1+y)$ target space, given exact same covariates)
  - `gbt_tuned` (stacked global `(window, h)` `HistGradientBoostingRegressor(loss="quantile")` with lags, exact seasonal lag at `t+h`, rolling stats, past covariates, exact step-aligned known-future covariates at `t+h`, native categorical `series_id`, and `h`; tuned strictly on validation wQL across 24 configs: `target_transform ∈ {raw, context_norm}` × `(max_leaf_nodes, min_samples_leaf) ∈ {(15,50), (31,30), (63,20)}` × `lr ∈ {0.03, 0.08}` × `l2 ∈ {0.0, 1.0}`, then all 21 quantiles fitted at the validation winner; scored both raw and `+ CQR`)
  - `28M zero_shot`, `28M + CQR`, `28M + ACI`
  - `28M` fine-tuned across all four modes (`head_only`, `lora_variate`, `lora_temporal`, `full_ft`) at shipped defaults (`max_steps=400, eval_every=40, early_stopping_patience=5, restore_best=True`, 3 seeds), scored raw, `+ CQR`, and `+ ACI` (plus `5M` for cross-size comparison).
- **Pre-registered success thresholds:**
  1. *Beats released 28M:* Fine-tuning cuts wQL and Winkler_80 by **>= 10%** vs `28M zero_shot` with series 95% CI strictly below 0.
  2. *Beats tuned GBT (the practical user question):* Best validation-selected fine-tuned Timestral mode beats `gbt_tuned` by **>= 3%** on wQL and/or Winkler_80/Pinball_0.90 with paired 95% CI strictly below 0. If `gbt_tuned` ties or beats fine-tuned Timestral, report plainly that gradient boosting is the better practical choice for that domain.

### Work Package 4 — Breadth & serving cost

1. **Three-seed 123M & 209M main table:** Covered by WP1 (`seeds = (0, 1, 2)` across `head_only`, `lora_variate`, `lora_temporal`, `full_ft`, `zero_shot`, `CQR`, `ACI` on `retail_demand` and `realised_volatility`). Check whether 3-seed means match the existing single-seed numbers in `docs/BENCHMARKS.md` §3 within ±0.003 wQL.
2. **Retail covariate ablation (`5M` & `28M`, 3 seeds):** Compare `with_covariates` (3 past + 5 known) vs `no_covariates` (univariate hourly sales only) for `zero_shot`, `zero_shot + CQR`, `head_only`, `lora_temporal`, `full_ft` (noting `lora_variate` requires `>= 2` rows per T10). Measure how much of both zero-shot accuracy and the fine-tuning gain comes from covariates vs temporal adaptation.
3. **Inference latency & throughput (`5M`, `28M`, `123M`, `209M`):** Upstream `TimestralForecaster.forecast()` (`symmetric_averaging=True` default, plus `False` reference) on:
   - **NVIDIA L4** (`bf16` and `fp32`),
   - **x86 CPU** (4 threads on the same L4 VM), and
   - **Apple M4 CPU & MPS** (`fp32`).
   Workloads: `(ctx=256, H=48)` and `(ctx=512, H=96)` for `uni` and `cov=8` at `B=1` (interactive latency: `p50/p95 ms`) and `B=16, 64` (batched throughput: `series/s`, `ms/series`, peak VRAM GiB). Quantify exact speed-up and VRAM savings of serving fine-tuned `5M` in place of released `28M` / `123M` / `209M`.

---

## Execution Log

Completed `2026-11-23` on branch `practical-benchmarks`. Following maintainer review, all six GBT
baselines (`retail_demand`, `realised_volatility`, `realised_volatility_2026_10m`, `de_power_15m`,
`india_air_quality_daily`, `sl_marine_waves_daily`) and all WP1 (`5M`–`209M`) and WP3 (`28M`, `5M`)
paired comparisons were regenerated together in one single pass from the tip code on a short-lived
OS-Login NVIDIA L4 VM (`g2-standard-32`, `OMP_NUM_THREADS=4` per baseline, `bf16` training on L4).
All temporary VMs were deleted immediately afterward with verified empty GCP project metadata and
zero leftover GPU VMs/disks; `README.md` and `assets/results.png` were left untouched on the branch.

*After the merge (2026-11-24):* the branch was squash-merged into `main`. The README's Results section
now states the tuned-GBT comparison for `28M` in one sentence; the fuller plain-language account,
including `5M`, is the Summary at the top of `docs/BENCHMARKS.md` (`docs/DEVLOG.md`, 2026-11-24). The
chart is unchanged.

### Methodological disclosures (including all three GBT search revisions)

- **WP3 screening split & India rule (`docs/benchmarks/wp3_screening.json`):** The initial exploratory
  screening numbers written in §WP3 above had evaluated zero-shot `28M` vs an ad-hoc Ridge baseline on
  the **test split**, which can inflate "beats the released model" on the three new datasets (it does
  not favour Timestral over GBT). Re-checking with committed `screen_wp3_candidates.py` against the
  final `run_tuned_classical` baseline confirms `de_power_15m` (`+6.5%` val / `+15.4%` test) and
  `sl_marine_waves_daily` (`+8.5%` val / `+10.0%` test) meet the `>= 5%` rule on validation, whereas
  `india_air_quality_daily` (`−4.5%` val / `−0.2%` test) does **not** meet the `>= 5%` classical-gap
  rule and qualifies under the niche-domain branch only.
- **German power `a69_da` publication timing:** Under EU Transparency Regulation 543/2013 (Art. 14.1.d),
  ENTSO-E day-ahead forecasts (`a69_da`) are published by 18:00 CET/CEST on D−1 for the German CET/CEST
  delivery day. Because `load_de_power_15m` aligns windows to `00:00 UTC`, the first 92 winter (88 summer)
  steps come from day D's D−1 publication while the final 4 winter (`23:00–23:45 UTC`) / 8 summer
  (`22:00–23:45 UTC`) steps belong to day D+1's publication (17:00/16:00 UTC on day D). **Consequence:**
  for the last hour of each German power window in winter (and the last two hours in summer), every method
  (Timestral and all baselines alike) sees a day-ahead forecast that had not yet been published at the
  `00:00 UTC` forecast origin. Source parquets carry no publication timestamps; Timestral and all baselines
  receive the exact same `da_mw` array, so relative comparisons are symmetric. Exact zero-target fraction
  in `de_power_15m` training targets is `10.22%` (`41,496 / 406,080`; `19.59%` val, `23.85%` test).
- **3-seed cluster+seed paired bootstraps:** Every paired CI in `wp1_decision_metrics.json`,
  `wp3_unfamiliar_domains.json`, and `wp4_covariate_ablation.json` draws both training seed
  `s ∈ {0, 1, 2}` and clusters (`10,000` resamples), with zero consistency violations between 3-seed
  means and 95% CIs.
- **Three GBT baseline search revisions & cross-machine spread:**
  1. *Revision 1 (initial run):* 24-config grid (`2` scalings `{raw, context_norm}` × `3` leaf pairs
     `{(15,50), (31,30), (63,20)}` × `2` LRs `{0.03, 0.08}` × `2` L2 `{0.0, 1.0}`, fixed `max_iter ∈
     {200, 350}`, no early stopping). Four of six splits hit the smallest leaf corner `(15, 50)`.
  2. *Revision 2 (first review pass):* ~46-config two-stage grid (`2` scalings × `5` leaf pairs
     `(4,200)..(63,20)` × `3` LRs `{0.02, 0.05, 0.10}`, Stage 1 fixed `l2=1.0`, Stage 2 `l2 ∈ {0, 1,
     10, 50}`, chronological early stopping `1..800` via `staged_predict`). Exposed a scikit-learn
     `loss="quantile"` bug on zero-inflated `de_power_15m`: `l2=1.0` froze `tau=0.10` at a constant
     (`281.63`), rejecting `raw` and forcing `context_norm` (`0.1143`).
  3. *Revision 3 (tip code, single pass across all 6 splits):* 45 Stage-1 configs (`3` scalings
     `{raw, context_norm, median_residual}` × `5` leaf pairs × `3` LRs at `l2=0.0`) carrying the best
     Stage-1 config of each scaling into Stage 2 across `l2 ∈ {0, 1, 10, 50}` and adjacent leaf pairs
     (`71–85` total configs), with exact per-quantile chronological early stopping (`1..800`) via
     `staged_predict` and separate 4-thread CPU `fit_time_s` / `tuning_time_s` reporting (no GPU-vs-CPU
     speed ratios). All six splits and all WP1/WP3 comparisons in the committed JSONs were regenerated
     together from Revision 3 in one pass. Four of six splits (all four post-release slices) have zero
     boundary hits; retail (`max_leaves=63`, `max_iter`) and 10m volatility (`max_iter`) touch upper
     bounds only.
  - *Cross-machine baseline spread:* running the same `HistGradientBoostingRegressor` code across
    different CPU architectures (macOS arm64 vs Linux x86_64) introduces up to `~0.4%` relative wQL
    spread from floating-point histogram/tie-breaking differences (e.g., `retail_demand` `0.2121` on
    Apple M4 vs `0.2114` on x86_64, `0.33%`; `sl_marine_waves_daily` Revision 1 `0.0824` vs `0.0823`);
    the committed single-pass Revision-3 numbers all come from one uniform x86_64 machine (`g2-standard-32`).

### Outcome against pre-registered thresholds

- **WP1 — Width at matched coverage (`>= 10%` narrower within `2 pp` of `zero_shot + CQR`):**
  **PASSES cleanly on retail `28M`** (`−19.9%`, `[-21.5%, -18.4%]`, cov `+1.2 pp`) and **`de_power_15m`**
  (`−31.9%`, cov `−1.4 pp`), and is **borderline on retail `5M`** (`full_ft` `−16.0%` at `−2.0 pp`; LoRA
  `−14.6%..−16.0%` at `−1.6..−1.8 pp`). **FAILS the equal-coverage condition on retail `123M` (`−4.6 pp`)
  and `209M` (`−4.0 pp`)** (released CQR over-covered at `83%–84%`, fine-tuned landed near `79%–80%`),
  **MIXED/FAILS on volatility** (`28M`/`123M` post-release `full_ft` hit `−10.3%`/`−12.6%`, `5M` `−2.4%`,
  `209M` `−8.9%`, 10m 2026 `−3.1%` with CI `[-6.8%, +0.5%]`), and **FAILS on daily air/marine**
  (`india_air_quality_daily`: `+6.1%` wider for `+4.5 pp` cov; `sl_marine_waves_daily`: `−7.2%`).
- **WP1 — Winkler score (`>= 5%` vs zero-shot and vs CQR alone):** **PASSES** across all four sizes on
  retail (`−11.3%` to `−18.4%`, confirming real interval-quality gains even on `123M`/`209M`), on
  `5M`/`28M`/`209M` post-release volatility (`−5.3%` to `−7.5%`; `123M` misses at `−3.4%..−3.6%`), on
  `28M` 10m volatility (`−5.8%`), and on all three WP3 post-release domains (`−9.7%` to `−23.9%`).
- **WP1 — 90% service-level cost (`>= 5%`):** **PASSES** across all four sizes on retail (`−13.7%` to
  `−22.9%`), on `5M`/`28M`/`209M` post-release volatility (`−5.2%` to `−6.9%`; `123M` misses at `−1.8%..−2.5%`),
  on `28M full_ft` 10m volatility (`−5.3%`), and across all three WP3 domains (`−15.0%` to `−26.1%`).
- **WP1 — Cross-size claim (`5M` fine-tuned vs larger released):** **PASSES decisively on retail**
  (`−8.1%` vs `28M`, `−6.8%` vs `123M`, `−9.6%` vs `209M`, all 3-seed CIs `< 0`) and on all three WP3
  domains vs `28M ZS` by series (`−5.8%` to `−16.8%`). On volatility, `5M full_ft` beats `28M ZS`
  (`−4.1%` post-rel, `−3.8%` 10m) and **ties `123M`/`209M ZS` (`−0.1%`/`−0.1%`)**.
- **WP2 — Scaling curves:** `28M` retail plateaus by `200` steps (`0.1951` across `200..2,000`); `5M`
  gains `1.3 pp` at `500` steps then flattens. `30%` of training series (`15–19` series) captures
  `80%–113%` of full-data gain (`10%` / `5–6` series: `41%–78%`).
- **WP3 — Unfamiliar non-hourly post-release domains vs Revision-3 GBT:**
  - *Beats released `28M` (`>= 10%`):* **PASSES** on `de_power_15m` (`−23.0%`) and `sl_marine_waves_daily`
    (`−22.6%`); narrowly misses on `india_air_quality_daily` (`−9.0%`). *(Caveat: test-split initial screening.)*
  - *Beats tuned GBT (`>= 3%`) — `28M` and `5M`:*
    - **Pre-release / mostly-pre-release splits (PASS):** fine-tuned `28M full_ft` leads cleanly on
      retail (`−7.7%`, `[-9.5%, -5.9%]`, time `[-11.3%, -4.7%]`) and 10m volatility (`−6.0%`,
      `[-7.9%, -4.2%]`, time `[-8.6%, -3.3%]`). Fine-tuned `5M full_ft` also beats tuned GBT on both:
      retail `−4.7%` (`[-6.4%, -3.0%]`, time `[-9.2%, -0.4%]`, `47/64`) and 10m volatility `−3.9%`
      (`[-5.6%, -2.1%]`, time `[-6.2%, -1.4%]`, `34/51`).
    - **All 4 clean post-release test slices (FAIL — registered conclusion):** fine-tuned `28M full_ft`
      fails to beat tuned GBT on any of the four post-release slices: 18-day post-release volatility
      `+2.7%` (`[-1.5%, +7.3%]`, GBT ahead), `de_power_15m` `+2.3%` (`[-0.9%, +6.4%]`, GBT ahead),
      `india_air_quality_daily` `−0.1%` (`[-2.1%, +2.2%]`, dead tie), and `sl_marine_waves_daily`
      `−5.1%` (`[-13.3%, +2.9%]`, time `[-20.5%, +10.5%]`, not separable). Fine-tuned `5M` trails GBT
      on all four post-release slices — significantly on three of the four: 18-day post-release
      volatility `+4.5%` (`[+0.9%, +8.3%]`, `15/51`; all four `5M` modes `+3.9%..+4.7%`), `de_power_15m`
      `+10.4%` (`[+6.5%, +14.9%]`, `0/10`), `india_air_quality_daily` `+3.4%` (`[+0.5%, +6.7%]` for
      `full_ft`; `+1.9%`, `[-0.3%, +4.1%]` for `lora_variate`), and `sl_marine_waves_daily` `+3.3%`
      (`[-5.2%, +12.6%]`, `5/16`). **Registered conclusion:** per §WP3 above, because fine-tuned
      Timestral (`5M` or `28M`) only ties or loses to tuned GBT across all four clean post-release
      domains, **gradient boosting is the better practical choice for those domains.**
- **WP4 — Breadth & serving:** `123M`/`209M` 3-seed means match §3 within `0.0022` wQL. Retail covariate
  ablation shows zero-shot ignores covariates (`+0.1%`/`+0.4%`) while fine-tuning unlocks `−2.1%`/`−2.6%`
  (`54–56/64` SKUs). Serving `5M` in place of `209M` is `11.9×` faster on L4 `bf16` (`B=64`, `6.0×` less
  VRAM), `18.9×` on L4 `fp32`, `24.4×` on x86 CPU (`B=16`, 2 default physical-core threads), and `17.5×`/`25.1×`
  on Apple M4 CPU (`4` default P-core threads) / MPS.
- **Artefacts:** `docs/benchmarks/{wp1_decision_metrics,wp2_scaling,wp3_unfamiliar_domains,wp3_screening,wp4_covariate_ablation,wp4_inference_latency_cuda_cpu,wp4_inference_latency_apple}.json`, `docs/BENCHMARKS.md` §6, `docs/VERIFICATION.md` §16.

