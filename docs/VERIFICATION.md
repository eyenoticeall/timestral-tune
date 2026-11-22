# Verification Record

Measured evidence behind every **[V]** tag in `docs/SPEC.md`. Each section names the probe or test that
reproduces it. Probes live in `scripts/probes/` and run the released checkpoints through the library.

When you add a `[V]` claim to the spec, add its probe or test and a row here in the same change.

## Environment of the recorded runs

| | |
|---|---|
| Date | 2026-11-21 |
| Machine | Apple M4, 10 CPU threads, 16 GB memory |
| PyTorch | 2.15.0 for all probes; the test suite also ran on 2.4.0 and 2.8.0 (§10) |
| Python | 3.13 for the probes; 3.10, 3.12 and 3.13 for the test suite |
| Checkpoints | Timestral-5M and Timestral-28M at the Hub revisions pinned in `docs/UPSTREAM.md` |

CUDA results (§13) come from a second machine: a Google Cloud VM with one NVIDIA L4 (23 GB), 4 CPU threads,
Ubuntu 22.04, PyTorch 2.15.0+cu132, Python 3.12.16.

Not verified anywhere: the 123M and 209M checkpoints.

## 1. Model structure

Probe: `uv run python scripts/probes/model_facts.py --size 5M` (and `--size 28M`).

| Claim | 5M | 28M |
|---|---|---|
| Modules named `q_proj`, `v_proj` or `variate_attn` | none | none |
| Attention modules (temporal / variate) | 8 / 2 | 12 / 3 |
| Variate attention located at layers | 3, 7 | 3, 7, 11 |
| Parameters of each attention module | `wQKV.weight`, `wO.weight`, `_gate.weight` | same |
| `wQKV.weight` shape | (576, 192) | (1152, 384) |
| Total parameters | 5,208,960 | 27,821,184 |
| Head parameters (share) | 792,576 (15.2%) | 1,880,064 (6.8%) |
| Head size equals `4D² + 3360·D` | yes | yes |

## 2. Roles, gradients and the firewall

Probe: `model_facts.py`. One series with a target, a past-only covariate and a known-future covariate.
The firewall is also asserted permanently by acceptance test T6.

| Claim | 5M | 28M |
|---|---|---|
| Gradient of a target-only loss w.r.t. the past-only covariate's context (sum of abs.) | 8.5e-1 | 8.4e-1 |
| … w.r.t. the known-future covariate's context | 5.3e0 | 2.5e0 |
| … w.r.t. the known-future covariate's horizon values | 1.1e0 | 2.1e-1 |
| Noise on the target's context → max change in (target, past-only, known-future) outputs | 14.0, 0, 0 | 9.4, 0, 0 |
| Noise on the past-only covariate → same | 0.75, 6.3, 0 | 0.77, 15.2, 0 |
| Noise on the known-future covariate → same | 2.3, 3.3, 41.5 | 2.2, 4.8, 8.3 |

Covariate inputs receive non-zero gradients from a target-only loss, so "zero gradients on covariates" is
not a valid leakage test. Targets never influence covariate rows, and past-only rows never influence
known-future rows. The zeros are exact.

## 3. Single-row variate attention

Probe: `model_facts.py`. A univariate series with no covariates; random noise added to slices of
`wQKV.weight` in every variate-attention module.

| Perturbed rows | Max output change, 5M | 28M |
|---|---|---|
| Rows `[0, 2D)` (query and key) | 0 (exact) | 0 (exact) |
| Rows `[2D, 3D)` (value) | 4.9 | 16.3 |

This confirms the last third of the rows is the value projection and the first two-thirds are query and
key together. The order of query and key within the first two-thirds is from the code only.

## 4. Training forward against upstream inference

Tests: `uv run pytest tests/acceptance/test_forward_parity.py tests/test_collator.py tests/test_dataset.py`.
The last column says which models each claim ran on; "all" means the tiny random model, 5M and 28M.

| Claim | Measured | Models |
|---|---|---|
| T1: denormalised training forward against `predict_block`, no covariates | Bit-identical on PyTorch 2.4.0 and 2.15.0; 8e-10 on 2.8.0. Bound: 1e-6 | all |
| T2: denormalised training forward against `predict_block`, trimmed `(L + 2h)` covariate layout (including past-only covariates) | Float32: 6.1e-8 to 5.2e-6 (bound 1e-5). Float64: 1.1e-16 to 9.8e-15 (bound 1e-12). Trimmed mask bit-identical to sliced `build_two_stream_time_mask` with zero edges to dropped tokens | all |
| T3: median-folded frame against `_refine_block`, and the full pipeline against `forecast()` | 1e-7 to 2e-7 relative. Bound: 1e-5 | all |
| T3: `true_values` frame of every horizon patch after the first, against upstream's `model.scale` on the context extended by the true values | Within 1e-6 | all |
| Collated batch against upstream's `_assemble_rows` and `_encoding_target` | Every tensor exactly equal, including missing values, a partial final patch and several targets | all |
| Categorical encodings against upstream's `_encode_categorical` | Exactly equal | tiny only |
| Extra left padding (1 and 3 patches beyond upstream's), both layouts | Forecast unchanged within 1e-5 | all |
| Frame → dataset → collator → forward → frame, against `forecast()` on the same raw values, for polars, pandas and pyarrow; and with two targets and categoricals | Within 1e-5 | tiny only |
| The batch the dataset and collator build passes `check_forward_parity` | Pass | all |

## 5. Apple GPU (MPS) parity

Probe: `uv run python scripts/probes/device_parity.py --device mps --size 5M` (and `--size 28M`). Asserted
permanently by acceptance test T14. Without the device shims the official code raises
`TypeError: Cannot convert a MPS Tensor to float64` on this backend
(`tests/test_loader_device.py::test_plain_upstream_fails_on_mps_and_the_shims_fix_it`).

Forward, relative L2 difference from CPU:

| Case | 5M | 28M |
|---|---|---|
| `predict_block`, no covariates | 1.9e-7 | 1.9e-7 |
| `predict_block`, with covariates | 1.7e-6 | 3.4e-6 |
| `forecast`, horizon 96 | 1.0e-7 | 1.3e-7 |
| `forecast`, horizon 2,100 (block-autoregressive) | 1.8e-5 | 1.4e-6 |
| `forecast`, numeric and categorical covariates | 9.4e-8 | 9.3e-8 |

Autocast on MPS, forward relative difference from CPU fp32:

| | 5M | 28M |
|---|---|---|
| `bfloat16` | 1.4e-2 | 5.9e-2 |
| `float16` | 2.2e-3 | 7.5e-3 |

## 6. Float32 gradient accuracy

Probe: `uv run python scripts/probes/gradient_accuracy.py --device mps`.

Relative L2 error of the float32 training-loss gradient against a float64 reference. The reference uses
float64 weights and inputs on CPU and computes upstream's RMSNorm in float64 (upstream's own always
computes in float32; see below). RoPE still builds its table in float32, so it is not a pure float64
computation.

| Model | Layout | CPU | Apple GPU |
|---|---|---|---|
| Tiny random | univariate | 4.9e-7 | 3.6e-7 |
| Tiny random | multivariate | 3.5e-7 | 3.2e-7 |
| Tiny random | covariates | 2.8e-7 | 2.9e-7 |
| Tiny random | past-only covariates | 3.5e-7 | 3.5e-7 |
| 5M | univariate | 1.1e-3 | 3.1e-5 |
| 5M | multivariate | 2.1e-3 | 5.4e-5 |
| 5M | covariates | 1.2e-3 | 7.7e-5 |
| 5M | past-only covariates | 5.3e-4 | 5.9e-6 |
| 28M | univariate | 8.2e-4 | 1.8e-5 |
| 28M | multivariate | 5.0e-4 | 1.4e-5 |
| 28M | covariates | 9.1e-4 | 1.8e-4 |
| 28M | past-only covariates | 4.6e-4 | 9.5e-6 |

Across the five scenarios of acceptance test T14, on 5M and 28M (other data than the probe's;
`uv run pytest tests/acceptance/test_device_parity.py -k "gradients" -s` prints each value). The x86 CPU and
CUDA columns are from the VM of §13. One case, 5M with past-only covariates, is ill-conditioned in float32
(see below) and has its own rows and its own bound:

| Check | Bound | Apple CPU | Apple GPU | x86 CPU | CUDA |
|---|---|---|---|---|---|
| Tiny random model, every kernel | 1e-5 | 4.0e-7 | 4.5e-7 | 4.4e-7 | 3.5e-7 |
| Released, default kernel, all ten cases | 5e-2 | 3.4e-3 | 2.1e-3 | 2.4e-3 | 1.1e-2 |
| Released, explicit matmul-and-softmax, nine cases | 5e-3 | 6.1e-4 | 1.2e-3 | 2.1e-4 | 1.1e-4 |
| Released, PyTorch's unfused kernel, nine cases | 5e-3 | 3.7e-4 | 1.2e-3 | 5.4e-4 | 3.0e-4 |
| Near-tie case, default kernel | 5e-2 | 2.2e-3 | 2.1e-3 | 2.4e-3 | 1.1e-2 |
| Near-tie case, explicit matmul-and-softmax | 2e-2 | 1.2e-3 | 2.1e-3 | 5.3e-3 | 1.9e-3 |
| Near-tie case, PyTorch's unfused kernel | 2e-2 | 9.0e-3 | 2.1e-3 | 1.4e-3 | 1.4e-3 |
| `head_only` (everything but the head frozen), all models | 1e-5 | 2.3e-6 | 2.8e-6 | 1.7e-6 | 2.2e-6 |

One minus the cosine similarity with the reference (bound 1e-4): with the default kernel at most 5.7e-6,
2.1e-6, 2.5e-6 and 5.6e-5 in the same column order; with the unfused kernel at most 3.7e-5, 2.1e-6, 8.5e-7
and 8.5e-7.

The near-tie case shows that the residual error on such a batch is a matter of how each kernel happens to
round on each machine: the worst kernel is the unfused one on an Apple CPU, explicit attention on an x86
CPU, and the default kernel on CUDA. No kernel is reliably best on it.

Probe Experiment [2] (`default` vs `attention_context("math")` vs explicit matmul-and-softmax) and Experiment
[5] (strictly same-row peak logit with its own top-1/top-2 gap, and strictly same-row largest near-tie logit
with `gap < 10`):

| Model | Layout | CPU `default` / `math` / `explicit` | MPS `default` (= `math`) / `explicit` | Peak `|logit|` (same-row gap) | Max near-tie (same-row gap) |
|---|---|---|---|---|---|
| 5M | univariate | 1.1e-3 / 1.4e-5 / 2.6e-5 | 3.1e-5 / 1.9e-5 | 370,272 (4,753.4) at `layers.5.attns.0` | 13,718 (0.14) at `layers.3.attns.0` |
| 5M | multivariate | 2.1e-3 / 5.4e-5 / 5.4e-5 | 5.4e-5 / 2.2e-5 | 403,726 (2,760.9) at `layers.5.attns.0` | 319,403 (8.82) at `layers.5.attns.0` |
| 5M | covariates | 1.2e-3 / 2.3e-5 / 5.9e-5 | 7.7e-5 / 7.6e-5 | 378,399 (1,337.4) at `layers.5.attns.0` | 335,379 (6.97) at `layers.5.attns.0` |
| 5M | past-only | 5.3e-4 / 1.1e-5 / 7.3e-6 | 5.9e-6 / 9.9e-6 | 403,726 (2,760.9) at `layers.5.attns.0` | 13,377 (0.73) at `layers.3.attns.0` |
| 28M | univariate | 8.2e-4 / 8.3e-5 / 8.3e-5 | 1.8e-5 / 1.8e-5 | 203,146 (24,934.2) at `layers.1.attns.0` | 13,788 (4.04) at `layers.1.attns.0` |
| 28M | multivariate | 5.0e-4 / 1.2e-5 / 1.3e-5 | 1.4e-5 / 6.5e-6 | 291,241 (54,047.5) at `layers.1.attns.0` | 2,791 (8.11) at `layers.11.attns.1` |
| 28M | covariates | 9.1e-4 / 3.6e-4 / 3.9e-4 | 1.8e-4 / 3.7e-5 | 503,429 (20,194.0) at `layers.1.attns.0` | 37,548 (4.08) at `layers.1.attns.0` |
| 28M | past-only | 4.6e-4 / 1.4e-5 / 1.2e-5 | 9.5e-6 / 1.2e-5 | 291,241 (54,047.5) at `layers.1.attns.0` | 2,814 (6.07) at `layers.11.attns.1` |

Peak-layer weight norms, standalone attention, RMSNorm reference sensitivity, and the 15-seed 5M past-only
CPU sweep (Experiments [3]–[6] of the same probe):

| Experiment | Result |
|---|---|
| 5M `transformer.layers.5.attns.0` weight norms | `||norm.scale||_∞ = 5.01`, `||Wq||_2 = 141.6`, `||Wk||_2 = 82.4`, `||Wv||_2 = 17.8` |
| 28M `transformer.layers.1.attns.0` weight norms | `||norm.scale||_∞ = 2.10`, `||Wq||_2 = 196.1`, `||Wk||_2 = 160.6`, `||Wv||_2 = 16.5` |
| Standalone attention, CPU fp32 vs fp64, max `|logit|` 4.5 / 71.4 / 1,143 | default 5.5e-7 / 4.0e-6 / 5.0e-5; `math` 3.6e-7 / 3.8e-6 / 4.6e-5; `explicit` 3.5e-7 / 3.7e-6 / 4.8e-5 |
| Standalone attention, max `|logit|` 401,874 (large gap) | default 1.6e-2, `math` 5.6e-3, `explicit` 1.3e-2 |
| Standalone attention, max `|logit|` 348,488 (near-tie keys) | default 1.6e-2, `math` 1.5e-2, `explicit` 1.6e-2 |
| A float64-weights reference with upstream's float32 RMSNorm, against the reference with float64 RMSNorm, over the twelve cells above | 6.0e-8 to 1.3e-4 (largest: 28M covariates) |
| 5M past-only CPU sweep across 15 seeds (Experiment [6]) | `default` median 1.05e-3 (max 5.36e-3); `math` median 1.24e-5 (max 5.93e-5); `explicit` median 1.11e-5 (max 5.82e-5) |

Reading. The large Q/K weights in the released checkpoints produce peak attention logits of 203k–503k (a
random model has O(1) logits), where one float32 ULP is `0.015625` below 262k and `0.03125` from 262k to
524k, causing two effects:

1. **Fused SDPA backward** (CPU `FLASH_ATTENTION`, CUDA fused SDPA) loses accuracy on typical batches (4.6e-4
   to 2.1e-3 on CPU, up to 1.2e-2 on CUDA §13). Switching to PyTorch's unfused math kernel via
   `attention_context("math")` (no vendored edits) reduces CPU error by 2.5×–79× across the eight probe
   layouts (10×–79× on seven, 2.5× on 28M covariates) and cuts the 15-seed 5M past-only median 85×. On the
   Apple GPU `"math"` is identical to `"default"`.
2. **Float32 ULP quantization on near-tied large logits** affects every float32 kernel. When two keys in the
   same query row sit at large logits within an O(1) gap, float32's ULP quantizes the gap by 1–2 ULPs,
   shifting softmax weights by a few percent. Consequently `"math"` can be *less* accurate than `"default"`
   on individual batches: on T14's 5M past-only-covariates batch, `"math"` gives 9.0e-3 on CPU against
   2.2e-3 for `"default"` and 1.2e-3 for explicit matmul-and-softmax (`1 - cosine = 3.7e-5`; a one-off trace
   attributes this to one near-tie query at ~351k, §11).

Separately, upstream's float32 RMSNorm alone moved the old reference by 3.8e-3 on T14's 5M
past-only-covariates batch (`uv run pytest tests/acceptance/test_device_parity.py -k
"device_parity_gradients and cpu" -s`), so all references compute RMSNorm in float64 (`tests/helpers.py`
and `scripts/probes/_common.py`, without editing vendored code). The CUDA numbers in §13 use the same
reference.

`head_only` is not affected: with the body frozen the backward pass never enters attention or the body's
norms; this holds on CUDA too (§13). Forward outputs are unaffected (about 3e-7 from the reference, §11).

Consequence: T14 keeps its tight `5e-3` explicit-attention tier alongside the three existing tiers, adds a
separate check for `attention_context("math")` (spec D12). That helper is experimental and no training
path uses it; 20-step L4 CUDA trajectories in §13 show default SDPA suffices (spec D15).

## 7. Training-step timing, CPU against the Apple GPU

Probe: `uv run python scripts/probes/step_timing.py --device mps`. The library's own step: `compute_loss`
(normalised forward, median-folded frame, pinball loss), backward over all parameters, AdamW. Context 1,024
steps, horizon 96, batch of 32 series, fp32. Median of five steps after a warm-up step.

| Model | Batch | CPU | Apple GPU | Speed-up | GPU memory allocated |
|---|---|---|---|---|---|
| 5M | univariate | 124 ms | 56 ms | 2.2× | 0.3 GiB |
| 5M | 5 rows per series, with covariates | 492 ms | 276 ms | 1.8× | 2.3 GiB |
| 28M | univariate | 334 ms | 194 ms | 1.7× | 2.3 GiB |
| 28M | 5 rows per series, with covariates | 1,845 ms | 986 ms | 1.9× | 4.5 GiB |

Before trimming the `L` dead covariate decoder tokens (decision D16), the two covariate rows measured 907 ms
CPU / 474 ms MPS (5M) and 6,216 ms CPU / 1,957 ms MPS / 7.3 GiB (28M). Timings vary between runs. Memory
is the driver allocation after the step, cumulative over the script, not a per-configuration peak.

One-off 5-epoch `head_only` timing with per-epoch validation (24 covariate series, 72 windows, batch 16,
including epoch 1 cache build): 5M 615 → 151 ms on CPU (4.1×) and 1,002 → 290 ms on MPS (3.5×); 28M
1,705 → 393 ms on CPU (4.3×) and 1,384 → 385 ms on MPS (3.6×).

## 8. Leakage, loss mask, training, LoRA, conformal and export

Tests: `uv run pytest tests/acceptance tests/test_trainer.py tests/test_lora.py tests/test_conformal.py tests/test_export.py`.
Run on the tiny random model, 5M and 28M.

| Claim | Test | Result |
|---|---|---|
| Horizon target values changed in the raw frame leave inputs and predictions bit-identical | T4 | Pass |
| A past-only covariate changed after the origin changes nothing | T5 | Pass |
| Noise on targets changes no covariate row's hidden state or output; noise on past-only rows changes no known-future row | T6 | Exactly 0 |
| Noise on known-future horizon values changes the target forecast | T7 | Pass |
| Loss gradient is exactly zero at covariate rows, unobserved steps and horizon padding, under both frames | T8 | Pass |
| Freshly initialised LoRA leaves outputs unchanged (`< 1e-12` fp64, `< 1e-5` fp32); merged and unmerged agree (`< 1e-12` fp64, `< 1e-4` fp32) | T9 | Pass on tiny, 5M and 28M (variate and time axes) |
| `lora_variate` raises on single-row data; `allow_single_row=True` rejects `q`/`k` and permits `v`/`o`/`gate` | T10 | Pass on tiny, 5M and 28M (and via `TimestralTrainer`) |
| Exported checkpoint loads in vendored upstream `load_model` with `strict=True`, reproduces forecasts within `1e-4`, and writes `config.json`, `model.safetensors`, adapters (LoRA), `conformal.json`, `tune_metadata.json`, `LICENSE`, `NOTICE` | T11 | Pass on tiny, 5M and 28M across all four modes (`head_only`, `lora_variate`, `lora_temporal`, `full_ft`) |
| Categorical encodings are unchanged when data after the origin changes; a whole-series encoding would differ | T12 | Pass |
| With two targets, a past-only category that first occurs after the origin (sorting first, or in the middle) changes no input; the dataset-wide codes do change | T12, several targets | Pass |
| Static CQR meets nominal coverage across all 10 symmetric pairs and single default `0.80` on under- and over-covering exchangeable data; online ACI tracks nominal coverage under regime shift without `alpha_t` wind-up, evicts stale forecast windows, and keeps quantiles non-decreasing with median unshifted | T13, `test_conformal.py` | Pass |
| Power-of-two shape bucketing (`pad_batch_to_series`) preserves loss and gradients to `< 1e-12` in float64 and trains/evaluates non-divisible datasets under `compile_mode="reduce-overhead"` | `test_trainer.py` | Pass |
| External isolated install of `mistralai/timestral@a534bdb` (without `timestral-tune`) loads an exported 5M LoRA checkpoint | One-off isolated check (2026-11-22) | Bit-identical to library (`max_abs = 0.0`), `4.9%` relative diff from zero-shot |
| A category first seen after a `split_by_time` cutoff changes no training batch, with one or two targets | `test_dataset.py` | Pass (tiny model) |
| Series that start and end at different times: no training window holds a step at or after the cutoff | `test_dataset.py` | Pass |
| `head_only`, `lora_variate`, `lora_temporal`, `full_ft` on a fixed batch: loss falls, only targeted weights change, deterministic for a seed | `test_trainer.py` | Pass on CPU, Apple GPU and CUDA (including `bf16` on CUDA and `torch.compile`) |

## 9. Target-frame check

Probe: `uv run python scripts/probes/target_frame_check.py --size 5M` (and `--size 28M`). The pretrained
model's normalised quantiles are scored against targets normalised in each candidate frame, by horizon
patch. Data: 1,024 synthetic series per cell (random walks, trend plus seasonality, level shifts, changing
volatility), contexts of 2 and 8 patches, horizon of 16 patches, seed 0.

Coverage of the 50% / 80% / 90% central intervals:

| Model, context | Patches | `median_folded` | `true_values` |
|---|---|---|---|
| 5M, 2 patches | 0 (control) | 0.51 / 0.81 / 0.91 | identical |
| | 1–3 | 0.51 / 0.81 / 0.92 | 0.74 / 0.96 / 0.99 |
| | 4–15 | 0.40 / 0.79 / 0.94 | 0.95 / 1.00 / 1.00 |
| 5M, 8 patches | 0 (control) | 0.49 / 0.80 / 0.90 | identical |
| | 1–3 | 0.51 / 0.81 / 0.90 | 0.58 / 0.88 / 0.95 |
| | 4–15 | 0.43 / 0.77 / 0.90 | 0.76 / 0.97 / 0.99 |
| 28M, 2 patches | 0 (control) | 0.52 / 0.83 / 0.92 | identical |
| | 1–3 | 0.55 / 0.85 / 0.94 | 0.78 / 0.97 / 0.99 |
| | 4–15 | 0.44 / 0.78 / 0.92 | 0.95 / 1.00 / 1.00 |
| 28M, 8 patches | 0 (control) | 0.51 / 0.81 / 0.91 | identical |
| | 1–3 | 0.52 / 0.81 / 0.91 | 0.59 / 0.88 / 0.95 |
| | 4–15 | 0.43 / 0.76 / 0.89 | 0.77 / 0.97 / 0.99 |

Pinball loss and frame scales (5M; 28M is within a few percent):

| Context | Patches | Loss, `median_folded` | Loss, `true_values` | Signed mean `log(scale_A / scale_B)` |
|---|---|---|---|---|
| 2 patches | 1–3 | 0.429 | 0.316 | −0.37 |
| 2 patches | 4–15 | 0.723 | 0.410 | −1.18 |
| 8 patches | 1–3 | 0.236 | 0.204 | −0.08 |
| 8 patches | 4–15 | 0.410 | 0.270 | −0.44 |

By generator, the 80% interval's coverage on patches 1–15 ranges over 0.68–0.96 under `median_folded`
(random walks and changing volatility below nominal, level shifts above) and over 0.83–1.00 under
`true_values`, which over-covers for every generator.

Reading. The model is calibrated on this data (the control sits at nominal). Under `median_folded` it stays
near nominal at 80% and 90%; under `true_values` it over-covers heavily. A model trained against
true-values targets would be calibrated in that frame. So the checkpoints were trained in the
median-folded frame or something equivalent, which confirms decision D2.

What the rule fixed before the run said. It also compared pinball loss, and by that rule the outcome is
"inconclusive": `true_values` has the lower loss. That criterion was ill-posed. Each frame's loss is in its
own normalised units, and true-values frames have larger scales here (the signed log ratio above), so they
score lower whatever the model predicts. The probe's docstring keeps the original rule with a dated note.

Limits. Synthetic, univariate, fully observed series; one seed; two model sizes. Under `median_folded` the
50% interval under-covers at long lead times (0.40–0.44 on patches 4–15).

## 10. PyTorch versions

Test suite: `uv run --isolated --python 3.12 --with "torch==<version>" pytest`.

| PyTorch | Python | Result |
|---|---|---|
| 2.4.0 | 3.12 | 313 passed, 25 skipped on `main` (`-m "not network"`, including all Phase 3 & Phase 4 replay/distillation tests); earlier full run at `3fe5bc9`: 411 passed, 26 skipped. Skipped: accelerator tests (MPS requires 2.8+), DataLoader-worker test (below), optional `lacuna`, and 14 non-network CUDA-only tests |
| 2.8.0 | 3.12 | 356 passed, 15 skipped on `main` (`-m "not network"`, including Apple GPU and all Phase 3 & Phase 4 tests); earlier full run at `3fe5bc9`: 510 passed, 4 skipped |
| 2.14.1 | 3.10 | 356 passed, 15 skipped on `main` (`-m "not network"`, `uv run --isolated --python 3.10 pytest -m "not network"`); earlier full run at `3fe5bc9`: 510 passed, 4 skipped |
| 2.15.0 | 3.13 | 567 passed, 17 skipped (full suite on `main`, including tiny, 5M, 28M, 123M MPS parity, and all Phase 3 & Phase 4 replay/distillation tests; 16 CUDA-only + 1 optional `lacuna` skipped) |

These are the counts on `main` (2026-11-23), all on macOS. Zero acceptance test placeholders remain.
The CUDA run (§13) used Linux x86 (`89ac249`, merged into `main`) and passed all 16 CUDA tests as well (580 passed, 4 skipped).

**Declared dependency floors and the built package (2026-11-23, version `0.1.0`).**

| Check | Command | Result |
|---|---|---|
| Lowest declared versions together (Python 3.10: torch 2.4.0, numpy 1.24.0, safetensors 0.4.0, huggingface_hub 0.23.0) | `uv run --isolated --python 3.10 --resolution lowest-direct --with "setuptools==75.1.0" pytest -m "not network"` | 313 passed, 25 skipped |
| The same without `setuptools` | `uv run --isolated --python 3.10 --resolution lowest-direct pytest -m "not network"` | 310 passed, 25 skipped, 3 failed: the three `torch.compile` tests raise `ModuleNotFoundError: No module named 'setuptools'` inside PyTorch's compiler. The Python 3.10 wheel of torch 2.4.0 declares no `setuptools` requirement; the Python 3.12 wheel does. With torch 2.8.0 on Python 3.10 the same tests pass |
| Package metadata | `uv build && uvx twine check --strict dist/*` | Both files pass; the wheel holds only `timestral_tune/` and the three licence files |
| Clean install | Wheel installed into a fresh Python 3.12 environment outside the repository, then the README quick start run as written on synthetic hourly data | `Timestral-5M` loads from the Hub, `lora_variate` trains 12 steps on the Apple GPU, the export loads in the vendored upstream loader, `run_checks` reports six passes, and the `ReplayConfig()` example trains and exports |
| Published package | `uv pip install "timestral-tune==0.1.0"` from PyPI into a fresh Python 3.12 environment, then the README quick start as written | Version `0.1.0` installs with its four dependencies; the quick start gives the same losses and six `run_checks` passes as the local wheel, and the export loads upstream |

**The Apple GPU needs PyTorch 2.8.** `torch.Tensor.cummax` on that backend, which upstream's input assembly
calls. Probe (it imports nothing from this library):
`uv run --isolated --python 3.12 --with "torch==2.7.1" python scripts/probes/mps_cummax.py`.

| PyTorch | 2.4.0 | 2.5.1 | 2.6.0 | 2.7.1 | 2.8.0 | 2.9.1 |
|---|---|---|---|---|---|---|
| `cummax` on MPS | raises | raises | raises | raises | works | works |

2.7.1, 2.8.0 and 2.15.0 were re-run with the probe; 2.4.0, 2.5.1, 2.6.0 and 2.9.1 were measured with an
equivalent one-line command before the probe existed.

`resolve_device` skips the Apple GPU under `"auto"` on older versions and raises for an explicit `"mps"`.

**DataLoader workers hang the exit on macOS with PyTorch 2.4.0.** Probe (it imports nothing from this
library): `uv run --isolated --python 3.12 --with "torch==2.4.0" python scripts/probes/worker_exit_hang.py`.

| PyTorch | 2.4.0 | 2.8.0 | 2.15.0 |
|---|---|---|---|
| Process exits after reading batches through one worker | no (killed after 300 s) | yes | yes |

While it hung, PyTorch's `torch_shm_manager` helper held the write end of the pipe that Python's
multiprocessing resource tracker reads, so the interpreter waited for the tracker forever at shutdown.
Versions 2.5 to 2.7, Python versions other than 3.12, and Linux were not tried. The suite skips
`test_dataset_pickles_and_works_in_a_worker_process` on macOS below PyTorch 2.8 for this reason; with it
included, all tests passed on 2.4.0 and the process then never exited.

## 11. One-off results not covered by a probe or test

Measured once in throwaway scripts on 2026-11-21 and not reproduced by anything in the repository.
Re-measure before relying on them.

| Result | Value |
|---|---|
| Forward agreement with the float64-weights reference (5M) | CPU fp32 3.1e-7, MPS fp32 4.3e-7 |
| Autocast speed on MPS, 28M covariate batch | fp32 1,778 ms; `bfloat16` 1,731 ms; `float16` 1,732 ms (no speed-up) |
| Training step at context 4,096, univariate, batch 16 | 5M: CPU 138 ms, MPS 93 ms. 28M: CPU 526 ms, MPS 347 ms |
| RMSNorm via `sqrt` instead of `rsqrt` (5M) | CPU fp32 gradient error moved from 3.9e-3 to 1.5e-3 |
| The 5M past-only-covariates case with explicit attention, against the earlier (float32-RMSNorm) reference | CPU 2.6e-3, Apple GPU 5.9e-3; the error sat in layers 0 to 4 and the patch embedding, with the same distribution on both devices |
| Why `attention_context("math")` regresses on T14's 5M `past_only_covariates` batch on CPU | 100% of layer 5 decoder `dq` squared error sits at one query `(row=3, head=2, query=11)` whose top-2 keys sit at ~351,272 with true float64 gap 3.4577 (float32 ULP 0.03125): `default` quantized the gap to 3.4375 (`p2` +2.0%, grad rel L2 2.24e-3), `explicit` to 3.46875 (`p2` −1.1%, 1.16e-3), and `math` to 3.375 (`p2` +8.3%, 9.02e-3) |
| 25-seed CPU sweep on T14-shaped 5M `past_only_covariates` batches | `default` median 1.89e-3 (max 3.67e-3); `math` median 2.12e-5 (max 5.38e-3); `explicit` median 2.31e-5 (max 1.40e-2 on seed 8, where `math` gave 9.57e-4) |

## 12. Facts read from sources, not measured

These `[H]` and `[U]` facts were read, not run: the four checkpoints' `config.json` values (spec §3.1), and
that the released `quantile_width_factor` is 3.0 while the code default is 2.0.

## 13. CUDA

Run: `bash scripts/cuda_check.sh` on an NVIDIA L4 VM (`g2-standard-4`, Ubuntu 22.04 x86_64, 4 CPU threads,
PyTorch 2.15.0+cu132, Python 3.12) on `phase-4` at commit `89ac249` (2026-11-22; `step_timing.py`
`--bf16`/`--compile` probe rows from the same hardware at `b0980cb`). All gradient numbers here use the
float64-RMSNorm reference of §6.

| Check | Result |
|---|---|
| T14 and all other device-parametrised tests | 255 passed, 3 skipped (the 3 Apple GPU tests) |
| Whole suite (including T1–T14, all 4 `bf16` CUDA modes + dual-orientation `head_only` cache, shape-bucketed `reduce-overhead` compile, `load_lora` resume, Phase 3 metrics/sign-flip/Lacuna, and all 12 Phase 4 CUDA replay tests incl. cached+live `reduce-overhead` and released `Timestral-5M` `bf16`) | 580 passed, 4 skipped (3 Apple GPU tests + 1 optional `lacuna` test) |
| `head_only` gradients on CUDA against the reference | At most 1.7e-6 (bound 1e-5) |
| Verifiers (`run_checks`) on CUDA | Pass (part of the suite) |
| `step_timing.py --device cuda --bf16 --compile default` (`full_ft`, batch 32, ctx 1,024; `b0980cb`) | 5M uni 21 ms / 0.18 GiB, cov5 25 ms / 0.58 GiB; 28M uni 32 ms / 0.73 GiB, cov5 62 ms / 1.90 GiB |
| `step_timing.py --device cuda --mode lora_variate --bf16 --compile reduce-overhead` (`b0980cb`) | 5M cov5 18 ms; 28M cov5 43 ms |
| Real Hub export of `mistralai/Timestral-5M` (`lora_variate`, `bf16`, run at `3fe5bc9`) | Copies verbatim 1,687-byte `Modified MIT License` from Hub; reloads via upstream `load_model(..., strict=True)` on CUDA (`5,208,960` params) |

The first attempt at this run (at `163ae4a`, one commit earlier) failed
`test_shape_bucketing_preserves_exact_loss_gradients_and_works_under_reduce_overhead` because
`head_bucket_trainer` hardcoded `device="cpu"` with `compile_model=True` on a minimal Linux VM without
`g++`; switching it to `device` (`"cuda"` when available) in `6880061` resolved the failure and exercised
the compiled head-cache bucketing path on CUDA. (Historical Phase 1 note: the first attempt at the Phase 1
CUDA run before `50c170a` failed one x86 CPU case — explicit attention on 5M with past-only covariates at
5.3e-3 against a 5e-3 bound, the near-tie case of §6 which now has its own bound.)

Forward, relative L2 difference from CPU (`device_parity.py --device cuda`):

| Case | 5M | 28M |
|---|---|---|
| `predict_block`, no covariates | 2.2e-7 | 3.0e-7 |
| `predict_block`, with covariates | 1.4e-6 | 2.2e-6 |
| `forecast`, horizon 96 | 1.4e-7 | 1.4e-7 |
| `forecast`, horizon 2,100 | 1.5e-5 | 1.8e-6 |
| `forecast`, numeric and categorical covariates | 1.2e-7 | 1.0e-7 |
| Autocast `bfloat16` (CUDA `bf16` training mode; fp32 only on CPU/MPS) | 1.5e-2 | 5.7e-2 |
| Autocast `float16` | 2.0e-3 | 5.9e-3 |

Float32 training-loss gradients against the reference, by attention kernel (`gradient_accuracy.py --device
cuda`, experiments 1 and 2; the CPU columns are that VM's x86 CPU):

| Model | Layout | CPU default / unfused / explicit | CUDA default / unfused / explicit |
|---|---|---|---|
| Tiny random | all four | 2.3e-7 to 3.6e-7 (default) | 1.6e-7 to 3.1e-7 (default) |
| 5M | univariate | 2.0e-3 / 2.1e-5 / 1.9e-5 | 1.2e-2 / 1.3e-5 / 6.0e-6 |
| 5M | multivariate | 1.1e-3 / 1.6e-4 / 5.6e-5 | 1.2e-2 / 1.5e-5 / 1.4e-5 |
| 5M | covariates | 9.1e-4 / 1.0e-4 / 1.3e-4 | 3.2e-3 / 2.3e-5 / 1.9e-5 |
| 5M | past-only covariates | 7.9e-4 / 8.3e-6 / 1.1e-5 | 2.7e-3 / 3.4e-6 / 5.5e-6 |
| 28M | univariate | 5.9e-4 / 5.8e-5 / 5.8e-5 | 1.4e-3 / 2.6e-5 / 3.5e-6 |
| 28M | multivariate | 9.6e-4 / 7.5e-6 / 7.9e-6 | 2.3e-3 / 5.5e-6 / 8.9e-6 |
| 28M | covariates | 1.1e-3 / 4.0e-4 / 1.5e-4 | 1.6e-3 / 2.0e-5 / 1.8e-5 |
| 28M | past-only covariates | 1.3e-3 / 1.2e-5 / 2.0e-5 | 3.2e-3 / 6.9e-6 / 1.2e-5 |

On CUDA both unfused kernels lower the error on every one of these eight layouts, by 50 to 900 times. The
15-seed sweep on that VM's CPU (experiment 6): default median 1.24e-3 (max 4.94e-3), unfused median 1.41e-5
(max 1.32e-3, on a seed with a near-tie), explicit median 1.16e-5 (max 8.3e-5). T14's per-tier maxima on
this VM are in the table of §6.

Cost of each kernel on CUDA: a full training step, all parameters, context 1,024, horizon 96, batch of 32
series. Measured once in a throwaway script on the VM (median of six steps after two warm-up steps); not
reproduced by a probe:

| Model | Batch | Default | PyTorch's unfused kernel | Explicit matmul-and-softmax |
|---|---|---|---|---|
| 5M | univariate | 47 ms, 0.3 GiB | 59 ms, 0.3 GiB | 58 ms, 0.3 GiB |
| 5M | 5 rows per series, with covariates | 95 ms, 1.9 GiB | 102 ms, 1.9 GiB | 100 ms, 1.9 GiB |
| 28M | univariate | 72 ms, 0.9 GiB | 89 ms, 0.9 GiB | 90 ms, 0.9 GiB |
| 28M | 5 rows per series, with covariates | 393 ms, 5.7 GiB | 411 ms, 5.9 GiB | 408 ms, 5.9 GiB |

So at this context length the unfused kernels cost 5% to 25% more time and at most 0.2 GiB more memory. The
attention matrix grows with the square of the context, so longer contexts will cost more; that was not
measured.

These rows were measured once in a throwaway script during the first CUDA run (univariate batch, earlier
reference) and are not reproduced by a probe:

| Experiment | 5M | 28M |
|---|---|---|
| CUDA with float64 weights | 1.2e-5 | 1.2e-5 |
| CUDA float32 with TF32 disabled; and with `TRITON_DEFAULT_FP_FUSION=0` | 1.2e-2 (unchanged) | 1.4e-3 (unchanged) |

Training-step timing against the VM's CPU (`step_timing.py --device cuda` on the same L4 VM, comparing
baseline `main` with `perf/phase1-kernels` at `9fc6735`; the CPU has only 4 threads, so CPU/CUDA ratios
flatter the GPU compared with a workstation CPU):

| Model | Batch | CPU (`main` → `perf`) | CUDA (`main` → `perf`) | CUDA speed-up vs `main` | CUDA memory (`main` → `perf`) |
|---|---|---|---|---|---|
| 5M | univariate | 298 → 296 ms | 47 → 47 ms | 1.00× | 0.3 → 0.3 GiB |
| 5M | 5 rows per series, with covariates | 2,591 → 1,426 ms | 94 → 56 ms | **1.68×** | 1.9 → 1.1 GiB (−42%) |
| 28M | univariate | 1,132 → 1,130 ms | 74 → 73 ms | 1.01× | 1.9 → 1.1 GiB |
| 28M | 5 rows per series, with covariates | 10,837 → 5,993 ms | 390 → 208 ms | **1.88×** | 5.7 → 3.3 GiB (−42%) |

On the same VM, a warmed-up 5-epoch `head_only` run with per-epoch validation on 28M (24 covariate series,
batch 16, including epoch 1 cache build) dropped from 850 ms uncached to 273 ms cached (3.1× faster).

Phase 2 CUDA step timing and peak memory (`bf16`, `torch.compile`, fused `AdamW`) on the L4 VM (`ctx=1,024`,
`H=96`, batch 32, 5 rows/series with covariates; median of 6 steps after 3 warm-up steps; one-off benchmark):

| Model | Mode | `fp32` | `bf16` | `bf16 + compile(default)` | `bf16 + compile(reduce-overhead)` |
|---|---|---|---|---|---|
| 5M | `lora_variate` (`r=8`, Q+V) | 37.9 ms, 0.47 GiB | 39.0 ms, 0.32 GiB | 17.8 ms, 0.26 GiB (2.1×) | 18.0 ms |
| 5M | `full_ft` | 54.1 ms, 1.08 GiB | 53.8 ms, 0.73 GiB | 23.5 ms, 0.56 GiB (2.3×) | 23.0 ms |
| 28M | `lora_variate` (`r=8`, Q+V) | 139.8 ms, 1.63 GiB | 69.7 ms, 1.12 GiB (2.0×) | 42.9 ms, 0.94 GiB (3.3×) | 43.6 ms |
| 28M | `full_ft` | 197.0 ms, 3.28 GiB | 93.8 ms, 2.30 GiB (2.1×) | 59.8 ms, 1.80 GiB (3.3×) | 59.9 ms |

(For `reduce-overhead`, CUDA graphs allocate from a private pool not reflected in `max_memory_allocated()`,
so memory is reported for `compile(default)`.)

Decision D15 resolution — 20-step AdamW training trajectories on L4 CUDA (16 covariate series, batch 16;
comparing `fp32` default SDPA, `fp32` `attention_context("math")`, and `bf16` autocast; one-off script):

| Model | Mode | `fp32` default val loss | `fp32` math val loss (forecast `max_rel` vs default) | `bf16` val loss (forecast `max_rel` vs default) |
|---|---|---|---|---|
| 5M | `lora_variate` (`lr=3e-4`) | 0.11511 | 0.11509 (4.3e-4) | 0.11519 (1.2e-3) |
| 5M | `lora_temporal` (`lr=3e-4`) | 0.10853 | 0.10873 (7.8e-3) | 0.10876 (5.9e-3) |
| 5M | `full_ft` (`lr=1e-5`) | 0.11559 | 0.11566 (8.7e-4) | 0.11548 (1.4e-3) |
| 28M | `lora_variate` (`lr=3e-4`) | 0.09029 | 0.09029 (6.0e-6) | 0.09039 (3.0e-3) |
| 28M | `lora_temporal` (`lr=3e-4`) | 0.08629 | 0.08629 (1.2e-3) | 0.08618 (1.2e-2) |
| 28M | `full_ft` (`lr=1e-5`) | 0.09022 | 0.09022 (7.2e-5) | 0.09034 (3.9e-3) |

Large checkpoints — `Timestral-123M` (122.6M params, `d_model=640`, 20 layers) and `Timestral-209M`
(209.3M params, `d_model=768`, 24 layers) probed on the same NVIDIA L4 architecture (one-off probe,
2026-11-22, `B=2`, `ctx=128`, `H=64`, float64-RMSNorm reference on CUDA):

| Model | Layout | Peak attention logit (layer) | Peak `||Wq||_2` / `||Wk||_2` | CUDA fp32 default grad rel L2 (`1 - cos`) | CUDA `math` SDPA | CUDA explicit | CUDA `head_only` | Fwd `fp32` / `bf16` vs CPU |
|---|---|---|---:|---|---:|---:|---:|---|
| 123M | univariate | 131.6k (`layers.17.attns.0`, gap 1,928) | 81.7 / 61.9 | 4.96e-4 (`1.2e-7`) | 9.39e-6 | 1.34e-5 | 4.86e-7 | 7.4e-7 / 1.0e-2 |
| 123M | covariates | 132.4k (`layers.17.attns.0`, gap 191) | 81.7 / 61.9 | 6.94e-4 (`2.4e-7`) | 7.89e-6 | 1.07e-5 | 1.91e-6 | 3.0e-6 / 4.1e-2 |
| 209M | univariate | 123.4k (`layers.1.attns.0`, gap 30,731) | 167.9 / 150.6 | 4.85e-4 (`1.2e-7`) | 4.03e-6 | 7.13e-6 | 4.24e-7 | 2.8e-7 / 5.7e-3 |
| 209M | covariates | 131.4k (`layers.1.attns.0`, gap 2,268) | 167.9 / 150.6 | 2.63e-4 (`3.5e-8`) | 2.30e-5 | 2.32e-5 | 5.49e-7 | 5.5e-6 / 5.6e-2 |

Both large checkpoints have 3×–4× lower peak attention logits than 5M (404k) and 28M (503k), and 10×–45×
smaller CUDA default-SDPA gradient error (`2.6e-4`–`6.9e-4`, cosine `>= 0.9999997`). Real 120-step
`retail_demand` `lora_temporal` training on `123M` and `209M` (train-only series selection,
`docs/benchmarks/precision_ablation_123m_209m.json`) gave `fp32` default / `fp32` math / `bf16` default
test wQL of `0.1982 / 0.1978 / 0.1981` on `123M` and `0.1980 / 0.1951 / 0.1947` on `209M` (`docs/BENCHMARKS.md` §3).

Environment requirement found on the way. With the VM's system Python 3.10 (PyTorch 2.14.1+cu130), every
CUDA test failed: PyTorch routed an operation in upstream's attention layer through a Triton kernel, and
Triton compiles a C helper on first use, which stopped at `fatal error: Python.h: No such file or directory`.
With a uv-managed Python 3.12, which ships its headers, everything passed. `scripts/cuda_check.sh` now
checks for the headers and a compiler first and stops with advice (verified on the VM: exit code 4).

## 14. Phase 3 defaults, sign-flip augmentation, and large-checkpoint sweeps

Backs the `[V]` claims in spec §5.3, §5.4, and D19. All sweeps ran on an NVIDIA L4 (`bf16`, PyTorch
2.15.0+cu132, Python 3.12, 2026-11-22), scored strictly on validation wQL and equal-weighted 3-domain
macro GIFT-Eval forgetting (`ett1`, `jena_weather`, `loop_seattle`; raw JSON in `docs/benchmarks/`):

| Claim (`SPEC.md`) | Command / artefact | Verified result |
|---|---|---|
| `TrainingConfig.sign_flip=True` default (§5.3, §5.4, D19) | `run_sweeps.py --device cuda --sizes 123M 209M --bf16 --batch-size 16 --eval-batch-size 32 --datasets realised_volatility retail_demand signed_returns --modes lora_temporal full_ft --mode-lrs lora_temporal=1e-3,3e-3 full_ft=1e-4,3e-4 --sign-flip-modes off on --seeds 0 1 2 --check-forgetting --skip-lora-sweep` (`docs/benchmarks/sweeps_large_decisions.json`) | On signed `logret_20`, flipping stayed within seed spread on both `123M` and `209M` (improved 5/6 cells; worst delta `+0.0008` vs `0.0025` spread). On retail, cut validation wQL by 5.9%–6.7%; improved volatility on 3/3 seeds on both sizes. |
| `lora_temporal=1e-3` default (§5.4, D19) | Same command (`docs/benchmarks/sweeps_large_decisions.json`) | `3e-3` never beat `1e-3` by more than seed spread on `123M`/`209M`, lost outright on 3/6 splits, and hit `+19.6%`–`+23.8%` worst-seed forgetting on signed returns (vs `+0.3%`–`+0.7%` at `1e-3`). |
| `full_ft=1e-4` default (§5.4, D19) | Same command (`docs/benchmarks/sweeps_large_decisions.json`) | On `123M`/`209M`, `1e-4` won 3/6 splits outright (including both `209M` volatility and retail), `3e-4` never beat `1e-4` by more than seed spread on the other 3, and `1e-4` halved worst-seed retail forgetting (`+1.6%`/`+2.1%` vs `+4.0%`/`+4.2%`). |
| `head_only=3e-3`, `lora_variate=3e-3`, LoRA `r=8, targets=("q","k","v","o")` (§5.4, D19) | `run_sweeps.py --device cuda --sizes 5M 28M --bf16 --max-steps 120 --batch-size 32 --sign-flip-modes on --check-forgetting` (`docs/benchmarks/sweeps_bracketed.json`) | Full `head_only` (`3e-5..3e-2`) and `lora_variate` (`3e-4..3e-2`) grids bracket `3e-3` cleanly on `28M`; `r=8, ("q","k","v","o")` beat `("q","v")` across 3/4 5M LoRA comparisons. |
| Dual-orientation `head_only` cache parity with `sign_flip=True` (§5.4) | `tests/test_trainer.py::test_sign_flip_augmentation_coherence_determinism_and_head_cache_bypass` | 3 frozen-body passes across 4 epochs (`+x` train, `-x` train, val) matching streaming `sign_flip=True` head weights within `< 1e-4` L2. |
| `123M`/`209M` `fp32` default vs `fp32` math vs `bf16` on train-only retail (D15) | `run_benchmarks.py --datasets retail_demand --modes lora_temporal --max-steps 120 --batch-size 16 --eval-batch-size 16` (`docs/benchmarks/precision_ablation_123m_209m.json`) | `123M`: `0.1982 / 0.1978 / 0.1981` test wQL; `209M`: `0.1980 / 0.1951 / 0.1947` (`bf16` 1.8× faster, `2.79 GiB` VRAM). |

## 15. Phase 4: Replay/distillation and Apple GPU 123M, 209M, and long-context coverage

Backs the `[V]` claims in spec §5.3, §5.4 (Apple GPU scope), §10, and D21. Measured on an Apple M4
(16 GiB unified memory, macOS, PyTorch 2.15.0, Python 3.13, `fp32`, 2026-11-22):

### 15.1 Replay mixing & frozen-base distillation (`docs/benchmarks/replay_ablation.json`)

Command: `uv run python scripts/benchmarks/run_replay_ablation.py --device mps --output docs/benchmarks/replay_ablation.json`.

| Claim (`SPEC.md` §5.3, D21) | Artefact / Test | Verified result |
|---|---|---|
| `head_only` vs lower-LR `1e-3` control (`5M` 3 seeds; `28M` 1 seed) | `docs/benchmarks/replay_ablation.json` | Default `distill_0.1` strictly beats plain `1e-3` on retail on both `5M` (`0.2105 / +2.5%` vs `0.2114 / +3.6%`) and `28M` (`0.2079 / +3.6%` vs `0.2090 / +4.2%`). On `5M` vol, `replay_0.25` matches `1e-3` wQL (`0.2231` vs `0.2236`) at lower worst-seed forgetting (`+5.0%` vs `+7.8%`), while `distill_0.1` cuts forgetting to `+1.4%` (`−0.7%` on `28M`). On `28M` vol (1 seed), plain `1e-3` gives lower wQL (`0.2169` vs `0.2195–0.2196`). Most runs selected step `200`. |
| `lora_temporal=3e-3` forgetting cliff vs shipped `1e-3` (3 seeds) | `docs/benchmarks/replay_ablation.json` | Default `distill_0.1` eliminates the `+31.3%` cliff (`+0.6%` worst-seed, mean `−0.2%`) and edges out shipped `1e-3` on 3/3 seeds (`0.2178` vs `0.2186`, small margin vs seed spread). `replay_0.25` (`0.2181`, worst `+1.1%`) matches shipped `1e-3`. |
| `full_ft=3e-4` vs `1e-4` & `2e-4` controls and 256-series pool (3 seeds) | `docs/benchmarks/replay_ablation.json` | Plain `2e-4` (`0.1995`, worst `+1.3%`) matches or slightly beats `3e-4 + distill_0.1` (`0.1999`, worst `+1.3%`, mean `+0.9%` vs `+1.2%`). Expanding the synthetic pool from 64 to 256 series lowers synthetic-pinball worst-seed forgetting from `+4.1%` to `+2.8%` (still no better than plain `3e-4`'s `+2.8%`). |
| Cached vs live teacher parity, CUDA `bf16` `smooth_beta` probe (`tiny` & released `5M`), cached+live `reduce-overhead`, and released `5M` CUDA `bf16` replay | `tests/test_replay.py` (42 passed on Linux x86 + L4 CUDA at `89ac249`; 30 passed + 12 CUDA skipped on macOS) & `scripts/probes/replay_bf16_parity.py --device cuda --sizes tiny 5M` (`docs/benchmarks/replay_bf16_parity.json`) | Direct CPU teacher parity `< 1e-5` on `+x`/`-x`; CPU `fp32` `full_ft` 2-step divergence drops from `1.01e-2` (`beta=0.0`) to `4.65e-6` (`beta=1e-3`); on NVIDIA L4 CUDA `bf16`, cached vs live teacher diff is `0.00e+00` across `smooth_beta` `{0.0, 1e-3, 1e-2}` for `lora_temporal` and `full_ft` on both `tiny` and released `Timestral-5M`. Both cached and live-LoRA `reduce-overhead` CUDA graph paths and released `Timestral-5M` CUDA `bf16` replay+export pass. |

### 15.2 Apple GPU (MPS) `123M`, `209M`, and long-context coverage (`docs/benchmarks/mps_coverage.json`)

Command: `uv run python scripts/probes/mps_large_and_long_context.py --output docs/benchmarks/mps_coverage.json`.
Measured on PyTorch **2.15.0** only (16 GiB Apple M4, `fp32`; freshly regenerated with `assert torch.isfinite(loss)` active).

**Large checkpoints (`123M` & `209M`, `fp32` on MPS):**

| Model | `predict_block` uni / cov rel L2 vs CPU | `forecast(96)` cov+cat rel L2 | Grad rel L2 vs f64 (`cpu` / `mps`, `1 - cos`) | `ctx=256 cov5 B=8` median ms / driver GiB (`head` / `lora_v` / `lora_t` / `full_ft`) |
|---|---|---|---|---|
| **123M** | `1.39e-7` / `1.09e-6` | `1.09e-7` | `1.11e-4` / **`7.95e-6`** (`2.95e-11`) | `85 ms / 0.59` · `192 ms / 1.26` · `224 ms / 1.28` · `362 ms / 2.57` |
| **209M** | `1.18e-7` / `3.93e-7` | `9.51e-8` | `2.00e-4` / **`4.54e-6`** (`9.17e-12`) | `130 ms / 1.05` · `318 ms / 2.06` · `375 ms / 2.04` · `664 ms / 4.07` |

**Long contexts (`4,096` to `32,768` steps = `128..1,024` patches, `B=1`, `fp32` on 16 GiB M4 MPS, PyTorch 2.15.0):**
Training entries run **one optimiser step** and assert `torch.isfinite(loss)` (no gradient parity check at `4,096..32,768` steps).

| Model | Context (`steps / layout`) | Inference rel L2 vs CPU (`ms`) | Training mode (`ms` / alloc GiB / driver GiB) |
|---|---|---|---|
| **28M** | `4,096` / `8,192` / `16,384` / `32,768` uni | `3.8e-7` (`30 ms`) .. `4.8e-7` (`76 ms`) | `lora_t`: `123..273 ms` (`0.23 / 4.06`); `full_ft` 32k: `294 ms` (`0.56 / 4.11`) |
| **28M** | `8,192` / `16,384` `cov3` | `1.1e-5` (`122 ms`) / `3.5e-7` (`247 ms`) | `lora_t`: `186 ms` / `335 ms` (`0.22 / 4.06`) |
| **123M** | `4,096` / `8,192` / `16,384` / `32,768` uni | `2.9e-7` (`50 ms`) .. `5.1e-7` (`257 ms`) | `lora_t`: `128..759 ms` (`0.95 / 4.03`); `full_ft` 32k: `936 ms` (`2.35 / 4.05`) |
| **123M** | `8,192` / `16,384` `cov3` | `6.3e-6` (`383 ms`) / `2.2e-7` (`759 ms`) | `lora_t`: `495 ms` (`0.95 / 2.78`) / `1,030 ms` (`0.95 / 4.56`) |
| **209M** | `4,096` / `8,192` / `16,384` / `32,768` uni | `3.7e-7` (`73 ms`) .. `5.3e-7` (`375 ms`) | `lora_t`: `165..1,170 ms` (`1.59 / 5.07`); `full_ft` 32k: `1,469 ms` (`4.06 / 6.09`) |
| **209M** | `8,192` / `16,384` `cov3` | `1.7e-6` (`605 ms`) / `4.7e-7` (`1,376 ms`) | `lora_t`: `825 ms` (`1.59 / 4.13`) / `1,692 ms` (`1.60 / 6.10`) |

Also covered in pytest by `tests/test_loader_device.py::test_mps_long_context_and_large_checkpoint_parity` (`@pytest.mark.network`).
