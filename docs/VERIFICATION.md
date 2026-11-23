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

CUDA results (§13, §17.4) come from Linux x86_64 cloud VMs with one NVIDIA L4 (23 GB), PyTorch
2.15.0+cu132: Phase 1–4 runs used Ubuntu 22.04, 4 CPU threads, and Python 3.12.16 (`89ac249`); the first
2026-11-23 Phase 5 run (`71514a6`, §11, §13, §17.2) used driver `580.178.04`, CUDA runtime `13.2`, NCCL
`2.30.7`, and `uv`-selected Python `3.15.2` (outside the CI matrix of Python `3.10–3.13`); and the
2026-11-23 tracked Stage 3 session (`82a21ca`, §13, §17.4) used an 8-vCPU Linux x86_64 cloud VM with one
NVIDIA L4 (`23034 MiB`, driver `595.112.07`), CUDA runtime `13.2`, NCCL `2.30.7`, and Python `3.12.16`.
Linux CPU results in §11, §16, and §17 come from 8-vCPU Linux x86_64 cloud VMs (Ubuntu 22.04, including an
AMD EPYC 7B12 VM for post-fix runs) with `uv`-managed Python `3.12.16` (`torch 2.15.0+cu132`) and
`3.10.22` (`torch 2.14.1+cu130`).

The 123M and 209M checkpoints were measured later and separately: on CUDA in §13 and §14, and on the
Apple GPU in §15. Sections that do not name them used 5M and 28M only.

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
| A local checkpoint of a shape that is neither the tiny model nor a released size (width 48, 8 blocks, 3 heads, patch 8, released layer pattern) loads by path in `TimestralTrainer`, trains in all four modes, and exports to a directory the vendored upstream loader accepts with `strict=True` | `test_trainer.py` | Pass (CPU, random weights; no checkpoint above 209M has been run) |
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

Measured once in throwaway scripts on 2026-11-21 (unless dated otherwise) and not reproduced by anything in the repository.
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
| `ReduceOp.AVG` vs `ReduceOp.SUM` on CPU `gloo` (2026-11-23, macOS arm64; 1-rank `gloo` group on `tcp://127.0.0.1:<free port>`, `all_reduce` on a 1-element `float64` tensor, run under `uv run python -c ...` and `uv run --isolated --python 3.12 --with "torch==2.4.0" python -c ...`) | PyTorch 2.4.0: `ReduceOp.SUM` succeeds (`[2.0]`); `ReduceOp.AVG` raises `RuntimeError: Cannot use ReduceOp.AVG with Gloo`. PyTorch 2.15.0: both `ReduceOp.SUM` and `ReduceOp.AVG` succeed (`[2.0]`) |
| One-process bit-identity vs pre-implementation commit `57a781c` (2026-11-23; out-of-repo 12-scenario script covering all 4 modes, replay/distillation, cached/streamed `head_only`, sign flips, early stopping, and restore, run twice at `57a781c` for self-reproduction and once at branch tip) | Linux x86_64 CPU (8-vCPU VM at `3347f70`, and L4 VM CPU at `71514a6`): `57a781c` self-reproduction `BITWISE MATCH`, tip vs `57a781c` `BITWISE MATCH`. NVIDIA L4 CUDA (`71514a6`, driver `580.178.04`, Python `3.15.2`, PyTorch `2.15.0+cu132`): `57a781c` self-reproduction `BITWISE MATCH`, tip vs `57a781c` `BITWISE MATCH` |
| One-process float32 intra-op thread sensitivity on T15(j) scenarios and pre-fix thread-count reproduction (2026-11-23, commit `3347f70`, 8-vCPU Linux x86_64 VM, out-of-repo script running one-process `fit()` only at 1, 2, and 4 threads with no multi-process code; plus acceptance file alone at default 4 threads vs `OMP_NUM_THREADS=1`) | One-process `param_update` rel L2 across threads `1/2/4` was identical on `torch 2.14.1+cu130` (Python `3.10.12`) and `torch 2.15.0+cu132` (Python `3.12.16`): worst `N=2 lora_variate distill_only` `1.14e-04` (`1v2`), `4.23e-05` (`1v4`), `7.65e-05` (`2v4`); next worst `N=4 lora_variate both` `4.71e-05` (`1v4`) and `N=4 lora_variate replay_only` `4.33e-05` (`2v4`); `head_only distill_only` `4.48e-08`–`2.73e-06`; `full_ft replay_only` `2.92e-06`–`6.93e-06`. Running `tests/acceptance/test_distributed.py` alone (subset of 6 tests) passed `6/6` at default 4 threads and failed the exact 2 float32 `param_update` checks at `OMP_NUM_THREADS=1` (`lora_variate_distill_only_wrap_float32` `1.07e-04` and `default_synth` `1.71e-04` vs `1e-4`), identically on both PyTorch versions |
| Pre-fix 10×-tightened float32 margin scans (2026-11-23, commit `3347f70`, out-of-repo edit dividing every float32 tier by 10; lower bounds on what was thin because each test stops at its first failing assert) | macOS arm64: `9 failed, 24 passed, 4 deselected` (`param_update` up to `4.82e-05` in `T15(l)[4]`, `3.32e-05` in `T15(j)[4]`, `2.97e-05` in `T15(a)[4]`; `T15(b)[4]` step-0 `param` `1.67e-05`). Linux x86_64 (8-vCPU VM, threads `1/2/4`): `12 failed / 21 passed` (`t=1`), `9 failed / 24 passed` (`t=2`), `12 failed / 21 passed` (`t=4`), `4 deselected` — `param_update` up to `1.94e-04` (`t=2`, `T15(l)[4]` `default_synth` before switching to `lora_temporal`), `1.71e-04` (`t=1`), `7.78e-05` (`t=4`, `T15(a)[4]`); single-step `param` up to `3.38e-06` (`t=2`, replay whitebox `lora_variate:b_replay_zero_one_rank`) and `1.07e-05` (`t=1`, `T15(b)[4]` step 1 at old `lr=1e-2`); single-step `grad` `1.03e-06` (`t=4`, `T15(b)[4]` step 1 at old `lr=1e-2`) |
| Single-rank NCCL collective-dtype and primitive smoke (2026-11-23, commit `71514a6`, one NVIDIA L4, driver `580.178.04`, Python `3.15.2`, PyTorch `2.15.0+cu132`, CUDA runtime `13.2`, NCCL `2.30.7`, out-of-repo script; kept as history — multi-rank NCCL collectives on `TimestralTrainer` are now exercised by the tracked Stage 3 driver in §17.4) | `19/19 passed`: `all_reduce_sum` (`float64` n=8, `int64` n=4, `float32` n=1,048,576, `bfloat16` n=1,024), `broadcast` (`float32`, `float64`, `bfloat16`, `float16`, `int64`, `int32`, `bool`), `propagate_failure` (no error), `check_consistency`, `broadcast_status`, `broadcast_module_state` (`tiny` 47 f32, `5M` 75 f32, `5M` 75 bf16), and `GradientReducer` (`tiny` 47 tensors / `490,624` elements, reduce `1.50 ms`; `5M` 75 tensors / `5,208,960` elements, reduce `2.15 ms`) |
| Multi-rank NCCL trainer trial on one shared GPU (2026-11-23, commit `71514a6`, one NVIDIA L4, driver `580.178.04`, Python `3.15.2`, PyTorch `2.15.0+cu132`, NCCL `2.30.7`, out-of-repo script run with `NCCL_MULTI_RANK_GPU_ENABLE=1` because NCCL refuses multiple ranks on one GPU by default; 23 6-step scenarios each at `N = 2` and `N = 4` vs one-process CUDA on the same GPU; **kept as history — reproduced and superseded by the tracked Stage 3 driver session in §17.4**) | `N = 2` exit `[0, 0]`, `hung=[]`, wall `19 s`; `N = 4` exit `[0, 0, 0, 0]`, `hung=[]`, wall `35 s`; **`xrank_equal=True`** (bitwise cross-rank equality) across all 46 runs. **`float64`** (10 scenarios × 2: all 4 modes `none`, plus `head_only`/`lora_variate`/`full_ft` `replay_only` and `both`): `final_rel` `7.28e-17`–`1.43e-15`, `update_rel` `6.10e-16`–`2.99e-14`, `loss_rel` `1.33e-16`–`6.51e-16`. **`float32`** (`bf16=False`, same 10 scenarios × 2): `final_rel` `6.81e-08`–`9.86e-06`, `update_rel` `4.56e-07`–`1.80e-04` (peaks on `lora_variate`: `1.37e-04` at `N=2 none`, `1.80e-04` at `N=4 both`), `loss_rel` `7.76e-08`–`1.88e-05`. **`bf16`** (3 scenarios × 2: `lora_temporal none`, `full_ft none`, `full_ft both`; no parity tier set): `N = 2` `final_rel` `1.70e-04`–`8.14e-04`, `update_rel` `1.05e-02`–`1.89e-02`, `loss_rel` `1.96e-03`–`2.28e-03`; `N = 4` `final_rel` `1.77e-04`–`1.17e-03`, `update_rel` `1.03e-02`–`2.50e-02`, `loss_rel` `6.59e-04`–`2.85e-03` |

## 12. Facts read from sources, not measured

These `[H]`, `[U]`, and dependency-source facts were read, not run:

- The four checkpoints' `config.json` values (spec §3.1), and that the released `quantile_width_factor` is 3.0 while the code default is 2.0.
- `huggingface_hub` per-file download locking (spec §5.6, D29; source reading only, not a concurrent-download test): in both installed `huggingface_hub==2.4.0` (`_snapshot_download.py:521`; `file_download.py:1258, 1273, 1485, 1503`) and the declared floor `huggingface_hub==0.23.0` (inspected 2026-11-23 via `uv run --isolated --no-project --python 3.12 --with "huggingface_hub==0.23.0"`: `_snapshot_download.py:267-268` delegates to `hf_hub_download`; `file_download.py:797, 1366, 1478, 1485` acquires `WeakFileLock`), `snapshot_download` routes file downloads through `hf_hub_download` with `WeakFileLock`.

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

Phase 5 branch re-runs of `bash scripts/cuda_check.sh` on `2026-11-23` on one NVIDIA L4 (`23034 MiB`,
Linux x86_64, `torch 2.15.0+cu132`, CUDA runtime `13.2`, NCCL `2.30.7`): first at `71514a6` (`40a580a`
differs in comments only; driver `580.178.04`, `uv`-selected `CPython 3.15.2`, outside the CI matrix of
Python `3.10–3.13`), and then at `82a21ca` on an 8-vCPU cloud VM (driver `595.112.07`, `Python 3.12.16`,
closing the Python `3.12` gap on this branch):

| Check (one NVIDIA L4, `torch 2.15.0+cu132`) | `71514a6` (driver `580.178.04`, Python `3.15.2`) | `82a21ca` (driver `595.112.07`, Python `3.12.16`) |
|---|---|---|
| T14 and all other device-parametrised tests | `266 passed, 3 skipped` (`188.12 s`; 3 Apple GPU tests skipped) | `266 passed, 3 skipped` (`128.61 s`; 3 Apple GPU tests skipped) |
| Whole suite (`uv run pytest -q`, incl. T1–T14, network tests, and all 142 CPU/`gloo` multi-process items) | `735 passed, 4 skipped` (`603.65 s`; 3 Apple GPU tests + 1 optional `lacuna` skipped) | `735 passed, 4 skipped` (`433.05 s`; 3 Apple GPU tests + 1 optional `lacuna` skipped) |
| `head_only` gradients on CUDA against the reference | At most `1.68e-6` (bound `1e-5`) | At most `1.68e-06` (bound `1e-5`) |
| `device_parity.py --device cuda` (`5M` / `28M`) | `5M` fwd `1.19e-7`–`1.49e-5` (`bf16` `1.49e-2`, `fp16` `1.96e-3`), grad `1.15e-2`; `28M` fwd `1.04e-7`–`2.20e-6` (`bf16` `5.74e-2`, `fp16` `5.92e-3`), grad `2.24e-3` | `5M` fwd `1.08e-07`–`3.19e-05` (`bf16` `1.49e-02`, `fp16` `1.96e-03`), grad `1.15e-02`; `28M` fwd `1.13e-07`–`2.53e-06` (`bf16` `5.74e-02`, `fp16` `5.92e-03`), grad `2.24e-03` |
| `gradient_accuracy.py --device cuda` (`default` / `math` / `explicit`) | Tiny `1.6e-7`–`3.1e-7`; `5M` `1.7e-3`–`1.2e-2` / `3.6e-6`–`5.4e-5` / `6.0e-6`–`2.5e-5`; `28M` `1.4e-3`–`2.8e-3` / `4.8e-6`–`7.8e-5` / `3.5e-6`–`7.0e-5` | Tiny `1.6e-7`–`3.1e-7`; `5M` `1.7e-3`–`1.2e-2` / `3.6e-6`–`5.4e-5` / `6.0e-6`–`2.5e-5`; `28M` `1.4e-3`–`2.8e-3` / `4.8e-6`–`7.8e-5` / `3.5e-6`–`7.0e-5` |
| `step_timing.py --device cuda` (`full_ft`, `fp32`, eager, batch 32, ctx 1,024, `cpu threads 8`) | `5M` uni `52 ms` / `0.27 GiB` (`3.7×` vs CPU `190 ms`), cov5 `55 ms` / `1.10 GiB` (`14.3×` vs `786 ms`); `28M` uni `68 ms` / `0.98 GiB` (`9.7×` vs `654 ms`), cov5 `209 ms` / `3.39 GiB` (`14.7×` vs `3,081 ms`) | `5M` uni `34 ms` / `0.27 GiB` (`5.1×` vs CPU `171 ms`), cov5 `51 ms` / `1.10 GiB` (`16.2×` vs `834 ms`); `28M` uni `49 ms` / `0.98 GiB` (`13.8×` vs `678 ms`), cov5 `194 ms` / `3.39 GiB` (`19.2×` vs `3,715 ms`) |

The earlier `71514a6` one-off CUDA checks (one-process bit-identity vs `57a781c`, single-rank NCCL
collective-dtype smoke, and 2-/4-rank shared-GPU trainer trial) are kept as history in §11; the tracked
`82a21ca` Stage 3 driver and rank-failure session on the Python `3.12.16` L4 VM is recorded in §17.4.

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

## 16. Distributed training-step parity prototype (§5.6, standalone probe outside `src/`)

Backs the prototype `[V]` claims in spec §5.6 and §9 T15 (referenced by D22, D24, and D26). Every number in
this section measures the **standalone prototype outside `src/`** (`scripts/probes/distributed_parity.py`)
run with `torch.multiprocessing.spawn` on loopback (`127.0.0.1`), CPU, `gloo` backend, on `2026-11-23`
(macOS arm64). Implemented Stages 1–2 trainer tests (T15) are recorded separately in §17.

The prototype checks distributed steps against two single-process references: `probe_ref` (a union-batch
`_local_step`, run for every case and model) and independent `shipped_trainer` (a one-process
`TimestralTrainer`, run on `tiny_random` only: `training_step()` for `[2] unequal`, `[3] head_only`,
`lora_variate_step1`, `lora_temporal_step1`, and all four `[4]` replay/distillation regimes in `float64`
and `float32`; `fit()` with `cache_frozen_features=True` for `[3] head_only_cached` in `float32`, while
`float64` `head_only_cached` is reported `n/a` because `TimeSeriesDataset` stores `np.float32` inputs—a
`float64` model works with the shipped head cache in `fit()`, but `float64` input values cannot be fed
through `TimeSeriesDataset`).

Commands run:

```bash
uv run python scripts/probes/distributed_parity.py
uv run --isolated --python 3.12 --with "torch==2.4.0" python scripts/probes/distributed_parity.py
uv run python scripts/probes/distributed_parity.py --sizes 5M
```

### 16.1 Tiny random model across PyTorch 2.15.0 (Python 3.12.13) and PyTorch 2.4.0 (Python 3.12)

| Probe section & check | Scope | Bound | PyTorch 2.15.0 (`CPU`/`gloo`) | PyTorch 2.4.0 (`CPU`/`gloo`) |
|---|---|---:|---|---|
| `[1]` `ddp.module` bypass cross-rank grad rel L2 (`N=2`, ungraded control) | Prototype outside `src/` | — | `2.46e+00` (`max_abs=5.49e-02`, `identical=False`, f64 & f32) | `2.46e+00` (`max_abs=5.49e-02`, `identical=False`, f64 & f32) |
| `[1]` Proposed step (`N=2`, 6 checks: xrank, `probe_ref` grad & loss) | Prototype outside `src/` | `0.0e+00` / `1.0e-12` f64 / `1.0e-05` f32 | xrank `0.00e+00`; `probe_ref` grad `4.37e-16` f64, `2.32e-07` f32; loss `0.00e+00` f64, `1.29e-07` f32 | xrank `0.00e+00`; `probe_ref` grad `4.37e-16` f64, `2.43e-07` f32; loss `0.00e+00` f64, `6.46e-08` f32 |
| `[2]` Domain-loss parity (`equal`, `unequal`, `zero_scored`, `empty_shard`, `N=2` & `N=4`; 48 vs `probe_ref` + 8 vs `shipped_trainer` on `unequal`) | Prototype outside `src/` | `0.0e+00` / `1.0e-12` f64 / `1.0e-05` f32 | `plain_avg` `3.72e-01`–`9.85e-01` (non-equal); `N=2 empty_shard` (`W_d=[0, 112]`) `empty_r0_recv_ok=True`, loss `0.00e+00`, xrank `0.00e+00`; multi-rank `probe_ref` grad `3.31e-16`–`4.72e-16` f64 (`N=4 empty_shard` `4.72e-16`), `1.68e-07`–`2.82e-07` f32 (`N=4 empty_shard` `2.58e-07`); loss `0.00e+00`–`3.01e-16` f64, `0.00e+00`–`1.08e-07` f32; xrank `0.00e+00`; `shipped_trainer` (`unequal`) grad `3.86e-16`–`4.53e-16` f64 / `2.14e-07`–`2.82e-07` f32, loss `0.00e+00` f64 / `1.91e-08`–`9.73e-08` f32 | `plain_avg` `3.72e-01`–`9.85e-01` (non-equal); `N=2 empty_shard` (`W_d=[0, 112]`) `empty_r0_recv_ok=True`, loss `0.00e+00`, xrank `0.00e+00`; multi-rank `probe_ref` grad `3.11e-16`–`4.58e-16` f64 (`N=4 empty_shard` `4.16e-16`), `1.64e-07`–`2.69e-07` f32 (`N=4 empty_shard` `2.69e-07`); loss `0.00e+00`–`3.01e-16` f64, `0.00e+00`–`1.47e-07` f32; xrank `0.00e+00`; `shipped_trainer` (`unequal`) grad `4.03e-16`–`4.58e-16` f64 / `2.24e-07`–`2.45e-07` f32, loss `0.00e+00`–`1.68e-16` f64 / `1.74e-08`–`2.16e-08` f32 |
| `[2]` `all_zero_skip` & single-rank `non_finite` (`N=2` & `N=4`, f64 & f32; 8 checks) | Prototype outside `src/` | `0.0e+00` | `all_ranks_skipped=True`, `all_ranks_raised=True` (`no_grad=N/N`, `ref_raised=True`, clean follow-up collective) | `all_ranks_skipped=True`, `all_ranks_raised=True` (`no_grad=N/N`, `ref_raised=True`, clean follow-up collective) |
| `[3]` All four modes under unequal counts (`head_only`, `head_only_cached`, `lora_variate` step 0/1, `lora_temporal` step 0/1, `full_ft`, `N=2` & `N=4`; 100 vs `probe_ref`/LoRA inv + 28 vs `shipped_trainer`, f64 `head_only_cached` `n/a`) | Prototype outside `src/` | `0.0e+00` / `1.0e-12` f64 / `1.0e-05` f32 | `plain_avg` `6.75e-01`–`2.78e+00`; `probe_ref` grad `2.82e-16`–`6.89e-16` f64, `2.01e-07`–`4.41e-07` f32; loss `0.00e+00`–`2.28e-16` f64, `6.53e-09`–`1.04e-07` f32; xrank `0.00e+00`; LoRA step-0/1 invariants `0.00e+00`; `shipped_trainer` grad `2.82e-16`–`6.64e-16` f64 / `2.01e-07`–`3.67e-07` f32, loss `0.00e+00`–`2.28e-16` f64 / `6.53e-09`–`1.04e-07` f32 | `plain_avg` `6.75e-01`–`2.78e+00`; `probe_ref` grad `2.75e-16`–`7.51e-16` f64, `1.45e-07`–`3.76e-07` f32; loss `0.00e+00`–`2.30e-16` f64, `1.78e-08`–`1.48e-07` f32; xrank `0.00e+00`; LoRA step-0/1 invariants `0.00e+00`; `shipped_trainer` grad `2.75e-16`–`7.44e-16` f64 / `1.45e-07`–`3.48e-07` f32, loss `0.00e+00`–`2.30e-16` f64 / `1.78e-08`–`1.48e-07` f32 |
| `[4]` Replay & distillation parity (4 regimes, `N=2` & `N=4`; 48 vs `probe_ref` + 32 vs `shipped_trainer`) | Prototype outside `src/` | `0.0e+00` / `1.0e-12` f64 / `1.0e-05` f32 | `plain_avg` `8.97e-02`–`3.39e-01`; `probe_ref` and `shipped_trainer` (matching at 2 d.p.): grad `1.49e-16`–`1.92e-16` f64, `1.01e-07`–`5.57e-07` f32; loss `0.00e+00`–`1.37e-16` f64, `9.18e-09`–`9.98e-08` f32; xrank `0.00e+00` | `plain_avg` `8.97e-02`–`3.39e-01`; `probe_ref` and `shipped_trainer` (matching at 2 d.p.): grad `1.53e-16`–`7.41e-16` f64, `1.00e-07`–`2.27e-07` f32; loss `0.00e+00`–`2.13e-16` f64, `7.82e-09`–`1.28e-07` f32; xrank `0.00e+00` |
| `[5]` 10-active-step (+1 zero-scored skip) AdamW trajectories (`wd=0.01`, `warmup=3` + cosine, `clip=1.0`, `full_ft` & `lora_variate`, `N=2` & `N=4`; 48 checks) | Prototype outside `src/` | `1.0e-10` f64 / `1.0e-04` f32; xrank/clip/skip `0.0e+00` | `full_ft`: `final_params` `1.16e-16`–`1.52e-16` f64, `6.26e-08`–`6.52e-08` f32; `param_update` `2.69e-15`–`3.31e-15` f64, `1.37e-06`–`1.50e-06` f32; clipped `2/10` (`N=2`), `0/10` (`N=4`). `lora_variate`: `final_params` `1.34e-16`–`2.23e-16` f64, `7.75e-08`–`3.01e-07` f32; `param_update` `5.31e-15`–`7.73e-15` f64, `2.79e-06`–`1.26e-05` f32; clipped `0/10`. All: xrank `0.00e+00`, `clipped_vs_ref=0`, `clipped_xrank=0`, `skip_noop=True` | `full_ft`: `final_params` `9.50e-17`–`1.09e-16` f64, `5.27e-08`–`8.34e-08` f32; `param_update` `2.07e-15`–`2.52e-15` f64, `1.22e-06`–`1.82e-06` f32; clipped `2/10` (`N=2`), `0/10` (`N=4`). `lora_variate`: `final_params` `1.09e-16`–`5.32e-16` f64, `4.23e-08`–`2.28e-07` f32; `param_update` `4.33e-15`–`1.84e-14` f64, `1.52e-06`–`9.50e-06` f32; clipped `0/10`. All: xrank `0.00e+00`, `clipped_vs_ref=0`, `clipped_xrank=0`, `skip_noop=True` |
| `[6]` `DistributedSampler` (10 items, 4 replicas, `shuffle=False`) | Prototype outside `src/` | observation | `drop_last=False`: `duplicated=[0, 1]`, `dropped=[]`; `drop_last=True`: `duplicated=[]`, `dropped=[8, 9]` | identical |
| **Summary (`tiny_random`)** | Prototype outside `src/` | all enforced | **`326/326` passed, `0` failed** | **`326/326` passed, `0` failed** |

Also run on Linux x86_64 (`CPU`/`gloo`, `tiny_random`, `2026-11-23`): **`326/326` passed, `0` failed** on both
`torch 2.14.1+cu130` (8-vCPU cloud VM at `3347f70`) and `torch 2.15.0+cu132` (L4 VM's CPU at `71514a6`).

### 16.2 Released `Timestral-5M` run (`--sizes 5M`, PyTorch 2.15.0, Python 3.12.13, `CPU`/`gloo`)

The `--sizes 5M` run reports `516/516 passed, 0 failed, 68 info (ungraded)`: the `516` enforced passes are
the `326` `tiny_random` checks (values identical to §16.1) plus **`190` enforced `Timestral-5M` checks**
(cross-rank agreement across `[1]`–`[5]`, scalar global loss parity across `[1]`–`[4]`, `N=2 empty_shard`
empty-rank-0 delivery, global skip and non-finite control flow, `head_only` & `head_only_cached` gradients,
LoRA step-0/1 invariants, and `[5]` clipping-parity and skipped-step no-op invariants). All **`68`
`Timestral-5M` body-backprop gradient and parameter-trajectory comparisons against `probe_ref` are ungraded
`[INFO]` values** with no pass/fail verdict (`shipped_trainer` runs on `tiny_random` only); `516/516` does
**not** mean released-weights body-backprop parity passed a bound.

| Probe section & check | Scope & grading | Bound | Measured (`Timestral-5M`, `CPU`/`gloo`, PyTorch 2.15.0) |
|---|---|---:|---|
| `[1]` `ddp.module` bypass cross-rank grad rel L2 (`N=2`) | Prototype outside `src/` (control) | — | `1.04e+00` (`max_abs=7.52e-01` f64 / `7.53e-01` f32, `identical=False`) |
| `[1]`–`[4]` Cross-rank grad rel L2 (`N=2` & `N=4`, 62 checks) | Prototype outside `src/` (**enforced**) | `0.0e+00` | `0.00e+00` (`[1]` 2, `[2]` 16, `[3]` 28, `[4]` 16; f64 & f32; 62 PASS) |
| `[1]`–`[4]` Scalar global loss vs `probe_ref` (`N=2` & `N=4`, 62 checks) | Prototype outside `src/` (**enforced**) | `1.0e-12` f64 / `1.0e-05` f32 | `0.00e+00`–`6.77e-16` f64, `0.00e+00`–`4.75e-07` f32 (`[1]` `3.84e-16` / `2.06e-07`; `[2]` `0.00e+00`–`6.77e-16` / `0.00e+00`–`4.75e-07`; `[3]` `0.00e+00`–`3.80e-16` / `2.14e-08`–`2.72e-07`; `[4]` `0.00e+00`–`5.52e-16` / `1.33e-08`–`1.69e-07`; 62 PASS) |
| `[2]` `N=2 empty_shard` (`W_d=[0, 112]`), `all_zero_skip`, & `non_finite` (`N=2` & `N=4`, 10 checks) | Prototype outside `src/` (**enforced**) | `0.0e+00` | `empty_r0_recv_ok=True` (2 PASS), `all_ranks_skipped=True` (4 PASS), `all_ranks_raised=True` (`no_grad=N/N`, `ref_raised=True`, 4 PASS) |
| `[3]` `head_only` & `head_only_cached` grad under unequal counts (`N=2` & `N=4`, 8 checks) | Prototype outside `src/` (**enforced**) | `1.0e-12` f64 / `1.0e-05` f32 | `plain_avg` `7.89e-01`–`1.79e+00`; proposed `5.20e-16`–`1.12e-15` f64, `3.59e-07`–`4.00e-07` f32 (8 PASS) |
| `[3]` LoRA step-0/1 invariants (`lora_variate` & `lora_temporal`, `N=2` & `N=4`, 16 checks) | Prototype outside `src/` (**enforced**) | `0.0e+00` | `0.00e+00` (step-0 `|B|_max=0.0e+00, ||gA||=0.0e+00, ||gB|| > 0`; step-1 all `> 0`; 16 PASS) |
| `[5]` 10-active-step (+1 skip) trajectory invariants (`full_ft` & `lora_variate`, `N=2` & `N=4`, 32 checks) | Prototype outside `src/` (**enforced**) | `0.0e+00` | cross-rank params `0.00e+00` (8 PASS), `clipped_vs_ref=0` (`full_ft`: `4/10` at `N=2`, `3/10` at `N=4`; `lora_variate`: `0/10`; 8 PASS), `clipped_xrank=0` (8 PASS), `skip_noop=True` (8 PASS) |
| `[1]` Proposed step vs `probe_ref`, body backprop (`N=2`, 2 values) | Prototype outside `src/` (**ungraded `[INFO]`**) | none | `1.97e-12` f64, `7.55e-04` f32 |
| `[2]` Domain-loss body backprop vs `probe_ref` (`N=2` & `N=4`, 14 values) | Prototype outside `src/` (**ungraded `[INFO]`**) | none | `plain_avg` `3.66e-01`–`1.78e+00` (non-equal); proposed `1.13e-12`–`2.42e-12` f64 (`N=4 empty_shard` `1.24e-12`), `6.64e-04`–`1.43e-03` f32 (`N=4 empty_shard` `1.16e-03`) |
| `[3]` `lora_variate` step 0/1, `lora_temporal` step 0/1, `full_ft` vs `probe_ref` (`N=2` & `N=4`, 20 values) | Prototype outside `src/` (**ungraded `[INFO]`**) | none | `plain_avg` `4.63e-01`–`1.35e+00`; proposed `2.22e-14`–`1.91e-12` f64, `1.62e-04`–`1.58e-03` f32 |
| `[4]` Replay & distillation body backprop vs `probe_ref` (`N=2` & `N=4`, 16 values) | Prototype outside `src/` (**ungraded `[INFO]`**) | none | `plain_avg` `2.39e-01`–`7.38e-01`; proposed `3.00e-13`–`2.16e-12` f64, `3.24e-04`–`1.31e-03` f32 |
| `[5]` 10-active-step (+1 skip) trajectories vs `probe_ref` (`full_ft` & `lora_variate`, `N=2` & `N=4`, 16 values) | Prototype outside `src/` (**ungraded `[INFO]`**) | none | `full_ft`: `final_params` `3.37e-13`–`4.48e-13` f64 / `2.85e-05`–`3.02e-05` f32, `param_update` `5.35e-10`–`7.40e-10` f64 / `4.70e-02`–`4.80e-02` f32; `lora_variate`: `final_params` `2.70e-14`–`1.12e-13` f64 / `5.25e-05`–`1.22e-04` f32, `param_update` `2.33e-12`–`9.60e-12` f64 / `4.57e-03`–`1.06e-02` f32 |
| **Summary (`tiny_random` + `Timestral-5M`)** | Prototype outside `src/` | enforced only | **`516/516` passed (`326` tiny + `190` `5M`), `0` failed, `68` `[INFO]` (ungraded `5M` body backprop)** |

**Not verified by the prototype (all design choices `[D]`, spec §12):** CUDA, NCCL, `cuda:{LOCAL_RANK}` /
`cuda:{rank}` device binding, `bf16` autocast, `torch.compile`, `torchrun` on Linux, released checkpoints
`28M`/`123M`/`209M`, multi-GPU throughput or scaling efficiency, and rank-failure behaviour under NCCL. The
`ReduceOp.AVG` vs `ReduceOp.SUM` check on CPU `gloo` across PyTorch 2.4.0 and 2.15.0 is recorded as a
one-off result in §11 (not covered by this probe).

## 17. Phase 5 Stages 1–2: Implemented multi-process training (CPU/`gloo`)

### 17.1 Multi-process trainer, acceptance test, and headroom runs (CPU/`gloo`)

Measured on `2026-11-23` across macOS arm64 (commit `40a580a`, Apple M4, Python `3.12.13`) and Linux x86_64
(commit `71514a6`—`40a580a` differs in comments only—on an 8-vCPU AMD EPYC 7B12 Ubuntu 22.04 cloud VM and on
the single-L4 VM's CPU), exercising the implemented Stages 1–2 multi-process trainer
(`src/timestral_tune/trainer/distributed.py`, `_distributed_strategy.py`, `_types.py`, `trainer.py`, and
`export.py`) on CPU/`gloo` via `tests/distributed_harness.py` (`spawn`, file-based rendezvous, 1 intra-op CPU
thread per worker restored in `finally`, shared parity tiers and `TIMESTRAL_TEST_F32_TOL_SCALE` headroom knob,
hard wall-clock deadlines; tiny random model).

Commands run on macOS arm64 (`40a580a`):

```bash
uv run pytest tests/acceptance/test_distributed.py --collect-only -q
TMPDIR=/tmp uv run pytest -m "not network" -q -p no:cacheprovider
TMPDIR=/tmp TIMESTRAL_TEST_F32_TOL_SCALE=0.1 uv run pytest tests/test_distributed.py tests/test_trainer_distributed.py tests/test_trainer_distributed_replay.py tests/acceptance/test_distributed.py -q -p no:cacheprovider
TMPDIR=/tmp uv run --isolated --python 3.12 --with "torch==2.4.0" pytest tests/test_distributed.py tests/test_trainer_distributed.py tests/test_trainer_distributed_replay.py tests/acceptance/test_distributed.py -q -p no:cacheprovider
```

| Suite (macOS arm64, `40a580a`) | PyTorch `2.15.0` (`CPU`/`gloo`, Python `3.12.13`) | PyTorch `2.4.0` (`CPU`/`gloo`, Python `3.12`) |
|---|---|---|
| Full network-free suite (`-m "not network"`) | **`511 passed, 15 skipped, 213 deselected`** (`265.63 s`, `0` failed; skips: 1 optional `lacuna`, 14 CUDA `bf16`) | Not run in full here (four distributed files below) |
| Acceptance test T15 (`tests/acceptance/test_distributed.py`, cases `(a)–(l)`, 19 test functions / 34 collected items, `N = 2` and `N = 4`, tiny random model) | **`34 passed`** (`0` skipped, `0` failed, within both the `-m "not network"` run above and the `scale=0.1` 4-file run below; includes `test_t15_f_dataloader_workers` at `N = 4`, `test_t15_h_compile_default_smoke`, and `test_t15_l_compile_default_replay_distill_smoke`) | **`33 passed, 1 skipped`** within the 4-file run below (`test_t15_f_dataloader_workers` skipped on macOS PyTorch `< 2.8` per §10; both compiled smoke tests passed) |
| All four multi-process test files (`tests/test_distributed.py`, `tests/test_trainer_distributed.py`, `tests/test_trainer_distributed_replay.py`, `tests/acceptance/test_distributed.py`; `142` items collected) | **`142 passed`** at default scale (`1.0`, inside `-m "not network"`) and **`142 passed`** at **`TIMESTRAL_TEST_F32_TOL_SCALE=0.1`** (`139.02 s`, `>= 10×` float32 headroom) | **`138 passed, 4 skipped`** (`142.43 s`; all 4 skips are `num_workers > 0` on macOS PyTorch `< 2.8` per §10: 2 in `tests/test_distributed.py`, 1 in `tests/test_trainer_distributed_replay.py`, 1 in `tests/acceptance/test_distributed.py`) |

Linux x86_64 runs (`71514a6`, 8-vCPU AMD EPYC 7B12 Ubuntu 22.04 VM with `uv`-managed Python `3.12.16` /
`torch 2.15.0+cu132` and `uv`-managed Python `3.10.22` / `torch 2.14.1+cu130`; plus standalone 4-file run on
the L4 VM's CPU with Python `3.15.2` / `torch 2.15.0+cu132`):

| Suite & configuration (Linux x86_64, `71514a6`, `CPU`/`gloo`) | Result |
|---|---|
| Full network-free suite (`-m "not network"`), Python `3.12.16` / `torch 2.15.0+cu132` | **`469 passed, 24 skipped, 169 deselected`** (`265.26 s`; skips: 6 device parity, 1 `lacuna`, 2 MPS loader, 1 `test_pinball`, 14 CUDA `bf16`) |
| Full network-free suite (`-m "not network"`), managed Python `3.10.22` / `torch 2.14.1+cu130` | **`469 passed, 24 skipped, 169 deselected`** (`424.19 s`; identical skip breakdown) |
| Four multi-process files (`142` items), Python `3.12.16` / `torch 2.15.0+cu132`, `TIMESTRAL_TEST_F32_TOL_SCALE=0.2` (`>= 5×` headroom), `OMP_NUM_THREADS=1` / `2` / `4` (acceptance file last) | **`142 passed`** at `t=1` (`229.72 s`), **`142 passed`** at `t=2` (`167.44 s`), **`142 passed`** at `t=4` (`167.43 s`) |
| Four multi-process files (`142` items), Python `3.12.16` / `torch 2.15.0+cu132`, `TIMESTRAL_TEST_F32_TOL_SCALE=0.2`, acceptance file first (default threads) | **`142 passed`** (`167.33 s`) |
| Four multi-process files (`142` items), Python `3.12.16` / `torch 2.15.0+cu132`, `TIMESTRAL_TEST_F32_TOL_SCALE=0.1` (`>= 10×` headroom), default threads | **`142 passed`** (`168.02 s`) |
| Four multi-process files (`142` items), managed Python `3.10.22` / `torch 2.14.1+cu130`, `TIMESTRAL_TEST_F32_TOL_SCALE=0.2`, `OMP_NUM_THREADS=2` | **`142 passed`** (`177.22 s`) |
| Four multi-process files (`142` items), L4 VM CPU, Python `3.15.2` / `torch 2.15.0+cu132`, default scale (`1.0`) | **`142 passed`** (`199.97 s`) |

*Pre-fix Linux x86_64 baseline (`3347f70`, separate 8-vCPU Ubuntu 22.04 VM):* `-m "not network"` passed with
`468 passed, 24 skipped, 169 deselected` on both managed Python `3.12.16` (`468.61 s`) and managed Python
`3.10.22` (`573.93 s`; running under Ubuntu's system Python `3.10.12` failed the 5 `torch.compile` tests on
missing `Python.h`, an environment artefact). Running the four multi-process files alone (`acceptance` last)
failed two float32 `param_update` checks (`1.07e-04` in `T15(j)[2]` and `1.71e-04` in `T15(l)[4]` vs `1e-4`;
`2 failed, 139 passed` on Python `3.12.16`, and `4 failed, 137 passed` across three repeats under headerless
system Python `3.10.12` where the two compiled smoke tests also hit missing `Python.h`) because `_worker_main`
leaked `num_threads=1` into the pytest parent process (`e33cbf5` fixed the leak and added `_guard_torch_global_state`
in `tests/conftest.py`; `0f80f4c` recalibrated the float32 tiers).

### 17.2 PyTorch `F.linear` `requires_grad` dispatch probe (`scripts/probes/linear_requires_grad_dispatch.py`)

Backs the `[V]` claims in spec §9 T15(j) and §12. Measured on `2026-11-23` on CPU (`threads=4`, `seed=0`)
across macOS arm64 (at `3347f70`, PyTorch `2.15.0` and `2.4.0`) and three Linux x86_64 cloud VMs: VM 1 (at
`3347f70`, 8 vCPU, `torch 2.14.1+cu130`), VM 2 (at `71514a6`, 8 vCPU AMD EPYC 7B12, `torch 2.15.0+cu132` and
`torch 2.14.1+cu130`), and the L4 VM's CPU (at `71514a6`, `torch 2.15.0+cu132`). GPU/CUDA `F.linear` dispatch
is unmeasured by this CPU-only probe.

Commands run:

```bash
uv run python scripts/probes/linear_requires_grad_dispatch.py
uv run --isolated --python 3.12 --with "torch==2.4.0" python scripts/probes/linear_requires_grad_dispatch.py
```

**macOS arm64 CPU:**

| Probe section & case | PyTorch `2.15.0` (macOS arm64 CPU) | PyTorch `2.4.0` (macOS arm64 CPU) |
|---|---|---|
| `[a]` Standalone `F.linear`, `float32` contiguous (all 7 shapes) | `0.00e+00` across all four columns (`grad:T-F`, `nograd:T-F`, `rgT:g-ng`, `rgF:g-ng`) | Identical (`0.00e+00` everywhere) |
| `[a]` Standalone `F.linear`, `float32` permuted `(1, 0, 2)` non-contiguous (7 shapes) | `grad:T-F == nograd:T-F`: `(8,5,32)->32`: `3.81e-06`; `(2,6,64)->64`: `1.34e-05`; `(4,6,64)->64`: `9.54e-06`; `(2,6,64)->192`: `7.63e-06`; `(1,6,64)->64`, `(8,6,64)->64`, `(8,6,64)->192`: `0.00e+00`. `rgT:g-ng` and `rgF:g-ng`: `0.00e+00` for every shape | Identical in every cell |
| `[a]` Standalone `F.linear`, `float64` (contiguous & permuted, all 7 shapes) | `0.00e+00` in every cell | Identical (`0.00e+00` everywhere) |
| `[b]` Tiny random model, `student(train, rg=T)` vs `teacher(eval, no_grad, rg=T)` | `0.00e+00` (`max_abs` & `rel_l2`) across all `float32`/`float64`, univariate/covariates, `bs=1,2,4,8` | Identical (`0.00e+00` everywhere) |
| `[b]` Tiny random model, `student(train, rg=T)` vs `frozen(rg=F)` teacher, `float32` | Uni `bs=1`: `0.00e+00`; `bs=2`: `max_abs 4.77e-07` (`rel_l2 1.94e-07`); `bs=4`: `8.34e-07` (`2.23e-07`); `bs=8`: `5.96e-07` (`2.06e-07`). Cov `bs=1`: `3.58e-07` (`2.14e-07`); `bs=2`: `7.15e-07` (`2.03e-07`); `bs=4,8`: `0.00e+00` | **Differs from 2.15.0:** Uni `bs=1`: `4.77e-07` (`1.64e-07`); `bs=2`: `5.96e-07` (`2.32e-07`); `bs=4`: `5.96e-07` (`2.15e-07`); `bs=8`: `5.96e-07` (`2.16e-07`). Cov `bs=1`: `4.77e-07` (`1.80e-07`); `bs=2`: `5.96e-07` (`2.10e-07`); `bs=4,8`: `0.00e+00` |
| `[b]` Tiny random model, `student(train, rg=T)` vs `frozen(rg=F)` teacher, `float64` | `0.00e+00` across all 8 cells (`uni`/`cov`, `bs=1,2,4,8`) | **Differs from 2.15.0:** Uni `bs=1`: `max_abs 8.88e-16` (`rel_l2 3.76e-16`); all other 7 cells `0.00e+00` |
| `[c]` Step-0 `distillation_loss` & `grad_l2` at identical weights, `unfrozen(rg=T)` teacher | `0.00e+00` / `0.00e+00` in every cell (`float32` & `float64`, `beta=1e-3` & `0.0`) | Identical (`0.00e+00` / `0.00e+00` everywhere) |
| `[c]` Step-0 `distillation_loss` & `grad_l2`, `frozen(rg=F)` teacher, `float32` | Non-zero on uni `bs=2,4,8` and cov `bs=1,2` (zero on uni `bs=1`, cov `bs=4,8`): `beta=1e-3` loss `9.11e-12`–`1.25e-11`, `grad_l2` `8.82e-05`–`3.04e-04` (peak cov `bs=1`); `beta=0.0` loss `1.05e-07`–`1.21e-07`, `grad_l2` `5.69e-01`–`1.76e+00` | **Differs from 2.15.0:** Non-zero on uni `bs=1,2,4,8` and cov `bs=1,2` (zero on cov `bs=4,8`): `beta=1e-3` loss `6.73e-12`–`1.31e-11`, `grad_l2` `8.58e-05`–`2.55e-04` (peak uni `bs=2`); `beta=0.0` loss `8.88e-08`–`1.20e-07`, `grad_l2` `5.41e-01`–`1.47e+00` |
| `[c]` Step-0 `distillation_loss` & `grad_l2`, `frozen(rg=F)` teacher, `float64` | `0.00e+00` / `0.00e+00` in every cell | **Differs from 2.15.0:** Uni `bs=1`: `beta=1e-3` loss `3.53e-29`, `grad_l2` `5.07e-13`; `beta=0.0` loss `2.17e-16`, `grad_l2` `1.76e+00`; all other 7 cells `0.00e+00` / `0.00e+00` |

**Linux x86_64 CPU (three cloud VMs; contiguous `[a]`, `rgT:g-ng`, `rgF:g-ng`, and `unfrozen(rg=T)` `[b]`/`[c]` are `0.00e+00` in every cell on all three VMs):**

| Probe section & case | VM 1 (`3347f70`, `torch 2.14.1+cu130`) | VM 2 AMD EPYC 7B12 (`71514a6`, `torch 2.15.0+cu132` & `2.14.1+cu130`, cell-for-cell identical) | L4 VM CPU (`71514a6`, `torch 2.15.0+cu132`) |
|---|---|---|---|
| `[a]` Standalone `F.linear`, `float32` permuted `(1, 0, 2)` (`grad:T-F == nograd:T-F`) | 5/7 non-zero: `(8,5,32)->32`: `4.77e-06`; `(2,6,64)->64`: `9.54e-06`; `(8,6,64)->64`: `7.63e-06`; `(2,6,64)->192`: `5.72e-06`; `(8,6,64)->192`: `7.63e-06`; `(1,6,64)->64` & `(4,6,64)->64`: `0.00e+00` | 2/7 non-zero: `(2,6,64)->64`: `1.14e-05`; `(2,6,64)->192`: `7.63e-06`; other 5 shapes `0.00e+00` | Same 2/7 non-zero, different magnitudes: `(2,6,64)->64`: `9.54e-06`; `(2,6,64)->192`: `6.68e-06`; other 5 shapes `0.00e+00` |
| `[a]` Standalone `F.linear`, `float64` permuted `(1, 0, 2)` (`grad:T-F == nograd:T-F`) | **Non-zero (3/7):** `(8,5,32)->32`: `5.77e-15`; `(2,6,64)->64`: `1.42e-14`; `(8,6,64)->64`: `1.07e-14`; other 4 shapes `0.00e+00` | **Non-zero (2/7):** `(2,6,64)->64`: `1.42e-14`; `(2,6,64)->192`: `1.78e-14`; other 5 shapes `0.00e+00` | **Non-zero (2/7):** `(2,6,64)->64`: `1.07e-14`; `(2,6,64)->192`: `1.78e-14`; other 5 shapes `0.00e+00` |
| `[b]` Tiny random model, `frozen(rg=F)`, `float32` | Uni `bs=1`: `0.00e+00`; other 7/8 cells non-zero (incl. `cov bs=4,8`): `max_abs` `4.17e-07`–`7.15e-07` (`rel_l2` `1.42e-07`–`2.47e-07`) | Uni `bs=1`: `0.00e+00`; other 7/8 cells non-zero: `max_abs` `4.17e-07`–`6.56e-07` (`rel_l2` `1.48e-07`–`2.50e-07`) | Uni `bs=1`: `0.00e+00`; other 7/8 cells non-zero: `max_abs` `4.77e-07`–`5.96e-07` (`rel_l2` `1.50e-07`–`2.18e-07`) |
| `[b]` Tiny random model, `frozen(rg=F)`, `float64` | Uni `bs=1`: `0.00e+00`; other 7/8 cells non-zero: `max_abs` `8.88e-16`–`1.11e-15` (`rel_l2` `1.57e-16`–`4.08e-16`) | Uni `bs=1`: `0.00e+00`; other 7/8 cells non-zero: `max_abs` `8.88e-16`–`1.33e-15` (`rel_l2` `3.12e-16`–`4.71e-16`) | Uni `bs=1`: `0.00e+00`; other 7/8 cells non-zero: `max_abs` `8.88e-16`–`1.33e-15` (`rel_l2` `2.80e-16`–`3.99e-16`) |
| `[c]` Step-0 `distillation_loss` & `grad_l2`, `frozen(rg=F)`, `float32` (7 non-zero cells) | `beta=1e-3`: loss `5.06e-12`–`1.48e-11`, `grad_l2` `7.82e-05`–`2.94e-04` (peak cov `bs=1`); `beta=0.0`: loss `6.76e-08`–`1.37e-07`, `grad_l2` `4.97e-01`–`1.46e+00` | `beta=1e-3`: loss `5.74e-12`–`1.52e-11`, `grad_l2` `4.99e-05`–`2.72e-04` (peak uni `bs=2`); `beta=0.0`: loss `7.92e-08`–`1.32e-07`, `grad_l2` `3.83e-01`–`1.71e+00` | `beta=1e-3`: loss `5.91e-12`–`1.16e-11`, `grad_l2` `6.82e-05`–`2.16e-04` (peak cov `bs=1`); `beta=0.0`: loss `7.97e-08`–`1.16e-07`, `grad_l2` `4.57e-01`–`1.61e+00` |
| `[c]` Step-0 `distillation_loss` & `grad_l2`, `frozen(rg=F)`, `float64` (7 non-zero cells) | `beta=1e-3`: loss `6.20e-30`–`4.17e-29`, `grad_l2` `8.36e-14`–`5.42e-13`; `beta=0.0`: loss `4.03e-17`–`2.22e-16`, `grad_l2` `2.53e-01`–`1.65e+00` | `beta=1e-3`: loss `2.57e-29`–`5.40e-29`, `grad_l2` `1.18e-13`–`5.31e-13`; `beta=0.0`: loss `1.66e-16`–`2.67e-16`, `grad_l2` `4.37e-01`–`1.40e+00` | `beta=1e-3`: loss `1.98e-29`–`3.88e-29`, `grad_l2` `1.05e-13`–`6.19e-13`; `beta=0.0`: loss `1.47e-16`–`2.23e-16`, `grad_l2` `4.99e-01`–`1.96e+00` |

### 17.3 Float32 intra-op thread trajectory noise probe (`scripts/probes/float32_trajectory_noise.py`)

Backs the `[V]` platform thread-sensitivity claim in spec §9 (T15 shared tolerance tiers). Runs deterministic
6-step one-process `float32` `TimestralTrainer.fit()` trajectories on `tiny_random` across all four modes and
all four replay/distillation regimes (`bs=[4, 8]`, `seed=0`) at intra-op thread counts `1`, `2`, and `4`.

Commands run:

```bash
uv run python scripts/probes/float32_trajectory_noise.py
uv run --isolated --python 3.12 --with "torch==2.4.0" python scripts/probes/float32_trajectory_noise.py
```

| Platform & PyTorch | `max_param_update_rel_l2` (worst cell) | `max_final_params_rel_l2` | `max_loss_rel` | Pattern across thread pairs (`1v2`, `1v4`, `2v4`) |
|---|---|---|---|---|
| macOS arm64 (`40a580a`, `threads=4`), PyTorch `2.15.0` and `2.4.0` | `0.00e+00` (all 32 rows) | `0.00e+00` | `0.00e+00` | Bit-identical across threads `1`, `2`, and `4` in every cell on both versions |
| Linux x86_64 AMD EPYC 7B12 (`71514a6`, `threads=4`), PyTorch `2.15.0+cu132` and `2.14.1+cu130` (cell-for-cell identical) | `8.08e-06` (`bs=4 full_ft/both (1v2)`) | `2.66e-07` (`bs=4 lora_temporal/both`) | `7.84e-07` (`bs=4 lora_temporal/both`) | `1v4` is `0.00e+00` in every row; `1v2 == 2v4` is non-zero (`1.22e-07`–`8.08e-06`) across all 32 rows |

Note: this probe demonstrates that Linux x86_64 one-process `float32` trajectories depend on intra-op thread
count while macOS arm64 trajectories do not; on this synthetic workload the probe reaches `8.08e-06`, whereas
the larger `1.14e-04` one-process thread-count difference that motivated `TRAJ_UPDATE_TOL["float32"] = 1e-3`
occurred on a T15(j) `lora_variate` distillation scenario (one-off measurement in §11).

### 17.4 Stage 3 measuring driver & rank-failure sessions (`scripts/multi_gpu_check.sh`, commit `82a21ca`)

Backs the `[V]` claims in spec §5.6, §7.6, §9, D28, D35, and §12 for the tracked Stage 3 scripts
(`scripts/probes/distributed_trainer_check.py`, `scripts/probes/distributed_rank_failure_check.py`, and
`scripts/multi_gpu_check.sh`). **All numbers in this subsection were produced by commit `82a21ca` of those
scripts on `2026-11-23`.** An independent code review of the `82a21ca` scripts identified tooling defects
summarised under *Known limitations of the tooling at `82a21ca`* below; the GPU session will be repeated
after those review fixes land, and no Stage 3 roadmap box is ticked on the strength of this session.

`scripts/probes/distributed_trainer_check.py` is an ungraded measuring driver (41 scenarios across groups
`g1`–`g8`, 6 steps per scenario except 4 for `export_multirank_f32`; `B = 2`, `b = 2`, `seed = 0`), separate
from acceptance test T15 (`tests/acceptance/test_distributed.py`, CPU/`gloo` only). Reproduces and supersedes
the earlier `71514a6` one-off shared-GPU NCCL trial in §11.

**Environments and commands (`82a21ca`, `2026-11-23`):**

- **Single-L4 shared-GPU NCCL + Linux CPU/`gloo` session:** one NVIDIA L4 (`23034 MiB`, driver `595.112.07`)
  on a Linux x86_64 cloud VM (8 vCPU), Python `3.12.16`, `torch 2.15.0+cu132`, CUDA runtime `13.2`, NCCL
  `2.30.7`, `python_h=True`, `c_compiler=True`.
  ```bash
  bash scripts/multi_gpu_check.sh --shared-gpu --python 3.12
  ```
  Sets `NCCL_MULTI_RANK_GPU_ENABLE=1` and `--device-map shared` (all ranks share `cuda:0`). `summary.txt`
  recorded 16 stages: **15 exited `0`**, 1 skipped (`02_prototype_parity_cuda`: `SKIPPED (needs 4 GPUs)`),
  zero hangs. Reference archives (`--reference-mode sketch`, `k = 8,192`): `tiny_random` `2.18 MiB` (40 CUDA
  entries) / `1.86 MiB` (34 CPU entries); `5M`+`28M` combined `5.14 MiB` (80 CUDA entries).
- **Reference-mode check (`exact` vs `sketch`, same L4 VM and commit `82a21ca`):** five scenarios
  (`domain_full_ft_f64`, `domain_lora_temporal_f32`, `replay_full_ft_distill_only_f64`,
  `bf16_lora_temporal`, `bf16_full_ft_replay`) on `tiny_random` and `5M` at `N = 2`
  (`--backend nccl --device-map shared`, `NCCL_MULTI_RANK_GPU_ENABLE=1`), writing and running both
  `--reference-mode exact` and `--reference-mode sketch`.
- **macOS arm64 CPU/`gloo` session (reproduced locally on Apple M4, commit `82a21ca`):** Darwin arm64,
  Python `3.12.13`, `torch 2.15.0`, `python_h=True`, `c_compiler=True`, `threads=1`.
  ```bash
  bash scripts/multi_gpu_check.sh --backend gloo --ranks 2 4 --pg-timeout 10 --log-dir /tmp/tt_docs_mgc
  ```
  8 stages listed (`350.0 s` wall): **7 exited `0`**, 1 skipped (`02_prototype_parity_cuda`: `SKIPPED (needs
  4 GPUs)`), zero hangs (temporary log directory deleted afterwards).

**What the `82a21ca` parity numbers measure (and how to read the table):** across all executed rows,
`xr_eq`/`xr_max` compare every **parameter** (`model.named_parameters()`, omitting buffers), and
`param_l2`/`upd_l2` compare **trainable parameters** (`trainer._trainable`) against the one-process reference
on the same device. In `--reference-mode sketch` (`k = 8,192`), `float32`, `bf16`, and `float64` differences
of `>= 1e-14` agree with `--reference-mode exact` to within ~1–2% (see the comparison table below), whereas
`float64` values below `~1e-14` on large parameter sets (`head_only`/`full_ft` on `5M` and `28M`) sit at the
sketch's own float64 bucket-accumulation noise floor (1.3×–7× inflation measured for true diffs of
`1e-16`–`3e-15`; values `>= 1e-13` are unaffected) and are marked `(<= ~1e-14 floor)` below rather than
treated as precise sub-`1e-14` measurements. On CUDA, the two `moved_student` `float32` rows
(`replay_full_ft_distill_only_moved_f32`, `replay_full_ft_both_moved_f32`) are provisional because rank 0 and
the reference loaded a CPU-computed `rope.inv_freq` buffer while peer ranks kept the CUDA-computed one
(neither row was the worst `float32` row on `tiny`, `5M`, or `28M`).

#### Driver parity summary (`distributed_trainer_check.py`, commit `82a21ca`, `--reference-mode sketch`)

Every executed row across all runs below reported **`xr_eq = yes` and `xr_max = 0.00e+00`** over all model
parameters (`246/246` NCCL rows on L4: `tiny_random`, `5M`, `28M` × `N = 2, 4`; `70/70` Linux CPU/`gloo` rows
on L4; `70/70` macOS arm64 CPU/`gloo` rows). `val_restore_early_stop_*` returned
`early=True restored=True best_step=2`, `export_multirank_f32` returned `export=ok(N/N)`, and
`stochastic_lora_variate_f32` matched bitwise across ranks in every run.

| Platform, backend, model, `N` | Executed / skipped | `float64` (14 rows): `param_l2` range (med); `upd_l2` min / med / max (worst scenario); `loss_rel` max; `val_rel` | `float32` (20 rows; CPU splits 18 non-id + 2 id-start): `param_l2` range (med); `upd_l2` min / med / max (worst scenario); `loss_rel` max; `val_rel` | `bf16` (6 rows, CUDA only; **no tier set**): `param_l2` range; `upd_l2` min / max (worst); `loss_rel` max |
|---|---|---|---|---|
| **L4 shared NCCL**, `tiny_random`, `N = 2` (`82a21ca`) | `41 / 0` | `param_l2` `<= 1.86e-15` (all `< 1e-14` floor); `upd_l2` `1.94e-15` / `4.70e-15` (floor) / **`4.62e-14`** (`replay_full_ft_distill_only_f64`, ~`2,160×` below T15 `1e-10`); `loss_rel` `<= 1.38e-15`; `val_rel` `2.23e-16` | `param_l2` `5.22e-08`–`3.43e-06` (med `1.83e-07`); `upd_l2` `1.88e-06` / `3.45e-06` / **`6.96e-05`** (`compile_default_lora_replay`; non-compile max `5.62e-06`; id-start `3.48e-06`/`5.56e-06`); `loss_rel` `<= 4.50e-06`; `val_rel` `3.94e-08`–`4.22e-08` | `param_l2` `3.96e-04`–`2.95e-03`; `upd_l2` `9.26e-03` (`bf16_head_only`) / **`5.75e-02`** (`bf16_lora_temporal`); `loss_rel` `<= 4.77e-03` (`bf16_lora_variate_replay`) |
| **L4 shared NCCL**, `tiny_random`, `N = 4` (`82a21ca`) | `41 / 0` | `param_l2` `<= ~1e-14` floor except 2 rows up to `1.80e-13`; `upd_l2` `1.54e-15` / `4.50e-15` (floor) / **`7.81e-12`** (`replay_full_ft_distill_only_f64`, `12.8×` below T15 `1e-10`; 2nd `1.75e-12` `replay_full_ft_both_f64`; all other 12 `<= 9.78e-15`); `loss_rel` `<= 3.53e-14`; `val_rel` `0.00e+00` | `param_l2` `6.30e-08`–`4.81e-06` (med `5.19e-07`); `upd_l2` `9.78e-07` / `3.82e-06` / **`1.00e-04`** (`compile_default_lora_replay`; non-compile max `3.03e-05` `replay_lora_variate_distill_only_f32` and provisional `replay_full_ft_both_moved_f32`; id-start `3.34e-06`/`4.40e-06`); `loss_rel` `<= 9.49e-06`; `val_rel` `4.23e-08`–`4.67e-08` | `param_l2` `3.69e-04`–`2.39e-03`; `upd_l2` `4.18e-03` (`bf16_head_only`) / **`4.75e-02`** (`bf16_lora_temporal`); `loss_rel` `<= 2.14e-03` (`bf16_lora_variate_replay`) |
| **L4 shared NCCL**, `5M`, `N = 2` (`82a21ca`, **no tier set**) | `41 / 0` | `param_l2` 13/14 `< 1e-14` floor, max `3.00e-14`; `upd_l2` `1.02e-14` / `5.00e-14` / **`5.94e-12`** (`domain_lora_temporal_f64`; `replay_full_ft_both_f64` `4.84e-12`, `replay_full_ft_distill_only_f64` `3.46e-12`); `loss_rel` `<= 3.57e-13`; `val_rel` `2.19e-16` | `param_l2` `2.38e-08`–**`5.58e-04`** (`domain_lora_temporal_f32`; `5.61e-04` exact; med `8.29e-06`); `upd_l2` `4.42e-06` / `4.13e-03` / **`1.04e-01`** (`domain_lora_temporal_f32`; 2nd `3.27e-02` `domain_full_ft_f32`; `head_only <= 1.78e-04`); `loss_rel` `<= 7.28e-04` (`domain_full_ft_f32`); `val_rel` `1.71e-08`–`1.56e-07` | `param_l2` `7.63e-05`–`1.09e-03`; `upd_l2` `1.02e-01` (`bf16_head_only`) / **`3.48e-01`** (`bf16_full_ft_replay`; `3.51e-01` exact); `loss_rel` `<=` **`1.27e-01`** (`bf16_full_ft_replay`) |
| **L4 shared NCCL**, `5M`, `N = 4` (`82a21ca`, **no tier set**) | `41 / 0` | `param_l2` 11/14 `< 1e-14` floor, max `3.00e-14`; `upd_l2` `1.33e-14` / `1.44e-13` / **`8.73e-11`** (`replay_full_ft_both_f64`; 2nd `8.05e-11` `replay_full_ft_distill_only_f64`; next `4.33e-12` `domain_lora_temporal_f64`; all other 11 `<= 3.19e-13`); `loss_rel` `<= 2.31e-12`; `val_rel` `0.00e+00` | `param_l2` `6.72e-08`–**`2.34e-04`** (`domain_lora_temporal_f32`; med `6.29e-06`); `upd_l2` `1.28e-05` / `1.48e-03` / **`4.03e-02`** (`domain_lora_temporal_f32`; 2nd provisional `2.35e-02` `replay_full_ft_both_moved_f32`, non-moved 2nd `1.95e-02` `replay_full_ft_both_f32`); `loss_rel` `<= 1.38e-04` (`compile_default_lora_replay`); `val_rel` `1.09e-07`–`2.65e-07` | `param_l2` `9.51e-05`–`1.31e-03`; `upd_l2` `8.91e-02` (`bf16_head_only`) / **`2.77e-01`** (`bf16_full_ft_replay`); `loss_rel` `<= 3.90e-02` (`bf16_full_ft`) |
| **L4 shared NCCL**, `28M`, `N = 2` (`82a21ca`, **no tier set**) | `41 / 0` | `param_l2` 12/14 `< 1e-14` floor, max `8.45e-14`; `upd_l2` `1.56e-14` / `7.95e-14` / **`1.55e-10`** (`replay_full_ft_both_f64`; 2nd **`1.21e-10`** `replay_full_ft_distill_only_f64`; all other 12 `<= 2.11e-13`); `loss_rel` `<= 2.80e-11`; `val_rel` `2.82e-16` | `param_l2` `5.37e-08`–**`3.38e-04`** (`domain_lora_temporal_f32`; med `6.09e-06`); `upd_l2` `6.44e-06` / `2.18e-03` / **`7.33e-02`** (`domain_lora_temporal_f32`; 2nd `2.25e-02` `replay_full_ft_distill_only_f32`); `loss_rel` `<= 4.43e-04` (`compile_default_lora_replay`); `val_rel` `4.85e-07`–`5.05e-07` | `param_l2` `7.94e-05`–`1.05e-03`; `upd_l2` `3.37e-02` (`bf16_head_only`) / **`2.30e-01`** (`bf16_lora_temporal`); `loss_rel` `<= 3.05e-02` (`bf16_full_ft_replay`) |
| **L4 shared NCCL**, `28M`, `N = 4` (`82a21ca`, **no tier set**) | `41 / 0` | `param_l2` 12/14 `< 1e-14` floor, max `4.46e-14`; `upd_l2` `1.93e-14` / `1.12e-13` / **`7.68e-11`** (`replay_full_ft_distill_only_f64`; 2nd `3.89e-11` `replay_full_ft_both_f64`; all other 12 `<= 1.94e-13`); `loss_rel` `<= 4.96e-12`; `val_rel` `2.87e-16` | `param_l2` `6.57e-08`–**`3.01e-04`** (`domain_lora_temporal_f32`; med `7.32e-06`); `upd_l2` `7.83e-06` / `6.70e-03` / **`6.18e-02`** (`domain_lora_temporal_f32`; 2nd `3.33e-02` `replay_full_ft_both_f32`); `loss_rel` `<= 1.27e-04` (`compile_default_lora_replay`); `val_rel` `5.60e-08`–`9.12e-08` | `param_l2` `1.14e-04`–`1.26e-03`; `upd_l2` `1.10e-01` (`bf16_head_only`) / **`2.59e-01`** (`bf16_lora_temporal`); `loss_rel` `<= 3.06e-02` (`bf16_full_ft_replay`) |
| **Linux L4 CPU/`gloo`**, `tiny_random`, `N = 2` (`82a21ca`, `threads=1`) | `35 / 6` (`bf16` skipped) | `param_l2` 12/14 `< 1e-14` floor, max `6.75e-14`; `upd_l2` `2.21e-15` / `5.55e-15` (floor) / **`3.00e-12`** (`replay_full_ft_distill_only_f64`, `33×` below `1e-10`; 2nd `2.01e-12` `replay_full_ft_both_f64`); `loss_rel` `<= 2.75e-14`; `val_rel` `1.54e-16` | Non-id (18): `param_l2` `6.42e-08`–`4.16e-06` (med `2.81e-07`), `upd_l2` `9.73e-07` / `3.30e-06` / **`8.44e-05`** (`compile_default_lora_replay`), `loss_rel` `<= 2.05e-05`, `val_rel` `6.38e-08`–`1.07e-07`. Id-start (2, §17.2): `upd_l2` `1.90e-03`–`1.95e-03` | — |
| **Linux L4 CPU/`gloo`**, `tiny_random`, `N = 4` (`82a21ca`, `threads=1`) | `35 / 6` | `param_l2` 12/14 `< 1e-14` floor, max `1.14e-13`; `upd_l2` `1.37e-15` / `4.00e-15` (floor) / **`4.94e-12`** (`replay_full_ft_distill_only_f64`, `20×` below `1e-10`; 2nd `1.15e-12` `replay_full_ft_both_f64`); `loss_rel` `<= 3.22e-14`; `val_rel` `0.00e+00` | Non-id (18): `param_l2` `5.25e-08`–`7.79e-06` (med `3.31e-07`), `upd_l2` `7.93e-07` / `3.63e-06` / **`1.62e-04`** (`compile_default_lora_replay`), `loss_rel` `<= 1.18e-05`, `val_rel` `7.17e-08`–`1.06e-07`. Id-start (2): `upd_l2` `2.09e-03`–`2.14e-03` | — |
| **macOS arm64 CPU/`gloo`**, `tiny_random`, `N = 2` (`82a21ca`, `threads=1`) | `35 / 6` | `param_l2` `<= 1.33e-15` (all `< 1e-14` floor); `upd_l2` `2.10e-15` / `4.80e-15` (floor) / **`1.49e-14`** (`domain_lora_variate_f64`, ~`6,710×` below `1e-10`); `loss_rel` `<= 8.63e-16`; `val_rel` `3.34e-16` | Non-id (18): `param_l2` `6.27e-08`–`7.78e-06` (med `2.66e-07`), `upd_l2` `1.58e-06` / `3.14e-06` / **`1.58e-04`** (`compile_default_lora_replay`; non-compile max `1.53e-05`), `loss_rel` `<= 1.34e-05`, `val_rel` `6.38e-08`–`1.23e-07`. Id-start (2): `upd_l2` `1.95e-03`–`2.18e-03` | — |
| **macOS arm64 CPU/`gloo`**, `tiny_random`, `N = 4` (`82a21ca`, `threads=1`) | `35 / 6` | `param_l2` `<= 2.07e-15` (all `< 1e-14` floor); `upd_l2` `1.43e-15` / `5.20e-15` (floor) / **`1.38e-14`** (`replay_lora_variate_distill_only_f64`, ~`7,250×` below `1e-10`); `loss_rel` `<= 7.93e-16`; `val_rel` `3.03e-16` | Non-id (18): `param_l2` `5.10e-08`–`8.32e-06` (med `2.84e-07`), `upd_l2` `1.08e-06` / `3.97e-06` / **`1.73e-04`** (`compile_default_lora_replay`; non-compile max `1.19e-05`), `loss_rel` `<= 2.29e-05`, `val_rel` `5.03e-08`–`1.22e-07`. Id-start (2): `upd_l2` `1.89e-03`–`2.60e-03` | — |

**Feature rows (`compile_mode="default"`, `num_workers=2`, `vram_mb`) at `82a21ca`:**
- *`compile_mode="default"`:* across every distributed run (L4 NCCL `tiny`/`5M`/`28M`, Linux CPU/`gloo`, and macOS arm64 CPU/`gloo`, at both `N = 2` and `N = 4`), `compile_default_lora_variate` reported `graphs=2 recompiled=yes` and `compile_default_lora_replay` reported `graphs=5 recompiled=yes` (against `graphs=3 recompiled=yes` in `N = 2` cached one-process references and `graphs=2 recompiled=no` in `N = 4` cached references).
- *`workers_uneven_f32` `tail_s` (time from last optimiser step to `fit()` returning):* Linux L4 NCCL `0.16 s` (`tiny` `N = 2`), `0.32 s` (`tiny` `N = 4`), `0.19 s` (`5M` `N = 2`), `0.32 s` (`5M` `N = 4`), `0.20 s` (`28M` `N = 2`), `0.32 s` (`28M` `N = 4`); Linux L4 CPU/`gloo` `0.06 s` (`N = 2`), `0.08 s` (`N = 4`); macOS arm64 CPU/`gloo` `20.02 s` (`N = 2`) and `20.04 s` (`N = 4`, vs `10.02 s` one-process reference). (`step_ms` and `wall_s` are intentionally not quoted as step latency or scaling metrics; see *Known limitations* below.)
- *`vram_mb` (per-rank peak allocated MiB during `fit()`, not a sum across shared-GPU ranks):* `5M` NCCL rows used `30.5–135.6 MiB` on `f32` (`36.0–132.2 MiB` on `bf16`; `53.4–265.2 MiB` on `f64`, peaking on `replay_full_ft_distill_only_*` with live `_frozen_base`); `28M` NCCL rows used `123.4–671.5 MiB` on `f32` (`143.1–663.6 MiB` on `bf16`; `243.6–1,350.5 MiB` on `f64`).
- *Interpretation note (not a demonstrated cause):* on `5M` and `28M` NCCL, `float32` `head_only` stays small (`param_l2` `2.38e-08`–`1.61e-06`, `upd_l2` `4.42e-06`–`1.78e-04`), while `lora_temporal`/`full_ft` `upd_l2` reaches `1.04e-01` (`5M`) / `7.33e-02` (`28M`) even though `float64` on the exact same scenarios stays at `8.87e-14`–`5.94e-12`; this is consistent with §13 / spec §12's CUDA default-SDPA `float32` gradient inaccuracy on released weights. By contrast, the cause of the large `bf16` differences (`upd_l2` up to `3.48e-01` / `3.51e-01` exact and `loss_rel` up to `1.27e-01` on `5M` `N = 2` `bf16_full_ft_replay`) has not been investigated (spec §12).

#### Rank-failure timeout probe (`distributed_rank_failure_check.py`, commit `82a21ca`)

Kills rank 1 via `os._exit(1)` at step 2 mid-`fit()` and observes surviving ranks (note: `gloo` rank failure was run on macOS arm64 only; `multi_gpu_check.sh` runs Stage 9 for the primary `--backend` only, so the Linux L4 session ran NCCL rank failure only):

| Platform, backend, `N` (`82a21ca`) | Timeouts (`pg` / `wall`) | Killed rank 1 | Surviving rank(s) outcome | Elapsed (`exc_s` / `exit_s`) | Result |
|---|---|---|---|---|---|
| **macOS arm64 CPU/`gloo`**, `N = 2` | `10.0 s` / `60.0 s` | `exit:1` (`0.04 s`) | `exit:2`; raises Python `builtins.RuntimeError` (Gloo `recv` timeout `10000ms`); normal `except`/`finally` unwinds | `10.01 s` / `10.42 s` | `PASS (no hang)` (`parent_killed=none`) |
| **macOS arm64 CPU/`gloo`**, `N = 4` | `10.0 s` / `60.0 s` | `exit:1` (`0.04 s`) | Ranks 0, 2, 3 all `exit:2`; raise Python `builtins.RuntimeError` (Gloo `recv`/`send` timeout `10000ms`) | `10.00–10.01 s` / `10.58 s` | `PASS (no hang)` (`parent_killed=none`) |
| **L4 shared NCCL**, `N = 2` (`--device-map shared`) | `20.0 s` / `65.0 s` | `exit:1` (`0.18 s`) | Rank 0 killed by **`signal:SIGABRT`** (C++ watchdog: `Operation timed out after 20780 ms`, then `Waiting 15000ms for health check and flight recorder dump before aborting`, then `Fatal Python error: Aborted` in `_reduce_step_stats`); **no Python exception raised; `except`/`finally` blocks do not run** | `-` / `36.19 s` | `PASS (no hang)` (`parent_killed=none`) |
| **L4 shared NCCL**, `N = 4` (`--device-map shared`) | `20.0 s` / `65.0 s` | `exit:1` (`0.19 s`) | Ranks 0, 2, 3 all killed by **`signal:SIGABRT`** (`Operation timed out after 20555 / 20535 / 20552 ms` + `15000 ms` wait); **no Python exception raised; `except`/`finally` blocks do not run** | `-` / `35.85 / 35.90 / 35.95 s` | `PASS (no hang)` (`parent_killed=none`) |

#### Reference-mode comparison (`--reference-mode exact` vs `sketch`, `k = 8,192`, commit `82a21ca`, L4 shared NCCL `N = 2`)

Ten entries across `tiny_random` and `5M`: `ref_exact.zip` was **`114,469,429` bytes (`109.17 MiB`)** versus
`ref_sketch.zip` at **`674,359` bytes (`0.64 MiB`)** (~`170×` smaller). `loss_rel` agreed to `0.00%` on all
10 entries; `param_l2` and `upd_l2` matched within `0.00%`–`1.60%` (`tiny` `lora_temporal` has `8,192`
trainable elements `<= SKETCH_DIM` and stores `exact_delta`, so exact and sketch are identical there):

| Model & scenario (`N = 2`, L4 shared NCCL, `82a21ca`) | `param_l2` (`exact` / `sketch`) | `upd_l2` (`exact` / `sketch`) | `loss_rel` (`exact` / `sketch`) | Max relative diff (`sketch` vs `exact`) |
|---|---|---|---|---:|
| `tiny_random` `domain_full_ft_f64` | `1.03e-15` / `1.02e-15` | `3.46e-14` / `3.46e-14` | `3.99e-16` / `3.99e-16` | `0.97%` (`param_l2`, sub-`1e-14` floor) |
| `tiny_random` `domain_lora_temporal_f32` (`exact_delta`) | `2.66e-07` / `2.66e-07` | `5.58e-06` / `5.58e-06` | `2.36e-07` / `2.36e-07` | `0.00%` |
| `tiny_random` `replay_full_ft_distill_only_f64` | `1.04e-15` / `1.04e-15` | `4.61e-14` / `4.62e-14` | `3.13e-16` / `3.13e-16` | `0.22%` (`upd_l2`) |
| `tiny_random` `bf16_lora_temporal` (`exact_delta`) | `2.75e-03` / `2.75e-03` | `5.75e-02` / `5.75e-02` | `2.84e-03` / `2.84e-03` | `0.00%` |
| `tiny_random` `bf16_full_ft_replay` | `8.46e-04` / `8.46e-04` | `3.72e-02` / `3.73e-02` | `1.44e-03` / `1.44e-03` | `0.27%` (`upd_l2`) |
| `5M` `domain_full_ft_f64` | `1.25e-15` / `1.23e-15` | `2.46e-12` / `2.43e-12` | `2.65e-14` / `2.65e-14` | `1.60%` (`param_l2`, sub-`1e-14` floor; `1.22%` `upd_l2`) |
| `5M` `domain_lora_temporal_f32` | `5.61e-04` / `5.58e-04` | `1.04e-01` / `1.04e-01` | `4.66e-06` / `4.66e-06` | `0.53%` (`param_l2`) |
| `5M` `replay_full_ft_distill_only_f64` | `1.18e-15` / `1.17e-15` | `3.47e-12` / `3.46e-12` | `9.02e-14` / `9.02e-14` | `0.85%` (`param_l2`, sub-`1e-14` floor; `0.29%` `upd_l2`) |
| `5M` `bf16_lora_temporal` | `1.09e-03` / `1.09e-03` | `2.04e-01` / `2.04e-01` | `3.20e-03` / `3.20e-03` | `0.00%` |
| `5M` `bf16_full_ft_replay` | `1.12e-04` / `1.12e-04` | `3.51e-01` / `3.48e-01` | `1.27e-01` / `1.27e-01` | `0.85%` (`upd_l2`) |

**Known limitations of the tooling at `82a21ca` (review fixes pending; GPU session will be repeated after fixes):**
1. *Parameters only, no buffers:* `xr_eq` and `xr_max` inspect `model.named_parameters()` and omit `model.named_buffers()` (the model has one buffer, `transformer.rope.inv_freq`). `param_l2` and `upd_l2` inspect trainable parameters (`trainer._trainable`) only.
2. *CountSketch float64 floor:* in `sketch` mode (`k = 8,192`), `float64` bucket accumulation has a noise floor below `~1e-14` on large parameter sets (`head_only`/`full_ft` on `5M` and `28M`: 1.3×–7× inflation measured on CPU for true diffs of `1e-16`–`3e-15`; values `>= 1e-13` are unaffected, and all worst-case `float64` distillation rows at `1e-12`–`1.55e-10` sit well above it). Sparse single-coordinate sub-ULP diffs can also be absorbed to `0.00e+00` in a bucket sum.
3. *Timing columns (`step_ms`, `wall_s`):* `step_ms` on scenarios with periodic validation (`val_restore_early_stop_*`, `workers_uneven_f32`) includes validation passes and lazy validation worker spawning, and `wall_s` has no pre-`t0` barrier (absorbing pre-`fit()` setup/hashing skew across ranks). Neither is quoted here as training-step latency or scaling evidence. `tail_s` and `vram_mb` are as labelled (`vram_mb` is per-rank peak allocated MiB during `fit()`, not a sum across ranks sharing a GPU).
4. *Provisional CUDA `moved_student` parity rows:* on CUDA, `replay_full_ft_distill_only_moved_f32` and `replay_full_ft_both_moved_f32` loaded a CPU-computed `rope.inv_freq` on rank 0 and in the reference while ranks `1..N-1` kept the CUDA-computed table; whether those differ in the last bit on that GPU is unknown, so their CUDA parity numbers are provisional (cross-rank parameter equality was exact).
5. *Incomplete reference-mismatch guard:* `distributed_trainer_check.py`'s reference check hashes trainable parameters only (not frozen base weights in LoRA modes) and omits several hardcoded hyperparameters; in this `82a21ca` session both reference and distributed runs came from the same script invocation and checkpoints, so the comparison is sound, but the guard itself is incomplete.
6. *Session script timeout and resume:* `scripts/multi_gpu_check.sh`'s stage timeout does not reliably kill `torchrun`'s rank processes on a real hang (they run in their own sessions via `start_new_session=True`), and its `.done` resume markers do not record CLI flags or guard against a stale reference archive after a failed `--write-reference` re-run. Fixes for all of the above are pending.

**What this session does NOT cover:** ranks on separate GPUs; real `torch.cuda.set_device` / `LOCAL_RANK` per-rank binding; acceptance test T15 on NCCL (`tests/distributed_harness.py` still enforces CPU/`gloo` via `_ALLOWED_CPU_BACKENDS`); `scripts/probes/distributed_parity.py --device cuda` on 4 separate GPUs (both tiny and `5M`/`28M`); cross-GPU NCCL reduction determinism; separate-GPU NCCL rank-failure behaviour; Linux CPU/`gloo` rank-failure checks; `bf16` and released-weights parity tier calibration; and 1/2/4-GPU throughput, VRAM, and scaling measurements.

**Not verified anywhere (Stage 3, spec §10 & §12):** Stages 1–2 multi-process training is verified on
CPU/`gloo` on macOS arm64 and Linux x86_64 (§17.1), and on one NVIDIA L4 (Linux x86_64, Python `3.12.16`,
`82a21ca`) `scripts/cuda_check.sh` passed (§13) and the tracked Stage 3 driver and rank-failure probe ran
under `torchrun` with `N = 2` and `N = 4` NCCL ranks sharing that GPU (§17.4; to be re-run after tooling
review fixes). Still unverified: T15 on NCCL (requires adding a backend and per-rank device switch to
`tests/distributed_harness.py`, which currently restricts to CPU/`gloo` via `_ALLOWED_CPU_BACKENDS`), ranks
on separate GPUs and real multi-GPU `torch.cuda.set_device` / `LOCAL_RANK` binding (monkeypatched unit tests
only), `scripts/probes/distributed_parity.py --device cuda` on 4 separate GPUs, cross-GPU NCCL reduction
determinism, separate-GPU NCCL rank-failure behaviour, `bf16` parity tiers and the cause of the large
released-checkpoint `bf16` differences in §17.4, multi-GPU throughput/VRAM/scaling (including per-parameter
pack/unpack, 3–4 collectives per step, flat gradient buffer live through validation, and per-rank
`_frozen_base` VRAM in `full_ft` distillation), released-weights T15 bounds (with or without replay, including
cosine similarity), live concurrent `huggingface_hub==0.23.0` downloads (source reading only, §12), and
`F.linear` `requires_grad` dispatch on GPUs/CUDA.
