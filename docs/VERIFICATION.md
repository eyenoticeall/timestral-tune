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

CUDA results (§13, §17.4) come from two rented Linux x86_64 GPU setups, both running PyTorch
`2.15.0+cu132`, CUDA runtime `13.2`, and NCCL `2.30.7`:
- **Single NVIDIA L4 (`23034 MiB`) cloud VMs:** Phase 1–4 runs used Ubuntu 22.04, 4 CPU threads, and Python
  `3.12.16` (`89ac249`); the first 2026-11-23 Phase 5 run (`71514a6`, §11, §13, §17.2) used driver
  `580.178.04` and `uv`-selected Python `3.15.2` (outside the CI matrix of Python `3.10–3.13`); and the
  2026-11-23/24 tracked Stage 3 shared-GPU sessions (`82a21ca`, `a20d386`, `866816a`, the detailed primary
  record at `8e8f490`, and the final-tip repeat at `d474f82`, §13, §17.4) used rented Linux x86_64 cloud VMs
  (8 vCPU per the sessions' provisioning records) with one NVIDIA L4 (drivers `595.112.07` and `580.178.04`)
  and project Python `3.12.16` (with `scripts/multi_gpu_check.py` at `8e8f490` and `d474f82` running on
  `python3` `3.10.12` on `PATH`).
- **4× NVIDIA GeForce RTX 3060 (`12288 MiB`, Ampere compute capability `8.6`) host (session S6 at `3ac7c6d`,
  2026-11-24, §13, §17.4):** a rented GPU-marketplace Linux x86_64 Docker container host (per the session's
  provisioning record), with a dual-socket `Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz` (`80` logical CPUs),
  driver `580.178.04`, project Python `3.12.16`, and `scripts/multi_gpu_check.py` running on `python3`
  `3.12.3` on `PATH` (full host, topology, and NCCL transport inventory in §17.4; single-GPU `cuda_check.sh`
  in §13 ran on one RTX 3060 of this host). `scripts/`, `src/`, and `tests/` at `3ac7c6d` are identical to
  `d474f82`.

Linux CPU results in §11, §16, and §17 come from 8-vCPU Linux x86_64 cloud VMs (Ubuntu 22.04, including an
AMD EPYC 7B12 VM for post-fix runs) with `uv`-managed Python `3.12.16` (`torch 2.15.0+cu132`) and
`3.10.22` (`torch 2.14.1+cu130`), plus the dual-socket Xeon `E5-2673 v4` host above for S6's Linux `gloo`
comparison stages (§17.4).

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
| `load_model(..., dtype=torch.float64)` export-reload truncation on `tiny_random` (2026-11-24, commit `8e8f490`, macOS arm64 CPU [Apple M4 per the session's provisioning record], Python `3.12.13`, PyTorch `2.15.0`; **one-off check without a tracked probe**, backs pre-existing observation #12 in spec §12 and the `1e-6` float64 export-reload tolerance in `distributed_trainer_check.py`). Exact command below this table | `model.safetensors` stores `torch.float64` weights, but `load_model(..., dtype=torch.float64)` constructs `TuneTimestralModel(config)` in `float32`, calls `load_state_dict(...)` into those `float32` parameters (truncating `float64` weights to `float32`), and only then casts to `float64`. Across 6 modes (4 steps each under `float64_reference()`, relative forecast diff `max(|reloaded - live|) / max(|live|, 1e-12)`): `head_only` `1.52e-08` (`param_max_abs 7.45e-09`), `lora_variate` `5.38e-09` (`1.33e-08`), `lora_temporal` `2.76e-09` (`7.45e-09`), `full_ft` `1.49e-08` (`5.93e-08`), `lora_variate+replay` `3.36e-09` (`1.41e-08`), `full_ft+replay` `1.55e-08` (`5.96e-08`). Casting the model to `torch.float64` *before* `load_state_dict` dropped the same forecast differences to `1.67e-16`–`4.54e-16` |

Exact command for the one-off `load_model(..., dtype=torch.float64)` check above (no tracked probe):

```bash
env -u PYTHONUNBUFFERED TMPDIR=/tmp uv run python -c '
import copy, sys, tempfile
from pathlib import Path
sys.path.insert(0, "scripts/probes")
import torch
from safetensors.torch import load_file
from _common import float64_reference, tiny_random_forecaster
from _dist_common import make_synthetic_dataset
from timestral_tune import LoraConfig, ReplayConfig, TimestralTrainer, TrainingConfig
from timestral_tune.data import DHFDataCollator
from timestral_tune.models import denormalise_quantiles, load_model, training_forward

base = tiny_random_forecaster(seed=0).model.double()
dom_ds = make_synthetic_dataset(8, covariates=True, seed=1001)
rep_ds = make_synthetic_dataset(8, covariates=False, seed=1003)
collator = DHFDataCollator.for_model(base.config, dom_ds.context_length)
batch = collator([dom_ds[0]])
lora_v = LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("q", "k", "v", "o"))
lora_t = LoraConfig(r=4, lora_alpha=8, axes=("time",), targets=("q", "k", "v", "o"))
rcfg = ReplayConfig(replay_fraction=0.25, distill_weight=0.5, batch_size=2, cache_base_outputs=False)
cases = [
    ("head_only", "head_only", None, None),
    ("lora_variate", "lora_variate", lora_v, None),
    ("lora_temporal", "lora_temporal", lora_t, None),
    ("full_ft", "full_ft", None, None),
    ("lora_variate+replay", "lora_variate", lora_v, rcfg),
    ("full_ft+replay", "full_ft", None, rcfg),
]
with float64_reference():
    for label, mode, lora, rc in cases:
        cfg = TrainingConfig(model_name_or_path="local-f64-reload-check", mode=mode, device="cpu", batch_size=4, epochs=2, max_steps=4, learning_rate=1e-3 if mode == "full_ft" else 1e-2, shuffle=False, sign_flip=False, cache_frozen_features=False, log_every=100, seed=107)
        tr = TimestralTrainer(cfg, dom_ds, lora_config=lora, replay_config=rc, replay_dataset=rep_ds if rc else None, model=copy.deepcopy(base))
        tr.fit()
        with tempfile.TemporaryDirectory() as td:
            out = tr.export(Path(td) / "ckpt")
            saved = load_file(str(out / "model.safetensors"), device="cpu")
            reloaded = load_model(out, device="cpu", dtype=torch.float64, local_files_only=True)
            tr.model.eval()
            with torch.no_grad():
                live_o = training_forward(tr.model, batch, validate=False)
                live_q = denormalise_quantiles(tr.model, live_o.quantiles, live_o.anchor_loc, live_o.anchor_scale)
                rel_q = reloaded.predict_block(batch.context, batch.context_mask, batch.group_ids, batch.horizon_steps // base.config.patch_size, batch.role_ids, batch.known_future, batch.known_future_mask)
            scale = max(float(live_q.double().abs().max().item()), 1e-12)
            f_rel = float((rel_q.double() - live_q.double()).abs().max().item()) / scale
            p_abs = max(float((dict(reloaded.named_parameters())[k].double() - saved[k].double()).abs().max().item()) for k in saved)
            print(f"{label}: param_max_abs={p_abs:.2e} forecast_rel={f_rel:.2e}")
'
```

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

Phase 5 branch re-runs of `bash scripts/cuda_check.sh` on `2026-11-23`/`24` (Linux x86_64, `torch
2.15.0+cu132`, CUDA runtime `13.2`, NCCL `2.30.7`): three on one NVIDIA L4 (`23034 MiB`) — first at `71514a6`
(`40a580a` differs in comments only; driver `580.178.04`, `uv`-selected `CPython 3.15.2`, outside the CI
matrix of Python `3.10–3.13`), then at `82a21ca` on an 8-vCPU cloud VM (driver `595.112.07`, `Python 3.12.16`,
closing the Python `3.12` gap on this branch), and at `d474f82` (`2026-11-24`, driver `580.178.04`,
`uv`-managed `Python 3.12.16`, 8 vCPU per the session's provisioning record, including all 105 tests in
`tests/test_multi_gpu_check.py`) — plus a fourth run on **one NVIDIA GeForce RTX 3060** (on the 4-GPU host,
`12288 MiB`, Ampere `8.6`; single-process tests and probes run on `cuda:0`) at `3ac7c6d` (`2026-11-24`, driver
`580.178.04`, `uv`-managed `Python 3.12.16`, `80` logical Xeon CPUs; `scripts/`, `src/`, and `tests/`
identical to `d474f82`):

| Check (`torch 2.15.0+cu132`, CUDA `13.2`) | `71514a6` (one L4, driver `580.178.04`, Python `3.15.2`) | `82a21ca` (one L4, driver `595.112.07`, Python `3.12.16`) | `d474f82` (one L4, driver `580.178.04`, Python `3.12.16`) | `3ac7c6d` (one RTX 3060 on 4-GPU host, driver `580.178.04`, Python `3.12.16`) |
|---|---|---|---|---|
| T14 and all other device-parametrised tests | `266 passed, 3 skipped` (`188.12 s`; 3 Apple GPU tests skipped) | `266 passed, 3 skipped` (`128.61 s`; 3 Apple GPU tests skipped) | `266 passed, 3 skipped` (`173.84 s`; 3 Apple GPU tests skipped) | `266 passed, 3 skipped` (`153.71 s`; 3 Apple GPU tests skipped) |
| Whole suite (`uv run pytest -q`, incl. T1–T14, network tests, 142 CPU/`gloo` multi-process items, and at `d474f82`/`3ac7c6d` 105 `tests/test_multi_gpu_check.py` items) | `735 passed, 4 skipped` (`603.65 s`; 3 Apple GPU tests + 1 optional `lacuna` skipped) | `735 passed, 4 skipped` (`433.05 s`; 3 Apple GPU tests + 1 optional `lacuna` skipped) | `840 passed, 4 skipped` (`419.94 s`; 3 Apple GPU tests + 1 optional `lacuna` skipped) | `840 passed, 4 skipped` (`620.05 s`; 3 Apple GPU tests + 1 optional `lacuna` skipped) |
| `head_only` gradients on CUDA against the reference | At most `1.68e-6` (bound `1e-5`) | At most `1.68e-06` (bound `1e-5`) | At most `1.68e-06` (bound `1e-5`) | At most `1.95e-06` (bound `1e-5`) |
| `device_parity.py --device cuda` (`5M` / `28M`) | `5M` fwd `1.19e-7`–`1.49e-5` (`bf16` `1.49e-2`, `fp16` `1.96e-3`), grad `1.15e-2`; `28M` fwd `1.04e-7`–`2.20e-6` (`bf16` `5.74e-2`, `fp16` `5.92e-3`), grad `2.24e-3` | `5M` fwd `1.08e-07`–`3.19e-05` (`bf16` `1.49e-02`, `fp16` `1.96e-03`), grad `1.15e-02`; `28M` fwd `1.13e-07`–`2.53e-06` (`bf16` `5.74e-02`, `fp16` `5.92e-03`), grad `2.24e-03` | `5M` fwd `1.19e-07`–`1.49e-05` (`bf16` `1.49e-02`, `fp16` `1.96e-03`), grad `1.15e-02`; `28M` fwd `1.04e-07`–`2.20e-06` (`bf16` `5.74e-02`, `fp16` `5.92e-03`), grad `2.24e-03` | `5M` fwd `1.11e-07`–`3.98e-05` (`bf16` `1.44e-02`, `fp16` `1.96e-03`), grad `1.15e-02`; `28M` fwd `1.05e-07`–`2.07e-06` (`bf16` `6.02e-02`, `fp16` `6.77e-03`), grad `1.75e-03` |
| `gradient_accuracy.py --device cuda` (`default` / `math` / `explicit`) | Tiny `1.6e-7`–`3.1e-7`; `5M` `1.7e-3`–`1.2e-2` / `3.6e-6`–`5.4e-5` / `6.0e-6`–`2.5e-5`; `28M` `1.4e-3`–`2.8e-3` / `4.8e-6`–`7.8e-5` / `3.5e-6`–`7.0e-5` | Tiny `1.6e-7`–`3.1e-7`; `5M` `1.7e-3`–`1.2e-2` / `3.6e-6`–`5.4e-5` / `6.0e-6`–`2.5e-5`; `28M` `1.4e-3`–`2.8e-3` / `4.8e-6`–`7.8e-5` / `3.5e-6`–`7.0e-5` | Tiny `1.6e-7`–`3.1e-7`; `5M` `1.7e-3`–`1.2e-2` / `3.6e-6`–`5.4e-5` / `6.0e-6`–`2.5e-5`; `28M` `1.4e-3`–`2.8e-3` / `4.8e-6`–`7.8e-5` / `3.5e-6`–`7.0e-5` | Tiny `1.8e-7`–`3.0e-7`; `5M` `2.6e-3`–`1.3e-2` / `7.7e-6`–`5.7e-5` / `5.3e-6`–`6.1e-5`; `28M` `1.7e-3`–`4.4e-3` / `5.4e-6`–`1.4e-4` / `3.3e-6`–`6.2e-5` |
| `step_timing.py --device cuda` (`full_ft`, `fp32`, eager, batch 32, ctx 1,024; `cpu threads 8` on L4, **`cpu threads 80`** on RTX 3060 host) | `5M` uni `52 ms` / `0.27 GiB` (`3.7×` vs CPU `190 ms`), cov5 `55 ms` / `1.10 GiB` (`14.3×` vs `786 ms`); `28M` uni `68 ms` / `0.98 GiB` (`9.7×` vs `654 ms`), cov5 `209 ms` / `3.39 GiB` (`14.7×` vs `3,081 ms`) | `5M` uni `34 ms` / `0.27 GiB` (`5.1×` vs CPU `171 ms`), cov5 `51 ms` / `1.10 GiB` (`16.2×` vs `834 ms`); `28M` uni `49 ms` / `0.98 GiB` (`13.8×` vs `678 ms`), cov5 `194 ms` / `3.39 GiB` (`19.2×` vs `3,715 ms`) | `5M` uni `48 ms` / `0.27 GiB` (`4.0×` vs CPU `192 ms`), cov5 `54 ms` / `1.10 GiB` (`14.8×` vs `795 ms`); `28M` uni `68 ms` / `0.98 GiB` (`9.4×` vs `636 ms`), cov5 `200 ms` / `3.39 GiB` (`15.9×` vs `3,187 ms`) | (`cpu threads 80`) `5M` uni `56 ms` / `0.27 GiB` (`4.7×` vs CPU `261 ms`), cov5 `83 ms` / `1.10 GiB` (`6.5×` vs `546 ms`); `28M` uni `70 ms` / `0.98 GiB` (`6.3×` vs `440 ms`), cov5 `272 ms` / `3.39 GiB` (`6.6×` vs `1,799 ms`) |

*(Note on the `3ac7c6d` RTX 3060 column: `step_timing.py` ran with `cpu threads 80` on the dual-socket Xeon
host versus `cpu threads 8` on the L4 VMs, so the `CPU/CUDA` speed-up ratios are not comparable across GPUs;
and TorchInductor emitted `Not enough SMs to use max_autotune_gemm mode` on RTX 3060 during compiled tests.)*

The earlier `71514a6` one-off CUDA checks (one-process bit-identity vs `57a781c`, single-rank NCCL
collective-dtype smoke, and 2-/4-rank shared-GPU trainer trial) are kept as history in §11; the tracked
Stage 3 driver and rank-failure sessions on rented single-L4 VMs (S1–S5, `82a21ca`–`d474f82`, plus the Linux
non-network test runs) and the first separate-GPU session on 4× RTX 3060 (S6 at `3ac7c6d`) are recorded in
§17.4.

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
| Full network-free suite (`-m "not network"`) | **`511 passed, 15 skipped, 213 deselected`** (`265.63 s`, `0` failed; skips: 1 optional `lacuna`, 14 CUDA `bf16`; superseded at `d474f82` by `616 passed, 15 skipped, 213 deselected` after adding `tests/test_multi_gpu_check.py`, §17.4) | Not run in full here (four distributed files below) |
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

### 17.4 Stage 3 measuring driver & rank-failure sessions (`scripts/multi_gpu_check.sh` / `multi_gpu_check.py`: shared-GPU L4 S1–S5 through `d474f82`; separate-GPU 4× RTX 3060 S6 at `3ac7c6d`)

Backs the `[V]` claims in spec §5.6, §7.6, §9, D28, D35, and §12 for the tracked Stage 3 scripts
(`scripts/probes/distributed_trainer_check.py`, `scripts/probes/distributed_rank_failure_check.py`,
`scripts/probes/_dist_common.py`, and `scripts/multi_gpu_check.sh` wrapping `scripts/multi_gpu_check.py`).
This subsection records two sets of runs:
- **Single-L4 shared-GPU sessions (S1–S5, `82a21ca`–`d474f82`):** **session S4 at `8e8f490` (2026-11-24)**
  is the detailed single-L4 shared-GPU primary record (after three probe/driver review-fix rounds
  `713aaac`–`1963050` and the rewrite into stdlib-only `scripts/multi_gpu_check.py`, `bd24e47`–`8e8f490`;
  `src/` did not change), followed by the compact **final-tip repeat S5 at `d474f82`** (after the pure-BSD
  `ps` fix `335cf7d` and pre-stage `--log-dir` hardening `d474f82`) and earlier sessions S1–S3 (`82a21ca`,
  `a20d386`, `866816a`) as cross-session reproducibility evidence. All five ran with **one GPU shared by all
  ranks** (`NCCL_MULTI_RANK_GPU_ENABLE=1`, `--device-map shared`).
- **Separate-GPU PCIe/SHM session (S6, `3ac7c6d`, 2026-11-24):** recorded in full in its own subsection
  below, running on **4× NVIDIA GeForce RTX 3060** with one rank per GPU (`--device-map per-rank`,
  `NCCL_MULTI_RANK_GPU_ENABLE` unset, PCIe gen 3 ×16, NCCL `via SHM/direct`; `scripts/`, `src/`, and `tests/`
  at `3ac7c6d` are identical to `d474f82`). S6 satisfies the separate-GPU PCIe/SHM session sub-item in spec
  §10 while leaving T15 on NCCL, NVLink / GPU P2P + cross-GPU determinism, tier calibration, and multi-GPU
  scaling open.

`scripts/probes/distributed_trainer_check.py` is an ungraded measuring driver (41 scenarios across groups
`g1`–`g8`; 6 steps default, 10 for `val_restore_early_stop_*`, 4 for `export_multirank_f32`; `B = 2`,
`b = 2`, `seed = 0`), separate from acceptance test T15 (`tests/acceptance/test_distributed.py`, CPU/`gloo`
only). Reproduces and supersedes the earlier `71514a6` one-off shared-GPU NCCL trial in §11.

**Primary environments and commands (`8e8f490`, `2026-11-24`):**

- **Single-L4 shared-GPU NCCL + Linux CPU/`gloo` session (S4, `8e8f490`):** one rented NVIDIA L4
  (`23034 MiB`, driver `580.178.04`), Linux x86_64 (8 vCPU per the session's provisioning record), project
  environment Python `3.12.16`, `torch 2.15.0+cu132`, CUDA runtime `13.2`, NCCL `2.30.7`, `python_h=True`,
  `c_compiler=True`; `scripts/multi_gpu_check.py` ran on `python3` `3.10.12` on `PATH` (interpreter path not
  recorded); `tree_state_method=git` (`2d896c0c4b41`).
  ```bash
  bash scripts/multi_gpu_check.sh --shared-gpu --python 3.12 --log-dir mgc_logs
  ```
  Exports `NCCL_MULTI_RANK_GPU_ENABLE=1`, `TMPDIR=/tmp`, `PYTHONFAULTHANDLER=1`, and passes
  `--device-map shared` (all ranks share `cuda:0`; `threads=1` in `torchrun` and CPU/`gloo` stages,
  `threads=4` in CUDA reference stages). Wall time `47 min 51 s` (`~48 min`), exit code `0`.
  **17 of 17 runnable stages exited `0` (18 listed, `02_prototype_parity_cuda` `SKIPPED (shared GPU)`):**
  `01_env`, `03_ref_nccl_tiny_n{2,4}` / `04_nccl_tiny_n{2,4}`,
  `05_ref_nccl_released_n{2,4}` / `06_nccl_released_n{2,4}`, `07_ref_gloo_tiny_n{2,4}` /
  `08_gloo_tiny_n{2,4}`, `09_rank_failure_gloo_n{2,4}`, and `10_rank_failure_nccl_n{2,4}`. Zero leftover
  processes and 0 GPU compute processes afterwards. Reference archives (`--reference-mode auto`,
  `auto(f64<=1000000,k=8192)`, schema v5): `tiny_random` CUDA `25.44 MiB` (40 entries per `N`),
  `5M`+`28M` CUDA `37.47 MiB` (80 entries per `N`), `tiny_random` CPU `25.11 MiB` (34 entries per `N`).
  **316 executed rows** in total (`246` NCCL = `41+41` `tiny_random`, `82+82` `5M`+`28M` at `N = 2, 4`;
  `70` Linux CPU/`gloo` = `35+35` `tiny_random` at `N = 2, 4` with 6 `bf16` rows skipped per `N`).
- **macOS arm64 CPU/`gloo` session (`8e8f490`):** Darwin arm64 (Apple M4 per the session's provisioning
  record), project Python `3.12.13`, `torch 2.15.0`, `python_h=True`, `c_compiler=True`, `threads=1`,
  `tree_state_method=git` (`2d896c0c4b41`). Per the session's provisioning record, the wrapper took its first
  branch (`python3` `3.13.x` on `PATH`) here as on Linux (`3.10.12`); **no real run in this evidence set
  exercised the wrapper's `uv python find` or `uv run` fallback branches** (covered only by the stubbed unit
  test `test_wrapper_interpreter_fallback_order_and_opaque_arg_passthrough`).
  ```bash
  bash scripts/multi_gpu_check.sh --backend gloo --ranks 2 4 --pg-timeout 10 --log-dir <dir>
  ```
  Wall time `156 s`, exit code `0`; **7 of 7 runnable stages exited `0` (8 listed, `02_prototype_parity_cuda`
  `SKIPPED (gloo backend)`)**; reference archives `25.11 MiB` (34 entries per `N`); **70 executed rows**
  (`35+35`), zero hangs.

**What `--reference-mode auto` and each driver column measure (as built at `8e8f490` / `d474f82`):**

- *`--reference-mode auto` (default; `auto(f64<=1000000,k=8192)`, schema v5):* stores exact `float64`
  `delta = final - init` (`exact_delta`) for every `float64` scenario with `numel <= 1,000,000` (all 14
  `tiny_random` `float64` scenarios, all `5M` LoRA + `head_only`, and all `28M` LoRA) and for any
  `float32`/`bf16` scenario with `numel <= 8,192`; uses a deterministic `k = 8,192` `float64` CountSketch
  (`64 KiB`/scenario) for larger vectors (`5M`/`28M` `full_ft` and `28M` `head_only` in `float64`, plus
  large `float32`/`bf16` sets). On sketch entries, differences within `10×` of the expected `float64`
  bucket-accumulation floor `0.25 * sqrt(n/k) * eps_f64` (~`8.4e-16` for `28M head_only` with `1,880,064`
  trainable parameters, ~`1.4e-15` for `5M full_ft`, ~`3.2e-15` for `28M full_ft`) are flagged
  `sketch_bound(...)` and hash mismatches with zero sketch diff are flagged `sketch_underflow(sha_differs)`.
  Every-rank pre-flight validation (`_read_reference_metadata` and `_validate_reference_spec`) verifies zip
  CRCs, `spec.json` / `entries.json`, exact tensor member byte sizes, base model digests (parameters +
  buffers), per-scenario resolved `TrainingConfig`/`LoraConfig`/`ReplayConfig` + `moved_student` warmup
  digests, `torch`/`CUDA` versions, `git_head`, and `script_sha256` (over `distributed_trainer_check.py`,
  `_dist_common.py`, and `_common.py`). Per scenario on rank 0, `_compare_to_reference` additionally verifies
  `init_model_sha256`, `init_trainable_sha256`, `frozen_base_sha256`, and dataset fingerprints (`dom_fp`,
  `val_fp`, `rep_fp`), broadcasting any mismatch flag so all ranks abort cleanly.
- *Columns:*
  - `xr_eq`: exact 256-bit SHA-256 unanimity across all ranks over **all `model.named_parameters()` and all
    `model.named_buffers()`** (`TrainingResult` 10-field unanimity is checked separately; mismatch appends
    `RESULT_MISMATCH` and fails deterministic runs).
  - `xr_max`: `max_{r >= 1} ||state_r - state_0||_inf` in `float64` over **all parameters and all buffers**
    (NaN-propagating).
  - `param_l2`, `upd_l2`: relative L2 of `final_trainable` and `delta = final_trainable - init_trainable`
    over `trainer._trainable` only against the reference's exact `float64` norms (bitwise-identical
    non-finite vectors report `nan` + `NONFINITE_PARAMS`, never `0.0`).
  - `loss_rel`, `val_rel`: NaN-propagating maximum relative step-loss and validation-loss difference.
  - `wall_s`: `max_r` `fit()` wall time starting after a 1-scalar `all_reduce` barrier + CUDA sync and
    ending after a final CUDA sync.
  - `step_ms`: `max_r` mean full per-step wall time over steps `3..K` (excluding steps 1–2 startup/compile),
    measured symmetrically in reference and distributed modes from `strategy.on_step_end(k-1)` to
    `strategy.on_step_end(k)` (with CUDA sync) minus any `strategy.evaluate_val()` + `strategy.on_evaluate()`
    time between them.
  - `val_s` (in `notes` when validating): `max_r` cumulative `evaluate_val()` + `on_evaluate()` wall time
    (including lazy validation worker spawn).
  - `tail_s`: `max_r` wall time from the final `on_step_end(K)` (with CUDA sync) to `fit()` returning.
  - `vram_mib`: `max_r` per-rank peak `torch.cuda.max_memory_allocated()` during `fit()` in MiB (`2**20`
    bytes, CUDA only; not a sum across ranks sharing a GPU).
  - Compiled graph note: `graphs=s1:A->end:B(+C,bumps_post_s1=D)` (`A` Dynamo `unique_graphs` after step 1,
    `B` after `fit()`, `+C = B - A`, `D` distinct post-step-1 steps on which `unique_graphs` increased).
  - `export_multirank_f32`: validates shared-directory export (`tune_metadata.json`, `strict=True` reload via
    `load_model`, forecast parity within `1e-4` in `float32` / `1e-6` in `float64` per §11) and proves
    non-zero ranks perform zero disk writes against a private non-existent directory.

#### Driver parity summary (`distributed_trainer_check.py`, commit `8e8f490`, `--reference-mode auto`)

Every executed row across all runs below reported **`xr_eq = yes` and `xr_max = 0.00e+00`** over all model
parameters and buffers (`246/246` L4 shared NCCL rows across `tiny_random`, `5M`, `28M` × `N = 2, 4`;
`70/70` Linux CPU/`gloo` rows; `70/70` macOS arm64 CPU/`gloo` rows) and **zero warning/error flags**.
`val_restore_early_stop_*` returned `early=True restored=True best_step=2`, `export_multirank_f32` returned
`export=ok(2/2)` / `export=ok(4/4)`, and `stochastic_lora_variate_f32` matched bitwise across ranks in every
table.

| Platform, backend, model, `N` (`8e8f490`) | Executed / skipped | `float64` (14 rows): `param_l2` range (med); `upd_l2` min / med / max (worst scenario); `loss_rel` max; `val_rel` | `float32` (20 rows; CPU splits 18 non-id + 2 id-start): `param_l2` range (med); `upd_l2` min / med / max (worst scenario); `loss_rel` max; `val_rel` | `bf16` (6 rows, CUDA only; **no tier set**): `param_l2` range; `upd_l2` min / max (worst); `loss_rel` max |
|---|---|---|---|---|
| **L4 shared NCCL**, `tiny_random`, `N = 2` | `41 / 0` | `param_l2` `1.04e-16`–`1.86e-15` (med `7.33e-16`, exact); `upd_l2` `1.93e-15` / `4.70e-15` / **`4.61e-14`** (`replay_full_ft_distill_only_f64`, ~`2,170×` below T15 `1e-10`); `loss_rel` `<= 1.38e-15`; `val_rel` `2.23e-16` | `param_l2` `5.83e-08`–`3.47e-06` (med `1.83e-07`); `upd_l2` `1.96e-06` / `3.45e-06` / **`7.04e-05`** (`compile_default_lora_replay`; non-compile max `5.62e-06` `val_restore_early_stop_f32`; moved `2.10e-06`/`3.17e-06`); `loss_rel` `<= 4.69e-06` (`compile_default_lora_replay`); `val_rel` `3.94e-08`–`4.22e-08` | `param_l2` `3.96e-04`–`2.95e-03` (`bf16_head_only`); `upd_l2` `9.26e-03` (`bf16_head_only`) / **`5.75e-02`** (`bf16_lora_temporal`); `loss_rel` `<= 4.77e-03` (`bf16_lora_variate_replay`) |
| **L4 shared NCCL**, `tiny_random`, `N = 4` | `41 / 0` | `param_l2` `7.58e-17`–**`1.80e-13`** (med `2.67e-16`, exact; max `replay_full_ft_distill_only_f64`); `upd_l2` `1.54e-15` / `4.52e-15` / **`7.79e-12`** (`replay_full_ft_distill_only_f64`, `12.8×` below T15 `1e-10`; 2nd `1.76e-12` `replay_full_ft_both_f64`; all other 12 `<= 9.77e-15`); `loss_rel` `<= 3.53e-14`; `val_rel` `0.00e+00` | `param_l2` `6.30e-08`–`3.61e-06` (med `3.19e-07`); `upd_l2` `9.78e-07` / `3.82e-06` / **`7.52e-05`** (`compile_default_lora_replay`; non-compile max `3.03e-05` `replay_lora_variate_distill_only_f32`; moved `8.81e-06`/`1.28e-05`); `loss_rel` `<= 5.29e-06` (`compile_default_lora_replay`); `val_rel` `4.23e-08`–`4.67e-08` | `param_l2` `3.69e-04`–`2.39e-03`; `upd_l2` `4.18e-03` (`bf16_head_only`) / **`4.75e-02`** (`bf16_lora_temporal`); `loss_rel` `<= 2.14e-03` (`bf16_lora_variate_replay`) |
| **L4 shared NCCL**, `5M`, `N = 2` (**no tier set**) | `41 / 0` | `param_l2` `5.42e-17`–`3.02e-14` (med `2.32e-16`); `upd_l2` `1.01e-14` / `5.01e-14` / **`5.97e-12`** (`domain_lora_temporal_f64`; `replay_full_ft_both_f64` `4.84e-12`, `replay_full_ft_distill_only_f64` `3.46e-12`); `loss_rel` `<= 3.57e-13`; `val_rel` `2.19e-16` | `param_l2` `2.38e-08`–**`5.58e-04`** (`domain_lora_temporal_f32`; med `8.22e-06`); `upd_l2` `4.42e-06` / `4.13e-03` / **`1.04e-01`** (`domain_lora_temporal_f32`; 2nd `3.27e-02` `domain_full_ft_f32`; `head_only <= 2.00e-05`; moved `1.10e-02`/`1.28e-02`); `loss_rel` `<=` **`7.28e-04`** (`domain_full_ft_f32`); `val_rel` `1.71e-08`–`1.56e-07` | `param_l2` `7.63e-05`–`1.09e-03`; `upd_l2` `1.02e-01` (`bf16_head_only`) / **`3.48e-01`** (`bf16_full_ft_replay`); `loss_rel` `<=` **`1.27e-01`** (`bf16_full_ft_replay`) |
| **L4 shared NCCL**, `5M`, `N = 4` (**no tier set**) | `41 / 0` | `param_l2` `6.97e-17`–`3.00e-14` (med `3.44e-16`); `upd_l2` `1.33e-14` / `1.43e-13` / **`8.73e-11`** (`replay_full_ft_both_f64`; 2nd `8.05e-11` `replay_full_ft_distill_only_f64`; next `4.35e-12` `domain_lora_temporal_f64`); `loss_rel` `<= 2.31e-12`; `val_rel` `0.00e+00` | `param_l2` `6.72e-08`–**`2.34e-04`** (`domain_lora_temporal_f32`; med `6.10e-06`); `upd_l2` `1.28e-05` / `1.48e-03` / **`4.03e-02`** (`domain_lora_temporal_f32`; 2nd `1.95e-02` `replay_full_ft_both_f32`; `head_only <= 1.27e-04`; moved `1.39e-02`/`1.44e-02`); `loss_rel` `<= 1.08e-04` (`replay_full_ft_both_moved_f32`); `val_rel` `1.09e-07`–`2.65e-07` | `param_l2` `9.51e-05`–`1.31e-03`; `upd_l2` `8.91e-02` (`bf16_head_only`) / **`2.77e-01`** (`bf16_full_ft_replay`); `loss_rel` `<= 3.90e-02` (`bf16_full_ft`) |
| **L4 shared NCCL**, `28M`, `N = 2` (**no tier set**) | `41 / 0` | `param_l2` `5.47e-17`–`8.45e-14` (med `4.40e-16`); `upd_l2` `1.56e-14` / `7.95e-14` / **`1.55e-10`** (`replay_full_ft_both_f64`; 2nd **`1.21e-10`** `replay_full_ft_distill_only_f64`; all other 12 `<= 2.11e-13`); `loss_rel` `<=` **`2.80e-11`**; `val_rel` `2.82e-16` | `param_l2` `5.37e-08`–**`3.38e-04`** (`domain_lora_temporal_f32`; med `7.02e-06`); `upd_l2` `6.44e-06` / `2.37e-03` / **`7.33e-02`** (`domain_lora_temporal_f32`; 2nd `2.25e-02` `replay_full_ft_distill_only_f32`; `head_only <= 1.78e-04`; moved `9.51e-03`/`1.52e-02`); `loss_rel` `<= 4.50e-04` (`compile_default_lora_replay`); `val_rel` `4.85e-07`–`5.05e-07` | `param_l2` `7.94e-05`–`1.05e-03`; `upd_l2` `3.37e-02` (`bf16_head_only`) / **`2.30e-01`** (`bf16_lora_temporal`); `loss_rel` `<= 3.05e-02` (`bf16_full_ft_replay`) |
| **L4 shared NCCL**, `28M`, `N = 4` (**no tier set**) | `41 / 0` | `param_l2` `4.79e-17`–`4.46e-14` (med `5.56e-16`); `upd_l2` `1.93e-14` / `1.12e-13` / **`7.68e-11`** (`replay_full_ft_distill_only_f64`; 2nd `3.89e-11` `replay_full_ft_both_f64`; all other 12 `<= 1.95e-13`); `loss_rel` `<= 4.96e-12`; `val_rel` `2.87e-16` | `param_l2` `6.57e-08`–**`3.01e-04`** (`domain_lora_temporal_f32`; med `7.32e-06`); `upd_l2` `7.83e-06` / `6.70e-03` / **`6.18e-02`** (`domain_lora_temporal_f32`; 2nd `3.33e-02` `replay_full_ft_both_f32`; `head_only <= 3.50e-05`; moved `1.39e-02`/`1.88e-02`); `loss_rel` `<= 1.27e-04` (`compile_default_lora_replay`); `val_rel` `5.60e-08`–`9.12e-08` | `param_l2` `1.14e-04`–`1.26e-03`; `upd_l2` `1.10e-01` (`bf16_head_only`) / **`2.59e-01`** (`bf16_lora_temporal`); `loss_rel` `<= 3.06e-02` (`bf16_full_ft_replay`) |
| **Linux L4 CPU/`gloo`**, `tiny_random`, `N = 2` (`threads=1`) | `35 / 6` (`bf16` skipped) | `param_l2` `8.00e-17`–`1.13e-13` (med `4.88e-16`, exact); `upd_l2` `1.21e-15` / `5.07e-15` / **`5.03e-12`** (`replay_full_ft_distill_only_f64`, `19.9×` below `1e-10`; 2nd `1.57e-12` `replay_full_ft_both_f64`); `loss_rel` `<= 7.62e-14`; `val_rel` `2.33e-16` | Non-id (18): `param_l2` `1.40e-08`–`5.75e-06` (med `1.23e-07`), `upd_l2` `1.09e-07` / `2.50e-06` / **`1.17e-04`** (`compile_default_lora_replay`; non-compile max `1.01e-05`), `loss_rel` `<= 1.33e-05`, `val_rel` `5.90e-08`–`1.09e-07`. Id-start (2, §17.2): `param_l2` `3.04e-05`–`4.28e-05`, `upd_l2` `1.34e-03`–`1.90e-03` | — |
| **Linux L4 CPU/`gloo`**, `tiny_random`, `N = 4` (`threads=1`) | `35 / 6` | `param_l2` `6.48e-17`–`9.74e-14` (med `3.35e-16`, exact); `upd_l2` `1.11e-15` / `4.72e-15` / **`4.21e-12`** (`replay_full_ft_distill_only_f64`, `23.8×` below `1e-10`; 2nd `1.25e-12` `replay_full_ft_both_f64`); `loss_rel` `<= 2.91e-14`; `val_rel` `1.15e-16` | Non-id (18): `param_l2` `4.53e-08`–`5.32e-06` (med `2.29e-07`), `upd_l2` `7.35e-07` / `3.02e-06` / **`1.11e-04`** (`compile_default_lora_replay`; non-compile max `1.67e-05`), `loss_rel` `<= 1.12e-05`, `val_rel` `9.24e-08`–`1.42e-07`. Id-start (2, §17.2): `param_l2` `4.04e-05`–`4.80e-05`, `upd_l2` `1.73e-03`–`2.08e-03` | — |
| **macOS arm64 CPU/`gloo`**, `tiny_random`, `N = 2` (`threads=1`) | `35 / 6` | `param_l2` `1.12e-16`–`1.34e-15` (med `3.61e-16`, exact); `upd_l2` `2.10e-15` / `4.82e-15` / **`1.49e-14`** (`domain_lora_variate_f64`, ~`6,710×` below `1e-10`); `loss_rel` `<= 8.63e-16`; `val_rel` `3.34e-16` | Non-id (18): `param_l2` `6.27e-08`–`7.78e-06` (med `2.66e-07`), `upd_l2` `1.58e-06` / `3.14e-06` / **`1.58e-04`** (`compile_default_lora_replay`; non-compile max `1.53e-05`), `loss_rel` `<= 1.34e-05`, `val_rel` `6.38e-08`–`1.23e-07`. Id-start (2, §17.2): `param_l2` `4.42e-05`–`4.90e-05`, `upd_l2` `1.95e-03`–`2.18e-03` | — |
| **macOS arm64 CPU/`gloo`**, `tiny_random`, `N = 4` (`threads=1`) | `35 / 6` | `param_l2` `8.36e-17`–`2.07e-15` (med `2.59e-16`, exact); `upd_l2` `1.42e-15` / `5.20e-15` / **`1.38e-14`** (`replay_lora_variate_distill_only_f64`, ~`7,250×` below `1e-10`); `loss_rel` `<= 7.93e-16`; `val_rel` `3.03e-16` | Non-id (18): `param_l2` `5.10e-08`–`8.32e-06` (med `2.84e-07`), `upd_l2` `1.08e-06` / `3.97e-06` / **`1.73e-04`** (`compile_default_lora_replay`; non-compile max `1.19e-05`), `loss_rel` `<= 2.29e-05`, `val_rel` `5.03e-08`–`1.22e-07`. Id-start (2, §17.2): `param_l2` `4.39e-05`–`6.02e-05`, `upd_l2` `1.89e-03`–`2.60e-03` | — |

**Worst multi-process vs one-process differences across all 316 L4 rows (S4, `8e8f490`; identical printed worst-case `param_l2`, `upd_l2`, and `loss_rel` across S2 `a20d386`, S3 `866816a`, S4 `8e8f490`, and S5 `d474f82`):**

| Dtype tier | Max `param_l2` (row) | Max `upd_l2` (row) | Max `loss_rel` (row) |
|---|---|---|---|
| `float64` | `1.80e-13` (L4 NCCL `tiny_random`, `N = 4`, `replay_full_ft_distill_only_f64`) | `1.55e-10` (L4 NCCL `28M`, `N = 2`, `replay_full_ft_both_f64`) | `2.80e-11` (L4 NCCL `28M`, `N = 2`, `replay_full_ft_both_f64`) |
| `float32` | `5.58e-04` (L4 NCCL `5M`, `N = 2`, `domain_lora_temporal_f32`) | `1.04e-01` (L4 NCCL `5M`, `N = 2`, `domain_lora_temporal_f32`) | `7.28e-04` (L4 NCCL `5M`, `N = 2`, `domain_full_ft_f32`) |
| `bf16` autocast | `2.95e-03` (L4 NCCL `tiny_random`, `N = 2`, `bf16_head_only`) | `3.48e-01` (L4 NCCL `5M`, `N = 2`, `bf16_full_ft_replay`) | `1.27e-01` (L4 NCCL `5M`, `N = 2`, `bf16_full_ft_replay`) |

On released `float32` weights (`5M`/`28M`, `N = 2` and `N = 4`, S4 `8e8f490`), the moved-student `full_ft`
distillation rows (`*_moved_f32`, now unprovisional with buffers untouched) measured `param_l2`
`5.38e-06`–`1.25e-05` and `upd_l2` `9.51e-03`–`1.88e-02`.

*Interpretation discipline:* `float64` agreement across all 316 L4 rows and 70 macOS rows supports that the
multi-process and one-process paths execute the same algorithm. The larger `float32` and `bf16` update
differences on released weights are **measured and not explained**: for `float32`, the working reading that
CUDA default-SDPA `float32` gradient round-off on released weights (§13, spec §12) is amplified through a
short AdamW run—while `head_only` `upd_l2` stays `<= 1.78e-04` and `float64` `upd_l2` on the exact same
`lora_temporal` / `full_ft` domain rows stays at `8.87e-14`–`5.97e-12`—is a **hypothesis**, not a
demonstrated cause; for `bf16`, the cause of `upd_l2` up to `3.48e-01` and `loss_rel` up to `1.27e-01` is
uninvestigated (spec §12).

**Feature rows (`compile_mode="default"`, `num_workers=2`, `val_s`, `vram_mib`) at `8e8f490`:**

- *`compile_mode="default"`:* across every distributed run (L4 NCCL `tiny`/`5M`/`28M`, Linux `gloo`, and
  macOS arm64 `gloo`, at both `N = 2` and `N = 4`), `compile_default_lora_variate` reported
  `graphs=s1:1->end:2(+1,bumps_post_s1=1)` (matching its one-process reference) and
  `compile_default_lora_replay` reported `graphs=s1:3->end:5(+2,bumps_post_s1=1)` (versus
  `s1:2->end:3(+1,bumps_post_s1=1)` in `N = 2` cached one-process references and
  `s1:2->end:2(+0,bumps_post_s1=0)` in `N = 4` cached one-process references). Every run had at most one
  post-step-1 graph-count bump (shorter trailing batch specialisation), not repeated recompilation.
- *`workers_uneven_f32` `tail_s` and `val_s`:* L4 NCCL `tail_s` `0.19 s` (`tiny` `N = 2`), `0.28 s` (`tiny`
  `N = 4`), `0.17 s` (`5M` `N = 2`), `0.28 s` (`5M` `N = 4`), `0.27 s` (`28M` `N = 2`), `0.35 s` (`28M`
  `N = 4`), with `val_s` `0.26–0.59 s`; Linux L4 CPU/`gloo` `tail_s` `0.07 s` (`N = 2`), `0.17 s` (`N = 4`),
  `val_s` `0.13–0.25 s`; macOS arm64 CPU/`gloo` `tail_s` `20.02 s` (`N = 2` and `N = 4`, vs `10.01 s`
  one-process reference), `val_s` `0.78–1.22 s`. (`step_ms` and `wall_s` on shared-GPU NCCL are intentionally
  not quoted as multi-GPU scaling metrics because all ranks share one GPU.)
- *`vram_mib` (per-rank peak allocated MiB during `fit()`, not summed across shared-GPU ranks):*
  `tiny_random` `10.6–58.1` (`f32`), `11.2–20.4` (`bf16`), `13.1–34.4` (`f64`); `5M` `30.5–135.6` (`f32`),
  `36.0–132.2` (`bf16`), `53.4–265.2` (`f64`); `28M` `123.4–671.5` (`f32`), `143.1–663.6` (`bf16`),
  `243.6–1,350.5` (`f64`, tied peak on `replay_full_ft_distill_only_f64` and `replay_full_ft_both_f64` at
  both `N = 2` and `N = 4` with live `_frozen_base`).

#### Rank-failure timeout probe (`distributed_rank_failure_check.py`, commit `8e8f490`)

Kills rank 1 via `os._exit(1)` at step 2 mid-`fit()` and monitors surviving ranks under separate
`--startup-timeout` (`90.0 s`) and post-kill `--wall-timeout` (`60.0 s` / `65.0 s`) budgets with a strict
survivor-abort allow-list. In `8e8f490`, NCCL sessions run both Linux CPU/`gloo` (`09_rank_failure_gloo_n*`)
and CUDA/NCCL (`10_rank_failure_nccl_n*`):

| Platform, backend, `N` (`8e8f490`) | Timeouts (`pg` / `startup` / `wall`) | Killed rank 1 | Surviving rank(s) outcome | Elapsed (`exc_s` / `exit_s`) | Result |
|---|---|---|---|---|---|
| **L4 shared NCCL**, Linux x86_64, `N = 2` (`--device-map shared`) | `20.0 s` / `90.0 s` / `65.0 s` | `exit:1` (`0.21 s`) | Rank 0 killed by **`signal:SIGABRT`** (C++ watchdog: `Operation timed out after 20079 ms`, then `Waiting 15000ms for health check and flight recorder dump before aborting`, `Fatal Python error: Aborted` in `_reduce_step_stats`); **no Python exception raised; `except`/`finally` blocks do not run** | `-` / `35.50 s` | `PASS (no hang)` (`parent_killed=none`) |
| **L4 shared NCCL**, Linux x86_64, `N = 4` (`--device-map shared`) | `20.0 s` / `90.0 s` / `65.0 s` | `exit:1` (`0.21 s`) | Ranks 0, 2, 3 all killed by **`signal:SIGABRT`** (`Operation timed out after 20927–20932 ms` + `15000 ms` wait); **no Python exception raised; `except`/`finally` blocks do not run** | `-` / `36.34–36.44 s` | `PASS (no hang)` (`parent_killed=none`) |
| **Linux x86_64 CPU/`gloo`**, `N = 2` | `20.0 s` / `90.0 s` / `65.0 s` | `exit:1` (`0.12 s`) | Rank 0 `exit:2`; raises Python `builtins.RuntimeError` (`Read error ... Connection reset by peer`); normal `except`/`finally` unwinds | `0.07 s` / `1.62 s` | `PASS (no hang)` (`parent_killed=none`) |
| **Linux x86_64 CPU/`gloo`**, `N = 4` | `20.0 s` / `90.0 s` / `65.0 s` | `exit:1` (`0.11 s`) | Ranks 0, 2 (`Read error ... Connection reset by peer`, `0.07 s` / `1.71–1.76 s`) and rank 3 (`Connection closed by peer`, `1.67 s` / `2.87 s`) all `exit:2`; normal `except`/`finally` unwinds | `0.07–1.67 s` / `1.71–2.87 s` | `PASS (no hang)` (`parent_killed=none`) |
| **macOS arm64 CPU/`gloo`**, `N = 2` | `10.0 s` / `90.0 s` / `60.0 s` | `exit:1` (`0.04 s`) | Rank 0 `exit:2`; raises Python `builtins.RuntimeError` (Gloo `recv` timeout `10000ms`); normal `except`/`finally` unwinds | `10.01 s` / `10.27 s` | `PASS (no hang)` (`parent_killed=none`) |
| **macOS arm64 CPU/`gloo`**, `N = 4` | `10.0 s` / `90.0 s` / `60.0 s` | `exit:1` (`0.06 s`) | Ranks 0, 2, 3 all `exit:2`; raise Python `builtins.RuntimeError` (Gloo `recv`/`send` timeout `10000ms`) | `10.01 s` / `10.38–10.44 s` | `PASS (no hang)` (`parent_killed=none`) |

#### Session-driver behaviour & suite checks (`scripts/multi_gpu_check.py`, commits `8e8f490` and `d474f82`)

| Check | Platform & scope | Observed |
|---|---|---|
| Resume (identical command on finished log dir) | L4 Linux (`mgc_logs`) & macOS arm64 (`gloo`) at `8e8f490` and `d474f82` | L4: exit `0`, `17 of 17` runnable stages `0 (resumed)` at both `8e8f490` and `d474f82`. macOS: exit `0` in `1 s`, `7 of 7` runnable stages `0 (resumed)` at both commits. Nothing re-executed |
| Default-disposition `SIGINT` / `SIGTERM` / `SIGHUP` sent to session process group mid-stage | L4 Linux (`04_nccl_tiny_n2`, `--ranks 2 --sizes ""`, all 41 tiny scenarios / 7 runnable stages) & macOS arm64 (`04_gloo_tiny_n2`, `--ranks 2 --pg-timeout 10 --only domain_head_only_f32`, 1 scenario / 4 runnable stages per session harness) | `8e8f490` L4: exit `130` / `143` / `129` in `1.2 s` / `1.2 s` / `1.0 s` (signalled `4` PIDs per stage); summary records `<code> (interrupted:SIG...)` and `"Session interrupted by SIG... (exit <code>); remaining stages were not started."`; `0` leftover and `0` GPU compute processes (`d474f82` L4 `SIGINT`: exit `130` in `0.6 s`, `0` leftover/GPU procs). macOS: exit `130` / `143` / `129` in `0.4–0.5 s` (`8e8f490`) and `0.4 s` (`d474f82`), `0` leftover |
| Ignored-on-entry signals (as under `nohup`) sent at the same moment | L4 Linux (`SIGHUP`, `SIGINT` mid-`04_nccl_tiny_n2`, all 41 tiny scenarios / 7 runnable stages) & macOS arm64 (`SIGHUP`, `SIGINT`, `SIGTERM` mid-`04_gloo_tiny_n2`, `--only domain_head_only_f32` / 4 runnable stages per session harness) | `8e8f490` L4 (`SIGHUP`, `SIGINT`): session unaffected; all `7` runnable stages exited `0`, session exit `0` (`after_signal` `194.1 s` / `228.1 s`, `0` leftover / `0` GPU processes); `d474f82` L4 (`SIGHUP`): all `7` runnable stages exited `0` (`after_signal` `180.2 s`, `0` leftover/GPU procs). macOS (single-scenario run): all three completed with exit `0` (`12.8–13.5 s` at `8e8f490`, `12.8–12.9 s` at `d474f82`), `0` leftover |
| Reference-stage interrupt (`SIGINT` default mid-`03_ref_nccl_tiny_n2`, `8e8f490`) | L4 Linux (ranks 2 tiny) | Exit `130` in `0.3 s` (signalled `2` PIDs); only `01_env.done` remains; `refs/` empty (`[]`); `0` leftover and `0` GPU processes |
| Stage timeout (`--stage-timeout 20`, ranks 2 tiny) | L4 Linux (`8e8f490` and `d474f82`) | `01_env` `0`, `02_prototype_parity_cuda` `SKIPPED (shared GPU)`, `03_ref_nccl_tiny_n2` `124` (signalled `2` PIDs), `04_nccl_tiny_n2` `SKIPPED (ref failed)`, `07_ref_gloo_tiny_n2` `124` (`2` PIDs), `08_gloo_tiny_n2` `SKIPPED (ref failed)`, `09_rank_failure_gloo_n2` `0`, `10_rank_failure_nccl_n2` `124` (`4` PIDs); session exit `1`; `0` leftover, `0` GPU processes, `0` files left in `refs/` (identical stage outcomes and PID counts at `8e8f490` and `d474f82`) |
| Tree without `.git` (exported via `git archive HEAD \| tar -x` per session harness, `--backend gloo --ranks 2 --only domain_head_only_f32`, `8e8f490`) | L4 Linux | Fresh: exit `0`, `tree_state_method=content_hash` (`c0f97d37748a`). Resume: exit `0`, `4 of 4` runnable stages resumed. After appending `# touched` to `scripts/probes/_dist_common.py` (per session harness): `tree_state_sig` changed to `5e1e553b7cb5`, exit `0`, `0` resumed (all `4` re-ran) |
| Unusable `--log-dir` (`d474f82`: read-only parent dir and regular file at `--log-dir`) | L4 Linux (`--backend gloo --ranks 2`) & macOS arm64 | L4: both cases exited `2` with `1` `error:` line naming the supplied path and `0` tracebacks (`Permission denied` / `File exists`), before any stage started. macOS: both `ro_p/logs` (`Permission denied`) and `not_a_dir` (`File exists`) exited `2` with `1` `error:` line and no traceback |
| Relative `--log-dir` path hygiene | L4 Linux (`--log-dir mgc_logs`) & macOS arm64 (single-scenario default-dir run `--ranks 2 --pg-timeout 5 --only domain_head_only_f32` per session harness) | L4: zero `$HOME` hits in `.done` markers or `summary.txt` at both `8e8f490` and `d474f82`; at `8e8f490`, the session harness's first-3-lines check over stage logs reported `2` hits, both Python-emitted Triton `UserWarning` lines inside `.venv/` in `10_rank_failure_nccl_n{2,4}.log` (not stage arguments), and full stage logs also contain Python-emitted paths from `.venv/`, the `uv`-managed interpreter, and repository `src/`/`scripts/` frames in `faulthandler` output on NCCL `SIGABRT`. macOS: the harness grepped `multi_gpu_check_logs` and its session output for the local `$HOME` path and local `$(whoami)` username, finding `0` hits for both at `8e8f490` and `d474f82`; that default-dir run's own `09_rank_failure_gloo_n2` output (and session output) still contains PyTorch's compiled C++ build path (which did not match the local home path or username), and interrupted macOS gate stage logs contain Python traceback paths from both `.venv/` and the `uv`-managed interpreter |
| Test suite & driver unit tests on macOS arm64 | macOS arm64, Python `3.12.13`, `torch 2.15.0` | **`d474f82` (current):** `-m "not network"` **`616 passed, 15 skipped, 213 deselected`** (`212.76 s`; includes all `105` tests in `tests/test_multi_gpu_check.py`; skips: 1 `lacuna`, 14 CUDA `bf16`; `0` leftover). Earlier `8e8f490`: `-m "not network"` `614 passed, 15 skipped, 213 deselected` (`211.39 s`) and `tests/test_multi_gpu_check.py` `103 passed` across 3 consecutive standalone runs (`88.21 s`, `85.70 s`, `85.68 s`, `0` leftover) |
| Test suite & driver unit tests on Linux L4 VM (**failed at `8e8f490` → fixed in `335cf7d` → green at `d474f82`**) | Linux x86_64 L4 VM, CUDA visible, Python `3.12.16` (plus driver tests on Python `3.10`; `ps from procps-ng 3.3.17`) | **At `8e8f490`:** `-m "not network"` (Python `3.12.16`) was `625 passed, 3 skipped, 1 failed, 213 deselected` (`745.23 s`) and `tests/test_multi_gpu_check.py` was `102 passed, 1 failed` (×3 on Python `3.12` [`67.95–76.40 s`], ×1 on Python `3.10` [`78.66 s`]); the failure (`test_find_token_pids_exact_matching_empty_proc_fallback_and_scan_order`) caught a real defect where the `ps` fallback mixed BSD `eww` with `-axo`, which Linux `procps-ng` rejects with exit `1` (live Linux `/proc` scans were unaffected). **Fixed in `335cf7d`** (pure BSD `axewwo`/`axo`) and **green at `d474f82`:** `tests/test_multi_gpu_check.py` **`105 passed`** across 3 consecutive Python `3.12` runs (`68.29 s`, `67.94 s`, `67.78 s`, `0` leftover) and 1 isolated Python `3.10` run (`67.93 s`); `-m "not network"` (Python `3.12.16`) **`628 passed, 3 skipped, 213 deselected`** (`620.00 s`; skips: 1 `lacuna`, 2 Apple GPU; `0` failed, `0` leftover/GPU procs); full `scripts/cuda_check.sh` **`840 passed, 4 skipped`** (§13) |

#### Repeat at the final tip (`d474f82`, S5) and cross-session history (`82a21ca`, `a20d386`, `866816a`, `8e8f490`)

Five shared-GPU NVIDIA L4 sessions (`23034 MiB`, Linux x86_64; 8 vCPU per the sessions' provisioning records;
project Python `3.12.16`, `torch 2.15.0+cu132`, CUDA `13.2`, NCCL `2.30.7`, `--shared-gpu --python 3.12`) ran
across the review-fix and driver-rewrite stretch:

| Session & commit | Driver | Reference mode | Stages / wall time | Notes |
|---|---|---|---|---|
| S1 (`82a21ca`, 2026-11-23) | `595.112.07` | `sketch` (`k=8,192`) | 15 of 15 runnable stages exited `0` (16 listed, `02_prototype_parity_cuda` skipped); no Linux `gloo` rank-failure stage yet | First tracked session; also ran `cuda_check.sh` on Python `3.12.16` (§13) and the `exact` vs `sketch` comparison below. Review findings prompted S2–S5 |
| S2 (`a20d386`, 2026-11-23) | `595.112.07` | `auto` | 17 of 17 runnable stages exited `0` (18 listed, `02_prototype_parity_cuda` skipped); `30 min 05 s` | Post-round-1 fixes (buffers, `auto` mode, schema v4, split rank-failure stages) on a different host from S3–S5 per the provisioning record; `316/316` `xr_eq=yes` |
| S3 (`866816a`, 2026-11-23) | `580.178.04` | `auto` | 17 of 17 runnable stages exited `0` (18 listed, `02_prototype_parity_cuda` skipped); `42 min 22 s` | Post-round-2 fixes; `316/316` `xr_eq=yes`, plus resume (`17 of 17`), `--stage-timeout 20` (`0` leftover/GPU procs), and a `SIGINT` check. Per the S3 session harness, that interrupt check launched as a background job of a non-interactive shell with `SIGINT` ignored on entry, and the `866816a` embedded runner overrode the ignore and exited `130`—a defect removed in `4682669`—so the default-disposition Linux interrupt path was first exercised in S4 |
| S4 (`8e8f490`, 2026-11-24) | `580.178.04` | `auto` | 17 of 17 runnable stages exited `0` (18 listed, `02_prototype_parity_cuda` skipped); `47 min 51 s` | Detailed primary parity record above on the same machine type as S3 (a different VM instance per the provisioning record): Python `multi_gpu_check.py` driver, schema v5, `1e-6` f64 reload tol, signal/timeout/no-git/hygiene checks (table above); 1 Linux `ps`-fallback unit test failed (fixed in `335cf7d`) |
| S5 (`d474f82`, 2026-11-24) | `580.178.04` | `auto` | 17 of 17 runnable stages exited `0` (18 listed, `02_prototype_parity_cuda` skipped); `42 min 15 s` | Final-tip repeat on the same machine type and image as S3/S4 (another VM instance per the provisioning record; `python3` `3.10.12` on `PATH`, `ps from procps-ng 3.3.17`, `tree_state_method=git` `8a13c6c4c86d`): `316/316` `xr_eq=yes`, resume (`17 of 17`), `SIGINT` default / `SIGHUP` ignored gates, `--stage-timeout 20`, unusable-`--log-dir` checks, green Linux driver & non-network tests, and green `cuda_check.sh` (§13) |

**Compact repeat summary at `d474f82` (S5 on one NVIDIA L4 + macOS arm64):**

- *L4 S5 parity vs S4 (`8e8f490`):* all `316` rows (`246` NCCL + `70` Linux `gloo`) reported `xr_eq=yes,
  xr_max=0.00e+00` and `0` flags, and overall per-tier worst cases across all `316` rows matched S4. Row by
  row against S4 across all 8 parity columns (`steps`, `skip`, `xr_eq`, `xr_max`, `param_l2`, `upd_l2`,
  `loss_rel`, `val_rel`): every non-compiled NCCL row (`234/234`) and every Linux CPU/`gloo` row (`70/70`,
  including compiled CPU/`gloo`) matched S4 identically, and all `316` rows matched S4 in `vram_mib`; all
  **12 of 12** compiled NCCL rows differed from S4 in at least one parity column (`28M` `N = 4`
  `compile_default_lora_replay` differed in `param_l2` `9.46e-05` vs `9.44e-05` while printing the same
  `upd_l2` `1.81e-02`). For `workers_uneven_f32` (`num_workers=2`), `tail_s` stayed sub-second across every
  L4 stage: `0.16–0.34 s` on shared-L4 NCCL (`tiny_random`, `5M`, `28M` at `N = 2, 4`) and `0.08–0.09 s` on
  Linux CPU/`gloo` (`N = 2, 4`).
- *L4 S5 rank-failure stages:* Linux `gloo` survivors raised `builtins.RuntimeError` (`Connection
  reset/closed by peer`) in `exc_s` `0.05 s` / `exit_s` `0.80 s` (`N = 2`) and `0.06–0.85 s` / `0.88–1.68 s`
  (`N = 4`); shared-L4 NCCL survivors aborted via `SIGABRT` at `36.30 s` (`N = 2`, watchdog `20988 ms` +
  `15000 ms`) and `35.42–35.52 s` (`N = 4`, watchdog `20072–20130 ms` + `15000 ms`) with no Python exception;
  all `PASS (no hang)`, `parent_killed=none`.
- *macOS arm64 repeat at `d474f82`:* `--backend gloo --ranks 2 4 --pg-timeout 10` finished in `155 s` (`7 of
  7` runnable stages exited `0`, 8 listed), all `70` rows matched `8e8f490` across all 8 parity columns (`0`
  diffs), `workers_uneven_f32` `tail_s` measured `20.02 s` at both `N = 2` and `N = 4`, `gloo` rank-failure
  survivors raised `builtins.RuntimeError` at `10.01 s` (`exit_s` `10.24–10.44 s`), and
  resume/signal/path/unusable-`--log-dir`/pytest gates all passed (table above).

**Observed cross-session reproducibility and variation (S2–S5, reported as observations without claiming
unverified causes):**

- *Worst-case parity identical across sessions:* S2 (`a20d386`, driver `595.112.07`), S3 (`866816a`, driver
  `580.178.04`), S4 (`8e8f490`, driver `580.178.04`), and S5 (`d474f82`, driver `580.178.04`) produced the
  exact same printed worst-case `param_l2`, `upd_l2`, and `loss_rel` across all 316 L4 rows in every dtype
  tier (`float64`, `float32`, `bf16`).
- *Same machine type, different VM instances (S3 vs S4 vs S5, per the provisioning records; same driver
  `580.178.04`):* every non-compiled NCCL row and every Linux CPU/`gloo` row (including compiled CPU/`gloo`
  rows) was identical in the printed parity columns (`steps`, `skip`, `xr_eq`, `xr_max`, `param_l2`, `upd_l2`,
  `loss_rel`, `val_rel`) and `vram_mib` (timing columns and `val_s` varied between sessions); **only compiled
  NCCL rows varied** (10 of 12 model×rank entries differed between S3 and S4, while `5M` `N = 2` matched; all
  12 of 12 differed between S4 and S5).
- *Compiled rows on CUDA vary run-to-run (4-session sample S2–S5; cause uninvestigated):* across S2, S3, S4,
  and S5, `compile_default_*` NCCL rows varied on both tiny and released weights—for example:
  - `28M` `N = 2` `compile_default_lora_variate` `upd_l2`: `7.83e-04` (S2), `5.09e-04` (S3), `2.25e-03` (S4),
    `5.09e-04` (S5);
  - `28M` `N = 2` `compile_default_lora_replay` `upd_l2`: `1.39e-02` (S2), `1.44e-02` (S3), `1.25e-02` (S4),
    `1.44e-02` (S5);
  - `28M` `N = 4` `compile_default_lora_variate` `upd_l2`: `2.24e-04` (S2), `1.45e-04` (S3), `2.24e-04` (S4),
    `1.45e-04` (S5);
  - `tiny_random` `N = 4` `compile_default_lora_replay` `upd_l2`: `9.99e-05` (S2), `8.14e-05` (S3), `7.52e-05`
    (S4), `1.34e-04` (S5).
  Graph counts (`s1:1->end:2` / `s1:3->end:5`) and bitwise cross-rank equality (`xr_eq=yes, xr_max=0.00e+00`)
  held in every run. The compiled `float32` distance from its one-process compiled reference is not
  reproducible run-to-run on CUDA (spec §12).
- *CPU-dependent rows vary across hosts (S2 vs S3–S5; all pairs below ordered S2 vs S3–S5):* against S2 (a
  different host per the provisioning record, driver `595.112.07`; CPU models were not recorded), all 34
  compared Linux CPU/`gloo` rows at each `N` and all 12 `*_moved_f32` NCCL rows (whose deterministic warm-up
  runs on CPU) differed in their printed values—by up to **12×** on CPU/`gloo` (`08_gloo_tiny_n2`
  `domain_head_only_f32` `param_l2` `5.97e-07` vs `4.98e-08`, `upd_l2` `1.87e-06` vs `1.56e-07`; plus two
  zero-vs-round-off `float64` cells: `08_gloo_tiny_n2` `domain_head_only_f64` `loss_rel` `3.71e-16` in S2 vs
  `0.00e+00` in S3–S5, and `08_gloo_tiny_n4` `val_restore_early_stop_f64` `val_rel` `0.00e+00` in S2 vs
  `1.15e-16` in S3–S5) and up to **4.1×** on NCCL `*_moved_f32` (`28M` `N = 4`
  `replay_full_ft_distill_only_moved_f32` `loss_rel` `2.06e-05` vs `8.54e-05`; `replay_full_ft_both_moved_f32`
  `upd_l2` up to `2.37×` on `tiny_random` `N = 4` `3.03e-05` vs `1.28e-05`, and `1.63×` on `5M` `N = 4`
  `2.35e-02` vs `1.44e-02`)—while all other non-compiled NCCL rows matched in their printed parity columns and
  `vram_mib` across S2–S5.

*Historical `exact` vs `sketch` comparison (`82a21ca`, L4 shared NCCL `N = 2`, `k = 8,192`; not re-run at
`8e8f490`, `d474f82`, or `3ac7c6d`):* ten entries across `tiny_random` and `5M` gave `ref_exact.zip`
`114,469,429` bytes (`109.17 MiB`) vs `ref_sketch.zip` `674,359` bytes (`0.64 MiB`, ~`170×` smaller), `0.00%`
`loss_rel` diff on all 10 entries, and `0.00%`–`1.60%` max relative diff in `param_l2`/`upd_l2` (`5M`
`domain_lora_temporal_f32` `param_l2` `5.61e-04` exact vs `5.58e-04` sketch, `upd_l2` `1.04e-01` both; `5M`
`bf16_full_ft_replay` `upd_l2` `3.51e-01` exact vs `3.48e-01` sketch, `loss_rel` `1.27e-01` both; `tiny`
`lora_temporal` `0.00%` because `numel = 8,192 <= SKETCH_DIM` stores `exact_delta`).

#### Separate-GPU session S6 (`3ac7c6d`, 2026-11-24, 4× NVIDIA GeForce RTX 3060, `--device-map per-rank`)

First separate-GPU Stage 3 session, run at commit `3ac7c6d` (`scripts/`, `src/`, and `tests/` identical to
`d474f82`; `tree_state_method=git`, `a45b72e7fa66`) with **one rank per GPU** (`--device-map per-rank`,
`NCCL_MULTI_RANK_GPU_ENABLE` unset, `CUDA_VISIBLE_DEVICES` unset):

- **Environment, topology, and provenance:** rented GPU-marketplace Linux x86_64 Docker container host (per
  the session's provisioning record). Logged host CPU/memory (one-off session-harness pre-flight check, not a
  tracked probe): `Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz`, 2 sockets × 20 cores/socket × 2 threads/core =
  `80` logical CPUs across 2 NUMA nodes; `94 GiB` RAM (`90 GiB` available), `32G` `/` overlay, `47G`
  `/dev/shm`. GPUs: **4× `NVIDIA GeForce RTX 3060`** (`12288 MiB` each, driver `580.178.04` in tracked
  `01_env`; Ampere compute capability `(8, 6)`, `distinct GPU models: 1`, PCIe gen `3` max, width current `16`
  / max `16`, `bf16 native: True`, `is_bf16_supported(): True`, `matmul ok on all 4`, topology GPU0–GPU1 `PHB`
  [NUMA 0, CPU affinity `0-19,40-59`] / GPU2–GPU3 `PHB` [NUMA 1, CPU affinity `20-39,60-79`] / cross-socket
  `SYS`, and `torch.cuda.can_device_access_peer` all `0` — one-off session-harness observations, not tracked
  probes). Software: project Python `3.12.16`, `torch 2.15.0+cu132`, CUDA runtime `13.2`, NCCL `2.30.7`
  (`2.30.7+cuda13.3` in NCCL init), `arch_list ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`,
  `python_h=True`, `c_compiler=True`, `uv 0.13.9`, `/usr/bin/cc`, `/usr/bin/git`, and
  `scripts/multi_gpu_check.py` running on `python3` `3.12.3` on `PATH`.
- **4-rank NCCL transport & bandwidth check (one-off session-harness observation with `NCCL_DEBUG=INFO`, not a
  tracked probe):** exited `0`; `Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 0`; 2 ring channels
  (`Channel 00/02` and `01/02`: `0 1 2 3`), all 8 ring links **`via SHM/direct`** (PCIe/SHM transport, no
  NVLink or GPU P2P); `all_reduce 64 MiB on 4 ranks: 11.0 ms/iter, bus bandwidth 9.17 GB/s, values
  correct=True`.
- **Full session command and summary:**
  ```bash
  bash scripts/multi_gpu_check.sh --python 3.12 --log-dir mgc_logs
  ```
  Ran with `backend=nccl`, `device_map=per-rank`, `ranks=2 4`, `NCCL_MULTI_RANK_GPU_ENABLE` unset;
  real `LOCAL_RANK` per-rank binding (`device="auto"` → `cuda:LOCAL_RANK`) logged `device=cuda:0..1`
  (`N = 2`) and `device=cuda:0..3` (`N = 4`), `gpu_count=4`. Wall time **`47 min 42 s`** (`06:06:41`–`06:54:23`),
  exit code `0`. **18 of 18 listed stages exited `0` (0 skipped)**, including `02_prototype_parity_cuda` on 4
  separate GPUs; `0` leftover and `0` GPU compute processes afterwards; `0` `$HOME` hits in `.done`/summary.
  Threads: `threads=40` in single-process CUDA reference stages, `threads=1` in all `torchrun` and CPU/`gloo`
  stages. Reference archives (`auto(f64<=1000000,k=8192)`, schema v5): `tiny_random` CUDA `25.44 MiB` (40
  entries per `N`), `5M`+`28M` CUDA `37.47 MiB` (80 entries per `N`), `tiny_random` CPU `25.11 MiB` (34
  entries per `N`). TorchInductor emitted `Not enough SMs to use max_autotune_gemm mode` on RTX 3060 during
  compiled runs.

**4-GPU `02_prototype_parity_cuda` (`scripts/probes/distributed_parity.py --device cuda --sizes 5M 28M`, `N = 2` and `N = 4`, rank `r` on `cuda:r`):**

- **Total:** **`706/706 passed, 0 failed, 136 info (ungraded)`** — `tiny_random` `326 PASS / 0 INFO`,
  `Timestral-5M` `190 PASS / 68 INFO`, `Timestral-28M` `190 PASS / 68 INFO`; all `210` cross-rank checks
  (`70` per model) measured **`0.00e+00`**.
- **Enforced PASS maxima:** `tiny_random` `float64` `2.60e-14` (`[5] lora_variate N=4 param_update <= 1e-10`),
  `float32` `4.56e-06` (`[5] lora_variate N=2 param_update <= 1e-4`); `Timestral-5M` `float64` `3.63e-14`
  (two-way tie: `[3] N=2 head_only grad` and `[3] N=2 head_only_cached grad`, `<= 1e-12`), `float32`
  `6.93e-07` (`[2] N=4 empty_shard loss <= 1e-5`); `Timestral-28M` `float64` `2.03e-15` (two-way tie: `[2]
  N=4 unequal loss` and `[3] N=2 head_only grad`, `<= 1e-12`), `float32` `1.08e-06` (two-way tie: `[3] N=2
  head_only grad` and `[3] N=2 head_only_cached grad`, `<= 1e-5`).
- **Ungraded `[INFO]` released body-backprop ranges:**
  - `Timestral-5M`: single-step `float64` `4.94e-15`–`2.59e-11` (`[3] N=2 full_ft grad`), `float32`
    `7.14e-05`–`2.11e-03` (`[3] N=4 full_ft grad`); 10-step `[5]` `float64` `9.76e-17`–`1.81e-12` (`full_ft
    N=4 param_update`), `float32` `1.14e-05`–`3.02e-02` (`full_ft N=2 param_update`).
  - `Timestral-28M`: single-step `float64` `1.21e-14`–`1.55e-11` (`[2] N=4 unequal grad`), `float32`
    `4.92e-05`–`9.05e-03` (`[2] N=4 unequal grad`); 10-step `[5]` `float64` `1.31e-16`–`4.77e-13` (`full_ft
    N=4 param_update`), `float32` `1.23e-06`–`1.55e-02` (`full_ft N=4 param_update`).

**S6 driver parity summary (`distributed_trainer_check.py`, `3ac7c6d`, `--reference-mode auto`, `--device-map per-rank`):**

Every executed row across S6 reported **`xr_eq = yes` and `xr_max = 0.00e+00`** over all model parameters and
buffers (`316/316`: `246` separate-GPU NCCL rows across `tiny_random`, `5M`, `28M` × `N = 2, 4` + `70` Linux
Xeon CPU/`gloo` rows; `12` skipped `bf16` rows on `gloo`) and **zero warning/error flags**.
`val_restore_early_stop_*` returned `early=True restored=True best_step=2`, `export_multirank_f32` returned
`export=ok(2/2)` / `export=ok(4/4)`, and `stochastic_lora_variate_f32` matched bitwise across ranks in every
table.

| Platform, backend, model, `N` (`3ac7c6d`, S6) | Executed / skipped | `float64` (14 rows): `param_l2` range (med); `upd_l2` min / med / max (worst scenario); `loss_rel` max; `val_rel` | `float32` (20 rows; CPU splits 18 non-id + 2 id-start): `param_l2` range (med); `upd_l2` min / med / max (worst scenario); `loss_rel` max; `val_rel` | `bf16` (6 rows, CUDA only; **no tier set**): `param_l2` range; `upd_l2` min / max (worst); `loss_rel` max |
|---|---|---|---|---|
| **RTX 3060 separate NCCL**, `tiny_random`, `N = 2` (`cuda:0..1`) | `41 / 0` | `param_l2` `1.03e-16`–`2.03e-15` (med `2.63e-16`, exact); `upd_l2` `2.17e-15` / `5.37e-15` / **`1.09e-14`** (`replay_full_ft_distill_only_f64`, ~`9,170×` below T15 `1e-10`); `loss_rel` `<= 6.90e-16`; `val_rel` `2.23e-16` | `param_l2` `4.48e-08`–`8.68e-06` (med `1.76e-07`); `upd_l2` `1.12e-06` / `2.42e-06` / **`1.76e-04`** (`compile_default_lora_replay`; non-compile max `9.17e-06` `export_multirank_f32`; moved `2.28e-06`/`1.61e-06`); `loss_rel` `<= 1.67e-05` (`compile_default_lora_replay`); `val_rel` `4.22e-08`–`1.79e-07` | `param_l2` `4.86e-04`–**`1.53e-02`** (`bf16_head_only`); `upd_l2` `1.64e-02` (`bf16_full_ft`) / **`4.78e-02`** (`bf16_head_only`); `loss_rel` `<= 9.17e-03` (`bf16_lora_temporal`) |
| **RTX 3060 separate NCCL**, `tiny_random`, `N = 4` (`cuda:0..3`) | `41 / 0` | `param_l2` `1.04e-16`–**`1.52e-13`** (med `2.53e-16`, exact; max `replay_full_ft_distill_only_f64`); `upd_l2` `1.81e-15` / `4.33e-15` / **`6.59e-12`** (`replay_full_ft_distill_only_f64`, `15.2×` below T15 `1e-10`; 2nd `2.44e-12` `replay_full_ft_both_f64`; all other 12 `<= 1.03e-14`); `loss_rel` `<= 4.18e-14`; `val_rel` `0.00e+00` | `param_l2` `7.20e-08`–`3.89e-06` (med `2.60e-07`); `upd_l2` `1.18e-06` / `4.12e-06` / **`8.09e-05`** (`compile_default_lora_replay`; non-compile max `3.51e-05` `replay_lora_variate_distill_only_f32`; moved `8.69e-06`/`8.94e-06`); `loss_rel` `<= 1.01e-05` (`compile_default_lora_replay`); `val_rel` `3.97e-08`–`4.04e-08` | `param_l2` `8.85e-04`–**`2.34e-02`** (`bf16_head_only`); `upd_l2` `2.99e-02` (`bf16_full_ft`) / **`1.08e-01`** (`bf16_lora_temporal`); `loss_rel` `<= 6.10e-03` (`bf16_head_only`) |
| **RTX 3060 separate NCCL**, `5M`, `N = 2` (**no tier set**) | `41 / 0` | `param_l2` `2.94e-17`–`2.10e-14` (med `1.09e-16`); `upd_l2` `6.44e-15` / `3.32e-14` / **`6.18e-11`** (`replay_full_ft_distill_only_f64`; 2nd `6.02e-11` `replay_full_ft_both_f64`; 3rd `1.20e-12` `domain_full_ft_f64`); `loss_rel` `<= 9.84e-12`; `val_rel` `0.00e+00` | `param_l2` `2.50e-08`–**`2.03e-04`** (`domain_lora_temporal_f32`; med `5.65e-06`); `upd_l2` `4.66e-06` / `1.22e-03` / **`3.80e-02`** (`domain_lora_temporal_f32`; 2nd `3.26e-02` `domain_full_ft_f32`; `head_only <= 3.11e-05`; moved `2.37e-02`/`1.12e-02`); `loss_rel` `<=` **`5.01e-04`** (`domain_full_ft_f32`); `val_rel` `1.10e-07`–`1.68e-07` | `param_l2` `8.83e-05`–`1.62e-03`; `upd_l2` `1.07e-01` (`bf16_head_only`) / **`3.03e-01`** (`bf16_lora_temporal`; 2nd `3.00e-01` `bf16_full_ft_replay`); `loss_rel` `<=` **`1.00e-01`** (`bf16_full_ft_replay`) |
| **RTX 3060 separate NCCL**, `5M`, `N = 4` (**no tier set**) | `41 / 0` | `param_l2` `6.27e-17`–`2.08e-14` (med `2.18e-16`); `upd_l2` `1.19e-14` / `1.66e-13` / **`6.05e-11`** (`replay_full_ft_both_f64`; 2nd `4.51e-11` `replay_full_ft_distill_only_f64`; 3rd `1.52e-12` `domain_lora_temporal_f64`); `loss_rel` `<= 1.19e-12`; `val_rel` `0.00e+00` | `param_l2` `7.63e-08`–**`3.19e-04`** (`domain_lora_temporal_f32`; med `8.74e-06`); `upd_l2` `1.45e-05` / `8.48e-03` / **`5.50e-02`** (`domain_lora_temporal_f32`; 2nd `2.49e-02` `replay_full_ft_both_f32`; `head_only <= 1.02e-04`; moved `1.97e-02`/`2.40e-02`); `loss_rel` `<= 2.25e-04` (`replay_lora_variate_distill_only_f32`); `val_rel` `1.99e-07`–`1.15e-06` | `param_l2` `8.37e-05`–`1.59e-03`; `upd_l2` `7.85e-02` (`bf16_head_only`) / **`2.83e-01`** (`bf16_lora_variate_replay`; 2nd `2.75e-01` `bf16_lora_temporal`); `loss_rel` `<= 3.03e-02` (`bf16_lora_variate_replay`) |
| **RTX 3060 separate NCCL**, `28M`, `N = 2` (**no tier set**) | `41 / 0` | `param_l2` `6.91e-17`–`3.31e-14` (med `5.44e-16`); `upd_l2` `2.15e-14` / `1.02e-13` / **`6.05e-11`** (`replay_full_ft_both_f64`; 2nd `4.81e-11` `replay_full_ft_distill_only_f64`; all other 12 `<= 4.99e-13`); `loss_rel` `<=` **`9.99e-12`**; `val_rel` `2.11e-16` | `param_l2` `5.76e-08`–**`3.70e-04`** (`domain_lora_temporal_f32`; med `6.41e-06`); `upd_l2` `7.72e-06` / `2.79e-03` / **`8.02e-02`** (`domain_lora_temporal_f32`; 2nd `2.87e-02` `replay_full_ft_distill_only_f32`; `head_only <= 3.53e-04`; moved `1.08e-02`/`9.67e-03`); `loss_rel` `<= 1.15e-04` (`compile_default_lora_replay`); `val_rel` `2.32e-07`–`1.09e-06` | `param_l2` `1.22e-04`–`1.42e-03`; `upd_l2` `1.29e-01` (`bf16_lora_variate_replay`) / **`3.07e-01`** (`bf16_lora_temporal`; 2nd `2.22e-01` `bf16_full_ft_replay`); `loss_rel` `<= 4.70e-02` (`bf16_full_ft_replay`) |
| **RTX 3060 separate NCCL**, `28M`, `N = 4` (**no tier set**) | `41 / 0` | `param_l2` `6.72e-17`–`1.80e-15` (med `5.86e-16`); `upd_l2` `1.55e-14` / `1.28e-13` / **`3.10e-12`** (`replay_full_ft_distill_only_f64`; 2nd `2.00e-12` `replay_full_ft_both_f64`; all other 12 `<= 2.07e-13`); `loss_rel` `<= 8.91e-14`; `val_rel` `2.87e-16` | `param_l2` `6.25e-08`–**`3.50e-04`** (`domain_lora_temporal_f32`; med `6.43e-06`); `upd_l2` `7.45e-06` / `5.81e-03` / **`7.18e-02`** (`domain_lora_temporal_f32`; 2nd `2.36e-02` `replay_full_ft_both_f32`; `head_only <= 4.43e-05`; moved `1.76e-02`/`1.71e-02`); `loss_rel` `<= 1.34e-04` (`compile_default_lora_replay`); `val_rel` `9.87e-08`–`1.59e-07` | `param_l2` `1.16e-04`–`1.10e-03`; `upd_l2` `1.02e-01` (`bf16_head_only`) / **`2.29e-01`** (`bf16_full_ft_replay`; 2nd `2.27e-01` `bf16_lora_temporal`); `loss_rel` `<= 1.29e-02` (`bf16_lora_variate_replay`) |
| **Linux Xeon CPU/`gloo`**, `tiny_random`, `N = 2` (`threads=1`) | `35 / 6` (`bf16` skipped) | `param_l2` `9.52e-17`–**`1.84e-13`** (med `6.72e-16`, exact; max `replay_full_ft_distill_only_f64`); `upd_l2` `2.43e-15` / `5.58e-15` / **`8.19e-12`** (`replay_full_ft_distill_only_f64`, `12.2×` below `1e-10`; 2nd `1.65e-12` `replay_full_ft_both_f64`); `loss_rel` `<= 6.00e-14`; `val_rel` `1.16e-16` | Non-id (18): `param_l2` `6.79e-08`–`3.12e-06` (med `3.06e-07`), `upd_l2` `1.67e-06` / `5.29e-06` / **`6.33e-05`** (`compile_default_lora_replay`; non-compile max `8.31e-06` `export_multirank_f32`), `loss_rel` `<= 7.85e-06`, `val_rel` `5.90e-08`–`1.58e-07`. Id-start (2, §17.2): `param_l2` `3.14e-05`–`5.74e-05`, `upd_l2` `1.38e-03`–`2.55e-03` | — |
| **Linux Xeon CPU/`gloo`**, `tiny_random`, `N = 4` (`threads=1`) | `35 / 6` | `param_l2` `7.34e-17`–`1.50e-13` (med `4.50e-16`, exact); `upd_l2` `1.50e-15` / `5.72e-15` / **`6.48e-12`** (`replay_full_ft_distill_only_f64`, `15.4×` below `1e-10`; 2nd `2.20e-12` `replay_full_ft_both_f64`); `loss_rel` `<= 5.64e-14`; `val_rel` `2.30e-16` | Non-id (18): `param_l2` `4.62e-08`–`5.09e-06` (med `2.77e-07`), `upd_l2` `1.12e-06` / `3.78e-06` / **`1.06e-04`** (`compile_default_lora_replay`; non-compile max `7.79e-06` `export_multirank_f32`), `loss_rel` `<= 8.02e-06`, `val_rel` `9.24e-08`–`1.06e-07`. Id-start (2, §17.2): `param_l2` `3.60e-05`–`6.18e-05`, `upd_l2` `1.54e-03`–`2.67e-03` | — |

**Worst multi-process vs one-process differences across all 316 rows — 4× RTX 3060 separate-GPU (S6, `3ac7c6d`) alongside single NVIDIA L4 shared-GPU (S4/S5, `8e8f490`/`d474f82`):**

*Non-comparability disclaimer (observation only, no explanation claimed):* the two columns are shown side by
side as an observation without explanation. Parity values are **not comparable digit-for-digit** across GPUs:
each session measures its multi-rank run against a one-process reference computed on its own GPU and host CPU,
so differences reflect a different GPU architecture, kernel set, and host CPU rather than a direct
shared-GPU-vs-separate-GPU delta.

| Dtype tier & metric | Single L4 shared-GPU (S4/S5, `8e8f490`/`d474f82`, 316 rows) | 4× RTX 3060 separate-GPU (S6, `3ac7c6d`, 316 rows) |
|---|---|---|
| `float64` max `param_l2` | `1.80e-13` (L4 NCCL `tiny_random` `N = 4`, `replay_full_ft_distill_only_f64`) | `1.84e-13` (**Linux Xeon `gloo`** `tiny_random` `N = 2`, `replay_full_ft_distill_only_f64`; worst separate-GPU NCCL `param_l2` is `1.52e-13`, `tiny_random` `N = 4`) |
| `float64` max `upd_l2` | `1.55e-10` (L4 NCCL `28M` `N = 2`, `replay_full_ft_both_f64`) | `6.18e-11` (RTX 3060 NCCL `5M` `N = 2`, `replay_full_ft_distill_only_f64`) |
| `float64` max `loss_rel` | `2.80e-11` (L4 NCCL `28M` `N = 2`, `replay_full_ft_both_f64`) | `9.99e-12` (RTX 3060 NCCL `28M` `N = 2`, `replay_full_ft_both_f64`) |
| `float32` max `param_l2` | `5.58e-04` (L4 NCCL `5M` `N = 2`, `domain_lora_temporal_f32`) | `3.70e-04` (RTX 3060 NCCL `28M` `N = 2`, `domain_lora_temporal_f32`) |
| `float32` max `upd_l2` | `1.04e-01` (L4 NCCL `5M` `N = 2`, `domain_lora_temporal_f32`) | `8.02e-02` (RTX 3060 NCCL `28M` `N = 2`, `domain_lora_temporal_f32`) |
| `float32` max `loss_rel` | `7.28e-04` (L4 NCCL `5M` `N = 2`, `domain_full_ft_f32`) | `5.01e-04` (RTX 3060 NCCL `5M` `N = 2`, `domain_full_ft_f32`) |
| `bf16` max `param_l2` | `2.95e-03` (L4 NCCL `tiny_random` `N = 2`, `bf16_head_only`) | `2.34e-02` (RTX 3060 NCCL `tiny_random` `N = 4`, `bf16_head_only`) |
| `bf16` max `upd_l2` | `3.48e-01` (L4 NCCL `5M` `N = 2`, `bf16_full_ft_replay`) | `3.07e-01` (RTX 3060 NCCL `28M` `N = 2`, `bf16_lora_temporal`) |
| `bf16` max `loss_rel` | `1.27e-01` (L4 NCCL `5M` `N = 2`, `bf16_full_ft_replay`) | `1.00e-01` (RTX 3060 NCCL `5M` `N = 2`, `bf16_full_ft_replay`) |

On released `float32` weights (`5M`/`28M`, `N = 2` and `N = 4`, S6 `3ac7c6d`), the moved-student `full_ft`
distillation rows (`*_moved_f32`) measured `param_l2` `5.49e-06`–`1.17e-05` and `upd_l2`
`9.67e-03`–`2.40e-02`; on those same released models, `float64` `upd_l2` on `domain_lora_temporal_f64` and
`domain_full_ft_f64` stayed at `5.20e-14`–`1.52e-12` while `float32` `head_only` `upd_l2` stayed
`<= 3.53e-04` (`param_l2 <= 1.91e-06`).

**S6 feature rows (`compile_mode="default"`, `num_workers=2`, `val_s`, `vram_mib`):**

- *`compile_mode="default"`:* identical graph counts to S4/S5 across every S6 distributed run (RTX 3060
  separate NCCL `tiny`/`5M`/`28M` and Linux Xeon `gloo`, `N = 2` and `N = 4`): `compile_default_lora_variate`
  reported `graphs=s1:1->end:2(+1,bumps_post_s1=1)` and `compile_default_lora_replay` reported
  `graphs=s1:3->end:5(+2,bumps_post_s1=1)` (versus `s1:2->end:3(+1,bumps_post_s1=1)` at `N = 2` and
  `s1:2->end:2(+0,bumps_post_s1=0)` at `N = 4` in cached one-process references), with `xr_eq=yes,
  xr_max=0.00e+00` everywhere.
- *`workers_uneven_f32` (`num_workers=2`) `tail_s` and `val_s`:* separate-GPU RTX 3060 NCCL `tail_s` `0.19 s`
  (`tiny` `N = 2`), `0.20 s` (`tiny` `N = 4`), `0.23 s` (`5M` `N = 2`), `0.21 s` (`5M` `N = 4`), `0.25 s`
  (`28M` `N = 2`), `0.26 s` (`28M` `N = 4`), with `val_s` `0.17–0.26 s`; Linux Xeon CPU/`gloo` `tail_s`
  `0.10 s` (`N = 2`), `0.09 s` (`N = 4`), `val_s` `0.10 s`.
- *`vram_mib` (per-rank peak allocated MiB on each dedicated RTX 3060 during `fit()`):* `tiny_random`
  `10.6–21.2` (`f32`), `11.2–20.4` (`bf16`), `13.1–34.4` (`f64`); `5M` `30.5–135.6` (`f32`), `36.0–132.2`
  (`bf16`), `53.4–265.2` (`f64`); `28M` `123.4–671.5` (`f32`), `143.1–663.6` (`bf16`), `243.6–1,350.5` (`f64`,
  tied peak on `replay_full_ft_distill_only_f64` and `replay_full_ft_both_f64` at both `N = 2` and `N = 4`).
- *Not a scaling study:* `step_ms` and `wall_s` from this 6-step `B = 2` measuring driver are intentionally
  not quoted as multi-GPU scaling or throughput measurements; 1/2/4-GPU scaling and overhead micro-benchmarks
  remain open in Stage 3.

**S6 rank-failure timeout probe (`distributed_rank_failure_check.py`, `3ac7c6d`, `--device-map per-rank`):**

| Platform, backend, `N` (`3ac7c6d`, S6) | Timeouts (`pg` / `startup` / `wall`) | Killed rank 1 | Surviving rank(s) outcome | Elapsed (`exc_s` / `exit_s`) | Result |
|---|---|---|---|---|---|
| **RTX 3060 separate NCCL**, Linux x86_64, `N = 2` (`cuda:0..1`) | `20.0 s` / `90.0 s` / `65.0 s` | `exit:1` (`0.17 s`) | Rank 0 killed by **`signal:SIGABRT`** (C++ watchdog: `Operation timed out after 20729 ms` + `15000ms` health-check wait, `Fatal Python error: Aborted` in `_reduce_step_stats`); **no Python exception raised; `except`/`finally` blocks do not run** | `-` / `36.05 s` | `PASS (no hang)` (`parent_killed=none`) |
| **RTX 3060 separate NCCL**, Linux x86_64, `N = 4` (`cuda:0..3`) | `20.0 s` / `90.0 s` / `65.0 s` | `exit:1` (`0.25 s`) | Ranks 0, 2, 3 all killed by **`signal:SIGABRT`** (`Operation timed out after 20693–20699 ms` + `15000ms` wait); **no Python exception raised; `except`/`finally` blocks do not run** | `-` / `35.95–36.05 s` | `PASS (no hang)` (`parent_killed=none`) |
| **Linux Xeon CPU/`gloo`**, `N = 2` | `20.0 s` / `90.0 s` / `65.0 s` | `exit:1` (`0.14 s`) | Rank 0 `exit:2`; raises Python `builtins.RuntimeError` (`Connection reset by peer`); normal `except`/`finally` unwinds | `0.09 s` / `0.79 s` | `PASS (no hang)` (`parent_killed=none`) |
| **Linux Xeon CPU/`gloo`**, `N = 4` | `20.0 s` / `90.0 s` / `65.0 s` | `exit:1` (`0.16 s`) | Ranks 0, 2 (`Connection reset by peer`, `0.08 s` / `0.81–0.86 s`) and rank 3 (`Connection closed by peer`, `0.75 s` / `1.57 s`) all `exit:2`; normal `except`/`finally` unwinds | `0.08–0.75 s` / `0.81–1.57 s` | `PASS (no hang)` (`parent_killed=none`) |

**S6 session-driver gates (one-off session-harness checks, not tracked probes), driver unit tests, and single-GPU `cuda_check.sh` (`3ac7c6d`, 4× RTX 3060 host):**

| Check (`3ac7c6d`, 4× RTX 3060 host) | Scope | Observed |
|---|---|---|
| Resume (identical command on finished `mgc_logs`) | Full 4-GPU separate-GPU session | Exit `0`, **`18 of 18`** stages `0 (resumed)`; nothing re-executed |
| Default-disposition `SIGINT` / `SIGTERM` / `SIGHUP` mid-`04_nccl_tiny_n2` | `--ranks 2 --sizes ""` (`device_map=per-rank`) | Exit `130` / `143` / `129` in `1.1 s` / `1.0 s` / `1.0 s` (signalled `4` PIDs per stage); `0` leftover and `0` GPU procs |
| Ignored-on-entry `SIGHUP` / `SIGINT` mid-`04_nccl_tiny_n2` | `--ranks 2 --sizes ""` (`device_map=per-rank`) | Session unaffected; exit `0` (`after_signal` `165.8 s` / `163.5 s`), all `7` runnable stages exited `0` (`02_prototype_parity_cuda SKIPPED (rank 4 not requested)`), `0` leftover and `0` GPU procs |
| Reference-stage interrupt (`SIGINT` default mid-`03_ref_nccl_tiny_n2`) | `--ranks 2 --sizes ""` | Exit `130` in `0.5 s` (signalled `2` PIDs); only `01_env.done` remains (`['01_env.done']`), `refs: []`, `0` leftover and `0` GPU procs |
| Stage timeout (`--stage-timeout 20`, `--ranks 2 --sizes ""`) | Separate-GPU `device_map=per-rank` | `01_env` `0`, **`02_prototype_parity_cuda SKIPPED (rank 4 not requested)`**, `03_ref_nccl_tiny_n2` `124` (`2` PIDs), `04_nccl_tiny_n2` `SKIPPED (ref failed)`, `07_ref_gloo_tiny_n2` `124` (`2` PIDs), `08_gloo_tiny_n2` `SKIPPED (ref failed)`, `09_rank_failure_gloo_n2` `0`, `10_rank_failure_nccl_n2` `124` (`4` PIDs); session exit `1`; `0` leftover, `0` GPU procs, `0` files left in `refs/` |
| Relative `--log-dir` path hygiene | `--log-dir mgc_logs` | `0` `$HOME` hits in `.done` markers or `summary.txt` |
| Driver unit tests (`tests/test_multi_gpu_check.py`) | Python `3.12.16` on 4× RTX 3060 host (single run) | **`105 passed in 66.95s`**, `0` leftover |
| Full `scripts/cuda_check.sh` | Python `3.12.16` on **one RTX 3060** (`cuda:0`) of the 4-GPU host | **`266 passed, 3 skipped`** device tests (`153.71 s`); **`840 passed, 4 skipped`** whole suite (`620.05 s`); full probe breakdown in §13 |

*(Checks from S4/S5 not re-run in S6: isolated Python `3.10` driver tests and unusable `--log-dir` exit `2`
checks last ran on Linux L4 at `d474f82` [S5, where `scripts/` matches `3ac7c6d`]; the no-`.git`
`content_hash` export-tree check last ran on Linux L4 at `8e8f490` [S4, before the `335cf7d` and `d474f82`
changes to `scripts/multi_gpu_check.py`]; and S6 ran `tests/test_multi_gpu_check.py` once rather than S5's
three-run repeat and omitted a standalone `pytest -m "not network"` run, superseded by `scripts/cuda_check.sh`'s
whole-suite run.)*

#### Resolution of the `82a21ca` tooling limitations and remaining limitations at `d474f82` / `3ac7c6d`

Status of the six limitations previously recorded at `82a21ca`:

1. *Parameters only, no buffers* → **Resolved (`fd24a2a`)**: `xr_eq` and `xr_max` now hash and compare all
   `model.named_parameters()` and all `model.named_buffers()` (casting `bool` buffers to `uint8` for NCCL
   broadcast). `param_l2`/`upd_l2` intentionally remain over `trainer._trainable` only.
2. *CountSketch `float64` floor* → **Resolved for `tiny_random` and LoRA/`5M head_only` (`fd24a2a`)**:
   default `--reference-mode auto` stores exact `float64` deltas for `dtype=float64` with
   `numel <= 1,000,000` (100% of `tiny_random` `float64` rows, all `5M` LoRA + `head_only`, and all `28M`
   LoRA). Larger vectors still using CountSketch (`k = 8,192`) emit `sketch_bound(...)` within `10×` of
   `floor_rel` and `sketch_underflow(sha_differs)` on hash mismatch with zero sketch diff (`0` flags fired in
   S2–S6).
3. *Timing columns (`step_ms`, `wall_s`)* → **Resolved (`fd24a2a`, `9ab347d`)**: `wall_s` starts after a
   1-scalar `all_reduce` barrier + CUDA sync; `step_ms` measures symmetric full per-step wall time on steps
   `3..K` minus `evaluate_val()` + `on_evaluate()`, and `val_s` is reported separately in `notes`.
4. *Provisional CUDA `moved_student` parity rows* → **Resolved (`fd24a2a`)**: `_build_moved_full_ft_params`
   and `_apply_moved_student_params` touch parameters only, leaving device-built `rope.inv_freq` untouched.
5. *Incomplete reference-mismatch guard* → **Resolved (`fd24a2a`, `9ab347d`, `ef92d53`)**: schema v5 checks
   zip CRCs, member byte sizes, base model digests (parameters + buffers), `frozen_base_sha256`, per-scenario
   resolved config + `moved_student` warmup digests, dataset fingerprints, `torch`/`CUDA`/`git_head`, and
   `script_sha256` covering `distributed_trainer_check.py`, `_dist_common.py`, and `_common.py`.
6. *Session script timeout, signals, and resume* → **Resolved (`a20d386`–`d474f82`)**: replaced by the single
   stdlib-only `scripts/multi_gpu_check.py` behind `scripts/multi_gpu_check.sh`, with content-addressed resume
   markers (`git` or `content_hash`), confirmed-live `/proc` + pure-BSD `ps` (`335cf7d`) token-tagged
   process-tree teardown, partial archive cleanup, preserved ignored-on-entry signals (`nohup`), `2 s`
   post-exit pipe-drain grace, relative `--log-dir` hygiene, pre-stage `--log-dir` writability checks and
   clean `OSError` reporting (`d474f82`), and broken-stdio exit-code preservation.

**Remaining tooling limitations at `d474f82` / `3ac7c6d`:**

- *Process-escape edge case:* as documented in `scripts/multi_gpu_check.py`, a descendant that strips
  `_TIMESTRAL_STAGE_TOKEN` from its environment and outlives its parent (being reparented outside the tracked
  PPID tree), or runs on a system where neither `/proc` nor `ps` can inspect processes (note that on Linux
  `procps-ng` `ps` itself reads `/proc`, so an OS-level unmounted or inaccessible `/proc` cannot be worked
  around via `ps`), can escape tree termination (neither `torchrun` nor PyTorch `DataLoader` workers do this).
- *Non-reproducible compiled CUDA parity:* `compile_default_*` rows on CUDA varied run-to-run against their
  one-process compiled reference across single-L4 sessions S2–S5 (cause uninvestigated; S6 ran once on 4× RTX
  3060, so separate-GPU run-to-run variation is unmeasured).
- *Float64 export-reload precision:* `distributed_trainer_check.py` uses `1e-6` rather than float64
  round-off for float64 export-reload forecast checks because `load_model(..., dtype=torch.float64)`
  truncates weights through float32 on load (§11, spec §12).
- *Large `full_ft` `float64` CountSketch entries under `auto`:* `5M`/`28M` `full_ft` and `28M` `head_only`
  exceed `1,000,000` parameters and still use CountSketch (`k = 8,192`, floor ~`8.4e-16` for `28M head_only`,
  ~`1.4e-15` for `5M full_ft`, ~`3.2e-15` for `28M full_ft`; pass `--reference-mode exact` when exact
  sub-`1e-14` measurement is needed on those rows).

**What S6 (and S1–S5) do NOT cover:** acceptance test T15 on NCCL (at `3ac7c6d`, `tests/distributed_harness.py`
still restricted to CPU/`gloo` via `_ALLOWED_CPU_BACKENDS`); NVLink or GPU P2P NCCL transport (S6 ran on PCIe gen 3
×16 with `can_device_access_peer` all `0`, NCCL `via SHM/direct`); cross-GPU run-to-run determinism (S6 was a
single separate-GPU session); `bf16` and released-weights parity tier calibration (across `5M`/`28M`/`123M`/
`209M`, seeds, devices, replay regimes, and cosine similarity); and 1/2/4-GPU throughput, VRAM, and scaling
measurements (the 6-step `B = 2` measuring driver is not a scaling benchmark).

**Not verified anywhere (Stage 3, spec §10 & §12):** Stages 1–2 multi-process training is verified on
CPU/`gloo` on macOS arm64 and Linux x86_64 (§17.1), single-GPU `scripts/cuda_check.sh` passed on one NVIDIA L4
(`d474f82`) and on one RTX 3060 of the 4-GPU host (`3ac7c6d`) with Python `3.12.16` (§13), and the hardened
Stage 3 driver and rank-failure probes ran both on one shared NVIDIA L4 (S4/S5) and on 4 separate RTX 3060
GPUs over PCIe/SHM (S6 at `3ac7c6d`, including `device="auto"` → `cuda:LOCAL_RANK` binding, 4-GPU
`02_prototype_parity_cuda`, 316 driver rows, and separate-GPU NCCL rank failure; §17.4). Still unverified: T15
execution on NCCL hardware (opt-in CUDA/NCCL harness support in `tests/distributed_harness.py` and the
`scripts/run_t15_nccl.py` runner are implemented and unit-tested without a GPU at `0d14138`, but 2-GPU and
4-GPU CUDA runs have not executed yet), NVLink / GPU P2P multi-GPU setups and
cross-GPU run-to-run determinism, `bf16` parity tiers and the cause of the large released-checkpoint `bf16`
differences and compiled CUDA run-to-run variation in §17.4, multi-GPU throughput/VRAM/scaling (including
per-parameter pack/unpack, 3–4 collectives per step, flat gradient buffer live through validation, and
per-rank `_frozen_base` VRAM in `full_ft` distillation), released-weights T15 bounds (with or without replay,
including cosine similarity), live concurrent `huggingface_hub==0.23.0` downloads (source reading only, §12),
and `F.linear` `requires_grad` dispatch on GPUs/CUDA.
