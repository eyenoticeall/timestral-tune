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
| T1/T2: denormalised training forward against `predict_block`, both layouts, including covariates with none known in advance | Bit-identical on PyTorch 2.4.0 and 2.15.0. One tiny-model case differs by 8e-10 on 2.8.0. Bound: 1e-6 | all |
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
`uv run pytest tests/acceptance/test_device_parity.py -k "gradients" -s` prints each value):

| Tier | Bound | CPU, worst case | Apple GPU, worst case |
|---|---|---|---|
| Tiny random model, as shipped and with explicit attention | 1e-5 | 4.1e-7 | 4.5e-7 |
| Released, as shipped | 5e-2 | 3.4e-3 (5M, two targets with covariates) | 2.1e-3 (5M, past-only covariates) |
| Released, explicit attention | 5e-3 | 1.2e-3 (5M, past-only covariates) | 2.1e-3 (5M, past-only covariates) |
| `head_only` (everything but the head frozen), all models | 1e-5 | 2.3e-6 | 2.8e-6 |

One minus the cosine similarity with the reference, as shipped: at most 5.7e-6 on CPU and 2.1e-6 on the
Apple GPU.

What narrows it down, and what does not (probe, covariates batch unless stated):

| Experiment | Result |
|---|---|
| Upstream's attention call replaced by explicit matmul and softmax, 5M | CPU 1.2e-3 → 5.9e-5; Apple GPU 7.7e-5 → 7.6e-5 |
| Same, 28M | CPU 9.1e-4 → 3.9e-4; Apple GPU 1.8e-4 → 3.7e-5 |
| Standalone attention call, fp32 vs fp64 gradient, max logit 5.5 | stock 5.5e-7, explicit 3.5e-7 |
| Same, max logit 88 | stock 4.0e-6, explicit 3.7e-6 |
| Same, max logit 1,409 | stock 5.0e-5, explicit 4.8e-5 |
| A float64-weights reference with upstream's float32 RMSNorm, against the reference with float64 RMSNorm, over the twelve cells above | 6e-8 to 1.3e-4 (largest: 28M covariates) |

Reading. The error depends on the weights (a random model shows none) and appears on every device. Two
sources are identified:

1. The float32 backward pass of upstream's attention call, on CUDA (§13) and CPU. On the Apple GPU the swap
   changes little.
2. Upstream's RMSNorm, which computes in float32 whatever the model's dtype. On T14's 5M scenario with
   past-only covariates this moved the float64-weights reference itself by 3.8e-3 (§11), enough to fail a
   check whose bound is 5e-3. All references now compute RMSNorm in float64 (`tests/helpers.py` and
   `scripts/probes/_common.py`, by replacing the method while the reference runs; vendored files are not
   edited). Numbers in this section were re-measured with that reference on 2026-11-21. The CUDA numbers in
   §13 were not: they used the earlier reference.

What remains with explicit attention and the better reference (at most 1.2e-3 on CPU, 2.1e-3 on the Apple
GPU) is unexplained beyond being float32 arithmetic through the same layers.

`head_only` is not affected: with the body frozen the backward pass never enters attention or the body's
norms. Not yet run on CUDA. Forward outputs are unaffected (about 3e-7 from the reference, §11).

Consequence: T14 compares gradients with the reference on every device, in three tiers plus the `head_only`
check (spec D12).

## 7. Training-step timing, CPU against the Apple GPU

Probe: `uv run python scripts/probes/step_timing.py --device mps`. The library's own step: `compute_loss`
(normalised forward, median-folded frame, pinball loss), backward over all parameters, AdamW. Context 1,024
steps, horizon 96, batch of 32 series, fp32. Median of five steps after a warm-up step.

| Model | Batch | CPU | Apple GPU | Speed-up | GPU memory allocated |
|---|---|---|---|---|---|
| 5M | univariate | 104 ms | 53 ms | 1.9× | 0.3 GiB |
| 5M | 5 rows per series, with covariates | 907 ms | 474 ms | 1.9× | 2.3 GiB |
| 28M | univariate | 317 ms | 192 ms | 1.6× | 2.3 GiB |
| 28M | 5 rows per series, with covariates | 6,216 ms | 1,957 ms | 3.2× | 7.3 GiB |

Timings vary between runs. Across four runs on this machine the speed-up ranged from 1.4× to 3.2×. Memory
is the driver allocation after the step, cumulative over the script, not a per-configuration peak.

## 8. Leakage, loss mask and training

Tests: `uv run pytest tests/acceptance tests/test_trainer.py`. Run on the tiny random model, 5M and 28M.

| Claim | Test | Result |
|---|---|---|
| Horizon target values changed in the raw frame leave inputs and predictions bit-identical | T4 | Pass |
| A past-only covariate changed after the origin changes nothing | T5 | Pass |
| Noise on targets changes no covariate row's hidden state or output; noise on past-only rows changes no known-future row | T6 | Exactly 0 |
| Noise on known-future horizon values changes the target forecast | T7 | Pass |
| Loss gradient is exactly zero at covariate rows, unobserved steps and horizon padding, under both frames | T8 | Pass |
| Categorical encodings are unchanged when data after the origin changes; a whole-series encoding would differ | T12 | Pass |
| With two targets, a past-only category that first occurs after the origin (sorting first, or in the middle) changes no input; the dataset-wide codes do change | T12, several targets | Pass |
| A category first seen after a `split_by_time` cutoff changes no training batch, with one or two targets | `test_dataset.py` | Pass (tiny model) |
| Series that start and end at different times: no training window holds a step at or after the cutoff | `test_dataset.py` | Pass |
| `head_only` on a fixed batch: loss falls, and every non-head parameter is bit-identical afterwards | `test_trainer.py` | Pass on CPU and the Apple GPU, with and without covariates |

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
| 2.4.0 | 3.12 | 331 passed, 26 skipped. Skipped: 21 accelerator tests, one DataLoader-worker test (below) and the 4 Phase 2 placeholders; 52 further Apple GPU cases are not generated at all. CPU passes everything else, including all of T14's gradient checks and the tests on 5M and 28M |
| 2.8.0 | 3.12 | 405 passed, 4 skipped, including every Apple GPU test |
| 2.14.1 | 3.10 | 405 passed, 4 skipped (`uv run --isolated --python 3.10 pytest`; NumPy 2.2.6, pandas 2.3.3, polars 2.0.0, pyarrow 25.0.1) |
| 2.15.0 | 3.13 | 405 passed, 4 skipped |

These are the counts after the review fixes of 2026-11-21, all on macOS. The CUDA run (§13) used Linux and
predates them.

The 4 skips common to every row are the Phase 2 acceptance placeholders.

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
| Float64-weights reference with upstream's float32 RMSNorm against the same with float64 RMSNorm, on T14's scenarios | 5M past-only covariates 3.8e-3; 28M covariates 1.4e-4; 5M univariate 1.5e-5 |
| The 5M past-only-covariates case with explicit attention, against the earlier (float32-RMSNorm) reference | CPU 2.6e-3, Apple GPU 5.9e-3; the error sat in layers 0 to 4 and the patch embedding, with the same distribution on both devices |

## 12. Facts read from sources, not measured

These `[H]` and `[U]` facts were read, not run: the four checkpoints' `config.json` values (spec §3.1), and
that the released `quantile_width_factor` is 3.0 while the code default is 2.0.

## 13. CUDA

Run: `bash scripts/cuda_check.sh` on the VM described at the top, at commit `ecbab9d`. It runs the
device-parametrised tests, the whole suite and the device probes. Everything added later has not run on
CUDA: T14's `head_only` gradient cases, the past-only-covariate scenario and the review fixes. The gradient
numbers below were measured against the earlier reference, which left upstream's RMSNorm in float32 (§6).

| Check | Result |
|---|---|
| T14 and all other device-parametrised tests | 127 passed, 2 skipped (the Apple GPU tests) |
| Whole suite | 290 passed, 6 skipped (4 Phase 2 placeholders, 2 Apple GPU tests) |
| `head_only` training on CUDA: loss falls, nothing outside the head changes | Pass (part of the suite) |
| Verifiers (`run_checks`) on CUDA | Pass (part of the suite) |

Forward, relative L2 difference from CPU (`device_parity.py --device cuda`):

| Case | 5M | 28M |
|---|---|---|
| `predict_block`, no covariates | 2.2e-7 | 3.0e-7 |
| `predict_block`, with covariates | 1.4e-6 | 2.2e-6 |
| `forecast`, horizon 96 | 1.4e-7 | 1.4e-7 |
| `forecast`, horizon 2,100 | 1.5e-5 | 1.8e-6 |
| `forecast`, numeric and categorical covariates | 1.2e-7 | 1.0e-7 |
| Autocast `bfloat16` (not a supported mode yet) | 1.5e-2 | 5.7e-2 |
| Autocast `float16` | 2.0e-3 | 5.9e-3 |

Float32 training-loss gradients against the float64-weights reference (`gradient_accuracy.py --device cuda`;
the CPU column is that VM's CPU):

| Model | Layout | CPU | CUDA |
|---|---|---|---|
| Tiny random | univariate / multivariate / covariates | 3.4e-7 / 3.0e-7 / 2.3e-7 | 3.2e-7 / 2.4e-7 / 1.8e-7 |
| 5M | univariate | 2.0e-3 | 1.2e-2 |
| 5M | multivariate | 1.2e-3 | 1.2e-2 |
| 5M | covariates | 9.2e-4 | 3.2e-3 |
| 28M | univariate | 5.9e-4 | 1.4e-3 |
| 28M | multivariate | 9.6e-4 | 2.3e-3 |
| 28M | covariates | 1.1e-3 | 1.5e-3 |

Across T14's scenarios on that VM: CUDA 2.2e-7 to 7.4e-3 as shipped (one minus the cosine at most 2.7e-5),
and at most 1.1e-4 with explicit attention; CPU at most 1.4e-3 as shipped and 1.2e-4 with explicit attention.
On the Mac, explicit attention gives at most 5.4e-4 on CPU and 1.2e-3 on the Apple GPU; both maxima are the
28M covariates scenario, and every other case is at 6e-5 or less.

Where the CUDA error comes from:

| Experiment | 5M | 28M |
|---|---|---|
| CUDA float32, as shipped (covariates batch) | 3.2e-3 | 1.5e-3 |
| CUDA float32, attention written out as matmul and softmax (same batch) | 7.1e-5 | 7.7e-5 |
| CPU float32, as shipped / explicit attention (same batch) | 9.2e-4 / 1.8e-4 | 1.1e-3 / 9.5e-5 |

The next four rows were measured once in a throwaway script on the VM (univariate batch) and are not
reproduced by a probe:

| Experiment | 5M | 28M |
|---|---|---|
| CUDA float32, as shipped | 1.2e-2 | 1.4e-3 |
| CUDA float32, explicit attention | 2.3e-5 | 4.1e-6 |
| CUDA with float64 weights | 1.2e-5 | 1.2e-5 |
| CUDA float32 with TF32 disabled; and with `TRITON_DEFAULT_FP_FUSION=0` | 1.2e-2 (unchanged) | 1.4e-3 (unchanged) |

Training-step timing (`step_timing.py --device cuda`; the CPU has only 4 threads, so the ratios flatter the
GPU compared with a workstation CPU):

| Model | Batch | CPU | CUDA | Speed-up | GPU memory allocated |
|---|---|---|---|---|---|
| 5M | univariate | 309 ms | 52 ms | 5.9× | 0.3 GiB |
| 5M | 5 rows per series, with covariates | 2,644 ms | 94 ms | 28× | 1.9 GiB |
| 28M | univariate | 1,163 ms | 79 ms | 15× | 1.9 GiB |
| 28M | 5 rows per series, with covariates | 10,833 ms | 380 ms | 29× | 5.7 GiB |

Environment requirement found on the way. With the VM's system Python 3.10 (PyTorch 2.14.1+cu130), every
CUDA test failed: PyTorch routed an operation in upstream's attention layer through a Triton kernel, and
Triton compiles a C helper on first use, which stopped at `fatal error: Python.h: No such file or directory`.
With a uv-managed Python 3.12, which ships its headers, everything passed. `scripts/cuda_check.sh` now
checks for the headers and a compiler first and stops with advice (verified on the VM: exit code 4).
