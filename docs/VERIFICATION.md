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

Not verified anywhere: CUDA, and the 123M and 209M checkpoints. Linux has run only the network-free tests,
in CI.

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
Run on the tiny random model, 5M and 28M.

| Claim | Measured |
|---|---|
| T1/T2: denormalised training forward against `predict_block`, both layouts | Bit-identical on PyTorch 2.4.0 and 2.15.0. One tiny-model case differs by 8e-10 on 2.8.0. Bound: 1e-6 |
| T3: median-folded frame against `_refine_block`, and the full pipeline against `forecast()` | 1e-7 to 2e-7 relative. Bound: 1e-5 |
| Collated batch against upstream's `_assemble_rows`, `_encoding_target`, `_encode_categorical` | Every tensor exactly equal, including missing values, a partial final patch, categoricals and several targets |
| Extra left padding (1 and 3 patches beyond upstream's), both layouts | Forecast unchanged within 1e-5 |
| Frame → dataset → collator → forward → frame, against `forecast()` on the same raw values | Within 1e-5, for polars, pandas and pyarrow |

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

## 6. Float32 gradient accuracy (cause unexplained)

Probe: `uv run python scripts/probes/gradient_accuracy.py --device mps`.

Relative L2 error of the float32 training-loss gradient against a float64-weights reference. The reference
uses float64 weights and inputs on CPU; upstream's RMSNorm and RoPE still compute in float32 internally, so
it is not a pure float64 computation.

| Model | Layout | CPU | Apple GPU |
|---|---|---|---|
| Tiny random | univariate | 5.0e-7 | 3.9e-7 |
| Tiny random | multivariate | 3.5e-7 | 3.3e-7 |
| Tiny random | covariates | 2.7e-7 | 3.0e-7 |
| 5M | univariate | 1.1e-3 | 9.8e-6 |
| 5M | multivariate | 2.1e-3 | 4.6e-5 |
| 5M | covariates | 1.2e-3 | 5.6e-5 |
| 28M | univariate | 8.1e-4 | 5.9e-6 |
| 28M | multivariate | 5.0e-4 | 1.4e-5 |
| 28M | covariates | 9.8e-4 | 4.5e-5 |

Across the scenarios of acceptance test T14 (other batches; `uv run pytest
tests/acceptance/test_device_parity.py -k gradients -s` prints each value): CPU 1.6e-7 to 3.4e-3, Apple GPU
2.2e-7 to 1.1e-3. The largest Apple GPU value is 28M with covariates. In an earlier one-off run of the same
scenarios, cosine similarity with the reference stayed above 0.999999 in every case.

What narrows it down, and what does not:

| Experiment | Result |
|---|---|
| CPU error with upstream's attention call replaced by explicit matmul and softmax, 5M | 1.2e-3 → 3.9e-5 |
| Same, 28M | 9.8e-4 → 2.6e-4 |
| Standalone attention call, fp32 vs fp64 gradient, max logit 5.5 | stock 5.5e-7, explicit 3.5e-7 |
| Same, max logit 88 | stock 4.0e-6, explicit 3.7e-6 |
| Same, max logit 1,409 | stock 5.0e-5, explicit 4.8e-5 |

Reading: the error depends on the weights (a random model shows none), appears on both devices, and is
usually 10 to 100 times larger on CPU. Swapping the attention implementation changes it inside the model,
but the same call is accurate in isolation at every logit scale tried, so no single cause is established.
Forward outputs are unaffected (about 3e-7 from the reference, §11).

Consequence: T14 compares gradients with the reference on every device, with two tiers (spec D12).

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
| 2.4.0 | 3.12 | 227 passed, 21 skipped. The skips are the Apple GPU tests and the Phase 2 placeholders. CPU passes everything, including T14's gradient check and the tests on 5M and 28M |
| 2.8.0 | 3.12 | 262 passed, 4 skipped, including every Apple GPU test |
| 2.15.0 | 3.13 | 262 passed, 4 skipped |

The 4 skips are the Phase 2 acceptance placeholders. Python 3.10 has run only the scaffold tests locally
(with PyTorch 2.14.1); CI runs the network-free tests on 3.10 to 3.13.

The Apple GPU needs PyTorch 2.8. `torch.Tensor.cummax` on that backend, which upstream's input assembly
calls:

| PyTorch | 2.4.0 | 2.5.1 | 2.6.0 | 2.7.1 | 2.8.0 | 2.9.1 |
|---|---|---|---|---|---|---|
| `cummax` on MPS | raises | raises | raises | raises | works | works |

Before the library refused the Apple GPU on older versions, the suite on 2.4.0 failed there and then hung
in one test. `resolve_device` now skips it under `"auto"` and raises for an explicit `"mps"`.

## 11. One-off results not covered by a probe or test

Measured once in throwaway scripts on 2026-11-21 and not reproduced by anything in the repository.
Re-measure before relying on them.

| Result | Value |
|---|---|
| Forward agreement with the float64-weights reference (5M) | CPU fp32 3.1e-7, MPS fp32 4.3e-7 |
| Cosine similarity of float32 gradients with the reference, T14 scenarios | Above 0.999999 on CPU and the Apple GPU |
| Autocast speed on MPS, 28M covariate batch | fp32 1,778 ms; `bfloat16` 1,731 ms; `float16` 1,732 ms (no speed-up) |
| Training step at context 4,096, univariate, batch 16 | 5M: CPU 138 ms, MPS 93 ms. 28M: CPU 526 ms, MPS 347 ms |
| RMSNorm via `sqrt` instead of `rsqrt` (5M) | CPU fp32 gradient error moved from 3.9e-3 to 1.5e-3 |

## 12. Facts read from sources, not measured

These `[H]` and `[U]` facts were read, not run: the four checkpoints' `config.json` values (spec §3.1), and
that the released `quantile_width_factor` is 3.0 while the code default is 2.0.
