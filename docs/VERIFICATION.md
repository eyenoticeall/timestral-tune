# Verification Record

Measured evidence behind every **[V]** tag in `timestral_tune_handoff_document.md`. Each row names the
probe that reproduces it. Probes live in `scripts/probes/` and run the released checkpoints through the
vendored upstream code.

When you add a `[V]` claim to the handoff, add its probe and a row here in the same change.

## Environment of the recorded runs

| | |
|---|---|
| Date | 2026-11-21 |
| Machine | Apple M4, 10 CPU threads, 16 GB memory |
| PyTorch | 2.15.0 for all probes. The scaffold tests also passed on Python 3.10 with PyTorch 2.14.1, the only run below 2.15 |
| Python | 3.13 |
| Checkpoints | Timestral-5M and Timestral-28M at the Hub revisions pinned in `docs/UPSTREAM.md` |

Not yet verified anywhere: CUDA, the PyTorch 2.4 floor, the 123M and 209M checkpoints, Linux.

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

| Claim | 5M | 28M |
|---|---|---|
| Gradient of a target-only loss w.r.t. the past-only covariate's context (sum of abs.) | 8.5e-1 | 8.4e-1 |
| … w.r.t. the known-future covariate's context | 5.3e0 | 2.5e0 |
| … w.r.t. the known-future covariate's horizon values | 1.1e0 | 2.1e-1 |
| Noise on the target's context → max change in (target, past-only, known-future) outputs | 14.0, 0, 0 | 9.4, 0, 0 |
| Noise on the past-only covariate → same | 0.75, 6.3, 0 | 0.77, 15.2, 0 |
| Noise on the known-future covariate → same | 2.3, 3.3, 41.5 | 2.2, 4.8, 8.3 |

What this shows: covariate inputs receive non-zero gradients from a target-only loss (so "zero gradients
on covariates" is not a valid leakage test); targets never influence covariate rows; past-only rows never
influence known-future rows. The zeros are exact.

## 3. Single-row variate attention

Probe: `model_facts.py`. A univariate series with no covariates; random noise added to slices of
`wQKV.weight` in every variate-attention module.

| Perturbed rows | Max output change, 5M | 28M |
|---|---|---|
| Rows `[0, 2D)` (query and key) | 0 (exact) | 0 (exact) |
| Rows `[2D, 3D)` (value) | 4.9 | 16.3 |

This confirms the last third of the rows is the value projection and the first two-thirds are query and
key together. The order of query and key within the first two-thirds is from the code only
(`layers.py`, `_qkv`).

## 4. Apple GPU (MPS) parity

Probe: `uv run python scripts/probes/device_parity.py --device mps --size 5M` (and `--size 28M`).
The float64 statistics run on CPU via `install_cpu_offload` in `scripts/probes/_common.py`. Without that
offload, `predict_block` on MPS raises `TypeError: Cannot convert a MPS Tensor to float64`.

Forward, relative L2 difference from CPU:

| Case | 5M | 28M |
|---|---|---|
| `predict_block`, no covariates | 2.0e-7 | 2.7e-7 |
| `predict_block`, with covariates | 1.8e-6 | 3.4e-6 |
| `forecast`, horizon 96 | 1.1e-7 | 1.3e-7 |
| `forecast`, horizon 2,100 (block-autoregressive) | 2.3e-5 | 1.4e-6 |
| `forecast`, numeric and categorical covariates | 9.1e-8 | 9.6e-8 |

Gradients of the full training loss (normalised forward, median-folded frame, pinball), relative L2
difference from a float64-weights reference:

| | 5M | 28M |
|---|---|---|
| MPS fp32 | 1.3e-5 | 8.1e-5 |
| CPU fp32 | 4.0e-3 | 9.4e-4 |
| Cosine similarity between CPU fp32 and MPS fp32 gradients | 0.9999923 | 0.9999996 |

The reference uses float64 weights and inputs on CPU; upstream's RMSNorm and RoPE still compute in
float32 internally, so it is not a pure float64 computation.

Autocast on MPS, forward relative difference from CPU fp32 (gradients finite in all cases):

| | 5M | 28M |
|---|---|---|
| `bfloat16` | 1.4e-2 | 5.9e-2 |
| `float16` | 2.2e-3 | 7.5e-3 |

## 5. Training-step timing, CPU against MPS

Probe: `uv run python scripts/probes/step_timing.py --device mps`. The full step: normalised forward,
median-folded frame with its CPU round trip, pinball loss, backward, AdamW. Context 1,024 steps, horizon
96, batch of 32 series, fp32. Median of five steps after a warm-up step.

| Model | Batch | CPU | MPS | Speed-up | MPS memory allocated |
|---|---|---|---|---|---|
| 5M | univariate | 109 ms | 55 ms | 2.0× | 0.3 GiB |
| 5M | 5 rows per series, with covariates | 1,027 ms | 576 ms | 1.8× | 3.3 GiB |
| 28M | univariate | 330 ms | 206 ms | 1.6× | 3.3 GiB |
| 28M | 5 rows per series, with covariates | 5,113 ms | 2,528 ms | 2.0× | 7.3 GiB |

Timings vary between runs. Across three runs on this machine the speed-up ranged from 1.4× to 2.7×.
Memory is the driver allocation after the step, cumulative over the script, not a per-configuration peak.

Autocast gave no speed-up on MPS in a one-off run (see §7); the timing probe does not yet repeat this.

## 6. The CPU fp32 gradient gap (unexplained)

Probe: `uv run python scripts/probes/cpu_gradient_gap.py --size 5M` (and `--size 28M`).

| Experiment | 5M | 28M |
|---|---|---|
| 1. Stock attention call: CPU fp32 gradient vs float64-weights reference | 4.0e-3 | 9.4e-4 |
| 2. Attention replaced by explicit matmul and softmax, same model | 8.9e-6 | 8.8e-5 |
| 3. Standalone attention call, fp32 vs fp64 gradient | 5.4e-7 | 5.4e-7 |

Forward outputs on CPU fp32 agree with the reference to about 3e-7, so only the backward pass is affected.
Replacing the attention call removes the gap inside the model, yet the same call is accurate in isolation.
In a one-off run, changing only RMSNorm's `rsqrt` to a division by `sqrt` also moved the 5M gap (see §7).
No single cause is established. Tracked in handoff §12.

Practical consequence: device gradient checks (acceptance test T14) compare against the float64-weights
reference, never against CPU fp32.

## 7. One-off results not covered by a probe

These were measured once in throwaway scripts during the 2026-11-21 session and are cited in the handoff.
They are not reproduced by anything in `scripts/probes/`. Re-measure before relying on them, and move
them into a probe if they start to matter.

| Result | Value |
|---|---|
| Official package on MPS without the CPU offload | `predict_block` raises `TypeError: Cannot convert a MPS Tensor to float64` |
| Forward agreement with the float64-weights reference (5M) | CPU fp32 3.1e-7, MPS fp32 4.3e-7 |
| Autocast speed on MPS, 28M covariate batch | fp32 1,778 ms; `bfloat16` 1,731 ms; `float16` 1,732 ms (no speed-up) |
| Training step at context 4,096, univariate, batch 16 | 5M: CPU 138 ms, MPS 93 ms. 28M: CPU 526 ms, MPS 347 ms |
| RMSNorm via `sqrt` instead of `rsqrt` (5M) | CPU fp32 gradient gap moved from 3.9e-3 to 1.5e-3 |
| `bfloat16` autocast called with an empty block on MPS | Enters without error (says nothing about correctness) |

## 8. Facts read from sources, not measured

For completeness, these `[H]` and `[U]` facts were read, not run: the four checkpoints'
`config.json` values (handoff §3.1), and that the released `quantile_width_factor` is 3.0 while the code
default is 2.0.
