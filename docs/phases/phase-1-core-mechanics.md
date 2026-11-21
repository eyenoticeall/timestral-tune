# Phase 1 — Core Mechanics & Verification

**Status:** Built and passing. One verification item is open: T14 has never run on CUDA.
**Goal:** A training forward pass and loss that provably match upstream inference, a leak-free data layer, and the first end-to-end `head_only` run.
**Spec sections:** `docs/SPEC.md` §3.3, §3.4, §4, §6.2, §6.3, §9 (T1–T8, T12), §10

The task checklist and exit criteria are in the spec, §10. This file holds working notes for whoever
maintains or extends the phase.

## How the pieces fit

- `data/dataset.py` yields raw `Window`s (views). `data/collator.py` turns a list of them into a
  `TrainingBatch` (`data/batch.py`), the packed row layout upstream uses.
- `models/forward.py::training_forward` returns normalised quantiles; `target_frames` returns the per-patch
  frames; `trainer/step.py::compute_loss` composes forward, frame, normalised targets and pinball loss.
  Trainer, tests and probes all go through `compute_loss`, so they cannot drift apart.
- `utils/validation.py` holds the acceptance checks as functions; the acceptance tests call them.
- `trainer/trainer.py` is the loop. `_select_trainable` is where Phase 2 adds modes.

## Things worth knowing before changing this code

- **Parity is the safety net.** T1–T3 and `tests/test_collator.py` compare against upstream's own code on
  the same inputs. If you change the forward or the collator, run them on 5M and 28M, not only the tiny
  model. Expect agreement at float32 round-off; investigate anything looser.
- **Do not assert bit-identity across PyTorch versions.** T1/T2 are bit-identical on 2.4.0 and 2.15.0 but
  one case differs by 8e-10 on 2.8.0.
- **The Apple GPU has no float64.** Every bug found on that device in this phase was a float64 conversion
  executed before moving to CPU: `.to(device, torch.float64)` in one call, `.double()` before `.cpu()`,
  `loss.double().item()`. Move first, then widen. Only running on the device catches these.
- **Batches are padded further than upstream pads** (to one context length per dataset). That is safe
  because leading unobserved patches are never attended to; `test_extra_left_padding_does_not_change_the_forecast`
  guards it.
- **Values at unobserved positions do not matter to the model** (they are zeroed after scaling and carry
  zero weight in the statistics), which is why forward-filling inside a truncated window matches upstream.
- **Pinball losses are not comparable between the two target frames.** Each is in its own units.
- **The tiny random model** (`tests/helpers.py`) makes structural tests network-free. It does not show the
  float32 gradient inaccuracy the released weights do, so gradient tests need both.
- **Validation loss under `median_folded` moves its own units** as the model's medians change (spec §12).
  Early stopping compares those losses; prefer raw-space metrics for model selection once they exist.
- **Floor runs on macOS.** PyTorch 2.4.0 hangs at exit after a DataLoader worker has run
  (`scripts/probes/worker_exit_hang.py`); the suite skips the one worker test there.
- **`split_by_time`**: training gets nothing at or after the cutoff; the embargo is an extra gap.

## Open questions

- T14 on CUDA (no hardware so far).
- The cause of the float32 gradient inaccuracy on released weights (`docs/VERIFICATION.md` §6).
- The target-frame evidence is synthetic, univariate and fully observed. A check on real data with
  covariates would strengthen decision D2.

## Log of what was done

- 2026-11-21: All Phase 1 items built. Evidence: 268 passed, 4 skipped on PyTorch 2.15.0, 2.14.1
  (Python 3.10) and 2.8.0; 231 passed, 22 skipped on 2.4.0. T1–T8 and T12 pass on 5M and 28M. Target-frame check run and
  recorded; D2 confirmed. Not done: CUDA.
