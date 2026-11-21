# Phase 2 — Adaptation & Calibration

**Status:** Complete on branch `phase-2` (2026-11-21); unmerged and unpushed.
**Goal:** All five adaptation modes, both conformal calibrators, and exports that load in the official package.
**Spec sections:** `docs/SPEC.md` §3.2, §5.1, §5.2, §7.3, §7.4, §9 (T9–T11, T13), §10

The task checklist and exit criteria are in the spec, §10. This file holds working notes for whoever
implements the phase. Add to it as you learn things; do not copy the checklist here.

## Notes for the implementer

- **LoRA slices.** In `wQKV.weight`, rows `[0, D)` are query, `[D, 2D)` key, `[2D, 3D)` value. The
  query/key order is from the code only; the value third is measured (`docs/VERIFICATION.md` §3).
  `LoRASliceLinear` fuses `lora_A` projections across adapted slices of `wQKV` into one `F.linear`, uses
  out-of-place `chunk(3) + cat` for autograd and `torch.compile` safety, and strips `_orig_mod.` keys in
  `merged_state_dict` / `extract_lora_state_dict`.
- **Adapter naming.** Wrapping a linear changes parameter names, so `merged_state_dict` always exports
  merged weights under exact upstream names without mutating a live unmerged trainer. T11 guards this.
- **Single-row guard** is measured, not theoretical: query and key perturbations change nothing on
  univariate input (`validate_lora_for_rows`, enforced in both `inject_lora` and `TimestralTrainer`).
- **Conformal formulas** are quoted from the papers in spec §5.2; the PDFs are in `papers/` locally.
  Both `ConformalCalibrator` (CQR) and `OnlineConformalCalibrator` (ACI) run in float64 NumPy internally
  (safe on Apple GPU) and return the caller's array/tensor type, device, and dtype.
- **`config.json` must stay byte-compatible** with upstream; metadata goes in `tune_metadata.json`.
  Because upstream's `resolve_checkpoint` only downloads `config.json` and `model.safetensors`, `export.py`
  copies `LICENSE` from a local base directory or fetches it from Hub via `hf_hub_download`.
- **`bf16` and `torch.compile` on CUDA:** verified on NVIDIA L4 (`512 passed, 2 skipped`). `bf16` alone
  gives ~2× speed-up on 28M; `bf16 + compile("default")` gives 3.3× speed-up and 45% less VRAM.
  CPU `torch.compile` requires a C++ compiler (`g++` or `clang++`).

## Open questions

- Default LoRA rank and which slices to enable by default (settled by Phase 3 sweeps).

## Log of what was done

- 2026-11-21 (`c1396f8`): added `LoraConfig` and slice LoRA (`models/adapter.py`), T9 and T10 passing on
  tiny, 5M, and 28M.
- 2026-11-21 (`c179949`): added `lora_variate`, `lora_temporal`, `full_ft`, fused CUDA `AdamW`, `bf16`
  autocast on CUDA, and optional `torch.compile` in `TimestralTrainer`.
- 2026-11-21 (`01fdd53`): implemented `ConformalCalibrator` (CQR) and `OnlineConformalCalibrator` (ACI)
  in `models/conformal.py`, T13 passing.
- 2026-11-21 (`1d08f02`): implemented `export_checkpoint` (`export.py`) and `TimestralTrainer.export()`,
  T11 passing on tiny, 5M, and 28M across all four modes.
- 2026-11-21 (`3fe5bc9`): verified across PyTorch 2.4/2.8/2.14/2.15 locally (510 passed, 4 skipped; 411/26
  on 2.4.0) and on NVIDIA L4 CUDA (**512 passed, 2 skipped**), resolved D15 via 20-step CUDA trajectories,
  and verified verbatim Hub `LICENSE` export on `mistralai/Timestral-5M`.
- 2026-11-21 (review-fix pass, `4ec2e46`..`352f832`): addressed all Important findings from a 4-agent review
  (`load_lora`, exact pre-merge snapshot in `unmerge`, exact O(n) CQR order statistic, repaired-interval ACI
  `err_t` + `bisect`-sorted window, strict `from_dict`, CPU `best_state`, `evaluate()` shape guard, local
  LoRA RNG, content `dataset_fingerprint`, dirty-dir cleanup, unfitted-calibrator guard, fallback `LICENSE`
  warning, `--bf16`/`--compile` flags on `step_timing.py`). Local suite: **518 passed, 4 skipped**.
