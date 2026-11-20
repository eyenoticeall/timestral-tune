"""Training forward pass in normalised space.

Runs the two-stream transformer for both upstream layouts and returns normalised quantiles at the
horizon prediction tokens, plus the target frame (`median_folded` by default, `true_values` for
ablation). A working prototype is `norm_forward` / `real_loss` in `scripts/probes/_common.py`.

Status: not implemented (scaffold only).

- Spec: `timestral_tune_handoff_document.md` §3.4, §4.3, §5.4
- Roadmap phase: 1
- Acceptance tests: T1, T2, T3, T4, T6, T7
- Planned public names: `training_forward`, `target_frames`
"""
