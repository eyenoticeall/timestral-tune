"""Configuration dataclasses.

`TrainingConfig` (model, mode, optimiser, precision, target frame, output), `LoraConfig`
(rank, alpha, axes, targets, dropout, single-row override) and `ReplayConfig` (replay fraction,
source, distillation weight).

Status: not implemented (scaffold only).

- Spec: `docs/SPEC.md` §5 (modes), §5.1 (LoRA), §5.3 (replay), §5.4 (trainer), §7.1 (API example)
- Roadmap phase: 1 (TrainingConfig), 2 (LoraConfig), 4 (ReplayConfig)
- Acceptance tests: none directly; exercised by all
- Planned public names: `TrainingConfig`, `LoraConfig`, `ReplayConfig`
"""
