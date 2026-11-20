"""DHF batch collator.

Packs windows into the upstream row layout (one row per variate, `group_ids`, `role_ids`),
forward-fills targets, left-pads to a patch multiple, builds known-future tensors and the loss
weight mask. Must mirror vendored `TimestralForecaster._assemble_rows` exactly.

Status: not implemented (scaffold only).

- Spec: `timestral_tune_handoff_document.md` §3.3, §3.4, §4.2, §6.2
- Roadmap phase: 1
- Acceptance tests: T1, T2, T4, T8
- Planned public names: `DHFDataCollator`
"""
