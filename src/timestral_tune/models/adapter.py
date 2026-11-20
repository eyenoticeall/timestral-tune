"""Slice LoRA for the fused attention projections.

Low-rank adapters on row slices of `wQKV` (query / key / value) and optionally on `wO` and
`_gate`, selected by attention axis via each module's `is_time`. Includes merge / unmerge and the
single-row guard for variate attention.

Status: not implemented (scaffold only).

- Spec: `timestral_tune_handoff_document.md` §3.2, §5.1
- Roadmap phase: 2
- Acceptance tests: T9, T10
- Planned public names: `inject_lora`, `merge_lora`, `LoRASliceLinear`
"""
