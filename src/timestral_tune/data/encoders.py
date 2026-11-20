"""Leak-free categorical covariate encoding.

Per-window, context-only target encoding that reuses the vendored
`covariate_encoding.target_encode`; horizon codes are looked up in the context's table.

Status: not implemented (scaffold only).

- Spec: `timestral_tune_handoff_document.md` §6.3 rule 4
- Roadmap phase: 1
- Acceptance tests: T12
- Planned public names: `encode_categorical_window`
"""
