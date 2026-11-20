"""Checkpoint export.

Writes the layout of handoff §7.4: an upstream-loadable `config.json` and merged
`model.safetensors`, the adapter files, conformal offsets, the `tune_metadata.json` sidecar, and
the weights LICENSE and NOTICE. Never adds keys to `config.json`.

Status: not implemented (scaffold only).

- Spec: `timestral_tune_handoff_document.md` §7.4, §8
- Roadmap phase: 2
- Acceptance tests: T11
- Planned public names: `export_checkpoint`
"""
