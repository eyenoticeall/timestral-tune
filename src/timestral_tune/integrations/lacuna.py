"""Optional Lacuna integration (Python 3.11+, `timestral-tune[lacuna]`).

Appends trial records to a `lacuna.ExperimentRegistry` and writes signal frames with the
column names `lacuna.SignalStudy` expects. Does not write Lacuna bundles and computes no Sharpe
ratios.

Status: not implemented (scaffold only).

- Spec: `timestral_tune_handoff_document.md` §7.5
- Roadmap phase: 3
- Acceptance tests: none yet
- Planned public names: `LacunaExporter`, `write_signal_frame`
"""
