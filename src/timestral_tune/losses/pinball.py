"""21-quantile pinball loss with weight-mask reduction.

Averages equally over quantile levels and scored positions; a batch with no scored position is
skipped rather than divided by an epsilon.

Status: not implemented (scaffold only).

- Spec: `timestral_tune_handoff_document.md` §4.1, §4.2
- Roadmap phase: 1
- Acceptance tests: T8
- Planned public names: `PinballLoss`
"""
