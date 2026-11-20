"""Checkpoint loading at pinned Hub revisions.

Wraps the vendored `load_model` / `TimestralForecaster.from_pretrained`, always reading
hyperparameters from the checkpoint's `config.json`, and returns the device-shimmed subclasses of
`models/device.py`. `load_forecaster(path, device=...)` is the inference entry point of §7.2.

Status: not implemented (scaffold only).

- Spec: `docs/SPEC.md` §3.1, §5.4, §7.2
- Roadmap phase: 0
- Acceptance tests: T11, T14
- Planned public names: `load_model`, `load_forecaster`
"""
