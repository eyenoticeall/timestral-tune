"""Device shims: run upstream's float64 statistics on CPU.

Subclasses of the vendored model and forecaster that move tensors to CPU for the four float64
sites (model `scale`, forecaster `_cap_quantiles`, `_refine_block`, `_encode_categorical`) and call
the parent. Needed for the Apple GPU backend, which has no float64. A monkeypatched prototype of
the same idea lives in `scripts/probes/_common.py`.

Status: not implemented (scaffold only).

- Spec: `docs/SPEC.md` §3.5, §5.4
- Roadmap phase: 0
- Acceptance tests: T14
- Planned public names: `TuneTimestralModel`, `TuneTimestralForecaster`
"""
