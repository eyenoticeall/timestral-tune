"""Long-format frame loader, windowing and time splits.

`TimeSeriesDataset.from_polars` (and PyArrow / pandas equivalents) takes an id column, a time
column and explicit covariate roles; produces forecast-origin windows; `split_by_time` applies a
cutoff plus an embargo of at least one forecast horizon.

Status: not implemented (scaffold only).

- Spec: `docs/SPEC.md` §6.2, §6.3, §7.1
- Roadmap phase: 1
- Acceptance tests: T5, T12
- Planned public names: `TimeSeriesDataset`
"""
