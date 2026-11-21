"""Tests for the optional Lacuna integration (spec §7.5, decision D8)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

from timestral_tune import LoraConfig, TrainingConfig
from timestral_tune.integrations.lacuna import LacunaExporter, write_signal_frame
from timestral_tune.models import DEFAULT_QUANTILE_LEVELS
from timestral_tune.utils import MetricsSummary

from .helpers import make_toy_dataset

HAS_LACUNA = importlib.util.find_spec("lacuna") is not None


def test_core_package_does_not_import_lacuna():
    import timestral_tune

    assert timestral_tune.__version__
    assert "lacuna" not in sys.modules or HAS_LACUNA


def test_write_signal_frame_built_ins_callable_and_persistence(tmp_path):
    q = len(DEFAULT_QUANTILE_LEVELS)
    n, h = 6, 4
    # Non-uniform quantiles and horizon-varying predictions so swapping any signal_fn or horizon_index fails!
    base = np.array([q_val**2 * 10.0 for q_val in DEFAULT_QUANTILE_LEVELS], dtype=np.float64)[None, None, :]
    preds = np.broadcast_to(base, (n, h, q)).copy()
    preds += np.arange(n, dtype=np.float64)[:, None, None]
    preds += np.arange(h, dtype=np.float64)[None, :, None] * 0.5
    times = [datetime(2025, 1, i + 1, tzinfo=timezone.utc) for i in range(n)]
    insts = ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"]

    # Default median at horizon step 0 vs step 2.
    csv_path = tmp_path / "signals.csv"
    pq_path = tmp_path / "signals.parquet"
    frame_h0 = write_signal_frame(
        preds,
        timestamps=times,
        instruments=insts,
        signal_fn="median",
        horizon_index=0,
        path=csv_path,
        output_format="dict",
    )
    frame_h2 = write_signal_frame(
        preds, timestamps=times, instruments=insts, signal_fn="median", horizon_index=2, output_format="dict"
    )
    assert list(frame_h0.keys()) == ["time", "instrument", "signal"]
    med0 = float(base[0, 0, DEFAULT_QUANTILE_LEVELS.index(0.50)])
    np.testing.assert_allclose(frame_h0["signal"], med0 + np.arange(n, dtype=np.float64))
    np.testing.assert_allclose(frame_h2["signal"], med0 + np.arange(n, dtype=np.float64) + 1.0)
    assert csv_path.is_file() and csv_path.read_text(encoding="utf-8").splitlines()[0] == "time,instrument,signal"

    # All built-in spread reductions (iqr, spread_80, spread_90) give exact distinct values + Parquet write.
    for name, lo_q, hi_q in [("iqr", 0.25, 0.75), ("spread_80", 0.10, 0.90), ("spread_90", 0.05, 0.95)]:
        tbl = write_signal_frame(
            preds,
            timestamps=times,
            instruments=insts,
            signal_fn=name,
            horizon_index=None,
            path=pq_path if name == "spread_80" else None,
        )
        lo_i, hi_i = DEFAULT_QUANTILE_LEVELS.index(lo_q), DEFAULT_QUANTILE_LEVELS.index(hi_q)
        expected_spread = float(base[0, 0, hi_i] - base[0, 0, lo_i])
        np.testing.assert_allclose(tbl["signal"].to_pylist(), [expected_spread] * n)
    assert pq_path.is_file()

    custom = write_signal_frame(
        preds, timestamps=times, instruments=insts, signal_fn=lambda arr: arr[:, -1, -1], output_format="dict"
    )
    np.testing.assert_allclose(custom["signal"], float(base[0, 0, -1]) + np.arange(n, dtype=np.float64) + 1.5)

    with pytest.raises(ValueError, match="unsupported signal frame"):
        write_signal_frame(preds, timestamps=times, instruments=insts, path=tmp_path / "bad.json")


def test_lacuna_exporter_works_with_duck_typed_registry_or_raises_without_lacuna(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size, series=4, covariates=True)

    class _FakeRecord:
        def __init__(self, trial_id: str, kwargs: dict):
            self.trial_id = trial_id
            self.kwargs = kwargs

    class _FakeRegistry:
        def __init__(self):
            self.records: list[_FakeRecord] = []
            self.selections_list: list[dict] = []

        def record(self, **kwargs):
            rec = _FakeRecord(f"trial_{len(self.records)}", kwargs)
            self.records.append(rec)
            return rec

        def record_selection(self, **kwargs):
            self.selections_list.append(kwargs)
            return kwargs

        def attempts(self):
            return tuple(self.records)

        def selections(self):
            return tuple(self.selections_list)

    fake = _FakeRegistry()
    exp = LacunaExporter(fake)
    cfg = TrainingConfig(model_name_or_path="mistralai/Timestral-5M", mode="lora_variate", learning_rate=3e-4)
    lora = LoraConfig(r=8, lora_alpha=16, axes=("variate",), targets=("q", "v"))
    summary = MetricsSummary(
        pinball=0.1,
        wql=0.2,
        mase=0.8,
        coverage_80=0.81,
        width_80=1.2,
        coverage_90=0.91,
        width_90=1.8,
        num_series=4,
        num_scored=64,
    )
    r1 = exp.record_run(
        metric=summary.wql,
        metric_name="wql",
        training_config=cfg,
        lora_config=lora,
        dataset=dataset,
        metrics=summary,
    )
    r2 = exp.record_run(metric=0.25, metric_name="wql", parameters={"mode": "head_only"}, dataset=dataset)
    exp.record_selection(
        [r1.trial_id, r2.trial_id], [r1.trial_id], metric="wql", direction="minimize", used_holdout=True
    )
    assert len(exp.attempts()) == 2 and len(exp.selections()) == 1
    assert r1.kwargs["data_fingerprint"].startswith("sha256:")
    assert r1.kwargs["code_fingerprint"].startswith("timestral-tune:")
    assert r1.kwargs["parameters"]["lora"]["r"] == 8

    if not HAS_LACUNA:
        with pytest.raises((ImportError, RuntimeError)):
            LacunaExporter("missing-lacuna")


@pytest.mark.skipif(not HAS_LACUNA, reason="requires optional extra timestral-tune[lacuna]")
def test_lacuna_live_experiment_registry_and_signal_study(tiny_forecaster, tmp_path):
    import lacuna
    import polars as pl

    dataset = make_toy_dataset(tiny_forecaster.patch_size, series=4, covariates=True)
    with LacunaExporter("timestral-phase3-test", path=tmp_path / "registry.sqlite") as exp:
        r1 = exp.record_run(
            metric=0.11,
            metric_name="wql",
            training_config=TrainingConfig(model_name_or_path="mistralai/Timestral-5M", mode="lora_variate"),
            lora_config=LoraConfig(r=8, lora_alpha=16, axes=("variate",), targets=("q", "v")),
            dataset=dataset,
        )
        r2 = exp.record_run(
            metric=0.16,
            metric_name="wql",
            training_config=TrainingConfig(model_name_or_path="mistralai/Timestral-5M", mode="head_only"),
            dataset=dataset,
        )
        sel = exp.record_selection(
            [r1.trial_id, r2.trial_id],
            [r1.trial_id],
            metric="wql",
            direction="minimize",
            used_holdout=True,
        )
        assert len(exp.attempts()) == 2 and len(exp.selections()) == 1
        assert sel.selected_trial_ids == (r1.trial_id,)

    # Feed write_signal_frame output directly into lacuna.SignalStudy.
    dates = [datetime(2025, 1, d, tzinfo=timezone.utc) for d in range(1, 25)]
    n = 72
    q = len(DEFAULT_QUANTILE_LEVELS)
    rng = np.random.default_rng(0)
    preds = np.sort(rng.normal(size=(n, 4, q)), axis=-1)
    sig_frame = write_signal_frame(
        preds,
        timestamps=dates * 3,
        instruments=["AAA"] * 24 + ["BBB"] * 24 + ["CCC"] * 24,
        signal_fn="median",
        output_format="pyarrow",
    )
    prices = pl.DataFrame(
        {
            "time": dates * 3,
            "instrument": ["AAA"] * 24 + ["BBB"] * 24 + ["CCC"] * 24,
            "close": [100.0 + 0.3 * i + float(rng.normal(scale=0.5)) for i in range(n)],
        }
    )
    study = lacuna.SignalStudy(signal=sig_frame, prices=prices, horizons=("1D",))
    ic_res = study.ic()
    assert ic_res is not None
