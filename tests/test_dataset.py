"""Unit tests for `TimeSeriesDataset`: loading, validation, windowing, splits. Network-free."""

from __future__ import annotations

import datetime
import pickle
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from timestral_tune.data import SeriesData, TimeSeriesDataset, window_category_codes
from timestral_tune.models import denormalise_quantiles, target_frames, training_forward

from .helpers import FRAME_KINDS, make_columns, max_rel, role_kwargs, to_frame

FIELDS = ("timestamps", "targets", "past", "known", "categorical_past", "categorical_known")


def build(columns, kind="pandas", **overrides) -> TimeSeriesDataset:
    kwargs = {"context_length": 24, "forecast_horizon": 8, **role_kwargs(columns), **overrides}
    return TimeSeriesDataset.from_frame(to_frame(columns, kind), **kwargs)


def series_equal(a: TimeSeriesDataset, b: TimeSeriesDataset) -> bool:
    return len(a.series) == len(b.series) and all(
        np.array_equal(getattr(x, f), getattr(y, f), equal_nan=True) and x.series_id == y.series_id
        for x, y in zip(a.series, b.series, strict=True)
        for f in FIELDS
    )


def simple_series(steps: int, missing_from: int | None = None) -> SeriesData:
    targets = np.arange(steps, dtype=np.float32)[None]
    if missing_from is not None:
        targets[:, missing_from:] = np.nan
    return SeriesData(
        series_id="s",
        timestamps=np.arange(steps, dtype=np.int64),
        targets=targets,
        past=np.empty((0, steps), dtype=np.float32),
        known=np.empty((0, steps), dtype=np.float32),
        categorical_past=np.empty((0, steps), dtype=np.float64),
        categorical_known=np.empty((0, steps), dtype=np.float64),
    )


FULL = {"targets": 2, "past": 1, "known": 1, "categorical_past": 1, "categorical_known": 1}


# ---------------------------------------------------------------- loading


def test_the_three_frame_libraries_give_identical_datasets():
    columns = make_columns(0, **FULL)
    pandas, polars, arrow = (build(columns, kind) for kind in FRAME_KINDS)
    assert series_equal(pandas, polars) and series_equal(pandas, arrow)
    assert len(pandas) == len(polars) == len(arrow)
    assert pandas.rows_per_series == 6
    assert [s.series_id for s in pandas.series] == ["s0", "s1", "s2"]
    assert pandas.roles.targets == ("y0", "y1") and pandas.roles.categorical_known == ("ck0",)


@pytest.mark.parametrize("kind", FRAME_KINDS)
def test_named_constructors_check_the_frame_type(kind):
    columns = make_columns(0)
    frame = to_frame(columns, kind)
    kwargs = {"context_length": 24, "forecast_horizon": 8, **role_kwargs(columns)}
    constructors = {
        "pandas": TimeSeriesDataset.from_pandas,
        "polars": TimeSeriesDataset.from_polars,
        "arrow": TimeSeriesDataset.from_arrow,
    }
    assert len(constructors[kind](frame, **kwargs)) > 0
    other = next(k for k in FRAME_KINDS if k != kind)
    with pytest.raises(TypeError, match="needs a"):
        constructors[other](frame, **kwargs)


def test_row_order_does_not_matter():
    columns = make_columns(1, **FULL)
    order = np.random.default_rng(1).permutation(len(columns["id"]))
    shuffled = {name: values[order] for name, values in columns.items()}
    assert series_equal(build(columns), build(shuffled))


def test_categories_are_coded_by_sorted_value_and_nulls_are_missing():
    columns = make_columns(2, series=1, steps=40, categorical_known=1, missing=0.2)
    dataset = build(columns)
    assert dataset.categories["ck0"].tolist() == ["a", "b", "c", "d"]
    codes = dataset.series[0].categorical_known[0]
    labels = columns["ck0"]
    for code, label in zip(codes, labels, strict=True):
        assert np.isnan(code) if label is None else dataset.categories["ck0"][int(code)] == label


def test_single_series_without_an_id_column():
    columns = make_columns(3, series=1)
    del columns["id"]
    dataset = build(columns, id_col=None)
    assert len(dataset.series) == 1 and dataset.series[0].num_steps == 60


def test_integer_timestamps():
    columns = make_columns(4)
    columns["time"] = np.tile(np.arange(60) * 5 + 1000, 3)
    dataset = build(columns)
    assert dataset.series[0].timestamps.dtype == np.int64
    assert dataset.series[0].timestamps.tolist() == list(range(1000, 1300, 5))


# ---------------------------------------------------------- regular grid


def test_absent_time_steps_become_missing_values():
    columns = make_columns(5, series=2, steps=30, missing=0.0)
    keep = np.ones(60, dtype=bool)
    keep[[7, 8, 40]] = False
    dataset = build({name: values[keep] for name, values in columns.items()}, context_length=12, forecast_horizon=4)
    first, second = dataset.series
    assert first.num_steps == second.num_steps == 30
    assert np.isnan(first.targets[0, [7, 8]]).all() and np.isfinite(np.delete(first.targets[0], [7, 8])).all()
    assert np.isnan(second.targets[0, 10]) and np.isfinite(np.delete(second.targets[0], 10)).all()
    assert (np.diff(first.timestamps) == np.timedelta64(1, "h")).all()


def test_off_grid_timestamps_raise_and_regularize_false_accepts_them():
    columns = make_columns(6, series=1, steps=30)
    columns["time"] = columns["time"].copy()
    columns["time"][10] += np.timedelta64(7, "m")
    with pytest.raises(ValueError, match="not on the regular grid"):
        build(columns, context_length=12, forecast_horizon=4)
    dataset = build(columns, context_length=12, forecast_horizon=4, regularize=False)
    assert dataset.series[0].num_steps == 30


def test_calendar_irregular_data_needs_regularize_false():
    months = np.arange("2015-01", "2020-01", dtype="datetime64[M]").astype("datetime64[us]")
    columns = {"time": months, "y0": np.arange(months.size, dtype=np.float64)}
    kwargs = {"time_col": "time", "target_cols": ["y0"], "context_length": 12, "forecast_horizon": 3}
    with pytest.raises(ValueError, match="not on the regular grid"):
        TimeSeriesDataset.from_frame(to_frame(columns, "pandas"), **kwargs)
    dataset = TimeSeriesDataset.from_frame(to_frame(columns, "pandas"), regularize=False, **kwargs)
    assert dataset.series[0].num_steps == months.size
    assert not np.isnan(dataset.series[0].targets).any()


def test_explicit_freq_and_the_wrong_frequency_guard():
    columns = make_columns(7, series=1, steps=30, missing=0.0)
    half_hourly = build(columns, context_length=12, forecast_horizon=4, freq=np.timedelta64(30, "m"))
    assert half_hourly.series[0].num_steps == 59
    assert np.isnan(half_hourly.series[0].targets[0, 1::2]).all()
    assert (
        build(columns, context_length=12, forecast_horizon=4, freq=datetime.timedelta(hours=1)).series[0].num_steps
        == 30
    )
    with pytest.raises(ValueError, match="frequency looks wrong"):
        build(columns, context_length=12, forecast_horizon=4, freq=np.timedelta64(1, "s"))
    with pytest.raises(ValueError, match="timedelta"):
        build(columns, context_length=12, forecast_horizon=4, freq=3600)


# ------------------------------------------------------------- bad input


def test_duplicate_timestamps_raise():
    columns = make_columns(8, series=2, steps=20)
    columns["time"] = columns["time"].copy()
    columns["time"][25] = columns["time"][24]
    with pytest.raises(ValueError, match="more than one row at timestamp"):
        build(columns, context_length=8, forecast_horizon=4)


def test_null_ids_and_null_timestamps_raise():
    columns = make_columns(9, series=2, steps=20)
    bad_id = dict(columns, id=columns["id"].copy())
    bad_id["id"][3] = None
    with pytest.raises(ValueError, match="'id' holds 1 null"):
        build(bad_id, context_length=8, forecast_horizon=4)
    bad_time = dict(columns, time=columns["time"].copy())
    bad_time["time"][3] = np.datetime64("NaT", "us")
    with pytest.raises(ValueError, match="'time' holds null"):
        build(bad_time, context_length=8, forecast_horizon=4)


def test_non_numeric_and_infinite_value_columns_raise():
    columns = make_columns(10, series=1, steps=20, categorical_known=1)
    with pytest.raises(ValueError, match="'ck0' is not numeric"):
        build(columns, context_length=8, forecast_horizon=4, known_covariates=["ck0"], categorical_known_covariates=[])
    infinite = dict(columns, y0=columns["y0"].copy())
    infinite["y0"][4] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        build(infinite, context_length=8, forecast_horizon=4)
    huge = dict(columns, y0=columns["y0"].copy())
    huge["y0"][4] = 1e300
    with pytest.raises(ValueError, match="too large for float32"):
        build(huge, context_length=8, forecast_horizon=4)


def test_missing_columns_wrong_frame_types_and_bad_roles_raise():
    columns = make_columns(11, series=1, steps=20)
    with pytest.raises(ValueError, match=r"not found in the frame: \['nope'\]"):
        build(columns, context_length=8, forecast_horizon=4, target_cols=["nope"])
    with pytest.raises(TypeError, match="expected a polars.DataFrame"):
        TimeSeriesDataset.from_frame(columns, time_col="time", target_cols=["y0"], context_length=8, forecast_horizon=4)
    import polars

    with pytest.raises(TypeError, match="collect"):
        TimeSeriesDataset.from_frame(
            polars.DataFrame({"time": [1], "y0": [1.0]}).lazy(),
            time_col="time",
            target_cols=["y0"],
            context_length=8,
            forecast_horizon=4,
        )
    with pytest.raises(ValueError, match="one role each"):
        build(columns, context_length=8, forecast_horizon=4, past_covariates=["y0"])
    with pytest.raises(ValueError, match="at least one target"):
        build(columns, context_length=8, forecast_horizon=4, target_cols=[])
    with pytest.raises(ValueError, match="must hold integers or datetimes"):
        build(dict(columns, time=np.linspace(0, 1, 20)), context_length=8, forecast_horizon=4)


def test_time_zone_aware_timestamps_raise():
    import pandas

    frame = pandas.DataFrame(
        {"time": pandas.date_range("2024-01-01", periods=20, freq="h", tz="Europe/Paris"), "y0": np.arange(20.0)}
    )
    with pytest.raises(ValueError, match="time-zone aware"):
        TimeSeriesDataset.from_pandas(frame, time_col="time", target_cols=["y0"], context_length=8, forecast_horizon=4)


def test_pandas_nullable_dtypes_are_supported():
    import pandas

    frame = pandas.DataFrame(
        {
            "time": pandas.date_range("2024-01-01", periods=20, freq="h"),
            "y0": pandas.array([1, None] * 10, dtype="Int64"),
            "c": pandas.array(["a", None] * 10, dtype="string"),
        }
    )
    dataset = TimeSeriesDataset.from_pandas(
        frame,
        time_col="time",
        target_cols=["y0"],
        categorical_known_covariates=["c"],
        context_length=8,
        forecast_horizon=4,
    )
    assert np.isnan(dataset.series[0].targets[0, 1::2]).all() and (dataset.series[0].targets[0, ::2] == 1).all()
    assert np.isnan(dataset.series[0].categorical_known[0, 1::2]).all()


# -------------------------------------------------------------- windows


def test_origins_are_anchored_at_the_end_and_step_by_stride():
    dataset = TimeSeriesDataset([simple_series(50)], context_length=20, forecast_horizon=8, stride=10)
    origins = [dataset[i].origin for i in range(len(dataset))]
    assert origins == [2, 12, 22, 32, 42]
    assert dataset.stats.candidates == dataset.stats.kept == 5
    last = dataset[-1]
    assert last.target_future[0].tolist() == list(range(42, 50))
    assert last.target_context[0].tolist() == list(range(22, 42))
    assert dataset[0].context_steps == 2, "near the start a window has less context than context_length"
    assert dataset.origin_time(-1) == 42


def test_min_context_removes_early_origins():
    dataset = TimeSeriesDataset([simple_series(50)], context_length=20, forecast_horizon=8, stride=10, min_context=15)
    assert [dataset[i].origin for i in range(len(dataset))] == [22, 32, 42]


def test_windows_are_views_into_the_series():
    dataset = TimeSeriesDataset([simple_series(50)], context_length=20, forecast_horizon=8)
    window = dataset[-1]
    assert np.shares_memory(window.target_context, dataset.series[0].targets)
    assert np.shares_memory(window.target_future, dataset.series[0].targets)


def test_windows_without_observed_targets_are_dropped_and_counted():
    late_start = simple_series(60)
    late_start.targets[:, :30] = np.nan
    dataset = TimeSeriesDataset(
        [simple_series(60, missing_from=40), late_start], context_length=10, forecast_horizon=5, stride=5
    )
    stats = dataset.stats
    # First series (observed 0..39): origins 50 and 55 see no target in their context; 40 and 45 have
    # nothing to score. Second series (observed from 30): origins 5..30 see no target in their context.
    assert stats.candidates == 22
    assert stats.dropped_no_context == 2 + 6
    assert stats.dropped_unscored == 2
    assert stats.kept == len(dataset) == 12
    assert stats.series_without_windows == 0
    for i in range(len(dataset)):
        window = dataset[i]
        assert np.isfinite(window.target_context).any() and np.isfinite(window.target_future).any()


def test_no_usable_window_raises_with_the_reason():
    with pytest.raises(ValueError, match="no usable window"):
        TimeSeriesDataset([simple_series(5)], context_length=10, forecast_horizon=8)
    with pytest.raises(ValueError, match="no usable window"):
        TimeSeriesDataset([simple_series(30, missing_from=0)], context_length=10, forecast_horizon=8)


def test_indexing_errors_and_argument_validation():
    dataset = TimeSeriesDataset([simple_series(30)], context_length=10, forecast_horizon=5)
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    with pytest.raises(IndexError):
        dataset[-len(dataset) - 1]
    for bad in ({"context_length": 0}, {"forecast_horizon": 0}, {"stride": 0}, {"min_context": 0}):
        with pytest.raises(ValueError, match="must be positive"):
            TimeSeriesDataset([simple_series(30)], **{"context_length": 10, "forecast_horizon": 5, **bad})
    with pytest.raises(ValueError, match="at least one series"):
        TimeSeriesDataset([], context_length=10, forecast_horizon=5)


def test_available_lag_delays_a_past_covariate():
    columns = make_columns(12, series=1, steps=30, past=2, categorical_past=1, missing=0.0)
    plain = build(columns, context_length=10, forecast_horizon=4)
    lagged = build(columns, context_length=10, forecast_horizon=4, available_lag={"p1": 3, "cp0": 2})
    a, b = plain.series[0], lagged.series[0]
    assert np.array_equal(a.past[0], b.past[0])
    assert np.isnan(b.past[1, :3]).all() and np.array_equal(b.past[1, 3:], a.past[1, :-3])
    assert np.isnan(b.categorical_past[0, :2]).all() and np.array_equal(
        b.categorical_past[0, 2:], a.categorical_past[0, :-2]
    )
    with pytest.raises(ValueError, match="not a past-only covariate"):
        build(columns, context_length=10, forecast_horizon=4, available_lag={"y0": 1})
    with pytest.raises(ValueError, match="non-negative integer"):
        build(columns, context_length=10, forecast_horizon=4, available_lag={"p0": -1})


# ---------------------------------------------------------------- splits


@pytest.mark.parametrize("embargo", [0, 3])
def test_split_by_time_keeps_training_strictly_before_the_cutoff(embargo):
    columns = make_columns(13, series=3, steps=80, missing=0.0)
    dataset = build(columns, context_length=16, forecast_horizon=6, stride=2)
    cutoff = np.datetime64("2024-01-03T00:00")  # step 48
    train, validation = dataset.split_by_time(cutoff, embargo=embargo)
    assert len(train) > 0 and len(validation) > 0
    assert train.series is dataset.series, "splits share the underlying arrays"
    for i in range(len(train)):
        window = train[i]
        assert window.origin + train.forecast_horizon <= 48, "a training window reaches the cutoff"
    for i in range(len(validation)):
        assert validation[i].origin >= 48 + embargo
        assert validation.origin_time(i) >= cutoff
    straddling = len(dataset) - len(train) - len(validation)
    assert straddling > 0
    assert train.stats.kept == len(train) and validation.stats.kept == len(validation)


def test_split_accepts_strings_datetimes_and_integers_and_rejects_empty_sides():
    columns = make_columns(14, series=1, steps=60, missing=0.0)
    dataset = build(columns, context_length=16, forecast_horizon=6)
    by_string = dataset.split_by_time("2024-01-02T06:00")
    by_datetime = dataset.split_by_time(datetime.datetime(2024, 1, 2, 6))
    assert [len(d) for d in by_string] == [len(d) for d in by_datetime]

    integer = TimeSeriesDataset([simple_series(60)], context_length=16, forecast_horizon=6)
    train, validation = integer.split_by_time(30)
    assert train[-1].origin == 24 and validation[0].origin == 30

    with pytest.raises(ValueError, match="both must be non-empty"):
        dataset.split_by_time("2030-01-01")
    with pytest.raises(ValueError, match="both must be non-empty"):
        dataset.split_by_time("2000-01-01")
    with pytest.raises(ValueError, match="non-negative"):
        dataset.split_by_time("2024-01-02", embargo=-1)
    with pytest.raises(ValueError, match="not comparable"):
        integer.split_by_time("yesterday")


# ------------------------------------------------- loaders and end to end


# A DataLoader worker on macOS with PyTorch 2.4.0 leaves the interpreter hanging at exit, with or without
# this library (scripts/probes/worker_exit_hang.py). Not reproduced on 2.8.0 or newer; 2.5 to 2.7 are untested.
WORKERS_HANG_AT_EXIT = sys.platform == "darwin" and tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


@pytest.mark.skipif(WORKERS_HANG_AT_EXIT, reason="PyTorch below 2.8 on macOS: a DataLoader worker hangs the exit")
def test_dataset_pickles_and_works_in_a_worker_process(tiny_forecaster):
    columns = make_columns(15, **FULL)
    dataset = build(columns, context_length=16, forecast_horizon=6, stride=4)
    clone = pickle.loads(pickle.dumps(dataset))
    assert len(clone) == len(dataset) and series_equal(clone, dataset)

    collate = dataset.collator(tiny_forecaster.config)
    in_process = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate)))
    in_worker = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate, num_workers=1)))
    assert torch.equal(in_process.context, in_worker.context) and torch.equal(in_process.targets, in_worker.targets)
    in_process.validate(tiny_forecaster.patch_size)


@pytest.mark.parametrize("targets", [1, 2])
def test_a_category_first_seen_after_the_cutoff_changes_no_training_window(tiny_forecaster, targets):
    """Spec §6.3 rules 4 and 5: training inputs do not depend on which categories exist after the cutoff."""
    columns = make_columns(21, series=2, steps=60, **{**FULL, "targets": targets})
    step = np.arange(columns["time"].shape[0]) % 60
    changed = dict(columns, cp0=columns["cp0"].copy(), ck0=columns["ck0"].copy())
    changed["cp0"][step >= 44] = "A_sorts_first"
    changed["ck0"][step >= 44] = np.where(step[step >= 44] % 2 == 0, "A_sorts_first", "bb_sorts_in_the_middle")
    cutoff = columns["time"][44]

    def training_batch(frame_columns):
        dataset = build(frame_columns, context_length=16, forecast_horizon=6, stride=3)
        train, _ = dataset.split_by_time(cutoff)
        return train, train.collator(tiny_forecaster.config)([train[i] for i in range(len(train))])

    train, batch = training_batch(columns)
    changed_train, changed_batch = training_batch(changed)
    assert len(train) == len(changed_train) > 4
    assert train.categories["ck0"].tolist() != changed_train.categories["ck0"].tolist()
    for name in ("context", "context_mask", "known_future", "known_future_mask", "targets", "loss_weight"):
        assert torch.equal(getattr(batch, name), getattr(changed_batch, name)), name


def test_frame_to_forecast_matches_upstream_with_several_targets(tiny_forecaster):
    """With several targets the category codes are model inputs. Feeding upstream's `forecast` the codes of
    the dataset's window (`window_category_codes` of the raw codes) reproduces the training forward.
    """
    forecaster = tiny_forecaster
    patch = forecaster.patch_size
    context_length, horizon, steps = 9 * patch + 1, 2 * patch + 1, 60
    columns = make_columns(17, series=2, steps=steps, **FULL)
    dataset = build(columns, context_length=context_length, forecast_horizon=horizon, stride=1000)
    windows = [dataset[0], dataset[1]]
    batch = dataset.collator(forecaster.config)(windows)

    model = forecaster.model
    with torch.no_grad():
        output = training_forward(model, batch)
        loc, scale = target_frames(model, batch, output)
        ours = denormalise_quantiles(model, output.quantiles, loc, scale)
    ours = ours.reshape(2, dataset.rows_per_series, batch.horizon_steps, -1)[:, :2, :horizon]

    span = slice(steps - horizon - context_length, steps)
    raw_past = np.stack([s.categorical_past[:, span][:, :context_length] for s in dataset.series])
    raw_known = np.stack([s.categorical_known[:, span] for s in dataset.series])
    codes_past, _ = window_category_codes(raw_past)
    codes_known = np.concatenate(
        window_category_codes(raw_known[..., :context_length], raw_known[..., context_length:]), -1
    )
    assert np.array_equal(codes_past, np.stack([w.categorical_past for w in windows]), equal_nan=True)
    assert np.array_equal(codes_known, np.stack([w.categorical_known for w in windows]), equal_nan=True)

    expected = forecaster.forecast(
        np.stack([w.target_context for w in windows]),
        horizon,
        past_covariates=np.stack([w.past for w in windows]),
        known_covariates=np.stack([w.known for w in windows]),
        categorical_past_covariates=codes_past,
        categorical_known_covariates=codes_known,
        symmetric_averaging=False,
        clip_nonnegative=False,
    ).quantiles
    assert max_rel(ours, expected) < 1e-5


@pytest.mark.parametrize("kind", FRAME_KINDS)
def test_frame_to_forecast_matches_upstream_on_the_same_raw_values(tiny_forecaster, kind):
    """End to end: frame -> dataset -> collator -> training forward -> median-folded frame equals upstream's
    `forecast` called on the raw arrays of the same window, including categorical covariates.
    """
    forecaster = tiny_forecaster
    patch = forecaster.patch_size
    context_length, horizon, steps = 9 * patch + 1, 2 * patch + 1, 60
    columns = make_columns(
        16, series=2, steps=steps, targets=1, past=1, known=1, categorical_past=1, categorical_known=1
    )
    dataset = build(columns, kind, context_length=context_length, forecast_horizon=horizon, stride=1000)
    assert len(dataset) == 2, "one window per series: the last one"
    batch = dataset.collator(forecaster.config)([dataset[0], dataset[1]])

    model = forecaster.model
    with torch.no_grad():
        output = training_forward(model, batch)
        loc, scale = target_frames(model, batch, output)
        ours = denormalise_quantiles(model, output.quantiles, loc, scale)
    ours = ours.reshape(2, dataset.rows_per_series, batch.horizon_steps, -1)[:, :1, :horizon]

    origin = steps - horizon
    window = slice(origin - context_length, steps)
    data = dataset.series

    def stack(field: str, stop: int | None) -> np.ndarray:
        return np.stack([getattr(s, field)[:, window][:, :stop] for s in data])

    expected = forecaster.forecast(
        stack("targets", context_length),
        horizon,
        past_covariates=stack("past", context_length),
        known_covariates=stack("known", None),
        categorical_past_covariates=stack("categorical_past", context_length),
        categorical_known_covariates=stack("categorical_known", None),
        symmetric_averaging=False,
        clip_nonnegative=False,
    ).quantiles
    assert max_rel(ours, expected) < 1e-5
