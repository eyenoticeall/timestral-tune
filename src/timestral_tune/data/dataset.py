"""Long-format frame loader, point-in-time windowing and time splits.

A `TimeSeriesDataset` holds every series as contiguous arrays on a regular time grid and yields
forecast-origin `Window`s: up to `context_length` steps before an origin and `forecast_horizon`
steps from it. A window's numeric arrays are views into the series arrays; the collator makes the
padded copies.

Point-in-time rules enforced here (spec §6.3):

- A window's past-only covariates stop at its origin. Later values are never read.
- A window's target values from the origin on are returned separately, as labels.
- `available_lag` shifts a past-only covariate that is published late, so that the value at each
  step is the one that was known at that step.
- `split_by_time` gives training windows nothing at or after the cutoff.
- A window's category codes are numbered among the categories that window can see, so a category
  that first occurs later cannot change them (`window_category_codes`).

Input problems raise instead of being papered over: null ids or timestamps, duplicate
(id, timestamp) pairs, timestamps off the regular grid, non-numeric value columns, infinities.

- Spec: `docs/SPEC.md` §6.2, §6.3, §7.1
- Acceptance tests: T4, T5, T12
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from .._vendor.timestral.config import TimestralConfig
from .collator import DHFDataCollator, Window
from .encoders import window_category_codes

__all__ = ["ColumnRoles", "SeriesData", "TimeSeriesDataset", "WindowStats"]

logger = logging.getLogger(__name__)

# Regularising a series inserts a missing row for every absent time step. If that would multiply the
# data by more than this factor, the inferred frequency is almost certainly wrong.
_MAX_FILL_FACTOR = 100


@dataclass(frozen=True)
class ColumnRoles:
    """Which input columns play which role. Order is the row order the model sees."""

    targets: tuple[str, ...]
    past: tuple[str, ...] = ()
    known: tuple[str, ...] = ()
    categorical_past: tuple[str, ...] = ()
    categorical_known: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.targets) == 0:
            raise ValueError("at least one target column is required")
        names = [*self.targets, *self.past, *self.known, *self.categorical_past, *self.categorical_known]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"columns may have one role each; repeated: {repeated}")

    @property
    def value_columns(self) -> tuple[str, ...]:
        return (*self.targets, *self.past, *self.known)

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return (*self.categorical_past, *self.categorical_known)


@dataclass(frozen=True)
class SeriesData:
    """One series on a regular time grid. NaN marks a missing value.

    Attributes:
        series_id: The series' identifier from the id column.
        timestamps: `(N,)` strictly increasing, integer or `datetime64`.
        targets: `(V, N)` float32 target variates.
        past: `(C_past, N)` float32 past-only covariates.
        known: `(C_known, N)` float32 known-future covariates.
        categorical_past: `(D_past, N)` float64 category codes.
        categorical_known: `(D_known, N)` float64 category codes.
    """

    series_id: Hashable
    timestamps: np.ndarray
    targets: np.ndarray
    past: np.ndarray
    known: np.ndarray
    categorical_past: np.ndarray
    categorical_known: np.ndarray

    def __post_init__(self) -> None:
        steps = self.timestamps.shape[0]
        if self.timestamps.ndim != 1 or self.timestamps.dtype.kind not in "iuM":
            raise ValueError("timestamps must be a 1-D integer or datetime64 array")
        if steps > 1 and not bool((np.diff(self.timestamps.astype(np.int64)) > 0).all()):
            raise ValueError(f"series {self.series_id!r}: timestamps must be strictly increasing")
        for name, dtype in (
            ("targets", np.float32),
            ("past", np.float32),
            ("known", np.float32),
            ("categorical_past", np.float64),
            ("categorical_known", np.float64),
        ):
            array = getattr(self, name)
            if array.ndim != 2 or array.shape[1] != steps or array.dtype != dtype:
                raise ValueError(
                    f"series {self.series_id!r}: {name} must be a (rows, {steps}) {np.dtype(dtype).name} array, "
                    f"got {array.shape} {array.dtype}"
                )
            if np.isinf(array).any():
                raise ValueError(f"series {self.series_id!r}: {name} holds infinite values; use NaN for missing")
        if self.targets.shape[0] < 1:
            raise ValueError(f"series {self.series_id!r}: at least one target variate is required")

    @property
    def num_steps(self) -> int:
        return self.timestamps.shape[0]

    @property
    def layout(self) -> tuple[int, int, int, int, int]:
        return (
            self.targets.shape[0],
            self.past.shape[0],
            self.known.shape[0],
            self.categorical_past.shape[0],
            self.categorical_known.shape[0],
        )


@dataclass(frozen=True)
class WindowStats:
    """How the window index was built.

    Attributes:
        series: Number of series.
        series_without_windows: Series that yield no window (too short, or all windows dropped).
        candidates: Origins considered before filtering.
        dropped_no_context: Windows dropped because no target value is observed in the context.
        dropped_unscored: Windows dropped because no target value is observed in the horizon; they
            would contribute no loss.
        kept: Windows in the dataset.
    """

    series: int
    series_without_windows: int
    candidates: int
    dropped_no_context: int
    dropped_unscored: int
    kept: int


class TimeSeriesDataset(Dataset[Window]):
    """Forecast-origin windows over a collection of series.

    Build one with `from_polars`, `from_pandas`, `from_arrow` (or `from_frame`, which accepts any of
    the three), or directly from `SeriesData` objects.

    Origins. For a series of `N` steps, origins are placed at `N - forecast_horizon`,
    `N - forecast_horizon - stride`, and so on, back to `min_context`. Anchoring at the end means
    the most recent data is always used. A window is dropped when its context holds no observed
    target value or its horizon holds none.

    Args:
        series: The series, all with the same number of rows per role.
        context_length: Maximum number of steps before the origin in a window.
        forecast_horizon: Number of steps from the origin that are forecast.
        stride: Distance between consecutive origins of one series.
        min_context: Minimum number of steps before an origin.
        roles: Column names per role, for reference; set by the frame constructors.
        categories: For each categorical column, the sorted category values. The codes stored in
            `series` index this list. Windows do not expose those codes: each window renumbers
            them among the categories it can see (`window_category_codes`). Set by the frame
            constructors.
    """

    def __init__(
        self,
        series: Sequence[SeriesData],
        *,
        context_length: int,
        forecast_horizon: int,
        stride: int = 1,
        min_context: int = 1,
        roles: ColumnRoles | None = None,
        categories: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        if context_length < 1:
            raise ValueError(f"context_length must be positive, got {context_length}")
        if forecast_horizon < 1:
            raise ValueError(f"forecast_horizon must be positive, got {forecast_horizon}")
        if stride < 1:
            raise ValueError(f"stride must be positive, got {stride}")
        if min_context < 1:
            raise ValueError(f"min_context must be positive, got {min_context}")
        self.series: tuple[SeriesData, ...] = tuple(series)
        if len(self.series) == 0:
            raise ValueError("a dataset needs at least one series")
        layouts = {s.layout for s in self.series}
        if len(layouts) != 1:
            raise ValueError(f"all series must have the same number of rows per role; got {sorted(layouts)}")
        kinds = {s.timestamps.dtype for s in self.series}
        if len(kinds) != 1:
            raise ValueError(f"all series must share one timestamp dtype; got {sorted(map(str, kinds))}")

        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.min_context = min_context
        self.roles = roles
        self.categories: dict[str, np.ndarray] = dict(categories or {})
        self._series_index, self._origins, self.stats = self._build_index()
        if self.stats.kept == 0:
            raise ValueError(
                "no usable window: every series is shorter than min_context + forecast_horizon "
                f"({min_context} + {forecast_horizon} steps) or has no observed target where one is needed. "
                f"{self.stats}"
            )
        logger.info("built %d windows from %d series: %s", self.stats.kept, self.stats.series, self.stats)

    # ------------------------------------------------------------------ index

    def _build_index(self) -> tuple[np.ndarray, np.ndarray, WindowStats]:
        horizon, context = self.forecast_horizon, self.context_length
        series_parts: list[np.ndarray] = []
        origin_parts: list[np.ndarray] = []
        candidates = dropped_context = dropped_unscored = empty_series = 0
        for index, data in enumerate(self.series):
            last_origin = data.num_steps - horizon
            if last_origin < self.min_context:
                empty_series += 1
                continue
            origins = np.arange(last_origin, self.min_context - 1, -self.stride, dtype=np.int64)[::-1]
            candidates += origins.size
            # Running count of steps with any observed target, so each window's counts are two lookups.
            observed = np.concatenate([[0], np.isfinite(data.targets).any(axis=0).cumsum()])
            in_context = observed[origins] - observed[np.maximum(origins - context, 0)]
            in_horizon = observed[origins + horizon] - observed[origins]
            no_context = in_context == 0
            unscored = ~no_context & (in_horizon == 0)
            dropped_context += int(no_context.sum())
            dropped_unscored += int(unscored.sum())
            kept = origins[~no_context & ~unscored]
            if kept.size == 0:
                empty_series += 1
                continue
            series_parts.append(np.full(kept.size, index, dtype=np.int64))
            origin_parts.append(kept)
        series_index = np.concatenate(series_parts) if series_parts else np.empty(0, dtype=np.int64)
        origins = np.concatenate(origin_parts) if origin_parts else np.empty(0, dtype=np.int64)
        stats = WindowStats(
            series=len(self.series),
            series_without_windows=empty_series,
            candidates=candidates,
            dropped_no_context=dropped_context,
            dropped_unscored=dropped_unscored,
            kept=int(origins.size),
        )
        return series_index, origins, stats

    def _with_windows(self, keep: np.ndarray) -> TimeSeriesDataset:
        """A dataset sharing this one's series, restricted to the windows where `keep` is True."""
        subset = object.__new__(type(self))
        subset.__dict__.update(self.__dict__)
        subset._series_index = self._series_index[keep]
        subset._origins = self._origins[keep]
        remaining = {int(i) for i in np.unique(subset._series_index)}
        subset.stats = WindowStats(
            series=self.stats.series,
            series_without_windows=len(self.series) - len(remaining),
            candidates=self.stats.candidates,
            dropped_no_context=self.stats.dropped_no_context,
            dropped_unscored=self.stats.dropped_unscored,
            kept=int(subset._origins.size),
        )
        return subset

    # ------------------------------------------------------------- container

    def __len__(self) -> int:
        return int(self._origins.size)

    def __getitem__(self, index: int) -> Window:
        size = len(self)
        if not -size <= index < size:
            raise IndexError(f"window index {index} is out of range for {size} windows")
        series_index = int(self._series_index[index])
        origin = int(self._origins[index])
        data = self.series[series_index]
        start = max(0, origin - self.context_length)
        end = origin + self.forecast_horizon
        # The stored codes index every category of the dataset, including ones that first occur after this
        # window. Renumber them among the categories the window can see, so its inputs depend on nothing later.
        categorical_past, _ = window_category_codes(data.categorical_past[:, start:origin])
        known_context, known_horizon = window_category_codes(
            data.categorical_known[:, start:origin], data.categorical_known[:, origin:end]
        )
        return Window(
            series_index=series_index,
            origin=origin,
            target_context=data.targets[:, start:origin],
            target_future=data.targets[:, origin:end],
            past=data.past[:, start:origin],
            known=data.known[:, start:end],
            categorical_past=categorical_past,
            categorical_known=np.concatenate([known_context, known_horizon], axis=-1),
        )

    def __repr__(self) -> str:
        targets, past, known, cat_past, cat_known = self.series[0].layout
        return (
            f"TimeSeriesDataset(windows={len(self)}, series={len(self.series)}, targets={targets}, past={past}, "
            f"known={known}, categorical_past={cat_past}, categorical_known={cat_known}, "
            f"context_length={self.context_length}, forecast_horizon={self.forecast_horizon}, stride={self.stride})"
        )

    @property
    def rows_per_series(self) -> int:
        """Number of rows each series contributes to a batch."""
        return sum(self.series[0].layout)

    def origin_time(self, index: int) -> Any:
        """Timestamp of the first horizon step of window `index`."""
        window_index = range(len(self))[index]
        return self.series[int(self._series_index[window_index])].timestamps[int(self._origins[window_index])]

    def collator(self, config: TimestralConfig) -> DHFDataCollator:
        """The collator for this dataset and a model with `config`."""
        return DHFDataCollator.for_model(config, self.context_length)

    # ----------------------------------------------------------------- split

    def split_by_time(self, cutoff: Any, embargo: int = 0) -> tuple[TimeSeriesDataset, TimeSeriesDataset]:
        """Split windows into a training and a validation set at a point in time.

        - Training windows end before the cutoff: every step they contain, context and horizon
          alike, is earlier than `cutoff`. Training sees nothing at or after it.
        - Validation windows have their origin at least `embargo` steps after the first step at
          or after the cutoff. Their context may reach back before the cutoff, as it would in
          real use.
        - Windows whose horizon straddles the cutoff are in neither set.

        So no validation label ever appears in a training window, with or without an embargo. The
        embargo adds a gap for data where neighbouring steps are strongly dependent.

        Args:
            cutoff: A value comparable with the timestamps: a date string, `datetime`,
                `numpy.datetime64` for datetime timestamps, or an integer for integer timestamps.
            embargo: Extra steps skipped after the cutoff before validation origins start.

        Raises:
            ValueError: If either side would be empty.
        """
        if embargo < 0:
            raise ValueError(f"embargo must be non-negative, got {embargo}")
        dtype = self.series[0].timestamps.dtype
        try:
            cutoff_value = (
                np.asarray(cutoff).astype(dtype) if dtype.kind == "M" else np.asarray(int(cutoff), dtype=dtype)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"cutoff {cutoff!r} is not comparable with timestamps of dtype {dtype}") from error

        # Index, per series, of the first step at or after the cutoff.
        first_after = np.array([np.searchsorted(s.timestamps, cutoff_value, side="left") for s in self.series])
        boundary = first_after[self._series_index]
        is_train = self._origins + self.forecast_horizon <= boundary
        is_validation = self._origins >= boundary + embargo
        if not is_train.any() or not is_validation.any():
            raise ValueError(
                f"split at {cutoff!r} with embargo {embargo} leaves {int(is_train.sum())} training and "
                f"{int(is_validation.sum())} validation windows; both must be non-empty"
            )
        logger.info(
            "split at %r: %d training, %d validation, %d dropped windows",
            cutoff,
            int(is_train.sum()),
            int(is_validation.sum()),
            int((~is_train & ~is_validation).sum()),
        )
        return self._with_windows(is_train), self._with_windows(is_validation)

    # ---------------------------------------------------------- constructors

    @classmethod
    def from_polars(cls, frame: Any, **kwargs: Any) -> TimeSeriesDataset:
        """Build from a `polars.DataFrame`. See `from_frame` for the arguments."""
        if _frame_kind(frame) != "polars":
            raise TypeError(f"from_polars needs a polars.DataFrame, got {type(frame).__name__}")
        return cls.from_frame(frame, **kwargs)

    @classmethod
    def from_pandas(cls, frame: Any, **kwargs: Any) -> TimeSeriesDataset:
        """Build from a `pandas.DataFrame`. See `from_frame` for the arguments."""
        if _frame_kind(frame) != "pandas":
            raise TypeError(f"from_pandas needs a pandas.DataFrame, got {type(frame).__name__}")
        return cls.from_frame(frame, **kwargs)

    @classmethod
    def from_arrow(cls, table: Any, **kwargs: Any) -> TimeSeriesDataset:
        """Build from a `pyarrow.Table`. See `from_frame` for the arguments."""
        if _frame_kind(table) != "pyarrow":
            raise TypeError(f"from_arrow needs a pyarrow.Table, got {type(table).__name__}")
        return cls.from_frame(table, **kwargs)

    @classmethod
    def from_frame(
        cls,
        frame: Any,
        *,
        time_col: str,
        target_cols: Sequence[str],
        context_length: int,
        forecast_horizon: int,
        id_col: str | None = None,
        past_covariates: Sequence[str] = (),
        known_covariates: Sequence[str] = (),
        categorical_past_covariates: Sequence[str] = (),
        categorical_known_covariates: Sequence[str] = (),
        stride: int = 1,
        min_context: int = 1,
        regularize: bool = True,
        freq: int | np.timedelta64 | datetime.timedelta | None = None,
        available_lag: Mapping[str, int] | None = None,
    ) -> TimeSeriesDataset:
        """Build from a long-format frame: one row per (series, timestamp).

        Args:
            frame: A `polars.DataFrame`, `pandas.DataFrame` or `pyarrow.Table`. Row order does not
                matter; rows are sorted by series and time.
            time_col: Timestamp column: integers or datetimes without a time zone.
            target_cols: Numeric columns to forecast.
            context_length, forecast_horizon, stride, min_context: See the class.
            id_col: Series identifier column. None means the frame holds a single series.
            past_covariates: Numeric columns known up to the forecast origin only.
            known_covariates: Numeric columns known in advance over the horizon.
            categorical_past_covariates: Categorical columns known up to the origin only.
            categorical_known_covariates: Categorical columns known in advance.
            regularize: If True (default), put every series on a regular grid with step `freq`,
                inserting missing values for absent time steps; a timestamp that is off the grid
                raises. If False, consecutive rows of a series are taken as consecutive steps
                whatever their timestamps: use this for calendar-irregular data such as monthly
                or business-day series.
            freq: Grid step, in the timestamp column's own units for integers, or as a timedelta
                for datetimes. Inferred as the most common step when None.
            available_lag: For past-only covariates published late: column name to the number of
                steps by which the column is delayed. A lag of 2 means the value stamped `t`
                only became known at `t + 2`.
        """
        roles = ColumnRoles(
            targets=tuple(target_cols),
            past=tuple(past_covariates),
            known=tuple(known_covariates),
            categorical_past=tuple(categorical_past_covariates),
            categorical_known=tuple(categorical_known_covariates),
        )
        lags = dict(available_lag or {})
        past_columns = {*roles.past, *roles.categorical_past}
        for name, lag in lags.items():
            if name not in past_columns:
                raise ValueError(f"available_lag names {name!r}, which is not a past-only covariate")
            if int(lag) != lag or lag < 0:
                raise ValueError(f"available_lag for {name!r} must be a non-negative integer, got {lag!r}")

        kind = _frame_kind(frame)
        _require_columns(
            frame, kind, [*([id_col] if id_col else []), time_col, *roles.value_columns, *roles.categorical_columns]
        )
        times = _time_column(_column(frame, kind, time_col), time_col)
        rows = times.shape[0]
        if rows == 0:
            raise ValueError("the frame has no rows")
        if id_col is None:
            id_codes, id_values = np.zeros(rows, dtype=np.int64), np.array([0])
        else:
            id_codes, id_values = _factorize(_column(frame, kind, id_col), id_col, allow_missing=False)

        values = np.empty((rows, len(roles.value_columns)), dtype=np.float32)
        for position, name in enumerate(roles.value_columns):
            column = _float_column(frame, kind, name)
            # NaN compares False, so missing values pass; anything beyond float32's range would become inf.
            if (np.abs(column) > np.finfo(np.float32).max).any():
                raise ValueError(f"column {name!r} holds values too large for float32, which the model computes in")
            values[:, position] = column
        codes = np.empty((rows, len(roles.categorical_columns)), dtype=np.float64)
        categories: dict[str, np.ndarray] = {}
        for position, name in enumerate(roles.categorical_columns):
            column_codes, categories[name] = _factorize(_column(frame, kind, name), name, allow_missing=True)
            codes[:, position] = column_codes

        series = _split_series(
            times=times,
            id_codes=id_codes,
            id_values=id_values,
            values=values,
            codes=codes,
            roles=roles,
            regularize=regularize,
            freq=freq,
            lags=lags,
        )
        return cls(
            series,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
            stride=stride,
            min_context=min_context,
            roles=roles,
            categories=categories,
        )


# ------------------------------------------------------------------ frame I/O


def _frame_kind(frame: Any) -> str:
    """`"polars"`, `"pandas"` or `"pyarrow"`, without importing any of them."""
    module = type(frame).__module__.split(".")[0]
    name = type(frame).__name__
    if module == "polars" and name == "LazyFrame":
        raise TypeError("got a polars.LazyFrame; call .collect() first")
    if (module, name) in {("polars", "DataFrame"), ("pandas", "DataFrame"), ("pyarrow", "Table")}:
        return module
    raise TypeError(
        f"expected a polars.DataFrame, pandas.DataFrame or pyarrow.Table, got {type(frame).__module__}.{name}"
    )


def _require_columns(frame: Any, kind: str, names: Sequence[str]) -> None:
    available = set(frame.column_names if kind == "pyarrow" else frame.columns)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"columns not found in the frame: {missing}")


def _column(frame: Any, kind: str, name: str) -> np.ndarray:
    if kind == "polars":
        return frame.get_column(name).to_numpy()
    if kind == "pandas":
        return frame[name].to_numpy()
    return frame.column(name).to_numpy()


def _float_column(frame: Any, kind: str, name: str) -> np.ndarray:
    """A numeric column as float64 with NaN for missing; raises if it is not numeric or holds infinities."""
    try:
        if kind == "pandas":
            # Handles pandas' nullable dtypes, whose missing marker NumPy cannot convert by itself.
            array = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            array = np.asarray(_column(frame, kind, name), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"column {name!r} is not numeric; pass categorical columns as categorical covariates"
        ) from error
    if np.isinf(array).any():
        raise ValueError(f"column {name!r} holds infinite values; use nulls for missing values")
    return array


def _is_missing(array: np.ndarray) -> np.ndarray:
    if array.dtype.kind in "fc":
        return np.isnan(array)
    if array.dtype.kind in "mM":
        return np.isnat(array)
    if array.dtype.kind == "O":
        # None, float NaN, pandas.NA and pandas.NaT; without importing pandas.
        return np.fromiter(
            (
                v is None or type(v).__name__ in ("NAType", "NaTType") or (isinstance(v, float) and v != v)
                for v in array
            ),
            dtype=bool,
            count=array.shape[0],
        )
    return np.zeros(array.shape[0], dtype=bool)


def _factorize(array: np.ndarray, name: str, *, allow_missing: bool) -> tuple[np.ndarray, np.ndarray]:
    """`(codes, categories)`: float64 codes indexing the sorted unique values; NaN where missing."""
    missing = _is_missing(array)
    if missing.any() and not allow_missing:
        raise ValueError(f"column {name!r} holds {int(missing.sum())} null values; every row needs one")
    try:
        categories, inverse = np.unique(array[~missing], return_inverse=True)
    except TypeError as error:
        raise ValueError(f"column {name!r} mixes value types that cannot be ordered") from error
    codes = np.full(array.shape[0], np.nan, dtype=np.float64)
    codes[~missing] = inverse
    return codes, categories


def _time_column(array: np.ndarray, name: str) -> np.ndarray:
    """Timestamps as an integer or `datetime64` array; raises on nulls and unsupported types."""
    if array.dtype.kind == "O":
        if _is_missing(array).any():
            raise ValueError(f"time column {name!r} holds null values")
        if any(getattr(v, "tzinfo", None) is not None for v in array[:1]):
            raise ValueError(
                f"time column {name!r} is time-zone aware; convert it to naive timestamps (e.g. UTC) first"
            )
        try:
            array = array.astype("datetime64[ns]")
        except (TypeError, ValueError) as error:
            raise ValueError(f"time column {name!r} must hold integers or datetimes") from error
    if array.dtype.kind == "M":
        if np.isnat(array).any():
            raise ValueError(f"time column {name!r} holds null values")
        return array
    if array.dtype.kind in "iu":
        return array.astype(np.int64)
    raise ValueError(f"time column {name!r} must hold integers or datetimes, got dtype {array.dtype}")


def _step_as_int(freq: int | np.timedelta64 | datetime.timedelta, dtype: np.dtype) -> int:
    if dtype.kind == "M":
        if not isinstance(freq, (np.timedelta64, datetime.timedelta)):
            raise ValueError(f"freq must be a timedelta for datetime timestamps, got {freq!r}")
        unit = np.datetime_data(dtype)[0]
        exact = np.timedelta64(freq)
        step = int(exact.astype(f"timedelta64[{unit}]").astype(np.int64))
        if step < 1 or np.timedelta64(step, unit) != exact:
            raise ValueError(f"freq {freq!r} is not a positive whole number of the time column's unit ({unit})")
        return step
    if isinstance(freq, (np.timedelta64, datetime.timedelta)) or int(freq) != freq or freq < 1:
        raise ValueError(f"freq must be a positive integer for integer timestamps, got {freq!r}")
    return int(freq)


def _split_series(
    *,
    times: np.ndarray,
    id_codes: np.ndarray,
    id_values: np.ndarray,
    values: np.ndarray,
    codes: np.ndarray,
    roles: ColumnRoles,
    regularize: bool,
    freq: int | np.timedelta64 | datetime.timedelta | None,
    lags: Mapping[str, int],
) -> list[SeriesData]:
    """Sort rows by (series, time), validate them, put each series on its grid and split the roles."""
    time_dtype = times.dtype
    ticks = times.astype(np.int64)
    order = np.lexsort((ticks, id_codes))
    ticks, sorted_ids, values, codes = ticks[order], id_codes[order], values[order], codes[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_ids)) + 1, [ticks.shape[0]]])
    groups = [(int(sorted_ids[a]), slice(int(a), int(b))) for a, b in zip(starts[:-1], starts[1:], strict=True)]

    for series_code, rows in groups:
        steps = np.diff(ticks[rows])
        if (steps <= 0).any():
            duplicate = ticks[rows][1:][steps <= 0][0]
            raise ValueError(
                f"series {id_values[series_code]!r} has more than one row at timestamp "
                f"{np.asarray(duplicate).astype(time_dtype)!r}; each (series, timestamp) pair must be unique"
            )

    step = 1
    if regularize:
        if freq is not None:
            step = _step_as_int(freq, time_dtype)
        else:
            all_steps = np.concatenate([np.diff(ticks[rows]) for _, rows in groups])
            if all_steps.size:
                candidates, counts = np.unique(all_steps, return_counts=True)
                step = int(candidates[np.argmax(counts)])  # most common step; ties go to the smallest
        grid_total = sum(int((ticks[rows][-1] - ticks[rows][0]) // step) + 1 for _, rows in groups)
        if grid_total > _MAX_FILL_FACTOR * ticks.shape[0]:
            raise ValueError(
                f"a regular grid with step {step} would have {grid_total} steps for {ticks.shape[0]} rows. The "
                "frequency looks wrong: pass freq explicitly, or regularize=False for calendar-irregular data"
            )

    n_targets, n_past, n_known = len(roles.targets), len(roles.past), len(roles.known)
    d_past = len(roles.categorical_past)
    series: list[SeriesData] = []
    for series_code, rows in groups:
        series_ticks = ticks[rows]
        if regularize:
            offsets = series_ticks - series_ticks[0]
            off_grid = offsets % step != 0
            if off_grid.any():
                bad = np.asarray(series_ticks[off_grid][0]).astype(time_dtype)
                raise ValueError(
                    f"series {id_values[series_code]!r}: timestamp {bad!r} is not on the regular grid with step "
                    f"{step} that starts at the series' first timestamp. Pass freq explicitly, or regularize=False "
                    "to treat consecutive rows as consecutive steps"
                )
            positions = offsets // step
            length = int(positions[-1]) + 1
            grid_ticks = series_ticks[0] + np.arange(length, dtype=np.int64) * step
        else:
            positions = np.arange(series_ticks.shape[0])
            length = series_ticks.shape[0]
            grid_ticks = series_ticks

        value_grid = np.full((values.shape[1], length), np.nan, dtype=np.float32)
        value_grid[:, positions] = values[rows].T
        code_grid = np.full((codes.shape[1], length), np.nan, dtype=np.float64)
        code_grid[:, positions] = codes[rows].T

        past = value_grid[n_targets : n_targets + n_past]
        categorical_past = code_grid[:d_past]
        for names, block in ((roles.past, past), (roles.categorical_past, categorical_past)):
            for row, name in enumerate(names):
                lag = int(lags.get(name, 0))
                if lag:
                    block[row] = _delay(block[row], lag)

        series.append(
            SeriesData(
                series_id=id_values[series_code].item()
                if hasattr(id_values[series_code], "item")
                else id_values[series_code],
                timestamps=grid_ticks.astype(time_dtype),
                targets=np.ascontiguousarray(value_grid[:n_targets]),
                past=np.ascontiguousarray(past),
                known=np.ascontiguousarray(value_grid[n_targets + n_past : n_targets + n_past + n_known]),
                categorical_past=np.ascontiguousarray(categorical_past),
                categorical_known=np.ascontiguousarray(code_grid[d_past:]),
            )
        )
    return series


def _delay(row: np.ndarray, lag: int) -> np.ndarray:
    """Shift a row `lag` steps later, so step `t` holds what was stamped `t - lag`; the start becomes missing."""
    delayed = np.full_like(row, np.nan)
    if lag < row.shape[0]:
        delayed[lag:] = row[: row.shape[0] - lag]
    return delayed
