"""Reproducible, permissively licensed benchmark datasets for Phase 3 (spec §10, §12).

Datasets (all downloaded at pinned Hugging Face Hub revisions via `huggingface_hub`):

1. **Realised volatility** — `twelvedata/financial-world-model` (`MIT`, sha `da186e5fd362`):
   51 large-cap US equities on 1-hour trading bars (2024-01 through 2026-11-20).
   Target: `rv_20` (20-bar causal realised volatility).
   Past covariates: `log1p(volume)`, `vix_level`, `spy_logret_1`, `sector_logret_1`, `atr_14`.
   Known-future covariates: trading-session hour and day of week.
   Test split is strictly post-October 27, 2026 (`2026-10-28` to `2026-11-20`), after Timestral's
   release date, eliminating pretraining contamination on the test window.

2. **Retail demand** — `Dingdong-Inc/FreshRetailNet-50K` (`CC BY 4.0`, sha `c374e85975c8`):
   Deterministic top-activity subset of 64 store-SKU hourly sales series over 97 days (2,328 hours
   each). Target: hourly sales (non-negative, zero-inflated).
   Past covariates: hourly stockout indicator, daily precipitation, average temperature.
   Known-future covariates: promotional discount, holiday flag, activity flag, hour of day, day of week.
   Contamination note: published prior to October 2026; may have been seen during post-training.

3. **General forgetting set** — `Salesforce/GiftEval` (`Apache-2.0`, sha `30841734ac5c`):
   Held-out hourly series from energy (`ett1/H`), weather (`jena_weather/H`), and traffic
   (`LOOP_SEATTLE/H`) — completely disjoint from equity volatility and grocery retail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import pyarrow.ipc as ipc
from huggingface_hub import hf_hub_download

from timestral_tune.data import ColumnRoles, SeriesData, TimeSeriesDataset

TWELVEDATA_REPO = "twelvedata/financial-world-model"
TWELVEDATA_REV = "da186e5fd362"
FRESHRETAIL_REPO = "Dingdong-Inc/FreshRetailNet-50K"
FRESHRETAIL_REV = "c374e85975c8"
GIFTEVAL_REPO = "Salesforce/GiftEval"
GIFTEVAL_REV = "30841734ac5c"


@dataclass(frozen=True)
class BenchmarkSplits:
    """Strictly chronological train / validation (calibration) / test datasets."""

    name: str
    license: str
    source: str
    contamination_note: str
    test_date_range: str
    seasonality: int
    train: TimeSeriesDataset
    val: TimeSeriesDataset
    test: TimeSeriesDataset


def _load_twelvedata_1h(
    *,
    target_col: str,
    name: str,
    context_length: int = 256,
    forecast_horizon: int = 32,
    train_stride: int = 32,
    eval_stride: int = 32,
) -> BenchmarkSplits:
    """Shared loader for 51-equity 1-hour Twelvedata benchmarks (`rv_20` or signed `logret_20`)."""
    p_val = hf_hub_download(TWELVEDATA_REPO, "bars_1h/val.parquet", repo_type="dataset", revision=TWELVEDATA_REV)
    p_test = hf_hub_download(TWELVEDATA_REPO, "bars_1h/test.parquet", repo_type="dataset", revision=TWELVEDATA_REV)
    cols = [
        "symbol",
        "datetime",
        target_col,
        "volume",
        "vix_level",
        "spy_logret_1",
        "sector_logret_1",
        "atr_14",
    ]
    df = pl.concat([pl.read_parquet(p_val, columns=cols), pl.read_parquet(p_test, columns=cols)])
    df = df.sort(["symbol", "datetime"]).with_columns(
        pl.col("datetime").dt.convert_time_zone("UTC").alias("dt_utc"),
        pl.col("volume").clip(lower_bound=0.0).log1p().cast(pl.Float32).alias("log_volume"),
        pl.col("datetime").dt.hour().cast(pl.Float32).alias("hour_of_day"),
        pl.col("datetime").dt.weekday().cast(pl.Float32).alias("day_of_week"),
    )

    val_cutoff_ns = int(np.datetime64("2026-06-01T00:00:00", "ns").astype(np.int64))
    test_cutoff_ns = int(np.datetime64("2026-10-28T00:00:00", "ns").astype(np.int64))

    series_list: list[SeriesData] = []
    dt_by_series: list[np.ndarray] = []
    for sym, grp in df.group_by("symbol", maintain_order=True):
        sym_name = str(sym[0] if isinstance(sym, tuple) else sym)
        n = grp.height
        dt_ns = grp["dt_utc"].dt.epoch("ns").to_numpy()
        val_pos = int(np.searchsorted(dt_ns, val_cutoff_ns))
        test_pos = int(np.searchsorted(dt_ns, test_cutoff_ns))
        if val_pos < context_length + forecast_horizon or (n - test_pos) < forecast_horizon:
            continue
        dt_by_series.append(grp["dt_utc"].to_numpy())

        bar_idx = np.arange(n, dtype=np.int64) - test_pos
        targets = grp[target_col].to_numpy().astype(np.float32)[None, :]
        past = np.stack(
            [
                grp["log_volume"].to_numpy().astype(np.float32),
                grp["vix_level"].to_numpy().astype(np.float32),
                grp["spy_logret_1"].to_numpy().astype(np.float32),
                grp["sector_logret_1"].to_numpy().astype(np.float32),
                grp["atr_14"].to_numpy().astype(np.float32),
            ],
            axis=0,
        )
        known = np.stack(
            [
                grp["hour_of_day"].to_numpy().astype(np.float32),
                grp["day_of_week"].to_numpy().astype(np.float32),
            ],
            axis=0,
        )
        series_list.append(
            SeriesData(
                series_id=sym_name,
                timestamps=bar_idx,
                targets=targets,
                past=past,
                known=known,
                categorical_past=np.empty((0, n), dtype=np.float64),
                categorical_known=np.empty((0, n), dtype=np.float64),
            )
        )

    roles = ColumnRoles(
        targets=(target_col,),
        past=("log_volume", "vix_level", "spy_logret_1", "sector_logret_1", "atr_14"),
        known=("hour_of_day", "day_of_week"),
    )
    val_start_bar = -700
    full_train = TimeSeriesDataset(
        series_list,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=train_stride,
        min_context=context_length,
        roles=roles,
    )
    full_eval = TimeSeriesDataset(
        series_list,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=eval_stride,
        min_context=context_length,
        roles=roles,
    )
    train_ds, _ = full_train.split_by_time(cutoff=val_start_bar)
    pre_test, test_ds = full_eval.split_by_time(cutoff=0)
    _, val_ds = pre_test.split_by_time(cutoff=val_start_bar)

    scored_starts = [dt_by_series[int(test_ds._series_index[i])][int(test_ds._origins[i])] for i in range(len(test_ds))]
    scored_ends = [
        dt_by_series[int(test_ds._series_index[i])][int(test_ds._origins[i]) + forecast_horizon - 1]
        for i in range(len(test_ds))
    ]
    test_range = f"{str(min(scored_starts))[:10]} to {str(max(scored_ends))[:10]}"

    return BenchmarkSplits(
        name=name,
        license="MIT",
        source=f"{TWELVEDATA_REPO}@{TWELVEDATA_REV} (bars_1h, target={target_col})",
        contamination_note=(
            f"Scored test targets span {test_range} (strictly after Timestral's 2026-10-27 release; "
            "no pretraining contamination on test targets). Training/validation bars precede release."
        ),
        test_date_range=test_range,
        seasonality=7,
        train=train_ds,
        val=val_ds,
        test=test_ds,
    )


def load_realised_volatility(
    *,
    context_length: int = 256,
    forecast_horizon: int = 32,
    train_stride: int = 32,
    eval_stride: int = 32,
) -> BenchmarkSplits:
    """Load the 51-equity 1-hour realised-volatility (`rv_20`) benchmark with post-release test split."""
    return _load_twelvedata_1h(
        target_col="rv_20",
        name="realised_volatility",
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        train_stride=train_stride,
        eval_stride=eval_stride,
    )


def load_signed_returns(
    *,
    context_length: int = 256,
    forecast_horizon: int = 32,
    train_stride: int = 32,
    eval_stride: int = 32,
) -> BenchmarkSplits:
    """Load the 51-equity 1-hour signed 20-bar log-returns (`logret_20`) benchmark for sign-flip checks."""
    return _load_twelvedata_1h(
        target_col="logret_20",
        name="signed_returns",
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        train_stride=train_stride,
        eval_stride=eval_stride,
    )


def load_retail_demand(
    *,
    num_series: int = 64,
    context_length: int = 256,
    forecast_horizon: int = 48,
    train_stride: int = 48,
    eval_stride: int = 48,
) -> BenchmarkSplits:
    """Load a deterministic 64-series hourly retail-demand benchmark from FreshRetailNet-50K."""
    p_train = hf_hub_download(FRESHRETAIL_REPO, "data/train.parquet", repo_type="dataset", revision=FRESHRETAIL_REV)
    p_eval = hf_hub_download(FRESHRETAIL_REPO, "data/eval.parquet", repo_type="dataset", revision=FRESHRETAIL_REV)
    cols = [
        "store_id",
        "product_id",
        "dt",
        "sale_amount",
        "hours_sale",
        "hours_stock_status",
        "discount",
        "holiday_flag",
        "activity_flag",
        "precpt",
        "avg_temperature",
    ]
    df_tr = pl.read_parquet(p_train, columns=cols)
    df_ev = pl.read_parquet(p_eval, columns=cols)

    # Deterministically select `num_series` store-SKU pairs with full calendar presence using
    # ONLY training-period sales (first 70 days, strictly before val/test) so zero test-period
    # sales influence series selection.
    tr_dates = sorted(df_tr["dt"].unique().to_list())
    train_days_cutoff_dt = tr_dates[69]
    tr_days = len(tr_dates)
    ev_days = df_ev["dt"].n_unique()
    tr_ok = (
        df_tr.group_by(["store_id", "product_id"])
        .agg(
            pl.len().alias("n_tr"),
            pl.col("sale_amount").filter(pl.col("dt") <= train_days_cutoff_dt).sum().alias("train_only_sale"),
        )
        .filter(pl.col("n_tr") == tr_days)
    )
    ev_ok = df_ev.group_by(["store_id", "product_id"]).agg(pl.len().alias("n_ev")).filter(pl.col("n_ev") == ev_days)
    chosen = (
        tr_ok.join(ev_ok, on=["store_id", "product_id"], how="inner")
        .sort(["train_only_sale", "store_id", "product_id"], descending=[True, False, False])
        .head(num_series)
        .select(["store_id", "product_id"])
    )
    df = pl.concat(
        [
            df_tr.join(chosen, on=["store_id", "product_id"], how="semi"),
            df_ev.join(chosen, on=["store_id", "product_id"], how="semi"),
        ]
    ).sort(["store_id", "product_id", "dt"])

    total_days = tr_days + ev_days
    total_hours = total_days * 24
    hours_grid = np.arange(total_hours, dtype=np.int64)
    hour_of_day = (hours_grid % 24).astype(np.float32)

    series_list: list[SeriesData] = []
    for key, grp in df.group_by(["store_id", "product_id"], maintain_order=True):
        if grp.height != total_days:
            continue
        store_id, prod_id = key
        # Expand 24 hourly values per day cleanly.
        sales = np.concatenate([np.asarray(x, dtype=np.float32) for x in grp["hours_sale"].to_list()])
        stock = np.concatenate([np.asarray(x, dtype=np.float32) for x in grp["hours_stock_status"].to_list()])
        if sales.shape[0] != total_hours:
            continue
        precpt = np.repeat(grp["precpt"].to_numpy().astype(np.float32), 24)
        temp = np.repeat(grp["avg_temperature"].to_numpy().astype(np.float32), 24)
        discount = np.repeat(grp["discount"].to_numpy().astype(np.float32), 24)
        holiday = np.repeat(grp["holiday_flag"].to_numpy().astype(np.float32), 24)
        activity = np.repeat(grp["activity_flag"].to_numpy().astype(np.float32), 24)
        dts = pl.Series(grp["dt"]).str.to_date("%Y-%m-%d").dt.weekday().to_numpy().astype(np.float32)
        dow = np.repeat(dts, 24)

        series_list.append(
            SeriesData(
                series_id=f"s{store_id}_p{prod_id}",
                timestamps=hours_grid,
                targets=sales[None, :],
                past=np.stack([stock, precpt, temp], axis=0),
                known=np.stack([discount, holiday, activity, hour_of_day, dow], axis=0),
                categorical_past=np.empty((0, total_hours), dtype=np.float64),
                categorical_known=np.empty((0, total_hours), dtype=np.float64),
            )
        )

    roles = ColumnRoles(
        targets=("sales",),
        past=("stock_status", "precpt", "avg_temperature"),
        known=("discount", "holiday_flag", "activity_flag", "hour_of_day", "day_of_week"),
    )
    # Chronological split: train days 0..69 (1,680h), val days 70..83 (336h), test days 84..96 (312h).
    val_cutoff = 70 * 24
    test_cutoff = 84 * 24

    full_train = TimeSeriesDataset(
        series_list,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=train_stride,
        min_context=context_length,
        roles=roles,
    )
    full_eval = TimeSeriesDataset(
        series_list,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=eval_stride,
        min_context=context_length,
        roles=roles,
    )
    train_ds, _ = full_train.split_by_time(cutoff=val_cutoff)
    pre_test, test_ds = full_eval.split_by_time(cutoff=test_cutoff)
    _, val_ds = pre_test.split_by_time(cutoff=val_cutoff)

    all_dates = sorted(df["dt"].unique().to_list())
    test_start_day = min(int(test_ds._origins[i]) // 24 for i in range(len(test_ds)))
    test_end_day = max((int(test_ds._origins[i]) + forecast_horizon - 1) // 24 for i in range(len(test_ds)))
    test_range = f"{all_dates[test_start_day]} to {all_dates[test_end_day]}"

    return BenchmarkSplits(
        name="retail_demand",
        license="CC BY 4.0",
        source=f"{FRESHRETAIL_REPO}@{FRESHRETAIL_REV} (top {len(series_list)} train-period active store-SKU series)",
        contamination_note=(
            f"FreshRetailNet-50K (scored test targets {test_range}) was published before Timestral's "
            "October 2026 release and may have been seen during Mistral's post-training."
        ),
        test_date_range=test_range,
        seasonality=24,  # 24-hour daily retail cycle
        train=train_ds,
        val=val_ds,
        test=test_ds,
    )


def load_general_forgetting_by_domain(
    *,
    context_length: int = 256,
    forecast_horizon: int = 64,
    stride: int = 64,
    tail_steps: int = 512,
    max_traffic_series: int = 20,
) -> dict[str, TimeSeriesDataset]:
    """Load per-domain non-finance, non-retail forgetting datasets from GIFT-Eval (Apache-2.0).

    Returns `{"ett1": ..., "jena_weather": ..., "loop_seattle": ...}` so forgetting can be
    reported per domain and as an equal-weighted macro average (preventing high-magnitude channels
    such as Jena atmospheric pressure from dominating the pooled wQL denominator).
    """
    out: dict[str, TimeSeriesDataset] = {}
    sources = (
        ("ett1", "ett1/H/data-00000-of-00001.arrow", 7),
        ("jena_weather", "jena_weather/H/data-00000-of-00001.arrow", 21),
        ("loop_seattle", "LOOP_SEATTLE/H/data-00000-of-00001.arrow", max_traffic_series),
    )
    for domain, rel_path, max_count in sources:
        local = hf_hub_download(GIFTEVAL_REPO, rel_path, repo_type="dataset", revision=GIFTEVAL_REV)
        with open(local, "rb") as fh:
            tbl = ipc.open_stream(fh).read_all()
        domain_series: list[SeriesData] = []
        for row_idx in range(tbl.num_rows):
            if len(domain_series) >= max_count:
                break
            raw = tbl["target"][row_idx].as_py()
            channels = raw if isinstance(raw[0], list) else [raw]
            for ch_idx, ch in enumerate(channels):
                if len(domain_series) >= max_count:
                    break
                arr = np.asarray(ch[-tail_steps:], dtype=np.float32)
                if arr.shape[0] < context_length + forecast_horizon or not np.isfinite(arr).all():
                    continue
                if float(np.std(arr)) <= 1e-6:
                    continue
                n = arr.shape[0]
                domain_series.append(
                    SeriesData(
                        series_id=f"{domain}_{row_idx}_{ch_idx}",
                        timestamps=np.arange(n, dtype=np.int64),
                        targets=arr[None, :],
                        past=np.empty((0, n), dtype=np.float32),
                        known=np.empty((0, n), dtype=np.float32),
                        categorical_past=np.empty((0, n), dtype=np.float64),
                        categorical_known=np.empty((0, n), dtype=np.float64),
                    )
                )
        out[domain] = TimeSeriesDataset(
            domain_series,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
            stride=stride,
            min_context=context_length,
            roles=ColumnRoles(targets=("value",)),
        )
    return out


def load_general_forgetting(
    *,
    context_length: int = 256,
    forecast_horizon: int = 64,
    stride: int = 64,
    tail_steps: int = 512,
    max_traffic_series: int = 20,
) -> TimeSeriesDataset:
    """Pooled multi-domain GIFT-Eval forgetting dataset (union of `load_general_forgetting_by_domain`)."""
    by_dom = load_general_forgetting_by_domain(
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
        tail_steps=tail_steps,
        max_traffic_series=max_traffic_series,
    )
    all_series = [s for ds in by_dom.values() for s in ds.series]
    return TimeSeriesDataset(
        all_series,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
        min_context=context_length,
        roles=ColumnRoles(targets=("value",)),
    )
