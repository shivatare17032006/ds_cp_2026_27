"""Leakage-safe demand feature engineering with Polars."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

DEFAULT_WINDOWS = (7, 14, 28)


def build_complete_grid(train: pl.LazyFrame) -> pl.LazyFrame:
    """Create a dense store-family-date grid and fill unrecorded demand with zero."""
    required = {"date", "store_nbr", "family", "sales", "onpromotion"}
    missing = required - set(train.collect_schema().names())
    if missing:
        raise ValueError(f"Feature input missing columns: {sorted(missing)}")
    bounds = train.select(
        pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")
    ).collect().row(0)
    if bounds[0] is None or bounds[1] is None:
        raise ValueError("Cannot build a grid from an empty training table")
    dates = pl.DataFrame(
        {"date": pl.date_range(bounds[0], bounds[1], interval="1d", eager=True)}
    ).lazy()
    entities = train.select(["store_nbr", "family"]).unique()
    grid = entities.join(dates, how="cross")
    observed = train.group_by(["date", "store_nbr", "family"]).agg(
        pl.col("sales").sum().clip(lower_bound=0).alias("sales"),
        pl.col("onpromotion").max().fill_null(0).alias("onpromotion"),
    )
    return (
        grid.join(observed, on=["date", "store_nbr", "family"], how="left")
        .with_columns(
            pl.col("sales").fill_null(0.0),
            pl.col("onpromotion").fill_null(0).cast(pl.Int32),
        )
        .sort(["store_nbr", "family", "date"])
    )


def add_features(grid: pl.LazyFrame, horizon: int = 14) -> pl.LazyFrame:
    """Add calendar, lagged, rolling and exponentially weighted demand features.

    Every demand-derived feature is shifted by ``horizon`` or more, ensuring
    that a direct horizon forecast cannot read future target values.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    keys = ["store_nbr", "family"]
    result = grid.with_columns(
        [
            pl.col("sales").log1p().alias("target"),
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.day().alias("day"),
            pl.col("date").dt.weekday().alias("weekday"),
            pl.col("date").dt.ordinal_day().alias("day_of_year"),
            (pl.col("date").dt.day() >= 25).or_(pl.col("date").dt.day() <= 5).cast(pl.Int8).alias("payday_window"),
            (pl.col("date").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        ]
    )
    for window in DEFAULT_WINDOWS:
        shifted = pl.col("sales").shift(horizon).over(keys)
        result = result.with_columns(
            [
                shifted.rolling_mean(window).over(keys).alias(f"sales_mean_{window}"),
                shifted.rolling_std(window).over(keys).fill_null(0).alias(f"sales_std_{window}"),
                (shifted == 0).cast(pl.Float64).rolling_mean(window).over(keys).alias(
                    f"zero_rate_{window}"
                ),
            ]
        )
    result = result.with_columns(
        [
            pl.col("sales").shift(horizon).over(keys).alias(f"sales_lag_{horizon}"),
            pl.col("sales").shift(horizon + 7).over(keys).alias(f"sales_lag_{horizon + 7}"),
        ]
    )
    result = result.with_columns(
        pl.col("sales").shift(horizon).ewm_mean(span=14, adjust=False).over(keys).alias("sales_ewm_14")
    )
    return result.with_columns(
        [
            pl.col("store_nbr").cast(pl.String).alias("store_id"),
            pl.col("family").cast(pl.String),
        ]
    )


def join_context(
    features: pl.LazyFrame,
    *,
    holidays: pl.LazyFrame | None = None,
    oil: pl.LazyFrame | None = None,
    transactions: pl.LazyFrame | None = None,
    stores: pl.LazyFrame | None = None,
    horizon: int = 14,
) -> pl.LazyFrame:
    """Join optional context tables without changing the target grain.

    Store attributes and holidays are known in advance and join on the target
    date. Transactions and oil prices are only observed after the fact, so
    they are joined as of ``date - horizon`` (``transactions_lag_<h>``,
    ``oil_price_lag_<h>``): a direct ``horizon``-day forecast can use them.
    Auxiliary tables are optional because the Store Sales prototype may be
    run with only its train table.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    result = features
    store_dtype = result.collect_schema().get("store_nbr", pl.Int32)
    lag = pl.duration(days=horizon)
    if stores is not None and "store_nbr" in stores.collect_schema().names():
        result = result.join(
            stores.with_columns(pl.col("store_nbr").cast(store_dtype)), on="store_nbr", how="left"
        )
    if transactions is not None:
        transaction_columns = set(transactions.collect_schema().names())
        if {"date", "store_nbr", "transactions"}.issubset(transaction_columns):
            result = result.join(
                transactions.select(
                    (pl.col("date").cast(pl.Date) + lag).alias("date"),
                    pl.col("store_nbr").cast(store_dtype),
                    pl.col("transactions")
                    .cast(pl.Float64, strict=False)
                    .alias(f"transactions_lag_{horizon}"),
                ),
                on=["date", "store_nbr"],
                how="left",
            )
    if oil is not None and {"date", "dcoilwtico"}.issubset(oil.collect_schema().names()):
        # Oil is missing on non-trading days; the as-of join carries the
        # latest known price forward.
        oil_prices = (
            oil.select(
                (pl.col("date").cast(pl.Date) + lag).alias("date"),
                pl.col("dcoilwtico").cast(pl.Float64, strict=False).alias(f"oil_price_lag_{horizon}"),
            )
            .drop_nulls()
            .sort("date")
        )
        result = result.sort("date").join_asof(oil_prices, on="date", strategy="backward")
    if holidays is not None and "date" in holidays.collect_schema().names():
        result = _join_holiday_flag(result, holidays)
    return result


def _join_holiday_flag(features: pl.LazyFrame, holidays: pl.LazyFrame) -> pl.LazyFrame:
    """Flag rows whose date is an observed holiday for that store's location.

    Transferred holidays (celebrated on another date) and ``Work Day`` rows
    (compensating working days) are excluded. National holidays apply to all
    stores; Regional and Local ones match the store's ``state`` / ``city``
    when those columns are present. Without a ``locale`` column every row is
    treated as national.
    """
    holiday_schema = holidays.collect_schema()
    holiday_columns = set(holiday_schema.names())
    feature_columns = set(features.collect_schema().names())
    observed = holidays.with_columns(pl.col("date").cast(pl.Date)).drop_nulls("date")
    if "transferred" in holiday_columns:
        transferred = (
            pl.col("transferred").str.to_lowercase() == "true"
            if holiday_schema["transferred"] == pl.String
            else pl.col("transferred").cast(pl.Boolean)
        )
        observed = observed.filter(~transferred.fill_null(False))
    if "type" in holiday_columns:
        observed = observed.filter(pl.col("type") != "Work Day")
    if "locale" not in holiday_columns:
        observed = observed.with_columns(pl.lit("National").alias("locale"))

    scopes = [("National", [], "_holiday_national")]
    if "locale_name" in holiday_columns:
        scopes += [("Regional", ["state"], "_holiday_regional"), ("Local", ["city"], "_holiday_local")]
    result = features
    flags: list[str] = []
    for locale, keys, flag in scopes:
        if not set(keys).issubset(feature_columns):
            continue
        matches = (
            observed.filter(pl.col("locale") == locale)
            .select([pl.col("date")] + [pl.col("locale_name").alias(key) for key in keys])
            .unique()
            .with_columns(pl.lit(1, dtype=pl.Int8).alias(flag))
        )
        result = result.join(matches, on=["date", *keys], how="left")
        flags.append(flag)
    return result.with_columns(
        pl.max_horizontal([pl.col(f).fill_null(0) for f in flags]).cast(pl.Int8).alias("holiday_flag")
    ).drop(flags)


def append_future(
    grid: pl.LazyFrame, horizon: int, future: pl.LazyFrame | None = None
) -> pl.LazyFrame:
    """Append ``horizon`` forecast dates per entity with unknown (null) sales.

    Lag features shifted by ``horizon`` stay computable on these rows, which
    is what lets the model forecast t+1..t+h. Promotions are known in advance:
    they are taken from ``future`` (e.g. Kaggle's ``test.csv``) when given,
    otherwise assumed to be zero.
    """
    keys = ["store_nbr", "family", "date"]
    schema = grid.collect_schema()
    last = grid.select(pl.col("date").max()).collect().item()
    if last is None:
        raise ValueError("Cannot append forecast dates to an empty grid")
    dates = pl.DataFrame(
        {"date": pl.date_range(last + timedelta(days=1), last + timedelta(days=horizon), eager=True)}
    ).lazy()
    rows = (
        grid.select(["store_nbr", "family"]).unique().join(dates, how="cross")
        .with_columns(pl.lit(None, dtype=pl.Float64).alias("sales"))
    )
    if future is not None:
        promotions = future.select(
            pl.col("store_nbr").cast(schema["store_nbr"]),
            pl.col("family").cast(pl.String),
            pl.col("date").cast(pl.Date),
            pl.col("onpromotion").cast(pl.Int32, strict=False),
        ).group_by(keys).agg(pl.col("onpromotion").max())
        rows = rows.join(promotions, on=keys, how="left")
    else:
        rows = rows.with_columns(pl.lit(None, dtype=pl.Int32).alias("onpromotion"))
    rows = rows.with_columns(pl.col("onpromotion").fill_null(0))
    rows = rows.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])
    return pl.concat([grid, rows]).sort(keys)


def build_features(
    train: pl.LazyFrame,
    horizon: int = 14,
    *,
    forecast_rows: bool = True,
    future: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Build the complete dense feature table from normalized training data.

    With ``forecast_rows`` the table also holds ``horizon`` future dates per
    entity; they have null ``sales``/``target`` and are what the model
    forecasts. Consumers that need history only should filter on non-null
    ``sales``.
    """
    grid = build_complete_grid(train)
    if forecast_rows:
        grid = append_future(grid, horizon, future)
    return add_features(grid, horizon=horizon)


def write_features(train_path: str | Path, output_path: str | Path, horizon: int = 14) -> None:
    """Read a Parquet training table lazily and persist engineered features."""
    build_features(pl.scan_parquet(train_path), horizon=horizon).sink_parquet(output_path)
