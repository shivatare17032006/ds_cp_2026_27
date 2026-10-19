from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.data.features import build_features, join_context
from src.forecasting import prepare_features, seasonal_naive, train_global_model

HORIZON = 7
START = date(2024, 1, 1)
DAYS = 90


def _train(stores: int = 2, families: tuple[str, ...] = ("DAIRY", "PRODUCE")) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for store in range(1, stores + 1):
        for family in families:
            for day in range(DAYS):
                current = START + timedelta(days=day)
                base = 10 * store + (5 if family == "DAIRY" else 0)
                rows.append(
                    {
                        "date": current,
                        "store_nbr": store,
                        "family": family,
                        "sales": float(base + 3 * (current.weekday() >= 5) + rng.poisson(2)),
                        "onpromotion": int(day % 10 == 0),
                    }
                )
    return pl.DataFrame(rows).with_columns(pl.col("store_nbr").cast(pl.Int32))


def _context(train: pl.DataFrame) -> dict[str, pl.LazyFrame]:
    dates = train["date"].unique().sort()
    transactions = train.group_by(["date", "store_nbr"]).agg(
        (pl.col("sales").sum() * 3).alias("transactions")
    )
    oil = pl.DataFrame({"date": dates, "dcoilwtico": np.linspace(50, 80, len(dates))})
    return {"transactions": transactions.lazy(), "oil": oil.lazy()}


def _features(train: pl.DataFrame) -> pl.DataFrame:
    context = _context(train)
    return (
        join_context(build_features(train.lazy(), horizon=HORIZON), horizon=HORIZON, **context)
        .collect()
        .sort(["store_nbr", "family", "date"])
    )


def test_features_do_not_depend_on_recent_sales_or_context() -> None:
    train = _train()
    cutoff = START + timedelta(days=60)
    baseline = _features(train)
    # Scramble everything observed in the last `HORIZON` days before the cutoff
    # and afterwards; nothing known only after cutoff - horizon may change
    # the features at the cutoff.
    perturbed = train.with_columns(
        pl.when(pl.col("date") > cutoff - timedelta(days=HORIZON))
        .then(pl.col("sales") * 7 + 100)
        .otherwise(pl.col("sales"))
        .alias("sales")
    )
    changed = _features(perturbed)
    at_cutoff = pl.col("date") == cutoff
    columns = [
        c for c in baseline.columns
        if c.startswith(("sales_", "zero_rate_", "transactions_lag", "oil_price_lag"))
    ]
    assert f"transactions_lag_{HORIZON}" in columns
    assert f"oil_price_lag_{HORIZON}" in columns
    assert baseline.filter(at_cutoff).select(columns).equals(changed.filter(at_cutoff).select(columns))


def test_oil_and_transactions_are_lagged_by_horizon() -> None:
    train = _train(stores=1, families=("DAIRY",))
    features = _features(train)
    context = _context(train)
    oil = context["oil"].collect()
    row = features.filter(pl.col("date") == START + timedelta(days=30))
    expected = oil.filter(pl.col("date") == START + timedelta(days=30 - HORIZON))["dcoilwtico"]
    assert row[f"oil_price_lag_{HORIZON}"].item() == pytest.approx(expected.item())


def test_forecast_rows_are_appended_with_lag_features() -> None:
    train = _train()
    features = build_features(train.lazy(), horizon=HORIZON).collect()
    future = features.filter(pl.col("sales").is_null())
    assert future.height == 4 * HORIZON
    last = train["date"].max()
    assert future["date"].min() == last + timedelta(days=1)
    assert future["date"].max() == last + timedelta(days=HORIZON)
    assert future["target"].null_count() == future.height
    assert future[f"sales_lag_{HORIZON}"].null_count() == 0
    assert future["sales_mean_28"].null_count() == 0


def test_future_promotions_come_from_future_table() -> None:
    train = _train(stores=1, families=("DAIRY",))
    first_future = train["date"].max() + timedelta(days=1)
    promotions = pl.DataFrame(
        {"date": [first_future], "store_nbr": [1], "family": ["DAIRY"], "onpromotion": [12]}
    )
    features = build_features(train.lazy(), horizon=HORIZON, future=promotions.lazy()).collect()
    future = features.filter(pl.col("sales").is_null()).sort("date")
    assert future["onpromotion"].to_list() == [12] + [0] * (HORIZON - 1)


def test_prepare_features_shares_categories_and_keeps_family() -> None:
    frame = pd.DataFrame(
        {"date": [1, 2], "sales": [1.0, None], "target": [0.1, None],
         "family": ["DAIRY", "PRODUCE"], "store_id": ["1", "1"]}
    )
    prepared, columns = prepare_features(frame)
    assert columns == ["family"]
    assert list(prepared["family"].cat.categories) == ["DAIRY", "PRODUCE"]


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost"])
def test_global_model_forecasts_each_future_day(model_name: str) -> None:
    pytest.importorskip(model_name)
    train = _train()
    features = _features(train).to_pandas()
    _, forecast, columns = train_global_model(features, horizon=HORIZON, model_name=model_name)
    assert "family" in columns
    assert len(forecast) == 4 * HORIZON
    assert sorted(forecast["horizon_day"].unique()) == list(range(1, HORIZON + 1))
    last = pd.Timestamp(train["date"].max())
    assert forecast["date"].min() == last + pd.Timedelta(days=1)
    assert (forecast["forecast"] >= 0).all()
    # The model must tell families apart: DAIRY series are 5 units higher.
    by_family = forecast.groupby("family")["forecast"].mean()
    assert by_family["DAIRY"] > by_family["PRODUCE"]


def test_seasonal_naive_returns_tail_of_every_series() -> None:
    frame = _train().to_pandas()
    result = seasonal_naive(frame, horizon=5)
    assert len(result) == 4 * 5
    assert result.groupby(["store_nbr", "family"]).size().eq(5).all()
