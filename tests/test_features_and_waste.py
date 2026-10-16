import pandas as pd
import polars as pl
from datetime import date

from src.data.features import build_complete_grid, add_features
from src.waste import WastePolicy, simulate_fifo


def test_grid_fills_missing_days_with_zero() -> None:
    frame = pl.DataFrame(
        {
            "date": pl.date_range(date(2024, 1, 1), date(2024, 1, 3), eager=True),
            "store_nbr": [1, 1, 2],
            "family": ["PRODUCE", "PRODUCE", "PRODUCE"],
            "sales": [2.0, 0.0, 4.0],
            "onpromotion": [0, 0, 1],
        }
    )
    result = build_complete_grid(frame.lazy()).collect()
    assert result.height == 6
    assert result.filter((pl.col("store_nbr") == 2) & (pl.col("date") == pl.date(2024, 1, 1)))[
        "sales"
    ].item() == 0


def test_features_use_horizon_shift() -> None:
    dates = pl.date_range(date(2024, 1, 1), date(2024, 2, 1), eager=True)
    frame = pl.DataFrame(
        {
            "date": dates,
            "store_nbr": [1] * len(dates),
            "family": ["PRODUCE"] * len(dates),
            "sales": list(range(len(dates))),
            "onpromotion": [0] * len(dates),
        }
    )
    result = add_features(frame.lazy(), horizon=3).collect()
    assert result["sales_mean_7"][3] is None
    assert result["sales_mean_7"][10] == 4.0


def test_fifo_simulation_has_service_and_waste_columns() -> None:
    index = pd.date_range("2024-01-01", periods=5)
    demand = pd.Series([5, 5, 5, 5, 5], index=index)
    forecast = pd.Series([5, 5, 5, 5, 5], index=index)
    result = simulate_fifo(demand, forecast, shelf_life_days=3, policy=WastePolicy("median"))
    assert {"waste", "stockout", "service_level"}.issubset(result.columns)
