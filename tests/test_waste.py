import numpy as np
import pandas as pd
import pytest

from src.waste import WastePolicy, compare_policies, critical_ratio, simulate_fifo


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)), dtype=float)


def test_waste_is_counted_once_at_expiry() -> None:
    # Ordering 6 against demand 5 leaves 1 unit per day. FIFO sells the oldest
    # stock first, and the order-up-to rule never orders above forecast, so
    # only the very first surplus unit can ever age out.
    result = simulate_fifo(_series([5] * 10), _series([6] * 10), shelf_life_days=3)
    assert result["order"].sum() == pytest.approx(51.0)
    assert result["sold"].sum() == pytest.approx(50.0)
    assert result["waste"].sum() <= 1.0


def test_unsold_batch_expires_on_shelf_life_day() -> None:
    demand = _series([0, 0, 0, 0])
    forecast = _series([4, 0, 0, 0])
    result = simulate_fifo(demand, forecast, shelf_life_days=3)
    assert result["waste"].tolist() == [0.0, 0.0, 0.0, 4.0]
    assert result["on_hand"].tolist() == [4.0, 4.0, 4.0, 0.0]


@pytest.mark.parametrize("policy", ["median", "moving_average", "newsvendor"])
def test_inventory_is_conserved(policy: str) -> None:
    rng = np.random.default_rng(0)
    demand = _series(rng.poisson(8, 60).tolist())
    forecast = _series((rng.normal(8, 2, 60).clip(0)).tolist())
    result = simulate_fifo(demand, forecast, 4, WastePolicy(policy, 0.77, 0.1))
    assert result["order"].sum() == pytest.approx(
        result["sold"].sum() + result["waste"].sum() + result["on_hand"].iloc[-1]
    )


def test_critical_ratio() -> None:
    assert critical_ratio(1.0, 0.3) == pytest.approx(1 / 1.3)
    with pytest.raises(ValueError):
        critical_ratio(0.0, 0.3)


def test_high_quantile_newsvendor_reduces_stockouts_vs_median() -> None:
    rng = np.random.default_rng(1)
    demand = _series(rng.poisson(10, 90).tolist())
    forecast = _series([10.0] * 90)
    median = simulate_fifo(demand, forecast, 3, WastePolicy("median"))
    newsvendor = simulate_fifo(demand, forecast, 3, WastePolicy("newsvendor", 0.9))
    assert newsvendor["stockout"].sum() < median["stockout"].sum()
    assert newsvendor["order"].sum() >= median["order"].sum()


def test_newsvendor_uses_only_past_errors() -> None:
    # Before enough history exists, newsvendor orders exactly the forecast;
    # a demand spike on day 0 must not change the day-0 order.
    demand = _series([100, 5, 5])
    forecast = _series([5, 5, 5])
    result = simulate_fifo(demand, forecast, 3, WastePolicy("newsvendor", 0.9))
    assert result["order"].iloc[0] == pytest.approx(5.0)


def test_compare_policies_uses_cost_based_quantile() -> None:
    rng = np.random.default_rng(2)
    demand = _series(rng.poisson(10, 60).tolist())
    forecast = _series([10.0] * 60)
    table = compare_policies(demand, forecast, shelf_life_days=3).set_index("policy")
    assert table.loc["newsvendor", "service_level"] > table.loc["median", "service_level"]


def test_invalid_quantile_rejected() -> None:
    with pytest.raises(ValueError, match="quantile"):
        simulate_fifo(_series([1]), _series([1]), 3, WastePolicy("newsvendor", 1.0))
