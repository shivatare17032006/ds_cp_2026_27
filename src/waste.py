"""FIFO inventory simulation and ordering policy comparison."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

RESIDUAL_WINDOW = 28
MIN_RESIDUAL_HISTORY = 7
MAX_SERVICE_QUANTILE = 0.99


@dataclass(frozen=True)
class WastePolicy:
    """Ordering policy parameters.

    ``newsvendor`` orders up to the ``quantile`` of the demand distribution,
    estimated as the forecast plus the empirical quantile of recent forecast
    errors. Other policies order up to ``forecast * (1 + safety_stock)``.
    """

    name: str
    quantile: float = 0.5
    safety_stock: float = 0.0


def critical_ratio(lost_margin: float, waste_cost: float) -> float:
    """Return the newsvendor service quantile ``cu / (cu + co)``.

    Capped at ``MAX_SERVICE_QUANTILE``: with a waste cost of zero the
    formula gives 1.0, a quantile no finite order can reach.
    """
    if lost_margin <= 0 or waste_cost < 0:
        raise ValueError("lost_margin must be positive and waste_cost non-negative")
    return min(lost_margin / (lost_margin + waste_cost), MAX_SERVICE_QUANTILE)


def _order_up_to(policy: WastePolicy, forecast: float, residuals: list[float]) -> float:
    if policy.name == "newsvendor":
        history = residuals[-RESIDUAL_WINDOW:]
        if len(history) < MIN_RESIDUAL_HISTORY:
            return forecast
        return max(forecast + float(np.quantile(history, policy.quantile)), 0.0)
    return forecast * (1 + policy.safety_stock)


def simulate_fifo(
    demand: pd.Series,
    forecast: pd.Series,
    shelf_life_days: int = 7,
    policy: WastePolicy = WastePolicy("forecast_only"),
    lost_margin: float = 1.0,
    waste_cost: float = 0.3,
) -> pd.DataFrame:
    """Simulate daily ordering, FIFO consumption, expiry and stockouts.

    A batch ordered on day 0 can be sold on days ``0 .. shelf_life_days - 1``
    and is discarded as waste at the start of day ``shelf_life_days``. Waste
    is recorded once, on the day a batch expires, so over the whole run
    ``order.sum() == sold.sum() + waste.sum() + on_hand.iloc[-1]``.
    """
    if shelf_life_days < 1:
        raise ValueError("shelf_life_days must be positive")
    if not 0 < policy.quantile < 1:
        raise ValueError("policy.quantile must be between 0 and 1")
    inventory: list[list[float]] = []
    residuals: list[float] = []
    results: list[dict[str, float]] = []
    for timestamp, actual in demand.items():
        aged = [[qty, age + 1] for qty, age in inventory]
        waste = sum(qty for qty, age in aged if age >= shelf_life_days)
        inventory = [[qty, age] for qty, age in aged if age < shelf_life_days]
        available = sum(qty for qty, _ in inventory)
        predicted = float(forecast.loc[timestamp])
        order = max(_order_up_to(policy, predicted, residuals) - available, 0.0)
        inventory.append([order, 0])
        available += order
        sold = min(float(actual), available)
        stockout = max(float(actual) - available, 0.0)
        remaining = sold
        new_inventory: list[list[float]] = []
        for quantity, age in inventory:
            consumed = min(quantity, remaining)
            remaining -= consumed
            if quantity - consumed > 0:
                new_inventory.append([quantity - consumed, age])
        inventory = new_inventory
        # Errors are only known after the day closes, so they inform later orders.
        residuals.append(float(actual) - predicted)
        results.append(
            {
                "date": timestamp,
                "demand": float(actual),
                "order": order,
                "sold": sold,
                "waste": waste,
                "stockout": stockout,
                "on_hand": sum(qty for qty, _ in inventory),
                "cost": waste * waste_cost + stockout * lost_margin,
            }
        )
    result = pd.DataFrame(results).set_index("date")
    result["service_level"] = result["sold"] / result["demand"].replace(0, np.nan)
    return result.fillna({"service_level": 1.0})


def default_policies(lost_margin: float = 1.0, waste_cost: float = 0.3) -> list[WastePolicy]:
    """Forecast +10% buffer, forecast as-is, and the cost-optimal newsvendor."""
    return [
        WastePolicy("forecast_plus_10pct", 0.5, 0.10),
        WastePolicy("forecast_only", 0.5, 0.0),
        WastePolicy("newsvendor", critical_ratio(lost_margin, waste_cost), 0.0),
    ]


def simulate_portfolio(
    frame: pd.DataFrame,
    forecast_column: str,
    shelf_life: Callable[[str], int],
    lost_margin: float = 1.0,
    waste_cost: float = 0.3,
) -> pd.DataFrame:
    """Simulate every store/family separately and aggregate per policy.

    ``frame`` needs ``store_nbr``, ``family``, ``date``, ``sales`` and the
    forecast column; each series uses its own family's shelf life. Summing
    stores into one series first would let one store's surplus cover
    another's shortfall and hide most of the waste.
    """
    totals: dict[str, dict[str, float]] = {}
    for (_, family), series in frame.sort_values("date").groupby(["store_nbr", "family"], observed=True):
        demand = series.set_index("date")["sales"]
        forecast = series.set_index("date")[forecast_column]
        for policy in default_policies(lost_margin, waste_cost):
            result = simulate_fifo(
                demand, forecast, shelf_life(str(family)), policy, lost_margin, waste_cost
            )
            total = totals.setdefault(
                policy.name,
                {"ordered": 0.0, "sold": 0.0, "demand": 0.0, "waste": 0.0, "cost": 0.0,
                 "stockout_days": 0.0, "days": 0.0},
            )
            total["ordered"] += result["order"].sum()
            total["sold"] += result["sold"].sum()
            total["demand"] += result["demand"].sum()
            total["waste"] += result["waste"].sum()
            total["cost"] += result["cost"].sum()
            total["stockout_days"] += float((result["stockout"] > 0).sum())
            total["days"] += len(result)
    return pd.DataFrame(
        [
            {
                "policy": name,
                "waste_pct": t["waste"] / max(t["ordered"], 1),
                "stockout_rate": t["stockout_days"] / max(t["days"], 1),
                "service_level": t["sold"] / max(t["demand"], 1),
                "cost": t["cost"],
                "units_ordered": t["ordered"],
                "units_wasted": t["waste"],
            }
            for name, t in totals.items()
        ]
    )


def compare_policies(
    demand: pd.Series,
    forecast: pd.Series,
    shelf_life_days: int = 7,
    lost_margin: float = 1.0,
    waste_cost: float = 0.3,
) -> pd.DataFrame:
    """Compare forecast +10%, forecast-only and cost-optimal newsvendor ordering policies."""
    policies = default_policies(lost_margin, waste_cost)
    rows = []
    for policy in policies:
        result = simulate_fifo(demand, forecast, shelf_life_days, policy, lost_margin, waste_cost)
        rows.append(
            {
                "policy": policy.name,
                "waste_pct": float(result["waste"].sum() / max(result["order"].sum(), 1)),
                "stockout_rate": float((result["stockout"] > 0).mean()),
                "service_level": float(result["sold"].sum() / max(result["demand"].sum(), 1)),
                "cost": float(result["cost"].sum()),
            }
        )
    return pd.DataFrame(rows)
