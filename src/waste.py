"""FIFO inventory simulation and ordering policy comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WastePolicy:
    """Ordering policy parameters."""

    name: str
    quantile: float = 0.5
    safety_stock: float = 0.0


def simulate_fifo(
    demand: pd.Series,
    forecast: pd.Series,
    shelf_life_days: int = 7,
    policy: WastePolicy = WastePolicy("median"),
    lost_margin: float = 1.0,
    waste_cost: float = 0.3,
) -> pd.DataFrame:
    """Simulate daily ordering, FIFO consumption, expiry and stockouts."""
    if shelf_life_days < 1:
        raise ValueError("shelf_life_days must be positive")
    inventory: list[list[float]] = []
    results: list[dict[str, float]] = []
    for timestamp, actual in demand.items():
        inventory = [[qty, age + 1] for qty, age in inventory if age + 1 < shelf_life_days]
        available = sum(qty for qty, _ in inventory)
        order = max(float(forecast.loc[timestamp]) * (1 + policy.safety_stock) - available, 0.0)
        if policy.name == "newsvendor":
            order *= policy.quantile
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
        waste = sum(quantity for quantity, _ in inventory)
        results.append(
            {
                "date": timestamp,
                "demand": float(actual),
                "order": order,
                "sold": sold,
                "waste": waste,
                "stockout": stockout,
                "cost": waste * waste_cost + stockout * lost_margin,
            }
        )
    result = pd.DataFrame(results).set_index("date")
    result["service_level"] = result["sold"] / result["demand"].replace(0, np.nan)
    return result.fillna({"service_level": 1.0})


def compare_policies(
    demand: pd.Series, forecast: pd.Series, shelf_life_days: int = 7
) -> pd.DataFrame:
    """Compare moving-average, median, and newsvendor ordering policies."""
    policies = [
        WastePolicy("moving_average", 0.5, 0.10),
        WastePolicy("median", 0.5, 0.0),
        WastePolicy("newsvendor", 0.9, 0.0),
    ]
    rows = []
    for policy in policies:
        result = simulate_fifo(demand, forecast, shelf_life_days, policy)
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
