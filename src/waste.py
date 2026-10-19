"""FIFO inventory simulation and ordering policy comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RESIDUAL_WINDOW = 28
MIN_RESIDUAL_HISTORY = 7


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
    """Return the newsvendor service quantile ``cu / (cu + co)``."""
    if lost_margin <= 0 or waste_cost < 0:
        raise ValueError("lost_margin must be positive and waste_cost non-negative")
    return lost_margin / (lost_margin + waste_cost)


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
    policy: WastePolicy = WastePolicy("median"),
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


def compare_policies(
    demand: pd.Series,
    forecast: pd.Series,
    shelf_life_days: int = 7,
    lost_margin: float = 1.0,
    waste_cost: float = 0.3,
) -> pd.DataFrame:
    """Compare moving-average, median, and cost-optimal newsvendor ordering policies."""
    policies = [
        WastePolicy("moving_average", 0.5, 0.10),
        WastePolicy("median", 0.5, 0.0),
        WastePolicy("newsvendor", critical_ratio(lost_margin, waste_cost), 0.0),
    ]
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
