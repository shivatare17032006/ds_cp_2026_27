"""Dashboard calculations kept free of Streamlit so they can be unit tested."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd

KEYS = ["store_nbr", "family"]
RISK_WINDOW = 28
BASELINE_WINDOW = 14
HIGH_RISK = 0.6
MEDIUM_RISK = 0.4
HIGH_LABEL, MEDIUM_LABEL, LOW_LABEL = "🔴 High", "🟡 Medium", "🟢 Low"

PIPELINE_STEPS: list[tuple[str, list[str]]] = [
    ("pipeline", ["-m", "src.pipeline", "--config", "configs/base.yaml"]),
    ("train", ["-m", "src.train"]),
    ("report", ["-m", "src.report"]),
]


def stockout_risk(recent_sales: pd.Series, order: float) -> float:
    """Share of recent days on which demand exceeded ``order``.

    This is the empirical chance of a stockout if ``order`` units were
    available every day, i.e. if the store ordered exactly the forecast.
    """
    recent = recent_sales.dropna()
    return float((recent > order).mean()) if len(recent) else 0.0


def risk_level(risk: float, spike: float = 0.0, zero_rate: float = 0.0) -> str:
    if risk > HIGH_RISK or spike > 0.5 or zero_rate > 0.6:
        return HIGH_LABEL
    if risk > MEDIUM_RISK or abs(spike) > 0.25 or zero_rate > 0.35:
        return MEDIUM_LABEL
    return LOW_LABEL


def next_day_forecast(history: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Next-day forecast per store/family: the model's first future day,
    else a 14-day moving average of sales."""
    ordered = history.sort_values("date")
    baseline = (
        ordered.groupby(KEYS, observed=True)["sales"]
        .apply(lambda s: s.tail(BASELINE_WINDOW).mean())
        .rename("forecast")
        .reset_index()
        .assign(source="moving average")
    )
    baseline["family"] = baseline["family"].astype(str)
    if forecast.empty:
        return baseline
    model = (
        forecast.assign(family=forecast["family"].astype(str))
        .sort_values("date")
        .groupby(KEYS, observed=True)
        .head(1)[[*KEYS, "forecast"]]
        .assign(source="model")
    )
    merged = baseline.merge(model, on=KEYS, how="left", suffixes=("_baseline", ""))
    missing = merged["forecast"].isna()
    merged.loc[missing, "forecast"] = merged.loc[missing, "forecast_baseline"]
    merged.loc[missing, "source"] = merged.loc[missing, "source_baseline"]
    return merged[[*KEYS, "forecast", "source"]]


def compute_alerts(history: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Risk table with one row per store/family."""
    next_day = next_day_forecast(history, forecast).set_index(KEYS)
    records = []
    for (store, family), group in history.groupby(KEYS, observed=True):
        sales = group.sort_values("date")["sales"]
        avg7 = sales.tail(7).mean()
        avg_prev7 = sales.tail(14).head(7).mean()
        spike = (avg7 / max(avg_prev7, 0.1)) - 1
        zero_rate = float((sales.tail(RISK_WINDOW) == 0).mean())
        order = float(next_day.loc[(store, str(family)), "forecast"])
        risk = stockout_risk(sales.tail(RISK_WINDOW), order)
        records.append(
            {
                "Store": store,
                "Category": str(family),
                "Risk Level": risk_level(risk, spike, zero_rate),
                "Stockout Risk": risk,
                "Next-Day Forecast": order,
                "Demand Spike": spike,
                "Zero-Day Rate": zero_rate,
                "Avg 28-Day Sales": float(sales.tail(RISK_WINDOW).mean()),
            }
        )
    order = {HIGH_LABEL: 0, MEDIUM_LABEL: 1, LOW_LABEL: 2}
    table = pd.DataFrame(records)
    if table.empty:
        return table
    return (
        table.assign(_rank=table["Risk Level"].map(order))
        .sort_values(["_rank", "Stockout Risk"], ascending=[True, False])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def run_pipeline(
    root: Path,
    steps: list[tuple[str, list[str]]] = PIPELINE_STEPS,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Iterator[tuple[str, bool, str]]:
    """Run each step in order, yielding ``(name, ok, output)``; stop on failure."""
    for name, args in steps:
        try:
            result = runner(
                [sys.executable, *args], capture_output=True, text=True, cwd=str(root)
            )
        except OSError as exc:
            yield name, False, str(exc)
            return
        output = (result.stdout or "") + (result.stderr or "")
        yield name, result.returncode == 0, output
        if result.returncode != 0:
            return


def _has(message: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", message) for word in words)


def rule_based_answer(
    question: str,
    history: pd.DataFrame,
    holdout: pd.DataFrame,
    waste: pd.DataFrame,
    stores: pd.DataFrame,
    alerts: pd.DataFrame,
) -> str:
    """Offline answers for the assistant when no API key is configured."""
    message = question.lower()
    if _has(message, "hello", "hi", "hey"):
        return "👋 Hello! Ask me about stores, forecasts, categories, waste policies, or metrics."
    if _has(message, "stockout", "stock-out", "risk", "alert", "alerts"):
        if alerts.empty:
            return "No alert data available yet."
        if _has(message, "category", "categories", "family", "product"):
            by_category = alerts.groupby("Category")["Stockout Risk"].mean().sort_values(ascending=False)
            return "⚠️ **Average stockout risk by category** (if ordering the forecast):\n" + "\n".join(
                f"- {name}: {value:.0%}" for name, value in by_category.head(5).items()
            )
        top = alerts.sort_values("Stockout Risk", ascending=False).head(5)
        return "⚠️ **Highest stockout risk** (if ordering the forecast):\n" + "\n".join(
            f"- Store {store} · {category}: {risk:.0%}"
            for store, category, risk in zip(top["Store"], top["Category"], top["Stockout Risk"])
        )
    if _has(message, "wape", "accuracy", "accurate", "metric", "metrics", "rmsse", "bias", "model"):
        if holdout.empty:
            return "No holdout metrics available. Run `python -m src.report` first."
        best = holdout.loc[holdout["wape"].idxmin()]
        return (
            f"📊 **Best model: {best['model'].replace('_', ' ').title()}**\n"
            f"- WAPE (weighted absolute % error, lower is better): {best['wape']:.2%}\n"
            f"- Bias: {best['bias']:.2%}\n- RMSSE: {best['rmsse']:.3f}"
        )
    if _has(message, "waste", "policy", "policies", "newsvendor"):
        if waste.empty:
            return "No waste metrics found. Run `python -m src.report` first."
        best = waste.loc[waste["cost"].idxmin()]
        return (
            f"🏆 **Lowest-cost policy: {best['policy'].replace('_', ' ').title()}**\n"
            f"- Cost (relative units): {best['cost']:,.0f}\n"
            f"- Service level: {best['service_level']:.1%}\n"
            f"- Stockout rate: {best['stockout_rate']:.1%}"
        )
    if _has(message, "category", "categories", "family", "families", "product", "products"):
        average = history.groupby("family", observed=True)["sales"].mean().sort_values(ascending=False)
        return f"📦 **{average.index[0]}** is the top category with **{average.iloc[0]:.1f} avg daily sales**.\n\n" + "\n".join(
            f"- {name}: {value:.1f}" for name, value in average.items()
        )
    if _has(message, "store", "stores"):
        if _has(message, "highest", "top", "best", "most", "largest", "biggest"):
            average = history.groupby("store_nbr")["sales"].mean()
            return (
                f"🏪 **Store {average.idxmax()}** has the highest average daily sales "
                f"(**{average.max():.1f} units/day**).\n\nTop 5 stores:\n"
                + "\n".join(f"- Store {s}: {v:.1f} units/day" for s, v in average.nlargest(5).items())
            )
        cities = stores["city"].nunique() if "city" in stores else None
        where = f" across **{cities} cities**" if cities else ""
        return f"🏪 There are **{history['store_nbr'].nunique()} stores**{where}."
    top_store = history.groupby("store_nbr")["sales"].mean().idxmax()
    top_category = history.groupby("family", observed=True)["sales"].mean().idxmax()
    return (
        "📊 Here's a quick summary of your data:\n"
        f"- **{history['store_nbr'].nunique()} stores**, **{history['family'].nunique()} product categories**\n"
        f"- **{len(history):,} daily records** from {history['date'].min().date()} "
        f"to {history['date'].max().date()}\n"
        f"- Top store: Store **{top_store}** | Top category: **{top_category}**"
    )
