"""Streamlit dashboard for forecast review and waste-policy comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# `streamlit run app/streamlit_app.py` only puts app/ on the path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.waste import compare_policies  # noqa: E402
FEATURE_PATH = ROOT / "data" / "processed" / "features.parquet"
FORECAST_PATH = ROOT / "data" / "artifacts" / "forecast.parquet"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return observed history and the model's future forecast (may be empty)."""
    future = pd.DataFrame(columns=["store_nbr", "family", "date", "forecast"])
    if FEATURE_PATH.exists():
        frame = pd.read_parquet(FEATURE_PATH, columns=["date", "store_nbr", "family", "sales"])
        # Forecast rows appended by the feature build have no observed sales.
        frame = frame[frame["sales"].notna()].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        if FORECAST_PATH.exists():
            future = pd.read_parquet(FORECAST_PATH)
            future["date"] = pd.to_datetime(future["date"])
        return frame, future
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    demo = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "store_nbr": [1] * 120 + [2] * 120,
            "family": ["PRODUCE"] * 240,
            "sales": [10 + (i % 7) * 2 for i in range(120)] * 2,
            "forecast": [11.0] * 240,
        }
    )
    return demo, future


st.set_page_config(page_title="FreshCast | Supply Chain Control Tower", page_icon="🥬", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1400px; padding-top: 2rem;}
    .hero {padding: 1.4rem 1.8rem; border-radius: 18px;
           background: linear-gradient(120deg,#0b3d2e,#197c5a); color:white;}
    .hero h1 {margin:0; font-size:2.4rem;}
    .muted {color:#b9d8c8;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>FreshCast Control Tower</h1><p class="muted">'
    "Demand forecasts, order recommendations, and simulated waste impact.</p></div>",
    unsafe_allow_html=True,
)

data, future_forecast = load_data()
st.sidebar.header("Scenario controls")
stores = sorted(data["store_nbr"].dropna().unique().tolist())
families = sorted(data["family"].dropna().astype(str).unique().tolist())
store = st.sidebar.selectbox("Store", stores)
family = st.sidebar.selectbox("Category", families)
shelf_life = st.sidebar.slider("Shelf life (days)", min_value=1, max_value=30, value=7)

selection = data[(data["store_nbr"] == store) & (data["family"].astype(str) == family)].copy()
selection["date"] = pd.to_datetime(selection["date"])
selection = selection.sort_values("date")
if "forecast" not in selection:
    # Historical baseline used for the policy simulation; the model only
    # forecasts future dates.
    selection["forecast"] = selection["sales"].rolling(14, min_periods=1).mean().shift(1).fillna(0)
future = future_forecast[
    (future_forecast["store_nbr"] == store) & (future_forecast["family"].astype(str) == family)
].sort_values("date")

latest = selection.tail(28)[["date", "sales", "forecast"]]
if future.empty:
    forecast_value = float(selection["forecast"].iloc[-1])
    plotted = latest
else:
    forecast_value = float(future["forecast"].iloc[0])
    plotted = pd.concat([latest, future[["date", "forecast"]]], ignore_index=True)
plotted["lower"] = (plotted["forecast"] * 0.75).clip(lower=0)
plotted["upper"] = plotted["forecast"] * 1.25
avg_sales = float(selection["sales"].tail(28).mean())
stockout_risk = max(0.0, 1.0 - forecast_value / max(avg_sales, 1.0))

metric_cols = st.columns(4)
metric_cols[0].metric("Recommended order", f"{forecast_value:,.1f} units")
metric_cols[1].metric("28-day demand", f"{avg_sales:,.1f}/day")
metric_cols[2].metric("Stockout risk", f"{stockout_risk:.1%}")
metric_cols[3].metric("Data coverage", f"{len(selection):,} days")

st.subheader("Demand and forecast")
if future.empty:
    st.caption("No model forecast found; run `python -m src.train`. Showing a 14-day moving average.")
else:
    st.caption(f"Model forecast for the next {len(future)} days follows the observed history.")
chart = plotted.set_index("date")[["sales", "forecast", "lower", "upper"]].rename(
    columns={"sales": "Observed sales", "forecast": "Forecast"}
)
st.line_chart(chart)

st.subheader("Ordering policy simulation")
demand = selection.set_index("date")["sales"].tail(56)
forecast = selection.set_index("date")["forecast"].reindex(demand.index).fillna(forecast_value)
policy_table = compare_policies(demand, forecast, shelf_life)
st.dataframe(
    policy_table.style.format(
        {"waste_pct": "{:.1%}", "stockout_rate": "{:.1%}", "service_level": "{:.1%}", "cost": "{:,.2f}"}
    ),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Methodology and caveats"):
    st.write(
        "Forecast intervals and model SHAP views appear once processed model artifacts are available. "
        "Waste is simulated with FIFO inventory and configurable shelf life; Kaggle sales do not contain "
        "observed waste or inventory records."
    )
