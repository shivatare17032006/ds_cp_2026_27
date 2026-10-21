"""Ecuador Store Map — demand bubbles and waste risk by location."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data_loader import ECUADOR_COORDS, load_alerts, load_all
from app.theme import PLOT_LAYOUT, apply_theme

st.set_page_config(page_title="Store Map · FreshCast", page_icon="🗺️", layout="wide")
apply_theme()

data, forecast_df, stores_df, results = load_all()

st.markdown("## 🗺️ Ecuador Store Network")
st.caption(
    f"Demand volume and average stockout risk across {data['store_nbr'].nunique()} store locations. "
    "Stores are placed at their city's coordinates with a small offset so they don't overlap."
)

# ── Build map data ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_map_data(data: pd.DataFrame, stores_df: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    agg = (
        data.groupby("store_nbr", observed=True)["sales"]
        .agg(total_sales="sum", avg_daily="mean", days="count")
        .reset_index()
    )
    if not alerts.empty:
        risk = alerts.groupby("Store")["Stockout Risk"].mean().rename("stockout_risk")
        agg = agg.merge(risk, left_on="store_nbr", right_index=True, how="left")
    else:
        agg["stockout_risk"] = 0.0
    if not stores_df.empty:
        agg = agg.merge(stores_df[["store_nbr", "city", "state", "type", "cluster"]],
                        on="store_nbr", how="left")
    else:
        agg["city"] = "Quito"
        agg["state"] = "Pichincha"
        agg["type"] = "D"
        agg["cluster"] = 1

    agg["lat"] = agg["city"].map(lambda c: ECUADOR_COORDS.get(c, (-1.8, -78.2))[0])
    agg["lon"] = agg["city"].map(lambda c: ECUADOR_COORDS.get(c, (-1.8, -78.2))[1])
    # Add slight jitter so stores in same city don't overlap
    import numpy as np
    rng = np.random.default_rng(42)
    agg["lat"] += rng.uniform(-0.05, 0.05, len(agg))
    agg["lon"] += rng.uniform(-0.05, 0.05, len(agg))
    return agg

with st.spinner("Building store map…"):
    map_df = build_map_data(data, stores_df, load_alerts())

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
color_by = col1.selectbox("Color by", ["total_sales", "avg_daily", "stockout_risk", "cluster", "type"],
                          key="map_color")
size_by  = col2.selectbox("Bubble size by", ["total_sales", "avg_daily", "days"], key="map_size")

# ── Map ───────────────────────────────────────────────────────────────────────
continuous_scale = {"stockout_risk": "RdYlGn_r", "type": None}.get(color_by, "Viridis")
fig = px.scatter_map(
    map_df,
    lat="lat", lon="lon",
    size=size_by, color=color_by,
    hover_name="store_nbr",
    hover_data={"city": True, "state": True, "type": True,
                "total_sales": ":.0f", "avg_daily": ":.1f", "stockout_risk": ":.0%",
                "lat": False, "lon": False},
    color_continuous_scale=continuous_scale,
    size_max=30,
    zoom=5.5,
    center={"lat": -1.8, "lon": -78.2},
    map_style="carto-darkmatter",
    title="Ecuador Store Demand Bubbles",
)
fig.update_layout(
    **PLOT_LAYOUT,
    margin=dict(l=0, r=0, t=40, b=0),
    height=580,
    coloraxis_colorbar=dict(title=color_by.replace("_", " ").title()),
)
st.plotly_chart(fig, width='stretch')

# ── City-level bar ────────────────────────────────────────────────────────────
st.markdown("### 📊 Total Sales by City")
city_sales = map_df.groupby("city")["total_sales"].sum().sort_values(ascending=False).reset_index()
fig2 = px.bar(
    city_sales, x="city", y="total_sales",
    color="total_sales", color_continuous_scale="Teal",
    labels={"total_sales": "Total Sales", "city": "City"},
)
fig2.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,20,40,0.6)",
    margin=dict(l=0, r=0, t=20, b=0), height=300,
    font=dict(family="Inter"),
    showlegend=False,
)
st.plotly_chart(fig2, width='stretch')

# ── Table ─────────────────────────────────────────────────────────────────────
with st.expander("📋 Store Details Table"):
    st.dataframe(
        map_df[["store_nbr","city","state","type","cluster","total_sales","avg_daily"]]
        .sort_values("total_sales", ascending=False)
        .style.format({"total_sales": "{:,.0f}", "avg_daily": "{:,.2f}"}),
        width='stretch', hide_index=True,
    )

