"""Alerts & Anomaly Detection page — red/yellow/green risk badges per store+category."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data_loader import load_alerts, load_all
from app.insights import HIGH_LABEL, HIGH_RISK, LOW_LABEL, MEDIUM_LABEL, MEDIUM_RISK
from app.theme import PLOT_LAYOUT, apply_theme

st.set_page_config(page_title="Alerts · FreshCast", page_icon="⚠️", layout="wide")
apply_theme(".section-hd { font-size:1.1rem; font-weight:600; color:#e2e8f0; margin: 1.2rem 0 0.4rem; }")

data, forecast_df, stores_df, results = load_all()

st.markdown("## ⚠️ Alerts & Anomaly Detection")
st.caption(
    "Stockout risk = share of the last 28 days on which demand exceeded the next-day forecast, "
    "i.e. the chance of running out if you order exactly the forecast. "
    f"High above {HIGH_RISK:.0%}, medium above {MEDIUM_RISK:.0%}; sharp demand spikes and "
    "frequent zero-sales days also raise the level."
)

n_stores, n_categories = data["store_nbr"].nunique(), data["family"].nunique()
with st.spinner(f"Scanning {n_stores} stores × {n_categories} categories…"):
    alert_df = load_alerts()

high   = alert_df[alert_df["Risk Level"] == HIGH_LABEL]
medium = alert_df[alert_df["Risk Level"] == MEDIUM_LABEL]
low    = alert_df[alert_df["Risk Level"] == LOW_LABEL]

# ── Summary KPIs ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 High Risk", len(high))
c2.metric("🟡 Medium Risk", len(medium))
c3.metric("🟢 Low Risk", len(low))
c4.metric("Total Monitored", len(alert_df))

st.markdown("---")

# ── Filter controls ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
filter_level = col1.multiselect("Filter by Risk Level", [HIGH_LABEL, MEDIUM_LABEL, LOW_LABEL],
                                 default=[HIGH_LABEL, MEDIUM_LABEL])
filter_cat   = col2.multiselect("Filter by Category", sorted(alert_df["Category"].unique()),
                                 default=[])

filtered = alert_df[alert_df["Risk Level"].isin(filter_level)]
if filter_cat:
    filtered = filtered[filtered["Category"].isin(filter_cat)]

# ── Table ─────────────────────────────────────────────────────────────────────
def fmt_pct(v: float) -> str:
    return f"{v:.1%}"
def fmt_num(v: float) -> str:
    return f"{v:,.1f}"

st.dataframe(
    filtered.style
        .format({"Stockout Risk": fmt_pct, "Demand Spike": fmt_pct,
                 "Zero-Day Rate": fmt_pct, "Avg 28-Day Sales": fmt_num,
                 "Next-Day Forecast": fmt_num})
        .map(lambda v: "color:#fca5a5" if "High" in str(v) else
                            ("color:#fde68a" if "Medium" in str(v) else "color:#86efac"),
                  subset=["Risk Level"]),
    width='stretch', hide_index=True,
)

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hd">Stockout Risk Heatmap — Store × Category</div>', unsafe_allow_html=True)
pivot = alert_df.pivot_table(index="Store", columns="Category", values="Stockout Risk", aggfunc="mean").fillna(0)
fig = px.imshow(
    pivot, color_continuous_scale="RdYlGn_r",
    labels=dict(color="Stockout Risk"),
    aspect="auto", text_auto=".0%",
)
fig.update_layout(**PLOT_LAYOUT, margin=dict(l=0, r=0, t=20, b=0), height=420)
st.plotly_chart(fig, width='stretch')

# ── Download ──────────────────────────────────────────────────────────────────
csv = alert_df.to_csv(index=False).encode()
st.download_button("⬇️ Download Alert Report CSV", csv, "alerts.csv", "text/csv")

