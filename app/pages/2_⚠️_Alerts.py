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

from app.data_loader import load_all

st.set_page_config(page_title="Alerts · FreshCast", page_icon="⚠️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.badge-red  { background:#7f1d1d; color:#fca5a5; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
.badge-yel  { background:#78350f; color:#fde68a; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
.badge-grn  { background:#14532d; color:#86efac; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
.section-hd { font-size:1.1rem; font-weight:600; color:#e2e8f0; margin: 1.2rem 0 0.4rem; }
</style>
""", unsafe_allow_html=True)

data, forecast_df, stores_df, results = load_all()

st.markdown("## ⚠️ Alerts & Anomaly Detection")
st.caption("Auto-detected stockout risks, demand spikes, and anomalies across all stores and categories.")

# ── Compute per-entity risk metrics ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_alerts(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (store, family), grp in data.groupby(["store_nbr", "family"], observed=True):
        grp = grp.sort_values("date")
        sales = grp["sales"]
        avg28 = sales.tail(28).mean()
        avg7  = sales.tail(7).mean()
        avg_prev7 = sales.tail(14).head(7).mean()
        spike = (avg7 / max(avg_prev7, 0.1)) - 1  # relative spike
        zero_rate = (sales.tail(28) == 0).mean()   # fraction of zero-demand days

        if "forecast" in grp.columns:
            forecast_val = grp["forecast"].iloc[-1]
        else:
            forecast_val = avg28

        stockout_risk = max(0.0, 1.0 - forecast_val / max(avg28, 1.0))

        if stockout_risk > 0.25 or spike > 0.5 or zero_rate > 0.6:
            level = "🔴 High"
        elif stockout_risk > 0.1 or abs(spike) > 0.25 or zero_rate > 0.35:
            level = "🟡 Medium"
        else:
            level = "🟢 Low"

        records.append({
            "Store": store,
            "Category": str(family),
            "Risk Level": level,
            "Stockout Risk": stockout_risk,
            "Demand Spike": spike,
            "Zero-Day Rate": zero_rate,
            "Avg 28-Day Sales": avg28,
        })
    return pd.DataFrame(records).sort_values(["Risk Level", "Stockout Risk"], ascending=[True, False])

with st.spinner("Scanning all 54 stores × 9 categories…"):
    alert_df = compute_alerts(data)

high   = alert_df[alert_df["Risk Level"] == "🔴 High"]
medium = alert_df[alert_df["Risk Level"] == "🟡 Medium"]
low    = alert_df[alert_df["Risk Level"] == "🟢 Low"]

# ── Summary KPIs ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 High Risk", len(high))
c2.metric("🟡 Medium Risk", len(medium))
c3.metric("🟢 Low Risk", len(low))
c4.metric("Total Monitored", len(alert_df))

st.markdown("---")

# ── Filter controls ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
filter_level = col1.multiselect("Filter by Risk Level", ["🔴 High", "🟡 Medium", "🟢 Low"],
                                 default=["🔴 High", "🟡 Medium"])
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
                 "Zero-Day Rate": fmt_pct, "Avg 28-Day Sales": fmt_num})
        .applymap(lambda v: "color:#fca5a5" if "High" in str(v) else
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
fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,20,40,0.6)",
    margin=dict(l=0, r=0, t=20, b=0), height=420,
    font=dict(family="Inter"),
)
st.plotly_chart(fig, width='stretch')

# ── Download ──────────────────────────────────────────────────────────────────
csv = alert_df.to_csv(index=False).encode()
st.download_button("⬇️ Download Alert Report CSV", csv, "alerts.csv", "text/csv")

