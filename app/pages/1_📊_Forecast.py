"""Demand Forecast page — interactive Plotly charts with confidence bands."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import load_all

st.set_page_config(page_title="Forecast · FreshCast", page_icon="📊", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.metric-card {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.2rem 1.5rem;
    color: white; text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 700; color: #4ade80; }
.metric-lbl { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
.alert-red { background:#7f1d1d22; border-left:4px solid #ef4444; padding:0.6rem 1rem; border-radius:8px; color:#fca5a5; }
.alert-grn { background:#14532d22; border-left:4px solid #22c55e; padding:0.6rem 1rem; border-radius:8px; color:#86efac; }
</style>
""", unsafe_allow_html=True)

data, forecast_df, stores_df, results = load_all()

st.markdown("## 📊 Demand Forecast Explorer")
st.caption("Interactive forecast with confidence bands, per store and product category.")

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Controls")
stores = sorted(data["store_nbr"].dropna().unique().tolist())
families = sorted(data["family"].dropna().astype(str).unique().tolist())
store = st.sidebar.selectbox("Store", stores, key="fc_store")
family = st.sidebar.selectbox("Category", families, key="fc_family")
lookback = st.sidebar.slider("History window (days)", 14, 120, 56, key="fc_lookback")
ci_pct = st.sidebar.slider("Confidence band (%)", 10, 40, 25, key="fc_ci")

# ── Filter data ───────────────────────────────────────────────────────────────
sel = data[(data["store_nbr"] == store) & (data["family"].astype(str) == family)].copy()
sel["date"] = pd.to_datetime(sel["date"])
sel = sel.sort_values("date")

if "forecast" not in sel.columns:
    sel["forecast"] = sel["sales"].rolling(14, min_periods=1).mean().shift(1).fillna(0)

fut = forecast_df[
    (forecast_df["store_nbr"] == store) & (forecast_df["family"].astype(str) == family)
].sort_values("date").copy() if not forecast_df.empty else pd.DataFrame()

history = sel.tail(lookback)[["date", "sales", "forecast"]].copy()
if not fut.empty:
    forecast_value = float(fut["forecast"].iloc[0])
    combined = pd.concat([history, fut[["date", "forecast"]]], ignore_index=True)
else:
    forecast_value = float(sel["forecast"].iloc[-1])
    combined = history.copy()

combined["lower"] = (combined["forecast"] * (1 - ci_pct / 100)).clip(lower=0)
combined["upper"] = combined["forecast"] * (1 + ci_pct / 100)

avg_sales = float(sel["sales"].tail(28).mean())
stockout_risk = max(0.0, 1.0 - forecast_value / max(avg_sales, 1.0))
trend_7 = sel["sales"].tail(7).mean() - sel["sales"].tail(14).head(7).mean()

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"""<div class="metric-card">
  <div class="metric-val">{forecast_value:,.0f}</div>
  <div class="metric-lbl">Recommended Order</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="metric-card">
  <div class="metric-val">{avg_sales:,.1f}</div>
  <div class="metric-lbl">28-Day Avg Sales</div>
</div>""", unsafe_allow_html=True)
risk_color = "#ef4444" if stockout_risk > 0.2 else "#4ade80"
c3.markdown(f"""<div class="metric-card">
  <div class="metric-val" style="color:{risk_color}">{stockout_risk:.1%}</div>
  <div class="metric-lbl">Stockout Risk</div>
</div>""", unsafe_allow_html=True)
trend_arrow = "↑" if trend_7 > 0 else "↓"
trend_color = "#4ade80" if trend_7 > 0 else "#f87171"
c4.markdown(f"""<div class="metric-card">
  <div class="metric-val" style="color:{trend_color}">{trend_arrow} {abs(trend_7):.1f}</div>
  <div class="metric-lbl">7-Day Trend</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Alert
if stockout_risk > 0.25:
    st.markdown(f'<div class="alert-red">⚠️ High stockout risk ({stockout_risk:.1%}) detected for Store {store} · {family}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="alert-grn">✅ Demand is stable for Store {store} · {family}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Plotly chart ──────────────────────────────────────────────────────────────
fig = go.Figure()

# Confidence band
fig.add_trace(go.Scatter(
    x=pd.concat([combined["date"], combined["date"][::-1]]),
    y=pd.concat([combined["upper"], combined["lower"][::-1]]),
    fill="toself", fillcolor="rgba(99,102,241,0.12)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip", name=f"±{ci_pct}% Band",
))

# Forecast line
fig.add_trace(go.Scatter(
    x=combined["date"], y=combined["forecast"],
    mode="lines", name="Forecast",
    line=dict(color="#818cf8", width=2.5, dash="dot"),
))

# Observed sales
if "sales" in combined.columns:
    fig.add_trace(go.Scatter(
        x=combined["date"], y=combined["sales"],
        mode="lines", name="Observed Sales",
        line=dict(color="#4ade80", width=2),
    ))

# Vertical line at today
if not fut.empty:
    cutoff_date = history["date"].max()
    fig.add_vline(x=cutoff_date, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  annotation_text="Forecast Start", annotation_font_color="#94a3b8")

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,20,40,0.6)",
    font=dict(family="Inter", color="#e2e8f0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=0, r=0, t=40, b=0),
    height=420,
    hovermode="x unified",
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, title="Units"),
)
st.plotly_chart(fig, width='stretch')

# ── Download ──────────────────────────────────────────────────────────────────
csv = combined.to_csv(index=False).encode()
st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv", "text/csv")

