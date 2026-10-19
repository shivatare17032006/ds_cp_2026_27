"""Model Comparison — leaderboard with holdout metrics and policy rankings."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from app.data_loader import load_all, load_holdout_metrics, load_waste_metrics

st.set_page_config(page_title="Model Comparison · FreshCast", page_icon="🏆", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.gold   { color:#fbbf24; font-weight:700; }
.silver { color:#94a3b8; font-weight:700; }
.leader-card { background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(139,92,246,0.1));
               border:1px solid rgba(99,102,241,0.3); border-radius:16px;
               padding:1.2rem 1.5rem; margin-bottom:0.8rem; }
</style>""", unsafe_allow_html=True)

data, forecast_df, stores_df, results = load_all()
holdout = load_holdout_metrics()
waste   = load_waste_metrics()

st.markdown("## 🏆 Model Comparison Leaderboard")
st.caption("Holdout metrics (WAPE, Bias, RMSSE) and waste-policy cost rankings.")

if holdout.empty:
    st.warning("No holdout metrics found — run `python -m src.report` first.")
else:
    # ── Leaderboard ───────────────────────────────────────────────────────────
    best_wape = holdout.loc[holdout["wape"].idxmin(), "model"]
    st.markdown(f'<div class="leader-card">🥇 <span class="gold">Best WAPE: {best_wape.replace("_"," ").title()}</span> &nbsp;·&nbsp; WAPE = {holdout["wape"].min():.2%}</div>', unsafe_allow_html=True)

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.markdown("### 📋 Holdout Metrics")
    st.dataframe(
        holdout.style.format({"wape": "{:.2%}", "bias": "{:.2%}", "rmsse": "{:.3f}"})
               .highlight_min(subset=["wape", "rmsse"], color="#14532d44")
               .highlight_min(subset=["bias"], color="#14532d44"),
        width='stretch', hide_index=True,
    )

    # ── Radar / bar charts ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### WAPE by Model")
        fig = px.bar(holdout, x="model", y="wape", color="model",
                     color_discrete_sequence=px.colors.qualitative.Vivid,
                     labels={"wape": "WAPE (lower=better)", "model": "Model"},
                     text_auto=".2%")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(15,20,40,0.6)",
                          showlegend=False, height=280, margin=dict(l=0,r=0,t=20,b=0),
                          font=dict(family="Inter"))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("#### RMSSE by Model")
        fig2 = px.bar(holdout, x="model", y="rmsse", color="model",
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      labels={"rmsse": "RMSSE (lower=better)", "model": "Model"},
                      text_auto=".3f")
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(15,20,40,0.6)",
                           showlegend=False, height=280, margin=dict(l=0,r=0,t=20,b=0),
                           font=dict(family="Inter"))
        st.plotly_chart(fig2, width='stretch')

    # ── Bias chart ────────────────────────────────────────────────────────────
    st.markdown("#### Forecast Bias (positive = over-forecast)")
    fig3 = go.Figure(go.Bar(
        x=holdout["model"], y=holdout["bias"],
        marker_color=["#f87171" if b > 0 else "#4ade80" for b in holdout["bias"]],
        text=[f"{b:.2%}" for b in holdout["bias"]],
        textposition="outside",
    ))
    fig3.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(15,20,40,0.6)",
                       height=260, margin=dict(l=0,r=0,t=20,b=0),
                       yaxis_title="Bias", font=dict(family="Inter"))
    st.plotly_chart(fig3, width='stretch')

if not waste.empty:
    st.markdown("---")
    st.markdown("### 🗑️ Waste Policy Rankings")
    best_cost = waste.loc[waste["cost"].idxmin(), "policy"]
    st.markdown(f'<div class="leader-card">🥇 <span class="gold">Lowest Cost Policy: {best_cost.replace("_"," ").title()}</span> &nbsp;·&nbsp; Cost = €{waste["cost"].min():,.0f}</div>', unsafe_allow_html=True)

    st.dataframe(
        waste.style.format({"waste_pct": "{:.1%}", "stockout_rate": "{:.1%}",
                            "service_level": "{:.1%}", "cost": "€{:,.0f}"})
             .highlight_min(subset=["cost", "stockout_rate"], color="#14532d44")
             .highlight_max(subset=["service_level"], color="#14532d44"),
        width='stretch', hide_index=True,
    )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Policy Cost Comparison")
        fig4 = px.bar(waste, x="policy", y="cost", color="policy",
                      color_discrete_sequence=["#f87171","#fbbf24","#34d399"],
                      labels={"cost":"Total Cost (€)","policy":"Policy"}, text_auto=",.0f")
        fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(15,20,40,0.6)", showlegend=False,
                           height=280, margin=dict(l=0,r=0,t=20,b=0), font=dict(family="Inter"))
        st.plotly_chart(fig4, width='stretch')

    with col4:
        st.markdown("#### Service Level vs Stockout Rate")
        fig5 = px.scatter(waste, x="stockout_rate", y="service_level",
                          text="policy", size="cost",
                          color_discrete_sequence=["#818cf8"],
                          labels={"stockout_rate":"Stockout Rate","service_level":"Service Level"})
        fig5.update_traces(textposition="top center")
        fig5.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(15,20,40,0.6)",
                           height=280, margin=dict(l=0,r=0,t=20,b=0), font=dict(family="Inter"))
        st.plotly_chart(fig5, width='stretch')

