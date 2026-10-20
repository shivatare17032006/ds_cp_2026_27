"""What-If Scenario Planner — interactive sliders, live waste + cost recalculation."""

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
from app.theme import apply_theme
from src.waste import compare_policies, simulate_fifo, WastePolicy, critical_ratio

st.set_page_config(page_title="Scenario Planner · FreshCast", page_icon="🎛️", layout="wide")
apply_theme("""
.kpi-box { background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.3);
           border-radius:12px; padding:1rem; text-align:center; }
.kpi-val { font-size:1.8rem; font-weight:700; color:#a5b4fc; }
.kpi-lbl { font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; }
.vs-arrow { font-size:2rem; display:flex; align-items:center; justify-content:center; color:#6366f1; }
""")

data, forecast_df, stores_df, results = load_all()

st.markdown("## 🎛️ What-If Scenario Planner")
st.caption("Adjust ordering assumptions and instantly see the impact on waste, cost and service level.")

# ── Store/Category selector ───────────────────────────────────────────────────
stores  = sorted(data["store_nbr"].dropna().unique().tolist())
families = sorted(data["family"].dropna().astype(str).unique().tolist())
c1, c2 = st.columns(2)
store  = c1.selectbox("Store", stores, key="wp_store")
family = c2.selectbox("Category", families, key="wp_family")

sel = data[(data["store_nbr"] == store) & (data["family"].astype(str) == family)].copy()
sel["date"] = pd.to_datetime(sel["date"])
sel = sel.sort_values("date")
if "forecast" not in sel.columns:
    sel["forecast"] = sel["sales"].rolling(14, min_periods=1).mean().shift(1).fillna(0)

demand   = sel.set_index("date")["sales"].tail(56)
forecast = sel.set_index("date")["forecast"].reindex(demand.index).fillna(demand.mean())

st.markdown("---")
st.markdown("### ⚙️ Scenario A vs Scenario B")

col_a, col_mid, col_b = st.columns([5, 1, 5])

with col_a:
    st.markdown("**📦 Scenario A (Baseline)**")
    shelf_a      = st.slider("Shelf life (days)", 1, 30, 7,  key="sh_a")
    safety_a     = st.slider("Safety stock (%)",  0, 50, 10, key="ss_a", help="Ignored by newsvendor, which sizes its buffer from the lost-margin / waste-cost ratio.") / 100
    lost_margin_a = st.slider("Lost margin (per unit)", 0.1, 5.0, 1.0, step=0.1, key="lm_a")
    waste_cost_a  = st.slider("Waste cost (per unit)",  0.0, 3.0, 0.3, step=0.1, key="wc_a")
    policy_a_name = st.selectbox("Policy", ["moving_average", "median", "newsvendor"], key="pol_a")

with col_mid:
    st.markdown('<div class="vs-arrow">VS</div>', unsafe_allow_html=True)

with col_b:
    st.markdown("**🚀 Scenario B (Improved)**")
    shelf_b      = st.slider("Shelf life (days)", 1, 30, 7,  key="sh_b")
    safety_b     = st.slider("Safety stock (%)",  0, 50, 5,  key="ss_b", help="Ignored by newsvendor, which sizes its buffer from the lost-margin / waste-cost ratio.") / 100
    lost_margin_b = st.slider("Lost margin (per unit)", 0.1, 5.0, 1.0, step=0.1, key="lm_b")
    waste_cost_b  = st.slider("Waste cost (per unit)",  0.0, 3.0, 0.5, step=0.1, key="wc_b")
    policy_b_name = st.selectbox("Policy", ["newsvendor", "moving_average", "median"], key="pol_b")


def run_scenario(demand, forecast, shelf, safety, lost_m, waste_c, pol_name):
    if pol_name == "newsvendor":
        q = critical_ratio(lost_m, waste_c)
        policy = WastePolicy("newsvendor", q, 0.0)
    else:
        policy = WastePolicy(pol_name, 0.5, safety)
    return simulate_fifo(demand, forecast, shelf, policy, lost_m, waste_c)


res_a = run_scenario(demand, forecast, shelf_a, safety_a, lost_margin_a, waste_cost_a, policy_a_name)
res_b = run_scenario(demand, forecast, shelf_b, safety_b, lost_margin_b, waste_cost_b, policy_b_name)

# ── KPI comparison ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Results Comparison")
metrics = ["waste_pct", "stockout_rate", "service_level", "cost"]


def summarise(res):
    tot_orders = res["order"].sum()
    return {
        "waste_pct":     res["waste"].sum() / max(tot_orders, 1),
        "stockout_rate": (res["stockout"] > 0).mean(),
        "service_level": res["sold"].sum() / max(res["demand"].sum(), 1),
        "cost":          res["cost"].sum(),
    }


sa, sb = summarise(res_a), summarise(res_b)

kpi_cols = st.columns(4)
labels = ["Waste %", "Stockout Rate", "Service Level", "Total Cost (relative units)"]
keys   = ["waste_pct", "stockout_rate", "service_level", "cost"]
fmt    = ["{:.1%}", "{:.1%}", "{:.1%}", "{:,.0f}"]

for i, (lbl, key, f) in enumerate(zip(labels, keys, fmt)):
    va = sa[key]; vb = sb[key]
    delta_pct = (vb - va) / max(abs(va), 1e-9) * 100
    better = (delta_pct < 0 if key != "service_level" else delta_pct > 0)
    arrow = ("🟢 ↓" if better else "🔴 ↑") if key != "service_level" else ("🟢 ↑" if better else "🔴 ↓")
    kpi_cols[i].markdown(f"""<div class="kpi-box">
      <div class="kpi-lbl">{lbl}</div>
      <div class="kpi-val">{f.format(va)}</div>
      <div style="font-size:0.75rem; color:#64748b;">A</div>
      <div style="font-size:1.3rem; font-weight:700; color:#a5b4fc;">{f.format(vb)}</div>
      <div style="font-size:0.75rem; color:#64748b;">B &nbsp; {arrow} {abs(delta_pct):.1f}%</div>
    </div>""", unsafe_allow_html=True)

# ── Daily waste/stockout chart ────────────────────────────────────────────────
st.markdown("### 📈 Daily Waste & Stockout — A vs B")
fig = go.Figure()
fig.add_trace(go.Scatter(x=res_a.index, y=res_a["waste"], name="A — Waste",
                          line=dict(color="#f87171", width=1.5)))
fig.add_trace(go.Scatter(x=res_b.index, y=res_b["waste"], name="B — Waste",
                          line=dict(color="#fbbf24", width=1.5, dash="dot")))
fig.add_trace(go.Scatter(x=res_a.index, y=res_a["stockout"], name="A — Stockout",
                          line=dict(color="#818cf8", width=1.5)))
fig.add_trace(go.Scatter(x=res_b.index, y=res_b["stockout"], name="B — Stockout",
                          line=dict(color="#34d399", width=1.5, dash="dot")))
fig.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,20,40,0.6)",
    font=dict(family="Inter"), height=320,
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(orientation="h", y=1.05, x=0),
    hovermode="x unified",
)
st.plotly_chart(fig, width='stretch')

