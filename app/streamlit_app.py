"""FreshCast — Home dashboard with live KPIs and pipeline retraining."""

from __future__ import annotations
import sys
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data_loader import load_all, load_holdout_metrics, load_waste_metrics

st.set_page_config(
    page_title="FreshCast Control Tower",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0b3d2e 0%, #0f5a42 40%, #1a7a5e 100%);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top:-50%; right:-10%;
    width:400px; height:400px; border-radius:50%;
    background: radial-gradient(circle, rgba(74,222,128,0.08), transparent 60%);
}
.hero h1 { margin:0; font-size:2.8rem; font-weight:800;
           background: linear-gradient(90deg,#4ade80,#34d399,#a7f3d0);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p  { color:#a7f3d0; margin:0.5rem 0 0; font-size:1.05rem; }

/* KPI cards */
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1.2rem 0; }
.kpi { background:linear-gradient(135deg,rgba(15,20,40,0.9),rgba(20,30,60,0.9));
       border:1px solid rgba(255,255,255,0.07); border-radius:16px;
       padding:1.2rem 1.5rem; position:relative; overflow:hidden; }
.kpi::after { content:''; position:absolute; bottom:0; left:0; right:0; height:3px;
              background:var(--accent,#4ade80); border-radius:0 0 16px 16px; }
.kpi-val { font-size:2.2rem; font-weight:700; color:var(--accent,#4ade80); }
.kpi-lbl { font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; margin-top:0.3rem; }
.kpi-sub { font-size:0.82rem; color:#64748b; margin-top:0.2rem; }

/* Sidebar nav style */
section[data-testid="stSidebar"] { background: #0a0f1e; }

/* Retrain button */
.stButton > button {
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    color: white; border:none; border-radius:10px;
    font-weight:600; padding:0.6rem 1.2rem;
    transition: all 0.2s ease;
}
.stButton > button:hover { opacity:0.9; transform:translateY(-1px); }
</style>
""", unsafe_allow_html=True)

data, forecast_df, stores_df, results = load_all()
holdout = load_holdout_metrics()
waste   = load_waste_metrics()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🥬 FreshCast Control Tower</h1>
  <p>Predictive supply chain · Demand forecasting · Food waste reduction · AI-powered insights</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
total_stores   = data["store_nbr"].nunique()
total_families = data["family"].nunique()
total_records  = len(data)
date_range     = f"{data['date'].min().date()} → {data['date'].max().date()}"
avg_daily_all  = data["sales"].mean()

best_wape = holdout["wape"].min() if not holdout.empty else None
best_cost = waste["cost"].min() if not waste.empty else None

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"""<div class="kpi" style="--accent:#4ade80">
  <div class="kpi-val">{total_stores}</div>
  <div class="kpi-lbl">Stores Monitored</div>
  <div class="kpi-sub">Ecuador · Favorita</div>
</div>""", unsafe_allow_html=True)

c2.markdown(f"""<div class="kpi" style="--accent:#818cf8">
  <div class="kpi-val">{total_families}</div>
  <div class="kpi-lbl">Product Families</div>
  <div class="kpi-sub">Fresh + Perishable</div>
</div>""", unsafe_allow_html=True)

c3.markdown(f"""<div class="kpi" style="--accent:#f59e0b">
  <div class="kpi-val">{best_wape:.1%}</div>
  <div class="kpi-lbl">Best WAPE</div>
  <div class="kpi-sub">Holdout evaluation</div>
</div>""" if best_wape is not None else f"""<div class="kpi" style="--accent:#f59e0b">
  <div class="kpi-val">{avg_daily_all:,.0f}</div>
  <div class="kpi-lbl">Avg Daily Sales</div>
  <div class="kpi-sub">All stores</div>
</div>""", unsafe_allow_html=True)

c4.markdown(f"""<div class="kpi" style="--accent:#34d399">
  <div class="kpi-val">{total_records/1e6:.2f}M</div>
  <div class="kpi-lbl">Data Records</div>
  <div class="kpi-sub">{date_range}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Total Daily Sales (All Stores)")
    daily = data.groupby("date")["sales"].sum().reset_index()
    daily["ma_14"] = daily["sales"].rolling(14).mean()
    fig = px.line(daily, x="date", y=["sales", "ma_14"],
                  color_discrete_map={"sales": "rgba(74,222,128,0.4)", "ma_14": "#4ade80"},
                  labels={"value":"Units","date":"Date","variable":"Series"})
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(15,20,40,0.6)", height=300,
                      margin=dict(l=0,r=0,t=20,b=0), legend=dict(orientation="h",y=1.05),
                      font=dict(family="Inter"))
    st.plotly_chart(fig, width='stretch')

with col2:
    st.markdown("### 🥧 Sales by Category")
    cat_sales = data.groupby("family")["sales"].sum().sort_values(ascending=False).reset_index()
    fig2 = px.pie(cat_sales, names="family", values="sales",
                  color_discrete_sequence=px.colors.qualitative.Vivid, hole=0.4)
    fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       height=300, margin=dict(l=0,r=0,t=20,b=0),
                       font=dict(family="Inter"), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig2, width='stretch')

# ── Results summary ───────────────────────────────────────────────────────────
if results:
    st.markdown("---")
    st.markdown("### 📋 Latest Report Results")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Model Accuracy**")
        if not holdout.empty:
            st.dataframe(holdout.style.format({"wape":"{:.2%}","bias":"{:.2%}","rmsse":"{:.3f}"}),
                         width='stretch', hide_index=True)

    with col4:
        st.markdown("**Waste Policy Simulation**")
        if not waste.empty:
            st.dataframe(
                waste.style.format({"waste_pct":"{:.1%}","stockout_rate":"{:.1%}",
                                    "service_level":"{:.1%}","cost":"€{:,.0f}"}),
                width='stretch', hide_index=True)

# ── Live retraining ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔁 Retrain Pipeline")
st.caption("Regenerate features, retrain model, and refresh the report in one click.")

if "retrain_log" not in st.session_state:
    st.session_state.retrain_log = ""
if "retrain_running" not in st.session_state:
    st.session_state.retrain_running = False

if st.button("🚀 Run Full Pipeline (pipeline → train → report)", disabled=st.session_state.retrain_running):
    st.session_state.retrain_running = True
    st.session_state.retrain_log = "⏳ Starting pipeline…\n"

    def _retrain():
        log = ""
        for step, cmd in [
            ("pipeline", [sys.executable, "-m", "src.pipeline", "--config", "configs/base.yaml"]),
            ("train",    [sys.executable, "-m", "src.train"]),
            ("report",   [sys.executable, "-m", "src.report"]),
        ]:
            log += f"\n▶️ Running {step}…\n"
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
                log += r.stdout or ""
                if r.returncode != 0:
                    log += f"❌ {step} failed:\n{r.stderr}"
                    break
                else:
                    log += f"✅ {step} done.\n"
            except Exception as e:
                log += f"❌ Error: {e}\n"
                break
        log += "\n🎉 Pipeline complete! Refresh the page to see updated results."
        st.session_state.retrain_log = log
        st.session_state.retrain_running = False

    t = threading.Thread(target=_retrain, daemon=True)
    t.start()
    st.rerun()

if st.session_state.retrain_running:
    st.spinner("Running pipeline…")

if st.session_state.retrain_log:
    st.code(st.session_state.retrain_log, language="bash")
    if not st.session_state.retrain_running:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

st.markdown("---")
st.caption("FreshCast · Predictive Supply Chain & Food Waste Reduction · Kaggle Store Sales Dataset")

