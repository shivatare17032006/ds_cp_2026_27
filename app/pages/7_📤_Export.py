"""Export & Reports page — download CSV/Excel and view raw data."""

from __future__ import annotations
import sys
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.data_loader import load_all, load_holdout_metrics, load_waste_metrics

st.set_page_config(page_title="Export · FreshCast", page_icon="📤", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.dl-card { background:rgba(30,41,59,0.8); border:1px solid rgba(99,102,241,0.25);
           border-radius:14px; padding:1.2rem; margin-bottom:0.8rem; }
</style>""", unsafe_allow_html=True)

data, forecast_df, stores_df, results = load_all()
holdout = load_holdout_metrics()
waste   = load_waste_metrics()

st.markdown("## 📤 Export & Reports")
st.caption("Download forecasts, metrics, and data exports in CSV or Excel format.")

# ── Download cards ────────────────────────────────────────────────────────────
st.markdown("### 📥 Download Files")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="dl-card">', unsafe_allow_html=True)
    st.markdown("**📊 Full Forecast**")
    st.caption("All store+category forecasts")
    if not forecast_df.empty:
        csv = forecast_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv, "forecast_all.csv", "text/csv", width='stretch')

        buf = io.BytesIO()
        forecast_df.to_excel(buf, index=False, engine="openpyxl")
        st.download_button("⬇️ Download Excel", buf.getvalue(),
                           "forecast_all.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width='stretch')
    else:
        st.warning("No forecast. Run `python -m src.train`")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="dl-card">', unsafe_allow_html=True)
    st.markdown("**📋 Holdout Metrics**")
    st.caption("WAPE, Bias, RMSSE per model")
    if not holdout.empty:
        csv2 = holdout.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv2, "holdout_metrics.csv", "text/csv", width='stretch')
    else:
        st.warning("No metrics. Run `python -m src.report`")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="dl-card">', unsafe_allow_html=True)
    st.markdown("**🗑️ Waste Policy Results**")
    st.caption("Cost, service level, stockout")
    if not waste.empty:
        csv3 = waste.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv3, "waste_policy.csv", "text/csv", width='stretch')
    else:
        st.warning("No waste data. Run `python -m src.report`")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ── Raw data explorer ─────────────────────────────────────────────────────────
st.markdown("### 🔍 Raw Data Explorer")
tab1, tab2, tab3, tab4 = st.tabs(["📅 Sales History", "🔮 Forecast", "🏪 Stores", "📊 Metrics"])

with tab1:
    families = sorted(data["family"].dropna().astype(str).unique().tolist())
    stores   = sorted(data["store_nbr"].dropna().unique().tolist())
    c1, c2 = st.columns(2)
    sel_fam   = c1.multiselect("Filter Category", families, default=families[:2], key="exp_fam")
    sel_store = c2.multiselect("Filter Store", stores, default=stores[:3], key="exp_str")
    filt = data.copy()
    if sel_fam:
        filt = filt[filt["family"].astype(str).isin(sel_fam)]
    if sel_store:
        filt = filt[filt["store_nbr"].isin(sel_store)]
    st.dataframe(filt.tail(500).sort_values("date", ascending=False), width='stretch', hide_index=True)
    st.caption(f"Showing last 500 of {len(filt):,} filtered rows")

with tab2:
    if not forecast_df.empty:
        st.dataframe(forecast_df.sort_values(["store_nbr","family","date"]), width='stretch', hide_index=True)
    else:
        st.info("No forecast data found.")

with tab3:
    if not stores_df.empty:
        st.dataframe(stores_df, width='stretch', hide_index=True)
    else:
        st.info("No store data found.")

with tab4:
    if not holdout.empty:
        st.markdown("**Holdout Metrics**")
        st.dataframe(holdout.style.format({"wape":"{:.2%}","bias":"{:.2%}","rmsse":"{:.3f}"}),
                     width='stretch', hide_index=True)
    if not waste.empty:
        st.markdown("**Waste Policies**")
        st.dataframe(waste.style.format({"waste_pct":"{:.1%}","stockout_rate":"{:.1%}",
                                         "service_level":"{:.1%}","cost":"€{:,.0f}"}),
                     width='stretch', hide_index=True)

