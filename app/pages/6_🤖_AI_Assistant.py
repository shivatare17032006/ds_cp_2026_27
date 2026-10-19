"""AI Chatbot page — Gemini-powered Q&A grounded in the supply chain data."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import streamlit as st

from app.data_loader import load_all, load_holdout_metrics, load_waste_metrics

st.set_page_config(page_title="AI Assistant · FreshCast", page_icon="🤖", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.chat-bubble-user { background:linear-gradient(135deg,#4f46e5,#7c3aed);
    color:white; border-radius:18px 18px 4px 18px;
    padding:0.8rem 1.1rem; margin:0.4rem 0; max-width:75%; margin-left:auto; }
.chat-bubble-bot { background:rgba(30,41,59,0.9); border:1px solid rgba(99,102,241,0.25);
    color:#e2e8f0; border-radius:18px 18px 18px 4px;
    padding:0.8rem 1.1rem; margin:0.4rem 0; max-width:80%; }
.api-key-box { background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.3);
    border-radius:12px; padding:1rem; margin-bottom:1rem; }
.suggestion-btn { background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.3);
    border-radius:8px; padding:0.4rem 0.8rem; font-size:0.82rem; cursor:pointer;
    color:#a5b4fc; margin:0.2rem; display:inline-block; }
</style>""", unsafe_allow_html=True)

data, forecast_df, stores_df, results = load_all()
holdout = load_holdout_metrics()
waste   = load_waste_metrics()

st.markdown("## 🤖 AI Supply Chain Assistant")
st.caption("Ask anything about your data, forecasts, and waste policies.")

# ── API key ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input("Enter your Gemini API key", type="password",
                             key="gemini_key",
                             help="Get a free key at https://aistudio.google.com/app/apikey")
    if api_key:
        st.success("✅ Key saved for this session")
    else:
        st.info("Enter your Gemini API key to enable AI responses. "
                "Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).")

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    suggestions = [
        "Which store has the highest average daily sales?",
        "What is the best waste policy and why?",
        "Which product category has the most stockout risk?",
        "Summarize the holdout forecast accuracy.",
        "How does newsvendor policy compare to moving average?",
        "What is WAPE and which model scored best?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s[:20]}", width='stretch'):
            st.session_state["chat_input_prefill"] = s


# ── Build grounding context ───────────────────────────────────────────────────
def build_context() -> str:
    lines: list[str] = ["=== FreshCast Supply Chain Data Summary ===\n"]

    # Top-level stats
    lines.append(f"Total records: {len(data):,}")
    lines.append(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    lines.append(f"Stores: {data['store_nbr'].nunique()}")
    lines.append(f"Product families: {data['family'].nunique()}")

    # Top stores
    top_stores = (
        data.groupby("store_nbr")["sales"].mean()
        .nlargest(5).reset_index()
        .rename(columns={"sales": "avg_daily_sales"})
    )
    lines.append("\nTop 5 stores by average daily sales:")
    lines.append(top_stores.to_string(index=False))

    # Top categories
    top_cats = (
        data.groupby("family")["sales"].mean()
        .nlargest(5).reset_index()
        .rename(columns={"sales": "avg_daily_sales"})
    )
    lines.append("\nTop 5 categories by average daily sales:")
    lines.append(top_cats.to_string(index=False))

    if not holdout.empty:
        lines.append("\nHoldout forecast metrics (lower WAPE = better):")
        lines.append(holdout.to_string(index=False))

    if not waste.empty:
        lines.append("\nWaste policy simulation results:")
        lines.append(waste.to_string(index=False))

    if results:
        lines.append(f"\nHoldout period: {results.get('holdout_start','?')} to {results.get('holdout_end','?')}")
        lines.append(f"Horizon: {results.get('horizon_days','?')} days")

    if not stores_df.empty:
        lines.append("\nStore location summary (city counts):")
        city_counts = stores_df["city"].value_counts().head(8).to_string()
        lines.append(city_counts)

    return "\n".join(lines)


# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content":
         "👋 Hi! I'm FreshCast AI. I can answer questions about your store sales data, "
         "forecast accuracy, waste policies, and supply chain insights. What would you like to know?"}
    ]


def get_gemini_response(user_msg: str, api_key: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=(
                "You are FreshCast AI, an expert supply chain analyst assistant. "
                "You answer questions about grocery demand forecasting, waste reduction, "
                "and ordering policies. Be concise, data-driven, and helpful. "
                "Always ground your answers in the data context provided. "
                "Format numbers nicely. Use bullet points when listing multiple items."
            )
        )
        context = build_context()
        prompt = f"Data Context:\n{context}\n\nUser question: {user_msg}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ API Error: {str(e)}\n\nPlease check your API key or try again."


def get_rule_based_response(user_msg: str) -> str:
    """Fallback rule-based Q&A when no API key is provided."""
    msg = user_msg.lower()
    top_store = data.groupby("store_nbr")["sales"].mean().idxmax()
    top_cat   = data.groupby("family")["sales"].mean().idxmax()
    total_sales = data["sales"].sum()

    if any(w in msg for w in ["best store", "top store", "highest store", "most sales"]):
        avg = data.groupby("store_nbr")["sales"].mean()
        return f"🏪 **Store {avg.idxmax()}** has the highest average daily sales of **{avg.max():.1f} units/day**.\n\nTop 5 stores:\n" + \
               "\n".join([f"- Store {s}: {v:.1f} units/day" for s, v in avg.nlargest(5).items()])

    if any(w in msg for w in ["category", "family", "product", "top cat"]):
        avg = data.groupby("family")["sales"].mean()
        return f"📦 **{avg.idxmax()}** is the top category with **{avg.max():.1f} avg daily sales**.\n\nAll categories:\n" + \
               "\n".join([f"- {f}: {v:.1f}" for f, v in avg.sort_values(ascending=False).items()])

    if any(w in msg for w in ["wape", "accuracy", "metric", "rmsse", "bias"]):
        if not holdout.empty:
            best = holdout.loc[holdout["wape"].idxmin()]
            return f"📊 **Best model: {best['model'].replace('_',' ').title()}**\n- WAPE: {best['wape']:.2%}\n- Bias: {best['bias']:.2%}\n- RMSSE: {best['rmsse']:.3f}"
        return "No holdout metrics available. Run `python -m src.report` first."

    if any(w in msg for w in ["waste", "policy", "stockout", "newsvendor"]):
        if not waste.empty:
            best = waste.loc[waste["cost"].idxmin()]
            return f"🏆 **Best policy: {best['policy'].replace('_',' ').title()}**\n- Cost: €{best['cost']:,.0f}\n- Service Level: {best['service_level']:.1%}\n- Stockout Rate: {best['stockout_rate']:.1%}"
        return "No waste metrics found. Run `python -m src.report` first."

    if any(w in msg for w in ["store", "how many", "count"]):
        return f"🏪 There are **{data['store_nbr'].nunique()} stores** across **{stores_df['city'].nunique() if not stores_df.empty else 'multiple'} cities** in Ecuador."

    if any(w in msg for w in ["hello", "hi", "hey"]):
        return "👋 Hello! Ask me about stores, forecasts, categories, waste policies, or metrics."

    return (f"📊 Here's a quick summary of your data:\n"
            f"- **{data['store_nbr'].nunique()} stores** across Ecuador\n"
            f"- **{data['family'].nunique()} product categories**\n"
            f"- **{len(data):,} daily records** from {data['date'].min().date()} to {data['date'].max().date()}\n"
            f"- Top store: Store **{top_store}** | Top category: **{top_cat}**\n\n"
            f"💡 *Add your Gemini API key in the sidebar for full AI responses!*")


# ── Render chat history ───────────────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-bot">{msg["content"]}</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("chat_input_prefill", "")
user_input = st.chat_input("Ask about your data, forecasts, or policies…")

if user_input or prefill:
    question = user_input or prefill
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🤖 Thinking…"):
        if api_key and api_key.strip():
            answer = get_gemini_response(question, api_key.strip())
        else:
            answer = get_rule_based_response(question)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# ── Clear chat ────────────────────────────────────────────────────────────────
if st.sidebar.button("🗑️ Clear Chat", width='stretch'):
    st.session_state.messages = [st.session_state.messages[0]]
    st.rerun()

