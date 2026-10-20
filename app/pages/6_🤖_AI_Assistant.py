"""AI Chatbot page — Gemini-powered Q&A grounded in the supply chain data."""

from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.data_loader import load_alerts, load_all, load_holdout_metrics, load_waste_metrics
from app.insights import rule_based_answer
from app.theme import apply_theme

DEFAULT_MODEL = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = (
    "You are FreshCast AI, an expert supply chain analyst assistant. "
    "You answer questions about grocery demand forecasting, waste reduction, "
    "and ordering policies. Be concise, data-driven, and helpful. "
    "Ground every answer in the data context provided and say so when the context "
    "does not contain the answer. Costs are in relative units, not a currency. "
    "Format numbers nicely. Use bullet points when listing multiple items."
)

st.set_page_config(page_title="AI Assistant · FreshCast", page_icon="🤖", layout="wide")
apply_theme()

data, forecast_df, stores_df, results = load_all()
holdout = load_holdout_metrics()
waste   = load_waste_metrics()
alerts  = load_alerts()

st.markdown("## 🤖 AI Supply Chain Assistant")
st.caption("Ask anything about your data, forecasts, and waste policies.")

# ── API key ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input("Enter your Gemini API key", type="password",
                             key="gemini_key", value=os.environ.get("GEMINI_API_KEY", ""),
                             help="Get a free key at https://aistudio.google.com/app/apikey. "
                                  "You can also set the GEMINI_API_KEY environment variable.")
    model_name = st.text_input("Gemini model", value=DEFAULT_MODEL, key="gemini_model")
    if api_key:
        st.success("✅ Key set for this session")
    else:
        st.info("Without a key the assistant answers common questions offline. "
                "Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).")

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    suggestions = [
        "Which store has the highest average daily sales?",
        "What is the best waste policy and why?",
        "Which product category has the most stockout risk?",
        "Summarize the holdout forecast accuracy.",
        "How does newsvendor ordering compare to ordering the forecast?",
        "What is WAPE and which model scored best?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s[:20]}", width='stretch'):
            st.session_state["chat_input_prefill"] = s


# ── Build grounding context ───────────────────────────────────────────────────
def build_context() -> str:
    lines: list[str] = ["=== FreshCast Supply Chain Data Summary ===\n"]

    lines.append(f"Total records: {len(data):,}")
    lines.append(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    lines.append(f"Stores: {data['store_nbr'].nunique()}")
    lines.append(f"Product families: {data['family'].nunique()}")

    top_stores = (
        data.groupby("store_nbr")["sales"].mean()
        .nlargest(5).reset_index()
        .rename(columns={"sales": "avg_daily_sales"})
    )
    lines.append("\nTop 5 stores by average daily sales:")
    lines.append(top_stores.to_string(index=False))

    top_cats = (
        data.groupby("family")["sales"].mean()
        .nlargest(5).reset_index()
        .rename(columns={"sales": "avg_daily_sales"})
    )
    lines.append("\nTop 5 categories by average daily sales:")
    lines.append(top_cats.to_string(index=False))

    if not alerts.empty:
        lines.append("\nHighest stockout risk (share of last 28 days demand exceeded the next-day forecast):")
        lines.append(alerts.head(10)[["Store", "Category", "Stockout Risk", "Risk Level"]].to_string(index=False))

    if not holdout.empty:
        lines.append("\nHoldout forecast metrics (lower WAPE = better):")
        lines.append(holdout.to_string(index=False))

    if not waste.empty:
        lines.append("\nWaste policy simulation results (cost in relative units):")
        lines.append(waste.to_string(index=False))

    if results:
        lines.append(f"\nHoldout period: {results.get('holdout_start','?')} to {results.get('holdout_end','?')}")
        lines.append(f"Horizon: {results.get('horizon_days','?')} days")

    if not stores_df.empty:
        lines.append("\nStore location summary (city counts):")
        lines.append(stores_df["city"].value_counts().head(8).to_string())

    return "\n".join(lines)


# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content":
         "👋 Hi! I'm FreshCast AI. I can answer questions about your store sales data, "
         "forecast accuracy, waste policies, and supply chain insights. What would you like to know?"}
    ]


def get_gemini_response(user_msg: str, api_key: str, model: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return ("⚠️ The Gemini SDK is not installed. Run `pip install -e \".[ai]\"` "
                "(installs `google-genai`) and reload the page.")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=f"Data Context:\n{build_context()}\n\nUser question: {user_msg}",
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
        )
        return response.text or "⚠️ The model returned an empty response."
    except Exception as e:  # surface API/key/model errors in the chat
        return f"⚠️ API Error: {e}\n\nPlease check your API key and model name, or try again."


# ── Render chat history ───────────────────────────────────────────────────────
# st.chat_message renders Markdown only, so neither user input nor model output
# can inject HTML into the page.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("chat_input_prefill", "")
user_input = st.chat_input("Ask about your data, forecasts, or policies…")

if user_input or prefill:
    question = user_input or prefill
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🤖 Thinking…"):
        if api_key and api_key.strip():
            answer = get_gemini_response(question, api_key.strip(), model_name.strip() or DEFAULT_MODEL)
        else:
            answer = rule_based_answer(question, data, holdout, waste, stores_df, alerts)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# ── Clear chat ────────────────────────────────────────────────────────────────
if st.sidebar.button("🗑️ Clear Chat", width='stretch'):
    st.session_state.messages = [st.session_state.messages[0]]
    st.rerun()
