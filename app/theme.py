"""Shared page styling so every dashboard page loads the same base CSS once."""

from __future__ import annotations

import streamlit as st

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,20,40,0.6)",
    font=dict(family="Inter, system-ui, sans-serif"),
)

# Fonts are set on text containers rather than `*`, which would also
# override Streamlit's icon font and turn icons into their ligature names.
_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Inter', system-ui, sans-serif;
}
"""


def apply_theme(extra_css: str = "") -> None:
    """Inject the shared CSS plus any page-specific rules."""
    st.markdown(f"<style>{_BASE_CSS}{extra_css}</style>", unsafe_allow_html=True)
