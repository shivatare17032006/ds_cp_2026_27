"""Shared data loader — cached once, imported by every page."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.insights import compute_alerts

FEATURE_PATH = ROOT / "data" / "processed" / "features.parquet"
FORECAST_PATH = ROOT / "data" / "artifacts" / "forecast.parquet"
STORES_PATH = ROOT / "data" / "interim" / "stores.parquet"
RESULTS_PATH = ROOT / "reports" / "generated" / "results.json"
HOLDOUT_PATH = ROOT / "reports" / "generated" / "holdout_metrics.csv"
WASTE_PATH = ROOT / "reports" / "generated" / "waste_policy_metrics.csv"


@st.cache_data(show_spinner=False)
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return (history, forecast, stores, results_json)."""
    forecast_df = pd.DataFrame(columns=["store_nbr", "family", "date", "forecast"])
    stores_df = pd.DataFrame()
    results: dict = {}

    if FEATURE_PATH.exists():
        data = pd.read_parquet(FEATURE_PATH, columns=["date", "store_nbr", "family", "sales"])
        data = data[data["sales"].notna()].copy()
        data["date"] = pd.to_datetime(data["date"])
    else:
        # Demo fallback
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        data = pd.DataFrame({
            "date": list(dates) * 2,
            "store_nbr": [1] * 120 + [2] * 120,
            "family": ["PRODUCE"] * 240,
            "sales": [10 + (i % 7) * 2 for i in range(120)] * 2,
        })

    if FORECAST_PATH.exists():
        forecast_df = pd.read_parquet(FORECAST_PATH)
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    if STORES_PATH.exists():
        stores_df = pd.read_parquet(STORES_PATH)

    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    return data, forecast_df, stores_df, results


@st.cache_data(show_spinner=False)
def load_alerts() -> pd.DataFrame:
    """Per store/category risk table (see ``app.insights.compute_alerts``)."""
    history, forecast, _, _ = load_all()
    return compute_alerts(history, forecast)


@st.cache_data(show_spinner=False)
def load_report_table(name: str) -> pd.DataFrame:
    """A CSV written by ``src.report`` (e.g. ``family_metrics``), or empty."""
    path = HOLDOUT_PATH.parent / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_holdout_metrics() -> pd.DataFrame:
    if HOLDOUT_PATH.exists():
        return pd.read_csv(HOLDOUT_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_waste_metrics() -> pd.DataFrame:
    if WASTE_PATH.exists():
        return pd.read_csv(WASTE_PATH)
    return pd.DataFrame()


# Ecuador city → (lat, lon) lookup
ECUADOR_COORDS: dict[str, tuple[float, float]] = {
    "Quito": (-0.2295, -78.5243),
    "Guayaquil": (-2.1962, -79.8862),
    "Cuenca": (-2.8974, -79.0045),
    "Santo Domingo": (-0.2543, -79.1719),
    "Machala": (-3.2581, -79.9554),
    "Manta": (-0.9677, -80.7089),
    "Esmeraldas": (0.9592, -79.6536),
    "Ambato": (-1.2491, -78.6167),
    "Riobamba": (-1.6635, -78.6536),
    "Ibarra": (0.3517, -78.1222),
    "Loja": (-3.9931, -79.2042),
    "Latacunga": (-0.9333, -78.6167),
    "Cayambe": (0.0408, -78.1411),
    "Salinas": (-2.2226, -80.9584),
    "Daule": (-1.8650, -79.9778),
    "Babahoyo": (-1.8013, -79.5342),
    "Quevedo": (-1.0225, -79.4626),
    "Playas": (-2.6333, -80.3833),
    "Libertad": (-2.2333, -80.9000),
    "Guaranda": (-1.5936, -78.9997),
    "Puyo": (-1.4924, -77.9981),
    "El Carmen": (-0.2667, -79.4500),
}

