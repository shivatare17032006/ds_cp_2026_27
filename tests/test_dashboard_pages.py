"""Smoke tests: every dashboard page renders without errors, with and without data."""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

import streamlit as st
from streamlit.testing.v1 import AppTest

from app import data_loader, insights

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP_DIR / "streamlit_app.py", *sorted((APP_DIR / "pages").glob("*.py"))]
PATH_NAMES = ["FEATURE_PATH", "FORECAST_PATH", "STORES_PATH", "RESULTS_PATH", "HOLDOUT_PATH", "WASTE_PATH"]


def _point_loader_at(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    for name in PATH_NAMES:
        monkeypatch.setattr(data_loader, name, root / getattr(data_loader, name).name)
    st.cache_data.clear()


def _write_artifacts(root: Path) -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=60)
    keys = [(store, family) for store in (1, 2, 3) for family in ("DAIRY", "PRODUCE")]
    history = pd.DataFrame(
        [{"date": d, "store_nbr": s, "family": f, "sales": float(rng.poisson(10 * s))}
         for s, f in keys for d in dates]
    )
    future = history.groupby(["store_nbr", "family"]).tail(1).assign(sales=np.nan)
    pd.concat([history, future]).to_parquet(root / "features.parquet")
    forecast_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=14)
    pd.DataFrame(
        [{"store_nbr": s, "family": f, "date": d, "horizon_day": i + 1, "forecast": 8.0 * s}
         for s, f in keys for i, d in enumerate(forecast_dates)]
    ).to_parquet(root / "forecast.parquet")
    pd.DataFrame(
        {"store_nbr": [1, 2, 3], "city": ["Quito", "Guayaquil", "Cuenca"],
         "state": ["Pichincha", "Guayas", "Azuay"], "type": ["A", "B", "C"], "cluster": [1, 2, 3]}
    ).to_parquet(root / "stores.parquet")
    pd.DataFrame(
        {"model": ["seasonal_naive", "moving_average"], "wape": [0.3, 0.25],
         "bias": [0.01, -0.02], "rmsse": [1.0, 0.9]}
    ).to_csv(root / "holdout_metrics.csv", index=False)
    pd.DataFrame(
        {"policy": ["forecast_plus_10pct", "forecast_only", "newsvendor"], "waste_pct": [0.1, 0.05, 0.08],
         "stockout_rate": [0.1, 0.4, 0.2], "service_level": [0.98, 0.9, 0.95],
         "cost": [100.0, 400.0, 150.0]}
    ).to_csv(root / "waste_policy_metrics.csv", index=False)
    (root / "results.json").write_text(
        json.dumps({"holdout_start": "2024-01-20", "holdout_end": "2024-02-29", "horizon_days": 14}),
        encoding="utf-8",
    )


@pytest.fixture
def demo_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _point_loader_at(monkeypatch, tmp_path)
    yield
    st.cache_data.clear()


@pytest.fixture
def data_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _write_artifacts(tmp_path)
    _point_loader_at(monkeypatch, tmp_path)
    yield
    st.cache_data.clear()


def _run(page: Path) -> AppTest:
    return AppTest.from_file(str(page), default_timeout=120).run()


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem.encode("ascii", "ignore").decode())
def test_page_renders_in_demo_mode(page: Path, demo_mode: None) -> None:
    app = _run(page)
    assert not app.exception, [e.value for e in app.exception]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem.encode("ascii", "ignore").decode())
def test_page_renders_with_pipeline_outputs(page: Path, data_mode: None) -> None:
    app = _run(page)
    assert not app.exception, [e.value for e in app.exception]


def _page(fragment: str) -> Path:
    return next(p for p in PAGES if fragment in p.name)


def test_alerts_page_shows_nonzero_stockout_risk(data_mode: None) -> None:
    app = _run(_page("Alerts"))
    app.multiselect[0].set_value([insights.HIGH_LABEL, insights.MEDIUM_LABEL, insights.LOW_LABEL]).run()
    table = app.dataframe[0].value
    assert len(table) == 6
    # Forecast 8×store vs demand ~10×store: demand usually beats the forecast.
    assert table["Stockout Risk"].max() > 0.5


def test_ai_assistant_answers_sample_question_offline(data_mode: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = _run(_page("AI_Assistant"))
    app.sidebar.button[0].click().run()  # "Which store has the highest average daily sales?"
    messages = [m.markdown[0].value for m in app.chat_message]
    assert "has the highest average daily sales" in messages[-1]
    assert not app.exception


def test_retrain_button_reports_success_and_stays_enabled(
    data_mode: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_pipeline(root: Path) -> Iterator[tuple[str, bool, str]]:
        calls.append(root)
        yield from [("pipeline", True, "ok"), ("train", True, "ok"), ("report", True, "ok")]

    monkeypatch.setattr(insights, "run_pipeline", fake_pipeline)
    app = _run(APP_DIR / "streamlit_app.py")
    button = next(b for b in app.button if "Run Full Pipeline" in b.label)
    button.click().run()
    assert calls, "pipeline was not started"
    assert not app.exception, [e.value for e in app.exception]
    assert "report" in app.session_state["retrain_log"]
    assert not next(b for b in app.button if "Run Full Pipeline" in b.label).disabled


def test_retrain_button_reports_failure(data_mode: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_pipeline(root: Path) -> Iterator[tuple[str, bool, str]]:
        yield "pipeline", False, "Missing configured train file"

    monkeypatch.setattr(insights, "run_pipeline", failing_pipeline)
    app = _run(APP_DIR / "streamlit_app.py")
    next(b for b in app.button if "Run Full Pipeline" in b.label).click().run()
    assert "failed" in app.session_state["retrain_log"]
    assert "Pipeline complete" not in app.session_state["retrain_log"]
