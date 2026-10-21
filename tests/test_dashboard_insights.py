import sys
from pathlib import Path

import pandas as pd
import pytest

from app.insights import (
    HIGH_LABEL,
    LOW_LABEL,
    compute_alerts,
    model_label,
    next_day_forecast,
    rule_based_answer,
    run_pipeline,
    same_weekday_sales,
    stockout_risk,
)


def _history(levels: dict[tuple[int, str], float], days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days)
    rows = [
        {"date": d, "store_nbr": store, "family": family, "sales": level + (i % 3)}
        for (store, family), level in levels.items()
        for i, d in enumerate(dates)
    ]
    return pd.DataFrame(rows)


def _forecast(values: dict[tuple[int, str], float], start: str = "2024-02-10") -> pd.DataFrame:
    dates = pd.date_range(start, periods=3)
    return pd.DataFrame(
        [
            {"store_nbr": s, "family": f, "date": d, "forecast": value + i}
            for (s, f), value in values.items()
            for i, d in enumerate(dates)
        ]
    )


def test_stockout_risk_is_share_of_days_above_order() -> None:
    assert stockout_risk(pd.Series([1, 2, 3, 4]), 2.5) == 0.5
    assert stockout_risk(pd.Series([], dtype=float), 1.0) == 0.0


def test_same_weekday_sales_picks_the_last_eight_matching_days() -> None:
    sales = pd.Series(range(70), index=pd.date_range("2024-01-01", periods=70), dtype=float)
    target = pd.Timestamp("2024-03-11")  # the day after the history, a Monday
    picked = same_weekday_sales(sales, target)
    assert len(picked) == 8
    assert (picked.index.weekday == target.weekday()).all()
    assert picked.index.max() == target - pd.Timedelta(days=7)


def test_accurate_weekday_forecast_is_not_flagged_because_of_weekend_peaks() -> None:
    # Weekdays sell 10, weekends 30. An exact weekday forecast used to be
    # compared with every recent day, so the weekend peaks made it look risky.
    dates = pd.date_range("2024-01-01", periods=56)
    history = pd.DataFrame({"date": dates, "store_nbr": 1, "family": "DAIRY",
                            "sales": [30.0 if d.weekday() >= 5 else 10.0 for d in dates]})
    weekday_forecast = pd.DataFrame({"store_nbr": [1], "family": ["DAIRY"],
                                     "date": [pd.Timestamp("2024-02-26")], "forecast": [10.0]})
    alerts = compute_alerts(history, weekday_forecast)
    assert alerts.loc[0, "Stockout Risk"] == 0.0
    assert alerts.loc[0, "Risk Level"] == LOW_LABEL
    assert stockout_risk(history["sales"].tail(28), 10.0) > 0.25  # what the old metric saw


def test_model_label() -> None:
    assert model_label("lightgbm") == "LightGBM"
    assert model_label("moving_average") == "Moving Average"


def test_next_day_forecast_prefers_model_and_falls_back_to_baseline() -> None:
    history = _history({(1, "DAIRY"): 10, (2, "DAIRY"): 20})
    forecast = _forecast({(1, "DAIRY"): 4.0})
    result = next_day_forecast(history, forecast).set_index(["store_nbr", "family"])
    assert result.loc[(1, "DAIRY"), "forecast"] == 4.0  # first future day, not later ones
    assert result.loc[(1, "DAIRY"), "source"] == "model"
    assert result.loc[(1, "DAIRY"), "date"] == pd.Timestamp("2024-02-10")
    assert result.loc[(2, "DAIRY"), "date"] == history["date"].max() + pd.Timedelta(days=1)
    assert result.loc[(2, "DAIRY"), "source"] == "moving average"
    assert result.loc[(2, "DAIRY"), "forecast"] == pytest.approx(
        history[history.store_nbr == 2]["sales"].tail(14).mean()
    )


def test_alerts_use_the_forecast_so_risk_is_not_always_zero() -> None:
    history = _history({(1, "DAIRY"): 10, (2, "DAIRY"): 10})
    forecast = _forecast({(1, "DAIRY"): 5.0, (2, "DAIRY"): 50.0})
    alerts = compute_alerts(history, forecast).set_index("Store")
    assert alerts.loc[1, "Stockout Risk"] == 1.0
    assert alerts.loc[1, "Risk Level"] == HIGH_LABEL
    assert alerts.loc[2, "Stockout Risk"] == 0.0
    assert alerts.loc[2, "Risk Level"] == LOW_LABEL
    # High-risk rows are listed first.
    assert compute_alerts(history, forecast)["Store"].iloc[0] == 1


def _answer(question: str) -> str:
    history = _history({(1, "DAIRY"): 10, (2, "PRODUCE"): 30, (3, "DAIRY"): 5})
    holdout = pd.DataFrame(
        {"model": ["seasonal_naive", "moving_average"], "wape": [0.3, 0.2],
         "bias": [0.0, 0.01], "rmsse": [1.0, 0.9]}
    )
    waste = pd.DataFrame(
        {"policy": ["forecast_only", "newsvendor"], "waste_pct": [0.1, 0.05],
         "stockout_rate": [0.3, 0.1], "service_level": [0.9, 0.97], "cost": [500.0, 200.0]}
    )
    stores = pd.DataFrame({"store_nbr": [1, 2, 3], "city": ["Quito", "Quito", "Cuenca"]})
    forecast = _forecast({(1, "DAIRY"): 1.0, (2, "PRODUCE"): 100.0, (3, "DAIRY"): 100.0})
    return rule_based_answer(question, history, holdout, waste, stores, compute_alerts(history, forecast))


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which store has the highest average daily sales?", "Store 2** has the highest"),
        ("What is the best waste policy and why?", "Lowest-cost policy: Newsvendor"),
        ("Which product category has the most stockout risk?", "by category"),
        ("Summarize the holdout forecast accuracy.", "Best model: Moving Average"),
        ("How does newsvendor ordering compare to ordering the forecast?", "Lowest-cost policy"),
        ("What is WAPE and which model scored best?", "Best model: Moving Average"),
        ("How many stores are there?", "3 stores** across **2 cities"),
        ("hello", "Hello!"),
    ],
)
def test_sample_questions_get_the_matching_offline_answer(question: str, expected: str) -> None:
    assert expected in _answer(question)


def test_which_is_not_mistaken_for_a_greeting() -> None:
    # "which" contains "hi"; greetings must match whole words only.
    assert "Hello!" not in _answer("Which store sells the most?")


def test_run_pipeline_stops_at_first_failure(tmp_path: Path) -> None:
    steps = [
        ("first", ["-c", "print('one')"]),
        ("second", ["-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"]),
        ("third", ["-c", "print('never')"]),
    ]
    results = list(run_pipeline(tmp_path, steps))
    assert [(name, ok) for name, ok, _ in results] == [("first", True), ("second", False)]
    assert "one" in results[0][2] and "boom" in results[1][2]


def test_run_pipeline_reports_launch_errors(tmp_path: Path) -> None:
    def broken_runner(*args: object, **kwargs: object) -> object:
        raise OSError("no interpreter")

    results = list(run_pipeline(tmp_path, [("only", ["-c", "pass"])], runner=broken_runner))
    assert results == [("only", False, "no interpreter")]
    assert sys.executable  # the default runner launches the current interpreter
