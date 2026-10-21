import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest, summarize
from src.metrics import mean_series_rmsse


def _weekly_frame(days: int = 120, future_days: int = 14) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=days + future_days)
    for store, level in ((1, 10.0), (2, 50.0)):
        for i, current in enumerate(dates):
            sales = level + 4 * current.weekday() if i < days else np.nan
            rows.append({"date": current, "store_nbr": store, "family": "DAIRY", "sales": sales})
    return pd.DataFrame(rows)


def test_mean_series_rmsse_scales_each_series_by_its_own_history() -> None:
    training = pd.DataFrame(
        {"id": ["a"] * 8 + ["b"] * 8, "sales": [0, 1] * 4 + [0, 10] * 4}
    )
    holdout = pd.DataFrame({"id": ["a", "b"], "sales": [1.0, 10.0], "pred": [0.0, 0.0]})
    a_scale = np.mean((np.array([0, 1] * 4)[7:] - np.array([0, 1] * 4)[:-7]) ** 2)
    b_scale = np.mean((np.array([0, 10] * 4)[7:] - np.array([0, 10] * 4)[:-7]) ** 2)
    expected = np.mean([np.sqrt(1 / a_scale), np.sqrt(100 / b_scale)])
    assert mean_series_rmsse(holdout, training, "pred", ["id"]) == pytest.approx(expected)


def test_backtest_baselines_use_full_history_and_ignore_forecast_rows() -> None:
    frame = _weekly_frame().assign(target=lambda f: np.log1p(f["sales"]))
    predictions = backtest(frame, horizon=14, folds=2, models=())
    overall, _, _ = summarize(predictions, frame, ["seasonal_naive", "moving_average"])
    # Sales repeat weekly, so a 14-day seasonal naive is exact from the first
    # window day; computing it inside the window would zero-fill the start.
    assert overall.set_index("model").loc["seasonal_naive", "wape"] == pytest.approx(0.0)
    assert predictions["date"].max() == pd.Timestamp("2024-04-29")  # forecast rows ignored
    assert predictions["date"].nunique() == 28


def test_time_split_separates_dates() -> None:
    from src.tuning import time_split

    frame = _weekly_frame(days=40, future_days=0)
    train, valid = time_split(frame, valid_days=14)
    assert train["date"].max() < valid["date"].min()
    assert valid["date"].nunique() == 14


def test_tuning_runs_with_seeded_sampler() -> None:
    pytest.importorskip("optuna")
    pytest.importorskip("lightgbm")
    from src.tuning import tune_lightgbm
    from tests.test_leakage_and_forecast import _features, _train

    features = _features(_train()).to_pandas()
    first = tune_lightgbm(features, trials=2, valid_days=7)
    second = tune_lightgbm(features, trials=2, valid_days=7)
    assert first == second
    assert {"tweedie_variance_power", "n_estimators", "learning_rate", "num_leaves", "min_child_samples"} == set(first)
