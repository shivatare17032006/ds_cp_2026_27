from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metrics import mean_series_rmsse
from src.report import generate_report


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


def test_report_baselines_use_full_history_and_ignore_forecast_rows(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.parquet"
    _weekly_frame().to_parquet(feature_path)
    result = generate_report(feature_path, tmp_path / "out", holdout_days=28, horizon=14)
    metrics = {row["model"]: row for row in result["metrics"]}
    # Sales repeat weekly, so a 14-day seasonal naive is exact from the first
    # holdout day; computing it inside the holdout would zero-fill the start.
    assert metrics["seasonal_naive"]["wape"] == pytest.approx(0.0)
    assert result["holdout_end"] == "2024-04-29"
    assert (tmp_path / "out" / "results.json").exists()


def test_time_split_separates_dates() -> None:
    from src.tuning import time_split

    frame = _weekly_frame(days=40, future_days=0)
    train, valid = time_split(frame, valid_days=14)
    assert train["date"].max() < valid["date"].min()
    assert valid["date"].nunique() == 14


def test_tuning_runs_with_seeded_sampler() -> None:
    pytest.importorskip("optuna")
    pytest.importorskip("lightgbm")
    from tests.test_leakage_and_forecast import _features, _train
    from src.tuning import tune_lightgbm

    features = _features(_train()).to_pandas()
    first = tune_lightgbm(features, trials=2, valid_days=7)
    second = tune_lightgbm(features, trials=2, valid_days=7)
    assert first == second
    assert {"n_estimators", "learning_rate", "num_leaves", "min_child_samples"} == set(first)
