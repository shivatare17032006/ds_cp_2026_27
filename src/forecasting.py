"""Baselines and global forecasting models with optional LightGBM/XGBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import bias, interval_coverage, pinball_loss, rmsse, wape


@dataclass
class ForecastResult:
    """Forecast values, intervals and validation metrics."""

    predictions: pd.DataFrame
    metrics: dict[str, float]
    model_name: str


def seasonal_naive(frame: pd.DataFrame, horizon: int = 14, season: int = 7) -> pd.DataFrame:
    """Forecast each entity with its latest available value from one season ago."""
    data = frame.sort_values(["store_nbr", "family", "date"]).copy()
    data["forecast"] = data.groupby(["store_nbr", "family"])["sales"].shift(season)
    data["forecast"] = data["forecast"].fillna(0).clip(lower=0)
    return data.tail(horizon)


def moving_average(frame: pd.DataFrame, window: int = 14, horizon: int = 14) -> pd.DataFrame:
    """Return a direct multi-step moving-average forecast for the final date."""
    data = frame.sort_values(["store_nbr", "family", "date"]).copy()
    rows: list[pd.DataFrame] = []
    for keys, group in data.groupby(["store_nbr", "family"], observed=True):
        value = float(group["sales"].tail(window).mean())
        last = pd.Timestamp(group["date"].max())
        future = pd.DataFrame(
            {
                "store_nbr": keys[0],
                "family": keys[1],
                "date": pd.date_range(last + pd.Timedelta(days=1), periods=horizon),
                "forecast": value,
            }
        )
        rows.append(future)
    return pd.concat(rows, ignore_index=True)


def _make_estimator(model_name: str, seed: int) -> Any:
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression", n_estimators=300, learning_rate=0.05,
            num_leaves=63, random_state=seed, verbosity=-1
        )
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, random_state=seed,
            objective="reg:squarederror", n_jobs=2
        )
    raise ValueError(f"Unsupported model: {model_name}")


def train_global_model(
    features: pd.DataFrame,
    horizon: int = 14,
    model_name: str = "lightgbm",
    seed: int = 42,
) -> tuple[Any, pd.DataFrame, list[str]]:
    """Train a global regressor and produce a future forecast table."""
    excluded = {"date", "sales", "target", "family", "store_id"}
    feature_columns = [c for c in features.columns if c not in excluded]
    train = features.dropna(subset=feature_columns + ["target"]).copy()
    if train.empty:
        raise ValueError("No rows remain after removing unavailable lag features.")
    categorical = [c for c in feature_columns if str(train[c].dtype) in {"object", "category"}]
    for column in categorical:
        train[column] = train[column].astype("category").cat.codes
    estimator = _make_estimator(model_name, seed)
    estimator.fit(train[feature_columns], train["target"])
    latest = features.sort_values(["store_nbr", "family", "date"]).groupby(
        ["store_nbr", "family"], observed=True
    ).tail(1)
    latest = latest.copy()
    for column in categorical:
        latest[column] = latest[column].astype("category").cat.codes
    latest["forecast_log"] = estimator.predict(latest[feature_columns])
    latest["forecast"] = np.expm1(latest["forecast_log"]).clip(0)
    latest["date"] = pd.to_datetime(latest["date"]) + pd.Timedelta(days=1)
    return estimator, latest[["store_nbr", "family", "date", "forecast"]], feature_columns


def evaluate(actual: np.ndarray, prediction: np.ndarray, training: np.ndarray | None = None) -> dict[str, float]:
    """Calculate the portfolio metrics for one validation slice."""
    values = {
        "wape": wape(actual, prediction),
        "bias": bias(actual, prediction),
    }
    if training is not None:
        values["rmsse"] = rmsse(actual, prediction, training)
    return values


def evaluate_intervals(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, median: np.ndarray
) -> dict[str, float]:
    """Calculate coverage and median pinball loss."""
    return {
        "coverage": interval_coverage(actual, lower, upper),
        "pinball_0.5": pinball_loss(actual, median, 0.5),
    }
