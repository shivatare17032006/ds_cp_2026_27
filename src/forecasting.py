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
    return data.groupby(["store_nbr", "family"], observed=True).tail(horizon)


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
            objective="reg:squarederror", n_jobs=2,
            tree_method="hist", enable_categorical=True,
        )
    raise ValueError(f"Unsupported model: {model_name}")


NON_FEATURE_COLUMNS = {"date", "sales", "target", "store_id"}
WARMUP_PREFIXES = ("sales_", "zero_rate_")


def prepare_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return model-ready features and the feature column list.

    String columns become ``category`` with one category set for the whole
    frame, so training and forecast rows share the same encoding. Tree models
    use these natively.
    """
    frame = features.copy()
    feature_columns = [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]
    for column in feature_columns:
        dtype = frame[column].dtype
        if not isinstance(dtype, pd.CategoricalDtype) and (
            pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
        ):
            frame[column] = frame[column].astype("category")
    return frame, feature_columns


def split_history_future(
    frame: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into trainable history and the forecast rows (null target).

    History rows are dropped only while demand lags are still warming up;
    other missing context values (e.g. store-days without transactions) are
    left for the tree model to handle.
    """
    warmup = [c for c in feature_columns if c.startswith(WARMUP_PREFIXES)]
    history = frame[frame["target"].notna()].dropna(subset=warmup)
    future = frame[frame["target"].isna()]
    return history, future


def train_global_model(
    features: pd.DataFrame,
    horizon: int = 14,
    model_name: str = "lightgbm",
    seed: int = 42,
) -> tuple[Any, pd.DataFrame, list[str]]:
    """Train a global regressor and forecast the future rows of the feature table.

    ``features`` must include the forecast dates appended by
    ``build_features(..., forecast_rows=True)``. The forecast has one row per
    entity and future date, with ``horizon_day`` 1..``horizon``.
    """
    frame, feature_columns = prepare_features(features)
    frame["date"] = pd.to_datetime(frame["date"])
    history, future = split_history_future(frame, feature_columns)
    if history.empty:
        raise ValueError("No rows remain after removing unavailable lag features.")
    if future.empty:
        raise ValueError("No forecast rows found; build features with forecast_rows=True.")
    estimator = _make_estimator(model_name, seed)
    estimator.fit(history[feature_columns], history["target"])
    forecast = future[["store_nbr", "family", "date"]].copy()
    forecast["family"] = forecast["family"].astype(str)
    forecast["horizon_day"] = (forecast["date"] - history["date"].max()).dt.days
    forecast["forecast"] = np.expm1(estimator.predict(future[feature_columns])).clip(0)
    forecast = forecast[forecast["horizon_day"].between(1, horizon)]
    return estimator, forecast.reset_index(drop=True), feature_columns


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
