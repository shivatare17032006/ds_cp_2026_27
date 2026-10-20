"""Rolling-origin backtest of baselines and ML models on the same windows."""

from __future__ import annotations

import pandas as pd

from src.forecasting import fit_model, predict_demand, prepare_features, trainable_rows
from src.metrics import bias, mean_series_rmsse, wape

KEYS = ["store_nbr", "family"]
BASELINES = ("seasonal_naive", "moving_average")


def add_baselines(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add horizon-safe baseline forecasts to a dense, sorted history.

    Both only use sales at least ``horizon`` days old: the most recent
    same-weekday value (seasonal naive) and a 14-day mean (moving average).
    """
    frame = history.sort_values([*KEYS, "date"]).copy()
    grouped = frame.groupby(KEYS, observed=True)["sales"]
    seasonal_lag = -(-horizon // 7) * 7
    frame["seasonal_naive"] = grouped.shift(seasonal_lag).fillna(0)
    frame["moving_average"] = grouped.transform(
        lambda series: series.shift(horizon).rolling(14, min_periods=1).mean()
    ).fillna(0)
    return frame


def backtest(
    features: pd.DataFrame,
    horizon: int = 14,
    folds: int = 3,
    models: tuple[str, ...] | list[str] = ("lightgbm",),
    seed: int = 42,
) -> pd.DataFrame:
    """Forecast ``folds`` consecutive ``horizon``-day windows at the end of history.

    For each window the ML models are refit on data up to the window's
    origin. Features are lagged by ``horizon``, so every row in the window
    only uses information available at the origin. Returns one row per
    store/family/date in the windows (open days only) with ``sales``, the
    ``fold`` number, its ``origin`` and one prediction column per model.
    """
    if folds < 1:
        raise ValueError("folds must be at least 1")
    frame, feature_columns = prepare_features(features)
    frame["date"] = pd.to_datetime(frame["date"])
    history = add_baselines(frame[frame["target"].notna()], horizon)
    last = history["date"].max()
    windows: list[pd.DataFrame] = []
    for fold in range(folds):
        origin = last - pd.Timedelta(days=horizon * (folds - fold))
        window = history[(history["date"] > origin) & (history["date"] <= origin + pd.Timedelta(days=horizon))]
        if "is_open" in window.columns:
            window = window[window["is_open"] == 1]
        if window.empty:
            raise ValueError(f"Backtest window after {origin.date()} has no open rows.")
        result = window[[*KEYS, "date", "sales", *BASELINES]].copy()
        result["fold"] = fold + 1
        result["origin"] = origin
        train = trainable_rows(history[history["date"] <= origin], feature_columns)
        for model_name in models:
            estimator = fit_model(train, feature_columns, model_name, seed)
            result[model_name] = predict_demand(estimator, window, feature_columns)
        windows.append(result)
    predictions = pd.concat(windows, ignore_index=True)
    predictions["family"] = predictions["family"].astype(str)
    return predictions


def summarize(
    predictions: pd.DataFrame, history: pd.DataFrame, models: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (overall, per-fold, per-family) metrics.

    WAPE and bias pool all rows of the windows; RMSSE is the per-series
    average from ``mean_series_rmsse`` with each fold scaled by the history
    before its origin.
    """
    history = history.assign(date=pd.to_datetime(history["date"]), family=history["family"].astype(str))
    history = history[history["sales"].notna()]
    if "is_open" in history.columns:
        history = history[history["is_open"] == 1]
    per_fold = []
    for (fold, origin), window in predictions.groupby(["fold", "origin"]):
        training = history[history["date"] <= origin]
        for model in models:
            per_fold.append(
                {
                    "fold": fold,
                    "origin": str(pd.Timestamp(origin).date()),
                    "model": model,
                    "wape": wape(window["sales"].to_numpy(), window[model].to_numpy()),
                    "bias": bias(window["sales"].to_numpy(), window[model].to_numpy()),
                    "rmsse": mean_series_rmsse(window, training, model, KEYS),
                }
            )
    folds = pd.DataFrame(per_fold)
    overall = pd.DataFrame(
        [
            {
                "model": model,
                "wape": wape(predictions["sales"].to_numpy(), predictions[model].to_numpy()),
                "bias": bias(predictions["sales"].to_numpy(), predictions[model].to_numpy()),
                "rmsse": float(folds.loc[folds["model"] == model, "rmsse"].mean()),
            }
            for model in models
        ]
    ).sort_values("wape", ignore_index=True)
    by_family = pd.DataFrame(
        [
            {"family": family, **{m: wape(g["sales"].to_numpy(), g[m].to_numpy()) for m in models}}
            for family, g in predictions.groupby("family")
        ]
    )
    return overall, folds, by_family
