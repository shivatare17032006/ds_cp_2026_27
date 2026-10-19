"""Forecast metrics used by model validation and the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(actual: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.abs(actual).sum())
    return float(np.abs(actual - prediction).sum() / denominator) if denominator else 0.0


def bias(actual: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.abs(actual).sum())
    return float((prediction - actual).sum() / denominator) if denominator else 0.0


def pinball_loss(actual: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = actual - prediction
    return float(np.maximum(quantile * error, (quantile - 1) * error).mean())


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(((actual >= lower) & (actual <= upper)).mean())


def rmsse(actual: np.ndarray, prediction: np.ndarray, training: np.ndarray, season: int = 7) -> float:
    if len(training) <= season:
        return float(np.sqrt(np.mean((actual - prediction) ** 2)))
    scale = np.mean((training[season:] - training[:-season]) ** 2)
    return float(np.sqrt(np.mean((actual - prediction) ** 2) / scale)) if scale else 0.0


def mean_series_rmsse(
    holdout: pd.DataFrame,
    training: pd.DataFrame,
    prediction: str,
    keys: list[str],
    target: str = "sales",
    season: int = 7,
) -> float:
    """Average RMSSE over series, each scaled by its own training history.

    Series with too little history or a constant seasonal-naive error of zero
    have no defined scale and are skipped (as in the M5 evaluation).
    """
    scores: list[float] = []
    history = {name: group[target].to_numpy() for name, group in training.groupby(keys, observed=True)}
    for name, group in holdout.groupby(keys, observed=True):
        series = history.get(name)
        if series is None or len(series) <= season:
            continue
        scale = float(np.mean((series[season:] - series[:-season]) ** 2))
        if scale == 0:
            continue
        error = group[target].to_numpy() - group[prediction].to_numpy()
        scores.append(float(np.sqrt(np.mean(error**2) / scale)))
    return float(np.mean(scores)) if scores else float("nan")
