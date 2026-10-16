"""Forecast metrics used by model validation and the dashboard."""

from __future__ import annotations

import numpy as np


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
