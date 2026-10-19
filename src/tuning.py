"""Optuna time-aware hyperparameter tuning hook."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.forecasting import prepare_features, split_history_future
from src.metrics import wape


def time_split(
    history: pd.DataFrame, valid_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last ``valid_days`` dates; train on every earlier date.

    Splitting on dates (not row positions) keeps one day from straddling both
    sides. Features are already lagged by the forecast horizon, so this
    mirrors a forecast made at the end of the training window.
    """
    dates = pd.to_datetime(history["date"])
    cutoff = dates.max() - pd.Timedelta(days=valid_days - 1)
    return history[dates < cutoff], history[dates >= cutoff]


def tune_lightgbm(
    features: pd.DataFrame,
    seed: int = 42,
    trials: int = 20,
    valid_days: int = 14,
) -> dict[str, Any]:
    """Tune a compact LightGBM search space on a final ``valid_days`` time split."""
    try:
        import optuna
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("Install the modeling extra to use Optuna tuning.") from exc
    frame, feature_columns = prepare_features(features)
    history, _ = split_history_future(frame, feature_columns)
    x_train, x_valid = time_split(history, valid_days)
    if x_train.empty or x_valid.empty:
        raise ValueError("Not enough history for the requested validation window.")
    actual = np.expm1(x_valid["target"].to_numpy())

    def objective(trial: Any) -> float:
        model = LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            random_state=seed,
            verbosity=-1,
        )
        model.fit(x_train[feature_columns], x_train["target"])
        pred = np.expm1(model.predict(x_valid[feature_columns])).clip(0)
        return wape(actual, pred)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials)
    return dict(study.best_params)
