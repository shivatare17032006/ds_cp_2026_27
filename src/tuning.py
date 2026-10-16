"""Optuna time-aware hyperparameter tuning hook."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.metrics import wape


def tune_lightgbm(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target",
    seed: int = 42,
    trials: int = 20,
) -> dict[str, Any]:
    """Tune a compact LightGBM search space using an expanding time split."""
    try:
        import optuna
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("Install the modeling extra to use Optuna tuning.") from exc
    ordered = train.sort_values("date").dropna(subset=feature_columns + [target_column])
    split = max(int(len(ordered) * 0.8), 1)
    x_train, x_valid = ordered.iloc[:split], ordered.iloc[split:]

    def objective(trial: Any) -> float:
        model = LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            random_state=seed,
            verbosity=-1,
        )
        model.fit(x_train[feature_columns], x_train[target_column])
        pred = np.expm1(model.predict(x_valid[feature_columns])).clip(0)
        actual = np.expm1(x_valid[target_column].to_numpy())
        return wape(actual, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials, seed=seed)
    return dict(study.best_params)
