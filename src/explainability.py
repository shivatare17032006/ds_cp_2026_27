"""Optional SHAP explainability helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def global_shap(model: Any, data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return mean absolute SHAP importance, or a clear dependency error."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install the modeling extra to generate SHAP explanations.") from exc
    values = shap.TreeExplainer(model)(data[feature_columns]).values
    return pd.DataFrame(
        {"feature": feature_columns, "mean_abs_shap": abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
