"""Generate final holdout metrics and waste-policy results from processed data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.metrics import bias, rmsse, wape
from src.waste import compare_policies


def generate_report(
    feature_path: str | Path = "data/processed/features.parquet",
    output_dir: str | Path = "reports/generated",
    holdout_days: int = 42,
) -> dict[str, object]:
    """Evaluate seasonal-naive and moving-average forecasts on the final holdout."""
    frame = pd.read_parquet(feature_path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["store_nbr", "family", "date"])
    cutoff = frame["date"].max() - pd.Timedelta(days=holdout_days - 1)
    holdout = frame[frame["date"] >= cutoff].copy()
    holdout["seasonal_naive"] = holdout.groupby(["store_nbr", "family"], observed=True)[
        "sales"
    ].shift(7).fillna(0)
    holdout["moving_average"] = holdout.groupby(["store_nbr", "family"], observed=True)[
        "sales"
    ].transform(lambda series: series.shift(7).rolling(14, min_periods=1).mean()).fillna(0)
    rows: list[dict[str, float | str]] = []
    for model in ("seasonal_naive", "moving_average"):
        actual = holdout["sales"].to_numpy()
        prediction = holdout[model].to_numpy()
        rows.append(
            {
                "model": model,
                "wape": wape(actual, prediction),
                "bias": bias(actual, prediction),
                "rmsse": rmsse(actual, prediction, frame["sales"].to_numpy()),
            }
        )
    representative = holdout.groupby("date", observed=True)["sales"].sum()
    representative_forecast = holdout.groupby("date", observed=True)["moving_average"].sum()
    policies = compare_policies(representative, representative_forecast, shelf_life_days=7)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "holdout_metrics.csv", index=False)
    policies.to_csv(output / "waste_policy_metrics.csv", index=False)
    result: dict[str, object] = {
        "dataset": "Kaggle Store Sales - Time Series Forecasting",
        "holdout_days": holdout_days,
        "holdout_start": str(cutoff.date()),
        "holdout_end": str(frame["date"].max().date()),
        "metrics": rows,
        "waste_policies": policies.to_dict(orient="records"),
        "caveat": "Waste is simulated from sales and assumptions; it is not observed waste.",
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    report = generate_report()
    print(json.dumps(report, indent=2))
