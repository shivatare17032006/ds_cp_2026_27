"""Generate final holdout metrics and waste-policy results from processed data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.metrics import bias, mean_series_rmsse, wape
from src.waste import compare_policies


def generate_report(
    feature_path: str | Path = "data/processed/features.parquet",
    output_dir: str | Path = "reports/generated",
    holdout_days: int = 42,
    horizon: int = 14,
) -> dict[str, object]:
    """Evaluate seasonal-naive and moving-average forecasts on the final holdout.

    Both baselines are computed on the full history and only use sales at
    least ``horizon`` days old, matching what a direct ``horizon``-day
    forecast could know. Forecast rows (null sales) are ignored.
    """
    keys = ["store_nbr", "family"]
    frame = pd.read_parquet(feature_path)
    frame = frame[frame["sales"].notna()].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values([*keys, "date"])
    grouped = frame.groupby(keys, observed=True)["sales"]
    # The most recent same-weekday value that is at least `horizon` days old.
    seasonal_lag = -(-horizon // 7) * 7
    frame["seasonal_naive"] = grouped.shift(seasonal_lag)
    frame["moving_average"] = grouped.transform(
        lambda series: series.shift(horizon).rolling(14, min_periods=1).mean()
    )
    cutoff = frame["date"].max() - pd.Timedelta(days=holdout_days - 1)
    training = frame[frame["date"] < cutoff]
    holdout = frame[frame["date"] >= cutoff].copy()
    rows: list[dict[str, float | str]] = []
    for model in ("seasonal_naive", "moving_average"):
        holdout[model] = holdout[model].fillna(0)
        actual = holdout["sales"].to_numpy()
        prediction = holdout[model].to_numpy()
        rows.append(
            {
                "model": model,
                "wape": wape(actual, prediction),
                "bias": bias(actual, prediction),
                "rmsse": mean_series_rmsse(holdout, training, model, keys),
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
        "horizon_days": horizon,
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
