"""Generate backtest metrics and waste-policy results from processed data."""

from __future__ import annotations

import argparse
import importlib.util
import json

import pandas as pd

from src.backtest import BASELINES, backtest, summarize
from src.config import DEFAULT_CONFIG, PipelineConfig, load_config
from src.waste import simulate_portfolio

HEADLINE_POLICY = "newsvendor"
NEGLIGIBLE_WASTE_PCT = 0.005


def available_models(models: list[str]) -> list[str]:
    """Configured ML models whose libraries are installed."""
    return [m for m in models if importlib.util.find_spec(m) is not None]


def headline(waste: pd.DataFrame, model: str, baseline: str = "moving_average") -> dict[str, object]:
    """Compare model-driven and baseline-driven ordering under the same policy."""
    rows = waste[waste["policy"] == HEADLINE_POLICY].set_index("forecast")
    if model == baseline or model not in rows.index:
        return {"available": False, "reason": "No ML model was evaluated."}
    ours, base = rows.loc[model], rows.loc[baseline]
    negligible = base["waste_pct"] < NEGLIGIBLE_WASTE_PCT
    result: dict[str, object] = {
        "available": True,
        "policy": HEADLINE_POLICY,
        "forecast": model,
        "baseline_forecast": baseline,
        "units_wasted": float(ours["units_wasted"]),
        "baseline_units_wasted": float(base["units_wasted"]),
        "waste_pct": float(ours["waste_pct"]),
        "baseline_waste_pct": float(base["waste_pct"]),
        # A relative change between two near-zero amounts is noise, not a result.
        "waste_reduction_pct": None if negligible else float(1 - ours["units_wasted"] / base["units_wasted"]),
        "service_level": float(ours["service_level"]),
        "baseline_service_level": float(base["service_level"]),
        "cost": float(ours["cost"]),
        "baseline_cost": float(base["cost"]),
        "cost_reduction_pct": float(1 - ours["cost"] / base["cost"]) if base["cost"] else None,
    }
    if negligible:
        result["note"] = (
            f"Simulated waste is below {NEGLIGIBLE_WASTE_PCT:.1%} of units ordered for both "
            "forecasts: with daily ordering and same-day delivery, leftovers sell before they "
            "expire. Compare cost and service level instead."
        )
    return result


def generate_report(config: PipelineConfig) -> dict[str, object]:
    """Backtest baselines and ML models, simulate waste per store/family, write results.

    Outputs in ``config.reports_dir``: ``holdout_metrics.csv`` (overall,
    one row per model), ``backtest_metrics.csv`` (per fold),
    ``family_metrics.csv`` (WAPE per family), ``waste_comparison.csv``
    (every forecast x policy), ``waste_policy_metrics.csv`` (policies for
    the best forecast) and ``results.json``.
    """
    feature_path = config.processed_dir / "features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing features: {feature_path}. Run src.pipeline first.")
    features = pd.read_parquet(feature_path)
    models = available_models(config.models)
    predictions = backtest(features, config.horizon, config.backtest_folds, models, config.seed)
    evaluated = [*BASELINES, *models]
    overall, per_fold, by_family = summarize(predictions, features, evaluated)

    best_model = min(models, key=lambda m: float(overall.set_index("model").loc[m, "wape"])) if models else None
    forecasts = ["moving_average", *([best_model] if best_model else [])]
    waste = pd.concat(
        [
            simulate_portfolio(
                predictions, forecast, config.shelf_life, config.lost_margin, config.waste_cost
            ).assign(forecast=forecast)
            for forecast in forecasts
        ],
        ignore_index=True,
    )
    best_forecast = best_model or "moving_average"
    policies = waste[waste["forecast"] == best_forecast].drop(columns="forecast")

    output = config.reports_dir
    output.mkdir(parents=True, exist_ok=True)
    overall.to_csv(output / "holdout_metrics.csv", index=False)
    per_fold.to_csv(output / "backtest_metrics.csv", index=False)
    by_family.to_csv(output / "family_metrics.csv", index=False)
    waste.to_csv(output / "waste_comparison.csv", index=False)
    policies.to_csv(output / "waste_policy_metrics.csv", index=False)
    skipped = sorted(set(config.models) - set(models))
    result: dict[str, object] = {
        "dataset": "Kaggle Store Sales - Time Series Forecasting",
        "horizon_days": config.horizon,
        "folds": config.backtest_folds,
        "holdout_days": config.horizon * config.backtest_folds,
        "holdout_start": str(predictions["date"].min().date()),
        "holdout_end": str(predictions["date"].max().date()),
        "families": sorted(predictions["family"].unique().tolist()),
        "shelf_life_days": {f: config.shelf_life(f) for f in sorted(predictions["family"].unique())},
        "metrics": overall.to_dict(orient="records"),
        "best_model": best_model,
        "skipped_models": skipped,
        "waste_forecast": best_forecast,
        "waste_policies": policies.to_dict(orient="records"),
        "headline": headline(waste, best_forecast),
        "caveat": (
            "Waste is simulated per store and family from sales, forecasts and the configured "
            "shelf lives and costs; it is not observed waste. Costs are in relative units."
        ),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    try:
        print(json.dumps(generate_report(load_config(args.config)), indent=2))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
