"""Train a global model and persist forecast artifacts for the dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_CONFIG, load_config
from src.forecasting import train_global_model


def run(config_path: str | Path, model_name: str = "lightgbm") -> Path:
    """Train the configured model from processed features."""
    config = load_config(config_path)
    feature_path = config.root / config.values["paths"]["processed_dir"] / "features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing features: {feature_path}. Run src.pipeline first.")
    features = pd.read_parquet(feature_path)
    _, forecast, _ = train_global_model(
        features,
        horizon=int(config.values["project"]["horizon_days"]),
        model_name=model_name,
        seed=int(config.values["project"]["seed"]),
    )
    output = config.root / config.values["paths"]["artifacts_dir"] / "forecast.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", choices=["lightgbm", "xgboost"], default="lightgbm")
    args = parser.parse_args()
    try:
        print(run(args.config, args.model))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
