"""Command-line orchestration for the reproducible data-to-features phase."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.config import load_config
from src.data.features import build_features, join_context
from src.data.ingest import convert_raw_to_parquet


def run(config_path: str | Path) -> Path:
    """Convert raw data and write the processed feature table."""
    config = load_config(config_path)
    convert_raw_to_parquet(config)
    horizon = int(config.values["project"]["horizon_days"])
    context = {}
    for table in ("holidays_events", "oil", "transactions", "stores", "test"):
        if table not in config.dataset_files:
            continue
        path = config.interim_dir / f"{table}.parquet"
        if path.exists():
            context[table] = pl.scan_parquet(path)
    features = build_features(
        pl.scan_parquet(config.interim_dir / "train.parquet"),
        horizon=horizon,
        future=context.get("test"),
    )
    features = join_context(
        features,
        holidays=context.get("holidays_events"),
        oil=context.get("oil"),
        transactions=context.get("transactions"),
        stores=context.get("stores"),
        horizon=horizon,
    )
    output = config.root / config.values["paths"]["processed_dir"] / "features.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    features.sink_parquet(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
