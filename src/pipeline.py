"""Command-line orchestration for the reproducible data-to-features phase."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.config import load_config
from src.data.features import build_features, join_context
from src.data.ingest import convert_raw_to_parquet


def apply_scope(train: pl.LazyFrame, families: list[str], stores: list[int]) -> pl.LazyFrame:
    """Keep the configured families/stores; fail on names absent from the data.

    A typo (e.g. ``BAKERY`` for ``BREAD/BAKERY``) would otherwise silently
    drop a whole category.
    """
    if families:
        available = set(train.select("family").unique().collect()["family"].to_list())
        unknown = sorted(set(families) - available)
        if unknown:
            raise ValueError(
                f"Configured families not found in the data: {unknown}. "
                f"Available: {sorted(available)}"
            )
        train = train.filter(pl.col("family").is_in(families))
    if stores:
        train = train.filter(pl.col("store_nbr").is_in(stores))
    return train


def run(config_path: str | Path) -> Path:
    """Convert raw data and write the processed feature table."""
    config = load_config(config_path)
    convert_raw_to_parquet(config)
    horizon = config.horizon
    context = {}
    for table in ("holidays_events", "oil", "transactions", "stores", "test"):
        if table not in config.dataset_files:
            continue
        path = config.interim_dir / f"{table}.parquet"
        if path.exists():
            context[table] = pl.scan_parquet(path)
    train = apply_scope(
        pl.scan_parquet(config.interim_dir / "train.parquet"), config.families, config.stores
    )
    features = build_features(train, horizon=horizon, future=context.get("test"))
    features = join_context(
        features,
        holidays=context.get("holidays_events"),
        oil=context.get("oil"),
        transactions=context.get("transactions"),
        stores=context.get("stores"),
        horizon=horizon,
    )
    output = config.processed_dir / "features.parquet"
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
