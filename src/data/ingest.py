"""Raw Kaggle CSV validation and conversion to Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.config import PipelineConfig, load_config

DATE_COLUMNS = {"date", "Date"}
REQUIRED_TRAIN_COLUMNS = {"date", "store_nbr", "family", "sales", "onpromotion"}
# Kaggle's test.csv holds future promotions; forecasting assumes none without it.
OPTIONAL_TABLES = {"test"}


def _resolve_raw_path(config: PipelineConfig, filename: str) -> Path:
    path = Path(filename)
    return path if path.is_absolute() else config.raw_dir / path


def _normalize_train(lf: pl.LazyFrame) -> pl.LazyFrame:
    columns = set(lf.collect_schema().names())
    missing = REQUIRED_TRAIN_COLUMNS - columns
    if missing:
        raise ValueError(f"Training file is missing required columns: {sorted(missing)}")
    date_expr = (
        pl.col("date").str.to_date(strict=False)
        if lf.collect_schema()["date"] == pl.String
        else pl.col("date").cast(pl.Date, strict=False)
    )
    return (
        lf.with_columns(
            date_expr,
            pl.col("store_nbr").cast(pl.Int32, strict=False),
            pl.col("family").cast(pl.String),
            pl.col("sales").cast(pl.Float64, strict=False).fill_null(0).clip(lower_bound=0),
            pl.col("onpromotion").cast(pl.Int32, strict=False).fill_null(0),
        )
        .filter(pl.col("date").is_not_null())
        .select(["date", "store_nbr", "family", "sales", "onpromotion"])
    )


def convert_raw_to_parquet(config: PipelineConfig) -> list[Path]:
    """Validate configured CSVs and write normalized Parquet files.

    The conversion is lazy until collection/write time, so later phases can
    extend this function with projection and predicate pushdown.
    """

    config.interim_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for table, filename in config.dataset_files.items():
        source = _resolve_raw_path(config, filename)
        if not source.exists() and table in OPTIONAL_TABLES:
            continue
        if not source.exists():
            raise FileNotFoundError(
                f"Missing configured {table} file: {source}. "
                "Download the Kaggle data separately or update configs/base.yaml."
            )
        destination = config.interim_dir / f"{table}.parquet"
        lazy = pl.scan_csv(source, try_parse_dates=True, infer_schema_length=10_000)
        if table == "train":
            lazy = _normalize_train(lazy)
        else:
            # Preserve auxiliary columns now; schema-specific joins are Phase 2.
            lazy = lazy.with_columns(
                [
                    pl.col(column).cast(pl.Date, strict=False)
                    for column in DATE_COLUMNS
                    if column in lazy.collect_schema().names()
                ]
            )
        lazy.sink_parquet(destination)
        outputs.append(destination)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    outputs = convert_raw_to_parquet(load_config(args.config))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
