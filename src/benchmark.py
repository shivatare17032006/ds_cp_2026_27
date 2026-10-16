"""Small reproducible Polars/Pandas feature-step benchmark."""

from __future__ import annotations

import time

import pandas as pd
import polars as pl

from src.data.features import add_features


def benchmark(frame: pl.DataFrame, horizon: int = 14) -> pd.DataFrame:
    """Compare wall time and approximate memory for the feature transformation."""
    start = time.perf_counter()
    add_features(frame.lazy(), horizon).collect()
    polars_seconds = time.perf_counter() - start
    pandas_frame = frame.to_pandas()
    start = time.perf_counter()
    pandas_frame.sort_values(["store_nbr", "family", "date"]).groupby(
        ["store_nbr", "family"], observed=True
    )["sales"].shift(horizon).rolling(7).mean()
    pandas_seconds = time.perf_counter() - start
    return pd.DataFrame(
        [
            {"engine": "polars", "seconds": polars_seconds, "memory_mb": frame.estimated_size("mb")},
            {"engine": "pandas", "seconds": pandas_seconds, "memory_mb": pandas_frame.memory_usage(deep=True).sum() / 1e6},
        ]
    )
