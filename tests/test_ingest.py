from pathlib import Path

import polars as pl
import pytest

from src.config import load_config
from src.data.ingest import _normalize_train, convert_raw_to_parquet


def test_normalize_train_clips_returns_and_casts_schema() -> None:
    frame = pl.DataFrame(
        {
            "date": ["2017-01-01", "2017-01-02"],
            "store_nbr": ["1", "1"],
            "family": ["PRODUCE", "PRODUCE"],
            "sales": pl.Series("sales", [-2, 3.5], dtype=pl.Float64),
            "onpromotion": [None, 1],
        }
    )
    result = _normalize_train(frame.lazy()).collect()
    assert result["sales"].to_list() == [0.0, 3.5]
    assert result["store_nbr"].dtype == pl.Int32
    assert result["date"].dtype == pl.Date
    assert result["onpromotion"].to_list() == [0, 1]


def test_normalize_train_rejects_missing_columns() -> None:
    frame = pl.DataFrame({"date": ["2017-01-01"], "sales": [1]}).lazy()
    with pytest.raises(ValueError, match="missing required columns"):
        _normalize_train(frame)


def test_ingestion_reports_missing_raw_file(tmp_path: Path) -> None:
    config = load_config(Path("configs/base.yaml"))
    values = dict(config.values)
    values["paths"] = dict(values["paths"])
    values["paths"]["raw_dir"] = str(tmp_path)
    altered = type(config)(root=Path.cwd(), values=values)
    with pytest.raises(FileNotFoundError, match="Missing configured"):
        convert_raw_to_parquet(altered)
