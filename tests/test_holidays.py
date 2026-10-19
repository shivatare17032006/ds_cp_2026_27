from datetime import date

import polars as pl

from src.data.features import join_context

DATES = pl.date_range(date(2024, 1, 1), date(2024, 1, 10), eager=True)


def _features(stores: list[int]) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "date": [d for _ in stores for d in DATES],
            "store_nbr": [s for s in stores for _ in DATES],
        }
    ).lazy()


def _flagged(frame: pl.DataFrame, store: int = 1) -> list[date]:
    return (
        frame.filter((pl.col("store_nbr") == store) & (pl.col("holiday_flag") == 1))
        .sort("date")["date"]
        .to_list()
    )


def test_flag_only_on_the_holiday_date() -> None:
    holidays = pl.DataFrame({"date": [date(2024, 1, 5)]}).lazy()
    result = join_context(_features([1]), holidays=holidays).collect()
    assert _flagged(result) == [date(2024, 1, 5)]
    assert result.height == len(DATES)


def test_transferred_and_work_days_excluded() -> None:
    holidays = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "type": ["Holiday", "Holiday", "Work Day"],
            "locale": ["National"] * 3,
            "locale_name": ["Ecuador"] * 3,
            "transferred": ["False", "True", "False"],
        }
    ).lazy()
    result = join_context(_features([1]), holidays=holidays).collect()
    assert _flagged(result) == [date(2024, 1, 2)]


def test_regional_and_local_holidays_match_store_location() -> None:
    stores = pl.DataFrame(
        {"store_nbr": [1, 2], "city": ["Quito", "Cuenca"], "state": ["Pichincha", "Azuay"]}
    ).lazy()
    holidays = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 4)],
            "type": ["Holiday"] * 4,
            "locale": ["National", "Local", "Regional", "Local"],
            "locale_name": ["Ecuador", "Quito", "Azuay", "Cuenca"],
            "transferred": [False] * 4,
        }
    ).lazy()
    result = join_context(_features([1, 2]), stores=stores, holidays=holidays).collect()
    assert _flagged(result, 1) == [date(2024, 1, 2), date(2024, 1, 3)]
    # Overlapping regional + local holidays on the same day must not duplicate rows.
    assert _flagged(result, 2) == [date(2024, 1, 2), date(2024, 1, 4)]
    assert result.height == 2 * len(DATES)
