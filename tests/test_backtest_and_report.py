import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.backtest import backtest
from src.config import load_config
from src.data.features import build_complete_grid, build_features, mark_open_days
from src.forecasting import prepare_features, trainable_rows
from src.pipeline import apply_scope
from src.report import generate_report, headline
from src.waste import simulate_fifo, simulate_portfolio

HORIZON = 7
START = date(2024, 1, 1)


def _train(days: int = 90, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [
        {
            "date": START + timedelta(days=d),
            "store_nbr": store,
            "family": family,
            "sales": float(rng.poisson(5 * store + (3 if family == "DAIRY" else 1))),
            "onpromotion": 0,
        }
        for store in (1, 2)
        for family in ("DAIRY", "PRODUCE")
        for d in range(days)
    ]
    return pl.DataFrame(rows).with_columns(pl.col("store_nbr").cast(pl.Int32))


# ── Config and scope ─────────────────────────────────────────────────────────
def test_config_matches_store_sales_names_and_shelf_lives() -> None:
    config = load_config(Path("configs/base.yaml"))
    assert "BREAD/BAKERY" in config.families and "BAKERY" not in config.families
    assert config.shelf_life("BREAD/BAKERY") == 3
    assert config.shelf_life("NOT CONFIGURED") == 7  # default
    assert config.backtest_folds * config.horizon == 42


def test_apply_scope_rejects_unknown_families() -> None:
    train = _train(days=5).lazy()
    with pytest.raises(ValueError, match=r"not found in the data: \['BAKERY'\].*Available"):
        apply_scope(train, ["DAIRY", "BAKERY"], [])
    scoped = apply_scope(train, ["DAIRY"], [2]).collect()
    assert set(scoped["family"]) == {"DAIRY"} and set(scoped["store_nbr"]) == {2}


# ── Closed / not-yet-stocked days ────────────────────────────────────────────
def test_mark_open_days_flags_prelaunch_and_store_closures() -> None:
    dates = [START + timedelta(days=d) for d in range(5)]
    frame = pl.DataFrame(
        {
            "date": dates * 2,
            "store_nbr": [1] * 10,
            "family": ["DAIRY"] * 5 + ["PRODUCE"] * 5,
            # PRODUCE is first stocked on day 2; the store is closed on day 3.
            "sales": [4.0, 4.0, 4.0, 0.0, 4.0] + [0.0, 0.0, 2.0, 0.0, 2.0],
            "onpromotion": [0] * 10,
        }
    )
    marked = mark_open_days(build_complete_grid(frame.lazy())).collect().sort(["family", "date"])
    assert marked.filter(pl.col("family") == "DAIRY")["is_open"].to_list() == [1, 1, 1, 0, 1]
    assert marked.filter(pl.col("family") == "PRODUCE")["is_open"].to_list() == [0, 0, 1, 0, 1]


def test_lags_do_not_treat_days_before_first_sale_as_zero_demand() -> None:
    dates = [START + timedelta(days=d) for d in range(40)]
    frame = pl.DataFrame(
        {"date": dates, "store_nbr": [1] * 40, "family": ["DAIRY"] * 40,
         # Not stocked for the first 20 days, then a steady 10 a day.
         "sales": [0.0] * 20 + [10.0] * 20, "onpromotion": [0] * 40}
    )
    features = build_features(frame.lazy(), horizon=HORIZON, forecast_rows=False).collect()
    launch = START + timedelta(days=20)
    after = features.filter(pl.col("date") >= launch + timedelta(days=HORIZON))
    # A lag that reaches back before the launch is unknown, not zero.
    assert after.filter(pl.col("date") < launch + timedelta(days=HORIZON + 6))["sales_mean_7"].null_count() == 6
    assert after.filter(pl.col("date") >= launch + timedelta(days=HORIZON + 6))["sales_mean_7"].to_list() == [10.0] * 7
    assert after[f"sales_lag_{HORIZON}"].to_list() == [10.0] * 13


def test_cli_reports_missing_inputs_without_a_traceback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.report

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text(
        "project: {seed: 1, dataset: store_sales, horizon_days: 7}\n"
        "paths: {raw_dir: raw, interim_dir: interim, processed_dir: processed, artifacts_dir: artifacts}\n"
        "dataset: {store_sales: {train: train.csv}}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["report", "--config", str(tmp_path / "configs" / "base.yaml")])
    with pytest.raises(SystemExit, match="error: Missing features: .*Run src.pipeline first"):
        src.report.main()


def test_closed_days_are_not_training_targets() -> None:
    features = build_features(_train().lazy(), horizon=HORIZON).collect().to_pandas()
    closed_day = pd.Timestamp(START + timedelta(days=60))
    closed = pd.to_datetime(features["date"]) == closed_day
    assert closed.sum() == 4  # 2 stores x 2 families
    features.loc[closed, "is_open"] = 0
    frame, columns = prepare_features(features)
    rows = trainable_rows(frame, columns)
    assert "is_open" not in columns
    assert closed_day not in set(pd.to_datetime(rows["date"]))
    assert pd.Timestamp(START + timedelta(days=61)) in set(pd.to_datetime(rows["date"]))
    assert rows["target"].notna().all()


# ── Backtest ─────────────────────────────────────────────────────────────────
def test_backtest_windows_are_consecutive_and_end_at_last_observed_day() -> None:
    features = build_features(_train().lazy(), horizon=HORIZON).collect().to_pandas()
    predictions = backtest(features, horizon=HORIZON, folds=3, models=())
    last = pd.Timestamp(START + timedelta(days=89))
    assert predictions["date"].max() == last
    assert predictions["date"].nunique() == 3 * HORIZON
    spans = predictions.groupby("fold")["date"].agg(["min", "max"])
    assert list(spans["max"] - spans["min"]) == [pd.Timedelta(days=HORIZON - 1)] * 3
    assert (predictions.groupby("fold")["origin"].first().diff().dropna() == pd.Timedelta(days=HORIZON)).all()


def test_backtest_predictions_do_not_see_data_after_the_origin() -> None:
    pytest.importorskip("lightgbm")
    train = _train()
    baseline = backtest(build_features(train.lazy(), horizon=HORIZON).collect().to_pandas(),
                        horizon=HORIZON, folds=2, models=("lightgbm",))
    origin = baseline.loc[baseline["fold"] == 1, "origin"].iloc[0].date()
    perturbed_train = train.with_columns(
        pl.when(pl.col("date") > origin).then(pl.col("sales") * 5 + 50).otherwise(pl.col("sales")).alias("sales")
    )
    perturbed = backtest(build_features(perturbed_train.lazy(), horizon=HORIZON).collect().to_pandas(),
                         horizon=HORIZON, folds=2, models=("lightgbm",))
    columns = ["seasonal_naive", "moving_average", "lightgbm"]
    first = baseline[baseline["fold"] == 1].sort_values(["store_nbr", "family", "date"])[columns]
    again = perturbed[perturbed["fold"] == 1].sort_values(["store_nbr", "family", "date"])[columns]
    np.testing.assert_allclose(first.to_numpy(), again.to_numpy())


# ── Per-store waste simulation ───────────────────────────────────────────────
def test_portfolio_simulates_each_store_separately() -> None:
    dates = pd.date_range("2024-01-01", periods=6)
    frame = pd.DataFrame(
        {
            "store_nbr": [1] * 6 + [2] * 6,
            "family": ["DAIRY"] * 12,
            "date": list(dates) * 2,
            # Store 1 over-forecasts, store 2 under-forecasts: summed together
            # they would cancel out and show no waste.
            "sales": [0.0] * 6 + [10.0] * 6,
            "forecast": [10.0] * 6 + [0.0] * 6,
        }
    )
    lives: list[str] = []
    table = simulate_portfolio(frame, "forecast", lambda f: lives.append(f) or 2).set_index("policy")
    assert table.loc["forecast_only", "units_wasted"] > 0
    assert table.loc["forecast_only", "service_level"] == 0.0
    assert set(lives) == {"DAIRY"}
    combined = simulate_fifo(frame.groupby("date")["sales"].sum(), frame.groupby("date")["forecast"].sum(), 2)
    assert combined["waste"].sum() == 0  # the aggregate view hides it


def test_headline_withholds_relative_change_when_waste_is_negligible() -> None:
    waste = pd.DataFrame(
        {
            "policy": ["newsvendor", "newsvendor"],
            "forecast": ["lightgbm", "moving_average"],
            "waste_pct": [0.0001, 0.0004],
            "service_level": [0.95, 0.93],
            "cost": [90.0, 100.0],
            "units_ordered": [10_000.0, 10_000.0],
            "units_wasted": [1.0, 4.0],
        }
    )
    result = headline(waste, "lightgbm")
    assert result["waste_reduction_pct"] is None and "note" in result
    assert result["cost_reduction_pct"] == pytest.approx(0.1)
    meaningful = headline(waste.assign(waste_pct=[0.02, 0.04], units_wasted=[200.0, 400.0]), "lightgbm")
    assert meaningful["waste_reduction_pct"] == pytest.approx(0.5)
    assert headline(waste, "moving_average")["available"] is False


# ── End to end ───────────────────────────────────────────────────────────────
def test_generate_report_writes_all_outputs(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text(
        "project: {seed: 1, dataset: store_sales, horizon_days: 7}\n"
        "paths: {raw_dir: raw, interim_dir: interim, processed_dir: processed, "
        "artifacts_dir: artifacts, reports_dir: out}\n"
        "evaluation: {folds: 2, models: [lightgbm, not_a_real_model]}\n"
        "assumptions: {lost_margin: 1.0, waste_cost: 0.3, shelf_life_days: {default: 4, DAIRY: 2}}\n"
        "dataset: {store_sales: {train: train.csv}}\n",
        encoding="utf-8",
    )
    config = load_config(tmp_path / "configs" / "base.yaml")
    config.processed_dir.mkdir()
    build_features(_train().lazy(), horizon=HORIZON).collect().write_parquet(
        config.processed_dir / "features.parquet"
    )
    result = generate_report(config)
    names = {"holdout_metrics", "backtest_metrics", "family_metrics", "waste_comparison", "waste_policy_metrics"}
    assert all((tmp_path / "out" / f"{name}.csv").exists() for name in names)
    saved = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert saved["best_model"] == "lightgbm"
    assert saved["skipped_models"] == ["not_a_real_model"]
    assert saved["shelf_life_days"] == {"DAIRY": 2, "PRODUCE": 4}
    assert saved["headline"]["available"] is True
    assert {row["model"] for row in result["metrics"]} == {"seasonal_naive", "moving_average", "lightgbm"}
    waste = pd.read_csv(tmp_path / "out" / "waste_comparison.csv")
    assert set(waste["forecast"]) == {"moving_average", "lightgbm"}
