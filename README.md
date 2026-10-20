# Predictive Supply Chain & Food Waste Reduction

An end-to-end portfolio project for forecasting grocery demand and evaluating
ordering policies that reduce simulated food waste while maintaining service.

## Status

The end-to-end implementation is now present: ingestion, dense-grid feature
engineering, leakage-safe baselines/model adapters, validation metrics, FIFO
waste simulation, optional SHAP explanations, a Polars/Pandas benchmark, and a
Streamlit control-tower UI. The Kaggle Store Sales data has been processed and
the first real-data holdout and waste-policy results are generated under
`reports/generated/`.

## Data caveats

- Kaggle data is not distributed with this repository. Download the approved
  dataset separately and place files under `data/raw/` (or change the YAML
  paths).
- Favorita records are sales transactions, not observed inventory. A missing
  record is treated as zero only after the configured store-item-date grid is
  created.
- Food waste is simulated using configurable shelf-life, ordering, and cost
  assumptions; it is not observed waste.
- Returns are clipped at zero for demand modeling. This avoids negative demand
  targets but may hide return-related operational behavior.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,modeling,app]"
python -m src.pipeline --config configs\base.yaml
python -m src.train
pytest
```

The pipeline validates the configured raw files, writes normalized Parquet
tables to `data/interim/`, and writes leakage-safe features to
`data/processed/features.parquet`. It does not download Kaggle data or infer
credentials.

The feature table also contains `horizon_days` future dates per store and
family (with empty `sales`); `src.train` forecasts those dates into
`data/artifacts/forecast.parquet`. Transactions and oil prices are only used
lagged by the horizon, since they are not known in advance. Kaggle's
`test.csv` is optional: when present, its promotions are used for the
forecast dates, otherwise future promotions are assumed to be zero.

Launch the dashboard after installing the app extra:

```powershell
streamlit run app\streamlit_app.py
```

The dashboard includes a demo view when no processed data exists, so the UI
can be reviewed before downloading Kaggle files. Demo values are not model
results.

The AI Assistant page answers common questions offline. For free-form
answers it calls Google Gemini: install the `ai` extra
(`pip install -e ".[ai]"`) and enter a Gemini API key in the sidebar or set
`GEMINI_API_KEY`. Stockout risk on the dashboard is the share of the last 28
days on which demand exceeded the next-day forecast; simulated costs are in
relative units, not a currency.

Generate the backtest and simulated waste-policy report:

```powershell
python -m src.report --config configs\base.yaml
```

The report runs a rolling-origin backtest: `evaluation.folds` consecutive
windows of `horizon_days` at the end of the history (3 × 14 = the last 42
days by default). In each window the baselines (seasonal naive, 14-day
moving average) and the configured ML models (`evaluation.models`) forecast
using only data available before the window starts. Days before a store first
sold a category, and days a whole store sold nothing (closures), are excluded
from training and scoring.

Waste is then simulated for every store and category separately, with the
category's shelf life from `assumptions.shelf_life_days`, for three ordering
policies (forecast + 10%, forecast only, cost-optimal newsvendor). The
headline in `results.json` compares the best model's forecasts with the
moving-average forecasts under newsvendor ordering. Orders arrive the same
day in this simulation, which keeps simulated waste very low; the headline
says so when waste is too small to compare.

Outputs in `reports/generated/`: `holdout_metrics.csv` (one row per model),
`backtest_metrics.csv` (per window), `family_metrics.csv` (WAPE per
category), `waste_comparison.csv` (every forecast × policy),
`waste_policy_metrics.csv` (policies for the best forecast) and
`results.json`.

## Repository layout

```text
src/       reusable ingestion, features, models, metrics, simulation code
notebooks/ exploratory work only
configs/   YAML configuration and assumptions
tests/     unit and data-contract tests
app/       Streamlit control-tower dashboard
reports/   generated charts and results
data/      ignored raw/interim/processed artifacts
```
