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
pytest
```

The pipeline validates the configured raw files, writes normalized Parquet
tables to `data/interim/`, and writes leakage-safe features to
`data/processed/features.parquet`. It does not download Kaggle data or infer
credentials.

Launch the dashboard after installing the app extra:

```powershell
streamlit run app\streamlit_app.py
```

The dashboard includes a demo view when no processed data exists, so the UI
can be reviewed before downloading Kaggle files. Demo values are not model
results.

Generate the final holdout and simulated waste-policy report:

```powershell
python -m src.report
```

Results are written to `reports/generated/holdout_metrics.csv`,
`reports/generated/waste_policy_metrics.csv`, and
`reports/generated/results.json`.

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
