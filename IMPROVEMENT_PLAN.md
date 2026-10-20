# FreshCast — Improvement Plan

Review of `shivatare17032006/ds_cp_2026_27` at commit `6fb86b7` (single commit, ~1,000 LOC).

**Progress:** Phases 1–3 are done (P0 #1–11, the pytest path from #12, and
P1 #1, #4, #5, #6, #8). Still open from P0 #12: the packaging/entry-point
layout. The dashboard's `No module named 'src'` crash was fixed along the way.
Next up is Phase 4; delivery lead time (P1 #9) matters most, because with
same-day delivery the simulated waste is close to zero for every forecast.

## Summary

The structure is good: a clean `src/` layout, Polars-based lazy ingestion, a YAML config,
a FIFO waste simulator, a Streamlit dashboard, and an honest README about data caveats.
The weak spot is **correctness**. Several bugs silently skew the headline numbers.
The waste metric is inflated, the holiday feature is ~always 1, two context features leak
future information, the ML model's forecast never reaches the dashboard, and the ML model
is never evaluated on a holdout. The README says the end-to-end work is done, but the
results it produces today can't be trusted.

Priorities: **P0** = produces wrong results, **P1** = missing core capability,
**P2** = quality / polish.

---

## P0 — Correctness bugs (fix first)

Items marked ✅ were reproduced with a small script against the cloned code.

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 ✅ | `src/waste.py:51` | `waste` = *all stock still on hand* at end of day, so the same unit is counted as waste every day it sits. Repro: 51 units ordered, 50 sold, **10 reported as waste**. | Count waste only when a batch expires (the batches dropped by the age filter on line 34). Add a conservation test: `ordered == sold + wasted + ending_inventory`. |
| 2 ✅ | `src/waste.py:37-38` | The "newsvendor" policy just does `order *= 0.9`, so it orders *less* than the median policy. That is the opposite of a 0.9 service quantile. | Use the critical ratio `cu/(cu+co)` (with the defaults 1.0/0.3 → 0.77) and order up to that quantile of a demand distribution (e.g. a residual-based or quantile-model forecast). |
| 3 ✅ | `src/data/features.py:130-138` | The holiday `join_asof(strategy="backward")` matches the most recent past holiday, so every date after the first holiday gets `holiday_flag = 1` (26 of 30 days in the repro). | Use an exact left join on date for the flag, plus `days_to_next_holiday` / `days_since_holiday`. Respect `locale` (National/Regional/Local) and the `transferred` / `Work Day` rows. |
| 4 | `src/data/features.py:110-129` | **Leakage**: `transactions` and `oil_price` are joined on the *same* date as the target. For a 14-day-ahead forecast these values are not known yet. | Lag them by `horizon` (like the sales features), or drop them. Promotions (`onpromotion`) are known in advance, so they can stay. |
| 5 | `src/forecasting.py:86-94` | The forecast predicts from the **last observed row** and labels it `date + 1`. Features at row *t* predict target *t*, so this is a prediction of an existing day, shifted by one. It also gives only 1 day, not the 14-day horizon. | Build future rows for *t+1…t+h* (the lags are horizon-safe, so they can be computed for future dates) and predict those. |
| 6 | `src/forecasting.py:81-91` | Categorical columns from `stores` (city, state, type) get `cat.codes` computed separately for the train frame and the latest rows. The integer codes don't match. | Fit one category mapping on the training data and reuse it. Better: pass `pd.Categorical` to LightGBM's native categorical support. |
| 7 | `app/streamlit_app.py:25-29` | `forecast.parquet` dates (max date + 1) never appear in `features.parquet`, so the left merge always gives NaN. The dashboard **never shows the ML forecast**. | Fixed by #5. Then concatenate the future rows instead of merging them onto history. |
| 8 | `src/tuning.py:45` | `study.optimize(..., seed=seed)`: `optimize` has no `seed` parameter, so this raises `TypeError`. The categoricals aren't encoded either. | Use `optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))`. Split by **date** with a `horizon`-day gap, not by row index. |
| 9 | `src/report.py:25-30` | The baselines are computed *inside* the holdout slice, so the first 7 days get `shift(7)` → 0 and are penalised. A 7-day lag also isn't available at a 14-day horizon. | Compute the baselines on the full history with `shift(horizon)`, then slice the holdout. |
| 10 | `src/report.py:40`, `src/metrics.py:27` | RMSSE is scaled with diffs over the **concatenated** series of every store/family, holdout included, so it crosses series boundaries. | Compute the scale per series on training data only, then average (M5 style). |
| 11 | `src/forecasting.py:28` | `seasonal_naive` returns `data.tail(horizon)`, which is the tail of the whole frame (i.e. the last series only). | Use `groupby(...).tail(horizon)`. |
| 12 ✅ | README / `pyproject.toml` | Bare `pytest` (as the README says) fails with `ModuleNotFoundError: No module named 'src'`. Only `python -m pytest` works. Also, `packages.find where=["src"]` installs a top-level `data` package, so the `freshcast-pipeline` entry point (`src.pipeline:main`) breaks for a normal install. | Set `pythonpath = ["."]`. Better: rename to a real package (`src/freshcast/...`), import it as `freshcast.*`, and fix the entry point. |

---

## P1 — Missing core capabilities

### Modeling and validation
1. **Evaluate the ML model.** Right now only the baselines get holdout metrics. Add a
   `backtest.py` that does rolling-origin evaluation (e.g. 4 folds × 14 days, with a
   `horizon` gap). Report WAPE / bias / RMSSE for seasonal-naive, moving-average,
   LightGBM and XGBoost side by side, overall and per family.
2. **Real prediction intervals.** The dashboard's `±25%` band is made up
   (`streamlit_app.py:77-78`). Train LightGBM quantile models (`objective="quantile"`,
   α = 0.1 / 0.5 / 0.9) or use conformal residuals. Report coverage and pinball loss with the
   existing `evaluate_intervals`. These quantiles also feed the newsvendor policy (P0-2).
3. **Wire up `tuning.py` and `explainability.py`.** Both are unused. Add `--tune` to
   `train.py`, persist the model (`joblib`) plus its feature list, and write SHAP importances
   to `reports/generated/` so the dashboard can show them.

### Data and features
4. **Use the config.** `scope.families` and `assumptions.shelf_life_days` are never read.
   `report.py` hard-codes paths, the shelf life (7) and the holdout length. Pass
   `PipelineConfig` everywhere, and add typed accessors with validation (dataclass or pydantic).
5. **Fix the family names.** In Store Sales the bakery family is `BREAD/BAKERY`, not
   `BAKERY`. Validate the configured families against the data and fail loudly on a mismatch.
6. **Grid hygiene.** The dense grid zero-fills days *before a store/item first sold* and on
   store-wide closures (e.g. Dec 25). The model then learns fake zeros. Start each series
   at its first sale and mark closures separately.
7. **Richer features:** promo lags/leads, day-of-week × family target encoding, the
   Ecuador paydays (15th and last day; the current `>=25 or <=5` window is a rough guess),
   and an April 2016 earthquake flag.

### Waste simulation
8. Simulate **per store × family** with the family's shelf life, then aggregate.
   Summing all stores into one series (`report.py:43-45`) hides almost all the waste.
9. Add **lead time** (orders currently arrive the same day), an order-review cadence, and
   a sensitivity sweep over shelf life and cost ratios. Show it as a cost-vs-service frontier
   chart.
10. Add an explicit "forecast-driven vs naive" comparison. This is the headline result the
    project should be able to defend ("X% less simulated waste at equal service level").

### Dashboard
11. Show history plus the 14-day future forecast, with real intervals, backtest metrics,
    SHAP importance, and per-family shelf life taken from the config.
12. Read only the columns and rows needed (filter the Parquet with Polars by store/family)
    instead of `pd.read_parquet` on the whole feature table.
13. Fix the "Stockout risk" KPI. Today it's an ad-hoc formula; derive it from the
    simulation or the forecast quantiles. Make "Recommended order" subtract on-hand
    inventory.

---

## P2 — Engineering quality

- **Tests:** coverage is 6 tests. Add tests for each P0 fix (waste conservation, holiday
  flag, no-leakage check that asserts features at *t* are unchanged when future sales are
  perturbed), plus `join_context`, `report`, `train` on a tiny synthetic dataset,
  and an end-to-end smoke test.
- **CI:** add a GitHub Actions workflow (ruff, mypy, pytest with coverage) on Python 3.10–3.12.
  `mypy --strict` is configured but the code would not pass it yet.
- **Repo hygiene:** remove the committed `src/*.egg-info/` and `.vscode/` and add them to
  `.gitignore`. Add a `LICENSE`, and a `Makefile`/`tasks.py`
  (`ingest → features → train → report → app`).
- **Benchmark fairness:** `benchmark.py` compares the full Polars feature step with a
  single Pandas rolling mean. Implement the same features in both or drop the claim.
- **Reproducibility:** pin a lockfile (`uv.lock` / `requirements.txt`), save run metadata
  (config hash, git SHA, data date range) into `results.json`, and add a data download
  helper that uses the Kaggle CLI and the user's own credentials.
- **Docs:** fix the Quick start, add an architecture diagram, a results table, a "Known
  limitations" section, and screenshots of the dashboard. Put EDA in `notebooks/`, which is
  empty now.

---

## Suggested order of work

| Phase | Scope | Outcome |
|---|---|---|
| 1 | P0 #1–3, #12 + tests | Waste and holiday numbers are correct; `pytest` works |
| 2 | P0 #4–11 | No leakage; the ML forecast is real and multi-day; metrics are correct |
| 3 | P1 #1, #4, #5, #8 | Backtested ML vs baselines; config-driven per-family waste |
| 4 | P1 #2, #3, #9, #10 | Quantile forecasts, a true newsvendor, the headline waste result |
| 5 | P1 #11–13, P2 | Dashboard shows real results; CI, docs, hygiene |

After Phase 2, regenerate `reports/generated/`. Any results produced before that are affected
by bugs #1, #3, #4, #9 and #10.
