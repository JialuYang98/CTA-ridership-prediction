# CTA 'L' Station Ridership Forecasting

Next-day entry forecasts for 143 Chicago Transit Authority (CTA) 'L' rail stations. The project compares gradient-boosted trees (XGBoost), a residual LSTM and a GCN-LSTM that uses the physical rail network. All three are scored against a same-weekday-last-week baseline on identical targets.

中文说明见 [README.zh-CN.md](README.zh-CN.md).

## Results

Chronological split by target date: training Jan 29, 2018 – Dec 31, 2023; validation 2024; test Jan 1 – Nov 30, 2025. Every model is scored on the same 52,156 validation and 47,762 test station-days. All model selection used the 2024 validation year only.

| Model | Validation WMAPE | Test WMAPE | Test MAE | Test RMSE | Test bias |
|---|---:|---:|---:|---:|---:|
| Same weekday last week (baseline) | 13.64% | 13.29% | 287.4 | 795.0 | −0.2% |
| GCN-LSTM (physical rail graph) | 8.63% | 9.13% | 197.6 | 527.5 | −2.6% |
| Residual LSTM (identity graph) | 8.60% | 9.00% | 194.6 | 523.2 | −1.9% |
| **XGBoost** | **7.77%** | **8.10%** | **175.3** | **498.5** | −1.7% |

WMAPE = sum of absolute errors / sum of actual entries. MAE and RMSE are in entries per station-day.

Key findings:

- XGBoost cuts test WMAPE by 39% relative to the seasonal baseline.
- The rail graph adds no measurable accuracy at daily resolution. The residual LSTM and GCN-LSTM share code, parameter count, seed and budget; only the adjacency matrix differs.
- Holiday features matter most. Removing them from XGBoost raises validation WMAPE by 0.44 points, while removing the network-neighbor features changes it by 0.02.
- The largest remaining errors are special events (St. Patrick's Day parade, Chicago Marathon, Cubs games at Addison), which no model has features for.

## Repository layout

```text
cta_data.py                  # shared preprocessing: ridership panel, station mapping, GTFS rail graph
cta_xgboost_features.py      # feature table for XGBoost
cta_xgboost_train.py         # XGBoost search, training and test evaluation
cta_neural_forecast.py       # residual LSTM / GCN-LSTM (--graph identity | physical)
compare_models.py            # recomputes metrics on common targets and draws reports/best_models.png
test_cta_xgboost.py          # behavioral tests (synthetic data, no download needed)
test_cta_neural_forecast.py
models/
  xgboost/model.ubj          # retained XGBoost model + config, metrics, predictions, feature importance
  lstm/model.pt              # residual LSTM (identity adjacency)
  gcn_lstm/model.pt          # GCN-LSTM (physical rail adjacency)
  best_models.json           # model registry and final metrics
reports/                     # metric tables, figures, XGBoost search history, graph ablation records
CTA_RailLines.kmz, CTA_RailStations.kmz   # CTA rail line and station geometry
```

## Data

The raw data are not stored in this repository. Download them into the repository root:

1. **Ridership.** [CTA – Ridership – 'L' Station Entries – Daily Totals](https://data.cityofchicago.org/Transportation/CTA-Ridership-L-Station-Entries-Daily-Totals/5neh-572f) from the City of Chicago Data Portal. Export it as CSV and save it as `CTA_-_Ridership_-__L__Station_Entries_-_Daily_Totals.csv`.
2. **Rail network.** The [CTA GTFS feed](https://www.transitchicago.com/developers/gtfs/). Unzip it into `google_transit/`, so that the folder contains `stops.txt`, `routes.txt`, `trips.txt`, `stop_times.txt` and `transfers.txt`.

The saved models used ridership through Nov 30, 2025 and a GTFS feed downloaded in March 2026. Each model's `provenance.json` stores SHA-256 fingerprints of its inputs and source code. Evaluation refuses to run if those inputs have changed, so a newer download needs a fresh training run in a new output directory.

## Usage

```bash
pip install -r requirements.txt

# Behavioral tests. Run them separately: torch and xgboost native libraries
# have conflicted when finalized in one process.
python -m unittest -v test_cta_neural_forecast.py
python -m unittest -v test_cta_xgboost.py

# Recompute metrics and the comparison figure from the saved predictions (no data download needed)
python compare_models.py

# Rebuild the feature cache from raw data (requires the downloads above)
python cta_xgboost_features.py data/cta_features.parquet

# Retrain into a NEW directory so the retained models are never overwritten
python cta_xgboost_train.py --output experiments/new_xgboost
python cta_neural_forecast.py --graph identity --output experiments/new_lstm
python cta_neural_forecast.py --graph physical --output experiments/new_gcn_lstm

# Evaluate a frozen model on the test period (requires the exact original inputs)
python cta_xgboost_train.py --evaluate-test --output models/xgboost
python cta_neural_forecast.py --evaluate-test --output models/lstm
```

XGBoost predictions start from last week's ridership through `base_margin`, which is not stored in `model.ubj`. Load the model through `cta_xgboost_train.py`. A bare Booster without the base margin gives wrong predictions.

## Limitations

- One random seed and one validation year. Differences of about 0.1 points, such as LSTM versus GCN-LSTM, are not conclusive.
- The 2025 test period was examined during earlier exploratory work, so it is a historical benchmark rather than an untouched holdout.
- The CTA day type is read from the ridership file, and the current GTFS network is applied to all years.
- No event or weather features, and no uncertainty intervals.

## License

Apache License 2.0. See [LICENSE](LICENSE).
