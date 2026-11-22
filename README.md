# CTA 'L' Station Ridership Forecasting

Next-day entry forecasts for 143 Chicago Transit Authority (CTA) 'L' rail stations. The project compares gradient-boosted trees (XGBoost), a residual LSTM and a GCN-LSTM that uses the physical rail network. All three are scored against a same-weekday-last-week baseline on identical targets.


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

### Exact snapshot used by the saved models

Each model's `provenance.json` stores SHA-256 fingerprints of its inputs and source code, and evaluation refuses to run if they have changed. Both public sources are updated regularly, so fresh downloads will not match.

- **Ridership:** download `CTA_L_station_entries_2001-01-01_to_2025-11-30.zip` from the [data-2025-11-30 release](https://github.com/JialuYang98/CTA-ridership-prediction/releases/tag/data-2025-11-30) and unzip it into the repository root. It contains the exact CSV used: 2001-01-01 to 2025-11-30, 1,298,399 rows, retrieved March 8, 2026.
- **GTFS:** not included, because CTA's developer license limits redistribution. The models used the feed with files dated February 4, 2026 (service calendar January 6 – March 31, 2026), retrieved March 9, 2026. CTA hosts only the current feed; third-party GTFS archives may hold this version.

| File | SHA-256 |
|---|---|
| `CTA_-_Ridership_-__L__Station_Entries_-_Daily_Totals.csv` | `0dfbb526903900c01269802584818404358c873c9144a28988ea47665b59c56e` |
| `google_transit/stops.txt` | `fa90e850540de3e58068cd518591c1754fc43da99d5d9093b97af3807ff44f34` |
| `google_transit/routes.txt` | `5a13d264674ff85b3f679dc3b0f46b1d2e4dc779b0f1bc91969b194005301412` |
| `google_transit/trips.txt` | `faa96e865399c33328746a3ed852ca987365885a2b49e6bf0dc59c82fc801cc3` |
| `google_transit/stop_times.txt` | `4cd34872e4c4f3fabb5e1b7cfe2bbbedda4bf7ee85815457978f29fac2968b78` |
| `google_transit/transfers.txt` | `2de6f47ab836c7ac0209b643cc303f5c81a50b387c8ec10edbc182afa511307d` |

With newer downloads, retrain into a new output directory instead.

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
