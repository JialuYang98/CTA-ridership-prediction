#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost baseline for CTA daily rail ridership next-day prediction.

This script mirrors the existing CTA GCN+LSTM pipeline in several ways:
- same ridership cleaning logic
- same GTFS-based graph construction
- same target-date time splits
- same reported metrics: MAE / RMSE / WMAPE on raw rides

Modeling choice:
- one row = one (station, target_date) sample
- target = log1p(rides) for that station/date
- features include:
    * lag demand features
    * rolling statistics
    * calendar/date features
    * station static / network features
    * neighbor interaction features derived from the GTFS graph
    * categorical station_id and daytype
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb


# =========================================================
# Configuration
# =========================================================
RIDERSHIP_CSV = "CTA_-_Ridership_-__L__Station_Entries_-_Daily_Totals.csv"
STOPS_TXT = "google_transit/stops.txt"
ROUTES_TXT = "google_transit/routes.txt"
TRIPS_TXT = "google_transit/trips.txt"
STOP_TIMES_TXT = "google_transit/stop_times.txt"
TRANSFERS_TXT = "google_transit/transfers.txt"

OUTPUT_DIR = Path("cta_xgboost_outputs")

START_DATE = "2018-01-01"
SEQ_LEN = 28
SEED = 42

TRACK_EDGE_WEIGHT = 1.0
TRANSFER_EDGE_WEIGHT = 0.5

MANUAL_STATION_ID_MAP = {
    "40200": "41700",  # Randolph/Wabash -> Washington/Wabash
    "40640": "41700",  # Madison/Wabash -> Washington/Wabash
    "40500": "40370",  # Washington/State -> Washington
    "40260": "41660",  # State/Lake -> Lake (prototype assumption)
}
DROP_STATION_IDS = {"41580"}  # Homan, obsolete and no GTFS parent station

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "n_jobs": 1,
}
EARLY_STOPPING_ROUNDS = 50


# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def wmape(y_true, y_pred) -> float:
    denom = float(np.sum(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def clean_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


# =========================================================
# 1. Load ridership
# =========================================================
def load_ridership(csv_path: str, start_date: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["station_id"] = df["station_id"].astype(str)

    df = (
        df.groupby(["station_id", "stationname", "date", "daytype"], as_index=False)["rides"]
        .mean()
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )

    df = df[df["date"] >= pd.Timestamp(start_date)].copy()

    df["station_id"] = df["station_id"].replace(MANUAL_STATION_ID_MAP)
    df = df[~df["station_id"].isin(DROP_STATION_IDS)].copy()

    # Re-aggregate after remapping
    df = (
        df.groupby(["station_id", "date", "daytype"], as_index=False)["rides"]
        .sum()
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )
    return df


# =========================================================
# 2. Build CTA rail graph from GTFS
# =========================================================
def load_gtfs_graph():
    stops = pd.read_csv(STOPS_TXT)
    routes = pd.read_csv(ROUTES_TXT)
    trips = pd.read_csv(TRIPS_TXT)
    stop_times = pd.read_csv(STOP_TIMES_TXT)
    transfers = pd.read_csv(TRANSFERS_TXT)

    stops["stop_id"] = stops["stop_id"].map(clean_id)
    stops["parent_station"] = stops["parent_station"].map(clean_id)
    routes["route_id"] = routes["route_id"].map(clean_id)
    trips["route_id"] = trips["route_id"].map(clean_id)
    trips["trip_id"] = trips["trip_id"].map(clean_id)
    stop_times["trip_id"] = stop_times["trip_id"].map(clean_id)
    stop_times["stop_id"] = stop_times["stop_id"].map(clean_id)
    transfers["from_stop_id"] = transfers["from_stop_id"].map(clean_id)
    transfers["to_stop_id"] = transfers["to_stop_id"].map(clean_id)

    rail_routes = set(routes.loc[routes["route_type"] == 1, "route_id"].astype(str))
    rail_trips = trips[trips["route_id"].isin(rail_routes)][["trip_id", "route_id"]].copy()

    stops["node_station_id"] = np.where(
        stops["location_type"] == 1,
        stops["stop_id"],
        stops["parent_station"],
    )
    stops["node_station_id"] = stops["node_station_id"].map(clean_id)
    stop_to_parent = dict(zip(stops["stop_id"], stops["node_station_id"]))

    parent_rows = stops[stops["location_type"] == 1].copy()
    parent_meta = parent_rows[["stop_id", "stop_name", "stop_lat", "stop_lon"]].rename(
        columns={"stop_id": "station_id", "stop_name": "station_name"}
    )

    missing_parent_ids = (
        pd.Series(list(set(stops["node_station_id"].unique()) - set(parent_meta["station_id"].unique())))
        .replace("", np.nan)
        .dropna()
        .astype(str)
        .tolist()
    )
    if missing_parent_ids:
        fallback = (
            stops[stops["node_station_id"].isin(missing_parent_ids)]
            .groupby("node_station_id", as_index=False)
            .agg(
                station_name=("stop_name", "first"),
                stop_lat=("stop_lat", "mean"),
                stop_lon=("stop_lon", "mean"),
            )
            .rename(columns={"node_station_id": "station_id"})
        )
        parent_meta = pd.concat([parent_meta, fallback], ignore_index=True)

    parent_meta["station_id"] = parent_meta["station_id"].astype(str)
    parent_meta = parent_meta.drop_duplicates("station_id").reset_index(drop=True)

    rail_stop_times = stop_times[stop_times["trip_id"].isin(set(rail_trips["trip_id"]))].copy()
    rail_stop_times["parent_station_id"] = rail_stop_times["stop_id"].map(stop_to_parent)
    rail_stop_times = rail_stop_times[rail_stop_times["parent_station_id"].notna()].copy()
    rail_stop_times = rail_stop_times.sort_values(["trip_id", "stop_sequence"])
    rail_stop_times["prev_parent_station_id"] = rail_stop_times.groupby("trip_id")["parent_station_id"].shift(1)

    track_pairs = rail_stop_times[
        rail_stop_times["prev_parent_station_id"].notna()
        & (rail_stop_times["parent_station_id"] != rail_stop_times["prev_parent_station_id"])
    ][["prev_parent_station_id", "parent_station_id"]].drop_duplicates()

    track_edges = set()
    for _, row in track_pairs.iterrows():
        a = clean_id(row["prev_parent_station_id"])
        b = clean_id(row["parent_station_id"])
        if a and b and a != b:
            track_edges.add(tuple(sorted((a, b))))

    transfer_edges = set()
    for _, row in transfers.iterrows():
        a = clean_id(stop_to_parent.get(row["from_stop_id"], ""))
        b = clean_id(stop_to_parent.get(row["to_stop_id"], ""))
        if a and b and a != b:
            transfer_edges.add(tuple(sorted((a, b))))

    return parent_meta, track_edges, transfer_edges


# =========================================================
# 3. Build station-day panel aligned to graph nodes
# =========================================================
def build_station_panel(ridership_df: pd.DataFrame, parent_meta: pd.DataFrame, track_edges, transfer_edges):
    graph_nodes = set(parent_meta["station_id"].astype(str))
    ridership_nodes = set(ridership_df["station_id"].astype(str))
    node_ids = sorted(graph_nodes & ridership_nodes)

    meta = parent_meta[parent_meta["station_id"].isin(node_ids)].copy().reset_index(drop=True)
    meta = meta.sort_values("station_id").reset_index(drop=True)

    all_dates = pd.date_range(ridership_df["date"].min(), ridership_df["date"].max(), freq="D")

    rides_pivot = ridership_df.pivot_table(index="date", columns="station_id", values="rides", aggfunc="sum")
    rides_pivot = rides_pivot.reindex(index=all_dates, columns=node_ids)

    obs_mask = (~rides_pivot.isna()).astype(np.float32).values
    rides_raw = rides_pivot.fillna(0.0).values.astype(np.float32)
    rides_log = np.log1p(rides_raw)

    daytype_df = ridership_df.groupby("date", as_index=False)["daytype"].agg(lambda x: x.mode().iloc[0])
    daytype_series = daytype_df.set_index("date")["daytype"].reindex(all_dates)
    daytype_series = daytype_series.fillna("W").astype(str)

    cal = pd.DataFrame({"date": all_dates})
    cal["dow"] = cal["date"].dt.dayofweek
    cal["month"] = cal["date"].dt.month
    cal["day_of_year"] = cal["date"].dt.dayofyear
    cal["is_weekend"] = (cal["dow"] >= 5).astype(np.float32)
    cal["post_covid"] = (cal["date"] >= pd.Timestamp("2021-01-01")).astype(np.float32)
    cal["dow_sin"] = np.sin(2 * np.pi * cal["dow"] / 7.0)
    cal["dow_cos"] = np.cos(2 * np.pi * cal["dow"] / 7.0)
    cal["month_sin"] = np.sin(2 * np.pi * cal["month"] / 12.0)
    cal["month_cos"] = np.cos(2 * np.pi * cal["month"] / 12.0)
    cal["doy_sin"] = np.sin(2 * np.pi * cal["day_of_year"] / 365.25)
    cal["doy_cos"] = np.cos(2 * np.pi * cal["day_of_year"] / 365.25)
    cal["trend"] = np.arange(len(cal), dtype=np.float32) / max(1, len(cal) - 1)
    cal = cal.set_index("date")

    id_to_idx = {sid: i for i, sid in enumerate(meta["station_id"].astype(str))}
    degree = np.zeros(len(meta), dtype=np.float32)
    is_transfer = np.zeros(len(meta), dtype=np.float32)

    edge_list = []
    for a, b in track_edges:
        if a in id_to_idx and b in id_to_idx:
            ia, ib = id_to_idx[a], id_to_idx[b]
            edge_list.append((ia, ib, TRACK_EDGE_WEIGHT))
            degree[ia] += 1
            degree[ib] += 1
    for a, b in transfer_edges:
        if a in id_to_idx and b in id_to_idx:
            ia, ib = id_to_idx[a], id_to_idx[b]
            edge_list.append((ia, ib, TRANSFER_EDGE_WEIGHT))
            is_transfer[ia] = 1.0
            is_transfer[ib] = 1.0

    def zscore(v: np.ndarray) -> np.ndarray:
        mu = float(v.mean())
        sd = float(v.std() + 1e-6)
        return ((v - mu) / sd).astype(np.float32)

    meta["lat_z"] = zscore(meta["stop_lat"].values.astype(np.float32))
    meta["lon_z"] = zscore(meta["stop_lon"].values.astype(np.float32))
    meta["degree_z"] = zscore(degree)
    meta["is_transfer"] = is_transfer.astype(np.float32)
    meta["station_id"] = meta["station_id"].astype(str)
    meta["station_name"] = meta["station_name"].astype(str)

    # Neighbor-weight matrix without self-loops, row-normalized
    n = len(meta)
    neighbor_w = np.zeros((n, n), dtype=np.float32)
    for i, j, w in edge_list:
        neighbor_w[i, j] += w
        neighbor_w[j, i] += w
    row_sum = neighbor_w.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    neighbor_w = neighbor_w / row_sum

    info = {
        "dates": all_dates,
        "station_ids": meta["station_id"].tolist(),
        "station_names": meta["station_name"].tolist(),
        "num_nodes": len(meta),
        "edge_list": edge_list,
        "neighbor_weight_matrix": neighbor_w,
        "meta": meta,
        "calendar": cal,
        "daytype_series": daytype_series,
        "rides_raw": rides_raw,
        "rides_log": rides_log,
        "obs_mask": obs_mask,
    }
    return info


# =========================================================
# 4. Build XGBoost table
# =========================================================
def build_feature_table(info: Dict, seq_len: int) -> pd.DataFrame:
    dates = pd.DatetimeIndex(info["dates"])
    station_ids = info["station_ids"]
    station_names = info["station_names"]
    meta = info["meta"].set_index("station_id")
    cal = info["calendar"]
    daytype_series = info["daytype_series"]
    rides_raw = info["rides_raw"]
    rides_log = info["rides_log"]
    obs_mask = info["obs_mask"]
    neighbor_w = info["neighbor_weight_matrix"]

    # Precompute neighbor aggregates over stations for each day
    neighbor_lag_base = rides_log @ neighbor_w.T  # [T, N]

    rows: List[Dict] = []
    n_dates, n_nodes = rides_log.shape

    for t in range(seq_len, n_dates):
        target_date = dates[t]
        split = "train" if target_date < pd.Timestamp("2024-01-01") else (
            "valid" if target_date < pd.Timestamp("2025-01-01") else "test"
        )

        # Date-level features for target date
        cal_row = cal.loc[target_date]
        daytype = str(daytype_series.loc[target_date])

        hist7 = rides_log[t - 7:t, :]
        hist14 = rides_log[t - 14:t, :]
        hist28 = rides_log[t - 28:t, :]

        neigh_hist7_mean = hist7.mean(axis=0) @ neighbor_w.T
        neigh_hist14_mean = hist14.mean(axis=0) @ neighbor_w.T
        neigh_hist28_mean = hist28.mean(axis=0) @ neighbor_w.T

        hist_mask28 = obs_mask[t - 28:t, :]
        hist_mask7 = obs_mask[t - 7:t, :]

        for i in range(n_nodes):
            if obs_mask[t, i] <= 0:
                continue

            sid = station_ids[i]
            sname = station_names[i]
            station_meta = meta.loc[sid]

            row = {
                "date": target_date,
                "split": split,
                "station_id": sid,
                "station_name": sname,
                "daytype": daytype,
                "target_log": float(rides_log[t, i]),
                "target_raw": float(rides_raw[t, i]),
                # calendar/date features
                "is_weekend": float(cal_row["is_weekend"]),
                "post_covid": float(cal_row["post_covid"]),
                "dow_sin": float(cal_row["dow_sin"]),
                "dow_cos": float(cal_row["dow_cos"]),
                "month_sin": float(cal_row["month_sin"]),
                "month_cos": float(cal_row["month_cos"]),
                "doy_sin": float(cal_row["doy_sin"]),
                "doy_cos": float(cal_row["doy_cos"]),
                "trend": float(cal_row["trend"]),
                # static/network features
                "lat_z": float(station_meta["lat_z"]),
                "lon_z": float(station_meta["lon_z"]),
                "degree_z": float(station_meta["degree_z"]),
                "is_transfer": float(station_meta["is_transfer"]),
                # lag demand features (log scale)
                "lag1": float(rides_log[t - 1, i]),
                "lag7": float(rides_log[t - 7, i]),
                "lag14": float(rides_log[t - 14, i]),
                "lag28": float(rides_log[t - 28, i]),
                # rolling features (log scale)
                "roll_mean_7": float(hist7[:, i].mean()),
                "roll_std_7": float(hist7[:, i].std()),
                "roll_mean_14": float(hist14[:, i].mean()),
                "roll_std_14": float(hist14[:, i].std()),
                "roll_mean_28": float(hist28[:, i].mean()),
                "roll_std_28": float(hist28[:, i].std()),
                "roll_min_28": float(hist28[:, i].min()),
                "roll_max_28": float(hist28[:, i].max()),
                # history missingness helpers
                "missing_hist_7": float((1.0 - hist_mask7[:, i]).sum()),
                "missing_hist_28": float((1.0 - hist_mask28[:, i]).sum()),
                # neighbor interaction features
                "nbr_lag1": float(neighbor_lag_base[t - 1, i]),
                "nbr_lag7": float(neighbor_lag_base[t - 7, i]),
                "nbr_lag14": float(neighbor_lag_base[t - 14, i]),
                "nbr_roll_mean_7": float(neigh_hist7_mean[i]),
                "nbr_roll_mean_14": float(neigh_hist14_mean[i]),
                "nbr_roll_mean_28": float(neigh_hist28_mean[i]),
                "nbr_gap_lag1": float(rides_log[t - 1, i] - neighbor_lag_base[t - 1, i]),
                "nbr_ratio_lag1": float(rides_log[t - 1, i] / max(neighbor_lag_base[t - 1, i], 1e-6)),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df["station_id"] = df["station_id"].astype("category")
    df["daytype"] = df["daytype"].astype("category")
    return df


# =========================================================
# 5. Split / train / evaluate
# =========================================================
def make_feature_matrices(df: pd.DataFrame):
    feature_cols = [
        c for c in df.columns
        if c not in {"date", "split", "station_name", "target_log", "target_raw"}
    ]
    X = df[feature_cols].copy()
    y = df["target_log"].astype(np.float32).copy()
    return X, y, feature_cols


def encode_features_for_xgb(X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame):
    """
    Safer than XGBoost native categorical support on some local environments.
    Uses ordinal encoding for string/category columns and float32 arrays for XGBoost.
    """
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = enc.fit_transform(X_train[cat_cols].astype(str))
        X_valid_cat = enc.transform(X_valid[cat_cols].astype(str))
        X_test_cat = enc.transform(X_test[cat_cols].astype(str))
    else:
        enc = None
        X_train_cat = np.empty((len(X_train), 0), dtype=np.float32)
        X_valid_cat = np.empty((len(X_valid), 0), dtype=np.float32)
        X_test_cat = np.empty((len(X_test), 0), dtype=np.float32)

    X_train_num = X_train[num_cols].to_numpy(dtype=np.float32, copy=True)
    X_valid_num = X_valid[num_cols].to_numpy(dtype=np.float32, copy=True)
    X_test_num = X_test[num_cols].to_numpy(dtype=np.float32, copy=True)

    X_train_enc = np.hstack([X_train_num, X_train_cat.astype(np.float32)])
    X_valid_enc = np.hstack([X_valid_num, X_valid_cat.astype(np.float32)])
    X_test_enc = np.hstack([X_test_num, X_test_cat.astype(np.float32)])

    encoded_feature_names = num_cols + cat_cols
    return X_train_enc, X_valid_enc, X_test_enc, encoded_feature_names, cat_cols, enc


def evaluate_predictions(df_pred: pd.DataFrame, label: str) -> Dict[str, float]:
    y_true = df_pred["target_raw"].to_numpy(dtype=np.float64)
    y_pred = df_pred["pred_rides"].to_numpy(dtype=np.float64)

    mae = mean_absolute_error(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    wmape_val = wmape(y_true, y_pred)

    print(f"\n{label} results")
    print(f"MAE   : {mae:,.2f}")
    print(f"RMSE  : {rmse_val:,.2f}")
    print(f"WMAPE : {wmape_val:.4%}")
    return {"MAE": mae, "RMSE": rmse_val, "WMAPE": wmape_val}


# =========================================================
# 6. Main
# =========================================================
def main() -> None:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ridership...")
    ridership_df = load_ridership(RIDERSHIP_CSV, START_DATE)

    print("Building GTFS graph...")
    parent_meta, track_edges, transfer_edges = load_gtfs_graph()
    print(f"Track edges    : {len(track_edges)}")
    print(f"Transfer edges : {len(transfer_edges)}")

    print("Building station-day panel...")
    info = build_station_panel(ridership_df, parent_meta, track_edges, transfer_edges)
    print(f"Num dates  : {len(info['dates'])}")
    print(f"Num nodes  : {info['num_nodes']}")
    print(f"Edge count : {len(info['edge_list'])}")

    print("Building XGBoost feature table...")
    feat_df = build_feature_table(info, SEQ_LEN)
    feat_df.to_parquet(OUTPUT_DIR / "feature_table.parquet", index=False)

    train_df = feat_df[feat_df["split"] == "train"].copy()
    valid_df = feat_df[feat_df["split"] == "valid"].copy()
    test_df = feat_df[feat_df["split"] == "test"].copy()

    print(f"Train samples: {len(train_df):,}")
    print(f"Valid samples: {len(valid_df):,}")
    print(f"Test samples : {len(test_df):,}")

    X_train_df, y_train, feature_cols = make_feature_matrices(train_df)
    X_valid_df, y_valid, _ = make_feature_matrices(valid_df)
    X_test_df, y_test, _ = make_feature_matrices(test_df)

    X_train, X_valid, X_test, encoded_feature_names, cat_cols, encoder = encode_features_for_xgb(
        X_train_df, X_valid_df, X_test_df
    )

    model = xgb.XGBRegressor(**XGB_PARAMS)

    print("Training XGBoost...")
    print(f"Using {X_train.shape[1]} encoded features ({len(cat_cols)} categorical columns ordinal-encoded).")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=50,
    )

    evals_result = model.evals_result()
    eval_rows = []
    for dataset_name, metric_dict in evals_result.items():
        for metric_name, values in metric_dict.items():
            for epoch, val in enumerate(values, start=1):
                eval_rows.append({
                    "dataset": dataset_name,
                    "metric": metric_name,
                    "iteration": epoch,
                    "value": float(val),
                })
    pd.DataFrame(eval_rows).to_csv(OUTPUT_DIR / "training_eval_history.csv", index=False)

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        pred_valid_log = model.predict(X_valid)
        pred_test_log = model.predict(X_test)
    else:
        pred_valid_log = model.predict(X_valid, iteration_range=(0, best_iteration + 1))
        pred_test_log = model.predict(X_test, iteration_range=(0, best_iteration + 1))

    valid_pred = valid_df[["date", "station_id", "station_name", "target_raw"]].copy()
    valid_pred["pred_log"] = pred_valid_log
    valid_pred["pred_rides"] = np.clip(np.expm1(valid_pred["pred_log"].to_numpy()), 0, None)

    test_pred = test_df[["date", "station_id", "station_name", "target_raw"]].copy()
    test_pred["pred_log"] = pred_test_log
    test_pred["pred_rides"] = np.clip(np.expm1(test_pred["pred_log"].to_numpy()), 0, None)

    valid_metrics = evaluate_predictions(valid_pred, label="Validation")
    test_metrics = evaluate_predictions(test_pred, label="Test")

    valid_pred = valid_pred.rename(columns={"target_raw": "actual_rides"})
    test_pred = test_pred.rename(columns={"target_raw": "actual_rides"})
    valid_pred.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    test_pred.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    metrics_df = pd.DataFrame([
        {"split": "validation", **valid_metrics},
        {"split": "test", **test_metrics},
    ])
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    fi_df = pd.DataFrame({
        "feature": encoded_feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    model.save_model(OUTPUT_DIR / "xgboost_model.json")
    joblib.dump({
        "feature_columns": feature_cols,
        "encoded_feature_columns": encoded_feature_names,
        "categorical_columns": cat_cols,
        "ordinal_encoder": encoder,
        "params": XGB_PARAMS,
        "seq_len": SEQ_LEN,
        "manual_station_id_map": MANUAL_STATION_ID_MAP,
        "track_edge_weight": TRACK_EDGE_WEIGHT,
        "transfer_edge_weight": TRANSFER_EDGE_WEIGHT,
        "best_iteration": getattr(model, "best_iteration", None),
    }, OUTPUT_DIR / "metadata.joblib")

    # Save graph tables to align with the existing project outputs
    edge_rows = []
    for i, j, w in info["edge_list"]:
        edge_rows.append({
            "from_station_id": info["station_ids"][i],
            "from_station_name": info["station_names"][i],
            "to_station_id": info["station_ids"][j],
            "to_station_name": info["station_names"][j],
            "weight": w,
        })
    pd.DataFrame(edge_rows).to_csv(OUTPUT_DIR / "graph_edges.csv", index=False)
    pd.DataFrame({
        "station_id": info["station_ids"],
        "station_name": info["station_names"],
    }).to_csv(OUTPUT_DIR / "graph_nodes.csv", index=False)

    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "train_samples": int(len(train_df)),
            "valid_samples": int(len(valid_df)),
            "test_samples": int(len(test_df)),
            "num_nodes": int(info["num_nodes"]),
            "seq_len": int(SEQ_LEN),
            "best_iteration": int(getattr(model, "best_iteration", -1)) if getattr(model, "best_iteration", None) is not None else None,
            "metrics": {
                "validation": valid_metrics,
                "test": test_metrics,
            },
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
