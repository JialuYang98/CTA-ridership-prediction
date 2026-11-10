"""Shared CTA preprocessing, extracted unchanged from the original experiment."""
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch


RIDERSHIP_CSV = "CTA_-_Ridership_-__L__Station_Entries_-_Daily_Totals.csv"

STOPS_TXT = "google_transit/stops.txt"

ROUTES_TXT = "google_transit/routes.txt"

TRIPS_TXT = "google_transit/trips.txt"

STOP_TIMES_TXT = "google_transit/stop_times.txt"

TRANSFERS_TXT = "google_transit/transfers.txt"

START_DATE = "2018-01-01"

TRACK_EDGE_WEIGHT = 1.0

TRANSFER_EDGE_WEIGHT = 0.5

MANUAL_STATION_ID_MAP = {
    "40200": "41700",  # Randolph/Wabash -> Washington/Wabash
    "40640": "41700",  # Madison/Wabash -> Washington/Wabash
    "40500": "40370",  # Washington/State -> Washington
    "40260": "41660",  # State/Lake -> Lake (GTFS name mismatch; prototype assumption)
}

DROP_STATION_IDS = {"41580"}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def build_normalized_adjacency(num_nodes: int, edge_list: List[Tuple[int, int, float]]) -> torch.Tensor:
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j, w in edge_list:
        A[i, j] += w
        A[j, i] += w

    # self-loops
    A += np.eye(num_nodes, dtype=np.float32)

    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_hat, dtype=torch.float32)

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

    # Rail routes
    rail_routes = set(routes.loc[routes["route_type"] == 1, "route_id"].astype(str))
    rail_trips = trips[trips["route_id"].isin(rail_routes)][["trip_id", "route_id"]].copy()

    # Map every stop/platform to a parent station node
    stops["node_station_id"] = np.where(
        stops["location_type"] == 1,
        stops["stop_id"],
        stops["parent_station"],
    )
    stops["node_station_id"] = stops["node_station_id"].map(clean_id)

    stop_to_parent = dict(zip(stops["stop_id"], stops["node_station_id"]))

    # Parent-station metadata
    parent_rows = stops[stops["location_type"] == 1].copy()
    parent_meta = parent_rows[["stop_id", "stop_name", "stop_lat", "stop_lon"]].rename(
        columns={"stop_id": "station_id", "stop_name": "station_name"}
    )

    # In case some parent node exists only via platform references, backfill from platform coords
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

    # Track edges from consecutive parent stations along rail trips
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

    # Transfer edges from transfers.txt, mapped from platform stops to parent stations
    transfer_edges = set()
    for _, row in transfers.iterrows():
        a = stop_to_parent.get(row["from_stop_id"], "")
        b = stop_to_parent.get(row["to_stop_id"], "")
        a = clean_id(a)
        b = clean_id(b)
        if a and b and a != b:
            transfer_edges.add(tuple(sorted((a, b))))

    return parent_meta, track_edges, transfer_edges

def build_station_panel(ridership_df: pd.DataFrame, parent_meta: pd.DataFrame,
                        track_edges, transfer_edges):
    graph_nodes = set(parent_meta["station_id"].astype(str))
    ridership_nodes = set(ridership_df["station_id"].astype(str))
    node_ids = sorted(graph_nodes & ridership_nodes)

    meta = parent_meta[parent_meta["station_id"].isin(node_ids)].copy().reset_index(drop=True)

    # Date range
    all_dates = pd.date_range(ridership_df["date"].min(), ridership_df["date"].max(), freq="D")

    # Station x date matrix
    rides_pivot = ridership_df.pivot_table(index="date", columns="station_id", values="rides", aggfunc="sum")
    rides_pivot = rides_pivot.reindex(index=all_dates, columns=node_ids)

    obs_mask = (~rides_pivot.isna()).astype(np.float32).values
    rides_raw = rides_pivot.fillna(0.0).values.astype(np.float32)
    rides_log = np.log1p(rides_raw)

    # daytype is date-level
    daytype_df = ridership_df.groupby("date", as_index=False)["daytype"].agg(lambda x: x.mode().iloc[0])
    daytype_series = daytype_df.set_index("date")["daytype"].reindex(all_dates)
    daytype_series = daytype_series.fillna("W")
    daytype_vocab = sorted(daytype_series.unique().tolist())
    daytype_to_idx = {d: i for i, d in enumerate(daytype_vocab)}
    daytype_idx = daytype_series.map(daytype_to_idx).astype(int).values

    # Calendar features (date-level)
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

    date_features = cal[[
        "is_weekend", "post_covid",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        "doy_sin", "doy_cos",
        "trend"
    ]].values.astype(np.float32)

    # Static node features
    meta = meta.sort_values("station_id").reset_index(drop=True)
    lat = meta["stop_lat"].values.astype(np.float32)
    lon = meta["stop_lon"].values.astype(np.float32)

    id_to_idx = {sid: i for i, sid in enumerate(meta["station_id"].astype(str))}
    degree = np.zeros(len(meta), dtype=np.float32)
    is_transfer = np.zeros(len(meta), dtype=np.float32)

    for a, b in track_edges:
        if a in id_to_idx and b in id_to_idx:
            degree[id_to_idx[a]] += 1
            degree[id_to_idx[b]] += 1
    for a, b in transfer_edges:
        if a in id_to_idx and b in id_to_idx:
            is_transfer[id_to_idx[a]] = 1.0
            is_transfer[id_to_idx[b]] = 1.0

    def zscore(v):
        mu = float(v.mean())
        sd = float(v.std() + 1e-6)
        return (v - mu) / sd

    node_static = np.stack([
        zscore(lat),
        zscore(lon),
        zscore(degree),
        is_transfer
    ], axis=1).astype(np.float32)

    # Build feature tensor [T, N, F]
    T = len(all_dates)
    N = len(meta)

    date_feat_3d = np.repeat(date_features[:, None, :], N, axis=1)
    static_feat_3d = np.repeat(node_static[None, :, :], T, axis=0)
    rides_feat_3d = rides_log[:, :, None]

    X = np.concatenate([rides_feat_3d, date_feat_3d, static_feat_3d], axis=2).astype(np.float32)
    y = rides_log.astype(np.float32)

    # Graph edge list for model adjacency
    edge_list = []
    for a, b in track_edges:
        if a in id_to_idx and b in id_to_idx:
            edge_list.append((id_to_idx[a], id_to_idx[b], TRACK_EDGE_WEIGHT))
    for a, b in transfer_edges:
        if a in id_to_idx and b in id_to_idx:
            edge_list.append((id_to_idx[a], id_to_idx[b], TRANSFER_EDGE_WEIGHT))

    info = {
        "dates": all_dates,
        "station_ids": meta["station_id"].astype(str).tolist(),
        "station_names": meta["station_name"].tolist(),
        "daytype_to_idx": daytype_to_idx,
        "num_features": X.shape[2],
        "num_nodes": N,
    }

    return X, y, obs_mask, daytype_idx, edge_list, info

def fit_ride_scaler(y_log: np.ndarray, obs_mask: np.ndarray, dates: pd.DatetimeIndex):
    train_mask = dates < pd.Timestamp("2024-01-01")
    train_values = y_log[train_mask][obs_mask[train_mask] > 0]
    mean = float(train_values.mean())
    std = float(train_values.std() + 1e-6)
    return {"ride_mean": mean, "ride_std": std}
