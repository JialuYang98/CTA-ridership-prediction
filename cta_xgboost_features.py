"""Rebuildable original XGBoost feature table; no training or model artifacts."""
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import cta_data
from cta_data import TRACK_EDGE_WEIGHT, TRANSFER_EDGE_WEIGHT


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

def rebuild(output):
    root = Path(__file__).resolve().parent
    for name in ("STOPS_TXT", "ROUTES_TXT", "TRIPS_TXT", "STOP_TIMES_TXT", "TRANSFERS_TXT"):
        setattr(cta_data, name, str(root / getattr(cta_data, name)))
    raw = cta_data.load_ridership(str(root / cta_data.RIDERSHIP_CSV), cta_data.START_DATE)
    meta, tracks, transfers = cta_data.load_gtfs_graph()
    panel = build_station_panel(raw, meta, tracks, transfers)
    table = build_feature_table(panel, 28)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    return table


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    table = rebuild(args.output)
    print(f"Wrote {len(table)} rows to {args.output}")

