
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset


# =========================================================
# Configuration
# =========================================================
RIDERSHIP_CSV = "CTA_-_Ridership_-__L__Station_Entries_-_Daily_Totals.csv"
STOPS_TXT = "google_transit/stops.txt"
ROUTES_TXT = "google_transit/routes.txt"
TRIPS_TXT = "google_transit/trips.txt"
STOP_TIMES_TXT = "google_transit/stop_times.txt"
TRANSFERS_TXT = "google_transit/transfers.txt"

OUTPUT_DIR = Path("cta_gcn_lstm_outputs")

START_DATE = "2018-01-01"
SEQ_LEN = 28
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 15
SEED = 42

TRACK_EDGE_WEIGHT = 1.0
TRANSFER_EDGE_WEIGHT = 0.5

# A few ridership station IDs do not appear directly in current GTFS parent stations.
# These aliases make the prototype runnable. You can revise them later if you want.
MANUAL_STATION_ID_MAP = {
    "40200": "41700",  # Randolph/Wabash -> Washington/Wabash
    "40640": "41700",  # Madison/Wabash -> Washington/Wabash
    "40500": "40370",  # Washington/State -> Washington
    "40260": "41660",  # State/Lake -> Lake (GTFS name mismatch; prototype assumption)
}
DROP_STATION_IDS = {"41580"}  # Homan, obsolete and no GTFS parent station


# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def wmape(y_true, y_pred):
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


# =========================================================
# 3. Build station-day panel aligned to graph nodes
# =========================================================
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


# =========================================================
# 4. Sequence dataset
# =========================================================
class GraphSequenceDataset(Dataset):
    def __init__(self, X, daytype_idx, y_scaled, y_raw, obs_mask, dates, seq_len, split_name):
        self.X_seq = []
        self.daytype_target = []
        self.y_scaled = []
        self.y_raw = []
        self.mask = []
        self.target_dates = []

        for t in range(seq_len, len(dates)):
            target_date = dates[t]
            use = False
            if split_name == "train":
                use = target_date < pd.Timestamp("2024-01-01")
            elif split_name == "valid":
                use = pd.Timestamp("2024-01-01") <= target_date < pd.Timestamp("2025-01-01")
            elif split_name == "test":
                use = target_date >= pd.Timestamp("2025-01-01")
            else:
                raise ValueError(f"Unknown split_name={split_name}")

            if use:
                self.X_seq.append(X[t - seq_len:t])
                self.daytype_target.append(daytype_idx[t])
                self.y_scaled.append(y_scaled[t])
                self.y_raw.append(y_raw[t])
                self.mask.append(obs_mask[t])
                self.target_dates.append(target_date)

        self.X_seq = np.asarray(self.X_seq, dtype=np.float32)
        self.daytype_target = np.asarray(self.daytype_target, dtype=np.int64)
        self.y_scaled = np.asarray(self.y_scaled, dtype=np.float32)
        self.y_raw = np.asarray(self.y_raw, dtype=np.float32)
        self.mask = np.asarray(self.mask, dtype=np.float32)

    def __len__(self):
        return len(self.y_scaled)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_seq[idx], dtype=torch.float32),    # [S, N, F]
            torch.tensor(self.daytype_target[idx], dtype=torch.long),
            torch.tensor(self.y_scaled[idx], dtype=torch.float32), # [N]
            torch.tensor(self.y_raw[idx], dtype=torch.float32),    # [N]
            torch.tensor(self.mask[idx], dtype=torch.float32),     # [N]
        )


# =========================================================
# 5. Scaling
# =========================================================
def fit_ride_scaler(y_log: np.ndarray, obs_mask: np.ndarray, dates: pd.DatetimeIndex):
    train_mask = dates < pd.Timestamp("2024-01-01")
    train_values = y_log[train_mask][obs_mask[train_mask] > 0]
    mean = float(train_values.mean())
    std = float(train_values.std() + 1e-6)
    return {"ride_mean": mean, "ride_std": std}


def apply_ride_scaler(X: np.ndarray, y_log: np.ndarray, scaler: Dict[str, float]):
    X_scaled = X.copy()
    X_scaled[:, :, 0] = (X[:, :, 0] - scaler["ride_mean"]) / scaler["ride_std"]
    y_scaled = (y_log - scaler["ride_mean"]) / scaler["ride_std"]
    return X_scaled.astype(np.float32), y_scaled.astype(np.float32)


# =========================================================
# 6. GCN + LSTM model
# =========================================================
class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], adj: [N, N]
        h = self.linear(x)
        return torch.einsum("ij,bjf->bif", adj, h)


class GCNLSTM(nn.Module):
    def __init__(self, num_nodes: int, num_features: int, num_daytypes: int):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, 8)
        self.daytype_emb = nn.Embedding(num_daytypes, 2)

        gcn_in = num_features + 8
        self.gcn1 = GraphConv(gcn_in, 32)
        self.gcn2 = GraphConv(32, 32)
        self.dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.head = nn.Sequential(
            nn.Linear(64 + 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq: torch.Tensor, daytype_idx: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, S, N, F]
        B, S, N, F = x_seq.shape

        node_ids = torch.arange(N, device=x_seq.device)
        node_emb = self.node_emb(node_ids)                       # [N, 8]
        node_emb = node_emb.unsqueeze(0).unsqueeze(0).expand(B, S, N, -1)

        x = torch.cat([x_seq, node_emb], dim=-1)                # [B, S, N, F+8]
        x = x.reshape(B * S, N, F + 8)

        h = torch.relu(self.gcn1(x, adj))
        h = self.dropout(h)
        h = torch.relu(self.gcn2(h, adj))
        h = self.dropout(h)

        h = h.reshape(B, S, N, 32)
        h = h.permute(0, 2, 1, 3).reshape(B * N, S, 32)         # [B*N, S, 32]

        _, (h_n, _) = self.lstm(h)
        h_last = h_n[-1]                                        # [B*N, 64]
        h_last = h_last.reshape(B, N, 64)

        day_emb = self.daytype_emb(daytype_idx)                 # [B, 2]
        day_emb = day_emb.unsqueeze(1).expand(B, N, 2)

        out = self.head(torch.cat([h_last, day_emb], dim=-1)).squeeze(-1)  # [B, N]
        return out


# =========================================================
# 7. Training and evaluation
# =========================================================
def masked_mse_loss(pred, target, mask):
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return diff2.sum() / denom


def train_one_epoch(model, loader, optimizer, adj, device):
    model.train()
    total_loss = 0.0
    total_weight = 0.0

    for x_seq, daytype_idx, y_scaled, _, mask in loader:
        x_seq = x_seq.to(device)
        daytype_idx = daytype_idx.to(device)
        y_scaled = y_scaled.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        pred = model(x_seq, daytype_idx, adj)
        loss = masked_mse_loss(pred, y_scaled, mask)
        loss.backward()
        optimizer.step()

        batch_weight = float(mask.sum().item())
        total_loss += float(loss.item()) * batch_weight
        total_weight += batch_weight

    return total_loss / max(total_weight, 1.0)


@torch.no_grad()
def predict_raw(model, loader, adj, scaler, device):
    model.eval()
    preds_raw = []
    truths_raw = []
    masks = []

    for x_seq, daytype_idx, _, y_raw, mask in loader:
        x_seq = x_seq.to(device)
        daytype_idx = daytype_idx.to(device)

        pred_scaled = model(x_seq, daytype_idx, adj).cpu().numpy()
        pred_log = pred_scaled * scaler["ride_std"] + scaler["ride_mean"]
        pred_raw = np.expm1(pred_log)
        pred_raw = np.clip(pred_raw, 0, None)

        preds_raw.append(pred_raw)
        truths_raw.append(y_raw.numpy())
        masks.append(mask.numpy())

    return (
        np.concatenate(preds_raw, axis=0),
        np.concatenate(truths_raw, axis=0),
        np.concatenate(masks, axis=0),
    )


def evaluate_model(model, loader, adj, scaler, device, label="set"):
    pred_raw, true_raw, mask = predict_raw(model, loader, adj, scaler, device)

    pred_vec = pred_raw[mask > 0]
    true_vec = true_raw[mask > 0]

    mae = mean_absolute_error(true_vec, pred_vec)
    rmse_val = rmse(true_vec, pred_vec)
    wmape_val = wmape(true_vec, pred_vec)

    print(f"\n{label} results")
    print(f"MAE   : {mae:,.2f}")
    print(f"RMSE  : {rmse_val:,.2f}")
    print(f"WMAPE : {wmape_val:.4%}")

    return {"MAE": mae, "RMSE": rmse_val, "WMAPE": wmape_val}, pred_raw, true_raw, mask


def train_model(model, train_loader, valid_loader, adj, scaler, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_valid = float("inf")
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, adj, device)
        valid_metrics, _, _, _ = evaluate_model(model, valid_loader, adj, scaler, device, label=f"Validation @ epoch {epoch}")
        valid_score = valid_metrics["WMAPE"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_wmape": valid_score,
        })
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | valid_wmape={valid_score:.4%}")

        if valid_score < best_valid:
            best_valid = valid_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    return model, history_df


# =========================================================
# 8. Save predictions
# =========================================================
def save_prediction_table(dataset: GraphSequenceDataset, pred_raw: np.ndarray, info: Dict, out_csv: Path):
    rows = []
    node_ids = info["station_ids"]
    node_names = info["station_names"]

    for sample_idx, target_date in enumerate(dataset.target_dates):
        for node_idx, station_id in enumerate(node_ids):
            rows.append({
                "date": pd.Timestamp(target_date).strftime("%Y-%m-%d"),
                "station_id": station_id,
                "station_name": node_names[node_idx],
                "actual_rides": float(dataset.y_raw[sample_idx, node_idx]),
                "pred_rides": float(pred_raw[sample_idx, node_idx]),
                "observed_mask": int(dataset.mask[sample_idx, node_idx]),
            })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_csv, index=False)
    return pred_df


# =========================================================
# 9. Main
# =========================================================
def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading ridership...")
    ridership_df = load_ridership(RIDERSHIP_CSV, START_DATE)

    print("Building GTFS graph...")
    parent_meta, track_edges, transfer_edges = load_gtfs_graph()
    print(f"Track edges    : {len(track_edges)}")
    print(f"Transfer edges : {len(transfer_edges)}")

    print("Building station panel...")
    X, y_log, obs_mask, daytype_idx, edge_list, info = build_station_panel(
        ridership_df, parent_meta, track_edges, transfer_edges
    )
    dates = pd.DatetimeIndex(info["dates"])

    print(f"Num dates  : {len(dates)}")
    print(f"Num nodes  : {info['num_nodes']}")
    print(f"Num feats  : {info['num_features']}")
    print(f"Edge count : {len(edge_list)}")

    scaler = fit_ride_scaler(y_log, obs_mask, dates)
    X_scaled, y_scaled = apply_ride_scaler(X, y_log, scaler)

    train_ds = GraphSequenceDataset(X_scaled, daytype_idx, y_scaled, np.expm1(y_log), obs_mask, dates, SEQ_LEN, "train")
    valid_ds = GraphSequenceDataset(X_scaled, daytype_idx, y_scaled, np.expm1(y_log), obs_mask, dates, SEQ_LEN, "valid")
    test_ds = GraphSequenceDataset(X_scaled, daytype_idx, y_scaled, np.expm1(y_log), obs_mask, dates, SEQ_LEN, "test")

    print(f"Train samples: {len(train_ds)}")
    print(f"Valid samples: {len(valid_ds)}")
    print(f"Test samples : {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    adj = build_normalized_adjacency(info["num_nodes"], edge_list).to(device)

    model = GCNLSTM(
        num_nodes=info["num_nodes"],
        num_features=info["num_features"],
        num_daytypes=len(info["daytype_to_idx"]),
    ).to(device)

    print("Training GCN + LSTM...")
    model, history_df = train_model(model, train_loader, valid_loader, adj, scaler, device)

    print("\nEvaluating best model...")
    valid_metrics, valid_pred_raw, _, _ = evaluate_model(model, valid_loader, adj, scaler, device, label="Validation")
    test_metrics, test_pred_raw, _, _ = evaluate_model(model, test_loader, adj, scaler, device, label="Test")

    save_prediction_table(valid_ds, valid_pred_raw, info, OUTPUT_DIR / "validation_predictions.csv")
    save_prediction_table(test_ds, test_pred_raw, info, OUTPUT_DIR / "test_predictions.csv")

    # Graph edge table
    edge_rows = []
    for i, j, w in edge_list:
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

    pd.DataFrame([
        {"split": "validation", **valid_metrics},
        {"split": "test", **test_metrics},
    ]).to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "info": info,
            "seq_len": SEQ_LEN,
            "manual_station_id_map": MANUAL_STATION_ID_MAP,
        },
        OUTPUT_DIR / "gcn_lstm_model.pt",
    )
    joblib.dump(
        {
            "info": info,
            "scaler": scaler,
            "manual_station_id_map": MANUAL_STATION_ID_MAP,
        },
        OUTPUT_DIR / "metadata.joblib",
    )

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
