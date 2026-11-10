"""Residual temporal models: physical graph for GCN-LSTM, identity for LSTM.

Retains the original graph, station mapping, dates, and targets. Writes to a new
directory. Train first; --evaluate-test loads the frozen checkpoint separately.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import cta_data as original

ROOT = Path(__file__).resolve().parent
SEQ_LEN = 28


def prepare_data():
    for name in ("STOPS_TXT", "ROUTES_TXT", "TRIPS_TXT", "STOP_TIMES_TXT", "TRANSFERS_TXT"):
        setattr(original, name, str(ROOT / getattr(original, name)))
    raw = original.load_ridership(str(ROOT / original.RIDERSHIP_CSV), original.START_DATE)
    meta, tracks, transfers = original.load_gtfs_graph()
    x, y, mask, daytype, edges, info = original.build_station_panel(raw, meta, tracks, transfers)
    dates = pd.DatetimeIndex(info["dates"])
    scaler = original.fit_ride_scaler(y, mask, dates)
    train = dates < "2024-01-01"
    scaler["raw_mean"] = float(np.expm1(y[train][mask[train] > 0]).mean())
    # A fixed time reference, independent of validation/test date range.
    x[:, :, 9] = np.asarray((dates - pd.Timestamp("2018-01-01")).days)[:, None] / 365.25
    x[:, :, 0] = (y - scaler["ride_mean"]) / scaler["ride_std"]
    x[:, :, 0] = np.where(mask > 0, x[:, :, 0], 0.0)
    # Mask is an explicit history feature; imputed values are neutral on scaled log space.
    x = np.concatenate([x, mask[..., None]], axis=-1)
    calendar = np.concatenate([
        x[:, 0, 1:10], np.eye(len(info["daytype_to_idx"]), dtype=np.float32)[daytype]
    ], axis=1)
    adj = original.build_normalized_adjacency(len(info["station_ids"]), edges)
    return {
        "x": torch.from_numpy(x), "calendar": torch.from_numpy(calendar),
        "y": torch.from_numpy(y), "mask": torch.from_numpy(mask),
        "dates": dates, "info": info, "scaler": scaler, "adj": adj,
    }


class Windows(Dataset):
    def __init__(self, data, split):
        self.data = data
        dates = data["dates"]
        select = {
            "train": dates < "2024-01-01",
            "validation": (dates >= "2024-01-01") & (dates < "2025-01-01"),
            "test": dates >= "2025-01-01",
        }[split]
        self.indices = np.flatnonzero(select & (np.arange(len(dates)) >= SEQ_LEN))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        t = self.indices[index]
        d = self.data
        return d["x"][t-SEQ_LEN:t], d["calendar"][t], d["y"][t], d["mask"][t]


class ResidualGCNLSTM(nn.Module):
    def __init__(self, n_nodes, n_features, n_calendar, adjacency, hidden=32):
        super().__init__()
        self.register_buffer("adjacency", adjacency.clone())
        self.node_embedding = nn.Embedding(n_nodes, 8)
        # Preserve an own-station path; graph aggregation cannot overwrite it.
        self.local = nn.Linear(n_features, 16)
        self.graph = nn.Linear(n_features, 16)
        self.graph_gate = nn.Parameter(torch.tensor(-1.0))
        self.lstm = nn.LSTM(32, hidden, batch_first=True)
        # Four explicit seasonal lags, observation flags, and masked mean/std.
        self.head = nn.Sequential(
            nn.Linear(hidden + n_calendar + 8 + 10, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history, calendar):
        b, s, n, _ = history.shape
        neighbors = torch.einsum("ij,bsjf->bsif", self.adjacency, history)
        local = torch.relu(self.local(history))
        spatial = torch.sigmoid(self.graph_gate) * torch.relu(self.graph(neighbors))
        sequence = torch.cat([local, spatial], dim=-1)
        sequence = sequence.permute(0, 2, 1, 3).reshape(b*n, s, 32)
        _, (hidden, _) = self.lstm(sequence)
        hidden = hidden[-1].reshape(b, n, -1)
        values, observed = history[..., 0], history[..., -1]
        count = observed.sum(dim=1).clamp_min(1)
        mean = (values * observed).sum(dim=1) / count
        var = ((values - mean[:, None])**2 * observed).sum(dim=1) / count
        lag_indices = [-1, -7, -14, -28]
        lags = values[:, lag_indices].transpose(1, 2)
        lag_masks = observed[:, lag_indices].transpose(1, 2)
        summaries = torch.cat([lags, lag_masks, mean[..., None], var.sqrt()[..., None]], dim=-1)
        # Fall back to an observed history mean if the seasonal reference is missing.
        seasonal = torch.where(lag_masks[..., 1] > 0, lags[..., 1], mean)
        node = self.node_embedding.weight[None].expand(b, -1, -1)
        target_calendar = calendar[:, None].expand(-1, n, -1)
        correction = self.head(torch.cat([hidden, summaries, node, target_calendar], dim=-1)).squeeze(-1)
        return seasonal + correction


def raw_prediction(scaled, scaler):
    log = scaled * scaler["ride_std"] + scaler["ride_mean"]
    # Numerical guard for catastrophic excursions; normal forecasts are far below exp(14).
    return torch.expm1(log.clamp(0, 14))


def objective(scaled, target_log, mask, scaler):
    raw = raw_prediction(scaled, scaler)
    target = target_log.expm1()
    raw_mae = (raw - target).abs() / scaler["raw_mean"]
    target_scaled = (target_log - scaler["ride_mean"]) / scaler["ride_std"]
    log_aux = nn.functional.smooth_l1_loss(scaled, target_scaled, reduction="none")
    return ((raw_mae + 0.05 * log_aux) * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, loader, scaler):
    model.eval()
    predictions, truths, masks = [], [], []
    for history, calendar, target, mask in loader:
        predictions.append(raw_prediction(model(history, calendar), scaler).numpy())
        truths.append(target.expm1().numpy())
        masks.append(mask.numpy())
    prediction, truth, mask = map(np.concatenate, (predictions, truths, masks))
    error = (prediction.astype(np.float64) - truth)[mask > 0]
    actual = truth[mask > 0].astype(np.float64)
    metrics = {
        "MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.mean(error**2))),
        "WMAPE": float(np.abs(error).sum()/actual.sum()),
        "bias": float(error.sum()/actual.sum()), "samples": int(len(actual)),
    }
    return metrics, prediction


def save_predictions(dataset, predictions, path):
    data = dataset.data
    indices = dataset.indices
    n = len(data["info"]["station_ids"])
    pd.DataFrame({
        "date": np.repeat(data["dates"][indices].strftime("%Y-%m-%d"), n),
        "station_id": np.tile(data["info"]["station_ids"], len(indices)),
        "station_name": np.tile(data["info"]["station_names"], len(indices)),
        "actual_rides": data["y"][indices].expm1().numpy().ravel(),
        "pred_rides": predictions.ravel(),
        "observed_mask": data["mask"][indices].numpy().ravel().astype(int),
    }).to_csv(path, index=False)


def make_model(data, graph):
    adj = data["adj"] if graph == "physical" else torch.eye(len(data["info"]["station_ids"]))
    return ResidualGCNLSTM(len(adj), data["x"].shape[-1], data["calendar"].shape[-1], adj)


def fingerprints():
    paths = [Path(__file__), Path(original.__file__), ROOT / original.RIDERSHIP_CSV]
    paths += [Path(getattr(original, name)) for name in (
        "STOPS_TXT", "ROUTES_TXT", "TRIPS_TXT", "STOP_TIMES_TXT", "TRANSFERS_TXT")]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/gcn_lstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph", choices=["physical", "identity"], default="physical")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    original.set_seed(args.seed)
    checkpoint_path = args.output / "model.pt"
    if checkpoint_path.exists() and not args.evaluate_test:
        parser.error("Output already has a model. Use a different --output to preserve experiments.")
    print("Preparing original station/date panel...", flush=True)
    data = prepare_data()
    signature = fingerprints()
    if args.evaluate_test:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["fingerprints"] != signature:
            raise ValueError("Sources or input data changed since training; evaluation aborted.")
        if checkpoint["station_ids"] != data["info"]["station_ids"]:
            raise ValueError("Station order changed.")
        model = make_model(data, checkpoint["config"]["graph"])
        model.load_state_dict(checkpoint["model_state_dict"])
        dataset = Windows(data, "test")
        metrics, predictions = evaluate(model, DataLoader(dataset, batch_size=args.batch_size), checkpoint["scaler"])
        save_predictions(dataset, predictions, args.output / "test_predictions.csv")
        (args.output / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        print("TEST", json.dumps(metrics), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    train, valid = Windows(data, "train"), Windows(data, "validation")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=args.batch_size)
    model = make_model(data, args.graph)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    initial, _ = evaluate(model, valid_loader, data["scaler"])
    print("Seasonal initial validation", json.dumps(initial), flush=True)
    best, stale, history = float("inf"), 0, []
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, weight = 0., 0.
        for x, cal, y, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x, cal), y, mask, data["scaler"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * mask.sum().item()
            weight += mask.sum().item()
        metrics, predictions = evaluate(model, valid_loader, data["scaler"])
        scheduler.step(metrics["WMAPE"])
        history.append({"epoch": epoch, "train_loss": total/weight, **metrics,
                        "elapsed_seconds": time.monotonic()-start, "lr": optimizer.param_groups[0]["lr"]})
        pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
        print(f"epoch {epoch:02d}: validation WMAPE={metrics['WMAPE']:.4%}, MAE={metrics['MAE']:.2f}, elapsed={time.monotonic()-start:.1f}s", flush=True)
        if metrics["WMAPE"] < best:
            best, stale = metrics["WMAPE"], 0
            config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            torch.save({"model_state_dict": model.state_dict(), "scaler": data["scaler"],
                        "station_ids": data["info"]["station_ids"], "config": config,
                        "fingerprints": signature, "epoch": epoch,
                        "torch_version": str(torch.__version__), "validation": metrics}, checkpoint_path)
            save_predictions(valid, predictions, args.output / "validation_predictions.csv")
            (args.output / "validation_metrics.json").write_text(json.dumps({"epoch": epoch, **metrics}, indent=2))
        else:
            stale += 1
            if stale >= args.patience:
                break
    print(f"Best validation WMAPE: {best:.4%}. Test not evaluated.", flush=True)


if __name__ == "__main__":
    main()
