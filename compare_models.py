"""Recompute the retained three models' metrics on exactly the same targets."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FAMILIES = {"xgboost": "XGBoost", "lstm": "Residual LSTM", "gcn_lstm": "GCN-LSTM"}


def main():
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    rows = []
    for split in ("validation", "test"):
        reference = None
        for family, name in FAMILIES.items():
            frame = pd.read_csv(ROOT / "models" / family / f"{split}_predictions.csv", dtype={"station_id": str})
            if "observed_mask" in frame:
                frame = frame[frame.observed_mask > 0]
            if frame.duplicated(["date", "station_id"]).any():
                raise ValueError(f"Duplicate keys: {family}/{split}")
            frame = frame.set_index(["date", "station_id"]).sort_index()
            if reference is None:
                reference = frame.actual_rides.astype(np.float64)
            elif not reference.index.equals(frame.index) or not np.allclose(reference, frame.actual_rides, atol=.1, rtol=0):
                raise ValueError(f"Mismatched targets: {family}/{split}")
            prediction = frame.pred_rides.to_numpy(np.float64)
            if not np.isfinite(prediction).all() or np.any(prediction < 0):
                raise ValueError("Invalid predictions")
            error = prediction - reference.to_numpy()
            rows.append({"family": family, "model": name, "split": split, "samples": len(frame),
                         "MAE": np.abs(error).mean(), "RMSE": np.sqrt(np.mean(error**2)),
                         "WMAPE": np.abs(error).sum()/reference.sum(), "bias": error.sum()/reference.sum()})
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "best_model_metrics.csv", index=False)
    registry = {"selection_metric": "2024 validation raw-count WMAPE", "test_used_for_selection": False,
                "test_period": "2025-01-01 through 2025-11-30", "models": {}}
    for family in FAMILIES:
        registry["models"][family] = {
            "checkpoint": f"models/{family}/" + ("model.ubj" if family == "xgboost" else "model.pt"),
            "implementation": "cta_xgboost_train.py" if family == "xgboost" else "cta_neural_forecast.py",
            "graph_mode": None if family == "xgboost" else ("identity" if family == "lstm" else "physical"),
            "metrics": scores[scores.family == family].drop(columns=["family", "model"]).to_dict("records"),
        }
    (ROOT / "models/best_models.json").write_text(json.dumps(registry, indent=2))
    print(scores.to_string(index=False))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cta-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    test = scores[scores.split == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, scale in zip(axes, ["WMAPE", "RMSE"], [100, 1]):
        bars = ax.bar(test.model, test[col]*scale, color=["#2d86b5", "#67a688", "#dfa341"])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set(title=f"Test {col}", ylabel="Percent" if col == "WMAPE" else "Rides")
        ax.set_ylim(0, (test[col]*scale).max()*1.2)
    fig.suptitle("Best retained model per family — selected on validation WMAPE")
    fig.tight_layout()
    fig.savefig(out / "best_models.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
