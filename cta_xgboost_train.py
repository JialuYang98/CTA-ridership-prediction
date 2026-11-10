"""Validation-selected XGBoost search; test evaluation is a separate command."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data/cta_features.parquet"
EXCLUDED = {"date", "split", "station_name", "target_log", "target_raw"}
TRIALS = [
    {"name": "log_squared_d6", "scale": "log", "objective": "reg:squarederror", "max_depth": 6, "seasonal": False},
    {"name": "log_seasonal_d6", "scale": "log", "objective": "reg:squarederror", "max_depth": 6, "seasonal": True},
    {"name": "raw_absolute_d6", "scale": "raw", "objective": "reg:absoluteerror", "max_depth": 6, "seasonal": True},
    {"name": "raw_absolute_d8", "scale": "raw", "objective": "reg:absoluteerror", "max_depth": 8, "seasonal": True},
    {"name": "raw_squared_d6", "scale": "raw", "objective": "reg:squarederror", "max_depth": 6, "seasonal": True},
]


def enrich(frame):
    """Only fixed calendar information and existing strictly historical features."""
    frame = frame.copy()
    dates = pd.to_datetime(frame.date)
    frame["trend"] = (dates - pd.Timestamp("2018-01-01")).dt.days / 365.25
    frame["dow"] = dates.dt.dayofweek
    frame["day_of_month"] = dates.dt.day
    frame["month"] = dates.dt.month
    frame["year"] = dates.dt.year
    frame["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    holidays = USFederalHolidayCalendar().holidays("2017-01-01", "2040-12-31")
    for offset in (-1, 0, 1):
        frame[f"federal_holiday_offset_{offset}"] = (dates + pd.Timedelta(days=offset)).isin(holidays).astype(int)
    frame["christmas"] = ((dates.dt.month == 12) & (dates.dt.day == 25)).astype(int)
    frame["new_year"] = ((dates.dt.month == 1) & (dates.dt.day == 1)).astype(int)
    frame["thanksgiving"] = ((dates.dt.month == 11) & (dates.dt.dayofweek == 3) & dates.dt.day.between(22, 28)).astype(int)
    for lag in (1, 7, 14, 28):
        frame[f"raw_lag{lag}"] = np.expm1(frame[f"lag{lag}"])
    frame["weekly_raw_mean"] = frame[["raw_lag7", "raw_lag14", "raw_lag28"]].mean(axis=1)
    frame["weekly_log_change"] = frame.lag7 - frame.lag14
    frame["recent_log_change"] = frame.lag1 - frame.lag7
    return frame


def encode(frame, metadata=None):
    if metadata is None:
        cat = ["station_id", "daytype"]
        num = [c for c in frame if c not in EXCLUDED and c not in cat]
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(frame[cat].astype(str))
        metadata = {"categorical": cat, "numeric": num, "encoder": encoder, "feature_names": num + cat}
    numeric = frame[metadata["numeric"]].to_numpy(np.float32)
    categorical = metadata["encoder"].transform(frame[metadata["categorical"]].astype(str)).astype(np.float32)
    return np.column_stack([numeric, categorical]), metadata


def matrix(values, frame, metadata, trial, labels=True):
    target = frame.target_log if trial["scale"] == "log" else frame.target_raw
    margin = None
    if trial["seasonal"]:
        margin = frame.lag7.to_numpy() if trial["scale"] == "log" else np.expm1(frame.lag7.to_numpy())
    return xgb.DMatrix(values, label=target if labels else None, base_margin=margin,
                       feature_names=metadata["feature_names"], nthread=2)


def inverse(prediction, scale):
    if scale == "log":
        prediction = np.expm1(np.clip(prediction, 0, 14))
    return np.maximum(prediction, 0)


def metrics(actual, prediction):
    error = np.asarray(prediction, np.float64) - np.asarray(actual, np.float64)
    return {"MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.mean(error**2))),
            "WMAPE": float(np.abs(error).sum()/np.sum(actual)), "bias": float(error.sum()/np.sum(actual)),
            "samples": len(error)}


def export_predictions(frame, prediction, path):
    result = frame[["date", "station_id", "station_name", "target_raw"]].rename(columns={"target_raw": "actual_rides"}).copy()
    result["pred_rides"] = prediction
    result.to_csv(path, index=False)


def fingerprint():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), CACHE)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/xgboost_search")
    parser.add_argument("--rounds", type=int, default=1800)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    if args.rebuild_features or not CACHE.exists():
        # Keep the torch-based shared data module out of the XGBoost process:
        # the installed native runtimes can conflict when finalized together.
        subprocess.run([sys.executable, str(ROOT / "cta_xgboost_features.py"), str(CACHE)], check=True)
    frame = enrich(pd.read_parquet(CACHE))
    if args.evaluate_test:
        metadata = joblib.load(args.output / "metadata.joblib")
        if metadata["fingerprints"] != fingerprint():
            raise ValueError("Training source or cached features changed; refusing evaluation.")
        test = frame[frame.split == "test"]
        values, _ = encode(test, metadata)
        model = xgb.Booster(model_file=args.output / "model.ubj")
        pred = inverse(model.predict(matrix(values, test, metadata, metadata["trial"], labels=False)), metadata["trial"]["scale"])
        result = metrics(test.target_raw, pred)
        export_predictions(test, pred, args.output / "test_predictions.csv")
        (args.output / "test_metrics.json").write_text(json.dumps(result, indent=2))
        print("TEST", json.dumps(result), flush=True)
        return
    if (args.output / "model.ubj").exists():
        parser.error("Choose a new output directory rather than overwrite an existing model.")
    args.output.mkdir(parents=True, exist_ok=True)
    train, valid = frame[frame.split == "train"], frame[frame.split == "valid"]
    train_values, schema = encode(train)
    valid_values, _ = encode(valid, schema)
    signature = fingerprint()
    print(f"Train={len(train)}, validation={len(valid)}, features={train_values.shape[1]}; test excluded from selection", flush=True)
    best, results = float("inf"), []
    for trial in TRIALS:
        started = time.monotonic()
        params = {"objective": trial["objective"], "max_depth": trial["max_depth"],
                  "eta": .04, "min_child_weight": 20, "subsample": .85, "colsample_bytree": .9,
                  "lambda": 10., "alpha": 0., "tree_method": "hist", "seed": 42,
                  "nthread": 2, "disable_default_eval_metric": 1}
        dtrain, dvalid = matrix(train_values, train, schema, trial), matrix(valid_values, valid, schema, trial)
        actual = valid.target_raw.to_numpy(np.float64)

        def score(pred, _):
            return "raw_wmape", float(np.abs(inverse(pred, trial["scale"])-actual).sum()/actual.sum())

        history = {}
        print("TRAIN", trial["name"], flush=True)
        model = xgb.train(params, dtrain, num_boost_round=args.rounds, evals=[(dvalid, "validation")],
                          custom_metric=score, evals_result=history, verbose_eval=200,
                          callbacks=[xgb.callback.EarlyStopping(rounds=80, metric_name="raw_wmape",
                                                               data_name="validation", maximize=False, save_best=True)])
        prediction = inverse(model.predict(dvalid), trial["scale"])
        result = {"trial": trial["name"], "best_iteration": int(model.best_iteration),
                  "elapsed_seconds": time.monotonic()-started, **metrics(actual, prediction)}
        results.append(result)
        pd.DataFrame(results).to_csv(args.output / "search_results.csv", index=False)
        pd.DataFrame({"iteration": np.arange(len(history["validation"]["raw_wmape"])),
                      "validation_wmape": history["validation"]["raw_wmape"]}).to_csv(args.output / f"history_{trial['name']}.csv", index=False)
        print("RESULT", json.dumps(result), flush=True)
        if result["WMAPE"] < best:
            best = result["WMAPE"]
            model.save_model(args.output / "model.ubj")
            metadata = {**schema, "trial": trial, "params": params, "fingerprints": signature,
                        "xgboost_version": xgb.__version__, "validation": result,
                        "selection": "2024 validation raw-count WMAPE", "best_iteration": int(model.best_iteration)}
            joblib.dump(metadata, args.output / "metadata.joblib")
            (args.output / "validation_metrics.json").write_text(json.dumps(result, indent=2))
            export_predictions(valid, prediction, args.output / "validation_predictions.csv")
            importance = model.get_score(importance_type="gain")
            pd.DataFrame({"feature": schema["feature_names"], "gain": [importance.get(f, 0) for f in schema["feature_names"]]}).sort_values("gain", ascending=False).to_csv(args.output / "feature_importance.csv", index=False)
    print(f"Best validation WMAPE: {best:.4%}. Test has not been evaluated.", flush=True)


if __name__ == "__main__":
    main()
