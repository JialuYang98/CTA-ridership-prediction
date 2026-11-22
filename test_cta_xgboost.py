import unittest
import numpy as np
import pandas as pd
import xgboost as xgb

from cta_xgboost_train import enrich, encode, matrix, inverse


class XGBoostTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "station_id": ["a", "b"], "station_name": ["A", "B"],
            "daytype": ["U", "W"], "split": ["train", "train"],
            "target_raw": [10., 20.], "target_log": np.log1p([10., 20.]),
            **{f"lag{lag}": np.log1p([8., 18.]) for lag in (1, 7, 14, 28)},
        })

    def test_features_do_not_depend_on_current_or_future_targets(self):
        frame = self.frame()
        before = enrich(frame)
        frame.target_raw = 9999.
        frame.target_log = np.log1p(9999.)
        after = enrich(frame)
        x, schema = encode(before)
        y, _ = encode(after, schema)
        np.testing.assert_array_equal(x, y)
        self.assertNotIn("target_raw", schema["feature_names"])
        self.assertNotIn("target_log", schema["feature_names"])
        self.assertNotIn("split", schema["feature_names"])

    def test_unknown_categories_do_not_refit_encoder(self):
        frame = enrich(self.frame())
        _, schema = encode(frame)
        frame.station_id = "unseen"
        encoded, _ = encode(frame, schema)
        np.testing.assert_array_equal(encoded[:, -2], -np.ones(2))

    def test_seasonal_base_margin_is_preserved_at_inference(self):
        frame = enrich(self.frame())
        values, schema = encode(frame)
        for scale in ["raw", "log"]:
            trial = {"scale": scale, "seasonal": True}
            d = matrix(values, frame, schema, trial, labels=False)
            model = xgb.Booster({"objective": "reg:squarederror", "nthread": 1}, cache=[d])
            prediction = inverse(model.predict(d), scale)
            np.testing.assert_allclose(prediction, np.expm1(frame.lag7), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
