"""Behavioral checks for forecast timing, masking, and the seasonal residual path."""
import unittest

import numpy as np
import pandas as pd
import torch

from cta_neural_forecast import ResidualGCNLSTM, Windows, objective
from cta_data import fit_ride_scaler


class ForecastTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.model = ResidualGCNLSTM(3, 15, 12, torch.eye(3)).eval()
        self.x = torch.randn(2, 28, 3, 15)
        self.x[..., -1] = 1
        self.calendar = torch.zeros(2, 12)

    def test_initial_prediction_is_previous_week(self):
        with torch.no_grad():
            prediction = self.model(self.x, self.calendar)
        torch.testing.assert_close(prediction, self.x[:, -7, :, 0])

    def test_missing_previous_week_uses_observed_history(self):
        self.x[:, -7, :, -1] = 0
        with torch.no_grad():
            prediction = self.model(self.x, self.calendar)
        expected = (self.x[..., 0] * self.x[..., -1]).sum(1) / 27
        torch.testing.assert_close(prediction, expected)

    def test_window_excludes_target_and_future_counts(self):
        dates = pd.date_range("2023-12-01", periods=65)
        data = {"dates": dates, "x": torch.randn(65, 3, 15),
                "calendar": torch.zeros(65, 12), "y": torch.zeros(65, 3),
                "mask": torch.ones(65, 3)}
        ds = Windows(data, "validation")
        target_index = ds.indices[0]
        self.assertEqual(dates[target_index], pd.Timestamp("2024-01-01"))
        before = ds[0][0].clone()
        data["x"][target_index:, :, 0] = 99999
        torch.testing.assert_close(ds[0][0], before)
        self.assertEqual(len(before), 28)

    def test_scaler_ignores_validation_and_test_labels(self):
        dates = pd.date_range("2023-12-30", periods=5)
        y = np.arange(15, dtype=np.float32).reshape(5, 3)
        mask = np.ones_like(y)
        before = fit_ride_scaler(y, mask, dates)
        y[2:] = 10000
        self.assertEqual(fit_ride_scaler(y, mask, dates), before)

    def test_unobserved_labels_do_not_change_loss_or_gradient(self):
        pred = torch.zeros(1, 3, requires_grad=True)
        y = torch.tensor([[1., 2., 3.]])
        mask = torch.tensor([[1., 0., 1.]])
        scaler = {"ride_mean": 2., "ride_std": 1., "raw_mean": 10.}
        before = objective(pred, y, mask, scaler)
        before.backward()
        self.assertEqual(pred.grad[0, 1].item(), 0)
        y[0, 1] = 10.
        torch.testing.assert_close(before, objective(pred, y, mask, scaler))


if __name__ == "__main__":
    unittest.main()
