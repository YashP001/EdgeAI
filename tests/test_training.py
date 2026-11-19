"""Synthetic learning check; does not load any team's data."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from train import fit, prepare
from app.model import FEATURES


class TrainingTests(unittest.TestCase):
    def test_preparation_retains_every_row_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            output = root / "results"
            output.mkdir()
            frame = pd.DataFrame({
                "battery_v": [3.8] * 25, "temp_c": [25.] * 25,
                "rings_per_day": [5.] * 25, "wifi_rssi_dbm": [-60.] * 25,
                "age_months": [12.] * 25, "days_left": np.arange(25, dtype=float),
            })
            frame.loc[0, "battery_v"] = 0
            frame.loc[1, "temp_c"] = -127
            frame.loc[2, "wifi_rssi_dbm"] = 0
            path = root / "data" / "team_04_train.csv"
            frame.to_csv(path, index=False)
            original_bytes = path.read_bytes()
            with patch("train.ROOT", root):
                X, y, Xv, yv, audit = prepare(4, output)
            self.assertEqual(len(X) + len(Xv), 25)
            self.assertEqual(audit["rows_removed"], 0)
            self.assertEqual(audit["rows_flagged_but_retained"], 3)
            self.assertEqual(path.read_bytes(), original_bytes)
            # Every original target identifies a row and survives the split.
            np.testing.assert_array_equal(np.sort(np.concatenate([y, yv])), np.arange(25))

    def test_learning_and_training_only_normalisation(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(42)
        X = rng.normal(size=(240, 5)).astype(np.float32)
        Xv = (rng.normal(size=(60, 5)) + .3).astype(np.float32)
        y = (8 + X[:, 0] * 2 - X[:, 1]).astype(np.float32)
        yv = (8 + Xv[:, 0] * 2 - Xv[:, 1]).astype(np.float32)
        net, mu, sd, result, history, predicted = fit(X, y, Xv, yv, (16, 16), 200)
        np.testing.assert_allclose(mu, X.mean(axis=0))
        np.testing.assert_allclose(sd, X.std(axis=0))
        baseline = np.mean(np.abs(yv - y.mean()))
        self.assertLess(result["val_mae"], baseline * .5)
        self.assertEqual(len(history), 200)
        with torch.inference_mode():
            reloaded_prediction = net(torch.from_numpy((Xv - mu) / sd)).numpy().clip(0)
        np.testing.assert_allclose(predicted, reloaded_prediction)


if __name__ == "__main__":
    unittest.main()
