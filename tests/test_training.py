"""Synthetic learning check; does not load any team's data."""
import unittest

import numpy as np
import torch

from train import fit


class TrainingTests(unittest.TestCase):
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
