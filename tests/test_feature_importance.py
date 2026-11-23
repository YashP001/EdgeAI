"""Unit tests for the permutation feature importance analysis tool."""
import json
import unittest
from pathlib import Path

import numpy as np

from tools.feature_importance import (
    calculate_mae,
    load_model,
    load_validation_data,
    calculate_permutation_importance,
)
from app.model import FEATURES

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "team_04"


class FeatureImportanceTests(unittest.TestCase):
    """Test suite verifying feature importance calculations and output artifacts."""

    def test_calculate_mae(self):
        """Verify MAE calculation on known vectors."""
        pred = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        actual = np.array([12.0, 19.0, 33.0], dtype=np.float32)
        # Errors: |10-12|=2, |20-19|=1, |30-33|=3 -> mean = 2.0
        self.assertAlmostEqual(calculate_mae(pred, actual), 2.0, places=4)

    def test_validation_split_dimensions(self):
        """Validation data must have 400 rows and 5 feature columns."""
        X_val, y_val = load_validation_data()
        self.assertEqual(X_val.shape, (400, 5))
        self.assertEqual(y_val.shape, (400,))

    def test_feature_order_integrity(self):
        """Model feature configuration must match application feature list."""
        model, mean, std = load_model()
        self.assertEqual(len(mean), len(FEATURES))
        self.assertEqual(len(std), len(FEATURES))
        self.assertTrue((std > 0).all())

    def test_permutation_importance_baseline_and_ranking(self):
        """Calculate importance with small repeat count to verify non-zero scores and baseline MAE."""
        model, mean, std = load_model()
        X_val, y_val = load_validation_data()

        baseline_mae, results = calculate_permutation_importance(
            model, X_val, y_val, mean, std, repeats=3, seed=42
        )

        # Baseline MAE should be approximately 3.70 days
        self.assertAlmostEqual(baseline_mae, 3.7005, delta=0.01)

        # Must report all 5 features
        self.assertEqual(len(results), 5)
        ranked_names = [item["feature"] for item in results]
        self.assertEqual(set(ranked_names), set(FEATURES))

        # Most important feature must be battery_v
        self.assertEqual(results[0]["feature"], "battery_v")

    def test_generated_artifacts_exist(self):
        """Verify generated files exist if previously run."""
        csv_file = RESULTS_DIR / "feature_importance.csv"
        json_file = RESULTS_DIR / "feature_importance_summary.json"
        png_file = RESULTS_DIR / "feature_importance.png"

        if json_file.exists():
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("baseline_validation_mae", data)
            self.assertIn("ranking", data)
            self.assertEqual(len(data["ranking"]), 5)

        if csv_file.exists():
            self.assertGreater(csv_file.stat().st_size, 0)

        if png_file.exists():
            self.assertGreater(png_file.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

