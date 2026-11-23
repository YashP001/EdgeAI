
"""Permutation feature importance for the EdgeAI battery prediction model.

Uses the original deterministic 80/20 validation split and the saved model.
A feature is more important when shuffling it increases validation MAE.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet

MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "team_04"


def calculate_mae(predicted, actual):
    """Calculate mean absolute error."""
    return float(np.mean(np.abs(predicted - actual)))


def load_model():
    """Load the existing trained model and its normalisation settings."""
    config_path = MODEL_DIR / "config.json"
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    if config["features"] != FEATURES:
        raise ValueError("Feature order does not match the saved model.")

    model = BatteryNet(tuple(config["hidden"]))
    state = torch.load(
        MODEL_DIR / "model.pt",
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    mean = np.asarray(config["mu"], dtype=np.float32)
    std = np.asarray(config["sd"], dtype=np.float32)

    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Invalid normalisation values.")

    if (std <= 0).any():
        raise ValueError("Standard deviations must be positive.")

    return model, mean, std


def load_validation_data():
    """Reproduce the original training script's validation split."""
    data_path = DATA_DIR / "team_04_train.csv"
    frame = pd.read_csv(data_path)

    required_columns = FEATURES + ["days_left"]
    missing = set(required_columns) - set(frame.columns)

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    frame[required_columns] = frame[required_columns].apply(
        pd.to_numeric, errors="raise"
    )

    if not np.isfinite(frame[required_columns].to_numpy()).all():
        raise ValueError("Dataset contains missing or non-finite values.")

    # Match train.py exactly: seed 0, first 20% of shuffled indices.
    indices = np.random.default_rng(0).permutation(len(frame))
    validation_count = len(frame) // 5
    validation = frame.iloc[indices[:validation_count]]

    X_val = validation[FEATURES].to_numpy(dtype=np.float32)
    y_val = validation["days_left"].to_numpy(dtype=np.float32)

    return X_val, y_val


def predict(model, X, mean, std):
    """Predict days remaining using the saved model."""
    normalised = (X - mean) / std
    tensor = torch.from_numpy(normalised.astype(np.float32))

    with torch.inference_mode():
        predictions = model(tensor).numpy()

    return np.maximum(predictions, 0)


def calculate_permutation_importance(
    model, X_val, y_val, mean, std, repeats=30, seed=42
):
    """Measure the validation MAE increase after shuffling each feature."""

    baseline_predictions = predict(model, X_val, mean, std)
    baseline_mae = calculate_mae(baseline_predictions, y_val)

    rng = np.random.default_rng(seed)
    results = []

    for feature_index, feature_name in enumerate(FEATURES):
        increases = []

        for _ in range(repeats):
            shuffled_X = X_val.copy()

            shuffled_X[:, feature_index] = rng.permutation(
                shuffled_X[:, feature_index]
            )

            shuffled_predictions = predict(
                model, shuffled_X, mean, std
            )

            shuffled_mae = calculate_mae(
                shuffled_predictions, y_val
            )

            increases.append(shuffled_mae - baseline_mae)

        results.append({
            "feature": feature_name,
            "baseline_mae": baseline_mae,
            "importance_mean": float(np.mean(increases)),
            "importance_std": float(np.std(increases)),
            "repeats": repeats
        })

    results.sort(
        key=lambda item: item["importance_mean"],
        reverse=True
    )

    return baseline_mae, results


def save_results(baseline_mae, results):
    """Save importance scores and generate a horizontal bar chart."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = RESULTS_DIR / "feature_importance.csv"
    output_plot = RESULTS_DIR / "feature_importance.png"
    output_json = RESULTS_DIR / "feature_importance_summary.json"

    pd.DataFrame(results).to_csv(output_csv, index=False)

    summary = {
        "method": "Permutation Feature Importance",
        "baseline_validation_mae": baseline_mae,
        "metric": "Increase in validation MAE (days)",
        "repeats_per_feature": results[0]["repeats"],
        "ranking": results
    }

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    names = [item["feature"] for item in results]
    means = [item["importance_mean"] for item in results]
    errors = [item["importance_std"] for item in results]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        names[::-1],
        means[::-1],
        xerr=errors[::-1],
        capsize=4
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Increase in validation MAE (days)")
    ax.set_ylabel("Input features")
    ax.set_title("EdgeAI - Permutation Feature Importance")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_plot, dpi=200)
    plt.close(fig)

    print("\nFeature importance results")
    print("-" * 55)
    print(f"Baseline validation MAE: {baseline_mae:.4f} days\n")

    for rank, item in enumerate(results, start=1):
        print(
            f"{rank}. {item['feature']}: "
            f"{item['importance_mean']:.4f} "
            f"+/- {item['importance_std']:.4f} days"
        )

    print("\nFiles generated:")
    print(output_csv)
    print(output_json)
    print(output_plot)


def main():
    torch.set_num_threads(1)

    model, mean, std = load_model()
    X_val, y_val = load_validation_data()

    baseline_mae, results = calculate_permutation_importance(
        model,
        X_val,
        y_val,
        mean,
        std,
        repeats=30
    )

    save_results(baseline_mae, results)


if __name__ == "__main__":
    main()
