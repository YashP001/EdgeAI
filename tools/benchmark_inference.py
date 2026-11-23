"""Inference latency benchmarking tool for the EdgeAI battery prediction model.

Profiles single-reading and batch inference latency across 1,000 iterations.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from app.model import FEATURES, BatteryNet

MODEL_DIR = ROOT / "model"


def load_model():
    """Load model and normalisation statistics."""
    with (MODEL_DIR / "config.json").open(encoding="utf-8") as f:
        config = json.load(f)

    model = BatteryNet(tuple(config["hidden"]))
    state = torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    mean = np.asarray(config["mu"], dtype=np.float32)
    std = np.asarray(config["sd"], dtype=np.float32)
    return model, mean, std


def benchmark(iterations=1000):
    """Profile inference latency in microseconds."""
    torch.set_num_threads(1)
    model, mean, std = load_model()

    sample = np.array([[3.86, 28.0, 5.0, -66.0, 22.0]], dtype=np.float32)

    # Warmup
    for _ in range(50):
        norm = (sample - mean) / std
        t = torch.from_numpy(norm)
        with torch.inference_mode():
            _ = model(t).numpy()

    latencies_us = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        norm = (sample - mean) / std
        t = torch.from_numpy(norm)
        with torch.inference_mode():
            _ = model(t).numpy()
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)

    latencies = np.asarray(latencies_us)

    print("\nInference Latency Profile (1,000 runs)")
    print("-" * 45)
    print(f"Mean Latency:    {np.mean(latencies):.2f} us")
    print(f"Median (p50):    {np.percentile(latencies, 50):.2f} us")
    print(f"95th Percentile: {np.percentile(latencies, 95):.2f} us")
    print(f"99th Percentile: {np.percentile(latencies, 99):.2f} us")
    print(f"Min / Max:       {np.min(latencies):.2f} us / {np.max(latencies):.2f} us")


if __name__ == "__main__":
    benchmark()
