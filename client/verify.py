"""Verify a running endpoint and compare 400 single calls with one batch."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--team", type=int, required=True)
    parser.add_argument("--output", default="results/endpoint_check.json")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    session = requests.Session()
    health = session.get(url + "/health", timeout=30)
    health.raise_for_status()
    health = health.json()
    assert health["status"] == "ok" and health["team"] == args.team, health
    info = session.get(url + "/model_info", timeout=30)
    info.raise_for_status()
    info = info.json()
    assert info["total_params"] == health["params"]
    assert info["layers"][0]["in"] == 5 and info["layers"][-1]["out"] == 1
    for index, layer in enumerate(info["layers"]):
        assert layer["params"] == layer["in"] * layer["out"] + layer["out"]
        assert layer["bytes_f32"] == layer["params"] * 4
        assert layer["macs"] == layer["in"] * layer["out"]
        if index:
            assert info["layers"][index - 1]["out"] == layer["in"]
    assert info["total_params"] == sum(layer["params"] for layer in info["layers"])
    assert info["total_bytes_f32"] == info["total_params"] * 4
    assert info["total_bytes_int8"] == info["total_params"]
    assert info["total_macs"] == sum(layer["macs"] for layer in info["layers"])
    for dtype in ("f32", "int8"):
        assert info[f"fits_300kb_sram_{dtype}"] == (info[f"total_bytes_{dtype}"] <= 300 * 1024)
    rng = np.random.default_rng(7)
    rows = [{"battery_v": float(rng.uniform(3.4, 4.2)), "temp_c": float(rng.uniform(10, 45)),
             "rings_per_day": int(rng.integers(0, 30)), "wifi_rssi_dbm": int(rng.integers(-88, -45)),
             "age_months": int(rng.integers(0, 37))} for _ in range(400)]
    singles = []
    start = time.perf_counter()
    for row in rows:
        response = session.post(url + "/predict", json=row, timeout=30)
        response.raise_for_status()
        body = response.json()
        assert {"parse", "normalise", "inference", "build", "server_total"} <= body["timings_ms"].keys()
        assert all(np.isfinite(value) and value >= 0 for value in body["timings_ms"].values())
        assert np.isfinite(body["days_left"]) and body["days_left"] >= 0
        singles.append(body["days_left"])
    single_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    response = session.post(url + "/predict_batch", json={"rows": rows}, timeout=30)
    response.raise_for_status()
    batch_ms = (time.perf_counter() - start) * 1000
    np.testing.assert_allclose(response.json()["days_left"], singles, atol=.011, rtol=0)
    glitches = {"battery_v": -1, "temp_c": -273, "rings_per_day": -1,
                "wifi_rssi_dbm": 20, "age_months": -1}
    statuses = {}
    for field, bad in glitches.items():
        response = session.post(url + "/predict", json=dict(rows[0], **{field: bad}), timeout=30)
        statuses[field] = response.status_code
        assert response.status_code == 422, (field, response.text)
    result = {"url": url, "health": health, "model_info": info,
              "single_400_ms": single_ms, "batch_400_ms": batch_ms,
              "batch_speedup": single_ms / batch_ms, "glitch_statuses": statuses,
              "note": "Generated readings test the API contract, not predictive accuracy."}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
