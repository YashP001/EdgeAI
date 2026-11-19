"""Contract checks use a temporary RANDOM model; they do not measure accuracy."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from fastapi.testclient import TestClient

from app.metrics import describe_model
from app.model import BatteryNet, FEATURES

fixture = tempfile.TemporaryDirectory()
folder = Path(fixture.name)
torch.manual_seed(0)
torch.save(BatteryNet().state_dict(), folder / "model.pt")
(folder / "config.json").write_text(json.dumps({
    "team": 0, "hidden": [32, 32], "mu": [0] * 5, "sd": [1] * 5,
    "features": FEATURES, "val_mae": 0,
}))
os.environ["BATTERY_MODEL_DIR"] = str(folder)
from app.main import app
client = TestClient(app)
READING = {"battery_v": 3.86, "temp_c": 28, "rings_per_day": 5,
           "wifi_rssi_dbm": -66, "age_months": 22}


class ContractTests(unittest.TestCase):
    def test_model_counts(self):
        info = describe_model(BatteryNet())
        self.assertEqual([r["params"] for r in info["layers"]], [192, 1056, 33])
        self.assertEqual(info["total_params"], 1281)
        self.assertEqual(info["total_bytes_f32"], 5124)
        self.assertEqual(info["total_bytes_int8"], 1281)
        self.assertEqual(info["total_macs"], 1216)
        self.assertTrue(info["fits_300kb_sram_f32"])
        self.assertFalse(describe_model(BatteryNet((512, 512)))["fits_300kb_sram_f32"])

    def test_health_matches_info(self):
        health = client.get("/health").json()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["params"], client.get("/model_info").json()["total_params"])

    def test_prediction_and_timings(self):
        reply = client.post("/predict", json=READING)
        self.assertEqual(reply.status_code, 200)
        body = reply.json()
        self.assertTrue(np.isfinite(body["days_left"]))
        self.assertGreaterEqual(body["days_left"], 0)
        timing = body["timings_ms"]
        self.assertEqual(set(timing), {"parse", "normalise", "inference", "build", "server_total"})
        self.assertTrue(all(v >= 0 for v in timing.values()))
        self.assertGreaterEqual(timing["server_total"] + 0.00001,
                                sum(v for k, v in timing.items() if k != "server_total"))

    def test_batch_matches_fifty_singles(self):
        rows = [dict(READING, battery_v=3.4 + i * .015, rings_per_day=i) for i in range(50)]
        single = [client.post("/predict", json=row).json()["days_left"] for row in rows]
        batch = client.post("/predict_batch", json={"rows": rows})
        self.assertEqual(batch.status_code, 200)
        np.testing.assert_allclose(batch.json()["days_left"], single, atol=.011, rtol=0)

    def test_glitches_for_all_five_fields(self):
        for name, bad in {"battery_v": -1, "temp_c": -273, "rings_per_day": -1,
                          "wifi_rssi_dbm": 10, "age_months": -1}.items():
            with self.subTest(field=name):
                self.assertEqual(client.post("/predict", json=dict(READING, **{name: bad})).status_code, 422)
        for name, bad in {"battery_v": 99, "temp_c": 500, "rings_per_day": 1e9,
                          "wifi_rssi_dbm": -999, "age_months": 9999}.items():
            with self.subTest(field=name):
                self.assertEqual(client.post("/predict", json=dict(READING, **{name: bad})).status_code, 422)

    def test_malformed_missing_nonfinite_and_batches(self):
        for raw in ['{', '{}', json.dumps(dict(READING, battery_v=float("nan"))),
                    json.dumps(dict(READING, temp_c=float("inf")))]:
            self.assertEqual(client.post("/predict", content=raw).status_code, 422)
        for rows in [[], [dict(READING, age_months=-1)], [READING] * 10001]:
            self.assertEqual(client.post("/predict_batch", json={"rows": rows}).status_code, 422)

    def test_plausible_values_beyond_bench_ranges_are_accepted(self):
        row = dict(READING, battery_v=3.1, temp_c=-10, wifi_rssi_dbm=-95, age_months=60)
        self.assertEqual(client.post("/predict", json=row).status_code, 200)


if __name__ == "__main__":
    unittest.main()
