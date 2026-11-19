"""Serve the saved model: uvicorn app.main:app --host 0.0.0.0 --port 8000."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.metrics import describe_model
from app.model import FEATURES, BatteryNet
from app.validation import LIMITS

MODEL_DIR = Path(os.environ.get("BATTERY_MODEL_DIR", Path(__file__).resolve().parents[1] / "model"))
with (MODEL_DIR / "config.json").open() as fh:
    cfg = json.load(fh)
if cfg["features"] != FEATURES:
    raise ValueError("Saved feature order does not match the server")
torch.set_num_threads(1)
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=True))
net.eval()
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
if not (np.isfinite(MU).all() and np.isfinite(SD).all() and (SD > 0).all()):
    raise ValueError("Saved normalisation must be finite with positive deviations")
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    battery_v: float = Field(gt=LIMITS["battery_v"][0], le=LIMITS["battery_v"][1])
    temp_c: float = Field(ge=LIMITS["temp_c"][0], le=LIMITS["temp_c"][1])
    rings_per_day: float = Field(ge=LIMITS["rings_per_day"][0], le=LIMITS["rings_per_day"][1])
    wifi_rssi_dbm: float = Field(ge=LIMITS["wifi_rssi_dbm"][0], lt=LIMITS["wifi_rssi_dbm"][1])
    age_months: float = Field(ge=LIMITS["age_months"][0], le=LIMITS["age_months"][1])


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")


def elapsed_ms(start):
    return round((time.perf_counter() - start) * 1000, 6)


@app.get("/model_info")
def model_info():
    return describe_model(net)


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    start = time.perf_counter()
    timings = {}
    stage = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as exc:
        # Excluding inputs also makes NaN/Infinity validation errors JSON safe.
        detail = json.loads(exc.json(include_input=False, include_context=False))
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": detail})
    timings["parse"] = elapsed_ms(stage)

    stage = time.perf_counter()
    rows = data.rows if isinstance(data, Batch) else [data]
    x = (np.array([[getattr(row, f) for f in FEATURES] for row in rows], np.float32) - MU) / SD
    timings["normalise"] = elapsed_ms(stage)

    stage = time.perf_counter()
    with torch.inference_mode():
        y = net(torch.from_numpy(x)).numpy()
    timings["inference"] = elapsed_ms(stage)

    stage = time.perf_counter()
    days = [round(float(value), 2) for value in np.maximum(y, 0)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    timings["build"] = elapsed_ms(stage)
    timings["server_total"] = elapsed_ms(start)
    body["timings_ms"] = timings
    return body


@app.post("/predict")
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request):
    return await run(request, Batch)
