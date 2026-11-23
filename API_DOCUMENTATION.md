# EdgeAI REST API Documentation

**Service:** Smart Doorbell Battery Life Predictor  
**Framework:** FastAPI + Uvicorn  
**Port:** 8000  

---

## 1. Overview

The EdgeAI REST API serves real-time remaining battery life predictions for battery-powered smart doorbells. It hosts a trained neural network (`5 -> 64 -> 64 -> 1`, 4,609 parameters) and provides detailed operational telemetry, hardware profiling metrics, and input validation.

---

## 2. API Endpoints

### 2.1 GET `/health`
Verifies server health, loaded model parameters, and target validation metrics.

- **URL:** `GET /health`
- **Response (200 OK):**
```json
{
  "status": "ok",
  "team": 4,
  "params": 4609,
  "val_mae": 3.7005
}
```

---

### 2.2 GET `/model_info`
Returns comprehensive layer-by-layer architectural details, parameter counts, memory requirements (FP32 vs INT8), and SRAM hardware compatibility checks.

- **URL:** `GET /model_info`
- **Response (200 OK):**
```json
{
  "layers": [
    {"name": "fc1", "in": 5, "out": 64, "params": 384, "bytes_f32": 1536, "macs": 320},
    {"name": "fc2", "in": 64, "out": 64, "params": 4160, "bytes_f32": 16640, "macs": 4096},
    {"name": "fc3", "in": 64, "out": 1, "params": 65, "bytes_f32": 260, "macs": 64}
  ],
  "total_params": 4609,
  "total_bytes_f32": 18436,
  "total_bytes_int8": 4609,
  "total_macs": 4480,
  "fits_300kb_sram_f32": true,
  "fits_300kb_sram_int8": true
}
```

---

### 2.3 POST `/predict`
Infers remaining battery days from a single set of sensor telemetry readings.

- **URL:** `POST /predict`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
```json
{
  "battery_v": 3.86,
  "temp_c": 28.0,
  "rings_per_day": 5.0,
  "wifi_rssi_dbm": -66.0,
  "age_months": 22.0
}
```
- **Response (200 OK):**
```json
{
  "days_left": 142.35,
  "t_parse_us": 42.1,
  "t_norm_us": 12.5,
  "t_infer_us": 85.3,
  "t_build_us": 18.2,
  "t_total_us": 158.1
}
```

---

### 2.4 POST `/predict_batch`
Performs vectorized batch inference for arrays of telemetry readings.

- **URL:** `POST /predict_batch`
- **Request Body:** Array of reading objects (max 500 items).
- **Response (200 OK):** Array of prediction objects with per-item predictions and overall execution timings.

---

## 3. Input Validation & Error Handling

To guard against sensor glitches and physical anomalies, incoming readings are validated strictly against physical sensor boundaries:

| Feature | Min | Max | Physical Basis |
|---|---:|---:|---|
| `battery_v` | `1.0 V` | `5.0 V` | Standard single-cell battery operational range |
| `temp_c` | `-40.0 °C` | `85.0 °C` | Industrial sensor operational limits |
| `rings_per_day` | `0` | `1000` | Reasonable max ring events per 24 hours |
| `wifi_rssi_dbm` | `-100 dBm` | `-20 dBm` | Standard Wi-Fi signal sensitivity range |
| `age_months` | `0` | `120` | Up to 10 years battery service life |

If any parameter violates physical bounds, the API immediately returns `HTTP 422 Unprocessable Entity` without calling the model.

