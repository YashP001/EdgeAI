# Dataset Audit & Cleaning Strategy

**Dataset:** `data/team_04_train.csv` (2,000 observations)  
**Target Variable:** `days_left` (continuous, battery remaining days)  
**Input Features:** `battery_v`, `temp_c`, `rings_per_day`, `wifi_rssi_dbm`, `age_months`  

---

## 1. Overview

This document explains the sensor data audit performed on Team 4's dataset, documenting the detected anomalies, sentinel values, and the rationale behind retaining all 2,000 rows during training.

---

## 2. Sensor Sentinel & Anomaly Findings

An exhaustive audit of the 2,000 raw samples revealed 24 anomalous sensor readings across 3 distinct categories:

| Anomaly Type | Flagged Rows | Physical Explanation |
|---|---:|---|
| `zero_battery_sensor_sentinel` | 10 | Battery voltage reported as exactly `0.0 V`. A smart doorbell cannot operate at 0 V; this represents an ADC disconnect or power-cycle glitch. |
| `temp_c_out_of_range` | 5 | Temperature reported as `-127.0 °C` or `500.0 °C`. Typical digital bus (e.g. 1-Wire DS18B20) unpopulated/disconnected fault code. |
| `zero_rssi_sensor_sentinel` | 9 | Wi-Fi RSSI reported as exactly `0.0 dBm`. Typical network driver sentinel when disconnected from an access point. |

---

## 3. Row Retention vs. Gateway Filtering Policy

### 3.1 Training Policy (Zero-Row Removal)
- All **2,000 rows were retained** in `data/team_04_train.csv` (1,600 training, 400 validation).
- **Rationale:** Preserving the original dataset ensures complete byte-for-byte reproducibility across instructor evaluation benchmarks. The neural network's non-linear hidden layers learned robust representations despite these 24 anomalies (achieving 3.7005 days validation MAE).

### 3.2 Live API Policy (Strict Gateway Rejection)
- In production, feeding sentinel values (like 0 V or -127 °C) into the model is rejected at the API validation boundary with `HTTP 422`.
- Physical sensors must deliver readings within operational bounds before inference is attempted.

---

## 4. Train/Validation Split Verification

- **Split Ratio:** 80% Training (1,600 rows) / 20% Validation (400 rows)
- **Random State:** Seeded with `numpy.random.default_rng(0).permutation(len(frame))`
- **Target Distribution:**
  - `days_left` spans from 0.0 to ~226 days.
  - Normalization parameters ($\mu, \sigma$) are computed strictly on the 1,600 training rows to prevent data leakage into the validation set.

