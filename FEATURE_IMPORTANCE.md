# Permutation Feature Importance Analysis

**Module:** Model Explainability & Feature Ranking  
**Author:** Priyanshi Gupta  
**Team:** Team 4 (EdgeX)  
**Date:** November 2026  
**Repository:** [EdgeAI](https://github.com/YashP001/EdgeAI)

---

## 1. Executive Summary

To deploy machine learning models safely on resource-constrained IoT edge devices, model explainability is critical. This analysis measures the global importance of each sensor feature in predicting the remaining battery life (in days) for smart doorbells.

We employ **Permutation Feature Importance**, an established model-agnostic technique that evaluates the degradation in predictive performance when a feature's values are randomly shuffled. If shuffling a feature causes a significant increase in Mean Absolute Error (MAE), the model relies heavily on that feature.

### Key Findings
1. **`battery_v` (Battery Voltage)** is by far the single most critical feature (+43.58 days error increase).
2. **`rings_per_day`** and **`wifi_rssi_dbm`** form a strong second tier (+12.11 and +10.94 days error increase), reflecting operational power drain.
3. **`age_months`** and **`temp_c`** provide essential refinement (+7.26 and +4.88 days error increase), capturing physical battery degradation and thermal kinetics.

---

## 2. Methodology

### 2.1 The Permutation Importance Algorithm
1. **Model Loading:** Load the committed 3-layer architecture (`5 -> 64 -> 64 -> 1`, 4,609 parameters) with saved weights from epoch 334.
2. **Validation Split:** Use the exact deterministic 80/20 train/validation split (1,600 training, 400 validation samples) seeded with `seed=0` to ensure reproducibility.
3. **Baseline MAE:** Evaluate model predictions against actual `days_left` on the unperturbed validation set.
4. **Shuffling & Scoring:**
   - For each feature $j \in \{1, \dots, 5\}$:
     - Perform $N = 30$ independent shuffling iterations.
     - In each iteration, permute the values of column $j$ across all 400 validation samples while leaving all other 4 columns intact.
     - Compute the perturbed validation MAE:
       $$\Delta \text{MAE}_{j, k} = \text{MAE}_{\text{shuffled}, j, k} - \text{MAE}_{\text{baseline}}$$
     - Record the mean importance $\mu_j$ and standard deviation $\sigma_j$:
       $$\mu_j = \frac{1}{N} \sum_{k=1}^N \Delta \text{MAE}_{j, k}, \quad \sigma_j = \sqrt{\frac{1}{N-1}\sum_{k=1}^N (\Delta \text{MAE}_{j, k} - \mu_j)^2}$$
5. **Ranking:** Sort features in descending order of $\mu_j$.

> [!NOTE]
> Negative scores can naturally occur if shuffling random noise slightly improves prediction by chance. Negative scores are never artificially clamped to zero.

---

## 3. Quantitative Results

- **Baseline Validation MAE:** `3.7005 days` (matches original training validation evaluation exactly)
- **Validation Dataset Size:** 400 samples
- **Permutation Repeats ($N$):** 30 per feature

| Rank | Feature | Mean MAE Increase (Days) | Std Dev ($\pm$) | Relative Impact |
|:---:|:---|:---:|:---:|:---|
| **1** | `battery_v` | **+43.5848** | $\pm 1.9354$ | Dominant primary indicator |
| **2** | `rings_per_day` | **+12.1129** | $\pm 0.5386$ | Major mechanical/active load |
| **3** | `wifi_rssi_dbm` | **+10.9374** | $\pm 0.4931$ | Major RF power consumption |
| **4** | `age_months` | **+7.2609** | $\pm 0.4883$ | Chemical cell degradation |
| **5** | `temp_c` | **+4.8791** | $\pm 0.2807$ | Temperature-dependent capacity |

---

## 4. Domain & Physical Interpretation

### 4.1 `battery_v` (Battery Voltage) — +43.58 days
Battery voltage is the fundamental thermodynamic indicator of remaining chemical potential in lithium-ion/alkaline cells. Shuffling voltage destroys the model's primary discharge curve tracking, causing errors to skyrocket from 3.7 days to over 47 days.

### 4.2 `rings_per_day` (Ring Frequency) — +12.11 days
Every doorbell ring event activates multiple power-hungry sub-circuits: the chime solenoid/buzzer, camera sensor, video encoding ISP, and bright LED halo. High daily ring counts rapidly accelerate battery depletion.

### 4.3 `wifi_rssi_dbm` (Signal Strength) — +10.94 days
In IoT devices, Wi-Fi radio power management scales with signal strength:
- Weak signals (e.g., $-85\text{ dBm}$) trigger maximum transmission power amplification (PA) and frequent packet retransmissions.
- Strong signals (e.g., $-45\text{ dBm}$) allow low-power transmission states and shorter active radio periods.
Hence, Wi-Fi RSSI strongly modulates real-world battery drain.

### 4.4 `age_months` (Battery Age) — +7.26 days
Rechargeable and primary batteries undergo internal impedance growth, active material loss, and SEI (Solid Electrolyte Interphase) layer thickening over time. A 24-month-old battery delivers fewer total milliampere-hours at a given voltage than a fresh cell.

### 4.5 `temp_c` (Ambient Temperature) — +4.88 days
Sub-zero temperatures increase electrolyte viscosity and internal cell resistance, temporarily reducing usable capacity, whereas high ambient heat accelerates self-discharge. While non-trivial, ambient temperature is secondary to voltage and active workload.

---

## 5. Implications for Edge Deployment

1. **ADC Sensor Integrity:** Because `battery_v` drives >55% of the overall model sensitivity, the analog-to-digital converter (ADC) monitoring battery voltage requires high sampling precision and low-pass filtering to remove motor/RF load transients.
2. **Sensor Duty Cycling:** In deep-sleep battery-saver modes, ambient temperature (`temp_c`) and Wi-Fi RSSI (`wifi_rssi_dbm`) can be sampled less frequently (e.g., once per hour) with minimal accuracy penalty, whereas active ring counts must be accumulated accurately via hardware interrupts.

---

## 6. How to Reproduce

Run the standalone analysis script from the project root:

```powershell
py -3.10 tools/feature_importance.py
```

### Generated Artifacts
- **CSV Data:** `results/team_04/feature_importance.csv`
- **JSON Metadata:** `results/team_04/feature_importance_summary.json`
- **Bar Chart Plot:** `results/team_04/feature_importance.png`

