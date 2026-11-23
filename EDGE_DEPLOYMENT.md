# Edge Hardware Deployment & Feasibility Report

**Project:** Smart Doorbell Battery Life Predictor  
**Team:** Team 4 (EdgeX)  
**Target Hardware:** Embedded Microcontrollers (ARM Cortex-M4, ESP32-S3, STM32F4/F7)  

---

## 1. Executive Summary

A core objective of the EdgeAI project is to verify whether the trained battery life prediction model can execute directly on low-power, memory-constrained edge hardware. This report documents the memory budget, computational cost (MACs), flash storage, and operational considerations for running on microcontroller units (MCUs) with limited SRAM (e.g., 300 KB budget).

---

## 2. Model Footprint Breakdown

The selected architecture is `5 -> 64 -> 64 -> 1` with ReLU activations, containing 4,609 trainable parameters.

### 2.1 Memory Requirements
| Representation | Weight Storage | Activation Buffer (Peak) | Total SRAM Requirement | Target Feasibility (<300 KB) |
|---|---:|---:|---:|:---:|
| **Float32 (Unquantized)** | 18,436 bytes (~18.0 KB) | 512 bytes (~0.5 KB) | **~19.0 KB** | **PASS** (6.3% of 300 KB) |
| **INT8 (Post-Quantization)** | 4,609 bytes (~4.5 KB) | 128 bytes (~0.1 KB) | **~4.7 KB** | **PASS** (1.6% of 300 KB) |

Both representations easily fit inside the strict 300 KB SRAM budget of modern edge microcontrollers, leaving >93% of RAM free for the RTOS, network stack, and sensor drivers.

---

## 3. Computational Complexity (MACs)

Computational load is measured in Multiply-Accumulate operations (MACs):
- **Layer 1 (`5 -> 64`):** $5 \times 64 = 320\text{ MACs}$
- **Layer 2 (`64 -> 64`):** $64 \times 64 = 4,096\text{ MACs}$
- **Layer 3 (`64 -> 1`):** $64 \times 1 = 64\text{ MACs}$
- **Total:** **4,480 MACs per inference**

### Latency Estimates on Common MCUs
- **ARM Cortex-M4 @ 80 MHz (STM32L4):**
  - Native single-cycle MAC instructions: ~4,480 cycles
  - Estimated inference time: **< 0.15 ms (150 µs)**
  - Energy per inference: **~2.4 µJ**
- **ESP32-S3 (Xtensa Dual-Core @ 240 MHz):**
  - SIMD vector extensions: ~2,500 cycles
  - Estimated inference time: **< 0.05 ms (50 µs)**

---

## 4. Hardware Optimization Recommendations

1. **Wake-up Triggering:** Instead of running the neural network continuously, the MCU can remain in deep-sleep (~10 µA standby) and execute inference only once every 6 hours or immediately after ring burst events.
2. **Quantization:** Converting weights from float32 to INT8 reduces flash storage from 18 KB to 4.6 KB with negligible loss in MAE.
3. **ADC Noise Filtering:** Since `battery_v` has the largest feature importance impact (+43.58 days), hardware decoupling capacitors and a median filter over 8 ADC samples should be used before feeding the voltage reading to the model.

