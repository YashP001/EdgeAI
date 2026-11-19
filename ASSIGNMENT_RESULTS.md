# Team 4 - EdgeX: assignment results

Public endpoint: https://redesigned-eureka-x5gq4qgxj4g92px76-8000.app.github.dev

Repository: https://github.com/YashP001/EdgeAI

## Data and baselines

All 2,000 original rows were retained: 1,600 training and 400 validation. The CSV is byte-for-byte unchanged. The audit flags 24 sensor sentinel readings, but removes zero rows at the team's request. This intentionally differs from the README instruction to remove glitches. API validation still rejects broken incoming readings.

| Model | Validation MAE (days) |
|---|---:|
| Mean baseline | 42.6824 |
| Linear least squares | 25.8684 |
| Original instructor starter, unchanged | 9.12 |

The original starter was run separately using the instructor ZIP and this team's CSV. It did not overwrite the submission model. The controlled sweep below uses a learning-rate schedule and restores each run's best validation checkpoint, so its unnormalised run is a separate experiment.

The relationships plot shows strong voltage curvature and declining battery life as ringing frequency increases. Linear regression misses high-life readings most severely: CSV row 1505 has 225.8 actual days versus 118.48 predicted days, a 107.32-day error. The ten largest errors are recorded in linear_worst_errors.csv.

## Architecture and normalisation comparisons

| Hidden layers | Normalised | Parameters | Training MAE | Validation MAE | Best epoch |
|---|---|---:|---:|---:|---:|
| 32 -> 32 | False | 1281 | 10.5283 | 11.4502 | 1116 |
| 4 | True | 29 | 10.6454 | 10.6126 | 614 |
| 4 -> 4 | True | 49 | 8.5326 | 8.4106 | 1493 |
| 8 | True | 57 | 13.5197 | 12.3043 | 465 |
| 8 -> 8 | True | 129 | 5.2124 | 5.7368 | 1499 |
| 16 | True | 113 | 13.4358 | 12.3196 | 352 |
| 16 -> 16 | True | 385 | 4.1764 | 4.6966 | 1274 |
| 32 | True | 225 | 13.3646 | 12.4474 | 464 |
| 32 -> 32 | True | 1281 | 3.3948 | 3.7864 | 1465 |
| 64 | True | 449 | 13.3265 | 12.4447 | 412 |
| 64 -> 64 | True | 4609 | 3.3013 | 3.7005 | 334 |

The saved model is 5 -> 64 -> 64 -> 1, restored from epoch 334. Its training MAE is 3.3013 days and validation MAE is 3.7005 days. Later epochs reduce training error while validation worsens; the earlier checkpoint limits overfitting. Normalisation statistics use training rows only. Validation selected both architecture and checkpoint; the hidden-test error is unknown. A provisional forecast is about 4 days MAE.

## Per-layer memory and work

| Layer | Parameters | Float32 bytes | MACs |
|---|---:|---:|---:|
| 5 -> 64 | 384 | 1536 | 320 |
| 64 -> 64 | 4160 | 16640 | 4096 |
| 64 -> 1 | 65 | 260 | 64 |
| Total | 4609 | 18436 | 4480 |

Each dense layer has input_width * output_width weights plus output_width biases. Each connection costs one MAC by the assignment convention. The starter 5 -> 32 -> 32 -> 1 counts are 192 + 1056 + 33 = 1281 parameters, 5124 float32 bytes and 1216 MACs.

## Request timing

All modes used the supplied, unchanged client/bench.py with 200 requests. The first request is separate; p50/p95/p99 exclude it. These first requests follow a health request and are not fresh-process cold-start measurements.

| Mode | First request ms | Round trip p50 | p95 | p99 | Inference p50 | Server total p50 |
|---|---:|---:|---:|---:|---:|---:|
| Windows localhost, keepalive | 5.050 | 3.231 | 4.129 | 4.751 | 0.329 | 0.506 |
| Windows localhost, no keepalive | 5.757 | 4.480 | 6.826 | 22.046 | 0.322 | 0.499 |
| Codespace localhost, keepalive | 95.441 | 2.186 | 3.994 | 6.158 | 0.137 | 0.245 |
| Codespace localhost, no keepalive | 8.285 | 2.665 | 4.028 | 7.280 | 0.132 | 0.223 |
| Laptop to Codespace, keepalive | 620.969 | 53.791 | 80.510 | 160.991 | 0.236 | 0.379 |
| Laptop to Codespace, no keepalive | 544.290 | 507.940 | 882.613 | 1355.453 | 0.224 | 0.373 |

Every stage's p50/p95/p99 and all raw requests are preserved in benchmarks/comparison.json and the per-mode CSVs. Network plus framework is round trip minus server_total; it includes response serialization and routing overhead as well as transport.

The LLM forecast was that transport/HTTP would dominate remote inference, with 0.1-1 ms inference and 50-300 ms remote round trip. Compare those guesses with the measured percentiles above. Connection reuse avoids reconnecting and TLS setup for every remote request. Localhost omits the internet/TLS tunnel but still includes HTTP and framework work. The first inside-Codespace inference took 89.516 ms, versus 0.137 ms at the warm median, showing one-time forward-path overhead. The first remote prediction used a new HTTP session and took 621.0 ms even after the model had warmed up. Different machines and run conditions also affect results; these comparisons do not isolate hardware speed alone.

| Location | 400 single calls ms | One 400-row batch ms | Speedup |
|---|---:|---:|---:|
| Windows localhost | 1197.981 | 7.317 | 163.7x |
| Codespace localhost | 1045.805 | 26.358 | 39.7x |
| Laptop to Codespace | 28181.321 | 109.747 | 256.8x |

Batching amortises HTTP/tunnel overhead and uses one array and network call for many readings. The generated readings in these checks test API behaviour, not predictive accuracy. A separate 50-reading public batch check matched all 50 single predictions exactly (maximum difference 0 days). Three invalid readings were also tried directly in Swagger and all returned 422; a valid reading returned 200. A final 50-request public benchmark passed.

## Doorbell memory budget

Memory budgets use 300 * 1024 SRAM bytes and 8 * 1024 * 1024 PSRAM bytes. The flags concern parameter storage. Runtime allocations, stack and scratch space also need memory. Int8 is a theoretical one-byte-per-parameter estimate; no quantised model has been exported or accuracy-tested.

| Width w in 5 -> w -> w -> 1 | Parameters | Float32 bytes | Int8 bytes | SRAM f32 | SRAM int8 | PSRAM f32 | PSRAM int8 | MACs |
|---|---:|---:|---:|---|---|---|---|---:|
| 32 | 1281 | 5124 | 1281 | True | True | True | True | 1216 |
| 128 | 17409 | 69636 | 17409 | True | True | True | True | 17152 |
| 256 | 67585 | 270340 | 67585 | True | True | True | True | 67072 |
| 512 | 266241 | 1064964 | 266241 | False | True | True | True | 265216 |
| 2048 | 4210689 | 16842756 | 4210689 | False | False | False | True | 4206592 |

P(w) = w*w + 8*w + 1. Float32: w=273 uses 306856 bytes and fits, while w=274 uses 309076 bytes and fails. Int8: w=550 uses 306901 bytes and fits, while w=551 uses 308010 bytes and fails. Thus the first failing widths are 274 and 551.

The selected model uses 18436 float32 parameter bytes (4609 theoretical int8 bytes), and at most 128 simultaneous dense-layer input/output activation values: 512 float32 bytes or 128 int8 bytes. It fits the stated SRAM budget by a wide margin before runtime overhead.

At an assumed conservative 10 million MAC/s, 4480 MACs take 0.448 ms of compute. This is an estimate, not an ESP32 measurement. Espressif's [ESP-NN benchmark](https://github.com/espressif/esp-nn/blob/master/docs/performance.md) lists a 271-by-3 int8 fully connected kernel at 8439 ANSI C cycles or 1543 optimised cycles at 240 MHz, about 23 or 126 million MAC/s for that shape. The lower assumption leaves margin; it is not a measured or guaranteed rate for this model, float32 code, preprocessing or invocation overhead.

On-device inference is a reasonable production direction: it avoids remote latency and per-request cloud costs, keeps readings local, and works when Wi-Fi is down. It also avoids a radio transmission for each prediction, although net energy savings require board measurements. Cloud inference simplifies updates and central monitoring. The public cloud endpoint is still required for this assignment.

## Submission and team review

Submit the repository and verified endpoint address https://redesigned-eureka-x5gq4qgxj4g92px76-8000.app.github.dev using the channel the instructor announces. Keep the Codespace and server running with port 8000 Public during the announced check window; verify visibility after restarts. Deadline: Monday, 23 November 2026.

The team still needs to review and understand the code, rotate the assignment roles, add collaborators using their GitHub usernames, and submit via the announced channel. Those human actions are not represented as completed by these automated measurements. Pre-experiment forecasts in forecasts.txt are labelled LLM estimates, not invented student observations.
