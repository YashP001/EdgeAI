# Team walkthrough and remaining submission steps

Read the original README.md first. The assignment asks every member to understand every line and says to use an LLM for explanations and hints rather than finished code. The implementation here is AI-assisted; review it together and check with your instructor if you intend to submit it as provided.

## Status

Team 4: EdgeX (Yash Purwar). Repository: https://github.com/YashP001/EdgeAI. All 2,000 rows are retained unchanged, with a deterministic 1,600/400 training/validation split. The architecture sweep is running; deployment and final metrics will be recorded after verification.

## Run your team's experiments

From this directory in Codespaces (use team number 4):

```sh
pip install -r requirements.txt
python train.py --team 4 --sweep --epochs 1500
python tools/hardware.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The sweep runs ten normalised architectures (widths 4, 8, 16, 32, 64; one and two hidden layers) and the unnormalised 32,32 comparison. It chooses the normalised checkpoint with the lowest validation MAE and saves model/model.pt and model/config.json. Other results are under results/team_NN/. The split is deterministic: 80% training, 20% validation, seed 0. Every experiment uses the same split.

Training uses minibatches of 128, MSE loss and Adam. MSE penalises large misses more heavily; MAE is the reported average absolute miss in days. Learning rate decreases later in training. Each run restores its best validation checkpoint. The validation score is somewhat optimistic after choosing among multiple models: the instructor's unseen 400 readings are the real test. Never claim their error has already been measured.

Normalisation computes (reading - training_mean) / training_std. Learning the mean and standard deviation from validation data would leak information about the evaluation set. Those five means and deviations are saved with the weights so the server applies the same transformation.

At the team's explicit request, EVERY CSV row is retained unchanged, including flagged sensor readings. The audit counts each rule and their union, but removes zero rows. row_audit.csv maps flags to original CSV row numbers. Training uses all retained rows across the training/validation split. Nonfinite or missing numbers cause a clear error rather than silent row removal. This differs from the README mission asking for glitch removal. API validation still refuses impossible incoming readings: positive battery voltage up to 5 V, temperature -40 to 85 C, 0 to 86400 rings/day, RSSI from -127 dBm up to but excluding 0, and age 0 to 600 months. These are conservative operating assumptions, not manufacturer specifications or CSV minima/maxima.

The linear baseline adds a column of ones to learn an intercept, then solves least squares on training rows only. Compare its predicted/true plot and largest errors with relationships.png. Curvature in the battery-voltage relationship is one reason a ReLU network may improve on a straight line; whether that happens here must come from your measurements.

## Predictions to write BEFORE measurements

Do not invent or backdate your team's predictions. Before running the experiments, record your team's estimates of mean-baseline MAE, normalisation's benefit, the best width, network inference time, and the laptop-to-Codespace round trip. Rotate Builder, Measurer, Skeptic and Scribe.

LLM forecast, recorded before any endpoint benchmark: for this tiny model, transport and HTTP/framework overhead will probably dominate the round trip. A provisional numerical guess is 0.1-1 ms for local model inference and 50-300 ms for a remote request. These are guesses, not measured results; geographic placement, warmup and connection reuse can change them substantially.

## Understand the API

app/model.py builds dense layers and ReLU activations. The final output is a scalar. app/main.py loads the saved model once, validates the five named fields, normalises them, runs inference without gradients, clips negative days to zero, rounds to two decimals and returns timings. Batch validation rejects the entire request if one row is invalid. Tiny float32 differences can occur between vectorised batches and single rows; the checker allows a one-hundredth-of-a-day rounding difference.

Timing boundaries: parse includes reading/validating the body; normalise includes constructing the array; inference is the tensor forward pass; build includes converting predictions to response values. server_total ends before FastAPI serialises the response and the client receives it. Round trip minus server_total therefore includes framework/serialization and transport, not just network latency. The first benchmark prediction occurs after the health check; it is not a clean measure of a fresh server boot.

## Hand calculations for 5 -> 32 -> 32 -> 1

| Layer | Weights | Biases | Parameters | Float32 bytes | MACs |
|---|---:|---:|---:|---:|---:|
| 5 -> 32 | 160 | 32 | 192 | 768 | 160 |
| 32 -> 32 | 1024 | 32 | 1056 | 4224 | 1024 |
| 32 -> 1 | 32 | 1 | 33 | 132 | 32 |
| Total | 1216 | 65 | 1281 | 5124 | 1216 |

For width w and two hidden layers: parameters = w*w + 8*w + 1; MACs = w*w + 6*w. ReLU has no learned parameters. Bias additions are not MACs under this assignment's convention. Largest simultaneous dense-layer input/output activation buffers use max(5+w, 2*w, w+1) values, or 256 float32 bytes at w=32. This excludes runtime, scratch space, allocation overhead and stack.

The memory flags compare weights and biases against 300*1024 bytes (300 KiB). The int8 count is a theoretical one-byte-per-parameter estimate required by the exercise, not an exported quantised model. Real integer runtimes often use wider biases plus scale metadata. Quantisation also needs calibration and a new accuracy check. tools/hardware.py reuses the API counting function for the five requested widths and calculates the exact first width exceeding the budget.

Board timing is an estimate, not a board measurement. The default assumption is 10 million MAC/s, selected as a conservative illustrative rate. Espressif's ESP-NN benchmark lists a 271-input, 3-output fully connected kernel at 8439 ANSI C cycles and 1543 optimised cycles on an ESP32-S3 at 240 MHz. That is about 23 and 126 million MAC/s for that particular int8 kernel, respectively. The lower assumption leaves margin but is not a guaranteed rate for this model, float32 code or tiny unaligned layers. Estimate compute_ms = MACs / assumed_MACs_per_second * 1000. Include preprocessing and invocation overhead when measuring a real board.

Source: [Espressif ESP-NN kernel performance](https://github.com/espressif/esp-nn/blob/master/docs/performance.md), consulted 19 November 2026.

An on-device model is a reasonable production direction if measured accuracy and resource use meet requirements: readings stay local, predictions work without Wi-Fi, and no server request fee is needed. Sending data to the cloud adds connectivity dependence and radio activity; whether local inference actually saves energy requires measurement. Cloud serving simplifies updates and monitoring. For THIS assignment the public server endpoint remains compulsory regardless of the production recommendation.

## Verify locally and remotely

Development contract checks use a temporary random model, not a trained submission model:

```sh
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

After training and starting the real server, in another terminal:

```sh
python client/verify.py --url http://localhost:8000 --team 4
python client/bench.py --url http://localhost:8000 --n 200
python client/bench.py --url http://localhost:8000 --n 200 --no-keepalive
```

Do not change client/bench.py. It saves raw measurements. Preserve each CSV under a descriptive directory, because filenames only have one-second precision. From your laptop, repeat both benchmark modes against your public Codespace address, then run client/verify.py there. Its generated inputs check behaviour, not accuracy.

Connection reuse can avoid connection setup costs, especially TLS for a remote endpoint. Localhost avoids the internet path; Windows-local measurements are not measurements inside Codespaces. A 400-row batch reduces request overhead and can use vectorised matrix operations. Explain differences from the actual recorded numbers, not from guesses alone.

## Deployment and submission

Create a GitHub repository for your team, upload this directory including hidden .devcontainer and model/, and invite the other members. Exclude .venv and __pycache__. Open a Codespace on the repository, install requirements, then run uvicorn. Set port 8000 Public and test /health, /model_info and /docs from outside the Codespace. Keep the Codespace and server running during the instructor's check window. Restarting a Codespace requires checking port visibility again.

The two submission items are the repository link and public endpoint address. The deadline is Monday, 23 November 2026. The instructor has not yet provided the submission channel or check-window time. Do not treat a local URL, a random-model contract test, or a saved screenshot as a deployed endpoint.
