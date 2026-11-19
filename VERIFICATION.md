# Verification performed locally

- Eight automated tests passed on 19 November 2026.
- API contract checks used a temporary random model, with no team data.
- Training smoke check used generated synthetic data only; validation MAE was 0.607 days for that artificial task. This is NOT the assignment's accuracy result.
- Checked model-count arithmetic, required timing fields, valid readings, malformed JSON, missing inputs, NaN/Infinity, batch size limits, invalid readings for all five fields, and 50 single/batch predictions.
- Ran tools/hardware.py: first failing two-hidden-layer width is 274 for float32 weights and 551 for theoretical int8 weights, using a 307200-byte budget. Outputs are in results/hardware/.
- Team 4 confirmed. Training retains all 2,000 rows; 24 sensor glitches are flagged but retained. Model sweep and public deployment are in progress.
