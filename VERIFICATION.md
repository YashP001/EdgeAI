# Verification - 19 November 2026

- Nine automated tests passed, including explicit all-row retention and source-byte preservation.
- Original Team 4 CSV remains byte-for-byte identical to the instructor ZIP: 2000 rows, zero removed. The supplied client/bench.py and .devcontainer/devcontainer.json are also byte-identical. Proof: results/team_04/protected_files_integrity.json.
- Only Team 4 data was used for training and analysis. All 11 architecture/normalisation experiments ran for 1500 epochs on the same 1600/400 split.
- The original unchanged starter was separately measured at 9.12 days validation MAE; the final model was not overwritten.
- Committed model: 5 -> 64 -> 64 -> 1, 4609 parameters, validation MAE 3.7005 days, training MAE 3.3013 days.
- Public /health responds without authentication at the endpoint in SUBMISSION.md. /model_info matches /health parameter count, layer connections, storage counts, MACs and both SRAM flags.
- Actual deployed model passed client/verify.py from the laptop and inside Codespaces: health, per-layer arithmetic, five timing fields, 400 single/batch predictions, all five invalid input fields.
- A separate 50-reading public check produced exactly equal predictions for all 50 single/batch pairs; maximum absolute difference was 0 days.
- Through public Swagger: negative battery, impossible temperature and negative ring count all returned 422; a valid reading returned 200 with a prediction and timings.
- Supplied benchmark completed with 200 requests each for Windows localhost keepalive/no-keepalive, Codespace localhost keepalive/no-keepalive, and laptop-to-Codespace keepalive/no-keepalive. Raw CSVs and full percentiles are committed.
- Final public 50-request benchmark completed successfully.
- Hardware tables, exact SRAM thresholds, activation memory and estimated board compute are documented. No physical board or quantised model was measured.
- Public health screenshot and Swagger proof are in results/team_04/.
- The live endpoint depends on the Codespace remaining running with port 8000 Public. Future check-window availability and submission require the team's follow-through.
