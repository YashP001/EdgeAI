# Verified on 19 November 2026

- Nine automated tests passed, including explicit row-retention and source-byte preservation checks.
- Team 4 CSV remains byte-for-byte identical to the instructor ZIP: 2000 rows; 0 removed. SHA256 is recorded in results/team_04/data_integrity.json.
- All 11 requested architecture/normalisation experiments ran for 1500 epochs on the same 1600/400 split.
- Saved final model is 5 -> 64 -> 64 -> 1, validation MAE 3.7005 days, training MAE 3.3013 days.
- Actual trained model passed the local HTTP checker: health, per-layer arithmetic, timing fields, 400 single/batch predictions, and all five invalid-field tests.
- Swagger request-body schemas expose all five input fields and batch rows.
- Both 200-request local benchmark modes completed. Raw measurements and summaries are under results/team_04/benchmarks/.
- Hardware width table and exact 300 KiB thresholds are under results/hardware/.
- Public Codespaces endpoint remains UNVERIFIED. Browser automation was stopped because it could not confidently determine the current browser URL on Windows. Existing Git credentials can push the repository but the Codespaces API returned HTTP 404.
