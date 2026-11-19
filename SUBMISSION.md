# EdgeX, Team 4: submission and endpoint startup

Repository: https://github.com/YashP001/EdgeAI

Model: 5 -> 64 -> 64 -> 1; 4609 parameters; validation MAE 3.7005 days. All 2000 original rows retained; no CSV modifications.

## Remaining step: start the public endpoint

A Codespace was created for this repository:
https://redesigned-eureka-x5gq4qgxj4g92px76.github.dev/

The public API has NOT yet been started or verified. In that Codespace's terminal, run:

```sh
git pull --ff-only
bash tools/start_codespace.sh
```

Then select Ports, right-click port 8000, and set Port Visibility to Public. Copy the displayed forwarded address. Confirm that /health returns team 4, params 4609 and val_mae 3.7005. The expected address for this Codespace is https://redesigned-eureka-x5gq4qgxj4g92px76-8000.app.github.dev, but it is not a working submission until those steps and external checks succeed.

From a laptop, run (replace ADDRESS with the verified forwarded URL):

```sh
python client/verify.py --url ADDRESS --team 4 --output results/team_04/public_endpoint_check.json
python client/bench.py --url ADDRESS --n 200
python client/bench.py --url ADDRESS --n 200 --no-keepalive
```

Also run client/bench.py inside the Codespace against http://localhost:8000. Preserve the resulting CSVs and update the comparison in TEAM_GUIDE.md. Raw local Windows results are already committed; they are not substitutes for remote measurements.

Submit the repository link and verified public endpoint via the method the instructor announces. Keep the Codespace open, the server running and port 8000 Public during the check window. Recheck visibility after each restart. Deadline: Monday, 23 November 2026.
