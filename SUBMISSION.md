# EdgeX, Team 4 - submission details

- **Endpoint:** [https://redesigned-eureka-x5gq4qgxj4g92px76-8000.app.github.dev](https://redesigned-eureka-x5gq4qgxj4g92px76-8000.app.github.dev)
- **Repository:** [YashP001/EdgeAI](https://github.com/YashP001/EdgeAI)
- [Health](https://redesigned-eureka-x5gq4qgxj4g92px76-8000.app.github.dev/health) | [Model information](https://redesigned-eureka-x5gq4qgxj4g92px76-8000.app.github.dev/model_info) | [Try predictions](https://redesigned-eureka-x5gq4qgxj4g92px76-8000.app.github.dev/docs)

Verified publicly on 19 November 2026 without authentication. Health: status ok, team 4, 4609 parameters, validation MAE 3.7005 days. All 2000 original CSV rows retained unchanged, split 1600/400 for training/validation.

The public API passed model-count checks, stage timing checks, all five invalid-reading tests, and single/batch consistency. An additional 50-reading check produced exactly identical single and batch answers. Three glitches were also tested through Swagger and returned 422; a valid example returned 200. Both public 200-request benchmark modes, inside-Codespace benchmark modes, the 400-call/batch comparisons, and the final public 50-request benchmark are saved in results/team_04/.

Full measurements, plots and explanations: [ASSIGNMENT_RESULTS.md](ASSIGNMENT_RESULTS.md). Code walkthrough: [TEAM_GUIDE.md](TEAM_GUIDE.md).

## Keep the endpoint available for grading

The Codespace is [redesigned eureka](https://redesigned-eureka-x5gq4qgxj4g92px76.github.dev/). Its API is currently running in the Python terminal. Keep it running, keep the Codespace open during the instructor's announced check window, and verify port 8000 is Public. The endpoint will stop when the Codespace stops; it is not permanent hosting.

After a restart, from the repository directory:

```sh
git pull --ff-only
bash tools/start_codespace.sh
```

Then set Ports -> 8000 -> Port Visibility -> Public. If GitHub CLI is installed and authorised, you can instead run this in a second terminal:

```sh
gh codespace ports visibility 8000:public -c "$CODESPACE_NAME"
```

Before submitting, open /health from outside the Codespace and run:

```sh
python client/bench.py --url https://redesigned-eureka-x5gq4qgxj4g92px76-8000.app.github.dev --n 50
```

Submit the two links above using the instructor's announced channel. Deadline: Monday, 23 November 2026. Review the implementation as a team; role rotation, collaborator invitations and the actual submission are human follow-up actions. The submission channel and check-window time were not supplied.
