"""Use EVERY row of the selected team's data, compare models, and save the API model.

python train.py --team NN --sweep
python train.py --team NN --hidden 32,32 --epochs 1500
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from app.metrics import describe_model
from app.model import FEATURES, BatteryNet
from app.validation import LIMITS

ROOT = Path(__file__).resolve().parent


def mae(predicted, actual):
    return float(np.mean(np.abs(predicted - actual)))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prepare(team, output):
    path = ROOT / "data" / f"team_{team:02d}_train.csv"
    frame = pd.read_csv(path)
    columns = FEATURES + ["days_left"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    rules = {"nonfinite_or_missing": ~np.isfinite(frame[columns]).all(axis=1)}
    for name, (low, high) in LIMITS.items():
        rules[name + "_out_of_range"] = (frame[name] < low) | (frame[name] > high)
    rules["zero_battery_sensor_sentinel"] = frame.battery_v == 0
    rules["zero_rssi_sensor_sentinel"] = frame.wifi_rssi_dbm == 0
    rules["negative_days_left"] = frame.days_left < 0
    bad = np.logical_or.reduce([mask.to_numpy() for mask in rules.values()])
    if rules["nonfinite_or_missing"].any():
        raise ValueError("Nonfinite/missing values found. No rows were removed; choose an explicit imputation policy before training.")
    if len(frame) < 20:
        raise ValueError("Too few readings for training and validation")
    counts = {name: int(mask.sum()) for name, mask in rules.items()}
    cleaning = {"rows_original": len(frame), "rows_removed": 0,
                "rows_retained": len(frame), "rows_flagged_but_retained": int(bad.sum()),
                "policy": "Keep every row unchanged, including flagged readings, as requested by the team.",
                "rule_counts_overlap": counts,
                "limits": LIMITS, "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    print(json.dumps(cleaning, indent=2), flush=True)
    save_json(output / "cleaning.json", cleaning)
    pd.DataFrame({"csv_row": np.arange(len(frame)) + 2, "flagged_but_retained": bad,
                  **{name: mask.to_numpy() for name, mask in rules.items()}}).to_csv(output / "row_audit.csv", index=False)
    for label, data in [("all_rows", frame)]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for axis, name in zip(axes.flat, columns):
            values = data[name].to_numpy()
            axis.hist(values[np.isfinite(values)], bins=35)
            axis.set_title(name)
        fig.tight_layout()
        fig.savefig(output / f"histograms_{label}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, name in zip(axes.flat, FEATURES):
        axis.scatter(frame[name], frame.days_left, s=5, alpha=.3)
        axis.set(xlabel=name, ylabel="days_left")
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output / "relationships.png", dpi=150)
    plt.close(fig)
    idx = np.random.default_rng(0).permutation(len(frame))
    n_val = len(frame) // 5
    val, tr = frame.iloc[idx[:n_val]], frame.iloc[idx[n_val:]]
    return (tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32),
            val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32), cleaning)


def fit(X, y, Xv, yv, hidden, epochs, normalise=True):
    mu = X.mean(axis=0) if normalise else np.zeros(X.shape[1], np.float32)
    sd = X.std(axis=0) if normalise else np.ones(X.shape[1], np.float32)
    sd = np.where(sd > 1e-6, sd, 1).astype(np.float32)
    torch.manual_seed(0)
    net = BatteryNet(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[max(1, int(epochs * .6)), max(2, int(epochs * .85))], gamma=.3)
    Xt = torch.from_numpy((X - mu) / sd)
    yt = torch.from_numpy(y)
    Xvt = torch.from_numpy((Xv - mu) / sd)
    best, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(Xt))
        for batch in perm.split(128):
            loss = torch.nn.functional.mse_loss(net(Xt[batch]), yt[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()
        net.eval()
        with torch.inference_mode():
            validation = mae(net(Xvt).numpy().clip(0), yv)
            training = mae(net(Xt).numpy().clip(0), y)
        history.append({"epoch": epoch, "train_mae": training, "val_mae": validation})
        if validation < best:
            best, best_epoch, best_state = validation, epoch, deepcopy(net.state_dict())
        if epoch == 1 or epoch % 300 == 0 or epoch == epochs:
            print(f"hidden={hidden} normalise={normalise} epoch={epoch}: train={training:.3f} val={validation:.3f}", flush=True)
    net.load_state_dict(best_state)
    with torch.inference_mode():
        predicted = net(Xvt).numpy().clip(0)
        training = mae(net(Xt).numpy().clip(0), y)
    result = {"hidden": list(hidden), "normalise": normalise, "epochs": epochs,
              "best_epoch": best_epoch, "train_mae": training, "val_mae": best,
              "params": sum(p.numel() for p in net.parameters())}
    return net, mu, sd, result, history, predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", type=int, required=True, choices=range(1, 28))
    parser.add_argument("--hidden", default="32,32")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--sweep", action="store_true", help="Compare all ten required architectures and unnormalised starter")
    args = parser.parse_args()
    hidden = tuple(int(h) for h in args.hidden.split(","))
    if args.epochs < 1 or not hidden or any(h < 1 for h in hidden):
        parser.error("epochs and hidden widths must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = ROOT / "results" / f"team_{args.team:02d}"
    output.mkdir(parents=True, exist_ok=True)
    X, y, Xv, yv, cleaning = prepare(args.team, output)
    design = np.column_stack([X, np.ones(len(X))])
    weights = np.linalg.lstsq(design, y, rcond=None)[0]
    linear = np.column_stack([Xv, np.ones(len(Xv))]) @ weights
    baselines = {"mean_validation_mae": mae(np.full_like(yv, y.mean()), yv),
                 "linear_validation_mae": mae(linear, yv),
                 "train_rows": len(X), "validation_rows": len(Xv)}
    save_json(output / "baselines.json", baselines)
    print("Baselines:", baselines, flush=True)
    pd.DataFrame({"true": yv, "linear_prediction": linear,
                  "absolute_error": np.abs(linear - yv)}).to_csv(output / "linear_predictions.csv", index=False)
    fig, ax = plt.subplots()
    ax.scatter(yv, linear, s=8, alpha=.4)
    ax.plot([yv.min(), yv.max()], [yv.min(), yv.max()], "k--")
    ax.set(xlabel="True days left", ylabel="Linear prediction")
    fig.tight_layout()
    fig.savefig(output / "linear_predictions.png", dpi=150)
    plt.close(fig)
    jobs = [(hidden, True)]
    if args.sweep:
        jobs = [((32, 32), False)] + [(h, True) for w in (4, 8, 16, 32, 64) for h in [(w,), (w, w)]]
    results = []
    best_model = None
    best_error = float("inf")
    for widths, normalise in jobs:
        net, mu, sd, result, history, predicted = fit(X, y, Xv, yv, widths, args.epochs, normalise)
        label = "x".join(map(str, widths)) + ("_normalised" if normalise else "_raw")
        pd.DataFrame(history).to_csv(output / f"history_{label}.csv", index=False)
        results.append(result)
        pd.DataFrame(results).to_csv(output / "experiments.csv", index=False)
        if normalise and result["val_mae"] < best_error:
            best_error = result["val_mae"]
            best_model = (net, mu, sd, result, predicted)
    net, mu, sd, result, predicted = best_model
    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    torch.save(net.state_dict(), model_dir / "model.pt")
    config = {"team": args.team, "hidden": result["hidden"], "mu": mu.tolist(), "sd": sd.tolist(),
              "features": FEATURES, "val_mae": round(result["val_mae"], 4),
              "train_mae": result["train_mae"], "seed": 0, "best_epoch": result["best_epoch"],
              "data_sha256": cleaning["data_sha256"]}
    save_json(model_dir / "config.json", config)
    save_json(output / "model_info.json", describe_model(net))
    save_json(output / "summary.json", {"team": args.team, "cleaning": cleaning,
              "baselines": baselines, "experiments": results, "selected": result,
              "caution": "Validation was used to choose the model; hidden-test MAE is unknown."})
    pd.DataFrame({"true": yv, "network_prediction": predicted,
                  "absolute_error": np.abs(predicted - yv)}).to_csv(output / "network_predictions.csv", index=False)
    fig, ax = plt.subplots()
    for normalise in (False, True):
        runs = [r for r in results if r["normalise"] == normalise]
        if runs:
            ax.scatter([r["params"] for r in runs], [r["val_mae"] for r in runs], label=f"normalise={normalise}")
            for r in runs:
                ax.annotate("x".join(map(str, r["hidden"])), (r["params"], r["val_mae"]), fontsize=7)
    ax.set(xlabel="Parameters", ylabel="Validation MAE (days)", xscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "mae_vs_parameters.png", dpi=150)
    plt.close(fig)
    print("Saved model:", json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
