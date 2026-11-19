"""Mission 4: reuse the API counting function for all requested widths."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.metrics import describe_model, SRAM_BYTES
from app.model import BatteryNet


def first_failure(bytes_per_param):
    # P(w) = (5w+w) + (w*w+w) + (w+1) = w*w+8w+1.
    width = 1
    while bytes_per_param * (width * width + 8 * width + 1) <= SRAM_BYTES:
        width += 1
    return width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--macs-per-second", type=float, default=10_000_000,
                        help="Illustrative conservative assumption, not measured board speed")
    args = parser.parse_args()
    if args.macs_per_second <= 0:
        parser.error("Speed must be positive")
    output = Path(__file__).resolve().parents[1] / "results" / "hardware"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for width in (32, 128, 256, 512, 2048):
        info = describe_model(BatteryNet((width, width)))
        row = {"width": width, **{k: v for k, v in info.items() if k != "layers"}}
        row["estimated_compute_ms"] = info["total_macs"] / args.macs_per_second * 1000
        rows.append(row)
    with (output / "widths.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    thresholds = {
        "budget_bytes": SRAM_BYTES, "interpretation": "300 KiB; weights and biases only",
        "first_failing_width_f32": first_failure(4),
        "first_failing_width_int8": first_failure(1),
        "assumed_macs_per_second": args.macs_per_second,
        "estimate_is_not_a_measurement": True,
    }
    (output / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    print(json.dumps(thresholds, indent=2))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
