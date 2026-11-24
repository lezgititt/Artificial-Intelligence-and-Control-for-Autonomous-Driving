"""Plot anchored PPO runs: held-out validation, training outcomes and the BC anchor KL."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID = "#2b2b2a", "#76756d", "#e4e3de"


def read_log(run):
    with (run / "training_log.csv").open(newline="") as file:
        return list(csv.DictReader(file))


def validation(directory, name):
    points = []
    for path in sorted(directory.glob(f"{name}_iter*/evaluation_summary.json")):
        iteration = int(path.parent.name.rsplit("iter", 1)[1])
        points.append((iteration, 100 * json.loads(path.read_text())["overall"]["success_rate"]))
    return points


def rolling(rows, key, window=10):
    episodes = np.array([float(r["episodes"]) for r in rows])
    values = np.array([float(r[key]) for r in rows]) * episodes
    kernel = np.ones(window)
    return 100 * np.convolve(values, kernel, "valid") / np.convolve(episodes, kernel, "valid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path, help="Directory with <name>_iterNNN and bc evaluations")
    parser.add_argument("runs", type=Path, nargs="+", help="Run directories; the name is the part after the last _")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or [run.name for run in args.runs]
    bc = 100 * json.loads((args.validation / "bc/evaluation_summary.json").read_text())["overall"]["success_rate"]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = (("Held-out validation success (%)", "Success (%)"),
              ("Training arrival rate (%, 10-iteration window)", "Arrival (%)"),
              ("Acceleration KL to the BC policy", "KL"))
    for ax, (title, ylabel) in zip(axes, panels):
        ax.set_title(title, color=INK, loc="left", fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    baseline = axes[0].axhline(bc, color=MUTED, linestyle="--", linewidth=1.5,
                               label=f"BC validation ({bc:.2f}%)")
    for run, label, color in zip(args.runs, labels, COLORS):
        rows = read_log(run)
        points = validation(args.validation, run.name.rsplit("_", 1)[1])
        if points:
            x, y = zip(*points)
            axes[0].plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=label)
        arrival = rolling(rows, "arrival_rate")
        axes[1].plot(np.arange(10, len(rows) + 1), arrival, color=color, linewidth=2, label=label)
        actor = [(int(r["iteration"]), float(r["anchor_accel_kl"])) for r in rows if r.get("anchor_accel_kl")]
        if actor:
            axes[2].plot(*zip(*actor), color=color, linewidth=2, label=label)
    handles, names = axes[1].get_legend_handles_labels()
    handles, names = [*handles, baseline], [*names, baseline.get_label()]
    fig.legend(handles, names, frameon=False, loc="lower center", ncol=len(names))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(args.output, dpi=150)
    print(args.output)


if __name__ == "__main__":
    main()
