#!/usr/bin/env python3
"""Render the selected aggregate example; no data access or model fitting."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

REPRESENTATIONS = ("Morgan", "DINOv2", "DINOv3", "DINOv3 + Morgan")
SPLITS = ("compound_disjoint", "scaffold_disjoint")
INPUTS = ("single_time", "cumulative_history")


def load_metrics(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    expected = {(condition, split, model) for condition in INPUTS for split in SPLITS for model in REPRESENTATIONS}
    metrics = {}
    for row in records:
        key = row["input"], row["split"], row["representation"]
        if key not in expected or key in metrics:
            raise ValueError("Unexpected or duplicate example row")
        mean = float(row["balanced_accuracy_mean"])
        sd = float(row["balanced_accuracy_seed_sd"])
        if int(row["time_hours"]) != 48 or int(row["n_seeds"]) != 3:
            raise ValueError("The selected example requires 48 h and all three seeds")
        if not (math.isfinite(mean) and math.isfinite(sd) and 0 <= mean <= 1 and 0 <= sd <= 1):
            raise ValueError("Invalid aggregate metric")
        metrics[key] = mean * 100, sd * 100
    if set(metrics) != expected:
        raise ValueError("Incomplete selected example grid")
    return metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=root / "results/lctp_48h_v1/metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results/lctp_48h_v1")
    args = parser.parse_args()
    metrics = load_metrics(args.metrics)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "activepheno-demo-v1"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    titles = {"single_time": "Single observed scan at 48 h", "cumulative_history": "Observed image history through 48 h"}
    colors = ("#64748b", "#2563eb", "#0f766e", "#b45309")
    for condition in INPUTS:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
        for ax, split in zip(axes, SPLITS):
            for index, (model, color) in enumerate(zip(REPRESENTATIONS, colors)):
                mean, sd = metrics[condition, split, model]
                ax.errorbar(mean, index, xerr=sd, fmt="o", color=color, markersize=9, capsize=5, elinewidth=2)
            ax.axvline(100 / 6, color="#94a3b8", linewidth=1.2, linestyle="--", label="Uniform chance (16.7%)")
            ax.set(xlim=(0, 100), xticks=range(0, 101, 20), yticks=range(4), yticklabels=REPRESENTATIONS,
                   title=split.replace("_", " ").capitalize(), xlabel="Compound-level balanced accuracy (%)")
            ax.grid(axis="x", color="#e2e8f0", zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=10)
        axes[0].invert_yaxis()
        axes[0].legend(loc="lower left", bbox_to_anchor=(-0.02, -0.37), frameon=False, fontsize=10)
        fig.suptitle("ActivePheno · " + titles[condition], fontsize=15, fontweight="bold", y=0.97)
        fig.text(0.5, 0.035, "Test results · seeds 42 / 43 / 44 · mean ± sample seed SD (not a confidence interval)", ha="center", fontsize=10, color="#475569")
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.27, top=0.81, wspace=0.16)
        for extension in ("png", "svg"):
            target = args.output_dir / f"{condition}.{extension}"
            metadata = {"Software": "ActivePheno"} if extension == "png" else {"Date": None, "Creator": "ActivePheno"}
            fig.savefig(target, dpi=180, metadata=metadata, facecolor="white")
            if extension == "svg":
                target.write_text("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
            print(target.name)
        plt.close(fig)


if __name__ == "__main__":
    main()
