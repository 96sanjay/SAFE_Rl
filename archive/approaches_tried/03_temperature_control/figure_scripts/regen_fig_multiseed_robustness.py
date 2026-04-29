#!/usr/bin/env python3
"""
Regenerate fig_multiseed_robustness.pdf with ALL 6 CSAC-LB seeds
AND all 5 algorithms side-by-side.

Uses multiseed_eval_summary.json as the data source.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.3,
    "grid.linestyle": "--",
    "axes.linewidth": 0.5,
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent.parent.parent
SUMMARY = BASE / "runs" / "multiseed_eval_summary.json"
THESIS_FIG = Path(__file__).resolve().parent.parent.parent.parent.parent / "studienarbeiten-master" / "thesis" / "figures"
LOCAL_FIG = Path(__file__).resolve().parent.parent / "figures"

with open(SUMMARY) as f:
    data = json.load(f)

# Order: CSAC-LB first (best), then CPO, SAC-Lag, CUP, FOCOPS
algo_order = ["CSAC-LB", "CPO", "SAC-Lag", "CUP", "FOCOPS"]
colors = {
    "CSAC-LB": "#C65D4B",  # coral
    "CPO":     "#2F5D8A",  # blue
    "SAC-Lag": "#6D597A",  # purple
    "CUP":     "#2A7F7F",  # teal
    "FOCOPS":  "#B26A00",  # amber
}

RBC_VIOL = 0.516  # Cooling RBC baseline

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.5, 2.8))

positions = []
labels = []
for i, algo in enumerate(algo_order):
    ad = data[algo]
    vrates = [s["violation_rate"] * 100 for s in ad["per_seed"]]
    n = ad["n_seeds"]
    mean_vr = np.mean(vrates)

    pos = i
    positions.append(pos)
    labels.append(f"{algo} ({n} seeds)")

    # Box plot
    bp = ax.boxplot(
        [vrates], positions=[pos], widths=0.5,
        patch_artist=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=4),
        medianprops=dict(color="black", linewidth=1.0),
        boxprops=dict(facecolor=colors[algo], alpha=0.3,
                      edgecolor=colors[algo], linewidth=0.8),
        whiskerprops=dict(color=colors[algo], linewidth=0.8),
        capprops=dict(color=colors[algo], linewidth=0.8),
        flierprops=dict(marker="o", markerfacecolor=colors[algo],
                        markeredgecolor=colors[algo], markersize=3),
    )

    # Individual seed points
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vrates))
    ax.scatter(
        [pos + j for j in jitter], vrates,
        color=colors[algo], s=18, zorder=5, edgecolors="white", linewidths=0.3
    )

    # Annotate mean
    ax.annotate(
        f"mean = {mean_vr:.1f}%",
        xy=(pos + 0.35, mean_vr), fontsize=5.5, color=colors[algo],
        ha="left", va="center",
    )

# RBC baseline
ax.axhline(RBC_VIOL * 100, color="gray", ls="--", lw=0.8, zorder=1)
ax.annotate(
    f"Cooling RBC = {RBC_VIOL*100:.1f}%",
    xy=(len(algo_order) - 0.5, RBC_VIOL * 100 + 1.5),
    fontsize=5.5, color="gray", ha="right",
)

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Comfort Violation Rate (%)")
ax.set_title("Multi-Seed Robustness: All Algorithms")
ax.set_ylim(-2, 100)

fig.tight_layout()

# Save to both locations
for out_dir in [LOCAL_FIG, THESIS_FIG]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_multiseed_robustness.pdf")
    fig.savefig(out_dir / "fig_multiseed_robustness.png")
    print(f"Saved to {out_dir / 'fig_multiseed_robustness.pdf'}")

plt.close(fig)
