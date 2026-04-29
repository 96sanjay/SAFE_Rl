#!/usr/bin/env python3
"""
Publication-quality 2x2 energy-comfort Pareto scatter.
One panel per algorithm + combined overview.

Replaces the cluttered single-panel fig_paper_energy_vs_temp.
"""
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Publication style (matching thesis figures)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.25,
    "grid.alpha": 0.20,
    "grid.linestyle": "-",
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[3]
CKPT = BASE / "docs" / "temperature_case_study" / "data" / "all_checkpoint_evals.json"
THESIS_FIG = BASE.parent / "studienarbeiten-master" / "thesis" / "figures"
LOCAL_FIG = Path(__file__).resolve().parent.parent / "figures"

with open(CKPT) as f:
    raw = json.load(f)

# Parse into per-algorithm lists
data = defaultdict(list)
for key, vals in raw.items():
    parts = key.split("|")
    algo = parts[0]
    seed = int(parts[1].replace("seed", ""))
    epoch = int(parts[2].replace("epoch", ""))
    vals["algo"] = algo
    vals["seed"] = seed
    vals["epoch"] = epoch
    data[algo].append(vals)

# RBC baseline (from eval data)
RBC_ENERGY = 7200.0  # approximate from original script
RBC_MAX_DEV = 1.93   # approximate

# Try to get exact RBC values
import os
for root, dirs, files in os.walk(str(BASE / "runs" / "csac_lb_multi_seed" / "seed_1")):
    if "eval_case_study.json" in files:
        with open(os.path.join(root, "eval_case_study.json")) as f:
            ed = json.load(f)
        for r in ed["rows"]:
            if r.get("name") == "Cooling RBC":
                RBC_ENERGY = r.get("district_import_kwh", RBC_ENERGY)
                RBC_MAX_DEV = r.get("kpi::discomfort_hot_delta_maximum",
                                    r.get("max_hot_dev", RBC_MAX_DEV))
                break
        break

# ---------------------------------------------------------------------------
# Algorithm config
# ---------------------------------------------------------------------------
ALGO_ORDER = ["CSAC-LB", "SAC-Lag", "CPO"]
PANEL_TITLES = {
    "CSAC-LB": "(a) CSAC-LB ($n$=6 seeds)",
    "SAC-Lag": "(b) SAC-Lag ($n$=3 seeds)",
    "CPO":     "(c) CPO ($n$=3 seeds)",
}
MARKERS = {"CSAC-LB": "o", "SAC-Lag": "s", "CPO": "^"}
EDGE_COLORS = {"CSAC-LB": "#888888", "SAC-Lag": "#888888", "CPO": "#888888"}

MAX_EPOCH = 60
STEPS_PER_EPOCH = 2208

# Shared colormap
cmap = plt.get_cmap("coolwarm")
norm = mcolors.Normalize(vmin=0, vmax=MAX_EPOCH)

# ---------------------------------------------------------------------------
# Pareto front helper
# ---------------------------------------------------------------------------
def pareto_mask(energy, deviation):
    """Lower energy AND lower deviation is better."""
    n = len(energy)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                if energy[j] <= energy[i] and deviation[j] <= deviation[i]:
                    if energy[j] < energy[i] or deviation[j] < deviation[i]:
                        mask[i] = False
                        break
    return mask

# ---------------------------------------------------------------------------
# Figure: 2x2 grid
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), sharex=True, sharey=True)
axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

# Panels (a)-(c): one algorithm each
for idx, algo in enumerate(ALGO_ORDER):
    ax = axes_flat[idx]
    pts = data[algo]

    energy = np.array([p["district_import_kwh"] for p in pts])
    max_dev = np.array([max(p.get("max_hot_dev", 0), p.get("max_cold_dev", 0)) for p in pts])
    epochs = np.array([p["epoch"] for p in pts])
    seeds = np.array([p["seed"] for p in pts])

    # Scatter colored by epoch
    sc = ax.scatter(
        energy, max_dev,
        c=epochs, cmap=cmap, norm=norm,
        marker=MARKERS[algo], s=18, alpha=0.75,
        linewidths=0.2, edgecolors=EDGE_COLORS[algo],
        zorder=4,
    )

    # Pareto front markers
    pm = pareto_mask(energy, max_dev)
    if pm.any():
        ax.scatter(
            energy[pm], max_dev[pm],
            marker="x", s=30, c="black", linewidths=0.8, zorder=6,
        )

    # RBC baseline
    ax.scatter([RBC_ENERGY], [RBC_MAX_DEV], marker="*", s=80,
               color="black", zorder=8)

    # Seed count annotation
    n_seeds = len(set(seeds))
    ax.set_title(PANEL_TITLES[algo], fontweight="bold", pad=4)
    ax.grid(True)

# Panel (d): combined overview (all algorithms, no epoch color — use algorithm color)
ax_d = axes_flat[3]
ALGO_COLORS = {"CSAC-LB": "#D62728", "SAC-Lag": "#9467BD", "CPO": "#1F77B4"}

for algo in ALGO_ORDER:
    pts = data[algo]
    energy = np.array([p["district_import_kwh"] for p in pts])
    max_dev = np.array([max(p.get("max_hot_dev", 0), p.get("max_cold_dev", 0)) for p in pts])

    ax_d.scatter(
        energy, max_dev,
        color=ALGO_COLORS[algo], marker=MARKERS[algo],
        s=12, alpha=0.4, linewidths=0.15, edgecolors="grey",
        zorder=4, label=algo,
    )

# RBC in combined panel
ax_d.scatter([RBC_ENERGY], [RBC_MAX_DEV], marker="*", s=80,
             color="black", zorder=8, label="Cooling RBC")
ax_d.set_title("(d) All algorithms", fontweight="bold", pad=4)
ax_d.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc",
            markerscale=1.5, handletextpad=0.2, borderpad=0.3)
ax_d.grid(True)

# Shared axis labels
for ax in [axes[1, 0], axes[1, 1]]:
    ax.set_xlabel("Energy Consumption (kWh)")
for ax in [axes[0, 0], axes[1, 0]]:
    ax.set_ylabel(r"Max Temperature Deviation ($^\circ$C)")

# Colorbar (shared, right side)
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Training Epoch", fontsize=8)
cbar.ax.tick_params(labelsize=6.5)

fig.tight_layout(rect=[0, 0, 0.88, 1])

# Save
for out_dir in [LOCAL_FIG, THESIS_FIG]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_paper_energy_vs_temp.pdf")
    fig.savefig(out_dir / "fig_paper_energy_vs_temp.png")
    print(f"Saved to {out_dir / 'fig_paper_energy_vs_temp.pdf'}")

plt.close(fig)
