#!/usr/bin/env python3
"""
Publication-quality 2-panel training curves with multi-seed shaded bands.
(a) Episode Return — mean ± std across seeds
(b) Episode Cost   — mean ± std across seeds

Replaces the 5-panel single-run diagnostics figure.
Only plots the two metrics that matter for a thesis: what the agent
achieves (return) and how safe it is (cost). Internal optimizer
diagnostics (entropy, value loss, policy loss) are dropped.
"""
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "lines.linewidth": 1.8,
})

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
BASE = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs")
THESIS_FIG = Path("/home/christmas/studienarbeiten-master/thesis/figures")
LOCAL_FIG = Path(__file__).resolve().parent.parent / "figures"

# Algorithm run directories (all seeds)
ALGO_SEEDS = {
    "CSAC-LB": [
        BASE / "csac_lb_multi_seed" / f"seed_{s}" for s in [0, 1, 2, 7, 13, 42]
    ],
    "CPO": [
        BASE / "cpo_temp_cooling_only" / "seed_0",
        BASE / "cpo_temp_cooling_only" / "seed_1",
        BASE / "cpo_temp_cooling_only" / "CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}",
    ],
    "SAC-Lag": [
        BASE / "saclag_temp_cooling_only_v1" / "seed_0",
        BASE / "saclag_temp_cooling_only_v1" / "seed_1",
        BASE / "saclag_temp_cooling_only_v1" / "SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}",
    ],
    "CUP": [
        BASE / "cup_temp_cooling_only_v2" / "seed_0",
        BASE / "cup_temp_cooling_only_v2" / "seed_1",
        BASE / "cup_temp_cooling_only_v2" / "CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}",
    ],
    "FOCOPS": [
        BASE / "focops_temp_cooling_only_v2" / "seed_0",
        BASE / "focops_temp_cooling_only_v2" / "seed_1",
        BASE / "focops_temp_cooling_only_v2" / "FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}",
    ],
}

# Style — matching Fig 4.1/4.2 palette (Zhang et al. paper style)
COLORS = {
    "CSAC-LB": "#ff7f0e",   # orange (paper style)
    "SAC-Lag": "#d62728",   # red (paper style)
    "CPO":     "#1f77b4",   # blue (paper style)
    "CUP":     "#17becf",   # cyan
    "FOCOPS":  "#8c564b",   # brown
}
LINESTYLES = {
    "CSAC-LB": "-",
    "CPO":     "-",
    "SAC-Lag": "-",
    "CUP":     "--",
    "FOCOPS":  "--",
}
ALGO_ORDER = ["CSAC-LB", "CPO", "SAC-Lag", "CUP", "FOCOPS"]

# ---------------------------------------------------------------------------
# Load multi-seed data
# ---------------------------------------------------------------------------
def find_progress_csvs(seed_dirs):
    """Walk seed directories and find all progress.csv files."""
    csvs = []
    for sd in seed_dirs:
        for root, dirs, files in os.walk(str(sd)):
            if "progress.csv" in files:
                csvs.append(os.path.join(root, "progress.csv"))
    return csvs


def load_multi_seed(algo):
    """Load EpRet and EpCost across all seeds, aligned by epoch."""
    csvs = find_progress_csvs(ALGO_SEEDS[algo])
    all_ret = []
    all_cost = []

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        epoch_col = "Train/Epoch"
        ret_col = "Metrics/EpRet"
        cost_col = "Metrics/EpCost"

        if epoch_col not in df.columns:
            continue

        epochs = df[epoch_col].values
        ret = df[ret_col].values if ret_col in df.columns else None
        cost = df[cost_col].values if cost_col in df.columns else None

        if ret is not None:
            all_ret.append((epochs, ret))
        if cost is not None:
            all_cost.append((epochs, cost))

    return all_ret, all_cost


def compute_mean_std(traces, max_epoch=60):
    """Align traces to common epoch grid and compute mean ± std."""
    epoch_grid = np.arange(0, max_epoch + 1)
    aligned = []

    for epochs, values in traces:
        # Interpolate to common grid
        interp = np.interp(epoch_grid, epochs, values,
                           left=values[0], right=values[-1])
        aligned.append(interp)

    if not aligned:
        return epoch_grid, np.zeros_like(epoch_grid), np.zeros_like(epoch_grid)

    aligned = np.array(aligned)
    mean = np.mean(aligned, axis=0)
    std = np.std(aligned, axis=0)
    return epoch_grid, mean, std


# ---------------------------------------------------------------------------
# Training phase definitions (Nweye-style background bands)
# ---------------------------------------------------------------------------
PHASES = [
    (0, 10,  "#fff3e0", "Exploration"),              # light orange
    (10, 30, "#e8f5e9", "Constraint\nLearning"),     # light green — use newline for spacing
    (30, 60, "#e3f2fd", "Convergence"),              # light blue
]

# ---------------------------------------------------------------------------
# Generate two separate figures for LaTeX subfigure
# ---------------------------------------------------------------------------
def make_panel(ax, ylabel):
    """Add phase bands and labels to a single panel."""
    for start, end, color, label in PHASES:
        ax.axvspan(start, end, facecolor=color, alpha=0.45, zorder=0)
    for start, end, color, label in PHASES:
        mid = (start + end) / 2
        ax.text(mid, 1.04, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color="#444444",
                fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.set_xlim(0, 60)


# Also generate the combined 2-panel version
fig, (ax_ret, ax_cost) = plt.subplots(1, 2, figsize=(10.0, 4.2))

for ax in [ax_ret, ax_cost]:
    for start, end, color, label in PHASES:
        ax.axvspan(start, end, facecolor=color, alpha=0.45, zorder=0)
    for start, end, color, label in PHASES:
        mid = (start + end) / 2
        ax.text(mid, 1.04, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="#444444",
                fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7))

# Plot algorithms with markers at key epochs
MARKERS = {"CSAC-LB": "o", "CPO": "s", "SAC-Lag": "^", "CUP": "D", "FOCOPS": "v"}
KEY_EPOCHS = [0, 5, 10, 15, 20, 30, 40, 50, 60]

for algo in ALGO_ORDER:
    ret_traces, cost_traces = load_multi_seed(algo)
    n_seeds = len(ret_traces)
    col = COLORS[algo]
    ls = LINESTYLES[algo]
    mk = MARKERS[algo]

    # Episode Return
    if ret_traces:
        epochs, mean, std = compute_mean_std(ret_traces)
        ax_ret.plot(epochs, mean, color=col, ls=ls, lw=1.8,
                    label=f"{algo} ($n$={n_seeds})", zorder=3)
        ax_ret.fill_between(epochs, mean - std, mean + std,
                            color=col, alpha=0.15, zorder=2)
        # Markers at key epochs
        key_idx = [e for e in KEY_EPOCHS if e <= epochs[-1]]
        key_vals = np.interp(key_idx, epochs, mean)
        ax_ret.scatter(key_idx, key_vals, color=col, marker=mk,
                       s=20, zorder=5, edgecolors="white", linewidths=0.3)

    # Episode Cost
    if cost_traces:
        epochs, mean, std = compute_mean_std(cost_traces)
        ax_cost.plot(epochs, mean, color=col, ls=ls, lw=1.8,
                     label=f"{algo} ($n$={n_seeds})", zorder=3)
        ax_cost.fill_between(epochs, mean - std, mean + std,
                             color=col, alpha=0.15, zorder=2)
        key_idx = [e for e in KEY_EPOCHS if e <= epochs[-1]]
        key_vals = np.interp(key_idx, epochs, mean)
        ax_cost.scatter(key_idx, key_vals, color=col, marker=mk,
                        s=20, zorder=5, edgecolors="white", linewidths=0.3)

# Panel (a) formatting — no title (LaTeX subcaption handles it)
ax_ret.set_xlabel("Epoch")
ax_ret.set_ylabel("Episode Return")
ax_ret.legend(loc="lower right", framealpha=0.95, edgecolor="#cccccc",
              ncol=1, handlelength=1.8, fontsize=7.5)
ax_ret.grid(True, axis="y")
ax_ret.set_xlim(0, 60)

# Panel (b) formatting — no title
ax_cost.set_xlabel("Epoch")
ax_cost.set_ylabel("Episode Cost")
ax_cost.legend(loc="upper right", framealpha=0.95, edgecolor="#cccccc",
               ncol=1, handlelength=1.8, fontsize=7.5)
ax_cost.grid(True, axis="y")
ax_cost.set_xlim(0, 60)

fig.tight_layout(w_pad=2.5)

# Save combined version
for out_dir in [LOCAL_FIG, THESIS_FIG]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_training_diagnostics.pdf")
    print(f"Saved combined to {out_dir / 'fig_training_diagnostics.pdf'}")
plt.close(fig)

# --- Save individual panels for LaTeX subfigure ---
for panel_name, ylabel, trace_key in [("return", "Episode Return", "ret"),
                                        ("cost", "Episode Cost", "cost")]:
    fig_s, ax_s = plt.subplots(figsize=(5.0, 3.8))
    make_panel(ax_s, ylabel)

    for algo in ALGO_ORDER:
        ret_traces, cost_traces = load_multi_seed(algo)
        traces = ret_traces if trace_key == "ret" else cost_traces
        n_seeds = len(traces)
        col = COLORS[algo]
        ls = LINESTYLES[algo]
        mk = MARKERS[algo]

        if traces:
            epochs, mean, std = compute_mean_std(traces)
            ax_s.plot(epochs, mean, color=col, ls=ls, lw=1.8,
                      label=f"{algo} ($n$={n_seeds})", zorder=3)
            ax_s.fill_between(epochs, mean - std, mean + std,
                              color=col, alpha=0.15, zorder=2)
            key_idx = [e for e in KEY_EPOCHS if e <= epochs[-1]]
            key_vals = np.interp(key_idx, epochs, mean)
            ax_s.scatter(key_idx, key_vals, color=col, marker=mk,
                         s=22, zorder=5, edgecolors="white", linewidths=0.3)

    loc = "lower right" if trace_key == "ret" else "upper right"
    ax_s.legend(loc=loc, framealpha=0.95, edgecolor="#cccccc",
                handlelength=1.8, fontsize=8)
    fig_s.tight_layout()

    for out_dir in [LOCAL_FIG, THESIS_FIG]:
        fig_s.savefig(out_dir / f"fig_training_{panel_name}.pdf")
        print(f"Saved {panel_name} to {out_dir / f'fig_training_{panel_name}.pdf'}")
    plt.close(fig_s)
