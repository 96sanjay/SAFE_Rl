#!/usr/bin/env python3
"""
Publication-quality stress test figure — horizontal grouped bars.
Fixes: title wrapping, inconsistent fonts, cramped x-labels.
Uses horizontal bars (Nweye style) instead of vertical bars.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style (matching thesis figures)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.25,
    "grid.alpha": 0.20,
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
BASE = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs")
STRESS_JSON = BASE / "stress_test_72h_all_algos.json"
THESIS_FIG = Path("/home/christmas/studienarbeiten-master/thesis/figures")
LOCAL_FIG = Path(__file__).resolve().parent.parent / "figures"

with open(STRESS_JSON) as f:
    raw = json.load(f)

# Parse per_algorithm: keys like "CSAC-LB (seed 1) [best (ep 29)]"
# Keep only [best ...] entries, flatten kpis dict into top level
pa = raw.get("per_algorithm", raw)
best = {}
ALGO_NAME_MAP = {
    "CSAC-LB (seed 1)": "CSAC-LB",
    "CPO": "CPO",
    "SAC-Lag": "SAC-Lag",
    "PPO-Lag (tight v2)": "PPO-Lag",
    "PPO (unconstrained BC)": "PPO",
    "CUP": "CUP",
    "FOCOPS": "FOCOPS",
    "Cooling RBC": "Cooling RBC",
}

for key, val_list in pa.items():
    if "[best" not in key and key not in ["Cooling RBC", "Zero (no cooling)"]:
        continue
    if not isinstance(val_list, list) or not val_list:
        continue
    entry = val_list[0]
    # Map algo name
    raw_name = entry.get("algo_name", key.split("[")[0].strip())
    algo = ALGO_NAME_MAP.get(raw_name, raw_name)
    # Flatten kpis dict
    flat = dict(entry)
    if "kpis" in flat and isinstance(flat["kpis"], dict):
        for kk, vv in flat["kpis"].items():
            flat[f"kpi::{kk}"] = vv
    best[algo] = flat

print(f"Algorithms found: {list(best.keys())}")

# Algorithm order
ALGO_ORDER = ["CSAC-LB", "CPO", "SAC-Lag", "PPO-Lag", "PPO", "CUP", "FOCOPS", "Cooling RBC"]
ALGO_ORDER = [a for a in ALGO_ORDER if a in best and a != "Zero"]

COLORS = {
    "CSAC-LB": "#ff7f0e", "CPO": "#1f77b4", "SAC-Lag": "#d62728",
    "PPO-Lag": "#9467bd", "PPO": "#e377c2", "CUP": "#17becf",
    "FOCOPS": "#8c564b", "Cooling RBC": "#7f7f7f",
}

# Metrics
METRICS = [
    ("violation_rate",              "Comfort Violation Rate (%)",     100, "{:.1f}"),
    ("kpi::cost_total",             "Electricity Cost (normalised)",    1, "{:.2f}"),
    ("kpi::carbon_emissions_total", "Carbon Emissions (normalised)",   1, "{:.2f}"),
    ("kpi::ramping_average",        "Ramping (normalised)",             1, "{:.2f}"),
]

# ---------------------------------------------------------------------------
# Figure: 2x2 grid of horizontal bar charts
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

panel_labels = ["(a)", "(b)", "(c)", "(d)"]

for idx, (mkey, mlabel, scale, vfmt) in enumerate(METRICS):
    ax = axes_flat[idx]
    vals = []
    colors = []
    labels = []

    for algo in reversed(ALGO_ORDER):  # reversed so top = first algorithm
        v = best[algo].get(mkey, 0)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0
        vals.append(v * scale)
        colors.append(COLORS.get(algo, "#999999"))
        labels.append(algo)

    y = np.arange(len(labels))
    bars = ax.barh(y, vals, height=0.6, color=colors,
                   edgecolor="white", linewidth=0.3)

    # Value annotations
    xmax = max(vals) if vals else 1
    for yi, v in zip(y, vals):
        ax.text(v + xmax * 0.02, yi, vfmt.format(v),
                va="center", ha="left", fontsize=6.5, fontweight="bold",
                color="#333333")

    # Baseline reference line at 1.0 for normalised metrics
    if scale == 1:
        ax.axvline(1.0, color="#888888", ls="--", lw=0.6, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_title(f"{panel_labels[idx]} {mlabel}", fontweight="bold", pad=4)
    ax.set_xlim(0, xmax * 1.20)
    ax.grid(axis="x")

fig.suptitle("72-h Heat Wave Stress Test ($2.5\\times$ Cooling Demand)",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()

# Save
for out_dir in [LOCAL_FIG, THESIS_FIG]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_stress_test_multimetric.pdf")
    fig.savefig(out_dir / "fig_stress_test_multimetric.png")
    print(f"Saved to {out_dir / 'fig_stress_test_multimetric.pdf'}")

plt.close(fig)
