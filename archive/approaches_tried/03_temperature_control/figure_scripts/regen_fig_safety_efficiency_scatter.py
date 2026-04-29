#!/usr/bin/env python3
"""
Publication-quality safety–efficiency scatter plot.
Each point = one seed's best checkpoint.
Replaces fig_multiseed_robustness (box plots) and fig_eval_bar_chart (bars).

X-axis: Comfort Violation Rate (lower = safer)
Y-axis: Total Episode Reward (higher = more efficient)
"""
import json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

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
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.20,
    "grid.linestyle": "-",
    "mathtext.fontset": "dejavuserif",
})

# ---------------------------------------------------------------------------
# Colors and markers (colorblind-friendly, distinct shapes)
# ---------------------------------------------------------------------------
ALGO_STYLE = {
    "CSAC-LB": {"color": "#D62728", "marker": "o", "label": "CSAC-LB ($n$=6)"},
    "CPO":     {"color": "#1F77B4", "marker": "s", "label": "CPO ($n$=3)"},
    "SAC-Lag": {"color": "#9467BD", "marker": "^", "label": "SAC-Lag ($n$=3)"},
    "CUP":     {"color": "#17BECF", "marker": "D", "label": "CUP ($n$=3)"},
    "FOCOPS":  {"color": "#FF7F0E", "marker": "v", "label": "FOCOPS ($n$=3)"},
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
BASE = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs")
THESIS_FIG = Path("/home/christmas/studienarbeiten-master/thesis/figures")
LOCAL_FIG = Path(__file__).resolve().parent.parent / "figures"

# CSAC-LB seeds
csac_base = BASE / "csac_lb_multi_seed"
all_data = {}

csac_seeds = []
for seed_name in ["seed_0", "seed_1", "seed_2", "seed_7", "seed_13", "seed_42"]:
    for root, dirs, files in os.walk(str(csac_base / seed_name)):
        for fname in ["eval_case_study.json", "eval_case_study_multiseed.json"]:
            fpath = os.path.join(root, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    d = json.load(f)
                for r in d["rows"]:
                    if "best" in r.get("name", "").lower():
                        csac_seeds.append((r["violation_rate"] * 100, r["total_reward"]))
                        break
                break
all_data["CSAC-LB"] = csac_seeds

# Other algorithms from summary
with open(str(BASE / "multiseed_eval_summary.json")) as f:
    summary = json.load(f)
for algo in ["CPO", "SAC-Lag", "CUP", "FOCOPS"]:
    seeds = []
    for s in summary[algo]["per_seed"]:
        seeds.append((s["violation_rate"] * 100,
                       s.get("total_reward", s.get("reward", 0))))
    all_data[algo] = seeds

# RBC baseline
rbc_viol, rbc_reward = 51.6, -448.7

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 4.0))

# Plot each algorithm
for algo in ["CSAC-LB", "CPO", "SAC-Lag", "CUP", "FOCOPS"]:
    style = ALGO_STYLE[algo]
    pts = np.array(all_data[algo])
    viol, rew = pts[:, 0], pts[:, 1]

    # Individual seeds
    ax.scatter(
        viol, rew,
        color=style["color"], marker=style["marker"],
        s=55, zorder=5, edgecolors="white", linewidths=0.5,
        label=style["label"], alpha=0.9,
    )

    # Mean crosshair (thin lines, no large marker)
    mv, mr = np.mean(viol), np.mean(rew)
    ax.plot([mv, mv], [mr - 60, mr + 60], color=style["color"],
            lw=0.6, zorder=4, alpha=0.6)
    ax.plot([mv - 1.5, mv + 1.5], [mr, mr], color=style["color"],
            lw=0.6, zorder=4, alpha=0.6)

# RBC baseline
ax.scatter(
    rbc_viol, rbc_reward,
    color="#666666", marker="*", s=120, zorder=5,
    edgecolors="black", linewidths=0.4,
)
ax.annotate(
    "Cooling RBC",
    xy=(rbc_viol, rbc_reward),
    xytext=(rbc_viol - 8, rbc_reward + 250),
    fontsize=6, color="#666666", fontstyle="italic",
    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5),
    ha="center",
)

# Axes
ax.set_xlabel("Comfort Violation Rate (%)")
ax.set_ylabel("Total Episode Reward")
ax.set_xlim(-3, 100)
ax.set_ylim(min(rbc_reward - 200, -700), None)

# Legend — IEEE style: outside data area, single column, serif font
handles = []
for algo in ["CSAC-LB", "CPO", "SAC-Lag", "CUP", "FOCOPS"]:
    s = ALGO_STYLE[algo]
    handles.append(Line2D(
        [0], [0], marker=s["marker"], color=s["color"],
        markerfacecolor=s["color"], markeredgecolor="black",
        markeredgewidth=0.5, markersize=7,
        label=s["label"], linestyle="None",
    ))
handles.append(Line2D(
    [0], [0], marker="*", color="#666666",
    markerfacecolor="#666666", markeredgecolor="black",
    markeredgewidth=0.5, markersize=9,
    label="Cooling RBC", linestyle="None",
))

ax.legend(
    handles=handles,
    loc="upper right",
    frameon=True, framealpha=0.95,
    edgecolor="#999999",
    borderpad=0.6,
    handletextpad=0.5,
    labelspacing=0.4,
    ncol=1,
    fancybox=False,
)

ax.grid(True)
fig.tight_layout()

# Save
for out_dir in [LOCAL_FIG, THESIS_FIG]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_safety_efficiency_scatter.pdf")
    fig.savefig(out_dir / "fig_safety_efficiency_scatter.png")
    print(f"Saved to {out_dir / 'fig_safety_efficiency_scatter.pdf'}")

plt.close(fig)
