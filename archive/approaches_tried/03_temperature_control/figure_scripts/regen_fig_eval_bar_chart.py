#!/usr/bin/env python3
"""
Regenerate fig_eval_bar_chart.pdf using CSAC-LB seed 1 (best seed)
from the multi-seed benchmark, not the old single-run.

This ensures the bar chart matches Table 5 (seed 1: viol=0.056, etc.)
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
# Data — use seed 1 from multi-seed benchmark
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent.parent.parent
SEED1_DIR = BASE / "runs" / "csac_lb_multi_seed" / "seed_1" / \
    "CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}" / \
    "seed-001-2026-04-09-11-15-54"
THESIS_FIG = BASE.parent / "studienarbeiten-master" / "thesis" / "figures"
LOCAL_FIG = Path(__file__).resolve().parent.parent / "figures"

# Try eval_case_study.json first, fallback to multiseed
eval_path = SEED1_DIR / "eval_case_study.json"
if not eval_path.exists():
    eval_path = SEED1_DIR / "eval_case_study_multiseed.json"

with open(eval_path) as f:
    eval_data = json.load(f)

# Find best epoch and get the row
best_epoch = eval_data["best_epoch"]
best_row = None
rbc_row = None
for r in eval_data["rows"]:
    if r.get("epoch") == best_epoch or r.get("name", "").startswith("CSAC"):
        if best_row is None or r.get("epoch", 999) == best_epoch:
            best_row = r
    # Explicitly match "Cooling RBC" — the dumb bang-bang baseline,
    # NOT "RBC 2" or "RBC 3" which are smarter variants
    if r.get("name", "") == "Cooling RBC":
        rbc_row = r

if rbc_row is None:
    raise RuntimeError(
        "Could not find 'Cooling RBC' row in eval data. "
        f"Available names: {[r.get('name','') for r in eval_data['rows']]}"
    )

if best_row is None:
    # Find the row at best_epoch
    for r in eval_data["rows"]:
        if r.get("epoch") == best_epoch:
            best_row = r
            break
    if best_row is None:
        raise RuntimeError(f"Could not find best epoch {best_epoch} in {eval_path}")

print(f"CSAC-LB seed 1 best epoch: {best_epoch}")
print(f"  violation_rate: {best_row.get('violation_rate', '?')}")
print(f"  total_reward:   {best_row.get('total_reward', '?')}")

# ---------------------------------------------------------------------------
# Metrics to plot
# ---------------------------------------------------------------------------
metrics = [
    ("violation_rate",                     "Comfort Violation Rate", True),
    ("discomfort_rate",                    "Discomfort Proportion",  True),
    ("kpi::cost_total",                    "Cost (KPI ratio)",       True),
    ("kpi::carbon_emissions_total",        "Carbon Emissions (KPI)", True),
    ("kpi::electricity_consumption_total", "Electricity Consumption (KPI)", True),
    ("kpi::ramping_average",               "Ramping Average (KPI)", True),
]

# Try alternate key names
def get_val(row, key):
    if key in row:
        return row[key]
    # Try without kpi:: prefix
    alt = key.replace("kpi::", "")
    if alt in row:
        return row[alt]
    # Try discomfort variants
    if "discomfort" in key:
        for k in ["discomfort_rate", "discomfort_proportion",
                   "kpi::discomfort_proportion"]:
            if k in row:
                return row[k]
    return 0.0

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5))

colors_bar = ["#7f8c8d", "#C65D4B"]
methods = ["RBC", "CSAC-LB"]
rows = [rbc_row, best_row]

for idx, (key, title, lower_better) in enumerate(metrics):
    r, c = divmod(idx, 3)
    ax = axes[r, c]
    vals = [get_val(row, key) for row in rows]
    bars = ax.bar(range(len(methods)), vals, color=colors_bar,
                  edgecolor="black", linewidth=0.4, width=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_title(title)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)

    for bar, val in zip(bars, vals):
        if "rate" in key or "proportion" in key or "discomfort" in key:
            fmt = f"{val:.1%}"
        else:
            fmt = f"{val:.3f}"
        ax.annotate(fmt, xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, fontweight="bold")

fig.suptitle("Cooling-Season Evaluation: RBC vs CSAC-LB (seed 1, epoch 29)",
             fontsize=9, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])

for out_dir in [LOCAL_FIG, THESIS_FIG]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_eval_bar_chart.pdf")
    fig.savefig(out_dir / "fig_eval_bar_chart.png")
    print(f"Saved to {out_dir / 'fig_eval_bar_chart.pdf'}")

plt.close(fig)
