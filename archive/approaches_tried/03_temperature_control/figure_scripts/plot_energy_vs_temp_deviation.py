#!/usr/bin/env python3
"""
Replicates paper Fig 5 axes: Energy Consumption (kWh) vs Maximum Temperature
Deviation (°C) — ALL algorithms in a SINGLE combined scatter plot.

Each dot = one trained seed (final best-epoch evaluation).
Color + marker = algorithm. Pareto-optimal points marked with ×.
Black ★ = Cooling RBC baseline.

Source: eval_case_study_multiseed.json files (best-epoch evaluation row).

Output:
  figures/fig_energy_vs_temp_deviation_combined.pdf / .png
"""

from __future__ import annotations
from pathlib import Path
import json
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[3]   # repo root
OUT  = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# ── Eval-JSON paths (one per seed per algo) ──────────────────────────────────
ALGO_EVAL_DIRS: dict[str, list[Path]] = {
    "CSAC-LB": [
        BASE / "runs/csac_lb_multi_seed/seed_0/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-09-10-08-23",
        BASE / "runs/csac_lb_multi_seed/seed_1/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-09-11-15-54",
        BASE / "runs/csac_lb_multi_seed/seed_2/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-002-2026-04-09-12-26-54",
        BASE / "runs/csac_lb_multi_seed/seed_7/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-007-2026-04-10-16-33-51",
        BASE / "runs/csac_lb_multi_seed/seed_13/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-013-2026-04-10-17-40-31",
        BASE / "runs/csac_lb_multi_seed/seed_42/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-09-13-41-32",
    ],
    "SAC-Lag": [
        BASE / "runs/saclag_temp_cooling_only_v1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-15-33-32",
        BASE / "runs/saclag_temp_cooling_only_v1/seed_0/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-15-00-17",
        BASE / "runs/saclag_temp_cooling_only_v1/seed_1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-15-45-24",
    ],
    "CPO": [
        BASE / "runs/cpo_temp_cooling_only/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-20-54-15",
        BASE / "runs/cpo_temp_cooling_only/seed_0/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-12-34-57",
        BASE / "runs/cpo_temp_cooling_only/seed_1/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-12-58-45",
    ],
    "CUP": [
        BASE / "runs/cup_temp_cooling_only_v2/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-23-10-06",
        BASE / "runs/cup_temp_cooling_only_v2/seed_0/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-13-22-49",
        BASE / "runs/cup_temp_cooling_only_v2/seed_1/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-13-46-22",
    ],
    "FOCOPS": [
        BASE / "runs/focops_temp_cooling_only_v2/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-22-44-31",
        BASE / "runs/focops_temp_cooling_only_v2/seed_0/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-14-09-24",
        BASE / "runs/focops_temp_cooling_only_v2/seed_1/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-14-35-00",
    ],
}

ALGO_ORDER = ["CSAC-LB", "SAC-Lag", "CPO", "CUP", "FOCOPS"]

# Colors matching paper-style (one per algo)
COLORS = {
    "CSAC-LB": "#e41a1c",   # red
    "SAC-Lag": "#ff7f00",   # orange
    "CPO":     "#377eb8",   # blue
    "CUP":     "#4daf4a",   # green
    "FOCOPS":  "#984ea3",   # purple
}

MARKERS = {
    "CSAC-LB": "o",
    "SAC-Lag": "s",
    "CPO":     "^",
    "CUP":     "D",
    "FOCOPS":  "v",
}

# ── RBC / Zero reference baselines ───────────────────────────────────────────
# Extracted from eval_case_study_multiseed.json (Cooling RBC row)
RBC_ENERGY  = 7175.9    # kWh
RBC_MAX_DEV = 1.885     # °C max hot deviation

# ── IEEE style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth":     0.7,
    "axes.grid":          True,
    "grid.alpha":         0.22,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.4,
})

# ── Load eval JSON → extract best-epoch metrics ───────────────────────────────
def load_eval(run_dir: Path) -> dict | None:
    """Return best-epoch row dict from eval_case_study_multiseed.json."""
    json_path = run_dir / "eval_case_study_multiseed.json"
    if not json_path.exists():
        json_path = run_dir / "eval_case_study.json"   # fallback
    if not json_path.exists():
        print(f"  WARNING: no eval JSON in {run_dir.name}")
        return None
    with open(json_path) as f:
        d = json.load(f)
    rows = d.get("rows", [])
    # Prefer "best" row; otherwise take first
    for row in rows:
        if "best" in row.get("name", "").lower():
            return row
    return rows[0] if rows else None

def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Pareto-optimal mask: lower x (energy) AND lower y (deviation) simultaneously.
    """
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        dominated[i] = False
        if dominated.any():
            mask[i] = False
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# Build data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading evaluation data …")

algo_data: dict[str, dict] = {}   # algo → {"energy": [], "max_dev": [], "avg_dev": []}

for algo in ALGO_ORDER:
    energies, max_devs, avg_devs = [], [], []
    for run_dir in ALGO_EVAL_DIRS[algo]:
        row = load_eval(run_dir)
        if row is None:
            continue
        e  = row.get("district_import_kwh", None)
        md = row.get("kpi::discomfort_hot_delta_maximum", None)
        ad = row.get("kpi::discomfort_hot_delta_average", None)
        if e is not None and md is not None:
            energies.append(e)
            max_devs.append(md)
            avg_devs.append(ad or 0.0)
            print(f"  {algo}: energy={e:.0f} kWh, max_dev={md:.3f}°C")
    algo_data[algo] = {
        "energy":  np.array(energies),
        "max_dev": np.array(max_devs),
        "avg_dev": np.array(avg_devs),
    }

# ── Collect all RL points for Pareto computation ─────────────────────────────
all_energy  = np.concatenate([algo_data[a]["energy"]  for a in ALGO_ORDER if len(algo_data[a]["energy"]) > 0])
all_max_dev = np.concatenate([algo_data[a]["max_dev"] for a in ALGO_ORDER if len(algo_data[a]["max_dev"]) > 0])
all_labels  = np.concatenate([[a] * len(algo_data[a]["energy"]) for a in ALGO_ORDER if len(algo_data[a]["energy"]) > 0])

pm = pareto_mask(all_energy, all_max_dev)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE  — Combined scatter: all algorithms in ONE plot
# X = Energy Consumption (kWh)
# Y = Maximum Temperature Deviation (°C)
# ══════════════════════════════════════════════════════════════════════════════
print("Building combined scatter …")

fig, ax = plt.subplots(figsize=(5.0, 3.8))

offset = 0   # pointer into all_energy for Pareto labels
for algo in ALGO_ORDER:
    e   = algo_data[algo]["energy"]
    md  = algo_data[algo]["max_dev"]
    n   = len(e)
    if n == 0:
        continue

    color  = COLORS[algo]
    marker = MARKERS[algo]

    # Individual seed dots
    ax.scatter(e, md,
               c=color, marker=marker, s=55,
               alpha=0.85, linewidths=0.5, edgecolors="white",
               zorder=4, label=algo)

    # Mark Pareto-optimal seeds with ×
    seg_pm = pm[offset : offset + n]
    if seg_pm.any():
        ax.scatter(e[seg_pm], md[seg_pm],
                   marker="x", s=80, c=color,
                   linewidths=1.5, zorder=6)

    # Mean ± std error bars
    if n > 1:
        mu_e, sd_e = e.mean(), e.std()
        mu_md, sd_md = md.mean(), md.std()
        ax.errorbar(mu_e, mu_md,
                    xerr=sd_e, yerr=sd_md,
                    fmt="none", color=color, capsize=3,
                    capthick=0.9, elinewidth=0.9,
                    alpha=0.70, zorder=5)
        # Larger filled mean marker
        ax.scatter([mu_e], [mu_md],
                   c=color, marker=marker, s=120,
                   linewidths=0.8, edgecolors="black",
                   zorder=7)

    offset += n

# RBC baseline star
ax.scatter([RBC_ENERGY], [RBC_MAX_DEV], marker="*", s=260,
           color="black", zorder=8, label="Cooling RBC")

# Axis labels and limits
ax.set_xlabel("Energy Consumption (kWh)", fontsize=9)
ax.set_ylabel("Maximum Temperature Deviation (°C)", fontsize=9)
ax.set_title(
    "Energy Consumption vs. Maximum Temperature Deviation\n"
    "(× = Pareto-optimal seeds; large markers = mean ± std over seeds)",
    fontsize=9, pad=5,
)

# Legend — outside to the right
legend = ax.legend(
    loc="upper right", fontsize=7.5, framealpha=0.85,
    handletextpad=0.3, borderpad=0.45, labelspacing=0.30,
    handlelength=1.3,
)

fig.tight_layout()

# Save
for ext in ("pdf", "png"):
    p = OUT / f"fig_energy_vs_temp_deviation_combined.{ext}"
    fig.savefig(p)
    print(f"  Saved {p.name}")

plt.close(fig)
print("Done.")
