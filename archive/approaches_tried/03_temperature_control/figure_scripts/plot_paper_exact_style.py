#!/usr/bin/env python3
"""
Exact replication of Zhang et al. (CSAC-LB paper, arXiv:2409.19716) figure
style, adapted to our CityLearn temperature-control benchmark.

Produces 3 figures matching the paper EXACTLY:

  Figure 5  — Energy Consumption vs Max Temperature Deviation  (paper Fig 5)
              Scatter: each dot = one checkpoint evaluation, color = training
              step (rainbow colormap). × on Pareto front. ★ = Rule-based.
              ALL algorithms in ONE single plot.

  Figure 6a — Yearly Energy Consumption vs Training Steps  (paper Fig 6 top)
              Mean ± std bands across seeds. ALL algorithms overlaid.

  Figure 6b — Temperature Deviation vs Training Steps  (paper Fig 6 bottom)
              Dual y-axis: average deviation (solid, left) + maximum deviation
              (dashed, right). Mean ± std bands. ALL algorithms overlaid.

Data source: docs/temperature_case_study/data/all_checkpoint_evals.json
             (produced by scripts/batch_eval_all_checkpoints.py)

Fallback: If checkpoint eval JSON is not ready, uses progress.csv (EpRet/EpCost
          proxy metrics) and eval_case_study_multiseed.json (final points only).

Style: IEEE double-column, serif font (matching paper exactly).
"""

from __future__ import annotations
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[3]
OUT  = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)

CHECKPOINT_JSON = BASE / "docs" / "temperature_case_study" / "data" / "all_checkpoint_evals.json"

# ── Run directories (for progress.csv fallback) ──────────────────────────────
ALGO_RUN_DIRS: dict[str, dict[int, Path]] = {
    "CSAC-LB": {
        0:  BASE / "runs/csac_lb_multi_seed/seed_0/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-09-10-08-23",
        1:  BASE / "runs/csac_lb_multi_seed/seed_1/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-09-11-15-54",
        2:  BASE / "runs/csac_lb_multi_seed/seed_2/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-002-2026-04-09-12-26-54",
        7:  BASE / "runs/csac_lb_multi_seed/seed_7/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-007-2026-04-10-16-33-51",
        13: BASE / "runs/csac_lb_multi_seed/seed_13/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-013-2026-04-10-17-40-31",
        42: BASE / "runs/csac_lb_multi_seed/seed_42/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-09-13-41-32",
    },
    "SAC-Lag": {
        42: BASE / "runs/saclag_temp_cooling_only_v1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-15-33-32",
        0:  BASE / "runs/saclag_temp_cooling_only_v1/seed_0/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-15-00-17",
        1:  BASE / "runs/saclag_temp_cooling_only_v1/seed_1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-15-45-24",
    },
    "CPO": {
        42: BASE / "runs/cpo_temp_cooling_only/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-20-54-15",
        0:  BASE / "runs/cpo_temp_cooling_only/seed_0/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-12-34-57",
        1:  BASE / "runs/cpo_temp_cooling_only/seed_1/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-12-58-45",
    },
}

ALGO_ORDER = ["CSAC-LB", "SAC-Lag", "CPO"]

# Steps per epoch (from progress.csv TotalEnvSteps / epoch)
STEPS_PER_EPOCH = 2208

# ── Colors matching paper Fig 6 ──────────────────────────────────────────────
COLORS = {
    "CPO":     "#1f77b4",   # blue
    "SAC-Lag": "#d62728",   # red
    "CSAC-LB": "#ff7f0e",   # orange
}

MARKERS_SCATTER = {
    "CPO":     "^",
    "SAC-Lag": "s",
    "CSAC-LB": "o",
}

# ── RBC baseline (from eval JSON) ───────────────────────────────────────────
RBC_ENERGY    = 7175.9       # district_import_kwh
RBC_MAX_DEV   = 1.885        # max hot deviation (°C)
RBC_AVG_DEV   = 0.074        # avg hot deviation (°C)

# ══════════════════════════════════════════════════════════════════════════════
# EXACT paper style  (matching Zhang et al. IEEE submission)
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth":     0.8,
    "axes.grid":          False,
    "lines.linewidth":    1.5,
    "lines.markersize":   4,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
})

# ── Helpers ──────────────────────────────────────────────────────────────────
def pareto_mask_lower(x, y):
    """Pareto mask: lower x AND lower y = better."""
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dom = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        dom[i] = False
        if dom.any():
            mask[i] = False
    return mask

def save(fig, name):
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p)
        print(f"  Saved {p.name}")


# ── Load checkpoint evaluation data ─────────────────────────────────────────
def load_checkpoint_data() -> dict[str, list[dict]] | None:
    """Load all_checkpoint_evals.json, return {algo: [rows sorted by epoch]}."""
    if not CHECKPOINT_JSON.exists():
        print(f"  Checkpoint JSON not found: {CHECKPOINT_JSON}")
        return None

    with open(CHECKPOINT_JSON) as f:
        raw = json.load(f)

    # Group by algorithm
    by_algo: dict[str, list[dict]] = {a: [] for a in ALGO_ORDER}
    for key, row in raw.items():
        algo = row.get("algo", "")
        if algo in by_algo:
            by_algo[algo].append(row)

    # Sort by epoch within each algo
    for algo in by_algo:
        by_algo[algo].sort(key=lambda r: (r.get("seed", 0), r.get("epoch", 0)))

    total = sum(len(v) for v in by_algo.values())
    print(f"  Loaded {total} checkpoint evaluations from JSON")
    for algo in ALGO_ORDER:
        n = len(by_algo[algo])
        seeds = set(r["seed"] for r in by_algo[algo])
        print(f"    {algo}: {n} evaluations across seeds {sorted(seeds)}")
    return by_algo


def aggregate_by_epoch(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate metric across seeds at each epoch → (epochs, mean, std)."""
    # Group by epoch
    by_epoch: dict[int, list[float]] = {}
    for r in rows:
        ep = r["epoch"]
        val = r.get(metric)
        if val is not None:
            by_epoch.setdefault(ep, []).append(float(val))

    if not by_epoch:
        return np.array([]), np.array([]), np.array([])

    epochs = sorted(by_epoch.keys())
    means = np.array([np.mean(by_epoch[e]) for e in epochs])
    stds  = np.array([np.std(by_epoch[e]) for e in epochs])
    return np.array(epochs), means, stds


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
ckpt_data = load_checkpoint_data()

if ckpt_data is None or sum(len(v) for v in ckpt_data.values()) < 50:
    print("\n*** Insufficient checkpoint data — batch evaluation may still be running. ***")
    print("*** Run:  python scripts/batch_eval_all_checkpoints.py  ***")
    print("*** Then re-run this plot script. ***\n")
    # Still produce fallback plots from progress.csv
    USE_CHECKPOINT = False
else:
    USE_CHECKPOINT = True


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5  —  Energy Consumption vs Max Temperature Deviation
#              (Paper Fig 5 style: rainbow scatter, × on Pareto, ★ baseline)
#              ALL algorithms in ONE single plot
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 5 — Energy vs Max Temperature Deviation …")

fig5, ax5 = plt.subplots(figsize=(5.5, 4.2))

if USE_CHECKPOINT:
    # ── Dense scatter from checkpoint evals ──────────────────────────────
    # Collect all points
    all_energy = []
    all_max_dev = []
    all_steps = []
    all_algo_labels = []

    for algo in ALGO_ORDER:
        for row in ckpt_data[algo]:
            e  = row.get("district_import_kwh")
            md = row.get("max_hot_dev")
            ep = row.get("epoch", 0)
            if e is not None and md is not None:
                all_energy.append(float(e))
                all_max_dev.append(float(md))
                all_steps.append(ep * STEPS_PER_EPOCH)
                all_algo_labels.append(algo)

    all_energy = np.array(all_energy)
    all_max_dev = np.array(all_max_dev)
    all_steps = np.array(all_steps)

    # Rainbow colormap by training step (matching paper exactly)
    max_step = all_steps.max() if len(all_steps) > 0 else 1.0
    norm = mcolors.Normalize(vmin=0, vmax=max_step)
    cmap = plt.get_cmap("rainbow")

    # Plot each algorithm with its marker shape, colored by training step
    for algo in ALGO_ORDER:
        mask = np.array([a == algo for a in all_algo_labels])
        if not mask.any():
            continue
        e_a  = all_energy[mask]
        md_a = all_max_dev[mask]
        s_a  = all_steps[mask]

        sc = ax5.scatter(e_a, md_a,
                         c=s_a, cmap=cmap, norm=norm,
                         marker=MARKERS_SCATTER[algo],
                         s=35, alpha=0.80,
                         linewidths=0.3, edgecolors="grey",
                         zorder=4, label=algo)

    # Pareto front — × marks
    pm = pareto_mask_lower(all_energy, all_max_dev)
    if pm.any():
        ax5.scatter(all_energy[pm], all_max_dev[pm],
                    marker="x", s=80, c="black",
                    linewidths=1.5, zorder=6)

    # Colorbar
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig5.colorbar(sm, ax=ax5, pad=0.02, aspect=30)
    cbar.set_label("Epoch", fontsize=9)
    # Format colorbar ticks as epoch numbers
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v / STEPS_PER_EPOCH)}"))

else:
    # ── Fallback: final eval points only ─────────────────────────────────
    for algo in ALGO_ORDER:
        energies, max_devs = [], []
        for run_dir in ALGO_RUN_DIRS[algo].values():
            for name in ("eval_case_study_multiseed.json", "eval_case_study.json"):
                p = run_dir / name
                if p.exists():
                    with open(p) as f:
                        d = json.load(f)
                    for row in d.get("rows", []):
                        if "best" in row.get("name", "").lower():
                            e  = row.get("district_import_kwh")
                            md = row.get("kpi::discomfort_hot_delta_maximum")
                            if e is not None and md is not None:
                                energies.append(e)
                                max_devs.append(md)
                            break
                    break
        if energies:
            ax5.scatter(energies, max_devs,
                        c=COLORS[algo], marker=MARKERS_SCATTER[algo],
                        s=60, alpha=0.85, linewidths=0.5,
                        edgecolors="white", zorder=4, label=algo)

    # Pareto (no dense data available)
    print("  (Fallback mode — sparse dots only)")

# RBC baseline — black star
ax5.scatter([RBC_ENERGY], [RBC_MAX_DEV], marker="*", s=260,
            color="black", zorder=8, label="Rule-based")

ax5.set_xlabel("Energy Consumption (kWh)", fontsize=10)
ax5.set_ylabel(r"Maximum Temperature Deviation ($^\circ$C)", fontsize=10)
ax5.legend(loc="upper left", fontsize=8, framealpha=0.85,
           handletextpad=0.3, borderpad=0.45, labelspacing=0.35)

fig5.tight_layout()
save(fig5, "fig_paper_energy_vs_temp")
plt.close(fig5)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6a  —  Yearly Energy Consumption vs Training Steps
#               (Paper Fig 6 top row: mean ± std bands, all algos overlaid)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 6a — Energy Consumption vs Training Steps …")

fig6a, ax6a = plt.subplots(figsize=(5.5, 3.5))

if USE_CHECKPOINT:
    for algo in ALGO_ORDER:
        epochs, mu, sd = aggregate_by_epoch(ckpt_data[algo], "district_import_kwh")
        if len(epochs) == 0:
            continue
        steps = epochs * STEPS_PER_EPOCH
        ax6a.plot(steps, mu, color=COLORS[algo], lw=1.5, label=algo)
        ax6a.fill_between(steps, mu - sd, mu + sd,
                          color=COLORS[algo], alpha=0.15)
else:
    # Fallback: EpRet from progress.csv
    print("  (Fallback: using EpRet proxy from progress.csv)")
    OFFPOLICY = {"CSAC-LB"}
    for algo in ALGO_ORDER:
        ret_col = "Metrics/TestEpRet" if algo in OFFPOLICY else "Metrics/EpRet"
        arrays, step_arrays = [], []
        for run_dir in ALGO_RUN_DIRS[algo].values():
            csv = run_dir / "progress.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                if ret_col in df.columns:
                    arrays.append(df[ret_col].values.astype(float))
                    step_arrays.append(df["TotalEnvSteps"].values.astype(float))
        if arrays:
            min_len = min(len(a) for a in arrays)
            mat = np.stack([a[:min_len] for a in arrays])
            s = step_arrays[0][:min_len]
            ax6a.plot(s, mat.mean(0), color=COLORS[algo], lw=1.5, label=algo)
            ax6a.fill_between(s, mat.mean(0) - mat.std(0), mat.mean(0) + mat.std(0),
                              color=COLORS[algo], alpha=0.15)

# RBC baseline — black line
ax6a.axhline(RBC_ENERGY, color="black", lw=1.5, ls="-", label="Rule-based")

ax6a.set_xlabel("Epoch", fontsize=10)
ax6a.set_ylabel("Energy Consumption (kWh)", fontsize=10)
ax6a.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v / STEPS_PER_EPOCH:.0f}"))
ax6a.set_xlim(left=0)
ax6a.legend(loc="best", fontsize=8, ncol=2)

fig6a.tight_layout()
save(fig6a, "fig_paper_energy_vs_steps")
plt.close(fig6a)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6b  —  Temperature Deviation vs Training Steps
#               (Paper Fig 6 bottom row: dual y-axis)
#               Left y-axis: Average Temperature Deviation (solid)
#               Right y-axis: Maximum Temperature Deviation (dashed)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 6b — Temperature Deviation vs Training Steps …")

fig6b, ax_avg = plt.subplots(figsize=(5.5, 3.5))
ax_max = ax_avg.twinx()

if USE_CHECKPOINT:
    for algo in ALGO_ORDER:
        # Average deviation (solid line, left y-axis)
        epochs_avg, mu_avg, sd_avg = aggregate_by_epoch(ckpt_data[algo], "avg_hot_dev")
        if len(epochs_avg) > 0:
            steps_avg = epochs_avg * STEPS_PER_EPOCH
            line_avg, = ax_avg.plot(steps_avg, mu_avg, color=COLORS[algo],
                                    lw=1.5, ls="-", label=f"{algo}")
            ax_avg.fill_between(steps_avg, mu_avg - sd_avg, mu_avg + sd_avg,
                                color=COLORS[algo], alpha=0.10)

        # Maximum deviation (dashed line, right y-axis)
        epochs_max, mu_max, sd_max = aggregate_by_epoch(ckpt_data[algo], "max_hot_dev")
        if len(epochs_max) > 0:
            steps_max = epochs_max * STEPS_PER_EPOCH
            ax_max.plot(steps_max, mu_max, color=COLORS[algo],
                        lw=1.2, ls="--")
            ax_max.fill_between(steps_max, mu_max - sd_max, mu_max + sd_max,
                                color=COLORS[algo], alpha=0.08)

    # RBC baselines
    ax_avg.axhline(RBC_AVG_DEV, color="black", lw=1.5, ls="-", label="Rule-based")
    ax_max.axhline(RBC_MAX_DEV, color="black", lw=1.2, ls="--")

else:
    # Fallback: violation rate from progress.csv
    print("  (Fallback: using violation rate proxy from progress.csv)")
    OFFPOLICY = {"CSAC-LB"}
    EPISODE_LEN = 2207.0
    for algo in ALGO_ORDER:
        cost_col = "Metrics/TestEpCost" if algo in OFFPOLICY else "Metrics/EpCost"
        len_col  = "Metrics/TestEpLen"  if algo in OFFPOLICY else "Metrics/EpLen"
        viol_arrays, step_arrays = [], []
        for run_dir in ALGO_RUN_DIRS[algo].values():
            csv = run_dir / "progress.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                if cost_col in df.columns:
                    eplen = df[len_col].values.astype(float) if len_col in df.columns \
                            else np.full(len(df), EPISODE_LEN)
                    eplen = np.where(eplen > 0, eplen, EPISODE_LEN)
                    viol_arrays.append(df[cost_col].values.astype(float) / eplen)
                    step_arrays.append(df["TotalEnvSteps"].values.astype(float))
        if viol_arrays:
            min_len = min(len(a) for a in viol_arrays)
            mat = np.stack([a[:min_len] for a in viol_arrays])
            s = step_arrays[0][:min_len]
            ax_avg.plot(s, mat.mean(0), color=COLORS[algo], lw=1.5, label=algo)
            ax_avg.fill_between(s, mat.mean(0) - mat.std(0), mat.mean(0) + mat.std(0),
                                color=COLORS[algo], alpha=0.15)

ax_avg.set_xlabel("Epoch", fontsize=10)
ax_avg.set_ylabel(r"Average Temperature Deviation ($^\circ$C)", fontsize=10)
ax_max.set_ylabel(r"Maximum Temperature Deviation ($^\circ$C)", fontsize=10)
ax_avg.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{v / STEPS_PER_EPOCH:.0f}"))
ax_avg.set_xlim(left=0)

# Combined legend (solid = avg, dashed = max)
lines_avg, labels_avg = ax_avg.get_legend_handles_labels()
# Add a note about solid vs dashed
from matplotlib.lines import Line2D
legend_extra = [
    Line2D([0], [0], color="grey", lw=1.2, ls="-",  label="Avg. (solid)"),
    Line2D([0], [0], color="grey", lw=1.2, ls="--", label="Max. (dashed)"),
]
ax_avg.legend(handles=lines_avg + legend_extra,
              loc="upper right", fontsize=7.5, ncol=2)

fig6b.tight_layout()
save(fig6b, "fig_paper_temp_dev_vs_steps")
plt.close(fig6b)


print("\nDone — 3 figures saved.")
