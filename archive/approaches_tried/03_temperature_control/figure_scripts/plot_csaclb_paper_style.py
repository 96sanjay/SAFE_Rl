#!/usr/bin/env python3
"""
Replicates the two key figures from Zhang et al. (CSAC-LB paper, arXiv:2409.19716)
adapted to our CityLearn temperature-control benchmark.

  Fig 5  — Pareto scatter during training (paper Fig 5 style):
            Each dot = one evaluation episode, colour = training steps (rainbow).
            Crosses (×) mark Pareto-optimal points.
            Black ★ = RBC baseline.  Dashed red = cost-limit boundary.
            X = Violation Rate (EpCost/EpLen), Y = Episode Return.
            One subplot per algorithm (5 total).

  Fig 6  — 2-row training curves (paper Fig 6 style):
            Top row    : Episode Return vs Training Steps (mean ± std over seeds)
            Bottom row : Violation Rate vs Training Steps  (mean ± std over seeds)
            Dashed red = cost-limit threshold, dashed black = RBC baseline.
            One column per algorithm (5 total).
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[3]   # repo root
OUT  = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# ── Run directories ─────────────────────────────────────────────────────────────
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
    "CUP": {
        42: BASE / "runs/cup_temp_cooling_only_v2/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-23-10-06",
        0:  BASE / "runs/cup_temp_cooling_only_v2/seed_0/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-13-22-49",
        1:  BASE / "runs/cup_temp_cooling_only_v2/seed_1/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-13-46-22",
    },
    "FOCOPS": {
        42: BASE / "runs/focops_temp_cooling_only_v2/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-22-44-31",
        0:  BASE / "runs/focops_temp_cooling_only_v2/seed_0/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-14-09-24",
        1:  BASE / "runs/focops_temp_cooling_only_v2/seed_1/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-14-35-00",
    },
}

# CSAC-LB is off-policy: use separate TestEp* evaluation columns
OFFPOLICY = {"CSAC-LB"}

ALGO_ORDER = ["CSAC-LB", "SAC-Lag", "CPO", "CUP", "FOCOPS"]

# Colors matching paper-style (one distinct color per algo)
COLORS = {
    "CSAC-LB": "#e41a1c",   # red
    "SAC-Lag": "#ff7f00",   # orange
    "CPO":     "#377eb8",   # blue
    "CUP":     "#4daf4a",   # green
    "FOCOPS":  "#984ea3",   # purple
}

# ── Constants ───────────────────────────────────────────────────────────────────
EPISODE_LEN   = 2207.0        # timesteps per episode (from data)
COST_LIMIT    = 0.10          # 10 % violation-rate threshold used during training
RBC_VIOL_FRAC = 0.414         # RBC rule-based controller violation fraction
RBC_REWARD    = 1680.0        # approximate RBC episode reward

# ── IEEE / paper style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    6.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.04,
    "axes.linewidth":     0.7,
    "axes.grid":          True,
    "grid.alpha":         0.20,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.35,
    "lines.linewidth":    1.2,
})

# ── Helpers ─────────────────────────────────────────────────────────────────────
def _ret_col(algo):
    return "Metrics/TestEpRet" if algo in OFFPOLICY else "Metrics/EpRet"

def _cost_col(algo):
    return "Metrics/TestEpCost" if algo in OFFPOLICY else "Metrics/EpCost"

def _len_col(algo):
    return "Metrics/TestEpLen" if algo in OFFPOLICY else "Metrics/EpLen"

def load_seeds(algo) -> list[pd.DataFrame]:
    dfs = []
    for run_dir in ALGO_RUN_DIRS[algo].values():
        csv = run_dir / "progress.csv"
        if csv.exists():
            dfs.append(pd.read_csv(csv))
        else:
            print(f"  WARNING: missing {csv}")
    return dfs

def stack_metric(dfs, col) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, mean, std) aligned to shortest seed."""
    arrays, steps = [], []
    for df in dfs:
        if col in df.columns and "TotalEnvSteps" in df.columns:
            arrays.append(df[col].values.astype(float))
            steps.append(df["TotalEnvSteps"].values.astype(float))
    if not arrays:
        return np.array([]), np.array([]), np.array([])
    min_len = min(len(a) for a in arrays)
    mat = np.stack([a[:min_len] for a in arrays])
    s   = steps[0][:min_len]          # use first seed's step axis
    return s, mat.mean(0), mat.std(0)

def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Boolean mask for Pareto-optimal points.
    Optimality: lower x (violation) AND higher y (reward) simultaneously.
    """
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        # point i is dominated if some j has x[j] <= x[i] AND y[j] >= y[i]
        # with at least one strict inequality
        dominated = ((x <= x[i]) & (y >= y[i]) &
                     ((x < x[i]) | (y > y[i])))
        dominated[i] = False
        if dominated.any():
            mask[i] = False
    return mask

def save(fig, name):
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p)
        print(f"  Saved {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5  —  Pareto Scatter  (paper Fig 5 style)
# X = Violation Rate (EpCost / EpLen)
# Y = Episode Return
# Colour = TotalEnvSteps   (rainbow: blue=early → red=late)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Fig 5 — Pareto scatter …")

n_algos   = len(ALGO_ORDER)
# Paper uses ~2.2 inches wide per subplot for 3 algos → we use 2.2 for 5
fig5, axes5 = plt.subplots(1, n_algos, figsize=(2.2 * n_algos, 3.0),
                            sharey=False)

max_steps = 135_000          # ~60 epochs × 2208 steps
cmap5     = cm.rainbow
norm5     = mcolors.Normalize(vmin=0, vmax=max_steps)

for ax, algo in zip(axes5, ALGO_ORDER):
    dfs      = load_seeds(algo)
    ret_col  = _ret_col(algo)
    cost_col = _cost_col(algo)
    len_col  = _len_col(algo)

    all_x, all_y, all_c = [], [], []

    for df in dfs:
        if ret_col not in df.columns or cost_col not in df.columns:
            continue
        eplen = df[len_col].values.astype(float) if len_col in df.columns else np.full(len(df), EPISODE_LEN)
        x = df[cost_col].values.astype(float) / np.where(eplen > 0, eplen, EPISODE_LEN)
        y = df[ret_col].values.astype(float)
        s = df["TotalEnvSteps"].values.astype(float) if "TotalEnvSteps" in df.columns else np.linspace(0, max_steps, len(df))
        all_x.append(x)
        all_y.append(y)
        all_c.append(s)

    if all_x:
        X = np.concatenate(all_x)
        Y = np.concatenate(all_y)
        C = np.concatenate(all_c)
        colors = cmap5(norm5(C))

        # Scatter — all evaluation episodes
        ax.scatter(X, Y, c=colors, s=14, alpha=0.80, linewidths=0,
                   rasterized=True)

        # Pareto-optimal crosses
        pm = pareto_mask(X, Y)
        if pm.any():
            ax.scatter(X[pm], Y[pm], marker="x", s=35, c=colors[pm],
                       linewidths=1.0, zorder=6)

    # RBC baseline ★ (black)
    ax.scatter([RBC_VIOL_FRAC], [RBC_REWARD], marker="*", s=150,
               color="black", zorder=8, label="Rule-based")

    # Cost-limit vertical dashed line
    ax.axvline(COST_LIMIT, color="#e74c3c", linewidth=0.9,
               linestyle="--", alpha=0.85, label=f"Cost limit ({int(COST_LIMIT*100)}%)")

    ax.set_title(algo, fontweight="bold", pad=4, fontsize=9)
    ax.set_xlabel("Violation Rate", labelpad=3, fontsize=7.5)
    if algo == ALGO_ORDER[0]:
        ax.set_ylabel("Episode Return", fontsize=7.5)
    ax.tick_params(labelsize=6.5)

# Legend on first subplot only (matches paper)
axes5[0].legend(fontsize=6, loc="upper right", framealpha=0.75,
                handletextpad=0.3, borderpad=0.4, handlelength=1.2)

# Shared colorbar on right
sm5 = cm.ScalarMappable(cmap=cmap5, norm=norm5)
sm5.set_array([])
cbar5 = fig5.colorbar(sm5, ax=axes5.tolist(), shrink=0.82, pad=0.01,
                      aspect=22)
cbar5.set_label("Training Steps", fontsize=7.5)
cbar5.ax.tick_params(labelsize=6.5)
cbar5.ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f"{int(v/1e3)}k" if v > 0 else "0"))

fig5.suptitle(
    "Evaluation Trajectory during Training\n"
    "(each dot = one evaluation episode, × = Pareto-optimal, colour = training steps)",
    fontsize=8.5, y=1.02,
)
fig5.subplots_adjust(wspace=0.32)
save(fig5, "fig_csaclb_paper_pareto_scatter")
plt.close(fig5)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6  —  2-row training curves  (paper Fig 6 style)
# Top row   : Episode Return  (mean ± std over seeds)
# Bottom row: Violation Rate  (mean ± std) + cost-limit + RBC reference
# X-axis    : TotalEnvSteps  (×10⁵)
# One column per algorithm
# ══════════════════════════════════════════════════════════════════════════════
print("Building Fig 6 — 2-row training curves …")

fig6, axes6 = plt.subplots(2, n_algos,
                            figsize=(2.3 * n_algos, 4.4),
                            gridspec_kw={"hspace": 0.08, "wspace": 0.28})

for col_i, algo in enumerate(ALGO_ORDER):
    ax_top = axes6[0, col_i]
    ax_bot = axes6[1, col_i]
    color  = COLORS[algo]

    dfs      = load_seeds(algo)
    ret_col  = _ret_col(algo)
    cost_col = _cost_col(algo)
    len_col  = _len_col(algo)

    # ── Violation rate metric (cost normalised by episode length) ─────────────
    # Build per-seed violation rate arrays, then stack
    viol_arrays, step_arrays = [], []
    ret_arrays = []
    for df in dfs:
        if cost_col not in df.columns:
            continue
        eplen = df[len_col].values.astype(float) if len_col in df.columns else np.full(len(df), EPISODE_LEN)
        eplen = np.where(eplen > 0, eplen, EPISODE_LEN)
        viol_arrays.append((df[cost_col].values.astype(float) / eplen))
        step_arrays.append(df["TotalEnvSteps"].values.astype(float))
        if ret_col in df.columns:
            ret_arrays.append(df[ret_col].values.astype(float))

    # ── Top: Reward ───────────────────────────────────────────────────────────
    steps_r, mu_r, sd_r = stack_metric(dfs, ret_col)
    if len(steps_r):
        ax_top.plot(steps_r, mu_r, color=color, lw=1.3, label=algo)
        ax_top.fill_between(steps_r, mu_r - sd_r, mu_r + sd_r,
                            color=color, alpha=0.18)

    ax_top.axhline(RBC_REWARD, color="black", lw=0.9,
                   linestyle="--", alpha=0.75, label="Rule-based")
    ax_top.set_title(algo, fontweight="bold", pad=3, fontsize=9)
    if col_i == 0:
        ax_top.set_ylabel("Episode Return", fontsize=7.5)
    ax_top.tick_params(labelsize=6.5)
    ax_top.set_xticklabels([])   # hide x ticks on top row

    # ── Bottom: Violation Rate ────────────────────────────────────────────────
    if viol_arrays:
        min_len = min(len(a) for a in viol_arrays)
        mat = np.stack([a[:min_len] for a in viol_arrays])
        s   = step_arrays[0][:min_len]
        mu_v, sd_v = mat.mean(0), mat.std(0)
        ax_bot.plot(s, mu_v, color=color, lw=1.3)
        ax_bot.fill_between(s, mu_v - sd_v, mu_v + sd_v,
                            color=color, alpha=0.18)

    # Cost-limit dashed threshold
    ax_bot.axhline(COST_LIMIT, color="#e74c3c", lw=1.0,
                   linestyle="--", alpha=0.90,
                   label=f"Cost limit ({int(COST_LIMIT*100)}%)")
    # RBC violation reference
    ax_bot.axhline(RBC_VIOL_FRAC, color="black", lw=0.9,
                   linestyle=":", alpha=0.75, label="Rule-based")

    ax_bot.set_ylim(bottom=0)
    if col_i == 0:
        ax_bot.set_ylabel("Violation Rate", fontsize=7.5)

    # X-axis: format in units of ×10⁵
    ax_bot.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1e5:.1f}"))
    ax_bot.set_xlabel("Training Steps (×10⁵)", fontsize=7.5, labelpad=2)
    ax_bot.tick_params(labelsize=6.5)

    # Legend only on first column
    if col_i == 0:
        ax_top.legend(fontsize=5.5, loc="lower right", framealpha=0.75,
                      handlelength=1.4, borderpad=0.35)
        ax_bot.legend(fontsize=5.5, loc="upper right", framealpha=0.75,
                      handlelength=1.4, borderpad=0.35)

fig6.suptitle(
    "Training Dynamics: Mean ± Std over Seeds\n"
    "(top: episode return; bottom: violation rate; dashed red = cost limit)",
    fontsize=8.5, y=1.02,
)
save(fig6, "fig_csaclb_paper_training_curves")
plt.close(fig6)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7  —  Combined Pareto Scatter — ALL algorithms in ONE single plot
# X = Violation Rate   |  Y = Episode Return
# Color = algorithm    |  Opacity = training progression (fades in over time)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Fig 7 — combined single-plot scatter (all algos) …")

fig7, ax7 = plt.subplots(figsize=(5.2, 3.8))

for algo in ALGO_ORDER:
    dfs      = load_seeds(algo)
    ret_col  = _ret_col(algo)
    cost_col = _cost_col(algo)
    len_col  = _len_col(algo)
    color    = COLORS[algo]

    all_x, all_y, all_s = [], [], []
    for df in dfs:
        if ret_col not in df.columns or cost_col not in df.columns:
            continue
        eplen = df[len_col].values.astype(float) if len_col in df.columns else np.full(len(df), EPISODE_LEN)
        eplen = np.where(eplen > 0, eplen, EPISODE_LEN)
        x = df[cost_col].values.astype(float) / eplen
        y = df[ret_col].values.astype(float)
        s = df["TotalEnvSteps"].values.astype(float) if "TotalEnvSteps" in df.columns else np.linspace(0, max_steps, len(df))
        all_x.append(x)
        all_y.append(y)
        all_s.append(s)

    if not all_x:
        continue

    X = np.concatenate(all_x)
    Y = np.concatenate(all_y)
    S = np.concatenate(all_s)

    # Alpha fades from 0.15 (early) to 0.85 (late) to show training progression
    alpha_vals = 0.15 + 0.70 * (S / max_steps)

    # Scatter all dots with per-point alpha using RGBA colours
    import matplotlib.colors as mc
    base_rgb  = mc.to_rgb(color)
    rgba_arr  = np.array([[*base_rgb, a] for a in alpha_vals])
    ax7.scatter(X, Y, color=rgba_arr, s=10, linewidths=0, rasterized=True)

    # Draw a larger opaque mean marker at the LAST epoch (max step) per seed
    for df in dfs:
        if ret_col not in df.columns or cost_col not in df.columns:
            continue
        eplen_v = df[len_col].values.astype(float) if len_col in df.columns else np.full(len(df), EPISODE_LEN)
        eplen_v = np.where(eplen_v > 0, eplen_v, EPISODE_LEN)
        x_last = float(df[cost_col].iloc[-1]) / float(eplen_v[-1])
        y_last = float(df[ret_col].iloc[-1])
        ax7.scatter([x_last], [y_last], c=color, s=55,
                    linewidths=0.6, edgecolors="white", zorder=6)

# Dummy handles for legend (filled squares showing algo color)
handles = [
    plt.scatter([], [], c=COLORS[a], s=45, marker="o",
                linewidths=0, label=a)
    for a in ALGO_ORDER
]
# RBC reference (no label on scatter so legend stays clean)
ax7.scatter([RBC_VIOL_FRAC], [RBC_REWARD], marker="*", s=200,
            color="black", zorder=9)
handles.append(plt.scatter([], [], marker="*", s=100, c="black",
                            label="Rule-based", linewidths=0))

# Cost-limit line (label added via handles list in legend below)
ax7.axvline(COST_LIMIT, color="#e74c3c", lw=0.9, linestyle="--", alpha=0.85)

ax7.set_xlabel("Violation Rate", fontsize=9)
ax7.set_ylabel("Episode Return", fontsize=9)
ax7.set_title(
    "Evaluation Trajectory during Training — All Algorithms\n"
    "(fading dots = training progression; bright = late training)",
    fontsize=9, pad=4,
)

ax7.legend(handles=handles + [
               plt.Line2D([0],[0], color="#e74c3c", lw=0.9,
                          linestyle="--", label=f"Cost limit ({int(COST_LIMIT*100)}%)")
           ],
           fontsize=7, loc="lower right", framealpha=0.80,
           handletextpad=0.3, borderpad=0.40)

fig7.tight_layout()
save(fig7, "fig_csaclb_paper_pareto_scatter_combined")
plt.close(fig7)

print("Done.")
