#!/usr/bin/env python3
"""
Multi-seed training curves — all algorithms on the same graph per metric.
Legend placed OUTSIDE the plot area so it never overlaps the curves.

Figure 1: 1x4  — Return | Cost | Value Loss | Policy Loss  (all algos overlaid)
Figure 2: 1x2  — Return | Cost only  (cleaner thesis version)
Figure 3: 1x4  — CSAC-LB only with barrier penalty panel
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# IEEE style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size":          8,
    "axes.titlesize":     8,
    "axes.labelsize":     7.5,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.04,
    "axes.grid":          True,
    "grid.alpha":         0.20,
    "grid.linewidth":     0.3,
    "grid.linestyle":     "--",
    "axes.linewidth":     0.5,
    "lines.linewidth":    1.2,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "0.8",
    "legend.fancybox":    False,
    "text.usetex":        False,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.width":  0.4,
    "ytick.major.width":  0.4,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
})

COL2 = 7.16   # IEEE double-column width (inches)

BASE    = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
FIG_DIR = BASE / "docs" / "temperature_case_study" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------
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

# Column to use per algo-type (on-policy vs off-policy differ)
OFFPOLICY = {"CSAC-LB", "SAC-Lag"}

METRICS = {
    # key: (on-policy col, off-policy col, y-axis label)
    "reward":   ("Metrics/EpRet",            "Metrics/TestEpRet",  "Episode Return"),
    "cost":     ("Metrics/EpCost",           "Metrics/TestEpCost", "Episode Cost"),
    "val_loss": ("Loss/Loss_reward_critic",  "Loss/Loss_reward_critic", "Value Loss"),
    "pol_loss": ("Loss/Loss_pi",             "Loss/Loss_pi",        "Policy Loss"),
}

ALGO_COLORS = {
    "CSAC-LB": "#c0392b",
    "SAC-Lag": "#795548",
    "CPO":     "#2471a3",
    "CUP":     "#e67e22",
    "FOCOPS":  "#27ae60",
}

ALGO_ORDER = ["CSAC-LB", "SAC-Lag", "CPO", "CUP", "FOCOPS"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_matrix(algo: str, metric_key: str) -> np.ndarray | None:
    """
    Returns shape (n_seeds, n_epochs) array for the given algo + metric.
    Returns None if no data found.
    """
    onpol_col, offpol_col, _ = METRICS[metric_key]
    col = offpol_col if algo in OFFPOLICY else onpol_col

    arrays = []
    for run_dir in ALGO_RUN_DIRS[algo].values():
        csv = run_dir / "progress.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if col not in df.columns:
            continue
        arrays.append(df[col].values.astype(float))

    if not arrays:
        return None

    min_len = min(len(a) for a in arrays)
    return np.stack([a[:min_len] for a in arrays])   # (n_seeds, n_epochs)


def smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


# ---------------------------------------------------------------------------
# Core plot primitive: mean ± std band into a single axes
# ---------------------------------------------------------------------------
def plot_mean_std(ax: plt.Axes, mat: np.ndarray, color: str,
                  label: str, smooth_w: int = 5) -> None:
    epochs = np.arange(1, mat.shape[1] + 1)
    smoothed = np.stack([smooth(mat[i], smooth_w) for i in range(len(mat))])
    mu  = smoothed.mean(axis=0)
    std = smoothed.std(axis=0)

    ax.plot(epochs, mu, color=color, linewidth=1.4, label=label, zorder=3)
    ax.fill_between(epochs, mu - std, mu + std,
                    color=color, alpha=0.22, linewidth=0, zorder=2)


def fmt_ax(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("Epoch", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  -> {stem}.pdf/.png")


# ===========================================================================
# Figure 1 — 1×4: all algos on same graph per metric, legend OUTSIDE
# ===========================================================================
def fig_1x4_overlay() -> None:
    metric_keys = ["reward", "cost", "val_loss", "pol_loss"]
    col_titles  = ["(a) Episode Return", "(b) Episode Cost",
                   "(c) Value Loss",     "(d) Policy Loss"]

    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.4))

    handles, labels = [], []

    for algo in ALGO_ORDER:
        color = ALGO_COLORS[algo]
        for ax, mkey in zip(axes, metric_keys):
            mat = load_matrix(algo, mkey)
            if mat is None:
                continue
            plot_mean_std(ax, mat, color=color,
                          label=f"{algo} (n={mat.shape[0]})")

        # Collect legend handle from first panel that worked
        mat0 = load_matrix(algo, "reward")
        if mat0 is not None:
            import matplotlib.lines as mlines
            h = mlines.Line2D([], [], color=color, linewidth=1.4,
                              label=f"{algo} (n={mat0.shape[0]})")
            handles.append(h)
            labels.append(f"{algo} (n={mat0.shape[0]})")

    for ax, title in zip(axes, col_titles):
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.tick_params(labelsize=6.5)

    axes[0].set_ylabel("Value", fontsize=7)

    # Legend OUTSIDE — below the figure, horizontal, no overlap
    fig.legend(handles=handles, labels=labels,
               loc="lower center",
               bbox_to_anchor=(0.5, -0.18),
               ncol=len(ALGO_ORDER),
               fontsize=7,
               framealpha=0.9,
               edgecolor="0.8")

    fig.suptitle("Multi-Seed Training Curves — Mean ± 1σ Shaded Band",
                 fontsize=9)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.22)   # make room for legend below
    save(fig, "fig_multiseed_training_overlay")


# ===========================================================================
# Figure 2 — 1×2: reward + cost only, legend OUTSIDE (cleaner thesis fig)
# ===========================================================================
def fig_1x2_reward_cost() -> None:
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(COL2 * 0.65, 2.4))

    handles = []
    import matplotlib.lines as mlines

    for algo in ALGO_ORDER:
        color = ALGO_COLORS[algo]
        for ax, mkey in [(ax_r, "reward"), (ax_c, "cost")]:
            mat = load_matrix(algo, mkey)
            if mat is not None:
                plot_mean_std(ax, mat, color=color,
                              label=f"{algo} (n={mat.shape[0]})")

        mat0 = load_matrix(algo, "reward")
        if mat0 is not None:
            handles.append(mlines.Line2D(
                [], [], color=color, linewidth=1.4,
                label=f"{algo} (n={mat0.shape[0]})"))

    ax_r.set_title("(a) Episode Return", fontsize=8)
    ax_c.set_title("(b) Episode Cost",   fontsize=8)
    for ax, ylabel in [(ax_r, "Episode Return"), (ax_c, "Episode Cost")]:
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.tick_params(labelsize=6.5)

    # Legend OUTSIDE — to the right of the figure
    fig.legend(handles=handles,
               loc="center left",
               bbox_to_anchor=(1.01, 0.5),
               fontsize=7,
               framealpha=0.9,
               edgecolor="0.8")

    fig.suptitle("Episode Return & Cost — Mean ± 1σ", fontsize=9)
    fig.tight_layout()
    plt.subplots_adjust(right=0.78)   # make room for legend on right
    save(fig, "fig_multiseed_reward_cost_overlay")


# ===========================================================================
# Figure 3 — CSAC-LB only, 1×4 with barrier penalty, legend OUTSIDE
# ===========================================================================
def fig_csaclb_1x4() -> None:
    algo  = "CSAC-LB"
    color = ALGO_COLORS[algo]

    special_metrics = [
        ("Metrics/TestEpRet",        "Episode Return"),
        ("Metrics/TestEpCost",       "Episode Cost"),
        ("Loss/Loss_reward_critic",  "Value Loss"),
        ("Value/barrier_penalty",    "Barrier Penalty"),
    ]

    arrays = {ylabel: [] for _, ylabel in special_metrics}
    for run_dir in ALGO_RUN_DIRS[algo].values():
        csv = run_dir / "progress.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        for col, ylabel in special_metrics:
            if col in df.columns:
                arrays[ylabel].append(df[col].values.astype(float))

    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.1))

    for ax, (col, ylabel) in zip(axes, special_metrics):
        lst = arrays[ylabel]
        if not lst:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="gray")
            ax.set_title(ylabel, fontsize=8)
            continue
        min_len = min(len(a) for a in lst)
        mat = np.stack([a[:min_len] for a in lst])
        plot_mean_std(ax, mat, color=color, label=f"n={mat.shape[0]}")
        ax.set_title(ylabel, fontsize=8)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.tick_params(labelsize=6.5)

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    h_line = mlines.Line2D([], [], color=color, linewidth=1.4,
                           label=f"CSAC-LB mean (n=6)")
    h_band = mpatches.Patch(color=color, alpha=0.22, label="± 1 std dev")

    fig.legend(handles=[h_line, h_band],
               loc="lower center",
               bbox_to_anchor=(0.5, -0.15),
               ncol=2,
               fontsize=7,
               framealpha=0.9,
               edgecolor="0.8")

    fig.suptitle("CSAC-LB — 6-Seed Training Curves (mean ± 1σ)", fontsize=9)
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.20, wspace=0.4)
    save(fig, "fig_csaclb_multiseed_training")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("Figure 1: 1×4 overlay (all algos, 4 metrics, legend below) ...")
    fig_1x4_overlay()

    print("Figure 2: 1×2 reward+cost (legend right) ...")
    fig_1x2_reward_cost()

    print("Figure 3: CSAC-LB 1×4 with barrier penalty ...")
    fig_csaclb_1x4()

    print(f"\nDone. Saved to: {FIG_DIR}")
