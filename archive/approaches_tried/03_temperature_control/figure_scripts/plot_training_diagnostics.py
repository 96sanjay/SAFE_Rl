#!/usr/bin/env python3
"""
Training diagnostics curves for all benchmark algorithms.

Produces a 5-panel figure (IEEE double-column) with:
  Panel A — Mean Reward (Metrics/EpRet)           ★★★★★  Upward slope
  Panel B — Policy Entropy / Alpha                ★★★★   Slow, steady decrease
  Panel C — Episode Cost (Metrics/EpCost)         ★★★    Downward = good
  Panel D — Value Loss (Loss/Loss_reward_critic)  ★★     Downward, then stabilizing
  Panel E — Policy Loss (Loss/Loss_pi)            ★★     Stabilizing near zero
"""

from __future__ import annotations

import json
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
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.3,
    "grid.linestyle": "--",
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "text.usetex": False,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

COL1 = 3.5   # IEEE single column (in)
COL2 = 7.16  # IEEE double column (in)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
FIG_DIR = BASE / "docs" / "temperature_case_study" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Algorithm registry — CSV paths and display info
# ---------------------------------------------------------------------------
ALGO_REGISTRY = {
    "CSAC-LB": {
        "csv": BASE / "runs/csac_lb_multi_seed/seed_1/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-09-11-15-54/progress.csv",
        "color": "#c0392b",
        "ls": "-",
        "marker": "o",
        "type": "off-policy",
    },
    "CPO": {
        "csv": BASE / "runs/cpo_temp_cooling_only/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-20-54-15/progress.csv",
        "color": "#2980b9",
        "ls": "-",
        "marker": "s",
        "type": "on-policy",
    },
    "PPO-Lag": {
        "csv": BASE / "runs/ppolag_temp_cooling_only_tight_v2/PPOLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-11-44-33/progress.csv",
        "color": "#27ae60",
        "ls": "-",
        "marker": "^",
        "type": "on-policy",
    },
    "SAC-Lag": {
        "csv": BASE / "runs/saclag_temp_cooling_only_v1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-15-33-32/progress.csv",
        "color": "#8e44ad",
        "ls": "-",
        "marker": "D",
        "type": "off-policy",
    },
    "PPO": {
        "csv": BASE / "runs/ppo_temp_cooling_only_masked_reward_bc_40ep/PPOTempMasked-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-03-28-19-39-49/progress.csv",
        "color": "#f39c12",
        "ls": "--",
        "marker": "v",
        "type": "on-policy",
    },
    "CUP": {
        "csv": BASE / "runs/cup_temp_cooling_only/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-21-43-13/progress.csv",
        "color": "#1abc9c",
        "ls": "--",
        "marker": "P",
        "type": "on-policy",
    },
    "FOCOPS": {
        "csv": BASE / "runs/focops_temp_cooling_only/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-21-17-52/progress.csv",
        "color": "#e74c3c",
        "ls": "--",
        "marker": "X",
        "type": "on-policy",
    },
}

# ---------------------------------------------------------------------------
# Multi-seed CSAC-LB paths
# ---------------------------------------------------------------------------
CSAC_LB_SEEDS = {
    "s0": BASE / "runs/csac_lb_multi_seed/seed_0/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-09-10-08-23/progress.csv",
    "s1": BASE / "runs/csac_lb_multi_seed/seed_1/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-09-11-15-54/progress.csv",
    "s2": BASE / "runs/csac_lb_multi_seed/seed_2/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-002-2026-04-09-12-26-54/progress.csv",
    "s42": BASE / "runs/csac_lb_multi_seed/seed_42/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-09-13-41-32/progress.csv",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_all():
    """Load progress CSVs for all algorithms."""
    data = {}
    for name, info in ALGO_REGISTRY.items():
        try:
            df = pd.read_csv(info["csv"])
            data[name] = df
            print(f"  Loaded {name}: {len(df)} epochs")
        except Exception as e:
            print(f"  WARNING: Failed to load {name}: {e}")
    return data


def smooth(y, window=5):
    """Exponential moving average for smoothing."""
    if len(y) < window:
        return y
    s = pd.Series(y).ewm(span=window, adjust=False).mean().values
    return s


# ---------------------------------------------------------------------------
# FIGURE 1: 5-panel training diagnostics (all algorithms)
# ---------------------------------------------------------------------------
def fig_training_diagnostics(data):
    """5-panel training diagnostic curves for all algorithms."""
    print("[Fig] Training diagnostics (5-panel) ...")

    fig, axes = plt.subplots(5, 1, figsize=(COL2, 8.5), sharex=True)

    # Panel definitions
    panels = [
        {
            "key": "Metrics/EpRet",
            "title": "(a) Mean episode reward",
            "ylabel": "Episode return",
        },
        {
            "key": "entropy",  # special handling
            "title": "(b) Policy entropy",
            "ylabel": "Entropy / \\alpha",
        },
        {
            "key": "Metrics/EpCost",
            "title": "(c) Episode cost",
            "ylabel": "Episode cost",
        },
        {
            "key": "Loss/Loss_reward_critic",
            "title": "(d) Value loss",
            "ylabel": "Critic loss",
        },
        {
            "key": "Loss/Loss_pi",
            "title": "(e) Policy loss",
            "ylabel": "Policy loss",
        },
    ]

    for ax, panel in zip(axes, panels):
        for name, info in ALGO_REGISTRY.items():
            if name not in data:
                continue
            df = data[name]

            # Get the right column
            if panel["key"] == "entropy":
                # On-policy: Train/Entropy; Off-policy: Value/alpha (entropy temperature)
                if info["type"] == "on-policy" and "Train/Entropy" in df.columns:
                    y = df["Train/Entropy"].values
                elif "Value/alpha" in df.columns:
                    y = df["Value/alpha"].values
                elif "Loss/alpha_loss" in df.columns:
                    # SAC-Lag has alpha but might be named differently
                    y = df.get("Value/alpha", pd.Series([np.nan]*len(df))).values
                else:
                    continue
            else:
                col = panel["key"]
                if col not in df.columns:
                    continue
                y = df[col].values

            x = np.arange(len(y))
            y_smooth = smooth(y, window=5)

            # Plot raw with low alpha, smoothed with full
            ax.plot(x, y, color=info["color"], alpha=0.15, linewidth=0.5)
            ax.plot(x, y_smooth, color=info["color"], ls=info["ls"],
                    linewidth=1.2, label=name, marker=info["marker"],
                    markevery=max(1, len(x)//8), markersize=3.5)

        ax.set_ylabel(panel["ylabel"])
        ax.set_title(panel["title"], fontsize=8, loc="left")
        ax.legend(loc="best", ncol=4, fontsize=5.5)

        # Log scale for value loss
        if panel["key"] == "Loss/Loss_reward_critic":
            ax.set_yscale("symlog", linthresh=1.0)

    axes[-1].set_xlabel("Epoch")
    fig.align_ylabels(axes)
    plt.tight_layout(h_pad=0.4)

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_training_diagnostics.{ext}")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig_training_diagnostics.pdf/.png'}")


# ---------------------------------------------------------------------------
# FIGURE 2: Multi-seed CSAC-LB training diagnostics
# ---------------------------------------------------------------------------
def fig_multiseed_diagnostics():
    """4-panel CSAC-LB multi-seed training curves."""
    print("[Fig] Multi-seed CSAC-LB diagnostics ...")

    seed_data = {}
    for sname, csv_path in CSAC_LB_SEEDS.items():
        try:
            seed_data[sname] = pd.read_csv(csv_path)
        except:
            print(f"  WARNING: Failed to load {sname}")

    if not seed_data:
        print("  No seed data, skipping.")
        return

    seed_colors = {"s0": "#e74c3c", "s1": "#2980b9", "s2": "#27ae60", "s42": "#f39c12"}
    seed_labels = {"s0": "Seed 0", "s1": "Seed 1", "s2": "Seed 2", "s42": "Seed 42"}

    fig, axes = plt.subplots(4, 1, figsize=(COL2, 7.0), sharex=True)

    panels = [
        ("Metrics/EpRet", "Episode Return", "(a) Mean Reward"),
        ("Metrics/EpCost", "Episode Cost", "(b) Episode Cost"),
        ("Loss/Loss_reward_critic", "Critic Loss", "(c) Value Loss"),
        ("Value/alpha", "α (entropy temp.)", "(d) Entropy Temperature α"),
    ]

    for ax, (col, ylabel, title) in zip(axes, panels):
        for sname, df in seed_data.items():
            if col not in df.columns:
                continue
            y = df[col].values
            x = np.arange(len(y))
            y_sm = smooth(y, window=5)
            ax.plot(x, y, color=seed_colors[sname], alpha=0.2, linewidth=0.5)
            ax.plot(x, y_sm, color=seed_colors[sname], linewidth=1.2,
                    label=seed_labels[sname], marker="o",
                    markevery=max(1, len(x)//8), markersize=3)

        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8, loc="left")
        ax.legend(loc="best", ncol=4, fontsize=6)

        if col == "Loss/Loss_reward_critic":
            ax.set_yscale("symlog", linthresh=1.0)

    axes[-1].set_xlabel("Epoch")
    fig.align_ylabels(axes)
    plt.tight_layout(h_pad=0.4)

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_multiseed_diagnostics.{ext}")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig_multiseed_diagnostics.pdf/.png'}")


# ---------------------------------------------------------------------------
# FIGURE 3: Explained Variance proxy — reward prediction quality
# ---------------------------------------------------------------------------
def fig_reward_prediction_quality(data):
    """
    Plot reward vs value-critic estimate for on-policy algorithms that have both.
    For off-policy, plot critic value vs actual return.
    """
    print("[Fig] Reward prediction quality ...")

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.8))

    # Left: On-policy — EpRet vs Value/reward
    ax = axes[0]
    for name, info in ALGO_REGISTRY.items():
        if name not in data:
            continue
        df = data[name]
        if info["type"] != "on-policy":
            continue
        if "Value/reward" not in df.columns:
            continue
        x = np.arange(len(df))
        epret = smooth(df["Metrics/EpRet"].values, 5)
        vr = smooth(df["Value/reward"].values, 5)
        ax.plot(x, epret, color=info["color"], ls="-", linewidth=1.0,
                label=f"{name} (actual)", alpha=0.8)
        ax.plot(x, vr, color=info["color"], ls="--", linewidth=0.8,
                label=f"{name} (V est.)", alpha=0.6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Return / Value Estimate")
    ax.set_title("(a) On-policy: Return vs Value Estimate", fontsize=8, loc="left")
    ax.legend(fontsize=5, ncol=2, loc="best")

    # Right: Off-policy — EpRet vs Value/reward_critic
    ax = axes[1]
    for name, info in ALGO_REGISTRY.items():
        if name not in data:
            continue
        df = data[name]
        if info["type"] != "off-policy":
            continue
        epret = smooth(df["Metrics/EpRet"].values, 5)
        x = np.arange(len(df))
        ax.plot(x, epret, color=info["color"], ls="-", linewidth=1.0,
                label=f"{name} (return)", alpha=0.8)
        if "Value/reward_critic" in df.columns:
            vc = smooth(df["Value/reward_critic"].values, 5)
            ax.plot(x, vc, color=info["color"], ls="--", linewidth=0.8,
                    label=f"{name} (critic)", alpha=0.6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Return / Critic Value")
    ax.set_title("(b) Off-policy: Return vs Critic Value", fontsize=8, loc="left")
    ax.legend(fontsize=5, ncol=2, loc="best")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_reward_prediction.{ext}")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig_reward_prediction.pdf/.png'}")


# ---------------------------------------------------------------------------
# FIGURE 4: Cost critic and Lagrange multiplier evolution
# ---------------------------------------------------------------------------
def fig_constraint_dynamics(data):
    """
    2-panel: (a) cost critic value over training, (b) Lagrange multiplier evolution.
    """
    print("[Fig] Constraint enforcement dynamics ...")

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.8))

    # (a) Cost critic / cost value
    ax = axes[0]
    for name, info in ALGO_REGISTRY.items():
        if name not in data:
            continue
        df = data[name]
        # Try multiple cost value column names
        for col in ["Value/cost_critic", "Value/cost", "Value/cost_critic_max"]:
            if col in df.columns:
                y = smooth(df[col].values, 5)
                x = np.arange(len(y))
                ax.plot(x, y, color=info["color"], ls=info["ls"],
                        linewidth=1.0, label=name, marker=info["marker"],
                        markevery=max(1, len(x)//8), markersize=3)
                break

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cost Value Estimate")
    ax.set_title("(a) Cost Critic / Value Estimate", fontsize=8, loc="left")
    ax.legend(fontsize=5, ncol=2, loc="best")

    # (b) Lagrange multiplier
    ax = axes[1]
    for name, info in ALGO_REGISTRY.items():
        if name not in data:
            continue
        df = data[name]
        for col in ["Metrics/LagrangeMultiplier", "Metrics/PID_Lambda",
                     "Value/barrier_penalty"]:
            if col in df.columns:
                vals = df[col].values
                if np.all(np.isnan(vals)) or np.all(vals == 0):
                    continue
                y = smooth(vals, 5)
                x = np.arange(len(y))
                lbl = name
                if col == "Value/barrier_penalty":
                    lbl += " (barrier)"
                elif col == "Metrics/PID_Lambda":
                    lbl += " (PID-λ)"
                ax.plot(x, y, color=info["color"], ls=info["ls"],
                        linewidth=1.0, label=lbl, marker=info["marker"],
                        markevery=max(1, len(x)//8), markersize=3)
                break

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lagrange Multiplier / Penalty")
    ax.set_title("(b) Lagrange Multiplier Evolution", fontsize=8, loc="left")
    ax.legend(fontsize=5, ncol=2, loc="best")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_constraint_dynamics.{ext}")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig_constraint_dynamics.pdf/.png'}")


# ---------------------------------------------------------------------------
# FIGURE 5: Summary dashboard — 3x2 compact
# ---------------------------------------------------------------------------
def fig_compact_dashboard(data):
    """
    Compact 3x2 dashboard: reward, cost, entropy, value loss, policy loss,
    and learning rate / KL divergence.
    """
    print("[Fig] Compact training dashboard (3x2) ...")

    fig, axes = plt.subplots(3, 2, figsize=(COL2, 7.0), sharex=True)

    panel_defs = [
        # row 0
        ("Metrics/EpRet", "Episode return", "(a) Mean episode reward"),
        ("Metrics/EpCost", "Episode cost", "(b) Episode cost"),
        # row 1
        ("entropy", "Entropy / alpha", "(c) Policy entropy"),
        ("Loss/Loss_reward_critic", "Critic loss", "(d) Value loss"),
        # row 2
        ("Loss/Loss_pi", "Policy loss", "(e) Policy loss"),
        ("kl_or_lr", "KL / LR", "(f) KL divergence"),
    ]

    for idx, (col, ylabel, title) in enumerate(panel_defs):
        row, c = divmod(idx, 2)
        ax = axes[row, c]

        for name, info in ALGO_REGISTRY.items():
            if name not in data:
                continue
            df = data[name]

            if col == "entropy":
                if info["type"] == "on-policy" and "Train/Entropy" in df.columns:
                    y = df["Train/Entropy"].values
                elif "Value/alpha" in df.columns:
                    y = df["Value/alpha"].values
                else:
                    continue
            elif col == "kl_or_lr":
                if "Train/KL" in df.columns:
                    y = df["Train/KL"].values
                elif "Train/LR" in df.columns:
                    y = df["Train/LR"].values
                else:
                    continue
            else:
                if col not in df.columns:
                    continue
                y = df[col].values

            x = np.arange(len(y))
            y_sm = smooth(y, 5)
            ax.plot(x, y, color=info["color"], alpha=0.12, linewidth=0.4)
            ax.plot(x, y_sm, color=info["color"], ls=info["ls"],
                    linewidth=1.0, label=name, marker=info["marker"],
                    markevery=max(1, len(x)//8), markersize=2.5)

        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=7.5, loc="left")

        if col == "Loss/Loss_reward_critic":
            ax.set_yscale("symlog", linthresh=1.0)
        if col == "kl_or_lr":
            ax.set_yscale("log")

        if row == 0 and c == 0:
            ax.legend(loc="lower right", ncol=4, fontsize=5)

    for ax in axes[-1]:
        ax.set_xlabel("Epoch")

    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    plt.tight_layout(h_pad=0.5, w_pad=0.8)

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_training_dashboard.{ext}")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'fig_training_dashboard.pdf/.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Training Diagnostics — All Algorithms")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 60)

    print("\nLoading training logs ...")
    data = load_all()

    print(f"\nLoaded {len(data)} algorithms.")
    print()

    fig_training_diagnostics(data)
    fig_multiseed_diagnostics()
    fig_reward_prediction_quality(data)
    fig_constraint_dynamics(data)
    fig_compact_dashboard(data)

    print()
    print("=" * 60)
    print("All training diagnostic figures generated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
