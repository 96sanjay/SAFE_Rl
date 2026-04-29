#!/usr/bin/env python3
"""
IEEE-style benchmark figures for safe RL temperature control case study.

Figures:
  1. Training Dynamics (double-col) — EpCost + EpRet vs epoch
  2. Comfort Violation Rate (single-col) — horizontal bar, all algorithms
  3. Energy–Comfort Tradeoff (single-col) — scatter
  4. Hot vs Cold Discomfort (double-col) — grouped bar
  5. 72-h Stress Test Violation (single-col) — horizontal bar
  6. 72-h Stress Test Multi-Metric (double-col) — 4-panel vertical bars
  7. Normalised KPI Comparison (double-col) — heatmap, all algorithms
  8. Multi-Seed Robustness (single-col) — box plot
  9. Multi-Seed Training Convergence (single-col) — line plot
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
# Algorithm registry  (order = display order in legends)
# ---------------------------------------------------------------------------
ALGO_ORDER = ["CSAC-LB", "CPO", "SAC-Lag", "PPO-Lag", "PPO", "CUP", "FOCOPS"]

ALGO_COLORS = {
    "CSAC-LB": "#c0392b",
    "CPO":     "#2471a3",
    "FOCOPS":  "#27ae60",
    "CUP":     "#e67e22",
    "PPO-Lag": "#8e44ad",
    "SAC-Lag": "#795548",
    "PPO":     "#c2185b",
}

ALGO_MARKERS = {
    "CSAC-LB": "o", "CPO": "s", "FOCOPS": "D", "CUP": "^",
    "PPO-Lag": "v", "SAC-Lag": "P", "PPO": "X",
}

BASELINE_COLORS = {"Cooling RBC": "#424242", "Zero": "#9e9e9e"}

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
EVAL_JSONS = {
    "CSAC-LB": BASE / "runs/csac_lb_multi_seed/seed_1/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-09-11-15-54/eval_case_study.json",
    "CPO":     BASE / "runs/cpo_temp_cooling_only/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-20-54-15/eval_case_study.json",
    "FOCOPS":  BASE / "runs/focops_temp_cooling_only/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-21-17-52/eval_case_study.json",
    "CUP":     BASE / "runs/cup_temp_cooling_only/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-21-43-13/eval_case_study.json",
    "PPO-Lag": BASE / "runs/ppolag_temp_cooling_only_tight_v2/PPOLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-11-44-33/eval_case_study.json",
    "SAC-Lag": BASE / "runs/saclag_temp_cooling_only_v1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-15-33-32/eval_case_study.json",
    "PPO":     BASE / "runs/ppo_temp_cooling_only_masked_reward_bc_40ep/PPOTempMasked-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-03-28-19-39-49/eval_case_study.json",
}

PROGRESS_CSVS = {
    "CSAC-LB": BASE / "runs/csac_lb_multi_seed/seed_1/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-09-11-15-54/progress.csv",
    "CPO":     BASE / "runs/cpo_temp_cooling_only/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-20-54-15/progress.csv",
    "FOCOPS":  BASE / "runs/focops_temp_cooling_only/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-21-17-52/progress.csv",
    "CUP":     BASE / "runs/cup_temp_cooling_only/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-21-43-13/progress.csv",
    "PPO-Lag": BASE / "runs/ppolag_temp_cooling_only_tight_v2/PPOLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-11-44-33/progress.csv",
    "SAC-Lag": BASE / "runs/saclag_temp_cooling_only_v1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-15-33-32/progress.csv",
    "PPO":     BASE / "runs/ppo_temp_cooling_only_masked_reward_bc_40ep/PPOTempMasked-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-03-28-19-39-49/progress.csv",
}

CSAC_LB_SEEDS = {
    0:  BASE / "runs/csac_lb_multi_seed/seed_0/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-09-10-08-23",
    1:  BASE / "runs/csac_lb_multi_seed/seed_1/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-09-11-15-54",
    2:  BASE / "runs/csac_lb_multi_seed/seed_2/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-002-2026-04-09-12-26-54",
    42: BASE / "runs/csac_lb_multi_seed/seed_42/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-09-13-41-32",
}

STRESS_JSON = BASE / "runs/stress_test_72h_all_algos.json"


# ===== Helpers ==============================================================

def load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [WARN] {path.name}: {e}")
        return None


def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [WARN] {path.name}: {e}")
        return None


def get_best(data: dict) -> dict | None:
    for r in data["rows"]:
        if "best" in r["name"].lower():
            return r
    return None


def get_baseline(data: dict, name: str) -> dict | None:
    for r in data["rows"]:
        if name.lower() in r["name"].lower():
            return r
    return None


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  -> {stem}.pdf/.png")


def _load_all_best() -> dict[str, dict]:
    """Load best-checkpoint eval data for every algorithm + baselines."""
    results: dict[str, dict] = {}
    for algo in ALGO_ORDER:
        data = load_json(EVAL_JSONS[algo])
        if data is None:
            continue
        row = get_best(data)
        if row:
            results[algo] = row
    # Baselines from first available
    ref = load_json(next(iter(EVAL_JSONS.values())))
    if ref:
        for bl in ("Cooling RBC", "Zero"):
            r = get_baseline(ref, bl)
            if r:
                results[bl] = r
    return results


def _parse_stress_best() -> dict[str, dict]:
    """Return one entry per algorithm from stress test (lowest violation)."""
    stress = load_json(STRESS_JSON)
    if stress is None:
        return {}
    groups: dict[str, list[tuple[str, dict]]] = {}
    for key, vals in stress["summary"].items():
        lbl = key.strip()
        if "RBC" in lbl:
            algo = "Cooling RBC"
        elif "Zero" in lbl:
            algo = "Zero"
        elif "CSAC-LB" in lbl:
            algo = "CSAC-LB"
        elif "CPO" in lbl:
            algo = "CPO"
        elif "FOCOPS" in lbl:
            algo = "FOCOPS"
        elif "CUP" in lbl:
            algo = "CUP"
        elif "PPO-Lag" in lbl:
            algo = "PPO-Lag"
        elif "SAC-Lag" in lbl:
            algo = "SAC-Lag"
        elif "PPO" in lbl:
            algo = "PPO"
        else:
            algo = lbl
        groups.setdefault(algo, []).append((lbl, vals))
    return {a: min(es, key=lambda x: x[1].get("violation_rate", 999))[1]
            for a, es in groups.items()}


def _bar_annotate(ax, bars, values, fmt="{:.1f}%", fontsize=6,
                  inside_thresh=0.75, max_val=None):
    """Annotate horizontal bars without overlap."""
    if max_val is None:
        max_val = max(values) if values else 100
    for bar, val in zip(bars, values):
        txt = fmt.format(val)
        if val / max_val > inside_thresh:
            x = bar.get_width() - max_val * 0.02
            ha = "right"
            color = "white"
        else:
            x = bar.get_width() + max_val * 0.015
            ha = "left"
            color = "black"
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                txt, va="center", ha=ha, fontsize=fontsize, color=color)


# ===== Figure 1: Training Dynamics ==========================================

def fig_training_dynamics() -> None:
    print("[Fig 1] Training dynamics ...")
    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(COL2, 2.4))
    fig.subplots_adjust(wspace=0.30, left=0.07, right=0.98, top=0.90, bottom=0.17)

    sac_family = {"CSAC-LB", "SAC-Lag"}

    for algo in ALGO_ORDER:
        df = load_csv(PROGRESS_CSVS[algo])
        if df is None:
            continue
        col = ALGO_COLORS[algo]
        ep = df["Train/Epoch"].values if "Train/Epoch" in df.columns else np.arange(len(df))

        c_col = "Metrics/TestEpCost" if (algo in sac_family and "Metrics/TestEpCost" in df.columns) else "Metrics/EpCost"
        r_col = "Metrics/TestEpRet" if (algo in sac_family and "Metrics/TestEpRet" in df.columns) else "Metrics/EpRet"

        ax_c.plot(ep, df[c_col].values, color=col, label=algo, alpha=0.85)
        ax_r.plot(ep, df[r_col].values, color=col, label=algo, alpha=0.85)

    for ax, ylabel, sub, loc in [
        (ax_c, "Episode Cost",   "(a) Constraint Cost", "upper right"),
        (ax_r, "Episode Return", "(b) Cumulative Return", "lower right"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(sub, fontsize=8, pad=4)
        ax.legend(loc=loc, ncol=2, fontsize=5.5,
                  handlelength=1.4, columnspacing=0.7, handletextpad=0.3,
                  borderpad=0.4)

    save(fig, "fig_training_dynamics")


# ===== Figure 1b: Loss Curves ==============================================

def fig_loss_curves() -> None:
    print("[Fig 1b] Loss curves ...")
    fig, (ax_pi, ax_rc, ax_cc) = plt.subplots(1, 3, figsize=(COL2, 2.4))
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.98, top=0.86, bottom=0.17)

    for algo in ALGO_ORDER:
        df = load_csv(PROGRESS_CSVS[algo])
        if df is None:
            continue
        col = ALGO_COLORS[algo]
        ep = df["Train/Epoch"].values if "Train/Epoch" in df.columns else np.arange(len(df))

        # Policy loss
        if "Loss/Loss_pi" in df.columns:
            ax_pi.plot(ep, df["Loss/Loss_pi"].values, color=col,
                       label=algo, alpha=0.85)

        # Reward critic loss
        if "Loss/Loss_reward_critic" in df.columns:
            ax_rc.plot(ep, df["Loss/Loss_reward_critic"].values, color=col,
                       label=algo, alpha=0.85)

        # Cost critic loss (not available for unconstrained PPO)
        if "Loss/Loss_cost_critic" in df.columns:
            ax_cc.plot(ep, df["Loss/Loss_cost_critic"].values, color=col,
                       label=algo, alpha=0.85)

    for ax, ylabel, sub in [
        (ax_pi, "Policy Loss",        "(a) Policy Loss"),
        (ax_rc, "Reward Critic Loss",  "(b) Reward Critic Loss"),
        (ax_cc, "Cost Critic Loss",    "(c) Cost Critic Loss"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(sub, fontsize=8, pad=4)
        ax.legend(loc="best", ncol=2, fontsize=5,
                  handlelength=1.2, columnspacing=0.6, handletextpad=0.3,
                  borderpad=0.3)

    save(fig, "fig_loss_curves")


# ===== Figure 1c: Train–Eval Gap (CSAC-LB) =================================

def fig_train_eval_gap() -> None:
    """Show train vs eval metrics for CSAC-LB (only algo with both logged).
    On-policy algorithms use rollout data for both training and evaluation,
    so the gap is only meaningful for off-policy methods."""
    print("[Fig 1c] Train–eval gap (CSAC-LB) ...")

    df = load_csv(PROGRESS_CSVS["CSAC-LB"])
    if df is None or "Metrics/TestEpCost" not in df.columns:
        print("  [SKIP] No separate test metrics.")
        return

    ep = df["Train/Epoch"].values
    col = ALGO_COLORS["CSAC-LB"]

    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(COL2, 2.3))
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.98, top=0.86, bottom=0.17)

    # Cost: train vs eval
    ax_c.plot(ep, df["Metrics/EpCost"].values, color=col, ls="--",
              alpha=0.7, label="Train", lw=0.9)
    ax_c.plot(ep, df["Metrics/TestEpCost"].values, color=col, ls="-",
              alpha=0.9, label="Eval", lw=1.1)
    ax_c.fill_between(ep, df["Metrics/EpCost"].values,
                       df["Metrics/TestEpCost"].values,
                       color=col, alpha=0.08)

    # Return: train vs eval
    ax_r.plot(ep, df["Metrics/EpRet"].values, color=col, ls="--",
              alpha=0.7, label="Train", lw=0.9)
    ax_r.plot(ep, df["Metrics/TestEpRet"].values, color=col, ls="-",
              alpha=0.9, label="Eval", lw=1.1)
    ax_r.fill_between(ep, df["Metrics/EpRet"].values,
                       df["Metrics/TestEpRet"].values,
                       color=col, alpha=0.08)

    for ax, ylabel, sub in [
        (ax_c, "Episode Cost",   "(a) Constraint Cost"),
        (ax_r, "Episode Return", "(b) Cumulative Return"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(sub, fontsize=8, pad=4)
        ax.legend(loc="best", fontsize=6.5)

    fig.suptitle("CSAC-LB: Training vs Evaluation Gap", fontsize=8, y=0.97)

    save(fig, "fig_train_eval_gap")


# ===== Figure 2: Comfort Violation Rate =====================================

def fig_violation_rate() -> None:
    print("[Fig 2] Comfort violation rate ...")
    all_best = _load_all_best()

    entries = [(algo, all_best[algo]["violation_rate"] * 100,
                ALGO_COLORS.get(algo, BASELINE_COLORS.get(algo, "#999")))
               for algo in list(ALGO_ORDER) + ["Cooling RBC", "Zero"]
               if algo in all_best]

    entries.sort(key=lambda x: x[1])
    labels, rates, colors = zip(*entries)

    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.92, bottom=0.14)

    bars = ax.barh(range(len(labels)), rates, color=colors,
                   edgecolor="white", linewidth=0.3, height=0.55)
    _bar_annotate(ax, bars, rates, max_val=max(rates))

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Comfort Violation Rate (%)")
    ax.set_title("Evaluation Period — Comfort Violations", fontsize=8, pad=4)
    ax.set_xlim(0, max(rates) * 1.10)

    save(fig, "fig_violation_rate")


# ===== Figure 3: Energy–Comfort Tradeoff ====================================

def fig_energy_comfort_tradeoff() -> None:
    print("[Fig 3] Energy–comfort tradeoff ...")
    from matplotlib.lines import Line2D

    all_best = _load_all_best()

    fig, ax = plt.subplots(figsize=(COL1, 2.8))
    fig.subplots_adjust(left=0.16, right=0.96, top=0.90, bottom=0.15)

    # Collect positions for smart label placement
    points = []  # (x, y, label, color, marker)

    for algo in ALGO_ORDER:
        if algo not in all_best:
            continue
        r = all_best[algo]
        vr = r["violation_rate"] * 100
        ecost = r["kpi::cost_total"]  # normalised energy cost (1.0 = RBC)
        ax.scatter(vr, ecost, c=ALGO_COLORS[algo], marker=ALGO_MARKERS[algo],
                   s=50, edgecolors="black", linewidths=0.4, zorder=5)
        points.append((vr, ecost, algo))

    # Baselines
    for bl, bc in BASELINE_COLORS.items():
        if bl in all_best:
            r = all_best[bl]
            vr = r["violation_rate"] * 100
            ecost = r["kpi::cost_total"]
            ax.scatter(vr, ecost, c=bc, marker="*", s=65,
                       edgecolors="black", linewidths=0.4, zorder=5)
            points.append((vr, ecost, bl))

    # Smart label placement — avoid overlap
    _annotate_no_overlap(ax, points, fontsize=6.5)

    ax.set_xlabel("Comfort Violation Rate (%)")
    ax.set_ylabel("Normalised Energy Cost")
    ax.set_title("Energy–Comfort Tradeoff", fontsize=8, pad=4)

    # Compact legend
    handles = []
    for algo in ALGO_ORDER:
        if algo in all_best:
            handles.append(Line2D([0], [0], marker=ALGO_MARKERS[algo], color="w",
                                  markerfacecolor=ALGO_COLORS[algo],
                                  markeredgecolor="black", markeredgewidth=0.3,
                                  markersize=4.5, label=algo))
    for bl, bc in BASELINE_COLORS.items():
        if bl in all_best:
            handles.append(Line2D([0], [0], marker="*", color="w",
                                  markerfacecolor=bc, markersize=6, label=bl))
    ax.legend(handles=handles, loc="best", fontsize=5.5, ncol=2,
              handlelength=1.0, columnspacing=0.5, handletextpad=0.2)

    save(fig, "fig_energy_comfort_tradeoff")


def _annotate_no_overlap(ax, points, fontsize=6.5):
    """Place labels with manual offset adjustments to avoid overlap."""
    # Pre-defined offsets per algo to avoid overlap (tuned for this data)
    offsets = {
        "CSAC-LB":    (8, -8),
        "CPO":        (8, 8),
        "SAC-Lag":    (8, -10),
        "PPO-Lag":    (8, 4),
        "PPO":        (-40, -10),
        "CUP":        (8, 4),
        "FOCOPS":     (8, -8),
        "Cooling RBC": (8, 4),
        "Zero":       (-30, 6),
    }
    for x, y, lbl in points:
        ofs = offsets.get(lbl, (8, 0))
        style = {"fontstyle": "italic"} if lbl in BASELINE_COLORS else {}
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=ofs, fontsize=fontsize, **style)


# ===== Figure 4: Hot vs Cold Discomfort =====================================

def fig_hot_cold_discomfort() -> None:
    print("[Fig 4] Hot vs cold discomfort ...")
    all_best = _load_all_best()

    display_order = [a for a in ALGO_ORDER + ["Cooling RBC"] if a in all_best]

    hot = [all_best[a]["kpi::discomfort_hot_proportion"] * 100 for a in display_order]
    cold = [all_best[a]["kpi::discomfort_cold_proportion"] * 100 for a in display_order]

    x = np.arange(len(display_order))
    w = 0.35

    fig, ax = plt.subplots(figsize=(COL2, 2.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18)

    bars_hot = ax.bar(x - w / 2, hot, w, label="Overheating",
                      color="#e74c3c", edgecolor="white", linewidth=0.3)
    bars_cold = ax.bar(x + w / 2, cold, w, label="Overcooling",
                       color="#3498db", edgecolor="white", linewidth=0.3)

    # Value labels
    for bars, vals in [(bars_hot, hot), (bars_cold, cold)]:
        for bar, v in zip(bars, vals):
            if v > 1.0:  # only annotate non-trivial
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(display_order, fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Discomfort Breakdown — Overheating vs Overcooling", fontsize=8, pad=4)
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_ylim(0, max(max(hot), max(cold)) * 1.15)

    save(fig, "fig_hot_cold_discomfort")


# ===== Figure 5: Stress Test Violation Rate =================================

def fig_stress_violation() -> None:
    print("[Fig 5] Stress test violation rate ...")
    best_per_algo = _parse_stress_best()
    if not best_per_algo:
        print("  [SKIP] No data.")
        return

    display_order = [a for a in ALGO_ORDER + ["Cooling RBC", "Zero"]
                     if a in best_per_algo]

    entries = [(a, best_per_algo[a]["violation_rate"] * 100,
                ALGO_COLORS.get(a, BASELINE_COLORS.get(a, "#999")))
               for a in display_order]
    entries.sort(key=lambda x: x[1])
    labels, rates, colors = zip(*entries)

    fig, ax = plt.subplots(figsize=(COL1, 2.8))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.90, bottom=0.14)

    bars = ax.barh(range(len(labels)), rates, color=colors,
                   edgecolor="white", linewidth=0.3, height=0.55)
    _bar_annotate(ax, bars, rates, max_val=110)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Comfort Violation Rate (%)")
    ax.set_title("72-h Heat Wave Stress Test", fontsize=8, pad=4)
    ax.set_xlim(0, 110)

    save(fig, "fig_stress_test_violation")


# ===== Figure 6: Stress Test Multi-Metric ===================================

def fig_stress_multimetric() -> None:
    print("[Fig 6] Stress test multi-metric ...")
    best_per_algo = _parse_stress_best()
    if not best_per_algo:
        print("  [SKIP] No data.")
        return

    display_order = [a for a in ALGO_ORDER + ["Cooling RBC"]
                     if a in best_per_algo]

    metrics = [
        ("violation_rate",              "Violation\nRate (%)",        100, "{:.1f}"),
        ("kpi::cost_total",             "Normalised\nCost",           1,   "{:.2f}"),
        ("kpi::carbon_emissions_total", "Normalised\nEmissions",      1,   "{:.2f}"),
        ("kpi::ramping_average",        "Normalised\nRamping",        1,   "{:.2f}"),
    ]
    n_m = len(metrics)
    n_a = len(display_order)
    x = np.arange(n_a)
    w = 0.55

    fig, axes = plt.subplots(1, n_m, figsize=(COL2, 2.8))
    fig.subplots_adjust(wspace=0.40, left=0.05, right=0.98, top=0.82, bottom=0.25)

    for idx, (mkey, mlabel, scale, vfmt) in enumerate(metrics):
        ax = axes[idx]
        vals, cols = [], []
        for algo in display_order:
            v = best_per_algo[algo].get(mkey, 0)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 0
            vals.append(v * scale)
            cols.append(ALGO_COLORS.get(algo, BASELINE_COLORS.get(algo, "#999")))

        bars = ax.bar(x, vals, w, color=cols, edgecolor="white", linewidth=0.2)

        # Value labels on top
        ymax = max(vals) if vals else 1
        for xi, v in zip(x, vals):
            ax.text(xi, v + ymax * 0.02, vfmt.format(v),
                    ha="center", va="bottom", fontsize=4.5)

        ax.set_xticks(x)
        ax.set_xticklabels(display_order, fontsize=5, rotation=45, ha="right")
        ax.set_title(mlabel, fontsize=7, pad=3)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.set_ylim(0, ymax * 1.15)

    fig.suptitle("72-h Heat Wave Stress Test — Multi-Metric Comparison",
                 fontsize=8, y=0.94)

    save(fig, "fig_stress_test_multimetric")


# ===== Figure 7: Full KPI Comparison Heatmap ================================

def fig_kpi_heatmap() -> None:
    print("[Fig 7] KPI heatmap (all algorithms) ...")

    all_best = _load_all_best()

    display_order = [a for a in ALGO_ORDER + ["Cooling RBC"]
                     if a in all_best]

    kpi_keys = [
        "kpi::cost_total",
        "kpi::carbon_emissions_total",
        "kpi::electricity_consumption_total",
        "kpi::ramping_average",
        "kpi::daily_peak_average",
        "kpi::discomfort_proportion",
    ]
    kpi_labels = ["Cost", "Emissions", "Consumption",
                  "Ramping", "Peak Demand", "Discomfort"]

    # Build matrix (algos × KPIs)
    n_a = len(display_order)
    n_k = len(kpi_keys)
    matrix = np.zeros((n_a, n_k))
    for i, algo in enumerate(display_order):
        for j, k in enumerate(kpi_keys):
            v = all_best[algo].get(k, 0)
            matrix[i, j] = v if (v is not None and not np.isnan(v)) else 0

    fig, ax = plt.subplots(figsize=(COL2, 2.6))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.05)

    # Use diverging colormap centred at 1.0 (RBC baseline)
    vmin = min(0.5, matrix.min() - 0.05)
    vmax = max(2.5, matrix.max() + 0.05)
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=vmin, vmax=vmax)

    # Annotate cells
    for i in range(n_a):
        for j in range(n_k):
            v = matrix[i, j]
            # Choose text color for readability
            tc = "white" if v > 1.8 or v < 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color=tc, fontweight="bold" if v < 0.95 else "normal")

    ax.set_xticks(range(n_k))
    ax.set_xticklabels(kpi_labels, fontsize=7)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_yticks(range(n_a))
    ax.set_yticklabels(display_order, fontsize=7)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Normalised KPI (1.0 = RBC baseline)", fontsize=6.5)

    ax.set_title("Evaluation Period — Normalised KPI Comparison", fontsize=8, pad=10)

    # Grid lines between cells
    for i in range(n_a + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.8)
    for j in range(n_k + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.8)

    ax.grid(False)

    save(fig, "fig_kpi_heatmap")


# ===== Figure 8: Multi-Seed Robustness =====================================

def fig_multiseed_robustness() -> None:
    print("[Fig 8] Multi-seed robustness ...")

    rates = []
    seeds_sorted = sorted(CSAC_LB_SEEDS.keys())

    for seed in seeds_sorted:
        data = load_json(CSAC_LB_SEEDS[seed] / "eval_case_study.json")
        if data is None:
            continue
        b = get_best(data)
        if b:
            rates.append(b["violation_rate"] * 100)

    if not rates:
        print("  [SKIP] No data.")
        return

    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    fig.subplots_adjust(left=0.15, right=0.90, top=0.88, bottom=0.13)

    bp = ax.boxplot([rates], positions=[1], widths=0.4, patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=4),
                    medianprops=dict(color="black", linewidth=0.8),
                    whiskerprops=dict(linewidth=0.6),
                    capprops=dict(linewidth=0.6))
    bp["boxes"][0].set_facecolor(ALGO_COLORS["CSAC-LB"])
    bp["boxes"][0].set_alpha(0.40)

    # Individual seed points (fixed jitter for reproducibility)
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.08, 0.08, len(rates))
    ax.scatter(1 + jitter, rates, c=ALGO_COLORS["CSAC-LB"], s=22, zorder=5,
               edgecolors="black", linewidths=0.3)

    # Seed annotations — stagger vertically if close
    sorted_by_val = sorted(enumerate(zip(rates, seeds_sorted)),
                           key=lambda x: x[1][0])
    prev_y = -999
    for rank, (i, (rate, seed)) in enumerate(sorted_by_val):
        y_shift = 0
        if abs(rate - prev_y) < 1.5:
            y_shift = 5
        ax.annotate(f"seed {seed}: {rate:.1f}%",
                    (1 + jitter[i], rate),
                    textcoords="offset points",
                    xytext=(14, y_shift), fontsize=5.5, ha="left", va="center")
        prev_y = rate

    # Mean annotation
    m = np.mean(rates)
    ax.annotate(f"mean = {m:.1f}%", (1.28, m), fontsize=6.5, va="center",
                fontstyle="italic")

    ax.set_xticks([1])
    ax.set_xticklabels(["CSAC-LB (4 seeds)"])
    ax.set_ylabel("Comfort Violation Rate (%)")
    ax.set_title("Multi-Seed Robustness", fontsize=8, pad=4)

    save(fig, "fig_multiseed_robustness")


# ===== Figure 9: Multi-Seed Training Convergence ============================

def fig_multiseed_training() -> None:
    print("[Fig 9] Multi-seed training convergence ...")

    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.16)

    cmap = plt.cm.Reds
    seed_colors = {0: cmap(0.85), 1: cmap(0.65), 2: cmap(0.45), 42: cmap(0.30)}
    seed_ls = {0: "-", 1: "--", 2: "-.", 42: ":"}

    for seed, seed_dir in sorted(CSAC_LB_SEEDS.items()):
        df = load_csv(seed_dir / "progress.csv")
        if df is None:
            continue
        ep = df["Train/Epoch"].values if "Train/Epoch" in df.columns else np.arange(len(df))
        c_col = "Metrics/TestEpCost" if "Metrics/TestEpCost" in df.columns else "Metrics/EpCost"
        ax.plot(ep, df[c_col].values, color=seed_colors[seed],
                ls=seed_ls[seed], lw=1.0, label=f"Seed {seed}", alpha=0.85)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Episode Cost")
    ax.set_title("CSAC-LB Cost Convergence Across Seeds", fontsize=8, pad=4)
    ax.legend(loc="upper right", fontsize=6)

    save(fig, "fig_multiseed_training")


# ===== Main =================================================================

def main() -> None:
    print("=" * 60)
    print("Benchmark Plotting — IEEE Style (v2)")
    print(f"Output: {FIG_DIR}")
    print("=" * 60)

    fig_training_dynamics()        # 1a
    fig_loss_curves()              # 1b
    fig_train_eval_gap()           # 1c
    fig_violation_rate()            # 2
    fig_energy_comfort_tradeoff()   # 3
    fig_hot_cold_discomfort()       # 4
    fig_stress_violation()          # 5
    fig_stress_multimetric()        # 6
    fig_kpi_heatmap()               # 7
    fig_multiseed_robustness()      # 8
    fig_multiseed_training()        # 9

    print("=" * 60)
    print("All 11 figures generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
