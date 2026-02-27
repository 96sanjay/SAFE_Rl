
"""
Paper-quality plots for baseline evaluation metrics (v2).

Fixes the "incomplete-looking" plots by avoiding mixed-units on one axis.
Creates:
  - CMDP cost breakdown (stacked)
  - EV deficit split (stacked)
  - Battery safety (3 aligned subplots)
  - Economics (2 subplots)
  - Reward + SoC (3 aligned subplots)
  - All-metrics heatmap (z-scored) with raw annotations

Input:
  runs/baselines/evaluation/baseline_comparison.csv

Output:
  runs/baselines/evaluation/figures_v2/*.pdf and *.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Paper style (research-ready)
# -------------------------
def set_paper_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# Colorblind-friendly palette (Okabe-Ito inspired)
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "gray":   "#4D4D4D",
}


def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def safe_get(df: pd.DataFrame, col: str, default=0.0):
    if col in df.columns:
        return df[col].astype(float).values
    return np.full(len(df), default, dtype=float)


def annotate_bars(ax, x, y, fmt="{:.1f}", dy=0.01):
    """Annotate bars with values above each bar."""
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for xi, yi in zip(x, y):
        ax.text(xi, yi + dy*y_range, fmt.format(yi), ha="center", va="bottom", fontsize=9)


# -------------------------
# 1) CMDP COST (stacked)
# -------------------------
def plot_stacked_cmdp_cost(df: pd.DataFrame, out_dir: Path):
    agents = df["agent"].astype(str).tolist()
    x = np.arange(len(agents))

    total = safe_get(df, "cmdp_cost_total")
    b_soc = safe_get(df, "cmdp_cost_building_soc")
    ev_dep = safe_get(df, "cmdp_cost_ev_departure")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    ax.bar(x, b_soc, label="Building SoC cost", color=PALETTE["blue"])
    ax.bar(x, ev_dep, bottom=b_soc, label="EV departure cost", color=PALETTE["orange"])

    ax.set_title("CMDP Cost Breakdown")
    ax.set_ylabel("Cost (sum over episode)")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")

    for i in range(len(x)):
        ax.text(x[i], b_soc[i] + ev_dep[i], f"{total[i]:.1f}",
                ha="center", va="bottom", fontsize=9)

    ax.legend(frameon=False, ncols=2, loc="upper right")
    save_fig(fig, out_dir, "fig1_cmdp_cost_breakdown")


# -------------------------
# 2) EV DEFICIT (stacked)
# -------------------------
def plot_ev_deficit_split(df: pd.DataFrame, out_dir: Path):
    agents = df["agent"].astype(str).tolist()
    x = np.arange(len(agents))

    total = safe_get(df, "ev_deficit_total_kwh")
    avoid = safe_get(df, "ev_deficit_avoidable_kwh")
    unavoid = safe_get(df, "ev_deficit_unavoidable_kwh")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    ax.bar(x, avoid, label="Avoidable (policy)", color=PALETTE["red"])
    ax.bar(x, unavoid, bottom=avoid, label="Unavoidable (physics)", color=PALETTE["green"])

    ax.set_title("EV Departure Energy Deficit Split")
    ax.set_ylabel("Energy deficit (kWh, sum over episode)")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")

    for i in range(len(x)):
        ax.text(x[i], avoid[i] + unavoid[i], f"{total[i]:.1f}",
                ha="center", va="bottom", fontsize=9)

    ax.legend(frameon=False, ncols=2, loc="upper right")
    save_fig(fig, out_dir, "fig2_ev_deficit_split")


# -------------------------
# 3) BATTERY SAFETY (3 subplots, no mixed units)
# -------------------------
def plot_battery_safety_v2(df: pd.DataFrame, out_dir: Path):
    agents = df["agent"].astype(str).tolist()
    x = np.arange(len(agents))

    viol_pct = safe_get(df, "soc_violation_pct")
    abuse_kwh = safe_get(df, "battery_abuse_total_kwh")
    abuse_hours = safe_get(df, "battery_abuse_hours")

    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(7.2, 6.8), sharex=True
    )

    # (a) violation %
    ax = axes[0]
    ax.bar(x, viol_pct, color=PALETTE["purple"])
    ax.set_title("Battery Safety Metrics")
    ax.set_ylabel("SoC violations (%)")
    annotate_bars(ax, x, viol_pct, fmt="{:.1f}")

    # (b) abuse kWh
    ax = axes[1]
    ax.bar(x, abuse_kwh, color=PALETTE["orange"])
    ax.set_ylabel("Battery abuse (kWh)")
    annotate_bars(ax, x, abuse_kwh, fmt="{:.0f}")

    # (c) abuse hours
    ax = axes[2]
    ax.bar(x, abuse_hours, color=PALETTE["blue"])
    ax.set_ylabel("Abuse time (hours)")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")
    annotate_bars(ax, x, abuse_hours, fmt="{:.0f}")

    save_fig(fig, out_dir, "fig3_battery_safety_metrics_v2")


# -------------------------
# 4) ECONOMICS (2 subplots)
# -------------------------
def plot_economic_metrics_v2(df: pd.DataFrame, out_dir: Path):
    agents = df["agent"].astype(str).tolist()
    x = np.arange(len(agents))

    imp = safe_get(df, "grid_import_total_kwh")
    exp = safe_get(df, "grid_export_total_kwh")
    net = safe_get(df, "net_grid_consumption_kwh")
    cost = safe_get(df, "electricity_cost_total")

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.2, 5.6), sharex=True)

    ax = axes[0]
    w = 0.25
    ax.bar(x - w, imp, width=w, label="Import", color=PALETTE["blue"])
    ax.bar(x,     exp, width=w, label="Export", color=PALETTE["green"])
    ax.bar(x + w, net, width=w, label="Net",    color=PALETTE["gray"])
    ax.set_title("Grid Energy Metrics")
    ax.set_ylabel("Energy (kWh)")
    ax.legend(frameon=False, ncols=3, loc="upper right")

    ax = axes[1]
    ax.bar(x, cost, color=PALETTE["orange"])
    ax.set_title("Electricity Cost")
    ax.set_ylabel("Cost ($)")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")
    annotate_bars(ax, x, cost, fmt="{:.0f}")

    save_fig(fig, out_dir, "fig4_economic_metrics_v2")


# -------------------------
# 5) REWARD + SOC (3 subplots, fixes your “empty” reward plot)
# -------------------------
def plot_reward_and_soc_v2(df: pd.DataFrame, out_dir: Path):
    agents = df["agent"].astype(str).tolist()
    x = np.arange(len(agents))

    total_reward = safe_get(df, "total_reward")
    avg_reward = safe_get(df, "avg_reward_per_step")
    avg_soc = safe_get(df, "avg_soc")

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7.2, 6.8), sharex=True)

    ax = axes[0]
    ax.bar(x, total_reward, color=PALETTE["red"])
    ax.set_title("Reward and SoC Summary")
    ax.set_ylabel("Total reward")
    annotate_bars(ax, x, total_reward, fmt="{:.0f}")

    ax = axes[1]
    ax.bar(x, avg_reward, color=PALETTE["orange"])
    ax.set_ylabel("Avg reward/step")
    annotate_bars(ax, x, avg_reward, fmt="{:.2f}")

    ax = axes[2]
    ax.bar(x, avg_soc, color=PALETTE["blue"])
    ax.set_ylabel("Avg SoC")
    ax.set_ylim(0.0, 1.0)  # SoC is naturally [0,1]
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")
    annotate_bars(ax, x, avg_soc, fmt="{:.3f}")

    save_fig(fig, out_dir, "fig5_reward_soc_summary_v2")


# -------------------------
# 6) ALL METRICS HEATMAP (includes every computed metric)
# -------------------------
def plot_all_metrics_heatmap(df: pd.DataFrame, out_dir: Path):
    ignore_cols = {"agent"}
    metric_cols = [c for c in df.columns if c not in ignore_cols]

    preferred_order = [
        "total_steps",
        "cmdp_cost_total", "cmdp_cost_building_soc", "cmdp_cost_ev_departure", "cmdp_cost_per_step",
        "ev_deficit_total_kwh", "ev_deficit_avoidable_kwh", "ev_deficit_unavoidable_kwh",
        "ev_deficit_avoidable_pct", "ev_failures_count",
        "soc_violation_count", "soc_violation_pct", "battery_abuse_total_kwh", "battery_abuse_hours", "soc_max_observed",
        "electricity_cost_total", "grid_import_total_kwh", "grid_export_total_kwh", "net_grid_consumption_kwh",
        "carbon_emissions_kg",
        "total_reward", "avg_reward_per_step", "avg_soc"
    ]
    metric_cols = [c for c in preferred_order if c in metric_cols] + [c for c in metric_cols if c not in preferred_order]

    agents = df["agent"].astype(str).tolist()

    raw = df[metric_cols].astype(float)

    z = raw.copy()
    for col in metric_cols:
        v = raw[col].values.astype(float)
        mu = np.nanmean(v)
        sigma = np.nanstd(v)
        z[col] = 0.0 if (sigma == 0 or np.isnan(sigma)) else (v - mu) / sigma

    mat = z.values.T  # metrics x agents

    fig = plt.figure(figsize=(8.2, max(4.2, 0.34 * len(metric_cols))))
    ax = fig.add_subplot(111)

    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-2.0, vmax=2.0)

    ax.set_title("All Evaluation Metrics (row-wise z-score; annotated with raw values)")
    ax.set_xticks(np.arange(len(agents)))
    ax.set_xticklabels(agents, rotation=15, ha="right")

    ax.set_yticks(np.arange(len(metric_cols)))
    ax.set_yticklabels(metric_cols)

    for i in range(len(metric_cols)):
        for j in range(len(agents)):
            val = raw.iloc[j, i]
            if abs(val) >= 1000:
                txt = f"{val:,.0f}"
            elif abs(val) >= 10:
                txt = f"{val:,.1f}"
            else:
                txt = f"{val:,.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("z-score (per metric across agents)")

    save_fig(fig, out_dir, "fig6_all_metrics_heatmap")


def main():
    set_paper_style()

    eval_dir = Path("runs/baselines/evaluation")
    comparison_csv = eval_dir / "baseline_comparison.csv"
    out_dir = eval_dir / "figures_v2"

    if not comparison_csv.exists():
        raise FileNotFoundError(f"Could not find: {comparison_csv}")

    df = pd.read_csv(comparison_csv)
    if "agent" not in df.columns:
        raise ValueError("baseline_comparison.csv must contain an 'agent' column.")
    df["agent"] = df["agent"].astype(str)

    plot_stacked_cmdp_cost(df, out_dir)
    plot_ev_deficit_split(df, out_dir)
    plot_battery_safety_v2(df, out_dir)
    plot_economic_metrics_v2(df, out_dir)
    plot_reward_and_soc_v2(df, out_dir)
    plot_all_metrics_heatmap(df, out_dir)

    print(f"\n✅ Plots saved to: {out_dir.resolve()}")
    print("   (PDF + PNG for each figure)")


if __name__ == "__main__":
    main()
