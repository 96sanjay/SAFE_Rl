#!/usr/bin/env python3
"""Generate thesis figures for the temperature control case study.

RBC vs CSAC-LB only. No best/final distinction. IEEE-style plots.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
})

BASE = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs"
CSAC_DIR = os.path.join(
    BASE,
    "csac_lb_temp_strict_40ep",
    "CSACLBTemp-{CityLearnTemp-CoolingOnly-CSACLB-v0}",
    "seed-000-2026-03-27-23-30-03",
)
OUT_DIR = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/docs/temperature_case_study/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────

csac_prog = pd.read_csv(os.path.join(CSAC_DIR, "progress.csv"))

with open(os.path.join(CSAC_DIR, "eval_case_study_bangbang.json")) as f:
    csac_eval = json.load(f)
with open(os.path.join(CSAC_DIR, "eval_stems_heatwave_72h_25p0.json")) as f:
    hw72 = json.load(f)


def get_row(eval_data, name):
    for r in eval_data["rows"]:
        if r["name"] == name:
            return r
    return None


rbc_row = get_row(csac_eval, "Cooling RBC")
csac_row = get_row(csac_eval, "CSAC-LB final")


# ── Figure 1: Training Curves (2x2 grid) ────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1a: Episode reward
ax = axes[0, 0]
ax.plot(csac_prog["Train/Epoch"], csac_prog["Metrics/TestEpRet"],
        label="CSAC-LB (eval)", color="#d62728", linewidth=1.5)
ax.plot(csac_prog["Train/Epoch"], csac_prog["Metrics/EpRet"],
        label="CSAC-LB (train)", color="#d62728", linewidth=1.0, alpha=0.4, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Episode Return")
ax.set_title("(a) Episode Return during Training")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# 1b: Episode cost
ax = axes[0, 1]
ax.plot(csac_prog["Train/Epoch"], csac_prog["Metrics/TestEpCost"],
        label="CSAC-LB (eval)", color="#d62728", linewidth=1.5)
ax.plot(csac_prog["Train/Epoch"], csac_prog["Metrics/EpCost"],
        label="CSAC-LB (train)", color="#d62728", linewidth=1.0, alpha=0.4, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Episode Cost")
ax.set_title("(b) Comfort Violation Cost during Training")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# 1c: Actor loss
ax = axes[1, 0]
ax.plot(csac_prog["Train/Epoch"], csac_prog["Loss/Loss_pi"],
        label="CSAC-LB", color="#d62728", linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Policy Loss")
ax.set_title("(c) Actor Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# 1d: CSAC-LB specific: barrier penalty and cost critic
ax = axes[1, 1]
ax2 = ax.twinx()
ax.plot(csac_prog["Train/Epoch"], csac_prog["Value/barrier_penalty"],
        label="Barrier penalty", color="#d62728", linewidth=1.5)
ax2.plot(csac_prog["Train/Epoch"], csac_prog["Value/cost_critic_max"],
         label="Max cost Q-value", color="#ff7f0e", linewidth=1.5, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Barrier Penalty", color="#d62728")
ax2.set_ylabel("Cost Q-value", color="#ff7f0e")
ax.set_title("(d) CSAC-LB Barrier Penalty & Cost Critic")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
ax.grid(True, alpha=0.3)

fig.suptitle("CSAC-LB Training Dynamics", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "fig_training_curves.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_training_curves.png"))
plt.close(fig)
print("[1/6] Training curves saved.")


# ── Figure 2: Pareto front (energy vs comfort) ──────────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

csac_costs = csac_prog["Metrics/TestEpCost"].values
csac_rets = csac_prog["Metrics/TestEpRet"].values
epochs = csac_prog["Train/Epoch"].values.astype(int)

scatter = ax.scatter(
    csac_rets, csac_costs,
    c=epochs, cmap="RdYlGn_r", s=50, edgecolors="k", linewidth=0.5,
    zorder=3, label="CSAC-LB (per epoch)",
)
cbar = plt.colorbar(scatter, ax=ax, label="Training Epoch")

# Mark final epoch (our chosen model)
final_idx = len(epochs) - 1
ax.scatter([csac_rets[final_idx]], [csac_costs[final_idx]],
           marker="*", s=200, c="#d62728", edgecolors="k", linewidth=1, zorder=4,
           label=f"CSAC-LB (ep {epochs[final_idx]})")

# RBC reference
ax.scatter([rbc_row["total_reward"]], [rbc_row["total_cost"]],
           marker="P", s=200, c="black", zorder=5, label="RBC")

ax.set_xlabel("Episode Return (higher is better)")
ax.set_ylabel("Episode Cost (lower is better)")
ax.set_title("Pareto Front: Reward vs Comfort Violation Cost")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_pareto_front.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_pareto_front.png"))
plt.close(fig)
print("[2/6] Pareto front saved.")


# ── Figure 3: Evaluation bar chart — RBC vs CSAC-LB only ────────────────────

methods = ["RBC", "CSAC-LB"]
rows = [rbc_row, csac_row]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

metrics_top = [
    ("violation_rate", "Comfort Violation Rate", True),
    ("kpi::discomfort_proportion", "Discomfort Proportion", True),
    ("kpi::cost_total", "Cost (KPI ratio)", True),
]
metrics_bot = [
    ("kpi::carbon_emissions_total", "Carbon Emissions (KPI)", True),
    ("kpi::electricity_consumption_total", "Electricity Consumption (KPI)", True),
    ("kpi::ramping_average", "Ramping Average (KPI)", True),
]

colors_bar = ["#7f8c8d", "#d62728"]

for idx, (key, title, lower_better) in enumerate(metrics_top + metrics_bot):
    r, c = divmod(idx, 3)
    ax = axes[r, c]
    vals = [row[key] if row else 0 for row in rows]
    bars = ax.bar(range(len(methods)), vals, color=colors_bar,
                  edgecolor="black", linewidth=0.5, width=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_title(title)
    ax.grid(True, alpha=0.2, axis="y")

    # Value labels on bars
    for bar, val in zip(bars, vals):
        fmt = f"{val:.1%}" if "rate" in key or "proportion" in key else f"{val:.3f}"
        ax.annotate(fmt, xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")

fig.suptitle("Full-Year Evaluation: RBC vs CSAC-LB", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "fig_eval_bar_chart.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_eval_bar_chart.png"))
plt.close(fig)
print("[3/6] Evaluation bar chart saved.")


# ── Figure 4: Heat-wave comparison — 72h only ───────────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

labels = ["Violation\nRate", "Discomfort\nProportion", "Cost\n(KPI)",
          "Emissions\n(KPI)", "Ramping\n(KPI)"]
rbc_vals = [
    hw72["cooling_rbc_avg"]["violation_rate"],
    hw72["cooling_rbc_avg"].get("kpi::discomfort_proportion", 0),
    hw72["cooling_rbc_avg"]["kpi::cost_total"],
    hw72["cooling_rbc_avg"]["kpi::carbon_emissions_total"],
    hw72["cooling_rbc_avg"]["kpi::ramping_average"],
]
csac_vals = [
    hw72["csac_lb_avg"]["violation_rate"],
    hw72["csac_lb_avg"].get("kpi::discomfort_proportion", 0),
    hw72["csac_lb_avg"]["kpi::cost_total"],
    hw72["csac_lb_avg"]["kpi::carbon_emissions_total"],
    hw72["csac_lb_avg"]["kpi::ramping_average"],
]

x = np.arange(len(labels))
width = 0.3
bars1 = ax.bar(x - width / 2, rbc_vals, width, label="RBC",
               color="#7f8c8d", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width / 2, csac_vals, width, label="CSAC-LB",
               color="#d62728", edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title("72-Hour Heat-Wave Stress Test: RBC vs CSAC-LB")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2, axis="y")

# Annotate values
for bar, val in zip(list(bars1) + list(bars2), rbc_vals + csac_vals):
    fmt = f"{val:.1%}" if val < 1.0 else f"{val:.3f}"
    color = "#7f8c8d" if bar in bars1 else "#d62728"
    ax.annotate(fmt, xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold", color=color)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_heatwave_comparison.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_heatwave_comparison.png"))
plt.close(fig)
print("[4/6] Heat-wave comparison saved.")


# ── Figure 5: Training reward vs cost with RBC reference ────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(csac_prog["Train/Epoch"], csac_prog["Metrics/TestEpRet"],
        color="#d62728", linewidth=2, label="CSAC-LB")
ax.axhline(y=rbc_row["total_reward"], color="black", linestyle="--",
           linewidth=1.5, label="RBC")
ax.set_xlabel("Epoch")
ax.set_ylabel("Episode Return")
ax.set_title("(a) Episode Return (Energy Efficiency)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(csac_prog["Train/Epoch"], csac_prog["Metrics/TestEpCost"],
        color="#d62728", linewidth=2, label="CSAC-LB")
ax.axhline(y=rbc_row["total_cost"], color="black", linestyle="--",
           linewidth=1.5, label="RBC")
ax.set_xlabel("Epoch")
ax.set_ylabel("Episode Cost (Comfort Violations)")
ax.set_title("(b) Comfort Violation Cost")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("Training Performance: Energy vs Comfort Trade-off", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, "fig_training_energy_comfort.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_training_energy_comfort.png"))
plt.close(fig)
print("[5/6] Energy vs comfort training curves saved.")


# ── Figure 6: Radar chart — RBC vs CSAC-LB (both as separate polygons) ─────

categories = ["Cost", "Emissions", "Consumption",
              "Daily Peak", "Ramping"]

keys_kpi = [
    "kpi::cost_total",
    "kpi::carbon_emissions_total",
    "kpi::electricity_consumption_total",
    "kpi::daily_peak_average",
    "kpi::ramping_average",
]

rbc_vals = [rbc_row[k] for k in keys_kpi]
csac_vals = [csac_row[k] for k in keys_kpi]

N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))

# RBC polygon
rbc_plot = rbc_vals + rbc_vals[:1]
ax.plot(angles, rbc_plot, color="#333333", linewidth=2.5, label="RBC",
        marker="D", markersize=7, linestyle="--")
ax.fill(angles, rbc_plot, color="#333333", alpha=0.08)

# CSAC-LB polygon
csac_plot = csac_vals + csac_vals[:1]
ax.plot(angles, csac_plot, color="#d62728", linewidth=2.5, label="CSAC-LB",
        marker="o", markersize=7)
ax.fill(angles, csac_plot, color="#d62728", alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_title("Full-Year KPI Comparison\n(lower is better)", fontsize=13, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True, alpha=0.3)

# Add value labels
for angle, rv, cv in zip(angles[:-1], rbc_vals, csac_vals):
    ax.annotate(f"{rv:.3f}", xy=(angle, rv), xytext=(8, 8),
                textcoords="offset points", fontsize=8, color="#333333",
                fontweight="bold")
    ax.annotate(f"{cv:.3f}", xy=(angle, cv), xytext=(8, -10),
                textcoords="offset points", fontsize=8, color="#d62728",
                fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_radar_chart.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_radar_chart.png"))
plt.close(fig)
print("[6/6] Radar chart saved.")

print(f"\nAll figures saved to {OUT_DIR}/")
