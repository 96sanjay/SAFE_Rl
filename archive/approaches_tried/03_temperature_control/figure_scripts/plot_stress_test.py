#!/usr/bin/env python3
"""Generate stress test figures — 72h heat wave only. RBC vs CSAC-LB.

IEEE-style: serif font, clean axes, high DPI, clear method distinction.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

# Load data
with open(os.path.join(CSAC_DIR, "eval_case_study_bangbang.json")) as f:
    eval_full = json.load(f)
with open(os.path.join(CSAC_DIR, "eval_stems_heatwave_72h_25p0.json")) as f:
    hw72 = json.load(f)


def get_row(data, name):
    for r in data["rows"]:
        if r["name"] == name:
            return r
    return None


# Full-year data
full_rbc = get_row(eval_full, "Cooling RBC")
full_csac = get_row(eval_full, "CSAC-LB final")

# 72h stress test data
hw72_rbc = hw72["cooling_rbc_avg"]
hw72_csac = hw72["csac_lb_avg"]


# ── Figure 1: Radar chart — RBC vs CSAC-LB for 72h stress test ──────────────

categories = ["Cost", "Emissions", "Daily\nPeak",
              "Consumption", "Ramping"]

keys = [
    "kpi::cost_total",
    "kpi::carbon_emissions_total",
    "kpi::daily_peak_average",
    "kpi::electricity_consumption_total",
    "kpi::ramping_average",
]

# 72h stress test radar — both methods as separate polygons
rbc_vals = [hw72_rbc[k] for k in keys]
csac_vals = [hw72_csac[k] for k in keys]

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
ax.set_title("72h Heat-Wave KPI Comparison\n(lower is better)",
             fontsize=13, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True, alpha=0.3)

# Value labels
for angle, rv, cv in zip(angles[:-1], rbc_vals, csac_vals):
    ax.annotate(f"{rv:.3f}", xy=(angle, rv), xytext=(8, 8),
                textcoords="offset points", fontsize=8, color="#333333",
                fontweight="bold")
    ax.annotate(f"{cv:.3f}", xy=(angle, cv), xytext=(8, -10),
                textcoords="offset points", fontsize=8, color="#d62728",
                fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_stress_test_radar.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_stress_test_radar.png"))
plt.close(fig)
print("[1/4] Stress test radar chart saved.")


# ── Figure 2: Discomfort + Violations — Full-Year vs 72h ────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

scenarios = ["Full-Year", "72h Heat Wave"]

# Left: discomfort proportion
rbc_discomfort = [
    full_rbc["kpi::discomfort_proportion"],
    hw72_rbc.get("kpi::discomfort_proportion", 0),
]
csac_discomfort = [
    full_csac["kpi::discomfort_proportion"],
    hw72_csac.get("kpi::discomfort_proportion", 0),
]

x = np.arange(len(scenarios))
width = 0.3
bars1 = ax1.bar(x - width / 2, rbc_discomfort, width, label="RBC",
                color="#7f8c8d", edgecolor="black", linewidth=0.5)
bars2 = ax1.bar(x + width / 2, csac_discomfort, width, label="CSAC-LB",
                color="#d62728", edgecolor="black", linewidth=0.5)
ax1.set_ylabel("Discomfort Proportion (lower is better)")
ax1.set_title("(a) Discomfort Levels")
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, fontsize=10)
ax1.legend()
ax1.grid(True, alpha=0.2, axis="y")

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.001:
            ax1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=9, fontweight="bold")

# Right: violation rate
rbc_violations = [
    full_rbc["violation_rate"],
    hw72_rbc.get("violation_rate", 0),
]
csac_violations = [
    full_csac["violation_rate"],
    hw72_csac.get("violation_rate", 0),
]

bars3 = ax2.bar(x - width / 2, [v * 100 for v in rbc_violations], width,
                label="RBC", color="#7f8c8d", edgecolor="black", linewidth=0.5)
bars4 = ax2.bar(x + width / 2, [v * 100 for v in csac_violations], width,
                label="CSAC-LB", color="#d62728", edgecolor="black", linewidth=0.5)
ax2.set_ylabel("Comfort Violation Rate (%)")
ax2.set_title("(b) Safety Violations")
ax2.set_xticks(x)
ax2.set_xticklabels(scenarios, fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.2, axis="y")

for i in range(len(scenarios)):
    rbc_v = rbc_violations[i] * 100
    csac_v = csac_violations[i] * 100
    ax2.annotate(f"{rbc_v:.1f}%",
                 xy=(bars3[i].get_x() + bars3[i].get_width() / 2, rbc_v),
                 xytext=(0, 3), textcoords="offset points",
                 ha="center", fontsize=9, fontweight="bold")
    ax2.annotate(f"{csac_v:.1f}%",
                 xy=(bars4[i].get_x() + bars4[i].get_width() / 2, csac_v),
                 xytext=(0, 3), textcoords="offset points",
                 ha="center", fontsize=9, fontweight="bold", color="#d62728")
    if rbc_v > 0:
        reduction = (rbc_v - csac_v) / rbc_v * 100
        mid_x = (bars3[i].get_x() + bars4[i].get_x() + bars4[i].get_width()) / 2
        ax2.annotate(f"\u2212{reduction:.0f}%",
                     xy=(mid_x, max(rbc_v, csac_v) + 4),
                     ha="center", fontsize=10, color="green", fontweight="bold")

fig.suptitle("Thermal Comfort: Full-Year vs 72h Heat Wave",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, "fig_stress_test_discomfort_violations.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_stress_test_discomfort_violations.png"))
plt.close(fig)
print("[2/4] Discomfort and violations comparison saved.")


# ── Figure 3: Detailed 72h metric comparison ────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

metric_configs = [
    ("violation_rate", "Violation Rate", True),
    ("kpi::cost_total", "Cost (KPI)", True),
    ("kpi::carbon_emissions_total", "Emissions (KPI)", True),
    ("kpi::electricity_consumption_total", "Elec. Consumption (KPI)", True),
    ("kpi::ramping_average", "Ramping (KPI)", True),
    ("kpi::discomfort_proportion", "Discomfort Proportion", True),
]

scenarios = ["Full-Year", "72h Heat Wave"]

for idx, (key, title, lower_better) in enumerate(metric_configs):
    r, c = divmod(idx, 3)
    ax = axes[r, c]

    rbc_vals_m = [full_rbc.get(key, 0), hw72_rbc.get(key, 0)]
    csac_vals_m = [full_csac.get(key, 0), hw72_csac.get(key, 0)]

    x = np.arange(len(scenarios))
    width = 0.3
    ax.bar(x - width / 2, rbc_vals_m, width, label="RBC",
           color="#7f8c8d", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, csac_vals_m, width, label="CSAC-LB",
           color="#d62728", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_title(title)
    if idx == 0:
        ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

fig.suptitle("Full-Year vs 72h Heat Wave: Detailed Metrics",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT_DIR, "fig_stress_test_detailed_metrics.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_stress_test_detailed_metrics.png"))
plt.close(fig)
print("[3/4] Detailed stress test metrics saved.")


# ── Figure 4: Hot/Cold discomfort breakdown ──────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, data_csac, data_rbc, title in [
    (axes[0], full_csac, full_rbc, "Full-Year"),
    (axes[1], hw72_csac, hw72_rbc, "72h Heat Wave"),
]:
    labels = ["Hot\nDiscomfort", "Cold\nDiscomfort"]
    rbc_hc = [
        data_rbc.get("kpi::discomfort_hot_proportion", 0),
        data_rbc.get("kpi::discomfort_cold_proportion", 0),
    ]
    csac_hc = [
        data_csac.get("kpi::discomfort_hot_proportion", 0),
        data_csac.get("kpi::discomfort_cold_proportion", 0),
    ]

    x = np.arange(len(labels))
    width = 0.3
    ax.bar(x - width / 2, rbc_hc, width, label="RBC",
           color="#e74c3c", edgecolor="black", linewidth=0.5, alpha=0.6)
    ax.bar(x + width / 2, csac_hc, width, label="CSAC-LB",
           color="#3498db", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylabel("Proportion")

    for i, (rv, cv) in enumerate(zip(rbc_hc, csac_hc)):
        if rv > 0.001:
            ax.annotate(f"{rv:.3f}", xy=(i - width / 2, rv),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=8)
        if cv > 0.001:
            ax.annotate(f"{cv:.3f}", xy=(i + width / 2, cv),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=8)

fig.suptitle("Hot vs Cold Discomfort Breakdown", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT_DIR, "fig_stress_test_hot_cold_breakdown.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_stress_test_hot_cold_breakdown.png"))
plt.close(fig)
print("[4/4] Hot/cold discomfort breakdown saved.")

print(f"\nAll stress test figures saved to {OUT_DIR}/")
