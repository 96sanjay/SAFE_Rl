#!/usr/bin/env python3
"""
Publication-quality multi-seed robustness figure (2-panel).
Left: violation rate box plots with individual seeds.
Right: total episode reward with error bars.

Replaces both fig_multiseed_summary_two_panel and fig_multiseed_robustness.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Publication style (IEEE / NeurIPS compatible)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "lines.linewidth": 0.8,
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent.parent.parent
SUMMARY = BASE / "runs" / "multiseed_eval_summary.json"
THESIS_FIG = BASE.parent / "studienarbeiten-master" / "thesis" / "figures"
LOCAL_FIG = Path(__file__).resolve().parent.parent / "figures"

with open(SUMMARY) as f:
    data = json.load(f)

# Ordered by performance (best safety first)
ALGO_ORDER = ["CSAC-LB", "SAC-Lag", "CPO", "CUP", "FOCOPS"]
ALGO_LABELS = {
    "CSAC-LB": "CSAC-LB",
    "SAC-Lag": "SAC-Lag",
    "CPO": "CPO",
    "CUP": "CUP",
    "FOCOPS": "FOCOPS",
}

# Consistent color palette (colorblind-friendly)
COLORS = {
    "CSAC-LB": "#D62728",  # red
    "CPO":     "#1F77B4",  # blue
    "SAC-Lag": "#9467BD",  # purple
    "CUP":     "#17BECF",  # cyan
    "FOCOPS":  "#FF7F0E",  # orange
}

RBC_VIOL = 51.6  # Cooling RBC baseline (%)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax_viol, ax_rew) = plt.subplots(
    1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.2, 1]}
)

# ── Panel (a): Violation rate box plots ──────────────────────────────────────

for i, algo in enumerate(ALGO_ORDER):
    ad = data[algo]
    vrates = np.array([s["violation_rate"] * 100 for s in ad["per_seed"]])
    n = ad["n_seeds"]
    col = COLORS[algo]

    # Box plot
    bp = ax_viol.boxplot(
        [vrates], positions=[i], widths=0.45, vert=True,
        patch_artist=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=3.5,
                       markeredgewidth=0.5),
        medianprops=dict(color="black", linewidth=0.8),
        boxprops=dict(facecolor=col, alpha=0.25,
                      edgecolor=col, linewidth=0.7),
        whiskerprops=dict(color=col, linewidth=0.7),
        capprops=dict(color=col, linewidth=0.7),
        flierprops=dict(marker="o", markerfacecolor=col,
                        markeredgecolor="none", markersize=3, alpha=0.7),
    )

    # Individual seed dots (jittered)
    rng = np.random.default_rng(42 + i)
    jitter = rng.uniform(-0.10, 0.10, len(vrates))
    ax_viol.scatter(
        i + jitter, vrates,
        color=col, s=14, zorder=5, edgecolors="white",
        linewidths=0.3, alpha=0.85
    )

    # Mean annotation
    mean_v = np.mean(vrates)
    ax_viol.annotate(
        f"{mean_v:.1f}%",
        xy=(i, mean_v), xytext=(18, -2),
        textcoords="offset points", fontsize=6,
        color=col, fontweight="bold", ha="left", va="center",
    )

# RBC baseline
ax_viol.axhline(RBC_VIOL, color="#666666", ls="--", lw=0.7, zorder=1)
ax_viol.text(
    len(ALGO_ORDER) - 0.5, RBC_VIOL + 2,
    f"Cooling RBC ({RBC_VIOL:.1f}%)",
    fontsize=6, color="#666666", ha="right", va="bottom",
    fontstyle="italic",
)

ax_viol.set_xticks(range(len(ALGO_ORDER)))
ax_viol.set_xticklabels(
    [f"{ALGO_LABELS[a]}\n($n$={data[a]['n_seeds']})" for a in ALGO_ORDER],
    fontsize=6.5,
)
ax_viol.set_ylabel("Comfort Violation Rate (%)")
ax_viol.set_title("(a) Safety: Violation Rate", fontweight="bold")
ax_viol.set_ylim(-3, 105)
ax_viol.yaxis.set_major_locator(mticker.MultipleLocator(20))
ax_viol.grid(axis="y")

# ── Panel (b): Reward bar chart ──────────────────────────────────────────────

means = []
stds = []
for algo in ALGO_ORDER:
    ad = data[algo]
    rewards = [s.get("total_reward", s.get("reward", 0))
               for s in ad["per_seed"]]
    means.append(ad.get("reward_mean", np.mean(rewards)))
    stds.append(ad.get("reward_std", np.std(rewards)))

bars = ax_rew.barh(
    range(len(ALGO_ORDER)), means,
    xerr=stds, height=0.55,
    color=[COLORS[a] for a in ALGO_ORDER],
    edgecolor="white", linewidth=0.3,
    error_kw=dict(lw=0.7, capsize=2.5, capthick=0.6, color="black"),
    alpha=0.85,
)

# Value labels
for i, (m, s) in enumerate(zip(means, stds)):
    ax_rew.text(
        m + s + 30, i, f"{m:.0f}",
        va="center", ha="left", fontsize=6, fontweight="bold",
        color=COLORS[ALGO_ORDER[i]],
    )

ax_rew.set_yticks(range(len(ALGO_ORDER)))
ax_rew.set_yticklabels(
    [ALGO_LABELS[a] for a in ALGO_ORDER], fontsize=6.5,
)
ax_rew.set_xlabel("Total Episode Reward")
ax_rew.set_title("(b) Efficiency: Episode Reward", fontweight="bold")
ax_rew.set_xlim(0, max(m + s for m, s in zip(means, stds)) * 1.15)
ax_rew.grid(axis="x")
ax_rew.invert_yaxis()  # Match order with left panel

# ── Finalize ─────────────────────────────────────────────────────────────────

fig.tight_layout(w_pad=2.0)

for out_dir in [LOCAL_FIG, THESIS_FIG]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_multiseed_robustness.pdf")
    fig.savefig(out_dir / "fig_multiseed_robustness.png")
    print(f"Saved to {out_dir / 'fig_multiseed_robustness.pdf'}")

plt.close(fig)
