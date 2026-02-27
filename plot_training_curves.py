"""
plot_training_curves.py — Publication-quality training curves
Generates 3 plots:
  1. Episode Reward vs Training Steps (all baselines + C1 ablation)
  2. Episode Cost vs Training Steps
  3. Lagrange Multiplier vs Training Steps
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUTDIR = "runs/plots"
os.makedirs(OUTDIR, exist_ok=True)

# === Discover CSVs ===
def find_csv(pattern):
    m = sorted(glob.glob(pattern))
    return m[-1] if m else None

BASELINE_RUNS = {
    "TRPOLag": "runs/trpolag_v2g_stems/TRPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-02-23-16-07-24/progress.csv",
    "CPO":     "runs/cpo_v2g_stems/CPO-{CityLearnSafety-SoC-v0}/seed-000-2026-02-23-16-08-15/progress.csv",
    "PPOLag":  "runs/ppolag_v2g_stems/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-02-23-19-08-07/progress.csv",
}

C1_RUNS = {
    "C1: Pure Lagrangian":  find_csv("runs/c1_A_pure/TRPOLag-{CityLearnSafety-SoC-v0}/seed-*/progress.csv"),
    "C1: + EV Shaping":    find_csv("runs/c1_B_shaped/TRPOLag-{CityLearnSafety-SoC-v0}/seed-*/progress.csv"),
    "C1: + Forecast Obs":  find_csv("runs/c1_C_forecast/TRPOLag-{CityLearnSafety-Forecast-v0}/seed-*/progress.csv"),
}

# Also check old slow-lambda runs
OLD_RUNS = {
    "C1-old: Pure (slow λ)":  find_csv("runs/trpolag_c1_pure/TRPOLag-{CityLearnSafety-SoC-v0}/seed-*/progress.csv"),
    "C1-old: Shaped (slow λ)": find_csv("runs/trpolag_c1_shaped/TRPOLag-{CityLearnSafety-SoC-v0}/seed-*/progress.csv"),
}

def load_progress(path):
    if path is None or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"  Error reading {path}: {e}")
        return None

def millions_formatter(x, pos):
    return f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}k'

# === COLORS ===
COLORS_BASELINE = {
    "TRPOLag": "#e74c3c",
    "CPO": "#2ecc71",
    "PPOLag": "#3498db",
}
COLORS_C1 = {
    "C1: Pure Lagrangian": "#e74c3c",
    "C1: + EV Shaping": "#2ecc71",
    "C1: + Forecast Obs": "#9b59b6",
}

def smooth(y, weight=0.85):
    """Exponential moving average."""
    s = np.zeros_like(y, dtype=float)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = weight * s[i-1] + (1 - weight) * y[i]
    return s

# ═══════════════════════════════════════════════════════════════
# PLOT 1: Baseline Training Curves (Reward + Cost, 2 subplots)
# ═══════════════════════════════════════════════════════════════
print("\n=== Plot 1: Baseline Training Curves ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name, path in BASELINE_RUNS.items():
    print(f"  {name}: {path}")
    df = load_progress(path)
    if df is None:
        print(f"    SKIP — not found")
        continue
    
    steps = df["TotalEnvSteps"].values if "TotalEnvSteps" in df.columns else np.arange(len(df)) * 8759
    
    # Reward
    if "Metrics/EpRet" in df.columns:
        y = df["Metrics/EpRet"].values.astype(float)
        ax1.plot(steps, smooth(y), label=name, color=COLORS_BASELINE[name], linewidth=2)
        ax1.fill_between(steps, smooth(y, 0.95), smooth(y, 0.5), alpha=0.15, color=COLORS_BASELINE[name])
    
    # Cost
    if "Metrics/EpCost" in df.columns:
        y = df["Metrics/EpCost"].values.astype(float)
        ax2.plot(steps, smooth(y), label=name, color=COLORS_BASELINE[name], linewidth=2)
        ax2.fill_between(steps, smooth(y, 0.95), smooth(y, 0.5), alpha=0.15, color=COLORS_BASELINE[name])

ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Episode Reward")
ax1.set_title("Baseline Training Reward (4-Constraint)")
ax1.legend(loc="lower right")
ax1.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

ax2.set_xlabel("Training Steps")
ax2.set_ylabel("Episode Cost")
ax2.set_title("Baseline Training Cost (4-Constraint)")
ax2.axhline(y=20000, color='black', linestyle='--', linewidth=1.5, label='Cost Limit (20k)')
ax2.legend(loc="upper right")
ax2.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "baseline_training_curves.png"))
fig.savefig(os.path.join(OUTDIR, "baseline_training_curves.pdf"))
print(f"  Saved: {OUTDIR}/baseline_training_curves.png/.pdf")
plt.close()

# ═══════════════════════════════════════════════════════════════
# PLOT 2: Baseline Lambda Dynamics
# ═══════════════════════════════════════════════════════════════
print("\n=== Plot 2: Baseline Lambda Dynamics ===")
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for name, path in BASELINE_RUNS.items():
    df = load_progress(path)
    if df is None:
        continue
    steps = df["TotalEnvSteps"].values if "TotalEnvSteps" in df.columns else np.arange(len(df)) * 8759
    
    if "Metrics/LagrangeMultiplier" in df.columns:
        y = df["Metrics/LagrangeMultiplier"].values.astype(float)
        ax.plot(steps, y, label=name, color=COLORS_BASELINE[name], linewidth=2)

ax.set_xlabel("Training Steps")
ax.set_ylabel("Lagrange Multiplier (λ)")
ax.set_title("Lagrange Multiplier Dynamics — Baseline (4-Constraint)")
ax.legend()
ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

# Annotate lambda collapse
ax.annotate("λ collapse\n(discharge exploit)", xy=(0.3, 0.5), xycoords='axes fraction',
            fontsize=11, ha='center', style='italic', color='gray')

fig.savefig(os.path.join(OUTDIR, "baseline_lambda_dynamics.png"))
fig.savefig(os.path.join(OUTDIR, "baseline_lambda_dynamics.pdf"))
print(f"  Saved: {OUTDIR}/baseline_lambda_dynamics.png/.pdf")
plt.close()

# ═══════════════════════════════════════════════════════════════
# PLOT 3: C1 Ablation Training Curves (Reward + Cost + Lambda)
# ═══════════════════════════════════════════════════════════════
print("\n=== Plot 3: C1 Ablation Training Curves ===")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax_r, ax_c, ax_l = axes

for name, path in C1_RUNS.items():
    print(f"  {name}: {path}")
    df = load_progress(path)
    if df is None:
        print(f"    SKIP — not found")
        continue
    
    steps = df["TotalEnvSteps"].values if "TotalEnvSteps" in df.columns else np.arange(len(df)) * 8759
    color = COLORS_C1[name]
    
    if "Metrics/EpRet" in df.columns:
        y = df["Metrics/EpRet"].values.astype(float)
        ax_r.plot(steps, smooth(y), label=name, color=color, linewidth=2)
        ax_r.fill_between(steps, smooth(y, 0.95), smooth(y, 0.5), alpha=0.15, color=color)
    
    if "Metrics/EpCost" in df.columns:
        y = df["Metrics/EpCost"].values.astype(float)
        ax_c.plot(steps, smooth(y), label=name, color=color, linewidth=2)
        ax_c.fill_between(steps, smooth(y, 0.95), smooth(y, 0.5), alpha=0.15, color=color)
    
    if "Metrics/LagrangeMultiplier" in df.columns:
        y = df["Metrics/LagrangeMultiplier"].values.astype(float)
        ax_l.plot(steps, y, label=name, color=color, linewidth=2)

ax_r.set_xlabel("Training Steps")
ax_r.set_ylabel("Episode Reward")
ax_r.set_title("C1-Only: Episode Reward")
ax_r.legend(loc="best", fontsize=9)
ax_r.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

ax_c.set_xlabel("Training Steps")
ax_c.set_ylabel("Episode Cost (C1)")
ax_c.set_title("C1-Only: EV Departure Cost")
ax_c.axhline(y=1000, color='black', linestyle='--', linewidth=1.5, label='Cost Limit (1k)')
ax_c.axhline(y=2320, color='gray', linestyle=':', linewidth=1, label='Zero-Action Cost')
ax_c.legend(loc="upper right", fontsize=9)
ax_c.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

ax_l.set_xlabel("Training Steps")
ax_l.set_ylabel("Lagrange Multiplier (λ)")
ax_l.set_title("C1-Only: Lambda Dynamics")
ax_l.legend(loc="best", fontsize=9)
ax_l.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "c1_ablation_training_curves.png"))
fig.savefig(os.path.join(OUTDIR, "c1_ablation_training_curves.pdf"))
print(f"  Saved: {OUTDIR}/c1_ablation_training_curves.png/.pdf")
plt.close()

# ═══════════════════════════════════════════════════════════════
# PLOT 4: Baseline Evaluation Table (bar chart)
# ═══════════════════════════════════════════════════════════════
print("\n=== Plot 4: Baseline Evaluation Summary ===")

eval_data = {
    "Method":  ["RBC",     "TRPOLag",  "CPO",      "PPOLag"],
    "Reward":  [-13505.6,  14004.3,    15326.7,    8114.2],
    "C1 (%)":  [5.66,      99.32,      100.0,      97.95],
    "C2 (%)":  [99.98,     0.0,        5.88,       7.76],
    "C3 (%)":  [26.64,     20.92,      19.57,      29.04],
    "C4 (%)":  [35.56,     2.91,       0.83,       8.81],
}
df_eval = pd.DataFrame(eval_data)

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
constraints = ["C1 (%)", "C2 (%)", "C3 (%)", "C4 (%)"]
constraint_names = [
    "C1: EV Departure\nViolation %",
    "C2: Battery SoC\nViolation %",
    "C3: Building Power\nViolation %",
    "C4: Grid Power\nViolation %",
]
bar_colors = ["#95a5a6", "#e74c3c", "#2ecc71", "#3498db"]

for i, (col, title) in enumerate(zip(constraints, constraint_names)):
    ax = axes[i]
    bars = ax.bar(df_eval["Method"], df_eval[col], color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Violation %" if i == 0 else "")
    # Add value labels on bars
    for bar, val in zip(bars, df_eval[col]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add structural floor for C3
    if col == "C3 (%)":
        ax.axhline(y=7.35, color='red', linestyle='--', linewidth=1, label='Structural floor (7.35%)')
        ax.legend(fontsize=8)

plt.suptitle("Baseline Evaluation — Epoch 100 (4-Constraint, STEMS Reward)", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "baseline_evaluation_bars.png"))
fig.savefig(os.path.join(OUTDIR, "baseline_evaluation_bars.pdf"))
print(f"  Saved: {OUTDIR}/baseline_evaluation_bars.png/.pdf")
plt.close()

# ═══════════════════════════════════════════════════════════════
# PLOT 5: Zero-Action Comparison (key thesis finding)
# ═══════════════════════════════════════════════════════════════
print("\n=== Plot 5: Zero-Action vs TRPOLag Comparison ===")
fig, ax = plt.subplots(figsize=(8, 5))

compare = {
    "Metric": ["EpCost\n(total)", "C1 Cost\n(EV)", "C4 Cost\n(Grid)", "C3 Cost\n(Building)"],
    "Zero-Action": [3410.8, 2320.1, 1023.0, 67.5],
    "TRPOLag (100ep)": [10569, 9930, 145, 418],
}

x = np.arange(len(compare["Metric"]))
width = 0.35
bars1 = ax.bar(x - width/2, compare["Zero-Action"], width, label='Zero-Action (do nothing)',
               color='#3498db', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, compare["TRPOLag (100ep)"], width, label='TRPOLag (100 epochs)',
               color='#e74c3c', edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(compare["Metric"])
ax.set_ylabel("Episode Cost (weighted)")
ax.set_title("TRPOLag Learned Policy vs Doing Nothing\n(Lower is Better)")
ax.legend()

# Value labels
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 50,
                f'{h:.0f}', ha='center', va='bottom', fontsize=9)

fig.savefig(os.path.join(OUTDIR, "zero_action_comparison.png"))
fig.savefig(os.path.join(OUTDIR, "zero_action_comparison.pdf"))
print(f"  Saved: {OUTDIR}/zero_action_comparison.png/.pdf")
plt.close()

print(f"\n{'='*60}")
print(f"All plots saved in: {OUTDIR}/")
print(f"{'='*60}")
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith(('.png', '.pdf')):
        print(f"  {f}")
