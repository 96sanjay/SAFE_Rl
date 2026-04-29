"""
Publication plots matching Prof. Kassler's email format:
  1. Reward over training steps (all baselines overlaid, shaded bands)
  2. Violation % over training steps (EpCost as % of episode length)
  3. C1 Ablation with shaded bands (same style)
"""
import os, glob, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTDIR = "runs/plots"
os.makedirs(OUTDIR, exist_ok=True)
EP_LEN = 8759

def smooth_with_bands(y, window=7):
    y = np.array(y, dtype=float)
    n = len(y)
    if n < window:
        return y, y, y
    center = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo_idx = max(0, i - half)
        hi_idx = min(n, i + half + 1)
        chunk = y[lo_idx:hi_idx]
        center[i] = np.median(chunk)
        upper[i] = np.percentile(chunk, 75)
        lower[i] = np.percentile(chunk, 25)
    return lower, center, upper

def millions_fmt(x, pos):
    if x >= 1e6: return f'{x/1e6:.1f}M'
    if x >= 1e3: return f'{x/1e3:.0f}k'
    return f'{x:.0f}'

def find_best(pattern):
    best, best_n = None, 0
    for f in glob.glob(pattern):
        try:
            n = sum(1 for _ in open(f))
            if n > best_n: best, best_n = f, n
        except: pass
    return best

def load_csv(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))

BASELINES = {
    "TRPOLag": find_best("runs/trpolag_v2g_stems/TRPOLag-{CityLearnSafety-SoC-v0}/seed-*/progress.csv"),
    "CPO":     find_best("runs/cpo_v2g_stems/CPO-{CityLearnSafety-SoC-v0}/seed-*/progress.csv"),
    "PPOLag":  find_best("runs/ppolag_v2g_stems/PPOLag-{CityLearnSafety-SoC-v0}/seed-*/progress.csv"),
}

C1_RUNS = {
    "A: Pure Lagrangian": find_best("runs/c1_A_pure/TRPOLag-{CityLearnSafety-SoC-v0}/seed-*/progress.csv"),
    "B: + EV Shaping":    find_best("runs/c1_B_shaped/TRPOLag-{CityLearnSafety-SoC-v0}/seed-*/progress.csv"),
    "C: + Forecast Obs":  find_best("runs/c1_C_forecast/TRPOLag-{CityLearnSafety-Forecast-v0}/seed-*/progress.csv"),
}

B_COLORS = {"TRPOLag": "#e74c3c", "CPO": "#2980b9", "PPOLag": "#27ae60"}
C_COLORS = {"A: Pure Lagrangian": "#2980b9", "B: + EV Shaping": "#e67e22", "C: + Forecast Obs": "#27ae60"}

for k, v in BASELINES.items():
    n = len(load_csv(v) or [])
    print(f"  Baseline {k}: {n} epochs")
for k, v in C1_RUNS.items():
    n = len(load_csv(v) or [])
    print(f"  C1 {k}: {n} epochs")

# ═══════════════════════════════════════════════════════════
# PLOT 1: Baseline Reward + Violation % (side by side)
# ═══════════════════════════════════════════════════════════
print("\n[1] Baseline Reward + Violation %...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for name, path in BASELINES.items():
    rows = load_csv(path)
    if not rows or len(rows) < 3: continue
    steps = np.array([float(r.get("TotalEnvSteps", i*EP_LEN)) for i, r in enumerate(rows)])
    color = B_COLORS[name]

    ret = np.array([float(r["Metrics/EpRet"]) for r in rows])
    lo, mid, hi = smooth_with_bands(ret, window=7)
    ax1.plot(steps, mid, label=name, color=color, lw=2.5)
    ax1.fill_between(steps, lo, hi, alpha=0.2, color=color)

    cost = np.array([float(r["Metrics/EpCost"]) for r in rows])
    viol_pct = cost / EP_LEN * 100.0
    lo, mid, hi = smooth_with_bands(viol_pct, window=7)
    ax2.plot(steps, mid, label=name, color=color, lw=2.5)
    ax2.fill_between(steps, lo, hi, alpha=0.2, color=color)

ax1.set_xlabel("Steps (millions)")
ax1.set_ylabel("Episode Reward")
ax1.set_title("Rewards")
ax1.legend(loc="best", framealpha=0.9)
ax1.xaxis.set_major_formatter(FuncFormatter(millions_fmt))

ax2.set_xlabel("Steps (millions)")
ax2.set_ylabel("Violation Rate (%)")
ax2.set_title("Violation Percentage")
ax2.legend(loc="best", framealpha=0.9)
ax2.xaxis.set_major_formatter(FuncFormatter(millions_fmt))
ax2.set_ylim(bottom=0)

plt.tight_layout()
fig.savefig(f"{OUTDIR}/1_baseline_reward_violation.png")
fig.savefig(f"{OUTDIR}/1_baseline_reward_violation.pdf")
print("  -> 1_baseline_reward_violation")
plt.close()

# ═══════════════════════════════════════════════════════════
# PLOT 2: Baseline Lambda
# ═══════════════════════════════════════════════════════════
print("[2] Baseline Lambda...")
fig, ax = plt.subplots(figsize=(7, 5))

for name, path in BASELINES.items():
    rows = load_csv(path)
    if not rows or len(rows) < 3: continue
    steps = np.array([float(r.get("TotalEnvSteps", i*EP_LEN)) for i, r in enumerate(rows)])
    if "Metrics/LagrangeMultiplier" not in rows[0]: continue
    lam = np.array([float(r["Metrics/LagrangeMultiplier"]) for r in rows])
    ax.plot(steps, lam, label=name, color=B_COLORS[name], lw=2.5)

ax.set_xlabel("Steps (millions)")
ax.set_ylabel("Lambda")
ax.set_title("Lagrange Multiplier Dynamics")
ax.legend(loc="best", framealpha=0.9)
ax.xaxis.set_major_formatter(FuncFormatter(millions_fmt))
ax.annotate("lambda collapse -> discharge exploit",
            xy=(0.5, 0.15), xycoords='axes fraction',
            fontsize=12, ha='center', style='italic', color='#888')

fig.savefig(f"{OUTDIR}/2_baseline_lambda.png")
fig.savefig(f"{OUTDIR}/2_baseline_lambda.pdf")
print("  -> 2_baseline_lambda")
plt.close()

# ═══════════════════════════════════════════════════════════
# PLOT 3: C1 Ablation Reward + Violation %
# ═══════════════════════════════════════════════════════════
print("[3] C1 Ablation Reward + Violation %...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for name, path in C1_RUNS.items():
    rows = load_csv(path)
    if not rows or len(rows) < 3:
        print(f"  SKIP {name} ({len(rows) if rows else 0} epochs)")
        continue
    steps = np.array([float(r.get("TotalEnvSteps", i*EP_LEN)) for i, r in enumerate(rows)])
    color = C_COLORS[name]

    ret = np.array([float(r["Metrics/EpRet"]) for r in rows])
    lo, mid, hi = smooth_with_bands(ret, window=7)
    ax1.plot(steps, mid, label=name, color=color, lw=2.5)
    ax1.fill_between(steps, lo, hi, alpha=0.2, color=color)

    cost = np.array([float(r["Metrics/EpCost"]) for r in rows])
    viol_pct = cost / EP_LEN * 100.0
    lo, mid, hi = smooth_with_bands(viol_pct, window=7)
    ax2.plot(steps, mid, label=name, color=color, lw=2.5)
    ax2.fill_between(steps, lo, hi, alpha=0.2, color=color)

ax1.axhline(y=-3862.6, color='gray', ls='--', lw=1.5, alpha=0.7, label='Zero-Action')
ax2.axhline(y=1000/EP_LEN*100, color='red', ls=':', lw=1.5, alpha=0.7, label='Cost Limit')

ax1.set_xlabel("Steps (millions)")
ax1.set_ylabel("Episode Reward")
ax1.set_title("C1-Only Ablation: Rewards")
ax1.legend(loc="best", fontsize=10, framealpha=0.9)
ax1.xaxis.set_major_formatter(FuncFormatter(millions_fmt))

ax2.set_xlabel("Steps (millions)")
ax2.set_ylabel("C1 Violation Rate (%)")
ax2.set_title("C1-Only Ablation: EV Departure Violations")
ax2.legend(loc="best", fontsize=10, framealpha=0.9)
ax2.xaxis.set_major_formatter(FuncFormatter(millions_fmt))
ax2.set_ylim(bottom=0)

plt.tight_layout()
fig.savefig(f"{OUTDIR}/3_c1_ablation_reward_violation.png")
fig.savefig(f"{OUTDIR}/3_c1_ablation_reward_violation.pdf")
print("  -> 3_c1_ablation_reward_violation")
plt.close()

# ═══════════════════════════════════════════════════════════
# PLOT 4: C1 Ablation Lambda
# ═══════════════════════════════════════════════════════════
print("[4] C1 Ablation Lambda...")
fig, ax = plt.subplots(figsize=(7, 5))

for name, path in C1_RUNS.items():
    rows = load_csv(path)
    if not rows or len(rows) < 3: continue
    steps = np.array([float(r.get("TotalEnvSteps", i*EP_LEN)) for i, r in enumerate(rows)])
    if "Metrics/LagrangeMultiplier" not in rows[0]: continue
    lam = np.array([float(r["Metrics/LagrangeMultiplier"]) for r in rows])
    ax.plot(steps, lam, label=name, color=C_COLORS[name], lw=2.5)

ax.set_xlabel("Steps (millions)")
ax.set_ylabel("Lambda")
ax.set_title("C1-Only Ablation: Lagrange Multiplier")
ax.legend(loc="best", framealpha=0.9)
ax.xaxis.set_major_formatter(FuncFormatter(millions_fmt))

fig.savefig(f"{OUTDIR}/4_c1_ablation_lambda.png")
fig.savefig(f"{OUTDIR}/4_c1_ablation_lambda.pdf")
print("  -> 4_c1_ablation_lambda")
plt.close()

# ═══════════════════════════════════════════════════════════
# PLOT 5: Eval comparison
# ═══════════════════════════════════════════════════════════
print("[5] Evaluation comparison...")
eval_file = "runs/plots/eval_results.json"
if os.path.exists(eval_file):
    eval_data = json.load(open(eval_file))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    methods = [r["method"] for r in eval_data]
    rewards = [r["reward"] for r in eval_data]
    costs   = [r["cost"]   for r in eval_data]
    colors = ["#95a5a6", "#f39c12", "#e74c3c", "#2980b9", "#27ae60"]
    y_pos = np.arange(len(methods))

    bars1 = ax1.barh(y_pos, rewards, color=colors, edgecolor='k', lw=0.5, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontsize=11)
    ax1.set_xlabel("Episode Reward")
    ax1.set_title("Evaluation: Episode Reward")
    ax1.axvline(x=0, color='k', lw=0.5)
    for bar, val in zip(bars1, rewards):
        xpos = val + 200 if val > 0 else val - 200
        ha = 'left' if val > 0 else 'right'
        ax1.text(xpos, bar.get_y() + bar.get_height()/2,
                 f'{val:.0f}', va='center', ha=ha, fontsize=9, fontweight='bold')

    bars2 = ax2.barh(y_pos, costs, color=colors, edgecolor='k', lw=0.5, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods, fontsize=11)
    ax2.set_xlabel("Total Episode Cost")
    ax2.set_title("Evaluation: Total Constraint Cost")
    for bar, val in zip(bars2, costs):
        ax2.text(val + 300, bar.get_y() + bar.get_height()/2,
                 f'{val:.0f}', va='center', ha='left', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/5_eval_comparison.png")
    fig.savefig(f"{OUTDIR}/5_eval_comparison.pdf")
    print("  -> 5_eval_comparison")
    plt.close()

print(f"\nDone! All plots in {OUTDIR}/")
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith('.png'):
        print(f"  {f}")
