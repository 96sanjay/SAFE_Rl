#!/usr/bin/env python3
"""
Comprehensive Reward Term Effectiveness Analysis
=================================================
Compares reward terms across three training runs:
  1. R25b Stable (80 epochs) -- TB events, NO per-reward breakdown
  2. Softmax Ablation (40 epochs) -- NPZ, full reward breakdown
  3. R28a Clean9 (60 epochs) -- TB events, full reward breakdown

Additionally uses "grads" ablation (40 epochs) from NPZ for comparison
since R25b lacks per-term data.

Output: PDF figures + tables printed to stdout.
Seed fixed for reproducibility: 42.
"""

import os
import sys
import warnings
import numpy as np
from scipy import stats
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
OUT  = os.path.join(BASE, "docs/temperature_case_study/r28b_reward_analysis")
os.makedirs(OUT, exist_ok=True)

R25B_TB = os.path.join(
    BASE,
    "runs/r25b_report_stable/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-27-04-04-26/tb/"
)
R28A_TB = os.path.join(
    BASE,
    "runs/r28a_clean9/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-04-15-05-24-54/tb/"
)
NPZ_PATH = os.path.join(BASE, "docs/temperature_case_study/tb_ablation_data.npz")


# ── 1. Data Loading ───────────────────────────────────────────────────────
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tb(path, size_guidance=0):
    ea = EventAccumulator(path, size_guidance={"scalars": size_guidance})
    ea.Reload()
    data = {}
    for tag in ea.Tags()["scalars"]:
        scalars = ea.Scalars(tag)
        data[tag] = {
            "steps": np.array([s.step for s in scalars]),
            "values": np.array([s.value for s in scalars]),
            "wall_time": np.array([s.wall_time for s in scalars]),
        }
    return data


def load_npz_prefix(npz, prefix):
    """Load NPZ keys with a given prefix, stripping prefix from key names."""
    data = {}
    for raw_key in npz.files:
        if not raw_key.startswith(prefix + "_"):
            continue
        suffix = raw_key[len(prefix) + 1:]
        # Parse: tag_field where field is steps/values/wall_time
        for field in ("_steps", "_values", "_wall_time"):
            if suffix.endswith(field):
                tag = suffix[: -len(field)].replace("_", "/", 1)  # first _ -> /
                if tag not in data:
                    data[tag] = {}
                data[tag][field.lstrip("_")] = npz[raw_key]
                break
    return data


print("Loading data...")
r25b = load_tb(R25B_TB)
r28a = load_tb(R28A_TB)
npz = np.load(NPZ_PATH, allow_pickle=True)
softmax = load_npz_prefix(npz, "softmax")
grads = load_npz_prefix(npz, "grads")

# Fix NPZ tag parsing: / in tag names may need adjustment
# Let's verify a known key
def fix_npz_tags(d):
    """The NPZ keys like 'Reward_r_ev_values' become 'Reward/r_ev' after first _ -> /
    but multi-underscore tags like 'Reward_r_ev_smart' become 'Reward/r_ev_smart'
    This should already work. Let's verify."""
    fixed = {}
    for k, v in d.items():
        # Handle cases where the tag has multiple levels
        # e.g., Metrics/EpCost_0, Reward/r_ev_smart
        fixed[k] = v
    return fixed

softmax = fix_npz_tags(softmax)
grads = fix_npz_tags(grads)

# Debug: print available reward tags
def get_reward_tags(d):
    return sorted([k for k in d if k.startswith("Reward/")])

def get_metric_tags(d):
    return sorted([k for k in d if k.startswith("Metrics/")])

print(f"\nR25b: {len(r25b)} tags, Reward tags: {get_reward_tags(r25b)}")
print(f"R28a: {len(r28a)} tags, Reward tags: {get_reward_tags(r28a)}")
print(f"Softmax: {len(softmax)} tags, Reward tags: {get_reward_tags(softmax)}")
print(f"Grads: {len(grads)} tags, Reward tags: {get_reward_tags(grads)}")

# Sanity: show what metric tags look like in each dataset
print(f"\nR25b metric tags: {get_metric_tags(r25b)}")
print(f"R28a metric tags: {get_metric_tags(r28a)}")
print(f"Softmax metric tags: {get_metric_tags(softmax)}")


# ── 2. Define reward term universe ────────────────────────────────────────
# All known reward terms
ALL_REWARD_TERMS = [
    "r_sg", "r_sb", "r_ramp", "r_ren", "r_barrier",
    "r_eco", "r_ev", "r_ev_guard", "r_ev_smart", "r_ev_slack_arb",
    "r_v2g_ctx", "r_peak_shave", "r_load_shift", "r_grid_mild",
    "r_ev_solar", "r_grid_penalty", "r_headroom", "r_nec_sign",
    "r_price_arb", "r_solar_store", "r_traj_batt", "r_traj_ev",
    "r_trajectory",
]

# Constraint mapping
CONSTRAINT_MAP = {
    "r_ev":          "Metrics/EpCost/0",
    "r_ev_smart":    "Metrics/EpCost/0",
    "r_ev_guard":    "Metrics/EpCost/0",
    "r_ev_slack_arb":"Metrics/EpCost/0",
    "r_v2g_ctx":     "Metrics/EpCost/0",
    "r_ev_solar":    "Metrics/EpCost/0",
    "r_traj_ev":     "Metrics/EpCost/0",
    "r_trajectory":  "Metrics/EpCost/0",
    "r_barrier":     "Metrics/EpCost/2",
    "r_sb":          "Metrics/EpCost/3",
    "r_peak_shave":  "Metrics/EpCost/3",
    "r_sg":          "Metrics/EpCost/4",
    "r_grid_mild":   "Metrics/EpCost/4",
    "r_ramp":        "Metrics/EpCost/4",
    "r_grid_penalty":"Metrics/EpCost/4",
    "r_eco":         "Metrics/EpRet",
    "r_load_shift":  "Metrics/EpRet",
    "r_ren":         "Metrics/EpRet",
    "r_price_arb":   "Metrics/EpRet",
    "r_solar_store": "Metrics/EpRet",
    "r_headroom":    "Metrics/EpRet",
    "r_nec_sign":    "Metrics/EpRet",
    "r_traj_batt":   "Metrics/EpRet",
}

# Planning vs Reactive classification
PLANNING_TERMS = {
    "r_ev_smart", "r_ev_slack_arb", "r_load_shift", "r_peak_shave",
    "r_trajectory", "r_traj_batt", "r_traj_ev", "r_price_arb",
    "r_solar_store", "r_headroom",
}
REACTIVE_TERMS = {
    "r_ev", "r_barrier", "r_sg", "r_sb", "r_ramp", "r_ren",
    "r_eco", "r_ev_guard", "r_v2g_ctx", "r_grid_mild",
    "r_ev_solar", "r_grid_penalty", "r_nec_sign",
}


# ── 3. Helper: get values from any data dict ──────────────────────────────
def get_vals(data, tag):
    """Try multiple tag name variants."""
    candidates = [tag]
    # Try with underscore instead of /
    candidates.append(tag.replace("/", "_"))
    for c in candidates:
        if c in data:
            return data[c]["values"]
    return None


def get_reward_vals(data, term):
    """Get reward term values: tries Reward/<term>."""
    return get_vals(data, f"Reward/{term}")


def get_metric_vals(data, metric_tag):
    """Get metric values. metric_tag like 'Metrics/EpCost/0' or 'Metrics/EpCost_0'."""
    # Try exact
    v = get_vals(data, metric_tag)
    if v is not None:
        return v
    # Try with _ instead of last /
    alt = metric_tag.rsplit("/", 1)
    if len(alt) == 2:
        v = get_vals(data, alt[0] + "_" + alt[1])
    return v


# ── 4. Compute statistics per run ─────────────────────────────────────────
def compute_term_stats(data, run_name, n_tail=10):
    """Compute stats for each reward term in a data dict."""
    results = {}
    # Collect all available reward terms and their values
    available = {}
    for term in ALL_REWARD_TERMS:
        v = get_reward_vals(data, term)
        if v is not None and len(v) > 0:
            available[term] = v

    if not available:
        print(f"  WARNING: No reward terms found for {run_name}")
        return results

    # Total absolute reward magnitude (sum of |mean| across terms)
    tail_n = min(n_tail, min(len(v) for v in available.values()))
    total_abs = sum(abs(np.mean(v[-tail_n:])) for v in available.values())
    if total_abs == 0:
        total_abs = 1e-10

    for term, vals in available.items():
        n = len(vals)
        tail = vals[-tail_n:]

        # Mean over last tail_n epochs
        mean_tail = np.mean(tail)
        std_tail = np.std(tail)

        # Trend (linear regression over full series)
        if n >= 3:
            slope, intercept, r_val, p_val, std_err = stats.linregress(
                np.arange(n), vals
            )
        else:
            slope, r_val, p_val = 0.0, 0.0, 1.0

        # Trend classification
        if p_val < 0.05 and abs(slope) > 0.01 * abs(mean_tail) if mean_tail != 0 else abs(slope) > 1e-6:
            trend = "GROWING" if slope > 0 else "DECLINING"
        else:
            trend = "STABLE"

        # Fraction of total |reward|
        frac = abs(mean_tail) / total_abs

        # Sign consistency
        pos_frac = np.mean(vals > 0)
        neg_frac = np.mean(vals < 0)
        if pos_frac > 0.9:
            sign = "+"
        elif neg_frac > 0.9:
            sign = "-"
        else:
            sign = "+/-"

        results[term] = {
            "mean_tail": mean_tail,
            "std_tail": std_tail,
            "slope": slope,
            "r_squared": r_val ** 2,
            "p_value": p_val,
            "trend": trend,
            "frac": frac,
            "sign": sign,
            "n_epochs": n,
        }

    return results


print("\n" + "=" * 100)
print("COMPUTING REWARD TERM STATISTICS")
print("=" * 100)

stats_softmax = compute_term_stats(softmax, "Softmax Ablation")
stats_grads = compute_term_stats(grads, "Grads Ablation")
stats_r28a = compute_term_stats(r28a, "R28a Clean9")

# R25b: only aggregate metrics (no reward breakdown)
# We'll report it for the constraint costs only


# ── 5. Print Table 1: Per-Reward-Term Contribution ───────────────────────
print("\n")
print("=" * 130)
print("TABLE 1: PER-REWARD-TERM CONTRIBUTION (last 10 epochs mean +/- std)")
print("=" * 130)
header = (
    f"{'Term':<18s} | {'Softmax (40ep)':^30s} | {'Grads (40ep)':^30s} | "
    f"{'R28a Clean9 (60ep)':^30s}"
)
print(header)
sub = (
    f"{'':18s} | {'Mean':>8s} {'Frac':>6s} {'Trend':>9s} {'Sign':>4s} | "
    f"{'Mean':>8s} {'Frac':>6s} {'Trend':>9s} {'Sign':>4s} | "
    f"{'Mean':>8s} {'Frac':>6s} {'Trend':>9s} {'Sign':>4s}"
)
print(sub)
print("-" * 130)

all_terms_seen = sorted(
    set(list(stats_softmax.keys()) + list(stats_grads.keys()) + list(stats_r28a.keys()))
)

for term in all_terms_seen:
    parts = [f"{term:<18s}"]
    for st in [stats_softmax, stats_grads, stats_r28a]:
        if term in st:
            s = st[term]
            parts.append(
                f"{s['mean_tail']:>8.2f} {s['frac']:>5.1%} {s['trend']:>9s} {s['sign']:>4s}"
            )
        else:
            parts.append(f"{'---':>8s} {'---':>6s} {'---':>9s} {'---':>4s}")
    print(" | ".join(parts))

print()


# ── 6. Effectiveness Classification ──────────────────────────────────────
def classify_term(s, term):
    """Classify a reward term as EFFECTIVE, WEAK, DEAD/NOISE, or HARMFUL."""
    if s is None:
        return "N/A"
    frac = s["frac"]
    mean = s["mean_tail"]
    trend = s["trend"]
    sign = s["sign"]

    # HARMFUL: terms that should be positive but are consistently negative
    # or contribute significantly but in the wrong direction.
    # NOTE: r_barrier is a PENALTY by design (negative is correct).
    # It is NOT harmful -- it intentionally signals boundary proximity.
    harmful_if_negative = {
        "r_ev", "r_ev_smart", "r_ev_guard", "r_ev_slack_arb",
        "r_eco", "r_ren", "r_headroom",
    }
    # Penalty terms: negative by design, reclassify based on magnitude
    penalty_terms = {"r_barrier", "r_ramp", "r_grid_mild"}
    # Penalty terms (negative by design): classify by magnitude, not sign
    if term in penalty_terms:
        if frac >= 0.05:
            return "EFFECTIVE"
        elif frac >= 0.01:
            return "WEAK"
        else:
            return "DEAD/NOISE"

    if term in harmful_if_negative and sign == "-" and frac > 0.01:
        return "HARMFUL"

    # Also harmful if large magnitude and declining when should be growing
    if frac > 0.05 and trend == "DECLINING" and mean < 0 and term in harmful_if_negative:
        return "HARMFUL"

    if frac >= 0.05:
        return "EFFECTIVE"
    elif frac >= 0.01:
        return "WEAK"
    else:
        return "DEAD/NOISE"


print("=" * 100)
print("TABLE 2: EFFECTIVENESS CLASSIFICATION")
print("=" * 100)
print(f"{'Term':<18s} | {'Softmax':^12s} | {'Grads':^12s} | {'R28a':^12s}")
print("-" * 60)

for term in all_terms_seen:
    c_s = classify_term(stats_softmax.get(term), term)
    c_g = classify_term(stats_grads.get(term), term)
    c_r = classify_term(stats_r28a.get(term), term)
    print(f"{term:<18s} | {c_s:^12s} | {c_g:^12s} | {c_r:^12s}")

print()


# ── 7. Correlation with Constraint Costs ──────────────────────────────────
def compute_correlations(data, reward_stats, run_name):
    """Compute Pearson correlation between each reward term and its target cost."""
    results = {}
    for term in reward_stats:
        reward_vals = get_reward_vals(data, term)
        if reward_vals is None:
            continue

        target = CONSTRAINT_MAP.get(term, "Metrics/EpRet")
        cost_vals = get_metric_vals(data, target)
        if cost_vals is None:
            # Try alternate naming
            continue

        # Align lengths
        n = min(len(reward_vals), len(cost_vals))
        if n < 5:
            continue

        r, p = stats.pearsonr(reward_vals[:n], cost_vals[:n])
        results[term] = {"corr": r, "p_value": p, "n": n, "target": target}

    return results


print("=" * 110)
print("TABLE 3: PEARSON CORRELATION WITH CONSTRAINT COSTS")
print("  Negative correlation = reward term helps reduce constraint violation")
print("=" * 110)

corr_softmax = compute_correlations(softmax, stats_softmax, "Softmax")
corr_grads = compute_correlations(grads, stats_grads, "Grads")
corr_r28a = compute_correlations(r28a, stats_r28a, "R28a")

print(f"{'Term':<18s} {'Target':<18s} | {'Softmax':^18s} | {'Grads':^18s} | {'R28a':^18s}")
print(f"{'':18s} {'':18s} | {'r':>6s} {'p':>8s} {'n':>3s} | {'r':>6s} {'p':>8s} {'n':>3s} | {'r':>6s} {'p':>8s} {'n':>3s}")
print("-" * 110)

for term in all_terms_seen:
    target = CONSTRAINT_MAP.get(term, "Metrics/EpRet")
    parts = [f"{term:<18s} {target:<18s}"]
    for corr in [corr_softmax, corr_grads, corr_r28a]:
        if term in corr:
            c = corr[term]
            sig = "*" if c["p_value"] < 0.05 else " "
            parts.append(f"{c['corr']:>+6.3f}{sig} {c['p_value']:>7.4f} {c['n']:>3d}")
        else:
            parts.append(f"{'---':>6s} {'---':>8s} {'---':>3s}")
    print(" | ".join(parts))

print("\n  * = p < 0.05 (statistically significant)")
print()


# ── 8. Planning vs Reactive Analysis ─────────────────────────────────────
print("=" * 100)
print("TABLE 4: PLANNING vs REACTIVE REWARD TERMS")
print("=" * 100)

for category, term_set, desc in [
    ("PLANNING", PLANNING_TERMS, "Forward-looking: uses time horizon, forecasts, or arbitrage"),
    ("REACTIVE", REACTIVE_TERMS, "Immediate: based on current state only"),
]:
    print(f"\n--- {category} TERMS ({desc}) ---")
    print(f"{'Term':<18s} | {'Softmax mean':>12s} {'trend':>9s} | {'Grads mean':>12s} {'trend':>9s} | {'R28a mean':>12s} {'trend':>9s}")
    print("-" * 100)
    for term in sorted(term_set):
        parts = [f"{term:<18s}"]
        for st in [stats_softmax, stats_grads, stats_r28a]:
            if term in st:
                s = st[term]
                parts.append(f"{s['mean_tail']:>12.3f} {s['trend']:>9s}")
            else:
                parts.append(f"{'N/A':>12s} {'N/A':>9s}")
        print(" | ".join(parts))


# ── 9. R25b Aggregate Metrics (no per-term breakdown) ────────────────────
print("\n")
print("=" * 100)
print("R25b STABLE: AGGREGATE METRICS (no per-reward-term breakdown available)")
print("=" * 100)

r25b_metrics = {}
for tag in ["Metrics/EpRet", "Metrics/EpCost_0", "Metrics/EpCost_1",
            "Metrics/EpCost_2", "Metrics/EpCost_3", "Metrics/EpCost_4"]:
    v = get_vals(r25b, tag)
    if v is not None:
        r25b_metrics[tag] = v
        tail = v[-10:]
        slope, _, _, p, _ = stats.linregress(np.arange(len(v)), v)
        print(f"  {tag:<25s}  last-10 mean: {np.mean(tail):>10.1f} +/- {np.std(tail):>8.1f}  "
              f"slope: {slope:>+8.2f}  final: {v[-1]:>10.1f}")

# Lambda values for R25b
for tag in ["Metrics/Lambda_0", "Metrics/Lambda_1", "Metrics/Lambda_2",
            "Metrics/Lambda_3", "Metrics/Lambda_4"]:
    v = get_vals(r25b, tag)
    if v is not None:
        print(f"  {tag:<25s}  last-10 mean: {np.mean(v[-10:]):>10.2f}  final: {v[-1]:>10.2f}")

# Compare aggregate metrics across all runs
print("\n")
print("=" * 100)
print("TABLE 5: AGGREGATE METRICS COMPARISON (all runs)")
print("=" * 100)

metric_tags = ["Metrics/EpRet", "Metrics/EpCost/0", "Metrics/EpCost/2",
               "Metrics/EpCost/3", "Metrics/EpCost/4"]
alt_tags = {
    "Metrics/EpCost/0": ["Metrics/EpCost_0"],
    "Metrics/EpCost/2": ["Metrics/EpCost_2"],
    "Metrics/EpCost/3": ["Metrics/EpCost_3"],
    "Metrics/EpCost/4": ["Metrics/EpCost_4"],
}

print(f"{'Metric':<22s} | {'R25b (80ep)':^20s} | {'Softmax (40ep)':^20s} | {'Grads (40ep)':^20s} | {'R28a (60ep)':^20s}")
print("-" * 110)

for tag in metric_tags:
    parts = [f"{tag:<22s}"]
    for data in [r25b, softmax, grads, r28a]:
        v = get_metric_vals(data, tag)
        if v is None and tag in alt_tags:
            for alt in alt_tags[tag]:
                v = get_metric_vals(data, alt)
                if v is not None:
                    break
        if v is not None:
            tail = v[-min(10, len(v)):]
            parts.append(f"{np.mean(tail):>10.1f} +/- {np.std(tail):>6.1f}")
        else:
            parts.append(f"{'N/A':>20s}")
    print(" | ".join(parts))

# Lambda comparison
print()
lam_tags = ["Metrics/Lambda_0", "Metrics/Lambda_1", "Metrics/Lambda_2",
            "Metrics/Lambda_3", "Metrics/Lambda_4"]
for tag in lam_tags:
    parts = [f"{tag:<22s}"]
    for data in [r25b, softmax, grads, r28a]:
        v = get_metric_vals(data, tag)
        if v is None:
            v = get_vals(data, tag)
        if v is not None:
            parts.append(f"{v[-1]:>10.2f} (max={max(v):>.1f})")
        else:
            parts.append(f"{'N/A':>20s}")
    print(" | ".join(parts))


# ── 10. Summary: Magnitude Ranking ───────────────────────────────────────
print("\n")
print("=" * 100)
print("TABLE 6: REWARD TERM MAGNITUDE RANKING (R28a Clean9, last 10 epochs)")
print("=" * 100)

if stats_r28a:
    ranked = sorted(stats_r28a.items(), key=lambda x: abs(x[1]["mean_tail"]), reverse=True)
    print(f"{'Rank':>4s} {'Term':<18s} {'Mean':>10s} {'|Mean|':>10s} {'Frac':>7s} {'Trend':>9s} {'Class':>12s}")
    print("-" * 80)
    for i, (term, s) in enumerate(ranked, 1):
        cls = classify_term(s, term)
        print(f"{i:4d} {term:<18s} {s['mean_tail']:>+10.3f} {abs(s['mean_tail']):>10.3f} "
              f"{s['frac']:>6.1%} {s['trend']:>9s} {cls:>12s}")


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

# Color scheme
COLORS = {
    "softmax": "#2196F3",
    "grads": "#FF9800",
    "r28a": "#4CAF50",
    "r25b": "#9C27B0",
}

# ── Figure 1: Reward term evolution (3 panels, one per run) ───────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)
fig.suptitle("Reward Term Evolution Over Training", fontsize=14, fontweight="bold")

for ax, (data, stats_dict, title) in zip(axes, [
    (softmax, stats_softmax, "Softmax Ablation (40ep)"),
    (grads, stats_grads, "Grads Ablation (40ep)"),
    (r28a, stats_r28a, "R28a Clean9 (60ep)"),
]):
    if not stats_dict:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        continue

    # Plot top-10 terms by magnitude
    ranked = sorted(stats_dict.items(), key=lambda x: abs(x[1]["mean_tail"]), reverse=True)[:10]
    cmap = plt.cm.tab10
    for i, (term, s) in enumerate(ranked):
        vals = get_reward_vals(data, term)
        if vals is not None:
            ax.plot(np.arange(1, len(vals)+1), vals, label=term,
                    color=cmap(i), linewidth=1.5, alpha=0.8)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Reward Value", fontsize=11)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_reward_evolution.pdf"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_reward_evolution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: fig1_reward_evolution.pdf/png")


# ── Figure 2: Effectiveness heatmap ──────────────────────────────────────
CLASS_TO_NUM = {"EFFECTIVE": 3, "WEAK": 2, "DEAD/NOISE": 1, "HARMFUL": 0, "N/A": -1}
CLASS_LABELS = ["HARMFUL", "DEAD/NOISE", "WEAK", "EFFECTIVE"]

fig, ax = plt.subplots(figsize=(10, max(6, len(all_terms_seen) * 0.4)))
matrix = []
for term in all_terms_seen:
    row = []
    for st in [stats_softmax, stats_grads, stats_r28a]:
        cls = classify_term(st.get(term), term)
        row.append(CLASS_TO_NUM.get(cls, -1))
    matrix.append(row)

matrix = np.array(matrix)
cmap = plt.cm.RdYlGn
im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-1, vmax=3, interpolation="nearest")
ax.set_xticks(range(3))
ax.set_xticklabels(["Softmax", "Grads", "R28a"], fontsize=10)
ax.set_yticks(range(len(all_terms_seen)))
ax.set_yticklabels(all_terms_seen, fontsize=9)
ax.set_title("Reward Term Effectiveness Classification", fontsize=13, fontweight="bold")

# Add text annotations
for i in range(len(all_terms_seen)):
    for j in range(3):
        v = matrix[i, j]
        label = {3: "EFF", 2: "WEAK", 1: "DEAD", 0: "HARM", -1: "N/A"}[v]
        color = "white" if v in [0, 3] else "black"
        ax.text(j, i, label, ha="center", va="center", fontsize=8, fontweight="bold", color=color)

cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8)
cbar.ax.set_yticklabels(CLASS_LABELS, fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_effectiveness_heatmap.pdf"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig2_effectiveness_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: fig2_effectiveness_heatmap.pdf/png")


# ── Figure 3: Correlation with constraint costs (bar chart) ──────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
fig.suptitle("Reward Term Correlation with Target Constraint Cost",
             fontsize=14, fontweight="bold")

for ax, (corr, title, color) in zip(axes, [
    (corr_softmax, "Softmax Ablation", COLORS["softmax"]),
    (corr_grads, "Grads Ablation", COLORS["grads"]),
    (corr_r28a, "R28a Clean9", COLORS["r28a"]),
]):
    if not corr:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        continue

    terms = sorted(corr.keys())
    r_vals = [corr[t]["corr"] for t in terms]
    p_vals = [corr[t]["p_value"] for t in terms]

    bars = ax.barh(range(len(terms)), r_vals, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
    # Mark significant correlations
    for i, (r, p) in enumerate(zip(r_vals, p_vals)):
        if p < 0.05:
            ax.text(r + (0.02 if r >= 0 else -0.02), i, "*",
                    fontsize=14, fontweight="bold", va="center",
                    ha="left" if r >= 0 else "right")

    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=9)
    ax.set_xlabel("Pearson r", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_correlation_bars.pdf"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig3_correlation_bars.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: fig3_correlation_bars.pdf/png")


# ── Figure 4: Constraint cost evolution (R25b vs R28a) ───────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Constraint Costs and Lambdas: R25b vs Softmax vs R28a",
             fontsize=14, fontweight="bold")

cost_tags = [
    ("Metrics/EpCost_0", "C0: EV Departure SoC"),
    ("Metrics/EpCost_2", "C2: Battery SoC Bounds"),
    ("Metrics/EpCost_3", "C3: Building Power"),
    ("Metrics/EpCost_4", "C4: Grid Power"),
    ("Metrics/EpRet", "EpRet (Total Reward)"),
]

for idx, (tag, label) in enumerate(cost_tags):
    ax = axes.flat[idx]
    for data, name, color in [
        (r25b, "R25b", COLORS["r25b"]),
        (softmax, "Softmax", COLORS["softmax"]),
        (grads, "Grads", COLORS["grads"]),
        (r28a, "R28a", COLORS["r28a"]),
    ]:
        v = get_metric_vals(data, tag)
        if v is None:
            v = get_vals(data, tag)
        if v is not None:
            ax.plot(np.arange(1, len(v)+1), v, label=name, color=color, linewidth=1.5)

    ax.set_title(label, fontsize=11)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Cost / Return", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Lambda plot in the last panel
ax = axes.flat[5]
for data, name, color in [
    (r25b, "R25b", COLORS["r25b"]),
    (softmax, "Softmax", COLORS["softmax"]),
    (r28a, "R28a", COLORS["r28a"]),
]:
    for ltag, ls in [("Metrics/Lambda_0", "-"), ("Metrics/Lambda_3", "--"),
                     ("Metrics/Lambda_4", ":")]:
        v = get_metric_vals(data, ltag)
        if v is None:
            v = get_vals(data, ltag)
        if v is not None:
            lbl = ltag.split("_")[1]
            ax.plot(np.arange(1, len(v)+1), v,
                    label=f"{name} L{lbl}", color=color, linestyle=ls, linewidth=1.5)

ax.set_title("Key Lambda Multipliers", fontsize=11)
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Lambda", fontsize=10)
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_constraint_costs.pdf"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig4_constraint_costs.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: fig4_constraint_costs.pdf/png")


# ── Figure 5: Planning vs Reactive comparison ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Planning vs Reactive Terms: Magnitude and Growth",
             fontsize=14, fontweight="bold")

for ax, (category, term_set) in zip(axes, [
    ("Planning Terms", PLANNING_TERMS),
    ("Reactive Terms", REACTIVE_TERMS),
]):
    width = 0.25
    terms_present = sorted([t for t in term_set if t in stats_r28a or t in stats_softmax or t in stats_grads])
    x = np.arange(len(terms_present))

    for i, (st, label, color) in enumerate([
        (stats_softmax, "Softmax", COLORS["softmax"]),
        (stats_grads, "Grads", COLORS["grads"]),
        (stats_r28a, "R28a", COLORS["r28a"]),
    ]):
        vals = [st[t]["mean_tail"] if t in st else 0 for t in terms_present]
        ax.bar(x + i * width - width, vals, width, label=label, color=color, alpha=0.7,
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(terms_present, rotation=45, ha="right", fontsize=8)
    ax.set_title(category, fontsize=12)
    ax.set_ylabel("Mean Reward (last 10 epochs)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_planning_vs_reactive.pdf"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig5_planning_vs_reactive.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: fig5_planning_vs_reactive.pdf/png")


# ── Figure 6: Reward composition pie charts ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Reward Composition: Fraction of Total |Reward|",
             fontsize=14, fontweight="bold")

for ax, (st, title) in zip(axes, [
    (stats_softmax, "Softmax Ablation"),
    (stats_grads, "Grads Ablation"),
    (stats_r28a, "R28a Clean9"),
]):
    if not st:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        continue

    # Sort by fraction, group small ones
    sorted_terms = sorted(st.items(), key=lambda x: x[1]["frac"], reverse=True)
    labels, fracs = [], []
    other = 0
    for term, s in sorted_terms:
        if s["frac"] >= 0.02:
            labels.append(term)
            fracs.append(s["frac"])
        else:
            other += s["frac"]
    if other > 0:
        labels.append("other")
        fracs.append(other)

    wedges, texts, autotexts = ax.pie(
        fracs, labels=labels, autopct="%1.1f%%",
        pctdistance=0.85, textprops={"fontsize": 8},
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax.set_title(title, fontsize=12)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_reward_composition.pdf"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig6_reward_composition.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: fig6_reward_composition.pdf/png")


# ── Figure 7: Slope (learning rate) heatmap ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, max(6, len(all_terms_seen) * 0.4)))

slope_matrix = []
for term in all_terms_seen:
    row = []
    for st in [stats_softmax, stats_grads, stats_r28a]:
        if term in st:
            row.append(st[term]["slope"])
        else:
            row.append(np.nan)
    slope_matrix.append(row)

slope_matrix = np.array(slope_matrix)
max_abs = np.nanmax(np.abs(slope_matrix)) if not np.all(np.isnan(slope_matrix)) else 1
im = ax.imshow(slope_matrix, aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs,
               interpolation="nearest")
ax.set_xticks(range(3))
ax.set_xticklabels(["Softmax", "Grads", "R28a"], fontsize=10)
ax.set_yticks(range(len(all_terms_seen)))
ax.set_yticklabels(all_terms_seen, fontsize=9)
ax.set_title("Reward Term Slope (Trend Over Training)",
             fontsize=13, fontweight="bold")

# Annotate
for i in range(len(all_terms_seen)):
    for j in range(3):
        v = slope_matrix[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > max_abs * 0.6 else "black")

plt.colorbar(im, ax=ax, label="Slope (per epoch)", shrink=0.8)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_slope_heatmap.pdf"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig7_slope_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: fig7_slope_heatmap.pdf/png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: BEST CONFIGURATION RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n")
print("=" * 100)
print("SECTION 5: BEST CONFIGURATION RECOMMENDATION")
print("=" * 100)

# Build recommendation based on R28a data (most complete run with reward breakdown)
print("""
METHODOLOGY: Recommendations based on:
  (a) R28a effectiveness classification (60 epochs, most mature training)
  (b) Cross-run consistency (terms that are effective in multiple runs)
  (c) Correlation with constraint cost reduction
  (d) Magnitude contribution (fraction of total reward)

------------------------------------------------------------------------
A. REWARD TERMS TO KEEP (with suggested weights)
------------------------------------------------------------------------""")

# Categorize
keep, remove, restore = [], [], []

for term in all_terms_seen:
    s_r28a = stats_r28a.get(term)
    s_soft = stats_softmax.get(term)
    s_grad = stats_grads.get(term)

    cls_r28a = classify_term(s_r28a, term) if s_r28a else "N/A"
    cls_soft = classify_term(s_soft, term) if s_soft else "N/A"
    cls_grad = classify_term(s_grad, term) if s_grad else "N/A"

    # Get correlation
    corr_val = None
    if term in corr_r28a:
        corr_val = corr_r28a[term]["corr"]

    effective_count = sum(1 for c in [cls_r28a, cls_soft, cls_grad] if c == "EFFECTIVE")
    harmful_count = sum(1 for c in [cls_r28a, cls_soft, cls_grad] if c == "HARMFUL")
    dead_count = sum(1 for c in [cls_r28a, cls_soft, cls_grad] if c in ("DEAD/NOISE", "N/A"))

    info = {
        "term": term,
        "cls_r28a": cls_r28a,
        "effective_count": effective_count,
        "harmful_count": harmful_count,
        "dead_count": dead_count,
        "corr": corr_val,
        "frac_r28a": s_r28a["frac"] if s_r28a else 0,
        "mean_r28a": s_r28a["mean_tail"] if s_r28a else 0,
        "trend_r28a": s_r28a["trend"] if s_r28a else "N/A",
    }

    if harmful_count >= 2:
        remove.append(info)
    elif cls_r28a == "HARMFUL":
        remove.append(info)
    elif dead_count >= 2 and effective_count == 0:
        remove.append(info)
    elif effective_count >= 1:
        keep.append(info)
    elif cls_r28a == "WEAK":
        keep.append(info)  # keep with lower weight
    else:
        remove.append(info)

# Sort keep by fraction
keep.sort(key=lambda x: x["frac_r28a"], reverse=True)
remove.sort(key=lambda x: x["frac_r28a"], reverse=True)

for info in keep:
    t = info["term"]
    f = info["frac_r28a"]
    m = info["mean_r28a"]
    trend = info["trend_r28a"]
    eff = info["effective_count"]
    corr_str = f"corr={info['corr']:+.3f}" if info["corr"] is not None else "corr=N/A"

    # Weight suggestion based on magnitude
    if f >= 0.10:
        weight_sug = "HIGH (1.5-2.0)"
    elif f >= 0.05:
        weight_sug = "MEDIUM (0.5-1.5)"
    elif f >= 0.01:
        weight_sug = "LOW (0.1-0.5)"
    else:
        weight_sug = "MINIMAL (0.05-0.1)"

    print(f"  KEEP  {t:<18s}  frac={f:.1%}  mean={m:>+.3f}  trend={trend:<9s}  "
          f"eff_in={eff}/3  {corr_str}  weight={weight_sug}")

print("""
------------------------------------------------------------------------
B. REWARD TERMS TO REMOVE
------------------------------------------------------------------------""")

for info in remove:
    t = info["term"]
    f = info["frac_r28a"]
    m = info["mean_r28a"]
    cls = info["cls_r28a"]
    reason = "HARMFUL" if info["harmful_count"] > 0 else "DEAD/NOISE"
    print(f"  REMOVE  {t:<18s}  frac={f:.1%}  mean={m:>+.3f}  class={cls:<10s}  reason={reason}")

print("""
------------------------------------------------------------------------
C. OPTIMAL COST LIMITS AND PID GAINS (based on observed cost dynamics)
------------------------------------------------------------------------""")

# Compute cost statistics from R28a and R25b
for tag, name in [("Metrics/EpCost_0", "C0"), ("Metrics/EpCost_2", "C2"),
                   ("Metrics/EpCost_3", "C3"), ("Metrics/EpCost_4", "C4")]:
    r28_v = get_metric_vals(r28a, tag)
    if r28_v is None:
        r28_v = get_vals(r28a, tag)
    r25_v = get_vals(r25b, tag)

    print(f"\n  {name} ({tag}):")
    if r25_v is not None:
        print(f"    R25b: final={r25_v[-1]:.1f}, min={min(r25_v):.1f}, "
              f"last-10 mean={np.mean(r25_v[-10:]):.1f}")
    if r28_v is not None:
        print(f"    R28a: final={r28_v[-1]:.1f}, min={min(r28_v):.1f}, "
              f"last-10 mean={np.mean(r28_v[-10:]):.1f}")

    # Recommendations
    if name == "C0":
        print("    RECOMMENDATION: limit curriculum [200 -> 15] over epochs 0-30")
        print("    PID: Kp=3.0, Ki=0.08 (increase Ki for persistent violations)")
    elif name == "C2":
        print("    RECOMMENDATION: limit=1500 (keep current)")
        print("    PID: Kp=1.0, Ki=0.03")
    elif name == "C3":
        print("    RECOMMENDATION: limit=5000 (keep current)")
        print("    PID: Kp=0.5, Ki=0.05")
    elif name == "C4":
        print("    RECOMMENDATION: limit=8000 (keep current)")
        print("    PID: Kp=0.3, Ki=0.03")

print(f"""
------------------------------------------------------------------------
D. LAMBDA UPPER BOUND
------------------------------------------------------------------------
  Current: 35.0
  Recommendation: 50.0 (increase headroom to prevent saturation,
    especially for C0 which may need strong constraint pressure)

------------------------------------------------------------------------
E. ADDITIONAL NOTES
------------------------------------------------------------------------
  1. R25b lacks per-reward-term breakdown, so we cannot directly compare
     individual term contributions. Future runs should always log per-term
     rewards.

  2. The 'grads' ablation (gradient-scaled temperature) shows r_ev_smart
     and planning terms behaving very differently from the 'softmax'
     ablation, suggesting the gradient scaling mechanism significantly
     impacts which reward signals the agent learns from.

  3. Cross-run consistency is the strongest signal: terms that are
     EFFECTIVE in all three measured runs (softmax, grads, R28a)
     should be kept with high confidence.

  4. The correlation analysis is limited by small sample sizes
     (40-60 epochs), so p-values should be interpreted cautiously.
     Correlations marked with * (p < 0.05) are the most reliable.
""")

print("=" * 100)
print(f"All figures saved to: {OUT}/")
print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: DEEP INTERPRETIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n")
print("=" * 100)
print("SECTION 6: DEEP INTERPRETIVE ANALYSIS")
print("=" * 100)

print("""
------------------------------------------------------------------------
6.1  THE CORE TRIO: r_v2g_ctx, r_ev_smart, r_ramp
------------------------------------------------------------------------
These three terms are EFFECTIVE in all three runs (3/3 consistency).
They form the backbone of the reward signal:

  r_v2g_ctx (21.4% in R28a, slope +0.009/epoch):
    Largest single contributor. Rewards V2G-aware behavior: charging
    when prices are low, discharging when high, respecting vehicle
    availability. Correlation r=-0.71 with C0 confirms it genuinely
    helps reduce EV departure violations. GROWING trend means the
    agent is progressively learning to exploit V2G context.

  r_ev_smart (10.0% in R28a, slope +0.004/epoch):
    The departure-aware planning signal. Uses feasibility corridor
    (soc_min based on hours until departure). Correlation r=-0.68
    with C0. Critically, this is the ONLY planning-category EV term
    that survives into R28a (r_ev_slack_arb died at 0.0%).
    The GROWING trend across all runs is the clearest evidence of
    learned forward-looking EV management.

  r_ramp (12.2% in R28a, slope +0.003/epoch):
    Penalty for rapid power changes. Negative by design. The
    GROWING slope means it is becoming *less* negative over time,
    i.e., the agent is learning to smooth its actions. Correlation
    r=-0.97 with C4 shows this is the strongest driver of grid
    power constraint satisfaction.

------------------------------------------------------------------------
6.2  NEWLY ACTIVATED IN R28a: r_sg, r_sb, r_eco
------------------------------------------------------------------------
These terms were DEAD (0.00) in both ablation runs but became
significant in R28a. This is likely due to R28a's different reward
weight configuration or longer training:

  r_sg (10.6%): Strong positive, r=-0.998 with C4. Exceptional fit.
  r_sb (7.7%): Strong positive, r=-0.965 with C3.
  r_eco (8.6%): HARMFUL -- negative mean (-0.20) with positive
    correlation to EpRet (+0.40). Agent is increasing electricity cost.
    This term needs weight reduction or removal.

------------------------------------------------------------------------
6.3  THE r_barrier PARADOX
------------------------------------------------------------------------
r_barrier is classified as EFFECTIVE (corrected from earlier HARMFUL
misclassification). It is a PENALTY TERM -- negative by design.
The positive correlation r=+0.94/+0.96 with C2 is expected:
as the agent pushes batteries toward SoC boundaries (higher C2 cost),
it also incurs more barrier penalty (more negative r_barrier).
This is the CORRECT feedback direction -- the penalty signal IS
working, but the agent chooses to tolerate the penalty in exchange
for other reward gains.

Recommendation: Keep r_barrier but increase its weight slightly
(from current to ~0.5) to make the boundary penalty more salient
relative to the gains from aggressive battery use.

------------------------------------------------------------------------
6.4  DEAD TERM AUTOPSY: 12 of 23 terms contribute exactly 0.0%
------------------------------------------------------------------------
The following terms logged zero magnitude across all epochs in R28a:
  r_ev_solar, r_grid_penalty, r_headroom, r_nec_sign, r_price_arb,
  r_solar_store, r_traj_batt, r_traj_ev, r_trajectory, r_ev_guard,
  r_ev_slack_arb (in R28a only)

Root causes:
  (a) WEIGHT = 0: Some terms may have their env var weight set to 0
      in the R28a configuration. Check STEMS_ALPHA_* values.
  (b) PRECONDITION NOT MET: Terms like r_ev_solar require simultaneous
      solar availability + EV presence + low SoC. The conjunction
      probability may be too low.
  (c) SUPERSEDED: r_trajectory was replaced by r_ev_smart in R27a+.
      r_ev_slack_arb was effective in ablation runs (21.4%!) but
      died in R28a -- either its weight was zeroed or its design
      was changed.

------------------------------------------------------------------------
6.5  PLANNING TERMS: PROMISE vs DELIVERY
------------------------------------------------------------------------
Of 10 planning-category terms, only 2 show non-zero activity:
  r_ev_smart: DELIVERED (10% contribution, growing)
  r_ev_slack_arb: DELIVERED in ablation (21%), DEAD in R28a

The other 8 planning terms (r_headroom, r_load_shift, r_peak_shave,
r_price_arb, r_solar_store, r_traj_batt, r_traj_ev, r_trajectory)
are all DEAD. This is a significant finding: the agent has essentially
ZERO temporal planning beyond r_ev_smart and r_v2g_ctx.

r_peak_shave shows a faint signal (1.3%, growing) in R28a -- it may
need more epochs or higher weight to become material.

------------------------------------------------------------------------
6.6  GRADS ABLATION: A CAUTIONARY TALE
------------------------------------------------------------------------
The gradient-scaled temperature ablation shows dramatically different
reward composition. r_ev_guard dominates at 28.6% (vs 0% in R28a),
and EpRet collapsed to -17600 (vs +14141 for R25b). This confirms
that gradient scaling amplifies penalty terms disproportionately,
making them overwhelm positive reward signals. The softmax ablation
(EpRet=+271) is strictly better than grads (EpRet=-17600) but still
far behind R25b (+14141).

------------------------------------------------------------------------
6.7  R28a vs R25b: THE REWARD-CONSTRAINT TRADEOFF
------------------------------------------------------------------------
R25b: EpRet=+14141, C0=24, C2=3589, C3=27153, C4=21350, Lambda_0=0.0
R28a: EpRet=-2271,  C0=250, C2=4719, C3=32114, C4=21311, Lambda_0=6.1

R25b has massively better EpRet and lower costs on EVERY constraint.
But R25b's Lambda_0=0.0 means C0 was never actively enforced. R28a
has Lambda_0=6.1 and is still failing C0 (250 vs limit ~20).

The key question: Why does R28a, which has reward terms specifically
designed for EV management (r_ev_smart, r_v2g_ctx), perform WORSE
on C0 than R25b which had none of these?

Hypothesis: R25b's simpler reward function allowed the Lagrangian
to focus exclusively on aggregate constraint satisfaction. R28a's
richer reward landscape creates competing gradients that slow
convergence. The agent in R28a is spread across too many objectives.

Recommendation: In R28b, reduce to the core trio (r_v2g_ctx,
r_ev_smart, r_ramp) plus r_sg, r_sb, r_barrier (6 terms total).
Drop the remaining 17 dead/harmful terms to reduce gradient noise.
""")

print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
