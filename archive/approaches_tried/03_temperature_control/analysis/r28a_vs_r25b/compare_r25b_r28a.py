#!/usr/bin/env python3
"""
Comprehensive comparison: R25b (baseline, 40 epochs) vs R28a (current, ~38 epochs).
Generates publication-quality figures for thesis.

R25b data: tb_ablation_data.npz (softmax_ prefix)
R28a data: /tmp/r28a_clean9.log (parsed from training log tables)
"""

import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────
NPZ_PATH = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/docs/temperature_case_study/tb_ablation_data.npz"
LOG_PATH = "/tmp/r28a_clean9.log"
OUT_DIR  = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/docs/temperature_case_study/r28a_vs_r25b"

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.8,
    'figure.facecolor': 'white',
})
COLOR_R25B = '#2166AC'  # blue
COLOR_R28A = '#B2182B'  # red
LIMIT_STYLE = dict(linestyle='--', linewidth=1.0, alpha=0.7)

# ── Load R25b (NPZ) ───────────────────────────────────────────────────
print("[1/7] Loading R25b data from NPZ...")
npz = np.load(NPZ_PATH)

def get_r25b(metric_suffix):
    """Return (steps, values) for a softmax_ prefixed metric."""
    key_v = f"softmax_{metric_suffix}_values"
    key_s = f"softmax_{metric_suffix}_steps"
    if key_v in npz and key_s in npz:
        return npz[key_s], npz[key_v]
    return None, None

# ── Parse R28a (log) ──────────────────────────────────────────────────
print("[2/7] Parsing R28a training log...")
with open(LOG_PATH, 'r') as f:
    log_content = f.read()

# The log uses rich-style tables with characters:
#   Top border:    ┏━━━━┳━━━━┓
#   Header row:    ┃ Metrics ┃ Value ┃
#   Header sep:    ┡━━━━╇━━━━┩
#   Data rows:     │ metric  │ value │
#   Bottom border: └────┴────┘
metric_pattern = re.compile(r'\u2502\s+([\w/_.]+)\s+\u2502\s+([^\s\u2502]+)\s+\u2502')

# Find table top borders (┏━...┓)
table_start_positions = [m.start() for m in re.finditer(r'\u250F[\u2501\u2533]+\u2513', log_content)]
# Find table bottom borders (└─...┘)
table_end_positions = [m.end() for m in re.finditer(r'\u2514[\u2500\u2534]+\u2518', log_content)]

print(f"  Found {len(table_start_positions)} table starts, {len(table_end_positions)} table ends")

epochs_r28a = []
for start, end in zip(table_start_positions, table_end_positions):
    block = log_content[start:end]
    metrics = {}
    for name, val in metric_pattern.findall(block):
        try:
            metrics[name] = float(val)
        except ValueError:
            pass
    if metrics and 'Metrics/EpRet' in metrics:
        epochs_r28a.append(metrics)

n_epochs_r28a = len(epochs_r28a)
print(f"  Parsed {n_epochs_r28a} complete epochs for R28a")

def get_r28a(metric_name):
    """Return (epochs_array, values_array) for R28a metric."""
    vals = []
    for ep in epochs_r28a:
        if metric_name in ep:
            vals.append(ep[metric_name])
        else:
            vals.append(np.nan)
    epochs = np.arange(1, n_epochs_r28a + 1)
    return epochs, np.array(vals)

# ── Cost limits ────────────────────────────────────────────────────────
R25B_LIMITS = {0: 200, 2: 1500, 3: 5000, 4: 8000}
R28A_LIMITS = {0: 100, 2: 1000, 3: 5000, 4: 8000}
LAMBDA_MAX = 35.0

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Constraint Costs
# ══════════════════════════════════════════════════════════════════════
print("[3/7] Figure 1: Constraint Costs...")
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("Figure 1: Constraint Costs -- R25b (baseline) vs R28a", fontsize=14, fontweight='bold')

constraint_labels = {0: 'C0 (EV departure SOC)', 2: 'C2 (grid stress)',
                     3: 'C3 (battery wear)', 4: 'C4 (V2G abuse)'}

for idx, (ci, label) in enumerate(constraint_labels.items()):
    ax = axes1.flat[idx]

    # R25b
    s25, v25 = get_r25b(f"Metrics_EpCost_{ci}")
    if s25 is not None:
        ax.plot(s25, v25, color=COLOR_R25B, label='R25b', marker='o', markersize=2)

    # R28a
    s28, v28 = get_r28a(f"Metrics/EpCost_{ci}")
    ax.plot(s28, v28, color=COLOR_R28A, label='R28a', marker='s', markersize=2)

    # Limits
    xlim_max = max(len(s25) if s25 is not None else 0, len(s28)) + 2
    ax.axhline(R25B_LIMITS[ci], color=COLOR_R25B, label=f'R25b limit={R25B_LIMITS[ci]}', **LIMIT_STYLE)
    ax.axhline(R28A_LIMITS[ci], color=COLOR_R28A, label=f'R28a limit={R28A_LIMITS[ci]}', **LIMIT_STYLE)

    ax.set_title(label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Episode Cost')
    ax.legend(loc='best', fontsize=8)

fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig(f"{OUT_DIR}/fig1_constraint_costs.png", dpi=200, bbox_inches='tight')
fig1.savefig(f"{OUT_DIR}/fig1_constraint_costs.pdf", bbox_inches='tight')
print("  Saved fig1_constraint_costs.png/pdf")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Lambda Multipliers
# ══════════════════════════════════════════════════════════════════════
print("[4/7] Figure 2: Lambda Multipliers...")
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("Figure 2: Lagrange Multipliers -- R25b vs R28a", fontsize=14, fontweight='bold')

lambda_labels = {0: r'$\lambda_0$ (EV SOC)', 2: r'$\lambda_2$ (grid stress)',
                 3: r'$\lambda_3$ (battery wear)', 4: r'$\lambda_4$ (V2G abuse)'}

for idx, (ci, label) in enumerate(lambda_labels.items()):
    ax = axes2.flat[idx]

    s25, v25 = get_r25b(f"Metrics_Lambda_{ci}")
    if s25 is not None:
        ax.plot(s25, v25, color=COLOR_R25B, label='R25b', marker='o', markersize=2)

    s28, v28 = get_r28a(f"Metrics/Lambda_{ci}")
    ax.plot(s28, v28, color=COLOR_R28A, label='R28a', marker='s', markersize=2)

    ax.axhline(LAMBDA_MAX, color='gray', label=f'$\\lambda_{{max}}$={LAMBDA_MAX}', **LIMIT_STYLE)

    ax.set_title(label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Lambda value')
    ax.legend(loc='best', fontsize=8)

fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(f"{OUT_DIR}/fig2_lambda_multipliers.png", dpi=200, bbox_inches='tight')
fig2.savefig(f"{OUT_DIR}/fig2_lambda_multipliers.pdf", bbox_inches='tight')
print("  Saved fig2_lambda_multipliers.png/pdf")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Reward Terms Comparison
# ══════════════════════════════════════════════════════════════════════
print("[5/7] Figure 3: Reward Terms...")

# Shared terms
shared_terms = ['r_ev', 'r_ev_smart', 'r_v2g_ctx', 'r_barrier', 'r_ramp', 'r_ren']
# R28a-only terms (new)
r28a_only = ['r_sb', 'r_sg', 'r_eco', 'r_peak_shave', 'r_load_shift']
# R25b-only terms (removed)
r25b_only = ['r_ev_guard', 'r_grid_mild', 'r_ev_slack_arb']

all_terms = shared_terms + r28a_only + r25b_only
n_terms = len(all_terms)
n_cols = 3
n_rows = (n_terms + n_cols - 1) // n_cols

fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
fig3.suptitle("Figure 3: Per-Reward-Term Comparison", fontsize=14, fontweight='bold')

for idx, term in enumerate(all_terms):
    ax = axes3.flat[idx]

    # R25b
    s25, v25 = get_r25b(f"Reward_{term}")
    has_r25b = s25 is not None and not np.all(np.isnan(v25))
    if has_r25b:
        ax.plot(s25, v25, color=COLOR_R25B, label='R25b', marker='o', markersize=2)

    # R28a
    s28, v28 = get_r28a(f"Reward/{term}")
    has_r28a = not np.all(np.isnan(v28))
    if has_r28a:
        ax.plot(s28, v28, color=COLOR_R28A, label='R28a', marker='s', markersize=2)

    # Annotate category
    if term in shared_terms:
        cat = "SHARED"
        bg = '#E8E8E8'
    elif term in r28a_only:
        cat = "R28a ONLY"
        bg = '#FFE0E0'
    else:
        cat = "R25b ONLY"
        bg = '#E0E0FF'

    ax.set_title(f"{term} [{cat}]", fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor=bg, alpha=0.5))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward/step')
    ax.legend(loc='best', fontsize=7)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

# Hide unused subplots
for idx in range(n_terms, n_rows * n_cols):
    axes3.flat[idx].set_visible(False)

fig3.tight_layout(rect=[0, 0, 1, 0.95])
fig3.savefig(f"{OUT_DIR}/fig3_reward_terms.png", dpi=200, bbox_inches='tight')
fig3.savefig(f"{OUT_DIR}/fig3_reward_terms.pdf", bbox_inches='tight')
print("  Saved fig3_reward_terms.png/pdf")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Total EV Signal
# ══════════════════════════════════════════════════════════════════════
print("[5/7] Figure 4: Total EV Signal...")
fig4, ax4 = plt.subplots(1, 1, figsize=(12, 5))
fig4.suptitle("Figure 4: Total EV Reward Signal (sum of all EV-related terms)", fontsize=14, fontweight='bold')

# R25b EV total: r_ev + r_ev_guard + r_ev_smart + r_v2g_ctx + r_ev_slack_arb
r25b_ev_terms = ['r_ev', 'r_ev_guard', 'r_ev_smart', 'r_v2g_ctx', 'r_ev_slack_arb']
r25b_ev_total = None
r25b_steps = None
for term in r25b_ev_terms:
    s, v = get_r25b(f"Reward_{term}")
    if s is not None:
        if r25b_ev_total is None:
            r25b_ev_total = np.zeros_like(v)
            r25b_steps = s
        r25b_ev_total += v

# R28a EV total: r_ev + r_ev_smart + r_v2g_ctx
r28a_ev_terms = ['r_ev', 'r_ev_smart', 'r_v2g_ctx']
r28a_ev_total = np.zeros(n_epochs_r28a)
r28a_steps = np.arange(1, n_epochs_r28a + 1)
for term in r28a_ev_terms:
    _, v = get_r28a(f"Reward/{term}")
    r28a_ev_total += np.nan_to_num(v)

ax4.plot(r25b_steps, r25b_ev_total, color=COLOR_R25B, label=f'R25b ({" + ".join(r25b_ev_terms)})',
         marker='o', markersize=3, linewidth=2)
ax4.plot(r28a_steps, r28a_ev_total, color=COLOR_R28A, label=f'R28a ({" + ".join(r28a_ev_terms)})',
         marker='s', markersize=3, linewidth=2)

ax4.axhline(0, color='black', linewidth=0.5, alpha=0.5)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Total EV reward / step')
ax4.legend(fontsize=9)
ax4.set_title('Aggregate EV Charging Incentive Over Training')

# Also plot individual component breakdown as stacked area in inset
# R25b breakdown
ax_inset_l = fig4.add_axes([0.12, 0.55, 0.25, 0.35])  # left inset
for term in r25b_ev_terms:
    s, v = get_r25b(f"Reward_{term}")
    if s is not None and np.any(v != 0):
        ax_inset_l.plot(s, v, label=term, linewidth=1.2, alpha=0.8)
ax_inset_l.set_title('R25b components', fontsize=8)
ax_inset_l.legend(fontsize=6, loc='best')
ax_inset_l.tick_params(labelsize=7)
ax_inset_l.axhline(0, color='black', linewidth=0.3)
ax_inset_l.grid(True, alpha=0.2)

# R28a breakdown
ax_inset_r = fig4.add_axes([0.65, 0.55, 0.25, 0.35])  # right inset
for term in r28a_ev_terms:
    _, v = get_r28a(f"Reward/{term}")
    if np.any(~np.isnan(v)) and np.any(v != 0):
        ax_inset_r.plot(r28a_steps, v, label=term, linewidth=1.2, alpha=0.8)
ax_inset_r.set_title('R28a components', fontsize=8)
ax_inset_r.legend(fontsize=6, loc='best')
ax_inset_r.tick_params(labelsize=7)
ax_inset_r.axhline(0, color='black', linewidth=0.3)
ax_inset_r.grid(True, alpha=0.2)

fig4.savefig(f"{OUT_DIR}/fig4_ev_signal.png", dpi=200, bbox_inches='tight')
fig4.savefig(f"{OUT_DIR}/fig4_ev_signal.pdf", bbox_inches='tight')
print("  Saved fig4_ev_signal.png/pdf")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Training Health
# ══════════════════════════════════════════════════════════════════════
print("[6/7] Figure 5: Training Health...")
fig5, axes5 = plt.subplots(1, 3, figsize=(16, 5))
fig5.suptitle("Figure 5: Training Health Indicators", fontsize=14, fontweight='bold')

health_metrics = [
    ('EpRet', 'Metrics_EpRet', 'Metrics/EpRet', 'Episode Return'),
    ('Entropy', 'Train_Entropy', 'Train/Entropy', 'Policy Entropy'),
    ('KL', 'Train_KL', 'Train/KL', 'KL Divergence'),
]

for idx, (short, r25b_key, r28a_key, title) in enumerate(health_metrics):
    ax = axes5[idx]

    s25, v25 = get_r25b(r25b_key)
    if s25 is not None:
        ax.plot(s25, v25, color=COLOR_R25B, label='R25b', marker='o', markersize=2)

    s28, v28 = get_r28a(r28a_key)
    ax.plot(s28, v28, color=COLOR_R28A, label='R28a', marker='s', markersize=2)

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(short)
    ax.legend(loc='best')

fig5.tight_layout(rect=[0, 0, 1, 0.93])
fig5.savefig(f"{OUT_DIR}/fig5_training_health.png", dpi=200, bbox_inches='tight')
fig5.savefig(f"{OUT_DIR}/fig5_training_health.pdf", bbox_inches='tight')
print("  Saved fig5_training_health.png/pdf")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Constraint Convergence Rate (normalized violation ratio)
# ══════════════════════════════════════════════════════════════════════
print("[6/7] Figure 6: Constraint Convergence Rate...")
fig6, axes6 = plt.subplots(2, 2, figsize=(14, 10))
fig6.suptitle("Figure 6: Normalized Violation Ratio (EpCost / Limit) -- lower is better",
              fontsize=14, fontweight='bold')

for idx, (ci, label) in enumerate(constraint_labels.items()):
    ax = axes6.flat[idx]

    # R25b
    s25, v25 = get_r25b(f"Metrics_EpCost_{ci}")
    if s25 is not None:
        ratio_25 = v25 / R25B_LIMITS[ci]
        ax.plot(s25, ratio_25, color=COLOR_R25B, label=f'R25b (limit={R25B_LIMITS[ci]})',
                marker='o', markersize=2)

    # R28a
    s28, v28 = get_r28a(f"Metrics/EpCost_{ci}")
    ratio_28 = v28 / R28A_LIMITS[ci]
    ax.plot(s28, ratio_28, color=COLOR_R28A, label=f'R28a (limit={R28A_LIMITS[ci]})',
            marker='s', markersize=2)

    # Feasibility line at ratio=1
    ax.axhline(1.0, color='green', linewidth=1.5, linestyle='--', alpha=0.7, label='Feasible (ratio=1)')

    ax.set_title(label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Violation ratio (cost / limit)')
    ax.legend(loc='best', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.05)

fig6.tight_layout(rect=[0, 0, 1, 0.95])
fig6.savefig(f"{OUT_DIR}/fig6_convergence_rate.png", dpi=200, bbox_inches='tight')
fig6.savefig(f"{OUT_DIR}/fig6_convergence_rate.pdf", bbox_inches='tight')
print("  Saved fig6_convergence_rate.png/pdf")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 7: Summary Table
# ══════════════════════════════════════════════════════════════════════
print("[7/7] Figure 7: Summary Table...")

# Compute summary stats
all_reward_terms = sorted(set(shared_terms + r28a_only + r25b_only))

table_rows = []
for term in all_reward_terms:
    # R25b stats
    _, v25 = get_r25b(f"Reward_{term}")
    if v25 is not None and not np.all(v25 == 0):
        r25b_mean = np.nanmean(v25)
        r25b_final = v25[-1]
        r25b_status = "ACTIVE" if abs(r25b_mean) > 0.05 else ("weak" if abs(r25b_mean) > 0.01 else "DEAD")
    else:
        r25b_mean = 0.0
        r25b_final = 0.0
        r25b_status = "N/A" if v25 is None or np.all(v25 == 0) else "DEAD"

    # R28a stats
    _, v28 = get_r28a(f"Reward/{term}")
    if not np.all(np.isnan(v28)) and not np.all(v28 == 0):
        r28a_mean = np.nanmean(v28)
        r28a_final = v28[-1] if not np.isnan(v28[-1]) else 0.0
        r28a_status = "ACTIVE" if abs(r28a_mean) > 0.05 else ("weak" if abs(r28a_mean) > 0.01 else "DEAD")
    else:
        r28a_mean = 0.0
        r28a_final = 0.0
        r28a_status = "N/A" if np.all(np.isnan(v28)) or np.all(v28 == 0) else "DEAD"

    category = "SHARED" if term in shared_terms else ("NEW" if term in r28a_only else "REMOVED")
    table_rows.append([term, category, f"{r25b_mean:+.4f}", r25b_status,
                       f"{r28a_mean:+.4f}", r28a_status])

# Constraint convergence summary
constraint_summary = []
for ci, label in constraint_labels.items():
    _, v25 = get_r25b(f"Metrics_EpCost_{ci}")
    _, v28 = get_r28a(f"Metrics/EpCost_{ci}")

    r25b_init = v25[0] if v25 is not None else np.nan
    r25b_final = v25[-1] if v25 is not None else np.nan
    r25b_ratio_final = r25b_final / R25B_LIMITS[ci] if v25 is not None else np.nan

    r28a_init = v28[0] if not np.isnan(v28[0]) else np.nan
    r28a_final = v28[-1] if not np.isnan(v28[-1]) else np.nan
    r28a_ratio_final = r28a_final / R28A_LIMITS[ci]

    # Determine which converges faster
    if v25 is not None:
        r25b_below = np.where(v25 / R25B_LIMITS[ci] < 1.0)[0]
        r25b_first_feasible = r25b_below[0] + 1 if len(r25b_below) > 0 else "never"
    else:
        r25b_first_feasible = "N/A"

    r28a_below = np.where(v28 / R28A_LIMITS[ci] < 1.0)[0]
    r28a_first_feasible = r28a_below[0] + 1 if len(r28a_below) > 0 else "never"

    constraint_summary.append([
        f"C{ci}", f"{r25b_init:.0f}", f"{r25b_final:.0f}", f"{r25b_ratio_final:.1f}x",
        str(r25b_first_feasible),
        f"{r28a_init:.0f}", f"{r28a_final:.0f}", f"{r28a_ratio_final:.1f}x",
        str(r28a_first_feasible)
    ])

# Create table figure
fig7, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 12),
                                       gridspec_kw={'height_ratios': [3, 2]})
fig7.suptitle("Figure 7: Summary Comparison Table", fontsize=14, fontweight='bold')

# Top: Reward terms table
ax_top.axis('off')
col_labels = ['Reward Term', 'Category', 'R25b Mean', 'R25b Status', 'R28a Mean', 'R28a Status']
table1 = ax_top.table(cellText=table_rows, colLabels=col_labels,
                       loc='center', cellLoc='center')
table1.auto_set_font_size(False)
table1.set_fontsize(8)
table1.scale(1.0, 1.4)

# Color cells by status
for (row, col), cell in table1.get_celld().items():
    if row == 0:  # header
        cell.set_facecolor('#D4D4D4')
        cell.set_text_props(fontweight='bold')
    elif col in [3, 5]:  # status columns
        text = cell.get_text().get_text()
        if text == 'ACTIVE':
            cell.set_facecolor('#C8E6C9')
        elif text == 'weak':
            cell.set_facecolor('#FFF9C4')
        elif text == 'DEAD':
            cell.set_facecolor('#FFCDD2')
        elif text == 'N/A':
            cell.set_facecolor('#E0E0E0')
    elif col == 1:  # category
        text = cell.get_text().get_text()
        if text == 'NEW':
            cell.set_facecolor('#E3F2FD')
        elif text == 'REMOVED':
            cell.set_facecolor('#FCE4EC')

ax_top.set_title("Reward Term Health Summary", fontsize=11, pad=10)

# Bottom: Constraint convergence table
ax_bot.axis('off')
c_col_labels = ['Constraint', 'R25b Init', 'R25b Final', 'R25b Ratio',
                'R25b 1st Feasible', 'R28a Init', 'R28a Final', 'R28a Ratio',
                'R28a 1st Feasible']
table2 = ax_bot.table(cellText=constraint_summary, colLabels=c_col_labels,
                       loc='center', cellLoc='center')
table2.auto_set_font_size(False)
table2.set_fontsize(8)
table2.scale(1.0, 1.6)

for (row, col), cell in table2.get_celld().items():
    if row == 0:
        cell.set_facecolor('#D4D4D4')
        cell.set_text_props(fontweight='bold')
    elif col in [3, 7]:  # ratio columns
        text = cell.get_text().get_text()
        try:
            ratio_val = float(text.replace('x', ''))
            if ratio_val <= 1.0:
                cell.set_facecolor('#C8E6C9')
            elif ratio_val <= 5.0:
                cell.set_facecolor('#FFF9C4')
            else:
                cell.set_facecolor('#FFCDD2')
        except ValueError:
            pass

ax_bot.set_title("Constraint Convergence Summary", fontsize=11, pad=10)

fig7.tight_layout(rect=[0, 0.02, 1, 0.95])
fig7.savefig(f"{OUT_DIR}/fig7_summary_table.png", dpi=200, bbox_inches='tight')
fig7.savefig(f"{OUT_DIR}/fig7_summary_table.pdf", bbox_inches='tight')
print("  Saved fig7_summary_table.png/pdf")

# ══════════════════════════════════════════════════════════════════════
# Combined PDF
# ══════════════════════════════════════════════════════════════════════
print("\nCombining all figures into single PDF...")
with PdfPages(f"{OUT_DIR}/r28a_vs_r25b_full_comparison.pdf") as pdf:
    for fig in [fig1, fig2, fig3, fig4, fig5, fig6, fig7]:
        pdf.savefig(fig, bbox_inches='tight')

print(f"\nAll outputs saved to: {OUT_DIR}/")
print("  Individual PNGs: fig1-7_*.png")
print("  Individual PDFs: fig1-7_*.pdf")
print("  Combined PDF:    r28a_vs_r25b_full_comparison.pdf")

# ══════════════════════════════════════════════════════════════════════
# Print quick text summary to console
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("QUICK COMPARISON SUMMARY")
print("=" * 70)

# EpRet
_, v25_ret = get_r25b("Metrics_EpRet")
_, v28_ret = get_r28a("Metrics/EpRet")
print(f"\nEpRet:")
print(f"  R25b: start={v25_ret[0]:.0f}, end={v25_ret[-1]:.0f}, best={v25_ret.max():.0f}")
print(f"  R28a: start={v28_ret[0]:.0f}, end={v28_ret[-1]:.0f}, best={np.nanmax(v28_ret):.0f}")

print(f"\nConstraint Final Costs (violation ratio):")
for ci, label in constraint_labels.items():
    _, v25 = get_r25b(f"Metrics_EpCost_{ci}")
    _, v28 = get_r28a(f"Metrics/EpCost_{ci}")
    r25b_r = v25[-1] / R25B_LIMITS[ci] if v25 is not None else float('nan')
    r28a_r = v28[-1] / R28A_LIMITS[ci]
    winner = "R25b" if r25b_r < r28a_r else "R28a"
    print(f"  C{ci}: R25b={v25[-1]:.0f} ({r25b_r:.1f}x), R28a={v28[-1]:.0f} ({r28a_r:.1f}x) -- {winner} better")

print(f"\nLambda Final Values:")
for ci in [0, 2, 3, 4]:
    _, v25 = get_r25b(f"Metrics_Lambda_{ci}")
    _, v28 = get_r28a(f"Metrics/Lambda_{ci}")
    print(f"  L{ci}: R25b={v25[-1]:.2f}, R28a={v28[-1]:.2f} (max={LAMBDA_MAX})")

print(f"\nActive Reward Terms (|mean| > 0.05):")
for row in table_rows:
    term, cat, r25b_m, r25b_s, r28a_m, r28a_s = row
    if r25b_s == 'ACTIVE' or r28a_s == 'ACTIVE':
        print(f"  {term:20s} [{cat:8s}] R25b={r25b_m} ({r25b_s}), R28a={r28a_m} ({r28a_s})")

print(f"\nDead Terms (|mean| < 0.01 in BOTH):")
dead = [row[0] for row in table_rows if row[3] in ('DEAD', 'N/A') and row[5] in ('DEAD', 'N/A')]
print(f"  {', '.join(dead) if dead else 'None'}")

plt.close('all')
print("\nDone.")
