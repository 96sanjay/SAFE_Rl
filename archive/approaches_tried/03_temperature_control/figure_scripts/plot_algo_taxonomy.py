#!/usr/bin/env python3
"""
Clean taxonomy diagram: 5 benchmark safe RL algorithms classified by
policy type (on/off-policy) and constraint-handling mechanism.

Output: figures/fig_algo_taxonomy.pdf + .png
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Colours ────────────────────────────────────────────────────────────────────
C_ROOT   = "#2c3e50"
C_OFFPOL = "#1a6496"
C_ONPOL  = "#8e3a2e"
C_OFF_LT = "#d6eaf8"
C_ON_LT  = "#fde8e8"
C_LEAF   = "#f4f6f7"
C_LINE   = "#7f8c8d"

# ── Helper: rounded box + text ─────────────────────────────────────────────────
def box(ax, cx, cy, w, h, text, fc, ec,
        fontsize=9, bold=False, sub=None, text_color=None):
    patch = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=3,
    )
    ax.add_patch(patch)
    tc = text_color or ("white" if fc in (C_ROOT, C_OFFPOL, C_ONPOL) else "#1a1a1a")
    fw = "bold" if bold else "normal"
    if sub:
        ax.text(cx, cy + 0.018, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fw, color=tc, zorder=4)
        ax.text(cx, cy - 0.024, sub,  ha="center", va="center",
                fontsize=fontsize - 1.5, color="#555555", style="italic", zorder=4)
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fw, color=tc, zorder=4)

def hline(ax, x1, x2, y):
    ax.plot([x1, x2], [y, y], color=C_LINE, lw=0.9, zorder=2)

def vline(ax, x, y1, y2):
    ax.plot([x, x], [y1, y2], color=C_LINE, lw=0.9, zorder=2)

def connector(ax, px, py_top, cx, cy_bot):
    """Elbow: down from parent, across, down to child."""
    mid = (py_top + cy_bot) / 2
    vline(ax, px, py_top, mid)
    hline(ax, px, cx, mid)
    vline(ax, cx, mid, cy_bot)

# ── Layout ─────────────────────────────────────────────────────────────────────
# X positions for 5 leaf/sub-branch columns
# Off-policy: cols 0,1  |  On-policy: cols 2,3,4
XS = [0.11, 0.29, 0.52, 0.70, 0.88]   # leaf / sub-branch centres

X_OFF  = (XS[0] + XS[1]) / 2          # = 0.20 — off-policy branch centre
X_ON   = (XS[2] + XS[3] + XS[4]) / 3 # = 0.70 — on-policy branch centre
X_ROOT = (X_OFF + X_ON) / 2           # ≈ 0.45

# Y levels (top → bottom)
Y0  = 0.90   # root
Y1  = 0.73   # policy-type branch
Y2  = 0.55   # constraint-type sub-branch
Y3  = 0.32   # algorithm leaf

# Box sizes
RW, RH = 0.58, 0.10   # root  (wide enough for full text)
BW, BH = 0.20, 0.09   # policy-type branch
SW, SH = 0.155, 0.08  # sub-branch  (narrow — 5 fit without touching)
LW, LH = 0.155, 0.11  # leaf

# ── Figure canvas ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.set_xlim(0, 1)
ax.set_ylim(0.18, 1.02)
ax.axis("off")

# ── Root ───────────────────────────────────────────────────────────────────────
box(ax, X_ROOT, Y0, RW, RH,
    "Safe RL Benchmark Algorithms", C_ROOT, C_ROOT,
    fontsize=10, bold=True)

# ── Root → branches (stem + horizontal + drops) ───────────────────────────────
stem_y = Y0 - RH/2
cross_y = (stem_y + Y1 + BH/2) / 2
vline(ax, X_ROOT, stem_y, cross_y)
hline(ax, X_OFF, X_ON, cross_y)
vline(ax, X_OFF, cross_y, Y1 + BH/2)
vline(ax, X_ON,  cross_y, Y1 + BH/2)

# ── Policy-type branches ───────────────────────────────────────────────────────
box(ax, X_OFF, Y1, BW, BH, "Off-Policy", C_OFFPOL, C_OFFPOL, fontsize=9.5, bold=True)
box(ax, X_ON,  Y1, BW, BH, "On-Policy",  C_ONPOL,  C_ONPOL,  fontsize=9.5, bold=True)

# ── Off-policy → sub-branches ─────────────────────────────────────────────────
cross_off_y = (Y1 - BH/2 + Y2 + SH/2) / 2
vline(ax, X_OFF,  Y1 - BH/2,  cross_off_y)
hline(ax, XS[0],  XS[1],       cross_off_y)
vline(ax, XS[0],  cross_off_y, Y2 + SH/2)
vline(ax, XS[1],  cross_off_y, Y2 + SH/2)

box(ax, XS[0], Y2, SW, SH, "Log-Barrier",  C_OFF_LT, C_OFFPOL, fontsize=8)
box(ax, XS[1], Y2, SW, SH, "Lagrangian",   C_OFF_LT, C_OFFPOL, fontsize=8)

# ── On-policy → sub-branches ──────────────────────────────────────────────────
cross_on_y = (Y1 - BH/2 + Y2 + SH/2) / 2
vline(ax, X_ON,   Y1 - BH/2,  cross_on_y)
hline(ax, XS[2],  XS[4],       cross_on_y)
vline(ax, XS[2],  cross_on_y, Y2 + SH/2)
vline(ax, XS[3],  cross_on_y, Y2 + SH/2)
vline(ax, XS[4],  cross_on_y, Y2 + SH/2)

box(ax, XS[2], Y2, SW, SH, "Primal-Dual",  C_ON_LT, C_ONPOL, fontsize=8)
box(ax, XS[3], Y2, SW, SH, "First-Order",  C_ON_LT, C_ONPOL, fontsize=8)
box(ax, XS[4], Y2, SW, SH, "Trust Region", C_ON_LT, C_ONPOL, fontsize=8)

# ── Sub-branches → leaf algorithms ────────────────────────────────────────────
for xi in XS:
    vline(ax, xi, Y2 - SH/2, Y3 + LH/2)

box(ax, XS[0], Y3, LW, LH, "CSAC-LB",
    C_LEAF, C_OFFPOL, fontsize=9, bold=True, sub="(proposed)")
box(ax, XS[1], Y3, LW, LH, "SAC-Lag",
    C_LEAF, C_OFFPOL, fontsize=9, bold=True, sub="(Stooke 2020)")
box(ax, XS[2], Y3, LW, LH, "CUP",
    C_LEAF, C_ONPOL,  fontsize=9, bold=True, sub="(Yang 2022)")
box(ax, XS[3], Y3, LW, LH, "FOCOPS",
    C_LEAF, C_ONPOL,  fontsize=9, bold=True, sub="(Zhang 2020)")
box(ax, XS[4], Y3, LW, LH, "CPO",
    C_LEAF, C_ONPOL,  fontsize=9, bold=True, sub="(Achiam 2017)")

# ── Save ───────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    p = OUT / f"fig_algo_taxonomy.{ext}"
    fig.savefig(p)
    print(f"Saved: {p}")

plt.close(fig)
