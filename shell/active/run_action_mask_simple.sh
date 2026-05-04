#!/usr/bin/env bash
# R29: Action Mask + Simplified 4-Term Reward
#
# Architecture:
#   - Action mask (safety layer) → prevents C3/C2 violations structurally
#   - 4 reward terms (0 conflicts): price_arb, ev_urgency, ev_v2g, grid_penalty
#   - Lagrangian active for ALL constraints (mask keeps lambdas low)
#   - Saute MDP for C0 (EV departure)
#
# What's different from R28:
#   - Action mask NEW
#   - r_load_shift → replaced by simple R_batt = -action × price
#   - r_ramp, r_ev_guard, r_barrier → REMOVED (conflicted with cycling)
#   - r_grid_penalty → NEW (simple quadratic)
#   - Saute shaped_alpha: 10.0 → 2.0 (was drowning reward)
#   - No curriculum (mask keeps lambdas low from epoch 0)
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT"

if [ -n "${CONDA_EXE:-}" ]; then
    eval "$(${CONDA_EXE} shell.bash hook)"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not found. Install conda or set CONDA_EXE." >&2
    exit 1
fi
conda activate citylearn

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Schema (full year) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# --- Thresholds ---
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# =====================================================================
# R29 NEW: ACTION MASK (safety layer for C3/C2)
# =====================================================================
export CITYLEARN_ACTION_MASK="1"

# =====================================================================
# R29: SIMPLIFIED REWARD (4 terms, 0 conflicts)
# =====================================================================
# Term 1: Battery price arbitrage (AL-SAC style)
export STEMS_ALPHA_PRICE_ARB="2.0"

# Term 2: EV urgency (charge before departure)
export STEMS_LAMBDA_EV="4.0"

# Term 3: EV V2G arbitrage (peak hours 17-23 only)
export STEMS_EV_SLACK_ARB_SCALE="2.5"

# Term 4: Grid stability penalty (quadratic)
export STEMS_ALPHA_GRID_PENALTY="1.5"

# =====================================================================
# ALL OTHER REWARD TERMS DISABLED
# =====================================================================
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_GRID="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_BETA_RAMP="0.0"           # REMOVED: conflicts with cycling
export STEMS_XI_RENEWABLE="0.0"
export STEMS_ALPHA_LOAD_SHIFT="0.0"    # REPLACED by STEMS_ALPHA_PRICE_ARB
export STEMS_ALPHA_GRID_MILD="0.0"
export STEMS_ALPHA_EV_GUARD="0.0"      # REMOVED: conflicts with V2G
export STEMS_ALPHA_V2G_CONTEXT="0.0"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_BARRIER="0.0"       # REMOVED: conflicts with cycling
export STEMS_ALPHA_EV_SOLAR="0.0"
export STEMS_ALPHA_SOLAR_STORE="0.0"
export STEMS_SOLAR_STORE_BATT_ONLY="0"
export STEMS_ALPHA_HEADROOM="0.0"      # REDUNDANT: mask handles this
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_SG_THRESHOLD="0.5"

# =====================================================================
# Saute MDP for C0 (reduced shaped_alpha: 10→2)
# =====================================================================
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="2.0"   # R29: DOWN from 10.0

# PID Lagrangian
export CITYLEARN_PID_LAGRANGE="1"

# Cost weights (for single-lambda fallback)
export COST_W_C2="0.0"
export COST_W_C3="5.0"
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# Clamps OFF (mask handles safety)
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="0"
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"

# C3 controllable (kept for cost computation, mask handles enforcement)
export CITYLEARN_C3_CONTROLLABLE="1"

echo ""
echo "============================================"
echo "  R29: Action Mask + Simplified Reward"
echo "  4 terms: price_arb(2.0) + ev(4.0) + v2g(2.5) + grid(1.5)"
echo "  Action mask: C3/C2 enforcement"
echo "  Lagrangian: all constraints (backed by mask)"
echo "  Saute: C0 (shaped_alpha=2.0)"
echo "  No curriculum, no conflicting terms"
echo "============================================"
echo ""

python scripts/train_multi_lag.py \
    --cfg configs/active/action_mask_simple.yaml
