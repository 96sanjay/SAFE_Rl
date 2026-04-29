#!/usr/bin/env bash
# Gradient Conflict Diagnostic — 5-building schema, R21-style 8-term STEMS config
#
# Runs one full episode (8759 steps) with a random policy, computes per-term
# REINFORCE gradients, and prints pairwise cosine similarity to identify
# which reward terms pull the policy in conflicting directions.
#
# Active terms (8-term config):
#   r_ramp, r_ren, r_ev, r_load_shift, r_grid_penalty,
#   r_barrier, r_grid_mild, r_eco  (+ r_sg, r_sb from base STEMS)
#
# Runtime: ~5-10 minutes (one episode + gradient computation)
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate citylearn

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Schema: 5-building, full year
# ---------------------------------------------------------------------------
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# ---------------------------------------------------------------------------
# Power limits (auto-calibrated for 5 buildings)
# ---------------------------------------------------------------------------
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# ---------------------------------------------------------------------------
# STEMS reward weights: R21 8-term config
# ---------------------------------------------------------------------------
# Base STEMS terms
export STEMS_MU_ECONOMIC="0.3"        # r_eco (price * NEC)
export STEMS_ALPHA_GRID="1.0"         # r_sg (quadratic grid stability)
export STEMS_ALPHA_BUILD="2.0"        # r_sb (building power stability)
export STEMS_BETA_RAMP="0.5"          # r_ramp (grid ramping penalty)
export STEMS_XI_RENEWABLE="0.2"       # r_ren (renewable self-consumption)

# EV terms
export STEMS_LAMBDA_EV="1.5"          # r_ev (EV urgency × shortfall)
export STEMS_ALPHA_EV_GUARD="0.0"     # r_ev_guard — OFF
export STEMS_ALPHA_V2G_CONTEXT="0.0"  # r_v2g_ctx — OFF

# Battery terms
export STEMS_ALPHA_LOAD_SHIFT="0.15"  # r_load_shift (solar-aware price arb)
export STEMS_ALPHA_GRID_MILD="0.0"    # r_grid_mild — OFF (use grid_penalty instead)
export STEMS_ALPHA_BARRIER="0.5"      # r_barrier (SoC boundary penalty)
export STEMS_ALPHA_PRICE_ARB="0.0"    # r_price_arb — OFF
export STEMS_ALPHA_NEC_SIGN="0.0"     # r_nec_sign — OFF

# Grid penalty
export STEMS_ALPHA_GRID_PENALTY="0.1" # r_grid_penalty (simple quadratic)

# Disabled terms
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_SOLAR="0.0"
export STEMS_ALPHA_SOLAR_STORE="0.0"
export STEMS_SOLAR_STORE_BATT_ONLY="0"
export STEMS_EV_SLACK_ARB_SCALE="0.0"
export STEMS_ALPHA_HEADROOM="0.0"

# V2G flags
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_SG_THRESHOLD="0.0"

# ---------------------------------------------------------------------------
# CMDP cost weights
# ---------------------------------------------------------------------------
export COST_W_C1="10.0"
export COST_W_C1_DENSE="5.0"
export COST_W_C2="1.0"
export COST_W_C3="0.1"
export COST_W_C4="5.0"

# ---------------------------------------------------------------------------
# EV / safety config
# ---------------------------------------------------------------------------
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"
export CITYLEARN_EV_DENSE_COST_SCALE="1.0"
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

# Feature flags
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_SPATIAL_OBS="1"  # Spatial obs for 5-building
export CITYLEARN_TEMPORAL_WINDOW="0"
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="1"
export CITYLEARN_ACTION_MASK="0"  # No action mask for gradient diagnostic
export CITYLEARN_BETA_ACTOR="0"
export CITYLEARN_EV_SAUTE="0"

echo ""
echo "============================================"
echo "  Gradient Conflict Diagnostic"
echo "  Schema: 5-building, full year (8759 steps)"
echo "  R21 8-term STEMS config"
echo "  Active: eco=$STEMS_MU_ECONOMIC sg=$STEMS_ALPHA_GRID sb=$STEMS_ALPHA_BUILD"
echo "          ramp=$STEMS_BETA_RAMP ren=$STEMS_XI_RENEWABLE ev=$STEMS_LAMBDA_EV"
echo "          load_shift=$STEMS_ALPHA_LOAD_SHIFT barrier=$STEMS_ALPHA_BARRIER"
echo "          grid_penalty=$STEMS_ALPHA_GRID_PENALTY"
echo "============================================"
echo ""

python scripts/gradient_conflict_diagnostic.py \
    --gamma 0.99 \
    --device cpu \
    --output conflict_matrix.pt \
    "$@"

echo ""
echo "Diagnostic complete. Results saved to $PROJECT/conflict_matrix.pt"
