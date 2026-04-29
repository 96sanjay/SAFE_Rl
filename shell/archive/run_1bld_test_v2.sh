#!/usr/bin/env bash
# 1-Building V2 Test: KL-Regularized PPOLag + NEC-Sign Reward + Action Mask + BC Warmstart
#
# Addresses ALL identified failure modes:
#   1. Action mask → C3/C2 safety layer → lambdas stay low
#   2. NEC-sign reward → cycling 2.5x better than static (EVLearn proven)
#   3. BC warmstart → starts near cycling (SmartV2GRBC)
#   4. KL regularization → prevents drift to static discharge
#   5. Price arbitrage → timing optimization within NEC-gated direction
#   6. All Lagrangian constraints active → agent learns constraint awareness
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Schema (full year) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_1building_3month.json"
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
# R30: ACTION MASK (safety layer for C3/C2)
# =====================================================================
export CITYLEARN_ACTION_MASK="1"

# =====================================================================
# R30: KL REGULARIZATION (prevents drift from BC cycling behavior)
# =====================================================================
export CITYLEARN_KL_BETA="0.1"
export CITYLEARN_KL_BETA_DECAY="0.995"    # ~60% remaining at epoch 100

# =====================================================================
# R30: REWARD TERMS (5 terms, 0 conflicts)
# =====================================================================
# Term 1: NEC-sign gating (EVLearn-inspired, primary cycling signal)
export STEMS_ALPHA_NEC_SIGN="3.0"

# Term 2: Battery price arbitrage (timing within NEC direction)
export STEMS_ALPHA_PRICE_ARB="1.0"

# Term 3: EV urgency (charge before departure)
export STEMS_LAMBDA_EV="15.0"

# Term 4: EV V2G arbitrage (peak hours 17-23)
export STEMS_EV_SLACK_ARB_SCALE="2.5"

# Term 5: Grid stability penalty (quadratic)
export STEMS_ALPHA_GRID_PENALTY="1.5"

# Battery cost scale
export CITYLEARN_STEMS_BATTERY_COST_SCALE="1.0"

# =====================================================================
# ALL OTHER REWARD TERMS DISABLED
# =====================================================================
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_GRID="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_BETA_RAMP="0.0"
export STEMS_XI_RENEWABLE="0.0"
export STEMS_ALPHA_LOAD_SHIFT="0.0"
export STEMS_ALPHA_GRID_MILD="0.0"
export STEMS_ALPHA_EV_GUARD="0.0"
export STEMS_ALPHA_V2G_CONTEXT="0.0"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_BARRIER="0.0"
export STEMS_ALPHA_EV_SOLAR="0.0"
export STEMS_ALPHA_SOLAR_STORE="0.0"
export STEMS_SOLAR_STORE_BATT_ONLY="0"
export STEMS_ALPHA_HEADROOM="0.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_SG_THRESHOLD="0.5"

# =====================================================================
# Saute MDP for C0
# =====================================================================
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="2.0"

# PID Lagrangian
export CITYLEARN_PID_LAGRANGE="1"

# Cost weights
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
export CITYLEARN_C3_CONTROLLABLE="1"

echo ""
echo "============================================"
echo "  1-Building V2 Test: KL-Reg + NEC-Sign + Mask + BC"
echo "  KL: beta=${CITYLEARN_KL_BETA} decay=${CITYLEARN_KL_BETA_DECAY}"
echo "  Reward: nec_sign=${STEMS_ALPHA_NEC_SIGN} price=${STEMS_ALPHA_PRICE_ARB}"
echo "          ev=${STEMS_LAMBDA_EV} v2g=${STEMS_EV_SLACK_ARB_SCALE}"
echo "          grid=${STEMS_ALPHA_GRID_PENALTY}"
echo "  Mask: ON | BC warmstart: ON"
echo "============================================"
echo ""

/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_multi_lag.py \
    --cfg configs/on-policy/test_1bld_v2.yaml \
    --bc
