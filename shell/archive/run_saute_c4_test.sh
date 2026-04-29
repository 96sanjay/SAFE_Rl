#!/usr/bin/env bash
# Sauté C4 test: Grid power constraint handled via Sauté MDP
# C4 (cost_stems_grid_power) is zeroed for Lagrangian; budget wrapper provides
# shaped penalty instead. λ4 should → 0 while Sauté penalty handles C4.
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Base env ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_3buildings_3month.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Thresholds
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# PID Lagrangian
export CITYLEARN_PID_LAGRANGE="1"

# C1 Sauté: DISABLED (C1 is already 0, waste of obs dim)
export CITYLEARN_EV_SAUTE="0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="0.0"
export CITYLEARN_EV_DENSE_COST_SCALE="1.0"

# =====================================================================
# NEW: Sauté C4 (grid power) — replaces Lagrangian for C4
# =====================================================================
export SAUTE_C4_ENABLED="1"
export SAUTE_C4_BUDGET="1500"           # = C4 limit (depletes at ~59% of episode)
export SAUTE_C4_PENALTY="5.0"
export SAUTE_C4_GAMMA="1.0"
export SAUTE_C4_SHAPED_ALPHA="10.0"     # Smooth gradient: reward - 10 * |deficit|

# Reward signals (same as saute_test_off baseline)
export STEMS_ALPHA_GRID="0.0"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_ALPHA_LOAD_SHIFT="0.0"
export STEMS_ALPHA_GRID_MILD="0.3"
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_XI_RENEWABLE="0.2"
export STEMS_BETA_RAMP="0.3"

# C3_CONTROLLABLE
export CITYLEARN_C3_CONTROLLABLE="1"

# V2G reward signals
export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_GUARD="1.0"
export STEMS_ALPHA_V2G_CONTEXT="3.0"

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

export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"

# Clamps OFF
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="0"

export STEMS_EV_SLACK_ARB_SCALE="2.0"

echo ""
echo "============================================"
echo "  Sauté C4 Test"
echo "  C4 (grid power) → Sauté MDP (d=1500)"
echo "  λ4 should → 0 (C4 cost zeroed for Lagrangian)"
echo "  Shaped penalty α=10.0 handles C4 via reward"
echo "  Budget depletes at ~59% of episode"
echo "============================================"
echo ""

/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_multi_lag.py \
    --cfg configs/on-policy/saute_c4_test.yaml
