#!/usr/bin/env bash
# R16b: R16a + faster PID + C3_CONTROLLABLE=1
# Goal: Isolate effect of PID tuning + C3 controllability on top of reward reform
#
# Changes from R16a:
#   - C3 PID: Kp=0.5, Ki=0.05 (was 0.3/0.03) — more responsive
#   - C4 PID: Kp=0.3, Ki=0.03 (was 0.1/0.01) — more responsive
#   - C3_CONTROLLABLE=1: only penalize agent's contribution to building power
#     (battery exports no longer cause false C3 violations)
#
# Inherited from R16a:
#   - All reward reform (r_load_shift, r_grid_mild, disabled r_eco/r_sg/r_sb)
#   - All R15b EV features (clamp, guard, V2G context)
#   - PID Lagrangian + Sauté MDP
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Base env ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Thresholds
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# --- PID Lagrangian (from R14) ---
export CITYLEARN_PID_LAGRANGE="1"

# --- Sauté MDP for C1 (from R15b) ---
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

# =====================================================================
# R16a: REWARD REFORM (inherited)
# =====================================================================
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_GRID="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_ALPHA_LOAD_SHIFT="3.0"
export STEMS_ALPHA_GRID_MILD="0.3"
export STEMS_XI_RENEWABLE="1.5"
export STEMS_BETA_RAMP="0.3"

# =====================================================================
# R16b: FASTER PID + C3_CONTROLLABLE=1 (NEW)
# =====================================================================
# C3_CONTROLLABLE=1: battery exports don't cause false C3 violations
export CITYLEARN_C3_CONTROLLABLE="1"

# EV features (from R15b)
export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export CITYLEARN_EV_ACTION_CLAMP="1"
export CITYLEARN_EV_CLAMP_MARGIN="0.1"
export STEMS_ALPHA_EV_GUARD="5.0"
export STEMS_ALPHA_V2G_CONTEXT="3.0"

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

echo ""
echo "============================================"
echo "  R16b: R16a + Faster PID + C3_CONTROLLABLE"
echo "  Changes from R16a:"
echo "    - C3 PID: Kp=0.5, Ki=0.05 (was 0.3/0.03)"
echo "    - C4 PID: Kp=0.3, Ki=0.03 (was 0.1/0.01)"
echo "    - C3_CONTROLLABLE=1 (exports don't cause C3)"
echo "  Inherited from R16a:"
echo "    - Reward reform (load_shift, grid_mild)"
echo "    - All R15b EV features"
echo "    - PID Lagrangian + Sauté MDP"
echo "============================================"
echo ""

python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r16b_pid_c3ctrl.yaml
