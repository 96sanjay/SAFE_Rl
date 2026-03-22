#!/usr/bin/env bash
# TD3 + SP-RL: Deterministic policy with differentiable projection (Markgraf et al. 2025)
# Usage: ./run_td3_sp_rl.sh [seed]
#
# Same reward terms as SAC v4 (SE-RL baseline).
# Only difference: projection is INSIDE the policy (SP-RL) instead of environment (SE-RL).
# Uses cvxpylayers for differentiable QP projection of C2/C3/C4.
# C0/C1 handled via multi-lambda Lagrangian + PID.
set -euo pipefail

SEED="${1:-42}"

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Schema (1-building, 3-month) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_1building_3month.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# --- Thresholds ---
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.91"
export CITYLEARN_STEMS_PNORM_P="4.0"

# --- NO action mask (SP-RL uses differentiable projection instead) ---
export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_BETA_ACTOR="0"

# --- NO Saute (Lagrangian handles C0/C1 directly) ---
export CITYLEARN_EV_SAUTE="0"
export CITYLEARN_EV_DENSE_COST_SCALE="1.0"

# --- PID Lagrangian for C0/C1 ---
export CITYLEARN_PID_LAGRANGE="1"

# --- Reward terms (IDENTICAL to SAC v4) ---
export STEMS_ALPHA_NEC_SIGN="1.5"
export STEMS_ALPHA_PRICE_ARB="1.0"
export STEMS_LAMBDA_EV="15.0"
export STEMS_EV_SLACK_ARB_SCALE="2.5"
export STEMS_ALPHA_GRID_PENALTY="1.5"
export STEMS_BETA_RAMP="0.5"
export STEMS_XI_RENEWABLE="0.7"
export CITYLEARN_STEMS_BATTERY_COST_SCALE="1.0"

# --- All other reward terms disabled ---
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_GRID="0.0"
export STEMS_ALPHA_BUILD="0.0"
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

# --- Cost weights ---
export COST_W_C2="0.0"
export COST_W_C3="5.0"
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# --- Safety clamps OFF (projection handles everything) ---
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="0"
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"
export CITYLEARN_C3_CONTROLLABLE="1"

echo ""
echo "============================================"
echo "  TD3 + SP-RL (seed=$SEED)"
echo "  Differentiable projection (cvxpylayers)"
echo "  C2/C3/C4: hard QP projection"
echo "  C0/C1: multi-lambda Lagrangian + PID"
echo "  PenC: penalty critic (Markgraf et al. 2025)"
echo "  200 epochs, 1-building, 3-month"
echo "============================================"
echo ""

RUN_DIR="runs/td3_sp_rl_1bld_s${SEED}"
mkdir -p "$RUN_DIR"

/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_td3_sp_rl.py \
    --cfg configs/off-policy/td3_sp_rl_1bld.yaml \
    2>&1 | tee "$RUN_DIR/full_log.txt"
