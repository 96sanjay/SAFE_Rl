#!/usr/bin/env bash
# R15b SMOKE TEST: 1 epoch only
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

# --- PID Lagrangian ---
export CITYLEARN_PID_LAGRANGE="1"

# --- Sauté MDP for C1 (fixed: budget 25000, shaped penalty) ---
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

# --- STEMS weights ---
export STEMS_MU_ECONOMIC="1.5"
export STEMS_ALPHA_GRID="3.0"
export STEMS_ALPHA_BUILD="1.0"
export STEMS_BETA_RAMP="1.0"
export STEMS_XI_RENEWABLE="0.2"
export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"

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

# No spatial/temporal
export CITYLEARN_C3_CONTROLLABLE="0"
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"

# --- R15a: Hard EV action clamp + anti-discharge penalty ---
export CITYLEARN_EV_ACTION_CLAMP="1"
export CITYLEARN_EV_CLAMP_MARGIN="0.1"
export STEMS_ALPHA_EV_GUARD="5.0"

# --- R15b: Context-aware V2G signal ---
export STEMS_ALPHA_V2G_CONTEXT="3.0"

echo "[R15b SMOKE] 1 epoch — verifying V2G context reward"

python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r15b_smoke.yaml
