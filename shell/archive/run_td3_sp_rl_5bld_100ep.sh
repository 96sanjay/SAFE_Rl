#!/usr/bin/env bash
# TD3 SP-RL: 5-building, 100 epochs, same reward as R21
# Projection handles C2/C3/C4. Lagrangian handles C0/C1.
set -euo pipefail

SEED="${1:-42}"
PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_BETA_ACTOR="0"

export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"
export CITYLEARN_PID_LAGRANGE="1"

export STEMS_ALPHA_GRID="0.0"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_ALPHA_LOAD_SHIFT="0.0"
export STEMS_ALPHA_GRID_MILD="0.3"
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_XI_RENEWABLE="0.2"
export STEMS_BETA_RAMP="0.3"
export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_GUARD="1.0"
export STEMS_ALPHA_V2G_CONTEXT="3.0"
export STEMS_ALPHA_EV_SOLAR="0.0"
export STEMS_ALPHA_SOLAR_STORE="0.0"
export STEMS_SOLAR_STORE_BATT_ONLY="0"
export STEMS_ALPHA_HEADROOM="0.0"
export STEMS_ALPHA_PRICE_ARB="0.0"
export STEMS_ALPHA_GRID_PENALTY="0.0"
export STEMS_ALPHA_NEC_SIGN="0.0"
export STEMS_EV_SLACK_ARB_SCALE="0.0"

export COST_W_C2="0.0"
export COST_W_C3="5.0"
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="0"
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"

RUN_DIR="runs/td3_sp_rl_5bld_100ep_s${SEED}"

echo ""
echo "============================================"
echo "  TD3 SP-RL 5-building 100ep (seed=$SEED)"
echo "  DiffProjector: C2/C3/C4 hard (cvxpylayers)"
echo "  Lagrangian: C0/C1 (PID)"
echo "  PenC penalty critic"
echo "  100 epochs, 8759 steps/epoch"
echo "  Reward: IDENTICAL to R21"
echo "============================================"
echo ""

mkdir -p "$RUN_DIR"
/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_td3_sp_rl.py \
    --cfg configs/off-policy/td3_sp_rl_5bld_100ep.yaml \
    2>&1 | tee "$RUN_DIR/full_log.txt"
