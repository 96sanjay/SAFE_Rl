#!/usr/bin/env bash
# SE-RL Paper-Exact: 5-building, Markgraf et al. 2025
# Closest-point projection (clip) + penalty (Eq. 23-24)
# Same reward as SP-RL run for direct comparison
set -euo pipefail

SEED="${1:-42}"
PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- 5-building schema ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# --- Thresholds (same as R21 / SP-RL) ---
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# --- SE-RL PROJECTION (paper-exact clip + penalty) ---
export CITYLEARN_SERL_PROJECTION="1"
export SE_RL_PENALTY_WEIGHT="0.5"
export MASK_C4_ENABLED="1"

# --- NO action mask, NO beta actor, NO SP-RL projector ---
export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_BETA_ACTOR="0"

# --- Sauté + PID (same as R21) ---
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"
export CITYLEARN_PID_LAGRANGE="1"

# --- Optimized 8-term reward (same as SP-RL run) ---
export STEMS_ALPHA_GRID="0.0"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_ALPHA_LOAD_SHIFT="0.15"
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
export STEMS_ALPHA_V2G_CONTEXT="0.0"
export STEMS_ALPHA_EV_SOLAR="0.0"
export STEMS_ALPHA_SOLAR_STORE="0.0"
export STEMS_SOLAR_STORE_BATT_ONLY="0"
export STEMS_ALPHA_HEADROOM="0.0"
export STEMS_ALPHA_PRICE_ARB="0.0"
export STEMS_ALPHA_GRID_PENALTY="0.1"
export STEMS_ALPHA_NEC_SIGN="0.0"
export STEMS_EV_SLACK_ARB_SCALE="0.0"

# --- Cost weights (same as R21) ---
export COST_W_C2="0.0"
export COST_W_C3="5.0"
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# --- Safety clamps (BATT_CLAMP off — projection handles C2) ---
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="0"
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"
export CITYLEARN_STEMS_BATTERY_COST_SCALE="1.0"

RUN_DIR="runs/serl_5bld_s${SEED}"

echo ""
echo "============================================"
echo "  SE-RL Paper-Exact 5-building (seed=$SEED)"
echo "  Projection: clip (Eq. 12)"
echo "  Penalty: w=0.5 (Eq. 23-24)"
echo "  C2/C3/C4 projected, C0/C1 Lagrangian"
echo "  Reward: optimized 8-term"
echo "  90 epochs, PPOLagMulti + PID"
echo "============================================"
echo ""

mkdir -p "$RUN_DIR"

# Create temp config with correct log_dir and seed
TMP_CFG="/tmp/serl_5bld.yaml"
sed "s/^seed: 42/seed: ${SEED}/" configs/on-policy/ablation_mask.yaml | \
    sed "s|log_dir: .*|log_dir: ./${RUN_DIR}/5bld|" > "$TMP_CFG"
/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_multi_lag.py \
    --cfg /tmp/serl_5bld.yaml \
    2>&1 | tee "$RUN_DIR/full_log.txt"
