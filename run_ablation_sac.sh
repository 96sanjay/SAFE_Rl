#!/usr/bin/env bash
# Algorithm Ablation: SACLagMulti (off-policy) vs PPOLagMulti (R21 on-policy)
# Usage: ./run_ablation_sac.sh [seed]
#
# Reward terms IDENTICAL to R21 — only the RL algorithm differs.
# Baseline = R21 (PPOLagMulti, already trained).
# This script runs the SACLagMulti treatment.
set -euo pipefail

SEED="${1:-42}"

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Environment (IDENTICAL to R21) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# --- Thresholds (IDENTICAL to R21) ---
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# --- PID Lagrangian (IDENTICAL to R21) ---
export CITYLEARN_PID_LAGRANGE="1"

# --- Sauté MDP for C0 (IDENTICAL to R21) ---
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

# --- Reward terms (IDENTICAL to R21) ---
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

# --- Cost weights (IDENTICAL to R21) ---
export COST_W_C2="0.0"
export COST_W_C3="5.0"
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# --- Safety clamps (IDENTICAL to R21) ---
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="1"
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"

# --- NO action mask (same as R21 baseline) ---
export CITYLEARN_ACTION_MASK="0"

RUN_DIR="runs/ablation_sac_5bld_s${SEED}"

echo ""
echo "============================================"
echo "  ABLATION: SACLagMulti (seed=$SEED)"
echo "  Reward: IDENTICAL to R21"
echo "  Algorithm: SACLagMulti (off-policy)"
echo "  Baseline: R21 PPOLagMulti (on-policy)"
echo "  90 epochs, 5 buildings"
echo "============================================"
echo ""

# Create temp YAML with right seed and log_dir
TMP_CFG="/tmp/ablation_sac_5bld_s${SEED}.yaml"
sed "s/^seed: 42/seed: ${SEED}/" configs/off-policy/sac_lag_5bld_ablation.yaml | \
    sed "s|log_dir: .*|log_dir: ./${RUN_DIR}/5bld|" > "$TMP_CFG"

mkdir -p "$RUN_DIR"
/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_sac_lag.py \
    --cfg "$TMP_CFG" \
    2>&1 | tee "$RUN_DIR/full_log.txt"
