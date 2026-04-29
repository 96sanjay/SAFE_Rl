#!/usr/bin/env bash
# ABLATION: Prove C3/C4 Lagrangian kills battery cycling
# Runs two 15-epoch experiments with 3-month episodes (~45 min each):
#   1. CONTROL: R28 settings with C3/C4 active (limits 20K/15K)
#   2. ABLATION: Same but C3/C4 limits=999999 (lambdas stay ~0)
#
# If battery cycling persists in ablation but dies in control → PROOF
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- 3-MONTH SCHEMA ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings_3month.json"
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

# Saute MDP for C1
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

# Reward (same as R28)
export STEMS_ALPHA_GRID="0.0"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_ALPHA_LOAD_SHIFT="5.0"
export STEMS_ALPHA_GRID_MILD="0.0"
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_XI_RENEWABLE="0.0"
export STEMS_BETA_RAMP="0.3"

# V2G reward (same as R28)
export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_GUARD="1.0"
export STEMS_ALPHA_V2G_CONTEXT="0.0"
export STEMS_EV_SLACK_ARB_SCALE="2.0"
export STEMS_ALPHA_SOLAR_STORE="0.0"
export STEMS_SOLAR_STORE_BATT_ONLY="0"
export STEMS_ALPHA_EV_SOLAR="0.0"

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
export CITYLEARN_C3_CONTROLLABLE="1"

# Clamps OFF
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="0"

PYTHON="/home/christmas/miniconda3/envs/citylearn/bin/python"

echo ""
echo "============================================"
echo "  ABLATION: C3/C4 Lagrangian Impact"
echo "  3-month episodes (2190 steps), 15 epochs, TIGHT LIMITS"
echo "  Control: C3=5000, C4=3750 (scaled for 3-month)"
echo "  Ablation: C3=999999, C4=999999 (disabled)"
echo "============================================"

echo ""
echo "--- RUN 1/2: CONTROL (C3/C4 active) ---"
echo ""
$PYTHON scripts/train_multi_lag.py \
    --cfg configs/on-policy/ablation_c3c4_control.yaml 2>&1 | tee runs/ablation_c3c4_v2/control_log.txt

echo ""
echo "--- RUN 2/2: ABLATION (C3/C4 disabled) ---"
echo ""
$PYTHON scripts/train_multi_lag.py \
    --cfg configs/on-policy/ablation_c3c4_disabled.yaml 2>&1 | tee runs/ablation_c3c4_v2/disabled_log.txt

echo ""
echo "============================================"
echo "  ABLATION COMPLETE"
echo "  Compare: runs/ablation_c3c4_v2/control/ vs runs/ablation_c3c4_v2/disabled/"
echo "============================================"
