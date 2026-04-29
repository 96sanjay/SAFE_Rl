#!/bin/bash
set -e
PROJECT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Schema
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"

# Building/grid thresholds (same as all runs)
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Cost weights (same as Run C)
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"
export CITYLEARN_C3_CONTROLLABLE="1"
export COST_W_C2="0.0"
export COST_W_C3="5.0"

# Clamps OFF (same as all runs)
export CITYLEARN_BATT_CLAMP="0"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"

# NO Saute (Saute OFF for runs 0-3)
export CITYLEARN_EV_SAUTE="0"

# NO PID (single lambda uses standard OmniSafe)
# NO action mask
export CITYLEARN_ACTION_MASK="0"

# ===== OLD STEMS REWARD (all original terms enabled) =====
export STEMS_MU_ECONOMIC="1.0"         # ORIGINAL — discharge exploit
export STEMS_ALPHA_GRID="0.5"          # ORIGINAL — discharge exploit
export STEMS_ALPHA_BUILD="0.3"         # ORIGINAL — discharge exploit
export STEMS_BETA_RAMP="0.2"           # ORIGINAL
export STEMS_XI_RENEWABLE="0.6"        # ORIGINAL
export STEMS_SB_ASYMMETRIC="0"         # ORIGINAL (symmetric, not asymmetric)
export STEMS_SG_EXPORT_CREDIT="0.0"    # ORIGINAL (no export credit)
export STEMS_SG_THRESHOLD="0.0"        # ORIGINAL (no threshold)

# NO custom EV terms
export STEMS_LAMBDA_EV="0.0"
export STEMS_ALPHA_EV_GUARD="0.0"
export STEMS_ALPHA_V2G_CONTEXT="0.0"
export STEMS_EV_SLACK_ARB_SCALE="0.0"
export STEMS_ALPHA_BARRIER="0.0"
export STEMS_ALPHA_GRID_MILD="0.0"
export STEMS_ALPHA_LOAD_SHIFT="0.0"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_SOLAR="0.0"
export STEMS_ALPHA_SOLAR_STORE="0.0"
export STEMS_ALPHA_HEADROOM="0.0"
export STEMS_ALPHA_PRICE_ARB="0.0"
export STEMS_ALPHA_NEC_SIGN="0.0"
export STEMS_ALPHA_GRID_PENALTY="0.0"

echo "=============================="
echo "  ABLATION RUN 0: Single-Lambda + OLD STEMS"
echo "  Saute=OFF, C2=OFF, PID=NO"
echo "=============================="

/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_omnisafe.py \
    --cfg configs/on-policy/ablation_0.yaml
