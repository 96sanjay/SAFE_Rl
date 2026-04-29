#!/usr/bin/env bash
# Structural conflict test:
# Same env/reward recipe as r25b_report_stable, but C4 is relaxed only.
set -euo pipefail

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

export CITYLEARN_PID_LAGRANGE="1"

export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

export STEMS_ALPHA_GRID="0.0"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_ALPHA_LOAD_SHIFT="0.0"
export STEMS_ALPHA_GRID_MILD="0.3"
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_XI_RENEWABLE="0.2"
export STEMS_BETA_RAMP="0.3"

export CITYLEARN_C3_CONTROLLABLE="1"

export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_GUARD="1.0"
export STEMS_ALPHA_V2G_CONTEXT="3.0"
export STEMS_EV_SLACK_ARB_SCALE="2.0"

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

export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="0"
export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_POLICY_ACTION_MASK="0"

echo ""
echo "============================================"
echo "  R25b Report-Stable C4-Relaxed"
echo "  Same as report-stable, only C4 limit relaxed"
echo "  Purpose: test C0/C1 vs C4 structural conflict"
echo "============================================"
echo ""

/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r25b_report_stable_c4_relaxed.yaml
