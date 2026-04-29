#!/usr/bin/env bash
# 1-Building 3-Month Cycling Test
# Purpose: Test if NEC-sign reward + action mask produces cycling behavior
# Expected runtime: ~15-20 min (50 epochs x 2190 steps)
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Schema (1 building, 3 months) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_1building_3month.json"

# --- Action mask ---
export CITYLEARN_ACTION_MASK="1"

# --- KL regularization (anchor to BC policy) ---
export CITYLEARN_KL_BETA="0.1"
export CITYLEARN_KL_BETA_DECAY="0.995"

# --- NEC-sign reward (R30c weights exactly) ---
export STEMS_ALPHA_NEC_SIGN="3.0"
export STEMS_ALPHA_PRICE_ARB="1.0"
export STEMS_LAMBDA_EV="15.0"
export STEMS_EV_SLACK_ARB_SCALE="2.5"
export STEMS_ALPHA_GRID_PENALTY="1.5"
export STEMS_ALPHA_GRID_PENALTY_SOLAR="0.0"
export CITYLEARN_STEMS_BATTERY_COST_SCALE="1.0"

# --- All other STEMS terms = 0.0 ---
export STEMS_ALPHA_NEC="0.0"
export STEMS_ALPHA_PEAK="0.0"
export STEMS_ALPHA_RAMP="0.0"
export STEMS_ALPHA_CARBON="0.0"
export STEMS_ALPHA_LOAD="0.0"

# --- C1 tolerance (matches SmartV2GRBC headroom) ---
export CITYLEARN_C1_SOC_TOLERANCE="0.20"

echo "============================================="
echo "  1-Building 3-Month Cycling Test"
echo "  Schema: $CITYLEARN_SCHEMA"
echo "  Action mask: ON"
echo "  KL beta: $CITYLEARN_KL_BETA"
echo "  NEC-sign alpha: $STEMS_ALPHA_NEC_SIGN"
echo "  50 epochs x 2190 steps = 109,500 total"
echo "============================================="

/home/christmas/miniconda3/envs/citylearn/bin/python \
    scripts/train_multi_lag.py \
    --cfg configs/on-policy/test_1bld.yaml \
    --bc

echo "Done. Check runs/test_1bld/ for results."
