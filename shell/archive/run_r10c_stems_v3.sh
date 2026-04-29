#!/usr/bin/env bash
# R10c: STEMS v3 — Lambda capping fix
# Only change from R10b: lagrangian_upper_bound: 1.0
# Everything else identical to isolate the effect of lambda capping
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Base env vars ---
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

# --- STEMS v3 encoder with temporal ---
export STEMS_ENCODER_VERSION="v3"
export CITYLEARN_TEMPORAL_WINDOW="12"

# --- CMDPv2 reward weights ---
export STEMS_BETA_RAMP="1.5"

# --- CMDPv2 cost weights ---
export COST_W_C3="5.0"

# Base env cost weights
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

# EV settings
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# NO spatial obs, NO controllable C3
export CITYLEARN_C3_CONTROLLABLE="0"
export CITYLEARN_SPATIAL_OBS="0"

# --- R10c fix: cap lambda at 1.0 ---
export LAMBDA_UPPER_BOUND="1.0"

SMOKE_FLAG=""
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE_FLAG="--smoke_1epoch"
fi

echo ""
echo "=========================================="
echo "  R10c: STEMS v3 — Lambda Capping Fix"
echo "  5 buildings, 50 epochs"
echo "  Only change from R10b:"
echo "    lagrangian_upper_bound: 1.0"
if [[ -n "$SMOKE_FLAG" ]]; then
    echo "  *** SMOKE TEST (1 epoch) ***"
fi
echo "=========================================="
echo ""

python scripts/train_stems_5bld.py \
    --cfg configs/on-policy/r10c_stems_v3.yaml \
    $SMOKE_FLAG
