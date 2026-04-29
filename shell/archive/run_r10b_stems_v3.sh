#!/usr/bin/env bash
# R10b: STEMS v3b — Fixed temporal bypass + stable Lagrangian
#
# Fixes from R10:
#   1. Encoder: concat+proj replaces residual addition (forces temporal usage)
#   2. cost_limit: 600K (matches epoch-0 cost), lambda_lr: 0.0005 (slower growth)
#   3. update_iters: 30 (halves update time from 31min to ~15min)
#   4. target_kl: 0.12 (more room per step)
#
# Same reward/cost weights as R9/R10.
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

SMOKE_FLAG=""
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE_FLAG="--smoke_1epoch"
fi

echo ""
echo "=========================================="
echo "  R10b: STEMS v3b — Fixed Temporal + Stable Lambda"
echo "  5 buildings, 50 epochs"
echo "  Fixes from R10:"
echo "    Temporal: residual → concat+proj (forces temporal usage)"
echo "    cost_limit: 300K → 600K"
echo "    lambda_lr: 0.001 → 0.0005"
echo "    update_iters: 60 → 30"
echo "    target_kl: 0.10 → 0.12"
if [[ -n "$SMOKE_FLAG" ]]; then
    echo "  *** SMOKE TEST (1 epoch) ***"
fi
echo "=========================================="
echo ""

python scripts/train_stems_5bld.py \
    --cfg configs/on-policy/r10b_stems_v3.yaml \
    $SMOKE_FLAG
