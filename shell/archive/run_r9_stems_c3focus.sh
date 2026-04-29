#!/usr/bin/env bash
# R9: Intelligent STEMS — Rebalanced cost weights + ramping penalty + fixed Lagrangian
#
# ROOT CAUSE: omni_env_v2.py (CMDPv2) is the actual training wrapper.
# Key issue: COST_W_C3=0.1 (default) is 50x too low → agent ignores C3 violations.
#
# Changes from R8 (all via CMDPv2 env vars):
#   1. COST_W_C3: 0.1 → 5.0  (50x increase! Forces agent to care about building power)
#   2. STEMS_BETA_RAMP: 0.5 → 1.5  (3x increase, penalizes temporal degeneration)
#   3. Lagrangian: cost_limit 24400→150000, lambda_init 10→1, lambda_lr 0.05→0.005
#      (lambda stays bounded, reward signal visible throughout training)
#
# CMDPv2 defaults kept (already good):
#   STEMS_MU_ECONOMIC=0.3, STEMS_ALPHA_GRID=3.0, STEMS_ALPHA_BUILD=2.0
#   STEMS_XI_RENEWABLE=0.2, STEMS_LAMBDA_EV=5.0
#   COST_W_C1=10.0, COST_W_C1_DENSE=5.0, COST_W_C2=1.0, COST_W_C4=5.0
#
# Usage:
#   bash run_r9_stems_c3focus.sh              # full 50 epochs
#   bash run_r9_stems_c3focus.sh --smoke      # 1 epoch smoke test
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# --- Base env vars (same as R8) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Thresholds (same as R8)
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# --- R9 CMDPv2 reward weight (CHANGED) ---
export STEMS_BETA_RAMP="1.5"          # was 0.5 (3x ramping penalty)

# --- R9 CMDPv2 cost weight (CHANGED — this is the critical fix) ---
export COST_W_C3="5.0"                # was 0.1 (50x! building power constraint)

# Base env cost weights (for safety_env_v3 info dict computation)
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

# EV settings (same as R8)
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# NO spatial obs, NO controllable C3 (matching R8)
export CITYLEARN_C3_CONTROLLABLE="0"
export CITYLEARN_SPATIAL_OBS="0"

SMOKE_FLAG=""
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE_FLAG="--smoke_1epoch"
fi

echo ""
echo "=========================================="
echo "  R9: Intelligent STEMS — C3 Focus"
echo "  5 buildings, 50 epochs"
echo "  Key changes from R8 (CMDPv2 wrapper):"
echo "    COST_W_C3:     0.1 → 5.0  (50x increase!)"
echo "    STEMS_BETA_RAMP: 0.5 → 1.5  (3x ramping)"
echo "    cost_limit:    24400 → 150000"
echo "    lambda_init:   10 → 1"
echo "    lambda_lr:     0.05 → 0.005"
if [[ -n "$SMOKE_FLAG" ]]; then
    echo "  *** SMOKE TEST (1 epoch) ***"
fi
echo "=========================================="
echo ""

python scripts/train_stems_5bld.py \
    --cfg configs/on-policy/r9_stems_c3focus.yaml \
    $SMOKE_FLAG
