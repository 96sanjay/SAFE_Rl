#!/usr/bin/env bash
# GradS v2: Fixed multi-Lagrangian with per-constraint gradient shaping
#
# Fixes from v1:
#   1. CRITICAL: Sign bug fixed (was subtracting cost gradient, now adding)
#   2. Per-constraint limits tightened (sum went from 393K to 70K)
#   3. lambda_lr=0.0001 with upper_bound=50 (prevents explosion)
#   4. conflict_threshold=0.5 (enables real conflict detection)
#
# Usage:
#   bash run_grads_v2.sh              # full 50 epochs
#   bash run_grads_v2.sh --smoke      # 1 epoch smoke test
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# --- Shared env vars (5-building schema, same as R6/R7/R8) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Thresholds (same as all other runs)
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# Cost weights (default CMDPv2 weights — GradS handles per-constraint balancing)
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

# CMDP cost weights for GradS sampling
export COST_W_C1="10.0"
export COST_W_C1_DENSE="5.0"
export COST_W_C2="1.0"
export COST_W_C3="0.1"
export COST_W_C4="5.0"

# EV
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"
export CITYLEARN_EV_DENSE_COST_SCALE="1.0"

# P0 + P1 features (same as GradS v1)
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_SPATIAL_OBS="1"

# STEMS reward weights (same as v1)
export STEMS_ALPHA_GRID="1.0"
export STEMS_BETA_RAMP="2.0"

SMOKE_FLAG=""
CFG="configs/on-policy/ppolag_grads_5bld_v2.yaml"
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE_FLAG="--smoke_1epoch"
    # For smoke test, override total_steps in a temp config
    echo "*** SMOKE TEST MODE ***"
fi

echo ""
echo "==========================================="
echo "  GradS v2: Fixed Multi-Lagrangian"
echo "  5 buildings, 50 epochs"
echo "  Fixes: sign bug, tight limits, lambda cap"
echo "  Per-constraint limits:"
echo "    C1=50, C1d=300, C2=35K, C3=35K, C4=35K"
echo "  lambda_lr=0.0001, upper_bound=50"
echo "  conflict_threshold=0.5"
echo "==========================================="
echo ""

python scripts/train_grads.py --cfg "$CFG"
