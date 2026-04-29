#!/usr/bin/env bash
# R8: STEMS GCN-Transformer ablation on 5 buildings
#
# Identical to R5a (MLP baseline) except the policy architecture:
#   R5a uses default OmniSafe MLP [256,256]
#   R8  uses STEMS GCN-Transformer (GCN + Temporal Attention + Gated Fusion)
#
# Same env vars, same cost weights, same Lagrange params, same LR.
# This isolates the architecture's contribution.
#
# Usage:
#   bash run_r8_stems_ablation.sh              # full 50 epochs
#   bash run_r8_stems_ablation.sh --smoke      # 1 epoch smoke test
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# --- Shared env vars (IDENTICAL to R5a in run_r6_comparison_5bld.sh) ---
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

# Cost weights (same as R5a)
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

# EV
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# NO spatial obs, NO controllable C3 (matching R5a baseline)
export CITYLEARN_C3_CONTROLLABLE="0"
export CITYLEARN_SPATIAL_OBS="0"

SMOKE_FLAG=""
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE_FLAG="--smoke_1epoch"
fi

echo ""
echo "=========================================="
echo "  R8: STEMS GCN-Transformer Ablation"
echo "  5 buildings, 50 epochs"
echo "  Architecture: GCN + Temporal Transformer + Gated Fusion"
echo "  All other params identical to R5a"
if [[ -n "$SMOKE_FLAG" ]]; then
    echo "  *** SMOKE TEST (1 epoch) ***"
fi
echo "=========================================="
echo ""

python scripts/train_stems_5bld.py \
    --cfg configs/on-policy/r8_stems_5bld.yaml \
    $SMOKE_FLAG
