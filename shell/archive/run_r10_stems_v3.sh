#!/usr/bin/env bash
# R10: STEMS v3 — Per-Node Temporal Transformer + GCN + Gated Fusion
#
# Changes from R9:
#   - STEMS_ENCODER_VERSION=v3 (adds per-building temporal Transformer)
#   - CITYLEARN_TEMPORAL_WINDOW=12 (12h history embedded in observation)
#   - cost_normalize=true (fixes StopIter collapse)
#   - actor_lr=0.0003 (slightly lower for larger encoder)
#
# Same reward weights and cost weights as R9.
#
# Usage:
#   bash run_r10_stems_v3.sh              # full 50 epochs
#   bash run_r10_stems_v3.sh --smoke      # 1 epoch smoke test
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# --- Base env vars (same as R8/R9) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Thresholds (same as R8/R9)
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# --- R10: STEMS v3 encoder with temporal ---
export STEMS_ENCODER_VERSION="v3"
export CITYLEARN_TEMPORAL_WINDOW="12"

# --- CMDPv2 reward weights (same as R9) ---
export STEMS_BETA_RAMP="1.5"

# --- CMDPv2 cost weights (same as R9) ---
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
echo "  R10: STEMS v3 — Temporal Transformer"
echo "  5 buildings, 50 epochs"
echo "  Architecture: Per-node Temporal Transformer + GCN + Fusion"
echo "  Temporal window: 12 hours"
echo "  Key changes from R9:"
echo "    STEMS encoder:    v2 → v3 (temporal)"
echo "    Temporal window:  0 → 12"
echo "    cost_normalize:   false → true"
echo "    actor_lr:         0.0005 → 0.0003"
if [[ -n "$SMOKE_FLAG" ]]; then
    echo "  *** SMOKE TEST (1 epoch) ***"
fi
echo "=========================================="
echo ""

python scripts/train_stems_5bld.py \
    --cfg configs/on-policy/r10_stems_v3.yaml \
    $SMOKE_FLAG
