#!/usr/bin/env bash
# R6 Training: Smart C3 Agent
# Combines: P0 (spatial obs) + P1 (controllable C3) + P2 (StopIter + sparse Saute)
#
# Usage: bash train_r6_smart_c3.sh [SEED]
set -euo pipefail

SEED="${1:-0}"
PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# --- Dataset ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
export CITYLEARN_CENTRAL_AGENT="1"

# --- Reward ---
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# --- Constraint thresholds (calibrated, unchanged from R5) ---
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="29.6915"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# --- Cost weights (unchanged from R5) ---
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

# --- EV ---
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# ============================================
# NEW FOR R6
# ============================================
export CITYLEARN_C3_CONTROLLABLE="1"    # P1: agent-controllable C3
export CITYLEARN_SPATIAL_OBS="1"        # P0: spatial observations (+N*4 dims)

echo "=== R6 Smart C3 Training ==="
echo "  P0: Spatial obs = $CITYLEARN_SPATIAL_OBS"
echo "  P1: Controllable C3 = $CITYLEARN_C3_CONTROLLABLE"
echo "  P2: StopIter + Sparse Saute (in config)"
echo "  Seed: $SEED"
echo "  Schema: $CITYLEARN_SCHEMA"
echo ""

# NOTE: Replace the python command below with your actual training invocation.
# The multilag PPOLag training script and config from R5-A/R5-B should be used here,
# with the R5-A StopIter fix (train_iters > 1) and R5-B sparse Saute fix combined.
#
# Example (adjust to your actual training script):
# python train_omnisafe_multilag.py \
#     --algo PPOLag \
#     --env CityLearnSafety-V2G-v2-multilag \
#     --seed $SEED \
#     --epochs 200 \
#     --tag "r6_smart_c3_seed${SEED}"

echo "TODO: Add your training command here (see comments above)"
echo "Env vars are set. You can also source this script and run manually."
