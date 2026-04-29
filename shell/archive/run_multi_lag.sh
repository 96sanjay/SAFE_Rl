#!/usr/bin/env bash
# PPOLag-Multi: Per-constraint Lagrange + softmax advantage selection
#
# Key innovation: per-timestep softmax weighting focuses on most-violated
# constraint, preventing gradient cancellation between conflicting constraints.
#
# Hyperparameters match R5a (baseline, intelligence=0.76) with additions:
#   - 5 per-constraint lambdas (init=1.0, lr=0.035, uncapped)
#   - 5 per-constraint cost critics
#   - Softmax tau=1.0
#   - Z-scored per-constraint cost advantages
#
# Usage:
#   bash run_multi_lag.sh              # full 50 epochs
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate citylearn

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# --- Shared env vars (5-building schema, same as all runs) ---
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

# Cost weights (default CMDPv2 weights)
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

# CMDP cost weights for per-constraint tracking
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

# P0 + P1 features
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_SPATIAL_OBS="1"

# STEMS reward weights
export STEMS_ALPHA_GRID="1.0"
export STEMS_BETA_RAMP="2.0"

CFG="configs/on-policy/ppolag_multi_5bld.yaml"

echo ""
echo "==========================================="
echo "  PPOLag-Multi: Softmax Advantage Selection"
echo "  5 buildings, 50 epochs"
echo "  Per-constraint lambdas (uncapped)"
echo "  Softmax tau=1.0"
echo "  Cost limits: C0=500, C1=1200, C2=15K,"
echo "    C3=40K, C4=20K"
echo "  Lambda LR=0.035, init=1.0"
echo "  Actor LR=0.0005, update_iters=60"
echo "==========================================="
echo ""

python scripts/train_multi_lag.py --cfg "$CFG"
