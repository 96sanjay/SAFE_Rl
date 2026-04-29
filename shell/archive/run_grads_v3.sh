#!/usr/bin/env bash
# GradS v3: Balanced gradients + proper sampling
#
# Fixes from v2 (50-epoch run, StopIter=1 throughout, reward collapsed -49K→-105K):
#   1. Z-score cost advantages (was only mean-centered → 9x gradient imbalance)
#   2. Gradient normalization (cost grad rescaled to reward grad magnitude)
#   3. sampling=lambda (softmax(λ_i) not softmax(λ_i/limit_i) — v2 skewed by 700x limit range)
#   4. actor LR 0.0005→0.0002 (more conservative)
#   5. max_grad_norm 40→5 (tighter clip with balanced grads)
#   6. lambda upper_bound 50→10 (balanced grads need less lambda pressure)
#   7. target_kl 0.10→0.15 (allow more updates per epoch)
#
# Usage:
#   bash run_grads_v3.sh              # full 50 epochs
#   bash run_grads_v3.sh --smoke      # 1 epoch smoke test
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate citylearn

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# --- Shared env vars (5-building schema, same as all GradS runs) ---
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

# P0 + P1 features (same as v1/v2)
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_SPATIAL_OBS="1"

# STEMS reward weights (same as v1/v2)
export STEMS_ALPHA_GRID="1.0"
export STEMS_BETA_RAMP="2.0"

CFG="configs/on-policy/ppolag_grads_5bld_v3.yaml"

echo ""
echo "==========================================="
echo "  GradS v3: Balanced Gradients"
echo "  5 buildings, 50 epochs"
echo "  Fixes: z-score cost adv, grad norm,"
echo "    sampling=lambda, LR=0.0002, clip=5"
echo "  Per-constraint limits:"
echo "    C1=50, C1d=300, C2=35K, C3=35K, C4=35K"
echo "  lambda_lr=0.0001, upper_bound=10"
echo "  target_kl=0.15, conflict_threshold=0.5"
echo "==========================================="
echo ""

python scripts/train_grads.py --cfg "$CFG"
