#!/usr/bin/env bash
# R11b: PPOLagMulti (per-constraint lambdas) + Step 1 improvements
# A/B comparison Run B
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Base env ---
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

# --- Step 1 improvements ---
export STEMS_LAMBDA_EV="0.0"         # r_ev disabled (C1 handles EV)
export STEMS_ALPHA_BARRIER="0.5"     # SoC barrier reward
export COST_W_C2="0.0"              # C2 zeroed (clamp+barrier handle SoC)

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

# No spatial obs, no temporal (MLP baseline)
export CITYLEARN_C3_CONTROLLABLE="0"
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"

SMOKE_FLAG=""
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE_FLAG="--smoke_1epoch"
    # Override total_steps for smoke test
fi

echo ""
echo "=========================================="
echo "  R11b: PPOLagMulti + Step 1 Improvements"
echo "  Per-constraint lambdas, MLP, 50 epochs"
echo "  Changes from multi v5:"
echo "    - r_ev removed (STEMS_LAMBDA_EV=0)"
echo "    - SoC safety clamp (in env)"
echo "    - SoC barrier reward (alpha=0.5)"
echo "    - C2 zeroed"
echo "    - Cost adv clipping (±2×std(adv_r))"
echo "    - Hyperparams matched to Run A"
if [[ -n "$SMOKE_FLAG" ]]; then
    echo "  *** SMOKE TEST (1 epoch) ***"
fi
echo "=========================================="
echo ""

python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r11b_multi_improved.yaml
