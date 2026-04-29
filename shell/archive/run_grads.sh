#!/usr/bin/env bash
# PPOLag + GradS: Per-constraint lambdas (5x) with Gradient Shaping
#
# Same env vars as R6/R7 (P0 spatial obs + P1 controllable C3)
# Adds GradS gradient selection with 5 per-constraint cost critics + lambdas
#
# Usage: bash run_grads.sh
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# --- Shared env vars (5-building schema, same as R6/R7) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Thresholds (same as R6/R7)
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# Cost weights (same as R6/R7)
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

# CMDP cost weights for GradS sampling (read by ppo_lag_grads.py)
# Must match omni_env_v2.py weights so GradS constraints decompose the aggregate cost
export COST_W_C1="10.0"
export COST_W_C1_DENSE="5.0"
export COST_W_C2="1.0"
export COST_W_C3="0.1"
export COST_W_C4="5.0"

# EV (same as R6/R7)
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"
export CITYLEARN_EV_DENSE_COST_SCALE="1.0"    # activate C1d dense EV charging signal

# P0 + P1 features (same as R6/R7)
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_SPATIAL_OBS="1"

# STEMS reward weights (use R7 tuned values)
export STEMS_ALPHA_GRID="1.0"
export STEMS_BETA_RAMP="2.0"

# GradS thresholds (optional, defaults are from the paper)
export GRADS_SIM_THRESHOLD="0.8"
export GRADS_CONFLICT_THRESHOLD="0.999"

echo ""
echo "==========================================="
echo "  PPOLag + GradS (5 buildings, 50 epochs)"
echo "  P0=spatial obs, P1=controllable C3"
echo "  R7 reward tuning (alpha_grid=1, beta_ramp=2)"
echo "  GradS: 5 per-constraint cost critics + lambdas"
echo "  Per-constraint lambdas (5x), gradient shaping"
echo "==========================================="
echo ""

python scripts/train_grads.py --cfg configs/on-policy/ppolag_grads_5bld.yaml
