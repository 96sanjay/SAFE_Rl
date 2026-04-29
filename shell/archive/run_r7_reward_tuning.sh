#!/usr/bin/env bash
# R7: Fix myopic discharge exploit via reward tuning
#
# Identical to R6 (P0 spatial obs + P1 controllable C3) but with:
#   STEMS_ALPHA_GRID  3.0 → 1.0  (reduce dominant grid-import incentive)
#   STEMS_BETA_RAMP   0.5 → 2.0  (penalize discharge-then-spike pattern)
#
# Usage: bash run_r7_reward_tuning.sh
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# --- Shared env vars (5-building schema, same as R6) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Thresholds (same as R6)
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# Cost weights (same as R6)
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

# EV (same as R6)
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# P0 + P1 features (same as R6)
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_SPATIAL_OBS="1"

# === R7 REWARD TUNING (only changes from R6) ===
# R6 defaults: STEMS_ALPHA_GRID=3.0, STEMS_BETA_RAMP=0.5
# R7 tuned:    STEMS_ALPHA_GRID=1.0, STEMS_BETA_RAMP=2.0
export STEMS_ALPHA_GRID="1.0"
export STEMS_BETA_RAMP="2.0"

echo ""
echo "=========================================="
echo "  R7: Reward Tuning (5 buildings, 50 ep)"
echo "  P0=spatial obs, P1=controllable C3"
echo "  STEMS_ALPHA_GRID=1.0 (was 3.0)"
echo "  STEMS_BETA_RAMP=2.0  (was 0.5)"
echo "  All other params identical to R6"
echo "=========================================="
echo ""

python scripts/train_omnisafe.py --cfg configs/on-policy/r6_compare_r7_5bld.yaml
