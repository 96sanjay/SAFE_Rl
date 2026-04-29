#!/usr/bin/env bash
# R12b: PPOLagMulti + Sauté MDP C1 + C2 re-enabled + C3 tightened
# Full combined improvement over R11b
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

# --- Inherited from R11b ---
export STEMS_LAMBDA_EV="0.0"         # r_ev disabled (C1 handles EV)
export STEMS_ALPHA_BARRIER="0.5"     # SoC barrier reward

# --- R12b CHANGES ---
# Sauté MDP for C1 (Sootla et al., ICML 2022)
export CITYLEARN_EV_SAUTE="1"                       # Proper Sauté MDP
export CITYLEARN_EV_SAUTE_BUDGET="1500"              # Budget d
export CITYLEARN_EV_SAUTE_PENALTY="5.0"              # Reward penalty when exhausted
export CITYLEARN_EV_SAUTE_GAMMA="1.0"                # Budget discount
export CITYLEARN_BATT_SOC_UPPER_CLAMP="0.88"         # Lower clamp (was 0.94)
export COST_W_C2="1.0"                                # C2 re-enabled (was 0.0)

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

echo ""
echo "============================================"
echo "  R12b: PPOLagMulti + Full Combined Fix"
echo "  Changes from R11b:"
echo "    - Proper Sauté MDP for C1"
echo "    - CITYLEARN_BATT_SOC_UPPER_CLAMP=0.88"
echo "    - COST_W_C2=1.0 (C2 re-enabled)"
echo "    - cost_limit_2=2000 (tightened)"
echo "    - cost_limit_3=10000 (tightened)"
echo "============================================"
echo ""

python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r12b_full.yaml
