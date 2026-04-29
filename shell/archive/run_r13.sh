#!/usr/bin/env bash
# R13: Sauté MDP + V2G-aware reward structure
# Goal: Agent learns SMART V2G policy (charge solar/cheap, discharge peak)
#
# 4 reward fixes over R12a:
#   Fix 1: r_sb asymmetric — exports not penalized (STEMS_SB_ASYMMETRIC=1)
#   Fix 2: Weight rebalance — economic signal competitive with stability
#   Fix 3: Dense EV shaping — urgency×shortfall gradient (STEMS_LAMBDA_EV=2.0)
#   Fix 4: r_sg export credit — V2G exports reduce grid penalty (STEMS_SG_EXPORT_CREDIT=0.5)
#
# + Sauté MDP for C1 (from R12a): budget state augmentation + reward reshaping
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

# --- Sauté MDP for C1 (from R12a, unchanged) ---
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="1500"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"

# --- Fix 2: Rebalanced STEMS weights (V2G-aware) ---
# Economic signal 5x stronger so price-responsive behavior can emerge
export STEMS_MU_ECONOMIC="1.5"
# alpha_grid kept at 3.0 (critic analysis: reducing to 2.0 weakens C3 enforcement)
# V2G incentive comes from r_eco + asymmetric r_sb, NOT from weakening grid penalty
export STEMS_ALPHA_GRID="3.0"
export STEMS_ALPHA_BUILD="1.0"
export STEMS_BETA_RAMP="1.0"
export STEMS_XI_RENEWABLE="0.2"

# --- Fix 3: Dense EV charging reward (complementary to Sauté) ---
# Provides per-step urgency×shortfall gradient within safe regime
export STEMS_LAMBDA_EV="2.0"

# --- Fix 1: Building stability only penalizes imports, not exports ---
# Enables V2G: exporting from a building is NOT penalized
export STEMS_SB_ASYMMETRIC="1"

# --- Fix 4: Grid stability export credit ---
# V2G exports get 50% credit in grid penalty calculation
# Makes charge-discharge cycle profitable even without solar
export STEMS_SG_EXPORT_CREDIT="0.5"

# SoC barrier reward (unchanged from R12a)
export STEMS_ALPHA_BARRIER="0.5"

# C2 stays OFF (isolate V2G reward effect vs R12a)
export COST_W_C2="0.0"

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
echo "  R13: Sauté MDP + V2G-Aware Reward"
echo "  Changes from R12a:"
echo "    - Fix 1: r_sb asymmetric (exports not penalized)"
echo "    - Fix 2: mu=1.5, grid=3.0, build=1.0, ramp=1.0"
echo "    - Fix 3: lambda_ev=2.0 (dense EV shaping)"
echo "    - Fix 4: sg_export_credit=0.5 (V2G grid credit)"
echo "    - Sauté MDP for C1 (unchanged from R12a)"
echo "============================================"
echo ""

python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r13_v2g_smart.yaml
