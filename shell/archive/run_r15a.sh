#!/usr/bin/env bash
# R15a: Hard EV action clamp + anti-discharge penalty
# Goal: Eliminate V2G discharge exploit (99.8% EV departure deficit)
#
# Changes from R14:
#   - Hard EV action clamp: prevents discharge when SoC < required + 0.1
#   - Anti-discharge penalty (alpha=5.0): gradient for policy learning
#
# Inherited from R14:
#   - PID Lagrangian (per-constraint gains)
#   - Fix 1: r_sb asymmetric
#   - Fix 3: lambda_ev=2.0
#   - Fix 4: sg_export_credit=0.5
#   - Sauté MDP C1
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

# --- PID Lagrangian (from R14) ---
export CITYLEARN_PID_LAGRANGE="1"

# --- Sauté MDP for C1 (fixed: budget 1500→25000, shaped penalty) ---
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

# --- STEMS weights (from R14) ---
export STEMS_MU_ECONOMIC="1.5"
export STEMS_ALPHA_GRID="3.0"
export STEMS_ALPHA_BUILD="1.0"
export STEMS_BETA_RAMP="1.0"
export STEMS_XI_RENEWABLE="0.2"

# --- Fix 3: Dense EV charging reward ---
export STEMS_LAMBDA_EV="5.0"

# --- Fix 1: Building stability asymmetric ---
export STEMS_SB_ASYMMETRIC="1"

# --- Fix 4: Grid stability export credit ---
export STEMS_SG_EXPORT_CREDIT="0.5"

# SoC barrier reward
export STEMS_ALPHA_BARRIER="0.5"

# C2 stays OFF
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

# No spatial obs, no temporal
export CITYLEARN_C3_CONTROLLABLE="0"
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"

# --- R15a: Hard EV action clamp + anti-discharge penalty ---
export CITYLEARN_EV_ACTION_CLAMP="1"
export CITYLEARN_EV_CLAMP_MARGIN="0.1"
export STEMS_ALPHA_EV_GUARD="5.0"

echo ""
echo "============================================"
echo "  R15a: Hard EV Clamp + Anti-Discharge Guard"
echo "  Changes from R14:"
echo "    - Hard EV action clamp (margin=0.1)"
echo "    - Anti-discharge penalty (alpha=5.0)"
echo "  Inherited from R14:"
echo "    - PID Lagrangian (per-constraint gains)"
echo "    - Fix 1: r_sb asymmetric"
echo "    - Fix 3: lambda_ev=2.0"
echo "    - Fix 4: sg_export_credit=0.5"
echo "    - Sauté MDP C1 (d=25000, shaped alpha=10)"
echo "============================================"
echo ""

python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r15a_clamp.yaml
