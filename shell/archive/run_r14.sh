#!/usr/bin/env bash
# R14: PID Lagrangian + V2G-aware reward + Sauté MDP
# Goal: Stable lambda dynamics (no integral windup) + smart V2G policy
#
# Changes from R13:
#   - PID Lagrangian (Stooke et al., ICML 2020) replaces SGD for lambda updates
#   - Cost-limit normalized delta → scale-invariant gains across all 5 constraints
#   - D-term brakes lambda when costs are DECREASING (prevents StopIter oscillation)
#
# Inherited from R13:
#   - Fix 1: r_sb asymmetric (exports not penalized)
#   - Fix 2: Weight rebalance (mu=1.5, grid=2.0, build=1.0, ramp=1.0)
#   - Fix 3: Dense EV shaping (lambda_ev=2.0)
#   - Fix 4: r_sg export credit (0.5)
#   - Sauté MDP for C1
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

# --- PID Lagrangian (NEW in R14) ---
# Replaces SGD lambda update with PID controller
# Prevents integral windup that caused StopIter 30/1 oscillation in R12a
export CITYLEARN_PID_LAGRANGE="1"

# --- Sauté MDP for C1 (from R12a, unchanged) ---
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="1500"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"

# --- Fix 2: Rebalanced STEMS weights (V2G-aware) ---
# mu_economic boosted to make price-responsive behavior competitive
export STEMS_MU_ECONOMIC="1.5"
# alpha_grid KEPT at 3.0 (R13 had 2.0 but critic found this weakens C3 enforcement)
# V2G incentive comes from r_eco + asymmetric r_sb, NOT from weakening grid penalty
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

# No spatial obs, no temporal (MLP baseline)
export CITYLEARN_C3_CONTROLLABLE="0"
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"

echo ""
echo "============================================"
echo "  R14: PID Lagrangian + V2G-Aware Reward"
echo "  Changes from R13:"
echo "    - PID Lagrangian (per-constraint gains)"
echo "    - C0: pure proportional (Kp=5.0, no integral)"
echo "    - C3: 3x gains (Kp=0.3, Ki=0.03)"
echo "    - alpha_grid=3.0 (kept strong, not 2.0)"
echo "  Inherited from R13:"
echo "    - Fix 1: r_sb asymmetric"
echo "    - Fix 3: lambda_ev=2.0"
echo "    - Fix 4: sg_export_credit=0.5"
echo "    - Sauté MDP C1"
echo "============================================"
echo ""

python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r14_pid_v2g.yaml
