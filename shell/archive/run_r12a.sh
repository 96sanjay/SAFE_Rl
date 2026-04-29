#!/usr/bin/env bash
# R12a: PPOLagMulti + Sauté MDP for C1 (EV charging)
# Ablation: Sauté C1 ON, C2 stays OFF (isolate Sauté effect)
# Proper Sauté: budget state augmentation + penalty switching
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

# --- Inherited from R11b (unchanged) ---
export STEMS_LAMBDA_EV="0.0"         # r_ev disabled (C1 handles EV)
export STEMS_ALPHA_BARRIER="0.5"     # SoC barrier reward
export COST_W_C2="0.0"              # C2 stays OFF (isolate C1 effect)

# --- R12a: Proper Sauté MDP for C1 (Sootla et al., ICML 2022) ---
# Eliminates C1 constraint from CMDP via state augmentation + reward reshaping.
# C1 Lagrangian sees cost=0 always; C0 (departure) remains separate Lagrangian.
export CITYLEARN_EV_SAUTE="1"              # Enable Sauté
export CITYLEARN_EV_SAUTE_BUDGET="1500"    # Budget d (= cost_limit_1)
export CITYLEARN_EV_SAUTE_PENALTY="5.0"    # Reward penalty when budget exhausted
export CITYLEARN_EV_SAUTE_GAMMA="1.0"      # Budget discount (1.0 = no discounting)

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
echo "=========================================="
echo "  R12a: PPOLagMulti + Sauté MDP (C1)"
echo "  Ablation: Sauté C1 ON, C2 OFF"
echo "  Changes from R11b:"
echo "    - Proper Sauté MDP (Sootla et al. ICML 2022)"
echo "    - C1 ELIMINATED from CMDP (cost=0, reward reshaping)"
echo "    - Obs augmented: +1 dim (budget state λ)"
echo "    - Budget d=1500, penalty=5.0, gamma=1.0"
echo "    - C0 (departure) still Lagrangian (separate)"
echo "    - Everything else identical to R11b"
echo "=========================================="
echo ""

python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r12a_saute_c1.yaml
