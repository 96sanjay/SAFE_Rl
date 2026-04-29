#!/usr/bin/env bash
# R17: Mathematically verified reward reform with threshold r_sg
#
# Mathematical proof (docs/r17_mathematical_proof.md) showed:
#   - Original R17 proposal FAILS: r_sg penalizes ALL imports, making charging
#     never optimal → batteries permanently depleted
#   - FIX: Threshold r_sg at 50% P_grid_max → creates 4-hour afternoon charging
#     window (hours 14-17) while preserving evening peak discharge incentive
#
# Key changes from R16b:
#   1. r_sg RESTORED at alpha=1.5 with threshold at 50% P_grid_max (was 0.0)
#   2. r_load_shift alpha=6.0 (was 3.0) — battery price arbitrage
#   3. r_ren xi=0.2 (was 1.5) — reverted, eliminates solar-charging conflict
#   4. Lambda cap raised to 6.0 (was 3.0) — compensates reduced implicit pressure
#   5. Keeps: C3_CONTROLLABLE, faster PID, all EV features
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

# --- PID Lagrangian ---
export CITYLEARN_PID_LAGRANGE="1"

# --- Sauté MDP for C1 ---
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

# =====================================================================
# R17: VERIFIED REWARD CONFIGURATION
# =====================================================================
# r_sg RESTORED with threshold (proof: charging never optimal without threshold)
export STEMS_ALPHA_GRID="1.5"
export STEMS_SG_THRESHOLD="0.5"       # Only penalize imports > 50% P_grid_max

# r_load_shift increased (proven >15% variance contribution at alpha=6.0)
export STEMS_ALPHA_LOAD_SHIFT="6.0"

# r_grid_mild kept for gentle linear awareness
export STEMS_ALPHA_GRID_MILD="0.3"

# r_eco and r_sb REMOVED (redundant with r_sg + r_load_shift)
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_BUILD="0.0"

# r_ren reverted from 1.5 → 0.2 (1.5 penalized solar charging by -0.125/step)
export STEMS_XI_RENEWABLE="0.2"

# r_ramp kept moderate
export STEMS_BETA_RAMP="0.3"

# C3_CONTROLLABLE (from R16b — battery exports don't cause false C3 violations)
export CITYLEARN_C3_CONTROLLABLE="1"

# EV features (from R15b)
export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export CITYLEARN_EV_ACTION_CLAMP="1"
export CITYLEARN_EV_CLAMP_MARGIN="0.1"
export STEMS_ALPHA_EV_GUARD="5.0"
export STEMS_ALPHA_V2G_CONTEXT="3.0"

export COST_W_C2="0.0"
export COST_W_C3="5.0"

export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"

echo ""
echo "============================================"
echo "  R17: Threshold r_sg + Verified Rewards"
echo "  Key changes from R16b:"
echo "    - r_sg=1.5 with threshold=0.5 P_grid_max"
echo "    - r_load_shift=6.0 (battery price arbitrage)"
echo "    - r_ren=0.2 (reverted from 1.5)"
echo "    - Lambda cap=6.0 (raised from 3.0)"
echo "    - Keeps: C3_CONTROLLABLE, faster PID, EV features"
echo "  Proof: docs/r17_mathematical_proof.md"
echo "============================================"
echo ""

python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r17_threshold_sg.yaml
