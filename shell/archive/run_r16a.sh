#!/usr/bin/env bash
# R16a: Reward Reform — battery price arbitrage + gentle grid awareness
# Goal: Give agent clear price intelligence while keeping EV compliance
#
# Changes from R15b:
#   - r_eco DISABLED (mu=0) — replaced by r_load_shift
#   - r_sg DISABLED (alpha_grid=0) — replaced by r_grid_mild
#   - r_sb DISABLED (alpha_build=0) — was time-blind, no useful signal
#   - r_load_shift=3.0: battery-only price arbitrage (charge cheap, discharge expensive)
#   - r_grid_mild=0.3: gentle linear grid import awareness
#   - r_ren xi=1.5 (was 0.2): stronger solar incentive
#   - r_ramp beta=0.3 (was 1.0): reduced ramp penalty
#
# Inherited from R15b:
#   - Hard EV action clamp (margin=0.1)
#   - Anti-discharge penalty (alpha=5.0)
#   - Context-aware V2G signal (alpha=3.0)
#   - Dense EV shaping (lambda_ev=5.0)
#   - PID Lagrangian + Sauté MDP
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

# --- Sauté MDP for C1 (from R15b) ---
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

# =====================================================================
# R16a: REWARD REFORM — replace time-blind r_eco/r_sg/r_sb
# =====================================================================

# DISABLE old STEMS components (set to 0)
export STEMS_MU_ECONOMIC="0.0"      # r_eco OFF (replaced by r_load_shift)
export STEMS_ALPHA_GRID="0.0"       # r_sg OFF (replaced by r_grid_mild)
export STEMS_ALPHA_BUILD="0.0"      # r_sb OFF (was time-blind)

# NEW: Battery-only price arbitrage
# r_load_shift = 3.0 × Σ_batt [-act_i × (price/mean_price - 1)] / N_batt
export STEMS_ALPHA_LOAD_SHIFT="3.0"

# NEW: Gentle linear grid awareness
# r_grid_mild = -0.3 × (import / P_grid_max)
export STEMS_ALPHA_GRID_MILD="0.3"

# INCREASED: Renewable incentive (was 0.2 — too weak vs old r_sg)
export STEMS_XI_RENEWABLE="1.5"

# REDUCED: Ramp penalty (was 1.0 — was competing with load shifting)
export STEMS_BETA_RAMP="0.3"

# =====================================================================
# Inherited from R15b (EV features — unchanged)
# =====================================================================
export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"  # Disabled (worsened C3 in R15c)

# R15a: Hard EV action clamp + anti-discharge penalty
export CITYLEARN_EV_ACTION_CLAMP="1"
export CITYLEARN_EV_CLAMP_MARGIN="0.1"
export STEMS_ALPHA_EV_GUARD="5.0"

# R15b: Context-aware V2G signal
export STEMS_ALPHA_V2G_CONTEXT="3.0"

# C2 stays OFF
export COST_W_C2="0.0"

# CMDPv2 cost weights
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

echo ""
echo "============================================"
echo "  R16a: Reward Reform"
echo "  Changes from R15b:"
echo "    - r_eco/r_sg/r_sb DISABLED (all set to 0)"
echo "    - r_load_shift=3.0 (battery price arbitrage)"
echo "    - r_grid_mild=0.3 (gentle linear grid awareness)"
echo "    - r_ren=1.5 (was 0.2, stronger solar incentive)"
echo "    - r_ramp=0.3 (was 1.0, reduced ramp penalty)"
echo "  Inherited from R15b:"
echo "    - Hard EV clamp (margin=0.1)"
echo "    - Anti-discharge guard (alpha=5.0)"
echo "    - V2G context (alpha=3.0)"
echo "    - PID Lagrangian + Sauté MDP (d=25000, shaped)"
echo "============================================"
echo ""

python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r16a_reward_reform.yaml
