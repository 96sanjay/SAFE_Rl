#!/usr/bin/env bash
# R23: PPOLagMulti — Solar-Aligned EV Charging Reward
#
# Based on R21 (best PPO run). Single change:
#   - STEMS_ALPHA_EV_SOLAR=3.0 (new r_ev_solar reward component)
#
# R21 diagnosis:
#   - EV charging anti-solar: r=-0.21 correlation, only 28.6% during solar hours
#   - Battery charging IS solar-aware: r=+0.28, 58.3% during solar
#   - Grid import=48,520 kWh while solar=57,824 kWh — massive solar waste
#   - CityLearn KPIs=2.18x (worse than no-control baseline)
#
# Fix: r_ev_solar = alpha * mean(max(0,ev_action) * solar_norm)
#   - Rewards EV charging proportional to solar availability
#   - Only positive actions (charging), not discharge
#   - All connected EVs (not just surplus like r_v2g_context)
#   - alpha=3.0 (same scale as r_v2g_context, won't override r_ev urgency=5.0)
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Base env (IDENTICAL to R21) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Thresholds (IDENTICAL to R21)
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# PID Lagrangian (IDENTICAL to R21)
export CITYLEARN_PID_LAGRANGE="1"

# Sauté MDP for C1 (IDENTICAL to R21)
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

# Reward (IDENTICAL to R21 — zeroed conflicting signals)
export STEMS_ALPHA_GRID="0.0"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_ALPHA_LOAD_SHIFT="0.0"
export STEMS_ALPHA_GRID_MILD="0.3"
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_XI_RENEWABLE="0.2"
export STEMS_BETA_RAMP="0.3"

# C3_CONTROLLABLE (IDENTICAL to R21)
export CITYLEARN_C3_CONTROLLABLE="1"

# V2G reward signals (IDENTICAL to R21)
export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_GUARD="1.0"
export STEMS_ALPHA_V2G_CONTEXT="3.0"

# =====================================================================
# R23: NEW — Solar-aligned EV charging reward
# alpha=3.0 (same scale as r_v2g_context; won't override r_ev urgency=5.0)
# =====================================================================
export STEMS_ALPHA_EV_SOLAR="3.0"

# Cost weights (IDENTICAL to R21)
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

# Safety clamps (IDENTICAL to R21)
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="1"

echo ""
echo "============================================"
echo "  R23: PPOLagMulti — Solar-Aligned EV Charging"
echo "  Based on R21 (best PPO run)"
echo "  NEW: STEMS_ALPHA_EV_SOLAR=3.0"
echo "  Target: EV solar charging 28.6% → 45-55%"
echo "  Target: Solar-EV correlation -0.21 → >0"
echo "  90 epochs, all other config identical to R21"
echo "============================================"
echo ""

/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r23_ppo.yaml
