#!/usr/bin/env bash
# R22: PPOLagMulti + GradS + Lambda Cap 12.0
#
# Builds on R21 (which fixed KL instability + Safety Projection for C2).
# R21 ep 34 status: C3=35,369, C4=18,459 — flattening in Phase 2.
# Root cause: C0 gradient conflicts with C3/C4 in Phase 2 (78% slowdown).
#
# R22 adds:
#   1. GradS — gradient surgery integrated into PPOLagMulti
#      (preserves PID Lagrangian, curriculum, multi-critic)
#      Detects C0 vs C3/C4 conflicts per mini-batch, picks non-conflicting gradient
#   2. lambda_upper_bound 5.0 → 12.0 — more pressure headroom in Phase 3
#   3. 120 epochs — Phase 3 needs more time at lower decline rate
#
# Unchanged from R21:
#   actor_lr=0.0001, target_kl=0.08, batch=256 (trust region stable)
#   BATT_CLAMP=1 (Safety Projection for C2, λ2=0.000 confirmed all epochs)
#   Same reward, same curriculum (Phase 2 ep 20-40), same PID gains
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

export CITYLEARN_PID_LAGRANGE="1"

export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

# Reward (identical to R21)
export STEMS_ALPHA_GRID="0.0"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_ALPHA_LOAD_SHIFT="0.0"
export STEMS_ALPHA_GRID_MILD="0.3"
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_XI_RENEWABLE="0.2"
export STEMS_BETA_RAMP="0.3"

export CITYLEARN_C3_CONTROLLABLE="1"

export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_GUARD="1.0"
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

export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="1"          # Safety Projection for C2 (Dalal 2018)

echo ""
echo "============================================"
echo "  R22: PPOLagMulti + GradS + Lambda Cap 12"
echo "  GradS: gradient surgery integrated into PPOLagMulti"
echo "         (PID + curriculum + multi-critic preserved)"
echo "  lambda_cap: 5.0 → 12.0 (2.4x more headroom)"
echo "  120 epochs (Phase 3 needs time at lower rate)"
echo "  Inherited from R21:"
echo "    actor_lr=0.0001, target_kl=0.08, batch=256"
echo "    BATT_CLAMP=1 (C2 Safety Projection, λ2=0)"
echo "  Phase 1 (ep 0-19):  C0 disabled"
echo "  Phase 2 (ep 20-40): C0 anneals 999999→1800"
echo "  Phase 3 (ep 40-119): All constraints, GradS resolves conflicts"
echo "============================================"
echo ""

/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r22_ppo.yaml
