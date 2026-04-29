#!/usr/bin/env bash
# R21: STEMS GCN-Transformer + PPOLagMulti — Raised Lambda Cap
#
# KEY FIX: lambda_upper_bound 3.0 → 12.0
# Root cause from analyze_r20.py: Lambda_3 and Lambda_4 saturated at cap=3.0
# from epoch 1-3. Agent never felt increasing constraint pressure. Raising cap
# gives 4x more gradient signal when C3/C4 costs exceed limits.
#
# Other changes from R20:
#   - 120 epochs (was 80)
#   - Phase 2 delayed to ep 40-60 (C3 declined 78% slower when Phase 2 started in R20)
#   - pid_ki_4 raised: 0.03 → 0.05 (faster C4 integral buildup)
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Base env (IDENTICAL to R20) ---
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

# PID Lagrangian
export CITYLEARN_PID_LAGRANGE="1"

# Sauté MDP for C1
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

# Reward (IDENTICAL to R20 — zeroed conflicting signals)
export STEMS_ALPHA_GRID="0.0"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_ALPHA_LOAD_SHIFT="0.0"
export STEMS_ALPHA_GRID_MILD="0.3"
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_XI_RENEWABLE="0.2"
export STEMS_BETA_RAMP="0.3"

# C3_CONTROLLABLE
export CITYLEARN_C3_CONTROLLABLE="1"

# V2G reward signals (IDENTICAL to R20)
export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_GUARD="1.0"
export STEMS_ALPHA_V2G_CONTEXT="3.0"

# Cost weights (IDENTICAL to R20)
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

# Clamps (both OFF — same as R20)
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="0"

# STEMS v3 + Temporal Window + GPU (IDENTICAL to R20)
export STEMS_ENCODER_VERSION="v3"
export CITYLEARN_TEMPORAL_WINDOW="12"

echo ""
echo "============================================"
echo "  R21: STEMS GCN-Transformer + PPOLagMulti"
echo "  KEY FIX: lambda_cap 3.0 → 12.0"
echo "  Architecture: STEMS v3 (GCN + Temporal Transformer)"
echo "  120 epochs (was 80)"
echo "  Phase 1 (ep 0-39): C0 disabled — MORE time for C3/C4"
echo "  Phase 2 (ep 40-60): C0 anneals 999999→1800"
echo "  Phase 3 (ep 60-119): All active, full λ pressure"
echo "  λ cap = 12.0 (4x R20's 3.0 — saturated since ep 1!)"
echo "============================================"
echo ""

python scripts/train_multi_lag_stems.py \
    --cfg configs/on-policy/r21_stems.yaml
