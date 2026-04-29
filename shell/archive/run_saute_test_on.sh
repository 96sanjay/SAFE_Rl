#!/usr/bin/env bash
# R25b: R19 + EV Slack-Gated Arbitrage
# SINGLE CHANGE: STEMS_EV_SLACK_ARB_SCALE=2.0
set -euo pipefail
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Base env (IDENTICAL to R18) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_3buildings_3month.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Thresholds (IDENTICAL to R18)
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# PID Lagrangian (IDENTICAL to R18)
export CITYLEARN_PID_LAGRANGE="1"

# Sauté MDP for C1 (IDENTICAL to R18)
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="500"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"
export CITYLEARN_EV_DENSE_COST_SCALE="1.0"

# =====================================================================
# R19 CHANGE 2: Reward — zero 3 conflicting signals
# =====================================================================
export STEMS_ALPHA_GRID="0.0"          # R19: was 1.5 — redundant with C3/C4 λ
export STEMS_SG_THRESHOLD="0.5"
export STEMS_ALPHA_LOAD_SHIFT="0.0"    # R19: was 6.0 — biggest C3 offender
export STEMS_ALPHA_GRID_MILD="0.3"     # KEEP — soft signal
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_XI_RENEWABLE="0.2"        # KEEP — solar usage
export STEMS_BETA_RAMP="0.3"           # KEEP — smoothness

# C3_CONTROLLABLE (IDENTICAL to R18)
export CITYLEARN_C3_CONTROLLABLE="1"

# V2G reward signals
export STEMS_LAMBDA_EV="5.0"           # KEEP — EV charging guidance
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"       # KEEP — C2 needs help (44% violation)
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_GUARD="1.0"      # R19: was 5.0 — reduced, keeps mild anti-discharge during Phase 1
export STEMS_ALPHA_V2G_CONTEXT="3.0"   # KEEP — thesis contribution

# Cost weights (IDENTICAL to R18)
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

# Clamps (IDENTICAL to R18 — both OFF)
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="0"

echo ""
echo "============================================"
echo "  R25b: EV Slack-Gated Arbitrage"
echo "  Change 1: Cost advantage clipping REMOVED"
echo "  Change 2: r_load_shift=0, r_ev_guard 5→1, r_sg=0"
echo "  Change 3: C3 limit 12000→3000, C4 2500→1500, λ cap 6→3"
echo "  Change 4: 80 epochs, Phase 2 → ep 40"
echo "  Phase 1 (ep 0-19): C0 disabled, V2G discovery"
echo "  Phase 2 (ep 20-40): C0 anneals 999999→1800"
echo "  Phase 3 (ep 40-79): All constraints, fine-tune"
echo "============================================"
echo ""

export STEMS_EV_SLACK_ARB_SCALE="2.0"
/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r25b_ev_slack_arb.yaml
