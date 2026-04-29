#!/bin/bash
# Ablation: GradS vs Softmax — GradS arm
# R25b base with r_ev=1.0 (reduced from 5.0 so Lambda_0 stays active)
# Compare with run_ablation_softmax.sh (identical except gradient method)
# Uses MLP actor (no STEMS), matching original R25b
set -euo pipefail

eval "$(/home/christmas/miniconda3/bin/conda shell.bash hook)"
conda activate citylearn

cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export CITYLEARN_SCHEMA="$PWD/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# ── No STEMS / No temporal (R25b exact) ──
export CITYLEARN_TEMPORAL_WINDOW="0"

# ── Power thresholds (R25b exact) ──
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# ── PID Lagrangian ──
export CITYLEARN_PID_LAGRANGE="1"

# ── Sauté OFF (let Lagrangian handle constraints) ──
export CITYLEARN_EV_SAUTE="0"

# ── R25b reward weights (exact) ──
export STEMS_ALPHA_GRID="0.0"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_ALPHA_LOAD_SHIFT="0.0"
export STEMS_ALPHA_GRID_MILD="0.3"
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_XI_RENEWABLE="0.2"
export STEMS_BETA_RAMP="0.3"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_LAMBDA_EV="1.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_GUARD="1.0"
export STEMS_ALPHA_V2G_CONTEXT="3.0"
export STEMS_EV_SLACK_ARB_SCALE="2.0"
export STEMS_ALPHA_EV_SMART="1.5"

# ── Cost weights (R25b exact) ──
export COST_W_C2="0.0"
export COST_W_C3="5.0"
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# ── Controllability (R25b exact) ──
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_BATT_CLAMP="0"

# ── Observation/Action (R25b exact) ──
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_POLICY_ACTION_MASK="0"

echo "=== Ablation: GradS (paper-faithful) ==="
echo "  Gradient method: GradS (cosine similarity + uniform sampling)"
echo "  Actor: MLP [256,256] (no STEMS)"
echo "  Rewards: r_ev=1.0, r_ev_smart=1.5, r_ev_guard=1.0, r_v2g_ctx=3.0"
echo "  Sauté: OFF"
echo "  Lagrangian: PID, upper_bound=35, C0=200"
echo ""

nohup python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r25b_grads_ablation.yaml \
    > /tmp/ablation_grads.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/ablation_grads.log"
