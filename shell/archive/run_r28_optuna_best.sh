#!/bin/bash
# R28 Optuna Best: best trial from r28_sweep (score=343.31, 63 complete trials)
# Reward weights and PID/limit params taken directly from Optuna best_trial.params
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

# ── No STEMS encoder / No temporal ──
export CITYLEARN_TEMPORAL_WINDOW="0"

# ── Power thresholds ──
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# ── PID Lagrangian ──
export CITYLEARN_PID_LAGRANGE="1"

# ── Sauté OFF ──
export CITYLEARN_EV_SAUTE="0"

# ══════════════════════════════════════════════
# 9 REWARD TERMS — Optuna best values
# ══════════════════════════════════════════════

# --- EV charging (4 terms) ---
export STEMS_LAMBDA_EV="2.9196"
export STEMS_ALPHA_EV_SMART="1.2280"
export STEMS_EV_SLACK_ARB_SCALE="0.8791"
export STEMS_ALPHA_V2G_CONTEXT="1.7226"

# --- Battery SoC ---
export STEMS_ALPHA_BARRIER="0.2181"

# --- Grid/Building ---
export STEMS_ALPHA_GRID="3.9518"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BUILD="2.7973"
export STEMS_SB_ASYMMETRIC="1"

# --- Grid quality ---
export STEMS_BETA_RAMP="0.0736"
export STEMS_XI_RENEWABLE="0.3809"

# --- DISABLED ---
export STEMS_ALPHA_EV_GUARD="0.0"
export STEMS_ALPHA_GRID_MILD="0.0"
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_LOAD_SHIFT="0.0"

# ══════════════════════════════════════════════
# COST WEIGHTS
# ══════════════════════════════════════════════
export COST_W_C2="0.0"
export COST_W_C3="5.0"
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# ── Controllability ──
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_BATT_CLAMP="0"

# ── Observation/Action ──
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_POLICY_ACTION_MASK="0"

echo "=== R28 Optuna Best: 9-Term Reward, Optuna-tuned Limits & PID ==="
echo "  r_ev=2.9196, r_ev_smart=1.2280, r_ev_slack_arb=0.8791, r_v2g_ctx=1.7226"
echo "  r_barrier=0.2181, r_sg=3.9518, r_sb=2.7973, r_ramp=0.0736, r_ren=0.3809"
echo "  C0=105.6, C2=3313, C3=19066, C4=11055, lambda_cap=25.83"
echo "  C3 PID: kp=0.5027 ki=0.00795 | C4 PID: kp=0.10514 ki=0.00679"
echo "  40 epochs, CPU"
echo ""

nohup python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r28_optuna_best.yaml \
    > /tmp/r28_optuna_best.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/r28_optuna_best.log"
