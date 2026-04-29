#!/bin/bash
# R28_CPO: CPO benchmark — same 9-term reward as R28b, local GPU
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

# ── PID Lagrangian OFF (CPO uses trust region, not Lagrangian) ──
export CITYLEARN_PID_LAGRANGE="0"

# ── Sauté OFF ──
export CITYLEARN_EV_SAUTE="0"

# ══════════════════════════════════════════════
# 9 ACTIVE REWARD TERMS (same as R28b)
# ══════════════════════════════════════════════

# --- EV charging (4 terms) ---
export STEMS_LAMBDA_EV="2.0"
export STEMS_ALPHA_EV_SMART="1.5"
export STEMS_EV_SLACK_ARB_SCALE="1.0"
export STEMS_ALPHA_V2G_CONTEXT="1.5"

# --- Battery (1 term) ---
export STEMS_ALPHA_BARRIER="0.5"

# --- Grid/Building (2 terms) ---
export STEMS_ALPHA_GRID="0.5"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BUILD="0.3"
export STEMS_SB_ASYMMETRIC="1"

# --- Grid quality (2 terms) ---
export STEMS_BETA_RAMP="0.3"
export STEMS_XI_RENEWABLE="0.2"

# --- DISABLED ---
export STEMS_ALPHA_EV_GUARD="0.0"
export STEMS_ALPHA_GRID_MILD="0.0"
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_LOAD_SHIFT="0.0"

# ══════════════════════════════════════════════
# COST WEIGHTS (same as R28b — sets aggregate cost signal for CPO)
# ══════════════════════════════════════════════
export COST_W_C1_DENSE="0.0"
export COST_W_C2="5.0"
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

echo "=== R28_CPO: CPO Benchmark, 9-Term Reward, GPU ==="
echo "  Active rewards: r_ev=2.0, r_ev_smart=1.5, r_ev_slack_arb=1.0,"
echo "                  r_v2g_ctx=1.5, r_barrier=0.5,"
echo "                  r_sg=0.5, r_sb=0.3, r_ramp=0.3, r_ren=0.2"
echo "  CPO: trust-region, single aggregate constraint"
echo "  cost_limit=100000 (aggregate of 10*C0 + 5*C3 + 5*C4)"
echo "  100 epochs, cuda:0"
echo ""

nohup python scripts/train_omnisafe.py \
    --cfg configs/active/benchmark_cpo.yaml \
    > /tmp/benchmark_cpo.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/benchmark_cpo.log"
