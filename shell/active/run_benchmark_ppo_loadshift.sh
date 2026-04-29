#!/bin/bash
# R28c: R28b + r_load_shift (battery price arbitrage)
# Only change from R28b:
#   - r_load_shift 0.0→0.5 (solar-aware battery price arbitrage)
#   - Now 10 active reward terms (was 9)
#   - C3=18000 gives batteries room to actually do arbitrage
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
# 10 ACTIVE REWARD TERMS (R28b + r_load_shift)
# ══════════════════════════════════════════════

# --- EV charging (4 terms, layered) ---
export STEMS_LAMBDA_EV="2.0"              # dense charge urgency
export STEMS_ALPHA_EV_SMART="1.5"         # headroom-gated price signal
export STEMS_EV_SLACK_ARB_SCALE="1.0"     # solar-aware arb + peak V2G
export STEMS_ALPHA_V2G_CONTEXT="1.5"      # surplus EV context

# --- Battery SoC ---
export STEMS_ALPHA_BARRIER="0.5"          # SoC boundary

# --- Battery timing (NEW in R28c) ---
export STEMS_ALPHA_LOAD_SHIFT="0.5"       # solar-aware battery price arbitrage

# --- Grid/Building ---
export STEMS_ALPHA_GRID="0.5"             # quadratic grid peak (C4)
export STEMS_SG_THRESHOLD="0.5"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BUILD="0.3"            # per-building power (C3)
export STEMS_SB_ASYMMETRIC="1"

# --- Grid quality ---
export STEMS_BETA_RAMP="0.3"              # ramping
export STEMS_XI_RENEWABLE="0.2"           # renewable self-consumption

# --- DISABLED ---
export STEMS_ALPHA_EV_GUARD="0.0"         # redundant with r_ev
export STEMS_ALPHA_GRID_MILD="0.0"        # replaced by r_sg
export STEMS_MU_ECONOMIC="0.0"            # harmful — agent learned wrong direction
export STEMS_ALPHA_PEAK_SHAVE="0.0"       # dead — batteries too constrained

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

echo "=== R28c: R28b + Battery Price Arbitrage ==="
echo "  Active rewards: r_ev=2.0, r_ev_smart=1.5, r_ev_slack_arb=1.0,"
echo "                  r_v2g_ctx=1.5, r_barrier=0.5, r_load_shift=0.5,"
echo "                  r_sg=0.5, r_sb=0.3, r_ramp=0.3, r_ren=0.2"
echo "  Disabled: r_ev_guard, r_grid_mild, r_eco, r_peak_shave"
echo "  PID: C0(2.0/0.05) C2(0.1/0.01) C3(0.1/0.01) C4(0.1/0.01)"
echo "  Limits: C0=100 C2=3500 C3=18000 C4=13000"
echo "  Lambda cap: 25.0, Training: 100 epochs"
echo ""

nohup python scripts/train_multi_lag.py \
    --cfg configs/active/benchmark_ppo_loadshift.yaml \
    > /tmp/benchmark_ppo_loadshift.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/benchmark_ppo_loadshift.log"
