#!/bin/bash
# R28a: Clean 10-term reward, balanced Lagrangian
# Changes from R25b Softmax ablation:
#   - r_sg=0.5 ENABLED (was 0.0) → C4 quadratic peak penalty
#   - r_sb=0.3 ENABLED (was 0.0) → C3 building stability
#   - r_eco=0.3 ENABLED (was 0.0) → cost KPI optimization
#   - r_peak_shave=0.3 ENABLED (was 0.0) → peak demand KPI
#   - r_load_shift=0.3 ENABLED (was 0.0) → battery price arbitrage (kept low, was C3 offender at 6.0)
#   - r_grid_mild=0.0 OFF (was 0.3) → r_sg covers it
#   - r_v2g_ctx=1.0 reduced (was 3.0) → kept for C0 directional signal
#   - r_ev_slack_arb=0.0 OFF (was 2.0) → noise
#   - r_ev_guard=0.0 OFF (was 1.0) → redundant with r_ev (conflicting urgency)
#   - r_ev: 1.0→0.5, r_ev_smart: 1.5→1.0
#   - C4 PID: Kp 0.3→0.8, Ki 0.03→0.08
#   - 60 epochs (was 40)
set -euo pipefail

if [ -n "${CONDA_EXE:-}" ]; then
    eval "$(${CONDA_EXE} shell.bash hook)"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not found. Install conda or set CONDA_EXE." >&2
    exit 1
fi
conda activate citylearn

cd "$(dirname "$0")/../.."
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
# 10 ACTIVE REWARD TERMS (clean, no redundancy)
# ══════════════════════════════════════════════

# --- C0: EV charging (reduced — let Lagrangian co-drive) ---
export STEMS_LAMBDA_EV="0.5"              # was 1.0 — charge urgency
export STEMS_ALPHA_EV_GUARD="0.0"         # OFF — redundant with r_ev (conflicting urgency signals)
export STEMS_ALPHA_EV_SMART="1.0"         # was 1.5 — smart timing

# --- C2: Battery SoC ---
export STEMS_ALPHA_BARRIER="0.5"          # unchanged — SoC boundary

# --- C3: Building power (NEW) ---
export STEMS_ALPHA_BUILD="0.3"            # was 0.0 — per-building stability
export STEMS_SB_ASYMMETRIC="1"

# --- C4: Grid power (NEW quadratic, replaces linear) ---
export STEMS_ALPHA_GRID="0.5"             # was 0.0 — quadratic peak penalty
export STEMS_SG_THRESHOLD="0.5"
export STEMS_SG_EXPORT_CREDIT="0.5"

# --- Grid quality ---
export STEMS_BETA_RAMP="0.3"              # unchanged — ramping
export STEMS_XI_RENEWABLE="0.2"           # unchanged — renewable

# --- V2G context (reduced, not removed — critical for C0 signal) ---
export STEMS_ALPHA_V2G_CONTEXT="1.0"      # was 3.0 — reduced but kept (data shows removing kills C0 signal)

# --- KPI optimization (NEW — all were 0.0 in R25b) ---
export STEMS_MU_ECONOMIC="0.3"            # cost KPI — price × (import - 0.7×export)
export STEMS_ALPHA_PEAK_SHAVE="0.3"       # peak demand KPI — battery discharge during peaks
export STEMS_ALPHA_LOAD_SHIFT="0.3"       # battery arbitrage — charge cheap, discharge expensive (was 6.0→C3 offender, kept low)

# --- DISABLED ---
export STEMS_ALPHA_GRID_MILD="0.0"        # was 0.3 — r_sg covers it
export STEMS_EV_SLACK_ARB_SCALE="0.0"     # was 2.0 — noise

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

echo "=== R28a: Clean 10-Term Reward ==="
echo "  Active rewards: r_ev=0.5, r_ev_smart=1.0, r_v2g_ctx=1.0,"
echo "                  r_barrier=0.5, r_sb=0.3, r_sg=0.5,"
echo "                  r_ramp=0.3, r_ren=0.2, r_eco=0.3,"
echo "                  r_peak_shave=0.3, r_load_shift=0.3"
echo "  Disabled: r_ev_guard, r_grid_mild, r_ev_slack_arb"
echo "  PID: C0(1.0/0.03) C2(0.2/0.02) C3(0.5/0.05) C4(0.8/0.08)"
echo "  Limits: C0=100 C2=1000 C3=5000 C4=8000"
echo "  Training: 60 epochs, Softmax tau=1.0"
echo ""

nohup python scripts/train_multi_lag.py \
    --cfg configs/active/benchmark_reward_10term.yaml \
    > /tmp/benchmark_reward_10term.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/benchmark_reward_10term.log"
