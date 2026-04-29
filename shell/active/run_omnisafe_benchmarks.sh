#!/bin/bash
# =============================================================================
# OmniSafe Benchmark Suite — 5 algorithms on srv07 GPU
# =============================================================================
# Runs PPOLag, TRPOLag, FOCOPS, CPPOPID, PCPO in parallel on A100 GPU.
# All share the same 9-term reward (env vars) and aggregate cost signal.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
if [ -f "/home/sanjay/miniconda3/bin/conda" ]; then
    eval "$(/home/sanjay/miniconda3/bin/conda shell.bash hook)"
elif [ -f "/home/christmas/miniconda3/bin/conda" ]; then
    eval "$(/home/christmas/miniconda3/bin/conda shell.bash hook)"
else
    echo "ERROR: conda not found"
    exit 1
fi
conda activate citylearn

PROJECT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Limit CPU threads — Optuna is using 88/96 cores, leave headroom
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TORCH_NUM_THREADS=2

# ---------------------------------------------------------------------------
# Environment variables (shared across all benchmarks — same as R28b/R28_CPO)
# ---------------------------------------------------------------------------
export CITYLEARN_SCHEMA="$PWD/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# No STEMS encoder / No temporal
export CITYLEARN_TEMPORAL_WINDOW="0"

# Power thresholds
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# PID Lagrangian OFF (these algorithms use OmniSafe built-in constraint handling)
export CITYLEARN_PID_LAGRANGE="0"

# Saute OFF
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
# COST WEIGHTS (aggregate cost signal for single-constraint algorithms)
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

# Controllability
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_BATT_CLAMP="0"

# Observation/Action
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_POLICY_ACTION_MASK="0"

# ---------------------------------------------------------------------------
# Algorithm list
# ---------------------------------------------------------------------------
ALGOS=("ppolag" "trpolag" "focops" "cppopid" "pcpo")
LOG_DIR="/tmp/omnisafe_benchmarks"
mkdir -p "${LOG_DIR}"

echo "================================================================"
echo "  OmniSafe Benchmark Suite"
echo "================================================================"
echo "  Algorithms:  ${ALGOS[*]}"
echo "  Epochs:      100"
echo "  Device:      cuda:0"
echo "  Reward:      9-term STEMS (same as R28b)"
echo "  Constraint:  aggregate cost, limit=100000"
echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# Launch all benchmarks in parallel
# ---------------------------------------------------------------------------
PIDS=()

cleanup() {
    echo ""
    echo "Caught signal, killing all benchmarks..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    echo "All killed."
}
trap cleanup EXIT INT TERM

for algo in "${ALGOS[@]}"; do
    CFG="configs/active/benchmark_${algo}.yaml"
    LOG="${LOG_DIR}/${algo}.log"

    echo "  Launching ${algo} ..."
    nohup python scripts/train_omnisafe.py \
        --cfg "${CFG}" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
    echo "    PID: ${!}  Log: ${LOG}"
    sleep 2  # stagger to avoid GPU init contention
done

echo ""
echo "All ${#ALGOS[@]} benchmarks launched."
echo ""
echo "Monitor:"
for algo in "${ALGOS[@]}"; do
    echo "  tail -f ${LOG_DIR}/${algo}.log"
done
echo ""
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Waiting for all to finish..."

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || ((FAILED++))
done
echo ""
echo "================================================================"
echo "  All done. Failed: ${FAILED}/${#ALGOS[@]}"
echo "================================================================"

trap - EXIT INT TERM
