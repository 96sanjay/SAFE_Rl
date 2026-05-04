#!/bin/bash
# R27e: R27b + r_ev_guard=1.0 (optimal combo)
# r_ev=2.5 + r_ev_smart=1.5 + r_ev_guard=1.0 + r_sg=1.5 + r_sb=1.5
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

export CITYLEARN_SCHEMA="$PWD/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_TEMPORAL_RICH=1
export CITYLEARN_TEMPORAL_WINDOW=12
export STEMS_ENCODER_VERSION=v3

# ── Reward terms ──
export STEMS_MU_ECONOMIC=0.0
export STEMS_ALPHA_GRID=1.5
export STEMS_ALPHA_BUILD=1.5
export STEMS_BETA_RAMP=0.3
export STEMS_XI_RENEWABLE=0.3
export STEMS_LAMBDA_EV=2.5            # r_ev ON — dense charging (moderate)
export STEMS_ALPHA_BARRIER=0.3
export STEMS_ALPHA_EV_SMART=1.5       # departure-aware price signal
export STEMS_ALPHA_EV_GUARD=1.0       # NEW: prevents V2G during deficit

# ── OFF ──
export STEMS_ALPHA_TRAJECTORY=0.0
export STEMS_TRAJ_EV_WEIGHT=0.0
export STEMS_ALPHA_V2G_CONTEXT=0.0
export STEMS_ALPHA_PEAK_SHAVE=0.0
export STEMS_ALPHA_LOAD_SHIFT=0.0
export STEMS_ALPHA_GRID_MILD=0.0
export STEMS_ALPHA_EV_SOLAR=0.0
export STEMS_ALPHA_SOLAR_STORE=0.0
export STEMS_EV_SLACK_ARB_SCALE=0.0
export STEMS_ALPHA_HEADROOM=0.0
export STEMS_ALPHA_GRID_PENALTY=0.0
export STEMS_ALPHA_PRICE_ARB=0.0
export STEMS_ALPHA_NEC_SIGN=0.0

# ── No Sauté ──
export CITYLEARN_EV_SAUTE=0
export CITYLEARN_C0_DENSE_MERGE=0

export CITYLEARN_BATT_CLAMP=0
export CITYLEARN_EV_CLAMP=0
export CITYLEARN_STEMS_SOC_LOW=0.0
export CITYLEARN_STEMS_SOC_HIGH=0.95
export CITYLEARN_C3_COST=0.1
export CITYLEARN_C4_COST=5.0
export CITYLEARN_C3_CONTROLLABLE=1
export CITYLEARN_PID_LAGRANGE=1

echo "=== R27e: R27b + r_ev_guard=1.0 (40-epoch) ==="
echo "  Reward: r_sg=1.5 r_sb=1.5 r_ramp=0.3 r_ren=0.3 r_barrier=0.3"
echo "  EV:     r_ev=2.5 r_ev_smart=1.5 r_ev_guard=1.0"
echo "  C0: limit=200, Kp=2.0, Ki=0.05"
echo ""

nohup python scripts/train_multi_lag_stems.py \
    --cfg configs/active/headroom_dense_ev_guard.yaml \
    --hidden_dim 64 \
    --output_dim 256 \
    --num_gcn_layers 3 \
    --stems_critic \
    --temporal_layers 2 \
    --temporal_pool mean \
    --partial_obs_norm \
    > /tmp/headroom_dense_ev_guard.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/headroom_dense_ev_guard.log"
