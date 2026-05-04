#!/bin/bash
# R27f: Moderate dual — r_ev + r_ev_smart
# ==================================================================
# Key changes from R27a:
#   1. r_ev=2.0 (moderate charging incentive)
#   2. r_ev_smart=1.0 (headroom-gated price signal)
#   3. r_ev_guard=0.5 (light anti-discharge guard)
#   4. r_sg=1.0, r_sb=1.0 (REDUCED from 1.5 to make room for EV)
#   5. Test: does dual r_ev + r_ev_smart give best of both?
# ==================================================================
set -euo pipefail

# Activate conda environment
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

# -- Reward terms (REBALANCED for dual EV) --
export STEMS_MU_ECONOMIC=0.0
export STEMS_ALPHA_GRID=1.0             # REDUCED from 1.5 (make room for EV)
export STEMS_ALPHA_BUILD=1.0            # REDUCED from 1.5 (make room for EV)
export STEMS_BETA_RAMP=0.3
export STEMS_XI_RENEWABLE=0.3
export STEMS_LAMBDA_EV=2.0              # moderate r_ev
export STEMS_ALPHA_BARRIER=0.3

# -- Dual EV reward --
export STEMS_ALPHA_EV_SMART=1.0         # headroom-gated price signal
export STEMS_ALPHA_EV_GUARD=0.5         # light anti-discharge guard

# -- OFF reward terms --
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

# -- Dense C0 OFF --
export CITYLEARN_EV_SAUTE=0
export CITYLEARN_C0_DENSE_MERGE=0

# No clamps
export CITYLEARN_BATT_CLAMP=0
export CITYLEARN_EV_CLAMP=0

# C2 SoC range
export CITYLEARN_STEMS_SOC_LOW=0.0
export CITYLEARN_STEMS_SOC_HIGH=0.95

# Cost coefficients
export CITYLEARN_C3_COST=0.1
export CITYLEARN_C4_COST=5.0
export CITYLEARN_C3_CONTROLLABLE=1

# PID Lagrangian
export CITYLEARN_PID_LAGRANGE=1

echo "=== R27f: Moderate dual r_ev + r_ev_smart (40-epoch) ==="
echo "  Reward: r_sg=1.0 r_sb=1.0 r_ramp=0.3 r_ren=0.3 r_barrier=0.3"
echo "  EV:     r_ev=2.0 r_ev_smart=1.0 r_ev_guard=0.5 (dual)"
echo "  C0: limit=200, Kp=0.5, Ki=0.01"
echo "  C3: limit=5000, Kp=0.5, Ki=0.05"
echo "  C4: limit=8000, Kp=0.3, Ki=0.03"
echo ""

nohup python scripts/train_multi_lag_stems.py \
    --cfg configs/active/headroom_moderate.yaml \
    --hidden_dim 64 \
    --output_dim 256 \
    --num_gcn_layers 3 \
    --stems_critic \
    --temporal_layers 2 \
    --temporal_pool mean \
    --partial_obs_norm \
    > /tmp/headroom_moderate.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/headroom_moderate.log"
echo "Monitor: tail -f /tmp/headroom_moderate.log"
