#!/bin/bash
# R26j: Lagrangian Fix + Trajectory Boost
# ========================================
# Key changes from R26h:
#   1. C0 curriculum: [200,30,0,30] — immediate EV pressure (was [999999,50,20,40])
#   2. C0 Ki=0.05 — integral memory (was 0.0)
#   3. penalty_max=20 (was 10)
#   4. C3 limit: 5000 (was 15000 — now binding)
#   5. r_trajectory: 20.0 (was 2.0 — 10x boost)
#   6. r_sg: 1.0 (was 2.0 — halved)
#   7. r_sb: 2.0 (was 3.0 — reduced)
#
# Same 5 reward terms, rebalanced for temporal learning.
# NO ev_guard — C0 Lagrangian handles departure with fixed curriculum.
# ========================================
set -euo pipefail

# Activate conda environment
eval "$(/home/sanjay/miniconda3/bin/conda shell.bash hook)"
conda activate citylearn

export CITYLEARN_SCHEMA="$PWD/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_TEMPORAL_RICH=1
export CITYLEARN_TEMPORAL_WINDOW=12
export STEMS_ENCODER_VERSION=v3

# ── 5 active reward terms (REBALANCED) ──
export STEMS_MU_ECONOMIC=0.0          # r_eco OFF
export STEMS_ALPHA_GRID=1.0           # r_sg ON (was 2.0 — halved)
export STEMS_ALPHA_BUILD=2.0          # r_sb ON (was 3.0 — reduced)
export STEMS_BETA_RAMP=0.0            # r_ramp OFF
export STEMS_XI_RENEWABLE=0.3         # r_ren ON (unchanged)
export STEMS_LAMBDA_EV=0.0            # r_ev OFF
export STEMS_ALPHA_BARRIER=1.0        # r_barrier ON (unchanged)
export STEMS_ALPHA_TRAJECTORY=20.0    # r_trajectory ON (was 2.0 — 10x boost!)

# All other reward terms explicitly OFF
export STEMS_ALPHA_EV_GUARD=0.0       # OFF — C0 Lagrangian handles departure
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

# Forecast arbitrage — EV temporal enabled
export STEMS_TRAJ_EV_WEIGHT=1.0       # EVs learn temporal arbitrage
export STEMS_TRAJ_FORECAST_HOURS=24

# No clamps — pure RL
export CITYLEARN_BATT_CLAMP=0
export CITYLEARN_EV_CLAMP=0

# C2 SoC range: 0.0–0.95
export CITYLEARN_STEMS_SOC_LOW=0.0
export CITYLEARN_STEMS_SOC_HIGH=0.95

# Cost coefficients
export CITYLEARN_C3_COST=0.1
export CITYLEARN_C4_COST=5.0
export CITYLEARN_C3_CONTROLLABLE=1

# PID Lagrangian (CRITICAL — without this, falls back to SGD Lagrangian!)
export CITYLEARN_PID_LAGRANGE=1

echo "=== R26j: Lagrangian Fix + Trajectory Boost ==="
echo "  Reward: r_sg=1.0 r_sb=2.0 r_ren=0.3 r_barrier=1.0 r_trajectory=20.0"
echo "  C0: limit=[200→30], Kp=3.0, Ki=0.05, penalty_max=20"
echo "  C3: limit=5000, Kp=0.5, Ki=0.05"
echo "  C4: limit=8000, Kp=0.3, Ki=0.03"
echo ""

nohup python scripts/train_multi_lag_stems.py \
    --cfg configs/on-policy/r26j_lagfix.yaml \
    --hidden_dim 64 \
    --output_dim 256 \
    --num_gcn_layers 3 \
    --stems_critic \
    --temporal_layers 2 \
    --temporal_pool mean \
    --partial_obs_norm \
    > /tmp/r26j_lagfix.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/r26j_lagfix.log"
echo "Monitor: tail -f /tmp/r26j_lagfix.log"
