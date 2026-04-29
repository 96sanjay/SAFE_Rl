#!/bin/bash
# R27 Parallel Launcher: R27b through R27f
# ==========================================
# R27a is already running — do NOT relaunch.
# This script launches R27b-R27f in background.
#
# Experiment grid:
#   R27b: Strong r_ev=5.0 baseline (proven C0 solver)
#   R27c: Pure Lagrangian, aggressive PID (Kp=5.0, Ki=0.1)
#   R27d: Hybrid r_ev_smart + r_ev_guard + stronger PID
#   R27e: Dense C0 via Saute merge
#   R27f: Moderate dual r_ev=2.0 + r_ev_smart=1.0
# ==========================================
set -euo pipefail

cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork

echo "============================================"
echo "  R27 Parallel Launcher (b through f)"
echo "  R27a already running — skipped"
echo "============================================"
echo ""

# Activate conda environment
eval "$(/home/sanjay/miniconda3/bin/conda shell.bash hook)"
conda activate citylearn

# Common env vars (set once, inherited by all)
export CITYLEARN_SCHEMA="$PWD/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_TEMPORAL_RICH=1
export CITYLEARN_TEMPORAL_WINDOW=12
export STEMS_ENCODER_VERSION=v3
export STEMS_MU_ECONOMIC=0.0
export STEMS_BETA_RAMP=0.3
export STEMS_XI_RENEWABLE=0.3
export STEMS_ALPHA_BARRIER=0.3
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
export CITYLEARN_BATT_CLAMP=0
export CITYLEARN_EV_CLAMP=0
export CITYLEARN_STEMS_SOC_LOW=0.0
export CITYLEARN_STEMS_SOC_HIGH=0.95
export CITYLEARN_C3_COST=0.1
export CITYLEARN_C4_COST=5.0
export CITYLEARN_C3_CONTROLLABLE=1
export CITYLEARN_PID_LAGRANGE=1

TRAIN_CMD="python scripts/train_multi_lag_stems.py"
TRAIN_ARGS="--hidden_dim 64 --output_dim 256 --num_gcn_layers 3 --stems_critic --temporal_layers 2 --temporal_pool mean --partial_obs_norm"

# --- R27b: Strong r_ev baseline ---
echo "[R27b] Strong r_ev=5.0 baseline"
(
    export STEMS_ALPHA_GRID=1.5
    export STEMS_ALPHA_BUILD=1.5
    export STEMS_LAMBDA_EV=5.0
    export STEMS_ALPHA_EV_SMART=0.0
    export STEMS_ALPHA_EV_GUARD=0.0
    export CITYLEARN_EV_SAUTE=0
    export CITYLEARN_C0_DENSE_MERGE=0
    nohup $TRAIN_CMD --cfg configs/on-policy/r27b_cmdp.yaml $TRAIN_ARGS \
        > /tmp/r27b.log 2>&1 &
    echo "  PID: $!  Log: /tmp/r27b.log"
)

# --- R27c: Pure Lagrangian, aggressive PID ---
echo "[R27c] Pure Lagrangian, aggressive PID"
(
    export STEMS_ALPHA_GRID=1.5
    export STEMS_ALPHA_BUILD=1.5
    export STEMS_LAMBDA_EV=0.0
    export STEMS_ALPHA_EV_SMART=0.0
    export STEMS_ALPHA_EV_GUARD=0.0
    export CITYLEARN_EV_SAUTE=0
    export CITYLEARN_C0_DENSE_MERGE=0
    nohup $TRAIN_CMD --cfg configs/on-policy/r27c_cmdp.yaml $TRAIN_ARGS \
        > /tmp/r27c.log 2>&1 &
    echo "  PID: $!  Log: /tmp/r27c.log"
)

# --- R27d: Hybrid r_ev_smart + r_ev_guard + stronger PID ---
echo "[R27d] Hybrid r_ev_smart + r_ev_guard + stronger PID"
(
    export STEMS_ALPHA_GRID=1.5
    export STEMS_ALPHA_BUILD=1.5
    export STEMS_LAMBDA_EV=0.0
    export STEMS_ALPHA_EV_SMART=1.5
    export STEMS_ALPHA_EV_GUARD=1.0
    export CITYLEARN_EV_SAUTE=0
    export CITYLEARN_C0_DENSE_MERGE=0
    nohup $TRAIN_CMD --cfg configs/on-policy/r27d_cmdp.yaml $TRAIN_ARGS \
        > /tmp/r27d.log 2>&1 &
    echo "  PID: $!  Log: /tmp/r27d.log"
)

# --- R27e: Dense C0 via Saute merge ---
echo "[R27e] Dense C0 via Saute merge"
(
    export STEMS_ALPHA_GRID=1.5
    export STEMS_ALPHA_BUILD=1.5
    export STEMS_LAMBDA_EV=0.0
    export STEMS_ALPHA_EV_SMART=0.0
    export STEMS_ALPHA_EV_GUARD=2.0
    export CITYLEARN_EV_SAUTE=1
    export CITYLEARN_C0_DENSE_MERGE=1
    nohup $TRAIN_CMD --cfg configs/on-policy/r27e_cmdp.yaml $TRAIN_ARGS \
        > /tmp/r27e.log 2>&1 &
    echo "  PID: $!  Log: /tmp/r27e.log"
)

# --- R27f: Moderate dual r_ev + r_ev_smart ---
echo "[R27f] Moderate dual r_ev + r_ev_smart"
(
    export STEMS_ALPHA_GRID=1.0
    export STEMS_ALPHA_BUILD=1.0
    export STEMS_LAMBDA_EV=2.0
    export STEMS_ALPHA_EV_SMART=1.0
    export STEMS_ALPHA_EV_GUARD=0.5
    export CITYLEARN_EV_SAUTE=0
    export CITYLEARN_C0_DENSE_MERGE=0
    nohup $TRAIN_CMD --cfg configs/on-policy/r27f_cmdp.yaml $TRAIN_ARGS \
        > /tmp/r27f.log 2>&1 &
    echo "  PID: $!  Log: /tmp/r27f.log"
)

echo ""
echo "============================================"
echo "  All 5 runs launched. Summary:"
echo "============================================"
echo "  R27b: r_ev=5.0 baseline        -> /tmp/r27b.log"
echo "  R27c: Pure Lagrangian Kp=5     -> /tmp/r27c.log"
echo "  R27d: r_ev_smart+guard+PID     -> /tmp/r27d.log"
echo "  R27e: Dense C0 Saute           -> /tmp/r27e.log"
echo "  R27f: Dual r_ev+r_ev_smart     -> /tmp/r27f.log"
echo ""
echo "Monitor all: tail -f /tmp/r27{b,c,d,e,f}.log"
echo "Check GPUs:  nvidia-smi"
echo ""
echo "NOTE: All runs use cuda:0. If you have multiple GPUs,"
echo "edit the YAML configs to distribute across devices."
