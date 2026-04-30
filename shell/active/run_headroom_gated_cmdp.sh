#!/bin/bash
# R27a: Headroom-Gated CMDP (departure-aware EV + dense C0)
# ==========================================================
# Key changes from R26j:
#   1. r_ev_smart=1.5 (NEW) — headroom-gated price signal, departure-aware
#   2. r_ev=0 + r_ev_guard=0 — OFF (r_ev_smart + Lagrangian handle EV)
#   3. Dense C0 via Sauté (C1 channel) — per-step corridor cost
#   4. penalty_max=35 (R26j saturated at 20)
#   5. C0 curriculum [200→20] over 0-20 epochs (faster)
#   6. r_trajectory=0 (REMOVED — conflicts with C0)
#   7. 40-epoch test run (check at epoch 30 if working)
# ==========================================================
set -euo pipefail

# Activate conda environment
eval "$(/home/sanjay/miniconda3/bin/conda shell.bash hook)"
conda activate citylearn

export CITYLEARN_SCHEMA="$PWD/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_TEMPORAL_RICH=1
export CITYLEARN_TEMPORAL_WINDOW=12
export STEMS_ENCODER_VERSION=v3

# ── Reward terms (REBALANCED for CMDP) ──
export STEMS_MU_ECONOMIC=0.0          # r_eco OFF
export STEMS_ALPHA_GRID=1.5           # r_sg ON
export STEMS_ALPHA_BUILD=1.5          # r_sb ON
export STEMS_BETA_RAMP=0.3            # r_ramp ON (new)
export STEMS_XI_RENEWABLE=0.3         # r_ren ON
export STEMS_LAMBDA_EV=0.0            # r_ev OFF — dense C0 Lagrangian handles charging
export STEMS_ALPHA_BARRIER=0.3        # r_barrier ON (reduced, C2 Lambda helps)

# ── NEW: Headroom-gated EV smart reward ──
export STEMS_ALPHA_EV_SMART=1.5       # departure-aware price signal + V2G
export STEMS_ALPHA_EV_GUARD=0.0       # OFF — redundant with dense C0

# ── REMOVED/OFF reward terms ──
export STEMS_ALPHA_TRAJECTORY=0.0     # OFF — conflicts with C0
export STEMS_TRAJ_EV_WEIGHT=0.0
export STEMS_ALPHA_V2G_CONTEXT=0.0    # OFF — r_ev_smart handles V2G
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

# ── Dense C0 DISABLED — sparse C0 only ──
export CITYLEARN_EV_SAUTE=0           # DISABLED — sparse C0 + Lagrangian only
export CITYLEARN_C0_DENSE_MERGE=0     # OFF

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

# PID Lagrangian
export CITYLEARN_PID_LAGRANGE=1

echo "=== R27a: Headroom-Gated CMDP (40-epoch test) ==="
echo "  Reward: r_sg=1.5 r_sb=1.5 r_ramp=0.3 r_ren=0.3 r_barrier=0.3"
echo "  EV:     r_ev=0 r_ev_guard=0 r_ev_smart=1.5 (pure CMDP — Lagrangian handles C0)"
echo "  C0: fixed limit=200, Kp=0.5, Ki=0.01 (sparse max ~700)"
echo "  C1: unused"
echo "  C3: limit=5000, Kp=0.5, Ki=0.05"
echo "  C4: limit=8000, Kp=0.3, Ki=0.03"
echo "  penalty_max=35"
echo ""

nohup python scripts/train_multi_lag_stems.py \
    --cfg configs/active/headroom_gated_cmdp.yaml \
    --hidden_dim 64 \
    --output_dim 256 \
    --num_gcn_layers 3 \
    --stems_critic \
    --temporal_layers 2 \
    --temporal_pool mean \
    --partial_obs_norm \
    > /tmp/headroom_gated_cmdp.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/headroom_gated_cmdp.log"
echo "Monitor: tail -f /tmp/headroom_gated_cmdp.log"
echo ""
echo "Check at epoch 20:"
echo "  Lambda_0 should be >0 (cost=685 >> limit=200 → immediate pressure)"
echo "  EpCost_0 should be dropping from ~685"
echo "  Reward/r_ev_smart should appear in logs (now registered)"
