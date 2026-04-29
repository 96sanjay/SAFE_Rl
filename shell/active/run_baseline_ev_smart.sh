#!/bin/bash
# R27d: R25b exact + r_ev_smart=2.0
# Sauté=OFF, C1=unused, 4 constraints only
# Lagrangian: exact R25b (upper_bound=3.0, C0 anneal, C2=4000, C3=3000, C4=1500)
set -euo pipefail

eval "$(/home/sanjay/miniconda3/bin/conda shell.bash hook)"
conda activate citylearn

export CITYLEARN_SCHEMA="$PWD/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_TEMPORAL_RICH=1
export CITYLEARN_TEMPORAL_WINDOW=12
export STEMS_ENCODER_VERSION=v3

# ── R25b exact reward terms ──
export STEMS_MU_ECONOMIC=0.0
export STEMS_ALPHA_GRID=0.0           # R25b (OFF)
export STEMS_ALPHA_BUILD=0.0          # R25b (OFF)
export STEMS_BETA_RAMP=0.3            # R25b
export STEMS_XI_RENEWABLE=0.2         # R25b
export STEMS_LAMBDA_EV=5.0            # R25b
export STEMS_ALPHA_BARRIER=0.5        # R25b
export STEMS_ALPHA_EV_GUARD=1.0       # R25b
export STEMS_ALPHA_V2G_CONTEXT=3.0    # R25b
export STEMS_EV_SLACK_ARB_SCALE=2.0   # R25b
export STEMS_ALPHA_GRID_MILD=0.3      # R25b
export STEMS_SB_ASYMMETRIC=1          # R25b
export STEMS_SG_EXPORT_CREDIT=0.5     # R25b
export STEMS_ALPHA_EV_SMART=2.0       # NEW (departure-aware price signal)

# ── OFF ──
export STEMS_ALPHA_TRAJECTORY=0.0
export STEMS_TRAJ_EV_WEIGHT=0.0
export STEMS_ALPHA_PEAK_SHAVE=0.0
export STEMS_ALPHA_LOAD_SHIFT=0.0
export STEMS_ALPHA_EV_SOLAR=0.0
export STEMS_ALPHA_SOLAR_STORE=0.0
export STEMS_ALPHA_HEADROOM=0.0
export STEMS_ALPHA_GRID_PENALTY=0.0
export STEMS_ALPHA_PRICE_ARB=0.0
export STEMS_ALPHA_NEC_SIGN=0.0

# ── Sauté OFF, C1 unused ──
export CITYLEARN_EV_SAUTE=0
export CITYLEARN_C0_DENSE_MERGE=0

export CITYLEARN_BATT_CLAMP=0
export CITYLEARN_EV_CLAMP=0
export CITYLEARN_STEMS_SOC_LOW=0.0
export CITYLEARN_STEMS_SOC_HIGH=0.95

# ── R25b cost settings ──
export CITYLEARN_EV_COST_SCALE=3.0
export CITYLEARN_C3_CONTROLLABLE=1
export CITYLEARN_PID_LAGRANGE=1

echo "=== R27d: R25b exact + r_ev_smart=2.0 (40-epoch) ==="
echo "  Reward: r_ev=5.0 r_ev_guard=1.0 r_v2g_ctx=3.0 r_ev_slack_arb=2.0"
echo "          r_grid_mild=0.3 r_ren=0.2 r_ramp=0.3 r_barrier=0.5"
echo "  NEW:    r_ev_smart=2.0"
echo "  Lagrangian: R25b exact (upper_bound=3.0)"
echo "  C0: anneal [999999,1800,20,40], Kp=5.0"
echo "  C1: unused (999999)"
echo ""

nohup python scripts/train_multi_lag_stems.py \
    --cfg configs/active/baseline_ev_smart.yaml \
    --hidden_dim 64 \
    --output_dim 256 \
    --num_gcn_layers 3 \
    --stems_critic \
    --temporal_layers 2 \
    --temporal_pool mean \
    --partial_obs_norm \
    > /tmp/baseline_ev_smart.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/baseline_ev_smart.log"
