#!/bin/bash
# R26i: Forecast-Aware Arbitrage — FULL (8 reward terms)
# ======================================================
# Full suite + forecast arbitrage:
#   r_eco, r_sg, r_sb, r_ren, r_ev (REDUCED 0.5), r_ev_guard,
#   r_barrier, r_trajectory
#
# Comparison: if R26h >> R26i, confirms myopic terms hurt temporal.
# C2 SoC range: 0.0–0.95 (was 0.05–0.95)
# ======================================================
set -euo pipefail

# Activate conda environment
eval "$(/home/sanjay/miniconda3/bin/conda shell.bash hook)"
conda activate citylearn

export CITYLEARN_SCHEMA="$PWD/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_TEMPORAL_RICH=1
export CITYLEARN_TEMPORAL_WINDOW=12
export STEMS_ENCODER_VERSION=v3

# ── 8 active reward terms (CORRECT env var names) ──
export STEMS_MU_ECONOMIC=0.3          # r_eco ON
export STEMS_ALPHA_GRID=2.0           # r_sg ON
export STEMS_ALPHA_BUILD=3.0          # r_sb ON
export STEMS_BETA_RAMP=0.0            # r_ramp OFF (disabled by mask)
export STEMS_XI_RENEWABLE=0.3         # r_ren ON
export STEMS_LAMBDA_EV=0.5            # r_ev ON but REDUCED (was 2.5)
export STEMS_ALPHA_BARRIER=1.0        # r_barrier ON
export STEMS_ALPHA_TRAJECTORY=2.0     # r_trajectory ON (forecast arbitrage)
export STEMS_ALPHA_EV_GUARD=1.5       # r_ev_guard ON (was 3.0, reduced)

# All other reward terms explicitly OFF
export STEMS_ALPHA_V2G_CONTEXT=0.0
export STEMS_ALPHA_PEAK_SHAVE=0.0
export STEMS_ALPHA_LOAD_SHIFT=0.0     # OFF — replaced by r_trajectory
export STEMS_ALPHA_GRID_MILD=0.0
export STEMS_ALPHA_EV_SOLAR=0.0
export STEMS_ALPHA_SOLAR_STORE=0.0
export STEMS_EV_SLACK_ARB_SCALE=0.0
export STEMS_ALPHA_HEADROOM=0.0
export STEMS_ALPHA_GRID_PENALTY=0.0
export STEMS_ALPHA_PRICE_ARB=0.0      # OFF — replaced by r_trajectory
export STEMS_ALPHA_NEC_SIGN=0.0

# Forecast arbitrage sub-weights
export STEMS_TRAJ_EV_WEIGHT=1.0
export STEMS_TRAJ_FORECAST_HOURS=24

# No clamps — pure RL
export CITYLEARN_BATT_CLAMP=0
export CITYLEARN_EV_CLAMP=0

# C2 SoC range: 0.0–0.95 (allow full SoC range)
export CITYLEARN_STEMS_SOC_LOW=0.0
export CITYLEARN_STEMS_SOC_HIGH=0.95

# Cost coefficients
export CITYLEARN_C3_COST=0.1
export CITYLEARN_C4_COST=5.0
export CITYLEARN_C3_CONTROLLABLE=1

echo "=== R26i: Forecast-Aware Arbitrage — FULL ==="
echo "  8 terms: r_eco=0.3 r_sg=2.0 r_sb=3.0 r_ren=0.3 r_ev=0.5 r_ev_guard=1.5 r_barrier=1.0 r_trajectory=2.0"
echo "  OFF: r_load_shift, r_price_arb, r_nec_sign, r_v2g_ctx, r_peak_shave"
echo "  C2 SoC: 0.0–0.95"
echo ""

nohup python scripts/train_multi_lag_stems.py \
    --cfg configs/on-policy/r26i_forecast_full.yaml \
    --hidden_dim 64 \
    --output_dim 256 \
    --num_gcn_layers 3 \
    --stems_critic \
    --temporal_layers 2 \
    --temporal_pool mean \
    --partial_obs_norm \
    > /tmp/r26i_forecast_full.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/r26i_forecast_full.log"
echo "Monitor: tail -f /tmp/r26i_forecast_full.log"
