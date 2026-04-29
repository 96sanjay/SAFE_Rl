#!/bin/bash
# R26g: STEMS Critic + 2-Layer Temporal + Partial Obs Norm
# =========================================================
# Key fixes over R26f:
#   --stems_critic:      STEMS encoder for critics (not just actor)
#   --temporal_layers 2: 2-layer transformer (compositional reasoning)
#   --temporal_pool mean: mean-pool over all positions (not just last)
#   --partial_obs_norm:  don't normalize temporal history dims
#
# Full reward suite restored (R26e config, not R26f minimal)
# =========================================================
set -euo pipefail

export CITYLEARN_SCHEMA="$PWD/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_TEMPORAL_RICH=1
export CITYLEARN_TEMPORAL_WINDOW=12
export STEMS_ENCODER_VERSION=v3

# Reward weights: full suite (matching R26e, not R26f minimal)
export STEMS_ALPHA_ECO=0.3
export STEMS_ALPHA_GRID=2.0
export STEMS_ALPHA_BUILD=3.0
export STEMS_ALPHA_RAMP=0.3
export STEMS_ALPHA_RENEW=0.3
export STEMS_ALPHA_EV=2.5
export STEMS_ALPHA_BARRIER=1.0

# EV shaping
export CITYLEARN_EV_GUARD=3.0
export CITYLEARN_V2G_CONTEXT=2.0
export CITYLEARN_PEAK_SHAVE=3.0

# R16 reward extras
export CITYLEARN_LOAD_SHIFT=1.0
export CITYLEARN_GRID_MILD=0.5

# No clamps — pure RL
export CITYLEARN_BATT_CLAMP=0
export CITYLEARN_EV_CLAMP=0

# Cost coefficients
export CITYLEARN_C3_COST=0.1
export CITYLEARN_C4_COST=5.0

# Controllable C3: only penalize agent's marginal power, not exogenous load
export CITYLEARN_C3_CONTROLLABLE=1

echo "=== R26g: STEMS Critic + 2-Layer Transformer ==="
echo "  Fixes: STEMS critic, 2-layer temporal, mean-pool, partial obs norm"
echo "  Reward: full suite (eco=0.3 grid=2.0 build=3.0 ev=2.5 barrier=1.0)"
echo ""

nohup python scripts/train_multi_lag_stems.py \
    --cfg configs/on-policy/r26g_stems_critic.yaml \
    --hidden_dim 64 \
    --output_dim 256 \
    --num_gcn_layers 3 \
    --stems_critic \
    --temporal_layers 2 \
    --temporal_pool mean \
    --partial_obs_norm \
    > /tmp/r26g_stems_critic.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/r26g_stems_critic.log"
echo "Monitor: tail -f /tmp/r26g_stems_critic.log"
