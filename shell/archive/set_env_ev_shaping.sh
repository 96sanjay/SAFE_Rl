#!/usr/bin/env bash
# Environment variables for EV shaping experiment
# Source this before running train_ev_shaping.py

# === REWARD ===
export CITYLEARN_REWARD_TYPE=stems
export STEMS_ALPHA_GRID=1.5
export STEMS_ALPHA_BUILD=1.0
export STEMS_BETA_RAMP=0.5
export STEMS_ALPHA_RENEW=1.0

# === KEY CHANGE 1: EV reward shaping ===
# Per-step bonus for charging urgent EVs
# 0.0 = off (default, won't affect other runs)
# 5.0 = strong signal for EV charging
export CITYLEARN_EV_SHAPING_ALPHA=5.0

# === KEY CHANGE 2: PSF w_track ===
# Lower = QP ignores agent more, but agent has more room to learn
# when its actions DO pass through
export PSF_W_TRACK=10.0

# === COST WEIGHTS (corrected names) ===
export CITYLEARN_W_COST_EV=1.0
export CITYLEARN_W_COST_SOC=0.1
export CITYLEARN_W_COST_BUILDING=0.002
export CITYLEARN_W_COST_GRID=0.5

# === STEMS constraint thresholds ===
export CITYLEARN_STEMS_SOC_LOW=0.0
export CITYLEARN_STEMS_SOC_HIGH=0.95
export CITYLEARN_STEMS_P_BUILDING_MAX=2.273834
export CITYLEARN_STEMS_P_GRID_MAX=27.127751
export CITYLEARN_STEMS_BATTERY_COST_SCALE=50.0

# === Other ===
export CITYLEARN_EXPORT_FACTOR=1.0
export CITYLEARN_REWARD_SCALE=1.0
export CITYLEARN_EV_COST_SCALE=1.0
export CITYLEARN_EV_MISSING_ACTION_MODE=assume_full
export CITYLEARN_COST_MODE=hinge

# KPI logging
export CITYLEARN_KPI_RUN_NAME=ev_shaping_experiment

echo "[set_env_ev_shaping] All env vars set for EV shaping experiment"
echo "  CITYLEARN_EV_SHAPING_ALPHA=$CITYLEARN_EV_SHAPING_ALPHA"
echo "  PSF_W_TRACK=$PSF_W_TRACK"
echo "  cost_limit=5.0 (in train_ev_shaping.py)"
