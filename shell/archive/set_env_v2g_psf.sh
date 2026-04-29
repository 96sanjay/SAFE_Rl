#!/bin/bash
# Environment variables for TRPOLag + PSF V2G training
# Source this before running: source set_env_v2g_psf.sh
export CITYLEARN_SCHEMA="data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
export CITYLEARN_CENTRAL_AGENT=1
export CITYLEARN_STEMS_SOC_LOW=0.0
export CITYLEARN_STEMS_SOC_HIGH=0.95
export CITYLEARN_STEMS_P_BUILDING_MAX=2.273834
export CITYLEARN_STEMS_P_GRID_MAX=27.127751
export CITYLEARN_W_COST_EV=1.0
export CITYLEARN_W_COST_SOC=0.1
export CITYLEARN_W_COST_BUILDING=0.002
export CITYLEARN_W_COST_GRID=0.5
export PSF_HORIZON=24
export PSF_VERBOSE=1
export SERL_PENALTY_WEIGHT=10.0
export STEMS_ALPHA_GRID=1.5
export STEMS_ALPHA_BUILD=1.0
export STEMS_BETA_RAMP=0.5
export STEMS_ALPHA_RENEW=1.0
export USE_FORECAST=1
echo "[set_env_v2g_psf] Environment configured:"
echo "  Algo: TRPOLag"
echo "  PSF: H=${PSF_HORIZON}, V2G=[-1,1]"
echo "  SE-RL penalty: w=${SERL_PENALTY_WEIGHT}"
echo "  STEMS: alpha_grid=${STEMS_ALPHA_GRID} (reduced from 3.0)"
echo "  Cost weights: C1=${CITYLEARN_W_COST_EV} C2=${CITYLEARN_W_COST_SOC} C3=${CITYLEARN_W_COST_BUILDING} C4=${CITYLEARN_W_COST_GRID}"
