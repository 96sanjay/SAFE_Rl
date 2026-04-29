#!/usr/bin/env bash
# R26f: Minimal Reward CMDP + STEMS V3 (GCN-Transformer)
# HYPOTHESIS: 16-component reward fights the Lagrangian (positive baselines +46K/ep).
# Solution: Keep only 5 reward terms, let Lagrangian handle C2/C3/C4.
#   KEEP:    r_eco(1.0), r_ren(0.3), r_ramp(0.3), r_ev(2.0), r_ev_guard(2.0)
#   DISABLE: r_sg, r_sb, r_headroom, r_peak_shave, r_grid_mild, r_barrier,
#            r_nec_sign, r_load_shift, r_solar_store, r_ev_slack_arb, r_v2g_ctx
# STEMS fix: lr=1e-4, target_kl=0.12, batch=512
# 120 epochs (extended for 162K param encoder)
set -euo pipefail

PROJECT="/home/sanjay/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Environment ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# --- STEMS V3 Temporal ---
export STEMS_ENCODER_VERSION="v3"
export CITYLEARN_TEMPORAL_WINDOW="12"
export CITYLEARN_TEMPORAL_RICH="1"

# --- Thresholds ---
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# --- C3 controllable (import-only violations) ---
export CITYLEARN_C3_CONTROLLABLE="1"

# --- PID Lagrangian ---
export CITYLEARN_PID_LAGRANGE="1"

# =====================================================================
# MINIMAL REWARD: 5 components (down from 16)
# =====================================================================

# KEEP: r_eco — PRIMARY economic objective (minimize bill)
export STEMS_MU_ECONOMIC="1.0"

# KEEP: r_ren — renewable utilization (no constraint overlap)
export STEMS_XI_RENEWABLE="0.3"

# KEEP: r_ramp — operation smoothness (no constraint overlap)
export STEMS_BETA_RAMP="0.3"

# KEEP: r_ev — dense EV charging urgency (reinforces C0, not duplicates it)
export STEMS_LAMBDA_EV="2.0"

# KEEP: r_ev_guard — prevents bad EV discharge (immediate penalty)
export STEMS_ALPHA_EV_GUARD="2.0"

# =====================================================================
# DISABLE: 11 components that duplicate C2/C3/C4 constraints
# =====================================================================

# r_sg: duplicates C4 + has +2.0/step positive baseline → DISABLE
export STEMS_ALPHA_GRID="0.0"
export STEMS_SG_THRESHOLD="0.0"
export STEMS_SG_EXPORT_CREDIT="0.0"

# r_sb: duplicates C3 + has +3.0/step positive baseline → DISABLE
export STEMS_ALPHA_BUILD="0.0"
export STEMS_SB_ASYMMETRIC="1"

# r_headroom: duplicates C3 → DISABLE
export STEMS_ALPHA_HEADROOM="0.0"

# r_peak_shave: duplicates C3 → DISABLE
export STEMS_ALPHA_PEAK_SHAVE="0.0"

# r_grid_mild: duplicates C4 → DISABLE
export STEMS_ALPHA_GRID_MILD="0.0"

# r_barrier: duplicates C2 → DISABLE
export STEMS_ALPHA_BARRIER="0.0"

# r_nec_sign: duplicates C3/C4 → DISABLE
export STEMS_ALPHA_NEC_SIGN="0.0"

# r_load_shift: duplicates C2/C3 → DISABLE
export STEMS_ALPHA_LOAD_SHIFT="0.0"

# r_solar_store: potential C3 conflict → DISABLE
export STEMS_ALPHA_SOLAR_STORE="0.0"

# r_ev_slack_arb: duplicates C0 → DISABLE
export STEMS_EV_SLACK_ARB_SCALE="0.0"

# r_v2g_ctx: duplicates C0/C4 overlap → DISABLE
export STEMS_ALPHA_V2G_CONTEXT="0.0"

# Already disabled (default 0)
export STEMS_ALPHA_EV_SOLAR="0.0"
export STEMS_ALPHA_GRID_PENALTY="0.0"
export STEMS_ALPHA_PRICE_ARB="0.0"

# --- No clamps, no masks, no Sauté ---
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="0"
export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_POLICY_ACTION_MASK="0"
export CITYLEARN_EV_SAUTE="0"
export CITYLEARN_WM_DISABLE="1"

# --- Cost weights ---
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="1.0"
export CITYLEARN_W_COST_BUILDING="1.0"
export CITYLEARN_W_COST_GRID="1.0"
export CITYLEARN_EV_COST_SCALE="1.0"
export CITYLEARN_INCLUDE_EV_COST="1"

echo ""
echo "================================================"
echo "  R26f: Minimal Reward CMDP + STEMS V3"
echo "  KEEP:    r_eco(1.0) r_ren(0.3) r_ramp(0.3)"
echo "           r_ev(2.0) r_ev_guard(2.0)"
echo "  DISABLE: 11 components (r_sg r_sb r_headroom"
echo "           r_peak r_grid_mild r_barrier r_nec_sign"
echo "           r_load r_solar r_slack r_v2g_ctx)"
echo "  STEMS:   lr=1e-4 target_kl=0.12 batch=512"
echo "  EPOCHS:  120 (extended for 162K params)"
echo "  Phase 1: ep 0-19 (C0 disabled, V2G discovery)"
echo "  Phase 2: ep 20-40 (C0 anneals 999999→50)"
echo "  Phase 3: ep 40-119 (all active, fine-tune)"
echo "================================================"
echo ""

/home/sanjay/miniconda3/envs/citylearn/bin/python scripts/train_multi_lag_stems.py \
    --cfg configs/on-policy/r26f_minimal_reward.yaml \
    --hidden_dim 64 \
    --output_dim 256 \
    --num_gcn_layers 3
