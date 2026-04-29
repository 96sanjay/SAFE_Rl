#!/usr/bin/env bash
# R21: PPOLagMulti — Targeted fixes on top of R19
#
# Root causes fixed (all identified from R19 analysis):
#   1. actor_lr 0.0003 → 0.0001   (StopIter=1 in 52/80 R19 epochs = 65% wasted)
#   2. target_kl 0.12 → 0.08      (tighter trust region, paired with lower LR)
#   3. batch_size 128 → 256        (stable gradient estimates, less KL per pass)
#   4. lambda_upper_bound 3.0→5.0  (λ_3/λ_4 saturated at 3.0 since epoch 1-3)
#   5. BATT_CLAMP=1               (Lambda_2 grew 0.073→2.153, stealing C3/C4 budget)
#
# Scientific basis for #5 (thesis framing):
#   Battery SoC is a LINEAR constraint in action space: SoC_next = SoC + action*eta
#   → Feasible action interval is analytically computable at each step
#   → Safety Projection Layer (Dalal et al. 2018) is optimal; no Lagrangian needed
#   → Freeing Lambda_2 budget redirects gradient pressure to C3/C4
#
# Unchanged from R19:
#   - MLP architecture (beats STEMS on C4: 275 vs 156 steps/ep decline rate)
#   - Same curriculum schedule (Phase 2 ep 20-40)
#   - Same PID gains, same cost limits
#   - Same reward config (zeroed r_load_shift, r_sg, reduced r_ev_guard)
#   - 90 epochs (80 + 10 extra for lower LR warmup)
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Base env (IDENTICAL to R19) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Thresholds (IDENTICAL to R19)
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# PID Lagrangian (IDENTICAL to R19)
export CITYLEARN_PID_LAGRANGE="1"

# Sauté MDP for C1 (IDENTICAL to R19)
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

# Reward (IDENTICAL to R19 — zeroed conflicting signals)
export STEMS_ALPHA_GRID="0.0"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_ALPHA_LOAD_SHIFT="0.0"
export STEMS_ALPHA_GRID_MILD="0.3"
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_XI_RENEWABLE="0.2"
export STEMS_BETA_RAMP="0.3"

# C3_CONTROLLABLE (IDENTICAL to R19)
export CITYLEARN_C3_CONTROLLABLE="1"

# V2G reward signals (IDENTICAL to R19)
export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_GUARD="1.0"
export STEMS_ALPHA_V2G_CONTEXT="3.0"

# Cost weights (IDENTICAL to R19)
export COST_W_C2="0.0"
export COST_W_C3="5.0"
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"

# =====================================================================
# R21 FIX #5: BATT_CLAMP ON
# Battery SoC is a linear constraint → Safety Projection (Dalal 2018)
# In R19: Lambda_2 grew 0.073→2.153 despite COST_W_C2=0.0
# (PPOLagMulti reads cost_stems_battery from info dict directly)
# Clamping at env level removes C2 from the CMDP entirely,
# freeing the Lambda budget for C3/C4.
# =====================================================================
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="1"          # R21: 0→1 (KEY FIX for Lambda_2 bloat)

echo ""
echo "============================================"
echo "  R21: PPOLagMulti — Targeted R19 Fixes"
echo "  Fix 1: actor_lr 0.0003→0.0001 (stop 65% wasted epochs)"
echo "  Fix 2: target_kl 0.12→0.08 (tighter trust region)"
echo "  Fix 3: batch_size 128→256 (stable gradients)"
echo "  Fix 4: lambda_cap 3.0→5.0 (prevent Phase 3 reversal)"
echo "  Fix 5: BATT_CLAMP=1 (Safety Projection for C2 — Dalal 2018)"
echo "  Architecture: MLP (beats STEMS on C4 by 1.76x)"
echo "  90 epochs (80 + 10 for lower LR warmup)"
echo "  Phase 1 (ep 0-19):  C0 disabled"
echo "  Phase 2 (ep 20-40): C0 anneals 999999→1800"
echo "  Phase 3 (ep 40-89): All constraints, full λ pressure"
echo "============================================"
echo ""

/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_multi_lag.py \
    --cfg configs/on-policy/r21_ppo.yaml
