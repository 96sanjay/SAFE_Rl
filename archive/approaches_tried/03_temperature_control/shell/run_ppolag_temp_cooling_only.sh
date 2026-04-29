#!/usr/bin/env bash
# PPOLag + PID Lagrangian + BC warm-start for temperature case study
# Fixes all 5 root causes: 3D env, BC warmstart, PID lambda, aggressive gains, 40ep
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Temperature case study env settings
export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_POLICY_ACTION_MASK="1"

echo ""
echo "============================================"
echo "  PPOLag + PID + BC: Temperature Case Study"
echo "  Env: CityLearnTemp-CoolingOnly-Masked-Reward-v0"
echo "  40 epochs, cost_limit=55, PID Kp=0.5"
echo "============================================"
echo ""

python scripts/train_ppolag_temp_cooling_only.py \
    --cfg configs/on-policy/ppolag_temp_cooling_only_pid_bc_40ep.yaml \
    2>&1 | tee /tmp/ppolag_temp_cooling_only.log
