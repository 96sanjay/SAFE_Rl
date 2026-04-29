#!/usr/bin/env bash
# SAC-Lag + PID Lagrangian + Action Masking + BC warm-start
# Temperature cooling-only case study
# Key advantage: Q(s,a) critic predicts future violations better than PPO's V(s)
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_POLICY_ACTION_MASK="1"

echo ""
echo "============================================"
echo "  SAC-Lag + PID + Masking + BC"
echo "  Q(s,a) critic for better prediction"
echo "  PID: Kp=1.0, Ki=0.1, cost_limit=40"
echo "  Twin cost critics (conservative)"
echo "============================================"
echo ""

python scripts/train_saclag_temp_cooling_only.py \
    --cfg configs/off-policy/saclag_temp_cooling_only_v1.yaml \
    2>&1 | tee /tmp/saclag_temp_cooling_only_v1.log
