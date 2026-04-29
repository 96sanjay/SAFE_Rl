#!/usr/bin/env bash
# PPOLag + PID Lagrangian + BC warm-start — TIGHTENED v2
# Key changes: force_trigger=25, Kp=1.0, Ki=0.1, cost_limit=40, 60 epochs
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_POLICY_ACTION_MASK="1"

echo ""
echo "============================================"
echo "  PPOLag + PID + BC: TIGHTENED v2"
echo "  force_trigger=25, Kp=1.0, Ki=0.1"
echo "  cost_limit=40, penalty_max=10, 60 epochs"
echo "============================================"
echo ""

python scripts/train_ppolag_temp_cooling_only.py \
    --cfg configs/on-policy/ppolag_temp_cooling_only_tight_v2.yaml \
    2>&1 | tee /tmp/ppolag_temp_cooling_only_tight_v2.log
