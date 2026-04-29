#!/usr/bin/env bash
# PPOLag + PID + BC — TIGHTENED v3
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_POLICY_ACTION_MASK="1"

echo ""
echo "============================================"
echo "  PPOLag + PID + BC: TIGHTENED v3"
echo "  force_trigger=24, deadband=0.1"
echo "  cost_limit=25, penalty_max=15, 60 epochs"
echo "============================================"
echo ""

python scripts/train_ppolag_temp_cooling_only.py \
    --cfg configs/on-policy/ppolag_temp_cooling_only_tight_v3.yaml \
    2>&1 | tee /tmp/ppolag_v3.log
