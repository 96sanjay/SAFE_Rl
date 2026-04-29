#!/usr/bin/env bash
# PPOLag + PID Lagrangian + BC warm-start — NEVER-BLOCK v4
# Key insight: mask was blocking the policy's LEARNED preemptive cooling.
# Fix: never set bounds to [0,0] — always allow cooling, only force minimum when hot.
# Uses v2's proven PID gains + moderate mask settings.
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_POLICY_ACTION_MASK="1"

echo ""
echo "============================================"
echo "  PPOLag + PID + BC: NEVER-BLOCK v4"
echo "  Never block cooling — let policy cool"
echo "  preemptively when cost critic says so."
echo "  PID: Kp=1.0, Ki=0.1, cost_limit=40"
echo "============================================"
echo ""

python scripts/train_ppolag_temp_cooling_only.py \
    --cfg configs/on-policy/ppolag_temp_cooling_only_neverblock_v4.yaml \
    2>&1 | tee /tmp/ppolag_temp_cooling_only_neverblock_v4.log
