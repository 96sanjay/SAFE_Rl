#!/bin/bash
set -e

echo "=========================================="
echo "AGGREGATE CONSTRAINT TEST - 1 EPOCH"
echo "=========================================="

# Set environment variables
export PYTHONPATH="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork:$PYTHONPATH"
export CITYLEARN_SCHEMA="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"

# Economic settings
export CITYLEARN_EXPORT_FACTOR="0.7"
export CITYLEARN_REWARD_SCALE="1.0"
export CITYLEARN_EV_COST_SCALE="3.0"

# CRITICAL: Cost weights
export CITYLEARN_W_COST_SOC="50.0"
export CITYLEARN_W_COST_EV="3.0"
export CITYLEARN_W_COST_BUILDING="1.0"
export CITYLEARN_W_COST_GRID="1.0"

# STEMS constraint bounds (aggregate district-level)
export CITYLEARN_STEMS_SOC_LOW="0.20"
export CITYLEARN_STEMS_SOC_HIGH="0.80"

# KPI logging
export CITYLEARN_KPI_RUN_NAME="TEST_AGGREGATE_1EP"

# ✅ Run with new training script
python scripts/train_omnisafe_ev.py \
  --cfg configs/on-policy/ppo_lag_1ep_aggregate_test.yaml \
  2>&1 | tee logs/test_aggregate_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "TEST COMPLETE - Analyzing results..."
echo "=========================================="
