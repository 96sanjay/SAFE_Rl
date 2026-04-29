#!/bin/bash

# Exit on error
set -e

echo "=========================================="
echo "PPOLag V2 Training: Fixed Architecture"
echo "=========================================="

# Environment thresholds (P95 calibrated)
export CITYLEARN_STEMS_P_GRID_MAX="27.127751"
export CITYLEARN_STEMS_P_BUILDING_MAX="2.273834"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_zero"

# Balanced cost scales
export CITYLEARN_STEMS_BATTERY_COST_SCALE="8.775"
export CITYLEARN_STEMS_BUILDING_COST_SCALE="0.1313"
export CITYLEARN_STEMS_GRID_COST_SCALE="1.1396"
export CITYLEARN_EV_COST_SCALE="10.7898"

# KEY FIX: Reduce EV dense cost from 5.0 to 1.0
export CITYLEARN_EV_DENSE_COST_SCALE="1.0"

# Equal constraint weights
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="1.0"
export CITYLEARN_W_COST_BUILDING="1.0"
export CITYLEARN_W_COST_GRID="1.0"

# Schema path - UPDATE THIS!
export CITYLEARN_SCHEMA="${CITYLEARN_SCHEMA:-/path/to/your/schema.json}"

# Logging
export CITYLEARN_KPI_RUN_NAME="PPOLag_V2_Deep512_Batch512_Dense1_Limit45k"

echo "Configuration:"
echo "  Cost limit: 45000"
echo "  EV dense scale: 1.0 (was 5.0)"
echo "  Architecture: [512, 512, 256] (deep network)"
echo "  Batch size: 512"
echo "  Schema: $CITYLEARN_SCHEMA"
echo ""

# Start training
python scripts/train_omnisafe.py \
  --cfg configs/on-policy/ppolag_P95_V2.yaml \
  2>&1 | tee logs/PPOLag_V2_Deep512_Batch512.log

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Check: runs/ppolag_P95_V2/"
echo "=========================================="
