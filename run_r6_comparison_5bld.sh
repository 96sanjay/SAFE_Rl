#!/usr/bin/env bash
# R6 Comparison Experiment: 3 runs on 5 buildings (50 epochs each)
#
# Run 1: R5-A baseline (StopIter fix only)
# Run 2: R5-B baseline (broken StopIter)
# Run 3: R6 (StopIter + spatial obs + controllable C3)
#
# Usage: bash run_r6_comparison_5bld.sh
# Or run individual: bash run_r6_comparison_5bld.sh r5a|r5b|r6
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# --- Shared env vars (5-building schema) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Thresholds: P_building_max same, P_grid_max recalibrated for 5 buildings
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# Cost weights
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

# EV
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

RUN="${1:-all}"

run_r5a() {
    echo ""
    echo "=========================================="
    echo "  RUN 1/3: R5-A Baseline (5 buildings)"
    echo "  StopIter=60, NO spatial obs, NO controllable C3"
    echo "=========================================="
    export CITYLEARN_C3_CONTROLLABLE="0"
    export CITYLEARN_SPATIAL_OBS="0"
    python scripts/train_omnisafe.py --cfg configs/on-policy/r6_compare_r5a_5bld.yaml
}

run_r5b() {
    echo ""
    echo "=========================================="
    echo "  RUN 2/3: R5-B Baseline (5 buildings)"
    echo "  StopIter=1 (broken), NO spatial obs, NO controllable C3"
    echo "=========================================="
    export CITYLEARN_C3_CONTROLLABLE="0"
    export CITYLEARN_SPATIAL_OBS="0"
    python scripts/train_omnisafe.py --cfg configs/on-policy/r6_compare_r5b_5bld.yaml
}

run_r6() {
    echo ""
    echo "=========================================="
    echo "  RUN 3/3: R6 Full (5 buildings)"
    echo "  StopIter=60, spatial obs ON, controllable C3 ON"
    echo "=========================================="
    export CITYLEARN_C3_CONTROLLABLE="1"
    export CITYLEARN_SPATIAL_OBS="1"
    python scripts/train_omnisafe.py --cfg configs/on-policy/r6_compare_r6_5bld.yaml
}

case "$RUN" in
    r5a)  run_r5a ;;
    r5b)  run_r5b ;;
    r6)   run_r6 ;;
    all)
        run_r5a
        run_r5b
        run_r6
        echo ""
        echo "=========================================="
        echo "  ALL 3 RUNS COMPLETE"
        echo "  Results in: runs/r6_compare/"
        echo "  Compare with: tensorboard --logdir runs/r6_compare"
        echo "=========================================="
        ;;
    *)
        echo "Usage: bash run_r6_comparison_5bld.sh [r5a|r5b|r6|all]"
        exit 1
        ;;
esac
