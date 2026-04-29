#!/usr/bin/env bash
# =============================================================================
# Launch EV shaping experiment
# Run from project root:
#   bash run_ev_shaping_experiment.sh
# =============================================================================
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

echo "=============================================="
echo " EV Shaping Experiment — Launch"
echo "=============================================="

# Source environment variables
source set_env_ev_shaping.sh

# Verify patches applied
if ! grep -q "EV_REWARD_SHAPING_PATCH" citylearn_safe/safety_env_v3.py; then
    echo "ERROR: safety_env_v3.py not patched. Run apply_ev_shaping_patch.sh first."
    exit 1
fi
if ! grep -q "PSF_W_TRACK_ENVVAR_PATCH" citylearn_safe/lookahead_psf_v2g.py; then
    echo "ERROR: lookahead_psf_v2g.py not patched. Run apply_ev_shaping_patch.sh first."
    exit 1
fi

echo ""
echo "Starting training..."
echo "  Log dir: ./runs/ev_shaping/"
echo "  Use 'tail -f runs/ev_shaping/*/progress.csv' to monitor"
echo ""

# Run with nohup so it survives SSH disconnect
nohup python3 -u train_ev_shaping.py > runs/ev_shaping_stdout.log 2>&1 &
PID=$!
echo "  Training started with PID=$PID"
echo "  stdout -> runs/ev_shaping_stdout.log"
echo "  To follow: tail -f runs/ev_shaping_stdout.log"
echo ""
echo "  To stop: kill $PID"
echo "$PID" > runs/ev_shaping.pid
