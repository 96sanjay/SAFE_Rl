#!/bin/bash
source set_env_v2.sh

export PYTHONPATH=/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork:$PYTHONPATH

export CITYLEARN_W_COST_EV=1.0        # was 10.0
export CITYLEARN_W_COST_SOC=0.1       # was 1.0
export CITYLEARN_W_COST_BUILDING=0.002 # was 0.02
export CITYLEARN_W_COST_GRID=0.5      # was 5.0

export CITYLEARN_USE_SHIELD=0
export SHIELD_ENABLE_C3=0
export SHIELD_ENABLE_C4=0

echo "[set_env_baselines] Weights:"
echo "  C1 EV       = $CITYLEARN_W_COST_EV"
echo "  C2 SoC      = $CITYLEARN_W_COST_SOC"
echo "  C3 Building = $CITYLEARN_W_COST_BUILDING"
echo "  C4 Grid     = $CITYLEARN_W_COST_GRID"
echo "  Shield      = $CITYLEARN_USE_SHIELD"
