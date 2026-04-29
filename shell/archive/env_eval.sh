#!/usr/bin/env bash
set -e

# --------- Environment (A) ----------
export CITYLEARN_SCHEMA="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"

export CITYLEARN_STEMS_P_GRID_MAX="27.127751"
export CITYLEARN_STEMS_P_BUILDING_MAX="2.273834"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_zero"

# --------- Cost shaping (B) ----------
export CITYLEARN_EV_DENSE_COST_SCALE="1.0"

# If your safety_env_v3 reads these weights, keep them identical to training:
export CITYLEARN_W_COST_EV_DENSE="1.0"
export CITYLEARN_W_COST_EV_DEPARTURE="2.0"
export CITYLEARN_W_COST_BUILDING="1.5"
export CITYLEARN_W_COST_SOC="1.0"
export CITYLEARN_W_COST_GRID="1.0"

# If you used these scale factors during training, keep them for eval too:
export CITYLEARN_STEMS_BATTERY_COST_SCALE="8.775"
export CITYLEARN_STEMS_BUILDING_COST_SCALE="0.1313"
export CITYLEARN_STEMS_GRID_COST_SCALE="1.1396"
export CITYLEARN_EV_COST_SCALE="10.7898"

# Some older scripts used these names too:
export CITYLEARN_W_COST_EV="1.0"
