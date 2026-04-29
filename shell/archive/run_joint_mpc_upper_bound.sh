#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$ROOT"

export CITYLEARN_SCHEMA="${CITYLEARN_SCHEMA:-$ROOT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json}"

python scripts/evaluate_joint_mpc_upper_bound.py \
  --schema "$CITYLEARN_SCHEMA" \
  --horizon "${MPC_HORIZON:-12}" \
  --control-interval "${MPC_CONTROL_INTERVAL:-4}" \
  --ev-efficiency "${MPC_EV_EFFICIENCY:-0.95}" \
  --batt-efficiency "${MPC_BATT_EFFICIENCY:-0.95}" \
  --w-ev-slack "${MPC_W_EV_SLACK:-10000}" \
  --w-c3-slack "${MPC_W_C3_SLACK:-2500}" \
  --w-c4-slack "${MPC_W_C4_SLACK:-5000}" \
  --w-grid-cost "${MPC_W_GRID_COST:-10}" \
  --w-cycle-batt "${MPC_W_CYCLE_BATT:-1}" \
  --w-cycle-ev "${MPC_W_CYCLE_EV:-0.5}" \
  --solver "${MPC_SOLVER:-CLARABEL}" \
  --output-json "${MPC_OUTPUT_JSON:-$ROOT/runs/joint_mpc_upper_bound.json}"
