#!/bin/bash
export CITYLEARN_SCHEMA="data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
export CITYLEARN_STEMS_PNORM_P="4.0"

python3 scripts/run_rbc_comparison_CORRECT.py \
    --run-name "RBC_AFTER_FIX" \
    --episodes 1 \
    --ev-mode "greedy"
