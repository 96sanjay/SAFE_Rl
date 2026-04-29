#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CITYLEARN_SCHEMA="$ROOT_DIR/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_REWARD_TYPE=stems
export MASK_C4_ENABLED=1

python scripts/train_stems_sp_rl_single_agent.py \
  --cfg configs/on-policy/stems_sp_rl_single_agent_5bld_smoke.yaml
