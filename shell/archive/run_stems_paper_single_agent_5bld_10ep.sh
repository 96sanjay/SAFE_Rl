#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

python scripts/train_stems_paper_single_agent.py \
  --cfg configs/on-policy/stems_paper_single_agent_5bld_10ep.yaml
