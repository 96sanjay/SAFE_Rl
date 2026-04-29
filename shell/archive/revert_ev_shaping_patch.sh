#!/usr/bin/env bash
# Revert all patches from apply_ev_shaping_patch.sh
set -euo pipefail
PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

for f in citylearn_safe/safety_env_v3.py citylearn_safe/lookahead_psf_v2g.py; do
    if [ -f "${f}.bak_pre_evshaping" ]; then
        cp "${f}.bak_pre_evshaping" "$f"
        echo "Reverted: $f"
    else
        echo "No backup found for $f — nothing to revert"
    fi
done
echo "Revert complete. Training scripts (train_ev_shaping.py etc.) left in place."
