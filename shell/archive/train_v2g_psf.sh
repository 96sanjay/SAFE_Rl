#!/bin/bash
# Launch TRPOLag + PSF V2G SE-RL training
set -e
source set_env_v2g_psf.sh
echo "[Train] Verifying imports..."
python3 -c "
from citylearn_safe.lookahead_psf_v2g import LookaheadPSFv2G
from citylearn_safe.omni_env_v2g_psf import CityLearnV2GPSF
print('[OK] All imports successful')
"
echo "[Train] Starting TRPOLag + PSF V2G training..."
echo "[Train] Estimated time: ~10 hours (100 epochs x 8759 steps)"
echo "[Train] PSF H=${PSF_HORIZON}, SE-RL w=${SERL_PENALTY_WEIGHT}, alpha_grid=${STEMS_ALPHA_GRID}"

mkdir -p runs/trpolag_v2g_psf

python3 train_trpolag_v2g_psf.py
    --config configs/on-policy/trpolag_v2g_psf.yaml \
    2>&1 | tee runs/trpolag_v2g_psf/train.log

echo "[Train] Done. Check runs/trpolag_v2g_psf/ for results."
