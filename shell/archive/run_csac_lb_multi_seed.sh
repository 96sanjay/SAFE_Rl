#!/usr/bin/env bash
# Multi-seed CSAC-LB replication study
# Tests whether <5% violation rate is robust across seeds
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CITYLEARN_USE_DEFAULT_TEMP_SCHEMA=1

BASE_CFG="configs/off-policy/csac_lb_temp_multi_seed.yaml"
SEEDS=(0 1 2 42)

for SEED in "${SEEDS[@]}"; do
    LOG_DIR="./runs/csac_lb_multi_seed/seed_${SEED}"
    echo ""
    echo "============================================"
    echo "  CSAC-LB Multi-Seed: seed=${SEED}"
    echo "  Log dir: ${LOG_DIR}"
    echo "============================================"
    echo ""

    # Create per-seed config by patching seed and log_dir
    SEED_CFG="/tmp/csac_lb_seed_${SEED}.yaml"
    sed -e "s/^seed: .*/seed: ${SEED}/" \
        -e "s|log_dir: .*|log_dir: ${LOG_DIR}|" \
        "$BASE_CFG" > "$SEED_CFG"

    # Train
    python scripts/train_csac_lb_temp.py --cfg "$SEED_CFG" \
        2>&1 | tee "/tmp/csac_lb_seed_${SEED}.log"

    # Find the run directory (contains progress.csv and torch_save/)
    RUN_DIR=$(find "$LOG_DIR" -name "progress.csv" -printf '%h\n' | head -1)
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: No progress.csv found in ${LOG_DIR}"
        continue
    fi

    echo ""
    echo "  Evaluating seed=${SEED}..."
    python scripts/evaluate_csaclb_temp_case_study.py \
        --run-dir "$RUN_DIR" \
        --seed "$SEED" \
        --output-json "${RUN_DIR}/eval_case_study.json"

    echo ""
    echo "  seed=${SEED} complete."
    echo ""
done

echo ""
echo "============================================"
echo "  All seeds complete. Summary:"
echo "============================================"

# Quick summary
python3 -c "
import json, glob
for path in sorted(glob.glob('runs/csac_lb_multi_seed/seed_*/*/seed-*/eval_case_study.json')):
    seed = path.split('/')[2]
    with open(path) as f:
        data = json.load(f)
    for row in data['rows'][:2]:
        vr = row.get('violation_rate', -1)
        rew = row.get('total_reward', 0)
        print(f'  {seed} {row[\"name\"]:<25} Viol={vr*100:.2f}%  Rew={rew:.1f}')
"
