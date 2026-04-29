#!/usr/bin/env bash
# Benchmark: CPO, FOCOPS, CUP on temperature cooling-only case study
# NO action masking — pure algorithm comparison
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# NO masking for fair algorithm comparison
export CITYLEARN_ACTION_MASK="0"
export CITYLEARN_POLICY_ACTION_MASK="0"

for algo in cpo focops cup; do
    echo ""
    echo "============================================"
    echo "  Benchmark: ${algo^^}"
    echo "  No masking — pure algorithm comparison"
    echo "============================================"
    echo ""

    python scripts/train_benchmark_temp_cooling_only.py \
        --cfg "configs/on-policy/${algo}_temp_cooling_only.yaml" \
        2>&1 | tee "/tmp/${algo}_temp_cooling_only.log"

    echo ""
    echo "  ${algo^^} training complete."
    echo ""
done

echo "All benchmarks complete."
