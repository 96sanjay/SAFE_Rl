#!/usr/bin/env bash
# Multi-seed training for top-5 temperature case study algorithms
# Runs 2 new seeds (0, 1) for each algorithm, 1 at a time (sequential)
# CSAC-LB already has seeds 0,1,2,42 — we add seeds 7,13 instead
#
# Total: 10 runs (5 algos x 2 seeds), ~80 min each on-policy, ~80 min off-policy
# Memory: on-policy ~1 GB, off-policy ~1-2 GB (SAC-Lag 200K buffer, CSAC-LB 3M buffer)
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

LOG="/tmp/multiseed_top5.log"
echo "=== Multi-seed top-5 training started $(date) ===" | tee "$LOG"

COMPLETED=0
FAILED=0

run_one() {
    local ALGO="$1"
    local SEED="$2"
    local TRAINER="$3"
    local BASE_CFG="$4"
    local LOG_DIR="$5"

    echo "" | tee -a "$LOG"
    echo "============================================" | tee -a "$LOG"
    echo "  [$((COMPLETED+FAILED+1))/10] $ALGO seed=$SEED" | tee -a "$LOG"
    echo "  Started: $(date)" | tee -a "$LOG"
    echo "============================================" | tee -a "$LOG"

    # Generate per-seed config
    local SEED_CFG="/tmp/${ALGO}_seed_${SEED}.yaml"
    sed -e "s/^seed: .*/seed: ${SEED}/" \
        -e "s|log_dir: .*|log_dir: ${LOG_DIR}|" \
        "$BASE_CFG" > "$SEED_CFG"

    # For BC warm-start configs, also update the BC seed
    if grep -q "bc_cfgs:" "$SEED_CFG" 2>/dev/null; then
        sed -i "s/^\(  *\)seed: .*/\1seed: ${SEED}/" "$SEED_CFG"
    fi

    local START_TIME=$(date +%s)

    if python "$TRAINER" --cfg "$SEED_CFG" 2>&1 | tee -a "$LOG"; then
        local END_TIME=$(date +%s)
        local ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
        echo "  DONE: $ALGO seed=$SEED (${ELAPSED} min)" | tee -a "$LOG"
        COMPLETED=$((COMPLETED + 1))
    else
        local END_TIME=$(date +%s)
        local ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
        echo "  FAILED: $ALGO seed=$SEED (${ELAPSED} min)" | tee -a "$LOG"
        FAILED=$((FAILED + 1))
    fi
}

# ── 1. CPO — on-policy, no replay buffer, ~1 GB ──
for SEED in 0 1; do
    export CITYLEARN_ACTION_MASK="0"
    export CITYLEARN_POLICY_ACTION_MASK="0"
    run_one "CPO" "$SEED" \
        "scripts/train_benchmark_temp_cooling_only.py" \
        "configs/on-policy/cpo_temp_cooling_only.yaml" \
        "./runs/cpo_temp_cooling_only/seed_${SEED}"
done

# ── 2. CUP — on-policy, ~1 GB ──
for SEED in 0 1; do
    export CITYLEARN_ACTION_MASK="0"
    export CITYLEARN_POLICY_ACTION_MASK="0"
    run_one "CUP" "$SEED" \
        "scripts/train_benchmark_temp_cooling_only.py" \
        "configs/on-policy/cup_temp_cooling_only.yaml" \
        "./runs/cup_temp_cooling_only_v2/seed_${SEED}"
done

# ── 3. FOCOPS — on-policy, ~1 GB ──
for SEED in 0 1; do
    export CITYLEARN_ACTION_MASK="0"
    export CITYLEARN_POLICY_ACTION_MASK="0"
    run_one "FOCOPS" "$SEED" \
        "scripts/train_benchmark_temp_cooling_only.py" \
        "configs/on-policy/focops_temp_cooling_only.yaml" \
        "./runs/focops_temp_cooling_only_v2/seed_${SEED}"
done

# ── 4. SAC-Lag — off-policy, 200K buffer, ~1.5 GB ──
for SEED in 0 1; do
    export CITYLEARN_ACTION_MASK="0"
    export CITYLEARN_POLICY_ACTION_MASK="1"
    run_one "SAC-Lag" "$SEED" \
        "scripts/train_saclag_temp_cooling_only.py" \
        "configs/off-policy/saclag_temp_cooling_only_v1.yaml" \
        "./runs/saclag_temp_cooling_only_v1/seed_${SEED}"
done

# ── 5. CSAC-LB — off-policy, 3M buffer, ~2-3 GB ──
# Already has seeds 0,1,2,42. Adding seeds 7,13 for more diversity.
for SEED in 7 13; do
    run_one "CSAC-LB" "$SEED" \
        "scripts/train_csac_lb_temp.py" \
        "configs/off-policy/csac_lb_temp_multi_seed.yaml" \
        "./runs/csac_lb_multi_seed/seed_${SEED}"
done

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "  ALL DONE: $COMPLETED completed, $FAILED failed" | tee -a "$LOG"
echo "  Finished: $(date)" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "  Existing seeds:" | tee -a "$LOG"
echo "    CPO:     42, 0, 1" | tee -a "$LOG"
echo "    CUP:     42, 0, 1" | tee -a "$LOG"
echo "    FOCOPS:  42, 0, 1" | tee -a "$LOG"
echo "    SAC-Lag: 42, 0, 1" | tee -a "$LOG"
echo "    CSAC-LB: 0, 1, 2, 42, 7, 13" | tee -a "$LOG"
