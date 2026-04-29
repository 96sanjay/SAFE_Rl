#!/bin/bash
# =============================================================================
# Optuna PPO-Lag Hyperparameter Sweep — srv07 launcher
# =============================================================================
# Launches parallel workers, each running sequential trials.
# All workers share a single SQLite Optuna study (WAL mode) for coordination.
#
# Usage:
#   bash run_optuna_sweep.sh                   # defaults
#   bash run_optuna_sweep.sh my_study 44 4     # custom: study, workers, trials
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STUDY_NAME="${1:-r28_sweep}"
N_WORKERS="${2:-44}"
N_TRIALS_PER_WORKER="${3:-4}"

PROJECT="$(cd "$(dirname "$0")" && pwd)"
SWEEP_SCRIPT="${PROJECT}/scripts/optuna_ppo_sweep.py"
LOG_DIR="/tmp/optuna_sweep_${STUDY_NAME}"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
# Detect conda — srv07 uses sanjay's miniconda, local uses christmas's
if [ -f "/home/sanjay/miniconda3/bin/conda" ]; then
    eval "$(/home/sanjay/miniconda3/bin/conda shell.bash hook)"
elif [ -f "/home/christmas/miniconda3/bin/conda" ]; then
    eval "$(/home/christmas/miniconda3/bin/conda shell.bash hook)"
else
    echo "ERROR: conda not found"
    exit 1
fi
conda activate citylearn

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Force CPU — GPU reserved for top candidates later
export CUDA_VISIBLE_DEVICES=""

# Limit threads per process: 96 cores / 44 workers = ~2 threads each
# Leaves headroom for OS/Optuna overhead
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TORCH_NUM_THREADS=2

# ---------------------------------------------------------------------------
# Create directories
# ---------------------------------------------------------------------------
mkdir -p "${PROJECT}/runs/optuna_r28_sweep/${STUDY_NAME}"
mkdir -p "${PROJECT}/runs/optuna_r28_sweep/db"
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL_TRIALS=$((N_WORKERS * N_TRIALS_PER_WORKER))
echo "================================================================"
echo "  Optuna PPO-Lag Sweep: ${STUDY_NAME}"
echo "================================================================"
echo "  Workers:          ${N_WORKERS}"
echo "  Trials/worker:    ${N_TRIALS_PER_WORKER}"
echo "  Total trials:     ${TOTAL_TRIALS}"
echo "  Epochs/trial:     40"
echo "  Search params:    19 (4 limits, 5 PID, 9 rewards, 1 PPO)"
echo "  Device:           CPU"
echo "  Project:          ${PROJECT}"
echo "  Logs:             ${LOG_DIR}/worker_*.log"
echo "  Optuna DB:        ${PROJECT}/runs/optuna_r28_sweep/db/${STUDY_NAME}.db"
echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-initialize the Optuna study + enable WAL mode for SQLite concurrency
# ---------------------------------------------------------------------------
DB_PATH="${PROJECT}/runs/optuna_r28_sweep/db/${STUDY_NAME}.db"
STORAGE_URI="sqlite:///${DB_PATH}"
echo "Initializing study DB..."
python -c "
import optuna
optuna.create_study(
    study_name='${STUDY_NAME}',
    storage='${STORAGE_URI}',
    direction='minimize',
    load_if_exists=True,
)
print('Study initialized OK')
"
# Enable WAL mode for better concurrent read/write performance
python -c "
import sqlite3
conn = sqlite3.connect('${DB_PATH}')
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('PRAGMA busy_timeout=30000')
conn.close()
print('WAL mode enabled, busy_timeout=30s')
"

# ---------------------------------------------------------------------------
# Launch workers (staggered to avoid SQLite contention)
# ---------------------------------------------------------------------------
PIDS=()

# Trap to kill all workers on script termination
cleanup() {
    echo ""
    echo "Caught signal, killing all workers..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    echo "All workers killed."
}
trap cleanup EXIT INT TERM

for i in $(seq 1 "${N_WORKERS}"); do
    nohup python "${SWEEP_SCRIPT}" \
        --study-name "${STUDY_NAME}" \
        --storage "${STORAGE_URI}" \
        --n-trials "${N_TRIALS_PER_WORKER}" \
        --worker-id "${i}" \
        --timeout-hours 20.0 \
        > "${LOG_DIR}/worker_${i}.log" 2>&1 &
    PIDS+=($!)
    echo "  Started worker ${i} (PID: ${!})"
    sleep 0.5
done

echo ""
echo "All ${N_WORKERS} workers launched."
echo ""
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/worker_1.log"
echo "  ls ${PROJECT}/runs/optuna_r28_sweep/${STUDY_NAME}/ | wc -l"
echo ""
echo "Check study progress (requires optuna):"
echo "  python -c \"import optuna; s=optuna.load_study('${STUDY_NAME}', 'sqlite:///${PROJECT}/runs/optuna_r28_sweep/db/${STUDY_NAME}.db'); print(f'Completed: {len([t for t in s.trials if t.state.name==\\\"COMPLETE\\\"])}/{len(s.trials)}'); print(f'Best: {s.best_value:.1f}' if s.best_trial else 'No best yet')\""
echo ""
echo "Kill all workers:"
echo "  kill ${PIDS[*]}"
echo ""
echo "PIDs: ${PIDS[*]}"

# Wait for all workers to complete
echo ""
echo "Waiting for all workers to finish..."
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || ((FAILED++))
done
echo ""
echo "================================================================"
echo "  All workers done. Failed: ${FAILED}/${N_WORKERS}"
echo "================================================================"

# Disable the trap since we're done
trap - EXIT INT TERM
