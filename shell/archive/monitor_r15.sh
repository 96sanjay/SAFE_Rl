#!/usr/bin/env bash
# Monitor R14/R15a/R15b training, auto-start R15c/R15d as slots open.
# Max 3 concurrent runs. When all 5 done, run comprehensive evaluation.
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

# Current PIDs (set by caller or detected)
PID_R14=${PID_R14:-232927}
PID_R15A=${PID_R15A:-232663}
PID_R15B=${PID_R15B:-233089}

# Track what's been started
R15C_STARTED=0
R15D_STARTED=0
PID_R15C=""
PID_R15D=""

# All PIDs to monitor
declare -A PIDS
PIDS[R14]=$PID_R14
PIDS[R15a]=$PID_R15A
PIDS[R15b]=$PID_R15B

is_alive() {
    kill -0 "$1" 2>/dev/null
}

count_alive() {
    local count=0
    for pid in "${PIDS[@]}"; do
        if [ -n "$pid" ] && is_alive "$pid"; then
            ((count++)) || true
        fi
    done
    echo $count
}

start_r15c() {
    echo "[$(date)] Starting R15c..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate citylearn
    nohup bash "$PROJECT/run_r15c.sh" > "$PROJECT/r15c_train.log" 2>&1 &
    PID_R15C=$!
    PIDS[R15c]=$PID_R15C
    R15C_STARTED=1
    echo "[$(date)] R15c started with PID $PID_R15C"
}

start_r15d() {
    echo "[$(date)] Starting R15d..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate citylearn
    nohup bash "$PROJECT/run_r15d.sh" > "$PROJECT/r15d_train.log" 2>&1 &
    PID_R15D=$!
    PIDS[R15d]=$PID_R15D
    R15D_STARTED=1
    echo "[$(date)] R15d started with PID $PID_R15D"
}

echo "[$(date)] === R15 Ablation Monitor ==="
echo "[$(date)] Tracking: R14(PID=$PID_R14) R15a(PID=$PID_R15A) R15b(PID=$PID_R15B)"
echo "[$(date)] Will auto-start: R15c, R15d as slots open (max 3 concurrent)"
echo "[$(date)] Checking every 60 seconds..."

while true; do
    alive=$(count_alive)

    # Start R15c if slot available and not started
    if [ $R15C_STARTED -eq 0 ] && [ "$alive" -lt 3 ]; then
        start_r15c
        alive=$(count_alive)
    fi

    # Start R15d if slot available and not started
    if [ $R15D_STARTED -eq 0 ] && [ "$alive" -lt 3 ]; then
        start_r15d
        alive=$(count_alive)
    fi

    # Check if ALL are done
    if [ $R15C_STARTED -eq 1 ] && [ $R15D_STARTED -eq 1 ] && [ "$alive" -eq 0 ]; then
        echo "[$(date)] === ALL 5 RUNS COMPLETE ==="
        echo "[$(date)] Starting comprehensive evaluation..."

        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate citylearn
        export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

        python "$PROJECT/scripts/eval_r15_ablation.py" 2>&1 | tee "$PROJECT/r15_eval.log"

        echo "[$(date)] === EVALUATION COMPLETE ==="
        echo "[$(date)] Results: $PROJECT/r15_eval.log"
        echo "[$(date)] JSON:    $PROJECT/runs/r15_ablation/evaluation/"
        break
    fi

    # Status report
    status=""
    for name in R14 R15a R15b R15c R15d; do
        pid="${PIDS[$name]:-}"
        if [ -n "$pid" ]; then
            if is_alive "$pid"; then
                # Get epoch from log
                logfile="$PROJECT/${name,,}_train.log"
                if [ -f "$logfile" ]; then
                    epoch=$(grep -oP 'Ep \K[0-9]+' "$logfile" 2>/dev/null | tail -1 || echo "?")
                    status+=" $name:ep$epoch"
                else
                    status+=" $name:running"
                fi
            else
                status+=" $name:DONE"
            fi
        else
            status+=" $name:pending"
        fi
    done
    echo "[$(date)] alive=$alive |$status"

    sleep 60
done
