#!/usr/bin/env bash
# Multi-seed evaluation for top-5 temperature case study algorithms
# Evaluates best+final checkpoints for each seed, saves per-seed JSON,
# then aggregates into a single summary JSON.
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CITYLEARN_USE_DEFAULT_TEMP_SCHEMA=1

LOG="/tmp/multiseed_eval.log"
SUMMARY_JSON="runs/multiseed_eval_summary.json"

echo "=== Multi-seed evaluation started $(date) ===" | tee "$LOG"

# ── CPO: seeds 42, 0, 1 ──
declare -A CPO_DIRS=(
    [42]="runs/cpo_temp_cooling_only/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-20-54-15"
    [0]="runs/cpo_temp_cooling_only/seed_0/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-12-34-57"
    [1]="runs/cpo_temp_cooling_only/seed_1/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-12-58-45"
)

# ── CUP: seeds 42, 0, 1 ──
declare -A CUP_DIRS=(
    [42]="runs/cup_temp_cooling_only_v2/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-23-10-06"
    [0]="runs/cup_temp_cooling_only_v2/seed_0/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-13-22-49"
    [1]="runs/cup_temp_cooling_only_v2/seed_1/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-13-46-22"
)

# ── FOCOPS: seeds 42, 0, 1 ──
declare -A FOCOPS_DIRS=(
    [42]="runs/focops_temp_cooling_only_v2/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-22-44-31"
    [0]="runs/focops_temp_cooling_only_v2/seed_0/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-14-09-24"
    [1]="runs/focops_temp_cooling_only_v2/seed_1/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-14-35-00"
)

# ── SAC-Lag: seeds 42, 0, 1 ──
declare -A SACLAG_DIRS=(
    [42]="runs/saclag_temp_cooling_only_v1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-15-33-32"
    [0]="runs/saclag_temp_cooling_only_v1/seed_0/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-15-00-17"
    [1]="runs/saclag_temp_cooling_only_v1/seed_1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-15-45-24"
)

# ── CSAC-LB: seeds 0, 1, 2, 7, 13, 42 ──
declare -A CSACLB_DIRS=(
    [0]="runs/csac_lb_multi_seed/seed_0/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-09-10-08-23"
    [1]="runs/csac_lb_multi_seed/seed_1/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-09-11-15-54"
    [2]="runs/csac_lb_multi_seed/seed_2/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-002-2026-04-09-12-26-54"
    [7]="runs/csac_lb_multi_seed/seed_7/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-007-2026-04-10-16-33-51"
    [13]="runs/csac_lb_multi_seed/seed_13/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-013-2026-04-10-17-40-31"
    [42]="runs/csac_lb_multi_seed/seed_42/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-09-13-41-32"
)

eval_ppo_seed() {
    local ALGO="$1"
    local SEED="$2"
    local RUN_DIR="$3"
    local OUT_JSON="${RUN_DIR}/eval_case_study_multiseed.json"

    echo "  Evaluating $ALGO seed=$SEED ..." | tee -a "$LOG"
    python scripts/evaluate_ppo_temp_case_study.py \
        --run-dir "$RUN_DIR" \
        --seed "$SEED" \
        --output-json "$OUT_JSON" \
        >> "$LOG" 2>&1
    echo "  Done: $ALGO seed=$SEED -> $OUT_JSON" | tee -a "$LOG"
}

eval_csaclb_seed() {
    local SEED="$1"
    local RUN_DIR="$2"
    local OUT_JSON="${RUN_DIR}/eval_case_study_multiseed.json"

    echo "  Evaluating CSAC-LB seed=$SEED ..." | tee -a "$LOG"
    python scripts/evaluate_csaclb_temp_case_study.py \
        --run-dir "$RUN_DIR" \
        --seed "$SEED" \
        --output-json "$OUT_JSON" \
        >> "$LOG" 2>&1
    echo "  Done: CSAC-LB seed=$SEED -> $OUT_JSON" | tee -a "$LOG"
}

# ── Run evaluations ──

echo "" | tee -a "$LOG"
echo "--- CPO ---" | tee -a "$LOG"
for SEED in 42 0 1; do
    eval_ppo_seed "CPO" "$SEED" "${CPO_DIRS[$SEED]}"
done

echo "" | tee -a "$LOG"
echo "--- CUP ---" | tee -a "$LOG"
for SEED in 42 0 1; do
    eval_ppo_seed "CUP" "$SEED" "${CUP_DIRS[$SEED]}"
done

echo "" | tee -a "$LOG"
echo "--- FOCOPS ---" | tee -a "$LOG"
for SEED in 42 0 1; do
    eval_ppo_seed "FOCOPS" "$SEED" "${FOCOPS_DIRS[$SEED]}"
done

echo "" | tee -a "$LOG"
echo "--- SAC-Lag ---" | tee -a "$LOG"
for SEED in 42 0 1; do
    eval_ppo_seed "SAC-Lag" "$SEED" "${SACLAG_DIRS[$SEED]}"
done

echo "" | tee -a "$LOG"
echo "--- CSAC-LB ---" | tee -a "$LOG"
for SEED in 0 1 2 7 13 42; do
    eval_csaclb_seed "$SEED" "${CSACLB_DIRS[$SEED]}"
done

# ── Aggregate all JSON into summary ──
echo "" | tee -a "$LOG"
echo "--- Aggregating results ---" | tee -a "$LOG"

python3 - <<'PYEOF'
import json, glob, statistics, os
from pathlib import Path

PROJECT = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")

ALGO_SEEDS = {
    "CPO": [
        ("42", "runs/cpo_temp_cooling_only/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-20-54-15"),
        ("0",  "runs/cpo_temp_cooling_only/seed_0/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-12-34-57"),
        ("1",  "runs/cpo_temp_cooling_only/seed_1/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-12-58-45"),
    ],
    "CUP": [
        ("42", "runs/cup_temp_cooling_only_v2/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-23-10-06"),
        ("0",  "runs/cup_temp_cooling_only_v2/seed_0/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-13-22-49"),
        ("1",  "runs/cup_temp_cooling_only_v2/seed_1/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-13-46-22"),
    ],
    "FOCOPS": [
        ("42", "runs/focops_temp_cooling_only_v2/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-22-44-31"),
        ("0",  "runs/focops_temp_cooling_only_v2/seed_0/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-14-09-24"),
        ("1",  "runs/focops_temp_cooling_only_v2/seed_1/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-14-35-00"),
    ],
    "SAC-Lag": [
        ("42", "runs/saclag_temp_cooling_only_v1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-15-33-32"),
        ("0",  "runs/saclag_temp_cooling_only_v1/seed_0/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-10-15-00-17"),
        ("1",  "runs/saclag_temp_cooling_only_v1/seed_1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-10-15-45-24"),
    ],
    "CSAC-LB": [
        ("0",  "runs/csac_lb_multi_seed/seed_0/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-09-10-08-23"),
        ("1",  "runs/csac_lb_multi_seed/seed_1/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-09-11-15-54"),
        ("2",  "runs/csac_lb_multi_seed/seed_2/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-002-2026-04-09-12-26-54"),
        ("7",  "runs/csac_lb_multi_seed/seed_7/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-007-2026-04-10-16-33-51"),
        ("13", "runs/csac_lb_multi_seed/seed_13/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-013-2026-04-10-17-40-31"),
        ("42", "runs/csac_lb_multi_seed/seed_42/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-09-13-41-32"),
    ],
}

summary = {}
print("\n" + "="*80)
print(f"{'Algorithm':<12} {'Seeds':>6} {'Reward mean':>12} {'±std':>8} {'ViolRate mean':>14} {'±std':>8} {'Cost mean':>10} {'±std':>8}")
print("="*80)

for algo, seed_dirs in ALGO_SEEDS.items():
    seed_data = []
    for seed, run_dir in seed_dirs:
        json_path = PROJECT / run_dir / "eval_case_study_multiseed.json"
        if not json_path.exists():
            # fallback to original eval
            json_path = PROJECT / run_dir / "eval_case_study.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)
        # get "best" row
        rows = data.get("rows", [])
        best_row = next((r for r in rows if "best" in r["name"].lower()), rows[0] if rows else None)
        if best_row:
            seed_data.append({
                "seed": seed,
                "reward": best_row["total_reward"],
                "cost": best_row["total_cost"],
                "violation_rate": best_row["violation_rate"],
                "discomfort_rate": best_row["discomfort_rate"],
            })

    if not seed_data:
        continue

    rewards = [d["reward"] for d in seed_data]
    costs = [d["cost"] for d in seed_data]
    viols = [d["violation_rate"] for d in seed_data]

    mean_r = statistics.mean(rewards)
    std_r = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    mean_v = statistics.mean(viols)
    std_v = statistics.stdev(viols) if len(viols) > 1 else 0.0
    mean_c = statistics.mean(costs)
    std_c = statistics.stdev(costs) if len(costs) > 1 else 0.0

    summary[algo] = {
        "n_seeds": len(seed_data),
        "per_seed": seed_data,
        "reward_mean": mean_r, "reward_std": std_r,
        "cost_mean": mean_c, "cost_std": std_c,
        "violation_rate_mean": mean_v, "violation_rate_std": std_v,
    }
    print(f"{algo:<12} {len(seed_data):>6} {mean_r:>12.1f} {std_r:>8.1f} {mean_v:>14.4f} {std_v:>8.4f} {mean_c:>10.2f} {std_c:>8.2f}")

print("="*80)

out = PROJECT / "runs/multiseed_eval_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary -> {out}")
PYEOF

echo "" | tee -a "$LOG"
echo "=== Evaluation complete $(date) ===" | tee -a "$LOG"
