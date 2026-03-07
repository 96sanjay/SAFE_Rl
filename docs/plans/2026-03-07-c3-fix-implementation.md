# C3 Controllable Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make C3 controllable actually visible in the PPOLag cost signal so the agent can learn to reduce C3 violations.

**Architecture:** No code changes to the environment — all fixes are config/weight changes and a new run script. The `safety_env_v3.py` cost computation (line 1148) uses env vars for weights, so we only change env vars and YAML configs. We create a new R8 experiment that fixes all 3 root causes (weight imbalance, StopIter bug, lambda explosion).

**Tech Stack:** OmniSafe PPOLag, CityLearn environment, bash scripts, YAML configs

---

### Task 1: Compute Zero-Action Cost Floor with New Weights

**Files:**
- Create: `scripts/compute_cost_floor.py`
- Reference: `citylearn_safe/safety_env_v3.py:1037-1153` (cost formula)
- Reference: `run_r6_comparison_5bld.sh` (env var setup)

**Step 1: Write the cost floor computation script**

```python
#!/usr/bin/env python3
"""Compute the cost floor (zero-action baseline) with given env var weights.

Runs one full episode (8759 steps) with action=0 and prints per-constraint
weighted costs. Used to set a feasible cost_limit for PPOLag.
"""
import os, sys, json, numpy as np

# Must set env vars BEFORE importing citylearn_safe
# (they are read at __init__ time)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3


def main():
    schema = os.environ.get("CITYLEARN_SCHEMA",
        os.path.join(PROJECT, "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"))
    os.environ.setdefault("CITYLEARN_SCHEMA", schema)

    env = CityLearnSafetyEnvV3()
    obs, info = env.reset()

    totals = {
        "cost": 0.0,
        "cost_stems_battery": 0.0,
        "cost_stems_building_power": 0.0,
        "cost_stems_grid_power": 0.0,
        "cost_ev_departure": 0.0,
        "cost_ev_dense": 0.0,
        "cost_comfort": 0.0,
    }

    n_actions = env.action_space.shape[0]
    zero_action = np.zeros(n_actions, dtype=np.float32)

    for step in range(8759):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        for key in totals:
            totals[key] += float(info.get(key, 0.0))
        if terminated or truncated:
            break

    w_ev = float(os.environ.get("CITYLEARN_W_COST_EV", "1.0"))
    w_soc = float(os.environ.get("CITYLEARN_W_COST_SOC", "1.0"))
    w_bld = float(os.environ.get("CITYLEARN_W_COST_BUILDING", "1.0"))
    w_grid = float(os.environ.get("CITYLEARN_W_COST_GRID", "1.0"))

    print("\n=== Zero-Action Cost Floor ===")
    print(f"Steps completed: {step + 1}")
    print(f"\nWeights: w_ev={w_ev}, w_soc={w_soc}, w_bld={w_bld}, w_grid={w_grid}")
    print(f"\nPer-constraint raw totals:")
    print(f"  C1 EV departure:     {totals['cost_ev_departure']:>12.1f}")
    print(f"  C2 Battery SoC:      {totals['cost_stems_battery']:>12.1f}")
    print(f"  C3 Building power:   {totals['cost_stems_building_power']:>12.1f}")
    print(f"  C4 Grid power:       {totals['cost_stems_grid_power']:>12.1f}")
    print(f"  EV dense:            {totals['cost_ev_dense']:>12.1f}")
    print(f"  Comfort:             {totals['cost_comfort']:>12.1f}")
    print(f"\nWeighted totals (what OmniSafe sees):")
    print(f"  C1: {w_ev * totals['cost_ev_departure']:>12.1f}  ({w_ev * totals['cost_ev_departure'] / max(1, totals['cost']) * 100:.1f}%)")
    print(f"  C2: {w_soc * totals['cost_stems_battery']:>12.1f}  ({w_soc * totals['cost_stems_battery'] / max(1, totals['cost']) * 100:.1f}%)")
    print(f"  C3: {w_bld * totals['cost_stems_building_power']:>12.1f}  ({w_bld * totals['cost_stems_building_power'] / max(1, totals['cost']) * 100:.1f}%)")
    print(f"  C4: {w_grid * totals['cost_stems_grid_power']:>12.1f}  ({w_grid * totals['cost_stems_grid_power'] / max(1, totals['cost']) * 100:.1f}%)")
    print(f"\nTotal EpCost (from env): {totals['cost']:>12.1f}")
    print(f"\nSuggested cost_limit (80% of floor): {totals['cost'] * 0.80:.0f}")
    print(f"Suggested cost_limit (90% of floor): {totals['cost'] * 0.90:.0f}")


if __name__ == "__main__":
    main()
```

**Step 2: Run with current weights to verify script works**

```bash
cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
export CITYLEARN_SCHEMA="$PWD/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"
export CITYLEARN_C3_CONTROLLABLE="0"
export CITYLEARN_SPATIAL_OBS="0"
python scripts/compute_cost_floor.py
```

Expected: Script runs and prints per-constraint breakdown. Note the total EpCost value.

**Step 3: Run with NEW weights to find the new cost floor**

```bash
# Same as Step 2 but with rebalanced weights
export CITYLEARN_W_COST_SOC="5.0"
export CITYLEARN_W_COST_BUILDING="5.0"
export CITYLEARN_W_COST_GRID="1.0"
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_SPATIAL_OBS="1"
python scripts/compute_cost_floor.py
```

Expected: C3 is now ~35% of total cost. Note the suggested cost_limit values.

**Step 4: Commit**

```bash
git add scripts/compute_cost_floor.py
git commit -m "feat: add zero-action cost floor computation script"
```

---

### Task 2: Create R8 Training Config

**Files:**
- Create: `configs/on-policy/r8_c3fix_5bld.yaml`
- Reference: `configs/on-policy/r6_compare_r6_5bld.yaml` (base config)

**Step 1: Create the R8 config**

Based on R6 config but with fixes for all 3 root causes:
- Lower actor LR: 0.0005 -> 0.0001 (fixes StopIter=1 bug)
- cost_limit: set from Task 1 output (replaces infeasible 24,400)
- lambda_init: 1.0 (was 10.0)
- lambda_lr: 0.01 (was 0.05)

```yaml
# R8: C3 Controllable Fix (all 3 root causes addressed)
# Based on R6 but with:
#   1. Rebalanced weights (via env vars, not here)
#   2. Lower actor LR to fix StopIter=1 bug
#   3. Feasible cost_limit (from zero-action baseline)
#   4. Stable lambda params
# Set CITYLEARN_C3_CONTROLLABLE=1, CITYLEARN_SPATIAL_OBS=1 before running
algo: PPOLag
env_id: CityLearnSafety-V2G-v2
seed: 42

train_cfgs:
  total_steps: 437950          # 50 epochs x 8759 steps/epoch
  vector_env_nums: 1
  parallel: 1

algo_cfgs:
  steps_per_epoch: 8759
  update_iters: 60
  target_kl: 0.10
  kl_early_stop: true
  batch_size: 256
  obs_normalize: true
  reward_normalize: true
  cost_normalize: false
  entropy_coef: 0.005

lagrange_cfgs:
  cost_limit: REPLACE_WITH_TASK1_OUTPUT   # <-- fill from Task 1 Step 3
  lagrangian_multiplier_init: 1.0          # was 10.0
  lambda_lr: 0.01                          # was 0.05
  lambda_optimizer: SGD

model_cfgs:
  actor:
    hidden_sizes: [256, 256]
    activation: tanh
    lr: 0.0001                             # was 0.0005 -- fixes StopIter=1
  critic:
    hidden_sizes: [256, 256]
    activation: tanh
    lr: 0.001
  linear_lr_decay: false

logger_cfgs:
  use_wandb: false
  use_tensorboard: true
  save_model_freq: 5
  log_dir: ./runs/r8_c3fix
  window_lens: 1
```

**Step 2: Fill in cost_limit from Task 1 output**

Replace `REPLACE_WITH_TASK1_OUTPUT` with the "Suggested cost_limit (90% of floor)" value from Task 1 Step 3.

**Step 3: Commit**

```bash
git add configs/on-policy/r8_c3fix_5bld.yaml
git commit -m "feat: add R8 config with C3 fix (weight rebalance + StopIter fix + stable lambda)"
```

---

### Task 3: Create R8 Run Script

**Files:**
- Create: `run_r8_c3fix.sh`
- Reference: `run_r6_comparison_5bld.sh` (template)

**Step 1: Write the run script**

```bash
#!/usr/bin/env bash
# R8: C3 Controllable Fix
#
# Fixes 3 root causes from R6 failure:
#   1. Rebalanced weights: W_COST_BUILDING 0.5 -> 5.0, W_COST_SOC 10 -> 5
#   2. Lower actor LR: 0.0005 -> 0.0001 (fixes StopIter=1 bug)
#   3. Feasible cost_limit + stable lambda params
#
# Compare against R5a and R6 to measure C3 controllable improvement.
#
# Usage: bash run_r8_c3fix.sh
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# --- Shared env vars (5-building schema) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# Thresholds (same as R6)
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# === R8 COST WEIGHTS (rebalanced from R6) ===
# R6: W_SOC=10.0, W_BLD=0.5  ->  C3 was 6.5% of signal
# R8: W_SOC=5.0,  W_BLD=5.0  ->  C3 is ~35% of signal
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="5.0"
export CITYLEARN_W_COST_BUILDING="5.0"
export CITYLEARN_W_COST_GRID="1.0"

# EV (same as R6)
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# P0 + P1 features (same as R6)
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_SPATIAL_OBS="1"

echo ""
echo "=========================================="
echo "  R8: C3 Controllable Fix (5 buildings, 50 ep)"
echo "  Fixes: weight rebalance + actor LR + lambda"
echo "  W_COST_SOC=5.0 (was 10.0)"
echo "  W_COST_BUILDING=5.0 (was 0.5)"
echo "  W_COST_GRID=1.0 (was 0.05)"
echo "  Actor LR=0.0001 (was 0.0005)"
echo "  Lambda init=1.0, lr=0.01"
echo "=========================================="
echo ""

python scripts/train_omnisafe.py --cfg configs/on-policy/r8_c3fix_5bld.yaml
```

**Step 2: Make executable**

```bash
chmod +x run_r8_c3fix.sh
```

**Step 3: Commit**

```bash
git add run_r8_c3fix.sh
git commit -m "feat: add R8 run script with rebalanced C3 weights"
```

---

### Task 4: Create R8-Baseline (Same Weights, No C3 Controllable)

**Files:**
- Create: `configs/on-policy/r8_baseline_5bld.yaml`
- Create: `run_r8_baseline.sh`

**Purpose:** Run the SAME rebalanced weights but WITHOUT C3 controllable. This isolates the C3 controllable effect from the weight changes.

**Step 1: Copy R8 config with same cost_limit**

```yaml
# R8-Baseline: Same weights as R8 but WITHOUT C3 controllable
# Compare R8 vs R8-Baseline to isolate C3 controllable contribution
algo: PPOLag
env_id: CityLearnSafety-V2G-v2
seed: 42

train_cfgs:
  total_steps: 437950
  vector_env_nums: 1
  parallel: 1

algo_cfgs:
  steps_per_epoch: 8759
  update_iters: 60
  target_kl: 0.10
  kl_early_stop: true
  batch_size: 256
  obs_normalize: true
  reward_normalize: true
  cost_normalize: false
  entropy_coef: 0.005

lagrange_cfgs:
  cost_limit: REPLACE_WITH_TASK1_OUTPUT   # Same as R8
  lagrangian_multiplier_init: 1.0
  lambda_lr: 0.01
  lambda_optimizer: SGD

model_cfgs:
  actor:
    hidden_sizes: [256, 256]
    activation: tanh
    lr: 0.0001
  critic:
    hidden_sizes: [256, 256]
    activation: tanh
    lr: 0.001
  linear_lr_decay: false

logger_cfgs:
  use_wandb: false
  use_tensorboard: true
  save_model_freq: 5
  log_dir: ./runs/r8_baseline
  window_lens: 1
```

**Step 2: Write the baseline run script**

```bash
#!/usr/bin/env bash
# R8-Baseline: Same rebalanced weights as R8 but WITHOUT C3 controllable
# Compare R8 vs R8-Baseline to isolate the C3 controllable improvement
set -euo pipefail

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# Same rebalanced weights as R8
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="5.0"
export CITYLEARN_W_COST_BUILDING="5.0"
export CITYLEARN_W_COST_GRID="1.0"

export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# === KEY DIFFERENCE: C3 controllable OFF, spatial obs OFF ===
export CITYLEARN_C3_CONTROLLABLE="0"
export CITYLEARN_SPATIAL_OBS="0"

echo ""
echo "=========================================="
echo "  R8-Baseline (5 buildings, 50 ep)"
echo "  Same weights as R8, NO C3 controllable"
echo "  Compare against R8 to isolate P1 effect"
echo "=========================================="
echo ""

python scripts/train_omnisafe.py --cfg configs/on-policy/r8_baseline_5bld.yaml
```

**Step 3: Make executable and commit**

```bash
chmod +x run_r8_baseline.sh
git add configs/on-policy/r8_c3fix_5bld.yaml configs/on-policy/r8_baseline_5bld.yaml \
        run_r8_c3fix.sh run_r8_baseline.sh
git commit -m "feat: add R8 + R8-Baseline configs for clean C3 controllable comparison"
```

---

### Task 5: Run Both Experiments

**Step 1: Run R8-Baseline first (no C3 controllable)**

```bash
cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
bash run_r8_baseline.sh 2>&1 | tee runs/r8_baseline_train.log
```

Expected: ~6-7 hours for 50 epochs. Watch for:
- StopIter should start at 60 and decay (NOT stay at 1)
- Lambda should grow slowly (not explode)

**Step 2: Run R8 (with C3 controllable)**

```bash
bash run_r8_c3fix.sh 2>&1 | tee runs/r8_c3fix_train.log
```

Expected: Same duration. StopIter should also decay healthily.

**Step 3: Verify StopIter is NOT stuck at 1**

After first 5 epochs complete, check:

```bash
head -7 runs/r8_c3fix/PPOLag-*/seed-*/progress.csv | cut -d, -f7
head -7 runs/r8_baseline/PPOLag-*/seed-*/progress.csv | cut -d, -f7
```

Expected: Values > 1 (ideally 20-60 for early epochs). If still 1, the actor LR fix didn't work and you need to try `kl_early_stop: false` with `update_iters: 10`.

---

### Task 6: Compare Results

**Step 1: Extract final metrics**

```bash
cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork

echo "=== R8 (C3 controllable ON) ==="
tail -1 runs/r8_c3fix/PPOLag-*/seed-*/progress.csv | cut -d, -f1,2,30

echo "=== R8-Baseline (C3 controllable OFF) ==="
tail -1 runs/r8_baseline/PPOLag-*/seed-*/progress.csv | cut -d, -f1,2,30
```

**Step 2: Evaluate per-constraint violations**

Run evaluation on both checkpoints using `eval_all_v2.py` (or `eval_quick.py`):

```bash
# Evaluate R8
python eval_quick.py --checkpoint runs/r8_c3fix/PPOLag-*/seed-*/epoch-49.pt \
    --output runs/r8_c3fix/eval_results.csv

# Evaluate R8-Baseline
python eval_quick.py --checkpoint runs/r8_baseline/PPOLag-*/seed-*/epoch-49.pt \
    --output runs/r8_baseline/eval_results.csv
```

**Step 3: Compare C3 violation percentages**

The key metric: **C3 violation % should be lower in R8 than R8-Baseline**, while C1/C2/C4 should remain similar.

**Step 4: Commit results**

```bash
git add runs/r8_c3fix/eval_results.csv runs/r8_baseline/eval_results.csv
git commit -m "results: R8 vs R8-Baseline C3 controllable comparison"
```

---

## Checklist for Success

- [ ] Task 1: Cost floor computed with new weights, cost_limit filled in
- [ ] Task 2: R8 config created with all 3 fixes
- [ ] Task 3: R8 run script with rebalanced weights
- [ ] Task 4: R8-Baseline (control experiment) for clean A/B comparison
- [ ] Task 5: Both runs complete, StopIter > 1 verified
- [ ] Task 6: C3 violation % lower in R8 vs R8-Baseline

## Troubleshooting

**If StopIter still = 1 with actor LR 0.0001:**
- Try `kl_early_stop: false` + `update_iters: 10`
- Or try `target_kl: 0.20`

**If lambda still explodes:**
- Try `lambda_lr: 0.005` (even slower)
- Or try `cost_limit` at 95% of baseline floor

**If C3 violation % is same in R8 vs R8-Baseline:**
- C3 structural floor may be unavoidable (7.35% from c3_structural_proof.py)
- The controllable improvement is real but small — focus on what CAN be controlled
