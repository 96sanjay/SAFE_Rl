# Action Mask Ablation: R21 + C2/C3/C4 Safety Layer

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add C4 (grid-level) action masking to the existing C2+C3 ActionMaskWrapper, then run a clean ablation: R21 without mask vs R21 with C2+C3+C4 mask.

**Architecture:** Extend `ActionMaskWrapper._compute_safe_bounds()` with a second pass that enforces grid-level import constraint via two-pass slack redistribution (Stolz et al. NeurIPS 2024 interval-based masking approach). The mask is SE-RL: part of the environment, present at both training and deployment. Lagrangian (PPOLagMulti + PID) learns the policy; the mask provides structural safety. One run script with argument `[baseline|treatment]` controls mask ON/OFF.

**Tech Stack:** Python 3.10, PyTorch, OmniSafe PPOLagMulti, CityLearn, ActionMaskWrapper

**Paper reference:** Stolz, Krasowski, Thumm et al. "Excluding the Irrelevant: Focusing RL through Continuous Action Masking" (NeurIPS 2024) — proves policy gradient correctness under interval-based action rescaling.

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `citylearn_safe/action_mask_wrapper.py` | **MODIFY** | Add C4 grid-level mask (two-pass slack redistribution after existing C3 loop) |
| `run_ablation_mask.sh` | **CREATE** | Single run script with `[baseline\|treatment]` argument |
| `configs/on-policy/ablation_mask.yaml` | **CREATE** | R21 YAML copy with log_dir parameterized |
| `scripts/eval_ablation_mask.py` | **CREATE** | Evaluation: proper departure-based C0, per-building stats, CityLearn KPIs |

---

## Chunk 1: Extend ActionMaskWrapper with C4 Grid Mask

### Task 1: Add P_grid_max to ActionMaskWrapper.__init__

**Files:**
- Modify: `citylearn_safe/action_mask_wrapper.py:68-77`

- [ ] **Step 1: Add P_grid_max env var reading in __init__**

After line 76 (`self._soc_high = ...`), add:

```python
        self._p_gmax = float(os.environ.get(
            "CITYLEARN_STEMS_P_GRID_MAX", "10.2352"))
```

And update the init print at line 174:

```python
        print(f"[ActionMask] ENABLED: P_bmax={self._p_bmax:.2f} kW, "
              f"P_gmax={self._p_gmax:.2f} kW, "
              f"N_buildings={self._n_buildings}")
```

- [ ] **Step 2: Verify no other code references a grid max in this file**

Run: `grep -n "grid_max\|p_gmax\|P_grid" citylearn_safe/action_mask_wrapper.py`
Expected: Only the two new lines.

- [ ] **Step 3: Commit**

```bash
git add citylearn_safe/action_mask_wrapper.py
git commit -m "feat: add P_grid_max to ActionMaskWrapper init"
```

### Task 2: Add C4 two-pass grid enforcement to _compute_safe_bounds

**Files:**
- Modify: `citylearn_safe/action_mask_wrapper.py:260-404`

The C4 mask runs AFTER the existing per-building C3 loop (line 280-403). It tightens the **charge side only** (C4 is import-only: `max(0, sum(NEC)) ≤ P_grid_max`). Discharge/export does not violate C4.

- [ ] **Step 1: Add C4 enforcement after line 403 (after passthrough loop)**

Insert before the `return` statement at line 404. The new code goes between the passthrough loop (line 400-402) and the return:

```python
        # ── C4: Grid-level aggregate import constraint ──
        # C4 cost = max(0, grid_import - P_grid_max) where grid_import = max(0, sum(NEC))
        # The mask tightens the CHARGE side only (import = charge increases NEC).
        # Discharge reduces NEC → never violates C4 → safe_min unchanged.
        #
        # Two-pass slack redistribution:
        #   Pass 1: Equal C4 share per building. Effective = min(C3_charge, C4_share).
        #   Pass 2: Redistribute slack from buildings that can't use their share.

        total_exo_import = sum(max(0.0, e) for e in exo_nec)
        grid_headroom = max(0.0, self._p_gmax - total_exo_import)

        if grid_headroom < sum(
            max(0.0, safe_max[self._building_batt_act[b]] * self._batt_powers[b]
                + (safe_max[self._building_ev_act[b]] * self._ev_max_charge[b]
                   if b in self._building_ev_act else 0.0))
            for b in self._building_batt_act
        ):
            # C4 is binding — need to tighten charge limits

            # Compute each building's max charge power from current safe_max
            bld_charge_power = {}
            for b in self._building_batt_act:
                p = max(0.0, safe_max[self._building_batt_act[b]]) * self._batt_powers[b]
                if b in self._building_ev_act:
                    p += max(0.0, safe_max[self._building_ev_act[b]]) * self._ev_max_charge.get(b, 0.0)
                bld_charge_power[b] = p

            n_bld = len(bld_charge_power)

            # Pass 1: equal share
            equal_share = grid_headroom / max(n_bld, 1)
            alloc = {}
            slack = 0.0
            slack_receivers = []
            for b, cp in bld_charge_power.items():
                if cp <= equal_share:
                    # Building can't use its full share → keep as-is, release slack
                    alloc[b] = cp
                    slack += equal_share - cp
                else:
                    # Building wants more than its share → cap at share for now
                    alloc[b] = equal_share
                    slack_receivers.append(b)

            # Pass 2: redistribute slack to buildings that need more
            if slack > 0 and slack_receivers:
                extra_per = slack / len(slack_receivers)
                for b in slack_receivers:
                    alloc[b] = min(bld_charge_power[b], alloc[b] + extra_per)

            # Apply C4 allocation: tighten safe_max for charge side
            for b in self._building_batt_act:
                if b not in alloc:
                    continue
                c4_limit_kw = alloc[b]
                batt_act_idx = self._building_batt_act[b]
                p_batt = self._batt_powers[b]

                if b in self._building_ev_act:
                    ev_act_idx = self._building_ev_act[b]
                    ev_max_ch = self._ev_max_charge.get(b, 0.0)
                    total_dev = p_batt + ev_max_ch if ev_max_ch > 0 else p_batt + 1e-6
                    batt_share = c4_limit_kw * (p_batt / total_dev)
                    ev_share = c4_limit_kw - batt_share

                    # Tighten battery charge
                    c4_batt_smax = batt_share / p_batt if p_batt > 0 else 0.0
                    safe_max[batt_act_idx] = min(safe_max[batt_act_idx], c4_batt_smax)

                    # Tighten EV charge
                    c4_ev_smax = ev_share / ev_max_ch if ev_max_ch > 0 else 0.0
                    safe_max[ev_act_idx] = min(safe_max[ev_act_idx], c4_ev_smax)
                else:
                    # Battery only
                    c4_batt_smax = c4_limit_kw / p_batt if p_batt > 0 else 0.0
                    safe_max[batt_act_idx] = min(safe_max[batt_act_idx], c4_batt_smax)
```

- [ ] **Step 2: Add C4 diagnostic logging**

In the 1000-step print at line 466-469, add after `ev_range_mean`:

```python
            # C4 diagnostic
            total_charge_kw = sum(
                max(0.0, safe_max[self._building_batt_act[b]]) * self._batt_powers[b]
                + (max(0.0, safe_max[self._building_ev_act[b]]) * self._ev_max_charge.get(b, 0.0)
                   if b in self._building_ev_act else 0.0)
                for b in self._building_batt_act)
            c4_util = (total_exo_import + total_charge_kw) / max(self._p_gmax, 1e-6)
            print(f"[ActionMask] C4: grid_headroom={grid_headroom:.1f}kW "
                  f"max_charge={total_charge_kw:.1f}kW "
                  f"c4_util={c4_util:.2f}")
```

Note: `total_exo_import` needs to be stored as `self._last_total_exo_import` in `_apply_mask` for the diagnostic. Add `self._last_total_exo_import = sum(max(0.0, e) for e in exo_nec)` in `_apply_mask` after computing `exo_nec`.

- [ ] **Step 3: Test manually with a quick sanity check**

Run a short episode (100 steps) with mask enabled and verify C4 diagnostics appear:

```bash
cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && \
CITYLEARN_ACTION_MASK="1" \
CITYLEARN_SCHEMA="$PWD/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json" \
CITYLEARN_CENTRAL_AGENT="1" \
CITYLEARN_REWARD_TYPE="stems" \
CITYLEARN_STEMS_P_BUILDING_MAX="4.6083" \
CITYLEARN_STEMS_P_GRID_MAX="10.2352" \
PYTHONPATH="$PWD" \
python -c "
from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.action_mask_wrapper import ActionMaskWrapper
import numpy as np
base = make_base_env(central_agent=True)
safety = CityLearnSafetyEnvV3(base)
forecast = ForecastObsWrapper(safety, forecast_horizon=24)
env = ActionMaskWrapper(forecast)
obs, _ = env.reset()
for t in range(2000):
    action = np.random.uniform(-1, 1, size=env.action_space.shape)
    obs, r, term, trunc, info = env.step(action)
    if term or trunc: break
print('C4 test complete. Check for [ActionMask] C4: lines above.')
"
```

Expected: `[ActionMask] C4: grid_headroom=X.XkW max_charge=X.XkW c4_util=X.XX` at step 1000.

- [ ] **Step 4: Commit**

```bash
git add citylearn_safe/action_mask_wrapper.py
git commit -m "feat: add C4 grid-level mask via two-pass slack redistribution"
```

### Task 3: Update docstring

- [ ] **Step 1: Update module docstring at top of file**

Change line 2 from:
```python
Action masking wrapper for per-building power constraints (C3).
```
to:
```python
Action masking wrapper for C2 (battery SoC), C3 (per-building power), C4 (grid aggregate).
```

Add to the docstring after line 7:
```python
    |sum(NEC_b)| < P_grid_max     grid aggregate (C4, import-only)
```

Add `CITYLEARN_STEMS_P_GRID_MAX` to the env vars list.

- [ ] **Step 2: Commit**

```bash
git add citylearn_safe/action_mask_wrapper.py
git commit -m "docs: update action mask docstring for C4"
```

---

## Chunk 2: Create Ablation Run Script and Config

### Task 4: Create ablation YAML config

**Files:**
- Create: `configs/on-policy/ablation_mask.yaml`

- [ ] **Step 1: Write config (R21 exact, parameterized log_dir)**

```yaml
# Ablation: Action Mask (R21 baseline, only ACTION_MASK differs between arms)
# IDENTICAL to r21_ppo.yaml except log_dir and STEMS_BETA_RAMP=0.0 (see run script)

algo: PPOLagMulti
env_id: CityLearnSafety-V2G-v2
seed: 42

train_cfgs:
  total_steps: 787110
  vector_env_nums: 1
  parallel: 1

algo_cfgs:
  steps_per_epoch: 8759
  update_iters: 30
  target_kl: 0.08
  kl_early_stop: true
  batch_size: 256
  obs_normalize: true
  reward_normalize: true
  cost_normalize: false
  standardized_cost_adv: true
  entropy_coef: 0.005
  max_grad_norm: 40.0

lagrange_cfgs:
  cost_limit: 21800
  lagrangian_multiplier_init: 0.001
  lambda_lr: 0.035
  lambda_optimizer: Adam
  lagrangian_upper_bound: 5.0

multi_cfgs:
  tau: 1.0
  cost_limit_0: 999999
  cost_limit_1: 1500
  cost_limit_2: 4000
  cost_limit_3: 3000
  cost_limit_4: 1500
  anneal_cost_limit_0: [999999, 1800, 20, 40]
  pid_kp: 0.1
  pid_ki: 0.01
  pid_kd: 0.01
  pid_d_delay: 10
  pid_delta_p_ema_alpha: 0.95
  pid_delta_d_ema_alpha: 0.95
  pid_kp_0: 5.0
  pid_ki_0: 0.0
  pid_kd_0: 0.0
  pid_ema_p_0: 0.0
  pid_kp_3: 0.5
  pid_ki_3: 0.05
  pid_kp_4: 0.3
  pid_ki_4: 0.03

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
  log_dir: ./runs/ablation_mask/5bld
  window_lens: 1
```

- [ ] **Step 2: Diff against R21 YAML**

```bash
diff <(grep -v '^#\|log_dir' configs/on-policy/ablation_mask.yaml | sed '/^$/d') \
     <(grep -v '^#\|log_dir' configs/on-policy/r21_ppo.yaml | sed '/^$/d')
```

Expected: No differences (all hyperparameters identical).

- [ ] **Step 3: Commit**

```bash
git add configs/on-policy/ablation_mask.yaml
git commit -m "config: add ablation mask YAML (R21 exact copy)"
```

### Task 5: Create unified run script

**Files:**
- Create: `run_ablation_mask.sh`

- [ ] **Step 1: Write single run script with argument parsing**

```bash
#!/usr/bin/env bash
# Ablation Study: Action Mask (C2+C3+C4)
# Usage: ./run_ablation_mask.sh [baseline|treatment] [seed]
#   baseline:  ACTION_MASK=0 (Lagrangian only, identical to R21)
#   treatment: ACTION_MASK=1 (Lagrangian + C2/C3/C4 mask)
#
# All env vars IDENTICAL to R21 except:
#   1. STEMS_BETA_RAMP=0.0 in BOTH arms (mask forces r_ramp=0 when active,
#      so baseline must match to isolate the mask as sole variable)
#   2. CITYLEARN_ACTION_MASK=0 or 1 (the ablation variable)
set -euo pipefail

ARM="${1:?Usage: $0 [baseline|treatment] [seed]}"
SEED="${2:-42}"

if [[ "$ARM" != "baseline" && "$ARM" != "treatment" ]]; then
    echo "ERROR: first argument must be 'baseline' or 'treatment'"
    exit 1
fi

PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- Environment (IDENTICAL to R21) ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
export CITYLEARN_CENTRAL_AGENT="1"
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# --- Thresholds (IDENTICAL to R21) ---
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="10.2352"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# --- PID Lagrangian (IDENTICAL to R21) ---
export CITYLEARN_PID_LAGRANGE="1"

# --- Sauté MDP for C0 (IDENTICAL to R21) ---
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_PENALTY="5.0"
export CITYLEARN_EV_SAUTE_GAMMA="1.0"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"

# --- Reward terms (IDENTICAL to R21 EXCEPT ramp=0) ---
export STEMS_ALPHA_GRID="0.0"
export STEMS_SG_THRESHOLD="0.5"
export STEMS_ALPHA_LOAD_SHIFT="0.0"
export STEMS_ALPHA_GRID_MILD="0.3"
export STEMS_MU_ECONOMIC="0.0"
export STEMS_ALPHA_BUILD="0.0"
export STEMS_XI_RENEWABLE="0.2"
export STEMS_BETA_RAMP="0.0"              # ABLATION: zeroed in BOTH arms (mask forces r_ramp=0)
export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export STEMS_ALPHA_BARRIER="0.5"
export STEMS_ALPHA_PEAK_SHAVE="0.0"
export STEMS_ALPHA_EV_GUARD="1.0"
export STEMS_ALPHA_V2G_CONTEXT="3.0"
export STEMS_ALPHA_EV_SOLAR="0.0"
export STEMS_ALPHA_SOLAR_STORE="0.0"
export STEMS_SOLAR_STORE_BATT_ONLY="0"
export STEMS_ALPHA_HEADROOM="0.0"
export STEMS_ALPHA_PRICE_ARB="0.0"
export STEMS_ALPHA_GRID_PENALTY="0.0"
export STEMS_ALPHA_NEC_SIGN="0.0"
export STEMS_EV_SLACK_ARB_SCALE="0.0"

# --- Cost weights (IDENTICAL to R21) ---
export COST_W_C2="0.0"
export COST_W_C3="5.0"
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# --- Safety clamps (IDENTICAL to R21) ---
export CITYLEARN_C3_CONTROLLABLE="1"
export CITYLEARN_WM_DISABLE="1"
export CITYLEARN_EV_ACTION_CLAMP="0"
export CITYLEARN_BATT_CLAMP="1"
export CITYLEARN_SPATIAL_OBS="0"
export CITYLEARN_TEMPORAL_WINDOW="0"

# =====================================================================
# ABLATION VARIABLE
# =====================================================================
if [[ "$ARM" == "treatment" ]]; then
    export CITYLEARN_ACTION_MASK="1"
    RUN_DIR="runs/ablation_mask_treatment_s${SEED}"
else
    export CITYLEARN_ACTION_MASK="0"
    RUN_DIR="runs/ablation_mask_baseline_s${SEED}"
fi

echo ""
echo "============================================"
echo "  ABLATION: $ARM (seed=$SEED)"
echo "  ACTION_MASK=$CITYLEARN_ACTION_MASK"
echo "  Config: R21 exact (STEMS_BETA_RAMP=0.0)"
echo "  90 epochs, 5 buildings"
echo "============================================"
echo ""

# Create a temporary YAML with the right seed and log_dir
TMP_CFG="/tmp/ablation_mask_${ARM}_s${SEED}.yaml"
sed "s/^seed: 42/seed: ${SEED}/" configs/on-policy/ablation_mask.yaml | \
    sed "s|log_dir: .*|log_dir: ./${RUN_DIR}/5bld|" > "$TMP_CFG"

mkdir -p "$RUN_DIR"
/home/christmas/miniconda3/envs/citylearn/bin/python scripts/train_multi_lag.py \
    --cfg "$TMP_CFG" \
    2>&1 | tee "$RUN_DIR/full_log.txt"
```

- [ ] **Step 2: Make executable and verify**

```bash
chmod +x run_ablation_mask.sh
# Dry run: just check env vars parse correctly
bash -n run_ablation_mask.sh
```

- [ ] **Step 3: Commit**

```bash
git add run_ablation_mask.sh configs/on-policy/ablation_mask.yaml
git commit -m "scripts: add unified ablation mask run script (baseline/treatment + seed)"
```

---

## Chunk 3: Evaluation Script

### Task 6: Create evaluation script

**Files:**
- Create: `scripts/eval_ablation_mask.py`

- [ ] **Step 1: Write evaluation script**

Standard eval pattern: load checkpoint, build MLP [256,256], run 8760 steps deterministically, report:
- C0 EV departure: `deficit_count / departures` (departure-based)
- C2/C3: `violation_count / (N × 5)` building-steps
- C4: `violation_count / N` timesteps
- Per-building battery charge/discharge/idle %
- Per-charger EV V2G %
- CityLearn KPIs (7 metrics)
- Hourly action profile (cycling check)
- Side-by-side comparison table

For treatment: eval WITH mask active (it's part of the env).
For baseline: eval WITHOUT mask.

CLI: `python scripts/eval_ablation_mask.py --baseline CKPT --treatment CKPT`

- [ ] **Step 2: Commit**

```bash
git add scripts/eval_ablation_mask.py
git commit -m "scripts: add ablation mask evaluation script"
```

---

## Chunk 4: Run and Evaluate

### Task 7: Launch experiments

- [ ] **Step 1: Run seed 42 (both arms)**

```bash
./run_ablation_mask.sh baseline 42 &
./run_ablation_mask.sh treatment 42 &
```

- [ ] **Step 2: Check at epoch 5 (Rule 14)**

```bash
tail -20 runs/ablation_mask_baseline_s42/full_log.txt
tail -20 runs/ablation_mask_treatment_s42/full_log.txt
```

Verify: StopIter > 1, lambdas moving, C0 curriculum working.
For treatment: look for `[ActionMask]` and `[ActionMask] C4:` lines.

- [ ] **Step 3: Run seeds 123 and 456**

```bash
./run_ablation_mask.sh baseline 123 &
./run_ablation_mask.sh treatment 123 &
./run_ablation_mask.sh baseline 456 &
./run_ablation_mask.sh treatment 456 &
```

### Task 8: Evaluate and compare

- [ ] **Step 1: Find best checkpoint epoch for each run**

Check progress.csv for lowest EpCost or best EpRet in final 10 epochs.

- [ ] **Step 2: Run evaluation**

```bash
python scripts/eval_ablation_mask.py \
    --baseline runs/ablation_mask_baseline_s42/5bld/.../torch_save/epoch-90.pt \
    --treatment runs/ablation_mask_treatment_s42/5bld/.../torch_save/epoch-90.pt
```

- [ ] **Step 3: Aggregate across seeds and report mean ± std for each metric**

---

## Key Design Decisions (for thesis)

1. **C4 is import-only**: `cost = max(0, sum(NEC) - P_grid_max)`. Discharge reduces NEC → never violates C4. So the mask only tightens `safe_max` (charge side). This preserves gradient signal for discharge/V2G actions.

2. **Two-pass slack redistribution**: Equal C4 share → buildings that can't use their share release slack → slack goes to buildings that need more. This avoids over-restricting solar buildings while still enforcing the aggregate constraint.

3. **STEMS_BETA_RAMP=0.0 in both arms**: The code forces `r_ramp=0` when mask is active (line 1176). Setting it to 0 in both arms eliminates this hidden confound.

4. **BATT_CLAMP + mask C2 = double enforcement**: BATT_CLAMP has a stale-SoC bug (reads `time_step-1`). The mask reads `soc[-1]` (latest). With both active, the mask catches what BATT_CLAMP misses. The mask's C2 is the effective enforcement.

5. **Mask is SE-RL (part of environment)**: Present at both training and deployment. The policy learns within the constrained action space. This is validated by Stolz et al. (NeurIPS 2024) who prove policy gradient correctness under interval-based rescaling.
