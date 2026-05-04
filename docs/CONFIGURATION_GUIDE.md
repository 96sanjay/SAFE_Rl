# Configuration Guide

This document covers every configurable surface in the Safe RL V2G training
pipeline: YAML files, environment variables, CLI arguments, and PID
Lagrangian tuning.

---

## 1. Configuration Layers

The system reads configuration from four sources. When the same parameter
appears in multiple layers, higher-priority layers override lower ones.

| Priority | Layer                 | Scope                                  |
|----------|-----------------------|----------------------------------------|
| 1 (highest) | Environment variables | Reward weights, feature flags, safety toggles |
| 2        | YAML config file      | Algorithm hyperparameters, model architecture, per-constraint PID gains |
| 3        | CLI arguments         | STEMS encoder geometry, critic mode, init std |
| 4 (lowest) | Code defaults         | Fallback values hard-coded in `cmdp_env.py` and `train_multi_lag_stems.py` |

**How they interact:**

- The training script (`train_multi_lag_stems.py`) loads OmniSafe's built-in
  `PPOLag.yaml` defaults, then deep-merges the user YAML on top. CLI arguments
  override specific fields (e.g., `--hidden_dim` sets the STEMS encoder width).
- Environment variables are read at runtime by `cmdp_env.py` inside the
  environment constructor. They are invisible to OmniSafe's config system --
  the YAML and CLI have no effect on reward weights or feature flags.
- The shell launch script is the single source of truth for a run. It sets env
  vars, then invokes the training script with the YAML path and CLI flags.

---

## 2. YAML Configuration

YAML files live in `configs/`. The top-level keys are:

```yaml
algo: PPOLagMulti          # Algorithm class name
env_id: CityLearnSafety-V2G-v2
seed: 42

train_cfgs:
  device: cuda:0
  total_steps: 350360      # epochs * steps_per_epoch
  vector_env_nums: 1
  parallel: 1

algo_cfgs:
  steps_per_epoch: 8759    # 1 CityLearn year = 8759 timesteps
  update_iters: 10         # PPO update passes per epoch
  target_kl: 0.08          # KL divergence target for early stopping
  kl_early_stop: true
  batch_size: 512
  obs_normalize: true      # Running mean/std on observations
  reward_normalize: true   # Running mean/std on rewards
  cost_normalize: false    # Cost normalization (usually off)
  standardized_cost_adv: true
  entropy_coef: 0.005
  max_grad_norm: 0.5

lagrange_cfgs:
  cost_limit: 21800              # Global fallback (overridden by multi_cfgs)
  lagrangian_multiplier_init: 0.001
  lambda_lr: 0.02                # SGD learning rate (used when PID is off)
  lambda_optimizer: Adam
  lagrangian_upper_bound: 35.0   # Hard cap on all lambda values

multi_cfgs:
  tau: 1.0                       # Softmax temperature for multi-constraint weighting

  # ── Per-constraint cost limits ──
  # NOTE: Code uses 0-indexed channels. The thesis constraint mapping is:
  #   Index 0 → C1 (EV departure readiness)
  #   Index 1 → C1 dense variant (Saute, usually disabled)
  #   Index 2 → C2 (Battery SoC band)
  #   Index 3 → C3 (Building power envelope)
  #   Index 4 → C4 (Grid import ceiling)
  cost_limit_0: 200              # C1: EV departure readiness
  cost_limit_1: 999999           # C1 dense variant: disabled (Saute handles when enabled)
  cost_limit_2: 1500             # C2: Battery SoC band violation
  cost_limit_3: 5000             # C3: Building power limit
  cost_limit_4: 8000             # C4: Grid power limit

  # Default PID gains (applied to all constraints unless overridden)
  pid_kp: 0.1
  pid_ki: 0.01
  pid_kd: 0.01
  pid_d_delay: 10
  pid_delta_p_ema_alpha: 0.95
  pid_delta_d_ema_alpha: 0.95

  # Per-constraint PID overrides (suffix _N for code index N; see mapping above)
  pid_kp_0: 2.0
  pid_ki_0: 0.05
  pid_kd_0: 0.0
  pid_ema_p_0: 0.0
  pid_kp_3: 0.5
  pid_ki_3: 0.05
  pid_kp_4: 0.3
  pid_ki_4: 0.03

model_cfgs:
  actor_type: gaussian_learning
  actor:
    hidden_sizes: [256, 256]   # MLP head (after STEMS encoder)
    activation: tanh
    lr: 0.0002
  critic:
    hidden_sizes: [256, 256]
    activation: tanh
    lr: 0.001
  linear_lr_decay: false

logger_cfgs:
  use_wandb: false
  use_tensorboard: true
  save_model_freq: 5           # Save checkpoint every N epochs
  log_dir: ./runs/headroom_gated_cmdp/5bld
  window_lens: 1               # Smoothing window for logged metrics
```

---

## 3. Environment Variables

All environment variables are read in `citylearn_safe/cmdp_env.py` during
environment construction. Changes require restarting the training process.

### 3.1 STEMS Reward Coefficients

These control the weight of each reward component in the STEMS reward function.
Set to `0.0` to disable a component.

| Variable | Default | Description |
|----------|---------|-------------|
| `STEMS_MU_ECONOMIC` | `0.3` | Economic reward (price-based grid cost) |
| `STEMS_ALPHA_GRID` | `3.0` | Grid-level power import penalty (r_sg) |
| `STEMS_ALPHA_BUILD` | `2.0` | Building-level power import penalty (r_sb) |
| `STEMS_BETA_RAMP` | `0.5` | Ramp rate penalty (smooths action changes) |
| `STEMS_XI_RENEWABLE` | `0.2` | Renewable energy utilization bonus |
| `STEMS_LAMBDA_EV` | `0.0` | Basic EV charging reward |
| `STEMS_ALPHA_BARRIER` | `0.5` | Battery SoC barrier reward (soft C2) |
| `STEMS_ALPHA_EV_GUARD` | `0.0` | EV guard reward (penalizes low SoC near departure) |
| `STEMS_ALPHA_V2G_CONTEXT` | `0.0` | V2G contextual discharge reward |
| `STEMS_ALPHA_PEAK_SHAVE` | `0.0` | Peak shaving reward |
| `STEMS_ALPHA_LOAD_SHIFT` | `0.0` | Battery load-shifting (buy low / sell high) |
| `STEMS_ALPHA_GRID_MILD` | `0.0` | Gentle grid awareness (softer than r_sg) |
| `STEMS_ALPHA_TRAJECTORY` | `0.0` | Forecast-aware trajectory arbitrage |
| `STEMS_TRAJ_EV_WEIGHT` | `1.0` | EV weight within trajectory reward |
| `STEMS_TRAJ_FORECAST_HOURS` | `24` | Lookahead horizon for trajectory reward |
| `STEMS_ALPHA_EV_SMART` | `0.0` | Headroom-gated EV price signal (departure-aware) |
| `STEMS_ALPHA_EV_SOLAR` | `0.0` | Solar-aligned EV charging bonus |
| `STEMS_ALPHA_SOLAR_STORE` | `0.0` | Solar storage reward (battery) |
| `STEMS_SOLAR_STORE_BATT_ONLY` | `0` | Restrict solar store to batteries only (flag) |
| `STEMS_EV_SLACK_ARB_SCALE` | `0.0` | EV slack arbitrage scale |
| `STEMS_ALPHA_HEADROOM` | `0.0` | Headroom reward |
| `STEMS_ALPHA_GRID_PENALTY` | `0.0` | Explicit grid penalty |
| `STEMS_ALPHA_PRICE_ARB` | `0.0` | Simple battery price arbitrage (AL-SAC style) |
| `STEMS_ALPHA_NEC_SIGN` | `0.0` | NEC-sign reward (align storage with load direction) |

### 3.2 Reward Modifiers

| Variable | Default | Description |
|----------|---------|-------------|
| `STEMS_SB_ASYMMETRIC` | `0` | If `1`, r_sb only penalizes imports (enables V2G exports) |
| `STEMS_SG_EXPORT_CREDIT` | `0.0` | Partial credit for net exports in r_sg |
| `STEMS_SG_THRESHOLD` | `0.0` | Fraction of P_grid_max below which imports are free |

### 3.3 Feature Control

| Variable | Default | Description |
|----------|---------|-------------|
| `CITYLEARN_TEMPORAL_WINDOW` | `0` | Temporal history length (0 = disabled, 12 = typical) |
| `CITYLEARN_TEMPORAL_RICH` | `0` | Rich temporal features (solar, EV SoC, hour encoding) |
| `CITYLEARN_SPATIAL_OBS` | `0` | Spatial graph features (per-building headroom, SoC spread) |
| `CITYLEARN_NUM_BUILDINGS` | `0` | Override building count (0 = auto-detect) |
| `STEMS_ENCODER_VERSION` | `v3` | STEMS encoder variant (GCN + Temporal Transformer) |
| `STEMS_AUX_TARGETS` | `0` | Auxiliary temporal prediction targets |

### 3.4 Safety and Constraint Control

| Variable | Default | Description |
|----------|---------|-------------|
| `CITYLEARN_EV_SAUTE` | `0` | Saute MDP for C1 (dense per-step EV budget augmentation) |
| `CITYLEARN_EV_SAUTE_SHAPED_ALPHA` | `0.0` | Smooth shaping for Saute penalty (0 = binary) |
| ~~`CITYLEARN_PENALTY_MAX`~~ | -- | **Not an env var.** The penalty cap is set via YAML key `lagrangian_upper_bound` in `lagrange_cfgs` (see Section 2). |
| `CITYLEARN_PID_LAGRANGE` | `0` | Enable PID Lagrangian (1 = on) |
| `CITYLEARN_EXECUTION_SHIELD` | `1` | Execution-time safety projection |
| `SAUTE_C4_ENABLED` | `0` | Saute MDP for C4 (grid power budget) |
| `SAUTE_C4_BUDGET` | `1500` | C4 Saute budget |
| `SAUTE_C4_PENALTY` | `5.0` | C4 Saute penalty |
| `SAUTE_C4_GAMMA` | `1.0` | C4 Saute discount |
| `SAUTE_C4_SHAPED_ALPHA` | `0.0` | C4 Saute shaping |

### 3.5 Action Control

| Variable | Default | Description |
|----------|---------|-------------|
| `CITYLEARN_EV_ACTION_CLAMP` | `0` | Clamp EV actions to feasible range |
| `CITYLEARN_EV_CLAMP_MARGIN` | `0.1` | Margin for EV action clamping |
| `CITYLEARN_EV_DISCONNECT_MASK` | `0` | Mask EV actions when vehicle disconnected |
| `CITYLEARN_BATT_CLAMP` | `1` | Clamp battery actions to SoC bounds |
| `CITYLEARN_BATT_SOC_UPPER_CLAMP` | `0.94` | Upper SoC clamp for batteries |
| `CITYLEARN_WM_DISABLE` | `0` | Disable washing machine actions (clamp to 0) |
| `CITYLEARN_BETA_ACTOR` | `0` | Beta actor mode: action space (0,1) instead of (-1,1) |
| `CITYLEARN_ACTION_MASK` | `0` | Action mask wrapper for power constraints |
| `CITYLEARN_SERL_PROJECTION` | `0` | SE-RL action projection (clip + penalty) |
| `CITYLEARN_COUPLED_BUDGET_ACTION` | `0` | Coupled budget actions per building |

### 3.6 Cost Weights (Deprecated)

These only affect the single-cost `_rebalanced_cost()` path. Multi-constraint
algorithms bypass them.

| Variable | Default | Description |
|----------|---------|-------------|
| `COST_W_C1` | `10.0` | C1 weight (EV departure) |
| `COST_W_C1_DENSE` | `5.0` | C1 dense cost weight |
| `COST_W_C2` | `0.0` | C2 weight (battery SoC) |
| `COST_W_C3` | `0.1` | C3 weight (building power) |
| `COST_W_C4` | `5.0` | C4 weight (grid power) |

### 3.7 Schema and Physical Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `CITYLEARN_SCHEMA` | -- | Path to CityLearn schema JSON |
| `CITYLEARN_STEMS_P_BUILDING_MAX` | auto | Max building power (auto-calibrated from env) |
| `CITYLEARN_STEMS_P_GRID_MAX` | auto | Max grid power (auto-calibrated from env) |
| `CITYLEARN_STEMS_SOC_LOW` | `0.05` | Lower SoC bound for C2 |
| `CITYLEARN_STEMS_SOC_HIGH` | `0.95` | Upper SoC bound for C2 |
| `CITYLEARN_C3_CONTROLLABLE` | `0` | Use controllable load only for C3 (read by `safety_env.py`) |
| `CITYLEARN_C0_DENSE_MERGE` | `0` | Merge dense C1 variant (code index 1) into sparse C1 channel (code index 0) |

---

## 4. CLI Arguments

The training script `scripts/training/train_multi_lag_stems.py` accepts:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cfg` | str | required | Path to YAML config file |
| `--init_std` | float | `1.0` | Initial policy standard deviation |
| `--hidden_dim` | int | `64` | STEMS encoder hidden dimension per GCN layer |
| `--output_dim` | int | `256` | STEMS encoder output embedding dimension |
| `--num_gcn_layers` | int | `3` | Number of GCN message-passing layers |
| `--stems_critic` | flag | off | Use STEMS encoder for reward critic (not just actor) |
| `--temporal_layers` | int | `2` | Number of Transformer layers in temporal encoder |
| `--temporal_pool` | str | `mean` | Temporal pooling mode: `mean`, `last`, or `cls` |
| `--partial_obs_norm` | flag | off | Normalize only base obs dims (leave temporal history raw) |

**Notes:**

- `--stems_critic` creates a separate STEMS encoder with independent weights
  for the reward value critic. Cost critics always use plain MLPs because they
  predict instantaneous constraint violations that do not benefit from
  temporal reasoning.
- `--partial_obs_norm` prevents OmniSafe's running normalization from corrupting
  temporal history patterns. It freezes the normalizer statistics for temporal
  dimensions to identity (mean=0, std=1).

---

## 5. PID Lagrangian Configuration

The PID Lagrangian (`citylearn_safe/pid_lagrange.py`) replaces OmniSafe's
standard SGD Lagrangian with a PID controller per constraint. Enable it with
`CITYLEARN_PID_LAGRANGE=1`.

### 5.1 How It Works

Standard Lagrangian: `lambda += lr * (ep_cost - limit)`

PID Lagrangian: `lambda = Kp * error_p + Ki * integral + Kd * derivative`

The error is normalized by the cost limit:
`delta = (ep_cost - cost_limit) / cost_limit`

This makes gains transferable across constraints with different cost scales
(e.g., C1 with costs around 200 vs. C3 with costs around 5000).

### 5.2 Per-Constraint Gains (R27a Recommended Values)

| Constraint | Kp | Ki | Kd | Cost Limit | Rationale |
|------------|-----|------|------|------------|-----------|
| C1 (EV departure readiness) | 2.0 | 0.05 | 0.0 | 200 | Aggressive -- must compete with C3 Lambda for policy gradient bandwidth |
| C1 dense (usually disabled) | -- | -- | -- | 999999 | Disabled (Saute handles dense EV corridor when enabled) |
| C2 (battery SoC) | 0.1 | 0.01 | 0.01 | 1500 | Default gains, moderate limit |
| C3 (building power) | 0.5 | 0.05 | 0.01 | 5000 | Moderate -- binding constraint, needs steady pressure |
| C4 (grid power) | 0.3 | 0.03 | 0.01 | 8000 | Gentle -- large headroom, avoid over-constraining |

### 5.3 Key Parameters

| Parameter | YAML Key | Description |
|-----------|----------|-------------|
| Proportional gain | `pid_kp` / `pid_kp_N` | Immediate response to current violation |
| Integral gain | `pid_ki` / `pid_ki_N` | Accumulated persistent violations (with anti-windup) |
| Derivative gain | `pid_kd` / `pid_kd_N` | Dampens oscillation when cost changes rapidly |
| D-term delay | `pid_d_delay` | Lookback epochs for derivative computation |
| P-term EMA | `pid_delta_p_ema_alpha` / `pid_ema_p_N` | Smoothing for proportional term (0.95 = heavy) |
| D-term EMA | `pid_delta_d_ema_alpha` | Smoothing for derivative term |
| Lambda upper bound | `lagrangian_upper_bound` | Hard cap on all lambda values |

### 5.4 Curriculum Annealing

Cost limits can be annealed over training (e.g., `cost_limit_0: [200, 20]` --
code index 0 = thesis C1 -- over epochs 0-20). The PID controller's
`update_cost_limit()` method rescales the
integral term proportionally to prevent windup from stale accumulation at
different cost-limit scales.

---

## 6. Shell Script Template

A complete launch script with annotations. Use this as a starting point for new
runs.

```bash
#!/bin/bash
# R27a: Headroom-Gated CMDP
set -euo pipefail

# ── Conda activation (adjust path if needed) ──
if [ -n "${CONDA_EXE:-}" ]; then
    eval "$(${CONDA_EXE} shell.bash hook)"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
fi
conda activate citylearn

# ── Schema ──
export CITYLEARN_SCHEMA="$PWD/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"

# ── STEMS encoder settings ──
export CITYLEARN_TEMPORAL_RICH=1        # Rich temporal features
export CITYLEARN_TEMPORAL_WINDOW=12     # 12-step history
export STEMS_ENCODER_VERSION=v3          # GCN + Temporal Transformer

# ── Reward terms ──
export STEMS_MU_ECONOMIC=0.0            # r_eco OFF
export STEMS_ALPHA_GRID=1.5             # r_sg
export STEMS_ALPHA_BUILD=1.5            # r_sb
export STEMS_BETA_RAMP=0.3              # r_ramp
export STEMS_XI_RENEWABLE=0.3           # r_ren
export STEMS_ALPHA_BARRIER=0.3          # r_barrier
export STEMS_LAMBDA_EV=0.0              # r_ev OFF (Lagrangian handles C1)
export STEMS_ALPHA_EV_SMART=1.5         # r_ev_smart (departure-aware)
export STEMS_ALPHA_EV_GUARD=0.0         # OFF
export STEMS_ALPHA_TRAJECTORY=0.0       # OFF (conflicts with C1)

# ── Safety flags ──
export CITYLEARN_EV_SAUTE=0             # Saute OFF (sparse C1 only)
export CITYLEARN_BATT_CLAMP=0           # No action clamps
export CITYLEARN_EV_ACTION_CLAMP=0
export CITYLEARN_PID_LAGRANGE=1         # PID Lagrangian ON

# ── SoC bounds ──
export CITYLEARN_STEMS_SOC_LOW=0.05
export CITYLEARN_STEMS_SOC_HIGH=0.95

# ── Cost control ──
export CITYLEARN_C3_CONTROLLABLE=1      # Use controllable load only for C3

# ── Launch ──
nohup python scripts/training/train_multi_lag_stems.py \
    --cfg configs/active/headroom_gated_cmdp.yaml \
    --hidden_dim 64 \
    --output_dim 256 \
    --num_gcn_layers 3 \
    --stems_critic \
    --temporal_layers 2 \
    --temporal_pool mean \
    --partial_obs_norm \
    > /tmp/headroom_gated_cmdp.log 2>&1 &

echo "PID: $!"
echo "Log: tail -f /tmp/headroom_gated_cmdp.log"
```

**Key points:**

- Every reward weight not used should be explicitly set to `0.0`. The code
  defaults are non-zero for some weights (e.g., `STEMS_ALPHA_GRID=3.0`), so
  omitting a variable means inheriting the default, not disabling it.
- `set -euo pipefail` ensures the script fails fast on any error.
- `nohup` with output redirection allows the run to survive SSH disconnection.
- The YAML config path is relative to the working directory, not the script.
