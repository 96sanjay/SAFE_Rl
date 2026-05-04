# Safe-CityLearn-Fork: Technical Report
## OmniSafe × CityLearn Integration — End-to-End ML Pipeline

> **Authors:** Multi-Agent Analysis (Senior ML Architect · Codebase Analyzer · Technical Documentation Writer)
> **Codebase:** This repository (`SAFE_RL/`)
> **Branch:** `saf-omni-work`
> **Date:** 2026-04-09

---

## Table of Contents

1. [System & Environment Architecture](#1-system--environment-architecture)
2. [Wrapper Logic Deep Dive](#2-wrapper-logic-deep-dive)
3. [End-to-End ML Pipeline](#3-end-to-end-ml-pipeline)
4. [Algorithm Benchmarking Strategy](#4-algorithm-benchmarking-strategy)

---

## 1. System & Environment Architecture

### 1.1 High-Level Overview

This codebase implements a **Constrained Markov Decision Process (CMDP)** framework for safe reinforcement learning applied to multi-building energy management with Vehicle-to-Grid (V2G) control. It bridges two independent libraries:

| Layer | Library | Role |
|-------|---------|------|
| **Simulation** | [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) | Building energy environment (loads, solar, batteries, EVs, pricing) |
| **Learning** | [OmniSafe](https://github.com/PKU-Alignment/omnisafe) | Safe RL algorithms (PPO-Lag, CPO, TRPO-Lag, SAC-Lag, etc.) |
| **Glue** | `citylearn_safe/` (55 modules) | CMDP adapter, wrappers, reward shaping, constraint mapping, PID Lagrangian |

The system manages **5 buildings**, each with a battery energy storage system and EV charger(s), over a 1-year horizon (8,759 hourly timesteps per episode). A **central agent** issues actions for all devices via a single shared policy.

### 1.2 Constraint System (5 Independent Safety Constraints)

The CMDP formulation enforces 5 simultaneous safety constraints:

| ID | Constraint | Physical Meaning | Cost Signal | Typical Limit |
|----|-----------|------------------|-------------|---------------|
| **C0** | EV Departure Deficit | EVs must reach target SoC before departure | `cost_ev_departure` | 1,800 kWh |
| **C1** | EV Dense Charging | Step-level EV charging adequacy | `cost_ev_dense` | 1,500 |
| **C2** | Battery SoC Bounds | Battery SoC stays within [0.0, 0.95] | `cost_stems_battery` | 4,000 |
| **C3** | Building Power | Per-building |NEC| ≤ P_building_max (4.61 kW) | `cost_stems_building_power` | 12,000 |
| **C4** | Grid Power | Aggregate import ≤ P_grid_max (10.24 kW) | `cost_stems_grid_power` | 2,500 |

Each constraint is tracked independently with its own Lagrange multiplier, cost critic, and optional PID controller.

### 1.3 Environment Wrapper Stack

The interface between OmniSafe and CityLearn is realized through a composable wrapper chain. Each layer transforms observations, actions, or cost signals:

```
┌──────────────────────────────────────────────────────────────┐
│                    OmniSafe Algorithm                        │
│           (PPOLagMulti / SACLagMulti / CPO / ...)            │
│                                                              │
│  step(action) → (obs_t, reward_t, cost_t, term, trunc, info)│
└──────────────────────┬───────────────────────────────────────┘
                       │  CMDP interface (6-tuple, torch tensors)
┌──────────────────────▼───────────────────────────────────────┐
│              CityLearnCMDP(CMDP)                           │
│  @env_register → "CityLearnSafety-V2G-v2"                   │
│  · _stems_reward()  — 19-component reward                    │
│  · _rebalanced_cost() — weighted cost combination            │
│  · Sauté reward shaping (budget-gated penalty)               │
│  · V2G discharge tracking                                    │
│  · EV/battery action clamping (optional)                     │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│              Optional Wrapper Layers (configurable)          │
│                                                              │
│  ┌─ ActionMaskWrapper ──────────────────────────────────┐    │
│  │  Piecewise rescaling for C2/C3/C4 (gradient-safe)    │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌─ ActionProjectionSERL ───────────────────────────────┐    │
│  │  SE-RL execution shield (Markgraf et al., 2025)      │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌─ SauteEVBudgetWrapper ───────────────────────────────┐    │
│  │  Sauté MDP for C1 (budget state augmentation)        │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌─ SauteConstraintWrapper ─────────────────────────────┐    │
│  │  Generic Sauté MDP for any constraint (C3/C4)        │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌─ SpatialGraphFeaturesWrapper ────────────────────────┐    │
│  │  Inter-building spatial features (+68 dims)          │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌─ TemporalHistoryWrapper ─────────────────────────────┐    │
│  │  Sliding window of past observations                 │    │
│  └──────────────────────────────────────────────────────┘    │
│  ┌─ ForecastObsWrapper ────────────────────────────────┐     │
│  │  24h lookahead: price, load, solar, EV urgency       │    │
│  │  (+128 dims)                                         │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│            CityLearnSafetyEnv(gym.Env)                     │
│  · Action clipping (per-dimension, handles action_2 ∈ [0,1])│
│  · Cost computation (C0–C4) with p-norm violations           │
│  · 212+ KPI fields per step via KPILogger                    │
│  · Action-based EV deficit calculation (tau-aligned)         │
│  · CityLearn episode-end evaluation                          │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│            SingleAgentListAdapter(gym.Env)                    │
│  · Unwraps CityLearn [Box(...)] → Box(...)                   │
│  · Converts single-element lists to scalars                  │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│              CityLearn Base Environment                       │
│  · 5 buildings with batteries + EV chargers                  │
│  · Hourly simulation (8,759 steps/year)                      │
│  · Schema: citylearn_challenge_2022_phase_all_plus_evs       │
└──────────────────────────────────────────────────────────────┘
```

### 1.4 OmniSafe CMDP Interface Contract

OmniSafe's `CMDP` base class requires the following interface, which `CityLearnCMDP` implements:

```python
# Registration (cmdp_env.py)
@env_register
class CityLearnCMDP(CMDP):
    _support_envs: ClassVar[list[str]] = ['CityLearnSafety-V2G-v2']
    need_time_limit_wrapper: bool = False
    need_auto_reset_wrapper: bool = True

    def step(self, action) -> tuple:
        """
        Returns:
            obs:        torch.Tensor(float32)  — observation vector
            reward:     torch.Tensor(float32)  — scalar reward (STEMS 19-component)
            cost:       torch.Tensor(float32)  — scalar cost (rebalanced C0-C4)
            terminated: torch.Tensor(bool)     — episode done
            truncated:  torch.Tensor(bool)     — time limit reached
            info:       dict                   — 70+ fields (per-constraint costs, KPIs)
        """

    def reset(self, seed=None, options=None) -> tuple:
        """Returns: (obs_tensor, info_dict)"""
```

For **multi-constraint algorithms** (PPOLagMulti, SACLagMulti), the per-constraint costs are extracted directly from the `info` dict by the `_MultiCostAdapter`, bypassing the single `cost` return:

```python
# Extracted by _MultiCostAdapter during rollout:
info['cost_ev_departure']           # C0
info['cost_ev_dense']               # C1
info['cost_stems_battery']          # C2
info['cost_stems_building_power']   # C3
info['cost_stems_grid_power']       # C4
```

### 1.5 Directory Structure

```
Safe-CityLearn-Fork/
├── citylearn_safe/                    # Core Safe-RL package (55 Python modules)
│   ├── cmdp_env.py                 # CMDP adapter (1,766 lines)
│   ├── safety_env.py               # Inner safety layer (1,900+ lines)
│   ├── pid_lagrange.py                # PID Lagrangian controller (146 lines)
│   ├── saute_ev_wrapper.py            # Sauté MDP for EV (109 lines)
│   ├── saute_constraint_wrapper.py    # Generic Sauté MDP
│   ├── kpi_logger.py                  # 212-column CSV logging
│   ├── adapters.py                    # List→Box space adapter
│   ├── stems_encoder.py            # STEMS reward encoder
│   ├── extractors.py               # KPI extraction (1,900+ lines)
│   ├── grads/
│   │   ├── ppo_lag_multi.py           # PPOLagMulti algorithm (884 lines)
│   │   ├── ppo_lag_grads.py           # GradS variant (750+ lines)
│   │   ├── sac_lag_multi.py           # Off-policy multi-constraint SAC
│   │   └── td3_lag_multi.py           # Off-policy multi-constraint TD3
│   ├── wrappers/                      # Observation/action wrappers
│   │   ├── action_mask_wrapper.py
│   │   ├── forecast_obs_wrapper.py
│   │   ├── temporal_obs_wrapper.py
│   │   ├── stems_obs_wrapper.py
│   │   └── feasibility_obs_wrapper.py
│   └── psf/                           # Predictive Safety Filter
│       ├── psf_wrapper.py
│       ├── psf_filter.py
│       └── psf_lookahead.py
├── configs/
│   ├── on-policy/                     # 140+ YAML configs (PPO, TRPO, CPO, ...)
│   └── off-policy/                    # 35+ YAML configs (SAC, TD3, DDPG, ...)
├── scripts/
│   ├── train_omnisafe.py              # OmniSafe Agent-based training
│   ├── train_multi_lag.py             # Direct PPOLagMulti training
│   ├── benchmark_algos.py             # Multi-algorithm benchmarking
│   └── [151 utility scripts]
├── evaluation_pipeline/               # Evaluation framework
├── data/                              # CityLearn schemas and data
├── vendor_deps/                       # Vendored omnisafe + citylearn
│   ├── omnisafe/
│   └── citylearn/
└── [137 shell scripts]                # Experiment runners
```

### 1.6 Hardware & Software Requirements

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 3050 (4 GB VRAM) |
| RAM | 14 GB system memory |
| OS | Linux (kernel 6.17.0) |
| Python Environment | Conda (`citylearn`) |
| Key Dependencies | OmniSafe (vendored), CityLearn (vendored), PyTorch, Gymnasium |

---

## 2. Wrapper Logic Deep Dive

### 2.1 SingleAgentListAdapter (`adapters.py`)

**Purpose:** Bridge CityLearn's list-wrapped spaces to OmniSafe's expected `Box` spaces.

CityLearn, even in single-agent mode, wraps observations and actions as single-element lists: `[Box(...)]` instead of `Box(...)`. OmniSafe expects flat `Box` spaces.

**Input/Output Transformation:**

| Direction | CityLearn Format | Adapted Format |
|-----------|-----------------|----------------|
| Observation | `[np.array([...])]` | `np.array([...])` |
| Action | `np.array([...])` | `[np.array([...])]` |
| Reward | `[float]` | `float` |
| Terminated | `[bool]` | `bool` |

```python
class SingleAgentListAdapter(gym.Env):
    def step(self, action):
        obs, r, term, trunc, info = self.base.step([action])  # wrap action
        return (np.asarray(_first(obs), dtype=np.float32),     # unwrap obs
                float(_first(r)),                               # unwrap reward
                bool(_first(term)),                             # unwrap done
                bool(_first(trunc)), info)
```

### 2.2 CityLearnSafetyEnv (`safety_env.py`)

**Purpose:** The inner safety envelope that adds constraint cost computation, KPI logging, and action history tracking around the base CityLearn environment.

**Class Hierarchy:** `gym.Env` (not a `gym.Wrapper` — standalone env that holds `self.base`)

#### 2.2.1 Observation Space

Observations are flattened from CityLearn's structured output into a single `np.float32` vector (70–80 base dimensions depending on building count and features).

#### 2.2.2 Action Space

26-dimensional continuous action space (5 buildings × {battery, EV, washing_machine} + padding):
- Battery actions: [-1, 1] (discharge to charge)
- EV actions: [-1, 1] (V2G discharge to charge)
- Washing machine: [0, 1] (delay fraction, typically disabled)

#### 2.2.3 Cost Computation

Five independent cost signals computed per step:

**C0 — EV Departure Deficit** (action-based, tau-aligned):
```python
# When an EV departs at timestep t:
deficit_kwh = max(0, required_energy - actual_energy_delivered)
cost_ev_departure += deficit_kwh
```
The action-based computation uses stored action history aligned with `tau = pre_step_time_step + 1`, avoiding the "double-step" bug of earlier versions.

**C1 — EV Dense Charging:**
```python
# Per-step EV charging adequacy
cost_ev_dense = sum(max(0, urgency_i - charging_rate_i) for each EV_i)
```

**C2 — Battery SoC Bounds** (p-norm):
```python
low_violation  = max(0, soc_min - observed_soc)
high_violation = max(0, observed_soc - soc_max)
cost_stems_battery = (low_violation + high_violation) / band_width
```

**C3 — Building Power** (p-norm):
```python
# Per-building net energy consumption must stay within limits
for building in buildings:
    violation = max(0, |NEC_building| - P_building_max)
    cost_stems_building_power += (violation / P_building_max) ** p_norm
```

**C4 — Grid Power** (aggregate import):
```python
total_import = sum(max(0, NEC_building) for building in buildings)
cost_stems_grid_power = max(0, total_import - P_grid_max)
```

#### 2.2.4 Info Dict

Returns 70+ fields per step including all cost signals, battery SoC per building, all 26 actions, 8 EV actions, grid metrics, economic metrics, comfort metrics, and episode-end CityLearn KPIs.

### 2.3 ForecastObsWrapper (`forecast_obs_wrapper.py`)

**Purpose:** Append 24-hour lookahead features to enable temporal planning.

**Observation Augmentation:** +128 dimensions

| Feature Block | Dimensions | Computation |
|---------------|-----------|-------------|
| Price forecast | 24 | `price[t:t+24] / mean(prices) - 1.0` |
| Load forecast | 24 | Sum non-shiftable load, normalize by max |
| Solar forecast | 24 | Sum solar generation, normalize by max |
| EV urgency | 8 | `min(1.0, deficit / (max_rate × hours_to_departure))` per charger |
| Time encoding | 48 | `[sin(2π(t+k)/24), cos(2π(t+k)/24)]` for k=0..23 |

```python
class ForecastObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        forecast = self._compute_forecast()  # 128-dim vector
        return np.concatenate([obs, forecast])
```

### 2.4 TemporalHistoryWrapper (`temporal_obs_wrapper.py`)

**Purpose:** Append a sliding window of selected past observation features to approximate non-Markov temporal dependencies.

**Configuration:**
- `history_indices`: Which observation dimensions to track
- `window_size`: Number of past timesteps (default: 12)

**Observation Augmentation:** `+len(history_indices) × window_size` dimensions

```python
# Layout: [original_obs | hist_t-T | hist_t-T+1 | ... | hist_t-1]
# Zero-filled on reset, no warm-up artifacts
class TemporalHistoryWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        tracked = obs[self.history_indices]
        self._buffer = np.roll(self._buffer, -1, axis=0)
        self._buffer[-1] = tracked
        return np.concatenate([obs, self._buffer.flatten()])
```

### 2.5 SpatialGraphFeaturesWrapper (`stems_obs_wrapper.py`)

**Purpose:** Append inter-building spatial interaction features for coordination.

**Observation Augmentation:** +68 dimensions (4 features × 17 buildings max)

| Feature | Formula | Meaning |
|---------|---------|---------|
| Power headroom | `clip((P_max - |NEC|) / P_max, -5, 1)` | Remaining power capacity |
| SoC spread | `SoC_i - mean(SoC)` | Deviation from district average |
| Neighbour avg SoC | `(total_SoC - SoC_i) / (N-1)` | Average peer SoC |
| Grid contribution | `max(0, NEC_i) / total_import` | Share of district import |

### 2.6 SauteEVBudgetWrapper (`saute_ev_wrapper.py`)

**Purpose:** Implement Sauté MDP (Sootla et al., ICML 2022) to convert the EV charging constraint (C1) from a CMDP cost to a state-augmented reward signal.

**Core Mechanism:**

The Sauté MDP tracks a normalized safety budget λ ∈ (-∞, 1] that depletes as the EV cost accumulates:

```
λ_{t+1} = (λ_t - c_t / d) / γ
```

Where `d` is the total budget (default: 25,000) and `γ` is the discount factor.

**Observation Augmentation:** +1 dimension (budget value, clipped to [-1, 1])

**Cost Elimination:** Sets `cost_ev_dense = 0.0` in info dict, removing C1 from the Lagrangian entirely.

**Reward Reshaping:** When budget is exhausted (λ < 0):
- **Shaped mode** (`alpha > 0`): `reward = reward - alpha × |deficit|` — preserves gradient signal
- **Binary mode** (`alpha = 0`): `reward = -penalty` — kills 93% of gradient (deprecated)

**Configuration:**
```bash
CITYLEARN_EV_SAUTE=1                    # Enable
CITYLEARN_EV_SAUTE_BUDGET=25000         # Budget d
CITYLEARN_EV_SAUTE_SHAPED_ALPHA=10.0    # Smooth penalty weight
```

**Key Insight:** The original budget of d=1,500 depleted at step 580 (out of 8,759), killing gradient for 93% of the episode. Setting d=25,000 resolved this.

### 2.7 SauteConstraintWrapper (`saute_constraint_wrapper.py`)

**Purpose:** Generic Sauté MDP applicable to ANY constraint, parameterized by `cost_key`.

Identical mechanics to `SauteEVBudgetWrapper` but configurable:

```python
SauteConstraintWrapper(
    env=base_env,
    cost_key="cost_stems_grid_power",  # Target constraint
    budget_d=50000,
    penalty=5.0,
    label="c4"                         # Logging identifier
)
```

### 2.8 ActionMaskWrapper (`action_mask_wrapper.py`)

**Purpose:** Rescale policy outputs to satisfy power constraints (C2/C3/C4) **without clipping**, preserving gradient flow.

**Key Design:** Piecewise linear mapping with zero bias:

```
raw ∈ [-1, 0)  →  executed ∈ [safe_min, 0)    (discharge side)
raw = 0        →  executed = 0                  (identity)
raw ∈ (0, 1]   →  executed ∈ (0, safe_max]     (charge side)
```

**Safe Bounds Computation (per building, per step):**
1. Read exogenous NEC (non-shiftable load + solar)
2. Compute import headroom: `P_building_max - exo_NEC`
3. Compute export headroom: `P_building_max + exo_NEC`
4. For buildings with both battery and EV: proportional split (battery priority)
5. Enforce SoC limits: `a_min = (soc_low - current_soc) / soc_per_unit`
6. Optional C4 correction: scale down if aggregate grid import would exceed limit

**Info Augmentations:** `action_mask_safe_min/max`, `mask_penalty`, `mask_delta_l2`

### 2.9 ActionProjectionSERL (`action_projection_serl.py`)

**Purpose:** SE-RL execution shield (Markgraf et al., 2025) that projects unsafe actions onto the nearest feasible point at execution time.

**Mechanism:**
1. Predict next-step constraint violations from proposed action
2. If any violation predicted, solve a QP to find the nearest feasible action
3. Apply projected action instead of raw policy output
4. Log projection delta for debugging

**Configuration:** `CITYLEARN_SERL_PROJECTION=1`

### 2.10 PredictiveSafetyFilterWrapper (`psf/psf_wrapper.py`)

**Purpose:** Multi-horizon predictive safety filter that corrects RL actions BEFORE execution using H-step lookahead constraint predictions.

**Architecture:** RL policy → PSF → Environment

**Constraint Corrections (priority order):**
1. **EV Deadline (C0, highest priority):** Force minimum charge for urgent EVs (urgency ≥ 0.5)
2. **Battery SoC (C2):** Soft buffer zone [0.20, 0.80] + hard bounds clamp
3. **Building Power (C3):** Reduce EV (non-urgent only) and battery discharge
4. **Grid Import (C4):** Scale down imports if aggregate exceeds limit

**Configuration:**
```bash
PSF_HORIZON=24                    # Prediction horizon (steps)
PSF_CORRECTION_MODE="heuristic"   # or "passthrough" (disabled)
PSF_EV_URGENCY_THRESHOLD=0.5      # Min urgency to enforce charging
```

### 2.11 CoupledBudgetActionWrapper (`coupled_budget_action_wrapper.py`)

**Purpose:** Simplify the flat 26-dimensional action space to building-level budget semantics.

**Action Transformation:**
- **Input:** `[budget_per_building, ev_share_params, passthrough]`
- **Output:** Flat CityLearn action vector with per-device allocations

For each building with both battery and EV:
```
Budget ≥ 0 (charge):
  total_charge = ev_floor + budget × charge_cap
  ev_charge, batt_charge = spill_split(total, ev_cap, batt_cap, share)

Budget < 0 (discharge):
  total_discharge = |budget| × discharge_cap
  ev_discharge, batt_discharge = spill_split(total, ...)
```

### 2.12 ComfortWrapper (`comfort_wrapper.py`)

**Purpose:** Add optional temperature comfort constraint for buildings with LSTM dynamics models.

**Cost Computation:**
```python
if setpoint_available:
    deviation = |T_indoor - T_setpoint| - deadband
else:
    deviation = max(0, T_indoor - T_max) + max(0, T_min - T_indoor)
cost_comfort = w_comfort × max(0, deviation)
```

**Auto-Detection:** Enabled only when `LSTMDynamics` model detected in schema.

### 2.13 FeasibilityObsWrapper (`feasibility_obs_wrapper.py`)

**Purpose:** Append explicit safe action bounds to observations so the policy can "see" its feasible region.

**Observation Augmentation:** +2 × action_dim dimensions (safe_min, safe_max per action)

### 2.14 STEMSCombinedWrapper (`stems_obs_wrapper.py`)

**Purpose:** Stack multiple observation augmentation layers in the correct order.

**Composition Order:** Spatial → Forecast → Temporal

**Final Observation Layout:**
```
[base_obs | spatial_68 | forecast_128 | temporal_summary]
```

For base_obs ≈ 255 dims: total ~1,471 dims (summary mode).

### 2.15 Wrapper Summary Table

| Wrapper | Type | Augmentation | Constraint | Gradient-Safe |
|---------|------|-------------|------------|---------------|
| SingleAgentListAdapter | Space adapter | None (format only) | — | N/A |
| CityLearnSafetyEnv | Inner safety | 70+ info fields | C0–C4 costs | N/A |
| ForecastObsWrapper | Obs augmentation | +128 dims | — | Yes |
| TemporalHistoryWrapper | Obs augmentation | +n×window dims | — | Yes |
| SpatialGraphFeaturesWrapper | Obs augmentation | +68 dims | — | Yes |
| SauteEVBudgetWrapper | State augmentation | +1 dim (budget) | C1→reward | Yes (shaped) |
| SauteConstraintWrapper | State augmentation | +1 dim (budget) | Any→reward | Yes (shaped) |
| ActionMaskWrapper | Action rescaling | Same shape | C2/C3/C4 | Yes (piecewise) |
| ActionProjectionSERL | Action projection | Same shape | All | No (QP) |
| PredictiveSafetyFilter | Action correction | Same shape | All (H-step) | No (heuristic) |
| CoupledBudgetAction | Action structure | Reduced dims | — | Yes |
| ComfortWrapper | Cost signal | +5 info fields | Comfort | N/A |
| FeasibilityObsWrapper | Obs augmentation | +2×act_dim | — | Yes |

---

## 3. End-to-End ML Pipeline

### 3.1 Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│ Environment  │───▶│   Training   │───▶│  Checkpoint   │───▶│  Evaluation  │
│ Init & Data  │    │    Loop      │    │   Storage     │    │  & Analysis  │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────────┘
```

### 3.2 Phase 1: Environment Initialization & Data Ingestion

#### 3.2.1 Data Sources

The simulation data lives under `data/citylearn_challenge_2022_phase_all_plus_evs/`:

| File | Content | Resolution |
|------|---------|-----------|
| `schema_5buildings.json` | Building topology + device specs | Static |
| `Building_*.csv` | Non-shiftable load, solar generation, pricing | Hourly (8,760 rows) |
| `weather.csv` | Temperature, humidity, solar irradiance | Hourly |
| `carbon_intensity.csv` | Grid carbon intensity | Hourly |
| `pricing.csv` | Time-of-use electricity prices | Hourly |

#### 3.2.2 Environment Construction

```python
# scripts/train_multi_lag.py (simplified)
def main(cfg_path):
    # 1. Load algorithm defaults (PPO base)
    defaults = load_ppo_defaults()

    # 2. Load custom experiment config
    custom = yaml.safe_load(open(cfg_path))
    merged = deep_update(defaults, custom)

    # 3. Build OmniSafe Config
    cfgs = Config(**merged)

    # 4. Instantiate PPOLagMulti (triggers env construction)
    agent = PPOLagMulti(env_id='CityLearnSafety-V2G-v2', cfgs=cfgs)
    #       └─► _init_env() builds _MultiCostAdapter
    #              └─► CityLearnCMDP.__init__()
    #                     └─► Constructs full wrapper stack
```

#### 3.2.3 Wrapper Stack Assembly (in `CityLearnCMDP.__init__`)

```python
# 1. Base CityLearn environment
base = make_citylearn_env(schema_path, central_agent=True)

# 2. List→Box adapter
adapted = SingleAgentListAdapter(base)

# 3. Safety envelope (cost computation)
safe_env = CityLearnSafetyEnv(adapted, soc_min=0.0, soc_max=0.95)

# 4. Observation augmentation
env = ForecastObsWrapper(safe_env)                    # +128 dims

if temporal_window > 0:
    env = TemporalHistoryWrapper(env, ...)             # +history dims

if spatial_obs:
    env = SpatialGraphFeaturesWrapper(env)             # +68 dims

# 5. Constraint wrappers
if ev_saute:
    env = SauteEVBudgetWrapper(env, budget_d=25000)    # +1 dim, removes C1

if saute_c4:
    env = SauteConstraintWrapper(env, cost_key='cost_stems_grid_power')

# 6. Action wrappers (mutually exclusive)
if serl_projection:
    env = ActionProjectionSERL(env)
elif action_mask:
    env = ActionMaskWrapper(env)

self._env = env  # Final wrapped environment
```

### 3.3 Phase 2: Model Initialization

#### 3.3.1 Actor-Critic Architecture

```python
# PPOLagMulti._init_model()

# Standard OmniSafe actor-critic (reward)
super()._init_model()
# Actor:  MLP [obs_dim → 256 → 256 → act_dim] (Gaussian policy)
# Critic: MLP [obs_dim → 256 → 256 → 1]        (reward V-function)

# 5 per-constraint cost V-critics
self._cost_critics = nn.ModuleList()
for i in range(5):
    critic_i = CriticBuilder(obs_dim, hidden=[256,256], activation='tanh')
    self._cost_critics.append(critic_i.build_critic('v'))

# 5 per-constraint Lagrange multipliers
self._per_lagranges = []
for i in range(5):
    if use_pid:
        lag_i = PIDLagrange(
            cost_limit=cost_limits[i],
            kp=pid_kp_i, ki=pid_ki_i, kd=pid_kd_i,
            penalty_max=lagrangian_upper_bound
        )
    else:
        lag_i = Lagrange(cost_limit=cost_limits[i], ...)
    self._per_lagranges.append(lag_i)
```

#### 3.3.2 PID Lagrangian (Key Innovation — `pid_lagrange.py`)

Replaces OmniSafe's standard SGD lambda update with a PID controller (Stooke et al., ICML 2020):

```python
class PIDLagrange:
    def pid_update(self, ep_cost: float) -> float:
        """PID update for Lagrange multiplier."""
        # Cost-limit normalized delta (dimensionless, O(1))
        delta = (ep_cost - self.cost_limit) / self.cost_limit

        # P-term (optionally EMA-smoothed)
        self.p_term = ema_alpha * self.p_term + (1 - ema_alpha) * delta

        # I-term (clamped to prevent windup)
        self.i_term = clip(self.i_term + delta, 0, penalty_max)

        # D-term (only fires on cost INCREASES)
        d_raw = delta - self.prev_delta
        self.d_term = ema_d * self.d_term + (1 - ema_d) * max(0, d_raw)

        # Combined PID output
        pid_out = self.kp * self.p_term + self.ki * self.i_term + self.kd * self.d_term
        self.lagrangian = clip(pid_out, 0, penalty_max)
        return self.lagrangian
```

**Per-Constraint Gains (critical — uniform gains fail):**

| Constraint | Kp | Ki | Kd | Rationale |
|-----------|-----|-----|-----|-----------|
| C0 (EV departure) | 5.0 | 0.0 | 0.0 | Pure proportional — no integral windup risk |
| C1 (EV dense) | 0.1 | 0.01 | 0.01 | Default — Sauté handles most of C1 |
| C2 (Battery SoC) | 0.1 | 0.01 | 0.01 | Default — usually within limits |
| C3 (Building power) | 0.5 | 0.05 | 0.01 | 3× stronger — starts 6× over limit |
| C4 (Grid power) | 0.3 | 0.03 | 0.01 | Stronger than default for urgency |

**Curriculum Annealing Support:**
```python
def update_cost_limit(self, new_limit: float):
    """Rescale I-term proportionally during curriculum annealing."""
    ratio = new_limit / self.cost_limit if self.cost_limit > 0 else 1.0
    self.i_term *= ratio
    self.cost_limit = new_limit
```

### 3.4 Phase 3: Training Loop

#### 3.4.1 Epoch Structure

Each epoch = 1 CityLearn episode = 8,759 timesteps (1 year hourly).

```
For epoch in range(50):                        # Typically 50 epochs
    ┌─────────────────────────────────────────┐
    │ 1. ROLLOUT (8,759 steps)                │
    │    For step in range(8759):             │
    │      action = actor(obs)                │
    │      obs, rew, cost, done, trunc, info  │
    │        = env.step(action)               │
    │      Store: (obs, act, logp, val_r,     │
    │              val_c[0..4], cost[0..4])    │
    └─────────────────────────────────────────┘
                    │
    ┌───────────────▼─────────────────────────┐
    │ 2. CURRICULUM UPDATE                    │
    │    For each constraint with schedule:   │
    │      new_limit = interp(epoch)          │
    │      lagrange[i].update_cost_limit()    │
    └─────────────────────────────────────────┘
                    │
    ┌───────────────▼─────────────────────────┐
    │ 3. PER-CONSTRAINT GAE                   │
    │    For i in 0..4:                       │
    │      Compute GAE with costs[i], vals[i] │
    │      Z-score normalize advantages       │
    └─────────────────────────────────────────┘
                    │
    ┌───────────────▼─────────────────────────┐
    │ 4. POLICY UPDATE (30 iterations)        │
    │    For iter in range(update_iters):     │
    │      For minibatch in DataLoader:       │
    │        Update reward critic             │
    │        Update 5 cost critics            │
    │        Update actor (softmax or GradS)  │
    │      if KL > target_kl: break (early)   │
    └─────────────────────────────────────────┘
                    │
    ┌───────────────▼─────────────────────────┐
    │ 5. LAGRANGE UPDATE                      │
    │    For i in 0..4:                       │
    │      ep_cost_i = sum of costs[i]        │
    │      lambda[i] = pid_update(ep_cost_i)  │
    └─────────────────────────────────────────┘
                    │
    ┌───────────────▼─────────────────────────┐
    │ 6. LOGGING & CHECKPOINTING              │
    │    Log: EpRet, EpCost[0..4], Lambda[0..4]│
    │    Save checkpoint every N epochs       │
    └─────────────────────────────────────────┘
```

#### 3.4.2 Actor Update — Softmax Advantage Selection

The key innovation in PPOLagMulti is per-timestep softmax weighting over constraint advantages:

```python
def _compute_adv_surrogate(self, adv_r, adv_cs, lambdas):
    """
    Softmax-weighted constraint advantage prevents cancellation
    when constraints conflict.

    adv_r:   [B] reward advantage
    adv_cs:  [B, 5] per-constraint advantages (z-scored)
    lambdas: [5] Lagrange multipliers
    """
    # Softmax weights per timestep
    weights = softmax(lambdas * adv_cs / tau, dim=-1)  # [B, 5]

    # Weighted constraint advantage
    adv_c_combined = (weights * adv_cs).sum(dim=-1)     # [B]

    # Final advantage
    return adv_r - adv_c_combined
```

**Why softmax?** Standard PPO-Lag sums λ_i × A_c_i, which can cancel when one constraint wants charge and another wants discharge. Softmax focuses the gradient on the most-violated constraint at each timestep.

#### 3.4.3 Actor Update — GradS Alternative

Optional gradient surgery mode (Yao et al., L4DC 2024):

```python
def _update_actor_with_grads(self):
    """6 backward passes: 1 reward + 5 constraints."""
    # 1. Compute reward gradient
    g_reward = backward(L_reward)

    # 2. For each constraint, compute gradient
    for i in range(5):
        g_cost_i = backward(L_cost_i)

    # 3. Select non-conflicting constraint
    selected = GradSSelector.select(g_reward, g_costs, lambdas)

    # 4. Final gradient
    g_final = g_reward + lambda_sel * scale * g_cost_selected
```

#### 3.4.4 Curriculum Learning (R18 Innovation)

Three-phase training schedule that resolves gradient conflict between V2G exploration and EV departure safety:

```
Phase 1 (Epochs 0-19):  C0 disabled (limit=999,999)
  → Agent learns V2G freely from C3/C4 signals
  → Develops battery management strategy

Phase 2 (Epochs 20-35): C0 anneals 999,999 → 1,800
  → Agent gradually learns departure timing
  → PID λ_0 ramps up smoothly (no oscillation)

Phase 3 (Epochs 35-49): All constraints active
  → Fine-tuning with full constraint set
  → Agent balances V2G vs departure safety
```

**Configuration:**
```yaml
multi_cfgs:
  cost_limit_0: 999999
  anneal_cost_limit_0: [999999, 1800, 20, 35]
```

**Implementation:**
```python
# PPOLagMulti._update_cost_limits(epoch)
for i, schedule in self._anneal_schedules.items():
    start_val, end_val, start_ep, end_ep = schedule
    if start_ep <= epoch <= end_ep:
        progress = (epoch - start_ep) / (end_ep - start_ep)
        new_limit = start_val + progress * (end_val - start_val)
        self._per_lagranges[i].update_cost_limit(new_limit)
```

### 3.5 Phase 4: Reward Computation (STEMS 19-Component)

The reward function is a critical piece of domain engineering. It uses the **STEMS** (Sustainable TEnsor-based MultiScale) framework with 19 additive components:

```python
def _stems_reward(self, info, action_np, action_pre_ev_clamp=None):
    """19-component reward with progressive enhancements (R1→R30)."""

    # === Economic Signals ===
    r_eco        = -mu × price × (imports - export_factor × exports)
    r_price_arb  = -action × price  # Simple price arbitrage (R28)

    # === Grid Stability ===
    r_sg         = alpha_grid × (1 - (imports/P_grid_max)²)
    r_ramp       = -beta × |ΔNEC| / P_grid_max
    r_grid_mild  = -alpha_mild × (imports / P_grid_max)
    r_grid_penalty = quadratic grid penalty (R29)

    # === Building Stability ===
    r_sb         = alpha_build × mean(1 - |NEC_b|/P_bmax)  # Asymmetric for V2G
    r_headroom   = building power margin penalty (R26)
    r_nec_sign   = NEC-sign alignment reward (R30)

    # === Renewable Integration ===
    r_ren        = xi × min(solar / (solar + imports), 1.0)

    # === EV-Specific ===
    r_ev         = lambda_ev × dense_EV_urgency           # R13
    r_ev_guard   = anti-discharge penalty                  # R15a
    r_v2g_ctx    = context-aware V2G timing                # R15b
    r_ev_solar   = EV charge during solar surplus          # R23
    r_ev_slack   = slack-gated EV arbitrage                # R25

    # === Battery Management ===
    r_peak_shave = battery ↔ building peak correlation     # R15c
    r_solar_store = charge during solar surplus             # R24
    r_barrier    = SoC boundary penalty

    # === Load Shifting ===
    r_load_shift = solar-aware price arbitrage              # R16

    return sum(all_components)
```

**V2G-Specific Fixes (Critical for Correct Learning):**
- **Fix 1** (`STEMS_SB_ASYMMETRIC=1`): `r_sb` only penalizes imports, enabling V2G exports
- **Fix 3** (`STEMS_LAMBDA_EV=2.0`): Dense EV urgency shaping
- **Fix 4** (`STEMS_SG_EXPORT_CREDIT=0.5`): Partial credit for grid exports in `r_sg`

### 3.6 Phase 5: Logging & Monitoring

#### 3.6.1 Three-Tier CSV Logging (KPILogger)

| Tier | File | Granularity | Columns |
|------|------|-------------|---------|
| **Per-Step KPIs** | `{run}.csv` | 8,759 rows/episode | 212+ columns |
| **Cost Breakdown** | `{run}_costs.csv` | 8,759 rows/episode | 12 columns |
| **Episode Summary** | `{run}_episode_summary.csv` | 1 row/episode | 22 columns |

**Auto-Schema Upgrade:** If columns change mid-training, the logger rewrites the file with the new header and fills missing columns with 0.0.

#### 3.6.2 OmniSafe Native Logging

Standard OmniSafe metrics via TensorBoard:
- `EpRet`, `EpCost`, `EpLen`
- `LagrangianMultiplier`
- `PolicyLoss`, `ValueLoss`
- `StopIter`, `Entropy`, `KL`

#### 3.6.3 Per-Constraint Metrics

Custom metrics added to OmniSafe's logger:
- `Metrics/EpCost_0` through `Metrics/EpCost_4`
- `Metrics/Lambda_0` through `Metrics/Lambda_4`
- `Metrics/V2GDischargeCount`

### 3.7 Phase 6: Evaluation & Deployment

#### 3.7.1 Checkpoint Evaluation

```python
# eval_all_5bld.py
def run_eval(name, ckpt_path, algo, results_dir):
    cmd = [
        "python", "scripts/evaluate_trained_agent.py",
        "--checkpoint", ckpt_path,
        "--schema", "5bld",
        "--months", "12",        # Full year evaluation
        "--algo", algo,
        "--output", out_path,
    ]
    subprocess.run(cmd, timeout=900)  # 15 min timeout
```

#### 3.7.2 Evaluation Metrics Hierarchy

**Tier 1 — Step-Level Constraint Violations:**
```
C0_violation_pct = (EV departure deficits > 0) / total_steps × 100
C3_violation_pct = (|NEC_b| > P_bmax events) / total_steps × 100
C4_violation_pct = (grid_import > P_grid_max events) / total_steps × 100
```

**Tier 2 — Episode-Level CityLearn KPIs (vs RBC baseline):**
- `cost_ratio` — Electricity cost vs baseline (<1.0 is better)
- `carbon_ratio` — CO₂ emissions vs baseline
- `peak_demand` — Daily peak average (kW)
- `ramping_score` — Grid ramp stability

**Tier 3 — V2G-Specific Metrics:**
- `EV_charge_pct` — % of steps with EV charging
- `V2G_pct` — % of steps with vehicle-to-grid discharge
- `V2G_peak_pct` — % of V2G during peak hours
- `battery_cycling_days_pct` — Battery utilization

**Tier 4 — Training Quality:**
- Price correlation — Agent's response to price signals
- Peak NEC (P95) — 95th percentile net consumption

#### 3.7.3 Master Comparison Table

```
Run                      Ep  TotRew   MeanRew  C0 Viol% C0 Deps C3 Viol% C4 Viol%
────────────────────────────────────────────────────────────────────────────────────
r25b_benchmark_ep80      80  -28,412  -3.24    0.12%    3       2.41%    1.82%
r18_curriculum_ep49      49  -31,650  -3.61    0.89%    12      4.12%    2.95%
r15d_pid_fix_ep48        48  -35,100  -4.01    3.21%    45      5.87%    3.41%
...
```

**Sort Order:** Primary: C0 violation % (ascending) → Secondary: Total reward (descending)

### 3.8 Data Flow Diagram (Complete)

```
                        ┌──────────────────────────┐
                        │    CityLearn CSV Data     │
                        │  (loads, solar, pricing,  │
                        │   weather, EV schedules)  │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │  CityLearn Base Env       │
                        │  (hourly building sim)    │
                        └────────────┬─────────────┘
                                     │ obs (list), reward (list)
                        ┌────────────▼─────────────┐
                        │  SingleAgentListAdapter   │
                        │  [Box]→Box, [r]→r         │
                        └────────────┬─────────────┘
                                     │ obs (np), reward (float)
                        ┌────────────▼─────────────┐
                        │  CityLearnSafetyEnv    │
                        │  +costs, +KPIs, +actions  │
                        └────────────┬─────────────┘
                                     │ obs, reward, info{costs}
                ┌────────────────────▼──────────────────────┐
                │         Observation Wrappers               │
                │  Forecast(+128) → Temporal → Spatial(+68)  │
                └────────────────────┬──────────────────────┘
                                     │ augmented obs
                ┌────────────────────▼──────────────────────┐
                │       Constraint Wrappers                  │
                │  SauteEV(+1 budget) → SauteC4(+1 budget)   │
                └────────────────────┬──────────────────────┘
                                     │ obs + budgets
                ┌────────────────────▼──────────────────────┐
                │        Action Wrappers                     │
                │  ActionMask OR ActionProjectionSERL         │
                └────────────────────┬──────────────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │  CityLearnCMDP(CMDP)   │
                        │  · _stems_reward (19 comp)│
                        │  · _rebalanced_cost       │
                        │  · Sauté reward shaping   │
                        │  · torch tensor output    │
                        └────────────┬─────────────┘
                                     │ (obs_t, rew_t, cost_t, ...)
                        ┌────────────▼─────────────┐
                        │  _MultiCostAdapter        │
                        │  · Per-constraint costs   │
                        │  · Per-constraint critics  │
                        │  · Episode boundaries     │
                        └────────────┬─────────────┘
                                     │ rollout buffer
                        ┌────────────▼─────────────┐
                        │     PPOLagMulti           │
                        │  · Per-constraint GAE     │
                        │  · Softmax advantage      │
                        │  · PID lambda updates     │
                        │  · Curriculum annealing   │
                        └────────────┬─────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              ┌──────────┐   ┌──────────┐    ┌──────────┐
              │Checkpoint│   │TensorBoard│   │ KPI CSVs │
              │ .pt file │   │  Logs     │   │ (3 tiers)│
              └──────────┘   └──────────┘    └──────────┘
                    │
                    ▼
              ┌──────────┐
              │  Batch   │
              │  Eval    │
              │ Pipeline │
              └──────────┘
```

---

## 4. Algorithm Benchmarking Strategy

### 4.1 Algorithm Suite

The benchmarking framework evaluates Safe RL algorithms across three categories:

#### 4.1.1 On-Policy Algorithms

| Algorithm | Base | Constraint Method | Key Property |
|-----------|------|-------------------|-------------|
| **PPO** | PPO | None (unconstrained) | Baseline performance ceiling |
| **PPOLag** | PPO | Single Lagrange multiplier | Standard CMDP |
| **PPOLagMulti** | PPO | 5 per-constraint Lagrange (PID) | **Primary algorithm** |
| **TRPO** | TRPO | None | Trust region baseline |
| **TRPOLag** | TRPO | Lagrangian | Conservative policy updates |
| **CPO** | TRPO | Constrained optimization | Guaranteed improvement |
| **PCPO** | TRPO | Projected constraint | CPO with projection |
| **RCPO** | PPO | Reward-constrained | Penalty-based |

#### 4.1.2 Off-Policy Algorithms

| Algorithm | Base | Constraint Method |
|-----------|------|-------------------|
| **SAC** | SAC | None (unconstrained) |
| **SACLag** | SAC | Single Lagrange |
| **SACLagMulti** | SAC | 5 per-constraint Lagrange |
| **TD3** | TD3 | None |
| **TD3Lag** | TD3 | Single Lagrange |
| **DDPG** | DDPG | None |
| **DDPGLag** | DDPG | Single Lagrange |

#### 4.1.3 Custom Algorithm Variants

| Variant | Innovation | Reference |
|---------|-----------|-----------|
| PPOLagMulti + PID | Per-constraint PID λ control | Stooke et al., ICML 2020 |
| PPOLagMulti + GradS | Gradient surgery for conflict resolution | Yao et al., L4DC 2024 |
| PPOLagMulti + Curriculum | Phased constraint activation | Turchetta et al., NeurIPS 2020 |
| PPOLagMulti + Sauté | Budget-based constraint conversion | Sootla et al., ICML 2022 |

### 4.2 Benchmarking Methodology

#### 4.2.1 Standardized Configuration

All algorithms share a common base configuration for fair comparison:

```yaml
# Base configuration template
env_id: "CityLearnSafety-V2G-v2"
train_cfgs:
  total_steps: 437950               # 50 epochs × 8,759 steps
  vector_env_nums: 1
algo_cfgs:
  steps_per_epoch: 8759             # 1 epoch = 1 year
  batch_size: 128                   # or 256 for stable runs
  obs_normalize: true
  reward_normalize: true
  cost_normalize: false
  standardized_cost_adv: true       # Z-score cost advantages
model_cfgs:
  actor:
    hidden_sizes: [256, 256]
    activation: tanh
    lr: 0.0003
  critic:
    hidden_sizes: [256, 256]
    activation: tanh
    lr: 0.001
```

#### 4.2.2 Ablation Study Design (R15 Series)

Systematic ablation to isolate the contribution of each innovation:

| Run | Adds Over Previous | Key Env Vars |
|-----|-------------------|--------------|
| **R15a** | Sauté fix (d=25000, shaped=10) + EV clamp + guard | `CLAMP=1, GUARD=5.0` |
| **R15b** | + V2G context reward | `V2G_CONTEXT=3.0` |
| **R15c** | + Peak shaving reward | `PEAK_SHAVE=2.0` |
| **R15d** | + PID integral gain for C0 | `pid_ki_0: 0.05` (YAML) |

Each run isolates exactly one feature, enabling clear causal attribution.

#### 4.2.3 Run Lineage

```
R8 (STEMS encoder)
 └─► R9 (C3 focus)
      └─► R10 (V3 env)
           └─► R11b (multi-lag)
                └─► R12a (+ Sauté C1)
                     └─► R13 (V2G fixes)
                          └─► R14 (PID Lagrangian)
                               └─► R15a-d (4-run ablation)
                                    └─► R16-R17 (reward reformulation)
                                         └─► R18 (curriculum learning)
                                              └─► R19-R30 (variants)
                                                   └─► R25b (benchmark suite)
```

### 4.3 Evaluation Environment

#### 4.3.1 Standard Evaluation Protocol

```python
# Evaluation parameters (consistent across all runs)
SCHEMA        = "5bld"               # 5-building scenario
DURATION      = 12                    # months (full year)
SEED          = 42                    # Fixed for reproducibility
TIMEOUT       = 900                   # seconds (15 minutes)
DETERMINISTIC = False                 # Stochastic policy (no argmax)
```

#### 4.3.2 Baselines

| Baseline | Description | Purpose |
|----------|-------------|---------|
| **RBC Greedy** | Rule-based control (CityLearn default) | Primary comparison target |
| **No Control** | Zero actions (devices idle) | Lower bound |
| **Intelligent RBC** | Enhanced RBC with peak-aware scheduling | Strong heuristic |
| **SmartV2GRBC** | RBC with V2G awareness | V2G-specific baseline |

### 4.4 Metrics Framework

#### 4.4.1 Primary Metrics (Constraint Satisfaction)

| Metric | Formula | Target |
|--------|---------|--------|
| C0 Violation % | `departures_with_deficit / total_departures × 100` | < 1% |
| C3 Violation % | `steps_with_|NEC|>P_bmax / total_steps × 100` | < 5% |
| C4 Violation % | `steps_with_import>P_gmax / total_steps × 100` | < 3% |
| Total Cost | `Σ(all constraint costs over 8,759 steps)` | Minimize |

#### 4.4.2 Secondary Metrics (Performance)

| Metric | Formula | Target |
|--------|---------|--------|
| Total Reward | `Σ(STEMS reward over 8,759 steps)` | Maximize |
| Cost Ratio | `agent_cost / RBC_cost` | < 1.0 |
| Carbon Ratio | `agent_CO₂ / RBC_CO₂` | < 1.0 |
| Peak Demand | `P95(aggregate NEC)` | Minimize |
| ZNE | Binary zero-net-energy achievement | Achieve |

#### 4.4.3 V2G-Specific Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| V2G % | `discharge_steps / total_steps × 100` | > 5% |
| V2G Peak % | `V2G_during_peak / total_V2G × 100` | > 60% |
| EV Charge % | `charge_steps / total_steps × 100` | > 50% |
| Price Correlation | `corr(action, -price)` | > 0.3 |

#### 4.4.4 Training Quality Metrics

| Metric | Diagnostic Value |
|--------|-----------------|
| StopIter oscillation | λ windup → PID needed |
| Lambda trajectory | Convergence vs divergence |
| Reward plateau | Stuck agent → reward redesign |
| Cost decreasing despite reward flat | Constraint learning working |

### 4.5 Comparative Analysis Framework

#### 4.5.1 Master Table Generation

```python
# eval_all_5bld.py — Batch evaluation across all trained models
def print_master_table(all_results):
    """Print formatted comparison table sorted by C0 violation rate."""
    header = (
        f"{'Run':<30s} {'Ep':>4s} "
        f"{'TotRew':>8s} {'MeanRew':>8s} "
        f"{'C0 Viol%':>8s} {'C0 Deps':>7s} "
        f"{'C3 Viol%':>8s} {'C4 Viol%':>8s} "
        f"{'EV Chg%':>7s} {'V2G%':>5s} {'V2GPk%':>6s} "
        f"{'BtCyc%':>6s} {'PrCorr':>7s} "
        f"{'PkNEC':>7s} {'C0Cost':>7s} {'TotCost':>8s}"
    )
    # Sort: primary=C0_viol ascending, secondary=TotRew descending
    sorted_results = sorted(results, key=lambda r: (r['c0_viol'], -r['tot_rew']))
```

#### 4.5.2 Algorithm Comparison Dimensions

The benchmarking evaluates algorithms across 4 dimensions:

```
                    Constraint Satisfaction
                           ▲
                           │
                    ┌──────┼──────┐
                    │      │      │
                    │  ★   │      │  ← Ideal: low cost + low violation
                    │      │      │
    Performance ────┼──────┼──────┼──── Sample Efficiency
                    │      │      │
                    │      │      │
                    │      │      │
                    └──────┼──────┘
                           │
                           ▼
                    V2G Utilization
```

1. **Constraint Satisfaction:** C0/C3/C4 violation rates
2. **Performance:** Total reward, cost ratio vs RBC
3. **Sample Efficiency:** Epochs to convergence
4. **V2G Utilization:** Discharge %, peak discharge %, price correlation

#### 4.5.3 Statistical Significance

For multi-seed experiments:
- 3-5 random seeds per algorithm
- Report mean ± std for all metrics
- Paired comparison against RBC baseline
- Training curves plotted with confidence bands

### 4.6 Experiment Configuration Examples

#### 4.6.1 R18 Curriculum Learning (Current Best)

```yaml
# configs/on-policy/r18_curriculum.yaml
algo: PPOLagMulti
env_id: CityLearnSafety-V2G-v2
seed: 42

train_cfgs:
  total_steps: 437950                # 50 epochs

algo_cfgs:
  steps_per_epoch: 8759
  update_iters: 30
  target_kl: 0.12
  batch_size: 128
  entropy_coef: 0.005
  max_grad_norm: 40.0

multi_cfgs:
  tau: 1.0
  cost_limit_0: 999999              # C0: disabled initially
  cost_limit_1: 1500
  cost_limit_2: 4000
  cost_limit_3: 12000
  cost_limit_4: 2500
  anneal_cost_limit_0: [999999, 1800, 20, 35]  # Curriculum schedule
  pid_kp: 0.1
  pid_ki: 0.01
  pid_kp_0: 5.0                     # C0: strong proportional
  pid_ki_0: 0.0                     # C0: no integral
  pid_kp_3: 0.5                     # C3: 3× stronger
  pid_ki_3: 0.05
```

**Shell Script:**
```bash
# run_r18_curriculum.sh
export CITYLEARN_PID_LAGRANGE="1"
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"
export STEMS_LAMBDA_EV="5.0"
export STEMS_SB_ASYMMETRIC="1"
export STEMS_SG_EXPORT_CREDIT="0.5"
export CITYLEARN_EV_ACTION_CLAMP="0"    # R18: NO clamps
export CITYLEARN_BATT_CLAMP="0"         # R18: pure RL learning

python scripts/train_multi_lag.py --cfg configs/on-policy/r18_curriculum.yaml
```

#### 4.6.2 R25b Benchmark (Stable Reporting)

```yaml
# configs/on-policy/r25b_report_stable.yaml
algo: PPOLagMulti
train_cfgs:
  total_steps: 700720               # 80 epochs

algo_cfgs:
  target_kl: 0.06                   # Tighter KL
  batch_size: 256                   # Larger batches
  update_iters: 10                  # Fewer iterations

multi_cfgs:
  cost_limit_0: 999999
  anneal_cost_limit_0: [999999, 1800, 20, 40]  # Longer annealing
```

### 4.7 Key Benchmarking Findings

#### Historical Results (R12a, 48 Epochs)

| Metric | Value | Trend |
|--------|-------|-------|
| C1 (Sauté) | 0.0 all epochs | Sauté perfectly eliminates C1 |
| C3 (Building) | 76K → 24.5K | -68% (improving) |
| C4 (Grid) | 36K → 5.2K | -86% (strong improvement) |
| C0 (EV departure) | 989 → 1,955 | Getting WORSE (motivation for curriculum) |
| EpRet | ~-39,650 | Barely moving (reward plateau) |
| StopIter | 30/1 oscillation | Lambda integral windup |

**Key Insight:** C3/C4 improve while C0 degrades — conflicting gradients between V2G exploration and departure safety. This motivated the **curriculum learning** approach (R18).

#### Design Decisions from Benchmarking

| Observation | Decision | Run |
|-------------|----------|-----|
| StopIter 30/1 oscillation | Replace SGD with PID Lagrangian | R14 |
| C0 worsening during V2G learning | Curriculum annealing for C0 | R18 |
| Sauté budget depleted at step 580 | Increase d from 1,500 to 25,000 | R15 |
| Binary penalty kills 93% of gradient | Shaped penalty (α=10) | R15 |
| Lambda grows linearly despite cost decrease | Per-constraint gains | R14 |
| r_sb penalizes V2G exports | Asymmetric r_sb (Fix 1) | R13 |
| r_sg penalizes grid exports | Export credit (Fix 4) | R13 |

---

## Appendix A: Key File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `citylearn_safe/cmdp_env.py` | 1,766 | CMDP adapter + STEMS reward |
| `citylearn_safe/safety_env.py` | 1,900+ | Inner safety layer + KPI computation |
| `citylearn_safe/grads/ppo_lag_multi.py` | 884 | PPOLagMulti algorithm |
| `citylearn_safe/grads/ppo_lag_grads.py` | 750+ | GradS variant |
| `citylearn_safe/pid_lagrange.py` | 146 | PID controller for λ |
| `citylearn_safe/saute_ev_wrapper.py` | 109 | Sauté MDP for EV |
| `citylearn_safe/kpi_logger.py` | ~500 | 212-column CSV logging |
| `citylearn_safe/extractors.py` | 1,900+ | KPI extraction |
| `citylearn_safe/stems_encoder.py` | 1,900+ | STEMS reward encoding |
| `citylearn_safe/action_mask_wrapper.py` | ~400 | Gradient-safe action rescaling |
| `citylearn_safe/psf/psf_wrapper.py` | ~600 | Predictive safety filter |
| `scripts/train_multi_lag.py` | ~200 | Training entry point |
| `scripts/benchmark_algos.py` | ~150 | Multi-algorithm benchmarking |
| `eval_all_5bld.py` | ~300 | Batch evaluation |

## Appendix B: Environment Variable Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `CITYLEARN_SCHEMA` | — | Path to CityLearn schema JSON |
| `CITYLEARN_CENTRAL_AGENT` | `"1"` | Enable central agent mode |
| `CITYLEARN_REWARD_TYPE` | `"stems"` | Reward function selection |
| `CITYLEARN_PID_LAGRANGE` | `"0"` | Enable PID Lagrangian |
| `CITYLEARN_EV_SAUTE` | `"0"` | Enable Sauté MDP for C1 |
| `CITYLEARN_EV_SAUTE_BUDGET` | `"1500"` | Sauté budget d |
| `CITYLEARN_EV_SAUTE_SHAPED_ALPHA` | `"0"` | Shaped penalty (>0) vs binary (0) |
| `CITYLEARN_EV_ACTION_CLAMP` | `"0"` | Enable EV discharge guard |
| `CITYLEARN_BATT_CLAMP` | `"0"` | Enable battery SoC clamp |
| `CITYLEARN_SERL_PROJECTION` | `"0"` | Enable SE-RL execution shield |
| `CITYLEARN_ACTION_MASK` | `"0"` | Enable action mask wrapper |
| `CITYLEARN_WM_DISABLE` | `"0"` | Disable washing machine actions |
| `STEMS_MU_ECONOMIC` | `"0.3"` | Economic reward weight |
| `STEMS_ALPHA_GRID` | `"3.0"` | Grid stability weight |
| `STEMS_ALPHA_BUILD` | `"2.0"` | Building stability weight |
| `STEMS_LAMBDA_EV` | `"0.0"` | Dense EV reward weight |
| `STEMS_SB_ASYMMETRIC` | `"0"` | Fix 1: asymmetric r_sb |
| `STEMS_SG_EXPORT_CREDIT` | `"0.0"` | Fix 4: grid export credit |
| `STEMS_ALPHA_V2G_CONTEXT` | `"0.0"` | V2G context reward weight |
| `STEMS_ALPHA_PEAK_SHAVE` | `"0.0"` | Peak shaving reward weight |
| `STEMS_ALPHA_EV_GUARD` | `"0.0"` | Anti-discharge penalty weight |

## Appendix C: Academic References

| Innovation | Paper | Venue |
|-----------|-------|-------|
| PID Lagrangian | Stooke et al., "Responsive Safety in RL by PID Lagrangian Methods" | ICML 2020 |
| Sauté MDP | Sootla et al., "Sauté RL: Almost Surely Safe RL Using State Augmentation" | ICML 2022 |
| Curriculum Induction | Turchetta et al., "Safe RL via Curriculum Induction" | NeurIPS 2020 |
| GradS | Yao et al., "Gradient Surgery for Multi-Constraint Safe RL" | L4DC 2024 |
| SE-RL | Markgraf et al., "Safe Execution RL with Action Projection" | 2025 |
| STEMS | Rounsavall et al., "STEMS: Multi-Scale Reward for Smart Grid RL" | 2023 |
| CityLearn | Vazquez-Canteli et al., "CityLearn: Demand Response in Grid-Interactive Buildings" | ACM e-Energy |
| OmniSafe | Ji et al., "OmniSafe: An Infrastructure for Accelerating Safe RL Research" | JMLR 2024 |

---

*Report generated via multi-agent analysis of 55 Python modules, 190+ YAML configurations, and 137 shell scripts comprising ~23,000 lines of core Safe-RL code.*
