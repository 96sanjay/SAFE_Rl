# Safe-CityLearn: Project Architecture Overview

## 📋 Executive Summary

**Safe-CityLearn** is a benchmarking suite for Safe Reinforcement Learning (Safe RL) algorithms applied to multi-building energy management. It integrates the **CityLearn** environment (a building energy simulation) with **OmniSafe** (a Safe RL library) to learn battery control policies that optimize energy costs while respecting safety constraints (battery State of Charge bounds).

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OmniSafe Training                        │
│  (PPO-Lag, CPO, SACLag, etc.)                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           CityLearnCMDP (omni_env.py)                       │
│  • OmniSafe CMDP interface adapter                          │
│  • Converts torch tensors ↔ numpy arrays                    │
│  • Extracts cost from info dict                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│       CityLearnSafetyEnv (safety_env.py)                    │
│  • Adds safety cost (SoC band violation)                    │
│  • Computes KPIs at each step                               │
│  • Logs to custom CSV logger                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│    SingleAgentListAdapter (adapters.py)                    │
│  • Converts [Box] → Box (unwraps list)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│     NormalizedObservationWrapper (CityLearn)                 │
│  • Normalizes observations to [0, 1]                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│        CityLearnEnv (CityLearn library)                      │
│  • Multi-building energy simulation                          │
│  • 1 building (Building_1) in current config                │
│  • 8759 timesteps per episode (1 year, hourly)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

### Core Modules

#### 1. **`citylearn_safe/safety_env.py`** ⭐ Core Safety Environment
- **Purpose**: Wraps CityLearn environment to add CMDP-style safety costs
- **Key Features**:
  - **Safety Constraint**: Battery SoC must stay within `[0.1, 0.9]`
  - **Cost Function**: `_soc_band_cost()` computes normalized violation cost
    - Formula: `(low_violation + high_violation) / band_width`
    - Low violation: `max(0, soc_min - min_observed_soc)`
    - High violation: `max(0, max_observed_soc - soc_max)`
  - **KPI Computation**: Tracks 30+ metrics per step:
    - Observation stats (mean, std, min, max)
    - SoC stats (mean, min, max, std)
    - Action stats (mean, std, min, max)
    - Energy metrics (net consumption, import/export, solar generation)
    - Economic metrics (electricity price, step cost)
    - Environmental (outdoor/indoor temperature)
    - Constraint violations (binary flag)
  
- **Important Methods**:
  - `reset()`: Resets episode, computes initial KPIs, logs reset
  - `step()`: Steps environment, computes cost & KPIs, logs to CSV
  - `_compute_basic_kpis()`: Computes all step-wise KPIs
  - `_extract_citylearn_kpis()`: Extracts CityLearn episode-end KPIs
  - `_soc_band_cost()`: Computes safety cost from SoC violations

#### 2. **`citylearn_safe/omni_env.py`** 🔌 OmniSafe Adapter
- **Purpose**: Adapts `CityLearnSafetyEnv` to OmniSafe's CMDP interface
- **Key Features**:
  - Registers environment ID: `CityLearnSafety-SoC-v0`
  - Converts numpy arrays ↔ torch tensors
  - Extracts `cost` from `info` dict for OmniSafe
  - Handles reset/step/seed/close methods
  - Sets `max_episode_steps = 8759`

#### 3. **`citylearn_safe/adapters.py`** 🔄 Space Adapter
- **Purpose**: Converts CityLearn's list-wrapped spaces to plain Box
- **Problem Solved**: CityLearn returns `observation_space = [Box(...)]` but OmniSafe expects `Box(...)`
- **Solution**: Unwraps `[x]` → `x` for both observations and actions

#### 4. **`citylearn_safe/kpi_logger.py`** 📊 Custom KPI Logger
- **Purpose**: Tracks KPIs separately from OmniSafe's default logging
- **Features**:
  - Writes to CSV: `{run_dir}/kpis_kpis.csv`
  - Logs step-wise metrics (one row per timestep)
  - Logs episode-end summaries (with CityLearn KPIs)
  - Auto-detects run directory by timestamp
  - Writes incrementally (every 10 steps)

#### 5. **`citylearn_safe/schema_index.py`** 🔍 Observation Index Helper
- **Purpose**: Finds observation indices for specific features (e.g., SoC)
- **Key Function**: `soc_indices_from_schema_and_obs_dim()`
  - Parses CityLearn schema JSON
  - Calculates observation indices for `electrical_storage_soc`
  - Handles shared vs per-building observations

### Scripts

#### 6. **`scripts/make_env.py`** 🏭 Environment Factory
- **Purpose**: Creates base CityLearn environment with wrappers
- **Process**:
  1. Loads schema JSON from `CITYLEARN_SCHEMA`
  2. Creates `CityLearnEnv` with `central_agent=True`
  3. Applies `NormalizedObservationWrapper`
  4. Applies `SingleAgentListAdapter`
  5. Returns wrapped environment

#### 7. **`scripts/train_omnisafe.py`** 🚂 Training Entry Point
- **Purpose**: Main training script
- **Usage**: `python -m scripts.train_omnisafe --cfg configs/ppo_lag_soc.yaml`
- **Process**:
  1. Loads YAML config
  2. Creates OmniSafe agent with config
  3. Calls `agent.learn()` to start training
  4. Registers environment via `import citylearn_safe.omni_env`

#### 8. **`scripts/benchmark_algos.py`** 📈 Benchmark Runner
- **Purpose**: Trains multiple algorithms sequentially
- **Supported Algorithms**: 11 total
  - On-policy: PPO, PPOLag, TRPO, TRPOLag, CPO, PCPO, RCPO
  - Off-policy: SAC, SACLag, TD3, TD3Lag, DDPG, DDPGLag

#### 9. **`scripts/plot_kpis.py`** 📉 Visualization Tool
- **Purpose**: Generates plots from KPI CSV files
- **Features**:
  - 3x3 subplot layout
  - Plots SoC, violations, cost, consumption, actions
  - CityLearn episode-end KPIs (electricity, carbon, cost)
  - Saves to `kpi_plots.png`

---

## 🔬 Constraint & Cost Mechanism

### Safety Constraint: Battery SoC Band
- **Constraint**: Battery SoC must stay within `[0.1, 0.9]`
- **Violation Detection**: Binary flag (`constraint_violation = 1.0` if violated, `0.0` otherwise)
- **Cost Function**: Normalized violation magnitude
  ```python
  low_violation = max(0.0, soc_min - min_observed_soc)
  high_violation = max(0.0, max_observed_soc - soc_max)
  cost = (low_violation + high_violation) / band_width
  ```
- **Cost Interpretation**:
  - `cost = 0.0`: No violation (SoC within bounds)
  - `cost > 0.0`: Violation magnitude (normalized by band width)
  - Used by OmniSafe for constraint satisfaction

### Alternative Cost Functions (Commented Out)
The code includes commented examples for:
- **Binary cost**: Fixed penalty (1.0) per violation
- **Weighted binary**: Different penalties for undercharge vs overcharge
- **Per-building cost**: Count violations per building

---

## 📊 KPI Tracking System

### Step-Wise KPIs (Every Timestep)
1. **Observation Statistics**: `obs_mean`, `obs_std`, `obs_min`, `obs_max`
2. **SoC Statistics**: `soc_mean`, `soc_min`, `soc_max`, `soc_std`
3. **Action Statistics**: `action_mean`, `action_std`, `action_min`, `action_max`
4. **Energy Metrics**:
   - `step_net_consumption_kwh`: Net consumption this step (kWh)
   - `grid_import_kwh`: Import from grid (kWh, ≥ 0)
   - `grid_export_kwh`: Export to grid (kWh, ≥ 0)
   - `solar_generation_kwh`: Solar PV generation (kWh)
5. **Economic Metrics**:
   - `electricity_price`: Current price ($/kWh)
   - `step_cost`: Cost for this step ($)
6. **Environmental**:
   - `outdoor_temperature`: Outdoor temperature (°C)
   - `indoor_temperature`: Indoor temperature (°C)
7. **Safety**:
   - `constraint_violation`: Binary violation flag
   - `cost`: Safety cost from CMDP

### Episode-End KPIs (CityLearn Native)
These are computed by CityLearn's `evaluate()` method:
- `citylearn_electricity_consumption_total`: Total consumption ratio (vs baseline)
- `citylearn_carbon_emissions_total`: Total carbon emissions ratio
- `citylearn_cost_total`: Total cost ratio
- `citylearn_zero_net_energy`: Zero net energy metric
- `citylearn_daily_peak_average`: Average daily peak demand
- `citylearn_discomfort_proportion`: Proportion of time in discomfort

**Note**: Values > 1.0 indicate worse than baseline, < 1.0 indicate better than baseline.

---

## 🎯 Training Configuration

### Example Config (`configs/ppo_lag_soc.yaml`)
```yaml
algo: PPOLag
env_id: CityLearnSafety-SoC-v0

train_cfgs:
  total_steps: 87590        # 10 episodes
  vector_env_nums: 1
  parallel: 1

algo_cfgs:
  steps_per_epoch: 8759    # 1 episode = 1 epoch
  target_kl: 0.05
  batch_size: 128
  update_iters: 30
  obs_normalize: true
  reward_normalize: true
  cost_normalize: true

lagrange_cfgs:
  cost_limit: 0.05         # Average cost per step target
  lagrangian_multiplier_init: 0.5
```

### Key Parameters
- **Episodes**: 8759 timesteps = 1 year (hourly resolution)
- **Training**: 10 episodes (87590 steps) default
- **Cost Limit**: 0.05 (average cost per step)
- **SoC Bounds**: `[0.1, 0.9]` (hardcoded in `omni_env.py`)

---

## 🔄 Data Flow

### Step-by-Step Flow
1. **OmniSafe** selects action (torch tensor)
2. **CityLearnCMDP** converts to numpy array
3. **CityLearnSafetyEnv** computes:
   - Safety cost from SoC violations
   - Step-wise KPIs
   - Updates `info` dict with all metrics
4. **CityLearnSafetyEnv** logs KPIs to CSV (via `KPILogger`)
5. **CityLearnCMDP** extracts cost from `info` and returns to OmniSafe
6. **OmniSafe** uses cost for constraint satisfaction

### Episode End Flow
1. Episode terminates (8759 steps or truncation)
2. **CityLearnSafetyEnv** calls `_extract_citylearn_kpis()`
3. Accesses underlying CityLearn env via `self.base.base`
4. Calls `citylearn_env.evaluate()` to get episode KPIs
5. Logs episode summary to CSV with CityLearn KPIs

---

## 🗂️ File Organization

```
safe-citylearn/
├── citylearn_safe/          # Core package
│   ├── safety_env.py        # ⭐ Main safety environment
│   ├── omni_env.py          # OmniSafe adapter
│   ├── adapters.py          # Space adapter
│   ├── kpi_logger.py        # Custom CSV logger
│   └── schema_index.py      # Observation index helper
├── scripts/                 # Utility scripts
│   ├── train_omnisafe.py    # Training entry point
│   ├── make_env.py          # Environment factory
│   ├── benchmark_algos.py   # Multi-algorithm runner
│   └── plot_kpis.py         # Visualization
├── configs/                 # Training configs
│   ├── ppo_lag_soc.yaml     # Example config
│   └── on-policy/           # On-policy configs
│   └── off-policy/          # Off-policy configs
├── data/citylearn/          # CityLearn data
│   ├── schema.json          # Environment schema
│   ├── Building_1.csv       # Building data
│   ├── weather.csv          # Weather data
│   └── pricing.csv          # Electricity pricing
└── runs/                    # Training outputs
    └── PPOLag-{CityLearnSafety-SoC-v0}/
        └── seed-000-.../
            ├── progress.csv      # OmniSafe metrics
            ├── kpis_kpis.csv     # Custom KPIs
            ├── config.json        # Training config
            └── torch_save/       # Model checkpoints
```

---

## 🎓 Key Design Decisions

### 1. **Central Agent Mode**
- CityLearn supports multi-agent (17 buildings) but this project uses `central_agent=True`
- Converts multi-agent problem → single-agent CMDP
- Single policy controls all buildings' batteries

### 2. **Normalized Observations**
- Uses `NormalizedObservationWrapper` to scale observations to [0, 1]
- Improves training stability

### 3. **Custom KPI Logger**
- OmniSafe's default logging doesn't capture all KPIs
- Custom CSV logger provides full control over KPI structure
- Auto-detects run directory by timestamp

### 4. **Cost Function Design**
- Current: Normalized violation magnitude (continuous)
- Alternatives: Binary (commented out)
- Can be extended (see comments in `safety_env.py`)

### 5. **Episode Structure**
- 1 episode = 1 year (8759 timesteps, hourly)
- 1 epoch = 1 episode (for OmniSafe logging)

---

## 🔧 Extension Points

### Adding New Constraints
1. **Observation Extraction**: Add to `schema_index.py` to find new observation indices
2. **Cost Function**: Add method to `CityLearnSafetyEnv` (e.g., `_power_constraint_cost()`)
3. **Combine Costs**: Modify `_soc_band_cost()` to combine multiple constraints

### Adding New KPIs
1. **Compute in `_compute_basic_kpis()`**: Add new metric calculation
2. **Update `kpi_logger.py`**: Add field to `fieldnames` list
3. **Extract from CityLearn**: Use `_extract_citylearn_kpis()` pattern

### Changing Cost Function
- **Binary**: Uncomment/implement binary cost (see commented code)
- **Weighted**: Different penalties for different violation types
- **Per-building**: Aggregate costs per building then sum

---

## 📚 Dependencies

- **CityLearn**: Building energy simulation environment
- **OmniSafe**: Safe RL algorithm library
- **Gymnasium**: RL environment interface
- **PyTorch**: Deep learning backend
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation (for KPIs)
- **Matplotlib**: Visualization

---

## 🚀 Quick Start

```bash
# 1. Set environment variable
export CITYLEARN_SCHEMA="$PWD/data/citylearn/schema.json"

# 2. Train single algorithm
python -m scripts.train_omnisafe --cfg configs/ppo_lag_soc.yaml

# 3. View results
tensorboard --logdir runs/
python scripts/plot_kpis.py runs/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-...

# 4. Run benchmark
python scripts/benchmark_algos.py
```

---

## 📝 Notes

- **Current Setup**: Single building (Building_1) for faster testing
- **Schema**: Located at `data/citylearn/schema.json`
- **Run Detection**: KPI logger auto-detects most recent run directory
- **Episode Numbering**: Custom logger episode count may differ from OmniSafe epoch
- **CityLearn KPIs**: Only available at episode end (after `evaluate()` call)

---

This architecture enables benchmarking Safe RL algorithms on realistic building energy management problems while maintaining clear separation between environment, safety constraints, and logging systems.


