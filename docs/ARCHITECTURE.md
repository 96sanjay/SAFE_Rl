# Safe RL for V2G Control -- System Architecture

This document describes the full architecture of the Safe Reinforcement Learning
system for Vehicle-to-Grid (V2G) control in the CityLearn smart grid environment.
The system trains a constrained policy (CMDP) that optimizes energy costs while
satisfying four safety constraints (C1--C4) covering EV departure readiness,
battery SoC limits, building power capacity, and grid power capacity.

**Target audience:** Researchers who want to understand the system in order to
reproduce, evaluate, or extend it.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Environment Stack](#2-environment-stack)
3. [Observation Space](#3-observation-space)
4. [Action Space](#4-action-space)
5. [STEMS Encoder](#5-stems-encoder)
6. [Hybrid Critic](#6-hybrid-critic)
7. [Reward Design](#7-reward-design)
8. [Constraint System](#8-constraint-system)
9. [PID Lagrangian](#9-pid-lagrangian)
10. [Training Pipeline](#10-training-pipeline)
11. [Key Design Decisions](#11-key-design-decisions)

---

## 1. System Overview

The system is a Constrained Markov Decision Process (CMDP) pipeline. CityLearn
provides the multi-building smart grid simulation. A chain of gymnasium wrappers
adds safety costs, forecast observations, temporal history, and Saute budgets.
The wrapped environment is registered with OmniSafe, which provides PPO-based
on-policy training. A custom STEMS encoder replaces OmniSafe's default MLP
actor/critic with a GCN + Temporal Transformer architecture. Four independent
PID Lagrangian controllers manage per-constraint multipliers.

```
 +------------------------------------------------------------------+
 |                         Training Loop                             |
 |  train_multi_lag_stems.py                                         |
 |                                                                   |
 |  +-----------------------------+   +---------------------------+  |
 |  |  PPOLagMulti (PPO + CMDP)   |   |  PID Lagrangian x 4      |  |
 |  |  - On-policy rollouts       |   |  - Per-constraint lambda  |  |
 |  |  - Trust region (clip)      |   |  - Anti-windup            |  |
 |  |  - GAE advantage            |   |  - Cost-limit curriculum  |  |
 |  |  - Softmax cost weighting   |   |  - EMA smoothing          |  |
 |  +-------------+---------------+   +-------------+-------------+  |
 |                |                                   |               |
 |                v                                   v               |
 |  +-----------------------------+   +---------------------------+  |
 |  |  Actor (STEMS Encoder)      |   |  Hybrid Critics           |  |
 |  |  - GCN spatial              |   |  - STEMS reward V(s)      |  |
 |  |  - Temporal Transformer     |   |  - MLP cost V_c1..V_c4    |  |
 |  |  - Gated fusion             |   |                           |  |
 |  |  - Gaussian policy head     |   |                           |  |
 |  +-------------+---------------+   +---------------------------+  |
 |                |                                                   |
 +----------------|---------------------------------------------------+
                  |
                  v
 +------------------------------------------------------------------+
 |                     Environment Stack                             |
 |                                                                   |
 |  +----------------------------------------------------------+    |
 |  |  CityLearnCMDP (OmniSafe CMDP registration)            |    |
 |  |  - STEMS reward computation                              |    |
 |  |  - Rebalanced cost aggregation                            |    |
 |  |  - Torch tensor I/O                                      |    |
 |  +---+------------------------------------------------------+    |
 |      |                                                            |
 |  +---v------------------------------------------------------+    |
 |  |  [Optional] SauteEVBudgetWrapper (C1 dense budget)        |    |
 |  |  [Optional] ActionProjectionSERL / ActionMaskWrapper      |    |
 |  +---+------------------------------------------------------+    |
 |      |                                                            |
 |  +---v------------------------------------------------------+    |
 |  |  [Optional] TemporalHistoryWrapper (T-step feature stack) |    |
 |  +---+------------------------------------------------------+    |
 |      |                                                            |
 |  +---v------------------------------------------------------+    |
 |  |  ForecastObsWrapper (24h price/load/solar lookahead)      |    |
 |  +---+------------------------------------------------------+    |
 |      |                                                            |
 |  +---v------------------------------------------------------+    |
 |  |  CityLearnSafetyEnv (cost signals, KPI logging)         |    |
 |  +---+------------------------------------------------------+    |
 |      |                                                            |
 |  +---v------------------------------------------------------+    |
 |  |  CityLearnEnv (multi-building simulation)                 |    |
 |  +----------------------------------------------------------+    |
 +------------------------------------------------------------------+
```

---

## 2. Environment Stack

The environment is a chain of gymnasium wrappers, each adding a specific
capability. The order matters: wrappers closer to CityLearnEnv transform raw
simulation outputs, while outer wrappers add derived features visible to the
agent.

### 2.1 CityLearnEnv (Base)

The unmodified CityLearn simulator. Simulates N buildings (typically 5) over
8,760 hourly steps (one year). Each building has solar PV, battery storage,
non-shiftable load, and optionally EV chargers. The environment exposes a
central-agent interface: one flat observation vector and one flat action vector
controlling all buildings simultaneously.

**Source:** Third-party (`citylearn` package).

### 2.2 CityLearnSafetyEnv

**File:** `citylearn_safe/safety_env.py`

Wraps the base environment to add:

- **Per-step cost signals** for all four constraints (C1--C4), emitted in the
  `info` dict under standardized keys.
- **EV departure cost** (C1): action-based deficit computation using the agent's
  actual charging actions, not oracle SoC readings.
- **Action clipping** to per-dimension action space bounds.
- **KPI logging** via a streaming logger for episode-level metrics.
- **Schema-agnostic action map** that discovers battery and EV charger action
  indices from `action_names` at init time.
- **Auto-calibrated power limits** (`P_building_max`, `P_grid_max`) computed as
  the 95th percentile of non-shiftable load from the schema data.

### 2.3 ForecastObsWrapper

**File:** `citylearn_safe/forecast_obs_wrapper.py`

Appends a 24-hour lookahead to the observation vector. Forecast features include
electricity price (24 dims), district load (24 dims), district solar (24 dims),
EV urgency features, and hour-of-day encoding -- totaling 128 additional dims.

### 2.4 TemporalHistoryWrapper (Optional)

**File:** `citylearn_safe/temporal_obs_wrapper.py`

Maintains a rolling T-step history of selected observation features and appends
them to each observation. Controlled by `CITYLEARN_TEMPORAL_WINDOW` (typically
T=12). Two modes:

- **Basic** (default): 11 features per step -- 5 battery SoCs, 5 net
  consumptions, 1 price. Appends 132 dims for T=12.
- **Rich** (`CITYLEARN_TEMPORAL_RICH=1`): 32 features per step -- adds solar
  generation, EV SoC/departure, hour encoding. Appends 384 dims for T=12.

This wrapper is stateless from the agent's perspective: the history is part of
the observation, so PPO importance sampling is not affected.

### 2.5 SauteEVBudgetWrapper (Optional)

**File:** `citylearn_safe/saute_ev_wrapper.py`

Implements Saute MDP (Sootla et al., 2022) for constraint C1 (EV departure
readiness). Augments the observation with a per-EV budget variable that tracks
remaining safety budget. Converts the sparse departure-time constraint into a
dense per-step penalty when the budget is depleted.

### 2.6 ActionProjectionSERL / ActionMaskWrapper (Optional)

**Files:** `citylearn_safe/action_projection_serl.py`,
`citylearn_safe/action_mask_wrapper.py`

Hard constraint enforcement at the action level. The SE-RL projector clips
actions to satisfy C2/C3/C4 power constraints by construction, reducing the
constraint satisfaction burden on the learned policy.

### 2.7 CityLearnCMDP

**File:** `citylearn_safe/cmdp_env.py`

The outermost wrapper, registered with OmniSafe via `@env_register`. This class:

1. **Assembles the wrapper chain** in `__init__`, reading environment variables
   to enable/disable optional wrappers.
2. **Computes the STEMS reward** from per-building power flows, prices, and
   agent actions. This is a composite reward with 15+ individually weighted
   components (Section 7).
3. **Aggregates cost signals** into the 6-tuple `(obs, reward, cost, terminated,
   truncated, info)` required by OmniSafe's CMDP interface.
4. **Converts all outputs to PyTorch tensors**.
5. **Manages battery and EV action maps** for per-device reward/cost computation.

---

## 3. Observation Space

The observation is a flat vector whose structure depends on which wrappers are
active. With all wrappers enabled (STEMS encoder, temporal history T=12, basic mode):

```
obs [obs_dim]
 |
 +-- Base obs (70 dims)
 |    +-- Per-building (4 features x 5 buildings = 20)
 |    |    - non_shiftable_load
 |    |    - solar_generation
 |    |    - electrical_storage_soc
 |    |    - net_electricity_consumption
 |    |
 |    +-- Per-EV charger (7 features x 3 chargers = 21)
 |    |    - connected_state
 |    |    - departure_time
 |    |    - required_soc_departure
 |    |    - soc
 |    |    - battery_capacity
 |    |    - incoming_state
 |    |    - estimated_arrival_time
 |    |
 |    +-- Global features (13)
 |         - month_cos, month_sin
 |         - day_type_cos, day_type_sin
 |         - hour_cos, hour_sin
 |         - carbon_intensity
 |         - electricity_pricing (current + 3 predicted)
 |         - washing_machine timing (optional)
 |
 +-- Forecast (128 dims, from ForecastObsWrapper)
 |    - 24h electricity price forecast (24)
 |    - 24h district load forecast (24)
 |    - 24h district solar forecast (24)
 |    - EV urgency features (variable)
 |    - Hour-of-day encoding (variable)
 |
 +-- Temporal history (T x features_per_step dims)
      - T=12 steps of: [soc_0..soc_4, net_0..net_4, price]
      - Basic: 12 x 11 = 132 dims
      - Rich:  12 x 32 = 384 dims
```

The STEMS encoder parses this flat vector into structured per-building nodes
using pre-computed index buffers (`ObsIndex` from `schema_index.py`), avoiding
any naive `obs_dim // num_buildings` splitting that would break when buildings
have different feature counts.

---

## 4. Action Space

The action space is a continuous `Box` of dimension `act_dim`, where each
dimension is in [-1, 1]. For the 5-building schema:

| Action Type | Count | Range | Description |
|---|---|---|---|
| Battery storage | 5 | [-1, 1] | Charge (+) / Discharge (-) |
| EV charger | 3 | [0, 1] or [-1, 1] | Charge (+) / V2G (-) |
| Total | 8 | | |

Not all buildings have EV chargers. The `ActionMap` discovers which action
indices correspond to batteries vs. EV chargers by parsing `action_names` from
the CityLearn environment at init time.

**Action processing:**

1. The actor outputs raw actions in [-1, 1] (or [0, 1] in beta actor mode).
2. Optional `ActionMaskWrapper` rescales actions to state-dependent safe bounds.
3. Optional `ActionProjectionSERL` clips actions to satisfy power constraints.
4. Battery SoC clamping prevents actions that would violate physical SoC bounds.
5. EV disconnect masking zeroes out actions for disconnected EVs (no-op).

---

## 5. STEMS Encoder

STEMS (Spatial-Temporal Energy Management System) is a custom neural network
architecture that replaces OmniSafe's default MLP feature extractor. It exploits
the graph structure of multi-building energy systems.

### 5.1 Architecture Overview

**File:** `citylearn_safe/stems_encoder.py`

```
obs [B, obs_dim]
 |
 +-- Split: current_obs [B, current_obs_dim] | history [B, T*F]
 |
 |   +-- Parse current_obs:
 |   |    Building features [B, N, 4] -- via index gather
 |   |    EV features [B, N, 7] -- masked for non-EV buildings
 |   |    Global features [B, G] -- time encoding + forecast
 |   |    (optional) Forecast temporal [B, 24, 3]
 |   |
 |   +-- Global Encoder:
 |   |    global_raw [B, G] -> MLP(G->64->32) -> global_emb [B, 32]
 |   |    Broadcast to [B, N, 32]
 |   |
 |   +-- Node Input: concat [B, N, 4+7+32=43]
 |   |    -> ObservationEmbedding (MLP 43->64->64)
 |   |    -> node_embed [B, N, 64]
 |   |
 |   +-- Parse history:
 |        history [B, T*F] -> reshape [B, T, F]
 |        -> per-node gather -> [B, N, T, K]  (K=3: soc_i, net_i, price)
 |
 +-- Per-Node Temporal Transformer (shared weights across N nodes):
 |    [B*N, T, K] -> Linear(K, 32) -> + SinusoidalPE
 |    [Optional: concat forecast [B*N, 24, 3] -> Linear(3, 32)]
 |    -> TransformerEncoder (2 layers, 4 heads, GELU, d=32)
 |    -> mean-pool over time -> [B*N, 32]
 |    -> reshape [B, N, 32] -> Linear(32, 64)
 |    -> z_temporal [B, N, 64]
 |
 +-- Combine: concat(node_embed, z_temporal) [B, N, 128]
 |    -> Linear(128, 64) + LayerNorm + ReLU
 |    -> node_combined [B, N, 64]
 |
 +-- Adaptive Graph Constructor:
 |    node_combined [B, N, 64]
 |    -> Q = Linear(64, 64), K = Linear(64, 64)
 |    -> scores = softmax(Q @ K^T / sqrt(64))
 |    -> adj = (1-gate)*uniform + gate*learned + I
 |    -> row-normalize -> adj [N, N]
 |
 +-- Spatial GCN (3 layers of ResidualGCNBlock):
 |    node_combined [N, 64] + adj [N, N]
 |    -> LayerNorm -> GCN(A @ H @ W) -> ReLU -> Dropout + Skip
 |    -> h_spatial [B, N, 64]
 |
 +-- Gated Fusion:
 |    g = sigmoid(Linear(concat(h_spatial, z_temporal)))
 |    r = LayerNorm(g * W_s(h_spatial) + (1-g) * W_t(z_temporal))
 |    -> r [B, N, 64]
 |
 +-- Output:
      flatten [B, N*64=320]
      -> LayerNorm -> Linear(320, 256)
      -> features [B, 256]
```

### 5.2 Key Components

**SimpleGCNLayer:** Pure PyTorch GCN without torch_geometric dependency.
`H' = Linear(A_norm @ H)`. Expects pre-normalized adjacency (no additional
symmetric normalization to avoid feature shrinkage).

**ResidualGCNBlock:** GCN + LayerNorm + ReLU + Dropout + skip connection. Three
blocks stacked in `SpatialEncoder`.

**AdaptiveGraphConstructor:** Learns the adjacency matrix via bilinear attention
over node embeddings. A learnable gate interpolates between a uniform prior
(all-ones) and the learned attention weights. Output is row-normalized with
self-loops.

**PerNodeTemporalTransformer:** Processes each building's T-step history
independently through a shared 2-layer Transformer encoder with 4 attention
heads. Optionally appends 24-step forecast positions. Mean-pooling aggregates
the temporal sequence into a fixed-size embedding.

**GatedFusion:** Learnable per-node gate combining spatial (GCN) and temporal
streams. Gate statistics are monitored: healthy range is 0.3--0.7 (both streams
contributing).

### 5.3 Parameter Count

For the standard configuration (5 buildings, hidden=64, output=256, T=12):

| Component | Parameters |
|---|---|
| Global encoder | ~4.3K |
| Observation embedding | ~8.5K |
| Temporal transformer (2 layers) | ~18K |
| Temporal projection | ~2.1K |
| Combine projection | ~8.3K |
| Graph constructor | ~8.2K |
| Spatial encoder (3 GCN layers) | ~25K |
| Gated fusion | ~12.5K |
| Output projection | ~82K |
| **Total encoder** | **~170K** |
| Action head (256->64->act_dim) | ~17K |
| **Total actor mean_net** | **~187K** |

### 5.4 V2 Encoder (Spatial Only)

**File:** `citylearn_safe/stems_encoder_5bld.py`

The v2 encoder omits the temporal transformer. Instead, it uses
`SpatialSelfAttention` (multi-head self-attention over building nodes) in place
of temporal processing. This is fully stateless and lighter, but cannot reason
about temporal patterns in history data.

### 5.5 Statelesness Guarantee

Both encoder versions are fully stateless: the same function is computed during
rollout and during PPO update. This is critical for PPO's importance sampling
ratio to be correct. Earlier versions (v1) used an internal `HistoryBuffer` that
caused a behavioral split between rollout and update, breaking PPO convergence.

---

## 6. Hybrid Critic

The system uses a hybrid critic architecture: one STEMS-based reward critic and
four MLP-based cost critics.

### 6.1 Reward Critic (STEMS)

**Class:** `STEMSVCritic` in `train_multi_lag_stems.py`

The reward critic uses a separate instance of the STEMS encoder (independent
weights from the actor) followed by a value head:

```
obs [B, obs_dim]
 -> STEMSEncoder (separate weights) -> [B, 256]
 -> Linear(256, 64) -> ReLU -> Linear(64, 1)
 -> V(s)
```

This gives the reward critic the same spatial-temporal inductive bias as the
actor. Advantage estimates for temporally-contingent actions (e.g., "charge now
because price will rise in 6 hours") require the critic to model temporal
dependencies accurately.

### 6.2 Cost Critics (MLP)

**Built by:** `CriticBuilder` in `PPOLagMulti._init_model()`

Each of the four constraints gets an independent MLP cost critic
`V_ci(s)`. These use OmniSafe's default architecture (two hidden layers,
typically [256, 256]):

```
obs [B, obs_dim] -> Linear(obs_dim, 256) -> ReLU
                  -> Linear(256, 256)     -> ReLU
                  -> Linear(256, 1)       -> V_ci(s)
```

### 6.3 Why Hybrid?

The STEMS encoder adds ~170K parameters per instance. Using it for all five
critics (1 reward + 4 cost) would increase training time roughly 5x. Cost
critics predict relatively simple instantaneous signals (power threshold
violations, SoC band violations) that do not benefit from temporal-spatial
reasoning. The reward critic, however, must capture temporally extended value
(e.g., price arbitrage over 24-hour cycles), justifying the structural
investment.

Only the first `CriticBuilder.build_critic('v')` call receives the STEMS
encoder. Subsequent calls (for cost critics) fall through to the default MLP
builder. This is implemented via a monkeypatch with a mutable counter in the
closure.

---

## 7. Reward Design

The STEMS reward is a weighted sum of individually designed components. Each
component targets a specific aspect of smart grid operation. All rewards are
computed per step in `CityLearnCMDP._stems_reward()`.

### 7.1 Core Reward Components

| Symbol | Weight Env Var | Description |
|---|---|---|
| `r_eco` | `STEMS_MU_ECONOMIC` | Economic: `-price * (import - export_factor * export)` |
| `r_sg` | `STEMS_ALPHA_GRID` | Grid stability: `1 - (import/P_grid_max)^2`, quadratic penalty for high grid import |
| `r_sb` | `STEMS_ALPHA_BUILD` | Building stability: `1 - |NEC|/P_building_max`, per-building average |
| `r_ramp` | `STEMS_BETA_RAMP` | Ramp penalty: `-|delta_net| / P_grid_max`, penalizes rapid grid changes |
| `r_ren` | `STEMS_XI_RENEWABLE` | Renewable self-consumption: `solar / (solar + import)` |
| `r_barrier` | `STEMS_ALPHA_BARRIER` | Battery SoC barrier: smooth penalty near SoC boundaries [0.05, 0.10] and [0.85, 0.90] |

### 7.2 EV-Specific Reward Components

| Symbol | Weight Env Var | Description |
|---|---|---|
| `r_ev` | `STEMS_LAMBDA_EV` | Dense urgency-weighted EV charging shortfall penalty |
| `r_ev_guard` | `STEMS_ALPHA_EV_GUARD` | Penalizes V2G discharge when EV SoC < required SoC |
| `r_ev_smart` | `STEMS_ALPHA_EV_SMART` | **Headroom-gated EV price signal** (R27a innovation) |

### 7.3 Headroom-Gated EV Smart Reward (r_ev_smart)

This is the key innovation for resolving the conflict between EV departure
satisfaction (C1) and price arbitrage. It uses a **feasibility corridor**:

```
soc_min_feasible = max(0, required_soc - hours_to_departure * soc_per_hour)
headroom = current_soc - soc_min_feasible
gate = clip(headroom / required_soc, 0, 1)
```

- **Gate closed (headroom <= 0):** The EV is behind schedule. The price signal
  is suppressed, and the agent charges regardless of price to satisfy C1.
- **Gate open (headroom > 0):** The EV has slack. The price signal activates:
  charge during cheap hours, V2G discharge during expensive hours.

This prevents the failure mode observed in R26j where `r_trajectory` (which
lacked departure awareness) actively conflicted with C1, contributing 57% of
negative episode return.

### 7.4 Additional Reward Components

| Symbol | Weight Env Var | Description |
|---|---|---|
| `r_load_shift` | `STEMS_ALPHA_LOAD_SHIFT` | Solar-aware battery price arbitrage with SoC gating |
| `r_trajectory` | `STEMS_ALPHA_TRAJECTORY` | Forecast-aware arbitrage: compares spot price to dynamic future mean |
| `r_peak_shave` | `STEMS_ALPHA_PEAK_SHAVE` | Rewards battery discharge during building power peaks |
| `r_v2g_ctx` | `STEMS_ALPHA_V2G_CONTEXT` | Context-aware V2G: rewards discharge during grid import, charge during solar |
| `r_ev_solar` | `STEMS_ALPHA_EV_SOLAR` | Rewards EV charging during solar abundance |
| `r_solar_store` | `STEMS_ALPHA_SOLAR_STORE` | Rewards charging any storage during real solar surplus |
| `r_headroom` | `STEMS_ALPHA_HEADROOM` | Direction-aware building power headroom penalty |
| `r_grid_penalty` | `STEMS_ALPHA_GRID_PENALTY` | Quadratic grid import penalty (import-only, no export penalty) |
| `r_price_arb` | `STEMS_ALPHA_PRICE_ARB` | Simple battery price arbitrage: `-action * (price/mean - 1)` |
| `r_nec_sign` | `STEMS_ALPHA_NEC_SIGN` | Align storage actions with exogenous net load direction |
| `r_grid_mild` | `STEMS_ALPHA_GRID_MILD` | Gentle linear grid penalty (alternative to quadratic r_sg) |

All reward component values are logged individually per step in the `info` dict,
enabling post-hoc ablation analysis.

---

## 8. Constraint System

The system enforces four safety constraints (C1--C4) as a multi-constraint
CMDP. Each constraint has an independent cost signal, cost critic, Lagrange
multiplier, and cost limit.

### 8.1 Constraint Definitions

| ID | Key | Description | Signal Type |
|---|---|---|---|
| C1 | `cost_ev_departure` | EV departs with SoC below required | Sparse (at departure) |
| C2 | `cost_stems_battery` | Battery SoC outside [soc_min, soc_max] band | Dense |
| C3 | `cost_stems_building_power` | Building net power exceeds P_building_max | Dense |
| C4 | `cost_stems_grid_power` | Grid total power exceeds P_grid_max | Dense |

> **Code index mapping.** The codebase uses 0-indexed cost channels internally.
> Config key `cost_limit_0` corresponds to C1 (EV departure). Channel index 1
> is reserved for a dense EV budget variant (Saute MDP) that supplements C1
> with per-step feedback. Indices 2--4 map directly to C2--C4. YAML keys like
> `pid_kp_0` therefore control C1's PID gains.

### 8.2 Cost Signal Computation

**C1 (EV Departure Deficit):** Computed in `CityLearnSafetyEnv`. At each
timestep where an EV departs, the deficit is `max(0, required_soc -
actual_soc)`. Uses action-based tracking: the agent's actual charging actions
are recorded and used to compute what SoC the EV should have, avoiding reliance
on stale simulator state.

**C1 Dense Variant (Saute MDP):** The `SauteEVBudgetWrapper` supplements C1's
sparse departure signal with per-step feedback. Each EV has a safety budget
that decreases when charging is insufficient. When the budget is depleted, a
penalty fires every step until departure. This occupies a separate code channel
(index 1) but is not a distinct thesis constraint -- it is an implementation
detail that provides dense gradient signal for C1.

**C2 (Battery SoC Band):** Computed per building per step. Cost = hinge
violation outside the [soc_min, soc_max] band. Complemented by the r_barrier
reward shaping that provides smooth gradient near boundaries.

**C3 (Building Power):** Per-building per step. Cost = `max(0, |NEC_i| -
P_building_max)` for each building. P_building_max is auto-calibrated from the
schema data (95th percentile of non-shiftable load).

**C4 (Grid Power):** Per-district per step. Cost = `max(0, |sum(NEC_i)| -
P_grid_max)`. P_grid_max is similarly auto-calibrated.

### 8.3 Multi-Constraint Adapter

The `_MultiCostAdapter` wraps OmniSafe's standard single-cost environment
adapter to track all cost channels independently. Per-constraint episode costs are
accumulated and exposed via `get_per_constraint_ep_cost(i)` for the PID
Lagrangian update.

---

## 9. PID Lagrangian

**File:** `citylearn_safe/pid_lagrange.py`

Standard Lagrangian updates (`lambda += lr * (cost - limit)`) suffer from
integral windup: lambda grows unboundedly while cost remains high, then
overshoots when cost finally drops. The PID Lagrangian (Stooke et al., ICML
2020) replaces SGD with a PID controller.

### 9.1 Update Rule

```
delta = (ep_cost - cost_limit) / cost_limit    # normalized error

P-term:  delta_p = EMA(delta, alpha=0.95)       # smoothed current error
I-term:  pid_i  += delta * Ki                    # accumulated error
D-term:  pid_d   = max(0, EMA(cost) - EMA(cost, delay=10))

lambda = clamp(Kp * delta_p + pid_i + Kd * pid_d, 0, penalty_max)
```

### 9.2 Cost-Limit Normalization

The key addition over the original PID paper: dividing by `cost_limit` makes
`delta` dimensionless and O(1) for all constraints. Without this, a single set
of PID gains cannot work across constraints with cost magnitudes differing by
100x (e.g., C1 gap ~154 vs C3 gap ~22,000).

### 9.3 Anti-Windup

The I-term is clamped to `penalty_max` at every update. This prevents unbounded
accumulation when the output (lambda) is saturated at its upper bound. Without
anti-windup, the I-term grows during saturation, causing a delayed and
overshooting response when costs finally decrease.

```python
self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
self._pid_i = min(self._pid_i, self._penalty_max)    # anti-windup
```

### 9.4 Cost-Limit Curriculum

The cost limit for any constraint can be annealed over training epochs. For
example, C1 starts at limit=200 (permissive) and tightens to limit=20
(demanding) over epochs 0--20:

```yaml
multi_cfgs:
  anneal_cost_limit_0: [200, 20, 0, 20]  # [start_val, end_val, start_ep, end_ep]
```

When the limit changes, the I-term is rescaled proportionally (`I *= old_limit /
new_limit`) to prevent stale accumulation from earlier, more permissive limits.

### 9.5 Per-Constraint PID Gains

Each constraint can have independent gains:

| Constraint | Typical Kp | Typical Ki | Rationale |
|---|---|---|---|
| C1 (EV departure) | 3.0 | 0.05 | Aggressive: sparse signal, must react fast |
| C2 (battery SoC) | default | default | Standard: smooth, dense |
| C3 (building power) | 0.5 | 0.05 | Lower Kp: high-magnitude cost |
| C4 (grid power) | 0.3 | 0.03 | Lowest Kp: aggregate signal |

---

## 10. Training Pipeline

**File:** `scripts/training/train_multi_lag_stems.py`

### 10.1 Initialization Sequence

1. **Build ObsIndex**: Create a temporary environment to discover observation
   indices via `schema_index.build_index()`. This tells the STEMS encoder which
   observation dimensions correspond to which building/EV/global features.

2. **Build STEMS Encoder**: Instantiate `STEMSEncoder` (or V2) with the
   discovered node info, temporal window, and architecture hyperparameters.

3. **Wrap in STEMSMeanNet**: The encoder is wrapped with a
   `Linear(256, 64) -> ReLU -> Linear(64, act_dim) -> tanh` action head. The
   `forward()` method strips any extra observation dimensions (e.g., Saute
   budget) before passing to the encoder.

4. **Monkeypatch ActorBuilder**: Replace OmniSafe's `ActorBuilder.build_actor()`
   so that when it builds a `"gaussian_learning"` actor, it returns a
   `CustomGaussianLearningActor` with the STEMS mean network instead of a
   default MLP.

5. **Monkeypatch CriticBuilder** (if `--stems_critic`): The first call to
   `build_critic('v')` returns a `STEMSVCritic` with a separate STEMS encoder.
   Subsequent calls return default MLP critics (for cost critics).

6. **Load config**: Merge OmniSafe's PPOLag defaults with a custom YAML config.
   Inject `multi_cfgs` for per-constraint cost limits, PID gains, curriculum
   schedules, and cost weights.

7. **Instantiate PPOLagMulti**: OmniSafe creates the algorithm, which triggers
   the patched builders. The algorithm initializes 4 cost critics, 4 PID
   Lagrange controllers, and the softmax advantage combiner.

8. **Partial obs normalization** (optional): Patches OmniSafe's observation
   normalizer to only normalize base observation dimensions, leaving temporal
   history dimensions as identity. This preserves inter-timestep temporal
   patterns.

### 10.2 Training Loop (PPOLagMulti._update)

Each epoch:

1. **Curriculum update**: Anneal cost limits according to schedule. Update PID
   controllers with new limits.

2. **Lambda update**: For each constraint, compute episode cost from the
   adapter, then call `PIDLagrange.pid_update(ep_cost)`.

3. **Compute per-constraint GAE**: For each cost critic, compute advantages
   using generalized advantage estimation.

4. **Build DataLoader**: Pack reward advantages, per-constraint cost advantages,
   observations, actions, and log-probs into mini-batches.

5. **Update actor**: For each mini-batch:
   - Set `_current_batch_adv_cs` (per-constraint advantages for this batch).
   - Call `_compute_adv_surrogate(adv_r, adv_c)` which uses softmax weighting:
     ```
     weighted_advs = stack(A_ci) * lambdas     # [batch, 4]
     weights = softmax(weighted_advs / tau)     # per-timestep focus
     A_cost = sum(weights * weighted_advs)      # [batch]
     combined = adv_r - A_cost                  # CMDP Lagrangian
     ```
   - Compute PPO clipped surrogate loss with the combined advantage.

6. **Update critics**: Train reward critic and all 4 cost critics with
   mini-batch TD targets.

### 10.3 GradS (Optional)

When `use_grads: true`, the softmax advantage combination is replaced by
gradient surgery (Yao et al., L4DC 2024). Per-constraint gradients are computed
separately. A `GradSSelector` identifies non-conflicting constraint gradients
via cosine similarity filtering, samples one, and applies it scaled by
`|G|/N`.

---

## 11. Key Design Decisions

### 11.1 Why Monkeypatch Instead of Fork?

OmniSafe provides a well-tested PPO implementation with trust region, GAE,
observation normalization, logging, and checkpointing. Forking the entire
codebase would create a maintenance burden -- every OmniSafe bugfix or
improvement would require manual porting.

Instead, the system injects custom components at two narrow interfaces:

- `ActorBuilder.build_actor()` -- swap MLP for STEMS encoder
- `CriticBuilder.build_critic()` -- swap first critic for STEMS critic

Everything else (PPO rollouts, advantage estimation, gradient clipping, logging)
uses unmodified OmniSafe code. The `PPOLagMulti` class inherits from `PPO` and
overrides only `_init_model()`, `_init()`, and `_update()` to add
multi-constraint CMDP machinery.

### 11.2 Why Hybrid Critic?

The reward function has temporally extended structure (price arbitrage over 24h
cycles, EV departure deadlines hours away). The reward critic needs temporal
reasoning to accurately estimate V(s), so it benefits from the STEMS encoder's
temporal transformer.

Cost functions are instantaneous: "is building power above threshold right now?"
An MLP can predict these from the current observation without temporal context.
Using STEMS for 4 additional critics would increase training time ~5x for
negligible accuracy gain.

### 11.3 Why PID over Standard Lagrangian?

Standard SGD Lagrangian has a single parameter (learning rate) controlling
lambda dynamics. This creates a fundamental tradeoff:

- **High lr**: lambda responds quickly to violations but oscillates and
  overshoots.
- **Low lr**: lambda is stable but too slow to enforce tightening curriculum
  limits.

The PID controller separates these concerns:

- **P-term (Kp)**: immediate proportional response to current violation
- **I-term (Ki)**: slow accumulation for persistent violations
- **D-term (Kd)**: damping to prevent oscillation

With cost-limit normalization, a single set of default gains works across
constraints with 100x different cost magnitudes. Per-constraint overrides
allow fine-tuning when defaults are insufficient (e.g., C1 needs aggressive
Kp=3.0 because its signal is sparse).

Experimental evidence from R26j confirmed that standard Lagrangian with lambda
saturated at 20.0 could not solve C1 -- the I-term accumulated without bound
while lambda was capped, then could not respond when costs changed. PID
anti-windup directly addresses this failure mode.

### 11.4 Why Stateless Temporal Encoding?

PPO importance sampling requires that `pi(a|s)` produces the same distribution
during update as during rollout for the same `(s, a)`. If the encoder maintains
internal state (e.g., a history buffer), the same observation produces different
features depending on what was seen previously. During rollout, the buffer
contains the actual history. During update (replaying stored transitions), the
buffer contains a different sequence, breaking the importance sampling ratio.

The solution: embed the history in the observation itself via
`TemporalHistoryWrapper`. The encoder is a pure function of the observation
vector -- same input always produces the same output, regardless of when or how
often it is called.

### 11.5 Why Separate EV and Battery Reward Signals?

EV chargers and batteries serve fundamentally different purposes:

- **Batteries** have no deadlines. They can freely arbitrage prices.
- **EVs** have departure deadlines (C1). Charging cannot be deferred
  indefinitely.

A unified price-arbitrage reward would tell EVs to delay charging until prices
drop -- potentially missing the departure deadline. The headroom-gated
`r_ev_smart` solves this by suppressing the price signal when the EV is behind
its feasibility corridor, ensuring C1 compliance takes priority over economics.

---

## File Index

| File | Purpose |
|---|---|
| `citylearn_safe/cmdp_env.py` | CMDP wrapper, STEMS reward, cost aggregation |
| `citylearn_safe/safety_env.py` | Cost signals, KPI logging, action mapping |
| `citylearn_safe/stems_encoder.py` | STEMS encoder: GCN + Temporal Transformer |
| `citylearn_safe/stems_encoder_5bld.py` | 5-building base encoder: GCN + Spatial Self-Attention |
| `citylearn_safe/pid_lagrange.py` | PID Lagrangian controller |
| `citylearn_safe/grads/ppo_lag_multi.py` | PPOLagMulti: multi-constraint PPO |
| `citylearn_safe/grads/ppo_lag_grads.py` | PPOLagGradS: gradient surgery variant |
| `citylearn_safe/extractors.py` | EV deficit computation, SoC utilities |
| `citylearn_safe/forecast_obs_wrapper.py` | 24h forecast observation wrapper |
| `citylearn_safe/temporal_obs_wrapper.py` | T-step history observation wrapper |
| `citylearn_safe/saute_ev_wrapper.py` | Saute MDP for dense EV budget |
| `citylearn_safe/schema_index.py` | Schema-agnostic observation index builder |
| `scripts/training/train_multi_lag_stems.py` | Main training script |
