# OmniSafe Integration: From CityLearn to CMDP

This section explains the three-layer integration architecture that bridges CityLearn (a building energy simulator) with OmniSafe (a safe RL library). It covers how the environment interface was adapted, how OmniSafe's built-in algorithms work off-the-shelf with a scalar cost, and how the custom `PPOLagMulti` algorithm extends OmniSafe's internals to support five independent safety constraints.

---

## 1. The Interface Mismatch: What OmniSafe Expects vs. What CityLearn Provides

Before any RL training can happen, CityLearn and OmniSafe must agree on a common interface. They were designed independently, so their APIs differ significantly.

### 1.1 What OmniSafe Expects

OmniSafe defines an abstract base class called `CMDP` (Constrained Markov Decision Process) in `vendor_deps/omnisafe/envs/core.py`. Any environment used with OmniSafe must subclass `CMDP` and implement these abstract methods:

```python
class CMDP(ABC):
    # Class-level: which env IDs this class supports
    _support_envs: ClassVar[list[str]]

    # Whether OmniSafe should auto-wrap with TimeLimit / AutoReset
    need_time_limit_wrapper: bool
    need_auto_reset_wrapper: bool

    @abstractmethod
    def step(self, action: torch.Tensor) -> tuple[
        torch.Tensor,      # observation
        torch.Tensor,      # reward  (scalar)
        torch.Tensor,      # cost    (scalar)  ← THIS IS THE KEY ADDITION
        torch.Tensor,      # terminated
        torch.Tensor,      # truncated
        dict[str, Any],    # info
    ]: ...

    @abstractmethod
    def reset(self, seed=None, options=None) -> tuple[
        torch.Tensor,      # observation
        dict[str, Any],    # info
    ]: ...

    @abstractmethod
    def set_seed(self, seed: int) -> None: ...

    @abstractmethod
    def render(self) -> Any: ...

    @abstractmethod
    def close(self) -> None: ...
```

The critical difference from standard Gymnasium is the **third return value in `step()`**: a scalar `cost` tensor. In a standard `gym.Env`, `step()` returns `(obs, reward, terminated, truncated, info)` — five values. OmniSafe's `CMDP` returns **six values**, inserting `cost` between `reward` and `terminated`. This cost is the constraint violation signal that the Lagrangian algorithms use to enforce safety.

**All return values must be `torch.Tensor`** (not numpy arrays). OmniSafe's training loop operates entirely in PyTorch and expects tensors on the correct device.

**Registration:** OmniSafe maintains a global registry of known environments. A class decorated with `@env_register` is automatically added to this registry, allowing `omnisafe.Agent('PPOLag', 'CityLearnSafety-V2G-v2')` to find and instantiate it by string name.

### 1.2 What CityLearn Provides

CityLearn is a standard Gymnasium environment for simulating district-level energy management. Its API:

```python
# CityLearn returns standard gym 5-tuple, with list wrapping
obs, reward, terminated, truncated, info = env.step([action])
#  obs:        [np.ndarray]   ← list-wrapped (even for single agent)
#  reward:     [float]        ← list-wrapped
#  terminated: [bool]         ← list-wrapped
#  truncated:  [bool]         ← list-wrapped
#  info:       dict           ← no cost signal at all
```

There are four mismatches that must be resolved:

| Aspect | CityLearn | OmniSafe CMDP |
|--------|-----------|---------------|
| **Return count** | 5 values (no cost) | 6 values (includes cost) |
| **Wrapping** | Single-element lists: `[obs]`, `[reward]` | Unwrapped scalars/arrays |
| **Tensor type** | `numpy.ndarray`, Python `float`/`bool` | `torch.Tensor` (float32/bool) |
| **Cost signal** | Not provided | Required: scalar cost per step |

### 1.3 The Mismatch Diagram

```
CityLearn.step([action])
  Returns: ([np.array], [float], [bool], [bool], dict)
           ↑ lists      ↑ no cost   ↑ numpy types

OmniSafe expects:
  Returns: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict)
           ↑ obs         ↑ reward    ↑ COST         ↑ terminated  ↑ truncated
```

Three adapter layers resolve this mismatch, described in the next sections.

---

## 2. Layer 1 — Wrapping CityLearn into a CMDP

The adaptation happens through a three-layer wrapper stack. Each layer fixes one category of mismatch:

```
┌─────────────────────────────────────────────────────────────────┐
│  CityLearnCMDP(CMDP)              ← Layer 3: OmniSafe CMDP  │
│  @env_register                         torch tensors, 6-tuple  │
│  · step() returns (obs_t, rew_t, cost_t, term_t, trunc_t, info)│
│  · _stems_reward(): 19-component reward                        │
│  · _rebalanced_cost(): weighted sum of C0–C4                   │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  CityLearnSafetyEnv(gym.Env)  ← Layer 2: Cost signals  │  │
│  │  · Computes 5 constraint costs (C0–C4) per step           │  │
│  │  · Puts them in info dict as individual keys               │  │
│  │  · Clips actions per-dimension                             │  │
│  │  · Logs 100+ diagnostic fields                             │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │  SingleAgentListAdapter(gym.Env)  ← Layer 1: Unwrap │  │  │
│  │  │  · [obs] → obs,  [reward] → reward                  │  │  │
│  │  │  · [terminated] → terminated                         │  │  │
│  │  │  · action → [action]  (re-wraps for CityLearn)       │  │  │
│  │  │                                                      │  │  │
│  │  │  ┌────────────────────────────────────────────────┐  │  │  │
│  │  │  │  CityLearn Base Environment                    │  │  │  │
│  │  │  │  · 5 buildings, batteries, EV chargers         │  │  │  │
│  │  │  │  · 8,759 hourly steps per episode              │  │  │  │
│  │  │  └────────────────────────────────────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Layer 1: SingleAgentListAdapter (`adapters.py`)

**Problem solved:** CityLearn wraps everything in single-element lists, even in single-agent mode.

CityLearn was designed for multi-agent scenarios where each agent gets its own element. When `central_agent=True`, there is only one agent, but the returns are still list-wrapped: `[obs]` instead of `obs`. OmniSafe (and the safety envelope) expects bare arrays.

```python
class SingleAgentListAdapter(gym.Env):
    """Unwraps CityLearn's list-wrapped single-agent spaces."""

    def __init__(self, base_env):
        self.base = base_env
        # [Box(shape=(N,))] → Box(shape=(N,))
        self.observation_space = base_env.observation_space[0]
        self.action_space = base_env.action_space[0]

    def step(self, action):
        # Re-wrap action into list for CityLearn, unwrap everything that comes back
        obs, r, term, trunc, info = self.base.step([action])
        return (
            np.asarray(obs[0], dtype=np.float32),  # [obs] → obs
            float(r[0]),                            # [reward] → reward
            bool(term[0]),                          # [bool] → bool
            bool(trunc[0]),
            info,
        )

    def reset(self, **kwargs):
        obs, info = self.base.reset(**kwargs)
        return np.asarray(obs[0], dtype=np.float32), info
```

**After this layer:** Returns are `(np.ndarray, float, bool, bool, dict)` — standard Gymnasium format, no list wrapping. But still no cost signal.

### 2.2 Layer 2: CityLearnSafetyEnv (`safety_env.py`)

**Problem solved:** CityLearn has no concept of constraint costs. This layer computes them.

This is the largest wrapper (~1,900 lines). It wraps the adapted CityLearn environment and adds five independent safety cost signals. On each `step()`, after the base environment executes, it inspects the resulting state and computes how much each constraint was violated:

```python
class CityLearnSafetyEnv(gym.Env):
    def step(self, action):
        # 1. Clip actions per-dimension (respects per-device bounds)
        action_clipped = np.clip(action, self.action_space.low, self.action_space.high)

        # 2. Execute in base CityLearn environment
        obs, reward_base, terminated, truncated, info = self.base.step(action_clipped)

        # 3. Compute 5 constraint costs from post-step state
        info['cost_ev_departure']         = self._compute_c0(...)  # C0
        info['cost_ev_dense']             = self._compute_c1(...)  # C1
        info['cost_stems_battery']        = self._compute_c2(...)  # C2
        info['cost_stems_building_power'] = self._compute_c3(...)  # C3
        info['cost_stems_grid_power']     = self._compute_c4(...)  # C4

        return obs, reward_base, terminated, truncated, info
```

The five costs are computed as follows:

**C0 — EV Departure Deficit:** When an EV departs, how many kWh short of its target SoC was it? This is a sparse signal — it is zero on most timesteps and spikes when an EV leaves undercharged.
```python
deficit_kwh = max(0, required_energy - actual_energy_delivered)
cost_ev_departure += deficit_kwh  # accumulated over all chargers
```

**C1 — EV Dense Charging:** A per-step measure of whether EVs are being charged fast enough relative to their urgency. Unlike C0, this fires every step (dense signal).

**C2 — Battery SoC Bounds:** Measures how far battery state-of-charge drifts outside the safe band [0.05, 0.95]:
```python
violation = max(0, soc_min - soc) + max(0, soc - soc_max)
cost_stems_battery = sum(violation per building)
```

**C3 — Building Power:** Per-building check that net energy consumption stays within the building's power limit:
```python
violation = max(0, |NEC_building| - P_building_max)
cost_stems_building_power = sum(violation per building)
```

**C4 — Grid Power:** Aggregate grid import must stay below the district limit:
```python
total_import = sum(max(0, NEC_building) for each building)
cost_stems_grid_power = max(0, total_import - P_grid_max)
```

**After this layer:** Returns are still `(np.ndarray, float, bool, bool, dict)` — but `info` now contains five cost keys. The cost signal exists but hasn't been converted to OmniSafe's required format yet.

### 2.3 Layer 3: CityLearnCMDP (`cmdp_env.py`)

**Problem solved:** Convert everything to torch tensors, add the `cost` return value, replace the reward with the 19-component STEMS reward, and register with OmniSafe.

This is the outermost layer that OmniSafe directly interacts with. It subclasses OmniSafe's `CMDP` base class and implements the full 6-tuple contract:

```python
@env_register
class CityLearnCMDP(CMDP):
    _support_envs: ClassVar[list[str]] = ['CityLearnSafety-V2G-v2']
    need_time_limit_wrapper: bool = False    # CityLearn handles its own episode length
    need_auto_reset_wrapper: bool = True     # OmniSafe auto-resets on episode end

    def __init__(self, env_id: str, **kwargs):
        super().__init__(env_id, **kwargs)
        # Build the inner wrapper stack
        base = make_base_env(central_agent=True)         # raw CityLearn
        safety = CityLearnSafetyEnv(base)              # + cost computation
        forecast = ForecastObsWrapper(safety, horizon=24) # + 24h lookahead obs
        # ... optional wrappers: temporal, spatial, Sauté, action mask ...
        self._env = forecast  # (or final wrapper in chain)
```

**The `step()` method** does three transformations:

```python
def step(self, action):
    # 1. torch → numpy (OmniSafe passes torch tensors, CityLearn needs numpy)
    if isinstance(action, torch.Tensor):
        a = action.detach().cpu().numpy().ravel()
    else:
        a = np.asarray(action, dtype=np.float32).ravel()

    # 2. Optional safety clamps (battery, EV discharge guards)
    #    ... clamp logic ...

    # 3. Execute inner env (Layer 2 + Layer 1 + CityLearn)
    obs, _reward_base, terminated, truncated, info = self._env.step(a)

    # 4. REPLACE reward with 19-component STEMS reward
    reward = self._stems_reward(info, a)

    # 5. AGGREGATE 5 constraint costs into single scalar
    cost = self._rebalanced_cost(info)

    # 6. Convert EVERYTHING to torch tensors
    obs_t   = torch.as_tensor(obs, dtype=torch.float32)
    rew_t   = torch.as_tensor(reward, dtype=torch.float32)
    cost_t  = torch.as_tensor(cost, dtype=torch.float32)
    term_t  = torch.as_tensor(terminated, dtype=torch.bool)
    trunc_t = torch.as_tensor(truncated, dtype=torch.bool)

    # 7. Return OmniSafe's required 6-tuple
    return obs_t, rew_t, cost_t, term_t, trunc_t, info
```

**The scalar cost aggregation** (`_rebalanced_cost`) combines the five individual costs into one weighted sum for backward compatibility with single-constraint algorithms:

```python
def _rebalanced_cost(self, info):
    return float(
        10.0 * info.get('cost_ev_departure', 0.0) +       # C0: high weight (sparse)
         5.0 * info.get('cost_ev_dense', 0.0) +            # C1
         1.0 * info.get('cost_stems_battery', 0.0) +       # C2
         0.1 * info.get('cost_stems_building_power', 0.0) + # C3: low weight (frequent)
         5.0 * info.get('cost_stems_grid_power', 0.0)       # C4
    )
```

**The per-constraint costs are also preserved in `info`** — this is critical. The scalar `cost` satisfies OmniSafe's interface, but the individual cost keys (`cost_ev_departure`, etc.) remain in the info dict for multi-constraint algorithms to read directly.

### 2.4 The `@env_register` Mechanism

The decorator `@env_register` (from `omnisafe.envs.core`) inserts the class into OmniSafe's global `ENV_REGISTRY` dictionary. When OmniSafe later calls `make('CityLearnSafety-V2G-v2')`, the registry looks up which class declared that string in its `_support_envs` list and instantiates it. The registration happens at import time — so the training script must `import citylearn_safe.cmdp_env` (or `import citylearn_safe.register_env`) before calling `omnisafe.Agent(...)`.

---

## 3. Using OmniSafe's Built-In Algorithms Off-the-Shelf

Once `CityLearnCMDP` is registered, OmniSafe's built-in algorithms work without modification. The scalar cost from `_rebalanced_cost()` flows through the standard training pipeline.

### 3.1 The Standard Single-Constraint Flow

Here is what happens when you run a built-in algorithm like PPOLag, CPO, or FOCOPS with our registered environment:

```python
import omnisafe
import citylearn_safe.register_env  # triggers @env_register

agent = omnisafe.Agent('PPOLag', 'CityLearnSafety-V2G-v2')
agent.learn()
```

OmniSafe handles everything:

**Step 1 — Environment construction.** OmniSafe's `Agent.__init__()` calls `make('CityLearnSafety-V2G-v2')`, which finds `CityLearnCMDP` in the registry and instantiates it. OmniSafe then wraps it with its own wrapper stack:

```
CityLearnCMDP  (our CMDP)
  ↓ wrapped by
AutoReset        (auto-resets on episode end, stores final_observation)
  ↓ wrapped by
ObsNormalize     (running mean/variance normalization, optional)
  ↓ wrapped by
RewardNormalize  (running mean/variance, optional)
  ↓ wrapped by
CostNormalize    (running mean/variance, optional)
  ↓ wrapped by
ActionScale      (maps [-1,1] → env action bounds)
  ↓ wrapped by
Unsqueeze        (adds batch dimension for single env)
```

**Step 2 — Rollout.** The `OnPolicyAdapter` (OmniSafe's rollout manager) calls `env.step(action)` and receives the 6-tuple. It stores `obs`, `act`, `reward`, `cost`, `value_r` (reward critic), `value_c` (cost critic), and `logp` (log-probability) in a rollout buffer.

**Step 3 — GAE computation.** At the end of each epoch (8,759 steps = 1 simulated year), the buffer computes Generalized Advantage Estimation for both reward and cost:

```python
# Reward advantage: "how much better was this action than expected?"
adv_r = GAE(rewards, value_r_predictions, gamma, lambda)

# Cost advantage: "how much more costly was this action than expected?"
adv_c = GAE(costs, value_c_predictions, gamma, lambda_c)
```

Both use the standard GAE formula. There is one reward critic and one cost critic — both are standard MLPs mapping observations to scalar values.

**Step 4 — Policy update.** PPOLag combines the two advantages using a single Lagrange multiplier λ:

```python
# PPOLag's _compute_adv_surrogate() — from vendor_deps/omnisafe PPOLag:
penalty = self._lagrange.lagrangian_multiplier.item()
combined_advantage = (adv_r - penalty * adv_c) / (1 + penalty)
```

This is the standard Lagrangian relaxation: the policy tries to maximize reward while being penalized proportionally to cost violations. The `/(1 + penalty)` normalization prevents the cost term from dominating when λ grows large.

**Step 5 — Lambda update.** After each epoch, PPOLag updates λ via SGD on the constraint violation:

```python
# From OmniSafe's Lagrange class:
Jc = average_episode_cost  # scalar: total cost / steps
self._lagrange.update_lagrange_multiplier(Jc)
# Internally: λ += lr * (Jc - cost_limit)
# If Jc > cost_limit: λ increases (more penalty)
# If Jc < cost_limit: λ decreases (less penalty)
```

### 3.2 The Single-Constraint Limitation

This works — but with five independent constraints collapsed into one scalar cost via `_rebalanced_cost()`, the single λ cannot distinguish between them. If C3 (building power) is heavily violated but C0 (EV departure) is satisfied, the single λ increases for both. The fixed weights (C0×10, C3×0.1, etc.) are a crude attempt to balance them, but they cannot adapt during training.

This is why the multi-constraint extension was necessary.

---

## 4. PPOLagMulti: Extending OmniSafe for Five Independent Constraints

`PPOLagMulti` (`citylearn_safe/grads/ppo_lag_multi.py`) extends OmniSafe's PPO to handle five constraints independently, each with its own cost critic, Lagrange multiplier, and optional PID controller. It does this by overriding four initialization hooks and the update loop.

### 4.1 Inheritance: Why PPO, Not PPOLag

```
OmniSafe's hierarchy:           Our extension:

BaseAlgo
  └→ PolicyGradient
       └→ PPO ◄──────────────── PPOLagMulti inherits from HERE
            └→ PPOLag            (skipped — single-λ machinery not needed)
```

`PPOLagMulti` inherits from `PPO`, **not** `PPOLag`. This is deliberate: PPOLag hardcodes a single `_lagrange: Lagrange` object and a single `_compute_adv_surrogate()` that combines `adv_r` with one `adv_c`. If we inherited from PPOLag, we would have to fight its single-constraint machinery. By inheriting from PPO (which has no constraint handling), we build the multi-constraint layer cleanly on top.

### 4.2 What Gets Overridden

PPOLagMulti overrides five methods from OmniSafe's PPO:

| Method | PPO (OmniSafe) | PPOLagMulti (ours) |
|--------|----------------|-------------------|
| `_init_env()` | Creates `OnPolicyAdapter` | Creates `_MultiCostAdapter` (tracks per-constraint costs) |
| `_init_model()` | 1 actor + 1 reward critic + 1 cost critic | + **5 additional cost critics** (one per constraint) |
| `_init()` | Buffer setup | + **5 Lagrange multipliers** (PID or SGD) + curriculum schedules |
| `_init_log()` | Standard metrics | + **per-constraint metrics** (EpCost_i, Lambda_i, Loss_cost_critic_i) |
| `_update()` | Single-λ update, then PPO update | **Per-constraint λ updates** → per-constraint GAE → per-constraint critic updates → **softmax-weighted actor update** |
| `_compute_adv_surrogate()` | `(adv_r - λ*adv_c) / (1+λ)` | **Softmax over 5 weighted constraint advantages** |

### 4.3 The _MultiCostAdapter: Extracting Per-Constraint Costs

OmniSafe's standard `OnPolicyAdapter` stores only the scalar `cost` from `env.step()`. To train five independent cost critics, we need the five individual cost values. The `_MultiCostAdapter` extends `OnPolicyAdapter` to extract them from the info dict during rollout:

```python
class _MultiCostAdapter(OnPolicyAdapter):
    """Stores per-constraint costs and critic values at every step."""

    COST_KEYS = [
        'cost_ev_departure',          # C0
        'cost_ev_dense',              # C1
        'cost_stems_battery',         # C2
        'cost_stems_building_power',  # C3
        'cost_stems_grid_power',      # C4
    ]

    def rollout(self, steps_per_epoch, agent, buffer, logger):
        # Allocate storage for 5 cost streams + 5 critic value streams
        self.per_cost_steps = {
            f'cost_{i}': torch.zeros(steps_per_epoch)
            for i in range(5)
        }
        self.per_cost_steps.update({
            f'value_c_{i}': torch.zeros(steps_per_epoch)
            for i in range(5)
        })
        self.episode_boundaries = []
        self._per_ep_costs = [0.0] * 5

        obs, _ = self.reset()
        for step in range(steps_per_epoch):
            act, value_r, value_c, logp = agent.step(obs)

            # Evaluate all 5 cost critics on current observation
            for i, critic in enumerate(self._cost_critics_ref):
                with torch.no_grad():
                    vc_i = critic(obs)[0]
                    self.per_cost_steps[f'value_c_{i}'][step] = vc_i.cpu().squeeze()

            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            # Extract per-constraint costs from info dict
            for i, key in enumerate(self.COST_KEYS):
                raw_val = float(info.get(key, 0.0))
                self.per_cost_steps[f'cost_{i}'][step] = raw_val * cost_weights[i]
                self._per_ep_costs[i] += raw_val  # raw (for PID lambda updates)

            # Standard buffer storage (reward critic, scalar cost critic)
            buffer.store(obs=obs, act=act, reward=reward, cost=cost,
                         value_r=value_r, value_c=value_c, logp=logp)

            # On episode end: record boundary for per-constraint GAE
            if terminated or truncated or (step == steps_per_epoch - 1):
                # Bootstrap: evaluate all 5 cost critics on terminal obs
                last_values_c = [critic(obs)[0] for critic in cost_critics]
                self.episode_boundaries.append((path_start, step+1, last_values_c))
                self._last_completed_ep_costs = list(self._per_ep_costs)
                self._per_ep_costs = [0.0] * 5

            obs = next_obs
```

The key insight: the scalar `cost` still flows through OmniSafe's standard buffer for backward compatibility. The per-constraint costs are stored separately in `per_cost_steps` and used exclusively by PPOLagMulti's custom update loop.

### 4.4 Five Cost Critics: Independent Value Functions

In `_init_model()`, PPOLagMulti creates five additional V-critics using OmniSafe's own `CriticBuilder`:

```python
def _init_model(self):
    super()._init_model()  # Creates actor + reward critic + 1 cost critic (standard)

    # 5 per-constraint cost V-critics
    self._cost_critics = nn.ModuleList()
    self._cost_critic_optimizers = []
    for i in range(5):
        critic = CriticBuilder(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            hidden_sizes=[256, 256],       # same architecture as reward critic
            activation='tanh',
        ).build_critic('v').to(self._device)   # V-critic: obs → scalar value
        self._cost_critics.append(critic)
        self._cost_critic_optimizers.append(
            torch.optim.Adam(critic.parameters(), lr=critic_lr)
        )

    # Make critics accessible to the adapter during rollout
    self._env._cost_critics_ref = self._cost_critics
```

Each critic learns `V_ci(s)` — the expected cumulative cost for constraint `i` from state `s` onward. They are trained independently with their own optimizer and target values (no parameter sharing).

### 4.5 Five Lagrange Multipliers: PID or SGD

In `_init()`, PPOLagMulti creates five independent Lagrange multipliers:

```python
def _init(self):
    super()._init()  # PPO buffer setup

    self._per_lagranges = []
    for i in range(5):
        cost_limit_i = multi_cfgs.get(f'cost_limit_{i}', 5000.0)

        if use_pid:
            # PID controller (Stooke et al., ICML 2020)
            # Per-constraint gains — critical because constraints have different scales
            kp = multi_cfgs.get(f'pid_kp_{i}', default_kp)
            ki = multi_cfgs.get(f'pid_ki_{i}', default_ki)
            kd = multi_cfgs.get(f'pid_kd_{i}', default_kd)
            self._per_lagranges.append(PIDLagrange(
                cost_limit=cost_limit_i, pid_kp=kp, pid_ki=ki, pid_kd=kd,
                penalty_max=lagrangian_upper_bound,
            ))
        else:
            # Standard SGD (OmniSafe's built-in Lagrange class)
            self._per_lagranges.append(Lagrange(cost_limit=cost_limit_i, ...))
```

The PID controller (`pid_lagrange.py`) is a drop-in replacement for OmniSafe's `Lagrange` class. Instead of SGD (`λ += lr * (Jc - limit)`), it uses proportional-integral-derivative control:

```python
class PIDLagrange:
    def pid_update(self, ep_cost):
        # Normalize error by cost limit (makes gains scale-invariant)
        delta = (ep_cost - self.cost_limit) / self.cost_limit

        # P-term: proportional to current violation (EMA-smoothed)
        self._p_term = ema_alpha * self._p_term + (1 - ema_alpha) * delta

        # I-term: accumulated violation (clamped to prevent windup)
        self._i_term = clamp(self._i_term + delta, 0, penalty_max)

        # D-term: rate of change (only fires on cost INCREASES)
        d_raw = delta - self._prev_delta
        self._d_term = ema_d * self._d_term + (1 - ema_d) * max(0, d_raw)

        # Output
        pid_out = self.kp * self._p_term + self.ki * self._i_term + self.kd * self._d_term
        self.lagrangian = clamp(pid_out, 0, penalty_max)
```

**Per-constraint gains** are critical because the five constraints operate at very different scales:

| Constraint | Kp | Ki | Rationale |
|-----------|-----|-----|-----------|
| C0 (EV departure) | 5.0 | 0.0 | Pure proportional — no integral to prevent windup on sparse signal |
| C1 (EV dense) | 0.1 | 0.01 | Default — Sauté MDP handles most of C1 |
| C2 (Battery SoC) | 0.1 | 0.01 | Default — usually within limits |
| C3 (Building power) | 0.5 | 0.05 | 5× stronger — starts far over limit, needs urgency |
| C4 (Grid power) | 0.3 | 0.03 | 3× stronger for faster response |

### 4.6 The Update Loop: Per-Constraint GAE and Softmax Weighting

The `_update()` method is where everything comes together. It replaces OmniSafe's standard single-constraint update with a five-constraint version:

```python
def _update(self):
    # ── Step 1: Curriculum annealing (R18) ──
    # Gradually tighten cost limits during training
    self._update_cost_limits(self._epoch_counter)

    # ── Step 2: Update 5 Lagrange multipliers independently ──
    for i in range(5):
        ep_cost_i = self._env.get_per_constraint_ep_cost(i)  # raw episode cost
        if self._use_pid:
            self._per_lagranges[i].pid_update(ep_cost_i)
        else:
            self._per_lagranges[i].update_lagrange_multiplier(ep_cost_i)

    # ── Step 3: Compute per-constraint GAE ──
    # 5 independent advantage estimates, one per constraint
    per_cost_advs = []
    for i in range(5):
        costs_i = self._env.per_cost_steps[f'cost_{i}']
        values_i = self._env.per_cost_steps[f'value_c_{i}']

        adv_i = torch.zeros_like(costs_i)
        for start, end, last_values_c in self._env.episode_boundaries:
            # Standard GAE-lambda with per-constraint bootstrapping
            deltas = costs_i[start:end] + gamma * values_i[start+1:end+1] - values_i[start:end]
            adv_i[start:end] = discount_cumsum(deltas, gamma * lam_c)

        # Z-score normalize (same treatment as reward advantages)
        if standardized_cost_adv:
            adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)

        per_cost_advs.append(adv_i)

    # ── Step 4: Mini-batch training ──
    for iteration in range(update_iters):
        for batch in DataLoader(dataset, batch_size, shuffle=True):
            b_obs, b_act, b_logp, b_adv_r = batch[:4]
            b_adv_cs = [batch[7 + 2*i] for i in range(5)]      # per-constraint advs
            b_target_cs = [batch[7 + 2*i + 1] for i in range(5)] # per-constraint targets

            # Update reward critic (standard OmniSafe)
            self._update_reward_critic(b_obs, b_target_value_r)

            # Update 5 cost critics independently
            for i in range(5):
                self._update_per_cost_critic(b_obs, b_target_cs[i], i)

            # Update actor with softmax-weighted advantage
            self._current_batch_adv_cs = b_adv_cs  # pass to _compute_adv_surrogate
            self._update_actor(b_obs, b_act, b_logp, b_adv_r, b_adv_c_unused)

        if kl_divergence > target_kl:
            break  # early stopping (PPO trust region)
```

### 4.7 Softmax Advantage Selection: Resolving Constraint Conflicts

The core innovation is in `_compute_adv_surrogate()`. Standard PPOLag computes:

```
A_combined = (A_reward - λ × A_cost) / (1 + λ)
```

With five constraints, a naive sum `Σ λ_i × A_ci` causes **cancellation** — when one constraint wants the battery to charge (to satisfy C0) and another wants it to discharge (to satisfy C3), the gradients partially cancel and the policy learns slowly.

PPOLagMulti uses **per-timestep softmax weighting** to focus the cost gradient on whichever constraint is most violated at each timestep:

```python
def _compute_adv_surrogate(self, adv_r, adv_c_ignored):
    # adv_c_ignored: OmniSafe's single cost advantage — we don't use it

    # Get current lambda values from 5 PID controllers
    lambdas = [self._get_lambda(i) for i in range(5)]

    # Stack per-constraint advantages: [batch_size, 5]
    adv_stack = torch.stack(self._current_batch_adv_cs, dim=-1)

    # Weight by lambdas: [batch_size, 5]
    lambda_tensor = torch.tensor(lambdas, device=adv_r.device)
    weighted_advs = adv_stack * lambda_tensor.unsqueeze(0)

    # Softmax: at each timestep, focus on the most-violated constraint
    logits = torch.clamp(weighted_advs / self._tau, min=-50, max=50)
    softmax_weights = F.softmax(logits, dim=-1)     # [batch_size, 5]
    A_cost = (softmax_weights * weighted_advs).sum(dim=-1)  # [batch_size]

    # Standard Lagrangian: maximize reward, minimize weighted cost
    combined = adv_r - A_cost

    return combined
```

**Why softmax works:** At timestep `t`, if C3 has `λ_3 × A_c3 = 2.0` and all others are near zero, the softmax gives C3 nearly 100% of the weight. At timestep `t+1`, if C0 has the highest weighted advantage, C0 gets the focus. This prevents conflicting constraints from canceling each other — the policy receives a clear, focused gradient signal at every timestep.

The temperature parameter `τ` controls how sharp the selection is: `τ → 0` is hard argmax (winner-take-all), `τ → ∞` is uniform averaging (equivalent to naive sum).

### 4.8 Curriculum Annealing: Phased Constraint Activation

The R18 curriculum learning variant gradually tightens cost limits during training to resolve the fundamental conflict between V2G exploration and EV departure safety:

```python
def _update_cost_limits(self, epoch):
    for i, schedule in self._cost_limit_schedules.items():
        start_val, end_val, start_ep, end_ep = schedule
        if epoch < start_ep:
            new_limit = start_val
        elif epoch >= end_ep:
            new_limit = end_val
        else:
            progress = (epoch - start_ep) / (end_ep - start_ep)
            new_limit = start_val + progress * (end_val - start_val)

        # Update the PID controller's target
        self._per_lagranges[i].update_cost_limit(new_limit)
```

Example: C0 starts at 999,999 (effectively disabled) and anneals to 1,800 over epochs 20–35. This lets the agent first learn V2G behavior from C3/C4 signals, then gradually learn to respect EV departure deadlines.

The `update_cost_limit()` method on `PIDLagrange` proportionally rescales the I-term to prevent discontinuous jumps in λ when the limit changes.

---

## 5. Summary: Three Levels of Integration

| Level | What | How | Code |
|-------|------|-----|------|
| **Interface bridge** | Make CityLearn look like an OmniSafe CMDP | 3-layer wrapper: unwrap lists → add costs → convert to torch | `adapters.py` → `safety_env.py` → `cmdp_env.py` |
| **Off-the-shelf** | Run PPOLag/CPO/FOCOPS with scalar cost | `@env_register` + `_rebalanced_cost()` aggregates 5 costs into 1 | `cmdp_env.py` + standard `omnisafe.Agent()` |
| **Multi-constraint** | 5 independent λ, critics, PID controllers | Subclass PPO, override 5 hooks, inject `_MultiCostAdapter` | `ppo_lag_multi.py` + `pid_lagrange.py` |

The design keeps the layers cleanly separated: the CMDP adapter works with any OmniSafe algorithm (single-constraint or multi-constraint), the off-the-shelf path requires zero custom algorithm code, and the multi-constraint extension only overrides what it needs while reusing OmniSafe's rollout, buffer, and trust region infrastructure.
