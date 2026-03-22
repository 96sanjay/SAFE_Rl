# TD3 + SP-RL: Safe Policy RL with Differentiable Projection

**Date**: 2026-03-21
**Status**: Design
**Paper**: Markgraf et al. "Safe RL using Action Projection: Safeguard the Policy or the Environment?" arXiv:2509.12833, 2025

---

## Problem Statement

Implement SP-RL (Safe Policy RL) where the action projection Φ is embedded inside the policy
as a differentiable layer, enabling gradient flow from the critic through the projection back to the
actor. Compare against our existing SE-RL (SAC v4 with Beta actor + ActionMaskWrapper).

## Architecture

### Data Flow (Figure 1b of paper)

```
obs → Actor MLP → u (unsafe, ∈ [-1,1]⁹)
          ↓ differentiable (torch autograd)
      DiffProjector Φ(x, u) → u^φ (safe, ∈ U^φ_x)
          ↓ cvxpylayers (implicit differentiation via KKT, Eq. 36)
      CityLearn env → obs', r, cost_C0, cost_C1
          ↓
      Reward critics Q₁ᵣ(s, u^φ), Q₂ᵣ(s, u^φ)   [twin, on SAFE actions]
      Cost critics Q_C0(s, u^φ), Q_C1(s, u^φ)     [per-constraint Lagrangian]
      Penalty critic q_pen(s, u)                    [PenC, on UNSAFE actions, Eq. 29]
          ↓
      Actor loss (Eq. 30):
        L = -min(Q₁ᵣ, Q₂ᵣ)(s, u^φ)     ← reward maximization, gradient through Φ
            + λ₀·Q_C0(s, u^φ)            ← C0 Lagrangian, gradient through Φ
            + λ₁·Q_C1(s, u^φ)            ← C1 Lagrangian, gradient through Φ
            - q_pen(s, u)                 ← PenC penalty, NO gradient through Φ
```

### Constraint Handling Split

| Constraint | Method | Why |
|---|---|---|
| C0 (EV departure) | Lagrangian (λ₀ + PID) | Temporal: depends on future departure time |
| C1 (EV dense) | Lagrangian (λ₁ + PID) | Temporal: dense charging signal over episode |
| C2 (Battery SoC) | Hard projection (QP) | Instantaneous: SoC bounds computable from state |
| C3 (Building power) | Hard projection (QP) | Instantaneous: NEC computable from state |
| C4 (Grid power) | Hard projection (QP) | Instantaneous: grid import computable from state |

### QP Formulation (DiffProjector)

```
minimize   ½ ||ũ - u||²
subject to:
  ũ ∈ [-1, 1]⁹                                          (action bounds)
  SoC_next(b) = SoC(b) + ũ_batt(b) × scale(b) ∈ [soc_low, soc_high]   ∀b  (C2)
  |base_nec(b) + Σ device_power(b)| ≤ P_building_max     ∀b  (C3)
  Σ max(0, base_nec(b) + device_power(b)) ≤ P_grid_max        (C4)
```

Solved via cvxpylayers (SCS solver). Gradients computed via implicit function theorem (KKT conditions, Eq. 35-36).

### Penalty Critic (PenC, Section 7.2 of paper)

The penalty at each step:
```
h_t = w × ||u_t - u^φ_t||²    if u_t ∉ U^φ_x
h_t = 0                        if u_t ∈ U^φ_x
```

The penalty critic learns:
```
q_pen(x, u) = E[Σ γ^k h_{t+k+1} | x_t = x, u_t = u]
```

Trained with standard TD target: `y_pen = h + γ(1-d) × q_pen_target(s', π_target(s'))`

Actor gradient (Eq. 30): `∇_θ J = E[∇_θ π^⊥(x) ∇_{u^φ} q^SP(x, u^φ)] - E[∇_θ π(x) ∇_u q_pen(x, u)]`

The first term flows through Φ (SP-RL gradient). The second does NOT flow through Φ — it tells the actor to produce actions that need less projection correction.

### TD3LagMulti Class Design

```python
class TD3LagMulti(TD3):
    """TD3 with multi-constraint Lagrangian (C0, C1) + differentiable projection (C2/C3/C4).

    Extends OmniSafe's TD3 with:
    1. Per-constraint cost critics (Q_C0, Q_C1) + PID lambda updates
    2. DiffProjector: cvxpylayers QP for C2/C3/C4 hard constraints
    3. Penalty critic q_pen (Markgraf et al. 2025, Eq. 29-30)
    """

    # Key methods:

    def _init_model(self):
        # Standard TD3 actor + twin reward critics
        # + 2 cost critics (C0, C1)
        # + 1 penalty critic (q_pen)
        # + DiffProjector

    def _loss_pi(self, obs):
        # 1. Actor produces unsafe u = π(obs)
        # 2. Project: u_phi = DiffProjector(obs, u)  ← differentiable
        # 3. Penalty: h = w * ||u - u_phi||²
        # 4. Actor loss = -min(Q1r, Q2r)(obs, u_phi)     ← through Φ
        #              + λ0 * Q_C0(obs, u_phi)            ← through Φ
        #              + λ1 * Q_C1(obs, u_phi)            ← through Φ
        #              - q_pen(obs, u)                     ← NOT through Φ

    def _update_reward_critic(self, obs, act_phi, reward, next_obs, done):
        # Standard TD3 twin critic update on SAFE actions
        # Target: y = r + γ(1-d) * min(Q1_targ, Q2_targ)(s', Φ(s', π_targ(s')))

    def _update_cost_critics(self, obs, act_phi, costs, next_obs, done):
        # Per-constraint cost critic update (C0, C1)
        # Target: y_ci = c_i + γ(1-d) * Q_Ci_targ(s', Φ(s', π_targ(s')))

    def _update_penalty_critic(self, obs, act_unsafe, penalty, next_obs, done):
        # PenC update on UNSAFE actions
        # Target: y_pen = h + γ(1-d) * q_pen_targ(s', π_targ(s'))

    def _update_lambdas(self):
        # PID update per constraint (C0, C1)
        # λ_i += lr * (mean_cost_i - cost_limit_i)

    def step(self):
        # 1. obs → actor → u (unsafe)
        # 2. u → DiffProjector → u_phi (safe)
        # 3. env.step(u_phi) → obs', r, cost
        # 4. buffer.store(obs, u, u_phi, r, cost_C0, cost_C1, h, obs', done)
        # 5. Every update_cycle steps:
        #    - update reward critics
        #    - update cost critics
        #    - update penalty critic
        #    - every policy_delay steps: update actor + target networks
```

### Replay Buffer Extension

Standard off-policy buffer + extra fields:
```
(obs, act_unsafe, act_safe, reward, cost_C0, cost_C1, penalty_h, next_obs, done)
```

Both unsafe and safe actions stored. Reward/cost critics train on safe actions. Penalty critic trains on unsafe actions.

### Training Configuration

Identical to SAC v4 where possible:
```yaml
algo: TD3LagMulti
env_id: CityLearnSafety-V2G-v2
seed: 42
total_steps: 109500          # 50 epochs × 2190 (1-building 3-month)
steps_per_epoch: 2190
batch_size: 256
replay_buffer: 100000
gamma: 0.99
polyak: 0.005
actor_lr: 0.0003
critic_lr: 0.0003
policy_delay: 2
exploration_noise: 0.1       # TD3 Gaussian noise (not SAC entropy)
target_noise: 0.2            # TD3 target smoothing
noise_clip: 0.5
start_learning_steps: 2190   # 1 episode random exploration
warmup_epochs: 5             # no lambda updates for first 5 epochs

# Projection
projector_w_slack: 10.0      # not used (all hard constraints now)
projector_solver: SCS
projector_eps: 1e-4
projector_max_iters: 5000

# Penalty critic (PenC)
penalty_w: 1.0               # weight in h = w * ||u - u^φ||²
penalty_critic_lr: 0.0003

# Multi-lambda (C0, C1 only)
cost_limit_0: 80             # C0: EV departure
cost_limit_1: 50             # C1: EV dense
lambda_lr: 0.001
lambda_upper_bound: 3.0
pid_kp_0: 0.3
pid_ki_0: 0.0
pid_kp_1: 0.1
pid_ki_1: 0.01

# Same reward as SAC v4
STEMS_ALPHA_NEC_SIGN: 1.5
STEMS_ALPHA_PRICE_ARB: 1.0
STEMS_LAMBDA_EV: 15.0
STEMS_EV_SLACK_ARB_SCALE: 2.5
STEMS_ALPHA_GRID_PENALTY: 1.5
STEMS_BETA_RAMP: 0.5
STEMS_XI_RENEWABLE: 0.7
```

### Comparison Matrix (for thesis)

| | SE-RL (SAC v4) | SP-RL (TD3 + PenC) |
|---|---|---|
| Policy | Stochastic (Beta) | Deterministic (tanh MLP) |
| Projection | In environment (numpy) | In policy (cvxpylayers, differentiable) |
| C2/C3/C4 | ActionMaskWrapper (interval rescaling) | DiffProjector (QP, hard) |
| C0/C1 | Lagrangian (multi-λ + PID) | Lagrangian (multi-λ + PID) |
| Critic input | x ∈ (0,1) (normalized) | u^φ (safe physical action) |
| Gradient through projection | No | Yes (implicit function theorem) |
| Action aliasing mitigation | Beta distribution (no aliasing) | PenC (penalty for projection distance) |
| Reward terms | Identical | Identical |
| Schema | 1-building 3-month | 1-building 3-month |

### File Structure

```
citylearn_safe/
  grads/
    td3_lag_multi.py          # TD3LagMulti class (extends OmniSafe TD3)
  diff_projector.py           # DiffProjector (cleaned from trial, C2/C3/C4 hard)
  penalty_critic.py           # PenC network + penalty computation

scripts/
  train_td3_sp_rl.py          # Training script (like train_sac_lag.py)

configs/off-policy/
  td3_sp_rl_1bld.yaml         # 1-building config

run_td3_sp_rl.sh              # Run script with env vars (same as SAC v4)
```
