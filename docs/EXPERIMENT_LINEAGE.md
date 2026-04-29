# Experiment Lineage: Safe RL for V2G Control

This document traces the full experimental progression from R6 through R30.
Each run built on the failures and insights of its predecessors. The lineage
is not a straight line; it contains dead ends, abandoned approaches, and
course corrections that ultimately converged on the headroom-gated CMDP
design (R27a).

---

## Timeline Overview

| Run   | Phase                  | Description                                  | Status     |
|-------|------------------------|----------------------------------------------|------------|
| R6    | Foundation             | First 5-building comparison, basic PPO-Lag   | Completed  |
| R7    | Foundation             | Reward tuning experiments                    | Completed  |
| R8    | Foundation             | STEMS ablation: GCN+Transformer vs MLP      | Completed  |
| R9    | Foundation             | C3 (building peak power) focus               | Completed  |
| R10   | Foundation             | STEMSv3 encoder introduced                  | Completed  |
| R11a  | Multi-Constraint       | PPO-Lag, single constraint (SoC)             | Superseded |
| R11b  | Multi-Constraint       | First multi-constraint (4 constraints)       | Completed  |
| R12a  | Multi-Constraint       | Cost normalization stabilization             | Completed  |
| R15a-d| Multi-Constraint       | Hard EV clamp, dense C1 Saute, reward tuning | Completed  |
| R18   | Baseline Stabilization | Curriculum learning for constraint limits    | Completed  |
| R25b  | Baseline Stabilization | **Stable baseline** (reference config)       | Completed  |
| R26f  | Reward/Constraint      | Minimal reward (Lagrangian-only test)        | Completed  |
| R26g  | Reward/Constraint      | STEMS critic architecture validation         | Completed  |
| R26h  | Reward/Constraint      | Forecast-lean (24h price arbitrage)          | Failed     |
| R26i  | Reward/Constraint      | Forecast-full (trajectory reward)            | Failed     |
| R26j  | Reward/Constraint      | Lagrangian fix (PID gains, Ki integration)   | Failed     |
| R27a  | CMDP Innovation        | **Headroom-gated CMDP** (key innovation)     | Best       |
| R27b-f| CMDP Innovation        | R27a tuning variants                         | Completed  |
| R28   | Benchmarks             | OmniSafe stock algorithm comparison          | Completed  |
| R29   | Recent                 | Mask-simple variant                          | Completed  |
| R30   | Recent                 | KL-NEC constraint approach                   | Completed  |

**Lineage graph:**

```
R6 -> R7 -> R8 -> R9 -> R10
                          |
            R11a -----> R11b -> R12a -> R15a-d -> R18 -> R25b (baseline)
                                                          |
                  R26f  R26g  R26h  R26i  R26j  <-- all branch from R25b
                                                          |
                                                        R27a (CMDP) -> R27b-f
                                                          |
                                                        R28 (benchmarks)
                                                          |
                                                     R29, R30
```

---

## Phase 1: Foundation (R6--R10)

These runs established the environment, observation space, and encoder
architecture. The core question was whether graph-structured encoders
outperform flat MLPs on the CityLearn multi-building V2G problem.

### R6: First 5-Building Comparison

Basic PPO-Lagrangian on the 5-building CityLearn schema with EV chargers.
Used a single aggregate constraint signal. Demonstrated that PPO could learn
non-trivial battery and EV control policies but violated safety constraints
frequently.

### R7: Reward Tuning

Systematic sweep of reward coefficients for grid stability (`r_sg`), building
stability (`r_sb`), and economic terms (`r_eco`). Showed that naive reward
balancing leads to reward hacking: the agent exploits whichever term is
cheapest to maximize.

### R8: STEMS Ablation Studies

Compared three encoder architectures on the same task:
- Flat MLP (concatenated observations)
- GCN only (spatial graph, no temporal)
- GCN + temporal transformer (STEMS)

> **Key Insight:** GCN + temporal transformer outperformed the MLP baseline on
> all constraint satisfaction metrics. Spatial structure matters: buildings
> share a grid bus, and the GCN captures inter-building coupling that a flat
> MLP misses.

### R9: C3 (Peak Demand) Focus

Dedicated investigation of Constraint 3 (building-level peak power). Tuned
the C3 cost coefficient and power threshold. Learned that per-building
thresholds vary significantly; a single global threshold leaves some buildings
under-constrained and others over-constrained.

### R10: STEMSv3 Encoder

Introduced the final encoder architecture:
- **GCN**: 3-layer graph convolution over the building topology
- **Temporal transformer**: 2-layer self-attention over a 12-step window
- **Pooling**: Mean-pool over buildings
- **Parameters**: ~175K

This architecture remained unchanged through R30.

**File:** `stems_encoder.py`, `stems_encoder_5bld.py`

---

## Phase 2: Multi-Constraint (R11--R15)

Phase 1 used a single aggregate constraint. Phase 2 introduced per-constraint
Lagrangian multipliers, exposing the fundamental tension between EV departure
SoC (C1) and grid-level objectives.

### R11a: Single-Constraint Lagrangian

PPO-Lagrangian with one constraint (battery SoC band). Demonstrated that a
single lambda cannot arbitrate multiple safety requirements. When the SoC
constraint dominated, the agent ignored grid and EV departure constraints
entirely.

> **Why It Failed:** A single Lagrangian multiplier creates a scalar penalty
> that cannot distinguish between different constraint violations. When C1
> (EV departure) and C2 (battery SoC) compete, the multiplier oscillates
> between penalizing one at the expense of the other.

### R11b: First Multi-Constraint Setup

Four constraints with independent Lagrangian multipliers (the code uses five
internal cost channels, but the dense Saute signal is an implementation detail
of C1, not a separate constraint):
- C1: EV departure SoC deficit (with optional dense Saute variant for shaping)
- C2: Battery SoC band [0.0, 0.95]
- C3: Building peak power
- C4: Grid aggregate power

**Config:** `configs/on-policy/r11b_multi_improved.yaml`
**Script:** `shell/archive/run_r11b_multi.sh`

Key settings from the shell script:
```bash
export STEMS_LAMBDA_EV="0.0"       # r_ev disabled, dense C1 signal handles EV
export STEMS_ALPHA_BARRIER="0.5"   # SoC barrier reward
export COST_W_C2="0.0"             # C2 zeroed (clamp+barrier handle SoC)
export STEMS_BETA_RAMP="1.5"
export COST_W_C3="5.0"
```

Training used `train_multi_lag.py` (MLP-based, no STEMS encoder yet in the
multi-constraint pipeline).

### R12a: Cost Normalization

The raw constraint costs spanned several orders of magnitude (C1 in hundreds,
C4 in tens of thousands). Without normalization, the Lagrangian optimizer
treated low-magnitude constraints as irrelevant.

R12a introduced per-constraint cost scaling and standardized cost advantages.
This stabilized lambda convergence across all four constraints.

### R15a--d: Iterative Improvements

Four sub-runs addressed the V2G discharge exploit discovered in R14: the agent
learned to discharge EVs below their required departure SoC because doing so
was reward-positive (selling energy back to the grid).

**R15a** introduced two countermeasures:
1. Hard EV action clamp: prevents discharge when SoC < required + margin
2. Anti-discharge penalty (`r_ev_guard` = 5.0): gradient signal for policy

**Script:** `shell/archive/run_r15a.sh`

Key additions:
```bash
export CITYLEARN_EV_ACTION_CLAMP="1"
export CITYLEARN_EV_CLAMP_MARGIN="0.1"
export STEMS_ALPHA_EV_GUARD="5.0"
```

R15a also inherited PID Lagrangian and dense C1 Saute signal from R14:
```bash
export CITYLEARN_PID_LAGRANGE="1"
export CITYLEARN_EV_SAUTE="1"
export CITYLEARN_EV_SAUTE_BUDGET="25000"
export CITYLEARN_EV_SAUTE_SHAPED_ALPHA="10.0"
```

**R15b--d** iterated on reward coefficients (`r_sg`, `r_sb`, `r_ren`),
PID gains, and Saute budget parameters.

> **Key Insight:** Action clamps are a necessary safety net but not sufficient
> for learning. The agent needs a shaped reward signal (r_ev_guard) to learn
> *why* discharging is bad, not just that it is blocked. Clamps without
> gradient signals produce flat loss landscapes near the constraint boundary.

---

## Phase 3: Baseline Stabilization (R18--R25)

### R18: Curriculum Learning

Introduced constraint limit annealing: start with loose limits and tighten
over training. This prevents the Lagrangian multipliers from exploding in
early epochs when the policy is random and violates everything.

### R25b: Stable Baseline

The reference configuration for all subsequent experiments. Every R26+ run
was compared against R25b.

**Script:** `shell/archive/run_r25b_report_stable.sh`
**Config:** `configs/on-policy/baseline_stable.yaml`

Architecture and training:
- Multi-constraint PPO-Lagrangian with per-constraint PID lambdas
- Dense C1 Saute signal (EV corridor shaping)
- Batch size 256, 10 update iterations per epoch
- `target_kl` = 0.06, `max_grad_norm` = 0.5
- No action clamps or masks (pure RL)

Reward configuration:
```bash
export STEMS_ALPHA_GRID="0.0"         # r_sg OFF
export STEMS_ALPHA_GRID_MILD="0.3"    # mild grid penalty instead
export STEMS_ALPHA_BUILD="0.0"        # r_sb OFF
export STEMS_XI_RENEWABLE="0.2"       # r_ren ON
export STEMS_BETA_RAMP="0.3"          # r_ramp ON
export STEMS_LAMBDA_EV="5.0"          # r_ev ON (dense EV charging reward)
export STEMS_ALPHA_BARRIER="0.5"      # r_barrier ON
export STEMS_ALPHA_EV_GUARD="1.0"     # r_ev_guard ON
export STEMS_ALPHA_V2G_CONTEXT="3.0"  # V2G context ON
export STEMS_EV_SLACK_ARB_SCALE="2.0" # EV slack arbitrage ON
```

R25b achieved stable training curves suitable for reporting but still showed
significant C1 violation rates. This set the stage for Phase 4's systematic
investigation of why C1 was hard.

---

## Phase 4: Reward and Constraint Variants (R26)

Phase 4 was a diagnostic campaign. Each R26 variant isolated one hypothesis
about why C1 (EV departure SoC) remained unsolved.

### R26f: Minimal Reward

Stripped all reward terms except the bare minimum. Tested whether the
Lagrangian multiplier alone could drive constraint satisfaction without reward
shaping. It could not: the policy had no gradient signal pointing toward
feasible behavior, only a penalty for infeasible outcomes.

### R26g: STEMS Critic Architecture Validation

Validated the hybrid critic design: STEMS encoder for the reward critic,
separate MLP heads for each cost critic. Confirmed that sharing encoder
parameters between reward and cost critics causes interference; independent
cost MLPs produce more stable lambda updates.

### R26h: Forecast-Lean (24h Price Arbitrage)

Hypothesis: if the agent can see future electricity prices, it will learn to
charge EVs during cheap periods and avoid last-minute panic charging.

**Script:** `shell/archive/run_r26h.sh`

Five active reward terms:
```bash
export STEMS_ALPHA_GRID=2.0           # r_sg
export STEMS_ALPHA_BUILD=3.0          # r_sb
export STEMS_XI_RENEWABLE=0.3         # r_ren
export STEMS_ALPHA_BARRIER=1.0        # r_barrier
export STEMS_ALPHA_TRAJECTORY=2.0     # r_trajectory (forecast arbitrage)
```

Used STEMSv3 with `--stems_critic`, 3 GCN layers, 2 temporal transformer
layers, mean pooling.

**Results (epoch 95):**

| Metric                        | Value   |
|-------------------------------|---------|
| EpRet                         | 4244.8  |
| C1 violation (per departure)  | 70.1%   |
| C3 violation (per timestep)   | 36.6%   |
| C4 violation                  | 0.3%    |
| EV actions                    | Negative (discharging) |

> **Why It Failed:** The agent learned to discharge EVs for immediate grid
> arbitrage profit. The trajectory reward was departure-blind: it rewarded
> temporal price arbitrage without checking whether the EV had enough time
> to recharge before departure. With zero incentive to charge, EV actions
> went negative (V2G discharge), producing 70% C1 violation at departure.

### R26i: Forecast-Full (Full Trajectory Reward)

Extended R26h with the full trajectory reward and additional forecast
features. C1 violation dropped to 45.2% but C3 worsened to 47.8%.
The trajectory reward still conflicted with C1 fundamentally.

### R26j: Lagrangian Fix

Hypothesis: R26h/i failed because the PID Lagrangian was misconfigured.
R26j applied aggressive corrections:

**Script:** `shell/archive/run_r26j.sh`

Key changes from R26h:
```bash
# C1 curriculum: immediate pressure (was [999999,...])
# C1 Ki=0.05 (was 0.0 -- zero integral memory)
# penalty_max=20 (was 10)
# C3 limit=5000 (was 15000 -- now binding)
# r_trajectory=20.0 (was 2.0 -- 10x boost)
```

PID gains:
- C1: Kp=3.0, Ki=0.05
- C3: Kp=0.5, Ki=0.05
- C4: Kp=0.3, Ki=0.03

**Results (120 epochs, full run):**

| Metric              | Value                                        |
|---------------------|----------------------------------------------|
| Lambda for C1       | Saturated at 20.0 by epoch 27                |
| EpCost for C1       | Stuck at ~254 (never decreased)              |
| Lambda for C3       | Saturated at 20.0 by epoch 18                |
| EpCost for C3       | 98k -> 42k (still 8x over limit)             |
| r_trajectory        | -0.15 average (57% of negative EpRet)        |

> **Why It Failed:** The Lagrangian multiplier for C1 saturated at its upper
> bound (20.0) by epoch 27 and had nowhere left to go, yet the C1 episode
> cost remained at 254 -- far above the limit of 30. The multiplier was
> maxed out but the penalty was still insufficient to override the reward
> signal. Meanwhile, r_trajectory was *actively harmful*: its average
> contribution was -0.15, accounting for 57% of the negative component of
> EpRet.
>
> **Conclusion:** Lagrangian multipliers alone cannot solve C1 without
> aligned reward help. When the reward signal points away from constraint
> satisfaction, no finite lambda can overcome it. The reward must at minimum
> not *conflict* with the constraint, and ideally should *guide* the policy
> toward feasible regions.

---

## Phase 5: CMDP Innovation (R27)

### R27a: Headroom-Gated CMDP

The central innovation of the thesis. R26j proved that the Lagrangian needs
reward-side cooperation. R27a provides it through a departure-aware reward
signal that inherently aligns with C1.

**Script:** `shell/active/run_headroom_gated_cmdp.sh`
**Config:** `configs/active/headroom_gated_cmdp.yaml`

#### The r_ev_smart Design

The key idea: reward EV charging proportional to the agent's *slack* --
the gap between current SoC trajectory and the minimum feasible trajectory.

**Feasibility corridor:**
```
soc_min = max(0, req_soc - hours_left * charge_rate)
```

This is the minimum SoC the EV must have right now to reach `req_soc` by
departure if it charges at maximum rate for every remaining hour.

**Headroom gate:**
- When `soc > soc_min + margin`: gate is OPEN. Agent has slack. Reward
  encourages price-optimal charging/V2G.
- When `soc <= soc_min + margin`: gate CLOSES. Agent is urgent. Reward
  strongly penalizes any action other than maximum charging.

This eliminates the R26h/i failure mode: the agent cannot profitably
discharge an EV that is close to its feasibility boundary, because the
headroom gate shuts off arbitrage rewards and activates charging penalties.

#### Configuration

Reward terms (rebalanced for CMDP):
```bash
export STEMS_ALPHA_GRID=1.5           # r_sg
export STEMS_ALPHA_BUILD=1.5          # r_sb
export STEMS_BETA_RAMP=0.3            # r_ramp
export STEMS_XI_RENEWABLE=0.3         # r_ren
export STEMS_ALPHA_BARRIER=0.3        # r_barrier
export STEMS_ALPHA_EV_SMART=1.5       # r_ev_smart (NEW)
export STEMS_ALPHA_TRAJECTORY=0.0     # r_trajectory OFF (conflicts with C1)
```

Constraint configuration:
```yaml
# C1: EV departure SoC -- sparse, PID Lagrangian (code index 0)
cost_limit_0: 200
pid_kp_0: 2.0
pid_ki_0: 0.05

# Dense C1 variant: unused (r_ev_smart replaces Saute corridor, code index 1)
cost_limit_1: 999999

# C2: Battery SoC band [0.0, 0.95] (code index 2)
cost_limit_2: 1500

# C3: Building peak power (code index 3)
cost_limit_3: 5000
pid_kp_3: 0.5
pid_ki_3: 0.05

# C4: Grid aggregate power (code index 4)
cost_limit_4: 8000
pid_kp_4: 0.3
pid_ki_4: 0.03

lagrangian_upper_bound: 35.0  # was 20.0 in R26j
```

Model architecture:
```bash
--hidden_dim 64 --output_dim 256 --num_gcn_layers 3
--stems_critic --temporal_layers 2 --temporal_pool mean
--partial_obs_norm
```

Actor: 2x256 hidden, tanh, lr=0.0002
Critic: 2x256 hidden, tanh, lr=0.001
PPO: batch_size=512, update_iters=10, target_kl=0.08, entropy=0.005

> **Key Insight:** The reward and the constraint must point in the same
> direction. r_ev_smart achieves this by construction: when the agent has
> slack, the reward encourages economically optimal behavior (including V2G);
> when slack runs out, the reward demands immediate charging -- exactly what
> the C1 constraint requires. The Lagrangian multiplier then handles residual
> violations without fighting the reward signal.

### R27b--f: Tuning Variants

Explored sensitivity to:
- r_ev_smart weight (0.5 to 3.0)
- Headroom gate margin
- C1 limit values
- PID gain schedules

R27a remained the best configuration.

---

## Phase 6: OmniSafe Benchmarks (R28)

Benchmarked stock OmniSafe algorithms on the same CityLearn environment for
thesis comparison. This establishes how off-the-shelf safe RL methods perform
without domain-specific reward engineering.

Algorithms tested:
- **On-policy:** PPO, PPO-Saute, PPO-Lag, TRPO-Lag, CPO, PCPO, FOCOPS
- **Off-policy:** SAC-Lag, CSAC-LB

During benchmarking, discovered and patched upstream device-mismatch bugs in
OmniSafe (tensors on different CUDA devices during cost critic updates).

Results and analysis are documented in the thesis evaluation chapter.

---

## Phase 7: Recent Experiments (R29--R30)

### R29: Mask-Simple Variant

Action masking approach: invalid actions (those guaranteed to violate a
constraint) are masked to zero probability before sampling. Simpler than
learned constraints but requires known constraint structure at action time.

### R30: KL-NEC Constraint

KL-divergence-based constraint with Normalized Expected Cost (NEC). Uses the
KL divergence between the current policy and a safe reference policy as a
regularization term, combined with normalized cost expectations.

---

## Abandoned Approaches

### Single Constraint Lagrangian (R11a)

**What:** One Lagrangian multiplier for an aggregate safety cost.

**Why it failed:** A scalar multiplier cannot arbitrate between four
constraints with different scales, dynamics, and urgency profiles. When C1
needed attention, the single lambda increased, but this also penalized the
agent for C2/C3 violations it was not causing, destroying the learning signal.

**Lesson:** Multi-constraint problems require per-constraint multipliers.

---

### Predictive Safety Filter (PSF)

**What:** Lookahead-based action filtering. Before executing an action,
simulate its consequences over a horizon and reject it if any constraint
would be violated.

**Why it failed:** Three issues killed it:
1. Computational cost: simulating the CityLearn environment forward is
   expensive, and doing it for every candidate action at every step was
   prohibitive.
2. Discrete events: EV arrivals and departures are stochastic and discrete.
   The lookahead cannot predict when an EV will disconnect.
3. Scalability: with 5 buildings and 10 action dimensions, the filter
   rejected most actions in congested states, leaving the agent with no
   useful policy gradient.

**Lesson:** Safety filters work for low-dimensional continuous systems
with predictable dynamics. CityLearn is neither.

---

### Temperature Control Case Study

**What:** Thermal comfort constraint as a safe RL case study.

**Why it was abandoned:** Completed as a thesis case study but superseded by
the V2G focus. The temperature problem is simpler (continuous dynamics, single
constraint, predictable occupancy) and does not exercise the multi-constraint
machinery that is the thesis contribution.

**Lesson:** Good for exposition but insufficient as a primary evaluation.

---

### Execution Shield

**What:** Post-hoc action correction. After the policy selects an action,
project it onto the feasible set before applying it to the environment.

**Why it failed:** The projection was too aggressive. It corrected actions
to the nearest feasible point, but the policy never learned *from* the
correction. The reward signal reflected the corrected action, not the
intended action, creating a distribution mismatch. The policy gradient
pointed toward the projected actions (which were safe but suboptimal)
rather than learning to produce safe actions natively.

**Lesson:** Safety layers that are invisible to the policy gradient
destroy the learning signal. The agent must observe the consequences of
its own actions.

---

### SERL Action Projection

**What:** Differentiable quadratic-programming-based action projection.
Unlike the Execution Shield, gradients flow through the projection layer.

**Why it failed:**
1. Numerical instability: the QP solver produced NaN gradients when
   constraints were nearly active (the Jacobian was ill-conditioned).
2. Per-step overhead: solving a QP at every environment step added 3--5ms,
   which compounded over 8759 steps per epoch and 100+ epochs.
3. Constraint formulation: the EV departure constraint is inherently
   non-Markovian (depends on future departure time), and the QP can only
   enforce instantaneous constraints.

**Lesson:** Differentiable optimization layers are elegant in theory but
fragile in practice when constraints involve temporal dependencies.

---

### Forecast Arbitrage (R26h/i)

**What:** Give the agent a 24-hour price forecast and reward temporal
arbitrage (buy low, sell high).

**Why it failed:** The trajectory reward was departure-blind. It rewarded
the agent for price-optimal charging schedules without checking whether
those schedules left EVs charged by departure time. The agent learned to
discharge EVs during high-price periods (V2G for grid profit) and never
recharged them, producing 70% C1 violation.

**Lesson:** Temporal rewards must be departure-aware. A forecast reward
that ignores deadlines creates a direct conflict with deadline constraints.
R27a's r_ev_smart fixed this by gating arbitrage rewards on feasibility
headroom.

---

### Pure Lagrangian (R26j)

**What:** Fix PID gains, increase penalty bounds, and rely on the Lagrangian
multiplier alone (with 10x trajectory reward boost) to solve C1.

**Why it failed:** The Lagrangian multiplier for C1 saturated at its upper
bound (20.0) by epoch 27 but the C1 episode cost remained stuck at 254. The
r_trajectory reward was actively harmful (average contribution: -0.15,
accounting for 57% of negative EpRet). Even after boosting r_trajectory 10x,
the reward still pointed away from C1 satisfaction.

**Lesson:** No finite Lagrangian multiplier can solve a constraint when the
reward signal actively opposes it. The reward must at minimum be neutral to
the constraint, and ideally aligned. This insight directly motivated R27a's
headroom-gated design, where the reward *cooperates* with the constraint by
construction.

---

## Constraint Reference

| ID | Constraint              | Description                               | Cost Signal |
|----|-------------------------|-------------------------------------------|-------------|
| C1 | EV departure readiness  | SoC deficit at EV disconnection           | Sparse      |
| C2 | Battery SoC band        | SoC outside [0.0, 0.95]                   | Dense       |
| C3 | Building peak power     | Per-building power exceeds threshold      | Dense       |
| C4 | Grid aggregate power    | Total grid power exceeds threshold        | Dense       |

Note: The code internally uses five cost channels (indices 0--4). Code index 0
corresponds to C1. Code index 1 is a dense Saute variant of C1 used for reward
shaping and is not a separate constraint.

---

## File Index

| File                                            | Purpose                              |
|-------------------------------------------------|--------------------------------------|
| `shell/active/run_headroom_gated_cmdp.sh`        | Current best run script (R27a CMDP)  |
| `configs/active/headroom_gated_cmdp.yaml`        | R27a hyperparameters and PID config  |
| `shell/archive/run_r11b_multi.sh`               | First multi-constraint run           |
| `shell/archive/run_r15a.sh`                     | Hard EV clamp + anti-discharge guard |
| `shell/archive/run_r25b_report_stable.sh`       | Stable baseline configuration        |
| `shell/archive/run_r26h.sh`                     | Forecast-lean arbitrage (failed)     |
| `shell/archive/run_r26j.sh`                     | Lagrangian fix attempt (failed)      |
| `scripts/train_multi_lag_stems.py`              | Training script (STEMS encoder)      |
| `scripts/train_multi_lag.py`                    | Training script (MLP, legacy)        |
| `scripts/eval_r26hi.py`                         | Evaluation: per-departure C1, per-building C3 |
| `stems_encoder.py`                           | STEMSv3 encoder implementation       |
| `stems_encoder_5bld.py`                         | 5-building STEMS configuration       |
