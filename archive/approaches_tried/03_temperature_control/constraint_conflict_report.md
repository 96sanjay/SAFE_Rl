# Constraint Conflict in Multi-Constraint Safe RL for V2G Smart Grid Control

## Technical Report

**Date:** April 2026
**System:** PPOLagMulti / PPOLagGradS on CityLearnSafety-V2G-v2
**Scope:** Empirical analysis of physically-coupled constraint conflicts, their invisibility to parameter-space gradient methods, and resolution mechanisms

---

## 1. Problem Setting

### 1.1 Environment

The CityLearnSafety-V2G-v2 environment simulates a district of 5 buildings, each equipped with a stationary battery and (for 3 of the 5 buildings) an electric vehicle charger. The simulation runs at hourly resolution over a full year (8759 timesteps per episode). Buildings have non-shiftable electrical loads (HVAC, appliances) that the agent cannot control. The agent controls battery charge/discharge and EV charge/discharge to minimize grid costs while satisfying safety constraints.

### 1.2 Action Space

The actor is a Gaussian MLP with architecture 198 -> 256 -> 256 -> 9 and tanh activations. The 9 continuous action dimensions are:

| Index | Action | Description |
|-------|--------|-------------|
| 0-5 | Batt_0 through Batt_5 | Charge (+) or discharge (-) building batteries |
| 6 | EV_0 | Charge (+) or discharge (-) EV at building 0 |
| 7 | EV_1 | Charge (+) or discharge (-) EV at building 1 |
| 8 | EV_2 | Charge (+) or discharge (-) EV at building 2 |

Total actor parameters: approximately 130,000 (198x256 + 256 + 256x256 + 256 + 256x9 + 9). The last layer mapping hidden features to actions contains 256x9 + 9 = 2313 parameters.

### 1.3 Safety Constraints

The system operates as a Constrained Markov Decision Process (CMDP) with 5 cost functions:

| Constraint | Description | Limit | Nature |
|------------|-------------|-------|--------|
| C0 | EV departure SoC deficit: penalizes shortfall when an EV departs below its required SoC | 200 | Sparse (fires only at departure events, ~1070/year) |
| C1 | Dense EV corridor (Saute-based) | 999999 | Disabled in this analysis |
| C2 | Battery SoC band violation: cost whenever stationary battery SoC exits [0.0, 0.95] | 1500 | Dense (every timestep) |
| C3 | Building power capacity: per-building power consumption exceeding 4.6083 kW | 5000 | Dense (every timestep) |
| C4 | Grid power capacity: total grid power exceeding 10.2352 kW | 8000 | Dense (every timestep) |

### 1.4 Lagrangian Methods Under Comparison

**PPOLagMulti (Softmax multi-Lagrangian):** Maintains per-constraint Lagrange multipliers updated via PID control (Stooke et al. 2020). At each policy update, all constraint gradients are combined through a softmax weighting:

```
g_combined = sum_i softmax(lambda_i * A_Ci / tau) * grad_theta V_Ci
```

All constraints contribute to every gradient step.

**PPOLagGradS (Gradient Shaping, Yao et al. 2024):** Pre-scales constraint gradients by their lambda values, computes pairwise cosine similarity in the full parameter space (130K dimensions), builds a candidate set by filtering constraints whose gradients are mutually non-conflicting (cosine similarity within [-sigma, kappa]), uniformly samples ONE constraint from this candidate set, and applies:

```
g_final = g_reward + g_selected * |G| / N
```

Only one constraint gradient is applied per mini-batch.

---

## 2. The Physical Coupling

### 2.1 Energy Balance Equation

At each timestep, the power consumption of building b is:

```
P_building_b = P_non_shiftable_b + P_EV_charging_b + P_battery_b
```

Where:
- **P_non_shiftable_b**: HVAC and appliance load (uncontrollable by the agent)
- **P_EV_charging_b**: EV charger power draw (directly controlled by the agent via EV actions)
- **P_battery_b**: Battery charge/discharge power (directly controlled by the agent via battery actions)

The total grid power is the sum across buildings:

```
P_grid = sum_b P_building_b
```

### 2.2 The Constraint Conflict

This energy balance creates a direct physical coupling between constraints:

- **C0 wants MORE EV charging:** To meet departure SoC targets, the agent must deliver energy to EVs before they depart. Higher EV action values mean more charging.
- **C3 wants LESS building power:** EV charging is a component of building power. Every kW delivered to an EV adds 1 kW to P_building_b.
- **C4 wants LESS grid power:** Grid power is the sum of all building power. EV charging propagates through to C4 as well.

The conflict is not a modeling artifact. It is an unavoidable consequence of energy conservation: the energy that charges an EV must flow through the building's electrical connection, counted against the building's power capacity.

### 2.3 Temporal Structure

EVs in the CityLearn environment follow realistic arrival/departure patterns:
- **Arrival:** typically 5-7 PM (evening return home)
- **Departure:** typically 6-8 AM (morning commute)
- **Charging window:** overnight, approximately 12 hours

Building base load (HVAC, appliances) is non-trivial during nighttime hours. The conflict peaks during hours 0-6 when:

1. EVs need charging before morning departure (C0 pressure).
2. Building nighttime HVAC load provides a non-negligible baseline power draw (C3 baseline).
3. EV charging adds to the building load, pushing total consumption toward or beyond the 4.6083 kW limit (C3 violation).

This temporal overlap is not incidental. In real residential V2G scenarios, overnight charging is the dominant strategy because vehicles are parked longest at night. The power capacity conflict with building load during these same hours is a fundamental challenge for V2G deployment.

---

## 3. Empirical Evidence

All results are from deterministic evaluation runs on the CityLearnSafety-V2G-v2 environment with 5 buildings, using models from the r25b ablation study.

### 3.1 Softmax (PPOLagMulti) -- Epoch 5

At epoch 5, the model has not yet learned to resolve C0. There are 99 C0 violations (1.4% of the ~7000 departure-relevant timesteps). The conflict is actively present.

#### 3.1.1 Cost-Cost Correlations

Pearson and Spearman rank correlations computed between per-timestep cost signals across 8759 timesteps:

| Pair | Pearson | Spearman | Interpretation |
|------|---------|----------|----------------|
| C0 vs C3 | **-0.14** | **-0.06** | CONFLICT (negative) |
| C0 vs C4 | **-0.28** | **-0.20** | CONFLICT (negative) |
| C0 vs C2 | +0.21 | +0.20 | Aligned (both want energy storage) |
| C2 vs C3 | **-0.30** | **-0.28** | CONFLICT (battery charging also adds load) |
| C3 vs C4 | +0.73 | +0.70 | Aligned (both are power constraints) |

**Interpretation:** The negative correlations between C0 and C3/C4 confirm the physical coupling. When C0 cost is high (EV undercharged at departure), C3 cost tends to be low (building power was kept within limits), and vice versa. The agent cannot simultaneously minimize both without temporal coordination.

The C2-C3 conflict is a secondary instance of the same physical mechanism: battery charging also adds to building power consumption.

#### 3.1.2 Total Costs and Violation Rates

| Constraint | Total Cost | Violation Rate |
|------------|-----------|----------------|
| C0 | 99.3 | 1.4% of departures |
| C2 | 20,534.8 | 78.3% of timesteps |
| C3 | 18,207.3 | 59.2% of timesteps |
| C4 | 18,481.5 | 32.4% of timesteps |

#### 3.1.3 Action-Cost Sensitivity (Spearman Correlations)

Spearman rank correlations between each action dimension and each constraint cost, computed across all 8759 timesteps. Stars indicate statistical significance.

Key EV action correlations:

| Action | vs C0 | vs C2 | vs C3 | vs C4 |
|--------|-------|-------|-------|-------|
| EV_0 | +0.035 | -0.020 | **+0.335***| **+0.434*** |
| EV_1 | +0.014 | -0.010 | **+0.156***| **+0.304*** |
| EV_2 | +0.023 | -0.015 | **+0.181***| **+0.332*** |

The strong positive correlations between EV actions and C3/C4 costs confirm the mechanism: higher EV charging (positive action) directly increases building power (C3) and grid power (C4).

The weak C0 correlations reflect C0's sparse nature -- it fires only at departure events, which occur at fewer than 2% of timesteps. The action-cost coupling to C0 is real but temporally diffuse: the agent must charge across many hours to prevent a deficit at the single departure moment.

A combined EV-action-vs-C3 scatter plot yields r = 0.553 with p = 0.0, confirming an extremely strong positive relationship between total EV charging effort and building power constraint violations.

#### 3.1.4 Hourly Conflict Profile

Analysis of mean action values and violation rates by hour of day reveals the temporal structure:

- **EV charging peaks at hours 0-6** (before morning departure), with positive mean EV actions averaging 0.3-0.5.
- **C0 violations peak at hours 6-7** (morning departure window), when EVs that were insufficiently charged depart below target SoC.
- **C3 violations peak at hours 0-4** (overnight), precisely overlapping with the EV charging window.
- **EV actions go negative (V2G discharge) during hours 8-16**, when the agent has learned that EVs are either absent or that building load is high enough that discharging into the building reduces grid import.

This hourly profile directly demonstrates the conflict: the hours when C0 demands the most charging are the same hours when C3 is most likely to be violated.

### 3.2 Softmax (PPOLagMulti) -- Epoch 40

At epoch 40, the model has learned to resolve C0. Only 17 violations remain (0.1% of departures). The conflict has been resolved through temporal coordination.

#### 3.2.1 Cost-Cost Correlations

| Pair | Pearson | Spearman | Interpretation |
|------|---------|----------|----------------|
| C0 vs C3 | **+0.15** | **+0.15** | Aligned (conflict resolved) |
| C0 vs C4 | +0.09 | +0.09 | Aligned (conflict resolved) |
| C3 vs C4 | +0.79 | +0.79 | Aligned (unchanged) |

The C0-C3 correlation has flipped from -0.14 to +0.15. The C0-C4 correlation has flipped from -0.28 to +0.09. The conflict has disappeared because C0 is no longer binding -- the agent has learned to satisfy EV charging requirements without creating C3/C4 violations.

#### 3.2.2 Violation Resolution

| Constraint | Epoch 5 Cost | Epoch 40 Cost | Change |
|------------|-------------|---------------|--------|
| C0 | 99.3 | 17.0 | -83% |
| C3 | 18,207 | ~19,000 | +4% (slight worsening) |
| C4 | 18,481 | ~15,000 | -19% (improvement) |

C0 was solved without C3 significantly worsening, confirming that the resolution was temporal (charging at low-load hours) rather than through a pure capacity trade-off.

### 3.3 GradS (PPOLagGradS) -- 18 Epochs

GradS was run for 18 epochs on the same environment and reward configuration. Its behavior diverges sharply from Softmax.

#### 3.3.1 Parameter-Space Cosine Similarity

GradS computes pairwise cosine similarity between constraint gradients in the full 130K-dimensional parameter space:

| Pair | CosSim (GradS, 130K dims) | Empirical Action-Space Correlation |
|------|---------------------------|------------------------------------|
| C0 vs C3 | **+0.44** (ALIGNED) | **-0.14** (CONFLICTING) |
| C0 vs C4 | **+0.40** (ALIGNED) | **-0.28** (CONFLICTING) |
| C3 vs C4 | **+0.73** (ALIGNED) | **+0.73** (ALIGNED) |

GradS reports the **wrong sign** for C0-C3 and C0-C4 pairs. It sees these physically-coupled opposing constraints as aligned.

#### 3.3.2 Candidate Set Behavior

- **CandidateSetSize:** approximately 4.0 out of 5 constraints (most constraints enter the candidate set).
- **InCandidate_0 (C0):** 60-94% of the time, C0 is included in the candidate set. GradS does not filter it out.

Because GradS sees C0 and C3 as aligned (positive cosine similarity), it does not detect a conflict and does not exclude either from the candidate set. Both are eligible for selection, but only one is applied per mini-batch.

#### 3.3.3 Lambda and Cost Trajectories

**Lambda_0 (C0 multiplier) -- GradS:**

```
Epoch  0:  9.72
Epoch  3:  4.88
Epoch  6:  8.50
Epoch  9:  3.02
Epoch 12: 11.08
Epoch 15:  7.20
Epoch 18:  9.50
```

Lambda_0 oscillates without converging. The pattern: when C0 is selected for a few mini-batches, the actor learns to charge EVs more, C0 cost drops, lambda decreases. Then C3 is selected, the actor learns to reduce building power (including EV charging), C0 cost rises, lambda increases. This oscillation is the direct behavioral signature of undetected constraint conflict.

**EpCost_0 (C0 cost) -- GradS:**

```
Epoch  0: 1148
Epoch  3:  636
Epoch  6:  938
Epoch  9:  323
Epoch 12: 1082
Epoch 15:  550
Epoch 18:  920
```

The see-sawing pattern mirrors the lambda oscillation.

**EpRet (reward) -- GradS:**

```
Epoch  0: -16,902
Epoch  6: -14,538
Epoch 12: -17,711
Epoch 18: -16,200
```

Reward diverges after a brief initial improvement. The CostRewardGradRatio climbs from 6 to 130, meaning constraint gradients become 130 times larger than the reward gradient. The actor is entirely dominated by conflicting constraint signals.

#### 3.3.4 Softmax Comparison

**Lambda_0 (C0 multiplier) -- Softmax:**

```
Epoch  0:  9.90
Epoch 10:  3.50
Epoch 20:  0.80
Epoch 30:  0.10
Epoch 40:  0.00
```

Lambda_0 converges monotonically to zero. C0 is solved.

**EpCost_0 (C0 cost) -- Softmax:**

```
Epoch  0: 1166
Epoch 10:  420
Epoch 20:  110
Epoch 30:   35
Epoch 40:   20
```

Monotonic decrease. No oscillation.

**EpRet (reward) -- Softmax:**

```
Epoch  0: -17,146
Epoch 10: -12,500
Epoch 20:  -8,200
Epoch 30:  -5,100
Epoch 40:  -3,651
```

Steady improvement throughout training.

---

## 4. Why GradS Fails: The Parameter-Space Masking Effect

### 4.1 The Mechanism

The actor network maps observations (198 dimensions) to actions (9 dimensions) through two hidden layers (256 units each). The gradient of any constraint cost with respect to the actor parameters decomposes across layers:

- **Input layer (198 x 256 = 50,688 params):** Gradients encode "which observations matter for this constraint." Both C0 and C3 care about EV SoC, time of day, and building load. Their gradients in this layer are **genuinely aligned** -- both constraints need the network to attend to the same features.

- **Hidden layer (256 x 256 = 65,536 params):** Gradients encode "how to combine features into useful representations." Again, both constraints benefit from similar internal representations (EV state, building state, temporal patterns). **Aligned.**

- **Output layer (256 x 9 = 2,304 params):** Gradients encode "which direction to push each action dimension." Here, C0 wants to increase EV action outputs (charge more) while C3 wants to decrease them (reduce building power). **Conflicting.**

The parameter counts:
- Aligned parameters (input + hidden layers): ~116,224
- Conflicting parameters (output layer, specifically EV-related columns): ~768 (256 x 3 EV outputs)
- **Dilution ratio: 116,224 : 768 = approximately 151:1**

Even taking the entire output layer (2,304 parameters) against the rest (approximately 128,000), the ratio is 56:1. The cosine similarity in 130K dimensions is dominated by the aligned early-layer gradients, drowning the conflicting output-layer signal.

### 4.2 Formal Characterization

Let theta = [theta_early; theta_last] be the parameter vector partitioned into early layers and last layer. The cosine similarity GradS computes is:

```
CosSim(C0, C3) = (g_C0 . g_C3) / (|g_C0| |g_C3|)
```

This decomposes as:

```
CosSim = (g_C0_early . g_C3_early + g_C0_last . g_C3_last) / (|g_C0| |g_C3|)
```

If g_C0_early . g_C3_early >> |g_C0_last . g_C3_last| (which follows from the dimensionality imbalance), then the sign of CosSim is determined by the early layers regardless of what the last layer shows.

In our measurements:
- Full parameter space: CosSim(C0, C3) = +0.44
- This masks the action-space conflict entirely.

### 4.3 The Sampling Bottleneck

Even if GradS could detect the conflict, its single-constraint sampling strategy would still create problems for physically-coupled constraints. The uniform sampling means:

1. With probability 1/|G|, C0 is selected and the actor gradient says "charge EVs more."
2. With probability 1/|G|, C3 is selected and the actor gradient says "reduce building power."
3. These opposing signals arrive at different mini-batches, creating a tug-of-war.

For the actor to learn a **temporal compromise** (charge EVs when building load is low), it needs to receive both signals simultaneously so that the gradient reflects the nuanced trade-off. Sequential application of opposing signals does not converge to a compromise; it oscillates.

### 4.4 The CostRewardGradRatio Explosion

The oscillation creates a secondary failure mode. As lambda_0 and lambda_3 both grow (each trying to enforce its constraint without success), the constraint gradient magnitudes increase. By epoch 18, the cost gradient is 130x larger than the reward gradient. The reward signal -- which encodes useful economic objectives and could help the actor find temporal compromises -- is completely drowned out.

---

## 5. Why Softmax Works: Simultaneous Gradient Application

### 5.1 The Mechanism

PPOLagMulti computes a softmax-weighted combination of all constraint advantages and applies a combined gradient at every mini-batch:

```
g_combined = sum_i softmax(lambda_i * A_Ci / tau) * grad_theta V_Ci
```

The critical difference: the actor receives both "charge more" (C0) and "reduce power" (C3) pressure at every parameter update. The network must find parameters that reduce the combined loss, which naturally leads to temporal coordination.

### 5.2 The Temporal Compromise

The learning trajectory from epoch 5 to epoch 40 reveals how Softmax resolves the conflict:

1. **Epochs 1-5:** The agent charges EVs aggressively during all available hours. C0 cost is high (99 violations). C3 cost is high (18,207). The agent has not yet differentiated between high-load and low-load hours for charging.

2. **Epochs 5-15:** Lambda_3 grows as C3 remains violated. The combined gradient now includes strong C3 pressure. The agent reduces EV charging during high building-load hours. C0 cost initially worsens slightly because some charging opportunities are foregone.

3. **Epochs 15-25:** The agent discovers the temporal compromise. It shifts EV charging to hours when building base load is low (midday during solar surplus, late night when HVAC demand drops). C0 cost drops sharply because the total energy delivered to EVs is maintained but distributed to low-conflict hours.

4. **Epochs 25-40:** C0 is essentially solved (cost 20, down from 1166). Lambda_0 drops to near zero because the constraint is no longer binding. C3 has not significantly worsened. The reward improves steadily as the agent refines its temporal strategy.

### 5.3 Why Simultaneous Application Enables Temporal Compromise

When both C0 and C3 gradients are applied simultaneously, the gradient at the output layer for an EV action at a particular timestep reflects the **net** constraint pressure:

- At a timestep with low building load: C3 gradient for the EV action is small (no violation risk). C0 gradient pushes toward charging. **Net: charge.**
- At a timestep with high building load: C3 gradient for the EV action is large and negative. C0 gradient pushes toward charging but is weaker (one of many charging opportunities). **Net: reduce or defer charging.**

The network learns a representation that distinguishes these contexts because both signals are present for every weight update. With GradS, each signal arrives alone, so the network cannot learn the conditional policy.

---

## 6. Comparison of Gradient Methods in Parameter Space vs Action Space

| Property | Parameter Space (GradS) | Action Space (Empirical) |
|----------|------------------------|-------------------------|
| **C0 vs C3 similarity** | +0.44 (ALIGNED) | -0.14 Pearson (CONFLICT) |
| **C0 vs C4 similarity** | +0.40 (ALIGNED) | -0.28 Pearson (CONFLICT) |
| **C3 vs C4 similarity** | +0.73 (ALIGNED) | +0.73 Pearson (ALIGNED) |
| **Dimensionality** | 130,000 | 9 |
| **Detects EV-building power conflict** | No | Yes |
| **Dominant signal** | Shared representations | Physical couplings |

Parameter-space and action-space analyses agree only for constraints that are truly aligned across both representations and actions (C3 vs C4: both are power constraints, aligned in all senses). For physically-coupled opposing constraints, the high-dimensional parameter space dilutes the conflict signal below detectability.

---

## 7. Secondary Findings

### 7.1 Battery-Power Conflict (C2 vs C3)

The C2-C3 Pearson correlation is -0.30 at epoch 5. This reflects the same physical mechanism: battery charging adds to building power consumption. However, this conflict is less severe than C0-C3 because:

- Battery charging rates are lower than EV charging rates in absolute terms.
- Battery discharge (V2G to building) can reduce grid import during high-load hours, partially compensating.
- The agent has more temporal flexibility with batteries (they have no departure deadlines).

### 7.2 Sparse vs Dense Constraint Interaction

C0 is sparse (fires only at departure events), while C3 is dense (fires every timestep). This asymmetry creates an additional challenge for Lagrangian methods:

- The C0 gradient is nonzero at fewer than 2% of timesteps.
- The C3 gradient is nonzero at ~59% of timesteps.
- In a Lagrangian that applies constraint gradients proportional to their current cost, C3 naturally dominates because it produces gradients more frequently.

PID Lagrangian (Stooke et al. 2020) with aggressive C0 gains (Kp=2.0, Ki=0.05) partially compensates by amplifying C0's lambda more aggressively. In the Softmax formulation, this ensures C0 receives sufficient weight despite its sparsity.

### 7.3 GradS Cost-Reward Gradient Ratio

The CostRewardGradRatio metric tracked by GradS reveals the severity of the oscillation-induced gradient explosion:

| Epoch | CostRewardGradRatio |
|-------|---------------------|
| 0 | 6 |
| 5 | 25 |
| 10 | 68 |
| 15 | 105 |
| 18 | 130 |

By epoch 18, the cost gradients are 130x larger than the reward gradient. The reward signal is effectively suppressed. This explains the reward divergence: the actor is entirely driven by oscillating constraint signals and cannot learn the economic objectives that would help it find efficient temporal compromises.

---

## 8. Resolution Strategies

### 8.1 Temporal Compromise via Softmax (Demonstrated)

As shown in Section 5, the Softmax multi-Lagrangian resolves the conflict by applying all constraint gradients simultaneously, enabling the actor to learn context-dependent policies. This requires no architectural modification -- only the choice of gradient combination method.

### 8.2 Reward-Driven C0 Resolution: r_ev_smart / Headroom Gating (R27a)

An alternative approach removes C0 from the Lagrangian entirely and provides a shaped reward signal that encodes departure-aware charging incentives:

**Feasibility corridor:**
```
soc_min = max(0, required_soc - hours_until_departure * max_charge_rate)
```

**Headroom gate:** The reward signal r_ev_smart activates only when the agent has slack (current SoC > soc_min + margin). When SoC is close to or below soc_min, the gate closes and the agent receives strong penalty regardless of building load.

This approach sidesteps the Lagrangian conflict entirely by making C0 satisfaction part of the reward, not a constraint. The Lagrangian then manages only C2-C4 (all power/energy constraints without the EV departure coupling).

Configuration from R27a (`configs/on-policy/r27a_cmdp.yaml`):
- r_ev_smart = 1.5 (env var: STEMS_ALPHA_EV_SMART)
- r_ev = 0, r_ev_guard = 0 (disabled -- Lagrangian no longer handles C0 directly)
- C0 limit retained as a safety backstop at 200

### 8.3 Action-Space Projection (Not Implemented, Proposed)

Methods that operate in action space rather than parameter space could detect the physical coupling directly:

- **Safety Layer (Dalal et al. 2018):** Projects the actor's proposed action onto the nearest feasible point satisfying linearized constraints. Would detect that increasing EV action violates C3.
- **ATACOM (Liu et al. 2024):** Projects onto the constraint manifold using geometric methods. Would naturally couple C0 and C3 through the energy balance.

These approaches bypass the parameter-space masking problem because they reason about constraint satisfaction in the 9-dimensional action space where the conflict is visible.

---

## 9. Figures Reference

All figures generated from the r25b_softmax_ablation evaluation.

### Epoch 5 (Conflict Present)

Located in `docs/temperature_case_study/constraint_conflict_epoch5/`:

| Figure | Filename | Description |
|--------|----------|-------------|
| Fig. 1 | `fig1_cost_correlation.pdf` | 5x5 cost correlation matrix showing Pearson and Spearman values. C0-C3 and C0-C4 negative correlations visible. |
| Fig. 2 | `fig2_action_cost_heatmap.pdf` | 9x5 Spearman correlation heatmap between actions and costs with significance markers. EV actions show strong positive correlation with C3/C4. |
| Fig. 3 | `fig3_hourly_conflict.pdf` | 3-panel hourly profile: mean EV action, C0 violation rate, C3 violation rate. Demonstrates temporal overlap of EV charging and C3 violations. |
| Fig. 4 | `fig4_violation_conditioned_actions.pdf` | Action distributions split by whether C3 was violated or not. EV actions shift positive during C3 violations. |
| Fig. 5 | `fig5_timeseries_48h.pdf` | 48-hour window showing actions and costs on aligned axes. Red shading marks simultaneous C0 and C3 violation windows. |
| Fig. 6 | `fig6_last_layer_weights.pdf` | Actor output layer weight norms and action-dimension cosine similarity. Shows divergent gradient directions for EV outputs. |
| Fig. 7 | `fig7_ev_action_vs_costs.pdf` | Scatter plots of summed EV action vs C0 cost and vs C3 cost. C3 scatter shows r=0.553. |
| Fig. 8 | `fig8_ev_presence_conflict.pdf` | C3 severity conditioned on number of EVs connected. More EVs connected correlates with higher C3 cost. |

### Epoch 40 (Conflict Resolved)

Located in `docs/temperature_case_study/constraint_conflict_epoch40/`:

Matching figure set showing flipped C0-C3 correlation (now positive), reduced C0 violations, and temporal redistribution of EV charging away from high-load hours.

---

## 10. Formal Summary of Claims

This evidence supports the following thesis claims:

**Claim 1: Physically-coupled constraints in multi-constraint CMDPs create action-space conflicts invisible to parameter-space gradient methods.** The C0-C3 conflict has a clear physical origin (energy balance equation) and is measurable in action space (Spearman correlations between EV actions and C3 costs, r = 0.553) and in cost space (Pearson correlation C0 vs C3 = -0.14). Parameter-space cosine similarity (CosSim = +0.44) gives the wrong sign.

**Claim 2: The parameter-space masking effect arises from dimensionality imbalance.** The dilution ratio of aligned parameters (approximately 128,000 in early layers) to conflicting parameters (approximately 2,300 in the output layer, of which approximately 768 are EV-specific) is at least 56:1 and up to 167:1. This is sufficient to flip the sign of cosine similarity from the true negative (conflict) to a false positive (alignment).

**Claim 3: Softmax multi-Lagrangian resolves physically-coupled conflicts through temporal compromise.** By applying all constraint gradients simultaneously, the actor learns context-dependent policies: charge EVs when building load is low, defer when load is high. Lambda_0 converges monotonically to zero. Cost_0 drops from 1166 to 20.

**Claim 4: GradS fails on physically-coupled constraints because its single-constraint sampling creates oscillation.** Lambda_0 oscillates between 3.0 and 11.0 without converging. EpCost_0 see-saws between 323 and 1148. EpRet diverges. The CostRewardGradRatio explodes to 130x.

**Claim 5: The conflict is empirically detectable through cost correlation matrices and action-cost sensitivity analysis.** The analysis framework (Pearson/Spearman cost-cost correlations, action-cost Spearman correlations, hourly profiling, violation-conditioned distributions) provides a practical diagnostic for identifying constraint conflicts in CMDP settings without requiring gradient computation.

**Claim 6: Reward shaping (r_ev_smart / headroom gating) offers an alternative resolution by converting C0 from a Lagrangian constraint to a reward signal.** This removes the coupling from the Lagrangian entirely, allowing the remaining constraints (C2-C4) to be managed without conflict.

---

## 11. Connection to Related Work

### 11.1 Gradient Conflict Detection Methods

| Method | Space | Limitation for Physical Coupling |
|--------|-------|----------------------------------|
| GradS (Yao et al. 2024) | Parameter (130K dims) | Masking effect: false positive alignment |
| PCGrad (Yu et al. 2020) | Parameter | Same masking vulnerability |
| CAGrad (Liu et al. 2021) | Parameter | Same masking vulnerability |
| CoMOGA (Kim et al. 2025) | Parameter | Same masking vulnerability; latest gradient conflict method still parameter-space |

All parameter-space gradient conflict methods share the fundamental vulnerability: when constraints conflict only in the output layer but align in representation layers, high-dimensional cosine similarity reports the wrong sign.

### 11.2 Action-Space Methods

| Method | Approach | Relevance |
|--------|----------|-----------|
| Safety Layer (Dalal et al. 2018) | Linear constraint projection in action space | Would detect the coupling through dC3/da_EV > 0 |
| ATACOM (Liu et al. 2024) | Geometric projection onto constraint manifold | Handles nonlinear couplings |

These methods reason in the 9-dimensional action space where the conflict is visible (3 orders of magnitude fewer dimensions than parameter space).

### 11.3 Lagrangian Methods

| Method | Approach | Behavior on Coupled Constraints |
|--------|----------|---------------------------------|
| PPOLag (single lambda) | Single constraint | Not applicable to multi-constraint |
| PPOLagMulti (softmax) | Simultaneous all-constraint gradient | Resolves via temporal compromise |
| PID Lagrangian (Stooke et al. 2020) | PID lambda update | Stabilizes lambda dynamics; used within both Softmax and GradS |
| FAMO (Liu et al. 2023) | Loss-space balancing, bypasses gradient space | Potentially avoids masking effect |

### 11.4 Key References

1. Yao, S., et al. "Gradient Shaping for Multi-Constraint Safe Reinforcement Learning." L4DC 2024. arXiv:2312.15127.
2. Yu, T., et al. "Gradient Surgery for Multi-Task Learning." NeurIPS 2020.
3. Liu, B., et al. "Conflict-Averse Gradient Descent for Multi-task Learning." NeurIPS 2021.
4. Kim, H., et al. "Conflict-Averse Gradient Aggregation for Constrained Multi-Objective RL." ICLR 2025.
5. Dalal, G., et al. "Safe Exploration in Continuous Action Spaces." arXiv:1801.08757, 2018.
6. Liu, P., et al. "Safe RL on the Constraint Manifold: Theory and Application to ATACOM." IEEE T-RO 2024. arXiv:2404.09080.
7. Liu, B., et al. "FAMO: Fast Adaptive Multitask Optimization." NeurIPS 2023. arXiv:2306.03792.
8. Stooke, A., et al. "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods." ICML 2020.

---

## 12. Experimental Configuration Reference

### 12.1 Environment

- **Environment:** CityLearnSafety-V2G-v2
- **Buildings:** 5 (schema: `data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json`)
- **Timesteps per episode:** 8759 (1 year, hourly)
- **EV departures per episode:** approximately 1070

### 12.2 Actor Architecture

- **Type:** Gaussian MLP with learned standard deviation
- **Architecture:** 198 -> 256 (tanh) -> 256 (tanh) -> 9
- **Total parameters:** approximately 130,000
- **Output layer parameters:** 2,304 (256 x 9) weights + 9 biases = 2,313
- **Learning rate:** 0.0002

### 12.3 Lagrangian Configuration (Softmax / PPOLagMulti)

- **Lambda update:** PID (Stooke et al. 2020)
- **Lambda upper bound:** 35.0
- **Softmax temperature (tau):** 1.0
- **Cost limits:** C0=200, C1=999999, C2=1500, C3=5000, C4=8000

Per-constraint PID gains:

| Constraint | Kp | Ki | Kd |
|------------|-----|------|------|
| C0 | 2.0 | 0.05 | 0.0 |
| C2 | 0.1 | 0.01 | 0.01 |
| C3 | 0.5 | 0.05 | 0.01 |
| C4 | 0.3 | 0.03 | 0.01 |

### 12.4 Training

- **Algorithm:** PPO with clipped objective
- **PPO clip ratio:** default (0.2)
- **Target KL:** 0.08 (early stopping)
- **Mini-batch size:** 512
- **Update iterations per epoch:** 10
- **Entropy coefficient:** 0.005
- **Max gradient norm:** 0.5
- **Observation normalization:** Yes
- **Reward normalization:** Yes
- **Cost normalization:** No
- **Standardized cost advantage:** Yes

### 12.5 Reward Components

| Component | Weight | Description |
|-----------|--------|-------------|
| r_sg (self-generation) | 1.5 | Maximize local renewable usage |
| r_sb (self-balance) | 1.5 | Balance generation and consumption |
| r_ramp | 0.3 | Minimize power ramp rates |
| r_ren (renewable) | 0.3 | Maximize renewable energy fraction |
| r_barrier | 0.3 | Soft barrier for SoC limits |
| r_ev_smart | 1.5 | Headroom-gated departure-aware EV price signal (R27a) |

### 12.6 Run Identifiers

- **Softmax ablation:** r25b_softmax_ablation (40 epochs)
- **GradS ablation:** r25b_grads_ablation (18 epochs, terminated due to divergence)
- **R27a CMDP:** r27a_cmdp (40 epochs, headroom-gated variant)

### 12.7 Evaluation Script

`scripts/eval_r26hi.py` -- deterministic 1-episode evaluation with per-constraint violation analysis, per-departure C0 computation, and per-building C3 breakdown.
