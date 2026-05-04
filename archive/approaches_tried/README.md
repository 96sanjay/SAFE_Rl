# Approaches Tried

This directory contains seven major design directions explored during the Safe RL V2G project. Each subdirectory preserves the code for one approach. Some were outright failures; one (Temperature Control) succeeded as an auxiliary case study validating that the pipeline generalises beyond V2G. All contributed insights that shaped the final per-channel CMDP pipeline.

The approaches are numbered in roughly chronological order. Read them in sequence to understand how the project's understanding of the problem evolved.

---

## 01: Single Constraint Lagrangian

**Period:** R11a era
**Status:** Abandoned

**Idea.** Use standard OmniSafe PPO-Lagrangian with a single SoC constraint to enforce EV battery state-of-charge requirements at departure.

**Key files:**
- `omni_env.py` -- original CMDP environment wrapper
- `safety_env.py` -- safety cost computation
- `extractors.py` -- observation feature extractors

**Why it failed.** Multi-building V2G control requires four constraints simultaneously:
- C1: EV departure SoC deadlines
- C2: Battery cycling limits
- C3: Peak demand caps
- C4: Grid import ceiling

A single Lagrangian multiplier cannot express trade-offs between these competing concerns. The agent would satisfy whichever constraint the multiplier was tuned for and ignore the rest.

**Lesson.** Multi-constraint framework was essential from the start. This motivated the move to multi-Lagrangian with PID control.

---

## 02: Predictive Safety Filter (PSF)

**Period:** R15 era
**Status:** Abandoned

**Idea.** Train an unconstrained RL agent for maximum reward, then apply a post-hoc safety filter. The filter uses lookahead optimization over a dynamics model to project the agent's proposed action forward in time, and solves a QP to find the nearest safe action if a constraint violation is predicted.

**Key files:**
- `psf/psf_extract.py` -- feature extraction for the filter
- `psf/psf_filter.py` -- core safety filter logic
- `psf/psf_lookahead.py` -- multi-step lookahead computation
- `psf/psf_wrapper.py` -- environment wrapper integrating the filter
- `omni_env_v2g_psf.py` -- V2G environment with PSF integration
- `lookahead_psf_v2g.py` -- lookahead-based PSF for V2G

**Why it failed.** Four compounding problems:

1. **Model accuracy.** Lookahead requires a faithful dynamics model. CityLearn's building thermal model, battery degradation, and EV connect/disconnect events are too complex to approximate with the linear model the QP needs.
2. **Latency.** QP solver added 100ms+ per timestep per building. With 5 buildings and 8760 timesteps per episode, this made training infeasible.
3. **Discrete events.** EV arrivals and departures are discrete, stochastic events. The continuous QP formulation cannot represent "the EV disconnects in 3 hours."
4. **Reward destruction.** When the filter overrides the agent's action, the agent receives reward for an action it did not choose. Over time, the agent learns to propose extreme actions and let the filter correct them, never developing internal safety awareness.

**Lesson.** End-to-end constrained learning (CMDP) outperforms train-then-filter approaches for complex multi-agent systems. The agent must internalize constraint satisfaction, not outsource it.

---

## 03: Temperature Control (Auxiliary Case Study)

**Period:** Throughout project
**Status:** Completed (auxiliary validation)

**Idea.** Apply the safe RL pipeline to HVAC temperature control with thermal comfort as a single safety constraint, confirming the pipeline generalises beyond the original V2G domain.

**Key files:**
- `omni_env_temp.py` -- base temperature control environment
- `omni_env_temp_cooling_only.py` -- cooling-only variant
- `omni_env_temp_masked.py` -- masked-action variant
- `omni_env_temp_masked_reward.py` -- masked-action with shaped reward
- `policy_action_mask_temp.py` -- action masking policy
- Multiple training scripts for temperature experiments

**Outcome.** This was a successful auxiliary case study confirming the pipeline generalises beyond V2G. It serves two roles in the thesis:

1. **Single-constraint baseline.** Standard methods (CSAC-LB at 11.3%, CPO at 8.8% violation) solve a single safety constraint effectively. This sharpens the thesis question: if these methods work for one constraint, why do they fail with four? The gap between solvable (single-constraint) and unsolvable (four-constraint) is what the per-channel Multi-Lagrangian architecture fills.
2. **Cross-domain Lagrangian instability.** SAC-Lag's vanilla Lagrangian fails in temperature control (20.5% rising to 94.7% under stress), independently confirming the same instability that drove V2G from single-Lagrangian to PID-augmented multi-channel control. Same root cause, different domain -- not an artefact of V2G specifically.

The temperature work is archived here because it is separate from the main V2G pipeline, not because it failed.

**Lesson.** Single-constraint safety is a solved problem; the thesis contribution lies in the four-constraint V2G case where standard methods break down. The temperature study supports this argument by establishing the baseline of what standard methods can achieve.

---

## 04: Execution Shield

**Period:** R15-R18 era
**Status:** Abandoned

**Idea.** Post-hoc action correction. Before executing the agent's action, check whether it would violate a constraint. If so, clip the action to the nearest value within the safe region.

**Key files:**
- `omni_env_v2_shield.py` -- environment wrapper with execution shield

**Why it failed.** Two related problems:

1. **Gradient destruction.** Aggressive clipping creates a flat reward landscape in the clipped region. The agent cannot distinguish between "slightly unsafe" and "extremely unsafe" actions because both get clipped to the same safe boundary.
2. **Learned laziness.** The agent discovered that the shield would always save it. It stopped learning to avoid unsafe regions and instead adopted "lazy" policies that relied entirely on the shield for constraint satisfaction. When the shield was removed (or when constraints changed), the policy had zero internal safety awareness.

**Lesson.** Safety must be internalized through training (constraints + reward shaping), not externally imposed post-hoc. This insight applies equally to PSF (approach 02) and the shield -- both suffer from the same decoupling of learning and safety.

---

## 05: SERL Action Projection

**Period:** R15 era
**Status:** Abandoned

**Idea.** Use a differentiable QP-based safety projection layer (inspired by the SERL framework) that maps any unsafe action to the nearest feasible point in constraint space. Unlike the shield (approach 04), the projection is differentiable, so gradients flow through it during training.

**Key files:**
- `action_projection_serl.py` -- SERL QP solver for action projection
- `action_projection_stems_all4.py` -- 4-action variant for multi-building setup

**Why it failed.** Four problems:

1. **Numerical instability.** CVXPyLayers QP solver produced NaN gradients in ~2% of steps, requiring fallback logic that broke the gradient chain.
2. **Computational cost.** 50ms+ per-step overhead for the projection layer, multiplied across 5 buildings.
3. **Model dependence.** Computing the constraint Jacobian requires knowledge of the environment dynamics, which is not available during online inference in a model-free setting.
4. **Diminishing returns.** Simple action clamping combined with Lagrangian multipliers achieved the same constraint satisfaction rate with zero computational overhead.

**Lesson.** Simpler constraint enforcement (Lagrangian + clamp) beats elegant optimization when computational budget is limited. The QP approach is theoretically appealing but practically inferior for this problem scale.

---

## 06: Forecast Arbitrage

**Period:** R26h/i
**Status:** Abandoned

**Idea.** Give the agent a 24-hour electricity price forecast and shape the reward to encourage economic arbitrage: charge EVs when electricity is cheap, discharge when expensive.

**Key files:**
- `omni_env_forecast.py` -- environment with price forecast observations
- `omni_env_aggregate.py` -- aggregated observation variant
- Configs for R26h/R26i experiments

**Why it failed.** The `r_trajectory` reward signal was departure-blind. It encouraged the agent to delay charging (waiting for cheaper prices) even when the EV was about to depart with insufficient charge. Results from R26h evaluation:

- **70.1% C1 violation rate** (EV departure SoC constraint)
- EV actions were NEGATIVE (discharging) when they should have been charging
- `r_trajectory` was actively harmful: it consumed 57% of the negative EpRet component

The forecast gave the agent information about WHEN electricity is cheap but said nothing about WHEN the EV must depart. The agent optimized for price and ignored urgency.

**Lesson.** EV reward signals MUST be departure-aware. This insight led directly to the successful `r_ev_smart` (headroom-gated) reward in R27a. The headroom gate closes when the EV is running out of time to charge, overriding any price-based incentive.

---

## 07: Pure Lagrangian (No Reward Help)

**Period:** R26j
**Status:** Abandoned (120 epochs completed)

**Idea.** Test whether Lagrangian multipliers alone -- without any reward shaping assistance -- can enforce all constraints. This was a controlled experiment to isolate the Lagrangian mechanism's capability.

**Key files:**
- Configs for R26j experiment

**Results (120 epochs):**

| Metric | Value | Assessment |
|--------|-------|------------|
| Lambda_1 | Saturated at 20.0 (epoch 27) | Maxed out, still not enough |
| EpCost_1 | Stuck at ~254 | Far above target |
| Lambda_3 | Saturated at 20.0 (epoch 18) | Maxed out |
| EpCost_3 | 42,000 (from 98,000) | Still 8x over limit |
| r_trajectory | -0.15 | 57% of negative EpRet |

Lambda saturation means the Lagrangian pushed as hard as it could and still could not drive constraint costs below their limits.

**Why it failed.** The Lagrangian multiplier communicates "violating C1 is bad" through a scalar penalty. But C1 satisfaction requires a specific sequence of actions: charge the EV steadily over the hours before departure, at a rate that reaches the target SoC. The scalar penalty tells the agent THAT it should satisfy C1 but not HOW. Without reward shaping to provide dense, step-by-step guidance, the agent cannot discover the correct sequential strategy through trial and error alone.

**Lesson.** Dense, informative reward shaping is necessary for constraints that require sequential decision-making. The Lagrangian provides the constraint pressure; the reward shaping provides the behavioral blueprint. Neither alone is sufficient. This is the project's most important finding, and it motivated the per-channel CMDP pipeline (PPO-Lag-Multi).

---

## Summary of Lessons

| # | Approach | Core Lesson |
|---|----------|-------------|
| 01 | Single Constraint | Multi-constraint framework required |
| 02 | Predictive Safety Filter | End-to-end learning beats train-then-filter |
| 03 | Temperature Control (auxiliary) | Single-constraint safety is solved; the contribution is the four-constraint gap |
| 04 | Execution Shield | Safety must be internalized, not imposed |
| 05 | SERL Action Projection | Simple methods beat elegant optimization at scale |
| 06 | Forecast Arbitrage | EV rewards must be departure-aware |
| 07 | Pure Lagrangian | Lagrangian + reward shaping, neither alone suffices |

Each lesson built on the previous ones. The primary thesis contribution is the per-channel CMDP pipeline (PPO-Lag-Multi), which achieves 0.65% C1 violation rate. CSAC-LB occupies a different Pareto corner -- better economic performance but 69% C1 violation -- confirming that no single algorithm dominates both objectives. The final design incorporates all seven lessons: multi-constraint PID Lagrangian (from 01, 07), end-to-end CMDP training (from 02, 04), dense domain-informed reward shaping (from 03, 06), simple action clamping (from 05), and departure-aware headroom gating (from 06, 07).
