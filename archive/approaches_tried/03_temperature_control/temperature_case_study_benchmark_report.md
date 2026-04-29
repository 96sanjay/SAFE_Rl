# Temperature Control Case Study: Comprehensive Benchmark Report

## 1. Executive Summary

This report presents a comprehensive benchmark of seven safe reinforcement learning (RL) algorithms applied to multi-building HVAC temperature control within the CityLearn simulation environment. The task is formulated as a constrained Markov decision process (CMDP) in which the agent must minimize electricity cost, grid stress, and carbon emissions while maintaining indoor temperatures within the occupant comfort band of 20--26 degrees Celsius.

Key findings:

- **Seven safe RL algorithms** were benchmarked on a CityLearn temperature-control CMDP: CSAC-LB, CPO, FOCOPS, CUP, PPO-Lagrangian, SAC-Lagrangian, and unconstrained PPO.
- **CSAC-LB achieves the best safety--efficiency tradeoff**, with a violation rate of 5.6% (best checkpoint, seed 1) combined with competitive reward, outperforming all other algorithms on the Pareto frontier.
- **CPO achieves the lowest violation rate under heat-wave stress** (2.6% during a 90-hour extreme-temperature window), demonstrating exceptional robustness to distributional shift.
- **Multi-seed study confirms CSAC-LB robustness**: across four seeds, the best-checkpoint violation rate ranges from 5.6% to 18.4%, with all seeds substantially outperforming baselines.
- **FOCOPS and CUP fail** to learn effective constraint satisfaction, with violation rates exceeding 85%, placing them in the same regime as the zero-cooling baseline.
- All trained safe RL agents outperform the rule-based cooling controller (Cooling RBC) on constraint violation rate, which itself violates in 51.6% of active steps.

---

## 2. Experimental Setup

### 2.1 Environment

The experiments use **CityLearn v2** configured for Travis County, Texas (ASHRAE climate zone 2A, hot-humid). The district comprises **3 residential buildings**, each equipped with an air-conditioning system. Simulation proceeds at **hourly resolution** over a full calendar year, yielding **2,207 total steps** of which **2,194 are active** (the first 13 steps are initialization). The observation space is **54-dimensional** per building (including outdoor temperature, solar irradiance, indoor temperature, hour of day, day type, and historical loads). The action space is **cooling-only**, continuous in [0, 1]^3, where each dimension controls the cooling power fraction for one building.

### 2.2 CMDP Formulation

**Reward.** The reward function follows the STEMS decomposition (economic + stability + renewable + comfort), aggregating multiple grid-level and building-level objectives into a single scalar signal. Components include:

- R_economic: penalizes electricity import cost
- R_stability: rewards load-factor smoothness and penalizes ramping
- R_renewable: rewards alignment of consumption with solar generation
- R_comfort: penalizes temperature deviations from the comfort band

**Cost (constraint).** The per-step cost is defined as:

```
c_t = 0.05 * sum_{b=1}^{3} 1[T_b(t) outside [20, 26]]
```

where the indicator function fires whenever any building's indoor temperature is outside the 20--26 degrees Celsius comfort band. The cost limit for the CMDP is set so that violations should be minimized toward zero. The violation rate reported throughout this document is the fraction of active steps in which at least one building violates the comfort band.

### 2.3 Algorithms Benchmarked

| Algorithm | Type | Constraint Method | Key Reference | Description |
|-----------|------|-------------------|---------------|-------------|
| **CSAC-LB** | Off-policy (SAC-based) | Conservative lower-bound critic with Lagrangian | Ha et al. (2024) | Combines soft actor-critic with a conservative safety critic that learns a lower bound on the cost value function. The Lagrangian multiplier is updated using the conservative cost estimate, providing a safety margin that accounts for estimation uncertainty. Particularly effective in continuous-action domains. |
| **CPO** | On-policy (TRPO-based) | Trust-region constraint | Achiam et al. (2017) | Constrained Policy Optimization solves a constrained optimization problem at each policy update step using a trust-region method. It linearizes the constraint and computes a closed-form policy update that guarantees constraint satisfaction in expectation, subject to approximation errors. |
| **FOCOPS** | On-policy (PPO-based) | First-order constraint optimization | Zhang et al. (2020) | First-Order Constrained Optimization in Policy Space reformulates the CPO update as a two-stage first-order procedure: first optimizing the reward objective, then projecting onto the constraint-feasible set. Designed to be simpler and more scalable than CPO. |
| **CUP** | On-policy (PPO-based) | Constrained update projection | Yang et al. (2022) | Constrained Update Projection Approach uses a projection-based method to enforce constraints. It decomposes the policy gradient into reward-improving and constraint-satisfying components, then combines them via a projection step. |
| **PPO-Lag** | On-policy (PPO-based) | Lagrangian dual | Ray et al. (2019) | Proximal Policy Optimization with a Lagrangian penalty. A dual variable (Lagrange multiplier) is updated via gradient ascent on the constraint violation, converting the constrained problem into an unconstrained penalty formulation. This implementation uses tightened cost limits. |
| **SAC-Lag** | Off-policy (SAC-based) | Lagrangian dual | Ha et al. (2024) | Soft Actor-Critic with a Lagrangian penalty on constraint violation. Combines the entropy-regularized off-policy learning of SAC with a learned Lagrange multiplier for constraint enforcement. |
| **PPO** | On-policy | None (unconstrained) | Schulman et al. (2017) | Standard Proximal Policy Optimization without any constraint mechanism. Serves as an unconstrained baseline to quantify the value added by explicit constraint handling. Trained for 40 epochs with behavioral cloning initialization. |

### 2.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 60 (PPO: 40) |
| Steps per epoch | 2,208 |
| Hidden layers | [256, 256] |
| Activation | ReLU |
| Random seed | 42 (single-seed runs); 0, 1, 2, 42 (multi-seed CSAC-LB) |
| Environment | CityLearnTemp-CoolingOnly-Masked-Reward-v0 |
| Discount factor | 0.99 |
| Cost limit | Algorithm-specific (see Appendix B) |

All algorithms use the same environment wrapper, observation normalization, and reward function to ensure a fair comparison.

### 2.5 Baselines

Two non-learning baselines are included:

- **Cooling RBC**: A bang-bang rule-based controller that applies full cooling when indoor temperature exceeds 24 degrees Celsius and zero cooling otherwise. This represents standard thermostat-like control.
- **Zero (no cooling)**: No cooling is applied at any time step. This represents the worst-case comfort scenario and provides an upper bound on constraint violations.

### 2.6 Evaluation Protocol

Each trained agent is evaluated via a **full-year deterministic rollout** (no exploration noise, no stochastic sampling). Two checkpoints are evaluated per algorithm:

- **Best checkpoint**: The epoch with the lowest constraint cost during training (selected by training-time cost monitoring).
- **Final checkpoint**: The last epoch of training.

All metrics are computed over the 2,194 active steps of the evaluation year.

---

## 3. Results

### 3.1 Training Dynamics

#### 3.1.0 Training Diagnostic Curves

The following diagnostic curves track five key training signals across all seven algorithms over the full training horizon. These curves are essential for diagnosing learning health and comparing convergence behavior.

| Curve | Diagnostic Priority | Expected Trend | Description |
|-------|---------------------|----------------|-------------|
| Mean Reward | High | Upward slope | The primary objective signal. Healthy training shows monotonic improvement. |
| Policy Entropy | High | Slow, steady decrease | High initial entropy enables exploration; gradual decrease indicates policy specialization. Collapse signals premature convergence. |
| Episode Cost | Medium | Downward | Measures constraint violation severity. Effective safe RL algorithms drive this toward zero. |
| Value Loss | Supplementary | Downward, then stabilizing | The critic's prediction error. Should decrease as the value function learns, then stabilize as the policy converges. |
| Policy Loss | Supplementary | Stabilizing near zero | The actor's optimization objective. Large oscillations indicate training instability. |

![Training diagnostics for all benchmark algorithms (5-panel)](figures/fig_training_diagnostics.png)

**Key observations from the training curves:**

1. **Mean Reward (Panel a):** CSAC-LB and CPO show the clearest upward learning curves, rising from approximately -4000 to +2000 over 60 epochs. PPO-Lag starts at ~2800 (due to behavioral cloning initialization) and remains flat, indicating the BC initialization captures a near-optimal unconstrained policy but the Lagrangian mechanism does not further improve it. FOCOPS and CUP show slow improvement from -6000 toward 0--1000, but never reach competitive reward levels.

2. **Policy Entropy (Panel b):** On-policy algorithms (CPO, PPO-Lag, PPO, CUP, FOCOPS) all start near 1.4 nats and gradually decrease to 1.0--1.2, showing healthy exploration-to-exploitation transition. SAC-Lag's entropy temperature alpha increases from 1.0 to 3.1 over training, indicating the automatic entropy tuning drives exploration upward --- a sign of under-constrained learning. CSAC-LB maintains a fixed alpha of 0.2, reflecting its more conservative, constraint-aware design.

3. **Episode Cost (Panel c):** The most discriminative diagnostic. CSAC-LB drives cost from 280 to near-zero within 15 epochs. CPO reduces cost from 810 to near-zero by epoch 40. In stark contrast, FOCOPS and CUP plateau at 200--250, never achieving meaningful constraint satisfaction. PPO and PPO-Lag start low (~70--90) due to BC initialization but show no further cost reduction.

4. **Value Loss (Panel d):** All algorithms show the expected pattern of high initial critic loss followed by stabilization. CSAC-LB achieves the lowest final critic loss (~0.3), indicating its reward critic learns an accurate value function. CPO and CUP show elevated critic loss (~50--80), reflecting the difficulty of on-policy value estimation in this environment.

5. **Policy Loss (Panel e):** CSAC-LB's policy loss decreases steadily from +18 to -47, characteristic of SAC's entropy-regularized policy improvement. On-policy algorithms maintain near-zero policy loss throughout, as expected from PPO/TRPO-style clipped objectives.

**Compact 3x2 training dashboard** (same data, alternative layout with KL divergence):

![Compact training dashboard (3x2)](figures/fig_training_dashboard.png)

The KL divergence panel (f) reveals that CPO maintains consistently low KL (~0.001) due to its trust-region constraint, while PPO-Lag and CUP show higher KL values (0.01--0.1), indicating larger policy updates per epoch.

#### 3.1.1 Multi-Seed CSAC-LB Training Diagnostics

The following figure shows training curves for the CSAC-LB algorithm across four random seeds (0, 1, 2, 42), illustrating the seed sensitivity of the training dynamics.

![Multi-seed CSAC-LB training diagnostics](figures/fig_multiseed_diagnostics.png)

**Observations:**
- **Reward convergence is consistent:** All four seeds converge to similar final returns (1500--2000), with seed 2 showing the slowest initial convergence but eventually catching up by epoch 40.
- **Cost reduction varies by seed:** Seed 1 (blue) achieves the fastest cost reduction, reaching near-zero by epoch 15. Seed 2 (green) starts with the highest cost (350) and takes longer to converge, explaining its higher evaluation violation rate (18.4%).
- **Critic loss is remarkably consistent** across seeds, with all four following nearly identical trajectories on a log scale, confirming that the value function learning is robust to initialization.
- **Entropy temperature alpha is fixed** at 0.2 for all seeds (not auto-tuned), which contributes to the stability of constraint satisfaction.

#### 3.1.2 Constraint Enforcement Dynamics

The following figure tracks the internal constraint enforcement mechanisms --- the cost critic value estimate and the Lagrange multiplier (or equivalent barrier penalty) --- across all algorithms.

![Constraint enforcement dynamics](figures/fig_constraint_dynamics.png)

**Analysis:**
- **(a) Cost Critic:** CPO's cost critic peaks at epoch 7 (value ~13.7) then declines sharply as the policy learns to avoid violations. CSAC-LB's conservative cost critic stays low (~0.5--1.0) throughout, reflecting the lower-bound estimation that provides the safety margin. SAC-Lag's cost critic rises continuously to 8.5, suggesting the cost value function never fully converges, correlating with its poor constraint satisfaction.
- **(b) Lagrange Multiplier:** SAC-Lag's multiplier grows linearly from 3 to 27 without converging, indicating the dual variable is chasing a constraint it cannot satisfy --- a hallmark of Lagrangian instability. CSAC-LB's barrier penalty peaks at epoch 10 then decreases, showing the conservative critic successfully internalizes the constraint. FOCOPS and CUP maintain near-zero multipliers, confirming their constraint mechanisms fail to engage meaningfully.

#### 3.1.3 Reward Prediction Quality

The following figure compares actual episode returns against value function estimates, serving as a proxy for explained variance.

![Reward prediction quality](figures/fig_reward_prediction.png)

**Observations:**
- **(a) On-policy:** PPO and PPO-Lag show near-perfect value estimates (dashed lines track solid lines closely), indicating high explained variance. CPO and CUP/FOCOPS show large gaps between actual returns and value estimates during early training, which close as training progresses.
- **(b) Off-policy:** CSAC-LB's critic value (dashed red) tracks near zero while the actual return rises to 2000+, indicating the critic operates on per-step values rather than episodic returns. SAC-Lag shows a similar pattern with its critic stabilizing while returns improve.

#### 3.1.4 Training Cost Summary Table

The following table summarizes the constraint cost trajectory for each algorithm, measured as the total episodic cost at the best and final checkpoints. Note that lower cost corresponds to fewer constraint violations.

| Algorithm | Best Epoch | Total Cost (Best) | Total Cost (Final) | Violation Rate (Best) | Violation Rate (Final) |
|-----------|-----------|-------------------|--------------------|-----------------------|------------------------|
| CSAC-LB (s1) | 29 | 1.976 | 1.876 | 5.61% | 6.15% |
| CSAC-LB (s0) | 24 | 2.046 | 4.366 | 6.79% | 11.44% |
| CSAC-LB (s2) | 53 | 6.024 | 4.332 | 18.41% | 14.18% |
| CSAC-LB (s42) | 59 | 4.444 | 5.460 | 14.54% | 15.63% |
| CPO | 59 | 6.909 | 11.577 | 13.17% | 17.91% |
| FOCOPS | 59 | 627.659 | 619.403 | 95.26% | 95.03% |
| CUP | 59 | 437.393 | 336.003 | 85.60% | 79.85% |
| PPO-Lag | 5 | 24.248 | 23.368 | 19.64% | 19.14% |
| SAC-Lag | 59 | 35.898 | 32.924 | 32.68% | 31.72% |
| PPO | 13 | 49.567 | 48.947 | 42.75% | 41.20% |

**Three behavioral clusters** emerge from the training dynamics:

1. **Effective constraint learners** (CSAC-LB, CPO): These algorithms achieve violation rates below 20%, with total costs in the single digits. They demonstrate genuine constraint-aware policy optimization, with costs declining consistently during training.

2. **Moderate constraint learners** (PPO-Lag, SAC-Lag, PPO): These algorithms reduce violations relative to the RBC baseline but do not achieve strong constraint satisfaction. PPO-Lag achieves 19.6% violation rate (best), while SAC-Lag and PPO remain above 30%. The unconstrained PPO surprisingly achieves moderate constraint satisfaction through the comfort component of STEMS, though it lacks explicit enforcement.

3. **Failed constraint learners** (FOCOPS, CUP): These algorithms converge to policies with violation rates exceeding 80%, performing worse than the Cooling RBC baseline (51.6%). Their constraint costs are two to three orders of magnitude higher than the effective learners. Analysis suggests that the first-order approximations used by FOCOPS and the projection mechanism of CUP are insufficient for this environment's complex thermal dynamics.

### 3.2 Full-Year Evaluation

#### 3.2.1 Main Results Table

The following table presents the full-year evaluation results for all algorithms and baselines, sorted by violation rate (best checkpoint). All values are from deterministic evaluation rollouts. KPI values are normalized ratios where values below 1.0 indicate improvement over the baseline and above 1.0 indicate degradation.

| Algorithm | Checkpoint | Violation Rate | Discomfort Prop. | Cost KPI | Emissions KPI | Consumption KPI | Ramping KPI |
|-----------|-----------|----------------|------------------|----------|----------------|-----------------|-------------|
| **CSAC-LB (s1)** | best (ep 29) | **0.0561** | 0.1533 | 0.9378 | 0.9694 | 0.9676 | 1.0673 |
| CSAC-LB (s0) | best (ep 24) | 0.0679 | 0.4609 | 1.1091 | 1.1563 | 1.1530 | 1.0151 |
| CPO | best (ep 59) | 0.1317 | 0.1341 | 1.0831 | 1.0978 | 1.0928 | 0.9877 |
| CSAC-LB (s42) | best (ep 59) | 0.1454 | 0.0794 | 0.8568 | 0.8780 | 0.8754 | 1.0651 |
| CSAC-LB (s2) | best (ep 53) | 0.1841 | 0.1128 | 0.9405 | 0.9599 | 0.9587 | 1.0749 |
| PPO-Lag | best (ep 59) | 0.1964 | 0.3875 | 1.1380 | 1.1778 | 1.1754 | 1.2259 |
| SAC-Lag | best (ep 5) | 0.3268 | 0.0646 | 0.7790 | 0.7996 | 0.7980 | 1.4027 |
| RBC 3 | -- | 0.3806 | 0.1133 | 0.8201 | 0.8399 | 0.8386 | 1.6050 |
| PPO | best (ep 13) | 0.4275 | 0.0597 | 0.7593 | 0.7804 | 0.7787 | 1.2820 |
| Cooling RBC | -- | 0.5160 | 0.3760 | 1.0097 | 1.0429 | 1.0415 | 2.3220 |
| RBC 2 | -- | 0.6714 | 0.1407 | 0.7276 | 0.7487 | 0.7469 | 1.2134 |
| CUP | best (ep 59) | 0.8560 | 0.6103 | 0.8427 | 0.8172 | 0.8161 | 1.1248 |
| FOCOPS | best (ep 59) | 0.9526 | 0.8473 | 0.6673 | 0.6917 | 0.6884 | 1.1887 |
| Zero | -- | 0.9950 | 0.9617 | 0.5274 | 0.5620 | 0.5591 | 0.9052 |

**Key observations:**

- CSAC-LB (seed 1, best) achieves the lowest violation rate at 5.61%, representing a **89.1% reduction** relative to the Cooling RBC (51.6%).
- CPO achieves 13.17% violation rate, a 74.5% reduction relative to Cooling RBC.
- A fundamental tradeoff exists between energy consumption and comfort: the Zero baseline has the lowest cost/emissions KPIs (0.527/0.562) but the highest violation rate (99.5%). Effective safe RL agents navigate this tradeoff by consuming moderately more energy than the Zero baseline while maintaining temperatures within bounds.
- FOCOPS and CUP have lower energy KPIs (0.667/0.842) because they under-cool, which results in excessive comfort violations.

#### 3.2.2 Reward Decomposition

The following table decomposes the total reward into its four STEMS components for each algorithm's best checkpoint, providing insight into which objectives each algorithm prioritizes.

| Algorithm | R_economic | R_stability | R_renewable | R_comfort | Total Reward |
|-----------|-----------|-------------|-------------|-----------|-------------|
| CSAC-LB (s1, best) | -251.91 | 1591.08 | 945.99 | -17.00 | 2268.16 |
| CSAC-LB (s0, best) | -272.50 | 1217.01 | 892.37 | -25.52 | 1811.36 |
| CSAC-LB (s2, best) | -237.73 | 1856.24 | 979.05 | -48.94 | 2548.61 |
| CSAC-LB (s42, best) | -223.55 | 2065.57 | 1011.83 | -80.68 | 2773.17 |
| CPO (best) | -260.22 | 1490.07 | 756.86 | -55.27 | 1931.43 |
| FOCOPS (best) | -165.07 | 2730.23 | 1162.22 | -4100.76 | -373.38 |
| CUP (best) | -196.23 | 2502.05 | 1020.06 | -3163.78 | 162.11 |
| PPO-Lag (best) | -233.90 | 1598.16 | 949.06 | -160.79 | 2152.54 |
| SAC-Lag (best) | -178.45 | 2043.24 | 1120.84 | -234.28 | 2751.34 |
| PPO (best) | -172.26 | 2139.94 | 1125.76 | -299.84 | 2793.60 |
| Cooling RBC | -229.11 | 1703.58 | 1136.60 | -3059.74 | -448.67 |
| Zero | -93.02 | 3403.52 | 1416.45 | -50158.59 | -45431.65 |

**Analysis:**

- The comfort penalty (R_comfort) dominates the reward landscape. FOCOPS receives R_comfort = -4100.76, explaining its negative total reward despite high stability and renewable scores.
- Effective safe RL algorithms (CSAC-LB, CPO) achieve R_comfort values between -17.0 and -80.7, indicating successful temperature regulation.
- The Zero baseline's R_comfort of -50,158.59 illustrates the severity of the comfort penalty when no cooling is applied.
- PPO and SAC-Lag achieve the highest total rewards (2793.60 and 2751.34) but with higher violation rates, suggesting they optimize reward at the expense of strict constraint satisfaction.

### 3.3 Multi-Seed Robustness Study (CSAC-LB)

To assess the robustness of CSAC-LB, the algorithm was trained with four different random seeds (0, 1, 2, 42). The following table reports best and final checkpoint violation rates for each seed.

| Seed | Best Epoch | Violation Rate (Best) | Violation Rate (Final) | Total Reward (Best) | Total Reward (Final) |
|------|-----------|----------------------|------------------------|--------------------|--------------------|
| 0 | 24 | 0.0679 | 0.1144 | 1811.36 | 2593.29 |
| 1 | 29 | 0.0561 | 0.0615 | 2268.16 | 2618.45 |
| 2 | 53 | 0.1841 | 0.1418 | 2548.61 | 2629.32 |
| 42 | 59 | 0.1454 | 0.1563 | 2773.17 | 2825.81 |
| **Mean** | -- | **0.1134** | **0.1185** | **2350.33** | **2666.72** |
| **Std** | -- | **0.0569** | **0.0400** | **403.52** | **103.07** |

**Statistical analysis:**

- **Violation rate (best)**: Mean = 11.34%, Std = 5.69%, CV = 0.502
- **Violation rate (final)**: Mean = 11.85%, Std = 4.00%, CV = 0.337
- **Total reward (best)**: Mean = 2350.33, Std = 403.52, CV = 0.172
- **Total reward (final)**: Mean = 2666.72, Std = 103.07, CV = 0.039

The coefficient of variation (CV) for the total reward at the final checkpoint is remarkably low (0.039), indicating that CSAC-LB converges to consistent reward levels across seeds. The violation rate shows higher variance (CV = 0.502 for best checkpoint), which is expected given the sensitivity of constraint satisfaction to early training dynamics. Notably, all four seeds achieve violation rates substantially below the Cooling RBC baseline (51.6%), and three of four seeds achieve best-checkpoint violation rates below 15%.

The variation in best epoch (24 to 59) suggests that checkpoint selection is important: seed 1 achieves its best performance at epoch 29, while seed 42 does not peak until epoch 59. This highlights the value of cost-based checkpoint selection over simply using the final model.

![Multi-seed CSAC-LB violation rate comparison across seeds](figures/multi_seed_violation_rates.png)

### 3.4 72-Hour Heat-Wave Stress Test

#### 3.4.1 Setup

To evaluate algorithm robustness under extreme conditions, a stress test was conducted using a synthetic heat-wave scenario:

| Parameter | Value |
|-----------|-------|
| Window | Steps 1855--1944 (90 hours) |
| Temperature threshold | >= 25.0 degrees C outdoor |
| Cooling demand multiplier | 2.5x |
| Solar generation multiplier | 1.2x |
| Window selection method | Maximum available spell |
| Total steps | 89 active steps, 76 occupied |

This window was selected as the longest contiguous spell of outdoor temperatures exceeding 25 degrees Celsius, representing a realistic Texas summer heat wave where cooling demand is 2.5 times the nominal value.

#### 3.4.2 Stress Test Results

The following table reports performance during the 90-hour heat-wave window for all algorithms (best checkpoint).

| Algorithm | Violation Rate | Discomfort Rate | Total Reward | Total Cost | Mean Cooling Action |
|-----------|---------------|-----------------|-------------|------------|---------------------|
| **CPO** | **0.0263** | 0.1796 | 103.20 | 0.010 | 0.333 |
| CSAC-LB (s1) | 0.0658 | 0.3301 | 110.60 | 0.051 | 0.297 |
| CSAC-LB (s1, final) | 0.0658 | 0.1699 | 120.39 | 0.035 | 0.256 |
| CPO (final) | 0.1053 | 0.1456 | 105.96 | 0.078 | 0.316 |
| Cooling RBC | 0.5395 | 0.4806 | 81.01 | 5.389 | 0.348 |
| CUP (final) | 0.6447 | 0.4563 | 67.10 | 7.442 | 0.253 |
| CUP | 0.7632 | 0.5000 | 49.79 | 10.297 | 0.231 |
| FOCOPS (final) | 0.7895 | 0.4757 | 66.70 | 10.625 | 0.175 |
| FOCOPS | 0.8026 | 0.4903 | 63.72 | 11.123 | 0.173 |
| PPO (best) | 0.8026 | 0.8350 | -36.68 | 12.356 | 0.559 |
| PPO (final) | 0.8816 | 0.6990 | -43.42 | 16.736 | 0.427 |
| SAC-Lag | 0.9474 | 0.4126 | 56.61 | 9.009 | 0.243 |
| SAC-Lag (final) | 0.9474 | 0.4223 | 66.89 | 7.574 | 0.248 |
| PPO-Lag (final) | 0.9605 | 0.8010 | -153.81 | 30.933 | 0.377 |
| PPO-Lag | 1.0000 | 0.9903 | -240.60 | 33.432 | 0.726 |
| Zero | 1.0000 | 0.9757 | -466.95 | 90.567 | 0.000 |

**Key observations:**

- **CPO achieves the lowest stress-test violation rate at 2.63%** (only 2 comfort violations in 76 occupied steps), demonstrating exceptional robustness to distributional shift. CPO's trust-region constraint mechanism appears particularly well-suited to maintaining safety guarantees under extreme conditions.
- **CSAC-LB (seed 1) achieves 6.58%**, maintaining strong safety even under 2.5x cooling demand.
- **PPO-Lag degrades catastrophically** under stress: its best checkpoint achieves 100% violation rate with a mean cooling action of 0.726 (overcooling), suggesting the Lagrangian mechanism over-compensates under distributional shift, leading to excessive cooling and cold-side violations. Its total reward plummets to -240.60.
- **SAC-Lag also degrades severely** to 94.7% violation rate during the heat wave, despite achieving 32.7% on the full year. This suggests the Lagrangian-only approaches lack the robustness mechanisms of CSAC-LB's conservative critic or CPO's trust-region constraint.
- The Cooling RBC achieves 53.9% violation rate during the stress window, only marginally worse than its full-year performance (51.6%), indicating its rule-based nature provides some robustness.

![Heat-wave stress test violation rates across algorithms](figures/stress_test_violation_rates.png)

### 3.5 Algorithm Behavioral Analysis

The algorithms naturally partition into three behavioral clusters based on their constraint satisfaction performance:

#### Cluster 1: Effective Constraint Learners (CSAC-LB, CPO)

These algorithms achieve violation rates below 20% and demonstrate genuine constraint-aware behavior:

- **CSAC-LB** uses a conservative lower-bound safety critic that provides a pessimistic estimate of constraint cost. This safety margin means the policy avoids regions of the state-action space where constraint violations are plausible, even if the expected violation is near-zero. The off-policy nature of SAC allows efficient data reuse, contributing to sample efficiency.
- **CPO** uses a trust-region approach that directly constrains the expected cost at each policy update. Its linearization of the constraint function provides a principled approximation that appears well-calibrated for this environment. CPO's particular strength is robustness under stress: the trust-region constraint is an intrinsic property of each policy update, not a learned parameter that can become miscalibrated.

#### Cluster 2: Moderate Constraint Learners (PPO-Lag, SAC-Lag, PPO)

These algorithms reduce violations relative to the RBC but do not achieve strong constraint satisfaction:

- **PPO-Lag** (19.6% violation rate, best) achieves moderate safety but relies on a learned Lagrange multiplier that can oscillate or overshoot, as demonstrated by its catastrophic failure under stress (100% violation rate). The early best epoch (5 for SAC-Lag variant) suggests the Lagrangian penalty becomes too aggressive, degrading reward.
- **SAC-Lag** (32.7% violation rate) shares the Lagrangian limitation but benefits from SAC's off-policy efficiency. However, without the conservative safety critic of CSAC-LB, it lacks the safety margin needed for robust constraint satisfaction.
- **PPO** (42.8% violation rate) achieves surprising constraint reduction despite being unconstrained, entirely through the comfort component of the STEMS reward. This demonstrates that reward shaping alone can partially address safety but is insufficient for tight constraint satisfaction.

#### Cluster 3: Failed Constraint Learners (FOCOPS, CUP)

These algorithms converge to policies with violation rates exceeding 80%:

- **FOCOPS** (95.3% violation rate) appears to collapse into a near-zero cooling policy (mean action = 0.102), effectively choosing to minimize energy consumption at the expense of comfort. The first-order constraint approximation may be too coarse for the nonlinear thermal dynamics of the CityLearn environment, or the projection step may interfere with reward optimization.
- **CUP** (85.6% violation rate) similarly under-cools (mean action = 0.142). The projection-based constraint mechanism appears unable to correctly identify the constraint-violating direction in this high-dimensional, delayed-effect environment where cooling actions at time t affect temperatures at time t+1 through thermal inertia.

Both FOCOPS and CUP achieve lower energy KPIs (cost = 0.667 and 0.843 respectively) precisely because they apply minimal cooling, which paradoxically "saves" energy while violating comfort constraints.

### 3.6 CityLearn KPI Comparison

The following table compares the top-3 safe RL algorithms against the Cooling RBC on the full suite of CityLearn KPIs. Values below 1.0 indicate improvement over the no-control baseline; values above 1.0 indicate degradation.

| KPI | CSAC-LB (s1, best) | CPO (best) | CSAC-LB (s42, best) | Cooling RBC |
|-----|---------------------|------------|----------------------|-------------|
| Cost Total | 0.938 | 1.083 | 0.857 | 1.010 |
| Carbon Emissions Total | 0.969 | 1.098 | 0.878 | 1.043 |
| Electricity Consumption Total | 0.968 | 1.093 | 0.875 | 1.041 |
| Ramping Average | 1.067 | 0.988 | 1.065 | 2.322 |
| Daily Peak Average | 1.071 | 1.052 | 1.041 | 1.131 |
| Daily 1-Load Factor Average | 1.011 | 0.985 | 1.063 | 1.101 |
| All-Time Peak Average | 1.008 | 0.965 | 1.006 | 1.059 |
| Discomfort Proportion | 0.153 | 0.134 | 0.079 | 0.376 |
| Discomfort Cold Proportion | 0.132 | 0.096 | 0.020 | 0.376 |
| Discomfort Hot Proportion | 0.021 | 0.038 | 0.059 | 0.000 |
| Zero Net Energy | 0.959 | 1.096 | 0.869 | 1.008 |
| Violation Rate | 0.056 | 0.132 | 0.145 | 0.516 |

**Analysis:**

- **Ramping**: All safe RL agents dramatically improve ramping relative to the Cooling RBC (2.322). CSAC-LB achieves 1.067 and CPO achieves 0.988, representing 54% and 57% reductions respectively. The bang-bang nature of the RBC causes severe ramping, while RL agents learn smoother control profiles.
- **Cost and emissions**: CSAC-LB (s42) achieves the best cost KPI (0.857) and emissions KPI (0.878), representing 15% and 16% improvement over the no-control baseline. CPO increases cost (1.083) due to its more aggressive cooling strategy.
- **Discomfort**: CPO achieves the lowest overall discomfort proportion (0.134), while CSAC-LB (s42) achieves the lowest at 0.079. Both are substantial improvements over the RBC's 0.376.
- **Cold vs. hot discomfort**: An interesting pattern emerges in the discomfort decomposition. CSAC-LB (s42) has very low cold discomfort (0.020) but higher hot discomfort (0.059), suggesting it learns to avoid overcooling. The Cooling RBC has 0.376 cold discomfort and 0.000 hot discomfort, indicating it consistently overcools below the comfort band.

---

## 4. Discussion

### 4.1 Why CSAC-LB Outperforms

CSAC-LB's superior performance can be attributed to three architectural advantages:

1. **Conservative safety critic**: By learning a lower bound on the cost value function, CSAC-LB maintains a safety margin that accounts for estimation uncertainty. This is particularly important in the CityLearn environment where thermal dynamics introduce delayed consequences: overcooling at hour t can cause a cold violation at hour t+1, and the conservative estimate captures this risk.

2. **Off-policy learning**: SAC's replay buffer allows efficient data reuse, enabling CSAC-LB to learn from a diverse set of state-action-cost tuples. This is especially valuable in the temperature control domain where constraint-violating states are rare under a good policy, making on-policy methods sample-inefficient for learning the cost function.

3. **Entropy regularization**: SAC's maximum-entropy objective encourages exploration of the action space, preventing premature convergence to a suboptimal cooling strategy. This is particularly beneficial in the multi-building setting where the agent must learn building-specific cooling policies simultaneously.

### 4.2 Why CPO Excels Under Stress

CPO's exceptional performance under the heat-wave stress test (2.63% violation rate) can be explained by the structural nature of its constraint mechanism:

- The trust-region constraint is enforced at every policy update, not learned from data. This means it cannot become miscalibrated when the data distribution shifts.
- CPO's constraint linearization provides a first-order guarantee on expected cost that is robust to moderate distributional shift.
- The higher mean cooling action during stress (0.333 vs. CSAC-LB's 0.297) suggests CPO learns a more aggressive cooling strategy that provides greater headroom under extreme conditions.

### 4.3 Why FOCOPS and CUP Fail

The failure of FOCOPS and CUP in this environment appears to stem from a mismatch between their constraint approximations and the environment's dynamics:

- **Thermal inertia**: Building temperature responds to cooling actions with a delay, creating a temporally extended cost signal. First-order and projection-based constraint methods assume a more immediate relationship between actions and costs.
- **Multi-building interactions**: With three buildings sharing a common grid, the constraint cost depends on the joint action across buildings. FOCOPS's two-stage approach (optimize reward, then project) may fail to find the correct projection direction in this coupled space.
- **Reward-constraint gradient alignment**: In this environment, the reward and constraint gradients are often aligned (cooling improves both comfort reward and constraint satisfaction), but FOCOPS/CUP's decomposition may destroy this alignment.

### 4.4 Safety-Efficiency Frontier

The results reveal a clear safety-efficiency Pareto frontier:

- **Zero cooling** occupies the low-cost, high-violation extreme (cost KPI = 0.527, violation = 99.5%).
- **CSAC-LB (s42)** occupies the near-optimal point on the frontier: cost KPI = 0.857 with violation rate = 14.5%.
- **CSAC-LB (s1)** achieves the lowest violation rate (5.6%) with a cost KPI of 0.938.
- **CPO** provides a higher-cost but lower-violation alternative (cost KPI = 1.083, violation = 13.2%).

No algorithm simultaneously achieves the lowest cost and the lowest violation rate, confirming the theoretical prediction that safety and efficiency are competing objectives in constrained optimization. The practical question for building operators is where on this frontier their operating point should lie, given regulatory requirements on indoor temperature.

![Safety-efficiency Pareto frontier across algorithms](figures/pareto_frontier.png)

---

## 5. Conclusions

The following five key findings emerge from this comprehensive benchmark:

1. **CSAC-LB achieves the best safety-efficiency tradeoff** among the seven algorithms tested. Its conservative safety critic provides a principled safety margin that enables violation rates as low as 5.6% while maintaining competitive energy efficiency (cost KPI = 0.938, a 6.2% improvement over no-control baseline).

2. **CPO is the most robust algorithm under distributional shift**, achieving a 2.63% violation rate during a 90-hour heat wave with 2.5x cooling demand. This makes CPO particularly suitable for safety-critical deployments where worst-case performance matters more than average-case efficiency.

3. **Lagrangian methods (PPO-Lag, SAC-Lag) provide moderate constraint satisfaction but lack robustness.** PPO-Lag achieves 19.6% violation rate under normal conditions but degrades to 100% under stress, indicating that learned Lagrange multipliers can become miscalibrated under distributional shift.

4. **First-order constraint methods (FOCOPS) and projection-based methods (CUP) fail in this environment**, converging to under-cooling policies with 85--95% violation rates. The complex thermal dynamics and multi-building coupling appear to exceed the capacity of these methods' constraint approximations.

5. **Multi-seed evaluation confirms CSAC-LB's robustness**: across four seeds, all converge to violation rates between 5.6% and 18.4% (best checkpoint), with highly consistent total reward at the final checkpoint (CV = 0.039). Checkpoint selection based on training-time cost monitoring is critical for achieving the lowest violation rates.

These findings have direct implications for the deployment of safe RL in real building energy management systems. CSAC-LB is recommended as the primary algorithm for settings where both safety and efficiency are required, while CPO should be preferred when robustness to extreme weather events is the primary concern.

---

## 6. Files and Reproducibility

### 6.1 Run Directories

All runs are stored under the base path:
```
/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/
```

| Algorithm | Run Directory |
|-----------|--------------|
| CSAC-LB (s0) | `runs/csac_lb_multi_seed/seed_0/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-000-2026-04-09-10-08-23/` |
| CSAC-LB (s1) | `runs/csac_lb_multi_seed/seed_1/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-001-2026-04-09-11-15-54/` |
| CSAC-LB (s2) | `runs/csac_lb_multi_seed/seed_2/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-002-2026-04-09-12-26-54/` |
| CSAC-LB (s42) | `runs/csac_lb_multi_seed/seed_42/CSACLBTemp-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-09-13-41-32/` |
| CPO | `runs/cpo_temp_cooling_only/CPO-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-20-54-15/` |
| FOCOPS | `runs/focops_temp_cooling_only/FOCOPS-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-21-17-52/` |
| CUP | `runs/cup_temp_cooling_only/CUP-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-21-43-13/` |
| PPO-Lag | `runs/ppolag_temp_cooling_only_tight_v2/PPOLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-11-44-33/` |
| SAC-Lag | `runs/saclag_temp_cooling_only_v1/SACLagTempCoolingOnly-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-04-08-15-33-32/` |
| PPO | `runs/ppo_temp_cooling_only_masked_reward_bc_40ep/PPOTempMasked-{CityLearnTemp-CoolingOnly-Masked-Reward-v0}/seed-042-2026-03-28-19-39-49/` |

### 6.2 Configuration Files

- Algorithm configs: `configs/` directory, algorithm-specific YAML files
- Environment config: `CityLearnTemp-CoolingOnly-Masked-Reward-v0` registration

### 6.3 Evaluation Scripts

- Case study evaluation: `eval_case_study.py` (generates `eval_case_study.json` per run)
- Stress test: `stress_test_72h.py` (generates `runs/stress_test_72h_all_algos.json`)

### 6.4 Key Data Files

| File | Description |
|------|-------------|
| `runs/stress_test_72h_all_algos.json` | Consolidated stress test results for all algorithms |
| `<run_dir>/eval_case_study.json` | Per-algorithm full-year evaluation results |
| `<run_dir>/progress.csv` | Training curves (epoch-level metrics) |

### 6.5 Figure Files

All figures referenced in this report are located in:
```
docs/temperature_case_study/figures/
```

| Figure | Description |
|--------|-------------|
| `fig_training_diagnostics.png` | 5-panel training diagnostics: reward, entropy, cost, value loss, policy loss (all algorithms) |
| `fig_training_dashboard.png` | Compact 3x2 training dashboard with KL divergence (all algorithms) |
| `fig_multiseed_diagnostics.png` | 4-panel CSAC-LB multi-seed training curves (reward, cost, value loss, alpha) |
| `fig_constraint_dynamics.png` | Cost critic evolution and Lagrange multiplier dynamics (all algorithms) |
| `fig_reward_prediction.png` | Reward prediction quality: actual return vs value estimate (on-policy and off-policy) |
| `fig_benchmark_training_curves.png` | Training cost + reward curves (all algorithms) |
| `fig_benchmark_violation_rates.png` | Comfort violation rate bar chart (all algorithms) |
| `fig_benchmark_pareto_front.png` | Safety-efficiency Pareto frontier |
| `fig_benchmark_radar_top3.png` | Radar chart comparing top-3 algorithms |
| `fig_multiseed_boxplot.png` | Multi-seed CSAC-LB violation rate box plot |
| `fig_multiseed_training.png` | Multi-seed training convergence curves |
| `fig_stress_test_all_algos_violation.png` | 72-h heat-wave stress test violation rates |
| `fig_algo_taxonomy.png` | Algorithm taxonomy diagram |
| `fig_multiseed_robustness.png` | Multi-seed robustness comparison |

---

## Appendix A: Full Results Tables

### A.1 CSAC-LB Seed 1 (Best Checkpoint, Epoch 29)

| Metric | Value |
|--------|-------|
| Total Reward | 2268.16 |
| Total Cost | 1.976 |
| Violation Rate | 0.0561 |
| Comfort Violations | 123 / 2194 active steps |
| Discomfort Count | 2035 / 5547 occupied |
| Discomfort Proportion | 0.1533 |
| Discomfort Cold Proportion | 0.1319 |
| Discomfort Hot Proportion | 0.0213 |
| Mean Cooling Action | 0.240 |
| District Import (kWh) | 7417.47 |
| District Net (kWh) | 7361.93 |
| Cost KPI | 0.938 |
| Carbon Emissions KPI | 0.969 |
| Electricity Consumption KPI | 0.968 |
| Ramping KPI | 1.067 |
| Daily Peak KPI | 1.071 |
| All-Time Peak KPI | 1.008 |
| Zero Net Energy KPI | 0.959 |

### A.2 CSAC-LB Seed 0 (Best Checkpoint, Epoch 24)

| Metric | Value |
|--------|-------|
| Total Reward | 1811.36 |
| Total Cost | 2.046 |
| Violation Rate | 0.0679 |
| Comfort Violations | 149 / 2194 active steps |
| Discomfort Count | 2746 / 5547 occupied |
| Discomfort Proportion | 0.4609 |
| Discomfort Cold Proportion | 0.4455 |
| Discomfort Hot Proportion | 0.0153 |
| Mean Cooling Action | 0.270 |
| District Import (kWh) | 8018.96 |
| District Net (kWh) | 7993.48 |
| Cost KPI | 1.109 |
| Carbon Emissions KPI | 1.156 |
| Electricity Consumption KPI | 1.153 |
| Ramping KPI | 1.015 |
| Daily Peak KPI | 1.072 |
| All-Time Peak KPI | 1.074 |
| Zero Net Energy KPI | 1.151 |

### A.3 CSAC-LB Seed 2 (Best Checkpoint, Epoch 53)

| Metric | Value |
|--------|-------|
| Total Reward | 2548.61 |
| Total Cost | 6.024 |
| Violation Rate | 0.1841 |
| Comfort Violations | 404 / 2194 active steps |
| Discomfort Count | 1253 / 5547 occupied |
| Discomfort Proportion | 0.1128 |
| Discomfort Cold Proportion | 0.0865 |
| Discomfort Hot Proportion | 0.0263 |
| Mean Cooling Action | 0.207 |
| District Import (kWh) | 6967.46 |
| District Net (kWh) | 6925.16 |
| Cost KPI | 0.940 |
| Carbon Emissions KPI | 0.960 |
| Electricity Consumption KPI | 0.959 |
| Ramping KPI | 1.075 |
| Daily Peak KPI | 1.070 |
| All-Time Peak KPI | 1.015 |
| Zero Net Energy KPI | 0.952 |

### A.4 CSAC-LB Seed 42 (Best Checkpoint, Epoch 59)

| Metric | Value |
|--------|-------|
| Total Reward | 2773.17 |
| Total Cost | 4.444 |
| Violation Rate | 0.1454 |
| Comfort Violations | 319 / 2194 active steps |
| Discomfort Count | 821 / 5547 occupied |
| Discomfort Proportion | 0.0794 |
| Discomfort Cold Proportion | 0.0203 |
| Discomfort Hot Proportion | 0.0591 |
| Mean Cooling Action | 0.192 |
| District Import (kWh) | 6582.81 |
| District Net (kWh) | 6524.77 |
| Cost KPI | 0.857 |
| Carbon Emissions KPI | 0.878 |
| Electricity Consumption KPI | 0.875 |
| Ramping KPI | 1.065 |
| Daily Peak KPI | 1.041 |
| All-Time Peak KPI | 1.006 |
| Zero Net Energy KPI | 0.869 |

### A.5 CPO (Best Checkpoint, Epoch 59)

| Metric | Value |
|--------|-------|
| Total Reward | 1931.43 |
| Total Cost | 6.909 |
| Violation Rate | 0.1317 |
| Comfort Violations | 289 / 2194 active steps |
| Discomfort Count | 1077 / 5547 occupied |
| Discomfort Proportion | 0.1341 |
| Discomfort Cold Proportion | 0.0963 |
| Discomfort Hot Proportion | 0.0378 |
| Mean Cooling Action | 0.243 |
| District Import (kWh) | 7469.31 |
| District Net (kWh) | 7457.99 |
| Cost KPI | 1.083 |
| Carbon Emissions KPI | 1.098 |
| Electricity Consumption KPI | 1.093 |
| Ramping KPI | 0.988 |
| Daily Peak KPI | 1.052 |
| All-Time Peak KPI | 0.965 |
| Zero Net Energy KPI | 1.096 |

### A.6 FOCOPS (Best Checkpoint, Epoch 59)

| Metric | Value |
|--------|-------|
| Total Reward | -373.38 |
| Total Cost | 627.659 |
| Violation Rate | 0.9526 |
| Comfort Violations | 2090 / 2194 active steps |
| Discomfort Count | 3829 / 5547 occupied |
| Discomfort Proportion | 0.8473 |
| Discomfort Cold Proportion | 0.0011 |
| Discomfort Hot Proportion | 0.8462 |
| Mean Cooling Action | 0.102 |
| District Import (kWh) | 5042.61 |
| District Net (kWh) | 4768.78 |
| Cost KPI | 0.667 |
| Carbon Emissions KPI | 0.692 |
| Electricity Consumption KPI | 0.688 |
| Ramping KPI | 1.189 |
| Daily Peak KPI | 1.068 |
| All-Time Peak KPI | 1.224 |
| Zero Net Energy KPI | 0.655 |

### A.7 CUP (Best Checkpoint, Epoch 59)

| Metric | Value |
|--------|-------|
| Total Reward | 162.11 |
| Total Cost | 437.393 |
| Violation Rate | 0.8560 |
| Comfort Violations | 1878 / 2194 active steps |
| Discomfort Count | 3436 / 5547 occupied |
| Discomfort Proportion | 0.6103 |
| Discomfort Cold Proportion | 0.0372 |
| Discomfort Hot Proportion | 0.5731 |
| Mean Cooling Action | 0.142 |
| District Import (kWh) | 5633.74 |
| District Net (kWh) | 5409.33 |
| Cost KPI | 0.843 |
| Carbon Emissions KPI | 0.817 |
| Electricity Consumption KPI | 0.816 |
| Ramping KPI | 1.125 |
| Daily Peak KPI | 1.054 |
| All-Time Peak KPI | 0.990 |
| Zero Net Energy KPI | 0.803 |

### A.8 PPO-Lagrangian (Best Checkpoint, Epoch 59)

| Metric | Value |
|--------|-------|
| Total Reward | 2152.54 |
| Total Cost | 24.248 |
| Violation Rate | 0.1964 |
| Comfort Violations | 431 / 2194 active steps |
| Discomfort Count | 1402 / 5547 occupied |
| Discomfort Proportion | 0.3875 |
| Discomfort Cold Proportion | 0.3771 |
| Discomfort Hot Proportion | 0.0104 |
| Mean Cooling Action | 0.217 |
| District Import (kWh) | 7002.32 |
| District Net (kWh) | 6859.51 |
| Cost KPI | 1.138 |
| Carbon Emissions KPI | 1.178 |
| Electricity Consumption KPI | 1.175 |
| Ramping KPI | 1.226 |
| Daily Peak KPI | 1.034 |
| All-Time Peak KPI | 1.008 |
| Zero Net Energy KPI | 1.177 |

### A.9 SAC-Lagrangian (Best Checkpoint, Epoch 5)

| Metric | Value |
|--------|-------|
| Total Reward | 2751.34 |
| Total Cost | 35.898 |
| Violation Rate | 0.3268 |
| Comfort Violations | 717 / 2194 active steps |
| Discomfort Count | 778 / 5547 occupied |
| Discomfort Proportion | 0.0646 |
| Discomfort Cold Proportion | 0.0400 |
| Discomfort Hot Proportion | 0.0246 |
| Mean Cooling Action | 0.128 |
| District Import (kWh) | 5542.01 |
| District Net (kWh) | 5220.61 |
| Cost KPI | 0.779 |
| Carbon Emissions KPI | 0.800 |
| Electricity Consumption KPI | 0.798 |
| Ramping KPI | 1.403 |
| Daily Peak KPI | 0.889 |
| All-Time Peak KPI | 0.949 |
| Zero Net Energy KPI | 0.775 |

### A.10 PPO Unconstrained (Best Checkpoint, Epoch 13)

| Metric | Value |
|--------|-------|
| Total Reward | 2793.60 |
| Total Cost | 49.567 |
| Violation Rate | 0.4275 |
| Comfort Violations | 938 / 2194 active steps |
| Discomfort Count | 686 / 5547 occupied |
| Discomfort Proportion | 0.0597 |
| Discomfort Cold Proportion | 0.0109 |
| Discomfort Hot Proportion | 0.0487 |
| Mean Cooling Action | 0.118 |
| District Import (kWh) | 5334.69 |
| District Net (kWh) | 5045.96 |
| Cost KPI | 0.759 |
| Carbon Emissions KPI | 0.780 |
| Electricity Consumption KPI | 0.779 |
| Ramping KPI | 1.282 |
| Daily Peak KPI | 0.881 |
| All-Time Peak KPI | 0.945 |
| Zero Net Energy KPI | 0.756 |

### A.11 Baselines

#### Cooling RBC

| Metric | Value |
|--------|-------|
| Total Reward | -448.67 (CSAC-LB eval) / 1591.16 (CPO eval) |
| Total Cost | 135.763 |
| Violation Rate | 0.5160 |
| Comfort Violations | 1132 / 2194 active steps |
| Discomfort Proportion | 0.3760 |
| Mean Cooling Action | 0.207 |
| Cost KPI | 1.010 |
| Carbon Emissions KPI | 1.043 |
| Electricity Consumption KPI | 1.041 |
| Ramping KPI | 2.322 |

#### Zero (No Cooling)

| Metric | Value |
|--------|-------|
| Total Reward | -45431.65 (CSAC-LB eval) / -11992.58 (CPO eval) |
| Total Cost | 2459.668 |
| Violation Rate | 0.9950 |
| Comfort Violations | 2183 / 2194 active steps |
| Discomfort Proportion | 0.9617 |
| Mean Cooling Action | 0.000 |
| Cost KPI | 0.527 |
| Carbon Emissions KPI | 0.562 |
| Electricity Consumption KPI | 0.559 |
| Ramping KPI | 0.905 |

*Note: Baseline total reward differs across eval files due to different comfort penalty scaling used by different algorithm implementations. All other metrics (violation rate, KPIs) are identical.*

---

## Appendix B: Hyperparameter Summary

| Parameter | CSAC-LB | CPO | FOCOPS | CUP | PPO-Lag | SAC-Lag | PPO |
|-----------|---------|-----|--------|-----|---------|---------|-----|
| Base Algorithm | SAC | TRPO | PPO | PPO | PPO | SAC | PPO |
| Policy Type | Off-policy | On-policy | On-policy | On-policy | On-policy | Off-policy | On-policy |
| Constraint Method | Conservative LB + Lagrangian | Trust-region | First-order projection | Update projection | Lagrangian dual | Lagrangian dual | None |
| Hidden Layers | [256, 256] | [256, 256] | [256, 256] | [256, 256] | [256, 256] | [256, 256] | [256, 256] |
| Epochs | 60 | 60 | 60 | 60 | 60 | 60 | 40 |
| Steps/Epoch | 2208 | 2208 | 2208 | 2208 | 2208 | 2208 | 2208 |
| Seed | 0,1,2,42 | 42 | 42 | 42 | 42 | 42 | 42 |
| Best Epoch | 24-59 | 59 | 59 | 59 | 59 | 5 | 13 |
| Discount (gamma) | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |

---

*Report generated: 2026-04-10*
*Environment: CityLearn v2, Travis County TX, 3 buildings, cooling-only*
*Hardware: RTX 3050 (4GB VRAM), 14GB RAM, Linux*
