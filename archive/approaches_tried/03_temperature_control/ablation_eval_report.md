# Softmax vs GradS Ablation Study: Multi-Constraint Gradient Aggregation for Safe V2G Control

**Study ID:** R27a Ablation (Softmax vs GradS)
**Date:** 2026-04-15
**Checkpoint:** Epoch 40
**Domain:** Safe Reinforcement Learning for Vehicle-to-Grid (V2G) Energy Management

---

## 1. Introduction and Motivation

Constrained Markov Decision Processes (CMDPs) with multiple constraints require a mechanism to aggregate per-constraint gradient signals into a single policy update direction. Two paradigms dominate the literature:

1. **Softmax aggregation** --- temperature-weighted averaging of all constraint gradients, producing a blended update that simultaneously addresses every active constraint.
2. **Gradient Surgery (GradS)** --- cosine-similarity-based conflict detection followed by single-constraint selection per update step, inspired by Yu et al. (2020).

This ablation isolates the gradient aggregation method as the sole experimental variable. Both models share identical architectures, reward shaping, PID Lagrangian parameters, and training hyperparameters. The study answers a specific question: **which aggregation strategy produces safer, higher-return policies when constraints exhibit action-space conflicts?**

The V2G smart-grid domain provides an ideal testbed because its constraints are physically coupled --- charging an electric vehicle satisfies EV departure requirements (C0) but increases building power consumption (C3). This coupling creates a structured conflict that gradient aggregation must navigate.

---

## 2. Experimental Setup

### 2.1 Environment

- **Domain:** CityLearn with 5 buildings, each equipped with EV chargers, battery storage, and solar PV.
- **Horizon:** 8759 timesteps per episode (one full year at hourly resolution).
- **Action space:** 9-dimensional continuous (5 EV chargers + 4 batteries), bounded in [-1, 1].
- **Schema:** `data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json`.

### 2.2 Shared Configuration

| Parameter | Value |
|---|---|
| Actor architecture | MLP [256, 256], tanh activation |
| Critic (reward) | STEMS encoder (GCN + 2-layer Temporal Transformer, 175K params) |
| Critic (cost) | 6 independent MLPs |
| Training epochs | 40 |
| Random seed | 42 |
| PID Lagrangian upper bound | 35.0 |
| Saute safety wrapper | OFF (Lagrangian handles all constraints) |

### 2.3 Reward Weights

| Component | Symbol | Weight |
|---|---|---|
| Ramping penalty | r_ramp | 0.3 |
| Renewable utilization | r_ren | 0.2 |
| EV charge urgency | r_ev | 1.0 |
| EV anti-discharge guard | r_ev_guard | 1.0 |
| SoC boundary barrier | r_barrier | 0.5 |
| Headroom-gated EV signal | r_ev_smart | 1.5 |
| V2G context reward | r_v2g_ctx | 3.0 |
| Grid mildness | r_grid_mild | 0.3 |
| EV slack arbitrage | r_ev_slack_arb | 2.0 |

### 2.4 PID Lagrangian Parameters

| Constraint | Kp | Ki | Limit (final) | Curriculum |
|---|---|---|---|---|
| C0 (EV departure SoC deficit) | 3.0 | 0.05 | 20 | 200 to 20, epochs 0--20 |
| C1 (dense EV corridor, Saute) | 1.0 | 0.03 | 50 | None (Saute OFF) |
| C2 (battery SoC band) | default | default | 1500 | None |
| C3 (building power capacity) | 0.5 | 0.05 | 5000 | None |
| C4 (grid power capacity) | 0.3 | 0.03 | 8000 | None |

### 2.5 Ablation Variable

- **Softmax:** Temperature tau = 1.0. All constraint gradients are combined via softmax-weighted average at every update step. Every constraint influences the policy proportionally to its current violation magnitude and Lagrangian multiplier.
- **GradS:** Cosine similarity is computed between all pairs of constraint gradients. Conflicting pairs (negative cosine similarity) trigger gradient projection. A single constraint is selected per update step via uniform sampling among active constraints.

---

## 3. Results

### 3.1 Constraint Violations (Evaluation Rollout, Epoch 40)

| Constraint | Softmax Total | GradS Total | Softmax % Violated | GradS % Violated | GradS / Softmax Ratio |
|---|---|---|---|---|---|
| C0 (EV departure SoC deficit) | 17.1 | 206.1 | 0.1% | 2.6% | 12.0x |
| C2 (battery SoC band) | 7441.8 | 44419.2 | 28.1% | 92.4% | 6.0x |
| C3 (building power capacity) | 19557.0 | 53524.6 | 30.2% | 47.8% | 2.7x |
| C4 (grid power capacity) | 19193.8 | 43548.1 | 21.1% | 26.1% | 2.3x |

Softmax achieves lower violation totals on every constraint. The improvement is most dramatic on C0, where Softmax reduces total EV departure SoC deficit by 12x compared to GradS. C2 (battery SoC band) shows a 6x improvement, indicating Softmax learns battery management far more effectively.

### 3.2 Episode Returns (Evaluation)

| Model | EpRet (eval) |
|---|---|
| Softmax | -4178.4 |
| GradS | -14238.0 |
| Difference | +10059.6 (Softmax 2.4x better) |

### 3.3 Training Curves (40 Epochs)

| Model | EpRet (epoch 1) | EpRet (epoch 40) | Trend |
|---|---|---|---|
| Softmax | ~-17000 | ~+1544 | Steady improvement throughout training |
| GradS | ~-16900 | ~-16770 | Flat; never improved meaningfully |

GradS episode returns remain effectively stationary across all 40 epochs. The policy does not learn to improve reward while satisfying constraints. Softmax, by contrast, recovers approximately 18500 reward units over the training run.

### 3.4 CityLearn KPIs (Normalized, Lower = Better Except Load Factor)

| KPI | Softmax | GradS | Better |
|---|---|---|---|
| Electricity Consumption | 2.1514 | 2.8690 | Softmax |
| Carbon Emissions | 2.2715 | 3.0493 | Softmax |
| Electricity Cost | 2.1341 | 3.0528 | Softmax |
| Ramping | 1.7257 | 2.5167 | Softmax |
| 1 - Load Factor (daily) | 0.7683 | 0.8620 | Softmax |
| Daily Peak | 2.5964 | 2.7225 | Softmax |
| All-time Peak | 1.1350 | 1.4028 | Softmax |

Softmax dominates GradS on every CityLearn KPI. Note that both models produce KPI ratios above 1.0 for most metrics (meaning the controlled building consumes more than the uncontrolled baseline), which reflects the additional energy demand from EV charging --- a demand absent in the baseline. The meaningful comparison is between the two controlled policies, where Softmax achieves 25--30% lower electricity cost and carbon emissions.

---

## 4. Reward Component Decomposition

Per-step reward means, averaged over the last 5 training epochs:

| Component | Weight | Softmax | GradS | Difference | Analysis |
|---|---|---|---|---|---|
| r_ramp | 0.3 | -0.2417 | -0.2501 | +0.0084 | Similar; ramping penalty comparable |
| r_ren | 0.2 | +0.0789 | +0.0862 | -0.0073 | Similar; renewable utilization comparable |
| r_ev | 1.0 | -0.0439 | -0.6207 | +0.5768 | **Softmax 14x better** |
| r_ev_guard | 1.0 | -0.0569 | -1.3706 | +1.3136 | **Softmax 24x better** |
| r_barrier | 0.5 | -0.3538 | -0.4294 | +0.0755 | Similar |
| r_ev_smart | 1.5 | +0.2968 | +0.3410 | -0.0442 | Similar |
| r_v2g_ctx | 3.0 | +1.1451 | +0.6824 | +0.4627 | **Softmax 68% higher** |
| r_grid_mild | 0.3 | -0.2006 | -0.2555 | +0.0549 | Softmax moderately better |
| r_ev_slack_arb | 2.0 | +0.6751 | +0.5883 | +0.0868 | Similar |

**Key finding:** The dominant reward difference between Softmax and GradS lies in two EV-related components:

- **r_ev** (charge urgency): Softmax achieves -0.04 per step vs GradS at -0.62. GradS fails to charge EVs before departure, incurring 14x the penalty.
- **r_ev_guard** (anti-discharge): Softmax achieves -0.06 per step vs GradS at -1.37. GradS actively discharges EVs when it should not, incurring 24x the penalty.

Grid-related and battery-related rewards (r_ramp, r_ren, r_barrier, r_ev_smart, r_ev_slack_arb) are comparable between the two models. The failure mode is specific to EV management under constraint conflict.

---

## 5. Lagrangian Multiplier Analysis

### 5.1 Final and Peak Multiplier Values

| Constraint | Softmax lambda_final | GradS lambda_final | Softmax lambda_max | GradS lambda_max |
|---|---|---|---|---|
| C0 (EV departure) | 0.000 | 3.225 | 9.905 | 13.306 |
| C2 (battery SoC) | 4.675 | 11.475 | 4.675 | 11.475 |
| C3 (building power) | 16.286 | 24.143 | 16.286 | 24.143 |
| C4 (grid power) | 3.247 | 4.868 | 3.247 | 4.868 |

### 5.2 Interpretation

GradS produces universally higher Lagrangian multipliers. This is consistent with the policy failing to reduce constraint costs: the PID controller continues increasing lambda because violations persist. The Softmax lambda values converge to lower levels because the policy actually reduces violations, relieving pressure on the Lagrangian.

The most revealing contrast is C0: Softmax drives lambda_0 to 0.000 (no active Lagrangian penalty needed at convergence), while GradS retains lambda_0 = 3.225 with violations still present. The Lagrangian is still actively trying to enforce C0 in GradS but failing.

---

## 6. Attribution Analysis: Reward Shaping vs Lagrangian Enforcement

A central question for CMDP design is whether constraint satisfaction is driven by reward shaping (intrinsic motivation) or Lagrangian enforcement (extrinsic penalty). The lambda and cost trajectories allow attribution.

### 6.1 C0 (EV Departure SoC Deficit)

**Softmax --- MIXED (Reward-Sustained) Attribution:**

- Cost trajectory: 1166.2 (epoch 1) to 32.8 (epoch 40), a 97.2% reduction.
- Lambda trajectory: peaked at 9.905 in early training, then declined to 0.000 by epoch 40.
- Mechanism: The Lagrangian provided initial gradient pressure in early epochs, pushing the policy toward EV charging behavior. Once reward shaping components (r_ev, r_ev_smart, r_ev_guard) reinforced this behavior, the policy maintained C0 satisfaction without Lagrangian help. Lambda decayed to zero because cost fell below the limit.
- This represents a successful "handoff" from Lagrangian enforcement to reward-sustained behavior.

**GradS --- FAILED Attribution:**

- Cost trajectory: 1148.4 (epoch 1) to 174.2 (epoch 40), only an 84.8% reduction. Final cost is 5.3x higher than Softmax.
- Lambda trajectory: oscillated throughout training (max 13.306, final 3.225), never converging.
- Mechanism: GradS selects one constraint per update step. When C0 is selected, the policy receives gradient signal to charge EVs. When C3 is selected (which is frequent, given C3's high cost), the policy receives gradient signal to reduce building power --- which conflicts with EV charging. The result is oscillation: C0 cost decreases when C0 is selected, then rebounds when C3 is selected. Neither constraint achieves stable satisfaction.

### 6.2 C3 (Building Power Capacity)

**Both Models --- LAGRANGIAN-DRIVEN Attribution:**

- No direct reward component targets building power reduction. C3 satisfaction depends entirely on the Lagrangian multiplier.
- Softmax: Cost 64482 (epoch 1) to 26184 (epoch 40). Lambda_3 = 16.286, actively penalizing.
- GradS: Cost substantially higher. Lambda_3 = 24.143, approaching the upper bound of 35.0.
- In both models, the Lagrangian is the sole driver of C3 cost reduction. Softmax achieves better C3 outcomes because simultaneous gradient aggregation allows the policy to find temporal compromises (charge EVs during low-demand hours).

### 6.3 C4 (Grid Power Capacity)

- Managed by Lagrangian in both models.
- Softmax lambda_4 = 3.247, GradS lambda_4 = 4.868.
- C4 is less contentious because grid power constraints are less tightly coupled to EV actions.

---

## 7. Failure Mechanism Analysis: Why GradS Fails

Four interacting failure mechanisms explain GradS's poor performance.

### 7.1 Gradient Magnitude Imbalance

**CostRewardGradRatio** (from TensorBoard): Average 76x, peaked at 130x in late training. Cost gradient norms averaged approximately 41; reward gradient norms averaged approximately 4.8.

The policy gradient is dominated by cost reduction signals. Reward learning is starved because the cost gradient magnitude overwhelms the reward gradient by nearly two orders of magnitude. Under Softmax, all gradients are blended with temperature-controlled weights, preventing any single signal from monopolizing the update. Under GradS, the selected constraint's gradient is applied directly, and with a 76x magnitude advantage, the cost gradient dictates the update direction.

### 7.2 Single-Constraint Selection Oscillation

GradS selects one constraint per update step. When multiple constraints have conflicting optimal actions, this produces a see-saw dynamic:

1. **Step t:** C0 selected. Policy gradient pushes toward EV charging. C0 cost decreases.
2. **Step t+1:** C3 selected. Policy gradient pushes toward reducing building power. EV charging decreases. C0 cost rebounds.
3. **Step t+2:** C0 selected again (cost increased). Cycle repeats.

The policy oscillates rather than converging to a compromise that satisfies both constraints simultaneously.

### 7.3 Invisible Action-Space Conflict (CosSim Masking)

GradS uses cosine similarity between constraint gradients in parameter space to detect conflicts. Two key measurements:

- **CosSim(C0, C3) = +0.44:** Positive, suggesting alignment. GradS does not project or correct for conflict.
- **CosSim(C3, C4) = +0.83:** Strongly positive, indicating near-identical gradient directions.

The C0-C3 cosine similarity is positive because the 130K shared representation parameters dominate the inner product. These shared parameters encode general features (time-of-day, building state) that both constraints benefit from. The conflict exists in the 9-dimensional action head, but this signal is drowned out by the high-dimensional shared representation.

**Proposition 1 (CosSim Masking).** *When the shared representation dimensionality D_shared far exceeds the action-head dimensionality D_action (here, 130K vs 9), cosine similarity in full parameter space can show positive alignment even when constraints impose directly opposing requirements in action space. GradS's conflict detection fails under this condition.*

### 7.4 Physical Constraint Coupling

The C0-C3 conflict has a clear physical interpretation:

- **C0** (EV departure SoC deficit): Requires the agent to **charge** EVs before their departure time.
- **C3** (building power capacity): Requires the agent to **reduce** total building electricity consumption.
- **Conflict:** Charging an EV increases building power consumption. The same action (positive EV charging) satisfies C0 but violates C3.
- **Resolution:** Charge EVs during **low-demand hours** (night, early morning) when the building has headroom under C3. Avoid charging during peak hours when C3 is binding.

This temporal compromise requires the policy to learn a coordinated strategy that considers both constraints simultaneously. Softmax aggregation provides gradient information from both constraints at every update step, enabling this coordination. GradS, by selecting a single constraint per step, prevents the policy from learning the temporal compromise.

---

## 8. Formal Claims

**Claim 1 (Softmax Dominance).** Under physically-coupled multi-constraint CMDPs where constraints impose conflicting action-space requirements, Softmax gradient aggregation achieves lower violation rates on all constraints simultaneously (12x on C0, 6x on C2, 2.7x on C3, 2.3x on C4) and 2.4x higher episode returns compared to GradS.

**Claim 2 (GradS Stagnation).** GradS fails to improve episode return over 40 epochs of training (EpRet: -16900 to -16770), while Softmax recovers from -17000 to +1544. The failure is attributable to single-constraint selection oscillation under action-space conflict, compounded by a 76x cost-to-reward gradient magnitude imbalance.

**Claim 3 (CosSim Masking).** Cosine similarity in full parameter space is an unreliable conflict detector when shared representation parameters outnumber action-head parameters by orders of magnitude. The measured CosSim(C0, C3) = +0.44 masked a genuine action-space conflict that produced 12x higher C0 violations under GradS.

**Claim 4 (Lagrangian-Reward Handoff).** Softmax enables a "handoff" pattern where the Lagrangian multiplier provides initial enforcement pressure, reward shaping sustains the learned behavior, and the multiplier decays to zero. This pattern was observed for C0 (lambda peaked at 9.905, converged to 0.000, cost reduced 97.2%). GradS does not achieve this handoff (lambda oscillated, final value 3.225, cost reduced only 84.8%).

---

## 9. Implications for Thesis

### 9.1 Contribution to Safe RL

This ablation provides empirical evidence that gradient aggregation strategy is a first-order design choice in multi-constraint safe RL, not a secondary implementation detail. The 12x constraint violation difference and 2.4x return difference from changing only the aggregation method exceed typical differences from reward weight tuning or architecture changes.

### 9.2 Limitations of Gradient Surgery in High-Dimensional Shared Representations

The CosSim masking phenomenon (Section 7.3) identifies a structural limitation of gradient surgery methods when applied to deep networks with large shared representations. This finding extends beyond V2G to any multi-constraint domain where constraints operate on a low-dimensional action space but share a high-dimensional feature extractor.

### 9.3 Domain-Specific Insight: Temporal Compromise in V2G

The C0-C3 conflict resolution via temporal load shifting (Section 7.4) demonstrates that V2G constraint satisfaction requires inter-temporal coordination. Aggregation methods that provide all constraint signals simultaneously (Softmax) enable this coordination; methods that alternate between constraints (GradS) do not.

### 9.4 Design Recommendation

For multi-constraint CMDPs with physically-coupled constraints, Softmax gradient aggregation with PID Lagrangian multipliers is strongly preferred over gradient surgery. The temperature parameter (tau = 1.0 in this study) provides a tunable knob for balancing constraint influence without the failure modes of single-constraint selection.

---

## 10. Figures

All figures are located in `docs/temperature_case_study/ablation_eval/`.

| Figure | Filename | Description |
|---|---|---|
| Fig. 1 | fig1 | Constraint violation comparison: 3-panel (total cost, violation frequency, severity) |
| Fig. 2 | fig2 | CityLearn KPI comparison: grouped bar chart |
| Fig. 3 | fig3 | Reward component decomposition: bar chart + EV reward trajectories over training |
| Fig. 4 | fig4 | Lagrangian multiplier and cost trajectories: 4 subplots (C0, C2, C3, C4) |
| Fig. 5 | fig5 | Reward vs Lagrangian attribution: 4 subplots (Softmax C0, GradS C0, Softmax C3, GradS C3) |
| Fig. 6 | fig6 | Per-step cost time-series overlay: 200-step window, both models |
| Fig. 7 | fig7 | EV action distribution comparison: per building, when EV connected |
| Fig. 8 | fig8 | Reward-cost Spearman correlation matrix: per model |
| Fig. 9 | fig9 | Lambda-cost phase portrait: scatter with epoch coloring and directional arrows |
| Fig. 10 | fig10 | Training curves comparison: EpRet, C0, C3, total cost |

---

## 11. Related Work

- **Gradient Surgery for Multi-Task Learning** (Yu et al., 2020): Introduced PCGrad, projecting conflicting task gradients onto the normal plane of each other. GradS in this study extends this idea to constraint gradients in CMDPs.
- **PID Lagrangian Methods** (Stooke et al., 2020): Responsive safety via PID control of Lagrangian multipliers. Both Softmax and GradS use PID Lagrangian in this study; the ablation isolates the gradient aggregation mechanism.
- **Safety Layer** (Dalal et al., 2018): Projects actions onto the constraint-satisfying set at execution time. Complementary to gradient aggregation; could be combined with Softmax in future work.
- **CoMOGA** (ICLR 2025): Conflict-averse gradient aggregation for constrained multi-objective RL. Addresses similar gradient conflict issues but uses a different resolution mechanism.
- **ATACOM** (Liu et al., 2024): Constraint manifold projection for safe RL. Operates in action space rather than gradient space, potentially avoiding the CosSim masking problem identified here.

---

## 12. Summary Table

| Metric | Softmax | GradS | Softmax Advantage |
|---|---|---|---|
| EpRet (eval) | -4178.4 | -14238.0 | 2.4x |
| EpRet (train, final) | +1544 | -16770 | -- |
| C0 total violation | 17.1 | 206.1 | 12.0x lower |
| C2 total violation | 7441.8 | 44419.2 | 6.0x lower |
| C3 total violation | 19557.0 | 53524.6 | 2.7x lower |
| C4 total violation | 19193.8 | 43548.1 | 2.3x lower |
| r_ev (per step) | -0.0439 | -0.6207 | 14x better |
| r_ev_guard (per step) | -0.0569 | -1.3706 | 24x better |
| Electricity Cost KPI | 2.1341 | 3.0528 | 30% lower |
| C0 lambda_final | 0.000 | 3.225 | Converged to zero |
| C3 lambda_final | 16.286 | 24.143 | 33% lower |
| CostRewardGradRatio | -- | 76x avg | GradS-specific pathology |

---

*Report generated for thesis integration. All numerical values are from evaluation rollouts at epoch 40 (constraint violations, returns, KPIs) or from TensorBoard training logs (reward components, Lagrangian multipliers, gradient ratios).*
