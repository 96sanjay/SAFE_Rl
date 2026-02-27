
📄 DETAILED EXPERIMENTAL PLAN DRAFT

Experimental Design for Safe Reinforcement Learning in Vehicle-to-Grid Energy Management
Student: Sanjay Sajeev
Supervisor: Professor Andreas Kassler
Date: January 14, 2026
Status: Draft for Thesis Integration

1. OVERVIEW
This experimental plan evaluates the effectiveness of Safe Reinforcement Learning (Safe RL) for managing energy in smart buildings with Vehicle-to-Grid (V2G) capabilities. The core hypothesis is that explicit safety constraints (via Constrained Markov Decision Processes and Lagrangian methods) produce more reliable and deployable policies than traditional reward-shaping approaches, even when both optimize similar objectives.
1.1 Motivation
The CityLearn Challenge environment uses a multi-objective reward function that includes terms for electricity consumption, carbon emissions, peak demand, ramping, and thermal comfort. However, preliminary experiments (Bill-35ep agent) demonstrated that agents trained with reward-only optimization can achieve excellent reward scores while violating critical operational requirements—specifically, failing to adequately charge electric vehicles. This finding motivates the need for explicit constraint enforcement rather than relying solely on weighted reward terms.
1.2 Research Gap
Existing work in building energy management predominantly uses reward shaping to discourage undesirable behaviors. However, reward shaping provides no formal guarantees that safety-critical requirements will be met. Safe RL, implemented via Constrained MDPs with Lagrangian multipliers, offers a principled alternative by treating safety requirements as hard constraints rather than soft penalties.

2. RESEARCH QUESTIONS
Primary Research Questions
RQ1: Safety vs Performance Trade-off
Does Safe RL with explicit constraints achieve adequate performance while guaranteeing safety, compared to unconstrained RL that optimizes only reward?

Hypothesis: Safe RL agents will satisfy all operational constraints (EV charging, peak demand, ramping) while achieving competitive but slightly lower CityLearn reward scores compared to unconstrained agents.
Rationale: Constraints limit the action space, potentially reducing optimization freedom, but ensure deployable policies.

RQ2: Safe RL vs Rule-Based Control
Do learned Safe RL policies outperform hand-crafted rule-based controllers in terms of energy efficiency metrics?

Hypothesis: Safe RL will achieve lower costs, better peak management, and improved renewable utilization compared to RBC, while maintaining equivalent safety.
Rationale: Learning from data should discover more sophisticated strategies than simple time-of-use rules.

RQ3: Cost of Safety
What is the performance penalty (in terms of CityLearn reward) for enforcing safety constraints?

Hypothesis: Safe RL incurs a 10-20% penalty in CityLearn reward compared to unconstrained RL, but this penalty is acceptable given the guarantee of constraint satisfaction.
Rationale: Based on preliminary data showing unconstrained-like behavior achieving -0.22 reward while violating constraints vs RBC achieving -1.95 with no violations.

Secondary Research Questions
RQ4: Action-Based EV Classification (V3 vs V2)
Does action-based deficit classification (V3) improve learning efficiency and final performance compared to electricity-based classification (V2)?

Hypothesis: V3 produces fairer cost signals and faster convergence because it only penalizes agent-controllable deficits.
Rationale: Penalizing unavoidable deficits (V2) adds noise to the cost signal, hindering learning.

RQ5: Constraint Contribution Analysis
How does each individual constraint (EV, Peak, Ramp) contribute to overall safety and performance?

Hypothesis: All three constraints are necessary; removing any one leads to operational violations.
Rationale: Ablation study will quantify the importance of each constraint.

RQ6: Lagrangian Multiplier Sensitivity
How does the initial Lagrangian multiplier value affect training convergence and final policy quality?

Hypothesis: Stronger initial lambda (35-50) leads to faster constraint satisfaction but may be more conservative; weaker lambda (<10) fails to enforce constraints adequately.
Rationale: Bill-35ep data showed lambda=10 was insufficient.


3. EXPERIMENTAL DESIGN
3.1 Environment Configuration
Base Environment: CityLearn 2.0
Dataset: citylearn_challenge_2022_phase_all_plus_evs
Schema: 26-action space (17 batteries, 8 EV chargers, 1 washing machine)
Episode Length: 8760 timesteps (1 year, hourly resolution)
Buildings: 17 buildings with solar PV, battery storage
EV Chargers: 8 chargers distributed across buildings
Safety Wrapper: CityLearnSafetyEnvV3

Action-based EV deficit classification (V3)
Three operational safety constraints:

EV Constraint: Vehicles must reach target SOC before departure
Peak Constraint: District import must not exceed 96.10 kW (97th percentile of RBC baseline)
Ramp Constraint: Change in district net signal must not exceed TBD kW/hour (97th percentile of RBC baseline)



3.2 CMDP Cost Function
The total cost at each timestep is defined as:
cost(t) = cost_EV(t) + cost_peak(t) + cost_ramp(t)

where:
  cost_EV(t) = w_EV × controllable_deficit(t)    [V3 classification]
  cost_peak(t) = w_peak × max(0, import(t) - threshold_peak)
  cost_ramp(t) = w_ramp × max(0, |Δ_import(t)| - threshold_ramp)
Cost Weights:

w_EV = 1.0 (scaled by CITYLEARN_EV_COST_SCALE environment variable)
w_peak = 0.05
w_ramp = 0.05

Cost Limits:

Derived from RBC baseline performance
Target: Allow 3-5% violation rate for peak/ramp (similar to RBC)
Estimated total budget: 250-300 per episode

3.3 Reward Functions
Reward Function 1: Bill-Based (Primary)
reward(t) = -[(import(t) × price(t)) - (export_factor × export(t) × price(t))]

where:
  import(t) = max(0, net_consumption(t))
  export(t) = max(0, -net_consumption(t))
  export_factor = 0.7 (feed-in tariff)
Reward Function 2: Cost-Only (Alternative)
reward(t) = -(import(t) × price(t))
Reward Function 3: CityLearn Multi-Objective (Baseline Comparison)
reward(t) = Σ w_i × KPI_i
where KPIs include: electricity, carbon, cost, peak, ramping, discomfort

4. EXPERIMENTS
4.1 Core Experiments (Essential)
Experiment 1: Rule-Based Controller (RBC) Baseline
Status: ✅ Completed (January 8, 2026)
Configuration:

Battery: Time-of-use scheduling (charge 10-16h, discharge 17-21h at ±0.5 rate)
EV: Greedy charging (action=1.0 when connected)
Washing machine: No control (action=0.0)

Results (Already Obtained):
Total CMDP Cost: 149.57
  - EV Cost: 0.00 kWh (zero controllable deficit)
  - Peak Cost: 149.57 (263 violations, 3.0% of timesteps)
  
Energy Metrics:
  - Grid Import: 207,124 kWh
  - Grid Export: 77,772 kWh
  - Export Ratio: 37.5%
  
Economic:
  - Total Bill: $25,868
  
CityLearn KPIs:
  - Electricity Consumption: 2.7816
  - Carbon Emissions: 2.8587
  - Cost: 2.8184
  - Daily Peak Average: 3.6060
  - Ramping Average: (TBD after ramp cost added)
Purpose: Establishes baseline performance achievable with simple rules. Shows that perfect safety is possible but leaves room for optimization.

Experiment 2: Vanilla PPO with CityLearn Multi-Objective Reward
Status: ⏳ Planned
Purpose: Demonstrate that reward-only optimization, even with terms for peaks and ramping, cannot guarantee constraint satisfaction.
Configuration:
yamlalgorithm: PPO (unconstrained, NOT PPO-Lagrangian)
reward_function: CityLearn default multi-objective
  weights: [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal weighting
  components: [electricity, carbon, cost, peak, ramping]
  
constraints: None (cost_limit: null)
cost_computation: Disabled or set to 0.0

training:
  total_steps: 876,000 (100 epochs × 8760 steps)
  steps_per_epoch: 8759
  batch_size: 128
  actor_lr: 0.0003
  critic_lr: 0.001
  gamma: 0.99
  hidden_sizes: [64, 64]
  linear_lr_decay: false
Expected Results:

✅ Excellent CityLearn reward (predicted: -0.3 to -0.5)
✅ Low peak and ramping KPIs (agent optimizes these)
❌ EV charging violations (agent may skip charging to save energy)
❌ High CMDP cost if measured (predicted: 300-500+)
❌ Policy unusable in practice despite good metrics

Success Criteria:

Achieves CityLearn reward better than RBC (-1.95)
Violates at least one safety constraint significantly
Demonstrates the insufficiency of reward shaping for safety

Analysis Focus:

Quantify EV deficit (compare controllable vs uncontrollable)
Measure peak violations (count and magnitude)
Compare CityLearn KPIs to RBC
Identify trade-offs agent made to optimize reward


Experiment 3: Safe PPO-Lagrangian (Primary Contribution)
Status: ⏳ Planned
Purpose: Demonstrate that Safe RL with explicit constraints can satisfy all operational requirements while achieving competitive performance.
Configuration:
yamlalgorithm: PPO-Lagrangian
reward_function: Bill-based
  export_factor: 0.7
  reward_scale: 1.0

constraints:
  cost_limit: 280  # Derived from RBC + buffer
  lagrangian_multiplier_init: 50  # Strong enforcement
  lambda_lr: 0.05  # Faster lambda adaptation
  use_lagrangian_penalty: true

training:
  total_steps: 876,000 (100 epochs)
  steps_per_epoch: 8759
  batch_size: 128
  actor_lr: 0.0003
  critic_lr: 0.001
  gamma: 0.99
  hidden_sizes: [64, 64]
  linear_lr_decay: false
  
environment:
  CITYLEARN_EV_COST_SCALE: 1.0
  peak_threshold: 96.10
  peak_weight: 0.05
  ramp_threshold: TBD (from calibration)
  ramp_weight: 0.05
Expected Results:

✅ Total CMDP cost under budget (< 280)
✅ EV charging: >95% of vehicles charged adequately
✅ Peak violations: Under budget (similar to RBC ~3%)
✅ Ramp violations: Under budget (~3-5%)
⚖️ CityLearn reward: Competitive but lower than vanilla PPO (predicted: -0.6 to -0.8)
✅ Policy deployable in practice

Success Criteria:

Constraint satisfaction: cost < budget by epoch 80
Lambda convergence: stable by epoch 70
Better CityLearn metrics than RBC
No critical safety violations (EV charging >95%)

Analysis Focus:

Convergence trajectory (cost and lambda over epochs)
Final policy behavior (action distributions, charging patterns)
Trade-off quantification vs vanilla PPO
Per-constraint breakdown (which constraints were hardest to satisfy)


4.2 Ablation Studies (Secondary)
Experiment 4: Constraint Ablation
Purpose: Understand the contribution of each individual constraint.
Variants:

4a: Safe RL with EV constraint only
4b: Safe RL with Peak constraint only
4c: Safe RL with Ramp constraint only
4d: Safe RL with EV + Peak (no ramp)
4e: Safe RL with EV + Ramp (no peak)
4f: Safe RL with Peak + Ramp (no EV)

Configuration: Same as Experiment 3, but enable/disable specific cost components.
Expected Results:

EV-only: Good charging, but may create peaks/ramps
Peak-only: Manages peaks but may undercharge EVs
Combined: Best overall performance

Analysis: Quantify violation rates for disabled constraints.

Experiment 5: V3 vs V2 Classification Comparison
Purpose: Validate V3 action-based classification improvement over V2 electricity-based.
Variants:

5a: Safe RL with V3 (action-based) classification
5b: Safe RL with V2 (electricity-based) classification

Configuration: Identical except for EV cost computation method.
Expected Results:

V3: Faster convergence, lower final cost (fairer signal)
V2: Slower convergence, noisier cost signal

Metrics:

Convergence speed (epochs to reach budget)
Final policy cost
EV charging behavior (controllable vs uncontrollable deficit)


Experiment 6: Lambda Sensitivity Analysis
Purpose: Understand the effect of initial Lagrangian multiplier on training.
Variants:

6a: lambda_init = 10 (weak, similar to Bill-35ep)
6b: lambda_init = 30 (moderate)
6c: lambda_init = 50 (strong, primary configuration)
6d: lambda_init = 100 (very strong)

Expected Results:

lambda=10: Fails to enforce constraints (like Bill-35ep)
lambda=30: Slower convergence but eventually succeeds
lambda=50: Fast convergence, good balance
lambda=100: Very conservative, may over-penalize

Metrics:

Constraint satisfaction rate over training
Final lambda value at convergence
Training stability (variance in cost)


Experiment 7: Reward Function Comparison
Purpose: Evaluate different reward formulations under Safe RL.
Variants:

7a: Bill-based reward (primary)
7b: Cost-only reward (simpler)
7c: CityLearn multi-objective reward (with Safe RL constraints)

Expected Results:

All should satisfy constraints (due to Safe RL)
Bill-based: Best economic performance
Cost-only: Simpler but may miss export optimization
Multi-objective: Most complex, may be harder to tune


4.3 Stress Tests (Optional/Future Work)
Experiment 8: Tighter Budgets
Purpose: Test robustness of Safe RL to stricter safety requirements.
Variants:

8a: cost_limit = 200 (30% tighter than baseline)
8b: cost_limit = 150 (50% tighter)
8c: Peak threshold = 90% of RBC max (instead of 97%)

Expected Results:

Agent adapts to tighter constraints
May require higher lambda or more training
Performance penalty increases with tightness


Experiment 9: Different Scenarios
Purpose: Evaluate generalization to other building configurations.
Variants:

Different weather years
Different pricing schemes
Different EV arrival patterns

Status: Out of scope for current thesis (mention as future work)

5. EVALUATION METRICS
5.1 Safety Metrics (Primary)
Constraint Satisfaction:

Total CMDP cost per episode
Cost relative to budget (% over/under)
Violation timesteps (count and percentage)
Per-constraint breakdown:

EV controllable deficit (kWh)
Peak violation magnitude (kW over threshold)
Ramp violation magnitude (kW change over threshold)



EV Reliability:

Percentage of EVs adequately charged (>= target SOC)
Mean SOC at departure
Controllable vs uncontrollable deficit ratio (V3 validation)

Grid Operational Safety:

Peak demand (max district import)
Peak violation frequency
Ramp rate (max hourly change)
Ramp violation frequency

5.2 Performance Metrics (Secondary)
CityLearn Official KPIs:

Electricity Consumption Total (normalized, lower = better)
Carbon Emissions Total (normalized, lower = better)
Cost Total (normalized, lower = better)
Daily Peak Average (normalized, lower = better)
All-Time Peak Average (normalized, lower = better)
Ramping Average (normalized, lower = better)
Discomfort Proportion (not applicable in dataset)
Zero Net Energy (closer to 0 = better)
Overall CityLearn Reward (aggregate of above)

Energy Management Efficiency:

Total grid import (kWh)
Total grid export (kWh)
Export ratio (export / generation)
Solar utilization (generation used / total generation)
Battery cycling (mean SOC standard deviation)

Economic Performance:

Total electricity cost ($)
Total bill (cost - export revenue, $)
Average electricity price paid ($/kWh)

5.3 Learning Metrics
Convergence:

Episodes to constraint satisfaction (cost < budget)
Lagrangian multiplier trajectory
Cost trajectory over training
Reward trajectory over training

Stability:

Variance in cost (last 10 epochs)
Variance in reward (last 10 epochs)
Final lambda standard deviation

Sample Efficiency:

Total environment interactions required
Training wall-clock time


6. EXPECTED OUTCOMES
6.1 Quantitative Predictions
Based on preliminary data (RBC, Bill-35ep) and Safe RL theory:
MetricRBC BaselineVanilla PPOSafe PPO-LagSafety MetricsTotal CMDP Cost149.57400-600<280 ✅EV Charging Rate100% ✅30-50% ❌>95% ✅Peak Violations (%)3.0%15-25%<5% ✅Ramp Violations (%)TBD (~3%)10-20%<5% ✅Performance MetricsCityLearn Reward-1.95-0.3 to -0.5 ✅-0.6 to -0.8 ⚖️Electricity KPI2.782.50-2.602.65-2.75Peak KPI3.611.20-1.40 ✅1.80-2.20Cost Total$25,868$28,000-32,000$26,000-28,000Learning MetricsConvergence (epochs)N/AN/A60-80Final LambdaN/AN/A40-60
Key Takeaways:

Vanilla PPO achieves best CityLearn metrics but violates safety
Safe RL achieves adequate CityLearn metrics with guaranteed safety
Performance penalty for safety: ~20-30% in CityLearn reward
All approaches outperform RBC in energy efficiency

6.2 Qualitative Predictions
Vanilla PPO Behavior:

Will learn to minimize total energy consumption aggressively
May skip EV charging during high-price periods
May create sharp demand spikes when charging (no ramp constraint)
Optimizes CityLearn metrics but at the expense of operational requirements

Safe PPO-Lag Behavior:

Will spread charging over multiple timesteps (smooth ramp)
Will prioritize EV charging even during high-price periods (constraint)
May be more conservative in battery utilization (avoid peaks)
Finds optimal policy within safety bounds

Learning Dynamics:

Lambda will increase rapidly in early epochs (high violations)
Lambda will stabilize once agent learns to satisfy constraints
Cost will decrease monotonically as lambda enforces constraints
Agent will explore aggressive strategies early, then become conservative


7. ANALYSIS METHODOLOGY
7.1 Statistical Analysis
Significance Testing:

Compare final episode costs using paired t-tests
Compare CityLearn KPIs using Mann-Whitney U tests
Report p-values and effect sizes (Cohen's d)

Confidence Intervals:

Report mean ± standard deviation for all metrics
Use bootstrapping for non-normal distributions

Multiple Seeds:

Run each experiment with 3 random seeds
Report mean and variance across seeds

7.2 Visualization Plan
Training Curves:

Cost vs epoch (all experiments overlaid)
Lambda vs epoch (Safe RL only)
Reward vs epoch (all experiments)
Violation rate vs epoch

Policy Behavior:

Action distribution histograms (battery, EV)
Hourly charging patterns (24-hour cycle)
Peak import by hour of day
Ramp magnitude distribution

Comparative Analysis:

Radar charts for CityLearn KPIs
Bar charts for constraint satisfaction
Scatter plots (CityLearn reward vs CMDP cost)
Pareto frontier (safety vs performance)

Case Studies:

Detailed timestep-by-timestep analysis of:

Representative day (typical behavior)
Worst violation day (stress scenario)
Best performance day (optimal behavior)



7.3 Ablation Analysis
Per-Constraint Contribution:

Remove each constraint individually
Measure resulting violations
Quantify performance improvement from removing constraint

Sensitivity Analysis:

Vary cost weights (w_EV, w_peak, w_ramp)
Vary thresholds (peak_threshold, ramp_threshold)
Measure robustness of learned policy


8. IMPLEMENTATION DETAILS
8.1 Software Stack
Core Libraries:

CityLearn 2.0 (environment)
OmniSafe 1.0+ (Safe RL algorithms)
PyTorch 2.0+ (deep learning)
Gymnasium 0.29+ (RL interface)

Custom Components:

CityLearnSafetyEnvV3 (safety wrapper)
V3 EV deficit extractor
KPI logger (98-column CSV output)

Compute Requirements:

GPU: NVIDIA (CUDA support)
RAM: 16GB minimum
Storage: 10GB per experiment (logs, checkpoints)

8.2 Reproducibility
Version Control:

Git repository with tagged releases for each experiment
Config files versioned alongside code

Logging:

Per-step KPI logs (runs/kpi_logs/*.csv)
Training progress logs (runs/*/progress.csv)
Checkpoints every 10 epochs

Seeds:

Fixed random seeds for reproducibility
Seeds: 0, 42, 123 (for 3-seed runs)


9. TIMELINE AND MILESTONES
Week 1 (January 13-19, 2026) - Current Week

✅ Day 1-2: Add ramp cost constraint

Calibrate threshold from RBC
Implement in safety_env_v3.py
Update KPI logger
Test with RBC evaluation


⏳ Day 3-4: Create experiment configs

Vanilla PPO config
Safe PPO-Lag config (strong lambda)
Validate configurations


⏳ Day 5-7: Launch core experiments

Start Vanilla PPO training (100 epochs, ~10h)
Start Safe PPO-Lag training (100 epochs, ~10h)
Monitor progress



Week 2 (January 20-26, 2026)

Day 1-2: Complete core experiments

Finish Vanilla PPO training
Finish Safe PPO-Lag training
Evaluate final checkpoints


Day 3-5: Initial analysis

Generate training curves
Compare safety metrics
Compare performance metrics
Identify key findings


Day 6-7: Prepare ablation experiments

Decide which ablations are critical
Create configs for selected ablations



Week 3 (January 27 - February 2, 2026)

Day 1-5: Run ablation experiments

Lambda sensitivity (2-3 values)
V3 vs V2 comparison
Optional: Constraint ablation


Day 6-7: Analysis and visualization

Create all figures for thesis
Statistical significance testing
Compile results tables



Week 4 (February 3-9, 2026)

Day 1-4: Write results section

Experimental setup description
Present findings with figures/tables
Interpret results


Day 5-7: Write discussion section

Relate findings to research questions
Compare to related work
Discuss limitations
Propose future work



Week 5-6 (February 10-23, 2026)

Complete remaining thesis sections
Professor review and feedback
Revisions

Week 7 (February 24-28, 2026)

Final revisions
Thesis submission


10. RISK MITIGATION
10.1 Potential Issues and Solutions
Issue 1: Vanilla PPO doesn't violate constraints

Unlikely but possible if CityLearn weights are well-tuned
Solution: Report this as a positive finding; show Safe RL matches performance with formal guarantees
Alternative: Increase training time or adjust weights to induce violations

Issue 2: Safe RL fails to satisfy constraints within 100 epochs

Possible if lambda is still too weak
Solution: Increase lambda_init to 100, extend training to 150 epochs
Backup: Use lambda=50 results as "work in progress" and discuss convergence challenges

Issue 3: Safe RL performance degrades significantly (>40% penalty)

Constraints may be too tight
Solution: Relax cost_limit slightly (increase buffer)
Analysis: Quantify trade-off curve (safety vs performance)

Issue 4: Training crashes or diverges

Hyperparameter issues
Solution: Reduce learning rates, adjust batch size, add gradient clipping
Debug: Check for NaN values in cost/reward, verify environment wrapper

Issue 5: Insufficient time for all ablations

Realistic concern given thesis deadline
Solution: Prioritize RQ1-RQ3 (core experiments), make RQ4-RQ6 optional
Backup: Present preliminary results, mention full ablation as future work


11. SUCCESS CRITERIA
11.1 Minimum Viable Results
For thesis to be successful, must achieve:

✅ RBC Baseline completed (already done)
✅ Vanilla PPO trained and evaluated
✅ Safe PPO-Lag trained and evaluated
✅ Clear demonstration that:

Vanilla PPO violates at least one constraint significantly OR
Vanilla PPO and Safe RL both satisfy constraints but Safe RL provides formal guarantees


✅ Safe RL achieves competitive performance (CityLearn reward within 30% of Vanilla)

11.2 Strong Results
For strong thesis contribution, should additionally achieve:

✅ V3 vs V2 comparison demonstrating V3 improvement
✅ Lambda sensitivity analysis showing optimal initialization
✅ Convergence analysis showing learning dynamics
✅ Statistical significance for all key comparisons

11.3 Excellent Results
For publication-quality work, would ideally include:

✅ Full constraint ablation quantifying each constraint's contribution
✅ Reward function comparison (Bill vs Cost vs Multi-objective)
✅ Pareto frontier analysis (safety-performance trade-off curve)
✅ Generalization study (different scenarios or datasets)


12. DOCUMENTATION AND DELIVERABLES
12.1 Code Deliverables

✅ citylearn_safe/safety_env_v3.py - Safety wrapper with V3 classification + peak + ramp costs
✅ citylearn_safe/kpi_logger.py - Comprehensive KPI logging (100+ columns)
✅ configs/on-policy/ppo_lag_*.yaml - Safe RL experiment configs
⏳ configs/vanilla_ppo/ppo_*.yaml - Vanilla RL experiment configs
✅ scripts/train_omnisafe.py - Training script
✅ evaluation_pipeline/scripts/evaluate_*.py - Evaluation scripts
⏳ analysis/compare_experiments.py - Automated comparison script
⏳ analysis/generate_figures.py - Automated figure generation

12.2 Data Deliverables

✅ runs/kpi_logs/RBC_Greedy_V3.csv - RBC baseline results
⏳ runs/kpi_logs/Vanilla_PPO_*.csv - Vanilla RL results
⏳ runs/kpi_logs/Safe_PPO_Lag_*.csv - Safe RL results
⏳ evaluation_pipeline/processed_tables/comparison_all.csv - Unified comparison
⏳ evaluation_pipeline/plots/*.png - All thesis figures

12.3 Thesis Deliverables

⏳ Results Chapter: Experimental results with figures and tables
⏳ Discussion Chapter: Interpretation and analysis
⏳ Methodology Chapter: Updated with experimental design
⏳ Appendices: Additional results, hyperparameter details, statistical tests


13. REFERENCES TO EXISTING DATA
13.1 Completed Experiments (Reference)
RBC Greedy Baseline (January 8, 2026):

Total CMDP Cost: 149.57 (peak violations only, no ramp yet)
EV Controllable Deficit: 0.00 kWh
Peak Violations: 263 steps (3.0%)
CityLearn Reward: -1.95
File: runs/kpi_logs/RBC_Greedy_V3.csv

Bill-35ep Agent (January 7-8, 2026):

Configuration: PPO-Lag, Bill reward, lambda_init=10 (too weak)
Total CMDP Cost: 327.30 (193% over budget)
EV Controllable Deficit: 163.6 kWh ❌
Peak Violations: 766 steps (8.74%)
CityLearn Reward: -0.22 (89% better than RBC!)
Key Finding: Lambda=10 insufficient; agent behaved like unconstrained RL
File: runs/kpi_logs/Evaluated_PPOLag_bill_evscale2_ep35_seed000.csv

Cost-85ep Agent (January 7, 2026):

Configuration: PPO-Lag, Cost-only reward, lambda_init=10, 85 epochs
Total CMDP Cost: 182.17 (63% over budget, but declining trend)
Status: Training incomplete (should have continued to 150 epochs)
Key Finding: Shows learning is happening but needs more time or stronger lambda
File: runs/ppo_lag_cost_based_reward_v3_e85/*/progress.csv

13.2 Use of Existing Results
These preliminary results will be incorporated into the thesis as:

Bill-35ep as negative example: Shows what happens when lambda is too weak
Cost-85ep as learning trajectory: Shows convergence is possible with more training
Motivation for lambda_init=50: Justified by failure of lambda=10


14. OPEN QUESTIONS FOR DISCUSSION
14.1 For Professor Review

Constraint Budget: Should we increase cost_limit to 300-350 to account for ramp cost being added, or keep it at 280 and see if agent can achieve tighter budget?
Training Duration: Is 100 epochs sufficient, or should we plan for 150 epochs based on Cost-85ep trends?
Multiple Seeds: Resources permitting, should we run 3 seeds for all experiments or only for the core comparisons (RBC, Vanilla, Safe RL)?
V3 vs V2 Priority: Is comparing V3 vs V2 classification essential for the thesis core, or can it be secondary/appendix?
CityLearn Reward Function: Should we implement the exact CityLearn Challenge reward function or use a simplified version for Vanilla PPO?
Thesis Scope: Given timeline constraints, should we limit ablations to only lambda sensitivity and V3 vs V2, or attempt full constraint ablation?


15. CONCLUSION
This experimental plan provides a comprehensive framework for validating the hypothesis that Safe RL with explicit constraints produces more reliable and deployable policies than reward-shaping approaches for V2G energy management. The three core experiments (RBC, Vanilla PPO, Safe PPO-Lag) directly address the primary research questions, while ablation studies provide deeper insights into mechanism and design choices.
Expected Timeline: 4 weeks for experiments, 2 weeks for analysis/writing
Risk Level: Moderate (main risk is convergence time for Safe RL)
Innovation: First application of Safe RL with action-based EV classification to V2G
Impact: Demonstrates formal safety guarantees for critical infrastructure applications

Status: Draft for review
Next Steps:

Add ramp cost constraint (immediate)
Professor review and approval (this week)
Begin core experiments (next week)


END OF EXPERIMENTAL PLAN DRAFT