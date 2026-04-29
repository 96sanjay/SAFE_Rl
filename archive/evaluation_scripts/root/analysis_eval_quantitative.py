"""
Quantitative analysis of PPOLagMulti evaluation data.
Seed fixed for reproducibility where randomness is involved.
"""
import numpy as np
np.random.seed(42)

print("=" * 80)
print("QUANTITATIVE ANALYSIS: PPOLagMulti Evaluation (5 Buildings, 8760 Steps)")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LAGRANGIAN COST DOMINANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("1. LAGRANGIAN COST DOMINANCE")
print("─" * 80)

raw_costs = {
    "C0 (EV Dense)":      7.6,
    "C1 (EV Departure)":  1.3,
    "C2 (Battery SoC)":   14033.2,
    "C3 (Building Power)":22406.0,
    "C4 (Grid Power)":    18300.9,
}

weights = {
    "C0 (EV Dense)":      10,
    "C1 (EV Departure)":  5,
    "C2 (Battery SoC)":   1,
    "C3 (Building Power)":0.1,
    "C4 (Grid Power)":    5,
}

weighted_costs = {}
for k in raw_costs:
    weighted_costs[k] = raw_costs[k] * weights[k]

total_weighted = sum(weighted_costs.values())
total_raw = sum(raw_costs.values())

print(f"\n{'Constraint':<25s} {'Raw Cost':>12s} {'Weight':>8s} {'Weighted':>12s} {'% Total':>10s}")
print("-" * 70)
for k in raw_costs:
    frac = weighted_costs[k] / total_weighted * 100
    print(f"{k:<25s} {raw_costs[k]:>12.1f} {weights[k]:>8.1f} {weighted_costs[k]:>12.1f} {frac:>9.1f}%")
print("-" * 70)
print(f"{'TOTAL':<25s} {total_raw:>12.1f} {'':>8s} {total_weighted:>12.1f} {'100.0':>9s}%")

# Identify dominant constraint
sorted_wc = sorted(weighted_costs.items(), key=lambda x: x[1], reverse=True)
print(f"\nDominant constraint: {sorted_wc[0][0]} at {sorted_wc[0][1]/total_weighted*100:.1f}% of total weighted cost")
print(f"Top-2 combined: {(sorted_wc[0][1]+sorted_wc[1][1])/total_weighted*100:.1f}%")

# Concentration ratio (HHI-like)
shares = np.array([v/total_weighted for v in weighted_costs.values()])
hhi = np.sum(shares**2)
print(f"Herfindahl-Hirschman Index of cost concentration: {hhi:.4f} (1/N={1/len(shares):.4f} would be perfectly balanced)")

# Lambda dominance implication
print(f"\nImplication for lambda: C4 (Grid Power) contributes {sorted_wc[0][1]/total_weighted*100:.1f}% of the")
print(f"gradient signal. Since OmniSafe uses a shared lambda, the policy optimizes")
print(f"almost exclusively for grid power. C0 and C1 ({(weighted_costs['C0 (EV Dense)']+weighted_costs['C1 (EV Departure)'])/total_weighted*100:.3f}%)")
print(f"are effectively invisible to the Lagrangian.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ENERGY BALANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("2. ENERGY BALANCE")
print("─" * 80)

elec_ratio = 2.189
baseline_reward = 7230
agent_reward = -4524

# Electricity ratio = agent_consumption / baseline_consumption
# Extra consumption = (ratio - 1) * baseline = 1.189 * baseline
extra_fraction = (elec_ratio - 1.0)
print(f"\nElectricity consumption ratio: {elec_ratio:.3f}x baseline")
print(f"Extra energy drawn: {extra_fraction:.3f}x baseline = +{extra_fraction*100:.1f}% above baseline")

# Zero Net Energy ratio is 3.452 -- this measures how much net import vs baseline
zne_ratio = 3.452
print(f"\nZero Net Energy ratio: {zne_ratio:.3f}x baseline")
print(f"  This means the agent imports {zne_ratio:.3f}x as much net energy as baseline.")
print(f"  Net import surplus vs baseline: {(zne_ratio-1)*100:.1f}%")

# Solar self-consumption inference
# ZNE is much worse than electricity ratio, meaning the agent is NOT using solar
# ZNE 3.45x vs Elec 2.19x => agent is exporting solar and reimporting from grid
solar_waste_indicator = zne_ratio / elec_ratio
print(f"\nSolar waste indicator (ZNE/Electricity ratio): {solar_waste_indicator:.3f}")
print(f"  If the agent consumed all solar, ZNE would track Electricity ~1:1.")
print(f"  Ratio of {solar_waste_indicator:.2f} means the agent's net grid dependence")
print(f"  grows {solar_waste_indicator:.2f}x faster than its gross consumption.")

# Battery behavior explains this: discharge 19/24 hours, mean=-0.511
# The battery is DISCHARGING during solar hours, pushing energy to grid,
# then the building must import from grid later
batt_mean = -0.511
ev_mean = 0.497
solar_pref = -1.250
ev_solar_pct = 31.2

print(f"\nBattery solar preference: {solar_pref:.3f} (negative = discharging during solar)")
print(f"  Battery discharges during {19}/24 hours including peak solar hours.")
print(f"  This EXPORTS stored/solar energy, inflating both grid import and ZNE.")
print(f"EV charge during solar: {ev_solar_pct:.1f}% (vs ~50% daylight hours = underweight)")

# Estimate energy waste from discharge bias
# With 19/24 discharge hours and mean=-0.511, the agent is a net energy sink
discharge_hours = 19
charge_hours = 4
# Net energy flow direction over 24h
# If balanced: 12h charge, 12h discharge, mean ~0
# Actual: heavy discharge bias
net_discharge_bias = (discharge_hours - charge_hours) / 24
print(f"\nDischarge time bias: {discharge_hours}h discharge vs {charge_hours}h charge")
print(f"  Net discharge fraction of day: {net_discharge_bias:.3f} ({net_discharge_bias*100:.1f}%)")
print(f"  62.5% of hours are net-discharge. A balanced agent would have ~50/50.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. COST-REWARD TRADEOFF EFFICIENCY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("3. COST-REWARD TRADEOFF EFFICIENCY (or INEFFICIENCY)")
print("─" * 80)

reward_sacrifice = baseline_reward - agent_reward  # positive = how much reward lost
cost_agent = 152459
cost_baseline = 110613
cost_change = cost_agent - cost_baseline  # positive = agent INCREASED cost

print(f"\nReward: agent={agent_reward}, baseline={baseline_reward}")
print(f"  Reward sacrifice: {reward_sacrifice:+d} (agent is {reward_sacrifice} worse)")
print(f"\nCost: agent={cost_agent}, baseline={cost_baseline}")
print(f"  Cost change: {cost_change:+d} (agent INCREASED cost by {cost_change})")
print(f"  Cost increase ratio: {cost_agent/cost_baseline:.3f}x baseline")

print(f"\nThis is a Pareto-DOMINATED outcome:")
print(f"  The agent sacrificed {reward_sacrifice} reward AND increased cost by {cost_change}.")
print(f"  Both objectives are WORSE than the do-nothing baseline.")

# Pain-per-unit analysis: there is no improvement, but we can compute the ratio
# of harm in each dimension
if cost_change > 0:
    pain_ratio = reward_sacrifice / cost_change
    print(f"\n  Harm ratio: {pain_ratio:.3f} reward-units lost per cost-unit added")
    print(f"  In other words, each unit of cost INCREASE cost the agent {pain_ratio:.3f}")
    print(f"  units of reward. The agent is actively self-harming on both axes.")

# Per-step analysis
steps = 8760
print(f"\n  Per-step reward: agent={agent_reward/steps:.2f}, baseline={baseline_reward/steps:.2f}")
print(f"  Per-step cost:   agent={cost_agent/steps:.2f}, baseline={cost_baseline/steps:.2f}")
print(f"  Per-step reward deficit: {reward_sacrifice/steps:.2f}")
print(f"  Per-step cost surplus:   {cost_change/steps:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ACTION BIAS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("4. ACTION BIAS ANALYSIS")
print("─" * 80)

batt_mean = -0.511
batt_std = 0.700
ev_mean = 0.497
ev_std = 0.725
overall_abs_mean = 0.828

print(f"\nBattery: mean={batt_mean:.3f}, std={batt_std:.3f}")
print(f"  Discharge hours: {discharge_hours}/24 ({discharge_hours/24*100:.1f}%)")
print(f"  Charge hours:    {charge_hours}/24 ({charge_hours/24*100:.1f}%)")

# A balanced agent: mean ~0, roughly equal charge/discharge
# The 'imbalance ratio'
batt_imbalance = abs(batt_mean) / batt_std  # how many SDs the mean is from zero
print(f"\n  Battery imbalance (|mean|/std): {batt_imbalance:.3f}")
print(f"    This means the mean is {batt_imbalance:.2f} standard deviations from zero.")
print(f"    A balanced agent would have ratio near 0.")

# For a Gaussian with mean=-0.511, std=0.700, what fraction is negative?
from scipy import stats
frac_negative_batt = stats.norm.cdf(0, loc=batt_mean, scale=batt_std)
print(f"\n  P(battery action < 0) assuming Gaussian: {frac_negative_batt:.3f} ({frac_negative_batt*100:.1f}%)")
print(f"  Observed discharge fraction: {discharge_hours/24:.3f} ({discharge_hours/24*100:.1f}%)")
print(f"  A balanced agent: P(negative) ~ 50%")

# Energy asymmetry
# With mean=-0.511, the agent discharges on average at 51.1% of max rate
# but charges at a lower average rate (in the few hours it charges)
# Net energy balance per day:
# If action in [-1,1] maps to power, net = mean * 24h
net_per_day = batt_mean * 24  # in action-hours
print(f"\n  Net battery action-hours per day: {net_per_day:.2f}")
print(f"  (negative = net discharge). A balanced agent: ~0 action-hours/day")

# EV analysis
frac_positive_ev = 1 - stats.norm.cdf(0, loc=ev_mean, scale=ev_std)
print(f"\n  EV: mean={ev_mean:.3f}, std={ev_std:.3f}")
print(f"  P(EV action > 0) assuming Gaussian: {frac_positive_ev:.3f} ({frac_positive_ev*100:.1f}%)")
print(f"  EV has strong charge bias (expected for EV management)")

# Overall action magnitude
print(f"\n  Overall |mean| of actions: {overall_abs_mean:.3f}")
print(f"  This is {overall_abs_mean*100:.1f}% of the action range maximum.")
print(f"  The agent operates near the EDGES of the action space, not the center.")
print(f"  Saturation this high often indicates policy collapse to a fixed pattern.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. INFORMATION UTILIZATION GAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("5. INFORMATION UTILIZATION GAP")
print("─" * 80)

mi_top5 = 0.950
cond_entropy = 0.279
price_gradient = 0.012
building_gradient = 0.067
current_fraction = 0.849

print(f"\nMutual Information (top-5 features): {mi_top5:.3f}")
print(f"  The policy's actions share {mi_top5:.3f} bits of information with the top-5")
print(f"  observation features. This is high -- the policy CAN see the inputs.")

print(f"\nConditional entropy (top-5): {cond_entropy:.3f}")
print(f"  Given the top-5 features, {cond_entropy:.3f} bits of action entropy remain.")
print(f"  Entropy reduction: MI / (MI + H_cond) = {mi_top5/(mi_top5+cond_entropy):.3f}")

print(f"\nPrice gradient: {price_gradient:.3f}")
print(f"  When price changes by 1 unit, the action changes by {price_gradient:.3f}")
print(f"  This is NEGLIGIBLE price responsiveness.")

print(f"\nBuilding gradient: {building_gradient:.3f}")
print(f"  Building-specific features influence actions {building_gradient/price_gradient:.1f}x more than price.")

# The gap
utilization_gap = mi_top5 - price_gradient
utilization_ratio = mi_top5 / max(price_gradient, 1e-10)
print(f"\nINFORMATION UTILIZATION GAP:")
print(f"  MI (sees):           {mi_top5:.3f}")
print(f"  Price gradient (uses): {price_gradient:.3f}")
print(f"  Gap:                 {utilization_gap:.3f}")
print(f"  Ratio (sees/uses):   {utilization_ratio:.1f}x")
print(f"\n  The agent sees information {utilization_ratio:.0f}x better than it uses price signal.")
print(f"  This suggests the policy has learned to ATTEND to features but not to")
print(f"  produce DIFFERENTIATED responses to economically meaningful signals.")

# Current fraction analysis
print(f"\nCurrent-state fraction: {current_fraction:.3f}")
print(f"  {current_fraction*100:.1f}% of gradient comes from current-timestep features.")
print(f"  Only {(1-current_fraction)*100:.1f}% comes from history/temporal features.")
print(f"  The temporal attention (STEMS) contributes minimally to final actions.")

# Behavioral variance ratio
bvr = 0.110
print(f"\nBehavioral variance ratio: {bvr:.3f}")
print(f"  Only {bvr*100:.1f}% of action variance is explained by observation variance.")
print(f"  {(1-bvr)*100:.1f}% of action variance is effectively NOISE or fixed bias.")
print(f"  An effective policy would have BVR >> 0.5.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. BUILDING DIFFERENTIATION DEFICIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("6. BUILDING DIFFERENTIATION DEFICIT")
print("─" * 80)

building_corr = 0.771
n_buildings = 5

print(f"\nInter-building action correlation: {building_corr:.3f}")
print(f"Number of buildings: {n_buildings}")

# Effective independent buildings
# If correlation = 1.0, effectively 1 building
# If correlation = 0.0, effectively N buildings
# Effective N = N / (1 + (N-1)*rho)  (from variance of sum formula)
effective_n = n_buildings / (1 + (n_buildings - 1) * building_corr)
print(f"\nEffective independent building count:")
print(f"  Formula: N_eff = N / (1 + (N-1)*rho)")
print(f"  N_eff = {n_buildings} / (1 + {n_buildings-1}*{building_corr:.3f})")
print(f"  N_eff = {n_buildings} / {1 + (n_buildings-1)*building_corr:.3f}")
print(f"  N_eff = {effective_n:.3f}")

# Theoretical loss from treating 5 as 1
redundancy = 1 - (effective_n / n_buildings)
print(f"\nRedundancy: {redundancy:.3f} ({redundancy*100:.1f}%)")
print(f"  {redundancy*100:.1f}% of the multi-building control capacity is WASTED.")
print(f"  The policy uses {effective_n:.2f} effective controllers out of {n_buildings} possible.")
print(f"  Lost control degrees of freedom: {n_buildings - effective_n:.2f} out of {n_buildings}")

# What this means for peak shaving
print(f"\nImplication for peak shaving:")
print(f"  Peak shaving requires STAGGERED responses across buildings.")
print(f"  With rho={building_corr:.3f}, buildings charge/discharge nearly in unison,")
print(f"  which AMPLIFIES grid peaks instead of flattening them.")
print(f"  Daily Peak KPI = 1.890 (89% WORSE than baseline) confirms this.")
daily_peak = 1.890
peak_excess = (daily_peak - 1.0) * 100
print(f"  Excess daily peak: +{peak_excess:.0f}% above baseline")


# ─────────────────────────────────────────────────────────────────────────────
# 7. C2 CLAMP DEPENDENCY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("7. C2 (BATTERY SoC) CLAMP DEPENDENCY")
print("─" * 80)

c2_raw_pct = 34.04
c2_clamped_pct = 9.16
c2_raw_cost = 14033.2
c2_clamped_cost = 429.7

# Steps
c2_raw_steps = c2_raw_pct / 100 * steps
c2_clamped_steps = c2_clamped_pct / 100 * steps
c2_prevented_steps = c2_raw_steps - c2_clamped_steps

print(f"\nRaw violations:     {c2_raw_pct:.2f}% = {c2_raw_steps:.0f} steps")
print(f"After clamp:        {c2_clamped_pct:.2f}% = {c2_clamped_steps:.0f} steps")
print(f"Prevented by clamp: {c2_prevented_steps:.0f} steps")

# What fraction of "compliance" is earned?
total_compliant_steps = steps - c2_clamped_steps
clamp_contribution_steps = c2_prevented_steps
policy_contribution_steps = steps - c2_raw_steps

pct_earned = policy_contribution_steps / total_compliant_steps * 100
pct_gifted = clamp_contribution_steps / total_compliant_steps * 100

print(f"\nCompliance attribution:")
print(f"  Total compliant steps (after clamp): {total_compliant_steps:.0f}")
print(f"  Compliant by policy alone:           {policy_contribution_steps:.0f} ({pct_earned:.1f}%)")
print(f"  Compliant only due to clamp:         {clamp_contribution_steps:.0f} ({pct_gifted:.1f}%)")

# Cost reduction
cost_reduction = c2_raw_cost - c2_clamped_cost
cost_reduction_pct = cost_reduction / c2_raw_cost * 100
print(f"\nCost reduction from clamp:")
print(f"  Raw cost:     {c2_raw_cost:.1f}")
print(f"  Clamped cost: {c2_clamped_cost:.1f}")
print(f"  Reduction:    {cost_reduction:.1f} ({cost_reduction_pct:.1f}%)")

# Dependency ratio
dependency = c2_clamped_steps / c2_raw_steps if c2_raw_steps > 0 else 0
prevented_fraction = 1 - dependency
print(f"\nClamp dependency ratio:")
print(f"  {prevented_fraction*100:.1f}% of the agent's raw violations are caught by the clamp")
print(f"  Only {dependency*100:.1f}% of raw violations pass through the clamp")
print(f"  The clamp reduces violations by {(1-c2_clamped_pct/c2_raw_pct)*100:.1f}%")
print(f"  The clamp reduces cost by {cost_reduction_pct:.1f}%")

print(f"\nInterpretation:")
print(f"  Without the safety clamp, the policy would violate SoC bounds in")
print(f"  {c2_raw_pct:.1f}% of steps -- more than 1 in 3.")
print(f"  The clamp is not a fallback; it is the PRIMARY safety mechanism.")
print(f"  The policy has NOT learned SoC constraint satisfaction.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. SUMMARY: TOP 5 QUANTITATIVE FINDINGS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("8. TOP 5 QUANTITATIVE FINDINGS (THESIS-READY)")
print("─" * 80)

print("""
1. PARETO-DOMINATED OUTCOME: The trained agent achieves reward = -4524 vs
   baseline +7230 (a deficit of 11754) while SIMULTANEOUSLY increasing total
   constraint cost from 110613 to 152459 (+37.8%). Both optimization objectives
   are strictly worse than the do-nothing baseline, indicating fundamental
   training failure rather than a reward-safety tradeoff.

2. LAGRANGIAN COST CONCENTRATION: Grid power (C4) contributes 84.8% of the
   total weighted cost signal (91505 out of 107861). With a single shared
   lambda, the policy optimizes almost exclusively for C4, rendering EV
   constraints (C0+C1, combined 0.08% of weighted cost) invisible to the
   Lagrangian. This is a structural limitation of the single-lambda PPOLag
   formulation applied to heterogeneous constraints.

3. SAFETY CLAMP DEPENDENCY: The policy violates battery SoC bounds in 34.0%
   of timesteps. The post-hoc safety clamp reduces this to 9.2%, catching
   73.1% of violations and reducing C2 cost by 96.9% (14033 -> 430). Only
   72.7% of the policy's apparent compliance is genuinely learned; 27.3% is
   "gifted" by the engineering safeguard.

4. INFORMATION-ACTION DISCONNECT: Mutual information with top features is
   0.950 (the policy observes price and load signals) but the price gradient
   is only 0.012 (actions change by 1.2% per unit price change). The
   behavioral variance ratio of 0.110 confirms that 89% of action variance
   is unrelated to observations. The policy has collapsed to a near-fixed
   discharge-heavy pattern (battery mean = -0.511, 79.2% of hours discharging).

5. BUILDING HOMOGENEITY WASTES MULTI-AGENT CAPACITY: Inter-building action
   correlation of 0.771 yields only 1.23 effective independent controllers
   out of 5 buildings (75.4% redundancy). This synchronized behavior directly
   causes the 89% daily peak increase (KPI = 1.890), since coordinated
   discharge amplifies rather than flattens grid peaks.
""")

# ─────────────────────────────────────────────────────────────────────────────
# ADDITIONAL: Composite failure score
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 80)
print("COMPOSITE METRICS")
print("─" * 80)

# Normalized performance vs baseline (1.0 = matches baseline, 0.0 = terrible)
reward_score = max(0, agent_reward / baseline_reward)  # negative/positive = 0 or worse
cost_score = max(0, 1 - (cost_agent - cost_baseline) / cost_baseline)  # 1 if same, 0 if 2x

print(f"\nReward score (agent/baseline, clipped [0,1]): {reward_score:.3f}")
print(f"  (Negative because agent reward is negative while baseline is positive)")
print(f"Cost score (1 - excess/baseline, clipped [0,1]): {cost_score:.3f}")
print(f"\nAverage KPI (CityLearn, 1.0=baseline): ", end="")
kpis = [2.189, 2.304, 2.206, 1.890, 1.118, 1.591, 3.452]
kpi_names = ["Electricity", "Carbon", "Cost", "Daily Peak", "All-time Peak", "Ramping", "ZNE"]
avg_kpi = np.mean(kpis)
print(f"{avg_kpi:.3f}")
print(f"\n{'KPI':<20s} {'Value':>8s} {'vs Baseline':>12s}")
print("-" * 42)
for name, val in zip(kpi_names, kpis):
    print(f"{name:<20s} {val:>8.3f} {(val-1)*100:>+11.1f}%")
print("-" * 42)
print(f"{'Mean':<20s} {avg_kpi:>8.3f} {(avg_kpi-1)*100:>+11.1f}%")
print(f"\nAll 7 KPIs are ABOVE 1.0 (worse than baseline).")
print(f"Worst: Zero Net Energy at {max(kpis):.3f}x baseline (+{(max(kpis)-1)*100:.1f}%)")
print(f"Best:  All-time Peak at {min(kpis):.3f}x baseline (+{(min(kpis)-1)*100:.1f}%)")

print("\n" + "=" * 80)
print("END OF ANALYSIS")
print("=" * 80)
