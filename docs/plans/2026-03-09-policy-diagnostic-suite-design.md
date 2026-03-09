# Policy Diagnostic Suite Design

**Date:** 2026-03-09
**Purpose:** Mathematically diagnose WHY the RL agent fails to learn intelligent decisions
**Output:** Single script `scripts/diagnose_policy_health.py` + structured report

---

## Problem Statement

The agent exhibits ALL failure symptoms simultaneously:
- Actions look random/uniform
- Costs stay high (590K vs 400K limit)
- Reward doesn't improve (or gets worse than zero-action)
- All buildings do the same thing

We need a mathematical framework to decompose "the agent doesn't work" into specific, actionable diagnoses.

---

## Architecture

```
scripts/diagnose_policy_health.py
    --checkpoint <path-to-epoch-N.pt>
    --config <path-to-yaml>
    --output-dir diagnostics/<run>_<epoch>/

Output:
    diagnostics/<run>_<epoch>/
        report.md           # Human-readable summary with pass/fail
        mi_matrix.npy       # 330x9 MI matrix
        phi_score.json      # Composite Policy Health Index
        rollout_data.npz    # Raw (obs, actions, rewards, costs) for reanalysis
        figures/
            mi_heatmap.png
            action_autocorrelation.png
            violation_timing.png
            value_calibration.png
            cross_temporal_corr.png
            building_correlation.png
```

### Execution Flow

1. Load checkpoint + reconstruct actor (STEMSMeanNet or MLP)
2. Load config + instantiate environment
3. Run one full evaluation episode (8,759 steps) with deterministic mean policy
4. Run zero-action baseline episode (for headroom comparison)
5. Execute 8 diagnostic tests on collected data
6. Compute composite PHI score
7. Run decision tree → output actionable diagnosis
8. Generate figures + write report

---

## Test Specifications

### Test 1: Value Function Accuracy (PRIORITY 1)

**Purpose:** If the critics are broken, ALL policy gradients are noise.

**Method:**
- Load reward critic V_r and cost critic V_c from checkpoint
- Compute actual discounted returns: R_t = Σ_{k=0}^{T-t-1} γ^k r_{t+k}
- Compute explained variance: EV = 1 - Var(R_actual - V_pred) / Var(R_actual)
- Compute TD errors: δ_t = r_t + γV(s_{t+1}) - V(s_t)
- Check TD error autocorrelation: corr(δ_t, δ_{t+1})

**Thresholds:**

| Metric | Healthy | Warning | Broken |
|--------|---------|---------|--------|
| EV_reward | > 0.5 | 0.1-0.5 | < 0.1 |
| EV_cost | > 0.5 | 0.1-0.5 | < 0.1 |
| TD autocorr | < 0.1 | 0.1-0.3 | > 0.3 |

**If broken → Action:** Increase critic LR, add critic update epochs, check critic architecture.

---

### Test 2: Feature-Action Mutual Information (PRIORITY 1)

**Purpose:** Does the policy extract useful information from observations?

**Method:**
- KSG estimator (k=5) via `sklearn.feature_selection.mutual_info_regression`
- Compute full 330×9 MI matrix
- Compare against null (uniform random policy MI ≈ 0.01-0.03 nats)

**Expected high-MI pairs (domain physics):**

| Feature | Action | Expected MI | Rationale |
|---------|--------|-------------|-----------|
| electricity_pricing | a_battery_i | > 0.1 | Buy low, sell high |
| solar_generation_i | a_battery_i | > 0.08 | Store excess solar |
| electrical_storage_soc_i | a_battery_i | > 0.15 | Don't over-charge/discharge |
| ev_connected_state | a_ev_charger | > 0.3 | Can only charge when connected |
| ev_required_soc_departure | a_ev_charger | > 0.1 | Charge urgency |
| hour_cos/sin | all actions | > 0.05 | Diurnal patterns |
| pricing_predicted_1/2/3 | a_battery_i | > 0.05 | Lookahead arbitrage |

**Calibration:**
- MI < 2× noise floor (~0.02) → feature ignored
- 0.02-0.1 → weak sensitivity
- 0.1-0.3 → moderate (policy uses it)
- > 0.3 → strong driver

**If all MI ≈ noise → Action:** Check gradient pathways (Test 5). Encoder is dead.

---

### Test 3: Conditional Entropy Decomposition (PRIORITY 2)

**Purpose:** Complement to MI — measures HOW MUCH each feature reduces action uncertainty.

**Method:**
- Bin each feature into 20 quantile bins
- Compute Var(action | feature_bin) for each bin
- Entropy reduction ratio: ρ = I(X,a) / H(a)
- Also: 2D conditional H(a | price, SOC) — tests whether policy learned the interaction

**Thresholds:**
- ρ < 0.01 → policy ignores feature
- 0.01-0.05 → weak conditioning
- 0.05-0.20 → moderate (feature is used)
- > 0.20 → strong driver

**If broken → Action:** Same as Test 2 — encoder pathway problem.

---

### Test 4: Inter-Building Action Correlation (PRIORITY 2)

**Purpose:** Is the GCN earning its architectural cost?

**Method:**
- 5×5 Pearson correlation matrix for battery actions
- 3×3 correlation matrix for EV actions
- Cross-type correlation: corr(a_battery_i, a_ev_i) per building
- Spatial variance ratio: η = spatial_variance / total_variance
  - where spatial = Var across buildings at each timestep
  - total = Var across all (building, timestep) pairs

**Action autocorrelation:**
- ACF(a_j, τ) for τ = 1..48 hours
- Healthy: ACF(24) ≈ 0.3 (daily cycle), ACF(1) ≈ 0.5 (smooth)
- Broken: ACF(τ) ≈ 0 ∀τ (white noise) or ACF(τ) ≈ 1 ∀τ (frozen)

**Thresholds:**

| Metric | Healthy | Broken |
|--------|---------|--------|
| mean \|R_ij\| (i≠j) | < 0.7 | > 0.9 |
| η (spatial ratio) | > 0.1 | < 0.05 |
| ACF(24) | > 0.1 | < 0.02 |

**If broken → Action:** GCN outputs are being averaged away. Check gated fusion weights, adjacency matrix learning.

---

### Test 5: Gradient Attribution by Pathway (PRIORITY 2)

**Purpose:** Which encoder pathway is carrying information? Which is dead?

**Method:**
- Compute ∂μ_j/∂s_i for all (i,j) pairs (330×9 Jacobian)
- Aggregate by pathway:
  - Temporal: gradient w.r.t. history dims (indices 198-329)
  - Current building: gradient w.r.t. per-building features
  - Global: gradient w.r.t. shared features (price, time, carbon)
  - EV: gradient w.r.t. EV-specific features
- Pathway fraction = |grad_pathway| / |grad_total|
- Also: Integrated Gradients for thesis-quality attribution

**Thresholds:**

| Pathway | Healthy fraction | Broken |
|---------|-----------------|--------|
| Temporal | > 0.10 | < 0.01 |
| Global (price) | > 0.15 | < 0.02 |
| Per-building | > 0.20 | < 0.05 |
| EV features | > 0.05 | < 0.005 |

**If temporal dead → Action:** Check temporal transformer weights, concat projection. May be a LayerNorm attenuation issue.
**If price dead → Action:** Global encoder → broadcast mechanism → per-node integration is broken.

---

### Test 6: Temporal Planning Assessment (PRIORITY 1 — UPGRADED)

**Purpose:** Quantify whether the agent does temporal PLANNING (intelligent multi-step decisions) vs myopic reacting. Temporal planning is THE key capability that should improve ALL KPIs: better cost management, higher rewards, and fewer constraint violations.

**Why temporal planning matters for each KPI:**
- **Reward (R_economic):** Price arbitrage requires buying low NOW because price will be HIGH in 4 hours → temporal planning
- **Cost (C1 - EV departure):** Must start charging early enough to reach required SOC → multi-hour planning
- **Constraint (C3 - building power):** Spread battery charging across hours to avoid peak → temporal smoothing
- **Constraint (C4 - grid peak):** Coordinate building-level discharge during district peak → temporal coordination

#### 6a. Cross-Temporal Correlation Matrix (the "planning fingerprint")

**Method:**
- Compute corr(a_{battery_i,t}, price_{t+τ}) for τ = -6..+24 for each building
- Also: corr(a_{battery_i,t}, solar_{t+τ}), corr(a_{ev,t}, ev_departure_time)

**Interpretation table:**

| Pattern | What it means | Agent quality |
|---------|--------------|---------------|
| Only corr(τ=0) ≠ 0 | Purely reactive — responds to current price only | Myopic |
| corr(τ=1..3) ≠ 0 | Short-horizon planning (1-3 hours ahead) | Basic |
| corr(τ=1..6) ≠ 0 with structure | Medium-horizon (uses price forecast window) | Good |
| corr(τ=1..12) with decay | Full planning horizon, diminishing future weight | Intelligent |
| corr(τ<0) ≠ 0, corr(τ>0) ≈ 0 | Backward-looking only (reacting to past, not planning) | Broken |

**Temporal Planning Score (TPS):**
```
TPS = Σ_{τ=1}^{24} |corr(a_battery, price_{t+τ})| × decay(τ)
    / Σ_{τ=1}^{24} decay(τ)

where decay(τ) = exp(-τ/6) (6-hour characteristic scale)
```
- TPS < 0.02 → No planning (myopic)
- TPS 0.02-0.05 → Weak planning
- TPS 0.05-0.15 → Moderate planning
- TPS > 0.15 → Strong temporal planning

#### 6b. Granger Causality Test

**Method:** F-test: do future prices help predict current actions beyond current state?

Restricted model: a_t = f(price_t, SOC_t, ..., a_{t-1..t-6})
Unrestricted model: a_t = f(price_t, SOC_t, ..., a_{t-1..t-6}, price_{t+1..t+6})

If F-test rejects H0 (p < 0.01) → agent uses forecast information for planning.

#### 6c. Temporal Attention Analysis

**Method:**
- Extract attention weights from PerNodeTemporalTransformer (4 heads × 12 positions)
- For each head, compute:
  - Entropy ratio = H(attn) / log(12) — how focused is attention?
  - Peak position — which history timestep gets most attention?
  - Attention gradient — does attention shift based on state (e.g., price spike)?

**Thresholds:**
- Entropy ratio > 0.95 → Uniform attention (transformer is an averaging layer = DEAD)
- Entropy ratio 0.70-0.85 → Focused attention (learning temporal patterns = HEALTHY)
- Entropy ratio < 0.50 → Over-focused (may be degenerately attending to one position)

#### 6d. Temporal Perturbation Tests

| Test | Method | Healthy | Broken |
|------|--------|---------|--------|
| Zero history | Set obs[198:330] = 0, measure Δaction | > 5% change | < 1% |
| Shuffle timesteps | Randomly permute 12-step history order | > 3% change | < 0.5% |
| Reverse history | Flip history order (newest→oldest) | > 2% change | < 0.5% |
| Future price perturbation | Replace price forecast with random | > 4% change | < 1% |

#### 6e. Temporal Planning Impact Estimator

**The key question:** If temporal planning were PERFECT, how much would each KPI improve?

**Method (oracle comparison):**
1. Run agent with actual observations → get KPIs
2. Run simple heuristic: "charge battery when price is below daily median, discharge above" → get KPIs
3. Run oracle: "charge at daily minimum price, discharge at daily maximum" → get KPIs

**The gap between (1) and (3) = maximum temporal planning headroom:**
```
temporal_headroom_reward = (oracle_reward - agent_reward) / (oracle_reward - random_reward)
temporal_headroom_C1 = (agent_C1 - oracle_C1) / (agent_C1 - 0)  # 0 is perfect
```

This tells you: "If I fix ONLY temporal planning, I can capture X% of the remaining performance gap."

**If broken → Action:** Check temporal transformer weights. Verify concat projection isn't attenuating temporal signal. Consider increasing temporal embedding dimension or adding more transformer layers.

---

### Test 7: Constraint Decomposition (PRIORITY 3)

**Purpose:** Which constraint violations are the agent's fault vs physics?

**Method:**
For each constraint at each violation timestep:
1. Record current cost
2. Sample 100 random actions, compute cost for each
3. If any random action achieves cost=0 → violation is BEHAVIORAL
4. If no random action achieves cost=0 → violation is STRUCTURAL

**Per-constraint metrics:**
- Violation rate: VR = fraction of steps with cost > 0
- Violation magnitude: mean cost when violated
- Structural fraction: fraction of violations no action can fix
- Timing: violation rate by hour of day (heatmap)

**Known structural floors (from theoretical analysis):**
- C1 (EV departure): ~0 structural (agent has time to charge)
- C2 (Battery SOC): ~200K structural (tight bands + battery dynamics)
- C3 (Building power): ~100K structural (12kW PV >> 4.6kW limit)
- C4 (Grid peak): 0 structural (never binding at 5 buildings)

**If high behavioral violations → Action:** Policy is taking wrong actions at wrong times. Focus training on those specific constraints.
**If mostly structural → Action:** Relax constraint thresholds or raise cost_limit to account for structural floor.

---

### Test 8: Headroom Analysis (PRIORITY 1)

**Purpose:** Know your ceiling and floor — where is improvement possible?

**Method:**
- Run zero-action baseline episode
- Decompose reward into 5 components (economic, stability_grid, stability_building, ramp, renewable)
- Decompose cost into 4 components (C1, C2, C3, C4)
- Compute per-component:
  ```
  headroom_i = (policy_i - zero_action_i) / (oracle_i - zero_action_i)
  ```
- Negative headroom = policy is WORSE than doing nothing

**Oracle estimates:**
- R_economic oracle: charge at min price, discharge at max price
- R_stability oracle: minimize |net_power| at all times (battery as buffer)
- C1 oracle: always charge EV to required SOC before departure
- C2 oracle: keep SOC in [0.05, 0.90] with margin

**Output:** Table with 9 rows (5 reward + 4 cost components), showing floor/ceiling/current/headroom.

---

## Composite Score: Policy Health Index (PHI)

```
PHI = 0.20×S_value + 0.20×S_mi + 0.10×S_entropy + 0.10×S_correlation +
      0.10×S_gradient + 0.20×S_temporal + 0.10×S_constraints
```

Each sub-score S ∈ [0, 1]:
- S_value = clip(mean(EV_r, EV_c), 0, 1)
- S_mi = clip(mean_MI_top5_pairs / 0.3, 0, 1)
- S_entropy = clip(mean_ρ_top5 / 0.2, 0, 1)
- S_correlation = clip(1 - mean|R_ij|/0.9, 0, 1) × clip(η/0.1, 0, 1)
- S_gradient = clip(temporal_frac/0.2, 0, 1) × clip(price_grad/0.01, 0, 1)
- S_temporal = clip(TPS/0.1, 0, 1) × clip(1 - attn_entropy_ratio/0.95, 0, 1) × clip(perturbation_effect/0.05, 0, 1)^(1/3)
- S_constraints = clip(1 - behavioral_VR/0.5, 0, 1)

Note: S_temporal is weighted 0.20 (upgraded from 0.10) because temporal planning is the primary capability that improves ALL KPIs simultaneously. It uses the geometric mean of three sub-metrics:
1. TPS (cross-temporal correlation) — does the agent correlate actions with future states?
2. Attention focus (1 - entropy ratio) — does the temporal transformer focus on informative timesteps?
3. Perturbation effect — does removing history actually change actions?

**Interpretation:**
- PHI < 0.1 → Dead policy (random)
- 0.1-0.3 → Failing (some signal, overwhelmed by noise)
- 0.3-0.5 → Learning (features extracted, actions not yet good)
- 0.5-0.7 → Functional (domain-appropriate behavior)
- > 0.7 → Intelligent (thesis-worthy)

---

## Automated Decision Tree

The script outputs a diagnosis based on test results:

```python
def diagnose(results):
    if results['ev_reward'] < 0.1 or results['ev_cost'] < 0.1:
        return "CRITIC BROKEN: Value function cannot predict returns. " \
               "Fix: increase critic LR, add critic update epochs."

    if results['mean_mi_top5'] < 0.02:
        if results['price_gradient'] < 0.001:
            return "ENCODER DEAD: Price→action gradient is zero. " \
                   "Fix: check global encoder → broadcast → per-node integration."
        else:
            return "POLICY TOO NOISY: Gradients exist but MI is low. " \
                   "Fix: reduce PolicyStd, increase training epochs."

    if results['mean_building_corr'] > 0.9:
        return "GCN DEAD: All buildings take identical actions. " \
               "Fix: check adjacency learning, gated fusion weights."

    if results['headroom_reward'] < 0:
        return "WORSE THAN ZERO-ACTION: Lambda erasing reward signal. " \
               "Fix: enable standardized_cost_adv, raise cost_limit, " \
               "drop non-binding constraints."

    if results['temporal_fraction'] < 0.01:
        return "TEMPORAL DEAD: History has no effect on actions. " \
               "Fix: check temporal transformer → concat projection."

    if results['behavioral_vr'] > 0.5:
        return "CONSTRAINT FAILURE: Agent causes avoidable violations. " \
               "Fix: dense cost shaping, increase cost critic capacity."

    return "ARCHITECTURE WORKS: Needs more training epochs with lambda cap."
```

---

## Known Issues to Fix (Informed by Analysis)

These 5 fixes should be applied based on diagnostic results:

1. **`standardized_cost_adv: true`** — z-score cost advantages (1-line config change)
   - Root cause: cost advantage σ_c ≈ 10 vs reward advantage σ_r = 1 → 10:1 gradient imbalance
   - Location: `ppo_lag_multi.py` line 393 (already implemented, just disabled)

2. **Raise `cost_limit` to ~600K** — match structural floor
   - Root cause: 590K zero-action cost vs 400K limit → 190K gap drives lambda up
   - Alternative: drop C4/C5 from CMDP cost (never binding at 5 buildings)

3. **Enable `CITYLEARN_EV_DENSE_COST_SCALE > 0`** — spread EV cost over connection window
   - Root cause: sparse departure-time-only cost is invisible during most of training

4. **Enable `CITYLEARN_C3_CONTROLLABLE=1`** — remove structural C3 floor
   - Root cause: 12kW PV buildings structurally violate 4.6kW building limit

5. **Z-score BOTH advantage types** in the PPOLag loss
   - Most important fix, directly addresses the gradient imbalance

---

## Dependencies

- Python 3.10+, PyTorch, NumPy, Matplotlib
- scikit-learn (for `mutual_info_regression` KSG estimator)
- OmniSafe (for environment + actor reconstruction)
- Existing: `stems_encoder_v3.py`, `schema_index.py`, `safety_env_v3.py`, `omni_env_v2.py`

## Estimated Runtime

- Rollout: ~10 min (8,759 steps)
- Zero-action baseline: ~5 min
- Tests 1-8: ~30 min total
- Figure generation: ~2 min
- **Total: ~50 min per checkpoint**
