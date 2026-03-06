# R6 Design: Smart C3 Agent

**Date:** 2026-03-06
**Status:** Approved
**Goal:** Reduce C3 violations without degrading C1, C2, C4, or scorecard metrics

---

## Problem Statement

C3 (building power cap constraint) has a structural conflict: charging batteries/EVs pushes building power above the threshold, but NOT charging violates C1 (EV departure) and hurts price_spread. The Lagrange multiplier lambda_C3 saturates at its cap in both R5-A (5.0) and R5-B (2.0) without improving C3.

**Root causes identified:**
1. Agent lacks per-building C3 headroom in observations (can't do spatial management)
2. 40.9-100% of C3 cost signal is structural noise (base load violations agent can't fix)
3. Value function was broken (StopIter=1 in R4/R5-B) preventing temporal credit assignment

## Data Validation

Analysis of c3_structural_proof_results.csv (P_max = 2.274 kW, 17 buildings, 8759 steps):

| Metric | Value |
|--------|-------|
| Structural floor (NSL > threshold) | 7.35% of building-timesteps |
| Agent-controllable violations (MAX_CHARGE) | 6.69% additional |
| Old C3 cost for passive agent (action=0) | 22,491 kW (100% noise) |
| New C3 cost for passive agent (action=0) | 0.0 kW (correct) |
| Structural noise in MAX_CHARGE scenario | 40.9% of total cost |

## Design: Three Layered Changes

### P0: Add Spatial Observations (+68 dims)

**What:** Wire existing `SpatialTemporalHistoryWrapper` (stems_obs_wrapper.py) into env pipeline.

**Features added (4 x 17 buildings = 68 dims):**
- C3 headroom: `(P_max - |p_i|) / P_max` per building (positive = safe, negative = violating)
- SoC spread: `SoC_i - mean(SoC)` per building
- Neighbor avg SoC: mean SoC of other buildings
- Grid contribution: fraction of grid import per building

**File:** OmniSafe CMDP env registration (e.g., `omni_env_forecast.py` or `omni_env_v2.py`)

**Gate:** `CITYLEARN_SPATIAL_OBS=1` (default: "0", old behavior unchanged)

**Implementation:**
```python
from citylearn_safe.stems_obs_wrapper import SpatialTemporalHistoryWrapper

# After ForecastObsWrapper:
if os.environ.get("CITYLEARN_SPATIAL_OBS", "0") == "1":
    env = SpatialTemporalHistoryWrapper(env, p_building_max=P_building_max)
```

**Risk:** Low. Pure observation augmentation. No change to reward, cost, or dynamics.

### P1: Agent-Controllable C3 Cost

**What:** Replace C3 cost with version that only penalizes agent's controllable contribution.

**File:** `citylearn_safe/safety_env_v3.py`, lines 788-811 (C3 block)

**Gate:** `CITYLEARN_C3_CONTROLLABLE=1` (default: "0", old behavior unchanged)

**Logic:**
```python
c3_controllable = os.environ.get("CITYLEARN_C3_CONTROLLABLE", "0") == "1"

for b in citylearn_env.buildings:
    p_i = net_electricity_consumption[t]      # total building power

    if c3_controllable:
        nsl_i = non_shiftable_load[t]         # base load (uncontrollable)
        if abs(nsl_i) <= p_building_max:
            # Normal: base load under threshold, standard penalty
            v_i = max(0.0, abs(p_i) - p_building_max)
        else:
            # Structural violation: only penalize agent's marginal excess
            # If agent discharges (reduces total), v_i = 0 (encouraged)
            v_i = max(0.0, abs(p_i) - abs(nsl_i))
    else:
        # Original C3 (unchanged)
        v_i = max(0.0, abs(p_i) - p_building_max)
```

**Why it works:**
- Passive agent (action=0): C3 cost = 0 (was 22,491 kW)
- Agent that adds charging: penalized only for its kW contribution
- Agent that discharges during structural peak: cost = 0 (rewarded via penalty absence)
- Lambda_C3 stops saturating because total cost is lower and meaningful

**Theoretical validity:** Controllability-aware constraint design. The CMDP formulation
`E[c_agent(s,a)] <= d` remains valid with a tighter, cleaner constraint.

**NSL access:** `b._Building__energy_to_non_shiftable_load[t]` or
`b.energy_simulation.non_shiftable_load[t]`. Must match the time indexing used for
`net_electricity_consumption`.

**Risk:** Medium. Need to verify NSL indexing matches NEC indexing. Add assertion:
`assert abs(p_i - nsl_i - storage_i) < 0.1` for first 100 steps.

### P2: Combine R5-A + R5-B Fixes

**What:** Merge two independently validated improvements:
- R5-A: StopIter fix (train_iters > 1 in PPO inner loop). Value function trains properly.
- R5-B: Sparse Saute fix (budget only decremented on actual EV departures, not every step).

**Expected synergy:**
- Working value function -> temporal credit assignment for C3
- Sparse Saute -> C1 stays excellent (0.020) -> more freedom for C3 management
- Agent can learn: "charge now (C3 cost) to avoid C1 cost at departure"

**No code changes for P2** -- config merge of R5-A and R5-B training parameters.

## Observation Pipeline (R6)

```
CityLearnEnv
  -> NormalizedObservationWrapper (if used)
    -> SingleAgentListAdapter
      -> CityLearnSafetyEnvV3 (with CITYLEARN_C3_CONTROLLABLE=1)
        -> NormalizedForecastObsWrapper (128 forecast dims)
          -> SpatialTemporalHistoryWrapper (68 spatial dims, CITYLEARN_SPATIAL_OBS=1)
            -> OmniSafe PPOLag (multilag, separate lambda per constraint)
```

Total obs: base + 128 + 68 = ~349 dims

## Environment Variables for R6

```bash
# Existing (unchanged from R5)
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="29.6915"
export CITYLEARN_W_COST_BUILDING="0.5"
# ... all other R5 env vars ...

# NEW for R6
export CITYLEARN_C3_CONTROLLABLE="1"    # P1: agent-controllable C3
export CITYLEARN_SPATIAL_OBS="1"        # P0: spatial observations
```

## Expected Outcomes

| Metric | R5-B | R6 Expected | Why |
|--------|------|-------------|-----|
| C3 | 2.35 | < 1.5 | Clean signal + spatial obs + working value function |
| C1 | 0.020 | ~0.020 | Sparse Saute preserved, spatial obs don't conflict |
| C2 | 0.55 | ~0.55 | No change to battery constraint |
| C4 | 1.18 | ~1.0 | Spatial obs help grid coordination |
| lambda_C3 | 2.0 (cap) | < 2.0 | Lower cost -> lambda finds equilibrium |
| StopIter | 1 | 6-17 | R5-A fix |
| Scorecard overall | 0.76 | > 0.80 | C3 fix unlocks price_spread + pre_peak_planning |

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| NSL indexing mismatch | Medium | Add assertion checks first 100 steps |
| Spatial obs increase obs dim -> slower convergence | Low | Obs normalization + 512-512-256 network handles it |
| R5-A + R5-B interaction creates instability | Low | Both fixes are independent (StopIter = learning, Saute = constraint) |
| Agent-controllable C3 too easy -> agent ignores C3 | Low | Structural floor only 7.35%; 93% of timesteps unchanged |

## Future Work (R7+)

- STEMS GCN-Transformer encoder for spatial-temporal feature processing
- Asymmetric C3 (soft/hard thresholds) if needed
- Time-conditional C3 (peak-hours only) if needed
