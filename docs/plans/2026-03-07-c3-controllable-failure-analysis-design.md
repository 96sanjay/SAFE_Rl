# C3 Controllable Failure Analysis & Fix Design

**Date:** 2026-03-07
**Context:** R6 comparison runs (March 6, 2026) — PPOLag with C3 controllable (P1) + spatial obs (P0)
**Objective:** Understand why 50-epoch training with C3 controllable showed no significant improvement

---

## Root Cause Analysis

### Finding 1: C3 Is Invisible in the Cost Signal (Primary Cause)

The total CMDP cost seen by OmniSafe is dominated by C2 battery SoC:

| Constraint | Raw Cost (8759 steps) | Weight | Weighted Cost | % of Total |
|-----------|----------------------|--------|---------------|-----------|
| C2 Battery SoC | 23,972 | 10.0 | 239,724 | 92.7% |
| C3 Building Power | 33,911 | 0.5 | 16,955 | 6.5% |
| C1 EV Departure | 1,541 | 1.0 | 1,541 | 0.6% |
| C4 Grid Power | 9,967 | 0.05 | 498 | 0.2% |
| **Total** | | | **258,718** | 100% |

The C3 controllable feature removed ~4,800 kWh of structural cost (14% of raw C3).
In weighted terms: 4,800 x 0.5 = 2,400 reduction = **0.93% of total cost**.

The policy optimizer cannot distinguish a 0.93% signal change. C3 is noise.

### Finding 2: R6 Has Broken Policy Updates (StopIter = 1)

Despite `update_iters: 60` in config, R6's `Train/StopIter` was **1 throughout all 50 epochs**.

Comparison of StopIter across runs:
- **R5a (baseline, no P0/P1):** 60 -> 44 -> 30 -> 22 -> ... -> 3 (healthy decay)
- **R5b (intentionally broken):** Always 1
- **R6 (P0 + P1):** Always 1 (same as broken baseline!)
- **R7 (P0 + P1 + reward tuning):** 60 -> 41 -> 31 -> ... -> 4 (healthy)

The spatial observations (P0) altered the gradient landscape such that a single policy update step immediately exceeds `target_kl: 0.10`, triggering KL early stopping. Evidence:
- R6 `Train/PolicyRatio` ~ 1.000 +/- 0.003 (barely moving)
- R5a `Train/PolicyRatio` ~ 1.0 +/- 0.02 (actually learning)
- R7 doesn't have this issue because its reward tuning (alpha_grid=1.0, beta_ramp=2.0) changes gradient magnitudes

### Finding 3: Lagrange Multiplier Explosion (All Runs)

- `cost_limit: 24,400` but minimum achievable cost after 50 epochs: ~87,000-90,000
- Lambda grew from ~9,000 to ~250,000 across all runs
- Per-epoch lambda increase: 0.05 x (90,000 - 24,400) = ~3,280
- After 50 epochs: ~9,000 + 164,000 = ~173,000 (observed: ~250K due to higher early costs)
- Massive lambda overwhelms reward signal, reward degrades in all runs

---

## Data Summary

### R6 Compare: Final Epoch (49) Metrics

| Metric | R5a | R5b | R6 | R7 |
|--------|-----|-----|-----|-----|
| Episode Return | -32,556 | -48,089 | -34,179 | -57,520 |
| Episode Cost | 87,985 | 90,067 | 86,724 | 86,052 |
| Lagrange Lambda | 254,734 | 259,298 | 245,838 | 245,721 |
| StopIter (final) | 3 | 1 | 1 | 4 |
| PolicyRatio | 1.000 | 0.997 | 1.010 | 1.006 |

Key observations:
- R6 cost (86,724) is marginally better than R5a (87,985) -- only 1.4% lower
- R6 reward (-34,179) is worse than R5a (-32,556) due to StopIter bug
- R7 reward is dramatically worse (-57,520) due to aggressive reward tuning

---

## Fix Plan

### Fix 1: Rebalance Cost Weights (Required)

**Goal:** Make C3 a meaningful fraction of the cost signal.

Current weights:
```
W_COST_SOC      = 10.0   (C2 -> 92.7% of signal)
W_COST_BUILDING = 0.5    (C3 -> 6.5% of signal)
W_COST_GRID     = 0.05   (C4 -> 0.2% of signal)
W_COST_EV       = 1.0    (C1 -> 0.6% of signal)
```

Proposed weights (equalized):
```
W_COST_SOC      = 5.0    (C2)
W_COST_BUILDING = 5.0    (C3 -- 10x increase)
W_COST_GRID     = 1.0    (C4)
W_COST_EV       = 1.0    (C1)
```

With these weights, C3 would represent ~35% of total cost, making the controllable improvement visible.

**Alternative:** If C2 must stay at 10.0, set `W_COST_BUILDING = 10.0` to equalize.

### Fix 2: Fix R6 StopIter Bug (Required for R6)

The spatial observations cause immediate KL violation. Options (ranked):

1. **Lower actor LR from 0.0005 to 0.0001** -- smaller gradient steps, less KL divergence
2. **Disable KL early stopping** -- set `kl_early_stop: false` with `update_iters: 10`
3. **Increase target_kl from 0.10 to 0.20** -- more permissive, but risks instability

Recommendation: Option 1 (lower actor LR). R7 shows the architecture works with P0+P1 when gradients are controlled.

### Fix 3: Adjust Cost Limit (Required)

The cost limit must be achievable. If the structural floor is ~87,000:

```yaml
# Current (infeasible):
cost_limit: 24400

# Proposed (tight but achievable):
cost_limit: 70000   # ~80% of structural minimum, allows lambda to stabilize
```

Or better: compute the cost limit from a zero-action baseline evaluation with the new weights.

### Fix 4: Recalibrate Lambda (Recommended)

Reset lambda to prevent inherited explosion:
```yaml
lagrangian_multiplier_init: 1.0     # was 10.0
lambda_lr: 0.01                      # was 0.05, slower adaptation
```

---

## Execution Order

1. Rebalance weights (Fix 1) -- makes C3 visible
2. Fix StopIter (Fix 2) -- enables policy learning with P0+P1
3. Adjust cost limit (Fix 3) -- prevents lambda explosion
4. Recalibrate lambda (Fix 4) -- stable convergence
5. Run zero-action baseline with new weights to establish cost floor
6. Retrain R6 for 50 epochs with all fixes
7. Compare C3 violation % between new R6 and R5a

---

## Expected Outcome

With C3 accounting for ~35% of total cost (instead of 6.5%):
- C3 controllable improvement becomes 5-15% of the relevant cost signal (instead of <1%)
- The cleaner gradient (structural noise removed) should reduce C3 violations by 10-20% vs baseline
- Lambda stabilizes instead of exploding, allowing reward optimization to proceed
- R6 should show better reward AND lower C3 violations than R5a

---

## Files Referenced

- Training configs: `configs/on-policy/r6_compare_*.yaml`
- Cost computation: `citylearn_safe/safety_env_v3.py` (lines 788-851, 1148)
- Training progress: `runs/r6_compare/*/progress.csv`
- Diagnostic data: `runs/r6_compare/r6_diagnosis/`
- C3 structural proof: `c3_structural_proof.py`, `c3_structural_proof_results.csv`
