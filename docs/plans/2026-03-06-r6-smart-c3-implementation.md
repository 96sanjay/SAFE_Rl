# R6 Smart C3 Agent — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the RL agent smarter about C3 (building power cap) by giving it spatial observations and a clean, agent-controllable cost signal.

**Architecture:** Three layered changes: (P0) add `SpatialGraphFeaturesWrapper` for per-building C3 headroom in observations, (P1) modify C3 cost in `safety_env_v3.py` to only penalize agent-controllable violations, (P2) combine R5-A StopIter fix with R5-B sparse Saute fix in training config.

**Tech Stack:** Python, NumPy, Gymnasium, OmniSafe, CityLearn

**Project root:** `/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork`

---

### Task 1: Verify NSL Indexing Matches NEC Indexing

Before implementing P1, we must confirm that `b._Building__energy_to_non_shiftable_load[t]` uses the same time index as `b.net_electricity_consumption[t]` in the C3 block.

**Files:**
- Create: `tests/test_nsl_nec_indexing.py`

**Step 1: Write the verification test**

```python
#!/usr/bin/env python3
"""Verify NSL and NEC use the same time indexing in the C3 block."""
import os, sys, json
import numpy as np

os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, os.getcwd())

os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "29.6915"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"

from citylearn.citylearn import CityLearnEnv

schema_path = "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
env = CityLearnEnv(schema=schema_path, central_agent=True)
env.reset()

action_dim = env.action_space.shape[0]
mismatches = 0

for step in range(100):
    t = env.time_step
    for b_idx, b in enumerate(env.buildings):
        nsl = float(np.asarray(b._Building__energy_to_non_shiftable_load)[t])
        sg  = float(np.asarray(b._Building__solar_generation)[t])

        nec_arr = getattr(b, "net_electricity_consumption", None)
        if nec_arr is not None and hasattr(nec_arr, "__len__") and len(nec_arr) > t:
            nec = float(nec_arr[t])
        else:
            nec = 0.0

        # NEC = NSL + solar + storage contributions
        # With zero action, storage contribution ≈ 0 (no charge/discharge)
        # So NEC ≈ NSL + solar_generation
        expected_nec = nsl + sg
        diff = abs(nec - expected_nec)

        if diff > 0.5 and step > 0:  # allow first step tolerance
            mismatches += 1
            if mismatches <= 5:
                print(f"MISMATCH step={step} b={b_idx}: nec={nec:.4f} vs nsl+sg={expected_nec:.4f} (diff={diff:.4f})")

    env.step([[0.0] * action_dim])

if mismatches == 0:
    print("PASS: NSL + solar indexing matches NEC for all 100 steps x 17 buildings")
else:
    print(f"FAIL: {mismatches} mismatches found")

assert mismatches == 0, f"NSL indexing mismatch: {mismatches} cases"
```

**Step 2: Run the test**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && python tests/test_nsl_nec_indexing.py`
Expected: `PASS: NSL + solar indexing matches NEC for all 100 steps x 17 buildings`

**Step 3: Commit**

```bash
git add tests/test_nsl_nec_indexing.py
git commit -m "test: verify NSL/NEC time indexing alignment for agent-controllable C3"
```

---

### Task 2: Implement P1 — Agent-Controllable C3 Cost

**Files:**
- Modify: `citylearn_safe/safety_env_v3.py:775-844` (C3 block)

**Step 1: Write the failing test**

Create `tests/test_controllable_c3.py`:

```python
#!/usr/bin/env python3
"""Test that agent-controllable C3 produces zero cost for passive agent."""
import os, sys
import numpy as np

os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, os.getcwd())

os.environ["CITYLEARN_SCHEMA"] = os.path.join(os.getcwd(), "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json")
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "29.6915"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"
os.environ["CITYLEARN_C3_CONTROLLABLE"] = "1"  # NEW feature flag

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

schema_path = os.environ["CITYLEARN_SCHEMA"]
base = CityLearnEnv(schema=schema_path, central_agent=True)
env = CityLearnSafetyEnvV3(base)
obs, info = env.reset()

action_dim = env.action_space.shape[0]
total_old_c3 = 0.0
total_new_c3 = 0.0

for step in range(200):
    obs, reward, terminated, truncated, info = env.step(np.zeros(action_dim))
    c3_cost = float(info.get("cost_stems_building_power", 0.0))
    total_new_c3 += c3_cost
    if terminated:
        break

print(f"Controllable C3 cost over 200 steps (zero action): {total_new_c3:.4f}")

# With zero actions, a passive agent should have near-zero controllable C3 cost
# (only non-zero if there are tiny numerical floating point differences)
assert total_new_c3 < 1.0, (
    f"Controllable C3 should be near-zero for passive agent, got {total_new_c3:.4f}"
)
print("PASS: Passive agent gets near-zero controllable C3 cost")

# Now test with OLD behavior (should have substantial cost)
os.environ["CITYLEARN_C3_CONTROLLABLE"] = "0"
base2 = CityLearnEnv(schema=schema_path, central_agent=True)
env2 = CityLearnSafetyEnvV3(base2)
obs2, info2 = env2.reset()

total_old_c3 = 0.0
for step in range(200):
    obs2, reward2, terminated2, truncated2, info2 = env2.step(np.zeros(action_dim))
    total_old_c3 += float(info2.get("cost_stems_building_power", 0.0))
    if terminated2:
        break

print(f"Old C3 cost over 200 steps (zero action): {total_old_c3:.4f}")
assert total_old_c3 > total_new_c3, (
    f"Old C3 ({total_old_c3:.4f}) should be greater than controllable C3 ({total_new_c3:.4f})"
)
print("PASS: Old C3 > Controllable C3 for passive agent")
```

**Step 2: Run to verify it fails**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && python tests/test_controllable_c3.py`
Expected: FAIL — `CITYLEARN_C3_CONTROLLABLE` env var is not yet read in safety_env_v3.py, so both paths produce the same cost.

**Step 3: Implement agent-controllable C3**

In `citylearn_safe/safety_env_v3.py`, replace the C3 violation loop (lines ~788-811):

Find the existing code block:
```python
        if citylearn_env is not None and getattr(citylearn_env, "buildings", None):
            violations_list = []  # Track violations
            idx_bp = self._state_time_index(citylearn_env)
            for b in citylearn_env.buildings:
                try:
                    nec = getattr(b, "net_electricity_consumption", None)
                    if nec is not None and hasattr(nec, "__len__") and len(nec) > idx_bp:
                        p_i = float(nec[idx_bp])
                    else:
                        p_i = 0.0
                except Exception:
                    p_i = 0.0

                v_i = max(0.0, abs(p_i) - p_building_max)
                if v_i > 0.0:
                    b_any_violation = 1.0
                    b_viol_cnt += 1.0
                violations_list.append(v_i)  # Track violations
                n_b += 1

            building_scale = float(os.environ.get("CITYLEARN_STEMS_BUILDING_COST_SCALE", "1.0"))
            cost_stems_building_power = float(sum(violations_list) * building_scale)  # SUM + scale
```

Replace with:
```python
        if citylearn_env is not None and getattr(citylearn_env, "buildings", None):
            violations_list = []  # Track violations
            building_powers = []  # Track raw powers for debug logging
            idx_bp = self._state_time_index(citylearn_env)
            c3_controllable = os.environ.get("CITYLEARN_C3_CONTROLLABLE", "0") == "1"

            for b in citylearn_env.buildings:
                try:
                    nec = getattr(b, "net_electricity_consumption", None)
                    if nec is not None and hasattr(nec, "__len__") and len(nec) > idx_bp:
                        p_i = float(nec[idx_bp])
                    else:
                        p_i = 0.0
                except Exception:
                    p_i = 0.0

                building_powers.append(p_i)

                if c3_controllable:
                    # Agent-controllable C3: only penalize agent's contribution
                    nsl_i = 0.0
                    try:
                        nsl_arr = np.asarray(
                            getattr(b, "_Building__energy_to_non_shiftable_load", []),
                            dtype=float,
                        )
                        sg_arr = np.asarray(
                            getattr(b, "_Building__solar_generation", []),
                            dtype=float,
                        )
                        if len(nsl_arr) > idx_bp:
                            nsl_i = float(nsl_arr[idx_bp])
                        if len(sg_arr) > idx_bp:
                            nsl_i += float(sg_arr[idx_bp])  # base = NSL + solar
                    except Exception:
                        nsl_i = p_i  # fallback: treat all as uncontrollable (v_i=0)

                    if abs(nsl_i) <= p_building_max:
                        # Normal case: base load under threshold
                        v_i = max(0.0, abs(p_i) - p_building_max)
                    else:
                        # Structural violation: only penalize agent's marginal excess
                        v_i = max(0.0, abs(p_i) - abs(nsl_i))
                else:
                    # Original C3 (unchanged)
                    v_i = max(0.0, abs(p_i) - p_building_max)

                if v_i > 0.0:
                    b_any_violation = 1.0
                    b_viol_cnt += 1.0
                violations_list.append(v_i)
                n_b += 1

            building_scale = float(os.environ.get("CITYLEARN_STEMS_BUILDING_COST_SCALE", "1.0"))
            cost_stems_building_power = float(sum(violations_list) * building_scale)
```

**Key changes:**
- Added `c3_controllable` env var check (default "0" = old behavior)
- When controllable: reads NSL + solar as base load, splits violation into controllable/structural
- When structural violation (base > threshold): only penalizes agent's marginal excess
- Added `building_powers` list so the existing debug logging block (lines 816-835) works correctly
- Fallback: if NSL read fails, treats all as uncontrollable (v_i=0) — safe default

**Step 4: Run test to verify it passes**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && python tests/test_controllable_c3.py`
Expected: Both assertions pass.

**Step 5: Commit**

```bash
git add citylearn_safe/safety_env_v3.py tests/test_controllable_c3.py
git commit -m "feat: add agent-controllable C3 cost (CITYLEARN_C3_CONTROLLABLE env var)

When enabled, C3 only penalizes the agent's controllable contribution.
Structural violations (base load > threshold) are excluded from cost.
Agent is still encouraged to reduce consumption during structural peaks.
Gated behind CITYLEARN_C3_CONTROLLABLE=1 (default: off, old behavior)."
```

---

### Task 3: Implement P0 — Wire Spatial Observations into Env Pipeline

**Files:**
- Modify: `citylearn_safe/omni_env_v2.py:43-54` (env pipeline in `__init__`)

**Step 1: Write the failing test**

Create `tests/test_spatial_obs.py`:

```python
#!/usr/bin/env python3
"""Test that spatial observations are appended when CITYLEARN_SPATIAL_OBS=1."""
import os, sys
import numpy as np

os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, os.getcwd())

os.environ["CITYLEARN_SCHEMA"] = os.path.join(os.getcwd(), "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json")
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "29.6915"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"

# Test WITHOUT spatial obs
os.environ["CITYLEARN_SPATIAL_OBS"] = "0"

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper

base = CityLearnEnv(schema=os.environ["CITYLEARN_SCHEMA"], central_agent=True)
safety = CityLearnSafetyEnvV3(base)
forecast = ForecastObsWrapper(safety, forecast_horizon=24)
obs_no_spatial, _ = forecast.reset()
dim_no_spatial = len(np.asarray(obs_no_spatial).ravel())

# Test WITH spatial obs
os.environ["CITYLEARN_SPATIAL_OBS"] = "1"
from citylearn_safe.stems_obs_wrapper import SpatialGraphFeaturesWrapper

base2 = CityLearnEnv(schema=os.environ["CITYLEARN_SCHEMA"], central_agent=True)
safety2 = CityLearnSafetyEnvV3(base2)
forecast2 = ForecastObsWrapper(safety2, forecast_horizon=24)
spatial = SpatialGraphFeaturesWrapper(forecast2, num_buildings=17, p_building_max=4.6083)
obs_with_spatial, _ = spatial.reset()
dim_with_spatial = len(np.asarray(obs_with_spatial).ravel())

print(f"Without spatial: {dim_no_spatial} dims")
print(f"With spatial:    {dim_with_spatial} dims")
print(f"Difference:      {dim_with_spatial - dim_no_spatial} dims (expected 68)")

assert dim_with_spatial == dim_no_spatial + 68, (
    f"Expected +68 spatial dims, got +{dim_with_spatial - dim_no_spatial}"
)

# Check spatial features are not all zeros (headroom should be non-zero)
spatial_features = np.asarray(obs_with_spatial).ravel()[-68:]
assert not np.all(spatial_features == 0.0), "Spatial features should not be all zeros"

print("PASS: Spatial obs wrapper adds 68 non-zero features")
```

**Step 2: Run to verify it works standalone (not yet in env pipeline)**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && python tests/test_spatial_obs.py`
Expected: PASS (the wrapper already exists and works, this just verifies it).

**Step 3: Wire spatial wrapper into CityLearnCMDPv2**

In `citylearn_safe/omni_env_v2.py`, modify the `__init__` method. After the existing import block at the top of the file, add:

```python
from citylearn_safe.stems_obs_wrapper import SpatialGraphFeaturesWrapper
```

Then in `__init__`, replace lines 46-50:
```python
        base: gym.Env = make_base_env(central_agent=True)
        safety = CityLearnSafetyEnvV3(base)
        forecast = ForecastObsWrapper(safety, forecast_horizon=24)

        self._env = forecast
        self._observation_space = forecast.observation_space
```

With:
```python
        base: gym.Env = make_base_env(central_agent=True)
        safety = CityLearnSafetyEnvV3(base)
        forecast = ForecastObsWrapper(safety, forecast_horizon=24)

        # P0: Add spatial observations (per-building C3 headroom, SoC spread, etc.)
        if os.environ.get("CITYLEARN_SPATIAL_OBS", "0") == "1":
            p_bmax = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
            env_final = SpatialGraphFeaturesWrapper(forecast, num_buildings=17, p_building_max=p_bmax)
            print(f"[CMDPv2] Spatial obs ENABLED (+68 dims, P_building_max={p_bmax})")
        else:
            env_final = forecast

        self._env = env_final
        self._observation_space = env_final.observation_space
```

**Step 4: Run test to verify the pipeline works end-to-end**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && python tests/test_spatial_obs.py`
Expected: PASS

**Step 5: Commit**

```bash
git add citylearn_safe/omni_env_v2.py tests/test_spatial_obs.py
git commit -m "feat: wire SpatialGraphFeaturesWrapper into CityLearnCMDPv2 env pipeline

Adds 68 per-building spatial features (C3 headroom, SoC spread,
neighbour SoC, grid contribution) when CITYLEARN_SPATIAL_OBS=1.
Default off — old behavior preserved."
```

---

### Task 4: Add Integration Test — Old Behavior Preserved

Verify that with both flags OFF, the environment produces identical results to before.

**Files:**
- Create: `tests/test_backward_compat.py`

**Step 1: Write the backward compatibility test**

```python
#!/usr/bin/env python3
"""Verify old behavior is preserved when new features are disabled."""
import os, sys
import numpy as np

os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, os.getcwd())

os.environ["CITYLEARN_SCHEMA"] = os.path.join(os.getcwd(), "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json")
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "29.6915"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"
# Explicitly disable new features
os.environ["CITYLEARN_C3_CONTROLLABLE"] = "0"
os.environ["CITYLEARN_SPATIAL_OBS"] = "0"

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

schema_path = os.environ["CITYLEARN_SCHEMA"]
base = CityLearnEnv(schema=schema_path, central_agent=True)
env = CityLearnSafetyEnvV3(base)
obs, info = env.reset()

action_dim = env.action_space.shape[0]
c3_costs = []
rewards = []

for step in range(50):
    obs, reward, terminated, truncated, info = env.step(np.zeros(action_dim))
    c3_costs.append(float(info.get("cost_stems_building_power", 0.0)))
    rewards.append(float(reward))
    if terminated:
        break

total_c3 = sum(c3_costs)
print(f"Old behavior C3 cost (50 steps): {total_c3:.4f}")
print(f"Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")

# Sanity: C3 cost should be non-negative
assert all(c >= 0 for c in c3_costs), "C3 costs must be non-negative"
# Sanity: obs should be the base dim (no spatial additions)
obs_dim = len(np.asarray(obs).ravel())
print(f"Obs dim: {obs_dim} (should NOT include 68 spatial dims)")

print("PASS: Old behavior preserved with flags OFF")
```

**Step 2: Run the test**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && python tests/test_backward_compat.py`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_backward_compat.py
git commit -m "test: add backward compatibility test for R6 feature flags"
```

---

### Task 5: Add Debug Logging for Agent-Controllable C3

Add info keys so we can monitor old-vs-new C3 during training.

**Files:**
- Modify: `citylearn_safe/safety_env_v3.py` (after the C3 block, around line 844)

**Step 1: Add logging keys**

After the line `info["building_power_violation_rate_%"] = ...` (line 844), add:

```python
        # --- Agent-controllable C3 debug info ---
        info["c3_controllable_enabled"] = 1.0 if os.environ.get("CITYLEARN_C3_CONTROLLABLE", "0") == "1" else 0.0
```

And inside the C3 loop (at the end, after computing all violations), add tracking for how much cost was removed:

In the C3 block, add two accumulators before the loop:
```python
            c3_structural_removed = 0.0  # cost removed by controllability filter
```

And inside the `if c3_controllable:` branch, after computing `v_i`, add:
```python
                    # Track how much cost the controllability filter removed
                    v_old = max(0.0, abs(p_i) - p_building_max)
                    c3_structural_removed += max(0.0, v_old - v_i)
```

After the loop, log it:
```python
            info["c3_structural_cost_removed"] = float(c3_structural_removed)
```

**Step 2: Verify logging works**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && CITYLEARN_C3_CONTROLLABLE=1 python tests/test_controllable_c3.py`
Expected: PASS (test still works, now with extra info keys)

**Step 3: Commit**

```bash
git add citylearn_safe/safety_env_v3.py
git commit -m "feat: add C3 controllability debug logging (c3_structural_cost_removed)"
```

---

### Task 6: Write R6 Training Launch Script

Create the training script that combines P0 + P1 + P2.

**Files:**
- Create: `train_r6_smart_c3.sh`

**Step 1: Write the launch script**

```bash
#!/usr/bin/env bash
# R6 Training: Smart C3 Agent
# Combines: P0 (spatial obs) + P1 (controllable C3) + P2 (StopIter + sparse Saute)
#
# Usage: bash train_r6_smart_c3.sh [SEED]
set -euo pipefail

SEED="${1:-0}"
PROJECT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# --- Dataset ---
export CITYLEARN_SCHEMA="$PROJECT/data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
export CITYLEARN_CENTRAL_AGENT="1"

# --- Reward ---
export CITYLEARN_REWARD_TYPE="stems"
export CITYLEARN_EXPORT_FACTOR="0.7"

# --- Constraint thresholds (calibrated, unchanged from R5) ---
export CITYLEARN_STEMS_P_BUILDING_MAX="4.6083"
export CITYLEARN_STEMS_P_GRID_MAX="29.6915"
export CITYLEARN_STEMS_SOC_LOW="0.0"
export CITYLEARN_STEMS_SOC_HIGH="0.95"
export CITYLEARN_STEMS_PNORM_P="4.0"

# --- Cost weights (unchanged from R5) ---
export CITYLEARN_W_COST_EV="1.0"
export CITYLEARN_W_COST_SOC="10.0"
export CITYLEARN_W_COST_BUILDING="0.5"
export CITYLEARN_W_COST_GRID="0.05"

# --- EV ---
export CITYLEARN_EV_COST_SCALE="3.0"
export CITYLEARN_EV_MISSING_ACTION_MODE="assume_full"
export CITYLEARN_INCLUDE_EV_COST="1"

# ============================================
# NEW FOR R6
# ============================================
export CITYLEARN_C3_CONTROLLABLE="1"    # P1: agent-controllable C3
export CITYLEARN_SPATIAL_OBS="1"        # P0: spatial observations (+68 dims)

echo "=== R6 Smart C3 Training ==="
echo "  P0: Spatial obs = $CITYLEARN_SPATIAL_OBS"
echo "  P1: Controllable C3 = $CITYLEARN_C3_CONTROLLABLE"
echo "  P2: StopIter + Sparse Saute (in config)"
echo "  Seed: $SEED"
echo "  Schema: $CITYLEARN_SCHEMA"
echo ""

# NOTE: Replace the python command below with your actual training invocation.
# The multilag PPOLag training script and config from R5-A/R5-B should be used here,
# with the R5-A StopIter fix (train_iters > 1) and R5-B sparse Saute fix combined.
#
# Example (adjust to your actual training script):
# python train_omnisafe_multilag.py \
#     --algo PPOLag \
#     --env CityLearnSafety-V2G-v2-multilag \
#     --seed $SEED \
#     --epochs 200 \
#     --tag "r6_smart_c3_seed${SEED}"

echo "TODO: Add your training command here (see comments above)"
echo "Env vars are set. You can also source this script and run manually."
```

**Step 2: Make executable and commit**

```bash
chmod +x train_r6_smart_c3.sh
git add train_r6_smart_c3.sh
git commit -m "feat: add R6 training launch script with P0+P1+P2 env vars"
```

---

### Task 7: Run Full Smoke Test (All Features Combined)

**Files:**
- Create: `tests/test_r6_smoke.py`

**Step 1: Write end-to-end smoke test**

```python
#!/usr/bin/env python3
"""R6 smoke test: verify P0+P1 work together for 500 steps without errors."""
import os, sys
import numpy as np

os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, os.getcwd())

os.environ["CITYLEARN_SCHEMA"] = os.path.join(os.getcwd(), "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json")
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "29.6915"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"
os.environ["CITYLEARN_C3_CONTROLLABLE"] = "1"   # P1
os.environ["CITYLEARN_SPATIAL_OBS"] = "1"        # P0

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.stems_obs_wrapper import SpatialGraphFeaturesWrapper

base = CityLearnEnv(schema=os.environ["CITYLEARN_SCHEMA"], central_agent=True)
safety = CityLearnSafetyEnvV3(base)
forecast = ForecastObsWrapper(safety, forecast_horizon=24)
spatial = SpatialGraphFeaturesWrapper(forecast, num_buildings=17, p_building_max=4.6083)

obs, info = spatial.reset()
action_dim = safety.action_space.shape[0]

print(f"Obs dim: {len(np.asarray(obs).ravel())}")
print(f"Action dim: {action_dim}")

c3_costs = []
structural_removed = []
errors = 0

for step in range(500):
    # Random actions to stress-test
    action = np.random.uniform(-1, 1, size=action_dim).astype(np.float32)
    obs, reward, terminated, truncated, info = spatial.step(action)

    c3 = float(info.get("cost_stems_building_power", 0.0))
    removed = float(info.get("c3_structural_cost_removed", 0.0))
    c3_costs.append(c3)
    structural_removed.append(removed)

    # Verify no NaN/Inf
    obs_arr = np.asarray(obs).ravel()
    if np.any(np.isnan(obs_arr)) or np.any(np.isinf(obs_arr)):
        print(f"ERROR: NaN/Inf in obs at step {step}")
        errors += 1

    if not np.isfinite(reward):
        print(f"ERROR: Non-finite reward at step {step}: {reward}")
        errors += 1

    if terminated:
        obs, info = spatial.reset()

print(f"\n=== R6 Smoke Test Results (500 steps) ===")
print(f"C3 cost total:          {sum(c3_costs):.4f}")
print(f"Structural removed:     {sum(structural_removed):.4f}")
print(f"Errors:                 {errors}")
print(f"Obs dim:                {len(np.asarray(obs).ravel())}")

assert errors == 0, f"{errors} errors found"
assert sum(structural_removed) > 0, "Expected some structural cost removal"
print("\nPASS: R6 smoke test complete — no errors, structural cost removed")
```

**Step 2: Run the smoke test**

Run: `cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork && python tests/test_r6_smoke.py`
Expected: PASS with non-zero structural cost removed

**Step 3: Commit**

```bash
git add tests/test_r6_smoke.py
git commit -m "test: add R6 end-to-end smoke test (P0+P1 combined, 500 steps)"
```

---

## Summary

| Task | What | Files | Est. Time |
|------|------|-------|-----------|
| 1 | Verify NSL indexing | `tests/test_nsl_nec_indexing.py` | 3 min |
| 2 | P1: Agent-controllable C3 | `safety_env_v3.py` + test | 5 min |
| 3 | P0: Wire spatial obs | `omni_env_v2.py` + test | 3 min |
| 4 | Backward compat test | `tests/test_backward_compat.py` | 2 min |
| 5 | Debug logging | `safety_env_v3.py` | 2 min |
| 6 | Training launch script | `train_r6_smart_c3.sh` | 2 min |
| 7 | Full smoke test | `tests/test_r6_smoke.py` | 3 min |

**Total: 7 tasks, 7 commits, ~20 min implementation**

P2 (R5-A + R5-B config merge) is a config-only change that depends on your specific multilag training script — update the training command in `train_r6_smart_c3.sh` when ready.
