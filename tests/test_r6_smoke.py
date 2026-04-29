#!/usr/bin/env python3
"""R6 smoke test: verify P0+P1 work together for 500 steps without errors."""
import os, sys
import numpy as np

os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, os.getcwd())

if "CITYLEARN_SCHEMA" not in os.environ:
    os.environ["CITYLEARN_SCHEMA"] = os.path.join(
        os.getcwd(), "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
    )
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "29.6915"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"
os.environ["CITYLEARN_C3_CONTROLLABLE"] = "1"   # P1
os.environ["CITYLEARN_SPATIAL_OBS"] = "1"        # P0

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env import CityLearnSafetyEnv
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.stems_obs_wrapper import SpatialGraphFeaturesWrapper

base = CityLearnEnv(schema=os.environ["CITYLEARN_SCHEMA"], central_agent=True)
num_buildings = len(base.buildings)
safety = CityLearnSafetyEnv(base)
forecast = ForecastObsWrapper(safety, forecast_horizon=24)
spatial = SpatialGraphFeaturesWrapper(forecast, num_buildings=num_buildings, p_building_max=4.6083)

obs, info = spatial.reset()
if isinstance(safety.action_space, list):
    action_dim = sum(int(np.prod(s.shape)) for s in safety.action_space)
else:
    action_dim = safety.action_space.shape[0]

print(f"Obs dim: {len(np.asarray(obs).ravel())}")
print(f"Action dim: {action_dim}")
print(f"Buildings: {num_buildings}")

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

print(f"\n=== R6 Smoke Test Results (500 steps, {num_buildings} buildings) ===")
print(f"C3 cost total:          {sum(c3_costs):.4f}")
print(f"Structural removed:     {sum(structural_removed):.4f}")
print(f"Errors:                 {errors}")
print(f"Obs dim:                {len(np.asarray(obs).ravel())}")

assert errors == 0, f"{errors} errors found"
assert sum(structural_removed) > 0, "Expected some structural cost removal"
print("\nPASS: R6 smoke test complete — no errors, structural cost removed")
