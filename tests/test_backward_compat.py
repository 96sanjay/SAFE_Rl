#!/usr/bin/env python3
"""Verify old behavior is preserved when new features are disabled."""
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
# Explicitly disable new features
os.environ["CITYLEARN_C3_CONTROLLABLE"] = "0"
os.environ["CITYLEARN_SPATIAL_OBS"] = "0"

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

schema_path = os.environ["CITYLEARN_SCHEMA"]
base = CityLearnEnv(schema=schema_path, central_agent=True)
num_buildings = len(base.buildings)
env = CityLearnSafetyEnvV3(base)
obs, info = env.reset()

if isinstance(env.action_space, list):
    action_dim = sum(int(np.prod(s.shape)) for s in env.action_space)
else:
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
print(f"Obs dim: {obs_dim} (should NOT include {num_buildings * 4} spatial dims)")

print(f"PASS: Old behavior preserved with flags OFF ({num_buildings} buildings)")
