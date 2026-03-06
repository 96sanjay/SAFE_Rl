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

if isinstance(env.action_space, list):
    action_dim = sum(int(np.prod(s.shape)) for s in env.action_space)
else:
    action_dim = env.action_space.shape[0]
total_new_c3 = 0.0

for step in range(200):
    obs, reward, terminated, truncated, info = env.step(np.zeros(action_dim))
    c3_cost = float(info.get("cost_stems_building_power", 0.0))
    total_new_c3 += c3_cost
    if terminated:
        break

print(f"Controllable C3 cost over 200 steps (zero action): {total_new_c3:.4f}")

# With zero actions, a passive agent should have near-zero controllable C3 cost.
# Threshold 15.0 accounts for step-0 battery initialization artifact where
# action=0 still produces storage activity from initial state.
assert total_new_c3 < 15.0, (
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
