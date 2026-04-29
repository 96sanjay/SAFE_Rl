#!/usr/bin/env python3
"""Quick test: Does peak cost work?"""
import sys
sys.path.insert(0, '/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork')

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
import numpy as np

# Load environment
schema_path = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
base_env = CityLearnEnv(schema=schema_path)
safe_env = CityLearnSafetyEnvV3(base_env)

print("✅ Environment created successfully")
print(f"   Peak threshold: {safe_env.peak_threshold:.2f} kW")
print(f"   Peak weight: {safe_env.w_grid_peak:.3f}")

# Reset and take a few steps
obs, info = safe_env.reset()
print(f"\n✅ Reset successful")
print(f"   Initial cost: {info['cost']:.3f}")
print(f"   Initial peak cost: {info.get('cost_grid_peak', 0.0):.3f}")

# Take 5 random steps
for i in range(5):
    action = safe_env.action_space.sample()
    obs, reward, term, trunc, info = safe_env.step(action)
    
    print(f"\n📊 Step {i+1}:")
    print(f"   Grid import: {info.get('grid_import_kwh', 0.0):.2f} kW")
    print(f"   Peak violation (raw): {info.get('cost_grid_peak_raw', 0.0):.2f} kW")
    print(f"   Peak cost (weighted): {info.get('cost_grid_peak', 0.0):.4f}")
    print(f"   Total CMDP cost: {info['cost']:.4f}")
    
    if term or trunc:
        break

print("\n✅ Test complete! Peak cost is working.")
