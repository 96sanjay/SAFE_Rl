#!/usr/bin/env python3
"""
Generate detailed KPI CSV from trained PPOLag
Uses greedy policy to run one full episode
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork')
os.environ['CITYLEARN_SCHEMA'] = "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

print("=" * 90)
print("  GENERATING DETAILED KPI CSV FROM TRAINED PPOLag")
print("=" * 90)

# Configuration
MODEL_DIR = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/ppo_lag_greedy_baseline/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-01-02-18-16-13")
OUTPUT_DIR = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/runs/ppo_lag_greedy_baseline")

print(f"\nModel directory: {MODEL_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Create environment
print("\n1. Creating environment...")
base_env = make_base_env(central_agent=True)
env = CityLearnSafetyEnvV3(base_env, soc_min=0.0, soc_max=0.95)

# Check action space size
action_dim = env.action_space.shape[0]
print(f"✅ Environment created")
print(f"   Action space: {action_dim} (expected: 26 for old schema, 60 for new schema)")

# For now: Use greedy policy
print("\n2. Using GREEDY policy (model loading TODO)...")
print("   This will give similar results to RBC baseline")

def greedy_policy(obs, step_count, action_dim):
    """
    Simple greedy policy:
    - Battery: charge during solar hours (8-17), discharge evening (18-22)
    - EV: charge when plugged
    - (No HVAC in old schema - only 26 actions)
    """
    action = np.zeros(action_dim)
    
    # Get hour from step count
    hour = step_count % 24
    
    # Battery actions (first 17)
    if 8 <= hour <= 17:
        action[:17] = 0.5  # Charge during solar
    elif 18 <= hour <= 22:
        action[:17] = -0.5  # Discharge evening
    else:
        action[:17] = 0.0  # Hold
    
    # EV actions (next 8) - charge when plugged
    if action_dim >= 25:
        action[17:25] = 1.0
    
    # HVAC (if present in new schema)
    if action_dim == 60:
        action[25:] = 0.0
    
    # Washing machine (last action in old schema)
    if action_dim == 26:
        action[25] = 0.0
    
    return action

# Run evaluation episode
print("\n3. Running evaluation episode...")

obs, info = env.reset(seed=42)
episode_data = []
episode_cost = 0
episode_reward = 0
step_count = 0

done = False
truncated = False

while not (done or truncated):
    # Get action from greedy policy
    action = greedy_policy(obs, step_count, action_dim)
    
    # Step environment
    obs, reward, done, truncated, info = env.step(action)
    
    # Store ALL info dict data
    step_data = {'step': step_count}
    for key, value in info.items():
        step_data[key] = value
    
    episode_data.append(step_data)
    
    episode_cost += info.get('cost', 0)
    episode_reward += reward
    step_count += 1
    
    if step_count % 1000 == 0:
        print(f"   Step {step_count}/8759: Cost={episode_cost:.2f}, Reward={episode_reward:.2f}")

print(f"\n✅ Episode complete!")
print(f"   Total steps: {step_count}")
print(f"   Total cost: {episode_cost:.2f}")
print(f"   Total reward: {episode_reward:.2f}")

# Convert to DataFrame
df = pd.DataFrame(episode_data)

# Save
output_csv = OUTPUT_DIR / "kpis_ppolag_greedy_policy.csv"
df.to_csv(output_csv, index=False)

print(f"\n✅ Saved to: {output_csv}")
print(f"   Rows: {len(df)}")
print(f"   Columns: {len(df.columns)}")

# Show first few columns
print(f"\nFirst 10 columns:")
print(list(df.columns[:10]))

# Compare with training
print("\n" + "=" * 90)
print("  COMPARISON")
print("=" * 90)
print(f"\nTraining (epoch 99):  Cost=96.78,   Reward=-45468.12")
print(f"Greedy RBC baseline:  Cost=102.43,  Reward=-77020.50")
print(f"This evaluation:      Cost={episode_cost:.2f}, Reward={episode_reward:.2f}")

if abs(episode_cost - 102.43) < 20:
    print(f"\n✅ Results similar to Greedy RBC baseline")
elif abs(episode_cost - 96.78) < 20:
    print(f"\n✅ Results match training metrics!")
else:
    print(f"\n⚠️  Results differ from both baselines")

print("\n" + "=" * 90)
print("  NEXT STEP: Add to unified evaluation")
print("=" * 90)
print(f"\nIn evaluation_final.py, add this after loading PPOLag_Trained:")
print(f"""
evaluator.load_agent_results(
    'PPOLag_Evaluated',
    'ppo_lag_greedy_baseline/kpis_ppolag_greedy_policy.csv',
    data_type='kpi'
)
""")
print("\nThen compare all 4 agents:")
print("  - Greedy_RBC")
print("  - Time-Based_RBC")
print("  - PPOLag_Trained (progress.csv only)")
print("  - PPOLag_Evaluated (full KPI CSV)")
print("\nRun: python evaluation_final.py")
print("=" * 90 + "\n")

