#!/usr/bin/env python3
"""Evaluate agent using KPI logger for consistent metrics"""
import os
import sys
sys.path.insert(0, "/home/extra-storage/THESIS/CityLearn")
sys.path.insert(0, "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/")

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from agents.intelligent_rbc_with_temp import RBCAgentWithTemp

# Config
SCHEMA = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/citylearn_challenge_2023_phase_2_online_evaluation_3/schema.json"
NUM_EPISODES = 5

# Set KPI logger name (IMPORTANT: unique per agent!)
os.environ["CITYLEARN_KPI_RUN_NAME"] = "rbc_baseline"
os.environ["CITYLEARN_KPI_FLUSH_EVERY_STEP"] = "0"  # Flush every 100 steps

print("="*60)
print("RBC EVALUATION WITH KPI LOGGER")
print("="*60)

# Create env (KPI logger auto-initializes)
base_env = CityLearnEnv(schema=SCHEMA)
env = CityLearnSafetyEnvV3(base_env)

# Create agent
agent = RBCAgentWithTemp(ev_mode="greedy", temp_deadband=0.5)

# Run episodes
for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    agent.reset(env)
    done = False
    step = 0
    
    print(f"\nEpisode {ep+1}/{NUM_EPISODES}")
    
    while not done:
        action = agent.act(obs, info)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        step += 1
        
        if step % 1000 == 0:
            print(f"  Step {step}")
    
    print(f"  Completed: {step} steps")

env.close()

print("\n" + "="*60)
print("✅ DONE - Check KPI logs:")
print("  runs/kpi_logs/rbc_baseline.csv")
print("="*60)
