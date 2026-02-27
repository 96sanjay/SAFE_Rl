#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, "/home/extra-storage/THESIS/CityLearn")
sys.path.insert(0, "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/")

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from agents.intelligent_rbc_with_temp import RBCAgentWithTemp

SCHEMA = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/citylearn_challenge_2023_phase_2_online_evaluation_3/schema.json"

os.environ["CITYLEARN_KPI_RUN_NAME"] = "rbc_heatwave"

base_env = CityLearnEnv(schema=SCHEMA)

# Find heat wave periods (>35°C)
outdoor = np.array(base_env.buildings[0].weather.outdoor_dry_bulb_temperature)
heatwave_steps = np.where(outdoor > 35)[0]

print(f"Heat wave steps: {len(heatwave_steps)} ({len(heatwave_steps)/len(outdoor)*100:.1f}%)")

env = CityLearnSafetyEnvV3(base_env)
agent = RBCAgentWithTemp(ev_mode="greedy", temp_deadband=0.5)

obs, info = env.reset()
agent.reset(env)

for step in range(len(outdoor)):
    action = agent.act(obs, info)
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break

env.close()
print("\n✅ Check: runs/kpi_logs/rbc_heatwave.csv")
