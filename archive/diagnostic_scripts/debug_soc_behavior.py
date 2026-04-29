import os
import numpy as np
import pandas as pd
from citylearn.citylearn import CityLearnEnv
try: from citylearn.agents import RBC as CityLearnRBC
except: from citylearn.agents.rbc import RBC as CityLearnRBC

def get_env():
    schema = os.environ.get("CITYLEARN_SCHEMA")
    return CityLearnEnv(schema=schema)

def run_debug():
    print(f"{'Hour':<5} | {'No Control SoC':<15} | {'Default RBC SoC':<15} | {'Action (Default)':<15}")
    print("-" * 60)

    # 1. Run No Control
    env1 = get_env()
    env1.reset()
    
    # 2. Run Default RBC
    env2 = get_env()
    agent2 = CityLearnRBC(env2)
    obs2 = env2.reset()
    if isinstance(obs2, tuple): obs2 = obs2[0]

    # Run for 24 Hours
    for t in range(24):
        # No Control Step
        zero_action = [ [0.0]*b.action_space.shape[0] for b in env1.buildings ]
        env1.step(zero_action)
        soc1 = env1.buildings[0].electrical_storage.soc[t]

        # Default RBC Step
        act2 = agent2.predict(obs2)
        res = env2.step(act2)
        if len(res) == 5: obs2, _, _, _, _ = res
        else: obs2, _, _, _ = res
        
        soc2 = env2.buildings[0].electrical_storage.soc[t]
        act_val = act2[0][0] # Action for first building

        print(f"{t:<5} | {soc1:<15.4f} | {soc2:<15.4f} | {act_val:<15.4f}")

if __name__ == "__main__":
    run_debug()
