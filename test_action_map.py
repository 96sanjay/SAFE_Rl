import json, os, sys, numpy as np
os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, os.getcwd())
os.environ["CITYLEARN_STEMS_SOC_LOW"] = "0.0"
os.environ["CITYLEARN_STEMS_SOC_HIGH"] = "0.95"
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "3.47"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "35.76"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"
from citylearn.citylearn import CityLearnEnv
schema_path = "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
with open(schema_path) as f:
    schema = json.load(f)
schema["root_directory"] = os.path.dirname(os.path.abspath(schema_path))
env = CityLearnEnv(schema=schema)
env.reset()

print("Action names per building:")
for b_idx, names in enumerate(env.action_names):
    print("  Building %d: %s" % (b_idx, names))

# Test: set action=1.0 for building 0 battery, run 3 steps, check SoC
actions = [[0.0] * len(sub) for sub in env.action_names]
for i, name in enumerate(env.action_names[0]):
    if name == "electrical_storage":
        actions[0][i] = 1.0
        print("\nSetting building 0 action[%d] = 1.0 (electrical_storage)" % i)

for step in range(3):
    t = env.time_step
    b = env.buildings[0]
    soc_before = float(np.asarray(b.electrical_storage.soc)[-1]) if len(b.electrical_storage.soc) > 0 else -1
    env.step(actions)
    soc_after = float(np.asarray(b.electrical_storage.soc)[-1])
    batt_ec = float(np.asarray(b.electrical_storage.electricity_consumption)[t])
    print("  step t=%d: soc_before=%.4f soc_after=%.4f batt_ec=%.3f kW" % (t, soc_before, soc_after, batt_ec))
