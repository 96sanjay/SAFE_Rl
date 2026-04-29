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

actions = [[0.0] * len(sub) for sub in env.action_names]
actions[0][0] = 1.0  # full charge building 0 battery

for step in range(5):
    t = env.time_step
    b = env.buildings[0]
    soc_arr = np.asarray(b.electrical_storage.soc)
    print("  t=%d len(soc)=%d soc[-1]=%.4f soc[t]=%.4f soc[:t+2]=%s" % (
        t, len(soc_arr), soc_arr[-1], soc_arr[t] if t < len(soc_arr) else -1,
        soc_arr[:min(t+2, len(soc_arr))].tolist()))
    env.step(actions)
# One more read after last step
t = env.time_step
soc_arr = np.asarray(b.electrical_storage.soc)
print("  t=%d soc[:t+1]=%s" % (t, soc_arr[:min(t+1, len(soc_arr))].tolist()))
