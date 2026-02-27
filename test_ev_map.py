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

print("Action names and indices per building:")
g = 0
for b_idx, sub in enumerate(env.action_names):
    chargers = getattr(env.buildings[b_idx], "electric_vehicle_chargers", None) or []
    max_powers = []
    for ch in chargers:
        mp = getattr(ch, "max_charging_power", getattr(ch, "_Charger__max_charging_power", 0))
        if isinstance(mp, np.ndarray):
            mp = float(mp.ravel()[0])
        max_powers.append(float(mp))
    print("  B%d: gidx %d-%d names=%s charger_max_kW=%s" % (
        b_idx, g, g+len(sub)-1, sub, max_powers))
    g += len(sub)

# Test: action=0.5 on building 0 EV charger
env2 = CityLearnEnv(schema=schema)
env2.reset()
for _ in range(200):
    env2.step([[0.0]*len(sub) for sub in env2.action_names])
for test_a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    t = env2.time_step
    actions = [[0.0]*len(sub) for sub in env2.action_names]
    actions[0][1] = test_a
    env2.step(actions)
    cec = float(getattr(env2.buildings[0], '_Building__chargers_electricity_consumption')[t])
    print("EV action=%.2f -> charger_ec=%.3f kW" % (test_a, cec))
