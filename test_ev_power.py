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

# Check which buildings have EV chargers and their max power
for b_idx, b in enumerate(env.buildings):
    chargers = getattr(b, "electric_vehicle_chargers", None) or []
    if chargers:
        for ch in chargers:
            max_p = getattr(ch, "max_charging_power", 
                    getattr(ch, "_Charger__max_charging_power", None))
            if isinstance(max_p, np.ndarray):
                max_p = float(max_p.ravel()[0])
            name = getattr(ch, "name", "?")
            print("Building %d charger %s: max_charging_power=%.2f kW" % (b_idx, name, max_p))

# Run a few steps with known EV actions to check mapping
# Find a step where EV is connected
for warmup in range(200):
    env.step([[0.0] * len(sub) for sub in env.action_names])

# Now try action=1.0 on building 0's EV charger (index 1 in action_names[0])
b = env.buildings[0]
chargers = b.electric_vehicle_chargers or []
print("\nBuilding 0 chargers:", len(chargers))
for ch in chargers:
    ev = getattr(ch, "connected_electric_vehicle", None)
    print("  connected EV:", ev is not None)
    if ev is not None:
        batt = getattr(ev, "battery", None)
        if batt:
            print("  EV battery capacity:", getattr(batt, "capacity", "?"))
            soc = getattr(batt, "soc", None)
            if soc is not None:
                t = env.time_step
                print("  EV soc[t-1]:", float(np.asarray(soc)[max(0,t-1)]))

# Test action=1.0 on EV charger
t = env.time_step
actions = [[0.0] * len(sub) for sub in env.action_names]
actions[0][1] = 1.0  # EV charger action for building 0
env.step(actions)
charger_ec = getattr(b, '_Building__chargers_electricity_consumption', None)
print("\nWith EV action=1.0:")
print("  charger_ec[t]=%.3f" % float(charger_ec[t]))
