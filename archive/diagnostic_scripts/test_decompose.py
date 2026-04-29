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

# Step with action=0.3 for building 0 battery
actions = [[0.0] * len(sub) for sub in env.action_names]
actions[0][0] = 0.3
t = env.time_step
env.step(actions)

b = env.buildings[0]
es = b.electrical_storage
print("t=%d components of net_electricity_consumption:" % t)
print("  cooling_device_ec:    %.4f" % b.cooling_device.electricity_consumption[t])
print("  heating_device_ec:    %.4f" % b.heating_device.electricity_consumption[t])
print("  dhw_device_ec:        %.4f" % b.dhw_device.electricity_consumption[t])
print("  nonshift_device_ec:   %.4f" % b.non_shiftable_load_device.electricity_consumption[t])
print("  elec_storage_ec:      %.4f" % es.electricity_consumption[t])
print("  solar_generation:     %.4f" % b.solar_generation[t])
charger_ec = getattr(b, '_Building__chargers_electricity_consumption', None)
if charger_ec is not None:
    print("  chargers_ec:          %.4f" % float(charger_ec[t]))
wm_ec = getattr(b, '_Building__washing_machines_electricity_consumption', None)
if wm_ec is not None:
    print("  washing_machines_ec:  %.4f" % float(wm_ec[t]))
print("  ---")
print("  net_elec_consumption: %.4f" % b.net_electricity_consumption[t])
print("")
print("  energy_balance:       %.4f" % es.energy_balance[t])
print("  raw nsl[t]:           %.4f" % float(np.asarray(b._Building__energy_to_non_shiftable_load)[t]))
print("  raw sg[t]:            %.4f" % float(np.asarray(b._Building__solar_generation)[t]))
print("")
print("  So: net = sum of all components above")
total = (b.cooling_device.electricity_consumption[t]
         + b.heating_device.electricity_consumption[t]
         + b.dhw_device.electricity_consumption[t]
         + b.non_shiftable_load_device.electricity_consumption[t]
         + es.electricity_consumption[t]
         + b.solar_generation[t])
if charger_ec is not None:
    total += float(charger_ec[t])
if wm_ec is not None:
    total += float(wm_ec[t])
print("  manual sum:           %.4f" % total)
