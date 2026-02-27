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
b = env.buildings[0]
es = b.electrical_storage
print("Battery attributes:")
print("  capacity:", es.capacity)
print("  nominal_power:", es.nominal_power)
print("  efficiency:", getattr(es, "efficiency", "N/A"))
print("  round_trip_efficiency:", getattr(es, "round_trip_efficiency", "N/A"))
print("  capacity_loss_coef:", getattr(es, "capacity_loss_coefficient", "N/A"))
print("  power_efficiency_curve:", getattr(es, "power_efficiency_curve", "N/A"))
print("  loss_coef:", getattr(es, "loss_coefficient", "N/A"))
print("  initial_soc:", getattr(es, "initial_soc", "N/A"))
# Check all relevant private attrs
for attr in sorted(dir(es)):
    if "effic" in attr.lower() or "loss" in attr.lower() or "soc" in attr.lower() or "init" in attr.lower():
        val = getattr(es, attr, "?")
        if not callable(val):
            print("  %s = %s" % (attr, val))
