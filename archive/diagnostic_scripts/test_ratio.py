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
print("seconds_per_time_step:", b.seconds_per_time_step)
print("time_step_ratio:", getattr(es, "time_step_ratio", "N/A"))
print("nominal_power:", es.nominal_power)
print("capacity:", es.capacity)
print("round_trip_efficiency:", es.round_trip_efficiency)
print("efficiency:", es.efficiency)
print("depth_of_discharge:", getattr(es, "depth_of_discharge", "N/A"))
print("degraded_capacity:", getattr(es, "degraded_capacity", "N/A"))
print("energy_init:", getattr(es, "energy_init", "N/A"))
print("get_max_input_power():", es.get_max_input_power())
print("get_max_output_power():", es.get_max_output_power())
print("available_nominal_power:", es.available_nominal_power)
