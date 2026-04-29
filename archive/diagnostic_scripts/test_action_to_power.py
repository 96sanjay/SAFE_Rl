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

# Run multiple episodes with different constant actions, measure batt_ec
for test_a in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, -0.3, -0.5, -1.0]:
    env = CityLearnEnv(schema=schema)
    env.reset()
    # Charge up first if testing discharge
    if test_a < 0:
        for _ in range(5):
            a = [[0.0] * len(sub) for sub in env.action_names]
            a[0][0] = 1.0
            env.step(a)
    # Now apply test action for 1 step
    t = env.time_step
    b = env.buildings[0]
    es = b.electrical_storage
    soc_before_idx = max(0, t - 1)
    soc_before = float(np.asarray(es.soc)[soc_before_idx])
    a = [[0.0] * len(sub) for sub in env.action_names]
    a[0][0] = test_a
    env.step(a)
    batt_ec = float(np.asarray(es.electricity_consumption)[t])
    soc_after = float(np.asarray(es.soc)[t])
    dsoc = soc_after - soc_before
    energy_stored = dsoc * es.capacity
    print("action=% .1f soc=%.3f->%.3f dsoc=%.4f energy_stored=%.3f kWh batt_ec=%.3f kW ratio_ec/nom=%.3f" % (
        test_a, soc_before, soc_after, dsoc, energy_stored, batt_ec, batt_ec / es.nominal_power))
