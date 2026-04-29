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
obs = env.reset()

# Check: for building 0, what does action=0.5 (half charge) produce?
# predicted_net = nsl[t] + sg[t] + action * nominal_power
# But CityLearn might scale action differently (efficiency, etc.)
errs = []
for step in range(48):
    t = env.time_step
    b = env.buildings[0]
    nsl = float(np.asarray(b._Building__energy_to_non_shiftable_load)[t])
    sg = float(np.asarray(b._Building__solar_generation)[t])
    nom_p = float(b.electrical_storage.nominal_power)
    cap = float(b.electrical_storage.capacity)
    soc_arr = np.asarray(b.electrical_storage.soc)
    cur_soc = float(soc_arr[-1]) if len(soc_arr) > 0 else 0.5
    # Use a non-zero action for building 0 battery
    test_action = 0.3
    actions = [[0.0] * len(sub) for sub in env.action_names]
    # Find battery action index for building 0
    for i, name in enumerate(env.action_names[0]):
        if name == "electrical_storage":
            actions[0][i] = test_action
    pred_net = nsl + sg + test_action * nom_p
    env.step(actions)
    actual_net = float(np.asarray(b.net_electricity_consumption)[t])
    err = actual_net - pred_net
    batt_ec = float(np.asarray(b.electrical_storage.electricity_consumption)[t])
    if abs(err) > 0.05:
        print("  t=%d nsl=%.2f sg=%.2f action=%.1f pred=%.2f actual=%.2f err=%.2f batt_ec=%.2f soc=%.3f" % (
            t, nsl, sg, test_action, pred_net, actual_net, err, batt_ec, cur_soc))
    errs.append(err)
ea = np.array(errs)
print("\nPrediction errors (action=0.3, building 0, 48 steps):")
print("  Mean abs: %.4f" % np.mean(np.abs(ea)))
print("  Max abs: %.4f" % np.max(np.abs(ea)))
print("  Steps with |err|>0.1: %d/48" % sum(1 for e in ea if abs(e) > 0.1))
