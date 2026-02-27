import json, os, sys, numpy as np
os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, os.getcwd())
os.environ["CITYLEARN_STEMS_SOC_LOW"] = "0.0"
os.environ["CITYLEARN_STEMS_SOC_HIGH"] = "0.95"
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "3.47"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "35.76"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"
from citylearn.citylearn import CityLearnEnv
from evaluation.agents.rbc import IntelligentRBC
schema_path = "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
with open(schema_path) as f:
    schema = json.load(f)
schema["root_directory"] = os.path.dirname(os.path.abspath(schema_path))
env = CityLearnEnv(schema=schema)
obs = env.reset()

# Setup RBC
flat_names = [n for sub in env.action_names for n in sub]
class _FlatProxy:
    def __init__(self, raw, names):
        self._raw = raw
        self.action_names = names
        self.action_space = type("S", (), {"shape": (len(names),)})()
    def __getattr__(self, name):
        return getattr(self._raw, name)
rbc = IntelligentRBC(_FlatProxy(env, flat_names), ev_mode="greedy")
dims = [len(sub) for sub in env.action_names]

# Track missing power components
charger_powers = []
wm_powers = []
nonshift_device_vs_raw = []
pred_errors = []

for step in range(min(8759, 8759)):
    t = env.time_step
    
    flat_a = rbc.predict(obs)
    actions = []
    offset = 0
    for d in dims:
        actions.append(flat_a[offset:offset+d].copy())
        offset += d
    obs, _, _, _, _ = env.step(actions)
    # Collect for all buildings
    for b_idx, b in enumerate(env.buildings):
        charger_ec = getattr(b, '_Building__chargers_electricity_consumption', None)
        wm_ec = getattr(b, '_Building__washing_machines_electricity_consumption', None)
        c_val = float(charger_ec[t]) if charger_ec is not None else 0.0
        w_val = float(wm_ec[t]) if wm_ec is not None else 0.0
        charger_powers.append(c_val)
        wm_powers.append(w_val)
        # Check nonshift device vs raw
        raw_nsl = float(np.asarray(b._Building__energy_to_non_shiftable_load)[t])
        dev_nsl = float(b.non_shiftable_load_device.electricity_consumption[t])
        nonshift_device_vs_raw.append(dev_nsl - raw_nsl)

ca = np.array(charger_powers)
wa = np.array(wm_powers)
na = np.array(nonshift_device_vs_raw)
print("EV Charger electricity_consumption (all buildings, all steps):")
print("  mean=%.3f max=%.3f nonzero=%d/%d" % (np.mean(ca), np.max(ca), np.count_nonzero(ca > 0.01), len(ca)))
print("Washing machine electricity_consumption:")
print("  mean=%.3f max=%.3f nonzero=%d/%d" % (np.mean(wa), np.max(wa), np.count_nonzero(wa > 0.01), len(wa)))
print("Nonshift device_ec - raw_nsl:")
print("  mean=%.3f max=%.3f nonzero=%d/%d" % (np.mean(na), np.max(np.abs(na)), np.count_nonzero(np.abs(na) > 0.01), len(na)))
