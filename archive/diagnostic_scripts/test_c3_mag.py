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

threshold = 3.47
c3_fixable = 0
c3_unfixable = 0
c3_sample = []
excess_list = []

for step in range(8759):
    t = env.time_step
    for b_idx, b in enumerate(env.buildings):
        nsl = float(np.asarray(b._Building__energy_to_non_shiftable_load)[t])
        sg = float(np.asarray(b._Building__solar_generation)[t])
        base_net = nsl + sg
        nom_p = float(b.electrical_storage.nominal_power)
        if abs(base_net) > threshold:
            excess = abs(base_net) - threshold
            excess_list.append(excess)
            fixable = excess <= nom_p
            if fixable:
                c3_fixable += 1
            else:
                c3_unfixable += 1
            if step < 48 and b_idx == 0:
                c3_sample.append((t, b_idx, base_net, excess, fixable))
    env.step([[0.0] * len(sub) for sub in env.action_names])

total = c3_fixable + c3_unfixable
pct_fix = 100.0 * c3_fixable / max(1, total)
pct_unfix = 100.0 * c3_unfixable / max(1, total)
print("C3 violations (zero actions, full year, all 17 buildings):")
print("  Total:", total)
print("  Fixable (excess <= 5kW): %d (%.1f%%)" % (c3_fixable, pct_fix))
print("  Unfixable (excess > 5kW): %d (%.1f%%)" % (c3_unfixable, pct_unfix))

if excess_list:
    ea = np.array(excess_list)
    print("\nExcess magnitude stats:")
    print("  Mean: %.2f kW" % np.mean(ea))
    print("  Median: %.2f kW" % np.median(ea))
    print("  p75: %.2f kW" % np.percentile(ea, 75))
    print("  p95: %.2f kW" % np.percentile(ea, 95))
    print("  Max: %.2f kW" % np.max(ea))

print("\nFirst 48h building 0 violations:")
for t, bi, net, exc, fix in c3_sample:
    print("  t=%d net=%.2f excess=%.2f fixable=%s" % (t, net, exc, fix))
