#!/usr/bin/env python3
import os, sys
import numpy as np

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

SCHEMA = "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
P_MAX = 2.273834

from citylearn.citylearn import CityLearnEnv

def run_scenario(schema, action_value, name):
    env = CityLearnEnv(schema=schema, central_agent=True)
    env.reset()
    action_dim = env.action_space[0].shape[0] if isinstance(env.action_space, list) else env.action_space.shape[0]
    
    num_b = len(env.buildings)
    violations = {i: 0 for i in range(num_b)}
    violations_no_storage = {i: 0 for i in range(num_b)}
    net_max = {i: -999.0 for i in range(num_b)}
    net_all = {i: [] for i in range(num_b)}
    nsl_exceeds = {i: 0 for i in range(num_b)}
    nsl_max = {i: -999.0 for i in range(num_b)}
    steps = 0
    terminated = False
    
    while not terminated:
        action = [[action_value] * action_dim]
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        
        for i, b in enumerate(env.buildings):
            # Use [-2] because [-1] is the current incomplete step (always 0)
            if len(b.net_electricity_consumption) >= 2:
                net = b.net_electricity_consumption[-2]
            else:
                continue
            
            net_all[i].append(net)
            if net > net_max[i]:
                net_max[i] = net
            if net > P_MAX:
                violations[i] += 1
            
            # Also check without storage (pure structural)
            if len(b.net_electricity_consumption_without_storage) >= 2:
                net_ns = b.net_electricity_consumption_without_storage[-2]
                if net_ns > P_MAX:
                    violations_no_storage[i] += 1
            
            nsl = b.energy_simulation.non_shiftable_load[b.time_step - 1]
            if nsl > nsl_max[i]:
                nsl_max[i] = nsl
            if nsl > P_MAX:
                nsl_exceeds[i] += 1
        
        if steps % 2000 == 0:
            print(f"  [{name}] Step {steps}...")
    
    print(f"  [{name}] Done: {steps} steps")
    return num_b, steps, violations, violations_no_storage, net_max, net_all, nsl_exceeds, nsl_max

print(f"Threshold: {P_MAX} kW\n")
scenarios = {"ZERO": 0.0, "MAX_DISCHARGE": -1.0, "MAX_CHARGE": 1.0}
results = {}
for name, val in scenarios.items():
    print(f"{'='*50}\nScenario: {name} (action={val})\n{'='*50}")
    results[name] = run_scenario(SCHEMA, val, name)

num_b, steps = results["ZERO"][0], results["ZERO"][1]
total_checks = num_b * steps

print(f"\n{'='*80}")
print(f"C3 STRUCTURAL FLOOR — threshold: {P_MAX} kW, {num_b} buildings, {steps} steps")
print(f"{'='*80}")

for name in scenarios:
    nb, st, viol, viol_ns, nmax, nall, nsl_exc, nsl_m = results[name]
    tv = sum(viol.values())
    tv_ns = sum(viol_ns.values())
    rate = tv / total_checks * 100
    rate_ns = tv_ns / total_checks * 100
    print(f"\n── {name} (action={scenarios[name]}) ──")
    print(f"  C3 violations (with storage):    {tv:>7}/{total_checks} = {rate:.2f}%")
    print(f"  C3 violations (without storage): {tv_ns:>7}/{total_checks} = {rate_ns:.2f}%")
    print(f"  {'Bldg':>6} {'Viol':>7} {'Rate':>8} {'NoStor':>7} {'Rate':>8} {'MaxNet':>9} {'MeanNet':>9} {'MaxNSL':>9} {'NSL>thr':>8}")
    for i in range(nb):
        vr = viol[i]/st*100
        vr_ns = viol_ns[i]/st*100
        mn = np.mean(nall[i]) if nall[i] else 0.0
        print(f"  {'B'+str(i):>6} {viol[i]:>7} {vr:>7.2f}% {viol_ns[i]:>7} {vr_ns:>7.2f}% {nmax[i]:>9.3f} {mn:>9.3f} {nsl_m[i]:>9.3f} {nsl_exc[i]:>8}")

nsl_total = sum(results["ZERO"][6].values())
nsl_rate = nsl_total / total_checks * 100

# Without-storage floor (most accurate structural measure)
ns_total = sum(results["ZERO"][3].values())
ns_rate = ns_total / total_checks * 100

print(f"\n{'='*80}")
print(f"CONCLUSION")
print(f"{'='*80}")
print(f"{'Scenario':<20} {'C3 Rate':>10} {'No-Storage Rate':>16}")
print(f"-"*48)
for name in scenarios:
    tv = sum(results[name][2].values())
    tv_ns = sum(results[name][3].values())
    print(f"{name:<20} {tv/total_checks*100:>9.2f}% {tv_ns/total_checks*100:>15.2f}%")
print(f"\nNSL-only floor:              {nsl_rate:.2f}%")
print(f"No-storage floor (ZERO):     {ns_rate:.2f}%  <-- THIS is the true structural floor")
print(f"\nThe no-storage rate shows C3 violations when batteries do NOTHING.")
print(f"No RL agent can reduce C3 below this floor.")
