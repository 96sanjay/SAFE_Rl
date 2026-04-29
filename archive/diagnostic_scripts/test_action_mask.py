#!/usr/bin/env python3
"""Test script for ActionMaskWrapper: verify C3 constraint enforcement.

Runs random actions through the masked env and categorizes any C3 violations
into: (1) step-0 HVAC initialization, (2) |exo|>P_bmax (inherent), 
(3) washing machine passthrough, (4) unexplained mask failure.
"""

import os
import sys
import numpy as np

os.environ['CITYLEARN_SCHEMA'] = '/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json'
os.environ['CITYLEARN_CENTRAL_AGENT'] = '1'
os.environ['CITYLEARN_REWARD_TYPE'] = 'stems'
os.environ['CITYLEARN_ACTION_MASK'] = '1'
os.environ['CITYLEARN_STEMS_P_BUILDING_MAX'] = '4.6083'
os.environ['CITYLEARN_STEMS_SOC_LOW'] = '0.0'
os.environ['CITYLEARN_STEMS_SOC_HIGH'] = '0.95'

project_root = '/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork'
sys.path.insert(0, project_root)

from citylearn.citylearn import CityLearnEnv
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.action_mask_wrapper import ActionMaskWrapper

P_BUILDING_MAX = 4.6083
N_STEPS = 200
N_BUILDINGS = 5
np.random.seed(42)


def main():
    print("=" * 70)
    print("ActionMaskWrapper Test: C3 Constraint Enforcement")
    print("=" * 70)

    print("\n--- Creating environment chain ---")
    base = CityLearnEnv(schema=os.environ['CITYLEARN_SCHEMA'], central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    masked = ActionMaskWrapper(safety)
    obs, info = masked.reset()

    if isinstance(obs, list):
        obs_arr = np.array(obs[0])
    else:
        obs_arr = np.array(obs)
    print(f"Observation shape: {obs_arr.shape}")
    print(f"Action space: {masked.action_space}")

    batt_act_map = masked._building_batt_act
    ev_act_map = masked._building_ev_act
    batt_powers = masked._batt_powers
    ev_powers = masked._ev_powers
    batt_act_indices = masked._batt_act_indices
    ev_act_indices = masked._ev_act_indices
    city = masked._city

    # Counters
    c3_with_mask = 0
    c3_without_mask = 0
    total_bs = 0
    cat_step0 = 0
    cat_exo = 0
    cat_wash = 0
    cat_unexplained = 0

    batt_ranges_log = []
    ev_ranges_log = []
    collapsed_count = 0
    exo_nec_log = []
    nec_actual_log = []

    print(f"\n--- Running {N_STEPS} steps with random actions ---\n")

    for step_i in range(N_STEPS):
        raw_action = np.random.uniform(-1, 1, 9).astype(np.float32)
        exo_nec = masked._get_exogenous_nec()

        obs, reward, terminated, truncated, info = masked.step(raw_action)

        safe_min = np.array(info['action_mask_safe_min'])
        safe_max = np.array(info['action_mask_safe_max'])

        t_idx = int(getattr(city, 'time_step', 1)) - 1
        actual_necs = []
        for b in city.buildings:
            nec = getattr(b, 'net_electricity_consumption', None)
            actual_necs.append(float(nec[t_idx]) if nec is not None and len(nec) > t_idx else 0)

        # Check C3 with mask + categorize violations
        for b_idx in range(N_BUILDINGS):
            total_bs += 1
            if abs(actual_necs[b_idx]) > P_BUILDING_MAX:
                c3_with_mask += 1
                b = city.buildings[b_idx]
                wash_ec = float(b.washing_machines_electricity_consumption[t_idx]) \
                    if len(b.washing_machines_electricity_consumption) > t_idx else 0
                if step_i == 0:
                    cat_step0 += 1
                elif abs(exo_nec[b_idx]) > P_BUILDING_MAX:
                    cat_exo += 1
                elif abs(wash_ec) > 0.01:
                    cat_wash += 1
                else:
                    cat_unexplained += 1
                    print(f"  UNEXPLAINED: step={step_i} B{b_idx}: "
                          f"|NEC|={abs(actual_necs[b_idx]):.4f}, "
                          f"exo={exo_nec[b_idx]:.4f}, wash={wash_ec:.4f}")

        # Counterfactual without mask
        for b_idx in range(N_BUILDINGS):
            exo = exo_nec[b_idx]
            device_power = 0.0
            if b_idx in batt_act_map:
                device_power += raw_action[batt_act_map[b_idx]] * batt_powers[b_idx]
            if b_idx in ev_act_map:
                device_power += raw_action[ev_act_map[b_idx]] * ev_powers[b_idx]
            if abs(exo + device_power) > P_BUILDING_MAX:
                c3_without_mask += 1

        # Track ranges
        batt_range = np.array([safe_max[ai] - safe_min[ai] for ai in batt_act_indices])
        ev_range = np.array([safe_max[ai] - safe_min[ai] for ai in ev_act_indices]) \
            if ev_act_indices else np.array([])
        batt_ranges_log.append(batt_range)
        ev_ranges_log.append(ev_range)
        for r in batt_range:
            if r <= 0:
                collapsed_count += 1
        for r in ev_range:
            if r <= 0:
                collapsed_count += 1

        exo_nec_log.append(exo_nec)
        nec_actual_log.append(actual_necs)

        if terminated or truncated:
            obs, info = masked.reset()

    # --- Summary ---
    batt_ranges_arr = np.array(batt_ranges_log)
    ev_ranges_arr = np.array(ev_ranges_log) if ev_act_indices else None

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nTotal building-steps: {total_bs}")
    print(f"\nC3 Violations WITH mask:    {c3_with_mask} "
          f"({100*c3_with_mask/total_bs:.2f}%)")
    print(f"C3 Violations WITHOUT mask: {c3_without_mask} "
          f"({100*c3_without_mask/total_bs:.2f}%)")

    print(f"\nViolation breakdown:")
    print(f"  Step 0 HVAC initialization:   {cat_step0}")
    print(f"  |exo| > P_bmax (inherent):    {cat_exo}")
    print(f"  Washing machine passthrough:  {cat_wash}")
    print(f"  Unexplained (mask failure):   {cat_unexplained}")

    print(f"\nBattery safe range width:")
    print(f"  Mean: {np.mean(batt_ranges_arr):.4f}")
    print(f"  Min:  {np.min(batt_ranges_arr):.4f}")
    print(f"  Max:  {np.max(batt_ranges_arr):.4f}")
    for i, b_idx in enumerate(sorted(batt_act_map.keys())):
        print(f"  Building {b_idx} (act {batt_act_map[b_idx]}): "
              f"mean={np.mean(batt_ranges_arr[:, i]):.4f}, "
              f"min={np.min(batt_ranges_arr[:, i]):.4f}, "
              f"max={np.max(batt_ranges_arr[:, i]):.4f}")

    if ev_ranges_arr is not None and len(ev_ranges_arr) > 0:
        print(f"\nEV safe range width:")
        print(f"  Mean: {np.mean(ev_ranges_arr):.4f}")
        print(f"  Min:  {np.min(ev_ranges_arr):.4f}")
        print(f"  Max:  {np.max(ev_ranges_arr):.4f}")
        for i, b_idx in enumerate(sorted(ev_act_map.keys())):
            print(f"  Building {b_idx} (act {ev_act_map[b_idx]}): "
                  f"mean={np.mean(ev_ranges_arr[:, i]):.4f}, "
                  f"min={np.min(ev_ranges_arr[:, i]):.4f}, "
                  f"max={np.max(ev_ranges_arr[:, i]):.4f}")

    print(f"\nCollapsed ranges (width <= 0): {collapsed_count}")

    # Final verdict
    print("\n" + "=" * 70)
    if cat_unexplained == 0:
        print("PASS: Zero unexplained C3 violations.")
        print("      All violations are from inherent causes (HVAC init, |exo|>P_bmax, washing).")
    else:
        print(f"FAIL: {cat_unexplained} unexplained C3 violations (mask failure).")
    print("=" * 70)


if __name__ == '__main__':
    main()
