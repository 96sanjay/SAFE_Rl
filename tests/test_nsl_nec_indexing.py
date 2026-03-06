#!/usr/bin/env python3
"""Verify NSL and NEC use the same time indexing in the C3 block.

The C3 block in safety_env_v3.py reads at idx_bp = max(0, time_step - 1).
This test verifies that NSL[idx_bp] and NEC[idx_bp] refer to the same timestep,
which is critical for the agent-controllable C3 (P1) to work correctly.

Uses 5-building schema by default; override with CITYLEARN_SCHEMA env var.
"""
import os, sys
import numpy as np

os.chdir("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, os.getcwd())

if "CITYLEARN_SCHEMA" not in os.environ:
    os.environ["CITYLEARN_SCHEMA"] = os.path.join(
        os.getcwd(), "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
    )
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "29.6915"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_zero"

from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv(schema=os.environ["CITYLEARN_SCHEMA"], central_agent=True)
num_buildings = len(env.buildings)
print(f"Schema: {os.environ['CITYLEARN_SCHEMA']}")
print(f"Num buildings: {num_buildings}")
env.reset()

# action_space is a list of Box for central_agent=True
if isinstance(env.action_space, list):
    action_dim = sum(s.shape[0] for s in env.action_space)
else:
    action_dim = env.action_space.shape[0]

mismatches = 0

for step in range(100):
    # Take a zero-action step first so NEC gets populated
    env.step([[0.0] * action_dim])

    t = env.time_step
    # C3 block uses idx_bp = max(0, time_step - 1)
    idx_bp = max(0, t - 1)

    # Skip idx_bp=0: at the very first timestep, batteries have initial
    # state that adds a large charge/discharge even with zero actions.
    # From step 1 onwards, zero actions mean zero storage contribution.
    if idx_bp == 0:
        continue

    for b_idx, b in enumerate(env.buildings):
        nsl = float(np.asarray(b._Building__energy_to_non_shiftable_load)[idx_bp])
        sg  = float(np.asarray(b._Building__solar_generation)[idx_bp])

        nec_arr = getattr(b, "net_electricity_consumption", None)
        if nec_arr is not None and hasattr(nec_arr, "__len__") and len(nec_arr) > idx_bp:
            nec = float(nec_arr[idx_bp])
        else:
            nec = 0.0

        # With zero action, NEC ≈ NSL + solar_generation
        # (storage contribution ≈ 0 when no charge/discharge)
        expected_nec = nsl + sg
        diff = abs(nec - expected_nec)

        if diff > 0.5:
            mismatches += 1
            if mismatches <= 5:
                print(f"MISMATCH step={step} b={b_idx} idx_bp={idx_bp}: "
                      f"nec={nec:.4f} vs nsl+sg={expected_nec:.4f} (diff={diff:.4f})")

if mismatches == 0:
    print(f"PASS: NSL + solar indexing matches NEC for all 100 steps x {num_buildings} buildings")
else:
    print(f"FAIL: {mismatches} mismatches found")

assert mismatches == 0, f"NSL indexing mismatch: {mismatches} cases"
