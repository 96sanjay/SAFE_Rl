"""
Evaluate TRPOLag single-lambda checkpoint on the 17-building schema.

Architecture: 153 -> 512 -> 512 -> 256 -> 26 (deterministic mean)
Obs normalizer from checkpoint.
8759 steps, seed=42.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# ── Environment setup ──────────────────────────────────────────────────────
os.environ["CITYLEARN_SCHEMA"] = (
    "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/"
    "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
)
os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"

CKPT_PATH = (
    "/media/christmas/KODAK/Safe-CityLearn-Fork/runs/trpolag_v2g_stems/"
    "TRPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-02-23-16-07-24/"
    "torch_save/epoch-100.pt"
)
SEED = 42
TOTAL_STEPS = 8759
SOC_DEPARTURE_TOL = 0.001


# ── Actor network (matches checkpoint) ────────────────────────────────────
class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(obs)


# ── Obs normalizer ────────────────────────────────────────────────────────
class ObsNormalizer:
    def __init__(self, state_dict: dict):
        self._mean = state_dict["_mean"].numpy()
        self._std = state_dict["_std"].numpy()
        self._clip = state_dict["_clip"].numpy()

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        normed = (obs - self._mean) / (self._std + 1e-8)
        return np.clip(normed, -self._clip, self._clip)


# ── Load checkpoint ───────────────────────────────────────────────────────
print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")

actor = DeterministicActor(obs_dim=153, act_dim=26)
# Map checkpoint keys (mean.0.weight -> mean network)
actor.load_state_dict({k: v for k, v in ckpt["pi"].items() if k.startswith("mean.")})
actor.eval()

normalizer = ObsNormalizer(ckpt["obs_normalizer"])
print("Actor and normalizer loaded successfully.")


# ── Build environment (same as omni_env.py, no safety wrapper for raw eval) ─
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper
from citylearn_safe.adapters import SingleAgentListAdapter

base_raw = CityLearnEnv(schema=os.environ["CITYLEARN_SCHEMA"], central_agent=True)

# ── Discover action mapping ───────────────────────────────────────────────
action_names = base_raw.action_names[0]  # flatten list-of-lists
print(f"\nAction mapping ({len(action_names)} actions):")
battery_indices = []
ev_indices = []
other_indices = []
ev_charger_names = {}  # idx -> charger suffix

for i, name in enumerate(action_names):
    tag = ""
    if name == "electrical_storage":
        battery_indices.append(i)
        tag = " [BATTERY]"
    elif "electric_vehicle_storage_charger_" in name.lower():
        ev_indices.append(i)
        suffix = name.split("electric_vehicle_storage_charger_")[1]
        ev_charger_names[i] = suffix
        tag = " [EV]"
    else:
        other_indices.append(i)
        tag = " [OTHER]"
    print(f"  action[{i:2d}] = {name}{tag}")

n_batteries = len(battery_indices)
n_evs = len(ev_indices)
print(f"\nBatteries: {n_batteries}, EVs: {n_evs}, Other: {len(other_indices)}")

# ── Discover EV chargers per building ─────────────────────────────────────
print("\nEV charger discovery:")
charger_info = {}  # charger_suffix -> {building_idx, building_name, charger_obj}
for b_idx, b in enumerate(base_raw.buildings):
    chargers = b.electric_vehicle_chargers if b.electric_vehicle_chargers else []
    for ch_idx, ch in enumerate(chargers):
        # The action name suffix encodes building+charger: e.g., "1_1", "15_2"
        cid = getattr(ch, "charger_id", None)
        print(f"  Building {b_idx} ({b.name}): charger {ch_idx}, charger_id={cid}")
        charger_info[cid] = {"building_idx": b_idx, "building_name": b.name, "charger_obj": ch}

# ── Wrap for running ──────────────────────────────────────────────────────
base = NormalizedObservationWrapper(base_raw)
base = SingleAgentListAdapter(base)

obs, info = base.reset()
obs_flat = np.asarray(obs, dtype=np.float32).ravel()
assert obs_flat.shape[0] == 153, f"Obs dim mismatch: expected 153, got {obs_flat.shape[0]}"

# Action space bounds
act_low = np.asarray(base.action_space.low, dtype=np.float32).ravel()
act_high = np.asarray(base.action_space.high, dtype=np.float32).ravel()


# ── Helper: get CityLearn env for state access ────────────────────────────
def get_citylearn_env():
    return base_raw


def get_time_index():
    t = int(getattr(get_citylearn_env(), "time_step", 0))
    return 0 if t <= 0 else t - 1


# ── Tracking structures ──────────────────────────────────────────────────
# EV departure tracking: per charger, track connected EV name at each step
prev_ev_connected = {}  # charger_id -> ev_name or None
departure_records = []  # list of (charger_id, required_soc, actual_soc)

# Battery SoC tracking
battery_soc_violations = 0  # steps where any battery SoC > 0.95
battery_soc_total = 0

# Battery action tracking
battery_charge_count = 0
battery_discharge_count = 0
battery_idle_count = 0
battery_total_actions = 0

# EV action tracking
ev_charge_count = 0
ev_v2g_count = 0
ev_idle_count = 0
ev_total_actions = 0

# Grid constraint tracking
building_nec_violations = 0
building_nec_total = 0
grid_nec_violations = 0

# Departure SoC accumulator
departure_soc_sum = 0.0
departure_count = 0

# Reward tracking
total_reward = 0.0

# Auto-calibrate P_building_max and P_grid_max from data
building_loads = []
total_load = None
for b in base_raw.buildings:
    es = getattr(b, "energy_simulation", None)
    if es is None:
        continue
    load = getattr(es, "non_shiftable_load", None)
    if load is None:
        continue
    arr = np.array(load, dtype=np.float64)
    building_loads.append(float(np.percentile(np.abs(arr), 95)))
    total_load = arr.copy() if total_load is None else total_load + arr

P_building_max = float(np.mean(building_loads)) if building_loads else 4.6083
P_grid_max = float(np.percentile(np.abs(total_load), 95)) if total_load is not None else 10.2352 * 17 / 5
n_buildings = len(base_raw.buildings)
print(f"\nAuto-calibrated: P_building_max={P_building_max:.4f}, P_grid_max={P_grid_max:.4f}")
print(f"Buildings: {n_buildings}")


# ── Initialize EV connected state ────────────────────────────────────────
def get_ev_state_snapshot():
    """Return dict: charger_id -> (ev_name, required_soc, actual_soc) or None."""
    city = get_citylearn_env()
    t = get_time_index()
    snapshot = {}
    for b in city.buildings:
        chargers = b.electric_vehicle_chargers if b.electric_vehicle_chargers else []
        for ch in chargers:
            cid = getattr(ch, "charger_id", None)
            ev = getattr(ch, "connected_electric_vehicle", None)
            if ev is None:
                snapshot[cid] = None
            else:
                ev_name = getattr(ev, "name", "unknown")
                batt = getattr(ev, "battery", None)
                if batt is not None and hasattr(batt, "soc"):
                    soc_arr = batt.soc
                    actual_soc = float(soc_arr[t]) if t < len(soc_arr) else 0.0
                else:
                    actual_soc = 0.0
                req_soc = getattr(ev, "required_soc_departure", None)
                if req_soc is None:
                    req_soc = getattr(ev, "_EVBattery__required_soc_departure", None)
                if req_soc is not None:
                    if hasattr(req_soc, "__len__"):
                        req_soc = float(req_soc[t]) if t < len(req_soc) else 1.0
                    else:
                        req_soc = float(req_soc)
                else:
                    req_soc = 1.0
                snapshot[cid] = (ev_name, req_soc, actual_soc)
    return snapshot


# Initial snapshot
prev_snapshot = get_ev_state_snapshot()
print(f"Initial EV snapshot: {len(prev_snapshot)} chargers tracked")
for cid, val in prev_snapshot.items():
    if val is not None:
        print(f"  {cid}: EV={val[0]}, req_soc={val[1]:.3f}, actual_soc={val[2]:.3f}")
    else:
        print(f"  {cid}: no EV connected")

# ── Main evaluation loop ─────────────────────────────────────────────────
print(f"\nRunning {TOTAL_STEPS} steps deterministically (seed={SEED})...")
np.random.seed(SEED)
torch.manual_seed(SEED)

obs_flat = np.asarray(obs, dtype=np.float32).ravel()

for step in range(TOTAL_STEPS):
    # Normalize obs and get action
    obs_normed = normalizer.normalize(obs_flat)
    obs_t = torch.as_tensor(obs_normed, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        action_t = actor(obs_t).squeeze(0).numpy()

    # Clip to action space bounds
    action_clipped = np.clip(action_t, act_low, act_high)

    # Track battery actions
    for idx in battery_indices:
        a = float(action_clipped[idx])
        battery_total_actions += 1
        if a > 0.01:
            battery_charge_count += 1
        elif a < -0.01:
            battery_discharge_count += 1
        else:
            battery_idle_count += 1

    # Track EV actions
    for idx in ev_indices:
        a = float(action_clipped[idx])
        ev_total_actions += 1
        if a > 0.01:
            ev_charge_count += 1
        elif a < -0.01:
            ev_v2g_count += 1
        else:
            ev_idle_count += 1

    # Step environment
    obs, reward, terminated, truncated, info = base.step(action_clipped)
    obs_flat = np.asarray(obs, dtype=np.float32).ravel()
    total_reward += float(reward)

    # Get post-step state
    city = get_citylearn_env()
    t_idx = get_time_index()

    # ── C2: Battery SoC > 0.95 ──
    for b in city.buildings:
        es = b.electrical_storage
        if es is not None and hasattr(es, "soc"):
            soc_arr = es.soc
            if t_idx < len(soc_arr):
                soc_val = float(soc_arr[t_idx])
                battery_soc_total += 1
                if soc_val > 0.95:
                    battery_soc_violations += 1

    # ── C3: Building NEC violations ──
    for b in city.buildings:
        nec = getattr(b, "net_electricity_consumption", None)
        if nec is not None and hasattr(nec, "__len__") and t_idx < len(nec):
            nec_val = abs(float(nec[t_idx]))
            building_nec_total += 1
            if nec_val > P_building_max:
                building_nec_violations += 1

    # ── C4: Grid NEC violations ──
    grid_nec = 0.0
    for b in city.buildings:
        nec = getattr(b, "net_electricity_consumption", None)
        if nec is not None and hasattr(nec, "__len__") and t_idx < len(nec):
            grid_nec += float(nec[t_idx])
    if abs(grid_nec) > P_grid_max:
        grid_nec_violations += 1

    # ── C0: EV departure tracking ──
    curr_snapshot = get_ev_state_snapshot()
    for cid, prev_val in prev_snapshot.items():
        curr_val = curr_snapshot.get(cid)
        # Departure: was connected (prev_val not None), now disconnected or different EV
        if prev_val is not None:
            departed = False
            if curr_val is None:
                departed = True
            elif curr_val[0] != prev_val[0]:
                # Different EV connected -> previous one departed
                departed = True

            if departed:
                ev_name, req_soc, actual_soc = prev_val
                departure_records.append((cid, req_soc, actual_soc, ev_name))
                departure_soc_sum += actual_soc
                departure_count += 1

    prev_snapshot = curr_snapshot

    if terminated or truncated:
        print(f"Episode ended at step {step + 1}")
        break

    if (step + 1) % 1000 == 0:
        print(f"  Step {step + 1}/{TOTAL_STEPS}, reward_so_far={total_reward:.1f}")


# ── Compute metrics ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EVALUATION RESULTS: TRPOLag Single-Lambda on 17-Building Schema")
print("=" * 70)

# C0: Departure violations
n_departures = len(departure_records)
n_violations = sum(
    1 for _, req, actual, _ in departure_records
    if actual < req - SOC_DEPARTURE_TOL
)
c0_rate = n_violations / n_departures if n_departures > 0 else 0.0
print(f"\n--- C0: EV Departure SoC Violations ---")
print(f"  Total departures:  {n_departures}")
print(f"  Violations:        {n_violations}")
print(f"  C0 violation rate: {c0_rate:.4f} ({c0_rate * 100:.2f}%)")
if departure_count > 0:
    print(f"  Mean departure SoC: {departure_soc_sum / departure_count:.4f}")

# C2: Battery SoC > 0.95
c2_rate = battery_soc_violations / battery_soc_total if battery_soc_total > 0 else 0.0
print(f"\n--- C2: Battery SoC > 0.95 ---")
print(f"  Violations: {battery_soc_violations} / {battery_soc_total}")
print(f"  C2 rate:    {c2_rate:.4f} ({c2_rate * 100:.2f}%)")

# C3: Building NEC violations
c3_rate = building_nec_violations / building_nec_total if building_nec_total > 0 else 0.0
print(f"\n--- C3: Building |NEC| > P_building_max ({P_building_max:.4f} kW) ---")
print(f"  Violations: {building_nec_violations} / {building_nec_total}")
print(f"  C3 rate:    {c3_rate:.4f} ({c3_rate * 100:.2f}%)")

# C4: Grid NEC violations
steps_ran = min(step + 1, TOTAL_STEPS)
c4_rate = grid_nec_violations / steps_ran if steps_ran > 0 else 0.0
print(f"\n--- C4: Grid |NEC| > P_grid_max ({P_grid_max:.4f} kW) ---")
print(f"  Violations: {grid_nec_violations} / {steps_ran}")
print(f"  C4 rate:    {c4_rate:.4f} ({c4_rate * 100:.2f}%)")

# Battery action profile
print(f"\n--- Battery Action Profile ---")
if battery_total_actions > 0:
    print(f"  Charge:    {battery_charge_count:6d} ({battery_charge_count / battery_total_actions * 100:.1f}%)")
    print(f"  Discharge: {battery_discharge_count:6d} ({battery_discharge_count / battery_total_actions * 100:.1f}%)")
    print(f"  Idle:      {battery_idle_count:6d} ({battery_idle_count / battery_total_actions * 100:.1f}%)")

# EV action profile
print(f"\n--- EV Action Profile ---")
if ev_total_actions > 0:
    print(f"  Charge: {ev_charge_count:6d} ({ev_charge_count / ev_total_actions * 100:.1f}%)")
    print(f"  V2G:    {ev_v2g_count:6d} ({ev_v2g_count / ev_total_actions * 100:.1f}%)")
    print(f"  Idle:   {ev_idle_count:6d} ({ev_idle_count / ev_total_actions * 100:.1f}%)")

# Total reward
print(f"\n--- Reward ---")
print(f"  Total episode reward: {total_reward:.1f}")
print(f"  Mean step reward:     {total_reward / steps_ran:.4f}")

# Departure details (first 20)
if departure_records:
    print(f"\n--- Departure Details (first 20 of {len(departure_records)}) ---")
    for i, (cid, req, actual, ev_name) in enumerate(departure_records[:20]):
        viol = "VIOLATION" if actual < req - SOC_DEPARTURE_TOL else "OK"
        print(f"  {cid}: {ev_name} req={req:.3f} actual={actual:.3f} [{viol}]")

print(f"\n{'=' * 70}")
print("Done.")
