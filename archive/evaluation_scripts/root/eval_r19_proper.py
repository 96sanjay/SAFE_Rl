#!/usr/bin/env python3
"""
Proper evaluation of R19 ablation using its ORIGINAL env vars.

Builds the exact same env wrapper chain used during training:
  CityLearnEnv -> NormalizedObsWrapper -> SingleAgentListAdapter
  -> CityLearnSafetyEnvV3 -> ForecastObsWrapper -> SauteEVBudgetWrapper

Loads the epoch-80 checkpoint, runs full year (8759 steps) deterministically.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1) Parse env vars from R19 run script
# ---------------------------------------------------------------------------
RUN_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "run_r19_ablation.sh",
)

PROJECT = os.path.dirname(os.path.abspath(__file__))


def parse_exports(script_path: str) -> dict[str, str]:
    """Extract all 'export KEY=VALUE' from a bash script."""
    exports = {}
    with open(script_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("export "):
                continue
            # Remove 'export '
            rest = line[len("export "):]
            # Handle comments after value
            # Split on first '='
            eq_idx = rest.find("=")
            if eq_idx < 0:
                continue
            key = rest[:eq_idx].strip()
            val_raw = rest[eq_idx + 1:].strip()
            # Remove inline comment (but not inside quotes)
            if val_raw.startswith('"'):
                end_q = val_raw.find('"', 1)
                if end_q > 0:
                    val_raw = val_raw[1:end_q]
            else:
                # Remove trailing comment
                for sep in ["  #", " #", "\t#"]:
                    ci = val_raw.find(sep)
                    if ci >= 0:
                        val_raw = val_raw[:ci]
                val_raw = val_raw.strip().strip('"').strip("'")
            # Expand $PROJECT
            val_raw = val_raw.replace("$PROJECT", PROJECT)
            exports[key] = val_raw
    return exports


env_vars = parse_exports(RUN_SCRIPT)

# Apply ALL parsed env vars
for k, v in env_vars.items():
    os.environ[k] = v

# Override schema to full year (same schema, just explicit)
schema_path = os.path.join(
    PROJECT,
    "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
)
os.environ["CITYLEARN_SCHEMA"] = schema_path
os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"

# Add project to PYTHONPATH
sys.path.insert(0, PROJECT)

print("=" * 60)
print("  R19 Proper Evaluation")
print("=" * 60)
print(f"Schema: {schema_path}")
print(f"Parsed {len(env_vars)} env vars from run script")
# Print key ones
for k in sorted(env_vars.keys()):
    if k.startswith("STEMS_") or k.startswith("CITYLEARN_"):
        print(f"  {k}={env_vars[k]}")
print()

# ---------------------------------------------------------------------------
# 2) Build env chain (same as omni_env_v2.py but without the CMDP wrapper)
# ---------------------------------------------------------------------------
from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.saute_ev_wrapper import SauteEVBudgetWrapper

base = make_base_env(central_agent=True)
safety = CityLearnSafetyEnvV3(base)
forecast = ForecastObsWrapper(safety, forecast_horizon=24)
env = SauteEVBudgetWrapper(forecast)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
print(f"Env obs_dim={obs_dim}, act_dim={act_dim}")

# ---------------------------------------------------------------------------
# 3) Load checkpoint
# ---------------------------------------------------------------------------
CKPT_DIR = os.path.join(
    PROJECT,
    "runs/r19_ablation/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-12-20-01-28/torch_save",
)

# Find latest epoch
epoch_files = [f for f in os.listdir(CKPT_DIR) if f.startswith("epoch-") and f.endswith(".pt")]
epochs = [int(f.replace("epoch-", "").replace(".pt", "")) for f in epoch_files]
latest_epoch = max(epochs)
ckpt_path = os.path.join(CKPT_DIR, f"epoch-{latest_epoch}.pt")
print(f"Loading checkpoint: epoch-{latest_epoch} from {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location="cpu")

# ---------------------------------------------------------------------------
# 4) Build actor MLP [256, 256] with tanh activation
# ---------------------------------------------------------------------------
pi_state = ckpt["pi"]

# OmniSafe PPOLag actor keys: mean.0.weight, mean.0.bias, mean.2.weight, mean.2.bias, mean.4.weight, mean.4.bias
# This is a Sequential: Linear(199,256) -> Tanh -> Linear(256,256) -> Tanh -> Linear(256,9)
actor = torch.nn.Sequential(
    torch.nn.Linear(obs_dim, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, 256),
    torch.nn.Tanh(),
    torch.nn.Linear(256, act_dim),
)

# Map OmniSafe keys -> Sequential keys
key_map = {
    "mean.0.weight": "0.weight",
    "mean.0.bias": "0.bias",
    "mean.2.weight": "2.weight",
    "mean.2.bias": "2.bias",
    "mean.4.weight": "4.weight",
    "mean.4.bias": "4.bias",
}

actor_sd = {}
for omnisafe_key, seq_key in key_map.items():
    if omnisafe_key not in pi_state:
        raise KeyError(f"Missing key {omnisafe_key} in checkpoint pi state")
    actor_sd[seq_key] = pi_state[omnisafe_key]

actor.load_state_dict(actor_sd)
actor.eval()
print(f"Actor loaded: {sum(p.numel() for p in actor.parameters())} params")

# ---------------------------------------------------------------------------
# 5) Load obs normalizer
# ---------------------------------------------------------------------------
norm_data = ckpt.get("obs_normalizer")
has_normalizer = norm_data is not None
if has_normalizer:
    norm_mean = norm_data["_mean"].numpy()
    norm_std = norm_data["_std"].numpy()
    norm_clip = norm_data["_clip"].numpy()
    # Ensure std > 0
    norm_std = np.maximum(norm_std, 1e-8)
    print(f"Obs normalizer loaded: dim={len(norm_mean)}")
else:
    print("WARNING: No obs normalizer in checkpoint")


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """Apply OmniSafe-style obs normalization: clip((obs - mean) / std, -clip, clip)."""
    if not has_normalizer:
        return obs
    # The normalizer was trained on the 199-dim obs (base 70 + forecast 128 + saute 1)
    # But obs may come back with different dims if env changed. Truncate/pad.
    obs_flat = obs.ravel()
    n = min(len(obs_flat), len(norm_mean))
    normed = np.zeros_like(obs_flat)
    normed[:n] = np.clip(
        (obs_flat[:n] - norm_mean[:n]) / norm_std[:n],
        -norm_clip[:n],
        norm_clip[:n],
    )
    return normed


# ---------------------------------------------------------------------------
# 6) Run full year deterministically
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

obs, info = env.reset(seed=42)
print(f"Reset obs shape: {obs.shape}")

# Access the CityLearn env for direct state reading
def get_citylearn_env(wrapper):
    """Walk wrapper chain to find the CityLearnEnv."""
    cur = wrapper
    seen = set()
    for _ in range(40):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "buildings") and hasattr(cur, "time_step"):
            blds = getattr(cur, "buildings", None)
            if blds is not None and len(blds) > 0:
                return cur
        for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return None


city = get_citylearn_env(env)
if city is None:
    print("ERROR: Could not find CityLearnEnv in wrapper chain")
    sys.exit(1)

print(f"CityLearnEnv found: {len(city.buildings)} buildings")

# Discover chargers via electric_vehicle_chargers (CityLearn API)
charger_info = []  # (building_idx, charger_obj, charger_id)
for bi, b in enumerate(city.buildings):
    for ch in getattr(b, "electric_vehicle_chargers", []):
        cid = getattr(ch, "charger_id", f"charger_{bi}")
        charger_info.append((bi, ch, cid))
        print(f"  Charger: {cid} in {b.name}")

if not charger_info:
    print("WARNING: No EV chargers found")

# Tracking variables
total_steps = 8759
step_data = []

departures = []  # list of (charger_id, departure_soc, required_soc)

# Building/grid metrics
total_grid_import = 0.0
total_grid_export = 0.0
total_solar_gen = 0.0
peak_values = []
c3_violations = 0
c3_total = 0
c4_violations = 0
c4_total = 0

# Battery metrics
batt_charge_steps = 0
batt_discharge_steps = 0
batt_total_steps = 0

# EV metrics
ev_charge_steps = 0
ev_v2g_steps = 0
ev_v2g_peak_steps = 0
ev_total_steps = 0

# Get action names for identifying battery vs EV actions
act_names_raw = None
_e = env
for _ in range(20):
    act_names_raw = getattr(_e, "action_names", None)
    if act_names_raw is not None:
        break
    _e = getattr(_e, "env", getattr(_e, "base", None))
    if _e is None:
        break

if act_names_raw is not None:
    if isinstance(act_names_raw, list) and len(act_names_raw) > 0 and isinstance(act_names_raw[0], list):
        act_names = [n for sub in act_names_raw for n in sub]
    else:
        act_names = list(act_names_raw)
    print(f"Action names: {act_names}")
else:
    act_names = []
    print("WARNING: Could not get action names")

# Identify battery and EV action indices
batt_indices = [i for i, n in enumerate(act_names) if n == "electrical_storage"]
ev_indices = [i for i, n in enumerate(act_names) if "electric_vehicle" in n.lower()]
print(f"Battery action indices: {batt_indices}")
print(f"EV action indices: {ev_indices}")

# Power thresholds for C3/C4
P_building_max = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
P_grid_max = float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "10.2352"))
c3_controllable = os.environ.get("CITYLEARN_C3_CONTROLLABLE", "0") == "1"

print(f"\nP_building_max={P_building_max}, P_grid_max={P_grid_max}, C3_controllable={c3_controllable}")
print(f"\nStarting {total_steps}-step evaluation...")
print()

# Peak hour detection (17:00-21:00 roughly)
# CityLearn uses hour_cos/hour_sin encoded, but we can get hour from time_step
def get_hour(time_step: int) -> int:
    """CityLearn time_step is 0-indexed hourly for the year."""
    return time_step % 24


for step_i in range(total_steps):
    # Normalize obs and get action
    obs_normed = normalize_obs(obs)
    obs_t = torch.as_tensor(obs_normed, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        mean = actor(obs_t)
        action = torch.tanh(mean).squeeze(0).numpy()

    # Step the env
    obs, reward, terminated, truncated, info = env.step(action)

    # Read post-step state for metrics
    t_after = int(getattr(city, "time_step", 0))
    hour = get_hour(t_after)
    is_peak = 17 <= hour <= 21

    # Grid metrics from info
    grid_import = float(info.get("grid_import_kwh", 0.0))
    grid_export = float(info.get("grid_export_kwh", 0.0))
    total_grid_import += grid_import
    total_grid_export += grid_export

    # Solar
    solar = 0.0
    for b in city.buildings:
        sg = getattr(b, "solar_generation", None)
        if sg is not None and hasattr(sg, "__len__") and t_after < len(sg):
            solar += abs(float(sg[t_after]))
    total_solar_gen += solar

    # Peak tracking
    step_net = float(info.get("step_net_consumption_kwh", 0.0))
    peak_values.append(step_net)

    # C3: building power violations
    for b in city.buildings:
        c3_total += 1
        nec = getattr(b, "net_electricity_consumption", None)
        if nec is not None and hasattr(nec, "__len__") and t_after < len(nec):
            p_i = float(nec[t_after])
            if c3_controllable:
                # Only count agent-controllable violations
                nsl_arr = getattr(b, "_Building__energy_to_non_shiftable_load", [])
                sg_arr = getattr(b, "_Building__solar_generation", [])
                nsl_i = 0.0
                if hasattr(nsl_arr, "__len__") and t_after < len(nsl_arr):
                    nsl_i = float(nsl_arr[t_after])
                if hasattr(sg_arr, "__len__") and t_after < len(sg_arr):
                    nsl_i += float(sg_arr[t_after])
                if abs(nsl_i) <= P_building_max:
                    if abs(p_i) > P_building_max:
                        c3_violations += 1
                else:
                    if abs(p_i) > abs(nsl_i):
                        c3_violations += 1
            else:
                if abs(p_i) > P_building_max:
                    c3_violations += 1

    # C4: grid import violation
    c4_total += 1
    if grid_import > P_grid_max:
        c4_violations += 1

    # Battery action stats
    for bi in batt_indices:
        if bi < len(action):
            a = float(action[bi])
            batt_total_steps += 1
            if a > 0.05:
                batt_charge_steps += 1
            elif a < -0.05:
                batt_discharge_steps += 1

    # EV action stats
    for ei in ev_indices:
        if ei < len(action):
            a = float(action[ei])
            ev_total_steps += 1
            if a > 0.05:
                ev_charge_steps += 1
            elif a < -0.05:
                ev_v2g_steps += 1
                if is_peak:
                    ev_v2g_peak_steps += 1

    # Track departures using extractor logic:
    # Departure = charger_state[t] == 1.0 AND departure_time[t] == 0.0
    # t_after already set above from city.time_step
    t_soc = max(0, t_after - 1)  # SoC index is time_step - 1 after step
    for bi, ch, cid in charger_info:
        sim = getattr(ch, "charger_simulation", None)
        if sim is None:
            continue
        state_arr = getattr(sim, "_electric_vehicle_charger_state", None)
        dep_time_arr = getattr(sim, "_electric_vehicle_departure_time", None)
        req_soc_arr = getattr(sim, "_electric_vehicle_required_soc_departure", None)
        if state_arr is None or dep_time_arr is None or req_soc_arr is None:
            continue
        if t_after >= len(state_arr):
            continue
        s = float(state_arr[t_after])
        d = float(dep_time_arr[t_after])
        r = float(req_soc_arr[t_after])
        if s == 1.0 and d == 0.0:
            # Departure detected! Read EV battery SoC
            ev = getattr(ch, "connected_electric_vehicle", None)
            actual_soc = 0.0
            if ev is not None:
                batt = getattr(ev, "battery", None)
                if batt is not None:
                    soc_series = getattr(batt, "soc", None)
                    if soc_series is not None and hasattr(soc_series, "__len__"):
                        if 0 <= t_soc < len(soc_series):
                            actual_soc = float(np.clip(soc_series[t_soc], 0.0, 1.0))
            departures.append((cid, actual_soc, r))

    if terminated or truncated:
        print(f"Episode ended at step {step_i + 1}")
        break

    if (step_i + 1) % 1000 == 0:
        dep_so_far = len(departures)
        viol_so_far = sum(1 for _, ds, dr in departures if ds < dr)
        print(f"  Step {step_i + 1}/{total_steps}: departures={dep_so_far}, "
              f"violated={viol_so_far}, grid_import_avg={total_grid_import / (step_i + 1):.2f}")

# ---------------------------------------------------------------------------
# 8) Print results
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("  R19 EVALUATION RESULTS (epoch-80, full year)")
print("=" * 60)

# C0: EV departure
n_departures = len(departures)
n_violated = sum(1 for _, ds, dr in departures if ds < dr)
viol_pct = 100.0 * n_violated / max(n_departures, 1)
mean_dep_soc = np.mean([ds for _, ds, _ in departures]) if departures else 0.0
mean_req_soc = np.mean([dr for _, _, dr in departures]) if departures else 0.0
mean_deficit = np.mean([max(0, dr - ds) for _, ds, dr in departures]) if departures else 0.0

print(f"\n--- C0: EV Departure Constraint ---")
print(f"  Total departures:    {n_departures}")
print(f"  Violated:            {n_violated}")
print(f"  Violation %:         {viol_pct:.1f}%")
print(f"  Mean departure SoC:  {mean_dep_soc:.4f}")
print(f"  Mean required SoC:   {mean_req_soc:.4f}")
print(f"  Mean deficit (when violated): {mean_deficit:.4f}")

# Per-charger breakdown
print(f"\n  Per-charger breakdown:")
for cname in sorted(set(cn for cn, _, _ in departures)):
    ch_deps = [(ds, dr) for cn, ds, dr in departures if cn == cname]
    ch_viols = sum(1 for ds, dr in ch_deps if ds < dr)
    ch_mean_soc = np.mean([ds for ds, _ in ch_deps])
    ch_mean_req = np.mean([dr for _, dr in ch_deps])
    print(f"    {cname}: {len(ch_deps)} departures, {ch_viols} violated "
          f"({100*ch_viols/max(len(ch_deps),1):.1f}%), "
          f"mean_soc={ch_mean_soc:.4f}, mean_req={ch_mean_req:.4f}")

# Battery
print(f"\n--- Battery ---")
bt = max(batt_total_steps, 1)
print(f"  Charge %:            {100.0 * batt_charge_steps / bt:.1f}%")
print(f"  Discharge %:         {100.0 * batt_discharge_steps / bt:.1f}%")
print(f"  Idle %:              {100.0 * (bt - batt_charge_steps - batt_discharge_steps) / bt:.1f}%")
avg_solar = total_solar_gen / total_steps
avg_peak = np.mean(np.maximum(peak_values, 0))
print(f"  Solar avg (kW):      {avg_solar:.2f}")
print(f"  Grid import avg:     {total_grid_import / total_steps:.2f} kW")
print(f"  Grid export avg:     {total_grid_export / total_steps:.2f} kW")
print(f"  Peak import avg:     {avg_peak:.2f} kW")

# EV
print(f"\n--- EV ---")
et = max(ev_total_steps, 1)
print(f"  Charge %:            {100.0 * ev_charge_steps / et:.1f}%")
print(f"  V2G %:               {100.0 * ev_v2g_steps / et:.1f}%")
peak_ev_total = sum(1 for _ in range(total_steps) if 17 <= (_ % 24) <= 21) * len(ev_indices)
print(f"  V2G at peak steps:   {ev_v2g_peak_steps} (of {peak_ev_total} peak EV steps)")

# C3
print(f"\n--- C3: Building Power Violation ---")
c3_pct = 100.0 * c3_violations / max(c3_total, 1)
print(f"  Violations:          {c3_violations}/{c3_total}")
print(f"  Violation %:         {c3_pct:.1f}%")
print(f"  P_building_max:      {P_building_max}")
print(f"  C3 controllable:     {c3_controllable}")

# C4
print(f"\n--- C4: Grid Import Violation ---")
c4_pct = 100.0 * c4_violations / max(c4_total, 1)
print(f"  Violations:          {c4_violations}/{c4_total}")
print(f"  Violation %:         {c4_pct:.1f}%")
print(f"  P_grid_max:          {P_grid_max}")

# Summary totals
print(f"\n--- Summary ---")
print(f"  Total grid import:   {total_grid_import:.1f} kWh")
print(f"  Total grid export:   {total_grid_export:.1f} kWh")
print(f"  Total solar gen:     {total_solar_gen:.1f} kWh")
print(f"  Net consumption:     {total_grid_import - total_grid_export:.1f} kWh")
