#!/usr/bin/env python3
"""
Debug C0: Why does 91% charging produce 65% departure violations?
Traces every EV session: raw action, masked action, SoC trajectory, energy charged.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# =====================================================================
# ENV VARS -- EXACT copy from run_1bld_test_v2.sh
# =====================================================================
# Clear all first
for k in list(os.environ.keys()):
    if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
        os.environ.pop(k, None)

os.environ["CITYLEARN_SCHEMA"] = f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_1building_3month.json"
os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"
os.environ["CITYLEARN_EXPORT_FACTOR"] = "0.7"

os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "10.2352"
os.environ["CITYLEARN_STEMS_SOC_LOW"] = "0.0"
os.environ["CITYLEARN_STEMS_SOC_HIGH"] = "0.95"
os.environ["CITYLEARN_STEMS_PNORM_P"] = "4.0"

os.environ["CITYLEARN_ACTION_MASK"] = "1"

os.environ["CITYLEARN_KL_BETA"] = "0.1"
os.environ["CITYLEARN_KL_BETA_DECAY"] = "0.995"

os.environ["STEMS_ALPHA_NEC_SIGN"] = "3.0"
os.environ["STEMS_ALPHA_PRICE_ARB"] = "1.0"
os.environ["STEMS_LAMBDA_EV"] = "15.0"
os.environ["STEMS_EV_SLACK_ARB_SCALE"] = "2.5"
os.environ["STEMS_ALPHA_GRID_PENALTY"] = "1.5"

os.environ["CITYLEARN_STEMS_BATTERY_COST_SCALE"] = "1.0"

os.environ["STEMS_MU_ECONOMIC"] = "0.0"
os.environ["STEMS_ALPHA_GRID"] = "0.0"
os.environ["STEMS_ALPHA_BUILD"] = "0.0"
os.environ["STEMS_BETA_RAMP"] = "0.0"
os.environ["STEMS_XI_RENEWABLE"] = "0.0"
os.environ["STEMS_ALPHA_LOAD_SHIFT"] = "0.0"
os.environ["STEMS_ALPHA_GRID_MILD"] = "0.0"
os.environ["STEMS_ALPHA_EV_GUARD"] = "0.0"
os.environ["STEMS_ALPHA_V2G_CONTEXT"] = "0.0"
os.environ["STEMS_ALPHA_PEAK_SHAVE"] = "0.0"
os.environ["STEMS_ALPHA_BARRIER"] = "0.0"
os.environ["STEMS_ALPHA_EV_SOLAR"] = "0.0"
os.environ["STEMS_ALPHA_SOLAR_STORE"] = "0.0"
os.environ["STEMS_SOLAR_STORE_BATT_ONLY"] = "0"
os.environ["STEMS_ALPHA_HEADROOM"] = "0.0"
os.environ["STEMS_SB_ASYMMETRIC"] = "1"
os.environ["STEMS_SG_EXPORT_CREDIT"] = "0.5"
os.environ["STEMS_SG_THRESHOLD"] = "0.5"

os.environ["CITYLEARN_EV_SAUTE"] = "1"
os.environ["CITYLEARN_EV_SAUTE_BUDGET"] = "25000"
os.environ["CITYLEARN_EV_SAUTE_PENALTY"] = "5.0"
os.environ["CITYLEARN_EV_SAUTE_GAMMA"] = "1.0"
os.environ["CITYLEARN_EV_SAUTE_SHAPED_ALPHA"] = "2.0"

os.environ["CITYLEARN_PID_LAGRANGE"] = "1"

os.environ["COST_W_C2"] = "0.0"
os.environ["COST_W_C3"] = "5.0"
os.environ["CITYLEARN_W_COST_EV"] = "1.0"
os.environ["CITYLEARN_W_COST_SOC"] = "10.0"
os.environ["CITYLEARN_W_COST_BUILDING"] = "0.5"
os.environ["CITYLEARN_W_COST_GRID"] = "0.05"
os.environ["CITYLEARN_EV_COST_SCALE"] = "3.0"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_full"
os.environ["CITYLEARN_INCLUDE_EV_COST"] = "1"

os.environ["CITYLEARN_WM_DISABLE"] = "1"
os.environ["CITYLEARN_EV_ACTION_CLAMP"] = "0"
os.environ["CITYLEARN_BATT_CLAMP"] = "0"
os.environ["CITYLEARN_SPATIAL_OBS"] = "0"
os.environ["CITYLEARN_TEMPORAL_WINDOW"] = "0"
os.environ["CITYLEARN_C3_CONTROLLABLE"] = "1"

# =====================================================================
# Model
# =====================================================================
class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.mean = nn.Sequential(*layers)

    def forward(self, obs):
        return torch.tanh(self.mean(obs))


def load_actor(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi_state = ckpt["pi"]
    h1 = pi_state["mean.0.weight"].shape[0]
    h2 = pi_state["mean.2.weight"].shape[0]
    obs_dim = pi_state["mean.0.weight"].shape[1]
    act_dim = pi_state["mean.4.weight"].shape[0]
    actor = MLPActor(obs_dim, act_dim, (h1, h2))
    filtered = {k: v for k, v in pi_state.items() if not k.startswith("log_std")}
    result = actor.load_state_dict(filtered, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Missing keys: {result.missing_keys}")
    actor.eval()

    obs_mean = obs_std = obs_clip = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        clip_t = norm["_clip"].float()
        obs_clip = float(clip_t.mean())

    log_std = pi_state.get("log_std", None)
    if log_std is not None:
        print(f"  Policy log_std: {log_std.numpy()}")
        print(f"  Policy std:     {log_std.exp().numpy()}")

    return actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim


# =====================================================================
# Find latest checkpoint
# =====================================================================
CKPT_DIR = (
    f"{PROJECT}/runs/test_1bld_v2/"
    "PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-19-22-56-55/torch_save"
)

# Find latest epoch
import glob
ckpts = glob.glob(os.path.join(CKPT_DIR, "epoch-*.pt"))
if not ckpts:
    print(f"No checkpoints found in {CKPT_DIR}")
    sys.exit(1)
epochs = [int(os.path.basename(c).replace("epoch-", "").replace(".pt", "")) for c in ckpts]
latest_epoch = max(epochs)
ckpt_path = os.path.join(CKPT_DIR, f"epoch-{latest_epoch}.pt")
print(f"Loading checkpoint: {ckpt_path}")

actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(ckpt_path)
print(f"  Actor: obs_dim={obs_dim}, act_dim={act_dim}")
print(f"  Obs normalizer: {'YES' if obs_mean is not None else 'NO'}")

# =====================================================================
# Create environment (same wrapper chain as training)
# =====================================================================
from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.action_mask_wrapper import ActionMaskWrapper
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env

base = make_base_env(central_agent=True)
safety = CityLearnSafetyEnvV3(base)
forecast = ForecastObsWrapper(safety, forecast_horizon=24)
env = ActionMaskWrapper(forecast)

raw = unwrap_to_raw_citylearn_env(env)
buildings = list(getattr(raw, "buildings", []))
n_buildings = len(buildings)

# --- Identify action indices ---
names_raw = getattr(raw, "action_names", [])
if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
    names = names_raw[0]
else:
    names = list(names_raw)

batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]

print(f"\n  Buildings: {n_buildings}")
print(f"  Obs dim (env): {env.observation_space.shape[0]}  (model expects: {obs_dim})")
print(f"  Act dim: {act_dim}")
print(f"  Action layout:")
for i, n in enumerate(names):
    tag = ""
    if i in batt_idx: tag = "  [BATTERY]"
    elif i in ev_idx: tag = "  [EV]"
    print(f"    [{i}] {n}{tag}")

# Handle obs dim mismatch
env_obs_dim = env.observation_space.shape[0]
need_pad = obs_dim > env_obs_dim
pad_dim = obs_dim - env_obs_dim if need_pad else 0
need_trim = obs_dim < env_obs_dim

# --- EV charger specs ---
print(f"\n  EV CHARGER SPECS:")
for b_idx, bld in enumerate(buildings):
    chargers = getattr(bld, "electric_vehicle_chargers", []) or []
    for ch_idx, ch in enumerate(chargers):
        print(f"    Building {b_idx}, Charger {ch_idx}:")
        print(f"      max_charging_power:    {ch.max_charging_power} kW")
        print(f"      min_charging_power:    {ch.min_charging_power} kW")
        print(f"      max_discharging_power: {ch.max_discharging_power} kW")
        print(f"      min_discharging_power: {ch.min_discharging_power} kW")
        print(f"      efficiency:            {ch.efficiency}")
        ev_obj = getattr(ch, 'connected_electric_vehicle', None)
        if ev_obj:
            bt = getattr(ev_obj, 'battery', None)
            if bt:
                print(f"      EV battery capacity:   {bt.capacity} kWh")
                print(f"      EV battery nominal_power: {bt.nominal_power} kW")

# =====================================================================
# ROLLOUT
# =====================================================================
obs, _ = env.reset(seed=42)
done = False
step = 0

# Session tracking: (b_idx, ch_idx) -> session data
sessions = []  # completed sessions
active_sessions = {}  # key -> session dict

# Track EV charger objects for SoC reading
def get_ev_info(bld, ch, t_now, t_idx):
    """Get EV SoC, required SoC, departure time, connected state."""
    sim = getattr(ch, 'charger_simulation',
                  getattr(ch, '_Charger__charger_simulation', None))
    if sim is None:
        return None

    sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
    connected = t_now < len(sa) and float(sa[t_now]) == 1.0

    ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
    dt = np.asarray(getattr(sim, '_electric_vehicle_departure_time'), dtype=float)

    req_soc = float(ra[t_now]) if t_now < len(ra) else 1.0
    if not np.isfinite(req_soc):
        req_soc = 1.0
    dep_time = float(dt[t_now]) if t_now < len(dt) else -1

    ev_soc = 0.0
    ev_capacity = 0.0
    if connected:
        ev_obj = getattr(ch, 'connected_electric_vehicle', None)
        if ev_obj is not None:
            bt = getattr(ev_obj, 'battery', None)
            if bt is not None:
                soc_arr = getattr(bt, 'soc', None)
                if soc_arr is not None:
                    sn = np.asarray(soc_arr, dtype=float)
                    ev_soc = float(np.clip(sn[t_idx], 0, 1)) if 0 <= t_idx < len(sn) else 0.0
                ev_capacity = float(getattr(bt, 'capacity', 0))

    return {
        'connected': connected,
        'ev_soc': ev_soc,
        'required_soc': req_soc,
        'departure_time': dep_time,
        'ev_capacity': ev_capacity,
    }


print(f"\n{'='*80}")
print(f"  RUNNING FULL ROLLOUT...")
print(f"{'='*80}")

max_steps = 9000  # full year = 8759 for 3-month schema = 2190

while not done and step < max_steps:
    # --- Prepare observation for model ---
    obs_np = np.asarray(obs, dtype=np.float32).ravel()
    if need_pad:
        obs_np = np.concatenate([obs_np, np.zeros(pad_dim, dtype=np.float32)])
    elif need_trim:
        obs_np = obs_np[:obs_dim]

    obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
    if obs_mean is not None:
        obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
        if obs_clip is not None:
            obs_t = obs_t.clamp(-obs_clip, obs_clip)

    with torch.no_grad():
        action_t = actor(obs_t).squeeze(0).numpy()
    raw_action = np.clip(action_t, -1.0, 1.0)

    t_now = int(getattr(raw, "time_step", 0))
    t_idx = max(0, t_now - 1)
    hour = t_now % 24

    # --- Record action mask state BEFORE step ---
    # We need to peek at the mask bounds that will be applied
    exo_nec = env._get_exogenous_nec()
    socs = env._get_battery_socs()
    safe_min, safe_max, _ = env._compute_safe_bounds(exo_nec, socs)

    # The masked (rescaled) action
    masked_action = ActionMaskWrapper._rescale(raw_action, safe_min, safe_max)

    # --- EV session tracking (BEFORE step, using current state) ---
    for b_idx, bld in enumerate(buildings):
        chargers = getattr(bld, "electric_vehicle_chargers", []) or []
        for ch_idx, ch in enumerate(chargers):
            ev_info = get_ev_info(bld, ch, t_now, t_idx)
            if ev_info is None:
                continue

            key = (b_idx, ch_idx)
            ev_act_idx = ev_idx[0] if ev_idx else None  # 1 building, 1 charger

            if ev_info['connected']:
                if key not in active_sessions:
                    # New session starts
                    active_sessions[key] = {
                        'start_hour': t_now,
                        'initial_soc': ev_info['ev_soc'],
                        'required_soc': ev_info['required_soc'],
                        'departure_time': ev_info['departure_time'],
                        'ev_capacity': ev_info['ev_capacity'],
                        'hourly': [],
                    }
                # Record this hour's data
                sess = active_sessions[key]
                raw_ev = float(raw_action[ev_act_idx]) if ev_act_idx is not None else 0.0
                masked_ev = float(masked_action[ev_act_idx]) if ev_act_idx is not None else 0.0
                ev_safe_min = float(safe_min[ev_act_idx]) if ev_act_idx is not None else 0.0
                ev_safe_max = float(safe_max[ev_act_idx]) if ev_act_idx is not None else 0.0
                sess['hourly'].append({
                    'step': step,
                    't_now': t_now,
                    'hour': hour,
                    'raw_action': raw_ev,
                    'masked_action': masked_ev,
                    'safe_min': ev_safe_min,
                    'safe_max': ev_safe_max,
                    'soc_before_step': ev_info['ev_soc'],
                    'required_soc': ev_info['required_soc'],
                    'exo_nec': float(exo_nec[b_idx]) if b_idx < len(exo_nec) else 0.0,
                })
            else:
                if key in active_sessions:
                    # Session just ended (departure)
                    sess = active_sessions.pop(key)
                    # Get the final SoC (from the last connected step)
                    if sess['hourly']:
                        sess['final_soc'] = sess['hourly'][-1]['soc_before_step']
                        sess['end_hour'] = sess['hourly'][-1]['t_now']
                    else:
                        sess['final_soc'] = sess['initial_soc']
                        sess['end_hour'] = sess['start_hour']

                    sess['violated'] = sess['final_soc'] < sess['required_soc'] - 0.01
                    sess['deficit'] = max(0, sess['required_soc'] - sess['final_soc'])

                    # Compute stats
                    raw_acts = [h['raw_action'] for h in sess['hourly']]
                    masked_acts = [h['masked_action'] for h in sess['hourly']]
                    safe_mins = [h['safe_min'] for h in sess['hourly']]
                    safe_maxs = [h['safe_max'] for h in sess['hourly']]

                    sess['n_hours'] = len(sess['hourly'])
                    sess['n_charge'] = sum(1 for a in masked_acts if a > 0.1)
                    sess['n_discharge'] = sum(1 for a in masked_acts if a < -0.1)
                    sess['n_idle'] = sum(1 for a in masked_acts if abs(a) <= 0.1)
                    sess['mean_raw_action'] = np.mean(raw_acts) if raw_acts else 0.0
                    sess['mean_masked_action'] = np.mean(masked_acts) if masked_acts else 0.0
                    sess['mean_safe_min'] = np.mean(safe_mins) if safe_mins else 0.0
                    sess['mean_safe_max'] = np.mean(safe_maxs) if safe_maxs else 0.0
                    sess['min_safe_max'] = min(safe_maxs) if safe_maxs else 0.0
                    sess['max_safe_min'] = max(safe_mins) if safe_mins else 0.0

                    # Estimate energy charged
                    max_charge_kw = 11.0
                    efficiency = 0.95
                    capacity = sess['ev_capacity'] if sess['ev_capacity'] > 0 else 60.0
                    total_energy = 0.0
                    for h in sess['hourly']:
                        a = h['masked_action']
                        if a > 0:
                            # CityLearn: power = action * max_charging_power
                            # energy = power * 1hr = action * 11 kWh
                            # But clamped to [min_charge, max_charge]
                            energy_raw = a * max_charge_kw
                            energy_clamped = max(min(energy_raw, max_charge_kw), 1.4)  # min_charge=1.4
                            total_energy += energy_clamped * efficiency
                        elif a < 0:
                            energy_raw = a * 7.2  # max_discharge = 7.2
                            total_energy += energy_raw / efficiency
                    sess['est_energy_kwh'] = total_energy
                    sess['soc_change_needed'] = (sess['required_soc'] - sess['initial_soc']) * capacity
                    sess['soc_change_actual'] = (sess['final_soc'] - sess['initial_soc']) * capacity

                    sessions.append(sess)

    # --- Step environment ---
    obs, reward, terminated, truncated, info = env.step(raw_action)
    done = terminated or truncated
    step += 1

    # Update SoC AFTER step for active sessions
    for b_idx, bld in enumerate(buildings):
        chargers = getattr(bld, "electric_vehicle_chargers", []) or []
        for ch_idx, ch in enumerate(chargers):
            key = (b_idx, ch_idx)
            if key in active_sessions:
                t_after = int(getattr(raw, "time_step", 0))
                t_idx_after = max(0, t_after - 1)
                ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                if ev_obj:
                    bt = getattr(ev_obj, 'battery', None)
                    if bt:
                        soc_arr = getattr(bt, 'soc', None)
                        if soc_arr is not None:
                            sn = np.asarray(soc_arr, dtype=float)
                            soc_after = float(np.clip(sn[t_idx_after], 0, 1)) if t_idx_after < len(sn) else 0.0
                            if active_sessions[key]['hourly']:
                                active_sessions[key]['hourly'][-1]['soc_after_step'] = soc_after

    if step % 500 == 0:
        print(f"  Step {step}, sessions completed: {len(sessions)}")

print(f"\nRollout complete: {step} steps, {len(sessions)} sessions")

# =====================================================================
# ANALYSIS
# =====================================================================
print(f"\n{'='*80}")
print(f"  SESSION ANALYSIS")
print(f"{'='*80}")

n_total = len(sessions)
n_violated = sum(1 for s in sessions if s['violated'])
n_ok = n_total - n_violated
print(f"\n  Total departures:     {n_total}")
print(f"  Violated (SoC < req): {n_violated} ({100*n_violated/max(1,n_total):.1f}%)")
print(f"  OK:                   {n_ok} ({100*n_ok/max(1,n_total):.1f}%)")

# A. Is the agent actually charging?
all_masked = []
all_raw = []
all_safe_mins = []
all_safe_maxs = []
for s in sessions:
    for h in s['hourly']:
        all_masked.append(h['masked_action'])
        all_raw.append(h['raw_action'])
        all_safe_mins.append(h['safe_min'])
        all_safe_maxs.append(h['safe_max'])

all_masked = np.array(all_masked)
all_raw = np.array(all_raw)
all_safe_mins = np.array(all_safe_mins)
all_safe_maxs = np.array(all_safe_maxs)

print(f"\n  --- A. RAW vs MASKED action distribution (connected hours only) ---")
print(f"  Raw action:    mean={all_raw.mean():.4f}, std={all_raw.std():.4f}, min={all_raw.min():.4f}, max={all_raw.max():.4f}")
print(f"  Masked action: mean={all_masked.mean():.4f}, std={all_masked.std():.4f}, min={all_masked.min():.4f}, max={all_masked.max():.4f}")
print(f"  Safe bounds:   min_of_safe_min={all_safe_mins.min():.4f}, max_of_safe_min={all_safe_mins.max():.4f}")
print(f"                 min_of_safe_max={all_safe_maxs.min():.4f}, max_of_safe_max={all_safe_maxs.max():.4f}")
print(f"  Mean safe range: [{all_safe_mins.mean():.4f}, {all_safe_maxs.mean():.4f}]")

n_mask_charge = np.sum(all_masked > 0.1)
n_mask_discharge = np.sum(all_masked < -0.1)
n_mask_idle = np.sum(np.abs(all_masked) <= 0.1)
n_mask_any_positive = np.sum(all_masked > 0.0)
n_mask_tiny = np.sum((all_masked > 0.0) & (all_masked <= 0.1))
print(f"\n  Masked action breakdown (connected hours):")
print(f"    Charge (>0.1):     {n_mask_charge} ({100*n_mask_charge/len(all_masked):.1f}%)")
print(f"    Tiny pos (0,0.1]:  {n_mask_tiny} ({100*n_mask_tiny/len(all_masked):.1f}%)")
print(f"    Any positive (>0): {n_mask_any_positive} ({100*n_mask_any_positive/len(all_masked):.1f}%)")
print(f"    Idle (|a|<=0.1):   {n_mask_idle} ({100*n_mask_idle/len(all_masked):.1f}%)")
print(f"    Discharge (<-0.1): {n_mask_discharge} ({100*n_mask_discharge/len(all_masked):.1f}%)")

n_raw_charge = np.sum(all_raw > 0.1)
print(f"\n  Raw action breakdown (what agent WANTED):")
print(f"    Charge (>0.1):     {n_raw_charge} ({100*n_raw_charge/len(all_raw):.1f}%)")
print(f"    Any positive (>0): {np.sum(all_raw > 0)} ({100*np.sum(all_raw > 0)/len(all_raw):.1f}%)")

# B. Charge magnitude
print(f"\n  --- B. Charge magnitude analysis ---")
pos_masked = all_masked[all_masked > 0]
if len(pos_masked) > 0:
    print(f"  When charging: mean={pos_masked.mean():.4f}, median={np.median(pos_masked):.4f}, "
          f"std={pos_masked.std():.4f}")
    print(f"  Histogram of charge magnitudes:")
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    counts, _ = np.histogram(pos_masked, bins=bins)
    for i in range(len(bins)-1):
        print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {counts[i]} ({100*counts[i]/len(pos_masked):.1f}%)")

# C. SoC trajectory
print(f"\n  --- C. SoC trajectory during sessions ---")
soc_increases = 0
soc_decreases = 0
soc_flat = 0
for s in sessions:
    for h in s['hourly']:
        soc_after = h.get('soc_after_step', h['soc_before_step'])
        delta = soc_after - h['soc_before_step']
        if delta > 0.001:
            soc_increases += 1
        elif delta < -0.001:
            soc_decreases += 1
        else:
            soc_flat += 1
n_total_hours = soc_increases + soc_decreases + soc_flat
print(f"  SoC increased: {soc_increases} ({100*soc_increases/max(1,n_total_hours):.1f}%)")
print(f"  SoC decreased: {soc_decreases} ({100*soc_decreases/max(1,n_total_hours):.1f}%)")
print(f"  SoC unchanged: {soc_flat} ({100*soc_flat/max(1,n_total_hours):.1f}%)")

# D. Energy analysis per session
print(f"\n  --- D. Energy charged per session ---")
violated_sessions = [s for s in sessions if s['violated']]
ok_sessions = [s for s in sessions if not s['violated']]

for label, subset in [("VIOLATED", violated_sessions), ("OK", ok_sessions)]:
    if not subset:
        continue
    energies = [s['est_energy_kwh'] for s in subset]
    needed = [s['soc_change_needed'] for s in subset]
    durations = [s['n_hours'] for s in subset]
    deficits = [s.get('deficit', 0) for s in subset]
    init_socs = [s['initial_soc'] for s in subset]
    req_socs = [s['required_soc'] for s in subset]
    final_socs = [s['final_soc'] for s in subset]
    mean_masked = [s['mean_masked_action'] for s in subset]
    mean_safe_maxs = [s['mean_safe_max'] for s in subset]

    print(f"\n  {label} sessions (n={len(subset)}):")
    print(f"    Duration:       mean={np.mean(durations):.1f}, min={min(durations)}, max={max(durations)}")
    print(f"    Initial SoC:    mean={np.mean(init_socs):.3f}, min={min(init_socs):.3f}")
    print(f"    Required SoC:   mean={np.mean(req_socs):.3f}, min={min(req_socs):.3f}")
    print(f"    Final SoC:      mean={np.mean(final_socs):.3f}, max={max(final_socs):.3f}")
    print(f"    Deficit:        mean={np.mean(deficits):.3f}, max={max(deficits):.3f}")
    print(f"    Est energy kWh: mean={np.mean(energies):.1f}, min={min(energies):.1f}")
    print(f"    Needed kWh:     mean={np.mean(needed):.1f}")
    print(f"    Mean masked act:mean={np.mean(mean_masked):.4f}")
    print(f"    Mean safe_max:  mean={np.mean(mean_safe_maxs):.4f}")

# E. Temporal pattern
print(f"\n  --- E. Temporal pattern of violations ---")
violated_starts = [s['start_hour'] for s in violated_sessions]
ok_starts = [s['start_hour'] for s in ok_sessions]
if violated_starts:
    print(f"  Violated session start hours: min={min(violated_starts)}, max={max(violated_starts)}")
    # Bucket by month (rough: 730 hours per month)
    months_v = [h // 730 for h in violated_starts]
    months_ok = [h // 730 for h in ok_starts]
    for m in range(4):  # 3-month schema ~= 3 months
        nv = months_v.count(m)
        nok = months_ok.count(m)
        print(f"    Month {m}: violated={nv}, ok={nok}, rate={100*nv/max(1,nv+nok):.0f}%")

# =====================================================================
# FIRST 10 VIOLATED DEPARTURES - DETAILED TRACE
# =====================================================================
print(f"\n{'='*80}")
print(f"  DETAILED TRACE: FIRST 10 VIOLATED DEPARTURES")
print(f"{'='*80}")

for i, s in enumerate(violated_sessions[:10]):
    print(f"\n  --- Violated Session #{i+1} ---")
    print(f"  Start: hour {s['start_hour']}, End: hour {s['end_hour']}")
    print(f"  Duration: {s['n_hours']} hours")
    print(f"  Initial SoC: {s['initial_soc']:.4f}")
    print(f"  Required SoC: {s['required_soc']:.4f}")
    print(f"  Final SoC: {s['final_soc']:.4f}")
    print(f"  Deficit: {s['deficit']:.4f}")
    print(f"  EV capacity: {s['ev_capacity']:.1f} kWh")
    print(f"  Estimated energy: {s['est_energy_kwh']:.1f} kWh (needed: {s['soc_change_needed']:.1f} kWh)")
    print(f"  Hours charged (mask>0.1): {s['n_charge']}, discharged: {s['n_discharge']}, idle: {s['n_idle']}")
    print(f"  Mean safe range: [{s['mean_safe_min']:.4f}, {s['mean_safe_max']:.4f}]")
    print(f"  Min safe_max across session: {s['min_safe_max']:.4f}")

    print(f"\n  Hour-by-hour trace:")
    print(f"  {'step':>6} {'t':>5} {'hr':>3} {'raw':>8} {'s_min':>8} {'s_max':>8} {'masked':>8} {'SoC_bef':>8} {'SoC_aft':>8} {'delta':>8} {'req':>8} {'gap':>8}")
    for h in s['hourly']:
        soc_after = h.get('soc_after_step', h['soc_before_step'])
        delta_soc = soc_after - h['soc_before_step']
        gap = h['required_soc'] - soc_after
        print(f"  {h['step']:>6} {h['t_now']:>5} {h['hour']:>3} {h['raw_action']:>8.4f} "
              f"{h['safe_min']:>8.4f} {h['safe_max']:>8.4f} {h['masked_action']:>8.4f} "
              f"{h['soc_before_step']:>8.4f} {soc_after:>8.4f} {delta_soc:>8.4f} "
              f"{h['required_soc']:>8.4f} {gap:>8.4f}")

# =====================================================================
# ALSO SHOW 3 OK DEPARTURES FOR COMPARISON
# =====================================================================
print(f"\n{'='*80}")
print(f"  COMPARISON: FIRST 3 OK DEPARTURES")
print(f"{'='*80}")

for i, s in enumerate(ok_sessions[:3]):
    print(f"\n  --- OK Session #{i+1} ---")
    print(f"  Start: hour {s['start_hour']}, End: hour {s['end_hour']}")
    print(f"  Duration: {s['n_hours']} hours")
    print(f"  Initial SoC: {s['initial_soc']:.4f}")
    print(f"  Required SoC: {s['required_soc']:.4f}")
    print(f"  Final SoC: {s['final_soc']:.4f}")
    print(f"  EV capacity: {s['ev_capacity']:.1f} kWh")
    print(f"  Hours charged (mask>0.1): {s['n_charge']}, discharged: {s['n_discharge']}, idle: {s['n_idle']}")
    print(f"  Mean safe range: [{s['mean_safe_min']:.4f}, {s['mean_safe_max']:.4f}]")

    print(f"\n  Hour-by-hour trace (first/last 5):")
    print(f"  {'step':>6} {'t':>5} {'hr':>3} {'raw':>8} {'s_min':>8} {'s_max':>8} {'masked':>8} {'SoC_bef':>8} {'SoC_aft':>8} {'delta':>8} {'req':>8}")
    show = s['hourly'][:5] + s['hourly'][-5:] if len(s['hourly']) > 10 else s['hourly']
    for h in show:
        soc_after = h.get('soc_after_step', h['soc_before_step'])
        delta_soc = soc_after - h['soc_before_step']
        print(f"  {h['step']:>6} {h['t_now']:>5} {h['hour']:>3} {h['raw_action']:>8.4f} "
              f"{h['safe_min']:>8.4f} {h['safe_max']:>8.4f} {h['masked_action']:>8.4f} "
              f"{h['soc_before_step']:>8.4f} {soc_after:>8.4f} {delta_soc:>8.4f} "
              f"{h['required_soc']:>8.4f}")

# =====================================================================
# SUMMARY ANSWERS
# =====================================================================
print(f"\n{'='*80}")
print(f"  DIAGNOSTIC ANSWERS")
print(f"{'='*80}")

# A
mask_zeroed = np.sum((all_raw > 0.1) & (all_masked <= 0.0))
print(f"\n  A. Is mask converting charge to zero?")
print(f"     Agent wanted charge (raw>0.1) but mask gave <=0: {mask_zeroed}/{len(all_raw)} "
      f"({100*mask_zeroed/max(1,len(all_raw)):.1f}%)")

# B
print(f"\n  B. Is charge magnitude sufficient?")
if len(pos_masked) > 0:
    tiny = np.sum(pos_masked < 0.05)
    print(f"     Tiny charges (<0.05): {tiny}/{len(pos_masked)} ({100*tiny/max(1,len(pos_masked)):.1f}%)")
    print(f"     With 11kW charger: action=0.05 -> 0.55 kW -> clamped to 1.4 kW min")
    print(f"     Even tiny actions give 1.4 kWh/hr * 0.95 eff = 1.33 kWh/hr")

# C
print(f"\n  C. Is SoC actually increasing during charge?")
charge_hours_soc_up = 0
charge_hours_soc_flat = 0
charge_hours_soc_down = 0
for s in sessions:
    for h in s['hourly']:
        if h['masked_action'] > 0.01:
            soc_after = h.get('soc_after_step', h['soc_before_step'])
            delta = soc_after - h['soc_before_step']
            if delta > 0.001: charge_hours_soc_up += 1
            elif delta < -0.001: charge_hours_soc_down += 1
            else: charge_hours_soc_flat += 1
total_ch = charge_hours_soc_up + charge_hours_soc_flat + charge_hours_soc_down
print(f"     During charge actions (masked>0.01):")
print(f"       SoC UP:   {charge_hours_soc_up} ({100*charge_hours_soc_up/max(1,total_ch):.1f}%)")
print(f"       SoC FLAT: {charge_hours_soc_flat} ({100*charge_hours_soc_flat/max(1,total_ch):.1f}%)")
print(f"       SoC DOWN: {charge_hours_soc_down} ({100*charge_hours_soc_down/max(1,total_ch):.1f}%)")

# D
print(f"\n  D. kWh charged per session vs needed:")
if violated_sessions:
    v_energies = [s['est_energy_kwh'] for s in violated_sessions]
    v_needed = [s['soc_change_needed'] for s in violated_sessions]
    print(f"     Violated: charged={np.mean(v_energies):.1f} kWh, needed={np.mean(v_needed):.1f} kWh")
if ok_sessions:
    o_energies = [s['est_energy_kwh'] for s in ok_sessions]
    o_needed = [s['soc_change_needed'] for s in ok_sessions]
    print(f"     OK:       charged={np.mean(o_energies):.1f} kWh, needed={np.mean(o_needed):.1f} kWh")

# E
print(f"\n  E. Violation pattern:")
if violated_sessions:
    v_durations = [s['n_hours'] for s in violated_sessions]
    o_durations = [s['n_hours'] for s in ok_sessions] if ok_sessions else [0]
    print(f"     Violated mean duration: {np.mean(v_durations):.1f} hours")
    print(f"     OK mean duration:       {np.mean(o_durations):.1f} hours")
    v_req = [s['required_soc'] for s in violated_sessions]
    o_req = [s['required_soc'] for s in ok_sessions] if ok_sessions else [0]
    print(f"     Violated mean required SoC: {np.mean(v_req):.3f}")
    print(f"     OK mean required SoC:       {np.mean(o_req):.3f}")

print(f"\n  DONE.")
