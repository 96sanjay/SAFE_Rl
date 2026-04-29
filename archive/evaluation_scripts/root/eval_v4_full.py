#!/usr/bin/env python3
"""
PPO v4 1-Building COMPREHENSIVE Evaluation
============================================
Runs 1 full year (8759 steps) with deterministic actions and computes
every possible metric with exact numbers.

Usage:
    python eval_v4_full.py [--epoch 50] [--seed 42]
"""
import os

# =====================================================================
# ENV VARS -- must be set BEFORE any project imports
# =====================================================================
os.environ["CITYLEARN_SCHEMA"] = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/citylearn_challenge_2022_phase_all_plus_evs/schema_1building.json"
os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"
os.environ["CITYLEARN_ACTION_MASK"] = "1"
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_SOC_LOW"] = "0.0"
os.environ["CITYLEARN_STEMS_SOC_HIGH"] = "0.95"
os.environ["CITYLEARN_STEMS_BATTERY_COST_SCALE"] = "1.0"
os.environ["CITYLEARN_EXPORT_FACTOR"] = "0.7"
os.environ["CITYLEARN_EV_SAUTE"] = "0"
os.environ["CITYLEARN_EV_DENSE_COST_SCALE"] = "1.0"
os.environ["STEMS_ALPHA_NEC_SIGN"] = "3.0"
os.environ["STEMS_ALPHA_PRICE_ARB"] = "1.0"
os.environ["STEMS_LAMBDA_EV"] = "15.0"
os.environ["STEMS_EV_SLACK_ARB_SCALE"] = "2.5"
os.environ["STEMS_ALPHA_GRID_PENALTY"] = "1.5"
os.environ["STEMS_MU_ECONOMIC"] = "0.0"
os.environ["STEMS_ALPHA_GRID"] = "0.0"
os.environ["STEMS_ALPHA_BUILD"] = "0.0"
os.environ["STEMS_BETA_RAMP"] = "0.0"
os.environ["STEMS_XI_RENEWABLE"] = "0.0"
os.environ["STEMS_ALPHA_BARRIER"] = "0.0"
os.environ["STEMS_ALPHA_EV_GUARD"] = "0.0"
os.environ["STEMS_ALPHA_V2G_CONTEXT"] = "0.0"
os.environ["STEMS_ALPHA_PEAK_SHAVE"] = "0.0"
os.environ["STEMS_ALPHA_EV_SOLAR"] = "0.0"
os.environ["STEMS_ALPHA_SOLAR_STORE"] = "0.0"
os.environ["STEMS_ALPHA_HEADROOM"] = "0.0"
os.environ["STEMS_ALPHA_LOAD_SHIFT"] = "0.0"
os.environ["STEMS_ALPHA_GRID_MILD"] = "0.0"

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# =====================================================================
# Checkpoint paths
# =====================================================================
CKPT_DIR = (
    f"{PROJECT}/runs/test_1bld_v4/"
    "PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-20-00-56-30/torch_save"
)
OUT_DIR = f"{PROJECT}/runs/test_1bld_v4"
OUT_TXT = f"{OUT_DIR}/eval_v4_full.txt"


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
    """Load actor network and obs normalizer from OmniSafe checkpoint."""
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


def pct(a, b):
    return 100.0 * a / b if b > 0 else 0.0


# =====================================================================
# Output helper -- prints and writes to file
# =====================================================================
_output_lines = []

def out(line=""):
    print(line)
    _output_lines.append(line)

def flush_output(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(_output_lines) + "\n")


# =====================================================================
# Main evaluation
# =====================================================================
def evaluate(epoch=50, seed=42):
    ckpt_path = os.path.join(CKPT_DIR, f"epoch-{epoch}.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    out(f"{'='*80}")
    out(f"  PPO v4 1-BUILDING COMPREHENSIVE EVALUATION")
    out(f"  Checkpoint: epoch-{epoch}.pt")
    out(f"  Seed: {seed}")
    out(f"{'='*80}")

    # --- Load model ---
    actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(ckpt_path)
    out(f"  Actor: obs_dim={obs_dim}, act_dim={act_dim}, hidden=[256,256]")
    out(f"  Obs normalizer: {'YES' if obs_mean is not None else 'NO'}")

    # --- Create environment ---
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

    # --- Action indices ---
    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)

    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]

    out(f"  Buildings: {n_buildings}")
    env_obs_dim = env.observation_space.shape[0]
    out(f"  Obs dim (env): {env_obs_dim}  (model expects: {obs_dim})")
    out(f"  Act dim: {act_dim}")
    out(f"  Action layout:")
    for i, n in enumerate(names):
        tag = ""
        if i in batt_idx:
            tag = "  [BATTERY]"
        elif i in ev_idx:
            tag = "  [EV]"
        out(f"    [{i}] {n}{tag}")

    # Handle obs dim mismatch
    need_pad = obs_dim > env_obs_dim
    pad_dim = obs_dim - env_obs_dim if need_pad else 0
    need_trim = obs_dim < env_obs_dim

    # --- Reset ---
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    # =====================================================================
    # TRACKING VARIABLES
    # =====================================================================
    P_BUILDING_MAX = 4.6083

    rewards = []
    costs = []
    actions_all = []
    c0_vals, c1_vals, c2_vals, c3_vals, c4_vals = [], [], [], [], []

    # Battery
    batt_actions_by_hour = [[] for _ in range(24)]
    batt_soc_all = []  # list of floats (building 0 only)

    # EV
    ev_actions_by_hour = [[] for _ in range(24)]
    ev_connected_mask = []  # bool per step
    ev_soc_all = []  # float per step (when connected)
    ev_actions_when_connected = []  # (hour, action) tuples

    # NEC
    nec_per_building = [[] for _ in range(n_buildings)]
    grid_nec_all = []

    # Price
    price_all = []
    hour_all = []

    # Departure tracking
    total_departures = 0
    violated_departures = 0
    departure_records = []  # list of dicts: {actual_soc, required_soc, deficit}
    ev_tracker = {}

    # Solar & load tracking
    solar_gen_all = []
    non_shiftable_load_all = []

    # =====================================================================
    # ROLLOUT
    # =====================================================================
    out(f"\n  Running evaluation rollout (8759 steps)...")

    while not done:
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
        action = np.clip(action_t, -1.0, 1.0)
        actions_all.append(action.copy())

        t_now = int(getattr(raw, "time_step", 0))
        t_idx = max(0, t_now - 1)
        hour = t_now % 24
        hour_all.append(hour)

        # Price
        try:
            pr = buildings[0].pricing.electricity_pricing
            price = float(pr[t_idx]) if t_idx < len(pr) else 0.0
        except Exception:
            price = 0.0
        price_all.append(price)

        # Solar generation
        try:
            sg = getattr(buildings[0], 'solar_generation', None)
            if sg is not None and len(sg) > t_idx:
                solar_val = float(sg[t_idx])
            else:
                solar_val = 0.0
        except Exception:
            solar_val = 0.0
        solar_gen_all.append(solar_val)

        # Non-shiftable load
        try:
            nsl = buildings[0].energy_simulation.non_shiftable_load
            if nsl is not None and t_idx < len(nsl):
                nsl_val = float(nsl[t_idx])
            else:
                nsl_val = 0.0
        except Exception:
            nsl_val = 0.0
        non_shiftable_load_all.append(nsl_val)

        # Battery SoC (building 0)
        try:
            es = getattr(buildings[0], "electrical_storage", None)
            soc_arr = getattr(es, "soc", None) if es else None
            if soc_arr is not None and len(soc_arr) > t_idx:
                batt_soc = float(np.clip(soc_arr[t_idx], 0, 1))
            else:
                batt_soc = 0.5
        except Exception:
            batt_soc = 0.5
        batt_soc_all.append(batt_soc)

        # Battery actions by hour
        for bi in batt_idx:
            if bi < len(action):
                batt_actions_by_hour[hour].append(float(action[bi]))

        # EV tracking
        ev_connected_this_step = False
        ev_soc_this_step = 0.0
        bld = buildings[0]
        chargers = getattr(bld, "electric_vehicle_chargers", None) or []
        for ch_idx, ch in enumerate(chargers):
            key = (0, ch_idx)
            sim = getattr(ch, 'charger_simulation',
                          getattr(ch, '_Charger__charger_simulation', None))
            if sim is None:
                continue
            try:
                sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                current_connected = t_now < len(sa) and float(sa[t_now]) == 1.0

                if current_connected:
                    ev_connected_this_step = True
                    ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                    if ev_obj is not None:
                        bt = getattr(ev_obj, 'battery', None)
                        if bt is not None:
                            soc_arr_ev = getattr(bt, 'soc', None)
                            if soc_arr_ev is not None:
                                sn = np.asarray(soc_arr_ev, dtype=float)
                                ev_soc_this_step = float(np.clip(sn[t_idx], 0, 1)) if 0 <= t_idx < len(sn) else 0.0

                # Departure tracking
                ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                if current_connected:
                    rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                    if not np.isfinite(rs):
                        rs = 1.0
                    ev_tracker[key] = {'was_connected': True, 'last_soc': ev_soc_this_step, 'required_soc': rs}
                else:
                    prev = ev_tracker.get(key, {})
                    if prev.get('was_connected', False):
                        total_departures += 1
                        last_soc = prev['last_soc']
                        rs = prev['required_soc']
                        deficit = max(0.0, rs - last_soc)
                        violated = deficit > 0.01
                        if violated:
                            violated_departures += 1
                        departure_records.append({
                            'actual_soc': last_soc,
                            'required_soc': rs,
                            'deficit': deficit,
                            'violated': violated,
                        })
                    ev_tracker[key] = {'was_connected': False}
            except Exception:
                pass

        ev_connected_mask.append(ev_connected_this_step)
        ev_soc_all.append(ev_soc_this_step)

        # EV actions by hour (connected only)
        if ev_connected_this_step and ev_idx:
            ei = ev_idx[0]
            if ei < len(action):
                a_ev = float(action[ei])
                ev_actions_by_hour[hour].append(a_ev)
                ev_actions_when_connected.append((hour, a_ev))

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        rewards.append(reward)
        costs.append(info.get("cost", 0.0))
        c0_vals.append(info.get("cost_ev_departure", 0.0))
        c1_vals.append(info.get("cost_ev_dense", 0.0))
        c2_vals.append(info.get("cost_stems_battery", 0.0))
        c3_vals.append(info.get("cost_stems_building_power", 0.0))
        c4_vals.append(info.get("cost_stems_grid_power", 0.0))

        # NEC per building (after step)
        t_after = max(0, int(getattr(raw, "time_step", 0)) - 1)
        for b_idx_loop, bld_loop in enumerate(buildings):
            try:
                nec = getattr(bld_loop, 'net_electricity_consumption', None)
                if nec is not None and len(nec) > t_after:
                    p = float(nec[t_after])
                else:
                    p = 0.0
            except Exception:
                p = 0.0
            nec_per_building[b_idx_loop].append(p)

        total_nec = sum(nec_per_building[b][-1] for b in range(n_buildings))
        grid_nec_all.append(total_nec)

        if step % 2000 == 0:
            print(f"    Step {step}/8759...")

    out(f"  Rollout complete: {step} steps")

    # =====================================================================
    # COMPUTE ALL METRICS
    # =====================================================================
    actions_arr = np.array(actions_all)
    hours_arr = np.array(hour_all)

    # ----------- Battery Metrics -----------
    batt_actions = actions_arr[:, batt_idx[0]] if batt_idx else np.array([])
    batt_soc_arr = np.array(batt_soc_all)

    n_total = len(batt_actions)
    batt_charge_count = int(np.sum(batt_actions > 0.1))
    batt_discharge_count = int(np.sum(batt_actions < -0.1))
    batt_idle_count = int(np.sum(np.abs(batt_actions) <= 0.1))
    batt_mean_action = float(np.mean(batt_actions))

    solar_mask = np.isin(hours_arr, [10, 11, 12, 13, 14, 15])
    peak_mask = np.isin(hours_arr, [17, 18, 19, 20, 21])
    offpeak_mask = np.isin(hours_arr, list(range(0, 10)) + [22, 23])

    batt_mean_solar = float(np.mean(batt_actions[solar_mask])) if solar_mask.any() else 0.0
    batt_mean_peak = float(np.mean(batt_actions[peak_mask])) if peak_mask.any() else 0.0
    batt_mean_offpeak = float(np.mean(batt_actions[offpeak_mask])) if offpeak_mask.any() else 0.0

    # Price-action correlation: corr(action, -(price - mean_price))
    prices_np = np.array(price_all)
    price_signal = -(prices_np - np.mean(prices_np))
    valid_corr = np.isfinite(prices_np) & np.isfinite(batt_actions)
    if valid_corr.sum() > 10:
        price_corr = float(np.corrcoef(batt_actions[valid_corr], price_signal[valid_corr])[0, 1])
    else:
        price_corr = 0.0

    batt_soc_min = float(np.min(batt_soc_arr))
    batt_soc_max = float(np.max(batt_soc_arr))
    batt_soc_mean = float(np.mean(batt_soc_arr))
    batt_soc_std = float(np.std(batt_soc_arr))

    # SoC at start of each day (hour 0)
    hour0_indices = np.where(hours_arr == 0)[0]
    batt_soc_hour0 = batt_soc_arr[hour0_indices] if len(hour0_indices) > 0 else np.array([])
    batt_soc_hour0_mean = float(np.mean(batt_soc_hour0)) if len(batt_soc_hour0) > 0 else 0.0

    # Daily charge/discharge cycle count
    n_days = step // 24
    cycle_days = 0
    for d in range(n_days):
        day_slice = batt_actions[d * 24 : (d + 1) * 24]
        has_charge = np.any(day_slice > 0.1)
        has_discharge = np.any(day_slice < -0.1)
        if has_charge and has_discharge:
            cycle_days += 1

    # ----------- EV Metrics -----------
    ev_connected_arr = np.array(ev_connected_mask)
    ev_connected_total = int(np.sum(ev_connected_arr))
    ev_connected_pct = pct(ev_connected_total, step)

    ev_actions_conn = []
    ev_actions_conn_hours = []
    ev_charge_conn = 0
    ev_discharge_conn = 0
    ev_idle_conn = 0
    ev_v2g_peak_conn = 0
    ev_v2g_nonpeak_conn = 0
    ev_peak_conn_total = 0
    ev_nonpeak_conn_total = 0

    ev_soc_when_connected = []

    for t in range(step):
        if ev_connected_arr[t]:
            a_ev = float(actions_arr[t, ev_idx[0]]) if ev_idx else 0.0
            h = hours_arr[t]
            ev_actions_conn.append(a_ev)
            ev_actions_conn_hours.append(h)
            if ev_soc_all[t] > 0:
                ev_soc_when_connected.append(ev_soc_all[t])

            if a_ev > 0.1:
                ev_charge_conn += 1
            elif a_ev < -0.1:
                ev_discharge_conn += 1
            else:
                ev_idle_conn += 1

            if 17 <= h <= 23:
                ev_peak_conn_total += 1
                if a_ev < -0.1:
                    ev_v2g_peak_conn += 1
            else:
                ev_nonpeak_conn_total += 1
                if a_ev < -0.1:
                    ev_v2g_nonpeak_conn += 1

    ev_mean_action_conn = float(np.mean(ev_actions_conn)) if ev_actions_conn else 0.0
    ev_peak_actions = [a for h, a in zip(ev_actions_conn_hours, ev_actions_conn) if 17 <= h <= 23]
    ev_mean_action_peak = float(np.mean(ev_peak_actions)) if ev_peak_actions else 0.0

    ev_soc_conn_arr = np.array(ev_soc_when_connected) if ev_soc_when_connected else np.array([0.0])
    ev_soc_min = float(np.min(ev_soc_conn_arr))
    ev_soc_max = float(np.max(ev_soc_conn_arr))
    ev_soc_mean = float(np.mean(ev_soc_conn_arr))

    # ----------- EV Departure Metrics (C0) -----------
    dep_actual = [d['actual_soc'] for d in departure_records]
    dep_required = [d['required_soc'] for d in departure_records]
    dep_deficits = [d['deficit'] for d in departure_records]
    dep_violated = [d for d in departure_records if d['violated']]
    dep_satisfied = [d for d in departure_records if not d['violated']]

    dep_mean_actual = float(np.mean(dep_actual)) if dep_actual else 0.0
    dep_mean_required = float(np.mean(dep_required)) if dep_required else 0.0
    dep_violated_mean_actual = float(np.mean([d['actual_soc'] for d in dep_violated])) if dep_violated else 0.0
    dep_violated_mean_deficit = float(np.mean([d['deficit'] for d in dep_violated])) if dep_violated else 0.0
    dep_max_deficit = float(np.max(dep_deficits)) if dep_deficits else 0.0

    # Departures by SoC range
    dep_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    dep_by_range = {}
    for lo, hi in dep_ranges:
        cnt = sum(1 for s in dep_actual if lo <= s < hi + (0.001 if hi == 1.0 else 0.0))
        dep_by_range[f"[{lo:.1f}-{hi:.1f}]"] = cnt

    # ----------- Building Power Metrics (C3) -----------
    nec_b0 = np.array(nec_per_building[0])
    abs_nec_b0 = np.abs(nec_b0)
    c3_exceed_count = int(np.sum(abs_nec_b0 > P_BUILDING_MAX))
    c3_exceed_pct = pct(c3_exceed_count, len(nec_b0))
    mean_abs_nec = float(np.mean(abs_nec_b0))
    max_abs_nec = float(np.max(abs_nec_b0))
    p95_abs_nec = float(np.percentile(abs_nec_b0, 95))
    mean_nec_signed = float(np.mean(nec_b0))
    total_import = float(np.sum(np.maximum(0, nec_b0)))
    total_export = float(np.sum(np.maximum(0, -nec_b0)))

    # ----------- Grid Power Metrics (C4) -----------
    grid_nec_arr = np.array(grid_nec_all)
    abs_grid_nec = np.abs(grid_nec_arr)
    P_GRID_MAX = P_BUILDING_MAX  # 1 building
    c4_exceed_count = int(np.sum(abs_grid_nec > P_GRID_MAX))
    c4_exceed_pct = pct(c4_exceed_count, len(grid_nec_arr))
    mean_abs_grid = float(np.mean(abs_grid_nec))
    max_abs_grid = float(np.max(abs_grid_nec))

    # ----------- Battery SoC Metrics (C2) -----------
    c2_below_0 = int(np.sum(batt_soc_arr < 0.0))
    c2_above_095 = int(np.sum(batt_soc_arr > 0.95))
    c2_outside = c2_below_0 + c2_above_095
    c2_outside_pct = pct(c2_outside, len(batt_soc_arr))

    # ----------- CityLearn KPIs -----------
    # Ramping: sum of |NEC_t - NEC_{t-1}|
    ramping = float(np.sum(np.abs(np.diff(nec_b0))))

    # Load factor: mean(NEC) / max(NEC) -- using positive NEC only (import)
    nec_pos = np.maximum(0, nec_b0)
    max_nec_pos = float(np.max(nec_pos)) if np.max(nec_pos) > 0 else 1.0
    load_factor = float(np.mean(nec_pos)) / max_nec_pos

    # Electricity cost (approximate: sum of NEC * price)
    # Only charge import from grid (positive NEC)
    elec_cost = float(np.sum(nec_pos * prices_np[:len(nec_pos)]))
    # With export credit
    export_factor = 0.7
    export_credit = float(np.sum(np.maximum(0, -nec_b0) * prices_np[:len(nec_b0)] * export_factor))
    net_elec_cost = elec_cost - export_credit

    # Peak demand
    peak_demand = float(np.max(nec_pos))

    # ----------- Energy Balance -----------
    solar_arr = np.array(solar_gen_all)
    total_solar = float(np.sum(np.abs(solar_arr)))  # solar_generation is typically negative
    nsl_arr = np.array(non_shiftable_load_all)
    total_nsl = float(np.sum(nsl_arr))

    # Battery energy throughput (approximate from actions)
    # NOTE: These are normalized actions, not kWh directly.
    # We report the action-space sums.
    batt_charge_energy = float(np.sum(np.maximum(0, batt_actions)))
    batt_discharge_energy = float(np.sum(np.minimum(0, batt_actions)))

    ev_actions_full = actions_arr[:, ev_idx[0]] if ev_idx else np.zeros(step)
    ev_charge_energy = float(np.sum(np.maximum(0, ev_actions_full)))
    ev_discharge_energy = float(np.sum(np.minimum(0, ev_actions_full)))

    net_grid = total_import - total_export

    # =====================================================================
    # PRINT EVERYTHING
    # =====================================================================
    out(f"\n{'='*80}")
    out(f"  PPO v4 1-BUILDING COMPREHENSIVE RESULTS (epoch-{epoch})")
    out(f"{'='*80}")

    out(f"\n  --- Episode Summary ---")
    out(f"  {'Total steps':<45} {step}")
    out(f"  {'Total reward':<45} {sum(rewards):.6f}")
    out(f"  {'Mean reward':<45} {np.mean(rewards):.6f}")
    out(f"  {'Min reward':<45} {np.min(rewards):.6f}")
    out(f"  {'Max reward':<45} {np.max(rewards):.6f}")
    out(f"  {'Total cost (aggregate)':<45} {sum(costs):.6f}")

    out(f"\n  --- Per-Constraint Cumulative Costs ---")
    out(f"  {'C0 (EV departure)':<45} {sum(c0_vals):.6f}")
    out(f"  {'C1 (EV dense)':<45} {sum(c1_vals):.6f}")
    out(f"  {'C2 (battery SoC)':<45} {sum(c2_vals):.6f}")
    out(f"  {'C3 (building power)':<45} {sum(c3_vals):.6f}")
    out(f"  {'C4 (grid power)':<45} {sum(c4_vals):.6f}")

    out(f"\n{'='*80}")
    out(f"  BATTERY METRICS (Building 0)")
    out(f"{'='*80}")
    out(f"  {'Total steps':<45} {n_total}")
    out(f"  {'Charge steps (action > 0.1)':<45} {batt_charge_count} ({pct(batt_charge_count, n_total):.2f}%)")
    out(f"  {'Discharge steps (action < -0.1)':<45} {batt_discharge_count} ({pct(batt_discharge_count, n_total):.2f}%)")
    out(f"  {'Idle steps (|action| <= 0.1)':<45} {batt_idle_count} ({pct(batt_idle_count, n_total):.2f}%)")
    out(f"  {'Mean action (all hours)':<45} {batt_mean_action:+.6f}")
    out(f"  {'Mean action during solar (10-15)':<45} {batt_mean_solar:+.6f}")
    out(f"  {'Mean action during peak (17-21)':<45} {batt_mean_peak:+.6f}")
    out(f"  {'Mean action during off-peak (22-9)':<45} {batt_mean_offpeak:+.6f}")
    out(f"  {'Price-action corr(a, -(p-mean_p))':<45} {price_corr:+.6f}")
    out(f"  {'Battery SoC min':<45} {batt_soc_min:.6f}")
    out(f"  {'Battery SoC max':<45} {batt_soc_max:.6f}")
    out(f"  {'Battery SoC mean':<45} {batt_soc_mean:.6f}")
    out(f"  {'Battery SoC std':<45} {batt_soc_std:.6f}")
    out(f"  {'Battery SoC at hour 0 (mean)':<45} {batt_soc_hour0_mean:.6f}")
    out(f"  {'Daily charge/discharge cycles':<45} {cycle_days}/{n_days} days ({pct(cycle_days, n_days):.2f}%)")

    out(f"\n{'='*80}")
    out(f"  EV METRICS (Charger on Building 0)")
    out(f"{'='*80}")
    out(f"  {'Total connected steps':<45} {ev_connected_total}")
    out(f"  {'Connected %':<45} {ev_connected_pct:.2f}%")
    out(f"  {'Charge when connected (action > 0.1)':<45} {ev_charge_conn} ({pct(ev_charge_conn, ev_connected_total):.2f}%)")
    out(f"  {'V2G discharge when connected (< -0.1)':<45} {ev_discharge_conn} ({pct(ev_discharge_conn, ev_connected_total):.2f}%)")
    out(f"  {'Idle when connected (|action| <= 0.1)':<45} {ev_idle_conn} ({pct(ev_idle_conn, ev_connected_total):.2f}%)")
    out(f"  {'V2G during peak (17-23) when connected':<45} {ev_v2g_peak_conn} ({pct(ev_v2g_peak_conn, ev_peak_conn_total):.2f}%)")
    out(f"  {'V2G during non-peak when connected':<45} {ev_v2g_nonpeak_conn} ({pct(ev_v2g_nonpeak_conn, ev_nonpeak_conn_total):.2f}%)")
    out(f"  {'Mean EV action when connected':<45} {ev_mean_action_conn:+.6f}")
    out(f"  {'Mean EV action during peak (connected)':<45} {ev_mean_action_peak:+.6f}")
    out(f"  {'EV SoC min (when connected)':<45} {ev_soc_min:.6f}")
    out(f"  {'EV SoC max (when connected)':<45} {ev_soc_max:.6f}")
    out(f"  {'EV SoC mean (when connected)':<45} {ev_soc_mean:.6f}")

    out(f"\n{'='*80}")
    out(f"  EV DEPARTURE METRICS (C0)")
    out(f"{'='*80}")
    out(f"  {'Total departures':<45} {total_departures}")
    out(f"  {'Departures with SoC >= required':<45} {len(dep_satisfied)} ({pct(len(dep_satisfied), total_departures):.2f}%)")
    out(f"  {'Departures with SoC < required (VIOLATED)':<45} {len(dep_violated)} ({pct(len(dep_violated), total_departures):.2f}%)")
    out(f"  {'C0 violation rate':<45} {pct(len(dep_violated), total_departures):.4f}%")
    out(f"  {'Mean SoC at departure (all)':<45} {dep_mean_actual:.6f}")
    out(f"  {'Mean SoC at departure (violated only)':<45} {dep_violated_mean_actual:.6f}")
    out(f"  {'Mean required SoC at departure':<45} {dep_mean_required:.6f}")
    out(f"  {'Mean deficit at violated departures':<45} {dep_violated_mean_deficit:.6f}")
    out(f"  {'Max deficit at any departure':<45} {dep_max_deficit:.6f}")
    out(f"  Departures by SoC range:")
    for rng, cnt in dep_by_range.items():
        out(f"    {rng:<42} {cnt}")

    out(f"\n{'='*80}")
    out(f"  BUILDING POWER METRICS (C3)")
    out(f"{'='*80}")
    out(f"  {'P_building_max':<45} {P_BUILDING_MAX:.4f} kW")
    out(f"  {'Total building-hours':<45} {len(nec_b0)}")
    out(f"  {'Hours where |NEC| > P_building_max':<45} {c3_exceed_count} ({c3_exceed_pct:.2f}%)")
    out(f"  {'Mean |NEC|':<45} {mean_abs_nec:.6f} kW")
    out(f"  {'Max |NEC|':<45} {max_abs_nec:.6f} kW")
    out(f"  {'P95 |NEC|':<45} {p95_abs_nec:.6f} kW")
    out(f"  {'Mean NEC (signed)':<45} {mean_nec_signed:+.6f} kW")
    out(f"  {'Total import (kWh)':<45} {total_import:.6f}")
    out(f"  {'Total export (kWh)':<45} {total_export:.6f}")

    out(f"\n{'='*80}")
    out(f"  GRID POWER METRICS (C4)")
    out(f"{'='*80}")
    out(f"  {'P_grid_max (1 building = P_building_max)':<45} {P_GRID_MAX:.4f} kW")
    out(f"  {'Hours where |total_NEC| > P_grid_max':<45} {c4_exceed_count} ({c4_exceed_pct:.2f}%)")
    out(f"  {'Mean |total_NEC|':<45} {mean_abs_grid:.6f} kW")
    out(f"  {'Max |total_NEC|':<45} {max_abs_grid:.6f} kW")

    out(f"\n{'='*80}")
    out(f"  BATTERY SOC CONSTRAINT METRICS (C2)")
    out(f"{'='*80}")
    out(f"  {'Steps where SoC < 0.0':<45} {c2_below_0}")
    out(f"  {'Steps where SoC > 0.95':<45} {c2_above_095}")
    out(f"  {'Steps outside [0.0, 0.95]':<45} {c2_outside} ({c2_outside_pct:.2f}%)")

    out(f"\n{'='*80}")
    out(f"  CITYLEARN KPIs")
    out(f"{'='*80}")
    out(f"  {'Electricity cost (import only, $)':<45} {elec_cost:.6f}")
    out(f"  {'Export credit ($, factor={export_factor})':<45} {export_credit:.6f}")
    out(f"  {'Net electricity cost ($)':<45} {net_elec_cost:.6f}")
    out(f"  {'Peak demand (kW)':<45} {peak_demand:.6f}")
    out(f"  {'Ramping (sum |NEC_t - NEC_t-1|)':<45} {ramping:.6f}")
    out(f"  {'Load factor (mean import / max import)':<45} {load_factor:.6f}")

    out(f"\n{'='*80}")
    out(f"  ENERGY BALANCE")
    out(f"{'='*80}")
    out(f"  {'Total solar generation (kWh)':<45} {total_solar:.6f}")
    out(f"  {'Total non-shiftable load (kWh)':<45} {total_nsl:.6f}")
    out(f"  {'Total battery charge (action-sum, >0)':<45} {batt_charge_energy:.6f}")
    out(f"  {'Total battery discharge (action-sum, <0)':<45} {batt_discharge_energy:.6f}")
    out(f"  {'Total EV charge (action-sum, >0)':<45} {ev_charge_energy:.6f}")
    out(f"  {'Total EV discharge/V2G (action-sum, <0)':<45} {ev_discharge_energy:.6f}")
    out(f"  {'Net grid consumption (import-export kWh)':<45} {net_grid:.6f}")

    out(f"\n{'='*80}")
    out(f"  EVALUATION COMPLETE")
    out(f"{'='*80}")

    # Save output
    flush_output(OUT_TXT)
    out(f"\n  Results saved to: {OUT_TXT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    evaluate(epoch=args.epoch, seed=args.seed)
