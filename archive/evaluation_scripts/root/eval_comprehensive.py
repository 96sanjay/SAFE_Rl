#!/usr/bin/env python3
"""
Comprehensive Evaluation: R30c (5-building) and 1-Building Test
================================================================
Evaluates both checkpoints on full-year schemas, collects every possible metric,
prints tables, and saves a 6-panel visualization.

Usage:
    python eval_comprehensive.py [--r30c_epoch 50] [--onebld_epoch 50]
"""
import os
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
# Constants
# =====================================================================
P_BUILDING_MAX = 4.6083
P_GRID_MAX = 10.2352
OUT_PNG = f"{PROJECT}/runs/eval_comprehensive.png"

R30C_CKPT_DIR = (
    f"{PROJECT}/runs/r30c_kl_nec/5bld/"
    "PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-19-18-08-21/torch_save"
)

ONEBLD_CKPT_DIR = (
    f"{PROJECT}/runs/test_1bld/"
    "PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-19-19-23-19/torch_save"
)

# =====================================================================
# ENV VARS for each run
# =====================================================================
# Common R30c env vars (from run_r30_kl_nec.sh)
R30C_ENV_VARS = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_PNORM_P": "4.0",
    "CITYLEARN_ACTION_MASK": "1",
    "CITYLEARN_KL_BETA": "0.1",
    "CITYLEARN_KL_BETA_DECAY": "0.995",
    "STEMS_ALPHA_NEC_SIGN": "3.0",
    "STEMS_ALPHA_PRICE_ARB": "1.0",
    "STEMS_LAMBDA_EV": "4.0",
    "STEMS_EV_SLACK_ARB_SCALE": "2.5",
    "STEMS_ALPHA_GRID_PENALTY": "1.5",
    "CITYLEARN_STEMS_BATTERY_COST_SCALE": "1.0",
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_GRID": "0.0",
    "STEMS_ALPHA_BUILD": "0.0",
    "STEMS_BETA_RAMP": "0.0",
    "STEMS_XI_RENEWABLE": "0.0",
    "STEMS_ALPHA_LOAD_SHIFT": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.0",
    "STEMS_ALPHA_EV_GUARD": "0.0",
    "STEMS_ALPHA_V2G_CONTEXT": "0.0",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "STEMS_ALPHA_BARRIER": "0.0",
    "STEMS_ALPHA_EV_SOLAR": "0.0",
    "STEMS_ALPHA_SOLAR_STORE": "0.0",
    "STEMS_SOLAR_STORE_BATT_ONLY": "0",
    "STEMS_ALPHA_HEADROOM": "0.0",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_SG_THRESHOLD": "0.5",
    "CITYLEARN_EV_SAUTE": "1",
    "CITYLEARN_EV_SAUTE_BUDGET": "25000",
    "CITYLEARN_EV_SAUTE_PENALTY": "5.0",
    "CITYLEARN_EV_SAUTE_GAMMA": "1.0",
    "CITYLEARN_EV_SAUTE_SHAPED_ALPHA": "2.0",
    "CITYLEARN_PID_LAGRANGE": "1",
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.05",
    "CITYLEARN_EV_COST_SCALE": "3.0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_BATT_CLAMP": "0",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_C3_CONTROLLABLE": "1",
}

# 1-building env vars (from run_1bld_test.sh / eval_1bld.py)
ONEBLD_ENV_VARS = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_1building.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_ACTION_MASK": "1",
    "CITYLEARN_KL_BETA": "0.1",
    "CITYLEARN_KL_BETA_DECAY": "0.995",
    "STEMS_ALPHA_NEC_SIGN": "3.0",
    "STEMS_ALPHA_PRICE_ARB": "1.0",
    "STEMS_LAMBDA_EV": "4.0",
    "STEMS_EV_SLACK_ARB_SCALE": "2.5",
    "STEMS_ALPHA_GRID_PENALTY": "1.5",
    "STEMS_ALPHA_GRID_PENALTY_SOLAR": "0.0",
    "CITYLEARN_STEMS_BATTERY_COST_SCALE": "1.0",
    "STEMS_ALPHA_NEC": "0.0",
    "STEMS_ALPHA_PEAK": "0.0",
    "STEMS_ALPHA_RAMP": "0.0",
    "STEMS_ALPHA_CARBON": "0.0",
    "STEMS_ALPHA_LOAD": "0.0",
    "CITYLEARN_C1_SOC_TOLERANCE": "0.20",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_EV_SAUTE": "0",
}


def clear_env():
    """Remove all CITYLEARN/STEMS/COST_W env vars."""
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(k, None)


def set_env(env_vars):
    clear_env()
    for k, v in env_vars.items():
        os.environ[k] = v


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


def pct(a, b):
    return 100.0 * a / b if b > 0 else 0.0


def find_latest_epoch(ckpt_dir):
    """Find the highest epoch number in a checkpoint directory."""
    epochs = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("epoch-") and f.endswith(".pt"):
            try:
                e = int(f.replace("epoch-", "").replace(".pt", ""))
                epochs.append(e)
            except ValueError:
                pass
    if not epochs:
        raise FileNotFoundError(f"No epoch checkpoints found in {ckpt_dir}")
    return max(epochs)


# =====================================================================
# Core evaluation function
# =====================================================================
def evaluate_run(run_name, ckpt_dir, epoch, env_vars, use_saute, seed=42):
    """Run a full-year evaluation and return a dict of all metrics."""
    ckpt_path = os.path.join(ckpt_dir, f"epoch-{epoch}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\n{'='*72}")
    print(f"  EVALUATING: {run_name}")
    print(f"  Checkpoint: epoch-{epoch}.pt")
    print(f"  Seed: {seed}")
    print(f"{'='*72}")

    # Load model
    actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(ckpt_path)
    print(f"  Actor: obs_dim={obs_dim}, act_dim={act_dim}, hidden=[256,256]")
    print(f"  Obs normalizer: {'YES' if obs_mean is not None else 'NO'}")

    # Set environment
    set_env(env_vars)

    # We need fresh imports each time since env vars changed
    # Use importlib to force re-evaluation if needed
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.action_mask_wrapper import ActionMaskWrapper
    from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    forecast = ForecastObsWrapper(safety, forecast_horizon=24)

    if use_saute:
        from citylearn_safe.saute_ev_wrapper import SauteEVBudgetWrapper
        saute = SauteEVBudgetWrapper(forecast)
        env = ActionMaskWrapper(saute)
    else:
        env = ActionMaskWrapper(forecast)

    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(getattr(raw, "buildings", []))
    n_buildings = len(buildings)

    # Identify action indices
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
        if i in batt_idx:
            tag = "  [BATTERY]"
        elif i in ev_idx:
            tag = "  [EV]"
        print(f"    [{i}] {n}{tag}")

    # Handle obs dim mismatch
    env_obs_dim = env.observation_space.shape[0]
    need_pad = obs_dim > env_obs_dim
    pad_dim = obs_dim - env_obs_dim if need_pad else 0
    need_trim = obs_dim < env_obs_dim
    if need_pad:
        print(f"  WARNING: Model expects {obs_dim} obs, env gives {env_obs_dim}. Padding {pad_dim} dims.")
    if need_trim:
        print(f"  WARNING: Model expects {obs_dim} obs, env gives {env_obs_dim}. Trimming.")

    # Reset
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    # =====================================================================
    # TRACKING VARIABLES
    # =====================================================================
    rewards = []
    costs = []
    actions_all = []

    c0_vals, c1_vals, c2_vals, c3_vals, c4_vals = [], [], [], [], []

    batt_actions_by_hour = [[] for _ in range(24)]
    batt_soc_all = []

    ev_actions_by_hour = [[] for _ in range(24)]
    ev_connected_mask = []
    ev_soc_all = []

    nec_per_building = [[] for _ in range(n_buildings)]
    grid_nec_all = []
    price_all = []
    hour_all = []

    c3_violations_per_building = [0] * n_buildings
    c3_total = 0
    c4_total = 0

    # Battery SoC violations (C2)
    c2_violation_count = 0

    total_departures = 0
    violated_departures = 0
    departure_deficits = []
    departure_socs = []
    ev_tracker = {}

    # EV per-charger tracking
    ev_per_charger_connected = defaultdict(int)
    ev_per_charger_charge = defaultdict(int)
    ev_per_charger_discharge = defaultdict(int)
    ev_per_charger_idle = defaultdict(int)
    ev_per_charger_peak_connected = defaultdict(int)
    ev_per_charger_peak_v2g = defaultdict(int)
    ev_per_charger_nonpeak_v2g = defaultdict(int)
    ev_per_charger_peak_v2g_magnitudes = defaultdict(list)
    ev_per_charger_departures = defaultdict(int)
    ev_per_charger_violated = defaultdict(int)
    ev_per_charger_deficits = defaultdict(list)
    ev_per_charger_departure_socs = defaultdict(list)

    # Per-building battery tracking
    batt_per_building_actions = defaultdict(list)

    # =====================================================================
    # ROLLOUT
    # =====================================================================
    print(f"\n  Running evaluation rollout (~8759 steps)...")

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

        # Battery SoC
        batt_soc_step = []
        for b_idx, bld in enumerate(buildings):
            try:
                es = getattr(bld, "electrical_storage", None)
                soc_arr = getattr(es, "soc", None) if es else None
                if soc_arr is not None and len(soc_arr) > t_idx:
                    s = float(np.clip(soc_arr[t_idx], 0, 1))
                else:
                    s = 0.5
            except Exception:
                s = 0.5
            batt_soc_step.append(s)
            # C2: battery SoC violation
            if s < 0.0 or s > 0.95:
                c2_violation_count += 1
        batt_soc_all.append(batt_soc_step)

        # Battery actions by hour and per building
        for bi_local, bi in enumerate(batt_idx):
            if bi < len(action):
                batt_actions_by_hour[hour].append(float(action[bi]))
                batt_per_building_actions[bi_local].append(float(action[bi]))

        # EV tracking
        ev_soc_step = []
        ev_connected_step = []
        charger_global_idx = 0
        for b_idx, bld in enumerate(buildings):
            chargers = getattr(bld, "electric_vehicle_chargers", None) or []
            for ch_idx, ch in enumerate(chargers):
                key = (b_idx, ch_idx)
                sim = getattr(ch, 'charger_simulation',
                              getattr(ch, '_Charger__charger_simulation', None))
                if sim is None:
                    ev_connected_step.append(False)
                    ev_soc_step.append(0.0)
                    charger_global_idx += 1
                    continue
                try:
                    sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                    current_connected = t_now < len(sa) and float(sa[t_now]) == 1.0
                    ev_connected_step.append(current_connected)

                    ev_soc = 0.0
                    if current_connected:
                        ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                        if ev_obj is not None:
                            bt = getattr(ev_obj, 'battery', None)
                            if bt is not None:
                                soc_arr = getattr(bt, 'soc', None)
                                if soc_arr is not None:
                                    sn = np.asarray(soc_arr, dtype=float)
                                    ev_soc = float(np.clip(sn[t_idx], 0, 1)) if 0 <= t_idx < len(sn) else 0.0
                    ev_soc_step.append(ev_soc)

                    # Per-charger stats
                    if current_connected:
                        ev_per_charger_connected[key] += 1
                        # Find the action for this charger
                        if charger_global_idx < len(ev_idx):
                            ei = ev_idx[charger_global_idx]
                            if ei < len(action):
                                a = float(action[ei])
                                if a > 0.1:
                                    ev_per_charger_charge[key] += 1
                                elif a < -0.1:
                                    ev_per_charger_discharge[key] += 1
                                else:
                                    ev_per_charger_idle[key] += 1

                                if 17 <= hour <= 23:
                                    ev_per_charger_peak_connected[key] += 1
                                    if a < -0.1:
                                        ev_per_charger_peak_v2g[key] += 1
                                        ev_per_charger_peak_v2g_magnitudes[key].append(abs(a))
                                else:
                                    if a < -0.1:
                                        ev_per_charger_nonpeak_v2g[key] += 1

                    # Departure tracking
                    ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                    if current_connected:
                        rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                        if not np.isfinite(rs):
                            rs = 1.0
                        ev_tracker[key] = {'was_connected': True, 'last_soc': ev_soc, 'required_soc': rs}
                    else:
                        prev = ev_tracker.get(key, {})
                        if prev.get('was_connected', False):
                            total_departures += 1
                            ev_per_charger_departures[key] += 1
                            last_soc = prev['last_soc']
                            rs = prev['required_soc']
                            deficit = max(0.0, rs - last_soc)
                            departure_socs.append(last_soc)
                            ev_per_charger_departure_socs[key].append(last_soc)
                            if deficit > 0.01:
                                violated_departures += 1
                                ev_per_charger_violated[key] += 1
                            departure_deficits.append(deficit)
                            ev_per_charger_deficits[key].append(deficit)
                        ev_tracker[key] = {'was_connected': False}
                except Exception:
                    ev_connected_step.append(False)
                    ev_soc_step.append(0.0)
                charger_global_idx += 1

        ev_connected_mask.append(ev_connected_step)
        ev_soc_all.append(ev_soc_step)

        # EV actions by hour (connected only)
        ev_charger_i = 0
        for ei in ev_idx:
            if ei < len(action):
                if ev_charger_i < len(ev_connected_step) and ev_connected_step[ev_charger_i]:
                    ev_actions_by_hour[hour].append(float(action[ei]))
                ev_charger_i += 1

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
        for b_idx, bld in enumerate(buildings):
            try:
                nec = getattr(bld, 'net_electricity_consumption', None)
                if nec is not None and len(nec) > t_after:
                    p = float(nec[t_after])
                else:
                    p = 0.0
            except Exception:
                p = 0.0
            nec_per_building[b_idx].append(p)
            if abs(p) > P_BUILDING_MAX:
                c3_violations_per_building[b_idx] += 1
                c3_total += 1

        total_nec = sum(nec_per_building[b][-1] for b in range(n_buildings))
        grid_nec_all.append(total_nec)
        if abs(total_nec) > P_GRID_MAX:
            c4_total += 1

        if step % 2000 == 0:
            print(f"    Step {step}/~8759...")

    print(f"  Rollout complete: {step} steps")

    # =====================================================================
    # COMPUTE ALL METRICS
    # =====================================================================
    actions_arr = np.array(actions_all)
    M = {}  # metrics dict
    M["run_name"] = run_name
    M["n_buildings"] = n_buildings
    M["steps"] = step
    M["total_reward"] = float(sum(rewards))
    M["mean_reward"] = float(np.mean(rewards))
    M["total_cost"] = float(sum(costs))

    # Per-constraint cumulative costs
    M["c0_cumulative"] = float(sum(c0_vals))
    M["c1_cumulative"] = float(sum(c1_vals))
    M["c2_cumulative"] = float(sum(c2_vals))
    M["c3_cumulative"] = float(sum(c3_vals))
    M["c4_cumulative"] = float(sum(c4_vals))

    # ----- BATTERY METRICS -----
    batt_actions = actions_arr[:, batt_idx] if batt_idx else np.zeros((step, 0))
    batt_flat = batt_actions.ravel()
    n_batt_total = len(batt_flat) if batt_flat.size > 0 else 1

    M["batt_charge_pct"] = pct(np.sum(batt_flat > 0.1), n_batt_total)
    M["batt_discharge_pct"] = pct(np.sum(batt_flat < -0.1), n_batt_total)
    M["batt_idle_pct"] = pct(np.sum(np.abs(batt_flat) <= 0.1), n_batt_total)
    M["batt_action_mean"] = float(batt_flat.mean()) if batt_flat.size > 0 else 0.0
    M["batt_action_std"] = float(batt_flat.std()) if batt_flat.size > 0 else 0.0
    M["batt_action_min"] = float(batt_flat.min()) if batt_flat.size > 0 else 0.0
    M["batt_action_max"] = float(batt_flat.max()) if batt_flat.size > 0 else 0.0

    solar_hours = set(range(10, 16))
    peak_hours = set(range(17, 22))
    offpeak_hours = set(range(22, 24)) | set(range(0, 10))
    solar_batt = [a for h, acts in enumerate(batt_actions_by_hour) if h in solar_hours for a in acts]
    peak_batt = [a for h, acts in enumerate(batt_actions_by_hour) if h in peak_hours for a in acts]
    offpeak_batt = [a for h, acts in enumerate(batt_actions_by_hour) if h in offpeak_hours for a in acts]

    M["batt_mean_solar"] = float(np.mean(solar_batt)) if solar_batt else 0.0
    M["batt_mean_peak"] = float(np.mean(peak_batt)) if peak_batt else 0.0
    M["batt_mean_offpeak"] = float(np.mean(offpeak_batt)) if offpeak_batt else 0.0

    # Price-action correlation
    if batt_flat.size > 0 and len(price_all) == step:
        # Expand prices to match per-building actions
        prices_expanded = np.repeat(price_all, len(batt_idx))
        price_centered = prices_expanded - np.mean(prices_expanded)
        neg_price = -price_centered
        valid = np.isfinite(batt_flat) & np.isfinite(neg_price)
        if valid.sum() > 10:
            M["batt_price_corr"] = float(np.corrcoef(batt_flat[valid], neg_price[valid])[0, 1])
        else:
            M["batt_price_corr"] = 0.0
    else:
        M["batt_price_corr"] = 0.0

    # Battery SoC
    batt_soc_arr = np.array(batt_soc_all)
    M["batt_soc_mean"] = float(batt_soc_arr.mean())
    M["batt_soc_min"] = float(batt_soc_arr.min())
    M["batt_soc_max"] = float(batt_soc_arr.max())
    M["batt_soc_per_building"] = []
    for b in range(n_buildings):
        M["batt_soc_per_building"].append({
            "mean": float(batt_soc_arr[:, b].mean()),
            "min": float(batt_soc_arr[:, b].min()),
            "max": float(batt_soc_arr[:, b].max()),
        })

    # Cycling ratio
    ch_pct = M["batt_charge_pct"]
    dch_pct = M["batt_discharge_pct"]
    M["cycling_ratio"] = min(ch_pct, dch_pct) / max(ch_pct, dch_pct) if max(ch_pct, dch_pct) > 0 else 0.0
    M["cycling_emerged"] = ch_pct > 20.0 and dch_pct > 20.0

    # Per-building battery stats
    M["batt_per_building"] = []
    for bi_local in range(len(batt_idx)):
        acts = np.array(batt_per_building_actions.get(bi_local, []))
        if acts.size > 0:
            M["batt_per_building"].append({
                "charge_pct": pct(np.sum(acts > 0.1), len(acts)),
                "discharge_pct": pct(np.sum(acts < -0.1), len(acts)),
                "idle_pct": pct(np.sum(np.abs(acts) <= 0.1), len(acts)),
                "mean": float(acts.mean()),
                "std": float(acts.std()),
            })
        else:
            M["batt_per_building"].append({"charge_pct": 0, "discharge_pct": 0, "idle_pct": 0, "mean": 0, "std": 0})

    # ----- EV METRICS -----
    ev_connected_total = 0
    ev_charge_connected = 0
    ev_discharge_connected = 0
    ev_idle_connected = 0
    ev_peak_connected = 0
    ev_peak_discharge = 0
    ev_nonpeak_discharge = 0
    ev_peak_v2g_magnitudes_all = []

    for t in range(len(actions_all)):
        h = hour_all[t]
        for ei_local, ei in enumerate(ev_idx):
            if ei_local < len(ev_connected_mask[t]) and ev_connected_mask[t][ei_local]:
                a = float(actions_all[t][ei])
                ev_connected_total += 1
                if a > 0.1:
                    ev_charge_connected += 1
                elif a < -0.1:
                    ev_discharge_connected += 1
                else:
                    ev_idle_connected += 1
                if 17 <= h <= 23:
                    ev_peak_connected += 1
                    if a < -0.1:
                        ev_peak_discharge += 1
                        ev_peak_v2g_magnitudes_all.append(abs(a))
                else:
                    if a < -0.1:
                        ev_nonpeak_discharge += 1

    M["ev_connected_steps"] = ev_connected_total
    M["ev_charge_pct"] = pct(ev_charge_connected, ev_connected_total)
    M["ev_discharge_pct"] = pct(ev_discharge_connected, ev_connected_total)
    M["ev_idle_pct"] = pct(ev_idle_connected, ev_connected_total)
    M["ev_v2g_peak_pct"] = pct(ev_peak_discharge, ev_peak_connected)
    M["ev_v2g_nonpeak_pct"] = pct(ev_nonpeak_discharge, ev_connected_total - ev_peak_connected) if ev_connected_total > ev_peak_connected else 0.0
    M["ev_v2g_peak_mean_magnitude"] = float(np.mean(ev_peak_v2g_magnitudes_all)) if ev_peak_v2g_magnitudes_all else 0.0

    # EV departures
    M["total_departures"] = total_departures
    M["compliant_departures"] = total_departures - violated_departures
    M["violated_departures"] = violated_departures
    M["c0_violation_rate"] = pct(violated_departures, total_departures)
    M["mean_deficit_violated"] = float(np.mean([d for d in departure_deficits if d > 0.01])) if any(d > 0.01 for d in departure_deficits) else 0.0
    M["mean_deficit_all"] = float(np.mean(departure_deficits)) if departure_deficits else 0.0
    M["mean_soc_at_departure"] = float(np.mean(departure_socs)) if departure_socs else 0.0

    # ----- CONSTRAINT METRICS -----
    total_building_hours = step * n_buildings
    M["c0_violations"] = violated_departures
    M["c0_violation_pct"] = M["c0_violation_rate"]
    M["c0_mean_deficit"] = M["mean_deficit_all"]

    M["c2_violations"] = c2_violation_count
    M["c2_violation_pct"] = pct(c2_violation_count, total_building_hours)

    M["c3_violations"] = c3_total
    M["c3_violation_pct"] = pct(c3_total, total_building_hours)
    M["c3_per_building"] = []
    for b in range(n_buildings):
        M["c3_per_building"].append({
            "violations": c3_violations_per_building[b],
            "violation_pct": pct(c3_violations_per_building[b], step),
            "mean_abs_nec": float(np.mean(np.abs(nec_per_building[b]))),
            "max_abs_nec": float(np.max(np.abs(nec_per_building[b]))),
        })

    M["c4_violations"] = c4_total
    M["c4_violation_pct"] = pct(c4_total, step)

    # ----- KPI METRICS -----
    total_import = sum(max(0, nec) for necs in nec_per_building for nec in necs)
    total_export = sum(abs(min(0, nec)) for necs in nec_per_building for nec in necs)
    total_elec_cost = sum(p * max(0, grid_nec_all[i]) for i, p in enumerate(price_all))

    M["total_import_kwh"] = float(total_import)
    M["total_export_kwh"] = float(total_export)
    M["total_elec_cost"] = float(total_elec_cost)

    # Building power
    all_bld_nec = [abs(p) for necs in nec_per_building for p in necs]
    M["mean_building_power_kw"] = float(np.mean(all_bld_nec))
    M["max_building_power_kw"] = float(np.max(all_bld_nec))

    # Grid power
    M["mean_grid_power_kw"] = float(np.mean(np.abs(grid_nec_all)))
    M["max_grid_power_kw"] = float(np.max(np.abs(grid_nec_all)))

    # Ramping
    M["total_ramping"] = float(sum(abs(grid_nec_all[i] - grid_nec_all[i-1]) for i in range(1, step)))

    # ----- HOURLY PATTERNS -----
    M["batt_hourly_mean"] = [float(np.mean(batt_actions_by_hour[h])) if batt_actions_by_hour[h] else 0.0 for h in range(24)]
    M["batt_hourly_std"] = [float(np.std(batt_actions_by_hour[h])) if batt_actions_by_hour[h] else 0.0 for h in range(24)]
    M["ev_hourly_mean"] = [float(np.mean(ev_actions_by_hour[h])) if ev_actions_by_hour[h] else 0.0 for h in range(24)]
    M["ev_hourly_std"] = [float(np.std(ev_actions_by_hour[h])) if ev_actions_by_hour[h] else 0.0 for h in range(24)]
    M["price_hourly_mean"] = []
    for h in range(24):
        hp = [price_all[t] for t in range(step) if hour_all[t] == h]
        M["price_hourly_mean"].append(float(np.mean(hp)) if hp else 0.0)

    # Raw data for plotting
    M["_actions_arr"] = actions_arr
    M["_batt_idx"] = batt_idx
    M["_ev_idx"] = ev_idx
    M["_price_all"] = price_all
    M["_hour_all"] = hour_all
    M["_batt_soc_all"] = batt_soc_all
    M["_grid_nec_all"] = grid_nec_all
    M["_batt_flat"] = batt_flat
    M["_batt_actions"] = batt_actions

    return M


# =====================================================================
# Print results
# =====================================================================
def print_results(M):
    name = M["run_name"]
    print(f"\n{'='*72}")
    print(f"  {name} -- FULL RESULTS")
    print(f"{'='*72}")

    print(f"\n  --- Episode Summary ---")
    print(f"  Buildings:      {M['n_buildings']}")
    print(f"  Steps:          {M['steps']}")
    print(f"  Total reward:   {M['total_reward']:.2f}")
    print(f"  Mean reward:    {M['mean_reward']:.4f}")
    print(f"  Total cost:     {M['total_cost']:.2f}")

    print(f"\n  --- Per-Constraint Cumulative Costs ---")
    print(f"  C0 (EV departure):     {M['c0_cumulative']:.1f}")
    print(f"  C1 (EV dense):         {M['c1_cumulative']:.1f}")
    print(f"  C2 (battery SoC):      {M['c2_cumulative']:.1f}")
    print(f"  C3 (building power):   {M['c3_cumulative']:.1f}")
    print(f"  C4 (grid power):       {M['c4_cumulative']:.1f}")

    print(f"\n  {'='*60}")
    print(f"  BATTERY METRICS (averaged across {M['n_buildings']} buildings)")
    print(f"  {'='*60}")
    print(f"  Charge % (action > 0.1):     {M['batt_charge_pct']:.1f}%")
    print(f"  Discharge % (action < -0.1): {M['batt_discharge_pct']:.1f}%")
    print(f"  Idle % (|action| <= 0.1):    {M['batt_idle_pct']:.1f}%")
    print(f"  Cycling ratio:               {M['cycling_ratio']:.3f}  (1.0 = perfect)")
    print(f"  CYCLING EMERGED:             {'YES' if M['cycling_emerged'] else 'NO'}")
    print(f"  Action mean: {M['batt_action_mean']:.4f}, std: {M['batt_action_std']:.4f}")
    print(f"  Action range: [{M['batt_action_min']:.4f}, {M['batt_action_max']:.4f}]")
    print(f"  Mean action solar (10-15):   {M['batt_mean_solar']:+.4f}")
    print(f"  Mean action peak  (17-21):   {M['batt_mean_peak']:+.4f}")
    print(f"  Mean action offpeak (22-9):  {M['batt_mean_offpeak']:+.4f}")
    print(f"  Price-action corr:           {M['batt_price_corr']:+.4f}")
    print(f"  SoC overall: mean={M['batt_soc_mean']:.3f}, min={M['batt_soc_min']:.3f}, max={M['batt_soc_max']:.3f}")

    print(f"\n  Battery SoC per building:")
    for i, bs in enumerate(M["batt_soc_per_building"]):
        print(f"    Bld {i}: mean={bs['mean']:.3f}, min={bs['min']:.3f}, max={bs['max']:.3f}")

    print(f"\n  Battery actions per building:")
    for i, bb in enumerate(M["batt_per_building"]):
        print(f"    Bld {i}: charge={bb['charge_pct']:.1f}%, discharge={bb['discharge_pct']:.1f}%, "
              f"idle={bb['idle_pct']:.1f}%, mean={bb['mean']:.4f}")

    print(f"\n  {'='*60}")
    print(f"  EV METRICS")
    print(f"  {'='*60}")
    print(f"  Connected steps:                {M['ev_connected_steps']}")
    print(f"  Charge % (connected):           {M['ev_charge_pct']:.1f}%")
    print(f"  V2G/discharge % (connected):    {M['ev_discharge_pct']:.1f}%")
    print(f"  Idle % (connected):             {M['ev_idle_pct']:.1f}%")
    print(f"  V2G during peak 17-23:          {M['ev_v2g_peak_pct']:.1f}%")
    print(f"  V2G during non-peak:            {M['ev_v2g_nonpeak_pct']:.1f}%")
    print(f"  Mean V2G magnitude at peak:     {M['ev_v2g_peak_mean_magnitude']:.4f}")
    print(f"  Total departures:               {M['total_departures']}")
    print(f"  Compliant departures:           {M['compliant_departures']}")
    print(f"  Violated departures:            {M['violated_departures']}")
    print(f"  C0 violation rate:              {M['c0_violation_rate']:.1f}%")
    print(f"  Mean SoC deficit (violated):    {M['mean_deficit_violated']:.4f}")
    print(f"  Mean SoC deficit (all):         {M['mean_deficit_all']:.4f}")
    print(f"  Mean SoC at departure:          {M['mean_soc_at_departure']:.4f}")

    print(f"\n  {'='*60}")
    print(f"  CONSTRAINT VIOLATIONS")
    print(f"  {'='*60}")
    print(f"  C0 (EV departure):  {M['c0_violations']} violations ({M['c0_violation_pct']:.1f}%), "
          f"mean deficit={M['c0_mean_deficit']:.4f}")
    print(f"  C2 (batt SoC):      {M['c2_violations']} violations ({M['c2_violation_pct']:.2f}%)")
    print(f"  C3 (building power): {M['c3_violations']} violations ({M['c3_violation_pct']:.2f}%)")
    for i, c3b in enumerate(M["c3_per_building"]):
        print(f"    Bld {i}: {c3b['violations']} violations ({c3b['violation_pct']:.2f}%), "
              f"mean|NEC|={c3b['mean_abs_nec']:.3f} kW, max|NEC|={c3b['max_abs_nec']:.3f} kW")
    print(f"  C4 (grid power):    {M['c4_violations']} violations ({M['c4_violation_pct']:.2f}%)")

    print(f"\n  {'='*60}")
    print(f"  KPI METRICS")
    print(f"  {'='*60}")
    print(f"  Total electricity import: {M['total_import_kwh']:.1f} kWh")
    print(f"  Total electricity export: {M['total_export_kwh']:.1f} kWh")
    print(f"  Total electricity cost:   ${M['total_elec_cost']:.2f}")
    print(f"  Mean building power:      {M['mean_building_power_kw']:.3f} kW")
    print(f"  Max building power:       {M['max_building_power_kw']:.3f} kW")
    print(f"  Mean grid power:          {M['mean_grid_power_kw']:.3f} kW")
    print(f"  Max grid power:           {M['max_grid_power_kw']:.3f} kW")
    print(f"  Total ramping:            {M['total_ramping']:.1f} kW")

    print(f"\n  --- Hourly Battery Action Mean ---")
    for h in range(24):
        bar = "#" * int(abs(M["batt_hourly_mean"][h]) * 40)
        sign = "+" if M["batt_hourly_mean"][h] >= 0 else "-"
        print(f"    {h:02d}:00  {M['batt_hourly_mean'][h]:+.4f}  {sign}{bar}")

    print(f"\n  --- Hourly EV Action Mean (connected only) ---")
    for h in range(24):
        bar = "#" * int(abs(M["ev_hourly_mean"][h]) * 40)
        sign = "+" if M["ev_hourly_mean"][h] >= 0 else "-"
        print(f"    {h:02d}:00  {M['ev_hourly_mean'][h]:+.4f}  {sign}{bar}")

    print(f"\n  --- Hourly Mean Price ---")
    for h in range(24):
        print(f"    {h:02d}:00  {M['price_hourly_mean'][h]:.4f}")


# =====================================================================
# Comparison table
# =====================================================================
def print_comparison(m1, m2):
    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE: {m1['run_name']}  vs  {m2['run_name']}")
    print(f"{'='*80}")

    rows = [
        ("Buildings", f"{m1['n_buildings']}", f"{m2['n_buildings']}"),
        ("Steps", f"{m1['steps']}", f"{m2['steps']}"),
        ("Total Reward", f"{m1['total_reward']:.2f}", f"{m2['total_reward']:.2f}"),
        ("Mean Reward", f"{m1['mean_reward']:.4f}", f"{m2['mean_reward']:.4f}"),
        ("Total Cost", f"{m1['total_cost']:.2f}", f"{m2['total_cost']:.2f}"),
        ("", "", ""),
        ("--- BATTERY ---", "", ""),
        ("Charge %", f"{m1['batt_charge_pct']:.1f}%", f"{m2['batt_charge_pct']:.1f}%"),
        ("Discharge %", f"{m1['batt_discharge_pct']:.1f}%", f"{m2['batt_discharge_pct']:.1f}%"),
        ("Idle %", f"{m1['batt_idle_pct']:.1f}%", f"{m2['batt_idle_pct']:.1f}%"),
        ("Cycling Ratio", f"{m1['cycling_ratio']:.3f}", f"{m2['cycling_ratio']:.3f}"),
        ("Cycling Emerged", f"{'YES' if m1['cycling_emerged'] else 'NO'}", f"{'YES' if m2['cycling_emerged'] else 'NO'}"),
        ("Solar Mean Action", f"{m1['batt_mean_solar']:+.4f}", f"{m2['batt_mean_solar']:+.4f}"),
        ("Peak Mean Action", f"{m1['batt_mean_peak']:+.4f}", f"{m2['batt_mean_peak']:+.4f}"),
        ("Off-Peak Mean Action", f"{m1['batt_mean_offpeak']:+.4f}", f"{m2['batt_mean_offpeak']:+.4f}"),
        ("Price-Action Corr", f"{m1['batt_price_corr']:+.4f}", f"{m2['batt_price_corr']:+.4f}"),
        ("SoC Mean", f"{m1['batt_soc_mean']:.3f}", f"{m2['batt_soc_mean']:.3f}"),
        ("SoC Range", f"[{m1['batt_soc_min']:.3f}, {m1['batt_soc_max']:.3f}]", f"[{m2['batt_soc_min']:.3f}, {m2['batt_soc_max']:.3f}]"),
        ("", "", ""),
        ("--- EV ---", "", ""),
        ("Connected Steps", f"{m1['ev_connected_steps']}", f"{m2['ev_connected_steps']}"),
        ("Charge %", f"{m1['ev_charge_pct']:.1f}%", f"{m2['ev_charge_pct']:.1f}%"),
        ("V2G %", f"{m1['ev_discharge_pct']:.1f}%", f"{m2['ev_discharge_pct']:.1f}%"),
        ("V2G Peak %", f"{m1['ev_v2g_peak_pct']:.1f}%", f"{m2['ev_v2g_peak_pct']:.1f}%"),
        ("V2G Non-Peak %", f"{m1['ev_v2g_nonpeak_pct']:.1f}%", f"{m2['ev_v2g_nonpeak_pct']:.1f}%"),
        ("V2G Peak Magnitude", f"{m1['ev_v2g_peak_mean_magnitude']:.4f}", f"{m2['ev_v2g_peak_mean_magnitude']:.4f}"),
        ("Total Departures", f"{m1['total_departures']}", f"{m2['total_departures']}"),
        ("Violated Departures", f"{m1['violated_departures']}", f"{m2['violated_departures']}"),
        ("C0 Violation Rate", f"{m1['c0_violation_rate']:.1f}%", f"{m2['c0_violation_rate']:.1f}%"),
        ("Mean SoC @ Departure", f"{m1['mean_soc_at_departure']:.4f}", f"{m2['mean_soc_at_departure']:.4f}"),
        ("", "", ""),
        ("--- CONSTRAINTS ---", "", ""),
        ("C0 Violations", f"{m1['c0_violations']} ({m1['c0_violation_pct']:.1f}%)", f"{m2['c0_violations']} ({m2['c0_violation_pct']:.1f}%)"),
        ("C2 Violations", f"{m1['c2_violations']} ({m1['c2_violation_pct']:.2f}%)", f"{m2['c2_violations']} ({m2['c2_violation_pct']:.2f}%)"),
        ("C3 Violations", f"{m1['c3_violations']} ({m1['c3_violation_pct']:.2f}%)", f"{m2['c3_violations']} ({m2['c3_violation_pct']:.2f}%)"),
        ("C4 Violations", f"{m1['c4_violations']} ({m1['c4_violation_pct']:.2f}%)", f"{m2['c4_violations']} ({m2['c4_violation_pct']:.2f}%)"),
        ("", "", ""),
        ("--- KPIs ---", "", ""),
        ("Import (kWh)", f"{m1['total_import_kwh']:.1f}", f"{m2['total_import_kwh']:.1f}"),
        ("Export (kWh)", f"{m1['total_export_kwh']:.1f}", f"{m2['total_export_kwh']:.1f}"),
        ("Elec Cost ($)", f"{m1['total_elec_cost']:.2f}", f"{m2['total_elec_cost']:.2f}"),
        ("Mean Bld Power (kW)", f"{m1['mean_building_power_kw']:.3f}", f"{m2['mean_building_power_kw']:.3f}"),
        ("Max Bld Power (kW)", f"{m1['max_building_power_kw']:.3f}", f"{m2['max_building_power_kw']:.3f}"),
        ("Mean Grid Power (kW)", f"{m1['mean_grid_power_kw']:.3f}", f"{m2['mean_grid_power_kw']:.3f}"),
        ("Max Grid Power (kW)", f"{m1['max_grid_power_kw']:.3f}", f"{m2['max_grid_power_kw']:.3f}"),
        ("Ramping (kW)", f"{m1['total_ramping']:.1f}", f"{m2['total_ramping']:.1f}"),
        ("", "", ""),
        ("--- COSTS (cumulative) ---", "", ""),
        ("C0 Cost", f"{m1['c0_cumulative']:.1f}", f"{m2['c0_cumulative']:.1f}"),
        ("C1 Cost", f"{m1['c1_cumulative']:.1f}", f"{m2['c1_cumulative']:.1f}"),
        ("C2 Cost", f"{m1['c2_cumulative']:.1f}", f"{m2['c2_cumulative']:.1f}"),
        ("C3 Cost", f"{m1['c3_cumulative']:.1f}", f"{m2['c3_cumulative']:.1f}"),
        ("C4 Cost", f"{m1['c4_cumulative']:.1f}", f"{m2['c4_cumulative']:.1f}"),
    ]

    # Print table
    col_w = 25
    header = f"  {'Metric':<30s} {m1['run_name']:>{col_w}s} {m2['run_name']:>{col_w}s}"
    print(header)
    print(f"  {'-'*30} {'-'*col_w} {'-'*col_w}")
    for label, v1, v2 in rows:
        if label == "":
            print()
        else:
            print(f"  {label:<30s} {v1:>{col_w}s} {v2:>{col_w}s}")


# =====================================================================
# Visualization
# =====================================================================
def make_visualization(m1, m2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("Comprehensive Evaluation: R30c (5-bld) vs 1-Building Test", fontsize=14, fontweight='bold')

    hours = list(range(24))

    # --- Panel 1: Battery action by hour (both overlaid) ---
    ax = axes[0, 0]
    bm1 = np.array(m1["batt_hourly_mean"])
    bm2 = np.array(m2["batt_hourly_mean"])
    width = 0.35
    ax.bar(np.array(hours) - width/2, bm1, width, label=m1["run_name"], color='steelblue', alpha=0.8)
    ax.bar(np.array(hours) + width/2, bm2, width, label=m2["run_name"], color='coral', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvspan(10, 15, alpha=0.06, color='gold', label='Solar')
    ax.axvspan(17, 21, alpha=0.06, color='salmon', label='Peak')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Battery Action")
    ax.set_title("Battery Action by Hour")
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xticks(hours)
    ax.set_xlim(-0.5, 23.5)

    # --- Panel 2: EV action by hour (both overlaid) ---
    ax = axes[0, 1]
    em1 = np.array(m1["ev_hourly_mean"])
    em2 = np.array(m2["ev_hourly_mean"])
    ax.bar(np.array(hours) - width/2, em1, width, label=m1["run_name"], color='steelblue', alpha=0.8)
    ax.bar(np.array(hours) + width/2, em2, width, label=m2["run_name"], color='coral', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvspan(17, 23, alpha=0.06, color='salmon', label='Peak (V2G)')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean EV Action (connected)")
    ax.set_title("EV Action by Hour (Connected Only)")
    ax.legend(fontsize=7)
    ax.set_xticks(hours)
    ax.set_xlim(-0.5, 23.5)

    # --- Panel 3: Battery action distribution (both overlaid) ---
    ax = axes[0, 2]
    bf1 = m1["_batt_flat"]
    bf2 = m2["_batt_flat"]
    if bf1.size > 0:
        ax.hist(bf1, bins=80, alpha=0.6, color='steelblue', label=m1["run_name"], density=True)
    if bf2.size > 0:
        ax.hist(bf2, bins=80, alpha=0.6, color='coral', label=m2["run_name"], density=True)
    ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.6, label='Charge thresh')
    ax.axvline(x=-0.1, color='red', linestyle='--', alpha=0.6, label='Discharge thresh')
    ax.set_xlabel("Battery Action Value")
    ax.set_ylabel("Density")
    ax.set_title("Battery Action Distribution")
    ax.legend(fontsize=7)

    # --- Panel 4: Price vs battery action (1 week, R30c) ---
    ax = axes[1, 0]
    batt_actions_r30 = m1["_batt_actions"]
    prices_r30 = m1["_price_all"]
    week_len = min(168, m1["steps"])
    t_range = np.arange(week_len)

    if batt_actions_r30.size > 0 and batt_actions_r30.ndim > 1:
        batt_avg = batt_actions_r30[:week_len].mean(axis=1)
    elif batt_actions_r30.size > 0:
        batt_avg = batt_actions_r30[:week_len].ravel()
    else:
        batt_avg = np.zeros(week_len)

    prices_week = np.array(prices_r30[:week_len])
    ax2 = ax.twinx()
    ax.plot(t_range, prices_week, color='darkgoldenrod', alpha=0.6, linewidth=0.8, label='Price')
    ax2.plot(t_range, batt_avg, color='steelblue', alpha=0.7, linewidth=0.8, label='Batt action')
    ax2.axhline(y=0, color='gray', linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Hour (first week)")
    ax.set_ylabel("Electricity Price", color='darkgoldenrod')
    ax2.set_ylabel("Battery Action", color='steelblue')
    ax.set_title(f"Price vs Battery Action (Week 1, {m1['run_name']})")

    if len(prices_week) > 10:
        valid = np.isfinite(prices_week) & np.isfinite(batt_avg)
        if valid.sum() > 10:
            corr = np.corrcoef(prices_week[valid], batt_avg[valid])[0, 1]
            ax.text(0.02, 0.95, f"corr={corr:.3f}", transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Panel 5: Battery SoC trajectory (1 week, R30c) ---
    ax = axes[1, 1]
    batt_soc = np.array(m1["_batt_soc_all"])
    n_bld = m1["n_buildings"]
    week_soc = batt_soc[:week_len]
    colors_bld = ['steelblue', 'coral', 'green', 'purple', 'orange']
    for b in range(min(n_bld, 5)):
        ax.plot(t_range, week_soc[:, b], color=colors_bld[b % len(colors_bld)],
                alpha=0.7, linewidth=0.8, label=f'Bld {b}')
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.4, label='SoC high (0.95)')
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.4, label='SoC low (0.0)')
    ax.set_xlabel("Hour (first week)")
    ax.set_ylabel("Battery SoC")
    ax.set_title(f"Battery SoC Trajectory (Week 1, {m1['run_name']})")
    ax.legend(fontsize=6, loc='upper right')
    ax.set_ylim(-0.05, 1.05)

    # --- Panel 6: Constraint violation summary (bar chart) ---
    ax = axes[1, 2]
    labels = ['C0\n(EV depart)', 'C3\n(bld power)', 'C4\n(grid power)']
    vals1 = [m1['c0_violation_pct'], m1['c3_violation_pct'], m1['c4_violation_pct']]
    vals2 = [m2['c0_violation_pct'], m2['c3_violation_pct'], m2['c4_violation_pct']]

    x = np.arange(len(labels))
    ax.bar(x - width/2, vals1, width, label=m1["run_name"], color='steelblue', alpha=0.8)
    ax.bar(x + width/2, vals2, width, label=m2["run_name"], color='coral', alpha=0.8)
    ax.set_ylabel("Violation %")
    ax.set_title("Constraint Violation Rates")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)

    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip(vals1, vals2)):
        ax.text(i - width/2, v1 + 0.3, f"{v1:.1f}%", ha='center', va='bottom', fontsize=7)
        ax.text(i + width/2, v2 + 0.3, f"{v2:.1f}%", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\n  Saved visualization: {OUT_PNG}")
    plt.close()


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Comprehensive evaluation of R30c and 1-building test")
    ap.add_argument("--r30c_epoch", type=int, default=None, help="R30c epoch (default: latest)")
    ap.add_argument("--onebld_epoch", type=int, default=None, help="1-building epoch (default: latest)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Determine epochs
    r30c_epoch = args.r30c_epoch
    if r30c_epoch is None:
        r30c_epoch = find_latest_epoch(R30C_CKPT_DIR)
        print(f"R30c: using latest epoch = {r30c_epoch}")

    onebld_epoch = args.onebld_epoch
    if onebld_epoch is None:
        onebld_epoch = find_latest_epoch(ONEBLD_CKPT_DIR)
        print(f"1-bld: using latest epoch = {onebld_epoch}")

    # ---- Evaluate R30c (5-building, uses Saute) ----
    m_r30c = evaluate_run(
        run_name="R30c (5-bld)",
        ckpt_dir=R30C_CKPT_DIR,
        epoch=r30c_epoch,
        env_vars=R30C_ENV_VARS,
        use_saute=True,
        seed=args.seed,
    )
    print_results(m_r30c)

    # ---- Evaluate 1-building (no Saute) ----
    m_1bld = evaluate_run(
        run_name="1-bld Test",
        ckpt_dir=ONEBLD_CKPT_DIR,
        epoch=onebld_epoch,
        env_vars=ONEBLD_ENV_VARS,
        use_saute=False,
        seed=args.seed,
    )
    print_results(m_1bld)

    # ---- Comparison ----
    print_comparison(m_r30c, m_1bld)

    # ---- Visualization ----
    print("\n  Generating combined visualization...")
    make_visualization(m_r30c, m_1bld)

    print(f"\n{'='*72}")
    print(f"  ALL EVALUATIONS COMPLETE")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
