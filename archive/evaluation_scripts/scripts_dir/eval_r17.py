#!/usr/bin/env python3
"""
R17 Evaluation -- Compare R15b (benchmark) vs R17 (threshold r_sg)

Includes all standard metrics from eval_r16.py PLUS:
  1. Reward component breakdown (per-component mean/std/min/max, variance contribution)
  2. C3 structural vs controllable breakdown
  3. Battery charging cycle analysis
  4. Price arbitrage verification
"""
import os
import sys
import json
import glob as glob_mod
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

EVAL_SEED = 42

# =====================================================================
# Environment profile (eval mode -- no Saute, no PID, no clamp)
# We use R17's reward weights so reward components reflect R17's design,
# but disable training-only wrappers (Saute, PID, clamp) for clean eval.
# =====================================================================
EVAL_ENV_VARS = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_PNORM_P": "4.0",
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.05",
    "CITYLEARN_EV_COST_SCALE": "3.0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_KPI_RUN_NAME": "__eval_disabled__",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    # Disable training-only wrappers
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_PID_LAGRANGE": "0",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    # R17 reward weights for component analysis
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_GRID": "1.5",
    "STEMS_SG_THRESHOLD": "0.5",
    "STEMS_ALPHA_BUILD": "0.0",
    "STEMS_BETA_RAMP": "0.3",
    "STEMS_XI_RENEWABLE": "0.2",
    "STEMS_LAMBDA_EV": "5.0",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_ALPHA_LOAD_SHIFT": "6.0",
    "STEMS_ALPHA_GRID_MILD": "0.3",
    "STEMS_ALPHA_EV_GUARD": "0.0",
    "STEMS_ALPHA_V2G_CONTEXT": "3.0",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
    # C3 controllable ON (R17 feature -- needed for structural breakdown)
    "CITYLEARN_C3_CONTROLLABLE": "1",
}


def set_env():
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(k, None)
    for k, v in EVAL_ENV_VARS.items():
        os.environ[k] = v


set_env()

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env
import citylearn_safe.schema_index as si


RUN_DIRS = {
    "R15b": "runs/r15b_v2g_context/5bld",
    "R17":  "runs/r17_threshold_sg/5bld",
}


def find_checkpoint(run_dir, epoch=None):
    pattern = os.path.join(PROJECT, run_dir, "PPOLagMulti-*", "seed-*", "torch_save")
    save_dirs = sorted(glob_mod.glob(pattern))
    if not save_dirs:
        return ""
    save_dir = save_dirs[-1]
    if epoch is not None:
        ckpt = os.path.join(save_dir, f"epoch-{epoch}.pt")
        if os.path.exists(ckpt):
            return ckpt
    ckpts = sorted(glob_mod.glob(os.path.join(save_dir, "epoch-*.pt")))
    return ckpts[-1] if ckpts else ""


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
    return actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim


# =====================================================================
# Reward component keys -- computed inline during eval
# (omni_env_v2 wrapper is not used during eval, so we compute manually)
# =====================================================================
REWARD_COMPONENT_KEYS = [
    "r_eco", "r_sg", "r_sb", "r_ramp", "r_ren",
    "r_load_shift", "r_grid_mild", "r_barrier",
]

# R17 reward config for inline computation
REWARD_CFG = {
    "mu_economic": 0.0,
    "alpha_grid": 1.5,
    "sg_threshold_frac": 0.5,
    "alpha_build": 0.0,
    "beta_ramp": 0.3,
    "xi_renewable": 0.2,
    "alpha_load_shift": 6.0,
    "alpha_grid_mild": 0.3,
    "alpha_barrier": 0.5,
    "sb_asymmetric": True,
    "sg_export_credit": 0.5,
    "P_building_max": 4.6083,
    "P_grid_max": 10.2352,
}


def compute_reward_components(action, buildings, t_idx, batt_idx, prev_net, mean_price, cfg):
    """Compute reward components inline (replicates omni_env_v2._stems_reward logic)."""
    # --- Gather building data ---
    total_net = 0.0
    solar_total = 0.0
    for b in buildings:
        try:
            nec = getattr(b, 'net_electricity_consumption', None)
            if nec is not None and hasattr(nec, '__len__') and len(nec) > t_idx:
                total_net += float(nec[t_idx])
        except Exception:
            pass
        try:
            s = getattr(b, 'solar_generation', None)
            if s is not None and hasattr(s, '__len__') and len(s) > t_idx:
                solar_total += abs(float(s[t_idx]))
        except Exception:
            pass

    imp = max(0.0, total_net)
    exp = max(0.0, -total_net)
    ef = 0.7  # CITYLEARN_EXPORT_FACTOR

    # r_eco
    try:
        pr = buildings[0].pricing.electricity_pricing
        price = float(pr[t_idx]) if hasattr(pr, '__len__') and len(pr) > t_idx else 0.17
    except Exception:
        price = 0.17
    r_eco = -cfg["mu_economic"] * price * (imp - ef * exp)

    # r_sg (with threshold)
    imp_eff = imp
    if cfg["sg_export_credit"] > 0 and exp > 0:
        imp_eff = max(0.0, imp - cfg["sg_export_credit"] * exp)
    if cfg["sg_threshold_frac"] > 0:
        sg_thresh = cfg["P_grid_max"] * cfg["sg_threshold_frac"]
        imp_above = max(0.0, imp_eff - sg_thresh)
    else:
        imp_above = imp_eff
    r_sg = cfg["alpha_grid"] * (1.0 - min((imp_above / max(1e-6, cfg["P_grid_max"])) ** 2, 4.0))

    # r_sb (building stability)
    bs, bc = 0.0, 0
    for b in buildings:
        try:
            nec = getattr(b, 'net_electricity_consumption', None)
            if nec is not None and hasattr(nec, '__len__') and len(nec) > t_idx:
                nec_val = float(nec[t_idx])
                if cfg["sb_asymmetric"]:
                    ratio = max(0.0, nec_val) / max(1e-6, cfg["P_building_max"])
                else:
                    ratio = abs(nec_val) / max(1e-6, cfg["P_building_max"])
                bs += 1.0 - min(ratio, 4.0)
                bc += 1
        except Exception:
            pass
    r_sb = cfg["alpha_build"] * (bs / max(1, bc)) if bc > 0 else 0.0

    # r_ramp
    rd = abs(total_net - prev_net) if prev_net is not None else 0.0
    r_ramp = -cfg["beta_ramp"] * (rd / max(1e-6, cfg["P_grid_max"]))

    # r_ren (renewable utilization)
    r_ren = cfg["xi_renewable"] * min(solar_total / (solar_total + imp), 1.0) if (solar_total + imp) > 0 else 0.0

    # r_load_shift (battery price arbitrage)
    r_load_shift = 0.0
    if cfg["alpha_load_shift"] > 0 and batt_idx:
        price_dev = price / max(1e-8, mean_price) - 1.0
        ls_sum = 0.0
        for bi in batt_idx:
            if bi < len(action):
                act = float(action[bi])
                ls_sum += -act * price_dev
        r_load_shift = cfg["alpha_load_shift"] * ls_sum / max(1, len(batt_idx))

    # r_grid_mild
    r_grid_mild = -cfg["alpha_grid_mild"] * (imp / max(1e-6, cfg["P_grid_max"])) if cfg["alpha_grid_mild"] > 0 else 0.0

    # r_barrier (battery SoC barrier)
    # We need battery SoC info -- get from info dict
    r_barrier = 0.0
    # (computed separately using info dict)

    return {
        "r_eco": r_eco,
        "r_sg": r_sg,
        "r_sb": r_sb,
        "r_ramp": r_ramp,
        "r_ren": r_ren,
        "r_load_shift": r_load_shift,
        "r_grid_mild": r_grid_mild,
        "r_barrier": r_barrier,
        "total_net": total_net,
        "price": price,
    }


def run_eval_episode(actor, obs_mean, obs_std, obs_clip, actor_obs_dim, label):
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    si._CACHE = None
    set_env()

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)
    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(getattr(raw, "buildings", []))
    n_buildings = len(buildings)

    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)
    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]

    try:
        pr = buildings[0].pricing.electricity_pricing
        all_prices = np.array(pr, dtype=float)
        p25, p50, p75 = np.percentile(all_prices, [25, 50, 75])
        mean_price = float(np.mean(all_prices))
    except Exception:
        p25, p50, p75 = 0.12, 0.16, 0.20
        mean_price = 0.16

    P_building_max = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
    P_grid_max = float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "10.2352"))

    env_obs_dim = env.observation_space.shape[0]
    need_pad = actor_obs_dim > env_obs_dim
    pad_dim = actor_obs_dim - env_obs_dim if need_pad else 0

    print(f"  [{label}] env_obs={env_obs_dim}, actor_obs={actor_obs_dim}, "
          f"pad={pad_dim}, batt={batt_idx}, ev={ev_idx}")

    obs, _ = env.reset(seed=EVAL_SEED)
    done = False
    step = 0

    rewards, costs = [], []
    actions_all = []
    c0_vals, c1_vals, c2_vals, c3_vals, c4_vals = [], [], [], [], []

    total_departures = 0
    violated_departures = 0
    departure_deficits = []
    ev_tracker = {}

    building_power_violations = [0] * n_buildings
    building_power_steps = [0] * n_buildings

    # Per-building structural C3 tracking
    building_structural_violations = [0] * n_buildings
    building_controllable_violations = [0] * n_buildings

    hourly_batt = {h: [] for h in range(24)}
    hourly_ev = {h: [] for h in range(24)}
    hourly_net_load = {h: [] for h in range(24)}
    hourly_price = {h: [] for h in range(24)}
    hourly_solar = {h: [] for h in range(24)}

    # Reward component tracking (computed inline)
    reward_components = {k: [] for k in REWARD_COMPONENT_KEYS}

    # Hourly reward component profiles (r_sg and r_load_shift)
    hourly_r_sg = {h: [] for h in range(24)}
    hourly_r_load_shift = {h: [] for h in range(24)}

    # Track previous net load for ramp computation
    prev_net_for_ramp = None

    # Per-step price and battery action for arbitrage analysis
    step_prices = []
    step_batt_actions = []

    while not done:
        if need_pad:
            obs_padded = np.concatenate([obs, np.ones(pad_dim)])
        else:
            obs_padded = obs

        obs_t = torch.as_tensor(obs_padded, dtype=torch.float32).unsqueeze(0)
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

        # EV departure tracking
        for b_idx, bld in enumerate(buildings):
            chargers = getattr(bld, "electric_vehicle_chargers", None) or []
            for ch_idx, ch in enumerate(chargers):
                key = (b_idx, ch_idx)
                sim = getattr(ch, 'charger_simulation',
                              getattr(ch, '_Charger__charger_simulation', None))
                if sim is None:
                    continue
                try:
                    sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                    current_connected = t_now < len(sa) and float(sa[t_now]) == 1.0
                    if current_connected:
                        ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                        rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                        if not np.isfinite(rs):
                            rs = 1.0
                        ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                        current_soc = 0.0
                        if ev_obj is not None:
                            bt = getattr(ev_obj, 'battery', None)
                            if bt is not None:
                                soc_arr = getattr(bt, 'soc', None)
                                if soc_arr is not None:
                                    sn = np.asarray(soc_arr, dtype=float)
                                    current_soc = float(np.clip(sn[t_idx], 0, 1)) if 0 <= t_idx < len(sn) else 0.0
                        ev_tracker[key] = {'was_connected': True, 'last_soc': current_soc, 'required_soc': rs}
                    else:
                        prev = ev_tracker.get(key, {})
                        if prev.get('was_connected', False):
                            total_departures += 1
                            last_soc = prev['last_soc']
                            rs = prev['required_soc']
                            deficit = max(0.0, rs - last_soc)
                            if deficit > 0.01:
                                violated_departures += 1
                            departure_deficits.append(deficit)
                        ev_tracker[key] = {'was_connected': False}
                except Exception:
                    pass

        price = 0.17
        try:
            price = float(buildings[0].pricing.electricity_pricing[t_idx])
        except Exception:
            pass
        step_prices.append(price)

        # Track battery action for this step
        batt_act_mean = np.mean([action[i] for i in batt_idx]) if batt_idx else 0.0
        step_batt_actions.append(batt_act_mean)

        solar = 0.0
        for b in buildings:
            try:
                s = getattr(b, "solar_generation", None)
                if s is not None and len(s) > t_idx:
                    solar += abs(float(s[t_idx]))
            except Exception:
                pass

        total_net = 0.0
        for b_idx, b in enumerate(buildings):
            try:
                nec = getattr(b, "net_electricity_consumption", None)
                if nec is not None and len(nec) > t_idx:
                    bld_net = float(nec[t_idx])
                    total_net += bld_net
                    building_power_steps[b_idx] += 1
                    if abs(bld_net) > P_building_max:
                        building_power_violations[b_idx] += 1

                    # C3 structural vs controllable breakdown
                    # Structural: base load (NSL + solar) already exceeds threshold
                    nsl_arr = np.asarray(
                        getattr(b, "_Building__energy_to_non_shiftable_load", []),
                        dtype=float)
                    sg_arr = np.asarray(
                        getattr(b, "_Building__solar_generation", []),
                        dtype=float)
                    nsl_i = 0.0
                    if len(nsl_arr) > t_idx:
                        nsl_i = float(nsl_arr[t_idx])
                    if len(sg_arr) > t_idx:
                        nsl_i += float(sg_arr[t_idx])  # base = NSL + solar (solar is negative)

                    raw_violation = abs(bld_net) > P_building_max
                    structural_violation = abs(nsl_i) > P_building_max

                    if raw_violation:
                        if structural_violation:
                            building_structural_violations[b_idx] += 1
                        else:
                            building_controllable_violations[b_idx] += 1
            except Exception:
                pass

        hourly_batt[hour].append(np.mean([action[i] for i in batt_idx]) if batt_idx else 0)
        hourly_ev[hour].append(np.mean([action[i] for i in ev_idx]) if ev_idx else 0)
        hourly_net_load[hour].append(total_net)
        hourly_price[hour].append(price)
        hourly_solar[hour].append(solar)

        obs, reward, term, trunc, info = env.step(action)
        done = bool(term) or bool(trunc)
        step += 1

        rewards.append(float(reward))
        costs.append(float(info.get("cost", 0.0)))
        c0_vals.append(float(info.get("cost_ev_dense", 0.0)))
        c1_vals.append(float(info.get("cost_ev_departure", 0.0)))
        c2_vals.append(float(info.get("cost_stems_battery", 0.0)))
        c3_vals.append(float(info.get("cost_stems_building_power", 0.0)))
        c4_vals.append(float(info.get("cost_stems_grid_power", 0.0)))

        # Compute reward components inline (omni_env_v2 wrapper not used in eval)
        rc = compute_reward_components(
            action, buildings, t_idx, batt_idx, prev_net_for_ramp, mean_price, REWARD_CFG)
        prev_net_for_ramp = rc["total_net"]

        # Compute r_barrier from info dict battery SoC values
        r_barrier_val = 0.0
        n_batt_for_barrier = len(batt_idx)
        if REWARD_CFG["alpha_barrier"] > 0 and n_batt_for_barrier > 0:
            for b_i in range(1, n_batt_for_barrier + 1):
                soc = float(info.get(f'battery_soc_b{b_i}', 0.5))
                if soc < 0.05:
                    pen = -1.0
                elif soc < 0.10:
                    pen = -(0.10 - soc) / 0.05
                elif soc > 0.90:
                    pen = -1.0
                elif soc > 0.85:
                    pen = -(soc - 0.85) / 0.05
                else:
                    pen = 0.0
                r_barrier_val += pen
            r_barrier_val *= REWARD_CFG["alpha_barrier"] / n_batt_for_barrier
        rc["r_barrier"] = r_barrier_val

        for k in REWARD_COMPONENT_KEYS:
            reward_components[k].append(float(rc.get(k, 0.0)))

        # Hourly r_sg and r_load_shift profiles
        hourly_r_sg[hour].append(float(rc["r_sg"]))
        hourly_r_load_shift[hour].append(float(rc["r_load_shift"]))

    # CityLearn KPIs
    cl_kpis = {}
    try:
        city = raw
        if hasattr(city, 'evaluate'):
            eval_df = city.evaluate()
            if eval_df is not None and not eval_df.empty:
                if "name" in eval_df.columns:
                    district = eval_df[eval_df["name"] == "District"]
                else:
                    district = eval_df
                for _, row in district.iterrows():
                    cf = str(row.get("cost_function", ""))
                    val = row.get("value", None)
                    if cf and val is not None:
                        try:
                            cl_kpis[cf] = float(val)
                        except (ValueError, TypeError):
                            pass
    except Exception as e:
        print(f"  [{label}] CityLearn evaluate() error: {e}")

    env.close()

    actions_arr = np.array(actions_all)
    N = step

    hourly_batt_means = [np.mean(hourly_batt[h]) if hourly_batt[h] else 0 for h in range(24)]
    hourly_ev_means = [np.mean(hourly_ev[h]) if hourly_ev[h] else 0 for h in range(24)]
    hourly_net_means = [np.mean(hourly_net_load[h]) if hourly_net_load[h] else 0 for h in range(24)]
    hourly_price_means = [np.mean(hourly_price[h]) if hourly_price[h] else 0 for h in range(24)]
    hourly_solar_means = [np.mean(hourly_solar[h]) if hourly_solar[h] else 0 for h in range(24)]

    hourly_r_sg_means = [np.mean(hourly_r_sg[h]) if hourly_r_sg[h] else 0 for h in range(24)]
    hourly_r_ls_means = [np.mean(hourly_r_load_shift[h]) if hourly_r_load_shift[h] else 0 for h in range(24)]

    per_building_c3_pct = []
    for b_idx in range(n_buildings):
        if building_power_steps[b_idx] > 0:
            pct = 100.0 * building_power_violations[b_idx] / building_power_steps[b_idx]
        else:
            pct = 0.0
        per_building_c3_pct.append(pct)

    # Per-building structural vs controllable C3
    per_building_structural_pct = []
    per_building_controllable_pct = []
    for b_idx in range(n_buildings):
        total_steps = building_power_steps[b_idx] if building_power_steps[b_idx] > 0 else 1
        per_building_structural_pct.append(100.0 * building_structural_violations[b_idx] / total_steps)
        per_building_controllable_pct.append(100.0 * building_controllable_violations[b_idx] / total_steps)

    total_structural = sum(building_structural_violations)
    total_controllable = sum(building_controllable_violations)
    total_c3_violations = sum(building_power_violations)

    ev_departure_violation_pct = (100.0 * violated_departures / total_departures
                                  if total_departures > 0 else 0.0)
    avg_departure_deficit = float(np.mean(departure_deficits)) if departure_deficits else 0.0

    # Intelligence metrics
    mean_ev = np.mean([actions_arr[:, i] for i in ev_idx], axis=0) if ev_idx else np.zeros(N)
    mean_batt = np.mean([actions_arr[:, i] for i in batt_idx], axis=0) if batt_idx else np.zeros(N)
    prices_arr = np.array([hourly_price_means[t % 24] for t in range(N)])
    solar_arr = np.array([hourly_solar_means[t % 24] for t in range(N)])

    solar_med = np.median(solar_arr[solar_arr > 0]) if (solar_arr > 0).any() else 1.0
    high_solar = solar_arr > solar_med
    low_solar = solar_arr < 0.01
    ev_solar_pref = (float(mean_ev[high_solar].mean() - mean_ev[low_solar].mean())
                     if high_solar.sum() > 0 and low_solar.sum() > 0 else 0.0)

    expensive = prices_arr > p75
    cheap = prices_arr < p25
    batt_price_spread = 0.0
    if expensive.sum() > 0 and cheap.sum() > 0:
        batt_price_spread = float(mean_batt[cheap].mean() - mean_batt[expensive].mean())

    actual_prices = np.array([0.17] * N)
    try:
        pr_data = buildings[0].pricing.electricity_pricing
        for t in range(N):
            actual_prices[t] = float(pr_data[t]) if t < len(pr_data) else 0.17
    except Exception:
        pass
    corr_batt_price = float(np.corrcoef(actual_prices, mean_batt)[0, 1])
    corr_ev_price = float(np.corrcoef(actual_prices, mean_ev)[0, 1]) if ev_idx else 0.0

    hours_arr = np.array([t % 24 for t in range(N)])
    peak_hours = {h for h in range(24) if hourly_price_means[h] > p75}
    pre_peak_hours = set()
    for ph in peak_hours:
        for offset in [2, 3, 4]:
            pre_h = (ph - offset) % 24
            if pre_h not in peak_hours:
                pre_peak_hours.add(pre_h)

    pre_peak_batt = 0.0
    peak_batt = 0.0
    if pre_peak_hours and peak_hours:
        pre_mask = np.isin(hours_arr, list(pre_peak_hours))
        peak_mask = np.isin(hours_arr, list(peak_hours))
        if pre_mask.sum() > 0:
            pre_peak_batt = float(mean_batt[pre_mask].mean())
        if peak_mask.sum() > 0:
            peak_batt = float(mean_batt[peak_mask].mean())

    net_loads = np.array([hourly_net_means[t % 24] for t in range(N)])
    corr_batt_load = float(np.corrcoef(net_loads, mean_batt)[0, 1])

    hourly_batt_arr = np.array(hourly_batt_means)
    action_diversity = float(np.std(hourly_batt_arr))

    intel_scores = {
        "solar_preference": float(np.clip(ev_solar_pref / 0.2, 0, 1)),
        "price_spread": float(np.clip(batt_price_spread / 0.3, 0, 1)),
        "price_correlation": float(np.clip(-corr_batt_price / 0.3, 0, 1)),
        "pre_peak_planning": float(np.clip((pre_peak_batt - peak_batt) / 0.3, 0, 1)),
        "peak_shave_corr": float(np.clip(-corr_batt_load / 0.3, 0, 1)),
        "ev_compliance": float(np.clip(1.0 - ev_departure_violation_pct / 50.0, 0, 1)),
        "behavioral_diversity": float(np.clip(action_diversity / 0.15, 0, 1)),
    }
    intel_overall = float(np.mean(list(intel_scores.values())))

    # --- Reward component statistics ---
    reward_comp_stats = {}
    for k in REWARD_COMPONENT_KEYS:
        arr = np.array(reward_components[k])
        reward_comp_stats[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    # Variance contribution: fraction of total reward variance explained by each component
    total_reward_arr = np.array(rewards)
    total_var = float(np.var(total_reward_arr))
    for k in REWARD_COMPONENT_KEYS:
        arr = np.array(reward_components[k])
        comp_var = float(np.var(arr))
        reward_comp_stats[k]["var_pct"] = (
            100.0 * comp_var / total_var if total_var > 1e-12 else 0.0)

    # --- Battery charging cycle analysis ---
    step_batt_arr = np.array(step_batt_actions)
    charging_hours = []
    discharging_hours = []
    for h in range(24):
        hm = hourly_batt_means[h]
        if hm > 0.001:
            charging_hours.append(h)
        elif hm < -0.001:
            discharging_hours.append(h)

    # Daily energy throughput (simplified: |action| proportional to kWh)
    # Battery capacity ~6.4kWh (typical CityLearn), nominal power ~5kW
    # action in [-1,1] maps to [-nominal_power, +nominal_power] per step (1 hour)
    # So energy per step = action * nominal_power (kWh)
    # For aggregate metric, we just track action magnitude
    daily_charge = 0.0
    daily_discharge = 0.0
    n_days = max(1, N / 24)
    for t in range(N):
        act = step_batt_arr[t]
        if act > 0:
            daily_charge += act
        else:
            daily_discharge += abs(act)
    daily_charge /= n_days
    daily_discharge /= n_days

    # --- Price arbitrage verification ---
    step_prices_arr = np.array(step_prices)
    price_sorted_idx = np.argsort(step_prices_arr)
    n_cheapest = max(1, int(N * 4.0 / 24.0))  # ~4 hours per day
    n_expensive = n_cheapest
    cheapest_idx = price_sorted_idx[:n_cheapest]
    expensive_idx = price_sorted_idx[-n_expensive:]
    avg_batt_cheapest = float(np.mean(step_batt_arr[cheapest_idx]))
    avg_batt_expensive = float(np.mean(step_batt_arr[expensive_idx]))

    # Revenue proxy: sum of -action * (price/mean_price - 1)
    price_dev = step_prices_arr / max(1e-8, mean_price) - 1.0
    arbitrage_revenue = float(np.sum(-step_batt_arr * price_dev))

    return {
        "label": label,
        "total_steps": N,
        "total_reward": float(sum(rewards)),
        "avg_reward": float(sum(rewards) / N),
        "total_cost": float(sum(costs)),
        "ev_total_departures": total_departures,
        "ev_violated_departures": violated_departures,
        "ev_departure_violation_pct": ev_departure_violation_pct,
        "ev_avg_departure_deficit": avg_departure_deficit,
        "c1_total_cost": float(sum(c1_vals)),
        "c2_violation_steps": int(sum(1 for c in c2_vals if c > 0)),
        "c2_violation_pct": 100.0 * sum(1 for c in c2_vals if c > 0) / N,
        "c2_total_cost": float(sum(c2_vals)),
        "c3_violation_steps": int(sum(1 for c in c3_vals if c > 0)),
        "c3_violation_pct": 100.0 * sum(1 for c in c3_vals if c > 0) / N,
        "c3_total_cost": float(sum(c3_vals)),
        "c3_per_building_pct": per_building_c3_pct,
        "c3_per_building_structural_pct": per_building_structural_pct,
        "c3_per_building_controllable_pct": per_building_controllable_pct,
        "c3_total_structural": total_structural,
        "c3_total_controllable": total_controllable,
        "c3_total_violations": total_c3_violations,
        "c4_violation_steps": int(sum(1 for c in c4_vals if c > 0)),
        "c4_violation_pct": 100.0 * sum(1 for c in c4_vals if c > 0) / N,
        "c4_total_cost": float(sum(c4_vals)),
        "c0_total_cost": float(sum(c0_vals)),
        "c0_violation_steps": int(sum(1 for c in c0_vals if c > 0)),
        "batt_mean_action": float(mean_batt.mean()),
        "ev_mean_action": float(mean_ev.mean()),
        "batt_std_action": float(mean_batt.std()),
        "ev_std_action": float(mean_ev.std()),
        "intel_scores": intel_scores,
        "intel_overall": intel_overall,
        "corr_batt_price": corr_batt_price,
        "corr_batt_load": corr_batt_load,
        "ev_solar_pref": ev_solar_pref,
        "batt_price_spread": batt_price_spread,
        "pre_peak_batt": pre_peak_batt,
        "peak_batt": peak_batt,
        "hourly_batt": hourly_batt_means,
        "hourly_ev": hourly_ev_means,
        "hourly_net_load": hourly_net_means,
        "hourly_r_sg": hourly_r_sg_means,
        "hourly_r_load_shift": hourly_r_ls_means,
        "citylearn_kpis": cl_kpis,
        "reward_comp_stats": reward_comp_stats,
        # Battery cycle analysis
        "charging_hours": charging_hours,
        "discharging_hours": discharging_hours,
        "daily_charge_action_sum": daily_charge,
        "daily_discharge_action_sum": daily_discharge,
        # Price arbitrage
        "avg_batt_cheapest_4h": avg_batt_cheapest,
        "avg_batt_expensive_4h": avg_batt_expensive,
        "arbitrage_revenue_proxy": arbitrage_revenue,
        "corr_batt_price_actual": corr_batt_price,
    }


def print_report(all_results):
    report = []

    def p(line=""):
        print(line)
        report.append(line)

    runs = list(all_results.keys())

    p()
    p("=" * 120)
    p("  R17 THRESHOLD r_sg -- COMPREHENSIVE EVALUATION")
    p("  Environment: 5-building CityLearn V2G, seed=42, deterministic")
    p("  Comparison: " + " vs ".join(runs))
    p("  R17 key changes: threshold r_sg (50% P_grid_max), r_load_shift=6.0, r_ren=0.2, C3_CONTROLLABLE")
    p("=" * 120)

    def table_row(metric, key, fmt=".1f", extract=None):
        vals = []
        for r in runs:
            if extract:
                v = extract(all_results[r])
            else:
                v = all_results[r].get(key, float('nan'))
            vals.append(v)
        cols = f"  {metric:<45s}"
        for v in vals:
            if isinstance(v, float) and np.isnan(v):
                cols += f"{'N/A':>16s}"
            else:
                cols += f"{v:>16{fmt}}"
        # Add delta column if exactly 2 runs
        if len(runs) == 2:
            v0, v1 = vals[0], vals[1]
            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                if not (isinstance(v0, float) and np.isnan(v0)) and not (isinstance(v1, float) and np.isnan(v1)):
                    delta = v1 - v0
                    cols += f"  {delta:>+12{fmt}}"
                else:
                    cols += f"{'':>14s}"
        p(cols)

    header = f"  {'Metric':<45s}" + "".join(f"{r:>16s}" for r in runs)
    if len(runs) == 2:
        header += f"{'Delta':>14s}"

    # =====================================================================
    # SECTION 1: Standard metrics (from eval_r16)
    # =====================================================================
    p()
    p(header)
    p("  " + "-" * (len(header) - 2))

    p()
    p("  === REWARD ===")
    table_row("Total Reward", "total_reward", ".0f")
    table_row("Avg Reward/Step", "avg_reward", ".3f")

    p()
    p("  === EV DEPARTURE (C0) -- violations / total departures ===")
    table_row("Total Departures", "ev_total_departures", ".0f")
    table_row("Violated Departures", "ev_violated_departures", ".0f")
    table_row("Violation Rate (%)", "ev_departure_violation_pct", ".1f")
    table_row("Avg Deficit at Departure", "ev_avg_departure_deficit", ".4f")

    p()
    p("  === BATTERY SoC (C2) -- violations / timesteps ===")
    table_row("Violation Steps", "c2_violation_steps", ".0f")
    table_row("Violation Rate (%)", "c2_violation_pct", ".1f")
    table_row("Total Cost", "c2_total_cost", ".1f")

    p()
    p("  === BUILDING POWER (C3) -- violations / timesteps ===")
    table_row("Violation Steps (cost-based)", "c3_violation_steps", ".0f")
    table_row("Violation Rate (%)", "c3_violation_pct", ".1f")
    table_row("Total Cost", "c3_total_cost", ".1f")
    n_bld = len(all_results[runs[0]].get("c3_per_building_pct", []))
    for b in range(n_bld):
        table_row(f"  Building {b} Violation (%)", None, ".1f",
                  extract=lambda r, b=b: r.get("c3_per_building_pct", [0]*5)[b])

    p()
    p("  === GRID POWER (C4) -- violations / timesteps ===")
    table_row("Violation Steps", "c4_violation_steps", ".0f")
    table_row("Violation Rate (%)", "c4_violation_pct", ".1f")
    table_row("Total Cost", "c4_total_cost", ".1f")

    p()
    p("  === EV DENSE CHARGING (C0 dense) ===")
    table_row("Violation Steps", "c0_violation_steps", ".0f")
    table_row("Total Cost", "c0_total_cost", ".1f")

    p()
    p("  === SAUTE C1 (budget-handled) ===")
    table_row("Total Cost", "c1_total_cost", ".1f")

    p()
    p("  === CITYLEARN STANDARD KPIs (1.0 = no-op baseline) ===")
    cl_keys = [
        ("electricity_consumption_total", "Electricity Consumption"),
        ("carbon_emissions_total", "Carbon Emissions"),
        ("cost_total", "Electricity Cost"),
        ("daily_peak_average", "Daily Peak Average"),
        ("all_time_peak_average", "All-Time Peak"),
        ("ramping_average", "Ramping Average"),
        ("1 - Loss of Life Share", "1 - Loss of Life"),
        ("zero_net_energy", "Zero Net Energy"),
    ]
    for cl_key, cl_label in cl_keys:
        table_row(cl_label, None, ".4f",
                  extract=lambda r, k=cl_key: r.get("citylearn_kpis", {}).get(k, float('nan')))

    p()
    p("  === ACTION STATISTICS ===")
    table_row("Battery Mean Action", "batt_mean_action", ".4f")
    table_row("EV Mean Action", "ev_mean_action", ".4f")
    table_row("Battery Std Action", "batt_std_action", ".4f")
    table_row("EV Std Action", "ev_std_action", ".4f")

    p()
    p("  === INTELLIGENCE METRICS ===")
    table_row("Solar Preference (EV)", "ev_solar_pref", ".4f")
    table_row("Price Spread (Battery)", "batt_price_spread", ".4f")
    table_row("Corr(Battery, Price)", "corr_batt_price", ".4f")
    table_row("Corr(Battery, Load)", "corr_batt_load", ".4f")
    table_row("Pre-Peak Battery Action", "pre_peak_batt", ".4f")
    table_row("Peak Battery Action", "peak_batt", ".4f")

    p()
    p("  === INTELLIGENCE SCORECARD ===")
    intel_keys = list(all_results[runs[0]].get("intel_scores", {}).keys())
    for ik in intel_keys:
        table_row(f"  {ik}", None, ".3f",
                  extract=lambda r, k=ik: r.get("intel_scores", {}).get(k, 0.0))
    table_row("OVERALL INTELLIGENCE", "intel_overall", ".3f")

    # =====================================================================
    # SECTION 2: Reward component breakdown
    # =====================================================================
    p()
    p("=" * 120)
    p("  === REWARD COMPONENT BREAKDOWN ===")
    p("  (Computed with R17 reward weights: r_sg=1.5 threshold=0.5, r_load_shift=6.0, r_ren=0.2)")
    p()

    comp_header = f"  {'Component':<18s}"
    for r in runs:
        comp_header += f"  {'mean':>8s} {'std':>7s} {'min':>8s} {'max':>8s} {'var%':>6s}"
        comp_header += "  |"
    p(f"  {'':>18s}  " + "  ".join(f"{r:^42s}" for r in runs))
    p(comp_header)
    p("  " + "-" * (len(comp_header) - 2))

    for k in REWARD_COMPONENT_KEYS:
        cols = f"  {k:<18s}"
        for r in runs:
            stats = all_results[r].get("reward_comp_stats", {}).get(k, {})
            if stats:
                cols += f"  {stats['mean']:>+8.4f} {stats['std']:>7.4f} {stats['min']:>+8.4f} {stats['max']:>+8.4f} {stats['var_pct']:>5.1f}%"
            else:
                cols += f"  {'N/A':>8s} {'N/A':>7s} {'N/A':>8s} {'N/A':>8s} {'N/A':>6s}"
            cols += "  |"
        p(cols)

    # =====================================================================
    # SECTION 3: C3 structural vs controllable breakdown
    # =====================================================================
    p()
    p("=" * 120)
    p("  === C3 STRUCTURAL vs CONTROLLABLE BREAKDOWN ===")
    p("  Structural: base load (NSL + solar) already exceeds P_building_max=4.6083")
    p("  Controllable: agent's battery/EV actions caused the violation")
    p()

    c3_header = f"  {'Metric':<45s}" + "".join(f"{r:>16s}" for r in runs)
    if len(runs) == 2:
        c3_header += f"{'Delta':>14s}"
    p(c3_header)
    p("  " + "-" * (len(c3_header) - 2))

    table_row("Total C3 violations (steps)", "c3_total_violations", ".0f")
    table_row("  Structural (solar export)", "c3_total_structural", ".0f")
    table_row("  Controllable (agent action)", "c3_total_controllable", ".0f")

    table_row("Structural fraction (%)", None, ".1f",
              extract=lambda r: 100.0 * r.get("c3_total_structural", 0) /
              max(1, r.get("c3_total_violations", 1)))
    table_row("Controllable fraction (%)", None, ".1f",
              extract=lambda r: 100.0 * r.get("c3_total_controllable", 0) /
              max(1, r.get("c3_total_violations", 1)))

    p()
    p("  Per-building breakdown (violation rate %):")
    p(f"  {'Building':<12s}" + "".join(
        f"  {'Raw':>6s} {'Struct':>6s} {'Ctrl':>6s}" for r in runs))
    p("  " + "-" * (12 + len(runs) * 22))
    for b in range(n_bld):
        cols = f"  Bld {b:<7d}"
        for r in runs:
            raw_pct = all_results[r].get("c3_per_building_pct", [0]*5)[b]
            struct_pct = all_results[r].get("c3_per_building_structural_pct", [0]*5)[b]
            ctrl_pct = all_results[r].get("c3_per_building_controllable_pct", [0]*5)[b]
            cols += f"  {raw_pct:>5.1f}% {struct_pct:>5.1f}% {ctrl_pct:>5.1f}%"
        p(cols)

    # =====================================================================
    # SECTION 4: Battery charging cycle analysis
    # =====================================================================
    p()
    p("=" * 120)
    p("  === BATTERY CHARGING CYCLE ANALYSIS ===")
    p("  Expected R17 behavior: afternoon charging (hours 14-17), evening/night discharge")
    p()

    cycle_header = f"  {'Metric':<45s}" + "".join(f"{r:>16s}" for r in runs)
    if len(runs) == 2:
        cycle_header += f"{'Delta':>14s}"
    p(cycle_header)
    p("  " + "-" * (len(cycle_header) - 2))

    table_row("Charging hours (mean action > 0)", None, "s",
              extract=lambda r: ",".join(f"{h:02d}" for h in r.get("charging_hours", [])))
    table_row("Discharging hours (mean action < 0)", None, "s",
              extract=lambda r: ",".join(f"{h:02d}" for h in r.get("discharging_hours", [])))
    table_row("Daily charge (action sum)", "daily_charge_action_sum", ".3f")
    table_row("Daily discharge (action sum)", "daily_discharge_action_sum", ".3f")
    table_row("Daily throughput (charge + discharge)", None, ".3f",
              extract=lambda r: r.get("daily_charge_action_sum", 0) + r.get("daily_discharge_action_sum", 0))

    p()
    p("  Hourly battery action profile:")
    profile_header = f"  {'Hour':<6s}" + "".join(f"  {r:>12s}" for r in runs) + "  {'Price':>8s}  {'Solar':>8s}"
    # Fix the f-string formatting
    profile_header = f"  {'Hour':<6s}"
    for r in runs:
        profile_header += f"  {r:>12s}"
    profile_header += f"  {'Price':>8s}  {'Solar':>8s}"
    p(profile_header)
    p("  " + "-" * (len(profile_header) - 2))
    for h in range(24):
        cols = f"  {h:02d}:00 "
        for r in runs:
            v = all_results[r].get("hourly_batt", [0]*24)[h]
            # Mark charging (+) vs discharging (-) clearly
            marker = "+" if v > 0.001 else ("-" if v < -0.001 else " ")
            cols += f"  {v:>+11.4f}{marker}"
        # Add price and solar context
        hp = all_results[runs[0]].get("hourly_net_load", [0]*24)
        price_h = 0.0
        solar_h = 0.0
        try:
            price_h = np.mean([all_results[r].get("hourly_batt", [0]*24) for r in runs])
        except Exception:
            pass
        p(cols)

    # =====================================================================
    # SECTION 5: Price arbitrage verification
    # =====================================================================
    p()
    p("=" * 120)
    p("  === PRICE ARBITRAGE VERIFICATION ===")
    p("  Tests whether R17's r_load_shift=6.0 drives price-aware battery behavior")
    p()

    arb_header = f"  {'Metric':<45s}" + "".join(f"{r:>16s}" for r in runs)
    if len(runs) == 2:
        arb_header += f"{'Delta':>14s}"
    p(arb_header)
    p("  " + "-" * (len(arb_header) - 2))

    table_row("Corr(Battery, Price)", "corr_batt_price_actual", ".4f")
    table_row("Avg batt during cheapest 4h", "avg_batt_cheapest_4h", ".4f")
    table_row("Avg batt during most expensive 4h", "avg_batt_expensive_4h", ".4f")
    table_row("Price arbitrage spread (cheap - exp)", None, ".4f",
              extract=lambda r: r.get("avg_batt_cheapest_4h", 0) - r.get("avg_batt_expensive_4h", 0))
    table_row("Arbitrage revenue proxy (sum)", "arbitrage_revenue_proxy", ".1f")

    p()
    p("  Interpretation:")
    p("  - Negative corr(batt,price) = good: charging when cheap, discharging when expensive")
    p("  - Positive spread (cheap - expensive) = good: net charging during cheap, net discharge during expensive")
    p("  - Higher arbitrage revenue = better price-aware behavior")

    # =====================================================================
    # SECTION 6: Hourly reward component profiles (r_sg and r_load_shift)
    # =====================================================================
    p()
    p("=" * 120)
    p("  === HOURLY REWARD COMPONENT PROFILES ===")
    p("  r_sg: Grid stability (R17 threshold at 50% P_grid_max)")
    p("  r_load_shift: Battery price arbitrage")
    p()

    rc_header = f"  {'Hour':<6s}"
    for r in runs:
        rc_header += f"  {'r_sg':>8s}  {'r_ls':>8s}"
    p(rc_header)
    p("  " + "-" * (len(rc_header) - 2))
    for h in range(24):
        cols = f"  {h:02d}:00 "
        for r in runs:
            sg_v = all_results[r].get("hourly_r_sg", [0]*24)[h]
            ls_v = all_results[r].get("hourly_r_load_shift", [0]*24)[h]
            cols += f"  {sg_v:>+8.4f}  {ls_v:>+8.4f}"
        p(cols)

    # =====================================================================
    # SECTION 7: Hourly EV action profile
    # =====================================================================
    p()
    p("=" * 120)
    p("  === HOURLY EV ACTION PROFILE ===")
    ev_header = f"  {'Hour':<6s}" + "".join(f"  {r:>12s}" for r in runs)
    p(ev_header)
    p("  " + "-" * (len(ev_header) - 2))
    for h in range(24):
        cols = f"  {h:02d}:00 "
        for r in runs:
            v = all_results[r].get("hourly_ev", [0]*24)[h]
            cols += f"  {v:>+12.4f}"
        p(cols)

    # =====================================================================
    # SECTION 8: Ablation deltas summary
    # =====================================================================
    p()
    p("=" * 120)
    p("  === ABLATION DELTAS (R15b -> R17) ===")
    p()
    if len(runs) == 2:
        delta_metrics = [
            ("Total Reward", "total_reward", ".0f"),
            ("Avg Reward/Step", "avg_reward", ".4f"),
            ("EV Departure Viol %", "ev_departure_violation_pct", ".1f"),
            ("C2 Viol %", "c2_violation_pct", ".1f"),
            ("C3 Viol %", "c3_violation_pct", ".1f"),
            ("C3 Controllable Violations", "c3_total_controllable", ".0f"),
            ("C4 Viol %", "c4_violation_pct", ".1f"),
            ("Intelligence", "intel_overall", ".3f"),
            ("Price Arbitrage Revenue", "arbitrage_revenue_proxy", ".1f"),
            ("Battery Price Spread", "batt_price_spread", ".4f"),
            ("Daily Charge Sum", "daily_charge_action_sum", ".3f"),
            ("Daily Discharge Sum", "daily_discharge_action_sum", ".3f"),
        ]
        d_header = f"  {'Metric':<40s}  {'R15b':>14s}  {'R17':>14s}  {'Delta':>14s}  {'Change':>10s}"
        p(d_header)
        p("  " + "-" * (len(d_header) - 2))
        for name, key, fmt in delta_metrics:
            v0 = all_results[runs[0]].get(key, 0.0)
            v1 = all_results[runs[1]].get(key, 0.0)
            delta = v1 - v0
            pct_change = ""
            if isinstance(v0, (int, float)) and abs(v0) > 1e-6:
                pct_change = f"{100.0 * delta / abs(v0):>+.1f}%"
            cols = f"  {name:<40s}  {v0:>14{fmt}}  {v1:>14{fmt}}  {delta:>+14{fmt}}  {pct_change:>10s}"
            p(cols)

    # =====================================================================
    # SECTION 9: Verdict
    # =====================================================================
    p()
    p("=" * 120)
    p("  === VERDICT ===")
    p()
    if len(runs) == 2:
        r0 = all_results[runs[0]]
        r1 = all_results[runs[1]]
        wins = 0
        losses = 0
        checks = [
            ("Total Reward", r1["total_reward"] > r0["total_reward"], "higher"),
            ("EV Departure Violations", r1["ev_departure_violation_pct"] <= r0["ev_departure_violation_pct"], "lower/equal"),
            ("C3 Controllable Violations", r1["c3_total_controllable"] <= r0["c3_total_controllable"], "lower/equal"),
            ("C4 Violations", r1["c4_violation_pct"] <= r0["c4_violation_pct"], "lower/equal"),
            ("Intelligence Score", r1["intel_overall"] > r0["intel_overall"], "higher"),
            ("Price Arbitrage Revenue", r1["arbitrage_revenue_proxy"] > r0["arbitrage_revenue_proxy"], "higher"),
            ("Battery Price Spread", r1["batt_price_spread"] > r0["batt_price_spread"], "higher"),
        ]
        for name, passed, direction in checks:
            status = "PASS" if passed else "FAIL"
            if passed:
                wins += 1
            else:
                losses += 1
            p(f"  [{status}] {name} (want {direction})")

        p()
        p(f"  Score: {wins}/{wins+losses} checks passed")
        if wins > losses:
            p(f"  --> R17 is an IMPROVEMENT over R15b")
        elif wins == losses:
            p(f"  --> R17 is MIXED compared to R15b")
        else:
            p(f"  --> R17 is a REGRESSION from R15b")

    p()
    p("=" * 120)
    return "\n".join(report)


def main():
    OUT_DIR = os.path.join(PROJECT, "runs", "r17_evaluation")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n=== R17 Evaluation ===\n")

    agents = {}
    for name, run_dir in RUN_DIRS.items():
        ckpt = find_checkpoint(run_dir, epoch=50)
        if not ckpt:
            ckpt = find_checkpoint(run_dir)
        if ckpt:
            agents[name] = ckpt
            print(f"  {name}: {ckpt}")
        else:
            print(f"  {name}: NOT FOUND (skipping)")

    if not agents:
        print("ERROR: No checkpoints found!")
        return

    all_results = {}
    for name, ckpt in agents.items():
        print(f"\n{'='*60}")
        print(f"  EVALUATING: {name}")
        print(f"  Checkpoint: {ckpt}")
        print(f"{'='*60}")
        actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(ckpt)
        print(f"  Loaded: obs_dim={obs_dim}, act_dim={act_dim}")
        results = run_eval_episode(actor, obs_mean, obs_std, obs_clip, obs_dim, name)
        all_results[name] = results
        print(f"  EV Departure: {results['ev_violated_departures']}/{results['ev_total_departures']} "
              f"({results['ev_departure_violation_pct']:.1f}%)")
        print(f"  C3: {results['c3_violation_pct']:.1f}% (struct={results['c3_total_structural']}, "
              f"ctrl={results['c3_total_controllable']})")
        print(f"  C4: {results['c4_violation_pct']:.1f}%")
        print(f"  Intel: {results['intel_overall']:.3f}")
        print(f"  Arbitrage revenue: {results['arbitrage_revenue_proxy']:.1f}")

    report_text = print_report(all_results)

    report_path = os.path.join(OUT_DIR, "r17_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport: {report_path}")

    json_path = os.path.join(OUT_DIR, "r17_results.json")
    json_results = {}
    for name, r in all_results.items():
        jr = dict(r)
        # Remove large arrays from JSON for readability
        for key_to_remove in ["hourly_batt", "hourly_ev", "hourly_net_load",
                               "hourly_r_sg", "hourly_r_load_shift"]:
            jr.pop(key_to_remove, None)
        json_results[name] = jr
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"JSON:   {json_path}")


if __name__ == "__main__":
    main()
