#!/usr/bin/env python3
"""
R16 Evaluation — Compare R15b (best previous) vs R16a vs R16b

Same metrics as R15 ablation eval, adapted for R16 runs.
"""
import os
import sys
import json
import glob as glob_mod
import random
import numpy as np
import torch
import torch.nn as nn

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

EVAL_SEED = 42

# Environment profile (eval mode — no Sauté, no PID, no clamp)
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
    "CITYLEARN_C3_CONTROLLABLE": "0",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    # Disable training-only wrappers
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_PID_LAGRANGE": "0",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    # STEMS weights (for reward computation)
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_GRID": "0.0",
    "STEMS_ALPHA_BUILD": "0.0",
    "STEMS_BETA_RAMP": "0.3",
    "STEMS_XI_RENEWABLE": "1.5",
    "STEMS_LAMBDA_EV": "5.0",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_ALPHA_LOAD_SHIFT": "3.0",
    "STEMS_ALPHA_GRID_MILD": "0.3",
    "STEMS_ALPHA_EV_GUARD": "0.0",
    "STEMS_ALPHA_V2G_CONTEXT": "0.0",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
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
    "R16a": "runs/r16a_reward_reform/5bld",
    "R16b": "runs/r16b_pid_c3ctrl/5bld",
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
    except Exception:
        p25, p50, p75 = 0.12, 0.16, 0.20

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

    hourly_batt = {h: [] for h in range(24)}
    hourly_ev = {h: [] for h in range(24)}
    hourly_net_load = {h: [] for h in range(24)}
    hourly_price = {h: [] for h in range(24)}
    hourly_solar = {h: [] for h in range(24)}

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
                    if bld_net > P_building_max:
                        building_power_violations[b_idx] += 1
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

    per_building_c3_pct = []
    for b_idx in range(n_buildings):
        if building_power_steps[b_idx] > 0:
            pct = 100.0 * building_power_violations[b_idx] / building_power_steps[b_idx]
        else:
            pct = 0.0
        per_building_c3_pct.append(pct)

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
        "citylearn_kpis": cl_kpis,
    }


def print_report(all_results):
    report = []

    def p(line=""):
        print(line)
        report.append(line)

    runs = list(all_results.keys())

    p()
    p("=" * 120)
    p("  R16 REWARD REFORM — COMPREHENSIVE EVALUATION")
    p("  Environment: 5-building CityLearn V2G, seed=42, deterministic")
    p("  Runs: " + " → ".join(runs))
    p("=" * 120)

    def table_row(metric, key, fmt=".1f", extract=None):
        vals = []
        for r in runs:
            if extract:
                v = extract(all_results[r])
            else:
                v = all_results[r].get(key, float('nan'))
            vals.append(v)
        cols = f"  {metric:<40s}"
        for v in vals:
            if isinstance(v, float) and np.isnan(v):
                cols += f"{'N/A':>14s}"
            else:
                cols += f"{v:>14{fmt}}"
        p(cols)

    header = f"  {'Metric':<40s}" + "".join(f"{r:>14s}" for r in runs)
    p()
    p(header)
    p("  " + "-" * (len(header) - 2))

    p()
    p("  === REWARD ===")
    table_row("Total Reward", "total_reward", ".0f")
    table_row("Avg Reward/Step", "avg_reward", ".3f")

    p()
    p("  === EV DEPARTURE (C0) — violations / total departures ===")
    table_row("Total Departures", "ev_total_departures", ".0f")
    table_row("Violated Departures", "ev_violated_departures", ".0f")
    table_row("Violation Rate (%)", "ev_departure_violation_pct", ".1f")
    table_row("Avg Deficit at Departure", "ev_avg_departure_deficit", ".4f")

    p()
    p("  === BATTERY SoC (C2) — violations / timesteps ===")
    table_row("Violation Steps", "c2_violation_steps", ".0f")
    table_row("Violation Rate (%)", "c2_violation_pct", ".1f")
    table_row("Total Cost", "c2_total_cost", ".1f")

    p()
    p("  === BUILDING POWER (C3) — violations / timesteps ===")
    table_row("Violation Steps", "c3_violation_steps", ".0f")
    table_row("Violation Rate (%)", "c3_violation_pct", ".1f")
    table_row("Total Cost", "c3_total_cost", ".1f")
    n_bld = len(all_results[runs[0]].get("c3_per_building_pct", []))
    for b in range(n_bld):
        table_row(f"  Building {b} Violation (%)", None, ".1f",
                  extract=lambda r, b=b: r.get("c3_per_building_pct", [0]*5)[b])

    p()
    p("  === GRID POWER (C4) — violations / timesteps ===")
    table_row("Violation Steps", "c4_violation_steps", ".0f")
    table_row("Violation Rate (%)", "c4_violation_pct", ".1f")
    table_row("Total Cost", "c4_total_cost", ".1f")

    p()
    p("  === EV DENSE CHARGING (C0 dense) ===")
    table_row("Violation Steps", "c0_violation_steps", ".0f")
    table_row("Total Cost", "c0_total_cost", ".1f")

    p()
    p("  === SAUTÉ C1 (budget-handled) ===")
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

    # Deltas
    p()
    p("  === ABLATION DELTAS ===")
    p()
    delta_metrics = [
        ("EV Departure Viol %", "ev_departure_violation_pct"),
        ("C2 Viol %", "c2_violation_pct"),
        ("C3 Viol %", "c3_violation_pct"),
        ("C4 Viol %", "c4_violation_pct"),
        ("Total Reward", "total_reward"),
        ("Intelligence", "intel_overall"),
    ]
    header = f"  {'Metric':<30s}"
    for i in range(1, len(runs)):
        header += f"  {runs[i-1]}→{runs[i]:>12s}"
    if len(runs) > 1:
        header += f"  {runs[0]}→{runs[-1]:>12s}"
    p(header)
    p("  " + "-" * (len(header) - 2))
    for name, key in delta_metrics:
        cols = f"  {name:<30s}"
        vals = [all_results[r].get(key, 0.0) for r in runs]
        for i in range(1, len(runs)):
            delta = vals[i] - vals[i-1]
            cols += f"  {delta:>+14.1f}"
        if len(runs) > 1:
            total_delta = vals[-1] - vals[0]
            cols += f"  {total_delta:>+14.1f}"
        p(cols)

    # Hourly profiles
    p()
    p("  === HOURLY BATTERY ACTION PROFILE ===")
    header = f"  {'Hour':<6s}" + "".join(f"  {r:>10s}" for r in runs)
    p(header)
    p("  " + "-" * (len(header) - 2))
    for h in range(24):
        cols = f"  {h:02d}:00 "
        for r in runs:
            v = all_results[r].get("hourly_batt", [0]*24)[h]
            cols += f"  {v:>+10.3f}"
        p(cols)

    p()
    p("  === HOURLY EV ACTION PROFILE ===")
    header = f"  {'Hour':<6s}" + "".join(f"  {r:>10s}" for r in runs)
    p(header)
    p("  " + "-" * (len(header) - 2))
    for h in range(24):
        cols = f"  {h:02d}:00 "
        for r in runs:
            v = all_results[r].get("hourly_ev", [0]*24)[h]
            cols += f"  {v:>+10.3f}"
        p(cols)

    p()
    p("=" * 120)
    return "\n".join(report)


def main():
    OUT_DIR = os.path.join(PROJECT, "runs", "r16_evaluation")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n=== R16 Evaluation ===\n")

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
        print(f"  C3: {results['c3_violation_pct']:.1f}% | C4: {results['c4_violation_pct']:.1f}%")
        print(f"  Intel: {results['intel_overall']:.3f}")

    report_text = print_report(all_results)

    report_path = os.path.join(OUT_DIR, "r16_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport: {report_path}")

    json_path = os.path.join(OUT_DIR, "r16_results.json")
    json_results = {}
    for name, r in all_results.items():
        jr = dict(r)
        jr.pop("hourly_batt", None)
        jr.pop("hourly_ev", None)
        jr.pop("hourly_net_load", None)
        json_results[name] = jr
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"JSON:   {json_path}")


if __name__ == "__main__":
    main()
