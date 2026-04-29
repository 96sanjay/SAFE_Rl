#!/usr/bin/env python3
"""
Unified KPI evaluation for R21, R23, R30c + no-control + greedy RBC baselines.
Reports: CityLearn KPIs, departure-based C0 violations, per-building battery stats,
per-charger EV V2G stats.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# =====================================================================
# Environment variable configs per run
# =====================================================================
# Common base env vars (shared across all runs at eval time)
BASE_ENV = {
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
    "CITYLEARN_C3_CONTROLLABLE": "1",
    # DISABLE training-only wrappers
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_PID_LAGRANGE": "0",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_ACTION_MASK": "0",
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
}

# R21 reward env vars
R21_REWARD = {
    "STEMS_ALPHA_GRID": "0.0",
    "STEMS_SG_THRESHOLD": "0.5",
    "STEMS_ALPHA_LOAD_SHIFT": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.3",
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_BUILD": "0.0",
    "STEMS_XI_RENEWABLE": "0.2",
    "STEMS_BETA_RAMP": "0.3",
    "STEMS_LAMBDA_EV": "5.0",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "STEMS_ALPHA_EV_GUARD": "1.0",
    "STEMS_ALPHA_V2G_CONTEXT": "3.0",
    "CITYLEARN_BATT_CLAMP": "1",
}

R23_REWARD = {**R21_REWARD, "STEMS_ALPHA_EV_SOLAR": "3.0"}

R30_REWARD = {
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
    "CITYLEARN_BATT_CLAMP": "0",
}

RUNS = {
    "R21": {
        "ckpt": "runs/r21_ppo/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-13-04-07-53/torch_save/epoch-89.pt",
        "reward_env": R21_REWARD,
        "batt_clamp": True,
    },
    "R23": {
        "ckpt": "runs/r23_ppo/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-15-04-32-16/torch_save/epoch-89.pt",
        "reward_env": R23_REWARD,
        "batt_clamp": True,
    },
    "R30c": {
        "ckpt": "runs/r30c_kl_nec/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-03-19-18-08-21/torch_save/epoch-50.pt",
        "reward_env": R30_REWARD,
        "batt_clamp": False,
    },
}

P_BUILDING_MAX = 4.6083
P_GRID_MAX = 10.2352
SOC_UPPER_CLAMP = 0.94
BATT_DT = 1.0


def set_env(reward_env=None):
    """Clear all CITYLEARN_/STEMS_/COST_ env vars, then set base + reward."""
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(k, None)
    for k, v in BASE_ENV.items():
        os.environ[k] = v
    if reward_env:
        for k, v in reward_env.items():
            os.environ[k] = v


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
    ckpt = torch.load(os.path.join(PROJECT, ckpt_path), map_location="cpu", weights_only=False)
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


def make_env():
    """Create eval env. Must call set_env() first."""
    import importlib
    import citylearn_safe.schema_index as si
    si._CACHE = None
    # Force reimport to pick up new env vars
    import citylearn_safe.safety_env_v3
    importlib.reload(citylearn_safe.safety_env_v3)
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)
    raw = unwrap_to_raw_citylearn_env(env)
    return env, raw


def discover_battery_actions(raw):
    buildings = list(getattr(raw, 'buildings', []))
    names_raw = getattr(raw, 'action_names', [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        flat_names = names_raw[0]
    else:
        flat_names = list(names_raw)
    batt_pos = [i for i, n in enumerate(flat_names) if str(n).lower() == 'electrical_storage']
    result = []
    for b_idx, b in enumerate(buildings):
        es = getattr(b, 'electrical_storage', None)
        if es is None:
            continue
        if b_idx < len(batt_pos):
            act_idx = batt_pos[b_idx]
        else:
            continue
        cap = float(getattr(es, 'capacity', 6.4) or 6.4)
        p_max = float(getattr(es, 'nominal_power', 5.0) or 5.0)
        eta = float(getattr(es, 'efficiency', 0.9) or 0.9)
        result.append((act_idx, b_idx, cap, p_max, eta))
    return result


def clamp_battery_actions(a, batt_action_map, raw):
    if not batt_action_map:
        return a
    buildings = list(getattr(raw, 'buildings', []))
    t_idx = max(0, int(getattr(raw, 'time_step', 0)) - 1)
    a_clamped = a.copy()
    for act_idx, bld_idx, cap, p_max, eta in batt_action_map:
        if act_idx >= len(a_clamped) or bld_idx >= len(buildings):
            continue
        es = getattr(buildings[bld_idx], 'electrical_storage', None)
        if es is None:
            continue
        soc_arr = getattr(es, 'soc', None)
        if soc_arr is None or not hasattr(soc_arr, '__len__') or len(soc_arr) <= t_idx:
            continue
        soc = float(np.clip(soc_arr[t_idx], 0.0, 1.0))
        denom_charge = p_max * BATT_DT * eta
        max_charge = (SOC_UPPER_CLAMP - soc) * cap / denom_charge if denom_charge > 0 else 1.0
        denom_discharge = p_max * BATT_DT
        max_discharge = soc * cap * eta / denom_discharge if denom_discharge > 0 else 1.0
        a_clamped[act_idx] = float(np.clip(a_clamped[act_idx], -max_discharge, max_charge))
    return a_clamped


def get_action_indices(raw):
    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)
    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]
    return names, batt_idx, ev_idx


def run_episode(env, raw, action_fn, model_obs_dim=None, batt_clamp=False, seed=42):
    """
    Run one full episode.
    action_fn(obs, env, raw) -> action array
    Returns dict of metrics.
    """
    buildings = list(getattr(raw, "buildings", []))
    n_buildings = len(buildings)
    names, batt_idx, ev_idx = get_action_indices(raw)
    act_dim = len(names)
    batt_action_map = discover_battery_actions(raw) if batt_clamp else []

    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    # Trackers
    rewards = []
    actions_all = []

    # EV departure tracking
    total_departures = 0
    violated_departures = 0
    departure_deficits = []
    ev_tracker = {}

    # Per-building battery charge/discharge energy
    batt_charge_energy = [0.0] * n_buildings  # kWh charged
    batt_discharge_energy = [0.0] * n_buildings  # kWh discharged

    # Per-charger EV V2G tracking
    # Map: charger global index -> {charge_steps, discharge_steps, idle_steps, connected_steps}
    charger_stats = defaultdict(lambda: {"charge": 0, "discharge": 0, "idle": 0, "connected": 0})

    # C3/C4 tracking
    c3_violation_steps = 0
    c4_violation_steps = 0
    building_power_violations = [0] * n_buildings

    # CityLearn KPIs
    citylearn_kpis = {}
    last_info = {}

    while not done:
        # Get action
        action = action_fn(obs, env, raw)
        action = np.clip(action, -1.0, 1.0)

        # Battery clamp if enabled
        if batt_clamp and batt_action_map:
            action = clamp_battery_actions(action, batt_action_map, raw)

        actions_all.append(action.copy())
        t_now = int(getattr(raw, "time_step", 0))
        t_idx = max(0, t_now - 1)

        # === EV departure tracking ===
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
                        charger_stats[key]["connected"] += 1
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

        # === Per-charger V2G action tracking ===
        charger_global_idx = 0
        for b_idx, bld in enumerate(buildings):
            chargers = getattr(bld, "electric_vehicle_chargers", None) or []
            for ch_idx, ch in enumerate(chargers):
                key = (b_idx, ch_idx)
                # Find the action index for this charger
                if charger_global_idx < len(ev_idx):
                    ai = ev_idx[charger_global_idx]
                    if ai < len(action):
                        a_val = action[ai]
                        # Only count when EV is connected
                        sim = getattr(ch, 'charger_simulation',
                                      getattr(ch, '_Charger__charger_simulation', None))
                        if sim is not None:
                            try:
                                sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                                connected = t_now < len(sa) and float(sa[t_now]) == 1.0
                                if connected:
                                    if a_val > 0.05:
                                        charger_stats[key]["charge"] += 1
                                    elif a_val < -0.05:
                                        charger_stats[key]["discharge"] += 1
                                    else:
                                        charger_stats[key]["idle"] += 1
                            except Exception:
                                pass
                charger_global_idx += 1

        # === Per-building battery action tracking ===
        for act_idx_b, bld_idx, cap, p_max, eta in (batt_action_map if batt_action_map else discover_battery_actions(raw)):
            if act_idx_b < len(action):
                a_val = action[act_idx_b]
                if a_val > 0:
                    # Charging: energy = action * p_max * dt (kWh)
                    batt_charge_energy[bld_idx] += a_val * p_max * BATT_DT
                elif a_val < 0:
                    batt_discharge_energy[bld_idx] += abs(a_val) * p_max * BATT_DT

        # === Step environment ===
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        rewards.append(reward)
        last_info = info

        # C3 violations
        bld_viol = info.get("building_power_violation", 0.0)
        if bld_viol > 0:
            c3_violation_steps += 1
        # Per-building
        for b_idx, bld in enumerate(buildings):
            try:
                nec = getattr(bld, 'net_electricity_consumption', None)
                if nec is not None and len(nec) > t_idx:
                    p = abs(float(nec[t_idx]))
                    if p > P_BUILDING_MAX:
                        building_power_violations[b_idx] += 1
            except Exception:
                pass

        # C4 violations
        grid_viol = info.get("grid_power_violation", 0.0)
        if grid_viol > 0:
            c4_violation_steps += 1

        # CityLearn KPIs on last step
        if done:
            for kpi_key in ["citylearn_electricity_consumption_total",
                            "citylearn_carbon_emissions_total",
                            "citylearn_cost_total",
                            "citylearn_daily_peak_average",
                            "citylearn_all_time_peak_average",
                            "citylearn_ramping_average",
                            "citylearn_discomfort_proportion",
                            "citylearn_zero_net_energy"]:
                citylearn_kpis[kpi_key] = info.get(kpi_key, float("nan"))

        if step % 2000 == 0:
            sys.stdout.write(f"  step {step}...\r")
            sys.stdout.flush()

    actions_arr = np.array(actions_all)

    # Per-building battery charge/discharge %
    batt_stats = []
    for b in range(n_buildings):
        total = batt_charge_energy[b] + batt_discharge_energy[b]
        batt_stats.append({
            "charge_kwh": batt_charge_energy[b],
            "discharge_kwh": batt_discharge_energy[b],
            "charge_pct": 100 * batt_charge_energy[b] / total if total > 0 else 50.0,
            "discharge_pct": 100 * batt_discharge_energy[b] / total if total > 0 else 50.0,
        })

    # Per-charger EV V2G %
    ev_charger_results = {}
    for key, stats in sorted(charger_stats.items()):
        b_idx, ch_idx = key
        total_connected = stats["connected"]
        ev_charger_results[f"B{b_idx}_CH{ch_idx}"] = {
            "connected_steps": total_connected,
            "charge_pct": 100 * stats["charge"] / total_connected if total_connected > 0 else 0,
            "discharge_pct": 100 * stats["discharge"] / total_connected if total_connected > 0 else 0,
            "idle_pct": 100 * stats["idle"] / total_connected if total_connected > 0 else 0,
        }

    # EV departure stats
    ev_viol_rate = 100.0 * violated_departures / total_departures if total_departures > 0 else 0
    violated_deficits = [d for d in departure_deficits if d > 0.01]

    return {
        "steps": step,
        "total_reward": sum(rewards),
        "mean_reward": sum(rewards) / step if step > 0 else 0,
        "ev_departures": total_departures,
        "ev_violated": violated_departures,
        "ev_violation_rate": ev_viol_rate,
        "ev_mean_deficit": float(np.mean(departure_deficits)) if departure_deficits else 0,
        "ev_mean_deficit_violated": float(np.mean(violated_deficits)) if violated_deficits else 0,
        "c3_violation_steps": c3_violation_steps,
        "c3_violation_rate": 100.0 * c3_violation_steps / step,
        "c4_violation_steps": c4_violation_steps,
        "c4_violation_rate": 100.0 * c4_violation_steps / step,
        "building_power_violations": building_power_violations,
        "battery_stats": batt_stats,
        "ev_charger_stats": ev_charger_results,
        "citylearn_kpis": citylearn_kpis,
    }


def run_model_eval(run_name, run_cfg, seed=42):
    """Evaluate a trained model checkpoint."""
    print(f"\n{'='*70}")
    print(f"  Evaluating: {run_name}")
    print(f"  Checkpoint: {run_cfg['ckpt']}")
    print(f"{'='*70}")

    # Load actor
    actor, obs_mean, obs_std, obs_clip, model_obs_dim, act_dim = load_actor(run_cfg["ckpt"])
    print(f"  Model: obs_dim={model_obs_dim}, act_dim={act_dim}")
    print(f"  Obs normalizer: {'YES' if obs_mean is not None else 'NO'}")

    # Set env vars
    set_env(run_cfg["reward_env"])

    # Create env
    env, raw = make_env()
    env_obs_dim = env.observation_space.shape[0]
    pad_dim = model_obs_dim - env_obs_dim
    print(f"  Env obs_dim={env_obs_dim}, pad_dim={pad_dim}")

    def action_fn(obs, env_inner, raw_inner):
        if pad_dim > 0:
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
        return action_t

    t0 = time.time()
    results = run_episode(env, raw, action_fn, model_obs_dim,
                          batt_clamp=run_cfg.get("batt_clamp", False), seed=seed)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    return results


def run_nocontrol(seed=42):
    """No-control baseline: action = 0 for all timesteps."""
    print(f"\n{'='*70}")
    print(f"  Evaluating: No-Control Baseline (action=0)")
    print(f"{'='*70}")

    set_env(R21_REWARD)  # reward doesn't matter for no-control, but env needs valid config
    os.environ["CITYLEARN_BATT_CLAMP"] = "0"
    env, raw = make_env()
    names, batt_idx, ev_idx = get_action_indices(raw)
    act_dim = len(names)

    def action_fn(obs, env_inner, raw_inner):
        return np.zeros(act_dim, dtype=np.float32)

    t0 = time.time()
    results = run_episode(env, raw, action_fn, batt_clamp=False, seed=seed)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    return results


def run_greedy_rbc(seed=42):
    """Greedy RBC: charge=+1 for batteries and EVs (CityLearn default RBC behavior)."""
    print(f"\n{'='*70}")
    print(f"  Evaluating: Greedy RBC Baseline (action=+1)")
    print(f"{'='*70}")

    set_env(R21_REWARD)
    os.environ["CITYLEARN_BATT_CLAMP"] = "0"
    env, raw = make_env()
    names, batt_idx, ev_idx = get_action_indices(raw)
    act_dim = len(names)

    def action_fn(obs, env_inner, raw_inner):
        # Greedy: charge everything maximally
        return np.ones(act_dim, dtype=np.float32)

    t0 = time.time()
    results = run_episode(env, raw, action_fn, batt_clamp=False, seed=seed)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    return results


def print_results_table(all_results):
    """Print comparison table."""
    print(f"\n\n{'='*120}")
    print(f"  COMPREHENSIVE COMPARISON TABLE")
    print(f"{'='*120}")

    names = list(all_results.keys())
    W = 25

    # CityLearn KPIs
    print(f"\n--- CityLearn KPIs (ratio vs no-control; lower = better for most) ---")
    kpi_keys = [
        "citylearn_electricity_consumption_total",
        "citylearn_carbon_emissions_total",
        "citylearn_cost_total",
        "citylearn_daily_peak_average",
        "citylearn_all_time_peak_average",
        "citylearn_ramping_average",
        "citylearn_zero_net_energy",
    ]
    kpi_short = {
        "citylearn_electricity_consumption_total": "Electricity",
        "citylearn_carbon_emissions_total": "Carbon",
        "citylearn_cost_total": "Cost",
        "citylearn_daily_peak_average": "DailyPeak",
        "citylearn_all_time_peak_average": "AllTimePeak",
        "citylearn_ramping_average": "Ramping",
        "citylearn_zero_net_energy": "ZeroNetEnergy",
    }

    header = f"  {'KPI':<20}"
    for n in names:
        header += f" {n:>{W}}"
    print(header)
    print(f"  {'-'*(20 + (W+1)*len(names))}")

    for kk in kpi_keys:
        short = kpi_short.get(kk, kk.split("_")[-1])
        row = f"  {short:<20}"
        for n in names:
            v = all_results[n].get("citylearn_kpis", {}).get(kk, float("nan"))
            if np.isnan(v):
                row += f" {'N/A':>{W}}"
            else:
                row += f" {v:>{W}.4f}"
        print(row)

    # Constraint violations
    print(f"\n--- Constraint Violations ---")
    header = f"  {'Metric':<30}"
    for n in names:
        header += f" {n:>{W}}"
    print(header)
    print(f"  {'-'*(30 + (W+1)*len(names))}")

    metrics = [
        ("C0: EV Departures (total)", "ev_departures"),
        ("C0: Violated Departures", "ev_violated"),
        ("C0: Violation Rate (%)", "ev_violation_rate"),
        ("C0: Mean Deficit (all)", "ev_mean_deficit"),
        ("C0: Mean Deficit (violated)", "ev_mean_deficit_violated"),
        ("C3: Viol Steps", "c3_violation_steps"),
        ("C3: Viol Rate (%)", "c3_violation_rate"),
        ("C4: Viol Steps", "c4_violation_steps"),
        ("C4: Viol Rate (%)", "c4_violation_rate"),
        ("Total Reward", "total_reward"),
    ]

    for label, key in metrics:
        row = f"  {label:<30}"
        for n in names:
            v = all_results[n].get(key, 0)
            if isinstance(v, float):
                row += f" {v:>{W}.4f}"
            else:
                row += f" {v:>{W}}"
        print(row)

    # Per-building battery stats
    print(f"\n--- Per-Building Battery Charge/Discharge (kWh and %) ---")
    n_bld = len(all_results[names[0]].get("battery_stats", []))
    for b in range(n_bld):
        print(f"\n  Building {b}:")
        header = f"    {'Metric':<20}"
        for n in names:
            header += f" {n:>{W}}"
        print(header)
        print(f"    {'-'*(20 + (W+1)*len(names))}")
        for metric, key in [("Charge kWh", "charge_kwh"), ("Discharge kWh", "discharge_kwh"),
                             ("Charge %", "charge_pct"), ("Discharge %", "discharge_pct")]:
            row = f"    {metric:<20}"
            for n in names:
                stats = all_results[n].get("battery_stats", [])
                v = stats[b][key] if b < len(stats) else 0
                row += f" {v:>{W}.1f}"
            print(row)

    # Per-charger EV V2G stats
    print(f"\n--- Per-Charger EV V2G (% of connected steps) ---")
    # Collect all charger keys
    all_charger_keys = set()
    for n in names:
        all_charger_keys.update(all_results[n].get("ev_charger_stats", {}).keys())
    all_charger_keys = sorted(all_charger_keys)

    for ck in all_charger_keys:
        print(f"\n  {ck}:")
        header = f"    {'Metric':<20}"
        for n in names:
            header += f" {n:>{W}}"
        print(header)
        print(f"    {'-'*(20 + (W+1)*len(names))}")
        for metric, key in [("Connected Steps", "connected_steps"),
                             ("Charge %", "charge_pct"),
                             ("V2G Discharge %", "discharge_pct"),
                             ("Idle %", "idle_pct")]:
            row = f"    {metric:<20}"
            for n in names:
                stats = all_results[n].get("ev_charger_stats", {}).get(ck, {})
                v = stats.get(key, 0)
                if isinstance(v, float):
                    row += f" {v:>{W}.1f}"
                else:
                    row += f" {v:>{W}}"
            print(row)

    # Per-building C3 violation breakdown
    print(f"\n--- Per-Building C3 Power Violations (steps > P_building_max={P_BUILDING_MAX}) ---")
    header = f"  {'Building':<12}"
    for n in names:
        header += f" {n:>{W}}"
    print(header)
    print(f"  {'-'*(12 + (W+1)*len(names))}")
    for b in range(n_bld):
        row = f"  {'Bld '+str(b):<12}"
        for n in names:
            v = all_results[n].get("building_power_violations", [])[b] if b < len(all_results[n].get("building_power_violations", [])) else 0
            row += f" {v:>{W}}"
        print(row)

    print(f"\n{'='*120}")


def main():
    all_results = {}

    # 1. No-control baseline
    all_results["NoControl"] = run_nocontrol(seed=42)

    # 2. Greedy RBC
    all_results["GreedyRBC"] = run_greedy_rbc(seed=42)

    # 3. R21
    all_results["R21"] = run_model_eval("R21", RUNS["R21"], seed=42)

    # 4. R23
    all_results["R23"] = run_model_eval("R23", RUNS["R23"], seed=42)

    # 5. R30c
    all_results["R30c"] = run_model_eval("R30c", RUNS["R30c"], seed=42)

    # Print comparison table
    print_results_table(all_results)

    # Save JSON
    out_path = os.path.join(PROJECT, "runs", "eval_r21_r23_r30c_comparison.json")
    # Convert to JSON-safe
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
