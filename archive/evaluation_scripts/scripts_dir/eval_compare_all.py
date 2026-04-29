#!/usr/bin/env python3
"""
Unified Evaluation & Comparison Script
=======================================
Evaluates individual runs and produces a unified comparison table.

Usage:
    # Evaluate a single run (saves JSON)
    python scripts/eval_compare_all.py --run r21 --epoch 89 --seed 42
    python scripts/eval_compare_all.py --run no_control
    python scripts/eval_compare_all.py --run rbc

    # Print comparison table from all saved JSONs
    python scripts/eval_compare_all.py --run summary
"""
import os
import sys
import json
import glob as glob_mod
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from scipy import stats as scipy_stats

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# =========================================================================
# Run configurations
# =========================================================================
COMMON_ENV = {
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
    # Disable training wrappers
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_PID_LAGRANGE": "0",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_WM_DISABLE": "1",
    # Reward weights (common to R19/R21/R22/R23)
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
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
}

RUNS = {
    "r19": {
        "run_dir": "runs/r19_ablation/5bld",
        "overrides": {"CITYLEARN_BATT_CLAMP": "0"},
        "description": "R19: Ablation baseline (no safety projection)",
    },
    "r21": {
        "run_dir": "runs/r21_ppo/5bld",
        "overrides": {"CITYLEARN_BATT_CLAMP": "1"},
        "description": "R21: Best PPO (Safety Projection)",
    },
    "r22": {
        "run_dir": "runs/r22_ppo/5bld",
        "overrides": {"CITYLEARN_BATT_CLAMP": "1"},
        "description": "R22: GradS + Lambda Cap 12",
    },
    "r23": {
        "run_dir": "runs/r23_ppo/5bld",
        "overrides": {"CITYLEARN_BATT_CLAMP": "1", "STEMS_ALPHA_EV_SOLAR": "3.0"},
        "description": "R23: Solar-Aligned EV Charging",
    },
    "no_control": {
        "run_dir": None,
        "overrides": {"CITYLEARN_BATT_CLAMP": "0"},
        "description": "No control (zero actions)",
    },
    "rbc": {
        "run_dir": None,
        "overrides": {"CITYLEARN_BATT_CLAMP": "0"},
        "description": "IntelligentRBC (greedy EV charging)",
    },
}

COST_LIMITS = {
    "C0_ev_departure": 1800,
    "C1_ev_saute": 1500,
    "C2_battery_soc": 4000,
    "C3_building_power": 3000,
    "C4_grid_power": 1500,
}

P_BUILDING_MAX = 4.6083
P_GRID_MAX = 10.2352
SOC_UPPER_CLAMP = 0.94
BATT_DT = 1.0
WM_ACTION_INDEX = 2  # washing_machine_1 is at index 2


# =========================================================================
# Helpers
# =========================================================================
def divz(a, b, default=0.0):
    return a / b if b > 0 else default


def pct(a, b):
    return 100.0 * divz(a, b)


def set_env(run_name):
    """Clear all CityLearn env vars and set the ones for this run."""
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(k, None)
    for k, v in COMMON_ENV.items():
        os.environ[k] = v
    run_cfg = RUNS[run_name]
    for k, v in run_cfg.get("overrides", {}).items():
        os.environ[k] = v


def find_checkpoint(run_dir, epoch=None, seed=None):
    """Discover the checkpoint path for an RL run."""
    seed_pattern = f"seed-{seed:03d}*" if seed is not None else "seed-*"
    pattern = os.path.join(PROJECT, run_dir, f"PPOLagMulti-*", seed_pattern, "torch_save")
    save_dirs = sorted(glob_mod.glob(pattern))
    if not save_dirs:
        # Try without PPOLagMulti prefix
        pattern2 = os.path.join(PROJECT, run_dir, f"PPOLag-*", seed_pattern, "torch_save")
        save_dirs = sorted(glob_mod.glob(pattern2))
    if not save_dirs:
        return ""
    save_dir = save_dirs[-1]
    if epoch is not None:
        ckpt = os.path.join(save_dir, f"epoch-{epoch}.pt")
        if os.path.exists(ckpt):
            return ckpt
    # Find latest epoch
    ckpts = sorted(glob_mod.glob(os.path.join(save_dir, "epoch-*.pt")))
    return ckpts[-1] if ckpts else ""


# =========================================================================
# MLP Actor (matches OmniSafe PPOLag architecture)
# =========================================================================
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
    """Load actor weights and obs normalizer from a checkpoint."""
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


# =========================================================================
# Battery safety projection (Dalal et al. 2018)
# =========================================================================
def discover_battery_actions(raw_env):
    """Identify battery action indices and their physical parameters."""
    buildings = list(getattr(raw_env, 'buildings', []))
    names_raw = getattr(raw_env, 'action_names', [])
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


def clamp_battery_actions(a, batt_action_map, raw_env):
    """Safety Projection Layer (Dalal et al. 2018) for battery SoC."""
    if not batt_action_map:
        return a
    buildings = list(getattr(raw_env, 'buildings', []))
    t_idx = max(0, int(getattr(raw_env, 'time_step', 0)) - 1)
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


# =========================================================================
# Environment factory
# =========================================================================
def make_env(run_name):
    """Create environment with correct env vars for the given run."""
    import citylearn_safe.schema_index as si
    si._CACHE = None
    set_env(run_name)

    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)
    raw = unwrap_to_raw_citylearn_env(env)
    return env, raw


# =========================================================================
# Core evaluation loop
# =========================================================================
def evaluate(run_name, epoch=None, seed=42):
    """Evaluate a single run and return a results dict."""
    run_cfg = RUNS[run_name]
    description = run_cfg["description"]
    is_rl = run_name not in ("no_control", "rbc")
    use_batt_clamp = run_cfg.get("overrides", {}).get("CITYLEARN_BATT_CLAMP", "0") == "1"

    print(f"\n{'='*70}")
    print(f"  Evaluating: {description}")
    print(f"  Mode: {'RL checkpoint' if is_rl else run_name}")
    print(f"  Seed: {seed}")
    print(f"{'='*70}")

    # --- Load model (for RL runs) ---
    actor = obs_mean = obs_std = obs_clip = None
    obs_dim_model = act_dim_model = 0
    ckpt_path = ""

    if is_rl:
        ckpt_path = find_checkpoint(run_cfg["run_dir"], epoch=epoch, seed=seed)
        if not ckpt_path:
            print(f"  ERROR: No checkpoint found for {run_name}!")
            print(f"  Searched: {os.path.join(PROJECT, run_cfg['run_dir'], 'PPOLagMulti-*', 'seed-*', 'torch_save')}")
            return None
        print(f"  Checkpoint: {ckpt_path}")
        actor, obs_mean, obs_std, obs_clip, obs_dim_model, act_dim_model = load_actor(ckpt_path)
        print(f"  Actor: obs_dim={obs_dim_model}, act_dim={act_dim_model}, hidden=[256,256]")
        print(f"  Obs normalizer: {'YES' if obs_mean is not None else 'NO'}")

    # --- Create environment ---
    env, raw = make_env(run_name)
    buildings = list(getattr(raw, "buildings", []))
    n_buildings = len(buildings)
    print(f"  Buildings: {n_buildings}")

    # --- Identify action indices ---
    from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env
    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)
    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]
    act_dim = len(names)

    print(f"  Action dim: {act_dim} (battery: {len(batt_idx)}, EV: {len(ev_idx)})")
    for i, n in enumerate(names):
        print(f"    [{i}] {n}")

    # Safety Projection Layer (for RL runs with batt_clamp=1)
    batt_action_map = discover_battery_actions(raw) if use_batt_clamp else []
    batt_clamp_count = 0
    if batt_action_map:
        print(f"  Battery Safety Projection: {len(batt_action_map)} batteries")

    # --- RBC setup ---
    rbc = None
    if run_name == "rbc":
        from scripts.rbc_policy import IntelligentRBC
        rbc = IntelligentRBC(env, ev_mode="greedy")
        print(f"  RBC: IntelligentRBC (ev_mode=greedy)")

    # --- Obs padding (Saute dim) ---
    need_pad = False
    pad_dim = 0
    if is_rl:
        env_obs_dim = env.observation_space.shape[0]
        if obs_dim_model > env_obs_dim:
            need_pad = True
            pad_dim = obs_dim_model - env_obs_dim
            print(f"  Padding obs: {env_obs_dim} -> {obs_dim_model} (+{pad_dim} for Saute budget)")

    # --- Reset ---
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    # =====================================================================
    # TRACKING VARIABLES
    # =====================================================================
    rewards = []
    costs = []
    actions_all = []

    # Per-constraint cumulative costs
    c0_vals, c1_vals, c2_vals, c3_vals, c4_vals = [], [], [], [], []

    # EV Departure tracking
    total_departures = 0
    violated_departures = 0
    departure_deficits = []
    ev_tracker = {}

    # C2: Battery SoC tracking
    soc_violation_steps = 0
    soc_all_steps = []

    # C3: Building power violations
    building_power_violations = [0] * n_buildings
    building_power_max = [0.0] * n_buildings
    building_power_all = [[] for _ in range(n_buildings)]
    c3_violation_steps = 0

    # C4: Grid power violations
    grid_violation_steps = 0
    grid_power_max = 0.0
    grid_power_all = []

    # V2G behavior
    ev_charge_steps = 0
    ev_discharge_steps = 0
    ev_idle_steps = 0
    total_ev_steps = 0

    # Energy tracking
    grid_imports = []
    grid_exports = []

    # Solar-EV tracking (hourly buckets)
    hourly_ev_charge_sums = np.zeros(24)
    hourly_ev_charge_counts = np.zeros(24)
    hourly_batt_charge_sums = np.zeros(24)
    hourly_batt_charge_counts = np.zeros(24)
    hourly_solar_sums = np.zeros(24)
    hourly_solar_counts = np.zeros(24)
    ev_charges_during_solar = 0
    ev_charges_total_positive = 0
    batt_charges_during_solar = 0
    batt_charges_total_positive = 0

    # CityLearn KPIs
    citylearn_kpis = {}

    # =====================================================================
    # ROLLOUT
    # =====================================================================
    print(f"\n  Running evaluation rollout...")

    while not done:
        # --- Compute action ---
        if run_name == "no_control":
            action = np.zeros(act_dim, dtype=np.float32)
        elif run_name == "rbc":
            action = rbc.predict(obs)
        else:
            # RL: prepare observation
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

        # WM disable: clamp WM action to 0 for RL runs
        if is_rl and WM_ACTION_INDEX < len(action):
            action[WM_ACTION_INDEX] = 0.0

        # Safety Projection Layer: clamp battery actions
        if use_batt_clamp and batt_action_map:
            action_pre_clamp = action.copy()
            action = clamp_battery_actions(action, batt_action_map, raw)
            if not np.array_equal(action, action_pre_clamp):
                batt_clamp_count += 1

        actions_all.append(action.copy())

        t_now = int(getattr(raw, "time_step", 0))
        t_idx = max(0, t_now - 1)
        hour = t_now % 24

        # === Solar generation for this step ===
        step_solar_kwh = 0.0
        for bld in buildings:
            sg = getattr(bld, "solar_generation", None)
            if sg is not None and hasattr(sg, '__len__') and t_idx < len(sg):
                # solar_generation is typically negative (generation), take absolute
                step_solar_kwh += abs(float(sg[t_idx]))
        is_solar_hour = step_solar_kwh > 0.1  # Threshold: >0.1 kWh total across buildings

        hourly_solar_sums[hour] += step_solar_kwh
        hourly_solar_counts[hour] += 1

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

        # === V2G action tracking + solar-EV correlation ===
        for ei in ev_idx:
            if ei < len(action):
                a = action[ei]
                total_ev_steps += 1
                if a > 0.05:
                    ev_charge_steps += 1
                    ev_charges_total_positive += 1
                    if is_solar_hour:
                        ev_charges_during_solar += 1
                elif a < -0.05:
                    ev_discharge_steps += 1
                else:
                    ev_idle_steps += 1
                # Track hourly EV charge
                if a > 0.0:
                    hourly_ev_charge_sums[hour] += a
                    hourly_ev_charge_counts[hour] += 1

        # === Battery action tracking for solar correlation ===
        for bi in batt_idx:
            if bi < len(action):
                a = action[bi]
                if a > 0.05:
                    batt_charges_total_positive += 1
                    if is_solar_hour:
                        batt_charges_during_solar += 1
                if a > 0.0:
                    hourly_batt_charge_sums[hour] += a
                    hourly_batt_charge_counts[hour] += 1

        # === Step environment ===
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cost = info.get("cost", 0.0)
        step += 1

        rewards.append(reward)
        costs.append(cost)

        # --- Per-constraint costs ---
        c0_vals.append(info.get("cost_ev_departure", 0.0) + info.get("cost_ev_dense", 0.0))
        c1_vals.append(info.get("cost_ev_dense", 0.0))
        c2_vals.append(info.get("cost_soc_pnorm", info.get("cost_stems_battery", 0.0)))
        c3_vals.append(info.get("cost_stems_building_power", 0.0))
        c4_vals.append(info.get("cost_stems_grid_power", 0.0))

        # --- Energy tracking ---
        grid_imports.append(info.get("grid_import_kwh", 0.0))
        grid_exports.append(info.get("grid_export_kwh", 0.0))

        # --- C2: Battery SoC ---
        soc_viol_any = info.get("battery_soc_violation_any", 0.0)
        if soc_viol_any > 0:
            soc_violation_steps += 1
        soc_step = []
        for b_idx, bld in enumerate(buildings):
            try:
                es = getattr(bld, "electrical_storage", None)
                soc = getattr(es, "soc", None) if es else None
                if soc is not None and len(soc) > t_idx:
                    s = float(np.clip(soc[t_idx], 0, 1))
                else:
                    s = 0.5
                soc_step.append(s)
            except Exception:
                soc_step.append(0.5)
        soc_all_steps.append(soc_step)

        # --- C3: Building power ---
        bld_viol = info.get("building_power_violation", 0.0)
        if bld_viol > 0:
            c3_violation_steps += 1
        for b_idx, bld in enumerate(buildings):
            try:
                nec = getattr(bld, 'net_electricity_consumption', None)
                if nec is not None and len(nec) > t_idx:
                    p = abs(float(nec[t_idx]))
                    building_power_all[b_idx].append(p)
                    building_power_max[b_idx] = max(building_power_max[b_idx], p)
                    if p > P_BUILDING_MAX:
                        building_power_violations[b_idx] += 1
            except Exception:
                pass

        # --- C4: Grid power ---
        grid_viol = info.get("grid_power_violation", 0.0)
        if grid_viol > 0:
            grid_violation_steps += 1
        gi = info.get("grid_import_kwh", 0.0)
        grid_power_all.append(gi)
        grid_power_max = max(grid_power_max, gi)

        # --- CityLearn KPIs (available on last step) ---
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

        # Progress indicator
        if step % 2000 == 0:
            print(f"    Step {step}: reward={reward:.2f}, cost={cost:.2f}")

    # =====================================================================
    # COMPUTE METRICS
    # =====================================================================
    total_reward = sum(rewards)
    total_cost = sum(costs)
    total_c0 = sum(c0_vals)
    total_c1 = sum(c1_vals)
    total_c2 = sum(c2_vals)
    total_c3 = sum(c3_vals)
    total_c4 = sum(c4_vals)

    ev_viol_pct = pct(violated_departures, total_departures)
    mean_deficit = float(np.mean(departure_deficits)) if departure_deficits else 0.0

    c2_viol_pct = pct(soc_violation_steps, step)
    c3_viol_pct = pct(c3_violation_steps, step)
    c4_viol_pct = pct(grid_violation_steps, step)

    total_import = sum(grid_imports)
    total_export = sum(grid_exports)

    # Solar-EV correlation
    # Compute hourly mean EV charge and hourly mean solar
    hourly_ev_mean = np.zeros(24)
    hourly_batt_mean = np.zeros(24)
    hourly_solar_mean = np.zeros(24)
    for h in range(24):
        if hourly_ev_charge_counts[h] > 0:
            hourly_ev_mean[h] = hourly_ev_charge_sums[h] / hourly_ev_charge_counts[h]
        if hourly_batt_charge_counts[h] > 0:
            hourly_batt_mean[h] = hourly_batt_charge_sums[h] / hourly_batt_charge_counts[h]
        if hourly_solar_counts[h] > 0:
            hourly_solar_mean[h] = hourly_solar_sums[h] / hourly_solar_counts[h]

    # Pearson correlation: hourly mean EV charge vs hourly mean solar
    ev_solar_corr = 0.0
    if np.std(hourly_ev_mean) > 1e-8 and np.std(hourly_solar_mean) > 1e-8:
        ev_solar_corr, _ = scipy_stats.pearsonr(hourly_ev_mean, hourly_solar_mean)
    batt_solar_corr = 0.0
    if np.std(hourly_batt_mean) > 1e-8 and np.std(hourly_solar_mean) > 1e-8:
        batt_solar_corr, _ = scipy_stats.pearsonr(hourly_batt_mean, hourly_solar_mean)

    ev_solar_charging_pct = pct(ev_charges_during_solar, ev_charges_total_positive)
    batt_solar_charging_pct = pct(batt_charges_during_solar, batt_charges_total_positive)

    # Total solar generation
    total_solar = float(np.sum(hourly_solar_sums))

    # =====================================================================
    # PRINT RESULTS
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  RESULTS: {description}")
    print(f"{'='*70}")
    print(f"  Steps:              {step}")
    print(f"  Total Reward:       {total_reward:.2f}")
    print(f"  Total Cost (CMDP):  {total_cost:.2f}")

    print(f"\n  Constraints:")
    print(f"  {'Constraint':<25} {'EpCost':>10} {'Limit':>10} {'Status':>10}")
    print(f"  {'-'*55}")
    for name, cval, lim in [
        ("C0 EV departure", total_c0, COST_LIMITS["C0_ev_departure"]),
        ("C1 EV Saute", total_c1, COST_LIMITS["C1_ev_saute"]),
        ("C2 Battery SoC", total_c2, COST_LIMITS["C2_battery_soc"]),
        ("C3 Building power", total_c3, COST_LIMITS["C3_building_power"]),
        ("C4 Grid power", total_c4, COST_LIMITS["C4_grid_power"]),
    ]:
        status = "OK" if cval <= lim else "VIOLATED"
        print(f"  {name:<25} {cval:>10.1f} {lim:>10} {status:>10}")

    print(f"\n  EV Departures: {violated_departures}/{total_departures} violated ({ev_viol_pct:.1f}%)")
    print(f"  V2G: charge={pct(ev_charge_steps, total_ev_steps):.1f}% "
          f"discharge={pct(ev_discharge_steps, total_ev_steps):.1f}% "
          f"idle={pct(ev_idle_steps, total_ev_steps):.1f}%")
    print(f"  Solar-EV charging: {ev_solar_charging_pct:.1f}% of EV charges during solar hours")
    print(f"  Solar-EV correlation: {ev_solar_corr:.3f}")
    print(f"  Energy: import={total_import:.0f} export={total_export:.0f} net={total_import-total_export:.0f} solar={total_solar:.0f} kWh")
    if use_batt_clamp:
        print(f"  Battery clamp active: {batt_clamp_count}/{step} steps ({pct(batt_clamp_count, step):.1f}%)")

    # =====================================================================
    # BUILD JSON RESULTS
    # =====================================================================
    # CityLearn KPIs: clean up
    cl_kpis = {}
    kpi_name_map = {
        "citylearn_electricity_consumption_total": "electricity_consumption",
        "citylearn_carbon_emissions_total": "carbon_emissions",
        "citylearn_cost_total": "cost",
        "citylearn_daily_peak_average": "daily_peak_avg",
        "citylearn_all_time_peak_average": "all_time_peak_avg",
        "citylearn_ramping_average": "ramping_avg",
        "citylearn_discomfort_proportion": "discomfort",
        "citylearn_zero_net_energy": "zero_net_energy",
    }
    for raw_key, short_key in kpi_name_map.items():
        val = citylearn_kpis.get(raw_key, float("nan"))
        cl_kpis[short_key] = val if np.isfinite(val) else None

    results = {
        "run": run_name,
        "description": description,
        "checkpoint": ckpt_path,
        "seed": seed,
        "steps": step,
        "citylearn_kpis": cl_kpis,
        "constraints": {
            "C0_ev_departure": {"cost": round(total_c0, 2), "limit": COST_LIMITS["C0_ev_departure"]},
            "C1_ev_saute": {"cost": round(total_c1, 2), "limit": COST_LIMITS["C1_ev_saute"]},
            "C2_battery_soc": {"cost": round(total_c2, 2), "limit": COST_LIMITS["C2_battery_soc"]},
            "C3_building_power": {"cost": round(total_c3, 2), "limit": COST_LIMITS["C3_building_power"]},
            "C4_grid_power": {"cost": round(total_c4, 2), "limit": COST_LIMITS["C4_grid_power"]},
        },
        "violation_rates_pct": {
            "C2_soc_steps": round(c2_viol_pct, 2),
            "C3_building_steps": round(c3_viol_pct, 2),
            "C4_grid_steps": round(c4_viol_pct, 2),
        },
        "ev_departures": {
            "total": total_departures,
            "violated": violated_departures,
            "violation_pct": round(ev_viol_pct, 2),
            "mean_deficit": round(mean_deficit, 4),
        },
        "solar_ev": {
            "ev_solar_charging_pct": round(ev_solar_charging_pct, 2),
            "batt_solar_charging_pct": round(batt_solar_charging_pct, 2),
            "solar_ev_correlation": round(float(ev_solar_corr), 3),
        },
        "energy": {
            "total_import_kwh": round(total_import, 2),
            "total_export_kwh": round(total_export, 2),
            "net_consumption_kwh": round(total_import - total_export, 2),
            "total_solar_kwh": round(total_solar, 2),
        },
        "v2g": {
            "charge_pct": round(pct(ev_charge_steps, total_ev_steps), 2),
            "discharge_pct": round(pct(ev_discharge_steps, total_ev_steps), 2),
            "idle_pct": round(pct(ev_idle_steps, total_ev_steps), 2),
        },
        "total_reward": round(total_reward, 2),
        "total_cost": round(total_cost, 2),
    }

    # Save to JSON
    out_path = os.path.join(PROJECT, "runs", f"{run_name}_eval_compare.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return results


# =========================================================================
# Summary table
# =========================================================================
def print_summary():
    """Read all *_eval_compare.json files and print a unified comparison table."""
    pattern = os.path.join(PROJECT, "runs", "*_eval_compare.json")
    json_files = sorted(glob_mod.glob(pattern))

    if not json_files:
        print("No eval_compare.json files found in runs/")
        print(f"  Searched: {pattern}")
        return

    # Load all results
    all_results = {}
    for fp in json_files:
        with open(fp) as f:
            data = json.load(f)
        run_name = data["run"]
        all_results[run_name] = data

    # Define display order
    display_order = ["no_control", "rbc", "r19", "r21", "r22", "r23"]
    runs = [r for r in display_order if r in all_results]
    # Add any extra runs not in the predefined order
    for r in all_results:
        if r not in runs:
            runs.append(r)

    if not runs:
        print("No results loaded.")
        return

    # Column widths
    label_w = 30
    col_w = 12
    n_cols = len(runs)
    total_w = label_w + col_w * n_cols + 4

    def hline(char="="):
        return char * total_w

    def header_row():
        """Print the header with run labels."""
        labels = {
            "no_control": "NoCtrl",
            "rbc": "RBC",
            "r19": "R19",
            "r21": "R21",
            "r22": "R22",
            "r23": "R23",
        }
        row = f"  {'Metric':<{label_w}}"
        for r in runs:
            lbl = labels.get(r, r.upper())
            row += f"{lbl:>{col_w}}"
        return row

    def data_row(label, values, fmt=".3f"):
        """Format a single data row."""
        row = f"  {label:<{label_w}}"
        for v in values:
            if v is None:
                row += f"{'N/A':>{col_w}}"
            elif isinstance(v, str):
                row += f"{v:>{col_w}}"
            else:
                row += f"{v:>{col_w}{fmt}}"
        return row

    def get_val(run, *keys, default=None):
        """Drill into nested dict using key path."""
        d = all_results.get(run, {})
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d

    # =====================================================================
    # PRINT TABLE
    # =====================================================================
    print(f"\n{'='*total_w}")
    print(f"{'COMPREHENSIVE RUN COMPARISON':^{total_w}}")
    print(f"{'='*total_w}")

    # Description row
    print(f"\n  {'Run descriptions:'}")
    for r in runs:
        desc = get_val(r, "description", default=r)
        print(f"    {r:>12}: {desc}")

    # --- 1. CityLearn KPIs ---
    print(f"\n{hline('-')}")
    print(f"  1. CityLearn KPIs (lower = better, 1.0 = no-control baseline)")
    print(f"{hline('-')}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    kpi_keys = ["electricity_consumption", "carbon_emissions", "cost", "daily_peak_avg",
                "all_time_peak_avg", "ramping_avg", "discomfort", "zero_net_energy"]
    for kpi in kpi_keys:
        vals = [get_val(r, "citylearn_kpis", kpi) for r in runs]
        print(data_row(kpi, vals))

    # --- 2. Constraint Violations ---
    print(f"\n{hline('-')}")
    print(f"  2. Constraint Violations (EpCost / Limit)")
    print(f"{hline('-')}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    constraint_keys = ["C0_ev_departure", "C1_ev_saute", "C2_battery_soc",
                       "C3_building_power", "C4_grid_power"]
    for ck in constraint_keys:
        limit = COST_LIMITS[ck]
        vals = []
        for r in runs:
            c = get_val(r, "constraints", ck, "cost")
            if c is not None:
                ratio = c / limit if limit > 0 else 0
                vals.append(ratio)
            else:
                vals.append(None)
        print(data_row(f"{ck} (/{limit})", vals, fmt=".2f"))

    # Also show raw cost values
    print()
    print(f"  {'(Raw EpCost values)'}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)
    for ck in constraint_keys:
        vals = [get_val(r, "constraints", ck, "cost") for r in runs]
        print(data_row(ck, vals, fmt=".0f"))

    # Violation rate (% of steps)
    print()
    print(f"  {'Step-level violation rates (%)'}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)
    for vk in ["C2_soc_steps", "C3_building_steps", "C4_grid_steps"]:
        vals = [get_val(r, "violation_rates_pct", vk) for r in runs]
        print(data_row(vk, vals, fmt=".2f"))

    # --- 3. EV Departure Reliability ---
    print(f"\n{hline('-')}")
    print(f"  3. EV Departure Reliability")
    print(f"{hline('-')}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    for metric, keys, fmt in [
        ("total_departures", ("ev_departures", "total"), ".0f"),
        ("violated_departures", ("ev_departures", "violated"), ".0f"),
        ("violation_pct (%)", ("ev_departures", "violation_pct"), ".2f"),
        ("mean_deficit (SoC)", ("ev_departures", "mean_deficit"), ".4f"),
    ]:
        vals = [get_val(r, *keys) for r in runs]
        print(data_row(metric, vals, fmt=fmt))

    # --- 4. Solar-EV Charging Intelligence ---
    print(f"\n{hline('-')}")
    print(f"  4. Solar-EV Charging Intelligence")
    print(f"{hline('-')}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    for metric, keys, fmt in [
        ("EV solar charging (%)", ("solar_ev", "ev_solar_charging_pct"), ".1f"),
        ("Batt solar charging (%)", ("solar_ev", "batt_solar_charging_pct"), ".1f"),
        ("Solar-EV correlation", ("solar_ev", "solar_ev_correlation"), ".3f"),
    ]:
        vals = [get_val(r, *keys) for r in runs]
        print(data_row(metric, vals, fmt=fmt))

    # --- 5. Energy Balance ---
    print(f"\n{hline('-')}")
    print(f"  5. Energy Balance (kWh)")
    print(f"{hline('-')}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    for metric, keys, fmt in [
        ("total_import", ("energy", "total_import_kwh"), ".0f"),
        ("total_export", ("energy", "total_export_kwh"), ".0f"),
        ("net_consumption", ("energy", "net_consumption_kwh"), ".0f"),
        ("total_solar", ("energy", "total_solar_kwh"), ".0f"),
    ]:
        vals = [get_val(r, *keys) for r in runs]
        print(data_row(metric, vals, fmt=fmt))

    # --- 6. V2G Behavior ---
    print(f"\n{hline('-')}")
    print(f"  6. V2G Behavior (%)")
    print(f"{hline('-')}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    for metric, keys, fmt in [
        ("charge (%)", ("v2g", "charge_pct"), ".1f"),
        ("discharge (%)", ("v2g", "discharge_pct"), ".1f"),
        ("idle (%)", ("v2g", "idle_pct"), ".1f"),
    ]:
        vals = [get_val(r, *keys) for r in runs]
        print(data_row(metric, vals, fmt=fmt))

    # --- 7. Episode Summary ---
    print(f"\n{hline('-')}")
    print(f"  7. Episode Summary")
    print(f"{hline('-')}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    for metric, key, fmt in [
        ("total_reward", "total_reward", ".0f"),
        ("total_cost", "total_cost", ".0f"),
        ("steps", "steps", ".0f"),
    ]:
        vals = [get_val(r, key) for r in runs]
        print(data_row(metric, vals, fmt=fmt))

    print(f"\n{hline('=')}")

    # --- Best/worst indicators ---
    print(f"\n  Key Takeaways:")

    # Best reward
    reward_vals = {r: get_val(r, "total_reward") for r in runs if get_val(r, "total_reward") is not None}
    if reward_vals:
        best_reward_run = max(reward_vals, key=reward_vals.get)
        print(f"    Best reward:           {best_reward_run} ({reward_vals[best_reward_run]:.0f})")

    # Best EV compliance
    ev_viol_vals = {r: get_val(r, "ev_departures", "violation_pct")
                    for r in runs if get_val(r, "ev_departures", "violation_pct") is not None}
    if ev_viol_vals:
        best_ev_run = min(ev_viol_vals, key=ev_viol_vals.get)
        print(f"    Best EV compliance:    {best_ev_run} ({ev_viol_vals[best_ev_run]:.1f}% violation)")

    # Best C3 (lowest cost ratio)
    c3_vals = {r: (get_val(r, "constraints", "C3_building_power", "cost") or 0) / COST_LIMITS["C3_building_power"]
               for r in runs if get_val(r, "constraints", "C3_building_power", "cost") is not None}
    if c3_vals:
        best_c3_run = min(c3_vals, key=c3_vals.get)
        print(f"    Best C3 (building):    {best_c3_run} ({c3_vals[best_c3_run]:.2f}x limit)")

    # Best solar-EV correlation
    solar_corr_vals = {r: get_val(r, "solar_ev", "solar_ev_correlation")
                       for r in runs if get_val(r, "solar_ev", "solar_ev_correlation") is not None}
    if solar_corr_vals:
        best_solar_run = max(solar_corr_vals, key=solar_corr_vals.get)
        print(f"    Best solar-EV corr:    {best_solar_run} ({solar_corr_vals[best_solar_run]:.3f})")

    print()


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified evaluation and comparison for Safe-CityLearn runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/eval_compare_all.py --run r21 --epoch 89
    python scripts/eval_compare_all.py --run no_control
    python scripts/eval_compare_all.py --run rbc
    python scripts/eval_compare_all.py --run summary
        """,
    )
    parser.add_argument("--run", required=True,
                        choices=list(RUNS.keys()) + ["summary"],
                        help="Which run to evaluate, or 'summary' to print comparison table")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific checkpoint epoch (default: latest)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for evaluation (default: 42)")

    args = parser.parse_args()

    if args.run == "summary":
        print_summary()
    else:
        result = evaluate(args.run, epoch=args.epoch, seed=args.seed)
        if result is None:
            sys.exit(1)
