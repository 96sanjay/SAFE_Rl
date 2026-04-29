#!/usr/bin/env python3
"""
Comprehensive Evaluation & Comparison Script for Thesis
========================================================
Evaluates ALL runs (no_control, rbc, r19, r21, r22, r23) and produces:
  A) CityLearn KPIs (8 metrics)
  B) Per-constraint violation percentages and costs
  C) EV departure reliability
  D) Intelligence metrics (solar, price, V2G, temporal, diversity, forecast)
  E) Intelligence scorecard (0-1 scores, overall)
  F) Energy balance, V2G behavior, solar-EV correlation

Output:
  1. Individual run details (verbose)
  2. Comparison tables (8 tables side-by-side)
  3. JSON: runs/full_comparison.json

Usage:
    python scripts/eval_full_comparison.py                   # all runs
    python scripts/eval_full_comparison.py --run r21         # single run
    python scripts/eval_full_comparison.py --run all         # all runs (explicit)
    python scripts/eval_full_comparison.py --seed 42         # custom seed
    python scripts/eval_full_comparison.py --epoch 89        # specific epoch
"""
import os
import sys
import json
import glob as glob_mod
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# =========================================================================
# Configuration
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
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_PID_LAGRANGE": "0",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_WM_DISABLE": "1",
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
    "r19": {
        "run_dir": "runs/r19_ablation/5bld",
        "overrides": {"CITYLEARN_BATT_CLAMP": "0"},
        "description": "R19: Ablation (no safety projection)",
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
WM_ACTION_INDEX = 2

DISPLAY_ORDER = ["no_control", "rbc", "r19", "r21", "r22", "r23"]

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


def find_checkpoint(run_dir, epoch=None, seed=42):
    """Discover the checkpoint path for an RL run."""
    seed_pattern = f"seed-{seed:03d}*"
    pattern = os.path.join(PROJECT, run_dir, "PPOLagMulti-*", seed_pattern, "torch_save")
    save_dirs = sorted(glob_mod.glob(pattern))
    if not save_dirs:
        pattern2 = os.path.join(PROJECT, run_dir, "PPOLag-*", seed_pattern, "torch_save")
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
    buildings = list(getattr(raw_env, "buildings", []))
    names_raw = getattr(raw_env, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        flat_names = names_raw[0]
    else:
        flat_names = list(names_raw)
    batt_pos = [i for i, n in enumerate(flat_names) if str(n).lower() == "electrical_storage"]
    result = []
    for b_idx, b in enumerate(buildings):
        es = getattr(b, "electrical_storage", None)
        if es is None:
            continue
        if b_idx < len(batt_pos):
            act_idx = batt_pos[b_idx]
        else:
            continue
        cap = float(getattr(es, "capacity", 6.4) or 6.4)
        p_max = float(getattr(es, "nominal_power", 5.0) or 5.0)
        eta = float(getattr(es, "efficiency", 0.9) or 0.9)
        result.append((act_idx, b_idx, cap, p_max, eta))
    return result


def clamp_battery_actions(a, batt_action_map, raw_env):
    """Safety Projection Layer (Dalal et al. 2018) for battery SoC."""
    if not batt_action_map:
        return a
    buildings = list(getattr(raw_env, "buildings", []))
    t_idx = max(0, int(getattr(raw_env, "time_step", 0)) - 1)
    a_clamped = a.copy()
    for act_idx, bld_idx, cap, p_max, eta in batt_action_map:
        if act_idx >= len(a_clamped) or bld_idx >= len(buildings):
            continue
        es = getattr(buildings[bld_idx], "electrical_storage", None)
        if es is None:
            continue
        soc_arr = getattr(es, "soc", None)
        if soc_arr is None or not hasattr(soc_arr, "__len__") or len(soc_arr) <= t_idx:
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
# Forecast perturbation test
# =========================================================================
def compute_forecast_utilization(obs_raw_all, actions_orig_all, actor, obs_mean, obs_std, obs_clip):
    """Run perturbation test on forecast dimensions to measure utilization."""
    obs_raw = np.array(obs_raw_all)
    actions_orig = np.array(actions_orig_all)
    obs_dim = obs_raw.shape[1]

    base_obs_dim = 70
    fc_start = base_obs_dim
    n_forecast = 128

    if fc_start + 72 > obs_dim:
        return {
            "forecast_utilization": 0.0,
            "price_sensitivity": 0.0,
            "solar_sensitivity": 0.0,
            "all_sensitivity": 0.0,
            "price_score": 0.0,
            "solar_score": 0.0,
            "all_score": 0.0,
        }

    price_range = (fc_start, fc_start + 24)
    solar_range = (fc_start + 48, fc_start + 72)
    all_range = (fc_start, min(fc_start + n_forecast, obs_dim))

    SCALE_THRESHOLD = 0.1

    @torch.no_grad()
    def batched_forward(obs_np):
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        if obs_mean is not None:
            obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
            if obs_clip is not None:
                obs_t = obs_t.clamp(-obs_clip, obs_clip)
        acts = actor(obs_t).numpy()
        return np.clip(acts, -1.0, 1.0)

    perturbations = {
        "price_forecast": price_range,
        "solar_forecast": solar_range,
        "all_forecast": (all_range[0], all_range[1]),
    }

    deltas = {}
    for name, (start, end) in perturbations.items():
        end = min(end, obs_dim)
        obs_perturbed = obs_raw.copy()
        obs_perturbed[:, start:end] = 0.0
        actions_perturbed = batched_forward(obs_perturbed)
        step_deltas = np.mean(np.abs(actions_orig - actions_perturbed), axis=1)
        deltas[name] = float(np.mean(step_deltas))

    price_score = float(np.clip(deltas["price_forecast"] / SCALE_THRESHOLD, 0, 1))
    solar_score = float(np.clip(deltas["solar_forecast"] / SCALE_THRESHOLD, 0, 1))
    all_score = float(np.clip(deltas["all_forecast"] / SCALE_THRESHOLD, 0, 1))
    forecast_util = float(np.mean([price_score, solar_score, all_score]))

    return {
        "forecast_utilization": forecast_util,
        "price_sensitivity": deltas["price_forecast"],
        "solar_sensitivity": deltas["solar_forecast"],
        "all_sensitivity": deltas["all_forecast"],
        "price_score": price_score,
        "solar_score": solar_score,
        "all_score": all_score,
    }


# =========================================================================
# Core evaluation loop
# =========================================================================
def evaluate(run_name, epoch=None, seed=42):
    """Evaluate a single run and return a comprehensive results dict."""
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
            print(f"  Searched: {os.path.join(PROJECT, run_cfg['run_dir'])}")
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

    # --- Price percentiles ---
    try:
        all_prices_arr = np.array(buildings[0].pricing.electricity_pricing, dtype=float)
        p25 = float(np.percentile(all_prices_arr, 25))
        p50 = float(np.percentile(all_prices_arr, 50))
        p75 = float(np.percentile(all_prices_arr, 75))
    except Exception:
        p25, p50, p75 = 0.12, 0.16, 0.20
    print(f"  Price percentiles: P25={p25:.4f}, P50={p50:.4f}, P75={p75:.4f}")

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

    # C3: Building power violations
    building_power_violations = [0] * n_buildings
    building_power_max = [0.0] * n_buildings
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

    # Intelligence data collection (per-step)
    step_hours = []
    step_prices = []
    step_solars = []
    step_ev_actions = []
    step_batt_actions = []
    step_obs_raw = []  # for forecast perturbation (RL only)
    step_actions_raw = []  # for forecast perturbation (RL only)

    # Solar-EV tracking
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
    print(f"\n  Running evaluation rollout (8760 steps)...")

    while not done:
        # --- Compute action ---
        if run_name == "no_control":
            action = np.zeros(act_dim, dtype=np.float32)
        elif run_name == "rbc":
            action = rbc.predict(obs)
        else:
            # RL: prepare observation
            obs_np = np.asarray(obs, dtype=np.float32)
            if need_pad:
                obs_padded = np.concatenate([obs_np, np.ones(pad_dim)])
            else:
                obs_padded = obs_np
            # Store raw obs for forecast perturbation
            step_obs_raw.append(obs_padded.copy())

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
        if is_rl:
            step_actions_raw.append(action.copy())

        t_now = int(getattr(raw, "time_step", 0))
        t_idx = max(0, t_now - 1)
        hour = t_now % 24

        # === Price for this step ===
        try:
            step_price = float(buildings[0].pricing.electricity_pricing[t_idx])
        except Exception:
            step_price = 0.17

        # === Solar generation for this step ===
        step_solar_kwh = 0.0
        for bld in buildings:
            sg = getattr(bld, "solar_generation", None)
            if sg is not None and hasattr(sg, "__len__") and t_idx < len(sg):
                step_solar_kwh += abs(float(sg[t_idx]))
        is_solar_hour = step_solar_kwh > 0.1

        # Store intelligence data
        step_hours.append(hour)
        step_prices.append(step_price)
        step_solars.append(step_solar_kwh)
        step_ev_actions.append([float(action[i]) for i in ev_idx if i < len(action)])
        step_batt_actions.append([float(action[i]) for i in batt_idx if i < len(action)])

        hourly_solar_sums[hour] += step_solar_kwh
        hourly_solar_counts[hour] += 1

        # === EV departure tracking ===
        for b_idx, bld in enumerate(buildings):
            chargers = getattr(bld, "electric_vehicle_chargers", None) or []
            for ch_idx, ch in enumerate(chargers):
                key = (b_idx, ch_idx)
                sim = getattr(ch, "charger_simulation",
                              getattr(ch, "_Charger__charger_simulation", None))
                if sim is None:
                    continue
                try:
                    sa = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                    current_connected = t_now < len(sa) and float(sa[t_now]) == 1.0
                    if current_connected:
                        ra = np.asarray(getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
                        rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                        if not np.isfinite(rs):
                            rs = 1.0
                        ev_obj = getattr(ch, "connected_electric_vehicle", None)
                        current_soc = 0.0
                        if ev_obj is not None:
                            bt = getattr(ev_obj, "battery", None)
                            if bt is not None:
                                soc_arr = getattr(bt, "soc", None)
                                if soc_arr is not None:
                                    sn = np.asarray(soc_arr, dtype=float)
                                    current_soc = float(np.clip(sn[t_idx], 0, 1)) if 0 <= t_idx < len(sn) else 0.0
                        ev_tracker[key] = {"was_connected": True, "last_soc": current_soc, "required_soc": rs}
                    else:
                        prev = ev_tracker.get(key, {})
                        if prev.get("was_connected", False):
                            total_departures += 1
                            last_soc = prev["last_soc"]
                            rs = prev["required_soc"]
                            deficit = max(0.0, rs - last_soc)
                            if deficit > 0.01:
                                violated_departures += 1
                            departure_deficits.append(deficit)
                        ev_tracker[key] = {"was_connected": False}
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

        # --- C3: Building power ---
        bld_viol = info.get("building_power_violation", 0.0)
        if bld_viol > 0:
            c3_violation_steps += 1
        for b_idx_c3, bld in enumerate(buildings):
            try:
                nec = getattr(bld, "net_electricity_consumption", None)
                if nec is not None and len(nec) > t_idx:
                    p = abs(float(nec[t_idx]))
                    building_power_max[b_idx_c3] = max(building_power_max[b_idx_c3], p)
                    if p > P_BUILDING_MAX:
                        building_power_violations[b_idx_c3] += 1
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
            for kpi_key in [
                "citylearn_electricity_consumption_total",
                "citylearn_carbon_emissions_total",
                "citylearn_cost_total",
                "citylearn_daily_peak_average",
                "citylearn_all_time_peak_average",
                "citylearn_ramping_average",
                "citylearn_discomfort_proportion",
                "citylearn_zero_net_energy",
            ]:
                citylearn_kpis[kpi_key] = info.get(kpi_key, float("nan"))

        # Progress indicator
        if step % 2000 == 0:
            print(f"    Step {step}: reward={reward:.2f}, cost={cost:.2f}")

    # =====================================================================
    # COMPUTE BASE METRICS
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

    c1_step_rate = pct(sum(1 for v in c1_vals if v > 0), step)
    c2_viol_pct = pct(soc_violation_steps, step)
    c3_viol_pct = pct(c3_violation_steps, step)
    c4_viol_pct = pct(grid_violation_steps, step)

    total_import = sum(grid_imports)
    total_export = sum(grid_exports)
    total_solar = float(np.sum(hourly_solar_sums))

    ev_solar_charging_pct = pct(ev_charges_during_solar, ev_charges_total_positive)
    batt_solar_charging_pct = pct(batt_charges_during_solar, batt_charges_total_positive)

    # =====================================================================
    # INTELLIGENCE METRICS
    # =====================================================================
    hours_arr = np.array(step_hours)
    prices_arr = np.array(step_prices)
    solars_arr = np.array(step_solars)
    ev_act_arr = np.array(step_ev_actions)
    batt_act_arr = np.array(step_batt_actions)

    mean_ev = ev_act_arr.mean(axis=1) if ev_act_arr.ndim == 2 and ev_act_arr.shape[1] > 0 else np.zeros(len(step_hours))
    mean_batt = batt_act_arr.mean(axis=1) if batt_act_arr.ndim == 2 and batt_act_arr.shape[1] > 0 else np.zeros(len(step_hours))
    N = len(hours_arr)

    # --- 1. Solar Charging ---
    solar_med = float(np.median(solars_arr[solars_arr > 0])) if (solars_arr > 0).any() else 1.0
    high_solar = solars_arr > solar_med
    if high_solar.sum() > 0:
        ev_charge_during_solar_rate = float((mean_ev[high_solar] > 0.05).mean())
        mean_ev_act_solar = float(mean_ev[high_solar].mean())
    else:
        ev_charge_during_solar_rate = 0.0
        mean_ev_act_solar = 0.0
    low_solar = solars_arr < 0.01
    mean_ev_act_nosolar = float(mean_ev[low_solar].mean()) if low_solar.sum() > 0 else 0.0
    solar_diff = mean_ev_act_solar - mean_ev_act_nosolar

    # --- 2. Price-Aware V2G ---
    expensive = prices_arr > p75
    cheap = prices_arr < p25
    if expensive.sum() > 0:
        batt_discharge_expensive = float((mean_batt[expensive] < -0.05).mean())
        mean_batt_expensive = float(mean_batt[expensive].mean())
        ev_discharge_expensive = float((mean_ev[expensive] < -0.05).mean())
        mean_ev_expensive = float(mean_ev[expensive].mean())
    else:
        batt_discharge_expensive = mean_batt_expensive = ev_discharge_expensive = mean_ev_expensive = 0.0

    # --- 3. Off-Peak Charging ---
    if cheap.sum() > 0:
        batt_charge_cheap = float((mean_batt[cheap] > 0.05).mean())
        mean_batt_cheap = float(mean_batt[cheap].mean())
        ev_charge_cheap = float((mean_ev[cheap] > 0.05).mean())
        mean_ev_cheap = float(mean_ev[cheap].mean())
    else:
        batt_charge_cheap = mean_batt_cheap = ev_charge_cheap = mean_ev_cheap = 0.0
    price_batt_diff = mean_batt_cheap - mean_batt_expensive

    # --- 4. Temporal Planning: hourly profiles ---
    hourly_batt_profile = np.zeros(24)
    hourly_ev_profile = np.zeros(24)
    hourly_price_profile = np.zeros(24)
    hourly_solar_profile = np.zeros(24)
    for h in range(24):
        mask = hours_arr == h
        if mask.sum() > 0:
            hourly_batt_profile[h] = float(mean_batt[mask].mean())
            hourly_ev_profile[h] = float(mean_ev[mask].mean())
            hourly_price_profile[h] = float(prices_arr[mask].mean())
            hourly_solar_profile[h] = float(solars_arr[mask].mean())

    # --- 5. Pre-peak preparation ---
    peak_hours = {h for h in range(24) if hourly_price_profile[h] > p75}
    pre_peak_hours = set()
    for ph in peak_hours:
        for offset in [2, 3, 4]:
            pre_h = (ph - offset) % 24
            if pre_h not in peak_hours:
                pre_peak_hours.add(pre_h)

    if pre_peak_hours:
        pre_peak_mask = np.isin(hours_arr, list(pre_peak_hours))
        pre_peak_batt = float(mean_batt[pre_peak_mask].mean()) if pre_peak_mask.sum() > 0 else 0.0
        peak_mask = np.isin(hours_arr, list(peak_hours))
        peak_batt = float(mean_batt[peak_mask].mean()) if peak_mask.sum() > 0 else 0.0
    else:
        pre_peak_batt = peak_batt = 0.0

    # --- 6. Correlations ---
    corr_batt_price = 0.0
    if np.std(mean_batt) > 1e-8 and np.std(prices_arr) > 1e-8:
        corr_batt_price = float(np.corrcoef(prices_arr, mean_batt)[0, 1])
    corr_ev_price = 0.0
    if np.std(mean_ev) > 1e-8 and np.std(prices_arr) > 1e-8:
        corr_ev_price = float(np.corrcoef(prices_arr, mean_ev)[0, 1])

    solar_mask_corr = solars_arr > 0
    corr_ev_solar = 0.0
    if solar_mask_corr.sum() > 100 and np.std(mean_ev[solar_mask_corr]) > 1e-8:
        corr_ev_solar = float(np.corrcoef(solars_arr[solar_mask_corr], mean_ev[solar_mask_corr])[0, 1])

    # Hourly-level correlations
    hourly_ev_mean_charge = np.zeros(24)
    hourly_batt_mean_charge = np.zeros(24)
    hourly_solar_mean = np.zeros(24)
    for h in range(24):
        if hourly_ev_charge_counts[h] > 0:
            hourly_ev_mean_charge[h] = hourly_ev_charge_sums[h] / hourly_ev_charge_counts[h]
        if hourly_batt_charge_counts[h] > 0:
            hourly_batt_mean_charge[h] = hourly_batt_charge_sums[h] / hourly_batt_charge_counts[h]
        if hourly_solar_counts[h] > 0:
            hourly_solar_mean[h] = hourly_solar_sums[h] / hourly_solar_counts[h]

    ev_solar_corr_hourly = 0.0
    if np.std(hourly_ev_mean_charge) > 1e-8 and np.std(hourly_solar_mean) > 1e-8:
        ev_solar_corr_hourly, _ = scipy_stats.pearsonr(hourly_ev_mean_charge, hourly_solar_mean)
        ev_solar_corr_hourly = float(ev_solar_corr_hourly)
    batt_solar_corr_hourly = 0.0
    if np.std(hourly_batt_mean_charge) > 1e-8 and np.std(hourly_solar_mean) > 1e-8:
        batt_solar_corr_hourly, _ = scipy_stats.pearsonr(hourly_batt_mean_charge, hourly_solar_mean)
        batt_solar_corr_hourly = float(batt_solar_corr_hourly)

    # --- 7. Behavioral diversity ---
    n_batt_charge_hours = sum(1 for h in range(24) if hourly_batt_profile[h] > 0.05)
    n_batt_discharge_hours = sum(1 for h in range(24) if hourly_batt_profile[h] < -0.05)
    n_batt_idle_hours = 24 - n_batt_charge_hours - n_batt_discharge_hours
    action_variance = float(np.std(hourly_batt_profile))

    # --- 8. Forecast utilization (RL only) ---
    forecast_results = {
        "forecast_utilization": 0.0,
        "price_sensitivity": 0.0,
        "solar_sensitivity": 0.0,
        "all_sensitivity": 0.0,
        "price_score": 0.0,
        "solar_score": 0.0,
        "all_score": 0.0,
    }
    if is_rl and actor is not None and len(step_obs_raw) > 0:
        forecast_results = compute_forecast_utilization(
            step_obs_raw, step_actions_raw, actor, obs_mean, obs_std, obs_clip
        )

    # --- Intelligence Scorecard ---
    scores = {}
    scores["solar_preference"] = float(np.clip(solar_diff / 0.2, 0, 1))
    scores["price_spread"] = float(np.clip(price_batt_diff / 0.3, 0, 1))
    scores["v2g_timing"] = float(np.clip(batt_discharge_expensive, 0, 1))
    scores["ev_compliance"] = float(np.clip(1.0 - c1_step_rate / 10.0, 0, 1))
    scores["grid_stability"] = float(np.clip(1.0 - c4_viol_pct / 30.0, 0, 1))
    scores["pre_peak_planning"] = float(np.clip((pre_peak_batt - peak_batt) / 0.3, 0, 1))
    scores["price_correlation"] = float(np.clip(-corr_batt_price / 0.3, 0, 1))
    scores["behavioral_diversity"] = float(np.clip(action_variance / 0.15, 0, 1))
    scores["forecast_utilization"] = forecast_results["forecast_utilization"]
    overall_intelligence = float(np.mean(list(scores.values())))

    # =====================================================================
    # PRINT INDIVIDUAL RUN DETAILS
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  RESULTS: {description}")
    print(f"{'='*70}")
    print(f"  Steps:              {step}")
    print(f"  Total Reward:       {total_reward:.2f}")
    print(f"  Total Cost (CMDP):  {total_cost:.2f}")

    # Constraints
    print(f"\n  Constraints:")
    print(f"  {'Constraint':<25} {'EpCost':>10} {'Limit':>10} {'Status':>10}")
    print(f"  {'-'*55}")
    for cname, cval, lim in [
        ("C0 EV departure", total_c0, COST_LIMITS["C0_ev_departure"]),
        ("C1 EV Saute", total_c1, COST_LIMITS["C1_ev_saute"]),
        ("C2 Battery SoC", total_c2, COST_LIMITS["C2_battery_soc"]),
        ("C3 Building power", total_c3, COST_LIMITS["C3_building_power"]),
        ("C4 Grid power", total_c4, COST_LIMITS["C4_grid_power"]),
    ]:
        status = "OK" if cval <= lim else "VIOLATED"
        print(f"  {cname:<25} {cval:>10.1f} {lim:>10} {status:>10}")

    # EV Departures
    print(f"\n  EV Departures: {violated_departures}/{total_departures} violated ({ev_viol_pct:.1f}%)")
    print(f"  Mean deficit:  {mean_deficit:.4f}")
    print(f"  V2G: charge={pct(ev_charge_steps, total_ev_steps):.1f}% "
          f"discharge={pct(ev_discharge_steps, total_ev_steps):.1f}% "
          f"idle={pct(ev_idle_steps, total_ev_steps):.1f}%")

    if use_batt_clamp:
        print(f"  Battery clamp: {batt_clamp_count}/{step} steps ({pct(batt_clamp_count, step):.1f}%)")

    # Intelligence metrics
    print(f"\n  --- Intelligence Metrics ---")
    print(f"  1. Solar Charging:")
    print(f"     EV charge rate during high solar:  {ev_charge_during_solar_rate*100:.1f}%")
    print(f"     Mean EV action (high solar):       {mean_ev_act_solar:+.3f}")
    print(f"     Mean EV action (no solar):         {mean_ev_act_nosolar:+.3f}")
    print(f"     Solar preference (diff):           {solar_diff:+.3f}")

    print(f"\n  2. Price-Aware V2G (price > P75={p75:.4f}):")
    print(f"     Batt discharge rate:  {batt_discharge_expensive*100:.1f}%  (mean={mean_batt_expensive:+.3f})")
    print(f"     EV discharge rate:    {ev_discharge_expensive*100:.1f}%  (mean={mean_ev_expensive:+.3f})")

    print(f"\n  3. Off-Peak Charging (price < P25={p25:.4f}):")
    print(f"     Batt charge rate:     {batt_charge_cheap*100:.1f}%  (mean={mean_batt_cheap:+.3f})")
    print(f"     EV charge rate:       {ev_charge_cheap*100:.1f}%  (mean={mean_ev_cheap:+.3f})")
    print(f"     Batt price spread:    {price_batt_diff:+.3f}")

    print(f"\n  4. Temporal Planning:")
    print(f"     Peak hours:       {sorted(peak_hours)}")
    print(f"     Pre-peak hours:   {sorted(pre_peak_hours)}")
    print(f"     Batt pre-peak:    {pre_peak_batt:+.3f}")
    print(f"     Batt at peak:     {peak_batt:+.3f}")
    print(f"     Swing:            {pre_peak_batt - peak_batt:+.3f}")

    print(f"\n  5. Correlations:")
    print(f"     Batt vs Price:    r={corr_batt_price:+.3f}")
    print(f"     EV vs Price:      r={corr_ev_price:+.3f}")
    print(f"     EV vs Solar:      r={corr_ev_solar:+.3f}")

    print(f"\n  6. Behavioral Diversity:")
    print(f"     Batt charge hours:    {n_batt_charge_hours}/24")
    print(f"     Batt discharge hours: {n_batt_discharge_hours}/24")
    print(f"     Batt idle hours:      {n_batt_idle_hours}/24")
    print(f"     Hourly action StdDev: {action_variance:.3f}")

    print(f"\n  7. Forecast Utilization:")
    print(f"     Price sensitivity:    {forecast_results['price_sensitivity']:.6f} (score={forecast_results['price_score']:.2f})")
    print(f"     Solar sensitivity:    {forecast_results['solar_sensitivity']:.6f} (score={forecast_results['solar_score']:.2f})")
    print(f"     All sensitivity:      {forecast_results['all_sensitivity']:.6f} (score={forecast_results['all_score']:.2f})")
    print(f"     Forecast util score:  {forecast_results['forecast_utilization']:.2f}")

    print(f"\n  8. Hourly Action Profile:")
    print(f"     Hour  Batt    EV     Price   Solar   Interpretation")
    print(f"     {'---'*20}")
    for h in range(24):
        b = hourly_batt_profile[h]
        e = hourly_ev_profile[h]
        p = hourly_price_profile[h]
        s = hourly_solar_profile[h]
        interp = ""
        if b > 0.05:
            interp += "B:charge "
        elif b < -0.05:
            interp += "B:discharge "
        if e > 0.1:
            interp += "EV:charge "
        elif e < -0.05:
            interp += "EV:V2G "
        if s > solar_med and s > 0:
            interp += "[solar] "
        if p > p75:
            interp += "[PEAK$] "
        elif p < p25:
            interp += "[cheap$] "
        print(f"     {h:02d}:00 {b:+.3f}  {e:+.3f}  {p:.4f}  {s:5.2f}   {interp}")

    # Intelligence scorecard
    print(f"\n  9. Intelligence Scorecard:")
    print(f"     {'Metric':<25s}  {'Score':>6s}  {'Rating'}")
    print(f"     {'---'*20}")
    for sname, sval in scores.items():
        rating = "***" if sval > 0.7 else "** " if sval > 0.4 else "*  " if sval > 0.1 else ".  "
        bar_len = int(sval * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"     {sname:<25s}  {sval:5.2f}   {bar} {rating}")
    print(f"\n     {'OVERALL INTELLIGENCE':<25s}  {overall_intelligence:5.2f}   "
          f"{'INTELLIGENT' if overall_intelligence > 0.5 else 'LEARNING' if overall_intelligence > 0.3 else 'DUMB'}")

    # Energy balance
    print(f"\n  Energy: import={total_import:.0f} export={total_export:.0f} net={total_import-total_export:.0f} solar={total_solar:.0f} kWh")
    print(f"  Solar-EV charging: {ev_solar_charging_pct:.1f}% | Solar-EV correlation (hourly): {ev_solar_corr_hourly:.3f}")
    print(f"  Solar-Batt charging: {batt_solar_charging_pct:.1f}% | Solar-Batt correlation (hourly): {batt_solar_corr_hourly:.3f}")

    # =====================================================================
    # BUILD RESULTS DICT
    # =====================================================================
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
    cl_kpis = {}
    for raw_key, short_key in kpi_name_map.items():
        val = citylearn_kpis.get(raw_key, float("nan"))
        cl_kpis[short_key] = round(val, 6) if np.isfinite(val) else None

    results = {
        "run": run_name,
        "description": description,
        "checkpoint": ckpt_path,
        "seed": seed,
        "steps": step,
        "total_reward": round(total_reward, 2),
        "total_cost": round(total_cost, 2),
        "citylearn_kpis": cl_kpis,
        "constraints": {
            "C0_ev_departure": {"cost": round(total_c0, 2), "limit": COST_LIMITS["C0_ev_departure"]},
            "C1_ev_saute": {"cost": round(total_c1, 2), "limit": COST_LIMITS["C1_ev_saute"]},
            "C2_battery_soc": {"cost": round(total_c2, 2), "limit": COST_LIMITS["C2_battery_soc"]},
            "C3_building_power": {"cost": round(total_c3, 2), "limit": COST_LIMITS["C3_building_power"]},
            "C4_grid_power": {"cost": round(total_c4, 2), "limit": COST_LIMITS["C4_grid_power"]},
        },
        "violation_rates_pct": {
            "C1_ev_steps": round(c1_step_rate, 2),
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
        "intelligence": {
            "solar_charging": {
                "ev_charge_rate_high_solar": round(ev_charge_during_solar_rate, 4),
                "mean_ev_action_solar": round(mean_ev_act_solar, 4),
                "mean_ev_action_nosolar": round(mean_ev_act_nosolar, 4),
                "solar_preference_diff": round(solar_diff, 4),
            },
            "price_aware_v2g": {
                "batt_discharge_expensive_rate": round(batt_discharge_expensive, 4),
                "mean_batt_expensive": round(mean_batt_expensive, 4),
                "ev_discharge_expensive_rate": round(ev_discharge_expensive, 4),
                "mean_ev_expensive": round(mean_ev_expensive, 4),
            },
            "off_peak_charging": {
                "batt_charge_cheap_rate": round(batt_charge_cheap, 4),
                "mean_batt_cheap": round(mean_batt_cheap, 4),
                "ev_charge_cheap_rate": round(ev_charge_cheap, 4),
                "mean_ev_cheap": round(mean_ev_cheap, 4),
                "batt_price_spread": round(price_batt_diff, 4),
            },
            "temporal_planning": {
                "peak_hours": sorted(peak_hours),
                "pre_peak_hours": sorted(pre_peak_hours),
                "pre_peak_batt": round(pre_peak_batt, 4),
                "peak_batt": round(peak_batt, 4),
                "swing": round(pre_peak_batt - peak_batt, 4),
            },
            "correlations": {
                "batt_vs_price": round(corr_batt_price, 4),
                "ev_vs_price": round(corr_ev_price, 4),
                "ev_vs_solar": round(corr_ev_solar, 4),
                "ev_solar_hourly": round(ev_solar_corr_hourly, 4),
                "batt_solar_hourly": round(batt_solar_corr_hourly, 4),
            },
            "behavioral_diversity": {
                "batt_charge_hours": n_batt_charge_hours,
                "batt_discharge_hours": n_batt_discharge_hours,
                "batt_idle_hours": n_batt_idle_hours,
                "hourly_action_stddev": round(action_variance, 4),
            },
            "forecast": {
                "price_sensitivity": round(forecast_results["price_sensitivity"], 6),
                "solar_sensitivity": round(forecast_results["solar_sensitivity"], 6),
                "all_sensitivity": round(forecast_results["all_sensitivity"], 6),
                "price_score": round(forecast_results["price_score"], 4),
                "solar_score": round(forecast_results["solar_score"], 4),
                "all_score": round(forecast_results["all_score"], 4),
                "forecast_utilization": round(forecast_results["forecast_utilization"], 4),
            },
        },
        "scorecard": {k: round(v, 4) for k, v in scores.items()},
        "overall_intelligence": round(overall_intelligence, 4),
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
        "solar_ev": {
            "ev_solar_charging_pct": round(ev_solar_charging_pct, 2),
            "batt_solar_charging_pct": round(batt_solar_charging_pct, 2),
            "solar_ev_correlation_hourly": round(ev_solar_corr_hourly, 3),
            "solar_batt_correlation_hourly": round(batt_solar_corr_hourly, 3),
        },
        "hourly_profiles": {
            "batt": [round(float(v), 4) for v in hourly_batt_profile],
            "ev": [round(float(v), 4) for v in hourly_ev_profile],
            "price": [round(float(v), 4) for v in hourly_price_profile],
            "solar": [round(float(v), 4) for v in hourly_solar_profile],
        },
        "batt_clamp_count": batt_clamp_count,
        "batt_clamp_pct": round(pct(batt_clamp_count, step), 2),
    }

    print(f"\n  Evaluation complete for {run_name}.")
    return results


# =========================================================================
# Comparison tables
# =========================================================================
def print_comparison_tables(all_results):
    """Print comprehensive side-by-side comparison tables."""
    runs = [r for r in DISPLAY_ORDER if r in all_results]
    for r in all_results:
        if r not in runs:
            runs.append(r)

    if not runs:
        print("No results to compare.")
        return

    # Column layout
    label_w = 32
    col_w = 13
    n_cols = len(runs)
    total_w = label_w + col_w * n_cols + 4

    LABELS = {
        "no_control": "NoCtrl",
        "rbc": "RBC",
        "r19": "R19",
        "r21": "R21",
        "r22": "R22",
        "r23": "R23",
    }

    def hline(char="="):
        return char * total_w

    def header_row():
        row = f"  {'Metric':<{label_w}}"
        for r in runs:
            lbl = LABELS.get(r, r.upper())
            row += f"{lbl:>{col_w}}"
        return row

    def data_row(label, values, fmt=".3f"):
        row = f"  {label:<{label_w}}"
        for v in values:
            if v is None:
                row += f"{'N/A':>{col_w}}"
            elif isinstance(v, str):
                row += f"{v:>{col_w}}"
            else:
                # Format value first, then right-align
                formatted = f"{v:{fmt}}"
                row += f"{formatted:>{col_w}}"
        return row

    def get_val(run, *keys, default=None):
        d = all_results.get(run, {})
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d

    # =====================================================================
    # Print header
    # =====================================================================
    print(f"\n\n{'#'*total_w}")
    print(f"{'#'*total_w}")
    print(f"{'COMPREHENSIVE RUN COMPARISON':^{total_w}}")
    print(f"{'#'*total_w}")
    print(f"{'#'*total_w}")

    # Description row
    print(f"\n  Run descriptions:")
    for r in runs:
        desc = get_val(r, "description", default=r)
        print(f"    {LABELS.get(r, r):>8}: {desc}")

    # =====================================================================
    # TABLE 1: CityLearn KPIs
    # =====================================================================
    print(f"\n{hline()}")
    print(f"  TABLE 1: CityLearn KPIs (lower = better, 1.0 = no-control baseline)")
    print(f"{hline()}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    kpi_keys = [
        "electricity_consumption", "carbon_emissions", "cost", "daily_peak_avg",
        "all_time_peak_avg", "ramping_avg", "discomfort", "zero_net_energy",
    ]
    for kpi in kpi_keys:
        vals = [get_val(r, "citylearn_kpis", kpi) for r in runs]
        print(data_row(kpi, vals))

    # =====================================================================
    # TABLE 2: Constraint Costs
    # =====================================================================
    print(f"\n{hline()}")
    print(f"  TABLE 2: Constraint Costs (raw EpCost + violation % steps)")
    print(f"{hline()}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    constraint_keys = ["C0_ev_departure", "C1_ev_saute", "C2_battery_soc", "C3_building_power", "C4_grid_power"]
    for ck in constraint_keys:
        limit = COST_LIMITS[ck]
        vals = [get_val(r, "constraints", ck, "cost") for r in runs]
        print(data_row(f"{ck} (cost)", vals, fmt=".1f"))

    print()
    for ck in constraint_keys:
        limit = COST_LIMITS[ck]
        vals = []
        for r in runs:
            c = get_val(r, "constraints", ck, "cost")
            if c is not None:
                vals.append(c / limit)
            else:
                vals.append(None)
        print(data_row(f"{ck} (ratio)", vals, fmt=".2f"))

    print()
    print(f"  {'Step-level violation rates (%):'}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)
    for vk in ["C1_ev_steps", "C2_soc_steps", "C3_building_steps", "C4_grid_steps"]:
        vals = [get_val(r, "violation_rates_pct", vk) for r in runs]
        print(data_row(vk, vals, fmt=".2f"))

    # =====================================================================
    # TABLE 3: EV Departure Reliability
    # =====================================================================
    print(f"\n{hline()}")
    print(f"  TABLE 3: EV Departure Reliability")
    print(f"{hline()}")
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

    # =====================================================================
    # TABLE 4: Intelligence Scorecard
    # =====================================================================
    print(f"\n{hline()}")
    print(f"  TABLE 4: Intelligence Scorecard (0-1, higher = better)")
    print(f"{hline()}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    score_names = [
        "solar_preference", "price_spread", "v2g_timing", "ev_compliance",
        "grid_stability", "pre_peak_planning", "price_correlation",
        "behavioral_diversity", "forecast_utilization",
    ]
    for sn in score_names:
        vals = [get_val(r, "scorecard", sn) for r in runs]
        print(data_row(sn, vals, fmt=".3f"))

    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)
    vals = [get_val(r, "overall_intelligence") for r in runs]
    print(data_row("OVERALL INTELLIGENCE", vals, fmt=".3f"))

    # Best intelligence
    intel_vals = {r: get_val(r, "overall_intelligence") for r in runs if get_val(r, "overall_intelligence") is not None}
    if intel_vals:
        best_intel = max(intel_vals, key=intel_vals.get)
        print(f"  --> Best: {LABELS.get(best_intel, best_intel)} ({intel_vals[best_intel]:.3f})")

    # =====================================================================
    # TABLE 5: Temporal Planning Summary
    # =====================================================================
    print(f"\n{hline()}")
    print(f"  TABLE 5: Temporal Planning Summary")
    print(f"{hline()}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    for metric, keys, fmt in [
        ("pre_peak_batt", ("intelligence", "temporal_planning", "pre_peak_batt"), "+.3f"),
        ("peak_batt", ("intelligence", "temporal_planning", "peak_batt"), "+.3f"),
        ("swing", ("intelligence", "temporal_planning", "swing"), "+.3f"),
        ("batt_vs_price corr", ("intelligence", "correlations", "batt_vs_price"), "+.3f"),
        ("ev_vs_price corr", ("intelligence", "correlations", "ev_vs_price"), "+.3f"),
        ("ev_vs_solar corr", ("intelligence", "correlations", "ev_vs_solar"), "+.3f"),
        ("batt_charge_hours", ("intelligence", "behavioral_diversity", "batt_charge_hours"), ".0f"),
        ("batt_discharge_hours", ("intelligence", "behavioral_diversity", "batt_discharge_hours"), ".0f"),
        ("hourly_action_stddev", ("intelligence", "behavioral_diversity", "hourly_action_stddev"), ".4f"),
    ]:
        vals = [get_val(r, *keys) for r in runs]
        print(data_row(metric, vals, fmt=fmt))

    # =====================================================================
    # TABLE 6: Energy Balance
    # =====================================================================
    print(f"\n{hline()}")
    print(f"  TABLE 6: Energy Balance (kWh)")
    print(f"{hline()}")
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

    # =====================================================================
    # TABLE 7: V2G Behavior
    # =====================================================================
    print(f"\n{hline()}")
    print(f"  TABLE 7: V2G Behavior (%)")
    print(f"{hline()}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    for metric, keys, fmt in [
        ("charge (%)", ("v2g", "charge_pct"), ".1f"),
        ("discharge (%)", ("v2g", "discharge_pct"), ".1f"),
        ("idle (%)", ("v2g", "idle_pct"), ".1f"),
        ("EV solar charging (%)", ("solar_ev", "ev_solar_charging_pct"), ".1f"),
        ("Batt solar charging (%)", ("solar_ev", "batt_solar_charging_pct"), ".1f"),
        ("Solar-EV corr (hourly)", ("solar_ev", "solar_ev_correlation_hourly"), ".3f"),
        ("Solar-Batt corr (hourly)", ("solar_ev", "solar_batt_correlation_hourly"), ".3f"),
    ]:
        vals = [get_val(r, *keys) for r in runs]
        print(data_row(metric, vals, fmt=fmt))

    # =====================================================================
    # TABLE 8: Forecast Sensitivity
    # =====================================================================
    print(f"\n{hline()}")
    print(f"  TABLE 8: Forecast Sensitivity (perturbation test)")
    print(f"{hline()}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    for metric, keys, fmt in [
        ("price sensitivity", ("intelligence", "forecast", "price_sensitivity"), ".6f"),
        ("price score", ("intelligence", "forecast", "price_score"), ".3f"),
        ("solar sensitivity", ("intelligence", "forecast", "solar_sensitivity"), ".6f"),
        ("solar score", ("intelligence", "forecast", "solar_score"), ".3f"),
        ("all sensitivity", ("intelligence", "forecast", "all_sensitivity"), ".6f"),
        ("all score", ("intelligence", "forecast", "all_score"), ".3f"),
        ("forecast utilization", ("intelligence", "forecast", "forecast_utilization"), ".3f"),
    ]:
        vals = [get_val(r, *keys) for r in runs]
        print(data_row(metric, vals, fmt=fmt))

    # =====================================================================
    # Episode Summary
    # =====================================================================
    print(f"\n{hline()}")
    print(f"  Episode Summary")
    print(f"{hline()}")
    print(header_row())
    print(f"  {'-'*label_w}" + f"{'-'*col_w}" * n_cols)

    for metric, key, fmt in [
        ("total_reward", "total_reward", ".0f"),
        ("total_cost", "total_cost", ".0f"),
        ("steps", "steps", ".0f"),
        ("batt_clamp_pct", "batt_clamp_pct", ".1f"),
    ]:
        vals = [get_val(r, key) for r in runs]
        print(data_row(metric, vals, fmt=fmt))

    print(f"\n{hline()}")

    # =====================================================================
    # Key Takeaways
    # =====================================================================
    print(f"\n  Key Takeaways:")

    # Best reward
    reward_vals = {r: get_val(r, "total_reward") for r in runs if get_val(r, "total_reward") is not None}
    if reward_vals:
        best_r = max(reward_vals, key=reward_vals.get)
        print(f"    Best reward:             {LABELS.get(best_r, best_r)} ({reward_vals[best_r]:.0f})")

    # Best EV compliance
    ev_vals = {r: get_val(r, "ev_departures", "violation_pct")
               for r in runs if get_val(r, "ev_departures", "violation_pct") is not None}
    if ev_vals:
        best_ev = min(ev_vals, key=ev_vals.get)
        print(f"    Best EV compliance:      {LABELS.get(best_ev, best_ev)} ({ev_vals[best_ev]:.1f}% violation)")

    # Best C3
    c3_vals_comp = {r: (get_val(r, "constraints", "C3_building_power", "cost") or 0) / COST_LIMITS["C3_building_power"]
                    for r in runs if get_val(r, "constraints", "C3_building_power", "cost") is not None}
    if c3_vals_comp:
        best_c3 = min(c3_vals_comp, key=c3_vals_comp.get)
        print(f"    Best C3 (building):      {LABELS.get(best_c3, best_c3)} ({c3_vals_comp[best_c3]:.2f}x limit)")

    # Best C4
    c4_vals_comp = {r: (get_val(r, "constraints", "C4_grid_power", "cost") or 0) / COST_LIMITS["C4_grid_power"]
                    for r in runs if get_val(r, "constraints", "C4_grid_power", "cost") is not None}
    if c4_vals_comp:
        best_c4 = min(c4_vals_comp, key=c4_vals_comp.get)
        print(f"    Best C4 (grid):          {LABELS.get(best_c4, best_c4)} ({c4_vals_comp[best_c4]:.2f}x limit)")

    # Best intelligence
    if intel_vals:
        best_intel = max(intel_vals, key=intel_vals.get)
        print(f"    Best intelligence:       {LABELS.get(best_intel, best_intel)} ({intel_vals[best_intel]:.3f})")

    # Best solar-EV correlation
    solar_corr = {r: get_val(r, "solar_ev", "solar_ev_correlation_hourly")
                  for r in runs if get_val(r, "solar_ev", "solar_ev_correlation_hourly") is not None}
    if solar_corr:
        best_solar = max(solar_corr, key=solar_corr.get)
        print(f"    Best solar-EV corr:      {LABELS.get(best_solar, best_solar)} ({solar_corr[best_solar]:.3f})")

    # Best forecast utilization (RL only)
    fc_vals = {r: get_val(r, "intelligence", "forecast", "forecast_utilization")
               for r in runs if get_val(r, "intelligence", "forecast", "forecast_utilization") is not None
               and get_val(r, "intelligence", "forecast", "forecast_utilization") > 0}
    if fc_vals:
        best_fc = max(fc_vals, key=fc_vals.get)
        print(f"    Best forecast util:      {LABELS.get(best_fc, best_fc)} ({fc_vals[best_fc]:.3f})")

    print()


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation and comparison for Safe-CityLearn runs (thesis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/eval_full_comparison.py                    # evaluate all runs
    python scripts/eval_full_comparison.py --run r21          # single run
    python scripts/eval_full_comparison.py --run all          # all runs (explicit)
    python scripts/eval_full_comparison.py --seed 42 --epoch 89
        """,
    )
    parser.add_argument(
        "--run", default="all",
        choices=list(RUNS.keys()) + ["all"],
        help="Which run to evaluate, or 'all' for everything + comparison (default: all)",
    )
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific checkpoint epoch (default: latest)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for evaluation (default: 42)")

    args = parser.parse_args()

    # Set global seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.run == "all":
        runs_to_eval = DISPLAY_ORDER
    else:
        runs_to_eval = [args.run]

    all_results = {}
    for run_name in runs_to_eval:
        # Reset seeds before each run for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        result = evaluate(run_name, epoch=args.epoch, seed=args.seed)
        if result is not None:
            all_results[run_name] = result

    # Print comparison tables if more than one run
    if len(all_results) > 1:
        print_comparison_tables(all_results)

    # Save comprehensive JSON
    out_path = os.path.join(PROJECT, "runs", "full_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Convert for JSON serialization: ensure all values are JSON-safe
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Comprehensive results saved to: {out_path}")
    print(f"  Evaluated {len(all_results)} runs: {list(all_results.keys())}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
