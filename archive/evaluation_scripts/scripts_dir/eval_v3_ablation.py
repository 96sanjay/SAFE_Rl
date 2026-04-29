#!/usr/bin/env python3
"""
V3 Ablation Evaluation: Compare 4 agents side by side.

Agents:
  1. R5a          - MLP baseline, 198-dim obs, no spatial, 50 epochs
  2. PPOLagMulti v2 - 218-dim obs, spatial, 50 epochs
  3. PPOLagMulti v3a - 218-dim obs, spatial, 100 epochs, 5 constraints
  4. PPOLagMulti v3b - 218-dim obs, spatial, 100 epochs, C1 disabled

Outputs:
  - Per-constraint violation percentages (C0-C4)
  - Per-constraint cost totals
  - Total STEMS reward
  - CityLearn standard KPIs
  - Intelligence metrics (solar, price, V2G, compliance, etc.)
  - Full comparison table with deltas vs R5a baseline

Usage:
  python scripts/eval_v3_ablation.py
"""
import os
import sys
import json
import glob
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

EVAL_SEED = 42
OUT_DIR = f"{PROJECT}/runs/multi_lag_v3_ablation/evaluation"

# =========================================================================
# Environment profiles
# =========================================================================
COMMON_ENV_VARS = {
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
}

PROFILE_198 = {
    **COMMON_ENV_VARS,
    "CITYLEARN_C3_CONTROLLABLE": "0",
    "CITYLEARN_SPATIAL_OBS": "0",
}

PROFILE_218 = {
    **COMMON_ENV_VARS,
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_SPATIAL_OBS": "1",
    "COST_W_C1": "10.0",
    "COST_W_C1_DENSE": "5.0",
    "COST_W_C2": "1.0",
    "COST_W_C3": "0.1",
    "COST_W_C4": "5.0",
    "CITYLEARN_EV_DENSE_COST_SCALE": "1.0",
    "STEMS_ALPHA_GRID": "1.0",
    "STEMS_BETA_RAMP": "2.0",
}

# Keys that must be cleared when switching profiles to avoid stale values
STALE_KEYS = [
    "CITYLEARN_C3_CONTROLLABLE", "CITYLEARN_SPATIAL_OBS",
    "COST_W_C1", "COST_W_C1_DENSE", "COST_W_C2", "COST_W_C3", "COST_W_C4",
    "CITYLEARN_EV_DENSE_COST_SCALE", "STEMS_ALPHA_GRID", "STEMS_BETA_RAMP",
]


def set_env_profile(profile: dict):
    """Set environment variables for an agent profile, clearing stale keys first."""
    for k in STALE_KEYS:
        os.environ.pop(k, None)
    for k, v in profile.items():
        os.environ[k] = v


# Set initial profile so imports work
set_env_profile(PROFILE_198)

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.stems_obs_wrapper import SpatialGraphFeaturesWrapper
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env
import citylearn_safe.schema_index as si


# =========================================================================
# Agent definitions
# =========================================================================
AGENTS = {
    "R5a (MLP)": {
        "ckpt_glob": f"{PROJECT}/runs/r6_compare/r5a_5bld/PPOLag-{{CityLearnSafety-V2G-v2}}/seed-*/torch_save/epoch-50.pt",
        "obs_dim": 198,
        "profile": PROFILE_198,
        "use_spatial": False,
    },
    "PPOLagMulti v2": {
        "ckpt_glob": f"{PROJECT}/runs/multi_lag_v2/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-*/torch_save/epoch-50.pt",
        "obs_dim": 218,
        "profile": PROFILE_218,
        "use_spatial": True,
    },
    "PPOLagMulti v3a": {
        "ckpt_glob": f"{PROJECT}/runs/multi_lag_v3a/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-*/torch_save/epoch-100.pt",
        "obs_dim": 218,
        "profile": PROFILE_218,
        "use_spatial": True,
    },
    "PPOLagMulti v3b": {
        "ckpt_glob": f"{PROJECT}/runs/multi_lag_v3b/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-*/torch_save/epoch-100.pt",
        "obs_dim": 218,
        "profile": PROFILE_218,
        "use_spatial": True,
    },
    "PPOLagMulti v5": {
        "ckpt_glob": f"{PROJECT}/runs/multi_lag_v5/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-*/torch_save/epoch-50.pt",
        "obs_dim": 218,
        "profile": PROFILE_218,
        "use_spatial": True,
    },
}


# =========================================================================
# Actor model
# =========================================================================
class MLPActor(nn.Module):
    """Simple MLP actor with tanh output for deterministic evaluation."""
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


def load_actor(ckpt_path, obs_dim, act_dim):
    """Load actor weights and obs normalizer from an OmniSafe checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi_state = ckpt["pi"]

    h1 = pi_state["mean.0.weight"].shape[0]
    h2 = pi_state["mean.2.weight"].shape[0]
    actual_obs_dim = pi_state["mean.0.weight"].shape[1]
    actual_act_dim = pi_state["mean.4.weight"].shape[0]

    if actual_obs_dim != obs_dim:
        print(f"  WARNING: expected obs_dim={obs_dim}, checkpoint has {actual_obs_dim}. Using checkpoint value.")
        obs_dim = actual_obs_dim
    if actual_act_dim != act_dim:
        print(f"  WARNING: expected act_dim={act_dim}, checkpoint has {actual_act_dim}. Using checkpoint value.")
        act_dim = actual_act_dim

    actor = MLPActor(obs_dim, act_dim, (h1, h2))
    filtered = {k: v for k, v in pi_state.items() if not k.startswith("log_std")}
    result = actor.load_state_dict(filtered, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Missing keys in actor state_dict: {result.missing_keys}")
    actor.eval()

    obs_mean = obs_std = obs_clip = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        clip_t = norm["_clip"].float()
        obs_clip = float(clip_t.mean())

    return actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim


def discover_checkpoint(ckpt_glob):
    """Find the latest checkpoint matching a glob pattern. Returns path or None."""
    matches = sorted(glob.glob(ckpt_glob))
    if not matches:
        return None
    # Return the last match (latest seed timestamp)
    return matches[-1]


# =========================================================================
# Episode runner (KPI + cost + intelligence data collection)
# =========================================================================
def run_full_episode(actor, obs_mean, obs_std, obs_clip, label, use_spatial=False, seed=42):
    """
    Run one deterministic episode. Collects all data needed for:
    - Per-constraint costs and violation rates
    - CityLearn standard KPIs
    - Intelligence metrics
    Returns (kpi_results, intelligence_data, price_percentiles).
    """
    # Reset schema index cache to avoid stale obs parsing
    si._CACHE = None

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)

    if use_spatial:
        p_bmax = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
        env = SpatialGraphFeaturesWrapper(env, num_buildings=5, p_building_max=p_bmax)

    # Get raw CityLearn env for evaluate() and building data
    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(getattr(raw, "buildings", []))

    # Parse action indices
    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)
    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]

    # Get price percentiles for intelligence analysis
    try:
        pr = buildings[0].pricing.electricity_pricing
        all_prices = np.array(pr, dtype=float)
        p25 = float(np.percentile(all_prices, 25))
        p50 = float(np.percentile(all_prices, 50))
        p75 = float(np.percentile(all_prices, 75))
    except Exception:
        p25, p50, p75 = 0.12, 0.16, 0.20

    # Walk through the city wrapper chain to find evaluate()
    city = base
    for _ in range(20):
        if hasattr(city, 'buildings') and len(getattr(city, 'buildings', [])) > 0:
            break
        city = getattr(city, 'env', getattr(city, 'base', getattr(city, 'unwrapped', None)))

    # Run episode
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    rewards = []
    costs = []
    actions_all = []

    c0_vals, c1_vals, c2_vals, c3_vals, c4_vals = [], [], [], [], []
    c0_steps = c1_steps = c2_steps = c3_steps = c4_steps = 0

    # Intelligence tracking data
    intel_data = {
        "hour": [], "price": [], "solar": [],
        "actions": [], "ev_actions": [], "batt_actions": [],
        "c0": [], "c1": [], "c2": [], "c3": [], "c4": [],
        "net_load": [], "reward": [], "obs_raw": [],
    }
    hourly_actions = {h: [] for h in range(24)}
    hourly_batt = {h: [] for h in range(24)}
    hourly_ev = {h: [] for h in range(24)}

    while not done:
        obs_np = np.asarray(obs, dtype=np.float32)
        intel_data["obs_raw"].append(obs_np.copy())

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
        hour = t_now % 24
        hourly_actions[hour].append(action.copy())
        if batt_idx:
            hourly_batt[hour].append(np.mean([action[i] for i in batt_idx]))
        if ev_idx:
            hourly_ev[hour].append(np.mean([action[i] for i in ev_idx]))

        # Price and solar for intelligence
        try:
            price = float(buildings[0].pricing.electricity_pricing[max(0, t_now - 1)])
        except Exception:
            price = 0.17
        solar_total = 0.0
        for b in buildings:
            try:
                s = getattr(b, "solar_generation", None)
                if s is not None and len(s) > max(0, t_now - 1):
                    solar_total += abs(float(s[max(0, t_now - 1)]))
            except Exception:
                pass

        obs, reward, term, trunc, info = env.step(action)
        done = bool(term) or bool(trunc)
        step += 1

        r = float(reward)
        c = float(info.get("cost", 0.0))
        rewards.append(r)
        costs.append(c)

        c0 = float(info.get("cost_ev_dense", 0.0))
        c1 = float(info.get("cost_ev_departure", 0.0))
        c2 = float(info.get("cost_stems_battery", 0.0))
        c3 = float(info.get("cost_stems_building_power", 0.0))
        c4 = float(info.get("cost_stems_grid_power", 0.0))

        c0_vals.append(c0)
        c1_vals.append(c1)
        c2_vals.append(c2)
        c3_vals.append(c3)
        c4_vals.append(c4)

        if c0 > 0: c0_steps += 1
        if c1 > 0: c1_steps += 1
        if c2 > 0: c2_steps += 1
        if c3 > 0: c3_steps += 1
        if c4 > 0: c4_steps += 1

        # Intelligence data
        intel_data["hour"].append(hour)
        intel_data["price"].append(price)
        intel_data["solar"].append(solar_total)
        intel_data["actions"].append(action.copy())
        intel_data["ev_actions"].append([float(action[i]) for i in ev_idx])
        intel_data["batt_actions"].append([float(action[i]) for i in batt_idx])
        intel_data["c0"].append(c0)
        intel_data["c1"].append(c1)
        intel_data["c2"].append(c2)
        intel_data["c3"].append(c3)
        intel_data["c4"].append(c4)
        intel_data["reward"].append(r)

        net = 0.0
        for b in buildings:
            try:
                nec = getattr(b, "net_electricity_consumption", None)
                if nec is not None and len(nec) > max(0, t_now - 1):
                    net += float(nec[max(0, t_now - 1)])
            except Exception:
                pass
        intel_data["net_load"].append(net)

    # Convert intelligence data to numpy
    for k in intel_data:
        intel_data[k] = np.array(intel_data[k])

    # CityLearn evaluate()
    cl_kpis = {}
    try:
        if city and hasattr(city, 'evaluate'):
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

    actions_arr = np.array(actions_all)
    total_steps = step

    # Hourly profiles
    hourly_batt_means = np.array([np.mean(hourly_batt[h]) if hourly_batt[h] else 0 for h in range(24)])
    hourly_ev_means = np.array([np.mean(hourly_ev[h]) if hourly_ev[h] else 0 for h in range(24)])

    kpi_results = {
        "label": label,
        "total_steps": total_steps,

        # Reward
        "total_reward": float(sum(rewards)),
        "avg_reward": float(sum(rewards) / total_steps),

        # Total CMDP cost
        "total_cost": float(sum(costs)),
        "avg_cost": float(sum(costs) / total_steps),
        "cost_positive_steps": int(sum(1 for c in costs if c > 0)),
        "cost_positive_pct": float(100.0 * sum(1 for c in costs if c > 0) / total_steps),

        # Per-constraint cost totals
        "c0_ev_dense_total": float(sum(c0_vals)),
        "c1_ev_departure_total": float(sum(c1_vals)),
        "c2_battery_soc_total": float(sum(c2_vals)),
        "c3_building_power_total": float(sum(c3_vals)),
        "c4_grid_power_total": float(sum(c4_vals)),

        # Violation percentages
        "c0_ev_dense_viol_pct": float(100.0 * c0_steps / total_steps),
        "c1_ev_departure_viol_pct": float(100.0 * c1_steps / total_steps),
        "c2_battery_soc_viol_pct": float(100.0 * c2_steps / total_steps),
        "c3_building_power_viol_pct": float(100.0 * c3_steps / total_steps),
        "c4_grid_power_viol_pct": float(100.0 * c4_steps / total_steps),

        # Action statistics
        "action_mean": float(actions_arr.mean()),
        "action_std": float(actions_arr.std()),
        "action_abs_mean": float(np.abs(actions_arr).mean()),
        "batt_mean": float(np.mean([actions_arr[:, i].mean() for i in batt_idx])) if batt_idx else 0.0,
        "ev_mean": float(np.mean([actions_arr[:, i].mean() for i in ev_idx])) if ev_idx else 0.0,

        # Hourly profiles
        "hourly_batt": hourly_batt_means.tolist(),
        "hourly_ev": hourly_ev_means.tolist(),

        # CityLearn KPIs
        **{f"cl_{k}": v for k, v in cl_kpis.items()},

        # Determinism hashes
        "_reward_hash": float(np.array(rewards).sum()),
        "_cost_hash": float(np.array(costs).sum()),
        "_action_hash": float(actions_arr.sum()),
    }

    env.close()
    return kpi_results, intel_data, (p25, p50, p75)


# =========================================================================
# Intelligence analysis
# =========================================================================
def compute_forecast_utilization(data, actor, obs_mean, obs_std, obs_clip):
    """Perturbation test: zero out forecast dims and measure action change."""
    obs_raw = data["obs_raw"]
    actions_orig = data["actions"]
    obs_dim = obs_raw.shape[1]

    n_forecast = 128
    base_obs_dim = 70
    fc_start = base_obs_dim

    if fc_start + 72 > obs_dim:
        return {
            "forecast_utilization": 0.0,
            "price_sensitivity": 0.0,
            "solar_sensitivity": 0.0,
            "all_sensitivity": 0.0,
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

    deltas = {}
    for name, (start, end) in [
        ("price_forecast", price_range),
        ("solar_forecast", solar_range),
        ("all_forecast", all_range),
    ]:
        end = min(end, obs_dim)
        obs_perturbed = obs_raw.copy()
        obs_perturbed[:, start:end] = 0.0
        actions_perturbed = batched_forward(obs_perturbed)
        step_deltas = np.mean(np.abs(actions_orig - actions_perturbed), axis=1)
        deltas[name] = float(np.mean(step_deltas))

    price_score = float(np.clip(deltas["price_forecast"] / SCALE_THRESHOLD, 0, 1))
    solar_score = float(np.clip(deltas["solar_forecast"] / SCALE_THRESHOLD, 0, 1))
    all_score = float(np.clip(deltas["all_forecast"] / SCALE_THRESHOLD, 0, 1))
    forecast_utilization = float(np.mean([price_score, solar_score, all_score]))

    return {
        "forecast_utilization": forecast_utilization,
        "price_sensitivity": deltas["price_forecast"],
        "solar_sensitivity": deltas["solar_forecast"],
        "all_sensitivity": deltas["all_forecast"],
    }


def compute_intelligence_scores(data, p25, p50, p75, actor, obs_mean, obs_std, obs_clip):
    """
    Compute intelligence metrics from episode data.
    Returns dict of named scores (0-1 each) and an overall score.
    """
    hours = data["hour"]
    prices = data["price"]
    solar = data["solar"]
    ev_act = data["ev_actions"]
    batt_act = data["batt_actions"]
    c0, c1, c2, c3, c4 = data["c0"], data["c1"], data["c2"], data["c3"], data["c4"]
    rewards = data["reward"]
    N = len(hours)

    mean_ev = ev_act.mean(axis=1) if ev_act.ndim == 2 else ev_act
    mean_batt = batt_act.mean(axis=1) if batt_act.ndim == 2 else batt_act

    # 1. Solar preference
    solar_med = np.median(solar[solar > 0]) if (solar > 0).any() else 1.0
    high_solar = solar > solar_med
    mean_ev_act_solar = mean_ev[high_solar].mean() if high_solar.sum() > 0 else 0.0
    low_solar = solar < 0.01
    mean_ev_act_nosolar = mean_ev[low_solar].mean() if low_solar.sum() > 0 else 0.0
    solar_diff = mean_ev_act_solar - mean_ev_act_nosolar

    # 2. Price spread
    expensive = prices > p75
    cheap = prices < p25
    mean_batt_expensive = mean_batt[expensive].mean() if expensive.sum() > 0 else 0.0
    mean_batt_cheap = mean_batt[cheap].mean() if cheap.sum() > 0 else 0.0
    price_batt_diff = mean_batt_cheap - mean_batt_expensive

    # 3. V2G timing
    batt_discharge_expensive = (mean_batt[expensive] < -0.05).mean() if expensive.sum() > 0 else 0.0

    # 4. EV compliance
    c1_rate = float(100 * (c1 > 0).sum() / N)

    # 5. Grid stability
    c4_rate = float(100 * (c4 > 0).sum() / N)

    # 6. Pre-peak planning
    hourly_batt_arr = np.zeros(24)
    hourly_price_arr = np.zeros(24)
    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            hourly_batt_arr[h] = mean_batt[mask].mean()
            hourly_price_arr[h] = prices[mask].mean()

    peak_hours = {h for h in range(24) if hourly_price_arr[h] > p75}
    pre_peak_hours = set()
    for ph in peak_hours:
        for offset in [2, 3, 4]:
            pre_h = (ph - offset) % 24
            if pre_h not in peak_hours:
                pre_peak_hours.add(pre_h)

    if pre_peak_hours:
        pre_peak_mask = np.isin(hours, list(pre_peak_hours))
        pre_peak_batt = mean_batt[pre_peak_mask].mean() if pre_peak_mask.sum() > 0 else 0
        peak_mask = np.isin(hours, list(peak_hours))
        peak_batt = mean_batt[peak_mask].mean() if peak_mask.sum() > 0 else 0
    else:
        pre_peak_batt = peak_batt = 0

    # 7. Price correlation
    corr_batt_price = np.corrcoef(prices, mean_batt)[0, 1] if N > 10 else 0.0

    # 8. Behavioral diversity
    action_variance = np.std([hourly_batt_arr[h] for h in range(24)])

    # 9. Forecast utilization
    forecast_results = compute_forecast_utilization(data, actor, obs_mean, obs_std, obs_clip)

    # Compile scores (each 0-1, higher = better)
    scores = {
        "solar_preference": float(np.clip(solar_diff / 0.2, 0, 1)),
        "price_spread": float(np.clip(price_batt_diff / 0.3, 0, 1)),
        "v2g_timing": float(np.clip(batt_discharge_expensive, 0, 1)),
        "ev_compliance": float(np.clip(1.0 - c1_rate / 10.0, 0, 1)),
        "grid_stability": float(np.clip(1.0 - c4_rate / 30.0, 0, 1)),
        "pre_peak_planning": float(np.clip((pre_peak_batt - peak_batt) / 0.3, 0, 1)),
        "price_correlation": float(np.clip(-corr_batt_price / 0.3, 0, 1)),
        "behavioral_diversity": float(np.clip(action_variance / 0.15, 0, 1)),
        "forecast_utilization": forecast_results["forecast_utilization"],
    }

    overall = float(np.mean(list(scores.values())))

    # Raw values for detailed analysis
    details = {
        "solar_diff": float(solar_diff),
        "price_batt_diff": float(price_batt_diff),
        "batt_discharge_expensive_rate": float(batt_discharge_expensive),
        "c1_violation_rate_pct": float(c1_rate),
        "c4_violation_rate_pct": float(c4_rate),
        "pre_peak_swing": float(pre_peak_batt - peak_batt),
        "corr_batt_price": float(corr_batt_price),
        "hourly_action_stddev": float(action_variance),
        **forecast_results,
    }

    return scores, overall, details


# =========================================================================
# Main evaluation loop
# =========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start_time = time.time()

    print("=" * 80)
    print("  V3 ABLATION EVALUATION")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Seed: {EVAL_SEED}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 80)

    # =====================================================================
    # Discover checkpoints
    # =====================================================================
    available_agents = {}
    skipped_agents = []

    for agent_name, agent_cfg in AGENTS.items():
        ckpt_path = discover_checkpoint(agent_cfg["ckpt_glob"])
        if ckpt_path is None:
            print(f"\n  WARNING: No checkpoint found for {agent_name}")
            print(f"           Glob: {agent_cfg['ckpt_glob']}")
            print(f"           SKIPPING this agent.\n")
            skipped_agents.append(agent_name)
        else:
            available_agents[agent_name] = {**agent_cfg, "ckpt": ckpt_path}
            print(f"\n  Found: {agent_name}")
            print(f"         {ckpt_path}")

    if not available_agents:
        print("\nERROR: No agents found. Nothing to evaluate.")
        sys.exit(1)

    print(f"\n  Evaluating {len(available_agents)} agents, skipped {len(skipped_agents)}")

    # =====================================================================
    # Run evaluations
    # =====================================================================
    all_kpis = {}
    all_intel_scores = {}
    all_intel_details = {}

    for agent_name, agent_cfg in available_agents.items():
        print(f"\n{'=' * 70}")
        print(f"  EVALUATING: {agent_name}")
        print(f"  Checkpoint: {agent_cfg['ckpt']}")
        print(f"  Expected obs_dim: {agent_cfg['obs_dim']}, spatial: {agent_cfg['use_spatial']}")
        print(f"{'=' * 70}")

        set_env_profile(agent_cfg["profile"])

        actor, obs_mean, obs_std, obs_clip, actual_obs_dim, act_dim = load_actor(
            agent_cfg["ckpt"], agent_cfg["obs_dim"], 9
        )
        print(f"  Loaded: obs_dim={actual_obs_dim}, act_dim={act_dim}")
        if obs_mean is not None:
            print(f"  Obs normalizer: mean_norm={float(obs_mean.norm()):.2f}, std_mean={float(obs_std.mean()):.4f}")

        # Run 1: primary
        random.seed(EVAL_SEED)
        np.random.seed(EVAL_SEED)
        torch.manual_seed(EVAL_SEED)

        print(f"  Run 1 (primary)...")
        t0 = time.time()
        kpi_r1, intel_data, price_pcts = run_full_episode(
            actor, obs_mean, obs_std, obs_clip, agent_name,
            use_spatial=agent_cfg["use_spatial"], seed=EVAL_SEED
        )
        t1 = time.time()
        print(f"  Run 1 done in {t1 - t0:.1f}s  (reward={kpi_r1['total_reward']:.0f}, cost={kpi_r1['total_cost']:.0f})")

        # Run 2: determinism check
        random.seed(EVAL_SEED)
        np.random.seed(EVAL_SEED)
        torch.manual_seed(EVAL_SEED)

        print(f"  Run 2 (determinism check)...")
        kpi_r2, _, _ = run_full_episode(
            actor, obs_mean, obs_std, obs_clip, agent_name,
            use_spatial=agent_cfg["use_spatial"], seed=EVAL_SEED
        )
        t2 = time.time()

        reward_match = abs(kpi_r1["_reward_hash"] - kpi_r2["_reward_hash"]) < 1e-4
        cost_match = abs(kpi_r1["_cost_hash"] - kpi_r2["_cost_hash"]) < 1e-4
        action_match = abs(kpi_r1["_action_hash"] - kpi_r2["_action_hash"]) < 1e-4
        if reward_match and cost_match and action_match:
            print(f"  DETERMINISM VERIFIED (Run 2 in {t2 - t1:.1f}s)")
        else:
            print(f"  WARNING: RESULTS DIFFER BETWEEN RUNS!")
            print(f"    Reward: {kpi_r1['_reward_hash']:.6f} vs {kpi_r2['_reward_hash']:.6f}")
            print(f"    Cost:   {kpi_r1['_cost_hash']:.6f} vs {kpi_r2['_cost_hash']:.6f}")
            print(f"    Action: {kpi_r1['_action_hash']:.6f} vs {kpi_r2['_action_hash']:.6f}")

        # Intelligence metrics
        p25, p50, p75 = price_pcts
        print(f"  Computing intelligence scores...")
        scores, overall, details = compute_intelligence_scores(
            intel_data, p25, p50, p75, actor, obs_mean, obs_std, obs_clip
        )
        print(f"  Intelligence overall: {overall:.3f}")

        all_kpis[agent_name] = kpi_r1
        all_intel_scores[agent_name] = (scores, overall)
        all_intel_details[agent_name] = details

    # =====================================================================
    # REPORT GENERATION
    # =====================================================================
    report = []
    def p(line=""):
        print(line)
        report.append(line)

    agents = list(all_kpis.keys())
    baseline_name = "R5a (MLP)"
    has_baseline = baseline_name in all_kpis

    # Header
    p()
    p("=" * 120)
    p("  V3 ABLATION EVALUATION REPORT")
    p(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"  Agents evaluated: {len(agents)}")
    if skipped_agents:
        p(f"  Agents skipped (no checkpoint): {', '.join(skipped_agents)}")
    p(f"  Seed: {EVAL_SEED}, Deterministic policy (tanh on mean)")
    p("=" * 120)

    # ---- Helper for table rows ----
    def short_name(name, max_len=16):
        """Shorten agent name for table columns."""
        mapping = {
            "R5a (MLP)": "R5a",
            "PPOLagMulti v2": "v2",
            "PPOLagMulti v3a": "v3a",
            "PPOLagMulti v3b": "v3b",
            "PPOLagMulti v5": "v5",
        }
        return mapping.get(name, name[:max_len])

    col_w = 14

    def table_header():
        h = f"  {'Metric':<42}"
        for a in agents:
            h += f"{short_name(a):>{col_w}}"
        if has_baseline and len(agents) > 1:
            for a in agents:
                if a != baseline_name:
                    h += f"{'d_' + short_name(a):>{col_w}}"
        return h

    def table_row(metric_label, key, fmt=".1f", lower_better=None, source="kpi"):
        """Print one row of the comparison table."""
        vals = []
        for a in agents:
            if source == "kpi":
                v = all_kpis[a].get(key, float('nan'))
            elif source == "intel":
                v = all_intel_scores[a][0].get(key, float('nan'))
            elif source == "intel_detail":
                v = all_intel_details[a].get(key, float('nan'))
            else:
                v = float('nan')
            vals.append(v)

        strs = []
        for v in vals:
            if isinstance(v, float) and np.isnan(v):
                strs.append("N/A")
            else:
                strs.append(f"{v:{fmt}}")

        line = f"  {metric_label:<42}"
        for s in strs:
            line += f"{s:>{col_w}}"

        # Delta columns vs baseline
        if has_baseline and len(agents) > 1:
            base_idx = agents.index(baseline_name)
            base_val = vals[base_idx]
            for i, a in enumerate(agents):
                if a == baseline_name:
                    continue
                if not np.isnan(base_val) and not np.isnan(vals[i]):
                    delta = vals[i] - base_val
                    if lower_better is not None:
                        better = (lower_better and delta < 0) or (not lower_better and delta > 0)
                        marker = " *" if better else ""
                    else:
                        marker = ""
                    line += f"{f'{delta:+{fmt}}{marker}':>{col_w}}"
                else:
                    line += f"{'N/A':>{col_w}}"

        p(line)

    # =====================================================================
    # 1. REWARD
    # =====================================================================
    p()
    p(table_header())
    sep_len = 42 + col_w * len(agents)
    if has_baseline and len(agents) > 1:
        sep_len += col_w * (len(agents) - 1)
    p("  " + "-" * (sep_len - 2))

    p()
    p("  --- TOTAL REWARD (STEMS) ---")
    table_row("Total Reward", "total_reward", ".0f", lower_better=False)
    table_row("Avg Reward/Step", "avg_reward", ".4f", lower_better=False)

    # =====================================================================
    # 2. PER-CONSTRAINT COST TOTALS
    # =====================================================================
    p()
    p("  --- PER-CONSTRAINT COST TOTALS (episode sum) ---")
    table_row("C0: EV Dense Charging", "c0_ev_dense_total", ".1f", lower_better=True)
    table_row("C1: EV Departure", "c1_ev_departure_total", ".1f", lower_better=True)
    table_row("C2: Battery SoC", "c2_battery_soc_total", ".1f", lower_better=True)
    table_row("C3: Building Power", "c3_building_power_total", ".1f", lower_better=True)
    table_row("C4: Grid Power", "c4_grid_power_total", ".1f", lower_better=True)
    table_row("Total CMDP Cost (weighted)", "total_cost", ".0f", lower_better=True)

    # =====================================================================
    # 3. PER-CONSTRAINT VIOLATION PERCENTAGES
    # =====================================================================
    p()
    p("  --- PER-CONSTRAINT VIOLATION RATES (% of steps with cost > 0) ---")
    table_row("C0: EV Dense Viol %", "c0_ev_dense_viol_pct", ".2f", lower_better=True)
    table_row("C1: EV Departure Viol %", "c1_ev_departure_viol_pct", ".2f", lower_better=True)
    table_row("C2: Battery SoC Viol %", "c2_battery_soc_viol_pct", ".2f", lower_better=True)
    table_row("C3: Building Power Viol %", "c3_building_power_viol_pct", ".2f", lower_better=True)
    table_row("C4: Grid Power Viol %", "c4_grid_power_viol_pct", ".2f", lower_better=True)
    table_row("Any Constraint Viol %", "cost_positive_pct", ".1f", lower_better=True)

    # =====================================================================
    # 4. CITYLEARN STANDARD KPIs
    # =====================================================================
    p()
    p("  --- CITYLEARN STANDARD KPIs (from env.evaluate(), 1.0 = no-op baseline) ---")
    cl_kpi_list = [
        ("Electricity Consumption", "cl_electricity_consumption_total"),
        ("Carbon Emissions", "cl_carbon_emissions_total"),
        ("Electricity Cost ($)", "cl_cost_total"),
        ("Daily Peak Average", "cl_daily_peak_average"),
        ("All-Time Peak Average", "cl_all_time_peak_average"),
        ("Ramping Average", "cl_ramping_average"),
        ("1 - Loss of Life Share", "cl_1 - Loss of Life Share"),
        ("Zero Net Energy", "cl_zero_net_energy"),
    ]
    for label, key in cl_kpi_list:
        # Lower is better for most CityLearn KPIs (ratio to baseline)
        lb = True if key not in ("cl_zero_net_energy",) else None
        table_row(label, key, ".4f", lower_better=lb)

    # =====================================================================
    # 5. INTELLIGENCE SCORES
    # =====================================================================
    p()
    p("  --- INTELLIGENCE SCORES (0.0 = worst, 1.0 = best) ---")
    intel_metrics = [
        ("Solar Preference", "solar_preference"),
        ("Price Spread", "price_spread"),
        ("V2G Timing", "v2g_timing"),
        ("EV Compliance", "ev_compliance"),
        ("Grid Stability", "grid_stability"),
        ("Pre-Peak Planning", "pre_peak_planning"),
        ("Price Correlation", "price_correlation"),
        ("Behavioral Diversity", "behavioral_diversity"),
        ("Forecast Utilization", "forecast_utilization"),
    ]
    for label, key in intel_metrics:
        table_row(label, key, ".3f", lower_better=False, source="intel")

    # Overall intelligence
    line = f"  {'OVERALL INTELLIGENCE':<42}"
    for a in agents:
        overall = all_intel_scores[a][1]
        line += f"{overall:>{col_w}.3f}"
    if has_baseline and len(agents) > 1:
        base_overall = all_intel_scores[baseline_name][1]
        for a in agents:
            if a == baseline_name:
                continue
            delta = all_intel_scores[a][1] - base_overall
            better = " *" if delta > 0 else ""
            line += f"{f'{delta:+.3f}{better}':>{col_w}}"
    p(line)

    # =====================================================================
    # 6. ACTION STATISTICS
    # =====================================================================
    p()
    p("  --- ACTION STATISTICS ---")
    table_row("Action Mean (all dims)", "action_mean", ".4f")
    table_row("Action Std (all dims)", "action_std", ".4f")
    table_row("Action |Mean| (all dims)", "action_abs_mean", ".4f")
    table_row("Battery Mean Action", "batt_mean", ".4f")
    table_row("EV Mean Action", "ev_mean", ".4f")

    # =====================================================================
    # 7. INTELLIGENCE DETAILS (raw values)
    # =====================================================================
    p()
    p("  --- INTELLIGENCE DETAILS (raw values) ---")
    intel_detail_metrics = [
        ("Solar Diff (EV hi-lo)", "solar_diff", ".4f"),
        ("Price Batt Spread (cheap-exp)", "price_batt_diff", ".4f"),
        ("Batt Discharge@Expensive %", "batt_discharge_expensive_rate", ".3f"),
        ("C1 Violation Rate %", "c1_violation_rate_pct", ".2f"),
        ("C4 Violation Rate %", "c4_violation_rate_pct", ".2f"),
        ("Pre-Peak Swing", "pre_peak_swing", ".4f"),
        ("Corr(Batt, Price)", "corr_batt_price", ".4f"),
        ("Hourly Action StdDev", "hourly_action_stddev", ".4f"),
        ("Price Forecast Sensitivity", "price_sensitivity", ".6f"),
        ("Solar Forecast Sensitivity", "solar_sensitivity", ".6f"),
        ("All Forecast Sensitivity", "all_sensitivity", ".6f"),
    ]
    for label, key, fmt in intel_detail_metrics:
        table_row(label, key, fmt, source="intel_detail")

    # =====================================================================
    # 8. SUMMARY COMPARISON vs R5a BASELINE
    # =====================================================================
    if has_baseline and len(agents) > 1:
        p()
        p("=" * 120)
        p("  SUMMARY: All Agents vs R5a Baseline")
        p("=" * 120)

        r5a = all_kpis[baseline_name]
        comparison_metrics = [
            ("Total Reward", "total_reward", False),
            ("Total Cost", "total_cost", True),
            ("C0 EV Dense Viol %", "c0_ev_dense_viol_pct", True),
            ("C1 EV Depart Viol %", "c1_ev_departure_viol_pct", True),
            ("C2 SoC Viol %", "c2_battery_soc_viol_pct", True),
            ("C3 Building Viol %", "c3_building_power_viol_pct", True),
            ("C4 Grid Viol %", "c4_grid_power_viol_pct", True),
            ("C0 Total Cost", "c0_ev_dense_total", True),
            ("C1 Total Cost", "c1_ev_departure_total", True),
            ("C2 Total Cost", "c2_battery_soc_total", True),
            ("C3 Total Cost", "c3_building_power_total", True),
            ("C4 Total Cost", "c4_grid_power_total", True),
            ("Intelligence", None, False),
        ]

        for agent_name_cmp in agents:
            if agent_name_cmp == baseline_name:
                continue
            p(f"\n  {agent_name_cmp} vs R5a:")
            p(f"  {'Metric':<30} {'R5a':>14} {short_name(agent_name_cmp):>14} {'Change':>10} {'%Change':>10} {'Verdict':>10}")
            p(f"  {'-' * 90}")

            cmp = all_kpis[agent_name_cmp]
            for metric_label, key, lower_better in comparison_metrics:
                if key is None:
                    # Special: intelligence overall
                    v5 = all_intel_scores[baseline_name][1]
                    vm = all_intel_scores[agent_name_cmp][1]
                    lower_better = False
                else:
                    v5 = r5a.get(key, 0)
                    vm = cmp.get(key, 0)

                delta = vm - v5
                if abs(v5) > 1e-6:
                    pct = 100 * delta / abs(v5)
                    if lower_better:
                        verdict = "BETTER" if delta < 0 else "WORSE"
                    else:
                        verdict = "BETTER" if delta > 0 else "WORSE"
                    p(f"  {metric_label:<30} {v5:>14.1f} {vm:>14.1f} {delta:>+10.1f} {pct:>+9.1f}% {verdict:>10}")
                else:
                    p(f"  {metric_label:<30} {v5:>14.1f} {vm:>14.1f} {delta:>+10.1f} {'N/A':>10} {'---':>10}")

    # =====================================================================
    # Footer
    # =====================================================================
    elapsed = time.time() - start_time
    p()
    p(f"  Total evaluation time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    p("=" * 120)

    # =====================================================================
    # Save outputs
    # =====================================================================

    # 1. Full comparison JSON
    json_results = {}
    for name in agents:
        r = dict(all_kpis[name])
        r.pop("hourly_batt", None)
        r.pop("hourly_ev", None)
        json_results[name] = {
            "kpi": r,
            "intelligence_scores": {k: float(v) for k, v in all_intel_scores[name][0].items()},
            "intelligence_overall": float(all_intel_scores[name][1]),
            "intelligence_details": {k: float(v) for k, v in all_intel_details[name].items()},
        }
    if skipped_agents:
        json_results["_skipped_agents"] = skipped_agents

    json_path = f"{OUT_DIR}/full_comparison.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # 2. Full report text
    report_path = f"{OUT_DIR}/full_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"Saved: {report_path}")

    # 3. Intelligence scores JSON (dedicated file)
    intel_json = {}
    for name in agents:
        scores, overall = all_intel_scores[name]
        intel_json[name] = {
            "overall": float(overall),
            "scores": {k: float(v) for k, v in scores.items()},
            "details": {k: float(v) for k, v in all_intel_details[name].items()},
        }
    intel_path = f"{OUT_DIR}/intelligence_scores.json"
    with open(intel_path, "w") as f:
        json.dump(intel_json, f, indent=2, default=str)
    print(f"Saved: {intel_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
