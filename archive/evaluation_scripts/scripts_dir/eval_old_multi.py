#!/usr/bin/env python3
"""
Comprehensive evaluation of the old PPOLagMulti agent (best EV performance).

Produces:
  1. Per-constraint violation percentages (C0-C4) — HONEST, per-departure C1
  2. CityLearn KPIs (from city.evaluate())
  3. Intelligence metrics (solar, price, V2G, pre-peak, correlation, diversity, forecast)
  4. C2 violation % WITH and WITHOUT the SoC safety clamp
  5. Action statistics

Uses CITYLEARN_SPATIAL_OBS=1 to match the 218-dim checkpoint.

Usage:
    python scripts/eval_old_multi.py
"""
import os
import sys
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

EVAL_SEED = 42

# =========================================================================
# Environment variable profile — matches run_multi_lag.sh exactly
# =========================================================================
PROFILE_OLD_MULTI = {
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
    # OLD multi_lag specific
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_SPATIAL_OBS": "1",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "STEMS_ALPHA_GRID": "1.0",
    "STEMS_BETA_RAMP": "2.0",
    "CITYLEARN_EV_DENSE_COST_SCALE": "1.0",
    "COST_W_C1": "10.0",
    "COST_W_C1_DENSE": "5.0",
    "COST_W_C2": "1.0",
    "COST_W_C3": "0.1",
    "COST_W_C4": "5.0",
}


def set_env_profile(profile: dict):
    stale_keys = [
        "CITYLEARN_C3_CONTROLLABLE", "CITYLEARN_SPATIAL_OBS",
        "COST_W_C1", "COST_W_C1_DENSE", "COST_W_C2", "COST_W_C3", "COST_W_C4",
        "CITYLEARN_EV_DENSE_COST_SCALE", "STEMS_ALPHA_GRID", "STEMS_BETA_RAMP",
        "STEMS_LAMBDA_EV", "STEMS_ALPHA_BARRIER", "CITYLEARN_TEMPORAL_WINDOW",
    ]
    for k in stale_keys:
        os.environ.pop(k, None)
    for k, v in profile.items():
        os.environ[k] = v


set_env_profile(PROFILE_OLD_MULTI)

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env
import citylearn_safe.schema_index as si


# =========================================================================
# Checkpoint
# =========================================================================
CHECKPOINT = f"{PROJECT}/runs/multi_lag/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-03-08-22-54-20/torch_save/epoch-50.pt"
OUT_DIR = f"{PROJECT}/runs/multi_lag/5bld/evaluation_comprehensive"
AGENT_NAME = "PPOLagMulti (old, best-EV)"


# =========================================================================
# MLPActor — with log_std to match checkpoint
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
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        return torch.tanh(self.mean(obs))


def load_actor(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi_state = ckpt["pi"]

    h1 = pi_state["mean.0.weight"].shape[0]
    h2 = pi_state["mean.2.weight"].shape[0]
    obs_dim = pi_state["mean.0.weight"].shape[1]
    act_dim = pi_state["mean.4.weight"].shape[0]

    actor = MLPActor(obs_dim, act_dim, (h1, h2))
    actor.load_state_dict(pi_state, strict=True)
    actor.eval()

    obs_mean = obs_std = obs_clip = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        if "_clip" in norm:
            obs_clip = float(norm["_clip"].float().mean())

    return actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim


# =========================================================================
# SoC safety clamp
# =========================================================================
def clamp_battery_actions(action, buildings, t_idx, batt_idx):
    SOC_UPPER = 0.94
    a = action.copy()
    for b_i, act_i in enumerate(batt_idx):
        if b_i >= len(buildings) or act_i >= len(a):
            continue
        es = getattr(buildings[b_i], 'electrical_storage', None)
        if es is None:
            continue
        soc_arr = getattr(es, 'soc', None)
        if soc_arr is None or not hasattr(soc_arr, '__len__') or len(soc_arr) <= t_idx:
            continue
        soc = float(np.clip(soc_arr[t_idx], 0.0, 1.0))
        cap = float(getattr(es, 'capacity', 6.4) or 6.4)
        p_max = float(getattr(es, 'nominal_power', 5.0) or 5.0)
        eta = float(getattr(es, 'efficiency', 0.9) or 0.9)
        denom_charge = p_max * 1.0 * eta
        max_charge = (SOC_UPPER - soc) * cap / denom_charge if denom_charge > 0 else 1.0
        denom_discharge = p_max * 1.0
        max_discharge = soc * cap * eta / denom_discharge if denom_discharge > 0 else 1.0
        a[act_i] = float(np.clip(a[act_i], -max_discharge, max_charge))
    return a


# =========================================================================
# Episode collection
# =========================================================================
def collect_episode(actor, obs_mean, obs_std, obs_clip, apply_clamp=False, seed=EVAL_SEED):
    si._CACHE = None
    set_env_profile(PROFILE_OLD_MULTI)

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)

    # Check if we need spatial wrapper
    env_obs_dim = env.observation_space.shape[0]
    actor_obs_dim = obs_mean.shape[0] if obs_mean is not None else 198

    if env_obs_dim < actor_obs_dim:
        from citylearn_safe.stems_obs_wrapper import SpatialGraphFeaturesWrapper
        p_bmax = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
        env = SpatialGraphFeaturesWrapper(env, num_buildings=5, p_building_max=p_bmax)
        print(f"  Added SpatialGraphFeaturesWrapper: {env_obs_dim} -> {env.observation_space.shape[0]}")

    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(getattr(raw, "buildings", []))

    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)
    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]

    # Price percentiles
    try:
        pr = buildings[0].pricing.electricity_pricing
        all_prices = np.array(pr, dtype=float)
        p25 = np.percentile(all_prices, 25)
        p50 = np.percentile(all_prices, 50)
        p75 = np.percentile(all_prices, 75)
    except Exception:
        p25, p50, p75 = 0.12, 0.16, 0.20

    clamp_tag = "[CLAMP]" if apply_clamp else "[RAW]"
    print(f"\n  {clamp_tag} {AGENT_NAME}")
    print(f"    batt_idx={batt_idx} ev_idx={ev_idx}")
    print(f"    env obs_dim={env.observation_space.shape[0]}, actor obs_dim={actor_obs_dim}")
    print(f"    price percentiles: P25={p25:.4f} P50={p50:.4f} P75={p75:.4f}")

    data = {
        "hour": [], "price": [], "solar": [],
        "actions": [], "ev_actions": [], "batt_actions": [],
        "c0": [], "c1": [], "c2": [], "c3": [], "c4": [],
        "net_load": [], "reward": [], "cost": [],
        "ev_departures": [], "ev_deficit_kwh": [],
        "ev_avoidable_deficit_kwh": [], "ev_unavoidable_deficit_kwh": [],
        "ev_violation_count": [],
        "obs_raw": [],
        "soc_per_building": [],
    }

    clamp_fire_count = 0

    obs, info = env.reset(seed=seed)
    for t in range(8760):
        obs_np = np.asarray(obs, dtype=np.float32)
        data["obs_raw"].append(obs_np.copy())

        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        if obs_mean is not None:
            obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
            if obs_clip is not None:
                obs_t = obs_t.clamp(-obs_clip, obs_clip)
        with torch.no_grad():
            action_t = actor(obs_t).squeeze(0).numpy()
        action = np.clip(action_t, -1.0, 1.0)

        t_now = int(getattr(raw, "time_step", 0))
        hour = t_now % 24

        try:
            price = float(buildings[0].pricing.electricity_pricing[max(0, t_now - 1)])
        except Exception:
            price = 0.17

        solar = 0.0
        for b in buildings:
            try:
                s = getattr(b, "solar_generation", None)
                if s is not None and len(s) > max(0, t_now - 1):
                    solar += abs(float(s[max(0, t_now - 1)]))
            except Exception:
                pass

        soc_t_idx = max(0, t_now - 1)
        step_socs = []
        for b in buildings:
            es = getattr(b, 'electrical_storage', None)
            if es is not None:
                soc_arr = getattr(es, 'soc', None)
                if soc_arr is not None and hasattr(soc_arr, '__len__') and len(soc_arr) > soc_t_idx:
                    step_socs.append(float(soc_arr[soc_t_idx]))
                else:
                    step_socs.append(-1.0)
            else:
                step_socs.append(-1.0)
        data["soc_per_building"].append(step_socs)

        if apply_clamp and batt_idx:
            action_before = action.copy()
            action = clamp_battery_actions(action, buildings, soc_t_idx, batt_idx)
            if not np.allclose(action_before, action, atol=1e-6):
                clamp_fire_count += 1

        obs, reward, terminated, truncated, info = env.step(action)

        data["hour"].append(hour)
        data["price"].append(price)
        data["solar"].append(solar)
        data["actions"].append(action.copy())
        data["ev_actions"].append([float(action[i]) for i in ev_idx])
        data["batt_actions"].append([float(action[i]) for i in batt_idx])

        c0 = float(info.get("cost_ev_dense", 0.0))
        c1 = float(info.get("cost_ev_departure", 0.0))
        c2 = float(info.get("cost_stems_battery", 0.0))
        c3 = float(info.get("cost_stems_building_power", 0.0))
        c4 = float(info.get("cost_stems_grid_power", 0.0))
        data["c0"].append(c0)
        data["c1"].append(c1)
        data["c2"].append(c2)
        data["c3"].append(c3)
        data["c4"].append(c4)
        data["reward"].append(float(reward))
        data["cost"].append(float(info.get("cost", 0.0)))
        data["ev_departures"].append(int(info.get("ev_departure_departures", 0)))
        data["ev_deficit_kwh"].append(float(info.get("ev_departure_deficit_kwh", 0.0)))
        data["ev_avoidable_deficit_kwh"].append(float(info.get("ev_avoidable_deficit_kwh", 0.0)))
        data["ev_unavoidable_deficit_kwh"].append(float(info.get("ev_unavoidable_deficit_kwh", 0.0)))
        data["ev_violation_count"].append(int(info.get("ev_departure_violation_count_deficit", 0)))
        data["net_load"].append(0.0)

        net = 0.0
        for b in buildings:
            try:
                nec = getattr(b, "net_electricity_consumption", None)
                if nec is not None and len(nec) > max(0, t_now - 1):
                    net += float(nec[max(0, t_now - 1)])
            except Exception:
                pass
        data["net_load"][-1] = net

        if t % 2000 == 0 and t > 0:
            print(f"    step {t}/8760 ...")

        if terminated or truncated:
            break

    # CityLearn KPIs
    cl_kpis = {}
    try:
        city = raw
        if hasattr(city, 'evaluate'):
            import pandas as pd
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
        print(f"    CityLearn evaluate() error: {e}")

    for k in data:
        data[k] = np.array(data[k])

    total_steps = len(data["hour"])
    print(f"    Episode done: {total_steps} steps, clamp_fires={clamp_fire_count}")

    env.close()
    return data, p25, p50, p75, cl_kpis, clamp_fire_count, batt_idx, ev_idx


# =========================================================================
# Compute violation stats
# =========================================================================
def compute_violations(data):
    N = len(data["hour"])
    c0 = np.array(data["c0"], dtype=float)
    c1 = np.array(data["c1"], dtype=float)
    c2 = np.array(data["c2"], dtype=float)
    c3 = np.array(data["c3"], dtype=float)
    c4 = np.array(data["c4"], dtype=float)
    reward = np.array(data["reward"], dtype=float)
    cost = np.array(data["cost"], dtype=float)

    ev_deps = np.array(data["ev_departures"], dtype=float)
    ev_deficit_total = np.array(data["ev_deficit_kwh"], dtype=float)
    ev_deficit_avoidable = np.array(data["ev_avoidable_deficit_kwh"], dtype=float)
    ev_deficit_unavoidable = np.array(data["ev_unavoidable_deficit_kwh"], dtype=float)
    ev_viol_count = np.array(data["ev_violation_count"], dtype=float)
    total_departures = int(ev_deps.sum())
    departure_steps = ev_deps > 0
    departure_with_avoidable_deficit = (ev_deps > 0) & (ev_deficit_avoidable > 0)
    departure_with_any_deficit = (ev_deps > 0) & (ev_deficit_total > 0)

    results = {
        "total_steps": N,
        "total_reward": float(reward.sum()),
        "avg_reward": float(reward.mean()),
        "total_cost": float(cost.sum()),
        "avg_cost": float(cost.mean()),
        "c0_total": float(c0.sum()),
        "c1_total": float(c1.sum()),
        "c2_total": float(c2.sum()),
        "c3_total": float(c3.sum()),
        "c4_total": float(c4.sum()),
        "c0_violation_pct": 100.0 * (c0 > 0).sum() / N,
        "c1_violation_pct": 100.0 * (c1 > 0).sum() / N,
        "c2_violation_pct": 100.0 * (c2 > 0).sum() / N,
        "c3_violation_pct": 100.0 * (c3 > 0).sum() / N,
        "c4_violation_pct": 100.0 * (c4 > 0).sum() / N,
        "c1_total_departures": total_departures,
        "c1_departures_with_avoidable_deficit": int(departure_with_avoidable_deficit.sum()),
        "c1_departures_with_any_deficit": int(departure_with_any_deficit.sum()),
        "c1_real_violation_pct": (100.0 * departure_with_avoidable_deficit.sum() / departure_steps.sum()
                                  if departure_steps.sum() > 0 else 0.0),
        "c1_total_violation_pct": (100.0 * departure_with_any_deficit.sum() / departure_steps.sum()
                                   if departure_steps.sum() > 0 else 0.0),
        "c1_departure_rate_pct": 100.0 * departure_steps.sum() / N,
        "c1_mean_avoidable_deficit_kwh": float(ev_deficit_avoidable[departure_steps].mean() if departure_steps.sum() > 0 else 0.0),
        "c1_mean_total_deficit_kwh": float(ev_deficit_total[departure_steps].mean() if departure_steps.sum() > 0 else 0.0),
        "c1_total_avoidable_deficit_kwh": float(ev_deficit_avoidable.sum()),
        "c1_total_unavoidable_deficit_kwh": float(ev_deficit_unavoidable.sum()),
        "c1_total_deficit_kwh": float(ev_deficit_total.sum()),
        "any_violation_pct": 100.0 * ((c0 + c1 + c2 + c3 + c4) > 0).sum() / N,
        "c0_violation_steps": int((c0 > 0).sum()),
        "c1_violation_steps": int((c1 > 0).sum()),
        "c2_violation_steps": int((c2 > 0).sum()),
        "c3_violation_steps": int((c3 > 0).sum()),
        "c4_violation_steps": int((c4 > 0).sum()),
    }

    total_cost_abs = abs(results["total_cost"]) + 1e-8
    results["c0_share_pct"] = 100.0 * results["c0_total"] / total_cost_abs
    results["c1_share_pct"] = 100.0 * results["c1_total"] / total_cost_abs
    results["c2_share_pct"] = 100.0 * results["c2_total"] / total_cost_abs
    results["c3_share_pct"] = 100.0 * results["c3_total"] / total_cost_abs
    results["c4_share_pct"] = 100.0 * results["c4_total"] / total_cost_abs

    return results


# =========================================================================
# Intelligence analysis
# =========================================================================
def analyze_intelligence(data, p25, p50, p75, actor, obs_mean, obs_std, obs_clip):
    hours = data["hour"]
    prices = data["price"]
    solar = data["solar"]
    ev_act = data["ev_actions"]
    batt_act = data["batt_actions"]
    c0, c1, c2, c3, c4 = data["c0"], data["c1"], data["c2"], data["c3"], data["c4"]
    rewards = data["reward"]
    N = len(hours)

    mean_ev = ev_act.mean(axis=1) if ev_act.ndim == 2 and ev_act.shape[1] > 0 else np.zeros(N)
    mean_batt = batt_act.mean(axis=1) if batt_act.ndim == 2 and batt_act.shape[1] > 0 else np.zeros(N)

    raw_metrics = {}

    print(f"\n{'='*70}")
    print(f"  INTELLIGENCE EVALUATION: {AGENT_NAME}")
    print(f"{'='*70}")

    # 1. Solar Charging
    solar_med = np.median(solar[solar > 0]) if (solar > 0).any() else 1.0
    high_solar = solar > solar_med
    ev_charge_during_solar = float((mean_ev[high_solar] > 0.05).mean()) if high_solar.sum() > 0 else 0.0
    mean_ev_act_solar = float(mean_ev[high_solar].mean()) if high_solar.sum() > 0 else 0.0
    low_solar = solar < 0.01
    mean_ev_act_nosolar = float(mean_ev[low_solar].mean()) if low_solar.sum() > 0 else 0.0
    solar_diff = mean_ev_act_solar - mean_ev_act_nosolar
    raw_metrics["ev_charge_during_solar_pct"] = ev_charge_during_solar * 100
    raw_metrics["solar_preference"] = solar_diff

    print(f"\n  1. Solar Charging:")
    print(f"     EV charge rate during high solar:  {ev_charge_during_solar*100:.1f}%")
    print(f"     Mean EV action (high solar):       {mean_ev_act_solar:+.3f}")
    print(f"     Mean EV action (no solar):         {mean_ev_act_nosolar:+.3f}")
    print(f"     Solar preference (diff):           {solar_diff:+.3f}")

    # 2. Price-Aware V2G
    expensive = prices > p75
    cheap = prices < p25
    batt_discharge_expensive = float((mean_batt[expensive] < -0.05).mean()) if expensive.sum() > 0 else 0.0
    mean_batt_expensive = float(mean_batt[expensive].mean()) if expensive.sum() > 0 else 0.0
    ev_discharge_expensive = float((mean_ev[expensive] < -0.05).mean()) if expensive.sum() > 0 else 0.0
    mean_ev_expensive = float(mean_ev[expensive].mean()) if expensive.sum() > 0 else 0.0
    raw_metrics["batt_discharge_expensive_pct"] = batt_discharge_expensive * 100
    raw_metrics["mean_batt_expensive"] = mean_batt_expensive
    raw_metrics["ev_discharge_expensive_pct"] = ev_discharge_expensive * 100

    print(f"\n  2. Price-Aware V2G (price > P75={p75:.4f}):")
    print(f"     Batt discharge rate:  {batt_discharge_expensive*100:.1f}%  (mean={mean_batt_expensive:+.3f})")
    print(f"     EV discharge rate:    {ev_discharge_expensive*100:.1f}%  (mean={mean_ev_expensive:+.3f})")

    # 3. Off-Peak Charging
    batt_charge_cheap = float((mean_batt[cheap] > 0.05).mean()) if cheap.sum() > 0 else 0.0
    mean_batt_cheap = float(mean_batt[cheap].mean()) if cheap.sum() > 0 else 0.0
    ev_charge_cheap = float((mean_ev[cheap] > 0.05).mean()) if cheap.sum() > 0 else 0.0
    mean_ev_cheap = float(mean_ev[cheap].mean()) if cheap.sum() > 0 else 0.0
    price_batt_diff = mean_batt_cheap - mean_batt_expensive
    raw_metrics["batt_charge_cheap_pct"] = batt_charge_cheap * 100
    raw_metrics["price_batt_spread"] = price_batt_diff

    print(f"\n  3. Off-Peak Charging (price < P25={p25:.4f}):")
    print(f"     Batt charge rate:     {batt_charge_cheap*100:.1f}%  (mean={mean_batt_cheap:+.3f})")
    print(f"     EV charge rate:       {ev_charge_cheap*100:.1f}%  (mean={mean_ev_cheap:+.3f})")
    print(f"     Batt price spread:    {price_batt_diff:+.3f}")

    # 4. Constraint Compliance
    c0_rate = 100 * (c0 > 0).sum() / N
    c1_rate = 100 * (c1 > 0).sum() / N
    c2_rate = 100 * (c2 > 0).sum() / N
    c3_rate = 100 * (c3 > 0).sum() / N
    c4_rate = 100 * (c4 > 0).sum() / N
    raw_metrics["c0_violation_rate"] = float(c0_rate)
    raw_metrics["c1_violation_rate"] = float(c1_rate)
    raw_metrics["c2_violation_rate"] = float(c2_rate)
    raw_metrics["c3_violation_rate"] = float(c3_rate)
    raw_metrics["c4_violation_rate"] = float(c4_rate)

    print(f"\n  4. Constraint Compliance:")
    print(f"     C0 EV dense:       {c0_rate:.1f}%")
    print(f"     C1 EV departure:   {c1_rate:.1f}%")
    print(f"     C2 Battery SoC:    {c2_rate:.1f}%")
    print(f"     C3 Building power: {c3_rate:.1f}%")
    print(f"     C4 Grid power:     {c4_rate:.1f}%")

    # 5. Hourly Profile
    print(f"\n  5. Hourly Action Profile:")
    print(f"     Hour  Batt    EV     Price   Solar   Interpretation")
    print(f"     {'---'*20}")
    hourly_batt = np.zeros(24)
    hourly_ev = np.zeros(24)
    hourly_price = np.zeros(24)
    hourly_solar = np.zeros(24)
    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            hourly_batt[h] = mean_batt[mask].mean()
            hourly_ev[h] = mean_ev[mask].mean()
            hourly_price[h] = prices[mask].mean()
            hourly_solar[h] = solar[mask].mean()

    for h in range(24):
        b, e, p, s = hourly_batt[h], hourly_ev[h], hourly_price[h], hourly_solar[h]
        interp = ""
        if b > 0.05: interp += "B:charge "
        elif b < -0.05: interp += "B:discharge "
        if e > 0.1: interp += "EV:charge "
        elif e < -0.05: interp += "EV:V2G "
        if s > solar_med and s > 0: interp += "[solar] "
        if p > p75: interp += "[PEAK$] "
        elif p < p25: interp += "[cheap$] "
        print(f"     {h:02d}:00 {b:+.3f}  {e:+.3f}  {p:.4f}  {s:5.2f}   {interp}")

    # 6. Pre-Peak Preparation
    peak_hours = {h for h in range(24) if hourly_price[h] > p75}
    pre_peak_hours = set()
    for ph in peak_hours:
        for offset in [2, 3, 4]:
            pre_h = (ph - offset) % 24
            if pre_h not in peak_hours:
                pre_peak_hours.add(pre_h)

    if pre_peak_hours:
        pre_peak_mask = np.isin(hours, list(pre_peak_hours))
        pre_peak_batt = float(mean_batt[pre_peak_mask].mean()) if pre_peak_mask.sum() > 0 else 0.0
        peak_mask = np.isin(hours, list(peak_hours))
        peak_batt = float(mean_batt[peak_mask].mean()) if peak_mask.sum() > 0 else 0.0
    else:
        pre_peak_batt = peak_batt = 0.0
    pre_peak_swing = pre_peak_batt - peak_batt
    raw_metrics["pre_peak_batt"] = pre_peak_batt
    raw_metrics["peak_batt"] = peak_batt
    raw_metrics["pre_peak_swing"] = pre_peak_swing

    print(f"\n  6. Pre-Peak Preparation:")
    print(f"     Peak hours:        {sorted(peak_hours)}")
    print(f"     Pre-peak hours:    {sorted(pre_peak_hours)}")
    print(f"     Batt pre-peak:     {pre_peak_batt:+.3f}")
    print(f"     Batt at peak:      {peak_batt:+.3f}")
    print(f"     Swing:             {pre_peak_swing:+.3f}")

    # 7. Price-Action Correlation
    corr_batt_price = float(np.corrcoef(prices, mean_batt)[0, 1]) if len(prices) > 2 else 0.0
    corr_ev_price = float(np.corrcoef(prices, mean_ev)[0, 1]) if len(prices) > 2 else 0.0
    if np.isnan(corr_batt_price): corr_batt_price = 0.0
    if np.isnan(corr_ev_price): corr_ev_price = 0.0
    raw_metrics["corr_batt_price"] = corr_batt_price
    raw_metrics["corr_ev_price"] = corr_ev_price

    print(f"\n  7. Price-Action Correlation:")
    print(f"     Batt vs Price:  r={corr_batt_price:+.3f}")
    print(f"     EV vs Price:    r={corr_ev_price:+.3f}")

    # 8. Solar-Charge Correlation
    solar_mask = solar > 0
    corr_ev_solar = float(np.corrcoef(solar[solar_mask], mean_ev[solar_mask])[0, 1]) if solar_mask.sum() > 100 else 0.0
    if np.isnan(corr_ev_solar): corr_ev_solar = 0.0
    raw_metrics["corr_ev_solar"] = corr_ev_solar

    print(f"\n  8. Solar-Charge Correlation:")
    print(f"     EV vs Solar:    r={corr_ev_solar:+.3f}")

    # 9. Behavioral Diversity
    n_batt_charge_hours = sum(1 for h in range(24) if hourly_batt[h] > 0.05)
    n_batt_discharge_hours = sum(1 for h in range(24) if hourly_batt[h] < -0.05)
    n_batt_idle_hours = 24 - n_batt_charge_hours - n_batt_discharge_hours
    action_variance = float(np.std([hourly_batt[h] for h in range(24)]))
    raw_metrics["n_batt_charge_hours"] = n_batt_charge_hours
    raw_metrics["n_batt_discharge_hours"] = n_batt_discharge_hours
    raw_metrics["hourly_action_std"] = action_variance

    print(f"\n  9. Behavioral Diversity:")
    print(f"     Batt charge hours:    {n_batt_charge_hours}/24")
    print(f"     Batt discharge hours: {n_batt_discharge_hours}/24")
    print(f"     Hourly action StdDev: {action_variance:.3f}")

    # 10. Forecast Utilization (Perturbation Test)
    forecast_scores = compute_forecast_utilization(data, actor, obs_mean, obs_std, obs_clip)

    # Scorecard
    scores = {}
    scores["solar_preference"] = float(np.clip(solar_diff / 0.2, 0, 1))
    scores["price_aware_v2g"] = float(np.clip(batt_discharge_expensive, 0, 1))
    scores["off_peak_charging"] = float(np.clip(price_batt_diff / 0.3, 0, 1))
    scores["ev_compliance"] = float(np.clip(1.0 - c1_rate / 10.0, 0, 1))
    scores["grid_stability"] = float(np.clip(1.0 - c4_rate / 30.0, 0, 1))
    scores["pre_peak_planning"] = float(np.clip(pre_peak_swing / 0.3, 0, 1))
    scores["price_correlation"] = float(np.clip(-corr_batt_price / 0.3, 0, 1))
    scores["behavioral_diversity"] = float(np.clip(action_variance / 0.15, 0, 1))
    scores["forecast_utilization"] = forecast_scores["forecast_utilization"]

    overall = float(np.mean(list(scores.values())))

    print(f"\n  -- INTELLIGENCE SCORECARD --\n")
    print(f"     {'Metric':<25s}  {'Score':>6s}  {'Rating'}")
    print(f"     {'---'*20}")
    for name, score in scores.items():
        rating = "***" if score > 0.7 else "** " if score > 0.4 else "*  " if score > 0.1 else ".  "
        bar_len = int(score * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"     {name:<25s}  {score:5.2f}   {bar} {rating}")
    print(f"\n     {'OVERALL INTELLIGENCE':<25s}  {overall:5.2f}")

    return scores, overall, forecast_scores, raw_metrics


def compute_forecast_utilization(data, actor, obs_mean, obs_std, obs_clip):
    obs_raw = data["obs_raw"]
    obs_dim = obs_raw.shape[1]

    base_obs_dim = 70
    fc_start = base_obs_dim
    n_forecast = 128

    print(f"\n  -- FORECAST UTILIZATION (Perturbation Test) --")
    print(f"     [obs_dim={obs_dim}, base={base_obs_dim}, forecast@{fc_start}, n_forecast={n_forecast}]")

    if fc_start + 72 > obs_dim:
        print(f"     Skipping: obs dim too small for forecast analysis")
        return {"forecast_utilization": 0.0, "price_sensitivity": 0.0,
                "solar_sensitivity": 0.0, "all_sensitivity": 0.0}

    price_range = (fc_start, fc_start + 24)
    solar_range = (fc_start + 48, fc_start + 72)
    all_range = (fc_start, fc_start + n_forecast)

    perturbations = {
        "price_forecast": price_range,
        "solar_forecast": solar_range,
        "all_forecast": all_range,
    }

    SCALE_THRESHOLD = 0.1

    @torch.no_grad()
    def batched_forward(obs_np):
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        if obs_mean is not None:
            obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
            if obs_clip is not None:
                obs_t = obs_t.clamp(-obs_clip, obs_clip)
        return np.clip(actor(obs_t).numpy(), -1.0, 1.0)

    actions_orig = data["actions"]
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

    print(f"     Price forecast |delta|: {deltas['price_forecast']:.6f}  score={price_score:.2f}")
    print(f"     Solar forecast |delta|: {deltas['solar_forecast']:.6f}  score={solar_score:.2f}")
    print(f"     All forecast |delta|:   {deltas['all_forecast']:.6f}  score={all_score:.2f}")

    forecast_utilization = float(np.mean([price_score, solar_score, all_score]))
    print(f"     Forecast Utilization Score: {forecast_utilization:.2f}")

    return {
        "forecast_utilization": forecast_utilization,
        "price_sensitivity": deltas["price_forecast"],
        "solar_sensitivity": deltas["solar_forecast"],
        "all_sensitivity": deltas["all_forecast"],
    }


def compute_action_stats(data, batt_idx, ev_idx):
    actions = data["actions"]
    stats = {
        "action_mean": float(actions.mean()),
        "action_std": float(actions.std()),
        "action_abs_mean": float(np.abs(actions).mean()),
    }
    if batt_idx and actions.shape[1] > max(batt_idx):
        bv = actions[:, batt_idx]
        stats["batt_mean"] = float(bv.mean())
        stats["batt_std"] = float(bv.std())
    else:
        stats["batt_mean"] = stats["batt_std"] = 0.0
    if ev_idx and actions.shape[1] > max(ev_idx):
        ev = actions[:, ev_idx]
        stats["ev_mean"] = float(ev.mean())
        stats["ev_std"] = float(ev.std())
    else:
        stats["ev_mean"] = stats["ev_std"] = 0.0
    return stats


# =========================================================================
# Main
# =========================================================================
def main():
    start_time = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    lines = []
    def p(line=""):
        print(line)
        lines.append(line)

    p("=" * 110)
    p(f"  COMPREHENSIVE EVALUATION: {AGENT_NAME}")
    p(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"  Checkpoint: {CHECKPOINT}")
    p(f"  Environment: 5-building CityLearn V2G, seed={EVAL_SEED}, deterministic")
    p(f"  Spatial obs: ON (218 dims)")
    p("=" * 110)

    # Load actor
    actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(CHECKPOINT)
    p(f"  Loaded: obs_dim={obs_dim}, act_dim={act_dim}")

    # Episode 1: RAW
    p(f"\n  --- Episode 1: RAW (no clamp) ---")
    random.seed(EVAL_SEED); np.random.seed(EVAL_SEED); torch.manual_seed(EVAL_SEED)
    data_raw, p25, p50, p75, cl_kpis, _, batt_idx, ev_idx = collect_episode(
        actor, obs_mean, obs_std, obs_clip, apply_clamp=False, seed=EVAL_SEED)

    violations_raw = compute_violations(data_raw)
    action_stats = compute_action_stats(data_raw, batt_idx, ev_idx)

    # Intelligence (on raw data)
    scores, overall, forecast_scores, raw_metrics = analyze_intelligence(
        data_raw, p25, p50, p75, actor, obs_mean, obs_std, obs_clip)

    # Episode 2: CLAMPED
    p(f"\n  --- Episode 2: CLAMPED (SoC safety clamp) ---")
    random.seed(EVAL_SEED); np.random.seed(EVAL_SEED); torch.manual_seed(EVAL_SEED)
    data_clamped, _, _, _, _, clamp_fires, _, _ = collect_episode(
        actor, obs_mean, obs_std, obs_clip, apply_clamp=True, seed=EVAL_SEED)

    violations_clamped = compute_violations(data_clamped)

    # =================================================================
    # REPORT
    # =================================================================
    col_w = 18

    def make_row(name, val, fmt=".1f"):
        formatted = f"{val:{fmt}}" if not (isinstance(val, float) and np.isnan(val)) else "N/A"
        return f"  {name:<45}  {formatted:>{col_w}}"

    p(f"\n\n{'='*110}")
    p(f"  SECTION 1: REWARD AND COST SUMMARY")
    p(f"{'='*110}")
    p(make_row("Total Reward", violations_raw["total_reward"], ".0f"))
    p(make_row("Avg Reward/Step", violations_raw["avg_reward"], ".4f"))
    p(make_row("Total CMDP Cost", violations_raw["total_cost"], ".0f"))
    p(make_row("Avg Cost/Step", violations_raw["avg_cost"], ".4f"))
    p(make_row("Steps w/ Any Violation %", violations_raw["any_violation_pct"], ".1f"))

    p(f"\n\n{'='*110}")
    p(f"  SECTION 2: PER-CONSTRAINT VIOLATION PERCENTAGES")
    p(f"{'='*110}")
    p("  Step-based (% of 8760 steps with cost > 0):")
    for cname, ckey in [
        ("C0: EV Dense Charging", "c0_violation_pct"),
        ("C1: EV Departure (step-based)", "c1_violation_pct"),
        ("C2: Battery SoC", "c2_violation_pct"),
        ("C3: Building Power", "c3_violation_pct"),
        ("C4: Grid Power", "c4_violation_pct"),
    ]:
        p(make_row(cname, violations_raw[ckey], ".2f"))

    p("")
    p("  C1 EV Departure — HONEST PER-DEPARTURE METRICS:")
    p(make_row("C1: Total Departures", violations_raw["c1_total_departures"], ".0f"))
    p(make_row("C1: Departure Rate (steps %)", violations_raw["c1_departure_rate_pct"], ".2f"))
    p(make_row("C1: Dep w/ Avoidable Deficit", violations_raw["c1_departures_with_avoidable_deficit"], ".0f"))
    p(make_row("C1: Dep w/ Any Deficit", violations_raw["c1_departures_with_any_deficit"], ".0f"))
    p(make_row("C1: REAL Violation % (avoidable/dep)", violations_raw["c1_real_violation_pct"], ".2f"))
    p(make_row("C1: Total Violation % (any/dep)", violations_raw["c1_total_violation_pct"], ".2f"))
    p(make_row("C1: Mean Avoidable Deficit (kWh)", violations_raw["c1_mean_avoidable_deficit_kwh"], ".4f"))
    p(make_row("C1: Total Avoidable Deficit (kWh)", violations_raw["c1_total_avoidable_deficit_kwh"], ".2f"))
    p(make_row("C1: Total Unavoidable Deficit (kWh)", violations_raw["c1_total_unavoidable_deficit_kwh"], ".2f"))
    p(make_row("C1: Total Deficit (kWh, all)", violations_raw["c1_total_deficit_kwh"], ".2f"))

    p(f"\n\n{'='*110}")
    p(f"  SECTION 3: PER-CONSTRAINT COST TOTALS")
    p(f"{'='*110}")
    for cname, ckey in [
        ("C0: EV Dense Total", "c0_total"),
        ("C1: EV Departure Total", "c1_total"),
        ("C2: Battery SoC Total", "c2_total"),
        ("C3: Building Power Total", "c3_total"),
        ("C4: Grid Power Total", "c4_total"),
    ]:
        p(make_row(cname, violations_raw[ckey], ".1f"))

    p("")
    p("  Cost Component Shares (% of total CMDP cost):")
    for cname, ckey in [
        ("C0 Share %", "c0_share_pct"),
        ("C1 Share %", "c1_share_pct"),
        ("C2 Share %", "c2_share_pct"),
        ("C3 Share %", "c3_share_pct"),
        ("C4 Share %", "c4_share_pct"),
    ]:
        p(make_row(cname, violations_raw[ckey], ".1f"))

    p(f"\n\n{'='*110}")
    p(f"  SECTION 4: C2 BATTERY SoC — WITH vs WITHOUT SAFETY CLAMP")
    p(f"{'='*110}")
    p(make_row("C2 Violation % (RAW, no clamp)", violations_raw["c2_violation_pct"], ".2f"))
    p(make_row("C2 Violation % (WITH clamp)", violations_clamped["c2_violation_pct"], ".2f"))
    delta_c2 = violations_clamped["c2_violation_pct"] - violations_raw["c2_violation_pct"]
    p(make_row("Delta (clamped - raw)", delta_c2, "+.2f"))
    p(make_row("C2 Cost Total (RAW)", violations_raw["c2_total"], ".1f"))
    p(make_row("C2 Cost Total (CLAMPED)", violations_clamped["c2_total"], ".1f"))
    p(make_row("Clamp Fire Count", clamp_fires, ".0f"))
    p(make_row("Clamp Fire Rate %", 100.0 * clamp_fires / 8760, ".2f"))

    p(f"\n\n{'='*110}")
    p(f"  SECTION 5: CITYLEARN STANDARD KPIs (1.0 = no-op baseline)")
    p(f"{'='*110}")
    cl_metric_labels = [
        ("electricity_consumption_total", "Electricity Consumption"),
        ("carbon_emissions_total", "Carbon Emissions"),
        ("cost_total", "Electricity Cost ($)"),
        ("daily_peak_average", "Daily Peak Average"),
        ("all_time_peak_average", "All-Time Peak Average"),
        ("ramping_average", "Ramping Average"),
        ("1 - Loss of Life Share", "1 - Loss of Life Share"),
        ("zero_net_energy", "Zero Net Energy"),
    ]
    for cl_key, cl_label in cl_metric_labels:
        v = cl_kpis.get(cl_key, float('nan'))
        p(make_row(cl_label, v, ".4f"))
    p("  (Values < 1.0 mean improvement over no-control baseline)")

    p(f"\n\n{'='*110}")
    p(f"  SECTION 6: ACTION STATISTICS")
    p(f"{'='*110}")
    for metric_name, key in [
        ("Action Mean (all dims)", "action_mean"),
        ("Action Std (all dims)", "action_std"),
        ("Action |Mean| (all dims)", "action_abs_mean"),
        ("Battery Mean Action", "batt_mean"),
        ("Battery Std Action", "batt_std"),
        ("EV Mean Action", "ev_mean"),
        ("EV Std Action", "ev_std"),
    ]:
        p(make_row(metric_name, action_stats.get(key, 0.0), ".4f"))

    p(f"\n\n{'='*110}")
    p(f"  SECTION 7: INTELLIGENCE SCORECARD (normalized [0,1])")
    p(f"{'='*110}")
    for name, score in scores.items():
        rating = "***" if score > 0.7 else "** " if score > 0.4 else "*  " if score > 0.1 else ".  "
        bar_len = int(score * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        p(f"  {name:<30s}  {score:5.3f}  {bar} {rating}")
    p(f"  {'OVERALL INTELLIGENCE':<30s}  {overall:5.3f}")

    p(f"\n\n{'='*110}")
    p(f"  SECTION 8: RAW INTELLIGENCE METRICS")
    p(f"{'='*110}")
    raw_metric_labels = [
        ("solar_preference", "Solar Preference (EV diff)"),
        ("ev_charge_during_solar_pct", "EV Charge During Solar %"),
        ("batt_discharge_expensive_pct", "Batt Discharge Expensive %"),
        ("mean_batt_expensive", "Mean Batt Action (expensive)"),
        ("batt_charge_cheap_pct", "Batt Charge Cheap %"),
        ("price_batt_spread", "Batt Price Spread"),
        ("pre_peak_swing", "Pre-Peak Planning Swing"),
        ("corr_batt_price", "Corr(Batt, Price)"),
        ("corr_ev_price", "Corr(EV, Price)"),
        ("corr_ev_solar", "Corr(EV, Solar)"),
        ("hourly_action_std", "Hourly Action StdDev"),
        ("n_batt_charge_hours", "Batt Charge Hours /24"),
        ("n_batt_discharge_hours", "Batt Discharge Hours /24"),
    ]
    for key, label in raw_metric_labels:
        v = raw_metrics.get(key, 0.0)
        fmt = ".3f" if isinstance(v, float) else ".0f"
        p(make_row(label, v, fmt))

    p(f"\n\n{'='*110}")
    p(f"  SECTION 9: FORECAST SENSITIVITY")
    p(f"{'='*110}")
    for key, label in [
        ("price_sensitivity", "Price Forecast |delta|"),
        ("solar_sensitivity", "Solar Forecast |delta|"),
        ("all_sensitivity", "All Forecast |delta|"),
        ("forecast_utilization", "Forecast Utilization Score"),
    ]:
        p(make_row(label, forecast_scores.get(key, 0.0), ".6f"))

    elapsed = time.time() - start_time
    p(f"\n\n{'='*110}")
    p(f"  Evaluation completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    p(f"{'='*110}")

    # Save
    json_results = {
        "agent": AGENT_NAME,
        "checkpoint": CHECKPOINT,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "violations_raw": {k: float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v
                           for k, v in violations_raw.items()},
        "violations_clamped": {k: float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v
                               for k, v in violations_clamped.items()},
        "clamp_fires": int(clamp_fires),
        "citylearn_kpis": {k: float(v) for k, v in cl_kpis.items()},
        "action_stats": {k: float(v) for k, v in action_stats.items()},
        "intelligence_scores": {k: float(v) for k, v in scores.items()},
        "intelligence_overall": float(overall),
        "forecast": {k: float(v) for k, v in forecast_scores.items()},
        "raw_intelligence_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                                     for k, v in raw_metrics.items()},
    }

    json_path = f"{OUT_DIR}/eval_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    p(f"\n  Saved JSON: {json_path}")

    report_path = f"{OUT_DIR}/eval_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved report: {report_path}")

    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
