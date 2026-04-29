#!/usr/bin/env python3
"""
Comprehensive R11 Evaluation: R5a (MLP baseline) vs R11a (single-lambda) vs R11b (multi-lambda).

Produces:
  1. Per-constraint violation percentages (C0-C4)
  2. CityLearn KPIs (from city.evaluate())
  3. Intelligence metrics (solar preference, price-aware V2G, off-peak charging,
     pre-peak planning, price-action correlation, behavioral diversity, forecast utilization)
  4. C2 violation % WITH and WITHOUT the SoC safety clamp
  5. Side-by-side comparison tables

Each agent runs TWO episodes: one without clamp (raw behavior), one with clamp (clamped C2).

Usage:
    python scripts/eval_r11_full.py
"""
import os
import sys
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from datetime import datetime

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

EVAL_SEED = 42

# =========================================================================
# Environment variable profiles
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

# R11 profile: same as 198 but with Step 1 env vars
PROFILE_R11 = {
    **COMMON_ENV_VARS,
    "CITYLEARN_C3_CONTROLLABLE": "0",
    "CITYLEARN_SPATIAL_OBS": "0",
    "STEMS_LAMBDA_EV": "0.0",
    "STEMS_ALPHA_BARRIER": "0.5",
    "COST_W_C2": "0.0",
    "STEMS_BETA_RAMP": "1.5",
    "COST_W_C3": "5.0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
}


def set_env_profile(profile: dict):
    """Set environment variables from a profile, clearing stale keys first."""
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


# Set initial profile so module imports work
set_env_profile(PROFILE_198)

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env
import citylearn_safe.schema_index as si


# =========================================================================
# Agent definitions
# =========================================================================
AGENTS = {
    "R5a (MLP baseline)": {
        "ckpt": f"{PROJECT}/runs/r6_compare/r5a_5bld/PPOLag-{{CityLearnSafety-V2G-v2}}/seed-000-2026-03-06-11-04-24/torch_save/epoch-50.pt",
        "obs_dim": 198,
        "profile": PROFILE_198,
        "use_spatial": False,
    },
    "R11a (single-lambda)": {
        "ckpt": f"{PROJECT}/runs/r11a_ppolag_improved/5bld/PPOLag-{{CityLearnSafety-V2G-v2}}/seed-000-2026-03-09-22-45-38/torch_save/epoch-50.pt",
        "obs_dim": 198,
        "profile": PROFILE_R11,
        "use_spatial": False,
    },
    "R11b (multi-lambda)": {
        "ckpt": f"{PROJECT}/runs/r11b_multi_improved/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-03-09-22-45-39/torch_save/epoch-50.pt",
        "obs_dim": 198,
        "profile": PROFILE_R11,
        "use_spatial": False,
    },
}


# =========================================================================
# MLPActor and loader
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


def load_actor(ckpt_path, obs_dim, act_dim):
    """Load MLP actor from OmniSafe checkpoint.

    Returns: actor, obs_mean, obs_std, obs_clip, actual_obs_dim, actual_act_dim
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi_state = ckpt["pi"]

    # Discover hidden sizes and true dimensions from weights
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
        raise RuntimeError(f"MLP: Missing keys: {result.missing_keys}")
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
# SoC safety clamp
# =========================================================================
def clamp_battery_actions(action, buildings, t_idx, batt_idx):
    """Apply SoC safety clamp during evaluation.

    Limits battery charge/discharge actions to prevent SoC from exceeding
    the upper bound (0.94) or going below 0.
    """
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
def collect_episode(actor, obs_mean, obs_std, obs_clip, label,
                    use_spatial=False, apply_clamp=False, seed=EVAL_SEED):
    """Run one full episode, collecting per-step data.

    Args:
        actor: MLPActor model
        obs_mean, obs_std, obs_clip: obs normalizer params
        label: agent name for logging
        use_spatial: not used (kept for API compat)
        apply_clamp: if True, apply SoC safety clamp to battery actions
        seed: random seed for determinism

    Returns:
        data: dict of per-step arrays
        p25, p50, p75: price percentiles
        cl_kpis: CityLearn KPIs from evaluate()
        clamp_fire_count: number of steps where clamp modified actions
    """
    # Reset schema index cache
    si._CACHE = None

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)

    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(getattr(raw, "buildings", []))

    # Action name parsing
    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)
    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]

    # Price percentiles for intelligence analysis
    try:
        pr = buildings[0].pricing.electricity_pricing
        all_prices = np.array(pr, dtype=float)
        p25 = np.percentile(all_prices, 25)
        p50 = np.percentile(all_prices, 50)
        p75 = np.percentile(all_prices, 75)
    except Exception:
        p25, p50, p75 = 0.12, 0.16, 0.20

    clamp_tag = "[CLAMP]" if apply_clamp else "[RAW]"
    print(f"\n  {clamp_tag} {label}")
    print(f"    batt_idx={batt_idx} ev_idx={ev_idx}")
    print(f"    env obs_dim={env.observation_space.shape[0]}")
    print(f"    price percentiles: P25={p25:.4f} P50={p50:.4f} P75={p75:.4f}")

    # Data containers
    data = {
        "hour": [], "price": [], "solar": [],
        "actions": [], "ev_actions": [], "batt_actions": [],
        "c0": [], "c1": [], "c2": [], "c3": [], "c4": [],
        "net_load": [], "reward": [], "cost": [],
        "ev_departures": [], "ev_deficit_kwh": [], "ev_avoidable_deficit_kwh": [], "ev_unavoidable_deficit_kwh": [], "ev_violation_count": [],
        "obs_raw": [],
        "soc_per_building": [],  # list of lists: per-step SoC for each building
    }

    clamp_fire_count = 0

    obs, info = env.reset(seed=seed)
    for t in range(8760):
        obs_np = np.asarray(obs, dtype=np.float32)
        data["obs_raw"].append(obs_np.copy())

        # Policy inference
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        if obs_mean is not None:
            obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
            if obs_clip is not None:
                obs_t = obs_t.clamp(-obs_clip, obs_clip)
        with torch.no_grad():
            action_t = actor(obs_t).squeeze(0).numpy()
        action = np.clip(action_t, -1.0, 1.0)

        # Environmental context
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

        # Collect per-building SoC before stepping
        # NOTE: SoC array is pre-allocated to 8760 entries. Use t_now-based
        # index, NOT len(soc_arr)-1 (which is always 8759).
        soc_t_idx = max(0, t_now - 1)  # current SoC is at this index
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

        # Apply SoC safety clamp if requested
        if apply_clamp and batt_idx:
            action_before = action.copy()
            action = clamp_battery_actions(action, buildings, soc_t_idx, batt_idx)
            if not np.allclose(action_before, action, atol=1e-6):
                clamp_fire_count += 1

        obs, reward, terminated, truncated, info = env.step(action)

        # Record data
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
        # EV departure details for proper C1 violation rate
        data["ev_departures"].append(int(info.get("ev_departure_departures", 0)))
        data["ev_deficit_kwh"].append(float(info.get("ev_departure_deficit_kwh", 0.0)))
        # FIX: use agent-controllable deficit (avoidable) for violation %, not total deficit
        data["ev_avoidable_deficit_kwh"].append(float(info.get("ev_avoidable_deficit_kwh", 0.0)))
        data["ev_unavoidable_deficit_kwh"].append(float(info.get("ev_unavoidable_deficit_kwh", 0.0)))
        data["ev_violation_count"].append(int(info.get("ev_departure_violation_count_deficit", 0)))
        data["net_load"].append(0.0)  # computed below

        # Net load
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

    # CityLearn KPIs from evaluate()
    cl_kpis = {}
    try:
        # Navigate to CityLearnEnv (may be wrapped by NormalizedObservationWrapper etc.)
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

    # Convert lists to arrays
    for k in data:
        data[k] = np.array(data[k])

    total_steps = len(data["hour"])
    print(f"    Episode done: {total_steps} steps, clamp_fires={clamp_fire_count}")

    env.close()
    return data, p25, p50, p75, cl_kpis, clamp_fire_count


# =========================================================================
# Constraint violation analysis
# =========================================================================
def compute_violations(data):
    """Compute per-constraint violation percentages and cost totals."""
    N = len(data["hour"])
    # Convert to numpy arrays for vectorized operations
    c0 = np.array(data["c0"], dtype=float)
    c1 = np.array(data["c1"], dtype=float)
    c2 = np.array(data["c2"], dtype=float)
    c3 = np.array(data["c3"], dtype=float)
    c4 = np.array(data["c4"], dtype=float)

    reward = np.array(data["reward"], dtype=float)
    cost = np.array(data["cost"], dtype=float)

    # EV departure data for proper C1 violation rate
    ev_deps = np.array(data["ev_departures"], dtype=float)
    ev_deficit_total = np.array(data["ev_deficit_kwh"], dtype=float)
    ev_deficit_avoidable = np.array(data["ev_avoidable_deficit_kwh"], dtype=float)
    ev_deficit_unavoidable = np.array(data["ev_unavoidable_deficit_kwh"], dtype=float)
    ev_viol_count = np.array(data["ev_violation_count"], dtype=float)
    total_departures = int(ev_deps.sum())
    # Steps with an EV departure
    departure_steps = ev_deps > 0
    # FIX: Use AVOIDABLE (agent-controllable) deficit for violation %, not total
    departure_with_avoidable_deficit = (ev_deps > 0) & (ev_deficit_avoidable > 0)
    # Keep total deficit for reference
    departure_with_any_deficit = (ev_deps > 0) & (ev_deficit_total > 0)

    results = {
        "total_steps": N,
        "total_reward": float(reward.sum()),
        "avg_reward": float(reward.mean()),
        "total_cost": float(cost.sum()),
        "avg_cost": float(cost.mean()),

        # Cost totals
        "c0_total": float(c0.sum()),
        "c1_total": float(c1.sum()),
        "c2_total": float(c2.sum()),
        "c3_total": float(c3.sum()),
        "c4_total": float(c4.sum()),

        # Violation percentages (% of steps with cost > 0)
        "c0_violation_pct": 100.0 * (c0 > 0).sum() / N,
        "c1_violation_pct": 100.0 * (c1 > 0).sum() / N,
        "c2_violation_pct": 100.0 * (c2 > 0).sum() / N,
        "c3_violation_pct": 100.0 * (c3 > 0).sum() / N,
        "c4_violation_pct": 100.0 * (c4 > 0).sum() / N,

        # REAL C1 violation: departures with AVOIDABLE (agent-controllable) deficit
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

        # Any violation
        "any_violation_pct": 100.0 * ((c0 + c1 + c2 + c3 + c4) > 0).sum() / N,

        # Violation step counts
        "c0_violation_steps": int((c0 > 0).sum()),
        "c1_violation_steps": int((c1 > 0).sum()),
        "c2_violation_steps": int((c2 > 0).sum()),
        "c3_violation_steps": int((c3 > 0).sum()),
        "c4_violation_steps": int((c4 > 0).sum()),
    }

    # Cost shares
    total_cost = abs(results["total_cost"]) + 1e-8
    results["c0_share_pct"] = 100.0 * results["c0_total"] / total_cost
    results["c1_share_pct"] = 100.0 * results["c1_total"] / total_cost
    results["c2_share_pct"] = 100.0 * results["c2_total"] / total_cost
    results["c3_share_pct"] = 100.0 * results["c3_total"] / total_cost
    results["c4_share_pct"] = 100.0 * results["c4_total"] / total_cost

    return results


# =========================================================================
# Intelligence analysis
# =========================================================================
def analyze_intelligence(data, p25, p50, p75, label, actor, obs_mean, obs_std, obs_clip):
    """Full intelligence analysis from collected episode data.

    Returns:
        scores: dict of normalized [0,1] scores per metric
        overall: mean of all scores
        forecast_scores: dict with forecast utilization details
        raw_metrics: dict of raw intelligence metric values
    """
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
    print(f"  INTELLIGENCE EVALUATION: {label}")
    print(f"{'='*70}")

    # -- A. TASK INTELLIGENCE --
    print(f"\n  -- A. TASK INTELLIGENCE --\n")

    # 1. Solar Charging Score
    solar_med = np.median(solar[solar > 0]) if (solar > 0).any() else 1.0
    high_solar = solar > solar_med
    if high_solar.sum() > 0:
        ev_charge_during_solar = float((mean_ev[high_solar] > 0.05).mean())
        mean_ev_act_solar = float(mean_ev[high_solar].mean())
    else:
        ev_charge_during_solar = 0.0
        mean_ev_act_solar = 0.0
    low_solar = solar < 0.01
    mean_ev_act_nosolar = float(mean_ev[low_solar].mean()) if low_solar.sum() > 0 else 0.0
    solar_diff = mean_ev_act_solar - mean_ev_act_nosolar

    raw_metrics["ev_charge_during_solar_pct"] = ev_charge_during_solar * 100
    raw_metrics["solar_preference"] = solar_diff

    print(f"  1. Solar Charging:")
    print(f"     EV charge rate during high solar:  {ev_charge_during_solar*100:.1f}%")
    print(f"     Mean EV action (high solar):       {mean_ev_act_solar:+.3f}")
    print(f"     Mean EV action (no solar):         {mean_ev_act_nosolar:+.3f}")
    print(f"     Solar preference (diff):           {solar_diff:+.3f}  {'GOOD' if solar_diff > 0.05 else 'WEAK' if solar_diff > 0 else 'BAD'}")

    # 2. Price-Aware V2G
    expensive = prices > p75
    cheap = prices < p25
    if expensive.sum() > 0:
        batt_discharge_expensive = float((mean_batt[expensive] < -0.05).mean())
        mean_batt_expensive = float(mean_batt[expensive].mean())
        ev_discharge_expensive = float((mean_ev[expensive] < -0.05).mean())
        mean_ev_expensive = float(mean_ev[expensive].mean())
    else:
        batt_discharge_expensive = mean_batt_expensive = 0.0
        ev_discharge_expensive = mean_ev_expensive = 0.0

    raw_metrics["batt_discharge_expensive_pct"] = batt_discharge_expensive * 100
    raw_metrics["mean_batt_expensive"] = mean_batt_expensive
    raw_metrics["ev_discharge_expensive_pct"] = ev_discharge_expensive * 100

    print(f"\n  2. Price-Aware V2G (price > P75={p75:.4f}):")
    print(f"     Batt discharge rate:  {batt_discharge_expensive*100:.1f}%  (mean={mean_batt_expensive:+.3f})")
    print(f"     EV discharge rate:    {ev_discharge_expensive*100:.1f}%  (mean={mean_ev_expensive:+.3f})")

    # 3. Off-Peak Charging
    if cheap.sum() > 0:
        batt_charge_cheap = float((mean_batt[cheap] > 0.05).mean())
        mean_batt_cheap = float(mean_batt[cheap].mean())
        ev_charge_cheap = float((mean_ev[cheap] > 0.05).mean())
        mean_ev_cheap = float(mean_ev[cheap].mean())
    else:
        batt_charge_cheap = mean_batt_cheap = 0.0
        ev_charge_cheap = mean_ev_cheap = 0.0
    price_batt_diff = mean_batt_cheap - mean_batt_expensive

    raw_metrics["batt_charge_cheap_pct"] = batt_charge_cheap * 100
    raw_metrics["price_batt_spread"] = price_batt_diff

    print(f"\n  3. Off-Peak Charging (price < P25={p25:.4f}):")
    print(f"     Batt charge rate:     {batt_charge_cheap*100:.1f}%  (mean={mean_batt_cheap:+.3f})")
    print(f"     EV charge rate:       {ev_charge_cheap*100:.1f}%  (mean={mean_ev_cheap:+.3f})")
    print(f"     Batt price spread:    {price_batt_diff:+.3f}  {'GOOD' if price_batt_diff > 0.1 else 'WEAK' if price_batt_diff > 0 else 'BAD'}")

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
    print(f"     C1 EV departure:   {c1_rate:.1f}%  ({'GOOD' if c1_rate < 1 else 'OK' if c1_rate < 5 else 'POOR'})")
    print(f"     C2 Battery SoC:    {c2_rate:.1f}%")
    print(f"     C3 Building power: {c3_rate:.1f}%")
    print(f"     C4 Grid power:     {c4_rate:.1f}%")
    print(f"     Total reward:      {rewards.sum():.1f}")
    print(f"     Total C0 cost:     {c0.sum():.1f}")
    print(f"     Total C1 cost:     {c1.sum():.1f}")
    print(f"     Total C2 cost:     {c2.sum():.1f}")
    print(f"     Total C3 cost:     {c3.sum():.1f}")
    print(f"     Total C4 cost:     {c4.sum():.1f}")

    # -- B. TEMPORAL PLANNING --
    print(f"\n  -- B. TEMPORAL PLANNING --\n")

    # 5. Diurnal Action Profile
    print(f"  5. Hourly Action Profile (mean battery | mean EV):")
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
    print(f"     Batt pre-peak:     {pre_peak_batt:+.3f}  {'CHARGING (GOOD)' if pre_peak_batt > 0.05 else 'not charging'}")
    print(f"     Batt at peak:      {peak_batt:+.3f}  {'DISCHARGING (GOOD)' if peak_batt < -0.05 else 'not discharging'}")
    print(f"     Swing:             {pre_peak_swing:+.3f}  {'PLANNING AHEAD' if pre_peak_swing > 0.1 else 'NO PLANNING'}")

    # 7. Price-Action Correlation
    corr_batt_price = float(np.corrcoef(prices, mean_batt)[0, 1]) if len(prices) > 2 else 0.0
    corr_ev_price = float(np.corrcoef(prices, mean_ev)[0, 1]) if len(prices) > 2 else 0.0
    if np.isnan(corr_batt_price):
        corr_batt_price = 0.0
    if np.isnan(corr_ev_price):
        corr_ev_price = 0.0

    raw_metrics["corr_batt_price"] = corr_batt_price
    raw_metrics["corr_ev_price"] = corr_ev_price

    print(f"\n  7. Price-Action Correlation:")
    print(f"     Batt vs Price:  r={corr_batt_price:+.3f}  {'GOOD' if corr_batt_price < -0.1 else 'WEAK' if corr_batt_price < 0 else 'BAD'}")
    print(f"     EV vs Price:    r={corr_ev_price:+.3f}  {'GOOD' if corr_ev_price < -0.1 else 'WEAK' if corr_ev_price < 0 else 'BAD'}")

    # 8. Solar-Charge Correlation
    solar_mask = solar > 0
    if solar_mask.sum() > 100:
        corr_ev_solar = float(np.corrcoef(solar[solar_mask], mean_ev[solar_mask])[0, 1])
        if np.isnan(corr_ev_solar):
            corr_ev_solar = 0.0
    else:
        corr_ev_solar = 0.0

    raw_metrics["corr_ev_solar"] = corr_ev_solar

    print(f"\n  8. Solar-Charge Correlation:")
    print(f"     EV vs Solar:    r={corr_ev_solar:+.3f}  {'GOOD' if corr_ev_solar > 0.1 else 'WEAK' if corr_ev_solar > 0 else 'BAD'}")

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
    print(f"     Batt idle hours:      {n_batt_idle_hours}/24")
    print(f"     Hourly action StdDev: {action_variance:.3f}  {'DIVERSE' if action_variance > 0.1 else 'FLAT'}")

    # -- D. FORECAST UTILIZATION (Perturbation Test) --
    forecast_scores = compute_forecast_utilization(data, actor, obs_mean, obs_std, obs_clip, label)

    # -- C. INTELLIGENCE SCORECARD --
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

    print(f"\n  -- C. INTELLIGENCE SCORECARD --\n")
    print(f"     {'Metric':<25s}  {'Score':>6s}  {'Rating'}")
    print(f"     {'---'*20}")
    for name, score in scores.items():
        rating = "***" if score > 0.7 else "** " if score > 0.4 else "*  " if score > 0.1 else ".  "
        bar_len = int(score * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"     {name:<25s}  {score:5.2f}   {bar} {rating}")

    print(f"\n     {'OVERALL INTELLIGENCE':<25s}  {overall:5.2f}   {'INTELLIGENT' if overall > 0.5 else 'LEARNING' if overall > 0.3 else 'DUMB'}")
    print(f"{'='*70}\n")

    return scores, overall, forecast_scores, raw_metrics


# =========================================================================
# Forecast Utilization (Perturbation Test)
# =========================================================================
def compute_forecast_utilization(data, actor, obs_mean, obs_std, obs_clip, label):
    """Perturbation test: zero out forecast dims, measure action change."""
    obs_raw = data["obs_raw"]   # (T, obs_dim)
    actions_orig = data["actions"]
    obs_dim = obs_raw.shape[1]

    base_obs_dim = 70  # base obs is always 70 for 5-building setup
    fc_start = base_obs_dim

    # Forecast layout: 24 price + 24 load + 24 solar + 24 carbon + 32 EV = 128
    n_forecast = 128

    print(f"\n  -- D. FORECAST UTILIZATION (Perturbation Test) --")
    print(f"     [obs_dim={obs_dim}, base={base_obs_dim}, forecast@{fc_start}, n_forecast={n_forecast}]")

    if fc_start + 72 > obs_dim:
        print(f"     Skipping: obs dim too small for forecast analysis")
        return {
            "forecast_utilization": 0.0,
            "price_sensitivity": 0.0,
            "solar_sensitivity": 0.0,
            "all_sensitivity": 0.0,
        }

    price_range = (fc_start, fc_start + 24)          # dims [70:94]
    solar_range = (fc_start + 48, fc_start + 72)      # dims [118:142]
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
        acts = actor(obs_t).numpy()
        return np.clip(acts, -1.0, 1.0)

    deltas = {}
    for name, (start, end) in perturbations.items():
        end = min(end, obs_dim)
        obs_perturbed = obs_raw.copy()
        obs_perturbed[:, start:end] = 0.0
        actions_perturbed = batched_forward(obs_perturbed)
        step_deltas = np.mean(np.abs(actions_orig - actions_perturbed), axis=1)
        deltas[name] = float(np.mean(step_deltas))

    print(f"\n  10. Price Forecast Sensitivity:")
    print(f"      Zeroed dims [{price_range[0]}:{price_range[1]}]")
    print(f"      Mean |action delta|: {deltas['price_forecast']:.6f}")
    price_score = float(np.clip(deltas["price_forecast"] / SCALE_THRESHOLD, 0, 1))
    print(f"      Score: {price_score:.2f}")

    print(f"\n  11. Solar Forecast Sensitivity:")
    print(f"      Zeroed dims [{solar_range[0]}:{solar_range[1]}]")
    print(f"      Mean |action delta|: {deltas['solar_forecast']:.6f}")
    solar_score = float(np.clip(deltas["solar_forecast"] / SCALE_THRESHOLD, 0, 1))
    print(f"      Score: {solar_score:.2f}")

    print(f"\n  12. All Forecast Sensitivity:")
    print(f"      Zeroed dims [{all_range[0]}:{min(all_range[1], obs_dim)}]")
    print(f"      Mean |action delta|: {deltas['all_forecast']:.6f}")
    all_score = float(np.clip(deltas["all_forecast"] / SCALE_THRESHOLD, 0, 1))
    print(f"      Score: {all_score:.2f}")

    forecast_utilization = float(np.mean([price_score, solar_score, all_score]))
    print(f"\n      Forecast Utilization Score: {forecast_utilization:.2f}")

    return {
        "forecast_utilization": forecast_utilization,
        "price_sensitivity": deltas["price_forecast"],
        "solar_sensitivity": deltas["solar_forecast"],
        "all_sensitivity": deltas["all_forecast"],
    }


# =========================================================================
# Action statistics
# =========================================================================
def compute_action_stats(data, batt_idx, ev_idx):
    """Compute action-level summary statistics."""
    actions = data["actions"]
    stats = {
        "action_mean": float(actions.mean()),
        "action_std": float(actions.std()),
        "action_abs_mean": float(np.abs(actions).mean()),
    }

    if batt_idx and actions.shape[1] > max(batt_idx):
        batt_vals = actions[:, batt_idx]
        stats["batt_mean"] = float(batt_vals.mean())
        stats["batt_std"] = float(batt_vals.std())
    else:
        stats["batt_mean"] = 0.0
        stats["batt_std"] = 0.0

    if ev_idx and actions.shape[1] > max(ev_idx):
        ev_vals = actions[:, ev_idx]
        stats["ev_mean"] = float(ev_vals.mean())
        stats["ev_std"] = float(ev_vals.std())
    else:
        stats["ev_mean"] = 0.0
        stats["ev_std"] = 0.0

    return stats


# =========================================================================
# Report printing utilities
# =========================================================================
class ReportPrinter:
    """Dual-output printer: stdout and text file accumulator."""

    def __init__(self):
        self.lines = []

    def p(self, line=""):
        print(line)
        self.lines.append(line)

    def get_text(self):
        return "\n".join(self.lines)


def format_table_row(values, widths, fmts=None):
    """Format a row of values with given column widths."""
    parts = []
    for i, (v, w) in enumerate(zip(values, widths)):
        fmt = fmts[i] if fmts else None
        if isinstance(v, float):
            if fmt:
                s = f"{v:{fmt}}"
            else:
                s = f"{v:.2f}"
        elif isinstance(v, int):
            s = str(v)
        else:
            s = str(v)
        parts.append(f"{s:>{w}}" if i > 0 else f"{s:<{w}}")
    return "  " + "  ".join(parts)


# =========================================================================
# Main
# =========================================================================
def main():
    start_time = time.time()
    OUT_DIR = f"{PROJECT}/runs/r11_evaluation"
    os.makedirs(OUT_DIR, exist_ok=True)

    rp = ReportPrinter()

    rp.p("=" * 110)
    rp.p("  COMPREHENSIVE R11 EVALUATION")
    rp.p(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rp.p(f"  Agents: R5a (MLP baseline), R11a (single-lambda), R11b (multi-lambda)")
    rp.p(f"  Environment: 5-building CityLearn V2G, seed={EVAL_SEED}, deterministic")
    rp.p(f"  Episodes: 2 per agent (raw + clamped)")
    rp.p("=" * 110)

    # Storage for all results
    all_violations_raw = {}
    all_violations_clamped = {}
    all_intelligence = {}
    all_forecasts = {}
    all_raw_metrics = {}
    all_cl_kpis = {}
    all_action_stats = {}
    all_clamp_fires = {}

    for agent_name, agent_cfg in AGENTS.items():
        rp.p(f"\n{'='*80}")
        rp.p(f"  LOADING: {agent_name}")
        rp.p(f"  Checkpoint: {agent_cfg['ckpt']}")
        rp.p(f"  Expected obs_dim: {agent_cfg['obs_dim']}")
        rp.p(f"{'='*80}")

        # Set environment profile for this agent
        set_env_profile(agent_cfg["profile"])

        # Load actor
        try:
            actor, obs_mean, obs_std, obs_clip, actual_obs_dim, act_dim = load_actor(
                agent_cfg["ckpt"], agent_cfg["obs_dim"], 9  # act_dim=9 for 5 buildings
            )
        except FileNotFoundError as e:
            rp.p(f"  ERROR: {e}")
            rp.p(f"  SKIPPING agent {agent_name}")
            continue

        rp.p(f"  Loaded: obs_dim={actual_obs_dim}, act_dim={act_dim}")

        # ---------------------------------------------------------------
        # Episode 1: WITHOUT clamp (raw agent behavior)
        # ---------------------------------------------------------------
        rp.p(f"\n  --- Episode 1: RAW (no clamp) ---")
        random.seed(EVAL_SEED)
        np.random.seed(EVAL_SEED)
        torch.manual_seed(EVAL_SEED)

        data_raw, p25, p50, p75, cl_kpis, _ = collect_episode(
            actor, obs_mean, obs_std, obs_clip, agent_name,
            use_spatial=agent_cfg["use_spatial"],
            apply_clamp=False, seed=EVAL_SEED,
        )

        violations_raw = compute_violations(data_raw)
        all_violations_raw[agent_name] = violations_raw
        all_cl_kpis[agent_name] = cl_kpis

        # Discover action indices from the env (need to redo for stats)
        si._CACHE = None
        base_tmp = make_base_env(central_agent=True)
        raw_tmp = unwrap_to_raw_citylearn_env(base_tmp)
        names_raw_tmp = getattr(raw_tmp, "action_names", [])
        if isinstance(names_raw_tmp, list) and len(names_raw_tmp) == 1 and isinstance(names_raw_tmp[0], list):
            names_tmp = names_raw_tmp[0]
        else:
            names_tmp = list(names_raw_tmp)
        batt_idx = [i for i, n in enumerate(names_tmp) if str(n).strip().lower() == "electrical_storage"]
        ev_idx = [i for i, n in enumerate(names_tmp) if "electric_vehicle_storage_charger_" in str(n).lower()]
        try:
            base_tmp.close()
        except Exception:
            pass

        action_stats = compute_action_stats(data_raw, batt_idx, ev_idx)
        all_action_stats[agent_name] = action_stats

        # Intelligence analysis (on raw episode data)
        scores, overall, forecast_scores, raw_intel_metrics = analyze_intelligence(
            data_raw, p25, p50, p75, agent_name, actor, obs_mean, obs_std, obs_clip
        )
        all_intelligence[agent_name] = (scores, overall)
        all_forecasts[agent_name] = forecast_scores
        all_raw_metrics[agent_name] = raw_intel_metrics

        # ---------------------------------------------------------------
        # Episode 2: WITH clamp (SoC safety clamp)
        # ---------------------------------------------------------------
        rp.p(f"\n  --- Episode 2: CLAMPED (SoC safety clamp) ---")
        random.seed(EVAL_SEED)
        np.random.seed(EVAL_SEED)
        torch.manual_seed(EVAL_SEED)

        # Reset env profile again (collect_episode resets si._CACHE internally)
        set_env_profile(agent_cfg["profile"])

        data_clamped, _, _, _, _, clamp_fires = collect_episode(
            actor, obs_mean, obs_std, obs_clip, agent_name,
            use_spatial=agent_cfg["use_spatial"],
            apply_clamp=True, seed=EVAL_SEED,
        )

        violations_clamped = compute_violations(data_clamped)
        all_violations_clamped[agent_name] = violations_clamped
        all_clamp_fires[agent_name] = clamp_fires

    # =====================================================================
    # REPORT GENERATION
    # =====================================================================
    agents = [a for a in AGENTS.keys() if a in all_violations_raw]

    if not agents:
        rp.p("\nERROR: No agents were successfully evaluated. Check checkpoint paths.")
        with open(f"{OUT_DIR}/eval_report.txt", "w") as f:
            f.write(rp.get_text())
        return

    # Short names for table headers
    short_names = {a: a.split("(")[0].strip() for a in agents}
    col_w = 18  # column width for values

    def make_header():
        header = f"  {'Metric':<40}"
        for a in agents:
            header += f"  {short_names[a]:>{col_w}}"
        return header

    def make_sep():
        return "  " + "-" * (40 + (col_w + 2) * len(agents))

    def make_row(metric_name, values, fmt=".1f"):
        row = f"  {metric_name:<40}"
        for v in values:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row += f"  {'N/A':>{col_w}}"
            else:
                formatted = f"{v:{fmt}}"
                row += f"  {formatted:>{col_w}}"
        return row

    # -----------------------------------------------------------------
    # Section 1: Reward and Cost Summary
    # -----------------------------------------------------------------
    rp.p(f"\n\n{'='*110}")
    rp.p("  SECTION 1: REWARD AND COST SUMMARY (RAW, no clamp)")
    rp.p(f"{'='*110}")
    rp.p(make_header())
    rp.p(make_sep())

    rp.p(make_row("Total Reward",
                   [all_violations_raw[a]["total_reward"] for a in agents], ".0f"))
    rp.p(make_row("Avg Reward/Step",
                   [all_violations_raw[a]["avg_reward"] for a in agents], ".4f"))
    rp.p(make_row("Total CMDP Cost",
                   [all_violations_raw[a]["total_cost"] for a in agents], ".0f"))
    rp.p(make_row("Avg Cost/Step",
                   [all_violations_raw[a]["avg_cost"] for a in agents], ".4f"))
    rp.p(make_row("Steps w/ Any Violation %",
                   [all_violations_raw[a]["any_violation_pct"] for a in agents], ".1f"))

    # -----------------------------------------------------------------
    # Section 2: Per-Constraint Violation Percentages (RAW)
    # -----------------------------------------------------------------
    rp.p(f"\n\n{'='*110}")
    rp.p("  SECTION 2: PER-CONSTRAINT VIOLATION PERCENTAGES (RAW, no clamp)")
    rp.p(f"{'='*110}")
    rp.p(make_header())
    rp.p(make_sep())

    for cname, ckey in [
        ("C0: EV Dense Charging", "c0_violation_pct"),
        ("C1: EV Departure (step %)", "c1_violation_pct"),
        ("C2: Battery SoC", "c2_violation_pct"),
        ("C3: Building Power", "c3_violation_pct"),
        ("C4: Grid Power", "c4_violation_pct"),
    ]:
        rp.p(make_row(cname,
                       [all_violations_raw[a][ckey] for a in agents], ".2f"))

    rp.p(make_sep())
    rp.p("  (% of 8760 steps where constraint cost > 0)")
    rp.p("")
    rp.p("  C1 EV Departure — CORRECTED (agent-controllable deficit only):")
    rp.p(make_header())
    rp.p(make_sep())
    rp.p(make_row("C1: Departure Rate (steps %)",
                   [all_violations_raw[a]["c1_departure_rate_pct"] for a in agents], ".2f"))
    rp.p(make_row("C1: Total Departures",
                   [all_violations_raw[a]["c1_total_departures"] for a in agents], ".0f"))
    rp.p(make_row("C1: Dep w/ Avoidable Deficit",
                   [all_violations_raw[a]["c1_departures_with_avoidable_deficit"] for a in agents], ".0f"))
    rp.p(make_row("C1: Dep w/ Any Deficit (incl unavoidable)",
                   [all_violations_raw[a]["c1_departures_with_any_deficit"] for a in agents], ".0f"))
    rp.p(make_row("C1: REAL Violation % (avoidable/dep)",
                   [all_violations_raw[a]["c1_real_violation_pct"] for a in agents], ".1f"))
    rp.p(make_row("C1: Total Violation % (any/dep)",
                   [all_violations_raw[a]["c1_total_violation_pct"] for a in agents], ".1f"))
    rp.p(make_row("C1: Mean Avoidable Deficit (kWh)",
                   [all_violations_raw[a]["c1_mean_avoidable_deficit_kwh"] for a in agents], ".3f"))
    rp.p(make_row("C1: Total Avoidable Deficit (kWh)",
                   [all_violations_raw[a]["c1_total_avoidable_deficit_kwh"] for a in agents], ".1f"))
    rp.p(make_row("C1: Total Unavoidable Deficit (kWh)",
                   [all_violations_raw[a]["c1_total_unavoidable_deficit_kwh"] for a in agents], ".1f"))
    rp.p(make_row("C1: Total Deficit (kWh, all)",
                   [all_violations_raw[a]["c1_total_deficit_kwh"] for a in agents], ".1f"))
    rp.p(make_sep())

    # -----------------------------------------------------------------
    # Section 3: Per-Constraint Cost Totals (RAW)
    # -----------------------------------------------------------------
    rp.p(f"\n\n{'='*110}")
    rp.p("  SECTION 3: PER-CONSTRAINT COST TOTALS (RAW)")
    rp.p(f"{'='*110}")
    rp.p(make_header())
    rp.p(make_sep())

    for cname, ckey in [
        ("C0: EV Dense Total", "c0_total"),
        ("C1: EV Departure Total", "c1_total"),
        ("C2: Battery SoC Total", "c2_total"),
        ("C3: Building Power Total", "c3_total"),
        ("C4: Grid Power Total", "c4_total"),
    ]:
        rp.p(make_row(cname,
                       [all_violations_raw[a][ckey] for a in agents], ".1f"))

    rp.p(make_sep())

    # Cost shares
    rp.p("")
    rp.p("  Cost Component Shares (% of total CMDP cost):")
    rp.p(make_header())
    rp.p(make_sep())
    for cname, ckey in [
        ("C0 Share %", "c0_share_pct"),
        ("C1 Share %", "c1_share_pct"),
        ("C2 Share %", "c2_share_pct"),
        ("C3 Share %", "c3_share_pct"),
        ("C4 Share %", "c4_share_pct"),
    ]:
        rp.p(make_row(cname,
                       [all_violations_raw[a][ckey] for a in agents], ".1f"))

    # -----------------------------------------------------------------
    # Section 4: C2 Violation WITH vs WITHOUT Safety Clamp
    # -----------------------------------------------------------------
    rp.p(f"\n\n{'='*110}")
    rp.p("  SECTION 4: C2 BATTERY SoC VIOLATION -- WITH vs WITHOUT SAFETY CLAMP")
    rp.p(f"{'='*110}")
    rp.p(make_header())
    rp.p(make_sep())

    rp.p(make_row("C2 Violation % (RAW, no clamp)",
                   [all_violations_raw[a]["c2_violation_pct"] for a in agents], ".2f"))
    rp.p(make_row("C2 Violation % (WITH clamp)",
                   [all_violations_clamped[a]["c2_violation_pct"] for a in agents], ".2f"))

    # Delta
    deltas = []
    for a in agents:
        raw_v = all_violations_raw[a]["c2_violation_pct"]
        clamp_v = all_violations_clamped[a]["c2_violation_pct"]
        deltas.append(clamp_v - raw_v)
    rp.p(make_row("Delta (clamped - raw)",
                   deltas, "+.2f"))

    rp.p(make_row("C2 Cost Total (RAW)",
                   [all_violations_raw[a]["c2_total"] for a in agents], ".1f"))
    rp.p(make_row("C2 Cost Total (CLAMPED)",
                   [all_violations_clamped[a]["c2_total"] for a in agents], ".1f"))

    rp.p(make_sep())
    rp.p(make_row("Clamp Fire Count (steps modified)",
                   [all_clamp_fires[a] for a in agents], ".0f"))
    rp.p(make_row("Clamp Fire Rate %",
                   [100.0 * all_clamp_fires[a] / 8760 for a in agents], ".2f"))

    # Also show all violations clamped vs raw
    rp.p("")
    rp.p("  All Constraint Violations (CLAMPED episode):")
    rp.p(make_header())
    rp.p(make_sep())
    for cname, ckey in [
        ("C0 Violation % (clamped)", "c0_violation_pct"),
        ("C1 Violation % (clamped)", "c1_violation_pct"),
        ("C2 Violation % (clamped)", "c2_violation_pct"),
        ("C3 Violation % (clamped)", "c3_violation_pct"),
        ("C4 Violation % (clamped)", "c4_violation_pct"),
    ]:
        rp.p(make_row(cname,
                       [all_violations_clamped[a][ckey] for a in agents], ".2f"))
    rp.p(make_row("Total Reward (clamped)",
                   [all_violations_clamped[a]["total_reward"] for a in agents], ".0f"))

    # -----------------------------------------------------------------
    # Section 5: CityLearn KPIs
    # -----------------------------------------------------------------
    rp.p(f"\n\n{'='*110}")
    rp.p("  SECTION 5: CITYLEARN STANDARD KPIs (from evaluate(), 1.0 = no-op baseline)")
    rp.p(f"{'='*110}")
    rp.p(make_header())
    rp.p(make_sep())

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
        vals = []
        for a in agents:
            v = all_cl_kpis.get(a, {}).get(cl_key, None)
            if v is None:
                v = float('nan')
            vals.append(v)
        rp.p(make_row(cl_label, vals, ".4f"))

    rp.p(make_sep())
    rp.p("  (Values < 1.0 mean improvement over no-control baseline)")

    # -----------------------------------------------------------------
    # Section 6: Action Statistics
    # -----------------------------------------------------------------
    rp.p(f"\n\n{'='*110}")
    rp.p("  SECTION 6: ACTION STATISTICS")
    rp.p(f"{'='*110}")
    rp.p(make_header())
    rp.p(make_sep())

    for metric_name, key in [
        ("Action Mean (all dims)", "action_mean"),
        ("Action Std (all dims)", "action_std"),
        ("Action |Mean| (all dims)", "action_abs_mean"),
        ("Battery Mean Action", "batt_mean"),
        ("Battery Std Action", "batt_std"),
        ("EV Mean Action", "ev_mean"),
        ("EV Std Action", "ev_std"),
    ]:
        vals = [all_action_stats.get(a, {}).get(key, 0.0) for a in agents]
        rp.p(make_row(metric_name, vals, ".4f"))

    # -----------------------------------------------------------------
    # Section 7: Intelligence Scorecard (side-by-side)
    # -----------------------------------------------------------------
    rp.p(f"\n\n{'='*110}")
    rp.p("  SECTION 7: INTELLIGENCE SCORECARD (normalized [0,1], higher = better)")
    rp.p(f"{'='*110}")
    rp.p(make_header() + f"  {'Best':>{col_w}}")
    rp.p(make_sep() + "-" * (col_w + 2))

    if agents:
        metric_names = list(all_intelligence[agents[0]][0].keys())
        for mname in metric_names:
            vals = []
            for a in agents:
                v = all_intelligence[a][0].get(mname, 0.0)
                vals.append(v)
            best_idx = int(np.argmax(vals))
            best_name = short_names[agents[best_idx]]
            row = f"  {mname:<40}"
            for v in vals:
                row += f"  {v:>{col_w}.3f}"
            row += f"  {best_name:>{col_w}}"
            rp.p(row)

        rp.p(make_sep() + "-" * (col_w + 2))

        # Overall row
        overalls = [all_intelligence[a][1] for a in agents]
        best_idx = int(np.argmax(overalls))
        best_name = short_names[agents[best_idx]]
        row = f"  {'OVERALL INTELLIGENCE':<40}"
        for o in overalls:
            row += f"  {o:>{col_w}.3f}"
        row += f"  {best_name:>{col_w}}"
        rp.p(row)

    # -----------------------------------------------------------------
    # Section 8: Raw Intelligence Metrics
    # -----------------------------------------------------------------
    rp.p(f"\n\n{'='*110}")
    rp.p("  SECTION 8: RAW INTELLIGENCE METRICS (unnormalized)")
    rp.p(f"{'='*110}")
    rp.p(make_header())
    rp.p(make_sep())

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
        vals = [all_raw_metrics.get(a, {}).get(key, 0.0) for a in agents]
        fmt = ".3f" if isinstance(vals[0], float) else ".0f"
        rp.p(make_row(label, vals, fmt))

    # -----------------------------------------------------------------
    # Section 9: Forecast Sensitivity Details
    # -----------------------------------------------------------------
    rp.p(f"\n\n{'='*110}")
    rp.p("  SECTION 9: FORECAST SENSITIVITY (perturbation test)")
    rp.p(f"{'='*110}")
    rp.p(make_header())
    rp.p(make_sep())

    for key, label in [
        ("price_sensitivity", "Price Forecast |delta|"),
        ("solar_sensitivity", "Solar Forecast |delta|"),
        ("all_sensitivity", "All Forecast |delta|"),
        ("forecast_utilization", "Forecast Utilization Score"),
    ]:
        vals = [all_forecasts.get(a, {}).get(key, 0.0) for a in agents]
        rp.p(make_row(label, vals, ".6f"))

    # -----------------------------------------------------------------
    # Section 10: Summary Comparison
    # -----------------------------------------------------------------
    rp.p(f"\n\n{'='*110}")
    rp.p("  SECTION 10: SUMMARY COMPARISON vs R5a BASELINE")
    rp.p(f"{'='*110}")

    r5a_name = "R5a (MLP baseline)"
    if r5a_name in all_violations_raw:
        r5a = all_violations_raw[r5a_name]
        other_agents = [a for a in agents if a != r5a_name]

        comparisons = [
            ("Total Reward", "total_reward", False, ".0f"),
            ("Total Cost", "total_cost", True, ".0f"),
            ("C0 EV Dense Viol %", "c0_violation_pct", True, ".2f"),
            ("C1 EV Departure Viol %", "c1_violation_pct", True, ".2f"),
            ("C2 Battery SoC Viol %", "c2_violation_pct", True, ".2f"),
            ("C3 Building Power Viol %", "c3_violation_pct", True, ".2f"),
            ("C4 Grid Power Viol %", "c4_violation_pct", True, ".2f"),
            ("C0 Total Cost", "c0_total", True, ".1f"),
            ("C1 Total Cost", "c1_total", True, ".1f"),
            ("C2 Total Cost", "c2_total", True, ".1f"),
            ("C3 Total Cost", "c3_total", True, ".1f"),
            ("C4 Total Cost", "c4_total", True, ".1f"),
        ]

        for other in other_agents:
            rp.p(f"\n  {short_names[other]} vs R5a:")
            rp.p(f"  {'Metric':<35}  {'R5a':>14}  {short_names[other]:>14}  {'Delta%':>10}  {'Verdict':>8}")
            rp.p(f"  {'-'*85}")

            other_data = all_violations_raw[other]
            for metric_label, key, lower_better, fmt in comparisons:
                v_base = r5a.get(key, 0)
                v_other = other_data.get(key, 0)
                if abs(v_base) > 1e-6:
                    pct = 100 * (v_other - v_base) / abs(v_base)
                    if lower_better:
                        direction = "BETTER" if pct < 0 else "WORSE"
                    else:
                        direction = "BETTER" if pct > 0 else "WORSE"
                    rp.p(f"  {metric_label:<35}  {v_base:>14{fmt}}  {v_other:>14{fmt}}  {pct:>+9.1f}%  {direction:>8}")
                else:
                    rp.p(f"  {metric_label:<35}  {v_base:>14{fmt}}  {v_other:>14{fmt}}  {'N/A':>10}  {'':>8}")

        # Also compare intelligence scores
        if r5a_name in all_intelligence:
            rp.p(f"\n  Intelligence Score Comparison:")
            header_line = f"  {'Metric':<35}  {'R5a':>14}"
            for other in other_agents:
                header_line += f"  {short_names[other]:>14}"
            rp.p(header_line)
            rp.p(f"  {'-'*85}")

            r5a_scores = all_intelligence[r5a_name][0]
            for mname in r5a_scores.keys():
                line = f"  {mname:<35}  {r5a_scores[mname]:>14.3f}"
                for other in other_agents:
                    line += f"  {all_intelligence[other][0].get(mname, 0.0):>14.3f}"
                rp.p(line)

            line = f"  {'OVERALL':<35}  {all_intelligence[r5a_name][1]:>14.3f}"
            for other in other_agents:
                line += f"  {all_intelligence[other][1]:>14.3f}"
            rp.p(line)

    # -----------------------------------------------------------------
    # Final footer
    # -----------------------------------------------------------------
    elapsed = time.time() - start_time
    rp.p(f"\n\n{'='*110}")
    rp.p(f"  Evaluation completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    rp.p(f"{'='*110}")

    # =====================================================================
    # Save results
    # =====================================================================
    # Build JSON-serializable results dict
    json_results = {}
    for a in agents:
        json_results[a] = {
            "violations_raw": {k: float(v) if isinstance(v, (float, np.floating, int, np.integer))
                               else v for k, v in all_violations_raw[a].items()},
            "violations_clamped": {k: float(v) if isinstance(v, (float, np.floating, int, np.integer))
                                   else v for k, v in all_violations_clamped[a].items()},
            "clamp_fires": int(all_clamp_fires.get(a, 0)),
            "citylearn_kpis": {k: float(v) for k, v in all_cl_kpis.get(a, {}).items()},
            "action_stats": {k: float(v) for k, v in all_action_stats.get(a, {}).items()},
            "intelligence_scores": {k: float(v) for k, v in all_intelligence[a][0].items()},
            "intelligence_overall": float(all_intelligence[a][1]),
            "forecast": {k: float(v) for k, v in all_forecasts.get(a, {}).items()},
            "raw_intelligence_metrics": {k: float(v) if isinstance(v, (float, np.floating))
                                         else int(v) for k, v in all_raw_metrics.get(a, {}).items()},
        }

    json_path = f"{OUT_DIR}/eval_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    rp.p(f"\n  Saved JSON: {json_path}")

    report_path = f"{OUT_DIR}/eval_report.txt"
    with open(report_path, "w") as f:
        f.write(rp.get_text())
    print(f"  Saved report: {report_path}")

    print(f"\nDone. Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
