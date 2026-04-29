#!/usr/bin/env python3
"""
Intelligence Evaluation: GradS v3 vs R5a, R5b, R6, R7 baselines.

Adapted from eval_intelligence_r8.py.
Measures whether agents exhibit intelligent temporal behavior:
  A. Task Intelligence (solar charging, price-aware V2G, off-peak charging, compliance)
  B. Temporal Planning (diurnal profiles, pre-peak preparation, correlations)
  D. Forecast Utilization (perturbation tests)
  C. Overall Intelligence Scorecard

Key difference from R8 eval: agents have different obs dims.
  - R5a, R5b:  obs_dim=198  (SPATIAL_OBS=0, C3_CONTROLLABLE=0)
  - R6, R7:    obs_dim=218  (SPATIAL_OBS=1, C3_CONTROLLABLE=1)  [from checkpoint weights]
  - GradS v3:  obs_dim=218  (SPATIAL_OBS=1, C3_CONTROLLABLE=1)

Each agent gets its own env with matching wrapper chain.
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# Global seeds for determinism
EVAL_SEED = 42
random.seed(EVAL_SEED)
np.random.seed(EVAL_SEED)
torch.manual_seed(EVAL_SEED)


# =========================================================================
# Environment variable profiles for different agents
# =========================================================================
# Common env vars shared by all agents
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

# Profile for R5a/R5b: no spatial obs, no C3 controllable
PROFILE_198 = {
    **COMMON_ENV_VARS,
    "CITYLEARN_C3_CONTROLLABLE": "0",
    "CITYLEARN_SPATIAL_OBS": "0",
}

# Profile for R6/R7/GradS v3: spatial obs + C3 controllable
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


def set_env_profile(profile: dict):
    """Set environment variables from a profile, clearing stale keys."""
    # Clear keys that might be left over from previous profile
    stale_keys = [
        "CITYLEARN_C3_CONTROLLABLE", "CITYLEARN_SPATIAL_OBS",
        "COST_W_C1", "COST_W_C1_DENSE", "COST_W_C2", "COST_W_C3", "COST_W_C4",
        "CITYLEARN_EV_DENSE_COST_SCALE", "STEMS_ALPHA_GRID", "STEMS_BETA_RAMP",
    ]
    for k in stale_keys:
        os.environ.pop(k, None)
    # Set the new profile
    for k, v in profile.items():
        os.environ[k] = v


# Set initial profile (will be overridden per agent)
set_env_profile(PROFILE_198)

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.stems_obs_wrapper import SpatialGraphFeaturesWrapper
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env
import citylearn_safe.schema_index as si


# =========================================================================
# Actor class (MLP only -- all agents in this comparison are MLP)
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
    """Load MLP actor from OmniSafe checkpoint."""
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
        raise RuntimeError(f"MLP: Missing keys: {result.missing_keys}")

    actor.eval()

    obs_mean = obs_std = obs_clip = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        clip_t = norm["_clip"].float()
        obs_clip = float(clip_t.mean())
        assert (clip_t == obs_clip).all(), f"Non-uniform clip values: {clip_t.unique()}"

    return actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim


# =========================================================================
# Episode collection with per-step intelligence data
# =========================================================================
def collect_episode(actor, obs_mean, obs_std, obs_clip, label, use_spatial=False):
    """Run one episode, record per-step data for intelligence analysis.

    Args:
        use_spatial: If True, wraps env with SpatialGraphFeaturesWrapper (for 218-dim agents).
    """
    si._CACHE = None
    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)

    if use_spatial:
        p_bmax = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
        env = SpatialGraphFeaturesWrapper(env, num_buildings=5, p_building_max=p_bmax)

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

    # Price percentiles
    try:
        pr = buildings[0].pricing.electricity_pricing
        all_prices = np.array(pr, dtype=float)
        p25 = np.percentile(all_prices, 25)
        p50 = np.percentile(all_prices, 50)
        p75 = np.percentile(all_prices, 75)
    except Exception:
        p25, p50, p75 = 0.12, 0.16, 0.20

    print(f"\n[Intel] {label}")
    print(f"[Intel] batt_idx={batt_idx} ev_idx={ev_idx} use_spatial={use_spatial}")
    print(f"[Intel] env obs_dim={env.observation_space.shape[0]}")
    print(f"[Intel] price percentiles: P25={p25:.4f} P50={p50:.4f} P75={p75:.4f}")

    data = {
        "hour": [], "price": [], "solar": [],
        "actions": [], "ev_actions": [], "batt_actions": [],
        "c1": [], "c2": [], "c3": [], "c4": [],
        "net_load": [], "reward": [],
        "obs_raw": [],
    }

    obs, info = env.reset(seed=EVAL_SEED)
    for t in range(8760):
        # Policy inference
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

        obs, reward, terminated, truncated, info = env.step(action)

        data["hour"].append(hour)
        data["price"].append(price)
        data["solar"].append(solar)
        data["actions"].append(action.copy())
        data["ev_actions"].append([float(action[i]) for i in ev_idx])
        data["batt_actions"].append([float(action[i]) for i in batt_idx])

        c1 = float(info.get("cost_ev_departure", 0))
        c2 = float(info.get("cost_stems_battery", 0))
        c3 = float(info.get("cost_stems_building_power", 0))
        c4 = float(info.get("cost_stems_grid_power", 0))
        data["c1"].append(c1)
        data["c2"].append(c2)
        data["c3"].append(c3)
        data["c4"].append(c4)
        data["reward"].append(float(reward))

        net = 0.0
        for b in buildings:
            try:
                nec = getattr(b, "net_electricity_consumption", None)
                if nec is not None and len(nec) > max(0, t_now - 1):
                    net += float(nec[max(0, t_now - 1)])
            except Exception:
                pass
        data["net_load"].append(net)

        if terminated or truncated:
            break

    for k in data:
        data[k] = np.array(data[k])

    env.close()
    return data, p25, p50, p75


# =========================================================================
# Forecast Utilization (Perturbation Test)
# =========================================================================
def compute_forecast_utilization(data, actor, obs_mean, obs_std, obs_clip, label):
    """Perturbation test: zero out forecast dims, measure action change."""
    obs_raw = data["obs_raw"]   # (T, obs_dim)
    actions_orig = data["actions"]
    obs_dim = obs_raw.shape[1]

    n_forecast = 128  # 24 price + 24 load + 24 solar + 24 carbon + 32 EV
    # Forecast dims start after base obs (70 dims for 5 buildings)
    base_obs_dim = 70  # base obs is always 70 for 5-building setup
    fc_start = base_obs_dim

    print(f"\n  -- D. FORECAST UTILIZATION (Perturbation Test) --")
    print(f"     [obs_dim={obs_dim}, base={base_obs_dim}, forecast@{fc_start}, n_forecast={n_forecast}]")

    if fc_start + 72 > obs_dim:
        print(f"     Skipping: obs dim too small for forecast analysis")
        return {"forecast_utilization": 0.0}

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

    print(f"\n  11. Price Forecast Sensitivity:")
    print(f"      Zeroed dims [{price_range[0]}:{price_range[1]}]")
    print(f"      Mean |action delta|: {deltas['price_forecast']:.6f}")
    price_score = float(np.clip(deltas["price_forecast"] / SCALE_THRESHOLD, 0, 1))
    print(f"      Score: {price_score:.2f}")

    print(f"\n  12. Solar Forecast Sensitivity:")
    print(f"      Zeroed dims [{solar_range[0]}:{solar_range[1]}]")
    print(f"      Mean |action delta|: {deltas['solar_forecast']:.6f}")
    solar_score = float(np.clip(deltas["solar_forecast"] / SCALE_THRESHOLD, 0, 1))
    print(f"      Score: {solar_score:.2f}")

    print(f"\n  13. All Forecast Sensitivity:")
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
# Intelligence Analysis
# =========================================================================
def analyze_intelligence(data, p25, p50, p75, label, actor, obs_mean, obs_std, obs_clip):
    """Full intelligence analysis from collected episode data."""
    hours = data["hour"]
    prices = data["price"]
    solar = data["solar"]
    ev_act = data["ev_actions"]
    batt_act = data["batt_actions"]
    c1, c2, c3, c4 = data["c1"], data["c2"], data["c3"], data["c4"]
    rewards = data["reward"]
    N = len(hours)

    mean_ev = ev_act.mean(axis=1) if ev_act.ndim == 2 else ev_act
    mean_batt = batt_act.mean(axis=1) if batt_act.ndim == 2 else batt_act

    print(f"\n{'='*70}")
    print(f"  INTELLIGENCE EVALUATION: {label}")
    print(f"{'='*70}")

    # -- A. TASK INTELLIGENCE --
    print(f"\n  -- A. TASK INTELLIGENCE --\n")

    # 1. Solar Charging Score
    solar_med = np.median(solar[solar > 0]) if (solar > 0).any() else 1.0
    high_solar = solar > solar_med
    if high_solar.sum() > 0:
        ev_charge_during_solar = (mean_ev[high_solar] > 0.05).mean()
        mean_ev_act_solar = mean_ev[high_solar].mean()
    else:
        ev_charge_during_solar = 0.0
        mean_ev_act_solar = 0.0
    low_solar = solar < 0.01
    mean_ev_act_nosolar = mean_ev[low_solar].mean() if low_solar.sum() > 0 else 0.0
    solar_diff = mean_ev_act_solar - mean_ev_act_nosolar
    print(f"  1. Solar Charging:")
    print(f"     EV charge rate during high solar:  {ev_charge_during_solar*100:.1f}%")
    print(f"     Mean EV action (high solar):       {mean_ev_act_solar:+.3f}")
    print(f"     Mean EV action (no solar):         {mean_ev_act_nosolar:+.3f}")
    print(f"     Solar preference (diff):           {solar_diff:+.3f}  {'GOOD' if solar_diff > 0.05 else 'WEAK' if solar_diff > 0 else 'BAD'}")

    # 2. Price-Aware V2G
    expensive = prices > p75
    cheap = prices < p25
    if expensive.sum() > 0:
        batt_discharge_expensive = (mean_batt[expensive] < -0.05).mean()
        mean_batt_expensive = mean_batt[expensive].mean()
        ev_discharge_expensive = (mean_ev[expensive] < -0.05).mean()
        mean_ev_expensive = mean_ev[expensive].mean()
    else:
        batt_discharge_expensive = mean_batt_expensive = ev_discharge_expensive = mean_ev_expensive = 0.0
    print(f"\n  2. Price-Aware V2G (price > P75={p75:.4f}):")
    print(f"     Batt discharge rate:  {batt_discharge_expensive*100:.1f}%  (mean={mean_batt_expensive:+.3f})")
    print(f"     EV discharge rate:    {ev_discharge_expensive*100:.1f}%  (mean={mean_ev_expensive:+.3f})")

    # 3. Off-Peak Charging
    if cheap.sum() > 0:
        batt_charge_cheap = (mean_batt[cheap] > 0.05).mean()
        mean_batt_cheap = mean_batt[cheap].mean()
        ev_charge_cheap = (mean_ev[cheap] > 0.05).mean()
        mean_ev_cheap = mean_ev[cheap].mean()
    else:
        batt_charge_cheap = mean_batt_cheap = ev_charge_cheap = mean_ev_cheap = 0.0
    price_batt_diff = mean_batt_cheap - mean_batt_expensive
    print(f"\n  3. Off-Peak Charging (price < P25={p25:.4f}):")
    print(f"     Batt charge rate:     {batt_charge_cheap*100:.1f}%  (mean={mean_batt_cheap:+.3f})")
    print(f"     EV charge rate:       {ev_charge_cheap*100:.1f}%  (mean={mean_ev_cheap:+.3f})")
    print(f"     Batt price spread:    {price_batt_diff:+.3f}  {'GOOD' if price_batt_diff > 0.1 else 'WEAK' if price_batt_diff > 0 else 'BAD'}")

    # 4. Compliance
    c1_rate = 100 * (c1 > 0).sum() / N
    c2_rate = 100 * (c2 > 0).sum() / N
    c3_rate = 100 * (c3 > 0).sum() / N
    c4_rate = 100 * (c4 > 0).sum() / N
    print(f"\n  4. Constraint Compliance:")
    print(f"     C1 EV departure:   {c1_rate:.1f}%  ({'GOOD' if c1_rate < 1 else 'OK' if c1_rate < 5 else 'POOR'})")
    print(f"     C2 Battery SoC:    {c2_rate:.1f}%")
    print(f"     C3 Building power: {c3_rate:.1f}%")
    print(f"     C4 Grid power:     {c4_rate:.1f}%")
    print(f"     Total reward:      {rewards.sum():.1f}")
    print(f"     Total C1 cost:     {c1.sum():.1f}")
    print(f"     Total C2 cost:     {c2.sum():.1f}")
    print(f"     Total C3 cost:     {c3.sum():.1f}")
    print(f"     Total C4 cost:     {c4.sum():.1f}")

    # -- B. TEMPORAL PLANNING --
    print(f"\n  -- B. TEMPORAL PLANNING --\n")

    # 6. Diurnal Action Profile
    print(f"  6. Hourly Action Profile (mean battery | mean EV):")
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

    # 7. Pre-Peak Preparation
    peak_hours = {h for h in range(24) if hourly_price[h] > p75}
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

    print(f"\n  7. Pre-Peak Preparation:")
    print(f"     Peak hours:        {sorted(peak_hours)}")
    print(f"     Pre-peak hours:    {sorted(pre_peak_hours)}")
    print(f"     Batt pre-peak:     {pre_peak_batt:+.3f}  {'CHARGING (GOOD)' if pre_peak_batt > 0.05 else 'not charging'}")
    print(f"     Batt at peak:      {peak_batt:+.3f}  {'DISCHARGING (GOOD)' if peak_batt < -0.05 else 'not discharging'}")
    print(f"     Swing:             {pre_peak_batt - peak_batt:+.3f}  {'PLANNING AHEAD' if (pre_peak_batt - peak_batt) > 0.1 else 'NO PLANNING'}")

    # 8. Price-Action Correlation
    corr_batt_price = np.corrcoef(prices, mean_batt)[0, 1]
    corr_ev_price = np.corrcoef(prices, mean_ev)[0, 1]
    print(f"\n  8. Price-Action Correlation:")
    print(f"     Batt vs Price:  r={corr_batt_price:+.3f}  {'GOOD' if corr_batt_price < -0.1 else 'WEAK' if corr_batt_price < 0 else 'BAD'}")
    print(f"     EV vs Price:    r={corr_ev_price:+.3f}  {'GOOD' if corr_ev_price < -0.1 else 'WEAK' if corr_ev_price < 0 else 'BAD'}")

    # 9. Solar-Charge Correlation
    solar_mask = solar > 0
    corr_ev_solar = np.corrcoef(solar[solar_mask], mean_ev[solar_mask])[0, 1] if solar_mask.sum() > 100 else 0.0
    print(f"\n  9. Solar-Charge Correlation:")
    print(f"     EV vs Solar:    r={corr_ev_solar:+.3f}  {'GOOD' if corr_ev_solar > 0.1 else 'WEAK' if corr_ev_solar > 0 else 'BAD'}")

    # 10. Behavioral Diversity
    n_batt_charge_hours = sum(1 for h in range(24) if hourly_batt[h] > 0.05)
    n_batt_discharge_hours = sum(1 for h in range(24) if hourly_batt[h] < -0.05)
    n_batt_idle_hours = 24 - n_batt_charge_hours - n_batt_discharge_hours
    action_variance = np.std([hourly_batt[h] for h in range(24)])
    print(f"\n  10. Behavioral Diversity:")
    print(f"     Batt charge hours:    {n_batt_charge_hours}/24")
    print(f"     Batt discharge hours: {n_batt_discharge_hours}/24")
    print(f"     Batt idle hours:      {n_batt_idle_hours}/24")
    print(f"     Hourly action StdDev: {action_variance:.3f}  {'DIVERSE' if action_variance > 0.1 else 'FLAT'}")

    # -- D. FORECAST UTILIZATION --
    forecast_scores = compute_forecast_utilization(data, actor, obs_mean, obs_std, obs_clip, label)

    # -- C. INTELLIGENCE SCORECARD --
    scores = {}
    scores["solar_preference"] = float(np.clip(solar_diff / 0.2, 0, 1))
    scores["price_spread"] = float(np.clip(price_batt_diff / 0.3, 0, 1))
    scores["v2g_timing"] = float(np.clip(batt_discharge_expensive, 0, 1))
    scores["ev_compliance"] = float(np.clip(1.0 - c1_rate / 10.0, 0, 1))
    scores["grid_stability"] = float(np.clip(1.0 - c4_rate / 30.0, 0, 1))
    scores["pre_peak_planning"] = float(np.clip((pre_peak_batt - peak_batt) / 0.3, 0, 1))
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

    # Extra summary metrics
    extra = {
        "total_reward": float(rewards.sum()),
        "total_c1": float(c1.sum()),
        "total_c2": float(c2.sum()),
        "total_c3": float(c3.sum()),
        "total_c4": float(c4.sum()),
        "c1_violation_rate": float(c1_rate),
        "c4_violation_rate": float(c4_rate),
    }

    return scores, overall, forecast_scores, extra


# =========================================================================
# Main
# =========================================================================
def main():
    OUT_DIR = f"{PROJECT}/runs/grads_v3/grads_5bld/evaluation"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Agent definitions
    # obs_dim determined from checkpoint weights:
    #   R5a, R5b = 198 (SPATIAL_OBS=0)
    #   R6, R7, GradS v3 = 218 (SPATIAL_OBS=1)
    AGENTS = {
        "R5a (MLP)": {
            "ckpt": f"{PROJECT}/runs/r6_compare/r5a_5bld/PPOLag-{{CityLearnSafety-V2G-v2}}/seed-000-2026-03-06-11-04-24/torch_save/epoch-50.pt",
            "obs_dim": 198,
            "profile": PROFILE_198,
            "use_spatial": False,
        },
        "R5b (MLP)": {
            "ckpt": f"{PROJECT}/runs/r6_compare/r5b_5bld/PPOLag-{{CityLearnSafety-V2G-v2}}/seed-000-2026-03-06-10-55-40/torch_save/epoch-50.pt",
            "obs_dim": 198,
            "profile": PROFILE_198,
            "use_spatial": False,
        },
        "R6 (PPOLag)": {
            "ckpt": f"{PROJECT}/runs/r6_compare/r6_5bld/PPOLag-{{CityLearnSafety-V2G-v2}}/seed-000-2026-03-06-10-55-43/torch_save/epoch-50.pt",
            "obs_dim": 218,
            "profile": PROFILE_218,
            "use_spatial": True,
        },
        "R7 (PPOLag)": {
            "ckpt": f"{PROJECT}/runs/r6_compare/r7_5bld/PPOLag-{{CityLearnSafety-V2G-v2}}/seed-000-2026-03-06-21-10-32/torch_save/epoch-50.pt",
            "obs_dim": 218,
            "profile": PROFILE_218,
            "use_spatial": True,
        },
        "GradS v3 (PPOLagGradS)": {
            "ckpt": f"{PROJECT}/runs/grads_v3/grads_5bld/PPOLagGradS-{{CityLearnSafety-V2G-v2}}/seed-042-2026-03-08-16-55-44/torch_save/epoch-50.pt",
            "obs_dim": 218,
            "profile": PROFILE_218,
            "use_spatial": True,
        },
    }

    all_scores = {}
    all_forecasts = {}
    all_extras = {}

    for agent_name, agent_cfg in AGENTS.items():
        print(f"\n{'='*60}")
        print(f"  LOADING: {agent_name}")
        print(f"  Checkpoint: {agent_cfg['ckpt']}")
        print(f"  Expected obs_dim: {agent_cfg['obs_dim']}")
        print(f"  Spatial obs: {agent_cfg['use_spatial']}")
        print(f"{'='*60}")

        # Set env profile for this agent
        set_env_profile(agent_cfg["profile"])

        actor, obs_mean, obs_std, obs_clip, actual_obs_dim, act_dim = load_actor(
            agent_cfg["ckpt"], agent_cfg["obs_dim"], 9  # act_dim=9 for 5 buildings
        )
        print(f"  Loaded: obs_dim={actual_obs_dim}, act_dim={act_dim}")

        # Reset seeds before each collection for consistency
        random.seed(EVAL_SEED)
        np.random.seed(EVAL_SEED)
        torch.manual_seed(EVAL_SEED)

        data, p25, p50, p75 = collect_episode(
            actor, obs_mean, obs_std, obs_clip, agent_name,
            use_spatial=agent_cfg["use_spatial"]
        )
        scores, overall, forecast, extra = analyze_intelligence(
            data, p25, p50, p75, agent_name, actor, obs_mean, obs_std, obs_clip
        )
        all_scores[agent_name] = (scores, overall)
        all_forecasts[agent_name] = forecast
        all_extras[agent_name] = extra

    # =========================================================================
    # Side-by-side comparison
    # =========================================================================
    agents = list(all_scores.keys())
    n_agents = len(agents)

    print(f"\n{'='*110}")
    print(f"  SIDE-BY-SIDE INTELLIGENCE COMPARISON")
    print(f"{'='*110}")

    # Header
    header = f"  {'Metric':<25s}"
    for a in agents:
        short = a.split("(")[0].strip()
        header += f"  {short:>12s}"
    header += "   Best"
    print(header)
    print(f"  {'---'*37}")

    # Score rows
    metric_names = list(all_scores[agents[0]][0].keys())
    for name in metric_names:
        row = f"  {name:<25s}"
        vals = []
        for a in agents:
            v = all_scores[a][0][name]
            vals.append(v)
            row += f"  {v:12.3f}"
        best_idx = int(np.argmax(vals))
        best_name = agents[best_idx].split("(")[0].strip()
        row += f"   {best_name}"
        print(row)

    print(f"  {'---'*37}")

    # Overall row
    row = f"  {'OVERALL':<25s}"
    overalls = []
    for a in agents:
        o = all_scores[a][1]
        overalls.append(o)
        row += f"  {o:12.3f}"
    best_idx = int(np.argmax(overalls))
    best_name = agents[best_idx].split("(")[0].strip()
    row += f"   {best_name}"
    print(row)

    # Summary metrics
    print(f"\n{'='*110}")
    print(f"  SUMMARY METRICS")
    print(f"{'='*110}")

    summary_metrics = ["total_reward", "total_c1", "total_c2", "total_c3", "total_c4",
                       "c1_violation_rate", "c4_violation_rate"]
    header = f"  {'Metric':<25s}"
    for a in agents:
        short = a.split("(")[0].strip()
        header += f"  {short:>12s}"
    print(header)
    print(f"  {'---'*37}")

    for m in summary_metrics:
        row = f"  {m:<25s}"
        for a in agents:
            v = all_extras[a].get(m, 0)
            row += f"  {v:12.1f}"
        print(row)

    # Forecast sensitivity details
    print(f"\n{'='*110}")
    print(f"  FORECAST SENSITIVITY DETAILS")
    print(f"{'='*110}")
    fc_metrics = ["price_sensitivity", "solar_sensitivity", "all_sensitivity", "forecast_utilization"]
    header = f"  {'Metric':<25s}"
    for a in agents:
        short = a.split("(")[0].strip()
        header += f"  {short:>12s}"
    print(header)
    print(f"  {'---'*37}")
    for m in fc_metrics:
        row = f"  {m:<25s}"
        for a in agents:
            v = all_forecasts[a].get(m, 0)
            row += f"  {v:12.6f}"
        print(row)

    # =========================================================================
    # Save results
    # =========================================================================
    results = {}
    for name in agents:
        scores, overall = all_scores[name]
        forecast = all_forecasts[name]
        extra = all_extras[name]
        results[name] = {
            "overall_intelligence": float(overall),
            "scores": {k: float(v) for k, v in scores.items()},
            "forecast": {k: float(v) for k, v in forecast.items()},
            "summary": {k: float(v) for k, v in extra.items()},
        }

    out_path = f"{OUT_DIR}/intelligence_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
