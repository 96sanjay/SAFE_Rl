#!/usr/bin/env python3
"""
Full Evaluation: R12a (PPOLagMulti + Sauté C1) epoch-50
========================================================

Reports:
  1. Per-constraint violation % with MEANINGFUL denominators
     (C1 = departures with deficit / total departure events)
  2. All CityLearn standard KPIs (from evaluate())
  3. Reward summary
  4. Intelligence Matrix (9 metrics + forecast utilization)

Key: builds env WITHOUT Sauté wrapper so we see raw costs,
but manually tracks budget + appends it to obs (actor expects 199 dims).
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

EVAL_SEED = 42
random.seed(EVAL_SEED)
np.random.seed(EVAL_SEED)
torch.manual_seed(EVAL_SEED)

# ── R12a environment profile ──
# From run_r12a.sh (no Sauté wrapper at eval — we track budget manually)
ENV_VARS = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_PNORM_P": "4.0",
    "STEMS_LAMBDA_EV": "0.0",
    "STEMS_ALPHA_BARRIER": "0.5",
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
    "STEMS_BETA_RAMP": "1.5",
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.05",
    "CITYLEARN_EV_COST_SCALE": "3.0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_C3_CONTROLLABLE": "0",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    # Disable Sauté (we handle budget manually)
    "CITYLEARN_EV_SAUTE": "0",
    # Disable KPI logger
    "CITYLEARN_KPI_RUN_NAME": "__eval_disabled__",
}

# Set all env vars
for k, v in ENV_VARS.items():
    os.environ[k] = v

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env
import citylearn_safe.schema_index as si

CKPT = f"{PROJECT}/runs/r12a_saute_c1/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-03-10-20-35-24/torch_save/epoch-50.pt"

# Sauté budget config (for manual tracking)
SAUTE_BUDGET_D = 1500.0
SAUTE_GAMMA = 1.0


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
    print(f"  Checkpoint: obs_dim={obs_dim}, act_dim={act_dim}, hidden=({h1},{h2})")

    actor = MLPActor(obs_dim, act_dim, (h1, h2))
    filtered = {k: v for k, v in pi_state.items() if not k.startswith("log_std")}
    actor.load_state_dict(filtered, strict=False)
    actor.eval()

    obs_mean = obs_std = obs_clip = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        obs_clip = float(norm["_clip"].float().mean())
        print(f"  Obs normalizer loaded: mean.shape={obs_mean.shape}, clip={obs_clip}")

    return actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim


def augment_obs_with_budget(obs, budget):
    """Append Sauté budget dim to obs (matching training wrapper)."""
    obs_flat = np.asarray(obs, dtype=np.float32).ravel()
    budget_clipped = np.clip(budget, -1.0, 1.0)
    return np.append(obs_flat, budget_clipped)


def run_episode(actor, obs_mean, obs_std, obs_clip):
    """Run 1 deterministic 8760-step episode. Returns all data."""
    si._CACHE = None
    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)

    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(getattr(raw, "buildings", []))

    # Action indices
    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)
    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]
    print(f"  Actions: {names}")
    print(f"  batt_idx={batt_idx}, ev_idx={ev_idx}")

    # Price percentiles
    try:
        pr = buildings[0].pricing.electricity_pricing
        all_prices = np.array(pr, dtype=float)
        p25 = np.percentile(all_prices, 25)
        p50 = np.percentile(all_prices, 50)
        p75 = np.percentile(all_prices, 75)
    except Exception:
        p25, p50, p75 = 0.12, 0.16, 0.20
    print(f"  Price percentiles: P25={p25:.4f}, P50={p50:.4f}, P75={p75:.4f}")

    data = {
        "hour": [], "price": [], "solar": [],
        "actions": [], "ev_actions": [], "batt_actions": [],
        "c0_dense": [], "c1_departure": [], "c2_soc": [],
        "c3_building": [], "c4_grid": [],
        "ev_departures": [], "ev_deficit_violations": [], "ev_80pct_violations": [],
        "reward": [], "cost": [],
        "obs_raw": [],
    }

    # Manual Sauté budget tracking
    budget = 1.0

    obs, info = env.reset(seed=EVAL_SEED)
    env_obs_dim = len(obs)
    print(f"  Env obs_dim={env_obs_dim} (will augment to {env_obs_dim + 1} with budget)")

    done = False
    step = 0
    while not done:
        # Augment obs with budget dim (matching training)
        obs_aug = augment_obs_with_budget(obs, budget)

        obs_t = torch.as_tensor(obs_aug, dtype=torch.float32).unsqueeze(0)
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

        obs_pre = obs_aug.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated) or bool(truncated)
        step += 1

        # Update Sauté budget (matching wrapper logic)
        raw_ev_dense_cost = float(info.get("cost_ev_dense", 0.0))
        if SAUTE_BUDGET_D > 0:
            budget = budget - raw_ev_dense_cost / SAUTE_BUDGET_D
            if SAUTE_GAMMA != 1.0:
                budget /= SAUTE_GAMMA

        # Collect data
        data["hour"].append(hour)
        data["price"].append(price)
        data["solar"].append(solar)
        data["actions"].append(action.copy())
        data["ev_actions"].append([float(action[i]) for i in ev_idx])
        data["batt_actions"].append([float(action[i]) for i in batt_idx])

        data["c0_dense"].append(raw_ev_dense_cost)
        data["c1_departure"].append(float(info.get("cost_ev_departure", 0.0)))
        data["c2_soc"].append(float(info.get("cost_stems_battery", 0.0)))
        data["c3_building"].append(float(info.get("cost_stems_building_power", 0.0)))
        data["c4_grid"].append(float(info.get("cost_stems_grid_power", 0.0)))

        data["ev_departures"].append(int(info.get("ev_departure_departures", 0)))
        data["ev_deficit_violations"].append(int(info.get("ev_departure_violation_count_deficit", 0)))
        data["ev_80pct_violations"].append(int(info.get("ev_departure_violation_count_80pct", 0)))

        data["reward"].append(float(reward))
        data["cost"].append(float(info.get("cost", 0.0)))
        data["obs_raw"].append(obs_pre)

    # Convert to numpy
    for k in data:
        data[k] = np.array(data[k])

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
        print(f"  WARNING: CityLearn evaluate() failed: {e}")

    env.close()
    return data, cl_kpis, p25, p50, p75, batt_idx, ev_idx


def report_constraints(data):
    """Section 1: Per-constraint costs + meaningful violation %."""
    N = len(data["hour"])
    c0 = data["c0_dense"]
    c1 = data["c1_departure"]
    c2 = data["c2_soc"]
    c3 = data["c3_building"]
    c4 = data["c4_grid"]

    total_departures = int(data["ev_departures"].sum())
    total_deficit = int(data["ev_deficit_violations"].sum())
    total_80pct = int(data["ev_80pct_violations"].sum())

    p = print
    p(f"\n{'='*70}")
    p(f"  1. CONSTRAINT ANALYSIS")
    p(f"{'='*70}")

    p(f"\n  --- Cost Totals ---")
    p(f"    C0 (EV Dense/Shortfall):  {c0.sum():>10.1f}  (steps>0: {(c0>0).sum():>5d} = {100*(c0>0).sum()/N:5.1f}%)")
    p(f"    C1 (EV Departure):        {c1.sum():>10.1f}  (steps>0: {(c1>0).sum():>5d} = {100*(c1>0).sum()/N:5.1f}%)")
    p(f"    C2 (Battery SoC):         {c2.sum():>10.1f}  (steps>0: {(c2>0).sum():>5d} = {100*(c2>0).sum()/N:5.1f}%)")
    p(f"    C3 (Building Power):      {c3.sum():>10.1f}  (steps>0: {(c3>0).sum():>5d} = {100*(c3>0).sum()/N:5.1f}%)")
    p(f"    C4 (Grid Power):          {c4.sum():>10.1f}  (steps>0: {(c4>0).sum():>5d} = {100*(c4>0).sum()/N:5.1f}%)")
    p(f"    TOTAL CMDP cost:          {data['cost'].sum():>10.1f}")

    p(f"\n  --- C1 Meaningful Violation Rate ---")
    p(f"    Total EV departure events:         {total_departures}")
    p(f"    Deficit violations (SoC < req):     {total_deficit}")
    p(f"    80% threshold violations:           {total_80pct}")
    if total_departures > 0:
        p(f"    Deficit violation rate:             {100*total_deficit/total_departures:.1f}% of departures")
        p(f"    80% threshold violation rate:       {100*total_80pct/total_departures:.1f}% of departures")
    else:
        p(f"    (No departure events found)")

    p(f"\n  --- Sauté Budget ---")
    # Budget trajectory: final budget tells us if budget was exhausted
    budget = 1.0
    budgets = []
    for t in range(N):
        budget = budget - float(c0[t]) / SAUTE_BUDGET_D if SAUTE_BUDGET_D > 0 else budget
        budgets.append(budget)
    budgets = np.array(budgets)
    p(f"    Final budget λ:   {budgets[-1]:.4f}")
    p(f"    Min budget λ:     {budgets.min():.4f}")
    p(f"    Steps unsafe (λ≤0): {(budgets <= 0).sum()} = {100*(budgets<=0).sum()/N:.1f}%")
    p(f"    Total EV dense cost: {c0.sum():.1f} vs Budget d={SAUTE_BUDGET_D}")
    if c0.sum() > SAUTE_BUDGET_D:
        p(f"    OVER BUDGET by {c0.sum() - SAUTE_BUDGET_D:.1f}")
    else:
        p(f"    WITHIN BUDGET (margin={SAUTE_BUDGET_D - c0.sum():.1f})")


def report_kpis(cl_kpis):
    """Section 2: CityLearn standard KPIs."""
    p = print
    p(f"\n{'='*70}")
    p(f"  2. CITYLEARN STANDARD KPIs (1.0 = no-op baseline)")
    p(f"{'='*70}\n")

    kpi_labels = [
        ("electricity_consumption_total", "Electricity Consumption"),
        ("carbon_emissions_total", "Carbon Emissions"),
        ("cost_total", "Electricity Cost ($)"),
        ("daily_peak_average", "Daily Peak Average"),
        ("all_time_peak_average", "All-Time Peak Average"),
        ("ramping_average", "Ramping Average"),
        ("1 - Loss of Life Share", "1 - Loss of Life Share"),
        ("zero_net_energy", "Zero Net Energy"),
    ]
    for key, label in kpi_labels:
        val = cl_kpis.get(key, None)
        if val is not None:
            good = val < 1.0
            marker = "  <-- BETTER than baseline" if good else "  <-- WORSE than baseline"
            p(f"    {label:<30s}  {val:.4f}{marker}")
        else:
            p(f"    {label:<30s}  N/A")


def report_reward(data):
    """Section 3: Reward summary."""
    p = print
    rewards = data["reward"]
    N = len(rewards)
    p(f"\n{'='*70}")
    p(f"  3. REWARD SUMMARY")
    p(f"{'='*70}\n")
    p(f"    Total reward:     {rewards.sum():.1f}")
    p(f"    Mean reward/step: {rewards.mean():.4f}")
    p(f"    Std reward/step:  {rewards.std():.4f}")
    p(f"    Min reward/step:  {rewards.min():.4f}")
    p(f"    Max reward/step:  {rewards.max():.4f}")


def report_actions(data, batt_idx, ev_idx):
    """Section 3b: Action statistics."""
    p = print
    actions = data["actions"]
    p(f"\n{'='*70}")
    p(f"  3b. ACTION STATISTICS")
    p(f"{'='*70}\n")
    p(f"    Overall mean:   {actions.mean():.4f}")
    p(f"    Overall std:    {actions.std():.4f}")
    p(f"    Overall |mean|: {np.abs(actions).mean():.4f}")

    if batt_idx:
        batt = actions[:, batt_idx]
        p(f"    Battery mean:   {batt.mean():.4f}")
        p(f"    Battery std:    {batt.std():.4f}")
        charge_pct = 100 * (batt > 0.05).mean()
        discharge_pct = 100 * (batt < -0.05).mean()
        p(f"    Battery charge%:    {charge_pct:.1f}%")
        p(f"    Battery discharge%: {discharge_pct:.1f}%")

    if ev_idx:
        ev = actions[:, ev_idx]
        p(f"    EV mean:        {ev.mean():.4f}")
        p(f"    EV std:         {ev.std():.4f}")
        charge_pct = 100 * (ev > 0.05).mean()
        discharge_pct = 100 * (ev < -0.05).mean()
        p(f"    EV charge%:     {charge_pct:.1f}%")
        p(f"    EV discharge%:  {discharge_pct:.1f}%")


def report_intelligence(data, p25, p50, p75, actor, obs_mean, obs_std, obs_clip):
    """Section 4: Intelligence Matrix (9 metrics + forecast)."""
    hours = data["hour"]
    prices = data["price"]
    solar = data["solar"]
    ev_act = data["ev_actions"]
    batt_act = data["batt_actions"]
    N = len(hours)

    mean_ev = ev_act.mean(axis=1) if ev_act.ndim == 2 else ev_act
    mean_batt = batt_act.mean(axis=1) if batt_act.ndim == 2 else batt_act

    p = print
    p(f"\n{'='*70}")
    p(f"  4. INTELLIGENCE MATRIX")
    p(f"{'='*70}")

    # 1. Solar Preference
    solar_med = np.median(solar[solar > 0]) if (solar > 0).any() else 1.0
    high_solar = solar > solar_med
    low_solar = solar < 0.01
    mean_ev_solar = mean_ev[high_solar].mean() if high_solar.sum() > 0 else 0.0
    mean_ev_nosolar = mean_ev[low_solar].mean() if low_solar.sum() > 0 else 0.0
    solar_diff = mean_ev_solar - mean_ev_nosolar
    p(f"\n  1. Solar Preference:")
    p(f"     EV action (high solar): {mean_ev_solar:+.3f}")
    p(f"     EV action (no solar):   {mean_ev_nosolar:+.3f}")
    p(f"     Diff (solar pref):      {solar_diff:+.3f}  {'GOOD' if solar_diff > 0.05 else 'WEAK' if solar_diff > 0 else 'BAD'}")

    # 2. Price Spread
    expensive = prices > p75
    cheap = prices < p25
    mean_batt_exp = mean_batt[expensive].mean() if expensive.sum() > 0 else 0.0
    mean_batt_chp = mean_batt[cheap].mean() if cheap.sum() > 0 else 0.0
    price_batt_diff = mean_batt_chp - mean_batt_exp
    p(f"\n  2. Price Spread (Battery):")
    p(f"     Batt (expensive): {mean_batt_exp:+.3f}")
    p(f"     Batt (cheap):     {mean_batt_chp:+.3f}")
    p(f"     Spread:           {price_batt_diff:+.3f}  {'GOOD' if price_batt_diff > 0.1 else 'WEAK' if price_batt_diff > 0 else 'BAD'}")

    # 3. V2G Timing
    batt_discharge_expensive = (mean_batt[expensive] < -0.05).mean() if expensive.sum() > 0 else 0.0
    p(f"\n  3. V2G Timing (expensive hours):")
    p(f"     Batt discharge rate: {batt_discharge_expensive*100:.1f}%")

    # 4. EV Compliance
    c1_rate = 100 * (data["c1_departure"] > 0).sum() / N
    total_deps = int(data["ev_departures"].sum())
    total_def = int(data["ev_deficit_violations"].sum())
    c1_meaningful = 100 * total_def / total_deps if total_deps > 0 else 0.0
    p(f"\n  4. EV Compliance:")
    p(f"     C1 departure violation (step%): {c1_rate:.1f}%")
    p(f"     C1 departure violation (event%): {c1_meaningful:.1f}% ({total_def}/{total_deps})")

    # 5. Grid Stability
    c4_rate = 100 * (data["c4_grid"] > 0).sum() / N
    p(f"\n  5. Grid Stability:")
    p(f"     C4 violation rate: {c4_rate:.1f}%")

    # 6. Pre-Peak Planning
    hourly_batt = np.zeros(24)
    hourly_ev = np.zeros(24)
    hourly_price = np.zeros(24)
    hourly_solar_h = np.zeros(24)
    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            hourly_batt[h] = mean_batt[mask].mean()
            hourly_ev[h] = mean_ev[mask].mean()
            hourly_price[h] = prices[mask].mean()
            hourly_solar_h[h] = solar[mask].mean()

    peak_hours = {h for h in range(24) if hourly_price[h] > p75}
    pre_peak_hours = set()
    for ph in peak_hours:
        for offset in [2, 3, 4]:
            pre_h = (ph - offset) % 24
            if pre_h not in peak_hours:
                pre_peak_hours.add(pre_h)

    pre_peak_mask = np.isin(hours, list(pre_peak_hours)) if pre_peak_hours else np.zeros(N, bool)
    peak_mask = np.isin(hours, list(peak_hours)) if peak_hours else np.zeros(N, bool)
    pre_peak_batt = mean_batt[pre_peak_mask].mean() if pre_peak_mask.sum() > 0 else 0
    peak_batt = mean_batt[peak_mask].mean() if peak_mask.sum() > 0 else 0
    swing = pre_peak_batt - peak_batt
    p(f"\n  6. Pre-Peak Planning:")
    p(f"     Peak hours:     {sorted(peak_hours)}")
    p(f"     Pre-peak hours: {sorted(pre_peak_hours)}")
    p(f"     Batt pre-peak:  {pre_peak_batt:+.3f}")
    p(f"     Batt at peak:   {peak_batt:+.3f}")
    p(f"     Swing:          {swing:+.3f}  {'PLANNING' if swing > 0.1 else 'NO PLANNING'}")

    # 7. Price Correlation
    corr_batt_price = np.corrcoef(prices, mean_batt)[0, 1]
    p(f"\n  7. Price-Battery Correlation:")
    p(f"     r = {corr_batt_price:+.3f}  {'GOOD' if corr_batt_price < -0.1 else 'WEAK' if corr_batt_price < 0 else 'BAD'}")

    # 8. Behavioral Diversity
    action_variance = np.std([hourly_batt[h] for h in range(24)])
    n_charge = sum(1 for h in range(24) if hourly_batt[h] > 0.05)
    n_discharge = sum(1 for h in range(24) if hourly_batt[h] < -0.05)
    p(f"\n  8. Behavioral Diversity:")
    p(f"     Batt charge hours:    {n_charge}/24")
    p(f"     Batt discharge hours: {n_discharge}/24")
    p(f"     Hourly action StdDev: {action_variance:.3f}  {'DIVERSE' if action_variance > 0.1 else 'FLAT'}")

    # 9. Forecast Utilization (perturbation test)
    obs_raw = data["obs_raw"]
    obs_dim = obs_raw.shape[1]
    base_obs_dim = 70
    fc_start = base_obs_dim
    n_forecast = 128
    SCALE = 0.1

    @torch.no_grad()
    def batched_forward(obs_np):
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        if obs_mean is not None:
            obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
            if obs_clip is not None:
                obs_t = obs_t.clamp(-obs_clip, obs_clip)
        return np.clip(actor(obs_t).numpy(), -1.0, 1.0)

    actions_orig = batched_forward(obs_raw)

    price_range = (fc_start, min(fc_start + 24, obs_dim))
    solar_range = (fc_start + 48, min(fc_start + 72, obs_dim))
    all_range = (fc_start, min(fc_start + n_forecast, obs_dim))

    deltas = {}
    for name, (s, e) in [("price", price_range), ("solar", solar_range), ("all", all_range)]:
        perturbed = obs_raw.copy()
        perturbed[:, s:e] = 0.0
        acts_p = batched_forward(perturbed)
        deltas[name] = float(np.mean(np.abs(actions_orig - acts_p)))

    price_score = float(np.clip(deltas["price"] / SCALE, 0, 1))
    solar_score = float(np.clip(deltas["solar"] / SCALE, 0, 1))
    all_score = float(np.clip(deltas["all"] / SCALE, 0, 1))
    forecast_util = float(np.mean([price_score, solar_score, all_score]))

    p(f"\n  9. Forecast Utilization:")
    p(f"     Price sensitivity: {deltas['price']:.6f} (score={price_score:.2f})")
    p(f"     Solar sensitivity: {deltas['solar']:.6f} (score={solar_score:.2f})")
    p(f"     All FC sensitivity: {deltas['all']:.6f} (score={all_score:.2f})")
    p(f"     Forecast utilization: {forecast_util:.2f}")

    # ── SCORECARD ──
    scores = {
        "solar_preference": float(np.clip(solar_diff / 0.2, 0, 1)),
        "price_spread": float(np.clip(price_batt_diff / 0.3, 0, 1)),
        "v2g_timing": float(np.clip(batt_discharge_expensive, 0, 1)),
        "ev_compliance": float(np.clip(1.0 - c1_meaningful / 10.0, 0, 1)),
        "grid_stability": float(np.clip(1.0 - c4_rate / 30.0, 0, 1)),
        "pre_peak_planning": float(np.clip(swing / 0.3, 0, 1)),
        "price_correlation": float(np.clip(-corr_batt_price / 0.3, 0, 1)),
        "behavioral_diversity": float(np.clip(action_variance / 0.15, 0, 1)),
        "forecast_utilization": forecast_util,
    }
    overall = float(np.mean(list(scores.values())))

    p(f"\n  {'─'*60}")
    p(f"  INTELLIGENCE SCORECARD")
    p(f"  {'─'*60}")
    p(f"  {'Metric':<25s}  {'Score':>6s}  {'Rating'}")
    p(f"  {'─'*60}")
    for name, score in scores.items():
        rating = "***" if score > 0.7 else "** " if score > 0.4 else "*  " if score > 0.1 else ".  "
        bar_len = int(score * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        p(f"  {name:<25s}  {score:5.2f}   {bar} {rating}")
    p(f"  {'─'*60}")
    verdict = "INTELLIGENT" if overall > 0.5 else "LEARNING" if overall > 0.3 else "DUMB"
    p(f"  {'OVERALL':<25s}  {overall:5.2f}   {verdict}")
    p(f"  {'─'*60}")

    # Hourly profile table
    p(f"\n  --- Hourly Action Profile ---")
    p(f"  {'Hr':>4s}  {'Batt':>7s}  {'EV':>7s}  {'Price':>7s}  {'Solar':>7s}  Notes")
    for h in range(24):
        notes = ""
        if hourly_batt[h] > 0.05: notes += "B:chg "
        elif hourly_batt[h] < -0.05: notes += "B:dis "
        if hourly_ev[h] > 0.1: notes += "EV:chg "
        elif hourly_ev[h] < -0.05: notes += "EV:v2g "
        if hourly_solar_h[h] > solar_med and hourly_solar_h[h] > 0: notes += "[solar]"
        if hourly_price[h] > p75: notes += "[PEAK$]"
        elif hourly_price[h] < p25: notes += "[cheap]"
        p(f"  {h:4d}  {hourly_batt[h]:+7.3f}  {hourly_ev[h]:+7.3f}  {hourly_price[h]:7.4f}  {hourly_solar_h[h]:7.2f}  {notes}")

    return scores, overall


def main():
    p = print
    p(f"\n{'='*70}")
    p(f"  R12a FULL EVALUATION: PPOLagMulti + Sauté C1 (epoch-50)")
    p(f"  Checkpoint: {CKPT}")
    p(f"{'='*70}\n")

    actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(CKPT)

    p(f"\n  Running 8760-step episode (seed={EVAL_SEED})...")
    data, cl_kpis, p25, p50, p75, batt_idx, ev_idx = run_episode(
        actor, obs_mean, obs_std, obs_clip
    )
    p(f"  Episode complete: {len(data['hour'])} steps")

    report_constraints(data)
    report_kpis(cl_kpis)
    report_reward(data)
    report_actions(data, batt_idx, ev_idx)
    report_intelligence(data, p25, p50, p75, actor, obs_mean, obs_std, obs_clip)

    p(f"\n{'='*70}")
    p(f"  EVALUATION COMPLETE")
    p(f"{'='*70}\n")


if __name__ == "__main__":
    main()
