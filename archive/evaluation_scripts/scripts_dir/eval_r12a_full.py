#!/usr/bin/env python3
"""
Comprehensive R12a Evaluation: Per-constraint costs, CityLearn KPIs, Intelligence metrics.

R12a: PPOLagMulti + Sauté MDP for C1 (EV charging)
  - obs_dim depends on checkpoint (198 base + Sauté augmentation)
  - SPATIAL_OBS=0, C3_CONTROLLABLE=0
  - Sauté MDP enabled (but eval doesn't need Sauté — just loads weights)
  - C2 disabled (COST_W_C2=0.0)

Usage:
    python scripts/eval_r12a_full.py
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

EVAL_SEED = 42

# =========================================================================
# R12a environment profile
# =========================================================================
R12A_ENV_VARS = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_PNORM_P": "4.0",
    # R12a specific
    "STEMS_LAMBDA_EV": "0.0",
    "STEMS_ALPHA_BARRIER": "0.5",
    "COST_W_C2": "0.0",
    "STEMS_BETA_RAMP": "1.5",
    "COST_W_C3": "5.0",
    # Base cost weights
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.05",
    # EV settings
    "CITYLEARN_EV_COST_SCALE": "3.0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_KPI_RUN_NAME": "__eval_disabled__",
    # No spatial/temporal
    "CITYLEARN_C3_CONTROLLABLE": "0",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    # Sauté MDP — enable for eval to match training env
    "CITYLEARN_EV_SAUTE": "1",
    "CITYLEARN_EV_SAUTE_BUDGET": "1500",
    "CITYLEARN_EV_SAUTE_PENALTY": "5.0",
    "CITYLEARN_EV_SAUTE_GAMMA": "1.0",
}

# Clean all potentially stale env vars, then set R12a profile
stale_keys = [
    "CITYLEARN_C3_CONTROLLABLE", "CITYLEARN_SPATIAL_OBS",
    "COST_W_C1", "COST_W_C1_DENSE", "COST_W_C2", "COST_W_C3", "COST_W_C4",
    "CITYLEARN_EV_DENSE_COST_SCALE", "STEMS_ALPHA_GRID", "STEMS_BETA_RAMP",
    "STEMS_LAMBDA_EV", "STEMS_ALPHA_BARRIER", "STEMS_MU_ECONOMIC",
    "STEMS_ALPHA_BUILD", "STEMS_XI_RENEWABLE", "STEMS_SB_ASYMMETRIC",
    "STEMS_SG_EXPORT_CREDIT", "STEMS_LAMBDA_EV",
    "CITYLEARN_EV_SAUTE", "CITYLEARN_EV_SAUTE_BUDGET",
    "CITYLEARN_EV_SAUTE_PENALTY", "CITYLEARN_EV_SAUTE_GAMMA",
    "CITYLEARN_PID_LAGRANGE",
]
for k in stale_keys:
    os.environ.pop(k, None)
for k, v in R12A_ENV_VARS.items():
    os.environ[k] = v

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.saute_ev_wrapper import SauteEVBudgetWrapper
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env
import citylearn_safe.schema_index as si


R12A_CKPT = (
    f"{PROJECT}/runs/r12a_saute_c1/5bld/"
    "PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-10-20-35-24/torch_save/epoch-50.pt"
)


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

    print(f"  Loaded: obs_dim={obs_dim}, act_dim={act_dim}, h=[{h1},{h2}]")
    if obs_mean is not None:
        print(f"  Obs normalizer: mean_norm={obs_mean.norm():.2f}, std_mean={obs_std.mean():.4f}, clip={obs_clip}")

    return actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim


def run_episode(actor, obs_mean, obs_std, obs_clip, seed=42):
    """Run 1 deterministic episode collecting all metrics."""
    si._CACHE = None
    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)
    # Add Sauté wrapper to match training (198 → 199 dims)
    env = SauteEVBudgetWrapper(env)

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

    print(f"  batt_idx={batt_idx}, ev_idx={ev_idx}")
    print(f"  env obs_dim={env.observation_space.shape[0]}")
    print(f"  price percentiles: P25={p25:.4f} P50={p50:.4f} P75={p75:.4f}")

    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    rewards, costs, actions_all = [], [], []
    c0_vals, c1_vals, c2_vals, c3_vals, c4_vals = [], [], [], [], []
    hours, prices_ts, solar_ts, net_loads = [], [], [], []
    ev_actions_ts, batt_actions_ts = [], []
    obs_raw_all = []

    # EV departure tracking (for correct C1 violation rate)
    ev_departure_counts = []       # number of departures per step
    ev_departure_deficit_counts = []  # number of departures with deficit > 0 per step
    ev_departure_80pct_counts = []   # number of departures below 80% charge

    # Per-building tracking
    per_building_power = {i: [] for i in range(len(buildings))}

    while not done:
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_raw_all.append(obs_np.copy())

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
        hours.append(hour)

        try:
            price = float(buildings[0].pricing.electricity_pricing[max(0, t_now - 1)])
        except Exception:
            price = 0.17
        prices_ts.append(price)

        solar = 0.0
        for b in buildings:
            try:
                s = getattr(b, "solar_generation", None)
                if s is not None and len(s) > max(0, t_now - 1):
                    solar += abs(float(s[max(0, t_now - 1)]))
            except Exception:
                pass
        solar_ts.append(solar)

        ev_actions_ts.append([float(action[i]) for i in ev_idx])
        batt_actions_ts.append([float(action[i]) for i in batt_idx])

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

        # EV departure event tracking
        ev_departure_counts.append(int(info.get("ev_departure_departures", 0)))
        ev_departure_deficit_counts.append(int(info.get("ev_departure_violation_count_deficit", 0)))
        ev_departure_80pct_counts.append(int(info.get("ev_departure_violation_count_80pct", 0)))

        net = 0.0
        for bi, b in enumerate(buildings):
            try:
                nec = getattr(b, "net_electricity_consumption", None)
                if nec is not None and len(nec) > max(0, t_now - 1):
                    val = float(nec[max(0, t_now - 1)])
                    net += val
                    per_building_power[bi].append(val)
            except Exception:
                per_building_power[bi].append(0.0)
        net_loads.append(net)

    # CityLearn evaluate()
    cl_kpis = {}
    try:
        city = base
        for _ in range(20):
            if hasattr(city, 'buildings') and len(getattr(city, 'buildings', [])) > 0:
                break
            city = getattr(city, 'env', getattr(city, 'base', getattr(city, 'unwrapped', None)))
        if city and hasattr(city, 'evaluate'):
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
        print(f"  CityLearn evaluate() error: {e}")

    env.close()

    return {
        "hours": np.array(hours),
        "prices": np.array(prices_ts),
        "solar": np.array(solar_ts),
        "actions": np.array(actions_all),
        "ev_actions": np.array(ev_actions_ts),
        "batt_actions": np.array(batt_actions_ts),
        "rewards": np.array(rewards),
        "costs": np.array(costs),
        "c0": np.array(c0_vals),
        "c1": np.array(c1_vals),
        "c2": np.array(c2_vals),
        "c3": np.array(c3_vals),
        "c4": np.array(c4_vals),
        "net_loads": np.array(net_loads),
        "obs_raw": np.array(obs_raw_all),
        "cl_kpis": cl_kpis,
        "batt_idx": batt_idx,
        "ev_idx": ev_idx,
        "p25": p25, "p50": p50, "p75": p75,
        "per_building_power": per_building_power,
        "total_steps": step,
        # EV departure events
        "ev_departures_total": sum(ev_departure_counts),
        "ev_departures_with_deficit": sum(ev_departure_deficit_counts),
        "ev_departures_below_80pct": sum(ev_departure_80pct_counts),
        "ev_departure_counts": np.array(ev_departure_counts),
        "ev_departure_deficit_counts": np.array(ev_departure_deficit_counts),
    }


def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def evaluate_constraints(d):
    """Section 1: Per-constraint violation rates and cost totals."""
    print_section("1. PER-CONSTRAINT COST ANALYSIS")
    N = d["total_steps"]

    # --- C1 (EV Departure): violation rate = deficit departures / total departures ---
    total_departures = d["ev_departures_total"]
    deficit_departures = d["ev_departures_with_deficit"]
    below_80_departures = d["ev_departures_below_80pct"]
    c1_viol_pct = 100.0 * deficit_departures / total_departures if total_departures > 0 else 0.0
    c1_80pct_viol_pct = 100.0 * below_80_departures / total_departures if total_departures > 0 else 0.0

    # --- C2 (Battery SoC): per-step, denominator = total steps ---
    c2_viol_steps = (d["c2"] > 0).sum()
    c2_viol_pct = 100.0 * c2_viol_steps / N

    # --- C3 (Building Power): per-step ---
    c3_viol_steps = (d["c3"] > 0).sum()
    c3_viol_pct = 100.0 * c3_viol_steps / N

    # --- C4 (Grid Power): per-step ---
    c4_viol_steps = (d["c4"] > 0).sum()
    c4_viol_pct = 100.0 * c4_viol_steps / N

    # --- C0 (EV Dense): zeroed by Saute ---
    c0_total = d["c0"].sum()

    print(f"\n  C1: EV DEPARTURE (event-based metric)")
    print(f"    Total departures:                   {total_departures}")
    print(f"    Departures with ANY deficit:         {deficit_departures}  ({c1_viol_pct:.1f}%)")
    print(f"    Departures below 80% charge:         {below_80_departures}  ({c1_80pct_viol_pct:.1f}%)")
    print(f"    Total C1 cost (sum):                 {d['c1'].sum():.1f}")
    print(f"    Mean deficit per departure:          {d['c1'].sum() / max(1, total_departures):.3f}")

    print(f"\n  C2: BATTERY SoC (per-step metric)")
    print(f"    Steps with violation:                {c2_viol_steps} / {N}  ({c2_viol_pct:.1f}%)")
    print(f"    Total C2 cost (sum):                 {d['c2'].sum():.1f}")
    print(f"    Mean per step:                       {d['c2'].mean():.4f}")
    print(f"    Max single step:                     {d['c2'].max():.3f}")
    print(f"    NOTE: C2 weight=0 during training (disabled)")

    print(f"\n  C3: BUILDING POWER (per-step metric)")
    print(f"    Steps with violation:                {c3_viol_steps} / {N}  ({c3_viol_pct:.1f}%)")
    print(f"    Total C3 cost (sum):                 {d['c3'].sum():.1f}")
    print(f"    Mean per step:                       {d['c3'].mean():.4f}")
    print(f"    Max single step:                     {d['c3'].max():.3f}")

    print(f"\n  C4: GRID POWER (per-step metric)")
    print(f"    Steps with violation:                {c4_viol_steps} / {N}  ({c4_viol_pct:.1f}%)")
    print(f"    Total C4 cost (sum):                 {d['c4'].sum():.1f}")
    print(f"    Mean per step:                       {d['c4'].mean():.4f}")
    print(f"    Max single step:                     {d['c4'].max():.3f}")

    print(f"\n  C0: EV DENSE CHARGING (zeroed by Saute)")
    print(f"    Total C0 cost:                       {c0_total:.1f}")

    # Summary table
    print(f"\n  SUMMARY TABLE:")
    print(f"  {'Constraint':<25s} {'Total Cost':>12s} {'Violation Rate':>15s} {'Denominator':>15s} {'vs Limit':>12s} {'Status':>8s}")
    print(f"  {'-'*93}")

    limits = {"C0": 1800, "C1": 1500, "C2": 4000, "C3": 12000, "C4": 2500}
    rows = [
        ("C0: EV Dense", d["c0"].sum(), 0.0, "N/A (Saute)", limits["C0"]),
        ("C1: EV Departure", d["c1"].sum(), c1_viol_pct, f"{deficit_departures}/{total_departures} deps", limits["C1"]),
        ("C2: Battery SoC", d["c2"].sum(), c2_viol_pct, f"{c2_viol_steps}/{N} steps", limits["C2"]),
        ("C3: Building Power", d["c3"].sum(), c3_viol_pct, f"{c3_viol_steps}/{N} steps", limits["C3"]),
        ("C4: Grid Power", d["c4"].sum(), c4_viol_pct, f"{c4_viol_steps}/{N} steps", limits["C4"]),
    ]
    for name, total, viol, denom, limit in rows:
        ratio = total / limit if limit > 0 else 0
        status = "OK" if total <= limit else f"{ratio:.1f}x"
        print(f"  {name:<25s} {total:12.1f} {viol:14.1f}% {denom:>15s} {total:7.0f}/{limit:<5d} {status:>8s}")

    # Total CMDP cost
    total_cost = d["costs"].sum()
    cost_steps = (d["costs"] > 0).sum()
    print(f"\n  Total CMDP Cost: {total_cost:.1f}  ({cost_steps} steps with cost > 0, {100*cost_steps/N:.1f}%)")

    # Weighted contributions
    w_ev, w_soc, w_bld, w_grid = 1.0, 10.0, 0.5, 0.05
    wc1 = d["c1"].sum() * w_ev
    wc2 = d["c2"].sum() * w_soc
    wc3 = d["c3"].sum() * w_bld
    wc4 = d["c4"].sum() * w_grid
    print(f"\n  Weighted Cost Contributions:")
    print(f"    C1 (EV depart):   {wc1:10.1f}  (w={w_ev})")
    print(f"    C2 (Battery SoC): {wc2:10.1f}  (w={w_soc}, disabled during training)")
    print(f"    C3 (Building):    {wc3:10.1f}  (w={w_bld})")
    print(f"    C4 (Grid):        {wc4:10.1f}  (w={w_grid})")


def evaluate_reward(d):
    """Section 2: Reward analysis."""
    print_section("2. REWARD ANALYSIS")
    rewards = d["rewards"]
    print(f"  Total Reward:     {rewards.sum():12.1f}")
    print(f"  Mean Reward/Step: {rewards.mean():12.4f}")
    print(f"  Std Reward/Step:  {rewards.std():12.4f}")
    print(f"  Min Step Reward:  {rewards.min():12.4f}")
    print(f"  Max Step Reward:  {rewards.max():12.4f}")


def evaluate_citylearn_kpis(d):
    """Section 3: CityLearn standard KPIs."""
    print_section("3. CITYLEARN STANDARD KPIs (1.0 = no-op baseline)")
    kpis = d["cl_kpis"]
    if not kpis:
        print("  No KPIs available (evaluate() returned empty)")
        return

    kpi_labels = [
        ("electricity_consumption_total", "Electricity Consumption"),
        ("carbon_emissions_total", "Carbon Emissions"),
        ("cost_total", "Electricity Cost ($)"),
        ("daily_peak_average", "Daily Peak Average"),
        ("all_time_peak_average", "All-Time Peak Average"),
        ("ramping_average", "Ramping Average"),
        ("1 - Load Factor", "1 - Load Factor"),
        ("1 - Loss of Life Share", "1 - Loss of Life Share"),
        ("zero_net_energy", "Zero Net Energy"),
    ]

    print(f"\n  {'KPI':<35s} {'Value':>10s} {'vs Baseline':>12s}")
    print(f"  {'-'*57}")
    for key, label in kpi_labels:
        val = kpis.get(key)
        if val is not None:
            delta = val - 1.0
            direction = "BETTER" if val < 1.0 else "WORSE" if val > 1.0 else "SAME"
            print(f"  {label:<35s} {val:10.4f} {delta:+10.4f}  ({direction})")
        else:
            print(f"  {label:<35s} {'N/A':>10s}")

    # Average score (lower is better for most KPIs)
    scoreable = [v for k, v in kpis.items() if isinstance(v, (int, float))]
    if scoreable:
        avg = np.mean(scoreable)
        print(f"\n  Average KPI Score: {avg:.4f}  ({'GOOD' if avg < 1.0 else 'POOR'})")


def evaluate_actions(d):
    """Section 4: Action statistics."""
    print_section("4. ACTION STATISTICS")
    actions = d["actions"]
    batt_idx = d["batt_idx"]
    ev_idx = d["ev_idx"]

    print(f"\n  Overall:")
    print(f"    Mean:     {actions.mean():+.4f}")
    print(f"    Std:      {actions.std():.4f}")
    print(f"    |Mean|:   {np.abs(actions).mean():.4f}")

    if batt_idx:
        batt_acts = actions[:, batt_idx]
        print(f"\n  Battery Actions (indices {batt_idx}):")
        print(f"    Mean:     {batt_acts.mean():+.4f}")
        print(f"    Std:      {batt_acts.std():.4f}")
        print(f"    Charge %: {100*(batt_acts > 0.05).mean():.1f}%")
        print(f"    Disch %:  {100*(batt_acts < -0.05).mean():.1f}%")
        print(f"    Idle %:   {100*((batt_acts >= -0.05) & (batt_acts <= 0.05)).mean():.1f}%")
        for i, bi in enumerate(batt_idx):
            print(f"    Bldg {i}: mean={actions[:, bi].mean():+.3f} std={actions[:, bi].std():.3f}")

    if ev_idx:
        ev_acts = actions[:, ev_idx]
        print(f"\n  EV Charger Actions (indices {ev_idx}):")
        print(f"    Mean:     {ev_acts.mean():+.4f}")
        print(f"    Std:      {ev_acts.std():.4f}")
        print(f"    Charge %: {100*(ev_acts > 0.05).mean():.1f}%")
        print(f"    V2G %:    {100*(ev_acts < -0.05).mean():.1f}%")
        print(f"    Idle %:   {100*((ev_acts >= -0.05) & (ev_acts <= 0.05)).mean():.1f}%")


def evaluate_intelligence(d, actor, obs_mean, obs_std, obs_clip):
    """Section 5: Intelligence metrics."""
    print_section("5. INTELLIGENCE EVALUATION")

    hours = d["hours"]
    prices = d["prices"]
    solar = d["solar"]
    p25, p50, p75 = d["p25"], d["p50"], d["p75"]
    N = d["total_steps"]

    ev_act = d["ev_actions"]
    batt_act = d["batt_actions"]
    mean_ev = ev_act.mean(axis=1) if ev_act.ndim == 2 and ev_act.shape[1] > 0 else np.zeros(N)
    mean_batt = batt_act.mean(axis=1) if batt_act.ndim == 2 and batt_act.shape[1] > 0 else np.zeros(N)

    # A. TASK INTELLIGENCE
    print(f"\n  -- A. TASK INTELLIGENCE --\n")

    # 1. Solar charging
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

    # 2. Price-aware V2G
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

    # 3. Off-peak charging
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

    # 4. Constraint compliance (correct denominators)
    total_deps = d["ev_departures_total"]
    deficit_deps = d["ev_departures_with_deficit"]
    c1_rate = 100.0 * deficit_deps / total_deps if total_deps > 0 else 0.0
    c2_rate = 100 * (d["c2"] > 0).sum() / N
    c3_rate = 100 * (d["c3"] > 0).sum() / N
    c4_rate = 100 * (d["c4"] > 0).sum() / N
    print(f"\n  4. Constraint Compliance:")
    print(f"     C1 EV departure:   {c1_rate:.1f}% ({deficit_deps}/{total_deps} departures)  ({'GOOD' if c1_rate < 5 else 'OK' if c1_rate < 20 else 'POOR'})")
    print(f"     C2 Battery SoC:    {c2_rate:.1f}% (steps)")
    print(f"     C3 Building power: {c3_rate:.1f}% (steps)")
    print(f"     C4 Grid power:     {c4_rate:.1f}% (steps)")

    # B. TEMPORAL PLANNING
    print(f"\n  -- B. TEMPORAL PLANNING --\n")

    # Hourly profiles
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

    print(f"  5. Hourly Action Profile:")
    print(f"     Hour  Batt    EV     Price   Solar   Interpretation")
    print(f"     {'-'*65}")
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

    # Pre-peak planning
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

    print(f"\n  6. Pre-Peak Preparation:")
    print(f"     Peak hours:        {sorted(peak_hours)}")
    print(f"     Pre-peak hours:    {sorted(pre_peak_hours)}")
    print(f"     Batt pre-peak:     {pre_peak_batt:+.3f}  {'CHARGING (GOOD)' if pre_peak_batt > 0.05 else 'not charging'}")
    print(f"     Batt at peak:      {peak_batt:+.3f}  {'DISCHARGING (GOOD)' if peak_batt < -0.05 else 'not discharging'}")
    print(f"     Swing:             {pre_peak_batt - peak_batt:+.3f}  {'PLANNING AHEAD' if (pre_peak_batt - peak_batt) > 0.1 else 'NO PLANNING'}")

    # Correlations
    corr_batt_price = np.corrcoef(prices, mean_batt)[0, 1]
    corr_ev_price = np.corrcoef(prices, mean_ev)[0, 1]
    print(f"\n  7. Price-Action Correlation:")
    print(f"     Batt vs Price:  r={corr_batt_price:+.3f}  {'GOOD' if corr_batt_price < -0.1 else 'WEAK' if corr_batt_price < 0 else 'BAD'}")
    print(f"     EV vs Price:    r={corr_ev_price:+.3f}  {'GOOD' if corr_ev_price < -0.1 else 'WEAK' if corr_ev_price < 0 else 'BAD'}")

    solar_mask = solar > 0
    corr_ev_solar = np.corrcoef(solar[solar_mask], mean_ev[solar_mask])[0, 1] if solar_mask.sum() > 100 else 0.0
    print(f"\n  8. Solar-Charge Correlation:")
    print(f"     EV vs Solar:    r={corr_ev_solar:+.3f}  {'GOOD' if corr_ev_solar > 0.1 else 'WEAK' if corr_ev_solar > 0 else 'BAD'}")

    # Behavioral diversity
    n_batt_charge_hours = sum(1 for h in range(24) if hourly_batt[h] > 0.05)
    n_batt_discharge_hours = sum(1 for h in range(24) if hourly_batt[h] < -0.05)
    n_batt_idle_hours = 24 - n_batt_charge_hours - n_batt_discharge_hours
    action_variance = np.std([hourly_batt[h] for h in range(24)])
    print(f"\n  9. Behavioral Diversity:")
    print(f"     Batt charge hours:    {n_batt_charge_hours}/24")
    print(f"     Batt discharge hours: {n_batt_discharge_hours}/24")
    print(f"     Batt idle hours:      {n_batt_idle_hours}/24")
    print(f"     Hourly action StdDev: {action_variance:.3f}  {'DIVERSE' if action_variance > 0.1 else 'FLAT'}")

    # C. FORECAST UTILIZATION
    print(f"\n  -- C. FORECAST UTILIZATION (Perturbation Test) --")
    obs_raw = d["obs_raw"]
    obs_dim = obs_raw.shape[1]
    base_obs_dim = 70
    fc_start = base_obs_dim
    n_forecast = 128

    if fc_start + 72 <= obs_dim:
        price_range = (fc_start, fc_start + 24)
        solar_range = (fc_start + 48, fc_start + 72)
        all_range = (fc_start, fc_start + n_forecast)
        SCALE_THRESHOLD = 0.1

        @torch.no_grad()
        def batched_forward(obs_np_in):
            obs_t = torch.tensor(obs_np_in, dtype=torch.float32)
            if obs_mean is not None:
                obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
                if obs_clip is not None:
                    obs_t = obs_t.clamp(-obs_clip, obs_clip)
            return np.clip(actor(obs_t).numpy(), -1.0, 1.0)

        actions_orig = d["actions"]
        perturbations = {
            "price_forecast": price_range,
            "solar_forecast": solar_range,
            "all_forecast": all_range,
        }
        deltas = {}
        for name, (start, end) in perturbations.items():
            end = min(end, obs_dim)
            obs_pert = obs_raw.copy()
            obs_pert[:, start:end] = 0.0
            act_pert = batched_forward(obs_pert)
            deltas[name] = float(np.mean(np.abs(actions_orig - act_pert)))

        price_score = float(np.clip(deltas["price_forecast"] / SCALE_THRESHOLD, 0, 1))
        solar_score = float(np.clip(deltas["solar_forecast"] / SCALE_THRESHOLD, 0, 1))
        all_score = float(np.clip(deltas["all_forecast"] / SCALE_THRESHOLD, 0, 1))
        forecast_utilization = float(np.mean([price_score, solar_score, all_score]))

        print(f"  10. Price Forecast:  delta={deltas['price_forecast']:.6f}  score={price_score:.2f}")
        print(f"  11. Solar Forecast:  delta={deltas['solar_forecast']:.6f}  score={solar_score:.2f}")
        print(f"  12. All Forecast:    delta={deltas['all_forecast']:.6f}   score={all_score:.2f}")
        print(f"      Forecast Util:   {forecast_utilization:.2f}")
    else:
        forecast_utilization = 0.0
        price_score = solar_score = all_score = 0.0
        print(f"  Skipped: obs_dim={obs_dim} too small for forecast analysis")

    # D. INTELLIGENCE SCORECARD
    print_section("6. INTELLIGENCE SCORECARD")

    scores = {}
    scores["solar_preference"] = float(np.clip(solar_diff / 0.2, 0, 1))
    scores["price_spread"] = float(np.clip(price_batt_diff / 0.3, 0, 1))
    scores["v2g_timing"] = float(np.clip(batt_discharge_expensive, 0, 1))
    scores["ev_compliance"] = float(np.clip(1.0 - c1_rate / 50.0, 0, 1))  # scale: 0%=1.0, 50%=0.0
    scores["grid_stability"] = float(np.clip(1.0 - c4_rate / 30.0, 0, 1))
    scores["pre_peak_planning"] = float(np.clip((pre_peak_batt - peak_batt) / 0.3, 0, 1))
    scores["price_correlation"] = float(np.clip(-corr_batt_price / 0.3, 0, 1))
    scores["behavioral_diversity"] = float(np.clip(action_variance / 0.15, 0, 1))
    scores["forecast_utilization"] = forecast_utilization

    overall = float(np.mean(list(scores.values())))

    print(f"\n  {'Metric':<25s}  {'Score':>6s}  {'Rating':>8s}  {'Bar'}")
    print(f"  {'-'*70}")
    for name, score in scores.items():
        rating = "***" if score > 0.7 else "** " if score > 0.4 else "*  " if score > 0.1 else ".  "
        bar_len = int(score * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"  {name:<25s}  {score:5.2f}   {rating:>8s}  {bar}")

    print(f"\n  {'OVERALL INTELLIGENCE':<25s}  {overall:5.2f}   {'INTELLIGENT' if overall > 0.5 else 'LEARNING' if overall > 0.3 else 'DUMB'}")

    return scores, overall


def main():
    OUT_DIR = f"{PROJECT}/runs/r12a_saute_c1/evaluation"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  COMPREHENSIVE R12a EVALUATION")
    print(f"  R12a: PPOLagMulti + Saute MDP for C1 (EV Charging)")
    print(f"  Checkpoint: epoch-50")
    print(f"  Seed: {EVAL_SEED}")
    print(f"{'='*80}")

    # Load actor
    print(f"\n  Loading checkpoint...")
    actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(R12A_CKPT)

    # Seed everything
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)

    # Run episode
    print(f"\n  Running evaluation episode (8760 steps)...")
    d = run_episode(actor, obs_mean, obs_std, obs_clip, seed=EVAL_SEED)
    print(f"  Episode complete: {d['total_steps']} steps")

    # Determinism check
    print(f"\n  Running determinism check...")
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    d2 = run_episode(actor, obs_mean, obs_std, obs_clip, seed=EVAL_SEED)
    r_match = abs(d["rewards"].sum() - d2["rewards"].sum()) < 1e-4
    c_match = abs(d["costs"].sum() - d2["costs"].sum()) < 1e-4
    if r_match and c_match:
        print(f"  DETERMINISM VERIFIED")
    else:
        print(f"  WARNING: non-deterministic! reward diff={abs(d['rewards'].sum() - d2['rewards'].sum()):.6f}")

    # Run all evaluation sections
    evaluate_constraints(d)
    evaluate_reward(d)
    evaluate_citylearn_kpis(d)
    evaluate_actions(d)
    scores, overall = evaluate_intelligence(d, actor, obs_mean, obs_std, obs_clip)

    # Save results
    N = d["total_steps"]
    total_deps = d["ev_departures_total"]
    deficit_deps = d["ev_departures_with_deficit"]
    results = {
        "run": "R12a (Saute MDP C1)",
        "epoch": 50,
        "seed": EVAL_SEED,
        "total_reward": float(d["rewards"].sum()),
        "total_cost": float(d["costs"].sum()),
        "per_constraint": {
            "C0": {
                "name": "EV Dense Charging",
                "total": float(d["c0"].sum()),
                "note": "zeroed by Saute MDP",
            },
            "C1": {
                "name": "EV Departure Deficit",
                "total": float(d["c1"].sum()),
                "total_departures": total_deps,
                "deficit_departures": deficit_deps,
                "violation_pct_departures": float(100.0 * deficit_deps / total_deps) if total_deps > 0 else 0.0,
                "below_80pct_departures": d["ev_departures_below_80pct"],
                "below_80pct_pct": float(100.0 * d["ev_departures_below_80pct"] / total_deps) if total_deps > 0 else 0.0,
                "mean_deficit_per_departure": float(d["c1"].sum() / max(1, total_deps)),
            },
            "C2": {
                "name": "Battery SoC",
                "total": float(d["c2"].sum()),
                "violation_steps": int((d["c2"] > 0).sum()),
                "violation_pct_steps": float(100.0 * (d["c2"] > 0).sum() / N),
                "mean_per_step": float(d["c2"].mean()),
                "max": float(d["c2"].max()),
                "note": "disabled during training (weight=0)",
            },
            "C3": {
                "name": "Building Power",
                "total": float(d["c3"].sum()),
                "violation_steps": int((d["c3"] > 0).sum()),
                "violation_pct_steps": float(100.0 * (d["c3"] > 0).sum() / N),
                "mean_per_step": float(d["c3"].mean()),
                "max": float(d["c3"].max()),
            },
            "C4": {
                "name": "Grid Power",
                "total": float(d["c4"].sum()),
                "violation_steps": int((d["c4"] > 0).sum()),
                "violation_pct_steps": float(100.0 * (d["c4"] > 0).sum() / N),
                "mean_per_step": float(d["c4"].mean()),
                "max": float(d["c4"].max()),
            },
        },
        "citylearn_kpis": d["cl_kpis"],
        "intelligence_scores": {k: float(v) for k, v in scores.items()},
        "overall_intelligence": float(overall),
        "action_stats": {
            "mean": float(d["actions"].mean()),
            "std": float(d["actions"].std()),
            "batt_mean": float(d["batt_actions"].mean()) if d["batt_actions"].size > 0 else 0,
            "ev_mean": float(d["ev_actions"].mean()) if d["ev_actions"].size > 0 else 0,
        },
    }

    out_json = f"{OUT_DIR}/r12a_eval_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_json}")
    print(f"\n  EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
