#!/usr/bin/env python3
"""
R22 PPO Comprehensive Evaluation (GradS + Lambda Cap 12)
=========================================================
Thorough evaluation of the R22 PPOLagMulti+GradS checkpoint covering:

  1. All constraint violations (C0-C4) — per-step AND per-departure rates
  2. CityLearn KPIs (electricity, carbon, cost, daily peak, ramping, etc.)
  3. Reward decomposition (STEMS components)
  4. V2G behavior analysis (charge/discharge/idle breakdown)
  5. Per-building power analysis
  6. Battery SoC statistics
  7. Action distribution analysis
  8. Energy balance (import/export/solar)
  9. Hourly violation heatmap data
 10. Comparison to constraint limits

Usage:
    python scripts/eval_r22_comprehensive.py [--epoch EPOCH] [--seed SEED]
"""
import os
import sys
import json
import glob as glob_mod
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# =====================================================================
# Eval env vars — match R21 training config but DISABLE training wrappers
# =====================================================================
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
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    # === DISABLE training-only wrappers (keep safety projections) ===
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_PID_LAGRANGE": "0",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_BATT_CLAMP": "1",          # Safety Projection (Dalal 2018) — deployment-time guarantee
    "CITYLEARN_WM_DISABLE": "1",
    # R21 reward weights (match training)
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

# R21 constraint limits (from config)
COST_LIMITS = {
    "C0_ev_departure": 1800,
    "C1_ev_saute": 1500,
    "C2_battery_soc": 4000,
    "C3_building_power": 3000,
    "C4_grid_power": 1500,
    "aggregate": 21800,
}

P_BUILDING_MAX = 4.6083
P_GRID_MAX = 10.2352

RUN_DIR = "runs/r22_ppo/5bld"


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

SOC_UPPER_CLAMP = 0.94
BATT_DT = 1.0


def discover_battery_actions(raw_env):
    """Replicate CityLearnCMDPv2._discover_battery_actions for eval."""
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


def find_checkpoint(epoch=None):
    pattern = os.path.join(PROJECT, RUN_DIR, "PPOLagMulti-*", "seed-*", "torch_save")
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


def divz(a, b, default=0.0):
    return a / b if b > 0 else default


def pct(a, b):
    return 100.0 * divz(a, b)


def evaluate(ckpt_path, seed=42):
    print(f"\n{'='*70}")
    print(f"  R22 PPO (GradS + Lambda Cap 12) — COMPREHENSIVE EVALUATION")
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"  Seed: {seed}")
    print(f"{'='*70}")

    # --- Load model ---
    actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(ckpt_path)
    print(f"  Actor: obs_dim={obs_dim}, act_dim={act_dim}, hidden=[256,256]")
    print(f"  Obs normalizer: {'YES' if obs_mean is not None else 'NO'}")

    # --- Create environment ---
    import citylearn_safe.schema_index as si
    si._CACHE = None
    set_env()

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)
    raw = unwrap_to_raw_citylearn_env(env)
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

    # Safety Projection Layer for batteries (Dalal et al. 2018)
    batt_action_map = discover_battery_actions(raw)
    batt_clamp_count = 0
    print(f"  Battery Safety Projection: {len(batt_action_map)} batteries")
    for act_idx, bld_idx, cap, p_max, eta in batt_action_map:
        print(f"    act[{act_idx}] → Bld {bld_idx}: cap={cap:.1f}kWh, Pmax={p_max:.1f}kW, eta={eta:.2f}")
    print(f"  Action dim: {len(names)} (battery: {len(batt_idx)}, EV: {len(ev_idx)})")
    for i, n in enumerate(names):
        print(f"    [{i}] {n}")

    need_pad = obs_dim > env.observation_space.shape[0]
    pad_dim = obs_dim - env.observation_space.shape[0] if need_pad else 0

    # --- Reset ---
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    # =====================================================================
    # TRACKING VARIABLES
    # =====================================================================

    # Episode-level
    rewards = []
    costs = []
    actions_all = []

    # Per-constraint cumulative costs
    c0_vals, c1_vals, c2_vals, c3_vals, c4_vals = [], [], [], [], []

    # STEMS reward components
    reward_components = defaultdict(list)

    # Energy tracking
    grid_imports = []
    grid_exports = []
    electricity_prices = []
    carbon_intensities = []
    step_costs_monetary = []
    step_carbon = []

    # === EV Departure tracking (THE REAL METRIC) ===
    total_departures = 0
    violated_departures = 0
    departure_deficits = []
    ev_tracker = {}

    # === C2: Battery SoC tracking ===
    soc_violation_steps = 0
    soc_all_steps = []  # list of [soc_b0, soc_b1, ...] per step
    soc_min_per_building = [1.0] * n_buildings
    soc_max_per_building = [0.0] * n_buildings

    # === C3: Building power violations ===
    building_power_violations = [0] * n_buildings
    building_power_max = [0.0] * n_buildings
    building_power_all = [[] for _ in range(n_buildings)]
    c3_violation_steps = 0

    # === C4: Grid power violations ===
    grid_violation_steps = 0
    grid_power_max = 0.0
    grid_power_all = []

    # === V2G behavior ===
    ev_charge_steps = 0
    ev_discharge_steps = 0
    ev_idle_steps = 0
    total_ev_steps = 0
    ev_discharge_kwh = 0.0

    # === Hourly violation tracking (for heatmap) ===
    hourly_c3_violations = [0] * 24
    hourly_c4_violations = [0] * 24
    hourly_counts = [0] * 24

    # === CityLearn KPIs (extracted at episode end) ===
    citylearn_kpis = {}

    # =====================================================================
    # ROLLOUT
    # =====================================================================
    print(f"\n  Running evaluation rollout...")

    while not done:
        # --- Prepare observation ---
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

        # Safety Projection Layer: clamp battery actions to prevent SoC violations
        action_pre_clamp = action.copy()
        action = clamp_battery_actions(action, batt_action_map, raw)
        if not np.array_equal(action, action_pre_clamp):
            batt_clamp_count += 1

        actions_all.append(action.copy())

        t_now = int(getattr(raw, "time_step", 0))
        t_idx = max(0, t_now - 1)
        hour = t_now % 24

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

        # === V2G action tracking ===
        for ei in ev_idx:
            if ei < len(action):
                a = action[ei]
                total_ev_steps += 1
                if a > 0.05:
                    ev_charge_steps += 1
                elif a < -0.05:
                    ev_discharge_steps += 1
                else:
                    ev_idle_steps += 1

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

        # --- STEMS reward components ---
        for key in ["reward_stems_total", "reward_economic", "reward_stability",
                     "reward_stability_grid", "reward_stability_building",
                     "reward_stability_ramp", "reward_renewable", "reward_comfort",
                     "reward_ev_shaping", "reward_bill"]:
            reward_components[key].append(info.get(key, 0.0))

        # --- Energy/monetary tracking ---
        grid_imports.append(info.get("grid_import_kwh", 0.0))
        grid_exports.append(info.get("grid_export_kwh", 0.0))
        electricity_prices.append(info.get("electricity_price", 0.0))
        carbon_intensities.append(info.get("carbon_intensity", 0.0))
        step_costs_monetary.append(info.get("step_cost", 0.0))
        step_carbon.append(info.get("step_carbon_kg", 0.0))

        # --- C2: Battery SoC ---
        soc_viol_any = info.get("battery_soc_violation_any", 0.0)
        if soc_viol_any > 0:
            soc_violation_steps += 1
        # Track per-building SoC
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
                soc_min_per_building[b_idx] = min(soc_min_per_building[b_idx], s)
                soc_max_per_building[b_idx] = max(soc_max_per_building[b_idx], s)
            except Exception:
                soc_step.append(0.5)
        soc_all_steps.append(soc_step)

        # --- C3: Building power ---
        bld_viol = info.get("building_power_violation", 0.0)
        if bld_viol > 0:
            c3_violation_steps += 1
            hourly_c3_violations[hour] += 1
        hourly_counts[hour] += 1
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
            hourly_c4_violations[hour] += 1
        gi = info.get("grid_import_kwh", 0.0)
        grid_power_all.append(gi)
        grid_power_max = max(grid_power_max, gi)

        # --- CityLearn KPIs (only available on last step) ---
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
            print(f"    Step {step}: reward={reward:.2f}, cost={cost:.2f}, "
                  f"C3_cum={sum(c3_vals):.0f}, C4_cum={sum(c4_vals):.0f}")

    # =====================================================================
    # COMPUTE ALL METRICS
    # =====================================================================
    total_reward = sum(rewards)
    total_cost = sum(costs)
    total_c0 = sum(c0_vals)
    total_c1 = sum(c1_vals)
    total_c2 = sum(c2_vals)
    total_c3 = sum(c3_vals)
    total_c4 = sum(c4_vals)

    ev_viol_pct = pct(violated_departures, total_departures)
    mean_deficit = np.mean(departure_deficits) if departure_deficits else 0.0
    violated_deficits = [d for d in departure_deficits if d > 0.01]

    actions_arr = np.array(actions_all)
    batt_actions = actions_arr[:, batt_idx] if batt_idx else np.array([])
    ev_actions = actions_arr[:, ev_idx] if ev_idx else np.array([])

    soc_arr = np.array(soc_all_steps)  # (steps, n_buildings)

    # =====================================================================
    # PRINT COMPREHENSIVE RESULTS
    # =====================================================================
    W = 70

    def section(title):
        print(f"\n{'='*W}")
        print(f"  {title}")
        print(f"{'='*W}")

    def subsection(title):
        print(f"\n  --- {title} ---")

    section("1. EPISODE SUMMARY")
    print(f"  Steps:              {step}")
    print(f"  Total Reward:       {total_reward:.2f}")
    print(f"  Mean Step Reward:   {total_reward/step:.4f}")
    print(f"  Total Cost (CMDP):  {total_cost:.2f}")
    print(f"  Cost Limit (agg):   {COST_LIMITS['aggregate']}")
    print(f"  Cost/Limit ratio:   {total_cost/COST_LIMITS['aggregate']:.2f}x")

    section("2. CONSTRAINT VIOLATION SUMMARY")
    print(f"  {'Constraint':<25} {'EpCost':>10} {'Limit':>10} {'Ratio':>8} {'Status':>10}")
    print(f"  {'-'*63}")
    constraints = [
        ("C0 EV departure", total_c0, COST_LIMITS["C0_ev_departure"]),
        ("C1 EV Sauté", total_c1, COST_LIMITS["C1_ev_saute"]),
        ("C2 Battery SoC", total_c2, COST_LIMITS["C2_battery_soc"]),
        ("C3 Building power", total_c3, COST_LIMITS["C3_building_power"]),
        ("C4 Grid power", total_c4, COST_LIMITS["C4_grid_power"]),
    ]
    for name, cost_val, limit in constraints:
        ratio = cost_val / limit if limit > 0 else 0
        status = "OK" if cost_val <= limit else "VIOLATED"
        print(f"  {name:<25} {cost_val:>10.1f} {limit:>10} {ratio:>7.2f}x {'  '+status:>10}")

    section("3. EV DEPARTURE VIOLATIONS (PER-DEPARTURE METRIC)")
    print(f"  Total departures:         {total_departures}")
    print(f"  Violated departures:      {violated_departures}")
    print(f"  Violation rate:           {ev_viol_pct:.2f}%  ({violated_departures}/{total_departures})")
    print(f"  Mean deficit (all dep):   {mean_deficit:.4f} SoC")
    if violated_deficits:
        print(f"  Mean deficit (violated):  {np.mean(violated_deficits):.4f} SoC")
        print(f"  Median deficit (violated):{np.median(violated_deficits):.4f} SoC")
        print(f"  Max deficit:              {max(violated_deficits):.4f} SoC")
        print(f"  P90 deficit:              {np.percentile(violated_deficits, 90):.4f} SoC")
        print(f"  P99 deficit:              {np.percentile(violated_deficits, 99):.4f} SoC")
    # Deficit histogram
    if departure_deficits:
        bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        hist, _ = np.histogram(departure_deficits, bins=bins)
        print(f"\n  Deficit distribution:")
        for i in range(len(bins)-1):
            bar = '#' * int(hist[i] * 40 / max(max(hist), 1))
            print(f"    [{bins[i]:.2f}-{bins[i+1]:.2f}): {hist[i]:>4d} {bar}")

    section("4. BATTERY SoC ANALYSIS (C2)")
    c2_viol_pct = pct(soc_violation_steps, step)
    print(f"  Safety Projection active:  {batt_clamp_count}/{step} steps clamped ({pct(batt_clamp_count, step):.1f}%)")
    print(f"  Steps with SoC violation:  {soc_violation_steps}/{step} ({c2_viol_pct:.2f}%)")
    print(f"  Cumulative C2 cost:        {total_c2:.1f} (limit: {COST_LIMITS['C2_battery_soc']})")
    print(f"\n  Per-building SoC range:")
    print(f"  {'Building':<12} {'Min SoC':>10} {'Max SoC':>10} {'Mean SoC':>10}")
    print(f"  {'-'*42}")
    for b in range(n_buildings):
        col = soc_arr[:, b] if b < soc_arr.shape[1] else np.array([0.5])
        print(f"  {'Bld '+str(b):<12} {soc_min_per_building[b]:>10.4f} {soc_max_per_building[b]:>10.4f} {np.mean(col):>10.4f}")

    section("5. BUILDING POWER ANALYSIS (C3)")
    c3_viol_pct = pct(c3_violation_steps, step)
    print(f"  Steps with ANY building violation: {c3_violation_steps}/{step} ({c3_viol_pct:.2f}%)")
    print(f"  Cumulative C3 cost:  {total_c3:.1f} (limit: {COST_LIMITS['C3_building_power']})")
    print(f"  P_building_max:      {P_BUILDING_MAX} kW")
    print(f"\n  Per-building breakdown:")
    print(f"  {'Building':<12} {'Violations':>12} {'Viol%':>8} {'MaxPwr(kW)':>12} {'MeanPwr':>10} {'P95 Pwr':>10}")
    print(f"  {'-'*64}")
    for b in range(n_buildings):
        pwr = np.array(building_power_all[b]) if building_power_all[b] else np.array([0.0])
        v = building_power_violations[b]
        v_pct = pct(v, len(pwr))
        p95 = np.percentile(pwr, 95) if len(pwr) > 0 else 0
        print(f"  {'Bld '+str(b):<12} {v:>12d} {v_pct:>7.2f}% {building_power_max[b]:>12.3f} {np.mean(pwr):>10.3f} {p95:>10.3f}")

    section("6. GRID POWER ANALYSIS (C4)")
    c4_viol_pct = pct(grid_violation_steps, step)
    gp = np.array(grid_power_all) if grid_power_all else np.array([0.0])
    print(f"  Steps with grid violation:  {grid_violation_steps}/{step} ({c4_viol_pct:.2f}%)")
    print(f"  Cumulative C4 cost:         {total_c4:.1f} (limit: {COST_LIMITS['C4_grid_power']})")
    print(f"  P_grid_max:                 {P_GRID_MAX} kW")
    print(f"  Actual max grid import:     {grid_power_max:.3f} kW")
    print(f"  Mean grid import:           {np.mean(gp):.3f} kW")
    print(f"  P95 grid import:            {np.percentile(gp, 95):.3f} kW")
    print(f"  P99 grid import:            {np.percentile(gp, 99):.3f} kW")

    section("7. HOURLY VIOLATION HEATMAP")
    print(f"  {'Hour':<6} {'C3 Viols':>10} {'C3%':>8} {'C4 Viols':>10} {'C4%':>8} {'Visual C3':>20}")
    print(f"  {'-'*62}")
    for h in range(24):
        c3h = hourly_c3_violations[h]
        c4h = hourly_c4_violations[h]
        cnt = hourly_counts[h]
        c3p = pct(c3h, cnt)
        c4p = pct(c4h, cnt)
        bar = '#' * int(c3p / 2)
        print(f"  {h:>4}h {c3h:>10} {c3p:>7.1f}% {c4h:>10} {c4p:>7.1f}%  {bar}")

    section("8. V2G BEHAVIOR ANALYSIS")
    print(f"  Total EV action steps:  {total_ev_steps}")
    print(f"  Charging (a > 0.05):    {ev_charge_steps} ({pct(ev_charge_steps, total_ev_steps):.1f}%)")
    print(f"  Discharging (a < -0.05):{ev_discharge_steps} ({pct(ev_discharge_steps, total_ev_steps):.1f}%)")
    print(f"  Idle (|a| <= 0.05):     {ev_idle_steps} ({pct(ev_idle_steps, total_ev_steps):.1f}%)")
    if len(ev_actions) > 0:
        print(f"\n  EV action distribution:")
        print(f"    Mean:   {ev_actions.mean():.4f}")
        print(f"    Std:    {ev_actions.std():.4f}")
        print(f"    Min:    {ev_actions.min():.4f}")
        print(f"    Max:    {ev_actions.max():.4f}")
        print(f"    P10:    {np.percentile(ev_actions, 10):.4f}")
        print(f"    P90:    {np.percentile(ev_actions, 90):.4f}")
        # Per-charger breakdown
        print(f"\n  Per-EV charger action stats:")
        for i, ei in enumerate(ev_idx):
            col = actions_arr[:, ei]
            n_charge = np.sum(col > 0.05)
            n_discharge = np.sum(col < -0.05)
            print(f"    EV[{i}] (act[{ei}]): mean={col.mean():.3f} std={col.std():.3f} "
                  f"charge={n_charge}({pct(n_charge,len(col)):.0f}%) "
                  f"discharge={n_discharge}({pct(n_discharge,len(col)):.0f}%)")

    section("9. ACTION STATISTICS")
    if len(batt_actions) > 0:
        print(f"  Battery actions:")
        print(f"    Mean: {batt_actions.mean():.4f}, Std: {batt_actions.std():.4f}")
        print(f"    Min:  {batt_actions.min():.4f}, Max: {batt_actions.max():.4f}")
        for i, bi in enumerate(batt_idx):
            col = actions_arr[:, bi]
            print(f"    Batt[{i}] (act[{bi}]): mean={col.mean():.3f} std={col.std():.3f} "
                  f"[{col.min():.3f}, {col.max():.3f}]")
    print(f"\n  Full action vector stats:")
    print(f"    {'Action':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"    {'-'*40}")
    for i in range(act_dim):
        col = actions_arr[:, i]
        print(f"    [{i:>2}]    {col.mean():>8.3f} {col.std():>8.3f} {col.min():>8.3f} {col.max():>8.3f}  {names[i] if i < len(names) else ''}")

    section("10. REWARD DECOMPOSITION (STEMS)")
    print(f"  Total episode reward:    {total_reward:.2f}")
    print(f"  Mean step reward:        {total_reward/step:.4f}")
    print(f"\n  {'Component':<30} {'Total':>12} {'Mean/Step':>12} {'% of Total':>12}")
    print(f"  {'-'*66}")
    for key in ["reward_stems_total", "reward_economic", "reward_stability",
                 "reward_stability_grid", "reward_stability_building",
                 "reward_stability_ramp", "reward_renewable", "reward_comfort",
                 "reward_ev_shaping", "reward_bill"]:
        vals = reward_components[key]
        total_comp = sum(vals)
        mean_comp = total_comp / step if step > 0 else 0
        pct_of_total = 100.0 * total_comp / abs(total_reward) if abs(total_reward) > 0 else 0
        print(f"  {key:<30} {total_comp:>12.2f} {mean_comp:>12.4f} {pct_of_total:>11.1f}%")

    section("11. ENERGY BALANCE")
    total_import = sum(grid_imports)
    total_export = sum(grid_exports)
    total_monetary = sum(step_costs_monetary)
    total_carbon_kg = sum(step_carbon)
    print(f"  Total grid import:       {total_import:.2f} kWh")
    print(f"  Total grid export:       {total_export:.2f} kWh")
    print(f"  Net consumption:         {total_import - total_export:.2f} kWh")
    print(f"  Self-consumption ratio:  {pct(total_import - total_export, total_import):.1f}%")
    print(f"  Total electricity cost:  ${total_monetary:.2f}")
    print(f"  Total carbon emissions:  {total_carbon_kg:.2f} kg CO2")
    print(f"  Mean electricity price:  ${np.mean(electricity_prices):.4f}/kWh")

    section("12. CITYLEARN KPIs (OFFICIAL)")
    if citylearn_kpis:
        for k, v in sorted(citylearn_kpis.items()):
            if np.isnan(v):
                print(f"  {k:<50} N/A")
            else:
                print(f"  {k:<50} {v:.6f}")
    else:
        print(f"  (No CityLearn KPIs available — episode may not have terminated normally)")

    section("13. OVERALL VERDICT")
    n_violated = sum(1 for _, c, l in constraints if c > l)
    n_ok = len(constraints) - n_violated
    print(f"  Constraints satisfied:  {n_ok}/{len(constraints)}")
    print(f"  Constraints violated:   {n_violated}/{len(constraints)}")
    for name, cost_val, limit in constraints:
        status = "PASS" if cost_val <= limit else f"FAIL ({cost_val/limit:.1f}x)"
        print(f"    {name:<25} {status}")
    print(f"\n  EV departure violation rate: {ev_viol_pct:.2f}%")
    print(f"  Building power violation rate: {c3_viol_pct:.2f}% of steps")
    print(f"  Grid power violation rate:     {c4_viol_pct:.2f}% of steps")
    print(f"  Total reward:                  {total_reward:.2f}")
    print(f"{'='*W}")

    # =====================================================================
    # SAVE RESULTS TO JSON
    # =====================================================================
    results = {
        "checkpoint": ckpt_path,
        "seed": seed,
        "steps": step,
        "total_reward": total_reward,
        "total_cost": total_cost,
        "per_constraint": {
            "C0_ev_departure": {"cost": total_c0, "limit": COST_LIMITS["C0_ev_departure"]},
            "C1_ev_saute": {"cost": total_c1, "limit": COST_LIMITS["C1_ev_saute"]},
            "C2_battery_soc": {"cost": total_c2, "limit": COST_LIMITS["C2_battery_soc"]},
            "C3_building_power": {"cost": total_c3, "limit": COST_LIMITS["C3_building_power"]},
            "C4_grid_power": {"cost": total_c4, "limit": COST_LIMITS["C4_grid_power"]},
        },
        "ev_departures": {
            "total": total_departures,
            "violated": violated_departures,
            "violation_pct": ev_viol_pct,
            "mean_deficit": mean_deficit,
        },
        "violation_rates_pct": {
            "C2_steps": c2_viol_pct,
            "C3_steps": c3_viol_pct,
            "C4_steps": c4_viol_pct,
        },
        "v2g_behavior": {
            "charge_pct": pct(ev_charge_steps, total_ev_steps),
            "discharge_pct": pct(ev_discharge_steps, total_ev_steps),
            "idle_pct": pct(ev_idle_steps, total_ev_steps),
        },
        "citylearn_kpis": {k: v for k, v in citylearn_kpis.items() if not np.isnan(v)},
        "energy_balance": {
            "total_import_kwh": total_import,
            "total_export_kwh": total_export,
            "net_consumption_kwh": total_import - total_export,
            "total_cost_usd": total_monetary,
            "total_carbon_kg": total_carbon_kg,
        },
    }

    out_path = os.path.join(PROJECT, "runs", "r22_ppo_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=None,
                    help="Specific epoch to evaluate (default: latest = epoch-89)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ckpt = find_checkpoint(epoch=args.epoch)
    if not ckpt:
        print("ERROR: No checkpoint found!")
        sys.exit(1)

    print(f"Using checkpoint: {ckpt}")
    evaluate(ckpt, seed=args.seed)
