#!/usr/bin/env python3
"""
Adversarial baseline evaluation: compare 4 policies side-by-side on the
5-building CityLearn schema.

Policies:
  1. zero_action:         all actions = 0 (no-control baseline)
  2. greedy_ev:           EVs = 1.0, batteries = 0.0
  3. adversarial_greedy:  EVs = 1.0, batteries = peak-charge / off-peak-discharge
  4. SmartV2GRBC:         intelligent price/solar-aware RBC

Reports per policy:
  - C0%, C2%, C3%, C4% violation rates
  - Total import/export kWh, peak grid power
  - Total reward
  - Hourly grid import profile
"""

import os
import sys
import time
import importlib
import numpy as np
from collections import defaultdict

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# =====================================================================
# Environment configuration (matches eval_r21_r23_r30c.py BASE_ENV)
# =====================================================================
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
    # Disable training-only wrappers
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_PID_LAGRANGE": "0",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_ACTION_MASK": "0",
    "CITYLEARN_BATT_CLAMP": "0",  # No battery clamp for baselines
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
    # Reward weights (needed for valid env, use R21 defaults)
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
}

P_BUILDING_MAX = 4.6083
P_GRID_MAX = 10.2352
SOC_LOW = 0.0
SOC_HIGH = 0.95
C0_TOLERANCE = 0.01  # SoC deficit > 0.01 counts as violation

SEED = 42


def set_env():
    """Clear all CITYLEARN/STEMS/COST env vars and set BASE_ENV."""
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(k, None)
    for k, v in BASE_ENV.items():
        os.environ[k] = v


def make_env():
    """Create evaluation environment. Must call set_env() first."""
    import citylearn_safe.schema_index as si
    si._CACHE = None
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


def get_action_indices(raw):
    """Detect battery and EV action indices from action_names."""
    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)
    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]
    wash_idx = [i for i, n in enumerate(names) if "washing_machine" in str(n).lower()]
    return names, batt_idx, ev_idx, wash_idx


def run_episode(env, raw, action_fn, seed=42):
    """
    Run one full episode (8759 steps) and collect all metrics.

    action_fn(obs, env, raw) -> np.ndarray of actions

    Returns dict with violation rates, energy stats, rewards, hourly profiles.
    """
    buildings = list(getattr(raw, "buildings", []))
    n_buildings = len(buildings)
    names, batt_idx, ev_idx, wash_idx = get_action_indices(raw)
    act_dim = len(names)

    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    # Trackers
    rewards = []
    actions_all = []

    # C0: EV departure tracking
    total_departures = 0
    violated_departures = 0
    departure_deficits = []
    ev_tracker = {}

    # C2: Battery SoC tracking
    c2_violation_steps = 0
    soc_values = []  # per step, list of per-building SoC

    # C3: Building power
    c3_violation_steps = 0
    building_power_violations = [0] * n_buildings

    # C4: Grid power
    c4_violation_steps = 0

    # Energy tracking
    hourly_grid_import = [0.0] * 24
    hourly_grid_export = [0.0] * 24
    hourly_step_counts = [0] * 24
    total_import = 0.0
    total_export = 0.0
    peak_grid_power = 0.0

    # CityLearn KPIs
    citylearn_kpis = {}

    while not done:
        action = action_fn(obs, env, raw)
        action = np.clip(action, -1.0, 1.0)
        actions_all.append(action.copy())

        t_now = int(getattr(raw, "time_step", 0))
        t_idx = max(0, t_now - 1)
        hour = t_now % 24

        # === C0: EV departure tracking ===
        for b_idx, bld in enumerate(buildings):
            chargers = getattr(bld, "electric_vehicle_chargers", None) or []
            for ch_idx, ch in enumerate(chargers):
                key = (b_idx, ch_idx)
                sim = getattr(ch, "charger_simulation",
                              getattr(ch, "_Charger__charger_simulation", None))
                if sim is None:
                    continue
                try:
                    sa = np.asarray(
                        getattr(sim, "_electric_vehicle_charger_state"), dtype=float
                    )
                    current_connected = t_now < len(sa) and float(sa[t_now]) == 1.0
                    if current_connected:
                        ra = np.asarray(
                            getattr(sim, "_electric_vehicle_required_soc_departure"),
                            dtype=float,
                        )
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
                                    if 0 <= t_idx < len(sn):
                                        current_soc = float(np.clip(sn[t_idx], 0, 1))
                        ev_tracker[key] = {
                            "was_connected": True,
                            "last_soc": current_soc,
                            "required_soc": rs,
                        }
                    else:
                        prev = ev_tracker.get(key, {})
                        if prev.get("was_connected", False):
                            total_departures += 1
                            last_soc = prev["last_soc"]
                            rs = prev["required_soc"]
                            deficit = max(0.0, rs - last_soc)
                            if deficit > C0_TOLERANCE:
                                violated_departures += 1
                            departure_deficits.append(deficit)
                        ev_tracker[key] = {"was_connected": False}
                except Exception:
                    pass

        # === Step environment ===
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        rewards.append(reward)

        # === C2: Battery SoC violations ===
        soc_viol = info.get("battery_soc_violation_any", 0.0)
        if soc_viol > 0:
            c2_violation_steps += 1

        # Track SoC per building
        step_socs = []
        for b_idx, bld in enumerate(buildings):
            try:
                es = getattr(bld, "electrical_storage", None)
                if es is not None:
                    soc_arr = getattr(es, "soc", None)
                    if soc_arr is not None and hasattr(soc_arr, "__len__") and len(soc_arr) > t_idx:
                        s = float(np.clip(soc_arr[t_idx], 0, 1))
                        step_socs.append(s)
                    else:
                        step_socs.append(0.5)
                else:
                    step_socs.append(0.5)
            except Exception:
                step_socs.append(0.5)
        soc_values.append(step_socs)

        # === C3: Building power violations ===
        bld_viol = info.get("building_power_violation", 0.0)
        if bld_viol > 0:
            c3_violation_steps += 1
        for b_idx, bld in enumerate(buildings):
            try:
                nec = getattr(bld, "net_electricity_consumption", None)
                if nec is not None and len(nec) > t_idx:
                    p = abs(float(nec[t_idx]))
                    if p > P_BUILDING_MAX:
                        building_power_violations[b_idx] += 1
            except Exception:
                pass

        # === C4: Grid power violations ===
        grid_viol = info.get("grid_power_violation", 0.0)
        if grid_viol > 0:
            c4_violation_steps += 1

        # === Energy tracking ===
        gi = info.get("grid_import_kwh", 0.0)
        ge = info.get("grid_export_kwh", 0.0)
        total_import += gi
        total_export += ge
        peak_grid_power = max(peak_grid_power, gi)
        hourly_grid_import[hour] += gi
        hourly_grid_export[hour] += ge
        hourly_step_counts[hour] += 1

        # === CityLearn KPIs on last step ===
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
                "citylearn_1_minus_load_factor",
            ]:
                citylearn_kpis[kpi_key] = info.get(kpi_key, float("nan"))

        if step % 2000 == 0:
            sys.stdout.write(f"  step {step}...\r")
            sys.stdout.flush()

    # Compute SoC stats
    soc_arr = np.array(soc_values)  # (steps, n_buildings)
    soc_min_per_bld = soc_arr.min(axis=0) if soc_arr.size > 0 else np.zeros(n_buildings)
    soc_max_per_bld = soc_arr.max(axis=0) if soc_arr.size > 0 else np.zeros(n_buildings)
    soc_mean_per_bld = soc_arr.mean(axis=0) if soc_arr.size > 0 else np.zeros(n_buildings)

    # Compute hourly averages
    hourly_import_avg = [
        hourly_grid_import[h] / max(1, hourly_step_counts[h]) for h in range(24)
    ]
    hourly_export_avg = [
        hourly_grid_export[h] / max(1, hourly_step_counts[h]) for h in range(24)
    ]

    c0_rate = 100.0 * violated_departures / total_departures if total_departures > 0 else 0.0
    c2_rate = 100.0 * c2_violation_steps / step
    c3_rate = 100.0 * c3_violation_steps / step
    c4_rate = 100.0 * c4_violation_steps / step

    actions_arr = np.array(actions_all)

    return {
        "steps": step,
        "total_reward": sum(rewards),
        "mean_reward": sum(rewards) / step if step > 0 else 0,
        # Violations
        "c0_departures": total_departures,
        "c0_violated": violated_departures,
        "c0_rate": c0_rate,
        "c0_mean_deficit": float(np.mean(departure_deficits)) if departure_deficits else 0.0,
        "c2_violation_steps": c2_violation_steps,
        "c2_rate": c2_rate,
        "c3_violation_steps": c3_violation_steps,
        "c3_rate": c3_rate,
        "c4_violation_steps": c4_violation_steps,
        "c4_rate": c4_rate,
        "building_power_violations": building_power_violations,
        # Energy
        "total_import_kwh": total_import,
        "total_export_kwh": total_export,
        "peak_grid_power_kwh": peak_grid_power,
        "hourly_import_avg": hourly_import_avg,
        "hourly_export_avg": hourly_export_avg,
        # SoC
        "soc_min_per_bld": soc_min_per_bld.tolist(),
        "soc_max_per_bld": soc_max_per_bld.tolist(),
        "soc_mean_per_bld": soc_mean_per_bld.tolist(),
        # CityLearn KPIs
        "citylearn_kpis": citylearn_kpis,
        # Action stats
        "action_means": actions_arr.mean(axis=0).tolist(),
        "action_stds": actions_arr.std(axis=0).tolist(),
    }


def print_comparison(all_results, names_list, act_names, batt_idx, ev_idx):
    """Print side-by-side comparison table."""
    W = 22  # column width

    print(f"\n{'='*120}")
    print(f"  ADVERSARIAL BASELINE COMPARISON (5-building schema, seed={SEED})")
    print(f"{'='*120}")

    # --- Constraint violations ---
    print(f"\n--- Constraint Violation Rates ---")
    header = f"  {'Metric':<30}"
    for n in names_list:
        header += f" {n:>{W}}"
    print(header)
    print(f"  {'-'*(30 + (W+1)*len(names_list))}")

    metrics = [
        ("C0: EV Violation Rate (%)", "c0_rate"),
        ("C0: Departures (total)", "c0_departures"),
        ("C0: Violated Departures", "c0_violated"),
        ("C0: Mean SoC Deficit", "c0_mean_deficit"),
        ("C2: Battery SoC Viol (%)", "c2_rate"),
        ("C3: Building Power Viol (%)", "c3_rate"),
        ("C4: Grid Power Viol (%)", "c4_rate"),
    ]
    for label, key in metrics:
        row = f"  {label:<30}"
        for n in names_list:
            v = all_results[n].get(key, 0)
            if isinstance(v, float):
                row += f" {v:>{W}.4f}"
            else:
                row += f" {v:>{W}}"
        print(row)

    # --- Energy ---
    print(f"\n--- Energy Statistics ---")
    header = f"  {'Metric':<30}"
    for n in names_list:
        header += f" {n:>{W}}"
    print(header)
    print(f"  {'-'*(30 + (W+1)*len(names_list))}")

    energy_metrics = [
        ("Total Import (kWh)", "total_import_kwh"),
        ("Total Export (kWh)", "total_export_kwh"),
        ("Peak Grid Power (kWh)", "peak_grid_power_kwh"),
        ("Total Reward", "total_reward"),
        ("Mean Step Reward", "mean_reward"),
    ]
    for label, key in energy_metrics:
        row = f"  {label:<30}"
        for n in names_list:
            v = all_results[n].get(key, 0)
            row += f" {v:>{W}.2f}"
        print(row)

    # --- CityLearn KPIs ---
    print(f"\n--- CityLearn KPIs (lower = better for most) ---")
    kpi_keys = [
        "citylearn_electricity_consumption_total",
        "citylearn_cost_total",
        "citylearn_daily_peak_average",
        "citylearn_all_time_peak_average",
        "citylearn_ramping_average",
        "citylearn_zero_net_energy",
    ]
    kpi_short = {
        "citylearn_electricity_consumption_total": "Electricity",
        "citylearn_cost_total": "Cost",
        "citylearn_daily_peak_average": "DailyPeak",
        "citylearn_all_time_peak_average": "AllTimePeak",
        "citylearn_ramping_average": "Ramping",
        "citylearn_zero_net_energy": "ZeroNetEnergy",
    }
    header = f"  {'KPI':<30}"
    for n in names_list:
        header += f" {n:>{W}}"
    print(header)
    print(f"  {'-'*(30 + (W+1)*len(names_list))}")
    for kk in kpi_keys:
        short = kpi_short.get(kk, kk)
        row = f"  {short:<30}"
        for n in names_list:
            v = all_results[n].get("citylearn_kpis", {}).get(kk, float("nan"))
            if np.isnan(v):
                row += f" {'N/A':>{W}}"
            else:
                row += f" {v:>{W}.4f}"
        print(row)

    # --- Battery SoC Stats ---
    n_bld = len(all_results[names_list[0]].get("soc_min_per_bld", []))
    print(f"\n--- Battery SoC Statistics (per building) ---")
    for b in range(n_bld):
        print(f"\n  Building {b}:")
        header = f"    {'Metric':<20}"
        for n in names_list:
            header += f" {n:>{W}}"
        print(header)
        print(f"    {'-'*(20 + (W+1)*len(names_list))}")
        for metric, key in [("Min SoC", "soc_min_per_bld"),
                            ("Max SoC", "soc_max_per_bld"),
                            ("Mean SoC", "soc_mean_per_bld")]:
            row = f"    {metric:<20}"
            for n in names_list:
                vals = all_results[n].get(key, [])
                v = vals[b] if b < len(vals) else 0.0
                row += f" {v:>{W}.4f}"
            print(row)

    # --- Per-building C3 violations ---
    print(f"\n--- Per-Building C3 Violations (steps > P_building_max={P_BUILDING_MAX}) ---")
    header = f"  {'Building':<12}"
    for n in names_list:
        header += f" {n:>{W}}"
    print(header)
    print(f"  {'-'*(12 + (W+1)*len(names_list))}")
    for b in range(n_bld):
        row = f"  {'Bld '+str(b):<12}"
        for n in names_list:
            viols = all_results[n].get("building_power_violations", [])
            v = viols[b] if b < len(viols) else 0
            row += f" {v:>{W}}"
        print(row)

    # --- Hourly grid import profile ---
    print(f"\n--- Hourly Average Grid Import (kWh) ---")
    header = f"  {'Hour':<6}"
    for n in names_list:
        header += f" {n:>{W}}"
    print(header)
    print(f"  {'-'*(6 + (W+1)*len(names_list))}")
    for h in range(24):
        row = f"  {h:>4}h"
        for n in names_list:
            avg_imports = all_results[n].get("hourly_import_avg", [])
            v = avg_imports[h] if h < len(avg_imports) else 0.0
            row += f" {v:>{W}.3f}"
        print(row)

    # --- Highlight peak hours ---
    print(f"\n--- Peak Hour Analysis (hours with highest avg import) ---")
    for n in names_list:
        avg_imports = all_results[n].get("hourly_import_avg", [0.0] * 24)
        top5 = sorted(range(24), key=lambda h: avg_imports[h], reverse=True)[:5]
        top_str = ", ".join(f"{h}h ({avg_imports[h]:.2f})" for h in top5)
        print(f"  {n:<20}: {top_str}")

    print(f"\n{'='*120}")


def main():
    all_results = {}

    # ---------------------------------------------------------------
    # 1. Zero Action (no-control baseline)
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Evaluating: zero_action (all zeros)")
    print(f"{'='*70}")
    set_env()
    env, raw = make_env()
    names, batt_idx, ev_idx, wash_idx = get_action_indices(raw)
    act_dim = len(names)
    print(f"  act_dim={act_dim} batt={batt_idx} ev={ev_idx} wash={wash_idx}")

    def zero_fn(obs, env_inner, raw_inner):
        return np.zeros(act_dim, dtype=np.float32)

    t0 = time.time()
    all_results["zero_action"] = run_episode(env, raw, zero_fn, seed=SEED)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---------------------------------------------------------------
    # 2. Greedy EV (EVs=1.0, batteries=0.0)
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Evaluating: greedy_ev (EVs=1.0, rest=0.0)")
    print(f"{'='*70}")
    set_env()
    env, raw = make_env()

    def greedy_ev_fn(obs, env_inner, raw_inner):
        a = np.zeros(act_dim, dtype=np.float32)
        for i in ev_idx:
            a[i] = 1.0
        return a

    t0 = time.time()
    all_results["greedy_ev"] = run_episode(env, raw, greedy_ev_fn, seed=SEED)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---------------------------------------------------------------
    # 3. Adversarial modes: try multiple battery strategies
    # ---------------------------------------------------------------
    from scripts.adversarial_baseline import AdversarialGreedyPolicy

    adv_modes = ["peak_charge", "always_charge", "oscillate", "contraflow"]
    for mode in adv_modes:
        label = f"adv_{mode}"
        print(f"\n{'='*70}")
        print(f"  Evaluating: adversarial ({mode})")
        print(f"{'='*70}")
        set_env()
        env, raw = make_env()

        adv_policy = AdversarialGreedyPolicy(env, battery_mode=mode)

        def make_adv_fn(policy):
            def fn(obs, env_inner, raw_inner):
                return policy.predict(obs)
            return fn

        t0 = time.time()
        all_results[label] = run_episode(env, raw, make_adv_fn(adv_policy), seed=SEED)
        elapsed = time.time() - t0
        r = all_results[label]
        print(f"  Done in {elapsed:.1f}s | C0={r['c0_rate']:.1f}% C2={r['c2_rate']:.1f}% "
              f"C3={r['c3_rate']:.1f}% C4={r['c4_rate']:.1f}%")

    # ---------------------------------------------------------------
    # 4. SmartV2GRBC
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Evaluating: SmartV2GRBC")
    print(f"{'='*70}")
    set_env()
    env, raw = make_env()

    from scripts.rbc_policy import SmartV2GRBC
    smart_policy = SmartV2GRBC(env)

    def smart_fn(obs, env_inner, raw_inner):
        return smart_policy.predict(obs)

    t0 = time.time()
    all_results["SmartV2GRBC"] = run_episode(env, raw, smart_fn, seed=SEED)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---------------------------------------------------------------
    # Print comparison
    # ---------------------------------------------------------------
    policy_names = list(all_results.keys())
    print_comparison(all_results, policy_names, names, batt_idx, ev_idx)

    # ---------------------------------------------------------------
    # Check each adversarial mode against targets
    # ---------------------------------------------------------------
    print(f"\n--- Adversarial Target Check (all modes) ---")
    print(f"  {'Mode':<20} {'C0%':>8} {'C2%':>8} {'C3%':>8} {'C4%':>8} {'Verdict':<20}")
    print(f"  {'-'*74}")
    best_mode = None
    best_score = float("-inf")
    for mode in adv_modes:
        label = f"adv_{mode}"
        if label not in all_results:
            continue
        r = all_results[label]
        c0 = r["c0_rate"]
        c2 = r["c2_rate"]
        c3 = r["c3_rate"]
        c4 = r["c4_rate"]
        c2_ok = 25 <= c2 <= 50
        c3_ok = 25 <= c3 <= 50
        c4_ok = 25 <= c4 <= 50
        n_ok = sum([c2_ok, c3_ok, c4_ok])
        verdict = f"{n_ok}/3 in range" + (" (C0 OK)" if c0 < 5 else " (C0 BAD!)")
        print(f"  {mode:<20} {c0:>7.2f}% {c2:>7.2f}% {c3:>7.2f}% {c4:>7.2f}% {verdict}")
        # Score: closer to 35% target for each
        score = -abs(c2 - 35) - abs(c3 - 35) - abs(c4 - 35) - (100 if c0 > 5 else 0)
        if score > best_score:
            best_score = score
            best_mode = mode

    if best_mode:
        print(f"\n  Best mode: {best_mode}")
        r = all_results[f"adv_{best_mode}"]
        print(f"    C0={r['c0_rate']:.2f}% C2={r['c2_rate']:.2f}% "
              f"C3={r['c3_rate']:.2f}% C4={r['c4_rate']:.2f}%")

    return all_results


if __name__ == "__main__":
    results = main()
