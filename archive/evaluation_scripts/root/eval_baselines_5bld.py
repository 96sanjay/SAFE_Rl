#!/usr/bin/env python3
"""
Comprehensive evaluation of baselines on the 5-building schema.

Policies evaluated:
  1. zero_action: all actions = 0.0
  2. greedy_ev: EV actions = 1.0 (full charge), battery actions = 0.0
  3. SmartV2GRBC: price/solar/grid-aware RBC with priority-based V2G

Metrics reported per policy:
  - C0 (EV departure SoC), C2 (battery SoC bounds), C3 (building power), C4 (grid power)
  - Total import/export kWh, peak grid power
  - Per-building import/export breakdown
  - EV behavior: charge%, V2G%, total V2G energy
  - Battery behavior: charge%, discharge%, idle%
  - Per-departure C0 details for any violations
  - Hourly average grid import profile (24 values)
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, PROJECT)

SCHEMA_PATH = os.path.join(
    PROJECT,
    "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
)
TOTAL_STEPS = 8759
SEED = 42

# Power thresholds (from run_r19_ablation.sh)
DEFAULT_P_BUILDING_MAX = 4.6083
DEFAULT_P_GRID_MAX = 10.2352

# C0 tolerance: ignore violations where deficit < this (structural noise)
C0_SOC_TOLERANCE = 0.001


# ---------------------------------------------------------------------------
# Helpers (adapted from eval_all_proper.py)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def suppress_stdout():
    """Temporarily suppress stdout (for noisy env debug prints)."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def log(msg: str = ""):
    """Print and flush immediately."""
    print(msg, flush=True)


def parse_exports(script_path: str) -> dict:
    """Extract all 'export KEY=VALUE' from a bash script."""
    exports = {}
    with open(script_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("export "):
                continue
            rest = line[len("export "):]
            eq_idx = rest.find("=")
            if eq_idx < 0:
                continue
            key = rest[:eq_idx].strip()
            val_raw = rest[eq_idx + 1:].strip()
            if val_raw.startswith('"'):
                end_q = val_raw.find('"', 1)
                if end_q > 0:
                    val_raw = val_raw[1:end_q]
            else:
                for sep in ["  #", " #", "\t#"]:
                    ci = val_raw.find(sep)
                    if ci >= 0:
                        val_raw = val_raw[:ci]
                val_raw = val_raw.strip().strip('"').strip("'")
            val_raw = val_raw.replace("$PROJECT", PROJECT)
            val_raw = val_raw.replace("${PYTHONPATH:-}", "")
            exports[key] = val_raw
    return exports


def get_citylearn_env(wrapper):
    """Walk wrapper chain to find the CityLearnEnv."""
    cur = wrapper
    seen = set()
    for _ in range(40):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "buildings") and hasattr(cur, "time_step"):
            blds = getattr(cur, "buildings", None)
            if blds is not None and len(blds) > 0:
                return cur
        for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return None


def get_action_names(env) -> list:
    """Walk wrapper chain to find action_names."""
    _e = env
    for _ in range(20):
        act_names_raw = getattr(_e, "action_names", None)
        if act_names_raw is not None:
            break
        _e = getattr(_e, "env", getattr(_e, "base", None))
        if _e is None:
            break
    if act_names_raw is None:
        return []
    if isinstance(act_names_raw, list) and len(act_names_raw) > 0 and isinstance(act_names_raw[0], list):
        return [n for sub in act_names_raw for n in sub]
    return list(act_names_raw)


def build_env(env_vars: dict):
    """Build the env wrapper chain with the given env vars."""
    for k, v in env_vars.items():
        os.environ[k] = v

    os.environ["CITYLEARN_SCHEMA"] = SCHEMA_PATH
    os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
    os.environ["CITYLEARN_REWARD_TYPE"] = "stems"

    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.saute_ev_wrapper import SauteEVBudgetWrapper

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    forecast = ForecastObsWrapper(safety, forecast_horizon=24)

    use_saute = env_vars.get("CITYLEARN_EV_SAUTE", "0") == "1"
    if use_saute:
        env = SauteEVBudgetWrapper(forecast)
    else:
        env = forecast

    return env


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------
def evaluate_policy(
    policy_name: str,
    env,
    get_action_fn,
    env_vars: dict,
) -> dict:
    """Run full-year eval and compute all metrics. Returns a results dict."""

    city = get_citylearn_env(env)
    if city is None:
        log(f"  ERROR: Could not find CityLearnEnv for {policy_name}")
        return {"error": "No CityLearnEnv found"}

    n_buildings = len(city.buildings)
    act_names = get_action_names(env)
    batt_indices = [i for i, n in enumerate(act_names) if str(n).strip().lower() == "electrical_storage"]
    ev_indices = [i for i, n in enumerate(act_names) if "electric_vehicle" in str(n).lower()]

    log(f"  Action names: {act_names}")
    log(f"  Battery indices: {batt_indices}")
    log(f"  EV indices: {ev_indices}")

    P_building_max = float(env_vars.get("CITYLEARN_STEMS_P_BUILDING_MAX", str(DEFAULT_P_BUILDING_MAX)))
    P_grid_max = float(env_vars.get("CITYLEARN_STEMS_P_GRID_MAX", str(DEFAULT_P_GRID_MAX)))
    c3_controllable = env_vars.get("CITYLEARN_C3_CONTROLLABLE", "0") == "1"

    # Discover chargers
    charger_info = []
    for bi, b in enumerate(city.buildings):
        for ch in getattr(b, "electric_vehicle_chargers", []):
            cid = getattr(ch, "charger_id", f"charger_{bi}")
            charger_info.append((bi, ch, cid))
    log(f"  Chargers: {len(charger_info)}")

    # Reset env
    np.random.seed(SEED)
    with suppress_stdout():
        obs, info = env.reset(seed=SEED)

    # Tracking arrays
    departures = []  # (charger_id, actual_soc, required_soc, time_step)
    hourly_grid_import = defaultdict(list)  # hour -> [import_kwh values]
    per_building_import = defaultdict(float)  # building_idx -> total import
    per_building_export = defaultdict(float)  # building_idx -> total export
    step_rewards = []
    step_infos_stems = defaultdict(list)  # stems component -> [values]

    nec_prev = 0.0
    ramping_total = 0.0
    total_import = 0.0
    total_export = 0.0
    electricity_cost_total = 0.0
    peak_nec = 0.0

    c2_violations = 0
    c2_total = 0
    c3_violations = 0
    c3_total = 0
    c4_violations = 0
    c4_total = 0

    batt_charge_steps = 0
    batt_discharge_steps = 0
    batt_total_steps = 0
    batt_daily_charge = defaultdict(bool)
    batt_daily_discharge = defaultdict(bool)

    ev_charge_steps = 0
    ev_v2g_steps = 0
    ev_connected_steps = 0
    ev_total_v2g_kwh = 0.0

    t0 = time.time()

    for step_i in range(TOTAL_STEPS):
        action = get_action_fn(obs)
        with suppress_stdout():
            obs, reward, terminated, truncated, info = env.step(action)

        step_rewards.append(float(reward))

        t_after = int(getattr(city, "time_step", 0))
        hour = t_after % 24
        day = t_after // 24
        is_solar = 10 <= hour <= 15
        is_peak = 17 <= hour <= 21

        # --- Grid-level metrics from info dict ---
        step_nec_total = float(info.get("step_net_consumption_kwh", 0.0))
        grid_import = float(info.get("grid_import_kwh", max(0.0, step_nec_total)))
        grid_export_step = float(info.get("grid_export_kwh", max(0.0, -step_nec_total)))

        hourly_grid_import[hour].append(grid_import)

        # --- NEC per building ---
        t_nec = max(0, t_after - 1)
        for bi, b in enumerate(city.buildings):
            c3_total += 1
            nec_arr = getattr(b, "net_electricity_consumption", None)
            if nec_arr is not None and hasattr(nec_arr, "__len__") and t_nec < len(nec_arr):
                nec_b = float(nec_arr[t_nec])
            else:
                nec_b = 0.0

            # Per-building import/export
            if nec_b > 0:
                per_building_import[bi] += nec_b
            else:
                per_building_export[bi] += abs(nec_b)

            # C3: building power violation
            if c3_controllable:
                nsl_arr = getattr(b, "_Building__energy_to_non_shiftable_load", [])
                sg_arr = getattr(b, "_Building__solar_generation", [])
                nsl_i = 0.0
                if hasattr(nsl_arr, "__len__") and t_nec < len(nsl_arr):
                    nsl_i = float(nsl_arr[t_nec])
                sg_i = 0.0
                if hasattr(sg_arr, "__len__") and t_nec < len(sg_arr):
                    sg_i = float(sg_arr[t_nec])
                exog = nsl_i + sg_i
                if abs(exog) <= P_building_max:
                    if abs(nec_b) > P_building_max:
                        c3_violations += 1
                else:
                    if abs(nec_b) > abs(exog):
                        c3_violations += 1
            else:
                if abs(nec_b) > P_building_max:
                    c3_violations += 1

        # C4: grid power
        c4_total += 1
        if grid_import > P_grid_max:
            c4_violations += 1

        total_import += grid_import
        total_export += grid_export_step
        peak_nec = max(peak_nec, abs(step_nec_total))

        # Ramping
        ramping_total += abs(step_nec_total - nec_prev)
        nec_prev = step_nec_total

        # Electricity cost
        price = 0.0
        for b in city.buildings:
            pricing_obj = getattr(b, "pricing", None)
            if pricing_obj is not None:
                pa = getattr(pricing_obj, "electricity_pricing", None)
                if pa is not None and hasattr(pa, "__len__") and t_nec < len(pa):
                    price = float(pa[t_nec])
                    break
        if grid_import > 0:
            electricity_cost_total += grid_import * price

        # C2: Battery SoC violations
        for bi_idx, bi in enumerate(batt_indices):
            bld_idx = bi_idx if bi_idx < n_buildings else 0
            if bld_idx < len(city.buildings):
                bld = city.buildings[bld_idx]
                for es in getattr(bld, "electrical_storage_devices", []):
                    soc_arr = getattr(es, "soc", None)
                    if soc_arr is not None and hasattr(soc_arr, "__len__") and t_nec < len(soc_arr):
                        soc_val = float(soc_arr[t_nec])
                        c2_total += 1
                        if soc_val > 0.95 or soc_val < 0.0:
                            c2_violations += 1
                    break

        # Battery action stats
        for bi in batt_indices:
            if bi < len(action):
                a = float(action[bi])
                batt_total_steps += 1
                if a > 0.1:
                    batt_charge_steps += 1
                    batt_daily_charge[day] = True
                elif a < -0.1:
                    batt_discharge_steps += 1
                    batt_daily_discharge[day] = True

        # EV action stats
        for ei in ev_indices:
            if ei < len(action):
                a = float(action[ei])
                ev_connected_steps += 1
                if a > 0.1:
                    ev_charge_steps += 1
                elif a < -0.1:
                    ev_v2g_steps += 1
                    ev_total_v2g_kwh += abs(a) * 6.0  # ~6 kW max charger

        # Track departures
        t_soc = max(0, t_after - 1)
        for bi, ch, cid in charger_info:
            sim = getattr(ch, "charger_simulation", None)
            if sim is None:
                continue
            state_arr = getattr(sim, "_electric_vehicle_charger_state", None)
            dep_time_arr = getattr(sim, "_electric_vehicle_departure_time", None)
            req_soc_arr = getattr(sim, "_electric_vehicle_required_soc_departure", None)
            if state_arr is None or dep_time_arr is None or req_soc_arr is None:
                continue
            if t_after >= len(state_arr):
                continue
            s = float(state_arr[t_after])
            d = float(dep_time_arr[t_after])
            r = float(req_soc_arr[t_after])
            if s == 1.0 and d == 0.0:
                ev = getattr(ch, "connected_electric_vehicle", None)
                actual_soc = 0.0
                if ev is not None:
                    batt = getattr(ev, "battery", None)
                    if batt is not None:
                        soc_series = getattr(batt, "soc", None)
                        if soc_series is not None and hasattr(soc_series, "__len__"):
                            if 0 <= t_soc < len(soc_series):
                                actual_soc = float(np.clip(soc_series[t_soc], 0.0, 1.0))
                departures.append((cid, actual_soc, r, t_after))

        # Collect STEMS reward components from info
        for key in info:
            if key.startswith("r_") or key.startswith("stems_"):
                step_infos_stems[key].append(float(info[key]))

        if terminated or truncated:
            break

        if (step_i + 1) % 2000 == 0:
            dep_so_far = len(departures)
            viol_so_far = sum(1 for _, ds, dr, _ in departures if (dr - ds) > C0_SOC_TOLERANCE)
            log(f"    Step {step_i+1}/{TOTAL_STEPS}: deps={dep_so_far}, "
                f"c0_viol={viol_so_far}, c3={c3_violations}, c4={c4_violations}")

    elapsed = time.time() - t0
    steps_completed = step_i + 1

    # --- Compute final metrics ---
    results = {}
    results["policy_name"] = policy_name
    results["steps_completed"] = steps_completed
    results["eval_seconds"] = elapsed

    # C0: EV departure (with tolerance)
    n_dep = len(departures)
    c0_violated = [(cid, ds, dr, ts) for cid, ds, dr, ts in departures if (dr - ds) > C0_SOC_TOLERANCE]
    n_viol = len(c0_violated)
    results["c0_total_departures"] = n_dep
    results["c0_violated"] = n_viol
    results["c0_violation_pct"] = 100.0 * n_viol / max(n_dep, 1)
    results["c0_mean_dep_soc"] = float(np.mean([ds for _, ds, _, _ in departures])) if departures else 0.0
    results["c0_mean_req_soc"] = float(np.mean([dr for _, _, dr, _ in departures])) if departures else 0.0
    violated_deficits = [dr - ds for _, ds, dr, _ in c0_violated]
    results["c0_mean_deficit"] = float(np.mean(violated_deficits)) if violated_deficits else 0.0
    results["c0_max_deficit"] = float(np.max(violated_deficits)) if violated_deficits else 0.0
    results["c0_violations_detail"] = c0_violated  # list of (charger_id, actual_soc, required_soc, timestep)

    # C2
    results["c2_violations"] = c2_violations
    results["c2_total"] = c2_total
    results["c2_violation_pct"] = 100.0 * c2_violations / max(c2_total, 1)

    # C3
    results["c3_violations"] = c3_violations
    results["c3_total"] = c3_total
    results["c3_violation_pct"] = 100.0 * c3_violations / max(c3_total, 1)

    # C4
    results["c4_violations"] = c4_violations
    results["c4_total"] = c4_total
    results["c4_violation_pct"] = 100.0 * c4_violations / max(c4_total, 1)

    # Battery
    bt = max(batt_total_steps, 1)
    results["batt_charge_pct"] = 100.0 * batt_charge_steps / bt
    results["batt_discharge_pct"] = 100.0 * batt_discharge_steps / bt
    results["batt_idle_pct"] = 100.0 * (bt - batt_charge_steps - batt_discharge_steps) / bt

    all_days = set(list(batt_daily_charge.keys()) + list(batt_daily_discharge.keys()))
    cycling_days = sum(1 for d in all_days if batt_daily_charge.get(d, False) and batt_daily_discharge.get(d, False))
    results["batt_daily_cycling_days"] = cycling_days

    # EV
    et = max(ev_connected_steps, 1)
    results["ev_charge_pct"] = 100.0 * ev_charge_steps / et
    results["ev_v2g_pct"] = 100.0 * ev_v2g_steps / et
    results["ev_idle_pct"] = 100.0 * (et - ev_charge_steps - ev_v2g_steps) / et
    results["ev_total_v2g_kwh"] = ev_total_v2g_kwh

    # CityLearn KPIs
    results["total_import_kwh"] = total_import
    results["total_export_kwh"] = total_export
    results["peak_nec_kw"] = peak_nec
    results["ramping_kwh"] = ramping_total
    results["electricity_cost"] = electricity_cost_total
    mean_import = total_import / max(steps_completed, 1)
    results["load_factor"] = mean_import / max(peak_nec, 1e-8)

    # Per-building breakdown
    results["per_building_import"] = dict(per_building_import)
    results["per_building_export"] = dict(per_building_export)

    # Hourly average grid import profile
    hourly_profile = {}
    for h in range(24):
        vals = hourly_grid_import.get(h, [])
        hourly_profile[h] = float(np.mean(vals)) if vals else 0.0
    results["hourly_avg_grid_import"] = hourly_profile

    # Reward stats
    if step_rewards:
        results["reward_mean"] = float(np.mean(step_rewards))
        results["reward_sum"] = float(np.sum(step_rewards))
        results["reward_std"] = float(np.std(step_rewards))
    else:
        results["reward_mean"] = 0.0
        results["reward_sum"] = 0.0
        results["reward_std"] = 0.0

    # STEMS components
    stems_summary = {}
    for key, vals in step_infos_stems.items():
        stems_summary[key] = {
            "mean": float(np.mean(vals)),
            "sum": float(np.sum(vals)),
            "std": float(np.std(vals)),
        }
    results["stems_components"] = stems_summary

    return results


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------
def print_results(r: dict):
    """Print comprehensive results for a policy."""
    name = r.get("policy_name", "unknown")
    log(f"\n{'='*70}")
    log(f"  RESULTS: {name}")
    log(f"{'='*70}")
    log(f"  Steps: {r['steps_completed']}, Time: {r['eval_seconds']:.1f}s")

    # Constraint violations
    log(f"\n  --- Constraint Violations ---")
    log(f"  C0 (EV departure SoC): {r['c0_violated']}/{r['c0_total_departures']} "
        f"({r['c0_violation_pct']:.1f}%) [tolerance={C0_SOC_TOLERANCE}]")
    log(f"    Mean departure SoC: {r['c0_mean_dep_soc']:.4f}")
    log(f"    Mean required SoC:  {r['c0_mean_req_soc']:.4f}")
    if r['c0_violated'] > 0:
        log(f"    Mean deficit:       {r['c0_mean_deficit']:.4f}")
        log(f"    Max deficit:        {r['c0_max_deficit']:.4f}")
    log(f"  C2 (battery SoC):     {r['c2_violations']}/{r['c2_total']} "
        f"({r['c2_violation_pct']:.1f}%)")
    log(f"  C3 (building power):  {r['c3_violations']}/{r['c3_total']} "
        f"({r['c3_violation_pct']:.1f}%)")
    log(f"  C4 (grid power):      {r['c4_violations']}/{r['c4_total']} "
        f"({r['c4_violation_pct']:.1f}%)")

    # CityLearn KPIs
    log(f"\n  --- CityLearn KPIs ---")
    log(f"  Total import:     {r['total_import_kwh']:.1f} kWh")
    log(f"  Total export:     {r['total_export_kwh']:.1f} kWh")
    log(f"  Peak NEC:         {r['peak_nec_kw']:.2f} kW")
    log(f"  Ramping:          {r['ramping_kwh']:.1f} kWh")
    log(f"  Electricity cost: {r['electricity_cost']:.2f}")
    log(f"  Load factor:      {r['load_factor']:.4f}")

    # Per-building breakdown
    log(f"\n  --- Per-Building Import/Export (kWh) ---")
    for bi in sorted(set(list(r['per_building_import'].keys()) + list(r['per_building_export'].keys()))):
        imp = r['per_building_import'].get(bi, 0.0)
        exp = r['per_building_export'].get(bi, 0.0)
        log(f"    Building {bi}: import={imp:.1f}, export={exp:.1f}")

    # Battery behavior
    log(f"\n  --- Battery Behavior ---")
    log(f"  Charge:     {r['batt_charge_pct']:.1f}%")
    log(f"  Discharge:  {r['batt_discharge_pct']:.1f}%")
    log(f"  Idle:       {r['batt_idle_pct']:.1f}%")
    log(f"  Days with charge+discharge cycling: {r['batt_daily_cycling_days']}")

    # EV behavior
    log(f"\n  --- EV Behavior ---")
    log(f"  Charge:     {r['ev_charge_pct']:.1f}%")
    log(f"  V2G:        {r['ev_v2g_pct']:.1f}%")
    log(f"  Idle:       {r['ev_idle_pct']:.1f}%")
    log(f"  V2G energy: {r['ev_total_v2g_kwh']:.1f} kWh (estimated)")

    # Reward
    log(f"\n  --- Reward ---")
    log(f"  Mean: {r['reward_mean']:.4f}, Sum: {r['reward_sum']:.1f}, Std: {r['reward_std']:.4f}")

    # STEMS components
    stems = r.get("stems_components", {})
    if stems:
        log(f"\n  --- STEMS Reward Components ---")
        for key in sorted(stems.keys()):
            s = stems[key]
            log(f"    {key}: mean={s['mean']:.4f}, sum={s['sum']:.1f}, std={s['std']:.4f}")

    # Hourly grid import profile
    log(f"\n  --- Hourly Avg Grid Import (kWh) ---")
    profile = r.get("hourly_avg_grid_import", {})
    bars = ""
    for h in range(24):
        val = profile.get(h, 0.0)
        bar_len = int(val / 0.2) if val > 0 else 0
        bar_len = min(bar_len, 40)
        bars = "#" * bar_len
        log(f"    {h:02d}:00  {val:6.3f} kWh  {bars}")

    # Per-departure C0 violations detail
    c0_detail = r.get("c0_violations_detail", [])
    if c0_detail:
        log(f"\n  --- C0 Violation Details (first 30) ---")
        log(f"    {'Charger':<16} {'ActualSoC':>10} {'ReqSoC':>10} {'Deficit':>10} {'Timestep':>10}")
        for cid, ds, dr, ts in c0_detail[:30]:
            deficit = dr - ds
            log(f"    {str(cid):<16} {ds:>10.4f} {dr:>10.4f} {deficit:>10.4f} {ts:>10d}")
        if len(c0_detail) > 30:
            log(f"    ... and {len(c0_detail) - 30} more violations")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 70)
    log("  BASELINE EVALUATION - 5 BUILDING SCHEMA")
    log(f"  Schema: {SCHEMA_PATH}")
    log(f"  Seed: {SEED}, Steps: {TOTAL_STEPS}")
    log(f"  C0 SoC tolerance: {C0_SOC_TOLERANCE}")
    log("=" * 70)
    log()

    # Parse env vars from run_r19_ablation.sh as the baseline config
    script_path = os.path.join(PROJECT, "run_r19_ablation.sh")
    if not os.path.exists(script_path):
        log(f"ERROR: Cannot find run script: {script_path}")
        sys.exit(1)
    env_vars = parse_exports(script_path)
    log(f"Parsed {len(env_vars)} env vars from {script_path}")

    all_results = []

    # -----------------------------------------------------------------------
    # 1. Zero action baseline
    # -----------------------------------------------------------------------
    log("\n" + "-" * 70)
    log("  POLICY: zero_action")
    log("-" * 70)
    try:
        with suppress_stdout():
            env = build_env(env_vars)
        act_dim = env.action_space.shape[0]
        log(f"  Env built: obs_dim={env.observation_space.shape[0]}, act_dim={act_dim}")

        def zero_fn(obs):
            return np.zeros(act_dim)

        result = evaluate_policy("zero_action", env, zero_fn, env_vars)
        all_results.append(result)
        print_results(result)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()
    log()

    # -----------------------------------------------------------------------
    # 2. Greedy EV baseline
    # -----------------------------------------------------------------------
    log("-" * 70)
    log("  POLICY: greedy_ev")
    log("-" * 70)
    try:
        with suppress_stdout():
            env = build_env(env_vars)
        act_dim = env.action_space.shape[0]
        act_names = get_action_names(env)
        log(f"  Env built: obs_dim={env.observation_space.shape[0]}, act_dim={act_dim}")
        log(f"  Action names: {act_names}")

        ev_idx = [i for i, n in enumerate(act_names) if "electric_vehicle" in str(n).lower()]
        batt_idx = [i for i, n in enumerate(act_names) if str(n).strip().lower() == "electrical_storage"]
        log(f"  EV indices: {ev_idx}, Battery indices: {batt_idx}")

        def greedy_ev_fn(obs):
            action = np.zeros(act_dim)
            for i in ev_idx:
                action[i] = 1.0  # full charge
            # battery indices remain 0.0
            return action

        result = evaluate_policy("greedy_ev", env, greedy_ev_fn, env_vars)
        all_results.append(result)
        print_results(result)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()
    log()

    # -----------------------------------------------------------------------
    # 3. SmartV2GRBC baseline
    # -----------------------------------------------------------------------
    log("-" * 70)
    log("  POLICY: SmartV2GRBC")
    log("-" * 70)
    try:
        with suppress_stdout():
            env = build_env(env_vars)
        log(f"  Env built: obs_dim={env.observation_space.shape[0]}, act_dim={env.action_space.shape[0]}")

        from scripts.rbc_policy import SmartV2GRBC
        rbc = SmartV2GRBC(env)

        def rbc_fn(obs):
            return rbc.predict(obs)

        result = evaluate_policy("SmartV2GRBC", env, rbc_fn, env_vars)
        all_results.append(result)
        print_results(result)
    except Exception as e:
        log(f"  ERROR: {e}")
        traceback.print_exc()
    log()

    # -----------------------------------------------------------------------
    # Comparison summary
    # -----------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("  COMPARISON SUMMARY")
    log("=" * 70)
    log()

    header = f"{'Policy':<16} {'C0%':>7} {'C0 viol':>8} {'C2%':>7} {'C3%':>7} {'C4%':>7} " \
             f"{'Import':>9} {'Export':>9} {'Peak':>7} {'Reward':>8} {'Time':>6}"
    log(header)
    log("-" * len(header))
    for r in all_results:
        if "error" in r:
            log(f"{r.get('policy_name', '?'):<16} ERROR")
            continue
        log(f"{r['policy_name']:<16} "
            f"{r['c0_violation_pct']:>6.1f}% "
            f"{r['c0_violated']:>7d} "
            f"{r['c2_violation_pct']:>6.1f}% "
            f"{r['c3_violation_pct']:>6.1f}% "
            f"{r['c4_violation_pct']:>6.1f}% "
            f"{r['total_import_kwh']:>8.0f} "
            f"{r['total_export_kwh']:>8.0f} "
            f"{r['peak_nec_kw']:>6.1f} "
            f"{r['reward_sum']:>7.0f} "
            f"{r['eval_seconds']:>5.0f}s")

    log()
    log("  --- Battery Behavior Comparison ---")
    header2 = f"{'Policy':<16} {'Charge%':>8} {'Discharge%':>11} {'Idle%':>7} {'Cycling':>8}"
    log(header2)
    log("-" * len(header2))
    for r in all_results:
        if "error" in r:
            continue
        log(f"{r['policy_name']:<16} "
            f"{r['batt_charge_pct']:>7.1f}% "
            f"{r['batt_discharge_pct']:>10.1f}% "
            f"{r['batt_idle_pct']:>6.1f}% "
            f"{r['batt_daily_cycling_days']:>7d}")

    log()
    log("  --- EV Behavior Comparison ---")
    header3 = f"{'Policy':<16} {'Charge%':>8} {'V2G%':>7} {'Idle%':>7} {'V2G kWh':>9}"
    log(header3)
    log("-" * len(header3))
    for r in all_results:
        if "error" in r:
            continue
        log(f"{r['policy_name']:<16} "
            f"{r['ev_charge_pct']:>7.1f}% "
            f"{r['ev_v2g_pct']:>6.1f}% "
            f"{r['ev_idle_pct']:>6.1f}% "
            f"{r['ev_total_v2g_kwh']:>8.1f}")

    log()
    log("  --- Hourly Grid Import Comparison ---")
    header4 = "Hour  " + "  ".join(f"{r['policy_name']:>12}" for r in all_results if "error" not in r)
    log(header4)
    log("-" * len(header4))
    for h in range(24):
        vals = []
        for r in all_results:
            if "error" in r:
                continue
            vals.append(f"{r['hourly_avg_grid_import'].get(h, 0.0):>12.3f}")
        log(f"{h:02d}:00 " + "  ".join(vals))

    log()
    log("Done.")


if __name__ == "__main__":
    main()
