#!/usr/bin/env python3
"""
Per-building violation rate evaluation for R19 and R21.

Revised metric:
  violation_rate = sum(buildings_violated_per_step) / (total_steps * n_buildings) * 100

This replaces the old "any building violated" binary metric which inflated rates.
Applies to C2 (battery SoC) and C3 (building power).
C0 uses per-departure metric, C4 is grid-level (single value).

Usage:
    python scripts/eval_per_building.py --run r21
    python scripts/eval_per_building.py --run r19
    python scripts/eval_per_building.py --run both
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
# Shared env vars (R19 and R21 use the same env config for eval)
# =====================================================================
BASE_ENV_VARS = {
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
    # Reward weights
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

RUN_CONFIGS = {
    "r21": {
        "run_dir": "runs/r21_ppo/5bld",
        "epoch": 89,
        "batt_clamp": "1",  # Safety Projection ON
        "label": "R21 (PPOLagMulti + Curriculum + Safety Proj.)",
    },
    "r19": {
        "run_dir": "runs/r19_ablation/5bld",
        "epoch": 80,
        "batt_clamp": "0",  # R19 did not use battery clamp
        "label": "R19 (PPOLagMulti + PID baseline)",
    },
}

P_BUILDING_MAX = 4.6083
P_GRID_MAX = 10.2352
SOC_LOW = 0.0
SOC_HIGH = 0.95
SOC_UPPER_CLAMP = 0.94
BATT_DT = 1.0

COST_LIMITS = {
    "C0_ev_departure": 1800,
    "C1_ev_saute": 1500,
    "C2_battery_soc": 4000,
    "C3_building_power": 3000,
    "C4_grid_power": 1500,
}


def set_env(batt_clamp="1"):
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(k, None)
    for k, v in BASE_ENV_VARS.items():
        os.environ[k] = v
    os.environ["CITYLEARN_BATT_CLAMP"] = batt_clamp


# Lazy imports after env set
def _do_imports():
    global make_base_env, CityLearnSafetyEnvV3, ForecastObsWrapper, unwrap_to_raw_citylearn_env
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env


def discover_battery_actions(raw_env):
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


def find_checkpoint(run_dir, epoch):
    pattern = os.path.join(PROJECT, run_dir, "PPOLagMulti-*", "seed-*", "torch_save")
    save_dirs = sorted(glob_mod.glob(pattern))
    if not save_dirs:
        return ""
    # Try all directories for the epoch
    for save_dir in reversed(save_dirs):
        ckpt = os.path.join(save_dir, f"epoch-{epoch}.pt")
        if os.path.exists(ckpt):
            return ckpt
    return ""


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


def pct(a, b):
    return 100.0 * a / b if b > 0 else 0.0


def evaluate_run(run_name, seed=42):
    """Run evaluation and return results dict with per-building violation metrics."""
    cfg = RUN_CONFIGS[run_name]
    ckpt_path = find_checkpoint(cfg["run_dir"], cfg["epoch"])
    if not ckpt_path:
        print(f"ERROR: No checkpoint found for {run_name}")
        return None

    print(f"\n{'='*70}")
    print(f"  {cfg['label']}")
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"  Battery clamp: {'ON' if cfg['batt_clamp'] == '1' else 'OFF'}")
    print(f"{'='*70}")

    # Reset env vars and schema cache
    import citylearn_safe.schema_index as si
    si._CACHE = None
    set_env(batt_clamp=cfg["batt_clamp"])
    _do_imports()

    actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(ckpt_path)

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)
    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(getattr(raw, "buildings", []))
    n_buildings = len(buildings)

    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]

    # Safety Projection
    use_clamp = cfg["batt_clamp"] == "1"
    batt_action_map = discover_battery_actions(raw) if use_clamp else []

    need_pad = obs_dim > env.observation_space.shape[0]
    pad_dim = obs_dim - env.observation_space.shape[0] if need_pad else 0

    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    # Tracking
    rewards = []
    costs = []
    c0_vals, c1_vals, c2_vals, c3_vals, c4_vals = [], [], [], [], []

    # EV departure
    total_departures = 0
    violated_departures = 0
    departure_deficits = []
    ev_tracker = {}

    # === PER-BUILDING violation counters ===
    # C2: per-building SoC violations
    c2_building_violations_total = 0   # sum of violated buildings across all steps
    c2_building_opportunities = 0      # total_steps * n_buildings

    # C3: per-building power violations
    c3_building_violations_total = 0
    c3_building_opportunities = 0

    # C4: grid-level (single value per step)
    c4_violation_steps = 0

    # V2G behavior
    ev_charge_steps = 0
    ev_discharge_steps = 0
    ev_idle_steps = 0
    total_ev_steps = 0

    # Hourly violations (per-building rates)
    hourly_c3_bld_violations = [0] * 24
    hourly_c3_bld_opportunities = [0] * 24
    hourly_c4_violations = [0] * 24
    hourly_counts = [0] * 24

    # CityLearn KPIs
    citylearn_kpis = {}

    # Energy
    grid_imports = []
    grid_exports = []

    print(f"  Running rollout ({n_buildings} buildings)...")

    while not done:
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

        if use_clamp:
            action = clamp_battery_actions(action, batt_action_map, raw)

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
        step += 1

        rewards.append(reward)
        costs.append(info.get("cost", 0.0))
        c0_vals.append(info.get("cost_ev_departure", 0.0) + info.get("cost_ev_dense", 0.0))
        c1_vals.append(info.get("cost_ev_dense", 0.0))
        c2_vals.append(info.get("cost_soc_pnorm", info.get("cost_stems_battery", 0.0)))
        c3_vals.append(info.get("cost_stems_building_power", 0.0))
        c4_vals.append(info.get("cost_stems_grid_power", 0.0))

        grid_imports.append(info.get("grid_import_kwh", 0.0))
        grid_exports.append(info.get("grid_export_kwh", 0.0))

        # === C2: Per-building battery SoC violation ===
        c2_step_violations = 0
        for b_idx, bld in enumerate(buildings):
            try:
                es = getattr(bld, "electrical_storage", None)
                soc = getattr(es, "soc", None) if es else None
                if soc is not None and len(soc) > t_idx:
                    s = float(np.clip(soc[t_idx], 0, 1))
                    violation = max(0.0, SOC_LOW - s) + max(0.0, s - SOC_HIGH)
                    if violation > 0:
                        c2_step_violations += 1
            except Exception:
                pass
        c2_building_violations_total += c2_step_violations
        c2_building_opportunities += n_buildings

        # === C3: Per-building power violation ===
        c3_step_violations = 0
        for b_idx, bld in enumerate(buildings):
            try:
                nec = getattr(bld, 'net_electricity_consumption', None)
                if nec is not None and len(nec) > t_idx:
                    p = abs(float(nec[t_idx]))
                    if p > P_BUILDING_MAX:
                        c3_step_violations += 1
            except Exception:
                pass
        c3_building_violations_total += c3_step_violations
        c3_building_opportunities += n_buildings

        # Hourly per-building tracking
        hourly_c3_bld_violations[hour] += c3_step_violations
        hourly_c3_bld_opportunities[hour] += n_buildings

        # === C4: Grid-level (unchanged) ===
        grid_viol = info.get("grid_power_violation", 0.0)
        if grid_viol > 0:
            c4_violation_steps += 1
            hourly_c4_violations[hour] += 1
        hourly_counts[hour] += 1

        # CityLearn KPIs at episode end
        if done:
            for kpi_key in ["citylearn_electricity_consumption_total",
                            "citylearn_carbon_emissions_total",
                            "citylearn_cost_total",
                            "citylearn_daily_peak_average",
                            "citylearn_all_time_peak_average",
                            "citylearn_ramping_average",
                            "citylearn_zero_net_energy"]:
                citylearn_kpis[kpi_key] = info.get(kpi_key, float("nan"))

        if step % 2000 == 0:
            print(f"    Step {step}/{8759}")

    # =====================================================================
    # COMPUTE RESULTS
    # =====================================================================
    total_reward = sum(rewards)
    total_cost = sum(costs)
    total_c0 = sum(c0_vals)
    total_c1 = sum(c1_vals)
    total_c2 = sum(c2_vals)
    total_c3 = sum(c3_vals)
    total_c4 = sum(c4_vals)

    # Per-building violation rates (THE REVISED METRIC)
    c2_per_building_pct = pct(c2_building_violations_total, c2_building_opportunities)
    c3_per_building_pct = pct(c3_building_violations_total, c3_building_opportunities)
    c4_step_pct = pct(c4_violation_steps, step)
    c0_departure_pct = pct(violated_departures, total_departures)

    # Hourly per-building rates
    hourly_c3_perbld = []
    hourly_c4_rate = []
    for h in range(24):
        c3r = pct(hourly_c3_bld_violations[h], hourly_c3_bld_opportunities[h])
        c4r = pct(hourly_c4_violations[h], hourly_counts[h])
        hourly_c3_perbld.append(round(c3r, 2))
        hourly_c4_rate.append(round(c4r, 2))

    # Print summary
    print(f"\n  {'='*60}")
    print(f"  RESULTS: {cfg['label']}")
    print(f"  {'='*60}")
    print(f"  Steps: {step},  Buildings: {n_buildings}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Total Cost:   {total_cost:.2f}")
    print(f"\n  Per-constraint costs:")
    for name, val, limit in [("C0 EV departure", total_c0, 1800),
                              ("C1 EV Sauté", total_c1, 1500),
                              ("C2 Battery SoC", total_c2, 4000),
                              ("C3 Building power", total_c3, 3000),
                              ("C4 Grid power", total_c4, 1500)]:
        ratio = val / limit if limit > 0 else 0
        status = "PASS" if val <= limit else "FAIL"
        print(f"    {name:<20} {val:>10.1f} / {limit:>6} = {ratio:.2f}x  {status}")

    print(f"\n  === REVISED PER-BUILDING VIOLATION RATES ===")
    print(f"  C0 EV departure:      {c0_departure_pct:.2f}%  ({violated_departures}/{total_departures} departures)")
    print(f"  C2 Battery SoC:       {c2_per_building_pct:.2f}%  ({c2_building_violations_total}/{c2_building_opportunities} building-steps)")
    print(f"  C3 Building power:    {c3_per_building_pct:.2f}%  ({c3_building_violations_total}/{c3_building_opportunities} building-steps)")
    print(f"  C4 Grid power:        {c4_step_pct:.2f}%  ({c4_violation_steps}/{step} steps)")

    print(f"\n  V2G behavior:")
    print(f"    Charge:    {pct(ev_charge_steps, total_ev_steps):.1f}%")
    print(f"    Discharge: {pct(ev_discharge_steps, total_ev_steps):.1f}%")
    print(f"    Idle:      {pct(ev_idle_steps, total_ev_steps):.1f}%")

    if citylearn_kpis:
        print(f"\n  CityLearn KPIs:")
        for k, v in sorted(citylearn_kpis.items()):
            if not np.isnan(v):
                print(f"    {k:<50} {v:.4f}")

    results = {
        "run": run_name,
        "label": cfg["label"],
        "checkpoint": ckpt_path,
        "epoch": cfg["epoch"],
        "batt_clamp": cfg["batt_clamp"],
        "steps": step,
        "n_buildings": n_buildings,
        "total_reward": total_reward,
        "total_cost": total_cost,
        "per_constraint": {
            "C0_ev_departure": {"cost": total_c0, "limit": 1800},
            "C1_ev_saute": {"cost": total_c1, "limit": 1500},
            "C2_battery_soc": {"cost": total_c2, "limit": 4000},
            "C3_building_power": {"cost": total_c3, "limit": 3000},
            "C4_grid_power": {"cost": total_c4, "limit": 1500},
        },
        "ev_departures": {
            "total": total_departures,
            "violated": violated_departures,
            "violation_pct": c0_departure_pct,
            "mean_deficit": float(np.mean(departure_deficits)) if departure_deficits else 0.0,
        },
        "per_building_violation_rates_pct": {
            "C0_departure": c0_departure_pct,
            "C2_battery_soc": c2_per_building_pct,
            "C3_building_power": c3_per_building_pct,
            "C4_grid_power": c4_step_pct,
        },
        "per_building_violation_counts": {
            "C2_violated": c2_building_violations_total,
            "C2_total": c2_building_opportunities,
            "C3_violated": c3_building_violations_total,
            "C3_total": c3_building_opportunities,
            "C4_violated": c4_violation_steps,
            "C4_total": step,
        },
        "v2g_behavior": {
            "charge_pct": pct(ev_charge_steps, total_ev_steps),
            "discharge_pct": pct(ev_discharge_steps, total_ev_steps),
            "idle_pct": pct(ev_idle_steps, total_ev_steps),
        },
        "citylearn_kpis": {k: v for k, v in citylearn_kpis.items() if not np.isnan(v)},
        "hourly_c3_perbld_pct": hourly_c3_perbld,
        "hourly_c4_pct": hourly_c4_rate,
        "energy_balance": {
            "total_import_kwh": sum(grid_imports),
            "total_export_kwh": sum(grid_exports),
            "net_consumption_kwh": sum(grid_imports) - sum(grid_exports),
        },
    }

    out_path = os.path.join(PROJECT, "runs", f"{run_name}_perbld_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["r19", "r21", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_env()
    _do_imports()

    results = {}
    if args.run in ("r21", "both"):
        results["r21"] = evaluate_run("r21", seed=args.seed)
    if args.run in ("r19", "both"):
        results["r19"] = evaluate_run("r19", seed=args.seed)

    if len(results) == 2:
        print(f"\n{'='*70}")
        print(f"  COMPARISON: R19 vs R21 (Per-Building Violation Rates)")
        print(f"{'='*70}")
        print(f"  {'Metric':<35} {'R19':>12} {'R21':>12} {'Change':>12}")
        print(f"  {'-'*71}")

        r19 = results["r19"]
        r21 = results["r21"]

        # Reward
        print(f"  {'Total Reward':<35} {r19['total_reward']:>12.1f} {r21['total_reward']:>12.1f} {r21['total_reward']-r19['total_reward']:>+12.1f}")
        print(f"  {'Total Cost':<35} {r19['total_cost']:>12.1f} {r21['total_cost']:>12.1f} {r21['total_cost']-r19['total_cost']:>+12.1f}")

        print(f"\n  --- Per-Building Violation Rates (%) ---")
        for key, label in [("C0_departure", "C0 EV departure (per-dep)"),
                           ("C2_battery_soc", "C2 Battery SoC (per-bld)"),
                           ("C3_building_power", "C3 Building power (per-bld)"),
                           ("C4_grid_power", "C4 Grid power (per-step)")]:
            v19 = r19["per_building_violation_rates_pct"][key]
            v21 = r21["per_building_violation_rates_pct"][key]
            delta = v21 - v19
            print(f"  {label:<35} {v19:>11.2f}% {v21:>11.2f}% {delta:>+11.2f}%")

        # Save combined results
        combined_path = os.path.join(PROJECT, "runs", "r19_r21_comparison.json")
        with open(combined_path, "w") as f:
            json.dump({
                "r19": results["r19"],
                "r21": results["r21"],
            }, f, indent=2)
        print(f"\n  Combined results: {combined_path}")
