#!/usr/bin/env python3
"""
3-Way Ablation Evaluation: R21 baseline vs Mask Treatment vs Single-lambda.

Runs each checkpoint through 8760 deterministic steps with matching env config,
then prints a side-by-side comparison table.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

P_BUILDING_MAX = 4.6083
P_GRID_MAX = 10.2352
SOC_UPPER_CLAMP = 0.94
BATT_DT = 1.0

# =========================================================================
# Common env vars (R21 config, 5-building schema)
# =========================================================================
COMMON_ENV = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_PID_LAGRANGE": "0",
    "CITYLEARN_ACTION_MASK": "0",
    "CITYLEARN_BATT_CLAMP": "1",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_KPI_RUN_NAME": "__eval__",
    "STEMS_ALPHA_GRID": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.3",
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_BUILD": "0.0",
    "STEMS_XI_RENEWABLE": "0.2",
    "STEMS_BETA_RAMP": "0.3",
    "STEMS_LAMBDA_EV": "5.0",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_ALPHA_EV_GUARD": "1.0",
    "STEMS_ALPHA_V2G_CONTEXT": "3.0",
}


def set_env(overrides=None):
    """Clear all CITYLEARN/STEMS env vars, then set common + overrides."""
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_", "MLAG_",
                         "SAUTE_", "SHAPING_", "EV_CLIP_", "FRAME_STACK_")):
            os.environ.pop(k, None)
    for k, v in COMMON_ENV.items():
        os.environ[k] = v
    if overrides:
        for k, v in overrides.items():
            os.environ[k] = v


# Set env so imports work
set_env()

from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.extractors import unwrap_to_raw_citylearn_env
from citylearn_safe.action_mask_wrapper import ActionMaskWrapper
import citylearn_safe.schema_index as si


# =========================================================================
# Agent definitions
# =========================================================================
AGENTS = [
    {
        "name": "R21 Baseline",
        "ckpt": f"{PROJECT}/runs/r21_ppo/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-03-13-04-07-53/torch_save/epoch-89.pt",
        "env_overrides": {},
        "use_mask": False,
    },
    {
        "name": "Mask Treatment",
        "ckpt": f"{PROJECT}/runs/ablation_mask_treatment_s42/5bld/PPOLagMulti-{{CityLearnSafety-V2G-v2}}/seed-042-2026-03-21-07-44-08/torch_save/epoch-89.pt",
        "env_overrides": {"CITYLEARN_ACTION_MASK": "1"},
        "use_mask": True,
    },
    {
        "name": "Single-lambda",
        "ckpt": f"{PROJECT}/runs/ablation_single_lag/5bld/PPOLag-{{CityLearnSafety-V2G-v2}}/seed-000-2026-03-20-06-48-52/torch_save/epoch-89.pt",
        "env_overrides": {},
        "use_mask": False,
    },
]


# =========================================================================
# Actor
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


# =========================================================================
# Battery safety projection
# =========================================================================
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


# =========================================================================
# Evaluation
# =========================================================================
def pct(a, b):
    return 100.0 * a / b if b > 0 else 0.0


def evaluate_agent(agent_def, seed=42):
    """Run one agent through 8760 steps. Returns dict of metrics."""
    name = agent_def["name"]
    ckpt_path = agent_def["ckpt"]
    use_mask = agent_def["use_mask"]

    print(f"\n{'='*70}")
    print(f"  EVALUATING: {name}")
    print(f"  Checkpoint: {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))}")
    print(f"  Action Mask: {'ON' if use_mask else 'OFF'}")
    print(f"{'='*70}")

    # Set environment
    set_env(agent_def["env_overrides"])
    si._CACHE = None

    # Load actor
    actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(ckpt_path)
    print(f"  Actor: obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  Obs normalizer: {'YES' if obs_mean is not None else 'NO'}")

    # Build environment
    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnv(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)

    if use_mask:
        env = ActionMaskWrapper(env)

    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(getattr(raw, "buildings", []))
    n_buildings = len(buildings)

    # Action indices
    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        action_names = names_raw[0]
    else:
        action_names = list(names_raw)
    batt_idx = [i for i, n in enumerate(action_names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(action_names) if "electric_vehicle_storage_charger_" in str(n).lower()]

    # Battery safety projection (only when mask is OFF -- mask handles C2 itself)
    batt_action_map = discover_battery_actions(raw) if not use_mask else []
    batt_clamp_count = 0

    # Handle obs_dim mismatch (saute adds +1)
    env_obs_dim = env.observation_space.shape[0]
    need_pad = obs_dim > env_obs_dim
    pad_dim = obs_dim - env_obs_dim if need_pad else 0
    if need_pad:
        print(f"  Padding obs: env={env_obs_dim} -> ckpt={obs_dim} (+{pad_dim})")

    # Reset
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    # Tracking
    rewards = []
    costs = []
    actions_all = []

    # EV departure tracking
    total_departures = 0
    violated_departures = 0
    departure_deficits = []
    ev_tracker = {}

    # Per-constraint
    c0_vals, c2_vals, c3_vals, c4_vals = [], [], [], []

    # Battery SoC
    soc_violation_steps = 0
    soc_all_steps = []
    soc_min_per_building = [1.0] * n_buildings
    soc_max_per_building = [0.0] * n_buildings

    # Building power
    building_power_violations = [0] * n_buildings
    building_power_max = [0.0] * n_buildings
    building_power_all = [[] for _ in range(n_buildings)]
    c3_violation_steps = 0

    # Grid
    grid_violation_steps = 0
    grid_power_max = 0.0
    grid_power_all = []

    # V2G
    ev_charge_steps = 0
    ev_discharge_steps = 0
    ev_idle_steps = 0
    total_ev_steps = 0

    # CityLearn KPIs
    citylearn_kpis = {}

    # STEMS components
    reward_components = defaultdict(list)

    # =====================================================================
    # ROLLOUT
    # =====================================================================
    print(f"  Running 8760-step rollout...")

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

        # Battery safety projection (when mask is OFF)
        if not use_mask:
            action_pre = action.copy()
            action = clamp_battery_actions(action, batt_action_map, raw)
            if not np.array_equal(action, action_pre):
                batt_clamp_count += 1

        actions_all.append(action.copy())
        t_now = int(getattr(raw, "time_step", 0))
        t_idx = max(0, t_now - 1)

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
                                soc_arr_ev = getattr(bt, 'soc', None)
                                if soc_arr_ev is not None:
                                    sn = np.asarray(soc_arr_ev, dtype=float)
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

        # === Step ===
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cost = info.get("cost", 0.0)
        step += 1

        rewards.append(reward)
        costs.append(cost)

        c0_vals.append(info.get("cost_ev_departure", 0.0) + info.get("cost_ev_dense", 0.0))
        c2_vals.append(info.get("cost_soc_pnorm", info.get("cost_stems_battery", 0.0)))
        c3_vals.append(info.get("cost_stems_building_power", 0.0))
        c4_vals.append(info.get("cost_stems_grid_power", 0.0))

        for key in ["reward_stems_total", "reward_ev_shaping"]:
            reward_components[key].append(info.get(key, 0.0))

        # C2: Battery SoC
        if info.get("battery_soc_violation_any", 0.0) > 0:
            soc_violation_steps += 1
        soc_step = []
        for b_idx_s, bld_s in enumerate(buildings):
            try:
                es = getattr(bld_s, "electrical_storage", None)
                soc = getattr(es, "soc", None) if es else None
                if soc is not None and len(soc) > t_idx:
                    s = float(np.clip(soc[t_idx], 0, 1))
                else:
                    s = 0.5
                soc_step.append(s)
                soc_min_per_building[b_idx_s] = min(soc_min_per_building[b_idx_s], s)
                soc_max_per_building[b_idx_s] = max(soc_max_per_building[b_idx_s], s)
            except Exception:
                soc_step.append(0.5)
        soc_all_steps.append(soc_step)

        # C3: Building power
        bld_viol = info.get("building_power_violation", 0.0)
        if bld_viol > 0:
            c3_violation_steps += 1
        for b_idx_p, bld_p in enumerate(buildings):
            try:
                nec = getattr(bld_p, 'net_electricity_consumption', None)
                if nec is not None and len(nec) > t_idx:
                    p = abs(float(nec[t_idx]))
                    building_power_all[b_idx_p].append(p)
                    building_power_max[b_idx_p] = max(building_power_max[b_idx_p], p)
                    if p > P_BUILDING_MAX:
                        building_power_violations[b_idx_p] += 1
            except Exception:
                pass

        # C4: Grid power
        if info.get("grid_power_violation", 0.0) > 0:
            grid_violation_steps += 1
        gi = info.get("grid_import_kwh", 0.0)
        grid_power_all.append(gi)
        grid_power_max = max(grid_power_max, gi)

        # CityLearn KPIs on last step
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

        if step % 2000 == 0:
            print(f"    Step {step}: R={sum(rewards[-2000:]):.1f}, "
                  f"C3_cum={sum(c3_vals):.0f}, C4_cum={sum(c4_vals):.0f}")

    # =====================================================================
    # Compute metrics
    # =====================================================================
    total_reward = sum(rewards)
    total_cost = sum(costs)
    total_c0 = sum(c0_vals)
    total_c2 = sum(c2_vals)
    total_c3 = sum(c3_vals)
    total_c4 = sum(c4_vals)

    ev_viol_pct = pct(violated_departures, total_departures)
    violated_deficits = [d for d in departure_deficits if d > 0.01]
    mean_deficit_violated = np.mean(violated_deficits) if violated_deficits else 0.0

    actions_arr = np.array(actions_all)
    soc_arr = np.array(soc_all_steps)

    c3_viol_pct = pct(c3_violation_steps, step)
    c4_viol_pct = pct(grid_violation_steps, step)

    # Per-building battery SoC stats
    batt_soc_stats = {}
    for b in range(n_buildings):
        col = soc_arr[:, b] if b < soc_arr.shape[1] else np.array([0.5])
        batt_soc_stats[f"bld{b}_mean"] = float(np.mean(col))
        batt_soc_stats[f"bld{b}_min"] = float(soc_min_per_building[b])
        batt_soc_stats[f"bld{b}_max"] = float(soc_max_per_building[b])

    # Per-building power stats
    bld_pwr_stats = {}
    for b in range(n_buildings):
        pwr = np.array(building_power_all[b]) if building_power_all[b] else np.array([0.0])
        bld_pwr_stats[f"bld{b}_max_kW"] = float(building_power_max[b])
        bld_pwr_stats[f"bld{b}_p95_kW"] = float(np.percentile(pwr, 95))
        bld_pwr_stats[f"bld{b}_viols"] = building_power_violations[b]

    # EV action stats per charger
    ev_action_stats = {}
    for i, ei in enumerate(ev_idx):
        col = actions_arr[:, ei]
        ev_action_stats[f"ev{i}_mean"] = float(col.mean())
        ev_action_stats[f"ev{i}_std"] = float(col.std())
        ev_action_stats[f"ev{i}_charge%"] = float(pct(np.sum(col > 0.05), len(col)))
        ev_action_stats[f"ev{i}_discharge%"] = float(pct(np.sum(col < -0.05), len(col)))

    gp = np.array(grid_power_all)

    metrics = {
        "name": name,
        "steps": step,
        "total_reward": total_reward,
        "mean_step_reward": total_reward / step,
        "total_cost": total_cost,
        "C0_ev_departure_cost": total_c0,
        "C2_battery_soc_cost": total_c2,
        "C3_building_power_cost": total_c3,
        "C4_grid_power_cost": total_c4,
        "ev_departures_total": total_departures,
        "ev_departures_violated": violated_departures,
        "ev_departure_viol_pct": ev_viol_pct,
        "ev_mean_deficit_violated": mean_deficit_violated,
        "c2_viol_steps_pct": pct(soc_violation_steps, step),
        "c3_viol_steps_pct": c3_viol_pct,
        "c4_viol_steps_pct": c4_viol_pct,
        "batt_clamp_count": batt_clamp_count,
        "batt_soc": batt_soc_stats,
        "bld_power": bld_pwr_stats,
        "ev_actions": ev_action_stats,
        "v2g_charge_pct": pct(ev_charge_steps, total_ev_steps),
        "v2g_discharge_pct": pct(ev_discharge_steps, total_ev_steps),
        "v2g_idle_pct": pct(ev_idle_steps, total_ev_steps),
        "grid_max_kW": grid_power_max,
        "grid_p95_kW": float(np.percentile(gp, 95)) if len(gp) > 0 else 0.0,
        "grid_mean_kW": float(np.mean(gp)) if len(gp) > 0 else 0.0,
        "citylearn_kpis": citylearn_kpis,
    }

    # Print summary for this agent
    print(f"\n  --- {name} SUMMARY ---")
    print(f"  Total Reward:         {total_reward:.2f}")
    print(f"  EV Departures:        {violated_departures}/{total_departures} violated ({ev_viol_pct:.1f}%)")
    print(f"  C3 (building) viols:  {c3_violation_steps}/{step} steps ({c3_viol_pct:.1f}%)")
    print(f"  C4 (grid) viols:      {grid_violation_steps}/{step} steps ({c4_viol_pct:.1f}%)")
    print(f"  V2G: charge={pct(ev_charge_steps, total_ev_steps):.1f}% "
          f"discharge={pct(ev_discharge_steps, total_ev_steps):.1f}% "
          f"idle={pct(ev_idle_steps, total_ev_steps):.1f}%")

    # Per-building details
    print(f"\n  Per-building battery SoC:")
    for b in range(n_buildings):
        print(f"    Bld{b}: mean={batt_soc_stats[f'bld{b}_mean']:.3f} "
              f"[{batt_soc_stats[f'bld{b}_min']:.3f}, {batt_soc_stats[f'bld{b}_max']:.3f}]")

    print(f"\n  Per-building power (kW):")
    for b in range(n_buildings):
        print(f"    Bld{b}: max={bld_pwr_stats[f'bld{b}_max_kW']:.2f} "
              f"p95={bld_pwr_stats[f'bld{b}_p95_kW']:.2f} "
              f"viols={bld_pwr_stats[f'bld{b}_viols']}")

    print(f"\n  Per-EV charger actions:")
    for i, ei in enumerate(ev_idx):
        print(f"    EV{i} (act[{ei}]): mean={ev_action_stats[f'ev{i}_mean']:.3f} "
              f"std={ev_action_stats[f'ev{i}_std']:.3f} "
              f"charge={ev_action_stats[f'ev{i}_charge%']:.0f}% "
              f"discharge={ev_action_stats[f'ev{i}_discharge%']:.0f}%")

    if citylearn_kpis:
        print(f"\n  CityLearn KPIs:")
        for k, v in sorted(citylearn_kpis.items()):
            if not np.isnan(v):
                print(f"    {k}: {v:.4f}")

    return metrics


# =========================================================================
# Comparison table
# =========================================================================
def print_comparison_table(results):
    """Print a side-by-side comparison table."""
    W = 90
    print(f"\n\n{'#'*W}")
    print(f"  SIDE-BY-SIDE COMPARISON TABLE")
    print(f"{'#'*W}")

    names = [r["name"] for r in results]
    col_w = 18

    def row(label, key_or_fn, fmt=".2f"):
        vals = []
        for r in results:
            if callable(key_or_fn):
                v = key_or_fn(r)
            else:
                v = r.get(key_or_fn, float("nan"))
            vals.append(v)
        val_strs = [f"{v:{fmt}}" for v in vals]
        line = f"  {label:<35}"
        for vs in val_strs:
            line += f"{vs:>{col_w}}"
        print(line)

    # Header
    header = f"  {'Metric':<35}"
    for n in names:
        header += f"{n:>{col_w}}"
    print(header)
    print(f"  {'='*35}" + f"{'='*col_w}" * len(names))

    print(f"\n  --- Episode ---")
    row("Total Reward", "total_reward", ".1f")
    row("Mean Step Reward", "mean_step_reward", ".4f")
    row("Total CMDP Cost", "total_cost", ".1f")

    print(f"\n  --- EV Departures (C0) ---")
    row("Total Departures", "ev_departures_total", ".0f")
    row("Violated Departures", "ev_departures_violated", ".0f")
    row("Violation Rate (%)", "ev_departure_viol_pct", ".1f")
    row("Mean Deficit (violated)", "ev_mean_deficit_violated", ".4f")
    row("C0 Cumulative Cost", "C0_ev_departure_cost", ".1f")

    print(f"\n  --- Battery SoC (C2) ---")
    row("C2 Cumulative Cost", "C2_battery_soc_cost", ".1f")
    row("C2 Violation Steps (%)", "c2_viol_steps_pct", ".2f")
    row("Batt Clamp Count", "batt_clamp_count", ".0f")

    print(f"\n  --- Building Power (C3) ---")
    row("C3 Cumulative Cost", "C3_building_power_cost", ".1f")
    row("C3 Violation Steps (%)", "c3_viol_steps_pct", ".2f")

    print(f"\n  --- Grid Power (C4) ---")
    row("C4 Cumulative Cost", "C4_grid_power_cost", ".1f")
    row("C4 Violation Steps (%)", "c4_viol_steps_pct", ".2f")
    row("Grid Max Import (kW)", "grid_max_kW", ".2f")
    row("Grid P95 Import (kW)", "grid_p95_kW", ".2f")
    row("Grid Mean Import (kW)", "grid_mean_kW", ".2f")

    print(f"\n  --- V2G Behavior ---")
    row("Charge (%)", "v2g_charge_pct", ".1f")
    row("Discharge (%)", "v2g_discharge_pct", ".1f")
    row("Idle (%)", "v2g_idle_pct", ".1f")

    # Per-building battery
    n_bld = 5
    print(f"\n  --- Per-Building Battery SoC ---")
    for b in range(n_bld):
        row(f"Bld{b} Mean SoC", lambda r, b=b: r["batt_soc"].get(f"bld{b}_mean", 0), ".3f")
        row(f"Bld{b} [min, max]",
            lambda r, b=b: r["batt_soc"].get(f"bld{b}_min", 0), ".3f")

    # Per-building power
    print(f"\n  --- Per-Building Power (kW) ---")
    for b in range(n_bld):
        row(f"Bld{b} Max Power", lambda r, b=b: r["bld_power"].get(f"bld{b}_max_kW", 0), ".2f")
        row(f"Bld{b} Violations", lambda r, b=b: r["bld_power"].get(f"bld{b}_viols", 0), ".0f")

    # Per-EV
    print(f"\n  --- Per-EV Charger ---")
    for i in range(4):  # up to 4 EVs
        k_mean = f"ev{i}_mean"
        if k_mean in results[0].get("ev_actions", {}):
            row(f"EV{i} Mean Action",
                lambda r, i=i: r["ev_actions"].get(f"ev{i}_mean", 0), ".3f")
            row(f"EV{i} Charge%",
                lambda r, i=i: r["ev_actions"].get(f"ev{i}_charge%", 0), ".1f")
            row(f"EV{i} Discharge%",
                lambda r, i=i: r["ev_actions"].get(f"ev{i}_discharge%", 0), ".1f")

    # CityLearn KPIs
    print(f"\n  --- CityLearn KPIs ---")
    kpi_keys = [
        "citylearn_electricity_consumption_total",
        "citylearn_carbon_emissions_total",
        "citylearn_cost_total",
        "citylearn_daily_peak_average",
        "citylearn_all_time_peak_average",
        "citylearn_ramping_average",
        "citylearn_discomfort_proportion",
        "citylearn_zero_net_energy",
    ]
    for k in kpi_keys:
        short_k = k.replace("citylearn_", "")
        row(short_k, lambda r, k=k: r["citylearn_kpis"].get(k, float("nan")), ".4f")

    print(f"\n{'#'*W}")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    all_results = []
    for agent_def in AGENTS:
        result = evaluate_agent(agent_def, seed=42)
        all_results.append(result)

    print_comparison_table(all_results)

    # Save results
    out_path = os.path.join(PROJECT, "runs", "3way_ablation_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    serializable = []
    for r in all_results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, dict):
                sr[k] = {kk: (float(vv) if isinstance(vv, (float, int, np.floating, np.integer)) else vv)
                         for kk, vv in v.items()}
            elif isinstance(v, (np.floating, np.integer)):
                sr[k] = float(v)
            else:
                sr[k] = v
        serializable.append(sr)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
