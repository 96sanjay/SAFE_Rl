#!/usr/bin/env python3
"""
R18 Curriculum Evaluation -- Per-departure EV violation rate + all constraint metrics.

CRITICAL: This computes the REAL per-departure violation rate:
  violation_pct = violated_departures / total_departures * 100

NOT the misleading per-timestep metric (deficit_kWh / 8760).

Usage:
    python scripts/eval_r18_curriculum.py [--epoch EPOCH]
"""
import os
import sys
import json
import glob as glob_mod
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

EVAL_SEED = 42

# =====================================================================
# Eval env vars -- DISABLE training-only wrappers (Saute, PID, clamp)
# Keep reward config matching R18 for reward component analysis
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
    # === DISABLE training-only wrappers ===
    "CITYLEARN_EV_SAUTE": "0",
    "CITYLEARN_PID_LAGRANGE": "0",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_BATT_CLAMP": "0",
    "CITYLEARN_WM_DISABLE": "1",
    # R18 reward weights
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_GRID": "1.5",
    "STEMS_SG_THRESHOLD": "0.5",
    "STEMS_ALPHA_BUILD": "0.0",
    "STEMS_BETA_RAMP": "0.3",
    "STEMS_XI_RENEWABLE": "0.2",
    "STEMS_LAMBDA_EV": "5.0",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_ALPHA_LOAD_SHIFT": "6.0",
    "STEMS_ALPHA_GRID_MILD": "0.3",
    "STEMS_ALPHA_EV_GUARD": "5.0",
    "STEMS_ALPHA_V2G_CONTEXT": "3.0",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
}


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

RUN_DIR = "runs/r18_curriculum/5bld"


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


def evaluate(ckpt_path):
    print(f"\n{'='*60}")
    print(f"  R18 Curriculum Evaluation")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim = load_actor(ckpt_path)
    print(f"  Actor: obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  Obs normalizer: {'YES' if obs_mean is not None else 'NO'}")

    import citylearn_safe.schema_index as si
    si._CACHE = None
    set_env()

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
    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]

    need_pad = obs_dim > env.observation_space.shape[0]
    pad_dim = obs_dim - env.observation_space.shape[0] if need_pad else 0

    obs, _ = env.reset(seed=EVAL_SEED)
    done = False
    step = 0

    rewards, costs = [], []
    actions_all = []
    c0_vals, c1_vals, c2_vals, c3_vals, c4_vals = [], [], [], [], []

    # === CRITICAL: Per-departure EV tracking ===
    total_departures = 0
    violated_departures = 0
    departure_deficits = []
    ev_tracker = {}

    # C2: Battery SoC violations (per-step, per-building)
    soc_violation_steps = 0  # steps where ANY building has SoC violation
    soc_violation_count_total = 0  # total per-building violations across all steps

    # C3: Building power violations (per-step, per-building)
    building_power_violations = [0] * n_buildings
    building_power_steps = [0] * n_buildings
    c3_violation_steps = 0  # steps where ANY building exceeds limit

    # C4: Grid power violations (per-step)
    grid_violation_steps = 0

    # V2G tracking
    ev_charge_steps = 0
    ev_discharge_steps = 0
    ev_idle_steps = 0
    total_ev_steps = 0

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
        actions_all.append(action.copy())

        t_now = int(getattr(raw, "time_step", 0))
        t_idx = max(0, t_now - 1)

        # === EV departure tracking (per-departure, NOT per-timestep) ===
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

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cost = info.get("cost", 0.0)
        step += 1

        rewards.append(reward)
        costs.append(cost)

        # Per-constraint costs from info
        c0_vals.append(info.get("cost_ev_departure", 0.0) + info.get("cost_ev_dense", 0.0))
        c1_vals.append(info.get("cost_ev_dense", 0.0))
        c2_vals.append(info.get("cost_soc_pnorm", 0.0))
        c3_vals.append(info.get("cost_stems_building_power", 0.0))
        c4_vals.append(info.get("cost_stems_grid_power", 0.0))

        # Also grab the env's own violation counts from info dict
        # C0: ev departures (already tracked manually above)
        # Use info dict values as cross-check
        info_departures = info.get("ev_departure_departures", 0)
        info_deficit_count = info.get("ev_departure_violation_count_deficit", 0)

        # C2: Battery SoC bound violations
        soc_viol_any = info.get("battery_soc_violation_any", 0.0)
        soc_viol_cnt = info.get("battery_soc_violation_count", 0.0)
        if soc_viol_any > 0:
            soc_violation_steps += 1
        soc_violation_count_total += int(soc_viol_cnt)

        # C3: Building power violations (from info dict)
        bld_viol = info.get("building_power_violation", 0.0)
        bld_viol_cnt = int(info.get("building_power_violation_count", 0.0))
        if bld_viol > 0:
            c3_violation_steps += 1
        for b_idx in range(n_buildings):
            building_power_steps[b_idx] += 1
        # Track per-building from environment directly
        for b_idx, bld in enumerate(buildings):
            try:
                nec = getattr(bld, 'net_electricity_consumption', None)
                if nec is not None and len(nec) > t_idx:
                    p = abs(float(nec[t_idx]))
                    limit = 4.6083
                    if p > limit:
                        building_power_violations[b_idx] += 1
            except Exception:
                pass

        # C4: Grid power violations
        grid_viol = info.get("grid_power_violation", 0.0)
        if grid_viol > 0:
            grid_violation_steps += 1

    # === Compute metrics ===
    total_reward = sum(rewards)
    total_cost = sum(costs)
    total_c0 = sum(c0_vals)
    total_c1 = sum(c1_vals)
    total_c2 = sum(c2_vals)
    total_c3 = sum(c3_vals)
    total_c4 = sum(c4_vals)

    ev_departure_violation_pct = (100.0 * violated_departures / total_departures
                                  if total_departures > 0 else 0.0)
    mean_deficit = np.mean(departure_deficits) if departure_deficits else 0.0

    bld_violations_total = sum(building_power_violations)
    bld_steps_total = sum(building_power_steps)
    bld_violation_pct = 100.0 * bld_violations_total / bld_steps_total if bld_steps_total > 0 else 0.0

    v2g_charge_pct = 100.0 * ev_charge_steps / total_ev_steps if total_ev_steps > 0 else 0.0
    v2g_discharge_pct = 100.0 * ev_discharge_steps / total_ev_steps if total_ev_steps > 0 else 0.0
    v2g_idle_pct = 100.0 * ev_idle_steps / total_ev_steps if total_ev_steps > 0 else 0.0

    actions_arr = np.array(actions_all)
    batt_actions = actions_arr[:, batt_idx] if batt_idx else np.array([])
    ev_actions = actions_arr[:, ev_idx] if ev_idx else np.array([])

    # === Print Results ===
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")

    print(f"\n--- Episode Summary ---")
    print(f"  Steps:        {step}")
    print(f"  Total Reward:  {total_reward:.1f}")
    print(f"  Total Cost:    {total_cost:.1f}")

    print(f"\n--- EV Departure Violations (THE REAL METRIC) ---")
    print(f"  Total departures:    {total_departures}")
    print(f"  Violated departures: {violated_departures}")
    print(f"  Violation rate:      {ev_departure_violation_pct:.2f}%"
          f"  ({violated_departures}/{total_departures})")
    print(f"  Mean deficit (all):  {mean_deficit:.4f}")
    if departure_deficits:
        violated_deficits = [d for d in departure_deficits if d > 0.01]
        if violated_deficits:
            print(f"  Mean deficit (violated only): {np.mean(violated_deficits):.4f}")
            print(f"  Max deficit:  {max(violated_deficits):.4f}")

    print(f"\n--- Per-Constraint Cumulative Costs ---")
    print(f"  C0 (EV departure + dense): {total_c0:.1f}")
    print(f"  C1 (EV dense):             {total_c1:.4f}")
    print(f"  C2 (SoC pnorm):            {total_c2:.1f}")
    print(f"  C3 (Building power):       {total_c3:.1f}")
    print(f"  C4 (Grid power):           {total_c4:.1f}")

    print(f"\n--- C2: Battery SoC Bound Violations ---")
    c2_viol_pct = 100.0 * soc_violation_steps / step if step > 0 else 0.0
    print(f"  Steps with ANY SoC violation: {soc_violation_steps}/{step} ({c2_viol_pct:.1f}%)")
    print(f"  Total per-building SoC violations: {soc_violation_count_total}")

    print(f"\n--- C3: Building Power Violations ---")
    c3_viol_pct = 100.0 * c3_violation_steps / step if step > 0 else 0.0
    print(f"  Steps with ANY building violation: {c3_violation_steps}/{step} ({c3_viol_pct:.1f}%)")
    print(f"  Per-building breakdown:")
    for b_idx in range(n_buildings):
        pct = (100.0 * building_power_violations[b_idx] / building_power_steps[b_idx]
               if building_power_steps[b_idx] > 0 else 0.0)
        print(f"    Building {b_idx}: {building_power_violations[b_idx]}/{building_power_steps[b_idx]} ({pct:.1f}%)")

    print(f"\n--- C4: Grid Power Violations ---")
    c4_viol_pct = 100.0 * grid_violation_steps / step if step > 0 else 0.0
    print(f"  Steps with grid violation: {grid_violation_steps}/{step} ({c4_viol_pct:.1f}%)")

    print(f"\n--- V2G Behavior ---")
    print(f"  EV actions total: {total_ev_steps}")
    print(f"  Charging:    {ev_charge_steps} ({v2g_charge_pct:.1f}%)")
    print(f"  Discharging: {ev_discharge_steps} ({v2g_discharge_pct:.1f}%)")
    print(f"  Idle:        {ev_idle_steps} ({v2g_idle_pct:.1f}%)")

    print(f"\n--- Action Statistics ---")
    if len(batt_actions) > 0:
        print(f"  Battery: mean={batt_actions.mean():.3f}, std={batt_actions.std():.3f}, "
              f"min={batt_actions.min():.3f}, max={batt_actions.max():.3f}")
    if len(ev_actions) > 0:
        print(f"  EV:      mean={ev_actions.mean():.3f}, std={ev_actions.std():.3f}, "
              f"min={ev_actions.min():.3f}, max={ev_actions.max():.3f}")

    print(f"\n{'='*60}")
    print(f"  ALL CONSTRAINT VIOLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  C0 EV departure:  {violated_departures}/{total_departures} = {ev_departure_violation_pct:.2f}%"
          f"  (deficit_count / departures)")
    print(f"  C2 SoC bounds:    {soc_violation_steps}/{step} = {c2_viol_pct:.1f}%"
          f"  (steps with any violation / total steps)")
    print(f"  C3 Building pwr:  {c3_violation_steps}/{step} = {c3_viol_pct:.1f}%"
          f"  (steps with any violation / total steps)")
    print(f"  C4 Grid pwr:      {grid_violation_steps}/{step} = {c4_viol_pct:.1f}%"
          f"  (steps with violation / total steps)")
    print(f"  V2G discharge:    {ev_discharge_steps}/{total_ev_steps} = {v2g_discharge_pct:.1f}%")
    print(f"  Reward:           {total_reward:.1f}")
    print(f"{'='*60}")

    return {
        "total_reward": total_reward,
        "total_departures": total_departures,
        "violated_departures": violated_departures,
        "ev_departure_violation_pct": ev_departure_violation_pct,
        "mean_deficit": mean_deficit,
        "total_c0": total_c0,
        "total_c1": total_c1,
        "total_c2": total_c2,
        "total_c3": total_c3,
        "total_c4": total_c4,
        "c2_violation_pct": c2_viol_pct,
        "c3_violation_pct": c3_viol_pct,
        "c4_violation_pct": c4_viol_pct,
        "v2g_charge_pct": v2g_charge_pct,
        "v2g_discharge_pct": v2g_discharge_pct,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=None,
                    help="Specific epoch to evaluate (default: latest)")
    args = ap.parse_args()

    ckpt = find_checkpoint(epoch=args.epoch)
    if not ckpt:
        print("ERROR: No checkpoint found!")
        sys.exit(1)

    print(f"Using checkpoint: {ckpt}")
    evaluate(ckpt)
