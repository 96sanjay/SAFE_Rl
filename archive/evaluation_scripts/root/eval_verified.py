#!/usr/bin/env python3
"""
Bulletproof Evaluation Script for PPO and SAC Checkpoints
==========================================================
Produces reproducible, deterministic results for fair comparison.

Usage:
    python eval_verified.py --checkpoint <path> --schema 1bld --months 3 --algo ppo
    python eval_verified.py --checkpoint <path> --schema 5bld --months 12 --algo sac

Features:
    - Fixed seeds for numpy, torch, random, and env
    - Weight shape verification on load
    - All metrics computed in one pass
    - Determinism check: runs twice, verifies identical results
    - Works for both PPO and SAC checkpoints
    - Auto-detects 1-building vs 5-building from env
"""
import os
import sys
import json
import argparse
import random
import hashlib
from collections import defaultdict

# =====================================================================
# ALL env vars set BEFORE any project imports
# =====================================================================
def set_all_env_vars(schema_path: str):
    """Set EVERY env var explicitly. No defaults, no missing."""
    # First, clear all CITYLEARN/STEMS/COST vars to avoid stale state
    for k in list(os.environ.keys()):
        if k.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(k, None)

    env_vars = {
        # --- Schema ---
        "CITYLEARN_SCHEMA": schema_path,
        "CITYLEARN_CENTRAL_AGENT": "1",
        "CITYLEARN_REWARD_TYPE": "stems",
        "CITYLEARN_EXPORT_FACTOR": "0.7",

        # --- Thresholds ---
        "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
        "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
        "CITYLEARN_STEMS_SOC_LOW": "0.0",
        "CITYLEARN_STEMS_SOC_HIGH": "0.95",
        "CITYLEARN_STEMS_PNORM_P": "4.0",

        # --- Action mask ---
        "CITYLEARN_ACTION_MASK": "1",

        # --- KL regularization (no effect at eval, but set for consistency) ---
        "CITYLEARN_KL_BETA": "0.1",
        "CITYLEARN_KL_BETA_DECAY": "0.995",

        # --- Active reward terms ---
        "STEMS_ALPHA_NEC_SIGN": "3.0",
        "STEMS_ALPHA_PRICE_ARB": "1.0",
        "STEMS_LAMBDA_EV": "15.0",
        "STEMS_EV_SLACK_ARB_SCALE": "2.5",
        "STEMS_ALPHA_GRID_PENALTY": "1.5",
        "CITYLEARN_STEMS_BATTERY_COST_SCALE": "1.0",

        # --- All disabled reward terms ---
        "STEMS_MU_ECONOMIC": "0.0",
        "STEMS_ALPHA_GRID": "0.0",
        "STEMS_ALPHA_BUILD": "0.0",
        "STEMS_BETA_RAMP": "0.0",
        "STEMS_XI_RENEWABLE": "0.0",
        "STEMS_ALPHA_LOAD_SHIFT": "0.0",
        "STEMS_ALPHA_GRID_MILD": "0.0",
        "STEMS_ALPHA_EV_GUARD": "0.0",
        "STEMS_ALPHA_V2G_CONTEXT": "0.0",
        "STEMS_ALPHA_PEAK_SHAVE": "0.0",
        "STEMS_ALPHA_BARRIER": "0.0",
        "STEMS_ALPHA_EV_SOLAR": "0.0",
        "STEMS_ALPHA_SOLAR_STORE": "0.0",
        "STEMS_SOLAR_STORE_BATT_ONLY": "0",
        "STEMS_ALPHA_HEADROOM": "0.0",
        "STEMS_SB_ASYMMETRIC": "1",
        "STEMS_SG_EXPORT_CREDIT": "0.5",
        "STEMS_SG_THRESHOLD": "0.5",

        # --- Saute MDP for C0 ---
        "CITYLEARN_EV_SAUTE": "0",
        "CITYLEARN_EV_DENSE_COST_SCALE": "1.0",
        "CITYLEARN_EV_SAUTE_BUDGET": "25000",
        "CITYLEARN_EV_SAUTE_PENALTY": "5.0",
        "CITYLEARN_EV_SAUTE_GAMMA": "1.0",
        "CITYLEARN_EV_SAUTE_SHAPED_ALPHA": "2.0",

        # --- PID Lagrangian ---
        "CITYLEARN_PID_LAGRANGE": "1",

        # --- Cost weights ---
        "COST_W_C2": "0.0",
        "COST_W_C3": "5.0",
        "CITYLEARN_W_COST_EV": "1.0",
        "CITYLEARN_W_COST_SOC": "10.0",
        "CITYLEARN_W_COST_BUILDING": "0.5",
        "CITYLEARN_W_COST_GRID": "0.05",
        "CITYLEARN_EV_COST_SCALE": "3.0",
        "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
        "CITYLEARN_INCLUDE_EV_COST": "1",

        # --- Feature flags ---
        "CITYLEARN_WM_DISABLE": "1",
        "CITYLEARN_EV_ACTION_CLAMP": "0",
        "CITYLEARN_BATT_CLAMP": "0",
        "CITYLEARN_SPATIAL_OBS": "0",
        "CITYLEARN_TEMPORAL_WINDOW": "0",
        "CITYLEARN_C3_CONTROLLABLE": "1",
    }

    for k, v in env_vars.items():
        os.environ[k] = v


# Set up paths before imports
PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# Actor Models
# =====================================================================
class PPOActor(nn.Module):
    """PPO/PPOLag actor: mean network with tanh output.

    Checkpoint key format: mean.0.weight, mean.2.weight, mean.4.weight
    """
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


class SACActor(nn.Module):
    """SAC/SACLag actor: outputs [mean, log_std] concatenated.

    For deterministic eval, we use only the mean (first act_dim outputs).
    Checkpoint key format: net.0.weight, net.2.weight, net.4.weight
    Output dim = 2 * act_dim (mean + log_std).
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())  # SAC typically uses ReLU
            in_dim = h
        # Output is 2 * act_dim: [mean, log_std]
        layers.append(nn.Linear(in_dim, 2 * act_dim))
        self.net = nn.Sequential(*layers)
        self.act_dim = act_dim

    def forward(self, obs):
        out = self.net(obs)
        mean = out[..., :self.act_dim]
        return torch.tanh(mean)


# =====================================================================
# Checkpoint Loading
# =====================================================================
def load_checkpoint(ckpt_path: str, algo: str):
    """Load actor and obs normalizer from checkpoint.

    Returns: (actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim, info_dict)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    top_keys = list(ckpt.keys())
    print(f"  Checkpoint top-level keys: {top_keys}")

    if "pi" not in ckpt:
        raise KeyError(f"Checkpoint missing 'pi' key. Found: {top_keys}")

    pi_state = ckpt["pi"]
    pi_keys = sorted(pi_state.keys())
    print(f"  Policy state keys: {pi_keys}")

    info = {}

    if algo == "ppo":
        # PPO format: mean.0.weight, mean.2.weight, mean.4.weight
        if "mean.0.weight" not in pi_state:
            raise KeyError(
                f"PPO checkpoint expected 'mean.0.weight' but found keys: {pi_keys}. "
                f"Try --algo sac if this is a SAC checkpoint."
            )
        h1 = pi_state["mean.0.weight"].shape[0]
        h2 = pi_state["mean.2.weight"].shape[0]
        obs_dim = pi_state["mean.0.weight"].shape[1]
        act_dim = pi_state["mean.4.weight"].shape[0]

        actor = PPOActor(obs_dim, act_dim, (h1, h2))
        filtered = {k: v for k, v in pi_state.items() if not k.startswith("log_std")}
        result = actor.load_state_dict(filtered, strict=False)
        if result.missing_keys:
            raise RuntimeError(f"Missing keys after load: {result.missing_keys}")

        log_std = pi_state.get("log_std", None)
        if log_std is not None:
            info["log_std"] = log_std.numpy()
            info["std"] = log_std.exp().numpy()

    elif algo == "sac":
        # SAC format: net.0.weight, net.2.weight, net.4.weight
        if "net.0.weight" not in pi_state:
            raise KeyError(
                f"SAC checkpoint expected 'net.0.weight' but found keys: {pi_keys}. "
                f"Try --algo ppo if this is a PPO checkpoint."
            )
        h1 = pi_state["net.0.weight"].shape[0]
        h2 = pi_state["net.2.weight"].shape[0]
        obs_dim = pi_state["net.0.weight"].shape[1]
        output_dim = pi_state["net.4.weight"].shape[0]
        act_dim = output_dim // 2  # SAC outputs [mean, log_std]

        actor = SACActor(obs_dim, act_dim, (h1, h2))
        # Filter out _log2 (SAC entropy coefficient)
        filtered = {k: v for k, v in pi_state.items() if k.startswith("net.")}
        result = actor.load_state_dict(filtered, strict=False)
        if result.missing_keys:
            raise RuntimeError(f"Missing keys after load: {result.missing_keys}")

        info["output_dim"] = output_dim
        info["_log2"] = float(pi_state.get("_log2", 0.0))
    else:
        raise ValueError(f"Unknown algo: {algo}. Use 'ppo' or 'sac'.")

    actor.eval()

    # --- Verify shapes ---
    print(f"\n  === Weight Shape Verification ===")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, hidden=[{h1},{h2}]")
    for name, param in actor.named_parameters():
        print(f"    {name}: {list(param.shape)}")

    # --- Obs normalizer ---
    obs_mean = obs_std = obs_clip = None
    if "obs_normalizer" in ckpt:
        norm = ckpt["obs_normalizer"]
        obs_mean = torch.as_tensor(norm["_mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(norm["_std"], dtype=torch.float32)
        clip_t = norm["_clip"].float()
        obs_clip = float(clip_t.mean())
        norm_obs_dim = obs_mean.shape[0]
        print(f"  Obs normalizer: YES (dim={norm_obs_dim}, clip={obs_clip:.1f})")
        if norm_obs_dim != obs_dim:
            print(f"  WARNING: Normalizer dim ({norm_obs_dim}) != model obs_dim ({obs_dim})")
    else:
        print(f"  Obs normalizer: NO")

    if info.get("log_std") is not None:
        print(f"  Policy log_std: {info['log_std']}")
        print(f"  Policy std:     {info['std']}")
    if "output_dim" in info:
        print(f"  SAC output_dim: {info['output_dim']} (act_dim={act_dim})")

    return actor, obs_mean, obs_std, obs_clip, obs_dim, act_dim, info


# =====================================================================
# Environment Setup
# =====================================================================
def build_env(schema_path: str):
    """Build the full env wrapper chain: base -> safety -> forecast -> action_mask."""
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.action_mask_wrapper import ActionMaskWrapper
    from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    forecast = ForecastObsWrapper(safety, forecast_horizon=24)
    env = ActionMaskWrapper(forecast)

    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(getattr(raw, "buildings", []))

    # Discover action layout
    names_raw = getattr(raw, "action_names", [])
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        names = names_raw[0]
    else:
        names = list(names_raw)

    batt_idx = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
    ev_idx = [i for i, n in enumerate(names)
              if "electric_vehicle_storage_charger_" in str(n).lower()]

    return env, raw, buildings, names, batt_idx, ev_idx


# =====================================================================
# Single Rollout
# =====================================================================
def run_episode(env, raw, buildings, names, batt_idx, ev_idx,
                actor, obs_mean, obs_std, obs_clip,
                model_obs_dim, act_dim, seed):
    """Run one full episode. Returns a dict of all metrics."""
    n_buildings = len(buildings)
    env_obs_dim = env.observation_space.shape[0]

    need_pad = model_obs_dim > env_obs_dim
    pad_dim = model_obs_dim - env_obs_dim if need_pad else 0
    need_trim = model_obs_dim < env_obs_dim

    P_BUILDING_MAX = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
    P_GRID_MAX = float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "10.2352"))
    SOC_HIGH = float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95"))
    SOC_LOW = float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0"))

    # --- Reset ---
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    # --- Tracking ---
    rewards = []
    costs = []
    actions_all = []
    c0_vals, c2_vals, c3_vals, c4_vals = [], [], [], []

    batt_actions_by_hour = [[] for _ in range(24)]
    batt_soc_all = []
    ev_actions_by_hour = [[] for _ in range(24)]
    ev_connected_mask = []
    ev_soc_all = []
    nec_per_building = [[] for _ in range(n_buildings)]
    grid_nec_all = []
    price_all = []
    hour_all = []

    # C3/C4 tracking
    c3_violations = 0
    c3_total_checks = 0
    c4_violations = 0
    c4_total_checks = 0

    # Battery SoC constraint tracking (C2)
    c2_soc_above_high = 0
    c2_soc_below_low = 0
    c2_total_checks = 0

    # EV departure tracking
    total_departures = 0
    violated_departures = 0
    departure_socs = []
    departure_required_socs = []
    departure_deficits = []
    ev_tracker = {}

    # Energy tracking
    batt_charge_kwh = 0.0
    batt_discharge_kwh = 0.0
    ev_charge_kwh = 0.0
    ev_v2g_kwh = 0.0

    # --- Rollout ---
    while not done:
        obs_np = np.asarray(obs, dtype=np.float32).ravel()
        if need_pad:
            obs_np = np.concatenate([obs_np, np.zeros(pad_dim, dtype=np.float32)])
        elif need_trim:
            obs_np = obs_np[:model_obs_dim]

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
        t_idx = max(0, t_now - 1)
        hour = t_now % 24
        hour_all.append(hour)

        # --- Price ---
        try:
            pr = buildings[0].pricing.electricity_pricing
            price = float(pr[t_idx]) if t_idx < len(pr) else 0.0
        except Exception:
            price = 0.0
        price_all.append(price)

        # --- Battery SoC (BEFORE step) ---
        batt_soc_step = []
        for b_idx, bld in enumerate(buildings):
            try:
                es = getattr(bld, "electrical_storage", None)
                soc_arr = getattr(es, "soc", None) if es else None
                if soc_arr is not None and len(soc_arr) > t_idx:
                    s = float(np.clip(soc_arr[t_idx], 0, 1))
                else:
                    s = 0.5
            except Exception:
                s = 0.5
            batt_soc_step.append(s)
            # C2 check
            c2_total_checks += 1
            if s > SOC_HIGH:
                c2_soc_above_high += 1
            if s < SOC_LOW:
                c2_soc_below_low += 1
        batt_soc_all.append(batt_soc_step)

        # Battery actions by hour
        for bi in batt_idx:
            if bi < len(action):
                batt_actions_by_hour[hour].append(float(action[bi]))

        # --- EV tracking ---
        ev_soc_step = []
        ev_connected_step = []
        for b_idx, bld in enumerate(buildings):
            chargers = getattr(bld, "electric_vehicle_chargers", None) or []
            for ch_idx, ch in enumerate(chargers):
                key = (b_idx, ch_idx)
                sim = getattr(ch, 'charger_simulation',
                              getattr(ch, '_Charger__charger_simulation', None))
                if sim is None:
                    ev_connected_step.append(False)
                    ev_soc_step.append(0.0)
                    continue
                try:
                    sa = np.asarray(
                        getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                    current_connected = (t_now < len(sa) and float(sa[t_now]) == 1.0)
                    ev_connected_step.append(current_connected)

                    ev_soc = 0.0
                    if current_connected:
                        ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                        if ev_obj is not None:
                            bt = getattr(ev_obj, 'battery', None)
                            if bt is not None:
                                soc_arr = getattr(bt, 'soc', None)
                                if soc_arr is not None:
                                    sn = np.asarray(soc_arr, dtype=float)
                                    if 0 <= t_idx < len(sn):
                                        ev_soc = float(np.clip(sn[t_idx], 0, 1))
                    ev_soc_step.append(ev_soc)

                    # Departure tracking
                    ra = np.asarray(
                        getattr(sim, '_electric_vehicle_required_soc_departure'),
                        dtype=float)
                    if current_connected:
                        rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                        if not np.isfinite(rs):
                            rs = 1.0
                        ev_tracker[key] = {
                            'was_connected': True,
                            'last_soc': ev_soc,
                            'required_soc': rs,
                        }
                    else:
                        prev = ev_tracker.get(key, {})
                        if prev.get('was_connected', False):
                            total_departures += 1
                            last_soc = prev['last_soc']
                            rs = prev['required_soc']
                            deficit = max(0.0, rs - last_soc)
                            departure_socs.append(last_soc)
                            departure_required_socs.append(rs)
                            departure_deficits.append(deficit)
                            if deficit > 0.01:
                                violated_departures += 1
                        ev_tracker[key] = {'was_connected': False}
                except Exception:
                    ev_connected_step.append(False)
                    ev_soc_step.append(0.0)

        ev_connected_mask.append(ev_connected_step)
        ev_soc_all.append(ev_soc_step)

        # EV actions by hour (connected only)
        ev_charger_i = 0
        for ei in ev_idx:
            if ei < len(action):
                if (ev_charger_i < len(ev_connected_step)
                        and ev_connected_step[ev_charger_i]):
                    ev_actions_by_hour[hour].append(float(action[ei]))
                ev_charger_i += 1

        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        rewards.append(reward)
        costs.append(info.get("cost", 0.0))
        c0_vals.append(info.get("cost_ev_departure", 0.0))
        c2_vals.append(info.get("cost_stems_battery", 0.0))
        c3_vals.append(info.get("cost_stems_building_power", 0.0))
        c4_vals.append(info.get("cost_stems_grid_power", 0.0))

        # --- NEC per building (after step) ---
        t_after = max(0, int(getattr(raw, "time_step", 0)) - 1)
        total_nec_step = 0.0
        for b_idx, bld in enumerate(buildings):
            try:
                nec = getattr(bld, 'net_electricity_consumption', None)
                if nec is not None and len(nec) > t_after:
                    p = float(nec[t_after])
                else:
                    p = 0.0
            except Exception:
                p = 0.0
            nec_per_building[b_idx].append(p)
            total_nec_step += p

            # C3 check per building
            c3_total_checks += 1
            if abs(p) > P_BUILDING_MAX:
                c3_violations += 1

        grid_nec_all.append(total_nec_step)

        # C4 check (grid)
        c4_total_checks += 1
        if abs(total_nec_step) > P_GRID_MAX:
            c4_violations += 1

        # Energy tracking from NEC
        if total_nec_step > 0:
            pass  # import
        # Battery energy: approximate from NEC changes
        for b_idx, bld in enumerate(buildings):
            try:
                es = getattr(bld, "electrical_storage", None)
                if es is not None:
                    edc = getattr(es, 'electricity_consumption', None)
                    if edc is not None and len(edc) > t_after:
                        ec = float(edc[t_after])
                        if ec > 0:
                            batt_charge_kwh += ec
                        else:
                            batt_discharge_kwh += abs(ec)
            except Exception:
                pass

        if step % 2000 == 0:
            sys.stdout.write(f"\r    Step {step}...")
            sys.stdout.flush()

    sys.stdout.write(f"\r    Rollout complete: {step} steps\n")

    # =====================================================================
    # COMPUTE ALL METRICS
    # =====================================================================
    actions_arr = np.array(actions_all)  # (T, act_dim)
    metrics = {}

    # --- Episode summary ---
    metrics["steps"] = step
    metrics["total_reward"] = float(sum(rewards))
    metrics["mean_reward"] = float(np.mean(rewards))
    metrics["total_cost"] = float(sum(costs))
    metrics["c0_cost"] = float(sum(c0_vals))
    metrics["c2_cost"] = float(sum(c2_vals))
    metrics["c3_cost"] = float(sum(c3_vals))
    metrics["c4_cost"] = float(sum(c4_vals))

    # --- Battery metrics ---
    batt_actions = actions_arr[:, batt_idx] if batt_idx else np.array([])
    if batt_actions.size > 0:
        bf = batt_actions.ravel()
        n_bf = len(bf)
        metrics["batt_charge_count"] = int(np.sum(bf > 0.1))
        metrics["batt_discharge_count"] = int(np.sum(bf < -0.1))
        metrics["batt_idle_count"] = int(np.sum(np.abs(bf) <= 0.1))
        metrics["batt_charge_pct"] = 100.0 * metrics["batt_charge_count"] / n_bf
        metrics["batt_discharge_pct"] = 100.0 * metrics["batt_discharge_count"] / n_bf
        metrics["batt_idle_pct"] = 100.0 * metrics["batt_idle_count"] / n_bf

        # Solar (10-15) and peak (17-21) and off-peak (0-6)
        solar_hours = set(range(10, 16))
        peak_hours = set(range(17, 22))
        offpeak_hours = set(range(0, 7))
        solar_batt = [a for h, acts in enumerate(batt_actions_by_hour)
                      if h in solar_hours for a in acts]
        peak_batt = [a for h, acts in enumerate(batt_actions_by_hour)
                     if h in peak_hours for a in acts]
        offpeak_batt = [a for h, acts in enumerate(batt_actions_by_hour)
                        if h in offpeak_hours for a in acts]
        metrics["batt_solar_avg"] = float(np.mean(solar_batt)) if solar_batt else 0.0
        metrics["batt_peak_avg"] = float(np.mean(peak_batt)) if peak_batt else 0.0
        metrics["batt_offpeak_avg"] = float(np.mean(offpeak_batt)) if offpeak_batt else 0.0

        # Price correlation
        if len(price_all) > 10:
            batt_avg_step = (batt_actions.mean(axis=1) if batt_actions.ndim > 1
                             else batt_actions.ravel())
            prices_np = np.array(price_all[:len(batt_avg_step)])
            valid = np.isfinite(prices_np) & np.isfinite(batt_avg_step)
            if valid.sum() > 10:
                metrics["batt_price_corr"] = float(
                    np.corrcoef(prices_np[valid], batt_avg_step[valid])[0, 1])
            else:
                metrics["batt_price_corr"] = 0.0
        else:
            metrics["batt_price_corr"] = 0.0

        # Battery SoC stats
        batt_soc_arr = np.array(batt_soc_all)  # (T, n_buildings)
        metrics["batt_soc_min"] = float(batt_soc_arr.min())
        metrics["batt_soc_max"] = float(batt_soc_arr.max())
        metrics["batt_soc_mean"] = float(batt_soc_arr.mean())
        metrics["batt_soc_std"] = float(batt_soc_arr.std())

        # Per-building SoC
        for b in range(n_buildings):
            col = batt_soc_arr[:, b]
            metrics[f"batt_soc_b{b}_min"] = float(col.min())
            metrics[f"batt_soc_b{b}_max"] = float(col.max())
            metrics[f"batt_soc_b{b}_mean"] = float(col.mean())
            metrics[f"batt_soc_b{b}_range"] = float(col.max() - col.min())

        # Daily cycling: count days with both charge and discharge
        n_days = step // 24
        cycling_days = 0
        for d in range(n_days):
            day_slice = bf[d * n_buildings * 24:(d + 1) * n_buildings * 24]
            if day_slice.size > 0 and np.any(day_slice > 0.1) and np.any(day_slice < -0.1):
                cycling_days += 1
        metrics["batt_cycling_days"] = cycling_days
        metrics["batt_cycling_days_total"] = n_days

        metrics["batt_action_mean"] = float(bf.mean())
        metrics["batt_action_std"] = float(bf.std())
        metrics["batt_action_min"] = float(bf.min())
        metrics["batt_action_max"] = float(bf.max())
    else:
        for k in ["batt_charge_count", "batt_discharge_count", "batt_idle_count",
                   "batt_charge_pct", "batt_discharge_pct", "batt_idle_pct",
                   "batt_solar_avg", "batt_peak_avg", "batt_offpeak_avg",
                   "batt_price_corr", "batt_soc_min", "batt_soc_max",
                   "batt_soc_mean", "batt_soc_std", "batt_cycling_days",
                   "batt_cycling_days_total", "batt_action_mean", "batt_action_std",
                   "batt_action_min", "batt_action_max"]:
            metrics[k] = 0.0

    # --- EV metrics ---
    ev_connected_total = 0
    ev_charge_connected = 0
    ev_discharge_connected = 0
    ev_idle_connected = 0
    ev_peak_connected = 0
    ev_peak_discharge = 0
    ev_connected_actions = []

    ev_actions = actions_arr[:, ev_idx] if ev_idx else np.array([])
    if ev_actions.size > 0:
        for t in range(len(actions_all)):
            h = hour_all[t]
            for ei_local, ei in enumerate(ev_idx):
                if (ei_local < len(ev_connected_mask[t])
                        and ev_connected_mask[t][ei_local]):
                    a = float(actions_all[t][ei])
                    ev_connected_total += 1
                    ev_connected_actions.append(a)
                    if a > 0.1:
                        ev_charge_connected += 1
                    elif a < -0.1:
                        ev_discharge_connected += 1
                    else:
                        ev_idle_connected += 1
                    if 17 <= h <= 23:
                        ev_peak_connected += 1
                        if a < -0.1:
                            ev_peak_discharge += 1

    metrics["ev_connected_steps"] = ev_connected_total
    metrics["ev_charge_count"] = ev_charge_connected
    metrics["ev_discharge_count"] = ev_discharge_connected
    metrics["ev_idle_count"] = ev_idle_connected
    pct_ = lambda a, b: 100.0 * a / b if b > 0 else 0.0
    metrics["ev_charge_pct"] = pct_(ev_charge_connected, ev_connected_total)
    metrics["ev_discharge_pct"] = pct_(ev_discharge_connected, ev_connected_total)
    metrics["ev_idle_pct"] = pct_(ev_idle_connected, ev_connected_total)
    metrics["ev_peak_connected"] = ev_peak_connected
    metrics["ev_peak_discharge"] = ev_peak_discharge
    metrics["ev_v2g_peak_pct"] = pct_(ev_peak_discharge, ev_peak_connected)
    metrics["ev_mean_action_connected"] = (
        float(np.mean(ev_connected_actions)) if ev_connected_actions else 0.0)

    # --- C0 (EV departure) ---
    metrics["c0_total_departures"] = total_departures
    metrics["c0_violated_departures"] = violated_departures
    metrics["c0_violation_pct"] = pct_(violated_departures, total_departures)
    metrics["c0_mean_soc_at_departure"] = (
        float(np.mean(departure_socs)) if departure_socs else 0.0)
    metrics["c0_mean_required_soc"] = (
        float(np.mean(departure_required_socs)) if departure_required_socs else 0.0)

    # Mean deficit at VIOLATED departures only
    violated_defs = [d for d in departure_deficits if d > 0.01]
    metrics["c0_mean_deficit_violated"] = (
        float(np.mean(violated_defs)) if violated_defs else 0.0)

    # SoC distribution at departure
    if departure_socs:
        dep_arr = np.array(departure_socs)
        for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
            count = int(np.sum((dep_arr >= lo) & (dep_arr < hi)))
            metrics[f"c0_dep_soc_{lo:.1f}_{hi:.1f}"] = count
    else:
        for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
            metrics[f"c0_dep_soc_{lo:.1f}_{hi:.1f}"] = 0

    # --- C2 (battery SoC bounds) ---
    metrics["c2_above_high_count"] = c2_soc_above_high
    metrics["c2_above_high_pct"] = pct_(c2_soc_above_high, c2_total_checks)
    metrics["c2_below_low_count"] = c2_soc_below_low
    metrics["c2_below_low_pct"] = pct_(c2_soc_below_low, c2_total_checks)

    # --- C3 (building power) ---
    metrics["c3_violations"] = c3_violations
    metrics["c3_total_checks"] = c3_total_checks
    metrics["c3_violation_pct"] = pct_(c3_violations, c3_total_checks)
    all_nec_flat = [abs(v) for necs in nec_per_building for v in necs]
    if all_nec_flat:
        metrics["c3_mean_abs_nec"] = float(np.mean(all_nec_flat))
        metrics["c3_max_abs_nec"] = float(np.max(all_nec_flat))
        metrics["c3_p95_abs_nec"] = float(np.percentile(all_nec_flat, 95))
    else:
        metrics["c3_mean_abs_nec"] = metrics["c3_max_abs_nec"] = metrics["c3_p95_abs_nec"] = 0.0

    # --- C4 (grid power) ---
    metrics["c4_violations"] = c4_violations
    metrics["c4_total_checks"] = c4_total_checks
    metrics["c4_violation_pct"] = pct_(c4_violations, c4_total_checks)
    grid_abs = [abs(v) for v in grid_nec_all]
    if grid_abs:
        metrics["c4_mean_abs_nec"] = float(np.mean(grid_abs))
        metrics["c4_max_abs_nec"] = float(np.max(grid_abs))
        metrics["c4_p95_abs_nec"] = float(np.percentile(grid_abs, 95))
    else:
        metrics["c4_mean_abs_nec"] = metrics["c4_max_abs_nec"] = metrics["c4_p95_abs_nec"] = 0.0

    # --- Energy ---
    total_import = sum(max(0, nec) for necs in nec_per_building for nec in necs)
    total_export = sum(abs(min(0, nec)) for necs in nec_per_building for nec in necs)
    metrics["energy_import_kwh"] = float(total_import)
    metrics["energy_export_kwh"] = float(total_export)
    metrics["energy_batt_charge_kwh"] = float(batt_charge_kwh)
    metrics["energy_batt_discharge_kwh"] = float(batt_discharge_kwh)

    # Fingerprint for determinism check: hash of all actions
    action_bytes = actions_arr.tobytes()
    metrics["_action_hash"] = hashlib.sha256(action_bytes).hexdigest()[:16]
    metrics["_reward_hash"] = hashlib.sha256(
        np.array(rewards, dtype=np.float64).tobytes()).hexdigest()[:16]

    return metrics


# =====================================================================
# Pretty Print
# =====================================================================
def print_results(metrics, algo, ckpt_path, schema_label, n_buildings):
    W = 72
    print(f"\n{'=' * W}")
    print(f"  EVALUATION RESULTS")
    print(f"  Algo:       {algo.upper()}")
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"  Schema:     {schema_label} ({n_buildings} building(s))")
    print(f"{'=' * W}")

    print(f"\n  --- Episode Summary ---")
    print(f"  Steps:          {metrics['steps']}")
    print(f"  Total reward:   {metrics['total_reward']:.2f}")
    print(f"  Mean reward:    {metrics['mean_reward']:.4f}")
    print(f"  Total cost:     {metrics['total_cost']:.2f}")

    print(f"\n  --- Per-Constraint Cumulative Costs ---")
    print(f"  C0 (EV departure):     {metrics['c0_cost']:.1f}")
    print(f"  C2 (battery SoC):      {metrics['c2_cost']:.1f}")
    print(f"  C3 (building power):   {metrics['c3_cost']:.1f}")
    print(f"  C4 (grid power):       {metrics['c4_cost']:.1f}")

    print(f"\n  {'=' * 60}")
    print(f"  BATTERY METRICS")
    print(f"  {'=' * 60}")
    print(f"  Charge   (action > 0.1):   {metrics['batt_charge_count']:>6d}  "
          f"({metrics['batt_charge_pct']:.1f}%)")
    print(f"  Discharge(action < -0.1):  {metrics['batt_discharge_count']:>6d}  "
          f"({metrics['batt_discharge_pct']:.1f}%)")
    print(f"  Idle     (|action| <= 0.1):{metrics['batt_idle_count']:>6d}  "
          f"({metrics['batt_idle_pct']:.1f}%)")
    print(f"  Solar avg  (10-15):  {metrics['batt_solar_avg']:+.4f}")
    print(f"  Peak avg   (17-21):  {metrics['batt_peak_avg']:+.4f}")
    print(f"  Off-peak avg (0-6):  {metrics['batt_offpeak_avg']:+.4f}")
    print(f"  Price correlation:   {metrics['batt_price_corr']:+.4f}")
    print(f"  SoC min/max/mean/std: {metrics['batt_soc_min']:.3f} / "
          f"{metrics['batt_soc_max']:.3f} / "
          f"{metrics['batt_soc_mean']:.3f} / "
          f"{metrics['batt_soc_std']:.3f}")
    print(f"  Action mean/std:     {metrics['batt_action_mean']:+.4f} / "
          f"{metrics['batt_action_std']:.4f}")
    print(f"  Action min/max:      {metrics['batt_action_min']:+.4f} / "
          f"{metrics['batt_action_max']:+.4f}")
    print(f"  Daily cycling:       {metrics['batt_cycling_days']}/"
          f"{metrics['batt_cycling_days_total']} days")

    print(f"\n  {'=' * 60}")
    print(f"  EV METRICS")
    print(f"  {'=' * 60}")
    print(f"  Connected steps:          {metrics['ev_connected_steps']}")
    print(f"  Charge (action > 0.1):    {metrics['ev_charge_pct']:.1f}%  "
          f"({metrics['ev_charge_count']} steps)")
    print(f"  V2G    (action < -0.1):   {metrics['ev_discharge_pct']:.1f}%  "
          f"({metrics['ev_discharge_count']} steps)")
    print(f"  Idle   (|action| <= 0.1): {metrics['ev_idle_pct']:.1f}%  "
          f"({metrics['ev_idle_count']} steps)")
    print(f"  V2G at peak (17-23):      {metrics['ev_v2g_peak_pct']:.1f}%  "
          f"({metrics['ev_peak_discharge']}/{metrics['ev_peak_connected']})")
    print(f"  Mean action (connected):  {metrics['ev_mean_action_connected']:+.4f}")

    print(f"\n  {'=' * 60}")
    print(f"  C0: EV DEPARTURE VIOLATIONS")
    print(f"  {'=' * 60}")
    print(f"  Total departures:    {metrics['c0_total_departures']}")
    print(f"  Violated:            {metrics['c0_violated_departures']}  "
          f"({metrics['c0_violation_pct']:.1f}%)")
    print(f"  Mean SoC at dep:     {metrics['c0_mean_soc_at_departure']:.4f}")
    print(f"  Mean required SoC:   {metrics['c0_mean_required_soc']:.4f}")
    print(f"  Mean deficit (viol): {metrics['c0_mean_deficit_violated']:.4f}")
    print(f"  SoC distribution at departure:")
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        k = f"c0_dep_soc_{lo:.1f}_{hi:.1f}"
        print(f"    [{lo:.1f}, {hi:.1f}): {metrics[k]}")

    print(f"\n  {'=' * 60}")
    print(f"  C2: BATTERY SoC BOUNDS")
    print(f"  {'=' * 60}")
    print(f"  SoC > {float(os.environ.get('CITYLEARN_STEMS_SOC_HIGH', 0.95))}: "
          f"{metrics['c2_above_high_count']}  ({metrics['c2_above_high_pct']:.2f}%)")
    print(f"  SoC < {float(os.environ.get('CITYLEARN_STEMS_SOC_LOW', 0.0))}: "
          f"{metrics['c2_below_low_count']}  ({metrics['c2_below_low_pct']:.2f}%)")

    print(f"\n  {'=' * 60}")
    print(f"  C3: BUILDING POWER (per-building |NEC| > P_bmax)")
    print(f"  {'=' * 60}")
    print(f"  Violations:  {metrics['c3_violations']}/{metrics['c3_total_checks']}  "
          f"({metrics['c3_violation_pct']:.2f}%)")
    print(f"  Mean |NEC|:  {metrics['c3_mean_abs_nec']:.4f}")
    print(f"  Max  |NEC|:  {metrics['c3_max_abs_nec']:.4f}")
    print(f"  P95  |NEC|:  {metrics['c3_p95_abs_nec']:.4f}")

    print(f"\n  {'=' * 60}")
    print(f"  C4: GRID POWER (total |NEC| > P_gmax)")
    print(f"  {'=' * 60}")
    print(f"  Violations:  {metrics['c4_violations']}/{metrics['c4_total_checks']}  "
          f"({metrics['c4_violation_pct']:.2f}%)")
    print(f"  Mean |NEC|:  {metrics['c4_mean_abs_nec']:.4f}")
    print(f"  Max  |NEC|:  {metrics['c4_max_abs_nec']:.4f}")
    print(f"  P95  |NEC|:  {metrics['c4_p95_abs_nec']:.4f}")

    print(f"\n  {'=' * 60}")
    print(f"  ENERGY")
    print(f"  {'=' * 60}")
    print(f"  Total import:        {metrics['energy_import_kwh']:.1f} kWh")
    print(f"  Total export:        {metrics['energy_export_kwh']:.1f} kWh")
    print(f"  Battery charge:      {metrics['energy_batt_charge_kwh']:.1f} kWh")
    print(f"  Battery discharge:   {metrics['energy_batt_discharge_kwh']:.1f} kWh")

    print(f"\n  --- Fingerprints ---")
    print(f"  Action hash: {metrics['_action_hash']}")
    print(f"  Reward hash: {metrics['_reward_hash']}")
    print(f"{'=' * W}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Bulletproof evaluation for PPO and SAC checkpoints")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--schema", required=True, choices=["1bld", "5bld"],
                        help="Schema: 1bld or 5bld")
    parser.add_argument("--months", required=True, type=int, choices=[3, 12],
                        help="Episode length: 3 or 12 months")
    parser.add_argument("--algo", required=True, choices=["ppo", "sac"],
                        help="Algorithm: ppo or sac")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-determinism-check", action="store_true",
                        help="Skip the second run for determinism verification")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    SEED = args.seed

    # --- Resolve schema path ---
    data_dir = os.path.join(
        PROJECT,
        "data/citylearn_challenge_2022_phase_all_plus_evs")
    schema_map = {
        ("1bld", 3): "schema_1building_3month.json",
        ("1bld", 12): "schema_1building.json",
        ("5bld", 3): "schema_5buildings_3month.json",
        ("5bld", 12): "schema_5buildings.json",
    }
    schema_file = schema_map.get((args.schema, args.months))
    if schema_file is None:
        print(f"ERROR: No schema for {args.schema} + {args.months} months")
        sys.exit(1)
    schema_path = os.path.join(data_dir, schema_file)
    if not os.path.exists(schema_path):
        print(f"ERROR: Schema file not found: {schema_path}")
        sys.exit(1)

    schema_label = f"{args.schema}-{args.months}mo"

    # --- Step 1: Set env vars ---
    print(f"\n[1/6] Setting environment variables...")
    set_all_env_vars(schema_path)
    print(f"  Schema: {schema_path}")

    # --- Step 2: Set seeds ---
    print(f"[2/6] Setting random seeds (seed={SEED})...")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- Step 3: Load checkpoint ---
    print(f"[3/6] Loading checkpoint: {args.checkpoint}")
    actor, obs_mean, obs_std, obs_clip, model_obs_dim, act_dim, ckpt_info = \
        load_checkpoint(args.checkpoint, args.algo)

    # --- Step 4: Build env ---
    print(f"[4/6] Building environment chain...")
    env, raw, buildings, names, batt_idx, ev_idx = build_env(schema_path)
    n_buildings = len(buildings)
    env_obs_dim = env.observation_space.shape[0]
    env_act_dim = env.action_space.shape[0]

    print(f"  Buildings:     {n_buildings}")
    print(f"  Obs dim (env): {env_obs_dim}  (model expects: {model_obs_dim})")
    print(f"  Act dim (env): {env_act_dim}  (model expects: {act_dim})")
    if model_obs_dim != env_obs_dim:
        diff = model_obs_dim - env_obs_dim
        if diff > 0:
            print(f"  WARNING: Model expects {diff} MORE obs dims. Will zero-pad.")
        else:
            print(f"  WARNING: Model expects {-diff} FEWER obs dims. Will trim.")
    if act_dim != env_act_dim:
        print(f"  WARNING: Action dim mismatch! Model={act_dim}, env={env_act_dim}")

    print(f"  Action layout:")
    for i, n in enumerate(names):
        tag = ""
        if i in batt_idx:
            tag = "  [BATTERY]"
        elif i in ev_idx:
            tag = "  [EV]"
        print(f"    [{i}] {n}{tag}")

    # --- Step 5: Run episode 1 ---
    print(f"\n[5/6] Running evaluation (run 1 of {'1' if args.no_determinism_check else '2'})...")

    # Reset seeds before each run
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    metrics1 = run_episode(
        env, raw, buildings, names, batt_idx, ev_idx,
        actor, obs_mean, obs_std, obs_clip,
        model_obs_dim, act_dim, SEED)

    print_results(metrics1, args.algo, args.checkpoint, schema_label, n_buildings)

    # --- Step 6: Determinism check ---
    if not args.no_determinism_check:
        print(f"\n[6/6] Determinism check (run 2)...")

        # Reset seeds
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # Rebuild env for clean state
        env2, raw2, buildings2, names2, batt_idx2, ev_idx2 = build_env(schema_path)

        metrics2 = run_episode(
            env2, raw2, buildings2, names2, batt_idx2, ev_idx2,
            actor, obs_mean, obs_std, obs_clip,
            model_obs_dim, act_dim, SEED)

        # Compare
        match = True
        mismatches = []
        for key in sorted(metrics1.keys()):
            v1 = metrics1[key]
            v2 = metrics2.get(key)
            if isinstance(v1, float):
                if abs(v1 - v2) > 1e-6:
                    mismatches.append((key, v1, v2))
                    match = False
            elif v1 != v2:
                mismatches.append((key, v1, v2))
                match = False

        if match:
            print(f"\n  DETERMINISM CHECK: PASSED")
            print(f"  Action hash run 1: {metrics1['_action_hash']}")
            print(f"  Action hash run 2: {metrics2['_action_hash']}")
        else:
            print(f"\n  DETERMINISM CHECK: FAILED")
            print(f"  Mismatched metrics:")
            for key, v1, v2 in mismatches:
                print(f"    {key}: run1={v1}, run2={v2}")
    else:
        print(f"\n[6/6] Determinism check: SKIPPED")

    # --- Save results ---
    if args.output:
        out_data = {
            "checkpoint": args.checkpoint,
            "algo": args.algo,
            "schema": schema_label,
            "seed": SEED,
            "metrics": {k: v for k, v in metrics1.items() if not k.startswith("_")},
            "fingerprints": {
                "action_hash": metrics1["_action_hash"],
                "reward_hash": metrics1["_reward_hash"],
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\n  Results saved to: {args.output}")
    else:
        # Default output path
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        default_out = os.path.join(
            PROJECT, "eval_results",
            f"eval_{args.algo}_{schema_label}_{ckpt_name}.json")
        os.makedirs(os.path.dirname(default_out), exist_ok=True)
        out_data = {
            "checkpoint": args.checkpoint,
            "algo": args.algo,
            "schema": schema_label,
            "seed": SEED,
            "metrics": {k: v for k, v in metrics1.items() if not k.startswith("_")},
            "fingerprints": {
                "action_hash": metrics1["_action_hash"],
                "reward_hash": metrics1["_reward_hash"],
            },
        }
        with open(default_out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\n  Results saved to: {default_out}")

    print(f"\n  EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
