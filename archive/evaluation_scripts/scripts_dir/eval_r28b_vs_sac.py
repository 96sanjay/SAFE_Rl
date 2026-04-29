#!/usr/bin/env python3
"""
Compare R28b PPO-Lag-Multi vs R28c SAC-Lag-Multi.

Runs deterministic 1-episode (8759 steps) evaluation for each and prints:
  - Per-constraint violation rates (C0, C2, C3, C4)
  - C0 per-departure violation rate
  - C3 per-building breakdown
  - CityLearn KPIs side-by-side
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

COST_KEYS = [
    "cost_ev_departure",          # C0
    "cost_ev_dense",              # C1
    "cost_stems_battery",         # C2
    "cost_stems_building_power",  # C3
    "cost_stems_grid_power",      # C4
]
COST_LABELS = ["C0 (EV departure)", "C1 (EV dense)", "C2 (battery SoC)",
               "C3 (building power)", "C4 (grid power)"]
COST_LIMITS = [100, 999999, 3500, 18000, 13000]

# ── Shared env vars (same for both R28b and R28c) ──
COMMON_ENV = {
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_PNORM_P": "4.0",
    "CITYLEARN_PID_LAGRANGE": "1",
    "CITYLEARN_EV_SAUTE": "0",
    # Reward weights (R28b 9-term)
    "STEMS_LAMBDA_EV": "2.0",
    "STEMS_ALPHA_EV_SMART": "1.5",
    "STEMS_EV_SLACK_ARB_SCALE": "1.0",
    "STEMS_ALPHA_V2G_CONTEXT": "1.5",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_ALPHA_GRID": "0.5",
    "STEMS_SG_THRESHOLD": "0.5",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_ALPHA_BUILD": "0.3",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_BETA_RAMP": "0.3",
    "STEMS_XI_RENEWABLE": "0.2",
    # Disabled
    "STEMS_ALPHA_EV_GUARD": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.0",
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "STEMS_ALPHA_LOAD_SHIFT": "0.0",
    # Cost weights
    "COST_W_C2": "0.0",
    "COST_W_C3": "5.0",
    "CITYLEARN_W_COST_EV": "1.0",
    "CITYLEARN_W_COST_SOC": "10.0",
    "CITYLEARN_W_COST_BUILDING": "0.5",
    "CITYLEARN_W_COST_GRID": "0.05",
    "CITYLEARN_EV_COST_SCALE": "3.0",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    # Controllability
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_BATT_CLAMP": "0",
    # Observation/Action
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_ACTION_MASK": "0",
    "CITYLEARN_POLICY_ACTION_MASK": "0",
    # Suppress noisy logs
    "CITYLEARN_KPI_FLUSH_EVERY_STEP": "0",
    "CITYLEARN_DEBUG_ACTION_CLIP": "0",
}

MODELS = {
    "R28b PPO-Lag": {
        "ckpt": "runs/r28b/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-04-15-09-32-30/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {},  # R28b: 9-term, no load_shift
    },
    "R28c PPO-Lag": {
        "ckpt": "runs/r28c/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-04-15-10-33-26/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {"STEMS_ALPHA_LOAD_SHIFT": "0.5"},  # R28c adds battery price arbitrage
    },
    "R28c SAC-Lag": {
        "ckpt": "runs/r28c_sac/5bld/SACLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-04-15-10-44-06/torch_save/epoch-95.pt",
        "actor_type": "sac",
        "extra_env": {"STEMS_ALPHA_LOAD_SHIFT": "0.5"},  # R28c adds load_shift
    },
    "R28 CPO": {
        "ckpt": "runs/r28_cpo/5bld/CPO-{CityLearnSafety-V2G-v2}/seed-000-2026-04-15-23-41-52/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {},
    },
    "R28 PPOLag (srv07)": {
        "ckpt": "runs/r28_ppolag/5bld/PPOLag-{CityLearnSafety-V2G-v2}/seed-000-2026-04-16-04-54-39/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
    },
    "R28 CPPOPID (srv07)": {
        "ckpt": "runs/r28_cppopid/5bld/CPPOPID-{CityLearnSafety-V2G-v2}/seed-000-2026-04-16-04-54-40/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
    },
    "R28 Optuna Best": {
        "ckpt": "runs/r28_optuna_best/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-04-16-18-23-28/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "1",
            "STEMS_LAMBDA_EV": "2.9196",
            "STEMS_ALPHA_EV_SMART": "1.2280",
            "STEMS_EV_SLACK_ARB_SCALE": "0.8791",
            "STEMS_ALPHA_V2G_CONTEXT": "1.7226",
            "STEMS_ALPHA_BARRIER": "0.2181",
            "STEMS_ALPHA_GRID": "3.9518",
            "STEMS_ALPHA_BUILD": "2.7973",
            "STEMS_BETA_RAMP": "0.0736",
            "STEMS_XI_RENEWABLE": "0.3809",
            "COST_W_C2": "0.0",
            "COST_W_C3": "5.0",
        },
    },
    # ── R28 stock OmniSafe benchmarks trained 2026-04-17 (100 epochs on V2G-v2) ──
    "R28 PPO (vanilla)": {
        "ckpt": "runs/r28_ppo/5bld/PPO-{CityLearnSafety-V2G-v2}/seed-000-2026-04-17-04-42-44/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
    },
    # PPOSaute: actor was trained with obs_dim = base+1 (safety-state augmentation).
    # build_eval_env() does NOT wrap with SauteAdapter, so obs dim will be 198, not 199.
    # If you hit a state_dict shape mismatch, either (a) pad obs with a constant 1.0
    # safety state, or (b) add SauteAdapter-equivalent dynamics in the eval loop that
    # tracks safety_state = safety_state - cost/safety_budget each step.
    "R28 PPOSaute (400k budget)": {
        "ckpt": "runs/r28_pposaute/5bld/PPOSaute-{CityLearnSafety-V2G-v2}/seed-000-2026-04-17-04-15-16/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
    },
    "R28 TRPOLag": {
        "ckpt": "runs/r28_trpolag/5bld/TRPOLag-{CityLearnSafety-V2G-v2}/seed-000-2026-04-17-04-42-41/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
    },
    "R28 SACLag": {
        "ckpt": "runs/r28_saclag/5bld/SACLag-{CityLearnSafety-V2G-v2}/seed-000-2026-04-17-04-42-43/torch_save/epoch-100.pt",
        "actor_type": "sac",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
    },
    # CSAC-LB R28-parity (100ep, 198-dim obs, 9-term STEMS reward). Retrained
    # 2026-04-18 with @env_register removed from CityLearnV2GCMDP and
    # omni_env_v2 imported first in train_csac_lb_v2g.py — so training used
    # the stock CityLearnCMDPv2 env (same as every other R28 algo).
    # Byte-identical env_vars to run_r28_benchmarks.sh.
    "R28 CSAC-LB": {
        "ckpt": "runs/csac_lb_v2g/CSACLBV2G-{CityLearnSafety-V2G-v2}/seed-042-2026-04-18-11-01-34/torch_save/epoch-100.pt",
        "actor_type": "sac",
        "extra_env": {
            "CITYLEARN_PID_LAGRANGE": "0",
            "COST_W_C2": "5.0",
            "COST_W_C1_DENSE": "0.0",
        },
    },
}


# ── Actor classes ──
class PPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.net(x))


class SACActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.act_dim = act_dim
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, act_dim * 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        mean = out[..., :self.act_dim]
        return torch.tanh(mean)


def normalize_obs(obs_np, obs_norm, device):
    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
    if obs_norm is None:
        return obs_t.unsqueeze(0)
    mean = obs_norm["_mean"].to(device).float()
    std = obs_norm["_std"].to(device).float()
    clip_val = obs_norm.get("_clip", torch.tensor(10.0)).to(device).float()
    obs_len = obs_t.shape[0]
    norm_len = mean.shape[0]
    if obs_len > norm_len:
        normed = torch.clamp((obs_t[:norm_len] - mean) / (std + 1e-8), -clip_val, clip_val)
        obs_t = torch.cat([normed, obs_t[norm_len:]], dim=0)
    elif obs_len < norm_len:
        clp = clip_val[:obs_len] if clip_val.dim() > 0 else clip_val
        obs_t = torch.clamp((obs_t - mean[:obs_len]) / (std[:obs_len] + 1e-8), -clp, clp)
    else:
        obs_t = torch.clamp((obs_t - mean) / (std + 1e-8), -clip_val, clip_val)
    return obs_t.unsqueeze(0)


def get_citylearn_env(env):
    cur = env
    seen = set()
    for _ in range(40):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "buildings") and hasattr(cur, "evaluate"):
            return cur
        for attr in ("base", "env", "unwrapped", "_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return None


def build_eval_env():
    import citylearn_safe.omni_env
    import citylearn_safe.omni_env_v2
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper

    base_env = make_base_env(central_agent=True)
    safety_env = CityLearnSafetyEnvV3(base_env)
    env = ForecastObsWrapper(safety_env, forecast_horizon=24)
    return env


def evaluate_model(name, cfg, device):
    """Run one episode and return results dict."""
    # Set env vars
    for k, v in COMMON_ENV.items():
        os.environ[k] = v
    for k, v in cfg["extra_env"].items():
        os.environ[k] = v
    schema_path = os.path.join(PROJECT_ROOT, "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json")
    os.environ["CITYLEARN_SCHEMA"] = schema_path

    env = build_eval_env()
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    # Load model
    ckpt_path = os.path.join(PROJECT_ROOT, cfg["ckpt"])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pi_sd = ckpt["pi"]

    # Detect Saute-augmented actor: ckpt first layer expects obs_dim + 1.
    # Happens with PPOSaute/TRPOSaute — SauteAdapter prepends a safety-state scalar.
    # Eval approximates by padding each obs with a constant 1.0 safety state (full budget),
    # which is a reasonable stand-in when we don't track safety state dynamically.
    saute_pad = False
    for k, v in pi_sd.items():
        if (k == "mean.0.weight" or k == "net.0.weight") and v.shape[1] == obs_dim + 1:
            saute_pad = True
            break
    actor_input_dim = obs_dim + (1 if saute_pad else 0)

    if cfg["actor_type"] == "ppo":
        model = PPOActor(actor_input_dim, act_dim)
        # PPO keys: 'mean.0.weight' etc → remap to 'net.0.weight'
        net_sd = {}
        for k, v in pi_sd.items():
            if k.startswith("mean."):
                net_sd["net." + k[5:]] = v  # 'mean.X' → 'net.X'
            elif k.startswith("net."):
                net_sd[k] = v
        if not net_sd:
            net_sd = {k: v for k, v in pi_sd.items() if 'weight' in k or 'bias' in k}
    else:
        model = SACActor(actor_input_dim, act_dim)
        net_sd = {k: v for k, v in pi_sd.items() if k.startswith("net.")}

    missing, unexpected = model.load_state_dict(net_sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys loading {name}: {missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")
    obs_norm = ckpt.get("obs_normalizer", None)
    model = model.to(device).eval()

    # Run episode
    obs, info = env.reset()
    step_rewards = []
    step_costs = defaultdict(list)
    step_violations = defaultdict(list)
    all_actions = []
    ev_departure_events = []
    c3_per_building_violations = []

    for step in range(8760):
        obs_normed = normalize_obs(obs, obs_norm, device)
        if saute_pad:
            # Append safety-state scalar (constant 1.0 = full budget) so actor input matches ckpt
            pad = torch.ones(obs_normed.shape[0], 1, device=device, dtype=obs_normed.dtype)
            obs_normed = torch.cat([obs_normed, pad], dim=-1)
        with torch.no_grad():
            action = model(obs_normed)
        action_np = action.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action_np)
        step_rewards.append(reward)
        all_actions.append(action_np.copy())

        for key in COST_KEYS:
            cost_val = float(info.get(key, 0.0))
            step_costs[key].append(cost_val)
            step_violations[key].append(1.0 if cost_val > 0.0 else 0.0)

        ev_departure_events.append(int(info.get("ev_departure_departures", 0)))
        c3_per_building_violations.append(float(info.get("building_power_violation_count", 0.0)))

        if terminated or truncated:
            break

    ep_len = len(step_rewards)

    # CityLearn KPIs
    citylearn_env = get_citylearn_env(env)
    cl_kpis = {}
    if citylearn_env is not None and hasattr(citylearn_env, "evaluate"):
        try:
            kpi_df = citylearn_env.evaluate()
            if kpi_df is not None:
                for _, row in kpi_df.iterrows():
                    level = str(row.get("level", ""))
                    cf = str(row.get("cost_function", "unknown"))
                    bname = str(row.get("name", "unknown"))
                    v = row.get("value", None)
                    if v is not None:
                        if level == "district":
                            cl_kpis[f"District|{cf}"] = float(v)
                        else:
                            cl_kpis[f"{bname}|{cf}"] = float(v)
        except Exception as e:
            print(f"  WARNING: KPI eval failed for {name}: {e}")

    # Compile results
    results = {
        "name": name,
        "ep_ret": sum(step_rewards),
        "ep_len": ep_len,
        "cl_kpis": cl_kpis,
    }

    # Per-constraint
    for i, (key, label, lim) in enumerate(zip(COST_KEYS, COST_LABELS, COST_LIMITS)):
        costs = step_costs[key]
        viols = step_violations[key]
        results[f"cost_{i}"] = sum(costs)
        results[f"viol_pct_{i}"] = 100.0 * np.mean(viols)
        results[f"limit_{i}"] = lim
        results[f"pass_{i}"] = sum(costs) <= lim

    # C0 per-departure
    total_deps = sum(ev_departure_events)
    c0_deficit = sum(1 for dep, cost in zip(ev_departure_events, step_costs[COST_KEYS[0]])
                     if dep > 0 and cost > 0)
    results["c0_departures"] = total_deps
    results["c0_deficit_departures"] = c0_deficit
    results["c0_per_dep_rate"] = 100.0 * c0_deficit / max(total_deps, 1)

    # C3 per-building
    c3_arr = np.array(c3_per_building_violations)
    results["c3_any_step_pct"] = 100.0 * np.sum(c3_arr > 0) / ep_len
    results["c3_per_bld_rate"] = 100.0 * np.sum(c3_arr) / (ep_len * 5)
    results["c3_mean_bld_per_step"] = c3_arr.mean()

    # Price correlation
    try:
        pricing = np.array(citylearn_env.buildings[0].pricing.electricity_pricing)[:ep_len]
        actions_arr = np.array(all_actions[:ep_len])
        safety_env = env
        while hasattr(safety_env, 'env'):
            if hasattr(safety_env, '_ev_charger_action_indices'):
                break
            safety_env = safety_env.env
        ev_idx = getattr(safety_env, '_ev_charger_action_indices', [])
        batt_idx = [i for i in range(actions_arr.shape[1]) if i not in ev_idx]
        batt_acts = actions_arr[:, batt_idx].mean(axis=1) if batt_idx else np.zeros(ep_len)
        ev_acts = actions_arr[:, ev_idx].mean(axis=1) if ev_idx else np.zeros(ep_len)
        results["price_corr_batt"] = float(np.corrcoef(pricing, batt_acts)[0, 1])
        results["price_corr_ev"] = float(np.corrcoef(pricing, ev_acts)[0, 1])
    except Exception:
        results["price_corr_batt"] = float('nan')
        results["price_corr_ev"] = float('nan')

    return results


def print_comparison_multi(all_results):
    """Print side-by-side comparison for N models."""
    w = 16  # column width
    models = list(all_results.values())
    n = len(models)

    print()
    print("=" * (40 + (w + 3) * n))
    print("  3-WAY ALGORITHM COMPARISON")
    print("=" * (40 + (w + 3) * n))

    # Header
    hdr = f"  {'Metric':<38s}"
    for m in models:
        hdr += f" | {m['name']:>{w}s}"
    print(f"\n{hdr}")
    print("  " + "-" * (36 + (w + 3) * n))

    # Episode metrics
    line = f"  {'Episode Return':<38s}"
    for m in models:
        line += f" | {m['ep_ret']:>{w}.1f}"
    print(line)

    # Constraint violations - violation % only
    print()
    print(f"  {'--- VIOLATION % (timesteps) ---':<38s}" + " |" * n)
    for i, label in enumerate(COST_LABELS):
        if i == 1:
            continue  # skip C1
        # Total cost + pass/fail
        line = f"  {label + ' total':<38s}"
        for m in models:
            cost = m[f"cost_{i}"]
            status = "PASS" if m[f"pass_{i}"] else "FAIL"
            line += f" | {cost:>{w-5}.0f} {status:>4s}"
        print(line)
        # Limit
        line = f"  {'  limit':<38s}"
        for m in models:
            line += f" | {m[f'limit_{i}']:>{w}d}"
        print(line)
        # Violation %
        line = f"  {'  violation %':<38s}"
        for m in models:
            vp = m[f"viol_pct_{i}"]
            line += f" | {vp:>{w-1}.1f}%"
        print(line)

    # C0 per-departure
    print()
    print(f"  {'--- C0 PER-DEPARTURE ---':<38s}" + " |" * n)
    line = f"  {'Total departures':<38s}"
    for m in models:
        line += f" | {m['c0_departures']:>{w}d}"
    print(line)
    line = f"  {'Departures with deficit':<38s}"
    for m in models:
        line += f" | {m['c0_deficit_departures']:>{w}d}"
    print(line)
    line = f"  {'Per-departure violation rate':<38s}"
    for m in models:
        line += f" | {m['c0_per_dep_rate']:>{w-1}.1f}%"
    print(line)

    # C3 per-building
    print()
    print(f"  {'--- C3 PER-BUILDING ---':<38s}" + " |" * n)
    line = f"  {'Steps with ANY bld violating':<38s}"
    for m in models:
        line += f" | {m['c3_any_step_pct']:>{w-1}.1f}%"
    print(line)
    line = f"  {'Per-building violation rate':<38s}"
    for m in models:
        line += f" | {m['c3_per_bld_rate']:>{w-1}.1f}%"
    print(line)

    # Price correlation
    print()
    print(f"  {'--- PRICE AWARENESS (neg=good) ---':<38s}" + " |" * n)
    line = f"  {'Battery-price correlation':<38s}"
    for m in models:
        line += f" | {m['price_corr_batt']:>{w}.4f}"
    print(line)
    line = f"  {'EV-price correlation':<38s}"
    for m in models:
        line += f" | {m['price_corr_ev']:>{w}.4f}"
    print(line)

    # CityLearn KPIs — district level
    district_cfs = sorted(set(
        k.split("|", 1)[1] for m in models for k in m["cl_kpis"] if k.startswith("District|")
    ))
    if district_cfs:
        print()
        print(f"  {'--- DISTRICT KPIs (<1 = better) ---':<38s}" + " |" * n)
        for cf in district_cfs:
            key = f"District|{cf}"
            line = f"  {cf:<38s}"
            for m in models:
                v = m["cl_kpis"].get(key, float('nan'))
                if np.isnan(v):
                    line += f" | {'N/A':>{w}s}"
                else:
                    line += f" | {v:>{w}.4f}"
            print(line)

    # Per-building average across cost functions
    bld_names = sorted(set(
        k.split("|", 1)[0] for m in models for k in m["cl_kpis"] if not k.startswith("District|")
    ))
    if bld_names and district_cfs:
        # Show key KPIs per building
        key_cfs = [cf for cf in district_cfs if cf in (
            "electricity_consumption_total", "cost_total", "carbon_emissions_total",
            "daily_peak_average", "ramping_average", "daily_one_minus_load_factor_average",
            "all_time_peak_average",
        )]
        if key_cfs:
            print()
            print(f"  {'--- PER-BUILDING KEY KPIs ---':<38s}" + " |" * n)
            for cf in key_cfs:
                for bname in bld_names:
                    key = f"{bname}|{cf}"
                    label = f"  {bname}:{cf[:20]}"
                    line = f"  {label:<38s}"
                    for m in models:
                        v = m["cl_kpis"].get(key, float('nan'))
                        if np.isnan(v):
                            line += f" | {'N/A':>{w}s}"
                        else:
                            line += f" | {v:>{w}.4f}"
                    print(line)

    print()
    print("=" * (40 + (w + 3) * n))


def main():
    device = torch.device("cpu")

    results = {}
    for name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"  Checkpoint: {cfg['ckpt']}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            results[name] = evaluate_model(name, cfg, device)
            print(f"  Done in {time.time()-t0:.1f}s (EpRet={results[name]['ep_ret']:.1f})")
        except Exception as e:
            import traceback
            print(f"  FAILED after {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"  skipping {name}, continuing with next model")

    print_comparison_multi(results)


if __name__ == "__main__":
    main()
