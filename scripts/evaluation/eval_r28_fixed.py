#!/usr/bin/env python3
"""
Fixed evaluation script (replaces eval_r28b_vs_sac.py for canonical reporting).

Fixes vs eval_r28b_vs_sac.py:
  A. Per-departure rate uses info["ev_departure_violation_count_deficit"]
     (the env's per-departure counter). The old code counted STEPS where any
     deficit happened, which under-counts when 2+ EVs depart in the same hour.
     Reportable ceiling under the old method = 756/1070 = 70.7%.
     Old metric retained as `c0_per_dep_rate_OLD_BUGGY` for side-by-side.
  B. Env-var leakage fix: pop `extra_env` keys at end of evaluate_model.
  C. Avoidable / unavoidable kWh aggregation from
     info["cost_ev_departure_avoidable" | "cost_ev_departure_unavoidable"]
     (divided by CITYLEARN_EV_COST_SCALE to convert cost units back to kWh).
  D. Absolute deficit kWh from info["ev_departure_deficit_kwh"]
     (already kWh; no division needed).
  E. PPOSaute dropped (its eval is biased — fake constant safety_state=1.0).
     CSAC-LB also excluded (still training).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

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

# Excluded vs original MODELS dict:
#  - "R28 PPOSaute (400k budget)": eval is biased (fake constant safety_state=1.0)
#  - "R28 CSAC-LB (custom)": still training as of 2026-04-18
MODELS = {
    "R28b PPO-Lag": {
        "ckpt": "runs/r28b/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-04-15-09-32-30/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {},
    },
    "R28c PPO-Lag": {
        "ckpt": "runs/r28c/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-04-15-10-33-26/torch_save/epoch-100.pt",
        "actor_type": "ppo",
        "extra_env": {"STEMS_ALPHA_LOAD_SHIFT": "0.5"},
    },
    "R28c SAC-Lag": {
        "ckpt": "runs/r28c_sac/5bld/SACLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-04-15-10-44-06/torch_save/epoch-95.pt",
        "actor_type": "sac",
        "extra_env": {"STEMS_ALPHA_LOAD_SHIFT": "0.5"},
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
    "R28 PPO (vanilla)": {
        "ckpt": "runs/r28_ppo/5bld/PPO-{CityLearnSafety-V2G-v2}/seed-000-2026-04-17-04-42-44/torch_save/epoch-100.pt",
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
    import citylearn_safe.omni_env  # noqa: F401
    import citylearn_safe.cmdp_env  # noqa: F401
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env import CityLearnSafetyEnv
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper

    base_env = make_base_env(central_agent=True)
    safety_env = CityLearnSafetyEnv(base_env)
    env = ForecastObsWrapper(safety_env, forecast_horizon=24)
    return env


def evaluate_model(name, cfg, device):
    """Run one episode and return results dict.

    Sets env vars from COMMON_ENV + cfg['extra_env'], and ALWAYS pops
    cfg['extra_env'] keys at the end (try/finally) to avoid leakage to
    subsequent models in the same process.
    """
    # Set env vars
    for k, v in COMMON_ENV.items():
        os.environ[k] = v
    for k, v in cfg["extra_env"].items():
        os.environ[k] = v
    schema_path = os.path.join(
        PROJECT_ROOT,
        "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    )
    os.environ["CITYLEARN_SCHEMA"] = schema_path

    try:
        ev_cost_scale = float(os.environ.get("CITYLEARN_EV_COST_SCALE", "1.0"))

        env = build_eval_env()
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])

        # Load model
        ckpt_path = os.path.join(PROJECT_ROOT, cfg["ckpt"])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        pi_sd = ckpt["pi"]

        # Detect Saute-augmented actor (still detected here for safety, even
        # though PPOSaute is not in MODELS — keeps the function defensive).
        saute_pad = False
        for k, v in pi_sd.items():
            if (k == "mean.0.weight" or k == "net.0.weight") and v.shape[1] == obs_dim + 1:
                saute_pad = True
                break
        actor_input_dim = obs_dim + (1 if saute_pad else 0)

        if cfg["actor_type"] == "ppo":
            model = PPOActor(actor_input_dim, act_dim)
            net_sd = {}
            for k, v in pi_sd.items():
                if k.startswith("mean."):
                    net_sd["net." + k[5:]] = v
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

        # ── Rollout ──
        obs, info = env.reset()
        step_rewards = []
        step_costs = defaultdict(list)
        step_violations = defaultdict(list)
        all_actions = []
        ev_departure_events = []
        c3_per_building_violations = []

        # NEW per-departure / kWh accumulators
        step_c0_violation_counts = []      # FIX A: per-departure deficit counts
        step_avoidable_cost = []           # FIX C: cost-units of avoidable deficit
        step_unavoidable_cost = []         # FIX C: cost-units of unavoidable deficit
        step_deficit_kwh = []              # FIX D: env-emitted kWh deficit per step

        for step in range(8760):
            obs_normed = normalize_obs(obs, obs_norm, device)
            if saute_pad:
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

            # ── FIX A: per-departure violation count (env's own counter) ──
            step_c0_violation_counts.append(int(info.get("ev_departure_violation_count_deficit", 0)))
            # ── FIX C: avoidable / unavoidable cost (units = ev_cost_scale * kWh) ──
            step_avoidable_cost.append(float(info.get("cost_ev_departure_avoidable", 0.0)))
            step_unavoidable_cost.append(float(info.get("cost_ev_departure_unavoidable", 0.0)))
            # ── FIX D: absolute deficit kWh (env emits in kWh; no scale division) ──
            step_deficit_kwh.append(float(info.get("ev_departure_deficit_kwh", 0.0)))

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

        # ── Compile results ──
        results = {
            "name": name,
            "ep_ret": float(sum(step_rewards)),
            "ep_len": ep_len,
            "cl_kpis": cl_kpis,
        }

        # Per-constraint
        for i, (key, label, lim) in enumerate(zip(COST_KEYS, COST_LABELS, COST_LIMITS)):
            costs = step_costs[key]
            viols = step_violations[key]
            results[f"cost_{i}"] = float(sum(costs))
            results[f"viol_pct_{i}"] = float(100.0 * np.mean(viols))
            results[f"limit_{i}"] = lim
            results[f"pass_{i}"] = bool(sum(costs) <= lim)

        # ── FIX A: C0 per-departure (NEW correct + OLD buggy side-by-side) ──
        total_deps = int(sum(ev_departure_events))
        c0_deficit_departures = int(sum(step_c0_violation_counts))   # NEW correct numerator
        c0_deficit_OLD = int(sum(                                    # OLD buggy = step-level
            1 for dep, cost in zip(ev_departure_events, step_costs[COST_KEYS[0]])
            if dep > 0 and cost > 0
        ))
        results["c0_departures"] = total_deps
        results["c0_deficit_departures"] = c0_deficit_departures
        results["c0_per_dep_rate"] = float(100.0 * c0_deficit_departures / max(total_deps, 1))
        results["c0_per_dep_rate_OLD_BUGGY"] = float(100.0 * c0_deficit_OLD / max(total_deps, 1))
        results["c0_deficit_steps_OLD_BUGGY"] = c0_deficit_OLD

        # ── FIX C + D: kWh aggregations ──
        total_avoidable_cost = float(sum(step_avoidable_cost))
        total_unavoidable_cost = float(sum(step_unavoidable_cost))
        results["avoidable_kwh"] = total_avoidable_cost / max(ev_cost_scale, 1e-9)
        results["unavoidable_kwh"] = total_unavoidable_cost / max(ev_cost_scale, 1e-9)
        # Prefer env-emitted ev_departure_deficit_kwh; fall back to cost/scale.
        env_deficit_kwh_total = float(sum(step_deficit_kwh))
        if env_deficit_kwh_total > 0:
            results["total_deficit_kwh"] = env_deficit_kwh_total
            results["total_deficit_kwh_source"] = "info[ev_departure_deficit_kwh]"
        else:
            results["total_deficit_kwh"] = (
                results["avoidable_kwh"] + results["unavoidable_kwh"]
            )
            results["total_deficit_kwh_source"] = "cost_ev_departure / EV_COST_SCALE (fallback)"
        results["ev_cost_scale"] = ev_cost_scale

        # C3 per-building
        c3_arr = np.array(c3_per_building_violations)
        results["c3_any_step_pct"] = float(100.0 * np.sum(c3_arr > 0) / ep_len)
        results["c3_per_bld_rate"] = float(100.0 * np.sum(c3_arr) / (ep_len * 5))
        results["c3_mean_bld_per_step"] = float(c3_arr.mean())

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

    finally:
        # ── FIX B: env-var leakage cleanup ──
        # Pop ONLY the keys this model added via extra_env (COMMON_ENV stays —
        # it's set fresh at the top of each evaluate_model call anyway).
        for k in cfg["extra_env"]:
            os.environ.pop(k, None)


def print_fixed_table(all_results):
    """Print the requested comparison table with OLD vs NEW per-dep rate."""
    print()
    print("=" * 130)
    print("  PER-DEPARTURE RATE FIX — OLD (buggy) vs NEW (correct) + kWh breakdown")
    print("=" * 130)
    header = (
        f"  {'Model':<24s} | {'OLD per-dep %':>14s} | {'NEW per-dep %':>14s} | "
        f"{'Δ abs':>7s} | {'total_kWh':>10s} | {'avoid_kWh':>10s} | {'unavoid_kWh':>11s}"
    )
    print(header)
    print("  " + "-" * 124)
    for name, m in all_results.items():
        old = m["c0_per_dep_rate_OLD_BUGGY"]
        new = m["c0_per_dep_rate"]
        delta = new - old
        tot_kwh = m["total_deficit_kwh"]
        av = m["avoidable_kwh"]
        unav = m["unavoidable_kwh"]
        print(
            f"  {name:<24s} | {old:>13.1f}% | {new:>13.1f}% | "
            f"{delta:>+6.1f}  | {tot_kwh:>10.1f} | {av:>10.1f} | {unav:>11.1f}"
        )
    print("=" * 130)
    print()


def main():
    device = torch.device("cpu")
    out_dir = os.path.join(PROJECT_ROOT, "scripts", "eval_outputs")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    json_path = os.path.join(out_dir, f"eval_r28_fixed_{ts}.json")

    results = {}
    for name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"  Checkpoint: {cfg['ckpt']}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            results[name] = evaluate_model(name, cfg, device)
            r = results[name]
            print(
                f"  Done in {time.time()-t0:.1f}s | "
                f"EpRet={r['ep_ret']:.1f} | "
                f"per-dep OLD={r['c0_per_dep_rate_OLD_BUGGY']:.1f}% NEW={r['c0_per_dep_rate']:.1f}% | "
                f"deficit_kWh={r['total_deficit_kwh']:.1f} "
                f"(avoid={r['avoidable_kwh']:.1f} unavoid={r['unavoidable_kwh']:.1f})"
            )
        except Exception as e:
            import traceback
            print(f"  FAILED after {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"  skipping {name}, continuing with next model")

    # Persist JSON (numeric-only — drop the cl_kpis dict to keep file small/portable)
    json_safe = {}
    for name, r in results.items():
        json_safe[name] = {k: v for k, v in r.items() if k != "cl_kpis"}
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2)
    print(f"\nResults JSON written to: {json_path}")

    print_fixed_table(results)


if __name__ == "__main__":
    main()
