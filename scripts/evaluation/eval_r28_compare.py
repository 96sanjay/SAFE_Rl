#!/usr/bin/env python3
"""
Comparative evaluation of R28b PPO, R28c PPO, R28c SAC.

Runs deterministic 1-episode evaluation for each model and prints
a unified comparison table with:
  - Per-constraint violation percentages
  - C0 per-departure violation rate
  - C3 per-building breakdown
  - Price-aware battery/EV action analysis
  - CityLearn KPIs

Usage:
    cd <PROJECT_ROOT>
    conda activate citylearn
    python scripts/evaluation/eval_r28_compare.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

REWARD_KEYS = [
    "r_eco", "r_sg", "r_sb", "r_ramp", "r_ren", "r_ev",
    "r_ev_guard", "r_barrier", "r_v2g_ctx", "r_ev_smart",
    "r_ev_slack_arb", "r_peak_shave", "r_load_shift", "r_grid_mild",
]

# ── Model configs ──
MODELS = {
    "R28b PPO": {
        "ckpt": "runs/r28b/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-04-15-09-32-30/torch_save/epoch-95.pt",
        "actor_type": "ppo",
        "reward_vars": {
            "STEMS_LAMBDA_EV": "2.0",
            "STEMS_ALPHA_EV_SMART": "1.5",
            "STEMS_EV_SLACK_ARB_SCALE": "1.0",
            "STEMS_ALPHA_V2G_CONTEXT": "1.5",
            "STEMS_ALPHA_BARRIER": "0.5",
            "STEMS_ALPHA_LOAD_SHIFT": "0.0",   # OFF in R28b
            "STEMS_ALPHA_GRID": "0.5",
            "STEMS_SG_THRESHOLD": "0.5",
            "STEMS_SG_EXPORT_CREDIT": "0.5",
            "STEMS_ALPHA_BUILD": "0.3",
            "STEMS_SB_ASYMMETRIC": "1",
            "STEMS_BETA_RAMP": "0.3",
            "STEMS_XI_RENEWABLE": "0.2",
            "STEMS_ALPHA_EV_GUARD": "0.0",
            "STEMS_ALPHA_GRID_MILD": "0.0",
            "STEMS_MU_ECONOMIC": "0.0",
            "STEMS_ALPHA_PEAK_SHAVE": "0.0",
        },
    },
    "R28c PPO": {
        "ckpt": "runs/r28c/5bld/PPOLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-04-15-10-33-26/torch_save/epoch-95.pt",
        "actor_type": "ppo",
        "reward_vars": {
            "STEMS_LAMBDA_EV": "2.0",
            "STEMS_ALPHA_EV_SMART": "1.5",
            "STEMS_EV_SLACK_ARB_SCALE": "1.0",
            "STEMS_ALPHA_V2G_CONTEXT": "1.5",
            "STEMS_ALPHA_BARRIER": "0.5",
            "STEMS_ALPHA_LOAD_SHIFT": "0.5",   # ON in R28c
            "STEMS_ALPHA_GRID": "0.5",
            "STEMS_SG_THRESHOLD": "0.5",
            "STEMS_SG_EXPORT_CREDIT": "0.5",
            "STEMS_ALPHA_BUILD": "0.3",
            "STEMS_SB_ASYMMETRIC": "1",
            "STEMS_BETA_RAMP": "0.3",
            "STEMS_XI_RENEWABLE": "0.2",
            "STEMS_ALPHA_EV_GUARD": "0.0",
            "STEMS_ALPHA_GRID_MILD": "0.0",
            "STEMS_MU_ECONOMIC": "0.0",
            "STEMS_ALPHA_PEAK_SHAVE": "0.0",
        },
    },
    "R28c SAC": {
        "ckpt": "runs/r28c_sac/5bld/SACLagMulti-{CityLearnSafety-V2G-v2}/seed-042-2026-04-15-10-44-06/torch_save/epoch-85.pt",
        "actor_type": "sac",
        "reward_vars": {
            "STEMS_LAMBDA_EV": "2.0",
            "STEMS_ALPHA_EV_SMART": "1.5",
            "STEMS_EV_SLACK_ARB_SCALE": "1.0",
            "STEMS_ALPHA_V2G_CONTEXT": "1.5",
            "STEMS_ALPHA_BARRIER": "0.5",
            "STEMS_ALPHA_LOAD_SHIFT": "0.5",
            "STEMS_ALPHA_GRID": "0.5",
            "STEMS_SG_THRESHOLD": "0.5",
            "STEMS_SG_EXPORT_CREDIT": "0.5",
            "STEMS_ALPHA_BUILD": "0.3",
            "STEMS_SB_ASYMMETRIC": "1",
            "STEMS_BETA_RAMP": "0.3",
            "STEMS_XI_RENEWABLE": "0.2",
            "STEMS_ALPHA_EV_GUARD": "0.0",
            "STEMS_ALPHA_GRID_MILD": "0.0",
            "STEMS_MU_ECONOMIC": "0.0",
            "STEMS_ALPHA_PEAK_SHAVE": "0.0",
        },
    },
}


# ── Actor classes ──
class PPOActor(nn.Module):
    """PPO gaussian_learning actor (Tanh activation)."""
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
    """SAC gaussian_sac actor (ReLU activation, outputs mean+log_std)."""
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


def set_common_env_vars():
    """Set env vars common to all R28 runs."""
    schema_path = os.path.join(
        PROJECT_ROOT,
        "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    )
    common = {
        "CITYLEARN_SCHEMA": schema_path,
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
        "COST_W_C2": "0.0",
        "COST_W_C3": "5.0",
        "CITYLEARN_W_COST_EV": "1.0",
        "CITYLEARN_W_COST_SOC": "10.0",
        "CITYLEARN_W_COST_BUILDING": "0.5",
        "CITYLEARN_W_COST_GRID": "0.05",
        "CITYLEARN_EV_COST_SCALE": "3.0",
        "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
        "CITYLEARN_INCLUDE_EV_COST": "1",
        "CITYLEARN_C3_CONTROLLABLE": "1",
        "CITYLEARN_BATT_CLAMP": "0",
        "CITYLEARN_SPATIAL_OBS": "0",
        "CITYLEARN_WM_DISABLE": "1",
        "CITYLEARN_EV_ACTION_CLAMP": "0",
        "CITYLEARN_ACTION_MASK": "0",
        "CITYLEARN_POLICY_ACTION_MASK": "0",
        "CITYLEARN_KPI_FLUSH_EVERY_STEP": "0",
        "CITYLEARN_DEBUG_ACTION_CLIP": "0",
    }
    for k, v in common.items():
        os.environ[k] = v


def build_eval_env():
    import citylearn_safe.cmdp_env    # noqa: F401  (registers env)
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env import CityLearnSafetyEnv
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper

    base_env = make_base_env(central_agent=True)
    safety_env = CityLearnSafetyEnv(base_env)
    env = ForecastObsWrapper(safety_env, forecast_horizon=24)
    return env


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
        cmin = -clip_val[:obs_len] if clip_val.dim() > 0 else -clip_val
        cmax = clip_val[:obs_len] if clip_val.dim() > 0 else clip_val
        obs_t = torch.clamp((obs_t - mean[:obs_len]) / (std[:obs_len] + 1e-8), cmin, cmax)
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


def load_model(cfg, obs_dim, act_dim, device):
    ckpt_path = cfg["ckpt"]
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pi_sd = ckpt["pi"]
    obs_norm = ckpt.get("obs_normalizer", None)

    if cfg["actor_type"] == "ppo":
        model = PPOActor(obs_dim, act_dim)
        # PPO keys: mean.0.weight → net.0.weight
        net_sd = {}
        for k, v in pi_sd.items():
            if k.startswith("mean."):
                net_sd["net." + k[len("mean."):]] = v
        model.load_state_dict(net_sd, strict=False)
    else:
        model = SACActor(obs_dim, act_dim)
        # SAC keys: net.0.weight directly
        net_sd = {k: v for k, v in pi_sd.items() if k.startswith("net.")}
        model.load_state_dict(net_sd, strict=False)

    return model.to(device).eval(), obs_norm


def run_eval(model, obs_norm, env, act_dim, device):
    """Run one deterministic episode, return results dict."""
    obs, info = env.reset()

    step_rewards = []
    step_costs = defaultdict(list)
    step_violations = defaultdict(list)
    all_actions = []
    ev_departure_events = []
    c3_per_building_violations = []
    ep_len = 0

    for step in range(8760):
        obs_normed = normalize_obs(obs, obs_norm, device)
        with torch.no_grad():
            action = model(obs_normed)
        action_np = action.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action_np)
        ep_len += 1
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

    # CityLearn KPIs
    citylearn_env = get_citylearn_env(env)
    cl_kpis = {}
    if citylearn_env is not None and hasattr(citylearn_env, "evaluate"):
        try:
            kpi_df = citylearn_env.evaluate()
            if kpi_df is not None:
                for _, row in kpi_df.iterrows():
                    name = str(row.get("name", row.get("cost_function", "unknown")))
                    val = row.get("value", row.get("cost", 0.0))
                    if name and val is not None:
                        cl_kpis[name] = float(val)
        except Exception:
            pass

    # Price analysis
    price_analysis = {}
    try:
        pricing = np.array(citylearn_env.buildings[0].pricing.electricity_pricing)[:ep_len]
        actions_arr = np.array(all_actions[:ep_len])

        safety_env = env
        while hasattr(safety_env, 'env'):
            if hasattr(safety_env, '_ev_charger_action_indices'):
                break
            safety_env = safety_env.env
        ev_indices = getattr(safety_env, '_ev_charger_action_indices', [])
        batt_indices = [i for i in range(act_dim) if i not in ev_indices]

        batt_actions = actions_arr[:, batt_indices].mean(axis=1) if batt_indices else np.zeros(ep_len)
        ev_actions = actions_arr[:, ev_indices].mean(axis=1) if ev_indices else np.zeros(ep_len)

        price_sorted = np.argsort(pricing)
        n = len(pricing)
        bounds = [0, n // 4, n // 2, 3 * n // 4, n]

        quartiles = []
        for qi in range(4):
            idx = price_sorted[bounds[qi]:bounds[qi + 1]]
            quartiles.append({
                "batt_mean": float(batt_actions[idx].mean()),
                "ev_mean": float(ev_actions[idx].mean()),
                "price_lo": float(pricing[idx].min()),
                "price_hi": float(pricing[idx].max()),
            })

        price_analysis = {
            "batt_corr": float(np.corrcoef(pricing, batt_actions)[0, 1]),
            "ev_corr": float(np.corrcoef(pricing, ev_actions)[0, 1]),
            "quartiles": quartiles,
        }
    except Exception:
        pass

    # C0 per-departure
    total_departures = sum(ev_departure_events)
    c0_deficit_departures = sum(
        1 for dep, cost in zip(ev_departure_events, step_costs[COST_KEYS[0]])
        if dep > 0 and cost > 0
    )

    # C3 per-building
    c3_arr = np.array(c3_per_building_violations)

    return {
        "ep_ret": sum(step_rewards),
        "ep_len": ep_len,
        "mean_reward": float(np.mean(step_rewards)),
        "costs": {k: sum(step_costs[k]) for k in COST_KEYS},
        "viol_pct": {k: 100.0 * np.mean(step_violations[k]) for k in COST_KEYS},
        "cost_mean": {k: float(np.mean(step_costs[k])) for k in COST_KEYS},
        "cost_max": {k: float(np.max(step_costs[k])) for k in COST_KEYS},
        "c0_departures": total_departures,
        "c0_deficit_departures": c0_deficit_departures,
        "c0_viol_per_dep": 100.0 * c0_deficit_departures / max(1, total_departures),
        "c3_timestep_viol": 100.0 * float(np.sum(c3_arr > 0)) / ep_len,
        "c3_building_viol": 100.0 * float(np.sum(c3_arr)) / (ep_len * 5),
        "c3_mean_bld_per_step": float(c3_arr.mean()),
        "cl_kpis": cl_kpis,
        "price": price_analysis,
    }


def main():
    device = torch.device("cpu")
    set_common_env_vars()

    all_results = {}

    for name, cfg in MODELS.items():
        print(f"\n{'=' * 76}")
        print(f"  Evaluating: {name}")
        print(f"  Checkpoint: {cfg['ckpt'].split('/')[-1]}")
        print(f"{'=' * 76}")

        # Set model-specific reward vars
        for k, v in cfg["reward_vars"].items():
            os.environ[k] = v

        env = build_eval_env()
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])

        model, obs_norm = load_model(cfg, obs_dim, act_dim, device)
        t0 = time.time()
        results = run_eval(model, obs_norm, env, act_dim, device)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({results['ep_len'] / elapsed:.0f} FPS)")

        all_results[name] = results
        del env

    # ════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ════════════════════════════════════════════════════════════════════
    names = list(all_results.keys())

    print("\n\n")
    print("=" * 90)
    print("  R28 COMPARATIVE EVALUATION")
    print("=" * 90)

    # ── Summary ──
    print(f"\n  {'Metric':<30s}", end="")
    for n in names:
        print(f" | {n:>15s}", end="")
    print()
    print("  " + "-" * (30 + 18 * len(names)))

    print(f"  {'Episode Return':<30s}", end="")
    for n in names:
        print(f" | {all_results[n]['ep_ret']:>15.1f}", end="")
    print()

    print(f"  {'Mean Step Reward':<30s}", end="")
    for n in names:
        print(f" | {all_results[n]['mean_reward']:>15.4f}", end="")
    print()

    # ── Constraint costs ──
    print(f"\n  {'--- CONSTRAINT COSTS ---':<30s}", end="")
    for n in names:
        print(f" | {'':>15s}", end="")
    print()

    for key, label, lim in zip(COST_KEYS, COST_LABELS, COST_LIMITS):
        if key == COST_KEYS[1]:
            continue  # skip C1
        print(f"  {label + ' (cost)':<30s}", end="")
        for n in names:
            c = all_results[n]["costs"][key]
            status = "ok" if c <= lim else "OVER"
            print(f" | {c:>10.0f} {status:>4s}", end="")
        print(f"  [limit={lim}]")

    # ── Violation percentages ──
    print(f"\n  {'--- VIOLATION PERCENTAGES ---':<30s}", end="")
    for n in names:
        print(f" | {'':>15s}", end="")
    print()

    for key, label in zip(COST_KEYS, COST_LABELS):
        if key == COST_KEYS[1]:
            continue
        print(f"  {label + ' (viol%)':<30s}", end="")
        for n in names:
            print(f" | {all_results[n]['viol_pct'][key]:>14.1f}%", end="")
        print()

    # ── C0 per-departure ──
    print(f"\n  {'--- C0 PER-DEPARTURE ---':<30s}", end="")
    for n in names:
        print(f" | {'':>15s}", end="")
    print()

    print(f"  {'Total departures':<30s}", end="")
    for n in names:
        print(f" | {all_results[n]['c0_departures']:>15d}", end="")
    print()

    print(f"  {'Departures with deficit':<30s}", end="")
    for n in names:
        print(f" | {all_results[n]['c0_deficit_departures']:>15d}", end="")
    print()

    print(f"  {'Violation rate (per dep)':<30s}", end="")
    for n in names:
        print(f" | {all_results[n]['c0_viol_per_dep']:>14.1f}%", end="")
    print()

    # ── C3 per-building ──
    print(f"\n  {'--- C3 PER-BUILDING ---':<30s}", end="")
    for n in names:
        print(f" | {'':>15s}", end="")
    print()

    print(f"  {'C3 timestep viol %':<30s}", end="")
    for n in names:
        print(f" | {all_results[n]['c3_timestep_viol']:>14.1f}%", end="")
    print()

    print(f"  {'C3 per-building viol %':<30s}", end="")
    for n in names:
        print(f" | {all_results[n]['c3_building_viol']:>14.1f}%", end="")
    print()

    print(f"  {'Mean bld violating/step':<30s}", end="")
    for n in names:
        print(f" | {all_results[n]['c3_mean_bld_per_step']:>13.2f}/5", end="")
    print()

    # ── Price-aware analysis ──
    print(f"\n  {'--- PRICE AWARENESS ---':<30s}", end="")
    for n in names:
        print(f" | {'':>15s}", end="")
    print()

    print(f"  {'Battery-price correlation':<30s}", end="")
    for n in names:
        pa = all_results[n].get("price", {})
        bc = pa.get("batt_corr", float("nan"))
        print(f" | {bc:>+15.4f}", end="")
    print("  (neg=good)")

    print(f"  {'EV-price correlation':<30s}", end="")
    for n in names:
        pa = all_results[n].get("price", {})
        ec = pa.get("ev_corr", float("nan"))
        print(f" | {ec:>+15.4f}", end="")
    print("  (neg=good)")

    # Price quartile table
    q_labels = ["Q1 (cheapest)", "Q2", "Q3", "Q4 (expensive)"]
    print(f"\n  Battery action by price quartile:")
    print(f"  {'Quartile':<18s}", end="")
    for n in names:
        print(f" | {n:>15s}", end="")
    print()
    print("  " + "-" * (18 + 18 * len(names)))
    for qi in range(4):
        print(f"  {q_labels[qi]:<18s}", end="")
        for n in names:
            pa = all_results[n].get("price", {})
            qs = pa.get("quartiles", [])
            if qi < len(qs):
                print(f" | {qs[qi]['batt_mean']:>+15.4f}", end="")
            else:
                print(f" | {'N/A':>15s}", end="")
        print()

    print(f"\n  EV action by price quartile:")
    print(f"  {'Quartile':<18s}", end="")
    for n in names:
        print(f" | {n:>15s}", end="")
    print()
    print("  " + "-" * (18 + 18 * len(names)))
    for qi in range(4):
        print(f"  {q_labels[qi]:<18s}", end="")
        for n in names:
            pa = all_results[n].get("price", {})
            qs = pa.get("quartiles", [])
            if qi < len(qs):
                print(f" | {qs[qi]['ev_mean']:>+15.4f}", end="")
            else:
                print(f" | {'N/A':>15s}", end="")
        print()

    # ── CityLearn KPIs ──
    print(f"\n  {'--- CITYLEARN KPIs ---':<30s}", end="")
    for n in names:
        print(f" | {'':>15s}", end="")
    print("  (<1.0 = better than baseline)")

    all_kpi_keys = set()
    for n in names:
        all_kpi_keys.update(all_results[n]["cl_kpis"].keys())

    for kpi in sorted(all_kpi_keys):
        print(f"  {kpi:<30s}", end="")
        for n in names:
            val = all_results[n]["cl_kpis"].get(kpi, float("nan"))
            print(f" | {val:>15.4f}", end="")
        print()

    print("\n" + "=" * 90)
    print("  Evaluation complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
