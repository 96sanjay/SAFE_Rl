#!/usr/bin/env python3
"""
Evaluation script for R28c SAC-Lag — GaussianSAC actor.

Runs deterministic 1-episode (8759 steps) evaluation and reports:
  - Per-constraint violation percentages (C0, C2, C3, C4)
  - C0 per-departure violation rate
  - C3 per-building breakdown
  - Reward component breakdown
  - Price-aware action analysis
  - CityLearn KPIs (normalized vs baseline)

Usage:
    cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
    conda activate citylearn
    python scripts/eval_r28c_sac.py [--checkpoint PATH] [--device cpu]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
PROJECT_ROOT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CKPT = (
    "runs/r28c_sac/5bld/"
    "SACLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-04-15-10-44-06/"
    "torch_save/epoch-85.pt"
)

# ---------------------------------------------------------------------------
COST_KEYS = [
    "cost_ev_departure",          # C0
    "cost_ev_dense",              # C1
    "cost_stems_battery",         # C2
    "cost_stems_building_power",  # C3
    "cost_stems_grid_power",      # C4
]
COST_LABELS = [
    "C0 (EV departure)",
    "C1 (EV dense)",
    "C2 (battery SoC)",
    "C3 (building power)",
    "C4 (grid power)",
]

REWARD_KEYS = [
    "r_eco", "r_sg", "r_sb", "r_ramp", "r_ren", "r_ev",
    "r_ev_guard", "r_barrier", "r_v2g_ctx", "r_ev_smart",
    "r_ev_slack_arb", "r_peak_shave", "r_load_shift", "r_grid_mild",
]


# ---------------------------------------------------------------------------
def set_env_vars():
    """Set env vars matching run_r28c_sac.sh exactly."""
    schema_path = os.path.join(
        PROJECT_ROOT,
        "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    )
    env_vars = {
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
        # Reward weights (R28c: 10 terms)
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
        # Disabled
        "STEMS_ALPHA_EV_GUARD": "0.0",
        "STEMS_ALPHA_GRID_MILD": "0.0",
        "STEMS_MU_ECONOMIC": "0.0",
        "STEMS_ALPHA_PEAK_SHAVE": "0.0",
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
    for k, v in env_vars.items():
        os.environ[k] = v
    print(f"[eval_r28c_sac] Schema: {schema_path}")


# ---------------------------------------------------------------------------
class SACActor(nn.Module):
    """Mirrors the omnisafe GaussianSACActor mean network.

    SAC outputs [mean, log_std] concatenated (dim = 2 * act_dim).
    For deterministic eval, we only use the mean half and apply tanh.
    Architecture: ReLU activations (SAC standard), not Tanh like PPO.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.act_dim = act_dim
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        # Output is 2*act_dim: [mean, log_std]
        layers.append(nn.Linear(prev, act_dim * 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        mean = out[..., :self.act_dim]
        return torch.tanh(mean)


# ---------------------------------------------------------------------------
def build_eval_env():
    """Build env with same wrapper chain as training (no temporal)."""
    import citylearn_safe.omni_env       # noqa: F401
    import citylearn_safe.omni_env_v2    # noqa: F401
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper

    base_env = make_base_env(central_agent=True)
    safety_env = CityLearnSafetyEnvV3(base_env)
    env = ForecastObsWrapper(safety_env, forecast_horizon=24)
    return env


# ---------------------------------------------------------------------------
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
        obs_t = torch.clamp((obs_t - mean[:obs_len]) / (std[:obs_len] + 1e-8),
                            -clip_val[:obs_len] if clip_val.dim() > 0 else -clip_val,
                            clip_val[:obs_len] if clip_val.dim() > 0 else clip_val)
    else:
        obs_t = torch.clamp((obs_t - mean) / (std + 1e-8), -clip_val, clip_val)
    return obs_t.unsqueeze(0)


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
def evaluate(args):
    set_env_vars()
    device = torch.device(args.device)

    ckpt_path = args.checkpoint or DEFAULT_CKPT
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
    if not os.path.isfile(ckpt_path):
        # Try to find the latest checkpoint
        ckpt_dir = os.path.dirname(ckpt_path)
        if os.path.isdir(ckpt_dir):
            pts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')],
                         key=lambda x: int(x.split('-')[-1].split('.')[0]) if x.split('-')[-1].split('.')[0].isdigit() else 0)
            if pts:
                ckpt_path = os.path.join(ckpt_dir, pts[-1])
                print(f"  Using latest checkpoint: {pts[-1]}")
        if not os.path.isfile(ckpt_path):
            # Search for any r28c_sac checkpoint
            import glob
            pattern = os.path.join(PROJECT_ROOT, "runs/r28c_sac/**/torch_save/*.pt")
            found = sorted(glob.glob(pattern, recursive=True))
            if found:
                ckpt_path = found[-1]
                print(f"  Found checkpoint: {ckpt_path}")
            else:
                raise FileNotFoundError(f"No checkpoint found for R28c SAC")

    # --- Build env ---
    print("\n=== Building evaluation environment ===")
    env = build_eval_env()
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")

    # --- Load model ---
    print(f"\n=== Loading checkpoint: {ckpt_path} ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # SAC checkpoint has 'pi' with 'net.*' keys directly
    pi_sd = ckpt["pi"]

    # Print checkpoint keys for debugging
    print(f"  Checkpoint pi keys: {list(pi_sd.keys())[:10]}")

    model = SACActor(obs_dim, act_dim, hidden_sizes=[256, 256])

    # SAC keys are already 'net.0.weight' etc — load directly
    # Filter out non-net keys (like '_log2')
    net_sd = {k: v for k, v in pi_sd.items() if k.startswith("net.")}
    if not net_sd:
        # Maybe keys have a different prefix
        print(f"  WARNING: No 'net.*' keys found. Available: {list(pi_sd.keys())}")
        # Try loading as-is
        net_sd = pi_sd

    missing, unexpected = model.load_state_dict(net_sd, strict=False)
    print(f"  Loaded {len(net_sd)} tensors (missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"  Missing: {missing}")

    obs_norm = ckpt.get("obs_normalizer", None)
    if obs_norm:
        print(f"  Obs normalizer: mean shape={obs_norm['_mean'].shape}")

    model = model.to(device).eval()

    # --- Run episode ---
    print(f"\n=== Running deterministic evaluation (1 episode, 8759 steps) ===")
    obs, info = env.reset()

    step_rewards = []
    step_costs = defaultdict(list)
    step_violations = defaultdict(list)
    step_rewards_detail = defaultdict(list)
    all_actions = []
    ev_departure_costs = []
    ev_departure_events = []
    c3_per_building_violations = []
    total_cost = 0.0
    ep_len = 0
    t_start = time.time()

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

        total_cost += float(info.get("cost", 0.0))

        for rk in REWARD_KEYS:
            step_rewards_detail[rk].append(float(info.get(rk, 0.0)))

        ev_departure_costs.append(float(info.get("cost_ev_departure", 0.0)))
        ev_departure_events.append(int(info.get("ev_departure_departures", 0)))
        c3_per_building_violations.append(float(info.get("building_power_violation_count", 0.0)))

        if (step + 1) % 2000 == 0:
            elapsed = time.time() - t_start
            print(f"  Step {step+1:>5d}/8759 | EpRet: {sum(step_rewards):>10.1f} | FPS: {(step+1)/elapsed:.0f}")

        if terminated or truncated:
            break

    elapsed = time.time() - t_start
    print(f"\n  Episode: {ep_len} steps in {elapsed:.1f}s ({ep_len/elapsed:.0f} FPS)")

    # --- CityLearn KPIs ---
    citylearn_env = get_citylearn_env(env)
    cl_kpis = {}
    if citylearn_env is not None and hasattr(citylearn_env, "evaluate"):
        try:
            kpi_df = citylearn_env.evaluate()
            if kpi_df is not None:
                for _, row in kpi_df.iterrows():
                    name = row.get("name", row.get("cost_function", "unknown"))
                    val = row.get("value", row.get("cost", 0.0))
                    if name and val is not None:
                        cl_kpis[str(name)] = float(val)
        except Exception as e:
            print(f"  WARNING: CityLearn evaluate() failed: {e}")

    # ====================================================================
    # RESULTS
    # ====================================================================
    ep_ret = sum(step_rewards)
    ckpt_name = os.path.basename(ckpt_path)
    print("\n")
    print("=" * 76)
    print(f"  R28c SAC EVALUATION RESULTS ({ckpt_name}, 10-term reward)")
    print("=" * 76)
    print(f"\n  Episode Return:       {ep_ret:>12.1f}")
    print(f"  Episode Length:       {ep_len:>12d}")
    print(f"  Mean Step Reward:     {np.mean(step_rewards):>12.4f}")
    print(f"  Total CMDP Cost:      {total_cost:>12.1f}")

    # --- Per-constraint table ---
    print(f"\n  " + "-" * 72)
    print(f"  {'Constraint':<25s} | {'Total Cost':>12s} | {'Limit':>8s} | {'Viol %':>8s} | "
          f"{'Mean/step':>10s} | {'Max/step':>10s}")
    print("  " + "-" * 72)
    limits = [100, 999999, 3500, 18000, 13000]
    for key, label, lim in zip(COST_KEYS, COST_LABELS, limits):
        costs = step_costs[key]
        viols = step_violations[key]
        total_c = sum(costs)
        viol_pct = 100.0 * np.mean(viols)
        mean_c = np.mean(costs)
        max_c = np.max(costs)
        status = "PASS" if total_c <= lim else "FAIL"
        print(f"  {label:<25s} | {total_c:>12.1f} | {lim:>8d} | {viol_pct:>7.1f}% | "
              f"{mean_c:>10.4f} | {max_c:>10.4f} | {status}")
    print("  " + "-" * 72)

    # --- C0: Per-departure ---
    total_departures = sum(ev_departure_events)
    c0_deficit_at_departure = sum(
        1 for dep, cost in zip(ev_departure_events, step_costs[COST_KEYS[0]])
        if dep > 0 and cost > 0
    )
    print(f"\n  C0 EV DEPARTURE METRICS:")
    print(f"    Total departures:              {total_departures}")
    print(f"    Departures with deficit:        {c0_deficit_at_departure}")
    if total_departures > 0:
        print(f"    Violation rate (per departure): {100.0 * c0_deficit_at_departure / total_departures:.1f}%")
        total_c0 = sum(step_costs[COST_KEYS[0]])
        print(f"    Mean deficit per departure:     {total_c0 / total_departures:.4f}")

    # --- C3: Per-building ---
    c3_arr = np.array(c3_per_building_violations)
    c3_any = int(np.sum(c3_arr > 0))
    c3_total_bld = int(np.sum(c3_arr))
    n_bld = 5
    print(f"\n  C3 BUILDING POWER - PER-BUILDING:")
    print(f"    Steps with ANY building violating: {c3_any} / {ep_len} ({100.0 * c3_any / ep_len:.1f}%)")
    print(f"    Total building-violations:          {c3_total_bld} / {ep_len * n_bld}")
    print(f"    Building-violation rate:             {100.0 * c3_total_bld / (ep_len * n_bld):.1f}%")
    print(f"    Mean buildings violating per step:   {c3_arr.mean():.2f} / {n_bld}")
    for nb in range(n_bld + 1):
        cnt = int(np.sum(c3_arr == nb))
        if cnt > 0:
            print(f"      {nb} buildings violating: {cnt} steps ({100.0 * cnt / ep_len:.1f}%)")

    # --- Reward component breakdown ---
    print(f"\n  " + "-" * 72)
    print(f"  REWARD COMPONENT BREAKDOWN")
    print(f"  " + "-" * 72)
    print(f"  {'Component':<20s} | {'Total':>10s} | {'Mean/step':>10s} | "
          f"{'Std':>8s} | {'Min':>8s} | {'Max':>8s} | {'% of |R|':>8s}")
    print(f"  " + "-" * 72)
    total_abs = sum(abs(np.array(step_rewards_detail[rk]).sum())
                    for rk in REWARD_KEYS if any(v != 0 for v in step_rewards_detail[rk]))
    for rk in REWARD_KEYS:
        vals = step_rewards_detail[rk]
        if all(v == 0.0 for v in vals):
            continue
        arr = np.array(vals)
        frac = 100.0 * abs(arr.sum()) / max(total_abs, 1e-8)
        print(f"  {rk:<20s} | {arr.sum():>+10.1f} | {arr.mean():>+10.4f} | "
              f"{arr.std():>8.4f} | {arr.min():>+8.4f} | {arr.max():>+8.4f} | {frac:>7.1f}%")
    zero_keys = [rk for rk in REWARD_KEYS if all(v == 0 for v in step_rewards_detail[rk])]
    if zero_keys:
        print(f"\n  Disabled: {', '.join(zero_keys)}")
    print(f"  " + "-" * 72)

    # --- Price-aware analysis ---
    print(f"\n  PRICE-AWARE ACTION ANALYSIS")
    print(f"  " + "-" * 72)
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
        bounds = [0, n//4, n//2, 3*n//4, n]
        labels = ["Q1 (cheapest)", "Q2", "Q3", "Q4 (expensive)"]
        print(f"  {'Quartile':<18s} | {'Price Range':>14s} | {'Batt Act':>10s} | {'EV Act':>10s}")
        print(f"  " + "-" * 60)
        for qi in range(4):
            idx = price_sorted[bounds[qi]:bounds[qi+1]]
            p = pricing[idx]
            print(f"  {labels[qi]:<18s} | {p.min():>6.3f}-{p.max():>6.3f} | "
                  f"{batt_actions[idx].mean():>+10.4f} | {ev_actions[idx].mean():>+10.4f}")

        batt_corr = np.corrcoef(pricing, batt_actions)[0, 1]
        ev_corr = np.corrcoef(pricing, ev_actions)[0, 1]
        print(f"\n  Price correlation:  battery={batt_corr:>+.4f}  EV={ev_corr:>+.4f}")
        print(f"  (negative = charge cheap / discharge expensive = GOOD)")
    except Exception as e:
        print(f"  Price analysis failed: {e}")

    # --- CityLearn KPIs ---
    if cl_kpis:
        print(f"\n  " + "-" * 72)
        print(f"  CITYLEARN KPIs (ratio to no-control baseline, <1.0 = improvement)")
        print(f"  " + "-" * 72)
        for name, val in sorted(cl_kpis.items()):
            marker = " PASS" if val < 1.0 else " FAIL" if val > 1.0 else ""
            print(f"    {name:<45s}: {val:>8.4f}{marker}")
    else:
        print(f"\n  CityLearn KPIs: not available")

    print("\n" + "=" * 76)
    print(f"  R28c SAC evaluation complete.")
    print("=" * 76)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate R28c SAC checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: latest)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
