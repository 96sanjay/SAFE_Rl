#!/usr/bin/env python3
"""
Evaluation script for R26h / R26i checkpoints.

Loads the R26h or R26i STEMS V3 encoder checkpoint (2-layer transformer,
mean pooling), runs a deterministic 1-episode (8759 steps) evaluation, and
reports per-constraint violation percentages, totals, reward component
breakdown, price-aware action analysis, and CityLearn KPIs.

Usage:
    cd <PROJECT_ROOT>
    conda activate citylearn
    python scripts/evaluation/eval_r26hi.py --run r26h [--checkpoint PATH] [--device cpu]
    python scripts/evaluation/eval_r26hi.py --run r26i [--checkpoint PATH] [--device cpu]
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
# Path setup: derive PROJECT_ROOT from this file's location
# (this file lives at <PROJECT_ROOT>/scripts/evaluation/eval_r26hi.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Default checkpoint paths (relative to PROJECT_ROOT)
# ---------------------------------------------------------------------------
DEFAULT_CKPT_R26H = (
    "runs/r26h_forecast_lean/5bld/"
    "PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-04-11-00-53-23/"
    "torch_save/epoch-95.pt"
)
DEFAULT_CKPT_R26I = (
    "runs/r26i_forecast_full/5bld/"
    "PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-04-11-00-53-28/"
    "torch_save/epoch-110.pt"
)

# ---------------------------------------------------------------------------
# Per-constraint cost keys (matches ppo_lag_grads.py / ppo_lag_multi.py)
# ---------------------------------------------------------------------------
COST_KEYS = [
    "cost_ev_departure",          # C0: EV departure SoC deficit
    "cost_ev_dense",              # C1: EV charging dense
    "cost_stems_battery",         # C2: Battery SoC band violation
    "cost_stems_building_power",  # C3: Building power capacity
    "cost_stems_grid_power",      # C4: Grid power capacity
]
COST_LABELS = [
    "C0 (EV departure)",
    "C1 (EV dense)",
    "C2 (battery SoC)",
    "C3 (building power)",
    "C4 (grid power)",
]

# ---------------------------------------------------------------------------
# Reward component keys to track
# ---------------------------------------------------------------------------
REWARD_KEYS = [
    "r_eco", "r_sg", "r_sb", "r_ramp", "r_ren", "r_ev",
    "r_ev_guard", "r_barrier", "r_trajectory",
    "r_traj_batt", "r_traj_ev", "traj_forecast_signal", "traj_forecast_mean",
]

# ---------------------------------------------------------------------------
# Run-specific reward configurations
# ---------------------------------------------------------------------------
REWARD_COMMON = {
    "STEMS_ALPHA_GRID": "2.0",
    "STEMS_ALPHA_BUILD": "3.0",
    "STEMS_XI_RENEWABLE": "0.3",
    "STEMS_ALPHA_BARRIER": "1.0",
    "STEMS_ALPHA_TRAJECTORY": "2.0",
    "STEMS_TRAJ_EV_WEIGHT": "1.0",
    "STEMS_TRAJ_FORECAST_HOURS": "24",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    # Disabled terms
    "STEMS_BETA_RAMP": "0.0",
    "STEMS_ALPHA_V2G_CONTEXT": "0.0",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "STEMS_ALPHA_LOAD_SHIFT": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.0",
    "STEMS_ALPHA_EV_SOLAR": "0.0",
    "STEMS_ALPHA_SOLAR_STORE": "0.0",
    "STEMS_EV_SLACK_ARB_SCALE": "0.0",
    "STEMS_ALPHA_HEADROOM": "0.0",
    "STEMS_ALPHA_GRID_PENALTY": "0.0",
    "STEMS_ALPHA_PRICE_ARB": "0.0",
    "STEMS_ALPHA_NEC_SIGN": "0.0",
}

REWARD_R26H = {
    **REWARD_COMMON,
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_LAMBDA_EV": "0.0",
    "STEMS_ALPHA_EV_GUARD": "0.0",
}

REWARD_R26I = {
    **REWARD_COMMON,
    "STEMS_MU_ECONOMIC": "0.3",
    "STEMS_LAMBDA_EV": "0.5",
    "STEMS_ALPHA_EV_GUARD": "1.5",
}


# ---------------------------------------------------------------------------
# Set environment variables BEFORE importing any project modules
# ---------------------------------------------------------------------------
def set_env_vars(run_name: str):
    schema_path = os.path.join(
        PROJECT_ROOT,
        "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    )
    env_vars = {
        "CITYLEARN_SCHEMA": schema_path,
        "CITYLEARN_TEMPORAL_RICH": "1",
        "CITYLEARN_TEMPORAL_WINDOW": "12",
        "STEMS_ENCODER_VERSION": "v3",
        "CITYLEARN_BATT_CLAMP": "0",
        "CITYLEARN_EV_CLAMP": "0",
        "CITYLEARN_C3_COST": "0.1",
        "CITYLEARN_C4_COST": "5.0",
        # Suppress noisy debug prints during eval
        "CITYLEARN_KPI_FLUSH_EVERY_STEP": "0",
        "CITYLEARN_DEBUG_ACTION_CLIP": "0",
    }

    # Apply run-specific reward env vars
    reward_cfg = REWARD_R26H if run_name == "r26h" else REWARD_R26I
    env_vars.update(reward_cfg)

    for k, v in env_vars.items():
        os.environ[k] = v

    print(f"[eval_{run_name}] Schema: {schema_path}")
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    # Print reward config summary
    print(f"[eval_{run_name}] Reward config ({run_name}):")
    for k in sorted(reward_cfg.keys()):
        val = reward_cfg[k]
        if float(val) != 0.0:
            print(f"  {k}={val}")


# ---------------------------------------------------------------------------
# STEMSMeanNet (mirrors train_multi_lag_stems.py exactly)
# ---------------------------------------------------------------------------
class STEMSMeanNet(nn.Module):
    def __init__(self, encoder, act_dim: int, encoder_obs_dim: int):
        super().__init__()
        self.encoder = encoder
        self.encoder_obs_dim = encoder_obs_dim
        self.action_head = nn.Sequential(
            nn.Linear(encoder.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        enc_obs = (
            obs[:, : self.encoder_obs_dim]
            if obs.size(-1) > self.encoder_obs_dim
            else obs
        )
        features = self.encoder(enc_obs)
        return torch.tanh(self.action_head(features))


# ---------------------------------------------------------------------------
# Build environment (same wrapper chain as training)
# ---------------------------------------------------------------------------
def build_eval_env():
    """Build the evaluation environment with the same wrapper chain as training."""
    # Register environments
    import citylearn_safe.cmdp_env    # noqa: F401  (registers env)

    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env import CityLearnSafetyEnv
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.temporal_obs_wrapper import TemporalHistoryWrapper

    # 1. Base CityLearn env
    base_env = make_base_env(central_agent=True)

    # 2. Safety wrapper
    safety_env = CityLearnSafetyEnv(base_env)

    # 3. Forecast wrapper
    forecast_env = ForecastObsWrapper(safety_env, forecast_horizon=24)

    # 4. Temporal history wrapper (rich temporal, applied when CITYLEARN_TEMPORAL_RICH=1)
    from citylearn_safe.schema_index import build_index, _CACHE
    import citylearn_safe.schema_index as si
    si._CACHE = None

    # Get obs_index for history config
    obs_index = build_index(safety_env, expected_buildings=5)

    from citylearn_safe.temporal_obs_wrapper import build_rich_history_config
    rich_cfg = build_rich_history_config(obs_index, num_buildings=5)
    history_indices = rich_cfg["history_indices"]

    temporal_window = int(os.environ.get("CITYLEARN_TEMPORAL_WINDOW", "12"))
    env = TemporalHistoryWrapper(
        forecast_env,
        history_indices=history_indices,
        window_size=temporal_window,
    )

    return env, obs_index, rich_cfg


# ---------------------------------------------------------------------------
# Build encoder and mean_net, load checkpoint
# ---------------------------------------------------------------------------
def build_and_load_model(obs_index, rich_cfg, env, checkpoint_path, device, run_name):
    """Build STEMSEncoder + STEMSMeanNet and load checkpoint weights."""
    from citylearn_safe.stems_encoder_5bld import build_node_indices
    from citylearn_safe.stems_encoder import STEMSEncoder

    num_buildings = 5
    temporal_window = int(os.environ.get("CITYLEARN_TEMPORAL_WINDOW", "12"))

    node_info = build_node_indices(obs_index, num_buildings)
    print(f"[eval_{run_name}] base_obs_dim={node_info['base_obs_dim']}")

    features_per_step = rich_cfg["features_per_step"]
    features_per_node = rich_cfg["features_per_node"]
    per_node_map = rich_cfg["per_node_map"]
    ev_mask = rich_cfg["history_ev_mask"]

    # Get obs_dim and act_dim from env
    obs_space = env.observation_space
    obs_dim = int(obs_space.shape[0])
    act_space = env.action_space
    act_dim = int(act_space.shape[0])

    # obs_dim_v3: base obs (after forecast wrapper) + temporal history
    # The env already includes temporal history, so obs_dim IS obs_dim_v3
    obs_dim_v3 = obs_dim

    print(
        f"[eval_{run_name}] obs_dim={obs_dim}, act_dim={act_dim}, "
        f"features/step={features_per_step}, features/node={features_per_node}"
    )

    stems_kwargs = dict(
        obs_dim=obs_dim_v3,
        node_info=node_info,
        num_buildings=num_buildings,
        hidden_dim=64,
        global_hidden=32,
        temporal_window=temporal_window,
        temporal_features_per_step=features_per_step,
        temporal_hidden=32,
        temporal_heads=4,
        num_gcn_layers=3,
        dropout=0.1,
        output_dim=256,
        temporal_features_per_node=features_per_node,
        per_node_history_map=per_node_map,
        history_ev_mask=ev_mask,
        temporal_num_layers=2,       # R26h/R26i use 2-layer transformer
        temporal_pool_mode="mean",   # R26h/R26i use mean pooling
    )
    encoder = STEMSEncoder(**stems_kwargs)
    mean_net = STEMSMeanNet(encoder, act_dim, encoder_obs_dim=obs_dim_v3)

    total_params = sum(p.numel() for p in mean_net.parameters())
    print(f"[eval_{run_name}] STEMSMeanNet params: {total_params:,}")

    # --- Load checkpoint ---
    print(f"[eval_{run_name}] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Load policy weights (pi state dict with mean.encoder.* keys)
    if "pi" in ckpt:
        pi_sd = ckpt["pi"]
        # Filter to only mean_net keys (strip 'mean.' prefix if present)
        mean_sd = {}
        for k, v in pi_sd.items():
            if k.startswith("mean."):
                mean_sd[k[len("mean."):]] = v
        if mean_sd:
            missing, unexpected = mean_net.load_state_dict(mean_sd, strict=False)
            if missing:
                print(f"  WARNING: Missing keys in mean_net: {missing}")
            if unexpected:
                print(f"  WARNING: Unexpected keys: {unexpected}")
            print(f"  Loaded {len(mean_sd)} weight tensors from pi['mean.*']")
        else:
            # Try loading directly (keys might already match)
            missing, unexpected = mean_net.load_state_dict(pi_sd, strict=False)
            print(f"  Loaded pi state dict directly (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        raise KeyError("Checkpoint missing 'pi' key. Available keys: " + str(list(ckpt.keys())))

    # Load obs normalizer
    obs_norm = None
    if "obs_normalizer" in ckpt:
        obs_norm = ckpt["obs_normalizer"]
        norm_keys = list(obs_norm.keys())
        print(f"  Obs normalizer loaded with keys: {norm_keys}")
        for k in ["_mean", "_std", "_clip"]:
            if k in obs_norm:
                t = obs_norm[k]
                if hasattr(t, "shape"):
                    print(f"    {k}: shape={t.shape}, dtype={t.dtype}")
    else:
        print("  WARNING: No obs_normalizer in checkpoint. Using raw observations.")

    mean_net = mean_net.to(device)
    mean_net.eval()

    return mean_net, obs_norm, act_dim


# ---------------------------------------------------------------------------
# Normalize observation using checkpoint's obs normalizer
# ---------------------------------------------------------------------------
def normalize_obs(obs_np, obs_norm, device):
    """Apply the obs normalizer from the checkpoint: (obs - mean) / (std + eps), clipped."""
    obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
    if obs_norm is None:
        return obs_t.unsqueeze(0)

    mean = obs_norm["_mean"].to(device).float()
    std = obs_norm["_std"].to(device).float()
    clip_val = obs_norm.get("_clip", torch.tensor(10.0)).to(device).float()

    # Handle shape mismatches (obs might be longer than normalizer if temporal was added)
    obs_len = obs_t.shape[0]
    norm_len = mean.shape[0]

    if obs_len > norm_len:
        # Normalize only the first norm_len dims, pass the rest through raw
        obs_base = obs_t[:norm_len]
        obs_extra = obs_t[norm_len:]
        normed_base = torch.clamp(
            (obs_base - mean) / (std + 1e-8), -clip_val, clip_val
        )
        obs_normed = torch.cat([normed_base, obs_extra], dim=0)
    elif obs_len < norm_len:
        # Should not happen, but handle gracefully
        obs_normed = torch.clamp(
            (obs_t - mean[:obs_len]) / (std[:obs_len] + 1e-8),
            -clip_val if clip_val.dim() == 0 else -clip_val[:obs_len],
            clip_val if clip_val.dim() == 0 else clip_val[:obs_len],
        )
    else:
        obs_normed = torch.clamp(
            (obs_t - mean) / (std + 1e-8), -clip_val, clip_val
        )

    return obs_normed.unsqueeze(0)  # [1, obs_dim]


# ---------------------------------------------------------------------------
# Unwrap CityLearn env for KPI evaluation
# ---------------------------------------------------------------------------
def get_citylearn_env(env):
    """Walk wrapper chain to find the CityLearnEnv with .buildings and .evaluate()."""
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
# Price-aware action analysis
# ---------------------------------------------------------------------------
def price_action_analysis(env, all_actions, ep_len, run_name):
    """Analyze correlation between electricity price and agent actions."""
    citylearn_env = get_citylearn_env(env)
    if citylearn_env is None:
        print(f"\n  [price analysis] Could not unwrap CityLearn env -- skipping.")
        return

    try:
        pricing = np.array(citylearn_env.buildings[0].pricing.electricity_pricing)
    except Exception as e:
        print(f"\n  [price analysis] Could not access pricing data: {e}")
        return

    # Trim pricing to episode length
    pricing = pricing[:ep_len]
    actions_arr = np.array(all_actions[:ep_len])  # shape: [ep_len, act_dim]

    if len(pricing) != len(actions_arr):
        print(
            f"\n  [price analysis] Length mismatch: pricing={len(pricing)}, "
            f"actions={len(actions_arr)} -- skipping."
        )
        return

    # Identify action indices for batteries and EVs from env
    safety_env = env
    while hasattr(safety_env, 'env'):
        if hasattr(safety_env, '_ev_charger_action_indices'):
            break
        safety_env = safety_env.env
    ev_indices = getattr(safety_env, '_ev_charger_action_indices', [])
    act_dim = actions_arr.shape[1]
    batt_indices = [i for i in range(act_dim) if i not in ev_indices]
    print(f"\n  [Price-Action] Battery indices: {batt_indices}, EV indices: {ev_indices}")

    batt_actions = actions_arr[:, batt_indices].mean(axis=1) if batt_indices else np.zeros(len(actions_arr))
    ev_actions = actions_arr[:, ev_indices].mean(axis=1) if ev_indices else np.zeros(len(actions_arr))

    # Quartile analysis
    price_sorted_idx = np.argsort(pricing)
    n = len(pricing)
    q1_end = n // 4
    q4_start = 3 * n // 4

    cheapest_idx = price_sorted_idx[:q1_end]
    expensive_idx = price_sorted_idx[q4_start:]

    quartile_boundaries = [0, q1_end, n // 2, q4_start, n]
    quartile_labels = ["Q1 (cheapest)", "Q2", "Q3", "Q4 (expensive)"]

    print(f"\n  " + "-" * 68)
    print(f"  PRICE-AWARE ACTION ANALYSIS ({run_name.upper()})")
    print(f"  " + "-" * 68)
    print(
        f"  {'Price Quartile':<20s} | {'Price Range':>14s} | "
        f"{'Mean Batt Act':>13s} | {'Mean EV Act':>11s}"
    )
    print(f"  " + "-" * 68)

    for qi in range(4):
        q_start = quartile_boundaries[qi]
        q_end = quartile_boundaries[qi + 1]
        q_idx = price_sorted_idx[q_start:q_end]

        q_prices = pricing[q_idx]
        q_batt = batt_actions[q_idx]
        q_ev = ev_actions[q_idx]

        price_lo = q_prices.min()
        price_hi = q_prices.max()

        print(
            f"  {quartile_labels[qi]:<20s} | "
            f"{price_lo:>6.3f}-{price_hi:>6.3f} | "
            f"{q_batt.mean():>+13.4f} | "
            f"{q_ev.mean():>+11.4f}"
        )

    print(f"  " + "-" * 68)

    # Correlation
    try:
        batt_corr = np.corrcoef(pricing, batt_actions)[0, 1]
        ev_corr = np.corrcoef(pricing, ev_actions)[0, 1]
    except Exception:
        batt_corr = float("nan")
        ev_corr = float("nan")

    print(f"\n  Correlation (price vs battery discharge): {batt_corr:>+.4f}")
    print(f"  Correlation (price vs EV action):          {ev_corr:>+.4f}")

    # Cheapest vs expensive summary
    cheap_batt = batt_actions[cheapest_idx].mean()
    exp_batt = batt_actions[expensive_idx].mean()
    cheap_ev = ev_actions[cheapest_idx].mean()
    exp_ev = ev_actions[expensive_idx].mean()

    print(f"\n  Cheapest 25% hours:  batt={cheap_batt:>+.4f}  ev={cheap_ev:>+.4f}")
    print(f"  Expensive 25% hours: batt={exp_batt:>+.4f}  ev={exp_ev:>+.4f}")
    print(
        f"  Delta (expensive - cheap): batt={exp_batt - cheap_batt:>+.4f}  "
        f"ev={exp_ev - cheap_ev:>+.4f}"
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate(args):
    run_name = args.run
    set_env_vars(run_name)

    device = torch.device(args.device)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Use run-specific default
        checkpoint_path = DEFAULT_CKPT_R26H if run_name == "r26h" else DEFAULT_CKPT_R26I
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # --- Build environment ---
    print("\n=== Building evaluation environment ===")
    env, obs_index, rich_cfg = build_eval_env()

    # --- Build model and load weights ---
    print("\n=== Building model and loading checkpoint ===")
    mean_net, obs_norm, act_dim = build_and_load_model(
        obs_index, rich_cfg, env, checkpoint_path, device, run_name
    )

    # --- Run evaluation ---
    print(f"\n=== Running deterministic evaluation ({run_name.upper()}, 1 episode, 8759 steps) ===")
    obs, info = env.reset()

    # Tracking
    step_rewards = []
    step_costs = defaultdict(list)       # cost_key -> list of per-step costs
    step_violations = defaultdict(list)  # cost_key -> list of 0/1 per step
    step_rewards_detail = defaultdict(list)  # reward_key -> list of per-step values
    all_actions = []                     # list of action arrays per step
    ev_departure_costs = []              # per-step cost_ev_departure for departure analysis
    ev_departure_events = []             # per-step departure count (0 or positive int)
    c3_per_building_violations = []      # per-step: how many buildings violated C3
    total_cost = 0.0
    ep_len = 0

    t_start = time.time()

    for step in range(8760):
        # Normalize obs
        obs_normed = normalize_obs(obs, obs_norm, device)

        # Get deterministic action
        with torch.no_grad():
            action = mean_net(obs_normed)  # [1, act_dim]
        action_np = action.squeeze(0).cpu().numpy()

        # Clip action to env bounds
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

        # Step
        obs, reward, terminated, truncated, info = env.step(action_np)
        ep_len += 1
        step_rewards.append(reward)
        all_actions.append(action_np.copy())

        # Track per-constraint costs
        for key in COST_KEYS:
            cost_val = float(info.get(key, 0.0))
            step_costs[key].append(cost_val)
            step_violations[key].append(1.0 if cost_val > 0.0 else 0.0)

        total_cost += float(info.get("cost", 0.0))

        # Track reward components
        for rk in REWARD_KEYS:
            step_rewards_detail[rk].append(float(info.get(rk, 0.0)))

        # Track EV departure cost and departure events
        ev_dep_cost = float(info.get("cost_ev_departure", 0.0))
        ev_departure_costs.append(ev_dep_cost)
        ev_departure_events.append(int(info.get("ev_departure_departures", 0)))

        # Track per-building C3 violations
        c3_per_building_violations.append(float(info.get("building_power_violation_count", 0.0)))

        # Progress
        if (step + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            fps = (step + 1) / elapsed
            print(
                f"  Step {step + 1:>5d}/8759 | "
                f"Reward: {reward:>8.2f} | "
                f"EpRet so far: {sum(step_rewards):>10.1f} | "
                f"FPS: {fps:.0f}"
            )

        if terminated or truncated:
            break

    elapsed = time.time() - t_start
    print(f"\n  Episode finished: {ep_len} steps in {elapsed:.1f}s ({ep_len/elapsed:.0f} FPS)")

    # --- CityLearn KPIs ---
    citylearn_env = get_citylearn_env(env)
    cl_kpis = {}
    if citylearn_env is not None and hasattr(citylearn_env, "evaluate"):
        try:
            kpi_df = citylearn_env.evaluate()
            if kpi_df is not None:
                # Convert DataFrame to dict for display
                for _, row in kpi_df.iterrows():
                    name = row.get("name", row.get("cost_function", "unknown"))
                    val = row.get("value", row.get("cost", 0.0))
                    if name and val is not None:
                        cl_kpis[str(name)] = float(val)
        except Exception as e:
            print(f"  WARNING: CityLearn evaluate() failed: {e}")

    # --- Print results ---
    print("\n")
    print("=" * 72)
    print(f"  R26h/R26i EVALUATION RESULTS  [{run_name.upper()}]")
    print("=" * 72)

    # Reward summary
    ep_ret = sum(step_rewards)
    print(f"\n  Run:                  {run_name.upper():>12s}")
    print(f"  Episode Return:       {ep_ret:>12.1f}")
    print(f"  Episode Length:       {ep_len:>12d}")
    print(f"  Mean Step Reward:     {np.mean(step_rewards):>12.4f}")
    print(f"  Total CMDP Cost:      {total_cost:>12.1f}")

    # Per-constraint table
    print("\n  " + "-" * 68)
    print(
        f"  {'Constraint':<25s} | {'Total Cost':>12s} | {'Viol %':>8s} | "
        f"{'Mean/step':>10s} | {'Max/step':>10s}"
    )
    print("  " + "-" * 68)

    for i, (key, label) in enumerate(zip(COST_KEYS, COST_LABELS)):
        costs = step_costs[key]
        viols = step_violations[key]
        total_c = sum(costs)
        viol_pct = 100.0 * np.mean(viols) if viols else 0.0
        mean_c = np.mean(costs) if costs else 0.0
        max_c = np.max(costs) if costs else 0.0

        print(
            f"  {label:<25s} | {total_c:>12.1f} | {viol_pct:>7.1f}% | "
            f"{mean_c:>10.4f} | {max_c:>10.4f}"
        )

    print("  " + "-" * 68)

    # Violation step counts
    print(f"\n  Per-constraint violation step counts (out of {ep_len}):")
    for i, (key, label) in enumerate(zip(COST_KEYS, COST_LABELS)):
        viols = step_violations[key]
        n_viol = int(sum(viols))
        print(f"    {label:<25s}: {n_viol:>5d} steps ({100.0 * n_viol / max(1, ep_len):.1f}%)")

    # --- C0: Correct violation rate using departure events as denominator ---
    total_departures = sum(ev_departure_events)
    c0_viol_steps = int(sum(step_violations[COST_KEYS[0]]))
    # Steps where a departure had deficit
    c0_deficit_at_departure = sum(
        1 for dep, cost in zip(ev_departure_events, step_costs[COST_KEYS[0]])
        if dep > 0 and cost > 0
    )
    print(f"\n  C0 DEPARTURE-BASED METRICS:")
    print(f"    Total departure events:       {total_departures}")
    print(f"    Departures with deficit:       {c0_deficit_at_departure}")
    if total_departures > 0:
        print(f"    Violation rate (per departure): {100.0 * c0_deficit_at_departure / total_departures:.1f}%")
        print(f"    Mean deficit at departure:     {sum(step_costs[COST_KEYS[0]]) / total_departures:.4f}")
    else:
        print(f"    Violation rate (per departure): N/A (no departures)")
    print(f"    (Previously reported {c0_viol_steps}/{ep_len} = {100.0 * c0_viol_steps / max(1, ep_len):.1f}% was per-timestep, NOT per-departure)")

    # --- C3: Per-building breakdown ---
    c3_viols_arr = np.array(c3_per_building_violations)
    c3_any_step = int(np.sum(c3_viols_arr > 0))
    c3_total_bld_violations = int(np.sum(c3_viols_arr))
    n_buildings = 5  # known from schema
    print(f"\n  C3 BUILDING POWER - PER-BUILDING BREAKDOWN:")
    print(f"    Steps with ANY building violating: {c3_any_step} / {ep_len} ({100.0 * c3_any_step / max(1, ep_len):.1f}%)")
    print(f"    Total building-violations:          {c3_total_bld_violations} (sum across all buildings & steps)")
    print(f"    Max possible building-violations:   {ep_len * n_buildings} ({ep_len} steps x {n_buildings} buildings)")
    print(f"    Building-violation rate:             {100.0 * c3_total_bld_violations / max(1, ep_len * n_buildings):.1f}%")
    print(f"    Mean buildings violating per step:   {c3_viols_arr.mean():.2f} / {n_buildings}")
    # Distribution of how many buildings violate per step
    for n_bld in range(n_buildings + 1):
        count = int(np.sum(c3_viols_arr == n_bld))
        if count > 0:
            print(f"      {n_bld} buildings violating: {count} steps ({100.0 * count / max(1, ep_len):.1f}%)")

    # --- Reward component breakdown ---
    print(f"\n  " + "-" * 68)
    print(f"  REWARD COMPONENT BREAKDOWN ({run_name.upper()})")
    print(f"  " + "-" * 68)
    print(
        f"  {'Component':<25s} | {'Total':>12s} | {'Mean/step':>10s} | "
        f"{'Std/step':>10s} | {'Min':>10s} | {'Max':>10s}"
    )
    print(f"  " + "-" * 68)

    for rk in REWARD_KEYS:
        vals = step_rewards_detail[rk]
        if not vals or all(v == 0.0 for v in vals):
            # Skip keys that are all zero (disabled terms)
            continue
        arr = np.array(vals)
        print(
            f"  {rk:<25s} | {arr.sum():>12.2f} | {arr.mean():>10.4f} | "
            f"{arr.std():>10.4f} | {arr.min():>10.4f} | {arr.max():>10.4f}"
        )

    # Also show disabled (all-zero) keys compactly
    zero_keys = [rk for rk in REWARD_KEYS if all(v == 0.0 for v in step_rewards_detail[rk])]
    if zero_keys:
        print(f"\n  Disabled (all-zero) reward terms: {', '.join(zero_keys)}")

    print(f"  " + "-" * 68)

    # --- EV departure analysis ---
    ev_dep_arr = np.array(ev_departure_costs)
    n_departures_with_violation = int(np.sum(ev_dep_arr > 0.0))
    total_ev_dep_cost = float(ev_dep_arr.sum())
    print(f"\n  EV Departure Analysis ({run_name.upper()}):")
    print(f"    Steps with departure violations: {n_departures_with_violation} / {ep_len}")
    print(f"    Total departure cost:            {total_ev_dep_cost:.4f}")
    if n_departures_with_violation > 0:
        violation_vals = ev_dep_arr[ev_dep_arr > 0.0]
        print(f"    Mean violation (when > 0):        {violation_vals.mean():.4f}")
        print(f"    Max violation:                    {violation_vals.max():.4f}")

    # --- Price-aware action analysis ---
    price_action_analysis(env, all_actions, ep_len, run_name)

    # CityLearn KPIs
    if cl_kpis:
        print(f"\n  CityLearn KPIs:")
        for name, val in sorted(cl_kpis.items()):
            print(f"    {name:<40s}: {val:>10.4f}")
    else:
        print(f"\n  CityLearn KPIs: not available")

    # Extra info from last step
    extra_keys = [
        "reward_type",
        "reward_stems_total",
        "reward_economic",
        "reward_stability",
        "reward_renewable",
        "grid_import_kwh",
        "battery_soc_violation_frac",
        "building_power_violation",
        "grid_power_violation",
    ]
    print(f"\n  Last step info snapshot:")
    for k in extra_keys:
        if k in info:
            val = info[k]
            if isinstance(val, float):
                print(f"    {k:<40s}: {val:>10.4f}")
            else:
                print(f"    {k:<40s}: {val}")

    print("\n" + "=" * 72)
    print(f"  {run_name.upper()} evaluation complete.")
    print("=" * 72)

    return ep_ret, total_cost, step_costs, step_violations


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate R26h/R26i STEMS V3 checkpoint (2-layer transformer, mean pooling)"
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        choices=["r26h", "r26i"],
        help="Which run to evaluate: r26h (lean) or r26i (full)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (absolute or relative to PROJECT_ROOT). "
             "If not provided, uses the default for the selected run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference (default: cpu)",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
