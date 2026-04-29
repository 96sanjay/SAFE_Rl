#!/usr/bin/env python3
"""
Evaluation script for R26f checkpoint.

Loads the R26f STEMS V3 encoder checkpoint, runs a deterministic 1-episode
(8759 steps) evaluation, and reports per-constraint violation percentages,
totals, reward statistics, and CityLearn KPIs.

Usage (on remote server):
    cd /home/sanjay/Safe-CityLearn-Fork/Safe-CityLearn-Fork
    conda activate citylearn
    python scripts/eval_r26f.py [--checkpoint PATH] [--device cpu]

Usage (local):
    cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
    conda activate citylearn
    python scripts/eval_r26f.py [--checkpoint PATH] [--device cpu]
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
# Path setup: detect which machine we are on
# ---------------------------------------------------------------------------
REMOTE_ROOT = "/home/sanjay/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
LOCAL_ROOT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"

if os.path.isdir(REMOTE_ROOT):
    PROJECT_ROOT = REMOTE_ROOT
elif os.path.isdir(LOCAL_ROOT):
    PROJECT_ROOT = LOCAL_ROOT
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Default checkpoint path (relative to PROJECT_ROOT)
# ---------------------------------------------------------------------------
DEFAULT_CKPT_REL = (
    "runs/r26f_minimal_reward/5bld/"
    "PPOLagMulti-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-04-10-01-08-05/"
    "torch_save/epoch-120.pt"
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
# Set environment variables BEFORE importing any project modules
# ---------------------------------------------------------------------------
def set_env_vars():
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
    for k, v in env_vars.items():
        os.environ[k] = v
    print(f"[eval_r26f] Schema: {schema_path}")
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"Schema not found: {schema_path}")


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
    import citylearn_safe.omni_env       # noqa: F401
    import citylearn_safe.omni_env_v2    # noqa: F401

    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.temporal_obs_wrapper import TemporalHistoryWrapper

    # 1. Base CityLearn env
    base_env = make_base_env(central_agent=True)

    # 2. Safety wrapper
    safety_env = CityLearnSafetyEnvV3(base_env)

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
def build_and_load_model(obs_index, rich_cfg, env, checkpoint_path, device):
    """Build STEMSEncoderV3 + STEMSMeanNet and load checkpoint weights."""
    from citylearn_safe.stems_encoder_5bld import build_node_indices
    from citylearn_safe.stems_encoder_v3 import STEMSEncoderV3

    num_buildings = 5
    temporal_window = int(os.environ.get("CITYLEARN_TEMPORAL_WINDOW", "12"))

    node_info = build_node_indices(obs_index, num_buildings)
    print(f"[eval_r26f] base_obs_dim={node_info['base_obs_dim']}")

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
        f"[eval_r26f] obs_dim={obs_dim}, act_dim={act_dim}, "
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
        temporal_num_layers=1,    # R26f uses 1 layer (not 2)
        temporal_pool_mode="last",  # R26f uses last position pooling (not mean)
    )
    encoder = STEMSEncoderV3(**stems_kwargs)
    mean_net = STEMSMeanNet(encoder, act_dim, encoder_obs_dim=obs_dim_v3)

    total_params = sum(p.numel() for p in mean_net.parameters())
    print(f"[eval_r26f] STEMSMeanNet params: {total_params:,}")

    # --- Load checkpoint ---
    print(f"[eval_r26f] Loading checkpoint: {checkpoint_path}")
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
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate(args):
    set_env_vars()

    device = torch.device(args.device)
    checkpoint_path = args.checkpoint
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
        obs_index, rich_cfg, env, checkpoint_path, device
    )

    # --- Run evaluation ---
    print("\n=== Running deterministic evaluation (1 episode, 8759 steps) ===")
    obs, info = env.reset()

    # Tracking
    step_rewards = []
    step_costs = defaultdict(list)       # cost_key -> list of per-step costs
    step_violations = defaultdict(list)  # cost_key -> list of 0/1 per step
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

        # Track per-constraint costs
        for key in COST_KEYS:
            cost_val = float(info.get(key, 0.0))
            step_costs[key].append(cost_val)
            step_violations[key].append(1.0 if cost_val > 0.0 else 0.0)

        total_cost += float(info.get("cost", 0.0))

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
    print("  R26f EVALUATION RESULTS")
    print("=" * 72)

    # Reward summary
    ep_ret = sum(step_rewards)
    print(f"\n  Episode Return:       {ep_ret:>12.1f}")
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
    print("  Evaluation complete.")
    print("=" * 72)

    return ep_ret, total_cost, step_costs, step_violations


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate R26f STEMS V3 checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CKPT_REL,
        help="Path to checkpoint (absolute or relative to PROJECT_ROOT)",
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
