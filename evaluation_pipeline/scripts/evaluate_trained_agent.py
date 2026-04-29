"""
Evaluate a trained OmniSafe PPOLag agent checkpoint on CityLearnSafetyEnv.

Outputs:
- KPI CSV in runs/kpi_logs/<KPI_RUN_NAME>.csv  (same old way)
- Prints summary metrics + V3 classification totals

Usage example:
python evaluate_trained_agent.py \
  --checkpoint runs/ppo_lag_bill_evscale2_35ep/PPOLag-{CityLearnSafety-SoC-v0}/seed-000-2026-01-07-23-57-58/torch_save/epoch-35.pt \
  --kpi_name Evaluated_epoch35_evscale2 \
  --seed 42
"""

import os
import sys
import re
import glob
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv


# --------------------------------------------------------------------------------------
# Robust helpers for loading OmniSafe checkpoint
# --------------------------------------------------------------------------------------

def _pick_state_dict(ckpt: dict):
    """Try common keys where OmniSafe stores the policy state dict."""
    for k in ["pi", "actor", "policy", "actor_state_dict", "pi_state_dict"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    # Sometimes whole ckpt is already a state dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise KeyError(f"Could not find policy state_dict in checkpoint keys: {list(ckpt.keys())}")


def _extract_obs_norm(ckpt: dict):
    """
    Return (mean, std) arrays if available, else (None, None).
    Supports a few formats:
      - obs_normalizer: {'_mean':..., '_std':...}
      - obs_normalizer: {'mean':..., 'var':...}  -> std=sqrt(var)
      - obs_rms: {'mean':..., 'var':...}
    """
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    for key in ["obs_normalizer", "obs_rms", "observation_normalizer"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            d = ckpt[key]
            if "_mean" in d and "_std" in d:
                mean = to_np(d["_mean"])
                std = to_np(d["_std"])
                return mean, std
            if "mean" in d and "var" in d:
                mean = to_np(d["mean"])
                var = to_np(d["var"])
                std = np.sqrt(np.maximum(var, 1e-12))
                return mean, std
            if "mean" in d and "std" in d:
                mean = to_np(d["mean"])
                std = to_np(d["std"])
                return mean, std

    return None, None


def _find_mlp_linear_layers(sd: dict):
    """
    Attempt to find the *mean network* linear layers in OmniSafe actor state_dict.

    Returns list of (W, b) tensors in forward order.

    Strategy:
    - Find candidate weight keys that look like sequential MLP layers:
        something.<int>.weight with matching .bias
    - Prefer keys that contain 'mean' or 'mu' (common for Gaussian mean network)
    - Fall back to any sequential MLP-like linear stack.
    """

    # helper: collect sequential linear layers for a given filter
    def collect(filter_fn):
        items = []
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            if not k.endswith(".weight"):
                continue
            if v.ndim != 2:
                continue
            if not filter_fn(k):
                continue

            # try to find an index like ".0.weight", ".2.weight", etc.
            m = re.search(r"\.(\d+)\.weight$", k)
            if not m:
                continue
            idx = int(m.group(1))
            bias_k = k[:-7] + ".bias"
            if bias_k not in sd:
                continue
            b = sd[bias_k]
            if not isinstance(b, torch.Tensor) or b.ndim != 1:
                continue
            items.append((idx, k, v, b))

        # sort by idx
        items.sort(key=lambda x: x[0])
        return [(W, b) for (_, _, W, b) in items]

    # Prefer mean-like keys
    layers = collect(lambda k: ("mean" in k.lower()) or (".mu" in k.lower()) or ("mu_net" in k.lower()) or ("mu" in k.lower()))
    if len(layers) >= 2:
        return layers

    # Next try anything that looks like sequential layers under "net"
    layers = collect(lambda k: (".net." in k.lower()) or ("mlp" in k.lower()))
    if len(layers) >= 2:
        return layers

    # Last resort: any sequential stack
    layers = collect(lambda k: True)
    if len(layers) >= 2:
        return layers

    raise RuntimeError("Could not infer MLP layers from checkpoint state_dict. "
                       "Print keys and adjust the parser.")


class MeanMLPPolicy:
    """
    Deterministic policy: uses the actor's mean network and tanh-squash.
    We implement forward using extracted linear weights to avoid key-name mismatch issues.
    """

    def __init__(self, sd: dict, obs_mean=None, obs_std=None, device="cpu"):
        self.device = device
        self.layers = _find_mlp_linear_layers(sd)
        self.obs_mean = None if obs_mean is None else torch.tensor(obs_mean, dtype=torch.float32, device=device)
        self.obs_std  = None if obs_std  is None else torch.tensor(obs_std,  dtype=torch.float32, device=device)

        # move weights to device
        self.layers = [(W.to(device), b.to(device)) for (W, b) in self.layers]

    @torch.no_grad()
    def __call__(self, obs_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.obs_mean is not None and self.obs_std is not None:
            x = (x - self.obs_mean) / (self.obs_std + 1e-8)

        # hidden tanh, final tanh
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.t() + b
            x = torch.tanh(x)

        a = x.squeeze(0).cpu().numpy()
        return np.clip(a, -1.0, 1.0)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

def evaluate(checkpoint_path: str, kpi_name: str, seed: int):
    ckpt_path = Path(checkpoint_path)
    assert ckpt_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    print("=" * 90)
    print("EVALUATION")
    print("=" * 90)
    print("checkpoint:", ckpt_path)

    # env vars (old way: KPI goes to runs/kpi_logs/)
    os.environ["CITYLEARN_SCHEMA"] = "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
    os.environ["CITYLEARN_KPI_RUN_NAME"] = kpi_name
    os.environ["CITYLEARN_KPI_FLUSH_EVERY_STEP"] = "1"
    os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "error"

    # IMPORTANT: keep evaluation consistent with your current reward setup
    # (do NOT rely on whatever is left in shell, set explicitly if you care)
    # os.environ["CITYLEARN_EXPORT_FACTOR"] = "0.2"
    # os.environ["CITYLEARN_REWARD_SCALE"]  = "1.0"

    # build env
    base_env = make_base_env(central_agent=True)
    env = CityLearnSafetyEnv(base_env, soc_min=0.0, soc_max=0.95)

    obs, info = env.reset(seed=seed)

    # load checkpoint
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = _pick_state_dict(ckpt)
    obs_mean, obs_std = _extract_obs_norm(ckpt)

    # build policy from weights
    policy = MeanMLPPolicy(sd, obs_mean=obs_mean, obs_std=obs_std, device="cpu")

    # run full episode
    total_cost = 0.0
    total_reward = 0.0
    violation_steps = 0
    steps = 0

    print("\nRunning rollout...")
    done = False
    while not done:
        act = policy(obs)
        obs, reward, terminated, truncated, info = env.step(act)
        done = bool(terminated or truncated)

        c = float(info.get("cost", 0.0))
        total_cost += c
        total_reward += float(reward)
        if c > 0:
            violation_steps += 1

        steps += 1
        if steps % 1000 == 0:
            print(f"  step {steps}/8759")

    # force flush
    try:
        env.close()
    except Exception:
        pass

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"steps: {steps}")
    print(f"total_reward: {total_reward:.3f}")
    print(f"total_cost:   {total_cost:.3f}")
    print(f"cost>0 steps: {violation_steps} ({100.0*violation_steps/max(1,steps):.2f}%)")

    # read KPI file and print EV V3 totals
    kpi_csv = Path("runs/kpi_logs") / f"{kpi_name}.csv"
    if not kpi_csv.exists():
        print("\n[WARNING] KPI CSV not found:", kpi_csv)
        print("If your KPI logger uses a different folder/name, search under runs/:")
        print("  find runs -name '*.csv' | grep", kpi_name)
        return

    import pandas as pd
    df = pd.read_csv(kpi_csv)

    total_missing = float(df.get("ev_missing_action_samples", 0.0).sum()) if "ev_missing_action_samples" in df.columns else 0.0
    v3_control = float(df.get("cost_ev_departure_agent_controllable_v3", 0.0).sum()) if "cost_ev_departure_agent_controllable_v3" in df.columns else 0.0
    v3_uncontrol = float(df.get("cost_ev_departure_uncontrollable_v3", 0.0).sum()) if "cost_ev_departure_uncontrollable_v3" in df.columns else 0.0

    print("\nEV V3 totals (from KPI CSV):")
    print(f"  controllable (avoidable):   {v3_control:.3f} kWh")
    print(f"  uncontrollable:             {v3_uncontrol:.3f} kWh")
    print(f"  missing action samples:     {int(total_missing)}")

    # Normalized KPI comparisons
    if "ev_departure_departures" in df.columns and "ev_departure_deficit_kwh" in df.columns and "ev_avoidable_deficit_kwh" in df.columns:
        dep = (df["ev_departure_departures"].astype(float).fillna(0) > 0)
        deficit = df["ev_departure_deficit_kwh"].astype(float).fillna(0)
        avoid = df["ev_avoidable_deficit_kwh"].astype(float).fillna(0)
        miss = int(((dep) & (deficit > 0)).sum())
        dep_steps = int(dep.sum())
        print("\nNormalized EV metrics:")
        print("  miss_rate:", miss / max(1, dep_steps))
        print("  avoidable_per_dep:", float(avoid.sum()) / max(1, dep_steps))
        print("  deficit_per_dep:", float(deficit.sum()) / max(1, dep_steps))

    print("\nKPI CSV:", kpi_csv)
    print("=" * 90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to epoch-XX.pt (or glob pattern)")
    p.add_argument("--kpi_name", required=True, help="KPI output name (CSV will be runs/kpi_logs/<kpi_name>.csv)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ckpt = args.checkpoint
    if "*" in ckpt:
        matches = glob.glob(ckpt)
        if not matches:
            raise SystemExit(f"No checkpoints matched: {ckpt}")
        ckpt = sorted(matches)[-1]

    evaluate(ckpt, args.kpi_name, args.seed)


if __name__ == "__main__":
    main()
