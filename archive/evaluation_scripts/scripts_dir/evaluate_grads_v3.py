#!/usr/bin/env python3
"""
Evaluate GradS v3 (PPOLagGradS) trained model on CityLearn 5-building environment.

Usage:
    cd /home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork
    eval "$(conda shell.bash hook)" && conda activate citylearn
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python scripts/evaluate_grads_v3.py

Outputs:
    - Console: summary table with total reward, total cost, per-constraint costs
    - CSV:  runs/grads_v3/grads_5bld/evaluation/eval_epoch50.csv
    - JSON: runs/grads_v3/grads_5bld/evaluation/eval_summary.json
"""
from __future__ import annotations

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

# ---------------------------------------------------------------------------
# Environment variables -- MUST match training exactly
# ---------------------------------------------------------------------------
ENV_VARS = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_SPATIAL_OBS": "1",
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
    "CITYLEARN_EV_DENSE_COST_SCALE": "1.0",
    "COST_W_C1": "10.0",
    "COST_W_C1_DENSE": "5.0",
    "COST_W_C2": "1.0",
    "COST_W_C3": "0.1",
    "COST_W_C4": "5.0",
    "STEMS_ALPHA_GRID": "1.0",
    "STEMS_BETA_RAMP": "2.0",
}

for k, v in ENV_VARS.items():
    os.environ[k] = v

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT = (
    f"{PROJECT}/runs/grads_v3/grads_5bld/"
    "PPOLagGradS-{CityLearnSafety-V2G-v2}/"
    "seed-042-2026-03-08-16-55-44/torch_save/epoch-50.pt"
)
OUTPUT_DIR = f"{PROJECT}/runs/grads_v3/grads_5bld/evaluation"
CSV_PATH = os.path.join(OUTPUT_DIR, "eval_epoch50.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "eval_summary.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Build the actor MLP (must match OmniSafe's build_mlp_network)
# ---------------------------------------------------------------------------
def build_actor_mlp(obs_dim: int, act_dim: int, hidden_sizes: list[int],
                    activation: str = "tanh") -> nn.Sequential:
    """
    Reproduce OmniSafe's MLP actor mean network.
    Structure: Linear -> Act -> Linear -> Act -> Linear -> Identity
    """
    act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    sizes = [obs_dim] + hidden_sizes + [act_dim]
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        if j < len(sizes) - 2:
            layers.append(act_fn())
        else:
            layers.append(nn.Identity())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint(path: str):
    """Load actor weights and obs normalizer from an OmniSafe PPOLag checkpoint."""
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")

    pi_state = ckpt["pi"]
    obs_norm = ckpt["obs_normalizer"]

    # Infer dimensions from weights
    first_weight = pi_state["mean.0.weight"]
    last_weight = [v for k, v in pi_state.items() if "weight" in k][-1]
    obs_dim = first_weight.shape[1]
    act_dim = last_weight.shape[0]

    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"  log_std={pi_state['log_std'].numpy().round(4).tolist()}")

    # Build and load actor
    actor = build_actor_mlp(obs_dim, act_dim, [256, 256], activation="tanh")
    # Map checkpoint keys to our sequential model
    actor_state = {}
    for k, v in pi_state.items():
        if k.startswith("mean."):
            actor_state[k.replace("mean.", "")] = v
    actor.load_state_dict(actor_state)
    actor.eval()

    # Obs normalizer
    obs_mean = obs_norm["_mean"].numpy()
    obs_std = obs_norm["_std"].numpy()
    obs_clip = obs_norm["_clip"].numpy() if "_clip" in obs_norm else np.full_like(obs_mean, 5.0)

    print(f"  obs_normalizer count={int(obs_norm['_count'].item())}")
    return actor, obs_mean, obs_std, obs_clip


# ---------------------------------------------------------------------------
# Build environment (same wrapper chain as training)
# ---------------------------------------------------------------------------
def build_eval_env():
    """
    Replicate the exact wrapper chain from CityLearnCMDPv2.__init__
    but without the OmniSafe CMDP wrapper (we drive it manually).
    """
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.stems_obs_wrapper import SpatialGraphFeaturesWrapper

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    forecast = ForecastObsWrapper(safety, forecast_horizon=24)

    # Spatial obs (matches training: CITYLEARN_SPATIAL_OBS=1)
    p_bmax = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
    n_buildings = 5
    env = SpatialGraphFeaturesWrapper(forecast, num_buildings=n_buildings, p_building_max=p_bmax)
    print(f"[Eval] Final env obs_space={env.observation_space.shape}, act_space={env.action_space.shape}")
    return env


# ---------------------------------------------------------------------------
# STEMS reward (from CityLearnCMDPv2)
# ---------------------------------------------------------------------------
class STEMSRewardCalculator:
    """Replicates the STEMS reward from omni_env_v2.py for evaluation."""

    def __init__(self, env):
        self.env = env
        self.mu_economic = float(os.environ.get("STEMS_MU_ECONOMIC", "0.3"))
        self.alpha_grid = float(os.environ.get("STEMS_ALPHA_GRID", "3.0"))
        self.alpha_build = float(os.environ.get("STEMS_ALPHA_BUILD", "2.0"))
        self.beta_ramp = float(os.environ.get("STEMS_BETA_RAMP", "0.5"))
        self.xi_renewable = float(os.environ.get("STEMS_XI_RENEWABLE", "0.2"))
        self.lambda_ev = float(os.environ.get("STEMS_LAMBDA_EV", "5.0"))
        self.P_building_max = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
        self.P_grid_max = float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "10.2352"))
        self._prev_net = None

    def _get_citylearn(self):
        cur = self.env
        seen = set()
        for _ in range(40):
            if cur is None or id(cur) in seen:
                break
            seen.add(id(cur))
            if hasattr(cur, 'buildings') and hasattr(cur, 'time_step') and \
               hasattr(cur.buildings, '__len__') and len(cur.buildings) > 0:
                return cur
            for attr in ('base', 'env', 'unwrapped', '_env', 'raw_env'):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
            else:
                break
        return None

    def reset(self):
        self._prev_net = None

    def compute(self, info: dict, action_np: np.ndarray) -> float:
        city = self._get_citylearn()
        if city is None:
            return 0.0
        t_idx = max(0, int(getattr(city, 'time_step', 0)) - 1)
        buildings = list(getattr(city, 'buildings', []))

        # Economic
        try:
            pr = buildings[0].pricing.electricity_pricing
            price = float(pr[t_idx]) if hasattr(pr, '__len__') and len(pr) > t_idx else 0.17
        except Exception:
            price = 0.17
        total_net = 0.0
        for b in buildings:
            try:
                nec = getattr(b, 'net_electricity_consumption', None)
                if nec is not None and hasattr(nec, '__len__') and len(nec) > t_idx:
                    total_net += float(nec[t_idx])
            except Exception:
                pass
        imp = max(0.0, total_net)
        exp = max(0.0, -total_net)
        ef = float(os.environ.get("CITYLEARN_EXPORT_FACTOR", "1.0"))
        r_eco = -self.mu_economic * price * (imp - ef * exp)

        # Grid stability
        r_sg = self.alpha_grid * (1.0 - min((imp / max(1e-6, self.P_grid_max)) ** 2, 4.0))

        # Building stability
        bs, bc = 0.0, 0
        for b in buildings:
            try:
                nec = getattr(b, 'net_electricity_consumption', None)
                if nec is not None and hasattr(nec, '__len__') and len(nec) > t_idx:
                    ratio = abs(float(nec[t_idx])) / max(1e-6, self.P_building_max)
                    bs += 1.0 - min(ratio, 4.0); bc += 1
            except Exception:
                pass
        r_sb = self.alpha_build * (bs / max(1, bc)) if bc > 0 else 0.0

        # Ramp
        rd = abs(total_net - self._prev_net) if self._prev_net is not None else 0.0
        self._prev_net = total_net
        r_ramp = -self.beta_ramp * (rd / max(1e-6, self.P_grid_max))

        # Renewable
        sg = 0.0
        for b in buildings:
            try:
                s = getattr(b, 'solar_generation', None)
                if s is not None and hasattr(s, '__len__') and len(s) > t_idx:
                    sg += abs(float(s[t_idx]))
            except Exception:
                pass
        r_ren = self.xi_renewable * min(sg / (sg + imp), 1.0) if (sg + imp) > 0 else 0.0

        return float(r_eco + r_sg + r_sb + r_ramp + r_ren)


# ---------------------------------------------------------------------------
# Rebalanced cost (from CityLearnCMDPv2)
# ---------------------------------------------------------------------------
def rebalanced_cost(info: dict) -> float:
    w_c1 = float(os.environ.get("COST_W_C1", "10.0"))
    w_c1d = float(os.environ.get("COST_W_C1_DENSE", "5.0"))
    w_c2 = float(os.environ.get("COST_W_C2", "1.0"))
    w_c3 = float(os.environ.get("COST_W_C3", "0.1"))
    w_c4 = float(os.environ.get("COST_W_C4", "5.0"))
    return float(
        w_c1 * float(info.get('cost_ev_departure', 0.0)) +
        w_c1d * float(info.get('cost_ev_dense', 0.0)) +
        w_c2 * float(info.get('cost_stems_battery', 0.0)) +
        w_c3 * float(info.get('cost_stems_building_power', 0.0)) +
        w_c4 * float(info.get('cost_stems_grid_power', 0.0))
    )


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
def run_evaluation(actor, obs_mean, obs_std, obs_clip, env, label="GradS_v3"):
    """Run a full 8759-step episode and collect per-step data."""
    stems_calc = STEMSRewardCalculator(env)

    obs, info = env.reset()
    stems_calc.reset()

    obs_np = np.asarray(obs, dtype=np.float32).ravel()
    assert obs_np.shape[0] == obs_mean.shape[0], \
        f"Obs dim mismatch: env={obs_np.shape[0]} vs checkpoint={obs_mean.shape[0]}"

    records = []
    total_reward = 0.0
    total_cost = 0.0
    cost_sums = {
        "cost_ev_departure": 0.0,
        "cost_ev_dense": 0.0,
        "cost_stems_battery": 0.0,
        "cost_stems_building_power": 0.0,
        "cost_stems_grid_power": 0.0,
    }
    step = 0
    t_start = time.time()

    while True:
        # Normalize observation
        obs_norm = np.clip(
            (obs_np - obs_mean) / (obs_std + 1e-8),
            -obs_clip, obs_clip
        )

        # Actor forward (deterministic)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            action_t = actor(obs_t)
            action_np = action_t.squeeze(0).numpy()

        # Clip actions to [-1, 1] (actor uses tanh implicitly via training, but
        # the final layer is Identity so raw output may slightly exceed bounds)
        action_np = np.clip(action_np, -1.0, 1.0)

        # Step environment
        obs_next, _reward_base, terminated, truncated, info = env.step(action_np)

        # Compute STEMS reward (same as training)
        reward = stems_calc.compute(info, action_np)
        cost = rebalanced_cost(info)

        total_reward += reward
        total_cost += cost

        # Per-constraint raw costs from info
        c1 = float(info.get("cost_ev_departure", 0.0))
        c1d = float(info.get("cost_ev_dense", 0.0))
        c2 = float(info.get("cost_stems_battery", 0.0))
        c3 = float(info.get("cost_stems_building_power", 0.0))
        c4 = float(info.get("cost_stems_grid_power", 0.0))

        cost_sums["cost_ev_departure"] += c1
        cost_sums["cost_ev_dense"] += c1d
        cost_sums["cost_stems_battery"] += c2
        cost_sums["cost_stems_building_power"] += c3
        cost_sums["cost_stems_grid_power"] += c4

        record = {
            "step": step,
            "reward": reward,
            "cost_total": cost,
            "cost_ev_departure": c1,
            "cost_ev_dense": c1d,
            "cost_stems_battery": c2,
            "cost_stems_building_power": c3,
            "cost_stems_grid_power": c4,
        }
        # Log actions
        for i, a in enumerate(action_np):
            record[f"action_{i}"] = float(a)

        records.append(record)

        step += 1
        if step % 1000 == 0:
            elapsed = time.time() - t_start
            print(f"  [{label}] step {step}/8759  cumR={total_reward:.1f}  "
                  f"cumCost={total_cost:.1f}  elapsed={elapsed:.1f}s")

        obs_np = np.asarray(obs_next, dtype=np.float32).ravel()

        if terminated or truncated:
            break

    elapsed = time.time() - t_start
    print(f"  [{label}] Episode done: {step} steps in {elapsed:.1f}s")

    return records, total_reward, total_cost, cost_sums


# ---------------------------------------------------------------------------
# RBC baseline (zero actions)
# ---------------------------------------------------------------------------
def run_rbc_baseline(env, label="RBC_zero"):
    """Run a full episode with zero actions as a naive baseline."""
    stems_calc = STEMSRewardCalculator(env)

    obs, info = env.reset()
    stems_calc.reset()

    act_dim = env.action_space.shape[0]
    zero_action = np.zeros(act_dim, dtype=np.float32)

    records = []
    total_reward = 0.0
    total_cost = 0.0
    cost_sums = {
        "cost_ev_departure": 0.0,
        "cost_ev_dense": 0.0,
        "cost_stems_battery": 0.0,
        "cost_stems_building_power": 0.0,
        "cost_stems_grid_power": 0.0,
    }
    step = 0
    t_start = time.time()

    while True:
        obs_next, _reward_base, terminated, truncated, info = env.step(zero_action)

        reward = stems_calc.compute(info, zero_action)
        cost = rebalanced_cost(info)

        total_reward += reward
        total_cost += cost

        c1 = float(info.get("cost_ev_departure", 0.0))
        c1d = float(info.get("cost_ev_dense", 0.0))
        c2 = float(info.get("cost_stems_battery", 0.0))
        c3 = float(info.get("cost_stems_building_power", 0.0))
        c4 = float(info.get("cost_stems_grid_power", 0.0))

        cost_sums["cost_ev_departure"] += c1
        cost_sums["cost_ev_dense"] += c1d
        cost_sums["cost_stems_battery"] += c2
        cost_sums["cost_stems_building_power"] += c3
        cost_sums["cost_stems_grid_power"] += c4

        record = {
            "step": step,
            "reward": reward,
            "cost_total": cost,
            "cost_ev_departure": c1,
            "cost_ev_dense": c1d,
            "cost_stems_battery": c2,
            "cost_stems_building_power": c3,
            "cost_stems_grid_power": c4,
        }
        for i in range(act_dim):
            record[f"action_{i}"] = 0.0
        records.append(record)

        step += 1
        if step % 1000 == 0:
            elapsed = time.time() - t_start
            print(f"  [{label}] step {step}/8759  cumR={total_reward:.1f}  "
                  f"cumCost={total_cost:.1f}  elapsed={elapsed:.1f}s")

        obs = obs_next
        if terminated or truncated:
            break

    elapsed = time.time() - t_start
    print(f"  [{label}] Episode done: {step} steps in {elapsed:.1f}s")
    return records, total_reward, total_cost, cost_sums


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("GradS v3 Evaluation -- PPOLagGradS epoch-50, 5-building CityLearn")
    print("=" * 72)

    # Load model
    actor, obs_mean, obs_std, obs_clip = load_checkpoint(CHECKPOINT)

    # Build env for GradS agent
    print("\n--- Building environment for GradS agent ---")
    env_agent = build_eval_env()

    # Run GradS evaluation
    print("\n--- Running GradS v3 evaluation ---")
    grads_records, grads_reward, grads_cost, grads_costs = run_evaluation(
        actor, obs_mean, obs_std, obs_clip, env_agent, label="GradS_v3"
    )

    # Build fresh env for RBC baseline
    print("\n--- Building environment for RBC baseline ---")
    env_rbc = build_eval_env()

    # Run RBC baseline
    print("\n--- Running RBC (zero-action) baseline ---")
    rbc_records, rbc_reward, rbc_cost, rbc_costs = run_rbc_baseline(
        env_rbc, label="RBC_zero"
    )

    # Save GradS per-step CSV
    df_grads = pd.DataFrame(grads_records)
    df_grads.to_csv(CSV_PATH, index=False)
    print(f"\nSaved per-step data to: {CSV_PATH}")

    # Also save RBC CSV
    rbc_csv = os.path.join(OUTPUT_DIR, "eval_rbc_zero.csv")
    df_rbc = pd.DataFrame(rbc_records)
    df_rbc.to_csv(rbc_csv, index=False)
    print(f"Saved RBC baseline data to: {rbc_csv}")

    # Summary
    summary = {
        "checkpoint": CHECKPOINT,
        "grads_v3": {
            "total_reward": round(grads_reward, 2),
            "total_cost": round(grads_cost, 2),
            "total_cost_ev_departure_raw": round(grads_costs["cost_ev_departure"], 2),
            "total_cost_ev_dense_raw": round(grads_costs["cost_ev_dense"], 2),
            "total_cost_stems_battery_raw": round(grads_costs["cost_stems_battery"], 2),
            "total_cost_stems_building_power_raw": round(grads_costs["cost_stems_building_power"], 2),
            "total_cost_stems_grid_power_raw": round(grads_costs["cost_stems_grid_power"], 2),
            "steps": len(grads_records),
        },
        "rbc_zero": {
            "total_reward": round(rbc_reward, 2),
            "total_cost": round(rbc_cost, 2),
            "total_cost_ev_departure_raw": round(rbc_costs["cost_ev_departure"], 2),
            "total_cost_ev_dense_raw": round(rbc_costs["cost_ev_dense"], 2),
            "total_cost_stems_battery_raw": round(rbc_costs["cost_stems_battery"], 2),
            "total_cost_stems_building_power_raw": round(rbc_costs["cost_stems_building_power"], 2),
            "total_cost_stems_grid_power_raw": round(rbc_costs["cost_stems_grid_power"], 2),
            "steps": len(rbc_records),
        },
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {SUMMARY_PATH}")

    # Print summary table
    print("\n" + "=" * 72)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 72)
    print(f"{'Metric':<40} {'GradS v3':>14} {'RBC (zero)':>14}")
    print("-" * 72)

    rows = [
        ("Total STEMS Reward", grads_reward, rbc_reward),
        ("Total Weighted Cost", grads_cost, rbc_cost),
        ("", None, None),
        ("Raw C1: EV departure", grads_costs["cost_ev_departure"], rbc_costs["cost_ev_departure"]),
        ("Raw C1d: EV dense", grads_costs["cost_ev_dense"], rbc_costs["cost_ev_dense"]),
        ("Raw C2: Battery SoC", grads_costs["cost_stems_battery"], rbc_costs["cost_stems_battery"]),
        ("Raw C3: Building power", grads_costs["cost_stems_building_power"], rbc_costs["cost_stems_building_power"]),
        ("Raw C4: Grid power", grads_costs["cost_stems_grid_power"], rbc_costs["cost_stems_grid_power"]),
        ("", None, None),
        ("Weighted C1 (x10)", grads_costs["cost_ev_departure"] * 10, rbc_costs["cost_ev_departure"] * 10),
        ("Weighted C1d (x5)", grads_costs["cost_ev_dense"] * 5, rbc_costs["cost_ev_dense"] * 5),
        ("Weighted C2 (x1)", grads_costs["cost_stems_battery"] * 1, rbc_costs["cost_stems_battery"] * 1),
        ("Weighted C3 (x0.1)", grads_costs["cost_stems_building_power"] * 0.1, rbc_costs["cost_stems_building_power"] * 0.1),
        ("Weighted C4 (x5)", grads_costs["cost_stems_grid_power"] * 5, rbc_costs["cost_stems_grid_power"] * 5),
        ("", None, None),
        ("Steps", len(grads_records), len(rbc_records)),
    ]

    for name, gval, rval in rows:
        if gval is None:
            print()
        else:
            print(f"{name:<40} {gval:>14.2f} {rval:>14.2f}")

    # Improvement percentages
    print("\n" + "-" * 72)
    print("Improvement vs RBC baseline:")
    if abs(rbc_reward) > 1e-6:
        r_pct = (grads_reward - rbc_reward) / abs(rbc_reward) * 100
        print(f"  Reward: {r_pct:+.1f}% ({'better' if r_pct > 0 else 'worse'})")
    if abs(rbc_cost) > 1e-6:
        c_pct = (grads_cost - rbc_cost) / abs(rbc_cost) * 100
        print(f"  Cost:   {c_pct:+.1f}% ({'lower is better' if c_pct < 0 else 'higher cost'})")

    for cname in ["cost_ev_departure", "cost_ev_dense", "cost_stems_battery",
                   "cost_stems_building_power", "cost_stems_grid_power"]:
        rval = rbc_costs[cname]
        gval = grads_costs[cname]
        if abs(rval) > 1e-6:
            pct = (gval - rval) / abs(rval) * 100
            short = cname.replace("cost_", "").replace("stems_", "")
            print(f"  {short}: {pct:+.1f}%")

    print("\n" + "=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
