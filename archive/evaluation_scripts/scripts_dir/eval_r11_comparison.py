#!/usr/bin/env python3
"""Evaluate R11a (single-lambda) vs R11b (multi-lambda) agents.

Loads final checkpoints, runs one full episode each with deterministic policy,
and compares KPIs: reward, per-constraint costs, violation rates.

Usage:
    python scripts/eval_r11_comparison.py
"""
import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

PROJECT = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("PYTHONPATH", str(PROJECT))

# Same env vars as the run scripts
os.environ["CITYLEARN_SCHEMA"] = str(PROJECT / "data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json")
os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"
os.environ["CITYLEARN_EXPORT_FACTOR"] = "0.7"
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "10.2352"
os.environ["CITYLEARN_STEMS_SOC_LOW"] = "0.0"
os.environ["CITYLEARN_STEMS_SOC_HIGH"] = "0.95"
os.environ["CITYLEARN_STEMS_PNORM_P"] = "4.0"
os.environ["STEMS_LAMBDA_EV"] = "0.0"
os.environ["STEMS_ALPHA_BARRIER"] = "0.5"
os.environ["COST_W_C2"] = "0.0"
os.environ["STEMS_BETA_RAMP"] = "1.5"
os.environ["COST_W_C3"] = "5.0"
os.environ["CITYLEARN_W_COST_EV"] = "1.0"
os.environ["CITYLEARN_W_COST_SOC"] = "10.0"
os.environ["CITYLEARN_W_COST_BUILDING"] = "0.5"
os.environ["CITYLEARN_W_COST_GRID"] = "0.05"
os.environ["CITYLEARN_EV_COST_SCALE"] = "3.0"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_full"
os.environ["CITYLEARN_INCLUDE_EV_COST"] = "1"
os.environ["CITYLEARN_C3_CONTROLLABLE"] = "0"
os.environ["CITYLEARN_SPATIAL_OBS"] = "0"
os.environ["CITYLEARN_TEMPORAL_WINDOW"] = "0"

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper


class MLPActor(nn.Module):
    """Matches OmniSafe's default GaussianLearningActor MLP structure."""
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        layers.append(nn.Tanh())
        self.mean = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        return self.mean(obs)


def make_eval_env():
    """Create the same env used during training."""
    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    forecast = ForecastObsWrapper(safety, forecast_horizon=24)
    return forecast


def load_agent(checkpoint_path, obs_dim, act_dim):
    """Load actor and obs normalizer from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    actor = MLPActor(obs_dim, act_dim)
    actor.load_state_dict(ckpt['pi'])
    actor.eval()

    norm = ckpt['obs_normalizer']
    obs_mean = torch.FloatTensor(norm['_mean'])
    obs_std = torch.FloatTensor(norm['_std'])
    return actor, obs_mean, obs_std


@torch.no_grad()
def evaluate_agent(env, actor, obs_mean, obs_std, label="Agent"):
    """Run one full episode, collect all KPIs."""
    obs, _ = env.reset(seed=42)
    metrics = defaultdict(float)
    step_data = []
    done = truncated = False
    step = 0

    # Cost keys to track
    cost_keys = [
        'cost_ev_departure', 'cost_ev_dense',
        'cost_stems_battery', 'cost_stems_building_power',
        'cost_stems_grid_power',
    ]
    violation_keys = [
        'battery_soc_violation_any', 'grid_power_violation',
    ]

    while not (done or truncated):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        obs_t = (obs_t - obs_mean) / (obs_std + 1e-8)
        action = actor(obs_t).cpu().numpy().squeeze()
        action = np.clip(action, -1.0, 1.0)

        obs, reward, done, truncated, info = env.step(action)

        metrics['total_reward'] += reward
        metrics['total_cost'] += info.get('cost', 0.0)
        for k in cost_keys:
            metrics[f'total_{k}'] += info.get(k, 0.0)
        for k in violation_keys:
            metrics[f'count_{k}'] += (1.0 if info.get(k, 0) > 0 else 0.0)

        # Track SoC stats
        for b in range(1, 18):
            soc = info.get(f'battery_soc_b{b}', -1)
            if soc >= 0:
                if soc < 0.05 or soc > 0.95:
                    metrics['soc_violation_steps'] += 1
                    break  # count per-step, not per-building

        step += 1
        if step % 2000 == 0:
            print(f"  [{label}] step {step}/8759")

    metrics['steps'] = step
    metrics['soc_violation_rate'] = metrics['soc_violation_steps'] / step * 100
    for k in violation_keys:
        metrics[f'rate_{k}'] = metrics[f'count_{k}'] / step * 100

    return dict(metrics)


def find_checkpoint(run_dir, epoch="50"):
    """Find checkpoint file."""
    pattern = str(PROJECT / run_dir / f"*/seed-*/torch_save/epoch-{epoch}.pt")
    matches = sorted(glob.glob(pattern))
    # Filter out smoke test runs (short seed dirs)
    for m in matches:
        if "22-45" in m or "22-37" in m:  # main run timestamps
            return m
    return matches[-1] if matches else None


def main():
    print("=" * 70)
    print("  R11 A/B Evaluation: Single-Lambda vs Multi-Lambda")
    print("=" * 70)

    # Create env to get dimensions
    env = make_eval_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Env: obs_dim={obs_dim}, act_dim={act_dim}")

    # Find checkpoints
    ckpt_a = find_checkpoint("runs/r11a_ppolag_improved/5bld")
    ckpt_b = find_checkpoint("runs/r11b_multi_improved/5bld")
    print(f"Run A checkpoint: {ckpt_a}")
    print(f"Run B checkpoint: {ckpt_b}")

    if not ckpt_a or not ckpt_b:
        print("ERROR: Missing checkpoints!")
        return

    results = {}

    # Evaluate Run A
    print(f"\n--- Evaluating Run A (PPOLag, single-lambda) ---")
    actor_a, mean_a, std_a = load_agent(ckpt_a, obs_dim, act_dim)
    env_a = make_eval_env()
    results['Run A (single-λ)'] = evaluate_agent(env_a, actor_a, mean_a, std_a, "R11a")

    # Evaluate Run B
    print(f"\n--- Evaluating Run B (PPOLagMulti, multi-lambda) ---")
    actor_b, mean_b, std_b = load_agent(ckpt_b, obs_dim, act_dim)
    env_b = make_eval_env()
    results['Run B (multi-λ)'] = evaluate_agent(env_b, actor_b, mean_b, std_b, "R11b")

    # Also evaluate random baseline
    print(f"\n--- Evaluating Random Baseline ---")
    class RandomActor(nn.Module):
        def __init__(self, act_dim):
            super().__init__()
            self.act_dim = act_dim
        def forward(self, obs):
            return torch.zeros(1, self.act_dim)
    rand_actor = RandomActor(act_dim)
    env_rand = make_eval_env()
    results['Random'] = evaluate_agent(
        env_rand, rand_actor,
        torch.zeros(obs_dim), torch.ones(obs_dim),
        "Random"
    )

    # Print comparison table
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)

    labels = list(results.keys())
    metrics_to_show = [
        ('total_reward', 'Total Reward', '.0f'),
        ('total_cost', 'Total Cost', '.0f'),
        ('total_cost_ev_departure', 'C0 (EV departure)', '.0f'),
        ('total_cost_ev_dense', 'C1 (EV dense)', '.0f'),
        ('total_cost_stems_battery', 'C2 (Battery SoC)', '.0f'),
        ('total_cost_stems_building_power', 'C3 (Building power)', '.0f'),
        ('total_cost_stems_grid_power', 'C4 (Grid power)', '.0f'),
        ('soc_violation_rate', 'SoC violation %', '.1f'),
        ('rate_battery_soc_violation_any', 'Battery viol %', '.1f'),
        ('rate_grid_power_violation', 'Grid viol %', '.1f'),
    ]

    # Header
    header = f"{'Metric':<25s}"
    for lab in labels:
        header += f" | {lab:>20s}"
    print(header)
    print("-" * len(header))

    # Rows
    for key, name, fmt in metrics_to_show:
        row = f"{name:<25s}"
        for lab in labels:
            val = results[lab].get(key, 0)
            row += f" | {val:>20{fmt}}"
        print(row)

    print("=" * 70)


if __name__ == "__main__":
    main()
