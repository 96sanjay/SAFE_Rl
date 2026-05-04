# scripts/training/train_grads.py
"""Training script for PPOLag + GradS.

Bypasses omnisafe.Agent (which needs a default config YAML per algorithm)
and directly instantiates PPOLagGradS with the merged config.

Usage:
    python scripts/training/train_grads.py --cfg configs/archive/on-policy/ppolag_grads_5bld.yaml
"""
from __future__ import annotations

import argparse
import copy
import os

import torch
import yaml

# Register environments FIRST (cmdp_env registers CityLearnSafety-V2G-v2)
import citylearn_safe.cmdp_env    # noqa: F401  @env_register side-effect

from omnisafe.utils.config import Config
from omnisafe.utils.tools import seed_all

# Import PPOLagGradS (triggers @registry.register)
from citylearn_safe.grads.ppo_lag_grads import PPOLagGradS


def load_ppolag_defaults() -> dict:
    """Load PPOLag default config from OmniSafe's installed configs."""
    import omnisafe
    pkg_dir = os.path.dirname(os.path.abspath(omnisafe.__file__))
    default_path = os.path.join(pkg_dir, 'configs', 'on-policy', 'PPOLag.yaml')
    with open(default_path) as f:
        raw = yaml.safe_load(f)
    return raw.get('defaults', raw)


def deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def main(cfg_path: str) -> None:
    # 1. Load PPOLag defaults
    defaults = load_ppolag_defaults()

    # 2. Load our custom config
    with open(cfg_path) as f:
        custom = yaml.safe_load(f)

    algo = custom.pop('algo', 'PPOLagGradS')
    env_id = custom.pop('env_id', 'CityLearnSafety-V2G-v2')
    seed = custom.pop('seed', 42)

    # 3. Merge: defaults + custom overrides
    merged = deep_update(defaults, custom)
    merged['seed'] = seed
    merged['env_id'] = env_id
    merged['algo'] = algo
    merged['exp_name'] = f'{algo}-{{{env_id}}}'

    # Compute epochs from total_steps
    total_steps = merged['train_cfgs']['total_steps']
    steps_per_epoch = merged['algo_cfgs']['steps_per_epoch']
    merged['train_cfgs']['epochs'] = total_steps // steps_per_epoch

    # Ensure device is set
    if 'device' not in merged['train_cfgs']:
        merged['train_cfgs']['device'] = 'cpu'

    # 4. Convert to OmniSafe Config object
    cfgs = Config(**merged)

    print(f"{'=' * 50}")
    print(f"  PPOLag + GradS Training")
    print(f"  Algo: {algo}")
    print(f"  Env: {env_id}")
    print(f"  Seed: {seed}")
    print(f"  Epochs: {cfgs.train_cfgs.epochs}")
    print(f"  Steps/epoch: {cfgs.algo_cfgs.steps_per_epoch}")
    print(f"  Cost limit: {cfgs.lagrange_cfgs.cost_limit}")
    if hasattr(cfgs, 'grads_cfgs'):
        print(f"  GradS sim_threshold: {cfgs.grads_cfgs.sim_threshold}")
        print(f"  GradS conflict_threshold: {cfgs.grads_cfgs.conflict_threshold}")
        print(f"  Per-constraint cost limits:")
        for i in range(5):
            lim = float(getattr(cfgs.grads_cfgs, f'cost_limit_{i}', 5000.0))
            print(f"    C{i}: {lim:.0f}")
    print(f"{'=' * 50}")

    # 5. Direct instantiation (bypasses omnisafe.Agent)
    agent = PPOLagGradS(env_id=env_id, cfgs=cfgs)
    ep_ret, ep_cost, ep_len = agent.learn()

    print(f"\nTraining complete.")
    print(f"  Final EpRet: {ep_ret:.1f}")
    print(f"  Final EpCost: {ep_cost:.1f}")
    print(f"  Final EpLen: {ep_len:.0f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='Path to YAML config')
    args = ap.parse_args()
    main(args.cfg)
