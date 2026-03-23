# scripts/train_sac_lag.py
"""Training script for SACLagMulti (off-policy, per-constraint Lagrange multipliers).

Bypasses omnisafe.Agent (which needs a default config YAML per algorithm)
and directly instantiates SACLagMulti with the merged config.

Usage:
    python scripts/train_sac_lag.py --cfg configs/off-policy/sac_lag_1bld.yaml
"""
from __future__ import annotations

import argparse
import copy
import os

import yaml

# Register environments FIRST
import citylearn_safe.omni_env       # noqa: F401
import citylearn_safe.omni_env_v2    # noqa: F401

from omnisafe.utils.config import Config

# Import SACLagMulti (triggers @registry.register)
from citylearn_safe.grads.sac_lag_multi import SACLagMulti


def load_saclag_defaults() -> dict:
    """Load SACLag default config from OmniSafe's installed configs."""
    import omnisafe
    pkg_dir = os.path.dirname(os.path.abspath(omnisafe.__file__))
    default_path = os.path.join(pkg_dir, 'configs', 'off-policy', 'SACLag.yaml')
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
    # 1. Load SACLag defaults
    defaults = load_saclag_defaults()

    # 2. Load our custom config
    with open(cfg_path) as f:
        custom = yaml.safe_load(f)

    algo = custom.pop('algo', 'SACLagMulti')
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

    print(f"{'=' * 60}")
    print(f"  SACLagMulti Training (Off-Policy Multi-Lambda Safe RL)")
    print(f"  Algo: {algo}")
    print(f"  Env: {env_id}")
    print(f"  Seed: {seed}")
    print(f"  Epochs: {cfgs.train_cfgs.epochs}")
    print(f"  Steps/epoch: {cfgs.algo_cfgs.steps_per_epoch}")
    print(f"  Replay buffer: {cfgs.algo_cfgs.size}")
    print(f"  Batch size: {cfgs.algo_cfgs.batch_size}")
    print(f"  Start learning: {cfgs.algo_cfgs.start_learning_steps}")
    print(f"  Auto alpha: {cfgs.algo_cfgs.auto_alpha}")
    print(f"  Warmup epochs: {cfgs.algo_cfgs.warmup_epochs}")
    if hasattr(cfgs, 'multi_cfgs'):
        print(f"  Softmax tau: {cfgs.multi_cfgs.tau}")
        print(f"  Per-constraint cost limits:")
        for i in range(5):
            lim = float(getattr(cfgs.multi_cfgs, f'cost_limit_{i}', 5000.0))
            print(f"    C{i}: {lim:.0f}")
    print(f"  Lambda LR: {cfgs.lagrange_cfgs.lambda_lr}")
    print(f"  Lambda init: {cfgs.lagrange_cfgs.lagrangian_multiplier_init}")
    ub = getattr(cfgs.lagrange_cfgs, 'lagrangian_upper_bound', None)
    print(f"  Lambda upper bound: {'None (uncapped)' if ub is None else ub}")
    print(f"{'=' * 60}")

    # 5. Direct instantiation (bypasses omnisafe.Agent)
    agent = SACLagMulti(env_id=env_id, cfgs=cfgs)

    ep_ret, ep_cost, ep_len = agent.learn()

    print(f"\nTraining complete.")
    print(f"  Final EpRet: {ep_ret:.1f}")
    print(f"  Final EpCost: {ep_cost:.1f}")
    print(f"  Final EpLen: {ep_len:.0f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Train SACLagMulti on CityLearn V2G'
    )
    ap.add_argument('--cfg', required=True, help='Path to YAML config')
    args = ap.parse_args()
    main(args.cfg)
