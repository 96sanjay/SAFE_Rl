#!/usr/bin/env python3
"""Train masked single-constraint PPO-Lag for the temperature case study."""
from __future__ import annotations

import argparse
import copy
import os
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import citylearn_safe.omni_env_temp_masked  # noqa: F401 - registers env
from citylearn_safe.ppo_lag_temp_masked import PPOLagTempMasked
from omnisafe.utils.config import Config


def load_ppolag_defaults() -> dict:
    """Load OmniSafe's default PPOLag config as the base config."""
    import omnisafe

    pkg_dir = os.path.dirname(os.path.abspath(omnisafe.__file__))
    default_path = os.path.join(pkg_dir, "configs", "on-policy", "PPOLag.yaml")
    with open(default_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("defaults", raw)


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge nested config dictionaries."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def build_cfg(cfg_path: str) -> tuple[str, Config]:
    """Merge custom YAML with OmniSafe PPOLag defaults."""
    defaults = load_ppolag_defaults()

    with open(cfg_path, encoding="utf-8") as f:
        custom = yaml.safe_load(f)

    algo = custom.pop("algo", "PPOLagTempMasked")
    env_id = custom.pop("env_id", "CityLearnTemp-Comfort-Masked-v0")
    seed = int(custom.pop("seed", 42))

    merged = deep_update(defaults, custom)
    merged["algo"] = algo
    merged["env_id"] = env_id
    merged["seed"] = seed
    merged["exp_name"] = f"{algo}-{{{env_id}}}"

    total_steps = int(merged["train_cfgs"]["total_steps"])
    steps_per_epoch = int(merged["algo_cfgs"]["steps_per_epoch"])
    merged["train_cfgs"]["epochs"] = total_steps // steps_per_epoch
    merged["train_cfgs"].setdefault("device", "cpu")

    return env_id, Config(**merged)


def main(cfg_path: str) -> None:
    """Load config, instantiate masked PPO-Lag, and train."""
    os.environ.setdefault("CITYLEARN_ACTION_MASK", "0")
    os.environ.setdefault("CITYLEARN_POLICY_ACTION_MASK", "1")

    env_id, cfgs = build_cfg(cfg_path)

    print(f"\n{'=' * 56}")
    print("  PPOLagTempMasked — Temperature Single-Constraint")
    print(f"  Env: {env_id}")
    print(f"  Schema: {os.environ.get('CITYLEARN_SCHEMA', '(default temp schema)')}")
    print(f"  Epochs: {cfgs.train_cfgs.epochs}")
    print(f"  Steps/epoch: {cfgs.algo_cfgs.steps_per_epoch}")
    print(f"  Cost limit: {cfgs.lagrange_cfgs.cost_limit}")
    print(f"{'=' * 56}\n")

    agent = PPOLagTempMasked(env_id=env_id, cfgs=cfgs)
    ep_ret, ep_cost, ep_len = agent.learn()

    print("\nTraining complete.")
    print(f"  Final EpRet: {ep_ret:.3f}")
    print(f"  Final EpCost: {ep_cost:.3f}")
    print(f"  Final EpLen: {ep_len:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.cfg)
