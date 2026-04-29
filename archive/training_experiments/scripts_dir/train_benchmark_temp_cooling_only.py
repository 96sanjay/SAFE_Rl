#!/usr/bin/env python3
"""Generic trainer for benchmarking OmniSafe algorithms on the cooling-only env.

Works with any registered OmniSafe algorithm (CPO, FOCOPS, CUP, PPOLag, etc.).

Usage:
    python scripts/train_benchmark_temp_cooling_only.py \
        --cfg configs/on-policy/cpo_temp_cooling_only.yaml
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import citylearn_safe.omni_env_temp_cooling_only  # noqa: F401 — registers env

from omnisafe.algorithms import registry
from omnisafe.utils.config import Config

ALGO_DEFAULTS = {
    "CPO": "on-policy/CPO.yaml",
    "FOCOPS": "on-policy/FOCOPS.yaml",
    "CUP": "on-policy/CUP.yaml",
    "PPO": "on-policy/PPO.yaml",
    "PPOLag": "on-policy/PPOLag.yaml",
    "TRPO": "on-policy/TRPO.yaml",
    "TRPOLag": "on-policy/TRPOLag.yaml",
}


def load_defaults(algo: str) -> dict:
    default_rel = ALGO_DEFAULTS.get(algo)
    if default_rel is None:
        raise ValueError(f"Unknown algo '{algo}'. Available: {list(ALGO_DEFAULTS.keys())}")
    default_path = os.path.join(PROJECT_ROOT, "vendor_deps", "omnisafe", "configs", default_rel)
    with open(default_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("defaults", raw)


def deep_update(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def build_cfg(cfg_path: str) -> tuple[str, str, Config]:
    with open(cfg_path, encoding="utf-8") as f:
        custom = yaml.safe_load(f)

    algo = custom.pop("algo")
    env_id = custom.pop("env_id", "CityLearnTemp-CoolingOnly-Masked-Reward-v0")
    seed = int(custom.pop("seed", 42))

    defaults = load_defaults(algo)
    merged = deep_update(defaults, custom)
    merged["algo"] = algo
    merged["env_id"] = env_id
    merged["seed"] = seed
    merged["exp_name"] = f"{algo}-{{{env_id}}}"
    merged["train_cfgs"]["epochs"] = (
        int(merged["train_cfgs"]["total_steps"]) // int(merged["algo_cfgs"]["steps_per_epoch"])
    )
    merged["train_cfgs"].setdefault("device", "cpu")
    return algo, env_id, Config(**merged)


def main(cfg_path: str) -> None:
    algo, env_id, cfgs = build_cfg(cfg_path)

    print(f"\n{'=' * 60}")
    print(f"  Benchmark: {algo}")
    print(f"  Env: {env_id}")
    print(f"  Epochs: {cfgs.train_cfgs.epochs}")
    print(f"  Steps/epoch: {cfgs.algo_cfgs.steps_per_epoch}")
    print(f"{'=' * 60}\n")

    AlgoClass = registry.get(algo)
    agent = AlgoClass(env_id=env_id, cfgs=cfgs)
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
