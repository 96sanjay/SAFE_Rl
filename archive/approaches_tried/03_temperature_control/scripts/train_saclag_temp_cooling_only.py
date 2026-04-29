#!/usr/bin/env python3
"""Train SAC-Lag with PID Lagrangian, action masking, and BC warm-start.

Usage:
    python scripts/train_saclag_temp_cooling_only.py \
        --cfg configs/off-policy/saclag_temp_cooling_only_v1.yaml
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
from citylearn_safe.sac_lag_temp_cooling_only import SACLagTempCoolingOnly
from omnisafe.utils.config import Config


def load_sac_defaults() -> dict:
    default_path = os.path.join(
        PROJECT_ROOT, "vendor_deps", "omnisafe", "configs", "off-policy", "SAC.yaml",
    )
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


def build_cfg(cfg_path: str) -> tuple[str, Config]:
    defaults = load_sac_defaults()
    with open(cfg_path, encoding="utf-8") as f:
        custom = yaml.safe_load(f)

    algo = custom.pop("algo", "SACLagTempCoolingOnly")
    env_id = custom.pop("env_id", "CityLearnTemp-CoolingOnly-Masked-Reward-v0")
    seed = int(custom.pop("seed", 42))

    merged = deep_update(defaults, custom)
    merged["algo"] = algo
    merged["env_id"] = env_id
    merged["seed"] = seed
    merged["exp_name"] = f"{algo}-{{{env_id}}}"
    merged["train_cfgs"]["epochs"] = (
        int(merged["train_cfgs"]["total_steps"]) // int(merged["algo_cfgs"]["steps_per_epoch"])
    )
    merged["train_cfgs"].setdefault("device", "cpu")
    return env_id, Config(**merged)


def main(cfg_path: str) -> None:
    os.environ.setdefault("CITYLEARN_ACTION_MASK", "0")
    os.environ.setdefault("CITYLEARN_POLICY_ACTION_MASK", "1")

    with open(cfg_path, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    env_id, cfgs = build_cfg(cfg_path)

    pid_cfgs = getattr(cfgs, "pid_lagrange_cfgs", None)
    pid_kp = float(getattr(pid_cfgs, "pid_kp", 1.0)) if pid_cfgs else 1.0
    pid_ki = float(getattr(pid_cfgs, "pid_ki", 0.1)) if pid_cfgs else 0.1
    penalty_max = float(getattr(pid_cfgs, "penalty_max", 100.0)) if pid_cfgs else 100.0
    cost_limit = float(getattr(pid_cfgs, "cost_limit", 40.0)) if pid_cfgs else 40.0

    print(f"\n{'=' * 60}")
    print("  SACLagTempCoolingOnly — PID Lagrangian + Masking + BC")
    print(f"  Env: {env_id}")
    print(f"  Epochs: {cfgs.train_cfgs.epochs}")
    print(f"  Steps/epoch: {cfgs.algo_cfgs.steps_per_epoch}")
    print(f"  Cost limit: {cost_limit}")
    print(f"  PID gains: Kp={pid_kp}, Ki={pid_ki}, max={penalty_max}")
    print(f"{'=' * 60}\n")

    agent = SACLagTempCoolingOnly(env_id=env_id, cfgs=cfgs)

    bc_cfg = getattr(cfgs, "bc_cfgs", None)
    if bc_cfg is not None:
        bc_epochs = int(getattr(bc_cfg, "epochs", 0))
        if bc_epochs > 0:
            agent.behavior_clone_warmstart(
                bc_epochs=bc_epochs,
                bc_rollout_steps=int(
                    getattr(bc_cfg, "rollout_steps", cfgs.algo_cfgs.steps_per_epoch)
                ),
                bc_batch_size=int(getattr(bc_cfg, "batch_size", 512)),
                seed=int(getattr(bc_cfg, "seed", cfgs.seed)),
            )

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
