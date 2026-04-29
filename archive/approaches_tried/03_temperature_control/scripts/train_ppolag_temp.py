#!/usr/bin/env python3
"""Train PPOLag for the temperature single-constraint case study."""
from __future__ import annotations

import argparse
import os
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import omnisafe
import citylearn_safe.omni_env_temp  # noqa: F401 - registers CityLearnTemp-Comfort-v0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to YAML config")
    ap.add_argument("--smoke_1epoch", action="store_true")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    algo = cfg.pop("algo", "PPOLag")
    env_id = cfg.pop("env_id", "CityLearnTemp-Comfort-v0")
    cfg.pop("seed", None)

    allowed = (
        "train_cfgs",
        "algo_cfgs",
        "logger_cfgs",
        "lagrange_cfgs",
        "model_cfgs",
        "save_cfgs",
        "env_cfgs",
    )
    custom_cfgs = {k: v for k, v in cfg.items() if k in allowed}

    if args.smoke_1epoch:
        custom_cfgs.setdefault("train_cfgs", {})
        custom_cfgs.setdefault("algo_cfgs", {})
        custom_cfgs["train_cfgs"]["total_steps"] = 2207
        custom_cfgs["algo_cfgs"]["steps_per_epoch"] = 2207
        custom_cfgs["algo_cfgs"]["update_iters"] = 5

    print(f"\n{'=' * 56}")
    print("  PPOLag — Temperature Single-Constraint Case Study")
    print(f"  Env: {env_id}")
    print(f"  Schema: {os.environ.get('CITYLEARN_SCHEMA', '(default temp schema)')}")
    print(f"  Log: {custom_cfgs.get('logger_cfgs', {}).get('log_dir', '(default)')}")
    print(f"{'=' * 56}\n")

    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()


if __name__ == "__main__":
    main()
