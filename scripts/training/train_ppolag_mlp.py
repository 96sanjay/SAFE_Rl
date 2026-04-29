#!/usr/bin/env python3
"""Train OmniSafe PPOLag with MLP on CityLearn (no STEMS encoder).

Simple wrapper around omnisafe.Agent for A/B comparison runs.

Usage:
    python scripts/train_ppolag_mlp.py --cfg configs/on-policy/r11a_ppolag_improved.yaml
    python scripts/train_ppolag_mlp.py --cfg configs/on-policy/r11a_ppolag_improved.yaml --smoke_1epoch
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import omnisafe
import citylearn_safe.cmdp_env  # noqa: F401 — registers CityLearnSafety-V2G-v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to YAML config")
    ap.add_argument("--smoke_1epoch", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    algo = cfg.pop("algo", "PPOLag")
    env_id = cfg.pop("env_id", "CityLearnSafety-V2G-v2")
    cfg.pop("seed", None)

    allowed = ("train_cfgs", "algo_cfgs", "logger_cfgs", "lagrange_cfgs",
               "model_cfgs", "save_cfgs", "env_cfgs")
    custom_cfgs = {k: v for k, v in cfg.items() if k in allowed}

    if args.smoke_1epoch:
        custom_cfgs.setdefault("train_cfgs", {})
        custom_cfgs.setdefault("algo_cfgs", {})
        custom_cfgs["train_cfgs"]["total_steps"] = 8759
        custom_cfgs["algo_cfgs"]["steps_per_epoch"] = 8759
        custom_cfgs["algo_cfgs"]["update_iters"] = 5

    print(f"\n{'=' * 50}")
    print(f"  {algo} — MLP baseline")
    print(f"  Env: {env_id}")
    print(f"  Log: {custom_cfgs.get('logger_cfgs', {}).get('log_dir', '(default)')}")
    print(f"{'=' * 50}\n")

    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)

    # Lambda cap (same monkeypatch as R10c)
    lambda_upper = float(os.environ.get('LAMBDA_UPPER_BOUND', '0'))
    if lambda_upper > 0 and hasattr(agent.agent, '_lagrange'):
        agent.agent._lagrange.lagrangian_upper_bound = lambda_upper
        print(f">>> LAMBDA CAP: {lambda_upper}")

    agent.learn()


if __name__ == "__main__":
    main()
