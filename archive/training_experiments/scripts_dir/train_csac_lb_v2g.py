#!/usr/bin/env python3
"""Train CSAC-LB on the V2G multi-building environment."""
from __future__ import annotations

import argparse
import copy
import os
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# IMPORTANT: import omni_env_v2 FIRST so its CityLearnV2G env class is
# registered in OmniSafe's env registry. CSAC-LB now reuses this stock
# R28 env (same reward, same obs space) instead of its own duplicate —
# this is what makes CSAC-LB's EpRet/EpCost directly comparable to every
# other R28 row in the benchmark table.
import citylearn_safe.omni_env_v2  # noqa: F401  registers CityLearnV2G (stock R28 env)
import citylearn_safe.csac_lb_v2g  # noqa: F401  registers CSACLBV2G algorithm
from citylearn_safe.csac_lb_v2g import CSACLBV2G
from omnisafe.utils.config import Config


def load_sac_defaults() -> dict:
    default_path = os.path.join(
        PROJECT_ROOT,
        "vendor_deps",
        "omnisafe",
        "configs",
        "off-policy",
        "SAC.yaml",
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

    algo = custom.pop("algo", "CSACLBV2G")
    env_id = custom.pop("env_id", "CityLearnSafety-V2G-v2")
    seed = int(custom.pop("seed", 0))

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
    with open(cfg_path, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    schema = raw_cfg.get("env_overrides", {}).get("CITYLEARN_SCHEMA") or os.environ.get("CITYLEARN_SCHEMA", "")
    if not schema:
        schema = os.path.join(
            PROJECT_ROOT,
            "data",
            "citylearn_challenge_2022_phase_all_plus_evs",
            "schema_5buildings.json",
        )
    os.environ["CITYLEARN_SCHEMA"] = schema

    for key, value in (raw_cfg.get("env_overrides") or {}).items():
        os.environ[str(key)] = str(value)

    env_id, cfgs = build_cfg(cfg_path)
    print(f"\n{'=' * 60}")
    print("  CSAC-LB V2G — Multi-Building Safe EV Control")
    print(f"  Env:    {env_id}")
    print(f"  Schema: {schema}")
    print(f"  Epochs: {cfgs.train_cfgs.epochs}")
    print(f"  Steps/epoch: {cfgs.algo_cfgs.steps_per_epoch}")
    print(f"  Cost limit:  {cfgs.algo_cfgs.cost_limit}")
    print(f"  Barrier factor: {cfgs.algo_cfgs.barrier_factor}")
    print(f"{'=' * 60}\n")

    agent = CSACLBV2G(env_id=env_id, cfgs=cfgs)

    init_checkpoint = raw_cfg.get("init_checkpoint")
    if init_checkpoint:
        import torch
        state = torch.load(str(init_checkpoint), map_location="cpu", weights_only=False)
        agent._actor_critic.actor.load_state_dict(state["pi"], strict=True)
        print(f"  Warm-start actor: {init_checkpoint}")

    ep_ret, ep_cost, ep_len = agent.learn()
    print("\nTraining complete.")
    print(f"  Final EpRet:  {ep_ret:.3f}")
    print(f"  Final EpCost: {ep_cost:.3f}")
    print(f"  Final EpLen:  {ep_len:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.cfg)
