#!/usr/bin/env python3
"""Train paper-style CSAC-LB on the cooling-only temperature case study."""
from __future__ import annotations

import argparse
import copy
import os
import sys

import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import citylearn_safe.omni_env_temp_cooling_only  # noqa: F401
from citylearn_safe.omni_env_temp import DEFAULT_TEMP_SCHEMA
from citylearn_safe.csac_lb_temp import CSACLBTemp
from omnisafe.utils.config import Config


def maybe_init_residual_actor(agent: CSACLBTemp, raw_cfg: dict) -> None:
    if raw_cfg.get("env_id") != "CityLearnTemp-CoolingResidual-CSACLB-v0":
        return
    if not bool(raw_cfg.get("init_residual_zero_mean", True)):
        return

    actor = agent._actor_critic.actor
    net = getattr(actor, "net", None)
    final_linear = None
    if net is not None:
        for module in reversed(list(net)):
            if isinstance(module, torch.nn.Linear):
                final_linear = module
                break
    if final_linear is None:
        raise RuntimeError("Residual zero-mean init requires GaussianSACActor.net final layer")

    with torch.no_grad():
        final_linear.weight.zero_()
        final_linear.bias.zero_()
        act_dim = agent._env.action_space.shape[0]
        log_std_bias = float(raw_cfg.get("init_residual_log_std_bias", -4.0))
        final_linear.bias[act_dim:].fill_(log_std_bias)


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

    algo = custom.pop("algo", "CSACLBTemp")
    env_id = custom.pop("env_id", "CityLearnTemp-CoolingOnly-CSACLB-v0")
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
    os.environ.setdefault("CITYLEARN_USE_DEFAULT_TEMP_SCHEMA", "1")
    with open(cfg_path, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    for key, value in (raw_cfg.get("env_overrides") or {}).items():
        os.environ[str(key)] = str(value)
    init_checkpoint = raw_cfg.get("init_checkpoint")

    env_id, cfgs = build_cfg(cfg_path)
    print(f"\n{'=' * 56}")
    print("  CSACLBTemp — Paper-Style Temperature Control")
    print(f"  Env: {env_id}")
    print(f"  Schema: {os.environ.get('CITYLEARN_SCHEMA', DEFAULT_TEMP_SCHEMA)}")
    print(f"  Epochs: {cfgs.train_cfgs.epochs}")
    print(f"  Steps/epoch: {cfgs.algo_cfgs.steps_per_epoch}")
    print(f"{'=' * 56}\n")

    agent = CSACLBTemp(env_id=env_id, cfgs=cfgs)
    if raw_cfg.get("env_id") == "CityLearnTemp-CoolingResidual-CSACLB-v0":
        import torch

        maybe_init_residual_actor(agent, raw_cfg)
        print(
            "  Residual actor init: zero mean, "
            f"log_std_bias={float(raw_cfg.get('init_residual_log_std_bias', -4.0)):.1f}",
        )
    if init_checkpoint:
        import torch

        state = torch.load(str(init_checkpoint), map_location="cpu", weights_only=False)
        agent._actor_critic.actor.load_state_dict(state["pi"], strict=True)
        print(f"  Warm start actor: {init_checkpoint}")
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
