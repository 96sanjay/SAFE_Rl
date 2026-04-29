# scripts/train_omnisafe_ev.py
"""
Training script for CityLearn Safe RL with AGGREGATE district-level constraints.

This script registers and uses CityLearnSafety-Aggregate-v0 which enforces
district-level average SOC constraints instead of per-building constraints.

Usage:
    python scripts/train_omnisafe_ev.py --cfg configs/on-policy/your_config.yaml
"""
from __future__ import annotations
import argparse
import yaml
import os
import omnisafe

# ✅ Register aggregate constraint environment
import citylearn_safe.omni_env_aggregate  # CityLearnSafety-Aggregate-v0

# Also keep old registration available for backward compatibility
import citylearn_safe.omni_env  # CityLearnSafety-SoC-v0


def main(cfg_path: str) -> None:
    """Main training function."""
    
    # Load config
    cfg = yaml.safe_load(open(cfg_path, "r"))
    algo = cfg["algo"]
    env_id = cfg["env_id"]
    
    # Print configuration info
    print("=" * 80)
    print("SAFE RL TRAINING - AGGREGATE CONSTRAINTS")
    print("=" * 80)
    print(f"Algorithm: {algo}")
    print(f"Environment: {env_id}")
    print(f"Config file: {cfg_path}")
    print()
    
    # Print environment-specific info
    if "Aggregate" in env_id:
        print("✅ Using AGGREGATE district-level SOC constraints")
        print("   → District average SOC enforced (achievable)")
        print("   → Individual buildings may hit limits (expected)")
        print("   → Target: <10% violation rate")
    else:
        print("⚠️  Using per-building SOC constraints")
        print("   → Each building enforced individually")
        print("   → May have high violation rates due to physics")
    
    print()
    print("Environment Variables:")
    print(f"  W_COST_SOC: {os.environ.get('CITYLEARN_W_COST_SOC', 'not set')}")
    print(f"  W_COST_EV: {os.environ.get('CITYLEARN_W_COST_EV', 'not set')}")
    print(f"  SOC_LOW: {os.environ.get('CITYLEARN_STEMS_SOC_LOW', 'not set')}")
    print(f"  SOC_HIGH: {os.environ.get('CITYLEARN_STEMS_SOC_HIGH', 'not set')}")
    print(f"  KPI_RUN_NAME: {os.environ.get('CITYLEARN_KPI_RUN_NAME', 'not set')}")
    print("=" * 80)
    print()
    
    # Extract custom configs
    allowed = (
        "train_cfgs",
        "algo_cfgs",
        "logger_cfgs",
        "lagrange_cfgs",
        "model_cfgs",
        "save_cfgs",
        "env_cfgs",
        "reward_model_cfgs",
    )
    custom_cfgs = {k: v for k, v in cfg.items() if k in allowed}
    
    # Create and train agent
    print("Initializing OmniSafe agent...")
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    
    print("Starting training...")
    print("=" * 80)
    agent.learn()
    
    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train Safe RL agent on CityLearn with aggregate constraints"
    )
    ap.add_argument(
        "--cfg",
        required=True,
        help="Path to config YAML file (e.g., configs/on-policy/ppo_lag_1ep_aggregate_test.yaml)"
    )
    args = ap.parse_args()
    main(args.cfg)
