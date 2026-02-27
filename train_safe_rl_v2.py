#!/usr/bin/env python3
"""
Train Safe RL agent (FOCOPS or PPOLag) for CityLearn V2G energy management.

Usage:
    python train_safe_rl_v2.py --algo FOCOPS --epochs 250
    python train_safe_rl_v2.py --algo PPOLag --epochs 250
    python train_safe_rl_v2.py --algo FOCOPS --epochs 5 --tag test
"""
from __future__ import annotations

import os
import sys
import argparse
import datetime

PROJECT_ROOT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

ENV_VARS = {
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "2.273834",
    "CITYLEARN_STEMS_P_GRID_MAX": "27.127751",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_zero",
    "CITYLEARN_SCHEMA": "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "1.0",
    "CITYLEARN_REWARD_SCALE": "1.0",
    "CITYLEARN_EV_DENSE_COST_SCALE": "1.0",
    "CITYLEARN_EV_COST_SCALE": "1.0",
    "CITYLEARN_STEMS_BATTERY_COST_SCALE": "50.0",
    "CITYLEARN_STEMS_BUILDING_COST_SCALE": "1.0",
    "CITYLEARN_STEMS_GRID_COST_SCALE": "1.0",
    "STEMS_MU_ECONOMIC": "0.3",
    "STEMS_ALPHA_GRID": "3.0",
    "STEMS_ALPHA_BUILD": "2.0",
    "STEMS_BETA_RAMP": "0.5",
    "STEMS_XI_RENEWABLE": "0.2",
    "STEMS_LAMBDA_EV": "5.0",
    "COST_W_C1": "10.0",
    "COST_W_C1_DENSE": "5.0",
    "COST_W_C2": "1.0",
    "COST_W_C3": "0.1",
    "COST_W_C4": "5.0",
    "CITYLEARN_W_COST_EV": "10.0",
    "CITYLEARN_W_COST_SOC": "1.0",
    "CITYLEARN_W_COST_BUILDING": "0.1",
    "CITYLEARN_W_COST_GRID": "5.0",
}


def set_env_vars():
    for k, v in ENV_VARS.items():
        if k not in os.environ:
            os.environ[k] = v


def register_env():
    import omnisafe
    from omni_env_v2 import CityLearnSafetyEnvV2Omni
    omnisafe.register(
        id='CityLearnSafety-V2G-v2',
        entry_point='omni_env_v2:CityLearnSafetyEnvV2Omni',
        max_episode_steps=8760,
    )
    print("[Train] Registered CityLearnSafety-V2G-v2")


def get_focops_config(epochs, cost_limit, tag):
    return {
        'train_cfgs': {
            'total_steps': epochs * 8760,
            'vector_env_nums': 1,
            'parallel': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'steps_per_epoch': 8760,
            'update_iters': 40,
            'batch_size': 512,
            'use_max_grad_norm': True,
            'max_grad_norm': 0.5,
            'use_critic_norm': True,
            'critic_norm_coeff': 0.001,
            'gamma': 0.995,
            'cost_gamma': 0.99,
            'lam': 0.95,
            'lam_c': 0.95,
            'clip': 0.2,
            'penalty_coeff': 0.0,
            'kl_early_stop': 0.02,
        },
        'model_cfgs': {
            'actor_type': 'gaussian_learning',
            'linear_lr_decay': False,
            'actor': {
                'hidden_sizes': [256, 256],
                'activation': 'tanh',
                'lr': 3e-4,
            },
            'critic': {
                'hidden_sizes': [256, 256],
                'activation': 'tanh',
                'lr': 1e-3,
            },
        },
        'lagrange_cfgs': {
            'cost_limit': cost_limit,
            'lagrangian_multiplier_init': 1.0,
            'lambda_lr': 0.035,
            'lambda_optimizer': 'Adam',
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': True,
            'log_dir': os.path.join(PROJECT_ROOT, 'runs', f'focops_stems_v2_{tag}'),
            'save_model_freq': 10,
            'window_lens': 10,
        },
    }


def get_ppolag_config(epochs, cost_limit, tag):
    cfg = get_focops_config(epochs, cost_limit, tag)
    cfg['logger_cfgs']['log_dir'] = os.path.join(
        PROJECT_ROOT, 'runs', f'ppolag_stems_v2_{tag}')
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train Safe RL for CityLearn V2G")
    parser.add_argument('--algo', type=str, default='FOCOPS',
                        choices=['FOCOPS', 'PPOLag', 'CUP', 'CPO', 'TRPOLag'])
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--cost-limit', type=float, default=5.0)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.tag is None:
        args.tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"  Safe RL Training: {args.algo}")
    print(f"  Epochs: {args.epochs} | Cost Limit: {args.cost_limit}")
    print(f"  Seed: {args.seed} | Tag: {args.tag}")
    print("=" * 70)

    set_env_vars()

    import omnisafe
    register_env()

    if args.algo in ('PPOLag', 'TRPOLag'):
        custom_cfgs = get_ppolag_config(args.epochs, args.cost_limit, args.tag)
    else:
        custom_cfgs = get_focops_config(args.epochs, args.cost_limit, args.tag)

    print("\n[Config] STEMS Reward Weights:")
    for k in ['STEMS_MU_ECONOMIC', 'STEMS_ALPHA_GRID', 'STEMS_ALPHA_BUILD',
              'STEMS_BETA_RAMP', 'STEMS_XI_RENEWABLE', 'STEMS_LAMBDA_EV']:
        print(f"  {k} = {os.environ.get(k)}")

    print("\n[Config] CMDP Cost Weights:")
    for k in ['COST_W_C1', 'COST_W_C1_DENSE', 'COST_W_C2', 'COST_W_C3', 'COST_W_C4']:
        print(f"  {k} = {os.environ.get(k)}")

    print(f"\n[Config] Cost Limit: {args.cost_limit}")
    print(f"[Config] Log Dir: {custom_cfgs['logger_cfgs']['log_dir']}")

    agent = omnisafe.Agent(
        args.algo,
        'CityLearnSafety-V2G-v2',
        custom_cfgs=custom_cfgs,
        seed=args.seed
    )

    print("\n" + "=" * 70)
    print("  Starting Training...")
    print("  Expected: ~4-8 hours for 250 epochs")
    print("  Monitor: tensorboard --logdir runs/")
    print("=" * 70 + "\n")

    agent.learn()

    print("\n" + "=" * 70)
    print("  Training Complete!")
    print(f"  Checkpoints: {custom_cfgs['logger_cfgs']['log_dir']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
