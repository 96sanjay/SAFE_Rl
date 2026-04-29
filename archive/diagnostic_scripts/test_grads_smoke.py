#!/usr/bin/env python3
"""Quick smoke test for PPOLag + GradS (500 steps, ~1 minute)."""
import os
import sys

PROJECT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)

# Env vars (minimal set for 5 buildings)
os.environ["CITYLEARN_SCHEMA"] = f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json"
os.environ["CITYLEARN_CENTRAL_AGENT"] = "1"
os.environ["CITYLEARN_REWARD_TYPE"] = "stems"
os.environ["CITYLEARN_EXPORT_FACTOR"] = "0.7"
os.environ["CITYLEARN_STEMS_P_BUILDING_MAX"] = "4.6083"
os.environ["CITYLEARN_STEMS_P_GRID_MAX"] = "10.2352"
os.environ["CITYLEARN_STEMS_SOC_LOW"] = "0.0"
os.environ["CITYLEARN_STEMS_SOC_HIGH"] = "0.95"
os.environ["CITYLEARN_STEMS_PNORM_P"] = "4.0"
os.environ["CITYLEARN_W_COST_EV"] = "1.0"
os.environ["CITYLEARN_W_COST_SOC"] = "10.0"
os.environ["CITYLEARN_W_COST_BUILDING"] = "0.5"
os.environ["CITYLEARN_W_COST_GRID"] = "0.05"
os.environ["CITYLEARN_EV_COST_SCALE"] = "3.0"
os.environ["CITYLEARN_EV_MISSING_ACTION_MODE"] = "assume_full"
os.environ["CITYLEARN_INCLUDE_EV_COST"] = "1"
os.environ["CITYLEARN_EV_DENSE_COST_SCALE"] = "1.0"
os.environ["CITYLEARN_C3_CONTROLLABLE"] = "1"
os.environ["CITYLEARN_SPATIAL_OBS"] = "1"
os.environ["STEMS_ALPHA_GRID"] = "1.0"
os.environ["STEMS_BETA_RAMP"] = "2.0"
os.environ["COST_W_C1"] = "10.0"
os.environ["COST_W_C1_DENSE"] = "5.0"
os.environ["COST_W_C2"] = "1.0"
os.environ["COST_W_C3"] = "0.1"
os.environ["COST_W_C4"] = "5.0"

import citylearn_safe.omni_env
import citylearn_safe.omni_env_v2
from citylearn_safe.grads.ppo_lag_grads import PPOLagGradS
from omnisafe.utils.config import Config

import yaml

# Load PPOLag defaults
import omnisafe
pkg_dir = os.path.dirname(os.path.abspath(omnisafe.__file__))
with open(os.path.join(pkg_dir, 'configs', 'on-policy', 'PPOLag.yaml')) as f:
    defaults = yaml.safe_load(f).get('defaults', {})

# Override for smoke test
defaults['seed'] = 42
defaults['env_id'] = 'CityLearnSafety-V2G-v2'
defaults['algo'] = 'PPOLagGradS'
defaults['exp_name'] = 'PPOLagGradS-{CityLearnSafety-V2G-v2}'
defaults['train_cfgs']['total_steps'] = 8759    # 1 full episode
defaults['train_cfgs']['vector_env_nums'] = 1
defaults['train_cfgs']['epochs'] = 1
defaults['algo_cfgs']['steps_per_epoch'] = 8759
defaults['algo_cfgs']['update_iters'] = 2        # minimal updates
defaults['algo_cfgs']['batch_size'] = 256
defaults['algo_cfgs']['obs_normalize'] = True
defaults['algo_cfgs']['reward_normalize'] = True
defaults['algo_cfgs']['cost_normalize'] = False
defaults['algo_cfgs']['entropy_coef'] = 0.005
defaults['lagrange_cfgs']['cost_limit'] = 24400.0
defaults['lagrange_cfgs']['lagrangian_multiplier_init'] = 0.1
defaults['lagrange_cfgs']['lambda_lr'] = 0.05
defaults['lagrange_cfgs']['lambda_optimizer'] = 'SGD'
defaults['model_cfgs']['actor']['hidden_sizes'] = [64, 64]
defaults['model_cfgs']['actor']['lr'] = 0.0005
defaults['model_cfgs']['critic']['hidden_sizes'] = [64, 64]
defaults['model_cfgs']['critic']['lr'] = 0.001
defaults['model_cfgs']['linear_lr_decay'] = False
defaults['logger_cfgs']['log_dir'] = './runs/grads_smoke_test'
defaults['logger_cfgs']['save_model_freq'] = 1
defaults['logger_cfgs']['window_lens'] = 1

# Add grads_cfgs with per-constraint cost limits
defaults['grads_cfgs'] = {
    'sim_threshold': 0.8,
    'conflict_threshold': 0.999,
    'cost_limit_0': 650,      # C1: EV departure
    'cost_limit_1': 36500,    # C1d: EV dense
    'cost_limit_2': 137000,   # C2: Battery SoC
    'cost_limit_3': 137000,   # C3: Building power
    'cost_limit_4': 83000,    # C4: Grid power
}

cfgs = Config(**defaults)

print("=" * 60)
print("  SMOKE TEST: PPOLag + GradS (8759 steps, 1 epoch)")
print("=" * 60)

try:
    agent = PPOLagGradS(env_id='CityLearnSafety-V2G-v2', cfgs=cfgs)
    ep_ret, ep_cost, ep_len = agent.learn()
    print(f"\n{'=' * 60}")
    print(f"  SMOKE TEST PASSED!")
    print(f"  EpRet: {ep_ret:.1f}")
    print(f"  EpCost: {ep_cost:.1f}")
    print(f"  EpLen: {ep_len:.0f}")
    print(f"{'=' * 60}")
except Exception as e:
    print(f"\n{'=' * 60}")
    print(f"  SMOKE TEST FAILED!")
    print(f"  Error: {e}")
    print(f"{'=' * 60}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
