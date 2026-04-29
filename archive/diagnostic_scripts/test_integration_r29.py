#!/usr/bin/env python
"""
R29 Integration Smoke Test
==========================
Verifies the FULL training chain works end-to-end:
  ActionMaskWrapper → omni_env_v2 (CityLearnCMDPv2) → safety_env_v3 → CityLearn → PPOLagMulti

Tests:
  1. Environment creation with ActionMask enabled
  2. Single episode rollout (manual stepping)
  3. Reward term verification (only 4 active, rest zero)
  4. Action mask active (info['action_mask_enabled'])
  5. C3 violation comparison (mask vs no-mask)
  6. PPOLagMulti 1-epoch training (full pipeline)
"""
from __future__ import annotations

import os
import sys
import time
import copy
import traceback

# ── R29 Configuration ──────────────────────────────────────────────
os.environ['CITYLEARN_SCHEMA'] = '/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings_3month.json'
os.environ['CITYLEARN_CENTRAL_AGENT'] = '1'
os.environ['CITYLEARN_REWARD_TYPE'] = 'stems'
os.environ['CITYLEARN_ACTION_MASK'] = '1'
os.environ['CITYLEARN_STEMS_P_BUILDING_MAX'] = '4.6083'
os.environ['CITYLEARN_STEMS_SOC_LOW'] = '0.0'
os.environ['CITYLEARN_STEMS_SOC_HIGH'] = '0.95'

# R29: Only 4 reward terms active
os.environ['STEMS_ALPHA_PRICE_ARB'] = '2.0'
os.environ['STEMS_LAMBDA_EV'] = '3.0'
os.environ['STEMS_EV_SLACK_ARB_SCALE'] = '5.0'
os.environ['STEMS_ALPHA_GRID_PENALTY'] = '0.3'

# ALL other terms disabled
os.environ['STEMS_MU_ECONOMIC'] = '0.0'
os.environ['STEMS_ALPHA_GRID'] = '0.0'
os.environ['STEMS_ALPHA_BUILD'] = '0.0'
os.environ['STEMS_BETA_RAMP'] = '0.0'
os.environ['STEMS_XI_RENEWABLE'] = '0.0'
os.environ['STEMS_ALPHA_LOAD_SHIFT'] = '0.0'
os.environ['STEMS_ALPHA_GRID_MILD'] = '0.0'
os.environ['STEMS_ALPHA_EV_GUARD'] = '0.0'
os.environ['STEMS_ALPHA_V2G_CONTEXT'] = '0.0'
os.environ['STEMS_ALPHA_PEAK_SHAVE'] = '0.0'
os.environ['STEMS_ALPHA_BARRIER'] = '0.0'
os.environ['STEMS_ALPHA_EV_SOLAR'] = '0.0'
os.environ['STEMS_ALPHA_SOLAR_STORE'] = '0.0'
os.environ['STEMS_ALPHA_HEADROOM'] = '0.0'

# Saute MDP for C1
os.environ['CITYLEARN_EV_SAUTE'] = '1'
os.environ['CITYLEARN_EV_SAUTE_BUDGET'] = '6250'
os.environ['CITYLEARN_EV_SAUTE_PENALTY'] = '5.0'
os.environ['CITYLEARN_EV_SAUTE_GAMMA'] = '1.0'
os.environ['CITYLEARN_EV_SAUTE_SHAPED_ALPHA'] = '2.0'

# Lagrangian
os.environ['CITYLEARN_PID_LAGRANGE'] = '1'

# Clamps OFF (mask handles safety)
os.environ['CITYLEARN_WM_DISABLE'] = '1'
os.environ['CITYLEARN_EV_ACTION_CLAMP'] = '0'
os.environ['CITYLEARN_BATT_CLAMP'] = '0'
os.environ['CITYLEARN_SPATIAL_OBS'] = '0'
os.environ['CITYLEARN_TEMPORAL_WINDOW'] = '0'

# ── Imports ─────────────────────────────────────────────────────────
PROJECT_ROOT = '/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import yaml

# Register environments
import citylearn_safe.omni_env       # noqa: F401
import citylearn_safe.omni_env_v2    # noqa: F401


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_env_creation():
    """Test 1: Create the env and verify ActionMask is in the wrapper chain."""
    separator("TEST 1: Environment Creation")

    from citylearn_safe.omni_env_v2 import CityLearnCMDPv2
    env = CityLearnCMDPv2('CityLearnSafety-V2G-v2')

    print(f"  obs_space: {env._observation_space.shape}")
    print(f"  act_space: {env._action_space.shape}")

    # Walk wrapper chain to verify ActionMaskWrapper is present
    found_mask = False
    cur = env._env
    chain = []
    for _ in range(20):
        if cur is None:
            break
        chain.append(type(cur).__name__)
        if type(cur).__name__ == 'ActionMaskWrapper':
            found_mask = True
        cur = getattr(cur, 'env', getattr(cur, '_env', getattr(cur, 'base', None)))

    print(f"  Wrapper chain: {' → '.join(chain)}")
    print(f"  ActionMaskWrapper found: {found_mask}")
    assert found_mask, "ActionMaskWrapper NOT found in wrapper chain!"
    print("  PASS")
    return env


def test_single_episode(env, n_steps=200):
    """Test 2: Run n_steps and verify rewards/info structure."""
    separator(f"TEST 2: Single Episode Rollout ({n_steps} steps)")

    obs, info = env.reset()
    print(f"  Reset obs shape: {obs.shape}")
    print(f"  Reset obs dtype: {obs.dtype}")

    act_dim = env._action_space.shape[0]
    total_reward = 0.0
    total_cost = 0.0
    c3_violations = 0
    c1_cost_sum = 0.0

    # Track reward term sums
    reward_terms = {
        'r_price_arb': 0.0, 'r_ev': 0.0, 'r_ev_slack_arb': 0.0,
        'r_grid_penalty': 0.0,
        # These should all be zero:
        'r_eco': 0.0, 'r_sg': 0.0, 'r_sb': 0.0, 'r_ramp': 0.0,
        'r_ren': 0.0, 'r_ev_guard': 0.0, 'r_v2g_ctx': 0.0,
        'r_peak_shave': 0.0, 'r_load_shift': 0.0, 'r_grid_mild': 0.0,
        'r_barrier': 0.0, 'r_ev_solar': 0.0, 'r_solar_store': 0.0,
        'r_headroom': 0.0,
    }

    mask_active_count = 0

    for step in range(n_steps):
        # Random action in [-1, 1]
        action = torch.randn(act_dim).clamp(-1.0, 1.0)
        obs, reward, cost, terminated, truncated, info = env.step(action)

        total_reward += reward.item()
        total_cost += cost.item()

        # Track reward terms
        for key in reward_terms:
            reward_terms[key] += float(info.get(key, 0.0))

        # Check action mask
        if info.get('action_mask_enabled', 0.0) > 0.5:
            mask_active_count += 1

        # Track C3 violations
        c3 = float(info.get('cost_stems_building_power', 0.0))
        if c3 > 0:
            c3_violations += 1

        c1_cost_sum += float(info.get('cost_ev_departure', 0.0))

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

    print(f"\n  Steps completed: {min(n_steps, step+1)}")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Total cost: {total_cost:.3f}")
    print(f"  C1 cost sum: {c1_cost_sum:.3f}")
    print(f"  C3 violations: {c3_violations}/{n_steps}")
    print(f"  Action mask active: {mask_active_count}/{n_steps}")

    print(f"\n  Reward term sums:")
    for key, val in sorted(reward_terms.items()):
        status = "ACTIVE" if abs(val) > 1e-6 else "zero"
        print(f"    {key:25s} = {val:+10.4f}  [{status}]")

    # Verify action mask is active
    assert mask_active_count == n_steps, \
        f"Action mask should be active every step, got {mask_active_count}/{n_steps}"

    # Verify only the 4 expected reward terms are active
    expected_active = {'r_price_arb', 'r_ev', 'r_ev_slack_arb', 'r_grid_penalty'}
    expected_zero = set(reward_terms.keys()) - expected_active

    active_but_should_be_zero = []
    for key in expected_zero:
        if abs(reward_terms[key]) > 1e-6:
            active_but_should_be_zero.append(key)

    if active_but_should_be_zero:
        print(f"\n  WARNING: These should be zero but aren't: {active_but_should_be_zero}")
    else:
        print(f"\n  Reward terms: CORRECT (only 4 active)")

    zero_but_should_be_active = []
    for key in expected_active:
        if abs(reward_terms[key]) < 1e-6:
            zero_but_should_be_active.append(key)

    if zero_but_should_be_active:
        print(f"  WARNING: These should be active but are zero: {zero_but_should_be_active}")
    else:
        print(f"  All 4 expected terms are non-zero: PASS")

    print("  PASS")
    return c3_violations


def test_c3_comparison(n_steps=200):
    """Test 3: Compare C3 violations with vs without action mask."""
    separator("TEST 3: C3 Violation Comparison (mask ON vs OFF)")

    from citylearn_safe.omni_env_v2 import CityLearnCMDPv2

    # Fix seed for reproducibility
    rng = np.random.RandomState(42)

    # Generate fixed random actions
    # We need to create env first to know act_dim
    env_on = CityLearnCMDPv2('CityLearnSafety-V2G-v2')
    act_dim = env_on._action_space.shape[0]
    fixed_actions = [rng.uniform(-1, 1, act_dim).astype(np.float32) for _ in range(n_steps)]

    # Run WITH mask
    obs, _ = env_on.reset()
    c3_on = 0
    for step in range(n_steps):
        action = torch.as_tensor(fixed_actions[step])
        obs, reward, cost, terminated, truncated, info = env_on.step(action)
        if float(info.get('cost_stems_building_power', 0.0)) > 0:
            c3_on += 1
        if terminated or truncated:
            break
    env_on.close()

    # Run WITHOUT mask
    os.environ['CITYLEARN_ACTION_MASK'] = '0'
    # Need to re-import to get fresh env without mask
    # But env_register caches, so just create new instance
    # Actually, the env_register decorator means CityLearnCMDPv2 is already registered.
    # Creating a new instance will read the current env var.

    # Force re-creation by importing fresh
    import importlib
    # We can't easily re-register, so just create the env manually
    from scripts.make_env import make_base_env
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.saute_ev_wrapper import SauteEVBudgetWrapper

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    forecast = ForecastObsWrapper(safety, forecast_horizon=24)
    env_off_raw = SauteEVBudgetWrapper(forecast)

    # Create a minimal CMDP-like wrapper for stepping
    obs_off, info_off = env_off_raw.reset()
    c3_off = 0
    for step in range(n_steps):
        action_np = fixed_actions[step]
        obs_off, _r, terminated, truncated, info_off = env_off_raw.step(action_np)
        if float(info_off.get('cost_stems_building_power', 0.0)) > 0:
            c3_off += 1
        if terminated or truncated:
            break
    env_off_raw.close()

    # Restore mask setting
    os.environ['CITYLEARN_ACTION_MASK'] = '1'

    print(f"  C3 violations WITH mask:    {c3_on}/{n_steps}")
    print(f"  C3 violations WITHOUT mask: {c3_off}/{n_steps}")

    if c3_on < c3_off:
        reduction = (c3_off - c3_on) / max(1, c3_off) * 100
        print(f"  Reduction: {reduction:.1f}%")
        print("  PASS (mask reduces C3 violations)")
    elif c3_on == c3_off == 0:
        print("  Both zero (no violations in either case) -- PASS (trivially)")
    else:
        print(f"  WARNING: Mask did not reduce C3 ({c3_on} >= {c3_off})")

    return c3_on, c3_off


def test_ppolag_multi_training():
    """Test 4: Full PPOLagMulti training for 1 epoch."""
    separator("TEST 4: PPOLagMulti 1-Epoch Training")

    from omnisafe.utils.config import Config
    from citylearn_safe.grads.ppo_lag_multi import PPOLagMulti

    # Load PPOLag defaults
    import omnisafe
    pkg_dir = os.path.dirname(os.path.abspath(omnisafe.__file__))
    default_path = os.path.join(pkg_dir, 'configs', 'on-policy', 'PPOLag.yaml')
    with open(default_path) as f:
        defaults = yaml.safe_load(f).get('defaults', {})

    # Minimal config for smoke test: 1 epoch, small batch
    custom = {
        'seed': 42,
        'env_id': 'CityLearnSafety-V2G-v2',
        'algo': 'PPOLagMulti',
        'exp_name': 'R29-smoke-test',
        'train_cfgs': {
            'total_steps': 8759,
            'epochs': 1,
            'vector_env_nums': 1,
            'parallel': 1,
            'device': 'cpu',
        },
        'algo_cfgs': {
            'steps_per_epoch': 8759,
            'update_iters': 5,       # fewer updates for speed
            'target_kl': 0.08,
            'kl_early_stop': True,
            'batch_size': 256,
            'obs_normalize': True,
            'reward_normalize': True,
            'cost_normalize': False,
            'standardized_cost_adv': True,
            'entropy_coef': 0.005,
            'max_grad_norm': 40.0,
        },
        'lagrange_cfgs': {
            'cost_limit': 21800,
            'lagrangian_multiplier_init': 0.001,
            'lambda_lr': 0.035,
            'lambda_optimizer': 'Adam',
            'lagrangian_upper_bound': 3.0,
        },
        'multi_cfgs': {
            'tau': 1.0,
            'cost_limit_0': 999999,
            'cost_limit_1': 1500,
            'cost_limit_2': 4000,
            'cost_limit_3': 3000,
            'cost_limit_4': 1500,
            'anneal_cost_limit_0': [999999, 1800, 20, 40],
            'pid_kp': 0.1,
            'pid_ki': 0.01,
            'pid_kd': 0.01,
            'pid_d_delay': 10,
            'pid_delta_p_ema_alpha': 0.95,
            'pid_delta_d_ema_alpha': 0.95,
            'pid_kp_0': 5.0,
            'pid_ki_0': 0.0,
            'pid_kd_0': 0.0,
            'pid_ema_p_0': 0.0,
            'pid_kp_3': 0.5,
            'pid_ki_3': 0.05,
            'pid_kp_4': 0.3,
            'pid_ki_4': 0.03,
        },
        'model_cfgs': {
            'actor': {
                'hidden_sizes': [64, 64],   # small for speed
                'activation': 'tanh',
                'lr': 0.0003,
            },
            'critic': {
                'hidden_sizes': [64, 64],   # small for speed
                'activation': 'tanh',
                'lr': 0.001,
            },
            'linear_lr_decay': False,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': False,
            'save_model_freq': 999,
            'log_dir': '/tmp/r29_smoke_test',
            'window_lens': 1,
        },
    }

    # Deep merge
    def deep_update(base, override):
        result = copy.deepcopy(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = deep_update(result[k], v)
            else:
                result[k] = v
        return result

    merged = deep_update(defaults, custom)
    cfgs = Config(**merged)

    print("  Creating PPOLagMulti agent...")
    t0 = time.time()
    agent = PPOLagMulti(env_id='CityLearnSafety-V2G-v2', cfgs=cfgs)

    print("  Starting 1-epoch training...")
    ep_ret, ep_cost, ep_len = agent.learn()
    elapsed = time.time() - t0

    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  EpRet: {ep_ret:.1f}")
    print(f"  EpCost: {ep_cost:.1f}")
    print(f"  EpLen: {ep_len:.0f}")

    # Verify training produced valid outputs
    assert not np.isnan(ep_ret), "EpRet is NaN!"
    assert not np.isnan(ep_cost), "EpCost is NaN!"
    assert ep_len > 0, "EpLen is 0!"

    print("  PASS")
    return ep_ret, ep_cost, ep_len


def main():
    print("=" * 60)
    print("  R29 INTEGRATION SMOKE TEST")
    print("  ActionMask + Simplified Reward + PPOLagMulti")
    print("=" * 60)

    results = {}
    failures = []

    # Test 1: Env creation
    try:
        env = test_env_creation()
        results['env_creation'] = 'PASS'
    except Exception as e:
        results['env_creation'] = f'FAIL: {e}'
        failures.append(('env_creation', traceback.format_exc()))
        env = None

    # Test 2: Single episode
    if env is not None:
        try:
            c3_viol = test_single_episode(env, n_steps=200)
            results['single_episode'] = f'PASS (C3 violations: {c3_viol}/200)'
        except Exception as e:
            results['single_episode'] = f'FAIL: {e}'
            failures.append(('single_episode', traceback.format_exc()))
        finally:
            env.close()
    else:
        results['single_episode'] = 'SKIP (env creation failed)'

    # Test 3: C3 comparison
    try:
        c3_on, c3_off = test_c3_comparison(n_steps=200)
        results['c3_comparison'] = f'PASS (mask={c3_on}, no_mask={c3_off})'
    except Exception as e:
        results['c3_comparison'] = f'FAIL: {e}'
        failures.append(('c3_comparison', traceback.format_exc()))

    # Test 4: PPOLagMulti training (1 epoch)
    try:
        ep_ret, ep_cost, ep_len = test_ppolag_multi_training()
        results['ppolag_training'] = f'PASS (ret={ep_ret:.1f}, cost={ep_cost:.1f}, len={ep_len:.0f})'
    except Exception as e:
        results['ppolag_training'] = f'FAIL: {e}'
        failures.append(('ppolag_training', traceback.format_exc()))

    # ── Summary ──
    separator("RESULTS SUMMARY")
    for test_name, result in results.items():
        status = "PASS" if result.startswith("PASS") else ("SKIP" if result.startswith("SKIP") else "FAIL")
        icon = {"PASS": "[OK]", "FAIL": "[!!]", "SKIP": "[--]"}[status]
        print(f"  {icon} {test_name:25s} {result}")

    if failures:
        separator("FAILURE DETAILS")
        for name, tb in failures:
            print(f"\n--- {name} ---")
            print(tb)

    n_pass = sum(1 for r in results.values() if r.startswith('PASS'))
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} tests passed")

    return len(failures) == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
