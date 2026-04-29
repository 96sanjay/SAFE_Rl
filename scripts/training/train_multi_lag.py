# scripts/train_multi_lag.py
"""Training script for PPOLagMulti (per-constraint Lagrange + softmax advantage).

Bypasses omnisafe.Agent (which needs a default config YAML per algorithm)
and directly instantiates PPOLagMulti with the merged config.

Usage:
    python scripts/train_multi_lag.py --cfg configs/on-policy/ppolag_multi_5bld.yaml
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import yaml

# Register environments FIRST
import citylearn_safe.omni_env       # noqa: F401
import citylearn_safe.cmdp_env    # noqa: F401

from omnisafe.utils.config import Config

# Import PPOLagMulti and PPOLagGradS (triggers @registry.register)
from citylearn_safe.grads.ppo_lag_multi import PPOLagMulti
from citylearn_safe.grads.ppo_lag_grads import PPOLagGradS


def load_ppo_defaults() -> dict:
    """Load PPO default config from OmniSafe and inject cost-related defaults.

    PPOLagMulti inherits from PPO (not PPOLag), so we start from PPO defaults
    and add the cost machinery that PPOLagMulti manages itself.
    """
    import omnisafe
    pkg_dir = os.path.dirname(os.path.abspath(omnisafe.__file__))
    default_path = os.path.join(pkg_dir, 'configs', 'on-policy', 'PPO.yaml')
    with open(default_path) as f:
        raw = yaml.safe_load(f)
    defaults = raw.get('defaults', raw)
    # PPOLagMulti needs cost tracking enabled (PPO defaults to False)
    defaults.setdefault('algo_cfgs', {})['use_cost'] = True
    # Provide lagrange_cfgs defaults (PPOLagMulti creates self._lagrange for
    # OmniSafe logging compatibility; user config can override these)
    defaults.setdefault('lagrange_cfgs', {
        'cost_limit': 25.0,
        'lagrangian_multiplier_init': 0.001,
        'lambda_lr': 0.035,
        'lambda_optimizer': 'Adam',
    })
    return defaults


def deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def main(cfg_path: str, use_bc: bool = False, no_curriculum: bool = False) -> None:
    # 1. Load PPO defaults (PPOLagMulti inherits from PPO, not PPOLag)
    defaults = load_ppo_defaults()

    # 2. Load our custom config
    with open(cfg_path) as f:
        custom = yaml.safe_load(f)

    algo = custom.pop('algo', 'PPOLagMulti')
    env_id = custom.pop('env_id', 'CityLearnSafety-V2G-v2')
    seed = custom.pop('seed', 42)

    # 3. Merge: defaults + custom overrides
    merged = deep_update(defaults, custom)
    merged['seed'] = seed
    merged['env_id'] = env_id
    merged['algo'] = algo
    merged['exp_name'] = f'{algo}-{{{env_id}}}'

    # Compute epochs from total_steps
    total_steps = merged['train_cfgs']['total_steps']
    steps_per_epoch = merged['algo_cfgs']['steps_per_epoch']
    merged['train_cfgs']['epochs'] = total_steps // steps_per_epoch

    # Ensure device is set
    if 'device' not in merged['train_cfgs']:
        merged['train_cfgs']['device'] = 'cpu'

    # 4. Convert to OmniSafe Config object
    cfgs = Config(**merged)

    print(f"{'=' * 50}")
    print(f"  {algo} Training")
    print(f"  Algo: {algo}")
    print(f"  Env: {env_id}")
    print(f"  Seed: {seed}")
    print(f"  Epochs: {cfgs.train_cfgs.epochs}")
    print(f"  Steps/epoch: {cfgs.algo_cfgs.steps_per_epoch}")
    if hasattr(cfgs, 'multi_cfgs'):
        print(f"  Softmax tau: {cfgs.multi_cfgs.tau}")
        print(f"  Per-constraint cost limits:")
        for i in range(5):
            lim = float(getattr(cfgs.multi_cfgs, f'cost_limit_{i}', 5000.0))
            print(f"    C{i}: {lim:.0f}")
    if hasattr(cfgs, 'grads_cfgs'):
        print(f"  GradS: sim={cfgs.grads_cfgs.sim_threshold}, "
              f"conflict={cfgs.grads_cfgs.conflict_threshold}, "
              f"sampling={cfgs.grads_cfgs.sampling}")
        print(f"  Per-constraint cost limits:")
        for i in range(5):
            lim = float(getattr(cfgs.grads_cfgs, f'cost_limit_{i}', 5000.0))
            print(f"    C{i}: {lim:.0f}")
    print(f"  Lambda LR: {cfgs.lagrange_cfgs.lambda_lr}")
    print(f"  Lambda init: {cfgs.lagrange_cfgs.lagrangian_multiplier_init}")
    ub = cfgs.lagrange_cfgs.lagrangian_upper_bound
    print(f"  Lambda upper bound: {'None (uncapped)' if ub is None else ub}")
    if no_curriculum and hasattr(cfgs, 'multi_cfgs'):
        # Override: C0 active from epoch 0 (no annealing)
        cfgs.multi_cfgs.cost_limit_0 = 1800  # immediate C0 enforcement
        cfgs.multi_cfgs.anneal_cost_limit_0 = [1800, 1800, 0, 0]  # no annealing
        print(f"  [NO-CURRICULUM] C0 limit set to 1800 from epoch 0")
    print(f"{'=' * 50}")

    # 5. Direct instantiation (bypasses omnisafe.Agent)
    algo_classes = {
        'PPOLagMulti': PPOLagMulti,
        'PPOLagGradS': PPOLagGradS,
    }
    algo_cls = algo_classes.get(algo)
    if algo_cls is None:
        raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(algo_classes.keys())}")
    agent = algo_cls(env_id=env_id, cfgs=cfgs)

    # 5b. Optional BC warmstart
    if use_bc:
        print("\n[BC] Running behavioral cloning warmstart from SmartV2GRBC...")
        _run_bc_warmstart(agent, cfgs)

    ep_ret, ep_cost, ep_len = agent.learn()

    print(f"\nTraining complete.")
    print(f"  Final EpRet: {ep_ret:.1f}")
    print(f"  Final EpCost: {ep_cost:.1f}")
    print(f"  Final EpLen: {ep_len:.0f}")


def _run_bc_warmstart(agent, cfgs):
    """Collect data from SmartV2GRBC and train actor via MSE."""
    import torch
    from scripts.rbc_policy import SmartV2GRBC

    algo = agent
    ac = algo._actor_critic
    device = algo._device

    # Collect one episode from SmartV2GRBC
    env = algo._env
    obs, info = env.reset()
    # Unwrap through adapter layers to find the CityLearn-compatible env
    inner = env
    for _ in range(20):
        if hasattr(inner, 'buildings') or (hasattr(inner, '_raw') and inner._raw is not None):
            break
        inner = getattr(inner, '_env', getattr(inner, 'env', getattr(inner, 'base', None)))
        if inner is None:
            break
    rbc = SmartV2GRBC(inner if inner is not None else env)

    obs_list, act_list = [], []
    done = False
    steps = 0
    while not done and steps < 8760:
        obs_np = obs.cpu().numpy() if hasattr(obs, 'cpu') else obs
        rbc_action = rbc.predict(obs_np)
        obs_list.append(obs_np.copy())
        act_list.append(rbc_action.copy())
        obs, _, _, terminated, truncated, info = env.step(
            torch.as_tensor(rbc_action, dtype=torch.float32).to(device))
        done = terminated or truncated
        steps += 1

    print(f"[BC] Collected {steps} steps from SmartV2GRBC")

    obs_t = torch.as_tensor(np.array(obs_list), dtype=torch.float32).to(device)
    act_t = torch.as_tensor(np.array(act_list), dtype=torch.float32).to(device)

    # Normalize obs using agent's normalizer
    if hasattr(algo, '_env') and hasattr(algo._env, '_obs_normalizer'):
        normalizer = algo._env._obs_normalizer
        if normalizer is not None:
            for i in range(len(obs_t)):
                obs_t[i] = normalizer.normalize(obs_t[i])

    # Train actor via MSE for 50 epochs
    actor = ac.actor
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    batch_size = 256

    for epoch in range(50):
        perm = torch.randperm(len(obs_t))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(obs_t), batch_size):
            idx = perm[i:i+batch_size]
            pred = actor(obs_t[idx])
            if hasattr(pred, 'mean'):
                pred = pred.mean
            loss = torch.nn.functional.mse_loss(pred, act_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 10 == 0:
            print(f"[BC] Epoch {epoch}: loss={total_loss/n_batches:.4f}")

    print(f"[BC] Warmstart complete.")

    # R30: Save frozen reference policy for KL regularization
    # NOTE: copy.deepcopy crashes on OmniSafe actors due to weight_norm /
    # non-leaf tensors. Instead we build a fresh GaussianLearningActor with
    # the same architecture and load a cloned state_dict.
    kl_beta = float(os.environ.get("CITYLEARN_KL_BETA", "0.0"))
    if kl_beta > 0:
        from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor

        ref_actor = GaussianLearningActor(
            obs_space=actor._obs_space,
            act_space=actor._act_space,
            hidden_sizes=actor._hidden_sizes,
            activation=actor._activation,
            weight_initialization_mode=actor._weight_initialization_mode,
        )
        # Clone state_dict (detach from compute graph)
        ref_state = {k: v.clone().detach() for k, v in actor.state_dict().items()}
        ref_actor.load_state_dict(ref_state)
        for p in ref_actor.parameters():
            p.requires_grad = False
        ref_actor.eval()
        ref_actor.to(device)
        algo._bc_ref_actor = ref_actor
        algo._kl_beta = kl_beta
        print(f"[BC] Saved reference policy for KL regularization (beta={kl_beta})")

    # Reset env for RL training
    env.reset()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='Path to YAML config')
    ap.add_argument('--bc', action='store_true', help='Enable BC warmstart from SmartV2GRBC')
    ap.add_argument('--no-curriculum', action='store_true', help='Disable C0 curriculum annealing')
    args = ap.parse_args()
    main(args.cfg, use_bc=args.bc, no_curriculum=args.no_curriculum)
