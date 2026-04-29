# scripts/train_td3_sp_rl.py
"""Training script for TD3LagMulti (SP-RL: Safety Projection for RL).

TD3 with per-constraint Lagrange multipliers (C0, C1 via PID) and DiffProjector
(C2/C3/C4 hard constraints via differentiable QP projection).

Bypasses omnisafe.Agent (which needs a default config YAML per algorithm)
and directly instantiates TD3LagMulti with the merged config.

Usage:
    python scripts/train_td3_sp_rl.py --cfg configs/off-policy/td3_sp_rl_1bld.yaml
"""
from __future__ import annotations

import argparse
import copy
import os

import yaml

# Register environments FIRST
import citylearn_safe.omni_env       # noqa: F401
import citylearn_safe.omni_env_v2    # noqa: F401

from omnisafe.utils.config import Config

# Import TD3LagMulti (triggers @registry.register)
from citylearn_safe.grads.td3_lag_multi import TD3LagMulti


def load_td3_defaults() -> dict:
    """Load TD3 default config from OmniSafe's installed configs.

    We load TD3 (not TD3Lag) because TD3LagMulti handles Lagrangian itself.
    """
    import omnisafe
    pkg_dir = os.path.dirname(os.path.abspath(omnisafe.__file__))
    default_path = os.path.join(pkg_dir, 'configs', 'off-policy', 'TD3.yaml')
    with open(default_path) as f:
        raw = yaml.safe_load(f)
    return raw.get('defaults', raw)


def deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def main(cfg_path: str) -> None:
    # 1. Load TD3 defaults
    defaults = load_td3_defaults()

    # 2. Load our custom config
    with open(cfg_path) as f:
        custom = yaml.safe_load(f)

    algo = custom.pop('algo', 'TD3LagMulti')
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

    # Extract projector config for display
    proj_cfgs = getattr(cfgs, 'projector_cfgs', None)
    penalty_w = float(getattr(proj_cfgs, 'penalty_w', 1.0)) if proj_cfgs else 1.0
    solver_eps = float(getattr(proj_cfgs, 'solver_eps', 1e-4)) if proj_cfgs else 1e-4
    solver_max_iters = int(getattr(proj_cfgs, 'solver_max_iters', 5000)) if proj_cfgs else 5000
    pen_critic_lr = float(getattr(proj_cfgs, 'penalty_critic_lr', 3e-4)) if proj_cfgs else 3e-4

    print(f"{'=' * 60}")
    print(f"  TD3LagMulti Training (SP-RL: Safety Projection for RL)")
    print(f"{'=' * 60}")
    print(f"  Algo: {algo}")
    print(f"  Env: {env_id}")
    print(f"  Seed: {seed}")
    print(f"  Epochs: {cfgs.train_cfgs.epochs}")
    print(f"  Steps/epoch: {cfgs.algo_cfgs.steps_per_epoch}")
    print(f"  Replay buffer: {cfgs.algo_cfgs.size}")
    print(f"  Batch size: {cfgs.algo_cfgs.batch_size}")
    print(f"  Start learning: {cfgs.algo_cfgs.start_learning_steps}")
    print(f"  Policy delay: {cfgs.algo_cfgs.policy_delay}")
    print(f"  Exploration noise: {cfgs.algo_cfgs.exploration_noise}")
    print(f"  Warmup epochs: {cfgs.algo_cfgs.warmup_epochs}")
    print(f"  --- Projection ---")
    print(f"  Penalty weight (w): {penalty_w}")
    print(f"  Solver eps: {solver_eps}")
    print(f"  Solver max iters: {solver_max_iters}")
    print(f"  Penalty critic LR: {pen_critic_lr}")
    if hasattr(cfgs, 'multi_cfgs'):
        print(f"  --- Lagrange (PID) ---")
        print(f"  Softmax tau: {cfgs.multi_cfgs.tau}")
        print(f"  Per-constraint cost limits:")
        for i in range(2):
            lim = float(getattr(cfgs.multi_cfgs, f'cost_limit_{i}', 5000.0))
            kp = float(getattr(cfgs.multi_cfgs, f'pid_kp_{i}',
                        getattr(cfgs.multi_cfgs, 'pid_kp', 0.1)))
            ki = float(getattr(cfgs.multi_cfgs, f'pid_ki_{i}',
                        getattr(cfgs.multi_cfgs, 'pid_ki', 0.01)))
            print(f"    C{i}: limit={lim:.0f}, Kp={kp}, Ki={ki}")
    print(f"  Lambda init: {cfgs.lagrange_cfgs.lagrangian_multiplier_init}")
    ub = getattr(cfgs.lagrange_cfgs, 'lagrangian_upper_bound', None)
    print(f"  Lambda upper bound: {'None (uncapped)' if ub is None else ub}")
    print(f"{'=' * 60}")

    # 5. Direct instantiation (bypasses omnisafe.Agent)
    agent = TD3LagMulti(env_id=env_id, cfgs=cfgs)

    # 5.5 Assert no double safety: SP-RL handles C2/C3/C4 via DiffProjector,
    # so env-side projection/masking must be disabled to avoid conflicts.
    for var_name, label in [
        ("CITYLEARN_SERL_PROJECTION", "SE-RL projection"),
        ("CITYLEARN_ACTION_MASK", "action mask"),
        ("CITYLEARN_BETA_ACTOR", "beta actor"),
    ]:
        if os.environ.get(var_name, "0") == "1":
            raise RuntimeError(
                f"SP-RL cannot run with env-side projection/masking enabled. "
                f"{var_name}=1 ({label}) conflicts with DiffProjector. "
                f"Unset {var_name} or set it to '0'."
            )

    # 6. Build hard shield (mandatory for SP-RL)
    controller_kind = os.environ.get("CITYLEARN_CONTROLLER_KIND", "").strip().lower()
    if not controller_kind:
        controller_kind = "hybrid" if os.environ.get("CITYLEARN_HYBRID_PROJECTOR", "1") == "1" else "diff"

    if controller_kind == "lex":
        from citylearn_safe.lexicographic_safety_controller import LexicographicSafetyController
        projector = LexicographicSafetyController(env=agent._env, verbose=0)
        projector_name = "LexicographicSafetyController"
    elif controller_kind == "hybrid":
        from citylearn_safe.hybrid_projector import HybridProjector
        projector = HybridProjector(
            env=agent._env,
            solver_eps=solver_eps,
            solver_max_iters=solver_max_iters,
        )
        projector_name = "HybridProjector"
    elif controller_kind == "diff":
        from citylearn_safe.diff_projector import DiffProjector
        projector = DiffProjector(
            env=agent._env,
            solver_eps=solver_eps,
            solver_max_iters=solver_max_iters,
        )
        projector_name = "DiffProjector"
    else:
        raise ValueError(
            f"Unknown CITYLEARN_CONTROLLER_KIND={controller_kind!r}. "
            "Expected one of: lex, hybrid, diff."
        )

    # Pass the adapter (which wraps the CMDP/CityLearn env chain);
    # DiffProjector._unwrap_to_citylearn walks ._env/.env/.unwrapped to
    # find the CityLearn env with .buildings and .time_step.
    projector.build()
    agent.attach_projector(projector)
    if hasattr(agent._env, "attach_execution_projector"):
        agent._env.attach_execution_projector(projector)
    print(f"[train_td3_sp_rl] {projector_name} built and attached successfully")
    print(f"[train_td3_sp_rl] Solver: eps={solver_eps}, max_iters={solver_max_iters}")

    # 7. Train
    ep_ret, ep_cost, ep_len = agent.learn()

    print(f"\nTraining complete.")
    print(f"  Final EpRet: {ep_ret:.1f}")
    print(f"  Final EpCost: {ep_cost:.1f}")
    print(f"  Final EpLen: {ep_len:.0f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Train TD3LagMulti (SP-RL) on CityLearn V2G'
    )
    ap.add_argument('--cfg', required=True, help='Path to YAML config')
    args = ap.parse_args()
    main(args.cfg)
