# scripts/training/train_multi_lag_stems.py
"""Training script for PPOLagMulti + STEMS encoder (GCN-Transformer).

Combines:
  - train_multi_lag.py: PPOLagMulti direct instantiation with per-constraint λ
  - train_stems_5bld.py: STEMS encoder monkeypatch into ActorBuilder

Usage:
    python scripts/training/train_multi_lag_stems.py --cfg configs/active/headroom_gated_cmdp.yaml
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import yaml
import torch
import torch.nn as nn

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Register environments FIRST (cmdp_env registers CityLearnSafety-V2G-v2)
import citylearn_safe.cmdp_env    # noqa: F401  @env_register side-effect

from omnisafe.utils.config import Config
from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.actor.actor_builder import ActorBuilder

from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.models.base import Critic

from citylearn_safe.grads.ppo_lag_multi import PPOLagMulti
from citylearn_safe.grads.ppo_lag_grads import PPOLagGradS
from citylearn_safe.safety_env import CityLearnSafetyEnv
from citylearn_safe.schema_index import build_index, _CACHE
from citylearn_safe.stems_encoder_5bld import STEMSEncoder5Bld, build_node_indices
from citylearn_safe.stems_encoder import STEMSEncoder


# ---------------------------------------------------------------------------
# From train_stems_5bld.py: Build ObsIndex
# ---------------------------------------------------------------------------
def build_obs_index_5bld():
    from scripts.make_env import make_base_env
    import citylearn_safe.schema_index as si
    si._CACHE = None

    base_env = make_base_env(central_agent=True)
    safety_env = CityLearnSafetyEnv(base_env)

    city = base_env
    for _ in range(20):
        if hasattr(city, 'buildings') and len(getattr(city, 'buildings', [])) > 0:
            break
        city = getattr(city, 'env', getattr(city, 'base', getattr(city, 'unwrapped', None)))
        if city is None:
            break
    num_buildings = len(city.buildings) if city and hasattr(city, 'buildings') else 5

    obs_index = build_index(safety_env, expected_buildings=num_buildings)
    num_evs = len(obs_index.ev)

    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    forecast_env = ForecastObsWrapper(safety_env, forecast_horizon=24)
    obs_space = forecast_env.observation_space
    obs_dim = int(obs_space[0].shape[0]) if isinstance(obs_space, (list, tuple)) else int(obs_space.shape[0])

    act_space = forecast_env.action_space
    act_dim = int(act_space[0].shape[0]) if isinstance(act_space, (list, tuple)) else int(act_space.shape[0])

    return obs_index, obs_dim, act_dim, num_buildings, num_evs


# ---------------------------------------------------------------------------
# From train_stems_5bld.py: STEMS Mean Network + Custom Actor
# ---------------------------------------------------------------------------
class STEMSMeanNet(nn.Module):
    def __init__(self, encoder, act_dim: int, encoder_obs_dim: int):
        super().__init__()
        self.encoder = encoder
        self.encoder_obs_dim = encoder_obs_dim  # expected obs dim without Sauté
        self.action_head = nn.Sequential(
            nn.Linear(encoder.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Strip extra dims (e.g., Sauté budget) beyond what encoder expects
        enc_obs = obs[:, :self.encoder_obs_dim] if obs.size(-1) > self.encoder_obs_dim else obs
        features = self.encoder(enc_obs)
        return torch.tanh(self.action_head(features))


class CustomGaussianLearningActor(GaussianLearningActor):
    def __init__(self, obs_space, act_space, hidden_sizes, mean_net: nn.Module,
                 init_std: float, activation='relu',
                 weight_initialization_mode='kaiming_uniform'):
        super().__init__(obs_space, act_space, hidden_sizes, activation,
                         weight_initialization_mode)
        self.mean = mean_net
        with torch.no_grad():
            self.log_std.fill_(torch.log(torch.tensor(init_std, device=self.log_std.device)))


# ---------------------------------------------------------------------------
# STEMS Critic: structural temporal-spatial encoder for value estimation
# ---------------------------------------------------------------------------
class STEMSValueNet(nn.Module):
    """STEMS encoder + value head for V(s) estimation.

    Gives the critic the same temporal-spatial inductive bias as the actor,
    so advantage estimates for temporally-contingent actions are accurate.
    """

    def __init__(self, encoder: nn.Module, encoder_obs_dim: int):
        super().__init__()
        self.encoder = encoder
        self.encoder_obs_dim = encoder_obs_dim
        self.value_head = nn.Sequential(
            nn.Linear(encoder.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        enc_obs = obs[:, :self.encoder_obs_dim] if obs.size(-1) > self.encoder_obs_dim else obs
        features = self.encoder(enc_obs)
        return self.value_head(features)  # [B, 1]


class STEMSVCritic(Critic):
    """VCritic replacement that uses STEMS encoder instead of plain MLP.

    Conforms to OmniSafe's VCritic interface (net_lst, forward → list[Tensor]).
    """

    def __init__(self, obs_space, act_space, hidden_sizes, stems_value_net,
                 activation='relu', weight_initialization_mode='kaiming_uniform'):
        super().__init__(obs_space, act_space, hidden_sizes, activation,
                         weight_initialization_mode, num_critics=1,
                         use_obs_encoder=False)
        self.net_lst = [stems_value_net]
        self.add_module('critic_0', stems_value_net)

    def forward(self, obs: torch.Tensor) -> list:
        return [torch.squeeze(self.net_lst[0](obs), -1)]


# ---------------------------------------------------------------------------
# From train_multi_lag.py: Config loading
# ---------------------------------------------------------------------------
def load_ppolag_defaults() -> dict:
    import omnisafe
    pkg_dir = os.path.dirname(os.path.abspath(omnisafe.__file__))
    default_path = os.path.join(pkg_dir, 'configs', 'on-policy', 'PPOLag.yaml')
    with open(default_path) as f:
        raw = yaml.safe_load(f)
    return raw.get('defaults', raw)


def deep_update(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# R26g: Partial obs normalization — only normalize base obs, not temporal history
# ---------------------------------------------------------------------------
def _apply_partial_obs_norm(agent, base_obs_dim: int):
    """Patch the agent's obs normalizer to only normalize the first base_obs_dim dims.

    OmniSafe's ObsNormalize applies per-dim running mean/std to ALL obs dims.
    For STEMS temporal history, this corrupts inter-timestep patterns because
    the same physical feature at different timesteps gets different statistics.

    Fix: zero out the running variance for temporal dims (std→1, mean→0),
    so normalization is identity for those dims.
    """
    # Access the normalizer through the env wrapper chain
    env = agent._env
    for _ in range(20):
        if hasattr(env, '_obs_normalizer'):
            normalizer = env._obs_normalizer
            # Freeze temporal dims: set mean=0, var=1, std=1 and prevent updates
            with torch.no_grad():
                normalizer._mean[base_obs_dim:] = 0.0
                normalizer._var[base_obs_dim:] = 1.0
                normalizer._std[base_obs_dim:] = 1.0
                normalizer._sumsq[base_obs_dim:] = 1.0

            # Monkeypatch _push to only update base dims (temporal dims stay identity)
            original_push = normalizer._push

            def make_partial_push(orig_push, bdim, norm):
                def partial_push(raw_data):
                    # Save temporal stats (identity normalization)
                    saved_mean = norm._mean[bdim:].clone()
                    saved_var = norm._var[bdim:].clone()
                    saved_std = norm._std[bdim:].clone()
                    saved_sumsq = norm._sumsq[bdim:].clone()
                    # Run normal update (updates ALL dims)
                    orig_push(raw_data)
                    # Restore temporal stats to identity
                    with torch.no_grad():
                        norm._mean[bdim:] = saved_mean
                        norm._var[bdim:] = saved_var
                        norm._std[bdim:] = saved_std
                        norm._sumsq[bdim:] = saved_sumsq
                return partial_push

            normalizer._push = make_partial_push(original_push, base_obs_dim, normalizer)

            print(f"  [PartialObsNorm] Normalizing dims 0-{base_obs_dim-1}, "
                  f"identity for dims {base_obs_dim}+")
            return True

        env = getattr(env, 'env', getattr(env, '_env', getattr(env, 'unwrapped', None)))
        if env is None:
            break

    print("  [PartialObsNorm] WARNING: Could not find obs normalizer in env chain")
    return False


# ---------------------------------------------------------------------------
# Main: Combined PPOLagMulti + STEMS
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to YAML config")
    ap.add_argument("--init_std", type=float, default=1.0)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--output_dim", type=int, default=256)
    ap.add_argument("--num_gcn_layers", type=int, default=3)
    ap.add_argument("--stems_critic", action="store_true",
                    help="R26g: Use STEMS encoder for critics too (not just actor)")
    ap.add_argument("--temporal_layers", type=int, default=2,
                    help="R26g: Number of transformer layers (default: 2)")
    ap.add_argument("--temporal_pool", type=str, default="mean",
                    choices=["mean", "last", "cls"],
                    help="R26g: Temporal pooling mode (default: mean)")
    ap.add_argument("--partial_obs_norm", action="store_true",
                    help="R26g: Only normalize base obs dims, leave temporal history raw")
    args = ap.parse_args()

    # --- 1. Build ObsIndex ---
    print("\n=== Building ObsIndex from 5-building environment ===")
    obs_index, obs_dim, act_dim, num_buildings, num_evs = build_obs_index_5bld()
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, "
          f"num_buildings={num_buildings}, num_evs={num_evs}")

    node_info = build_node_indices(obs_index, num_buildings)
    print(f"  base_obs_dim={node_info['base_obs_dim']}")

    # --- 2. Build STEMS Encoder ---
    encoder_version = os.environ.get("STEMS_ENCODER_VERSION", "v3")
    temporal_window = int(os.environ.get("CITYLEARN_TEMPORAL_WINDOW", "12"))
    use_rich_temporal = os.environ.get("CITYLEARN_TEMPORAL_RICH", "0") == "1"

    if encoder_version == "v3":
        if use_rich_temporal:
            # Rich temporal: solar, EV SoC/departure, hour encoding (10 features/node)
            from citylearn_safe.temporal_obs_wrapper import build_rich_history_config
            rich_cfg = build_rich_history_config(obs_index, num_buildings)
            history_indices = rich_cfg['history_indices']
            features_per_step = rich_cfg['features_per_step']
            features_per_node = rich_cfg['features_per_node']
            per_node_map = rich_cfg['per_node_map']
            ev_mask = rich_cfg['history_ev_mask']
            print(f"  Rich temporal: {features_per_step} features/step, "
                  f"{features_per_node} features/node")
        else:
            # Basic temporal: battery_soc + net_consumption + price (3 features/node)
            # MUST use contiguous blocks (not interleaved) to match encoder's
            # _build_default_history_mapping: [soc_0..N-1, net_0..N-1, price]
            from citylearn_safe.temporal_obs_wrapper import build_basic_history_indices
            history_indices = build_basic_history_indices(obs_index, num_buildings)
            features_per_step = len(history_indices)
            features_per_node = 3
            per_node_map = None
            ev_mask = None

        obs_dim_v3 = obs_dim + features_per_step * temporal_window
        print(f"  V3 obs_dim: {obs_dim} + {features_per_step}*{temporal_window} = {obs_dim_v3}")

        stems_kwargs = dict(
            obs_dim=obs_dim_v3,
            node_info=node_info,
            num_buildings=num_buildings,
            hidden_dim=args.hidden_dim,
            global_hidden=32,
            temporal_window=temporal_window,
            temporal_features_per_step=features_per_step,
            temporal_hidden=32,
            temporal_heads=4,
            num_gcn_layers=args.num_gcn_layers,
            dropout=0.1,
            output_dim=args.output_dim,
            temporal_features_per_node=features_per_node,
            per_node_history_map=per_node_map,
            history_ev_mask=ev_mask,
            temporal_num_layers=args.temporal_layers,
            temporal_pool_mode=args.temporal_pool,
        )
        encoder = STEMSEncoder(**stems_kwargs)
        print(f"  Using STEMSEncoder (temporal T={temporal_window}, "
              f"{'rich' if use_rich_temporal else 'basic'}, "
              f"layers={args.temporal_layers}, pool={args.temporal_pool})")

        # R26g: Build SEPARATE encoder for critics (same architecture, independent weights)
        critic_encoder = None
        if args.stems_critic:
            critic_encoder = STEMSEncoder(**stems_kwargs)
            print(f"  STEMS critic encoder: {sum(p.numel() for p in critic_encoder.parameters()):,} params")
    else:
        obs_dim_v3 = obs_dim  # no temporal history for v2
        encoder = STEMSEncoder5Bld(
            obs_dim=obs_dim,
            node_info=node_info,
            num_buildings=num_buildings,
            hidden_dim=args.hidden_dim,
            global_hidden=32,
            num_gcn_layers=args.num_gcn_layers,
            num_heads=4,
            temporal_window=24,
            output_dim=args.output_dim,
            dropout=0.1,
        )
        print(f"  Using STEMSEncoder5Bld (v2, spatial only)")

    mean_net = STEMSMeanNet(encoder, act_dim, encoder_obs_dim=obs_dim_v3)
    total_params = sum(p.numel() for p in mean_net.parameters())
    print(f"  STEMS encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Total mean_net params: {total_params:,}")

    # --- 3. Monkeypatch ActorBuilder ---
    original_build = ActorBuilder.build_actor

    def patched_build_actor(self, actor_type):
        if actor_type == "gaussian_learning":
            return CustomGaussianLearningActor(
                self._obs_space, self._act_space, self._hidden_sizes,
                mean_net=mean_net,
                init_std=args.init_std,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        return original_build(self, actor_type)

    ActorBuilder.build_actor = patched_build_actor

    # --- 3b. Monkeypatch CriticBuilder for STEMS critics ---
    # Only the reward critic (first build_critic('v') call) uses STEMS encoder.
    # Cost critics use default MLP — they predict instantaneous constraint violations
    # that don't benefit from temporal reasoning, and STEMS would be 5x slower.
    if critic_encoder is not None:
        original_build_critic = CriticBuilder.build_critic
        _stems_critic_count = [0]  # mutable counter for closure

        def patched_build_critic(self, critic_type):
            if critic_type == 'v' and _stems_critic_count[0] == 0:
                _stems_critic_count[0] += 1
                value_net = STEMSValueNet(critic_encoder, encoder_obs_dim=obs_dim_v3)
                print(f"  CriticBuilder: STEMS encoder for reward critic (call #{_stems_critic_count[0]})")
                return STEMSVCritic(
                    self._obs_space, self._act_space, self._hidden_sizes,
                    stems_value_net=value_net,
                    activation=self._activation,
                    weight_initialization_mode=self._weight_initialization_mode,
                )
            _stems_critic_count[0] += 1
            print(f"  CriticBuilder: MLP for cost critic (call #{_stems_critic_count[0]})")
            return original_build_critic(self, critic_type)

        CriticBuilder.build_critic = patched_build_critic
        print("  CriticBuilder monkeypatched: STEMS reward critic + MLP cost critics")

    # --- 4. Load config (same as train_multi_lag.py) ---
    with open(args.cfg) as f:
        custom = yaml.safe_load(f)

    algo = custom.pop('algo', 'PPOLagMulti')
    env_id = custom.pop('env_id', 'CityLearnSafety-V2G-v2')
    seed = custom.pop('seed', 42)

    defaults = load_ppolag_defaults()
    merged = deep_update(defaults, custom)
    merged['seed'] = seed
    merged['env_id'] = env_id
    merged['algo'] = algo
    merged['exp_name'] = f'{algo}-{{{env_id}}}'

    total_steps = merged['train_cfgs']['total_steps']
    steps_per_epoch = merged['algo_cfgs']['steps_per_epoch']
    merged['train_cfgs']['epochs'] = total_steps // steps_per_epoch

    if 'device' not in merged['train_cfgs']:
        merged['train_cfgs']['device'] = 'cpu'

    # Force gaussian_learning actor type
    merged.setdefault("model_cfgs", {})
    merged["model_cfgs"]["actor_type"] = "gaussian_learning"

    cfgs = Config(**merged)

    # --- 5. Print summary ---
    print(f"\n{'=' * 50}")
    print(f"  {algo} + STEMS Training")
    print(f"  Algo: {algo}")
    print(f"  Env: {env_id}")
    print(f"  Seed: {seed}")
    print(f"  Device: {cfgs.train_cfgs.device}")
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
    ub = cfgs.lagrange_cfgs.lagrangian_upper_bound
    print(f"  Lambda upper bound: {'None' if ub is None else ub}")
    print(f"  STEMS: {encoder_version}, T={temporal_window}")
    print(f"  STEMS critic: {'enabled' if args.stems_critic else 'disabled (MLP)'}")
    print(f"  Temporal: layers={args.temporal_layers}, pool={args.temporal_pool}")
    print(f"  Partial obs norm: {args.partial_obs_norm}")
    print(f"{'=' * 50}")

    # --- 6. Instantiate and train ---
    algo_classes = {
        'PPOLagMulti': PPOLagMulti,
        'PPOLagGradS': PPOLagGradS,
    }
    algo_cls = algo_classes.get(algo)
    if algo_cls is None:
        raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(algo_classes.keys())}")
    agent = algo_cls(env_id=env_id, cfgs=cfgs)

    # --- 6b. R26g: Partial obs normalization (only base dims, not temporal) ---
    if args.partial_obs_norm and encoder_version == "v3":
        _apply_partial_obs_norm(agent, base_obs_dim=obs_dim)
        print(f">>> Partial obs norm applied: normalizing first {obs_dim} dims only")

    # Verify STEMS is active
    actor = agent._actor_critic.actor
    print(f"\n>>> VERIFY actor class: {actor.__class__.__name__}")
    print(f">>> VERIFY mean  class: {actor.mean.__class__.__name__}")
    print(f">>> VERIFY policy std:  {actor.std}")

    # Verify critic architecture
    reward_critic = agent._actor_critic.reward_critic
    print(f">>> VERIFY reward critic class: {reward_critic.__class__.__name__}")
    if hasattr(reward_critic, 'net_lst') and len(reward_critic.net_lst) > 0:
        net0 = reward_critic.net_lst[0]
        print(f">>> VERIFY critic net class: {net0.__class__.__name__}")
        if hasattr(net0, 'encoder'):
            print(f">>> VERIFY critic encoder class: {net0.encoder.__class__.__name__}")

    # Verify with correct device and obs_dim (actor._obs_dim includes Sauté +1)
    device = torch.device(cfgs.train_cfgs.device)
    actual_obs_dim = actor._obs_dim  # includes all wrappers (temporal + sauté)
    test_obs = torch.zeros(2, actual_obs_dim, device=device)
    test_act = actor.predict(test_obs, deterministic=True)
    print(f">>> VERIFY obs_dim={actual_obs_dim}, action shape: {tuple(test_act.shape)} "
          f"min/max: {float(test_act.min()):.4f}/{float(test_act.max()):.4f}")

    # Verify critic forward pass
    test_v = reward_critic(test_obs)
    print(f">>> VERIFY critic output: {[v.shape for v in test_v]}")
    print(">>> STEMS encoder + critic verified.\n")

    ep_ret, ep_cost, ep_len = agent.learn()

    print(f"\nTraining complete.")
    print(f"  Final EpRet: {ep_ret:.1f}")
    print(f"  Final EpCost: {ep_cost:.1f}")
    print(f"  Final EpLen: {ep_len:.0f}")


if __name__ == "__main__":
    main()
