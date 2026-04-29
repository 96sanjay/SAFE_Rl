#!/usr/bin/env python3
"""
Train OmniSafe PPOLag with STEMS GCN-Transformer encoder on 5-building CityLearn.

Uses the same monkeypatch pattern as train_structured_brain.py to inject a custom
mean_net (STEMSEncoder5Bld) into OmniSafe's GaussianLearningActor.

Usage:
  # Full run (50 epochs)
  python scripts/train_stems_5bld.py --cfg configs/on-policy/r8_stems_5bld.yaml

  # Smoke test (1 epoch)
  python scripts/train_stems_5bld.py --cfg configs/on-policy/r8_stems_5bld.yaml --smoke_1epoch
"""
from __future__ import annotations

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import omnisafe
import citylearn_safe.cmdp_env  # registers CityLearnSafety-V2G-v2

from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.actor.actor_builder import ActorBuilder

from citylearn_safe.safety_env import CityLearnSafetyEnv
from citylearn_safe.schema_index import build_index, _CACHE
from citylearn_safe.stems_encoder_5bld import STEMSEncoder5Bld, build_node_indices
from citylearn_safe.stems_encoder import STEMSEncoder


# ---------------------------------------------------------------------------
# Build ObsIndex from a throwaway env (no OmniSafe agent needed)
# ---------------------------------------------------------------------------
def build_obs_index_5bld():
    """
    Construct a 5-building CityLearn env once to get ObsIndex.
    Returns (obs_index, obs_dim, act_dim, num_buildings, num_evs).
    """
    from scripts.make_env import make_base_env
    import citylearn_safe.schema_index as si

    # Clear cache so we get fresh index for this schema
    si._CACHE = None

    base_env = make_base_env(central_agent=True)
    safety_env = CityLearnSafetyEnv(base_env)

    # Count buildings from the underlying CityLearn env
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

    # Get obs_dim from the full env (with forecast wrapper)
    # We need to create the same env that OmniSafe will use
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    forecast_env = ForecastObsWrapper(safety_env, forecast_horizon=24)
    obs_space = forecast_env.observation_space
    if isinstance(obs_space, (list, tuple)):
        obs_dim = int(obs_space[0].shape[0])
    else:
        obs_dim = int(obs_space.shape[0])

    act_space = forecast_env.action_space
    if isinstance(act_space, (list, tuple)):
        act_dim = int(act_space[0].shape[0])
    else:
        act_dim = int(act_space.shape[0])

    return obs_index, obs_dim, act_dim, num_buildings, num_evs


# ---------------------------------------------------------------------------
# STEMS Mean Network: encoder + action head (outputs act_dim with tanh)
# ---------------------------------------------------------------------------
class STEMSMeanNet(nn.Module):
    """
    Wraps STEMSEncoder5Bld with a final linear projection to action space.
    Output is tanh-squashed to [-1, 1] matching GaussianLearningActor convention.
    """
    def __init__(self, encoder: STEMSEncoder5Bld, act_dim: int):
        super().__init__()
        self.encoder = encoder
        self.action_head = nn.Sequential(
            nn.Linear(encoder.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)  # [B, output_dim]
        return torch.tanh(self.action_head(features))  # [B, act_dim]


# ---------------------------------------------------------------------------
# Custom GaussianLearningActor (same pattern as train_structured_brain.py)
# ---------------------------------------------------------------------------
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
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to YAML config")
    ap.add_argument("--log_dir", default=None, help="Override log_dir")
    ap.add_argument("--init_std", type=float, default=1.0,
                    help="Initial Gaussian std (1.0 matches OmniSafe MLP default)")
    ap.add_argument("--smoke_1epoch", action="store_true",
                    help="Override to 1 epoch for smoke testing")
    ap.add_argument("--hidden_dim", type=int, default=64,
                    help="STEMS encoder hidden dimension")
    ap.add_argument("--output_dim", type=int, default=256,
                    help="STEMS encoder output dimension")
    ap.add_argument("--temporal_window", type=int, default=24,
                    help="Temporal attention window size")
    ap.add_argument("--num_gcn_layers", type=int, default=3,
                    help="Number of GCN layers")
    args = ap.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.cfg, "r"))
    algo = cfg["algo"]
    env_id = cfg["env_id"]

    allowed = ("train_cfgs", "algo_cfgs", "logger_cfgs", "lagrange_cfgs",
               "model_cfgs", "save_cfgs", "env_cfgs", "reward_model_cfgs")
    custom_cfgs = {k: v for k, v in cfg.items() if k in allowed}

    # Force gaussian_learning actor type
    custom_cfgs.setdefault("model_cfgs", {})
    custom_cfgs["model_cfgs"]["actor_type"] = "gaussian_learning"

    if args.log_dir:
        custom_cfgs.setdefault("logger_cfgs", {})
        custom_cfgs["logger_cfgs"]["log_dir"] = args.log_dir

    if args.smoke_1epoch:
        custom_cfgs.setdefault("train_cfgs", {})
        custom_cfgs.setdefault("algo_cfgs", {})
        custom_cfgs["train_cfgs"]["total_steps"] = 8759
        custom_cfgs["algo_cfgs"]["steps_per_epoch"] = 8759
        custom_cfgs["algo_cfgs"]["update_iters"] = 5

    # Build ObsIndex from 5-building env
    print("\n=== Building ObsIndex from 5-building environment ===")
    obs_index, obs_dim, act_dim, num_buildings, num_evs = build_obs_index_5bld()
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, "
          f"num_buildings={num_buildings}, num_evs={num_evs}")

    # Build node index mapping
    node_info = build_node_indices(obs_index, num_buildings)
    print(f"  base_obs_dim={node_info['base_obs_dim']}")
    print(f"  global features: {len(node_info['global_indices'])} dims")
    print(f"  forecast features: {obs_dim - node_info['base_obs_dim']} dims")
    print(f"  EV chargers per building: "
          f"{[('yes' if ev is not None else 'no') for ev in node_info['ev_indices']]}")

    # Build STEMS encoder + action head
    encoder_version = os.environ.get("STEMS_ENCODER_VERSION", "v2")

    if encoder_version == "v3":
        # V3: per-node temporal Transformer (PPO-compatible)
        history_indices = []
        for i in range(num_buildings):
            history_indices.append(obs_index.electrical_storage_soc[i])
            history_indices.append(obs_index.net_electricity_consumption[i])
        history_indices.append(obs_index.electricity_pricing)
        temporal_window = int(os.environ.get("CITYLEARN_TEMPORAL_WINDOW", "12"))

        # obs_dim from build_obs_index_5bld() is 198 (no temporal wrapper).
        # The actual env adds T * n_features history dims.
        obs_dim_v3 = obs_dim + len(history_indices) * temporal_window
        print(f"  V3 obs_dim: {obs_dim} (base) + {len(history_indices)*temporal_window} (history) = {obs_dim_v3}")

        encoder = STEMSEncoder(
            obs_dim=obs_dim_v3,
            node_info=node_info,
            num_buildings=num_buildings,
            hidden_dim=args.hidden_dim,
            global_hidden=32,
            temporal_window=temporal_window,
            temporal_features_per_step=len(history_indices),
            temporal_hidden=32,
            temporal_heads=4,
            num_gcn_layers=args.num_gcn_layers,
            dropout=0.1,
            output_dim=args.output_dim,
            history_indices=history_indices,
        )
        print(f"  Using STEMSEncoder (temporal T={temporal_window})")
    else:
        encoder = STEMSEncoder5Bld(
            obs_dim=obs_dim,
            node_info=node_info,
            num_buildings=num_buildings,
            hidden_dim=args.hidden_dim,
            global_hidden=32,
            num_gcn_layers=args.num_gcn_layers,
            num_heads=4,
            temporal_window=args.temporal_window,
            output_dim=args.output_dim,
            dropout=0.1,
        )
        print(f"  Using STEMSEncoder5Bld (v2, no temporal)")
    mean_net = STEMSMeanNet(encoder, act_dim)

    total_params = sum(p.numel() for p in mean_net.parameters())
    print(f"  STEMS encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Total mean_net params: {total_params:,}")
    print(f"  encoder output_dim={encoder.output_dim}, action_dim={act_dim}")

    # Monkeypatch ActorBuilder
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

    print("\n=== TRAIN STEMS GCN-TRANSFORMER (5 buildings) ===")
    print(f"Algo: {algo}")
    print(f"Env:  {env_id}")
    print(f"Architecture: STEMS GCN-Transformer")
    print(f"  hidden_dim={args.hidden_dim}, gcn_layers={args.num_gcn_layers}")
    print(f"  temporal_window={args.temporal_window}, output_dim={args.output_dim}")
    print(f"  init_std={args.init_std}")
    print(f"Log dir: {custom_cfgs.get('logger_cfgs', {}).get('log_dir', '(default)')}")
    print("=" * 50 + "\n")

    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)

    # --- Lambda capping (R10c fix) ---
    # OmniSafe's YAML validator rejects unknown keys, so we set it after init
    lambda_upper = float(os.environ.get('LAMBDA_UPPER_BOUND', '0'))
    if lambda_upper > 0 and hasattr(agent.agent, '_lagrange'):
        agent.agent._lagrange.lagrangian_upper_bound = lambda_upper
        print(f">>> LAMBDA CAP: lagrangian_upper_bound = {lambda_upper}")

    # Verify custom architecture is in use
    actor = agent.agent._actor_critic.actor
    print(f">>> VERIFY actor class: {actor.__class__.__name__}")
    print(f">>> VERIFY mean  class: {actor.mean.__class__.__name__}")
    print(f">>> VERIFY policy std:  {actor.std}")

    test_obs = torch.zeros(2, actor._obs_dim)
    test_act = actor.predict(test_obs, deterministic=True)
    print(f">>> VERIFY action shape: {tuple(test_act.shape)} "
          f"min/max: {float(test_act.min()):.4f}/{float(test_act.max()):.4f}")
    print(">>> VERIFY done. Custom STEMS encoder is active.\n")

    agent.learn()


if __name__ == "__main__":
    main()
