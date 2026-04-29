#!/usr/bin/env python3
"""
Train OmniSafe algorithms (e.g., PPOLag) with a custom "brain" (policy mean network)
WITHOUT modifying OmniSafe source code.

Key ideas:
- Keep model_cfgs.actor_type = "gaussian_learning" so OmniSafe logger keys (Train/PolicyStd) register.
- Monkeypatch ActorBuilder.build_actor at runtime in THIS script only.
- Use CityLearnSafetyEnvV3._obs_index (ObsIndex) so observation slicing is exact (no guessing).

Supports:
- Approach 1: Structured encoders + separate heads (battery/EV/washer)
- Approach 2: Approach 1 + ONE self-attention layer over device tokens (26 tokens)

Usage examples:
  # Approach 1 (default)
  python scripts/train_structured_brain.py --cfg configs/on-policy/ppo_lag_3constraints_lambda35_evweight3_100ep.yaml --log_dir runs/ppolag_structA1

  # Approach 2 (attention)
  python scripts/train_structured_brain.py --cfg configs/on-policy/ppo_lag_3constraints_lambda35_evweight3_100ep.yaml --attention --log_dir runs/ppolag_structA2

  # Smoke (1 epoch) faster updates
  python scripts/train_structured_brain.py --cfg configs/on-policy/ppo_lag_3constraints_lambda35_evweight3_100ep.yaml --smoke_1epoch
"""

from __future__ import annotations

import argparse
import yaml
import torch
import torch.nn as nn

import omnisafe
import citylearn_safe.omni_env  # registers env with OmniSafe
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.actor.actor_builder import ActorBuilder


# ----------------------------
# Utilities
# ----------------------------
def build_obs_index_from_local_env() -> object:
    """
    Build CityLearnSafetyEnvV3 once (WITHOUT OmniSafe Agent) to get ObsIndex.
    This avoids creating extra OmniSafe run directories just for probing.
    """
    # Your repo already has this helper (per your structure/docs).
    from scripts.make_env import make_base_env  # noqa

    base_env = make_base_env(central_agent=True)
    safety_env = CityLearnSafetyEnvV3(base_env)
    return safety_env._obs_index


def obsindex_to_groups(obs_index):
    """Return index tensors derived from ObsIndex (no guessing)."""
    time_idx = [
        obs_index.month_cos, obs_index.month_sin,
        obs_index.day_type_cos, obs_index.day_type_sin,
        obs_index.hour_cos, obs_index.hour_sin,
    ]
    # Weather/irradiance per ObsIndex.names ordering: indices 6..21 inclusive
    weather_idx = list(range(6, 22))

    price_idx = [
        obs_index.electricity_pricing,
        obs_index.electricity_pricing_predicted_1,
        obs_index.electricity_pricing_predicted_2,
        obs_index.electricity_pricing_predicted_3,
    ]
    carbon_idx = [obs_index.carbon_intensity]
    washer_idx = [obs_index.washing_machine_1_start_time_step, obs_index.washing_machine_1_end_time_step]

    b_nonshift = list(obs_index.non_shiftable_load)
    b_solar    = list(obs_index.solar_generation)
    b_soc      = list(obs_index.electrical_storage_soc)
    b_net      = list(obs_index.net_electricity_consumption)

    ev_dict = dict(obs_index.ev)
    ev_keys = sorted(ev_dict.keys())  # stable order
    ev_feature_names = [
        "connected_state","departure_time","required_soc_departure",
        "soc","battery_capacity","incoming_state","estimated_arrival_time",
    ]
    ev_indices = [[ev_dict[k][f] for f in ev_feature_names] for k in ev_keys]

    return dict(
        time_idx=time_idx,
        weather_idx=weather_idx,
        price_idx=price_idx,
        carbon_idx=carbon_idx,
        washer_idx=washer_idx,
        b_nonshift=b_nonshift,
        b_solar=b_solar,
        b_soc=b_soc,
        b_net=b_net,
        ev_indices=ev_indices,
        ev_keys=ev_keys,
        ev_feature_names=ev_feature_names,
    )


# ----------------------------
# Approach 1 Mean Network (Structured)
# ----------------------------
class StructuredMeanObsIndex(nn.Module):
    def __init__(self, groups, obs_dim=153, act_dim=26, d_dev=16, d_global=32, d_shared=64):
        super().__init__()
        assert obs_dim == 153
        assert act_dim == 26

        # store buffers for fast gather
        self.register_buffer("time_idx",   torch.tensor(groups["time_idx"], dtype=torch.long))
        self.register_buffer("weather_idx",torch.tensor(groups["weather_idx"], dtype=torch.long))
        self.register_buffer("price_idx",  torch.tensor(groups["price_idx"], dtype=torch.long))
        self.register_buffer("carbon_idx", torch.tensor(groups["carbon_idx"], dtype=torch.long))
        self.register_buffer("washer_idx", torch.tensor(groups["washer_idx"], dtype=torch.long))

        self.register_buffer("b_nonshift_idx", torch.tensor(groups["b_nonshift"], dtype=torch.long))
        self.register_buffer("b_solar_idx",    torch.tensor(groups["b_solar"], dtype=torch.long))
        self.register_buffer("b_soc_idx",      torch.tensor(groups["b_soc"], dtype=torch.long))
        self.register_buffer("b_net_idx",      torch.tensor(groups["b_net"], dtype=torch.long))

        self.register_buffer("ev_idx", torch.tensor(groups["ev_indices"], dtype=torch.long))  # [8,7]

        global_in = (
            len(groups["time_idx"]) + len(groups["weather_idx"]) +
            len(groups["price_idx"]) + len(groups["carbon_idx"]) + len(groups["washer_idx"])
        )

        # Encoders
        self.global_enc   = nn.Sequential(nn.Linear(global_in, 64), nn.ReLU(), nn.Linear(64, d_global), nn.ReLU())
        self.building_enc = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, d_dev), nn.ReLU())
        self.ev_enc       = nn.Sequential(nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, d_dev), nn.ReLU())
        self.washer_enc   = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, d_dev), nn.ReLU())

        trunk_in = d_global + (17*d_dev) + (8*d_dev) + d_dev
        self.trunk = nn.Sequential(nn.Linear(trunk_in, d_shared), nn.ReLU(), nn.Linear(d_shared, d_shared), nn.ReLU())

        # Heads
        self.batt_head = nn.Sequential(nn.Linear(d_shared + d_dev, 32), nn.ReLU(), nn.Linear(32, 1))
        self.ev_head   = nn.Sequential(nn.Linear(d_shared + d_dev, 32), nn.ReLU(), nn.Linear(32, 1))
        self.wash_head = nn.Sequential(nn.Linear(d_shared + d_dev, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]

        g = torch.cat([
            obs.index_select(1, self.time_idx),
            obs.index_select(1, self.weather_idx),
            obs.index_select(1, self.carbon_idx),
            obs.index_select(1, self.price_idx),
            obs.index_select(1, self.washer_idx),
        ], dim=-1)
        g_emb = self.global_enc(g)

        # buildings [B,17,4]
        b0 = obs.index_select(1, self.b_nonshift_idx)
        b1 = obs.index_select(1, self.b_solar_idx)
        b2 = obs.index_select(1, self.b_soc_idx)
        b3 = obs.index_select(1, self.b_net_idx)
        b  = torch.stack([b0, b1, b2, b3], dim=-1)
        b_emb = self.building_enc(b)

        # EVs [B,8,7]
        ev_rows = []
        for i in range(self.ev_idx.shape[0]):
            idxs = self.ev_idx[i]
            ev_rows.append(obs.index_select(1, idxs).unsqueeze(1))
        ev = torch.cat(ev_rows, dim=1)
        ev_emb = self.ev_enc(ev)

        # Washer [B,d_dev]
        w = obs.index_select(1, self.washer_idx)
        w_emb = self.washer_enc(w)

        z = torch.cat([g_emb, b_emb.reshape(B, -1), ev_emb.reshape(B, -1), w_emb], dim=-1)
        shared = self.trunk(z)

        batt_actions = [self.batt_head(torch.cat([shared, b_emb[:, i, :]], dim=-1)) for i in range(17)]
        ev_actions   = [self.ev_head(torch.cat([shared, ev_emb[:, j, :]], dim=-1)) for j in range(8)]
        wash_action  = self.wash_head(torch.cat([shared, w_emb], dim=-1))

        act = torch.cat(batt_actions + ev_actions + [wash_action], dim=-1)  # [B,26]
        return torch.tanh(act)


# ----------------------------
# Approach 2 Mean Network (Approach 1 + Light Attention)
# ----------------------------
class StructuredMeanObsIndexAttention(nn.Module):
    def __init__(self, groups, obs_dim=153, act_dim=26, d_dev=16, d_global=32, d_shared=64, n_heads=4):
        super().__init__()
        assert obs_dim == 153
        assert act_dim == 26

        # same gather buffers as A1
        self.register_buffer("time_idx",   torch.tensor(groups["time_idx"], dtype=torch.long))
        self.register_buffer("weather_idx",torch.tensor(groups["weather_idx"], dtype=torch.long))
        self.register_buffer("price_idx",  torch.tensor(groups["price_idx"], dtype=torch.long))
        self.register_buffer("carbon_idx", torch.tensor(groups["carbon_idx"], dtype=torch.long))
        self.register_buffer("washer_idx", torch.tensor(groups["washer_idx"], dtype=torch.long))

        self.register_buffer("b_nonshift_idx", torch.tensor(groups["b_nonshift"], dtype=torch.long))
        self.register_buffer("b_solar_idx",    torch.tensor(groups["b_solar"], dtype=torch.long))
        self.register_buffer("b_soc_idx",      torch.tensor(groups["b_soc"], dtype=torch.long))
        self.register_buffer("b_net_idx",      torch.tensor(groups["b_net"], dtype=torch.long))
        self.register_buffer("ev_idx", torch.tensor(groups["ev_indices"], dtype=torch.long))  # [8,7]

        global_in = (
            len(groups["time_idx"]) + len(groups["weather_idx"]) +
            len(groups["price_idx"]) + len(groups["carbon_idx"]) + len(groups["washer_idx"])
        )

        self.global_enc   = nn.Sequential(nn.Linear(global_in, 64), nn.ReLU(), nn.Linear(64, d_global), nn.ReLU())
        self.building_enc = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, d_dev), nn.ReLU())
        self.ev_enc       = nn.Sequential(nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, d_dev), nn.ReLU())
        self.washer_enc   = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, d_dev), nn.ReLU())

        # light attention over 26 device tokens
        self.attn = nn.MultiheadAttention(embed_dim=d_dev, num_heads=n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_dev)

        trunk_in = d_global + (26 * d_dev)
        self.trunk = nn.Sequential(nn.Linear(trunk_in, d_shared), nn.ReLU(), nn.Linear(d_shared, d_shared), nn.ReLU())

        self.batt_head = nn.Sequential(nn.Linear(d_shared + d_dev, 32), nn.ReLU(), nn.Linear(32, 1))
        self.ev_head   = nn.Sequential(nn.Linear(d_shared + d_dev, 32), nn.ReLU(), nn.Linear(32, 1))
        self.wash_head = nn.Sequential(nn.Linear(d_shared + d_dev, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]

        g = torch.cat([
            obs.index_select(1, self.time_idx),
            obs.index_select(1, self.weather_idx),
            obs.index_select(1, self.carbon_idx),
            obs.index_select(1, self.price_idx),
            obs.index_select(1, self.washer_idx),
        ], dim=-1)
        g_emb = self.global_enc(g)

        # buildings -> tokens [B,17,d_dev]
        b0 = obs.index_select(1, self.b_nonshift_idx)
        b1 = obs.index_select(1, self.b_solar_idx)
        b2 = obs.index_select(1, self.b_soc_idx)
        b3 = obs.index_select(1, self.b_net_idx)
        b  = torch.stack([b0, b1, b2, b3], dim=-1)
        b_emb = self.building_enc(b)

        # EVs -> tokens [B,8,d_dev]
        ev_rows = []
        for i in range(self.ev_idx.shape[0]):
            idxs = self.ev_idx[i]
            ev_rows.append(obs.index_select(1, idxs).unsqueeze(1))
        ev = torch.cat(ev_rows, dim=1)
        ev_emb = self.ev_enc(ev)

        # washer -> token [B,1,d_dev]
        w = obs.index_select(1, self.washer_idx)
        w_emb = self.washer_enc(w).unsqueeze(1)

        tokens = torch.cat([b_emb, ev_emb, w_emb], dim=1)  # [B,26,d_dev]

        # self-attention (1 layer) + residual
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.attn_norm(tokens + attn_out)

        z = torch.cat([g_emb, tokens.reshape(B, -1)], dim=-1)
        shared = self.trunk(z)

        batt_actions = [self.batt_head(torch.cat([shared, tokens[:, i, :]], dim=-1)) for i in range(17)]
        ev_actions   = [self.ev_head(torch.cat([shared, tokens[:, 17 + j, :]], dim=-1)) for j in range(8)]
        wash_action  = self.wash_head(torch.cat([shared, tokens[:, 25, :]], dim=-1))

        act = torch.cat(batt_actions + ev_actions + [wash_action], dim=-1)
        return torch.tanh(act)


# ----------------------------
# GaussianLearningActor wrapper (keeps OmniSafe happy)
# ----------------------------
class CustomGaussianLearningActor(GaussianLearningActor):
    def __init__(self, obs_space, act_space, hidden_sizes, mean_net: nn.Module, init_std: float, activation='relu', weight_initialization_mode='kaiming_uniform'):
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self.mean = mean_net
        with torch.no_grad():
            self.log_std.fill_(torch.log(torch.tensor(init_std, device=self.log_std.device)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to yaml config (your repo configs/...)")
    ap.add_argument("--log_dir", default=None, help="Override logger_cfgs.log_dir")
    ap.add_argument("--attention", action="store_true", help="Use Approach 2 (light attention)")
    ap.add_argument("--init_std", type=float, default=0.30, help="Initial Gaussian std")
    ap.add_argument("--smoke_1epoch", action="store_true", help="Override to 1 epoch (8759 steps) and fewer update iters")
    args = ap.parse_args()

    # Load cfg
    cfg = yaml.safe_load(open(args.cfg, "r"))
    algo = cfg["algo"]
    env_id = cfg["env_id"]

    allowed = ("train_cfgs","algo_cfgs","logger_cfgs","lagrange_cfgs","model_cfgs","save_cfgs","env_cfgs","reward_model_cfgs")
    custom_cfgs = {k: v for k, v in cfg.items() if k in allowed}

    # Keep actor_type gaussian_learning so OmniSafe registers Train/PolicyStd etc.
    custom_cfgs.setdefault("model_cfgs", {})
    custom_cfgs["model_cfgs"]["actor_type"] = "gaussian_learning"

    # Optional: override log dir
    custom_cfgs.setdefault("logger_cfgs", {})
    if args.log_dir is not None:
        custom_cfgs["logger_cfgs"]["log_dir"] = args.log_dir

    # Optional smoke
    if args.smoke_1epoch:
        custom_cfgs.setdefault("train_cfgs", {})
        custom_cfgs.setdefault("algo_cfgs", {})
        custom_cfgs["train_cfgs"]["total_steps"] = 8759
        custom_cfgs["algo_cfgs"]["steps_per_epoch"] = 8759
        custom_cfgs["algo_cfgs"]["update_iters"] = 5

    # Build ObsIndex once (no guessing)
    obs_index = build_obs_index_from_local_env()
    groups = obsindex_to_groups(obs_index)

    # Choose mean net
    if args.attention:
        mean_net = StructuredMeanObsIndexAttention(groups)
        brain_name = "Approach2_Attention"
    else:
        mean_net = StructuredMeanObsIndex(groups)
        brain_name = "Approach1_Structured"

    # Monkeypatch builder only inside this script
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

    print("\n=== TRAIN STRUCTURED BRAIN ===")
    print("Brain:", brain_name)
    print("Algo:", algo)
    print("Env :", env_id)
    print("Init std:", args.init_std)
    print("EV key order:", groups["ev_keys"])
    print("Log dir:", custom_cfgs.get("logger_cfgs", {}).get("log_dir", "(omnisafe default)"))
    print("==============================\n")

    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)

    # === VERIFY WE ARE USING THE CUSTOM BRAIN (NOT DEFAULT) ===
    actor = agent.agent._actor_critic.actor
    print(">>> VERIFY actor class:", actor.__class__.__name__)
    print(">>> VERIFY mean  class:", actor.mean.__class__.__name__)
    print(">>> VERIFY policy std  :", actor.std)
    # also verify a forward pass shape
    import torch
    test_obs = torch.zeros(2, actor._obs_dim)
    test_act = actor.predict(test_obs, deterministic=True)
    print(">>> VERIFY mean action shape:", tuple(test_act.shape), "min/max:", float(test_act.min()), float(test_act.max()))
    print(">>> VERIFY done. If actor is CustomGaussianLearningActor and mean is StructuredMean..., you're using the custom brain.\n")

    agent.learn()


if __name__ == "__main__":
    main()
