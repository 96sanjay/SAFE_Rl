"""Evaluate the isolated STEMS SP-RL-style projected controller."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import citylearn_safe.omni_env  # noqa: F401
import citylearn_safe.omni_env_v2  # noqa: F401

from citylearn_safe.action_projection_stems_all4 import StemsAll4Projector
from citylearn_safe.omni_env_v2 import CityLearnCMDPv2
from citylearn_safe.projector_state_obs_wrapper import ProjectorStateObsWrapper
from citylearn_safe.stems_5bld_factory import build_obs_index_5bld_spec, build_stems_encoder_v3_5bld
from citylearn_safe.stems_sp_rl import build_stems_sp_rl_actor_critic


DEFAULT_5BLD_SCHEMA = os.path.join(
    PROJECT_ROOT,
    "data",
    "citylearn_challenge_2022_phase_all_plus_evs",
    "schema_5buildings.json",
)


@contextmanager
def temporary_env_overrides(overrides: dict[str, str]):
    old_values = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def to_tensor(x: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def build_model_from_checkpoint(ckpt: dict[str, Any], env_obs_dim: int, device: torch.device):
    cfg = ckpt["cfg"]
    schema_path = cfg.get("schema_path", DEFAULT_5BLD_SCHEMA)
    encoder_cfgs = cfg["encoder_cfgs"]
    spec = build_obs_index_5bld_spec(
        schema_path=schema_path,
        temporal_window=int(encoder_cfgs["temporal_window"]),
    )

    encoder = build_stems_encoder_v3_5bld(
        spec,
        hidden_dim=int(encoder_cfgs["hidden_dim"]),
        global_hidden=int(encoder_cfgs["global_hidden"]),
        temporal_hidden=int(encoder_cfgs["temporal_hidden"]),
        temporal_heads=int(encoder_cfgs["temporal_heads"]),
        num_gcn_layers=int(encoder_cfgs["num_gcn_layers"]),
        output_dim=int(encoder_cfgs["output_dim"]),
        dropout=float(encoder_cfgs["dropout"]),
    )
    actor_in_dim = int(ckpt["model"]["actor.net.0.weight"].shape[1])
    extra_obs_dim = actor_in_dim - int(encoder_cfgs["output_dim"])
    if extra_obs_dim < 0:
        raise ValueError(
            f"Checkpoint implies negative extra_obs_dim: actor_in_dim={actor_in_dim}, "
            f"encoder_output_dim={int(encoder_cfgs['output_dim'])}.",
        )
    model = build_stems_sp_rl_actor_critic(
        spec=spec,
        encoder=encoder,
        actor_hidden_sizes=tuple(int(x) for x in encoder_cfgs["actor_hidden_sizes"]),
        critic_hidden_sizes=tuple(int(x) for x in encoder_cfgs["critic_hidden_sizes"]),
        extra_obs_dim=extra_obs_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def evaluate(checkpoint_path: Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    schema_path = cfg.get("schema_path", DEFAULT_5BLD_SCHEMA)
    device = torch.device(cfg["train_cfgs"].get("device", "cpu"))
    overrides = {
        "CITYLEARN_SCHEMA": schema_path,
        "CITYLEARN_TEMPORAL_WINDOW": str(cfg["encoder_cfgs"]["temporal_window"]),
        "STEMS_ENCODER_VERSION": "v3",
        "MASK_C4_ENABLED": "1",
    }

    with temporary_env_overrides(overrides):
        base_env = CityLearnCMDPv2(cfg["env_id"])
        projector = StemsAll4Projector(base_env)
        env = ProjectorStateObsWrapper(base_env, projector)
        model, _ = build_model_from_checkpoint(ckpt, int(env.observation_space.shape[0]), device)

        obs, _ = env.reset()
        total_reward = 0.0
        total_steps = 0

        c0_num = c0_den = 0
        c1_num = c1_den = 0
        c2_num = c2_den = 0
        c3_num = c3_den = 0
        c4_num = c4_den = 0

        changed_steps = 0
        delta_l2_sum = 0.0
        delta_linf_max = 0.0

        while True:
            obs_t = to_tensor(obs, device).unsqueeze(0)
            with torch.no_grad():
                raw_action_t = model.act(obs_t)
                proj_state_t = projector.extract_state_tensor().unsqueeze(0).to(device)
                safe_action_t, _ = projector.project(obs_t, raw_action_t, state_tensors=proj_state_t)
            raw = raw_action_t.squeeze(0).cpu().numpy()
            safe = safe_action_t.squeeze(0).cpu().numpy()
            delta = raw - safe
            if np.any(np.abs(delta) > 1e-6):
                changed_steps += 1
            delta_l2_sum += float(np.linalg.norm(delta))
            delta_linf_max = max(delta_linf_max, float(np.max(np.abs(delta))))

            obs, reward, _, terminated, truncated, info = env.step(safe)
            total_reward += float(reward)
            total_steps += 1

            c0 = float(info.get("cost_ev_departure", 0.0))
            c1 = float(info.get("cost_ev_dense", 0.0))
            c2 = float(info.get("cost_stems_battery", 0.0))
            c3 = float(info.get("cost_stems_building_power", 0.0))
            c4 = float(info.get("cost_stems_grid_power", 0.0))

            dep_events = int(info.get("ev_departure_events", 0) or (1 if c0 > 0.0 else 0))
            c0_num += int(c0 > 0.0) * max(dep_events, 1 if c0 > 0.0 else 0)
            c0_den += dep_events
            c1_num += int(c1 > 0.0)
            c1_den += 1

            num_buildings = int(getattr(base_env, "num_buildings", len(getattr(base_env, "buildings", [])) or 5))
            c2_num += int(c2 > 0.0) * num_buildings
            c2_den += num_buildings
            c3_num += int(c3 > 0.0) * num_buildings
            c3_den += num_buildings
            c4_num += int(c4 > 0.0)
            c4_den += 1

            done = bool(terminated.item() if hasattr(terminated, "item") else terminated) or bool(
                truncated.item() if hasattr(truncated, "item") else truncated
            )
            if done:
                break

        env.close()
        return {
            "checkpoint": str(checkpoint_path),
            "reward_sum": total_reward,
            "total_steps": total_steps,
            "C0": {"num": c0_num, "den": c0_den, "pct": (100.0 * c0_num / max(c0_den, 1))},
            "C1": {"num": c1_num, "den": c1_den, "pct": (100.0 * c1_num / max(c1_den, 1))},
            "C2": {"num": c2_num, "den": c2_den, "pct": (100.0 * c2_num / max(c2_den, 1))},
            "C3": {"num": c3_num, "den": c3_den, "pct": (100.0 * c3_num / max(c3_den, 1))},
            "C4": {"num": c4_num, "den": c4_den, "pct": (100.0 * c4_num / max(c4_den, 1))},
            "projection": {
                "changed_steps": changed_steps,
                "changed_pct": (100.0 * changed_steps / max(total_steps, 1)),
                "mean_l2": delta_l2_sum / max(total_steps, 1),
                "max_linf": delta_linf_max,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    result = evaluate(Path(args.checkpoint))
    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
