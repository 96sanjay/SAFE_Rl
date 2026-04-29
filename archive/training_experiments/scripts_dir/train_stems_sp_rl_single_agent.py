"""Isolated STEMS + SP-RL-style trainer with all-4 projected execution.

This path is separate from PPO and from the earlier value-based custom
single-agent trainer. Design:
  - actor proposes raw action
  - all-4 projector computes safe action (C0/C2/C3/C4)
  - environment executes safe action
  - reward / dense C1 critics learn from safe execution
  - penalty critic learns projection magnitude on raw actions
  - policy sees projector state appended to observation
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import citylearn_safe.omni_env  # noqa: F401
import citylearn_safe.omni_env_v2  # noqa: F401

from citylearn_safe.action_projection_stems_all4 import StemsAll4Projector
from citylearn_safe.omni_env_v2 import CityLearnCMDPv2
from citylearn_safe.pid_lagrange import PIDLagrange
from citylearn_safe.projector_state_obs_wrapper import ProjectorStateObsWrapper
from citylearn_safe.stems_5bld_factory import build_obs_index_5bld_spec, build_stems_encoder_v3_5bld
from citylearn_safe.stems_sp_rl import build_stems_sp_rl_actor_critic


DEFAULT_5BLD_SCHEMA = os.path.join(
    PROJECT_ROOT,
    "data",
    "citylearn_challenge_2022_phase_all_plus_evs",
    "schema_5buildings.json",
)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_defaults() -> dict[str, Any]:
    return {
        "algo": "STEMSSPRLAll4",
        "env_id": "CityLearnSafety-V2G-v2",
        "seed": 42,
        "schema_path": DEFAULT_5BLD_SCHEMA,
        "train_cfgs": {
            "epochs": 10,
            "steps_per_epoch": 1024,
            "gamma": 0.99,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "penalty_weight": 1.0,
            "bc_coef": 0.5,
            "tau": 0.01,
            "batch_size": 256,
            "buffer_size": 200000,
            "warmup_steps": 2048,
            "update_iters": 256,
            "policy_noise": 0.1,
            "policy_noise_clip": 0.2,
            "device": "cpu",
            "save_every": 1,
        },
        "lagrange_cfgs": {
            "cost_limit": 5.0,
            "pid_kp": 0.1,
            "pid_ki": 0.01,
            "pid_kd": 0.0,
            "penalty_max": 3.0,
            "lagrangian_multiplier_init": 0.001,
        },
        "encoder_cfgs": {
            "temporal_window": 12,
            "hidden_dim": 64,
            "global_hidden": 32,
            "temporal_hidden": 32,
            "temporal_heads": 4,
            "num_gcn_layers": 3,
            "output_dim": 256,
            "dropout": 0.1,
            "actor_hidden_sizes": [128, 128],
            "critic_hidden_sizes": [128, 128],
        },
    }


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


@dataclass
class Transition:
    obs: np.ndarray
    act_raw: np.ndarray
    act_safe: np.ndarray
    reward: float
    cost_c1: float
    penalty_h: float
    done: float
    next_obs: np.ndarray
    proj_state: np.ndarray
    next_proj_state: np.ndarray


class ReplayBuffer:
    def __init__(self, size: int):
        self._buf: deque[Transition] = deque(maxlen=int(size))

    def add(self, tr: Transition) -> None:
        self._buf.append(tr)

    def __len__(self) -> int:
        return len(self._buf)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        batch = random.sample(self._buf, batch_size)
        return {
            "obs": torch.as_tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=device),
            "act_raw": torch.as_tensor(np.stack([t.act_raw for t in batch]), dtype=torch.float32, device=device),
            "act_safe": torch.as_tensor(np.stack([t.act_safe for t in batch]), dtype=torch.float32, device=device),
            "reward": torch.as_tensor(np.asarray([t.reward for t in batch]), dtype=torch.float32, device=device),
            "cost_c1": torch.as_tensor(np.asarray([t.cost_c1 for t in batch]), dtype=torch.float32, device=device),
            "penalty_h": torch.as_tensor(np.asarray([t.penalty_h for t in batch]), dtype=torch.float32, device=device),
            "done": torch.as_tensor(np.asarray([t.done for t in batch]), dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=device),
            "proj_state": torch.as_tensor(np.stack([t.proj_state for t in batch]), dtype=torch.float32, device=device),
            "next_proj_state": torch.as_tensor(
                np.stack([t.next_proj_state for t in batch]),
                dtype=torch.float32,
                device=device,
            ),
        }


def make_lagrange(cfg: dict[str, Any]) -> PIDLagrange:
    return PIDLagrange(
        cost_limit=float(cfg["cost_limit"]),
        pid_kp=float(cfg["pid_kp"]),
        pid_ki=float(cfg["pid_ki"]),
        pid_kd=float(cfg["pid_kd"]),
        pid_d_delay=10,
        pid_delta_p_ema_alpha=0.95,
        pid_delta_d_ema_alpha=0.95,
        penalty_max=float(cfg["penalty_max"]),
        lagrangian_multiplier_init=float(cfg["lagrangian_multiplier_init"]),
    )


def build_run_dir(algo: str, env_id: str, seed: int) -> Path:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = (
        Path(PROJECT_ROOT)
        / "runs"
        / "stems_sp_rl_single_agent"
        / "5bld"
        / f"{algo}-{{{env_id}}}"
        / f"seed-{seed:03d}-{stamp}"
    )
    (run_dir / "torch_save").mkdir(parents=True, exist_ok=True)
    return run_dir


def init_progress_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "Train/Epoch",
                "TotalEnvSteps",
                "Metrics/EpRet",
                "Metrics/EpLen",
                "Loss/Actor",
                "Loss/RewardCritic",
                "Loss/CostCritic",
                "Loss/PenaltyCritic",
                "Lambda/C1",
                "Cost/C1",
                "Penalty/Mean",
            ],
        )


def append_progress_csv(path: Path, row: list[Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def save_checkpoint(path: Path, model, optimizers: dict[str, Any], epoch: int, lambda_c1: float, cfg: dict[str, Any]) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
            "lambda_c1": float(lambda_c1),
            "cfg": cfg,
        },
        path,
    )


def train(cfg: dict[str, Any]) -> Path:
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_cfgs = cfg["train_cfgs"]
    encoder_cfgs = cfg["encoder_cfgs"]
    schema_path = cfg["schema_path"]
    device = torch.device(train_cfgs["device"])

    overrides = {
        "CITYLEARN_SCHEMA": schema_path,
        "CITYLEARN_TEMPORAL_WINDOW": str(encoder_cfgs["temporal_window"]),
        "STEMS_ENCODER_VERSION": "v3",
        "MASK_C4_ENABLED": "1",
    }

    with temporary_env_overrides(overrides):
        spec = build_obs_index_5bld_spec(
            schema_path=schema_path,
            temporal_window=int(encoder_cfgs["temporal_window"]),
        )
        base_env = CityLearnCMDPv2(cfg["env_id"])
        projector = StemsAll4Projector(base_env)
        env = ProjectorStateObsWrapper(base_env, projector)

        env_obs_dim = int(env.observation_space.shape[0])
        env_act_dim = int(env.action_space.shape[0])
        if env_act_dim != spec.act_dim:
            raise AssertionError(f"Live env act_dim {env_act_dim} does not match STEMS spec act_dim {spec.act_dim}.")
        if env_obs_dim < spec.encoder_obs_dim:
            raise AssertionError(f"Live env obs_dim {env_obs_dim} smaller than STEMS encoder obs_dim {spec.encoder_obs_dim}.")
        extra_obs_dim = env_obs_dim - spec.encoder_obs_dim

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
        model = build_stems_sp_rl_actor_critic(
            spec=spec,
            encoder=encoder,
            actor_hidden_sizes=tuple(int(x) for x in encoder_cfgs["actor_hidden_sizes"]),
            critic_hidden_sizes=tuple(int(x) for x in encoder_cfgs["critic_hidden_sizes"]),
            extra_obs_dim=extra_obs_dim,
        ).to(device)

        actor_optimizer = torch.optim.Adam(
            list(model.backbone.parameters()) + list(model.actor.parameters()),
            lr=float(train_cfgs["actor_lr"]),
        )
        reward_critic_optimizer = torch.optim.Adam(
            list(model.reward_q1.parameters()) + list(model.reward_q2.parameters()),
            lr=float(train_cfgs["critic_lr"]),
        )
        cost_critic_optimizer = torch.optim.Adam(model.cost_q1.parameters(), lr=float(train_cfgs["critic_lr"]))
        penalty_critic_optimizer = torch.optim.Adam(model.penalty_q.parameters(), lr=float(train_cfgs["critic_lr"]))

        lagrange = make_lagrange(cfg["lagrange_cfgs"])
        replay = ReplayBuffer(int(train_cfgs["buffer_size"]))
        run_dir = build_run_dir(cfg["algo"], cfg["env_id"], seed)
        progress_csv = run_dir / "progress.csv"
        init_progress_csv(progress_csv)

        gamma = float(train_cfgs["gamma"])
        batch_size = int(train_cfgs["batch_size"])
        warmup_steps = int(train_cfgs["warmup_steps"])
        update_iters = int(train_cfgs["update_iters"])
        tau = float(train_cfgs["tau"])
        policy_noise = float(train_cfgs["policy_noise"])
        policy_noise_clip = float(train_cfgs["policy_noise_clip"])
        penalty_weight = float(train_cfgs["penalty_weight"])
        bc_coef = float(train_cfgs["bc_coef"])

        total_steps = 0
        obs, _ = env.reset()
        ep_ret = 0.0
        ep_len = 0
        ep_cost_c1 = 0.0

        for epoch in range(int(train_cfgs["epochs"])):
            completed_returns: list[float] = []
            completed_lens: list[int] = []
            completed_costs: list[float] = []
            penalties_epoch: list[float] = []

            for _ in range(int(train_cfgs["steps_per_epoch"])):
                obs_t = to_tensor(obs, device).unsqueeze(0)
                with torch.no_grad():
                    raw_action_t = model.act(obs_t)
                proj_state_t = projector.extract_state_tensor().unsqueeze(0).to(device)
                safe_action_t, proj_info = projector.project(obs_t, raw_action_t, state_tensors=proj_state_t)
                raw_action = raw_action_t.squeeze(0).cpu().numpy()
                safe_action = safe_action_t.squeeze(0).cpu().numpy()
                penalty_h = penalty_weight * float(((raw_action - safe_action) ** 2).sum())

                next_obs, reward, _, terminated, truncated, info = env.step(safe_action)
                done = bool(terminated.item() if hasattr(terminated, "item") else terminated) or bool(
                    truncated.item() if hasattr(truncated, "item") else truncated
                )
                next_proj_state_t = projector.extract_state_tensor().unsqueeze(0).to(device)
                cost_c1 = float(info.get("cost_ev_dense", 0.0))

                replay.add(
                    Transition(
                        obs=np.asarray(obs, dtype=np.float32),
                        act_raw=np.asarray(raw_action, dtype=np.float32),
                        act_safe=np.asarray(safe_action, dtype=np.float32),
                        reward=float(reward),
                        cost_c1=cost_c1,
                        penalty_h=penalty_h,
                        done=float(done),
                        next_obs=np.asarray(next_obs, dtype=np.float32),
                        proj_state=proj_state_t.squeeze(0).cpu().numpy().astype(np.float32),
                        next_proj_state=next_proj_state_t.squeeze(0).cpu().numpy().astype(np.float32),
                    ),
                )

                ep_ret += float(reward)
                ep_len += 1
                ep_cost_c1 += cost_c1
                penalties_epoch.append(penalty_h)
                total_steps += 1
                obs = next_obs

                if done:
                    completed_returns.append(ep_ret)
                    completed_lens.append(ep_len)
                    completed_costs.append(ep_cost_c1)
                    obs, _ = env.reset()
                    ep_ret = 0.0
                    ep_len = 0
                    ep_cost_c1 = 0.0

            actor_loss_v = 0.0
            reward_loss_v = 0.0
            cost_loss_v = 0.0
            penalty_loss_v = 0.0

            if len(replay) >= max(batch_size, warmup_steps):
                lambda_c1 = float(lagrange.lagrangian_multiplier)
                for _ in range(update_iters):
                    data = replay.sample(batch_size, device=device)
                    obs_b = data["obs"]
                    next_obs_b = data["next_obs"]
                    act_raw_b = data["act_raw"]
                    act_safe_b = data["act_safe"]
                    reward_b = data["reward"]
                    cost_c1_b = data["cost_c1"]
                    penalty_h_b = data["penalty_h"]
                    done_b = data["done"]
                    proj_state_b = data["proj_state"]
                    next_proj_state_b = data["next_proj_state"]

                    with torch.no_grad():
                        next_raw = model.target_act(next_obs_b)
                        if policy_noise > 0.0:
                            noise = torch.randn_like(next_raw) * policy_noise
                            noise = noise.clamp(-policy_noise_clip, policy_noise_clip)
                            next_raw = (next_raw + noise).clamp(-1.0, 1.0)
                        next_safe, _ = projector.project(next_obs_b, next_raw, state_tensors=next_proj_state_b)
                        next_feat = model.target_features(next_obs_b)
                        q1_next = model.target_reward_q1(next_feat, next_safe)
                        q2_next = model.target_reward_q2(next_feat, next_safe)
                        q_next = torch.min(q1_next, q2_next)
                        target_reward = reward_b + gamma * (1.0 - done_b) * q_next

                        q_c1_next = model.target_cost_q1(next_feat, next_safe)
                        target_cost = cost_c1_b + gamma * (1.0 - done_b) * q_c1_next

                        q_pen_next = model.target_penalty_q(next_feat, next_raw)
                        target_penalty = penalty_h_b + gamma * (1.0 - done_b) * q_pen_next

                    feat = model.features(obs_b)
                    reward_q1 = model.reward_q1(feat, act_safe_b)
                    reward_q2 = model.reward_q2(feat, act_safe_b)
                    reward_loss = F.mse_loss(reward_q1, target_reward) + F.mse_loss(reward_q2, target_reward)
                    reward_critic_optimizer.zero_grad(set_to_none=True)
                    reward_loss.backward()
                    reward_critic_optimizer.step()

                    feat = model.features(obs_b)
                    cost_q = model.cost_q1(feat, act_safe_b)
                    cost_loss = F.mse_loss(cost_q, target_cost)
                    cost_critic_optimizer.zero_grad(set_to_none=True)
                    cost_loss.backward()
                    cost_critic_optimizer.step()

                    feat = model.features(obs_b)
                    pen_q = model.penalty_q(feat, act_raw_b)
                    penalty_loss = F.mse_loss(pen_q, target_penalty)
                    penalty_critic_optimizer.zero_grad(set_to_none=True)
                    penalty_loss.backward()
                    penalty_critic_optimizer.step()

                    feat = model.features(obs_b)
                    raw_pred = model.actor(feat)
                    safe_pred, _ = projector.project(obs_b, raw_pred, state_tensors=proj_state_b)
                    safe_st = raw_pred + (safe_pred - raw_pred).detach()
                    reward_term = -torch.min(
                        model.reward_q1(feat, safe_st),
                        model.reward_q2(feat, safe_st),
                    ).mean()
                    cost_term = lambda_c1 * model.cost_q1(feat, safe_st).mean()
                    penalty_term = model.penalty_q(feat, raw_pred).mean()
                    bc_loss = F.mse_loss(raw_pred, safe_pred)
                    actor_loss = reward_term + cost_term + penalty_term + (bc_coef * bc_loss)

                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_optimizer.step()

                    model.polyak_update(tau)

                    actor_loss_v += float(actor_loss.detach().cpu().item())
                    reward_loss_v += float(reward_loss.detach().cpu().item())
                    cost_loss_v += float(cost_loss.detach().cpu().item())
                    penalty_loss_v += float(penalty_loss.detach().cpu().item())

                actor_loss_v /= update_iters
                reward_loss_v /= update_iters
                cost_loss_v /= update_iters
                penalty_loss_v /= update_iters

            mean_cost_c1 = float(np.mean(completed_costs)) if completed_costs else float(ep_cost_c1)
            lagrange.pid_update(mean_cost_c1)
            lambda_c1 = float(lagrange.lagrangian_multiplier)

            ep_ret_mean = float(np.mean(completed_returns)) if completed_returns else float(ep_ret)
            ep_len_mean = float(np.mean(completed_lens)) if completed_lens else float(ep_len or 1)
            penalty_mean = float(np.mean(penalties_epoch)) if penalties_epoch else 0.0

            append_progress_csv(
                progress_csv,
                [
                    epoch,
                    total_steps,
                    ep_ret_mean,
                    ep_len_mean,
                    actor_loss_v,
                    reward_loss_v,
                    cost_loss_v,
                    penalty_loss_v,
                    lambda_c1,
                    mean_cost_c1,
                    penalty_mean,
                ],
            )

            if (epoch + 1) % int(train_cfgs["save_every"]) == 0:
                save_checkpoint(
                    run_dir / "torch_save" / f"epoch-{epoch + 1}.pt",
                    model=model,
                    optimizers={
                        "actor": actor_optimizer,
                        "reward_critic": reward_critic_optimizer,
                        "cost_critic": cost_critic_optimizer,
                        "penalty_critic": penalty_critic_optimizer,
                    },
                    epoch=epoch + 1,
                    lambda_c1=lambda_c1,
                    cfg=cfg,
                )

            print(
                f"[Epoch {epoch:03d}] steps={total_steps} EpRet={ep_ret_mean:.2f} "
                f"ActorLoss={actor_loss_v:.4f} RewardCritic={reward_loss_v:.4f} "
                f"CostCritic={cost_loss_v:.4f} PenaltyCritic={penalty_loss_v:.4f} "
                f"LambdaC1={lambda_c1:.4f} CostC1={mean_cost_c1:.4f}"
            )

        env.close()
        return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train isolated STEMS SP-RL-style projected single-agent controller.")
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        custom = yaml.safe_load(f)
    cfg = deep_update(load_defaults(), custom)

    run_dir = train(cfg)
    print(f"\nRun complete: {run_dir}")


if __name__ == "__main__":
    main()
