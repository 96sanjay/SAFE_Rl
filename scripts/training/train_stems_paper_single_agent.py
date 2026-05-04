"""Isolated paper-style single-agent STEMS actor-critic trainer.

This is intentionally separate from PPO and other legacy trainers.
It follows the paper's actor-critic structure:
  - shared spatial-temporal STEMS representation
  - stochastic actor
  - TD critic
  - policy gradient with TD advantage

For the local CMDP adaptation, the paper's CBF-QP safety layer is replaced by
repo-local Lagrangian controllers. They enter only through:
  1. scalar constrained reward r_lambda = r - sum_i lambda_i * c_i
  2. lambda updates from rollout episode costs
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Register envs (cmdp_env registers CityLearnSafety-V2G-v2)
import citylearn_safe.cmdp_env  # noqa: F401  @env_register side-effect

from citylearn_safe.cmdp_env import CityLearnCMDP
from citylearn_safe.feasibility_obs_wrapper import FeasibilityObsWrapper
from citylearn_safe.pid_lagrange import PIDLagrange
from citylearn_safe.stems_5bld_factory import build_obs_index_5bld_spec, build_stems_encoder_5bld
from citylearn_safe.stems_paper_single_agent import build_paper_single_agent_stems_ac

DEFAULT_5BLD_SCHEMA = os.path.join(
    PROJECT_ROOT,
    "data",
    "citylearn_challenge_2022_phase_all_plus_evs",
    "schema_5buildings.json",
)

DEFAULT_COST_KEYS = [
    "cost_ev_departure",
    "cost_ev_dense",
    "cost_stems_battery",
    "cost_stems_building_power",
    "cost_stems_grid_power",
]

UNIFIED_EV_COST_KEYS = [
    "cost_ev_service",
    "cost_stems_battery",
    "cost_stems_building_power",
    "cost_stems_grid_power",
]


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
        "algo": "STEMSPaperSingleAgent",
        "env_id": "CityLearnSafety-V2G-v2",
        "seed": 42,
        "schema_path": DEFAULT_5BLD_SCHEMA,
        "train_cfgs": {
            "epochs": 10,
            "steps_per_epoch": 1024,
            "gamma": 0.99,
            "gae_lambda_r": 0.95,
            "gae_lambda_c": 0.95,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "backbone_lr": 3e-4,
            "weight_decay": 0.0,
            "entropy_coef": 1e-3,
            "update_batch_size": 256,
            "device": "cpu",
            "save_every": 1,
            "require_complete_episodes_for_lagrange": True,
        },
        "lagrange_cfgs": {
            "cost_limits": [5.0, 5.0, 5000.0, 1500.0, 1500.0],
            "pid_kp": [5.0, 0.1, 0.1, 0.3, 0.2],
            "pid_ki": [0.0, 0.01, 0.01, 0.03, 0.02],
            "pid_kd": [0.0, 0.0, 0.0, 0.0, 0.0],
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
        "observation_cfgs": {
            "append_feasibility_bounds": False,
        },
        "actor_cfgs": {
            "use_action_bounds_mask": False,
        },
        "cost_cfgs": {
            "use_unified_ev_service_cost": False,
        },
    }


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    dones: torch.Tensor
    next_obs: torch.Tensor
    logp_old: torch.Tensor
    value_r: torch.Tensor
    value_cs: torch.Tensor
    episode_cost_means: np.ndarray
    episode_return_mean: float
    episode_len_mean: float


def resolve_cost_keys(cfg: dict[str, Any]) -> list[str]:
    cost_cfgs = cfg.get("cost_cfgs", {})
    if bool(cost_cfgs.get("use_unified_ev_service_cost", False)):
        return list(UNIFIED_EV_COST_KEYS)
    return list(DEFAULT_COST_KEYS)


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


def extract_costs(info: dict[str, Any], cost_keys: list[str]) -> np.ndarray:
    return np.asarray([float(info.get(k, 0.0)) for k in cost_keys], dtype=np.float32)


def make_lagranges(cfg: dict[str, Any], num_costs: int) -> list[PIDLagrange]:
    limits = cfg["cost_limits"]
    kp = cfg["pid_kp"]
    ki = cfg["pid_ki"]
    kd = cfg["pid_kd"]
    if not (len(limits) == len(kp) == len(ki) == len(kd) == num_costs):
        raise ValueError(
            f"Lagrange config lengths must all match num_costs={num_costs}, got "
            f"limits={len(limits)}, kp={len(kp)}, ki={len(ki)}, kd={len(kd)}.",
        )
    return [
        PIDLagrange(
            cost_limit=float(limits[i]),
            pid_kp=float(kp[i]),
            pid_ki=float(ki[i]),
            pid_kd=float(kd[i]),
            pid_d_delay=10,
            pid_delta_p_ema_alpha=0.95,
            pid_delta_d_ema_alpha=0.95,
            penalty_max=float(cfg["penalty_max"]),
            lagrangian_multiplier_init=float(cfg["lagrangian_multiplier_init"]),
        )
        for i in range(num_costs)
    ]


def collect_rollout(
    env: CityLearnCMDP,
    model,
    device: torch.device,
    steps_per_epoch: int,
    cost_keys: list[str],
) -> RolloutBatch:
    obs, _ = env.reset()

    obs_buf, act_buf, rew_buf, cost_buf = [], [], [], []
    done_buf, next_obs_buf, logp_buf, value_r_buf, value_cs_buf = [], [], [], [], []

    ep_cost = np.zeros(len(cost_keys), dtype=np.float64)
    ep_ret = 0.0
    ep_len = 0
    completed_ep_costs: list[np.ndarray] = []
    completed_ep_returns: list[float] = []
    completed_ep_lens: list[int] = []

    steps_collected = 0
    require_complete_episodes = bool(getattr(env, "_require_complete_episodes_for_lagrange", False))
    while steps_collected < steps_per_epoch or (require_complete_episodes and not completed_ep_costs):
        obs_t = to_tensor(obs, device).unsqueeze(0)
        with torch.no_grad():
            out = model.act(obs_t, deterministic=False)
        action = out.action.squeeze(0).cpu().numpy()

        next_obs, reward, _, terminated, truncated, info = env.step(action)
        done = bool(terminated.item() if hasattr(terminated, "item") else terminated) or bool(
            truncated.item() if hasattr(truncated, "item") else truncated
        )
        costs = extract_costs(info, cost_keys)

        obs_buf.append(obs_t.squeeze(0))
        act_buf.append(out.action.squeeze(0))
        rew_buf.append(torch.tensor(float(reward), dtype=torch.float32, device=device))
        cost_buf.append(torch.tensor(costs, dtype=torch.float32, device=device))
        done_buf.append(torch.tensor(float(done), dtype=torch.float32, device=device))
        next_obs_buf.append(to_tensor(next_obs, device))
        logp_buf.append(out.log_prob.squeeze(0))
        value_r_buf.append(out.value_r.squeeze(0))
        if out.values_c:
            value_cs_buf.append(torch.stack([v.squeeze(0) for v in out.values_c], dim=0))
        else:
            value_cs_buf.append(torch.zeros(len(cost_keys), dtype=torch.float32, device=device))

        ep_cost += costs
        ep_ret += float(reward)
        ep_len += 1
        steps_collected += 1

        obs = next_obs
        if done:
            completed_ep_costs.append(ep_cost.copy())
            completed_ep_returns.append(ep_ret)
            completed_ep_lens.append(ep_len)
            obs, _ = env.reset()
            ep_cost = np.zeros(len(cost_keys), dtype=np.float64)
            ep_ret = 0.0
            ep_len = 0

    if completed_ep_costs:
        episode_cost_means = np.mean(np.stack(completed_ep_costs, axis=0), axis=0)
        episode_return_mean = float(np.mean(completed_ep_returns))
        episode_len_mean = float(np.mean(completed_ep_lens))
    else:
        episode_cost_means = np.sum(torch.stack(cost_buf, dim=0).cpu().numpy(), axis=0)
        episode_return_mean = float(torch.stack(rew_buf, dim=0).sum().cpu().item())
        episode_len_mean = float(len(rew_buf))

    return RolloutBatch(
        obs=torch.stack(obs_buf, dim=0),
        actions=torch.stack(act_buf, dim=0),
        rewards=torch.stack(rew_buf, dim=0),
        costs=torch.stack(cost_buf, dim=0),
        dones=torch.stack(done_buf, dim=0),
        next_obs=torch.stack(next_obs_buf, dim=0),
        logp_old=torch.stack(logp_buf, dim=0),
        value_r=torch.stack(value_r_buf, dim=0),
        value_cs=torch.stack(value_cs_buf, dim=0),
        episode_cost_means=np.asarray(episode_cost_means, dtype=np.float64),
        episode_return_mean=episode_return_mean,
        episode_len_mean=episode_len_mean,
    )


def compute_gae(
    rewards_or_costs: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation for 1D or per-cost signals.

    Args:
        rewards_or_costs: [T] or [T, K]
        values: [T] or [T, K]
        next_values: [T] or [T, K]
        dones: [T]
    Returns:
        advantages: same shape as rewards_or_costs
        targets: same shape as rewards_or_costs
    """
    if rewards_or_costs.shape != values.shape or values.shape != next_values.shape:
        raise ValueError("GAE inputs must have matching shapes for signal, values, and next_values.")

    not_done = 1.0 - dones
    if rewards_or_costs.dim() == 1:
        deltas = rewards_or_costs + gamma * not_done * next_values - values
        adv = torch.zeros_like(rewards_or_costs)
        gae = torch.zeros((), dtype=rewards_or_costs.dtype, device=rewards_or_costs.device)
        for t in range(rewards_or_costs.size(0) - 1, -1, -1):
            gae = deltas[t] + gamma * lam * not_done[t] * gae
            adv[t] = gae
        return adv, adv + values

    if rewards_or_costs.dim() == 2:
        deltas = rewards_or_costs + gamma * not_done.unsqueeze(-1) * next_values - values
        adv = torch.zeros_like(rewards_or_costs)
        gae = torch.zeros(rewards_or_costs.size(1), dtype=rewards_or_costs.dtype, device=rewards_or_costs.device)
        for t in range(rewards_or_costs.size(0) - 1, -1, -1):
            gae = deltas[t] + gamma * lam * not_done[t] * gae
            adv[t] = gae
        return adv, adv + values

    raise ValueError("GAE expects rank-1 or rank-2 tensors.")


def build_run_dir(algo: str, env_id: str, seed: int) -> Path:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = (
        Path(PROJECT_ROOT)
        / "runs"
        / "stems_paper_single_agent"
        / "5bld"
        / f"{algo}-{{{env_id}}}"
        / f"seed-{seed:03d}-{stamp}"
    )
    (run_dir / "torch_save").mkdir(parents=True, exist_ok=True)
    return run_dir


def init_progress_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Train/Epoch",
                "TotalEnvSteps",
                "Metrics/EpRet",
                "Metrics/EpLen",
                "Loss/Actor",
                "Loss/Critic",
                "Lambda/C0",
                "Lambda/C1",
                "Lambda/C2",
                "Lambda/C3",
                "Lambda/C4",
                "Cost/C0",
                "Cost/C1",
                "Cost/C2",
                "Cost/C3",
                "Cost/C4",
            ]
        )


def append_progress_csv(path: Path, row: list[Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def save_checkpoint(path: Path, model, optimizer, epoch: int, lambdas: list[float], cfg: dict[str, Any]) -> None:
    optimizer_state = (
        {
            key: (opt.state_dict() if hasattr(opt, "state_dict") else opt)
            for key, opt in optimizer.items()
        }
        if isinstance(optimizer, dict)
        else optimizer.state_dict()
    )
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer_state,
            "lambdas": lambdas,
            "cfg": cfg,
        },
        path,
    )


def train(cfg: dict[str, Any]) -> Path:
    seed = int(cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_cfgs = cfg["train_cfgs"]
    encoder_cfgs = cfg["encoder_cfgs"]
    observation_cfgs = cfg.get("observation_cfgs", {})
    actor_cfgs = cfg.get("actor_cfgs", {})
    schema_path = cfg["schema_path"]
    device = torch.device(train_cfgs["device"])

    overrides = {
        "CITYLEARN_SCHEMA": schema_path,
        "CITYLEARN_TEMPORAL_WINDOW": str(encoder_cfgs["temporal_window"]),
        "STEMS_ENCODER_VERSION": "v3",
    }

    with temporary_env_overrides(overrides):
        spec = build_obs_index_5bld_spec(
            schema_path=schema_path,
            temporal_window=int(encoder_cfgs["temporal_window"]),
        )
        env = CityLearnCMDP(cfg["env_id"])
        if bool(observation_cfgs.get("append_feasibility_bounds", False)):
            env = FeasibilityObsWrapper(env)

        env_obs_dim = int(env.observation_space.shape[0])
        env_act_dim = int(env.action_space.shape[0])
        if env_obs_dim < spec.encoder_obs_dim:
            raise AssertionError(
                f"Live env obs_dim {env_obs_dim} is smaller than STEMS spec encoder_obs_dim {spec.encoder_obs_dim}."
            )
        if env_act_dim != spec.act_dim:
            raise AssertionError(f"Live env act_dim {env_act_dim} does not match STEMS spec act_dim {spec.act_dim}.")
        extra_obs_dim = env_obs_dim - spec.encoder_obs_dim

        encoder = build_stems_encoder_5bld(
            spec,
            hidden_dim=int(encoder_cfgs["hidden_dim"]),
            global_hidden=int(encoder_cfgs["global_hidden"]),
            temporal_hidden=int(encoder_cfgs["temporal_hidden"]),
            temporal_heads=int(encoder_cfgs["temporal_heads"]),
            num_gcn_layers=int(encoder_cfgs["num_gcn_layers"]),
            output_dim=int(encoder_cfgs["output_dim"]),
            dropout=float(encoder_cfgs["dropout"]),
        )
        cost_keys = resolve_cost_keys(cfg)
        cfg["resolved_cost_keys"] = list(cost_keys)
        model = build_paper_single_agent_stems_ac(
            spec=spec,
            encoder=encoder,
            num_costs=len(cost_keys),
            actor_hidden_sizes=tuple(int(x) for x in encoder_cfgs["actor_hidden_sizes"]),
            critic_hidden_sizes=tuple(int(x) for x in encoder_cfgs["critic_hidden_sizes"]),
            extra_obs_dim=extra_obs_dim,
            use_action_bounds_mask=bool(actor_cfgs.get("use_action_bounds_mask", False)),
            action_bounds_dim=(2 * spec.act_dim if bool(observation_cfgs.get("append_feasibility_bounds", False)) else 0),
        ).to(device)

        env._require_complete_episodes_for_lagrange = bool(train_cfgs["require_complete_episodes_for_lagrange"])

        actor_lr = float(train_cfgs["actor_lr"])
        critic_lr = float(train_cfgs["critic_lr"])
        backbone_lr = float(train_cfgs["backbone_lr"])
        weight_decay = float(train_cfgs["weight_decay"])

        actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=actor_lr, weight_decay=weight_decay)
        critic_optimizer = torch.optim.Adam(
            list(model.value_r.parameters()) + list(model.value_costs.parameters()),
            lr=critic_lr,
            weight_decay=weight_decay,
        )
        backbone_optimizer = torch.optim.Adam(model.backbone.parameters(), lr=backbone_lr, weight_decay=weight_decay)
        lagranges = make_lagranges(cfg["lagrange_cfgs"], num_costs=len(cost_keys))

        run_dir = build_run_dir(cfg["algo"], cfg["env_id"], seed)
        progress_csv = run_dir / "progress.csv"
        init_progress_csv(progress_csv)

        epochs = int(train_cfgs["epochs"])
        steps_per_epoch = int(train_cfgs["steps_per_epoch"])
        gamma = float(train_cfgs["gamma"])
        gae_lambda_r = float(train_cfgs["gae_lambda_r"])
        gae_lambda_c = float(train_cfgs["gae_lambda_c"])
        entropy_coef = float(train_cfgs["entropy_coef"])
        update_batch_size = int(train_cfgs["update_batch_size"])
        save_every = int(train_cfgs["save_every"])

        total_steps = 0
        for epoch in range(epochs):
            batch = collect_rollout(env, model, device, steps_per_epoch=steps_per_epoch, cost_keys=cost_keys)
            total_steps += int(batch.obs.size(0))

            lambdas = torch.tensor(
                [float(lag.lagrangian_multiplier) for lag in lagranges],
                dtype=torch.float32,
                device=device,
            )

            with torch.no_grad():
                next_features = model.features(batch.next_obs)
                next_value_r, next_values_c = model.values(next_features)
                next_values_c_t = torch.stack(next_values_c, dim=1)

            adv_r, target_r = compute_gae(
                rewards_or_costs=batch.rewards,
                values=batch.value_r,
                next_values=next_value_r,
                dones=batch.dones,
                gamma=gamma,
                lam=gae_lambda_r,
            )
            adv_c, target_cs = compute_gae(
                rewards_or_costs=batch.costs,
                values=batch.value_cs,
                next_values=next_values_c_t,
                dones=batch.dones,
                gamma=gamma,
                lam=gae_lambda_c,
            )
            constrained_adv = adv_r - (adv_c * lambdas.unsqueeze(0)).sum(dim=-1)
            constrained_adv = (constrained_adv - constrained_adv.mean()) / (
                constrained_adv.std(unbiased=False) + 1e-8
            )
            actor_loss_total = 0.0
            critic_loss_total = 0.0
            sample_count = int(batch.obs.size(0))

            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            backbone_optimizer.zero_grad(set_to_none=True)

            for start in range(0, sample_count, update_batch_size):
                end = min(start + update_batch_size, sample_count)
                mb_obs = batch.obs[start:end]
                mb_actions = batch.actions[start:end]
                mb_adv = constrained_adv[start:end].detach()
                mb_target_r = target_r[start:end].detach()
                mb_target_cs = target_cs[start:end].detach()

                logp_new, entropy, value_r_new, value_cs_new = model.evaluate_actions(mb_obs, mb_actions)
                value_cs_new_t = torch.stack(value_cs_new, dim=1)

                actor_loss_mb = -(logp_new * mb_adv).mean() - entropy_coef * entropy.mean()
                reward_critic_loss_mb = F.mse_loss(value_r_new, mb_target_r)
                cost_critic_loss_mb = F.mse_loss(value_cs_new_t, mb_target_cs)
                critic_loss_mb = reward_critic_loss_mb + cost_critic_loss_mb
                total_loss_mb = (actor_loss_mb + critic_loss_mb) * ((end - start) / sample_count)

                total_loss_mb.backward()
                actor_loss_total += float(actor_loss_mb.detach().cpu().item()) * (end - start)
                critic_loss_total += float(critic_loss_mb.detach().cpu().item()) * (end - start)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            actor_optimizer.step()
            critic_optimizer.step()
            backbone_optimizer.step()

            actor_loss = actor_loss_total / sample_count
            critic_loss = critic_loss_total / sample_count

            for lag, ep_cost in zip(lagranges, batch.episode_cost_means):
                lag.pid_update(float(ep_cost))

            lambda_list = [float(lag.lagrangian_multiplier) for lag in lagranges]
            append_progress_csv(
                progress_csv,
                [
                    epoch,
                    total_steps,
                    batch.episode_return_mean,
                    batch.episode_len_mean,
                    float(actor_loss),
                    float(critic_loss),
                    *lambda_list,
                    *[float(x) for x in batch.episode_cost_means.tolist()],
                ],
            )

            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    run_dir / "torch_save" / f"epoch-{epoch + 1}.pt",
                    model=model,
                    optimizer={
                        "actor": actor_optimizer.state_dict(),
                        "critic": critic_optimizer.state_dict(),
                        "backbone": backbone_optimizer.state_dict(),
                    },
                    epoch=epoch + 1,
                    lambdas=lambda_list,
                    cfg=cfg,
                )

            print(
                f"[Epoch {epoch:03d}] steps={total_steps} "
                f"EpRet={batch.episode_return_mean:.2f} "
                f"ActorLoss={float(actor_loss):.4f} "
                f"CriticLoss={float(critic_loss):.4f} "
                f"Lambdas={[round(x, 4) for x in lambda_list]}"
            )

        env.close()
        return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train isolated paper-style single-agent STEMS actor-critic.")
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        custom = yaml.safe_load(f)
    cfg = deep_update(load_defaults(), custom)

    run_dir = train(cfg)
    print(f"\nRun complete: {run_dir}")


if __name__ == "__main__":
    main()
