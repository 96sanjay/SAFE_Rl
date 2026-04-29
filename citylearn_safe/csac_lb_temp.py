"""Paper-style CSAC-LB for the cooling-only temperature case study."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.adapter.offpolicy_adapter import OffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.envs.core import CMDP, make, support_envs
from omnisafe.envs.wrapper import (
    ActionScale,
    AutoReset,
    CostNormalize,
    ObsNormalize,
    RewardNormalize,
    TimeLimit,
    Unsqueeze,
)
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import Config
from omnisafe.utils.tools import get_device

from citylearn_safe.csac_lb_actor_q_critic import CSACLBActorQCritic


class _CSACLBOffPolicyAdapter(OffPolicyAdapter):
    """Off-policy adapter with separate train/eval env configs."""

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        assert env_id in support_envs(), f"Env {env_id} is not supported."
        self._cfgs = cfgs
        self._device = get_device(cfgs.train_cfgs.device)
        self._env_id = env_id

        env_cfgs = {}
        if hasattr(cfgs, "env_cfgs") and cfgs.env_cfgs is not None:
            env_cfgs = cfgs.env_cfgs.todict()
        eval_env_cfgs = env_cfgs
        if hasattr(cfgs, "eval_env_cfgs") and cfgs.eval_env_cfgs is not None:
            eval_env_cfgs = cfgs.eval_env_cfgs.todict()

        self._env: CMDP = make(env_id, num_envs=num_envs, device=self._device, **env_cfgs)
        self._eval_env: CMDP = make(env_id, num_envs=1, device=self._device, **eval_env_cfgs)

        self._wrapper(
            obs_normalize=cfgs.algo_cfgs.obs_normalize,
            reward_normalize=cfgs.algo_cfgs.reward_normalize,
            cost_normalize=cfgs.algo_cfgs.cost_normalize,
        )
        self._env.set_seed(seed)
        self._eval_env.set_seed(seed)
        self._current_obs, _ = self.reset()
        self._max_ep_len = 1000
        self._reset_log()

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        return obs.to(self._device), info

    def step(self, action: torch.Tensor):
        obs, reward, cost, terminated, truncated, info = super().step(action)
        obs = obs.to(self._device)
        reward = reward.to(self._device)
        cost = cost.to(self._device)
        terminated = terminated.to(self._device)
        truncated = truncated.to(self._device)
        final = info.get("final_observation")
        if isinstance(final, torch.Tensor):
            info["final_observation"] = final.to(self._device)
        return obs, reward, cost, terminated, truncated, info

    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ) -> None:
        if self._env.need_time_limit_wrapper:
            assert self._env.max_episode_steps and self._eval_env.max_episode_steps
            self._env = TimeLimit(
                self._env,
                time_limit=self._env.max_episode_steps,
                device=self._device,
            )
            self._eval_env = TimeLimit(
                self._eval_env,
                time_limit=self._eval_env.max_episode_steps,
                device=self._device,
            )
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env, device=self._device)
            self._eval_env = AutoReset(self._eval_env, device=self._device)
        if obs_normalize:
            self._env = ObsNormalize(self._env, device=self._device)
            self._eval_env = ObsNormalize(self._eval_env, device=self._device)
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        self._env = ActionScale(self._env, low=-1.0, high=1.0, device=self._device)
        self._eval_env = ActionScale(self._eval_env, low=-1.0, high=1.0, device=self._device)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)
        self._eval_env = Unsqueeze(self._eval_env, device=self._device)


@registry.register
class CSACLBTemp(SAC):
    """Constrained SAC with smoothed log-barrier loss for temperature control."""

    def _init_env(self) -> None:
        self._env: _CSACLBOffPolicyAdapter = _CSACLBOffPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (
            self._cfgs.algo_cfgs.steps_per_epoch % self._cfgs.train_cfgs.vector_env_nums == 0
        ), "steps_per_epoch must be divisible by vector_env_nums."
        assert (
            int(self._cfgs.train_cfgs.total_steps) % self._cfgs.algo_cfgs.steps_per_epoch == 0
        ), "total_steps must be divisible by steps_per_epoch."
        self._epochs = int(
            self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch,
        )
        self._epoch = 0
        self._steps_per_epoch = (
            self._cfgs.algo_cfgs.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        )
        self._update_cycle = self._cfgs.algo_cfgs.update_cycle
        assert self._steps_per_epoch % self._update_cycle == 0
        self._samples_per_epoch = self._steps_per_epoch // self._update_cycle
        self._update_count = 0

    def _init_model(self) -> None:
        self._cfgs.model_cfgs.critic["num_critics"] = 2
        self._actor_critic = CSACLBActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

    def _init(self) -> None:
        super()._init()
        self._cost_limit = float(self._cfgs.algo_cfgs.cost_limit)
        self._barrier_factor = float(self._cfgs.algo_cfgs.barrier_factor)
        self._barrier_shift = 1.0 / (self._barrier_factor ** 2)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key("Loss/Loss_cost_critic_1", delta=True)
        self._logger.register_key("Loss/Loss_cost_critic_2", delta=True)
        self._logger.register_key("Value/cost_critic_1")
        self._logger.register_key("Value/cost_critic_2")
        self._logger.register_key("Value/cost_critic_max")
        self._logger.register_key("Value/barrier_slack")
        self._logger.register_key("Value/barrier_penalty")

    def _smoothed_log_barrier(self, x: torch.Tensor) -> torch.Tensor:
        mu = self._barrier_factor
        threshold = -1.0 / (mu**2)
        out = torch.empty_like(x)
        log_mask = x <= threshold
        if log_mask.any():
            out[log_mask] = -(1.0 / mu) * torch.log(-x[log_mask])
        if (~log_mask).any():
            const = torch.log(torch.tensor(1.0 / (mu**2), device=x.device, dtype=x.dtype))
            out[~log_mask] = mu * x[~log_mask] - (1.0 / mu) * const + (1.0 / mu)
        return out

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_q1_c, next_q2_c = self._actor_critic.target_cost_critic(next_obs, next_action)
            next_q_c = torch.max(next_q1_c, next_q2_c)
            target_q_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_c

        q1_c, q2_c = self._actor_critic.cost_critic(obs, action)
        loss_1 = nn.functional.mse_loss(q1_c, target_q_c)
        loss_2 = nn.functional.mse_loss(q2_c, target_q_c)
        loss = loss_1 + loss_2

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()
        self._logger.store(
            {
                "Loss/Loss_cost_critic": loss.mean().item(),
                "Loss/Loss_cost_critic_1": loss_1.mean().item(),
                "Loss/Loss_cost_critic_2": loss_2.mean().item(),
                "Value/cost_critic": q1_c.mean().item(),
                "Value/cost_critic_1": q1_c.mean().item(),
                "Value/cost_critic_2": q2_c.mean().item(),
                "Value/cost_critic_max": torch.max(q1_c, q2_c).mean().item(),
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        q1_r, q2_r = self._actor_critic.reward_critic(obs, action)
        reward_term = self._alpha * log_prob - torch.min(q1_r, q2_r)

        q1_c, q2_c = self._actor_critic.cost_critic(obs, action)
        q_c = torch.max(q1_c, q2_c)
        barrier_slack = torch.relu(q_c - self._cost_limit) - self._barrier_shift
        barrier_penalty = self._smoothed_log_barrier(barrier_slack)
        self._logger.store(
            {
                "Value/barrier_slack": barrier_slack.mean().item(),
                "Value/barrier_penalty": barrier_penalty.mean().item(),
            },
        )
        return (reward_term + barrier_penalty).mean()
