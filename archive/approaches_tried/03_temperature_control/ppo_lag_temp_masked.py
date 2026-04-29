"""Masked PPO-Lag path for the temperature case study.

This module keeps the temperature case separate from the existing EV/V2G
masking stack. The policy samples a latent Gaussian action, squashes it with
``tanh``, and maps it into a temperature-safe action interval before the
environment step and PPO log-prob update.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed

from citylearn_safe.masked_onpolicy_buffer import MaskedVectorOnPolicyBuffer
import citylearn_safe.omni_env_temp_masked  # noqa: F401 - registers env id
from citylearn_safe.omni_env_temp_masked import (
    CityLearnTempMaskedSingleLagCMDP as CityLearnTempMaskedCMDP,
)
from citylearn_safe.policy_action_mask_temp import (
    CityLearnTempActionBoundsProvider,
    masked_action_from_pretanh,
    masked_log_prob_from_action,
)


def _find_policy_mask_hook(env: Any):
    """Return the first env object exposing a policy-action-mask provider hook."""
    cur = env
    seen = set()
    for _ in range(40):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        for name in (
            "make_policy_action_bounds_provider",
            "get_policy_action_bounds_provider",
            "policy_action_bounds_provider",
            "masked_action_bounds_provider",
        ):
            hook = getattr(cur, name, None)
            if callable(hook):
                return cur, hook
        for attr in ("unwrapped", "env", "base", "_env", "raw_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return env, None


class TempMaskedOnPolicyAdapter(OnPolicyAdapter):
    """On-policy adapter that samples and stores masked actions."""

    def _resolve_bounds_provider(self):
        hook_source, hook = _find_policy_mask_hook(self._env)
        if hook is not None:
            provider = hook()
            if provider is not None:
                return provider
        return CityLearnTempActionBoundsProvider(hook_source)

    def rollout(  # pylint: disable=too-many-locals,too-many-branches
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        self._reset_log()

        if self._cfgs.train_cfgs.vector_env_nums != 1:
            raise NotImplementedError(
                "CityLearnTemp masked PPO currently supports vector_env_nums=1 only.",
            )

        bounds_provider = self._resolve_bounds_provider()
        obs, _ = self.reset()
        for step in track(
            range(steps_per_epoch),
            description=f"Processing rollout for epoch: {logger.current_epoch}...",
        ):
            with torch.no_grad():
                value_r = agent.reward_critic(obs)[0]
                value_c = agent.cost_critic(obs)[0]
                dist = agent.actor(obs)
                pre_tanh = dist.rsample()

                safe_min_np, safe_max_np = bounds_provider.current_safe_bounds()
                safe_min = torch.as_tensor(safe_min_np, dtype=torch.float32, device=obs.device).unsqueeze(0)
                safe_max = torch.as_tensor(safe_max_np, dtype=torch.float32, device=obs.device).unsqueeze(0)

                act = masked_action_from_pretanh(pre_tanh, safe_min, safe_max)
                logp = masked_log_prob_from_action(dist, act, safe_min, safe_max)
                # The cooling-only env exposes physical actions in [0, 1], but
                # the adapter wraps it with ActionScale(-1, 1). Convert the
                # masked env-scale action back into wrapper scale before step().
                env_act = 2.0 * act - 1.0

            next_obs, reward, cost, terminated, truncated, info = self.step(env_act)

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({"Value/cost": value_c})
            logger.store({"Value/reward": value_r})

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
                safe_min=safe_min,
                safe_max=safe_max,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)

                    if not done:
                        if epoch_end:
                            logger.log(
                                f"Warning: trajectory cut off when rollout by epoch at {self._ep_len[idx]} steps.",
                            )
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(info["final_observation"][idx])
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)
                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0

                    buffer.finish_path(last_value_r, last_value_c, idx)


@registry.register
class PPOLagTempMasked(PPOLag):
    """Single-constraint PPO-Lag with policy-side action masking for temperature."""

    def _init_env(self) -> None:
        self._env: TempMaskedOnPolicyAdapter = TempMaskedOnPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0
        self._steps_per_epoch = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init(self) -> None:
        super()._init()
        if self._cfgs.train_cfgs.vector_env_nums != 1:
            raise NotImplementedError(
                "PPOLagTempMasked currently supports vector_env_nums=1 only.",
            )
        self._env._policy_action_mask = True
        self._buf = MaskedVectorOnPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
        safe_min: torch.Tensor | None = None,
        safe_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        distribution = self._actor_critic.actor(obs)
        if safe_min is not None and safe_max is not None:
            logp_ = masked_log_prob_from_action(distribution, act, safe_min, safe_max)
        else:
            logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)
        ratio_clipped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )
        loss = -torch.min(ratio * adv, ratio_clipped * adv).mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                "Train/Entropy": entropy,
                "Train/PolicyRatio": ratio,
                "Train/PolicyStd": std,
                "Loss/Loss_pi": loss.mean().item(),
            },
        )
        return loss

    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        safe_min: torch.Tensor | None = None,
        safe_max: torch.Tensor | None = None,
    ) -> None:
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, act, logp, adv, safe_min, safe_max)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

    def _update(self) -> None:
        Jc = self._logger.get_stats("Metrics/EpCost")[0]
        assert not np.isnan(Jc), "cost for updating lagrange multiplier is nan"
        self._lagrange.update_lagrange_multiplier(Jc)

        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data["obs"],
            data["act"],
            data["logp"],
            data["target_value_r"],
            data["target_value_c"],
            data["adv_r"],
            data["adv_c"],
        )
        safe_min = data.get("safe_min")
        safe_max = data.get("safe_max")

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        tensors = [obs, act, logp, target_value_r, target_value_c, adv_r, adv_c]
        has_mask = safe_min is not None and safe_max is not None
        if has_mask:
            tensors.extend([safe_min, safe_max])

        dataloader = DataLoader(
            dataset=TensorDataset(*tensors),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0
        for i in track(range(self._cfgs.algo_cfgs.update_iters), description="Updating..."):
            for batch in dataloader:
                if has_mask:
                    (
                        obs_b,
                        act_b,
                        logp_b,
                        target_value_r_b,
                        target_value_c_b,
                        adv_r_b,
                        adv_c_b,
                        safe_min_b,
                        safe_max_b,
                    ) = batch
                else:
                    (
                        obs_b,
                        act_b,
                        logp_b,
                        target_value_r_b,
                        target_value_c_b,
                        adv_r_b,
                        adv_c_b,
                    ) = batch
                    safe_min_b = safe_max_b = None

                self._update_reward_critic(obs_b, target_value_r_b)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs_b, target_value_c_b)
                self._update_actor(obs_b, act_b, logp_b, adv_r_b, adv_c_b, safe_min_b, safe_max_b)

            new_distribution = self._actor_critic.actor(original_obs)
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)
            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f"Early stopping at iter {i + 1} due to reaching max kl")
                break

        self._logger.store(
            {
                "Train/StopIter": update_counts,
                "Value/Adv": adv_r.mean().item(),
                "Train/KL": final_kl,
                "Metrics/LagrangeMultiplier": self._lagrange.lagrangian_multiplier,
            },
        )

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)


__all__ = [
    "CityLearnTempActionBoundsProvider",
    "CityLearnTempMaskedCMDP",
    "PPOLagTempMasked",
    "TempMaskedOnPolicyAdapter",
]
