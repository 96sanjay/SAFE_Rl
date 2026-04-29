"""Masked reward-only PPO path for the temperature case study."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed

from citylearn_safe.masked_onpolicy_buffer import MaskedVectorOnPolicyBuffer
import citylearn_safe.omni_env_temp_masked_reward  # noqa: F401 - registers env id
import citylearn_safe.omni_env_temp_cooling_only  # noqa: F401 - registers env id
from citylearn_safe.policy_action_mask_temp import (
    CityLearnTempActionBoundsProvider,
    masked_action_from_pretanh,
    masked_log_prob_from_action,
)
from citylearn_safe.ppo_lag_temp_masked import TempMaskedOnPolicyAdapter, _find_policy_mask_hook


@registry.register
class PPOTempMasked(PPO):
    """Reward-only PPO with policy-side action masking for temperature control."""

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
                "PPOTempMasked currently supports vector_env_nums=1 only.",
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

    def _find_teacher_env(self):
        cur = self._env
        seen = set()
        for _ in range(40):
            if cur is None or id(cur) in seen:
                break
            seen.add(id(cur))
            if hasattr(cur, "_rbc_cooling_action"):
                return cur
            for attr in ("unwrapped", "env", "base", "_env", "raw_env"):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
            else:
                break
        raise RuntimeError("Could not find cooling-only env with _rbc_cooling_action() for BC warm-start.")

    def behavior_clone_warmstart(
        self,
        bc_epochs: int,
        bc_rollout_steps: int | None = None,
        bc_batch_size: int = 512,
        seed: int | None = None,
    ) -> None:
        if bc_epochs <= 0:
            return

        teacher_env = self._find_teacher_env()
        hook_source, hook = _find_policy_mask_hook(self._env)
        bounds_provider = hook() if hook is not None else CityLearnTempActionBoundsProvider(hook_source)

        rollout_steps = int(bc_rollout_steps or self._steps_per_epoch)
        obs, _ = self._env.reset(seed=self._seed if seed is None else seed)
        obs_list: list[torch.Tensor] = []
        act_list: list[torch.Tensor] = []
        safe_min_list: list[torch.Tensor] = []
        safe_max_list: list[torch.Tensor] = []

        for _ in range(rollout_steps):
            teacher_action = np.asarray(teacher_env._rbc_cooling_action(), dtype=np.float32).reshape(-1)
            safe_min_np, safe_max_np = bounds_provider.current_safe_bounds()
            obs_list.append(obs.detach().clone().squeeze(0))
            act_list.append(torch.as_tensor(teacher_action, dtype=torch.float32))
            safe_min_list.append(torch.as_tensor(safe_min_np, dtype=torch.float32))
            safe_max_list.append(torch.as_tensor(safe_max_np, dtype=torch.float32))

            env_action = torch.as_tensor(2.0 * teacher_action - 1.0, dtype=torch.float32, device=obs.device).unsqueeze(0)
            obs, _, _, terminated, truncated, _ = self._env.step(env_action)
            if bool(terminated.item()) or bool(truncated.item()):
                obs, _ = self._env.reset(seed=self._seed if seed is None else seed)

        dataset = TensorDataset(
            torch.stack(obs_list),
            torch.stack(act_list),
            torch.stack(safe_min_list),
            torch.stack(safe_max_list),
        )
        dataloader = DataLoader(dataset=dataset, batch_size=bc_batch_size, shuffle=True)
        self._logger.log(
            f"Starting BC warm-start: epochs={bc_epochs}, samples={len(dataset)}, batch_size={bc_batch_size}",
        )

        for epoch in range(bc_epochs):
            losses = []
            for obs_b, act_b, safe_min_b, safe_max_b in dataloader:
                mean = self._actor_critic.actor.predict(obs_b, deterministic=True)
                pred_action = masked_action_from_pretanh(mean, safe_min_b, safe_max_b)
                loss = torch.nn.functional.mse_loss(pred_action, act_b)
                self._actor_critic.actor_optimizer.zero_grad()
                loss.backward()
                if self._cfgs.algo_cfgs.use_max_grad_norm:
                    clip_grad_norm_(
                        self._actor_critic.actor.parameters(),
                        self._cfgs.algo_cfgs.max_grad_norm,
                    )
                distributed.avg_grads(self._actor_critic.actor)
                self._actor_critic.actor_optimizer.step()
                losses.append(loss.detach())
            mean_loss = torch.stack(losses).mean().item() if losses else 0.0
            self._logger.log(f"BC warm-start epoch {epoch + 1}/{bc_epochs}: loss={mean_loss:.6f}")

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
        self._logger.store(
            {
                "Train/Entropy": distribution.entropy().mean().item(),
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
        safe_min: torch.Tensor | None = None,
        safe_max: torch.Tensor | None = None,
    ) -> None:
        loss = self._loss_pi(obs, act, logp, adv_r, safe_min, safe_max)
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
        data = self._buf.get()
        obs = data["obs"]
        act = data["act"]
        logp = data["logp"]
        target_value_r = data["target_value_r"]
        adv_r = data["adv_r"]
        safe_min = data.get("safe_min")
        safe_max = data.get("safe_max")

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        tensors = [obs, act, logp, target_value_r, adv_r]
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
                    obs_b, act_b, logp_b, target_value_r_b, adv_r_b, safe_min_b, safe_max_b = batch
                else:
                    obs_b, act_b, logp_b, target_value_r_b, adv_r_b = batch
                    safe_min_b = safe_max_b = None

                self._update_reward_critic(obs_b, target_value_r_b)
                self._update_actor(obs_b, act_b, logp_b, adv_r_b, safe_min_b, safe_max_b)

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
            },
        )


__all__ = [
    "CityLearnTempActionBoundsProvider",
    "ConstraintActorCritic",
    "Logger",
    "PPOTempMasked",
    "TempMaskedOnPolicyAdapter",
]
