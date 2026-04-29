"""SAC-Lag with PID Lagrangian, action masking, and BC warm-start for temperature cooling-only.

Combines:
- SAC's Q-function (state-action evaluation → better prediction of future violations)
- PID Lagrangian (fast, stable lambda adaptation)
- Policy-side action masking (never-block: always allow cooling)
- Twin cost critics with max(Q1_c, Q2_c) (conservative cost estimation)
- BC warm-start from RBC teacher

Inherits from OmniSafe's SACLag, replaces Lagrange with PID, adds masking.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter.offpolicy_adapter import OffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac_lag import SACLag
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
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import Config
from omnisafe.utils.tools import get_device

import citylearn_safe.omni_env_temp_cooling_only  # noqa: F401 — registers env
from citylearn_safe.csac_lb_actor_q_critic import CSACLBActorQCritic
from citylearn_safe.pid_lagrange import PIDLagrange
from citylearn_safe.ppo_lag_temp_masked import _find_policy_mask_hook
from citylearn_safe.policy_action_mask_temp import (
    CityLearnTempActionBoundsProvider,
    masked_action_from_pretanh,
)


def _sigmoid_masked_log_prob(
    dist: torch.distributions.Normal,
    pre_tanh: torch.Tensor,
    safe_min: torch.Tensor,
    safe_max: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Correct log-prob for sigmoid-based action masking.

    masked_action_from_pretanh uses: act = safe_min + sigmoid(x) * span
    where x ~ Normal(mean, std). This computes the exact change-of-variables
    log-prob, unlike masked_log_prob_from_action which uses a tanh inverse.
    """
    u = torch.sigmoid(pre_tanh)  # (0, 1)
    span = (safe_max - safe_min).clamp(min=eps)
    # log p(act) = log p_normal(x) - log(u) - log(1-u) - log(span)
    logp_normal = dist.log_prob(pre_tanh)  # Per dimension
    log_jac = torch.log(u.clamp(min=eps)) + torch.log((1.0 - u).clamp(min=eps))
    log_span = torch.log(span)
    return (logp_normal - log_jac - log_span).sum(dim=-1)


class _SACLagMaskedAdapter(OffPolicyAdapter):
    """Off-policy adapter with masked rollout and separate train/eval envs."""

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

        self._env: CMDP = make(env_id, num_envs=num_envs, device=self._device, **env_cfgs)
        self._eval_env: CMDP = make(env_id, num_envs=1, device=self._device, **env_cfgs)

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

    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ) -> None:
        if self._env.need_time_limit_wrapper:
            assert self._env.max_episode_steps and self._eval_env.max_episode_steps
            self._env = TimeLimit(self._env, time_limit=self._env.max_episode_steps, device=self._device)
            self._eval_env = TimeLimit(self._eval_env, time_limit=self._eval_env.max_episode_steps, device=self._device)
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

    def _resolve_bounds_provider(self):
        hook_source, hook = _find_policy_mask_hook(self._env)
        if hook is not None:
            provider = hook()
            if provider is not None:
                return provider
        return CityLearnTempActionBoundsProvider(hook_source)

    def _resolve_eval_bounds_provider(self):
        hook_source, hook = _find_policy_mask_hook(self._eval_env)
        if hook is not None:
            provider = hook()
            if provider is not None:
                return provider
        return CityLearnTempActionBoundsProvider(hook_source)

    def rollout(
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic,
        buffer: VectorOffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        bounds_provider = self._resolve_bounds_provider()

        for _ in range(rollout_step):
            safe_min_np, safe_max_np = bounds_provider.current_safe_bounds()
            safe_min = torch.as_tensor(safe_min_np, dtype=torch.float32, device=self._device).unsqueeze(0)
            safe_max = torch.as_tensor(safe_max_np, dtype=torch.float32, device=self._device).unsqueeze(0)

            if use_rand_action:
                # Random action within masked bounds
                u = torch.rand(1, self.action_space.shape[0], device=self._device)
                masked_act = safe_min + u * (safe_max - safe_min)
                env_act = 2.0 * masked_act - 1.0
            else:
                with torch.no_grad():
                    dist = agent.actor._distribution(self._current_obs)
                    pre_tanh = dist.rsample()
                    masked_act = masked_action_from_pretanh(pre_tanh, safe_min, safe_max)
                    env_act = 2.0 * masked_act - 1.0

            next_obs, reward, cost, terminated, truncated, info = self.step(env_act)

            self._log_value(reward=reward, cost=cost, info=info)
            real_next_obs = next_obs.clone()
            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    if "final_observation" in info:
                        real_next_obs[idx] = info["final_observation"][idx]
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)

            buffer.store(
                obs=self._current_obs,
                act=env_act,
                reward=reward,
                cost=cost,
                done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
                next_obs=real_next_obs,
                safe_min=safe_min,
                safe_max=safe_max,
            )

            self._current_obs = next_obs

    def eval_policy(
        self,
        episode: int,
        agent: ConstraintActorQCritic,
        logger: Logger,
    ) -> None:
        for _ in range(episode):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
            obs, _ = self._eval_env.reset()
            obs = obs.to(self._device)
            bounds_provider = self._resolve_eval_bounds_provider()

            done = False
            while not done:
                with torch.no_grad():
                    safe_min_np, safe_max_np = bounds_provider.current_safe_bounds()
                    safe_min = torch.as_tensor(safe_min_np, dtype=torch.float32, device=self._device).unsqueeze(0)
                    safe_max = torch.as_tensor(safe_max_np, dtype=torch.float32, device=self._device).unsqueeze(0)
                    dist = agent.actor._distribution(obs)
                    masked_act = masked_action_from_pretanh(dist.mean, safe_min, safe_max)
                    env_act = 2.0 * masked_act - 1.0

                obs, reward, cost, terminated, truncated, info = self._eval_env.step(env_act)
                obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (obs, reward, cost, terminated, truncated)
                )
                ep_ret += info.get("original_reward", reward).cpu()
                ep_cost += info.get("original_cost", cost).cpu()
                ep_len += 1
                done = bool(terminated[0].item()) or bool(truncated[0].item())

            logger.store(
                {
                    "Metrics/TestEpRet": ep_ret,
                    "Metrics/TestEpCost": ep_cost,
                    "Metrics/TestEpLen": ep_len,
                },
            )


@registry.register
class SACLagTempCoolingOnly(SACLag):
    """SAC-Lag with PID Lagrangian, action masking, and twin cost critics."""

    def _init_env(self) -> None:
        self._env: _SACLagMaskedAdapter = _SACLagMaskedAdapter(
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

        # Add mask bounds fields to buffer — safe_max must default to 1.0
        # (not 0.0) so unfilled entries represent full [0,1] range, not collapsed
        act_dim = self._env.action_space.shape[0]
        self._buf.add_field("safe_min", (act_dim,), torch.float32)
        self._buf.add_field("safe_max", (act_dim,), torch.float32)
        self._buf.data["safe_max"].fill_(1.0)

        # PID Lagrangian (replaces standard Lagrange)
        pid_cfgs = getattr(self._cfgs, "pid_lagrange_cfgs", None)
        cost_limit = float(
            getattr(pid_cfgs, "cost_limit", self._cfgs.lagrange_cfgs.cost_limit)
            if pid_cfgs else self._cfgs.lagrange_cfgs.cost_limit
        )
        pid_kp = float(getattr(pid_cfgs, "pid_kp", 1.0)) if pid_cfgs else 1.0
        pid_ki = float(getattr(pid_cfgs, "pid_ki", 0.1)) if pid_cfgs else 0.1
        pid_kd = float(getattr(pid_cfgs, "pid_kd", 0.01)) if pid_cfgs else 0.01
        pid_d_delay = int(getattr(pid_cfgs, "pid_d_delay", 10)) if pid_cfgs else 10
        pid_delta_p_ema_alpha = float(getattr(pid_cfgs, "pid_delta_p_ema_alpha", 0.95)) if pid_cfgs else 0.95
        pid_delta_d_ema_alpha = float(getattr(pid_cfgs, "pid_delta_d_ema_alpha", 0.95)) if pid_cfgs else 0.95
        penalty_max = float(getattr(pid_cfgs, "penalty_max", 100.0)) if pid_cfgs else 100.0
        lagrangian_multiplier_init = float(getattr(pid_cfgs, "lagrangian_multiplier_init", 0.001)) if pid_cfgs else 0.001
        normalize_by_limit = bool(getattr(pid_cfgs, "normalize_by_limit", True)) if pid_cfgs else True

        self._pid_lagrange = PIDLagrange(
            cost_limit=cost_limit,
            pid_kp=pid_kp,
            pid_ki=pid_ki,
            pid_kd=pid_kd,
            pid_d_delay=pid_d_delay,
            pid_delta_p_ema_alpha=pid_delta_p_ema_alpha,
            pid_delta_d_ema_alpha=pid_delta_d_ema_alpha,
            penalty_max=penalty_max,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            normalize_by_limit=normalize_by_limit,
        )
        self._last_pid_epoch = -1

        print(
            f"[PIDLag] Kp={pid_kp}, Ki={pid_ki}, Kd={pid_kd}, "
            f"cost_limit={cost_limit}, penalty_max={penalty_max}, "
            f"normalize={normalize_by_limit}"
        )

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key("Metrics/PID_Lambda", min_and_max=True)
        self._logger.register_key("Metrics/PID_I")
        self._logger.register_key("Metrics/PID_P")

    # ---- BC Warm-Start ----

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
        obs_list, act_list, safe_min_list, safe_max_list = [], [], [], []

        for _ in range(rollout_steps):
            teacher_action = np.asarray(teacher_env._rbc_cooling_action(), dtype=np.float32).reshape(-1)
            safe_min_np, safe_max_np = bounds_provider.current_safe_bounds()
            obs_list.append(obs.detach().clone().squeeze(0))
            act_list.append(torch.as_tensor(teacher_action, dtype=torch.float32))
            safe_min_list.append(torch.as_tensor(safe_min_np, dtype=torch.float32))
            safe_max_list.append(torch.as_tensor(safe_max_np, dtype=torch.float32))

            env_action = torch.as_tensor(
                2.0 * teacher_action - 1.0, dtype=torch.float32, device=obs.device
            ).unsqueeze(0)
            obs, _, _, terminated, truncated, _ = self._env.step(env_action)
            if bool(terminated.item()) or bool(truncated.item()):
                obs, _ = self._env.reset(seed=self._seed if seed is None else seed)

        dataset = TensorDataset(
            torch.stack(obs_list), torch.stack(act_list),
            torch.stack(safe_min_list), torch.stack(safe_max_list),
        )
        dataloader = DataLoader(dataset=dataset, batch_size=bc_batch_size, shuffle=True)
        self._logger.log(
            f"Starting BC warm-start: epochs={bc_epochs}, "
            f"samples={len(dataset)}, batch_size={bc_batch_size}",
        )

        for epoch in range(bc_epochs):
            losses = []
            for obs_b, act_b, safe_min_b, safe_max_b in dataloader:
                # Use raw mean (pre-tanh) for masking
                dist = self._actor_critic.actor._distribution(obs_b)
                pred_action = masked_action_from_pretanh(dist.mean, safe_min_b, safe_max_b)
                loss = torch.nn.functional.mse_loss(pred_action, act_b)
                self._actor_critic.actor_optimizer.zero_grad()
                loss.backward()
                if self._cfgs.algo_cfgs.max_grad_norm:
                    clip_grad_norm_(
                        self._actor_critic.actor.parameters(),
                        self._cfgs.algo_cfgs.max_grad_norm,
                    )
                self._actor_critic.actor_optimizer.step()
                losses.append(loss.detach())
            mean_loss = torch.stack(losses).mean().item() if losses else 0.0
            self._logger.log(f"BC warm-start epoch {epoch + 1}/{bc_epochs}: loss={mean_loss:.6f}")

    # ---- Actor Loss with Masking ----

    def _loss_pi(
        self,
        obs: torch.Tensor,
        safe_min: torch.Tensor | None = None,
        safe_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dist = self._actor_critic.actor._distribution(obs)
        pre_tanh = dist.rsample()

        if safe_min is not None and safe_max is not None:
            # Ensure span is never zero (collapsed bounds → fallback to full [0,1])
            span = safe_max - safe_min
            collapsed = (span < 1e-4).all(dim=-1, keepdim=True)
            safe_min_c = torch.where(collapsed, torch.zeros_like(safe_min), safe_min)
            safe_max_c = torch.where(collapsed, torch.ones_like(safe_max), safe_max)

            masked_act = masked_action_from_pretanh(pre_tanh, safe_min_c, safe_max_c)
            log_prob = _sigmoid_masked_log_prob(dist, pre_tanh, safe_min_c, safe_max_c)
            action = 2.0 * masked_act - 1.0  # Convert to [-1,1] for Q-critics
        else:
            action = torch.tanh(pre_tanh)
            self._actor_critic.actor._current_raw_action = pre_tanh
            self._actor_critic.actor._current_dist = dist
            self._actor_critic.actor._after_inference = True
            log_prob = self._actor_critic.actor.log_prob(action)

        q1_r, q2_r = self._actor_critic.reward_critic(obs, action)
        loss_r = self._alpha * log_prob - torch.min(q1_r, q2_r)

        q1_c, q2_c = self._actor_critic.cost_critic(obs, action)
        q_c = torch.max(q1_c, q2_c)  # Conservative cost estimate
        pid_lambda = self._pid_lagrange.lagrangian_multiplier
        loss_c = pid_lambda * q_c

        return (loss_r + loss_c).mean() / (1.0 + pid_lambda)

    def _update_actor(self, obs: torch.Tensor, safe_min: torch.Tensor = None, safe_max: torch.Tensor = None) -> None:
        loss = self._loss_pi(obs, safe_min, safe_max)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.actor_optimizer.step()
        self._logger.store({"Loss/Loss_pi": loss.mean().item()})

        # Auto-alpha update
        if self._cfgs.algo_cfgs.auto_alpha:
            with torch.no_grad():
                dist = self._actor_critic.actor._distribution(obs)
                pre_tanh = dist.rsample()
                if safe_min is not None and safe_max is not None:
                    span = safe_max - safe_min
                    collapsed = (span < 1e-4).all(dim=-1, keepdim=True)
                    sm = torch.where(collapsed, torch.zeros_like(safe_min), safe_min)
                    sx = torch.where(collapsed, torch.ones_like(safe_max), safe_max)
                    log_prob = _sigmoid_masked_log_prob(dist, pre_tanh, sm, sx)
                else:
                    action = torch.tanh(pre_tanh)
                    self._actor_critic.actor._current_raw_action = pre_tanh
                    self._actor_critic.actor._current_dist = dist
                    self._actor_critic.actor._after_inference = True
                    log_prob = self._actor_critic.actor.log_prob(action)

            alpha_loss = -self._log_alpha * (log_prob + self._target_entropy).mean()
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
            self._logger.store({"Loss/alpha_loss": alpha_loss.mean().item()})

        self._logger.store({"Value/alpha": self._alpha})

    # ---- Twin Cost Critic (conservative) ----

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
            next_q_c = torch.max(next_q1_c, next_q2_c)  # Conservative target
            target_q_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_c

        q1_c, q2_c = self._actor_critic.cost_critic(obs, action)
        loss = nn.functional.mse_loss(q1_c, target_q_c) + nn.functional.mse_loss(q2_c, target_q_c)

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
                "Value/cost_critic": torch.max(q1_c, q2_c).mean().item(),
            },
        )

    # ---- Update Loop ----

    def _update(self) -> None:
        # Standard SAC update loop (from DDPG._update)
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data["obs"], data["act"], data["reward"],
                data["cost"], data["done"], data["next_obs"],
            )
            safe_min = data.get("safe_min")
            safe_max = data.get("safe_max")

            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs, safe_min, safe_max)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

        # PID lambda update (once per epoch)
        if self._epoch != self._last_pid_epoch:
            Jc = self._logger.get_stats("Metrics/EpCost")[0]
            if not np.isnan(Jc):
                self._pid_lagrange.pid_update(Jc)
            self._last_pid_epoch = self._epoch

        pid_lambda = self._pid_lagrange.lagrangian_multiplier
        self._logger.store(
            {
                "Metrics/LagrangeMultiplier": pid_lambda,
                "Metrics/PID_Lambda": pid_lambda,
                "Metrics/PID_I": self._pid_lagrange._pid_i,
                "Metrics/PID_P": self._pid_lagrange._delta_p,
            },
        )

    def _log_when_not_update(self) -> None:
        super()._log_when_not_update()
        self._logger.store(
            {
                "Metrics/PID_Lambda": 0.0,
                "Metrics/PID_I": 0.0,
                "Metrics/PID_P": 0.0,
            },
        )


__all__ = ["SACLagTempCoolingOnly"]
