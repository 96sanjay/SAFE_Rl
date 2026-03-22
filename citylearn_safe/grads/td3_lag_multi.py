"""TD3LagMulti: TD3 with per-constraint Lagrange multipliers and DiffProjector.

Extends OmniSafe's TD3 (NOT TD3Lag) with:
  - Per-constraint cost Q-critics (C0: EV departure, C1: EV dense) with PID lambda updates
  - DiffProjector integration (differentiable safety projection for C2/C3/C4 hard constraints)
  - Penalty critic Q_pen (Markgraf et al. 2025, Eq. 29-30)

The SP-RL actor loss (Eq. 30, corrected sign) is:

    u       = pi_theta(s)                       # unsafe action from actor
    u_phi   = Phi(s, u)                         # projected safe action (differentiable)
    h       = w_pen * ||u - u_phi||^2           # projection penalty
    L_actor = (-min(Q1_r, Q2_r)(s, u_phi)       # reward (through projector)
               + lam_0 * Q_C0(s, u_phi)         # C0 cost (through projector)
               + lam_1 * Q_C1(s, u_phi)         # C1 cost (through projector)
               + Q_pen(s, u)                     # penalty (NOT through projector)
              ) / (1 + lam_0 + lam_1)

Critic targets use TD3-style target noise smoothing on the projected target action
for reward/cost critics, and on the raw target action for the penalty critic.

Replay buffer stores: (obs, act_unsafe, act_safe, reward, cost_c0, cost_c1,
                        penalty_h, next_obs, done)

References:
  - Markgraf et al. 2025, "Safety Projection for Reinforcement Learning", Eq. 29-30
  - Fujimoto et al. 2018, "Addressing Function Approximation Error in Actor-Critic Methods"
  - Stooke et al. 2020, "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods"
"""
from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.adapter import OffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.td3 import TD3
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.lagrange import Lagrange
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.models.critic.critic_builder import CriticBuilder

from citylearn_safe.pid_lagrange import PIDLagrange

# ---------------------------------------------------------------------------
# Per-constraint cost keys (from safety_env_v3 info dict).
# TD3LagMulti handles C0 and C1 via Lagrangian; C2/C3/C4 are handled by the
# DiffProjector as hard constraints.
# ---------------------------------------------------------------------------
COST_KEYS = [
    'cost_ev_departure',          # C0: EV departure SoC deficit (sparse)
    'cost_ev_dense',              # C1: EV charging incentive (dense)
]
NUM_COSTS = len(COST_KEYS)


# ===========================================================================
# Custom adapter: tracks per-constraint costs + stores (u_unsafe, u_safe, h)
# ===========================================================================
class _SPRLOffPolicyAdapter(OffPolicyAdapter):
    """Off-policy adapter that tracks per-constraint costs and safe/unsafe actions.

    During rollout the environment step uses the *safe* (projected) action, but
    the buffer stores both the raw actor output (u_unsafe) and the projected
    output (u_safe) plus the penalty h = w * ||u - u_phi||^2.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._per_ep_costs: list[float] = [0.0] * NUM_COSTS
        self._last_completed_ep_costs: list[float] = [0.0] * NUM_COSTS

    def get_per_constraint_ep_cost(self, idx: int) -> float:
        """Return the per-constraint cost from the last completed episode."""
        return self._last_completed_ep_costs[idx]

    def rollout(
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic,
        buffer: VectorOffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool,
        projector: Any = None,
        penalty_w: float = 1.0,
    ) -> None:
        """Off-policy rollout with DiffProjector integration.

        For each step:
          1. Get raw action u from actor (or random)
          2. Project u -> u_phi via DiffProjector (if available)
          3. Step environment with u_phi
          4. Store (obs, u, u_phi, reward, cost_c0, cost_c1, h, next_obs, done)
        """
        for _ in range(rollout_step):
            if use_rand_action:
                act_unsafe = (
                    torch.rand(self.action_space.shape) * 2 - 1
                ).unsqueeze(0).to(self._device)
            else:
                act_unsafe = agent.step(self._current_obs, deterministic=False)

            # Project through DiffProjector (if available)
            if projector is not None:
                with torch.no_grad():
                    act_safe, _ = projector.project(self._current_obs, act_unsafe)
            else:
                act_safe = act_unsafe.clone()

            # Compute penalty: h = w * ||u - u_phi||^2
            penalty_h = penalty_w * (act_unsafe - act_safe).pow(2).sum(dim=-1, keepdim=True)

            # Step environment with the SAFE action
            next_obs, reward, cost, terminated, truncated, info = self.step(act_safe)

            self._log_value(reward=reward, cost=cost, info=info)
            real_next_obs = next_obs.clone()

            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    if 'final_observation' in info:
                        real_next_obs[idx] = info['final_observation'][idx]
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)
                    self._last_completed_ep_costs = list(self._per_ep_costs)
                    self._per_ep_costs = [0.0] * NUM_COSTS

            # Extract per-constraint costs from info dict
            cost_info = (
                info.get('final_info', info)
                if info.get('final_info') is not None
                else info
            )

            per_cost_vals = {}
            for i, key in enumerate(COST_KEYS):
                val = cost_info.get(key, 0.0)
                if isinstance(val, torch.Tensor):
                    val = val.item()
                per_cost_vals[f'cost_{i}'] = torch.tensor(
                    [float(val)], dtype=torch.float32
                )
                self._per_ep_costs[i] += float(val)

            # Store in buffer: standard fields + per-constraint costs
            # + safe action + penalty. The 'act' field stores the UNSAFE action
            # (needed for penalty critic). We store safe action separately.
            buffer.store(
                obs=self._current_obs,
                act=act_unsafe,
                reward=reward,
                cost=cost,
                done=torch.logical_and(
                    terminated, torch.logical_xor(terminated, truncated)
                ),
                next_obs=real_next_obs,
                act_safe=act_safe,
                penalty_h=penalty_h,
                **per_cost_vals,
            )

            self._current_obs = next_obs


# ===========================================================================
# Custom buffer: stores per-constraint costs + safe action + penalty
# ===========================================================================
class _SPRLOffPolicyBuffer(VectorOffPolicyBuffer):
    """Replay buffer for SP-RL: stores (obs, act_unsafe, act_safe, reward,
    cost_c0, cost_c1, penalty_h, next_obs, done).

    The standard 'act' field holds the unsafe action. Additional fields:
      - act_safe: projected safe action (used for reward/cost critic training)
      - penalty_h: projection penalty (used for penalty critic training)
      - cost_0, cost_1: per-constraint costs
    """

    def __init__(
        self,
        obs_space,
        act_space,
        size: int,
        batch_size: int,
        num_envs: int,
        device: torch.device,
    ) -> None:
        super().__init__(
            obs_space=obs_space,
            act_space=act_space,
            size=size,
            batch_size=batch_size,
            num_envs=num_envs,
            device=device,
        )
        act_dim = act_space.shape[0]
        # Safe action buffer (same shape as act)
        self.data['act_safe'] = torch.zeros(
            (size, num_envs, act_dim), dtype=torch.float32, device=device
        )
        # Penalty buffer (scalar per sample)
        self.data['penalty_h'] = torch.zeros(
            (size, num_envs), dtype=torch.float32, device=device
        )
        # Per-constraint cost buffers
        for i in range(NUM_COSTS):
            self.data[f'cost_{i}'] = torch.zeros(
                (size, num_envs), dtype=torch.float32, device=device
            )


# ===========================================================================
# TD3LagMulti algorithm
# ===========================================================================
@registry.register
class TD3LagMulti(TD3):
    """TD3 with per-constraint Lagrange multipliers and DiffProjector.

    Extends OmniSafe's TD3 with:
      1. Per-constraint cost Q-critics (C0, C1) with PID lambda updates
      2. DiffProjector (differentiable safety layer for C2/C3/C4 hard constraints)
      3. Penalty critic Q_pen (Markgraf et al. 2025)

    Inherits from TD3 (which inherits from DDPG -> BaseAlgo).
    We handle Lagrangian ourselves rather than inheriting TD3Lag.
    """

    def _init_env(self) -> None:
        """Create SP-RL off-policy adapter."""
        self._env: _SPRLOffPolicyAdapter = _SPRLOffPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (
            self._cfgs.algo_cfgs.steps_per_epoch
            % self._cfgs.train_cfgs.vector_env_nums
            == 0
        ), 'steps_per_epoch must be divisible by vector_env_nums.'
        assert (
            int(self._cfgs.train_cfgs.total_steps)
            % self._cfgs.algo_cfgs.steps_per_epoch
            == 0
        ), 'total_steps must be divisible by steps_per_epoch.'

        self._epochs: int = int(
            self._cfgs.train_cfgs.total_steps
            // self._cfgs.algo_cfgs.steps_per_epoch
        )
        self._epoch: int = 0
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // self._cfgs.train_cfgs.vector_env_nums
        )
        self._update_cycle: int = self._cfgs.algo_cfgs.update_cycle
        assert (
            self._steps_per_epoch % self._update_cycle == 0
        ), 'steps_per_epoch must be divisible by update_cycle.'
        self._samples_per_epoch: int = self._steps_per_epoch // self._update_cycle
        self._update_count: int = 0

    def _init_model(self) -> None:
        """Build standard TD3 actor-critic plus per-constraint cost Q-critics
        and penalty critic."""
        # TD3._init_model sets num_critics=2 and builds ConstraintActorQCritic
        super()._init_model()

        obs_space = self._env.observation_space
        act_space = self._env.action_space
        model_cfgs = self._cfgs.model_cfgs

        # --- Per-constraint cost Q-critics (C0, C1) ---
        # Each is a single Q(s,a) network with its own target
        self._cost_q_critics = nn.ModuleList()
        self._cost_q_critic_targets = nn.ModuleList()
        self._cost_q_critic_optimizers = []

        for i in range(NUM_COSTS):
            critic = CriticBuilder(
                obs_space=obs_space,
                act_space=act_space,
                hidden_sizes=model_cfgs.critic.hidden_sizes,
                activation=model_cfgs.critic.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
                num_critics=1,
                use_obs_encoder=False,
            ).build_critic('q').to(self._device)

            target_critic = deepcopy(critic)
            for param in target_critic.parameters():
                param.requires_grad = False

            self._cost_q_critics.append(critic)
            self._cost_q_critic_targets.append(target_critic)
            self._cost_q_critic_optimizers.append(
                torch.optim.Adam(critic.parameters(), lr=model_cfgs.critic.lr)
            )

        # --- Penalty critic Q_pen(s, u_unsafe) ---
        # Learns to predict cumulative projection penalty (Eq. 29)
        proj_cfgs = getattr(self._cfgs, 'projector_cfgs', None)
        pen_lr = float(getattr(proj_cfgs, 'penalty_critic_lr', model_cfgs.critic.lr))

        self._penalty_critic = CriticBuilder(
            obs_space=obs_space,
            act_space=act_space,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
        ).build_critic('q').to(self._device)

        self._penalty_critic_target = deepcopy(self._penalty_critic)
        for param in self._penalty_critic_target.parameters():
            param.requires_grad = False

        self._penalty_critic_optimizer = torch.optim.Adam(
            self._penalty_critic.parameters(), lr=pen_lr
        )

        print(f"[TD3LagMulti] {NUM_COSTS} per-constraint cost Q-critics + 1 penalty critic")

    def _init(self) -> None:
        """Initialize replay buffer, DiffProjector, Lagrange multipliers."""
        # --- SP-RL replay buffer ---
        self._buf: _SPRLOffPolicyBuffer = _SPRLOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._cfgs.algo_cfgs.size,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

        # --- DiffProjector (built externally via agent._projector.build()) ---
        # Imported lazily; the training script must set self._projector before learn()
        self._projector = None

        # --- Projector config ---
        proj_cfgs = getattr(self._cfgs, 'projector_cfgs', None)
        self._penalty_w = float(getattr(proj_cfgs, 'penalty_w', 1.0))

        # --- Parent's single Lagrangian (for OmniSafe compatibility/logging) ---
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

        # --- Per-constraint PID Lagrange multipliers (C0, C1) ---
        multi_cfgs = getattr(self._cfgs, 'multi_cfgs', None)
        assert multi_cfgs is not None, (
            "multi_cfgs not found in config. TD3LagMulti requires per-constraint "
            "cost limits under multi_cfgs."
        )

        self._tau = float(getattr(multi_cfgs, 'tau', 1.0))

        self._per_cost_limits = [
            float(getattr(multi_cfgs, f'cost_limit_{i}', 5000.0))
            for i in range(NUM_COSTS)
        ]

        # PID gains (per-constraint overrides supported)
        pid_kp_default = float(getattr(multi_cfgs, 'pid_kp', 0.1))
        pid_ki_default = float(getattr(multi_cfgs, 'pid_ki', 0.01))
        pid_kd_default = float(getattr(multi_cfgs, 'pid_kd', 0.01))
        pid_d_delay = int(getattr(multi_cfgs, 'pid_d_delay', 10))
        pid_ema_p_default = float(getattr(multi_cfgs, 'pid_delta_p_ema_alpha', 0.95))
        pid_ema_d_default = float(getattr(multi_cfgs, 'pid_delta_d_ema_alpha', 0.95))

        lag_cfgs = self._cfgs.lagrange_cfgs
        penalty_max = float(getattr(lag_cfgs, 'lagrangian_upper_bound', 3.0))
        init_val = float(getattr(lag_cfgs, 'lagrangian_multiplier_init', 0.001))

        self._per_lagranges: list[PIDLagrange] = []
        for i in range(NUM_COSTS):
            kp = float(getattr(multi_cfgs, f'pid_kp_{i}', pid_kp_default))
            ki = float(getattr(multi_cfgs, f'pid_ki_{i}', pid_ki_default))
            kd = float(getattr(multi_cfgs, f'pid_kd_{i}', pid_kd_default))
            ema_p = float(getattr(multi_cfgs, f'pid_ema_p_{i}', pid_ema_p_default))
            ema_d = float(getattr(multi_cfgs, f'pid_ema_d_{i}', pid_ema_d_default))
            self._per_lagranges.append(
                PIDLagrange(
                    cost_limit=self._per_cost_limits[i],
                    pid_kp=kp,
                    pid_ki=ki,
                    pid_kd=kd,
                    pid_d_delay=pid_d_delay,
                    pid_delta_p_ema_alpha=ema_p,
                    pid_delta_d_ema_alpha=ema_d,
                    penalty_max=penalty_max,
                    lagrangian_multiplier_init=init_val,
                )
            )
            print(
                f"[TD3LagMulti] PID C{i}: Kp={kp}, Ki={ki}, Kd={kd}, "
                f"limit={self._per_cost_limits[i]:.0f}"
            )

        print(f"[TD3LagMulti] penalty_w={self._penalty_w}, tau={self._tau}")

    def _get_lambda(self, idx: int) -> float:
        """Get lambda value from PIDLagrange."""
        lam = self._per_lagranges[idx].lagrangian_multiplier
        return float(lam) if isinstance(lam, (int, float)) else lam.item()

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    def _init_log(self) -> None:
        """Register TD3LagMulti-specific log keys."""
        super()._init_log()

        # Per-constraint cost critic metrics
        for i in range(NUM_COSTS):
            self._logger.register_key(f'Loss/Loss_cost_critic_{i}')
            self._logger.register_key(f'Value/cost_critic_{i}')
            self._logger.register_key(f'Metrics/EpCost_{i}')
            self._logger.register_key(f'Metrics/Lambda_{i}')

        # Penalty critic metrics
        self._logger.register_key('Loss/Loss_penalty_critic')
        self._logger.register_key('Value/penalty_critic')

        # Lagrange compatibility
        self._logger.register_key('Metrics/LagrangeMultiplier')

        # SP-RL diagnostics
        self._logger.register_key('SPRL/RewardLoss')
        self._logger.register_key('SPRL/CostLoss')
        self._logger.register_key('SPRL/PenaltyLoss')
        self._logger.register_key('SPRL/ProjectionNorm')
        self._logger.register_key('SPRL/Lambda0')
        self._logger.register_key('SPRL/Lambda1')

    # -----------------------------------------------------------------------
    # Actor loss (Eq. 30, Markgraf et al. 2025)
    # -----------------------------------------------------------------------
    def _loss_pi(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute actor loss with DiffProjector and penalty critic.

        Eq. 30:
            u       = pi_theta(s)
            u_phi   = Phi(s, u)           (differentiable projection)
            L = -min(Q1_r, Q2_r)(s, u_phi)   # reward (through projector)
                + lam_0 * Q_C0(s, u)          # C0 cost (NOT through projector)
                + lam_1 * Q_C1(s, u)          # C1 cost (NOT through projector)
                + Q_pen(s, u)                  # penalty (NOT through projector)

        The reward critic receives the projected action (gradients flow through
        the projector back to the actor). The cost and penalty critics receive
        the raw action (gradients bypass the projector) to avoid action aliasing.
        """
        # 1. Raw action from actor
        u = self._actor_critic.actor.predict(obs, deterministic=True)

        # 2. Project through DiffProjector (differentiable)
        if self._projector is not None:
            u_phi, proj_info = self._projector.project(obs, u)
        else:
            # Fallback: no projection (for testing without projector)
            u_phi = u
            proj_info = {}

        # 3. Reward Q-values (through projector)
        q1_r, q2_r = self._actor_critic.reward_critic(obs, u_phi)
        loss_reward = -torch.min(q1_r, q2_r).mean()

        # 4. Per-constraint cost Q-values (NOT through projector -- uses raw u)
        # C0/C1 are Lagrangian constraints, not projection constraints.
        # Evaluating through the projector would cause action aliasing on the
        # cost critics (same issue as PSF).
        lam_0 = self._get_lambda(0)
        lam_1 = self._get_lambda(1)
        q_c0 = self._cost_q_critics[0](obs, u)[0]
        q_c1 = self._cost_q_critics[1](obs, u)[0]
        loss_c0 = lam_0 * q_c0.mean()
        loss_c1 = lam_1 * q_c1.mean()

        # 5. Penalty critic (NOT through projector -- uses raw u)
        # This penalizes the actor for producing actions that require large
        # projection corrections, encouraging the actor to learn safe behavior
        q_pen = self._penalty_critic(obs, u)[0]
        loss_pen = q_pen.mean()

        # 6. Combined loss (Eq. 30, no normalization -- paper has no denominator)
        total_loss = loss_reward + loss_c0 + loss_c1 + loss_pen

        # Log diagnostics
        proj_norm = (u - u_phi).pow(2).sum(dim=-1).mean().item()
        self._logger.store({
            'SPRL/RewardLoss': loss_reward.item(),
            'SPRL/CostLoss': (loss_c0 + loss_c1).item(),
            'SPRL/PenaltyLoss': loss_pen.item(),
            'SPRL/ProjectionNorm': proj_norm,
            'SPRL/Lambda0': lam_0,
            'SPRL/Lambda1': lam_1,
        })

        return total_loss

    # -----------------------------------------------------------------------
    # Critic updates
    # -----------------------------------------------------------------------
    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update twin reward critics using TD3-style target noise smoothing.

        Uses the SAFE action for both current Q-value and target computation.
        Target action is the projected target actor output with clipped noise.
        """
        with torch.no_grad():
            # Target action from target actor
            next_action_raw = self._actor_critic.target_actor.predict(
                next_obs, deterministic=True
            )
            # TD3 target noise smoothing
            policy_noise = self._cfgs.algo_cfgs.policy_noise
            policy_noise_clip = self._cfgs.algo_cfgs.policy_noise_clip
            noise = (torch.randn_like(next_action_raw) * policy_noise).clamp(
                -policy_noise_clip, policy_noise_clip
            )
            next_action_noisy = (next_action_raw + noise).clamp(-1.0, 1.0)

            # Project target action through DiffProjector
            if self._projector is not None:
                next_action_safe, _ = self._projector.project(
                    next_obs, next_action_noisy
                )
            else:
                next_action_safe = next_action_noisy

            # Twin target Q-values
            next_q1_r, next_q2_r = self._actor_critic.target_reward_critic(
                next_obs, next_action_safe
            )
            next_q_r = torch.min(next_q1_r, next_q2_r)
            target_q_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_r

        # Current Q-values (use safe action stored in buffer)
        q1_r, q2_r = self._actor_critic.reward_critic(obs, action)
        loss = F.mse_loss(q1_r, target_q_r) + F.mse_loss(q2_r, target_q_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()

        self._logger.store({
            'Loss/Loss_reward_critic': loss.mean().item(),
            'Value/reward_critic': q1_r.mean().item(),
        })

    def _update_per_cost_critic(
        self,
        obs: torch.Tensor,
        act_unsafe: torch.Tensor,
        cost_i: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
        idx: int,
    ) -> None:
        """Update a per-constraint cost Q-critic via TD learning.

        Uses the UNSAFE (raw) action for both current and target Q-values.
        C0/C1 are Lagrangian constraints -- their critics must see raw actions
        to avoid action aliasing through the projector.
        Target action uses TD3-style noise smoothing on the raw target action.
        """
        with torch.no_grad():
            next_action_raw = self._actor_critic.target_actor.predict(
                next_obs, deterministic=True
            )
            # TD3 target noise smoothing (no projection for cost critics)
            policy_noise = self._cfgs.algo_cfgs.policy_noise
            policy_noise_clip = self._cfgs.algo_cfgs.policy_noise_clip
            noise = (torch.randn_like(next_action_raw) * policy_noise).clamp(
                -policy_noise_clip, policy_noise_clip
            )
            next_action_noisy = (next_action_raw + noise).clamp(-1.0, 1.0)

            next_q_c = self._cost_q_critic_targets[idx](
                next_obs, next_action_noisy
            )[0]
            target_q_c = (
                cost_i + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_c
            )

        q_c = self._cost_q_critics[idx](obs, act_unsafe)[0]
        loss = F.mse_loss(q_c, target_q_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._cost_q_critics[idx].parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._cost_q_critic_optimizers[idx].zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._cost_q_critics[idx].parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._cost_q_critic_optimizers[idx].step()

        self._logger.store({
            f'Loss/Loss_cost_critic_{idx}': loss.mean().item(),
            f'Value/cost_critic_{idx}': q_c.mean().item(),
        })

    def _update_penalty_critic(
        self,
        obs: torch.Tensor,
        act_unsafe: torch.Tensor,
        penalty_h: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update penalty critic Q_pen(s, u_unsafe) via TD learning (Eq. 29).

        The penalty critic learns to predict cumulative projection penalty:
            Q_pen(s, u) = h(s, u) + gamma * Q_pen(s', u')

        Uses the RAW (unsafe) target action (NOT projected), because the penalty
        critic evaluates the actor's raw output to measure how far it is from
        being safe.
        """
        with torch.no_grad():
            next_action_raw = self._actor_critic.target_actor.predict(
                next_obs, deterministic=True
            )
            # TD3 target noise on raw action (no projection for penalty critic)
            policy_noise = self._cfgs.algo_cfgs.policy_noise
            policy_noise_clip = self._cfgs.algo_cfgs.policy_noise_clip
            noise = (torch.randn_like(next_action_raw) * policy_noise).clamp(
                -policy_noise_clip, policy_noise_clip
            )
            next_action_noisy = (next_action_raw + noise).clamp(-1.0, 1.0)

            next_q_pen = self._penalty_critic_target(
                next_obs, next_action_noisy
            )[0]
            target_q_pen = (
                penalty_h + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_pen
            )

        q_pen = self._penalty_critic(obs, act_unsafe)[0]
        loss = F.mse_loss(q_pen, target_q_pen)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._penalty_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._penalty_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._penalty_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._penalty_critic_optimizer.step()

        self._logger.store({
            'Loss/Loss_penalty_critic': loss.mean().item(),
            'Value/penalty_critic': q_pen.mean().item(),
        })

    # -----------------------------------------------------------------------
    # Main update loop
    # -----------------------------------------------------------------------
    def _update(self) -> None:
        """Update all critics, actor, and target networks.

        Per update iteration:
          1. Sample batch from replay buffer
          2. Update twin reward critics on (obs, act_safe, reward)
          3. Update per-constraint cost critics on (obs, act_safe, cost_i)
          4. Update penalty critic on (obs, act_unsafe, penalty_h)
          5. (Every policy_delay steps) Update actor and Polyak-update all targets
        """
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1

            obs = data['obs']
            act_unsafe = data['act']          # raw actor output
            act_safe = data['act_safe']       # projected safe action
            reward = data['reward']
            cost = data['cost']               # aggregate cost (for compatibility)
            done = data['done']
            next_obs = data['next_obs']
            penalty_h = data['penalty_h']

            # 1. Reward critics: trained on (obs, act_safe, reward)
            self._update_reward_critic(obs, act_safe, reward, done, next_obs)

            # 2. Standard cost critic (OmniSafe compatibility, uses aggregate cost)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act_safe, cost, done, next_obs)

            # 3. Per-constraint cost critics: trained on (obs, act_unsafe, cost_i)
            #    C0/C1 are Lagrangian constraints -- critics see raw actions
            for i in range(NUM_COSTS):
                cost_i = data[f'cost_{i}']
                self._update_per_cost_critic(
                    obs, act_unsafe, cost_i, done, next_obs, i
                )

            # 4. Penalty critic: trained on (obs, act_unsafe, penalty_h)
            self._update_penalty_critic(
                obs, act_unsafe, penalty_h, done, next_obs
            )

            # 5. Actor + Polyak updates (every policy_delay steps)
            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)

                # Polyak update for standard actor-critic targets
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

                # Polyak update for per-constraint cost critic targets
                tau = self._cfgs.algo_cfgs.polyak
                for i in range(NUM_COSTS):
                    for target_p, p in zip(
                        self._cost_q_critic_targets[i].parameters(),
                        self._cost_q_critics[i].parameters(),
                    ):
                        target_p.data.copy_(
                            tau * p.data + (1 - tau) * target_p.data
                        )

                # Polyak update for penalty critic target
                for target_p, p in zip(
                    self._penalty_critic_target.parameters(),
                    self._penalty_critic.parameters(),
                ):
                    target_p.data.copy_(
                        tau * p.data + (1 - tau) * target_p.data
                    )

    def _update_epoch(self) -> None:
        """Per-epoch: update PID Lagrange multipliers from last completed episode costs."""
        warmup = int(getattr(self._cfgs.algo_cfgs, 'warmup_epochs', 5))

        for i in range(NUM_COSTS):
            ep_cost_i = self._env.get_per_constraint_ep_cost(i)
            if self._epoch >= warmup:
                self._per_lagranges[i].pid_update(ep_cost_i)
            self._logger.store({
                f'Metrics/EpCost_{i}': ep_cost_i,
                f'Metrics/Lambda_{i}': self._get_lambda(i),
            })

        # Parent's single lambda (for OmniSafe logging compatibility)
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if self._epoch >= warmup:
            self._lagrange.update_lagrange_multiplier(Jc)
        self._logger.store({
            'Metrics/LagrangeMultiplier': (
                self._lagrange.lagrangian_multiplier.data.item()
            ),
        })

    def _log_when_not_update(self) -> None:
        """Log default values during initial random exploration (before learning starts)."""
        super()._log_when_not_update()

        for i in range(NUM_COSTS):
            self._logger.store({
                f'Loss/Loss_cost_critic_{i}': 0.0,
                f'Value/cost_critic_{i}': 0.0,
                f'Metrics/EpCost_{i}': 0.0,
                f'Metrics/Lambda_{i}': self._get_lambda(i),
            })
        self._logger.store({
            'Loss/Loss_penalty_critic': 0.0,
            'Value/penalty_critic': 0.0,
            'Metrics/LagrangeMultiplier': (
                self._lagrange.lagrangian_multiplier.data.item()
            ),
            'SPRL/RewardLoss': 0.0,
            'SPRL/CostLoss': 0.0,
            'SPRL/PenaltyLoss': 0.0,
            'SPRL/ProjectionNorm': 0.0,
            'SPRL/Lambda0': self._get_lambda(0),
            'SPRL/Lambda1': self._get_lambda(1),
        })

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------
    def learn(self) -> tuple[float, float, float]:
        """Main training loop with per-epoch lambda updates and DiffProjector rollouts.

        Overrides DDPG.learn() to:
          1. Pass DiffProjector to adapter rollout
          2. Inject _update_epoch() at the end of each epoch
        """
        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0

        for epoch in range(self._epochs):
            self._epoch = epoch
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            for sample_step in range(
                epoch * self._samples_per_epoch,
                (epoch + 1) * self._samples_per_epoch,
            ):
                step = (
                    sample_step
                    * self._update_cycle
                    * self._cfgs.train_cfgs.vector_env_nums
                )

                rollout_start = time.time()
                if self._cfgs.algo_cfgs.use_exploration_noise:
                    self._actor_critic.actor.noise = (
                        self._cfgs.algo_cfgs.exploration_noise
                    )

                # Rollout with projector integration
                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    use_rand_action=(
                        step <= self._cfgs.algo_cfgs.start_learning_steps
                    ),
                    projector=self._projector,
                    penalty_w=self._penalty_w,
                )
                rollout_time += time.time() - rollout_start

                update_start = time.time()
                if step > self._cfgs.algo_cfgs.start_learning_steps:
                    self._update()
                else:
                    self._log_when_not_update()
                update_time += time.time() - update_start

            # Per-epoch lambda updates
            self._update_epoch()

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.eval_episodes,
                agent=self._actor_critic,
                logger=self._logger,
            )
            eval_time = time.time() - eval_start

            self._logger.store({'Time/Update': update_time})
            self._logger.store({'Time/Rollout': rollout_time})
            self._logger.store({'Time/Evaluate': eval_time})

            if (
                step > self._cfgs.algo_cfgs.start_learning_steps
                and self._cfgs.model_cfgs.linear_lr_decay
            ):
                self._actor_critic.actor_scheduler.step()

            self._logger.store({
                'TotalEnvSteps': step + 1,
                'Time/FPS': (
                    self._cfgs.algo_cfgs.steps_per_epoch
                    / (time.time() - epoch_time)
                ),
                'Time/Total': (time.time() - start_time),
                'Time/Epoch': (time.time() - epoch_time),
                'Train/Epoch': epoch,
                'Train/LR': (
                    self._actor_critic.actor_scheduler.get_last_lr()[0]
                ),
            })

            self._logger.dump_tabular()

            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len
