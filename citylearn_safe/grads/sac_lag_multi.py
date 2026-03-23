"""SACLag with per-constraint Lagrange multipliers (multi-lambda).

Extends OmniSafe's SACLag with 5 independent Lagrange multipliers (one per
constraint) and per-constraint cost Q-critics. Analogous to PPOLagMulti but
for off-policy SAC.

Key differences from PPOLagMulti:
  - Off-policy: uses replay buffer, not on-policy rollouts
  - Q-critics: cost critics are Q(s,a) networks, not V(s) value networks
  - No GAE: TD learning for all critics
  - No trust region: SAC's entropy bonus replaces PPO's clipping/KL

The multi-lambda loss for SAC is:

    L_pi = alpha * log_pi(a|s) - min(Q1_r, Q2_r) + sum_i(lambda_i * Q_ci(s,a))

With softmax weighting over constraints at each sample:

    L_cost = sum_i( w_i * lambda_i * Q_ci )

where w_i = softmax(lambda_i * Q_ci / tau) focuses on the most-violated
constraint per sample.

Design: Follows the same architecture as PPOLagMulti (softmax advantage
selection) adapted for the off-policy Q-function setting.
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.adapter import OffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac_lag import SACLag
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.lagrange import Lagrange
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.models.critic.critic_builder import CriticBuilder

from citylearn_safe.pid_lagrange import PIDLagrange

# Beta actor mode: exact continuous action masking (Stolz et al. NeurIPS 2024)
BETA_MODE = os.environ.get("CITYLEARN_BETA_ACTOR", "0") == "1"
if BETA_MODE:
    from citylearn_safe.beta_sac_actor import BetaSACActor

# Per-constraint cost keys in the info dict (from safety_env_v3)
COST_KEYS = [
    'cost_ev_departure',          # C0: EV departure deficit (sparse, at departure)
    'cost_ev_dense',              # C1: EV charging incentive (dense, every step)
    'cost_stems_battery',         # C2: Battery SoC band violation
    'cost_stems_building_power',  # C3: Building power capacity
    'cost_stems_grid_power',      # C4: Grid power capacity
]
NUM_COSTS = len(COST_KEYS)


class _MultiCostOffPolicyAdapter(OffPolicyAdapter):
    """Extended off-policy adapter that tracks per-constraint costs.

    Stores per-constraint costs in the replay buffer alongside the standard
    (obs, act, reward, cost, done, next_obs) data. Also accumulates per-episode
    costs for lambda updates.
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
    ) -> None:
        """Standard off-policy rollout + per-constraint cost tracking.

        Stores per-constraint costs as additional keys in the replay buffer.
        """
        for _ in range(rollout_step):
            if use_rand_action:
                if BETA_MODE:
                    # Beta mode: actions in (0, 1)
                    act = torch.rand(self.action_space.shape).unsqueeze(0).to(self._device)
                else:
                    act = (
                        torch.rand(self.action_space.shape) * 2 - 1
                    ).unsqueeze(0).to(self._device)
            else:
                act = agent.step(self._current_obs.to(self._device), deterministic=False)

            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)
            real_next_obs = next_obs.clone()

            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    if 'final_observation' in info:
                        real_next_obs[idx] = info['final_observation'][idx]
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)
                    # Save completed episode costs for lambda updates
                    self._last_completed_ep_costs = list(self._per_ep_costs)
                    self._per_ep_costs = [0.0] * NUM_COSTS

            # Extract per-constraint costs from info dict
            # On terminal steps, AutoReset moves real info to info['final_info']
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

            # Store in buffer (standard fields + per-constraint costs)
            buffer.store(
                obs=self._current_obs,
                act=act,
                reward=reward,
                cost=cost,
                done=torch.logical_and(
                    terminated, torch.logical_xor(terminated, truncated)
                ),
                next_obs=real_next_obs,
                **per_cost_vals,
            )

            self._current_obs = next_obs


class _MultiCostOffPolicyBuffer(VectorOffPolicyBuffer):
    """Off-policy replay buffer with per-constraint cost storage.

    Extends VectorOffPolicyBuffer to store per-constraint cost signals
    alongside the standard (obs, act, reward, cost, done, next_obs) data.
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
        # Add per-constraint cost buffers
        for i in range(NUM_COSTS):
            self.data[f'cost_{i}'] = torch.zeros(
                (size, num_envs), dtype=torch.float32, device=device
            )


@registry.register
class SACLagMulti(SACLag):
    """SACLag with per-constraint lambdas and softmax Q-cost weighting.

    Uses 5 independent Lagrange multipliers (per-constraint) with per-sample
    softmax weighting to focus the cost signal on the most-violated constraint.

    Inherits from SACLag (which inherits from SAC -> DDPG -> BaseAlgo).
    Overrides:
      - _init_env: uses _MultiCostOffPolicyAdapter
      - _init_model: adds per-constraint cost Q-critics
      - _init: creates per-constraint lambdas + multi-cost buffer
      - _init_log: registers per-constraint metrics
      - _loss_pi: softmax-weighted multi-lambda loss
      - _update: updates per-constraint critics and lambdas
      - _update_cost_critic: updates per-constraint cost critics
    """

    def _init_env(self) -> None:
        """Create a MultiCost off-policy adapter."""
        self._env: _MultiCostOffPolicyAdapter = _MultiCostOffPolicyAdapter(
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
        """Build standard actor-critic plus per-constraint cost Q-critics."""
        super()._init_model()

        # Beta actor: replace GaussianSACActor with BetaSACActor
        if BETA_MODE:
            obs_dim = self._env.observation_space.shape[0]
            act_dim = self._env.action_space.shape[0]
            hidden = list(self._cfgs.model_cfgs.actor.hidden_sizes)
            beta_actor = BetaSACActor(
                obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=tuple(hidden),
                activation=self._cfgs.model_cfgs.actor.activation,
            ).to(self._device)
            self._actor_critic.actor = beta_actor
            # Rebuild actor optimizer with new parameters
            self._actor_critic.actor_optimizer = torch.optim.Adam(
                beta_actor.parameters(), lr=self._cfgs.model_cfgs.actor.lr,
            )
            print(f"[SACLagMulti] BetaSACActor: obs={obs_dim} act={act_dim} "
                  f"hidden={hidden}")

        obs_space = self._env.observation_space
        act_space = self._env.action_space
        model_cfgs = self._cfgs.model_cfgs

        # Per-constraint cost Q-critics (each is a Q(s,a) network)
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

        print(f"[SACLagMulti] Added {NUM_COSTS} per-constraint cost Q-critics")

    def _init(self) -> None:
        """Initialize multi-cost buffer and per-constraint Lagrange multipliers."""
        # SAC._init creates the standard buffer + auto_alpha.
        # We need to replace the buffer with our multi-cost version.
        # Call SAC's parent (DDPG._init) first for the buffer, then SAC's alpha init.

        # --- Multi-cost buffer (replaces DDPG._init's buffer creation) ---
        self._buf: _MultiCostOffPolicyBuffer = _MultiCostOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._cfgs.algo_cfgs.size,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

        # --- SAC's auto_alpha initialization (from SAC._init) ---
        if self._cfgs.algo_cfgs.auto_alpha:
            self._target_entropy = -torch.prod(
                torch.Tensor(self._env.action_space.shape)
            ).item()
            self._log_alpha = torch.zeros(
                1, requires_grad=True, device=self._device
            )
            assert self._cfgs.model_cfgs.critic.lr is not None
            self._alpha_optimizer = torch.optim.Adam(
                [self._log_alpha],
                lr=self._cfgs.model_cfgs.critic.lr,
            )
        else:
            self._log_alpha = torch.log(
                torch.tensor(
                    self._cfgs.algo_cfgs.alpha, device=self._device
                ),
            )

        # --- SACLag's single Lagrange multiplier (for compatibility) ---
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

        # --- Per-constraint Lagrange multipliers ---
        multi_cfgs = getattr(self._cfgs, 'multi_cfgs', None)
        assert multi_cfgs is not None, (
            "multi_cfgs not found in config. SACLagMulti requires per-constraint "
            "cost limits under multi_cfgs."
        )

        # Softmax temperature for per-sample constraint selection
        self._tau = float(getattr(multi_cfgs, 'tau', 1.0))

        # Per-constraint cost limits
        self._per_cost_limits = [
            float(getattr(multi_cfgs, f'cost_limit_{i}', 5000.0))
            for i in range(NUM_COSTS)
        ]

        # Check if PID Lagrangian is enabled
        self._use_pid = os.environ.get("CITYLEARN_PID_LAGRANGE", "0") == "1"

        lag_cfgs = self._cfgs.lagrange_cfgs

        if self._use_pid:
            # PID Lagrangian (Stooke et al., ICML 2020)
            pid_kp_default = float(getattr(multi_cfgs, 'pid_kp', 0.1))
            pid_ki_default = float(getattr(multi_cfgs, 'pid_ki', 0.01))
            pid_kd_default = float(getattr(multi_cfgs, 'pid_kd', 0.01))
            pid_d_delay = int(getattr(multi_cfgs, 'pid_d_delay', 10))
            pid_ema_p_default = float(
                getattr(multi_cfgs, 'pid_delta_p_ema_alpha', 0.95)
            )
            pid_ema_d_default = float(
                getattr(multi_cfgs, 'pid_delta_d_ema_alpha', 0.95)
            )
            penalty_max = float(
                getattr(lag_cfgs, 'lagrangian_upper_bound', 3.0)
            )
            init_val = float(
                getattr(lag_cfgs, 'lagrangian_multiplier_init', 0.001)
            )

            self._per_lagranges = []
            for i in range(NUM_COSTS):
                kp = float(getattr(multi_cfgs, f'pid_kp_{i}', pid_kp_default))
                ki = float(getattr(multi_cfgs, f'pid_ki_{i}', pid_ki_default))
                kd = float(getattr(multi_cfgs, f'pid_kd_{i}', pid_kd_default))
                ema_p = float(
                    getattr(multi_cfgs, f'pid_ema_p_{i}', pid_ema_p_default)
                )
                ema_d = float(
                    getattr(multi_cfgs, f'pid_ema_d_{i}', pid_ema_d_default)
                )
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
                if (
                    kp != pid_kp_default
                    or ki != pid_ki_default
                    or kd != pid_kd_default
                ):
                    print(
                        f"[SACLagMulti] PID C{i}: "
                        f"Kp={kp}, Ki={ki}, Kd={kd}, ema_p={ema_p}"
                    )
            print(
                f"[SACLagMulti] Using PID Lagrangian "
                f"(defaults: Kp={pid_kp_default}, Ki={pid_ki_default}, "
                f"Kd={pid_kd_default}, delay={pid_d_delay}, "
                f"penalty_max={penalty_max})"
            )

        if not self._use_pid:
            # Standard SGD Lagrangian
            upper_bound = getattr(lag_cfgs, 'lagrangian_upper_bound', None)
            if upper_bound is not None:
                upper_bound = float(upper_bound)

            self._per_lagranges: list[Lagrange] = []
            for i in range(NUM_COSTS):
                self._per_lagranges.append(
                    Lagrange(
                        cost_limit=self._per_cost_limits[i],
                        lagrangian_multiplier_init=float(
                            lag_cfgs.lagrangian_multiplier_init
                        ),
                        lambda_lr=float(lag_cfgs.lambda_lr),
                        lambda_optimizer=str(lag_cfgs.lambda_optimizer),
                        lagrangian_upper_bound=upper_bound,
                    )
                )
            print("[SACLagMulti] Using standard SGD Lagrangian")

        # --- Cost limit curriculum annealing (ported from PPOLagMulti) ---
        # Config format: anneal_cost_limit_<i>: [start_value, end_value, start_epoch, end_epoch]
        self._cost_limit_schedules = {}
        for i in range(NUM_COSTS):
            key = f'anneal_cost_limit_{i}'
            schedule = getattr(multi_cfgs, key, None)
            if schedule is not None:
                s = [float(x) for x in schedule]
                self._cost_limit_schedules[i] = {
                    'start_val': s[0], 'end_val': s[1],
                    'start_epoch': int(s[2]), 'end_epoch': int(s[3])
                }
                # Override initial cost limit to start value
                self._per_cost_limits[i] = s[0]
                if self._use_pid:
                    self._per_lagranges[i].update_cost_limit(s[0])
                else:
                    self._per_lagranges[i].cost_limit = s[0]
                print(f"[SACLagMulti Curriculum] C{i}: {s[0]} → {s[1]} "
                      f"(epochs {int(s[2])}-{int(s[3])})")

        # Epoch counter
        self._epoch_counter = 0

        print(f"[SACLagMulti] tau={self._tau}")
        print(
            f"[SACLagMulti] per-constraint cost limits: "
            f"{self._per_cost_limits}"
        )
        for i in range(NUM_COSTS):
            lam_val = self._get_lambda(i)
            print(
                f"[SACLagMulti]   Lambda_{i}: init={lam_val:.4f}, "
                f"limit={self._per_cost_limits[i]:.0f}"
            )

    def _get_lambda(self, idx: int) -> float:
        """Get lambda value from either Lagrange or PIDLagrange."""
        lam = self._per_lagranges[idx].lagrangian_multiplier
        return float(lam) if isinstance(lam, (int, float)) else lam.item()

    def _init_log(self) -> None:
        """Register multi-lagrangian specific log keys."""
        super()._init_log()
        # Per-constraint metrics
        for i in range(NUM_COSTS):
            self._logger.register_key(f'Loss/Loss_cost_critic_{i}')
            self._logger.register_key(f'Value/cost_critic_{i}')
            self._logger.register_key(f'Metrics/EpCost_{i}')
            self._logger.register_key(f'Metrics/Lambda_{i}')
        # Diagnostics
        self._logger.register_key('MultiLag/RewardFrac')
        self._logger.register_key('MultiLag/MaxLambda')
        self._logger.register_key('MultiLag/SumLambda')
        self._logger.register_key('MultiLag/DominantConstraint')

    def _loss_pi(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute actor loss with softmax-weighted multi-lambda cost.

        L = alpha * log_pi(a|s) - min(Q1_r, Q2_r)
            + sum_i( w_i * lambda_i * Q_ci(s,a) )

        where w_i = softmax(lambda_i * Q_ci / tau) focuses on the
        most-violated constraint per sample.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)

        # Reward Q-values
        loss_q_r_1, loss_q_r_2 = self._actor_critic.reward_critic(obs, action)
        loss_r = self._alpha * log_prob - torch.min(loss_q_r_1, loss_q_r_2)

        # Per-constraint cost Q-values: [batch, NUM_COSTS]
        lambdas = [self._get_lambda(i) for i in range(NUM_COSTS)]
        q_costs = []
        for i in range(NUM_COSTS):
            q_ci = self._cost_q_critics[i](obs, action)[0]  # [batch]
            q_costs.append(q_ci)
        q_cost_stack = torch.stack(q_costs, dim=-1)  # [batch, NUM_COSTS]

        # Weight by lambdas
        lambda_tensor = torch.tensor(
            lambdas, device=obs.device, dtype=obs.dtype
        )
        weighted_q = q_cost_stack * lambda_tensor.unsqueeze(0)  # [batch, NUM_COSTS]

        # Softmax weighting (focus on most-violated constraint per sample)
        logits = torch.clamp(weighted_q / self._tau, min=-50.0, max=50.0)
        softmax_weights = F.softmax(logits, dim=-1)  # [batch, NUM_COSTS]
        loss_c = (softmax_weights * weighted_q).sum(dim=-1)  # [batch]

        # Combined loss (no /(1+sum_lambda) normalization -- see PPOLagMulti)
        total_loss = (loss_r + loss_c).mean()

        # Log diagnostics
        sum_lambda = sum(lambdas)
        loss_r_mag = loss_r.abs().mean().item()
        loss_c_mag = loss_c.abs().mean().item()
        reward_frac = loss_r_mag / (loss_r_mag + loss_c_mag + 1e-8)
        dominant = softmax_weights.mean(dim=0).argmax().item()
        self._logger.store({
            'MultiLag/RewardFrac': reward_frac,
            'MultiLag/MaxLambda': max(lambdas),
            'MultiLag/SumLambda': sum_lambda,
            'MultiLag/DominantConstraint': dominant,
        })

        return total_loss

    def _update(self) -> None:
        """Update actor, critics, lambdas, and alpha.

        Overrides SAC._update to add:
        1. Per-constraint cost critic updates
        2. Per-constraint lambda updates (at epoch boundaries)
        3. Polyak updates for per-constraint target critics
        """
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            # Standard reward critic update
            self._update_reward_critic(obs, act, reward, done, next_obs)

            # Standard combined cost critic update (for SACLag compatibility)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)

            # Per-constraint cost critic updates
            for i in range(NUM_COSTS):
                cost_i = data[f'cost_{i}']
                self._update_per_cost_critic(obs, act, cost_i, done, next_obs, i)

            # Actor + alpha update (every policy_delay steps)
            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                # Polyak update for standard targets
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

    def _update_per_cost_critic(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost_i: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
        idx: int,
    ) -> None:
        """Update a per-constraint cost Q-critic via TD learning."""
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(
                next_obs, deterministic=True
            )
            next_q_c = self._cost_q_critic_targets[idx](
                next_obs, next_action
            )[0]
            target_q_c = (
                cost_i
                + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_c
            )

        q_c = self._cost_q_critics[idx](obs, act)[0]
        loss = nn.functional.mse_loss(q_c, target_q_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._cost_q_critics[idx].parameters():
                loss += (
                    param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff
                )

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

    def _update_cost_limits(self, epoch: int) -> None:
        """Update cost limits based on curriculum schedule."""
        for i, sched in self._cost_limit_schedules.items():
            start_e = sched['start_epoch']
            end_e = sched['end_epoch']
            start_v = sched['start_val']
            end_v = sched['end_val']

            if epoch < start_e:
                new_limit = start_v
            elif epoch >= end_e:
                new_limit = end_v
            else:
                progress = (epoch - start_e) / max(1, end_e - start_e)
                new_limit = start_v + progress * (end_v - start_v)

            if new_limit != self._per_cost_limits[i]:
                old = self._per_cost_limits[i]
                self._per_cost_limits[i] = new_limit
                if self._use_pid:
                    self._per_lagranges[i].update_cost_limit(new_limit)
                else:
                    self._per_lagranges[i].cost_limit = new_limit
                print(f"[SACLagMulti Curriculum] Epoch {epoch}: "
                      f"C{i} limit {old:.0f} → {new_limit:.0f}")

    def _update_epoch(self) -> None:
        """Per-epoch updates: update cost limits + per-constraint lambdas."""
        # Update cost limits from curriculum schedule
        self._update_cost_limits(self._epoch_counter)

        # Update per-constraint lambdas from last completed episode
        for i in range(NUM_COSTS):
            ep_cost_i = self._env.get_per_constraint_ep_cost(i)
            if self._use_pid:
                self._per_lagranges[i].pid_update(ep_cost_i)
            else:
                if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
                    self._per_lagranges[i].update_lagrange_multiplier(ep_cost_i)
            self._logger.store({
                f'Metrics/EpCost_{i}': ep_cost_i,
                f'Metrics/Lambda_{i}': self._get_lambda(i),
            })

        # Also update the parent's single lambda (for compatibility/logging)
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            self._lagrange.update_lagrange_multiplier(Jc)
        self._logger.store({
            'Metrics/LagrangeMultiplier': (
                self._lagrange.lagrangian_multiplier.data.item()
            ),
        })

        self._epoch_counter += 1

    def _log_when_not_update(self) -> None:
        """Log default values when not updating (during initial random steps)."""
        super()._log_when_not_update()
        for i in range(NUM_COSTS):
            self._logger.store({
                f'Loss/Loss_cost_critic_{i}': 0.0,
                f'Value/cost_critic_{i}': 0.0,
                f'Metrics/EpCost_{i}': 0.0,
                f'Metrics/Lambda_{i}': self._get_lambda(i),
            })
        self._logger.store({
            'MultiLag/RewardFrac': 1.0,
            'MultiLag/MaxLambda': max(self._get_lambda(i) for i in range(NUM_COSTS)),
            'MultiLag/SumLambda': sum(self._get_lambda(i) for i in range(NUM_COSTS)),
            'MultiLag/DominantConstraint': 0.0,
        })

    def learn(self) -> tuple[float, float, float]:
        """Main training loop with per-epoch lambda updates.

        Overrides SAC.learn() to inject _update_epoch() at the end of
        each epoch for per-constraint lambda updates.
        """
        import time

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

                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    use_rand_action=(
                        step <= self._cfgs.algo_cfgs.start_learning_steps
                    ),
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
