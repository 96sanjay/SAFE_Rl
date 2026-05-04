"""PPO with per-constraint Lagrange multipliers and softmax advantage selection.

Extends OmniSafe's PPO with 5 independent Lagrange multipliers (one per
constraint) and per-constraint cost critics. OmniSafe provides the RL engine
(PPO: on-policy rollouts, trust region, GAE); this module adds the
multi-constraint CMDP layer on top.

Key design:
  - Inherits from PPO (not PPOLag) — no orphaned single-cost machinery
  - 5 per-constraint cost V-critics with independent PID/SGD lambdas
  - Softmax weighting focuses actor loss on most-violated constraint
  - Cost weights (cost_weight_* in multi_cfgs) scale critic training signal
  - Lambda PID updates use raw costs (physical units, compared to cost_limits)

Based on PPO with per-constraint softmax advantage selection.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.lagrange import Lagrange
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.utils import distributed

from citylearn_safe.masked_onpolicy_buffer import MaskedVectorOnPolicyBuffer
from citylearn_safe.policy_action_mask import masked_log_prob_from_action
from citylearn_safe.pid_lagrange import PIDLagrange

# Reuse MultiCostAdapter and constants from GradS
from citylearn_safe.grads.ppo_lag_grads import (
    COST_KEYS,
    NUM_COSTS,
    _MultiCostAdapter,
)

# GradS selector (standalone, no OmniSafe dependency)
from citylearn_safe.grads.grads_selector import GradSSelector

# Import discount_cumsum for GAE computation
from omnisafe.utils.math import discount_cumsum


@registry.register
class PPOLagMulti(PPO):
    """PPO with per-constraint lambdas and softmax advantage selection.

    Uses 5 independent Lagrange multipliers (per-constraint) with per-timestep
    softmax weighting to focus the cost signal on the most-violated constraint.
    All constraints contribute every step — no constraint dropping.

    Inherits from PPO (which inherits from PolicyGradient -> BaseAlgo).
    OmniSafe provides: on-policy rollouts, trust region, GAE, value critics.
    This class adds: per-constraint cost critics, PID/SGD lambdas,
    softmax-weighted advantage combination, cost weights, curriculum annealing.
    """

    def _init_env(self) -> None:
        """Create a MultiCost adapter that tracks per-constraint costs."""
        self._env: _MultiCostAdapter = _MultiCostAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:
        """Build standard actor-critic plus per-constraint cost critics."""
        super()._init_model()

        obs_space = self._env.observation_space
        act_space = self._env.action_space
        model_cfgs = self._cfgs.model_cfgs

        self._cost_critics = nn.ModuleList()
        self._cost_critic_optimizers = []

        for i in range(NUM_COSTS):
            critic = CriticBuilder(
                obs_space=obs_space,
                act_space=act_space,
                hidden_sizes=model_cfgs.critic.hidden_sizes,
                activation=model_cfgs.critic.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode,
                num_critics=1,
                use_obs_encoder=False,
            ).build_critic('v').to(self._device)
            self._cost_critics.append(critic)
            self._cost_critic_optimizers.append(
                torch.optim.Adam(critic.parameters(), lr=model_cfgs.critic.lr)
            )

        # Set reference on adapter so rollout can access cost critics
        self._env._cost_critics_ref = self._cost_critics

        print(f"[MultiLag] Added {NUM_COSTS} per-constraint cost critics")

    def _init(self) -> None:
        """Initialize buffer and per-constraint Lagrange multipliers."""
        super()._init()  # PPO._init (buffer setup)
        self._policy_action_mask = os.environ.get("CITYLEARN_POLICY_ACTION_MASK", "0") == "1"
        if self._policy_action_mask:
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
            self._env._policy_action_mask = True
            print("[MultiLag] Policy-side continuous action mask ENABLED")

        # Read multi-lagrangian config
        multi_cfgs = getattr(self._cfgs, 'multi_cfgs', None)
        assert multi_cfgs is not None, (
            "multi_cfgs not found in config -- check YAML parsing. "
            "PPOLagMulti requires per-constraint cost limits under multi_cfgs."
        )

        # Softmax temperature for per-timestep constraint selection
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
            # Replaces SGD with PID controller for lambda updates
            # PID params live in multi_cfgs to avoid polluting parent's lagrange_cfgs
            # Supports per-constraint overrides: pid_kp_0, pid_ki_0, etc.
            pid_kp_default = float(getattr(multi_cfgs, 'pid_kp', 0.1))
            pid_ki_default = float(getattr(multi_cfgs, 'pid_ki', 0.01))
            pid_kd_default = float(getattr(multi_cfgs, 'pid_kd', 0.01))
            pid_d_delay = int(getattr(multi_cfgs, 'pid_d_delay', 10))
            pid_ema_p_default = float(getattr(multi_cfgs, 'pid_delta_p_ema_alpha', 0.95))
            pid_ema_d_default = float(getattr(multi_cfgs, 'pid_delta_d_ema_alpha', 0.95))
            penalty_max = float(getattr(lag_cfgs, 'lagrangian_upper_bound', 3.0))
            init_val = float(getattr(lag_cfgs, 'lagrangian_multiplier_init', 0.001))

            self._per_lagranges = []
            for i in range(NUM_COSTS):
                # Per-constraint overrides (e.g., pid_kp_0, pid_ki_3)
                kp = float(getattr(multi_cfgs, f'pid_kp_{i}', pid_kp_default))
                ki = float(getattr(multi_cfgs, f'pid_ki_{i}', pid_ki_default))
                kd = float(getattr(multi_cfgs, f'pid_kd_{i}', pid_kd_default))
                ema_p = float(getattr(multi_cfgs, f'pid_ema_p_{i}', pid_ema_p_default))
                ema_d = float(getattr(multi_cfgs, f'pid_ema_d_{i}', pid_ema_d_default))
                self._per_lagranges.append(PIDLagrange(
                    cost_limit=self._per_cost_limits[i],
                    pid_kp=kp,
                    pid_ki=ki,
                    pid_kd=kd,
                    pid_d_delay=pid_d_delay,
                    pid_delta_p_ema_alpha=ema_p,
                    pid_delta_d_ema_alpha=ema_d,
                    penalty_max=penalty_max,
                    lagrangian_multiplier_init=init_val,
                ))
                if kp != pid_kp_default or ki != pid_ki_default or kd != pid_kd_default:
                    print(f"[MultiLag] PID C{i}: Kp={kp}, Ki={ki}, Kd={kd}, ema_p={ema_p}")
            print(f"[MultiLag] Using PID Lagrangian (defaults: Kp={pid_kp_default}, "
                  f"Ki={pid_ki_default}, Kd={pid_kd_default}, delay={pid_d_delay}, "
                  f"penalty_max={penalty_max})")
        else:
            # Standard SGD Lagrangian
            upper_bound = getattr(lag_cfgs, 'lagrangian_upper_bound', None)
            if upper_bound is not None:
                upper_bound = float(upper_bound)

            self._per_lagranges: list[Lagrange] = []
            for i in range(NUM_COSTS):
                self._per_lagranges.append(Lagrange(
                    cost_limit=self._per_cost_limits[i],
                    lagrangian_multiplier_init=float(lag_cfgs.lagrangian_multiplier_init),
                    lambda_lr=float(lag_cfgs.lambda_lr),
                    lambda_optimizer=str(lag_cfgs.lambda_optimizer),
                    lagrangian_upper_bound=upper_bound,
                ))
            print("[MultiLag] Using standard SGD Lagrangian")

        # R18: Cost limit curriculum annealing
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
                # Also update the Lagrange controller's limit
                self._per_lagranges[i]._cost_limit = s[0] if self._use_pid else None
                if not self._use_pid:
                    self._per_lagranges[i].cost_limit = s[0]
                print(f"[Curriculum] C{i}: {s[0]} → {s[1]} "
                      f"(epochs {int(s[2])}-{int(s[3])})")

        # Epoch counter for curriculum (incremented in _update)
        self._epoch_counter = 0

        # --- Per-constraint cost weights (scale critic training signal) ---
        self._cost_weights = [
            float(getattr(multi_cfgs, f'cost_weight_{i}', 1.0))
            for i in range(NUM_COSTS)
        ]
        self._env._cost_weights = self._cost_weights
        if any(w != 1.0 for w in self._cost_weights):
            print(f"[MultiLag] Cost weights: {self._cost_weights}")

        # Instance variable for per-constraint advantages in current mini-batch
        # Set in _update() before _update_actor() is called
        self._current_batch_adv_cs: list[torch.Tensor] | None = None

        # ------------------------------------------------------------------ #
        # R22: GradS — gradient surgery in gradient space (opt-in)
        # Enabled by setting use_grads: true in multi_cfgs.
        # When enabled, replaces the single-backward softmax advantage
        # combination with per-constraint gradient surgery:
        #   1. Compute reward gradient (PPO surrogate)
        #   2. Compute 5 per-constraint gradients (policy gradient)
        #   3. GradSSelector picks the non-conflicting constraint
        #   4. Apply: g = g_reward + lambda_sel * scale * g_cost_sel
        # PID Lagrangian, curriculum, and multi-critic are all preserved.
        # ------------------------------------------------------------------ #
        self._use_grads = bool(getattr(multi_cfgs, 'use_grads', False))
        if self._use_grads:
            sim_thresh      = float(getattr(multi_cfgs, 'grads_sim_threshold',      0.8))
            conflict_thresh = float(getattr(multi_cfgs, 'grads_conflict_threshold', 0.999))
            sampling        = str(getattr(multi_cfgs,   'grads_sampling',           'lambda'))
            self._grads_selector = GradSSelector(
                num_costs=NUM_COSTS,
                sim_threshold=sim_thresh,
                conflict_threshold=conflict_thresh,
                sampling=sampling,
            )
            print(f"[MultiLag] GradS ENABLED: sim_thresh={sim_thresh}, "
                  f"conflict_thresh={conflict_thresh}, sampling={sampling}")
        else:
            self._grads_selector = None
            print("[MultiLag] GradS disabled (softmax advantage mode)")

        print(f"[MultiLag] tau={self._tau}")
        print(f"[MultiLag] per-constraint cost limits: {self._per_cost_limits}")
        for i in range(NUM_COSTS):
            lam_val = self._get_lambda(i)
            print(f"[MultiLag]   Lambda_{i}: init={lam_val:.4f}, limit={self._per_cost_limits[i]:.0f}")

    def _get_lambda(self, idx: int) -> float:
        """Get lambda value from either Lagrange or PIDLagrange."""
        lam = self._per_lagranges[idx].lagrangian_multiplier
        return float(lam) if isinstance(lam, (int, float)) else lam.item()

    def _update_cost_limits(self, epoch: int) -> None:
        """Update cost limits based on curriculum schedule (R18)."""
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
                if epoch % 5 == 0 or epoch == start_e or epoch == end_e:
                    print(f"[Curriculum] C{i} limit: {old:.0f} → {new_limit:.0f} "
                          f"(epoch {epoch})")

    def _init_log(self) -> None:
        """Register multi-lagrangian specific log keys."""
        super()._init_log()
        # OmniSafe compatibility (was registered by PPOLag parent)
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)
        # Per-constraint metrics
        for i in range(NUM_COSTS):
            self._logger.register_key(f'Loss/Loss_cost_critic_{i}')
            self._logger.register_key(f'Metrics/EpCost_{i}')
            self._logger.register_key(f'Metrics/Lambda_{i}')
        # Diagnostics
        self._logger.register_key('MultiLag/RewardAdvFrac')
        self._logger.register_key('MultiLag/MaxLambda')
        self._logger.register_key('MultiLag/SumLambda')
        self._logger.register_key('MultiLag/DominantConstraint')
        self._logger.register_key('MultiLag/EffectiveTau')
        # GradS diagnostics (only meaningful when use_grads=true)
        self._logger.register_key('GradS/SelectedConstraint')
        self._logger.register_key('GradS/CandidateSetSize')
        self._logger.register_key('GradS/Scale')
        self._logger.register_key('GradS/CosSim_C0_C3')
        self._logger.register_key('GradS/CosSim_C0_C4')
        self._logger.register_key('GradS/CosSim_C3_C4')
        # Reward-vs-constraint cosine similarities
        for i in range(NUM_COSTS):
            self._logger.register_key(f'GradS/CosSim_reward_C{i}')
        # Per-reward-term diagnostics (from cmdp_env info dict)
        _reward_keys = [
            'r_eco', 'r_sg', 'r_sb', 'r_ramp', 'r_ren', 'r_ev',
            'r_ev_guard', 'r_v2g_ctx', 'r_peak_shave', 'r_load_shift',
            'r_grid_mild', 'r_barrier', 'r_ev_solar', 'r_solar_store',
            'r_ev_slack_arb', 'r_headroom', 'r_grid_penalty', 'r_price_arb',
            'r_nec_sign', 'r_trajectory', 'r_ev_smart',
        ]
        for rk in _reward_keys:
            self._logger.register_key(f'Reward/{rk}')
        # Trajectory sub-components
        self._logger.register_key('Reward/r_traj_batt')
        self._logger.register_key('Reward/r_traj_ev')
        self._logger.register_key('Reward/traj_forecast_signal')
        self._logger.register_key('Reward/traj_forecast_mean')

    def _compute_adv_surrogate(
        self, adv_r: torch.Tensor, adv_c: torch.Tensor
    ) -> torch.Tensor:
        """Compute surrogate advantage using softmax per-timestep selection.

        Instead of summing all lambda_i * A_ci (which causes cancellation when
        constraints conflict), we use softmax weighting so the most-violated
        constraint dominates at each timestep.

        Args:
            adv_r: Reward advantage (z-scored by OmniSafe buffer). Shape [batch].
            adv_c: Combined cost advantage from buffer (IGNORED — we use per-constraint).

        Returns:
            Combined advantage for PPO loss. Shape [batch].
        """
        assert self._current_batch_adv_cs is not None, (
            "_current_batch_adv_cs not set. Must be set before calling _update_actor."
        )

        # Get per-constraint lambda values
        lambdas = [self._get_lambda(i) for i in range(NUM_COSTS)]
        sum_lambda = sum(lambdas)

        # Stack per-constraint advantages: [batch, NUM_COSTS]
        adv_stack = torch.stack(self._current_batch_adv_cs, dim=-1)

        # Weight by lambdas: [batch, NUM_COSTS]
        lambda_tensor = torch.tensor(lambdas, device=adv_r.device, dtype=adv_r.dtype)
        weighted_advs = adv_stack * lambda_tensor.unsqueeze(0)

        # Per-timestep softmax weighting (focus on most-violated)
        # Higher weighted_adv = more cost-increasing = more violated at this timestep
        # Clamp for numerical stability (prevents NaN if lambdas grow very large)
        logits = torch.clamp(weighted_advs / self._tau, min=-50.0, max=50.0)
        softmax_weights = F.softmax(logits, dim=-1)  # [batch, NUM_COSTS]
        A_cost = (softmax_weights * weighted_advs).sum(dim=-1)  # [batch]

        # R19: Removed cost advantage clipping (was 2x reward std).
        # The clipping created a hard ceiling that prevented λ from enforcing
        # constraints when reward signals were strong. PID + lagrangian_upper_bound
        # provide sufficient stability without artificial clipping.

        # Standard CMDP Lagrangian: L = adv_r - sum(lambda_i * A_ci)
        # No /(1+sum_lambda) normalization — that was killing 90%+ of reward signal
        combined = adv_r - A_cost

        # Log diagnostics (sample from batch)
        adv_r_mag = adv_r.abs().mean().item()
        a_cost_mag = A_cost.abs().mean().item()
        reward_frac = adv_r_mag / (adv_r_mag + a_cost_mag + 1e-8)
        dominant = softmax_weights.mean(dim=0).argmax().item()
        self._logger.store({
            'MultiLag/RewardAdvFrac': reward_frac,
            'MultiLag/MaxLambda': max(lambdas),
            'MultiLag/SumLambda': sum_lambda,
            'MultiLag/DominantConstraint': dominant,
            'MultiLag/EffectiveTau': self._tau,
        })

        return combined

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
        safe_min: torch.Tensor | None = None,
        safe_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """PPO clipped loss with optional KL regularization toward BC reference policy (R30).

        Overrides PPO._loss_pi to inject a KL penalty term that anchors the policy
        near the BC-warmstarted reference, preventing catastrophic forgetting of
        good initialization during early RL training.
        """
        distribution = self._actor_critic.actor(obs)
        if self._policy_action_mask and safe_min is not None and safe_max is not None:
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

        # R30: KL penalty toward BC reference policy
        ref_actor = getattr(self, '_bc_ref_actor', None)
        kl_beta = getattr(self, '_kl_beta', 0.0)
        if ref_actor is not None and kl_beta > 0:
            with torch.no_grad():
                ref_dist = ref_actor(obs)
            kl = torch.distributions.kl_divergence(
                distribution,
                torch.distributions.Normal(ref_dist.mean, ref_dist.stddev),
            ).sum(-1).mean()
            kl = torch.clamp(kl, max=100.0)  # prevent explosion
            loss = loss + kl_beta * kl

        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss

    def _update(self) -> None:
        """Update with per-constraint lambdas and softmax advantage selection.

        Overrides the full update loop to:
        1. Update per-constraint lambdas independently
        2. Compute per-constraint GAE
        3. Build DataLoader with per-constraint advantages
        4. For each mini-batch, set instance variable and call standard _update_actor
        """
        # R18: Update cost limits based on curriculum schedule
        self._update_cost_limits(self._epoch_counter)
        self._epoch_counter += 1

        # R30: KL beta decay
        if hasattr(self, '_kl_beta') and self._kl_beta > 0:
            decay = float(os.environ.get("CITYLEARN_KL_BETA_DECAY", "1.0"))
            if decay < 1.0:
                self._kl_beta *= decay

        # 1. Update per-constraint lambdas independently
        for i in range(NUM_COSTS):
            ep_cost_i = self._env.get_per_constraint_ep_cost(i)
            if self._use_pid:
                self._per_lagranges[i].pid_update(ep_cost_i)
            else:
                self._per_lagranges[i].update_lagrange_multiplier(ep_cost_i)
            self._logger.store({
                f'Metrics/EpCost_{i}': ep_cost_i,
                f'Metrics/Lambda_{i}': self._get_lambda(i),
            })

        # 3. Compute per-constraint GAE from adapter's stored data
        per_cost_advs = self._compute_per_constraint_gae()

        # 4. Get standard data from buffer
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'], data['act'], data['logp'],
            data['target_value_r'], data['target_value_c'],
            data['adv_r'], data['adv_c'],
        )
        safe_min = data.get('safe_min')
        safe_max = data.get('safe_max')

        # 5. Build per-constraint target values for critic updates
        per_cost_targets = self._compute_per_constraint_targets()

        # 6. Create DataLoader with all data including per-constraint advs/targets
        #    Move per-constraint data to same device as buffer data (prevents
        #    device mismatch if running on GPU)
        tensors = [obs, act, logp, target_value_r, target_value_c, adv_r, adv_c]
        if self._policy_action_mask and safe_min is not None and safe_max is not None:
            tensors.extend([safe_min, safe_max])
        for i in range(NUM_COSTS):
            tensors.append(per_cost_advs[i].to(obs.device))
            tensors.append(per_cost_targets[i].to(obs.device))

        original_obs = obs

        # R20: Compute old distribution in chunks to avoid CUDA OOM with STEMS encoder.
        # Store loc/scale tensors (small: N × act_dim) instead of full distribution.
        _kl_chunk = 512
        with torch.no_grad():
            _old_locs, _old_scales = [], []
            for _s in range(0, obs.shape[0], _kl_chunk):
                _e = min(_s + _kl_chunk, obs.shape[0])
                _d = self._actor_critic.actor(obs[_s:_e])
                _old_locs.append(_d.loc.detach())
                _old_scales.append(_d.scale.detach())
            _old_loc = torch.cat(_old_locs, dim=0)
            _old_scale = torch.cat(_old_scales, dim=0)

        dataloader = DataLoader(
            dataset=TensorDataset(*tensors),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        for epoch_i in track(
            range(self._cfgs.algo_cfgs.update_iters), description='Updating (MultiLag)...'
        ):
            for batch in dataloader:
                b_obs = batch[0]
                b_act = batch[1]
                b_logp = batch[2]
                b_target_value_r = batch[3]
                b_target_value_c = batch[4]
                b_adv_r = batch[5]
                b_adv_c = batch[6]
                offset = 7
                b_safe_min = None
                b_safe_max = None
                if self._policy_action_mask and safe_min is not None and safe_max is not None:
                    b_safe_min = batch[7]
                    b_safe_max = batch[8]
                    offset = 9
                b_adv_cs = []
                b_target_cs = []
                for i in range(NUM_COSTS):
                    b_adv_cs.append(batch[offset + 2 * i])
                    b_target_cs.append(batch[offset + 2 * i + 1])

                # Update reward critic (standard)
                self._update_reward_critic(b_obs, b_target_value_r)

                # Update combined cost critic (for buffer bootstrap values and logging)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(b_obs, b_target_value_c)

                # Update per-constraint cost critics
                for i in range(NUM_COSTS):
                    self._update_per_cost_critic(b_obs, b_target_cs[i], i)

                # Set per-constraint advantages for this mini-batch
                # (accessed by _compute_adv_surrogate via instance variable)
                self._current_batch_adv_cs = b_adv_cs
                # Also store reward adv for GradS path
                self._current_batch_adv_r = b_adv_r

                if self._use_grads:
                    # R22: GradS — gradient surgery in gradient space
                    # 6 backward passes; selects non-conflicting constraint gradient
                    self._update_actor_with_grads(
                        b_obs,
                        b_act,
                        b_logp,
                        b_safe_min,
                        b_safe_max,
                    )
                    # Log dummy values for softmax diagnostics (not used in GradS mode)
                    self._logger.store({
                        'MultiLag/RewardAdvFrac': 0.0,
                        'MultiLag/MaxLambda': max(self._get_lambda(i) for i in range(NUM_COSTS)),
                        'MultiLag/SumLambda': sum(self._get_lambda(i) for i in range(NUM_COSTS)),
                        'MultiLag/DominantConstraint': 0.0,
                        'MultiLag/EffectiveTau': self._tau,
                    })
                else:
                    # Standard softmax advantage combination (single backward pass)
                    self._update_actor(
                        b_obs,
                        b_act,
                        b_logp,
                        b_adv_r,
                        b_adv_c,
                        b_safe_min,
                        b_safe_max,
                    )
                    # Log dummy GradS values (not used in softmax mode)
                    self._logger.store({
                        'GradS/SelectedConstraint': -1.0,
                        'GradS/CandidateSetSize':    0.0,
                        'GradS/Scale':               0.0,
                        'GradS/CosSim_C0_C3':        0.0,
                        'GradS/CosSim_C0_C4':        0.0,
                        'GradS/CosSim_C3_C4':        0.0,
                        'GradS/CosSim_reward_C0':    0.0,
                        'GradS/CosSim_reward_C1':    0.0,
                        'GradS/CosSim_reward_C2':    0.0,
                        'GradS/CosSim_reward_C3':    0.0,
                        'GradS/CosSim_reward_C4':    0.0,
                    })

            # R20: Compute KL in chunks to avoid CUDA OOM with STEMS encoder.
            with torch.no_grad():
                _kl_sum = 0.0
                _n = original_obs.shape[0]
                for _s in range(0, _n, _kl_chunk):
                    _e = min(_s + _kl_chunk, _n)
                    _new_d = self._actor_critic.actor(original_obs[_s:_e])
                    _old_d = torch.distributions.Normal(_old_loc[_s:_e], _old_scale[_s:_e])
                    _kl_sum += (
                        torch.distributions.kl.kl_divergence(_old_d, _new_d)
                        .sum(-1).mean().item() * (_e - _s)
                    )
                kl = torch.tensor(_kl_sum / _n, device=original_obs.device)
            kl = distributed.dist_avg(kl)
            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {epoch_i + 1} due to reaching max kl')
                break

        # Clear instance variables
        self._current_batch_adv_cs = None
        self._current_batch_adv_r = None

        self._logger.store({
            'Train/StopIter': update_counts,
            'Value/Adv': adv_r.mean().item(),
            'Train/KL': final_kl,
            'Metrics/LagrangeMultiplier': max(self._get_lambda(i) for i in range(NUM_COSTS)),
        })

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
        """Actor update with optional policy-side masked log-prob."""
        adv = self._compute_adv_surrogate(adv_r=adv_r, adv_c=adv_c)
        loss = self._loss_pi(obs, act, logp, adv, safe_min, safe_max)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.actor_optimizer.step()

    def _update_actor_with_grads(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp_old: torch.Tensor,
        safe_min: torch.Tensor | None = None,
        safe_max: torch.Tensor | None = None,
    ) -> None:
        """Actor update using GradS gradient surgery (R22).

        Replaces the softmax advantage combination with gradient-space surgery:
          1. Compute reward gradient via PPO surrogate backward pass
          2. Compute per-constraint gradients via simple policy gradient
          3. GradSSelector picks a non-conflicting constraint
          4. Apply: g = g_reward + lambda_sel * scale * g_cost_sel

        Uses logp_old from rollout buffer (same as standard PPO importance
        sampling). Per-constraint advantages stored in _current_batch_adv_cs.

        Notes:
          - 6 separate backward passes (1 reward + 5 constraints) per mini-batch
          - Gradient clipping applied to final combined gradient
          - Entropy bonus added to reward loss only
          - PPO clipping applied to reward loss only (cost uses simple PG)
        """
        assert self._current_batch_adv_cs is not None
        lambdas = [self._get_lambda(i) for i in range(NUM_COSTS)]

        clip_eps    = self._cfgs.algo_cfgs.clip
        entropy_coef = self._cfgs.algo_cfgs.entropy_coef
        max_grad_norm = self._cfgs.algo_cfgs.max_grad_norm

        actor = self._actor_critic.actor
        actor_params = list(actor.parameters())

        def _collect_grads() -> torch.Tensor:
            """Flatten all actor param gradients into one vector."""
            grads = []
            for p in actor_params:
                grads.append(p.grad.detach().clone().flatten() if p.grad is not None
                             else torch.zeros(p.numel(), device=obs.device))
            return torch.cat(grads)

        def _apply_flat_grad(g_flat: torch.Tensor) -> None:
            """Write a flat gradient vector back to actor param .grad fields."""
            ptr = 0
            for p in actor_params:
                n = p.numel()
                p.grad = g_flat[ptr:ptr + n].view_as(p).clone()
                ptr += n

        # ---- 1. Reward gradient (PPO surrogate + entropy) ----
        actor.zero_grad()
        dist = actor(obs)
        if self._policy_action_mask and safe_min is not None and safe_max is not None:
            logp = masked_log_prob_from_action(dist, act, safe_min, safe_max)
        else:
            logp = dist.log_prob(act).sum(-1)
        ratio = torch.exp(logp - logp_old)
        adv_r = self._current_batch_adv_cs[0].new_empty(0)  # placeholder
        # Retrieve reward adv from instance (set externally via _current_batch_adv_r)
        adv_r = self._current_batch_adv_r

        surr1 = ratio * adv_r
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_r
        loss_r = -torch.min(surr1, surr2).mean()
        loss_r = loss_r - entropy_coef * dist.entropy().sum(-1).mean()

        # R30: KL penalty toward BC reference policy (GradS path)
        ref_actor = getattr(self, '_bc_ref_actor', None)
        kl_beta = getattr(self, '_kl_beta', 0.0)
        if ref_actor is not None and kl_beta > 0:
            with torch.no_grad():
                ref_dist = ref_actor(obs)
            kl = torch.distributions.kl_divergence(
                dist,
                torch.distributions.Normal(ref_dist.mean, ref_dist.stddev),
            ).sum(-1).mean()
            kl = torch.clamp(kl, max=100.0)
            loss_r = loss_r + kl_beta * kl

        loss_r.backward()
        g_reward = _collect_grads()

        # ---- 2. Per-constraint gradients (simple policy gradient) ----
        cost_grads: list[torch.Tensor] = []
        for i in range(NUM_COSTS):
            actor.zero_grad()
            dist_i = actor(obs)
            if self._policy_action_mask and safe_min is not None and safe_max is not None:
                logp_i = masked_log_prob_from_action(dist_i, act, safe_min, safe_max)
            else:
                logp_i = dist_i.log_prob(act).sum(-1)
            ratio_i = torch.exp(logp_i - logp_old)
            # Simple PG (no PPO clipping on cost side — standard for CMDP)
            loss_ci = -(ratio_i * self._current_batch_adv_cs[i]).mean()
            loss_ci.backward()
            cost_grads.append(_collect_grads())

        actor.zero_grad()

        # ---- Reward-vs-constraint cosine similarities ----
        g_r_np = g_reward.cpu().numpy()
        g_r_norm = np.linalg.norm(g_r_np) + 1e-8
        cos_reward_c = {}
        for i in range(NUM_COSTS):
            g_c_np = cost_grads[i].cpu().numpy()
            g_c_norm = np.linalg.norm(g_c_np) + 1e-8
            cos_reward_c[i] = float(np.dot(g_r_np, g_c_np) / (g_r_norm * g_c_norm))

        # ---- 3. GradS selection ----
        sel_idx, scale, diag = self._grads_selector.select(
            cost_grads, lambdas, self._per_cost_limits
        )

        # ---- 4. Combine and apply ----
        lambda_sel = lambdas[sel_idx]
        g_final = g_reward + lambda_sel * scale * cost_grads[sel_idx]

        _apply_flat_grad(g_final)
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(actor_params, max_grad_norm)
        self._actor_critic.actor_optimizer.step()
        actor.zero_grad()

        # ---- 5. Log GradS diagnostics ----
        cos = diag['cos_matrix']
        self._logger.store({
            'GradS/SelectedConstraint': float(sel_idx),
            'GradS/CandidateSetSize':   float(len(diag['candidate_set'])),
            'GradS/Scale':              float(scale),
            'GradS/CosSim_C0_C3':       float(cos[0, 3]),
            'GradS/CosSim_C0_C4':       float(cos[0, 4]),
            'GradS/CosSim_C3_C4':       float(cos[3, 4]),
            'GradS/CosSim_reward_C0':   cos_reward_c[0],
            'GradS/CosSim_reward_C1':   cos_reward_c[1],
            'GradS/CosSim_reward_C2':   cos_reward_c[2],
            'GradS/CosSim_reward_C3':   cos_reward_c[3],
            'GradS/CosSim_reward_C4':   cos_reward_c[4],
        })

    def _update_per_cost_critic(
        self, obs: torch.Tensor, target_value: torch.Tensor, idx: int
    ) -> None:
        """Update per-constraint cost critic."""
        critic = self._cost_critics[idx]
        optimizer = self._cost_critic_optimizers[idx]

        optimizer.zero_grad()
        loss = nn.functional.mse_loss(critic(obs)[0], target_value)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(critic.parameters(), self._cfgs.algo_cfgs.max_grad_norm)
        optimizer.step()

        self._logger.store({f'Loss/Loss_cost_critic_{idx}': loss.mean().item()})

    def _compute_per_constraint_gae(self) -> list[torch.Tensor]:
        """Compute per-constraint GAE with z-score normalization.

        Returns:
            List of NUM_COSTS tensors, each shape (total_steps,).
        """
        gamma = self._cfgs.algo_cfgs.gamma
        lam_c = self._cfgs.algo_cfgs.lam_c

        all_advs = []
        for i in range(NUM_COSTS):
            costs_i = self._env.per_cost_steps[f'cost_{i}']
            values_i = self._env.per_cost_steps[f'value_c_{i}']
            boundaries = self._env.episode_boundaries

            adv_i = torch.zeros_like(costs_i)
            for start, end, last_values_c in boundaries:
                lvc = last_values_c[i].flatten()[:1]
                path_costs = torch.cat([costs_i[start:end], lvc])
                path_values = torch.cat([values_i[start:end], lvc])

                deltas = path_costs[:-1] + gamma * path_values[1:] - path_values[:-1]
                path_adv = discount_cumsum(deltas, gamma * lam_c)
                adv_i[start:end] = torch.as_tensor(path_adv, dtype=adv_i.dtype)

            # Z-score normalization (critical: matches reward advantage treatment)
            if self._cfgs.algo_cfgs.standardized_cost_adv:
                adv_mean, adv_std = distributed.dist_statistics_scalar(adv_i)[:2]
                adv_i = (adv_i - adv_mean) / (adv_std + 1e-8)

            all_advs.append(adv_i)

        return all_advs

    def _compute_per_constraint_targets(self) -> list[torch.Tensor]:
        """Compute per-constraint TD targets for critic updates.

        Returns:
            List of NUM_COSTS tensors, each shape (total_steps,).
        """
        gamma = self._cfgs.algo_cfgs.gamma
        lam_c = self._cfgs.algo_cfgs.lam_c

        all_targets = []
        for i in range(NUM_COSTS):
            costs_i = self._env.per_cost_steps[f'cost_{i}']
            values_i = self._env.per_cost_steps[f'value_c_{i}']
            boundaries = self._env.episode_boundaries

            targets_i = torch.zeros_like(costs_i)
            for start, end, last_values_c in boundaries:
                lvc = last_values_c[i].flatten()[:1]
                path_costs = torch.cat([costs_i[start:end], lvc])
                path_values = torch.cat([values_i[start:end], lvc])

                deltas = path_costs[:-1] + gamma * path_values[1:] - path_values[:-1]
                path_adv = discount_cumsum(deltas, gamma * lam_c)
                targets_i[start:end] = torch.as_tensor(
                    path_adv, dtype=targets_i.dtype
                ) + path_values[:-1]

            all_targets.append(targets_i)

        return all_targets
