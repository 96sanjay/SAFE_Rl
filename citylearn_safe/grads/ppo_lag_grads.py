"""PPOLag + GradS: Per-constraint lambdas with Gradient Shaping.

Uses 5 independent Lagrange multipliers (one per constraint) matching
Yao et al. 2024 (L4DC). Each constraint has its own lambda, cost critic,
and cost limit. GradS selects which constraint gradient to apply using
candidate set construction with cosine similarity filtering, uniform
sampling, and |G|/N scaling (paper-faithful, Algorithm 1).

Paper-faithful implementation:
  - Lambda pre-scaling: gᵢ = λᵢ·∇Vcᵢ before cosine similarity (Alg 1, line 2)
  - Candidate set: -σ < sim(i,j) < κ (Alg 1, line 6)
  - Uniform sampling from candidate set (Alg 1, line 8)
  - Scale: ∇Jc = gc·|G|/N (Alg 1, line 9)
  - No 1/(1+λ) denominator (not in paper)

Based on Yao et al. 2024 (L4DC) Gradient Shaping algorithm.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.lagrange import Lagrange
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.utils import distributed
from omnisafe.utils.math import discount_cumsum

from citylearn_safe.grads.grads_selector import GradSSelector
from citylearn_safe.pid_lagrange import PIDLagrange
from citylearn_safe.policy_action_mask import (
    CityLearnActionBoundsProvider,
    masked_action_from_pretanh,
    masked_log_prob_from_action,
)

# Per-constraint cost keys in the info dict (from safety_env)
COST_KEYS = [
    'cost_ev_departure',          # C1: EV departure deficit (sparse, at departure)
    'cost_ev_dense',              # C1d: EV charging incentive (dense, every step)
    'cost_stems_battery',         # C2: Battery SoC band violation
    'cost_stems_building_power',  # C3: Building power capacity
    'cost_stems_grid_power',      # C4: Grid power capacity
]
NUM_COSTS = len(COST_KEYS)


@registry.register
class PPOLagGradS(PPOLag):
    """PPOLag with GradS gradient shaping for multi-constraint CMDPs.

    Uses 5 independent Lagrange multipliers (per-constraint) with
    paper-faithful GradS: lambda pre-scaling, cosine similarity
    filtering, uniform sampling, and |G|/N scaling (Yao et al. 2024).
    Adds per-constraint cost critics and uses GradS to select which
    constraint gradient to apply at each mini-batch update step.
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

        # Add per-constraint cost critics (same architecture as the combined one)
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

        print(f"[GradS] Added {NUM_COSTS} per-constraint cost critics")

    def _init(self) -> None:
        """Initialize buffer, GradS selector, and per-constraint Lagrange multipliers."""
        super()._init()

        # GradS selector
        grads_cfgs = getattr(self._cfgs, 'grads_cfgs', None)
        assert grads_cfgs is not None, (
            "grads_cfgs not found in config -- check YAML parsing. "
            "GradS requires per-constraint cost limits under grads_cfgs."
        )
        sim_thresh = float(getattr(grads_cfgs, 'sim_threshold', 0.8)) if grads_cfgs else 0.8
        conflict_thresh = float(getattr(grads_cfgs, 'conflict_threshold', 0.999)) if grads_cfgs else 0.999
        sampling = str(getattr(grads_cfgs, 'sampling', 'uniform')) if grads_cfgs else 'uniform'

        self._grads = GradSSelector(
            num_costs=NUM_COSTS,
            sim_threshold=sim_thresh,
            conflict_threshold=conflict_thresh,
            sampling=sampling,
        )

        # v3: gradient normalization flag
        self._normalize_cost_grad = bool(
            getattr(grads_cfgs, 'normalize_cost_grad', False)
        ) if grads_cfgs else False

        # v3: z-score per-constraint cost advantages (vs mean-center only)
        self._zscore_cost_adv = bool(
            getattr(grads_cfgs, 'zscore_cost_adv', False)
        ) if grads_cfgs else False

        # Per-constraint cost limits from grads_cfgs
        self._per_cost_limits = [
            float(getattr(grads_cfgs, f'cost_limit_{i}', 5000.0)) if grads_cfgs else 5000.0
            for i in range(NUM_COSTS)
        ]

        # Per-constraint Lagrange multipliers
        # Supports PID (same as PPOLagMulti) when CITYLEARN_PID_LAGRANGE=1
        lag_cfgs = self._cfgs.lagrange_cfgs
        upper_bound = getattr(lag_cfgs, 'lagrangian_upper_bound', None)
        if upper_bound is not None:
            upper_bound = float(upper_bound)

        self._use_pid = os.environ.get("CITYLEARN_PID_LAGRANGE", "0") == "1"

        if self._use_pid:
            # PID Lagrangian (Stooke et al., ICML 2020)
            # PID params from grads_cfgs (same field names as multi_cfgs)
            pid_kp_default = float(getattr(grads_cfgs, 'pid_kp', 0.1))
            pid_ki_default = float(getattr(grads_cfgs, 'pid_ki', 0.01))
            pid_kd_default = float(getattr(grads_cfgs, 'pid_kd', 0.01))
            pid_d_delay = int(getattr(grads_cfgs, 'pid_d_delay', 10))
            pid_ema_p_default = float(getattr(grads_cfgs, 'pid_delta_p_ema_alpha', 0.95))
            pid_ema_d_default = float(getattr(grads_cfgs, 'pid_delta_d_ema_alpha', 0.95))
            penalty_max = float(getattr(lag_cfgs, 'lagrangian_upper_bound', 3.0))
            init_val = float(getattr(lag_cfgs, 'lagrangian_multiplier_init', 0.001))

            self._per_lagranges = []
            for i in range(NUM_COSTS):
                # Per-constraint overrides (e.g., pid_kp_0, pid_ki_3)
                kp = float(getattr(grads_cfgs, f'pid_kp_{i}', pid_kp_default))
                ki = float(getattr(grads_cfgs, f'pid_ki_{i}', pid_ki_default))
                kd = float(getattr(grads_cfgs, f'pid_kd_{i}', pid_kd_default))
                ema_p = float(getattr(grads_cfgs, f'pid_ema_p_{i}', pid_ema_p_default))
                ema_d = float(getattr(grads_cfgs, f'pid_ema_d_{i}', pid_ema_d_default))
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
            print(f"[GradS] PID Lagrangian enabled (Kp={pid_kp_default}, Ki={pid_ki_default})")
        else:
            self._per_lagranges: list[Lagrange] = []
            for i in range(NUM_COSTS):
                self._per_lagranges.append(Lagrange(
                    cost_limit=self._per_cost_limits[i],
                    lagrangian_multiplier_init=float(lag_cfgs.lagrangian_multiplier_init),
                    lambda_lr=float(lag_cfgs.lambda_lr),
                    lambda_optimizer=str(lag_cfgs.lambda_optimizer),
                    lagrangian_upper_bound=upper_bound,
                ))
            print(f"[GradS] SGD Lagrangian (lr={lag_cfgs.lambda_lr})")

        print(f"[GradS] sim_threshold={sim_thresh}, conflict_threshold={conflict_thresh}")
        print(f"[GradS] sampling={sampling}, normalize_cost_grad={self._normalize_cost_grad}")
        print(f"[GradS] zscore_cost_adv={self._zscore_cost_adv}")
        print(f"[GradS] per-constraint cost limits: {self._per_cost_limits}")
        for i in range(NUM_COSTS):
            lam = self._per_lagranges[i].lagrangian_multiplier
            lam_val = float(lam) if isinstance(lam, (int, float)) else lam.item()
            print(f"[GradS]   Lambda_{i}: init={lam_val:.4f}, limit={self._per_cost_limits[i]:.0f}")

    def _init_log(self) -> None:
        """Register GradS-specific log keys."""
        super()._init_log()
        self._logger.register_key('GradS/CandidateSetSize')
        self._logger.register_key('GradS/SelectedConstraint')
        self._logger.register_key('GradS/Scale')
        self._logger.register_key('GradS/Anchor')
        # Cosine similarities between key conflicting pairs
        self._logger.register_key('GradS/CosSim_C0_C3')
        self._logger.register_key('GradS/CosSim_C0_C4')
        self._logger.register_key('GradS/CosSim_C3_C4')
        # Gradient diagnostics
        self._logger.register_key('GradS/GradNorm_reward')
        self._logger.register_key('GradS/GradNorm_selected')
        self._logger.register_key('GradS/EffectiveLambdaScale')
        self._logger.register_key('GradS/CostRewardGradRatio')
        self._logger.register_key('GradS/GradNorm_final')
        # Per-constraint candidate membership frequency
        for i in range(NUM_COSTS):
            self._logger.register_key(f'GradS/InCandidate_{i}')
        for i in range(NUM_COSTS):
            self._logger.register_key(f'Loss/Loss_cost_critic_{i}')
            self._logger.register_key(f'Metrics/EpCost_{i}')
            self._logger.register_key(f'Metrics/Lambda_{i}')
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
        self._logger.register_key('Reward/r_traj_batt')
        self._logger.register_key('Reward/r_traj_ev')
        self._logger.register_key('Reward/traj_forecast_signal')
        self._logger.register_key('Reward/traj_forecast_mean')

    def _update(self) -> None:
        """Update with GradS: per-constraint lambdas + gradient selection."""
        # 1. Update parent's single lambda (kept for inheritance compatibility)
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'
        self._lagrange.update_lagrange_multiplier(Jc)

        # 2. Update per-constraint lambdas independently + log
        for i in range(NUM_COSTS):
            ep_cost_i = self._env.get_per_constraint_ep_cost(i)
            if self._use_pid:
                self._per_lagranges[i].pid_update(ep_cost_i)
            else:
                self._per_lagranges[i].update_lagrange_multiplier(ep_cost_i)
            lam = self._per_lagranges[i].lagrangian_multiplier
            lam_val = float(lam) if isinstance(lam, (int, float)) else lam.item()
            self._logger.store({
                f'Metrics/EpCost_{i}': ep_cost_i,
                f'Metrics/Lambda_{i}': lam_val,
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

        # 5. Build per-constraint target values for critic updates
        per_cost_targets = self._compute_per_constraint_targets()

        # 6. Create DataLoader with all data
        tensors = [obs, act, logp, target_value_r, target_value_c, adv_r]
        for i in range(NUM_COSTS):
            tensors.append(per_cost_advs[i])
            tensors.append(per_cost_targets[i])

        original_obs = obs.to(self._device)
        old_distribution = self._actor_critic.actor(original_obs)

        dataloader = DataLoader(
            dataset=TensorDataset(*tensors),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        for epoch_i in track(
            range(self._cfgs.algo_cfgs.update_iters), description='Updating (GradS)...'
        ):
            for batch in dataloader:
                b_obs = batch[0].to(self._device)
                b_act = batch[1].to(self._device)
                b_logp = batch[2].to(self._device)
                b_target_value_r = batch[3].to(self._device)
                b_target_value_c = batch[4].to(self._device)
                b_adv_r = batch[5].to(self._device)
                b_adv_cs = []
                b_target_cs = []
                for i in range(NUM_COSTS):
                    b_adv_cs.append(batch[6 + 2 * i].to(self._device))
                    b_target_cs.append(batch[6 + 2 * i + 1].to(self._device))

                # Update reward critic (standard)
                self._update_reward_critic(b_obs, b_target_value_r)

                # Update combined cost critic (standard — for lambda)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(b_obs, b_target_value_c)

                # Update per-constraint cost critics
                for i in range(NUM_COSTS):
                    self._update_per_cost_critic(b_obs, b_target_cs[i], i)

                # GradS actor update (lambda-proportional sampling + rescaling)
                self._update_actor_grads(b_obs, b_act, b_logp, b_adv_r, b_adv_cs)

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
                self._logger.log(f'Early stopping at iter {epoch_i + 1} due to reaching max kl')
                break

        self._logger.store({
            'Train/StopIter': update_counts,
            'Value/Adv': adv_r.mean().item(),
            'Train/KL': final_kl,
            'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
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

    def _update_actor_grads(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_cs: list[torch.Tensor],
    ) -> None:
        """GradS actor update: per-constraint lambdas, lambda-proportional sampling, rescaling."""
        # Forward pass (shared)
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)
        clip = self._cfgs.algo_cfgs.clip

        # Clipped ratio for PPO
        ratio_clipped = torch.clamp(ratio, 1 - clip, 1 + clip)

        # 1. Reward gradient
        loss_r = -torch.min(ratio * adv_r, ratio_clipped * adv_r).mean()
        loss_r -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

        self._actor_critic.actor_optimizer.zero_grad()
        loss_r.backward(retain_graph=True)
        grad_r = self._flatten_actor_grad()

        # 2. Per-constraint gradients
        cost_grads = []
        for i in range(NUM_COSTS):
            self._actor_critic.actor_optimizer.zero_grad()
            loss_ci = torch.min(
                ratio * adv_cs[i], ratio_clipped * adv_cs[i]
            ).mean()
            retain = (i < NUM_COSTS - 1)
            loss_ci.backward(retain_graph=retain)
            cost_grads.append(self._flatten_actor_grad())

        # 3. GradS: lambda-proportional sampling (random permutation for fair access)
        lambdas = [
            float(lag.lagrangian_multiplier)
            if isinstance(lag.lagrangian_multiplier, (int, float))
            else lag.lagrangian_multiplier.item()
            for lag in self._per_lagranges
        ]
        selected_idx, scale, diag = self._grads.select(
            cost_grads=cost_grads,
            lambdas=lambdas,
            cost_limits=self._per_cost_limits,
        )

        # 4. Paper-faithful gradient combination (Yao et al. 2024)
        #    Paper: ∇J = -∇Vr + ∇Jc  where ∇Jc = gc·|G|/N
        #    In gradient domain (loss_r has negative sign, loss_ci has positive sign):
        #      g_final = grad_r + λ_sel · scale · cost_grad_sel
        #    NO 1/(1+λ) denominator — that was from OmniSafe, not the paper.
        lambda_sel = lambdas[selected_idx]
        cost_grad_sel = cost_grads[selected_idx]

        # Optional: normalize cost gradient magnitude to match reward gradient.
        if self._normalize_cost_grad:
            cost_norm = cost_grad_sel.norm()
            reward_norm = grad_r.norm()
            if cost_norm > 1e-8 and reward_norm > 1e-8:
                cost_grad_sel = cost_grad_sel * (reward_norm / cost_norm)

        g_final = grad_r + lambda_sel * scale * cost_grad_sel

        # 5. Apply combined gradient to actor
        self._actor_critic.actor_optimizer.zero_grad()
        self._set_actor_grad(g_final)

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

        # Log
        entropy = distribution.entropy().mean().item()
        cos = diag['cos_matrix']
        candidate_set = diag['candidate_set']
        log_dict = {
            'Train/Entropy': entropy,
            'Train/PolicyRatio': ratio,
            'Train/PolicyStd': std,
            'Loss/Loss_pi': loss_r.mean().item(),
            'GradS/CandidateSetSize': len(candidate_set),
            'GradS/SelectedConstraint': selected_idx,
            'GradS/Scale': scale,
            'GradS/Anchor': diag['anchor'],
            'GradS/CosSim_C0_C3': float(cos[0, 3]),
            'GradS/CosSim_C0_C4': float(cos[0, 4]),
            'GradS/CosSim_C3_C4': float(cos[3, 4]),
            'GradS/GradNorm_reward': float(grad_r.norm()),
            'GradS/GradNorm_selected': float(cost_grads[selected_idx].norm()),  # raw (pre-normalization)
            'GradS/EffectiveLambdaScale': lambda_sel * scale,
            # Effective ratio = actual cost contribution / reward contribution in g_final
            # Uses cost_grad_sel (possibly normalized) to reflect what was actually applied
            'GradS/CostRewardGradRatio': float(
                (cost_grad_sel.norm() * lambda_sel * scale) / (grad_r.norm() + 1e-8)
            ),
            'GradS/GradNorm_final': float(g_final.norm()),
        }
        for i in range(NUM_COSTS):
            log_dict[f'GradS/InCandidate_{i}'] = 1.0 if i in candidate_set else 0.0
        self._logger.store(log_dict)

    def _flatten_actor_grad(self) -> torch.Tensor:
        """Flatten all actor parameter gradients into a single vector."""
        grads = []
        for p in self._actor_critic.actor.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)

    def _set_actor_grad(self, flat_grad: torch.Tensor) -> None:
        """Set actor parameter gradients from a flattened vector."""
        offset = 0
        for p in self._actor_critic.actor.parameters():
            numel = p.numel()
            p.grad = flat_grad[offset:offset + numel].view_as(p).clone()
            offset += numel

    def _compute_per_constraint_gae(self) -> list[torch.Tensor]:
        """Compute per-constraint GAE from adapter's stored per-step data.

        Returns:
            List of NUM_COSTS tensors, each shape (total_steps,).
        """
        gamma = self._cfgs.algo_cfgs.gamma
        lam_c = self._cfgs.algo_cfgs.lam_c

        all_advs = []
        for i in range(NUM_COSTS):
            costs_i = self._env.per_cost_steps[f'cost_{i}']     # (steps,)
            values_i = self._env.per_cost_steps[f'value_c_{i}']  # (steps,)
            boundaries = self._env.episode_boundaries             # [(start, end, last_vc), ...]

            adv_i = torch.zeros_like(costs_i)
            for start, end, last_values_c in boundaries:
                lvc = last_values_c[i].flatten()[:1]  # ensure shape (1,)
                path_costs = torch.cat([costs_i[start:end], lvc])
                path_values = torch.cat([values_i[start:end], lvc])

                deltas = path_costs[:-1] + gamma * path_values[1:] - path_values[:-1]
                path_adv = discount_cumsum(deltas, gamma * lam_c)
                adv_i[start:end] = torch.as_tensor(path_adv, dtype=adv_i.dtype)

            # Standardize cost advantages
            if self._cfgs.algo_cfgs.standardized_cost_adv:
                adv_mean, adv_std = distributed.dist_statistics_scalar(adv_i)[:2]
                if self._zscore_cost_adv:
                    # v3: Full z-score (matches reward advantage normalization)
                    # This is critical — without it, cost gradients are ~9x larger
                    # than reward gradients due to raw cost magnitude differences
                    adv_i = (adv_i - adv_mean) / (adv_std + 1e-8)
                else:
                    # v2 behavior: mean-center only (same as OmniSafe default)
                    adv_i = adv_i - adv_mean

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
                lvc = last_values_c[i].flatten()[:1]  # ensure shape (1,)
                path_costs = torch.cat([costs_i[start:end], lvc])
                path_values = torch.cat([values_i[start:end], lvc])

                deltas = path_costs[:-1] + gamma * path_values[1:] - path_values[:-1]
                path_adv = discount_cumsum(deltas, gamma * lam_c)
                targets_i[start:end] = torch.as_tensor(path_adv, dtype=targets_i.dtype) + path_values[:-1]

            all_targets.append(targets_i)

        return all_targets


class _MultiCostAdapter(OnPolicyAdapter):
    """Extended adapter that tracks per-constraint costs during rollout.

    Stores per-constraint costs and per-constraint critic values at each step,
    plus episode boundaries for GAE computation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.per_cost_steps: dict[str, torch.Tensor] = {}
        self.episode_boundaries: list[tuple[int, int, list[torch.Tensor]]] = []
        self._per_ep_costs: list[float] = [0.0] * NUM_COSTS
        self._last_completed_ep_costs: list[float] = [0.0] * NUM_COSTS
        self._aux_targets_enabled = int(os.environ.get("STEMS_AUX_TARGETS", "0")) > 0
        self.aux_targets_steps: torch.Tensor | None = None

    def get_per_constraint_ep_cost(self, idx: int) -> float:
        """Return the per-constraint cost from the last completed episode."""
        return self._last_completed_ep_costs[idx]

    def rollout(
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Standard rollout + per-constraint cost tracking."""
        self._reset_log()

        # Get per-constraint cost critics from the algorithm
        # (they're stored on the PPOLagGradS instance, accessible via agent)
        # We'll access them via a stored reference set by _init_env
        cost_critics = getattr(self, '_cost_critics_ref', None)

        # Initialize per-constraint storage
        self.per_cost_steps = {
            f'cost_{i}': torch.zeros(steps_per_epoch, device=torch.device('cpu'))
            for i in range(NUM_COSTS)
        }
        self.per_cost_steps.update({
            f'value_c_{i}': torch.zeros(steps_per_epoch, device=torch.device('cpu'))
            for i in range(NUM_COSTS)
        })
        self.episode_boundaries = []
        self._per_ep_costs = [0.0] * NUM_COSTS
        path_start = 0
        policy_action_mask = bool(getattr(self, '_policy_action_mask', False))
        bounds_provider = None
        if policy_action_mask:
            if self._cfgs.train_cfgs.vector_env_nums != 1:
                raise NotImplementedError(
                    "CITYLEARN_POLICY_ACTION_MASK currently supports vector_env_nums=1 only.",
                )
            bounds_provider = CityLearnActionBoundsProvider(self)

        obs, _ = self.reset()
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            if policy_action_mask:
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
            else:
                act, value_r, value_c, logp = agent.step(obs)
                safe_min = safe_max = None

            # Get per-constraint critic values
            if cost_critics is not None:
                with torch.no_grad():
                    for i, critic in enumerate(cost_critics):
                        vc_i = critic(obs)[0]
                        # obs is (1, obs_dim), vc_i is (1,)
                        self.per_cost_steps[f'value_c_{i}'][step] = vc_i.cpu().squeeze()

            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            # Extract per-constraint costs from info dict
            # On terminal steps, AutoReset moves real info to info['final_info']
            cost_info = info.get('final_info', info) if info.get('final_info') is not None else info
            # Apply per-constraint cost weights (set by PPOLagMulti._init)
            cost_weights = getattr(self, '_cost_weights', None) or [1.0] * NUM_COSTS
            for i, key in enumerate(COST_KEYS):
                val = cost_info.get(key, 0.0)
                if isinstance(val, torch.Tensor):
                    val = val.item()
                raw_val = float(val)
                # Weighted cost → critic training signal (scales actor gradient)
                self.per_cost_steps[f'cost_{i}'][step] = raw_val * cost_weights[i]
                # Raw cost → lambda PID updates (compared against cost_limits)
                self._per_ep_costs[i] += raw_val

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            # Log per-reward-term and trajectory diagnostics from env info
            _reward_info_keys = [
                'r_eco', 'r_sg', 'r_sb', 'r_ramp', 'r_ren', 'r_ev',
                'r_ev_guard', 'r_v2g_ctx', 'r_peak_shave', 'r_load_shift',
                'r_grid_mild', 'r_barrier', 'r_ev_solar', 'r_solar_store',
                'r_ev_slack_arb', 'r_headroom', 'r_grid_penalty', 'r_price_arb',
                'r_nec_sign', 'r_trajectory', 'r_ev_smart',
                'r_traj_batt', 'r_traj_ev', 'traj_forecast_signal', 'traj_forecast_mean',
            ]
            for rk in _reward_info_keys:
                val = cost_info.get(rk, 0.0)
                if isinstance(val, torch.Tensor):
                    val = val.item()
                logger.store({f'Reward/{rk}': float(val)})

            buffer.store(
                obs=obs, act=act, reward=reward, cost=cost,
                value_r=value_r, value_c=value_c, logp=logp,
                **({'safe_min': safe_min, 'safe_max': safe_max} if policy_action_mask else {}),
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    last_values_c = [torch.zeros(1) for _ in range(NUM_COSTS)]

                    if not done:
                        if epoch_end:
                            logger.log(
                                f'Warning: trajectory cut off when rollout by epoch '
                                f'at {self._ep_len[idx]} steps.',
                            )
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                            if cost_critics is not None:
                                with torch.no_grad():
                                    obs_single = obs[idx].unsqueeze(0)
                                    for i, critic in enumerate(cost_critics):
                                        last_values_c[i] = critic(obs_single)[0].cpu().squeeze()
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx],
                            )
                            if cost_critics is not None:
                                with torch.no_grad():
                                    obs_single = info['final_observation'][idx].unsqueeze(0)
                                    for i, critic in enumerate(cost_critics):
                                        last_values_c[i] = critic(obs_single)[0].cpu().squeeze()
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)
                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0
                        self._last_completed_ep_costs = list(self._per_ep_costs)
                        self._per_ep_costs = [0.0] * NUM_COSTS

                    # Record episode boundary for per-constraint GAE
                    self.episode_boundaries.append(
                        (path_start, step + 1, last_values_c)
                    )
                    path_start = step + 1

                    buffer.finish_path(last_value_r, last_value_c, idx)
