"""PPO-Lag with PID Lagrangian and BC warm-start for temperature cooling-only case study.

Fixes all 5 root causes of PPOLagTempMasked's failure:
1. 3D cooling-only action space (via env_id config)
2. RBC behavioral cloning warm-start
3. PID Lagrangian (P-term reacts even when adv_c ~ 0)
4. Aggressive PID gains for fast lambda response
5. 40-epoch training budget to match CSAC-LB comparison

Inherits masked rollout, buffer, loss_pi, and update_actor from PPOLagTempMasked.
Replaces OmniSafe's Adam-based Lagrange with PIDLagrange from pid_lagrange.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.utils import distributed

import citylearn_safe.omni_env_temp_cooling_only  # noqa: F401 - registers cooling-only env
from citylearn_safe.ppo_lag_temp_masked import PPOLagTempMasked, _find_policy_mask_hook
from citylearn_safe.pid_lagrange import PIDLagrange
from citylearn_safe.policy_action_mask_temp import (
    CityLearnTempActionBoundsProvider,
    masked_action_from_pretanh,
    masked_log_prob_from_action,
)


@registry.register
class PPOLagTempCoolingOnly(PPOLagTempMasked):
    """PPO-Lag with PID Lagrangian and BC warm-start for temperature control.

    Key differences from PPOLagTempMasked:
    - Replaces OmniSafe Lagrange with PIDLagrange (fixes cost advantage collapse)
    - Adds behavior_clone_warmstart() from RBC teacher (fixes random init)
    - Env is cooling-only 3D via config (fixes wasted 9D capacity)
    """

    def _init(self) -> None:
        super()._init()

        # --- Replace OmniSafe Lagrange with PIDLagrange ---
        pid_cfgs = getattr(self._cfgs, "pid_lagrange_cfgs", None)

        cost_limit = float(
            getattr(pid_cfgs, "cost_limit", self._cfgs.lagrange_cfgs.cost_limit)
            if pid_cfgs
            else self._cfgs.lagrange_cfgs.cost_limit
        )
        pid_kp = float(getattr(pid_cfgs, "pid_kp", 0.5)) if pid_cfgs else 0.5
        pid_ki = float(getattr(pid_cfgs, "pid_ki", 0.05)) if pid_cfgs else 0.05
        pid_kd = float(getattr(pid_cfgs, "pid_kd", 0.01)) if pid_cfgs else 0.01
        pid_d_delay = int(getattr(pid_cfgs, "pid_d_delay", 10)) if pid_cfgs else 10
        pid_delta_p_ema_alpha = (
            float(getattr(pid_cfgs, "pid_delta_p_ema_alpha", 0.95)) if pid_cfgs else 0.95
        )
        pid_delta_d_ema_alpha = (
            float(getattr(pid_cfgs, "pid_delta_d_ema_alpha", 0.95)) if pid_cfgs else 0.95
        )
        penalty_max = float(getattr(pid_cfgs, "penalty_max", 5.0)) if pid_cfgs else 5.0
        lagrangian_multiplier_init = (
            float(getattr(pid_cfgs, "lagrangian_multiplier_init", 0.001)) if pid_cfgs else 0.001
        )
        normalize_by_limit = (
            bool(getattr(pid_cfgs, "normalize_by_limit", True)) if pid_cfgs else True
        )

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

    # ---- BC Warm-Start (from PPOTempMasked) ----

    def _find_teacher_env(self):
        """Walk env wrapper chain to find the cooling-only env with _rbc_cooling_action()."""
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
        raise RuntimeError(
            "Could not find cooling-only env with _rbc_cooling_action() for BC warm-start."
        )

    def behavior_clone_warmstart(
        self,
        bc_epochs: int,
        bc_rollout_steps: int | None = None,
        bc_batch_size: int = 512,
        seed: int | None = None,
    ) -> None:
        """Clone RBC cooling actions to warm-start the policy."""
        if bc_epochs <= 0:
            return

        teacher_env = self._find_teacher_env()
        hook_source, hook = _find_policy_mask_hook(self._env)
        bounds_provider = (
            hook() if hook is not None else CityLearnTempActionBoundsProvider(hook_source)
        )

        rollout_steps = int(bc_rollout_steps or self._steps_per_epoch)
        obs, _ = self._env.reset(seed=self._seed if seed is None else seed)
        obs_list: list[torch.Tensor] = []
        act_list: list[torch.Tensor] = []
        safe_min_list: list[torch.Tensor] = []
        safe_max_list: list[torch.Tensor] = []

        for _ in range(rollout_steps):
            teacher_action = np.asarray(
                teacher_env._rbc_cooling_action(), dtype=np.float32
            ).reshape(-1)
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
            torch.stack(obs_list),
            torch.stack(act_list),
            torch.stack(safe_min_list),
            torch.stack(safe_max_list),
        )
        dataloader = DataLoader(dataset=dataset, batch_size=bc_batch_size, shuffle=True)
        self._logger.log(
            f"Starting BC warm-start: epochs={bc_epochs}, "
            f"samples={len(dataset)}, batch_size={bc_batch_size}",
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
            self._logger.log(
                f"BC warm-start epoch {epoch + 1}/{bc_epochs}: loss={mean_loss:.6f}"
            )

    # ---- PID Lagrangian Update ----

    def _update(self) -> None:
        """Update with PID Lagrangian instead of OmniSafe's Adam Lagrange."""
        # 1. Get cost and update PID lambda
        Jc = self._logger.get_stats("Metrics/EpCost")[0]
        assert not np.isnan(Jc), "cost for updating lagrange multiplier is nan"
        self._pid_lagrange.pid_update(Jc)

        # Also update OmniSafe's Lagrange for compatibility
        self._lagrange.update_lagrange_multiplier(Jc)

        # 2. Run the masked PPO update loop
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
                self._update_actor(
                    obs_b, act_b, logp_b, adv_r_b, adv_c_b, safe_min_b, safe_max_b
                )

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

        # 3. Log PID-specific metrics
        pid_lambda = self._pid_lagrange.lagrangian_multiplier
        self._logger.store(
            {
                "Train/StopIter": update_counts,
                "Value/Adv": adv_r.mean().item(),
                "Train/KL": final_kl,
                "Metrics/LagrangeMultiplier": pid_lambda,
                "Metrics/PID_Lambda": pid_lambda,
                "Metrics/PID_I": self._pid_lagrange._pid_i,
                "Metrics/PID_P": self._pid_lagrange._delta_p,
            },
        )

    def _compute_adv_surrogate(
        self, adv_r: torch.Tensor, adv_c: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined advantage using PID lambda."""
        penalty = self._pid_lagrange.lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1.0 + penalty)


__all__ = ["PPOLagTempCoolingOnly"]
