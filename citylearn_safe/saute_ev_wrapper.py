"""
Sauté MDP wrapper for EV charging constraint (C1).

Implements the Sauté MDP formulation from:
  Sootla et al., "Sauté RL: Almost Surely Safe Reinforcement Learning
  Using State Augmentation", ICML 2022.

The wrapper:
  1. Tracks a normalized safety budget λ_t ∈ (-∞, 1] that depletes as
     EV charging cost accumulates:  λ_{t+1} = (λ_t - c_t / d) / γ
  2. Augments observations with λ_{t+1} (clipped to [-1, 1], one extra dim)
  3. Eliminates C1 from the Lagrangian: sets cost_ev_dense = 0.0 always
     (the constraint is handled via reward reshaping, not the CMDP)
  4. Provides ev_saute_unsafe flag for reward reshaping in CityLearnCMDP

Per the paper, the CMDP constraint is ELIMINATED by converting it to
state augmentation + reward reshaping. The Lagrangian multiplier for C1
should remain near zero because the Lagrangian sees zero cost.

Reward reshaping (r̃ = r if λ > 0, else unsafe_reward) is applied in
CityLearnCMDP.step() because STEMS reward is computed there, not here.

Gated by CITYLEARN_EV_SAUTE="1" env var. When disabled, this wrapper
is not inserted and everything works as before (backward compat).

Env vars:
  CITYLEARN_EV_SAUTE_BUDGET  - total budget d (default: 1500)
  CITYLEARN_EV_SAUTE_PENALTY - reward penalty per step when unsafe (default: 5.0)
  CITYLEARN_EV_SAUTE_GAMMA   - discount factor for budget (default: 1.0)
"""
from __future__ import annotations

import os
import gymnasium as gym
import numpy as np


class SauteEVBudgetWrapper(gym.Wrapper):
    """Sauté MDP budget wrapper for EV charging (C1 dense cost)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Budget config
        self._budget_d = float(os.environ.get("CITYLEARN_EV_SAUTE_BUDGET", "1500"))
        self._penalty = float(os.environ.get("CITYLEARN_EV_SAUTE_PENALTY", "5.0"))
        self._gamma = float(os.environ.get("CITYLEARN_EV_SAUTE_GAMMA", "1.0"))

        # Normalized budget: starts at 1.0, depletes toward 0 and below
        self._budget = 1.0

        # Extend observation space by 1 dim (budget state)
        inner_space = env.observation_space
        low = np.append(inner_space.low, -1.0)    # budget can go negative, clip to -1
        high = np.append(inner_space.high, 1.0)
        self.observation_space = gym.spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

        print(f"[SauteEV] Budget wrapper ENABLED: d={self._budget_d}, "
              f"penalty={self._penalty}, gamma={self._gamma}, "
              f"obs {inner_space.shape} → {self.observation_space.shape}")

    def _augment_obs(self, obs, budget_value):
        """Append clipped budget value to observation."""
        obs_flat = np.asarray(obs, dtype=np.float32).ravel()
        budget_clipped = np.clip(budget_value, -1.0, 1.0)
        return np.append(obs_flat, budget_clipped)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._budget = 1.0
        info["ev_saute_budget"] = self._budget
        info["ev_saute_unsafe"] = 0.0
        return self._augment_obs(obs, self._budget), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Read raw per-step EV shortfall cost from safety_env
        raw_cost = float(info.get("cost_ev_dense", 0.0))

        # Budget evolution: λ_{t+1} = (λ_t - c_t / d) / γ  (Eq. from paper)
        if self._budget_d > 0:
            self._budget = (self._budget - raw_cost / self._budget_d)
            if self._gamma != 1.0:
                self._budget /= self._gamma

        # Determine safe/unsafe using POST-depletion budget (per OmniSafe)
        unsafe = self._budget <= 0

        # ELIMINATE C1 from Lagrangian: cost passes as 0.0 always
        # The raw cost is only used for budget tracking (above).
        # The constraint is handled via reward reshaping in CityLearnCMDP.
        info["cost_ev_dense"] = 0.0

        # Provide flag + penalty for reward reshaping in CityLearnCMDP.step()
        info["ev_saute_unsafe"] = 1.0 if unsafe else 0.0
        info["ev_saute_penalty"] = self._penalty

        # Diagnostics
        info["ev_saute_budget"] = float(self._budget)
        info["ev_saute_raw_cost"] = float(raw_cost)

        # Augment obs with POST-depletion budget λ_{t+1} (per OmniSafe)
        return self._augment_obs(obs, self._budget), reward, terminated, truncated, info
