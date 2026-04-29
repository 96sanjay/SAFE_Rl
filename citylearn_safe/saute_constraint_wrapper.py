"""
Generic Sauté MDP wrapper for any constraint.

Implements the Sauté MDP formulation from:
  Sootla et al., "Sauté RL: Almost Surely Safe Reinforcement Learning
  Using State Augmentation", ICML 2022.

Unlike saute_ev_wrapper.py (which is hard-coded to C1/EV dense cost),
this wrapper accepts a configurable cost_key and can target any constraint:
  - "cost_stems_grid_power"      → C4 (grid import limit)
  - "cost_stems_building_power"  → C3 (building power limit)
  - "cost_ev_dense"              → C1 (EV charging, legacy)

The wrapper:
  1. Tracks a normalized safety budget λ_t ∈ (-∞, 1] that depletes as
     cost accumulates:  λ_{t+1} = (λ_t - c_t / d) / γ
  2. Augments observations with λ_{t+1} (clipped to [-1, 1], one extra dim)
  3. Zeros the target cost key in info dict (removes from Lagrangian)
  4. Preserves raw cost under a separate key for logging
  5. Provides saute_{label}_unsafe flag for reward reshaping in CityLearnCMDP
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class SauteConstraintWrapper(gym.Wrapper):
    """Sauté MDP budget wrapper for any single constraint."""

    def __init__(
        self,
        env: gym.Env,
        cost_key: str,
        budget_d: float,
        penalty: float = 5.0,
        gamma: float = 1.0,
        label: str = "c4",
    ):
        super().__init__(env)

        self._cost_key = cost_key
        self._budget_d = budget_d
        self._penalty = penalty
        self._gamma = gamma
        self._label = label

        # Normalized budget: starts at 1.0, depletes toward 0 and below
        self._budget = 1.0

        # Episode-level raw cost tracking (for logging)
        self._ep_raw_cost = 0.0
        self._ep_steps = 0

        # Extend observation space by 1 dim (budget state)
        inner_space = env.observation_space
        low = np.append(inner_space.low, -1.0)
        high = np.append(inner_space.high, 1.0)
        self.observation_space = gym.spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

        print(
            f"[Saute{label.upper()}] Budget wrapper ENABLED: "
            f"cost_key={cost_key}, d={budget_d}, "
            f"penalty={penalty}, gamma={gamma}, "
            f"obs {inner_space.shape} → {self.observation_space.shape}"
        )

    def _augment_obs(self, obs, budget_value):
        """Append clipped budget value to observation."""
        obs_flat = np.asarray(obs, dtype=np.float32).ravel()
        budget_clipped = np.clip(budget_value, -1.0, 1.0)
        return np.append(obs_flat, budget_clipped)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._budget = 1.0
        self._ep_raw_cost = 0.0
        self._ep_steps = 0
        info[f"saute_{self._label}_budget"] = self._budget
        info[f"saute_{self._label}_unsafe"] = 0.0
        return self._augment_obs(obs, self._budget), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Read raw per-step cost for the target constraint
        raw_cost = float(info.get(self._cost_key, 0.0))

        # Budget evolution: λ_{t+1} = (λ_t - c_t / d) / γ
        if self._budget_d > 0:
            self._budget = self._budget - raw_cost / self._budget_d
            if self._gamma != 1.0:
                self._budget /= self._gamma

        # Track raw cost for logging
        self._ep_raw_cost += raw_cost
        self._ep_steps += 1

        # Determine safe/unsafe using POST-depletion budget
        unsafe = self._budget <= 0

        # Preserve raw cost for logging under a separate key
        info[f"saute_{self._label}_raw_cost"] = raw_cost

        # ZERO the cost key — removes this constraint from Lagrangian
        info[self._cost_key] = 0.0

        # Provide flag + penalty for reward reshaping in CityLearnCMDP.step()
        info[f"saute_{self._label}_unsafe"] = 1.0 if unsafe else 0.0
        info[f"saute_{self._label}_penalty"] = self._penalty
        info[f"saute_{self._label}_budget"] = float(self._budget)

        # At episode end, emit episode-total raw cost for TensorBoard
        if terminated or truncated:
            info[f"saute_{self._label}_ep_raw_cost"] = self._ep_raw_cost
            info[f"saute_{self._label}_ep_steps"] = self._ep_steps
            pct = (1.0 - self._budget) * 100 if self._budget_d > 0 else 0.0
            print(
                f"  [Saute{self._label.upper()}] EP END: "
                f"raw_cost={self._ep_raw_cost:.1f} "
                f"budget={self._budget:.4f} "
                f"depleted={pct:.1f}% "
                f"unsafe_at_end={unsafe}"
            )

        # Augment obs with POST-depletion budget
        return self._augment_obs(obs, self._budget), reward, terminated, truncated, info
