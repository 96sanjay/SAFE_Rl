"""Observation wrapper that appends explicit action-feasibility features.

This is intentionally lightweight and isolated from legacy PPO paths.
It exposes the current safe action interval computed by the same logic used in
the policy-side masking path:

    feasibility(obs_t) = [safe_min_t, safe_max_t]

For the 5-building central-agent setup this adds 18 dims (9 safe minima +
9 safe maxima). These are normalized in the same action space as the actor,
so they are already well-scaled for direct policy consumption.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from citylearn_safe.policy_action_mask import CityLearnActionBoundsProvider


class FeasibilityObsWrapper(gym.ObservationWrapper):
    """Append explicit safe-action bounds to the flat observation."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._bounds_provider = CityLearnActionBoundsProvider(env)

        act_space = env.action_space
        if isinstance(act_space, (list, tuple)):
            raise NotImplementedError(
                "FeasibilityObsWrapper expects a flat central-agent action space.",
            )
        self._act_dim = int(act_space.shape[0])
        self.n_extra = 2 * self._act_dim

        orig = env.observation_space
        if isinstance(orig, spaces.Box):
            lo = orig.low.ravel()
            hi = orig.high.ravel()
        else:
            first = orig[0] if isinstance(orig, list) else orig
            lo = np.asarray(first.low).ravel()
            hi = np.asarray(first.high).ravel()

        extra_low = -np.ones(self.n_extra, dtype=np.float32)
        extra_high = np.ones(self.n_extra, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([lo, extra_low]).astype(np.float32),
            high=np.concatenate([hi, extra_high]).astype(np.float32),
            dtype=np.float32,
        )

        print(
            f"[FeasibilityObs] {len(lo)} + {self.n_extra} = "
            f"{len(lo) + self.n_extra} dims",
        )

    def _flatten_obs(self, obs) -> np.ndarray:
        if isinstance(obs, (list, tuple)):
            return np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs])
        return np.asarray(obs, dtype=np.float32).ravel()

    def _get_feasibility_features(self) -> np.ndarray:
        safe_min, safe_max = self._bounds_provider.current_safe_bounds()
        return np.concatenate([safe_min, safe_max]).astype(np.float32)

    def observation(self, obs):
        flat = self._flatten_obs(obs)
        extra = self._get_feasibility_features()
        return np.concatenate([flat, extra]).astype(np.float32)

    def step(self, action):
        step_out = self.env.step(action)
        if len(step_out) == 6:
            obs, reward, cost, terminated, truncated, info = step_out
            return self.observation(obs), reward, cost, terminated, truncated, info
        obs, reward, terminated, truncated, info = step_out
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
