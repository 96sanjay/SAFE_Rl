"""Wrapper that appends projector state features.

This env family uses a 6-value step signature, so we keep an explicit wrapper
instead of relying on Gym's ObservationWrapper step plumbing.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class ProjectorStateObsWrapper(gym.Wrapper):
    """Append the projector state tensor to the environment observation."""

    def __init__(self, env: gym.Env, projector: Any):
        super().__init__(env)
        self._projector = projector
        base_shape = tuple(env.observation_space.shape)
        state_dim = int(projector.state_tensor_dim)
        low = np.full((base_shape[0] + state_dim,), -np.inf, dtype=np.float32)
        high = np.full((base_shape[0] + state_dim,), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, observation):
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        proj_state = self._projector.extract_state_tensor().detach().cpu().numpy().astype(np.float32, copy=False)
        return np.concatenate([obs, proj_state], axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, cost, terminated, truncated, info
