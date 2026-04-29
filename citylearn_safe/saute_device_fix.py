"""Runtime device-mismatch fix for OmniSafe SauteAdapter.

Upstream bug: SauteAdapter._safety_step does not cast `cost` to the device of
`_safety_obs` before the in-place subtract, so training on GPU with a CPU
tensor-returning env crashes at step 1.

This module patches `_safety_step` to coerce `cost` onto `_safety_obs.device`.
It is a no-op on CPU-only runs.

Usage: import this module once before any code calls omnisafe.Agent(...).
"""
from __future__ import annotations

import torch

from omnisafe.adapter.saute_adapter import SauteAdapter


_ORIGINAL_SAFETY_STEP = SauteAdapter._safety_step
_ORIGINAL_SAFETY_REWARD = SauteAdapter._safety_reward
_ORIGINAL_STEP = SauteAdapter.step
_ORIGINAL_AUGMENT_OBS = SauteAdapter._augment_obs


def _safety_step_device_safe(self: SauteAdapter, cost: torch.Tensor) -> None:
    if cost.device != self._safety_obs.device:
        cost = cost.to(self._safety_obs.device)
    self._safety_obs -= cost.unsqueeze(-1) / self._safety_budget
    self._safety_obs /= self._cfgs.algo_cfgs.saute_gamma


def _safety_reward_device_safe(self: SauteAdapter, reward: torch.Tensor) -> torch.Tensor:
    safe = torch.as_tensor(
        self._safety_obs > 0, dtype=reward.dtype, device=reward.device
    ).squeeze(-1)
    return safe * reward + (1 - safe) * self._cfgs.algo_cfgs.unsafe_reward


def _step_device_safe(self: SauteAdapter, action: torch.Tensor):  # type: ignore[override]
    """Wrap the parent step, coercing all returned tensors to self._device before use."""
    from typing import Any
    next_obs, reward, cost, terminated, truncated, info = self._env.step(action)
    info['original_reward'] = reward

    self._safety_step(cost)
    reward = self._safety_reward(reward)

    # coerce `done` to the same device as _safety_obs before arithmetic
    done = torch.logical_or(terminated, truncated).float().unsqueeze(-1).float()
    done = done.to(self._safety_obs.device)
    self._safety_obs = self._safety_obs * (1 - done) + done

    augmented_obs = self._augment_obs(next_obs)

    if 'final_observation' in info:
        info['final_observation'] = self._augment_obs(info['final_observation'])

    return augmented_obs, reward, cost, terminated, truncated, info


def _augment_obs_device_safe(self: SauteAdapter, obs: torch.Tensor) -> torch.Tensor:
    # Ensure obs is on the same device as _safety_obs before concatenating
    if obs.device != self._safety_obs.device:
        obs = obs.to(self._safety_obs.device)
    return torch.cat([obs, self._safety_obs], dim=-1)


SauteAdapter._safety_step = _safety_step_device_safe
SauteAdapter._safety_reward = _safety_reward_device_safe
SauteAdapter.step = _step_device_safe
SauteAdapter._augment_obs = _augment_obs_device_safe
