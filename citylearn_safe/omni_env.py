from __future__ import annotations

from typing import Any, Tuple
import os
import numpy as np
import torch
import gymnasium as gym

from omnisafe.envs.core import CMDP, env_register

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper
from citylearn_safe.adapters import SingleAgentListAdapter
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3


@env_register
class CityLearnCMDP(CMDP):
    """Expose CityLearnSafetyEnvV3 to OmniSafe CMDP interface."""
    _support_envs = ['CityLearnSafety-SoC-v0']
    need_time_limit_wrapper: bool = False
    need_auto_reset_wrapper: bool = True

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)

        schema = os.environ.get("CITYLEARN_SCHEMA", "")
        if not schema:
            raise RuntimeError("CITYLEARN_SCHEMA is not set.")
        if not os.path.exists(schema):
            raise FileNotFoundError(f"CITYLEARN_SCHEMA not found: {schema}")

        # Build the same pipeline you already know works
        base: gym.Env = CityLearnEnv(schema=schema, central_agent=True)
        base = NormalizedObservationWrapper(base)
        base = SingleAgentListAdapter(base)

        # Safety wrapper
        env: gym.Env = CityLearnSafetyEnvV3(
            base,
            soc_min=float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0")),
            soc_max=float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95")),
            cost_mode="hinge",
        )

        self._env = env
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._num_envs = 1
        self._max_episode_steps = 8759

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def max_episode_steps(self) -> int | None:
        return self._max_episode_steps

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[torch.Tensor, dict]:
        obs, info = self._env.reset(seed=seed)
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        return obs_t, (info or {})

    def step(self, action: torch.Tensor):
        a = action.detach().cpu().numpy()
        a = np.asarray(a).squeeze()  # ensure (act_dim,)

        obs, reward, terminated, truncated, info = self._env.step(a)
        cost = float(info.get("cost", 0.0))

        return (
            torch.as_tensor(np.asarray(obs), dtype=torch.float32),
            torch.as_tensor(float(reward), dtype=torch.float32),
            torch.as_tensor(cost, dtype=torch.float32),
            torch.as_tensor(bool(terminated), dtype=torch.bool),
            torch.as_tensor(bool(truncated), dtype=torch.bool),
            info or {},
        )

    def set_seed(self, seed: int) -> None:
        try:
            self._env.reset(seed=seed)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    def render(self):
        try:
            return self._env.render()
        except Exception:
            return None
