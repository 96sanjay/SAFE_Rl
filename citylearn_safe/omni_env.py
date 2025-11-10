# citylearn_safe/omni_env.py
from __future__ import annotations
from typing import Any, Tuple
import torch
import gymnasium as gym
from omnisafe.envs.core import CMDP, env_register  # official hook

from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv
from citylearn_safe.kpi_logger import init_kpi_logger

@env_register
class CityLearnCMDP(CMDP):
    """Expose CityLearnSafetyEnv to OmniSafe's CMDP interface."""
    # List of IDs this class will serve:
    _support_envs = ['CityLearnSafety-SoC-v0']  # use your real ID here

    # Tell OmniSafe to add its standard wrappers
    need_time_limit_wrapper: bool = False   # OmniSafe wrappers also handle time limit
    need_auto_reset_wrapper: bool = True

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)

        # Build your Gym env as usual
        base: gym.Env = make_base_env(central_agent=True)
        env: gym.Env = CityLearnSafetyEnv(base, soc_min=0.1, soc_max=0.9)
        self._env = env

        # Declare Gym spaces
        self._observation_space = env.observation_space
        self._action_space = env.action_space

        # Parallelism (keep 1 unless you want vectorization handled here)
        self._num_envs = 1

        # If you know the episode horizon (e.g., 35040), expose it:
        self._max_episode_steps = 8759  # optional; aids logging/slicing
        
        # Initialize KPI logger - will be set up when OmniSafe creates the run directory
        # We'll initialize it later in the training process when we know the exact run directory

    # ---- required properties / methods (CMDP)
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
        return getattr(self, "_max_episode_steps", None)  # OmniSafe reads this

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[torch.Tensor, dict]:
        if seed is not None:
            self._env.reset(seed=seed)
        obs, info = self._env.reset()
        # OmniSafe expects torch tensors
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        
        # Add KPI metrics to info for OmniSafe logging
        # Try multiple formats to see what OmniSafe recognizes
        kpi_info = {}
        for key, value in info.items():
            if key.startswith(('obs_', 'soc_', 'action_', 'step_', 'total_', 'constraint_')):
                # Try different prefixes that OmniSafe might recognize
                kpi_info[f'Metrics/{key}'] = float(value)
                kpi_info[f'Train/{key}'] = float(value)
                kpi_info[key] = float(value)  # Also keep original key
                kpi_info[f'KPI/{key}'] = float(value)  # Keep KPI prefix too
        
        # Merge KPI info with existing info
        info.update(kpi_info)
        
        return obs_t, info or {}

    def step(self, action: torch.Tensor):
        # Convert incoming tensor to numpy for the underlying env
        a = action.detach().cpu().numpy()
        obs, reward, terminated, truncated, info = self._env.step(a)

        # Pull safety cost from info (your SafetyEnv already writes info['cost'])
        cost = info.get('cost', 0.0)
        
        # Add KPI metrics to info for OmniSafe logging
        # Try multiple formats to see what OmniSafe recognizes
        kpi_info = {}
        for key, value in info.items():
            if key.startswith(('obs_', 'soc_', 'action_', 'step_', 'total_', 'constraint_')):
                # Try different prefixes that OmniSafe might recognize
                kpi_info[f'Metrics/{key}'] = float(value)
                kpi_info[f'Train/{key}'] = float(value)
                kpi_info[key] = float(value)  # Also keep original key
                kpi_info[f'KPI/{key}'] = float(value)  # Keep KPI prefix too
        
        # Merge KPI info with existing info
        info.update(kpi_info)

        return (
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(cost, dtype=torch.float32),
            torch.as_tensor(terminated, dtype=torch.bool),
            torch.as_tensor(truncated, dtype=torch.bool),
            info,
        )

    def set_seed(self, seed: int) -> None:
        try:
            self._env.reset(seed=seed)
        except Exception:
            pass

    def close(self) -> None:
        self._env.close()

    def render(self):
        """OmniSafe requires this. Return underlying render or None."""
        try:
            return self._env.render()
        except Exception:
            return None
