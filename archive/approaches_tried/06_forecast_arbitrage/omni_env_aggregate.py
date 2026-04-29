# citylearn_safe/omni_env_aggregate.py
from __future__ import annotations
from typing import Any, Tuple
import torch
import gymnasium as gym
from omnisafe.envs.core import CMDP, env_register

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3  # ✅ Use the file you provided!

@env_register
class CityLearnCMDPAggregate(CMDP):
    """
    CityLearn CMDP with AGGREGATE district-level SOC constraints.
    
    Uses safety_env_v3.py (the file with aggregate constraints already implemented)
    
    Environment ID: CityLearnSafety-Aggregate-v0
    """
    _support_envs = ['CityLearnSafety-Aggregate-v0']

    need_time_limit_wrapper: bool = False
    need_auto_reset_wrapper: bool = True

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)

        base: gym.Env = make_base_env(central_agent=True)
        
        # ✅ Use CityLearnSafetyEnvV3 from safety_env_v3.py
        env: gym.Env = CityLearnSafetyEnvV3(
            base, 
            soc_min=0.20,  # Will be overridden by env vars
            soc_max=0.80,  # Will be overridden by env vars
            cost_mode="hinge",
            include_ev_in_cost=True
        )
        self._env = env

        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._num_envs = 1
        self._max_episode_steps = 8759

        print(f"[CityLearnCMDPAggregate] ✅ Initialized with safety_env_v3.py")
        print(f"[CityLearnCMDPAggregate]    Using AGGREGATE district-level constraints")

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
        return getattr(self, "_max_episode_steps", None)

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[torch.Tensor, dict]:
        if seed is not None:
            self._env.reset(seed=seed)
        obs, info = self._env.reset()
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        
        kpi_info = {}
        for key, value in info.items():
            if key.startswith(('obs_', 'soc_', 'action_', 'step_', 'total_', 
                             'constraint_', 'cost_', 'battery_', 'district_')):
                kpi_info[f'Metrics/{key}'] = float(value)
                kpi_info[f'Train/{key}'] = float(value)
                kpi_info[key] = float(value)
                kpi_info[f'KPI/{key}'] = float(value)
        
        info.update(kpi_info)
        return obs_t, info or {}

    def step(self, action: torch.Tensor):
        a = action.detach().cpu().numpy()
        obs, reward, terminated, truncated, info = self._env.step(a)

        cost = info.get('cost', 0.0)
        
        kpi_info = {}
        for key, value in info.items():
            if key.startswith(('obs_', 'soc_', 'action_', 'step_', 'total_', 
                             'constraint_', 'cost_', 'battery_', 'district_')):
                kpi_info[f'Metrics/{key}'] = float(value)
                kpi_info[f'Train/{key}'] = float(value)
                kpi_info[key] = float(value)
                kpi_info[f'KPI/{key}'] = float(value)
        
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
        try:
            return self._env.render()
        except Exception:
            return None
