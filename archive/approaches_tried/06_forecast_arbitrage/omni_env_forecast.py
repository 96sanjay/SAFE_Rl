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
from citylearn_safe.stems_obs_wrapper import ForecastObsWrapper


class NormalizedForecastObsWrapper(ForecastObsWrapper):
    """ForecastObsWrapper with built-in feature normalization.
    
    All 128 forecast dims are clipped to [-1, 1] before appending,
    so they match the scale of the NormalizedObservationWrapper base obs.
    """
    
    def __init__(self, env, forecast_horizon=24):
        super().__init__(env, forecast_horizon=forecast_horizon)
        # Running stats for online normalization
        self._forecast_mean = np.zeros(self.n_extra, dtype=np.float32)
        self._forecast_var = np.ones(self.n_extra, dtype=np.float32)
        self._forecast_count = 0
        print("[NormForecastObs] Forecast features will be online-normalized to ~N(0,1)")
    
    def _update_stats(self, feat):
        """Welford's online mean/variance."""
        self._forecast_count += 1
        n = self._forecast_count
        delta = feat - self._forecast_mean
        self._forecast_mean += delta / n
        delta2 = feat - self._forecast_mean
        self._forecast_var += (delta * delta2 - self._forecast_var) / n
    
    def _normalize_forecast(self, feat):
        """Normalize using running stats, clip to [-3, 3]."""
        std = np.sqrt(np.maximum(self._forecast_var, 1e-8))
        normed = (feat - self._forecast_mean) / std
        return np.clip(normed, -3.0, 3.0).astype(np.float32)
    
    def observation(self, obs):
        if isinstance(obs, (list, tuple)):
            flat = np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs])
        else:
            flat = np.asarray(obs, dtype=np.float32).ravel()
        
        raw_forecast = self._get_forecast()
        self._update_stats(raw_forecast)
        norm_forecast = self._normalize_forecast(raw_forecast)
        
        return np.concatenate([flat, norm_forecast]).astype(np.float32)
    
    def reset(self, **kwargs):
        self._citylearn_env = None
        self._buildings_cache = None
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


@env_register
class CityLearnForecastCMDP(CMDP):
    _support_envs = ['CityLearnSafety-Forecast-v0']
    need_time_limit_wrapper: bool = False
    need_auto_reset_wrapper: bool = True

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)
        schema = os.environ.get("CITYLEARN_SCHEMA", "")
        if not schema:
            raise RuntimeError("CITYLEARN_SCHEMA is not set.")
        if not os.path.exists(schema):
            raise FileNotFoundError(f"CITYLEARN_SCHEMA not found: {schema}")
        base: gym.Env = CityLearnEnv(schema=schema, central_agent=True)
        base = NormalizedObservationWrapper(base)
        base = SingleAgentListAdapter(base)
        env: gym.Env = CityLearnSafetyEnvV3(
            base,
            soc_min=float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0")),
            soc_max=float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95")),
            cost_mode="hinge",
        )
        env = NormalizedForecastObsWrapper(env, forecast_horizon=24)
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

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed)
        return torch.as_tensor(np.asarray(obs), dtype=torch.float32), (info or {})

    def step(self, action):
        a = action.detach().cpu().numpy().squeeze()
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

    def set_seed(self, seed):
        try: self._env.reset(seed=seed)
        except: pass
    def close(self):
        try: self._env.close()
        except: pass
    def render(self):
        try: return self._env.render()
        except: return None
