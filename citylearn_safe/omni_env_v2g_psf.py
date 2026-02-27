"""
OmniSafe CMDP Environment with Predictive Safety Filter (V2G-enabled).
Works with FOCOPS, TRPOLag, PPOLag, or any OmniSafe CMDP algorithm.

Architecture (during TRAINING):
    Agent -> LookaheadPSFv2G -> CityLearnSafetyEnvV3 -> CityLearn
                | SE-RL penalty
            reward -= w * ||a_safe - a_rl||^2
"""
from __future__ import annotations
import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register

# Local imports -- adjust paths to your project structure
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.lookahead_psf_v2g import LookaheadPSFv2G


def _make_base_env():
    schema = os.environ.get(
        "CITYLEARN_SCHEMA",
        "data/citylearn_challenge_2022_phase_all_plus_evs/schema.json")
    central = os.environ.get("CITYLEARN_CENTRAL_AGENT", "1") == "1"
    from citylearn.citylearn import CityLearnEnv
    env = CityLearnEnv(schema=schema, central_agent=central)
    return env


@env_register
class CityLearnV2GPSF(CMDP):
    """
    OmniSafe CMDP wrapper with Predictive Safety Filter for V2G.
    Works with any OmniSafe on-policy CMDP algo (FOCOPS, TRPOLag, PPOLag, etc.)
    """
    _support_envs = ["CityLearnV2GPSF-v0"]
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False

    def __init__(self, env_id: str, **kwargs):
        super().__init__(env_id, **kwargs)

        self._psf_horizon = int(os.environ.get("PSF_HORIZON", "24"))
        self._psf_verbose = int(os.environ.get("PSF_VERBOSE", "1"))
        self._use_forecast = os.environ.get("USE_FORECAST", "1") == "1"

        self._serl_penalty_weight = float(
            os.environ.get("SERL_PENALTY_WEIGHT", "10.0"))

        self._alpha_grid = float(os.environ.get("STEMS_ALPHA_GRID", "1.5"))
        self._alpha_build = float(os.environ.get("STEMS_ALPHA_BUILD", "1.0"))
        self._beta_ramp = float(os.environ.get("STEMS_BETA_RAMP", "0.5"))
        self._alpha_renew = float(os.environ.get("STEMS_ALPHA_RENEW", "1.0"))

        self._w_cost_ev = float(os.environ.get("CITYLEARN_W_COST_EV", "1.0"))
        self._w_cost_soc = float(os.environ.get("CITYLEARN_W_COST_SOC", "0.1"))
        self._w_cost_building = float(
            os.environ.get("CITYLEARN_W_COST_BUILDING", "0.002"))
        self._w_cost_grid = float(os.environ.get("CITYLEARN_W_COST_GRID", "0.5"))

        base = _make_base_env()
        safety = CityLearnSafetyEnvV3(base)

        self._psf = LookaheadPSFv2G(
            safety,
            horizon=self._psf_horizon,
            w_track=100.0,
            w_slack_c1=5000.0,
            w_slack_c3=500.0,
            w_slack_c4=1000.0,
            w_future_reg=0.01,
            exempt_ev_from_c3=True,
            c4_one_sided=True,
            verbose=self._psf_verbose,
        )

        if self._use_forecast:
            self._env = ForecastObsWrapper(self._psf, forecast_horizon=24)
        else:
            self._env = self._psf

        obs_space = self._env.observation_space
        if isinstance(obs_space, spaces.Box):
            self._observation_space = obs_space
        else:
            flat_lo = np.concatenate(
                [np.asarray(s.low).ravel() for s in obs_space])
            flat_hi = np.concatenate(
                [np.asarray(s.high).ravel() for s in obs_space])
            self._observation_space = spaces.Box(
                low=flat_lo, high=flat_hi, dtype=np.float32)

        act_space = self._env.action_space
        if isinstance(act_space, list):
            flat_lo = np.concatenate(
                [np.asarray(s.low).ravel() for s in act_space])
            flat_hi = np.concatenate(
                [np.asarray(s.high).ravel() for s in act_space])
            self._action_space = spaces.Box(
                low=flat_lo, high=flat_hi, dtype=np.float32)
        else:
            self._action_space = act_space

        self._num_envs = 1
        self._metadata = {}

        self._prev_grid_power = 0.0
        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_steps = 0
        self._episode_interventions = 0
        self._episode_ev_interventions = 0

        print(f"[V2G-PSF-CMDP] Initialized: "
              f"PSF H={self._psf_horizon} "
              f"SERL_w={self._serl_penalty_weight} "
              f"alpha_grid={self._alpha_grid} "
              f"forecast={'ON' if self._use_forecast else 'OFF'} "
              f"obs_dim={self._observation_space.shape[0]} "
              f"act_dim={self._action_space.shape[0]}")

    def _compute_stems_reward(self, info: dict) -> float:
        rew = 0.0
        grid_import = float(info.get("grid_import_kwh", 0.0))
        p_grid_max = float(os.environ.get(
            "CITYLEARN_STEMS_P_GRID_MAX", "27.127751"))
        if p_grid_max > 0:
            ratio = min(1.0, abs(grid_import) / p_grid_max)
            rew += self._alpha_grid * (1.0 - ratio ** 2)

        bld_violations = info.get("building_power_violations", [])
        p_bld_max = float(os.environ.get(
            "CITYLEARN_STEMS_P_BUILDING_MAX", "2.273834"))
        if isinstance(bld_violations, (list, np.ndarray)) and len(bld_violations) > 0:
            ratios = [min(1.0, abs(v) / max(p_bld_max, 1e-6))
                      for v in bld_violations]
            rew += self._alpha_build * (1.0 - np.mean(
                [r ** 2 for r in ratios]))

        grid_now = float(info.get("total_grid_power", 0.0))
        if p_grid_max > 0:
            ramp = abs(grid_now - self._prev_grid_power) / p_grid_max
            rew -= self._beta_ramp * min(1.0, ramp)
        self._prev_grid_power = grid_now

        solar = abs(float(info.get("total_solar_generation", 0.0)))
        curtailed = float(info.get("solar_curtailment", 0.0))
        if solar > 0.01:
            rew += self._alpha_renew * (1.0 - curtailed / solar)

        return float(rew)

    def _compute_cost(self, info: dict) -> float:
        c1 = float(info.get("cost_ev_departure", 0.0))
        c2 = float(info.get("cost_soc_violation", 0.0))
        c3 = float(info.get("cost_building_power", 0.0))
        c4 = float(info.get("cost_grid_power", 0.0))
        cost = (self._w_cost_ev * c1
                + self._w_cost_soc * c2
                + self._w_cost_building * c3
                + self._w_cost_grid * c4)
        return float(cost)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).ravel()
        action = np.clip(action, self._action_space.low, self._action_space.high)

        obs, _rew, terminated, truncated, info = self._env.step(action)
        info = dict(info) if info else {}

        if isinstance(obs, (list, tuple)):
            obs = np.concatenate(
                [np.asarray(o, dtype=np.float32).ravel() for o in obs])
        obs = torch.as_tensor(obs, dtype=torch.float32).ravel()

        reward = self._compute_stems_reward(info)

        delta_l2 = float(info.get("psf_action_delta_l2", 0.0))
        serl_penalty = self._serl_penalty_weight * (delta_l2 ** 2)
        reward -= serl_penalty

        cost = self._compute_cost(info)

        self._episode_reward += reward
        self._episode_cost += cost
        self._episode_steps += 1
        if info.get("psf_any_intervention", 0) > 0.5:
            self._episode_interventions += 1
        self._episode_ev_interventions += int(
            info.get("psf_ev_interventions", 0))

        if self._episode_steps % 2000 == 0 and self._psf_verbose >= 1:
            avg_rew = self._episode_reward / max(1, self._episode_steps)
            avg_cost = self._episode_cost / max(1, self._episode_steps)
            interv_pct = (100.0 * self._episode_interventions
                          / max(1, self._episode_steps))
            print(f"[V2G-PSF] step={self._episode_steps} "
                  f"avg_rew={avg_rew:.3f} avg_cost={avg_cost:.3f} "
                  f"interv={interv_pct:.1f}% "
                  f"ev_interv={self._episode_ev_interventions} "
                  f"serl_pen={serl_penalty:.3f} delta={delta_l2:.3f}")

        obs = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        cost = torch.as_tensor(cost, dtype=torch.float32)
        terminated = torch.as_tensor(terminated, dtype=torch.bool)
        truncated = torch.as_tensor(truncated, dtype=torch.bool)
        obs = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        cost = torch.as_tensor(cost, dtype=torch.float32)
        terminated = torch.as_tensor(terminated, dtype=torch.bool)
        truncated = torch.as_tensor(truncated, dtype=torch.bool)
        obs = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        cost = torch.as_tensor(cost, dtype=torch.float32)
        terminated = torch.as_tensor(terminated, dtype=torch.bool)
        truncated = torch.as_tensor(truncated, dtype=torch.bool)
        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if self._episode_steps > 0 and self._psf_verbose >= 1:
            interv_pct = (100.0 * self._episode_interventions
                          / max(1, self._episode_steps))
            print(f"[V2G-PSF] Episode done: "
                  f"steps={self._episode_steps} "
                  f"total_rew={self._episode_reward:.1f} "
                  f"total_cost={self._episode_cost:.1f} "
                  f"interventions={self._episode_interventions} "
                  f"({interv_pct:.1f}%) "
                  f"ev_interv={self._episode_ev_interventions}")

        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_steps = 0
        self._episode_interventions = 0
        self._episode_ev_interventions = 0
        self._prev_grid_power = 0.0

        obs, info = self._env.reset()
        if isinstance(obs, (list, tuple)):
            obs = np.concatenate(
                [np.asarray(o, dtype=np.float32).ravel() for o in obs])
        obs = torch.as_tensor(obs, dtype=torch.float32).ravel()
        return obs, info

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def metadata(self):
        return self._metadata

    def set_seed(self, seed):
        pass

    def render(self):
        pass

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
