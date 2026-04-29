"""CSAC-LB adapted for the V2G multi-building (R28) environment."""
from __future__ import annotations

import os
from typing import Any, Tuple

import numpy as np
import torch
import gymnasium as gym

from omnisafe.envs.core import CMDP, env_register
from omnisafe.algorithms import registry

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper
from citylearn_safe.adapters import SingleAgentListAdapter
from citylearn_safe.safety_env import CityLearnSafetyEnv
from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
from citylearn_safe.csac_lb_temp import CSACLBTemp, _CSACLBOffPolicyAdapter
from omnisafe.utils.config import Config


# NOTE: @env_register intentionally removed. CSAC-LB now uses the stock
# CityLearnV2G env from cmdp_env.py (registered via the import in
# scripts/train_csac_lb_v2g.py) for apples-to-apples comparison with the rest
# of the R28 benchmark suite (same 9-term STEMS reward, same obs space, same
# cost aggregation). This class is kept for reference only and is no longer
# instantiated at training time.
class CityLearnV2GCMDP(CMDP):
    """V2G multi-building CMDP wrapping safety_env directly.

    [DEPRECATED] Retained only as reference for the earlier CSAC-LB pipeline.
    Do not use — see cmdp_env.CityLearnV2G instead.
    """

    _support_envs = ["CityLearnSafety-V2G-v2"]
    need_time_limit_wrapper: bool = False
    need_auto_reset_wrapper: bool = True

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)

        # Capture the device passed by omnisafe.envs.core.make() so we can
        # return tensors on the correct device (fixes upstream device mismatch
        # when training with device='cuda').
        device = kwargs.get("device", torch.device("cpu"))
        self._device = device if isinstance(device, torch.device) else torch.device(str(device))

        schema = os.environ.get("CITYLEARN_SCHEMA", "")
        if not schema:
            raise RuntimeError("CITYLEARN_SCHEMA env var is not set.")
        if not os.path.exists(schema):
            raise FileNotFoundError(f"CITYLEARN_SCHEMA not found: {schema}")

        base: gym.Env = CityLearnEnv(schema=schema, central_agent=True)
        base = NormalizedObservationWrapper(base)
        base = SingleAgentListAdapter(base)

        env: gym.Env = CityLearnSafetyEnv(
            base,
            soc_min=float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0")),
            soc_max=float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95")),
            cost_mode="hinge",
        )
        # Wrap with ForecastObsWrapper so CSAC-LB uses the same 198-dim obs as
        # every other R28 algorithm (apples-to-apples benchmarking).
        env = ForecastObsWrapper(env, forecast_horizon=24)

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
        return (
            torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self._device),
            (info or {}),
        )

    def step(self, action: torch.Tensor):
        a = np.asarray(action.detach().cpu().numpy()).squeeze()
        obs, reward, terminated, truncated, info = self._env.step(a)

        cost = float(info.get("cost", 0.0))

        return (
            torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self._device),
            torch.as_tensor(float(reward), dtype=torch.float32, device=self._device),
            torch.as_tensor(cost, dtype=torch.float32, device=self._device),
            torch.as_tensor(bool(terminated), dtype=torch.bool, device=self._device),
            torch.as_tensor(bool(truncated), dtype=torch.bool, device=self._device),
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


class _CSACLBV2GAdapter(_CSACLBOffPolicyAdapter):
    """Off-policy adapter with V2G episode length (8759 steps)."""

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        super().__init__(env_id, num_envs, seed, cfgs)
        self._max_ep_len = 8759


@registry.register
class CSACLBV2G(CSACLBTemp):
    """CSAC-LB for the V2G multi-building environment.

    Drop-in replacement for CSACLBTemp targeting CityLearnSafety-V2G-v2.
    Uses the same smoothed log-barrier actor loss and twin cost critics,
    but with the correct episode length for the annual V2G dataset.
    """

    def _init_env(self) -> None:
        self._env: _CSACLBV2GAdapter = _CSACLBV2GAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (
            self._cfgs.algo_cfgs.steps_per_epoch % self._cfgs.train_cfgs.vector_env_nums == 0
        ), "steps_per_epoch must be divisible by vector_env_nums."
        assert (
            int(self._cfgs.train_cfgs.total_steps) % self._cfgs.algo_cfgs.steps_per_epoch == 0
        ), "total_steps must be divisible by steps_per_epoch."
        self._epochs = int(
            self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch,
        )
        self._epoch = 0
        self._steps_per_epoch = (
            self._cfgs.algo_cfgs.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        )
        self._update_cycle = self._cfgs.algo_cfgs.update_cycle
        assert self._steps_per_epoch % self._update_cycle == 0
        self._samples_per_epoch = self._steps_per_epoch // self._update_cycle
        self._update_count = 0
