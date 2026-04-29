"""Cooling-only masked temperature CMDP for the case study.

This wrapper exposes only one cooling action per building to the policy while
filling the storage channels internally with fixed actions. It is the cleanest
action interface for a temperature-control case study because the learned
policy only controls the actuator most directly tied to comfort.
"""
from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch

from omnisafe.envs.core import env_register

from citylearn_safe.omni_env_temp import CityLearnTempSingleLagCMDP


def _flatten_action_names(action_names: Any) -> list[str]:
    if isinstance(action_names, list) and len(action_names) == 1 and isinstance(action_names[0], list):
        action_names = action_names[0]
    if action_names is None:
        return []
    return [str(name).strip() for name in list(action_names)]


class CityLearnTempCoolingOnlyBoundsProvider:
    """Safety bounds provider for the 3D cooling-only action interface."""

    def __init__(self, env: "CityLearnTempCoolingOnlyMaskedRewardCMDP") -> None:
        self._env = env

    def current_safe_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        mins: list[float] = []
        maxs: list[float] = []
        for building_idx in range(self._env.cooling_action_dim):
            cooling_min, cooling_max = self._env.cooling_bounds_for_building(building_idx)
            mins.append(float(np.clip(cooling_min, 0.0, 1.0)))
            maxs.append(float(np.clip(max(cooling_min, cooling_max), 0.0, 1.0)))
        return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


class CityLearnTempCoolingResidualBoundsProvider:
    """Residual-action bounds provider around the RBC cooling backbone."""

    def __init__(self, env: "CityLearnTempCoolingResidualCMDP") -> None:
        self._env = env

    def current_safe_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        safe_min, safe_max = self._env._cooling_safe_bounds()
        base = self._env._rbc_cooling_action()
        mins = np.clip(safe_min - base, self._env._residual_low, self._env._residual_high)
        maxs = np.clip(safe_max - base, self._env._residual_low, self._env._residual_high)
        maxs = np.maximum(maxs, mins)
        return mins.astype(np.float32), maxs.astype(np.float32)


@env_register
class CityLearnTempCoolingOnlyMaskedRewardCMDP(CityLearnTempSingleLagCMDP):
    """Cooling-only temperature CMDP with policy-mask provider hook."""

    _support_envs: ClassVar[list[str]] = [
        "CityLearnTemp-CoolingOnly-Masked-Reward-v0",
        "CityLearnTemp-CoolingOnly-CSACLB-v0",
    ]
    policy_action_mask_enabled: ClassVar[bool] = True

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)

        city = self._get_citylearn()
        action_names = _flatten_action_names(getattr(city, "action_names", None))
        if not action_names:
            raise RuntimeError("[TempCoolingOnly] CityLearn action_names are missing")

        self._full_action_space = self._action_space
        self._full_action_low = np.asarray(self._full_action_space.low, dtype=np.float32).reshape(-1)
        self._full_action_high = np.asarray(self._full_action_space.high, dtype=np.float32).reshape(-1)
        if len(action_names) != int(self._full_action_low.shape[0]):
            raise ValueError("[TempCoolingOnly] action_names length does not match full action space")

        self._action_names = action_names
        self._cooling_indices = [
            idx for idx, name in enumerate(action_names) if str(name).strip().lower() == "cooling_device"
        ]
        self._battery_indices = [
            idx for idx, name in enumerate(action_names) if str(name).strip().lower() == "electrical_storage"
        ]
        self._dhw_indices = [
            idx for idx, name in enumerate(action_names) if str(name).strip().lower() == "dhw_storage"
        ]
        if not self._cooling_indices:
            raise ValueError("[TempCoolingOnly] No cooling_device actions found")

        self.cooling_action_dim = len(self._cooling_indices)
        self.policy_mask_cooling_disable_margin = float(
            kwargs.get("policy_mask_cooling_disable_margin", 0.0),
        )
        self.policy_mask_cooling_full_range = float(
            kwargs.get("policy_mask_cooling_full_range", 6.0),
        )
        self.policy_mask_cooling_force_range = float(
            kwargs.get("policy_mask_cooling_force_range", 4.0),
        )
        self.policy_mask_cooling_setpoint_deadband = float(
            kwargs.get("policy_mask_cooling_setpoint_deadband", 0.5),
        )
        self.policy_mask_cooling_min_cap = float(
            kwargs.get("policy_mask_cooling_min_cap", 0.6),
        )
        self.policy_mask_cooling_force_trigger = float(
            kwargs.get("policy_mask_cooling_force_trigger", 28.0),
        )
        self.policy_mask_cooling_rbc_divisor = float(
            kwargs.get("policy_mask_cooling_rbc_divisor", 3.0),
        )
        self.policy_mask_freeze_dhw_storage = int(kwargs.get("policy_mask_freeze_dhw_storage", 1))
        self.policy_mask_battery_follow_rbc = int(kwargs.get("policy_mask_battery_follow_rbc", 0))

        self._action_space = gym.spaces.Box(
            low=np.zeros(self.cooling_action_dim, dtype=np.float32),
            high=np.ones(self.cooling_action_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def _hour_of_day(self) -> int:
        city = self._get_citylearn()
        return self._state_time_index(city) % 24

    def _cooling_state(self, building_idx: int) -> tuple[float | None, float | None]:
        city = self._get_citylearn()
        if city is None:
            return None, None
        try:
            building = city.buildings[building_idx]
        except Exception:
            return None, None

        t_idx = self._state_time_index(city)
        es = getattr(building, "energy_simulation", None)
        tin = self._as_scalar_at(
            getattr(es, "indoor_dry_bulb_temperature", None) if es is not None else None,
            t_idx,
        )
        tset = self._as_scalar_at(
            getattr(es, "indoor_dry_bulb_temperature_cooling_set_point", None)
            if es is not None
            else None,
            t_idx,
        )
        if tin is not None:
            return tin, tset

        try:
            data = building._get_observations_data()
        except Exception:
            data = {}
        return (
            self._as_scalar_at(data.get("indoor_dry_bulb_temperature"), t_idx),
            self._as_scalar_at(data.get("indoor_dry_bulb_temperature_cooling_set_point"), t_idx),
        )

    def cooling_bounds_for_building(self, building_idx: int) -> tuple[float, float]:
        city = self._get_citylearn()
        t_idx = self._state_time_index(city)
        if t_idx < self._lstm_warmup_steps:
            return 0.0, 1.0

        tin, tset = self._cooling_state(building_idx)
        if tin is None:
            return 0.0, 1.0

        disable_temp = self.comfort_tmin + self.policy_mask_cooling_disable_margin
        if tset is not None:
            disable_temp = max(disable_temp, float(tset) + self.policy_mask_cooling_setpoint_deadband)
        if tin <= disable_temp:
            # Never block cooling — let the learned policy cool preemptively
            # if the cost critic says a violation is coming.
            return 0.0, 1.0

        force_start = max(self.policy_mask_cooling_force_trigger, disable_temp)
        rbc_like_max = max(
            0.0,
            (tin - disable_temp) / max(self.policy_mask_cooling_rbc_divisor, 1e-6),
        )
        cool_max = min(
            1.0,
            max(
                0.0,
                max(
                    (tin - disable_temp) / max(self.policy_mask_cooling_full_range, 1e-6),
                    rbc_like_max,
                ),
            ),
        )
        if tin <= force_start:
            cool_min = 0.0
        else:
            cool_min = min(
                self.policy_mask_cooling_min_cap,
                max(
                    0.0,
                    (tin - force_start) / max(self.policy_mask_cooling_force_range, 1e-6),
                ),
            )
        return cool_min, max(cool_min, cool_max)

    def _fixed_storage_action(self, idx: int) -> float:
        name = str(self._action_names[idx]).strip().lower()
        if name == "dhw_storage":
            return 0.0
        if name == "electrical_storage":
            if not bool(self.policy_mask_battery_follow_rbc):
                return 0.0
            hour = self._hour_of_day()
            if 10 <= hour <= 16:
                return 0.8
            if 17 <= hour <= 21:
                return -0.6
            return 0.0
        return 0.0

    def _compose_full_action(self, cooling_action: np.ndarray) -> np.ndarray:
        full = np.zeros_like(self._full_action_low, dtype=np.float32)
        for idx in range(len(full)):
            if idx in self._cooling_indices:
                continue
            full[idx] = self._fixed_storage_action(idx)
        for local_idx, full_idx in enumerate(self._cooling_indices):
            full[full_idx] = float(cooling_action[local_idx])
        return np.clip(full, self._full_action_low, self._full_action_high)

    def make_policy_action_bounds_provider(self) -> CityLearnTempCoolingOnlyBoundsProvider:
        return CityLearnTempCoolingOnlyBoundsProvider(self)

    def get_policy_action_bounds_provider(self) -> CityLearnTempCoolingOnlyBoundsProvider:
        return self.make_policy_action_bounds_provider()

    def policy_action_bounds_provider(self) -> CityLearnTempCoolingOnlyBoundsProvider:
        return self.make_policy_action_bounds_provider()

    def masked_action_bounds_provider(self) -> CityLearnTempCoolingOnlyBoundsProvider:
        return self.make_policy_action_bounds_provider()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            a = action.detach().cpu().numpy().ravel()
        else:
            a = np.asarray(action, dtype=np.float32).ravel()

        a = np.clip(a, 0.0, 1.0)
        full_action = self._compose_full_action(a)

        obs, _base_reward, terminated, truncated, info = self._env.step(full_action)
        self._step_count += 1

        metrics = self._collect_step_metrics()
        reward, cost, components = self._compute_reward_and_cost(metrics)
        self._prev_building_net = np.asarray(metrics["building_net"], dtype=np.float64).copy()

        info = dict(info)
        info["cost"] = float(cost)
        info["cost_comfort"] = float(cost)
        info["cost_comfort_raw"] = float(metrics["comfort_cost_raw"])
        info["comfort_violation"] = float(metrics["comfort_violation"])
        info["comfort_in_warmup"] = 1.0 if metrics["in_warmup"] else 0.0
        info["reward_stems_total"] = float(reward)
        info["cooling_action_mean"] = float(np.mean(a)) if a.size else 0.0
        for key, value in components.items():
            info[key] = float(value)
        info["district_import_kwh"] = float(metrics["district_import"])
        info["district_net_kwh"] = float(metrics["district_net"])
        info["electricity_price"] = float(metrics["price"])
        info["discomfort_count"] = float(metrics["discomfort_count"])
        info["occupied_count"] = float(metrics["occupied_count"])
        info["discomfort_rate"] = (
            float(metrics["discomfort_count"] / metrics["occupied_count"])
            if metrics["occupied_count"] > 0
            else 0.0
        )
        for key, value in list(info.items()):
            if isinstance(value, (int, float)):
                info[f"Metrics/{key}"] = float(value)

        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        reward_t = torch.as_tensor(reward, dtype=torch.float32)
        cost_t = torch.as_tensor(cost, dtype=torch.float32)
        terminated_t = torch.as_tensor(terminated, dtype=torch.bool)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool)
        return obs_t, reward_t, cost_t, terminated_t, truncated_t, info

    def _cooling_safe_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        mins: list[float] = []
        maxs: list[float] = []
        for building_idx in range(self.cooling_action_dim):
            cooling_min, cooling_max = self.cooling_bounds_for_building(building_idx)
            mins.append(float(np.clip(cooling_min, 0.0, 1.0)))
            maxs.append(float(np.clip(max(cooling_min, cooling_max), 0.0, 1.0)))
        return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)

    def _rbc_cooling_action(self) -> np.ndarray:
        """Bang-bang thermostat RBC: full cooling ON when tin > 24°C, OFF otherwise.

        Models a simple on/off thermostat common in older HVAC systems.
        Threshold is set 2°C below the upper comfort bound (26°C).
        """
        actions = np.zeros(self.cooling_action_dim, dtype=np.float32)
        bangbang_threshold = 24.0
        for i in range(self.cooling_action_dim):
            tin, tset = self._cooling_state(i)
            if tin is None or not np.isfinite(tin):
                actions[i] = 0.0
                continue
            actions[i] = 1.0 if tin > bangbang_threshold else 0.0
        return actions

    def _rbc2_cooling_action(self) -> np.ndarray:
        actions = np.zeros(self.cooling_action_dim, dtype=np.float32)
        weak_deadband = float(self.policy_mask_cooling_setpoint_deadband + 0.5)
        weak_divisor = float(max(self.policy_mask_cooling_rbc_divisor * 1.5, 1e-6))
        for i in range(self.cooling_action_dim):
            tin, tset = self._cooling_state(i)
            if tin is None or not np.isfinite(tin):
                actions[i] = 0.0
                continue
            if tset is None or not np.isfinite(tset):
                tset = 24.0
            threshold = float(tset) + weak_deadband
            if tin > threshold:
                error = tin - threshold
                actions[i] = float(min(1.0, error / weak_divisor))
            else:
                actions[i] = 0.0
        return actions

    def _rbc3_cooling_action(self) -> np.ndarray:
        actions = np.zeros(self.cooling_action_dim, dtype=np.float32)
        aggressive_deadband = float(max(self.policy_mask_cooling_setpoint_deadband - 0.3, 0.0))
        aggressive_divisor = float(max(self.policy_mask_cooling_rbc_divisor * 0.5, 1e-6))
        for i in range(self.cooling_action_dim):
            tin, tset = self._cooling_state(i)
            if tin is None or not np.isfinite(tin):
                actions[i] = 0.0
                continue
            if tset is None or not np.isfinite(tset):
                tset = 24.0
            threshold = float(tset) + aggressive_deadband
            if tin > threshold:
                error = tin - threshold
                actions[i] = float(min(1.0, error / aggressive_divisor))
            else:
                actions[i] = 0.0
        return actions


@env_register
class CityLearnTempCoolingResidualCMDP(CityLearnTempCoolingOnlyMaskedRewardCMDP):
    """Cooling-only residual controller around an RBC backbone."""

    _support_envs: ClassVar[list[str]] = [
        "CityLearnTemp-CoolingResidual-CSACLB-v0",
    ]
    policy_action_mask_enabled: ClassVar[bool] = True

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)
        self.residual_action_limit = float(kwargs.get("residual_action_limit", 0.25))
        self._residual_low = np.full(self.cooling_action_dim, -self.residual_action_limit, dtype=np.float32)
        self._residual_high = np.full(self.cooling_action_dim, self.residual_action_limit, dtype=np.float32)
        self._action_space = gym.spaces.Box(
            low=self._residual_low.copy(),
            high=self._residual_high.copy(),
            dtype=np.float32,
        )

    def make_policy_action_bounds_provider(self) -> CityLearnTempCoolingResidualBoundsProvider:
        return CityLearnTempCoolingResidualBoundsProvider(self)

    def get_policy_action_bounds_provider(self) -> CityLearnTempCoolingResidualBoundsProvider:
        return self.make_policy_action_bounds_provider()

    def policy_action_bounds_provider(self) -> CityLearnTempCoolingResidualBoundsProvider:
        return self.make_policy_action_bounds_provider()

    def masked_action_bounds_provider(self) -> CityLearnTempCoolingResidualBoundsProvider:
        return self.make_policy_action_bounds_provider()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            residual = action.detach().cpu().numpy().ravel()
        else:
            residual = np.asarray(action, dtype=np.float32).ravel()

        residual = np.clip(residual, self._residual_low, self._residual_high)
        base_action = self._rbc_cooling_action()
        safe_min, safe_max = self._cooling_safe_bounds()
        cooling_action = np.clip(base_action + residual, safe_min, safe_max)
        full_action = self._compose_full_action(cooling_action)

        obs, _base_reward, terminated, truncated, info = self._env.step(full_action)
        self._step_count += 1

        metrics = self._collect_step_metrics()
        reward, cost, components = self._compute_reward_and_cost(metrics)
        self._prev_building_net = np.asarray(metrics["building_net"], dtype=np.float64).copy()

        info = dict(info)
        info["cost"] = float(cost)
        info["cost_comfort"] = float(cost)
        info["cost_comfort_raw"] = float(metrics["comfort_cost_raw"])
        info["comfort_violation"] = float(metrics["comfort_violation"])
        info["comfort_in_warmup"] = 1.0 if metrics["in_warmup"] else 0.0
        info["reward_stems_total"] = float(reward)
        info["cooling_action_mean"] = float(np.mean(cooling_action)) if cooling_action.size else 0.0
        info["rbc_cooling_action_mean"] = float(np.mean(base_action)) if base_action.size else 0.0
        info["residual_action_mean"] = float(np.mean(residual)) if residual.size else 0.0
        for key, value in components.items():
            info[key] = float(value)
        info["district_import_kwh"] = float(metrics["district_import"])
        info["district_net_kwh"] = float(metrics["district_net"])
        info["electricity_price"] = float(metrics["price"])
        info["discomfort_count"] = float(metrics["discomfort_count"])
        info["occupied_count"] = float(metrics["occupied_count"])
        info["discomfort_rate"] = (
            float(metrics["discomfort_count"] / metrics["occupied_count"])
            if metrics["occupied_count"] > 0
            else 0.0
        )
        for key, value in list(info.items()):
            if isinstance(value, (int, float)):
                info[f"Metrics/{key}"] = float(value)

        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        reward_t = torch.as_tensor(reward, dtype=torch.float32)
        cost_t = torch.as_tensor(cost, dtype=torch.float32)
        terminated_t = torch.as_tensor(terminated, dtype=torch.bool)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool)
        return obs_t, reward_t, cost_t, terminated_t, truncated_t, info


__all__ = [
    "CityLearnTempCoolingOnlyBoundsProvider",
    "CityLearnTempCoolingResidualBoundsProvider",
    "CityLearnTempCoolingOnlyMaskedRewardCMDP",
    "CityLearnTempCoolingResidualCMDP",
]
