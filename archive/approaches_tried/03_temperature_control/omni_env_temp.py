"""
OmniSafe CMDP env for the temperature case study.

This environment is intentionally narrow:
  - LSTM-enabled 3-building schema by default
  - exact STEMS reward structure from Zhang et al. (arXiv:2510.14112)
  - single Lagrangian cost: district thermal comfort only

It is a thesis adaptation of STEMS, not a paper reproduction:
  - centralized single-agent control instead of multi-agent control
  - PPOLag instead of CBF shielding
  - comfort-only CMDP cost instead of battery/building/grid CBF constraints
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch

from omnisafe.envs.core import CMDP, env_register

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.make_env import make_base_env


DEFAULT_TEMP_SCHEMA = os.path.join(
    PROJECT_ROOT,
    "data",
    "citylearn_challenge_2023_phase_2_online_evaluation_3",
    "schema.json",
)


@env_register
class CityLearnTempSingleLagCMDP(CMDP):
    """OmniSafe CMDP for the LSTM temperature stress-test case study."""

    _support_envs: ClassVar[list[str]] = ["CityLearnTemp-Comfort-v0"]

    need_time_limit_wrapper: bool = False
    need_auto_reset_wrapper: bool = True

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)

        # This case-study env is bound to the LSTM temperature schema by default.
        # Set CITYLEARN_USE_DEFAULT_TEMP_SCHEMA=0 only when you intentionally
        # want to override it with another compatible temperature schema.
        if os.environ.get("CITYLEARN_USE_DEFAULT_TEMP_SCHEMA", "1") != "0":
            os.environ["CITYLEARN_SCHEMA"] = DEFAULT_TEMP_SCHEMA
        elif not os.environ.get("CITYLEARN_SCHEMA"):
            os.environ["CITYLEARN_SCHEMA"] = DEFAULT_TEMP_SCHEMA

        citylearn_env_kwargs = kwargs.get("citylearn_env_kwargs")
        base: gym.Env = make_base_env(
            central_agent=True,
            env_kwargs=citylearn_env_kwargs,
        )
        self._env = base
        self._observation_space = base.observation_space
        self._action_space = base.action_space
        self._num_envs = 1

        episode_cfg = citylearn_env_kwargs.get("episode_time_steps") if isinstance(citylearn_env_kwargs, dict) else None
        if isinstance(episode_cfg, int):
            self._max_episode_steps = int(episode_cfg)
        elif isinstance(episode_cfg, list) and len(episode_cfg) > 0:
            first_split = episode_cfg[0]
            if isinstance(first_split, (list, tuple)) and len(first_split) == 2:
                self._max_episode_steps = int(first_split[1]) - int(first_split[0]) + 1
            else:
                self._max_episode_steps = int(getattr(base, "time_steps", 2208) or 2208)
        else:
            self._max_episode_steps = int(getattr(base, "time_steps", 2208) or 2208)

        self.mu_economic = float(os.environ.get("CITYLEARN_STEMS_MU_ECONOMIC", "1.0"))
        self.alpha_grid = float(os.environ.get("CITYLEARN_STEMS_ALPHA_GRID", "0.5"))
        self.alpha_build = float(os.environ.get("CITYLEARN_STEMS_ALPHA_BUILD", "0.3"))
        self.beta_ramp = float(os.environ.get("CITYLEARN_STEMS_BETA_RAMP", "0.2"))
        self.lambda_indoor = float(os.environ.get("CITYLEARN_STEMS_LAMBDA_INDOOR", "0.4"))
        self.xi_renewable = float(os.environ.get("CITYLEARN_STEMS_XI_RENEWABLE", "0.6"))
        self.comfort_reward_occupancy_aware = (
            os.environ.get("CITYLEARN_COMFORT_REWARD_OCCUPANCY_AWARE", "1") != "0"
        )
        self.comfort_reward_vacant_eta = float(
            os.environ.get("CITYLEARN_COMFORT_REWARD_VACANT_ETA", "0.1"),
        )

        self.comfort_tmin = float(os.environ.get("CITYLEARN_COMFORT_TMIN", "20.0"))
        self.comfort_tmax = float(os.environ.get("CITYLEARN_COMFORT_TMAX", "26.0"))
        self.w_comfort = float(os.environ.get("CITYLEARN_W_COST_COMFORT", "0.05"))
        self._lstm_warmup_steps = int(os.environ.get("CITYLEARN_LSTM_WARMUP_STEPS", "13"))

        self.P_building_max, self.P_grid_max = self._calibrate_power_limits()
        self._prev_building_net: np.ndarray | None = None
        self._step_count = 0

        print(
            "[TempCMDP] schema=%s buildings=%d steps=%d"
            % (os.environ["CITYLEARN_SCHEMA"], self._num_buildings(), self._max_episode_steps)
        )
        print(
            "[TempCMDP] STEMS reward weights: mu=%.3f alpha_grid=%.3f alpha_build=%.3f "
            "beta_ramp=%.3f lambda_indoor=%.3f xi=%.3f"
            % (
                self.mu_economic,
                self.alpha_grid,
                self.alpha_build,
                self.beta_ramp,
                self.lambda_indoor,
                self.xi_renewable,
            )
        )
        print(
            "[TempCMDP] comfort reward occupancy-aware=%s vacant_eta=%.3f"
            % (
                "on" if self.comfort_reward_occupancy_aware else "off",
                self.comfort_reward_vacant_eta,
            )
        )
        print(
            "[TempCMDP] comfort cost: w=%.3f band=[%.1f, %.1f] warmup=%d "
            "P_building_max=%.3f P_grid_max=%.3f"
            % (
                self.w_comfort,
                self.comfort_tmin,
                self.comfort_tmax,
                self._lstm_warmup_steps,
                self.P_building_max,
                self.P_grid_max,
            )
        )

    def _get_citylearn(self):
        cur = self._env
        seen = set()
        for _ in range(40):
            if cur is None or id(cur) in seen:
                break
            seen.add(id(cur))
            if (
                hasattr(cur, "buildings")
                and hasattr(cur, "time_step")
                and hasattr(cur.buildings, "__len__")
                and len(cur.buildings) > 0
            ):
                return cur
            for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
            else:
                break
        return None

    def _num_buildings(self) -> int:
        city = self._get_citylearn()
        return len(getattr(city, "buildings", []) or [])

    def _state_time_index(self, city) -> int:
        if city is None:
            return 0
        return max(0, int(getattr(city, "time_step", 0)) - 1)

    def _as_scalar_at(self, x, t_idx: int) -> float | None:
        if x is None:
            return None
        try:
            if np.isscalar(x):
                v = float(x)
                return v if np.isfinite(v) else None
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 0:
                v = float(arr)
                return v if np.isfinite(v) else None
            if len(arr) <= t_idx:
                return None
            v = float(arr[t_idx])
            return v if np.isfinite(v) else None
        except Exception:
            return None

    def _calibrate_power_limits(self) -> tuple[float, float]:
        env_bld = os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX")
        env_grid = os.environ.get("CITYLEARN_STEMS_P_GRID_MAX")
        if env_bld is not None and env_grid is not None:
            return float(env_bld), float(env_grid)

        city = self._get_citylearn()
        if city is None:
            p_bld = float(env_bld) if env_bld is not None else 4.6083
            p_grid = float(env_grid) if env_grid is not None else 10.2352
            return p_bld, p_grid

        building_p95s: list[float] = []
        total_load = None
        try:
            for b in city.buildings:
                es = getattr(b, "energy_simulation", None)
                load = getattr(es, "non_shiftable_load", None) if es is not None else None
                if load is None:
                    continue
                arr = np.asarray(load, dtype=np.float64)
                building_p95s.append(float(np.percentile(np.abs(arr), 95)))
                total_load = arr.copy() if total_load is None else total_load + arr
        except Exception:
            total_load = None

        if not building_p95s or total_load is None:
            p_bld = float(env_bld) if env_bld is not None else 4.6083
            p_grid = float(env_grid) if env_grid is not None else 10.2352
            return p_bld, p_grid

        p_bld = float(np.mean(building_p95s))
        p_grid = float(np.percentile(np.abs(total_load), 95))
        if env_bld is not None:
            p_bld = float(env_bld)
        if env_grid is not None:
            p_grid = float(env_grid)
        return p_bld, p_grid

    def _collect_step_metrics(self) -> dict[str, Any]:
        city = self._get_citylearn()
        if city is None:
            return {
                "n_buildings": 0,
                "building_net": np.zeros(0, dtype=np.float64),
                "district_import": 0.0,
                "district_net": 0.0,
                "price": 0.0,
                "solar_gen": np.zeros(0, dtype=np.float64),
                "indoor_temp": np.zeros(0, dtype=np.float64),
                "temp_ref": np.zeros(0, dtype=np.float64),
                "occupancy": np.zeros(0, dtype=np.float64),
                "comfort_cost_raw": 0.0,
                "comfort_violation": 0.0,
                "discomfort_count": 0.0,
                "occupied_count": 0.0,
                "in_warmup": False,
                "t_idx": 0,
            }

        t_idx = self._state_time_index(city)
        buildings = list(city.buildings)
        n = len(buildings)

        building_net = np.zeros(n, dtype=np.float64)
        solar_gen = np.zeros(n, dtype=np.float64)
        indoor_temp = np.full(n, np.nan, dtype=np.float64)
        temp_ref = np.full(n, 0.5 * (self.comfort_tmin + self.comfort_tmax), dtype=np.float64)
        occupancy = np.zeros(n, dtype=np.float64)

        for i, b in enumerate(buildings):
            nec = getattr(b, "net_electricity_consumption", None)
            v = self._as_scalar_at(nec, t_idx)
            building_net[i] = float(v) if v is not None else 0.0

            sg = getattr(b, "solar_generation", None)
            solar_v = self._as_scalar_at(sg, t_idx)
            solar_gen[i] = abs(float(solar_v)) if solar_v is not None else 0.0

            es = getattr(b, "energy_simulation", None)
            tin = self._as_scalar_at(
                getattr(es, "indoor_dry_bulb_temperature", None) if es is not None else None,
                t_idx,
            )
            if tin is not None:
                indoor_temp[i] = float(tin)

            tref = self._as_scalar_at(
                getattr(es, "indoor_dry_bulb_temperature_cooling_set_point", None)
                if es is not None
                else None,
                t_idx,
            )
            if tref is not None:
                temp_ref[i] = float(tref)

            occ = self._as_scalar_at(
                getattr(es, "occupant_count", None) if es is not None else None,
                t_idx,
            )
            occupancy[i] = float(occ) if occ is not None else 0.0

        district_net = float(np.sum(building_net))
        district_import = float(np.sum(np.maximum(building_net, 0.0)))
        price = 0.0
        if buildings:
            price_v = self._as_scalar_at(
                getattr(getattr(buildings[0], "pricing", None), "electricity_pricing", None),
                t_idx,
            )
            if price_v is not None:
                price = float(price_v)

        hi = np.maximum(indoor_temp - self.comfort_tmax, 0.0)
        lo = np.maximum(self.comfort_tmin - indoor_temp, 0.0)
        raw_dev = np.where(np.isfinite(indoor_temp), hi + lo, 0.0)

        in_warmup = t_idx < self._lstm_warmup_steps
        comfort_cost_raw = 0.0 if in_warmup else float(np.sum(raw_dev))
        comfort_violation = 0.0 if in_warmup else float(np.any(raw_dev > 0.0))

        if in_warmup:
            discomfort_count = 0.0
            occupied_count = 0.0
        else:
            discomfort_mask = (
                (occupancy > 0.0)
                & np.isfinite(indoor_temp)
                & np.isfinite(temp_ref)
                & (np.abs(indoor_temp - temp_ref) > 2.0)
            )
            discomfort_count = float(np.sum(discomfort_mask))
            occupied_count = float(np.sum(occupancy > 0.0))

        return {
            "n_buildings": n,
            "building_net": building_net,
            "district_import": district_import,
            "district_net": district_net,
            "price": price,
            "solar_gen": solar_gen,
            "indoor_temp": indoor_temp,
            "temp_ref": temp_ref,
            "occupancy": occupancy,
            "comfort_cost_raw": comfort_cost_raw,
            "comfort_violation": comfort_violation,
            "discomfort_count": discomfort_count,
            "occupied_count": occupied_count,
            "in_warmup": in_warmup,
            "t_idx": t_idx,
        }

    def _compute_reward_and_cost(self, metrics: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
        n = int(metrics["n_buildings"])
        e = np.asarray(metrics["building_net"], dtype=np.float64)
        p = np.asarray(metrics["solar_gen"], dtype=np.float64)
        tin = np.asarray(metrics["indoor_temp"], dtype=np.float64)

        district_import = float(metrics["district_import"])
        price = float(metrics["price"])

        if n <= 0:
            return 0.0, 0.0, {
                "reward_economic": 0.0,
                "reward_stability": 0.0,
                "reward_stability_grid": 0.0,
                "reward_stability_building": 0.0,
                "reward_stability_ramp": 0.0,
                "reward_renewable": 0.0,
                "reward_comfort": 0.0,
                "reward_comfort_occupied": 0.0,
                "reward_comfort_vacant": 0.0,
            }

        reward_economic = -self.mu_economic * price * float(np.sum(e))

        # Normalize conservatively so schema-specific excursions cannot blow up
        # the reward magnitude on this 3-building case study.
        grid_ratio = np.clip(district_import / max(1e-6, self.P_grid_max), 0.0, 1.0)
        reward_stability_grid = float(n) * self.alpha_grid * (1.0 - grid_ratio**2)

        building_ratio = np.clip(np.abs(e) / max(1e-6, self.P_building_max), 0.0, 1.0)
        reward_stability_building = self.alpha_build * float(np.sum(1.0 - building_ratio))

        if self._prev_building_net is None or len(self._prev_building_net) != n:
            ramp_delta = np.zeros(n, dtype=np.float64)
        else:
            ramp_delta = np.abs(e - self._prev_building_net)
        reward_stability_ramp = -self.beta_ramp * float(
            np.sum(np.clip(ramp_delta / max(1e-6, self.P_building_max), 0.0, 1.0))
        )
        reward_stability = (
            reward_stability_grid + reward_stability_building + reward_stability_ramp
        )

        denom = p + np.maximum(e, 0.0)
        solar_ratio = np.divide(p, denom, out=np.zeros_like(p), where=denom > 0.0)
        reward_renewable = self.xi_renewable * float(np.sum(np.minimum(solar_ratio, 1.0)))

        if metrics["in_warmup"]:
            reward_comfort = 0.0
            reward_comfort_occupied = 0.0
            reward_comfort_vacant = 0.0
        else:
            # Align the reward with the same comfort-band objective used by the
            # CMDP cost while biasing the reward toward occupied discomfort.
            comfort_dev = np.maximum(tin - self.comfort_tmax, 0.0) + np.maximum(
                self.comfort_tmin - tin,
                0.0,
            )
            comfort_dev = np.where(np.isfinite(comfort_dev), comfort_dev, 0.0)
            occupied_mask = np.asarray(metrics["occupancy"], dtype=np.float64) > 0.0
            if self.comfort_reward_occupancy_aware:
                comfort_weights = np.where(
                    occupied_mask,
                    1.0,
                    self.comfort_reward_vacant_eta,
                )
            else:
                comfort_weights = np.ones_like(comfort_dev, dtype=np.float64)

            occupied_dev = np.where(occupied_mask, comfort_dev, 0.0)
            vacant_dev = np.where(~occupied_mask, comfort_dev, 0.0)
            reward_comfort_occupied = -self.lambda_indoor * float(
                np.sum(occupied_dev),
            )
            reward_comfort_vacant = -self.lambda_indoor * float(
                self.comfort_reward_vacant_eta * np.sum(vacant_dev),
            )
            reward_comfort = -self.lambda_indoor * float(np.sum(comfort_weights * comfort_dev))

        reward = reward_economic + reward_stability + reward_renewable + reward_comfort
        cost = self.w_comfort * float(metrics["comfort_cost_raw"])

        components = {
            "reward_economic": float(reward_economic),
            "reward_stability": float(reward_stability),
            "reward_stability_grid": float(reward_stability_grid),
            "reward_stability_building": float(reward_stability_building),
            "reward_stability_ramp": float(reward_stability_ramp),
            "reward_renewable": float(reward_renewable),
            "reward_comfort": float(reward_comfort),
            "reward_comfort_occupied": float(reward_comfort_occupied),
            "reward_comfort_vacant": float(reward_comfort_vacant),
        }
        return float(reward), float(cost), components

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._prev_building_net = None
        self._step_count = 0
        return torch.as_tensor(obs, dtype=torch.float32), info

    def set_seed(self, seed: int) -> None:
        try:
            self._env.reset(seed=seed)
        except Exception:
            pass

    def render(self) -> Any:
        if hasattr(self._env, "render"):
            return self._env.render()
        return None

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    def step(self, action):
        if isinstance(action, torch.Tensor):
            a = action.detach().cpu().numpy().ravel()
        else:
            a = np.asarray(action, dtype=np.float32).ravel()

        # Respect per-dimension CityLearn action bounds.
        low = np.asarray(self._action_space.low, dtype=np.float32).ravel()
        high = np.asarray(self._action_space.high, dtype=np.float32).ravel()
        if low.shape == a.shape and high.shape == a.shape:
            a = np.clip(a, low, high)

        obs, _base_reward, terminated, truncated, info = self._env.step(a)
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


def _default_schema_exists() -> bool:
    return os.path.exists(DEFAULT_TEMP_SCHEMA)


if __name__ == "__main__":
    print(f"default_schema={DEFAULT_TEMP_SCHEMA}")
    print(f"exists={_default_schema_exists()}")
