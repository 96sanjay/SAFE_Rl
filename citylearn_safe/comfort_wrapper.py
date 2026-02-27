from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym


class CityLearnComfortWrapper(gym.Env):
    """
    Dataset-agnostic comfort (temperature) constraint wrapper.
    Auto-enables only if LSTMDynamics exists in unwrapped CityLearn buildings.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: Any,
        *,
        enable_mode: str = "auto",
        w_comfort: float = 0.05,
        deadband: float = 0.5,
        tmin: float = 20.0,
        tmax: float = 26.0,
    ):
        super().__init__()
        self.base = base_env
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

        self.enable_mode = str(enable_mode).strip().lower()
        self.w_comfort = float(w_comfort)
        self.deadband = float(deadband)
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self._comfort_enabled = False

    def _get_citylearn_env(self):
        cur = self.base
        seen = set()
        for _ in range(40):
            if cur is None:
                break
            oid = id(cur)
            if oid in seen:
                break
            seen.add(oid)

            try:
                blds = getattr(cur, "buildings", None)
                ts = getattr(cur, "time_step", None)
                if blds is not None and hasattr(blds, "__len__") and len(blds) > 0 and ts is not None:
                    return cur
            except Exception:
                pass

            advanced = False
            for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
                if hasattr(cur, attr):
                    nxt = getattr(cur, attr, None)
                    if nxt is not None and nxt is not cur:
                        cur = nxt
                        advanced = True
                        break
            if not advanced:
                break
        return None

    def _has_lstm_dynamics(self, citylearn_env) -> bool:
        try:
            for b in getattr(citylearn_env, "buildings", []) if citylearn_env is not None else []:
                d = getattr(b, "dynamics", None)
                if isinstance(d, dict):
                    d = d.get("cooling") or next(iter(d.values()))
                if d is not None and ("LSTMDynamics" in type(d).__name__):
                    return True
        except Exception:
            return False
        return False

    def _state_time_index(self, citylearn_env) -> int:
        if citylearn_env is None:
            return 0
        t = int(getattr(citylearn_env, "time_step", 0))
        return max(0, t - 1)

    def _get_tin_tset(self, citylearn_env) -> Tuple[Optional[float], Optional[float]]:
        if citylearn_env is None or not getattr(citylearn_env, "buildings", None):
            return None, None

        b0 = citylearn_env.buildings[0]
        t_idx = self._state_time_index(citylearn_env)

        try:
            d = b0._get_observations_data()
            tin = d.get("indoor_dry_bulb_temperature", None)
            tset = None
            for k in (
                "indoor_dry_bulb_temperature_cooling_set_point",
                "indoor_dry_bulb_temperature_set_point",
                "indoor_dry_bulb_temperature_heating_set_point",
            ):
                if k in d:
                    tset = d.get(k, None)
                    break
            tin_f = float(tin) if tin is not None and np.isfinite(tin) else None
            tset_f = float(tset) if tset is not None and np.isfinite(tset) else None
            if tin_f is not None:
                return tin_f, tset_f
        except Exception:
            pass

        tin_f = None
        tset_f = None
        try:
            tin_series = getattr(b0, "indoor_dry_bulb_temperature", None)
            if tin_series is not None and hasattr(tin_series, "__len__") and len(tin_series) > t_idx:
                tin_f = float(tin_series[t_idx])
        except Exception:
            pass

        try:
            sp_series = getattr(b0, "indoor_dry_bulb_temperature_cooling_set_point", None)
            if sp_series is not None and hasattr(sp_series, "__len__") and len(sp_series) > t_idx:
                tset_f = float(sp_series[t_idx])
        except Exception:
            pass

        return tin_f, tset_f

    def _compute_comfort_raw(self, tin: float, tset: Optional[float]) -> float:
        if tset is not None and np.isfinite(tset):
            dev = abs(tin - float(tset)) - self.deadband
            return max(0.0, float(dev))
        hi = max(0.0, tin - self.tmax)
        lo = max(0.0, self.tmin - tin)
        return float(hi + lo)

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        obs, info = self.base.reset(seed=seed, options=options)
        info = dict(info) if info is not None else {}

        citylearn_env = self._get_citylearn_env()
        has_lstm = self._has_lstm_dynamics(citylearn_env)

        if self.enable_mode in ("0", "false", "no", "off"):
            self._comfort_enabled = False
        elif self.enable_mode in ("1", "true", "yes", "on"):
            self._comfort_enabled = True
        else:
            self._comfort_enabled = bool(has_lstm)

        info["comfort_enabled"] = 1.0 if self._comfort_enabled else 0.0
        info["cost_comfort"] = 0.0
        info["cost_comfort_raw"] = 0.0
        info["comfort_violation"] = 0.0
        info["comfort_tin"] = float("nan")
        info["comfort_tset"] = float("nan")

        if "cost" not in info:
            info["cost"] = 0.0

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base.step(action)
        info = dict(info) if info is not None else {}

        citylearn_env = self._get_citylearn_env()

        cost_raw = 0.0
        cost = 0.0
        viol = 0.0
        tin_out = float("nan")
        tset_out = float("nan")

        if self._comfort_enabled:
            tin, tset = self._get_tin_tset(citylearn_env)
            if tin is not None and np.isfinite(tin):
                tin_out = float(tin)
                if tset is not None and np.isfinite(tset):
                    tset_out = float(tset)

                cost_raw = self._compute_comfort_raw(float(tin), tset)
                cost = float(self.w_comfort) * float(cost_raw)
                viol = 1.0 if cost_raw > 0.0 else 0.0

        info["comfort_enabled"] = 1.0 if self._comfort_enabled else 0.0
        info["cost_comfort_raw"] = float(cost_raw)
        info["cost_comfort"] = float(cost)
        info["comfort_violation"] = float(viol)
        info["comfort_tin"] = float(tin_out)
        info["comfort_tset"] = float(tset_out)

        base_cost = float(info.get("cost", 0.0) or 0.0)
        info["cost"] = float(base_cost + cost)

        return obs, reward, terminated, truncated, info
