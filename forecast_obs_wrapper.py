"""
Forecast Observation Wrapper for CityLearn.

Adds 24-hour lookahead features to the observation space:
  - Electricity pricing forecast (24 values)
  - Per-building non-shiftable load aggregated forecast (24 values)
  - Per-building solar generation aggregated forecast (24 values)
  - EV urgency signals per charger (8 values: urgency 0-1)
  - Hour-of-day sine/cosine for each forecast step (48 values)

Total additional features: 24 + 24 + 24 + 8 + 48 = 128 dims
Original obs: 153 dims → New obs: 281 dims
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ForecastObsWrapper(gym.ObservationWrapper):
    """Augment observations with 24h forecast features for intelligent lookahead."""

    def __init__(self, env, forecast_horizon: int = 24):
        super().__init__(env)
        self.forecast_horizon = forecast_horizon
        self._citylearn_env = None

        self.n_price_features = forecast_horizon
        self.n_load_features = forecast_horizon
        self.n_solar_features = forecast_horizon
        self.n_ev_features = 8
        self.n_time_features = forecast_horizon * 2

        self.n_extra = (self.n_price_features +
                        self.n_load_features +
                        self.n_solar_features +
                        self.n_ev_features +
                        self.n_time_features)

        orig_space = env.observation_space
        if isinstance(orig_space, spaces.Box):
            orig_low = orig_space.low.ravel()
            orig_high = orig_space.high.ravel()
            new_low = np.concatenate([orig_low, -np.ones(self.n_extra) * 10.0])
            new_high = np.concatenate([orig_high, np.ones(self.n_extra) * 10.0])
            self.observation_space = spaces.Box(
                low=new_low.astype(np.float32),
                high=new_high.astype(np.float32),
                dtype=np.float32)
        else:
            first = orig_space[0] if isinstance(orig_space, list) else orig_space
            orig_low = np.asarray(first.low).ravel()
            orig_high = np.asarray(first.high).ravel()
            new_low = np.concatenate([orig_low, -np.ones(self.n_extra) * 10.0])
            new_high = np.concatenate([orig_high, np.ones(self.n_extra) * 10.0])
            self.observation_space = spaces.Box(
                low=new_low.astype(np.float32),
                high=new_high.astype(np.float32),
                dtype=np.float32)

        print(f"[ForecastObs] Original obs dim: {len(orig_low)}, "
              f"Forecast features: {self.n_extra}, "
              f"New obs dim: {len(orig_low) + self.n_extra}")

    def _get_citylearn(self):
        if self._citylearn_env is not None:
            return self._citylearn_env
        cur = self.env
        seen = set()
        for _ in range(40):
            if cur is None or id(cur) in seen:
                break
            seen.add(id(cur))
            if (hasattr(cur, 'buildings') and hasattr(cur, 'time_step')
                    and hasattr(cur.buildings, '__len__') and len(cur.buildings) > 0):
                self._citylearn_env = cur
                return cur
            for attr in ('base', 'env', 'unwrapped', '_env', 'raw_env'):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
            else:
                break
        return None

    def _get_forecast_features(self) -> np.ndarray:
        city = self._get_citylearn()
        H = self.forecast_horizon
        features = np.zeros(self.n_extra, dtype=np.float32)

        if city is None:
            return features

        t_now = int(getattr(city, 'time_step', 0))
        buildings = list(getattr(city, 'buildings', []))
        if not buildings:
            return features

        offset = 0

        # 1. Electricity price forecast (normalized by mean)
        try:
            b0 = buildings[0]
            prices = np.asarray(b0.pricing.electricity_pricing, dtype=float)
            price_mean = max(np.mean(prices), 1e-6)
            for k in range(H):
                idx = t_now + k
                if idx < len(prices):
                    features[offset + k] = float(prices[idx]) / price_mean - 1.0
        except Exception:
            pass
        offset += self.n_price_features

        # 2. District load forecast (normalized by max)
        try:
            district_load = np.zeros(H, dtype=float)
            for b in buildings:
                nsl = np.asarray(
                    getattr(b, '_Building__energy_to_non_shiftable_load', []),
                    dtype=float)
                for k in range(H):
                    idx = t_now + k
                    if idx < len(nsl):
                        district_load[k] += nsl[idx]
            load_max = max(np.max(np.abs(district_load)), 1e-6)
            features[offset:offset + H] = district_load / load_max
        except Exception:
            pass
        offset += self.n_load_features

        # 3. District solar forecast (normalized by max)
        try:
            district_solar = np.zeros(H, dtype=float)
            for b in buildings:
                sg = np.asarray(
                    getattr(b, '_Building__solar_generation', []),
                    dtype=float)
                for k in range(H):
                    idx = t_now + k
                    if idx < len(sg):
                        district_solar[k] += sg[idx]
            solar_max = max(np.max(np.abs(district_solar)), 1e-6)
            features[offset:offset + H] = district_solar / solar_max
        except Exception:
            pass
        offset += self.n_solar_features

        # 4. EV urgency signals (8 chargers)
        try:
            ev_idx = 0
            for b_idx, b in enumerate(buildings):
                chargers = getattr(b, 'electric_vehicle_chargers', None) or []
                for ch in chargers:
                    if ev_idx >= 8:
                        break
                    sim = getattr(ch, 'charger_simulation',
                                  getattr(ch, '_Charger__charger_simulation', None))
                    if sim is None:
                        ev_idx += 1
                        continue
                    try:
                        state_arr = np.asarray(
                            getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                        dep_arr = np.asarray(
                            getattr(sim, '_electric_vehicle_departure_time'), dtype=float)
                        req_arr = np.asarray(
                            getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)

                        if t_now >= len(state_arr) or float(state_arr[t_now]) != 1.0:
                            features[offset + ev_idx] = 0.0
                            ev_idx += 1
                            continue

                        dep_hours = float(dep_arr[t_now])
                        req_soc = float(req_arr[t_now])
                        if not np.isfinite(req_soc):
                            req_soc = 1.0

                        ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                        ev_soc = 0.0
                        ev_cap = 0.0
                        if ev_obj is not None:
                            batt = getattr(ev_obj, 'battery', None)
                            if batt is not None:
                                ev_cap = float(getattr(batt, 'capacity', 0) or 0)
                                soc_data = getattr(batt, 'soc', None)
                                t_idx = max(0, t_now - 1)
                                if soc_data is not None:
                                    soc_np = np.asarray(soc_data, dtype=float)
                                    if 0 <= t_idx < len(soc_np):
                                        ev_soc = float(np.clip(soc_np[t_idx], 0, 1))

                        max_p = float(getattr(ch, 'max_charging_power', 0) or 0)
                        if isinstance(max_p, np.ndarray):
                            max_p = float(max_p.ravel()[0])

                        deficit = max(0.0, req_soc - ev_soc)
                        if deficit <= 1e-6 or not np.isfinite(dep_hours) or dep_hours <= 0:
                            urgency = 0.0
                        else:
                            if ev_cap > 0 and max_p > 0:
                                max_soc_per_step = (max_p * 0.95) / ev_cap
                                steps_needed = deficit / max(max_soc_per_step, 1e-9)
                                urgency = min(1.0, steps_needed / max(1, dep_hours))
                            else:
                                urgency = 1.0

                        features[offset + ev_idx] = float(urgency)
                    except Exception:
                        pass
                    ev_idx += 1
        except Exception:
            pass
        offset += self.n_ev_features

        # 5. Time-of-day encoding for each forecast step
        for k in range(H):
            hour = (t_now + k) % 24
            features[offset + 2 * k] = np.sin(2 * np.pi * hour / 24.0)
            features[offset + 2 * k + 1] = np.cos(2 * np.pi * hour / 24.0)

        return features

    def observation(self, obs):
        if isinstance(obs, (list, tuple)):
            obs_flat = np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs])
        else:
            obs_flat = np.asarray(obs, dtype=np.float32).ravel()

        forecast = self._get_forecast_features()
        return np.concatenate([obs_flat, forecast]).astype(np.float32)

    def reset(self, **kwargs):
        self._citylearn_env = None
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
