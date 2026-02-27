"""
Forecast Observation Wrapper — adds 24h lookahead to obs space.
  +24 price, +24 load, +24 solar, +8 EV urgency, +48 time encoding = 128 dims
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ForecastObsWrapper(gym.ObservationWrapper):

    def __init__(self, env, forecast_horizon: int = 24):
        super().__init__(env)
        self.forecast_horizon = forecast_horizon
        self._citylearn_env = None
        H = forecast_horizon
        self.n_extra = H + H + H + 8 + H * 2  # 128

        orig = env.observation_space
        if isinstance(orig, spaces.Box):
            lo = orig.low.ravel()
            hi = orig.high.ravel()
        else:
            first = orig[0] if isinstance(orig, list) else orig
            lo = np.asarray(first.low).ravel()
            hi = np.asarray(first.high).ravel()

        self.observation_space = spaces.Box(
            low=np.concatenate([lo, -np.ones(self.n_extra) * 10.0]).astype(np.float32),
            high=np.concatenate([hi, np.ones(self.n_extra) * 10.0]).astype(np.float32),
            dtype=np.float32)
        print(f"[ForecastObs] {len(lo)} + {self.n_extra} = {len(lo) + self.n_extra} dims")

    def _get_citylearn(self):
        if self._citylearn_env is not None:
            return self._citylearn_env
        cur = self.env
        seen = set()
        for _ in range(40):
            if cur is None or id(cur) in seen:
                break
            seen.add(id(cur))
            if hasattr(cur, 'buildings') and hasattr(cur, 'time_step') and hasattr(cur.buildings, '__len__') and len(cur.buildings) > 0:
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

    def _get_forecast(self) -> np.ndarray:
        city = self._get_citylearn()
        H = self.forecast_horizon
        feat = np.zeros(self.n_extra, dtype=np.float32)
        if city is None:
            return feat
        t = int(getattr(city, 'time_step', 0))
        buildings = list(getattr(city, 'buildings', []))
        if not buildings:
            return feat
        off = 0
        # 1) price forecast (normalized by mean)
        try:
            pr = np.asarray(buildings[0].pricing.electricity_pricing, dtype=float)
            pm = max(np.mean(pr), 1e-6)
            for k in range(H):
                if t + k < len(pr):
                    feat[off + k] = float(pr[t + k]) / pm - 1.0
        except Exception:
            pass
        off += H
        # 2) district load forecast (normalized by max)
        try:
            dl = np.zeros(H)
            for b in buildings:
                nsl = np.asarray(getattr(b, '_Building__energy_to_non_shiftable_load', []), dtype=float)
                for k in range(H):
                    if t + k < len(nsl):
                        dl[k] += nsl[t + k]
            mx = max(np.max(np.abs(dl)), 1e-6)
            feat[off:off + H] = dl / mx
        except Exception:
            pass
        off += H
        # 3) district solar forecast (normalized by max)
        try:
            ds = np.zeros(H)
            for b in buildings:
                sg = np.asarray(getattr(b, '_Building__solar_generation', []), dtype=float)
                for k in range(H):
                    if t + k < len(sg):
                        ds[k] += sg[t + k]
            mx = max(np.max(np.abs(ds)), 1e-6)
            feat[off:off + H] = ds / mx
        except Exception:
            pass
        off += H
        # 4) EV urgency (8 chargers)
        try:
            ei = 0
            for b in buildings:
                for ch in (getattr(b, 'electric_vehicle_chargers', None) or []):
                    if ei >= 8:
                        break
                    sim = getattr(ch, 'charger_simulation', getattr(ch, '_Charger__charger_simulation', None))
                    if sim is None:
                        ei += 1; continue
                    try:
                        sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
                        da = np.asarray(getattr(sim, '_electric_vehicle_departure_time'), dtype=float)
                        ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
                        if t >= len(sa) or float(sa[t]) != 1.0:
                            ei += 1; continue
                        dh = float(da[t]); rs = float(ra[t])
                        if not np.isfinite(rs): rs = 1.0
                        ev_obj = getattr(ch, 'connected_electric_vehicle', None)
                        es, ec = 0.0, 0.0
                        if ev_obj:
                            bt = getattr(ev_obj, 'battery', None)
                            if bt:
                                ec = float(getattr(bt, 'capacity', 0) or 0)
                                sd = getattr(bt, 'soc', None)
                                if sd is not None:
                                    sn = np.asarray(sd, dtype=float)
                                    ti = max(0, t - 1)
                                    if 0 <= ti < len(sn): es = float(np.clip(sn[ti], 0, 1))
                        mp = float(getattr(ch, 'max_charging_power', 0) or 0)
                        if isinstance(mp, np.ndarray): mp = float(mp.ravel()[0])
                        deficit = max(0.0, rs - es)
                        if deficit <= 1e-6 or not np.isfinite(dh) or dh <= 0:
                            urg = 0.0
                        elif ec > 0 and mp > 0:
                            mps = (mp * 0.95) / ec
                            urg = min(1.0, (deficit / max(mps, 1e-9)) / max(1, dh))
                        else:
                            urg = 1.0
                        feat[off + ei] = urg
                    except Exception:
                        pass
                    ei += 1
        except Exception:
            pass
        off += 8
        # 5) time-of-day encoding (sin/cos)
        for k in range(H):
            h = (t + k) % 24
            feat[off + 2 * k] = np.sin(2 * np.pi * h / 24.0)
            feat[off + 2 * k + 1] = np.cos(2 * np.pi * h / 24.0)
        return feat

    def observation(self, obs):
        if isinstance(obs, (list, tuple)):
            flat = np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs])
        else:
            flat = np.asarray(obs, dtype=np.float32).ravel()
        return np.concatenate([flat, self._get_forecast()]).astype(np.float32)

    def reset(self, **kwargs):
        self._citylearn_env = None
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
