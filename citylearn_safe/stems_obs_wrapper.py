"""
stems_obs_wrapper.py — Spatial-Temporal Observation Wrappers for CityLearn + OmniSafe
======================================================================================

Designed for ablation study of the STEMS architecture components.
Drop this file into: citylearn_safe/stems_obs_wrapper.py

Ablation configurations (stack in train_omnisafe.py):
─────────────────────────────────────────────────────
  A) Base only          →  env
  B) + Forecast         →  ForecastObsWrapper(env)
  C) + History          →  SpatialTemporalHistoryWrapper(env)
  D) + Forecast+History →  STEMSCombinedWrapper(env)
  E) + Full STEMS       →  STEMSCombinedWrapper(env) + STEMSEncoder in model

Each wrapper is self-contained — use any combination.

Why separate wrappers instead of baking into the model:
  • OmniSafe 0.5.0 expects a standard gym obs space → wrappers handle this cleanly
  • Ablation = just change which wrappers you stack, same training script
  • No need to fork OmniSafe model code for A-D; only E needs model changes
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  FORECAST OBSERVATION WRAPPER  (improved version of your existing one)
# ═══════════════════════════════════════════════════════════════════════════════
#
# VERDICT ON YOUR EXISTING ForecastObsWrapper:
# ─────────────────────────────────────────────
# ✅ GOOD — keeps it; it provides genuinely useful information:
#   • Price forecast (24h)  → agent can time-shift battery charge/discharge
#   • Load forecast  (24h)  → helps anticipate C3 building-power violations
#   • Solar forecast (24h)  → agent knows when free energy arrives
#   • EV urgency    (8)     → CRITICAL for C1 — tells agent which EVs need
#                              charging soon vs which have time
#   • Time encoding (48)    → sin/cos hour-of-day for 24h ahead
#
# ⚠️  MINOR ISSUES in original (fixed below):
#   1. _get_citylearn() traversal can be fragile with deep wrapper chains
#      → Added 'base_env' attribute check (used by safety_env_v3)
#   2. No caching of building arrays — re-traverses every step
#      → Added lazy caching after first successful lookup
#   3. EV urgency computation accesses private attrs with try/except
#      → This is fine given CityLearn's API, but added fallback defaults
#
# WHY THIS MAKES SENSE FOR YOUR PROBLEM:
#   Your 4 constraints have fundamentally different time horizons:
#   • C1 (EV departure): needs FUTURE info — when does EV leave?
#   • C2 (battery SoC):  needs CURRENT info — where is SoC now?
#   • C3 (building power): needs FUTURE info — will NSL spike soon?
#   • C4 (grid power):    needs FUTURE info — will district peak soon?
#   Three of four constraints benefit from forecast features.
#   The ForecastObsWrapper directly addresses this.

class ForecastObsWrapper(gym.ObservationWrapper):
    """
    Appends 128-dim forecast features to the flat observation.
    Features: [24 price, 24 load, 24 solar, 8 EV_urgency, 48 time_enc]
    """

    def __init__(self, env: gym.Env, forecast_horizon: int = 24):
        super().__init__(env)
        self.forecast_horizon = forecast_horizon
        self._citylearn_env = None
        self._buildings_cache = None

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

        self._base_obs_dim = len(lo)
        self.observation_space = spaces.Box(
            low=np.concatenate([lo, -np.ones(self.n_extra) * 10.0]).astype(np.float32),
            high=np.concatenate([hi, np.ones(self.n_extra) * 10.0]).astype(np.float32),
            dtype=np.float32,
        )
        print(f"[ForecastObs] {self._base_obs_dim} + {self.n_extra} = "
              f"{self._base_obs_dim + self.n_extra} dims")

    # ── CityLearn env traversal (robust version) ──────────────────────────
    def _get_citylearn(self):
        if self._citylearn_env is not None:
            return self._citylearn_env
        cur = self.env
        seen = set()
        for _ in range(50):
            if cur is None or id(cur) in seen:
                break
            seen.add(id(cur))
            if (hasattr(cur, 'buildings') and hasattr(cur, 'time_step')
                    and hasattr(cur.buildings, '__len__') and len(cur.buildings) > 0):
                self._citylearn_env = cur
                self._buildings_cache = list(cur.buildings)
                return cur
            # Try multiple unwrapping paths
            for attr in ('base', 'env', 'unwrapped', '_env', 'raw_env', 'base_env'):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
            else:
                break
        return None

    # ── Forecast feature extraction ───────────────────────────────────────
    def _get_forecast(self) -> np.ndarray:
        city = self._get_citylearn()
        H = self.forecast_horizon
        feat = np.zeros(self.n_extra, dtype=np.float32)
        if city is None:
            return feat

        t = int(getattr(city, 'time_step', 0))
        buildings = self._buildings_cache or list(getattr(city, 'buildings', []))
        if not buildings:
            return feat

        off = 0

        # 1) Price forecast — normalised by running mean
        try:
            pr = np.asarray(buildings[0].pricing.electricity_pricing, dtype=float)
            pm = max(np.mean(pr), 1e-6)
            window = pr[t:t + H] if t + H <= len(pr) else pr[t:]
            feat[off:off + len(window)] = window / pm - 1.0
        except Exception:
            pass
        off += H

        # 2) District load forecast — normalised by max
        try:
            dl = np.zeros(min(H, 8760 - t))
            for b in buildings:
                nsl = np.asarray(
                    getattr(b, '_Building__energy_to_non_shiftable_load', []),
                    dtype=float)
                end = min(t + H, len(nsl))
                dl[:end - t] += nsl[t:end]
            mx = max(np.max(np.abs(dl)), 1e-6)
            feat[off:off + len(dl)] = dl / mx
        except Exception:
            pass
        off += H

        # 3) District solar forecast — normalised by max
        try:
            ds = np.zeros(min(H, 8760 - t))
            for b in buildings:
                sg = np.asarray(
                    getattr(b, '_Building__solar_generation', []),
                    dtype=float)
                end = min(t + H, len(sg))
                ds[:end - t] += sg[t:end]
            mx = max(np.max(np.abs(ds)), 1e-6)
            feat[off:off + len(ds)] = ds / mx
        except Exception:
            pass
        off += H

        # 4) EV urgency — one scalar per charger [0=fine, 1=critical]
        try:
            ei = 0
            for b in buildings:
                for ch in (getattr(b, 'electric_vehicle_chargers', None) or []):
                    if ei >= 8:
                        break
                    feat[off + ei] = self._compute_ev_urgency(ch, t)
                    ei += 1
        except Exception:
            pass
        off += 8

        # 5) Time-of-day encoding (sin/cos for 24h ahead)
        hours = (np.arange(H) + t) % 24
        feat[off::2][:H] = np.sin(2 * np.pi * hours / 24.0)
        feat[off + 1::2][:H] = np.cos(2 * np.pi * hours / 24.0)

        return feat

    @staticmethod
    def _compute_ev_urgency(charger, t: int) -> float:
        """Compute urgency ∈ [0,1] for a single EV charger at timestep t."""
        try:
            sim = getattr(charger, 'charger_simulation',
                          getattr(charger, '_Charger__charger_simulation', None))
            if sim is None:
                return 0.0
            sa = np.asarray(getattr(sim, '_electric_vehicle_charger_state'), dtype=float)
            if t >= len(sa) or float(sa[t]) != 1.0:
                return 0.0  # no EV connected

            da = np.asarray(getattr(sim, '_electric_vehicle_departure_time'), dtype=float)
            ra = np.asarray(getattr(sim, '_electric_vehicle_required_soc_departure'), dtype=float)
            dh = float(da[t])
            rs = float(ra[t]) if np.isfinite(ra[t]) else 1.0

            ev_obj = getattr(charger, 'connected_electric_vehicle', None)
            es, ec = 0.0, 0.0
            if ev_obj:
                bt = getattr(ev_obj, 'battery', None)
                if bt:
                    ec = float(getattr(bt, 'capacity', 0) or 0)
                    soc_arr = getattr(bt, 'soc', None)
                    if soc_arr is not None:
                        sn = np.asarray(soc_arr, dtype=float)
                        idx = max(0, t - 1)
                        if 0 <= idx < len(sn):
                            es = float(np.clip(sn[idx], 0, 1))

            mp = float(getattr(charger, 'max_charging_power', 0) or 0)
            if isinstance(mp, np.ndarray):
                mp = float(mp.ravel()[0])

            deficit = max(0.0, rs - es)
            if deficit <= 1e-6 or not np.isfinite(dh) or dh <= 0:
                return 0.0
            if ec > 0 and mp > 0:
                max_per_step = (mp * 0.95) / ec
                return min(1.0, (deficit / max(max_per_step, 1e-9)) / max(1.0, dh))
            return 1.0
        except Exception:
            return 0.0

    # ── Gym interface ─────────────────────────────────────────────────────
    def observation(self, obs):
        if isinstance(obs, (list, tuple)):
            flat = np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs])
        else:
            flat = np.asarray(obs, dtype=np.float32).ravel()
        return np.concatenate([flat, self._get_forecast()]).astype(np.float32)

    def reset(self, **kwargs):
        self._citylearn_env = None
        self._buildings_cache = None
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SPATIAL-TEMPORAL HISTORY WRAPPER  (the STEMS temporal component)
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY THIS IS NEEDED:
#   Your current obs is a single-timestep snapshot. The STEMS paper's
#   Transformer (Eq. 13-14) needs a WINDOW of past observations.
#   This wrapper maintains a sliding window and flattens it into the obs.
#
#   OmniSafe's MLP then sees: [current_obs | past_obs_t-1 | ... | past_obs_t-W]
#   This gives the MLP temporal context that approximates the Transformer's
#   attention — not as powerful, but zero model changes needed.
#
# HOW ABLATION WORKS:
#   • Without this wrapper: MLP sees 1 timestep only (Markov assumption)
#   • With this wrapper:    MLP sees W+1 timesteps (semi-recurrent via history)
#   • With full STEMS:      GCN+Transformer processes history properly
#
# DESIGN CHOICES:
#   • window_size=6: ~6 hours lookback. Enough to capture ramping trends
#     and EV charging progress. Smaller than STEMS paper's suggestion
#     because we're flattening (not using attention), so more = more noise.
#   • Per-node summary stats option: instead of raw history (huge obs),
#     compute [mean, std, delta, trend] per building over the window.
#     This compresses W*obs_dim into 4*obs_dim — much more MLP-friendly.

class SpatialTemporalHistoryWrapper(gym.ObservationWrapper):
    """
    Appends temporal history features to the observation.

    Two modes:
      mode='summary':  Appends [mean, std, delta, trend] over window.
                       Adds 4 × obs_dim features. DEFAULT — recommended.
      mode='raw':      Appends full flattened history window.
                       Adds window_size × obs_dim features. Use with small windows.
    """

    def __init__(
        self,
        env: gym.Env,
        window_size: int = 6,
        mode: str = "summary",  # 'summary' or 'raw'
        num_buildings: int = 17,
    ):
        super().__init__(env)
        self.window_size = window_size
        self.mode = mode
        self.num_buildings = num_buildings

        orig = env.observation_space
        if isinstance(orig, spaces.Box):
            lo = orig.low.ravel()
            hi = orig.high.ravel()
        else:
            first = orig[0] if isinstance(orig, list) else orig
            lo = np.asarray(first.low).ravel()
            hi = np.asarray(first.high).ravel()

        self._base_obs_dim = len(lo)

        if mode == "summary":
            # 4 summary stats per base obs dimension
            self.n_extra = 4 * self._base_obs_dim
        elif mode == "raw":
            # Full flattened window
            self.n_extra = window_size * self._base_obs_dim
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'summary' or 'raw'.")

        self.observation_space = spaces.Box(
            low=np.concatenate([lo, -np.ones(self.n_extra) * 100.0]).astype(np.float32),
            high=np.concatenate([hi, np.ones(self.n_extra) * 100.0]).astype(np.float32),
            dtype=np.float32,
        )

        self._history: deque = deque(maxlen=window_size)
        print(f"[HistoryObs] mode={mode}, window={window_size}, "
              f"{self._base_obs_dim} + {self.n_extra} = "
              f"{self._base_obs_dim + self.n_extra} dims")

    def _compute_summary(self) -> np.ndarray:
        """
        From the history window, compute 4 temporal features per obs dim:
          mean:  average over window (level)
          std:   standard deviation over window (volatility)
          delta: current - oldest (net change)
          trend: linear slope via least-squares (direction)
        """
        if len(self._history) < 2:
            return np.zeros(self.n_extra, dtype=np.float32)

        # Stack history: [W, obs_dim]
        H = np.stack(list(self._history), axis=0)
        W = H.shape[0]

        mean = H.mean(axis=0)                          # [obs_dim]
        std = H.std(axis=0)                             # [obs_dim]
        delta = H[-1] - H[0]                            # [obs_dim]

        # Linear trend: slope of least-squares fit
        t_axis = np.arange(W, dtype=np.float32)
        t_mean = t_axis.mean()
        t_var = ((t_axis - t_mean) ** 2).sum()
        if t_var > 1e-8:
            trend = ((t_axis[:, None] - t_mean) * (H - mean[None, :])).sum(axis=0) / t_var
        else:
            trend = np.zeros_like(mean)

        return np.concatenate([mean, std, delta, trend]).astype(np.float32)

    def _compute_raw(self) -> np.ndarray:
        """Flatten the full history window into the observation."""
        if len(self._history) == 0:
            return np.zeros(self.n_extra, dtype=np.float32)

        H = np.stack(list(self._history), axis=0)  # [W_actual, obs_dim]
        flat = H.ravel()

        # Pad if window not full yet
        if len(flat) < self.n_extra:
            flat = np.concatenate([np.zeros(self.n_extra - len(flat)), flat])

        return flat.astype(np.float32)

    def observation(self, obs):
        if isinstance(obs, (list, tuple)):
            flat = np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs])
        else:
            flat = np.asarray(obs, dtype=np.float32).ravel()

        # Store current obs in history buffer
        self._history.append(flat.copy())

        # Compute temporal features
        if self.mode == "summary":
            extra = self._compute_summary()
        else:
            extra = self._compute_raw()

        return np.concatenate([flat, extra]).astype(np.float32)

    def reset(self, **kwargs):
        self._history.clear()
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  BUILDING-GRAPH SPATIAL FEATURES WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════
#
# This implements the spatial component of STEMS (GCN equivalent) as a
# numpy-only observation wrapper. Instead of learnable GCN message passing,
# it computes inter-building interaction features analytically:
#   • Per-building power deltas (who is over/under threshold)
#   • Neighbourhood aggregation (mean, max of neighbours' features)
#   • Cross-building SoC disparity (battery balancing signal)
#
# This is the "poor man's GCN" — useful for ablation to test whether
# spatial features help AT ALL before investing in the full torch GCN.

class SpatialGraphFeaturesWrapper(gym.ObservationWrapper):
    """
    Appends inter-building spatial interaction features.
    No learnable parameters — pure analytic feature engineering.

    Features per building (N buildings):
      • power_headroom:     P_building_max - estimated_current_power  (N dims)
      • soc_spread:         per-battery SoC minus district mean SoC   (N dims)
      • neighbour_avg_soc:  mean SoC of neighbours (all-to-all graph) (N dims)
      • grid_contribution:  building power / total district power      (N dims)
    Total: 4*N dims
    """

    def __init__(
        self,
        env: gym.Env,
        num_buildings: int = 17,
        p_building_max: float = 2.273834,
    ):
        super().__init__(env)
        self.num_buildings = num_buildings
        self.p_building_max = p_building_max
        self._citylearn_env = None
        self.n_extra = num_buildings * 4  # 68

        orig = env.observation_space
        if isinstance(orig, spaces.Box):
            lo = orig.low.ravel()
            hi = orig.high.ravel()
        else:
            first = orig[0] if isinstance(orig, list) else orig
            lo = np.asarray(first.low).ravel()
            hi = np.asarray(first.high).ravel()

        self._base_obs_dim = len(lo)
        self.observation_space = spaces.Box(
            low=np.concatenate([lo, -np.ones(self.n_extra) * 10.0]).astype(np.float32),
            high=np.concatenate([hi, np.ones(self.n_extra) * 10.0]).astype(np.float32),
            dtype=np.float32,
        )
        print(f"[SpatialObs] {self._base_obs_dim} + {self.n_extra} = "
              f"{self._base_obs_dim + self.n_extra} dims")

    def _get_citylearn(self):
        if self._citylearn_env is not None:
            return self._citylearn_env
        cur = self.env
        seen = set()
        for _ in range(50):
            if cur is None or id(cur) in seen:
                break
            seen.add(id(cur))
            if (hasattr(cur, 'buildings') and hasattr(cur, 'time_step')
                    and len(getattr(cur, 'buildings', [])) > 0):
                self._citylearn_env = cur
                return cur
            for attr in ('base', 'env', 'unwrapped', '_env', 'raw_env', 'base_env'):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
            else:
                break
        return None

    def _get_spatial_features(self) -> np.ndarray:
        city = self._get_citylearn()
        N = self.num_buildings
        feat = np.zeros(self.n_extra, dtype=np.float32)
        if city is None:
            return feat

        buildings = list(getattr(city, 'buildings', []))[:N]
        if len(buildings) < N:
            return feat

        off = 0

        # Collect per-building data
        socs = np.zeros(N, dtype=np.float32)
        powers = np.zeros(N, dtype=np.float32)

        for i, b in enumerate(buildings):
            # Battery SoC
            try:
                es = getattr(b, 'electrical_storage', None)
                if es is not None:
                    soc_val = getattr(es, 'soc', [0.0])
                    if hasattr(soc_val, '__len__') and len(soc_val) > 0:
                        socs[i] = float(soc_val[-1])
                    else:
                        socs[i] = float(soc_val)
            except Exception:
                pass

            # Net building power
            try:
                net = getattr(b, 'net_electricity_consumption', [0.0])
                if hasattr(net, '__len__') and len(net) > 0:
                    powers[i] = float(net[-1])
                else:
                    powers[i] = float(net)
            except Exception:
                pass

        # Feature 1: Power headroom  (positive = safe, negative = violating C3)
        headroom = self.p_building_max - np.abs(powers)
        feat[off:off + N] = np.clip(headroom / self.p_building_max, -5.0, 1.0)
        off += N

        # Feature 2: SoC spread  (deviation from district mean)
        soc_mean = socs.mean()
        feat[off:off + N] = socs - soc_mean
        off += N

        # Feature 3: Neighbour average SoC  (all-to-all minus self)
        if N > 1:
            total_soc = socs.sum()
            feat[off:off + N] = (total_soc - socs) / (N - 1)
        off += N

        # Feature 4: Grid contribution fraction
        total_power = max(np.sum(np.maximum(0, powers)), 1e-6)
        feat[off:off + N] = np.maximum(0, powers) / total_power
        off += N

        return feat

    def observation(self, obs):
        if isinstance(obs, (list, tuple)):
            flat = np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs])
        else:
            flat = np.asarray(obs, dtype=np.float32).ravel()
        return np.concatenate([flat, self._get_spatial_features()]).astype(np.float32)

    def reset(self, **kwargs):
        self._citylearn_env = None
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  COMBINED STEMS WRAPPER  (forecast + spatial + temporal all-in-one)
# ═══════════════════════════════════════════════════════════════════════════════

class STEMSCombinedWrapper(gym.Wrapper):
    """
    Stacks all three observation augmentation layers in the correct order.
    This is the recommended wrapper for maximum performance.

    Observation layout:
      [base_obs | forecast_128 | spatial_68 | temporal_summary_4×base]

    For your env with base_obs ≈ 255:
      255 + 128 + 68 + 1020 = 1471 dims  (summary mode)
      or with raw temporal (window=6):
      255 + 128 + 68 + 1530 = 1981 dims  (raw mode, not recommended)
    """

    def __init__(
        self,
        env: gym.Env,
        forecast_horizon: int = 24,
        temporal_window: int = 6,
        temporal_mode: str = "summary",
        num_buildings: int = 17,
        p_building_max: float = 2.273834,
        enable_forecast: bool = True,
        enable_spatial: bool = True,
        enable_temporal: bool = True,
    ):
        """
        Args:
            env:              Base CityLearn environment (or safety wrapper).
            forecast_horizon: Hours ahead for price/load/solar forecasts.
            temporal_window:  Steps of history to keep.
            temporal_mode:    'summary' (4×obs) or 'raw' (W×obs).
            num_buildings:    Number of buildings in the district.
            p_building_max:   Building power threshold for C3 (2.273834 kW).
            enable_forecast:  Toggle forecast features.
            enable_spatial:   Toggle spatial features.
            enable_temporal:  Toggle temporal history features.
        """
        # Stack wrappers in order: spatial → forecast → temporal
        # (temporal goes last so its history includes forecast+spatial features)
        wrapped = env
        self._components = []

        if enable_spatial:
            wrapped = SpatialGraphFeaturesWrapper(
                wrapped, num_buildings=num_buildings,
                p_building_max=p_building_max)
            self._components.append("spatial")

        if enable_forecast:
            wrapped = ForecastObsWrapper(
                wrapped, forecast_horizon=forecast_horizon)
            self._components.append("forecast")

        if enable_temporal:
            wrapped = SpatialTemporalHistoryWrapper(
                wrapped, window_size=temporal_window,
                mode=temporal_mode, num_buildings=num_buildings)
            self._components.append("temporal")

        super().__init__(wrapped)
        print(f"[STEMSCombined] Active components: {self._components}")
        print(f"[STEMSCombined] Final obs dim: {self.observation_space.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  OMNISAFE INTEGRATION SNIPPET
# ═══════════════════════════════════════════════════════════════════════════════
#
# Paste this into your train_omnisafe.py or omni_env_v2_shield.py
# where you construct the environment.
#
# ──────────────────────────────────────────────────────────────────────────────
#
# OPTION 1: Quick integration — just wrappers, no model changes
# (Ablation levels A-D.  The standard OmniSafe MLP handles everything.)
#
# ```python
# import os
# from citylearn_safe.stems_obs_wrapper import (
#     ForecastObsWrapper,
#     SpatialTemporalHistoryWrapper,
#     SpatialGraphFeaturesWrapper,
#     STEMSCombinedWrapper,
# )
#
# def make_stems_env():
#     """
#     Drop-in replacement for your current env factory function.
#     Control ablation via environment variables.
#     """
#     from scripts.make_env import make_base_env
#     from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
#
#     base = make_base_env(central_agent=True)
#     env = CityLearnSafetyEnvV3(base, soc_min=0.0, soc_max=0.95)
#
#     # Read ablation config from environment variables
#     stems_mode = os.environ.get("STEMS_MODE", "none")
#     # "none"      = no STEMS features (baseline)
#     # "forecast"  = forecast only
#     # "spatial"   = spatial only
#     # "temporal"  = temporal history only
#     # "combined"  = all three
#
#     if stems_mode == "forecast":
#         env = ForecastObsWrapper(env)
#     elif stems_mode == "spatial":
#         env = SpatialGraphFeaturesWrapper(env)
#     elif stems_mode == "temporal":
#         env = SpatialTemporalHistoryWrapper(env, window_size=6, mode="summary")
#     elif stems_mode == "combined":
#         env = STEMSCombinedWrapper(env,
#             enable_forecast=True,
#             enable_spatial=True,
#             enable_temporal=True,
#         )
#     # else: no wrappers — pure baseline
#
#     return env
# ```
#
# Then in set_env_baselines.sh, add:
#   export STEMS_MODE=combined   # or "none", "forecast", "spatial", "temporal"
#
# And increase hidden layer sizes in your YAML configs to handle bigger obs:
#   model_cfgs:
#     actor:
#       hidden_sizes: [1024, 512, 256]  # was [512, 512, 256]
#     critic:
#       hidden_sizes: [1024, 512, 256]
#
# ──────────────────────────────────────────────────────────────────────────────
#
# OPTION 2: Full STEMS encoder (torch GCN + Transformer in the model)
# (Ablation level E.  Requires custom OmniSafe actor/critic.)
#
# This is more complex — you need to either:
#   a) Monkey-patch OmniSafe's actor to use STEMS encoder as first layer
#   b) Register a custom model with OmniSafe
#   c) Use the wrapper approach but with torch-based obs transformation
#
# The recommended approach (a) for OmniSafe 0.5.0:
#
# ```python
# # In train_omnisafe.py, AFTER creating the OmniSafe agent:
#
# import torch
# from stems_encoder import STEMSEncoder
#
# def inject_stems_encoder(agent, obs_dim, device='cpu'):
#     """
#     Monkey-patches OmniSafe agent to use STEMS encoder.
#     The encoder replaces the first linear layer of actor & critic.
#     """
#     encoder = STEMSEncoder(
#         obs_dim=obs_dim,
#         num_buildings=17,
#         hidden_dim=64,
#         num_gcn_layers=3,
#         num_heads=4,
#         temporal_window=24,
#         output_dim=256,  # feeds into remaining MLP layers
#     ).to(device)
#
#     # Store encoder on agent so its parameters get saved/loaded
#     agent.stems_encoder = encoder
#
#     # Patch the actor's forward to route through encoder first
#     original_actor_forward = agent.actor.forward
#
#     def patched_actor_forward(obs):
#         encoded = encoder(obs, push_history=True)
#         return original_actor_forward(encoded)
#
#     agent.actor.forward = patched_actor_forward
#
#     # Same for critic
#     original_critic_forward = agent.reward_critic.forward
#
#     def patched_critic_forward(obs):
#         encoded = encoder(obs, push_history=False)  # don't double-push
#         return original_critic_forward(encoded)
#
#     agent.reward_critic.forward = patched_critic_forward
#
#     # Add encoder params to the optimiser
#     encoder_params = list(encoder.parameters())
#     agent.actor_optimizer.add_param_group({'params': encoder_params})
#
#     print(f"[STEMS] Injected encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
#     return agent
# ```
#
# ⚠️  Option 2 is more powerful but also more fragile.  Start with Option 1
#     to establish baselines, then try Option 2 if wrapper results are promising.
#
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ABLATION SHELL COMMANDS — ready to copy-paste
# ═══════════════════════════════════════════════════════════════════════════════
#
# # ── A) Baseline: no STEMS features ──
# export STEMS_MODE=none
# source set_env_baselines.sh && python scripts/train_omnisafe.py --cfg configs/on-policy/trpolag_v2g.yaml
#
# # ── B) Forecast only ──
# export STEMS_MODE=forecast
# source set_env_baselines.sh && python scripts/train_omnisafe.py --cfg configs/on-policy/trpolag_v2g.yaml
#
# # ── C) Spatial only ──
# export STEMS_MODE=spatial
# source set_env_baselines.sh && python scripts/train_omnisafe.py --cfg configs/on-policy/trpolag_v2g.yaml
#
# # ── D) Temporal only ──
# export STEMS_MODE=temporal
# source set_env_baselines.sh && python scripts/train_omnisafe.py --cfg configs/on-policy/trpolag_v2g.yaml
#
# # ── E) All combined (wrapper-based STEMS) ──
# export STEMS_MODE=combined
# source set_env_baselines.sh && python scripts/train_omnisafe.py --cfg configs/on-policy/trpolag_v2g.yaml
#


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("STEMS Observation Wrapper — Standalone Test")
    print("=" * 70)

    # Create a dummy env that mimics CityLearn's flat obs
    OBS_DIM = 255  # adjust to your actual base obs_dim

    class DummyCityLearnEnv(gym.Env):
        """Minimal mock for testing wrappers without real CityLearn."""
        def __init__(self, obs_dim=OBS_DIM):
            self.observation_space = spaces.Box(
                low=-np.ones(obs_dim, dtype=np.float32),
                high=np.ones(obs_dim, dtype=np.float32),
                dtype=np.float32)
            self.action_space = spaces.Box(
                low=-np.ones(25, dtype=np.float32),
                high=np.ones(25, dtype=np.float32),
                dtype=np.float32)
            self.time_step = 0

        def reset(self, **kw):
            self.time_step = 0
            return np.random.randn(OBS_DIM).astype(np.float32), {}

        def step(self, action):
            self.time_step += 1
            obs = np.random.randn(OBS_DIM).astype(np.float32)
            return obs, 0.0, self.time_step >= 100, False, {}

    env = DummyCityLearnEnv()
    print(f"\nBase obs dim: {env.observation_space.shape}")

    # Test each wrapper independently
    print("\n── Test 1: ForecastObsWrapper ──")
    w1 = ForecastObsWrapper(DummyCityLearnEnv())
    obs, _ = w1.reset()
    print(f"  obs shape: {obs.shape}")
    for _ in range(3):
        obs, _, _, _, _ = w1.step(w1.action_space.sample())
    print(f"  after 3 steps: {obs.shape}")

    print("\n── Test 2: SpatialTemporalHistoryWrapper (summary) ──")
    w2 = SpatialTemporalHistoryWrapper(DummyCityLearnEnv(), window_size=6, mode="summary")
    obs, _ = w2.reset()
    print(f"  obs shape: {obs.shape}")
    for _ in range(8):
        obs, _, _, _, _ = w2.step(w2.action_space.sample())
    print(f"  after 8 steps (window full): {obs.shape}")

    print("\n── Test 3: SpatialGraphFeaturesWrapper ──")
    w3 = SpatialGraphFeaturesWrapper(DummyCityLearnEnv())
    obs, _ = w3.reset()
    print(f"  obs shape: {obs.shape}")

    print("\n── Test 4: STEMSCombinedWrapper (all enabled) ──")
    w4 = STEMSCombinedWrapper(
        DummyCityLearnEnv(),
        enable_forecast=True,
        enable_spatial=True,
        enable_temporal=True,
        temporal_window=6,
        temporal_mode="summary",
    )
    obs, _ = w4.reset()
    print(f"  obs shape: {obs.shape}")
    for _ in range(10):
        obs, _, _, _, _ = w4.step(w4.action_space.sample())
    print(f"  after 10 steps: {obs.shape}")

    print("\n── Test 5: STEMSCombinedWrapper (forecast only — for ablation) ──")
    w5 = STEMSCombinedWrapper(
        DummyCityLearnEnv(),
        enable_forecast=True,
        enable_spatial=False,
        enable_temporal=False,
    )
    obs, _ = w5.reset()
    print(f"  obs shape: {obs.shape}")

    # Summary table
    print("\n" + "=" * 70)
    print("ABLATION OBS DIMENSIONS SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<35} {'Obs Dim':>8}")
    print(f"  {'─' * 35} {'─' * 8}")
    configs = [
        ("A) Base only", OBS_DIM),
        ("B) + Forecast", OBS_DIM + 128),
        ("C) + Spatial", OBS_DIM + 68),
        ("D) + Temporal (summary, W=6)", OBS_DIM + 4 * OBS_DIM),
        ("E) Combined (all three)", w4.observation_space.shape[0]),
    ]
    for name, dim in configs:
        print(f"  {name:<35} {dim:>8}")
    print()

