from __future__ import annotations

"""
Central-agent wrapper + EV departure KPI utilities for CityLearn + EVLearn.

This file is self-contained:
- Does NOT import citylearn_safe.extractors.
- Works with CityLearnEnv(schema=..., central_agent=False).
- Provides:
    * unwrap_to_raw_citylearn_env
    * current_time_index + SoC helpers
    * ev_departure_cost_components (linear deficits)
    * ev_departure_safe_cost (scalar cost for safe RL, avoidable & optional quadratic)
    * CentralAgentEnv: single-agent Gymnasium/Gym interface over multi-building env.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 0. Optional Gymnasium / Gym import (for RL wrappers)
# ---------------------------------------------------------------------------

try:  # Gymnasium (preferred)
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # Fallback to classic Gym
    import gym  # type: ignore
    from gym import spaces  # type: ignore


# ---------------------------------------------------------------------------
# 1. Unwrapping helpers
# ---------------------------------------------------------------------------

def unwrap_to_raw_citylearn_env(env: Any) -> Any:
    """Follow .base / .env / .unwrapped links until we hit the raw CityLearnEnv.

    This is defensive: it works through multiple nested Gym wrappers.
    """
    seen = set()
    e = env
    while True:
        if id(e) in seen:
            break
        seen.add(id(e))

        # CityLearn's own Agent wraps env as .env
        if hasattr(e, "env"):
            e_next = getattr(e, "env")
        elif hasattr(e, "base"):
            e_next = getattr(e, "base")
        elif hasattr(e, "unwrapped"):
            e_next = getattr(e, "unwrapped")
        else:
            break

        if e_next is None:
            break
        e = e_next

    return e


# ---------------------------------------------------------------------------
# 2. Generic SoC helpers
# ---------------------------------------------------------------------------

def current_time_index(base_env: Any) -> int:
    """
    Return the index in battery.soc that represents the *current* SoC.

    CityLearn stores SoC as a time series. After each step, the new value
    is written at index (time_step - 1). At reset (time_step = 0), the
    current value is at index 0.

    So:
        t_soc = 0                     if time_step <= 0
        t_soc = time_step - 1         otherwise
    """
    env = unwrap_to_raw_citylearn_env(base_env)
    t = int(getattr(env, "time_step", 0))
    return 0 if t <= 0 else t - 1


def battery_soc_list(base_env: Any) -> List[float]:
    """Return SoC (0..1) for every building-level electricity storage, if any."""
    env = unwrap_to_raw_citylearn_env(base_env)
    vals: List[float] = []

    for b in getattr(env, "buildings", []):
        es = getattr(b, "electrical_storage", None)
        if es is None:
            es = getattr(b, "electricity_storage", None)  # older names
        soc = getattr(es, "soc", None) if es is not None else None
        if soc is None:
            continue
        try:
            vals.append(float(soc))
        except Exception:
            continue

    return vals


def battery_soc_stats(base_env: Any) -> Dict[str, float]:
    vals = battery_soc_list(base_env)
    if not vals:
        return {
            "soc_mean": 0.5,
            "soc_min_obs": 0.5,
            "soc_max_obs": 0.5,
            "num_storages": 0.0,
        }

    return {
        "soc_mean": float(sum(vals) / len(vals)),
        "soc_min_obs": float(min(vals)),
        "soc_max_obs": float(max(vals)),
        "num_storages": float(len(vals)),
    }


def ev_soc_by_charger(base_env: Any) -> Dict[Tuple[str, str], float]:
    """
    Return a dict mapping (building_name, charger_name) -> current EV SoC (0..1),
    for all chargers that currently have a connected EV.

    Uses the internal EV battery.soc time series, evaluated at current_time_index().
    """
    env = unwrap_to_raw_citylearn_env(base_env)
    t = current_time_index(env)

    out: Dict[Tuple[str, str], float] = {}

    for b in getattr(env, "buildings", []):
        bname = getattr(b, "name", "UNKNOWN_BUILDING")
        chargers = getattr(b, "electric_vehicle_chargers", [])
        for ch in chargers or []:
            cname = getattr(ch, "name", getattr(ch, "charger_id", "UNKNOWN_CHARGER"))
            ev = getattr(ch, "connected_electric_vehicle", None)
            if ev is None:
                continue

            batt = getattr(ev, "battery", None)
            if batt is None or not hasattr(batt, "soc"):
                continue

            try:
                soc_series = np.asarray(getattr(batt, "soc"), dtype=float)
                if 0 <= t < soc_series.shape[0]:
                    out[(bname, cname)] = float(soc_series[t])
            except Exception:
                continue

    return out


def ev_soc_by_name(base_env: Any) -> Dict[str, float]:
    """Return a dict mapping EV.name -> current SoC (0..1) for all connected EVs."""
    env = unwrap_to_raw_citylearn_env(base_env)
    t = current_time_index(env)

    out: Dict[str, float] = {}

    for b in getattr(env, "buildings", []):
        chargers = getattr(b, "electric_vehicle_chargers", [])
        for ch in chargers or []:
            ev = getattr(ch, "connected_electric_vehicle", None)
            if ev is None:
                continue
            name = getattr(ev, "name", "EV_UNKNOWN")
            batt = getattr(ev, "battery", None)
            if batt is None or not hasattr(batt, "soc"):
                continue

            try:
                soc_series = np.asarray(getattr(batt, "soc"), dtype=float)
                if 0 <= t < soc_series.shape[0]:
                    out[str(name)] = float(soc_series[t])
            except Exception:
                continue

    return out


def ev_soc_stats(base_env: Any) -> Dict[str, float]:
    """Aggregate stats over all connected EVs' current SoC."""
    vals = list(ev_soc_by_name(base_env).values())
    if not vals:
        return {
            "ev_soc_mean": 0.5,
            "ev_soc_min": 0.5,
            "ev_soc_max": 0.5,
            "num_evs": 0.0,
        }

    return {
        "ev_soc_mean": float(sum(vals) / len(vals)),
        "ev_soc_min": float(min(vals)),
        "ev_soc_max": float(max(vals)),
        "num_evs": float(len(vals)),
    }


# ---------------------------------------------------------------------------
# 3. EV departure KPI (components + detailed per-departure records)
# ---------------------------------------------------------------------------

@dataclass
class EVDepartureRecord:
    """Per-departure data at the current time step."""

    ev_id: str
    charger_id: str
    time_step: int
    required_soc: float      # r in [0, 1]
    actual_soc: float        # s_actual in [0, 1]
    arrival_soc: float       # s_arr in [0, 1]
    capacity_kwh: float      # EV battery capacity
    deficit_actual: float    # max(0, r - s_actual)
    deficit_avoidable: float # part that policy could have fixed
    deficit_unavoidable: float  # physically impossible part


def _get_dt_hours(env: Any) -> Optional[float]:
    seconds_per_step = getattr(env, "seconds_per_time_step", None)
    if seconds_per_step is None:
        return None
    try:
        return float(seconds_per_step) / 3600.0
    except Exception:
        return None


def _normalize_ev_id(x: Any) -> Optional[str]:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return x
    return None


def _is_valid_ev_id(x: Any) -> bool:
    name = _normalize_ev_id(x)
    return bool(name)


def _get_charger_pmax_kw(ch: Any) -> Optional[float]:
    # Public property in current CityLearn; fall back to private for older versions.
    p = getattr(ch, "max_charging_power", None)
    if p is None:
        p = getattr(ch, "_Charger__max_charging_power", None)

    if isinstance(p, np.ndarray):
        p = float(p.reshape(-1)[0])
    elif p is not None:
        p = float(p)

    return p


def _collect_ev_meta(env: Any) -> Dict[str, Dict[str, Any]]:
    """Map EV id/name -> {'cap_kwh', 'soc_series'}."""
    ev_by_name: Dict[str, Dict[str, Any]] = {}

    for ev in getattr(env, "electric_vehicles", []):
        name = getattr(ev, "name", None)
        if name is None:
            name = getattr(ev, "_ElectricVehicle__name", None)
        if isinstance(name, (bytes, np.bytes_)):
            name = name.decode("utf-8", errors="ignore")
        if not isinstance(name, str):
            continue

        batt = getattr(ev, "battery", None)
        if batt is None:
            batt = getattr(ev, "_ElectricVehicle__battery", None)

        cap_kwh = None
        soc_series = None
        if batt is not None:
            cap_kwh = getattr(batt, "capacity", None)
            if cap_kwh is None:
                cap_kwh = getattr(batt, "_EnergyStorage__capacity", None)

            soc_series = getattr(batt, "soc", None)
            if soc_series is None:
                soc_series = getattr(batt, "_EnergyStorage__soc", None)

        if isinstance(cap_kwh, np.ndarray):
            cap_kwh = float(cap_kwh.reshape(-1)[0])
        elif cap_kwh is not None:
            cap_kwh = float(cap_kwh)

        if soc_series is not None:
            soc_series = np.asarray(soc_series, dtype=float)
        else:
            soc_series = None

        ev_by_name[name] = {
            "cap_kwh": cap_kwh,
            "soc_series": soc_series,
        }

    return ev_by_name


def _iter_chargers(env: Any) -> List[Any]:
    chargers: List[Any] = []
    for b in getattr(env, "buildings", []):
        chs = getattr(b, "electric_vehicle_chargers", None)
        if chs is None:
            chs = getattr(b, "_Building__electric_vehicle_chargers", None)
        if chs:
            chargers.extend(chs)
    return chargers


def _get_charger_sim(ch: Any) -> Optional[Any]:
    # Public property in CityLearn v2.5+, fallback to private for older versions.
    sim = getattr(ch, "charger_simulation", None)
    if sim is None:
        sim = getattr(ch, "_Charger__charger_simulation", None)
    return sim


def _ev_departure_records(env: Any) -> List[EVDepartureRecord]:
    """Internal helper: compute per-departure records at current time step."""
    env = unwrap_to_raw_citylearn_env(env)
    t_state = int(getattr(env, "time_step", 0))
    if t_state < 0:
        t_state = 0

    t_soc = current_time_index(env)
    dt_hours = _get_dt_hours(env)
    ev_by_name = _collect_ev_meta(env)
    chargers = _iter_chargers(env)

    if not chargers or not ev_by_name:
        return []

    records: List[EVDepartureRecord] = []

    for ch in chargers:
        sim = _get_charger_sim(ch)
        if sim is None:
            continue

        # ChargerSimulation arrays (one entry per time-step) – see CityLearn docs.
        state = np.asarray(getattr(sim, "electric_vehicle_charger_state"), dtype=float)
        dep_time = np.asarray(getattr(sim, "electric_vehicle_departure_time"), dtype=float)
        req_soc = np.asarray(getattr(sim, "electric_vehicle_required_soc_departure"), dtype=float)
        ev_id = np.asarray(getattr(sim, "electric_vehicle_id"))
        arr_time = np.asarray(getattr(sim, "electric_vehicle_estimated_arrival_time"), dtype=float)
        arr_soc = np.asarray(getattr(sim, "electric_vehicle_estimated_soc_arrival"), dtype=float)

        if state.ndim != 1 or t_state >= state.shape[0]:
            continue

        s_state = float(state[t_state])
        d = float(dep_time[t_state])
        r = float(req_soc[t_state])
        e = ev_id[t_state]
        at_now = float(arr_time[t_state])
        as_now = float(arr_soc[t_state])

        # Departure event? Dataset conventions (CityLearn+EVLearn):
        # - state == 1: "plugged in, ready to charge"
        # - departure_time == 0: departure happens at this time-step
        if not (np.isfinite(s_state) and np.isfinite(d) and np.isfinite(r)):
            continue
        if not _is_valid_ev_id(e):
            continue
        if not (s_state == 1.0 and d <= 0.0):
            continue

        name = _normalize_ev_id(e)
        if name is None:
            continue

        ev_info = ev_by_name.get(name)
        if ev_info is None:
            continue

        cap_kwh = ev_info.get("cap_kwh")
        soc_series = ev_info.get("soc_series")
        p_max_kw = _get_charger_pmax_kw(ch)

        # ------------------ actual SoC at departure ------------------
        if soc_series is not None and soc_series.ndim == 1 and 0 <= t_soc < soc_series.shape[0]:
            s_actual = float(np.clip(soc_series[t_soc], 0.0, 1.0))
        else:
            # fallback to arrival soc if needed
            if np.isfinite(as_now) and as_now >= 0.0:
                s_actual = float(np.clip(as_now, 0.0, 1.0))
            else:
                s_actual = 0.0

        deficit_actual = max(0.0, r - s_actual)

        # Default split: everything avoidable until we prove otherwise.
        unavoidable = 0.0
        avoidable = deficit_actual

        # ------------------ find latest arrival for this EV ------------------
        arrival_found = False
        arrival_idx: Optional[int] = None
        s_arrival: Optional[float] = None

        # Scan backwards for last "incoming" event (state == 2) for same EV.
        for tau in range(t_state, -1, -1):
            if tau >= state.shape[0]:
                continue

            st_tau = float(state[tau])
            e_tau = ev_id[tau]
            name_tau = _normalize_ev_id(e_tau)
            if name_tau != name:
                continue

            at_tau = float(arr_time[tau])
            as_tau = float(arr_soc[tau])

            if (
                st_tau == 2.0
                and np.isfinite(at_tau)
                and at_tau >= 0.0
                and np.isfinite(as_tau)
                and as_tau >= 0.0
            ):
                arrival_idx = tau
                s_arrival = float(np.clip(as_tau, 0.0, 1.0))
                arrival_found = True
                break

        schedule_ok = (
            dt_hours is not None
            and cap_kwh is not None
            and cap_kwh > 0.0
            and p_max_kw is not None
            and p_max_kw > 0.0
            and arrival_found
        )

        if schedule_ok:
            arr_t = int(arrival_idx)  # type: ignore[arg-type]
            s_arr = float(s_arrival)  # type: ignore[arg-type]

            # Number of charging steps from arrival to current step (inclusive).
            if arr_t <= t_state:
                n_steps = max(1, t_state - arr_t + 1)
            else:
                n_steps = 0

            if n_steps > 0:
                max_delta_soc = (p_max_kw * dt_hours * n_steps) / cap_kwh  # type: ignore[arg-type]
            else:
                max_delta_soc = 0.0

            s_max_possible = min(1.0, s_arr + max(0.0, max_delta_soc))

            # Split deficit into avoidable + unavoidable (physically impossible).
            unavoidable = max(0.0, r - s_max_possible)
            unavoidable = min(unavoidable, deficit_actual)
            avoidable = max(0.0, deficit_actual - unavoidable)

        record = EVDepartureRecord(
            ev_id=name,
            charger_id=str(getattr(ch, "charger_id", getattr(ch, "name", "UNKNOWN_CHARGER"))),
            time_step=t_state,
            required_soc=float(np.clip(r, 0.0, 1.0)),
            actual_soc=s_actual,
            arrival_soc=float(np.clip(as_now if np.isfinite(as_now) and as_now >= 0.0 else 0.0, 0.0, 1.0)),
            capacity_kwh=float(cap_kwh) if cap_kwh is not None else float("nan"),
            deficit_actual=float(deficit_actual),
            deficit_avoidable=float(avoidable),
            deficit_unavoidable=float(unavoidable),
        )
        records.append(record)

    return records


def ev_departure_cost_components(citylearn_env: Any) -> Dict[str, float]:
    """Aggregate EV departure SoC deficits at the current CityLearn time-step.

    Returns:
        total        = sum of actual SoC deficits at this step (all causes)
        avoidable    = part of total that policy could, in principle, fix
        unavoidable  = part that is physically impossible given schedule + hardware
        departures   = number of EVs departing at this step

    All deficits are normalized (0..1) SoC fractions.
    """
    records = _ev_departure_records(citylearn_env)

    total_deficit = float(sum(r.deficit_actual for r in records))
    total_avoidable = float(sum(r.deficit_avoidable for r in records))
    total_unavoidable = float(sum(r.deficit_unavoidable for r in records))
    n_dep = int(len(records))

    return {
        "total": total_deficit,
        "avoidable": total_avoidable,
        "unavoidable": total_unavoidable,
        "departures": n_dep,
    }


def ev_departure_safe_cost(
    citylearn_env: Any,
    *,
    quadratic: bool = True,
    scale_by_capacity_kwh: bool = True,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Scalar safety cost for Safe RL based on *avoidable* EV departure deficits.

    For each EV departure j at the current time-step, define:

        d_j = deficit_avoidable_j              in [0, 1]
        E_j = d_j * capacity_kwh_j            in [0, cap_kwh]

    If scale_by_capacity_kwh is True (recommended), we use E_j (kWh deficit).
    Otherwise, we use d_j (normalized SoC deficit).

    The per-step cost is:

        cost_t = sum_j [ alpha * x_j + beta * x_j^2 ],

    where x_j is d_j or E_j depending on the scaling option, and the sum is over
    all departing EVs at this time-step.

    This matches common practice in safe RL and EV charging literature where
    (avoidable) energy not delivered by departure is treated as a safety cost.
    """
    records = _ev_departure_records(citylearn_env)
    if not records:
        return 0.0

    xs: List[float] = []
    for r in records:
        d = max(0.0, float(r.deficit_avoidable))
        if d <= 0.0:
            continue
        if scale_by_capacity_kwh and np.isfinite(r.capacity_kwh) and r.capacity_kwh > 0.0:
            x = d * float(r.capacity_kwh)  # kWh
        else:
            x = d  # pure SoC fraction
        xs.append(x)

    if not xs:
        return 0.0

    xs_arr = np.asarray(xs, dtype=float)
    if quadratic:
        cost = alpha * float(xs_arr.sum()) + beta * float((xs_arr ** 2).sum())
    else:
        cost = alpha * float(xs_arr.sum())

    return float(cost)


# ---------------------------------------------------------------------------
# 4. Central agent wrapper over CityLearnEnv (central_agent=False)
# ---------------------------------------------------------------------------

class CentralAgentEnv(gym.Env):
    """Single-agent Gym-compatible view over a multi-building CityLearnEnv.

    - Works with env.central_agent == False.
    - Observations: all building observations concatenated into a 1D Box.
    - Actions: flat 1D Box mapped back to per-building actions.
    - Reward: sum of per-building rewards.

    It also exposes:
        - action_index_map: [(flat_idx, building_idx, local_idx, action_name), ...]
        - ev_action_indices: list of flat indices corresponding to EV charger actions
                             (names containing 'electric_vehicle_storage').
    """

    metadata = {"render_modes": []}

    def __init__(self, citylearn_env: Any):
        super().__init__()
        self.env = citylearn_env
        self.raw_env = unwrap_to_raw_citylearn_env(citylearn_env)

        # Validate central_agent=False for underlying env.
        ca = getattr(self.raw_env, "central_agent", False)
        if ca:
            raise ValueError(
                "CentralAgentEnv is meant for env.central_agent == False. "
                "If you already have a central agent env, use CityLearn's "
                "StableBaselines3Wrapper instead."
            )

        # Observation space: concatenate building observation spaces into one Box.
        obs_spaces: List[spaces.Box] = getattr(self.raw_env, "observation_space")
        if not isinstance(obs_spaces, list):
            raise TypeError("Expected env.observation_space to be a list[Box].")

        obs_lows = []
        obs_highs = []
        obs_dtypes = []

        for sp in obs_spaces:
            if not isinstance(sp, spaces.Box):
                raise TypeError("Each observation space must be gymnasium.spaces.Box.")
            obs_lows.append(sp.low.reshape(-1))
            obs_highs.append(sp.high.reshape(-1))
            obs_dtypes.append(sp.dtype)

        obs_low = np.concatenate(obs_lows, axis=0)
        obs_high = np.concatenate(obs_highs, axis=0)
        # Use first dtype (they're usually all float32).
        obs_dtype = obs_dtypes[0] if obs_dtypes else np.float32
        self.observation_space = spaces.Box(
            low=obs_low.astype(obs_dtype),
            high=obs_high.astype(obs_dtype),
            dtype=obs_dtype,
        )

        # Action space: concatenate building action spaces into one Box.
        act_spaces: List[spaces.Box] = getattr(self.raw_env, "action_space")
        if not isinstance(act_spaces, list):
            raise TypeError("Expected env.action_space to be a list[Box].")

        act_lows = []
        act_highs = []
        act_sizes = []
        for sp in act_spaces:
            if not isinstance(sp, spaces.Box):
                raise TypeError("Each action space must be gymnasium.spaces.Box.")
            flat_low = sp.low.reshape(-1)
            flat_high = sp.high.reshape(-1)
            act_lows.append(flat_low)
            act_highs.append(flat_high)
            act_sizes.append(flat_low.shape[0])

        self._act_sizes = act_sizes
        self._act_offsets = np.cumsum([0] + act_sizes)
        act_low = np.concatenate(act_lows, axis=0)
        act_high = np.concatenate(act_highs, axis=0)
        self.action_space = spaces.Box(
            low=act_low.astype(np.float32),
            high=act_high.astype(np.float32),
            dtype=np.float32,
        )

        # Map flat action index -> (building index, local index, action name).
        self.action_index_map: List[Tuple[int, int, int, str]] = []
        ev_indices: List[int] = []

        action_names: List[List[str]] = getattr(self.raw_env, "action_names")
        if len(action_names) != len(act_spaces):
            raise ValueError(
                "len(env.action_names) != len(env.action_space); "
                "cannot build central index map."
            )

        for b_idx, names_b in enumerate(action_names):
            for a_idx, name in enumerate(names_b):
                flat_idx = int(self._act_offsets[b_idx] + a_idx)
                self.action_index_map.append((flat_idx, b_idx, a_idx, str(name)))
                if "electric_vehicle_storage" in str(name):
                    ev_indices.append(flat_idx)

        self.ev_action_indices: List[int] = sorted(ev_indices)

    # ------------------------ utility methods ------------------------

    def _split_flat_action(self, action: np.ndarray) -> List[np.ndarray]:
        """Convert flat action vector -> list[ per-building action vectors ]."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_space.shape[0]:
            raise ValueError(
                f"Expected action of shape {(self.action_space.shape[0],)}, "
                f"got {tuple(action.shape)}."
            )

        actions_per_building: List[np.ndarray] = []
        for i in range(len(self._act_sizes)):
            start = int(self._act_offsets[i])
            end = int(self._act_offsets[i + 1])
            actions_per_building.append(action[start:end])
        return actions_per_building

    def _flatten_obs(self, obs: Any) -> np.ndarray:
        """Flatten env.observations (list[list[float]]) into 1D array."""
        # CityLearn observations property is always a list of per-building lists.
        if isinstance(obs, (list, tuple)) and obs and isinstance(obs[0], (list, tuple, np.ndarray)):
            parts = [np.asarray(o, dtype=self.observation_space.dtype).reshape(-1) for o in obs]
            flat = np.concatenate(parts, axis=0)
        else:
            # Fallback: assume obs is already flat-ish.
            flat = np.asarray(obs, dtype=self.observation_space.dtype).reshape(-1)

        if flat.shape[0] != self.observation_space.shape[0]:
            # Defensive: pad or truncate if something is off.
            target = self.observation_space.shape[0]
            if flat.shape[0] > target:
                flat = flat[:target]
            else:
                pad = np.zeros(target - flat.shape[0], dtype=self.observation_space.dtype)
                flat = np.concatenate([flat, pad], axis=0)

        return flat

    # ------------------------ Gym API ------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if hasattr(self.env, "reset"):
            try:
                obs, info = self.env.reset(seed=seed, options=options)
            except TypeError:
                # Older Gym API without seed/options
                obs = self.env.reset()
                info = {}
        else:
            raise AttributeError("Underlying env has no reset method.")

        flat_obs = self._flatten_obs(obs)
        return flat_obs, info

    def step(self, action: np.ndarray):
        actions_per_building = self._split_flat_action(action)

        out = self.env.step(actions_per_building)

        if isinstance(out, tuple) and len(out) == 5:
            obs, rewards, terminated, truncated, info = out
        elif isinstance(out, tuple) and len(out) == 4:
            # Gym <=0.21 style: obs, reward, done, info
            obs, rewards, done, info = out
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError(
                "Unexpected env.step return format. "
                "Expected (obs, rewards, terminated, truncated, info) "
                "or (obs, rewards, done, info)."
            )

        flat_obs = self._flatten_obs(obs)

        # CityLearn multi-agent reward is List[float]; aggregate to scalar.
        if isinstance(rewards, (list, tuple, np.ndarray)):
            reward = float(np.asarray(rewards, dtype=float).sum())
        else:
            reward = float(rewards)

        return flat_obs, reward, terminated, truncated, info

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()
        return None

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


# ---------------------------------------------------------------------------
# 5. (Optional) quick sanity-check helper
# ---------------------------------------------------------------------------

def run_ev_departure_kpi_sanity_check(schema_path: str, mode: str = "no_control") -> Dict[str, float]:
    """Replicate your no-control vs full-charge sanity check in a single function.

    Args:
        schema_path: Path to CityLearn schema.json.
        mode: "no_control" or "ev_full_charge".

    Returns:
        dict with:
            total_reward
            total_ev_departure_cost (linear; from ev_departure_cost_components)
            total_ev_departure_safe_cost (advanced; from ev_departure_safe_cost)
    """
    from citylearn.citylearn import CityLearnEnv  # local import

    env = CityLearnEnv(schema=schema_path)  # central_agent as defined in schema
    assert not getattr(env, "central_agent", False), "Expected central_agent=False in schema."

    try:
        obs, _ = env.reset()
    except Exception:
        obs = env.reset()
    done = False
    total_reward = 0.0
    total_ev_cost = 0.0
    total_ev_safe_cost = 0.0
    steps = 0

    # Pre-compute EV action indices per building for mode="ev_full_charge".
    action_names: List[List[str]] = env.action_names
    ev_indices_per_building: List[List[int]] = []
    for names_b in action_names:
        ev_idx = [i for i, n in enumerate(names_b) if "electric_vehicle_storage" in str(n)]
        ev_indices_per_building.append(ev_idx)

    while True:
        # Build actions
        actions: List[np.ndarray] = []
        if mode == "no_control":
            for sp in env.action_space:
                size = int(np.prod(sp.shape)) if sp.shape else 1
                actions.append(np.zeros(size, dtype=float))
        elif mode == "ev_full_charge":
            for b_idx, sp in enumerate(env.action_space):
                size = int(np.prod(sp.shape)) if sp.shape else 1
                a = np.zeros(size, dtype=float)
                for j in ev_indices_per_building[b_idx]:
                    if 0 <= j < size:
                        a[j] = 1.0  # full charge when connected
                actions.append(a)
        else:
            raise ValueError("mode must be 'no_control' or 'ev_full_charge'.")

        # Step environment
        out = env.step(actions)
        if len(out) == 5:
            obs, rewards, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
        else:
            obs, rewards, done, info = out

        # Aggregate reward
        if isinstance(rewards, (list, tuple, np.ndarray)):
            r = float(np.asarray(rewards, dtype=float).sum())
        else:
            r = float(rewards)
        total_reward += r

        # EV KPI at THIS time-step (post-step, time_step already advanced)
        comps = ev_departure_cost_components(env)
        total_ev_cost += comps["total"]

        safe_c = ev_departure_safe_cost(env)
        total_ev_safe_cost += safe_c

        steps += 1
        if done:
            break

    return {
        "steps": float(steps),
        "total_reward": float(total_reward),
        "total_ev_departure_cost": float(total_ev_cost),
        "total_ev_departure_safe_cost": float(total_ev_safe_cost),
    }
