
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. Unwrapping helpers + generic SoC tools
# ---------------------------------------------------------------------------

def unwrap_to_raw_citylearn_env(env: Any) -> Any:
    """Follow .base / .env / .unwrapped links until we hit the raw CityLearnEnv-like object."""
    seen = set()
    e = env
    while True:
        if id(e) in seen:
            break
        seen.add(id(e))

        if hasattr(e, "base"):
            e = e.base
            continue
        if hasattr(e, "env"):
            e = e.env
            continue
        if hasattr(e, "unwrapped"):
            try:
                e = e.unwrapped
                continue
            except Exception:
                pass
        break
    return e


def battery_soc_list(base_env: Any) -> List[float]:
    """Return SoC (0..1) for every building-level electricity storage, if any."""
    env = unwrap_to_raw_citylearn_env(base_env)
    socs: List[float] = []
    for b in getattr(env, "buildings", []):
        es = getattr(b, "electrical_storage", None)
        if es is None:
            es = getattr(b, "electricity_storage", None)
        if es is not None and hasattr(es, "soc"):
            try:
                socs.append(float(es.soc))
            except Exception:
                pass
    return socs


def battery_soc_stats(base_env: Any) -> Dict[str, float]:
    socs = battery_soc_list(base_env)
    if not socs:
        return {
            "soc_mean": 0.5,
            "soc_min_obs": 0.5,
            "soc_max_obs": 0.5,
            "num_storages": 0.0,
        }
    return {
        "soc_mean": float(sum(socs) / len(socs)),
        "soc_min_obs": float(min(socs)),
        "soc_max_obs": float(max(socs)),
        "num_storages": float(len(socs)),
    }


def current_time_index(base_env: Any) -> int:
    """
    Return the index in battery.soc that represents the *current* SoC.

    CityLearn stores SoC as a time series. After each step, the new value
    is written at index (time_step - 1). At reset (time_step = 0), the
    current value is at index 0.
    """
    env = unwrap_to_raw_citylearn_env(base_env)
    t = int(getattr(env, "time_step", 0))
    return 0 if t <= 0 else t - 1


def ev_soc_by_charger(base_env: Any) -> Dict[Tuple[str, str], float]:
    """
    Return a dict mapping (building_name, charger_name) -> current EV SoC (0..1),
    for all chargers that currently have a connected EV.
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
                arr = getattr(batt, "soc")
                soc_t = float(arr[t])
                out[(bname, cname)] = soc_t
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
                arr = getattr(batt, "soc")
                soc_t = float(arr[t])
                out[str(name)] = soc_t
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
# 2. EV departure KPI internals
# ---------------------------------------------------------------------------

@dataclass
class EVDepartureRecord:
    """Per-departure data at the current time step."""
    ev_id: str
    charger_id: str
    time_step: int
    required_soc: float
    actual_soc: float
    arrival_soc: float
    capacity_kwh: float
    deficit_actual: float
    deficit_avoidable: float
    deficit_unavoidable: float


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

        ev_by_name[name] = {"cap_kwh": cap_kwh, "soc_series": soc_series}

    return ev_by_name


def _get_ev_soc_at_index(
    ev_by_name: Dict[str, Dict[str, Any]],
    ev_name: str,
    t_soc_idx: int
) -> Optional[float]:
    """Return EV battery.soc[t_soc_idx] if available and finite."""
    info = ev_by_name.get(ev_name)
    if info is None:
        return None
    series = info.get("soc_series")
    if series is None:
        return None
    series = np.asarray(series, dtype=float)
    if series.ndim != 1 or not (0 <= t_soc_idx < series.shape[0]):
        return None
    v = float(series[t_soc_idx])
    if not np.isfinite(v):
        return None
    return float(np.clip(v, 0.0, 1.0))


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

        state = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
        dep_time = np.asarray(getattr(sim, "_electric_vehicle_departure_time"), dtype=float)
        req_soc = np.asarray(getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
        ev_id = np.asarray(getattr(sim, "_electric_vehicle_id"))
        arr_soc = np.asarray(getattr(sim, "_electric_vehicle_estimated_soc_arrival"), dtype=float)

        if state.ndim != 1 or t_state >= state.shape[0]:
            continue

        s_state = float(state[t_state])
        d = float(dep_time[t_state])
        r = float(req_soc[t_state])
        e = ev_id[t_state]
        as_now = float(arr_soc[t_state]) if t_state < arr_soc.shape[0] else -1.0

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

        # Actual SOC at departure (best source: battery series)
        if soc_series is not None:
            soc_series_arr = np.asarray(soc_series, dtype=float)
            if soc_series_arr.ndim == 1 and 0 <= t_soc < soc_series_arr.shape[0]:
                s_actual = float(np.clip(soc_series_arr[t_soc], 0.0, 1.0))
            else:
                s_actual = 0.0
        else:
            # fallback only if schedule gives a valid number
            if np.isfinite(as_now) and as_now >= 0.0:
                s_actual = float(np.clip(as_now, 0.0, 1.0))
            else:
                s_actual = 0.0

        deficit_actual = max(0.0, r - s_actual)

        unavoidable = 0.0
        avoidable = deficit_actual

        # ---------------- Agent-controllable split ----------------
        schedule_ok = (
            dt_hours is not None
            and cap_kwh is not None
            and cap_kwh > 0.0
            and p_max_kw is not None
            and p_max_kw > 0.0
        )

        if schedule_ok and deficit_actual > 0.0:
            end = t_state

            # Find start of last continuous block of (state==1 AND ev_id==this EV)
            start = end
            for tau in range(end, -1, -1):
                if float(state[tau]) != 1.0:
                    break
                name_tau = _normalize_ev_id(ev_id[tau])
                if name_tau != name:
                    break
                start = tau

            # Count connected steps in [start, end] (strict: state==1 + valid ev_id + match)
            connected_steps = 0
            for tau in range(start, end + 1):
                if float(state[tau]) != 1.0:
                    continue
                e_tau = ev_id[tau]
                if not _is_valid_ev_id(e_tau):
                    continue
                name_tau = _normalize_ev_id(e_tau)
                if name_tau != name:
                    continue
                connected_steps += 1

            # Agent reaction delay: first plugged step is not controllable (action arrives one step later)
            effective_steps = max(0, connected_steps - 1)

            # Start SOC index for battery series = start-1 (time_step -> t_idx convention)
            start_soc_idx = 0 if start <= 0 else start - 1
            s_start = _get_ev_soc_at_index(ev_by_name, name, start_soc_idx)

            # If missing SOC series, be conservative: don't blame policy
            if s_start is None or effective_steps <= 0:
                unavoidable = deficit_actual
                avoidable = 0.0
            else:
                max_delta_soc = (p_max_kw * dt_hours * effective_steps) / cap_kwh  # type: ignore[operator]
                s_max_possible = min(1.0, float(s_start) + max(0.0, float(max_delta_soc)))

                unavoidable = max(0.0, r - s_max_possible)
                unavoidable = min(unavoidable, deficit_actual)
                avoidable = max(0.0, deficit_actual - unavoidable)

        record = EVDepartureRecord(
            ev_id=name,
            charger_id=str(getattr(ch, "charger_id", getattr(ch, "name", "UNKNOWN_CHARGER"))),
            time_step=t_state,
            required_soc=float(np.clip(r, 0.0, 1.0)),
            actual_soc=s_actual,
            # arrival_soc only for logging; CityLearn sometimes uses -0.1 sentinel
            arrival_soc=float(np.clip(as_now if np.isfinite(as_now) and as_now >= 0.0 else 0.0, 0.0, 1.0)),
            capacity_kwh=float(cap_kwh) if cap_kwh is not None else float("nan"),
            deficit_actual=float(deficit_actual),
            deficit_avoidable=float(avoidable),
            deficit_unavoidable=float(unavoidable),
        )
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# 3. Public EV KPI interfaces
# ---------------------------------------------------------------------------

def ev_departure_cost_components(citylearn_env: Any) -> Dict[str, float]:
    """Aggregate EV departure SoC deficits at the current CityLearn time-step."""
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
    """Scalar safety cost for Safe RL based on *avoidable* EV departure deficits."""
    records = _ev_departure_records(citylearn_env)
    if not records:
        return 0.0

    xs: List[float] = []
    for r in records:
        d = max(0.0, float(r.deficit_avoidable))
        if d <= 0.0:
            continue
        if scale_by_capacity_kwh and np.isfinite(r.capacity_kwh) and r.capacity_kwh > 0.0:
            x = d * float(r.capacity_kwh)
        else:
            x = d
        xs.append(x)

    if not xs:
        return 0.0

    xs_arr = np.asarray(xs, dtype=float)
    if quadratic:
        cost = alpha * float(xs_arr.sum()) + beta * float((xs_arr ** 2).sum())
    else:
        cost = alpha * float(xs_arr.sum())

    return float(cost)
