from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

def _norm(x: Any) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode("utf-8", errors="ignore")
    return str(x).strip()

def unwrap_to_citylearn(env: Any) -> Any:
    cur = env
    seen = set()
    for _ in range(60):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if getattr(cur, "buildings", None) is not None and getattr(cur, "time_step", None) is not None:
            return cur
        for a in ("base", "env", "unwrapped", "_env", "raw_env"):
            nxt = getattr(cur, a, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return cur

def action_name_list(env: Any) -> List[str]:
    names = getattr(env, "action_names", None)
    if names is None:
        raw = unwrap_to_citylearn(env)
        names = getattr(raw, "action_names", None)
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    return [str(n) for n in (names or [])]

def build_ev_action_map(names: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    key = "electric_vehicle_storage_charger_"
    for i, n in enumerate([x.lower() for x in names]):
        if key in n:
            suffix = n.split(key, 1)[1]
            out[f"charger_{suffix}"] = i
    return out

def flatten_action(action, action_space) -> np.ndarray:
    if isinstance(action, list):
        return np.concatenate([np.asarray(a, dtype=float).ravel() for a in action])
    return np.asarray(action, dtype=float).ravel()

def unflatten_action(action_flat: np.ndarray, action_space):
    if isinstance(action_space, list):
        out = []
        idx = 0
        for sp in action_space:
            n = int(np.prod(sp.shape))
            out.append(action_flat[idx:idx+n].reshape(sp.shape))
            idx += n
        return out
    return action_flat

def get_dt_hours(env: Any) -> float:
    raw = unwrap_to_citylearn(env)
    s = getattr(raw, "seconds_per_time_step", None)
    if s is None:
        return 1.0
    try:
        return float(s) / 3600.0
    except Exception:
        return 1.0

@dataclass
class EVState:
    charger_id: str
    building_idx: int
    action_idx: int
    E_def_kwh: float
    tau_steps: int
    Pmax_kw: float

def get_ev_states(env: Any, ev_map: Dict[str, int]) -> List[EVState]:
    raw = unwrap_to_citylearn(env)
    t_state = int(getattr(raw, "time_step", 0))
    buildings = getattr(raw, "buildings", []) or []
    evs: List[EVState] = []

    for bi, b in enumerate(buildings):
        chargers = getattr(b, "electric_vehicle_chargers", []) or []
        for ch in chargers:
            cid = _norm(getattr(ch, "charger_id", getattr(ch, "name", "")))
            if cid not in ev_map:
                continue

            sim = getattr(ch, "charger_simulation", getattr(ch, "_Charger__charger_simulation", None))
            if sim is None:
                continue

            state = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
            dep_time = np.asarray(getattr(sim, "_electric_vehicle_departure_time"), dtype=float)
            req_soc = np.asarray(getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
            ev_id = np.asarray(getattr(sim, "_electric_vehicle_id"))

            if t_state >= len(state):
                continue
            if float(state[t_state]) != 1.0:
                continue
            if ev_id[t_state] is None:
                continue

            d = float(dep_time[t_state])
            r = float(req_soc[t_state])
            if not np.isfinite(d) or not np.isfinite(r):
                continue

            tau = int(max(1, round(d)))  # conservative, at least 1 step

            ev = getattr(ch, "connected_electric_vehicle", None)
            if ev is None:
                continue
            batt = getattr(ev, "battery", None)
            if batt is None:
                continue

            cap = float(getattr(batt, "capacity", 0.0) or 0.0)
            soc = getattr(batt, "soc", None)
            t_soc = max(0, t_state - 1)
            soc_now = float(soc[t_soc]) if (hasattr(soc, "__len__") and t_soc < len(soc)) else float(soc or 0.0)

            E_def = max(0.0, (r - soc_now) * cap)

            pmax = getattr(ch, "max_charging_power", getattr(ch, "_Charger__max_charging_power", 0.0))
            pmax = float(np.asarray(pmax).reshape(-1)[0]) if pmax is not None else 0.0

            evs.append(EVState(
                charger_id=cid,
                building_idx=bi,
                action_idx=int(ev_map[cid]),
                E_def_kwh=float(E_def),
                tau_steps=int(tau),
                Pmax_kw=float(pmax),
            ))

    return evs
