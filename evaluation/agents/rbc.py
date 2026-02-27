from typing import Any, Dict
import numpy as np
from .base import BaseAgent


def _unwrap_action_names(names):
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        return names[0]
    return names


def _norm_id(x) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode("utf-8", errors="ignore")
    s = str(x).strip()
    if s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]
    return s.strip()


def _is_valid_ev_id(x) -> bool:
    s = _norm_id(x)
    return s != "" and s.lower() not in ("nan", "none")


def _unwrap_to_raw_citylearn_env(env):
    """Unwrap to find raw CityLearn env with buildings"""
    cur = env
    seen = set()
    for _ in range(40):
        if cur is None:
            break
        obj_id = id(cur)
        if obj_id in seen:
            break
        seen.add(obj_id)
        
        try:
            blds = getattr(cur, "buildings", None)
            ts = getattr(cur, "time_step", None)
            if blds is not None and hasattr(blds, "__len__") and len(blds) > 0 and ts is not None:
                return cur
        except Exception:
            pass
        
        for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
            if hasattr(cur, attr):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
    return cur


def _current_time_index(raw_env) -> int:
    """Get current time index from raw env"""
    return int(getattr(raw_env, "time_step", 0))


class IntelligentRBC:
    """RBC with configurable EV mode - connection-aware"""

    def __init__(self, env, ev_mode: str = "greedy"):
        self.env = env
        self.ev_mode = ev_mode

        raw = _unwrap_to_raw_citylearn_env(env)
        names = _unwrap_action_names(getattr(raw, "action_names", []))
        if not names:
            base = getattr(env, "base", None)
            if base is not None:
                names = _unwrap_action_names(getattr(base, "action_names", []))
        if not names:
            raise AttributeError("Could not find action_names on env/base/raw env")

        self.action_names = names
        self.action_dim = int(env.action_space.shape[0]) if hasattr(env.action_space, "shape") else len(names)

        self.battery_indices = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
        self.ev_indices = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger" in str(n).lower()]
        self.cooling_indices = [i for i, n in enumerate(names) if str(n).strip().lower() == "cooling_storage"]
        self.heating_indices = [i for i, n in enumerate(names) if str(n).strip().lower() == "heating_storage"]
        
        print(f"[IntelligentRBC] EV mode: {ev_mode}")
        print(f"[IntelligentRBC] Battery indices: {self.battery_indices}")
        print(f"[IntelligentRBC] EV indices: {self.ev_indices}")

    def _battery_action(self, hour: int) -> float:
        return 0.8 if 10 <= hour <= 16 else (-0.6 if 17 <= hour <= 21 else 0.0)

    def _ev_action_time_based(self, hour: int) -> float:
        return 1.0 if (hour >= 22 or hour < 6) else 0.0

    def _charger_connected_now(self, raw_env, action_name: str) -> bool:
        s = str(action_name).strip().lower()
        if "electric_vehicle_storage_charger_" not in s:
            return False
        suffix = s.split("electric_vehicle_storage_charger_", 1)[1]
        charger_id = f"charger_{suffix}"

        # IMPORTANT: aligns with wrapper storing action at tau = time_step + 1
        t_state = int(getattr(raw_env, "time_step", 0)) + 1
        if t_state < 0:
            t_state = 0

        for b in getattr(raw_env, "buildings", []) or []:
            for ch in getattr(b, "electric_vehicle_chargers", []) or []:
                cid = getattr(ch, "charger_id", getattr(ch, "name", None))
                if _norm_id(cid) != charger_id:
                    continue
                sim = getattr(ch, "charger_simulation", getattr(ch, "_Charger__charger_simulation", None))
                if sim is None:
                    return False
                try:
                    state = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                    ev_id = np.asarray(getattr(sim, "_electric_vehicle_id"))
                except Exception:
                    return False
                if state.ndim != 1 or t_state >= len(state):
                    return False
                st = float(state[t_state])
                eid = ev_id[t_state]
                return (st == 1.0) and _is_valid_ev_id(eid)
        return False

    def predict(self, obs) -> np.ndarray:
        a = np.zeros(self.action_dim, dtype=np.float32)
        raw = _unwrap_to_raw_citylearn_env(self.env)
        hour = int(_current_time_index(raw) % 24)

        # EV - connection-aware
        if self.ev_mode == "greedy":
            for i in self.ev_indices:
                a[i] = 1.0 if self._charger_connected_now(raw, self.action_names[i]) else 0.0
        elif self.ev_mode == "time_based":
            ev_action = self._ev_action_time_based(hour)
            for i in self.ev_indices:
                a[i] = ev_action

        # Batteries
        batt = float(self._battery_action(hour))
        for i in self.battery_indices:
            a[i] = batt

        # HVAC storages (harmless if empty)
        for i in self.cooling_indices:
            a[i] = 0.0
        for i in self.heating_indices:
            a[i] = 0.0

        return a


class RBCAgent(BaseAgent):
    """Wrapper for IntelligentRBC to match BaseAgent interface"""
    name = "rbc"
    
    def __init__(self, ev_mode: str = "greedy"):
        super().__init__()
        self.ev_mode = ev_mode
        self._rbc = None
    
    def reset(self, env: Any) -> None:
        super().reset(env)
        # Initialize IntelligentRBC with the env
        self._rbc = IntelligentRBC(env, ev_mode=self.ev_mode)
    
    def act(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if self._rbc is None:
            raise RuntimeError("RBCAgent not initialized. Call reset() first.")
        return self._rbc.predict(obs)
