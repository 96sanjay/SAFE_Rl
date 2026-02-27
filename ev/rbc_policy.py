import numpy as np
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env, current_time_index

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
    if s == "" or s.lower() in ("nan", "none"):
        return False
    return True

class IntelligentRBC:
    """RBC with configurable EV mode"""

    def __init__(self, env, ev_mode: str = "greedy"):
        self.env = env
        self.ev_mode = ev_mode

        raw = unwrap_to_raw_citylearn_env(env)
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

    def _battery_action(self, hour: int) -> float:
        return 0.8 if 10 <= hour <= 16 else (-0.6 if 17 <= hour <= 21 else 0.0)

    def _ev_action_time_based(self, hour: int) -> float:
        return 1.0 if (hour >= 22 or hour < 6) else 0.0

    def _get_indoor_temps(self, raw_env) -> np.ndarray:
        temps = []
        for b in getattr(raw_env, "buildings", []) or []:
            t_idx = int(getattr(raw_env, "time_step", 0))
            if t_idx < 0:
                t_idx = 0
            try:
                temp_arr = np.asarray(b.indoor_dry_bulb_temperature, dtype=float)
                if temp_arr.ndim == 1 and 0 <= t_idx < len(temp_arr):
                    temps.append(float(temp_arr[t_idx]))
                else:
                    temps.append(22.0)
            except Exception:
                temps.append(22.0)
        return np.array(temps)

    def _charger_connected_now(self, raw_env, action_name: str) -> bool:
        s = str(action_name).strip().lower()
        if "electric_vehicle_storage_charger_" not in s:
            return False
        suffix = s.split("electric_vehicle_storage_charger_", 1)[1]
        charger_id = f"charger_{suffix}"

        # IMPORTANT: aligns with your wrapper storing action at tau = time_step + 1
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
        raw = unwrap_to_raw_citylearn_env(self.env)
        hour = int(current_time_index(raw) % 24)

        # EV
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

        # HVAC storages (usually none in your dataset; harmless if empty)
        temps = self._get_indoor_temps(raw)
        for i in self.cooling_indices:
            a[i] = 0.5 if temps[i % len(temps)] > 24.0 else 0.0
        for i in self.heating_indices:
            a[i] = 0.5 if temps[i % len(temps)] < 20.0 else 0.0

        return a
