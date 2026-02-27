from typing import Any, Dict, Optional, Tuple, List
import numpy as np


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
    """Unwrap to find raw CityLearn env with buildings + time_step."""
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

        # unwrap common wrapper attributes
        moved = False
        for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
            if hasattr(cur, attr):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    moved = True
                    break
        if not moved:
            break

    return cur


def _current_time_index(raw_env) -> int:
    return int(getattr(raw_env, "time_step", 0))


def _as_scalar_at(x, t_idx: int) -> Optional[float]:
    """
    Convert x to a float scalar for the given time index.
    Handles scalars, lists, numpy arrays.
    """
    if x is None:
        return None
    try:
        # scalar
        if np.isscalar(x):
            v = float(x)
            return v if np.isfinite(v) else None
        # array/list-like
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


def _get_tin_tset(raw_env, building_idx: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (Tin, Tset_cool, Tset_heat) for a specific building.
    
    Priority:
    1. energy_simulation (ground truth after LSTM prediction)
    2. observation dict
    3. time series attributes
    """
    if raw_env is None or not getattr(raw_env, "buildings", None):
        return None, None, None

    buildings = getattr(raw_env, "buildings", [])
    if building_idx >= len(buildings):
        return None, None, None

    b = buildings[building_idx]
    # CityLearn time_step is 1-indexed during step, so actual index is time_step-1
    t_idx = max(0, int(getattr(raw_env, "time_step", 0)) - 1)

    # 1) Try energy_simulation (most reliable for LSTM buildings)
    try:
        es = getattr(b, "energy_simulation", None)
        if es is not None:
            tin = _as_scalar_at(getattr(es, "indoor_dry_bulb_temperature", None), t_idx)
            tset_cool = _as_scalar_at(getattr(es, "indoor_dry_bulb_temperature_cooling_set_point", None), t_idx)
            tset_heat = _as_scalar_at(getattr(es, "indoor_dry_bulb_temperature_heating_set_point", None), t_idx)
            if tin is not None:
                return tin, tset_cool, tset_heat
    except Exception:
        pass

    # 2) Try observation dict
    try:
        d = b._get_observations_data()
        tin = _as_scalar_at(d.get("indoor_dry_bulb_temperature", None), t_idx)
        tset_cool = _as_scalar_at(d.get("indoor_dry_bulb_temperature_cooling_set_point", None), t_idx)
        tset_heat = _as_scalar_at(d.get("indoor_dry_bulb_temperature_heating_set_point", None), t_idx)
        if tin is not None:
            return tin, tset_cool, tset_heat
    except Exception:
        pass

    # 3) Try time series attributes on building
    try:
        tin = _as_scalar_at(getattr(b, "indoor_dry_bulb_temperature", None), t_idx)
        tset_cool = _as_scalar_at(getattr(b, "indoor_dry_bulb_temperature_cooling_set_point", None), t_idx)
        tset_heat = _as_scalar_at(getattr(b, "indoor_dry_bulb_temperature_heating_set_point", None), t_idx)
        if tin is not None:
            return tin, tset_cool, tset_heat
    except Exception:
        pass

    return None, None, None


class IntelligentRBCWithTemp:
    """
    RBC with EV + Multi-Building Temperature control.
    
    Supports:
    - Multiple buildings with independent temperature control
    - Both separate (cooling_device, heating_device) and combined (cooling_or_heating_device) HVAC
    - Connection-aware EV charging
    - Solar-tracking battery control
    """

    def __init__(self, env, ev_mode: str = "greedy", temp_deadband: float = 0.5):
        self.env = env
        self.ev_mode = ev_mode
        self.temp_deadband = float(temp_deadband)

        raw = _unwrap_to_raw_citylearn_env(env)
        names = _unwrap_action_names(getattr(raw, "action_names", []))
        if not names:
            base = getattr(env, "base", None)
            if base is not None:
                names = _unwrap_action_names(getattr(base, "action_names", []))
        if not names:
            raise AttributeError("Could not find action_names on env/base/raw env")

        self.action_names = list(names)
        self.num_buildings = len(getattr(raw, "buildings", []))

        # action_dim for list vs box
        if hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
            # Box
            self.action_dim = int(env.action_space.shape[0])
        else:
            # list action space OR unknown: fall back to action_names length
            self.action_dim = len(self.action_names)

        # Battery (typically 1 per building)
        self.battery_indices = [
            i for i, n in enumerate(self.action_names)
            if str(n).strip().lower() == "electrical_storage"
        ]

        # EV chargers
        self.ev_indices = [
            i for i, n in enumerate(self.action_names)
            if "electric_vehicle_storage_charger" in str(n).strip().lower()
        ]

        # HVAC storage (TES)
        self.cooling_storage_indices = [
            i for i, n in enumerate(self.action_names)
            if str(n).strip().lower() == "cooling_storage"
        ]
        self.heating_storage_indices = [
            i for i, n in enumerate(self.action_names)
            if str(n).strip().lower() == "heating_storage"
        ]

        # HVAC devices: Map action indices to building indices
        self._map_hvac_to_buildings()

        self.has_temp_control = bool(
            self.cooling_device_map or 
            self.heating_device_map or 
            self.combined_hvac_map
        )

        print(f"[IntelligentRBC] Number of buildings: {self.num_buildings}")
        print(f"[IntelligentRBC] EV mode: {self.ev_mode}")
        print(f"[IntelligentRBC] Temperature deadband: {self.temp_deadband}°C")
        print(f"[IntelligentRBC] Battery indices: {self.battery_indices}")
        print(f"[IntelligentRBC] EV indices: {self.ev_indices}")
        print(f"[IntelligentRBC] Cooling device map: {self.cooling_device_map}")
        print(f"[IntelligentRBC] Heating device map: {self.heating_device_map}")
        print(f"[IntelligentRBC] Combined HVAC map: {self.combined_hvac_map}")
        print(f"[IntelligentRBC] Has temperature control: {self.has_temp_control}")
        print(f"[IntelligentRBC] Temperature-controlled buildings: {len(self.cooling_device_map) + len(self.combined_hvac_map)}")

    def _map_hvac_to_buildings(self):
        """
        Map HVAC action indices to building indices.
        
        CityLearn action_names format: actions are ordered by building
        Example: [dhw_b1, battery_b1, cooling_b1, dhw_b2, battery_b2, cooling_b2, ...]
        
        We detect which buildings have cooling/heating/combined devices by 
        counting occurrences in action_names.
        """
        # Count occurrences of each action type
        cooling_count = 0
        heating_count = 0
        combined_count = 0
        
        self.cooling_device_map = {}  # {building_idx: action_idx}
        self.heating_device_map = {}  # {building_idx: action_idx}
        self.combined_hvac_map = {}   # {building_idx: action_idx}
        
        for i, name in enumerate(self.action_names):
            name_lower = str(name).strip().lower()
            
            if name_lower == "cooling_device":
                self.cooling_device_map[cooling_count] = i
                cooling_count += 1
            elif name_lower == "heating_device":
                self.heating_device_map[heating_count] = i
                heating_count += 1
            elif name_lower == "cooling_or_heating_device":
                self.combined_hvac_map[combined_count] = i
                combined_count += 1

    def _battery_action(self, hour: int) -> float:
        """
        Solar tracking: charge during day, discharge at peak evening hours.
        
        Strategy:
        - 10-16h: Charge from solar (0.8)
        - 17-21h: Discharge for peak shaving (-0.6)
        - Other: Idle (0.0)
        """
        if 10 <= hour <= 16:
            return 0.8  # Charge during solar hours
        elif 17 <= hour <= 21:
            return -0.6  # Discharge during peak
        else:
            return 0.0  # Idle

    def _ev_action_time_based(self, hour: int) -> float:
        """
        Charge overnight during off-peak hours.
        
        Strategy: Charge 22:00 - 06:00 (low electricity prices)
        """
        return 1.0 if (hour >= 22 or hour < 6) else 0.0

    def _charger_connected_now(self, raw_env, action_name: str) -> bool:
        """Check if EV is connected to charger right now."""
        s = str(action_name).strip().lower()
        if "electric_vehicle_storage_charger_" not in s:
            return False
        suffix = s.split("electric_vehicle_storage_charger_", 1)[1]
        charger_id = f"charger_{suffix}"

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

    def _temp_control_action(self, raw_env, building_idx: int) -> Tuple[float, float, float]:
        """
        Thermostat logic with deadband for a specific building.
        
        Returns: (cooling_action, heating_action, combined_action)
        
        CityLearn convention for cooling_or_heating_device:
          combined_action < 0  => cooling
          combined_action > 0  => heating
          combined_action = 0  => off
        
        Strategy:
        - If T_in < T_heat - deadband: HEAT (proportional to error)
        - If T_in > T_cool + deadband: COOL (proportional to error)
        - Otherwise: OFF (within comfort band)
        """
        tin, tset_cool, tset_heat = _get_tin_tset(raw_env, building_idx)

        if tin is None or not np.isfinite(tin):
            return 0.0, 0.0, 0.0

        # Default setpoints if missing
        if tset_cool is None or not np.isfinite(tset_cool):
            tset_cool = 24.0  # Default cooling setpoint
        if tset_heat is None or not np.isfinite(tset_heat):
            tset_heat = 20.0  # Default heating setpoint

        # Heating mode: T_in below heating setpoint
        if tin < (tset_heat - self.temp_deadband):
            error = (tset_heat - tin) - self.temp_deadband
            # Proportional control: stronger heating for larger errors
            heating_action = min(1.0, error / 3.0)
            # Combined: POSITIVE = heating
            return 0.0, float(heating_action), float(+heating_action)

        # Cooling mode: T_in above cooling setpoint
        if tin > (tset_cool + self.temp_deadband):
            error = (tin - tset_cool) - self.temp_deadband
            # Proportional control: stronger cooling for larger errors
            cooling_action = min(1.0, error / 3.0)
            # Combined: NEGATIVE = cooling
            return float(cooling_action), 0.0, float(-cooling_action)

        # Within comfort band: do nothing
        return 0.0, 0.0, 0.0

    def predict(self, obs) -> np.ndarray:
        """
        Generate RBC action for current timestep.
        
        Control logic:
        1. EV: Connection-aware or time-based charging
        2. Battery: Solar tracking (charge day, discharge evening)
        3. Temperature: Independent thermostat per building
        4. HVAC storage: Disabled (set to 0)
        """
        a = np.zeros(self.action_dim, dtype=np.float32)
        raw = _unwrap_to_raw_citylearn_env(self.env)
        hour = int(_current_time_index(raw) % 24)

        # =====================================================================
        # 1. EV CHARGING: Connection-aware or time-based
        # =====================================================================
        if self.ev_mode == "greedy":
            # Charge whenever EV is connected (greedy)
            for i in self.ev_indices:
                a[i] = 1.0 if self._charger_connected_now(raw, self.action_names[i]) else 0.0
        elif self.ev_mode == "time_based":
            # Charge during off-peak hours (22:00-06:00)
            ev_action = self._ev_action_time_based(hour)
            for i in self.ev_indices:
                a[i] = ev_action
        else:
            # Unknown mode: disable EV charging
            for i in self.ev_indices:
                a[i] = 0.0

        # =====================================================================
        # 2. BATTERY: Solar tracking strategy
        # =====================================================================
        batt = float(self._battery_action(hour))
        for i in self.battery_indices:
            a[i] = batt

        # =====================================================================
        # 3. TEMPERATURE CONTROL: Per-building thermostat logic
        # =====================================================================
        if self.has_temp_control:
            # Control each building independently
            for building_idx in range(self.num_buildings):
                cool_action, heat_action, combined_action = self._temp_control_action(raw, building_idx)

                # Separate cooling device for this building
                if building_idx in self.cooling_device_map:
                    action_idx = self.cooling_device_map[building_idx]
                    a[action_idx] = cool_action

                # Separate heating device for this building
                if building_idx in self.heating_device_map:
                    action_idx = self.heating_device_map[building_idx]
                    a[action_idx] = heat_action

                # Combined HVAC device for this building
                if building_idx in self.combined_hvac_map:
                    action_idx = self.combined_hvac_map[building_idx]
                    a[action_idx] = combined_action

        # =====================================================================
        # 4. HVAC STORAGE: Disabled (not used in this strategy)
        # =====================================================================
        for i in self.cooling_storage_indices:
            a[i] = 0.0
        for i in self.heating_storage_indices:
            a[i] = 0.0

        return a


class RBCAgentWithTemp:
    """
    Wrapper for IntelligentRBCWithTemp.
    
    Compatible with OmniSafe and standard RL evaluation loops.
    
    Usage:
        agent = RBCAgentWithTemp(ev_mode="greedy", temp_deadband=0.5)
        obs, info = env.reset()
        agent.reset(env)
        
        for step in range(max_steps):
            action = agent.act(obs, info)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break
    """

    def __init__(self, ev_mode: str = "greedy", temp_deadband: float = 0.5):
        """
        Initialize RBC agent.
        
        Args:
            ev_mode: "greedy" (charge when connected) or "time_based" (charge 22:00-06:00)
            temp_deadband: Temperature deadband in °C (default: 0.5°C)
        """
        self.ev_mode = ev_mode
        self.temp_deadband = float(temp_deadband)
        self._rbc: Optional[IntelligentRBCWithTemp] = None

    def reset(self, env: Any) -> None:
        """Initialize RBC controller with environment."""
        self._rbc = IntelligentRBCWithTemp(
            env, 
            ev_mode=self.ev_mode, 
            temp_deadband=self.temp_deadband
        )

    def act(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """Generate action for current observation."""
        if self._rbc is None:
            raise RuntimeError("RBCAgentWithTemp not initialized. Call reset() first.")
        return self._rbc.predict(obs)
