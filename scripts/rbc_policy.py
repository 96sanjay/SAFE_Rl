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


class SmartV2GRBC:
    """Price-, solar-, and grid-aware RBC with priority-based V2G for BC warmstart.

    Battery logic:
      - Excess solar → charge proportional to surplus
      - High price → discharge (peak shaving)
      - Shoulder hours + above-median price → gentle discharge
      - Low price → charge from cheap grid
      - Default → idle

    EV logic (priority cascade):
      1. Must-charge: departure urgency overrides everything
      2. Solar window: cheap clean energy → charge 0.8
      3. V2G discharge: profitable + enough headroom → discharge -0.5
      4. Off-peak cheap: low price → charge 0.6
      5. Default connected: gentle charge 0.3
      6. Not connected: 0.0
    """

    def __init__(self, env):
        self.env = env
        self._raw = unwrap_to_raw_citylearn_env(env)
        if self._raw is None or not hasattr(self._raw, "buildings"):
            raise RuntimeError("Cannot find CityLearn env for SmartV2GRBC")

        names = _unwrap_action_names(getattr(self._raw, "action_names", []))
        if not names:
            raise AttributeError("Could not find action_names")

        self.action_names = names
        self.action_dim = int(env.action_space.shape[0]) if hasattr(env.action_space, "shape") else len(names)
        self.battery_indices = [i for i, n in enumerate(names) if str(n).strip().lower() == "electrical_storage"]
        self.ev_indices = [i for i, n in enumerate(names) if "electric_vehicle_storage_charger_" in str(n).lower()]

        # Pre-compute price percentiles from the full pricing series
        buildings = list(getattr(self._raw, "buildings", []))
        prices = np.array([], dtype=float)
        if buildings:
            try:
                prices = np.asarray(buildings[0].pricing.electricity_pricing, dtype=float)
                prices = prices[np.isfinite(prices)]
            except Exception:
                pass
        if len(prices) > 0:
            self.price_low = float(np.percentile(prices, 25))
            self.price_median = float(np.percentile(prices, 50))
            self.price_high = float(np.percentile(prices, 75))
        else:
            self.price_low, self.price_median, self.price_high = 0.10, 0.17, 0.22

        # Map action index → building index for batteries
        self._batt_to_bldg = {}
        for bi, idx in enumerate(self.battery_indices):
            if bi < len(buildings):
                self._batt_to_bldg[idx] = bi

        print(f"[SmartV2GRBC] action_dim={self.action_dim} batt={self.battery_indices} "
              f"ev={self.ev_indices}")
        print(f"[SmartV2GRBC] price: low={self.price_low:.4f} med={self.price_median:.4f} "
              f"high={self.price_high:.4f}")

    def _get_ev_state(self, buildings, action_idx, t_now):
        """Get EV state: (soc, required_soc, hours_to_departure) or None if not connected."""
        if action_idx >= len(self.action_names):
            return None
        action_name = str(self.action_names[action_idx]).strip().lower()
        if "electric_vehicle_storage_charger_" not in action_name:
            return None
        suffix = action_name.split("electric_vehicle_storage_charger_", 1)[1]
        charger_id = f"charger_{suffix}"

        t_state = t_now + 1
        if t_state < 0:
            t_state = 0

        for b in buildings:
            for ch in getattr(b, "electric_vehicle_chargers", []) or []:
                cid = getattr(ch, "charger_id", getattr(ch, "name", None))
                if _norm_id(cid) != charger_id:
                    continue
                sim = getattr(ch, "charger_simulation",
                              getattr(ch, "_Charger__charger_simulation", None))
                if sim is None:
                    return None
                try:
                    state = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                    ev_id_arr = getattr(sim, "_electric_vehicle_id")
                    req_soc = np.asarray(getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
                    dep_time = np.asarray(getattr(sim, "_electric_vehicle_departure_time"), dtype=float)
                except Exception:
                    return None
                if state.ndim != 1 or t_state >= len(state):
                    return None
                st = float(state[t_state])
                eid = ev_id_arr[t_state]
                if not (st == 1.0 and _is_valid_ev_id(eid)):
                    return None

                # Get current SoC from connected EV
                ev = getattr(ch, "connected_electric_vehicle", None)
                soc = 0.5
                if ev is not None:
                    batt = getattr(ev, "battery", None)
                    if batt is not None and hasattr(batt, "soc"):
                        try:
                            soc_arr = np.asarray(getattr(batt, "soc"), dtype=float)
                            soc_idx = max(0, t_now - 1) if t_now > 0 else 0
                            if soc_arr.ndim == 1 and soc_idx < len(soc_arr):
                                soc = float(np.clip(soc_arr[soc_idx], 0.0, 1.0))
                        except Exception:
                            pass

                required = float(req_soc[t_state]) if t_state < len(req_soc) else 0.8
                if not np.isfinite(required):
                    required = 1.0
                required = np.clip(required, 0.0, 1.0)
                dep = float(dep_time[t_state]) if t_state < len(dep_time) else 24.0
                hours_to_dep = max(0.0, dep) if np.isfinite(dep) else 24.0

                return (soc, required, hours_to_dep)
        return None

    def predict(self, obs) -> np.ndarray:
        a = np.zeros(self.action_dim, dtype=np.float32)
        raw = self._raw
        t_now = int(getattr(raw, "time_step", 0))
        t_idx = max(0, t_now - 1)
        hour = t_now % 24
        buildings = list(getattr(raw, "buildings", []))

        # Read current price
        current_price = self.price_median
        if buildings:
            try:
                pr = buildings[0].pricing.electricity_pricing
                if hasattr(pr, "__len__") and len(pr) > t_idx:
                    current_price = float(pr[t_idx])
            except Exception:
                pass

        # Aggregate solar and load across buildings
        solar_total = 0.0
        total_load = 0.0
        per_bldg_solar = []
        per_bldg_load = []
        for b in buildings:
            b_solar = 0.0
            b_load = 0.0
            try:
                s = getattr(b, "solar_generation", None)
                if s is not None and hasattr(s, "__len__") and len(s) > t_idx:
                    b_solar = abs(float(s[t_idx]))
            except Exception:
                pass
            try:
                nl = getattr(b, "non_shiftable_load", None)
                if nl is not None and hasattr(nl, "__len__") and len(nl) > t_idx:
                    b_load = float(nl[t_idx])
            except Exception:
                pass
            per_bldg_solar.append(b_solar)
            per_bldg_load.append(b_load)
            solar_total += b_solar
            total_load += b_load

        # ── BATTERIES (per building) ──
        for idx in self.battery_indices:
            b_idx = self._batt_to_bldg.get(idx)
            if b_idx is None or b_idx >= len(buildings):
                continue
            b = buildings[b_idx]
            b_solar = per_bldg_solar[b_idx]
            b_load = per_bldg_load[b_idx]
            solar_excess = b_solar - b_load

            # Read battery SoC
            batt_soc = 0.5
            try:
                es = getattr(b, "electrical_storage", None)
                if es is not None:
                    soc_val = getattr(es, "soc", None)
                    if soc_val is not None:
                        soc_arr = np.asarray(soc_val, dtype=float)
                        if soc_arr.ndim == 0:
                            batt_soc = float(soc_arr)
                        elif soc_arr.ndim == 1 and len(soc_arr) > t_idx:
                            batt_soc = float(soc_arr[t_idx])
                    cap = float(getattr(es, "capacity", 1.0) or 1.0)
            except Exception:
                cap = 1.0

            if solar_excess > 0.5:
                a[idx] = min(0.9, solar_excess / max(cap, 0.1))
            elif current_price > self.price_high and batt_soc > 0.2:
                a[idx] = -0.7
            elif current_price > self.price_median and 17 <= hour <= 21 and batt_soc > 0.3:
                a[idx] = -0.3
            elif current_price < self.price_low:
                a[idx] = 0.4
            else:
                a[idx] = 0.0

        # ── EVs (priority-based) ──
        for idx in self.ev_indices:
            ev_state = self._get_ev_state(buildings, idx, t_now)
            if ev_state is None:
                a[idx] = 0.0
                continue

            soc, req, hours = ev_state
            deficit = max(0.0, req - soc)

            # 1. MUST-CHARGE: departure urgency
            if deficit > 0 and (hours < 3 or deficit / 0.2 > hours * 0.8):
                a[idx] = 1.0
            # 2. SOLAR WINDOW: charge to 100% (build V2G headroom), headroom-limited
            elif solar_total > total_load * 0.5:
                a[idx] = 1.0  # charge max during solar for V2G surplus
            # 3. V2G: time-slack based (R28 fix — was impossible with soc > req+0.20)
            elif current_price > self.price_high and hours > 2:
                soc_per_step = 0.10  # conservative: ~10% SoC per hour
                soc_after = soc - soc_per_step
                deficit_after = max(0.0, req - soc_after)
                hours_to_recharge = deficit_after / max(soc_per_step, 1e-9)
                can_v2g = (hours_to_recharge < hours * 0.7)
                if can_v2g:
                    a[idx] = -0.5
                else:
                    a[idx] = 0.3  # can't V2G safely, gentle charge instead
            # 4. OFF-PEAK: cheap charging
            elif current_price < self.price_low:
                a[idx] = 0.6
            # 5. DEFAULT: gentle charge
            else:
                a[idx] = 0.3

        return a
