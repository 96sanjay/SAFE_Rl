"""
Predictive Safety Filter (PSF) Wrapper for CityLearn.

Sits ABOVE CityLearnSafetyEnvV3:
    RL policy -> PSF -> CityLearnSafetyEnvV3 -> base CityLearnEnv

On each step:
  1. Receive RL-proposed action
  2. Extract current state from inner env
  3. Predict constraint violations over horizon H
  4. If violation predicted, minimally modify action (heuristic or QP)
  5. Pass filtered action to inner env.step()

Schema-robust: all indices discovered from action_names / action_space.
Multi-agent safe: handles list-of-arrays obs and actions.
"""
from __future__ import annotations
import os, re, warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym

# ---------------------------------------------------------------------------
# Action mapping (schema-robust, discovered at init)
# ---------------------------------------------------------------------------
@dataclass
class PSFActionMap:
    flat_names: List[str] = field(default_factory=list)
    total_dim: int = 0
    agent_dims: List[int] = field(default_factory=list)
    battery_gidx: List[int] = field(default_factory=list)
    ev_gidx: List[int] = field(default_factory=list)
    charger_id_to_gidx: Dict[str, int] = field(default_factory=dict)
    gidx_to_charger_id: Dict[int, str] = field(default_factory=dict)
    gidx_to_building: Dict[int, int] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# EV state snapshot (per-charger, for deadline prediction)
# ---------------------------------------------------------------------------
@dataclass
class EVChargerState:
    charger_id: str
    building_idx: int
    action_gidx: int
    connected: bool = False
    ev_name: Optional[str] = None
    current_soc: float = 0.0
    required_soc: float = 1.0
    departure_time_remaining: int = 0
    capacity_kwh: float = 0.0
    max_charge_power_kw: float = 0.0
    dt_hours: float = 1.0

    @property
    def deficit_soc(self) -> float:
        return max(0.0, self.required_soc - self.current_soc)

    @property
    def max_soc_per_step(self) -> float:
        if self.capacity_kwh <= 0: return 0.0
        return (self.max_charge_power_kw * self.dt_hours) / self.capacity_kwh

    @property
    def steps_needed_full_charge(self) -> int:
        per_step = self.max_soc_per_step
        if per_step <= 1e-9: return 999999
        return int(np.ceil(self.deficit_soc / per_step))

    @property
    def feasible(self) -> bool:
        if self.deficit_soc <= 1e-6: return True
        return self.steps_needed_full_charge <= max(1, self.departure_time_remaining)

    @property
    def urgency(self) -> float:
        if self.deficit_soc <= 1e-6: return 0.0
        steps_avail = max(1, self.departure_time_remaining)
        return min(1.0, self.steps_needed_full_charge / steps_avail)

# ---------------------------------------------------------------------------
# Constraint prediction results
# ---------------------------------------------------------------------------
@dataclass
class ConstraintPrediction:
    ev_violations: Dict[str, float] = field(default_factory=dict)
    ev_min_actions: Dict[int, float] = field(default_factory=dict)
    battery_soc_violations: List[Tuple[int, float, str]] = field(default_factory=list)
    building_power_violations: List[Tuple[int, float]] = field(default_factory=list)
    grid_import_excess: float = 0.0

    @property
    def any_violation(self) -> bool:
        return (len(self.ev_violations) > 0
                or len(self.battery_soc_violations) > 0
                or len(self.building_power_violations) > 0
                or self.grid_import_excess > 0)

# ---------------------------------------------------------------------------
# Main PSF Wrapper
# ---------------------------------------------------------------------------
_EV_RE = re.compile(r"^electric_vehicle_storage_charger_(?P<suffix>.+)$", re.IGNORECASE)

class PredictiveSafetyFilterWrapper(gym.Wrapper):
    """
    Predictive Safety Filter wrapper for CityLearn.
    Wraps a CityLearnSafetyEnvV3 (or any gym.Env with compatible interface).
    """
    metadata = {"render_modes": []}

    def __init__(self, env: gym.Env, *, horizon: int = 24,
                 soc_low: float = 0.05, soc_high: float = 0.95,
                 p_building_max: float = 50.0, p_grid_max: float = 500.0,
                 ev_urgency_threshold: float = 0.5,
                 correction_mode: str = "heuristic",
                 enable_battery_correction: bool = True,
                 enable_grid_correction: bool = True,
                 verbose: int = 1):
        super().__init__(env)
        self.horizon = int(os.environ.get("PSF_HORIZON", str(horizon)))
        self.soc_low = float(os.environ.get(
            "CITYLEARN_STEMS_SOC_LOW", str(soc_low)))
        self.soc_high = float(os.environ.get(
            "CITYLEARN_STEMS_SOC_HIGH", str(soc_high)))
        self.p_building_max = float(os.environ.get(
            "CITYLEARN_STEMS_P_BUILDING_MAX", str(p_building_max)))
        self.p_grid_max = float(os.environ.get(
            "CITYLEARN_STEMS_P_GRID_MAX", str(p_grid_max)))
        self.ev_urgency_threshold = float(
            os.environ.get("PSF_EV_URGENCY_THRESHOLD", str(ev_urgency_threshold)))
        self.correction_mode = os.environ.get(
            "PSF_CORRECTION_MODE", correction_mode).strip().lower()
        self.enable_battery_correction = bool(enable_battery_correction)
        self.enable_grid_correction = bool(enable_grid_correction)
        self.verbose = int(os.environ.get("PSF_VERBOSE", str(verbose)))

        assert self.correction_mode in ("heuristic", "passthrough"), \
            f"Unknown PSF correction_mode: {self.correction_mode}"

        self._action_map: Optional[PSFActionMap] = None
        self._is_multi_agent = isinstance(self.action_space, list)
        self._step_count = 0
        self._episode_count = 0
        self._psf_interventions = 0
        self._psf_ev_interventions = 0
        self._psf_battery_interventions = 0
        self._psf_grid_interventions = 0

        print(f"[PSF] Initialized | horizon={self.horizon} "
              f"mode={self.correction_mode} multi_agent={self._is_multi_agent}")

    # -- Action flatten / unflatten --
    def _flatten_action(self, action) -> np.ndarray:
        if isinstance(action, (list, tuple)):
            parts = [np.asarray(a, dtype=np.float64).ravel() for a in action]
            return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
        return np.asarray(action, dtype=np.float64).ravel()

    def _unflatten_action(self, flat: np.ndarray) -> Any:
        if not self._is_multi_agent:
            return flat.astype(np.float32)
        am = self._get_action_map()
        result = []
        offset = 0
        for dim in am.agent_dims:
            result.append(flat[offset:offset + dim].astype(np.float32))
            offset += dim
        assert offset == len(flat), f"[PSF] unflatten mismatch: {offset} vs {len(flat)}"
        return result

    # -- Action map discovery --
    def _get_action_map(self) -> PSFActionMap:
        if self._action_map is not None:
            return self._action_map
        self._action_map = self._build_action_map()
        return self._action_map

    def _build_action_map(self) -> PSFActionMap:
        names_raw = getattr(self.env, "action_names", None)
        if names_raw is None:
            city = self._get_citylearn_env()
            if city is not None:
                names_raw = getattr(city, "action_names", None)
        if names_raw is None:
            raise RuntimeError("[PSF] Cannot find action_names on env.")

        am = PSFActionMap()
        if isinstance(names_raw, list) and len(names_raw) > 0 and isinstance(names_raw[0], list):
            g = 0
            for b_i, sub in enumerate(names_raw):
                if not isinstance(sub, list): sub = [sub]
                am.agent_dims.append(len(sub))
                for n in sub:
                    n = str(n)
                    am.flat_names.append(n)
                    am.gidx_to_building[g] = b_i
                    g += 1
        else:
            for g, n in enumerate(names_raw):
                n = str(n)
                am.flat_names.append(n)
                am.gidx_to_building[g] = g
            am.agent_dims = [len(am.flat_names)]

        am.total_dim = len(am.flat_names)
        for g, n in enumerate(am.flat_names):
            nl = n.lower()
            if nl == "electrical_storage":
                am.battery_gidx.append(g)
            m = _EV_RE.match(n)
            if m:
                suffix = m.group("suffix")
                cid = f"charger_{suffix}"
                am.ev_gidx.append(g)
                am.charger_id_to_gidx[cid] = g
                am.gidx_to_charger_id[g] = cid

        if self.verbose >= 1:
            print(f"[PSF] ActionMap: dim={am.total_dim} batt={len(am.battery_gidx)} "
                  f"ev={len(am.ev_gidx)} agents={len(am.agent_dims)}")
        return am

    # -- CityLearn env access --
    def _get_citylearn_env(self) -> Any:
        if hasattr(self.env, "_get_citylearn_env"):
            result = self.env._get_citylearn_env()
            if result is not None: return result
        cur = self.env
        seen = set()
        for _ in range(40):
            if cur is None or id(cur) in seen: break
            seen.add(id(cur))
            blds = getattr(cur, "buildings", None)
            ts = getattr(cur, "time_step", None)
            if blds is not None and hasattr(blds, "__len__") and len(blds) > 0 and ts is not None:
                return cur
            for attr in ("base", "env", "unwrapped", "_env", "raw_env"):
                nxt = getattr(cur, attr, None)
                if nxt is not None and nxt is not cur:
                    cur = nxt
                    break
            else:
                break
        return None

    def _get_time_step(self) -> int:
        city = self._get_citylearn_env()
        return int(getattr(city, "time_step", 0)) if city else 0

    def _get_state_index(self) -> int:
        return max(0, self._get_time_step() - 1)

    # -- EV state extraction --
    def _extract_ev_states(self) -> List[EVChargerState]:
        city = self._get_citylearn_env()
        if city is None: return []
        am = self._get_action_map()
        t_state = int(getattr(city, "time_step", 0))
        dt_h = float(getattr(city, "seconds_per_time_step", 3600)) / 3600.0
        states: List[EVChargerState] = []

        for b_idx, b in enumerate(getattr(city, "buildings", [])):
            chargers = getattr(b, "electric_vehicle_chargers", None) or []
            for ch in chargers:
                cid_raw = getattr(ch, "charger_id", getattr(ch, "name", ""))
                cid = self._norm_id(cid_raw)
                gidx = am.charger_id_to_gidx.get(cid)
                if gidx is None: continue

                es = EVChargerState(charger_id=cid, building_idx=b_idx,
                                    action_gidx=gidx, dt_hours=dt_h)
                sim = getattr(ch, "charger_simulation",
                              getattr(ch, "_Charger__charger_simulation", None))
                if sim is None:
                    states.append(es); continue

                try:
                    state_arr = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                    dep_time = np.asarray(getattr(sim, "_electric_vehicle_departure_time"), dtype=float)
                    req_soc = np.asarray(getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
                    ev_id = np.asarray(getattr(sim, "_electric_vehicle_id"))
                except (AttributeError, TypeError):
                    states.append(es); continue

                if t_state >= len(state_arr):
                    states.append(es); continue

                s = float(state_arr[t_state])
                es.connected = (s == 1.0)
                if not es.connected:
                    states.append(es); continue

                ev_name_raw = ev_id[t_state]
                if isinstance(ev_name_raw, (bytes, np.bytes_)):
                    ev_name_raw = ev_name_raw.decode("utf-8", errors="ignore")
                es.ev_name = str(ev_name_raw) if ev_name_raw else None
                es.required_soc = float(np.clip(req_soc[t_state], 0.0, 1.0))

                d = float(dep_time[t_state])
                es.departure_time_remaining = max(0, int(d)) if np.isfinite(d) else 999

                p_max = getattr(ch, "max_charging_power",
                                getattr(ch, "_Charger__max_charging_power", None))
                if isinstance(p_max, np.ndarray): p_max = float(p_max.ravel()[0])
                elif p_max is not None: p_max = float(p_max)
                else: p_max = 0.0
                es.max_charge_power_kw = p_max

                ev_obj = getattr(ch, "connected_electric_vehicle", None)
                if ev_obj is not None:
                    batt = getattr(ev_obj, "battery", None)
                    if batt is not None:
                        cap = getattr(batt, "capacity",
                                      getattr(batt, "_EnergyStorage__capacity", None))
                        if isinstance(cap, np.ndarray): cap = float(cap.ravel()[0])
                        elif cap is not None: cap = float(cap)
                        else: cap = 0.0
                        es.capacity_kwh = cap

                        soc_arr = getattr(batt, "soc",
                                          getattr(batt, "_EnergyStorage__soc", None))
                        t_soc = max(0, t_state - 1)
                        if soc_arr is not None:
                            soc_np = np.asarray(soc_arr, dtype=float)
                            if soc_np.ndim == 1 and 0 <= t_soc < len(soc_np):
                                es.current_soc = float(np.clip(soc_np[t_soc], 0.0, 1.0))
                states.append(es)
        return states

    # -- Battery SoC extraction --
    def _extract_battery_socs(self) -> List[Tuple[int, float]]:
        city = self._get_citylearn_env()
        if city is None: return []
        am = self._get_action_map()
        t_idx = self._get_state_index()
        results: List[Tuple[int, float]] = []
        for b_idx, b in enumerate(getattr(city, "buildings", [])):
            es = getattr(b, "electrical_storage", None)
            if es is None: continue
            soc_data = getattr(es, "soc", None)
            if soc_data is None: continue
            soc_arr = np.asarray(soc_data, dtype=float)
            soc_val = float(np.clip(soc_arr[t_idx], 0.0, 1.0)) \
                if soc_arr.ndim == 1 and 0 <= t_idx < len(soc_arr) else 0.5
            gidx = am.battery_gidx[b_idx] if b_idx < len(am.battery_gidx) else -1
            if gidx >= am.total_dim:
                gidx = -1  # OOB safety: skip if gidx beyond action space
            results.append((gidx, soc_val))
        return results

    # -- Building power extraction --
    def _extract_building_powers(self) -> List[float]:
        city = self._get_citylearn_env()
        if city is None: return []
        t_idx = self._get_state_index()
        powers = []
        for b in getattr(city, "buildings", []):
            nec = getattr(b, "net_electricity_consumption", None)
            if nec is not None and hasattr(nec, "__len__") and len(nec) > t_idx:
                powers.append(float(nec[t_idx]))
            else:
                powers.append(0.0)
        return powers

    # -- Constraint prediction --
    def _predict_constraints(self, action_flat: np.ndarray) -> ConstraintPrediction:
        pred = ConstraintPrediction()
        am = self._get_action_map()

        # 1. EV deadline
        ev_states = self._extract_ev_states()
        for es in ev_states:
            if not es.connected or es.deficit_soc <= 1e-6: continue
            urgency = es.urgency
            max_per_step = es.max_soc_per_step
            if max_per_step <= 1e-9: continue
            t_remain = max(1, es.departure_time_remaining)
            min_action_uniform = float(np.clip(
                es.deficit_soc / (t_remain * max_per_step), 0.0, 1.0))
            proposed = float(action_flat[es.action_gidx]) \
                if 0 <= es.action_gidx < len(action_flat) else 0.0
            if urgency >= self.ev_urgency_threshold or t_remain <= 2:
                if proposed < min_action_uniform - 1e-4:
                    pred.ev_violations[es.charger_id] = urgency
                    pred.ev_min_actions[es.action_gidx] = min_action_uniform

        # 2. Battery SoC bounds (V3: wide buffer zone [soc_low+margin, soc_high-margin])
        batt_margin = 0.15  # keep SoC well away from hard limits
        for gidx, soc in self._extract_battery_socs():
            if gidx < 0: continue
            proposed = float(action_flat[gidx]) if 0 <= gidx < len(action_flat) else 0.0
            soft_low = self.soc_low + batt_margin   # 0.20
            soft_high = self.soc_high - batt_margin  # 0.80
            if soc < self.soc_low:
                # Hard violation — already below 0.05
                pred.battery_soc_violations.append((gidx, self.soc_low - soc, "low"))
            elif soc < soft_low and proposed < 0.0:
                # In buffer zone and discharging — prevent
                pred.battery_soc_violations.append((gidx, max(0.0, self.soc_low - soc), "low"))
            elif soc > self.soc_high:
                # Hard violation — already above 0.95
                pred.battery_soc_violations.append((gidx, soc - self.soc_high, "high"))
            elif soc > soft_high and proposed > 0.0:
                # In buffer zone and charging — prevent
                pred.battery_soc_violations.append((gidx, max(0.0, soc - self.soc_high), "high"))

        # 3. Building power
        powers = self._extract_building_powers()
        for b_idx, p in enumerate(powers):
            if abs(p) > self.p_building_max:
                pred.building_power_violations.append((b_idx, abs(p) - self.p_building_max))

        # 4. Grid import
        total_import = sum(max(0.0, p) for p in powers)
        if total_import > self.p_grid_max:
            pred.grid_import_excess = total_import - self.p_grid_max

        return pred

    # -- Action correction (V5: EV+battery only, no grid override) --
    def _correct_action(self, action_flat: np.ndarray, pred: ConstraintPrediction
                        ) -> Tuple[np.ndarray, Dict[str, Any]]:
        corrected = action_flat.copy()
        info: Dict[str, Any] = {"psf_any_intervention": False, "psf_ev_interventions": 0,
                                "psf_battery_interventions": 0, "psf_grid_interventions": 0,
                                "psf_action_delta_l2": 0.0}
        if self.correction_mode == "passthrough":
            return corrected, info

        am = self._get_action_map()

        # STEP 1: EV deadline — force minimum charge (highest priority)
        for gidx, min_action in pred.ev_min_actions.items():
            if 0 <= gidx < len(corrected):
                old = corrected[gidx]
                corrected[gidx] = max(float(corrected[gidx]), min_action)
                if abs(corrected[gidx] - old) > 1e-6:
                    info["psf_ev_interventions"] += 1
                    info["psf_any_intervention"] = True

        # STEP 2: Battery SoC — closed-form exact clamp
        # For each battery: new_soc = current_soc + action * scale
        # Clamp action so new_soc stays in [soc_low, soc_high]
        if self.enable_battery_correction:
            city = self._get_citylearn_env()
            am = self._get_action_map()
            t_idx = self._get_state_index()
            if city is not None:
                for b_idx, b in enumerate(getattr(city, "buildings", [])):
                    es = getattr(b, "electrical_storage", None)
                    if es is None:
                        continue
                    gidx = am.battery_gidx[b_idx] if b_idx < len(am.battery_gidx) else -1
                    if gidx < 0 or gidx >= len(corrected):
                        continue
                    # Current SoC
                    soc_arr = np.asarray(getattr(es, "soc", []), dtype=float)
                    current_soc = float(np.clip(soc_arr[t_idx], 0.0, 1.0)) \
                        if soc_arr.ndim == 1 and 0 <= t_idx < len(soc_arr) else 0.5
                    # Battery parameters
                    capacity = float(getattr(es, "capacity", 6.4))
                    nominal_power = float(getattr(es, "nominal_power", 5.0))
                    if capacity <= 0 or nominal_power <= 0:
                        continue
                    # scale: fraction of capacity per unit action per timestep
                    scale = (nominal_power * 1.0) / capacity  # ~0.78 for 5kW/6.4kWh
                    # Exact bounds: action must satisfy soc_low <= current_soc + a*scale <= soc_high
                    a_min = max((self.soc_low - current_soc) / scale, -1.0)
                    a_max = min((self.soc_high - current_soc) / scale, 1.0)
                    old = corrected[gidx]
                    corrected[gidx] = float(np.clip(corrected[gidx], a_min, a_max))
                    if abs(corrected[gidx] - old) > 1e-6:
                        info["psf_battery_interventions"] += 1
                        info["psf_any_intervention"] = True

        # STEP 3+4: Building power (C3) + Grid import (C4) — PREDICTIVE correction
        # Uses battery AND EV charger actions. EV only reduced if urgency is low.
        if self.enable_battery_correction:
            city_c34 = self._get_citylearn_env()
            if city_c34 is not None:
                am_c34 = self._get_action_map()
                t_c34 = city_c34.time_step
                t_idx_c34 = self._get_state_index()
                buildings = list(getattr(city_c34, "buildings", []))

                # Build per-building data
                bldg_data = []
                for b_idx, b in enumerate(buildings):
                    bd = {"b_idx": b_idx, "base_net": 0.0,
                          "batt_gidx": -1, "batt_nom_p": 0.0,
                          "batt_a_min_c2": -1.0, "batt_a_max_c2": 1.0,
                          "ev_chargers": []}

                    # Base load + solar
                    raw_nsl = getattr(b, "_Building__energy_to_non_shiftable_load", None)
                    raw_sg = getattr(b, "_Building__solar_generation", None)
                    if raw_nsl is not None and raw_sg is not None:
                        nsl_arr = np.asarray(raw_nsl, dtype=float)
                        sg_arr = np.asarray(raw_sg, dtype=float)
                        if t_c34 < len(nsl_arr) and t_c34 < len(sg_arr):
                            bd["base_net"] = float(nsl_arr[t_c34]) + float(sg_arr[t_c34])

                    # Battery
                    es = getattr(b, "electrical_storage", None)
                    if es is not None:
                        gidx = am_c34.battery_gidx[b_idx] if b_idx < len(am_c34.battery_gidx) else -1
                        if 0 <= gidx < len(corrected):
                            nom_p = float(getattr(es, "nominal_power", 5.0))
                            cap = float(getattr(es, "capacity", 6.4))
                            if cap > 0 and nom_p > 0:
                                soc_arr = np.asarray(getattr(es, "soc", []), dtype=float)
                                csoc = float(np.clip(soc_arr[t_idx_c34], 0.0, 1.0))                                     if soc_arr.ndim == 1 and 0 <= t_idx_c34 < len(soc_arr) else 0.5
                                scale = (nom_p * 1.0) / cap
                                bd["batt_gidx"] = gidx
                                bd["batt_nom_p"] = nom_p
                                bd["batt_a_min_c2"] = max((self.soc_low - csoc) / scale, -1.0)
                                bd["batt_a_max_c2"] = min((self.soc_high - csoc) / scale, 1.0)

                    # EV chargers — use gidx_to_building for correct mapping
                    for ev_gidx in am_c34.ev_gidx:
                        if am_c34.gidx_to_building.get(ev_gidx) == b_idx:
                            # Find max_power for this charger
                            chargers = getattr(b, "electric_vehicle_chargers", None) or []
                            # Get EV urgency for this charger
                            urgency = pred.ev_violations.get(
                                am_c34.gidx_to_charger_id.get(ev_gidx, ""), 0.0)
                            c1_min = float(pred.ev_min_actions.get(ev_gidx, 0.0))
                            # Find matching charger max_power
                            max_p = 0.0
                            cid = am_c34.gidx_to_charger_id.get(ev_gidx, "")
                            for ch in chargers:
                                ch_max = getattr(ch, "max_charging_power",
                                                 getattr(ch, "_Charger__max_charging_power", 0))
                                if isinstance(ch_max, np.ndarray):
                                    ch_max = float(ch_max.ravel()[0])
                                elif ch_max is not None:
                                    ch_max = float(ch_max)
                                else:
                                    ch_max = 0.0
                                max_p = ch_max
                                break  # Use first unmatched charger
                            bd["ev_chargers"].append({
                                "gidx": ev_gidx, "max_p": max_p,
                                "c1_min": c1_min, "urgency": urgency})
                    bldg_data.append(bd)

                # --- C3: Per-building correction ---
                for bd in bldg_data:
                    batt_gidx = bd["batt_gidx"]
                    batt_nom_p = bd["batt_nom_p"]
                    batt_action = float(corrected[batt_gidx]) if batt_gidx >= 0 else 0.0

                    ev_power = 0.0
                    for evc in bd["ev_chargers"]:
                        g = evc["gidx"]
                        if 0 <= g < len(corrected):
                            ev_power += float(corrected[g]) * evc["max_p"]

                    pred_net = bd["base_net"] + batt_action * batt_nom_p * 2.0 + ev_power
                    if abs(pred_net) <= self.p_building_max:
                        continue

                    excess = abs(pred_net) - self.p_building_max

                    if pred_net > self.p_building_max:
                        remaining = excess
                        # 1. Reduce EV charging — ONLY if urgency < 0.3 (plenty of slack)
                        for evc in bd["ev_chargers"]:
                            if remaining <= 0.01:
                                break
                            g = evc["gidx"]
                            if g < 0 or g >= len(corrected) or evc["max_p"] <= 0:
                                continue
                            if evc["urgency"] >= 0.3:
                                continue  # Don't touch urgent EVs
                            cur_a = float(corrected[g])
                            # Floor: max of C1 minimum and current RBC action * 0.5
                            # Keep at least half the RBC action to preserve charging schedule
                            safe_min = max(evc["c1_min"], cur_a * 0.5)
                            reducible = (cur_a - safe_min) * evc["max_p"]
                            if reducible <= 0.01:
                                continue
                            reduce_kw = min(remaining, reducible)
                            reduce_a = reduce_kw / evc["max_p"]
                            old = corrected[g]
                            corrected[g] = max(safe_min, cur_a - reduce_a)
                            remaining -= reduce_kw
                            if abs(corrected[g] - old) > 1e-6:
                                info["psf_grid_interventions"] += 1
                                info["psf_any_intervention"] = True
                        # 2. Battery discharge
                        if remaining > 0.01 and batt_gidx >= 0 and batt_nom_p > 0:
                            desired_a = batt_action - remaining / (batt_nom_p * 2.0)
                            desired_a = float(np.clip(desired_a, bd["batt_a_min_c2"], bd["batt_a_max_c2"]))
                            old = corrected[batt_gidx]
                            corrected[batt_gidx] = desired_a
                            if abs(corrected[batt_gidx] - old) > 1e-6:
                                info["psf_grid_interventions"] += 1
                                info["psf_any_intervention"] = True
                    else:
                        # Export too high — charge battery
                        if batt_gidx >= 0 and batt_nom_p > 0:
                            desired_a = batt_action + excess / (batt_nom_p * 2.0)
                            desired_a = float(np.clip(desired_a, bd["batt_a_min_c2"], bd["batt_a_max_c2"]))
                            old = corrected[batt_gidx]
                            corrected[batt_gidx] = desired_a
                            if abs(corrected[batt_gidx] - old) > 1e-6:
                                info["psf_grid_interventions"] += 1
                                info["psf_any_intervention"] = True

                # --- C4: Grid import correction ---
                total_import = 0.0
                for bd in bldg_data:
                    ba = float(corrected[bd["batt_gidx"]]) if bd["batt_gidx"] >= 0 else 0.0
                    evp = sum(float(corrected[e["gidx"]]) * e["max_p"]
                              for e in bd["ev_chargers"]
                              if 0 <= e["gidx"] < len(corrected))
                    pn = bd["base_net"] + ba * bd["batt_nom_p"] * 2.0 + evp
                    total_import += max(0.0, pn)

                if total_import > self.p_grid_max:
                    excess = total_import - self.p_grid_max
                    reducibles = []
                    for bd in bldg_data:
                        # EV headroom (only low-urgency)
                        for evc in bd["ev_chargers"]:
                            g = evc["gidx"]
                            if g < 0 or g >= len(corrected) or evc["max_p"] <= 0:
                                continue
                            if evc["urgency"] >= 0.3:
                                continue
                            cur_a = float(corrected[g])
                            safe_min = max(evc["c1_min"], cur_a * 0.5)
                            headroom = (cur_a - safe_min) * evc["max_p"]
                            if headroom > 0.01:
                                reducibles.append(("ev", g, evc["max_p"], headroom, safe_min, bd))
                        # Battery headroom
                        bg = bd["batt_gidx"]
                        if bg >= 0 and bd["batt_nom_p"] > 0:
                            cur_a = float(corrected[bg])
                            headroom = (cur_a - bd["batt_a_min_c2"]) * bd["batt_nom_p"] * 2.0
                            if headroom > 0.01:
                                reducibles.append(("batt", bg, bd["batt_nom_p"], headroom, bd["batt_a_min_c2"], bd))

                    total_headroom = sum(r[3] for r in reducibles)
                    if total_headroom > 0.01:
                        for kind, gidx, power, headroom, min_a, bd in reducibles:
                            share = excess * (headroom / total_headroom)
                            old = corrected[gidx]
                            if kind == "ev":
                                corrected[gidx] = max(min_a, float(corrected[gidx]) - share / power)
                            else:
                                corrected[gidx] = float(np.clip(
                                    float(corrected[gidx]) - share / (power * 2.0),
                                    bd["batt_a_min_c2"], bd["batt_a_max_c2"]))
                            if abs(corrected[gidx] - old) > 1e-6:
                                info["psf_grid_interventions"] += 1
                                info["psf_any_intervention"] = True


        info["psf_action_delta_l2"] = float(np.linalg.norm(corrected - action_flat))
        return corrected, info

    # -- Gym reset --
    def reset(self, *, seed=None, options=None):
        if self._step_count > 0 and self.verbose >= 1:
            print(f"[PSF] Ep {self._episode_count} done | steps={self._step_count} "
                  f"interventions={self._psf_interventions} "
                  f"(ev={self._psf_ev_interventions} batt={self._psf_battery_interventions} "
                  f"grid={self._psf_grid_interventions})")

        obs, info = self.env.reset(seed=seed, options=options)
        self._action_map = None
        try: self._get_action_map()
        except Exception as e: warnings.warn(f"[PSF] ActionMap: {e}")

        self._step_count = 0
        self._psf_interventions = 0
        self._psf_ev_interventions = 0
        self._psf_battery_interventions = 0
        self._psf_grid_interventions = 0

        info = dict(info) if info else {}
        info.update({"psf_any_intervention": False, "psf_ev_interventions": 0,
                     "psf_battery_interventions": 0, "psf_grid_interventions": 0,
                     "psf_action_delta_l2": 0.0, "psf_mode": self.correction_mode,
                     "psf_horizon": self.horizon})
        return obs, info

    # -- Gym step --
    def step(self, action):
        self._step_count += 1
        action_flat = self._flatten_action(action)

        if self._is_multi_agent:
            lo = np.concatenate([np.asarray(sp.low, dtype=np.float64).ravel() for sp in self.action_space])
            hi = np.concatenate([np.asarray(sp.high, dtype=np.float64).ravel() for sp in self.action_space])
        else:
            lo = np.asarray(self.action_space.low, dtype=np.float64).ravel()
            hi = np.asarray(self.action_space.high, dtype=np.float64).ravel()
        action_flat = np.clip(action_flat, lo, hi)

        pred = self._predict_constraints(action_flat)
        corrected_flat, corr_info = self._correct_action(action_flat, pred)
        corrected_flat = np.clip(corrected_flat, lo, hi)

        if corr_info.get("psf_any_intervention", False):
            self._psf_interventions += 1
        self._psf_ev_interventions += corr_info.get("psf_ev_interventions", 0)
        self._psf_battery_interventions += corr_info.get("psf_battery_interventions", 0)
        self._psf_grid_interventions += corr_info.get("psf_grid_interventions", 0)

        if self.verbose >= 2 and corr_info.get("psf_any_intervention"):
            print(f"[PSF] step={self._step_count} INTERVENTION "
                  f"delta={corr_info['psf_action_delta_l2']:.4f}")

        action_out = self._unflatten_action(corrected_flat)
        obs, reward, term, trunc, info = self.env.step(action_out)
        info = dict(info) if info else {}
        info.update(corr_info)
        info["psf_mode"] = self.correction_mode
        info["psf_horizon"] = self.horizon

        if term or trunc:
            info["psf_total_interventions"] = self._psf_interventions
            info["psf_total_ev_interventions"] = self._psf_ev_interventions
            info["psf_total_battery_interventions"] = self._psf_battery_interventions
            info["psf_total_grid_interventions"] = self._psf_grid_interventions
            info["psf_intervention_rate"] = self._psf_interventions / max(1, self._step_count)
            self._episode_count += 1

        return obs, reward, term, trunc, info

    @staticmethod
    def _norm_id(x) -> str:
        if isinstance(x, (bytes, np.bytes_)):
            x = x.decode("utf-8", errors="ignore")
        s = str(x).strip()
        if s.startswith("b'") and s.endswith("'"): s = s[2:-1]
        return s.strip()
