
"""
scripts/run_intelligent_rbc_v3.py

V3 RBC baseline using action-based deficit calculation.
Imports from extractors_v2.py and safety_env_v2.py (which contain V3 logic).

Temperature control:
- Cooling: action = 0.5 if indoor_temp > 24°C, else 0.0
- Heating: action = 0.5 if indoor_temp < 20°C, else 0.0
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Set, Tuple
import numpy as np
import pandas as pd

REPO_ROOT = "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
sys.path.insert(0, REPO_ROOT)

from scripts.make_env import make_base_env
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3  # V3 logic, V2 filename
from citylearn_safe.extractors_v3 import (  # V3 logic, V2 filename
    unwrap_to_raw_citylearn_env,
    _ev_departure_records_v3,
    current_time_index,
)


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
    """Time-based battery + greedy EV + temperature control"""

    def __init__(self, env: CityLearnSafetyEnvV3, ev_mode: str = "always"):
        self.env = env
        self.ev_mode = ev_mode

        names = _unwrap_action_names(env.action_names)
        self.action_names = names

        if hasattr(env.action_space, "shape"):
            self.action_dim = int(env.action_space.shape[0])
        elif isinstance(env.action_space, list) and hasattr(env.action_space[0], "shape"):
            self.action_dim = int(env.action_space[0].shape[0])
        else:
            self.action_dim = len(names)

        self.battery_indices = [
            i for i, n in enumerate(names)
            if str(n).strip().lower() == "electrical_storage"
        ]

        self.ev_indices = [
            i for i, n in enumerate(names)
            if "electric_vehicle_storage_charger" in str(n).lower()
        ]

        self.cooling_indices = [
            i for i, n in enumerate(names)
            if str(n).strip().lower() == "cooling_storage"
        ]

        self.heating_indices = [
            i for i, n in enumerate(names)
            if str(n).strip().lower() == "heating_storage"
        ]

        self.other_indices = [
            i for i in range(len(names))
            if i not in set(self.battery_indices)
            and i not in set(self.ev_indices)
            and i not in set(self.cooling_indices)
            and i not in set(self.heating_indices)
        ]

        print("\n" + "=" * 90)
        print("IntelligentRBC V3 (ACTION-BASED)")
        print("=" * 90)
        print(f"Action dim: {self.action_dim}")
        print(f"Battery indices ({len(self.battery_indices)}): {self.battery_indices}")
        print(f"EV indices ({len(self.ev_indices)}): {self.ev_indices}")
        print(f"Cooling indices ({len(self.cooling_indices)}): {self.cooling_indices}")
        print(f"Heating indices ({len(self.heating_indices)}): {self.heating_indices}")
        if self.other_indices:
            print(f"Other indices ({len(self.other_indices)}): {self.other_indices}")
        print(f"EV mode: {self.ev_mode}")
        print("=" * 90 + "\n")

    def _battery_action(self, hour: int) -> float:
        return 0.8 if 10 <= hour <= 16 else (-0.6 if 17 <= hour <= 21 else 0.0)

    def _get_indoor_temps(self, raw_env: Any) -> np.ndarray:
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
            except:
                temps.append(22.0)
        return np.array(temps)

    def _charger_connected_now(self, raw_env: Any, action_name: str) -> bool:
        s = str(action_name).strip().lower()
        if "electric_vehicle_storage_charger_" not in s:
            return False

        suffix = s.split("electric_vehicle_storage_charger_", 1)[1]
        charger_id = f"charger_{suffix}"

        t_state = int(getattr(raw_env, "time_step", 0))
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

        if self.ev_mode == "always":
            for i in self.ev_indices:
                a[i] = 1.0
        elif self.ev_mode == "plugged_only":
            for i in self.ev_indices:
                a[i] = 1.0 if self._charger_connected_now(raw, self.action_names[i]) else 0.0
        else:
            raise ValueError(f"Unknown ev_mode={self.ev_mode}. Use 'always' or 'plugged_only'.")

        batt = float(self._battery_action(hour))
        for i in self.battery_indices:
            a[i] = batt

        temps = self._get_indoor_temps(raw)
        for i in self.cooling_indices:
            a[i] = 0.5 if temps[i % len(temps)] > 24.0 else 0.0
        for i in self.heating_indices:
            a[i] = 0.5 if temps[i % len(temps)] < 20.0 else 0.0

        return a


class PerBuildingMetrics:
    def __init__(self, citylearn_env):
        self.env = unwrap_to_raw_citylearn_env(citylearn_env)
        self.n_buildings = len(self.env.buildings)
        self.charger_to_building = self._build_charger_mapping()
        self._seen_departures: Set[Tuple[Any, ...]] = set()
        
        # V3: Track actions
        self.actions_history: Dict[int, Dict[int, float]] = {}
        self.timestep = 0
        self.ev_charger_action_indices = [3, 14, 18, 25, 35, 42, 52, 53]
        
        # Violation counters
        self.violation_counters = {
            "total_violations": 0,
            "building_soc_violations": 0,
            "ev_departures_with_deficit": 0,
            "ev_departures_controllable": 0,
            "ev_departures_uncontrollable": 0,
        }

    def reset_episode(self):
        self._seen_departures.clear()
        self.actions_history.clear()
        self.timestep = 0
        self.violation_counters = {
            "total_violations": 0,
            "building_soc_violations": 0,
            "ev_departures_with_deficit": 0,
            "ev_departures_controllable": 0,
            "ev_departures_uncontrollable": 0,
        }

    def store_actions(self, action: np.ndarray):
        """Store actions for V3 extractor"""
        action_flat = np.asarray(action, dtype=float).ravel()
        
        if self.timestep not in self.actions_history:
            self.actions_history[self.timestep] = {}
        
        for charger_idx in self.ev_charger_action_indices:
            if charger_idx < len(action_flat):
                self.actions_history[self.timestep][charger_idx] = float(action_flat[charger_idx])
        
        self.timestep += 1

    def _build_charger_mapping(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for bi, building in enumerate(self.env.buildings):
            chargers = getattr(building, "electric_vehicle_chargers", []) or []
            for charger in chargers:
                cid = (
                    getattr(charger, "charger_id", None)
                    or getattr(charger, "_Charger__charger_id", None)
                    or getattr(charger, "id", None)
                    or getattr(charger, "name", None)
                )
                if cid is None:
                    continue
                mapping[_norm_id(cid)] = bi
        return mapping

    def _record_key(self, record) -> Tuple[Any, ...]:
        t = int(getattr(record, "time_step", -1))
        ch = _norm_id(getattr(record, "charger_id", "UNKNOWN"))
        ev = _norm_id(getattr(record, "ev_id", "EV_UNKNOWN"))
        return (t, ch, ev)

    @staticmethod
    def _soc_to_kwh(def_soc: float, cap_kwh: float) -> float:
        if not np.isfinite(def_soc) or def_soc <= 0:
            return 0.0
        if not np.isfinite(cap_kwh) or cap_kwh <= 0:
            return 0.0
        return float(def_soc) * float(cap_kwh)

    def extract(self) -> Dict[int, Dict[str, float]]:
        t_idx = current_time_index(self.env)
        metrics: Dict[int, Dict[str, float]] = {}

        for bi, building in enumerate(self.env.buildings):
            bm: Dict[str, float] = {}

            es = getattr(building, "electrical_storage", None) or getattr(building, "electricity_storage", None)
            soc = 0.0
            if es is not None and hasattr(es, "soc"):
                try:
                    soc_arr = np.asarray(es.soc, dtype=float)
                    if soc_arr.ndim == 1 and 0 <= t_idx < len(soc_arr):
                        soc = float(soc_arr[t_idx])
                    elif np.isscalar(es.soc):
                        soc = float(es.soc)
                except Exception:
                    pass

            bm["soc"] = float(np.clip(soc, 0.0, 1.0))
            bm["soc_violation"] = float(max(0.0, bm["soc"] - 0.95))
            bm["soc_violation_flag"] = float(bm["soc"] > 0.95)

            bm["ev_deficit_soc"] = 0.0
            bm["ev_controllable_deficit_soc_v3"] = 0.0
            bm["ev_uncontrollable_deficit_soc_v3"] = 0.0
            bm["ev_deficit_kwh_true"] = 0.0
            bm["ev_controllable_deficit_kwh_true_v3"] = 0.0
            bm["ev_uncontrollable_deficit_kwh_true_v3"] = 0.0
            bm["ev_departures"] = 0.0

            metrics[bi] = bm

        # V3: Use action-based extractor from extractors_v2.py
        all_records = _ev_departure_records_v3(self.env, self.actions_history) or []
        
        departures_this_step = 0
        controllable_this_step = 0
        uncontrollable_this_step = 0
        
        for record in all_records:
            charger_id = _norm_id(getattr(record, "charger_id", "UNKNOWN"))
            bi = self.charger_to_building.get(charger_id)
            if bi is None:
                continue

            key = self._record_key(record)
            if key in self._seen_departures:
                continue
            self._seen_departures.add(key)

            cap = float(getattr(record, "capacity_kwh", float("nan")))

            da_soc = float(getattr(record, "deficit_actual", 0.0) or 0.0)
            controllable_soc = float(getattr(record, "deficit_agent_controllable", 0.0) or 0.0)
            uncontrollable_soc = float(getattr(record, "deficit_uncontrollable", 0.0) or 0.0)

            da_soc = max(0.0, da_soc)
            controllable_soc = max(0.0, controllable_soc)
            uncontrollable_soc = max(0.0, uncontrollable_soc)

            da_kwh = self._soc_to_kwh(da_soc, cap)
            controllable_kwh = self._soc_to_kwh(controllable_soc, cap)
            uncontrollable_kwh = self._soc_to_kwh(uncontrollable_soc, cap)

            metrics[bi]["ev_deficit_soc"] += da_soc
            metrics[bi]["ev_controllable_deficit_soc_v3"] += controllable_soc
            metrics[bi]["ev_uncontrollable_deficit_soc_v3"] += uncontrollable_soc

            metrics[bi]["ev_deficit_kwh_true"] += da_kwh
            metrics[bi]["ev_controllable_deficit_kwh_true_v3"] += controllable_kwh
            metrics[bi]["ev_uncontrollable_deficit_kwh_true_v3"] += uncontrollable_kwh

            metrics[bi]["ev_departures"] += 1.0
            
            if da_soc > 0:
                departures_this_step += 1
                if controllable_soc > 0:
                    controllable_this_step += 1
                if uncontrollable_soc > 0:
                    uncontrollable_this_step += 1

        self.violation_counters["ev_departures_with_deficit"] += departures_this_step
        self.violation_counters["ev_departures_controllable"] += controllable_this_step
        self.violation_counters["ev_departures_uncontrollable"] += uncontrollable_this_step
        
        step_has_building_violation = False
        step_has_ev_violation = (departures_this_step > 0)
        
        for bi in range(self.n_buildings):
            bm = metrics.get(bi, {})
            if bm.get("soc_violation_flag", 0.0) > 0:
                step_has_building_violation = True
                break
        
        if step_has_building_violation or step_has_ev_violation:
            self.violation_counters["total_violations"] += 1
        
        if step_has_building_violation:
            self.violation_counters["building_soc_violations"] += 1

        return metrics


def main():
    schema_path = os.environ.get("CITYLEARN_SCHEMA") or \
        "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs_WITH_TEMP_CONTROL/schema.json"
    os.environ["CITYLEARN_SCHEMA"] = schema_path
    os.environ["CITYLEARN_COST_MODE"] = "hinge"
    os.environ["CITYLEARN_INCLUDE_EV_COST"] = "1"

    base_env = make_base_env(central_agent=True)
    env = CityLearnSafetyEnvV3(base_env, soc_min=0.0, soc_max=0.95, cost_mode="hinge")

    agent = IntelligentRBC(env, ev_mode="plugged_only")
    building_metrics = PerBuildingMetrics(base_env)

    obs, info = env.reset(seed=42)
    building_metrics.reset_episode()

    raw_env = unwrap_to_raw_citylearn_env(base_env)

    runs_dir = Path("runs/baselines/intelligent_rbc_v3")
    runs_dir.mkdir(parents=True, exist_ok=True)

    out_full_csv = runs_dir / "kpis_with_buildings_v3.csv"

    print("\n" + "=" * 90)
    print("INTELLIGENT-RBC V3 (ACTION-BASED, MUTUALLY EXCLUSIVE)")
    print("=" * 90)
    print(f"Schema: {schema_path}")
    print(f"Output: {runs_dir}")
    print(f"Files: extractors_v2.py, safety_env_v2.py (V3 logic)")
    print("=" * 90 + "\n")

    full_rows = []
    done = False
    step = 0

    while not done:
        action = agent.predict(obs)
        
        # V3: Store actions BEFORE stepping
        building_metrics.store_actions(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        t_idx = current_time_index(raw_env)
        hour = int(t_idx % 24)

        per_building = building_metrics.extract()

        full_row: Dict[str, Any] = {
            "step": step + 1,
            "hour": hour,
            "episode": 0,
            "reward": float(reward),
        }

        N = building_metrics.n_buildings
        for building_id in range(N):
            bm = per_building.get(building_id, {})
            suffix = f"_b{building_id}"

            full_row[f"soc{suffix}"] = bm.get("soc", 0.0)
            full_row[f"soc_violation{suffix}"] = bm.get("soc_violation", 0.0)
            full_row[f"soc_violation_flag{suffix}"] = bm.get("soc_violation_flag", 0.0)

            full_row[f"ev_deficit_soc{suffix}"] = bm.get("ev_deficit_soc", 0.0)
            full_row[f"ev_controllable_v3{suffix}"] = bm.get("ev_controllable_deficit_soc_v3", 0.0)
            full_row[f"ev_uncontrollable_v3{suffix}"] = bm.get("ev_uncontrollable_deficit_soc_v3", 0.0)

            full_row[f"ev_departures{suffix}"] = bm.get("ev_departures", 0.0)

        full_rows.append(full_row)

        if (step + 1) % 500 == 0:
            print(f"  Step {step + 1:5d}")

        step += 1

    df_full = pd.DataFrame(full_rows)
    df_full.to_csv(out_full_csv, index=False)

    # ========== V3: VIOLATION SUMMARY ==========
    total_steps = step
    violations = building_metrics.violation_counters
    
    print("\n" + "=" * 90)
    print("V3 VIOLATION SUMMARY (ACTION-BASED, MUTUALLY EXCLUSIVE)")
    print("=" * 90)
    print(f"Total timesteps:                   {total_steps}")
    print(f"Total violations (timesteps):      {violations['total_violations']}")
    pct_violations = (violations['total_violations'] / total_steps * 100) if total_steps > 0 else 0.0
    print(f"Percentage violations:             {pct_violations:.2f}%")
    
    print(f"\nBreakdown (timesteps):")
    print(f"  Building SOC violations:         {violations['building_soc_violations']}")
    
    print(f"\nEV Departures:")
    print(f"  Total with deficit:              {violations['ev_departures_with_deficit']}")
    print(f"  Controllable (agent fault):      {violations['ev_departures_controllable']}")
    print(f"  Uncontrollable (physics/time):   {violations['ev_departures_uncontrollable']}")
    
    # Sanity check
    total_ev = violations['ev_departures_controllable'] + violations['ev_departures_uncontrollable']
    print(f"\n✅ V3 SANITY CHECK:")
    print(f"   Controllable + Uncontrollable = {total_ev}")
    print(f"   Total departures = {violations['ev_departures_with_deficit']}")
    if total_ev == violations['ev_departures_with_deficit']:
        print("   ✅ PASS: Categories are mutually exclusive!")
    else:
        print(f"   ⚠️  MISMATCH!")
    
    if violations['ev_departures_with_deficit'] > 0:
        pct_ctrl = (violations['ev_departures_controllable'] / violations['ev_departures_with_deficit'] * 100)
        pct_unctrl = (violations['ev_departures_uncontrollable'] / violations['ev_departures_with_deficit'] * 100)
        print(f"\nEV Departure Breakdown:")
        print(f"  Controllable:   {pct_ctrl:.2f}%")
        print(f"  Uncontrollable: {pct_unctrl:.2f}%")
    
    print("=" * 90 + "\n")
    
    # Save summary
    summary_data = {
        "total_timesteps": total_steps,
        "total_violations_timesteps": violations['total_violations'],
        "percentage_violations": pct_violations,
        "building_soc_violations": violations['building_soc_violations'],
        "ev_departures_total": violations['ev_departures_with_deficit'],
        "ev_departures_controllable": violations['ev_departures_controllable'],
        "ev_departures_uncontrollable": violations['ev_departures_uncontrollable'],
    }
    
    summary_csv = runs_dir / "violation_summary_v3.csv"
    pd.DataFrame([summary_data]).to_csv(summary_csv, index=False)
    print(f"✅ Violation summary: {summary_csv}")
    print(f"✅ Full CSV: {out_full_csv}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()