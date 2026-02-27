"""
COMPLETE RBC Comparison with ALL metrics (per-building + district + CityLearn KPIs)
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
from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3
from citylearn_safe.extractors_v3 import unwrap_to_raw_citylearn_env, current_time_index, _ev_departure_records_v3


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
        names = _unwrap_action_names(env.action_names)
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
            except:
                temps.append(22.0)
        return np.array(temps)

    def _charger_connected_now(self, raw_env, action_name: str) -> bool:
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
                except:
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

        if self.ev_mode == "greedy":
            for i in self.ev_indices:
                a[i] = 1.0 if self._charger_connected_now(raw, self.action_names[i]) else 0.0
        elif self.ev_mode == "time_based":
            ev_action = self._ev_action_time_based(hour)
            for i in self.ev_indices:
                a[i] = ev_action

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
    """Extract per-building metrics"""
    
    def __init__(self, citylearn_env):
        self.env = unwrap_to_raw_citylearn_env(citylearn_env)
        self.n_buildings = len(self.env.buildings)
        self.charger_to_building = self._build_charger_mapping()
        self._seen_departures: Set[Tuple[Any, ...]] = set()
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
        self.violation_counters = {k: 0 for k in self.violation_counters}

    def store_actions(self, action: np.ndarray):
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
                cid = getattr(charger, "charger_id", None) or getattr(charger, "name", None)
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

            # Battery SOC
            es = getattr(building, "electrical_storage", None)
            soc = 0.0
            if es is not None and hasattr(es, "soc"):
                try:
                    soc_arr = np.asarray(es.soc, dtype=float)
                    if soc_arr.ndim == 1 and 0 <= t_idx < len(soc_arr):
                        soc = float(soc_arr[t_idx])
                except:
                    pass

            bm["soc"] = float(np.clip(soc, 0.0, 1.0))
            bm["soc_violation"] = float(max(0.0, bm["soc"] - 0.95))
            bm["soc_violation_flag"] = float(bm["soc"] > 0.95)

            # Net consumption, solar, load
            net = 0.0
            if hasattr(building, "net_electricity_consumption"):
                try:
                    net_arr = np.asarray(building.net_electricity_consumption, dtype=float)
                    if net_arr.ndim == 1 and 0 <= t_idx < len(net_arr):
                        net = float(net_arr[t_idx])
                except:
                    pass
            bm["net_electricity_consumption"] = float(net)
            bm["grid_import_kwh"] = float(max(0.0, net))
            bm["grid_export_kwh"] = float(max(0.0, -net))

            sol = 0.0
            if hasattr(building, "solar_generation"):
                try:
                    sol_arr = np.asarray(building.solar_generation, dtype=float)
                    if sol_arr.ndim == 1 and 0 <= t_idx < len(sol_arr):
                        sol = float(sol_arr[t_idx])
                except:
                    pass
            bm["solar_generation"] = float(sol)

            load = 0.0
            if hasattr(building, "non_shiftable_load"):
                try:
                    load_arr = np.asarray(building.non_shiftable_load, dtype=float)
                    if load_arr.ndim == 1 and 0 <= t_idx < len(load_arr):
                        load = float(load_arr[t_idx])
                except:
                    pass
            bm["non_shiftable_load"] = float(load)

            # EV metrics
            bm["ev_deficit_soc"] = 0.0
            bm["ev_controllable_deficit_soc_v3"] = 0.0
            bm["ev_uncontrollable_deficit_soc_v3"] = 0.0
            bm["ev_deficit_kwh_true"] = 0.0
            bm["ev_controllable_deficit_kwh_true_v3"] = 0.0
            bm["ev_uncontrollable_deficit_kwh_true_v3"] = 0.0
            bm["ev_departures"] = 0.0

            metrics[bi] = bm

        # EV departures
        all_records = _ev_departure_records_v3(self.env, self.actions_history) or []
        
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
                self.violation_counters["ev_departures_with_deficit"] += 1
                if controllable_soc > 0:
                    self.violation_counters["ev_departures_controllable"] += 1
                if uncontrollable_soc > 0:
                    self.violation_counters["ev_departures_uncontrollable"] += 1

        # Count violations
        step_has_building_violation = any(metrics.get(bi, {}).get("soc_violation_flag", 0.0) > 0 for bi in range(self.n_buildings))
        
        if step_has_building_violation:
            self.violation_counters["total_violations"] += 1
            self.violation_counters["building_soc_violations"] += 1

        return metrics


def run_baseline(ev_mode: str, output_dir: Path):
    """Run baseline with COMPLETE metrics"""
    
    print(f"\n{'='*90}")
    print(f"RUNNING: {ev_mode.upper()}")
    print(f"{'='*90}")
    
    base_env = make_base_env(central_agent=True)
    env = CityLearnSafetyEnvV3(base_env, soc_min=0.0, soc_max=0.95, cost_mode="hinge")
    agent = IntelligentRBC(env, ev_mode=ev_mode)
    building_metrics = PerBuildingMetrics(base_env)

    obs, info = env.reset(seed=42)
    building_metrics.reset_episode()

    rows = []
    step = 0
    done = False

    while not done:
        action = agent.predict(obs)
        building_metrics.store_actions(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        per_building = building_metrics.extract()

        # Merge district info + per-building
        row = {"step": step + 1}
        row.update(info)  # District metrics from wrapper
        
        # Add per-building metrics
        for bi in range(building_metrics.n_buildings):
            bm = per_building.get(bi, {})
            for k, v in bm.items():
                row[f"{k}_b{bi}"] = v

        rows.append(row)

        if (step + 1) % 1000 == 0:
            print(f"  Step {step + 1:5d}")

        step += 1

    df = pd.DataFrame(rows)
    out_csv = output_dir / f"kpis_{ev_mode}_COMPLETE.csv"
    df.to_csv(out_csv, index=False)
    
    # Violation summary
    violations = building_metrics.violation_counters
    total_steps = step
    pct_violations = (violations['total_violations'] / total_steps * 100) if total_steps > 0 else 0.0
    
    summary = {
        "ev_mode": ev_mode,
        "total_timesteps": total_steps,
        "total_violations_timesteps": violations['total_violations'],
        "percentage_violations": pct_violations,
        "building_soc_violations": violations['building_soc_violations'],
        "ev_departures_total": violations['ev_departures_with_deficit'],
        "ev_departures_controllable": violations['ev_departures_controllable'],
        "ev_departures_uncontrollable": violations['ev_departures_uncontrollable'],
    }
    
    summary_csv = output_dir / f"violation_summary_{ev_mode}.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    
    print(f"\n✅ Saved: {out_csv}")
    print(f"   {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✅ Violations: {summary_csv}")
    
    return df, summary


def main():
    schema_path = os.environ.get("CITYLEARN_SCHEMA") or \
        "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs_WITH_TEMP_CONTROL/schema.json"
    os.environ["CITYLEARN_SCHEMA"] = schema_path
    
    runs_dir = Path("runs/baselines/rbc_comparison_COMPLETE")
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*90}")
    print("RBC COMPARISON WITH COMPLETE METRICS (Per-Building + District + KPIs)")
    print(f"{'='*90}")
    
    df_greedy, summary_greedy = run_baseline("greedy", runs_dir)
    df_time, summary_time = run_baseline("time_based", runs_dir)
    
    print(f"\n{'='*90}")
    print("✅ COMPLETE! Now you have EVERYTHING the professor needs!")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
