
# scripts/run_intelligent_rbc_with_buildings.py
"""Per-building KPI logging for Intelligent-RBC baseline."""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")

from scripts.make_env import make_base_env
from citylearn_safe.safety_env import CityLearnSafetyEnv
from citylearn_safe.extractors import (
    unwrap_to_raw_citylearn_env,
    _ev_departure_records,
    current_time_index,
)


class IntelligentRBC:
    def __init__(self, env):
        self.env = env
        self.action_dim = env.action_space.shape[0]

        names = env.action_names
        if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
            names = names[0]
        self.action_names = names

        self.battery_indices = [
            i
            for i, n in enumerate(names)
            if "electrical_storage" in str(n).lower() and "vehicle" not in str(n).lower()
        ]
        self.ev_indices = [
            i
            for i, n in enumerate(names)
            if "electric_vehicle" in str(n).lower() or "charger" in str(n).lower()
        ]

    def predict(self, obs):
        a = np.zeros(self.action_dim, dtype=np.float32)

        raw = unwrap_to_raw_citylearn_env(self.env)
        hour = int(current_time_index(raw) % 24)

        # Always try to charge EVs
        for i in self.ev_indices:
            a[i] = 1.0

        # Simple battery heuristic
        batt = 0.8 if 10 <= hour <= 16 else (-0.6 if 17 <= hour <= 21 else 0.0)
        for i in self.battery_indices:
            a[i] = batt

        return a


class PerBuildingMetrics:
    """
    Per-building metrics extracted from raw CityLearn env.

    EV logic is aligned to what your EOF debug proved:
      - record.deficit_* are already kWh (no multiplication by capacity)
      - record.time_step matches env.time_step (NOT t_idx = time_step - 1)
      - _ev_departure_records may reuse/mutate objects; scan all each step + dedup
    """

    def __init__(self, citylearn_env):
        self.env = unwrap_to_raw_citylearn_env(citylearn_env)
        self.n_buildings = len(self.env.buildings)

        self.charger_to_building = self._build_charger_mapping()
        self._seen_departures: Set[Tuple[Any, ...]] = set()

    def reset_episode(self):
        self._seen_departures.clear()

    def _normalize_id(self, x) -> str:
        if isinstance(x, (bytes, np.bytes_)):
            x = x.decode("utf-8", errors="ignore")
        s = str(x).strip()
        if s.startswith("b'") and s.endswith("'"):
            s = s[2:-1]
        return s.strip()

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
                mapping[self._normalize_id(cid)] = bi
        return mapping

    def _record_time_step(self, record) -> Optional[int]:
        for attr in (
            "departure_time_step",
            "departure_timestep",
            "time_step",
            "timestep",
            "time_index",
            "step",
            "time",
        ):
            if hasattr(record, attr):
                try:
                    v = getattr(record, attr)
                    if v is None:
                        continue
                    return int(v)
                except Exception:
                    pass
        return None

    def _record_key(self, record) -> Tuple[Any, ...]:
        """
        Stable dedup key. IMPORTANT:
          - DO NOT inject current timestep when record has no intrinsic time
          - DO NOT include capacity
          - include deficits and record time if present
        """
        charger = self._normalize_id(getattr(record, "charger_id", "UNKNOWN"))

        vid = None
        for attr in ("vehicle_id", "ev_id", "id", "name"):
            if hasattr(record, attr):
                try:
                    vid = self._normalize_id(getattr(record, attr))
                    break
                except Exception:
                    pass

        t_rec = self._record_time_step(record)  # may be None

        da = getattr(record, "deficit_actual", 0.0)
        dv = getattr(record, "deficit_avoidable", 0.0)
        du = getattr(record, "deficit_unavoidable", 0.0)

        da = float(da) if da is not None and np.isfinite(da) else 0.0
        dv = float(dv) if dv is not None and np.isfinite(dv) else 0.0
        du = float(du) if du is not None and np.isfinite(du) else 0.0

        da = max(0.0, da)
        dv = max(0.0, dv)
        du = max(0.0, du)

        return (charger, vid, t_rec, da, dv, du)

    def extract(self) -> Dict[int, Dict[str, float]]:
        # For time-series arrays: use t_idx aligned to CityLearn update semantics
        t_idx = current_time_index(self.env)

        # For EV records in YOUR fork: filter vs env.time_step (record.time_step == env.time_step)
        env_ts = int(getattr(self.env, "time_step", t_idx + 1))

        metrics: Dict[int, Dict[str, float]] = {}

        # ---------------- per-building time-series KPIs ----------------
        for bi, building in enumerate(self.env.buildings):
            bm: Dict[str, float] = {}

            # SoC
            es = getattr(building, "electrical_storage", None) or getattr(
                building, "electricity_storage", None
            )
            soc = 0.0
            if es is not None and hasattr(es, "soc"):
                soc_arr = np.asarray(es.soc, dtype=float)
                if soc_arr.ndim == 1 and 0 <= t_idx < len(soc_arr):
                    soc = float(soc_arr[t_idx])
                elif np.isscalar(es.soc):
                    soc = float(es.soc)

            bm["soc"] = float(np.clip(soc, 0.0, 1.0))
            bm["soc_violation"] = float(max(0.0, bm["soc"] - 0.95))
            bm["soc_violation_flag"] = float(bm["soc"] > 0.95)

            # Net electricity / grid split
            net = 0.0
            if hasattr(building, "net_electricity_consumption"):
                net_arr = np.asarray(building.net_electricity_consumption, dtype=float)
                if net_arr.ndim == 1 and 0 <= t_idx < len(net_arr):
                    net = float(net_arr[t_idx])

            bm["net_electricity_consumption"] = float(net)
            bm["grid_import_kwh"] = float(max(0.0, net))
            bm["grid_export_kwh"] = float(max(0.0, -net))

            # Solar
            sol = 0.0
            if hasattr(building, "solar_generation"):
                sol_arr = np.asarray(building.solar_generation, dtype=float)
                if sol_arr.ndim == 1 and 0 <= t_idx < len(sol_arr):
                    sol = float(sol_arr[t_idx])
            bm["solar_generation"] = float(sol)

            # Non-shiftable load
            load = 0.0
            if hasattr(building, "non_shiftable_load"):
                load_arr = np.asarray(building.non_shiftable_load, dtype=float)
                if load_arr.ndim == 1 and 0 <= t_idx < len(load_arr):
                    load = float(load_arr[t_idx])
            bm["non_shiftable_load"] = float(load)

            # EV init
            bm["ev_deficit_kwh"] = 0.0
            bm["ev_avoidable_deficit_kwh"] = 0.0
            bm["ev_unavoidable_deficit_kwh"] = 0.0
            bm["ev_departures"] = 0.0

            metrics[bi] = bm

        # ---------------- EV departures (kWh direct) ----------------
        all_records = _ev_departure_records(self.env) or []
        for record in all_records:
            charger_id = self._normalize_id(getattr(record, "charger_id", "UNKNOWN"))
            bi = self.charger_to_building.get(charger_id)
            if bi is None:
                continue

            rec_t = self._record_time_step(record)
            if rec_t is not None and int(rec_t) != env_ts:
                continue

            key = self._record_key(record)
            if key in self._seen_departures:
                continue
            self._seen_departures.add(key)

            # IMPORTANT: deficits already kWh in this fork
            da = float(getattr(record, "deficit_actual", 0.0) or 0.0)
            dv = float(getattr(record, "deficit_avoidable", 0.0) or 0.0)
            du = float(getattr(record, "deficit_unavoidable", 0.0) or 0.0)

            da = max(0.0, da)
            dv = max(0.0, dv)
            du = max(0.0, du)

            metrics[bi]["ev_deficit_kwh"] += da
            metrics[bi]["ev_avoidable_deficit_kwh"] += dv
            metrics[bi]["ev_unavoidable_deficit_kwh"] += du
            metrics[bi]["ev_departures"] += 1.0

        return metrics


def main():
    schema_path = os.environ.get("CITYLEARN_SCHEMA") or "/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
    os.environ["CITYLEARN_SCHEMA"] = schema_path
    os.environ["CITYLEARN_COST_MODE"] = "hinge"
    os.environ["CITYLEARN_INCLUDE_EV_COST"] = "1"

    base_env = make_base_env(central_agent=True)
    env = CityLearnSafetyEnv(base_env, soc_min=0.0, soc_max=0.95, cost_mode="hinge")

    agent = IntelligentRBC(env)

    # Important: PerBuildingMetrics should read from RAW CityLearn env (base_env)
    building_metrics = PerBuildingMetrics(base_env)

    obs, info = env.reset(seed=42)
    building_metrics.reset_episode()

    raw_env = unwrap_to_raw_citylearn_env(base_env)

    runs_dir = Path("runs/baselines/intelligent_rbc")
    runs_dir.mkdir(parents=True, exist_ok=True)

    step_rows = []
    done = False
    step = 0

    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        t_idx = current_time_index(raw_env)
        hour = int(t_idx % 24)

        row = {
            "step": step + 1,
            "hour": hour,
            "episode": 0,
            "reward": float(reward),
            "cost": float(info.get("cost", 0.0)),
            "cost_building_soc": float(info.get("cost_building_soc", 0.0)),
            "cost_ev_departure": float(info.get("cost_ev_departure", 0.0)),
            "soc_mean": float(info.get("soc_mean", 0.0)),
            "soc_max": float(info.get("soc_max", 0.0)),
            "soc_min": float(info.get("soc_min", 0.0)),
            "ev_departure_deficit_kwh": float(info.get("ev_departure_deficit_kwh", 0.0)),
            "ev_avoidable_deficit_kwh": float(info.get("ev_avoidable_deficit_kwh", 0.0)),
            "ev_unavoidable_deficit_kwh": float(info.get("ev_unavoidable_deficit_kwh", 0.0)),
            "ev_departure_departures": float(info.get("ev_departure_departures", 0.0)),
            "battery_abuse_kwh": float(info.get("battery_abuse_kwh", 0.0)),
            "grid_import_kwh": float(info.get("grid_import_kwh", 0.0)),
            "grid_export_kwh": float(info.get("grid_export_kwh", 0.0)),
            "step_cost": float(info.get("step_cost", 0.0)),
            "net_grid_kwh": float(info.get("grid_import_kwh", 0.0)) - float(info.get("grid_export_kwh", 0.0)),
        }

        per_building = building_metrics.extract()
        for building_id in range(building_metrics.n_buildings):
            bm = per_building.get(building_id, {})
            suffix = f"_b{building_id}"

            row[f"soc{suffix}"] = bm.get("soc", 0.0)
            row[f"soc_violation{suffix}"] = bm.get("soc_violation", 0.0)
            row[f"soc_violation_flag{suffix}"] = bm.get("soc_violation_flag", 0.0)

            row[f"ev_deficit_kwh{suffix}"] = bm.get("ev_deficit_kwh", 0.0)
            row[f"ev_avoidable_deficit_kwh{suffix}"] = bm.get("ev_avoidable_deficit_kwh", 0.0)
            row[f"ev_unavoidable_deficit_kwh{suffix}"] = bm.get("ev_unavoidable_deficit_kwh", 0.0)
            row[f"ev_departures{suffix}"] = bm.get("ev_departures", 0.0)

            row[f"net_electricity_consumption{suffix}"] = bm.get("net_electricity_consumption", 0.0)
            row[f"grid_import_kwh{suffix}"] = bm.get("grid_import_kwh", 0.0)
            row[f"grid_export_kwh{suffix}"] = bm.get("grid_export_kwh", 0.0)

            row[f"solar_generation{suffix}"] = bm.get("solar_generation", 0.0)
            row[f"non_shiftable_load{suffix}"] = bm.get("non_shiftable_load", 0.0)

        step_rows.append(row)
        step += 1

    df = pd.DataFrame(step_rows)
    N = building_metrics.n_buildings

    ev_def_cols = [f"ev_deficit_kwh_b{i}" for i in range(N) if f"ev_deficit_kwh_b{i}" in df.columns]
    ev_av_cols = [f"ev_avoidable_deficit_kwh_b{i}" for i in range(N) if f"ev_avoidable_deficit_kwh_b{i}" in df.columns]
    ev_un_cols = [f"ev_unavoidable_deficit_kwh_b{i}" for i in range(N) if f"ev_unavoidable_deficit_kwh_b{i}" in df.columns]
    net_cols = [f"net_electricity_consumption_b{i}" for i in range(N) if f"net_electricity_consumption_b{i}" in df.columns]

    df["sum_ev_deficit_kwh"] = df[ev_def_cols].sum(axis=1) if ev_def_cols else 0.0
    df["sum_ev_avoidable_deficit_kwh"] = df[ev_av_cols].sum(axis=1) if ev_av_cols else 0.0
    df["sum_ev_unavoidable_deficit_kwh"] = df[ev_un_cols].sum(axis=1) if ev_un_cols else 0.0

    df["sum_net_electricity_consumption"] = df[net_cols].sum(axis=1) if net_cols else 0.0
    df["recon_grid_import_kwh"] = df["sum_net_electricity_consumption"].clip(lower=0.0)
    df["recon_grid_export_kwh"] = (-df["sum_net_electricity_consumption"]).clip(lower=0.0)

    out_csv = runs_dir / "kpis_with_buildings.csv"
    df.to_csv(out_csv, index=False)

    district_ev_deficit = df["ev_departure_deficit_kwh"].sum()
    sum_ev_deficit = df["sum_ev_deficit_kwh"].sum()

    district_import = df["grid_import_kwh"].sum()
    recon_import = df["recon_grid_import_kwh"].sum()

    district_export = df["grid_export_kwh"].sum()
    recon_export = df["recon_grid_export_kwh"].sum()

    print(f"\n✅ Saved: {out_csv}")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")

    print(f"\nEV Deficit Total:")
    print(f"  District:   {district_ev_deficit:10.4f} kWh")
    print(f"  Sum(bldg):  {sum_ev_deficit:10.4f} kWh")
    print(f"  Diff:       {abs(district_ev_deficit - sum_ev_deficit):10.8f} kWh")

    print(f"\nGrid Totals:")
    print(f"  Dist Import:  {district_import:10.2f} kWh")
    print(f"  Recon Import: {recon_import:10.2f} kWh")
    print(f"  Dist Export:  {district_export:10.2f} kWh")
    print(f"  Recon Export: {recon_export:10.2f} kWh")

    env.close()


if __name__ == "__main__":
    main()
