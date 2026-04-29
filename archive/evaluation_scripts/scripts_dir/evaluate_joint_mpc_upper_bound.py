#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cvxpy as cp
import numpy as np

PROJECT = Path("/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork")
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

from citylearn_safe.extractors_v3 import _collect_ev_meta, unwrap_to_raw_citylearn_env


BASE_ENV = {
    "CITYLEARN_SCHEMA": f"{PROJECT}/data/citylearn_challenge_2022_phase_all_plus_evs/schema_5buildings.json",
    "CITYLEARN_CENTRAL_AGENT": "1",
    "CITYLEARN_REWARD_TYPE": "stems",
    "CITYLEARN_EXPORT_FACTOR": "0.7",
    "CITYLEARN_STEMS_P_BUILDING_MAX": "4.6083",
    "CITYLEARN_STEMS_P_GRID_MAX": "10.2352",
    "CITYLEARN_STEMS_SOC_LOW": "0.0",
    "CITYLEARN_STEMS_SOC_HIGH": "0.95",
    "CITYLEARN_EV_MISSING_ACTION_MODE": "assume_full",
    "CITYLEARN_INCLUDE_EV_COST": "1",
    "CITYLEARN_KPI_RUN_NAME": "__joint_mpc__",
    "CITYLEARN_SPATIAL_OBS": "0",
    "CITYLEARN_TEMPORAL_WINDOW": "0",
    "CITYLEARN_C3_CONTROLLABLE": "1",
    "CITYLEARN_WM_DISABLE": "1",
    "CITYLEARN_EV_ACTION_CLAMP": "0",
    "CITYLEARN_BATT_CLAMP": "0",
    "CITYLEARN_EV_SAUTE": "1",
    "CITYLEARN_EV_SAUTE_BUDGET": "25000",
    "CITYLEARN_EV_SAUTE_PENALTY": "5.0",
    "CITYLEARN_EV_SAUTE_GAMMA": "1.0",
    "CITYLEARN_EV_SAUTE_SHAPED_ALPHA": "10.0",
    "STEMS_ALPHA_GRID": "0.0",
    "STEMS_ALPHA_BUILD": "0.0",
    "STEMS_MU_ECONOMIC": "0.0",
    "STEMS_ALPHA_LOAD_SHIFT": "0.0",
    "STEMS_ALPHA_GRID_MILD": "0.3",
    "STEMS_ALPHA_EV_GUARD": "1.0",
    "STEMS_ALPHA_V2G_CONTEXT": "3.0",
    "STEMS_LAMBDA_EV": "5.0",
    "STEMS_ALPHA_BARRIER": "0.5",
    "STEMS_BETA_RAMP": "0.3",
    "STEMS_XI_RENEWABLE": "0.2",
    "STEMS_SB_ASYMMETRIC": "1",
    "STEMS_SG_EXPORT_CREDIT": "0.5",
    "STEMS_SG_THRESHOLD": "0.5",
    "STEMS_ALPHA_PEAK_SHAVE": "0.0",
    "STEMS_ALPHA_EV_SOLAR": "0.0",
    "STEMS_ALPHA_SOLAR_STORE": "0.0",
    "STEMS_SOLAR_STORE_BATT_ONLY": "0",
    "STEMS_EV_SLACK_ARB_SCALE": "2.0",
    "STEMS_ALPHA_HEADROOM": "0.0",
}


def _normalize_action_names(names_raw: Any, nb: int) -> List[List[str]]:
    if isinstance(names_raw, list) and len(names_raw) == 1 and isinstance(names_raw[0], list):
        flat_names = names_raw[0]
        batt_positions = [i for i, n in enumerate(flat_names) if str(n).lower() == "electrical_storage"]
        if len(batt_positions) == nb:
            rebuilt: List[List[str]] = []
            for b_idx in range(nb):
                start = batt_positions[b_idx]
                end = batt_positions[b_idx + 1] if b_idx + 1 < nb else len(flat_names)
                rebuilt.append([str(n) for n in flat_names[start:end]])
            return rebuilt
    if isinstance(names_raw, list) and all(isinstance(x, list) for x in names_raw):
        return [[str(n) for n in sub] for sub in names_raw]
    if isinstance(names_raw, list):
        return [[str(n)] for n in names_raw]
    return []


def _charger_sim(ch: Any) -> Any:
    return getattr(ch, "charger_simulation", getattr(ch, "_Charger__charger_simulation", None))


def _first_float_attr(obj: Any, names: List[str], default: float = 0.0) -> float:
    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            if value.size == 0:
                continue
            return float(value.reshape(-1)[0])
        try:
            return float(value)
        except Exception:
            continue
    return float(default)


def _safe_series_value(arr: Any, idx: int, default: float = 0.0) -> float:
    try:
        arr_np = np.asarray(arr, dtype=float)
        if 0 <= idx < len(arr_np):
            val = float(arr_np[idx])
            if np.isfinite(val):
                return val
    except Exception:
        pass
    return float(default)


def _search_arrival_soc(state: np.ndarray, ev_id: np.ndarray, arr_soc_raw: np.ndarray, idx: int, ev_name: str) -> float:
    for tau in range(idx, -1, -1):
        try:
            if str(ev_id[tau]) != ev_name:
                break
            if float(state[tau]) == 3.0:
                break
            if float(state[tau]) == 2.0:
                soc = float(arr_soc_raw[tau])
                if np.isfinite(soc) and soc >= 0.0:
                    return float(np.clip(soc, 0.0, 1.0))
        except Exception:
            break
    return 0.0


@dataclass
class BatteryMeta:
    b_idx: int
    action_idx: int
    capacity_kwh: float
    nominal_power_kw: float
    soc_now: float


@dataclass
class EVMeta:
    b_idx: int
    action_idx: int
    local_idx: int
    charge_power_kw: float
    discharge_power_kw: float
    state: np.ndarray
    dep_time: np.ndarray
    req_soc: np.ndarray
    ev_id: np.ndarray
    arrival_soc: np.ndarray
    capacity_by_name: Dict[str, float]
    soc_now: float
    current_ev_name: Optional[str]


@dataclass
class EVSegment:
    ev_idx: int
    abs_start: int
    abs_end: int
    ev_name: str
    init_soc: float
    capacity_kwh: float
    dep_constraints: Dict[int, float]


class JointUpperBoundMPC:
    def __init__(
        self,
        env: Any,
        horizon: int = 24,
        control_interval: int = 1,
        soc_low: float = 0.0,
        soc_high: float = 0.95,
        p_building_max: float = 4.6083,
        p_grid_max: float = 10.2352,
        ev_efficiency: float = 0.95,
        batt_efficiency: float = 0.95,
        w_ev_slack: float = 1.0e4,
        w_c3_slack: float = 2.5e3,
        w_c4_slack: float = 5.0e3,
        w_grid_cost: float = 10.0,
        w_cycle_batt: float = 1.0,
        w_cycle_ev: float = 0.5,
        solver: str = "OSQP",
    ):
        self.env = env
        self.raw = unwrap_to_raw_citylearn_env(env)
        self.horizon = int(horizon)
        self.control_interval = max(1, int(control_interval))
        self.soc_low = float(soc_low)
        self.soc_high = float(soc_high)
        self.p_building_max = float(p_building_max)
        self.p_grid_max = float(p_grid_max)
        self.ev_efficiency = float(ev_efficiency)
        self.batt_efficiency = float(batt_efficiency)
        self.w_ev_slack = float(w_ev_slack)
        self.w_c3_slack = float(w_c3_slack)
        self.w_c4_slack = float(w_c4_slack)
        self.w_grid_cost = float(w_grid_cost)
        self.w_cycle_batt = float(w_cycle_batt)
        self.w_cycle_ev = float(w_cycle_ev)
        self.solver = solver
        self.dt_hours = float(getattr(self.raw, "seconds_per_time_step", 3600)) / 3600.0
        self.buildings = list(getattr(self.raw, "buildings", []))
        self.nb = len(self.buildings)
        self.action_names = _normalize_action_names(getattr(self.raw, "action_names", []), self.nb)
        self.total_action_dim = int(np.prod(self.env.action_space.shape))
        self.mapping = self._build_mapping()
        self.solve_failures = 0
        self.solve_times_ms: List[float] = []
        self.status_counts: Dict[str, int] = {}
        self._cached_actions: List[np.ndarray] = []

    def _build_mapping(self) -> Dict[str, Any]:
        batteries: List[Dict[str, int]] = []
        evs: List[Dict[str, int]] = []
        wm_indices: List[int] = []
        gidx = 0
        for b_idx, names in enumerate(self.action_names):
            ev_local_idx = 0
            batt_seen = False
            for name in names:
                lname = str(name).lower()
                if lname == "electrical_storage" and not batt_seen:
                    batteries.append({"b_idx": b_idx, "action_idx": gidx})
                    batt_seen = True
                elif "electric_vehicle_storage_charger_" in lname:
                    evs.append({"b_idx": b_idx, "action_idx": gidx, "local_idx": ev_local_idx})
                    ev_local_idx += 1
                elif "washing_machine" in lname:
                    wm_indices.append(gidx)
                gidx += 1
        return {"batteries": batteries, "evs": evs, "wm_indices": wm_indices}

    def _battery_meta(self, t_now: int) -> List[BatteryMeta]:
        t_soc = 0 if t_now <= 0 else t_now - 1
        out: List[BatteryMeta] = []
        for item in self.mapping["batteries"]:
            b_idx = item["b_idx"]
            b = self.buildings[b_idx]
            es = getattr(b, "electrical_storage", getattr(b, "electricity_storage", None))
            if es is None:
                continue
            soc = _safe_series_value(getattr(es, "soc", []), t_soc, default=0.5)
            out.append(
                BatteryMeta(
                    b_idx=b_idx,
                    action_idx=item["action_idx"],
                    capacity_kwh=max(1e-6, _first_float_attr(es, ["capacity", "_EnergyStorage__capacity"], 6.4)),
                    nominal_power_kw=max(1e-6, _first_float_attr(es, ["nominal_power", "_EnergyStorage__nominal_power"], 5.0)),
                    soc_now=float(np.clip(soc, 0.0, 1.0)),
                )
            )
        return out

    def _ev_meta(self, t_now: int) -> List[EVMeta]:
        ev_by_name = _collect_ev_meta(self.raw)
        t_soc = 0 if t_now <= 0 else t_now - 1
        out: List[EVMeta] = []
        for item in self.mapping["evs"]:
            b = self.buildings[item["b_idx"]]
            chargers = getattr(b, "electric_vehicle_chargers", None) or []
            if item["local_idx"] >= len(chargers):
                continue
            ch = chargers[item["local_idx"]]
            sim = _charger_sim(ch)
            if sim is None:
                continue
            state = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
            dep_time = np.asarray(getattr(sim, "_electric_vehicle_departure_time"), dtype=float)
            req_soc = np.asarray(getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
            ev_id = np.asarray(getattr(sim, "_electric_vehicle_id"))
            arrival_soc = np.asarray(getattr(sim, "_electric_vehicle_estimated_soc_arrival"), dtype=float)
            current_ev_name: Optional[str] = None
            soc_now = 0.0
            if t_now < len(state) and float(state[t_now]) == 1.0 and t_now < len(ev_id):
                current_ev_name = str(ev_id[t_now])
                ev_obj = getattr(ch, "connected_electric_vehicle", None)
                if ev_obj is not None:
                    batt = getattr(ev_obj, "battery", None)
                    if batt is not None:
                        soc_now = _safe_series_value(getattr(batt, "soc", []), t_soc, default=0.0)
            cap_map = {}
            for name, meta in ev_by_name.items():
                cap_map[str(name)] = float(meta.get("cap_kwh") or 0.0)
            out.append(
                EVMeta(
                    b_idx=item["b_idx"],
                    action_idx=item["action_idx"],
                    local_idx=item["local_idx"],
                    charge_power_kw=max(1e-6, _first_float_attr(ch, ["max_charging_power", "_Charger__max_charging_power"], 0.0)),
                    discharge_power_kw=max(1e-6, _first_float_attr(ch, ["max_discharging_power", "_Charger__max_discharging_power"], 0.0)),
                    state=state,
                    dep_time=dep_time,
                    req_soc=req_soc,
                    ev_id=ev_id,
                    arrival_soc=arrival_soc,
                    capacity_by_name=cap_map,
                    soc_now=float(np.clip(soc_now, 0.0, 1.0)),
                    current_ev_name=current_ev_name,
                )
            )
        return out

    def _base_net_forecast(self, t_now: int) -> np.ndarray:
        base = np.zeros((self.nb, self.horizon), dtype=float)
        for b_idx, b in enumerate(self.buildings):
            nsl = np.asarray(getattr(b, "_Building__energy_to_non_shiftable_load", []), dtype=float)
            sg = np.asarray(getattr(b, "_Building__solar_generation", []), dtype=float)
            t_max = min(len(nsl), len(sg))
            for k in range(self.horizon):
                idx = t_now + k
                if idx < t_max:
                    base[b_idx, k] = float(nsl[idx] + sg[idx])
                elif t_max > 0:
                    base[b_idx, k] = float(nsl[t_max - 1] + sg[t_max - 1])
        return base

    def _price_forecast(self, t_now: int) -> np.ndarray:
        try:
            price = np.asarray(self.buildings[0].pricing.electricity_pricing, dtype=float)
        except Exception:
            return np.zeros(self.horizon, dtype=float)
        out = np.zeros(self.horizon, dtype=float)
        for k in range(self.horizon):
            idx = min(max(t_now + k, 0), len(price) - 1)
            raw_price = float(price[idx]) if len(price) > 0 else 0.0
            out[k] = max(0.0, raw_price)
        return out

    def _build_ev_segments(self, evs: List[EVMeta], t_now: int) -> List[EVSegment]:
        segments: List[EVSegment] = []
        for ev_idx, ev in enumerate(evs):
            k = 0
            while k < self.horizon:
                abs_idx = t_now + k
                if abs_idx >= len(ev.state):
                    break
                connected = float(ev.state[abs_idx]) == 1.0
                ev_name = str(ev.ev_id[abs_idx]) if abs_idx < len(ev.ev_id) else ""
                if not connected or not ev_name:
                    k += 1
                    continue
                start = k
                while k + 1 < self.horizon:
                    nxt_abs = t_now + k + 1
                    if nxt_abs >= len(ev.state):
                        break
                    same = float(ev.state[nxt_abs]) == 1.0 and str(ev.ev_id[nxt_abs]) == ev_name
                    if not same:
                        break
                    k += 1
                end = k
                if start == 0 and ev.current_ev_name == ev_name:
                    init_soc = ev.soc_now
                else:
                    init_soc = _search_arrival_soc(ev.state, ev.ev_id, ev.arrival_soc, t_now + start, ev_name)
                capacity = float(ev.capacity_by_name.get(ev_name, 0.0) or 0.0)
                dep_constraints: Dict[int, float] = {}
                for rel in range(start, end + 1):
                    abs_rel = t_now + rel
                    if abs_rel < len(ev.dep_time) and float(ev.dep_time[abs_rel]) == 0.0:
                        dep_constraints[rel - start] = float(np.clip(ev.req_soc[abs_rel], 0.0, 1.0))
                segments.append(
                    EVSegment(
                        ev_idx=ev_idx,
                        abs_start=start,
                        abs_end=end,
                        ev_name=ev_name,
                        init_soc=float(np.clip(init_soc, 0.0, 1.0)),
                        capacity_kwh=max(1e-6, capacity if capacity > 0 else 40.0),
                        dep_constraints=dep_constraints,
                    )
                )
                k += 1
        return segments

    def _fallback_action(self, t_now: int, batts: List[BatteryMeta], evs: List[EVMeta], base_net: np.ndarray) -> np.ndarray:
        action = np.zeros(self.total_action_dim, dtype=np.float32)
        for idx in self.mapping["wm_indices"]:
            action[idx] = 0.0

        district_base = float(np.sum(base_net[:, 0])) if base_net.size else 0.0
        over_grid = max(0.0, district_base - self.p_grid_max)
        batt_share = 0.0
        if batts and over_grid > 0.0:
            batt_nom_total = sum(b.nominal_power_kw for b in batts)
            batt_share = min(1.0, over_grid / max(batt_nom_total, 1e-6))

        for b in batts:
            if b.soc_now > self.soc_low + 0.05 and batt_share > 0.0:
                action[b.action_idx] = float(-min(1.0, batt_share))

        for ev in evs:
            if t_now >= len(ev.state):
                continue
            if float(ev.state[t_now]) != 1.0:
                continue
            req_soc = float(np.clip(ev.req_soc[t_now], 0.0, 1.0)) if t_now < len(ev.req_soc) else 1.0
            dep_time = float(ev.dep_time[t_now]) if t_now < len(ev.dep_time) else 999.0
            if req_soc > ev.soc_now + 0.01 and dep_time <= max(6.0, float(self.horizon)):
                action[ev.action_idx] = 1.0
        return action

    def _solve_problem(self, problem: cp.Problem) -> str:
        status = "unknown"
        solver_order = [self.solver, "SCIPY", "CLARABEL", "OSQP", "SCS"]
        seen_solvers = set()
        for solver_name in solver_order:
            if solver_name in seen_solvers:
                continue
            seen_solvers.add(solver_name)
            try:
                if solver_name == "OSQP":
                    problem.solve(
                        solver=cp.OSQP,
                        warm_start=True,
                        verbose=False,
                        max_iter=20000,
                        eps_abs=1e-4,
                        eps_rel=1e-4,
                    )
                elif solver_name == "SCIPY":
                    problem.solve(
                        solver=cp.SCIPY,
                        scipy_options={"method": "highs"},
                        verbose=False,
                    )
                elif solver_name == "SCS":
                    problem.solve(
                        solver=cp.SCS,
                        warm_start=True,
                        verbose=False,
                        max_iters=8000,
                        eps=1e-4,
                    )
                else:
                    problem.solve(
                        solver=getattr(cp, solver_name),
                        warm_start=True,
                        verbose=False,
                    )
                status = str(problem.status)
            except Exception:
                status = f"{solver_name.lower()}_error"
            if status in ("optimal", "optimal_inaccurate"):
                break
        return status

    def act(self) -> np.ndarray:
        if self._cached_actions:
            return self._cached_actions.pop(0).copy()

        raw = self.raw
        t_now = int(getattr(raw, "time_step", 0))
        batts = self._battery_meta(t_now)
        evs = self._ev_meta(t_now)
        base_net = self._base_net_forecast(t_now)
        price = self._price_forecast(t_now)
        ev_segments = self._build_ev_segments(evs, t_now)

        action = np.zeros(self.total_action_dim, dtype=np.float32)
        for idx in self.mapping["wm_indices"]:
            action[idx] = 0.0

        n_batt = len(batts)
        n_ev = len(evs)
        H = self.horizon
        batt_pos = cp.Variable((n_batt, H), nonneg=True)
        batt_neg = cp.Variable((n_batt, H), nonneg=True)
        ev_pos = cp.Variable((n_ev, H), nonneg=True)
        ev_neg = cp.Variable((n_ev, H), nonneg=True)
        c3_slack = cp.Variable((self.nb, H), nonneg=True)
        c4_slack = cp.Variable(H, nonneg=True)
        grid_import = cp.Variable(H, nonneg=True)
        constraints: List[Any] = []
        max_c3_slack = 50.0
        max_c4_slack = 100.0
        constraints += [c3_slack <= max_c3_slack, c4_slack <= max_c4_slack, grid_import <= self.p_grid_max + max_c4_slack]

        for i in range(n_batt):
            constraints += [batt_pos[i, :] <= 1.0, batt_neg[i, :] <= 1.0]
        for j in range(n_ev):
            constraints += [ev_pos[j, :] <= 1.0, ev_neg[j, :] <= 1.0]

        batt_by_bldg = {b.b_idx: i for i, b in enumerate(batts)}
        evs_by_bldg: Dict[int, List[int]] = {}
        for j, ev in enumerate(evs):
            evs_by_bldg.setdefault(ev.b_idx, []).append(j)

        batt_alpha_charge = np.array(
            [(self.dt_hours * self.batt_efficiency * b.nominal_power_kw) / b.capacity_kwh for b in batts],
            dtype=float,
        )
        batt_alpha_discharge = np.array(
            [(self.dt_hours * b.nominal_power_kw) / (self.batt_efficiency * b.capacity_kwh) for b in batts],
            dtype=float,
        )

        for i, b in enumerate(batts):
            soc = cp.Variable(H + 1)
            constraints += [soc[0] == b.soc_now, soc >= self.soc_low, soc <= self.soc_high]
            for k in range(H):
                constraints += [
                    soc[k + 1] == soc[k] + batt_alpha_charge[i] * batt_pos[i, k] - batt_alpha_discharge[i] * batt_neg[i, k]
                ]

        ev_slacks: List[Any] = []
        for j, ev in enumerate(evs):
            for k in range(H):
                abs_idx = t_now + k
                if abs_idx >= len(ev.state):
                    constraints += [ev_pos[j, k] == 0.0, ev_neg[j, k] == 0.0]
                    continue
                connected = float(ev.state[abs_idx]) == 1.0 and str(ev.ev_id[abs_idx]) != ""
                if not connected:
                    constraints += [ev_pos[j, k] == 0.0, ev_neg[j, k] == 0.0]
                if connected and float(ev.dep_time[abs_idx]) == 0.0:
                    constraints += [ev_pos[j, k] == 0.0, ev_neg[j, k] == 0.0]

        for seg in ev_segments:
            ev = evs[seg.ev_idx]
            seg_len = seg.abs_end - seg.abs_start + 1
            alpha_charge = (self.dt_hours * self.ev_efficiency * ev.charge_power_kw) / seg.capacity_kwh
            alpha_discharge = (self.dt_hours * ev.discharge_power_kw) / (self.ev_efficiency * seg.capacity_kwh)
            soc = cp.Variable(seg_len + 1)
            constraints += [soc[0] == seg.init_soc, soc >= 0.0, soc <= 1.0]
            for local in range(seg_len):
                abs_k = seg.abs_start + local
                constraints += [
                    soc[local + 1] == soc[local] + alpha_charge * ev_pos[seg.ev_idx, abs_k] - alpha_discharge * ev_neg[seg.ev_idx, abs_k]
                ]
            for local, req in seg.dep_constraints.items():
                slack = cp.Variable(nonneg=True)
                constraints += [slack <= 1.0]
                constraints += [soc[local] + slack >= req]
                ev_slacks.append(seg.capacity_kwh * slack)

        building_net_exprs: Dict[tuple, Any] = {}
        for b_idx in range(self.nb):
            for k in range(H):
                expr = float(base_net[b_idx, k])
                if b_idx in batt_by_bldg:
                    i = batt_by_bldg[b_idx]
                    expr = expr + batts[i].nominal_power_kw * (batt_pos[i, k] - batt_neg[i, k])
                for j in evs_by_bldg.get(b_idx, []):
                    expr = expr + evs[j].charge_power_kw * ev_pos[j, k] - evs[j].discharge_power_kw * ev_neg[j, k]
                building_net_exprs[(b_idx, k)] = expr
                constraints += [
                    expr <= self.p_building_max + c3_slack[b_idx, k],
                    expr >= -self.p_building_max - c3_slack[b_idx, k],
                ]

        for k in range(H):
            district = sum(building_net_exprs[(b_idx, k)] for b_idx in range(self.nb))
            constraints += [grid_import[k] >= district, grid_import[k] >= 0.0]
            constraints += [grid_import[k] <= self.p_grid_max + c4_slack[k]]

        ev_slack_expr = cp.sum(cp.hstack(ev_slacks)) if ev_slacks else cp.Constant(0.0)
        c4_slack_expr = cp.sum(c4_slack)
        c3_slack_expr = cp.sum(c3_slack)
        kpi_expr = (
            self.w_grid_cost * cp.sum(cp.multiply(price, grid_import))
            + self.w_cycle_batt * cp.sum(batt_pos + batt_neg)
            + self.w_cycle_ev * cp.sum(ev_pos + ev_neg)
        )

        t0 = time.time()
        stage_constraints = list(constraints)
        stage_statuses: List[str] = []
        stage_eps = 1e-6

        stage_1 = cp.Problem(cp.Minimize(ev_slack_expr), stage_constraints)
        status = self._solve_problem(stage_1)
        stage_statuses.append(f"ev:{status}")

        if status in ("optimal", "optimal_inaccurate"):
            ev_best = float(ev_slack_expr.value or 0.0)
            stage_constraints = stage_constraints + [ev_slack_expr <= ev_best + stage_eps]
            stage_2 = cp.Problem(cp.Minimize(c4_slack_expr), stage_constraints)
            status = self._solve_problem(stage_2)
            stage_statuses.append(f"c4:{status}")

        if status in ("optimal", "optimal_inaccurate"):
            c4_best = float(c4_slack_expr.value or 0.0)
            stage_constraints = stage_constraints + [c4_slack_expr <= c4_best + stage_eps]
            stage_3 = cp.Problem(cp.Minimize(c3_slack_expr), stage_constraints)
            status = self._solve_problem(stage_3)
            stage_statuses.append(f"c3:{status}")

        if status in ("optimal", "optimal_inaccurate"):
            c3_best = float(c3_slack_expr.value or 0.0)
            stage_constraints = stage_constraints + [c3_slack_expr <= c3_best + stage_eps]
            stage_4 = cp.Problem(cp.Minimize(kpi_expr), stage_constraints)
            status = self._solve_problem(stage_4)
            stage_statuses.append(f"kpi:{status}")

        self.solve_times_ms.append((time.time() - t0) * 1000.0)
        status_key = "|".join(stage_statuses) if stage_statuses else status
        self.status_counts[status_key] = self.status_counts.get(status_key, 0) + 1

        if status not in ("optimal", "optimal_inaccurate"):
            self.solve_failures += 1
            fallback = self._fallback_action(t_now, batts, evs, base_net)
            self._cached_actions = [fallback.copy() for _ in range(self.control_interval - 1)]
            return fallback

        hold = min(self.control_interval, H)
        planned_actions: List[np.ndarray] = []
        for k in range(hold):
            act_k = np.zeros(self.total_action_dim, dtype=np.float32)
            for idx in self.mapping["wm_indices"]:
                act_k[idx] = 0.0
            for i, b in enumerate(batts):
                act_k[b.action_idx] = float(np.clip(batt_pos.value[i, k] - batt_neg.value[i, k], -1.0, 1.0))
            for j, ev in enumerate(evs):
                act_k[ev.action_idx] = float(np.clip(ev_pos.value[j, k] - ev_neg.value[j, k], -1.0, 1.0))
            planned_actions.append(act_k)

        if not planned_actions:
            planned_actions = [action]
        self._cached_actions = [a.copy() for a in planned_actions[1:]]
        return planned_actions[0].copy()


def make_env():
    from scripts.make_env import make_base_env
    from citylearn_safe.forecast_obs_wrapper import ForecastObsWrapper
    from citylearn_safe.safety_env_v3 import CityLearnSafetyEnvV3

    base = make_base_env(central_agent=True)
    safety = CityLearnSafetyEnvV3(base)
    env = ForecastObsWrapper(safety, forecast_horizon=24)
    if os.environ.get("CITYLEARN_EV_SAUTE", "0") == "1":
        from citylearn_safe.saute_ev_wrapper import SauteEVBudgetWrapper
        env = SauteEVBudgetWrapper(env)
    return env, safety


def evaluate_mpc(args: argparse.Namespace) -> Dict[str, Any]:
    for key in list(os.environ.keys()):
        if key.startswith(("CITYLEARN_", "STEMS_", "COST_W_")):
            os.environ.pop(key)
    os.environ.update(BASE_ENV)
    if args.schema:
        os.environ["CITYLEARN_SCHEMA"] = args.schema

    env, safety = make_env()
    raw = unwrap_to_raw_citylearn_env(env)
    buildings = list(raw.buildings)
    n_b = len(buildings)
    p_bmax = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
    p_gmax = float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "10.2352"))

    mpc = JointUpperBoundMPC(
        env=env,
        horizon=args.horizon,
        control_interval=args.control_interval,
        soc_low=float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0")),
        soc_high=float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95")),
        p_building_max=p_bmax,
        p_grid_max=p_gmax,
        ev_efficiency=args.ev_efficiency,
        batt_efficiency=args.batt_efficiency,
        w_ev_slack=args.w_ev_slack,
        w_c3_slack=args.w_c3_slack,
        w_c4_slack=args.w_c4_slack,
        w_grid_cost=args.w_grid_cost,
        w_cycle_batt=args.w_cycle_batt,
        w_cycle_ev=args.w_cycle_ev,
        solver=args.solver,
    )

    obs, _ = env.reset(seed=args.seed)
    total_steps = 0
    total_reward = 0.0
    ev_departures = 0
    ev_violated_departures = 0
    ev_deficit_kwh = 0.0
    c2_violations = 0
    c2_total_checks = 0
    c3_per_building = [0] * n_b
    c3_total_per_building = [0] * n_b
    c4_violations = 0
    nec_history: List[float] = []
    nec_baseline_history: List[float] = []
    ev_tracker: Dict[Any, Dict[str, Any]] = {}
    done = False

    while not done:
        action = mpc.act()
        if args.limit_steps is not None and total_steps >= args.limit_steps:
            break

        t_now = int(getattr(raw, "time_step", 0))
        for b_idx, bld in enumerate(buildings):
            chargers = getattr(bld, "electric_vehicle_chargers", None) or []
            for ch_idx, ch in enumerate(chargers):
                key = (b_idx, ch_idx)
                sim = _charger_sim(ch)
                if sim is None:
                    continue
                try:
                    sa = np.asarray(getattr(sim, "_electric_vehicle_charger_state"), dtype=float)
                    connected = t_now < len(sa) and float(sa[t_now]) == 1.0
                    if connected:
                        ra = np.asarray(getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=float)
                        rs = float(ra[t_now]) if t_now < len(ra) else 1.0
                        ev_obj = getattr(ch, "connected_electric_vehicle", None)
                        soc = 0.0
                        if ev_obj and getattr(ev_obj, "battery", None):
                            soc_arr = np.asarray(ev_obj.battery.soc, dtype=float)
                            t_idx = max(0, t_now - 1)
                            soc = float(np.clip(soc_arr[t_idx], 0, 1)) if 0 <= t_idx < len(soc_arr) else 0.0
                        ev_tracker[key] = {"was": True, "soc": soc, "req": rs}
                    else:
                        prev = ev_tracker.get(key, {})
                        if prev.get("was", False):
                            ev_departures += 1
                            deficit = max(0.0, prev["req"] - prev["soc"])
                            if deficit > 0.01:
                                ev_violated_departures += 1
                                ev_deficit_kwh += deficit
                        ev_tracker[key] = {"was": False}
                except Exception:
                    pass

        obs, reward, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        total_steps += 1
        total_reward += float(reward)

        t_idx = max(0, int(getattr(raw, "time_step", 0)) - 1)
        district_nec = 0.0
        district_baseline = 0.0
        for b_idx, bld in enumerate(buildings):
            es = getattr(bld, "electrical_storage", None)
            if es is not None and hasattr(es, "soc") and len(es.soc) > t_idx:
                soc = float(es.soc[t_idx])
                c2_total_checks += 1
                if soc < 0.0 or soc > 0.95:
                    c2_violations += 1

            nec = getattr(bld, "net_electricity_consumption", None)
            if nec is not None and hasattr(nec, "__len__") and len(nec) > t_idx:
                p_i = float(nec[t_idx])
                district_nec += p_i
                c3_total_per_building[b_idx] += 1
                if abs(p_i) > p_bmax:
                    c3_per_building[b_idx] += 1
            try:
                nsl = getattr(bld, "_Building__energy_to_non_shiftable_load", [])
                sg = getattr(bld, "_Building__solar_generation", [])
                base_nec = 0.0
                if len(nsl) > t_idx:
                    base_nec += float(nsl[t_idx])
                if len(sg) > t_idx:
                    base_nec += float(sg[t_idx])
                district_baseline += base_nec
            except Exception:
                pass

        nec_history.append(district_nec)
        nec_baseline_history.append(district_baseline)
        if max(0.0, district_nec) > p_gmax:
            c4_violations += 1

        if total_steps % 500 == 0:
            avg_ms = float(np.mean(mpc.solve_times_ms)) if mpc.solve_times_ms else 0.0
            print(
                f"[joint-mpc] step={total_steps} reward={total_reward:.1f} "
                f"C0={ev_violated_departures}/{max(ev_departures,1)} "
                f"C3={sum(c3_per_building)}/{max(sum(c3_total_per_building),1)} "
                f"C4={c4_violations}/{total_steps} solve_ms={avg_ms:.1f}"
            )

    nec_arr = np.array(nec_history, dtype=float)
    base_arr = np.array(nec_baseline_history, dtype=float)
    agent_import = float(np.sum(np.maximum(0, nec_arr)))
    base_import = float(np.sum(np.maximum(0, base_arr)))
    kpi_consumption = agent_import / max(base_import, 1e-6)
    agent_ramp = float(np.sum(np.maximum(0, np.diff(nec_arr)))) if len(nec_arr) > 1 else 0.0
    base_ramp = float(np.sum(np.maximum(0, np.diff(base_arr)))) if len(base_arr) > 1 else 0.0
    kpi_ramping = agent_ramp / max(base_ramp, 1e-6)
    n_days = total_steps // 24
    if n_days > 0:
        agent_peaks = [float(np.max(nec_arr[d * 24:(d + 1) * 24])) for d in range(n_days)]
        base_peaks = [float(np.max(base_arr[d * 24:(d + 1) * 24])) for d in range(n_days)]
        kpi_daily_peak = float(np.mean(agent_peaks)) / max(float(np.mean(base_peaks)), 1e-6)
    else:
        kpi_daily_peak = 0.0

    c3_total = int(sum(c3_per_building))
    c3_total_checks = int(sum(c3_total_per_building))
    avg_solve_ms = float(np.mean(mpc.solve_times_ms)) if mpc.solve_times_ms else 0.0
    results = {
        "controller": "joint_mpc_upper_bound",
        "horizon": int(args.horizon),
        "control_interval": int(args.control_interval),
        "steps": int(total_steps),
        "reward": float(total_reward),
        "c0_violations": int(ev_violated_departures),
        "c0_departures": int(ev_departures),
        "c0_pct": 100.0 * ev_violated_departures / max(ev_departures, 1),
        "c0_deficit_kwh": float(ev_deficit_kwh),
        "c2_pct": 100.0 * c2_violations / max(c2_total_checks, 1),
        "c3_total_pct": 100.0 * c3_total / max(c3_total_checks, 1),
        "c3_per_building_pct": [
            100.0 * c3_per_building[i] / max(c3_total_per_building[i], 1) for i in range(n_b)
        ],
        "c4_pct": 100.0 * c4_violations / max(total_steps, 1),
        "kpi_consumption": float(kpi_consumption),
        "kpi_ramping": float(kpi_ramping),
        "kpi_daily_peak": float(kpi_daily_peak),
        "solve_failures": int(mpc.solve_failures),
        "avg_solve_ms": float(avg_solve_ms),
        "solver_status_counts": dict(sorted(mpc.status_counts.items())),
    }

    print("\n" + "=" * 72)
    print("JOINT MPC UPPER-BOUND RESULTS")
    print("=" * 72)
    print(f"C0: {results['c0_violations']}/{results['c0_departures']} = {results['c0_pct']:.2f}%")
    print(f"C2: {results['c2_pct']:.2f}%")
    print(f"C3: {c3_total}/{c3_total_checks} = {results['c3_total_pct']:.2f}%")
    print(f"C4: {c4_violations}/{total_steps} = {results['c4_pct']:.2f}%")
    print(f"Reward: {results['reward']:.2f}")
    print(f"KPI consumption: {results['kpi_consumption']:.3f}")
    print(f"KPI ramping: {results['kpi_ramping']:.3f}")
    print(f"KPI daily peak: {results['kpi_daily_peak']:.3f}")
    print(f"Solve failures: {results['solve_failures']}")
    print(f"Avg solve time: {results['avg_solve_ms']:.1f} ms")
    print(f"Statuses: {results['solver_status_counts']}")
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Saved JSON: {out_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a joint rolling-horizon MPC upper bound.")
    parser.add_argument("--schema", type=str, default=BASE_ENV["CITYLEARN_SCHEMA"])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--control-interval", type=int, default=1)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ev-efficiency", type=float, default=0.95)
    parser.add_argument("--batt-efficiency", type=float, default=0.95)
    parser.add_argument("--w-ev-slack", type=float, default=1.0e4)
    parser.add_argument("--w-c3-slack", type=float, default=2.5e3)
    parser.add_argument("--w-c4-slack", type=float, default=5.0e3)
    parser.add_argument("--w-grid-cost", type=float, default=10.0)
    parser.add_argument("--w-cycle-batt", type=float, default=1.0)
    parser.add_argument("--w-cycle-ev", type=float, default=0.5)
    parser.add_argument("--solver", type=str, default="OSQP")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_mpc(parse_args())
