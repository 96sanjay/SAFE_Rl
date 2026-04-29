"""Policy-side continuous action masking utilities for PPO.

This module keeps the existing CityLearn-specific feasible-set logic used by
ActionMaskWrapper, but moves the action transform into the policy/update path.
The policy samples a latent Gaussian action ``u``. We squash it with tanh to
``z in [-1, 1]`` and then map ``z`` into the current safe interval
``[safe_min(s), safe_max(s)]``.
"""
from __future__ import annotations

import os
import re
from typing import Any, Tuple

import numpy as np
import torch


_EV_RE = re.compile(
    r"^electric_vehicle_storage_charger_(?P<suffix>.+)$", re.IGNORECASE,
)


def _unwrap_citylearn(env: Any):
    """Walk the wrapper chain to find the CityLearn env with .buildings."""
    inner = env
    for _ in range(20):
        if hasattr(inner, 'buildings') and len(getattr(inner, 'buildings', [])) > 0:
            return inner
        inner = getattr(
            inner,
            '_env',
            getattr(inner, 'env', getattr(inner, 'base', None)),
        )
        if inner is None:
            break
    return None


class CityLearnActionBoundsProvider:
    """Compute state-dependent safe bounds using the current CityLearn state."""

    def __init__(self, env: Any):
        self._city = _unwrap_citylearn(env)
        if self._city is None:
            raise RuntimeError(
                "[PolicyActionMask] Cannot find CityLearn env with .buildings in wrapper chain",
            )

        self._p_bmax = float(os.environ.get("CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
        self._soc_low = float(os.environ.get("CITYLEARN_STEMS_SOC_LOW", "0.0"))
        self._soc_high = float(os.environ.get("CITYLEARN_STEMS_SOC_HIGH", "0.95"))
        self._p_gmax = float(os.environ.get("CITYLEARN_STEMS_P_GRID_MAX", "10.2352"))

        self._n_buildings = len(self._city.buildings)

        action_names = self._city.action_names
        if isinstance(action_names, list) and len(action_names) == 1 and isinstance(action_names[0], list):
            action_names_flat = action_names[0]
        else:
            action_names_flat = list(action_names)
        self._n_actions = len(action_names_flat)

        self._building_batt_act: dict[int, int] = {}
        self._building_ev_act: dict[int, int] = {}
        self._passthrough_indices: list[int] = []

        b_idx = 0
        batt_count_for_building = {}
        for act_idx, name in enumerate(action_names_flat):
            name_lower = name.lower()
            if _EV_RE.match(name):
                suffix = _EV_RE.match(name).group('suffix')
                parts = suffix.split('_')
                ev_b_idx = int(parts[0]) - 1
                self._building_ev_act[ev_b_idx] = act_idx
            elif 'washing_machine' in name_lower:
                self._passthrough_indices.append(act_idx)
            elif name_lower == 'electrical_storage':
                while b_idx in batt_count_for_building:
                    b_idx += 1
                self._building_batt_act[b_idx] = act_idx
                batt_count_for_building[b_idx] = True
                b_idx += 1
            else:
                self._passthrough_indices.append(act_idx)

        self._batt_powers: dict[int, float] = {}
        self._ev_max_charge: dict[int, float] = {}
        self._ev_min_charge: dict[int, float] = {}
        self._ev_max_discharge: dict[int, float] = {}
        self._ev_min_discharge: dict[int, float] = {}

        for bi, b in enumerate(self._city.buildings):
            try:
                es = b.electrical_storage
                self._batt_powers[bi] = float(getattr(es, 'nominal_power', 5.0) or 5.0)
            except Exception:
                self._batt_powers[bi] = 5.0

            chargers = getattr(b, 'electric_vehicle_chargers', None) or getattr(b, 'chargers', [])
            if chargers:
                c = chargers[0]
                try:
                    self._ev_max_charge[bi] = float(getattr(c, 'max_charging_power', 0) or 0)
                    self._ev_min_charge[bi] = float(getattr(c, 'min_charging_power', 0) or 0)
                    self._ev_max_discharge[bi] = float(getattr(c, 'max_discharging_power', 0) or 0)
                    self._ev_min_discharge[bi] = float(getattr(c, 'min_discharging_power', 0) or 0)
                except Exception:
                    self._ev_max_charge[bi] = 0.0
                    self._ev_min_charge[bi] = 0.0
                    self._ev_max_discharge[bi] = 0.0
                    self._ev_min_discharge[bi] = 0.0

        self._ev_discharge_margin = float(os.environ.get("CITYLEARN_EV_CLAMP_MARGIN", "0.1"))
        self._ev_departure_block_hours = float(os.environ.get("POLICY_MASK_EV_DEPARTURE_BLOCK_HOURS", "2.0"))
        self._ev_enforce_min_charge = os.environ.get("POLICY_MASK_EV_ENFORCE_MIN_CHARGE", "1") == "1"
        self._ev_departure_reserve_margin = float(
            os.environ.get("POLICY_MASK_EV_RESERVE_MARGIN", "0.02"),
        )
        self._ev_step_hours = float(os.environ.get("POLICY_MASK_EV_STEP_HOURS", "1.0"))
        self._ev_force_only_when_critical = os.environ.get(
            "POLICY_MASK_EV_FORCE_ONLY_WHEN_CRITICAL",
            "1",
        ) == "1"
        self._ev_force_slack_steps = float(
            os.environ.get("POLICY_MASK_EV_FORCE_SLACK_STEPS", "4.0"),
        )
        self._ev_force_rate_scale = float(
            os.environ.get("POLICY_MASK_EV_FORCE_RATE_SCALE", "1.0"),
        )
        self._batt_step_hours = float(os.environ.get("POLICY_MASK_BATT_STEP_HOURS", "1.0"))
        self._c3_soft_enabled = os.environ.get("POLICY_MASK_C3_SOFT", "1") == "1"
        self._c4_soft_enabled = os.environ.get("POLICY_MASK_C4_SOFT", "1") == "1"
        self._c3_soft_margin = float(os.environ.get("POLICY_MASK_C3_SOFT_MARGIN", "0.85"))
        self._c4_soft_margin = float(os.environ.get("POLICY_MASK_C4_SOFT_MARGIN", "0.85"))
        self._soft_min_scale = float(os.environ.get("POLICY_MASK_SOFT_MIN_SCALE", "0.1"))

    def _is_ev_connected(self, b_idx: int) -> bool:
        if b_idx not in self._building_ev_act:
            return False
        try:
            b = self._city.buildings[b_idx]
            chargers = getattr(b, 'electric_vehicle_chargers', None) or getattr(b, 'chargers', [])
            if not chargers:
                return False
            ch = chargers[0]
            sim = getattr(ch, 'charger_simulation', getattr(ch, '_Charger__charger_simulation', None))
            if sim is None:
                return False
            state_arr = getattr(sim, '_electric_vehicle_charger_state', None)
            if state_arr is None:
                return False
            t = int(getattr(self._city, 'time_step', 0))
            return t < len(state_arr) and float(state_arr[t]) == 1.0
        except Exception:
            return False

    def _ev_state(self, b_idx: int):
        if not self._is_ev_connected(b_idx):
            return None
        try:
            b = self._city.buildings[b_idx]
            chargers = getattr(b, 'electric_vehicle_chargers', None) or getattr(b, 'chargers', [])
            if not chargers:
                return None
            ch = chargers[0]
            ev = getattr(ch, 'connected_electric_vehicle', None)
            if ev is None:
                return None
            bt = getattr(ev, 'battery', None)
            if bt is None:
                return None
            t = int(getattr(self._city, 'time_step', 0))
            t_idx = max(0, t - 1)
            soc_arr = getattr(bt, 'soc', None)
            if soc_arr is None or not hasattr(soc_arr, '__len__') or t_idx >= len(soc_arr):
                return None
            soc_now = float(np.clip(float(soc_arr[t_idx]), 0.0, 1.0))
            cap_ev = float(getattr(bt, 'capacity', 60.0) or 60.0)
            eta_raw = getattr(bt, 'charging_efficiency', None)
            if eta_raw is not None and float(eta_raw) > 0:
                eta_ch = float(eta_raw)
            else:
                rte = float(getattr(bt, 'round_trip_efficiency', 0.9025) or 0.9025)
                eta_ch = float(np.sqrt(max(rte, 0.01)))
            sim = getattr(ch, 'charger_simulation', getattr(ch, '_Charger__charger_simulation', None))
            if sim is None:
                return None
            req_arr = getattr(sim, '_electric_vehicle_required_soc_departure', None)
            dep_arr = getattr(sim, '_electric_vehicle_departure_time', None)
            if req_arr is None or dep_arr is None or t >= len(req_arr) or t >= len(dep_arr):
                return None
            soc_req = float(req_arr[t])
            t_dep = float(dep_arr[t])
            return soc_now, soc_req, t_dep, cap_ev, eta_ch
        except Exception:
            return None

    def _ev_min_charge_kw(self, b_idx: int, target_soc: float | None = None) -> float:
        state = self._ev_state(b_idx)
        if state is None:
            return 0.0
        soc_now, soc_req, t_dep, cap_ev, eta_ch = state
        if target_soc is None:
            target_soc = soc_req
        target_soc = min(1.0, max(0.0, float(target_soc)))
        if t_dep <= 0 or soc_now >= target_soc - 0.01:
            return 0.0
        p_ev_max = self._ev_max_charge.get(b_idx, 0.0)
        if p_ev_max <= 0:
            return 0.0
        energy_deficit = max(0.0, target_soc - soc_now) * cap_ev
        input_deficit = energy_deficit / max(eta_ch, 0.01)
        return min(input_deficit / max(t_dep, 1e-6), p_ev_max)

    def _battery_charge_discharge_etas(self, b_idx: int) -> tuple[float, float]:
        try:
            es = self._city.buildings[b_idx].electrical_storage
            charge_eta = getattr(es, 'charging_efficiency', None)
            discharge_eta = getattr(es, 'discharging_efficiency', None)
            eff = getattr(es, 'efficiency', None)
            rte = getattr(es, 'round_trip_efficiency', None)

            if charge_eta is None or float(charge_eta) <= 0.0:
                if eff is not None and float(eff) > 0.0:
                    charge_eta = float(eff)
                elif rte is not None and float(rte) > 0.0:
                    charge_eta = float(np.sqrt(max(float(rte), 1e-6)))
                else:
                    charge_eta = 0.95
            else:
                charge_eta = float(charge_eta)

            if discharge_eta is None or float(discharge_eta) <= 0.0:
                if eff is not None and float(eff) > 0.0:
                    discharge_eta = float(eff)
                elif rte is not None and float(rte) > 0.0:
                    discharge_eta = float(np.sqrt(max(float(rte), 1e-6)))
                else:
                    discharge_eta = 0.95
            else:
                discharge_eta = float(discharge_eta)
        except Exception:
            charge_eta = 0.95
            discharge_eta = 0.95

        return max(charge_eta, 1e-6), max(discharge_eta, 1e-6)

    def _battery_action_caps_kw(self, b_idx: int, soc: float) -> tuple[float, float]:
        p_batt = self._batt_powers.get(b_idx, 0.0)
        if p_batt <= 0.0:
            return 0.0, 0.0

        cap = 6.4
        try:
            es = self._city.buildings[b_idx].electrical_storage
            cap = float(getattr(es, 'capacity', 6.4) or 6.4)
        except Exception:
            pass
        if cap <= 0.0:
            return 0.0, 0.0

        charge_eta, discharge_eta = self._battery_charge_discharge_etas(b_idx)
        dt = max(self._batt_step_hours, 1e-6)

        remaining_charge_soc = max(0.0, self._soc_high - soc)
        remaining_discharge_soc = max(0.0, soc - self._soc_low)

        max_charge_action = (remaining_charge_soc * cap) / max(p_batt * dt * charge_eta, 1e-9)
        max_discharge_action = (remaining_discharge_soc * cap * discharge_eta) / max(p_batt * dt, 1e-9)

        charge_cap_kw = max(0.0, min(p_batt, max_charge_action * p_batt))
        discharge_cap_kw = max(0.0, min(p_batt, max_discharge_action * p_batt))
        return charge_cap_kw, discharge_cap_kw

    def _ev_protected_charge_kw(self, b_idx: int, ev_state) -> float:
        if ev_state is None:
            return 0.0
        soc_now, soc_req, t_dep, cap_ev, eta_ch = ev_state
        target_soc = min(1.0, soc_req + self._ev_departure_reserve_margin)
        p_ev_max = self._ev_max_charge.get(b_idx, 0.0)
        if p_ev_max <= 0.0:
            return 0.0
        required_input = max(0.0, target_soc - soc_now) * cap_ev / max(eta_ch, 0.01)
        if required_input <= 0.0 or t_dep <= 0.0:
            return 0.0

        if not self._ev_force_only_when_critical:
            return self._ev_min_charge_kw(b_idx, target_soc=target_soc)

        dt = max(self._ev_step_hours, 1e-6)
        future_hours = max(0.0, t_dep - dt)
        max_future_input = p_ev_max * future_hours
        critical_input_now = max(0.0, required_input - max_future_input)
        critical_charge_kw = min(p_ev_max, critical_input_now / dt)

        nominal_charge_kw = min(p_ev_max, (required_input / max(t_dep, dt)) * self._ev_force_rate_scale)
        slack_input = max_future_input - required_input
        slack_steps = slack_input / max(p_ev_max * dt, 1e-9)

        if self._ev_force_slack_steps <= 1e-6:
            urgency = 1.0 if slack_steps <= 0.0 else 0.0
        else:
            urgency = float(np.clip(1.0 - (slack_steps / self._ev_force_slack_steps), 0.0, 1.0))

        planned_charge_kw = urgency * nominal_charge_kw
        return min(p_ev_max, max(critical_charge_kw, planned_charge_kw))

    def _ev_surplus_discharge_kw(self, b_idx: int, ev_state) -> float:
        if ev_state is None:
            return 0.0
        soc_now, soc_req, t_dep, cap_ev, eta_ch = ev_state
        p_ev_max = self._ev_max_charge.get(b_idx, 0.0)
        p_ev_dis = self._ev_max_discharge.get(b_idx, 0.0)
        if p_ev_max <= 0.0 or p_ev_dis <= 0.0:
            return 0.0
        if t_dep <= self._ev_departure_block_hours:
            return 0.0
        target_soc = min(1.0, soc_req + self._ev_departure_reserve_margin)
        required_input = max(0.0, target_soc - soc_now) * cap_ev / max(eta_ch, 0.01)
        remaining_hours = max(0.0, t_dep - self._ev_step_hours)
        recoverable_input = p_ev_max * remaining_hours
        surplus_input = max(0.0, recoverable_input - required_input)
        return min(p_ev_dis, surplus_input / max(self._ev_step_hours, 1e-6))

    def _soft_scale(self, value: float, limit: float, margin: float) -> float:
        if limit <= 1e-9:
            return 1.0
        margin = min(0.999, max(0.0, margin))
        start = margin * limit
        if value <= start:
            return 1.0
        if value >= limit:
            return self._soft_min_scale
        frac = (value - start) / max(limit - start, 1e-9)
        return max(self._soft_min_scale, 1.0 - frac)

    def _get_exogenous_nec(self) -> list[float]:
        city = self._city
        t_idx = max(0, int(getattr(city, 'time_step', 0)) - 1)
        exo_nec = []
        for b in city.buildings:
            nsl = getattr(b, '_Building__energy_to_non_shiftable_load', [])
            sg = getattr(b, '_Building__solar_generation', [])
            val = 0.0
            try:
                if len(nsl) > t_idx:
                    val += float(nsl[t_idx])
                if len(sg) > t_idx:
                    val += float(sg[t_idx])
            except Exception:
                pass
            exo_nec.append(val)
        return exo_nec

    def _get_battery_socs(self) -> list[float]:
        city = self._city
        socs = []
        t_idx = max(0, int(getattr(city, 'time_step', 0)) - 1)
        for b in city.buildings:
            try:
                es = b.electrical_storage
                soc_arr = getattr(es, 'soc', None)
                if soc_arr is not None and hasattr(soc_arr, '__len__') and len(soc_arr) > t_idx:
                    socs.append(float(soc_arr[t_idx]))
                else:
                    socs.append(0.5)
            except Exception:
                socs.append(0.5)
        return socs

    def _compute_safe_bounds(
        self,
        exo_nec: list[float],
        socs: list[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        safe_min = np.full(self._n_actions, -1.0, dtype=np.float32)
        safe_max = np.full(self._n_actions, 1.0, dtype=np.float32)
        total_import = sum(max(0.0, float(v)) for v in exo_nec)
        batt_charge_caps_raw: dict[int, float] = {}
        batt_discharge_caps_raw: dict[int, float] = {}
        ev_protected_charge: dict[int, float] = {}
        ev_optional_charge_raw: dict[int, float] = {}
        ev_surplus_discharge_raw: dict[int, float] = {}

        for b_idx in range(self._n_buildings):
            if b_idx >= len(exo_nec):
                continue
            if b_idx in self._building_batt_act:
                soc = socs[b_idx] if b_idx < len(socs) else 0.5
                batt_charge_cap_kw, batt_discharge_cap_kw = self._battery_action_caps_kw(b_idx, soc)
                batt_charge_caps_raw[b_idx] = batt_charge_cap_kw
                batt_discharge_caps_raw[b_idx] = batt_discharge_cap_kw
            if b_idx in self._ev_max_charge and b_idx in self._building_ev_act:
                ev_state = self._ev_state(b_idx)
                if ev_state is None:
                    ev_protected_charge[b_idx] = 0.0
                    ev_optional_charge_raw[b_idx] = 0.0
                    ev_surplus_discharge_raw[b_idx] = 0.0
                else:
                    protected = min(
                        self._ev_max_charge[b_idx],
                        max(0.0, self._ev_protected_charge_kw(b_idx, ev_state)),
                    )
                    ev_protected_charge[b_idx] = protected
                    ev_optional_charge_raw[b_idx] = max(0.0, self._ev_max_charge[b_idx] - protected)
                    ev_surplus_discharge_raw[b_idx] = max(0.0, self._ev_surplus_discharge_kw(b_idx, ev_state))

        total_protected_charge = sum(ev_protected_charge.values())
        total_optional_charge = sum(batt_charge_caps_raw.values()) + sum(ev_optional_charge_raw.values())
        grid_charge_scale = 1.0
        if self._c4_soft_enabled:
            grid_charge_scale = self._soft_scale(
                total_import + total_protected_charge + total_optional_charge,
                self._p_gmax,
                self._c4_soft_margin,
            )

        for b_idx in range(self._n_buildings):
            if b_idx >= len(exo_nec):
                continue
            exo = float(exo_nec[b_idx])
            c3_charge_scale = 1.0
            c3_discharge_scale = 1.0
            if self._c3_soft_enabled:
                predicted_import = (
                    max(0.0, exo)
                    + ev_protected_charge.get(b_idx, 0.0)
                    + batt_charge_caps_raw.get(b_idx, 0.0)
                    + ev_optional_charge_raw.get(b_idx, 0.0)
                )
                c3_charge_scale = self._soft_scale(predicted_import, self._p_bmax, self._c3_soft_margin)
                if exo < 0.0:
                    predicted_export = abs(exo) + batt_discharge_caps_raw.get(b_idx, 0.0) + ev_surplus_discharge_raw.get(b_idx, 0.0)
                    c3_discharge_scale = self._soft_scale(predicted_export, self._p_bmax, self._c3_soft_margin)

            has_batt = b_idx in self._building_batt_act
            if has_batt:
                batt_act_idx = self._building_batt_act[b_idx]
                p_batt = self._batt_powers[b_idx]
                batt_charge_cap_kw = batt_charge_caps_raw.get(b_idx, 0.0)
                batt_discharge_cap_kw = batt_discharge_caps_raw.get(b_idx, 0.0)
            else:
                batt_act_idx = None
                p_batt = 0.0
                batt_charge_cap_kw = 0.0
                batt_discharge_cap_kw = 0.0
            batt_charge_cap_kw *= c3_charge_scale * grid_charge_scale
            batt_discharge_cap_kw *= c3_discharge_scale

            has_ev = b_idx in self._ev_max_charge and b_idx in self._building_ev_act
            if has_ev:
                ev_max_ch = self._ev_max_charge[b_idx]
                ev_min_ch = self._ev_min_charge[b_idx]
                ev_max_dis = self._ev_max_discharge[b_idx]
                ev_min_dis = self._ev_min_discharge[b_idx]
                ev_act_idx = self._building_ev_act[b_idx]

                ev_state = self._ev_state(b_idx)
                ev_connected = ev_state is not None
                if not ev_connected:
                    safe_min[ev_act_idx] = 0.0
                    safe_max[ev_act_idx] = 0.0
                    if has_batt:
                        safe_min[batt_act_idx] = -min(1.0, batt_discharge_cap_kw / max(p_batt, 1e-6))
                        safe_max[batt_act_idx] = min(1.0, batt_charge_cap_kw / max(p_batt, 1e-6))
                    continue

                protected_charge_kw = self._ev_protected_charge_kw(b_idx, ev_state)
                protected_charge_kw = min(ev_max_ch, max(0.0, protected_charge_kw))

                ev_surplus_dis_kw = ev_surplus_discharge_raw.get(b_idx, 0.0)
                ev_discharge_cap_kw = min(ev_max_dis, ev_surplus_dis_kw) if ev_max_dis > 0 else 0.0
                ev_discharge_cap_kw *= c3_discharge_scale
                if 0.0 < ev_discharge_cap_kw < ev_min_dis:
                    ev_discharge_cap_kw = 0.0

                optional_ev_charge = ev_optional_charge_raw.get(b_idx, 0.0) * c3_charge_scale * grid_charge_scale
                ev_charge_cap_kw = min(ev_max_ch, protected_charge_kw + optional_ev_charge)
                if 0.0 < ev_charge_cap_kw < ev_min_ch:
                    ev_charge_cap_kw = ev_min_ch if ev_min_ch <= ev_max_ch else 0.0

                if self._ev_enforce_min_charge and protected_charge_kw > 0.0:
                    if protected_charge_kw < ev_min_ch:
                        if ev_min_ch > 0.0 and ev_min_ch <= ev_max_ch:
                            safe_min[ev_act_idx] = min(1.0, ev_min_ch / max(ev_max_ch, 1e-6))
                            safe_max[ev_act_idx] = min(1.0, ev_charge_cap_kw / max(ev_max_ch, 1e-6))
                        else:
                            safe_min[ev_act_idx] = 0.0
                            safe_max[ev_act_idx] = 0.0
                    else:
                        safe_min[ev_act_idx] = min(1.0, protected_charge_kw / max(ev_max_ch, 1e-6))
                        safe_max[ev_act_idx] = min(1.0, ev_charge_cap_kw / max(ev_max_ch, 1e-6))
                else:
                    safe_min[ev_act_idx] = (
                        -min(1.0, ev_discharge_cap_kw / max(ev_max_dis, 1e-6))
                        if ev_discharge_cap_kw > 0.0 else 0.0
                    )
                    safe_max[ev_act_idx] = (
                        min(1.0, ev_charge_cap_kw / max(ev_max_ch, 1e-6))
                        if ev_max_ch >= ev_min_ch and ev_max_ch > 0.0 else 0.0
                    )
                if has_batt:
                    safe_min[batt_act_idx] = -min(1.0, batt_discharge_cap_kw / max(p_batt, 1e-6))
                    safe_max[batt_act_idx] = min(1.0, batt_charge_cap_kw / max(p_batt, 1e-6))
            elif has_batt:
                safe_min[batt_act_idx] = -min(1.0, batt_discharge_cap_kw / max(p_batt, 1e-6))
                safe_max[batt_act_idx] = min(1.0, batt_charge_cap_kw / max(p_batt, 1e-6))

        for idx in self._passthrough_indices:
            safe_min[idx] = -1.0
            safe_max[idx] = 1.0

        safe_max = np.maximum(safe_max, safe_min)
        return safe_min, safe_max

    def current_safe_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        exo_nec = self._get_exogenous_nec()
        socs = self._get_battery_socs()
        return self._compute_safe_bounds(exo_nec, socs)


def masked_action_from_pretanh(
    pre_tanh: torch.Tensor,
    safe_min: torch.Tensor,
    safe_max: torch.Tensor,
) -> torch.Tensor:
    """Map latent Gaussian samples into the current safe action interval."""
    z = torch.tanh(pre_tanh)
    out = torch.zeros_like(z)
    eps = 1e-6
    collapsed = (safe_max - safe_min) <= eps
    pos_only = (safe_min >= 0.0) & (safe_max > safe_min + eps)
    neg_only = (safe_max <= 0.0) & (safe_max > safe_min + eps)
    cross_zero = ~(collapsed | pos_only | neg_only)

    affine = safe_min + 0.5 * (z + 1.0) * (safe_max - safe_min)
    out = torch.where(pos_only | neg_only, affine, out)

    neg = z < 0
    pos = z > 0
    cross_val = torch.zeros_like(z)
    cross_val = torch.where(neg, z * torch.abs(torch.clamp_max(safe_min, 0.0)), cross_val)
    cross_val = torch.where(pos, z * torch.clamp_min(safe_max, 0.0), cross_val)
    out = torch.where(cross_zero, cross_val, out)
    out = torch.where(collapsed, safe_min, out)
    return out


def masked_log_prob_from_action(
    dist: torch.distributions.Normal,
    action: torch.Tensor,
    safe_min: torch.Tensor,
    safe_max: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute log-prob of masked physical action under latent Gaussian policy."""
    active = (safe_max - safe_min) > eps
    pos_only = (safe_min >= 0.0) & active
    neg_only = (safe_max <= 0.0) & active
    same_sign = pos_only | neg_only
    cross_zero = active & ~same_sign

    z = torch.zeros_like(action)

    same_span = torch.clamp(safe_max - safe_min, min=eps)
    z_same = 2.0 * (action - safe_min) / same_span - 1.0
    z = torch.where(same_sign, z_same, z)

    scale_neg = torch.clamp(torch.abs(torch.clamp_max(safe_min, 0.0)), min=eps)
    scale_pos = torch.clamp(torch.clamp_min(safe_max, 0.0), min=eps)
    neg = action < 0
    pos = action > 0
    z_cross = torch.zeros_like(action)
    z_cross = torch.where(neg & cross_zero, action / scale_neg, z_cross)
    z_cross = torch.where(pos & cross_zero, action / scale_pos, z_cross)
    z = torch.where(cross_zero, z_cross, z)
    z = torch.clamp(z, -1.0 + eps, 1.0 - eps)

    pre_tanh = 0.5 * (torch.log1p(z) - torch.log1p(-z))
    logp_u = dist.log_prob(pre_tanh)
    log_det_tanh = torch.log(torch.clamp(1.0 - z.pow(2), min=eps))

    log_det_mask = torch.zeros_like(action)
    log_det_mask = torch.where(same_sign, torch.log(0.5 * same_span), log_det_mask)

    cross_scale = torch.ones_like(action)
    cross_scale = torch.where(neg & cross_zero, scale_neg, cross_scale)
    cross_scale = torch.where(pos & cross_zero, scale_pos, cross_scale)
    log_det_mask = torch.where(cross_zero, torch.log(torch.clamp(cross_scale, min=eps)), log_det_mask)

    per_dim = logp_u - log_det_tanh - log_det_mask
    per_dim = torch.where(active, per_dim, torch.zeros_like(per_dim))
    return per_dim.sum(dim=-1)
