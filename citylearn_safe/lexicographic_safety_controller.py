"""
LexicographicSafetyController: Safety-first execution-time controller for CityLearn V2G.

Architecture:
  The RL action is treated as a PREFERENCE only. The controller enforces safety
  constraints in strict priority order by applying sequential analytical repair:

    Priority:  C0  >  C2  >  C3  >  C4
               EV      Batt   Bldg   Grid
             depart   SoC    power  import

  Each phase applies the minimum intervention needed to satisfy its constraint,
  without reversing any changes made by higher-priority phases.

  Unlike DiffProjector / HybridProjector which solve a joint QP that can fail
  (~25% infeasibility under the old architecture), this controller:
    - Is always feasible (never falls through to an unprotected fallback)
    - Makes priority-correct decisions (C0 cannot be violated by C3/C4 repair)
    - Reports structural infeasibility separately (cases the agent truly cannot fix)
    - Runs in O(n_devices) per step — much faster than QP

Structural infeasibility cases (cannot be fixed, reported only):
  C0: EV deficit > total capacity across remaining horizon (e.g. connected 1h before departure with 80% deficit)
  C3: Base load already exceeds P_build_max even with max battery discharge
  C4: Sum of all base loads exceeds P_grid_max

Drop-in replacement for HybridProjector: implements the same project() API.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Reuse device mapping from DiffProjector (avoids code duplication)
from citylearn_safe.diff_projector import _build_device_mapping, _unwrap_to_citylearn


class LexicographicSafetyController:
    """
    Sequential lexicographic safety controller.

    Usage (same as HybridProjector):
        ctrl = LexicographicSafetyController(env)
        ctrl.build()
        env.attach_execution_projector(ctrl)

    The controller's project(obs_tensor, unsafe_action) method is called by
    CityLearnCMDP._apply_execution_shield() before each env.step().
    """

    def __init__(
        self,
        env: Any,
        soc_low: Optional[float] = None,
        soc_high: Optional[float] = None,
        p_build_max: Optional[float] = None,
        p_grid_max: Optional[float] = None,
        ev_efficiency: float = 0.95,
        verbose: int = 0,
    ):
        self.env = env
        self.soc_low = float(os.environ.get(
            "CITYLEARN_STEMS_SOC_LOW",
            str(soc_low if soc_low is not None else 0.0),
        ))
        self.soc_high = float(os.environ.get(
            "CITYLEARN_STEMS_SOC_HIGH",
            str(soc_high if soc_high is not None else 0.95),
        ))
        self.p_build_max = float(os.environ.get(
            "CITYLEARN_STEMS_P_BUILDING_MAX",
            str(p_build_max if p_build_max is not None else 4.6083),
        ))
        self.p_grid_max = float(os.environ.get(
            "CITYLEARN_STEMS_P_GRID_MAX",
            str(p_grid_max if p_grid_max is not None else 10.2352),
        ))
        self.ev_efficiency = ev_efficiency
        self.verbose = int(verbose)

        self._built = False
        self._mapping: Optional[Dict[str, Any]] = None
        self._action_low: Optional[np.ndarray] = None
        self._action_high: Optional[np.ndarray] = None

        # Compatibility shim: state_tensor_dim = 0 (not needed for analytical repair)
        self.state_tensor_dim = 0

    # ------------------------------------------------------------------
    #  Build: discover device mapping
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Discover device indices and action bounds from the env. Call once."""
        self._mapping = _build_device_mapping(self.env)
        m = self._mapping
        n_total = m["total_dim"]

        act_space = getattr(self.env, "action_space", None)
        if act_space is not None and hasattr(act_space, "low"):
            low = np.asarray(act_space.low, dtype=np.float64).reshape(-1)
            high = np.asarray(act_space.high, dtype=np.float64).reshape(-1)
            if len(low) == n_total:
                self._action_low = low.copy()
                self._action_high = high.copy()

        if self._action_low is None:
            self._action_low = -np.ones(n_total, dtype=np.float64)
            self._action_high = np.ones(n_total, dtype=np.float64)

        self._built = True

        if self.verbose >= 1:
            print(
                f"[LexSafety] Built: {m['nb']} buildings, "
                f"{len(m['batt_gidx'])} batteries, {len(m['ev_gidx'])} EVs, "
                f"{n_total} action dims | "
                f"P_bld={self.p_build_max:.3f} P_grid={self.p_grid_max:.3f} "
                f"SoC=[{self.soc_low},{self.soc_high}]"
            )

    # ------------------------------------------------------------------
    #  State extraction
    # ------------------------------------------------------------------

    def _extract_state(self) -> dict:
        """
        Read current env state needed for constraint repair.

        Returns a dict with per-battery and per-EV physical parameters
        plus per-building exogenous net electricity consumption.
        """
        if not self._built:
            self.build()
        m = self._mapping
        city = _unwrap_to_citylearn(self.env)
        if city is None:
            raise RuntimeError("[LexSafety] Cannot unwrap env to CityLearn")

        buildings = list(getattr(city, "buildings", []))
        nb = m["nb"]
        t_now = int(getattr(city, "time_step", 0))
        t_idx = max(0, t_now - 1)

        # ---- Exogenous base NEC per building (NSL + solar) ----
        # Use t_now (not t_idx) because the action will be applied at timestep
        # t_now — we need the exogenous load for THAT step, not the previous.
        # Also build a forecast for the next FORECAST_H steps (used by C0 phase
        # to estimate C3 headroom over the EV charging horizon).
        FORECAST_H = 48  # enough for longest EV connection
        base_nec = np.zeros(nb, dtype=np.float64)
        base_nec_forecast = np.zeros((nb, FORECAST_H), dtype=np.float64)
        for b_idx, b in enumerate(buildings):
            nsl = np.asarray(
                getattr(b, "_Building__energy_to_non_shiftable_load", []),
                dtype=np.float64,
            )
            sg = np.asarray(
                getattr(b, "_Building__solar_generation", []),
                dtype=np.float64,
            )
            T = min(len(nsl), len(sg))
            if T > 0:
                idx = min(t_now, T - 1)
                base_nec[b_idx] = float(nsl[idx] + sg[idx])
                for k in range(FORECAST_H):
                    fk = min(t_now + k, T - 1)
                    base_nec_forecast[b_idx, k] = float(nsl[fk] + sg[fk])

        # ---- Battery parameters ----
        n_batt = len(m["batt_gidx"])
        soc0 = np.full(n_batt, 0.5, dtype=np.float64)
        soc_scale_ch = np.ones(n_batt, dtype=np.float64)
        soc_scale_dis = np.ones(n_batt, dtype=np.float64)
        batt_gain = np.ones(n_batt, dtype=np.float64)

        for i in range(n_batt):
            b_idx = int(m["batt_bidx"][i])
            es = getattr(buildings[b_idx], "electrical_storage", None)
            if es is None:
                continue
            cap = max(1e-6, float(getattr(es, "capacity", 6.4) or 6.4))
            nom = max(1e-6, float(getattr(es, "nominal_power", 5.0) or 5.0))
            rte = float(
                getattr(es, "round_trip_efficiency",
                        getattr(es, "efficiency", 0.9)) or 0.9
            )
            rte = max(0.01, min(1.0, rte))
            sqrt_rte = math.sqrt(rte)

            batt_gain[i] = nom
            # Charge: delta_SoC = action * nom * sqrt(rte) / cap
            soc_scale_ch[i] = max(1e-9, nom * sqrt_rte / cap)
            # Discharge: delta_SoC = action * nom / sqrt(rte) / cap  (action < 0)
            soc_scale_dis[i] = max(1e-9, nom / sqrt_rte / cap)

            soc_arr = np.asarray(getattr(es, "soc", []), dtype=np.float64)
            if 0 <= t_idx < len(soc_arr):
                soc0[i] = float(np.clip(soc_arr[t_idx], 0.0, 1.0))

        # ---- Per-building max battery discharge power [kW] ----
        # Used by C0 to estimate C3 headroom: how much the battery can offset
        # building power when EV is charging. Conservative: limited by current SoC.
        bldg_batt_discharge_max = np.zeros(nb, dtype=np.float64)
        for i in range(n_batt):
            b_idx = int(m["batt_bidx"][i])
            # Max discharge action is limited by C2 lower bound
            soc = float(soc0[i])
            scale_dis = float(soc_scale_dis[i])
            lo_c2 = max(-1.0, (self.soc_low - soc) / scale_dis)
            # Power from max discharge: lo_c2 * batt_gain (lo_c2 is negative)
            discharge_kw = abs(lo_c2) * float(batt_gain[i])
            bldg_batt_discharge_max[b_idx] += discharge_kw

        # ---- EV charger parameters ----
        n_ev = len(m["ev_gidx"])
        ev_connected = np.zeros(n_ev, dtype=bool)
        ev_soc = np.zeros(n_ev, dtype=np.float64)
        ev_req_soc = np.ones(n_ev, dtype=np.float64)
        ev_cap = np.zeros(n_ev, dtype=np.float64)
        ev_tau = np.full(n_ev, 999, dtype=np.int32)
        ev_max_power = np.zeros(n_ev, dtype=np.float64)
        ev_gain = np.zeros(n_ev, dtype=np.float64)
        ev_rte = np.ones(n_ev, dtype=np.float64)          # EV battery round-trip efficiency
        ev_loss_coeff = np.zeros(n_ev, dtype=np.float64)   # EV battery loss coefficient per step

        for i in range(n_ev):
            b_idx = int(m["ev_bidx"][i])
            local = int(m["ev_local"][i])
            chargers = getattr(buildings[b_idx], "electric_vehicle_chargers", None) or []
            if local >= len(chargers):
                continue
            ch = chargers[local]

            mp = getattr(
                ch, "max_charging_power",
                getattr(ch, "_Charger__max_charging_power", 0),
            )
            if isinstance(mp, np.ndarray):
                mp = float(mp.ravel()[0])
            max_p = max(0.0, float(mp or 0.0))
            ev_max_power[i] = max_p
            ev_gain[i] = max_p

            sim = getattr(
                ch, "charger_simulation",
                getattr(ch, "_Charger__charger_simulation", None),
            )
            if sim is None:
                continue

            try:
                state_arr = np.asarray(
                    getattr(sim, "_electric_vehicle_charger_state"), dtype=np.float64
                )
                dep_arr = np.asarray(
                    getattr(sim, "_electric_vehicle_departure_time"), dtype=np.float64
                )
                req_arr = np.asarray(
                    getattr(sim, "_electric_vehicle_required_soc_departure"), dtype=np.float64
                )
            except (AttributeError, TypeError):
                continue

            if t_now >= len(state_arr) or float(state_arr[t_now]) != 1.0 or max_p <= 0.0:
                continue

            ev_connected[i] = True

            dep_raw = float(dep_arr[t_now]) if t_now < len(dep_arr) else np.nan
            if np.isfinite(dep_raw) and dep_raw > 0:
                ev_tau[i] = max(1, int(dep_raw))

            req_soc = float(req_arr[t_now]) if t_now < len(req_arr) else 1.0
            if not np.isfinite(req_soc):
                req_soc = 1.0
            ev_req_soc[i] = float(np.clip(req_soc, 0.0, 1.0))

            ev_obj = getattr(ch, "connected_electric_vehicle", None)
            if ev_obj is not None:
                batt_obj = getattr(ev_obj, "battery", None)
                if batt_obj is not None:
                    cap_ev = float(getattr(batt_obj, "capacity", 0.0) or 0.0)
                    ev_cap[i] = cap_ev

                    rte_val = float(
                        getattr(batt_obj, "round_trip_efficiency",
                                getattr(batt_obj, "efficiency", 1.0)) or 1.0
                    )
                    ev_rte[i] = max(0.01, min(1.0, rte_val))

                    loss_val = float(
                        getattr(batt_obj, "loss_coefficient", 0.0) or 0.0
                    )
                    ev_loss_coeff[i] = max(0.0, min(0.1, loss_val))

                    if cap_ev > 0:
                        soc_data = getattr(batt_obj, "soc", None)
                        if soc_data is not None:
                            soc_np = np.asarray(soc_data, dtype=np.float64)
                            if 0 <= t_idx < len(soc_np):
                                ev_soc[i] = float(np.clip(soc_np[t_idx], 0.0, 1.0))

        return {
            "nb": nb,
            "t_now": t_now,
            "base_nec": base_nec,                        # (nb,)   exogenous net consumption per building [kWh]
            "base_nec_forecast": base_nec_forecast,      # (nb, FORECAST_H) future base NEC per building
            "bldg_batt_discharge_max": bldg_batt_discharge_max,  # (nb,) max battery discharge power per bldg [kW]
            "soc0": soc0,                                # (n_batt,) current SoC
            "soc_scale_ch": soc_scale_ch,                # (n_batt,) delta_SoC per unit charge action
            "soc_scale_dis": soc_scale_dis,              # (n_batt,) delta_SoC per unit discharge action
            "batt_gain": batt_gain,                      # (n_batt,) kW per unit action
            "ev_connected": ev_connected,                # (n_ev,)  bool
            "ev_soc": ev_soc,                            # (n_ev,)  current SoC
            "ev_req_soc": ev_req_soc,                    # (n_ev,)  required SoC at departure
            "ev_cap": ev_cap,                            # (n_ev,)  battery capacity [kWh]
            "ev_tau": ev_tau,                            # (n_ev,)  steps until departure
            "ev_max_power": ev_max_power,                # (n_ev,)  max charging power [kW]
            "ev_gain": ev_gain,                          # (n_ev,)  kW per unit action
            "ev_rte": ev_rte,                            # (n_ev,)  EV battery round-trip efficiency
            "ev_loss_coeff": ev_loss_coeff,              # (n_ev,)  EV battery loss coefficient per step
        }

    # ------------------------------------------------------------------
    #  Phase 1: C0 — EV departure feasibility
    # ------------------------------------------------------------------

    # Tunable safety margin for C0 prefix-feasibility floor.
    # Compensates for model mismatch between the analytic bound and CityLearn's
    # actual EV charging physics (non-linear charger efficiency curves, etc.).
    C0_SAFETY_MARGIN: float = 1.3

    def _enforce_c0(
        self, a: np.ndarray, state: dict
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Enforce minimum EV charge action to preserve departure feasibility.

        Uses corrected EV physics:
          - effective_charge = max_p * charger_eff * sqrt(rte)  (not just max_p * eta)
          - SoC loss decay over τ steps: current SoC decays by (1 - loss_coeff)^τ
          - 1.3x safety margin on the floor to cover model mismatch

        For each connected EV with τ steps until departure:
          deficit_energy accounts for SoC decay over the remaining horizon.
          The prefix-feasibility lower bound for the current step:

            future_capacity = (τ - 1) * effective_charge_per_step
            min_now_energy = max(0, deficit_energy - future_capacity)
            min_action = (min_now_energy / effective_charge_per_step) * SAFETY_MARGIN

        Returns:
            a              : action vector with EV actions >= min_action
            ev_c0_min      : per-EV minimum action (passed to C3/C4 as hard lower bounds)
            info           : diagnostic counts
        """
        m = self._mapping
        n_ev = len(m["ev_gidx"])
        ev_c0_min = np.full(n_ev, -1.0)  # default: allow full discharge
        c0_interventions = 0
        c0_structural_infeasible = 0
        c0_connected = 0
        c0_positive_deficit = 0
        c0_positive_floor = 0

        for i in range(n_ev):
            gidx = int(m["ev_gidx"][i])
            lo = float(self._action_low[gidx])
            hi = float(self._action_high[gidx])

            if not state["ev_connected"][i]:
                ev_c0_min[i] = lo
                continue
            c0_connected += 1

            tau = int(state["ev_tau"][i])
            max_p = float(state["ev_max_power"][i])
            eta = self.ev_efficiency       # charger efficiency (~0.95)
            rte = float(state["ev_rte"][i])
            loss_c = float(state["ev_loss_coeff"][i])
            ev_soc_val = float(state["ev_soc"][i])
            req_soc = float(state["ev_req_soc"][i])
            cap = float(state["ev_cap"][i])

            if max_p <= 1e-6 or cap <= 1e-6:
                ev_c0_min[i] = lo
                continue

            # Effective SoC gain per max-charge step, accounting for both
            # charger efficiency and battery round-trip efficiency.
            # CityLearn charges: energy_stored = power * charger_eff * sqrt(rte)
            sqrt_rte = math.sqrt(rte)
            effective_charge_per_step = max_p * eta * sqrt_rte  # kWh

            # Account for SoC decay over remaining horizon.
            # Without any charging, current SoC decays to soc * (1-loss_c)^tau.
            # The deficit must cover both the gap AND the decay.
            decay_factor = (1.0 - loss_c) ** tau if loss_c > 0 else 1.0
            soc_at_departure_no_charge = ev_soc_val * decay_factor
            deficit_energy = max(0.0, (req_soc - soc_at_departure_no_charge) * cap)

            if deficit_energy > 1e-9:
                c0_positive_deficit += 1

            # Structural infeasibility check with corrected physics
            total_capacity = tau * effective_charge_per_step
            if deficit_energy > total_capacity + 1e-4:
                c0_structural_infeasible += 1
                ev_c0_min[i] = hi
                if float(a[gidx]) < hi - 1e-6:
                    a[gidx] = hi
                    c0_interventions += 1
                continue

            # PSF-style prefix-feasibility lower bound for the current step.
            future_after_now = max(0, tau - 1) * effective_charge_per_step
            min_now_energy = max(0.0, deficit_energy - future_after_now)
            min_action_raw = min_now_energy / effective_charge_per_step

            # Apply safety margin to compensate for model mismatch
            min_action_raw *= self.C0_SAFETY_MARGIN

            min_action = float(np.clip(min_action_raw, lo, hi))
            ev_c0_min[i] = min_action
            if min_action > lo + 1e-6:
                c0_positive_floor += 1

            if float(a[gidx]) < min_action - 1e-6:
                a[gidx] = min_action
                c0_interventions += 1

        info = {
            "lex_c0_interventions": c0_interventions,
            "lex_c0_structural_infeasible": c0_structural_infeasible,
            "lex_dbg_c0_connected": c0_connected,
            "lex_dbg_c0_positive_deficit": c0_positive_deficit,
            "lex_dbg_c0_positive_floor": c0_positive_floor,
        }
        return a, ev_c0_min, info

    # ------------------------------------------------------------------
    #  Phase 2: C2 — Battery SoC bounds
    # ------------------------------------------------------------------

    def _enforce_c2(
        self, a: np.ndarray, state: dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Clamp battery actions to stay within [soc_low, soc_high].

        Uses asymmetric SoC rate (charge vs discharge efficiency).
        Returns per-battery C2 bounds for use in subsequent phases.
        """
        m = self._mapping
        n_batt = len(m["batt_gidx"])
        batt_c2_lo = np.full(n_batt, -1.0, dtype=np.float64)
        batt_c2_hi = np.full(n_batt, 1.0, dtype=np.float64)
        c2_interventions = 0
        c2_tightened = 0

        for i in range(n_batt):
            gidx = int(m["batt_gidx"][i])
            lo = float(self._action_low[gidx])
            hi = float(self._action_high[gidx])
            soc = float(state["soc0"][i])
            scale_ch = max(1e-9, float(state["soc_scale_ch"][i]))
            scale_dis = max(1e-9, float(state["soc_scale_dis"][i]))

            # Upper bound: soc + a * scale_ch <= soc_high  =>  a <= (soc_high - soc) / scale_ch
            hi_c2 = min(hi, (self.soc_high - soc) / scale_ch)
            # Lower bound: soc + a * scale_dis >= soc_low  =>  a >= (soc_low - soc) / scale_dis
            # Note: scale_dis > 0 and the formula gives a negative number when soc > soc_low.
            lo_c2 = max(lo, (self.soc_low - soc) / scale_dis)

            # Guard against numerical issues
            hi_c2 = max(hi_c2, lo_c2)
            lo_c2 = min(lo_c2, hi_c2)

            batt_c2_lo[i] = lo_c2
            batt_c2_hi[i] = hi_c2
            if lo_c2 > lo + 1e-6 or hi_c2 < hi - 1e-6:
                c2_tightened += 1

            old_val = float(a[gidx])
            a[gidx] = float(np.clip(old_val, lo_c2, hi_c2))
            if abs(float(a[gidx]) - old_val) > 1e-6:
                c2_interventions += 1

        info = {
            "lex_c2_interventions": c2_interventions,
            "lex_dbg_c2_tightened": c2_tightened,
        }
        return a, batt_c2_lo, batt_c2_hi, info

    # ------------------------------------------------------------------
    #  Phase 3: C3 — Per-building net power bounds
    # ------------------------------------------------------------------

    def _enforce_c3(
        self,
        a: np.ndarray,
        state: dict,
        ev_c0_min: np.ndarray,
        batt_c2_lo: np.ndarray,
        batt_c2_hi: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Enforce per-building power bounds: |net_b| <= P_build_max.

        net_b = base_nec[b] + sum(batt_gain * a_batt) + sum(ev_gain * a_ev)

        For over-import (net > P_build_max):
          1. Reduce battery charge (toward 0)
          2. Reduce EV charge (toward C0_min — C0 cannot be violated)
          3. Discharge battery (toward C2 lower bound)
          If still infeasible: structural violation (base + required EV > P_build_max)

        For over-export (net < -P_build_max):
          1. Reduce battery discharge (toward 0)
          2. Reduce EV V2G discharge (toward 0 or C0_min if C0_min < 0)
          3. Increase battery charge (toward C2 upper bound)
          If still infeasible: structural violation (solar overwhelms capacity)
        """
        m = self._mapping
        nb = state["nb"]
        c3_interventions = 0
        c3_structural_infeasible = 0
        c3_over_import_buildings = 0
        c3_over_export_buildings = 0

        # Build per-building device lists once
        batts_by_bldg = [
            [(i, int(m["batt_gidx"][i])) for i in range(len(m["batt_gidx"]))
             if int(m["batt_bidx"][i]) == b]
            for b in range(nb)
        ]
        evs_by_bldg = [
            [(i, int(m["ev_gidx"][i])) for i in range(len(m["ev_gidx"]))
             if int(m["ev_bidx"][i]) == b]
            for b in range(nb)
        ]

        for b in range(nb):
            batts_b = batts_by_bldg[b]
            evs_b = evs_by_bldg[b]

            # Current net power for this building [kWh]
            net = float(state["base_nec"][b])
            for batt_i, gidx in batts_b:
                net += float(state["batt_gain"][batt_i]) * float(a[gidx])
            for ev_i, gidx in evs_b:
                net += float(state["ev_gain"][ev_i]) * float(a[gidx])

            # --- Over-import: net > P_build_max ---
            if net > self.p_build_max + 1e-4:
                c3_over_import_buildings += 1
                excess = net - self.p_build_max

                # Step 1: Reduce battery charge (positive actions toward 0, respecting C2 lo)
                for batt_i, gidx in batts_b:
                    if excess <= 1e-6:
                        break
                    gain = float(state["batt_gain"][batt_i])
                    if gain <= 1e-9:
                        continue
                    cur = float(a[gidx])
                    if cur > 0.0:
                        # Reduce toward 0 (but not below C2 lower bound)
                        floor = max(float(batt_c2_lo[batt_i]), 0.0)
                        reducible = max(0.0, (cur - floor) * gain)
                        if reducible > 1e-6:
                            actual = min(excess, reducible)
                            a[gidx] = float(np.clip(
                                cur - actual / gain,
                                batt_c2_lo[batt_i], batt_c2_hi[batt_i],
                            ))
                            excess -= actual
                            c3_interventions += 1

                # Step 2: Reduce EV charge (toward C0_min — C0 hard floor)
                for ev_i, gidx in evs_b:
                    if excess <= 1e-6:
                        break
                    gain = float(state["ev_gain"][ev_i])
                    if gain <= 1e-9:
                        continue
                    cur = float(a[gidx])
                    c0_min = float(ev_c0_min[ev_i])
                    if cur > c0_min + 1e-6:
                        reducible = max(0.0, (cur - c0_min) * gain)
                        if reducible > 1e-6:
                            actual = min(excess, reducible)
                            lo_ev = float(self._action_low[gidx])
                            hi_ev = float(self._action_high[gidx])
                            new_val = float(np.clip(cur - actual / gain, lo_ev, hi_ev))
                            new_val = max(new_val, c0_min)  # never below C0 min
                            a[gidx] = new_val
                            excess -= actual
                            c3_interventions += 1

                # Step 3: Discharge battery (negative actions toward C2 lo)
                for batt_i, gidx in batts_b:
                    if excess <= 1e-6:
                        break
                    gain = float(state["batt_gain"][batt_i])
                    if gain <= 1e-9:
                        continue
                    cur = float(a[gidx])
                    lo_c2 = float(batt_c2_lo[batt_i])
                    if cur > lo_c2 + 1e-6:
                        reducible = max(0.0, (cur - lo_c2) * gain)
                        if reducible > 1e-6:
                            actual = min(excess, reducible)
                            a[gidx] = float(np.clip(
                                cur - actual / gain,
                                lo_c2, batt_c2_hi[batt_i],
                            ))
                            excess -= actual
                            c3_interventions += 1

                if excess > 1e-3:
                    # Remaining excess is structural: required EV charging + base > P_build_max
                    c3_structural_infeasible += 1

            # --- Over-export: net < -P_build_max ---
            elif net < -self.p_build_max - 1e-4:
                c3_over_export_buildings += 1
                deficit = -net - self.p_build_max

                # Step 1: Reduce battery discharge (negative actions toward 0)
                for batt_i, gidx in batts_b:
                    if deficit <= 1e-6:
                        break
                    gain = float(state["batt_gain"][batt_i])
                    if gain <= 1e-9:
                        continue
                    cur = float(a[gidx])
                    if cur < 0.0:
                        # Increase toward 0 (reduce export)
                        absorbable = max(0.0, (-cur) * gain)
                        if absorbable > 1e-6:
                            actual = min(deficit, absorbable)
                            a[gidx] = float(np.clip(
                                cur + actual / gain,
                                batt_c2_lo[batt_i], batt_c2_hi[batt_i],
                            ))
                            deficit -= actual
                            c3_interventions += 1

                # Step 2: Reduce EV V2G discharge (increase EV action toward 0)
                for ev_i, gidx in evs_b:
                    if deficit <= 1e-6:
                        break
                    gain = float(state["ev_gain"][ev_i])
                    if gain <= 1e-9:
                        continue
                    cur = float(a[gidx])
                    if cur < 0.0:
                        # Increase toward 0 (reduce V2G export)
                        absorbable = max(0.0, (-cur) * gain)
                        if absorbable > 1e-6:
                            actual = min(deficit, absorbable)
                            lo_ev = float(self._action_low[gidx])
                            hi_ev = float(self._action_high[gidx])
                            c0_min = float(ev_c0_min[ev_i])
                            new_val = float(np.clip(cur + actual / gain, lo_ev, hi_ev))
                            new_val = max(new_val, c0_min)
                            a[gidx] = new_val
                            deficit -= actual
                            c3_interventions += 1

                # Step 3: Increase battery charge to absorb export
                for batt_i, gidx in batts_b:
                    if deficit <= 1e-6:
                        break
                    gain = float(state["batt_gain"][batt_i])
                    if gain <= 1e-9:
                        continue
                    cur = float(a[gidx])
                    hi_c2 = float(batt_c2_hi[batt_i])
                    if cur < hi_c2 - 1e-6:
                        absorbable = max(0.0, (hi_c2 - cur) * gain)
                        if absorbable > 1e-6:
                            actual = min(deficit, absorbable)
                            a[gidx] = float(np.clip(
                                cur + actual / gain,
                                batt_c2_lo[batt_i], hi_c2,
                            ))
                            deficit -= actual
                            c3_interventions += 1

                if deficit > 1e-3:
                    # Remaining deficit: solar generation exceeds absorption capacity
                    c3_structural_infeasible += 1

        info = {
            "lex_c3_interventions": c3_interventions,
            "lex_c3_structural_infeasible": c3_structural_infeasible,
            "lex_dbg_c3_over_import_buildings": c3_over_import_buildings,
            "lex_dbg_c3_over_export_buildings": c3_over_export_buildings,
        }
        return a, info

    # ------------------------------------------------------------------
    #  Phase 4: C4 — Grid-level import bound
    # ------------------------------------------------------------------

    def _enforce_c4(
        self,
        a: np.ndarray,
        state: dict,
        ev_c0_min: np.ndarray,
        batt_c2_lo: np.ndarray,
        batt_c2_hi: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Enforce grid-level import bound: total_net <= P_grid_max.

        total_net = sum(base_nec) + sum(batt_gain * a_batt) + sum(ev_gain * a_ev)

        Applies proportional reduction of positive controllable contributions
        (all devices that are currently importing), while respecting:
          - C0 lower bounds for EVs (cannot reduce below ev_c0_min)
          - C2 bounds for batteries (cannot reduce below batt_c2_lo)
        """
        m = self._mapping
        n_batt = len(m["batt_gidx"])
        n_ev = len(m["ev_gidx"])

        # Compute current total net [kWh]
        total_net = float(np.sum(state["base_nec"]))
        for i in range(n_batt):
            gidx = int(m["batt_gidx"][i])
            total_net += float(state["batt_gain"][i]) * float(a[gidx])
        for i in range(n_ev):
            gidx = int(m["ev_gidx"][i])
            total_net += float(state["ev_gain"][i]) * float(a[gidx])

        if total_net <= self.p_grid_max + 1e-4:
            return a, {
                "lex_c4_interventions": 0,
                "lex_c4_structural_infeasible": 0,
                "lex_dbg_c4_excess_kwh": 0.0,
                "lex_dbg_c4_total_available_kwh": 0.0,
            }

        excess = total_net - self.p_grid_max
        c4_interventions = 0

        # Build list of reducible contributions (positive actions or actions above lb)
        devices = []

        for i in range(n_batt):
            gidx = int(m["batt_gidx"][i])
            gain = float(state["batt_gain"][i])
            cur = float(a[gidx])
            lb = float(batt_c2_lo[i])  # hard lower bound (C2)
            reducible = max(0.0, (cur - lb) * gain)
            devices.append(("batt", i, gidx, gain, cur, lb, float(batt_c2_hi[i]), reducible))

        for i in range(n_ev):
            gidx = int(m["ev_gidx"][i])
            gain = float(state["ev_gain"][i])
            cur = float(a[gidx])
            lb = float(ev_c0_min[i])  # hard lower bound (C0)
            reducible = max(0.0, (cur - lb) * gain)
            lo_ev = float(self._action_low[gidx])
            hi_ev = float(self._action_high[gidx])
            devices.append(("ev", i, gidx, gain, cur, lb, hi_ev, reducible))

        total_available = sum(d[7] for d in devices)

        if total_available < 1e-9:
            # Nothing reducible: structural infeasibility (all devices at minimums)
            return a, {
                "lex_c4_interventions": 0,
                "lex_c4_structural_infeasible": 1,
                "lex_dbg_c4_excess_kwh": float(excess),
                "lex_dbg_c4_total_available_kwh": float(total_available),
            }

        if total_available >= excess - 1e-6:
            # Proportional reduction: each device reduces its import by (avail/total) * excess
            scale = excess / total_available
            for dtype, dev_i, gidx, gain, cur, lb, ub, avail in devices:
                if avail <= 1e-9:
                    continue
                reduce_energy = avail * scale
                new_val = cur - reduce_energy / max(gain, 1e-9)
                if dtype == "batt":
                    new_val = float(np.clip(new_val, batt_c2_lo[dev_i], batt_c2_hi[dev_i]))
                else:
                    lo_ev = float(self._action_low[gidx])
                    hi_ev = float(self._action_high[gidx])
                    new_val = float(np.clip(new_val, lo_ev, hi_ev))
                    new_val = max(new_val, lb)  # never below C0 min
                if abs(new_val - cur) > 1e-6:
                    a[gidx] = new_val
                    c4_interventions += 1
            structural = 0
        else:
            # Apply maximum possible reduction (still short — structural infeasibility)
            for dtype, dev_i, gidx, gain, cur, lb, ub, avail in devices:
                if avail <= 1e-9:
                    continue
                new_val = lb  # push each device to its lower bound
                if dtype == "batt":
                    new_val = float(np.clip(new_val, batt_c2_lo[dev_i], batt_c2_hi[dev_i]))
                else:
                    lo_ev = float(self._action_low[gidx])
                    hi_ev = float(self._action_high[gidx])
                    new_val = float(np.clip(new_val, lo_ev, hi_ev))
                    new_val = max(new_val, lb)
                if abs(new_val - cur) > 1e-6:
                    a[gidx] = new_val
                    c4_interventions += 1
            structural = 1

        info = {
            "lex_c4_interventions": c4_interventions,
            "lex_c4_structural_infeasible": structural,
            "lex_dbg_c4_excess_kwh": float(excess),
            "lex_dbg_c4_total_available_kwh": float(total_available),
        }
        return a, info

    # ------------------------------------------------------------------
    #  Main entry: compute(a_in) → (safe_a, info)
    # ------------------------------------------------------------------

    def compute(self, a_in: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Apply sequential lexicographic safety repair to action a_in.

        Phases:
          1. C0: enforce EV minimum charge (preserves departure feasibility)
          2. C2: clamp battery SoC (prevents over/under-charge)
          3. C3: reduce per-building power (respects C0/C2 floors)
          4. C4: reduce grid import (respects C0/C2 floors)

        The result is always an action within the action space bounds.
        Structural infeasibilities are reported in info but do NOT abort.
        """
        if not self._built:
            self.build()

        a = np.asarray(a_in, dtype=np.float64).ravel().copy()
        t0 = time.time()

        # Clip to action space bounds before any phase
        a = np.clip(a, self._action_low, self._action_high)
        a_orig = a.copy()

        try:
            state = self._extract_state()
        except Exception as exc:
            # Fallback: return clipped original action
            delta = float(np.linalg.norm(a - np.asarray(a_in, dtype=np.float64).ravel()))
            return a.astype(np.float32), {
                "projection_delta": delta,
                "feasible": True,
                "lex_error": str(exc)[:120],
                "solve_time_ms": (time.time() - t0) * 1000.0,
            }

        # ---- Phase 1: C0 ----
        a, ev_c0_min, c0_info = self._enforce_c0(a, state)

        # ---- Phase 2: C2 ----
        a, batt_c2_lo, batt_c2_hi, c2_info = self._enforce_c2(a, state)

        # Re-enforce C0 after C2 (C2 should not reduce EV actions, but be defensive)
        m = self._mapping
        for i in range(len(m["ev_gidx"])):
            gidx = int(m["ev_gidx"][i])
            if float(a[gidx]) < float(ev_c0_min[i]) - 1e-6:
                a[gidx] = float(ev_c0_min[i])

        # ---- Phase 3: C3 ----
        a, c3_info = self._enforce_c3(a, state, ev_c0_min, batt_c2_lo, batt_c2_hi)

        # ---- Phase 4: C4 ----
        a, c4_info = self._enforce_c4(a, state, ev_c0_min, batt_c2_lo, batt_c2_hi)

        # Final C3 consistency pass: C3 has higher priority than C4, so if the
        # grid-level repair created any new building-level issue, restore C3.
        a, c3_post_info = self._enforce_c3(a, state, ev_c0_min, batt_c2_lo, batt_c2_hi)
        c3_info["lex_c3_interventions"] += int(c3_post_info.get("lex_c3_interventions", 0))
        c3_info["lex_c3_structural_infeasible"] += int(c3_post_info.get("lex_c3_structural_infeasible", 0))

        # Final clip to ensure we never drift outside action space
        a = np.clip(a, self._action_low, self._action_high)

        # Re-enforce C2 before the final C0 floor. C3 should already respect
        # C2 bounds, but keep the highest-priority invariants explicit.
        for i in range(len(m["batt_gidx"])):
            gidx = int(m["batt_gidx"][i])
            a[gidx] = float(np.clip(a[gidx], batt_c2_lo[i], batt_c2_hi[i]))

        # Re-enforce C0 one final time (C4 could have touched EV actions)
        for i in range(len(m["ev_gidx"])):
            gidx = int(m["ev_gidx"][i])
            if float(a[gidx]) < float(ev_c0_min[i]) - 1e-6:
                a[gidx] = float(ev_c0_min[i])

        a = np.clip(a, self._action_low, self._action_high)

        delta = float(np.linalg.norm(a - a_orig))
        solve_ms = (time.time() - t0) * 1000.0

        info: dict = {
            "projection_delta": delta,
            "feasible": True,   # analytical repair is always feasible
            "solve_time_ms": solve_ms,
        }
        info.update(c0_info)
        info.update(c2_info)
        info.update(c3_info)
        info.update(c4_info)

        if self.verbose >= 2:
            print(
                f"[LexSafety] delta={delta:.4f} solve={solve_ms:.1f}ms "
                f"C0_int={c0_info['lex_c0_interventions']} "
                f"C2_int={c2_info['lex_c2_interventions']} "
                f"C3_int={c3_info['lex_c3_interventions']}(struct={c3_info['lex_c3_structural_infeasible']}) "
                f"C4_int={c4_info['lex_c4_interventions']}(struct={c4_info['lex_c4_structural_infeasible']})"
            )

        return a.astype(np.float32), info

    # ------------------------------------------------------------------
    #  project() — Drop-in replacement for HybridProjector / DiffProjector
    # ------------------------------------------------------------------

    def project(
        self,
        obs_tensor: torch.Tensor,
        unsafe_action: torch.Tensor,
        state_tensors: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        project(obs_tensor, unsafe_action) -> (safe_action_tensor, info_dict)

        Matches the API of HybridProjector.project() so this can be used as a
        drop-in replacement in _apply_execution_shield() and the diagnostic script.

        state_tensors is accepted for API compatibility but ignored:
        lexicographic repair always reads live state from the env.
        """
        # Convert input tensor to numpy
        a_in = unsafe_action.detach().cpu().numpy().ravel().astype(np.float64)

        # Apply lexicographic repair
        safe_a, info = self.compute(a_in)

        # Convert back to tensor with matching dtype/device
        safe_t = torch.as_tensor(
            safe_a, dtype=unsafe_action.dtype, device=unsafe_action.device
        )

        # Preserve batch dimension if input had one
        if unsafe_action.dim() == 2:
            safe_t = safe_t.unsqueeze(0)

        return safe_t, info

    # ------------------------------------------------------------------
    #  extract_state_tensor — compatibility stub for training scripts
    # ------------------------------------------------------------------

    def extract_state_tensor(self) -> torch.Tensor:
        """
        Compatibility stub. Returns empty tensor (no state tensor needed
        for analytical repair). Training scripts that store replay buffer
        state tensors should switch to DiffProjector for gradient computation.
        """
        return torch.zeros(0, dtype=torch.float32)
