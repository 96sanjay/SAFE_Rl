"""
Paper-exact SE-RL with closest-point QP projection and penalty.

Implements Markgraf et al. 2025 "Safe RL using Action Projection"
Sections 5.1 (SE-RL) and 7.1 (penalty augmentation).

Eq. 12: Phi(x, u) = argmin_{ũ} ½||ũ - u||²  s.t. s(x, ũ) ≤ 0
         Solved via cvxpy QP (non-differentiable, in environment).
Eq. 23: r_aug = r - h
Eq. 24: h = w * ||u - Phi(x,u)||^2  (zero when u is in the safe set)

Constraints enforced:
    C2: Battery SoC in [soc_low, soc_high]
    C3: Per-building |NEC| <= P_building_max
    C4: Grid import <= P_grid_max (joint urgency-weighted allocation)

C0/C1 (EV departure) handled by Lagrangian externally.

Env vars:
    CITYLEARN_SERL_PROJECTION="1"  -- enable this wrapper
    SE_RL_PENALTY_WEIGHT="0.5"     -- penalty weight w (paper tested {0.1, 0.5, 1.0, 2.0})
    MASK_C4_ENABLED="1"            -- enable C4 grid constraint for batteries
    CITYLEARN_STEMS_P_BUILDING_MAX -- per-building power limit kW (default 4.6083)
    CITYLEARN_STEMS_SOC_LOW        -- battery SoC lower bound (default 0.0)
    CITYLEARN_STEMS_SOC_HIGH       -- battery SoC upper bound (default 0.95)
    CITYLEARN_STEMS_P_GRID_MAX     -- grid aggregate power limit kW (default 10.2352)
"""
from __future__ import annotations

import os
import re
from typing import Tuple

import gymnasium as gym
import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


def _unwrap_citylearn(env):
    """Walk the wrapper chain to find the CityLearn env with .buildings."""
    inner = env
    for _ in range(20):
        if hasattr(inner, 'buildings') and len(getattr(inner, 'buildings', [])) > 0:
            return inner
        inner = getattr(inner, '_env',
                 getattr(inner, 'env',
                 getattr(inner, 'base', None)))
        if inner is None:
            break
    return None


_EV_RE = re.compile(
    r"^electric_vehicle_storage_charger_(?P<suffix>.+)$", re.IGNORECASE)


class ActionProjectionSERL(gym.Wrapper):
    """Paper-exact SE-RL with closest-point projection and penalty.

    Implements Markgraf et al. 2025 "Safe RL using Action Projection"
    Sections 5.1 (SE-RL) and 7.1 (penalty augmentation).

    Eq. 12: Phi(x, u) = clip(u, safe_min(x), safe_max(x))
    Eq. 23: r_aug = r - h
    Eq. 24: h = w * ||u - Phi(x,u)||^2  (zero when u is in the safe set)

    Constraints enforced:
      C2: Battery SoC in [soc_low, soc_high]
      C3: Per-building |NEC| <= P_building_max
      C4: Grid import <= P_grid_max (joint urgency-weighted allocation)

    C0/C1 (EV departure) handled by Lagrangian externally.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self._p_bmax = float(os.environ.get(
            "CITYLEARN_STEMS_P_BUILDING_MAX", "4.6083"))
        self._soc_low = float(os.environ.get(
            "CITYLEARN_STEMS_SOC_LOW", "0.0"))
        self._soc_high = float(os.environ.get(
            "CITYLEARN_STEMS_SOC_HIGH", "0.95"))
        self._p_gmax = float(os.environ.get(
            "CITYLEARN_STEMS_P_GRID_MAX", "10.2352"))

        # SoC margin for C2 enforcement
        self._soc_margin = 0.02

        # Eq. 24: SE-RL penalty weight
        self._penalty_w = float(os.environ.get("SE_RL_PENALTY_WEIGHT", "0.5"))

        # C4 grid mask toggle (cached, not per-step)
        self._c4_enabled = os.environ.get("MASK_C4_ENABLED", "0") == "1"

        # Urgency-aware joint allocation config
        self._urgency_alpha = float(os.environ.get("SERL_URGENCY_ALPHA", "5.0"))
        self._infeasibility_policy = os.environ.get("SERL_INFEASIBILITY_POLICY", "safety")
        self._mobility_overrides = 0  # cumulative count per episode

        # Beta actor incompatibility guard: SE-RL projection clips in [-1,1]
        # action space. Beta actor outputs in (0,1). These are incompatible.
        # Use ActionMaskWrapper (which has beta_mode affine transform) instead.
        if os.environ.get("CITYLEARN_BETA_ACTOR", "0") == "1":
            raise RuntimeError(
                "[ActionProjectionSERL] Incompatible with CITYLEARN_BETA_ACTOR=1. "
                "Beta actor outputs x∈(0,1) but SE-RL projection clips in [-1,1] space. "
                "Use CITYLEARN_ACTION_MASK=1 (ActionMaskWrapper) instead, which has "
                "beta_mode affine transform support."
            )

        # Discover CityLearn env and read device specs
        self._city = _unwrap_citylearn(env)
        if self._city is None:
            raise RuntimeError(
                "[SE-RL Projection] Cannot find CityLearn env with .buildings "
                "in wrapper chain")

        self._n_buildings = len(self._city.buildings)

        # --- Discover action layout from action_names ---
        action_names = self._city.action_names
        if isinstance(action_names, list) and len(action_names) == 1 \
                and isinstance(action_names[0], list):
            action_names_flat = action_names[0]
        else:
            action_names_flat = list(action_names)

        self._n_actions = len(action_names_flat)

        # Build building-number to list-index mapping
        # Building names might be "Building_1", "Building_2", "Building_4"
        # but list indices are 0, 1, 2
        self._bldg_num_to_idx = {}
        for list_idx, b in enumerate(self._city.buildings):
            bname = getattr(b, 'name', f'Building_{list_idx+1}')
            # Extract number from name (e.g., "Building_4" -> 4)
            nums = re.findall(r'\d+', str(bname))
            bnum = int(nums[0]) if nums else list_idx + 1
            self._bldg_num_to_idx[bnum] = list_idx

        # Parse action_names to build mapping
        self._building_batt_act = {}  # b_list_idx -> action_idx
        self._building_ev_act = {}    # b_list_idx -> action_idx
        self._passthrough_indices = []

        batt_list_idx = 0  # sequential assignment for battery actions
        for act_idx, name in enumerate(action_names_flat):
            name_lower = name.lower()
            if _EV_RE.match(name):
                m = _EV_RE.match(name)
                suffix = m.group('suffix')
                parts = suffix.split('_')
                bldg_num = int(parts[0])
                ev_list_idx = self._bldg_num_to_idx.get(bldg_num, bldg_num - 1)
                self._building_ev_act[ev_list_idx] = act_idx
            elif 'washing_machine' in name_lower:
                self._passthrough_indices.append(act_idx)
            elif name_lower == 'electrical_storage':
                self._building_batt_act[batt_list_idx] = act_idx
                batt_list_idx += 1
            else:
                self._passthrough_indices.append(act_idx)

        self._ev_building_indices = sorted(self._building_ev_act.keys())
        self._batt_act_indices = sorted(self._building_batt_act.values())
        self._ev_act_indices = [self._building_ev_act[b] for b in self._ev_building_indices]

        # Read battery nominal powers (kW)
        self._batt_powers = {}
        for bi in self._building_batt_act:
            b = self._city.buildings[bi]
            es = getattr(b, 'electrical_storage', None)
            if es is not None:
                p = getattr(es, 'nominal_power', None)
                self._batt_powers[bi] = float(p) if p and p > 0 else 5.0
            else:
                self._batt_powers[bi] = 5.0

        # Read EV charger powers (max and min for charge/discharge)
        self._ev_max_charge = {}
        self._ev_min_charge = {}
        self._ev_max_discharge = {}
        self._ev_min_discharge = {}
        for bi in self._ev_building_indices:
            b = self._city.buildings[bi]
            chargers = getattr(b, 'electric_vehicle_chargers',
                       getattr(b, 'chargers',
                       getattr(b, 'ev_chargers', [])))
            if chargers and len(chargers) > 0:
                c = chargers[0]
                self._ev_max_charge[bi] = float(getattr(c, 'max_charging_power', 0) or 0)
                self._ev_min_charge[bi] = float(getattr(c, 'min_charging_power', 0) or 0)
                self._ev_max_discharge[bi] = float(getattr(c, 'max_discharging_power', 0) or 0)
                self._ev_min_discharge[bi] = float(getattr(c, 'min_discharging_power', 0) or 0)
            else:
                self._ev_max_charge[bi] = 0
                self._ev_min_charge[bi] = 0
                self._ev_max_discharge[bi] = 0
                self._ev_min_discharge[bi] = 0

        # For backwards compatibility, keep _ev_powers as max_charging_power
        self._ev_powers = {bi: self._ev_max_charge[bi] for bi in self._ev_building_indices}

        # Diagnostics
        self._step_count = 0
        self._total_clipped = 0
        self._last_safe_min = np.full(self._n_actions, -1.0, dtype=np.float32)
        self._last_safe_max = np.full(self._n_actions, 1.0, dtype=np.float32)
        self._last_total_exo_import = 0.0

        print(f"[SE-RL Projection] ENABLED: w={self._penalty_w}, "
              f"P_bmax={self._p_bmax:.2f} kW, "
              f"P_gmax={self._p_gmax:.2f} kW, "
              f"N_buildings={self._n_buildings}")
        print(f"[SE-RL Projection] Action layout ({self._n_actions} dims):")
        print(f"  Battery actions: {self._building_batt_act}")
        print(f"  EV actions: {self._building_ev_act}")
        print(f"  Passthrough actions: {self._passthrough_indices}")
        print(f"[SE-RL Projection] Battery powers: {self._batt_powers}")
        print(f"[SE-RL Projection] EV max charge: {self._ev_max_charge}")
        print(f"[SE-RL Projection] EV min charge: {self._ev_min_charge}")
        print(f"[SE-RL Projection] EV max discharge: {self._ev_max_discharge}")
        print(f"[SE-RL Projection] EV min discharge: {self._ev_min_discharge}")
        print(f"[SE-RL Projection] SoC bounds: [{self._soc_low}, {self._soc_high}], "
              f"margin={self._soc_margin}")
        print(f"[SE-RL Projection] Urgency alpha={self._urgency_alpha}, "
              f"infeasibility_policy={self._infeasibility_policy}")

    def reset(self, **kwargs):
        self._step_count = 0
        self._total_clipped = 0
        self._mobility_overrides = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float32)

        # Compute state-dependent safe bounds
        exo_nec = self._get_exogenous_nec()
        socs = self._get_battery_socs()
        safe_min, safe_max, interventions, n_infeasible = self._compute_safe_bounds(exo_nec, socs)

        # Eq. 12: Closest-point QP projection
        # Phi(x, u) = argmin_{ũ} ½||ũ - u||²  s.t. constraints
        safe_action = self._qp_project(raw_action, safe_min, safe_max, exo_nec)

        # Eq. 24: Penalty (zero when inside bounds)
        delta = raw_action - safe_action
        h = self._penalty_w * float(np.sum(delta ** 2))

        # Count how many dimensions were clipped
        n_clipped = int(np.sum(np.abs(delta) > 1e-6))

        # Execute safe action
        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        # Eq. 23: Augmented reward
        reward = reward - h

        # Store diagnostics
        info["serl_projection_enabled"] = 1.0
        info["serl_penalty"] = h
        info["serl_delta_l2"] = float(np.linalg.norm(delta))
        info["serl_n_clipped"] = float(n_clipped)
        info["serl_n_total"] = float(len(raw_action))
        info["serl_clip_rate"] = float(n_clipped) / max(1, len(raw_action))
        info["serl_safe_min"] = safe_min.tolist()
        info["serl_safe_max"] = safe_max.tolist()
        info["serl_raw_action"] = raw_action.tolist()
        info["serl_n_infeasible"] = float(n_infeasible)  # dims where C3/C4 bounds conflict
        info["serl_structural_c4"] = float(getattr(self, '_last_structural_c4', 0))
        # QP solve path diagnostics
        qp_path = getattr(self, '_last_qp_path', 'unknown')
        info["serl_qp_path"] = qp_path
        info["serl_qp_solved"] = 1.0 if qp_path == "qp_solved" else 0.0
        info["serl_qp_fastpath"] = 1.0 if qp_path == "fastpath" else 0.0
        info["serl_qp_fallback"] = 1.0 if qp_path == "fallback" else 0.0
        info["serl_qp_solve_ms"] = getattr(self, '_last_qp_ms', 0.0)
        info["serl_c0_c4_conflict"] = float(getattr(self, '_last_c0_c4_conflict', 0))
        info["serl_c0_c3_conflict"] = float(getattr(self, '_last_c0_c3_conflict', 0))
        info["serl_c4_relaxation_kw"] = float(getattr(self, '_last_c4_relaxation_kw', 0))
        info["serl_mobility_overrides"] = float(self._mobility_overrides)

        # Store bounds on CityLearn env for other wrappers to access
        if self._city is not None:
            self._city._action_mask_safe_min = safe_min.copy()
            self._city._action_mask_safe_max = safe_max.copy()

        self._last_safe_min = safe_min.copy()
        self._last_safe_max = safe_max.copy()

        # Periodic logging
        self._step_count += 1
        self._total_clipped += n_clipped
        if self._step_count % 1000 == 0:
            clip_rate = self._total_clipped / (self._step_count * len(raw_action))
            print(f"[SE-RL Projection] step={self._step_count} "
                  f"clip_rate={clip_rate:.1%} "
                  f"penalty_mean={h:.4f} "
                  f"delta_l2={np.linalg.norm(delta):.4f}")

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helper methods (identical to ActionMaskWrapper)
    # ------------------------------------------------------------------

    def _get_exogenous_nec(self) -> list[float]:
        """Read exogenous NEC (non-shiftable load + solar) for each building.

        Uses energy_to_non_shiftable_load (thermal demand = electricity for
        this schema since HVAC is not actively controlled) plus solar_generation.
        """
        city = self._city
        t_idx = int(getattr(city, 'time_step', 0))

        exo_nec = []
        for b in city.buildings:
            nsl = getattr(b, '_Building__energy_to_non_shiftable_load', None)
            sg = getattr(b, '_Building__solar_generation', None)

            load_val = 0.0
            if nsl is not None and len(nsl) > t_idx:
                load_val = float(nsl[t_idx])

            solar_val = 0.0
            if sg is not None and len(sg) > t_idx:
                solar_val = float(sg[t_idx])

            exo_nec.append(load_val + solar_val)

        return exo_nec

    def _get_battery_socs(self) -> list[float]:
        """Read current battery SoC for each building."""
        socs = []
        for b in self._city.buildings:
            es = getattr(b, 'electrical_storage', None)
            if es is not None:
                soc = getattr(es, 'soc', None)
                if soc is not None:
                    if hasattr(soc, '__len__') and len(soc) > 0:
                        t = int(getattr(self._city, 'time_step', 0))
                        idx = max(0, t - 1)
                        if idx < len(soc):
                            socs.append(float(soc[idx]))
                        else:
                            socs.append(float(soc[-1]))
                    else:
                        socs.append(float(soc))
                else:
                    socs.append(0.5)
            else:
                socs.append(0.5)
        return socs

    def _is_ev_connected(self, b_idx: int) -> bool:
        """Check if EV charger at building b_idx has a connected vehicle."""
        if self._city is None or b_idx not in self._building_ev_act:
            return False
        try:
            b = self._city.buildings[b_idx]
            chargers = getattr(b, 'electric_vehicle_chargers',
                       getattr(b, 'chargers', []))
            if not chargers:
                return False
            ch = chargers[0]
            sim = getattr(ch, 'charger_simulation',
                  getattr(ch, '_Charger__charger_simulation', None))
            if sim is None:
                return True  # assume connected if can't check
            state_arr = getattr(sim, '_electric_vehicle_charger_state', None)
            if state_arr is None:
                return True
            t = int(getattr(self._city, 'time_step', 0))
            if t < len(state_arr):
                return float(state_arr[t]) == 1.0
            return False
        except Exception:
            return True  # assume connected on error (conservative)

    def _ev_has_deficit(self, b_idx: int) -> bool:
        """Check if connected EV at building b_idx has SoC below required.

        Uses _ev_state() to avoid duplicated access logic. Returns False if
        EV state unavailable, t_dep <= 0 (departing/away), or no deficit.
        """
        state = self._ev_state(b_idx)
        if state is None:
            return False
        soc_now, soc_req, t_dep, _, _ = state
        if t_dep <= 0:
            return False
        return soc_now < soc_req - 0.01

    def _ev_state(self, b_idx: int):
        """Read EV state variables for building b_idx.

        Returns (soc_now, soc_req, t_dep, cap_ev, eta_ch) or None if unavailable.
        """
        if not self._is_ev_connected(b_idx):
            return None
        try:
            b = self._city.buildings[b_idx]
            chargers = getattr(b, 'electric_vehicle_chargers',
                       getattr(b, 'chargers', []))
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
            soc_arr = getattr(bt, 'soc', None)
            if soc_arr is None or not hasattr(soc_arr, '__len__') or max(0, t-1) >= len(soc_arr):
                return None
            soc_now = float(soc_arr[max(0, t - 1)])
            cap_ev = float(getattr(bt, 'capacity', 60.0) or 60.0)
            # Prefer charging_efficiency; fallback to sqrt(round_trip_efficiency)
            eta_raw = getattr(bt, 'charging_efficiency', None)
            if eta_raw is not None and float(eta_raw) > 0:
                eta_ch = float(eta_raw)
            else:
                rte = float(getattr(bt, 'round_trip_efficiency', 0.9025) or 0.9025)
                eta_ch = float(np.sqrt(max(rte, 0.01)))
            sim = getattr(ch, 'charger_simulation',
                  getattr(ch, '_Charger__charger_simulation', None))
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

    def _ev_urgency(self, b_idx: int) -> float:
        """Compute urgency ratio for EV at building b_idx.

        Returns 0.0 if: EV not connected, no deficit, t_dep <= 0, or p_ev_max <= 0.
        """
        state = self._ev_state(b_idx)
        if state is None:
            return 0.0
        soc_now, soc_req, t_dep, cap_ev, eta_ch = state
        if t_dep <= 0 or soc_now >= soc_req - 0.01:
            return 0.0
        p_ev_max = self._ev_max_charge.get(b_idx, 0.0)
        if p_ev_max <= 0:
            return 0.0
        energy_deficit = max(0.0, soc_req - soc_now) * cap_ev
        input_deficit = energy_deficit / max(eta_ch, 0.01)
        hours_needed = input_deficit / p_ev_max
        return float(np.clip(hours_needed / t_dep, 0.0, 2.0))

    def _ev_min_charge_kw(self, b_idx: int) -> float:
        """Compute minimum charge rate (kW at charger input) to keep departure feasible.

        Accounts for charging efficiency. Returns 0.0 if no deficit, t_dep <= 0,
        or p_ev_max <= 0.
        """
        state = self._ev_state(b_idx)
        if state is None:
            return 0.0
        soc_now, soc_req, t_dep, cap_ev, eta_ch = state
        if t_dep <= 0 or soc_now >= soc_req - 0.01:
            return 0.0
        p_ev_max = self._ev_max_charge.get(b_idx, 0.0)
        if p_ev_max <= 0:
            return 0.0
        energy_deficit = max(0.0, soc_req - soc_now) * cap_ev
        input_deficit = energy_deficit / max(eta_ch, 0.01)
        return min(input_deficit / t_dep, p_ev_max)

    def _qp_project(
        self,
        raw_action: np.ndarray,
        safe_min: np.ndarray,
        safe_max: np.ndarray,
        exo_nec: list,
    ) -> np.ndarray:
        """Eq. 12: Fully joint closest-point QP projection.

        Solves one QP with ALL constraints (C2, C3, C4) simultaneously:
            min_{ũ}  ½||ũ - u||²
            s.t.  ũ ∈ [-1, 1]^n                                       (action bounds)
                  soc_low ≤ soc0 + ũ_batt × scale_ch ≤ soc_high       (C2: SoC bounds)
                  -P_bmax ≤ exo_b + Σ(gains × ũ_devices) ≤ P_bmax     (C3: building power)
                  Σ(exo_b + Σ(gains × ũ_devices)) ≤ P_gmax            (C4: grid import)

        Same formulation as DiffProjector (SP-RL) but solved non-differentiably.
        Falls back to np.clip if cvxpy unavailable or QP fails.
        """
        if not _HAS_CVXPY:
            self._last_qp_path = "no_cvxpy"
            self._last_qp_ms = 0.0
            return np.clip(raw_action, safe_min, safe_max)

        import time as _time
        _t0 = _time.monotonic()

        # Fast path: check if raw_action is already feasible for all constraints
        clipped = np.clip(raw_action, -1.0, 1.0)
        all_feasible = True

        # Check C2 (SoC bounds)
        for b_idx in self._building_batt_act:
            act_idx = self._building_batt_act[b_idx]
            if clipped[act_idx] < safe_min[act_idx] or clipped[act_idx] > safe_max[act_idx]:
                all_feasible = False
                break

        # Check C3 (building power) and C4 (grid)
        if all_feasible:
            grid_total = 0.0
            for b_idx in range(self._n_buildings):
                exo = exo_nec[b_idx] if b_idx < len(exo_nec) else 0.0
                net_b = exo
                if b_idx in self._building_batt_act:
                    net_b += clipped[self._building_batt_act[b_idx]] * self._batt_powers.get(b_idx, 0)
                if b_idx in self._building_ev_act:
                    net_b += clipped[self._building_ev_act[b_idx]] * self._ev_max_charge.get(b_idx, 0)
                if abs(net_b) > self._p_bmax:
                    all_feasible = False
                    break
                grid_total += net_b
            if all_feasible and self._c4_enabled and grid_total > self._p_gmax:
                all_feasible = False

        if all_feasible:
            self._last_qp_path = "fastpath"
            self._last_qp_ms = (_time.monotonic() - _t0) * 1000
            return clipped

        # Not feasible — solve full QP
        n = self._n_actions
        z = cp.Variable(n)
        objective = cp.Minimize(0.5 * cp.sum_squares(z - raw_action))
        constraints = [z >= -1.0, z <= 1.0]

        # C2: Battery SoC bounds (asymmetric charge/discharge)
        for b_idx in self._building_batt_act:
            act_idx = self._building_batt_act[b_idx]
            # Read actual SoC and efficiency
            try:
                es = self._city.buildings[b_idx].electrical_storage
                soc = float(es.soc[-1]) if hasattr(es.soc, '__len__') else float(es.soc)
                cap = float(getattr(es, 'capacity', 6.4) or 6.4)
                rte = float(getattr(es, 'round_trip_efficiency', 0.9) or 0.9)
                p_batt = self._batt_powers.get(b_idx, 5.0)
                import math
                sqrt_rte = math.sqrt(max(rte, 0.01))
                scale_ch = (p_batt * sqrt_rte) / max(cap, 0.01)
                scale_dis = (p_batt / sqrt_rte) / max(cap, 0.01)
            except Exception:
                soc = 0.5
                scale_ch = 0.74
                scale_dis = 0.83

            # charge: soc + z * scale_ch <= soc_high
            constraints.append(soc + z[act_idx] * scale_ch <= self._soc_high)
            # discharge: soc + z * scale_dis >= soc_low
            constraints.append(soc + z[act_idx] * scale_dis >= self._soc_low)

        # C3: Per-building net power magnitude bounds
        # net_b = exo_b + Σ(batt_gain * z_batt) + Σ(ev_gain * z_ev) ∈ [-P_bmax, P_bmax]
        net_exprs = []
        for b_idx in range(self._n_buildings):
            exo = float(exo_nec[b_idx]) if b_idx < len(exo_nec) else 0.0
            net_b = exo
            if b_idx in self._building_batt_act:
                act_idx = self._building_batt_act[b_idx]
                net_b = net_b + self._batt_powers.get(b_idx, 0) * z[act_idx]
            if b_idx in self._building_ev_act:
                act_idx = self._building_ev_act[b_idx]
                net_b = net_b + self._ev_max_charge.get(b_idx, 0) * z[act_idx]
            constraints.append(net_b <= self._p_bmax)
            constraints.append(net_b >= -self._p_bmax)
            net_exprs.append(net_b)

        # C4: Grid aggregate import bound
        if self._c4_enabled and net_exprs:
            grid_total = sum(net_exprs)
            constraints.append(grid_total <= self._p_gmax)

        # EV bounds from safe_min/safe_max (includes connection state, min charge power)
        for b_idx in self._building_ev_act:
            act_idx = self._building_ev_act[b_idx]
            constraints.append(z[act_idx] >= safe_min[act_idx])
            constraints.append(z[act_idx] <= safe_max[act_idx])

        # Passthrough actions unconstrained in [-1, 1] (already in bounds above)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000, eps=1e-4)
            if prob.status in ('optimal', 'optimal_inaccurate') and z.value is not None:
                result = np.asarray(z.value, dtype=np.float32).flatten()
                self._last_qp_path = "qp_solved"
                self._last_qp_ms = (_time.monotonic() - _t0) * 1000
                return np.clip(result, -1.0, 1.0)  # numerical safety
        except Exception:
            pass

        # Fallback: use pre-computed box bounds (C2+C3 enforced, C4 best-effort)
        self._last_qp_path = "fallback"
        self._last_qp_ms = (_time.monotonic() - _t0) * 1000
        return np.clip(raw_action, safe_min, safe_max)

    def _compute_safe_bounds(
        self,
        exo_nec: list[float],
        socs: list[float],
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Compute per-action [safe_min, safe_max] bounds for C2/C3/C4.

        Handles four building topologies:
        - Battery + EV (connected): proportional headroom split
        - Battery + EV (disconnected): battery gets all headroom, EV = [0, 0]
        - Battery only: battery gets all headroom
        - EV only: EV gets all headroom (no battery bounds computed)

        For EV chargers, respects min_charging_power / min_discharging_power:
        - If remaining headroom < min_charging_power, EV action is forced to 0.
        - Otherwise, EV safe range allows actions up to headroom.

        Returns:
            safe_min: array of shape (n_actions,), per-action lower bounds
            safe_max: array of shape (n_actions,), per-action upper bounds
            interventions: count of collapsed/inverted ranges
            n_infeasible: count of dimensions where C3/C4 bounds conflict
                          (exogenous load exceeds limits — structurally unsolvable)
        """
        safe_min = np.full(self._n_actions, -1.0, dtype=np.float32)
        safe_max = np.full(self._n_actions, 1.0, dtype=np.float32)
        interventions = 0

        for b_idx in range(self._n_buildings):
            if b_idx >= len(exo_nec):
                continue
            # Skip buildings with neither battery nor EV
            has_batt = b_idx in self._building_batt_act
            has_ev_device = b_idx in self._ev_max_charge
            if not has_batt and not has_ev_device:
                continue

            exo = exo_nec[b_idx]

            # Total headroom for all controllable devices (clamped to 0)
            import_headroom = max(0.0, self._p_bmax - exo)
            export_headroom = max(0.0, self._p_bmax + exo)

            # Get battery info (may not exist for EV-only buildings)
            batt_act_idx = self._building_batt_act.get(b_idx, None)
            p_batt = self._batt_powers.get(b_idx, 0.0)

            # ── Compute per-device bounds based on topology ──
            has_ev = b_idx in self._ev_max_charge
            b_smin, b_smax = -1.0, 1.0  # defaults

            if has_ev:
                ev_max_ch = self._ev_max_charge[b_idx]
                ev_min_ch = self._ev_min_charge[b_idx]
                ev_max_dis = self._ev_max_discharge[b_idx]
                ev_min_dis = self._ev_min_discharge[b_idx]
                ev_act_idx = self._building_ev_act[b_idx]

                ev_connected = self._is_ev_connected(b_idx)

                # Check departure time — departing EVs treated as disconnected
                ev_state = self._ev_state(b_idx)
                ev_departing = ev_connected and ev_state is not None and ev_state[2] <= 0

                if not ev_connected or ev_departing:
                    # EV disconnected or departing → EV idle, battery gets all headroom
                    safe_min[ev_act_idx] = 0.0
                    safe_max[ev_act_idx] = 0.0
                    if has_batt and p_batt > 0:
                        b_smin = max(-1.0, -export_headroom / p_batt)
                        b_smax = min(1.0, import_headroom / p_batt)
                else:
                    # EV connected — urgency-weighted C3 split
                    urgency = self._ev_urgency(b_idx)
                    has_deficit = self._ev_has_deficit(b_idx)
                    alpha = self._urgency_alpha

                    # --- CHARGE (import) side ---
                    if has_batt and p_batt > 0 and ev_max_ch > 0:
                        if has_deficit and urgency > 1.0:
                            # Behind schedule: EV-first
                            ev_import = min(import_headroom, ev_max_ch)
                            batt_import = max(0.0, import_headroom - ev_import)
                        elif has_deficit:
                            # Weighted split by urgency
                            p_b = 1.0
                            p_e = 1.0 + alpha * urgency
                            total_p = p_b + p_e
                            ev_import = min(import_headroom * p_e / total_p, ev_max_ch)
                            batt_import = max(0.0, import_headroom - ev_import)
                        else:
                            # No deficit: proportional split
                            total_power = p_batt + ev_max_ch
                            ev_import = import_headroom * ev_max_ch / total_power
                            batt_import = import_headroom * p_batt / total_power
                        b_smax = min(1.0, batt_import / p_batt)
                    elif ev_max_ch > 0:
                        # EV-only building
                        ev_import = import_headroom
                        b_smax = 1.0  # no battery
                    else:
                        ev_import = 0.0

                    # --- DISCHARGE (export) side ---
                    if has_batt and p_batt > 0 and ev_max_dis > 0:
                        if has_deficit:
                            # ANY deficit: zero EV V2G. Charge first, V2G after.
                            ev_export = 0.0
                            batt_export = export_headroom
                        else:
                            # No deficit: proportional split (V2G fine)
                            total_power = p_batt + ev_max_dis
                            ev_export = export_headroom * ev_max_dis / total_power
                            batt_export = export_headroom * p_batt / total_power
                        b_smin = max(-1.0, -batt_export / p_batt)
                    elif ev_max_dis > 0:
                        # EV-only building: V2G only if no deficit
                        ev_export = 0.0 if has_deficit else export_headroom
                        b_smin = -1.0
                    else:
                        ev_export = 0.0
                        if has_batt and p_batt > 0:
                            b_smin = max(-1.0, -export_headroom / p_batt)

                    # EV bounds from allocated headroom
                    if ev_max_ch > 0 and ev_import >= ev_min_ch:
                        ev_smax = min(1.0, ev_import / ev_max_ch)
                    elif ev_min_ch == 0 and ev_max_ch > 0:
                        ev_smax = min(1.0, ev_import / ev_max_ch)
                    else:
                        ev_smax = 0.0

                    if ev_max_dis > 0 and ev_export >= ev_min_dis:
                        ev_smin = max(-1.0, -ev_export / ev_max_dis)
                    elif ev_min_dis == 0 and ev_max_dis > 0:
                        ev_smin = max(-1.0, -ev_export / ev_max_dis)
                    else:
                        ev_smin = 0.0

                    if ev_smax < ev_smin:
                        interventions += 1
                    safe_min[ev_act_idx] = ev_smin
                    safe_max[ev_act_idx] = ev_smax

            elif has_batt and p_batt > 0:
                # Battery-only (no EV): all headroom goes to battery
                b_smin = max(-1.0, -export_headroom / p_batt)
                b_smax = min(1.0, import_headroom / p_batt)

            # ── Apply battery bounds + C2 SoC (only if building has battery) ──
            if has_batt and batt_act_idx is not None and p_batt > 0:
                if b_smax < b_smin:
                    interventions += 1
                safe_min[batt_act_idx] = b_smin
                safe_max[batt_act_idx] = b_smax

                # C2: SoC bounds
                # CityLearn formula: charge → SoC += energy × rte / cap
                #                    discharge → SoC += energy / rte / cap
                soc = socs[b_idx] if b_idx < len(socs) else 0.5
                try:
                    es = self._city.buildings[b_idx].electrical_storage
                    cap = float(getattr(es, 'capacity', 6.4) or 6.4)
                    rte = float(getattr(es, 'round_trip_efficiency', 0.9487))
                except Exception:
                    cap = 6.4
                    rte = 0.9487

                soc_per_unit_charge = (p_batt * rte) / max(cap, 1e-6)
                soc_per_unit_discharge = (p_batt / rte) / max(cap, 1e-6)

                remaining_charge = max(0.0, self._soc_high - soc)
                max_charge_action = remaining_charge / max(soc_per_unit_charge, 1e-9)
                safe_max[batt_act_idx] = min(safe_max[batt_act_idx], max_charge_action)

                remaining_discharge = max(0.0, soc - self._soc_low)
                max_discharge_action = remaining_discharge / max(soc_per_unit_discharge, 1e-9)
                safe_min[batt_act_idx] = max(safe_min[batt_act_idx], -max_discharge_action)

        # Passthrough indices keep [-1, 1]
        for idx in self._passthrough_indices:
            safe_min[idx] = -1.0
            safe_max[idx] = 1.0

        # C4: Grid-level aggregate import constraint (JOINT: batteries + EVs)
        # District net import: solar export at one building offsets import at another
        c4_enabled = self._c4_enabled
        self._last_total_exo_import = max(0.0, sum(exo_nec))

        # Infeasibility counters for this step
        self._last_structural_c4 = 0
        self._last_c0_c4_conflict = 0
        self._last_c0_c3_conflict = 0
        self._last_c4_relaxation_kw = 0.0

        if c4_enabled:
            total_exo_import = self._last_total_exo_import
            H = max(0.0, self._p_gmax - total_exo_import)

            # Type A: structural C4 infeasibility
            if total_exo_import > self._p_gmax:
                self._last_structural_c4 = 1

            # Collect all charging devices with their request and priority
            devices = []  # list of (act_idx, power_kw, r_kw, priority, device_type)
            for b in self._building_batt_act:
                act_idx = self._building_batt_act[b]
                p_kw = self._batt_powers.get(b, 5.0)
                r_kw = max(0.0, safe_max[act_idx]) * p_kw
                if r_kw > 1e-9:
                    devices.append((act_idx, p_kw, r_kw, 1.0, 'batt'))

            total_ev_min_kw = 0.0
            for b in self._building_ev_act:
                act_idx = self._building_ev_act[b]
                p_kw = self._ev_max_charge.get(b, 0.0)
                r_kw = max(0.0, safe_max[act_idx]) * p_kw
                if r_kw > 1e-9:
                    has_deficit = self._ev_has_deficit(b)
                    if has_deficit:
                        urgency = self._ev_urgency(b)
                        priority = 1.0 + self._urgency_alpha * urgency
                        ev_min = self._ev_min_charge_kw(b)
                        total_ev_min_kw += ev_min
                        # Type C: per-building C0-vs-C3 conflict
                        # Use same formula as C3: max(0, P_bmax - exo), not max(0, exo)
                        import_hr_b = max(0.0, self._p_bmax - (exo_nec[b] if b < len(exo_nec) else 0))
                        if ev_min > import_hr_b:
                            self._last_c0_c3_conflict += 1
                    else:
                        priority = 0.0  # no-deficit EV gets zero allocation
                    devices.append((act_idx, p_kw, r_kw, priority, 'ev'))

            # Type B: Stage-1 C0-vs-C4 conflict indicator
            if total_exo_import <= self._p_gmax and total_ev_min_kw > H:
                self._last_c0_c4_conflict = 1
                # Mobility policy: relax C4 to preserve EV minimum charge
                if self._infeasibility_policy == "mobility":
                    self._last_c4_relaxation_kw = total_ev_min_kw - H
                    H = total_ev_min_kw  # expand headroom
                    self._mobility_overrides += 1

            # Check if total request exceeds headroom
            total_requested = sum(d[2] for d in devices)

            if total_requested > H and total_requested > 0 and len(devices) > 0:
                # Iterative weighted allocation with saturation
                remaining = list(range(len(devices)))
                allocations = [0.0] * len(devices)
                H_remaining = H

                for _iteration in range(len(devices) + 1):
                    if not remaining or H_remaining <= 0:
                        break

                    total_weighted = sum(
                        devices[i][3] * devices[i][2] for i in remaining
                    )
                    if total_weighted <= 1e-9:
                        # All remaining have zero priority — give them nothing
                        break

                    saturated = []
                    for i in remaining:
                        _, _, r_kw, p_i, _ = devices[i]
                        trial = min(r_kw, (p_i * r_kw / total_weighted) * H_remaining)
                        if trial >= r_kw - 1e-9:
                            saturated.append(i)

                    if not saturated:
                        # No saturations — final proportional distribution
                        for i in remaining:
                            _, _, r_kw, p_i, _ = devices[i]
                            allocations[i] = (p_i * r_kw / total_weighted) * H_remaining
                        break

                    for i in saturated:
                        allocations[i] = devices[i][2]  # full request
                        H_remaining -= devices[i][2]
                        remaining.remove(i)

                # Apply allocations: C4 can only tighten, never loosen C3 bounds
                for idx_d, (act_idx, p_kw, r_kw, _, _) in enumerate(devices):
                    if p_kw > 0:
                        new_max = allocations[idx_d] / p_kw
                        safe_max[act_idx] = min(safe_max[act_idx], new_max)

        # Ensure no inverted ranges after C4 scaling
        inverted = safe_max < safe_min
        n_inverted = int(inverted.sum())
        if n_inverted > 0:
            safe_max = np.maximum(safe_max, safe_min)
            interventions += n_inverted

        # Detect structural infeasibility: buildings where exogenous NEC
        # exceeds P_building_max and no admissible corrective action can fix it.
        #
        # Two cases:
        #   Import violation (exo > +P_bmax): need devices to DISCHARGE (reduce NEC)
        #   Export violation (exo < -P_bmax): need devices to CHARGE (increase NEC)
        #
        # Check both battery AND EV corrective capacity in the needed direction.
        n_structural = 0
        for b_idx in range(self._n_buildings):
            if b_idx >= len(exo_nec):
                continue
            exo = exo_nec[b_idx]
            violation = abs(exo) - self._p_bmax
            if violation <= 0:
                continue  # no violation

            # How much corrective power can all devices at this building provide?
            corrective_kw = 0.0

            if exo > 0:
                # Import violation: need discharge (negative action) to reduce NEC
                if b_idx in self._building_batt_act:
                    act_idx = self._building_batt_act[b_idx]
                    corrective_kw += abs(safe_min[act_idx]) * self._batt_powers.get(b_idx, 0)
                if b_idx in self._building_ev_act:
                    act_idx = self._building_ev_act[b_idx]
                    corrective_kw += abs(safe_min[act_idx]) * self._ev_max_discharge.get(b_idx, 0)
            else:
                # Export violation: need charge (positive action) to increase NEC
                if b_idx in self._building_batt_act:
                    act_idx = self._building_batt_act[b_idx]
                    corrective_kw += safe_max[act_idx] * self._batt_powers.get(b_idx, 0)
                if b_idx in self._building_ev_act:
                    act_idx = self._building_ev_act[b_idx]
                    corrective_kw += safe_max[act_idx] * self._ev_max_charge.get(b_idx, 0)

            if corrective_kw < violation:
                n_structural += 1

        n_infeasible = n_inverted + n_structural
        return safe_min, safe_max, interventions, n_infeasible
