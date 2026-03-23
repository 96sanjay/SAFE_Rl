"""
Paper-exact SE-RL with closest-point projection and penalty.

Implements Markgraf et al. 2025 "Safe RL using Action Projection"
Sections 5.1 (SE-RL) and 7.1 (penalty augmentation).

Eq. 12: Phi(x, u) = clip(u, safe_min(x), safe_max(x))
Eq. 23: r_aug = r - h
Eq. 24: h = w * ||u - Phi(x,u)||^2  (zero when u is in the safe set)

Constraints enforced:
    C2: Battery SoC in [soc_low, soc_high]
    C3: Per-building |NEC| <= P_building_max
    C4: Grid import <= P_grid_max (battery-only, EV exempt for C1)

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
      C4: Grid import <= P_grid_max (battery-only, EV exempt for C1)

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

        # Parse action_names to build mapping
        self._building_batt_act = {}  # b_idx -> action_idx
        self._building_ev_act = {}    # b_idx -> action_idx
        self._passthrough_indices = []

        b_idx = 0
        batt_count_for_building = {}
        for act_idx, name in enumerate(action_names_flat):
            name_lower = name.lower()
            if _EV_RE.match(name):
                m = _EV_RE.match(name)
                suffix = m.group('suffix')
                parts = suffix.split('_')
                bldg_num = int(parts[0])
                ev_b_idx = bldg_num - 1
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

    def reset(self, **kwargs):
        self._step_count = 0
        self._total_clipped = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float32)

        # Compute state-dependent safe bounds
        exo_nec = self._get_exogenous_nec()
        socs = self._get_battery_socs()
        safe_min, safe_max, interventions, n_infeasible = self._compute_safe_bounds(exo_nec, socs)

        # Eq. 12: Closest-point projection (NOT rescaling)
        safe_action = np.clip(raw_action, safe_min, safe_max)

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

            # Total headroom for all controllable devices
            import_headroom = self._p_bmax - exo
            export_headroom = self._p_bmax + exo

            # Get battery info (may not exist for EV-only buildings)
            batt_act_idx = self._building_batt_act.get(b_idx, None)
            p_batt = self._batt_powers.get(b_idx, 0.0)

            # ── Compute per-device bounds based on topology ──
            has_ev = b_idx in self._ev_max_charge
            b_smin, b_smax = -1.0, 1.0  # defaults (overwritten below if has_batt)

            if has_ev:
                ev_max_ch = self._ev_max_charge[b_idx]
                ev_min_ch = self._ev_min_charge[b_idx]
                ev_max_dis = self._ev_max_discharge[b_idx]
                ev_min_dis = self._ev_min_discharge[b_idx]
                ev_act_idx = self._building_ev_act[b_idx]

                ev_connected = self._is_ev_connected(b_idx)

                if not ev_connected:
                    # EV disconnected → EV idle, battery gets all headroom
                    safe_min[ev_act_idx] = 0.0
                    safe_max[ev_act_idx] = 0.0
                    if has_batt and p_batt > 0:
                        b_smin = max(-1.0, -export_headroom / p_batt)
                        b_smax = min(1.0, import_headroom / p_batt)
                else:
                    # EV connected → split headroom between battery and EV
                    if has_batt and p_batt > 0:
                        # Proportional split
                        total_power = p_batt + ev_max_ch if ev_max_ch > 0 else p_batt + 1.0
                        batt_frac = p_batt / total_power
                        ev_frac = 1.0 - batt_frac
                        batt_import = import_headroom * batt_frac
                        batt_export = export_headroom * batt_frac
                        b_smin = max(-1.0, -batt_export / p_batt)
                        b_smax = min(1.0, batt_import / p_batt)
                        ev_import = import_headroom * ev_frac
                        ev_export = export_headroom * ev_frac
                    else:
                        # EV-only building: EV gets all headroom
                        ev_import = import_headroom
                        ev_export = export_headroom

                    # EV bounds with min power enforcement
                    if ev_import >= ev_min_ch and ev_min_ch > 0:
                        ev_smax = min(1.0, ev_import / ev_max_ch) if ev_max_ch > 0 else 0.0
                    elif ev_min_ch == 0 and ev_max_ch > 0:
                        ev_smax = min(1.0, ev_import / ev_max_ch)
                    else:
                        ev_smax = 0.0

                    if ev_export >= ev_min_dis and ev_min_dis > 0:
                        ev_smin = max(-1.0, -ev_export / ev_max_dis) if ev_max_dis > 0 else 0.0
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

        # C4: Grid-level aggregate import constraint (BATTERY ONLY)
        c4_enabled = self._c4_enabled
        self._last_total_exo_import = sum(max(0.0, e) for e in exo_nec)

        if c4_enabled:
            total_exo_import = self._last_total_exo_import
            grid_headroom = max(0.0, self._p_gmax - total_exo_import)

            total_batt_charge = sum(
                max(0.0, safe_max[self._building_batt_act[b]]) * self._batt_powers[b]
                for b in self._building_batt_act
            )

            total_ev_charge = sum(
                max(0.0, safe_max[self._building_ev_act[b]]) * self._ev_max_charge.get(b, 0.0)
                for b in self._building_ev_act
            ) if self._building_ev_act else 0.0

            remaining_for_batt = max(0.0, grid_headroom - total_ev_charge)

            if total_batt_charge > remaining_for_batt and total_batt_charge > 0:
                scale = max(0.0, remaining_for_batt / total_batt_charge)
                for b in self._building_batt_act:
                    batt_act_idx = self._building_batt_act[b]
                    safe_max[batt_act_idx] = max(safe_min[batt_act_idx],
                                                  safe_max[batt_act_idx] * scale)

        # Ensure no inverted ranges after C4 scaling
        inverted = safe_max < safe_min
        n_inverted = int(inverted.sum())
        if n_inverted > 0:
            safe_max = np.maximum(safe_max, safe_min)
            interventions += n_inverted

        # Detect structural infeasibility: buildings where exogenous load
        # already exceeds P_building_max and no controllable device can fix it.
        # This happens when base_load > P_bmax AND the device can't discharge
        # enough (or has no device at all). The wrapper sets [0,0] or tight
        # bounds, but the constraint is still violated by physics.
        n_structural = 0
        for b_idx in range(self._n_buildings):
            if b_idx >= len(exo_nec):
                continue
            if abs(exo_nec[b_idx]) > self._p_bmax:
                # Check if any device at this building can reduce NEC enough
                can_fix = False
                if b_idx in self._building_batt_act:
                    act_idx = self._building_batt_act[b_idx]
                    # Max discharge possible = |safe_min| × p_batt
                    max_discharge_kw = abs(safe_min[act_idx]) * self._batt_powers.get(b_idx, 0)
                    if max_discharge_kw > abs(exo_nec[b_idx]) - self._p_bmax:
                        can_fix = True
                if not can_fix:
                    n_structural += 1

        n_infeasible = n_inverted + n_structural
        return safe_min, safe_max, interventions, n_infeasible
